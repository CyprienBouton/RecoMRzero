import numpy as np
import torch
import MRzeroCore as mr0
import ggrappa

from .reco_tools import grappa_reconstruction

# -----------------------------------------------------------------------------------------------------------------
#  Utils
# -----------------------------------------------------------------------------------------------------------------


def get_IR(seq0: mr0.sequence.Sequence):
    """Get IR times and indices from a sequence.

    Args:
        seq (mr0.sequence.Sequence): MRzero base sequence.
        
    Returns:
        torch.Tensor: timing before the inversion recovery pulses.
        torch.Tensor: Indices of the inversion recovery pulses.
    """
    # compute the indices of each inversion
    angle_max = max([r.pulse.angle for r in seq0])
    idx_IRs = [i for i, r in enumerate(seq0) if r.pulse.angle==angle_max]
    # Compute the minimum durations of the TRs
    times_after = torch.cumsum(torch.tensor([r.event_time.sum() for r in seq0]), dim=0)
    times_before = torch.tensor([ 0. ] + list(times_after[:-1]))
    times_IR = times_before[idx_IRs]
    return times_IR, idx_IRs


def extract_indices_from_encoding(phase_enc):
    """Extract the indices from a list of phase encoding.
    Assume that the kspace is fully sampled at the centre

    Args:
        phase_enc (np.array): phase encoding (lines or partitions)

    Returns:
        np.array: Matching kspace indices.
    """
    unique_phase = np.unique(phase_enc) # sorted phases
    low_freq_idx = np.abs(unique_phase).argmin()
    delta_k = np.diff(unique_phase)[low_freq_idx]  # smallest k
    
    # compute phase gaps
    phase_gap = np.round(np.diff(unique_phase)/delta_k)
    phase_gap = np.insert(phase_gap, 0, 0)  # Ensure the same length as unique_phase
    
    # Map phase encoding to k-space indices
    k_idx = np.cumsum(phase_gap) 
    k_idx -= k_idx[low_freq_idx] # indices from the low frequency
    mapping_k = dict(zip(unique_phase, k_idx)) # map phases to indices
    return np.array([mapping_k[x] for x in phase_enc])


def crop_nonzero_region(
    tensor: torch.Tensor,
    crop_dims = (0, 1, 2)  # ky=0, kz=1, kx=2
):
    """
    Crops tensor along specified spatial axes (ky, kz, kx).

    Args:
        tensor (torch.Tensor): Input tensor of shape (C, ky, kz, kx)
        crop_dims (tuple): Dimensions to crop (ky=0, kz=1, kx=2)
    
    Returns:
        Cropped tensor
    """
    if isinstance(crop_dims, int):
        crop_dims = tuple(crop_dims)
    assert tensor.ndim == 4, "Expected tensor of shape (C, ky, kz, kx)"
    assert all(d in [0, 1, 2] for d in crop_dims), "Can only crop ky=0, kz=1, or kx=2"

    # Compute binary mask of nonzero values over channels
    mag = tensor.abs().sum(dim=0) > 0  # shape: (ky, kz, kx)
    mag_proj = mag

    # Collapse over non-cropped dims
    for d in list( set([0, 1, 2]) - set(crop_dims) )[::-1]:
        mag_proj = mag_proj.any(dim=d)

    coords = torch.nonzero(mag_proj, as_tuple=False)
    mins, maxs = coords.min(0).values, coords.max(0).values + 1

    # Prepare slicing
    slices = [slice(None)] * 4  # (C, ky, kz, kx)
    for i, d in enumerate(sorted(crop_dims)):
        slices[d + 1] = slice(mins[i].item(), maxs[i].item())  # d+1 to offset channel dim

    return tensor[slices[0], slices[1], slices[2], slices[3]]


# -----------------------------------------------------------------------------------------------------------------
#  KSpaceReconstructor
# -----------------------------------------------------------------------------------------------------------------


class RecoMRzero:
    def __init__(
        self,
        seq0: mr0.sequence.Sequence, 
        freq_os: int = 2, # Siemens default frequency oversampling
    ):
        # input values
        self.seq0 = seq0
        self.freq_os = freq_os
        # initialize values
        self.Nread = None
        self.Nlin_os = None
        self.Npar_os = None
        self.lin_enc = None
        self.par_enc = None
        self.unique_lin_enc = None
        self.unique_par_enc = None
        self.acquisition_order = None
        self.grappa_lin = None
        self.grappa_par = None
        self.times_after_rep = None
        self.is3D = None
        
        self._get_Nread()
        self._get_line_partition_enc()
        self._get_acquisition_order()
    
    ###############################
    # Get simulation parameters 
    ###############################
    
    def _get_Nread(self):
        if self.freq_os not in [1, 2]:
            raise ValueError("Oversampling factor should be 1 (no oversampling) or 2 (Siemens default).")
        for i, r in enumerate(self.seq0):
            if r.adc_usage.sum() > 0:
                kspace = self.seq0.get_full_kspace()[i]
                adc_mask = r.adc_usage>0
                center = kspace[adc_mask][:,0].abs().argmin()
                self.col_center_idx = center + np.where(adc_mask)[0][0]
                self.Nread = 2*( (adc_mask.sum()-center) // self.freq_os ).item()
                self.freq_acquired = np.arange(adc_mask.sum())
                # After the echo, all frequencies are sampled.
                self.freq_acquired += (self.Nread*self.freq_os - adc_mask.sum()).item()
                break
                            
    def _get_line_partition_enc(self):
        lines_enc, partitions_enc = [], []
        for kspace, r in zip(self.seq0.get_full_kspace(), self.seq0):
            if r.adc_usage.sum() > 0:
                ph1, ph2 = kspace[self.col_center_idx, 1:-1]
                lines_enc.append( torch.round(ph1, decimals=2) )
                partitions_enc.append( torch.round(ph2, decimals=2) )
        self.lin_enc = np.array(lines_enc)
        self.unique_lin_enc = np.unique(self.lin_enc)
        self.par_enc = np.array(partitions_enc)
        self.unique_par_enc = np.unique(self.par_enc)
        self.is3D = len(self.unique_par_enc)>1
        
    def _get_acquisition_order(self):
        self.klin = extract_indices_from_encoding(self.lin_enc)
        # Use kspace symmetry to complete missing lines (partial fourier)
        half_Nlin_os = max(-self.klin.min()+1, self.klin.max())
        full_klin = np.arange(half_Nlin_os, -half_Nlin_os, -1) # Siemens: highest frequency is positive
        self.Nlin_os = len(full_klin)
        if self.is3D:
            self.kpar = extract_indices_from_encoding(self.par_enc)
            # Use kspace symmetry to complete missing partitions (partial fourier)
            half_Npar_os = max(-self.kpar.min()+1, self.kpar.max()) 
            full_kpar = np.arange(half_Npar_os, -half_Npar_os, -1) # Siemens: highest frequency is positive
            # from top left to bottom right
            self.acquisition_order = (self.klin-full_klin.min()) + (self.kpar-full_kpar.min())*len(full_klin)
            self.Npar_os = len(full_kpar)  
        else:
            self.kpar = np.ones_like(self.par_enc)
            self.Npar_os = 1
            self.acquisition_order = (self.klin-full_klin.min())
    
    ###############################
    # Main functions
    ###############################    
    
    def get_kspace_from_signal( 
        self,
        signal: torch.Tensor,
        reorder_kspace: bool = False,
        ):
        """Simulate a pulseq sequence using MRzeroCore. Compute kspace (Npart, Nlin, Nread, Ncoil).

        Args:
            signal (torch.Tensor): The simulated signal of the sequence.
            reorder_kspace (bool, optional): Whether to reorder kspace to match MRZero format. Defaults to False.
        """
        Ncoil = signal.shape[-1]
        kspace = torch.zeros((self.Npar_os, self.Nlin_os, self.Nread*self.freq_os, Ncoil), dtype=torch.complex64)
        Nfreq = len(self.freq_acquired)
        acquired_mask = np.ix_(self.acquisition_order, self.freq_acquired, range(Ncoil))    
        kspace.view(-1, self.Nread*self.freq_os, Ncoil)[acquired_mask] = signal.reshape(-1, Nfreq, Ncoil)
        if reorder_kspace:
            kspace = torch.transpose(torch.flip(kspace, (0,1,2)), 1, 2) # reorder kspace
        return kspace
    
    def get_TR_matrix(self):
        """Get the temporal resolution matrix (TR_matrix) of the kspace.

        Returns:
            torch.tensor: temporal resolution matrix of the kspace (shape Npar x Nlin).
        """
        TR_matrix = np.zeros((self.Npar_os, self.Nlin_os))
        idx_adc = [i for i, r in enumerate(self.seq0) if r.adc_usage.sum() > 0]

        # Get IR times and indexes
        times_IR, idx_IRs = get_IR(self.seq0)
        TRs_before = np.insert(np.diff(times_IR), 0, 0)  # Equivalent to [0., *np.diff(times_IR)]

        # Assign TR values based on previous IR indexes
        idx_previous_IR = np.searchsorted(idx_IRs, idx_adc) - 1
        TR_matrix.flat[self.acquisition_order.astype(int)] = TRs_before[idx_previous_IR]

        return TR_matrix
    
    def get_timing_matrix(self):
        timing = np.zeros((self.Npar_os, self.Nlin_os))
        durations = np.cumsum([r.event_time.sum() for r in self.seq0])
        adc_mask = [r.adc_usage.sum()>0 for r in self.seq0]
        timing.flat[self.acquisition_order.astype(int)] = durations[adc_mask]
        return timing
    
    def runReconstruction(
            self,
            signal: torch.Tensor,
            reorder_kspace: bool = False,
        ):
        kspace = self.get_kspace_from_signal(signal, reorder_kspace)
        
        lin_not_null = (kspace.nonzero(as_tuple=True)[1]).unique()
        mask_lin = [lin_not_null.diff(prepend=torch.Tensor([0]))==1]
        min_lin = lin_not_null[mask_lin][0]-1
        max_lin = lin_not_null[mask_lin][-1]
        if lin_not_null.diff().max()>1:
            af_lin = lin_not_null.diff().max()
        else:
            af_lin = 1

        par_not_null = (kspace.nonzero(as_tuple=True)[2]).unique()
        if len(par_not_null)>1:
            mask_par = [par_not_null.diff(prepend=torch.Tensor([0]))==1]
            min_par = par_not_null[mask_par][0]
            max_par = par_not_null[mask_par][-1]
        
            if par_not_null.diff().max()>1:
                af_par = par_not_null.diff().max()
            else:
                af_par = 1
        else:
            min_par = 0
            max_par = -1
            af_par = 1
        
        af = [af_lin, af_par]
        acs = kspace[:,min_lin:max_lin+1, min_par:max_par+1]
        acs = to_recotwix_shape(acs)
        kspace = to_recotwix_shape(kspace)
        kspace_reco =  grappa_reconstruction(kspace, acs, af)
        return kspace_reco
    
    
def to_recotwix_shape(kspace: torch.Tensor):
    """
    Reorders a k-space tensor with shape (Par, Lin, Col, Cha) into a full 17D format
    following the `recotwix_order`.

    Output shape: (1,1,1,1,1,1,1,1,1,1,Par,1,1,Lin,Cha,Col)
    """
    recotwix_order = [
        'Ide', 'Idd', 'Idc', 'Idb', 'Ida',
        'Seg', 'Set', 'Rep', 'Phs', 'Eco',
        'Par', 'Sli', 'Ave', 'Lin', 'Cha', 'Col'
    ]
    
    # Map input tensor dimensions to their corresponding names
    source_dims = {'Par': 0, 'Lin': 1, 'Col': 2, 'Cha': 3}
    
    # Initialize the full 17D shape with ones (singleton dimensions)
    target_shape = [1] * len(recotwix_order)
    
    # Fill in the actual sizes from the input tensor
    for name, dim_idx in source_dims.items():
        target_pos = recotwix_order.index(name)
        target_shape[target_pos] = kspace.shape[dim_idx]
    
    # Reshape input tensor into the target shape
    kspace_reordered = kspace.permute(
        source_dims['Par'],
        source_dims['Lin'],
        source_dims['Col'],
        source_dims['Cha']
    ).reshape(target_shape)
    
    return kspace_reordered
