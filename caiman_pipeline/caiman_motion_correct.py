import tifffile
import sys
from glob import glob
import h5py
import dill
import matplotlib.pyplot as plt

import caiman as cm
from caiman.source_extraction import cnmf
from caiman.utils.visualization import inspect_correlation_pnr
from caiman.components_evaluation import estimate_components_quality_auto
from caiman.motion_correction import motion_correct_oneP_rigid

c, dview, n_procs = cm.cluster.setup_cluster(backend='local', n_processes=9)

# find tif files and make them if they don't exist
tifs = glob('*.tif')
if len(tifs) == 0 or len(sys.argv) > 1:
    # TODO: handle new extraction downsampled files also
    if len(sys.argv) > 1:
        fname = sys.argv[1]
    else:
        files = sorted(glob('*recording*downsample*.mat'), key=lambda a: len(a))
        fname = files[0]
    if not fname.endswith('tif'):
        with h5py.File(fname, 'r') as f:
            tifffile.imsave(fname[:-3] + 'tif', data=f['Y'], bigtiff=True)
    exp_file = fname[:-3] + 'tif'
else:
    # check to make sure that it's a processed tif
    candidates = list(filter(lambda a: len(glob(a[:-3] + 'mat')) > 0, tifs))
    exp_file = candidates[0]

gSig = 3
min_SNR = 3
r_values_min = 0.85
merge_thresh = 0.8
frate = 30
decay_time = 0.4

memmapped_files = glob('memmap_*')
motion_corrected = glob(exp_file[:-3] + '*.mmap')
if len(motion_corrected) == 0:
    # splits_rig is for keeping the memory low
    mc = motion_correct_oneP_rigid([exp_file], dview=dview, max_shifts=[5, 5], gSig_filt=[gSig]*2, splits_rig=50, save_movie=True)
    # transforming memoruy mapped file in C order (efficient to perform computing)
    fname_new = cm.save_memmap([mc.fname_tot_rig], base_name='memmap_', order = 'C')
elif len(memmapped_files) == 0:
    fname_new = cm.save_memmap([motion_corrected[0]], base_name='memmap_', order='C')
else:
    fname_new = memmapped_files[0]

Yr, dims, T = cm.load_memmap(fname_new)
Y = Yr.T.reshape((T,) + dims, order='F')

if len(glob('caiman-corr.png')) == 0:
    cn_filter, pnr = cm.summary_images.correlation_pnr(Y[:2000], gSig=gSig, swap_dim=False)
    inspect_correlation_pnr(cn_filter,pnr)
    plt.savefig('caiman-corr.png')

cm.cluster.stop_server()
