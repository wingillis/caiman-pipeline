import tifffile
from glob import glob
import h5py
import dill
import matplotlib.pyplot as plt

import caiman as cm
from caiman.source_extraction import cnmf
from caiman.utils.visualization import inspect_correlation_pnr
from caiman.components_evaluation import estimate_components_quality_auto
from caiman.motion_correction import motion_correct_oneP_rigid

c, dview, n_procs = cm.cluster.setup_cluster(backend='local', n_processes=19)

# find tif files and make them if they don't exist
tifs = glob('*.tif')
if len(tifs) == 0:
	# TODO: handle new extraction downsampled files also
	files = sorted(glob('*recording*downsample*.mat'), key=lambda a: len(a))
	with h5py.File(files[0], 'r') as f:
		tifffile.imsave(files[0][:-3] + 'tif', data=f['Y'], bigtiff=True)
	exp_file = files[0][:-3] + 'tif'
else:
	# check to make sure that it's a processed tif
	candidates = list(filter(lambda a: len(glob(a[:-3] + 'mat')) > 0, tifs))
	exp_file = candidates[0]

gSig = 3
min_corr = 0.8
min_pnr = 4
min_SNR = 3
r_values_min = 0.85
merge_thresh = 0.8
frate = 30
decay_time = 0.4

memmapped_files = glob('memmap_*')
if len(memmapped_files) == 0:
	mc = motion_correct_oneP_rigid([exp_file], dview=dview, max_shifts=[5, 5], gSig_filt=[gSig]*2, splits_rig=10, save_movie=True)
	# transforming memoruy mapped file in C order (efficient to perform computing)
	fname_new = cm.save_memmap([mc.fname_tot_rig], base_name='memmap_', order = 'C') 
else:
	fname_new = memmapped_files[0]

Yr, dims, T = cm.load_memmap(fname_new)
Y = Yr.T.reshape((T,) + dims, order='F')

if len(glob('caiman-corr.png')) == 0:
	cn_filter, pnr = cm.summary_images.correlation_pnr(Y[:2000], gSig=gSig, swap_dim=False)
	inspect_correlation_pnr(cn_filter,pnr)
	plt.savefig('caiman-corr.png')

min_corr = eval(input('corr: '))
min_pnr = eval(input('pnr: '))

cnm = cnmf.CNMF(n_processes=n_procs, 
                method_init='corr_pnr',                 # use this for 1 photon
                k=100,                                   # neurons per patch
                gSig=(gSig, gSig),                            # half size of neuron
                gSiz=(11, 11),                          # in general 3*gSig+1
                merge_thresh=merge_thresh,
                p=1,                     
                dview=dview,                            # if None it will run on a single thread
                tsub=2,                                 # downsampling factor in time for initialization, increase if you have memory problems             
                ssub=1,                                 # downsampling factor in space for initialization, increase if you have memory problems
                Ain=None,                               # if you want to initialize with some preselcted components you can pass them here as boolean vectors
                rf=(40, 40),                            # half size of the patch (final patch will be 100x100)
                stride=(20, 20),                        # overlap among patches (keep it at least large as 4 times the neuron size)
                only_init_patch=True,                   # just leave it as is
                gnb=16,                                 # number of background components
                nb_patch=0,                            # number of background components per patch
                method_deconvolution='oasis',          
                low_rank_background=True,               #leave as is
                update_background_components=True,      # sometimes setting to False improve the results
                min_corr=min_corr,                      # min peak value from correlation image 
                min_pnr=min_pnr,                        # min peak to noise ration from PNR image
                normalize_init=False,                   # just leave as is
                center_psf=True,                        # leave as is for 1 photon
                ring_size_factor=1.7,
                del_duplicates=True)                    # whether to remove duplicates from initialization

cnm.fit(Y)


idx_components, idx_components_bad, comp_SNR, r_values, pred_CNN=estimate_components_quality_auto(
                            Y, cnm.A, cnm.C, cnm.b, cnm.f, cnm.YrA, frate, 
                            decay_time, gSig, dims, dview=dview, 
                            min_SNR=min_SNR, r_values_min=r_values_min, min_std_reject=0.5, use_cnn=False)


cm.cluster.stop_server()

del cnm.dview 

with open('cnmf.dill', 'wb') as f:
    dill.dump(cnm, f)

# save the important data
with h5py.File('caiman-output.h5', 'w') as f:
    f.create_dataset('ca', data=cnm.C[idx_components])
    f.create_dataset('masks', data=cnm.A.toarray()[:, idx_components])

# crd = cm.utils.visualization.plot_contours(cnm.A.tocsc()[:,idx_components], cn_filter, thr=.8, vmax=0.95)
# plt.savefig('caiman-extracted-contours.png')
print(f'There are {cnm.C.shape[0]} neurons')
