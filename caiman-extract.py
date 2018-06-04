import os
import tifffile
import ruamel.yaml as yaml
from glob import glob
import h5py
import dill
import sys

import caiman as cm
from caiman.source_extraction import cnmf
from caiman.components_evaluation import estimate_components_quality_auto

c, dview, n_procs = cm.cluster.setup_cluster(backend='local', n_processes=8, single_thread=False)

gSig = 3
min_SNR = 3
r_values_min = 0.85
merge_thresh = 0.8
frate = 30
decay_time = 0.4
opts = {}

if os.path.exists('cnmfe.yaml'):
    with open('cnmfe.yaml', 'r') as f:
        opts = yaml.load(f, yaml.Loader)
else:
    print('cnmfe.yaml doesn\'t exist, falling back on defaults')

min_corr = opts.get('min_corr', 0.8)
min_pnr = opts.get('min_pnr', 8)

if len(sys.argv) > 1:
    fname_new = sys.argv[1]
else:
    fname_new = glob('memmap_*.mmap')[0]

Yr, dims, T = cm.load_memmap(fname_new)
Y = Yr.T.reshape((T,) + dims, order='F')

cnm = cnmf.CNMF(n_processes=n_procs, 
                method_init='corr_pnr',                 # use this for 1 photon
                k=100,                                   # neurons per patch
                gSig=(gSig, gSig),                            # half size of neuron
                gSiz=(11, 11),                          # in general 3*gSig+1
                merge_thresh=merge_thresh,
                p=1,                     
                dview=dview,                            # if None it will run on a single thread
                tsub=3,                                 # downsampling factor in time for initialization, increase if you have memory problems             
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
                ring_size_factor=1.4,
                del_duplicates=True)                    # whether to remove duplicates from initialization

cnm.fit(Y)

idx_components, idx_components_bad, comp_SNR, r_values, pred_CNN=estimate_components_quality_auto(
                            Y, cnm.A, cnm.C, cnm.b, cnm.f, cnm.YrA, frate, 
                            decay_time, gSig, dims, dview=dview, 
                            min_SNR=min_SNR, r_values_min=r_values_min, min_std_reject=0.5, use_cnn=False)

# save the important data
with h5py.File('caiman-output.h5', 'w') as f:
    f.create_dataset('ca', data=cnm.C[idx_components])
    f.create_dataset('masks', data=cnm.A.toarray()[:, idx_components])


del cnm.dview 

with open('cnmf.dill', 'wb') as f:
    dill.dump(cnm, f)


print(f'There are {cnm.C.shape[0]} neurons')

cm.cluster.stop_server(dview=dview)
