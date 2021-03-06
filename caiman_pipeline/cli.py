import os
import re
import tifffile
import click
import cv2
import dill
import h5py
import ruamel.yaml as yaml
import caiman_pipeline.motion_correct as mc
import caiman_pipeline.util as util
import caiman_pipeline.extract as extract
import caiman as cm
from glob import glob
from tqdm import tqdm
from os.path import join


@click.group()
def cli():
    pass


@cli.command(name='mat-to-tiff')
@click.argument('input-file', type=click.Path(resolve_path=True, exists=True))
@click.option('--output', '-o', default=None, type=click.Path(resolve_path=True))
def mat_to_tiff(input_file, output):
    output = mc.handle_mat_file(input_file, output)
    print('Data converted to tif: {}'.format(output))


@cli.command(name='downsample-isxd')
@click.argument('input-file', type=click.Path(exists=True, resolve_path=True))
@click.option('--downsample', '-d', default=4, type=float)
def downsample_isxd(input_file, downsample):
    fname = input_file[:-5] + '-dnsmpl-{}x-concat.tif'.format(downsample)
    import sys
    sys.path.append('/home/wg41/code/Inscopix Data Processing.linux/Contents/API/Python')
    import isx
    movie = isx.Movie.read(input_file)
    with tifffile.TiffWriter(fname, bigtiff=True) as tif:
        for i in tqdm(range(movie.timing.num_samples), desc='Downsampling',
                      total=movie.timing.num_samples):
            resized = cv2.resize(movie.get_frame_data(i), None, fx=1 / downsample,
                                 fy=1 / downsample, interpolation=cv2.INTER_AREA)
            tif.save(resized)


@cli.command(name='concat-tiffs')
@click.option('--input-dir', '-i', default=os.getcwd(), type=click.Path(resolve_path=True, exists=True))
@click.option('--downsample', '-d', default=4, type=int)
def concat_tiffs(input_dir, downsample):
    # TODO: add a failsafe so multiple recordings in the same dir don't get concatenated
    regex_pat = r'-[0-9]{3}\.tif$'
    # get tif files
    files = glob(join(input_dir, '*.tif'))
    # remove any files with 'concat' in the name - it means it's already concatenated
    files = [f for f in files if 'concat' not in os.path.basename(f)]
    processed = []
    for f in files:
        # append a 0 to the name if it doesn't have numbers
        if not re.findall(regex_pat, f):
            f2 = re.sub(r'\.tif$', '-000.tif', f)
            # rename the actual file - it's easier for everyone
            os.rename(f, f2)
            f = f2
        processed += [f]
    processed = sorted(processed)
    # go through each file one-by-one and write to a larger tiffile
    fname = re.sub(regex_pat, '-dnsmpl-{}x-concat.tif'.format(downsample), processed[0])
    with tifffile.TiffWriter(fname, bigtiff=True) as out_file:
        for file in tqdm(processed, desc='Concatenating data'):
            # read in data frame by frame
            with tifffile.TiffFile(file) as in_file:
                for img in in_file.pages:
                    resized = cv2.resize(img.asarray(), None, fx=1 / downsample,
                                         fy=1 / downsample, interpolation=cv2.INTER_AREA)
                    out_file.save(data=resized.reshape(1, *resized.shape))


@cli.command(name='extract')
@click.argument('input_file', type=click.Path(exists=True, resolve_path=True))
@click.option('--params', '-p', default=None, type=click.Path(exists=True, resolve_path=True))
@click.option('--out-file', '-o', default=None, type=str)
@click.option('--n-procs', default=9, type=int)
def extract_pipeline(input_file, params, out_file, n_procs):
    assert input_file.endswith('mmap'), 'Input file needs to be a memmap file!'
    dir = os.path.dirname(input_file)
    if out_file is None:
        search = re.search('_d1_[0-9]{3}_d2_[0-9]{3}_d3', input_file)
        if search is not None:
            out_file = input_file[:search.start()] + '-extract.h5'
        else:
            out_file = os.path.splitext(input_file)[0] + '-extract.h5'
    if params is None:
        params = glob(join(os.path.dirname(input_file), '*caiman*yaml'))[0]
        # TODO: if no params are found use default params
    with open(params, 'r') as f:
        cnmf_options = yaml.load(f, yaml.Loader)
    dview = util.create_dview(n_procs=n_procs)
    ca_traces, masks, cnmf, dims = extract.extract(input_file, cnmf_options, dview=dview)
    with h5py.File(out_file, 'w') as f:
        f.create_dataset('ca', data=ca_traces, compression='lzf')
        f.create_dataset('masks', data=masks, compression='lzf')
    util.plot_neurons(ca_traces, masks, join(dir, 'caiman-neurons'), dims)
    del cnmf.dview
    with open(os.path.splitext(out_file)[0] + '-estimates.dill', 'wb') as f:
        dill.dump(cnmf.estimates, f)
    with open(os.path.splitext(out_file)[0] + '-cnmf.dill', 'wb') as f:
        dill.dump(cnmf, f)
    # cnmf.save(os.path.splitext(out_file)[0] + '-cnmf.h5')

    print('There are {} neurons, baby!'.format(ca_traces.shape[0]))
    cm.stop_server(dview=dview)


@cli.command(name='mc')
@click.argument('input_file', type=click.Path(resolve_path=True))
@click.option('--gsig', default=3, type=int)
@click.option('--max-shifts', default=10, type=int)
@click.option('--rigid-splits', default=50, type=int)
@click.option('--save-movie', default=True, type=bool)
@click.option('--nprocs', default=8, type=int)
def motion_correct(input_file, gsig, max_shifts, rigid_splits, save_movie, nprocs):
    import matplotlib.pyplot as plt
    if input_file.endswith('mat'):
        tif = mc.handle_mat_file(input_file)
    elif input_file.endswith('tif') or input_file.endswith('tiff'):
        tif = input_file
    mc_params = dict(gSig_filt=[gsig]*2, max_shifts=[max_shifts]*2, splits_rig=rigid_splits,
                     save_movie=save_movie)
    dview = util.create_dview(n_procs=nprocs)
    corrected = mc.motion_correct(tif, mc_params, dview)
    mmapped = util.memmap_file(corrected.fname_tot_rig, dview=dview, basename=re.sub(r'\.tif$', '', tif))
    # remove unnecessary intermediate
    if isinstance(corrected.fname_tot_rig, (list, tuple)):
        os.remove(corrected.fname_tot_rig[0])
    else:
        os.remove(corrected.fname_tot_rig)

    Yr, dims, T = cm.load_memmap(mmapped)
    Y = Yr.T.reshape((T,) + dims, order='F')
    cn_filt, pnr = cm.summary_images.correlation_pnr(Y[:3000, max_shifts:-max_shifts, max_shifts:-max_shifts], gSig=gsig, swap_dim=False)
    cm.utils.visualization.inspect_correlation_pnr(cn_filt, pnr)
    plt.savefig(re.sub(r'\.tif$', '', tif) + '-caiman-corr.png')
    cm.stop_server(dview=dview)
