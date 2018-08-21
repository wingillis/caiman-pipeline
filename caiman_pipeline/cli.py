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


@click.group()
def cli():
    pass


@cli.command(name='mat-to-tiff')
@click.argument('input-file', type=click.Path(resolve_path=True, exists=True))
@click.option('--output', '-o', default=None, type=click.Path(resolve_path=True))
def mat_to_tiff(input_file, output):
    output = mc.handle_mat_file(input_file, output)
    print('Data converted to tif: {}'.format(output))


@cli.command(name='concat-tiffs')
@click.option('--input-dir', '-i', default=os.getcwd(), type=click.Path(resolve_path=True, exists=True))
@click.option('--downsample', '-d', default=4, type=int)
def concat_tiffs(input_dir, downsample):
    # TODO: add a failsafe so multiple recordings in the same dir don't get concatenated
    regex_pat = r'-[0-9]{3}\.tif$'
    # get tif files
    files = glob(os.path.join(input_dir, '*.tif'))
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
@click.argument('cnmf_options')
@click.option('--out-file', '-o', default='caiman-output.h5')
@click.option('--n-procs', default=9, type=int)
def extract_pipeline(input_file, cnmf_options, out_file, n_procs):
    assert input_file.endswith('mmap'), 'Input file needs to be a memmap file!'
    dir = os.path.dirname(input_file)
    with open(cnmf_options, 'r') as f:
        cnmf_options = yaml.load(f, yaml.Loader)
    dview = util.create_dview(n_procs=n_procs)
    ca_traces, masks, cnmf = extract.extract(input_file, cnmf_options, dview=dview)
    out_file = os.path.join(dir, out_file)
    with h5py.File(out_file, 'w') as f:
        f.create_dataset('ca', data=ca_traces, compression='lzf')
        f.create_dataset('masks', data=masks, compression='lzf')
    del cnmf.dview
    with open(os.path.splitext(out_file)[0] + '-cnmf.dill', 'wb') as f:
        dill.dump(cnmf, f)
    print('There are {} neurons, baby!'.format(ca_traces.shape[0]))
    cm.stop_server(dview=dview)


@cli.command(name='mc')
@click.argument('input_file', type=click.Path(resolve_path=True))
@click.option('--gsig', default=3, type=int)
@click.option('--max-shifts', default=5, type=int)
@click.option('--rigid-splits', default=50, type=int)
@click.option('--save-movie', default=True, type=bool)
@click.option('--nprocs', default=8, type=int)
def motion_correct(input_file, gsig, max_shifts, rigid_splits, save_movie, nprocs):
    import matplotlib.pyplot as plt
    if input_file.endswith('mat'):
        tif = mc.handle_mat_file(input_file)
    elif input_file.endswith('tif'):
        tif = input_file
    mc_params = dict(gSig_filt=[gsig]*2, max_shifts=[max_shifts]*2, splits_rig=rigid_splits,
                     save_movie=save_movie)
    dview = util.create_dview(n_procs=nprocs)
    corrected = mc.motion_correct(tif, mc_params, dview)
    mmapped = util.memmap_file(corrected.fname_tot_rig, dview=dview, basename=re.sub(r'\.tif$', '', tif))
    # remove unnecessary intermediate
    os.remove(corrected.fname_tot_rig)

    Yr, dims, T = cm.load_memmap(mmapped)
    Y = Yr.T.reshape((T,) + dims, order='F')
    cn_filt, pnr = cm.summary_images.correlation_pnr(Y[:3000, max_shifts:-max_shifts, max_shifts:-max_shifts], gSig=gsig, swap_dim=False)
    cm.utils.visualization.inspect_correlation_pnr(cn_filt, pnr)
    plt.savefig(re.sub(r'\.tif$', '', tif) + '-caiman-corr.png')
    cm.stop_server(dview=dview)
