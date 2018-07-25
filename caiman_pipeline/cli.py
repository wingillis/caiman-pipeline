import os
import re
import tifffile
import click
import cv2
from glob import glob
from tqdm import tqdm


@click.group()
def cli():
    pass


@cli.command(name='concat-tiffs')
@click.option('--input-dir', '-i', default=os.getcwd(), type=click.Path(resolve_path=True, exists=True))
@click.option('--downsample', '-d', default=4, type=int)
def concat_tiffs(input_dir, downsample):
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
def extract():
    pass


@cli.command(name='mc')
def motion_correct():
    pass
