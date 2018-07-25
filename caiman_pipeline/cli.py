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
    # get tif files
    files = glob(os.path.join(input_dir, '*.tif'))
    # remove any files with 'concat' in the name - it means it's already concatenated
    files = [f for f in files if 'concat' not in f]
    processed = []
    for f in files:
        # append a 0 to the name if it doesn't have numbers
        if len(f.split('-')) == 1:
            f2 = re.sub(r'\.tif$', '-000.tif', f)
            # rename the actual file - it's easier for everyone
            os.rename(f, f2)
            f = f2
        processed += [f]
    processed = sorted(processed)
    # go through each file one-by-one and write to a larger tiffile
    fname = processed[0].split('-')[0] + '-dnsmpl-{}x-concat.tif'.format(downsample)
    with tifffile.TiffWriter(fname, bigtiff=True) as out_file:
        for file in tqdm(processed, desc='Concatenating data'):
            # read in data frame by frame
            with tifffile.TiffFile(file) as in_file:
                for img in in_file.pages:
                    resized = cv2.resize(img.asarray(), None, fx=1 / downsample,
                                         fy=1 / downsample, interpolation=cv2.INTER_AREA)
                    out_file.save(data=resized.reshape(1, *resized.shape))
