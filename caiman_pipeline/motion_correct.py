import tifffile
import h5py
import re
import caiman_pipeline.util as util
import caiman.motion_correction as mc


def handle_mat_file(matfile, tifname=None):
    if not tifname:
        tifname = re.sub('mat$', 'tif', matfile)
    with h5py.File(matfile, 'r') as f:
        with tifffile.TiffWriter(tifname, bigtiff=True) as out_file:
            for start, end in util.generate_indices(1000, f['Y'].shape[0]):
                out_file.save(data=f['Y'][start:end])
    return tifname


def motion_correct(tif, mc_params, dview=None):
    if not dview:
        dview = util.create_dview()
    corrected = mc.motion_correct_oneP_rigid([tif], dview=dview, **mc_params)
    return corrected
