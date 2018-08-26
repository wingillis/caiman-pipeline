import caiman as cm
import numpy as np


def generate_indices(batch_size, max_frames):
    nbatches = int(np.ceil(max_frames / batch_size))
    for i in range(nbatches):
        # yield the start and end indices for a batched slice of data
        yield i * batch_size, min((i + 1) * batch_size, max_frames)


def create_dview(n_procs=9):
    _, dview, n_procs = cm.cluster.setup_cluster(backend='local', n_processes=n_procs, single_thread=False)
    return dview


def memmap_file(filename, basename='memmap_', dview=None):
    return cm.save_memmap([filename], base_name=basename, dview=dview, order='C')
