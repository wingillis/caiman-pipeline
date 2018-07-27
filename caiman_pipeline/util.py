import caiman as cm


def create_dview(n_procs=9):
    _, dview, n_procs = cm.cluster.setup_cluster(backend='local', n_processes=n_procs, single_thread=False)
    return dview


def memmap_file(filename, basename='memmap_', dview=None):
    return cm.save_memmap([filename], base_name=basename, dview=dview, order='C')
