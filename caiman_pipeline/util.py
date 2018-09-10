import os
import caiman as cm
import numpy as np
from glob import glob


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


def plot_neurons(calcium, masks, folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    else:
        # remove neurons from previous extractions
        neurons = glob(os.path.join(folder, '*.png'))
        for neuron in neurons:
            os.remove(neurons)

    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    for i in range(calcium.shape[0]):
        plt.subplot(2, 1, 1)
        plt.imshow(masks[:, i].reshape(270, 360))
        plt.subplot(2, 1, 2)
        plt.plot(calcium[i])
        plt.savefig(os.path.join(folder, 'neuron-{}.png'.format(i)))
        plt.close()
    all_masks = np.sum(masks, axis=1)
    plt.imshow(all_masks.reshape(270, 360))
    plt.savefig(os.path.join(folder, 'all-masks.png'))
    plt.close()