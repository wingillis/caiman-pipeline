import caiman as cm
import caiman.source_extraction as source
import caiman.components_evaluation as evaluate


def extract(mmap_file, cnmf_opts, nprocs=8, dview=None):
    yr, dims, t = cm.load_memmap(mmap_file)
    y = yr.T.reshape((t, *dims), order='F')
    cnmf = source.cnmf.CNMF(
        n_processes=nprocs,
        method_init=cnmf_opts['init-method'],
        gSiz=[cnmf_opts['gsiz']] * 2,
        gSig=[cnmf_opts['gsig']] * 2,
        merge_thresh=cnmf_opts['merge-thresh'],
        Ain=None,
        k=None,
        tsub=cnmf_opts['t-sub'],
        ssub=cnmf_opts['s-sub'],
        rf=[cnmf_opts['half-patch-size']]*2,
        p=1,
        dview=dview,
        stride=[cnmf_opts['patch-overlap']],
        only_init_patch=True,
        method_deconvolution='oasis',
        nb_patch=-1,
        gnb=cnmf_opts['background-components'],
        low_rank_background=None,
        update_background_components=True,
        min_corr=cnmf_opts['min-corr'],
        min_pnr=cnmf_opts['min-pnr'],
        normalize_init=False,
        center_psf=True,
        ring_size_factor=cnmf_opts['ring-size-factor'],
        del_duplicates=True,
        ssub_B=2,
        border_pix=cnmf_opts['border-px']
    )
    cnmf.fit(y)
    estimates = cnmf.estimates

    # TODO: remove hard-coded r values and std reject
    good_idx, bad_idx, _, _, _ = evaluate.estimate_components_quality_auto(
        Y=y, A=estimates.A, C=estimates.C, b=estimates.b, f=estimates.f, YrA=estimates.YrA, frate=cnmf_opts['fps'],
        decay_time=cnmf_opts['decay-time'], gSig=cnmf_opts['gsig'], dims=dims,
        dview=dview, min_SNR=cnmf_opts['min-snr'], use_cnn=False)

    return estimates.C[good_idx], estimates.A.toarray()[:, good_idx], cnmf
