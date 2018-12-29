from setuptools import setup

setup(
    name='caiman_pipeline',
    author='win.gillis@gmail.com',
    version='0.0.2',
    install_requires=[
        'numpy', 'h5py', 'opencv-python', 'click', 'tqdm', 'tifffile', 'dill',
        'ruamel.yaml'
    ],
    entry_points={'console_scripts': ['caiman-pipe = caiman_pipeline.cli:cli']}
)
