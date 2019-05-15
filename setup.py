from setuptools import setup, find_packages

with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='cherenkovdeconvolution',
    version='0.0.1',
    description='Deconvolution algorithms for Cherenkov astronomy and other use cases.',
    long_description=readme,
    author='Mirko Bunse',
    author_email='mirko.bunse@cs.tu-dortmund.de',
    url='https://github.com/mirkobunse/CherenkovDeconvolution.py',
    license=license,
    packages=find_packages(exclude=('tests', 'docs')),
    tests_require=['pytest'],
    setup_requires=['pytest-runner'],
    install_requires=[
        'numpy >= 1.16.2',
        'scipy >= 1.2.0',
        'scikit-learn >= 0.20.2',
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: Implementation :: CPython',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Machine Learning',
        'Topic :: Scientific/Engineering :: Astronomy',
        'Topic :: Scientific/Engineering :: Physics',
    ]
)
