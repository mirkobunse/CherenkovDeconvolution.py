from setuptools import setup, find_packages

with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='cherenkovdeconvolution',
    version='0.0.1',
    description='Deconvolution methods for Cherenkov astronomy and other use cases in experimental physics.',
    long_description=readme,
    author='Mirko Bunse',
    author_email='mirko.bunse@cs.tu-dortmund.de',
    url='https://github.com/mirkobunse/CherenkovDeconvolution.py',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)

