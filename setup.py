from setuptools import setup, find_packages


def readme():
    with open('README.md') as f:
        return f.read()


def requirements():
    with open('requirements.txt') as f:
        return f.read().splitlines()


def version():
    loc = dict()
    with open('zonal_variograms/__version__.py') as f:
        exec(f.read(), loc, loc)
    return loc['__version__']

setup(
    name='zonal_variograms',
    version=version(),
    author='Mirko Mälicke',
    author_email='mirko.maelicke@kit.edu',
    description='Zonal (Polygon) variograms on a input raster',
    long_description=readme(),
    long_description_content_type='text/markdown',
    install_requires=requirements(),
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'zonal_variograms = zonal_variograms.cli:process_segmented_files'
        ]
    }
)