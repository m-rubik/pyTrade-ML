from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()
long_description = (here / 'README.md').read_text(encoding='utf-8')

setup(
    name='pytrademl',
    version='0.0.1',
    description='ML for finance',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/m-rubik/pyTrade-ML',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    install_requires=['pandas', 'scikit-learn', 'alpha-vantage', 'requests', 'beautifulsoup4', 'pandas-datareader', 'matplotlib', 'ta', 'tensorflow', 'keras', 'numpy', 'mplfinance']
)
