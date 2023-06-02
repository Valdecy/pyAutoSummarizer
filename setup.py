from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / 'README.md').read_text()

setup(
    name='pyAutoSummarizer',
    version='1.0.9',
    license='GNU',
    author='Valdecy Pereira',
    author_email='valdecy.pereira@gmail.com',
    url='https://github.com/Valdecy/pyAutoSummarizer',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'chardet',
        'numpy',
        'regex',
        'scikit-learn',
        'sentence_transformers',
        'openai',
        'transformers'
    ],
    zip_safe=True,
    description='An Extractive and Abstractive Summarization Library Powered with Artificial Intelligence',
    long_description=long_description,
    long_description_content_type='text/markdown',
)
