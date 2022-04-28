from setuptools import setup


def readme():
    with open('README.md') as f:
        return f.read()


setup(name='icassp',
      version='0.0.1',
      description='Implementation of the simulations in our ICASSP2022 paper',
      long_description=readme(),
      long_description_content_type='text/markdown',
      classifiers=[
          'Programming Language :: Python :: 3',
          'License :: OSI Approved :: MIT License',
          'Operating System :: OS Independent',
      ],
      keywords='independent vector analysis, PARAFAC2',
      url='https://github.com/SSTGroup/ICASSP2022',
      author='Isabell Lehmann',
      author_email='isabell.lehmann@sst.upb.de',
      license='LICENSE',
      packages=['icassp'],
      python_requires='>=3.6',
      install_requires=[
          'numpy',
          'scipy',
          'pytest',
          'joblib',
          'tqdm',
          'matplotlib',
          'tensorly',
          'independent_vector_analysis',
          'argparse',
          'tikzplotlib'
      ],
      include_package_data=True,  # to include non .py-files listed in MANIFEST.in
      zip_safe=False)
