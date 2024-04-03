from setuptools import setup, find_packages

def readme():
  with open('README.md', 'r') as f:
    return f.read()

setup(
  name='neurotools',
  version='0.0.1',
  author='Ivan Mysin, Sergey Dubrovin, Sergey Skorokhod, Artem Vasilev',
  author_email='imysin@mail.ru',
  description='LFP and spikes processing',
  long_description=readme(),
  long_description_content_type='text/markdown',
  url='home_link',
  packages=find_packages(),
  install_requires=['requests>=2.25.1'],
  classifiers=[
    'Programming Language :: Python :: 3.10',
    'License :: OSI Approved :: MIT License',
    'Operating System :: OS Independent'
  ],
  keywords='brain rhythms neuron spike lfp place cells hippocampus',
  project_urls={
    'Documentation': 'link'
  },
  python_requires='>=3.7'
)