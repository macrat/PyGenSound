from setuptools import setup, find_packages


requires = [
    'numpy',
    'scipy',
    'PySoundFile',
    'PyAudio',
]

setup(name='PyGenSound',
      version='0.0.1',
      description='A sound generating and computing library',
      long_description=open('README.md', encoding='utf-8').read(),
      author='MacRat',
      author_email='m@crat.jp',
      url='http://github.com/macrat/gensound',
      packages=find_packages(exclude=['tests']),
      license='MIT License',
      requires=requires,
      install_requires=requires,
      test_suite='tests')
