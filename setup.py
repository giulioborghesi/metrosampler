from setuptools import setup, find_packages

setup(name='metrosampler',
      version=0.1,
      python_requires='<3',
      description='An adaptive Metropolis sampler algorithm',
      url='https://github.com/giulioborghesi/metrosampler',
      author='Giulio Borghesi',
      author_email='giulio.borghesi.1981@gmail.com',
      license='MIT',
      packages=find_packages(),
      install_requires=['numpy', 'scipy', 'matplotlib'],
      entry_points = {
          'console_scripts': [
              'metrosampler = scripts.gendist:gendist'
          ]
      },
      setup_requires=['pytest_runner'],
      tests_requires=['pytest'],
      zip_safe=False)
