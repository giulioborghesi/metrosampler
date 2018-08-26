from setuptools import setup

setup(name='adaptive_metropolis_sampler',
      version=0.1,
      description='An adaptive Metropolis sampler algorithm',
      url='https://github.com/giulioborghesi/Adaptive-Metropolis-Sampler',
      author='Giulio Borghesi',
      author_email='giulio.borghesi.1981@gmail.com',
      license='MIT',
      packages=['sampler'],
      install_requires=['numpy'],
      setup_requires=['pytest_runner'],
      tests_requires=['pytest'],
      zip_safe=False)
