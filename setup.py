from distutils.core import setup

setup(
    name='RL_toolkit',
    version='1.0',
    packages=[
        'environment',
        'simulator'
    ],
    package_dir={
        'environment': 'src/environment',
        'simulator': 'src/simulator'
    },
    author='Dorian Bagur',
    author_email='dorian.bagur@gmail.com',
    description='Reinforcement learning toolkit'
)