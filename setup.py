from distutils.core import setup

setup(
    name='RL_toolkit',
    version='1.0',
    packages=[
        'agent',
        'environment',
        'simulator'
    ],
    package_dir={
        'agent': 'src/agent',
        'environment': 'src/environment',
        'simulator': 'src/simulator'
    },
    author='Dorian Bagur',
    author_email='dorian.bagur@gmail.com',
    description='Reinforcement learning toolkit'
)