from distutils.core import setup

setup(
    name='lunar_lander_rl',
    version='1.0',
    packages=[
        'environment'
    ],
    package_dir={
        'environment': 'src/environment'
    },
    author='Dorian Bagur',
    author_email='dorian.bagur@gmail.com',
    description='Reinforcement learning applied to Lunar Lander'
)