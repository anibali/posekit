from setuptools import setup, find_packages


setup(
    name='posekit',
    version='0.0.0',
    author='Aiden Nibali',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    include_package_data=True,
    install_requires=[
        'numpy',
        'phx-class-registry',
    ]
)
