from setuptools import setup, find_packages

setup(
    name='pm',
    version='0.0.1',
    description='pm_0.0.1',
    packages=find_packages(),  # Automatically includes all packages under pm/
    include_package_data=True,
    install_requires=[],
)