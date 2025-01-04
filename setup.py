from setuptools import setup, find_packages

setup(
    name="SL_simulation",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        # Add other dependencies as needed
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="Strong Lensing Simulation Package",
    keywords="strong lensing, simulation, astronomy",
    url="https://github.com/yourusername/SL_simulation",  # If applicable
)