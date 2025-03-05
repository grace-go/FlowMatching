import setuptools

with open("README.md", "r", encoding='utf-8') as fh:
    long_description = fh.read()

install_requires = []

setuptools.setup(
    name="lieflow",
    author="F.M. Sherry",
    author_email="f.m.sherry@tue.nl",
    description="Flow Matching on Lie groups",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/finnsherry/FlowMatching",
    packages=setuptools.find_packages(),
    package_data={'lieflow' : ['lib/*']},
    install_requires=install_requires,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
        "Environment :: GPU :: NVIDIA CUDA",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Private :: Do Not Upload",
    ],
    python_requires=">=3.10",
)
