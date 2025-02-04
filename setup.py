from setuptools import find_packages, setup

setup(
    name="source",
    version="0.1.0",
    description="Hyperedge replacement grammar for Open Information Extraction",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="NLP graph source oie",
    url="https://github.com/Eszti/hrg",
    author="Eszter Iklodi, Gabor Recski",
    author_email="eszter.iklodi@tuwien.ac.at,gabor.recski@tuwien.ac.at",
    license="MIT",
    install_requires=[
        "matplotlib",
        "networkx",
        "ordered-set",
        "protobuf==3.20",
        "scikit-learn",
        "stanza",
    ],
    packages=find_packages(),
    include_package_data=True,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
    zip_safe=False,
)