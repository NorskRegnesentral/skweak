import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
     name='skweak',  
     version='0.2.9',
     author="Pierre Lison",
     author_email="plison@nr.no",
     description="Software toolkit for weak supervision in NLP",
     license='LICENSE.txt',
     packages=['skweak'],
     python_requires=">=3.6",
     install_requires=["spacy>=3.0","hmmlearn>=0.2", "pandas>=0.23", "numpy>=1.18"],
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://github.com/NorskRegnesentral/skweak",
     classifiers=[
         "Programming Language :: Python :: 3",
         "Programming Language :: Python :: 3.7",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
         "Intended Audience :: Developers",
         "Intended Audience :: Science/Research",
         "Topic :: Scientific/Engineering"
     ],
 )