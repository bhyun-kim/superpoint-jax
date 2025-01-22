from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="superpoint-jax",  
    version="0.1.0",
    author="Sean (Byunghyun) Kim",
    author_email="kimbh.mail@gmail.com",
    description="SuperPoint implementation in FLAX/NNX",
    long_description=long_description,    
    long_description_content_type="text/markdown",
    url="https://github.com/bhyun-kim/superpoint-jax", 
    packages=find_packages(),             
    include_package_data=True,            
    install_requires=[                     
        "numpy",
        "matplotlib",
        "opencv-python",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",   
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)
