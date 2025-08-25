from setuptools import setup, find_packages
import os

# Read requirements from requirements.txt
def read_requirements():
    reqs_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if not os.path.exists(reqs_path):
        return []
    
    with open(reqs_path) as f:
        requirements = []
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                # Handle version specifiers and extras
                if '[' in line:
                    # Handle jax[cuda12] format
                    requirements.append(line)
                else:
                    requirements.append(line)
        return requirements

setup(
    name="ae-node-stiffness-reduction",
    version="0.1.0",
    description="Neural ODE with Autoencoder for Stiffness Reduction",
    author="Saakaar Bhatnagar",
    author_email="saakaar28@gmail.com",
    packages=find_packages(),
    install_requires=read_requirements(),
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Time-Series Modeling",
        "Topic :: Scientific/Engineering :: Neural ODEs",
    ],
)
