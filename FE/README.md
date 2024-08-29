# FE

Python implementation of some existing functional encryption schemes supporting the inner product functionality.

**NOTE:** This implementation is only meant for educational and research purposes.

## Installation
The src file can be download from "https://github.com/cecyliaborek/FE-inner-product"

This library uses [Charm](https://github.com/JHUISI/charm) - a framework for rapidly prototyping cryptosystems. To install Charm, first verify that you have installed the following dependencies:

- [GMP 5.x](https://gmplib.org/)
- [PBC](https://crypto.stanford.edu/pbc/download.html)
- [OPENSSL](https://www.openssl.org/source/)

After that proceed with Charm installation. **NOTE:** You may encounter problems when installing Charm with Python version higher than 3.6. Therefore, it is recommended to install Python 3.6 and run Charm's configure script, ```./configure.sh```, with the *--python=PATH* option, where path points to your installation of Python3.6.

Finally, create a virtualenv from the provided Pipfile, by running ```pipenv install --site-packages``` (the ```--site-packages``` option will include Charm in the environment).

## How to use
Each scheme consists of four basic methods: 
- *set_up* - generates all parameters needed for the scheme and returns the pair of master public key and master secret key;
- *get_functional_key* - returns the key allowing for calculation of inner product of provided vector and some ciphertext encrypting other vector;
- *encrypt* - encrypts the provided vector;
- *decrypt* - recovers inner product of two vectors from provided vector, its functional key and ciphertext encrypting the other vector.

All methods are implemented as independent script methods and so can be used independently on different machines if we just provide the correct keys.

#### Example usage
python test_nddf.py
