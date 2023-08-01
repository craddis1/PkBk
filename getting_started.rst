Getting Started
===============

Nothing is required to be compiled! Simply clone the repository:

   .. code-block:: shell

      git clone ...

Requirements:

- numpy
- scipy
- Numba
- pyfftw

See the example notebooks `pk_example.ipynb` and `bk_example.ipynb` for a quick start into using PkBk for computing power spectrum and bispectrum.

Additional Information
----------------------

For more detailed information and usage instructions, refer to the specific sections below.

Interpolating 
-----------------------------------------------

To obtain additional information about interpolating a box 

Multiprocessing
---------------

Can be added on the frontend for computing spectra over many realisations. However I've found there can be some incompatibility (e.g. see `Link text https://github.com/pyFFTW/pyFFTW/issues/135`_) between the multithreading enabled in pyfftw and numba (bispectrum only) so would be advisable to disable that first!




Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
