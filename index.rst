.. PkBk documentation master file, created by
   sphinx-quickstart on Tue Jul 18 15:09:57 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PkBk's documentation!
================================

Computes Power Spectrum and Bispectrum multipoles using FFTs allowing for option of different Line-of-sights (LOS).

It can be used to quickly compute multipoles over many realisations with expansions for different LOS included as in ... . For the bispectrum several options are available but it can return values for the bispectrum over the full range of triangles for a given k-range.

It is intended as a fast user-friendly code written in python with heavy use of numpy, and so nothing is needed to be compiled! The FFTs are implemented in c with pyfftw and other key bottlenecks are optimised with Numba. Mulithreading is implented through pyfftw and Numba and is used in some key areas. Multiproccessing can then be added on the frontend.


See `pk_example.ipynb` and `bk_example.ipynb` for a quick start.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   
   Getting Started <getting_started>
   Compute_grid_info <cgi>
   Power Spectrum <pk>
   Bispectrum <bk> 

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
