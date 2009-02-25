================
PETSc for Python
================

:Authors:      Lisandro Dalcín
:Contact:      dalcinl@gmail.com
:Organization: CIMEC_
:Address:      PTLC, (3000) Santa Fe, Argentina
:Web Site:     http://petsc4py.googlecode.com
:Date:         |today|
:Copyright:    This document has been placed in the public domain.

This document describes petsc4py_, a Python_ port to the PETSc_
libraries.

PETSc_ (the Portable, Extensible Toolkit for Scientific Computation)
is a suite of data structures and routines for the scalable (parallel)
solution of scientific applications modeled by partial differential
equations. It employs the MPI_ standard for all message-passing
communication.

This package provides an important subset of PETSc functionalities and
uses NumPy_ to efficiently manage input and output of array data.

A *good friend* of petsc4py is:

    * mpi4py_: Python bindings for MPI_, 
      the *Message Passing Interface*.

Other two projects depend on petsc4py:

    * slepc4py_: Python bindings for SLEPc_, 
      the *Scalable Library for Eigenvalue Problem Computations*.

    * tao4py_: Python bindings for TAO_, 
      the *Toolkit for Advanced Optimization*.


Contents
========

.. toctree::
   :maxdepth: 2

   overview
   tutorial
   install


.. include:: links.txt
