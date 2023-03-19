================
PETSc for Python
================

.. only:: html or man

   :Author:       Lisandro Dalcin
   :Contact:      dalcinl@gmail.com
   :Web Site:     https://gitlab.com/petsc/petsc
   :Date:         |today|

.. topic:: Abstract

   This document describes petsc4py_, a Python_ wrapper to the PETSc_
   libraries.

   PETSc_ (the Portable, Extensible Toolkit for Scientific
   Computation) is a suite of data structures and routines for the
   scalable (parallel) solution of scientific applications modeled by
   partial differential equations. It employs the MPI_ standard for
   all message-passing communication.

   This package provides an important subset of PETSc functionalities
   and uses NumPy_ to efficiently manage input and output of array data.

   A *good friend* of petsc4py is:

      * mpi4py_: Python bindings for MPI_,
        the *Message Passing Interface*.

   Other projects depends on petsc4py:

      * slepc4py_: Python bindings for SLEPc_,
        the *Scalable Library for Eigenvalue Problem Computations*.

.. include:: links.txt


.. toctree::
   :caption: Contents
   :maxdepth: 2

   overview
   tutorial
   reference
   citing
   install
