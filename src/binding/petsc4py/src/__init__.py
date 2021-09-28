# Author:  Lisandro Dalcin
# Contact: dalcinl@gmail.com

# --------------------------------------------------------------------

"""
PETSc for Python
================

This package is an interface to PETSc libraries.

PETSc_ (the Portable, Extensible Toolkit for Scientific Computation)
is a suite of data structures and routines for the scalable (parallel)
solution of scientific applications modeled by partial differential
equations. It employs the MPI_ standard for all message-passing
communication.

.. _PETSc: http://www.mcs.anl.gov/petsc
.. _MPI:   http://www.mpi-forum.org

"""

__author__    = 'Lisandro Dalcin'
__version__   = '3.16.0'
__credits__   = 'PETSc Team <petsc-maint@mcs.anl.gov>'

# --------------------------------------------------------------------

def init(args=None, arch=None, comm=None):
    """
    Initialize PETSc.

    :Parameters:
      - `args`: command-line arguments, usually the 'sys.argv' list.
      - `arch`: specific configuration to use.
      - `comm`: MPI commmunicator

    .. note:: This function should be called only once, typically at
       the very beginning of the bootstrap script of an application.
    """
    import petsc4py.lib
    PETSc = petsc4py.lib.ImportPETSc(arch)
    args  = petsc4py.lib.getInitArgs(args)
    PETSc._initialize(args, comm)

# --------------------------------------------------------------------

def get_include():
    """
    Return the directory in the package that contains header files.

    Extension modules that need to compile against petsc4py should use
    this function to locate the appropriate include directory. Using
    Python distutils (or perhaps NumPy distutils)::

      import petsc4py
      Extension('extension_name', ...
                include_dirs=[..., petsc4py.get_include()])
    """
    from os.path import dirname, join
    return join(dirname(__file__), 'include')

# --------------------------------------------------------------------

def get_config():
    """Return a dictionary with information about PETSc."""
    import sys, os.path as p
    if sys.version_info[0] >= 3:
        from io import StringIO
        from configparser import ConfigParser
    else:
        from StringIO import StringIO
        from ConfigParser import ConfigParser
    filename = p.join(p.dirname(__file__), 'lib', 'petsc.cfg')
    with open(filename) as fp:
        stream = StringIO("[petsc]\n"+fp.read())
    parser = ConfigParser()
    parser.optionxform = str
    try:
        parser.read_file(stream, filename)
    except AttributeError:
        parser.readfp(stream, filename)
    return dict(parser.items('petsc'))

# --------------------------------------------------------------------
