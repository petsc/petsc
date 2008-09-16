# Author:  Lisandro Dalcin
# Contact: dalcinl@gmail.com
# Id: $Id$

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

.. _PETSc: http://www-unix.mcs.anl.gov/petsc
.. _MPI:   http://www.mpi-forum.org

"""

__author__    = 'Lisandro Dalcin'
__version__   = '1.0.0'
__credits__   = "PETSc Team <petsc-maint@mcs.anl.gov>"

# --------------------------------------------------------------------

def init(args=None, arch=None):
    """
    Initializes PETSc.

    :Parameters:
      - `args`: command-line arguments, usually the 'sys.argv' list.
      - `arch`: specific configuration to use.

    .. note:: This function should be called only once, typically at
      the very beginning of the bootstrap script of an application.
    """
    import petsc4py.lib
    PETSc = petsc4py.lib.Import(arch)
    if args is None:
        args = []
    elif isinstance(args, str):
        args = str(args).split()
    else:
        args = [str(a) for a in args]
        args = [a for a in args if a]
    if args and args[0].startswith('-'):
        import sys
        argv = getattr(sys, 'argv', None)
        program_name = None
        if argv and type(argv[0]) is str:
            program_name = argv[0]
        if not program_name:
            program_name = getattr(sys, 'executable', 'python')
        args.insert(0, program_name)
    PETSc._initialize(args)

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

if __name__ == '__main__':
    import petsc4py.help
    petsc4py.help.help()

# --------------------------------------------------------------------
