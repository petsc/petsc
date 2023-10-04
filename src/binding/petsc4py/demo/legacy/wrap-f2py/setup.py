#!/usr/bin/env python

#$ python setup.py build_ext --inplace

# a bit of monkeypatching ...
try:
    from numpy.distutils.fcompiler import FCompiler
    def runtime_library_dir_option(self, dir):
        return self.c_compiler.runtime_library_dir_option(dir)
    FCompiler.runtime_library_dir_option = \
        runtime_library_dir_option
except Exception:
    pass


def configuration(parent_package='',top_path=None):
    INCLUDE_DIRS = []
    LIBRARY_DIRS = []
    LIBRARIES    = []

    # PETSc
    import os
    PETSC_DIR  = os.environ['PETSC_DIR']
    PETSC_ARCH = os.environ.get('PETSC_ARCH', '')
    from os.path import join, isdir
    if PETSC_ARCH and isdir(join(PETSC_DIR, PETSC_ARCH)):
        INCLUDE_DIRS += [join(PETSC_DIR, PETSC_ARCH, 'include'),
                         join(PETSC_DIR, 'include')]
        LIBRARY_DIRS += [join(PETSC_DIR, PETSC_ARCH, 'lib')]
    else:
        if PETSC_ARCH: pass # XXX should warn ...
        INCLUDE_DIRS += [join(PETSC_DIR, 'include')]
        LIBRARY_DIRS += [join(PETSC_DIR, 'lib')]
    LIBRARIES += [#'petscts', 'petscsnes', 'petscksp',
                  #'petscdm', 'petscmat',  'petscvec',
                  'petsc']

    # PETSc for Python
    import petsc4py
    INCLUDE_DIRS += [petsc4py.get_include()]

    # Configuration
    from numpy.distutils.misc_util import Configuration
    config = Configuration('', parent_package, top_path)
    config.add_extension('Bratu2D',
                         sources = ['Bratu2D.pyf',
                                    'Bratu2D.F90'],
                         depends = ['Bratu2Dmodule.h'],
                         f2py_options=['--quiet'],
                         define_macros=[('F2PY_REPORT_ON_ARRAY_COPY',1)],
                         include_dirs=INCLUDE_DIRS + [os.curdir],
                         libraries=LIBRARIES,
                         library_dirs=LIBRARY_DIRS,
                         runtime_library_dirs=LIBRARY_DIRS)
    return config

if __name__ == "__main__":
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
