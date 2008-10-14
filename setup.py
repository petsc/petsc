#!/usr/bin/env python

"""
PETSc for Python
================

Python bindings for PETSc libraries.
"""


## try:
##     import setuptools
## except ImportError:
##     pass


# --------------------------------------------------------------------
# Metadata
# --------------------------------------------------------------------

from conf import metadata

name     = 'petsc4py'
version  = open('VERSION.txt').read().strip()
descr    = __doc__.strip().split('\n'); del descr[1:3]
devstat  = ['Development Status :: 3 - Alpha']
url      = 'http://%s.googlecode.com/' % name
download = 'http://%s.googlecode.com/files/%s-%s.tar.gz'
download = download % (name, name, version)

metadata['name'] = name
metadata['version'] = version
metadata['description'] = descr.pop(0)
metadata['long_description'] = '\n'.join(descr)
metadata['classifiers'] += devstat
metadata['url'] = url
metadata['download_url'] = download

# --------------------------------------------------------------------
# Extension modules
# --------------------------------------------------------------------

def get_ext_modules(Extension):
    from os   import walk, path
    from glob import glob
    depends = []
    for pth, dirs, files in walk('src'):
        depends += glob(path.join(pth, '*.h'))
    for pth, dirs, files in walk(path.join('src', 'source')):
        depends += glob(path.join(pth, '*.c'))
    return [Extension('petsc4py.lib.PETSc',
                      sources=['src/PETSc.c',
                               'src/source/libpetsc4py.c',],
                      include_dirs=['src/include',
                                    'src/source',],
                      depends=depends, language='c'),
            Extension('petsc4py.lib.PETSc',
                      sources=['src/PETSc.cpp',
                               'src/source/libpetsc4py.cpp',],
                      include_dirs=['src/include',
                                    'src/source',],
                      depends=depends, language='c++'),
            ]
# --------------------------------------------------------------------
# Setup
# --------------------------------------------------------------------

from conf.core import setup, Extension
from conf.core import config, build, build_py, build_ext

def main():
    setup(packages     = ['petsc4py',
                          'petsc4py.lib',],
          package_dir  = {'petsc4py'     : 'src',
                          'petsc4py.lib' : 'src/lib'},
          package_data = {'petsc4py'     : ['include/petsc4py/*.h',
                                            'include/petsc4py/*.i',
                                            'include/petsc4py/*.pxd',
                                            'include/petsc4py/*.pxi',
                                            'include/petsc4py/*.pyx',],
                          'petsc4py.lib' : ['petsc.cfg'],},
          ext_modules  = get_ext_modules(Extension),
          cmdclass     = {'config'     : config,
                          'build'      : build,
                          'build_py'   : build_py,
                          'build_ext'  : build_ext},
          **metadata)

if __name__ == '__main__':
    import sys, os
    C_SOURCE = os.path.join('src', 'petsc4py_PETSc.c')
    def cython_help():
        if os.path.exists(C_SOURCE): return
        warn = lambda msg='': sys.stderr.write(msg+'\n')
        warn("*"*70)
        warn()
        warn("You need to generate C source files with Cython !!!")
        warn("Please execute in your shell:")
        warn()
        warn("$ python ./conf/cythonize.py")
        warn()
        warn("*"*70)
        warn()
    ## from distutils import log
    ## log.set_verbosity(log.DEBUG)
    cython_help()
    main()

# --------------------------------------------------------------------
