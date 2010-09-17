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


# -----------------------------------------------------------------------------
# Metadata
# -----------------------------------------------------------------------------

from conf.metadata import metadata

def version():
    import os, re
    data = open(os.path.join('src', '__init__.py')).read()
    m = re.search(r"__version__\s*=\s*'(.*)'", data)
    return m.groups()[0]

name     = 'petsc4py'
version  = version()

url      = 'http://%(name)s.googlecode.com/' % vars()
download = url + 'files/%(name)s-%(version)s.tar.gz' % vars()

descr    = __doc__.strip().split('\n'); del descr[1:3]
devstat  = ['Development Status :: 3 - Alpha']
keywords = ['PETSc', 'MPI']

metadata['name'] = name
metadata['version'] = version
metadata['description'] = descr.pop(0)
metadata['long_description'] = '\n'.join(descr)
metadata['keywords'] += keywords
metadata['classifiers'] += devstat
metadata['url'] = url
metadata['download_url'] = download

metadata['provides'] = ['petsc4py']
metadata['requires'] = ['numpy']

# -----------------------------------------------------------------------------
# Extension modules
# -----------------------------------------------------------------------------

def get_ext_modules(Extension):
    from os   import walk, path
    from glob import glob
    depends = []
    for pth, dirs, files in walk('src'):
        depends += glob(path.join(pth, '*.h'))
    for pth, dirs, files in walk(path.join('src', 'source')):
        depends += glob(path.join(pth, '*.h'))
        depends += glob(path.join(pth, '*.c'))
    try:
        import numpy
        numpy_includes = [numpy.get_include()]
    except ImportError:
        numpy_includes = []
    return [Extension('petsc4py.lib.PETSc',
                      sources=['src/PETSc.c',
                               'src/source/libpetsc4py.c',
                               ],
                      include_dirs=['src/include',
                                    'src/source',
                                    ] + numpy_includes,
                      depends=depends)]

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------

from conf.petscconf import setup, Extension
from conf.petscconf import config, build, build_ext

def run_setup():
    import sys, os
    if (('distribute' in sys.modules) or
        ('setuptools' in sys.modules)):
        metadata['install_requires'] = ['numpy']
        if not os.environ.get('PETSC_DIR'):
            metadata['install_requires'].append('petsc')
    if 'setuptools' in sys.modules:
        metadata['zip_safe'] = False
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
                          'build_ext'  : build_ext},
          **metadata)

def run_cython(source):
    import sys, os
    source_c = os.path.splitext(source)[0] + '.c'
    if (os.path.exists(source_c)):
        return False
    try:
        import Cython
    except ImportError:
        warn = lambda msg='': sys.stderr.write(msg+'\n')
        warn("*"*80)
        warn()
        warn(" You need to generate C source files with Cython!!")
        warn(" Download and install Cython <http://www.cython.org>")
        warn()
        warn("*"*80)
        raise SystemExit
    from distutils import log
    from conf.cythonize import run as cythonize
    log.info("cythonizing '%s' source" % source)
    cythonize(source)
    return True

def main():
    import os
    try:
        run_setup()
    except:
        done = run_cython(os.path.join('src', 'petsc4py.PETSc.pyx'))
        if done:
            run_setup()
        else:
            raise

if __name__ == '__main__':
    main()

# -----------------------------------------------------------------------------
