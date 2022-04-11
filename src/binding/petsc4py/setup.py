#!/usr/bin/env python
# Author:  Lisandro Dalcin
# Contact: dalcinl@gmail.com

"""
PETSc for Python
"""

import sys
import os
import re

try:
    import setuptools
except ImportError:
    setuptools = None

pyver = sys.version_info[:2]
if pyver < (2, 6) or (3, 0) <= pyver < (3, 2):
    raise RuntimeError("Python version 2.6, 2.7 or >= 3.2 required")
if pyver == (2, 6) or pyver == (3, 2):
    sys.stderr.write(
        "WARNING: Python %d.%d is not supported.\n" % pyver)

# --------------------------------------------------------------------
# Metadata
# --------------------------------------------------------------------

topdir = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, topdir)

from conf.metadata import metadata

def name():
    return 'petsc4py'

def version():
    with open(os.path.join(topdir, 'src', '__init__.py')) as f:
        m = re.search(r"__version__\s*=\s*'(.*)'", f.read())
        return m.groups()[0]

def description():
    with open(os.path.join(topdir, 'DESCRIPTION.rst')) as f:
        return f.read()

name     = name()
version  = version()

url      = 'https://gitlab.com/petsc/petsc'
pypiroot = 'https://pypi.io/packages/source/%s/%s/' % (name[0], name)
download = pypiroot + '%(name)s-%(version)s.tar.gz' % vars()

devstat  = ['Development Status :: 5 - Production/Stable']
keywords = ['PETSc', 'MPI']

metadata['name'] = name
metadata['version'] = version
metadata['description'] = __doc__.strip()
metadata['long_description'] = description()
metadata['keywords'] += keywords
metadata['classifiers'] += devstat
metadata['url'] = url
metadata['download_url'] = download

metadata['provides'] = ['petsc4py']
metadata['requires'] = ['numpy']

# --------------------------------------------------------------------
# Extension modules
# --------------------------------------------------------------------

def get_ext_modules(Extension):
    from os import walk
    from glob import glob
    from os.path import join
    glob_join = lambda *args: glob(join(*args))
    depends = []
    for pth, dirs, files in walk('src'):
        depends += glob_join(pth, '*.h')
        depends += glob_join(pth, '*.c')
    if 'PETSC_DIR' in os.environ:
        pd = os.environ['PETSC_DIR']
        pa = os.environ.get('PETSC_ARCH', '')
        depends += glob_join(pd, 'include', '*.h')
        depends += glob_join(pd, 'include', 'petsc', 'private', '*.h')
        depends += glob_join(pd, pa, 'include', 'petscconf.h')
    numpy_include = os.environ.get('NUMPY_INCLUDE')
    if numpy_include is not None:
        numpy_includes = [numpy_include]
    else:
        try:
            import numpy
            numpy_includes = [numpy.get_include()]
        except ImportError:
            numpy_includes = []
    return [Extension('petsc4py.lib.PETSc',
                      sources=['src/PETSc.c',
                               'src/libpetsc4py.c',
                               ],
                      include_dirs=['src/include',
                                    ] + numpy_includes,
                      depends=depends)]

# --------------------------------------------------------------------
# Setup
# --------------------------------------------------------------------

from conf.petscconf import setup, Extension
from conf.petscconf import config, build, build_src, build_ext, install
from conf.petscconf import clean, test, sdist

CYTHON = '0.24'

def get_release():
    release = 1
    if topdir.endswith(os.path.join(os.path.sep, 'src', 'binding', name)):
        topname = name.replace('4py', '')
        rootdir = os.path.abspath(os.path.join(topdir, *[os.path.pardir]*3))
        version_h = os.path.join(rootdir, 'include', '%sversion.h' % topname)
        release_macro = '%s_VERSION_RELEASE' % topname.upper()
        version_re = re.compile(r"#define\s+%s\s+([-]*\d+)" % release_macro)
        if os.path.exists(version_h) and os.path.isfile(version_h):
            with open(version_h, 'r') as f:
                release = int(version_re.search(f.read()).groups()[0])
    return bool(release)

def requires(pkgname, major, minor, release=True):
    minor = minor + int(not release)
    devel = '' if release else '.dev0'
    vmin = "%s.%s%s" % (major, minor, devel)
    vmax = "%s.%s" % (major, minor + 1)
    return "%s>=%s,<%s" % (pkgname, vmin, vmax)

def run_setup():
    setup_args = metadata.copy()
    vstr = setup_args['version'].split('.')[:2]
    x, y = tuple(map(int, vstr))
    release = get_release()
    if not release:
        setup_args['version'] = "%d.%d.0.dev0" %(x, y+1)
    if setuptools:
        setup_args['zip_safe'] = False
        setup_args['install_requires'] = ['numpy']
        PETSC_DIR = os.environ.get('PETSC_DIR')
        if not (PETSC_DIR and os.path.isdir(PETSC_DIR)):
            petsc = requires('petsc', x, y, release)
            setup_args['install_requires'] += [petsc]
    if setuptools:
        src = os.path.join('src', 'petsc4py.PETSc.c')
        has_src = os.path.exists(os.path.join(topdir, src))
        has_git = os.path.isdir(os.path.join(topdir, '.git'))
        has_hg  = os.path.isdir(os.path.join(topdir, '.hg'))
        suffix = os.path.join('src', 'binding', 'petsc4py')
        in_petsc = topdir.endswith(os.path.sep + suffix)
        if not has_src or has_git or has_hg or in_petsc:
            setup_args['setup_requires'] = ['Cython>='+CYTHON]
    #
    setup(packages     = ['petsc4py',
                          'petsc4py.lib',],
          package_dir  = {'petsc4py'     : 'src',
                          'petsc4py.lib' : 'src/lib'},
          package_data = {'petsc4py'     : ['include/petsc4py/*.h',
                                            'include/petsc4py/*.i',
                                            'include/petsc4py/*.pxd',
                                            'include/petsc4py/*.pxi',
                                            'include/petsc4py/*.pyx',
                                            'PETSc.pxd',],
                          'petsc4py.lib' : ['petsc.cfg'],},
          ext_modules  = get_ext_modules(Extension),
          cmdclass     = {'config'     : config,
                          'build'      : build,
                          'build_src'  : build_src,
                          'build_ext'  : build_ext,
                          'install'    : install,
                          'clean'      : clean,
                          'test'       : test,
                          'sdist'      : sdist,
                          },
          **setup_args)

def chk_cython(VERSION):
    from distutils import log
    from distutils.version import LooseVersion
    from distutils.version import StrictVersion
    warn = lambda msg='': sys.stderr.write(msg+'\n')
    #
    try:
        import Cython
    except ImportError:
        warn("*"*80)
        warn()
        warn(" You need to generate C source files with Cython!!")
        warn(" Download and install Cython <http://www.cython.org>")
        warn()
        warn("*"*80)
        return False
    #
    try:
        CYTHON_VERSION = Cython.__version__
    except AttributeError:
        from Cython.Compiler.Version import version as CYTHON_VERSION
    REQUIRED = VERSION
    m = re.match(r"(\d+\.\d+(?:\.\d+)?).*", CYTHON_VERSION)
    if m:
        Version = StrictVersion
        AVAILABLE = m.groups()[0]
    else:
        Version = LooseVersion
        AVAILABLE = CYTHON_VERSION
    if (REQUIRED is not None and
        Version(AVAILABLE) < Version(REQUIRED)):
        warn("*"*80)
        warn()
        warn(" You need to install Cython %s (you have version %s)"
             % (REQUIRED, CYTHON_VERSION))
        warn(" Download and install Cython <http://www.cython.org>")
        warn()
        warn("*"*80)
        return False
    #
    return True

def run_cython(source, target=None,
               depends=(), includes=(),
               destdir_c=None, destdir_h=None,
               wdir=None, force=False, VERSION=None):
    from glob import glob
    from distutils import log
    from distutils import dep_util
    from distutils.errors import DistutilsError
    if target is None:
        target = os.path.splitext(source)[0]+'.c'
    cwd = os.getcwd()
    try:
        if wdir: os.chdir(wdir)
        alldeps = [source]
        for dep in depends:
            alldeps += glob(dep)
        if not (force or dep_util.newer_group(alldeps, target)):
            log.debug("skipping '%s' -> '%s' (up-to-date)",
                      source, target)
            return
    finally:
        os.chdir(cwd)
    if not chk_cython(VERSION):
        raise DistutilsError("requires Cython>=%s" % VERSION)
    log.info("cythonizing '%s' -> '%s'", source, target)
    from conf.cythonize import cythonize
    err = cythonize(source, target,
                    includes=includes,
                    destdir_c=destdir_c,
                    destdir_h=destdir_h,
                    wdir=wdir)
    if err:
        raise DistutilsError(
            "Cython failure: '%s' -> '%s'" % (source, target))

def build_sources(cmd):
    from os.path import exists, isdir, join

    # petsc4py.PETSc
    source = 'petsc4py.PETSc.pyx'
    target = 'petsc4py.PETSc.c'
    depends = ['include/*/*.pxd',
               'PETSc/*.pyx',
               'PETSc/*.pxi']
    includes = ['include']
    destdir_h = os.path.join('include', 'petsc4py')
    run_cython(source, target,
               depends=depends, includes=includes,
               destdir_c=None, destdir_h=destdir_h, wdir='src',
               force=cmd.force, VERSION=CYTHON)
    # libpetsc4py
    source = os.path.join('libpetsc4py', 'libpetsc4py.pyx')
    depends = ['include/petsc4py/*.pxd',
               'libpetsc4py/*.pyx',
               'libpetsc4py/*.pxi']
    includes = ['include']
    run_cython(source,
               depends=depends, includes=includes,
               destdir_c=None, destdir_h=None, wdir='src',
               force=cmd.force, VERSION=CYTHON)

build_src.run = build_sources

def run_testsuite(cmd):
    from distutils.errors import DistutilsError
    sys.path.insert(0, 'test')
    try:
        from runtests import main
    finally:
        del sys.path[0]
    if cmd.dry_run:
        return
    args = cmd.args[:] or []
    if cmd.verbose < 1:
        args.insert(0,'-q')
    if cmd.verbose > 1:
        args.insert(0,'-v')
    err = main(args)
    if err:
        raise DistutilsError("test")

test.run = run_testsuite

# --------------------------------------------------------------------

def main():
    run_setup()

if __name__ == '__main__':
    main()

# --------------------------------------------------------------------
