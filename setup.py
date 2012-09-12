#!/usr/bin/env python

"""
TAO: Toolkit for Advanced Optimization
======================================

The Toolkit for Advanced Optimization (TAO) is aimed at the solution
of large-scale optimization problems on high-performance
architectures. Its main goals are portability, performance, scalable
parallelism, and an interface independent of the architecture. TAO is
suitable for both single-processor and massively-parallel
architectures.

.. tip::

  You can also install `tao-dev`_ with::

    $ pip install petsc==dev tao==dev

  .. _tao-dev: http://bitbucket.org/sarich/tao-devel/
               get/tip.tar.gz#egg=tao-dev
"""

import sys, os
from distutils.core import setup
from distutils.util import get_platform
from distutils.spawn import find_executable
from distutils.command.build import build as _build
if 'setuptools' in sys.modules:
    from setuptools.command.install import install as _install
else:
    from distutils.command.install import install as _install
from distutils.command.sdist import sdist as _sdist
from distutils import log

init_py = """\
# Author:  TAO Team
# Contact: tao-comments@mcs.anl.gov

def get_tao_dir():
    import os
    return os.path.dirname(__file__)

def get_config():
    conf = {}
    conf['TAO_DIR'] = get_tao_dir()
    return conf
"""

metadata = {
    'provides' : ['tao'],
    'requires' : [],
}

def bootstrap():
    from os.path import join, isdir, abspath
    # Set TAO_DIR
    TAO_DIR  = abspath(os.getcwd())
    os.environ['TAO_DIR']  = TAO_DIR
    # Check PETSC_DIR/PETSC_ARCH
    PETSC_DIR  = os.environ.get('PETSC_DIR',  "")
    PETSC_ARCH = os.environ.get('PETSC_ARCH', "")
    if not (PETSC_DIR and isdir(PETSC_DIR)):
        PETSC_DIR = None
        try: del os.environ['PETSC_DIR']
        except KeyError: pass
        PETSC_ARCH = None
        try: del os.environ['PETSC_ARCH']
        except KeyError: pass
    elif not isdir(join(PETSC_DIR, PETSC_ARCH)):
        PETSC_ARCH = None
        try: del os.environ['PETSC_ARCH']
        except KeyError: pass
    # Generate package __init__.py file
    from distutils.dir_util import mkpath
    pkgdir = os.path.join(TAO_DIR, 'pypi')
    pkgfile = os.path.join(pkgdir, '__init__.py')
    if not os.path.exists(pkgdir): mkpath(pkgdir)
    fh = open(pkgfile, 'wt')
    fh.write(init_py)
    fh.close()
    if ('setuptools' in sys.modules):
        metadata['zip_safe'] = False
        if not PETSC_DIR:
            metadata['install_requires']= ['petsc>=3.3,<3.4']

def get_petsc_dir():
    PETSC_DIR = os.environ.get('PETSC_DIR')
    if PETSC_DIR: return PETSC_DIR
    try:
        import petsc
        PETSC_DIR = petsc.get_petsc_dir()
    except ImportError:
        log.warn("PETSC_DIR not specified")
        PETSC_DIR = os.path.join(os.path.sep, 'usr', 'local', 'petsc')
    return PETSC_DIR

def get_petsc_arch():
    PETSC_ARCH = os.environ.get('PETSC_ARCH') or ''
    return PETSC_ARCH

def config(dry_run=False):
    log.info('TAO: configure')
    if dry_run: return
    # Run TAO configure
    os.environ['PETSC_DIR'] = get_petsc_dir()
    status = 0#os.system("./configure")
    if status != 0: raise RuntimeError(status)

def build(dry_run=False):
    log.info('TAO: build')
    if dry_run: return
    # Run TAO build
    status = os.system(" ".join((
            find_executable('make'),
            'PETSC_DIR='+get_petsc_dir(),
            'PETSC_ARCH='+get_petsc_arch(),
            'all',
            )))
    if status != 0: raise RuntimeError

def install(dest_dir, prefix=None, dry_run=False):
    log.info('TAO: install')
    if dry_run: return
    TAO_DIR = os.environ['TAO_DIR']
    PETSC_ARCH = get_petsc_arch()
    # Run TAO install (python)
    from distutils.file_util import copy_file
    from distutils.dir_util import copy_tree
    copy_tree(os.path.join(TAO_DIR,  'include'),
              os.path.join(dest_dir, 'include'))
    copy_tree(os.path.join(TAO_DIR,  'conf'),
              os.path.join(dest_dir, 'conf'))
    copy_tree(os.path.join(TAO_DIR, PETSC_ARCH, 'lib'),
              os.path.join(dest_dir, 'lib'))
    # remove bad files
    badfiles = [os.path.join(dest_dir, 'conf', 'install.py')]
    for dirname, _, filenames in os.walk(
        os.path.join(dest_dir, 'include')):
        badfiles += [os.path.join(dirname, f)
                     for f in filenames
                     if f.endswith('.html')]
    for filename in badfiles:
        fullpath = os.path.join(dirname, filename)
        try:
            os.remove(fullpath)
            log.info("removing %s", fullpath)
        except:
            pass

class context:
    def __init__(self):
        self.sys_argv = sys.argv[:]
        self.wdir = os.getcwd()
    def enter(self):
        del sys.argv[1:]
        pdir = os.environ['TAO_DIR']
        os.chdir(pdir)
        return self
    def exit(self):
        sys.argv[:] = self.sys_argv
        os.chdir(self.wdir)

class cmd_build(_build):

    def initialize_options(self):
        _build.initialize_options(self)
        PETSC_ARCH = get_petsc_arch()
        self.build_base = os.path.join(PETSC_ARCH, 'build-python')

    def run(self):
        _build.run(self)
        ctx = context().enter()
        try:
            config(self.dry_run)
            build(self.dry_run)
        finally:
            ctx.exit()

class cmd_install(_install):

    def initialize_options(self):
        _install.initialize_options(self)
        self.optimize = 1

    def run(self):
        root_dir = self.install_platlib
        dest_dir = os.path.join(root_dir, 'tao')
        bdist_base = self.get_finalized_command('bdist').bdist_base
        if dest_dir.startswith(bdist_base):
            prefix = dest_dir[len(bdist_base)+1:]
            prefix = prefix[prefix.index(os.path.sep):]
        else:
            prefix = dest_dir
        dest_dir = os.path.abspath(dest_dir)
        prefix   = os.path.abspath(prefix)
        #
        _install.run(self)
        ctx = context().enter()
        try:
            install(dest_dir, prefix, self.dry_run)
        finally:
            ctx.exit()

manifest_in = """\
include makefile LICENSE

recursive-include include *
recursive-include src *
recursive-include conf *

recursive-exclude include *.html
recursive-exclude src *.html
recursive-exclude src/*/examples/* *.*
recursive-exclude pypi *.*
"""

class cmd_sdist(_sdist):

    def initialize_options(self):
        _sdist.initialize_options(self)
        self.force_manifest = 1
        self.template = os.path.join('pypi', 'manifest.in')
        # Generate manifest.in file
        from distutils.dir_util import mkpath
        TAO_DIR = os.environ['TAO_DIR']
        pkgdir = os.path.join(TAO_DIR, 'pypi')
        if not os.path.exists(pkgdir): mkpath(pkgdir)
        template = self.template
        fh = open(template, 'wt')
        fh.write(manifest_in)
        fh.close()

def version():
    import re
    version_re = {
        'major'  : re.compile(r"#define\s+TAO_VERSION_MAJOR\s+(\d+)"),
        'minor'  : re.compile(r"#define\s+TAO_VERSION_MINOR\s+(\d+)"),
        'micro'  : re.compile(r"#define\s+TAO_VERSION_SUBMINOR\s+(\d+)"),
        'patch'  : re.compile(r"#define\s+TAO_VERSION_PATCH\s+(\d+)"),
        'release': re.compile(r"#define\s+TAO_VERSION_RELEASE\s+(\d+)"),
        }
    taoversion_h = os.path.join('include','tao_version.h')
    data = open(taoversion_h, 'rt').read()
    major = int(version_re['major'].search(data).groups()[0])
    minor = int(version_re['minor'].search(data).groups()[0])
    micro = int(version_re['micro'].search(data).groups()[0])
    patch = int(version_re['patch'].search(data).groups()[0])
    release = int(version_re['release'].search(data).groups()[0])
    if release:
        v = "%d.%d" % (major, minor)
        if micro > 0:
            v += ".%d" % micro
        if patch > 0:
            v += ".%d" % patch
    else:
        v = "%d.%d.dev%d" % (major, minor+1, 0)
    return v

def tarball():
    VERSION = version()
    if '.dev' in VERSION:
        return None
    bits = VERSION.split('.')
    if len(bits) == 2: bits.append('0')
    TAO_VERSION = '.'.join(bits[:-1]) + '-p' + bits[-1]
    return ('http://www.mcs.anl.gov/research/projects/tao/download/'
            'tao-%s.tar.gz#egg=tao-%s' % (TAO_VERSION, VERSION))

description = __doc__.split('\n')[1:-1]; del description[1:3]
classifiers = """
License :: OSI Approved
Operating System :: POSIX
Intended Audience :: Developers
Intended Audience :: Science/Research
Programming Language :: C
Programming Language :: C++
Programming Language :: Fortran
Programming Language :: Python
Topic :: Scientific/Engineering
Topic :: Software Development :: Libraries
"""

bootstrap()
setup(name='tao',
      version=version(),
      description=description.pop(0),
      long_description='\n'.join(description),
      classifiers= classifiers.split('\n')[1:-1],
      keywords = ['TAO', 'PETSc', 'MPI'],
      platforms=['POSIX'],
      license='TAO',

      url='http://www.mcs.anl.gov/tao/',
      download_url=tarball(),

      author='TAO Team',
      author_email='tao-comments@mcs.anl.gov',
      maintainer='Lisandro Dalcin',
      maintainer_email='dalcinl@gmail.com',

      packages = ['tao'],
      package_dir = {'tao': 'pypi'},
      cmdclass={
        'build': cmd_build,
        'install': cmd_install,
        'sdist': cmd_sdist,
        },
      **metadata)
