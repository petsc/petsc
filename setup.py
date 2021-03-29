#!/usr/bin/env python

"""
PETSc: Portable, Extensible Toolkit for Scientific Computation
==============================================================

The Portable, Extensible Toolkit for Scientific Computation (PETSc),
is a suite of data structures and routines for the scalable (parallel)
solution of scientific applications modeled by partial differential
equations. It employs the Message Passing Interface (MPI) standard for
all message-passing communication.

.. note::

   To install ``PETSc`` and ``petsc4py`` (``mpi4py`` is optional
   but highly recommended) use::

     $ python -m pip install numpy mpi4py  (or pip install numpy mpi4py)
     $ python -m pip install petsc petsc4py (or pip install petsc petsc4py)

.. tip::

  You can also install the in-development versions with::

    $ python -m pip install Cython numpy mpi4py
    $ python -m pip install --no-deps https://gitlab.com/petsc/petsc/-/archive/main/petsc-main.tar.gz

  To set the MPI compilers use the environmental variables ``MPICC``, ``MPICXX``, ``MPIF90``.

  Provide any ``PETSc`` ./configure options using the environmental variable ``PETSC_CONFIGURE_OPTIONS``.

  Do not use the ``PETSc`` ``./configure`` options ``--with-cc``, ``--with-cxx``, ``--with-fc``, or ``--with-mpi-dir``.

  If ``mpi4py`` is installed the compilers will obtained from that installation and ``MPICC``, ``MPICXX``, ``MPIF90`` will be ignored.

"""

import sys, os
from setuptools import setup
from setuptools.command.install import install as _install
from distutils.util import get_platform, split_quoted
from distutils.spawn import find_executable
from distutils import log

init_py = """\
# Author:  PETSc Team
# Contact: petsc-maint@mcs.anl.gov

def get_petsc_dir():
    import os
    return os.path.dirname(__file__)

def get_config():
    conf = {}
    conf['PETSC_DIR'] = get_petsc_dir()
    return conf
"""

metadata = {
    'provides' : ['petsc'],
    'zip_safe' : False,
}

CONFIGURE_OPTIONS = []

def bootstrap():
    # Set PETSC_DIR and PETSC_ARCH
    PETSC_DIR  = os.path.abspath(os.getcwd())
    PETSC_ARCH = 'arch-python-' + get_platform()
    os.environ['PETSC_DIR']  = PETSC_DIR
    os.environ['PETSC_ARCH'] = PETSC_ARCH
    sys.path.insert(0, os.path.join(PETSC_DIR, 'config'))
    sys.path.insert(0, os.path.join(PETSC_DIR, 'lib','petsc','conf'))
    # Generate package __init__.py file
    from distutils.dir_util import mkpath
    pkgdir = os.path.join('config', 'pypi')
    if not os.path.exists(pkgdir): mkpath(pkgdir)
    pkgfile = os.path.join(pkgdir, '__init__.py')
    fh = open(pkgfile, 'w')
    fh.write(init_py)
    fh.close()
    # Configure options
    options = os.environ.get('PETSC_CONFIGURE_OPTIONS', '')
    CONFIGURE_OPTIONS.extend(split_quoted(options))
    for i in CONFIGURE_OPTIONS:
        if i.startswith('--with-mpi-dir='):
            raise RuntimeError("Do not use --with-mpi-dir, use the environmental variables MPICC, MPICXX, MPIF90")
        if i.startswith('--with-cc='):
            raise RuntimeError("Do not use --with-cc, use the environmental variable MPICC")
        if i.startswith('--with-cxx=') and i != "--with-cxx=0":
            raise RuntimeError("Do not use --with-cxx, use the environmental variable MPICXX")
        if i.startswith('--with-fc=') and i != "--with-fc=0":
            raise RuntimeError("Do not use --with-fc, use the environmental variable MPIF90")

    if '--with-mpi=0' not in CONFIGURE_OPTIONS:
        # Simple-minded lookup for MPI and mpi4py
        mpi4py = mpicc = None
        try:
            import mpi4py
            conf = mpi4py.get_config()
            mpicc = conf.get('mpicc')
        except ImportError: # mpi4py is not installed
            mpi4py = None
            mpicc = (os.environ.get('MPICC') or
                     find_executable('mpicc'))
        except AttributeError: # mpi4py is too old
            pass
        if not mpi4py and mpicc:
            metadata['install_requires'] = ['mpi4py>=1.2.2']

def config(prefix, dry_run=False):
    log.info('PETSc: configure')
    options = [
        '--prefix=' + prefix,
        'PETSC_ARCH='+os.environ['PETSC_ARCH'],
        '--with-shared-libraries=1',
        '--with-debugging=0',
        '--with-c2html=0', # not needed
        ]
    if '--with-fc=0' in CONFIGURE_OPTIONS:
        options.append('--with-sowing=0')
    if '--with-mpi=0' not in CONFIGURE_OPTIONS:
        try:
            import mpi4py
            conf = mpi4py.get_config()
            mpicc  = conf.get('mpicc')
            mpicxx = conf.get('mpicxx')
            mpif90 = conf.get('mpif90')
        except (ImportError, AttributeError):
            mpicc  = os.environ.get('MPICC')  or find_executable('mpicc')
            mpicxx = os.environ.get('MPICXX') or find_executable('mpicxx')
            mpif90 = os.environ.get('MPIF90') or find_executable('mpif90')
        if mpicc:
            options.append('--with-cc='+mpicc)
            if '--with-cxx=0' not in CONFIGURE_OPTIONS:
                if mpicxx:
                    options.append('--with-cxx='+mpicxx)
                else:
                    options.append('--with-cxx=0')
            if '--with-fc=0' not in CONFIGURE_OPTIONS:
                if mpif90:
                    options.append('--with-fc='+mpif90)
                else:
                    options.append('--with-fc=0')
                    options.append('--with-sowing=0')
        else:
            options.append('--with-mpi=0')
    options.extend(CONFIGURE_OPTIONS)
    #
    log.info('configure options:')
    for opt in options:
        log.info(' '*4 + opt)
    # Run PETSc configure
    if dry_run: return
    use_config_py = False
    if use_config_py:
        import configure
        configure.petsc_configure(options)
        import logger
        logger.Logger.defaultLog = None
    else:
        python = find_executable('python2') or find_executable('python')
        command = [python, './configure'] + options
        status = os.system(" ".join(command))
        if status != 0: raise RuntimeError(status)

def build(dry_run=False):
    log.info('PETSc: build')
    # Run PETSc build
    if dry_run: return
    use_builder_py = False
    if use_builder_py:
        import builder
        builder.PETScMaker().run()
        import logger
        logger.Logger.defaultLog = None
    else:
        make = find_executable('make')
        command = [make, 'all']
        status = os.system(" ".join(command))
        if status != 0: raise RuntimeError(status)

def install(dry_run=False):
    log.info('PETSc: install')
    # Run PETSc installer
    if dry_run: return
    use_install_py = False
    if use_install_py:
        import install
        install.Installer().run()
        import logger
        logger.Logger.defaultLog = None
    else:
        make = find_executable('make')
        command = [make, 'install']
        status = os.system(" ".join(command))
        if status != 0: raise RuntimeError(status)

class context(object):
    def __init__(self):
        self.sys_argv = sys.argv[:]
        self.wdir = os.getcwd()
    def enter(self):
        del sys.argv[1:]
        pdir = os.environ['PETSC_DIR']
        os.chdir(pdir)
        return self
    def exit(self):
        sys.argv[:] = self.sys_argv
        os.chdir(self.wdir)

class cmd_install(_install):

    def initialize_options(self):
        _install.initialize_options(self)
        self.optimize = 1

    def finalize_options(self):
        _install.finalize_options(self)
        self.install_lib = self.install_platlib
        self.install_libbase = self.install_lib

    def run(self):
        root_dir = os.path.abspath(self.install_lib)
        prefix = os.path.join(root_dir, 'petsc')
        #
        ctx = context().enter()
        try:
            config(prefix, self.dry_run)
            build(self.dry_run)
            install(self.dry_run)
        finally:
            ctx.exit()
        #
        self.outputs = []
        for dirpath, _, filenames in os.walk(prefix):
            for fn in filenames:
                self.outputs.append(os.path.join(dirpath, fn))
        #
        _install.run(self)

    def get_outputs(self):
        outputs = getattr(self, 'outputs', [])
        outputs += _install.get_outputs(self)
        return outputs

def version():
    import re
    version_re = {
        'major'  : re.compile(r"#define\s+PETSC_VERSION_MAJOR\s+(\d+)"),
        'minor'  : re.compile(r"#define\s+PETSC_VERSION_MINOR\s+(\d+)"),
        'micro'  : re.compile(r"#define\s+PETSC_VERSION_SUBMINOR\s+(\d+)"),
        'release': re.compile(r"#define\s+PETSC_VERSION_RELEASE\s+([-]*\d+)"),
        }
    petscversion_h = os.path.join('include','petscversion.h')
    data = open(petscversion_h, 'r').read()
    major = int(version_re['major'].search(data).groups()[0])
    minor = int(version_re['minor'].search(data).groups()[0])
    micro = int(version_re['micro'].search(data).groups()[0])
    release = int(version_re['release'].search(data).groups()[0])
    if release > 0 :
        v = "%d.%d.%d" % (major, minor, micro)
    else:
        v = "%d.%d.0.dev%d" % (major, minor+1, 0)
    return v

def tarball():
    VERSION = version()
    if '.dev' in VERSION: return None
    return ('http://ftp.mcs.anl.gov/pub/petsc/release-snapshots//'
            'petsc-%s.tar.gz#egg=petsc-%s' % (VERSION, VERSION))

description = __doc__.split('\n')[1:-1]; del description[1:3]
classifiers = """
Development Status :: 5 - Production/Stable
Intended Audience :: Developers
Intended Audience :: Science/Research
License :: OSI Approved :: BSD License
Operating System :: POSIX
Programming Language :: C
Programming Language :: C++
Programming Language :: Fortran
Programming Language :: Python
Topic :: Scientific/Engineering
Topic :: Software Development :: Libraries
"""

if 'bdist_wheel' in sys.argv:
    sys.stderr.write("petsc: this package cannot be built as a wheel\n")
    sys.exit(1)

bootstrap()
setup(name='petsc',
      version=version(),
      description=description.pop(0),
      long_description='\n'.join(description),
      classifiers= classifiers.split('\n')[1:-1],
      keywords = ['PETSc', 'MPI'],
      platforms=['POSIX'],
      license='BSD',

      url='https://www.mcs.anl.gov/petsc/',
      download_url=tarball(),

      author='PETSc Team',
      author_email='petsc-maint@mcs.anl.gov',
      maintainer='Lisandro Dalcin',
      maintainer_email='dalcinl@gmail.com',

      packages = ['petsc'],
      package_dir = {'petsc': 'config/pypi'},
      cmdclass={'install': cmd_install},
      **metadata)
