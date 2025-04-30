#!/usr/bin/env python3

"""
PETSc: Portable, Extensible Toolkit for Scientific Computation
==============================================================

The Portable, Extensible Toolkit for Scientific Computation (PETSc),
is a suite of data structures and routines for the scalable (parallel)
solution of scientific applications modeled by partial differential
equations. It employs the Message Passing Interface (MPI) standard for
all message-passing communication.

.. note::

   To install the ``PETSc`` and ``petsc4py`` packages use::

     $ python -m pip install numpy
     $ python -m pip install petsc petsc4py

.. tip::

  You can also install the in-development versions with::

    $ python -m pip install cython numpy
    $ python -m pip install --no-deps https://gitlab.com/petsc/petsc/-/archive/main/petsc-main.tar.gz

  Provide any ``PETSc`` ``./configure`` options using the environmental variable ``PETSC_CONFIGURE_OPTIONS``.

  Do not use the ``PETSc`` ``./configure`` options ``--with-cc``, ``--with-cxx``, ``--with-fc``, or ``--with-mpi-dir``.
  Compilers are detected from (in order): 1) the environmental variables ``MPICC``, ``MPICXX``, ``MPIFORT``,
  or 2) from running the ``which mpicc``, ``which mpicxx``, ``which mpifort`` commands.

"""

import re
import os
import sys
import shlex
import shutil
from setuptools import setup
from setuptools.command.install import install as _install
try:
    from setuptools.command.bdist_wheel import bdist_wheel as _bdist_wheel
except ImportError:
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel
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

main_py = """\
# Author:  PETSc Team
# Contact: petsc-maint@mcs.anl.gov

if __name__ == "__main__":
    import sys
    if "--prefix" in sys.argv:
        from . import get_petsc_dir
        print(get_petsc_dir())
        del get_petsc_dir
    del sys
"""

metadata = {
    'provides' : ['petsc'],
    'zip_safe' : False,
}

CONFIGURE_OPTIONS = []


def bootstrap():
    # Set PETSC_DIR and PETSC_ARCH
    PETSC_DIR  = os.path.abspath(os.getcwd())
    PETSC_ARCH = 'arch-python'
    os.environ['PETSC_DIR']  = PETSC_DIR
    os.environ['PETSC_ARCH'] = PETSC_ARCH
    sys.path.insert(0, os.path.join(PETSC_DIR, 'config'))
    sys.path.insert(0, os.path.join(PETSC_DIR, 'lib','petsc','conf'))

    # Generate package __init__.py and __main__.py files
    pkgdir = os.path.join('config', 'pypi')
    os.makedirs(pkgdir, exist_ok=True)
    for pyfile, contents in (
        ('__init__.py', init_py),
        ('__main__.py', main_py),
    ):
        with open(os.path.join(pkgdir, pyfile), 'w') as fh:
            fh.write(contents)

    # Configure options
    options = os.environ.get('PETSC_CONFIGURE_OPTIONS', '')
    CONFIGURE_OPTIONS.extend(shlex.split(options))
    for i in CONFIGURE_OPTIONS:
        if i.startswith('--with-mpi-dir='):
            raise RuntimeError("Do not use --with-mpi-dir, use the environmental variables MPICC, MPICXX, MPIFORT")
        if i.startswith('--with-cc='):
            raise RuntimeError("Do not use --with-cc, use the environmental variable MPICC")
        if i.startswith('--with-cxx=') and i != "--with-cxx=0":
            raise RuntimeError("Do not use --with-cxx, use the environmental variable MPICXX")
        if i.startswith('--with-fc=') and i != "--with-fc=0":
            raise RuntimeError("Do not use --with-fc, use the environmental variable MPIFORT")



def config(prefix, dry_run=False):
    log.info('PETSc: configure')
    options = [
        '--prefix=' + prefix,
        'PETSC_ARCH='+os.environ['PETSC_ARCH'],
        '--with-shared-libraries=1',
        '--with-c2html=0', # not needed
        ]
    if '--with-fc=0' in CONFIGURE_OPTIONS:
        options.append('--with-sowing=0')
    if '--with-debugging=1' not in CONFIGURE_OPTIONS:
        options.append('--with-debugging=0')
    if '--with-mpi=0' not in CONFIGURE_OPTIONS:
        mpicc = os.environ.get('MPICC') or shutil.which('mpicc')
        mpicxx = os.environ.get('MPICXX') or shutil.which('mpicxx')
        mpifort = (
            os.environ.get('MPIFORT')
            or os.environ.get('MPIF90')
            or shutil.which('mpifort')
            or shutil.which('mpif90')
        )
        if mpicc:
            options.append('--with-cc='+mpicc)
            if '--with-cxx=0' not in CONFIGURE_OPTIONS:
                if mpicxx:
                    options.append('--with-cxx='+mpicxx)
                else:
                    options.append('--with-cxx=0')
            if '--with-fc=0' not in CONFIGURE_OPTIONS:
                if mpifort:
                    options.append('--with-fc='+mpifort)
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
    if dry_run:
        return
    use_config_py = False
    if use_config_py:
        import configure
        configure.petsc_configure(options)
        import logger
        logger.Logger.defaultLog = None
    else:
        python = sys.executable
        command = [python, './configure'] + options
        status = os.system(" ".join(command))
        if status != 0:
            raise RuntimeError(status)

    # Fix PETSc configuration
    using_build_backend = any(
        os.environ.get(prefix + '_BUILD_BACKEND')
        for prefix in ('_PYPROJECT_HOOKS', 'PEP517')
    )
    if using_build_backend:
        pdir = os.environ['PETSC_DIR']
        parch = os.environ['PETSC_ARCH']
        include = os.path.join(pdir, parch, 'include')
        for filename in (
            'petscconf.h',
            'petscconfiginfo.h',
            'petscmachineinfo.h',
        ):
            filename = os.path.join(include, filename)
            with open(filename, 'r') as old_fh:
                contents = old_fh.read()
            contents = contents.replace(prefix, '${PETSC_DIR}')
            contents = re.sub(
                r'^(#define PETSC_PYTHON_EXE) "(.*)"$',
                r'\1 "python%d"' % sys.version_info[0],
                contents, flags=re.MULTILINE,
            )
            with open(filename, 'w') as new_fh:
                new_fh.write(contents)


def build(dry_run=False):
    log.info('PETSc: build')
    # Run PETSc build
    if dry_run:
        return
    use_builder_py = False
    if use_builder_py:
        import builder
        builder.PETScMaker().run()
        import logger
        logger.Logger.defaultLog = None
    else:
        make = shutil.which('make')
        command = [make, 'all']
        status = os.system(" ".join(command))
        if status != 0:
            raise RuntimeError(status)


def install(dry_run=False):
    log.info('PETSc: install')
    # Run PETSc installer
    if dry_run:
        return
    use_install_py = False
    if use_install_py:
        import install
        install.Installer().run()
        import logger
        logger.Logger.defaultLog = None
    else:
        make = shutil.which('make')
        command = [make, 'install']
        status = os.system(" ".join(command))
        if status != 0:
            raise RuntimeError(status)


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

    def finalize_options(self):
        _install.finalize_options(self)
        self.install_lib = self.install_platlib
        self.install_libbase = self.install_lib
        self.old_and_unmanageable = True

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


class cmd_bdist_wheel(_bdist_wheel):

    def finalize_options(self):
        super().finalize_options()
        self.root_is_pure = False
        self.build_number = None
        # self.keep_temp = True

    def get_tag(self):
        plat_tag = super().get_tag()[-1]
        return (self.python_tag, "none", plat_tag)


def version():
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
    if release:
        v = "%d.%d.%d" % (major, minor, micro)
    else:
        v = "%d.%d.0.dev%d" % (major, minor+1, 0)
    return v


def tarball():
    VERSION = version()
    if '.dev' in VERSION:
        return None
    return ('https://web.cels.anl.gov/projects/petsc/download/release-snapshots/'
            'petsc-%s.tar.gz#egg=petsc-%s' % (VERSION, VERSION))


description = __doc__.split('\n')[1:-1]
del description[1:3]

classifiers = """
Development Status :: 5 - Production/Stable
Intended Audience :: Developers
Intended Audience :: Science/Research
Operating System :: POSIX
Programming Language :: C
Programming Language :: C++
Programming Language :: Fortran
Programming Language :: Python
Programming Language :: Python :: 3
Topic :: Scientific/Engineering
Topic :: Software Development :: Libraries
"""

bootstrap()
setup(
    name='petsc',
    version=version(),
    description=description.pop(0),
    long_description='\n'.join(description),
    long_description_content_type='text/x-rst',
    classifiers=classifiers.split('\n')[1:-1],
    keywords = ['PETSc', 'MPI'],
    platforms=['POSIX'],
    license='BSD-2-Clause',

    url='https://petsc.org/',
    download_url=tarball(),

    author='PETSc Team',
    author_email='petsc-maint@mcs.anl.gov',
    maintainer='Lisandro Dalcin',
    maintainer_email='dalcinl@gmail.com',

    packages=['petsc'],
    package_dir= {'petsc': 'config/pypi'},
    cmdclass={
        'install': cmd_install,
        'bdist_wheel': cmd_bdist_wheel,
    },
    **metadata
)
