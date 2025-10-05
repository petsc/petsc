#!/usr/bin/env python3
# Author:  Lisandro Dalcin
# Contact: dalcinl@gmail.com

import re
import os
import sys
import warnings

try:
    import setuptools
except ImportError:
    setuptools = None

topdir = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(topdir, 'conf'))

pyver = sys.version_info[:2]
if pyver < (3, 6):
    raise RuntimeError('Python version 3.6 or higher is required')

PNAME = 'PETSc'
EMAIL = 'petsc-maint@mcs.anl.gov'
PLIST = [PNAME]

# --------------------------------------------------------------------
# Metadata
# --------------------------------------------------------------------

py_limited_api = (3, 10)


def F(string):
    return string.format(
        Name=PNAME,
        name=PNAME.lower(),
        pyname=PNAME.lower() + '4py',
    )


def get_name():
    return F('{pyname}')


def get_version():
    try:
        return get_version.result
    except AttributeError:
        pass
    pkg_init_py = os.path.join(F('{pyname}'), '__init__.py')
    with open(os.path.join(topdir, 'src', pkg_init_py)) as f:
        m = re.search(r"__version__\s*=\s*'(.*)'", f.read())
    version = m.groups()[0]
    get_version.result = version
    return version


def description():
    return F('{Name} for Python')


def long_description():
    with open(os.path.join(topdir, 'DESCRIPTION.rst')) as f:
        return f.read()


url = F('https://gitlab.com/{name}/{name}')
pypiroot = F('https://pypi.io/packages/source')
pypislug = F('{pyname}')[0] + F('/{pyname}')
tarball = F('{pyname}-%s.tar.gz' % get_version())
download = '/'.join([pypiroot, pypislug, tarball])

classifiers = """
Operating System :: POSIX
Intended Audience :: Developers
Intended Audience :: Science/Research
Programming Language :: C
Programming Language :: C++
Programming Language :: Cython
Programming Language :: Python
Programming Language :: Python :: 3
Programming Language :: Python :: Implementation :: CPython
Topic :: Scientific/Engineering
Topic :: Software Development :: Libraries :: Python Modules
Development Status :: 5 - Production/Stable
""".strip().split('\n')

keywords = """
scientific computing
parallel computing
MPI
""".strip().split('\n')

platforms = """
POSIX
Linux
macOS
FreeBSD
""".strip().split('\n')

metadata = {
    'name': get_name(),
    'version': get_version(),
    'description': description(),
    'long_description': long_description(),
    'url': url,
    'download_url': download,
    'classifiers': classifiers,
    'keywords': keywords + PLIST,
    'license': 'BSD-2-Clause',
    'platforms': platforms,
    'author': 'Lisandro Dalcin',
    'author_email': 'dalcinl@gmail.com',
    'maintainer': F('{Name} Team'),
    'maintainer_email': EMAIL,
}
metadata.update(
    {
        'requires': ['numpy'],
    }
)

metadata_extra = {
    'long_description_content_type': 'text/x-rst',
}

def get_build_pysabi():
    abi = os.environ.get("PETSC4PY_BUILD_PYSABI", "").lower()
    if abi and sys.implementation.name == "cpython":
        if abi in {"false", "no", "off", "n", "0"}:
            return None
        if abi in {"true", "yes", "on", "y", "1"} | {"abi3"}:
            return py_limited_api
        if abi.startswith("cp"):
            abi = abi[2:]
        if "." in abi:
            x, y = abi.split(".")
        else:
            x, y = abi[0], abi[1:]
        return (int(x), int(y))
    return None

# --------------------------------------------------------------------
# Extension modules
# --------------------------------------------------------------------


def sources():
    src = {
        'source': F('{pyname}/{Name}.pyx'),
        'depends': [
            F('{pyname}/*.pyx'),
            F('{pyname}/*.pxd'),
            F('{pyname}/{Name}/*.pyx'),
            F('{pyname}/{Name}/*.pxd'),
            F('{pyname}/{Name}/*.pxi'),
        ],
        'workdir': 'src',
    }
    return [src]


def extensions():
    from os import walk
    from glob import glob
    from os.path import join

    #
    depends = []
    glob_join = lambda *args: glob(join(*args))
    for pth, _, _ in walk('src'):
        depends += glob_join(pth, '*.h')
        depends += glob_join(pth, '*.c')
    for pkg in map(str.lower, reversed(PLIST)):
        if (pkg.upper() + '_DIR') in os.environ:
            pd = os.environ[pkg.upper() + '_DIR']
            pa = os.environ.get('PETSC_ARCH', '')
            depends += glob_join(pd, 'include', '*.h')
            depends += glob_join(pd, 'include', pkg, 'private', '*.h')
            depends += glob_join(pd, pa, 'include', '%sconf.h' % pkg)
    #
    include_dirs = []
    numpy_include = os.environ.get('NUMPY_INCLUDE')
    if numpy_include is not None:
        numpy_includes = [numpy_include]
    else:
        try:
            import numpy

            numpy_includes = [numpy.get_include()]
        except ImportError:
            numpy_includes = []
    include_dirs.extend(numpy_includes)
    if F('{pyname}') != 'petsc4py':
        try:
            import petsc4py

            petsc4py_includes = [petsc4py.get_include()]
        except ImportError:
            petsc4py_includes = []
        include_dirs.extend(petsc4py_includes)
    #
    ext = {
        'name': F('{pyname}.lib.{Name}'),
        'sources': [F('src/{pyname}/{Name}.c')],
        'depends': depends,
        'include_dirs': [
            'src',
            F('src/{pyname}/include'),
        ]
        + include_dirs,
        'define_macros': [
            ('MPICH_SKIP_MPICXX', 1),
            ('OMPI_SKIP_MPICXX', 1),
            ('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION'),
        ],
    }
    return [ext]


# --------------------------------------------------------------------
# Setup
# --------------------------------------------------------------------


def get_release():
    suffix = os.path.join('src', 'binding', F('{pyname}'))
    if not topdir.endswith(os.path.join(os.path.sep, suffix)):
        return True
    release = 1
    rootdir = os.path.abspath(os.path.join(topdir, *[os.path.pardir] * 3))
    version_h = os.path.join(rootdir, 'include', F('{name}version.h'))
    release_macro = '%s_VERSION_RELEASE' % F('{name}').upper()
    version_re = re.compile(r'#define\s+%s\s+([-]*\d+)' % release_macro)
    if os.path.exists(version_h) and os.path.isfile(version_h):
        with open(version_h, 'r') as f:
            release = int(version_re.search(f.read()).groups()[0])
    return bool(release)


def requires(pkgname, major, minor, release=True):
    minor = minor + int(not release)
    devel = '' if release else '.dev0'
    vmin = f'{major}.{minor}{devel}'
    vmax = f'{major}.{minor+1}'
    return f'{pkgname}>={vmin},<{vmax}'


def run_setup():
    is_sdist = 'sdist' in sys.argv
    setup_args = metadata.copy()
    vstr = setup_args['version'].split('.')[:2]
    x, y = tuple(map(int, vstr))
    release = get_release()
    if not release:
        setup_args['version'] = '%d.%d.0.dev0' % (x, y + 1)
    if setuptools:
        warnings.filterwarnings(
            'ignore', message=r'.*fetch_build_eggs', module='setuptools'
        )
        setup_args['zip_safe'] = False
        numpy_pin = 'numpy'
        if not is_sdist:
            try:
                import numpy

                major = int(numpy.__version__.partition('.')[0])
                numpy_pin = 'numpy>=1.19' if major >= 2 else 'numpy<2'
            except ImportError:
                pass
        setup_args['setup_requires'] = ['numpy']
        setup_args['install_requires'] = [numpy_pin]
        for pkg in map(str.lower, PLIST):
            PKG_DIR = os.environ.get(pkg.upper() + '_DIR')
            if not (PKG_DIR and os.path.isdir(PKG_DIR)):
                package = requires(pkg, x, y, release)
                setup_args['setup_requires'] += [package]
                setup_args['install_requires'] += [package]
        if F('{pyname}') != 'petsc4py':
            package = requires('petsc4py', x, y, release)
            setup_args['setup_requires'] += [package]
            setup_args['install_requires'] += [package]
        setup_args.update(metadata_extra)
    #
    conf = __import__(F('conf{name}'))
    cython_sources = [src for src in sources()]  # noqa: C416
    ext_modules = [conf.Extension(**ext) for ext in extensions()]
    #
    sabi = get_build_pysabi()
    if sabi and setuptools:
        api_tag = "cp{}{}".format(*sabi)
        options = {"bdist_wheel": {"py_limited_api": api_tag}}
        setup_args["options"] = options
        api_ver = "0x{:02X}{:02X}0000".format(*sabi)
        defines = [("Py_LIMITED_API", api_ver)]
        for ext in ext_modules:
            ext.define_macros.extend(defines)
            ext.py_limited_api = True
    #
    conf.setup(
        packages=[
            F('{pyname}'),
            F('{pyname}.lib'),
            F('{pyname}.lib._pytypes'),
            F('{pyname}.lib._pytypes.viewer'),
        ],
        package_dir={'': 'src'},
        package_data={
            F('{pyname}'): [
                F('{Name}.pxd'),
                F('{Name}*.h'),
                F('include/{pyname}/*.h'),
                F('include/{pyname}/*.i'),
                'py.typed',
                '*.pyi',
                '*/*.pyi',
            ],
            F('{pyname}.lib'): [
                F('{name}.cfg'),
            ],
        },
        cython_sources=cython_sources,
        ext_modules=ext_modules,
        **setup_args,
    )


# --------------------------------------------------------------------


def main():
    run_setup()


if __name__ == '__main__':
    main()

# --------------------------------------------------------------------
