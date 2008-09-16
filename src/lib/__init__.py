# Author:  Lisandro Dalcin
# Contact: dalcinl@gmail.com
# Id: $Id$

# --------------------------------------------------------------------

"""
Extension modules for different PETSc configurations.

PETSc can be configured with different options (eg. debug/optimized,
single/double precisionm, C/C++ compilers, external packages). Each
configuration variant is associated to a name, frequently available as
an environmental variable named ``PETSC_ARCH``.

This package is a holds all the available variants of the PETSc
extension module built agaist specific PETSc configurations. It also
provides a convenience function using of the builtin ``imp`` module
for easily importing any of the available extension modules depending
on the value of a user-provided configuration name, the ``PETSC_ARCH``
environmental variable, or a configuration file.
"""

# --------------------------------------------------------------------

def ImportPETSc(arch=None):
    """
    Import the PETSc extension module for a given configuration name.
    """
    return Import('petsc4py', 'PETSc', __file__, arch,
                  'PETSC_ARCH', 'petsc.cfg')


def Import(pkg, name, path, arch=None,
           conf_var=None, conf_file=None):
    """
    Import helper for PETSc-based extension modules.
    """
    import sys, os, imp
    # extension and full dotted module name
    extname =  name
    modname = '%s.%s' % (pkg, extname)
    # test if extension module was already imported
    module  = sys.modules.get(modname)
    fname = getattr(module, '__file__', '')
    shext = imp.get_suffixes()[0][0]
    if os.path.splitext(fname)[-1] == shext:
        # if 'arch' is None, do nothing; otherwise this
        # call may be invalid if extension module for
        # other 'arch' has been already imported.
        if arch is not None and arch != module.__arch__:
            raise ImportError("%s already imported" % module)
        return module
    # determine base path for import
    if os.path.isfile(path):
        path = os.path.dirname(path)
    elif not os.path.isdir(path):
        raise ImportError("invalid path '%s':" % path)
    # determine PETSC_ARCH value
    PETSC_ARCH = conf_var or 'PETSC_ARCH'
    if arch is None:
        arch_value = os.environ.get(PETSC_ARCH, '').strip()
        if not arch_value and conf_file:
            cfg = open(os.path.join(path, conf_file))
            try:
                lines = cfg.read().replace(' ', '').splitlines()
            finally:
                cfg.close()
            cfg = dict([line.split('=') for line in lines])
            arch_value = cfg[PETSC_ARCH]
        arch_list = arch_value.split(os.path.pathsep)
        arch = [a for a in arch_list if a][0]
        mpath = os.path.join(path, arch)
        if not os.path.isdir(mpath):
            raise ImportError("invalid '%s': '%s'" % (PETSC_ARCH, arch))
    else:
        if not isinstance(arch, str):
            raise TypeError("'arch' argument must be string")
        mpath = os.path.join(path, arch)
        if not os.path.isdir(mpath):
            raise ImportError("invalid 'arch' value: '%s'" % arch)
    # import extension module from 'path/arch' directory
    extpath = [os.path.join(path, arch),]
    fo, fn, stuff = imp.find_module(extname, extpath)
    module = imp.load_module(modname, fo, fn, stuff)
    module.__arch__ = arch # save arch value
    setattr(sys.modules[pkg], extname, module)
    return module

# --------------------------------------------------------------------
