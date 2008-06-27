# Author:  Lisandro Dalcin
# Contact: dalcinl@gmail.com
# Id: $Id$

"""
Low-level access to PETSc extension module.
"""

# --------------------------------------------------------------------

__date__     = '$Date$'
__version__  = '$Version$'
__revision__ = '$Revision$'

__docformat__ = 'reStructuredText'

# --------------------------------------------------------------------

def Import(arch=None):
    """
    Import the PETSc extension module for a given configuration name.
    """
    return ImportExt('petsc4py', 'PETSc',
                     __file__, arch,
                     'petsc.cfg')


def ImportExt(pkg, name, path, arch=None, conf=None):
    """
    Import helper for PETSc-based extension modules.
    """
    import sys, os, imp
    # test if extension module was imported before
    extname =  name
    modname = '%s.%s' % (pkg, extname)
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
    if arch is None:
        env_arch = os.environ.get('PETSC_ARCH')
        if not env_arch and conf:
            cfg = open(os.path.join(path, conf))
            try:
                lines = cfg.read().replace(' ', '').splitlines()
            finally:
                cfg.close()
            cfg = dict([line.split('=') for line in lines])
            cfg_arch = cfg['PETSC_ARCH'].split(os.path.pathsep)
            arch = [_a for _a in cfg_arch if _a][0]
        else:
            arch = env_arch
        mpath = os.path.join(path, arch)
        if not os.path.isdir(mpath):
            raise ImportError("invalid 'PETSC_ARCH': '%s'" % arch)
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
