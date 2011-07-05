# Author:  Lisandro Dalcin
# Contact: dalcinl@gmail.com

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
    path, arch = getPathArchPETSc(arch)
    return Import('petsc4py', 'PETSc', path, arch)

def getPathArchPETSc(arch=None):
    """
    Undocumented.
    """
    import sys, os
    path = os.path.dirname(__file__)
    rcvar, rcfile  =  'PETSC_ARCH', 'petsc.cfg'
    path, arch = getPathArch(path, arch, rcvar, rcfile)
    return (path, arch)

# --------------------------------------------------------------------

def Import(pkg, name, path, arch):
    """
    Import helper for PETSc-based extension modules.
    """
    import sys, os, imp
    # full dotted module name
    fullname = '%s.%s' % (pkg, name)
    # test if extension module was already imported
    module  = sys.modules.get(fullname)
    fname = getattr(module, '__file__', '')
    shext = imp.get_suffixes()[0][0]
    if os.path.splitext(fname)[-1] == shext:
        # if 'arch' is None, do nothing; otherwise this
        # call may be invalid if extension module for
        # other 'arch' has been already imported.
        if arch is not None and arch != module.__arch__:
            raise ImportError("%s already imported" % module)
        return module
    # import extension module from 'path/arch' directory
    pathlist = [os.path.join(path, arch)]
    fo, fn, stuff = imp.find_module(name, pathlist)
    module = imp.load_module(fullname, fo, fn, stuff)
    module.__arch__ = arch # save arch value
    setattr(sys.modules[pkg], name, module)
    return module

def getPathArch(path, arch, rcvar='PETSC_ARCH', rcfile='petsc.cfg'):
    """
    Undocumented.
    """
    import os, warnings
    # path
    if not path:
        path = '.'
    elif os.path.isfile(path):
        path = os.path.dirname(path)
    elif not os.path.isdir(path):
        raise ValueError("invalid path: '%s'" % path)
    # arch
    if arch is not None:
        if not isinstance(arch, str):
            raise TypeError( "arch argument must be string")
        if not os.path.isdir(os.path.join(path, arch)):
            raise TypeError("invalid arch value: '%s'" % arch)
        return (path, arch)
    # helper function
    def arch_list(arch):
        arch = arch.strip().split(os.path.pathsep)
        arch = [a.strip() for a in arch if a]
        arch = [a for a in arch if a]
        return arch
    # try to get arch from the environment
    arch_env = arch_list(os.environ.get(rcvar, ''))
    for arch in arch_env:
        if os.path.isdir(os.path.join(path, arch)):
            return (path, arch)
    # configuration file
    if not os.path.isfile(rcfile):
        rcfile = os.path.join(path, rcfile)
        if not os.path.isfile(rcfile):
            # now point to continue
            return (path, '')
    # helper function
    def parse_rc(rcfile):
        fh = open(rcfile)
        try: rcdata = fh.read()
        finally: fh.close()
        lines = [ln.strip() for ln in rcdata.splitlines()]
        lines = [ln for ln in lines if not ln.startswith('#')]
        entries = [ln.split('=') for ln in lines if ln]
        entries = [(k.strip(), v.strip()) for k, v in entries]
        return dict(entries)
    # try to get arch from data in config file
    configrc = parse_rc(rcfile)
    arch_cfg = arch_list(configrc.get(rcvar, ''))
    for arch in arch_cfg:
        if os.path.isdir(os.path.join(path, arch)):
            if arch_env:
                warnings.warn(
                    "ignored arch: '%s', using: '%s'" % \
                    (os.path.pathsep.join(arch_env), arch))
            return (path, arch)
    # nothing good found
    return (path, '')

def getInitArgs(args):
    """
    Undocumented.
    """
    import sys, shlex
    if args is None:
        args = []
    elif isinstance(args, str):
        args = shlex.split(args)
    else:
        args = [str(a) for a in args]
        args = [a for a in args if a]
    if args and args[0].startswith('-'):
        sys_argv = getattr(sys, 'argv', None)
        sys_exec = getattr(sys, 'executable', 'python')
        if (sys_argv and
            sys_argv[0] and
            sys_argv[0] != '-c'):
            prog_name = sys_argv[0]
        else:
            prog_name = sys_exec
        args.insert(0, prog_name)
    return args

# --------------------------------------------------------------------
