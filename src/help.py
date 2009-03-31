# Author:  Lisandro Dalcin
# Contact: dalcinl@gmail.com

"""
Command line access to the PETSc Options Database.

This module intends to provide command line access to the PETSc
Options Database. It outputs a listing of the many PETSc options
indicating option names, default values and descriptions. If you have
Python 2.4 and above, then you can issue at the command line::

  $ python -m petsc4py.help [vec|mat|ksp|pc|snes|ts] [<petsc-option-list>]

"""

def help(args=None):
    import sys
    # program name
    try:
        prog = sys.argv[0]
    except Exception:
        prog = getattr(sys, 'executable', 'python')
    if args is None:
        args = sys.argv[1:]
    elif isinstance(args, str):
        args = args.split()
    else:
        args = [str(a) for a in args]
    import petsc4py
    petsc4py.init([prog, '-help'] + args)
    from petsc4py import PETSc
    COMM = PETSc.COMM_SELF
    if 'vec' in args:
        vec = PETSc.Vec().create(comm=COMM)
        vec.setSizes(0)
        vec.setFromOptions()
        del vec
    if 'mat' in args:
        mat = PETSc.Mat().create(comm=COMM)
        mat.setSizes([0, 0])
        mat.setFromOptions()
        del mat
    if 'ksp' in args:
        ksp = PETSc.KSP().create(comm=COMM)
        ksp.setFromOptions()
        del ksp
    if 'pc' in args:
        pc = PETSc.PC().create(comm=COMM)
        pc.setFromOptions()
        del pc
    if 'snes' in args:
        snes = PETSc.SNES().create(comm=COMM)
        snes.setFromOptions()
        del snes
    if 'ts' in args:
        ts = PETSc.TS().create(comm=COMM)
        ts.setFromOptions()
        del ts

if __name__ == '__main__':
    help()
