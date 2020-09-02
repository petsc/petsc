# Author:  Lisandro Dalcin
# Contact: dalcinl@gmail.com

"""
Command line access to the PETSc Options Database.

This module provides command line access to PETSc Options
Database. It outputs a listing of the many PETSc options
indicating option names, default values and descriptions.
Usage::

  $ python -m petsc4py [vec|mat|pc|ksp|snes|ts|tao] [<petsc-option-list>]

"""

def help(args=None):
    import sys, shlex
    # program name
    try:
        prog = sys.argv[0]
    except Exception:
        prog = getattr(sys, 'executable', 'python')
    # arguments
    if args is None:
        args = sys.argv[1:]
    elif isinstance(args, str):
        args = shlex.split(args)
    else:
        args = [str(a) for a in args]
    # import and initialize
    import petsc4py
    petsc4py.init([prog, '-help'] + args)
    from petsc4py import PETSc
    # help dispatcher
    COMM = PETSc.COMM_SELF
    if 'vec' in args:
        vec = PETSc.Vec().create(comm=COMM)
        vec.setSizes(0)
        vec.setFromOptions()
        vec.destroy()
    if 'mat' in args:
        mat = PETSc.Mat().create(comm=COMM)
        mat.setSizes([0, 0])
        mat.setFromOptions()
        mat.destroy()
    if 'pc' in args:
        pc = PETSc.PC().create(comm=COMM)
        pc.setFromOptions()
        pc.destroy()
    if 'ksp' in args:
        ksp = PETSc.KSP().create(comm=COMM)
        ksp.setFromOptions()
        ksp.destroy()
    if 'snes' in args:
        snes = PETSc.SNES().create(comm=COMM)
        snes.setFromOptions()
        snes.destroy()
    if 'ts' in args:
        ts = PETSc.TS().create(comm=COMM)
        ts.setFromOptions()
        ts.destroy()
    if 'tao' in args:
        tao = PETSc.TAO().create(comm=COMM)
        tao.setFromOptions()
        tao.destroy()
    if 'dmda' in args:
        dmda = PETSc.DMDA().create(comm=COMM)
        dmda.setFromOptions()
        dmda.destroy()
    if 'dmplex' in args:
        dmplex = PETSc.DMPlex().create(comm=COMM)
        dmplex.setFromOptions()
        dmplex.destroy()

if __name__ == '__main__':
    help()
