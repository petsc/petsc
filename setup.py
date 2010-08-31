#!/usr/bin/env python

"""
     Make PETSc appear to be a Python package; so it can be depended on by petsc4py
"""

def setup():
    import os,sys
    os.environ['PETSC_DIR']  = os.path.abspath(os.getcwd())
    os.environ['PETSC_ARCH'] = 'arch-python-test'
    os.chdir(os.environ['PETSC_DIR'])
    sys.path.insert(0,os.path.join(os.environ['PETSC_DIR'],'config'))
    return

def configure(prefix):
    import configure
    configure.petsc_configure(['--with-fc=0','--with-mpi=0','--with-shared-libraries'])
    # force save of configure information, or build will fail to load it
    
    # work around bug in logger.Logger that when log file is closed Logger.defaultLog still points to something
    import logger
    logger.Logger.defaultLog = None
    return

def build():
    import builder
    builder.PETScMaker().run()
    return
    
if __name__ == '__main__':
    setup()
    configure("dummy")
    build()

# -----------------------------------------------------------------------------
