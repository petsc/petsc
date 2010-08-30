#!/usr/bin/env python

"""
     Make PETSc appear to be a Python package; so it can be depended on by petsc4py
"""



def configure(prefix):
    import os,sys
    os.environ['PETSC_DIR']  = os.path.abspath(os.getcwd())
    os.environ['PETSC_ARCH'] = 'arch-python-test'
    os.chdir(os.environ['PETSC_DIR'])
    sys.path.insert(0,os.path.join(os.environ['PETSC_DIR'],'config'))
    import configure
    configure.petsc_configure(['--with-fc=0','--with-mpi=0','--with-shared'])

def build():
    import os,sys

    # work around bug in logger.Logger that when log file is closed Logger.defaultLog still points to something
    import logger
    logger.Logger.defaultLog = None
    import builder
    builder.PETScMaker().run()

    
if __name__ == '__main__':
    configure("dummy")
    build()

# -----------------------------------------------------------------------------
