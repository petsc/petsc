import unittest

class MPITest (unittest.TestCase):
  '''Initialize and finalize MPI for every test'''
  mpi = None

  def setUpMPI():
    '''Initialize MPI'''
    if MPITest.mpi is None:
      import SIDL.Args
      import SIDL.Loader
      import MPIB.Base
      import atexit
      import sys

      SIDL.Args.set(sys.argv)
      MPITest.mpi = MPIB.Base.Base(SIDL.Loader.createClass('MPIB.Default.DefaultBase'))
      MPITest.mpi.Initialize()
      atexit.register(MPITest.mpi.Finalize)
    return
  setUpMPI = staticmethod(setUpMPI)

  def setUp(self):
    '''Initialize MPI and set "comm" to MPI_COMM_WORLD'''
    MPITest.setUpMPI()
    self.comm = MPITest.mpi.comm().WORLD()
    return

  def tearDown(self):
    '''We cannot finalize MPI here, since it can only be initialized once'''
    return

class PETScTest (MPITest):
  petsc = None

  def setUpPETSc():
    '''Initialize PETSc'''
    if PETScTest.petsc is None:
      import SIDL.Loader
      import SIDL.ProjectState
      import atexit

      PETScTest.petsc = SIDL.ProjectState.ProjectState(SIDL.Loader.createClass('PETSc.State'))
      PETScTest.petsc.Initialize()
      atexit.register(PETScTest.petsc.Finalize)
    return
  setUpPETSc = staticmethod(setUpPETSc)

  def setUp(self):
    '''Initialize PETSc'''
    MPITest.setUp(self)
    PETScTest.setUpPETSc()
    return

  def tearDown(self):
    '''Cannot finalize PETSc since it can only be initialized once'''
    return
