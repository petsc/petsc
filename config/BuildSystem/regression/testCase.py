import unittest

class MPITest (unittest.TestCase):
  '''Initialize and finalize MPI for every test'''
  mpi = None

  def setUpMPI(baseClass):
    '''Initialize MPI'''
    if MPITest.mpi is None:
      import ASE.Args
      import ASE.Loader
      import MPIB.Base
      import atexit
      import sys

      ASE.Args.Args.set(sys.argv)
      MPITest.mpi = MPIB.Base.Base(ASE.Loader.Loader.createClass(baseClass))
      MPITest.mpi.Initialize()
      atexit.register(ASE.Loader.Loader.setLibraries, [])
      atexit.register(MPITest.mpi.Finalize)
    return
  setUpMPI = staticmethod(setUpMPI)

  def setUp(self, baseClass = 'MPIB.Default.Base'):
    '''Initialize MPI and set "comm" to MPI_COMM_WORLD'''
    MPITest.setUpMPI(baseClass)
    self.comm = MPITest.mpi.comm().WORLD()
    return

  def tearDown(self):
    '''We cannot finalize MPI here, since it can only be initialized once'''
    self.comm = None
    return

class PETScTest (MPITest):
  petsc = None

  def setUpPETSc(baseClass):
    '''Initialize PETSc'''
    if PETScTest.petsc is None:
      import ASE.Loader
      import PETSc.Base
      import atexit

      PETScTest.petsc = PETSc.Base.Base(ASE.Loader.Loader.createClass(baseClass))
      PETScTest.petsc.Initialize()
      atexit.register(PETScTest.petsc.Finalize)
    return
  setUpPETSc = staticmethod(setUpPETSc)

  def setUp(self, baseClass = 'PETSc.Base', mpiBaseClass = 'MPIB.Default.Base'):
    '''Initialize PETSc'''
    MPITest.setUp(self, mpiBaseClass)
    PETScTest.setUpPETSc(baseClass)
    return

  def tearDown(self):
    '''Cannot finalize PETSc since it can only be initialized once'''
    MPITest.tearDown(self)
    return
