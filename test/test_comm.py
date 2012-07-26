from petsc4py import PETSc
import unittest

# --------------------------------------------------------------------

class TestComm(unittest.TestCase):

    def testInit(self):
        comm_null1  = PETSc.Comm()
        comm_null2  = PETSc.Comm(PETSc.COMM_NULL)
        comm_world = PETSc.Comm(PETSc.COMM_WORLD)
        comm_self  = PETSc.Comm(PETSc.COMM_SELF)
        self.assertEqual(comm_null1, PETSc.COMM_NULL)
        self.assertEqual(comm_null2, PETSc.COMM_NULL)
        self.assertEqual(comm_world, PETSc.COMM_WORLD)
        self.assertEqual(comm_self,  PETSc.COMM_SELF)

    def testDupDestr(self):
        self.assertRaises(ValueError, PETSc.COMM_NULL.duplicate)
        comm = PETSc.COMM_SELF.duplicate()
        comm.destroy()
        self.assertEqual(comm, PETSc.COMM_NULL)
        del comm
        comm = PETSc.COMM_WORLD.duplicate()
        comm.destroy()
        self.assertEqual(comm, PETSc.COMM_NULL)
        del comm

    def testBarrier(self):
        self.assertRaises(ValueError, PETSc.COMM_NULL.barrier)
        PETSc.COMM_SELF.barrier()
        PETSc.COMM_WORLD.barrier()

    def testSize(self):
        self.assertRaises(ValueError, PETSc.COMM_NULL.getSize)
        self.assertTrue(PETSc.COMM_WORLD.getSize() >= 1)
        self.assertEqual(PETSc.COMM_SELF.getSize(), 1)

    def testRank(self):
        self.assertRaises(ValueError, PETSc.COMM_NULL.getRank)
        self.assertEqual(PETSc.COMM_SELF.getRank(), 0)
        self.assertTrue(PETSc.COMM_WORLD.getRank() >= 0)

    def testProperties(self):
        self.assertEqual(PETSc.COMM_SELF.getSize(),
                         PETSc.COMM_SELF.size)
        self.assertEqual(PETSc.COMM_SELF.getRank(),
                         PETSc.COMM_SELF.rank)
        self.assertEqual(PETSc.COMM_WORLD.getSize(),
                         PETSc.COMM_WORLD.size)
        self.assertEqual(PETSc.COMM_WORLD.getRank(),
                         PETSc.COMM_WORLD.rank)

    def testCompatMPI4PY(self):
        try:
            from mpi4py import MPI
        except ImportError:
            return
        # mpi4py -> petsc4py
        cn = PETSc.Comm(MPI.COMM_NULL)
        cs = PETSc.Comm(MPI.COMM_SELF)
        cw = PETSc.Comm(MPI.COMM_WORLD)
        self.assertEqual(cn, PETSc.COMM_NULL)
        self.assertEqual(cs, PETSc.COMM_SELF)
        self.assertEqual(cw, PETSc.COMM_WORLD)
        # petsc4py - > mpi4py
        cn = PETSc.COMM_NULL.tompi4py()
        self.assertTrue(isinstance(cn, MPI.Comm))
        self.assertFalse(cn)
        cs = PETSc.COMM_SELF.tompi4py()
        self.assertTrue(isinstance(cs, MPI.Intracomm))
        self.assertEqual(cs.Get_size(), 1)
        self.assertEqual(cs.Get_rank(), 0)
        cw = PETSc.COMM_WORLD.tompi4py()
        self.assertTrue(isinstance(cw, MPI.Intracomm))
        self.assertEqual(cw.Get_size(), PETSc.COMM_WORLD.getSize())
        self.assertEqual(cw.Get_rank(), PETSc.COMM_WORLD.getRank())
        

# --------------------------------------------------------------------

if __name__ == '__main__':
    unittest.main()
