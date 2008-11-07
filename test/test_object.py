from petsc4py import PETSc
import unittest

# --------------------------------------------------------------------

class TestObjectBase(object):

    CLASS, FACTORY = None, None
    TARGS, KARGS = (), {}
    BUILD = None
    def setUp(self):
        self.obj = self.CLASS()
        getattr(self.obj,self.FACTORY)(*self.TARGS, **self.KARGS)
        if not self.obj: self.obj.create()

    def tearDown(self):
        self.obj = None

    def testClass(self):
        self.assertTrue(isinstance(self.obj, self.CLASS))
        self.assertTrue(type(self.obj) is self.CLASS)

    def testNonZero(self):
        self.assertTrue(bool(self.obj))

    def testDestroy(self):
        self.assertTrue(bool(self.obj))
        self.obj.destroy()
        self.assertFalse(bool(self.obj))
        ## self.assertRaises(PETSc.Error, self.obj.destroy)
        ## self.assertTrue(self.obj.this is this)

    def testOptions(self):
        self.assertFalse(self.obj.getOptionsPrefix())
        prefix1 = 'my_'
        self.obj.setOptionsPrefix(prefix1)
        self.assertEqual(self.obj.getOptionsPrefix(), prefix1)
        prefix2 = 'opt_'
        self.obj.setOptionsPrefix(prefix2)
        self.assertEqual(self.obj.getOptionsPrefix(), prefix2)
        ## self.obj.appendOptionsPrefix(prefix1)
        ## self.assertEqual(self.obj.getOptionsPrefix(),
        ##                  prefix2 + prefix1)
        ## self.obj.prependOptionsPrefix(prefix1)
        ## self.assertEqual(self.obj.getOptionsPrefix(),
        ##                  prefix1 + prefix2 + prefix1)
        self.obj.setFromOptions()

    def testName(self):
        oldname = self.obj.getName()
        newname = '%s-%s' %(oldname, oldname)
        self.obj.setName(newname)
        self.assertEqual(self.obj.getName(), newname)
        self.obj.setName(oldname)
        self.assertEqual(self.obj.getName(), oldname)

    def testComm(self):
        comm = self.obj.getComm()
        self.assertTrue(isinstance(comm, PETSc.Comm))
        self.assertTrue(comm in [PETSc.COMM_SELF, PETSc.COMM_WORLD])

    def testRefCount(self):
        self.assertEqual(self.obj.getRefCount(), 1)
        self.obj.incRef()
        self.assertEqual(self.obj.getRefCount(), 2)
        self.obj.incRef()
        self.assertEqual(self.obj.getRefCount(), 3)
        self.obj.decRef()
        self.assertEqual(self.obj.getRefCount(), 2)
        self.obj.decRef()
        self.assertEqual(self.obj.getRefCount(), 1)
        self.obj.decRef()
        self.assertFalse(bool(self.obj))

    def testComposeQuery(self):
        self.assertEqual(self.obj.getRefCount(), 1)
        self.obj.compose('myobj', self.obj)
        self.assertTrue(type(self.obj.query('myobj')) is self.CLASS)
        self.assertEqual(self.obj.query('myobj'), self.obj)
        self.assertEqual(self.obj.getRefCount(), 2)
        self.obj.compose('myobj', None)
        self.assertEqual(self.obj.getRefCount(), 1)
        self.assertEqual(self.obj.query('myobj'), None)

    def testProperties(self):
        self.assertEqual(self.obj.getCookie(),    self.obj.cookie)
        self.assertEqual(self.obj.getClassName(), self.obj.klass)
        self.assertEqual(self.obj.getType(),      self.obj.type)
        self.assertEqual(self.obj.getName(),      self.obj.name)
        self.assertEqual(self.obj.getComm(),      self.obj.comm)
        self.assertEqual(self.obj.getRefCount(),  self.obj.refcount)

# --------------------------------------------------------------------

class TestObjectRandom(TestObjectBase, unittest.TestCase):
    CLASS = PETSc.Random
    FACTORY = 'create'

class TestObjectViewer(TestObjectBase, unittest.TestCase):
    CLASS = PETSc.Viewer
    FACTORY = 'create'

class TestObjectIS(TestObjectBase, unittest.TestCase):
    CLASS  = PETSc.IS
    FACTORY = 'createGeneral'
    TARGS = ([],)

class TestObjectLGMap(TestObjectBase, unittest.TestCase):
    CLASS = PETSc.LGMap
    FACTORY = 'create'
    TARGS = ([],)

class TestObjectAO(TestObjectBase, unittest.TestCase):
    CLASS  = PETSc.AO
    FACTORY = 'createMapping'
    TARGS = ([], [])

class TestObjectDA(TestObjectBase, unittest.TestCase):
    CLASS  = PETSc.DA
    FACTORY = 'create'
    TARGS = ([3,3,3],)

class TestObjectVec(TestObjectBase, unittest.TestCase):
    CLASS   = PETSc.Vec
    FACTORY = 'createSeq'
    TARGS   = (0,)

class TestObjectScatter(TestObjectBase, unittest.TestCase):
    CLASS  = PETSc.Scatter
    FACTORY = 'create'
    def setUp(self):
        v1, v2 = PETSc.Vec().createSeq(0), PETSc.Vec().createSeq(0)
        i1, i2 = PETSc.IS().createGeneral([]), PETSc.IS().createGeneral([])
        self.obj = PETSc.Scatter().create(v1, i1, v2, i2)
        del v1, v2, i1, i2

class TestObjectMat(TestObjectBase, unittest.TestCase):
    CLASS  = PETSc.Mat
    FACTORY = 'createAIJ'
    TARGS = (0,)
    KARGS   = {'comm': PETSc.COMM_SELF}


class TestObjectNullSpace(TestObjectBase, unittest.TestCase):
    CLASS  = PETSc.NullSpace
    FACTORY = 'create'
    TARGS = (True, [])

class TestObjectKSP(TestObjectBase, unittest.TestCase):
    CLASS = PETSc.KSP
    FACTORY = 'create'

class TestObjectPC(TestObjectBase, unittest.TestCase):
    CLASS = PETSc.PC
    FACTORY = 'create'

class TestObjectSNES(TestObjectBase, unittest.TestCase):
    CLASS = PETSc.SNES
    FACTORY = 'create'

class TestObjectTS(TestObjectBase, unittest.TestCase):
    CLASS  = PETSc.TS
    FACTORY = 'create'
    def setUp(self):
        super(TestObjectTS, self).setUp()
        self.obj.setProblemType(PETSc.TS.ProblemType.NONLINEAR)
        self.obj.setType(PETSc.TS.Type.BEULER)

class TestObjectAOBasic(TestObjectBase, unittest.TestCase):
    CLASS  = PETSc.AO
    FACTORY = 'createBasic'
    TARGS = ([], [])

class TestObjectAOMapping(TestObjectBase, unittest.TestCase):
    CLASS  = PETSc.AO
    FACTORY = 'createMapping'
    TARGS = ([], [])

# --------------------------------------------------------------------

if __name__ == '__main__':
    unittest.main()
