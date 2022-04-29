import unittest

from petsc4py import PETSc

# --------------------------------------------------------------------

class BaseTestObject(object):

    CLASS, FACTORY = None, None
    TARGS, KARGS = (), {}
    BUILD = None
    def setUp(self):
        self.obj = self.CLASS()
        getattr(self.obj,self.FACTORY)(*self.TARGS, **self.KARGS)
        if not self.obj: self.obj.create()

    def tearDown(self):
        self.obj = None

    def testTypeRegistry(self):
        type_reg = PETSc.__type_registry__
        classid = self.obj.getClassId()
        typeobj = self.CLASS
        if isinstance(self.obj, PETSc.DMDA):
            typeobj = PETSc.DM
        self.assertTrue(type_reg[classid] is typeobj )

    def testLogClass(self):
        name = self.CLASS.__name__
        if name == 'DMDA': name = 'DM'
        logcls = PETSc.Log.Class(name)
        classid = self.obj.getClassId()
        self.assertEqual(logcls.id, classid)

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

    def testHandle(self):
        self.assertTrue(self.obj.handle)
        self.assertTrue(self.obj.fortran)
        h, f = self.obj.handle, self.obj.fortran
        if (h>0 and f>0) or (h<0 and f<0):
            self.assertEqual(h, f)
        self.obj.destroy()
        self.assertFalse(self.obj.handle)
        self.assertFalse(self.obj.fortran)

    def testComposeQuery(self):
        import copy
        try:
            myobj = copy.deepcopy(self.obj)
        except NotImplementedError:
            return
        self.assertEqual(myobj.getRefCount(), 1)
        self.obj.compose('myobj', myobj)
        self.assertTrue(type(self.obj.query('myobj')) is self.CLASS)
        self.assertEqual(self.obj.query('myobj'), myobj)
        self.assertEqual(myobj.getRefCount(), 2)
        self.obj.compose('myobj', None)
        self.assertEqual(myobj.getRefCount(), 1)
        self.assertEqual(self.obj.query('myobj'), None)
        myobj.destroy()

    def testProperties(self):
        self.assertEqual(self.obj.getClassId(),   self.obj.classid)
        self.assertEqual(self.obj.getClassName(), self.obj.klass)
        self.assertEqual(self.obj.getType(),      self.obj.type)
        self.assertEqual(self.obj.getName(),      self.obj.name)
        self.assertEqual(self.obj.getComm(),      self.obj.comm)
        self.assertEqual(self.obj.getRefCount(),  self.obj.refcount)

    def testShallowCopy(self):
        import copy
        rc = self.obj.getRefCount()
        obj = copy.copy(self.obj)
        self.assertTrue(obj is not self.obj)
        self.assertTrue(obj == self.obj)
        self.assertTrue(type(obj) is type(self.obj))
        self.assertEqual(obj.getRefCount(), rc+1)
        del obj
        self.assertEqual(self.obj.getRefCount(), rc)

    def testDeepCopy(self):
        import copy
        rc = self.obj.getRefCount()
        try:
            obj = copy.deepcopy(self.obj)
        except NotImplementedError:
            return
        self.assertTrue(obj is not self.obj)
        self.assertTrue(obj != self.obj)
        self.assertTrue(type(obj) is type(self.obj))
        self.assertEqual(self.obj.getRefCount(), rc)
        self.assertEqual(obj.getRefCount(), 1)
        del obj

    def testStateInspection(self):
        state = self.obj.stateGet()
        self.obj.stateIncrease()
        self.assertTrue(state < self.obj.stateGet())
        self.obj.stateSet(0)
        self.assertTrue(self.obj.stateGet() == 0)
        self.obj.stateSet(state)
        self.assertTrue(self.obj.stateGet() == state)


# --------------------------------------------------------------------

class TestObjectRandom(BaseTestObject, unittest.TestCase):
    CLASS = PETSc.Random
    FACTORY = 'create'

class TestObjectViewer(BaseTestObject, unittest.TestCase):
    CLASS = PETSc.Viewer
    FACTORY = 'create'

class TestObjectIS(BaseTestObject, unittest.TestCase):
    CLASS  = PETSc.IS
    FACTORY = 'createGeneral'
    TARGS = ([],)

class TestObjectLGMap(BaseTestObject, unittest.TestCase):
    CLASS = PETSc.LGMap
    FACTORY = 'create'
    TARGS = ([],)

class TestObjectAO(BaseTestObject, unittest.TestCase):
    CLASS  = PETSc.AO
    FACTORY = 'createMapping'
    TARGS = ([], [])

class TestObjectDMDA(BaseTestObject, unittest.TestCase):
    CLASS  = PETSc.DMDA
    FACTORY = 'create'
    TARGS = ([3,3,3],)

class TestObjectDS(BaseTestObject, unittest.TestCase):
    CLASS  = PETSc.DS
    FACTORY = 'create'

class TestObjectVec(BaseTestObject, unittest.TestCase):
    CLASS   = PETSc.Vec
    FACTORY = 'createSeq'
    TARGS   = (0,)

    def setUp(self):
        BaseTestObject.setUp(self)
        self.obj.assemble()

class TestObjectMat(BaseTestObject, unittest.TestCase):
    CLASS  = PETSc.Mat
    FACTORY = 'createAIJ'
    TARGS = (0,)
    KARGS   = {'nnz':0, 'comm': PETSc.COMM_SELF}

    def setUp(self):
        BaseTestObject.setUp(self)
        self.obj.assemble()

class TestObjectMatPartitioning(BaseTestObject, unittest.TestCase):
    CLASS  = PETSc.MatPartitioning
    FACTORY = 'create'

class TestObjectNullSpace(BaseTestObject, unittest.TestCase):
    CLASS  = PETSc.NullSpace
    FACTORY = 'create'
    TARGS = (True, [])

class TestObjectKSP(BaseTestObject, unittest.TestCase):
    CLASS = PETSc.KSP
    FACTORY = 'create'

class TestObjectPC(BaseTestObject, unittest.TestCase):
    CLASS = PETSc.PC
    FACTORY = 'create'

class TestObjectSNES(BaseTestObject, unittest.TestCase):
    CLASS = PETSc.SNES
    FACTORY = 'create'

class TestObjectTS(BaseTestObject, unittest.TestCase):
    CLASS  = PETSc.TS
    FACTORY = 'create'
    def setUp(self):
        super(TestObjectTS, self).setUp()
        self.obj.setProblemType(PETSc.TS.ProblemType.NONLINEAR)
        self.obj.setType(PETSc.TS.Type.BEULER)

class TestObjectTAO(BaseTestObject, unittest.TestCase):
    CLASS  = PETSc.TAO
    FACTORY = 'create'

class TestObjectAOBasic(BaseTestObject, unittest.TestCase):
    CLASS  = PETSc.AO
    FACTORY = 'createBasic'
    TARGS = ([], [])

class TestObjectAOMapping(BaseTestObject, unittest.TestCase):
    CLASS  = PETSc.AO
    FACTORY = 'createMapping'
    TARGS = ([], [])

# class TestObjectFE(BaseTestObject, unittest.TestCase):
#     CLASS  = PETSc.FE
#     FACTORY = 'create'
#
# class TestObjectQuad(BaseTestObject, unittest.TestCase):
#     CLASS  = PETSc.Quad
#     FACTORY = 'create'

class TestObjectDMLabel(BaseTestObject, unittest.TestCase):
    CLASS  = PETSc.DMLabel
    FACTORY = 'create'
    TARGS = ("test",)

class TestObjectSpace(BaseTestObject, unittest.TestCase):
    CLASS  = PETSc.Space
    FACTORY = 'create'

class TestObjectDualSpace(BaseTestObject, unittest.TestCase):
    CLASS  = PETSc.DualSpace
    FACTORY = 'create'

# --------------------------------------------------------------------

import numpy

if numpy.iscomplexobj(PETSc.ScalarType()):
    del TestObjectTAO

if __name__ == '__main__':
    unittest.main()
