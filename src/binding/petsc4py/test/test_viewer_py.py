# --------------------------------------------------------------------

from petsc4py import PETSc
import unittest

# --------------------------------------------------------------------


class EmptyViewer:
    def __init__(self):
        self.log = {}

    def _log(self, method, *args):
        self.log.setdefault(method, 0)
        self.log[method] += 1

    def setUp(self, viewer):
        self._log('setUp')

    def view(self, viewer, outviewer):
        self._log('view')

    def setFromOptions(self, viewer):
        self._log('setFromOptions')

    def flush(self, viewer):
        self._log('flush')


class PythonViewer(EmptyViewer):
    obj_viewed = []

    def viewObject(self, viewer, pobj):
        self.obj_viewed.append(pobj.klass)


# --------------------------------------------------------------------


class BaseTestViewPYTHON:
    ContextClass = None
    ContextName = None

    def setUp(self):
        self.viewer = PETSc.Viewer()
        self.viewer.create(PETSc.COMM_SELF)
        self.viewer.setType(PETSc.Viewer.Type.PYTHON)
        if self.ContextClass is not None:
            ctx = self.ContextClass()
            self.viewer.setPythonContext(ctx)
            self.viewer.setFromOptions()
            self.viewer.setUp()
        elif self.ContextName is not None:
            self.viewer.setPythonType(self.ContextName)
            self.viewer.setFromOptions()
            self.viewer.setUp()
        ctx = self.viewer.getPythonContext()
        if ctx is not None:
            self.assertTrue(ctx.log['setFromOptions'] == 1)
            self.assertTrue(ctx.log['setUp'] == 1)

    def tearDown(self):
        self.viewer.destroy()
        PETSc.garbage_cleanup()

    def testGetType(self):
        ctx = self.viewer.getPythonContext()
        pytype = None
        if ctx is not None:
            pytype = f'{ctx.__module__}.{type(ctx).__name__}'
        self.assertTrue(self.viewer.getPythonType() == pytype)

    def testViewObject(self):
        v = PETSc.Vec().create(PETSc.COMM_SELF)
        self.viewer.viewObjectPython(v)
        v.destroy()
        v = PETSc.KSP().create(PETSc.COMM_SELF)
        self.viewer.viewObjectPython(v)
        v.destroy()
        v = PETSc.DM().create(PETSc.COMM_SELF)
        v.setFromOptions()
        self.viewer.viewObjectPython(v)
        v.destroy()
        ctx = self.viewer.getPythonContext()
        if ctx is not None and hasattr(ctx, 'obj_viewed'):
            ov = ctx.obj_viewed
            self.assertTrue(len(ov) == 3)
            self.assertTrue(ov[0] == 'Vec')
            self.assertTrue(ov[1] == 'KSP')
            self.assertTrue(ov[2] == 'DM')

    # def testView(self):
    #     self.viewer.view()


class TestNone(BaseTestViewPYTHON, unittest.TestCase):
    ContextClass = None


class TestEmptyViewer(BaseTestViewPYTHON, unittest.TestCase):
    ContextClass = EmptyViewer


class TestPythonView(BaseTestViewPYTHON, unittest.TestCase):
    ContextClass = PythonViewer


# --------------------------------------------------------------------

if __name__ == '__main__':
    unittest.main()
