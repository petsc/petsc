from petsc4py import PETSc
import unittest

# --------------------------------------------------------------------


class TestDevice(unittest.TestCase):
    def testCurrent(self):
        dctx = PETSc.DeviceContext().getCurrent()
        if not dctx:
            return
        self.assertEqual(dctx.getRefCount(), 2)
        device = dctx.getDevice()
        del device
        del dctx
        dctx = PETSc.DeviceContext().getCurrent()
        self.assertEqual(dctx.getRefCount(), 2)
        device = dctx.getDevice()
        del device
        del dctx

    def testDevice(self):
        device = PETSc.Device.create()
        device.configure()
        _ = device.getDeviceType()
        _ = device.getDeviceId()
        del device

    def testDeviceContext(self):
        dctx = PETSc.DeviceContext().create()
        if not dctx:
            return
        self.assertEqual(dctx.getRefCount(), 1)
        dctx.setUp()
        self.assertTrue(dctx.idle())
        dctx.destroy()
        self.assertEqual(dctx.getRefCount(), 0)

    def testStream(self):
        dctx = PETSc.DeviceContext().getCurrent()
        if not dctx:
            return
        self.assertEqual(dctx.getRefCount(), 2)
        stype = dctx.getStreamType()
        dctx.setStreamType(stype)
        dctx.destroy()
        self.assertEqual(dctx.getRefCount(), 0)

    def testSetFromOptions(self):
        dctx = PETSc.DeviceContext().create()
        if not dctx:
            return
        self.assertEqual(dctx.getRefCount(), 1)
        dctx.setFromOptions()
        dctx.setUp()
        dctx.destroy()
        self.assertEqual(dctx.getRefCount(), 0)

    def testDuplicate(self):
        dctx = PETSc.DeviceContext().getCurrent()
        if not dctx:
            return
        self.assertEqual(dctx.getRefCount(), 2)
        dctx2 = dctx.duplicate()
        self.assertEqual(dctx2.getRefCount(), 1)
        dctx.destroy()
        self.assertEqual(dctx.getRefCount(), 0)
        dctx2.destroy()
        self.assertEqual(dctx2.getRefCount(), 0)

    def testWaitFor(self):
        dctx = PETSc.DeviceContext().create()
        if not dctx:
            return
        self.assertEqual(dctx.getRefCount(), 1)
        dctx.setUp()
        dctx2 = PETSc.DeviceContext().create()
        self.assertEqual(dctx2.getRefCount(), 1)
        dctx2.setUp()
        dctx.waitFor(dctx2)
        dctx.destroy()
        self.assertEqual(dctx.getRefCount(), 0)
        dctx2.destroy()
        dctx2.destroy()
        self.assertEqual(dctx2.getRefCount(), 0)

    def testForkJoin(self):
        dctx = PETSc.DeviceContext().getCurrent()
        if not dctx:
            return
        self.assertEqual(dctx.getRefCount(), 2)
        jdestroy = PETSc.DeviceContext.JoinMode.DESTROY
        jtypes = [
            PETSc.DeviceContext.JoinMode.SYNC,
            PETSc.DeviceContext.JoinMode.NO_SYNC,
        ]
        for j in jtypes:
            dctxs = dctx.fork(4)
            for ctx in dctxs:
                self.assertEqual(ctx.getRefCount(), 1)
            dctx.join(j, dctxs[0::2])
            dctx.join(j, dctxs[3::-2])
            for ctx in dctxs:
                self.assertEqual(ctx.getRefCount(), 1)
            dctx.join(jdestroy, dctxs)
            for ctx in dctxs:
                self.assertEqual(ctx.getRefCount(), 0)
        dctx.destroy()
        self.assertEqual(dctx.getRefCount(), 0)


# --------------------------------------------------------------------

if __name__ == '__main__':
    unittest.main()
