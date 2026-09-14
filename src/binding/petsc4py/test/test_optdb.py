import unittest
from petsc4py import PETSc
from sys import getrefcount
import numpy as np

# --------------------------------------------------------------------


class TestOptions(unittest.TestCase):
    PREFIX = 'myopts-'
    OPTLIST = [
        ('bool', True),
        ('int', -7),
        ('real', 5),
        ('scalar', 3),
        ('string', 'petsc4py'),
    ]

    def _putopts(self, opts=None, OPTLIST=None):
        if opts is None:
            opts = self.opts
        if OPTLIST is None:
            OPTLIST = self.OPTLIST
        for k, v in OPTLIST:
            opts[k] = v

    def _delopts(self, opts=None, OPTLIST=None):
        if opts is None:
            opts = self.opts
        if OPTLIST is None:
            OPTLIST = self.OPTLIST
        for k, _ in OPTLIST:
            del opts[k]

    def setUp(self):
        self.opts = PETSc.Options(self.PREFIX)

    def tearDown(self):
        self.opts = None
        PETSc.garbage_cleanup()

    def testHasOpts(self):
        self._putopts()
        for k, _ in self.OPTLIST:
            self.assertTrue(self.opts.hasName(k))
            self.assertTrue(k in self.opts)
            missing = k + '-missing'
            self.assertFalse(self.opts.hasName(missing))
            self.assertFalse(missing in self.opts)
        self._delopts()

    def testGetOpts(self):
        self._putopts()
        for k, v in self.OPTLIST:
            getopt = getattr(self.opts, 'get' + k.title())
            self.assertEqual(getopt(k), v)
        self._delopts()

    def testUsed(self):
        for prefix in (None, self.PREFIX):
            opts = PETSc.Options(prefix).create()
            for name in ('value', '-value'):
                with self.subTest(prefix=prefix, name=name):
                    opts.setValue(name, 7)
                    self.assertFalse(opts.used(name))
                    self.assertEqual(opts.getInt(name), 7)
                    self.assertTrue(opts.used(name))
                    opts.delValue(name)
                    self.assertFalse(opts.used(name))
            opts.destroy()

    def testGetAll(self):
        self._putopts()
        allopts = self.opts.getAll()
        self.assertTrue(isinstance(allopts, dict))
        optlist = [(k, str(v).lower()) for (k, v) in self.OPTLIST]
        for k, v in allopts.items():
            self.assertTrue((k, v) in optlist)
        self._delopts()

    def testGetAllQuoted(self):
        dct = {
            'o0': '"0 1 2"',
            'o1': '"a b c"',
            'o2': '"x y z"',
        }
        for k, v in dct.items():
            self.opts[k] = v
        allopts = self.opts.getAll()
        for k in dct:
            self.assertEqual(allopts[k], dct[k][1:-1])
            del self.opts[k]

    def testType(self):
        types = [
            (bool, bool, self.opts.getBool, self.opts.getBoolArray),
            (int, PETSc.IntType, self.opts.getInt, self.opts.getIntArray),
            (float, PETSc.RealType, self.opts.getReal, self.opts.getRealArray),
        ]
        if PETSc.ScalarType is PETSc.ComplexType:
            types.append(
                (
                    complex,
                    PETSc.ScalarType,
                    self.opts.getScalar,
                    self.opts.getScalarArray,
                )
            )
        else:
            types.append(
                (
                    float,
                    PETSc.ScalarType,
                    self.opts.getScalar,
                    self.opts.getScalarArray,
                )
            )
        toval = (lambda x: x, lambda x: np.array(x).tolist(), np.array)
        sv = 1
        av = (1, 0, 1)
        defv = 0
        defarrayv = (0, 0, 1, 0)
        for pyt, pat, pget, pgetarray in types:
            for tov in toval:
                self.opts.setValue('sv', tov(sv))
                self.opts.setValue('av', tov(av))

                v = pget('sv')
                self.assertTrue(isinstance(v, pyt))
                self.assertEqual(v, pyt(sv))

                v = pget('sv', defv)
                self.assertTrue(isinstance(v, pyt))
                self.assertEqual(v, pyt(sv))

                v = pget('missing', defv)
                self.assertTrue(isinstance(v, pyt))
                self.assertEqual(v, pyt(defv))

                if pgetarray is not None:
                    arrayv = pgetarray('av')
                    self.assertEqual(arrayv.dtype, pat)
                    self.assertEqual(len(arrayv), len(av))
                    for v1, v2 in zip(arrayv, av):
                        self.assertTrue(isinstance(v1.item(), pyt))
                        self.assertEqual(v1.item(), pyt(v2))

                    arrayv = pgetarray('av', defarrayv)
                    self.assertEqual(arrayv.dtype, pat)
                    self.assertEqual(len(arrayv), len(av))
                    for v1, v2 in zip(arrayv, av):
                        self.assertTrue(isinstance(v1.item(), pyt))
                        self.assertEqual(v1.item(), pyt(v2))

                    arrayv = pgetarray('missing', defarrayv)
                    self.assertEqual(arrayv.dtype, pat)
                    self.assertEqual(len(arrayv), len(defarrayv))
                    for v1, v2 in zip(arrayv, defarrayv):
                        self.assertTrue(isinstance(v1.item(), pyt))
                        self.assertEqual(v1.item(), pyt(v2))

                self.opts.delValue('sv')
                self.opts.delValue('av')

    def testMonitor(self):
        optlist = []
        mon = lambda n, v: optlist.append((n, v))
        self.opts.setMonitor(mon)
        self.assertEqual(getrefcount(mon) - 1, 2)
        self._putopts()
        target = [(self.PREFIX + k, str(v).lower()) for k, v in self.OPTLIST]
        self.assertEqual(optlist, target)
        self.opts.cancelMonitor()
        self.assertEqual(getrefcount(mon) - 1, 1)
        self._delopts()

    @unittest.skipUnless(
        PETSc.ScalarType is PETSc.ComplexType, 'requires complex scalars'
    )
    def testComplexValues(self):
        opts = PETSc.Options(self.PREFIX).create()
        values = (1 + 2j, 3 - 4j, 2j, -3j)
        for convert in (complex, PETSc.ScalarType, np.array):
            for value in values:
                with self.subTest(convert=convert, value=value):
                    opts.setValue('scalar', convert(value))
                    self.assertEqual(opts.getScalar('scalar'), value)
        for convert in (list, tuple, lambda v: np.array(v, dtype=PETSc.ScalarType)):
            with self.subTest(convert=convert):
                opts.setValue('array', convert(values))
                np.testing.assert_array_equal(opts.getScalarArray('array'), values)
        opts.setValue('string', 'jacobi')
        self.assertEqual(opts.getString('string'), 'jacobi')
        opts.destroy()


# --------------------------------------------------------------------

del TestOptions.testMonitor  # XXX

if __name__ == '__main__':
    unittest.main()
