import unittest
from petsc4py import PETSc
from sys import getrefcount

# --------------------------------------------------------------------

class TestOptions(unittest.TestCase):

    PREFIX  = 'myopts-'
    OPTLIST = [('bool',   True),
               ('int',    -7),
               ('real',   5),
               ('scalar', 3),
               ('string', 'petsc4py'),
               ]

    def _putopts(self, opts=None, OPTLIST=None):
        if opts is None:
            opts = self.opts
        if OPTLIST is None:
            OPTLIST = self.OPTLIST
        for k,v in OPTLIST:
            opts[k] = v
    def _delopts(self, opts=None, OPTLIST=None):
        if opts is None:
            opts = self.opts
        if OPTLIST is None:
            OPTLIST = self.OPTLIST
        for k,v in OPTLIST:
            del opts[k]

    def setUp(self):
        self.opts = PETSc.Options(self.PREFIX)

    def tearDown(self):
        self.opts = None
        PETSc.garbage_cleanup()

    def testHasOpts(self):
        self._putopts()
        for k, v in self.OPTLIST:
            self.assertTrue(self.opts.hasName(k))
            self.assertTrue(k in self.opts)
            missing = k+'-missing'
            self.assertFalse(self.opts.hasName(missing))
            self.assertFalse(missing in self.opts)
        self._delopts()

    def testGetOpts(self):
        self._putopts()
        for k, v in self.OPTLIST:
            getopt = getattr(self.opts, 'get'+k.title())
            self.assertEqual(getopt(k), v)
        self._delopts()

    def testGetAll(self):
        self._putopts()
        allopts = self.opts.getAll()
        self.assertTrue(type(allopts) is dict)
        optlist = [(k, str(v).lower())
                   for (k,v) in self.OPTLIST]
        for k,v in allopts.items():
            self.assertTrue((k, v) in optlist)
        self._delopts()

    def testGetAllQuoted(self):
        dct = {'o0' : '"0 1 2"',
               'o1' : '"a b c"',
               'o2' : '"x y z"',}
        for k in dct:
            self.opts[k] =  dct[k]
        allopts = self.opts.getAll()
        for k in dct:
            self.assertEqual(allopts[k], dct[k][1:-1])
            del self.opts[k]

    def testMonitor(self):
        optlist = []
        mon = lambda n,v: optlist.append((n,v))
        self.opts.setMonitor(mon)
        self.assertEqual(getrefcount(mon)-1, 2)
        self._putopts()
        target = [(self.PREFIX+k, str(v).lower())
                  for k, v in self.OPTLIST]
        self.assertEqual(optlist, target)
        self.opts.cancelMonitor()
        self.assertEqual(getrefcount(mon)-1, 1)
        self._delopts()

# --------------------------------------------------------------------

del TestOptions.testMonitor # XXX

if __name__ == '__main__':
    unittest.main()
