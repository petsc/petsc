class Tester:
  def __init__(self, clArgs = None, useMpi = 1, usePetsc = 1):
    self.numTests            = len(filter(self.isTest, dir(self)))
    self.numSuccesses        = 0
    self.numFailures         = 0
    self.numExpectedFailures = 0
    self.children            = []
    self.useMpi              = useMpi
    self.usePetsc            = usePetsc
    return

  def write(self, string):
    if self.comm and self.comm.rank() == 0:
      print string
    return

  def addChild(self, tester):
    self.children.append(tester)
    self.numTests += tester.numTests
    return

  def isTest(self, funcName):
    return funcName[0:4] == 'test'

  def describeTest(self, test):
    self.write('TEST: '+str(test.im_class)+'.'+test.im_func.__name__)
    self.write('DESCRIPTION: '+test.__doc__)
    return
        
  def finalize(self):
    numTests = self.numSuccesses + self.numFailures + self.numExpectedFailures
    if not numTests == self.numTests:
      raise RuntimeError('Invalid result logging: %d != %d' % (numTests == self.numTests))
    if self.numFailures > 0:
      self.write('TEST_RESULT FAIL')
      return 0
    elif self.numExpectedFailures > 0:
      self.write('TEST_RESULT XFAIL')
      return 0
    else:
      self.write('TEST_RESULT PASS')
    return 1

  def runTests(self):
    for funcName in dir(self):
      if not self.isTest(funcName): continue
      test = getattr(self, funcName)
      self.describeTest(test)
      if hasattr(self, 'setupTest'): getattr(self, 'setupTest')()
      try:
        ret = test()
      except:
        import sys
        e   = sys.exc_info()[1]
        msg = 'Unexpected exception: '
        if hasattr(e, 'getMessage'):
          msg += e.getMessage()
        else:
          msg += str(e)
        ret = (0, msg)
      if isinstance(ret, tuple):
        if not len(ret) == 2: raise RuntimeError('Invalid test return: '+str(ret))
        self.write('COMMENT: '+str(ret[1]))
        ret = ret[0]
      if ret:
        self.numSuccesses += 1
        self.write('RESULT: PASS')
      else:
        self.numFailures += 1
        self.write('RESULT: FAIL')
      if hasattr(self, 'cleanupTest'): getattr(self, 'cleanupTest')()
    return (self.numSuccesses, self.numFailures, self.numExpectedFailures)

  def run(self):
    import SIDL.Args
    import SIDL.Loader
    import SIDL.ProjectState
    import MPIB.Base
    import sys

    SIDL.Args.set(sys.argv)
    if self.useMpi:
      mpi = MPIB.Base.Base(SIDL.Loader.createClass('MPIB.Default.DefaultBase'))
      mpi.Initialize()
      self.comm = mpi.comm().WORLD()
      if self.usePetsc:
        petsc = SIDL.ProjectState.ProjectState(SIDL.Loader.createClass('PETSc.State'))
        petsc.Initialize()
    self.write('NUM_TESTS '+str(self.numTests))
    self.runTests()
    for child in self.children:
      child.comm = self.comm
      (numSuccesses, numFailures, numExpectedFailures) = child.runTests()
      self.numSuccesses        += numSuccesses
      self.numFailures         += numFailures
      self.numExpectedFailures += numExpectedFailures
    ret = self.finalize()
    if self.useMpi:
      mpi.Finalize()
      if self.usePetsc:
        petsc.Finalize()
    return ret
