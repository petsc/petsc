#!/usr/bin/env python
import os, sys
sys.path.insert(0, os.path.join(os.environ['PETSC_DIR'], 'config', 'BuildSystem'))
import script
import RDict

class MemoryTests(script.Script):
  def __init__(self):
    script.Script.__init__(self, clArgs = ['-log=memory.log'], argDB = RDict.RDict())
    self.executable = './memoryTests'
    self.debug      = 0
    return

  def createCmdLine(self, numProcs, test, num, numCells):
    args = ['$MPIEXEC']
    args.append('-np '+str(numProcs))
    args.append(self.executable)
    args.append('-'+test)
    args.append('-num '+str(num))
    args.append('-numCells '+str(numCells))
    args.append('-debug '+str(self.debug))
    return ' '.join(args)

  def run(self):
    self.setup()
    #for test in ['set', 'label', 'section', 'sectionDist']:
    for test in ['set', 'label', 'section']:
      for numProcs in [1, 2]:
        for num in [1, 2, 10]:
          for numCells in [8, 17]:
            self.logPrint('Running test for %s on %d procs for %d cells and %d iterations' % (test, numProcs, numCells, num))
            cmdLine = self.createCmdLine(numProcs, test, num, numCells)
            try:
              (output, error, status) = self.executeShellCommand(cmdLine)
              if output: print output
            except RuntimeError, e:
              print 'ERROR running',cmdLine
              self.logPrint(str(e), 4)
    return

if __name__ == '__main__':
  MemoryTests().run()
  
