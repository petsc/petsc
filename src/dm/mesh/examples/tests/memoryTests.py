#!/usr/bin/env python
import os, sys
sys.path.insert(0, os.path.join(os.environ['PETSC_DIR'], 'config', 'BuildSystem'))
import script
import RDict

class MemoryTests(script.Script):
  def __init__(self):
    script.Script.__init__(self, clArgs = ['-log=memory.log']+sys.argv[1:], argDB = RDict.RDict())
    self.executable = './memoryTests'
    self.debug      = 0
    return

  def setupHelp(self, help):
    '''This method should be overidden to provide help for arguments'''
    import nargs

    script.Script.setupHelp(self, help)
    help.addArgument('MemoryTests', '-tests', nargs.Arg(None, ['set', 'label', 'section', 'sectionDist'], 'Tests to run', isTemporary = 1))
    help.addArgument('MemoryTests', '-procs', nargs.Arg(None, [1, 2], 'Communicator sizes to test', isTemporary = 1))
    help.addArgument('MemoryTests', '-iters', nargs.Arg(None, [1, 2, 10], 'Iterations to test', isTemporary = 1))
    help.addArgument('MemoryTests', '-cells', nargs.Arg(None, [8, 17], 'Cell sizes to test', isTemporary = 1))
    return help

  def createCmdLine(self, numProcs, test, num, numCells):
    args = ['$MPIEXEC']
    args.append('-n '+str(numProcs))
    args.append(self.executable)
    args.append('-'+test)
    args.append('-num '+str(num))
    args.append('-numCells '+str(numCells))
    args.append('-debug '+str(self.debug))
    return ' '.join(args)

  def run(self):
    self.setup()
    #for test in ['set', 'label', 'section', 'sectionDist']:
    for test in self.argDB['tests']:
      for numProcs in map(int, self.argDB['procs']):
        for num in map(int, self.argDB['iters']):
          for numCells in map(int, self.argDB['cells']):
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
  
