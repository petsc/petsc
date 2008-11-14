#!/usr/bin/env python
import os, sys
sys.path.insert(0, os.path.join(os.environ['PETSC_DIR'], 'config', 'BuildSystem'))
import script
import RDict

class CheckError(RuntimeError):
  pass

class SieveTests(script.Script):
  def __init__(self):
    script.Script.__init__(self, clArgs = ['-log=sieve.log']+sys.argv[1:], argDB = RDict.RDict())
    self.mpiexec    = '$MPIEXEC'
    self.executable = './sieveTests'
    self.debug      = 0
    return

  def setupHelp(self, help):
    '''This method should be overidden to provide help for arguments'''
    import nargs

    script.Script.setupHelp(self, help)
    help.addArgument('SieveTests', '-tests', nargs.Arg(None, ['overlap', 'preallocation'], 'Tests to run', isTemporary = 1))
    help.addArgument('SieveTests', '-procs', nargs.Arg(None, [1], 'Communicator sizes to test', isTemporary = 1))
    help.addArgument('SieveTests', '-iters', nargs.Arg(None, [1], 'Iterations to test', isTemporary = 1))
    return help

  def createCmdLine(self, test, numProcs, num):
    args = [self.mpiexec]
    args.append('-n '+str(numProcs))
    args.append(self.executable)
    args.append('-'+test)
    #args.append('-num '+str(num))
    args.append('-debug '+str(self.debug))
    return ' '.join(args)

  def checkOverlapOutput(self, numProcs):
    # Assuming symmetric pattern
    sendMod = []
    [sendMod.append(__import__('sendOverlap_%d_%d' % (p, numProcs))) for p in range(numProcs)]
    recvMod = []
    [recvMod.append(__import__('recvOverlap_%d_%d' % (p, numProcs))) for p in range(numProcs)]
    # Check for symmetry
    for p in range(numProcs):
      if len(sendMod[p].sendOverlap) != len(recvMod[p].recvOverlap):
        raise CheckError('Communication pattern is not symmetric')
      if sendMod[p].sendOverlap.keys() != recvMod[p].recvOverlap.keys():
        raise CheckError('Communication pattern is not symmetric')
    # Check for send consistency
    for p in range(numProcs):
      for localPoint in sendMod[p].sendOverlap:
        for (rank, remotePoint) in sendMod[p].sendOverlap[localPoint]:
          if sendMod[p].coordinates[localPoint] != recvMod[rank].coordinates[remotePoint]:
            raise CheckError('Mismatch in send overlap')
    # Check for receive consistency
    for p in range(numProcs):
      for localPoint in recvMod[p].recvOverlap:
        for (rank, remotePoint) in recvMod[p].recvOverlap[localPoint]:
          if recvMod[p].coordinates[localPoint] != sendMod[rank].coordinates[remotePoint]:
            raise CheckError('Mismatch in receive overlap')
    # Cleanup
    for p in range(numProcs):
      os.remove('sendOverlap_%d_%d.py' % (p, numProcs))
      os.remove('recvOverlap_%d_%d.py' % (p, numProcs))
    return

  def checkPreallocationOutput(self, numProcs):
    return

  def run(self):
    self.setup()
    for test in self.argDB['tests']:
      for numProcs in map(int, self.argDB['procs']):
        for num in map(int, self.argDB['iters']):
          self.logPrint('Running Sieve test for %s on %d procs and %d iterations' % (test, numProcs, num), 4)
          cmdLine = self.createCmdLine(test, numProcs, num)
          try:
            (output, error, status) = self.executeShellCommand(cmdLine)
            if error:
              raise RuntimeError(error)
            if output: print output
            getattr(self, 'check'+test.capitalize()+'Output')(numProcs)
          except RuntimeError, e:
            print 'ERROR running',cmdLine
            self.logPrint(str(e), 4)
    return

if __name__ == '__main__':
  SieveTests().run()
