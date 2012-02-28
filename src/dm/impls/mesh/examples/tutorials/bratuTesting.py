#!/usr/bin/env python
'''
TODO:

  1) Put in Neumann conditions for Bratu FD

Testing Schedule:

  - Laplacian (bratu.cxx)
    - Dirichlet
      - many mesh sizes
        - SG, ICC(0), Jacobi, (GMG)
    - Neumann
      - many mesh sizes
        - SG, ICC(0), Jacobi, (GMG), maybe structured MG

  - Linear Elasticity (PyLith benchmarks)

  - Stokes (stokes.cxx)

  - Groundwater flow (PFLOTRAN)

Data:
  - Mesh maximum cell volume
  - Number of Rows
  - Number of iterates

Timing:
  - Total Time
  - MatMult()
  - PCSetUp()
  - PCApply()
  - KSPSolve()
'''
import script
import sys

class BratuTest(script.Script):
  def __init__(self):
    import RDict
    script.Script.__init__(self, argDB = RDict.RDict())
    self.executable = './bratu'
    self.data       = []
    return

  def setupHelp(self, help):
    import nargs

    script.Script.setupHelp(self, help)
    help.addArgument('BratuTest', '-num_refine=<int>', nargs.ArgInt(None, 1, 'Number of refinements', min = 0))
    help.addArgument('BratuTest', '-bc_type=<dirichlet or neumann>', nargs.Arg(None, 'dirichlet', 'PETSc boundary condition type'))
    help.addArgument('BratuTest', '-pc_type=<typename>', nargs.Arg(None, 'supportgraph', 'PETSc PC type'))
    help.addArgument('BratuTest', '-events=[event1,event2,...]', nargs.Arg(None, ['PCSetUp'], 'Events to monitor'))
    return

  def setupOptions(self):
    self.defaultOptions = ['-structured 0', '-ksp_rtol 1.0e-9', '-ksp_monitor', '-ksp_view', '-log_summary']
    self.defaultOptions.append('-pc_type '+self.argDB['pc_type'])
    self.defaultOptions.append('-bc_type '+self.argDB['bc_type'])
    return

  def setup(self):
    script.Script.setup(self)
    self.setupOptions()
    return

  def addOutput(self, maxVolume, numRows, numNonzeros, numIterates, TotalTime):
    self.data.append((maxVolume, numRows, numNonzeros, numIterates, TotalTime))
    return

  def processOutput(self, out):
    import re

    events  = re.compile(r'(?P<event>\w+)\s+(?P<count>\d+) \d\.\d (?P<time>\d\.\d+e[+-]\d\d) \d\.\d (?P<flops>\d\.\d+e[+-]\d\d) \d\.\d \d\.\d+e[+-]\d\d \d\.\d+e[+-]\d\d \d\.\d+e[+-]\d\d\s+\d+\s+\d+\s+\d+\s+\d+\s+\d+\s+\d+\s+\d+\s+\d+\s+\d+\s+\d+\s+(?P<mflops>\d+)')
    kspMon  = re.compile(r'\s*(?P<it>\d+) KSP Residual norm \d\.\d+e[+-]\d\d')
    matSize = re.compile(r'\s+type=\w+, rows=(?P<rows>\d+), cols=(?P=rows)')
    iters   = 0
    for line in out[0].split('\n'):
      if line.startswith('Reason for solver termination'):
        if not line.split(':')[1] == ' CONVERGED_FNORM_RELATIVE':
          sys.exit('ERROR: Invalid termination: '+str(line.split(':')[1]))
      else:
        matchEvent = events.match(line)
        if matchEvent:
          print matchEvent.group('event'), matchEvent.group('count'), matchEvent.group('time'), matchEvent.group('mflops')
          if matchEvent.group('event') in self.argDB['events']:
            print '  MONITORED'
        else:
          matchKsp = kspMon.match(line)
          if matchKsp:
            iters = max(int(matchKsp.group('it')), iters)
          else:
            matchMat = matSize.match(line)
            if matchMat:
              rows = int(matchMat.group('rows'))
    return (rows, iters)

  def testLS(self):
    print 'Test Low-Stretch'
    area   = 0.125
    factor = 2.0
    iters  = []
    for i in range(self.argDB['num_refine']+1):
      print '  testing refinement',i,'area', area
      cmd = ' '.join([self.executable]+self.defaultOptions+['-refinement_limit '+str(area)])
      numRows, numIterates = self.processOutput(self.executeShellCommand(cmd)[:-2])
      self.addOutput(area, numRows, -1, numIterates, -1)
      iters.append((numRows, numIterates))
      area /= factor
    print 'rows and iterations:', iters
    return iters

  def run(self):
    self.setup()
    self.testLS()
    return

if __name__ == '__main__':
  BratuTest().run()
