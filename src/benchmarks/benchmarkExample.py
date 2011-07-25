#!/usr/bin/env python
import os

class PETSc(object):
  def __init__(self):
    return

  def dir(self):
    '''Return the root directory for the PETSc tree (usually $PETSC_DIR)'''
    # This should search for a valid PETSc
    return os.environ['PETSC_DIR']

  def arch(self):
    '''Return the PETSc build label (usually $PETSC_ARCH)'''
    # This should be configurable
    return os.environ['PETSC_ARCH']

  def mpiexec(self):
    '''Return the path for the mpi launch executable'''
    mpiexec = os.path.join(self.dir(), self.arch(), 'bin', 'mpiexec')
    if not os.path.isexec(mpiexec):
      return None
    return mpiexec

  def example(self, num):
    '''Return the path to the executable for a given example number'''
    return os.path.join(self.dir(), self.arch(), 'lib', 'ex'+str(num)+'-obj', 'ex'+str(num))

class PETScExample(object):
  def __init__(self, library, num, **defaultOptions):
    self.petsc   = PETSc()
    self.library = library
    self.num     = num
    self.opts    = defaultOptions
    return

  @staticmethod
  def runShellCommand(command, cwd = None):
    import subprocess

    Popen = subprocess.Popen
    PIPE  = subprocess.PIPE
    print 'Executing: %s\n' % (command,)
    pipe = Popen(command, cwd=cwd, stdin=None, stdout=PIPE, stderr=PIPE, bufsize=-1, shell=True, universal_newlines=True)
    (out, err) = pipe.communicate()
    ret = pipe.returncode
    return (out, err, ret)

  def optionsToString(self, **opts):
    '''Convert a dictionary of options to a command line argument string'''
    a = []
    for key,value in opts.iteritems():
      if value is None:
        a.append('-'+key)
      else:
        a.append('-'+key+' '+str(value))
    return ' '.join(a)

  def run(self, **opts):
    if self.petsc.mpiexec() is None:
      cmd = self.petsc.example(self.num)
    else:
      cmd = ' '.join([self.petsc.mpiexec(), '-n 1', self.petsc.example(self.num)])
    cmd += ' '+self.optionsToString(**self.opts)+' '+self.optionsToString(**opts)
    out, err, ret = self.runShellCommand(cmd)
    if ret:
      print err
      print out
    return

def processSummary(moduleName, times, events):
  '''Process the Python log summary into plot data'''
  m = __import__(moduleName)
  reload(m)
  # Total Time
  times.append(m.Time[0])
  # Common events
  #   VecMAXPY and VecMDot essentially give KSPGMRESOrthog
  #   Add the time and flop rate
  for name in ['VecMDot', 'VecMAXPY', 'KSPGMRESOrthog', 'MatMult']:
    if not name in events:
      events[name] = []
    events[name].append((m.Solve.event[name].Time[0], m.Solve.event[name].Flops[0]/(m.Solve.event[name].Time[0] * 1e6)))
  # Particular events
  for name in ['VecCUSPCopyTo', 'VecCUSPCopyFrom', 'MatCUSPCopyTo']:
    if name in m.Solve.event:
      if not name in events:
        events[name] = []
      events[name].append((m.Solve.event[name].Time[0], m.Solve.event[name].Flops[0]/(m.Solve.event[name].Time[0] * 1e6)))
  return

def plotSummaryLine(library, num, sizes, times, events):
  from pylab import legend, plot, show, title, xlabel, ylabel
  import numpy as np
  showTime       = False
  showEventTime  = True
  showEventFlops = True
  arches         = sizes.keys()
  # Time
  if showTime:
    data = []
    for arch in arches:
      data.append(sizes[arch])
      data.append(times[arch])
    plot(*data)
    title('Performance on '+library+' Example '+str(num))
    xlabel('Number of Dof')
    ylabel('Time (s)')
    legend(arches, 'upper left', shadow = True)
    show()
  # Common event time
  #   We could make a stacked plot like Rio uses here
  if showEventTime:
    data  = []
    names = []
    for event, color in [('VecMDot', 'b'), ('VecMAXPY', 'g'), ('MatMult', 'r')]:
      for arch, style in zip(arches, ['-', ':']):
        names.append(arch+' '+event)
        data.append(sizes[arch])
        data.append(np.array(events[arch][event])[:,0])
        data.append(color+style)
    plot(*data)
    title('Performance on '+library+' Example '+str(num))
    xlabel('Number of Dof')
    ylabel('Time (s)')
    legend(names, 'upper left', shadow = True)
    show()
  # Common event flops
  #   We could make a stacked plot like Rio uses here
  if showEventFlops:
    data  = []
    names = []
    for event, color in [('VecMDot', 'b'), ('VecMAXPY', 'g'), ('MatMult', 'r')]:
      for arch, style in zip(arches, ['-', ':']):
        names.append(arch+' '+event)
        data.append(sizes[arch])
        data.append(np.array(events[arch][event])[:,1])
        data.append(color+style)
    plot(*data)
    title('Performance on '+library+' Example '+str(num))
    xlabel('Number of Dof')
    ylabel('Computation Rate (MF/s)')
    legend(names, 'upper left', shadow = True)
    show()
  return

def plotSummaryBar(library, num, sizes, times, events):
  import numpy as np
  import matplotlib.pyplot as plt

  eventNames  = ['VecMDot', 'VecMAXPY', 'MatMult']
  eventColors = ['b',       'g',        'r']
  arches = sizes.keys()
  names  = []
  N      = len(sizes[arches[0]])
  width  = 0.2
  ind    = np.arange(N) - 0.25
  bars   = {}
  for arch in arches:
    bars[arch] = []
    bottom = np.zeros(N)
    for event, color in zip(eventNames, eventColors):
      names.append(arch+' '+event)
      times = np.array(events[arch][event])[:,0]
      bars[arch].append(plt.bar(ind, times, width, color=color, bottom=bottom))
      bottom += times
    ind += 0.3

  plt.xlabel('Number of Dof')
  plt.ylabel('Time (s)')
  plt.title('GPU vs. CPU Performance on '+library+' Example '+str(num))
  plt.xticks(np.arange(N), map(str, sizes[arches[0]]))
  #plt.yticks(np.arange(0,81,10))
  #plt.legend( (p1[0], p2[0]), ('Men', 'Women') )
  plt.legend([bar[0] for bar in bars[arches[0]]], eventNames, 'upper right', shadow = True)

  plt.show()
  return

if __name__ == '__main__':
  library = 'SNES'
  num     = 19
  ex      = PETScExample(library, num, pc_type='none', dmmg_nlevels=1, log_summary='summary.dat', log_summary_python='summary.py', mat_no_inode=None, preload='off')
  sizes   = {}
  times   = {}
  events  = {}
  for name, vecType, matType, opts in [('CPU', 'seq', 'seqaij', {}), ('GPU', 'seqcusp', 'seqaijcusp', {'cusp_synchronize': None})]:
    sizes[name]  = []
    times[name]  = []
    events[name] = {}
    #for n in [10, 20, 50, 100, 150, 200]:
    for n in [10, 20]:
      ex.run(da_grid_x=n, da_grid_y=n, da_vec_type=vecType, da_mat_type=matType, **opts)
      sizes[name].append(n*n * 4)
      processSummary('summary', times[name], events[name])
  plotSummaryLine(library, num, sizes, times, events)
