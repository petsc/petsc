#!/usr/bin/env python
import os,sys
sys.path.append(os.path.join(os.environ['PETSC_DIR'], 'config'))
from builder2 import buildExample
from benchmarkBatch import generateBatchScript

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
    if not os.path.isfile(mpiexec):
      return None
    return mpiexec

  def example(self, num):
    '''Return the path to the executable for a given example number'''
    return os.path.join(self.dir(), self.arch(), 'lib', 'ex'+str(num)+'-obj', 'ex'+str(num))

  def source(self, library, num):
    '''Return the path to the sources for a given example number'''
    d = os.path.join(self.dir(), 'src', library.lower(), 'examples', 'tutorials')
    name = 'ex'+str(num)
    sources = []
    for f in os.listdir(d):
      if f == name+'.c':
        sources.insert(0, f)
      elif f.startswith(name) and f.endswith('.cu'):
        sources.append(f)
    return map(lambda f: os.path.join(d, f), sources)

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

  def run(self, numProcs = 1, **opts):
    if self.petsc.mpiexec() is None:
      cmd = self.petsc.example(self.num)
    else:
      cmd = ' '.join([self.petsc.mpiexec(), '-n', str(numProcs), self.petsc.example(self.num)])
    cmd += ' '+self.optionsToString(**self.opts)+' '+self.optionsToString(**opts)
    if 'batch' in opts and opts['batch']:
      del opts['batch']
      filename = generateBatchScript(self.num, numProcs, 120, ' '+self.optionsToString(**self.opts)+' '+self.optionsToString(**opts))
      # Submit job
      out, err, ret = self.runShellCommand('qsub -q gpu '+filename)
      if ret:
        print err
        print out
    else:
      out, err, ret = self.runShellCommand(cmd)
      if ret:
        print err
        print out
    return out

def processSummary(moduleName, defaultStage, eventNames, times, events):
  '''Process the Python log summary into plot data'''
  m = __import__(moduleName)
  reload(m)
  # Total Time
  times.append(m.Time[0])
  # Particular events
  for name in eventNames:
    if name.find(':') >= 0:
      stageName, name = name.split(':', 1)
      stage = getattr(m, stageName)
    else:
      stage = getattr(m, defaultStage)
    if name in stage.event:
      if not name in events:
        events[name] = []
      events[name].append((stage.event[name].Time[0], stage.event[name].Flops[0]/(stage.event[name].Time[0] * 1e6)))
  return

def plotSummaryLine(library, num, eventNames, sizes, times, events):
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
    for event, color in zip(eventName, ['b', 'g', 'r', 'y']:
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
    for event, color in zip(eventName, ['b', 'g', 'r', 'y']:
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

def plotSummaryBar(library, num, eventNames, sizes, times, events):
  import numpy as np
  import matplotlib.pyplot as plt

  eventColors = ['b', 'g', 'r', 'y']
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

def getDMMeshSize(dim, out):
  '''Retrieves the number of cells from '''
  size = 0
  for line in out.split('\n'):
    if line.strip().startswith(str(dim)+'-cells: '):
      size = int(line.strip()[9:])
      break
  return size

def run_DMDA(ex, name, opts, args, sizes, times, events):
  for n in map(int, args.size):
    ex.run(da_grid_x=n, da_grid_y=n, **opts)
    sizes[name].append(n*n * args.comp)
    processSummary('summary', args.stage, args.events, times[name], events[name])
  return

def run_DMMesh(ex, name, opts, args, sizes, times, events):
  # This should eventually be replaced by a direct FFC/Ignition interface
  if args.operator == 'laplacian':
    numComp  = 1
  elif args.operator == 'elasticity':
    numComp  = args.dim
  else:
    raise RuntimeError('Unknown operator: %s' % args.operator)

  for numBlock in [2**i for i in map(int, args.blockExp)]:
    opts['gpu_blocks'] = numBlock
    # Generate new block size
    cmd = './bin/pythonscripts/PetscGenerateFEMQuadrature.py %d %d %d %d %s %s.h' % (args.dim, args.order, numComp, numBlock, args.operator, os.path.splitext(source[0])[0])
    print(cmd)
    ret = os.system('python '+cmd)
    args.files = ['['+','.join(source)+']']
    buildExample(args)
    sizes[name][numBlock]  = []
    times[name][numBlock]  = []
    events[name][numBlock] = {}
    for r in map(float, args.refine):
      out = ex.run(refinement_limit=r, **opts)
      sizes[name][numBlock].append(getDMMeshSize(args.dim, out))
      processSummary('summary', args.stage, args.events, times[name][numBlock], events[name][numBlock])
  return

if __name__ == '__main__':
  import argparse

  parser = argparse.ArgumentParser(description     = 'PETSc Benchmarking',
                                   epilog          = 'This script runs src/<library>/examples/tutorials/ex<num>, For more information, visit http://www.mcs.anl.gov/petsc',
                                   formatter_class = argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--library', default='SNES',                     help='The PETSc library used in this example')
  parser.add_argument('--num',     type = int, default='5',            help='The example number')
  parser.add_argument('--module',  default='summary',                  help='The module for timing output')
  parser.add_argument('--stage',   default='Main_Stage',               help='The default logging stage')
  parser.add_argument('--events',  nargs='+',                          help='Events to process')
  parser.add_argument('--batch',   action='store_true', default=False, help='Generate batch files for the runs instead')
  subparsers = parser.add_subparsers(help='DM types')

  parser_dmda = subparsers.add_parser('DMDA', help='Use a DMDA for the problem geometry')
  parser_dmda.add_argument('--size', nargs='+',  default=['10'], help='Grid size (implementation dependent)')
  parser_dmda.add_argument('--comp', type = int, default='1',    help='Number of field components')
  parser_dmda.add_argument('runs',   nargs='*',                  help='Run descriptions: <name>=<args>')

  parser_dmmesh = subparsers.add_parser('DMMesh', help='Use a DMMesh for the problem geometry')
  parser_dmmesh.add_argument('--dim',      type = int, default='2',        help='Spatial dimension')
  parser_dmmesh.add_argument('--refine',   nargs='+',  default=['0.0'],    help='List of refinement limits')
  parser_dmmesh.add_argument('--order',    type = int, default='1',        help='Order of the finite element')
  parser_dmmesh.add_argument('--operator', default='laplacian',            help='The operator name')
  parser_dmmesh.add_argument('--blockExp', nargs='+', default=range(0, 5), help='List of block exponents j, block size is 2^j')
  parser_dmmesh.add_argument('runs',       nargs='*',                      help='Run descriptions: <name>=<args>')

  args = parser.parse_args()
  print(args)
  if hasattr(args, 'comp'):
    args.dmType = 'DMDA'
  else:
    args.dmType = 'DMMesh'

  ex     = PETScExample(args.library, args.num, log_summary='summary.dat', log_summary_python = None if args.batch else args.module+'.py', preload='off')
  source = ex.petsc.source(args.library, args.num)
  sizes  = {}
  times  = {}
  events = {}

  for run in args.runs:
    name, stropts = run.split('=', 1)
    opts = dict([t if len(t) == 2 else (t[0], None) for t in [arg.split('=', 1) for arg in stropts.split(' ')]])
    if args.dmType == 'DMDA':
      sizes[name]  = []
      times[name]  = []
      events[name] = {}
      run_DMDA(ex, name, opts, args, sizes, times, events)
    elif args.dmType == 'DMMesh':
      sizes[name]  = {}
      times[name]  = {}
      events[name] = {}
      run_DMMesh(ex, name, opts, args, sizes, times, events)
  print('sizes',sizes)
  print('times',times)
  print('events',events)
  if not args.batch: plotSummaryLine(args.library, args.num, args.events, sizes, times, events)
# Benchmark for ex50
# ./src/benchmarks/benchmarkExample.py --events VecMDot VecMAXPY KSPGMRESOrthog MatMult VecCUSPCopyTo VecCUSPCopyFrom MatCUSPCopyTo --num 50 DMDA --size 10 20 50 100 --comp 4 CPU='pc_type=none mat_no_inode dm_vec_type=seq dm_mat_type=seqaij' GPU='pc_type=none mat_no_inode dm_vec_type=seqcusp dm_mat_type=seqaijcusp cusp_synchronize'
# Benchmark for ex52
# ./src/benchmarks/benchmarkExample.py --events IntegBatchCPU IntegBatchGPU IntegGPUOnly --num 52 DMMesh --refine 0.0625 0.00625 0.000625 0.0000625 --blockExp 4 --order 1 CPU='dm_view show_residual=0 compute_function batch' GPU='dm_view show_residual=0 compute_function batch gpu gpu_batches=8'
