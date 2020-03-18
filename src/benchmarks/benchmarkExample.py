#!/usr/bin/env python
from __future__ import print_function
import os,sys
sys.path.append(os.path.join(os.environ['PETSC_DIR'], 'config'))
sys.path.append(os.getcwd())

class CSVLog(object): pass

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

  def example(self, library, num):
    '''Return the path to the executable for a given example number'''
    return os.path.join(self.dir(), self.arch(), 'lib', library.lower(), 'ex'+str(num))

  def source(self, library, num, filenametail):
    '''Return the path to the sources for a given example number'''
    d = os.path.join(self.dir(), 'src', library.lower(), 'examples', 'tutorials')
    name = 'ex'+str(num)
    sources = []
    for f in os.listdir(d):
      if f == name+'.c':
        sources.insert(0, f)
      elif f.startswith(name) and f.endswith(filenametail):
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
  def runShellCommand(command, cwd = None, log = True):
    import subprocess

    Popen = subprocess.Popen
    PIPE  = subprocess.PIPE
    if log: print('Executing: %s\n' % (command,))
    pipe = Popen(command, cwd=cwd, stdin=None, stdout=PIPE, stderr=PIPE, bufsize=-1, shell=True, universal_newlines=True)
    (out, err) = pipe.communicate()
    ret = pipe.returncode
    return (out, err, ret)

  def optionsToString(self, **opts):
    '''Convert a dictionary of options to a command line argument string'''
    a = []
    for key,value in opts.items():
      if value is None:
        a.append('-'+key)
      else:
        a.append('-'+key+' '+str(value))
    return ' '.join(a)

  def build(self, log = True):
    sdir = os.path.join(self.petsc.dir(), 'src', self.library.lower(), 'examples', 'tutorials')
    odir = os.getcwd()
    os.chdir(sdir)
    cmd = 'make ex'+str(self.num)
    out, err, ret = self.runShellCommand(cmd, cwd = sdir, log = log)
    if err:
      raise RuntimeError('Unable to build example:\n'+err+out)
    os.chdir(odir)
    bdir = os.path.dirname(self.petsc.example(self.library, self.num))
    try:
      os.makedirs(bdir)
    except OSError:
      if not os.path.isdir(bdir):
        raise
    cmd = 'mv '+os.path.join(sdir, 'ex'+str(self.num))+' '+self.petsc.example(self.library, self.num)
    out, err, ret = self.runShellCommand(cmd, log = log)
    if ret:
      print(err)
      print(out)
    return

  def run(self, numProcs = 1, log = True, **opts):
    cmd = ''
    if self.petsc.mpiexec() is not None:
      cmd += self.petsc.mpiexec() + ' '
      numProcs = os.environ.get('NUM_RANKS', numProcs)
      cmd += ' -n ' + str(numProcs) + ' '
      if 'PE_HOSTFILE' in os.environ:
        cmd += ' -hostfile hostfile '
    cmd += ' '.join([self.petsc.example(self.library, self.num), self.optionsToString(**self.opts), self.optionsToString(**opts)])
    if 'batch' in opts and opts['batch']:
      del opts['batch']
      from benchmarkBatch import generateBatchScript
      filename = generateBatchScript(self.num, numProcs, 120, ' '+self.optionsToString(**self.opts)+' '+self.optionsToString(**opts))
      # Submit job
      out, err, ret = self.runShellCommand('qsub -q gpu '+filename, log = log)
      if ret:
        print(err)
        print(out)
    else:
      out, err, ret = self.runShellCommand(cmd, log = log)
      if ret:
        print(err)
        print(out)
    return out

def processSummaryCSV(filename, defaultStage, eventNames, sizes, times, errors, events):
  '''Process the CSV log summary into plot data'''
  import csv
  m = CSVLog()
  setattr(m, 'Stages', {})
  with open(filename) as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
      stageName = row["Stage Name"]
      eventName = row["Event Name"]
      rank      = int(row["Rank"])
      if not stageName in m.Stages: m.Stages[stageName] = {}
      if not eventName in m.Stages[stageName]: m.Stages[stageName][eventName] = {}
      m.Stages[stageName][eventName][rank] = {"time" : float(row["Time"]), "numMessages": float(row["Num Messages"]), "messageLength": float(row["Message Length"]), "numReductions" : float(row["Num Reductions"]), "flop" : float(row["FLOP"])}
      for i in range(8):
        dname = "dof"+str(i)
        ename = "e"+str(i)
        if row[dname]:
          if not "dof" in m.Stages[stageName][eventName][rank]:
            m.Stages[stageName][eventName][rank]["dof"] = []
            m.Stages[stageName][eventName][rank]["error"] = []
          m.Stages[stageName][eventName][rank]["dof"].append(int(float(row[dname])))
          m.Stages[stageName][eventName][rank]["error"].append(float(row[ename]))
  return m

def processSummary(moduleName, defaultStage, eventNames, sizes, times, errors, events):
  '''Process the Python log summary into plot data'''
  if os.path.isfile(moduleName+'.csv'):
    m = processSummaryCSV(moduleName+'.csv', defaultStage, eventNames, sizes, times, errors, events)
  else:
    m = __import__(moduleName)
  # Total Time
  times.append(m.Stages[defaultStage]["summary"][0]["time"])
  # Particular events
  for name in eventNames:
    if name.find(':') >= 0:
      stageName, name = name.split(':', 1)
    else:
      stageName = defaultStage
    stage = m.Stages[stageName]
    if name in stage:
      if not name in events:
        events[name] = []
      event  = stage[name][0]
      etimes = [stage[name][p]["time"] for p in stage[name]]
      eflops = [stage[name][p]["flop"] for p in stage[name]]
      if "dof" in event:
        sizes.append(event["dof"][0])
        errors.append(event["error"][0])
      try:
        events[name].append((max(etimes), sum(eflops)/(max(etimes) * 1e6)))
      except ZeroDivisionError:
        events[name].append((max(etimes), 0))
  return

def plotTime(library, num, eventNames, sizes, times, events):
  from pylab import legend, plot, show, title, xlabel, ylabel
  import numpy as np

  arches = sizes.keys()
  data   = []
  for arch in arches:
    data.append(sizes[arch])
    data.append(times[arch])
  plot(*data)
  title('Performance on '+library+' Example '+str(num))
  xlabel('Number of Dof')
  ylabel('Time (s)')
  legend(arches, 'upper left', shadow = True)
  show()
  return

def plotEventTime(library, num, eventNames, sizes, times, events, filename = None):
  from pylab import close, legend, plot, savefig, show, title, xlabel, ylabel
  import numpy as np

  close()
  arches = sizes.keys()
  bs     = events[arches[0]].keys()[0]
  data   = []
  names  = []
  for event, color in zip(eventNames, ['b', 'g', 'r', 'y']):
    for arch, style in zip(arches, ['-', ':']):
      if event in events[arch][bs]:
        names.append(arch+'-'+str(bs)+' '+event)
        data.append(sizes[arch][bs])
        data.append(np.array(events[arch][bs][event])[:,0])
        data.append(color+style)
      else:
        print('Could not find %s in %s-%d events' % (event, arch, bs))
  print(data)
  plot(*data)
  title('Performance on '+library+' Example '+str(num))
  xlabel('Number of Dof')
  ylabel('Time (s)')
  legend(names, 'upper left', shadow = True)
  if filename is None:
    show()
  else:
    savefig(filename)
  return

def plotEventFlop(library, num, eventNames, sizes, times, events, filename = None):
  from pylab import legend, plot, savefig, semilogy, show, title, xlabel, ylabel
  import numpy as np

  arches = sizes.keys()
  bs     = events[arches[0]].keys()[0]
  data   = []
  names  = []
  for event, color in zip(eventNames, ['b', 'g', 'r', 'y']):
    for arch, style in zip(arches, ['-', ':']):
      if event in events[arch][bs]:
        names.append(arch+'-'+str(bs)+' '+event)
        data.append(sizes[arch][bs])
        data.append(1e-3*np.array(events[arch][bs][event])[:,1])
        data.append(color+style)
      else:
        print('Could not find %s in %s-%d events' % (event, arch, bs))
  semilogy(*data)
  title('Performance on '+library+' Example '+str(num))
  xlabel('Number of Dof')
  ylabel('Computation Rate (GF/s)')
  legend(names, 'upper left', shadow = True)
  if filename is None:
    show()
  else:
    savefig(filename)
  return

def plotEventScaling(library, num, eventNames, procs, events, filename = None):
  from pylab import legend, plot, savefig, semilogy, show, title, xlabel, ylabel
  import numpy as np

  arches = procs.keys()
  bs     = events[arches[0]].keys()[0]
  data   = []
  names  = []
  for arch, style in zip(arches, ['-', ':']):
    for event, color in zip(eventNames, ['b', 'g', 'r', 'y']):
      if event in events[arch][bs]:
        names.append(arch+'-'+str(bs)+' '+event)
        data.append(procs[arch][bs])
        data.append(1e-3*np.array(events[arch][bs][event])[:,1])
        data.append(color+style)
      else:
        print('Could not find %s in %s-%d events' % (event, arch, bs))
  plot(*data)
  title('Performance on '+library+' Example '+str(num))
  xlabel('Number of Processors')
  ylabel('Computation Rate (GF/s)')
  legend(names, 'upper left', shadow = True)
  if filename is None:
    show()
  else:
    savefig(filename)
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
    for event, color in zip(eventNames, ['b', 'g', 'r', 'y']):
      for arch, style in zip(arches, ['-', ':']):
        if event in events[arch]:
          names.append(arch+' '+event)
          data.append(sizes[arch])
          data.append(np.array(events[arch][event])[:,0])
          data.append(color+style)
        else:
          print('Could not find %s in %s events' % (event, arch))
    print(data)
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
    for event, color in zip(eventNames, ['b', 'g', 'r', 'y']):
      for arch, style in zip(arches, ['-', ':']):
        if event in events[arch]:
          names.append(arch+' '+event)
          data.append(sizes[arch])
          data.append(np.array(events[arch][event])[:,1])
          data.append(color+style)
        else:
          print('Could not find %s in %s events' % (event, arch))
    plot(*data)
    title('Performance on '+library+' Example '+str(num))
    xlabel('Number of Dof')
    ylabel('Computation Rate (MF/s)')
    legend(names, 'upper left', shadow = True)
    show()
  return

def plotMeshConvergence(library, num, eventNames, sizes, times, errors, events):
  import numpy as np
  import matplotlib.pyplot as plt
  data    = []
  legends = []
  print(sizes)
  print(errors)
  for run in sizes:
    rsizes = np.array(sizes[run])
    data.extend([rsizes, errors[run], rsizes, (errors[run][0]*rsizes[0]*2)*rsizes**(meshExp[run]/-2.0)])
    legends.extend(['Experiment '+run, r'Synthetic '+run+r' $\alpha = '+str(meshExp[run])+'$'])
  SizeError = plt.loglog(*data)
  plt.title(library+' ex'+str(num)+' Mesh Convergence')
  plt.xlabel('Size')
  plt.ylabel(r'Error $\|x - x^*\|_2$')
  plt.legend(legends)
  plt.show()
  return

def plotWorkPrecision(library, num, eventNames, sizes, times, errors, events):
  import numpy as np
  import matplotlib.pyplot as plt
  data    = []
  legends = []
  for run in times:
    rtimes = np.array(times[run])
    data.extend([rtimes, errors[run], rtimes, (errors[run][0]*rtimes[0]*2)*rtimes**(timeExp[run])])
    legends.extend(['Experiment '+run, 'Synthetic '+run+' exponent '+str(timeExp[run])])
  TimeError = plt.loglog(*data)
  plt.title(library+' ex'+str(num)+' Work Precision')
  plt.xlabel('Time (s)')
  plt.ylabel(r'Error $\|x - x^*\|_2$')
  plt.legend(legends)
  plt.show()
  return

def plotWorkPrecisionPareto(library, num, eventNames, sizes, times, errors, events):
  import numpy as np
  import matplotlib.pyplot as plt
  data    = []
  legends = []
  for run in times:
    rtimes = np.array(times[run])
    data.extend([rtimes, errors[run]])
    legends.append('Experiment '+run)
  TimeErrorPareto = plt.semilogy(*data)
  plt.title(library+' ex'+str(num)+' Work Precision: Pareto Front')
  plt.xlabel('Time (s)')
  plt.ylabel(r'Error $\|x - x^*\|_2$')
  plt.legend(legends)
  plt.show()
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

def processOptions(opts, name, n):
  newopts = {}
  for key, val in opts.items():
    val = opts[key]
    if val and val.find('%') >= 0:
      newval = val % (name.replace('/', '-'), n)
      newopts[key] = newval
    else:
      newopts[key] = val
  return newopts

def getLogName(opts):
  if 'log_view' in opts:
    val = opts['log_view']
    s   = val.find(':')
    e   = val.find(':', s+1)
    logName = os.path.splitext(val[s+1:e])[0]
    return logName
  return None

def run_DMDA(ex, name, opts, args, sizes, times, events, log=True, execute=True):
  for n in map(int, args.size):
    newopts = processOptions(opts, name, n)
    if execute:
      ex.run(log=log, da_grid_x=n, da_grid_y=n, **newopts)
    processSummary(getLogName(newopts), args.stage, args.events, sizes[name], times[name], errors[name], events[name])
  return

def run_DMPlex(ex, name, opts, args, sizes, times, events, log=True, execute=True):
  newopts = processOptions(opts, name, args.refine)
  if execute:
    ex.run(log=log, dim=args.dim, snes_convergence_estimate=None, convest_num_refine=args.refine, interpolate=1, **newopts)
  for r in range(args.refine+1):
    stage = args.stage
    if stage.find('%') >= 0: stage = stage % (r)
    processSummary(getLogName(newopts), stage, args.events, sizes[name], times[name], errors[name], events[name])
  return

def outputData(sizes, times, events, name = 'output.py'):
  if os.path.exists(name):
    base, ext = os.path.splitext(name)
    num = 1
    while os.path.exists(base+str(num)+ext):
      num += 1
    name = base+str(num)+ext
  with file(name, 'w') as f:
    f.write('#PETSC_ARCH='+os.environ['PETSC_ARCH']+' '+' '.join(sys.argv)+'\n')
    f.write('sizes  = '+repr(sizes)+'\n')
    f.write('times  = '+repr(times)+'\n')
    f.write('events = '+repr(events)+'\n')
  return

if __name__ == '__main__':
  import argparse
  import __main__

  parser = argparse.ArgumentParser(description     = 'PETSc Benchmarking',
                                   epilog          = 'This script runs src/<library>/tutorials/ex<num>, For more information, visit https://www.mcs.anl.gov/petsc',
                                   formatter_class = argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--library', default='SNES',                     help='The PETSc library used in this example')
  parser.add_argument('--num',     type = int, default='5',            help='The example number')
  parser.add_argument('--module',  default='summary',                  help='The module for timing output')
  parser.add_argument('--stage',   default='Main Stage',               help='The default logging stage')
  parser.add_argument('--events',  nargs='+',                          help='Events to process')
  parser.add_argument('--plotOnly',action='store_true', default=False, help='Flag to only plot existing data')
  parser.add_argument('--batch',   action='store_true', default=False, help='Generate batch files for the runs instead')
  parser.add_argument('--daemon',  action='store_true', default=False, help='Run as a daemon')
  parser.add_argument('--gpulang', default='OpenCL',                   help='GPU Language to use: Either CUDA or OpenCL (default)')
  parser.add_argument('--plots',   nargs='+',                          help='List of plots to show')
  subparsers = parser.add_subparsers(help='DM types')

  parser_dmda = subparsers.add_parser('DMDA', help='Use a DMDA for the problem geometry')
  parser_dmda.add_argument('--size', nargs='+',  default=['10'], help='Grid size (implementation dependent)')
  parser_dmda.add_argument('--comp', type = int, default='1',    help='Number of field components')
  parser_dmda.add_argument('runs',   nargs='*',                  help='Run descriptions: <name>=<args>')

  parser_dmmesh = subparsers.add_parser('DMPlex', help='Use a DMPlex for the problem geometry')
  parser_dmmesh.add_argument('--dim',      type = int, default='2',        help='Spatial dimension')
  parser_dmmesh.add_argument('--refine',   type = int, default='0',        help='Number of refinements')
  parser_dmmesh.add_argument('runs',       nargs='*',                      help='Run descriptions: <name>=<args>')

  args = parser.parse_args()
  if hasattr(args, 'comp'):
    args.dmType = 'DMDA'
  else:
    args.dmType = 'DMPlex'

  ex = PETScExample(args.library, args.num, preload='off')
  if args.gpulang == 'CUDA':
    source = ex.petsc.source(args.library, args.num, '.cu')
  else:
    source = ex.petsc.source(args.library, args.num, 'OpenCL.c')  # Using the convention of OpenCL code residing in source files ending in 'OpenCL.c' (at least for snes/ex52)
  sizes   = {}
  times   = {}
  errors  = {}
  meshExp = {}
  timeExp = {}
  events  = {}
  log     = not args.daemon

  if args.daemon:
    import daemon
    print('Starting daemon')
    daemon.createDaemon('.')

  for run in args.runs:
    name, stropts = run.split('=', 1)
    opts = dict([t if len(t) == 2 else (t[0], None) for t in [arg.split('=', 1) for arg in stropts.split(' ')]])
    #opts['log_view'] = 'summary.dat' if args.batch else ':'+args.module+'%s%d.py:ascii_info_detail'
    opts['log_view'] = 'summary.dat' if args.batch else ':'+args.module+'%s%d.csv:ascii_csv'
    meshExp[name] = float(opts['meshExp'])
    timeExp[name] = float(opts['timeExp'])
    sizes[name]   = []
    times[name]   = []
    errors[name]  = []
    events[name]  = {}
    getattr(__main__, 'run_'+args.dmType)(ex, name, opts, args, sizes, times, events, log=log, execute=(not args.plotOnly))
  outputData(sizes, times, events)
  if not args.batch and log:
    for plot in args.plots:
      print('Plotting ',plot)
      getattr(__main__, 'plot'+plot)(args.library, args.num, args.events, sizes, times, errors, events)

# ./src/benchmarks/benchmarkExample.py --events SNESSolve --plots MeshConvergence WorkPrecision WorkPrecisionPareto --num 5 DMDA --size 32 64 128 256 512 1024 --comp 1 GMRES/ILU0="snes_monitor par=0.0 ksp_rtol=1.0e-9 mms=2 ksp_type=gmres pc_type=ilu meshExp=2.0 timeExp=-0.5" GMRES/LU="snes_monitor par=0.0 ksp_rtol=1.0e-9 mms=2 ksp_type=gmres pc_type=lu meshExp=2.0 timeExp=-0.75" GMRES/GAMG="snes_monitor par=0.0 ksp_rtol=1.0e-9 mms=2 ksp_type=gmres pc_type=gamg meshExp=2.0 timeExp=-1.0"
# ./src/benchmarks/benchmarkExample.py --stage "ConvEst Refinement Level %d" --events SNESSolve "ConvEst Error" --plots MeshConvergence WorkPrecision WorkPrecisionPareto --num 13 DMPlex --refine 5 --dim 2 GMRES/ILU0="snes_monitor ksp_rtol=1.0e-9 ksp_type=gmres pc_type=ilu meshExp=2.0 timeExp=-0.75 dm_refine=4 potential_petscspace_order=1" GMRES/LU="snes_monitor ksp_rtol=1.0e-9 ksp_type=gmres pc_type=lu meshExp=2.0 timeExp=-0.75 dm_refine=4 potential_petscspace_order=1" GMRES/GAMG="snes_monitor ksp_rtol=1.0e-9 ksp_type=gmres pc_type=gamg meshExp=2.0 timeExp=-1.0 dm_refine=4 potential_petscspace_order=1"
# ./src/benchmarks/benchmarkExample.py --stage "ConvEst Refinement Level %d" --events SNESSolve "ConvEst Error" --plots MeshConvergence WorkPrecision WorkPrecisionPareto --num 13 DMPlex --refine 5 --dim 2 GMRES/ILU0="snes_monitor ksp_rtol=1.0e-9 ksp_type=gmres pc_type=ilu meshExp=3.0 timeExp=-0.75 dm_refine=3 potential_petscspace_order=2" GMRES/LU="snes_monitor ksp_rtol=1.0e-9 ksp_type=gmres pc_type=lu meshExp=3.0 timeExp=-0.75 dm_refine=3 potential_petscspace_order=2" GMRES/GAMG="snes_monitor ksp_rtol=1.0e-9 ksp_type=gmres pc_type=gamg meshExp=3.0 timeExp=-1.0 dm_refine=3 potential_petscspace_order=2"
# ./src/benchmarks/benchmarkExample.py --stage "ConvEst Refinement Level %d" --events SNESSolve "ConvEst Error" --plots MeshConvergence WorkPrecision WorkPrecisionPareto --num 13 DMPlex --refine 5 --dim 2 GMRES/ILU0="snes_monitor ksp_rtol=1.0e-9 ksp_type=gmres pc_type=ilu meshExp=3.0 timeExp=-0.75 dm_refine=3 potential_petscspace_order=2" GMRES/LU="snes_monitor ksp_rtol=1.0e-9 ksp_type=gmres pc_type=lu meshExp=3.0 timeExp=-0.75 dm_refine=3 potential_petscspace_order=2" GMRES/GAMG="snes_monitor ksp_rtol=1.0e-9 ksp_type=gmres pc_type=gamg meshExp=3.0 timeExp=-1.0 dm_refine=3 potential_petscspace_order=2"

# Old GPU benchmarks
# Benchmark for ex50
# ./src/benchmarks/benchmarkExample.py --events VecMDot VecMAXPY KSPGMRESOrthog MatMult VecCUSPCopyTo VecCUSPCopyFrom MatCUSPCopyTo --num 50 DMDA --size 10 20 50 100 --comp 4 CPU='pc_type=none mat_no_inode dm_vec_type=seq dm_mat_type=seqaij' GPU='pc_type=none mat_no_inode dm_vec_type=seqcusp dm_mat_type=seqaijcusp cusp_synchronize'
# Benchmark for ex52
# ./src/benchmarks/benchmarkExample.py --events IntegBatchCPU IntegBatchGPU IntegGPUOnly --num 52 DMPlex --refine 0.0625 0.00625 0.000625 0.0000625 --blockExp 4 --order=1 CPU='dm_view show_residual=0 compute_function batch' GPU='dm_view show_residual=0 compute_function batch gpu gpu_batches=8'
# ./src/benchmarks/benchmarkExample.py --events IntegBatchCPU IntegBatchGPU IntegGPUOnly --num 52 DMPlex --refine 0.0625 0.00625 0.000625 0.0000625 --blockExp 4 --order=1 --operator=elasticity CPU='dm_view op_type=elasticity show_residual=0 compute_function batch' GPU='dm_view op_type=elasticity show_residual=0 compute_function batch gpu gpu_batches=8'
# ./src/benchmarks/benchmarkExample.py --events IntegBatchCPU IntegBatchGPU IntegGPUOnly --num 52 DMPlex --dim=3 --refine 0.0625 0.00625 0.000625 0.0000625 --blockExp 4 --order=1 CPU='dim=3 dm_view show_residual=0 compute_function batch' GPU='dim=3 dm_view show_residual=0 compute_function batch gpu gpu_batches=8'
