#!/usr/bin/env python
from __future__ import print_function
import os
from benchmarkExample import PETScExample

savedTiming = {'baconost': {'ElemAssembly': [(0.040919999999999998, 0.0), (0.1242, 0.0), (0.24410000000000001, 0.0), (0.374, 0.0), (0.56259999999999999, 0.0), (0.79049999999999998, 0.0), (1.0880000000000001, 0.0), (1.351, 0.0), (1.6930000000000001, 0.0), (2.0609999999999999, 0.0), (2.4820000000000002, 0.0), (3.0640000000000001, 0.0)],
                            'MatCUSPSetValBch': [(0.0123, 0.0), (0.023429999999999999, 0.0), (0.043540000000000002, 0.0), (0.06608, 0.0), (0.09579, 0.0), (0.12920000000000001, 0.0), (0.17169999999999999, 0.0), (0.2172, 0.0), (0.27179999999999999, 0.0), (0.48309999999999997, 0.0), (0.44180000000000003, 0.0), (0.51529999999999998, 0.0)]}
               }

def calculateNonzeros(n):
  num = 0
  # corners
  num += 2*3 + 2*4
  # edges
  num += 4*(n-2)*5
  # interior
  num += (n-2)*(n-2)*7
  return num

def processSummary(moduleName, times, events):
  '''Process the Python log summary into plot data'''
  m = __import__(moduleName)
  reload(m)
  # Total Time
  times.append(m.Time[0])
  # Common events
  #   Add the time and flop rate
  for stageName, eventName in [('GPU_Stage','MatCUSPSetValBch'), ('CPU_Stage','ElemAssembly')]:
    s = getattr(m, stageName)
    if not eventName in events:
      events[eventName] = []
    events[eventName].append((s.event[eventName].Time[0], s.event[eventName].Flops[0]/(s.event[eventName].Time[0] * 1e6)))
  return

def plotSummary(library, num, sizes, nonzeros, times, events):
  from pylab import legend, plot, show, title, xlabel, ylabel, ylim
  import numpy as np
  showEventTime      = True
  showTimePerRow     = False
  showTimePerNonzero = True
  print(events)
  if showEventTime:
    data  = []
    names = []
    for event, style in [('MatCUSPSetValBch', 'b-'), ('ElemAssembly', 'b:')]:
      names.append(event)
      data.append(sizes)
      data.append(np.array(events[event])[:,0])
      data.append(style)
    plot(*data)
    title('Performance on '+library+' Example '+str(num))
    xlabel('Number of Dof')
    ylabel('Time (s)')
    legend(names, 'upper left', shadow = True)
    show()
  if showTimePerRow:
    data  = []
    names = []
    for event, style in [('MatCUSPSetValBch', 'b-'), ('ElemAssembly', 'b:')]:
      names.append(event)
      data.append(sizes)
      rows = np.sqrt(sizes)
      data.append(np.array(events[event])[:,0]/rows/3)
      data.append(style)
    plot(*data)
    title('Performance on '+library+' Example '+str(num))
    xlabel('Number of Dof')
    ylabel('Time/Row (s)')
    legend(names, 'upper left', shadow = True)
    show()
  if showTimePerNonzero:
    data  = []
    names = []
    for event, style in [('MatCUSPSetValBch', 'b-'), ('ElemAssembly', 'b:')]:
      names.append(event)
      data.append(sizes)
      data.append(np.array(events[event])[:,0]/nonzeros * 10**9)
      data.append(style)
    plot(*data)
    title('Performance on '+library+' Example '+str(num))
    xlabel('Number of Dof')
    ylabel('Time/Nonzero (ns)')
    legend(names, 'center right', shadow = True)
    show()
  return

if __name__ == '__main__':
  import argparse

  parser = argparse.ArgumentParser(description     = 'PETSc Benchmarking',
                                   epilog          = 'This script runs src/<library>/tutorials/ex<num>, For more information, visit https://petsc.org/',
                                   formatter_class = argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--library', default='SNES',          help='The PETSc library used in this example')
  parser.add_argument('--num',     type = int, default='5', help='The example number')
  parser.add_argument('--module',  default='summary',       help='The module for timing output')
  parser.add_argument('--saved',                            help='Name of saved data')
  parser.add_argument('--scaling',                          help='Run parallel scaling test')
  parser.add_argument('--small',   action='store_true', default=False, help='Use small sizes')
  parser.add_argument('--batch',   action='store_true', default=False, help='Generate batch files for the runs instead')

  args = parser.parse_args()
  print(args)
  ex       = PETScExample(args.library, args.num, log_summary_python = None if args.batch else args.module+'.py', preload='off')
  sizes    = []
  nonzeros = []
  times    = []
  if args.saved is None:
    events   = {}
    if args.scaling == 'strong':
      procs  = [1, 2, 4, 8]
      if args.small:
        grid = [10]*len(procs)
      else:
        grid = [1250]*len(procs)
    else:
      if args.small:
        grid = [100, 150, 200, 250, 300]
      else:
        grid = range(150, 1350, 100)
      procs  = [1]*len(grid)
    for n, p in zip(grid, procs):
      ex.run(p, da_grid_x=n, da_grid_y=n, cusp_synchronize=1, batch=args.batch)
      sizes.append(n*n)
      nonzeros.append(calculateNonzeros(n))
      if not args.batch:
        processSummary(args.module, times, events)
        os.remove(args.module+'.pyc')
  else:
    if args.batch: raise RuntimeException('Cannot use batch option with saved data')
    if args.saved in savedTiming:
      events = savedTiming[args.saved]
    else:
      # Process output to produce module
      events       = {}
      filenameBase = args.saved[:-7]
      jobnumBase   = int(args.saved[-7:])
      for i, n in enumerate(range(150, 1350, 100)):
        filename = filenameBase+str(jobnumBase+i)
        print('Processing',filename)
        headerSeen = False
        with file(filename) as f, file(args.module+'.py', 'w') as o:
          for line in f.readlines():
            if not headerSeen:
              if not line[0] == '#': continue
              headerSeen = True
            if line[0] == '#' and line[-6:] == '=====\n': break
            o.write(line)
            #print line
        processSummary(args.module, times, events)
        # I can't believe that this is necessary
        os.remove(args.module+'.pyc')
    for n in range(150, 1350, 100):
      sizes.append(n*n)
      nonzeros.append(calculateNonzeros(n))
  if not args.batch: plotSummary(args.library, args.num, sizes, nonzeros, times, events)
