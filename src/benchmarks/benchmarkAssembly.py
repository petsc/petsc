#!/usr/bin/env python
import os
from benchmarkExample import PETScExample

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

def plotSummary(library, num, sizes, times, events):
  from pylab import legend, plot, show, title, xlabel, ylabel
  import numpy as np
  showEventTime  = True
  print events
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
  return

if __name__ == '__main__':
  library = 'KSP'
  num     = 4
  ex      = PETScExample(library, num, log_summary_python='summary.py', preload='off')
  if 1:
    sizes   = []
    times   = []
    events  = {}
    for n in [10, 20, 50, 100, 150, 200, 250, 300, 350]:
      ex.run(da_grid_x=n, da_grid_y=n, cusp_synchronize=1)
      sizes.append(n*n)
      processSummary('summary', times, events)
    plotSummary(library, num, sizes, times, events)
  else:
    times   = []
    sizes   = []
    for n in range(150, 1350, 100):
      sizes.append(n*n)
    baconostEvents = {'ElemAssembly': [(0.040919999999999998, 0.0), (0.1242, 0.0), (0.24410000000000001, 0.0), (0.374, 0.0), (0.56259999999999999, 0.0), (0.79049999999999998, 0.0), (1.0880000000000001, 0.0), (1.351, 0.0), (1.6930000000000001, 0.0), (2.0609999999999999, 0.0), (2.4820000000000002, 0.0), (3.0640000000000001, 0.0)], 'MatCUSPSetValBch': [(0.0123, 0.0), (0.023429999999999999, 0.0), (0.043540000000000002, 0.0), (0.06608, 0.0), (0.09579, 0.0), (0.12920000000000001, 0.0), (0.17169999999999999, 0.0), (0.2172, 0.0), (0.27179999999999999, 0.0), (0.48309999999999997, 0.0), (0.44180000000000003, 0.0), (0.51529999999999998, 0.0)]}
    plotSummary(library, num, sizes, times, baconostEvents)
