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
  for name in ['MatCUSPSetValBch', 'ElemAssembly']:
    if not name in events:
      events[name] = []
    events[name].append((m.Main_Stage.event[name].Time[0], m.Main_Stage.event[name].Flops[0]/(m.Main_Stage.event[name].Time[0] * 1e6)))
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
  sizes   = []
  times   = []
  events  = {}
  for n in [10, 20, 50, 100, 150, 200, 250, 300, 350]:
    ex.run(da_grid_x=n, da_grid_y=n, cusp_synchronize=1)
    sizes.append(n*n)
    processSummary('summary', times, events)
  plotSummary(library, num, sizes, times, events)
