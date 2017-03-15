#!/usr/bin/env python
#
#    Reads in the output trajectory data from a run of extchem and
#  formats it for graphing
#
#  Make sure $PETSC_DIR/bin/petsc-pythonscripts is in your PYTHONPATH
#
import sys
import PetscBinaryIO
import numpy as np
import matplotlib.pyplot as pyplot

def ReadTrajectory(file):
  io = PetscBinaryIO.PetscBinaryIO()
  fh = open(file);

  v = []
  t = []
  flg = True

  while flg:
    try:
      objecttype = io.readObjectType(fh)
      v.append(io.readVec(fh))
      t.append(np.fromfile(fh, dtype=io._scalartype, count=1)[0])
    except:
      names = []
      nstrings = np.fromfile(fh, dtype=io._inttype, count=1)[0]
      sizes = np.fromfile(fh, dtype=io._inttype, count=nstrings)
      for i in range(0,nstrings):
        s = np.fromfile(fh, dtype=np.byte, count=sizes[i])
        names.append("".join(map(chr, s))[0:-1])
      flg = False

  return (t,v,names)

def PlotTrajectories(t,v,names,subnames):
  print names
  sub = []
  for s in subnames:
    sub.append(names.index(s))
  w = []
  for i in v:
    w.append(i[sub])

  pyplot.plot(t,w)
  pyplot.legend(subnames)
  pyplot.show()



if __name__ ==  '__main__':
  (t,v,names) = ReadTrajectory(sys.argv[1])
#  PlotTrajectories(t,v,names,['Temp','CO','CO2','H2O','H2','O2','CH4','C2H2','N2'])
#  for i in range(0,len(t)-1):
#    print t[i],v[i][names.index('Temp')],v[i][names.index('CH4')],v[i][names.index('H2')],v[i][names.index('CO')],v[i][names.index('CO2')],v[i][names.index('O')],v[i][names.index('OH')]
  for i in range(0,len(t)-1):
    print t[i],v[i][names.index('Temp')],v[i][names.index('CH4')],v[i][names.index('O2')],v[i][names.index('N2')],v[i][names.index('CO')],v[i][names.index('CO2')],v[i][names.index('O')],v[i][names.index('OH')],v[i][names.index('H2O')]


