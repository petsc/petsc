#!/usr/bin/env python
import Numeric

def parseHeader(f):
  # Ignore first line
  f.readline()
  # Ignore second line
  f.readline()
  return

def parseArc(f, arcs):
  print 'Parsing arc'
  args = f.readline().split()
  if int(args[0]) < 0: return False
  numPoints = int(args[-1])
  arc = Numeric.zeros((numPoints*2,), 'd')
  for i in range(numPoints/2):
    arc[i*4:i*4+4] = map(float, f.readline().split())
    print arc[i*4:i*4+4]
  if numPoints%2:
    arc[(numPoints-1)*2:numPoints*2] = map(float, f.readline().split())
    print arc[(numPoints-1)*2:numPoints*2]
  arcs.append(Numeric.reshape(arc, (numPoints,2)))
  print Numeric.reshape(arc, (numPoints,2))
  return True

def writeVTKHeader(f):
  f.write('''\
# vtk DataFile Version 2.0
IL State Boundary
ASCII
DATASET POLYDATA
''')
  return

def writeVTKPoints(f, arcs):
  numPoints = 0
  for arc in arcs: numPoints += len(arc)
  f.write('POINTS %d double\n' % numPoints)
  for arc in arcs:
    for x,y in arc:
      f.write(str(x)+' '+str(y)+' 0.0\n')
  return

def writeVTKLines(f, arcs):
  numPoints = 0
  for arc in arcs: numPoints += len(arc)
  f.write('LINES %d %d\n' % (len(arcs), numPoints+len(arcs)))
  point = 0
  for arc in arcs:
    f.write(str(len(arc)))
    for i in range(len(arc)):
      f.write(' '+str(point))
      point += 1
    f.write('\n')
  return

def writeVTK(arcs, filename):
  import os
  basename = os.path.splitext(filename)[0]
  f = file(basename+'.vtk', 'w')
  writeVTKHeader(f)
  writeVTKPoints(f, arcs)
  writeVTKLines(f, arcs)
  f.close()
  return

def writePCICEVertices(f, arcs):
  numPoints = 0
  for arc in arcs: numPoints += len(arc)
  f.write(str(numPoints)+'\n')
  point = 0
  for arc in arcs:
    for x,y in arc:
      f.write('%7d % 12.5E % 12.5E\n' % (point, x, y))
      point += 1
  return

def writePCICEElements(f, arcs):
  numPoints = 0
  for arc in arcs: numPoints += len(arc)
  f.write(str(numPoints)+'\n')
  point = 0
  for arc in arcs:
    for i in range(len(arc)):
      f.write('%d %d %d\n' % (point, point, (point+1)%numPoints))
      point += 1
  return

def writePCICE(arcs, filename):
  import os
  newArcs = [arcs[0], arcs[1][-1:0:-1], arcs[3][-1:0:-1], arcs[4][-1:0:-1], arcs[2]]
  basename = os.path.splitext(filename)[0]
  f = file(basename+'.lcon', 'w')
  writePCICEElements(f, newArcs)
  f.close()
  f = file(basename+'.nodes', 'w')
  writePCICEVertices(f, newArcs)
  f.close()
  return

def run(filename):
  f = file(filename)
  arcs = []
  parseHeader(f)
  while parseArc(f, arcs): pass
  f.close()
  writeVTK(arcs, filename)
  writePCICE(arcs, filename)
  return

if __name__ == '__main__':
  import sys
  run(sys.argv[1])
