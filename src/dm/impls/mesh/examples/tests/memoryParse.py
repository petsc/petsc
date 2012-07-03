#!/usr/bin/env python

def parse(filename, sizeHist, classHist):
  import re
  exp = re.compile(r'\[0\]Allocating (?P<num>\d+) bytes for class (?P<class>.*)$')
  f = open(filename)
  for line in f.readlines():
    m = exp.match(line)
    if m:
      size = int(m.group('num'))
      if size in sizeHist:
        sizeHist[size] += 1
      else:
        sizeHist[size]  = 1
      className = m.group('class')
      if className in classHist:
        classHist[className] += 1
      else:
        classHist[className]  = 1
  f.close()
  return

if __name__ == '__main__':
  import sys
  if len(sys.argv) < 2:
    sys.exit('Usage: memoryParse.py <filename>')
  sizeHist = {}
  classHist = {}
  parse(sys.argv[1], sizeHist, classHist)
  sizeOutput = []
  for (size, num) in sizeHist.iteritems():
    if num > 50: sizeOutput.append((size, num))
  sizeOutput.sort()
  print sizeOutput
  classOutput = []
  for (className, num) in classHist.iteritems():
    if num > 50: classOutput.append((className, num))
  classOutput.sort()
  print classOutput
