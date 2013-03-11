#!/usr/bin/env python
import user
import project
import RDict

import os
import sys

def getPythonPath():
  if 'PYTHONPATH' in os.environ:
    PYTHONPATH = filter(lambda p: len(p), os.environ['PYTHONPATH'].split(os.path.pathsep))
  else:  
    PYTHONPATH = [] 
  argsDB   = RDict.RDict(parentDirectory = os.path.abspath(os.path.dirname(sys.modules['RDict'].__file__)))
  projects = argsDB['installedprojects']
  for p in projects:
    try:
      root = p.getPythonPath()
      for r in root:
        if not r in PYTHONPATH:
          PYTHONPATH.append(r)
    except: pass
  return ':'.join(PYTHONPATH)
    
if __name__ ==  '__main__':
  if len(sys.argv) > 1: sys.exit('Usage: pythonpath.py')
  print getPythonPath()

