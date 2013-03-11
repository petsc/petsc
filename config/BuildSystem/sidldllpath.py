#!/usr/bin/env python
import user
import project
import RDict

import os
import sys

def getSIDLDLLPath():
  if 'SIDL_DLL_PATH' in os.environ:
    SIDL_DLL_PATH = filter(lambda p: len(p), os.environ['SIDL_DLL_PATH'].split(';'))
  else:  
    SIDL_DLL_PATH = [] 
  argDB    = RDict.RDict(parentDirectory = os.path.abspath(os.path.dirname(sys.modules['RDict'].__file__)))
  projects = argDB['installedprojects']
  for p in projects:
    try:
      root = os.path.join(p.getRoot(), 'lib')
      if not root in SIDL_DLL_PATH:
        SIDL_DLL_PATH.append(root)
    except: pass
  return ';'.join(SIDL_DLL_PATH)

def getSIDLDLLMap():
  argDB    = RDict.RDict(parentDirectory = os.path.abspath(os.path.dirname(sys.modules['RDict'].__file__)))
  projects = argDB['installedprojects']
  dllMap   = {}
  for p in projects:
    impls = p.getImplementations()
    for cls in impls:
      dllMap[cls] = impls[cls][0][0]
  return dllMap

if __name__ ==  '__main__':
  if len(sys.argv) > 2: sys.exit('Usage: sidldllpath.py [path | map]')
  if len(sys.argv) == 1 or sys.argv[1] == 'path':
    print getSIDLDLLPath()
  elif sys.argv[1] == 'map':
    print getSIDLDLLMap()
