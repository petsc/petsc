#!/usr/bin/env python
import user
import project
import RDict

import os
import sys

def getSIDLDLLPath():
  if 'SIDL_DLL_PATH' in os.environ:
    SIDL_DLL_PATH = os.environ['SIDL_DLL_PATH'].split(';')
  else:  
    SIDL_DLL_PATH = [] 
  argsDB   = RDict.RDict(parentDirectory = os.path.abspath(os.path.dirname(sys.modules['RDict'].__file__)))
  projects = argsDB['installedprojects']
  for p in projects:
    try:
      root = os.path.join(p.getRoot(), 'lib')
      if not root in SIDL_DLL_PATH:
        SIDL_DLL_PATH.append(root)
    except: pass
  return ';'.join(SIDL_DLL_PATH)
    
if __name__ ==  '__main__':
  if len(sys.argv) > 1: sys.exit('Usage: sidldllpath.py')
  print getSIDLDLLPath()

