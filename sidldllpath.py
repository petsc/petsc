#!/usr/bin/env python

import user
import os
import sys
import project
import RDict
import string

def getSIDLDLLPath():
  if os.environ.has_key('SIDL_DLL_PATH'):
    SIDL_DLL_PATH = os.environ['SIDL_DLL_PATH'].split(';')
  else:  
    SIDL_DLL_PATH = [] 
  argsDB = RDict.RDict(parentDirectory = os.path.abspath(os.path.dirname(sys.modules['RDict'].__file__)))
  projects = argsDB['installedprojects']
  for p in projects:
    try:
      root = p.getRoot()+'/lib'
      if not root in SIDL_DLL_PATH:
        SIDL_DLL_PATH += [root]
    except: pass
  return string.join(SIDL_DLL_PATH,';')
    
if __name__ ==  '__main__':
  if len(sys.argv) > 1: sys.exit('Usage: sidldllpath.py')
  print getSIDLDLLPath()
  

