#!/usr/bin/env python

import os
import sys
import nargs
import project


def getSIDLDLLPath():
  if os.environ.has_key('SIDL_DLL_PATH'):
    SIDL_DLL_PATH = os.environ['SIDL_DLL_PATH']
  else:  
    SIDL_DLL_PATH = '' 
  argsDB = nargs.ArgDict('ArgDict')
  projects = argsDB['installedprojects']
  for p in projects:
    try:
      root = p.getRoot()
      if SIDL_DLL_PATH:
        SIDL_DLL_PATH += ';' + root+'/lib'
      else:
        SIDL_DLL_PATH = root+'/lib'
    except: pass
  return SIDL_DLL_PATH  
    
if __name__ ==  '__main__':
  if len(sys.argv) > 1: sys.exit('Usage: sidldllpath.py')
  print getSIDLDLLPath()
  

