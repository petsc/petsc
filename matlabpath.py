#!/usr/bin/env python

import os
import sys
import nargs
import project


def getMatlabPath():
  if os.environ.has_key('MATLABPATH'):
    MATLABPATH = os.environ['MATLABPATH']
  else:  
    MATLABPATH = '' 
  argsDB = nargs.ArgDict('ArgDict')
  projects = argsDB['installedprojects']
  for p in projects:
    try:
      root = p.getMatlabPath()
      for k in p.getPackages():
        if MATLABPATH:
          MATLABPATH += ':' + root+'/'+k
        else:
          MATLABPATH = root+'/'+k
      if p.getUrl() == 'bk://sidl.bkbits.net/Runtime':
        MATLABPATH += ':' + p.getRoot() + '/matlab'
    except: pass
  return MATLABPATH  
    
if __name__ ==  '__main__':
  if len(sys.argv) > 1: sys.exit('Usage: matlabpath.py')
  print getMatlabPath()
  

