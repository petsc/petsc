#!/usr/bin/env python

import os
import sys
import nargs
import project
import string

def getMatlabPath():
  if os.environ.has_key('MATLABPATH'):
    MATLABPATH = os.environ['MATLABPATH'].split(':')
  else:  
    MATLABPATH = []
  argsDB = nargs.ArgDict('ArgDict')
  projects = argsDB['installedprojects']
  for p in projects:
    try:
      root = p.getMatlabPath()
      for k in p.getPackages():
        k = root+'/'+k
        if not k in MATLABPATH:
          MATLABPATH += [k]
      if p.getUrl() == 'bk://sidl.bkbits.net/Runtime':
        k =  p.getRoot() + '/matlab'
        if not k in MATLABPATH:
          MATLABPATH += [k]
    except: pass
  return string.join(MATLABPATH,':')  
    
if __name__ ==  '__main__':
  if len(sys.argv) > 1: sys.exit('Usage: matlabpath.py')
  print getMatlabPath()
  

