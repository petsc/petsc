#!/usr/bin/env python

import os
import os.path
import sys
import RDict
import project
import string

def getMatlabPath():
  if os.environ.has_key('MATLABPATH'):
    MATLABPATH = os.environ['MATLABPATH'].split(':')
  else:  
    MATLABPATH = []
  argsDB = RDict.RDict(parentDirectory = os.path.abspath(os.path.dirname(sys.modules['RDict'].__file__)))
  for p in argsDB['MATLABCLIENTPATH']:
    if os.path.exists(p) and (not p in MATLABPATH):
          MATLABPATH += [p]

  for p in argsDB['MATLABSERVERPATH']:
    if os.path.exists(p) and (not p in MATLABPATH):
          MATLABPATH += [p]

  for p in argsDB['installedprojects']:
    if p.getUrl() == 'bk://sidl.bkbits.net/Runtime':
      k =  p.getRoot() + '/matlab'
      if not k in MATLABPATH:
        MATLABPATH += [k]

  return string.join(MATLABPATH,':')  
    
if __name__ ==  '__main__':
  if len(sys.argv) > 1: sys.exit('Usage: matlabpath.py')
  print getMatlabPath()
  

