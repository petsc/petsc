#!/usr/bin/env python
import user
import importer

import os
import sys
import SIDL.Loader

def checkDLLLibs():
  dirs = SIDL.Loader.getSearchPath()
  dirs = dirs.split(';')
  for dir in dirs:
    if os.path.isdir(dir):
      for f in os.listdir(dir):
        if f and os.path.splitext(f)[1] == '.so':
          print 'Loading '+os.path.join(dir, f)
          SIDL.Loader.loadLibrary(os.path.join(dir, f))

    
if __name__ ==  '__main__':
  if len(sys.argv) > 1: sys.exit('Usage: checkdlllibs.py')
  checkDLLLibs()

