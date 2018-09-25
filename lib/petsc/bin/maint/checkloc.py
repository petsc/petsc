#!/usr/bin/env python
#!/bin/env python
#
# check if LOC is set correctly in ALL makefiles. Script
# assumes - the dir the command invoked in is PETSC_DIR
#
from __future__ import print_function
import os
#

def processLOCDIR(petscdir, dirpath, dirnames, filenames):
  if 'makefile' in filenames:
    mfile=os.path.join(dirpath,'makefile')
    # exclude list
    if dirpath.find('externalpackages') >=0 or mfile in [os.path.join(petscdir,'makefile'),os.path.join(petscdir,'python','makefile'),os.path.join(petscdir,'python','BuildSystem','makefile'),os.path.join(petscdir,'python','BuildSystem','docs','makefile'),os.path.join(petscdir,'projects','makefile')]:
      return
    try:
      fd=open(mfile,'r')
    except:
      print('Error! canot open ' + mfile)
      return
    buf = fd.read()
    locdir=''
    for line in buf.splitlines():
      if line.startswith('LOCDIR'):
        locdir = line
        break
    if locdir == '':
      print('Missing LOCDIR in: ' + mfile)
      return
    loc=locdir.split('=')[1].lstrip()
    if loc != loc.rstrip():
      print('Extra space at the end of LOCDIR in: ' + mfile)
    if loc == '' :
      print('Missing value for LOCDIR in: ' + mfile)
      return
    if loc[-1] != '/':
      print('Missing / at the end: ' + mfile)
    if (os.path.join(petscdir,loc,'makefile') != mfile):
      print('Wrong Entry: '+ loc + ' in ' + mfile)
  for skip in [n for n in dirnames if os.path.isdir(os.path.join(dirpath, n, 'lib', 'petsc', 'conf'))]:
    dirnames.remove(skip)
  return

def main():
  petscdir = os.getcwd()
  for dirpath, dirnames, filenames in os.walk(petscdir):
    processLOCDIR(petscdir, dirpath, dirnames, filenames)
  return
#
# The classes in this file can also be used in other python-programs by using 'import'
#
if __name__ ==  '__main__':
  import sys
  main()
