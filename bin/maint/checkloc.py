#!/usr/bin/env python
#!/bin/env python
#
# check if LOC is set correctly in ALL makefiles. Script
# assumes - the dir the command invoked in is PETSC_DIR
#
import os
#

def processLOCDIR(arg,dirname,names):
  import commands
  petscdir = arg[0]
  if 'makefile' in names:
    mfile=os.path.join(dirname,'makefile')
    # exclude list
    if dirname.find('externalpackages') >=0 or mfile in [os.path.join(petscdir,'makefile'),os.path.join(petscdir,'python','makefile'),os.path.join(petscdir,'python','BuildSystem','makefile'),os.path.join(petscdir,'python','BuildSystem','docs','makefile'),os.path.join(petscdir,'projects','makefile')]:
      return
    try:
      fd=open(mfile,'r')
    except:
      print 'Error! canot open ' + mfile
      return
    buf = fd.read()
    locdir=''
    for line in buf.splitlines():
      if line.startswith('LOCDIR'):
        locdir = line
        break
    if locdir == '':
      print 'Missing LOCDIR in: ' + mfile
      return
    loc=locdir.split('=')[1].lstrip()
    if loc != loc.rstrip():
      print 'Extra space at the end of LOCDIR in: ' + mfile
    if loc == '' :
      print 'Missing value for LOCDIR in: ' + mfile
      return
    if loc[-1] != '/':
      print 'Missing / at the end: ' + mfile
    if (os.path.join(petscdir,loc,'makefile') != mfile):
      print 'Wrong Entry: '+ loc + ' in ' + mfile
  return

def main():
  petscdir = os.getcwd()
  os.path.walk(petscdir, processLOCDIR, [petscdir])
  return
#
# The classes in this file can also be used in other python-programs by using 'import'
#
if __name__ ==  '__main__': 
  import sys
  main()
