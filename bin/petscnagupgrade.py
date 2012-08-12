#!/usr/bin/env python
#!/bin/env python
#
#    Nags the user to update to the latest version
#
import os
import os.path, time,sys
import re

def naggedtoday(file):
  if not os.path.exists(file): return 0
  if time.time() - os.path.getmtime(file) > 60*60*24: return 0
  return 1

def currentversion(petscdir):
  try:
    fd  = open(os.path.join(petscdir, 'include', 'petscversion.h'))
    pv = fd.read()
    fd.close()
    majorversion = int(re.compile(' PETSC_VERSION_MAJOR[ ]*([0-9]*)').search(pv).group(1))
    minorversion = int(re.compile(' PETSC_VERSION_MINOR[ ]*([0-9]*)').search(pv).group(1))
    patchversion = int(re.compile(' PETSC_VERSION_PATCH[ ]*([0-9]*)').search(pv).group(1))
  except:
    return 
  version=str(majorversion)+'.'+str(minorversion)+'.'+str(patchversion)
  try:
    import urllib
    fd = urllib.urlopen("http://www.mcs.anl.gov/petsc/petsc-dev/include/petscversion.h")
    pv = fd.read()
    fd.close()
    amajorversion = int(re.compile(' PETSC_VERSION_MAJOR[ ]*([0-9]*)').search(pv).group(1))
    aminorversion = int(re.compile(' PETSC_VERSION_MINOR[ ]*([0-9]*)').search(pv).group(1))
    apatchversion = int(re.compile(' PETSC_VERSION_PATCH[ ]*([0-9]*)').search(pv).group(1))
  except:
    return 
  aversion = str(amajorversion)+'.'+str(aminorversion)+'.'+str(apatchversion)
  if (amajorversion > majorversion) or ((amajorversion ==  majorversion) and (aminorversion > minorversion)) or \
  ((amajorversion ==  majorversion) and (aminorversion == minorversion) and (apatchversion > patchversion)):
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("The version of PETSc you are using is out-of-date, we recommend updating to the new release")
    print(" Available Version: "+aversion+"   Installed Version: "+version)
    print("http://www.mcs.anl.gov/petsc/download/index.html")
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
  try:
    fd = open(os.path.join(petscdir,'.nagged'),"w")
    fd.close()
  except:
    return

  return 0
#
#
if __name__ ==  '__main__': 
  if os.environ.has_key('PETSC_DIR'):
    petscdir = os.environ['PETSC_DIR']
  elif os.path.exists(os.path.join('.', 'include', 'petscversion.h')):
    petscdir  = '.'
  else:
    sys.exit(0)
  file     = os.path.join(petscdir,'.nagged')
  if not naggedtoday(file):
    currentversion(petscdir)


