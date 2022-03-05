#!/usr/bin/env python
#!/bin/env python
#
#    Nags the user to update to the latest version
#
from __future__ import print_function
import os
import os.path, time,sys
import re
try:
  from packaging.version import Version
except ImportError:
  try:
    from distutils.version import LooseVersion as Version
  except ImportError:
    sys.exit()

def naggedtoday(file):
  if not os.path.exists(file): return 0
  if time.time() - os.path.getmtime(file) > 60*60*24: return 0
  return 1

def parse_version_h(pv):
  release  = int(re.compile(' PETSC_VERSION_RELEASE[ ]*([0-9]*)').search(pv).group(1))
  major    = int(re.compile(' PETSC_VERSION_MAJOR[ ]*([0-9]*)').search(pv).group(1))
  minor    = int(re.compile(' PETSC_VERSION_MINOR[ ]*([0-9]*)').search(pv).group(1))
  subminor = int(re.compile(' PETSC_VERSION_SUBMINOR[ ]*([0-9]*)').search(pv).group(1))
  if release:
    return Version('%d.%d.%d' % (major, minor, subminor))
  else:
    return Version('%d.%d.0rc0' % (major,minor+1))

def currentversion(petscdir):
  try:
    with open(os.path.join(petscdir, 'include', 'petscversion.h')) as fd:
      pv = fd.read()
    version = parse_version_h(pv)
  except:
    return
  try:
    try:
      from urllib.request import urlopen
    except ImportError:
      from urllib2 import urlopen
    # with context manager not support in Python-2; would be preferred in Python-3
    fd = urlopen("https://gitlab.com/petsc/petsc/raw/release/include/petscversion.h",timeout = 2)
    pv = fd.read().decode('utf-8')
    fd.close()
    aversion = parse_version_h(pv)
  except:
    return
  if aversion > version:
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("The version of PETSc you are using is out-of-date, we recommend updating to the new release")
    print(" Available Version: "+str(aversion)+"   Installed Version: "+str(version))
    print("https://petsc.org/release/download/")
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
  if 'PETSC_DIR' in os.environ:
    petscdir = os.environ['PETSC_DIR']
  elif os.path.exists(os.path.join('.', 'include', 'petscversion.h')):
    petscdir  = '.'
  else:
    sys.exit(0)
  file     = os.path.join(petscdir,'.nagged')
  if not naggedtoday(file):
    currentversion(petscdir)
