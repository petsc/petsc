#!/usr/bin/env python
import os
import re
import exceptions
import sys

def getPETScDirectory():
  '''Checks PETSC_DIR and sets if not set'''
  if 'PETSC_DIR' in os.environ:
    return os.environ['PETSC_DIR']
  else:
    return os.path.realpath(os.path.dirname(os.path.dirname(sys.argv[0])))


def isGNUPatch(patch):
  '''Returns 1 if it is GNU patch or equivilent, exception if cannot run'''
  if sys.platform.startswith('sunos') or sys.platform.startswith('alpha'):
    try:
      output = commands.getoutputstatus(patch+' --help')
    except:
      raise RuntimeError('Unable to run '+patch+' command')
    if output.find('gnu.org') == -1: return 0
  return 1


  # should only apply patch if it truly has something new in it. Keep checksum somewhere?
def updatePatches():
  '''Updates the source code from any available patches'''
  if len(sys.argv) > 1 and sys.argv[1].startswith('--patch='):
    patch = sys.argv[1][8:]
    try:
      if not isGNUPatch(patch):
        raise RuntimeError('Patch program provided with --patch='+patch+' must be gnu patch')
    except:
      raise RuntimeError('Cannot run patch program provided with --patch='+patch)
  else:
    patch = 'patch'
    if not isGNUPatch(patch):
      raise RuntimeError('Solaris and Alpha require GNU patch, run with --with-patch=<full path of gnu patch> \n')
    
  # Get PETSc current version number
  dir = getPETScDirectory()
  print dir
  if not os.path.exists(os.path.join(dir, 'include', 'petscversion.h')):
    raise RuntimeError('Invalid PETSc directory '+str(dir)+' it may not exist?')
  fd  = open(os.path.join(dir, 'include', 'petscversion.h'))
  pv = fd.read()
  fd.close()
  import re
  try:
    majorversion    = re.compile(' PETSC_VERSION_MAJOR[ ]*([0-9]*)').search(pv).group(1)
    minorversion    = re.compile(' PETSC_VERSION_MINOR[ ]*([0-9]*)').search(pv).group(1)
    subminorversion = re.compile(' PETSC_VERSION_SUBMINOR[ ]*([0-9]*)').search(pv).group(1)
  except:
    raise RuntimeError('Unable to find version information from include/petscversion.h\nYour PETSc files are likely incomplete, get the PETSc code again')
  version=str(majorversion)+'.'+str(minorversion)+'.'+str(subminorversion)
    
  sys.stdout.write('Downloading latest patches for version '+version+'\n')
  patchfile1 =  'ftp://ftp.mcs.anl.gov/pub/petsc/patches/petsc_patch_all-'+version
  patchfile2 =  'ftp://ftp.mcs.anl.gov/pub/petsc/patches/buildsystem_patch_all-'+version
  import urllib
  try:
    urllib.urlretrieve(patchfile1, 'patches1')
    urllib.urlretrieve(patchfile2, 'patches2')
  except:
    raise RuntimeError('Unable to download patches. Perhaps you are off the network?\n')

  try:
    (output1,status1) = commands.getoutputstatus('echo '+patch+' -Np1 < '+patchfile1)
  except:
    raise RuntimeError('Unable to apply patch from '+patchfile1+'with '+patch) 
  if output1.find('error') >= 0:
    sys.stdout.write(output1+'\n')
    raise RuntimeError('Error applying '+patchfile1+' update.\n')

  try:
    (output1,status1) = commands.getoutputstatus('cd python/BuildSystem; echo '+patch+' -Np1 < '+patchfile2)
  except:
    raise RuntimeError('Unable to apply patch from '+patchfile2+'with '+patch) 
  if output1.find('error') >= 0:
    sys.stdout.write(output1+'\n')
    raise RuntimeError('Error applying '+patchfile2+' update.\n')

if __name__ ==  '__main__': 
  updatePatches()
