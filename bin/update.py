#!/usr/bin/env python
import commands
import os
import re
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
      output = commands.getoutput(patch+' --help')
      log.write(output+'\n')
    except:
      raise RuntimeError('Unable to run '+patch+' command')
    if output.find('gnu.org') == -1: return 0
  return 1

# should only apply patch if it truly has something new in it. Keep checksum somewhere?
def updatePatches():
  '''Updates the source code from any available patches'''
  log = open('patches.log','w')
  patch = 'patch'
  for i in range(1,len(sys.argv)):
    if sys.argv[i].startswith('--patch='):
      patch = sys.argv[i][8:]
      sys.stdout.write('Using '+patch+' program to apply patches\n')
      log.write('Using '+patch+' program to apply patches\n')      
      try:
        if not isGNUPatch(patch):
          raise RuntimeError('Patch program provided with --patch='+patch+' must be gnu patch')
      except:
        raise RuntimeError('Cannot run patch program provided with --patch='+patch)
  if not isGNUPatch(patch):
    raise RuntimeError('Solaris and Alpha require GNU patch, run with --patch=<full path of gnu patch> \n')
    
  # Get PETSc current version number
  dir = getPETScDirectory()
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
    
  patches1   = 'patches1'
  patchfile1 =  'http://ftp.mcs.anl.gov/pub/petsc/patches/petsc_patch_all-'+version
  for i in range(1,len(sys.argv)):
    if sys.argv[i].startswith('--patch1='):
      patches1 = sys.argv[i][9:]
      if patches1.startswith('ftp://'):
        patchfile1 = patches1
        patches1   = 'patches1'
  if patches1 == 'patches1':
    sys.stdout.write('Downloading patches '+patchfile1+' for PETSc version '+version+'\n')
    log.write('Downloading patches '+patchfile1+' for PETSc version '+version+'\n')    

    patches1   = 'patches1'
    import urllib
    try:
      urllib.urlretrieve(patchfile1, patches1)
    except Exception, e:
      raise RuntimeError('Unable to download patches. Perhaps you are off the network?\n  '+str(e))
  else:
    log.write('Using '+patches1+' for PETSc patches\n')

  patches2   = 'patches2'
  patchfile2 =  'http://ftp.mcs.anl.gov/pub/petsc/patches/buildsystem_patch_all-'+version
  for i in range(1,len(sys.argv)):
    if sys.argv[i].startswith('--patch2='):
      patches2 = sys.argv[i][9:]
      if patches2.startswith('ftp://'):
        patchfile2 = patches2
        patches2   = 'patches2'
  if patches2 == 'patches2':
    sys.stdout.write('Downloading patches '+patchfile2+' for PETSc version '+version+' BuildSystem\n')
    log.write('Downloading patches '+patchfile2+' for PETSc version '+version+' BuildSystem\n')    
    import urllib
    try:
      urllib.urlretrieve(patchfile2, patches2)
    except:
      raise RuntimeError('Unable to download patches. Perhaps you are off the network?\n')
  else:
    log.write('Using '+patches2+' for PETSc BuildSystem patches\n')

  try:
    (status1,output1) = commands.getstatusoutput(patch+' -Np1 < '+patches1)
  except:
    raise RuntimeError('Unable to apply patch from '+patches1+' with '+patch) 
  log.write(output1+'\n')
  if patches1 == 'patches1':
    os.unlink(patches1)
    
  try:
    (status1,output1) = commands.getstatusoutput('cd python/BuildSystem; '+patch+' -Np1 < '+os.path.join('..','..',patches2))
  except:
    raise RuntimeError('Unable to apply patch from '+patches2+' with '+patch) 
  log.write(output1+'\n')
  if patches2 == 'patches2':
    os.unlink(patches2)

  sys.stdout.write('Applied patches for version '+version+'\n')
  log.write('Applied patches for version '+version+'\n')
  log.close()
 
if __name__ ==  '__main__': 
  updatePatches()
