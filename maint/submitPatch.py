#!/usr/bin/env python
import commands
import os
import re
import sys
import time

class Patch (object):
  def __init__(self, clArgs = None):
    import bs.nargs
    self.argDB    = self.setupArgDB(clArgs)
    self.root     = os.getcwd()
    self.patchDir = os.path.join('/mcs', 'ftp', 'pub', 'petsc', 'patches')
    self.checkPetscRoot(self.root)
    return

  def setupArgDB(self, clArgs):
    import bs.nargs
    argDB = bs.nargs.ArgDict('ArgDict', localDict = 1)
    # Actions
    argDB.setLocalType('submit', bs.nargs.ArgBool('Construct and submit a patch for Petsc'))
    argDB.setLocalType('doVersion', bs.nargs.ArgBool('Update petscversion.h'))
    argDB.setLocalType('doPush', bs.nargs.ArgBool('Push changes'))
    argDB.setLocalType('makePatch', bs.nargs.ArgBool('Construct the patch for Petsc'))
    argDB.setLocalType('makeMasterPatch', bs.nargs.ArgBool('Construct the master patch for Petsc'))
    argDB.setLocalType('integrate', bs.nargs.ArgBool('Integrate changes into the Petsc development repository'))
    # Variables
    argDB.setLocalType('version', bs.nargs.ArgString('The version number, e.g. 2.1.0'))

    argDB.insertArgList(clArgs)
    return argDB

  def defaultCheckCommand(self, command, status, output):
    if status: raise RuntimeError('Could not execute\''+command+'\': '+output)
    return

  # Should use Maker.executeShellCommand() here
  def executeShellCommand(self, command, checkCommand = None):
    print 'Executing: '+command
    (status, output) = commands.getstatusoutput(command)
    if checkCommand:
      checkCommand(command, status, output)
    else:
      self.defaultCheckCommand(command, status, output)
    return output

  def checkPetscRoot(self, root):
    '''Find the root of the Petsc tree. Currently, we require it to be os.getcwd()'''
    if not os.path.isfile(os.path.join(root, 'include', 'petsc.h')):
      raise RuntimeError('Directory '+root+' is not the root of a Petsc tree')
    return

  def setVersion(self):
    '''If petscversion.h has not been edited, increment the patch level and checkin'''
    filename = os.path.join(self.root, 'include', 'petscversion.h')
    output   = self.executeShellCommand('bk sfiles -lg '+filename)
    if output.find(filename) >= 0: return
    self.executeShellCommand('bk edit '+filename)
    majorRE    = re.compile(r'^#define PETSC_VERSION_MAJOR([\s]+)(?P<versionNum>\d+)[\s]*$');
    minorRE    = re.compile(r'^#define PETSC_VERSION_MINOR([\s]+)(?P<versionNum>\d+)[\s]*$');
    subminorRE = re.compile(r'^#define PETSC_VERSION_SUBMINOR([\s]+)(?P<versionNum>\d+)[\s]*$');
    patchRE    = re.compile(r'^#define PETSC_VERSION_PATCH([\s]+)(?P<patchNum>\d+)[\s]*$');
    dateRE     = re.compile(r'^#define PETSC_VERSION_DATE([\s]+)"(?P<date>(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) \d\d?, \d\d\d\d)"[\s]*$');
    input   = file(filename)
    lines   = []
    for line in input.readlines():
      m1 = majorRE.match(line)
      m2 = minorRE.match(line)
      m3 = subminorRE.match(line)
      m4 = patchRE.match(line)
      m5 = dateRE.match(line)
      if m1:
        majorNum = int(m1.group('versionNum'))
      elif m2:
        minorNum = int(m2.group('versionNum'))
      elif m3:
        subminorNum = int(m3.group('versionNum'))

      if m4:
        self.patchNum = int(m4.group('patchNum'))+1
        lines.append('#define PETSC_VERSION_PATCH'+m4.group(1)+str(self.patchNum)+'\n')
      elif m5:
        self.date = time.strftime('%b %d, %Y', time.localtime(time.time()))
        lines.append('#define PETSC_VERSION_DATE'+m5.group(1)+'"'+self.date+'"\n')
      else:
        lines.append(line)
    input.close()
    output = file(filename, 'w')
    output.writelines(lines)
    output.close()
    self.executeShellCommand('bk ci -u -y"Cranked up patch level" '+filename)
    version = majorNum+'.'+minorNum+'.'+subminorNum
    if self.argDB.has_key('version'):
      if not self.argDB['version'] == verison:
        raise RuntimeError('Specified version ('+self.argDB['version']+') disagrees with petscversion.h ('+version+')')
    else:
      self.argDB['version'] = version
    print 'Changed version to '+version+'.'+self.patchNum
    return

  def pushChange(self):
    '''Push the change and parse the output to discover the change sets'''
    self.executeShellCommand('bk citool')
    output = self.executeShellCommand('bk push')
    # Get the change sets pushed
    self.changeSets = []
    m = re.search(r'----------------------- Sending the following csets -----------------------\n(?P<sets>[\d\. ]+)\n---------------------------------------------------------------------------', output)
    if m:
      sets = re.split(r'\s+', m.group('sets'))
      for set in sets:
        if not set: continue
        print 'Change Set: '+set
        m = re.match(r'^(\d)\.(\d+)$', set)
        if m and m.group(1) == '1':
          self.changeSets.append(int(m.group(2)))
        else:
          raise RuntimeError('Invalid change set: '+set)
    else:
      raise RuntimeError('Cannot parse push output: '+output)
    print 'Pushed changes in '+str(self.changeSets)
    return self.changeSets

  def makePatch(self):
    '''Make a patch, self.patchNum, from the pushed change sets, self.changeSets'''
    command = 'bk export -tpatch -T'
    if len(self.changeSets) == 1:
      command += ' -r1.'+str(self.changeSets[0])
    else:
      command += ' -r1.'+str(min(self.changeSets)-1)+',1.'+str(max(self.changeSets))
    self.patch = self.executeShellCommand(command)

    patchName = os.path.join(self.patchDir, 'petsc_patch-'+self.version+'.'+str(self.patchNum))
    patchFile = file(patchName, 'w')
    patchFile.write(self.patch)
    patchFile.close()
    os.chmod(patchName, 0644)
    print 'Made patch '+patchFile.name
    return

  def makeMasterPatch(self):
    '''Recreate the master patch from all patch files present'''
    masterName = os.path.join(self.patchDir, 'petsc_patch_all-'+self.version)
    if os.path.exists(masterName): os.remove(masterName)
    masterFile = file(masterName, 'w')
    for num in range(self.patchNum+1):
      try:
        patchFile = file(os.path.join(self.patchDir, 'petsc_patch-'+self.version+'.'+str(num)))
        patch     = patchFile.read()
        patchFile.close()
        masterFile.write(patch)
      except IOError:
        pass
    masterFile.close()
    os.chmod(masterName, 0664)
    print 'Made master patch '+patchFile.name
    return

  def integrateChange(self):
    # Precondition
    if not self.argDB.has_key('version'):
      raise RuntimeError('No version number specified')
    output = self.executeShellCommand('bk pull bk://petsc.bkbits.net/petsc-release-'+self.argDB['version'])
    print 'Integrated changes into development repository'
    return

  def setupRun(self):
    if self.argDB.has_key('submit') and self.argDB['submit']:
      self.doVersion     = 1
      self.doPush        = 1
      self.doPatch       = 1
      self.doMasterPatch = 1
    if self.argDB.has_key('updateVersion') and self.argDB['updateVersion']:
      self.doVersion = 1
    if self.argDB.has_key('pushChange') and self.argDB['pushChange']:
      self.doPush = 1
    if self.argDB.has_key('makePatch') and self.argDB['makePatch']:
      self.doPatch = 1
    if self.argDB.has_key('makeMasterPatch') and self.argDB['makeMasterPatch']:
      self.doMasterPatch = 1
    if self.argDB.has_key('integrate') and self.argDB['integrate']:
      self.doIntegrate = 1
    return

  def submit(self):
    self.setupRun()
    if doVersion:
      self.setVersion()
    if doPush:
      self.pushChange()
    if doPatch:
      self.makePatch()
    if doMasterPatch:
      self.makeMasterPatch()
    #self.updateWeb()
    if doIntegrate:
      self.integrateChange()
    return

if __name__ == '__main__':
  sys.path.insert(0, os.path.abspath('python'))
  try:
    Patch(sys.argv[1:]).submit()
    # Need self.patchNum and self.changeSets set in order to bypass the push
    # Need a better way of getting change sets
    #   Should be 
  except Exception, e:
    import sys
    import traceback

    print traceback.print_tb(sys.exc_info()[2])
    print str(e)
