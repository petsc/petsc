#!/usr/bin/env python

import commands
import os
import re
import time

class Patch (object):
  def __init__(self):
    self.root     = os.getcwd()
    self.patchDir = os.path.join('/mcs', 'ftp', 'pub', 'petsc', 'patches')
    self.checkPetscRoot(self.root)
    return

  def defaultCheckCommand(self, command, status, output):
    if status: raise RuntimeError('Could not execute\''+command+'\': '+output)

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

  def setVersion(self):
    '''If petscversion.h has not been edited, increment the patch level and checkin'''
    filename = os.path.join(self.root, 'include', 'petscversion.h')
    output   = self.executeShellCommand('bk sfiles -lg '+filename)
    if output.find(filename) >= 0: return
    self.executeShellCommand('bk edit '+filename)
    patchRE = re.compile(r'^#define PETSC_VERSION_PATCH([\s]+)(?P<patchNum>\d+)[\s]*$');
    dateRE  = re.compile(r'^#define PETSC_VERSION_DATE([\s]+)"(?P<date>(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) \d\d?, \d\d\d\d)"[\s]*$');
    input   = file(filename)
    lines   = []
    for line in input.readlines():
      m1 = patchRE.match(line)
      m2 = dateRE.match(line)
      if m1:
        self.patchNum = int(m1.group('patchNum'))+1
        lines.append('#define PETSC_VERSION_PATCH'+m1.group(1)+str(self.patchNum)+'\n')
      elif m2:
        self.date = time.strftime('%b %d, %Y', time.localtime(time.time()))
        lines.append('#define PETSC_VERSION_DATE'+m2.group(1)+'"'+self.date+'"\n')
      else:
        lines.append(line)
    input.close()
    output = file(filename, 'w')
    output.writelines(lines)
    output.close()
    self.executeShellCommand('bk ci -u -y"Cranked up patch level" '+filename)

  def pushChange(self):
    '''Push the change and parse the output to discover the change sets'''
    self.executeShellCommand('bk citool')
    output = self.executeShellCommand('bk push')
    # Get the change sets pushed
    self.changeSets = []
    m = re.match(r'----------------------- Sending the following csets -----------------------\n(?P<sets>[\d\. ]+)\n---------------------------------------------------------------------------', output)
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
    return self.changeSets

  def makePatch(self):
    '''Make a patch, self.patchNum, from the pushed change sets, self.changeSets'''
    command = 'bk export -tpatch -T'
    if len(self.changeSets) == 1:
      command += ' -r1.'+str(self.changeSets[0])
    else:
      command += ' -r1.'+str(min(self.changeSets)-1)+',1.'+str(max(self.changeSets))
    self.patch = self.executeShellCommand(command)

    patchName = os.path.join(self.patchDir, 'petsc_patch-2.1.3.'+str(self.patchNum))
    patchFile = file(patchName, 'w')
    patchFile.write(self.patch)
    patchFile.close()
    os.chmod(patchName, 0644)

  def makeMasterPatch(self):
    '''Recreate the master patch from all patch files present'''
    masterName = os.path.join(self.patchDir, 'petsc_patch_all-2.1.3')
    if os.path.exists(masterName): os.remove(masterName)
    masterFile = file(masterName, 'w')
    for num in range(self.patchNum+1):
      try:
        patchFile = file(os.path.join(self.patchDir, 'petsc_patch-2.1.3.'+str(num)))
        patch     = patchFile.read()
        patchFile.close()
        masterFile.write(patch)
      except IOError:
        pass
    masterFile.close()
    os.chmod(masterName, 0644)

  def submit(self):
    self.setVersion()
    self.pushChange()
    self.makePatch()
    self.makeMasterPatch()
    #self.updateWeb()

if __name__ == '__main__':
  try:
    Patch().submit()
    # Need self.patchNum and self.changeSets set in order to bypass the push
    # Need a better way of getting change sets
    #   Should be 
  except Exception, e:
    import sys
    import traceback

    print traceback.print_tb(sys.exc_info()[2])
    print str(e)
