#!/usr/bin/env python

import commands
import os
import re
import time

class Patch (object):
  def __init__(self):
    self.root = os.getcwd()
    self.checkPetscRoot(self.root)
    return

  def defaultCheckCommand(self, command, status, output):
    if status: raise RuntimeError('Could not execute\''+command+'\': '+output)

  # Should use Maker.executeShellCommand() here
  def executeShellCommand(self, command, checkCommand = None):
    (status, output) = commands.getstatusoutput(command)
    if checkCommand:
      checkCommand(command, status, output)
    else:
      self.defaultCheckCommand(command, status, output)
    return output

  def checkPetscRoot(self, root):
    if not os.path.isfile(os.path.join(root, 'include', 'petsc.h')):
      raise RuntimeError('Directory '+root+' is not the root of a Petsc tree')

  def setVersion(self):
    filename = os.path.join(self.root, 'include', 'petscversion.h')
    if len(self.executeShellCommand('bk sfiles -lg '+filename)):
      return
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
    self.executeShellCommand('bk citool')
    self.executeShellCommand('bk push')

  def submit(self):
    self.setVersion()
    self.pushChange()
    #self.makePatch()
    #self.makeMasterPatch()
    #self.updateWeb()

if __name__ == '__main__':
  try:
    Patch().submit()
  except Exception, e:
    import sys
    import traceback

    print traceback.print_tb(sys.exc_info()[3])
    print str(e)
