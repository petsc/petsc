#!/usr/bin/env python
import os
import sys
import urllib
import urlparse
# Fix parsing for nonstandard schemes
urlparse.uses_netloc.extend(['bk', 'ssh'])

class Patch (object):
  def __init__(self, clArgs = None, argDB = None):
    import help

    self.argDB      = self.setupArgDB(clArgs, argDB)
    self.root       = os.getcwd()
    self.logFile    = sys.stdout
    self.help       = help.Help(self.argDB)
    self.help.title = 'Patch Submission Help'
    self.checkPetscRoot(self.root)
    self.setupArguments(self.clArgs)
    return

  def setupArgDB(self, clArgs, initDB):
    self.clArgs = clArgs
    if initDB is None:
      import RDict
      argDB = RDict.RDict()
    else:
      argDB = initDB
    return argDB

  def setupArguments(self, clArgs = None):
    import nargs

    self.help.addArgument('Main', 'help', nargs.ArgBool(None, 0, 'Print this help message', isTemporary = 1))
    self.help.addArgument('Main', 'h',    nargs.ArgBool(None, 0, 'Print this help message', isTemporary = 1))
    self.help.addArgument('Actions', 'updateVersion',   nargs.ArgBool(None, 1, 'Update petscversion.h'))
    self.help.addArgument('Actions', 'pushChange',      nargs.ArgBool(None, 1, 'Push changes'))
    self.help.addArgument('Actions', 'makePatch',       nargs.ArgBool(None, 1, 'Construct the patch for Petsc'))
    self.help.addArgument('Actions', 'makeMasterPatch', nargs.ArgBool(None, 1, 'Construct the master patch for Petsc'))
    self.help.addArgument('Actions', 'integratePatch',  nargs.ArgBool(None, 1, 'Integrate changes into the Petsc development repository'))
    self.help.addArgument('Actions', 'updateWeb',       nargs.ArgBool(None, 1, 'Update the patches web page'))
    patchUrl = os.path.join('/mcs', 'ftp', 'pub', 'petsc', 'patches')
    if not os.path.isdir(patchUrl):
      patchUrl = urlparse.urlunparse(('ssh', 'petsc@login.mcs.anl.gov', patchUrl, '', '', ''))
    else:
      patchUrl = urlparse.urlunparse(('file', '', patchUrl, '', '', ''))
    self.help.addArgument('Variables', 'patchUrl=<dir>', nargs.Arg(None, patchUrl, 'The directory containing both the patch and master patch files'))
    # Variables necessary when some actions are excluded
    self.help.addArgument('Variables for missing actions', 'version=<num>',         nargs.Arg(None, None, 'The version number being patched (defined in updateVersion), e.g. 2.1.0', isTemporary = 1))
    self.help.addArgument('Variables for missing actions', 'patchNum=<num>',        nargs.ArgInt(None, None, 'The patch number (defined in updateVersion), e.g. 1', min = 1, isTemporary = 1))
    self.help.addArgument('Variables for missing actions', 'changeSets=[<num>...]', nargs.Arg(None, None, 'The ChangeSets which were pushed (defined in pushChange), e.g. 1.1052', isTemporary = 1))

    self.argDB.insertArgs(clArgs)
    return

  def writeLogLine(self, message):
    '''Writes the message to the log along with the current time'''
    import time
    self.logFile.write('('+str(os.getpid())+')('+str(id(self))+')'+message+' ['+time.asctime(time.localtime())+']\n')
    self.logFile.flush()
    return

  def defaultCheckCommand(self, command, status, output):
    if status: raise RuntimeError('Could not execute\''+command+'\': '+output)
    return

  # Should use Maker.executeShellCommand() here
  def executeShellCommand(self, command, checkCommand = None):
    '''Execute a shell command'''
    import commands

    self.writeLogLine('Executing shell cmd: '+command)
    (status, output) = commands.getstatusoutput(command)
    if checkCommand:
      checkCommand(command, status, output)
    else:
      self.defaultCheckCommand(command, status, output)
    self.writeLogLine('  Output: '+output)
    return output

  def checkPetscRoot(self, root):
    '''Find the root of the Petsc tree. Currently, we require it to be os.getcwd()'''
    if not os.path.isfile(os.path.join(root, 'include', 'petscsys.h')):
      raise RuntimeError('Directory '+root+' is not the root of a Petsc tree')
    return

  def setVersion(self):
    '''If petscversion.h has not been edited, increment the patch level and checkin'''
    if not self.argDB['updateVersion']: return
    import re
    import time

    filename = os.path.join(self.root, 'include', 'petscversion.h')
    # Check to see if petscversion.h was edited
    output   = self.executeShellCommand('bk sfiles -lg '+filename)
    if output and output.find(filename) >= 0: return
    # Parse petscversion.h and calculate the new version
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
        patchNum = int(m4.group('patchNum'))+1
        lines.append('#define PETSC_VERSION_PATCH'+m4.group(1)+str(patchNum)+'\n')
      elif m5:
        self.date = time.strftime('%b %d, %Y', time.localtime(time.time()))
        lines.append('#define PETSC_VERSION_DATE'+m5.group(1)+'"'+self.date+'"\n')
      else:
        lines.append(line)
    input.close()
    # Update the version and patchNum in argDB
    version = '%d.%d.%d' % (majorNum, minorNum, subminorNum)
    if 'version' in self.argDB:
      if not self.argDB['version'] == version:
        raise RuntimeError('Specified version ('+self.argDB['version']+') disagrees with petscversion.h ('+version+')')
    else:
      self.argDB['version'] = version
    if 'patchNum' in self.argDB:
      if not self.argDB['patchNum'] == patchNum:
        raise RuntimeError('Specified patch number ('+self.argDB['patchNum']+') disagrees with the one create ('+str(patchNum)+')')
    else:
      self.argDB['patchNum'] = patchNum
    # Write the new petscversion.h
    output = file(filename, 'w')
    output.writelines(lines)
    output.close()
    # Check in the new petscversion.h
    #self.executeShellCommand('bk ci -u -y"Cranked up patch level" '+filename)
    self.writeLogLine('Changed version to '+version+'.'+str(patchNum))
    return

  def pushChange(self):
    '''Push the change and parse the output to discover the change sets'''
    if not self.argDB['pushChange']: return
    import re

    # Create any remaining change set
    # FIX: Use bk sfiles to find out if any files are locked
    self.executeShellCommand('bk citool')
    # Get the change sets to be pushed
    changeSets = []
    output     = self.executeShellCommand('bk changes -L')
    setRE      = re.compile(r'ChangeSet@(?P<changeSet>\d+(\.\d+)*),.*')
    for line in output.split('\n'):
      m = setRE.match(line)
      if m:
        changeSets.append(m.group('changeSet'))
    if not len(changeSets): raise RuntimeError('No change sets to be submitted')
    changeSets.sort()
    output = self.executeShellCommand('bk push')
    self.writeLogLine('Pushed change sets '+str(changeSets))
    if 'changeSets' in self.argDB:
      if not self.argDB['changeSets'] == changeSets:
        raise RuntimeError('Specified ChangeSets ('+self.argDB['changeSets']+') disagree with those pushed ('+str(changeSets)+')')
    else:
      self.argDB['changeSets'] = changeSets
    return

  def prevChangeSet(self, set):
    '''Find the previous ChangeSet'''
    num = set.split('.')
    if len(num) > 2:
      raise RuntimeError('A merge set should not be the first change set')
    return num[0]+'.'+str(int(num[1])-1)

  def makePatch(self):
    '''Make a patch from the pushed change sets'''
    if not self.argDB['makePatch']: return
    changeSets = self.argDB['changeSets']
    command    = 'bk export -tpatch -T'
    if len(changeSets) == 1:
      command += ' -r'+str(changeSets[0])
    else:
      command += ' -r'+str(self.prevChangeSet(changeSets[0]))+','+str(changeSets[-1])
    self.patch = self.executeShellCommand(command)

    (scheme, location, path, parameters, query, fragment) = urlparse.urlparse(self.argDB['patchUrl'])
    path      = os.path.join(path, 'petsc_patch-'+self.argDB['version']+'.'+str(self.argDB['patchNum']))
    patchName = urlparse.urlunparse((scheme, location, path, parameters, query, fragment))
    if scheme == 'file':
      patchFile = file(path, 'w')
      patchFile.write(self.patch)
      patchFile.write('\n')
      patchFile.close()
      os.chmod(patchName, 0664)
    elif scheme == 'ssh':
      tmpName   = os.path.join('/tmp', os.tmpnam())
      patchFile = file(tmpName, 'w')
      patchFile.write(self.patch)
      patchFile.write('\n')
      patchFile.close()
      output    = self.executeShellCommand('scp '+tmpName+' '+location+':'+path)
      os.remove(tmpName)
      output    = self.executeShellCommand('ssh '+location+' chmod 664 '+path)
    else:
      raise RuntimeError('Cannot handle patch URL: '+self.argDB['patchUrl'])
    self.writeLogLine('Made patch '+patchName)
    return

  def makeMasterPatch(self):
    '''Recreate the master patch from all patch files present'''
    if not self.argDB['makeMasterPatch']: return
    (scheme, location, path, parameters, query, fragment) = urlparse.urlparse(self.argDB['patchUrl'])
    basePath   = path
    path       = os.path.join(path, 'petsc_patch_all-'+self.argDB['version'])
    masterName = urlparse.urlunparse((scheme, location, path, parameters, query, fragment))
    if scheme == 'file':
      if os.path.exists(masterName): os.remove(masterName)
      masterFile = file(masterName, 'w')
      for num in range(self.argDB['patchNum']+1):
        try:
          patchFile = file(os.path.join(basePath, 'petsc_patch-'+self.argDB['version']+'.'+str(num)))
          patch     = patchFile.read()
          patchFile.close()
          masterFile.write(patch)
        except IOError:
          pass
      masterFile.close()
      os.chmod(masterName, 0664)
    elif scheme == 'ssh':
      tmpName    = os.path.join('/tmp', os.tmpnam())
      masterFile = file(tmpName, 'w')
      for num in range(self.argDB['patchNum']+1):
        try:
          output = self.executeShellCommand('ssh '+location+' cat '+os.path.join(basePath, 'petsc_patch-'+self.argDB['version']+'.'+str(num)))
          masterFile.write(output)
        except RuntimeError:
          pass
      masterFile.close()
      output = self.executeShellCommand('scp '+tmpName+' '+location+':'+path)
      os.remove(tmpName)
      output = self.executeShellCommand('ssh '+location+' chmod 664 '+path)
    else:
      raise RuntimeError('Cannot handle patch URL: '+self.argDB['patchUrl'])
    self.writeLogLine('Made master patch '+masterName)
    return

  def getPetscDevelopmentDir(self):
    '''Find the development copy of PETSc'''
    dir = os.path.abspath(os.path.join(os.getcwd(), '..', 'petsc'))
    if not os.path.isdir(dir):
      dir = os.path.abspath(os.path.join(os.getcwd(), '..', 'petsc-dev'))
      if not os.path.isdir(dir):
        raise RuntimeError('Could not find PETSc development directory')
    # I should really check that its parent is the dev repository
    return dir

  def integratePatch(self):
    '''This method pulls the patch back into the development repository'''
    if not self.argDB['integratePatch']: return
    oldDir = os.getcwd()
    os.chdir(self.getPetscDevelopmentDir())
    output = self.executeShellCommand('bk pull bk://petsc.bkbits.net/petsc-release-'+self.argDB['version'])
    os.chdir(oldDir)
    self.writeLogLine('Integrated changes into development repository')
    return

  def updateWeb(self):
    '''Update the patches web page'''
    if not self.argDB['updateWeb']: return
    self.writeLogLine('Cannot update web until after merge with petsc/private')
    return

  def submit(self):
    if self.argDB['help'] or self.argDB['h']:
      self.help.output()
      return 0
    self.setVersion()
    self.pushChange()
    self.makePatch()
    self.makeMasterPatch()
    self.integratePatch()
    self.updateWeb()
    return 1

if __name__ == '__main__':
  sys.path.insert(0, os.path.abspath('python'))
  sys.path.insert(0, os.path.abspath(os.path.join('python', 'BuildSystem')))
  try:
    Patch(sys.argv[1:]).submit()
  except Exception, e:
    import traceback

    print traceback.print_tb(sys.exc_info()[2])
    print str(e)
