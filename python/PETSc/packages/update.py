from __future__ import generators
import config.base
import os
import re

class Configure(config.base.Configure):
  def __init__(self, framework):
    config.base.Configure.__init__(self, framework)
    self.headerPrefix = ''
    self.substPrefix  = ''
    self.updated  = 0
    self.strmsg   = ''
    return

  def __str__(self):
    return self.strmsg
     
  def configureHelp(self, help):
    import nargs
    help.addArgument('Update', '-enable-update',             nargs.ArgBool(None, 1, 'Update source code from PETSc website'))
    help.addArgument('Update', '-with-patch',                nargs.Arg(None, None, 'Location of GNU patch program'))
    help.addArgument('Update', '-with-patch-petsc',          nargs.Arg(None, None, 'Location of a patch for the PETSc source'))
    help.addArgument('Update', '-with-patch-buildsystem',    nargs.Arg(None, None, 'Location of a patch for the BuildSystem'))
    return

  def configureArchitecture(self):
    '''Sets PETSC_ARCH'''
    import sys
    # Find auxilliary directory by checking for config.sub
    auxDir = None
    for dir in [os.path.abspath(os.path.join('bin', 'config')), os.path.abspath('config')] + sys.path:
      if os.path.isfile(os.path.join(dir, 'config.sub')):
        auxDir      = dir
        configSub   = os.path.join(auxDir, 'config.sub')
        configGuess = os.path.join(auxDir, 'config.guess')
        break
    if not auxDir: raise RuntimeError('Unable to locate config.sub in order to determine architecture.Your PETSc directory is incomplete.\n Get PETSc again')
    try:
      # Guess host type (should allow user to specify this
      host = self.executeShellCommand(self.shell+' '+configGuess)
      # Get full host description
      output = self.executeShellCommand(self.shell+' '+configSub+' '+host)
    except:
      raise RuntimeError('Unable to determine host type using config.sub')
    # Parse output
    m = re.match(r'^(?P<cpu>[^-]*)-(?P<vendor>[^-]*)-(?P<os>.*)$', output)
    if not m: raise RuntimeError('Unable to parse output of config.sub: '+output)
    self.framework.host_cpu    = m.group('cpu')
    self.host_vendor = m.group('vendor')
    self.host_os     = m.group('os')

##    results = self.executeShellCode(self.macroToShell(self.hostMacro))
##    self.host_cpu    = results['host_cpu']
##    self.host_vendor = results['host_vendor']
##    self.host_os     = results['host_os']

    if not self.framework.argDB.has_key('PETSC_ARCH'):
      self.framework.arch = self.host_os
    else:
      self.framework.arch = self.framework.argDB['PETSC_ARCH']
    if not self.framework.arch.startswith(self.host_os):
      raise RuntimeError('PETSC_ARCH ('+self.framework.arch+') does not have our guess ('+self.host_os+') as a prefix!\nRun bin/petscarch --suggest and set the environment variable PETSC_ARCH to the suggested value.')
    self.addSubstitution('ARCH', self.framework.arch)
    self.framework.archBase = re.sub(r'^(\w+)[-_]?.*$', r'\1', self.framework.arch)
    self.addDefine('ARCH', self.framework.archBase)
    self.addDefine('ARCH_NAME', '"'+self.framework.arch+'"')
    return

  def configureDirectories(self):
    '''Checks PETSC_DIR and sets if not set'''
    if not self.framework.argDB.has_key('PETSC_DIR'):
      self.framework.argDB['PETSC_DIR'] = os.getcwd()
    self.dir = self.framework.argDB['PETSC_DIR']
    # Check for version
    if not os.path.exists(os.path.join(self.dir, 'include', 'petscversion.h')):
      raise RuntimeError('Invalid PETSc directory '+str(self.dir)+' it may not exist?')
    self.addSubstitution('DIR', self.dir)
    self.addDefine('DIR', self.dir)
    return

  def isGNUPatch(self,patch):
    '''Returns 1 if it is GNU patch or equivilent, exception if cannot run'''
    if self.framework.archBase.startswith('solaris') or self.framework.archBase.startswith('alpha'):
      try:
        output = self.executeShellCommand(patch+' --help')
      except:
        raise RuntimeError('Unable to run '+patch+' command')
      if output.find('gnu.org') == -1: return 0
    return 1

  #
  #  Issues: it tries to get the patch and apply everytime configure is called. Only safe way
  #  but still seems like overkill. 
  #
  def updateBK(self):
    '''Updates the source code from with bk pull'''
    self.framework.getExecutable('bk')
    if not hasattr(self.framework, 'bk'):
      self.framework.log.write('Cannot find bk program.\nContinuing configure without update.\n')
      return
    
    import re
    self.framework.log.write('Checking if can downloading latest source with bk\n')
    try:
      output1 = self.executeShellCommand('bk sfiles -lgpC')
      output2 = self.executeShellCommand('bk changes -L -v')
      if output2.startswith('Pseudo-terminal will not be allocated because stdin is not a terminal.') and len(output2) <= 72:
        output2 = ''
      if output1 or output2:
        self.framework.log.write('Cannot pull latest source code, you have changed files or bk change sets\n')
        self.framework.log.write(output1+'\n'+output2+'\n')
        return
    except RuntimeError:
      self.framework.log.write('BK failure in checking for latest changes\n')
      return

    self.framework.log.write('Downloading latest source with bk\n')
    try:
      output1 = self.executeShellCommand('bk pull')
      if output1.find('error') >= 0:
        raise RuntimeError('Error pulling latest source code from PETSc BK website\nRun with --enable-update=0 to configure without updating')
      if output1.find('Nothing to pull') >= 0:
        self.strmsg = 'Source is current with PETSc BK website\n'
      else: 
        self.strmsg = 'Updated source code from PETSc BK website\n'
    except RuntimeError:
      self.framework.log.write('Error doing bk pull. Continuing configure anyways.\n')
      self.framework.log.write(output1+'\n')
    return 

  def updateBKBuildSystem(self):
    '''Updates the source code from BuildSystem with bk pull'''
    self.framework.getExecutable('bk')
    if not hasattr(self.framework, 'bk'):
      self.framework.log.write('Cannot find bk program.\nContinuing configure without update.\n')
      return
    
    import re
    self.framework.log.write('Checking if can downloading latest source with bk\n')
    try:
      output1 = self.executeShellCommand('cd python/BuildSystem; bk sfiles -lgpC')
      output2 = self.executeShellCommand('cd python/BuildSystem; bk changes -L -v')
      if output2.startswith('Pseudo-terminal will not be allocated because stdin is not a terminal.') and len(output2) <= 72:
        output2 = ''
      if output1 or output2:
        self.framework.log.write('Cannot pull latest BuildSystem source code, you have changed files or bk change sets\n')
        self.framework.log.write(output1+'\n'+output2+'\n')
        return
    except RuntimeError:
      self.framework.log.write('BK failure in checking for latest changes in BuildSystem\n')
      return

    self.framework.log.write('Downloading latest BuildSystem source with bk\n')
    try:
      output2 = self.executeShellCommand('cd python/BuildSystem; bk pull')
      if output2.find('error') >= 0:
        raise RuntimeError('Error pulling latest source code from BuildSystem  BK website\nRun with --enable-update=0 to configure anyways')
      if output2.find('Nothing to pull') >= 0:
        self.strmsg = self.strmsg+'BuildSystem source is current with PETSc BK website\n'
      else: 
        self.strmsg = self.strmsg+'Updated BuildSystem source code from PETSc BK website\n'
    except RuntimeError:
      self.framework.log.write('Error doing bk pull on BuildSystem. Continuing configure anyways.\n')
      self.framework.log.write(output2+'\n')
    return 

  # should only apply patch if it truly has something new in it. Keep checksum somewhere?
  def updatePatches(self):
    '''Updates the source code from any available patches'''
    if self.framework.argDB.has_key('with-patch'):
      patch = self.framework.argDB['with-patch']
      try:
        if not self.isGNUPatch(patch):
          raise RuntimeError('Patch program provided with --with-patch='+patch+' must be gnu patch')
      except:
        raise RuntimeError('Cannot run patch program provided with --with-patch='+patch)
    else:
      self.framework.getExecutable('patch')
      if not hasattr(self.framework, 'patch'):
        self.framework.log.write('Cannot find patch program.\nContinuing configure without patches.\n')
        return
      patch = self.framework.patch
      try:
        if not self.isGNUPatch(patch):
          self.framework.log.write('Solaris and Alpha require GNU patch, run with --with-patch=<full path of gnu patch> \n')
          self.framework.log.write('Continuing configure without patch files\n')
          return
      except:
        self.framework.log.write('Error running patch --help, run with --with-patch=<full path of gnu patch> \n')
        self.framework.log.write('Continuing configure without patch files\n')
        return 0
    
    # Get PETSc current version number
    if not os.path.exists(os.path.join(self.dir, 'include', 'petscversion.h')):
      raise RuntimeError('Invalid PETSc directory '+str(self.dir)+' it may not exist?')
    fd  = open(os.path.join(self.dir, 'include', 'petscversion.h'))
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
    
    self.framework.log.write('Downloading latest patches for version '+version+'\n')
    patchfile1 =  'ftp://ftp.mcs.anl.gov/pub/petsc/patches/petsc_patch_all-'+version
    patchfile2 =  'ftp://ftp.mcs.anl.gov/pub/petsc/patches/buildsystem_patch_all-'+version
    import urllib
    try:
      urllib.urlretrieve(patchfile1, 'patches1')
      urllib.urlretrieve(patchfile2, 'patches2')
    except:
      self.framework.log.write('Unable to download patches. Perhaps you are off the network?\nContinuing configure without patches.\n')
      return
    try:
      output1 = self.executeShellCommand('echo '+patch+' -Np1 < patches1')
      os.unlink('patches1')
      output2 = self.executeShellCommand('cd python/BuildSystem; echo '+patch+' -Np1 < ../../patches2')
      os.unlink('patches2')
      if output1.find('error') >= 0 or output2.find('error') >= 0:
        self.framework.log.write(output1+'\n')
        self.framework.log.write(output2+'\n')
        raise RuntimeError('Error applying source update.\nRun with --enable-update=0 to configure anyways')
    except RuntimeError:
      try:
        os.unlink('patches1')
        os.unlink('patches2')        
      except: pass
      raise RuntimeError('Error applying source update.\nRun with --enable-update=0 to configure anyways')
    self.strmsg = 'Updated source code from PETSc website (using latest patches)'

    if self.framework.argDB.has_key('with-patch-petsc'):
      try:
        output1 = self.executeShellCommand(patch+' -Np1 < '+self.framework.argDB['with-patch-petsc'])
      except:
        raise RuntimeError('Unable to apply patch from '+self.framework.argDB['with-patch-petsc']+'with '+patch) 
      if output1.find('error') >= 0:
        self.framework.log.write(output1+'\n')
        raise RuntimeError('Error applying '+self.framework.argDB['with-patch-petsc']+' update.\n')
    if self.framework.argDB.has_key('with-patch-buildsystem'):
      try:
        output1 = self.executeShellCommand('cd python/BuildSystem; '+patch+' -Np1 < '+self.framework.argDB['with-patch-buildsystem'])
      except:
        raise RuntimeError('Unable to apply patch from '+self.framework.argDB['with-patch-buildsystem']+'with '+patch) 
      if output1.find('error') >= 0:
        self.framework.log.write(output1+'\n')
        raise RuntimeError('Error applying '+self.framework.argDB['with-patch-buildsystem']+' update.\n')

  def configure(self):
    self.executeTest(self.configureDirectories)
    self.executeTest(self.configureArchitecture)
    if not self.framework.argDB['enable-update']: return
    if os.path.isdir('BitKeeper'): 
      self.executeTest(self.updateBK)
      self.executeTest(self.updateBKBuildSystem)
    else:                          self.executeTest(self.updatePatches)
    return
