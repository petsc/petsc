from __future__ import generators
import config.base
import os

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
    help.addArgument('Matlab', '-enable-update',                nargs.ArgBool(None, 1, 'Update source code from PETSc website'))
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
    self.framework.getExecutable('patch')
    if not hasattr(self.framework, 'patch'):
      self.framework.log.write('Cannot find patch program.\nContinuing configure without patches.\n')
      return
    # on solaris and alpha make sure patch is gnupatch?

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
      output1 = self.executeShellCommand('echo patch -Np1 < patches1')
      os.unlink('patches1')
      output2 = self.executeShellCommand('cd python/BuildSystem; echo patch -Np1 < ../../patches2')
      os.unlink('patches2')
      if output1.find('error') >= 0 or output2.find('error') >= 0:
        self.framework.log.write(output1+'\n')
        self.framework.log.write(output2+'\n')
        raise RuntimeError('Error applying source update.\nRun with --enable-update=0 to configure anyways')
    except RuntimeError:
      self.framework.log.write('Error applying patches. Continuing configure anyways.\n')
      try: os.unlink('patches1')
      except: pass
    self.strmsg = 'Updated source code from PETSc website (using latest patches)'
    return 

  def configure(self):
    self.executeTest(self.configureDirectories)
    if not self.framework.argDB['enable-update']: return
    if os.path.isdir('BitKeeper'): 
      self.executeTest(self.updateBK)
      self.executeTest(self.updateBKBuildSystem)
    else:                          self.executeTest(self.updatePatches)
    return
