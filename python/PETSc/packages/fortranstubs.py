from __future__ import generators
import config.base
import os
import re
    
class Configure(config.base.Configure):
  def __init__(self, framework):
    config.base.Configure.__init__(self, framework)
    self.headerPrefix = 'PETSC'
    self.substPrefix  = 'PETSC'
    self.argDB        = framework.argDB
    return
     
  def configureHelp(self, help):
    import nargs
    help.addArgument('fortranstubs', '-with-bfort-if-needed',     nargs.ArgBool(None, 1, 'Download bfort if needed'))
    return

  def downLoadbfort(self):
    self.framework.log.write('Downloading bfort\n')
    # Check for SOWING
    dirs = []
    for dir in os.listdir(self.framework.argDB['PETSC_DIR']):
      if dir.startswith('sowing') and os.path.isdir(os.path.join(self.framework.argDB['PETSC_DIR'], dir)):
        dirs.append(dir)
    # Download SOWING if necessary
    if len(dirs) == 0:
      import urllib
      try:
        urllib.urlretrieve('ftp://ftp.mcs.anl.gov/pub/sowing/sowing.tar.gz', 'sowing.tar.gz')
      except Exception, e:
        raise RuntimeError('Error downloading Sowing: '+str(e))
      try:
        config.base.Configure.executeShellCommand('gunzip sowing.tar.gz', log = self.framework.log)
      except RuntimeError, e:
        raise RuntimeError('Error unzipping sowing.tar.gz: '+str(e))
      try:
        config.base.Configure.executeShellCommand('tar -xf sowing.tar', log = self.framework.log)
      except RuntimeError, e:
        raise RuntimeError('Error doing tar -xf sowing.tar: '+str(e))
      os.unlink('sowing.tar')
    # Get the SOWING directories
    sowingDir = None
    for dir in os.listdir(self.framework.argDB['PETSC_DIR']):
      if dir.startswith('sowing') and os.path.isdir(os.path.join(self.framework.argDB['PETSC_DIR'], dir)):
        sowingDir = dir
    if sowingDir is None:
      raise RuntimeError('Error locating sowing directory')
    installDir = os.path.join(self.framework.argDB['PETSC_DIR'],sowingDir, self.framework.argDB['PETSC_ARCH'])
    if not os.path.isdir(installDir):
      os.mkdir(installDir)
    # Configure and Build sowing
    args = ['--prefix='+installDir]
    args = ' '.join(args)
    try:
      fd = open(os.path.join(installDir,'config.args'),'r')
      oldargs = fd.readline()
      fd.close()
    except:
      oldargs = ''
    if not oldargs == args:
      try:
        output  = config.base.Configure.executeShellCommand('cd '+sowingDir+';./configure '+args, timeout=900, log = self.framework.log)[0]
      except RuntimeError, e:
        raise RuntimeError('Error running configure on Sowing: '+str(e))
      try:
        output  = config.base.Configure.executeShellCommand('cd '+sowingDir+';make; make install', timeout=2500, log = self.framework.log)[0]
      except RuntimeError, e:
        raise RuntimeError('Error running make; make install on Sowing: '+str(e))
      fd = open(os.path.join(installDir,'config.args'),'w')
      fd.write(args)
      fd.close()
    self.framework.bfort = os.path.join(installDir,'bin','bfort')

  def configureFortranStubs(self):
    '''Determine whether the Fortran stubs exist or not'''
    if os.path.exists(os.path.join(self.framework.argDB['PETSC_DIR'], 'BitKeeper')):
      self.framework.log.write('BitKeeper clone of PETSc, checking for bfort')
      self.framework.getExecutable('bfort', getFullPath = 1)

      # try to download bfort if not found
      if not hasattr(self.framework, 'bfort') and self.framework.argDB['with-bfort-if-needed']:
        self.downLoadbfort()
        
      if hasattr(self.framework, 'bfort'):
        self.framework.addSubstitution('BFORT', self.framework.bfort)
      else:
        message = 'See http:/www.mcs.anl.gov/petsc/petsc-2/developers for how\nto obtain bfort to generate the Fortran stubs or make sure\nbfort is in your path\n'
        self.framework.log.write(message)
        raise RuntimeError('You have a Fortran compiler but the PETSc Fortran stubs are not built and cannot be built.\n'+message+'or run with with --with-fc=0 to turn off the Fortran compiler')
    else:
      self.framework.log.write('Fortran stubs do exist in '+stubDir+'\n')
      # check if the SOWING directory exists and has bfort
      sowingDir = None
      for dir in os.listdir(self.framework.argDB['PETSC_DIR']):
        if dir.startswith('sowing') and os.path.isdir(os.path.join(self.framework.argDB['PETSC_DIR'], dir)):
          sowingDir = dir
      if sowingDir:
        bfort = os.path.join(self.framework.argDB['PETSC_DIR'],sowingDir, self.framework.arch,'bin','bfort')
        if os.path.isfile(bfort):
          self.framework.log.write('Found downloaded Sowing installed, will use this')
          self.framework.addSubstitution('BFORT', bfort)
    
      self.framework.log.write('Not BitKeeper clone of PETSc, assuming Fortran stubs already built')
    return

  def configure(self):
    self.framework.addSubstitution('BFORT', 'bfort')
    if 'FC' in self.framework.argDB:
      self.executeTest(self.configureFortranStubs)
    return
