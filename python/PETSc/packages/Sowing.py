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
    return

  def downLoadSowing(self):
    self.framework.log.write('Downloading Sowing\n')
    # Check for SOWING
    dirs     = []
    packages = os.path.join(self.framework.argDB['PETSC_DIR'],'packages')
    if not os.path.isdir(packages):
      os.mkdir(packages)
    for dir in os.listdir(packages):
      if dir.startswith('sowing') and os.path.isdir(os.path.join(packages, dir)):
        dirs.append(dir)
    # Download SOWING if necessary
    if len(dirs) == 0:
      import urllib
      try:
        urllib.urlretrieve('ftp://ftp.mcs.anl.gov/pub/sowing/sowing.tar.gz', os.path.join(packages,'sowing.tar.gz'))
      except Exception, e:
        raise RuntimeError('Error downloading Sowing: '+str(e))
      try:
        config.base.Configure.executeShellCommand('cd packages; gunzip sowing.tar.gz', log = self.framework.log)
      except RuntimeError, e:
        raise RuntimeError('Error unzipping sowing.tar.gz: '+str(e))
      try:
        config.base.Configure.executeShellCommand('cd packages ;tar -xf sowing.tar', log = self.framework.log)
      except RuntimeError, e:
        raise RuntimeError('Error doing tar -xf sowing.tar: '+str(e))
      os.unlink(os.path.join(packages,'sowing.tar'))
    # Get the SOWING directories
    sowingDir = None
    for dir in os.listdir(packages):
      if dir.startswith('sowing') and os.path.isdir(os.path.join(packages, dir)):
        sowingDir = dir
    if sowingDir is None:
      raise RuntimeError('Error locating sowing directory')
    installDir = os.path.join(packages,sowingDir, self.framework.argDB['PETSC_ARCH'])
    if not os.path.isdir(installDir):
      os.mkdir(installDir)
    # Configure and Build sowing
    args = ['--prefix='+installDir]
    args = ' '.join(args)
    try:
      fd      = file(os.path.join(installDir,'config.args'))
      oldargs = fd.readline()
      fd.close()
    except:
      oldargs = ''
    if not oldargs == args:
      try:
        output  = config.base.Configure.executeShellCommand('cd packages/'+sowingDir+';./configure '+args, timeout=900, log = self.framework.log)[0]
      except RuntimeError, e:
        raise RuntimeError('Error running configure on Sowing: '+str(e))
      try:
        output  = config.base.Configure.executeShellCommand('cd packages/'+sowingDir+';make; make install; make clean', timeout=2500, log = self.framework.log)[0]
      except RuntimeError, e:
        raise RuntimeError('Error running make; make install on Sowing: '+str(e))
      fd = file(os.path.join(installDir,'config.args'), 'w')
      fd.write(args)
      fd.close()
    self.framework.sowingDir = os.path.join(installDir,'bin')
    self.framework.bfort    = os.path.join(installDir,'bin','bfort')
    self.framework.doctext  = os.path.join(installDir,'bin','doctext')
    self.framework.mapnames = os.path.join(installDir,'bin','mapnames')        

  def configureSowing(self):
    '''Determine whether the Sowing exist or not'''
    if os.path.exists(os.path.join(self.framework.argDB['PETSC_DIR'], 'BitKeeper')):
      self.framework.log.write('BitKeeper clone of PETSc, checking for Sowing\n')
      self.framework.getExecutable('bfort', getFullPath = 1)
      self.framework.getExecutable('doctext', getFullPath = 1)
      self.framework.getExecutable('mapnames', getFullPath = 1)      
      if not hasattr(self.framework, 'bfort'):
        self.downLoadSowing()
        
      if hasattr(self.framework, 'bfort'):
        self.framework.addSubstitution('BFORT', self.framework.bfort)
        self.framework.addSubstitution('DOCTEXT', self.framework.doctext)
        self.framework.addSubstitution('MAPNAMES', self.framework.mapnames)

        self.framework.getExecutable('pdflatex', getFullPath = 1)
        if not hasattr(self.framework, 'pdflatex'): self.framework.pdflatex = 'CouldNotFind'
        self.framework.addSubstitution('PDFLATEX', self.framework.pdflatex)
      else:
        message = 'See http:/www.mcs.anl.gov/petsc/petsc-2/developers for how\nto obtain Sowing\n'
        self.framework.log.write(message)
        raise RuntimeError('Could not install Sowing\n'+message)
    else:
      self.framework.log.write("Not BitKeeper clone of PETSc, don't need Sowing\n")
    return

  def configure(self):
    self.executeTest(self.configureSowing)
    return
