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
    self.compilers    = self.framework.require('config.compilers', self)
    self.arch         = self.framework.require('PETSc.utilities.arch', self)
    return

  def getDir(self):
    '''Find the directory containing Sowing'''
    packages  = self.framework.argDB['with-external-packages-dir']
    if not os.path.isdir(packages):
      os.mkdir(packages)
      self.framework.actions.addArgument('PETSc', 'Directory creation', 'Created the packages directory: '+packages)
    sowingDir = None
    for dir in os.listdir(packages):
      if dir.startswith('sowing') and os.path.isdir(os.path.join(packages, dir)):
        sowingDir = dir
    if sowingDir is None:
      self.framework.log.write('Could not locate a downloading Sowing\n')
      raise RuntimeError('Error locating Sowing directory')
    return os.path.join(packages, sowingDir)

  def downLoadSowing(self):
    self.framework.log.write('Downloading Sowing\n')
    try:
      sowingDir = self.getDir()
      self.framework.log.write('Sowing tar file already downloaded\n')
    except RuntimeError:
      import urllib

      packages = self.framework.argDB['with-external-packages-dir']
      self.framework.log.write('Need to actually ftp Sowing\n')
      try:
        urllib.urlretrieve('ftp://ftp.mcs.anl.gov/pub/sowing/sowing.tar.gz', os.path.join(packages, 'sowing.tar.gz'))
      except Exception, e:
        raise RuntimeError('Error downloading Sowing: '+str(e))
      try:
        config.base.Configure.executeShellCommand('cd '+packages+'; gunzip sowing.tar.gz', log = self.framework.log)
      except RuntimeError, e:
        raise RuntimeError('Error unzipping sowing.tar.gz: '+str(e))
      try:
        config.base.Configure.executeShellCommand('cd '+packages+' ;tar -xf sowing.tar', log = self.framework.log)
      except RuntimeError, e:
        raise RuntimeError('Error doing tar -xf sowing.tar: '+str(e))
      os.unlink(os.path.join(packages,'sowing.tar'))
      self.framework.actions.addArgument('Sowing', 'Download', 'Downloaded Sowing into '+self.getDir())
    # Get the SOWING directories
    sowingDir = self.getDir()
    installDir = os.path.join(sowingDir, self.arch.arch)
    if not os.path.isdir(installDir):
      os.mkdir(installDir)
    # Configure and Build sowing
    args = ['--prefix='+installDir, '--with-cc='+'"'+self.framework.argDB['CC']+'"']
    args = ' '.join(args)
    try:
      fd      = file(os.path.join(installDir,'config.args'))
      oldargs = fd.readline()
      fd.close()
    except:
      oldargs = ''
    if not oldargs == args:
      self.framework.log.write('Need to configure and compile Sowing: old args = '+oldargs+' new args '+args+'\n')
      try:
        output  = config.base.Configure.executeShellCommand('cd '+sowingDir+';./configure '+args, timeout=900, log = self.framework.log)[0]
      except RuntimeError, e:
        raise RuntimeError('Error running configure on Sowing: '+str(e))
      try:
        output  = config.base.Configure.executeShellCommand('cd '+sowingDir+';make; make install; make clean', timeout=2500, log = self.framework.log)[0]
      except RuntimeError, e:
        raise RuntimeError('Error running make; make install on Sowing: '+str(e))
      fd = file(os.path.join(installDir,'config.args'), 'w')
      fd.write(args)
      fd.close()
      self.framework.actions.addArgument('Sowing', 'Install', 'Installed Sowing into '+installDir)
    self.binDir   = os.path.join(installDir, 'bin')
    self.bfort    = os.path.join(self.binDir, 'bfort')
    self.doctext  = os.path.join(self.binDir, 'doctext')
    self.mapnames = os.path.join(self.binDir, 'mapnames')
    # Bill's bug he does not install bib2html so use original location if needed
    if os.path.isfile(os.path.join(self.binDir, 'bib2html')):
      self.bib2html = os.path.join(self.binDir, 'bib2html')
    else:
      self.bib2html = os.path.join(sowingDir,'bin', 'bib2html')
    for prog in [self.bfort, self.doctext, self.mapnames]:
      if not (os.path.isfile(prog) and os.access(prog, os.X_OK)):
        raise RuntimeError('Error in Sowing installation: Could not find '+prog)
    self.addMakeMacro('BFORT ', self.bfort)
    self.addMakeMacro('DOCTEXT ', self.doctext)
    self.addMakeMacro('MAPNAMES ', self.mapnames)
    self.addMakeMacro('BIB2HTML ', self.bib2html)    
    return

  def configureSowing(self):
    '''Determine whether the Sowing exist or not'''
    if os.path.exists(os.path.join(self.arch.dir, 'BitKeeper')):
      self.framework.log.write('BitKeeper clone of PETSc, checking for Sowing\n')
      self.getExecutable('pdflatex', getFullPath = 1)
      self.getExecutable('bfort', getFullPath = 1)
      self.getExecutable('doctext', getFullPath = 1)
      self.getExecutable('mapnames', getFullPath = 1)
      self.getExecutable('bib2html', getFullPath = 1)            
      if not hasattr(self, 'bfort'):
        self.downLoadSowing()
    else:
      self.framework.log.write("Not BitKeeper clone of PETSc, don't need Sowing\n")
    return

  def configure(self):
    self.executeTest(self.configureSowing)
    if 'FC' in self.framework.argDB:
      self.framework.log.write('           Running '+self.bfort+' to generate fortran stubs\n')
      try:
        (output, error, status) = config.base.Configure.executeShellCommand(os.path.join('maint','generatefortranstubs.py') + ' ' +self.bfort, timeout = 15*60.0, log = self.framework.log)
        self.framework.actions.addArgument('PETSc', 'File creation', 'Generated Fortran stubs ')
      except RuntimeError, e:
        raise RuntimeError('*******Error generating Fortran stubs: '+str(e)+'*******\n')
    return
