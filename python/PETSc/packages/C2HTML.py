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

  def downLoadC2html(self):
    self.framework.log.write('Downloading C2HTML\n')
    packages = os.path.join(self.framework.argDB['PETSC_DIR'],'packages')
    if not os.path.isdir(packages):
      os.mkdir(packages)
    # Check for C2HTML
    dirs = []
    for dir in os.listdir(packages):
      if dir.startswith('c2html') and os.path.isdir(os.path.join(packages,dir)):
        dirs.append(dir)
    # Download C2HTML if necessary
    if len(dirs) == 0:
      import urllib
      try:
        urllib.urlretrieve('ftp://ftp.mcs.anl.gov/pub/petsc/c2html.tar.gz', os.path.join(packages,'c2html.tar.gz'))
      except Exception, e:
        raise RuntimeError('Error downloading C2html: '+str(e))
      try:
        config.base.Configure.executeShellCommand('cd packages; gunzip c2html.tar.gz', log = self.framework.log)
      except RuntimeError, e:
        raise RuntimeError('Error unzipping c2html.tar.gz: '+str(e))
      try:
        config.base.Configure.executeShellCommand('cd packages; tar -xf c2html.tar', log = self.framework.log)
      except RuntimeError, e:
        raise RuntimeError('Error doing tar -xf c2html.tar: '+str(e))
      os.unlink(os.path.join(packages,'c2html.tar'))
    # Get the C2HTML directories
    c2htmlDir = None
    for dir in os.listdir(packages):
      if dir.startswith('c2html') and os.path.isdir(os.path.join(packages,dir)):
        c2htmlDir = dir
    if c2htmlDir is None:
      raise RuntimeError('Error locating c2html directory')
    installDir = os.path.join(packages,c2htmlDir, self.framework.argDB['PETSC_ARCH'])
    if not os.path.isdir(installDir):
      os.mkdir(installDir)
    # Configure and Build c2html
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
        output  = config.base.Configure.executeShellCommand('cd packages/'+c2htmlDir+';./configure '+args, timeout=900, log = self.framework.log)[0]
      except RuntimeError, e:
        raise RuntimeError('Error running configure on C2html: '+str(e))
      try:
        output  = config.base.Configure.executeShellCommand('cd packages/'+c2htmlDir+';make; make install; make clean', timeout=2500, log = self.framework.log)[0]
      except RuntimeError, e:
        raise RuntimeError('Error running make; make install on C2html: '+str(e))
      fd = file(os.path.join(installDir,'config.args'), 'w')
      fd.write(args)
      fd.close()
    self.framework.c2htmlDir = os.path.join(installDir,'bin')
    self.framework.c2html    = os.path.join(installDir,'bin','c2html')

  def configureC2html(self):
    '''Determine whether the C2html exist or not'''
    if os.path.exists(os.path.join(self.framework.argDB['PETSC_DIR'], 'BitKeeper')):
      self.framework.log.write('BitKeeper clone of PETSc, checking for C2html\n')
      self.framework.getExecutable('c2html', getFullPath = 1)
      if not hasattr(self.framework, 'c2html'):
        self.downLoadC2html()
        
      if hasattr(self.framework, 'c2html'):
        self.framework.addSubstitution('C2HTML', self.framework.c2html)
      else:
        message = 'See http:/www.mcs.anl.gov/petsc/petsc-2/developers for how\nto obtain C2html\n'
        self.framework.log.write(message)
        raise RuntimeError('Could not install C2html\n'+message)
    else:
      self.framework.log.write("Not BitKeeper clone of PETSc, don't need C2html\n")
    return

  def configure(self):
    self.executeTest(self.configureC2html)
    return
