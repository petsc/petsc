#!/usr/bin/env python
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
    '''Find the directory containing c2html'''
    packages  = self.framework.argDB['with-external-packages-dir']
    if not os.path.isdir(packages):
      os.mkdir(packages)
      self.framework.actions.addArgument('PETSc', 'Directory creation', 'Created the packages directory: '+packages)
    c2htmlDir = None
    for dir in os.listdir(packages):
      if dir.startswith('c2html') and os.path.isdir(os.path.join(packages, dir)):
        c2htmlDir = dir
    if c2htmlDir is None:
      raise RuntimeError('Error locating c2html directory')
    return os.path.join(packages, c2htmlDir)

  def downLoadC2html(self):
    self.framework.log.write('Downloading C2HTML\n')
    try:
      c2htmlDir = self.getDir()
    except RuntimeError:
      import urllib

      packages = self.framework.argDB['with-external-packages-dir']
      try:
        urllib.urlretrieve('ftp://ftp.mcs.anl.gov/pub/petsc/c2html.tar.gz', os.path.join(packages, 'c2html.tar.gz'))
      except Exception, e:
        raise RuntimeError('Error downloading C2html: '+str(e))
      try:
        config.base.Configure.executeShellCommand('cd '+packages+'; gunzip c2html.tar.gz', log = self.framework.log)
      except RuntimeError, e:
        raise RuntimeError('Error unzipping c2html.tar.gz: '+str(e))
      try:
        config.base.Configure.executeShellCommand('cd '+packages+'; tar -xf c2html.tar', log = self.framework.log)
      except RuntimeError, e:
        raise RuntimeError('Error doing tar -xf c2html.tar: '+str(e))
      os.unlink(os.path.join(packages, 'c2html.tar'))
      self.framework.actions.addArgument('C2HTML', 'Download', 'Downloaded c2html into '+self.getDir())
    # Get the C2HTML directories
    c2htmlDir  = self.getDir()
    installDir = os.path.join(c2htmlDir, self.arch.arch)
    if not os.path.isdir(installDir):
      os.mkdir(installDir)
    # Configure and Build c2html
    args = ['--prefix='+installDir, '--with-cc='+'"'+self.framework.argDB['CC']+'"']
    args = ' '.join(args)
    try:
      fd      = file(os.path.join(installDir,'config.args'))
      oldargs = fd.readline()
      fd.close()
    except:
      oldargs = ''
    if not oldargs == args:
      try:
        output  = config.base.Configure.executeShellCommand('cd '+c2htmlDir+';./configure '+args, timeout=900, log = self.framework.log)[0]
      except RuntimeError, e:
        raise RuntimeError('Error running configure on C2html: '+str(e))
      try:
        output  = config.base.Configure.executeShellCommand('cd '+c2htmlDir+';make; make install; make clean', timeout=2500, log = self.framework.log)[0]
      except RuntimeError, e:
        raise RuntimeError('Error running make; make install on C2html: '+str(e))
      fd = file(os.path.join(installDir,'config.args'), 'w')
      fd.write(args)
      fd.close()
      self.framework.actions.addArgument('C2HTML', 'Install', 'Installed c2html into '+installDir)
    self.binDir = os.path.join(installDir, 'bin')
    self.c2html = os.path.join(self.binDir, 'c2html')
    self.addMakeMacro('C2HTML',self.c2html)
    return

  def configureC2html(self):
    '''Determine whether the C2html exist or not'''
    if os.path.exists(os.path.join(self.arch.dir, 'BitKeeper')):
      self.framework.log.write('BitKeeper clone of PETSc, checking for C2html\n')
      self.framework.getExecutable('c2html', getFullPath = 1)
      if hasattr(self.framework, 'c2html'):
        self.c2html = self.framework.c2html
      else:
        self.downLoadC2html()
        
      if not hasattr(self, 'c2html'):
        message = 'See http:/www.mcs.anl.gov/petsc/petsc-2/developers for how\nto obtain C2html\n'
        self.framework.log.write(message)
        raise RuntimeError('Could not install C2html\n'+message)
    else:
      self.framework.log.write("Not BitKeeper clone of PETSc, don't need C2html\n")
    return

  def configure(self):
    self.executeTest(self.configureC2html)
    return

if __name__ == '__main__':
  import config.framework
  import sys
  framework = config.framework.Framework(sys.argv[1:])
  framework.setupLogging(sys.argv[1:])
  framework.children.append(Configure(framework))
  framework.configure()
  framework.dumpSubstitutions()
