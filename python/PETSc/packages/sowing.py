from __future__ import generators
import config.base
import os
import re
import PETSc.package

class Configure(PETSc.package.Package):
  def __init__(self, framework):
    PETSc.package.Package.__init__(self, framework)
    self.download     = ['ftp://ftp.mcs.anl.gov/pub/sowing/sowing.tar.gz']
    return

  def Install(self):
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
    self.getExecutable('pdflatex', getFullPath = 1)
    return

  def buildFortranStubs(self):
    if 'FC' in self.framework.argDB:
      self.framework.log.write('           Running '+self.bfort+' to generate fortran stubs\n')
      try:
        import sys
        (output,error,status) = config.base.Configure.executeShellCommand(sys.executable+' '+os.path.join('maint','generatefortranstubs.py')+' ' +self.bfort,timeout = 15*60.0,log = self.framework.log)
        self.framework.actions.addArgument('PETSc', 'File creation', 'Generated Fortran stubs ')
      except RuntimeError, e:
        raise RuntimeError('*******Error generating Fortran stubs: '+str(e)+'*******\n')
    return

  def configure(self):
    '''Determine whether the Sowing exist or not'''
    if os.path.exists(os.path.join(self.framework.argDB['PETSC_DIR'], 'BitKeeper')):
      self.framework.log.write('BitKeeper clone of PETSc, checking for Sowing\n')
      self.Install()
      self.buildFortranStubs()
    else:
      self.framework.log.write("Not BitKeeper clone of PETSc, don't need Sowing\n")
    return

