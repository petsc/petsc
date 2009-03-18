from __future__ import generators
import config.base
import os
import re
import PETSc.package

class Configure(PETSc.package.Package):
  def __init__(self, framework):
    PETSc.package.Package.__init__(self, framework)
    self.download         = ['http://ftp.mcs.anl.gov/pub/petsc/externalpackages/sowing-1.1.11-a.tar.gz']
    self.complex          = 1
    self.double           = 0;
    self.requires32bitint = 0;
    return

  def Install(self):
    args = ['--prefix='+self.installDir]
    if not self.framework.argDB['with-batch']:
      self.framework.pushLanguage('C')
      args.append('CC="'+self.framework.getCompiler()+' '+self.framework.getCompilerFlags()+'"')
      args.append('CPP="'+self.framework.getPreprocessor()+'"')
      self.framework.popLanguage()
      if hasattr(self.compilers, 'CXX'):
        self.framework.pushLanguage('Cxx')
        args.append('CXX="'+self.framework.getCompiler()+' '+self.framework.getCompilerFlags()+'"')
        args.append('CPP="'+self.framework.getPreprocessor()+'"')
        self.framework.popLanguage()
      
    args = ' '.join(args)
    fd = file(os.path.join(self.packageDir,self.package), 'w')
    fd.write(args)
    fd.close()
    if self.installNeeded(self.package):
      try:
        output  = config.base.Configure.executeShellCommand('cd '+self.packageDir+';./configure '+args, timeout=900, log = self.framework.log)[0]
      except RuntimeError, e:
        raise RuntimeError('Error running configure on Sowing (install manually): '+str(e))
      try:
        output  = config.base.Configure.executeShellCommand('cd '+self.packageDir+';make; make install; make clean', timeout=2500, log = self.framework.log)[0]
      except RuntimeError, e:
        raise RuntimeError('Error running make; make install on Sowing (install manually): '+str(e))
      self.framework.actions.addArgument('Sowing', 'Install', 'Installed Sowing into '+self.installDir)
    self.binDir   = os.path.join(self.installDir, 'bin')
    self.bfort    = os.path.join(self.binDir, 'bfort')
    self.doctext  = os.path.join(self.binDir, 'doctext')
    self.mapnames = os.path.join(self.binDir, 'mapnames')
    # bug does not install bib2html so use original location if needed
    if os.path.isfile(os.path.join(self.binDir, 'bib2html')):
      self.bib2html = os.path.join(self.binDir, 'bib2html')
    else:
      self.bib2html = os.path.join(self.packageDir,'bin', 'bib2html')
    for prog in [self.bfort, self.doctext, self.mapnames]:
      if not (os.path.isfile(prog) and os.access(prog, os.X_OK)):
        raise RuntimeError('Error in Sowing installation: Could not find '+prog)
      output  = config.base.Configure.executeShellCommand('cp -f '+os.path.join(self.packageDir,'sowing')+' '+self.confDir+'/sowing', timeout=5, log = self.framework.log)[0]
    self.addMakeMacro('BFORT ', self.bfort)
    self.addMakeMacro('DOCTEXT ', self.doctext)
    self.addMakeMacro('MAPNAMES ', self.mapnames)
    self.addMakeMacro('BIB2HTML ', self.bib2html)    
    self.getExecutable('pdflatex', getFullPath = 1)
    return self.installDir

  def buildFortranStubs(self):
    if hasattr(self.compilers, 'FC'):
      if self.framework.argDB['with-batch'] and not hasattr(self,'bfort'):
        self.logPrintBox('Batch build that could not generate bfort, skipping generating Fortran stubs\n \
                          you will need to copy them from some other system (src/fortran/auto)')
      else:     
        self.framework.log.write('           Running '+self.bfort+' to generate fortran stubs\n')
        try:
          import sys
          sys.path.insert(0, os.path.abspath(os.path.join('bin','maint')))
          import generatefortranstubs
          del sys.path[0]
          generatefortranstubs.main(self.bfort)
          self.framework.actions.addArgument('PETSc', 'File creation', 'Generated Fortran stubs')
        except RuntimeError, e:
          raise RuntimeError('*******Error generating Fortran stubs: '+str(e)+'*******\n')
    return

  def alternateConfigureLibrary(self):
    self.checkDownload(1)

  def configure(self):
    '''Determine whether the Sowing exist or not'''

    # If download option is specified always build sowing
    if self.framework.argDB['download-sowing'] == 'ifneeded':
      self.framework.argDB['download-sowing'] = 0

    if self.framework.argDB['download-sowing']:
      PETSc.package.Package.configure(self)
      return

    if self.petscdir.isClone:
      self.framework.logPrint('PETSc clone, checking for Sowing or if it is needed\n')
#      if self.framework.argDB.has_key('with-sowing') and not self.framework.argDB['with-sowing']:
#        self.framework.logPrint('--with-sowing is turned off, skipping sowing')
#        return
      if not hasattr(self.compilers, 'FC'):
        self.framework.logPrint('No Fortran compiler, skipping sowing')
        return

      self.getExecutable('bfort', getFullPath = 1)
      self.getExecutable('doctext', getFullPath = 1)
      self.getExecutable('mapnames', getFullPath = 1)            
      self.getExecutable('bib2html', getFullPath = 1)
      self.getExecutable('pdflatex', getFullPath = 1)
      
      if hasattr(self, 'bfort'):
        self.framework.logPrint('Found bfort, not installing sowing')
      else:
        self.framework.logPrint('Installing bfort')
        self.framework.argDB['download-sowing'] = 1
        PETSc.package.Package.configure(self)
      self.buildFortranStubs()
    else:
      self.framework.logPrint("Not a clone of PETSc, don't need Sowing\n")
    return

