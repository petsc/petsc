#!/usr/bin/env python
from __future__ import generators
import user
import config.base
import os
import PETSc.package

class Configure(PETSc.package.Package):
  def __init__(self, framework):
    PETSc.package.Package.__init__(self, framework)
    self.mpi           = self.framework.require('PETSc.packages.MPI',self)
    self.blasLapack    = self.framework.require('PETSc.packages.BlasLapack',self)
    self.blacs         = self.framework.require('PETSc.packages.blacs',self)
    self.download      = ['http://www.netlib.org/scalapack/scalapack.tgz']
    self.deps          = [self.blacs,self.mpi,self.blasLapack]
    self.functions     = ['ssytrd']
    self.includes      = []
    self.libdir        = ''
    self.liblist       = [['libscalapack.a']]
    return

  def Install(self):
    # Get the SCALAPACK directories
    scalapackDir = self.getDir()
    installDir   = os.path.join(scalapackDir, self.arch.arch)

    # Configure and build SCALAPACK
    g = open(os.path.join(scalapackDir,'SLmake.inc'),'w')
    g.write('SHELL        = /bin/sh\n')
    g.write('home         = '+self.getDir()+'\n')
    g.write('USEMPI       = -DUsingMpiBlacs\n')
    g.write('SENDIS       = -DSndIsLocBlk\n')
    g.write('WHATMPI      = -DUseF77Mpi\n')
    g.write('BLACSDBGLVL  = -DBlacsDebugLvl=1\n')
    g.write('BLACSLIB     = '+self.libraries.toString(self.blacs.lib)+'\n') 
    g.write('SMPLIB       = '+self.libraries.toString(self.mpi.lib)+'\n')
    g.write('SCALAPACKLIB = '+os.path.join('$(home)',self.arch.arch,'libscalapack.a')+' \n')
    g.write('CBLACSLIB    = $(BLACSCINIT) $(BLACSLIB) $(BLACSCINIT)\n')
    g.write('FBLACSLIB    = $(BLACSFINIT) $(BLACSLIB) $(BLACSFINIT)\n')
    if self.compilers.fortranManglingDoubleUnderscore:
      blah = 'f77IsF2C'
    elif self.compilers.fortranMangling == 'underscore':
      blah = 'Add_'
    elif self.compilers.fortranMangling == 'capitalize':
      blah = 'UpCase'
    else:
      blah = 'NoChange'
    g.write('CDEFS        =-D'+blah+' -DUsingMpiBlacs\n')
    g.write('PBLASdir     = $(home)/PBLAS\n')
    g.write('SRCdir       = $(home)/SRC\n')
    g.write('TOOLSdir     = $(home)/TOOLS\n')
    g.write('REDISTdir    = $(home)/REDIST\n')
    self.setcompilers.pushLanguage('FC')  
    g.write('F77          = '+self.setcompilers.getCompiler()+'\n')
    g.write('F77FLAGS     = '+self.setcompilers.getCompilerFlags()+'\n')
    g.write('F77LOADER    = '+self.setcompilers.getLinker()+'\n')      
    g.write('F77LOADFLAGS = '+self.setcompilers.getLinkerFlags()+'\n')
    self.setcompilers.popLanguage()
    self.setcompilers.pushLanguage('C')
    g.write('CC           = '+self.setcompilers.getCompiler()+'\n')
    g.write('CCFLAGS      = '+self.setcompilers.getCompilerFlags()+'\n')      
    g.write('CCLOADER     = '+self.setcompilers.getLinker()+'\n')
    g.write('CCLOADFLAGS  = '+self.setcompilers.getLinkerFlags()+'\n')
    self.setcompilers.popLanguage()
    g.write('ARCH         = '+self.setcompilers.AR+'\n')
    g.write('ARCHFLAGS    = '+self.setcompilers.AR_FLAGS+'\n')    
    g.write('RANLIB       = '+self.setcompilers.RANLIB+'\n')    
    g.close()
    if not os.path.isdir(installDir):
      os.mkdir(installDir)
    if not os.path.isfile(os.path.join(installDir,'SLmake.inc')) or not (self.getChecksum(os.path.join(installDir,'SLmake.inc')) == self.getChecksum(os.path.join(scalapackDir,'SLmake.inc'))):
      self.framework.log.write('Have to rebuild SCALAPACK, SLmake.inc != '+installDir+'/SLmake.inc\n')
      try:
        output  = config.base.Configure.executeShellCommand('cd '+scalapackDir+';make cleanlib', timeout=2500, log = self.framework.log)[0]
      except RuntimeError, e:
        pass
      try:
        self.logPrintBox('Compiling Scalapack; this may take several minutes')
        output  = config.base.Configure.executeShellCommand('cd '+scalapackDir+';make', timeout=2500, log = self.framework.log)[0]
      except RuntimeError, e:
        raise RuntimeError('Error running make on SCALAPACK: '+str(e))
    else:
      self.framework.log.write('Did not need to compile downloaded SCALAPACK\n')
    if not os.path.isfile(os.path.join(installDir,'libscalapack.a')):
      self.framework.log.write('Error running make on SCALAPACK   ******(libraries not installed)*******\n')
      self.framework.log.write('********Output of running make on SCALAPACK follows *******\n')        
      self.framework.log.write(output)
      self.framework.log.write('********End of Output of running make on SCALAPACK *******\n')
      raise RuntimeError('Error running make on SCALAPACK, libraries not installed')
    
    output  = config.base.Configure.executeShellCommand('cp -f '+os.path.join(scalapackDir,'SLmake.inc')+' '+installDir, timeout=5, log = self.
framework.log)[0]
    self.framework.actions.addArgument('scalapack', 'Install', 'Installed scalapack into '+installDir)
    return self.getDir()

  def checkLib(self,lib,func,mangle,otherLibs = []):
    oldLibs = self.framework.argDB['LIBS']
    found = self.libraries.check(lib,func, otherLibs = otherLibs+self.mpi.lib+self.blasLapack.lib+self.compilers.flibs,fortranMangle=mangle)
    self.framework.argDB['LIBS']=oldLibs
    if found:
      self.framework.log.write('Found function '+str(func)+' in '+str(lib)+'\n')
    return found
  
  def configureLibrary(self): #almost same as package.py/configureLibrary()!
    '''Find an installation ando check if it can work with PETSc'''
    self.framework.log.write('==================================================================================\n')
    self.framework.log.write('Checking for a functional '+self.name+'\n')
    foundLibrary = 0
    foundHeader  = 0

    # get any libraries and includes we depend on
    libs         = []
    incls        = []
    for l in self.deps:
      if hasattr(l,'dlib'):    libs  += l.dlib
      if hasattr(l,self.includedir): incls += l.include
      
    for location, lib,incl in self.generateGuesses():
      if not isinstance(lib, list): lib = [lib]
      if not isinstance(incl, list): incl = [incl]
      self.framework.log.write('Checking for library '+location+': '+str(lib)+'\n')
      if self.executeTest(self.checkLib,[lib,self.functions,1]):     
        self.lib = lib
        self.framework.log.write('Checking for headers '+location+': '+str(incl)+'\n')
        if (not self.includes) or self.executeTest(self.libraries.checkInclude, [incl, self.includes],{'otherIncludes' : incls}):
          self.include = incl
          self.found   = 1
          self.dlib    = self.lib+libs
          self.framework.packages.append(self) 
          break
    if not self.found:
      raise RuntimeError('Could not find a functional '+self.name+'\n')

if __name__ == '__main__':
  import config.framework
  import sys
  framework = config.framework.Framework(sys.argv[1:])
  framework.setup()
  framework.addChild(Configure(framework))
  framework.configure()
  framework.dumpSubstitutions()
