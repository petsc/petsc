#!/usr/bin/env python
from __future__ import generators
import user
import config.base
import os
import PETSc.package

class Configure(PETSc.package.Package):
  def __init__(self, framework):
    PETSc.package.Package.__init__(self, framework)
    self.download  = ['bk://petsc.bkbits.net/blacs-dev']
    self.functions = ['blacs_pinfo']
    self.liblist   = [['libblacs.a']]
    self.includes  = []
    self.libdir    = ''
    self.fc        = 1
    return

  def setupDependencies(self, framework):
    PETSc.package.Package.setupDependencies(self, framework)
    self.mpi        = framework.require('PETSc.packages.MPI',self)
    self.blasLapack = framework.require('PETSc.packages.BlasLapack',self)
    self.deps       = [self.mpi,self.blasLapack]
    return
          
  def Install(self):
    # Get the BLACS directories
    blacsDir   = self.getDir()
    installDir = os.path.join(blacsDir, self.arch.arch)

    # Configure and build BLACS
    if os.path.isfile(os.path.join(blacsDir,'Bmake.Inc')):
      output  = config.base.Configure.executeShellCommand('cd '+blacsDir+'; rm -f Bmake.Inc', timeout=2500, log = self.framework.log)[0]
    g = open(os.path.join(blacsDir,'Bmake.Inc'),'w')
    g.write('SHELL     = /bin/sh\n')
    g.write('COMMLIB   = MPI\n')
    g.write('SENDIS    = -DSndIsLocBlk\n')
    g.write('WHATMPI   = -DUseF77Mpi\n')
    g.write('DEBUGLVL  = -DBlacsDebugLvl=1\n')
    g.write('BLACSdir  = '+blacsDir+'\n')
    g.write('BLACSLIB  = '+os.path.join(installDir,'libblacs.a')+'\n')
    g.write('MPIINCdir = '+self.mpi.include[0]+'\n')
    g.write('MPILIB    = '+self.libraries.toString(self.mpi.lib)+'\n')
    g.write('SYSINC    = -I$(MPIINCdir)\n')
    g.write('BTLIBS    = $(BLACSLIB)  $(MPILIB) \n')
    if self.compilers.fortranManglingDoubleUnderscore:
      blah = 'f77IsF2C'
    elif self.compilers.fortranMangling == 'underscore':
      blah = 'Add_'
    elif self.compilers.fortranMangling == 'capitalize':
      blah = 'UpCase'
    else:
      blah = 'NoChange'
    g.write('INTFACE   = -D'+blah+'\n')
    g.write('DEFS1     = -DSYSINC $(SYSINC) $(INTFACE) $(DEFBSTOP) $(DEFCOMBTOP) $(DEBUGLVL)\n')
    g.write('BLACSDEFS = $(DEFS1) $(SENDIS) $(BUFF) $(TRANSCOMM) $(WHATMPI) $(SYSERRORS)\n')
    self.setCompilers.pushLanguage('FC')  
    g.write('F77       = '+self.setCompilers.getCompiler()+'\n')
    g.write('F77FLAGS  = '+self.setCompilers.getCompilerFlags()+'\n')
    g.write('F77LOADER = '+self.setCompilers.getLinker()+'\n')      
    g.write('F77LOADFLAGS ='+self.setCompilers.getLinkerFlags()+'\n')
    self.setCompilers.popLanguage()     
    self.setCompilers.pushLanguage('C')
    g.write('CC          = '+self.setCompilers.getCompiler()+'\n')
    g.write('CCFLAGS     = '+self.setCompilers.getCompilerFlags()+'\n')      
    g.write('CCLOADER    = '+self.setCompilers.getLinker()+'\n')
    g.write('CCLOADFLAGS = '+self.setCompilers.getLinkerFlags()+'\n')
    self.setCompilers.popLanguage()
    g.write('ARCH        = '+self.setCompilers.AR+'\n')
    g.write('ARCHFLAGS   = '+self.setCompilers.AR_FLAGS+'\n')    
    g.write('RANLIB      = '+self.setCompilers.RANLIB+'\n')    
    g.close()
    if not os.path.isdir(installDir):
      os.mkdir(installDir)
    if not os.path.isfile(os.path.join(installDir,'Bmake.Inc')) or not (self.getChecksum(os.path.join(installDir,'Bmake.Inc')) == self.getChecksum(os.path.join(blacsDir,'Bmake.Inc'))):
      self.framework.log.write('Have to rebuild blacs, Bmake.Inc != '+installDir+'/Bmake.Inc\n')
      try:
        self.logPrintBox('Compiling Blacs; this may take several minutes')
        output  = config.base.Configure.executeShellCommand('cd '+os.path.join(blacsDir,'SRC','MPI')+';make', timeout=2500, log = self.framework.log)[0]
      except RuntimeError, e:
        raise RuntimeError('Error running make on BLACS: '+str(e))
    else:
      self.framework.log.write('Do NOT need to compile BLACS downloaded libraries\n')
    if not os.path.isfile(os.path.join(installDir,'libblacs.a')):
      self.framework.log.write('Error running make on BLACS   ******(libraries not installed)*******\n')
      self.framework.log.write('********Output of running make on BLACS follows *******\n')        
      self.framework.log.write(output)
      self.framework.log.write('********End of Output of running make on BLACS *******\n')
      raise RuntimeError('Error running make on BLACS, libraries not installed')
    
    output = config.base.Configure.executeShellCommand('cp -f '+os.path.join(blacsDir,'Bmake.Inc')+' '+installDir, timeout=5, log = self.framework.log)[0]
    self.framework.actions.addArgument('blacs', 'Install', 'Installed blacs into '+installDir)
    return self.getDir()

  def checkLib(self,lib,func,mangle,otherLibs = []):
    oldLibs = self.compilers.LIBS
    found = self.libraries.check(lib,func, otherLibs = otherLibs+self.mpi.lib+self.blasLapack.lib+self.compilers.flibs,fortranMangle=mangle)
    self.compilers.LIBS=oldLibs
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
      #if self.executeTest(self.libraries.check,[lib,self.functions],{'otherLibs' : libs}):
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
