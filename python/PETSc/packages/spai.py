#!/usr/bin/env python
from __future__ import generators
import user
import config.base
import PETSc.package

import re
import os

class Configure(PETSc.package.Package):
  def __init__(self, framework):
    PETSc.package.Package.__init__(self, framework)
    self.download  = ['ftp://ftp.mcs.anl.gov/pub/petsc/externalpackages/spai_3.0-mar-06.tar.gz']
    self.functions = ['bspai']
    self.includes  = ['spai.h']
    self.liblist   = [['libspai.a']]
    # SPAI include files are in the lib directory
    self.includedir = 'lib'
    return

  def setupDependencies(self, framework):
    PETSc.package.Package.setupDependencies(self, framework)
    self.mpi        = framework.require('config.packages.MPI',self)
    self.blasLapack = framework.require('config.packages.BlasLapack',self)
    self.deps       = [self.mpi,self.blasLapack]
    return

  def Install(self):
    spaiDir = self.getDir()
    installDir = os.path.join(spaiDir, self.arch.arch)
    if not os.path.isdir(os.path.join(installDir,'lib')):
      os.mkdir(os.path.join(installDir,'lib'))      
    self.framework.pushLanguage('C')
    if self.compilers.fortranMangling == 'underscore':
      FTNOPT = ''
    elif self.compilers.fortranMangling == 'capitalize':
      FTNOPT = ''
    else:
      FTNOPT = '-DSP2'
    args = 'CC = '+self.framework.getCompiler()+'\nCFLAGS = -DMPI '+FTNOPT+' '+self.framework.getCompilerFlags()+' '+self.headers.toString(self.mpi.include)+'\n'
    args = args+'AR         = '+self.setCompilers.AR+'\n'
    args = args+'ARFLAGS    = '+self.setCompilers.AR_FLAGS+'\n'
                                  
    self.framework.popLanguage()
    try:
      fd      = file(os.path.join(installDir,'Makefile.in'))
      oldargs = fd.readline()
      fd.close()
    except:
      oldargs = ''
    if not oldargs == args:
      self.framework.log.write('Have to rebuild Spai oldargs = '+oldargs+'\n new args ='+args+'\n')
      self.logPrintBox('Configuring and compiling Spai; this may take several minutes')
      fd = file(os.path.join(installDir,'Makefile.in'),'w')
      fd.write(args)
      fd.close()
      fd = file(os.path.join(spaiDir,'lib','Makefile.in'),'w')
      fd.write(args)
      fd.close()
      output  = config.base.Configure.executeShellCommand('cd '+os.path.join(spaiDir,'lib')+'; make ; mv libspai.a '+os.path.join(installDir,'lib','libspai.a'),timeout=250, log = self.framework.log)[0]
      output  = config.base.Configure.executeShellCommand('cd '+os.path.join(spaiDir,'lib')+'; cp *.h '+os.path.join(installDir,'lib'),timeout=250, log = self.framework.log)[0]      
      try:
        output  = config.base.Configure.executeShellCommand(self.setCompilers.RANLIB+' '+os.path.join(installDir,'lib')+'/libspai.a', timeout=250, log = self.framework.log)[0]
      except RuntimeError, e:
        raise RuntimeError('Error running ranlib on SPAI libraries: '+str(e))
        
    return spaiDir


  def configureLibrary(self):
    '''Calls the regular package configureLibrary and then does an additional test needed by SPAI'''
    '''Normally you do not need to provide this method'''
    if self.blasLapack.f2c:
      raise RuntimeError('SPAI requires a COMPLETE BLAS and LAPACK, it cannot be used with --download-c-blas-lapack=1 \nUse --download-f-blas-lapack option instead.')
    # SPAI requires dormqr() LAPACK routine
    if not self.blasLapack.checkForRoutine('dormqr'): 
      raise RuntimeError('SPAI requires the LAPACK routine dormqr(), the current Lapack libraries '+str(self.blasLapack.lib)+' does not have it\nTry using --download-f-blas-lapack=1 option \nIf you are using the IBM ESSL library, it does not contain this function.')
    self.framework.log.write('Found dormqr() in Lapack library as needed by SPAI\n')
    PETSc.package.Package.configureLibrary(self)
    return
  
if __name__ == '__main__':
  import config.framework
  import sys
  framework = config.framework.Framework(sys.argv[1:])
  framework.setup()
  framework.addChild(Configure(framework))
  framework.configure()
  framework.dumpSubstitutions()
