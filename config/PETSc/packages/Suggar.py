#!/usr/bin/env python
from __future__ import generators
import user
import config.base
import os
import PETSc.package

class Configure(PETSc.package.Package):
  def __init__(self, framework):
    PETSc.package.Package.__init__(self, framework)
    self.download     = ['Not available for download: use --download-Suggar=Suggar.tar.gz']
    self.functions    = ['ctkSortAllDonorsInGrid']
    self.machinename  = 'linux'
    self.liblist      = [['libsuggar_3d_'+self.machinename+'.a']]
    return

  def setupDependencies(self, framework):
    PETSc.package.Package.setupDependencies(self, framework)
    self.mpi        = framework.require('config.packages.MPI',self)
    self.expat      = framework.require('PETSc.packages.expat',self)
    self.cproto     = framework.require('PETSc.packages.cproto',self)
    self.p3dlib     = framework.require('PETSc.packages.P3Dlib',self)
    self.deps       = [self.p3dlib,self.expat,self.mpi]
    return

  def Install(self):

    self.framework.pushLanguage('C')
    g = open(os.path.join(self.packageDir,'src','FLAGS.local'),'w')
    g.write('MPICC ='+self.framework.getCompiler()+'\n')
    g.write('CFLAGS ='+self.framework.getCompilerFlags()+'\n')
    g.write('CPROTO = '+self.cproto.cproto+' -D__THROW= -D_STDARG_H ${TRACEMEM} -I..\n')
    g.write('P3DLIB_DIR = '+self.libraries.toString(self.p3dlib.lib)+'\n')
    g.write('P3DINC_DIR = '+self.headers.toString(self.p3dlib.include)+'\n')
    g.write('EXPATLIB_DIR = '+self.libraries.toString(self.expat.lib)+'\n')
    g.write('EXPATINC_DIR = '+self.headers.toString(self.expat.include)+'\n')
    g.write('MACHINE = '+self.machinename+'\n')    

    g.close()
    self.framework.popLanguage()

    if self.installNeeded(os.path.join('src','FLAGS.local')):
      try:
        self.logPrintBox('Compiling SUGGAR; this may take several minutes')
        output  = config.base.Configure.executeShellCommand('cd '+self.packageDir+'/src; make makedirs libs', timeout=2500, log = self.framework.log)[0]
      except RuntimeError, e:
        raise RuntimeError('Error running make on SUGGAR: '+str(e))
      output  = config.base.Configure.executeShellCommand('mv -f '+os.path.join(self.packageDir,'bin','libsuggar*')+' '+os.path.join(self.installDir,'lib'), timeout=5, log = self.framework.log)[0]      
                          
      self.checkInstall(output,os.path.join('src','FLAGS.local'))
    return self.installDir

if __name__ == '__main__':
  import config.framework
  import sys
  framework = config.framework.Framework(sys.argv[1:])
  framework.setupLogging(framework.clArgs)
  framework.children.append(Configure(framework))
  framework.configure()
  framework.dumpSubstitutions()
