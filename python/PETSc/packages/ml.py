#!/usr/bin/env python
from __future__ import generators
import user
import config.base
import os
import PETSc.package

class Configure(PETSc.package.Package):
  def __init__(self, framework):
    PETSc.package.Package.__init__(self, framework)
    self.mpi          = self.framework.require('PETSc.packages.MPI',self)
    self.blasLapack   = self.framework.require('PETSc.packages.BlasLapack',self)
    self.download     = ['ftp://ftp.mcs.anl.gov/pub/petsc/foobar/ml.tar.gz']
    self.deps         = [self.mpi,self.blasLapack]
    self.functions    = ['ML_Set_PrintLevel']
    self.includes     = ['ml_include.h']
    return

  def generateLibList(self,dir):
    libs = ['libml.a']
    alllibs = []
    for l in libs:
      alllibs.append(os.path.join(dir,l))
    import config.setCompilers
    self.framework.pushLanguage('C')
    self.framework.popLanguage()    
    return alllibs
          
        
  def Install(self):
    if not os.path.isfile(os.path.expanduser(os.path.join('~','.ml_license'))):
      print "**************************************************************************************************"
      print "You must register to use ml at http://software.sandia.gov/trilinos/downloads.html"
      print "    Once you have registered, configure will continue and download and install ml for you"
      print "**************************************************************************************************"
      fd = open(os.path.expanduser(os.path.join('~','.ml_license')),'w')
      fd.close()
      
    # Get the ML directories
    mlDir = self.getDir()
    installDir  = os.path.join(mlDir, self.arch.arch)
    # Configure and Build ML
    self.framework.pushLanguage('C')
    args = ['--prefix='+installDir, '--with-CC="'+self.framework.getCompiler()+' '+self.framework.getCompilerFlags()+'"']
    self.framework.popLanguage()
    if 'CXX' in self.framework.argDB:
      self.framework.pushLanguage('Cxx')
      args.append('--with-CXX="'+self.framework.getCompiler()+' '+self.framework.getCompilerFlags()+'"')
      self.framework.popLanguage()
    if 'FC' in self.framework.argDB:
      self.framework.pushLanguage('FC')
      args.append('--with-F77="'+self.framework.getCompiler()+' '+self.framework.getCompilerFlags()+'"')
      self.framework.popLanguage()
    if self.mpi.include:
      if len(self.mpi.include) > 1:
        raise RuntimeError("ml assumes there is a single MPI include directory")
      args.append('--with-mpi-include="'+self.mpi.include[0].replace('-I','')+'"')
    libdirs = []
    for l in self.mpi.lib:
      ll = os.path.dirname(l)
      libdirs.append(ll)
    libdirs = ' '.join(libdirs)
    args.append('--with-mpi-lib-dirs="'+libdirs+'"')
    libs = []
    for l in self.mpi.lib:
      ll = os.path.basename(l)
      libs.append(ll[3:-2])
    libs = ' '.join(libs)
    args.append('--with-mpi-libs="'+libs+'"')
    args.append('--with-blas="'+self.libraries.toString(self.blasLapack.dlib)+'"')        
    args = ' '.join(args)

    try:
      fd      = file(os.path.join(installDir,'config.args'))
      oldargs = fd.readline()
      fd.close()
    except:
      oldargs = ''
    if not oldargs == args:
      self.framework.log.write('Have to rebuild ML oldargs = '+oldargs+' new args '+args+'\n')
      try:
        self.logPrint("Configuring ml; this may take several minutes\n", debugSection='screen')
        output  = config.base.Configure.executeShellCommand('cd '+mlDir+'; ./configure '+args+' --disable-epetra --disable-aztecoo', timeout=900, log = self.framework.log)[0]
      except RuntimeError, e:
        raise RuntimeError('Error running configure on ML: '+str(e))
      try:
        self.logPrint("Compiling ml; this may take several minutes\n", debugSection='screen')
        output  = config.base.Configure.executeShellCommand('cd '+mlDir+'; ML_INSTALL_DIR='+installDir+'; export ML_INSTALL_DIR; make; make install', timeout=2500, log = self.framework.log)[0]
      except RuntimeError, e:
        raise RuntimeError('Error running make on ML: '+str(e))
      if not os.path.isdir(os.path.join(installDir,'lib')):
        self.framework.log.write('Error running make on ML   ******(libraries not installed)*******\n')
        self.framework.log.write('********Output of running make on ML follows *******\n')        
        self.framework.log.write(output)
        self.framework.log.write('********End of Output of running make on ML *******\n')
        raise RuntimeError('Error running make on ML, libraries not installed')
      
      fd = file(os.path.join(installDir,'config.args'), 'w')
      fd.write(args)
      fd.close()

      self.framework.actions.addArgument(self.PACKAGE, 'Install', 'Installed ML into '+installDir)
    return self.getDir()
  
if __name__ == '__main__':
  import config.framework
  import sys
  framework = config.framework.Framework(sys.argv[1:])
  framework.setup()
  framework.addChild(Configure(framework))
  framework.configure()
  framework.dumpSubstitutions()
