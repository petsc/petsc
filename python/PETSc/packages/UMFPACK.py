#!/usr/bin/env python
from __future__ import generators
import user
import config.base
import os
import PETSc.package

class Configure(PETSc.package.Package):
  def __init__(self, framework):
    PETSc.package.Package.__init__(self, framework)
    self.blasLapack   = self.framework.require('PETSc.packages.BlasLapack',self)
    self.download     = ['ftp://ftp.mcs.anl.gov/pub/petsc/externalpackages/UMFPACKv4.3.tar.gz']
    self.deps         = [self.blasLapack]
    self.functions    = ['umfpack_di_report_info'] 
    self.includes     = ['umfpack.h']
    self.libdir       = 'UMFPACK/Lib'
    self.includedir   = 'UMFPACK/Include'
    return

  def generateLibList(self,dir):  #dir = ~UMFPACKv4.3/UMFPACK/Lib
    libs    = ['libumfpack.a']
    alllibs = []
    for l in libs:
      alllibs.append(os.path.join(dir, l))
    # append libamd.a  
    (dirTmp,dummy) = os.path.split(dir)
    (dirTmp,dummy) = os.path.split(dirTmp) #dirTmp = ~UMFPACKv4.3
    alllibs.append(os.path.join(dirTmp, 'AMD/Lib/libamd.a'))
      
    import config.setCompilers
    self.framework.pushLanguage('C')
    self.framework.popLanguage()    
    return [alllibs]
          
  def Install(self):
    # Get the UMFPACK directories
    umfpackDir = self.getDir()
    installDir = os.path.join(umfpackDir, self.arch.arch)
    self.framework.log.write('umfpackDir = '+umfpackDir+' installDir '+installDir+'\n')
    # Configure and Build UMFPACK
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
    args.append('--with-blas="'+self.libraries.toString(self.blasLapack.dlib)+'"')        
    args = ' '.join(args)

    try:
      fd      = file(os.path.join(installDir,'config.args'))
      oldargs = fd.readline()
      fd.close()
    except:
      oldargs = ''
    if not oldargs == args:
      self.framework.log.write('Have to rebuild UMFPACK oldargs = '+oldargs+' new args '+args+'\n')
      try:
        self.logPrintBox('Compiling umfpack; this may take several minutes')
        output  = config.base.Configure.executeShellCommand('cd '+umfpackDir+'; UMFPACK_INSTALL_DIR='+installDir+';export UMFPACK_INSTALL_DIR; cp -r UMFPACK '+self.arch.arch+'/.; cp -r AMD '+self.arch.arch+'/.; cd '+self.arch.arch+'/UMFPACK; make lib; make clean', timeout=2500, log = self.framework.log)[0]
      except RuntimeError, e:
        raise RuntimeError('Error running make on UMFPACK: '+str(e))
      if not os.path.isdir(os.path.join(installDir,self.libdir)):
        self.framework.log.write('Error running make on UMFPACK   ******(libraries not installed)*******\n')
        self.framework.log.write('********Output of running make on UMFPACK follows *******\n')        
        self.framework.log.write(output)
        self.framework.log.write('********End of Output of running make on UMFPACK *******\n')
        raise RuntimeError('Error running make on UMFPACK, libraries not installed')
      
      fd = file(os.path.join(installDir,'config.args'), 'w')
      fd.write(args)
      fd.close()

      self.framework.actions.addArgument(self.PACKAGE, 'Install', 'Installed UMFPACK into '+installDir)
    return self.getDir()

if __name__ == '__main__':
  import config.framework
  import sys
  framework = config.framework.Framework(sys.argv[1:])
  framework.setup()
  framework.addChild(Configure(framework))
  framework.configure()
  framework.dumpSubstitutions()
  

