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
    self.download     = ['http://crd.lbl.gov/~xiaoye/SuperLU/superlu_3.0.tar.gz']
    self.deps         = [self.blasLapack]
    self.functions    = ['set_default_options']
    self.includes     = ['dsp_defs.h']
    self.libdir       = ''
    self.includedir   = 'SRC'
    return

  def generateLibList(self,dir):
    '''Normally the one in package.py is used'''
    alllibs = []
    alllibs.append(os.path.join(dir,'superlu_linux.a'))
    import config.setCompilers
    self.framework.pushLanguage('C')
    self.framework.popLanguage()    
    return alllibs
  
  def Install(self):
    # Get the SUPERLU directories
    superluDir = self.getDir()
    installDir = os.path.join(superluDir, self.arch.arch)

    # Configure and Build SUPERLU
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
      self.framework.log.write('Have to rebuild SUPERLU oldargs = '+oldargs+' new args '+args+'\n')
      try:
        self.logPrint("Compiling superlu; this may take several minutes\n", debugSection='screen')
        output = config.base.Configure.executeShellCommand('cd '+superluDir+'; SUPERLU_INSTALL_DIR='+installDir+'; export SUPERLU_INSTALL_DIR; cp MAKE_INC/make.inc .; make install lib; mv *.a '+os.path.join(installDir,self.libdir)+'; mkdir '+os.path.join(installDir,self.includedir)+'; cp SRC/*.h '+os.path.join(installDir,self.includedir)+'/.', timeout=2500, log = self.framework.log)[0]
      except RuntimeError, e:
        raise RuntimeError('Error running make on SUPERLU: '+str(e))
      if not os.path.isdir(os.path.join(installDir,self.libdir)):
        self.framework.log.write('Error running make on SUPERLU   ******(libraries not installed)*******\n')
        self.framework.log.write('********Output of running make on SUPERLU follows *******\n')        
        self.framework.log.write(output)
        self.framework.log.write('********End of Output of running make on SUPERLU *******\n')
        raise RuntimeError('Error running make on SUPERLU, libraries not installed')
      
      fd = file(os.path.join(installDir,'config.args'), 'w')
      fd.write(args)
      fd.close()

      self.framework.actions.addArgument(self.PACKAGE, 'Install', 'Installed SUPERLU into '+installDir)
    return self.getDir()
  
if __name__ == '__main__':
  import config.framework
  import sys
  framework = config.framework.Framework(sys.argv[1:])
  framework.setup()
  framework.addChild(Configure(framework))
  framework.configure()
  framework.dumpSubstitutions()
