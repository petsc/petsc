from __future__ import generators
import user
import config.base
import config.package

#
#   This and gmp.py and mpack.py CANNOT be used by PETSc because QD does not support
#       !dd_real, casts to int, bool, double from dd_real, dd_real++
#   and these cannot easily be fixed without a lot of work on QD
#
class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.download         = ['http://ftp.mcs.anl.gov/pub/petsc/externalpackages/qd-2.3.9.tar.gz']
    self.complex          = 0;
    self.double           = 0;
    self.includes         = ['qd/dd_real.h']
    self.liblist          = [['libqd.a','libqd_f_main.a','libqdmod.a']]
    self.requires32bitint = 0;
    self.cxx              = 1    # requires 
    self.fc               = 0    # 1 means requires fortran    return

  def Install(self):
    import os
    args = ['--prefix='+self.installDir]
    self.framework.pushLanguage('C')
    args.append('CC="'+self.framework.getCompiler()+' '+self.framework.getCompilerFlags()+'"')
    self.framework.popLanguage()
    self.framework.pushLanguage('Cxx')
    args.append('CXX="'+self.framework.getCompiler()+' '+self.framework.getCompilerFlags()+'"')
    self.framework.popLanguage()
      
    args = ' '.join(args)
    fd = file(os.path.join(self.packageDir,self.package), 'w')
    fd.write(args)
    fd.close()
    if self.installNeeded(self.package):
      self.logPrintBox('Configuring QD; this may take several minutes')
      try:
        output1,err1,ret1  = config.base.Configure.executeShellCommand('cd '+self.packageDir+';./configure '+args, timeout=900, log = self.framework.log)
      except RuntimeError, e:
        raise RuntimeError('Error running configure on QD (install manually): '+str(e))
      self.logPrintBox('Compiling QD; this may take several minutes')
      try:
        output2,err2,ret2  = config.base.Configure.executeShellCommand('cd '+self.packageDir+';make; make install; make clean', timeout=2500, log = self.framework.log)
      except RuntimeError, e:
        raise RuntimeError('Error running make; make install on QD (install manually): '+str(e))
      self.framework.actions.addArgument('QD', 'Install', 'Installed QD into '+self.installDir)
      self.postInstall(output1+err1+output2+err2,'qd')
    return self.installDir

  def configureLibrary(self):
    config.package.Package.configureLibrary(self)
    if self.found:
      self.pushLanguage('Cxx')      
      oldFlags = self.compilers.CPPFLAGS
      oldLibs  = self.compilers.LIBS
      self.compilers.CPPFLAGS += ' '+self.headers.toString(self.include)
      self.compilers.LIBS = self.libraries.toString(self.lib)+' '+self.compilers.LIBS
      if not self.checkLink('#include <qd/dd_real.h>\n', 'dd_real a = 1.0, b; b = sqrt(a);\n'):
        self.compilers.CPPFLAGS = oldFlags
        self.compilers.LIBS = oldLibs
        self.popLanguage()
        raise RuntimeError('Unable to use QD: ')
      self.popLanguage()
      self.addDefine('USE_QD_DD', 1)      
        
    return
  
