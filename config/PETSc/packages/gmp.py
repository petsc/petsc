from __future__ import generators
import user
import config.base
import config.package

#
# DOES NOT CURRENTLY WORK
#
class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.download         = ['http://ftp.mcs.anl.gov/pub/petsc/externalpackages/gmp.tar.gz']
    self.complex          = 0;
    self.double           = 0;
    self.includes         = ['gmp.h','mp_bases.h','fib_table.h','gmpxx.h','randmt.h','longlong.h','trialdivtab.h','gmp-mparam.h','mp.h']
    self.liblist          = [['']]
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
    args.append('--enable-cxx')
      
    args = ' '.join(args)
    fd = file(os.path.join(self.packageDir,self.package), 'w')
    fd.write(args)
    fd.close()
    if self.installNeeded(self.package):
      self.logPrintBox('Configuring GMP; this may take several minutes')
      try:
        output1,err1,ret1  = config.base.Configure.executeShellCommand('cd '+self.packageDir+';ABI=32 ./configure '+args, timeout=900, log = self.framework.log)
      except RuntimeError, e:
        raise RuntimeError('Error running configure on GMP (install manually): '+str(e))
      self.logPrintBox('Compiling GMP; this may take several minutes')
      try:
        output2,err2,ret2  = config.base.Configure.executeShellCommand('cd '+self.packageDir+';make; make install; make clean', timeout=2500, log = self.framework.log)
      except RuntimeError, e:
        raise RuntimeError('Error running make; make install on GMP (install manually): '+str(e))
      self.framework.actions.addArgument('GMP', 'Install', 'Installed GMP into '+self.installDir)
      self.postInstall(output1+err1+output2+err2,'gmp')
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
        raise RuntimeError('Unable to use GMP: ')
      self.popLanguage()
        
    return
  
