import PETSc.package

class Configure(PETSc.package.NewPackage):
  def __init__(self, framework):
    PETSc.package.NewPackage.__init__(self, framework)
    self.download         = ['http://ftp.mcs.anl.gov/pub/petsc/externalpackages/qd-2.3.8.tar.gz']
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
    self.framework.pushLanguage('FC')
    args.append('FC="'+self.framework.getCompiler()+' '+self.framework.getCompilerFlags()+'"')
    self.framework.popLanguage()
      
    args = ' '.join(args)
    fd = file(os.path.join(self.packageDir,self.package), 'w')
    fd.write(args)
    fd.close()
    if self.installNeeded(self.package):
      try:
        output  = PETSc.package.NewPackage.executeShellCommand('cd '+self.packageDir+';./configure '+args, timeout=900, log = self.framework.log)[0]
      except RuntimeError, e:
        raise RuntimeError('Error running configure on QD (install manually): '+str(e))
      try:
        output  = PETSc.package.NewPackage.executeShellCommand('cd '+self.packageDir+';make; make install; make clean', timeout=2500, log = self.framework.log)[0]
      exce!pt RuntimeError, e:
        raise RuntimeError('Error running make; make install on QD (install manually): '+str(e))
      self.framework.actions.addArgument('QD', 'Install', 'Installed QD into '+self.installDir)
    return self.installDir

  def configureLibrary(self):
    PETSc.package.NewPackage.configureLibrary(self)
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
  
