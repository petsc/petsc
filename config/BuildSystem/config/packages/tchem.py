import config.package

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.download          = ['git://https://bitbucket.org/jedbrown/tchem.git']
    self.gitcommit         = '81601d2'
    self.functions         = ['TC_getSrc']
    self.includes          = ['TC_interface.h']
    self.liblist           = [['libtchem.a']]
    return

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    self.compilerFlags = framework.require('config.compilerFlags', self)
    self.mathlib       = framework.require('config.packages.mathlib',self)
    self.deps          = [self.mathlib]
    return

  def Install(self):
    import os, glob

    libDir         = os.path.join(self.installDir, 'lib')
    includeDir     = os.path.join(self.installDir, 'include')
    shareDir       = os.path.join(self.installDir, 'share')

    args = []
    self.pushLanguage('C')
    args.append('CC='+self.getCompiler())
    args.append('CFLAGS='+self.updatePackageCFlags(self.getCompilerFlags()))
    self.popLanguage()
    if hasattr(self.compilers, 'CXX'):
      self.pushLanguage('Cxx')
      args.append('CXX='+self.getCompiler())
      args.append('CXXFLAGS='+self.updatePackageCxxFlags(self.getCompilerFlags()))
      self.popLanguage()

    conffile = os.path.join(self.packageDir, self.package)
    fd = open(conffile, 'w')
    fd.write(' '.join(args))
    fd.close()

    if self.installNeeded(conffile):
      try:
        self.logPrintBox('Configuring TChem')
        output1,err1,ret1  = config.package.Package.executeShellCommand(['./configure'] + args, cwd=self.packageDir, timeout=300, log = self.log)
      except RuntimeError as e:
        raise RuntimeError('Error running configure on TChem: '+str(e))
      try:
        self.logPrintBox('Compiling TChem; this may take several minutes')
        output2,err2,ret2  = config.package.Package.executeShellCommand(['make'], cwd=self.packageDir, timeout=500, log = self.log)
        output2,err2,ret2  = config.package.Package.executeShellCommandSeq([
          ['mkdir', '-p', includeDir, libDir, shareDir],
          ['cp'] + glob.glob(os.path.join(self.packageDir, 'include', 'TC_*')) + [includeDir],
          ['cp'] + glob.glob(os.path.join(self.packageDir, 'lib', 'libtchem*')) + [libDir],
          ['cp', os.path.join(self.packageDir, 'data', 'periodictable.dat'), shareDir],
          ], timeout=500, log = self.log)
      except RuntimeError as e:
        raise RuntimeError('Error running make on TChem: '+str(e))
      self.postInstall(output1+err1+output2+err2,'tchem')
    return self.installDir
