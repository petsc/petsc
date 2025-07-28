import config.package

#
# If --download-blis is used WITHOUT --download-f2cblaslapack it is possible that the installed BLIS libraries will NOT be used!
# This is because some automatically detected LAPACK libraries are so intimately connected to their own BLAS they do not utilize
# the other BLAS symbols provided in the link line (that, in this situation, come from BLIS)
#
class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.version   = '2.0'
    self.gitcommit = '2.0-rc0'
    self.download  = ['git://https://github.com/flame/blis.git', 'https://github.com/flame/blis/archive/%s.tar.gz' % self.gitcommit]
    self.functions = ['bli_init']
    self.includes  = ['blis/blis.h']
    self.liblist   = [['libblis.a']]
    self.complex_return = None
    return

  def setupHelp(self, help):
    import nargs
    config.package.Package.setupHelp(self, help)
    help.addArgument(self.PACKAGE,'-download-blis-use-pthreads=<bool>',nargs.ArgBool(None,1,'Use pthreads threading support for '+self.name ))
    help.addArgument(self.PACKAGE,'-download-blis-use-openmp=<bool>',nargs.ArgBool(None,1,'Use OpenMP threading support for '+self.name))
    help.addArgument(self.PACKAGE,'-download-blis-enable-cblas-headers=<bool>',nargs.ArgBool(None,0,'Enable CBLAS headers for '+self.name ))
    help.addArgument(self.PACKAGE,'-download-blis-complex-return=<string>',nargs.ArgString(None,None,'Specify the method of returning complex numbers from blas routines ('+self.name+' supports "gnu" and "intel")'))
    help.addArgument(self.PACKAGE,'-download-blis-confname=<string>',nargs.ArgString(None,'auto','Select blis confname: "auto", "generic", "sandybridge", "haswell", etc.'))
    return

  def configureLibrary(self):
    import os
    config.package.Package.configureLibrary(self)
    if not hasattr(self, 'known64'): self.known64 = 'unknown'
    if self.found:
      try:
        threading_model = re.compile(r'THREADING_MODEL\s*:=\s*(.*)')
        for line in os.path.join(self.directory, 'share', 'blis', 'config.mk'):
            match = threading_model.match(line)
            if match:
              self.threading_model = match.groups()[0]
              self.usesopenmp = 'no'
              self.usespthreads = 'no'
              if self.threading_model == 'openmp':
                self.usesopenmp = 'yes'
              if self.threading_model == 'pthreads':
                self.usespthreads = 'yes'
      except:
        pass

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    self.setCompilers    = framework.require('config.setCompilers',self)
    self.make            = framework.require('config.packages.make', self)
    self.openmp          = framework.require('config.packages.OpenMP',self)
    self.pthread         = framework.require('config.packages.pthread',self)

  def Install(self):
    import os
    self.logPrintBox('Configuring BLIS; this may take several minutes')
    args = ['./configure', '--prefix='+self.installDir]
    try:
      if self.argDB['with-64-bit-blas-indices']:
        args.append('--blas-int-size=64')
        self.known64 = '64'
      else:
        self.known64 = '32'
      threads = []
      if self.argDB['download-blis-use-pthreads'] and self.pthread.found:
          threads.append('pthreads')
          self.usespthreads = 'yes'
      if self.argDB['download-blis-use-openmp'] and self.openmp.found:
          threads.append('openmp')
          self.usesopenmp = 'yes'
      if threads:
        args.append('--enable-threading='+','.join(threads))
      if self.argDB['download-blis-enable-cblas-headers']:
        args.append('--enable-cblas')
      try:
        self.complex_return = self.argDB['download-blis-complex-return']
      except:
        pass
      if self.complex_return:
        args.append('--complex-return=' + self.complex_return)
      with self.Language('C'):
        args.append('CC=' + self.getCompiler())
      with self.Language('Cxx'):
        args.append('CXX=' + self.getCompiler())
      if hasattr(self.compilers, 'FC'):
        with self.Language('FC'):
          args.append('FC=' + self.getCompiler())
      args.append(str(self.argDB['download-blis-confname']))
      config.package.Package.executeShellCommand(args, cwd=self.packageDir, timeout=60, log=self.log)
    except RuntimeError as e:
      raise RuntimeError('Error running configure on BLIS: '+str(e))
    try:
      self.logPrintBox('Compiling and installing BLIS; this may take several minutes')
      config.package.Package.executeShellCommand(self.make.make_jnp_list, cwd=self.packageDir, timeout=500, log=self.log)
      config.package.Package.executeShellCommand(self.make.make_jnp_list + ['install'], cwd=self.packageDir, timeout=30, log=self.log)
    except RuntimeError as e:
      raise RuntimeError('Error running make on BLIS: '+str(e))
    return self.installDir
