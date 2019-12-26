import config.package

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.version   = '0.6.1'
    self.gitcommit = self.version
    self.download  = ['git://https://github.com/flame/blis.git', 'https://github.com/flame/blis/archive/%s.tar.gz' % self.version]
    self.functions = ['bli_init']
    self.includes  = ['blis/blis.h']
    self.liblist   = [['libblis.a']]
    return

  def setupHelp(self, help):
    config.package.Package.setupHelp(self,help)
    import nargs
    help.addArgument('BLIS', '-download-blis-threading', nargs.ArgString(None, 'no', regExp='^(?:openmp|pthreads|no)$', help='BLIS threading model [openmp|pthreads|no]'))

  def configureLibrary(self):
    import os
    config.package.Package.configureLibrary(self)
    if not hasattr(self, 'known64'): self.known64 = 'unknown'
    if not hasattr(self, 'usesopenmp'): self.usesopenmp = 'unknown'
    if self.found:
      try:
        threading_model = re.compile(r'THREADING_MODEL\s*:=\s*(.*)')
        for line in os.path.join(self.directory, 'share', 'blis', 'config.mk'):
            match = threading_model.match(line)
            if match:
              self.threading_model = match.groups()[0]
              if self.threading_model == 'openmp':
                self.usesopenmp = 'yes'
              else:
                self.usesopenmp = 'no'
      except:
        pass

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    self.setCompilers    = framework.require('config.setCompilers',self)
    self.make            = framework.require('config.packages.make', self)

  def Install(self):
    import os
    with self.Language('C'):
      cc = self.setCompilers.getCompiler()
    try:
      self.logPrintBox('Configuring BLIS; this may take several minutes')
      args = ['./configure', '--prefix='+self.installDir]
      if self.argDB['with-64-bit-blas-indices']:
        args.append('--blas-int-size=64')
        self.known64 = '64'
      else:
        self.known64 = '32'
      args.append('--enable-threading=' + self.argDB.get('with-download-blis-threading', 'no'))
      args.append('CC=' + cc)
      args.append('auto')
      config.package.Package.executeShellCommand(args, cwd=self.packageDir, timeout=60, log=self.log)
    except RuntimeError as e:
      raise RuntimeError('Error running configure on BLIS: '+str(e))
    try:
      self.logPrintBox('Compiling and installing BLIS; this may take several minutes')
      config.package.Package.executeShellCommand(self.make.make_jnp_list, cwd=self.packageDir, timeout=500, log=self.log)
      config.package.Package.executeShellCommand(self.make.make_sudo_list + ['install'], cwd=self.packageDir, timeout=30, log=self.log)
    except RuntimeError as e:
      raise RuntimeError('Error running make on BLIS: '+str(e))
    return self.installDir


