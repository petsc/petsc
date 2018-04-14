import config.package

#    We do not use CMAKE for OpenBLAS the cmake for OpenBLAS
#       does not have an install rule https://github.com/xianyi/OpenBLAS/issues/957
#       fails on mac due to argument list too long https://github.com/xianyi/OpenBLAS/issues/977
#       does not support 64 bit integers with INTERFACE64


class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.gitcommit        = 'v0.2.19'
    self.download         = ['git://https://github.com/xianyi/OpenBLAS.git','https://github.com/xianyi/OpenBLAS/archive/'+self.gitcommit+'.tar.gz']
    self.precisions       = ['single','double']
    self.fc               = 1
    self.installwithbatch = 1

  def setupHelp(self, help):
    config.package.Package.setupHelp(self,help)
    import nargs
    help.addArgument('OpenBLAS', '-download-openblas-64-bit-blas-indices', nargs.ArgBool(None, 0, 'Use 64 bit integers for OpenBLAS'))
    help.addArgument('OpenBLAS', '-download-openblas-make-options=<options>', nargs.Arg(None, None, 'additional options for building OpenBLAS'))
    return

  def configureLibrary(self):
    import os
    config.package.Package.configureLibrary(self)
    if self.found:
      self.libDir = os.path.join(self.directory,'lib')
    return

  def Install(self):
    import os

    if not hasattr(self.compilers, 'FC'):
      raise RuntimeError('Cannot request OpenBLAS without Fortran compiler, use --download-f2cblaslapack intead')

    cmdline = 'CC='+self.compilers.CC+' FC='+self.compilers.FC
    if self.argDB['download-openblas-64-bit-blas-indices']:
      cmdline += " INTERFACE64=1 "
    if 'download-openblas-make-options' in self.argDB and self.argDB['download-openblas-make-options']:
      cmdline+=" "+self.argDB['download-openblas-make-options']

    libdir = self.libDir
    blasDir = self.packageDir

    g = open(os.path.join(blasDir,'tmpmakefile'),'w')
    g.write(cmdline)
    g.close()
    if not self.installNeeded('tmpmakefile'): return self.installDir

    try:
      self.logPrintBox('Compiling OpenBLAS; this may take several minutes')
      output1,err1,ret  = config.package.Package.executeShellCommand('cd '+blasDir+' && make '+cmdline, timeout=2500, log = self.log)
    except RuntimeError as e:
      raise RuntimeError('Error running make on '+blasDir+': '+str(e))
    try:
      self.logPrintBox('Installing OpenBLAS')
      self.installDirProvider.printSudoPasswordMessage()
      output2,err2,ret  = config.package.Package.executeShellCommand('cd '+blasDir+' && '+self.installSudo+'mkdir -p '+libdir+' && '+self.installSudo+'cp -f libopenblas.* '+ libdir, timeout=30, log = self.log)
    except RuntimeError as e:
      raise RuntimeError('Error moving '+blasDir+' libraries: '+str(e))
    self.postInstall(output1+err1+output2+err2,'tmpmakefile')
    return self.installDir


