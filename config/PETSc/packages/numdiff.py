import PETSc.package

class Configure(PETSc.package.NewPackage):
  def __init__(self, framework):
    PETSc.package.NewPackage.__init__(self, framework)
    self.download          = ['http://ftp.mcs.anl.gov/pub/petsc/externalpackages/numdiff-5.2.1.a.tar.gz']
    self.complex           = 1
    self.double            = 0
    self.requires32bitint  = 0
    self.worksonWindows    = 1
    self.downloadonWindows = 1
    return

  def setupHelp(self, help):
    import nargs
    PETSc.package.NewPackage.setupHelp(self, help)
    help.addArgument('NUMDIFF', '-download-numdiff-cc=<prog>',                     nargs.Arg(None, None, 'C compiler for numdiff configure'))
    help.addArgument('NUMDIFF', '-download-numdiff-cxx=<prog>',                    nargs.Arg(None, None, 'CXX compiler for numdiff configure'))
    help.addArgument('NUMDIFF', '-download-numdiff-cpp=<prog>',                    nargs.Arg(None, None, 'CPP for numdiff configure'))
    help.addArgument('NUMDIFF', '-download-numdiff-cxxcpp=<prog>',                 nargs.Arg(None, None, 'CXX CPP for numdiff configure'))
    help.addArgument('NUMDIFF', '-download-numdiff-configure-options=<options>',   nargs.Arg(None, None, 'additional options for numdiff configure'))
    return

  def Install(self):
    import os
    args = ['--prefix='+self.installDir]
    if 'download-numdiff-cc' in self.framework.argDB and self.framework.argDB['download-numdiff-cc']:
      args.append('CC="'+self.framework.argDB['download-numdiff-cc']+'"')
    if 'download-numdiff-cxx' in self.framework.argDB and self.framework.argDB['download-numdiff-cxx']:
      args.append('CXX="'+self.framework.argDB['download-numdiff-cxx']+'"')
    if 'download-numdiff-cpp' in self.framework.argDB and self.framework.argDB['download-numdiff-cpp']:
      args.append('CPP="'+self.framework.argDB['download-numdiff-cpp']+'"')
    if 'download-numdiff-cxxcpp' in self.framework.argDB and self.framework.argDB['download-numdiff-cxxcpp']:
      args.append('CXXCPP="'+self.framework.argDB['download-numdiff-cxxcpp']+'"')
    if 'download-numdiff-configure-options' in self.framework.argDB and self.framework.argDB['download-numdiff-configure-options']:
      args.append(self.framework.argDB['download-numdiff-configure-options'])
    args.append('--enable-nls=no')
    args.append('--enable-gmp=no')
    args = ' '.join(args)
    fd = file(os.path.join(self.packageDir,self.package), 'w')
    fd.write(args)
    fd.close()
    if self.installNeeded(self.package):
      try:
        output,err,ret  = PETSc.package.NewPackage.executeShellCommand('cd '+self.packageDir+' && ./configure '+args, timeout=900, log = self.framework.log)
      except RuntimeError, e:
        raise RuntimeError('Error running configure on Numdiff (install manually): '+str(e))
      try:
        # TARGET_ARCH set by Intel compilers on windows can break this build
        output,err,ret  = PETSc.package.NewPackage.executeShellCommand('cd '+self.packageDir+' && make TARGET_ARCH="" &&  make install && make clean', timeout=2500, log = self.framework.log)
      except RuntimeError, e:
        raise RuntimeError('Error running make; make install on Numdiff (install manually): '+str(e))
      self.framework.actions.addArgument('Numdiff', 'Install', 'Installed Numdiff into '+self.installDir)
    self.numdiff = os.path.join(self.installDir,'bin','numdiff')
    self.addMakeMacro('DIFF ', self.numdiff + '  -a 1.e-6 -r 1.e-4 ')
    return self.installDir

  def configure(self):
   if (self.framework.clArgDB.has_key('with-numdiff') and not self.framework.argDB['with-numdiff']) or \
          (self.framework.clArgDB.has_key('download-numdiff') and not self.framework.argDB['download-numdiff']):
      self.framework.logPrint("Not checking numdiff on user request\n")
      return

   # If download option is specified always build sowing
   if self.framework.argDB['download-numdiff']:
     PETSc.package.NewPackage.configure(self)

   # autodetect if numdiff is required
   if self.petscdir.isClone:
     self.framework.logPrint('PETSc clone, checking for numdiff or if it is needed\n')

     self.getExecutable('numdiff', getFullPath = 1)

     if hasattr(self, 'numdiff'):
       self.framework.logPrint('Found numdiff, not installing numdiff')
     else:
       self.framework.logPrint('numdiff not found. Installing numdiff')
       self.framework.argDB['download-numdiff'] = 1
       PETSc.package.NewPackage.configure(self)
