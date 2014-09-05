import config.package
import os

class Configure(config.package.GNUPackage):
  def __init__(self, framework):
    config.package.GNUPackage.__init__(self, framework)
    self.download         = ['http://www.mpich.org/static/downloads/3.1/mpich-3.1.tar.gz']
    self.download_solaris = ['http://ftp.mcs.anl.gov/pub/petsc/tmp/mpich-master-v3.0.4-106-g3adb59c.tar.gz']
    self.downloadfilename = 'mpich'
    return

  def setupHelp(self, help):
    config.package.GNUPackage.setupHelp(self,help)
    import nargs
    help.addArgument('MPI', '-download-mpich-pm=<hydra, gforker or mpd>',              nargs.Arg(None, 'hydra', 'Launcher for MPI processes'))
    help.addArgument('MPI', '-download-mpich-device=<ch3:nemesis or see mpich2 docs>', nargs.Arg(None, 'ch3:sock', 'Communicator for MPI processes'))
    help.addArgument('MPI', '-download-mpich-mpe=<bool>',                              nargs.ArgBool(None, 0, 'Install MPE with MPICH'))
    help.addArgument('MPI', '-download-mpich-shared=<bool>',                           nargs.ArgBool(None, 1, 'Install MPICH with shared libraries'))
    return

  def checkDownload(self, requireDownload = 1):
    if config.setCompilers.Configure.isCygwin() and not config.setCompilers.Configure.isGNU(self.setCompilers.CC):
      raise RuntimeError('Sorry, cannot download-install MPICH on Windows with Microsoft or Intel Compilers. Suggest installing Windows version of MPICH manually')
    if config.setCompilers.Configure.isSolaris() or self.framework.argDB['with-gcov']:
      self.download         = self.download_solaris
    return config.package.Package.checkDownload(self, requireDownload)

  def formGNUConfigureExtraArgs(self):
    args = []
    if self.framework.argDB['download-mpich-shared']:
      args.append('--enable-shared') # --enable-sharedlibs can now be removed?
      if self.compilers.isGCC or config.setCompilers.Configure.isIntel(compiler):
        if config.setCompilers.Configure.isDarwin():
          args.append('--enable-sharedlibs=gcc-osx')
        else:
          args.append('--enable-sharedlibs=gcc')
      elif config.setCompilers.Configure.isSun(compiler):
        args.append('--enable-sharedlibs=solaris-cc')
      else:
        args.append('--enable-sharedlibs=libtool')
    if 'download-mpich-device' in self.argDB:
      args.append('--with-device='+self.argDB['download-mpich-device'])
    if self.argDB['download-mpich-mpe']:
      args.append('--with-mpe')
    else:
      args.append('--without-mpe')
    args.append('--with-pm='+self.argDB['download-mpich-pm'])
    # make MPICH behave properly for valgrind
    args.append('--enable-g=meminit')
    args.append('--enable-fast')
    return args

  def GNUConfigureRmArgs(self,args):
    '''MPICH configure errors out if given certain F90 arguments'''
    rejects = ['--disable-f90','--enable-f90']
    rejects.extend([arg for arg in args if arg.startswith('F90=') or arg.startswith('F90FLAGS=')])
    self.logPrint('MPICH is rejecting configure arguments '+str(rejects))
    return [arg for arg in args if not arg in rejects]

  def MPICHInstall(self):
    '''MPICH requires a custom install since make clean requires sudo! Remove this when you update the MPICH tarball'''
    args = self.formGNUConfigureArgs()
    args = ' '.join(args)
    conffile = os.path.join(self.packageDir,self.package)
    fd = file(conffile, 'w')
    fd.write(args)
    fd.close()
    ### Use conffile to check whether a reconfigure/rebuild is required
    if not self.installNeeded(conffile):
      return self.installDir
    ### Configure and Build package
    self.gitPreInstallCheck()
    try:
      self.logPrintBox('Running configure on ' +self.PACKAGE+'; this may take several minutes')
      output1,err1,ret1  = config.base.Configure.executeShellCommand('cd '+self.packageDir+' && ./configure '+args, timeout=2000, log = self.framework.log)
    except RuntimeError, e:
      raise RuntimeError('Error running configure on ' + self.PACKAGE+': '+str(e))
    try:
      self.logPrintBox('Running make on '+self.PACKAGE+'; this may take several minutes')
      output2,err2,ret2  = config.base.Configure.executeShellCommand('cd '+self.packageDir+' && '+self.make.make_jnp, timeout=6000, log = self.framework.log)
      self.logPrintBox('Running make install on '+self.PACKAGE+'; this may take several minutes')
      self.installDirProvider.printSudoPasswordMessage()
      output2,err2,ret2  = config.base.Configure.executeShellCommand('cd '+self.packageDir+' && '+self.installSudo+self.make.make+' install', timeout=300, log = self.framework.log)
      output3,err3,ret3  = config.base.Configure.executeShellCommand('cd '+self.packageDir+' && '+self.installSudo+self.make.make+' clean', timeout=200, log = self.framework.log)
    except RuntimeError, e:
      raise RuntimeError('Error running make; make install on '+self.PACKAGE+': '+str(e))
    self.postInstall(output1+err1+output2+err2+output3+err3, self.package)
    return self.installDir

  def Install(self):
    '''After downloading and installing MPICH we need to reset the compilers to use those defined by the MPICH install'''
    installDir = self.MPICHInstall()
    self.updateCompilers(installDir,'mpicc','mpicxx','mpif77','mpif90')
    return installDir

