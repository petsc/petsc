import config.package
import os

class Configure(config.package.GNUPackage):
  def __init__(self, framework):
    config.package.GNUPackage.__init__(self, framework)
    self.download          = ['http://ftp.gnu.org/gnu/make/make-4.1.tar.gz','http://ftp.mcs.anl.gov/pub/petsc/externalpackages/make-4.1.tar.gz']
    self.complex           = 1
    self.double            = 0
    self.downloadonWindows = 1
    self.useddirectly      = 0

    self.printdirflag      = ''
    self.noprintdirflag    = ''
    self.paroutflg         = ''
    self.haveGNUMake       = 0
    self.publicInstall     = 0  # always install in PETSC_DIR/PETSC_ARCH (not --prefix) since this is not used by users
    self.parallelMake      = 0  # sowing does not support make -j np
    return

  def setupHelp(self, help):
    import nargs
    config.package.GNUPackage.setupHelp(self, help)
    help.addArgument('MAKE', '-with-make-np=<np>',                           nargs.ArgInt(None, None, min=1, help='Default number of threads to use for parallel builds'))
    help.addArgument('MAKE', '-download-make-cc=<prog>',                     nargs.Arg(None, None, 'C compiler for GNU make configure'))
    help.addArgument('MAKE', '-download-make-configure-options=<options>',   nargs.Arg(None, None, 'additional options for GNU make configure'))
    return

  def formGNUConfigureArgs(self):
    '''Does not use the standard arguments at all since this does not use the MPI compilers etc
       Sowing will chose its own compilers if they are not provided explicitly here'''
    args = ['--prefix='+self.confDir]
    if 'download-make-cc' in self.argDB and self.argDB['download-make-cc']:
      args.append('CC="'+self.argDB['download-make-cc']+'"')
    if 'download-make-configure-options' in self.argDB and self.argDB['download-make-configure-options']:
      args.append(self.argDB['download-make-configure-options'])
    return args

  def Install(self):
    ''' Cannot use GNUPackage Install because that one uses make which does not yet exist
        This is almost a copy of GNUPackage Install just avoiding the use of make'''
    args = self.formGNUConfigureArgs()
    args = ' '.join(args)
    conffile = os.path.join(self.packageDir,self.package+'.petscconf')
    fd = file(conffile, 'w')
    fd.write(args)
    fd.close()
    ### Use conffile to check whether a reconfigure/rebuild is required
    if not self.installNeeded(conffile):
      return self.installDir
    ### Configure and Build package
    try:
      self.logPrintBox('Running configure on ' +self.PACKAGE+'; this may take several minutes')
      output1,err1,ret1  = config.base.Configure.executeShellCommand('cd '+self.packageDir+' && ./configure '+args, timeout=2000, log = self.log)
    except RuntimeError, e:
      raise RuntimeError('Error running configure on ' + self.PACKAGE+': '+str(e))
    try:
      self.logPrintBox('Running make on '+self.PACKAGE+'; this may take several minutes')
      output1,err1,ret  = config.package.Package.executeShellCommand('cd '+self.packageDir+' && ./build.sh && ./make install && ./make clean', timeout=2500, log = self.log)
    except RuntimeError, e:
      raise RuntimeError('Error building or installing make '+self.PACKAGE+': '+str(e))
    self.postInstall(output1+err1, conffile)
    return self.installDir

  def configureMake(self):
    '''Check for user specified make - or gmake, make'''
    if self.framework.clArgDB.has_key('with-make'):
      if not self.getExecutable(self.argDB['with-make'],getFullPath = 1,resultName = 'make'):
        raise RuntimeError('Error! User provided make not found :'+self.argDB['with-make'])
      self.found = 1
      return
    if not self.getExecutable('gmake', getFullPath = 1,resultName = 'make') and not self.getExecutable('make', getFullPath = 1,resultName = 'make'):
      import os
      if os.path.exists('/usr/bin/cygcheck.exe') and not os.path.exists('/usr/bin/make'):
        raise RuntimeError('''\
*** Incomplete cygwin install detected . /usr/bin/make is missing. **************
*** Please rerun cygwin-setup and select module "make" for install.**************''')
      else:
        raise RuntimeError('Could not locate the make utility on your system, try --download-make')
    self.found = 1
    return

  def configureCheckGNUMake(self):
    '''Check for GNU make'''
    try:
      import re
      # set self.haveGNUMake only if using gnumake version > 3.80 [as older version break with gmakefile]
      (output, error, status) = config.base.Configure.executeShellCommand(self.make+' --version', log = self.log)
      gver = re.compile('GNU Make ([0-9]+).([0-9]+)').match(output)
      if not status and gver:
        major = int(gver.group(1))
        minor = int(gver.group(2))
        if ((major > 3) or (major == 3 and minor > 80)):
          self.haveGNUMake = 1
        if (major > 3) and not self.setCompilers.isDarwin(self.log) and not self.setCompilers.isFreeBSD(self.log):
          self.paroutflg = "--output-sync=recurse"
    except RuntimeError, e:
      self.log.write('GNUMake check failed: '+str(e)+'\n')

    # Setup make flags
    if self.haveGNUMake:
      self.printdirflag = ' --print-directory'
      self.noprintdirflag = ' --no-print-directory'
      self.addMakeMacro('MAKE_IS_GNUMAKE',1)
    else:
      self.logPrintBox('Warning: '+self.make+' is not GNUMake (3.81 or higher). Suggest using --download-make')

    # Check to see if make allows rules which look inside archives
    if self.haveGNUMake:
      self.addMakeRule('libc','${LIBNAME}(${OBJSC})')
      self.addMakeRule('libcxx','${LIBNAME}(${OBJSCXX})')
      self.addMakeRule('libcu','${LIBNAME}(${OBJSCU})')
    else:
      self.addMakeRule('libc','${OBJSC}','-${AR} ${AR_FLAGS} ${LIBNAME} ${OBJSC}')
      self.addMakeRule('libcxx','${OBJSCXX}','-${AR} ${AR_FLAGS} ${LIBNAME} ${OBJSCXX}')
      self.addMakeRule('libcu','${OBJSCU}','-${AR} ${AR_FLAGS} ${LIBNAME} ${OBJSCU}')
    self.addMakeRule('libf','${OBJSF}','-${AR} ${AR_FLAGS} ${LIBNAME} ${OBJSF}')
    return

  def compute_make_np(self,i):
    f16 = .80
    f32 = .65
    f64 = .50
    f99 = .30
    if (i<=2):    return 2
    elif (i<=4):  return i
    elif (i<=16): return int(4+(i-4)*f16)
    elif (i<=32): return int(4+12*f16+(i-16)*f32)
    elif (i<=64): return int(4+12*f16+16*f32+(i-32)*f64)
    else:         return int(4+12*f16+16*f32+32*f64+(i-64)*f99)
    return

  def configureMakeNP(self):
    '''check no of cores on the build machine [perhaps to do make '-j ncores']'''
    try:
      import multiprocessing # python-2.6 feature
      cores = multiprocessing.cpu_count()
      make_np = self.compute_make_np(cores)
      self.logPrint('module multiprocessing found %d cores: using make_np = %d' % (cores,make_np))
    except (ImportError), e:
      cores = 2
      make_np = 2
      self.logPrint('module multiprocessing *not* found: using default make_np = %d' % make_np)

    if 'with-make-np' in self.argDB and self.argDB['with-make-np']:
        self.logPrint('using user-provided make_np = %d' % make_np)
        make_np = self.argDB['with-make-np']

    self.make_np = make_np
    self.addMakeMacro('MAKE_NP',str(make_np))
    self.addMakeMacro('NPMAX',str(cores))
    self.make_jnp = self.make + ' -j ' + str(self.make_np)
    return

  def configure(self):
    if (self.argDB['download-make']):
      config.package.GNUPackage.configure(self)
      self.getExecutable('make', path=os.path.join(self.installDir,'bin'), getFullPath = 1)
    else:
      self.executeTest(self.configureMake)
    self.executeTest(self.configureCheckGNUMake)
    self.executeTest(self.configureMakeNP)
    self.addMakeMacro('OMAKE_PRINTDIR ', self.make+' '+self.printdirflag)
    self.addMakeMacro('OMAKE', self.make+' '+self.noprintdirflag)
    self.addMakeMacro('MAKE_PAR_OUT_FLG', self.paroutflg)
    return
