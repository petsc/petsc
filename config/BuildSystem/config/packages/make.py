from __future__ import generators
import config.package

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.download          = ['http://ftp.gnu.org/gnu/make/make-3.82.tar.gz','http://ftp.mcs.anl.gov/pub/petsc/externalpackages/make-3.82.tar.gz']
    self.complex           = 1
    self.double            = 0
    self.requires32bitint  = 0
    self.worksonWindows    = 1
    self.downloadonWindows = 1
    self.useddirectly      = 0

    self.flags             = ''
    self.haveGNUMake       = 0
    return

  def setupHelp(self, help):
    import nargs
    help.addArgument('Make', '-with-make=<prog>',                            nargs.Arg(None, 'gmake', 'Specify GNU make'))
    help.addArgument('Make', '-with-make-np=<np>',                           nargs.ArgInt(None, None, min=1, help='Default number of threads to use for parallel builds'))
    help.addArgument('Make', '-download-make=<no,yes,filename>',             nargs.ArgDownload(None, 0, 'Download and install GNU make'))
    help.addArgument('Make', '-download-make-cc=<prog>',                     nargs.Arg(None, None, 'C compiler for GNU make configure'))
    help.addArgument('Make', '-download-make-configure-options=<options>',   nargs.Arg(None, None, 'additional options for GNU make configure'))
    return

  def Install(self):
    import os
    args = ['--prefix='+self.installDir]
    args.append('--program-prefix=g')
    if self.framework.argDB.has_key('download-make-cc'):
      args.append('CC="'+self.framework.argDB['download-make-cc']+'"')
    if self.framework.argDB.has_key('download-make-configure-options'):
      args.append(self.framework.argDB['download-make-configure-options'])
    args = ' '.join(args)
    fd = file(os.path.join(self.packageDir,'make.args'), 'w')
    fd.write(args)
    fd.close()
    if self.installNeeded('make.args'):
      try:
        self.logPrintBox('Configuring GNU Make; this may take several minutes')
        output,err,ret  = config.package.Package.executeShellCommand('cd '+self.packageDir+' && ./configure '+args, timeout=900, log = self.framework.log)
      except RuntimeError, e:
        raise RuntimeError('Error running configure on GNU make (install manually): '+str(e))
      try:
        self.logPrintBox('Compiling GNU Make; this may take several minutes')
        if self.getExecutable('make', getFullPath = 1,resultName='make',setMakeMacro = 0):
          output,err,ret  = config.package.Package.executeShellCommand('cd '+self.packageDir+' && '+self.make+' &&  '+self.make+' install && '+self.make+' clean', timeout=2500, log = self.framework.log)
        else:
          output,err,ret  = config.package.Package.executeShellCommand('cd '+self.packageDir+' && ./build.sh && ./make install && ./make clean', timeout=2500, log = self.framework.log)
      except RuntimeError, e:
        raise RuntimeError('Error running make; make install on GNU Make (install manually): '+str(e))
      self.postInstall(output+err,'make.args')
    self.binDir = os.path.join(self.installDir, 'bin')
    self.make = os.path.join(self.binDir, 'gmake')
    self.addMakeMacro('MAKE',self.make)
    return self.installDir

  def configureMake(self):
    '''Check for user specified make - or gmake, make'''
    if self.framework.clArgDB.has_key('with-make'):
      if not self.getExecutable(self.framework.argDB['with-make'],getFullPath = 1,resultName = 'make'):
        raise RuntimeError('Error! User provided make not found :'+self.framework.argDB['with-make'])
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
    self.getExecutable('strings', getFullPath = 1,setMakeMacro = 0)
    if hasattr(self, 'strings'):
      try:
        (output, error, status) = config.base.Configure.executeShellCommand(self.strings+' '+self.make, log = self.framework.log)
        if not status and output.find('GNU Make') >= 0:
          self.haveGNUMake = 1
      except RuntimeError, e:
        self.framework.log.write('Make check failed: '+str(e)+'\n')
      if not self.haveGNUMake:
        try:
          (output, error, status) = config.base.Configure.executeShellCommand(self.strings+' '+self.make+'.exe', log = self.framework.log)
          if not status and output.find('GNU Make') >= 0:
            self.haveGNUMake = 1
        except RuntimeError, e:
          self.framework.log.write('Make check failed: '+str(e)+'\n')
    # mac has fat binaries where 'string' check fails
    if not self.haveGNUMake:
      try:
        (output, error, status) = config.base.Configure.executeShellCommand(self.make+' -v dummy-foobar', log = self.framework.log)
        if not status and output.find('GNU Make') >= 0:
          self.haveGNUMake = 1
      except RuntimeError, e:
        self.framework.log.write('Make check failed: '+str(e)+'\n')

    # Setup make flags
    if self.haveGNUMake:
      self.flags += ' --no-print-directory'
      self.addMakeMacro('PETSC_BUILD_USING_GNUMAKE',1)

    # Check to see if make allows rules which look inside archives
    if self.haveGNUMake:
      self.addMakeRule('libc','${LIBNAME}(${OBJSC})')
      self.addMakeRule('libcu','${LIBNAME}(${OBJSCU})')
    else:
      self.addMakeRule('libc','${OBJSC}','-${AR} ${AR_FLAGS} ${LIBNAME} ${OBJSC}')
      self.addMakeRule('libcu','${OBJSCU}','-${AR} ${AR_FLAGS} ${LIBNAME} ${OBJSCU}')
    self.addMakeRule('libf','${OBJSF}','-${AR} ${AR_FLAGS} ${LIBNAME} ${OBJSF}')
    return

  def configureMakeNP(self):
    '''check no of cores on the build machine [perhaps to do make '-j ncores']'''
    make_np = self.framework.argDB.get('with-make-np')
    if make_np is not None:
      self.framework.logPrint('using user-provided make_np = %d' % make_np)
    else:
      try:
        import multiprocessing
        cores = multiprocessing.cpu_count()
        make_np = max(min(cores+1,5),cores/3)
        self.framework.logPrint('module multiprocessing found %d cores: using make_np = %d' % (cores,make_np))
      except (ImportError), e:
        make_np = 2
        self.framework.logPrint('module multiprocessing *not* found: using default make_np = %d' % make_np)
      try:
        import os
        import pwd
        if 'barrysmith' == pwd.getpwuid(os.getuid()).pw_name:
          # Barry wants to use exactly the number of physical cores (not logical cores) because it breaks otherwise.
          # Since this works for everyone else who uses a Mac, something must be wrong with their systems. ;-)
          try:
            (output, error, status) = config.base.Configure.executeShellCommand('/usr/sbin/system_profiler -detailLevel full SPHardwareDataType', log = self.framework.log)
            import re
            match = re.search(r'.*Total Number Of Cores: (\d+)', output)
            if match:
              make_np = int(match.groups()[0])
              self.framework.logPrint('Found number of cores using system_profiler: make_np = %d' % (make_np,))
          except:
            pass
      except:
        pass
    self.make_np = make_np
    self.addMakeMacro('MAKE_NP',str(make_np))
    self.make_jnp = self.make + ' -j ' + str(self.make_np)
    return

  def configure(self):
    '''Determine whether (GNU) make exist or not'''

    if self.framework.argDB['with-make'] == '0':
      return
    if (self.framework.argDB['download-make']):
      config.package.Package.configure(self)
    else:
      self.executeTest(self.configureMake)
    self.executeTest(self.configureCheckGNUMake)
    self.executeTest(self.configureMakeNP)
    self.addMakeMacro('OMAKE ', self.make+' '+self.flags)
    return
