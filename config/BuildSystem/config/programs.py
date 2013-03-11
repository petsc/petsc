#!/usr/bin/env python
import user
import config.base

import os

class Configure(config.base.Configure):
  def __init__(self, framework):
    config.base.Configure.__init__(self, framework)
    self.headerPrefix = ''
    self.substPrefix  = ''
    return

  def __str__(self):
    return ''

  def setupHelp(self, help):
    import nargs
    import nargs
    help.addArgument('PETSc', '-with-make=<prog>', nargs.Arg(None, 'make', 'Specify make'))
    help.addArgument('PETSc', '-with-make-np=<np>', nargs.ArgInt(None, None, min=1, help='Default number of threads to use for parallel builds'))
    return

  def configureMake(self):
    '''Check various things about make'''
    self.getExecutable(self.framework.argDB['with-make'], getFullPath = 1,resultName = 'make')

    if not hasattr(self,'make'):
      import os
      if os.path.exists('/usr/bin/cygcheck.exe') and not os.path.exists('/usr/bin/make'):
        raise RuntimeError('''\
*** Incomplete cygwin install detected . /usr/bin/make is missing. **************
*** Please rerun cygwin-setup and select module "make" for install.**************''')
      else:
        raise RuntimeError('Could not locate the make utility on your system, make sure\n it is in your path or use --with-make=/fullpathnameofmake\n and run ./configure again')    
    # Check for GNU make
    haveGNUMake = 0
    self.getExecutable('strings', getFullPath = 1)
    if hasattr(self, 'strings'):
      try:
        (output, error, status) = config.base.Configure.executeShellCommand(self.strings+' '+self.make, log = self.framework.log)
        if not status and output.find('GNU Make') >= 0:
          haveGNUMake = 1
      except RuntimeError, e:
        self.framework.log.write('Make check failed: '+str(e)+'\n')
      if not haveGNUMake:
        try:
          (output, error, status) = config.base.Configure.executeShellCommand(self.strings+' '+self.make+'.exe', log = self.framework.log)
          if not status and output.find('GNU Make') >= 0:
            haveGNUMake = 1
        except RuntimeError, e:
          self.framework.log.write('Make check failed: '+str(e)+'\n')
    # mac has fat binaries where 'string' check fails
    if not haveGNUMake:
      try:
        (output, error, status) = config.base.Configure.executeShellCommand(self.make+' -v dummy-foobar', log = self.framework.log)
        if not status and output.find('GNU Make') >= 0:
          haveGNUMake = 1
      except RuntimeError, e:
        self.framework.log.write('Make check failed: '+str(e)+'\n')
        
    # Setup make flags
    self.flags = ''
    if haveGNUMake:
      self.flags += ' --no-print-directory'
    self.addMakeMacro('OMAKE ', self.make+' '+self.flags)
      
    # Check to see if make allows rules which look inside archives
    if haveGNUMake:
      self.addMakeRule('libc','${LIBNAME}(${OBJSC} ${SOBJSC})')
    else:
      self.addMakeRule('libc','${OBJSC}','-${AR} ${AR_FLAGS} ${LIBNAME} ${OBJSC}')
    self.addMakeRule('libf','${OBJSF}','-${AR} ${AR_FLAGS} ${LIBNAME} ${OBJSF}')

    # check no of cores on the build machine [perhaps to do make '-j ncores']
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
    return

  def configureMkdir(self):
    '''Make sure we can have mkdir automatically make intermediate directories'''
    self.getExecutable('mkdir', getFullPath = 1, setMakeMacro = 0)
    if hasattr(self, 'mkdir'):
      confDir    = '.conftest'
      conftmpDir = os.path.join('.conftest', 'tmp')
      if os.path.exists(conftmpDir): os.rmdir(conftmpDir)
      if os.path.exists(confDir):    os.rmdir(confDir)
      try:
        (output, error, status) = config.base.Configure.executeShellCommand(self.mkdir+' -p '+conftmpDir, log = self.framework.log)
        if not status and os.path.isdir(conftmpDir):
          self.mkdir = self.mkdir+' -p'
          self.logPrint('Adding -p flag to '+self.mkdir+' to automatically create directories')
        else:
          self.logPrint('Could not determine flag for '+self.mkdir+' to automatically create directories')
      except RuntimeError:
        self.logPrint('Could not determine flag for '+self.mkdir+' to automatically create directories')
      self.addMakeMacro('MKDIR', self.mkdir)
      if os.path.exists(conftmpDir): os.rmdir(conftmpDir)
      if os.path.exists(confDir):    os.rmdir(confDir)
    return

  def configurePrograms(self):
    '''Check for the programs needed to build and run PETSc'''
    self.getExecutable('sh',   getFullPath = 1, resultName = 'SHELL')
    if not hasattr(self, 'SHELL'): raise RuntimeError('Could not locate sh executable')
    self.getExecutable('sed',  getFullPath = 1)
    if not hasattr(self, 'sed'): raise RuntimeError('Could not locate sed executable')
    # check if sed supports -i "" or -i option
    sed1 = os.path.join(self.tmpDir,'sed1')
    f = open(sed1, 'w')
    f.write('sed\n')
    f.close()
    for sedcmd in [self.sed+' -i',self.sed+' -i ""','perl -pi -e']:
      try:
        (out,err,status) = Configure.executeShellCommand('%s s/sed/sd/g "%s"'%(sedcmd,sed1))
        self.framework.logPrint('Adding SEDINPLACE cmd: '+sedcmd)
        self.addMakeMacro('SEDINPLACE',sedcmd)
        status = 1
        break
      except RuntimeError:
        self.framework.logPrint('Rejected SEDINPLACE cmd: '+sedcmd)
    os.unlink(sed1)
    if not status:
        self.framework.logPrint('No suitable SEDINPLACE found')
        self.addMakeMacro('SEDINPLACE','SEDINPLACE_NOT_FOUND')
    self.getExecutable('mv',   getFullPath = 1)
    if not hasattr(self, 'mv'): raise RuntimeError('Could not locate mv executable')
    self.getExecutable('cp',   getFullPath = 1)
    if not hasattr(self, 'cp'): raise RuntimeError('Could not locate cp executable')
    self.getExecutable('grep', getFullPath = 1)    
    if not hasattr(self, 'grep'): raise RuntimeError('Could not locate grep executable')
    self.getExecutable('rm -f',getFullPath = 1, resultName = 'RM')
    if not hasattr(self, 'RM'): raise RuntimeError('Could not locate rm executable')
    self.getExecutable('diff', getFullPath = 1,setMakeMacro=0)
    if hasattr(self, 'diff'):
      # check if diff supports -w option for ignoring whitespace
      def mkfile(base,contents):
        fname = os.path.join(self.tmpDir,base)
        f = open(fname,'w')
        f.write(contents)
        f.close
        return fname
      diff1 = mkfile('diff1','diff\n')
      diff2 = mkfile('diff2','diff  \n')
      try:
        (out,err,status) = Configure.executeShellCommand('"%s" -w "%s" "%s"' % (self.diff,diff1,diff2))
      except RuntimeError:
        status = 1
      os.unlink(diff1)
      os.unlink(diff2)
      if status:
        (buf,err,status) = Configure.executeShellCommand('/bin/rpm -q diffutils')
        if buf.find('diffutils-2.8.1-17.fc8') > -1:
          raise RuntimeError('''\
*** Fedora 8 Linux with broken diffutils-2.8.1-17.fc8 detected. ****************
*** Run "sudo yum update diffutils" to get the latest bugfixed version. ********''')
        raise RuntimeError(self.diff+' executable does not properly handle -w (whitespace) option')        
      self.diff = self.diff + ' -w'
      self.addMakeMacro('DIFF',self.diff)
    else:
      if os.path.exists('/usr/bin/cygcheck.exe') and not os.path.exists('/usr/bin/diff'):
        raise RuntimeError('''\
*** Incomplete cygwin install detected . /usr/bin/diff is missing. **************
*** Please rerun cygwin-setup and select module "diff" for install.**************''')
      else:
        raise RuntimeError('Could not locate diff executable')
    self.getExecutable('ps', path = '/usr/ucb:/usr/usb', resultName = 'UCBPS')
    if hasattr(self, 'UCBPS'):
      self.addDefine('HAVE_UCBPS', 1)
    self.getExecutable('gzip', getFullPath=1, resultName = 'GZIP')
    if hasattr(self, 'GZIP'):
      self.addDefine('HAVE_GZIP', 1)
    import sys
    self.addMakeMacro('PYTHON',sys.executable)
    return

  def configure(self):
    if not self.framework.argDB['with-make'] == '0':
      self.executeTest(self.configureMake)
      self.executeTest(self.configureMkdir)
      self.executeTest(self.configurePrograms)    
    return
