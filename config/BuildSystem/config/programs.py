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
    help.addArgument('PETSc', '-with-autoreconf=<prog>', nargs.Arg(None, 'autoreconf', 'Specify autoreconf'))
    help.addArgument('PETSc', '-with-libtoolize=<prog>', nargs.Arg(None, 'libtoolize', 'Specify libtoolize'))
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
        (output, error, status) = config.base.Configure.executeShellCommand(self.mkdir+' -p '+conftmpDir, log = self.log)
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

  def configureAutoreconf(self):
    '''Check for autoreconf'''
    self.autoreconf = None
    if self.getExecutable(self.argDB['with-autoreconf'], getFullPath = 1,resultName = 'autoreconf',setMakeMacro = 0):
      import shutil,os
      testdir = os.path.join(self.tmpDir, 'autoconfdir')
      acfile  = os.path.join(testdir,'configure.ac')
      if not os.path.isdir(testdir):
        os.mkdir(testdir)
      accode='AC_INIT(petscconftest, version-0.1)'
      fd = open(acfile,'w')
      fd.write(accode)
      fd.close()
      try:
        output,err,ret  = config.base.Configure.executeShellCommand([self.autoreconf], log = self.log, cwd=testdir)
        self.logPrint('autoreconf test successful!')
      except RuntimeError as e:
        self.autoreconf = None
        self.logPrint('autoreconf test error: '+str(e))
      shutil.rmtree(testdir)
    self.libtoolize = None
    if not self.getExecutable(self.argDB['with-libtoolize'], getFullPath = 1,resultName = 'libtoolize',setMakeMacro = 0):
      # it is called blibtoolize on Apple to prevent conflict with Apple's libtool
      self.getExecutable('glibtoolize', getFullPath = 1,resultName = 'libtoolize',setMakeMacro = 0\
)
    return

  def configurePrograms(self):
    '''Check for the programs needed to build and run PETSc'''
    for shell in ['bash','zsh','sh']:
      if self.getExecutable(shell,   getFullPath = 1, resultName = 'SHELL'): break
    if not hasattr(self, 'SHELL'): raise RuntimeError('Could not locate bash/zsh/sh executable')
    self.getExecutable('sed',  getFullPath = 1)
    if not hasattr(self, 'sed'): raise RuntimeError('Could not locate sed executable')
    # check if sed supports -i "" or -i option
    sed1 = os.path.join(self.tmpDir,'sed1')
    f = open(sed1, 'w')
    f.write('sed\n')
    f.close()
    for sedcmd in [self.sed+' -i',self.sed+' -i ""','perl -pi -e']:
      try:
        (out,err,status) = Configure.executeShellCommand('%s s/sed/sd/g "%s"'%(sedcmd,sed1), log = self.log)
        self.logPrint('Adding SEDINPLACE cmd: '+sedcmd)
        self.addMakeMacro('SEDINPLACE',sedcmd)
        status = 1
        break
      except RuntimeError:
        self.logPrint('Rejected SEDINPLACE cmd: '+sedcmd)
    os.unlink(sed1)
    if not status:
        self.logPrint('No suitable SEDINPLACE found')
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
        (out,err,status) = Configure.executeShellCommand('"%s" -w "%s" "%s"' % (self.diff,diff1,diff2), log = self.log)
      except RuntimeError:
        status = 1
      os.unlink(diff1)
      os.unlink(diff2)
      if status:
        (buf,err,status) = Configure.executeShellCommand('/bin/rpm -q diffutils', log = self.log)
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
    import sys
    self.addMakeMacro('PYTHON',sys.executable)
    self.getExecutable('m4', getFullPath=1, resultName = 'M4')
    return

  def configure(self):
    self.executeTest(self.configureMkdir)
    self.executeTest(self.configureAutoreconf)
    self.executeTest(self.configurePrograms)
    return
