import config.base
import os
import re

class Configure(config.base.Configure):
  def __init__(self, framework):
    config.base.Configure.__init__(self, framework)
    self.headerPrefix = 'PETSC'
    self.substPrefix  = 'PETSC'
    return

  def __str__(self):
    desc = ['PETSc:']
    carch = str(self.arch)
    envarch = os.getenv('PETSC_ARCH')
    if not carch == envarch :
      desc.append('  **\n  ** Configure has determined that your PETSC_ARCH must be specified as:')
      desc.append('  **  ** PETSC_ARCH: '+str(self.arch+'\n  **'))
    else:
      desc.append('  PETSC_ARCH: '+str(self.arch))
    return '\n'.join(desc)+'\n'
  
  def setupHelp(self, help):
    import nargs
    help.addArgument('PETSc', '-PETSC_ARCH',                       nargs.Arg(None, None, 'The configuration name'))
    return

  def setupDependencies(self, framework):
    config.base.Configure.setupDependencies(self, framework)
    self.petscdir = framework.require('PETSc.utilities.petscdir', self)
    self.languages = framework.require('PETSc.utilities.languages', self)
    self.compilerFlags = framework.require('config.compilerFlags', self)
    return

  def configureArchitecture(self):
    '''Checks PETSC_ARCH and sets if not set'''
    import sys

    auxDir = None
    searchDirs = [os.path.join(self.petscdir.dir, 'config'), os.path.join(self.petscdir.dir, 'bin', 'config')] + sys.path
    for dir in searchDirs:
      if os.path.isfile(os.path.join(dir, 'config.sub')):
        auxDir      = dir
        configSub   = os.path.join(auxDir, 'config.sub')
        configGuess = os.path.join(auxDir, 'config.guess')
        break
    if auxDir is None:
      raise RuntimeError('Unable to locate config.sub in '+str(searchDirs)+'.\nYour PETSc directory is incomplete.\n Get PETSc again')
    try:
      host   = config.base.Configure.executeShellCommand(self.shell+' '+configGuess, log = self.framework.log)[0]
      output = config.base.Configure.executeShellCommand(self.shell+' '+configSub+' '+host, log = self.framework.log)[0]
    except RuntimeError, e:
      fd = open(configGuess)
      data = fd.read()
      fd.close()
      if data.find('\r\n') >= 0:
        raise RuntimeError('''It appears petsc.tar.gz is uncompressed on Windows (perhaps with Winzip)
          and files copied over to Unix/Linux. Windows introduces LF characters which are
          inappropriate on other systems. Please use gunzip/tar on the install machine.\n''')
      raise RuntimeError('Unable to determine host type using '+configSub+': '+str(e))
    m = re.match(r'^(?P<cpu>[^-]*)-(?P<vendor>[^-]*)-(?P<os>.*)$', output)
    if not m:
      raise RuntimeError('Unable to parse output of '+configSub+': '+output)
    self.framework.host_cpu    = m.group('cpu')
    self.framework.host_vendor = m.group('vendor')
    self.framework.host_os     = m.group('os')


    # Warn if PETSC_ARCH doesnt match env variable
    if 'PETSC_ARCH' in self.framework.argDB and 'PETSC_ARCH' in os.environ and self.framework.argDB['PETSC_ARCH'] != os.environ['PETSC_ARCH']:
      self.logPrintBox('''\
Warning: PETSC_ARCH from environment does not match command-line.
Warning: Using from command-line: %s, ignoring environment: %s''' % (str(self.framework.argDB['PETSC_ARCH']), str(os.environ['PETSC_ARCH'])))
    if 'PETSC_ARCH' in self.framework.argDB:
      self.arch = self.framework.argDB['PETSC_ARCH']
    else:
      if 'PETSC_ARCH' in os.environ:
        self.arch = os.environ['PETSC_ARCH']
      else:
        self.arch = self.framework.host_os
        # use opt/debug, c/c++/complex tags.
        self.arch+= '-'+self.languages.clanguage.lower()+'-'+self.languages.scalartype
        if self.compilerFlags.debugging:
          self.arch += '-opt'
        else:
          self.arch += '-debug'

    self.archBase = re.sub(r'^(\w+)[-_]?.*$', r'\1', self.arch)
    self.hostOsBase = re.sub(r'^(\w+)[-_]?.*$', r'\1', self.framework.host_os)
    self.addDefine('ARCH', self.hostOsBase)
    self.addDefine('ARCH_NAME', '"'+self.arch+'"')
    self.addSubstitution('ARCH', self.arch)
    return

  def configure(self):
    self.executeTest(self.configureArchitecture)
    return
