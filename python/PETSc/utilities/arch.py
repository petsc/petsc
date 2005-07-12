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
          self.arch += '-debug'
        else:
          self.arch += '-opt'

    self.archBase = re.sub(r'^(\w+)[-_]?.*$', r'\1', self.arch)
    self.hostOsBase = re.sub(r'^(\w+)[-_]?.*$', r'\1', self.framework.host_os)
    self.addDefine('ARCH', self.hostOsBase)
    self.addDefine('ARCH_NAME', '"'+self.arch+'"')
    self.addSubstitution('ARCH', self.arch)
    return

  def configure(self):
    self.executeTest(self.configureArchitecture)
    return
