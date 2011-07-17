import config.base
import os
import re

class Configure(config.base.Configure):
  def __init__(self, framework):
    config.base.Configure.__init__(self, framework)
    self.headerPrefix = 'PETSC'
    self.substPrefix  = 'PETSC'
    return

  def __str1__(self):
    if not hasattr(self, 'arch'):
      return ''
    desc = ['PETSc:']
    desc.append('  PETSC_ARCH: '+str(self.arch))
    return '\n'.join(desc)+'\n'
  
  def setupHelp(self, help):
    import nargs
    help.addArgument('PETSc', '-PETSC_ARCH=<string>',     nargs.Arg(None, None, 'The configuration name'))
    help.addArgument('PETSc', '-with-petsc-arch=<string>',nargs.Arg(None, None, 'The configuration name'))
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
Warning: PETSC_ARCH from environment does not match command-line or name of script.
Warning: Using from command-line or name of script: %s, ignoring environment: %s''' % (str(self.framework.argDB['PETSC_ARCH']), str(os.environ['PETSC_ARCH'])))
    if 'with-petsc-arch' in self.framework.argDB:
      self.arch = self.framework.argDB['with-petsc-arch']
    elif 'PETSC_ARCH' in self.framework.argDB:
      self.arch = self.framework.argDB['PETSC_ARCH']
    else:
      if 'PETSC_ARCH' in os.environ:
        if not len(os.environ['PETSC_ARCH']):
          raise RuntimeError('PETSC_ARCH is the empty string in your environment. It must either be a valid string, or not be defined in the environment at all.')
        self.arch = os.environ['PETSC_ARCH']
      else:
        import sys
        self.arch = 'arch-' + sys.platform.replace('cygwin','mswin')
        # use opt/debug, c/c++ tags.
        self.arch+= '-'+self.languages.clanguage.lower()
        if self.compilerFlags.debugging:
          self.arch += '-debug'
        else:
          self.arch += '-opt'
    if self.arch.find('/') >= 0 or self.arch.find('\\') >= 0:
      raise RuntimeError('PETSC_ARCH should not contain path characters, but you have specified: '+str(self.arch))
    self.archBase = re.sub(r'^(\w+)[-_]?.*$', r'\1', self.arch)
    self.addDefine('ARCH', '"'+self.arch+'"')
    return

  def configure(self):
    self.executeTest(self.configureArchitecture)
    # required by top-level configure.py
    self.framework.arch = self.arch
    return
