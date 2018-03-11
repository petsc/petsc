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

  def createArchitecture(self):
    import sys
    arch = 'arch-' + sys.platform.replace('cygwin','mswin')
    # use opt/debug, c/c++ tags.s
    arch+= '-'+self.framework.argDB['with-clanguage'].lower().replace('+','x')
    if self.framework.argDB['with-debugging']:
      arch += '-debug'
    else:
      arch += '-opt'
    return arch

  def configureArchitecture(self):
    '''Checks PETSC_ARCH and sets if not set'''
    # Warn if PETSC_ARCH doesnt match env variable
    if 'PETSC_ARCH' in self.framework.argDB and 'PETSC_ARCH' in os.environ and self.framework.argDB['PETSC_ARCH'] != os.environ['PETSC_ARCH']:
      self.logPrintBox('''\
Warning: PETSC_ARCH from environment does not match command-line or name of script.
Warning: Using from command-line or name of script: %s, ignoring environment: %s''' % (str(self.framework.argDB['PETSC_ARCH']), str(os.environ['PETSC_ARCH'])))
      os.environ['PETSC_ARCH'] = self.framework.argDB['PETSC_ARCH']
    if 'with-petsc-arch' in self.framework.argDB:
      self.arch = self.framework.argDB['with-petsc-arch']
      msg = 'option -with-petsc-arch='+str(self.arch)
    elif 'PETSC_ARCH' in self.framework.argDB:
      self.arch = self.framework.argDB['PETSC_ARCH']
      msg = 'option PETSC_ARCH='+str(self.arch)
    elif 'PETSC_ARCH' in os.environ:
      self.arch = os.environ['PETSC_ARCH']
      msg = 'environment variable PETSC_ARCH='+str(self.arch)
    else:
      self.arch = self.createArchitecture()
    if self.arch.find('/') >= 0 or self.arch.find('\\') >= 0:
      raise RuntimeError('PETSC_ARCH should not contain path characters, but you have specified with '+msg)
    if self.arch.startswith('-'):
      raise RuntimeError('PETSC_ARCH should not start with "-", but you have specified with '+msg)
    if self.arch.startswith('.'):
      raise RuntimeError('PETSC_ARCH should not start with ".", but you have specified with '+msg)
    if not len(self.arch):
      raise RuntimeError('PETSC_ARCH cannot be empty string. Use a valid string or do not set one. Currently set with '+msg)
    self.archBase = re.sub(r'^(\w+)[-_]?.*$', r'\1', self.arch)
    return

  def configure(self):
    self.executeTest(self.configureArchitecture)
    # required by top-level configure.py
    self.framework.arch = self.arch
    return
