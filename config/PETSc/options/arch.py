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
    help.addArgument('PETSc', '-force=<bool>',            nargs.ArgBool(None, 0, 'Bypass configure hash caching, and run to completion'))
    return

  def setupDependencies(self, framework):
    self.sourceControl = framework.require('config.sourceControl',self)
    self.petscdir = framework.require('PETSc.options.petscdir', self)
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

  def makeDependency(self,hash,hashfile):
    '''Deletes the current hashfile and saves the hashfile name and its value in framework so that'''
    '''framework.Configure can create the file upon success of configure'''
    import os
    if hash:
      self.framework.hash = hash
      self.framework.hashfile = hashfile
    self.logPrint('Deleting configure hash file: '+hashfile)
    try:
      os.remove(hashfile)
    except:
      self.logPrint('Unable to delete configure hash file: '+hashfile)
      return
    self.logPrint('Deleted configure hash file: '+hashfile)

  def checkDependency(self):
    '''Checks if configure needs to be run'''
    '''Checks if files in config have changed, the command line options have changed or the PATH has changed'''
    import os
    import sys
    import hashlib
    hashfile = os.path.join(self.arch,'lib','petsc','conf','configure-hash')
    args = sorted(set(filter(lambda x: not (x.startswith('PETSC_ARCH') or x == '--force'),sys.argv[1:])))
    hash = 'args:\n' + '\n'.join('    '+a for a in args) + '\n'
    hash += 'PATH=' + os.environ.get('PATH', '') + '\n'
    try:
      for root, dirs, files in os.walk('config'):
        if root == 'config':
          dirs.remove('examples')
        for f in files:
          if not f.endswith('.py') or f.startswith('.') or f.startswith('#'):
            continue
          fname = os.path.join(root, f)
          with open(fname,'rb') as f:
            hash += hashlib.sha256(f.read()).hexdigest() + '  ' + fname + '\n'
    except:
      self.logPrint('Error generating file list/hash from config directory for configure hash, forcing new configuration')
      self.makeDependency(None,hashfile)
      return
    if self.argDB['force']:
      self.makeDependency(hash,hashfile)
      return
    a = ''
    try:
      with open(hashfile, 'r') as f:
        a = f.read()
    except:
      # no previous record so write current hash
      self.makeDependency(hash,hashfile)
      return
    if a == hash:
      self.logPrint('configure hash file: '+hashfile+' matches; no need to run configure.')
      print('Your configure options and state has not changed; no need to run configure')
      print('However you can force a configure run using the option: --force')
      sys.exit()
    self.makeDependency(hash,hashfile)
    self.logPrint('configure hash file: '+hashfile+' does not match\n'+a+'\n---\n'+hash+'\n need to run configure')

  def configure(self):
    self.executeTest(self.configureArchitecture)
    # required by top-level configure.py
    self.framework.arch = self.arch
    self.checkDependency()
    return
