from __future__ import generators
import config.base

class Configure(config.base.Configure):
  def __init__(self, framework):
    config.base.Configure.__init__(self, framework)
    self.headerPrefix = ''
    self.substPrefix  = ''
    return

  def __str1__(self):
    if not hasattr(self, 'clanguage'):
      return ''
    desc = ['PETSc:']
    desc.append('  Language used to compile PETSc: ' + self.clanguage)
    desc.append('  Language used to compile PetscDevice: ' + self.devicelanguage)
    return '\n'.join(desc)+'\n'

  def setupHelp(self, help):
    import nargs
    help.addArgument('PETSc', '-with-clanguage=<C or C++>', nargs.Arg(None, 'C', 'Specify C (recommended) or C++ to compile PETSc. You can use C++ in either case.'))
    help.addArgument('PETSc', '-with-devicelanguage=<C or C++>', nargs.Arg(None, None, 'Specify C or C++ to compile PetscDevice. You cannot use C if you either use --with-clanguage=C++ or you are using devices such as NVIDIA GPUs. You cannot use C++ if you use --with-cxx=0.'))
    return

  def setupDependencies(self, framework):
    config.base.Configure.setupDependencies(self, framework)
    return

  def configureCLanguage(self):
    '''Choose whether to compile the PETSc library using a C or C++ compiler'''
    self.clanguage = self.framework.argDB['with-clanguage'].upper().replace('+','x').replace('X','x')
    if not self.clanguage in ['C', 'Cxx']:
      raise RuntimeError('Invalid C language specified: '+str(self.clanguage))
    if self.clanguage == 'Cxx':
      self.logPrintBox('WARNING -with-clanguage=C++ is a developer feature and is *not* required for regular usage of PETSc either from C or C++')
    self.logPrint('C language is '+str(self.clanguage))
    self.addDefine('CLANGUAGE_'+self.clanguage.upper(),'1')
    self.addMakeMacro('CLANGUAGE',self.clanguage.upper())

  def configureDeviceLanguage(self):
    '''Choose whether to compile the PetscDevice code using a C or C++ compiler'''
    if 'with-devicelanguage' in self.argDB:
      self.devicelanguage = self.framework.argDB['with-devicelanguage'].upper().replace('+','x').replace('X','x')
      if not self.devicelanguage in ['C', 'Cxx']:
        raise RuntimeError('Invalid PetscDevice language specified: '+str(self.devicelanguage))
      if self.clanguage == 'Cxx' and self.devicelanguage == 'C':
        raise RuntimeError('Cannot use both --with-clanguage=C++ and --with-devicelanguage=C')
      self.logPrint('PetscDevice language is '+str(self.devicelanguage))
      self.addDefine('DEVICELANGUAGE_'+self.devicelanguage.upper(),'1')
      self.addMakeMacro('DEVICELANGUAGE',self.devicelanguage.upper())
    else:
      self.logPrint('PetscDevice language will be determined once all package dependencies have been configured')
      self.devicelanguage = ''

  def configure(self):
    self.executeTest(self.configureCLanguage)
    self.executeTest(self.configureDeviceLanguage)
    return
