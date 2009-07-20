import config.base

import re
import os

class Configure(config.base.Configure):
  def __init__(self, framework):
    config.base.Configure.__init__(self, framework)
    self.headerPrefix = ''
    self.substPrefix  = ''
    self.version = {}
    self.rejected = {}
    return

  def __str__(self):
    return ''

  def setupHelp(self, help):
    import nargs
    help.addArgument('Compiler Flags', '-optionsModule=<module name>', nargs.Arg(None, 'config.compilerOptions', 'The Python module used to determine compiler options and versions'))
    help.addArgument('Compiler Flags', '-with-debugging=<yes or no>', nargs.ArgBool(None, 1, 'Specify debugging version of libraries'))
    help.addArgument('Compiler Flags', '-C_VERSION',   nargs.Arg(None, 'Unknown', 'The version of the C compiler'))
    help.addArgument('Compiler Flags', '-CXX_VERSION', nargs.Arg(None, 'Unknown', 'The version of the C++ compiler'))
    help.addArgument('Compiler Flags', '-FC_VERSION',  nargs.Arg(None, 'Unknown', 'The version of the Fortran compiler'))
    help.addArgument('Compiler Flags', '-COPTFLAGS',   nargs.Arg(None, None, 'Override the debugging/optimization flags for the C compiler'))
    help.addArgument('Compiler Flags', '-CXXOPTFLAGS', nargs.Arg(None, None, 'Override the debugging/optimization flags for the C++ compiler'))
    help.addArgument('Compiler Flags', '-FOPTFLAGS',   nargs.Arg(None, None, 'Override the debugging/optimization flags for the Fortran compiler'))
    # not sure where to put this, currently gcov is handled in ../compilerOptions.py
    help.addArgument('Compiler Flags', '-with-gcov=<yes or no>', nargs.ArgBool(None, 0, 'Specify that GNUs coverage tool gcov is used'))
    return

  def setupDependencies(self, framework):
    config.base.Configure.setupDependencies(self, framework)
    self.setCompilers = framework.require('config.setCompilers', self)
    return

  def getOptionalFlagsName(self, language, compilerOnly = 0):
    if language == 'C':
      flagsArg = 'COPTFLAGS'
    elif language == 'Cxx':
      if compilerOnly:
        flagsArg = 'CXX_CXXOPTFLAGS'
      else:
        flagsArg = 'CXXOPTFLAGS'
    elif language == 'FC':
      flagsArg = 'FOPTFLAGS'
    else:
      raise RuntimeError('Unknown language: '+language)
    return flagsArg

  def getOptionsObject(self):
    '''Get a configure object which will return default options for each compiler'''
    options = None
    try:
      mod     = __import__(self.framework.argDB['optionsModule'], locals(), globals(), ['CompilerOptions'])
      options = mod.CompilerOptions(self.framework)
    except ImportError:
      self.framework.logPrint('ERROR: Failed to load user options module '+str(self.framework.argDB['optionsModule']))
    return options

  def configureCompilerFlags(self):
    '''Get the default compiler flags'''
    self.debugging = self.argDB['with-debugging']
    bopts = ['']
    if self.debugging:
      bopts.append('g')
    else:
      bopts.append('O')
    options = self.getOptionsObject()
    if not options:
      return
    for language, compiler in [('C', 'CC'), ('Cxx', 'CXX'), ('FC', 'FC')]:
      if not hasattr(self.setCompilers, compiler):
        continue
      self.setCompilers.pushLanguage(language)
      flagsArg = self.getCompilerFlagsArg()
      try:
        self.version[language] = self.argDB[language.upper()+'_VERSION']
        if self.version[language] == 'Unknown':
          self.version[language] = options.getCompilerVersion(language, self.setCompilers.getCompiler())
      except RuntimeError:
        pass
      try:
        self.rejected[language] = []
        for bopt in bopts:
          if not bopt == '' and self.getOptionalFlagsName(language) in self.framework.argDB:
            # treat user supplied optons as single option - as it coud include options separated by spaces '-tp k8-64'
            flags = [self.framework.argDB[self.getOptionalFlagsName(language)]]
          elif bopt == '' and self.getCompilerFlagsName(language) in self.framework.argDB and self.framework.argDB[self.getCompilerFlagsName(language)] != '':
            self.logPrint('Ignoring default options which were overridden using --'+self.getCompilerFlagsName(language)+ ' ' + self.framework.argDB[self.getCompilerFlagsName(language)])
            flags = []
          else:
            flags = options.getCompilerFlags(language, self.setCompilers.getCompiler(), bopt)
          for testFlag in flags:
            try:
              self.framework.logPrint('Trying '+language+' compiler flag '+testFlag)
              self.setCompilers.addCompilerFlag(testFlag)
            except RuntimeError:
              self.framework.logPrint('Rejected '+language+' compiler flag '+testFlag)
              self.rejected[language].append(testFlag)
      except RuntimeError:
        pass
      self.setCompilers.popLanguage()
    return

  def configure(self):
    self.executeTest(self.configureCompilerFlags)
    return
