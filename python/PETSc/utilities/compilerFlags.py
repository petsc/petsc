#!/usr/bin/env python
from __future__ import generators
import user
import config.base

class Configure(config.base.Configure):
  def __init__(self, framework):
    config.base.Configure.__init__(self, framework)
    self.headerPrefix = 'PETSC'
    self.substPrefix  = 'PETSC'
    self.setCompilers = self.framework.require('config.setCompilers', self)
    self.clanguage    = self.framework.require('PETSc.utilities.clanguage', self)
    self.debugging    = self.framework.require('PETSc.utilities.debugging', self)
    return

  def __str__(self):
    return ''
    
  def setupHelp(self, help):
    import nargs
    help.addArgument('PETSc Compiler Flags', 'optionsModule=<module name>', nargs.Arg(None, None,      'The Python module used to determine compiler options and versions'))
    help.addArgument('PETSc Compiler Flags', 'C_VERSION',   nargs.Arg(None, 'Unknown', 'The version of the C compiler'))
    help.addArgument('PETSc Compiler Flags', 'CXX_VERSION', nargs.Arg(None, 'Unknown', 'The version of the C++ compiler'))
    help.addArgument('PETSc Compiler Flags', 'F_VERSION',   nargs.Arg(None, 'Unknown', 'The version of the Fortran compiler'))
    help.addArgument('PETSc Compiler Flags', 'COPTFLAGS',   nargs.Arg(None, None, 'User flags for the C compiler'))
    help.addArgument('PETSc Compiler Flags', 'CXXOPTFLAGS', nargs.Arg(None, None, 'User flags for the C++ compiler'))
    help.addArgument('PETSc Compiler Flags', 'FOPTFLAGS',   nargs.Arg(None, None, 'User flags for the Fortran compiler'))
    # not sure where to put this, currently gcov is handled in ../compilerOptions.py
    help.addArgument('PETSc', '-with-gcov=<bool>',          nargs.ArgBool(None, 0, 'Specify that GNUs coverage tool gcov is used'))
    return

  def configureCompilerFlags(self):
    '''Get all compiler flags from the Petsc database'''
    options = None
    try:
      mod     = __import__('PETSc.compilerOptions', locals(), globals(), ['compilerOptionsFromArgDB'])
      options = mod.compilerOptions(self.framework)
    except ImportError:
      self.framework.logPrint('ERROR: Failed to load PETSc options module')
    try:
      if self.framework.argDB.has_key('optionsModule'):
        mod     = __import__(self.framework.argDB['optionsModule'], locals(), globals(), ['compilerOptions'])
        options = mod.compilerOptions(self.framework)
    except ImportError:
      self.framework.logPrint('ERROR: Failed to load user options module')
    if options:
      languages = [('C', 'CFLAGS'), ('FC', 'FFLAGS')]
      if self.clanguage.language == 'Cxx':
        languages.append(('Cxx', 'CXXFLAGS'))
      for language, flags in languages:
        self.pushLanguage(language)
        try:
          # Check compiler version
          if language == 'FC':
            versionName = 'F_VERSION'
          else:
            versionName = language.upper()+'_VERSION'
          if self.framework.argDB[versionName] == 'Unknown':
            self.framework.argDB[versionName]  = options.getCompilerVersion(language, self.getCompiler())
          self.addArgumentSubstitution(versionName, versionName)
          # Check normal compiler flags
          self.framework.argDB['REJECTED_'+flags] = []
          bopts = ['']
          if self.debugging.debugging:
            bopts.append('g')
          else:
            bopts.append('O')
          for bopt in bopts:
            for testFlag in options.getCompilerFlags(language, self.getCompiler(), bopt):
              try:
                self.framework.logPrint('Trying '+language+' compiler flag '+testFlag+'\n')
                self.addCompilerFlag(testFlag)
              except RuntimeError:
                self.framework.logPrint('Rejected '+language+' compiler flag '+testFlag+'\n')
                self.framework.argDB['REJECTED_'+flags].append(testFlag)
        except RuntimeError: pass
        self.popLanguage()
    # We just give empty string since we already put the flags into CFLAGS, etc.
    self.framework.addSubstitution('COPTFLAGS', '')
    self.framework.addSubstitution('COPTFLAGS', '')
    self.framework.addSubstitution('FOPTFLAGS', '')
    return

  def configure(self):
    self.executeTest(self.configureCompilerFlags)
    return
