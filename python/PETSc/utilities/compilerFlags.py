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
    help.addArgument('PETSc Compiler Flags', 'C_VERSION',                   nargs.Arg(None, 'Unknown', 'The version of the C compiler'))
    help.addArgument('PETSc Compiler Flags', 'CFLAGS_g',                    nargs.Arg(None, 'Unknown', 'Flags for the C compiler with debugging'))
    help.addArgument('PETSc Compiler Flags', 'CFLAGS_O',                    nargs.Arg(None, 'Unknown', 'Flags for the C compiler without debugging'))
    help.addArgument('PETSc Compiler Flags', 'CFLAGS_g_complex',            nargs.Arg(None, 'Unknown', 'Flags for the C compiler with debugging'))
    help.addArgument('PETSc Compiler Flags', 'CFLAGS_O_complex',            nargs.Arg(None, 'Unknown', 'Flags for the C compiler without debugging'))
    help.addArgument('PETSc Compiler Flags', 'CXX_VERSION',                 nargs.Arg(None, 'Unknown', 'The version of the C++ compiler'))
    help.addArgument('PETSc Compiler Flags', 'CXXFLAGS_g',                  nargs.Arg(None, 'Unknown', 'Flags for the C++ compiler with debugging'))
    help.addArgument('PETSc Compiler Flags', 'CXXFLAGS_O',                  nargs.Arg(None, 'Unknown', 'Flags for the C++ compiler without debugging'))
    help.addArgument('PETSc Compiler Flags', 'CXXFLAGS_g_complex',          nargs.Arg(None, 'Unknown', 'Flags for the C compiler with debugging'))
    help.addArgument('PETSc Compiler Flags', 'CXXFLAGS_O_complex',          nargs.Arg(None, 'Unknown', 'Flags for the C compiler without debugging'))
    help.addArgument('PETSc Compiler Flags', 'F_VERSION',                   nargs.Arg(None, 'Unknown', 'The version of the Fortran compiler'))
    help.addArgument('PETSc Compiler Flags', 'FFLAGS_g',                    nargs.Arg(None, 'Unknown', 'Flags for the Fortran compiler with debugging'))
    help.addArgument('PETSc Compiler Flags', 'FFLAGS_O',                    nargs.Arg(None, 'Unknown', 'Flags for the Fortran compiler without debugging'))
    help.addArgument('PETSc Compiler Flags', 'FFLAGS_g_complex',            nargs.Arg(None, 'Unknown', 'Flags for the C compiler with debugging'))
    help.addArgument('PETSc Compiler Flags', 'FFLAGS_O_complex',            nargs.Arg(None, 'Unknown', 'Flags for the C compiler without debugging'))

    # not sure where to put this, currently gcov is handled in ../compilerOptions.py
    help.addArgument('PETSc', '-with-gcov=<bool>',             nargs.ArgBool(None, 0, 'Specify that GNUs coverage tool gcov is used'))

    self.framework.argDB['PETSCFLAGS'] = ''
    self.framework.argDB['COPTFLAGS']  = ''
    self.framework.argDB['FOPTFLAGS']  = ''
    return

  def configureCompilerFlags(self):
    '''Get all compiler flags from the Petsc database'''
    options = None
    try:
      mod     = __import__('PETSc.compilerOptions', locals(), globals(), ['compilerOptions'])
      options = mod.compilerOptions(self.framework)
    except ImportError:
      self.framework.log.write('Failed to load generic options\n')
      print 'Failed to load generic options'
    try:
      if self.framework.argDB.has_key('optionsModule'):
        mod     = __import__(self.framework.argDB['optionsModule'], locals(), globals(), ['compilerOptions'])
        options = mod.compilerOptions(self.framework)
    except ImportError:
      self.framework.log.write('Failed to load custom options\n')
      print 'Failed to load custom options'
    if options:
      lang = self.clanguage.language
      # Need C++ if using C for building external packages
      if lang == 'C': languages = [('C', 'CFLAGS'), ('FC', 'FFLAGS')]
      else:           languages = [('C', 'CFLAGS'), ('Cxx', 'CXXFLAGS'), ('FC', 'FFLAGS')]
      for language, flags in languages:
        # Calling getCompiler() will raise an exception if a language is missing
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
                self.framework.log.write('Trying '+language+' compiler flag '+testFlag+'\n')
                self.addCompilerFlag(testFlag)
              except RuntimeError:
                self.framework.log.write('Rejected '+language+' compiler flag '+testFlag+'\n')
                self.framework.argDB['REJECTED_'+flags].append(testFlag)
        except RuntimeError: pass
        self.popLanguage()

    self.framework.addArgumentSubstitution('PETSCFLAGS', 'PETSCFLAGS')
    self.framework.addArgumentSubstitution('COPTFLAGS',  'COPTFLAGS')
    self.framework.addArgumentSubstitution('FOPTFLAGS',  'FOPTFLAGS')
    return

  def configure(self):
    self.executeTest(self.configureCompilerFlags)
    return
