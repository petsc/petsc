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
    return

  def __str__(self):
    return ''
    
  def configureHelp(self, help):
    import nargs
    help.addArgument('PETSc Compiler Flags', 'optionsModule=<module name>', nargs.Arg(None, None,      'The Python module used to determine compiler options and versions'))
    help.addArgument('PETSc Compiler Flags', 'C_VERSION',                   nargs.Arg(None, 'Unknown', 'The version of the C compiler'))
    help.addArgument('PETSc Compiler Flags', 'CFLAGS_g',                    nargs.Arg(None, 'Unknown', 'Flags for the C compiler with BOPT=g'))
    help.addArgument('PETSc Compiler Flags', 'CFLAGS_O',                    nargs.Arg(None, 'Unknown', 'Flags for the C compiler with BOPT=O'))
    help.addArgument('PETSc Compiler Flags', 'CFLAGS_g_complex',            nargs.Arg(None, 'Unknown', 'Flags for the C compiler with BOPT=g'))
    help.addArgument('PETSc Compiler Flags', 'CFLAGS_O_complex',            nargs.Arg(None, 'Unknown', 'Flags for the C compiler with BOPT=O'))
    help.addArgument('PETSc Compiler Flags', 'CXX_VERSION',                 nargs.Arg(None, 'Unknown', 'The version of the C++ compiler'))
    help.addArgument('PETSc Compiler Flags', 'CXXFLAGS_g',                  nargs.Arg(None, 'Unknown', 'Flags for the C++ compiler with BOPT=g'))
    help.addArgument('PETSc Compiler Flags', 'CXXFLAGS_O',                  nargs.Arg(None, 'Unknown', 'Flags for the C++ compiler with BOPT=O'))
    help.addArgument('PETSc Compiler Flags', 'CXXFLAGS_g_complex',          nargs.Arg(None, 'Unknown', 'Flags for the C compiler with BOPT=g'))
    help.addArgument('PETSc Compiler Flags', 'CXXFLAGS_O_complex',          nargs.Arg(None, 'Unknown', 'Flags for the C compiler with BOPT=O'))
    help.addArgument('PETSc Compiler Flags', 'F_VERSION',                   nargs.Arg(None, 'Unknown', 'The version of the Fortran compiler'))
    help.addArgument('PETSc Compiler Flags', 'FFLAGS_g',                    nargs.Arg(None, 'Unknown', 'Flags for the Fortran compiler with BOPT=g'))
    help.addArgument('PETSc Compiler Flags', 'FFLAGS_O',                    nargs.Arg(None, 'Unknown', 'Flags for the Fortran compiler with BOPT=O'))
    help.addArgument('PETSc Compiler Flags', 'FFLAGS_g_complex',            nargs.Arg(None, 'Unknown', 'Flags for the C compiler with BOPT=g'))
    help.addArgument('PETSc Compiler Flags', 'FFLAGS_O_complex',            nargs.Arg(None, 'Unknown', 'Flags for the C compiler with BOPT=O'))

    self.framework.argDB['PETSCFLAGS'] = ''
    self.framework.argDB['COPTFLAGS']  = ''
    self.framework.argDB['FOPTFLAGS']  = ''
    self.framework.argDB['BOPT']       = 'O'
    return

  def configureCompilerFlags(self):
    '''Get all compiler flags from the Petsc database'''
    options = None
    try:
      mod     = __import__('PETSc.Options', locals(), globals(), ['Options'])
      options = mod.Options(self.framework)
    except ImportError: print 'Failed to load generic options'
    try:
      if self.framework.argDB.has_key('optionsModule'):
        mod     = __import__(self.framework.argDB['optionsModule'], locals(), globals(), ['Options'])
        options = mod.Options(self.framework)
    except ImportError: print 'Failed to load custom options'
    if options:
      for language, flags in [('C', 'CFLAGS'), ('Cxx', 'CXXFLAGS'), ('F77', 'FFLAGS')]:
        # Calling getCompiler() will raise an exception if a language is missing
        self.pushLanguage(language)
        try:
          # Check compiler version
          if language == 'F77':
            versionName = 'F_VERSION'
          else:
            versionName = language.upper()+'_VERSION'
          if self.framework.argDB[versionName] == 'Unknown':
            self.framework.argDB[versionName]  = options.getCompilerVersion(language, self.getCompiler())
          self.addArgumentSubstitution(versionName, versionName)
          # Check normal compiler flags
          self.framework.argDB['REJECTED_'+flags] = []
          for testFlag in options.getCompilerFlags(language, self.getCompiler(), ''):
            try:
              self.addCompilerFlag(testFlag)
            except RuntimeError:
              self.framework.argDB['REJECTED_'+flags].append(testFlag)
          # Check special compiler flags
          for bopt in ['g', 'O', 'g_complex', 'O_complex']:
            flagsName = flags+'_'+bopt
            self.framework.argDB['REJECTED_'+flagsName] = []
            if self.framework.argDB[flagsName] == 'Unknown':
              testFlags = []
              for testFlag in options.getCompilerFlags(language, self.getCompiler(), bopt):
                if self.checkCompilerFlag(testFlag):
                  testFlags.append(testFlag)
                else:
                  self.framework.argDB['REJECTED_'+flagsName].append(testFlag)
              self.framework.argDB[flagsName] = ' '.join(testFlags)
            testFlags = self.framework.argDB[flagsName]
            if not self.checkCompilerFlag(testFlags):
              raise RuntimeError('Invalid '+language+' compiler flags for bopt '+bopt+': '+testFlags)
            self.addArgumentSubstitution(flagsName, flagsName)
        except RuntimeError: pass
        self.popLanguage()

    self.framework.addArgumentSubstitution('PETSCFLAGS', 'PETSCFLAGS')
    self.framework.addArgumentSubstitution('COPTFLAGS',  'COPTFLAGS')
    self.framework.addArgumentSubstitution('FOPTFLAGS',  'FOPTFLAGS')
    return

  def configure(self):
    self.executeTest(self.configureCompilerFlags)
    return
