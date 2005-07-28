import base

import os

class UsingC (base.Base):
  def __init__(self, argDB, sourceDB, project, usingSIDL):
    import config.base

    base.Base.__init__(self)
    self.language    = 'C'
    self.argDB       = argDB
    self.sourceDB    = sourceDB
    self.project     = project
    self.usingSIDL   = usingSIDL

    self.languageModule     = {}
    self.preprocessorObject = {}
    self.compilerObject     = {}
    self.linkerObject       = {}
    return

  def isCompiled(self):
    '''Returns True is source needs to be compiled in order to execute'''
    return 1

  def getCompileSuffix(self):
    '''Return the suffix for compilable files (.c)'''
    return self.getCompilerObject(self.language).sourceExtension

  def getLinker(self):
    if not hasattr(self, '_linker'):
      return self.argDB[self.getLinkerObject(self.language).name]
    return self._linker
  def setLinker(self, linker):
    self._linker = linker
  linker = property(getLinker, setLinker, doc = 'The linker corresponding to the C compiler')

  def getLinkerFlags(self):
    if not hasattr(self, '_linkerFlags'):
      return self.getLinkerObject(self.language).getFlags()
    return self._linkerFlags
  def setLinkerFlags(self, flags):
    self._linkerFlags = flags
  linkerFlags = property(getLinkerFlags, setLinkerFlags, doc = 'The flags for the C linker')

  #####################
  # Language Operations
  def getLanguageModule(self, language):
    if not language in self.languageModule:
      moduleName = 'config.compile.'+language
      components = moduleName.split('.')
      module     = __import__(moduleName)
      for component in components[1:]:
        module   = getattr(module, component)
      self.languageModule[language] = module
    return self.languageModule[language]

  def getPreprocessorObject(self, language):
    if not language in self.preprocessorObject:
      self.preprocessorObject[language] = self.getLanguageModule(language).Preprocessor(self.argDB)
      self.preprocessorObject[language].checkSetup()
    return self.preprocessorObject[language]

  def getCompilerObject(self, language):
    if not language in self.compilerObject:
      self.compilerObject[language] = self.getLanguageModule(language).Compiler(self.argDB)
      self.compilerObject[language].checkSetup()
    return self.compilerObject[language]

  def getLinkerObject(self, language):
    if not language in self.linkerObject:
      self.linkerObject[language] = self.getLanguageModule(language).Linker(self.argDB)
      self.linkerObject[language].checkSetup()
    return self.linkerObject[language]
