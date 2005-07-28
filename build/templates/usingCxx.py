import base
import build.buildGraph
import build.processor
import build.transform
import project

import os

class UsingCxx (base.Base):
  def __init__(self, argDB, sourceDB, project, usingSIDL, usingC = None):
    import config.base

    base.Base.__init__(self)
    self.language    = 'Cxx'
    self.argDB       = argDB
    self.sourceDB    = sourceDB
    self.project     = project
    self.usingSIDL   = usingSIDL
    self.usingC      = usingC
    self.clArgs      = None
    self.configBase  = config.base.Configure(self)
    if self.usingC is None:
      import build.templates.usingC
      self.usingC    = build.templates.usingC.UsingC(self.argDB, self.sourceDB, self.project, self.usingSIDL)
    self.setup()
    # Driver may need many outside includes and libraries
    self.programIncludeDirs = {}
    self.programLibraryTags = {}
    self.programLibraries   = {}

    self.languageModule     = {}
    self.preprocessorObject = {}
    self.compilerObject     = {}
    self.linkerObject       = {}
    return

  def __getstate__(self):
    '''Do not save the include directories and extra libraries'''
    d = base.Base.__getstate__(self)
    del d['includeDirs']
    del d['extraLibraries']
    return d

  def __setstate__(self, d):
    '''Recreate the include directories and extra libraries'''
    base.Base.__setstate__(self, d)
    self.setup()
    return

  def setup(self):
    '''Setup include directories and extra libraries'''
    self.setupIncludeDirectories()
    self.setupExtraLibraries()
    return

  def setupIncludeDirectories(self):
    self.includeDirs = []
    return self.includeDirs

  def setupExtraLibraries(self):
    self.extraLibraries = []
    return self.extraLibraries

  def isCompiled(self):
    '''Returns True is source needs to be compiled in order to execute'''
    return 1

  def getCompileSuffix(self):
    '''Return the suffix for compilable files (.cc)'''
    return self.getCompilerObject(self.language).sourceExtension

  def getLinker(self):
    if not hasattr(self, '_linker'):
      return self.argDB[self.getLinkerObject(self.language).name]
    return self._linker
  def setLinker(self, linker):
    self._linker = linker
  linker = property(getLinker, setLinker, doc = 'The linker corresponding to the Cxx compiler')

  def getLinkerFlags(self):
    if not hasattr(self, '_linkerFlags'):
      return self.getLinkerObject(self.language).getFlags()
    return self._linkerFlags
  def setLinkerFlags(self, flags):
    self._linkerFlags = flags
  linkerFlags = property(getLinkerFlags, setLinkerFlags, doc = 'The flags for the Cxx linker')

  def getServerLibrary(self, package, proj = None, lang = None):
    '''Server libraries follow the naming scheme: lib<project>-<lang>-<package>-server.a'''
    if proj is None: proj = self.project
    if lang is None: lang = self.language
    return project.ProjectPath(os.path.join('lib', 'lib'+proj.getName()+'-'+lang.lower()+'-'+package+'-server.a'), proj.getUrl())

  def getExecutableLibrary(self, program):
    '''Executable libraries follow the naming scheme: lib<project>-<lang>-<program>-exec.a'''
    return project.ProjectPath(os.path.join('lib', 'lib'+self.project.getName()+'-'+self.language.lower()+'-'+program+'-exec.a'), self.project.getUrl())

  def getGenericCompileTarget(self, action):
    '''All purposes are in Cxx, so only a Cxx compiler is necessary.'''
    import build.compile.Cxx
    inputTag  = map(lambda a: self.language.lower()+' '+a, action)
    outputTag = self.language.lower()+' '+action[0]+' '+self.language.lower()
    tagger    = build.fileState.GenericTag(self.sourceDB, outputTag, inputTag = inputTag, ext = 'cc', deferredExt = 'hh')
    compiler  = build.compile.Cxx.Compiler(self.sourceDB, self, inputTag = outputTag)
    compiler.includeDirs.extend(self.includeDirs)
    target    = build.buildGraph.BuildGraph()
    target.addVertex(tagger)
    target.addEdges(tagger, outputs = [compiler])
    return (target, compiler)

  def getIORCompileTarget(self, action):
    import build.compile.C
    outputTag = self.language.lower()+' '+action+' '+self.usingC.language.lower()
    tagger    = build.fileState.GenericTag(self.sourceDB, outputTag, inputTag = self.language.lower()+' '+action, ext = 'c', deferredExt = 'h')
    compiler  = build.compile.C.Compiler(self.sourceDB, self.usingC, inputTag = outputTag)
    compiler.includeDirs.extend(self.includeDirs)
    target    = build.buildGraph.BuildGraph()
    target.addVertex(tagger)
    target.addEdges(tagger, outputs = [compiler])
    return (target, compiler)

  def getServerCompileTarget(self, package):
    '''All purposes are in Cxx, so only a Cxx compiler is necessary for the skeleton and implementation.'''
    inputTag      = ['server '+package]
    if len(self.usingSIDL.staticPackages):
      inputTag.append('client')
    (target,    compiler)    = self.getGenericCompileTarget(inputTag)
    (iorTarget, iorCompiler) = self.getIORCompileTarget('server '+package)
    compiler.includeDirs.append(project.ProjectPath(self.usingSIDL.getServerRootDir(self.language, package), self.project.getUrl()))
    inputTags     = [compiler.output.tag, iorCompiler.output.tag]
    archiveTag    = self.language.lower()+' server library directory'
    sharedTag     = self.language.lower()+' server shared library'
    clientTag     = self.language.lower()+' client shared library'
    library       = self.getServerLibrary(package)
    linker        = build.buildGraph.BuildGraph()
    archiver      = build.processor.DirectoryArchiver(self.sourceDB, self, 'cp', inputTags, archiveTag, isSetwise = 1, library = library)
    consolidator  = build.transform.Consolidator(archiveTag, archiveTag, 'old '+archiveTag)
    sharedLinker  = build.processor.SharedLinker(self.sourceDB, self, None, archiveTag, sharedTag, isSetwise = 1, library = library)
    sharedLinker.extraLibraries.extend(self.extraLibraries)
    libraryAdder  = build.processor.LibraryAdder([clientTag, 'old '+clientTag], sharedLinker)
    archiveFilter = build.transform.Filter(archiveTag)
    linker.addVertex(archiver)
    linker.addEdges(consolidator, [archiver])
    linker.addEdges(libraryAdder, [consolidator])
    linker.addEdges(sharedLinker, [libraryAdder])
    linker.addEdges(archiveFilter, [sharedLinker])
    linker.addEdges(build.transform.Remover(inputTags), [archiveFilter])
    target.appendGraph(iorTarget)
    target.appendGraph(linker)
    return target

##  def getClientCompileTarget(self):
##    '''All purposes are in Cxx, so only a Cxx compiler is necessary for the stubs and cartilage.'''
##    if len(self.usingSIDL.staticPackages):
##      return build.buildGraph.BuildGraph()
##    (target, compiler) = self.getGenericCompileTarget(['client'])
##    sharedTag    = self.language.lower()+' client shared library'
##    importTag    = self.language.lower()+' client import library'
##    linker       = build.buildGraph.BuildGraph()
##    sharedLinker = build.processor.SharedLinker(self.sourceDB, self, None, compiler.output.tag, sharedTag)
##    sharedLinker.extraLibraries.extend(self.extraLibraries)
##    importLinker = build.processor.ImportSharedLinker(self.sourceDB, self, None, compiler.output.tag, importTag)
##    sharedAdder  = build.processor.LibraryAdder([importTag, 'old '+importTag], sharedLinker, prepend = 1)
##    linker.addVertex(importLinker)
##    linker.addEdges(sharedAdder,  [importLinker])
##    linker.addEdges(sharedLinker, [sharedAdder])
##    linker.addEdges(build.transform.Remover([compiler.output.tag, compiler.output.tag+' import']), [sharedLinker])
##    target.appendGraph(linker)
##    return target

  def getClientCompileTarget(self):
    '''All purposes are in Cxx, so only a Cxx compiler is necessary for the stubs and cartilage.'''
    if len(self.usingSIDL.staticPackages):
      return build.buildGraph.BuildGraph()
    (target, compiler) = self.getGenericCompileTarget(['client'])
    sharedTag    = self.language.lower()+' client shared library'
    linker       = build.buildGraph.BuildGraph()
    sharedLinker = build.processor.SharedLinker(self.sourceDB, self, None, compiler.output.tag, sharedTag)
    sharedLinker.extraLibraries.extend(self.extraLibraries)
    linker.addVertex(sharedLinker)
    linker.addEdges(build.transform.Remover([compiler.output.tag]), [sharedLinker])
    target.appendGraph(linker)
    return target

  def getExecutableCompileTarget(self, program):
    '''All source should be Cxx'''
    name         = os.path.basename(program)
    prefix       = 'executable '+name
    (target, compiler) = self.getGenericCompileTarget([prefix])
    if name in self.programIncludeDirs:
      compiler.includeDirs.extend(self.programIncludeDirs[name])
    sharedTag    = self.language.lower()+' '+prefix+' shared library'
    clientTag    = self.language.lower()+' client shared library'
    if name in self.programLibraryTags:
      progTags   = self.programLibraryTags[name]
    else:
      progTags   = []
    library      = self.getExecutableLibrary(name)
    linker       = build.buildGraph.BuildGraph()
    sharedLinker = build.processor.SharedLinker(self.sourceDB, self, None, compiler.output.tag, sharedTag, isSetwise = 1, library = library)
    sharedLinker.extraLibraries.extend(self.extraLibraries)
    if name in self.programLibraries:
      sharedLinker.extraLibraries.extend(self.programLibraries[name])
    sharedAdder  = build.processor.LibraryAdder([clientTag, 'old '+clientTag], sharedLinker)
    progLinker   = build.processor.Linker(self.sourceDB, compiler.processor, sharedTag, prefix, isSetwise = 1, library = program)
    progAdder    = build.processor.LibraryAdder([clientTag, 'old '+clientTag]+progTags, progLinker)
    progLinker.extraLibraries.extend(self.extraLibraries)
    if name in self.programLibraries:
      progLinker.extraLibraries.extend(self.programLibraries[name])
    linker.addVertex(sharedAdder)
    linker.addEdges(sharedLinker, [sharedAdder])
    linker.addEdges(progAdder,    [sharedLinker])
    linker.addEdges(progLinker,   [progAdder])
    linker.addEdges(build.transform.Remover(compiler.output.tag), [progLinker])
    target.appendGraph(linker)
    return target

  def installClient(self):
    '''Does nothing right now'''
    return

  def installClasses(self, package):
    for cls in self.usingSIDL.getClasses(package):
      self.project.addImplementation(cls, os.path.join(self.project.getRoot(), self.usingSIDL.getServerLibrary(self.project.getName(), self.language, package, isShared = 1)), self.language)
    return

  def installServer(self, package):
    '''Does nothing right now'''
    self.installClasses(package)
    return

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
