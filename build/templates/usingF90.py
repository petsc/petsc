import base
import build.buildGraph
import build.processor
import build.transform
import project

import os

class UsingF90 (base.Base):
  def __init__(self, sourceDB, project, usingSIDL, usingC = None, usingCxx = None):
    base.Base.__init__(self)
    self.sourceDB   = sourceDB
    self.project    = project
    self.usingSIDL  = usingSIDL
    self.usingC     = usingC
    if self.usingC is None:
      import build.templates.usingC
      self.usingC   = build.templates.usingC.UsingC(self.sourceDB, self.project, self.usingSIDL)
    self.usingCxx   = usingCxx
    if self.usingCxx is None:
      import build.templates.usingCxx
      self.usingCxx = build.templates.usingCxx.UsingCxx(self.sourceDB, self.project, self.usingSIDL, usingC = self.usingC)
    self.language   = 'F90'
    self.setup()
    return

  def __getstate__(self):
    '''Do not save the include directories and extra libraries'''
    d = self.__dict__.copy()
    del d['includeDirs']
    del d['extraLibraries']
    return d

  def __setstate__(self, d):
    '''Recreate the include directories and extra libraries'''
    self.__dict__.update(d)
    self.setup()
    return

  def setup(self):
    '''Setup include directories and extra libraries'''
    self.setupIncludeDirectories()
    self.setupExtraLibraries()
    return

  def setupIncludeDirectories(self):
    self.includeDirs = [project.ArgumentPath('F90_INCLUDE')]
    return self.includeDirs

  def setupExtraLibraries(self):
    self.extraLibraries = [project.ArgumentPath('F90_LIB')]
    return self.extraLibraries

  def getServerLibrary(self, package):
    '''Server libraries follow the naming scheme: lib<project>-<lang>-<package>-server.a'''
    return project.ProjectPath(os.path.join('lib', 'lib'+self.project.getName()+'-'+self.language.lower()+'-'+package+'-server.a'), self.project.getUrl())

  def getExecutableLibrary(self, program):
    '''Executable libraries follow the naming scheme: lib<project>-<lang>-<program>-exec.a'''
    return project.ProjectPath(os.path.join('lib', 'lib'+self.project.getName()+'-'+self.language.lower()+'-'+program+'-exec.a'), self.project.getUrl())

  def getGenericCompileTarget(self, action):
    '''All purposes are in Cxx, so only a F90 compiler is necessary.'''
    import build.buildGraph
    inputTag = map(lambda a: self.language.lower()+' '+a, action)
    target   = build.buildGraph.BuildGraph()
    # Cxx compilation
    import build.compile.Cxx
    outputTag   = self.language.lower()+' '+action[0]+' cxx'
    cxxTagger   = build.fileState.GenericTag(self.sourceDB, outputTag, inputTag = inputTag, ext = 'cc', deferredExt = 'hh')
    cxxCompiler = build.compile.Cxx.Compiler(self.sourceDB, self.usingCxx, inputTag = outputTag)
    cxxCompiler.includeDirs.extend(self.usingCxx.includeDirs)
    target.addVertex(cxxTagger)
    target.addEdges(cxxTagger, outputs = [cxxCompiler])
    # F90 compilation
    import build.compile.F90
    outputTag = self.language.lower()+' '+action[0]+' '+self.language.lower()
    tagger    = build.fileState.GenericTag(self.sourceDB, outputTag, inputTag = inputTag, ext = 'f90')
    compiler  = build.compile.F90.Compiler(self.sourceDB, self, inputTag = outputTag)
    compiler.includeDirs.extend(self.includeDirs)
    target.addEdges(cxxCompiler, outputs = [tagger])
    target.addEdges(tagger,      outputs = [compiler])
    return (target, [compiler, cxxCompiler])

  def getIORCompileTarget(self, action):
    import build.compile.C
    outputTag = self.language.lower()+' '+action+' '+self.usingC.language.lower()
    tagger    = build.fileState.GenericTag(self.sourceDB, outputTag, inputTag = self.language.lower()+' '+action, ext = 'c', deferredExt = 'h')
    compiler  = build.compile.C.Compiler(self.sourceDB, self.usingC, inputTag = outputTag)
    compiler.includeDirs.extend(self.includeDirs)
    target    = build.buildGraph.BuildGraph()
    target.addVertex(tagger)
    target.addEdges(tagger, outputs = [compiler])
    return (target, [compiler])

  def getServerCompileTarget(self, package):
    '''A Cxx compiler is necessary for the skeleton, and anF90 compiler for the implementation.'''
    inputTag      = ['server '+package]
#    if len(self.usingSIDL.staticPackages):
#      inputTag.append('client')
    (target,    compilers)    = self.getGenericCompileTarget(inputTag)
    (iorTarget, iorCompilers) = self.getIORCompileTarget('server '+package)
    for c in compilers: c.includeDirs.append(project.ProjectPath(self.usingSIDL.getServerRootDir(self.language, package), self.project.getUrl()))
    inputTags     = [c.output.tag for c in compilers+iorCompilers]
    archiveTag    = self.language.lower()+' server library directory'
    sharedTag     = self.language.lower()+' server shared library'
    clientTag     = self.language.lower()+' client shared library'
    library       = self.getServerLibrary(package)
    linker        = build.buildGraph.BuildGraph()
    archiver      = build.processor.DirectoryArchiver(self.sourceDB, 'cp', inputTags, archiveTag, isSetwise = 1, library = library)
    consolidator  = build.transform.Consolidator(archiveTag, archiveTag, 'old '+archiveTag)
    sharedLinker  = build.processor.SharedLinker(self.sourceDB, compilers[0].processor, archiveTag, sharedTag, isSetwise = 1, library = library)
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

  def getClientCompileTarget(self):
    '''An F90 compiler is necessary for the stubs, and a Cxx compiler for the cartilage.'''
#    if len(self.usingSIDL.staticPackages):
#      return build.buildGraph.BuildGraph()
    (target, compilers) = self.getGenericCompileTarget(['client'])
    inputTags    = [c.output.tag for c in compilers]
    sharedTag    = self.language.lower()+' client shared library'
    linker       = build.buildGraph.BuildGraph()
    sharedLinker = build.processor.SharedLinker(self.sourceDB, compilers[0].processor, inputTags, sharedTag)
    sharedLinker.extraLibraries.extend(self.extraLibraries)
    linker.addVertex(sharedLinker)
    linker.addEdges(build.transform.Remover(inputTags), [sharedLinker])
    target.appendGraph(linker)
    return target

  def getExecutableCompileTarget(self, program):
    '''All source should be F90'''
    name         = os.path.basename(program)
    prefix       = 'executable '+name
    (target, compilers) = self.getGenericCompileTarget([prefix])
    inputTags    = [c.output.tag for c in compilers]
    sharedTag    = self.language.lower()+' '+prefix+' shared library'
    clientTag    = self.language.lower()+' client shared library'
    library      = self.getExecutableLibrary(name)
    linker       = build.buildGraph.BuildGraph()
    sharedLinker = build.processor.SharedLinker(self.sourceDB, compilers[0].processor, inputTags, sharedTag, isSetwise = 1, library = library)
    sharedLinker.extraLibraries.extend(self.extraLibraries)
    sharedAdder  = build.processor.LibraryAdder([clientTag, 'old '+clientTag], sharedLinker)
    progLinker   = build.processor.Linker(self.sourceDB, compilers[0].processor, sharedTag, prefix, isSetwise = 1, library = program)
    progAdder    = build.processor.LibraryAdder([clientTag, 'old '+clientTag], progLinker)
    linker.addVertex(sharedAdder)
    linker.addEdges(sharedLinker, [sharedAdder])
    linker.addEdges(progAdder,    [sharedLinker])
    linker.addEdges(progLinker,   [progAdder])
    linker.addEdges(build.transform.Remover(inputTags), [progLinker])
    target.appendGraph(linker)
    return target

  def installClient(self):
    '''Does nothing right now'''
    return

  def installServer(self, package):
    '''Does nothing right now'''
    return
