'''
A Template is a default mechanism for constructing a BuildGraph. They are similar in spirit to
the default Make rules. Users with simple needs can use them to automatically build a project.
Any template should have a getTarget() call which provides a BuildGraph.
'''
import base
import build.buildGraph
import project

class Template(base.Base):
  '''This template constructs BuildGraphs capable of compiling source into object files'''
  def __init__(self, argDB, sourceDB, project, dependenceGraph, usingSIDL, packages):
    import build.compile.SIDL
    base.Base.__init__(self)
    self.argDB           = argDB
    self.sourceDB        = sourceDB
    self.project         = project
    self.dependenceGraph = dependenceGraph
    self.usingSIDL       = usingSIDL
    self.packages        = packages
    self.defines         = []
    self.includeDirs     = []
    self.extraLibraries  = []
    self.serverLanguages = build.compile.SIDL.SIDLLanguageList()
    self.clientLanguages = build.compile.SIDL.SIDLLanguageList()
    return

  def __getattr__(self, name):
    '''Handle requests for nonexistent using<lang> objects'''
    if name.startswith('using'):
      return self.getUsing(name)
    raise AttributeError('No attribute '+name)

  def addServer(self, lang):
    '''Designate that a server for lang should be built'''
    if not lang in self.serverLanguages and not lang in self.usingSIDL.serverLanguages:
      self.serverLanguages.append(lang)
    return

  def addClient(self, lang):
    '''Designate that a client for lang should be built'''
    if not lang in self.clientLanguages and not lang in self.usingSIDL.clientLanguages:
      self.clientLanguages.append(lang)
    return

  def getUsing(self, name):
    '''Create a using<lang> object from build.templates and name it _using<lang>'''
    if hasattr(self, '_'+name):
      return getattr(self, '_'+name)
    cls = 'Using'+name[5:]
    try:
      obj = getattr(__import__('build.templates.'+name, globals(), locals(), [cls]), cls)(self.argDB, self.sourceDB, self.project, self.usingSIDL)
    except ImportError:
      obj = getattr(__import__(name, globals(), locals(), [cls]), cls)(self.argDB, self.sourceDB, self.project, self.usingSIDL)
    setattr(self, '_'+name, obj)
    return obj

  def setupExtraOptions(self, lang, compileGraph):
    '''Set client include directories for all dependencies and the runtime library for linking'''
    import os

    # Hack for excluding build system: Should ask if it is a dependency for Runtime
    useRuntime = not self.project.getUrl() == 'bk://sidl.bkbits.net/BuildSystem'

    for vertex in compileGraph.vertices:
      if hasattr(vertex, 'defines'):
        # Custom defines
        vertex.defines.extend(self.defines)
      if hasattr(vertex, 'includeDirs'):
        dfs = build.buildGraph.BuildGraph.depthFirstVisit(self.dependenceGraph, self.project)
        # Client includes for project dependencies
        vertex.includeDirs.extend([project.ProjectPath(self.usingSIDL.getClientRootDir(lang), v.getUrl()) for v in dfs])
        # Runtime includes
        if useRuntime:
          vertex.includeDirs.extend(self.usingSIDL.getRuntimeIncludes())
        # Custom includes
        vertex.includeDirs.extend(self.includeDirs)
      if hasattr(vertex, 'extraLibraries'):
        if useRuntime:
          if not (self.project == self.usingSIDL.getRuntimeProject() and lang == self.usingSIDL.getRuntimeLanguage()):
            # Runtime libraries
            vertex.extraLibraries.extend(self.usingSIDL.getRuntimeLibraries())
        # Custom libraries
        vertex.extraLibraries.extend(self.extraLibraries)
    return compileGraph

  def getServerTarget(self, lang, package):
    using = getattr(self, 'using'+lang.capitalize())
    return self.setupExtraOptions(lang, using.getServerCompileTarget(package))

  def getServerTargets(self, isStatic = 0):
    '''Return a BuildGraph which will compile the servers specified
       - This is a linear array since all source is independent'''
    target = build.buildGraph.BuildGraph()
    for lang in self.usingSIDL.serverLanguages+self.serverLanguages:
      for package in self.packages:
        if (isStatic and not package in self.usingSIDL.staticPackages) or (not isStatic and package in self.usingSIDL.staticPackages): continue
        target.appendGraph(self.getServerTarget(lang, package))
    return target

  def getClientTarget(self, lang, fullTarget = 0):
    using  = getattr(self, 'using'+lang.capitalize())
    target = self.setupExtraOptions(lang, using.getClientCompileTarget())
    if fullTarget:
      target.appendGraph(build.buildGraph.BuildGraph([build.fileState.Update(self.sourceDB)]))
    return target

  def getClientTargets(self):
    '''Return a BuildGraph which will compile the clients specified
       - This is a linear array since all source is independent'''
    target = build.buildGraph.BuildGraph()
    for lang in self.usingSIDL.clientLanguages+self.clientLanguages:
      target.appendGraph(self.getClientTarget(lang))
    return target

  def getTarget(self):
    '''Return a BuildGraph which will compile source into object files'''
    target = build.buildGraph.BuildGraph()
    target.appendGraph(self.getServerTargets(isStatic = 1))
    target.appendGraph(self.getClientTargets())
    target.appendGraph(self.getServerTargets())
    target.appendGraph(build.buildGraph.BuildGraph([build.fileState.Update(self.sourceDB)]))
    return target

  def getExecutableTarget(self, program):
    '''Return a BuildGraph which will compile user provided source into an executable'''
    target = build.buildGraph.BuildGraph()
    for lang in self.usingSIDL.clientLanguages:
      using = getattr(self, 'using'+lang.capitalize())
      graph = self.setupExtraOptions(lang, using.getExecutableCompileTarget(program))
      target.appendGraph(graph)
    return target

  def install(self):
    for lang in self.usingSIDL.clientLanguages:
      getattr(self, 'using'+lang.capitalize()).installClient()
    for lang in self.usingSIDL.serverLanguages:
      for package in self.packages:
        getattr(self, 'using'+lang.capitalize()).installServer(package)
