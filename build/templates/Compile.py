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
  def __init__(self, sourceDB, project, dependenceGraph, usingSIDL, packages):
    import build.compile.SIDL
    base.Base.__init__(self)
    self.sourceDB        = sourceDB
    self.project         = project
    self.dependenceGraph = dependenceGraph
    self.usingSIDL       = usingSIDL
    self.packages        = packages
    self.includeDirs     = []
    self.extraLibraries  = []
    return

  def __getattr__(self, name):
    '''Handle requests for nonexistent using<lang> objects'''
    if name.startswith('using'):
      return self.getUsing(name)
    raise AttributeError('No attribute '+name)

  def getUsing(self, name):
    '''Create a using<lang> object from build.templates and name it _using<lang>'''
    if hasattr(self, '_'+name):
      return getattr(self, '_'+name)
    cls = 'Using'+name[5:]
    obj = getattr(__import__('build.templates.'+name, globals(), locals(), [cls]), cls)(self.sourceDB, self.project, self.usingSIDL)
    setattr(self, '_'+name, obj)
    return obj

  def setupExtraOptions(self, lang, compileGraph):
    '''Set client include directories for all dependencies and the runtime library for linking'''
    import os

    # Hack for excluding build system: Should ask if it is a dependency for Runtime
    useRuntime = not self.project.getUrl() == 'bk://sidl.bkbits.net/BuildSystem'

    for vertex in compileGraph.vertices:
      if hasattr(vertex, 'includeDirs'):
        dft = build.buildGraph.BuildGraph.depthFirstVisit(self.dependenceGraph, self.project)
        # Client includes for project dependencies
        vertex.includeDirs.extend([project.ProjectPath(self.usingSIDL.getClientRootDir(lang), v.getUrl()) for v in dft])
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

  def getServerTarget(self, isStatic = 0):
    '''Return a BuildGraph which will compile the servers specified
       - This is a linear array since all source is independent'''
    target = build.buildGraph.BuildGraph()
    for lang in self.usingSIDL.serverLanguages:
      for package in self.packages:
        if (isStatic and not package in self.usingSIDL.staticPackages) or (not isStatic and package in self.usingSIDL.staticPackages): continue
        using = getattr(self, 'using'+lang.capitalize())
        graph = self.setupExtraOptions(lang, using.getServerCompileTarget(package))
        target.appendGraph(graph)
    return target

  def getClientTarget(self):
    '''Return a BuildGraph which will compile the clients specified
       - This is a linear array since all source is independent'''
    target = build.buildGraph.BuildGraph()
    for lang in self.usingSIDL.clientLanguages:
      using = getattr(self, 'using'+lang.capitalize())
      graph = self.setupExtraOptions(lang, using.getClientCompileTarget())
      target.appendGraph(graph)
    return target

  def getTarget(self):
    '''Return a BuildGraph which will compile source into object files'''
    target = build.buildGraph.BuildGraph()
    target.appendGraph(self.getServerTarget(isStatic = 1))
    target.appendGraph(self.getClientTarget())
    target.appendGraph(self.getServerTarget())
    target.appendGraph(build.buildGraph.BuildGraph([build.fileState.Update(self.sourceDB)]))
    return target

  def install(self):
    for lang in self.usingSIDL.clientLanguages:
      getattr(self, 'using'+lang.capitalize()).installClient()
    for lang in self.usingSIDL.serverLanguages:
      for package in self.packages:
        getattr(self, 'using'+lang.capitalize()).installServer(package)
