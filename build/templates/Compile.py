'''
A Template is a default mechanism for constructing a BuildGraph. They are similar in spirit to
the default Make rules. Users with simple needs can use them to automatically build a project.
Any template should have a getTarget() call which provides a BuildGraph.
'''
import base
import build.buildGraph

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
    obj = getattr(__import__('build.templates.'+name, globals(), locals(), [cls]), cls)(self.sourceDB, self.project)
    setattr(self, '_'+name, obj)
    return obj

  def setupIncludes(self, lang, compileGraph):
    '''Include the client directories for all dependencies'''
    import os

    for compiler in compileGraph.vertices:
      if hasattr(compiler, 'includeDirs'):
        dft = build.buildGraph.BuildGraph.depthFirstVisit(self.dependenceGraph, self.project)
        # Client includes for project dependencies
        #compiler.includeDirs.extend([os.path.join(vertex.getRoot(), self.usingSIDL.getClientRootDir(lang)) for vertex in dft if not vertex == self.project])
        compiler.includeDirs.extend([os.path.join(vertex.getRoot(), self.usingSIDL.getClientRootDir(lang)) for vertex in dft])
        # Runtime includes
        compiler.includeDirs.extend(self.usingSIDL.getRuntimeIncludes())
    return compileGraph

  def getServerTarget(self):
    '''Return a BuildGraph which will compile the servers specified
       - This is a linear array since all source is independent'''
    target = build.buildGraph.BuildGraph()
    for lang in self.usingSIDL.serverLanguages:
      for package in self.packages:
        using = getattr(self, 'using'+lang.capitalize())
        graph = self.setupIncludes(lang, using.getServerCompileTarget(package))
        target.appendGraph(graph)
    return target

  def getClientTarget(self):
    '''Return a BuildGraph which will compile the clients specified
       - This is a linear array since all source is independent'''
    target = build.buildGraph.BuildGraph()
    for lang in self.usingSIDL.clientLanguages:
      using = getattr(self, 'using'+lang.capitalize())
      graph = self.setupIncludes(lang, using.getClientCompileTarget())
      target.appendGraph(graph)
    return target

  def getTarget(self):
    '''Return a BuildGraph which will compile source into object files'''
    target = build.buildGraph.BuildGraph()
    target.appendGraph(self.getClientTarget())
    target.appendGraph(self.getServerTarget())
    target.appendGraph(build.buildGraph.BuildGraph([build.fileState.Update(self.sourceDB)]))
    return target
