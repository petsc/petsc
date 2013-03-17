'''
A Template is a default mechanism for constructing a BuildGraph. They are similar in spirit to
the default Make rules. Users with simple needs can use them to automatically build a project.
Any template should have a getTarget() call which provides a BuildGraph.
'''
import base
import build.buildGraph
import build.compile.SIDL
import project

class ServerRootMap (project.ProjectPath):
  '''This class maps SIDL files to the server root directory they would generate'''
  def __init__(self, projectUrl, language, usingSIDL):
    project.ProjectPath.__init__(self, '', projectUrl)
    self.language  = language
    self.usingSIDL = usingSIDL
    return

  def __call__(self, f):
    import os
    return os.path.join(self.projectRoot, self.usingSIDL.getServerRootDir(self.language, os.path.splitext(os.path.basename(f))[0]))

class Template(base.Base):
  '''This template constructs BuildGraphs capable of compiling SIDL into server and client source'''
  def __init__(self, sourceDB, project, dependenceGraph, usingSIDL = None):
    base.Base.__init__(self)
    self.sourceDB        = sourceDB
    self.project         = project
    self.dependenceGraph = dependenceGraph
    self.usingSIDL       = usingSIDL
    if self.usingSIDL is None:
      import build.templates.usingSIDL
      self.usingSIDL = build.templates.usingSIDL.UsingSIDL(self.sourceDB, self.project)
    return

  def addServer(self, lang):
    '''Designate that a server for lang should be built, which also implies the client'''
    return self.usingSIDL.addServer(lang)

  def addClient(self, lang):
    '''Designate that a client for lang should be built'''
    return self.usingSIDL.addClient(lang)

  def addStaticPackage(self, package):
    '''For a static package, the client is statically linked to the server since dynamic loading is not feasible'''
    return self.usingSIDL.addStaticPackage(package)

  def addRepositoryDirs(self, compiler):
    compiler.repositoryDirs.extend([vertex.getRoot() for vertex in build.buildGraph.BuildGraph.depthFirstVisit(self.dependenceGraph, self.project)])
    return compiler

  def getServerTarget(self, lang):
    target = build.buildGraph.BuildGraph()
    rootFunc   = ServerRootMap(self.project.getUrl(), lang, self.usingSIDL)
    lastVertex = None
    vertex     = build.bk.Tag(rootFunc = rootFunc, inputTag = ['sidl', 'old sidl'])
    target.addEdges(lastVertex, outputs = [vertex])
    lastVertex = vertex
    vertex     = build.bk.Open()
    target.addEdges(lastVertex, outputs = [vertex])
    lastVertex = vertex
    vertex     = self.addRepositoryDirs(build.compile.SIDL.Compiler(self.sourceDB, lang, self.project.getRoot(), 1, self.usingSIDL))
    target.addEdges(lastVertex, outputs = [vertex])
    lastVertex = vertex
    vertex     = build.bk.Tag(rootFunc = rootFunc, inputTag = ['update sidl', 'old sidl'])
    target.addEdges(lastVertex, outputs = [vertex])
    lastVertex = vertex
    vertex     = build.bk.Close()
    target.addEdges(lastVertex, outputs = [vertex])
    return target

  def getServerTargets(self):
    '''Return a BuildGraph which will compile SIDL into the clients specified'''
    import build.bk
    import os

    target = build.buildGraph.BuildGraph()
    for lang in self.usingSIDL.serverLanguages:
      target.addSubgraph(self.getServerTarget(lang))
    return target

  def getClientTarget(self, lang, fullTarget = 0, forceRebuild = 0):
    '''Return a BuildGraph which will compile SIDL into the client specified'''
    target = build.buildGraph.BuildGraph()
    target.addVertex(self.addRepositoryDirs(build.compile.SIDL.Compiler(self.sourceDB, lang, self.project.getRoot(), 0, self.usingSIDL)))
    if fullTarget:
      target.prependGraph(build.buildGraph.BuildGraph([build.fileState.GenericTag(self.sourceDB, 'sidl', ext = 'sidl', force = forceRebuild)]))
      target.appendGraph(build.buildGraph.BuildGraph([build.fileState.Update(self.sourceDB)]))
    return target

  def getClientTargets(self):
    '''Return a BuildGraph which will compile SIDL into the clients specified
       - Currently this graph is just a list of unconnected vertices, which will be linked up in the final target'''
    target = build.buildGraph.BuildGraph()
    for lang in self.usingSIDL.clientLanguages:
      target.addSubgraph(self.getClientTarget(lang))
    return target

  def getTarget(self, forceRebuild = 0):
    '''Return a BuildGraph which will compile SIDL into the servers and clients specified'''
    import build.fileState

    target = build.buildGraph.BuildGraph()
    tagger = build.fileState.GenericTag(self.sourceDB, 'sidl', ext = 'sidl', force = forceRebuild)
    target.addVertex(tagger)
    client = self.getClientTargets()
    server = self.getServerTargets()
    target.addSubgraph(client)
    target.addSubgraph(server)
    target.addEdges(tagger, outputs = build.buildGraph.BuildGraph.getRoots(client)+build.buildGraph.BuildGraph.getRoots(server))
    target.appendGraph(build.buildGraph.BuildGraph([build.fileState.Update(self.sourceDB)]))
    return target
