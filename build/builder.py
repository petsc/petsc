import base

class Builder (base.Base):
  def __init__(self, buildGraph = None):
    base.Base.__init__(self)
    self.buildGraph = buildGraph
    return

  def execute(self, start = None, input = None):
    '''Execute the topologically sorted build graph, optionally starting from the transform "start" with the optional FileSet "input"'''
    import build.buildGraph

    started = 0
    self.debugPrint('Starting build', 1, 'build')
    for vertex in build.buildGraph.BuildGraph.topologicalSort(self.buildGraph):
      self.debugPrint('Executing vertex '+str(vertex), 2, 'build')
      if not started:
        if not start is None and not vertex == start:
          continue
        started = 1
        if not input is None:
          self.debugPrint('Processing initial input '+str(input), 3, 'build')
          vertex.handleFileSet(input)
      for parent in self.buildGraph.getEdges(vertex)[0]:
        self.debugPrint('Processing input '+self.debugFileSetStr(parent.output)+' from vertex: '+str(parent), 3, 'build')
        vertex.handleFileSet(parent.output)
      self.debugPrint('Generated output '+self.debugFileSetStr(vertex.output)+' from vertex: '+str(vertex), 3, 'build')
    return
