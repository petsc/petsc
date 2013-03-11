from __future__ import generators

class BuildGraph(object):
  def __init__(self, vertices = []):
    '''Create a graph'''
    self.vertices = []
    self.inEdges  = {}
    self.outEdges = {}
    map(self.addVertex, vertices)
    return

  def __str__(self):
    return 'BuildGraph with '+str(len(self.vertices))+' vertices and '+str(reduce(lambda k,l: k+l, [len(edgeList) for edgeList in self.inEdges.values()], 0))+' edges'

  def addVertex(self, vertex):
    '''Add a vertex if it does not already exist in the vertex list
       - Should be able to use Set in Python 2.3'''
    if vertex is None: return
    if not vertex in self.vertices:
      self.vertices.append(vertex)
      self.clearEdges(vertex)
    return

  def addEdges(self, vertex, inputs = [], outputs = []):
    '''Define the in and out edges for a vertex by listing the other vertices defining the edges
       - If any vertex does not exist in the graph, it is created'''
    self.addVertex(vertex)
    for input in inputs:
      self.addVertex(input)
      if not vertex is None and not input is None:
        if not input  in self.inEdges[vertex]: self.inEdges[vertex].append(input)
        if not vertex in self.outEdges[input]: self.outEdges[input].append(vertex)
    for output in outputs:
      self.addVertex(output)
      if not vertex is None and not output is None:
        if not vertex in self.inEdges[output]:  self.inEdges[output].append(vertex)
        if not output in self.outEdges[vertex]: self.outEdges[vertex].append(output)
    return

  def getEdges(self, vertex):
    return (self.inEdges[vertex], self.outEdges[vertex])

  def clearEdges(self, vertex, inOnly = 0, outOnly = 0):
    if inOnly and outOnly:
      raise RuntimeError('Inconsistent arguments')
    if not outOnly:
      self.inEdges[vertex]  = []
    if not inOnly:
      self.outEdges[vertex] = []
    return

  def removeVertex(self, vertex):
    '''Remove a vertex if already exists in the vertex list
       - Also removes all associated edges'''
    if vertex is None: return
    if vertex in self.vertices:
      self.vertices.remove(vertex)
      del self.inEdges[vertex]
      del self.outEdges[vertex]
      for v in self.vertices:
        if vertex in self.inEdges[v]:  self.inEdges[v].remove(vertex)
        if vertex in self.outEdges[v]: self.outEdges[v].remove(vertex)
    return

  def addSubgraph(self, graph):
    '''Add the vertices and edges of another graph into this one'''
    map(self.addVertex, graph.vertices)
    map(lambda v: apply(self.addEdges, (v,)+graph.getEdges(v)), graph.vertices)
    return

  def removeSubgraph(self, graph):
    '''Remove the vertices and edges of a subgraph, and all the edges connected to it'''
    map(self.removeVertex, graph.vertices)
    return

  def printIndent(self, indent):
    import sys
    for i in range(indent): sys.stdout.write('  ')

  def display(self):
    print 'I am a BuildGraph with '+str(len(self.vertices))+' vertices'
    for vertex in BuildGraph.breadthFirstSearch(self):
      self.printIndent(vertex.__level)
      print '('+str(self.vertices.index(vertex))+') '+str(vertex)+' in: '+str(map(self.vertices.index, self.inEdges[vertex]))+' out: '+str(map(self.vertices.index, self.outEdges[vertex]))
    return

  def appendGraph(self, graph):
    '''Join every leaf of this graph to every root of the input graph, leaving the result in this graph'''
    leaves = BuildGraph.getLeaves(self)
    self.addSubgraph(graph)
    map(lambda v: self.addEdges(v, outputs = BuildGraph.getRoots(graph)), leaves)
    return self

  def prependGraph(self, graph):
    '''Join every leaf of the input graph to every root of this graph, leaving the result in this graph'''
    roots = BuildGraph.getRoots(self)
    self.addSubgraph(graph)
    map(lambda v: self.addEdges(v, outputs = roots), BuildGraph.getLeaves(graph))
    return self

  def getRoots(graph):
    '''Return all the sources in the graph (nodes without entering edges)'''
    return filter(lambda v: not len(graph.getEdges(v)[0]), graph.vertices)
  getRoots = staticmethod(getRoots)

  def getLeaves(graph):
    '''Return all the sinks in the graph (nodes without exiting edges)'''
    return filter(lambda v: not len(graph.getEdges(v)[1]), graph.vertices)
  getLeaves = staticmethod(getLeaves)

  def depthFirstVisit(graph, vertex, seen = None, returnFinished = 0, outEdges = 1):
    '''This is a generator returning vertices in a depth-first traversal only for the subtree rooted at vertex
       - If returnFinished is True, return a vertex when it finishes
       - Otherwise, return a vertex when it is first seen
       - If outEdges is True, proceed along these, otherwise use inEdges'''
    if seen is None: seen = []
    seen.append(vertex)
    if not returnFinished:
      yield vertex
    # Cute trick since outEdges is index 1, and inEdges is index 0
    for v in graph.getEdges(vertex)[outEdges]:
      if not v in seen:
        try:
          for v2 in BuildGraph.depthFirstVisit(graph, v, seen, returnFinished, outEdges):
            yield v2
        except StopIteration:
          pass
    if returnFinished:
      yield vertex
    return
  depthFirstVisit = staticmethod(depthFirstVisit)

  def depthFirstSearch(graph, returnFinished = 0, outEdges = 1):
    '''This is a generator returning vertices in a depth-first traversal
       - If returnFinished is True, return a vertex when it finishes
       - Otherwise, return a vertex when it is first seen
       - If outEdges is True, proceed along these, otherwise use inEdges'''
    seen = []
    for vertex in graph.vertices:
      if not vertex in seen:
        try:
          for v in BuildGraph.depthFirstVisit(graph, vertex, seen, returnFinished, outEdges):
            yield v
        except StopIteration:
          pass
    return
  depthFirstSearch = staticmethod(depthFirstSearch)

  def breadthFirstSearch(graph, returnFinished = 0):
    '''This is a generator returning vertices in a breadth-first traversal
       - If returnFinished is True, return a vertex when it finishes
       - Otherwise, return a vertex when it is first seen'''
    queue = BuildGraph.getRoots(graph)[0:1]
    if not len(queue): return
    seen  = [queue[0]]
    if not returnFinished:
      queue[0].__level = 0
      yield queue[0]
    while len(queue):
      vertex = queue[-1]
      for v in graph.getEdges(vertex)[1]:
        if not v in seen:
          seen.append(v)
          v.__level = vertex.__level + 1
          queue.insert(0, v)
          if not returnFinished:
            yield v
      vertex = queue.pop()
      if returnFinished:
        yield vertex
    return

  def topologicalSort(graph, start = None):
    '''Reorder the vertices using topological sort'''
    if start is None:
      vertices = [vertex for vertex in BuildGraph.depthFirstSearch(graph, returnFinished = 1)]
    else:
      vertices = [vertex for vertex in BuildGraph.depthFirstVisit(graph, start, returnFinished = 1)]
    vertices.reverse()
    for vertex in vertices:
      yield vertex
    return
  topologicalSort = staticmethod(topologicalSort)
