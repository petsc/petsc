import base

import os

class UsingMathematica (base.Base):
  def __init__(self, sourceDB, project, usingCxx = None):
    base.Base.__init__(self)
    self.sourceDB = sourceDB
    self.project  = project
    self.usingCxx = usingCxx
    if self.usingCxx is None:
      import build.templates.usingCxx
      self.usingCxx = build.templates.usingCxx.UsingCxx(self.sourceDB, self.project)
    return

  def getServerCompileTarget(self):
    '''Mathematica code does not need compilation, so only a C++ compiler is necessary for the skeleton.'''
    import build.compile.Cxx
    target = build.buildGraph.BuildGraph()
    tagger = build.fileState.GenericTag(self.sourceDB, 'mathematica server cxx', inputTag = 'mathematica server', ext = 'cc')
    target.addVertex(tagger)
    target.addEdges(tagger, outputs = [build.compile.Cxx.Compiler(self.sourceDB, self.usingCxx, inputTag = 'mathematica server cxx')])
    return target

  def getClientCompileTarget(self):
    '''Mathematica code does not need compilation, so only a C++ compiler is necessary for the cartilage.'''
    import build.compile.Cxx
    target = build.buildGraph.BuildGraph()
    tagger = build.fileState.GenericTag(self.sourceDB, 'mathematica client cxx', inputTag = 'mathematica client', ext = 'cc')
    target.addVertex(tagger)
    target.addEdges(tagger, outputs = [build.compile.Cxx.Compiler(self.sourceDB, self.usingCxx, inputTag = 'mathematica client cxx')])
    return target
