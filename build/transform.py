import base
import build.fileset

class Transform(base.Base):
  '''This is a generic node in the build graph. It has hooks for processing a FileSet (handleFileSet), or an individual file (handleFile).'''
  def __init__(self):
    '''Reset the node'''
    base.Base.__init__(self)
    self.reset()
    return

  def reset(self):
    '''Clears all state in this node, usually in preparation for another execution'''
    self.output = build.fileset.FileSet()
    return

  def addOutputFile(self, f, tag = None):
    '''Add a file with an optional tag to the correct output set
       - This adds a set with the appropriate tag if necessary'''
    if tag is None:
      return self.output.append(f)
    for child in self.output.children:
      if tag == child.tag:
        return child.append(f)
    return self.output.children.append(build.fileset.FileSet([f], tag = tag))

  def handleFile(self, f, tag = None):
    '''Process a file which has an optional tag associated with it
       - This default method merely adds the file to the output set'''
    return self.addOutputFile(f, tag)

  def handleFileSet(self, set):
    '''Process a FileSet
       - This default method calls handleFile() on each member of the set'''
    map(lambda f: self.handleFile(f, set.tag), set)
    map(self.handleFileSet, set.children)
    return self.output
