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

class Operation (Transform):
  '''An Operation applys func to every file in sets matching inputTag'''
  def __init__(self, func, inputTag = None):
    Transform.__init__(self)
    self.func     = func
    self.inputTag = inputTag
    if not isinstance(self.inputTag, list):
      self.inputTag = [self.inputTag]
    return

  def __str__(self):
    return 'Operation('+str(self.func)+') for '+str(self.inputTag)

  def handleFile(self, f, tag):
    '''Call the supplied function on f (also giving tag)
       - If inputTag was specified, only handle files with this tag'''
    if self.inputTag is None or tag in self.inputTag:
      self.func(f, tag)
      return self.output
    return Transform.handleFile(self, f, tag)

class Consolidator (Transform):
  '''A Consolidator combines every file in sets matching inputTag into a single output set'''
  def __init__(self, inputTag, outputTag):
    Transform.__init__(self)
    self.inputTag = inputTag
    if not isinstance(self.inputTag, list):
      self.inputTag = [self.inputTag]
    self.output.tag = outputTag
    return

  def __str__(self):
    return 'Consolidating '+str(self.inputTag)+' into '+self.output.tag

  def handleFile(self, f, tag):
    '''Put all files matching inputTag in the output set'''
    if self.inputTag is None or tag in self.inputTag:
      self.output.append(f)
      return self.output
    return Transform.handleFile(self, f, tag)
