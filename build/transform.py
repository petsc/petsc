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

  def addOutputFile(self, f, set = None):
    '''Add a file to the correct output set
       - This adds a set with the appropriate metadata if necessary'''
    if set.isCompatible(self.output):
      return self.output.append(f)
    for child in self.output.children:
      if set.isCompatible(child):
        return child.append(f)
    newSet = set.clone()
    newSet.append(f)
    return self.output.children.append(newSet)

  def handleFile(self, f, set = None):
    '''Process a file which has an optional FileSet associated with it
       - This default method merely adds the file to an output set'''
    return self.addOutputFile(f, set)

  def handleFileSet(self, set):
    '''Process a FileSet
       - This default method calls handleFile() on each member of the set'''
    map(lambda f: self.handleFile(f, set), set)
    map(self.handleFileSet, set.children)
    return self.output

class Filter (Transform):
  '''A Filter removes every file in sets matching inputTag'''
  def __init__(self, inputTag):
    Transform.__init__(self)
    self.inputTag = inputTag
    if not isinstance(self.inputTag, list):
      self.inputTag = [self.inputTag]
    return

  def __str__(self):
    return 'Filter for '+str(self.inputTag)

  def handleFile(self, f, set):
    '''Drop files with inputTag'''
    if set.tag in self.inputTag:
      return self.output
    return Transform.handleFile(self, f, set)

class Remover (Transform):
  '''A Remover removes every file in sets matching inputTag'''
  def __init__(self, inputTag = None):
    Transform.__init__(self)
    self.inputTag = inputTag
    if not isinstance(self.inputTag, list):
      self.inputTag = [self.inputTag]
    return

  def __str__(self):
    return 'Remover for '+str(self.inputTag)

  def handleFile(self, f, set):
    '''Call the supplied function on f (also giving tag)
       - If inputTag was specified, only handle files with this tag'''
    if self.inputTag is None or set.tag in self.inputTag:
      import os
      os.remove(f)
      return self.output
    return Transform.handleFile(self, f, set)

class Consolidator (Transform):
  '''A Consolidator combines every file in sets matching inputTag into a single output set
     - If oldTag is provided, sets matching this tag are added only if at least one file with inputTag is present'''
  def __init__(self, inputTag, outputTag, oldTag = []):
    Transform.__init__(self)
    self.inputTag   = inputTag
    self.oldTag     = oldTag
    if not isinstance(self.inputTag, list):
      self.inputTag = [self.inputTag]
    if not isinstance(self.oldTag, list):
      self.oldTag   = [self.oldTag]
    self.output.tag = outputTag
    if len(self.oldTag):
      self.oldOutput     = build.fileset.FileSet()
      self.oldOutput.tag = self.oldTag[0]
      self.hasOutput     = 0
      self.output.children.append(self.oldOutput)
    return

  def __str__(self):
    return 'Consolidating '+str(self.inputTag)+'('+str(self.oldTag)+') into '+self.output.tag

  def handleFile(self, f, set):
    '''Put all files matching inputTag in the output set'''
    if self.inputTag is None or set.tag in self.inputTag:
      self.output.append(f)
      if not self.hasOutput:
        self.hasOutput = 1
        self.output.children.remove(self.oldOutput)
        self.output.extend(self.oldOutput)
      return self.output
    elif set.tag in self.oldTag:
      if self.hasOutput:
        self.output.append(f)
      else:
        self.oldOutput.append(f)
      return self.output
    return Transform.handleFile(self, f, set)
