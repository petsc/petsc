class Project:
  '''This class represents a SIDL project, and is the only BuildSystem class allowed in an RDict'''
  def __init__(self, name, url, root = None):
    import os
    if root is None: root = os.path.abspath(os.getcwd())
    # Read-only variables
    self.name = name # Needs to be immutable since it is the hash key
    self.url  = url
    self.root = root
    # Updated variables
    self.pythonPath = []
    return

  def __str__(self):
    return self.name

  def __hash__(self):
    return self.name.__hash__()

  def __lt__(self, other):
    return self.name.__lt__(other.getName())

  def __le__(self, other):
    return self.name.__le__(other.getName())

  def __eq__(self, other):
    return self.name.__eq__(other.getName())

  def __ne__(self, other):
    return self.name.__ne__(other.getName())

  def __gt__(self, other):
    return self.name.__gt__(other.getName())

  def __ge__(self, other):
    return self.name.__ge__(other.getName())

  def getName(self):
    '''Return the project name, e.g. PETSc'''
    return self.name

  def getUrl(self):
    '''Return the project URL, e.g. bk://petsc.bkbits.net/petsc-dev'''
    return self.url

  def getRoot(self):
    '''Return the root directory of the local installation'''
    return self.root

  def appendPythonPath(self, dir):
    '''Append a directory to the list of paths which must be given to Python for this project to function'''
    import os

    d = os.path.abspath(dir)
    if os.path.exists(d) and not d in self.pythonPath:
      self.pythonPath.append(d)
    return self.pythonPath

  def getPythonPath(self):
    '''Return the list of paths which must be given to Python for this project to function'''
    return self.pythonPath
