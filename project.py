import base

import os

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
    self.packages   = []
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

  def appendPackages(self, packages):
    '''Appends package names'''
    self.packages += packages

  def getPackages(self):
    '''Gets package names'''
    return self.packages

  def getMatlabPath(self):
    '''Return the path for the matlab directory for this project'''
    return self.matlabPath

  def setMatlabPath(self,path):
    '''Sets the path for the matlab directory for this project'''
    self.matlabPath = path

class ProjectPath (base.Base):
  '''This class represents a relocatable path based upon a project root. If not project is specified,
then the path remains unchanged. If an absolute path is given which conflicts with the project root,
a ValueError is raised.'''
  def __init__(self, path, projectUrl = None):
    base.Base.__init__(self)
    self.projectUrl = projectUrl
    self.path       = path
    return

  def __str__(self):
    return self.path

  def __getstate__(self):
    '''Remove the cached project root directory before pickling'''
    d = base.Base.__getstate__(self)
    if '_projectRoot' in d: del d['_projectRoot']
    return d

  def getProjectRoot(self):
    if not hasattr(self, '_projectRoot'):
      self._projectRoot = self.getInstalledProject(self.projectUrl).getRoot()
    return self._projectRoot

  def setProjectRoot(self):
    raise RuntimeError('Cannot set the project root. It is determined by the project URL.')
  projectRoot = property(getProjectRoot, setProjectRoot, doc = 'The project root for all files in the set')

  def getPath(self):
    path = self._path
    if not self.projectUrl is None:
      path = os.path.join(self.projectRoot, path)
    return path

  def setPath(self, path):
    if not self.projectUrl is None and path[0] == '/':
      root = self.projectRoot
      if not path.startswith(root+os.sep):
        raise ValueError('Absolute path '+path+' conflicts with project root '+root)
      else:
        path = path[len(root)+1:]
    self._path = path
    return
  path = property(getPath, setPath, doc = 'The absolute path, however it can be set as a path relative to the project')
