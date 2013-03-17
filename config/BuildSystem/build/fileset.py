import base

import os

class FileSet(list):
  def __init__(self, filenames = None, tag = None, filesets = [], mustExist = 1):
    list.__init__(self)
    self.children  = filesets[:]
    self.tag       = tag
    self.mustExist = mustExist
    if not filenames is None:
      self.extend(filenames)
    return

  def clone(self):
    '''Return a FileSet with the same tag and existence flag, but no members or children'''
    return FileSet(tag = self.tag, mustExist = self.mustExist)

  def checkFile(self, filename):
    '''If mustExist true, check for file existence'''
    if self.mustExist and not os.path.exists(filename):
      raise ValueError('File '+filename+' does not exist!')
    return filename

  def append(self, item):
    item = self.checkFile(item)
    if not item in self:
      list.append(self, item)
    return

  def extend(self, l):
    for item in l:
      self.append(item)
    return

  def insert(self, index, item):
    item = self.checkFile(item)
    if not item in self:
      list.insert(self, index, item)
    return

  def isCompatible(self, set):
    '''Return True if the set tags and mustExist flags match'''
    return (self.tag == set.tag) and (self.mustExist == set.mustExist)

class TreeFileSet (FileSet):
  def __init__(self, roots = None, fileTest = lambda file: 1, tag = None):
    if roots is None:
      self.roots  = FileSet(os.getcwd())
    else:
      if isinstance(roots, str):
        self.roots = FileSet([roots])
      else:
        self.roots = roots
    self.fileTest = fileTest
    FileSet.__init__(self, filenames = self.walkTree(), tag = tag)
    return

  def walkTree(self):
    files = []
    for root in self.roots:
      os.path.walk(root, self.walkFunc, files)
    return files

  def walkFunc(self, defaultFiles, directory, fileList):
    if (os.path.basename(directory) == 'SCCS'): return
    for file in fileList:
      fullPath = os.path.join(directory, file)
      if (os.path.isdir(fullPath)):            continue
      if (file[-1] == '~'):                    continue
      if (file[0] == '#' and file[-1] == '#'): continue
      if (self.fileTest(fullPath)): defaultFiles.append(fullPath)

class ExtensionFileSet (TreeFileSet):
  def __init__(self, roots, exts, tag = None):
    self.exts = exts
    if not isinstance(self.exts, list): self.exts = [self.exts]
    TreeFileSet.__init__(self, roots, self.extTest, tag = tag)
    return

  def extTest(self, file):
    (base, ext) = os.path.splitext(file)
    if (ext in self.exts):
      return 1
    else:
      return 0

class RootedFileSet(FileSet, base.Base):
  def __init__(self, projectUrl, filenames = None, tag = None, filesets = [], mustExist = 1):
    FileSet.__init__(self, None, tag, filesets, mustExist)
    base.Base.__init__(self)
    self.projectUrl = projectUrl
    if not filenames is None:
      self.extend(filenames)
    return

  def __str__(self):
    return '['+','.join(map(str, self))+']'

  def getProjectUrl(self):
    return self._projectUrl

  def setProjectUrl(self, url):
    self._projectUrl = url
  projectUrl = property(getProjectUrl, setProjectUrl, doc = 'The URL of the project which provides a root for all files in the set')

  def getProjectRoot(self):
    if not hasattr(self, '_projectRoot'):
      project = self.getInstalledProject(self.projectUrl)
      if project is None:
        self._projectRoot = ''
      else:
        self._projectRoot = project.getRoot()
    return self._projectRoot

  def setProjectRoot(self):
    raise RuntimeError('Cannot set the project root. It is determined by the project URL.')
  projectRoot = property(getProjectRoot, setProjectRoot, doc = 'The project root for all files in the set')

  def __getstate__(self):
    '''Remove the cached project root directory before pickling'''
    d = base.Base.__getstate__(self)
    if '_projectRoot' in d: del d['_projectRoot']
    return d

  def __getitem__(self, index):
    return os.path.join(self.projectRoot, list.__getitem__(self, index))

  def __getslice__(self, start, end):
    root = self.projectRoot
    return map(lambda f: os.path.join(root, f), list.__getslice__(self, start, end))

  def __setitem__(self, index, item):
    return list.__setitem__(self, index, self.checkFile(item))

  def __setslice__(self, start, end, s):
    root = self.projectRoot
    return list.__setslice__(self, start, end, map(lambda f: self.checkFile(f, root), s))

  def __iter__(self):
    return FileSetIterator(self)

  def clone(self):
    '''Return a RootedFileSet with the same root, tag and existence flag, but no members or children'''
    set = RootedFileSet(self.projectUrl, tag = self.tag, mustExist = self.mustExist)
    set._projectRoot = self._projectRoot
    return set

  def checkFile(self, filename, root = None):
    if root is None:
      root = self.projectRoot
    if os.path.isabs(filename):
      filename = FileSet.checkFile(self, filename)
      if not filename.startswith(root+os.sep):
        raise ValueError('Absolute path '+filename+' conflicts with project root '+root)
      else:
        filename = filename[len(root)+1:]
    else:
      filename = FileSet.checkFile(self, os.path.join(root, filename))
    return filename

  def isCompatible(self, set):
    '''Return True if the roots match and the superclass match returns True'''
    return isinstance(set, RootedFileSet) and (self.projectRoot == set.projectRoot) and FileSet.isCompatible(self, set)

class FileSetIterator (object):
  def __init__(self, set):
    self.set   = set
    self.index = -1
    self.max   = len(set)
    return

  def __iter__(self):
    return self

  def next(self):
    self.index += 1
    if self.index == self.max: raise StopIteration()
    return self.set[self.index]

class RootedExtensionFileSet (RootedFileSet, ExtensionFileSet):
  def __init__(self, projectUrl, roots, exts, tag = None):
    self.exts = exts
    if not isinstance(self.exts, list): self.exts = [self.exts]
    base.Base.__init__(self)
    self.projectUrl = projectUrl
    TreeFileSet.__init__(self, map(lambda d: os.path.join(self.projectRoot, d), roots), self.extTest, tag = tag)
    return
