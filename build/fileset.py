import os

class FileSet(list):
  def __init__(self, filenames = None, tag = None, filesets = []):
    list.__init__(self)
    if not filenames is None:
      list.extend(self, filenames)
    self.children  = filesets[:]
    self.tag       = tag
    self.mustExist = 1
    return

  def checkFile(self, filename):
    if self.mustExist and not os.path.exists(filename):
      raise RuntimeError('File '+filename+' does not exist!')

  def append(self, item):
    self.checkFile(item)
    if not item in self:
      list.append(self, item)
    return

  def extend(self, l):
    for item in l:
      self.append(item)
    return

  def insert(self, index, item):
    self.checkFile(item)
    if not item in self:
      list.insert(self, index, item)
    return

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
