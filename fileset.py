#!/usr/bin/env python
import bs

import os
import types

class FileSet:
  def __init__(self, data = [], func = None, children = [], tag = None):
    self.data     = data[:]
    self.func     = func
    self.children = children[:]
    self.tag      = tag
    self.cache    = None

  # The problem here is that FileSets are defined dynamically. We cannot just emulate
  # __getitem__ because the underlying definition of the set may change (the filesystem
  # could be altered, etc.). Thus the user must ask for the entire set at once.
  #   An alternative would be to freshen a set cache after each call to __len__. This
  # would update it for all loops I think, and also slices (since you must know the
  # length to request a slice)
  def getFiles(self):
    funcData = []
    if self.func:
      funcData = self.func(self)
    childData = []
    for child in self.children:
      childData += child.getFiles()
    self.cache = self.data+funcData+childData
    return self.cache

  def __len__(self):
    # This freshens the file cache
    return len(self.getFiles())

  def __getitem__(self, key):
    # This freshens the file cache if necessary
    if not self.cache: self.getFiles()
    return self.cache[key]

  def __setitem__(self, key, val):
    self.data[key] = val
    self.cache = None

  def __delitem__(self, key):
    del self.data[key]
    self.cache = None

  def append(self, item):
    if not item in self.data:
      self.data.append(item)
      self.cache = None

  def extend(self, list):
    if isinstance(list, FileSet):
      for item in list.getFiles():
        self.append(item)
    else:
      for item in list:
        self.append(item)
    self.cache = None

  def remove(self, item):
    self.data.remove(item)
    self.cache = None

class TreeFileSet (FileSet):
  def __init__(self, roots = None, fileTest = lambda file: 1):
    FileSet.__init__(self, func = self.walkTree)
    if roots:
      if type(roots) == types.StringType:
        self.roots  = FileSet(roots)
      else:
        self.roots  = roots
    else:
      self.roots  = FileSet(os.path.getcwd())
    self.fileTest = fileTest

  def walkTree(self, fileSet):
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
  def __init__(self, roots, exts):
    TreeFileSet.__init__(self, roots, self.extTest)
    self.exts = exts
    if not type(self.exts) == types.ListType:
      self.exts = [self.exts]

  def extTest(self, file):
    (base, ext) = os.path.splitext(file)
    if (ext in self.exts):
      return 1
    else:
      return 0
