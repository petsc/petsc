#!/usr/bin/env python
import fileset
import logging
import maker

import os
import string
import time

class Transform (maker.Maker):
  def __init__(self, sources = None):
    maker.Maker.__init__(self)
    if sources is None:
      self.sources = fileset.FileSet()
    else:
      self.sources = sources
    self.products  = []
    return

  def fileExecute(self, source):
    return self.currentSet.append(source)

  def setExecute(self, set):
    self.currentSet = fileset.FileSet(tag = set.tag)
    for file in set.getFiles():
      self.fileExecute(file)
    if len(self.currentSet): self.products.append(self.currentSet)
    return self.products

  def genericExecute(self, sources):
    if isinstance(sources, list):
      for set in sources:
        self.genericExecute(set)
    elif isinstance(sources, fileset.FileSet):
      self.setExecute(sources)
    return self.products

  def execute(self):
    self.genericExecute(self.sources)
    if isinstance(self.products, list) and len(self.products) == 1:
      self.products = self.products[0]
    return self.products
      
  def getIntermediateFileName(self, source, ext = '.o'):
    (dir, file) = os.path.split(source)
    (base, dum) = os.path.splitext(file)
    return os.path.join(self.tmpDir, string.replace(dir, '/', '_')+'_'+base+ext)

class SimpleFunction (Transform):
  def __init__(self, func):
    Transform.__init__(self, None)
    if not callable(func): raise RuntimeError('Invalid function: '+str(func))
    self.func = func

  def execute(self):
    self.func()
    return Transform.execute(self)

class FileFilter (Transform):
  def __init__(self, filter, sources = None, tags = None):
    Transform.__init__(self, sources)
    self.filter = filter
    self.tags   = tags
    if self.tags and not isinstance(self.tags, list):
      self.tags = [self.tags]

  def fileExecute(self, source):
    if self.tags:
      if (not self.currentSet.tag in self.tags) or (self.filter(source)):
        Transform.fileExecute(self, source)
    else:
      if self.filter(source): Transform.fileExecute(self, source)

class SetFilter (Transform):
  def __init__(self, tags, sources = None):
    Transform.__init__(self, sources)
    self.tags   = tags
    if self.tags and not isinstance(self.tags, list):
      self.tags = [self.tags]

  def setExecute(self, set):
    if not self.tags or not set.tag in self.tags:
      Transform.setExecute(self, set)
    return self.products

class FileChanged (Transform):
  def __init__(self, sourceDB, sources = None):
    Transform.__init__(self, sources)
    self.sourceDB      = sourceDB
    self.changed       = fileset.FileSet(tag = 'changed')
    self.unchanged     = fileset.FileSet(tag = 'unchanged')
    self.products      = [self.changed, self.unchanged]
    self.useUpdateFlag = 0 # self.argDB.has_key('restart') and int(self.argDB['restart'])

  def compare(self, source, sourceEntry):
    if sourceEntry[4] and self.useUpdateFlag:
      self.debugPrint('Update flag indicates '+source+' did not change', 3, 'sourceDB')
      return 0
    else:
      self.debugPrint('Checking for '+source+' in the source database', 3, 'sourceDB')
      checksum = self.sourceDB.getChecksum(source)
      if sourceEntry[0] == checksum:
        return 0
      else:
        self.debugPrint(source+' has changed relative to the source database: '+str(sourceEntry[0])+' <> '+str(checksum), 3, 'sourceDB')
        return 1

  def hasChanged(self, source):
    try:
      if not os.path.exists(source):
        self.debugPrint(source+' does not exist', 3, 'sourceDB')
        return 1
      else:
        changed = 0
        if self.compare(source, self.sourceDB[source]):
          return 1
        else:
          for dep in self.sourceDB[source][3]:
            try:
              if self.compare(dep, self.sourceDB[dep]):
                return 1
                break
            except KeyError:
              pass
        return 0
    except KeyError:
      self.debugPrint(source+' does not exist in source database', 3, 'sourceDB')
      return 1

  def fileExecute(self, source):
    if self.hasChanged(source):
      self.changed.append(source)
    else:
      self.unchanged.append(source)
      self.sourceDB.setUpdateFlag(source)
    return

  def execute(self):
    self.debugPrint('Checking for changes to sources '+self.debugFileSetStr(self.sources), 2, 'sourceDB')
    return Transform.execute(self)

class GenericTag (FileChanged):
  def __init__(self, sourceDB, tag, ext, sources = None, extraExt = '', root = None):
    FileChanged.__init__(self, sourceDB, sources)
    if isinstance(ext, list):
      self.ext           = map(lambda x: '.'+x, ext)
    else:
      self.ext           = ['.'+ext]
    if isinstance(extraExt, list):
      self.extraExt      = map(lambda x: '.'+x, extraExt)
    else:
      self.extraExt      = ['.'+extraExt]
    self.root            = root
    self.changed.tag     = tag
    self.unchanged.tag   = 'old '+tag
    self.deferredUpdates = fileset.FileSet(tag = 'update '+tag)
    self.products        = [self.changed, self.unchanged, self.deferredUpdates]

  def fileExecute(self, source):
    (base, ext) = os.path.splitext(source)
    if not self.root or self.root+os.sep == os.path.commonprefix([os.path.normpath(base), self.root+os.sep]):
      if ext in self.ext:
        FileChanged.fileExecute(self, source)
      elif ext in self.extraExt:
        if self.hasChanged(source):
          self.deferredUpdates.append(source)
      else:
        self.currentSet.append(source)
    else:
      self.currentSet.append(source)

  def setExecute(self, set):
    self.currentSet = fileset.FileSet(tag = set.tag)
    for file in set.getFiles():
      self.fileExecute(file)
    if len(self.currentSet): self.products.append(self.currentSet)
    return self.products

  def execute(self):
    if self.root:
      self.root = os.path.normpath(self.root)
      if not os.path.isdir(self.root):
        raise RuntimeError('Invalid root directory '+self.root+' for tagging operation')
    return FileChanged.execute(self)

class Update (Transform):
  def __init__(self, sourceDB, tags = None, sources = None):
    Transform.__init__(self, sources)
    self.sourceDB = sourceDB
    if tags is None:
      self.tags   = []
    else:
      self.tags   = tags
    if self.tags and not isinstance(self.tags, list):
      self.tags = [self.tags]
    self.tags = map(lambda tag: 'update '+tag, self.tags)
    self.products  = []

  def fileExecute(self, source):
    self.sourceDB.updateSource(source)

  def setExecute(self, set):
    if self.tags and set.tag in self.tags:
      Transform.setExecute(self, set)
    elif set.tag and set.tag[:6] == 'update':
      Transform.setExecute(self, set)
    else:
      if isinstance(self.products, fileset.FileSet):
        self.products = [self.products]
      self.products.append(set)
