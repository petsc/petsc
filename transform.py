#!/usr/bin/env python
import bs
import fileset
import logging
import maker

import os
import string
import time
import types

class Transform (maker.Maker):
  def __init__(self, sources = None):
    maker.Maker.__init__(self)
    if isinstance(sources, fileset.FileSet):
      self.sources = sources
    else:
      self.sources = fileset.FileSet()
    self.products  = []

  def fileExecute(self, source):
    self.currentSet.append(source)

  def setExecute(self, set):
    self.currentSet = fileset.FileSet(tag = set.tag)
    for file in set.getFiles():
      self.fileExecute(file)
    if len(self.currentSet): self.products.append(self.currentSet)
    return self.products

  def genericExecute(self, sources):
    if type(sources) == types.ListType:
      for set in sources:
        self.genericExecute(set)
    elif isinstance(sources, fileset.FileSet):
      self.setExecute(sources)
    return self.products

  def execute(self):
    self.genericExecute(self.sources)
    if type(self.products) == types.ListType and len(self.products) == 1:
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
    if self.tags and not type(self.tags) == types.ListType:
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
    if self.tags and not type(self.tags) == types.ListType:
      self.tags = [self.tags]

  def setExecute(self, set):
    if not self.tags or not set.tag in self.tags:
      Transform.setExecute(self, set)
    return self.products

class FileChanged (Transform):
  def __init__(self, sources = None):
    Transform.__init__(self, sources)
    self.changed   = fileset.FileSet(tag = 'changed')
    self.unchanged = fileset.FileSet(tag = 'unchanged')
    self.products  = [self.changed, self.unchanged]

  def compare(self, source, sourceEntry):
    self.debugPrint('Checking for '+source+' in the source database', 3, 'sourceDB')
    checksum = self.getChecksum(source)
    if sourceEntry[0] == checksum:
      return 0
    else:
      self.debugPrint(source+' has changed relative to the source database: '+str(sourceEntry[0])+' <> '+str(checksum), 3, 'sourceDB')
      return 1

  def fileExecute(self, source):
    try:
      if not os.path.exists(source):
        self.debugPrint(source+' does not exist', 3, 'sourceDB')
        self.changed.append(source)
      else:
        changed = 0
        if self.compare(source, bs.sourceDB[source]):
          changed = 1
        else:
          for dep in bs.sourceDB[source][3]:
            try:
              if self.compare(dep, bs.sourceDB[dep]):
                changed = 1
                break
            except KeyError:
              pass
        if changed:
          self.changed.append(source)
        else:
          self.unchanged.append(source)
    except KeyError:
      self.debugPrint(source+' does not exist in source database', 3, 'sourceDB')
      self.changed.append(source)

  def execute(self):
    self.debugPrint('Checking for changes to sources '+self.debugFileSetStr(self.sources), 2, 'sourceDB')
    return Transform.execute(self)

class GenericTag (FileChanged):
  def __init__(self, tag, ext, sources = None, extraExt = '', root = None):
    FileChanged.__init__(self, sources)
    if type(ext) == types.ListType:
      self.ext           = map(lambda x: '.'+x, ext)
    else:
      self.ext           = ['.'+ext]
    if type(extraExt) == types.ListType:
      self.extraExt      = map(lambda x: '.'+x, extraExt)
    else:
      self.extraExt      = ['.'+extraExt]
    self.root            = root
    self.changed.tag     = tag
    self.unchanged.tag   = 'old '+tag
    self.deferredUpdates = fileset.FileSet(tag = 'update '+tag)
    self.products        = [self.changed, self.unchanged]

  def fileExecute(self, source):
    (base, ext) = os.path.splitext(source)
    if not self.root or self.root == os.path.commonprefix([os.path.normpath(base), self.root]):
      if ext in self.ext:
        FileChanged.fileExecute(self, source)
      elif ext in self.extraExt:
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
        raise RuntimeError('Invalid tag root directory: '+self.root)
    return FileChanged.execute(self)

class Update (Transform):
  def __init__(self, tags = [], sources = None):
    Transform.__init__(self, sources)
    self.tags   = tags
    if self.tags and not type(self.tags) == types.ListType:
      self.tags = [self.tags]
    self.tags = map(lambda tag: 'update '+tag, self.tags)
    self.products  = []

  def fileExecute(self, source):
    bs.sourceDB.updateSource(source)

  def setExecute(self, set):
    if self.tags and set.tag in self.tags:
      Transform.setExecute(self, set)
    elif set.tag and set.tag[:6] == 'update':
      Transform.setExecute(self, set)
    else:
      if isinstance(self.products, fileset.FileSet):
        self.products = [self.products]
      self.products.append(set)
