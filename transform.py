#!/usr/bin/env python
import bs
import fileset

import os
import string
import time
import types

class ChecksumError (RuntimeError):
  def __init__(self, value):
    self.value = value

  def __str__(self):
    return str(self.value)

class Transform (bs.Maker):
  def __init__(self, sources = None):
    bs.Maker.__init__(self)
    if sources:
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

  def checkChecksumCall(self, command, status, output):
    if (status): raise ChecksumError(output)

  def getBKChecksum(self, source):
    output = self.executeShellCommand('bk checksum -s8 '+source, self.checkChecksumCall)
    return string.split(output)[1]

  def getMD5Checksum(self, source):
    output = self.executeShellCommand('md5sum --binary '+source, self.checkChecksumCall)
    return string.split(output)[0]

  def getChecksum(self, source):
    checksum = 0
    try:
      checksum = self.getBKChecksum(source)
    except ChecksumError:
      try:
        checksum = self.getMD5Checksum(source)
      except ChecksumError:
        pass
    return checksum

class FileFilter (Transform):
  def __init__(self, filter, sources = None, tag = None):
    Transform.__init__(self, sources)
    self.filter = filter
    self.tag    = tag

  def fileExecute(self, source):
    if self.tag:
      if (not self.tag == self.currentSet.tag) or (self.filter(source)):
        Transform.fileExecute(self, source)
    else:
      if self.filter(source): Transform.fileExecute(self, source)

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
  def __init__(self, tag, ext, sources = None, extraExt = ''):
    FileChanged.__init__(self, sources)
    if type(ext) == types.ListType:
      self.ext         = map(lambda x: '.'+x, ext)
    else:
      self.ext         = ['.'+ext]
    if type(extraExt) == types.ListType:
      self.extraExt    = map(lambda x: '.'+x, extraExt)
    else:
      self.extraExt    = ['.'+extraExt]
    self.changed.tag   = tag
    self.unchanged.tag = 'old '+tag
    self.products      = [self.changed, self.unchanged]

  def fileExecute(self, source):
    (base, ext) = os.path.splitext(source)
    if ext in self.ext:
      FileChanged.fileExecute(self, source)
    elif ext in self.extraExt:
      self.deferredUpdates.append(source)
    else:
      self.currentSet.append(source)

  def setExecute(self, set):
    self.currentSet = fileset.FileSet(tag = set.tag)
    for file in set.getFiles():
      self.fileExecute(file)
    if len(self.currentSet): self.products.append(self.currentSet)
    return self.products

  def execute(self):
    self.deferredUpdates = []
    products = FileChanged.execute(self)
    for source in self.deferredUpdates:
      self.updateSourceDB(source)
    return products
