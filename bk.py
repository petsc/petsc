#!/usr/bin/env python
import action
import fileset
import transform

import os
import string

class TagBKOpen (transform.Transform):
  def __init__(self, sources = None, root = None):
    transform.Transform.__init__(self, sources)
    if root: self.root = root
    else:    self.root = os.getcwd()

  def getBKFiles(self, set):
    return string.split(self.executeShellCommand('bk sfiles -g '+self.root))

  def execute(self):
    if isinstance(self.sources, fileset.FileSet): self.products = [self.sources]
    else:                                         self.products = self.sources[:]
    self.products.append(fileset.FileSet(func = self.getBKFiles, tag = 'bkedit'))
    return self.products

class BKOpen (action.Action):
  def __init__(self):
    action.Action.__init__(self, 'bk', None, 'edit', 1, self.checkEdit)
    self.buildProducts = 0

  def checkEdit(self, command, status, output):
    if (status):
      lines    = string.split(output, '\n')
      badLines = ''
      for line in lines:
        if line[0:4] == 'edit:':
          badLines += line
      if badLines:
        raise RuntimeError('Could not execute \''+command+'\': '+output)

  def shellSetAction(self, set):
    if set.tag == 'bkedit':
      action.Action.shellSetAction(self, set)
    else:
      if isinstance(self.products, fileset.FileSet):
        self.products = [self.products]
      self.products.append(set)

class TagBKClose (transform.Transform):
  def __init__(self, sources = None, root = None):
    transform.Transform.__init__(self, sources)
    if root: self.root = root
    else:    self.root = os.getcwd()

  def getNewFiles(self, set):
    fileList = string.split(self.executeShellCommand('bk sfiles -ax '+self.root))
    newFiles = []
    for file in fileList:
      if (file[-1] == '~'): continue
      if (file[0] == '#' and file[-1] == '#'): continue
      newFiles.append(file)
    return newFiles

  def getUnchangedFiles(self, set):
    lockedFiles  = string.split(self.executeShellCommand('bk sfiles -lg '+self.root))
    changedFiles = string.split(self.executeShellCommand('bk sfiles -cg '+self.root))
    map(lockedFiles.remove, changedFiles)
    return lockedFiles

  def execute(self):
    if isinstance(self.sources, fileset.FileSet): self.products = [self.sources]
    else:                                         self.products = self.sources[:]
    self.products.append(fileset.FileSet(func = self.getNewFiles,       tag = 'bkadd'))
    self.products.append(fileset.FileSet(func = self.getUnchangedFiles, tag = 'bkrevert'))
    return self.products

class BKClose (action.Action):
  def __init__(self):
    action.Action.__init__(self, self.close, None, 1)

  def addFiles(self, set):
    self.debugPrint('Putting new files under version control', 2, 'bk')
    if not len(set): return ''
    command = 'bk add '
    for file in set.getFiles():
      self.debugPrint('Adding '+file+' to version control', 3, 'bk')
      command += ' '+file
    output = self.executeShellCommand(command, self.errorHandler)
    command = 'bk co -q '
    for file in set.getFiles(): command += ' '+file
    output = self.executeShellCommand(command, self.errorHandler)
    return output

  def revertFiles(self, set):
    self.debugPrint('Reverting unchanged files', 2, 'bk')
    if not len(set): return ''
    command = 'bk unedit '
    for file in set.getFiles():
      self.debugPrint('Reverting '+file, 4, 'bk')
      command += ' '+file
    output = self.executeShellCommand(command, self.errorHandler)
    command = 'bk co -q '
    for file in set.getFiles(): command += ' '+file
    output = self.executeShellCommand(command, self.errorHandler)
    return output

  def close(self, set):
    if set.tag == 'bkadd':
      self.addFiles(set)
    elif set.tag == 'bkrevert':
      self.revertFiles(set)
    elif set.tag == 'bkcheckin':
      raise RuntimeError('Not yet supported')
    return self.products

  def shellSetAction(self, set):
    if set.tag in ['bkadd', 'bkrevert', 'bkcheckin']:
      action.Action.shellSetAction(self, set)
    else:
      if isinstance(self.products, fileset.FileSet):
        self.products = [self.products]
      self.products.append(set)
