#!/usr/bin/env python
import action
import fileset
import logging
import transform

import os
import string
import distutils.sysconfig

def cygwinFix(file):
  #  bad check for using cygwin!!
  if distutils.sysconfig.get_config_var('SO') == '.so': return file
  else:
    if not file[0:6] == '/cygwin': return '/cygwin'+file
    else:                          return file

class TagBK (transform.Transform):
  def __init__(self, mode, sources = None, roots = None):
    transform.Transform.__init__(self, sources)
    self.mode = mode
    if roots:
      self.roots = roots
    else:
      self.roots = fileset.FileSet([os.getcwd()])

  def getExistingFiles(self, set):
    files = []
    for root in self.roots:
      if not os.path.isdir(root): raise RuntimeError("Invalid BK source root directory: "+root)
      files.extend(string.split(self.executeShellCommand('bk sfiles -g '+root)))
    return files

  def getNewFiles(self, set):
    newFiles = []
    for root in self.roots:
      fileList = string.split(self.executeShellCommand('bk sfiles -ax '+root))
      for file in fileList:
        if (file[-1] == '~'): continue
        if (file[-1] == '#'): continue
        newFiles.append(file)
    return newFiles

  def getUnchangedFiles(self, set):
    lockedFiles  = []
    changedFiles = []
    for root in self.roots:
      lockedFiles.extend(string.split(self.executeShellCommand('bk sfiles -lg '+root)))
      changedFiles.extend(string.split(self.executeShellCommand('bk sfiles -cg '+root)))
    map(lockedFiles.remove, changedFiles)
    return lockedFiles

  def execute(self):
    if isinstance(self.sources, fileset.FileSet):
      self.products = [self.sources]
    else:
      self.products = self.sources[:]
    if self.mode == 'open':
      self.products.append(fileset.FileSet(func = self.getExistingFiles, tag = 'bkedit'))
    elif self.mode == 'close':
      self.products.append(fileset.FileSet(func = self.getNewFiles,       tag = 'bkadd'))
      self.products.append(fileset.FileSet(func = self.getUnchangedFiles, tag = 'bkrevert'))
    else:
      raise RuntimeError('Invalid BK tag mode: '+self.mode)
    return self.products

class TagBKOpen (TagBK):
  def __init__(self, sources = None, roots = None):
    TagBK.__init__(self, 'open', sources, roots)
      print 'Opening '+file

class TagBKClose (TagBK):
  def __init__(self, sources = None, roots = None):
    TagBK.__init__(self, 'close', sources, roots)

class BKOpen (action.Action):
  def __init__(self):
    action.Action.__init__(self, self.open, None, '', 1)
    self.buildProducts = 0

  def open(self, set):
    self.debugPrint('Opening files', 2, 'bk')
    if not len(set): return ''
    command = 'bk edit '
    for file in set.getFiles():
      self.debugPrint('Opening '+file, 4, 'bk')
      file = cygwinFix(file)
      command += ' '+file
    output = self.executeShellCommand(command, self.checkEdit)
    return output

  def checkEdit(self, command, status, output):
    if (status):
      lines    = string.split(output, '\n')
      badLines = ''
      for line in lines:
        if line[0:4] == 'edit:':
          badLines += line
      if badLines:
        raise RuntimeError('Could not execute \''+command+'\':\n'+output)

  def shellSetAction(self, set):
    if set.tag == 'bkedit':
      action.Action.shellSetAction(self, set)
    else:
      if isinstance(self.products, fileset.FileSet):
        self.products = [self.products]
      self.products.append(set)

class BKClose (action.Action):
  def __init__(self):
    action.Action.__init__(self, self.close, None, 1)
    self.buildProducts = 0

  def addFiles(self, set):
    self.debugPrint('Putting new files under version control', 2, 'bk')
    if not len(set): return ''
    command = 'bk add '
    for file in set.getFiles():
      self.debugPrint('Adding '+file+' to version control', 3, 'bk')
      file = cygwinFix(file)
      command += ' '+file
    output = self.executeShellCommand(command, self.errorHandler)
    command = 'bk co -q '
    for file in set.getFiles(): 
      file = cygwinFix(file)
      command += ' '+file
    output = self.executeShellCommand(command, self.errorHandler)
    return output

  def revertFiles(self, set):
    self.debugPrint('Reverting unchanged files', 2, 'bk')
    if not len(set): return ''
    command = 'bk unedit '
    for file in set.getFiles():
      self.debugPrint('Reverting '+file, 4, 'bk')
      file = cygwinFix(file)
      command += ' '+file
    output = self.executeShellCommand(command, self.errorHandler)
    command = 'bk co -q '
    for file in set.getFiles(): 
      file = cygwinFix(file)
      command += ' '+file
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

  def setAction(self, set):
    if set.tag in ['bkadd', 'bkrevert', 'bkcheckin']:
      action.Action.setAction(self, set)
    else:
      if isinstance(self.products, fileset.FileSet):
        self.products = [self.products]
      self.products.append(set)
