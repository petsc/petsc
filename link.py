#!/usr/bin/env python
import action
import fileset
import logging
import transform

import os
import tempfile
import distutils.sysconfig

class TagLibrary (transform.GenericTag):
  def __init__(self, sourceDB, tag = 'lib', ext = 'a', sources = None, extraExt = ''):
    transform.GenericTag.__init__(self, sourceDB, tag, ext, sources, extraExt)

class LinkSharedLibrary (action.Action):
  def __init__(self, sourceDB, sources = None, linker = 'g++', linkerFlags = '-g -shared', archiver = 'ar', archiverFlags = 'x', extraLibraries = None):
    action.Action.__init__(self, self.link, sources, linkerFlags, 0)
    self.sourceDB       = sourceDB
    self.linker         = linker
    self.linkerFlags    = linkerFlags
    self.archiver       = archiver
    self.archiverFlags  = archiverFlags
    if extraLibraries:
      self.extraLibraries = extraLibraries
    else:
      self.extraLibraries = fileset.FileSet()
    self.sharedLibs     = fileset.FileSet()
    self.products       = [self.sharedLibs]
    self.buildProducts  = 0
    self.doLibraryCheck = 1
    self.sharedext      = distutils.sysconfig.get_config_var('SO')
    
  def getSharedName(self, libName):
    (base, ext) = os.path.splitext(libName)
    return base+self.sharedext

  def checkLibrary(self, source):
    if not self.doLibraryCheck: return
    try:
      import BS.LinkCheckerI.Checker
      import BS.LinkError

      try:
        BS.LinkCheckerI.Checker.Checker().openLibrary(source)
      except BS.LinkError.Exception, e:
        raise RuntimeError(e.getMessage())
    except ImportError:
      # If BS is not yet built or unavailable
      self.debugPrint('Did not check shared library '+source, 3, 'link')

  def link(self, source):
    linkDir = tempfile.mktemp('link')
    oldDir  = os.getcwd()
    self.cleanupDir(linkDir)
    os.chdir(linkDir)
    sharedLibrary = self.getSharedName(source)
    self.sharedLibs.append(sharedLibrary)
    self.debugPrint('Linking '+source+' to '+sharedLibrary, 3, 'link')

    command = self.archiver+' '+self.archiverFlags+' '+source
    self.executeShellCommand(command)
    command = self.linker+' '+self.linkerFlags+' -o '+sharedLibrary+' *.o'
    for lib in self.extraLibraries.getFiles():
      if lib[0] == '-' or lib.endswith('.o'):
        command += ' '+lib
      # Big Intel F90 hack
      elif lib.endswith('intrins.a'):
        command += ' '+lib
      else:
        (dir, file) = os.path.split(lib)
        (base, ext) = os.path.splitext(file)
        if dir:
          command += ' -L'+dir+' -Wl,-rpath,'+dir+' -l'+base[3:]
        else:
          command += ' -l'+base[3:]
    self.executeShellCommand(command)
    os.chdir(oldDir)
    self.cleanupDir(linkDir, remove = 1)
    self.checkLibrary(sharedLibrary)
    self.sourceDB.updateSource(source)
    return self.products

  def setExecute(self, set):
    if set.tag == 'lib':
      transform.Transform.setExecute(self, set)
    elif set.tag == 'old lib':
      for file in set:
        sharedLibrary = self.getSharedName(file)
        if not os.path.exists(sharedLibrary):
          self.fileExecute(file)
        else:
          self.sharedLibs.append(sharedLibrary)
    else:
      if isinstance(self.products, fileset.FileSet):
        self.products = [self.products]
      self.products.append(set)
    return self.products

class TagShared (transform.GenericTag):
  def __init__(self, sourceDB, tag = 'sharedlib', ext = 'so', sources = None, extraExt = ''):
    transform.GenericTag.__init__(self, sourceDB, tag, ext, sources, extraExt)

class LinkExecutable (action.Action):
  def __init__(self, executable, sources = None, linker = 'g++', linkerFlags = '', extraLibraries = None):
    action.Action.__init__(self, self.link, sources, linkerFlags, setwiseExecute = 1)
    self.executable    = executable
    self.linker        = linker
    if extraLibraries:
      self.extraLibraries = extraLibraries
    else:
      self.extraLibraries = fileset.FileSet()
    self.rebuildAll    = 0
    self.products      = executable
    self.buildProducts = 0

  def constructFlags(self, source, baseFlags):
    return '-o '+self.executable[0]+' '+baseFlags

  def link(self, set):
    command = self.linker+' '+self.constructFlags(set, self.flags)
    files   = set.getFiles()
    if files:
      self.debugPrint('Linking '+str(files)+' into '+self.executable[0], 3, 'link')
      for file in files:
        command += ' '+file
      for lib in self.extraLibraries:
        (dir, file) = os.path.split(lib)
        (base, ext) = os.path.splitext(file)
        if dir:
          command += ' -L'+dir+' -l'+base[3:]
        else:
          command += ' -l'+base[3:]
      output = self.executeShellCommand(command, self.errorHandler)
#    for source in files:
#      self.sourceDB.updateSource(source)
    return self.products

  def setAction(self, set):
    if set.tag == 'sharedlib':
      action.Action.setAction(self, set)
    else:
      if isinstance(self.products, fileset.FileSet):
        self.products = [self.products]
      self.products.append(set)
    return self.products

  def execute(self):
    if self.executable:
      (dir, file) = os.path.split(self.executable[0])
      if not os.path.exists(dir): os.makedirs(dir)
    return action.Action.execute(self)
