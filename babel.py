#!/usr/bin/env python
import bs
import compile
import fileset
import logging
import transform

import os
import string

class CompileSIDL (compile.Process):
  def __init__(self, generatedSources, sources, compiler, compilerFlags, isRepository):
    if isRepository:
      compile.Process.__init__(self, generatedSources, 'sidl', sources, compiler, compilerFlags, 1, 'deferred')
    else:
      compile.Process.__init__(self, generatedSources, 'sidl', sources, compiler, compilerFlags, 0, 'deferred')
    self.outputDir      = 'generated'
    self.repositoryDirs = []
    self.errorHandler   = self.handleBabelErrors

  def handleBabelErrors(self, command, status, output):
    if status or string.find(output, 'Error:') >= 0:
      raise RuntimeError('Could not execute \''+command+'\':\n'+output)

  def constructAction(self, source, baseFlags):
    return baseFlags

  def constructOutputDir(self, source, baseFlags):
    if self.outputDir:
      baseFlags += ' --output-directory='+self.outputDir
    return baseFlags

  def constructRepositoryDir(self, source, baseFlags):
    if self.repositoryDirs:
      baseFlags += ' --repository-path=\"'
      for i in range(len(self.repositoryDirs)):
        dir = os.path.join(self.repositoryDirs[i], 'xml')
        if not os.path.exists(dir): raise RuntimeError('Invalid SIDL repository directory: '+dir)
        baseFlags += dir
        if i < len(self.repositoryDirs)-1: baseFlags += ';'
      baseFlags += '\"'
    return baseFlags

  def constructFlags(self, source, baseFlags):
    baseFlags = '--suppress-timestamp --suppress-metadata '+baseFlags
    baseFlags = self.constructAction(source, baseFlags)
    baseFlags = self.constructOutputDir(source, baseFlags)
    baseFlags = self.constructRepositoryDir(source, baseFlags)
    return baseFlags

class CompileSIDLRepository (CompileSIDL):
  def __init__(self, sources = None, compiler = 'babel', compilerFlags = ''):
    CompileSIDL.__init__(self, None, sources, compiler, compilerFlags, 1)
    self.outputDir = 'xml'

  def constructAction(self, source, baseFlags):
    return baseFlags+' --xml'

class CompileSIDLServer (CompileSIDL):
  def __init__(self, generatedSources, sources = None, compiler = 'babel', compilerFlags = ''):
    CompileSIDL.__init__(self, generatedSources, sources, compiler, compilerFlags, 0)
    self.language = 'C++'

  def constructAction(self, source, baseFlags):
    if self.language:
      baseFlags += ' --server='+self.language
    else:
      raise RuntimeError('No language specified for SIDL server compilation')
    return baseFlags

  def constructOutputDir(self, source, baseFlags):
    if self.outputDir:
      (base, ext) = os.path.splitext(os.path.split(source)[1])
      baseFlags  += ' --output-directory='+self.outputDir+'-'+base
    return baseFlags

  def constructFlags(self, source, baseFlags):
    return '--suppress-stubs '+CompileSIDL.constructFlags(self, source, baseFlags)

class CompileSIDLClient (CompileSIDL):
  def __init__(self, generatedSources = None, sources = None, compiler = 'babel', compilerFlags = ''):
    CompileSIDL.__init__(self, generatedSources, sources, compiler, compilerFlags, 1)
    self.language = 'Python'

  def constructAction(self, source, baseFlags):
    if self.language:
      baseFlags += ' --client='+self.language
    else:
      raise RuntimeError('No language specified for SIDL client compilation')
    return baseFlags

class PythonModuleFixup (transform.Transform):
  def __init__(self, library, pythonDir):
    transform.Transform.__init__(self)
    (base, ext)    = os.path.splitext(library[0])
    self.libName   = base+'.so'
    self.pythonDir = pythonDir

  def copySIDLInterface(self):
    babelPythonDir = os.path.join(bs.argDB['SIDL_DIR'], 'python', 'SIDL')
    sidlDir        = os.path.join(self.pythonDir, 'SIDL')
    if not os.path.exists(sidlDir): os.makedirs(sidlDir)
    for file in os.listdir(babelPythonDir):
      if file[-2:] == '.h':
        command = 'cp '+os.path.join(babelPythonDir, file)+' '+sidlDir
        self.executeShellCommand(command)

  def fileExecute(self, source):
    (dir, file) = os.path.split(source)
    (base, ext) = os.path.splitext(file)
    if not base[-7:] == '_Module': return
    if not ext       == '.c':      return
    package     = base[:-7]
    moduleName  = os.path.join(dir, package+'module.so')
    if not os.path.exists(moduleName) or not os.path.samefile(self.libName, moduleName):
      self.debugPrint('Symlinking '+self.libName+' to '+moduleName, 3, 'compile')
      if os.path.exists(moduleName) or os.path.islink(moduleName):
        os.remove(moduleName)
      os.symlink(self.libName, moduleName)

  def execute(self):
    #self.copySIDLInterface()
    return transform.Transform.execute(self)
