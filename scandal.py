import compile
import transform

import os

class CompileSIDL (compile.Process):
  def __init__(self, generatedSources, sources, compiler, compilerFlags, isRepository):
    if isRepository:
      compile.Process.__init__(self, generatedSources, 'sidl', sources, compiler, compilerFlags, 1, 'deferred')
    else:
      compile.Process.__init__(self, generatedSources, 'sidl', sources, compiler, compilerFlags, 0, 'deferred')
    self.repositoryDirs = []
    self.errorHandler   = self.handleBabelErrors

  def handleBabelErrors(self, command, status, output):
    if status or output.find('Error:') >= 0:
      raise RuntimeError('Could not execute \''+command+'\': '+output)

  def constructAction(self, source, baseFlags):
    return baseFlags

  def constructOutputDir(self, source, baseFlags):
    return baseFlags

  def constructIncludes(self, source, baseFlags):
    if self.repositoryDirs:
      baseFlags += ' -includes=['
      for i in range(len(self.repositoryDirs)):
        dir = os.path.join(self.repositoryDirs[i], 'sidl')
        if not os.path.exists(dir): raise RuntimeError('Invalid SIDL include directory: '+dir)
        for source in os.listdir(dir):
          source  = os.path.join(dir, source)
          if not os.path.exists(source): raise RuntimeError('Invalid SIDL include: '+source)
        baseFlags += source
        if i < len(self.repositoryDirs)-1: baseFlags += ','
      baseFlags += ']'
    return baseFlags

  def constructFlags(self, source, baseFlags):
    baseFlags = self.constructAction(source, baseFlags)
    baseFlags = self.constructOutputDir(source, baseFlags)
    baseFlags = self.constructIncludes(source, baseFlags)
    return baseFlags

class CompileSIDLRepository (CompileSIDL):
  def __init__(self, sources = None, compiler = 'scandal.py', compilerFlags = ''):
    CompileSIDL.__init__(self, None, sources, compiler, compilerFlags, 1)

  def constructAction(self, source, baseFlags):
    return baseFlags+' -updateRepository=1'

class CompileSIDLServer (CompileSIDL):
  def __init__(self, generatedSources, sources = None, compiler = 'scandal.py', compilerFlags = ''):
    CompileSIDL.__init__(self, generatedSources, sources, compiler, compilerFlags, 0)
    self.language = 'C++'

  def constructOutputDir(self, source, baseFlags):
    if self.outputDir:
      (base, ext) = os.path.splitext(os.path.basename(source))
      return baseFlags+' -serverDirs={'+self.language+':'+self.outputDir+'-'+base+'}'
    return baseFlags

  def constructAction(self, source, baseFlags):
    if self.language:
      baseFlags += ' -server='+self.language
    else:
      raise RuntimeError('No language specified for SIDL server compilation')
    return baseFlags

class CompileSIDLClient (CompileSIDL):
  def __init__(self, generatedSources = None, sources = None, compiler = 'scandal.py', compilerFlags = ''):
    CompileSIDL.__init__(self, generatedSources, sources, compiler, compilerFlags, 1)
    self.language = 'Python'

  def constructOutputDir(self, source, baseFlags):
    if self.outputDir:
      return baseFlags+' -clientDirs={'+self.language+':'+self.outputDir+'}'
    return baseFlags

  def constructAction(self, source, baseFlags):
    if self.language:
      baseFlags += ' -client='+self.language
    else:
      raise RuntimeError('No language specified for SIDL client compilation')
    return baseFlags
