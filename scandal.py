import compile
import transform

import os

class TagSIDL (transform.GenericTag):
  def __init__(self, tag = 'sidl', ext = 'sidl', sources = None, extraExt = ''):
    transform.GenericTag.__init__(self, tag, ext, sources, extraExt)

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
    if status or output.find('Error:') >= 0:
      raise RuntimeError('Could not execute \''+command+'\': '+output)

  def constructAction(self, source, baseFlags):
    return baseFlags

  def constructOutputDir(self, source, baseFlags):
    return baseFlags

  def constructIncludes(self, source, baseFlags):
    if self.repositoryDirs:
      baseFlags += ' -includes=['
      for source in self.repositoryDirs:
        if not os.path.exists(source): raise RuntimeError('Invalid SIDL include directory: '+source)
        baseFlags += source
        if not source == self.repositoryDirs[-1]: baseFlags += ','
      baseFlags += ']'
    return baseFlags

  def constructFlags(self, source, baseFlags):
    baseFlags = ''+baseFlags
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

  def constructAction(self, source, baseFlags):
    if self.language:
      baseFlags += ' -client='+self.language
    else:
      raise RuntimeError('No language specified for SIDL client compilation')
    return baseFlags
