#!/usr/bin/env python
import bs
import compile
import transform

import os

class TagSIDL (transform.GenericTag):
  def __init__(self, tag = 'sidl', ext = 'sidl', sources = None, useAll = 1, extraExt = ''):
    transform.GenericTag.__init__(self, tag, ext, sources, extraExt)
    self.useAll = useAll

  def execute(self):
    self.genericExecute(self.sources)
    if len(self.changed) and self.useAll:
      self.changed.extend(self.unchanged)
      # This is bad
      self.unchanged.data = []
    return self.products

class CompileSIDL (compile.Process):
  def __init__(self, generatedSources, sources, compiler, compilerFlags):
    compile.Process.__init__(self, generatedSources, 'sidl', sources, compiler, '--suppress-timestamp --suppress-metadata '+compilerFlags)
    self.outputDir      = 'generated'
    self.repositoryDirs = []

  def constructArgs(self):
    if self.outputDir:
      self.flags += ' --output-directory='+self.outputDir
    if self.repositoryDirs:
      self.flags += ' --repository-path='
      for dir in self.repositoryDirs:
        if not os.path.exists(dir): raise RuntimeError('Invalid SIDL repository directory: '+dir)
        self.flags += dir
        if not dir == self.repositoryDirs[-1]: self.flags += ';'

  def execute(self):
    self.constructArgs()
    return compile.Process.execute(self)

class CompileSIDLRepository (CompileSIDL):
  def __init__(self, sources = None, compiler = 'babel', compilerFlags = ''):
    CompileSIDL.__init__(self, None, sources, compiler, '--xml '+compilerFlags)
    self.outputDir = 'xml'

class CompileSIDLServer (CompileSIDL):
  def __init__(self, generatedSources, sources = None, compiler = 'babel', compilerFlags = ''):
    CompileSIDL.__init__(self, generatedSources, sources, compiler, compilerFlags)
    self.language = 'C++'

  def constructArgs(self):
    if self.language:
      self.flags += ' --server='+self.language
    else:
      raise RuntimeError('No language specified for SIDL server compilation')
    CompileSIDL.constructArgs(self)

class CompileSIDLClient (CompileSIDL):
  def __init__(self, generatedSources = None, sources = None, compiler = 'babel', compilerFlags = ''):
    CompileSIDL.__init__(self, generatedSources, sources, compiler, compilerFlags)
    self.language = 'Python'

  def constructArgs(self):
    if self.language:
      self.flags += ' --client='+self.language
    else:
      raise RuntimeError('No language specified for SIDL client compilation')
    CompileSIDL.constructArgs(self)

class PythonModuleFixup (transform.Transform):
  def __init__(self, library, pythonDir):
    transform.Transform.__init__(self)
    (base, ext)    = os.path.splitext(library[0])
    self.libName   = base+'.so'
    self.pythonDir = pythonDir

  def copySIDLInterface(self):
    babelPythonDir = os.path.join(bs.argDB['BABEL_DIR'], 'python', 'SIDL')
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
    package     = base[:-7]
    moduleName  = os.path.join(dir, package+'module.so')
    self.debugPrint('Symlinking '+self.libName+' to '+moduleName, 3, 'compile')
    if os.path.exists(moduleName) or os.path.islink(moduleName):
      os.remove(moduleName)
    os.symlink(self.libName, moduleName)

  def execute(self):
    self.copySIDLInterface()
    return transform.Transform.execute(self)
