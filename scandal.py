import compile
import install.base
import transform

import os

class CompileSIDL (compile.Process, install.base.Base):
  def __init__(self, sourceDB, generatedSources, sources, compiler, compilerFlags, isRepository):
    if isRepository:
      compile.Process.__init__(self, sourceDB, generatedSources, 'sidl', sources, compiler, compilerFlags, 1, 'deferred')
    else:
      compile.Process.__init__(self, sourceDB, generatedSources, 'sidl', sources, compiler, compilerFlags, 0, 'deferred')
    # TODO: Redo this whole initialization process
    #   Maker sets the default argDB, but that means we have to wait for that until figuring out compiler
    #     maybe I should make accessor functions
    if self.compiler is None:
      # used by Process
      self.compiler = self.getCompilerDriver()
      # used by Action
      self.program  = self.compiler
    self.repositoryDirs = []
    self.errorHandler   = self.handleScandalErrors
    self.spawn          = 1
    return

  def getCompilerDriver(self):
    project = self.getInstalledProject('bk://sidl.bkbits.net/Compiler')
    if project is None:
      raise ImportError('Project bk://sidl.bkbits.net/Compiler is not installed')
    return os.path.join(project.getRoot(), 'driver', 'python', 'scandal.py')

  def getCompilerModule(self, name = 'scandal'):
    import imp

    (fp, pathname, description) = imp.find_module(name, [os.path.dirname(self.getCompilerDriver(self))])
    try:
      return imp.load_module(name, fp, pathname, description)
    finally:
      if fp: fp.close()

  def handleScandalErrors(self, command, status, output):
    if status or output.find('Error:') >= 0:
      raise RuntimeError('Could not execute \''+str(command)+'\':\n'+str(output))

  def constructAction(self, source, baseFlags):
    return baseFlags

  def constructOutputDir(self, source, baseFlags):
    return baseFlags

  def constructIncludes(self, source, baseFlags):
    if not self.repositoryDirs: return baseFlags
    sources = []
    for dir in self.repositoryDirs:
      dir = os.path.join(dir, 'sidl')
      if not os.path.exists(dir):
        self.debugPrint('Invalid SIDL include directory: '+dir, 4, 'compile')
        continue
      for source in os.listdir(dir):
        if not os.path.splitext(source)[1] == '.sidl': continue
        source = os.path.join(dir, source)
        if not os.path.exists(source): raise RuntimeError('Invalid SIDL include: '+source)
        sources.append(source)
    arg = ' -includes=['
    for i in range(len(sources)):
      arg += sources[i]
      if i < len(sources)-1: arg += ','
    arg += ']'
    if self.spawn:
      baseFlags += arg
    else:
      baseFlags.append(arg)
    return baseFlags

  def constructFlags(self, source, baseFlags):
    baseFlags = self.constructAction(source, baseFlags)
    baseFlags = self.constructOutputDir(source, baseFlags)
    baseFlags = self.constructIncludes(source, baseFlags)
    return baseFlags

class CompileSIDLRepository (CompileSIDL):
  def __init__(self, sourceDB, sources = None, compiler = None, compilerFlags = ''):
    CompileSIDL.__init__(self, sourceDB, None, sources, compiler, compilerFlags, 1)
    #self.spawn = 0
    #self.flags = []
    return

  def constructAction(self, source, baseFlags):
    if self.spawn:
      return baseFlags+' -updateRepository'
    baseFlags.append('-updateRepository')
    return baseFlags

  def process(self, source):
    import StringIO
    import sys

    self.debugPrint(self.compiler+' processing '+source, 3, 'compile')
    # Compile file
    mod = self.getCompilerModule()
    self.constructFlags(source, self.flags)
    self.flags.append(source)

    oldStdout  = sys.stdout
    sys.stdout = StringIO.StringIO()
    mod.Scandal(self.flags).run()
    sys.stdout = oldStdout

    self.errorHandler('Scandal module run', 0, sys.stdout.getvalue())
    # Update source DB if it compiled successfully
    if self.updateType == 'immediate':
      self.sourceDB.updateSource(source)
    elif self.updateType == 'deferred':
      self.deferredUpdates.append(source)
    return source

class CompileSIDLServer (CompileSIDL):
  def __init__(self, sourceDB, generatedSources, sources = None, compiler = None, compilerFlags = ''):
    CompileSIDL.__init__(self, sourceDB, generatedSources, sources, compiler, compilerFlags, 0)
    self.language = 'C++'
    return

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
  def __init__(self, sourceDB, generatedSources = None, sources = None, compiler = None, compilerFlags = ''):
    CompileSIDL.__init__(self, sourceDB, generatedSources, sources, compiler, compilerFlags, 1)
    self.language = 'Python'
    return

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

class CompileSIDLPrint (CompileSIDL):
  def __init__(self, sourceDB, generatedSources = None, sources = None, compiler = None, compilerFlags = ''):
    CompileSIDL.__init__(self, sourceDB, generatedSources, sources, compiler, compilerFlags, 1)
    self.printer   = 'ANL.SIDLVisitorI.PrettyPrinterHTML'
    self.outputDir = None
    return

  def constructOutputDir(self, source, baseFlags):
    if self.outputDir:
      return baseFlags+' -outputDir='+self.outputDir
    return baseFlags

  def constructAction(self, source, baseFlags):
    return baseFlags+' -resolve=1 -dependency=1 -print='+self.printer
