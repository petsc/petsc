import base
import build.buildGraph
import build.processor
import build.transform

import os

class UsingPython (base.Base):
  def __init__(self, sourceDB, project, usingC = None):
    base.Base.__init__(self)
    self.sourceDB = sourceDB
    self.project  = project
    self.usingC   = usingC
    if self.usingC is None:
      import build.templates.usingC
      self.usingC = build.templates.usingC.UsingC(self.sourceDB, self.project)
    self.language = 'Python'
    self.setupIncludeDirectories()
    self.setupExtraLibraries()
    return

  def setupArgDB(self, argDB, clArgs):
    import nargs

    argDB.setType('PYTHON_INCLUDE', nargs.ArgDir(None, None, 'The directory containing Python.h', 1))
    argDB.setType('PYTHON_LIB',     nargs.ArgLibrary(None, None, 'The library containing PyInitialize()', 1))
    return base.Base.setupArgDB(self, argDB, clArgs)

  def setupIncludeDirectories(self):
    try:
      if not 'PYTHON_INCLUDE' in self.argDB:
        import distutils.sysconfig
        self.argDB['PYTHON_INCLUDE'] = distutils.sysconfig.get_python_inc()
    except: pass
    self.includeDirs = self.argDB['PYTHON_INCLUDE']
    if not isinstance(self.includeDirs, list):
      self.includeDirs = [self.includeDirs]
    return self.includeDirs

  def setupExtraLibraries(self):
    import distutils.sysconfig
    if not 'PYTHON_LIB' in self.argDB:
      SO = distutils.sysconfig.get_config_var('SO')
      try:
        # Look for the shared library
        lib = os.path.join(distutils.sysconfig.get_config_var('LIBPL'), distutils.sysconfig.get_config_var('LDLIBRARY'))
        # if .so was not built then need to strip .a off of end
        if lib[-2:] == '.a': lib = lib[0:-2]
        # may be stuff after .so like .0, so cannot use splitext()
        lib = lib.split(SO)[0]+SO
        self.argDB['PYTHON_LIB'] = lib
      except TypeError:
        try:
          # Try the archive instead
          lib = lib.split(SO)[0]+'.a'
          self.argDB['PYTHON_LIB'] = lib
        except: pass
      except: pass

    self.extraLibraries = [self.argDB['PYTHON_LIB']]
    if not distutils.sysconfig.get_config_var('LIBS') is None:
      for lib in distutils.sysconfig.get_config_var('LIBS').split():
        # Change -l<lib> to lib<lib>.so
        self.extraLibraries.append('lib'+lib[2:]+'.so')
    return self.extraLibraries

  def getServerLibrary(self, package):
    '''Server libraries follow the naming scheme: lib<project>-<lang>-<package>-server.a'''
    return os.path.join(self.project.getRoot(), 'lib', 'lib'+self.project.getName()+'-'+self.language.lower()+'-'+package+'-server.a')

  def getGenericCompileTarget(self, action):
    '''Python code does not need compilation, so only a C compiler is necessary.'''
    import build.compile.C
    tagger   = build.fileState.GenericTag(self.sourceDB, 'python '+action+' c', inputTag = 'python '+action, ext = 'c', deferredExt = 'h')
    compiler = build.compile.C.Compiler(self.sourceDB, self.usingC, inputTag = 'python '+action+' c')
    compiler.includeDirs.extend(self.includeDirs)
    target   = build.buildGraph.BuildGraph()
    target.addVertex(tagger)
    target.addEdges(tagger, outputs = [compiler])
    return (target, compiler)

  def getServerCompileTarget(self, package):
    '''Python code does not need compilation, so only a C compiler is necessary for the skeleton.'''
    (target, compiler) = self.getGenericCompileTarget('server '+package)
    archiveTag   = self.language.lower()+' server library'
    sharedTag    = self.language.lower()+' server shared library'
    library      = self.getServerLibrary(package)
    linker       = build.buildGraph.BuildGraph()
    archiver     = build.processor.Archiver(self.sourceDB, 'ar', compiler.output.tag, archiveTag, isSetwise = 1, library = library)
    sharedLinker = build.processor.SharedLinker(self.sourceDB, compiler.processor, compiler.output.tag, sharedTag, isSetwise = 1, library = library)
    linker.addVertex(archiver)
    linker.addEdges(sharedLinker, [archiver])
    linker.addEdges(build.transform.Operation(lambda f,tag: os.remove(f), compiler.output.tag), [sharedLinker])
    target.appendGraph(linker)
    return target

  def getClientCompileTarget(self):
    '''Python code does not need compilation, so only a C compiler is necessary for the cartilage.'''
    (target, compiler) = self.getGenericCompileTarget('client')
    archiveTag = self.language.lower()+' client library'
    sharedTag  = self.language.lower()+' client shared library'
    linker     = build.buildGraph.BuildGraph()
    archiver     = build.processor.Archiver(self.sourceDB, 'ar', compiler.output.tag, archiveTag)
    sharedLinker = build.processor.SharedLinker(self.sourceDB, compiler.processor, compiler.output.tag, sharedTag)
    linker.addVertex(archiver)
    linker.addEdges(sharedLinker, [archiver])
    linker.addEdges(build.transform.Operation(lambda f,tag: os.remove(f), compiler.output.tag), [sharedLinker])
    target.appendGraph(linker)
    return target
