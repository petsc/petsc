import base
import build.buildGraph
import build.processor
import build.transform
import project

import os

class UsingPython (base.Base):
  def __init__(self, sourceDB, project, usingSIDL, usingC = None):
    base.Base.__init__(self)
    self.sourceDB  = sourceDB
    self.project   = project
    self.usingSIDL = usingSIDL
    self.usingC    = usingC
    if self.usingC is None:
      import build.templates.usingC
      self.usingC = build.templates.usingC.UsingC(self.sourceDB, self.project, self.usingSIDL)
    self.language = 'Python'
    self.setup()
    return

  def __getstate__(self):
    '''Do not save the include directories and extra libraries'''
    d = self.__dict__.copy()
    del d['includeDirs']
    del d['extraLibraries']
    return d

  def __setstate__(self, d):
    '''Recreate the include directories and extra libraries'''
    self.__dict__.update(d)
    self.setup()
    return

  def setupArgDB(self, argDB, clArgs):
    return base.Base.setupArgDB(self, argDB, clArgs)

  def setup(self):
    '''Setup include directories and extra libraries'''
    self.setupIncludeDirectories()
    self.setupExtraLibraries()
    return

  def setupIncludeDirectories(self):
    try:
      if not 'PYTHON_INCLUDE' in self.argDB:
        import distutils.sysconfig
        self.argDB['PYTHON_INCLUDE'] = distutils.sysconfig.get_python_inc()
    except: pass
    if isinstance(self.argDB['PYTHON_INCLUDE'], list):
      # We need separate includes in separate keys
      self.includeDirs = []
      i = 0
      for dir in self.argDB['PYTHON_INCLUDE']:
        self.argDB['PYTHON_INCLUDE_'+str(i)] = dir
        self.includeDirs.append(project.ArgumentPath('PYTHON_INCLUDE_'+str(i)))
        i += 1
    else:
      self.includeDirs = [project.ArgumentPath('PYTHON_INCLUDE')]
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

    extraLibraries = [self.argDB['PYTHON_LIB']]
    if not distutils.sysconfig.get_config_var('LIBS') is None:
      for lib in distutils.sysconfig.get_config_var('LIBS').split():
        # Change -l<lib> to lib<lib>.so
        extraLibraries.append('lib'+lib[2:]+'.so')

    # We need separate libraries in separate keys
    self.extraLibraries = []
    i = 0
    for lib in extraLibraries:
      self.argDB['PYTHON_LIB_'+str(i)] = lib
      self.extraLibraries.append(project.ArgumentPath('PYTHON_LIB_'+str(i)))
      i += 1
    return self.extraLibraries

  def getServerLibrary(self, package, proj = None, lang = None):
    if proj is None: proj = self.project
    if lang is None: lang = self.language
    '''Server libraries follow the naming scheme: lib<project>-<lang>-<package>-server.a'''
    return project.ProjectPath(os.path.join('lib', 'lib'+proj.getName()+'-'+lang.lower()+'-'+package+'-server.a'), proj.getUrl())

  def getGenericCompileTarget(self, action):
    '''Python code does not need compilation, so only a C compiler is necessary.'''
    import build.compile.C
    outputTag = self.language.lower()+' '+action+' '+self.usingC.language.lower()
    tagger    = build.fileState.GenericTag(self.sourceDB, outputTag, inputTag = self.language.lower()+' '+action, ext = 'c', deferredExt = ['h', 'py'])
    compiler  = build.compile.C.Compiler(self.sourceDB, self.usingC, inputTag = outputTag)
    compiler.includeDirs.extend(self.includeDirs)
    target    = build.buildGraph.BuildGraph()
    target.addVertex(tagger)
    target.addEdges(tagger, outputs = [compiler])
    return (target, compiler)

  def getServerCompileTarget(self, package):
    '''Python code does not need compilation, so only a C compiler is necessary for the skeleton.'''
    (target, compiler) = self.getGenericCompileTarget('server '+package)
    archiveTag    = self.language.lower()+' server library directory'
    sharedTag     = self.language.lower()+' server shared library'
    library       = self.getServerLibrary(package)
    linker        = build.buildGraph.BuildGraph()
    archiver      = build.processor.DirectoryArchiver(self.sourceDB, 'cp', compiler.output.tag, archiveTag, isSetwise = 1, library = library)
    consolidator  = build.transform.Consolidator(archiveTag, archiveTag, 'old '+archiveTag)
    sharedLinker  = build.processor.SharedLinker(self.sourceDB, compiler.processor, archiveTag, sharedTag, isSetwise = 1, library = library)
    if not (self.project.getUrl() == 'bk://sidl.bkbits.net/Compiler' and package == 'pythonGenerator'):
      # Also need pythonGenerator library
      sharedLinker.extraLibraries.append(self.getServerLibrary('pythonGenerator', proj = self.getInstalledProject('bk://sidl.bkbits.net/Compiler')))
    sharedLinker.extraLibraries.extend(self.extraLibraries)
    archiveFilter = build.transform.Filter(archiveTag)
    linker.addVertex(archiver)
    linker.addEdges(consolidator, [archiver])
    linker.addEdges(sharedLinker, [consolidator])
    linker.addEdges(archiveFilter, [sharedLinker])
    linker.addEdges(build.transform.Remover(compiler.output.tag), [archiveFilter])
    target.appendGraph(linker)
    return target

  def getClientCompileTarget(self):
    '''Python code does not need compilation, so only a C compiler is necessary for the cartilage.'''
    (target, compiler) = self.getGenericCompileTarget('client')
    sharedTag    = self.language.lower()+' client shared library'
    linker       = build.buildGraph.BuildGraph()
    sharedLinker = build.processor.SharedLinker(self.sourceDB, compiler.processor, compiler.output.tag, sharedTag)
    sharedLinker.extraLibraries.extend(self.extraLibraries)
    linker.addVertex(sharedLinker)
    linker.addEdges(build.transform.Remover(compiler.output.tag), [sharedLinker])
    target.appendGraph(linker)
    return target

  def installClient(self):
    '''Add Python paths for clients to the project'''
    return self.project.appendPythonPath(os.path.join(self.project.getRoot(), self.usingSIDL.getClientRootDir(self.language)))

  def installServer(self, package):
    '''Add Python paths for servers to the project'''
    return self.project.appendPythonPath(os.path.join(self.project.getRoot(), self.usingSIDL.getServerRootDir(self.language, package)))
