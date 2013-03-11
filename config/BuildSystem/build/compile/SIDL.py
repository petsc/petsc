import build.fileset
import build.processor

import os

import cPickle

try:
  from hashlib import md5 as new_md5
except ImportError:
  from md5 import new as new_md5

class SIDLConstants:
  '''This class contains data about the SIDL language'''
  def getLanguages():
    '''Returns a list of all permissible SIDL target languages'''
    # This should be argDB['installedLanguages']
    return ['C', 'Cxx', 'C++', 'Python', 'F77', 'F90', 'Java', 'Mathematica', 'Matlab']
  getLanguages = staticmethod(getLanguages)

  def checkLanguage(language):
    '''Check for a valid SIDL target language, otherwise raise a ValueError'''
    if not language in SIDLConstants.getLanguages():
      raise ValueError('Invalid SIDL language: '+language)
  checkLanguage = staticmethod(checkLanguage)

class SIDLLanguageList (list):
  def __setitem__(self, key, value):
    SIDLConstants.checkLanguage(value)
    list.__setitem__(self, key, value)

class Compiler(build.processor.Processor):
  '''The SIDL compiler processes any FileSet with the tag "sidl", and outputs a FileSet of source code with the appropriate language tag.
     - Servers always compile a single SIDL file'''
  def __init__(self, sourceDB, language, outputDir, isServer, usingSIDL):
    SIDLConstants.checkLanguage(language)
    build.processor.Processor.__init__(self, sourceDB, None, ['sidl', 'old sidl'], language.lower(), not isServer, 'deferred')
    # Can't initialize processor in constructor since I have to wait for Base to set argDB
    self.processor = self.getCompilerDriver()
    self.language  = language
    self.outputDir = outputDir
    self.isServer  = isServer
    if isServer:
      self.action  = 'server'
    else:
      self.action  = 'client'
    self.usingSIDL = usingSIDL
    self.repositoryDirs = []
    self.outputTag = self.language.lower()+' '+self.action
    return

  def __str__(self):
    return 'SIDL Compiler for '+self.language+' '+self.action

  def handleErrors(self, command, status, output):
    if status or output.find('Error:') >= 0:
      raise RuntimeError('Could not execute \''+str(command)+'\':\n'+str(output))

  def getCompilerDriver(self):
    project = self.getInstalledProject('bk://sidl.bkbits.net/Compiler')
    if project is None:
      return 'scandal.py'
    return os.path.join(project.getRoot(), 'driver', 'python', 'scandal.py')

  def getCompilerModule(self, name = 'scandal'):
    import imp

    root = os.path.dirname(self.getCompilerDriver())
    if not root:
      raise ImportError('Project bk://sidl.bkbits.net/Compiler is not installed')
    (fp, pathname, description) = imp.find_module(name, [root])
    try:
      return imp.load_module(name, fp, pathname, description)
    finally:
      if fp: fp.close()

  def getActionFlags(self, source):
    '''Return a list of the compiler flags specifying the generation action.'''
    return ['-'+self.action+'='+self.language]

  def getDependenciesSIDL(self):
    '''Return all SIDL files found in project dependencies'''
    if not self.repositoryDirs: return []
    sources = []
    for dir in self.repositoryDirs:
      dir = os.path.join(dir, 'sidl')
      if not os.path.exists(dir):
        self.debugPrint('Invalid SIDL include directory: '+dir, 4, 'compile')
        continue
      for source in os.listdir(dir):
        if not os.path.splitext(source)[1] == '.sidl': continue
        source = os.path.join(dir, source)
        if not os.path.isfile(source): raise RuntimeError('Invalid SIDL include: '+source)
        sources.append(source)
    return sources

  def getIncludeFlags(self, source):
    return ['-includes=['+','.join(self.getDependenciesSIDL())+']']

  def getOutputFlags(self, source):
    '''Return a list of the compiler flags specifying the output directories'''
    if isinstance(source, build.fileset.FileSet): source = source[0]
    (package, ext) = os.path.splitext(os.path.basename(source))
    if not self.outputDir is None:
      if self.isServer:
        outputDir = os.path.join(self.outputDir, self.usingSIDL.getServerRootDir(self.language, package))
      else:
        outputDir = os.path.join(self.outputDir, self.usingSIDL.getClientRootDir(self.language))
      return ['-'+self.action+'Dirs={'+self.language+':'+outputDir+'}']
    return []

  def getFlags(self, source):
    return self.getActionFlags(source)+self.getIncludeFlags(source)+self.getOutputFlags(source)

  def processFileShell(self, source, set):
    '''Compile "source" using a shell command'''
    return self.processFileSetShell(build.fileset.FileSet([source], tag = set.tag))

  def processFileSetShell(self, set):
    '''Compile all the files in "set" using a shell command'''
    if not len(set) or set.tag.startswith('old'): return self.output
    self.debugPrint('Compiling '+str(set)+' into a '+self.language+' '+self.action, 3, 'compile')
    command = ' '.join([self.getProcessor()]+self.getFlags(set)+set)
    output  = self.executeShellCommand(command, self.handleErrors)
    #self.output.extend(map(self.getIntermediateFileName, set))
    return self.output

  def processFileModule(self, source, set):
    '''Compile "source" using a module directly'''
    return self.processFileSetModule(build.fileset.FileSet([source], tag = set.tag))

  def processFileSetModule(self, set):
    '''Compile all the files in "set" using a module directly'''
    if not len(set): return self.output
    import nargs
    import sourceDatabase

    # Check for cached output
    #   We could of course hash this big key again
    #   These keys could be local, but we can do that if they proliferate too much. It would mean
    #     that each project would have to compile the SIDL once
    flags    = self.getFlags(set)
    cacheKey = 'cacheKey'+''.join([sourceDatabase.SourceDB.getChecksum(f) for f in set]+[new_md5(''.join(flags)).hexdigest()])
    if set.tag.startswith('old') and cacheKey in self.argDB:
      self.debugPrint('Loading '+str(set)+' for a '+self.language+' '+self.action+' from argument database ('+cacheKey+')', 3, 'compile')
      outputFiles = cPickle.loads(self.argDB[cacheKey])
    else:
      # Save targets so that they do not interfere with Scandal
      target            = self.argDB.target
      self.argDB.target = []
      # Run compiler and reporter
      compiler = self.getCompilerModule().Scandal(flags+set)
      if not set.tag.startswith('old'):
        self.debugPrint('Compiling '+str(set)+' into a '+self.language+' '+self.action, 3, 'compile')
        self.debugPrint('  with flags '+str(flags), 4, 'compile')
        compiler.run()
      else:
        self.debugPrint('Reporting on '+str(set)+' for a '+self.language+' '+self.action, 3, 'compile')
        self.debugPrint('  with flags '+str(flags), 4, 'compile')
        compiler.report()
      outputFiles          = compiler.outputFiles
      self.argDB[cacheKey] = cPickle.dumps(outputFiles)
      # Restore targets and remove flags
      self.argDB.target = target
      for flag in flags:
        del self.argDB[nargs.Arg.parseArgument(flag)[0]]
    # Construct output
    tag = self.outputTag
    if self.isServer:
      (package, ext) = os.path.splitext(os.path.basename(set[0]))
      tag           += ' '+package
    self.output.children.append(build.fileset.RootedFileSet(self.usingSIDL.project.getUrl(), outputFiles, tag = tag))
    return self.output

  def processFile(self, source, set):
    '''Compile "source"'''
    return self.processFileModule(source, set)

  def processFileSet(self, set):
    '''Compile all the files in "set"'''
    return self.processFileSetModule(set)

  def processOldFile(self, source, set):
    '''Compile "source"'''
    return self.processFileModule(source, set)

  def processOldFileSet(self, set):
    '''Compile all the files in "set"'''
    return self.processFileSetModule(set)
