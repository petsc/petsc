import build.transform
import build.fileset

import os

class Processor(build.transform.Transform):
  '''Processor is the base class for source transformation, such as compilers, code generators, linkers, etc.
     - A FileSet with inputTag is transformed into a FileSet with outputTag.
     - Processed files are updated in the source database, either immediately, or put into a fileset tagged "update "<inputTag>
     - If isSetwise is true, FileSets are processed as a whole, otherwise individual files are processed'''
  def __init__(self, sourceDB, processor, inputTag, outputTag, isSetwise, updateType):
    build.transform.Transform.__init__(self)
    self.sourceDB   = sourceDB
    self.processor  = processor
    self.inputTag   = inputTag
    self.output.tag = outputTag
    self.isSetwise  = isSetwise
    self.updateType = updateType
    if self.updateType == 'deferred':
      self.deferredUpdates = build.fileset.FileSet(tag = 'update '+self.inputTag)
      self.output.children.append(self.deferredUpdates)
    return

  def setProcessor(self, processor):
    '''Set the processor executable'''
    self._processor = processor
    return

  def getProcessor(self):
    '''Return the processor executable'''
    return self._processor
  processor = property(getProcessor, setProcessor, doc = 'This is the executable which will process files')

  def handleErrors(self, command, status, output):
    if status:
      raise RuntimeError('Could not execute \''+command+'\':\n'+output)
    elif output.find('warning') >= 0:
      print('\''+command+'\': '+output)

  def checkTag(self, f, tag):
    '''- If the tag matches the transform tag, return True
       - Otherwise return False'''
    if tag == self.inputTag:
      return 1
    return 0

  def getIntermediateFileName(self, source, ext = '.o'):
    '''Get the name of the object file for "source"'''
    import tempfile
    (dir, file) = os.path.split(source)
    (base, dum) = os.path.splitext(file)
    return os.path.join(tempfile.tempdir, dir.replace('/', '+')+'+'+base+ext)

  def getSourceFileName(self, object):
    '''Get the name of the source file without extension for "object"'''
    (dir, file) = os.path.split(object)
    (base, dum) = os.path.splitext(file)
    return base.replace('+', '/')

  def processFile(self, f, tag):
    print 'Processing '+f
    return

  def processFileSet(self, set):
    for f in set:
      self.handleFile(f, set.tag)
    return self.output

  def updateFile(self, source):
    '''Update the file in the source database
       - With deferred updates, a new FileSet is created with an update tag'''
    if self.updateType == 'immediate':
      self.sourceDB.updateSource(source)
    elif self.updateType == 'deferred':
      self.deferredUpdates.append(source)
    return

  def handleFile(self, f, tag):
    '''Process and update the file if checkTag() returns True, otherwise call the default handler'''
    if not self.checkTag(f, tag):
      return build.transform.Transform.handleFile(self, f, tag)
    self.processFile(f, tag)
    self.updateFile(f)
    return self.output

  def handleFileSet(self, set):
    '''Process and update the set if execution is setwise and checkTag() returns True, otherwise call the default handler'''
    if self.isSetwise and self.checkTag(None, set.tag):
      self.processFileSet(set)
      map(self.updateFile, set)
      map(self.handleFileSet, set.children)
    else:
      build.transform.Transform.handleFileSet(self, set)
    return self.output

class Compiler(Processor):
  '''A Compiler processes any FileSet with source, and outputs a FileSet of the intermediate object files.'''
  def __init__(self, sourceDB, compiler, inputTag, outputTag = None, isSetwise = 0, updateType = 'immediate'):
    if outputTag is None:
      outputTag = inputTag+' object'
    Processor.__init__(self, sourceDB, compiler, inputTag, outputTag, isSetwise, updateType)
    self.includeDirs = []
    self.defines     = []
    return

  def checkIncludeDirectory(self, dirname):
    '''Check that the include directory exists
       - Arguments preceeded by dashes are ignored'''
    if not os.path.isdir(dirname):
      if not dirname or not dirname[0:2] == '-I':
        raise RuntimeError('Include directory '+dirname+' does not exist')

  def getIncludeFlags(self, source = None):
    '''Return a list of the compiler flags specifying include directories'''
    flags = []
    for dirname in self.includeDirs:
      try:
        self.checkIncludeDirectory(dirname)
        if dirname[0] == '-':
          flags.append(dirname)
        else:
          flags.append('-I'+dirname)
      except RuntimeError, e:
        self.debugPrint(str(e), 3, 'compile')
    return flags

  def getDefineFlags(self, source = None):
    '''Return a lsit of the compiler flags specifying defines'''
    flags = []
    for define in self.defines:
      if type(define) == types.TupleType:
        flags.append('-D'+define[0]+'='+define[1])
      else:
        flags.append('-D'+define)
    return flags

  def getOutputFlags(self, source):
    '''Return a list of the compiler flags specifying the intermediate output file'''
    if isinstance(source, build.fileset.FileSet): source = source[0]
    object = self.getIntermediateFileName(source)
    if object:
      return ['-c', '-o '+object]
    return []

  def getOptimizationFlags(self, source = None):
    '''Return a list of the compiler optimization flags. The default is -g.'''
    return ['-g']

  def getWarningFlags(self, source = None):
    '''Return a list of the compiler warning flags. The default is empty.'''
    return []

  def getFlags(self, source):
    return self.getOptimizationFlags(source)+self.getWarningFlags(source)+self.getDefineFlags(source)+self.getIncludeFlags(source)+self.getOutputFlags(source)

  def processFile(self, source, tag):
    '''Compile "source"'''
    return self.processFileSet(build.fileset.FileSet([source]))

  def processFileSet(self, set):
    '''Compile all the files in "set"'''
    objs = map(self.getIntermediateFileName, set)
    self.debugPrint('Compiling '+str(set)+' into '+str(objs), 3, 'compile')
    command = ' '.join([self.getProcessor()]+self.getFlags(set)+set)
    output  = self.executeShellCommand(command, self.handleErrors)
    self.output.extend(objs)
    return self.output

class Linker(Processor):
  '''A Linker processes any FileSet with intermediate object files, and outputs a FileSet of libraries.'''
  def __init__(self, sourceDB, archiver, inputTag, outputTag = None, isSetwise = 0, updateType = 'immediate', library = None, libExt = 'a'):
    Processor.__init__(self, sourceDB, archiver, inputTag, outputTag, isSetwise, updateType)
    self.library = library
    self.libExt  = libExt
    return

  def getLibrary(self, object):
    '''Get the library for "object", and ensures that a FileSet all has the same library'''
    if isinstance(object, build.fileset.FileSet):
      library = dict(zip(map(self.getLibrary, object), range(len(object))))
      if len(library) > 1: raise RuntimeError('Invalid setwise link due to incompatible libraries: '+str(library.keys()))
      return library.keys()[0]
    if not self.library is None:
      (library, ext) = os.path.splitext(self.library)
    else:
      source      = self.getSourceFileName(object)
      (dir, file) = os.path.split(source)
      (base, ext) = os.path.splitext(file)
      # Handle Python
      if base[-7:] == '_Module':
        library = os.path.join(dir, base[:-7]+'module')
      else:
        library = os.path.join(dir, base)
    return library+'.'+self.libExt

  def getOptimizationFlags(self, source):
    '''Return a list of the linker optimization flags.'''
    return []

  def getLinkerFlags(self, source):
    '''Return a list of the linker specific flags.'''
    return []

  def getOutputFlags(self, source):
    '''Return a list of the linker flags specifying the library'''
    return []

  def getFlags(self, source):
    return self.getOptimizationFlags(source)+self.getLinkerFlags(source)+self.getOutputFlags(source)

  def processFile(self, source, tag):
    '''Link "source"'''
    # Leave this set unchanged
    build.transform.Transform.handleFile(self, source, tag)
    return self.processFileSet(build.fileset.FileSet([source]))

  def processFileSet(self, set):
    '''Link all the files in "set"'''
    # Leave this set unchanged
    for f in set:
      build.transform.Transform.handleFile(self, f, set.tag)
    library = self.getLibrary(set)
    self.debugPrint('Linking '+str(set)+' into '+library, 3, 'compile')
    command = ' '.join([self.getProcessor()]+self.getFlags(set)+set)
    output  = self.executeShellCommand(command, self.handleErrors)
    self.output.append(library)
    return self.output

class Archiver(Linker):
  '''An Archiver processes any FileSet with intermediate object files, and outputs a FileSet of static libraries.'''
  def __init__(self, sourceDB, archiver, inputTag, outputTag = None, isSetwise = 0, updateType = 'immediate', library = None, libExt = 'a'):
    if outputTag is None:
      outputTag = inputTag+' library'
    Linker.__init__(self, sourceDB, archiver, inputTag, outputTag, isSetwise, updateType, library, libExt)
    return

  def __str__(self):
    return 'Archiver('+self.processor+') for '+self.inputTag

  def getOptimizationFlags(self, source):
    '''Return a list of the archiver optimization flags. The default is empty.'''
    return []

  def getLinkerFlags(self, source):
    '''Return a list of the archiver specific flags. The default is crv.'''
    return ['crv']

  def getOutputFlags(self, source):
    '''Return a list of the archiver flags specifying the archive'''
    return [self.getLibrary(source)]

class SharedLinker(Linker):
  '''A SharedLinker processes any FileSet oflibraries, and outputs a FileSet of shared libraries.'''
  def __init__(self, sourceDB, linker, inputTag, outputTag = None, isSetwise = 0, updateType = 'immediate', library = None, libExt = 'so'):
    if outputTag is None:
      outputTag = inputTag+' shared library'
    Linker.__init__(self, sourceDB, linker, inputTag, outputTag, isSetwise, updateType, library, libExt)
    return

  def __str__(self):
    return 'Shared linker('+self.processor+') for '+self.inputTag

  def checkSharedLibrary(self, source):
    '''Check that a shared library can be opened, otherwise throw a RuntimeException'''
    try:
      import BS.LinkCheckerI.Checker
      import BS.LinkError

      try:
        BS.LinkCheckerI.Checker.Checker().openLibrary(source)
      except BS.LinkError.Exception, e:
        raise RuntimeError(e.getMessage())
    except ImportError:
      self.debugPrint('Did not check shared library '+source, 3, 'link')

  def getOptimizationFlags(self, source):
    '''Return a list of the linker optimization flags. The default is -g.'''
    return ['-g']

  def getLinkerFlags(self, source):
    '''Return a list of the linker specific flags. The default is -shared.'''
    return ['-shared']

  def getOutputFlags(self, source):
    '''Return a list of the linker flags specifying the shared library'''
    return ['-o '+self.getLibrary(source)]
