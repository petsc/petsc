from __future__ import generators
import base
import build.transform
import build.fileset

import os

class Processor(build.transform.Transform):
  '''Processor is the base class for source transformation, such as compilers, code generators, linkers, etc.
     - A FileSet with inputTag is transformed into a FileSet with outputTag.
     - Processed files are updated in the source database, either immediately, or put into a fileset tagged "update "<inputTag>
     - If isSetwise is true, FileSets are processed as a whole, otherwise individual files are processed
     - Files tagged "old "<inputTag> will be placed in "old "<outputTag>'''
  def __init__(self, sourceDB, processor, inputTag, outputTag, isSetwise, updateType):
    build.transform.Transform.__init__(self)
    self.sourceDB   = sourceDB
    self.processor  = processor
    self.inputTag   = inputTag
    if not isinstance(self.inputTag, list): self.inputTag = [self.inputTag]
    self.output.tag = outputTag
    self.isSetwise  = isSetwise
    self.updateType = updateType
    if self.updateType == 'deferred':
      self.deferredUpdates = build.fileset.FileSet(tag = 'update '+self.inputTag[0])
      self.output.children.append(self.deferredUpdates)
    self.oldOutput = build.fileset.FileSet(tag = 'old '+outputTag, mustExist = 0)
    self.output.children.append(self.oldOutput)
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
      self.debugPrint('\''+command+'\': '+output, 1, 'compile')

  def checkTag(self, f, tag):
    '''- If the tag matches the transform tag, return True
       - Otherwise return False'''
    if tag in self.inputTag:
      return 1
    return 0

  def checkOldTag(self, f, tag):
    '''- If the tag matches the "old "<transform tag>, return True
       - Otherwise return False'''
    if tag in map(lambda t: 'old '+t, self.inputTag):
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

  def processFile(self, f, set):
    return

  def processOldFile(self, f, set):
    self.oldOutput.append(f)
    return

  def processFileSet(self, set):
    for f in set:
      self.handleFile(f, set)
    return self.output

  def processOldFileSet(self, set):
    for f in set:
      self.processOldFile(f, set)
    return self.output

  def updateFile(self, source):
    '''Update the file in the source database
       - With deferred updates, a new FileSet is created with an update tag'''
    if self.updateType == 'immediate':
      self.sourceDB.updateSource(source)
    elif self.updateType == 'deferred':
      self.deferredUpdates.append(source)
    return

  def handleFile(self, f, set):
    '''Process and update the file if checkTag() returns True, otherwise call the default handler'''
    if self.checkTag(f, set.tag):
      self.processFile(f, set)
      self.updateFile(f)
      return self.output
    elif self.checkOldTag(f, set.tag):
      self.processOldFile(f, set)
      return self.output
    return build.transform.Transform.handleFile(self, f, set)

  def handleFileSet(self, set):
    '''Process and update the set if execution is setwise and checkTag() returns True, otherwise call the default handler'''
    if self.isSetwise:
      if self.checkTag(None, set.tag):
        self.processFileSet(set)
        map(self.updateFile, set)
        map(self.handleFileSet, set.children)
      elif self.checkOldTag(None, set.tag):
        self.processOldFileSet(set)
        map(self.handleFileSet, set.children)
      else:
        build.transform.Transform.handleFileSet(self, set)
    else:
      build.transform.Transform.handleFileSet(self, set)
    return self.output

class Compiler(Processor):
  '''A Compiler processes any FileSet with source, and outputs a FileSet of the intermediate object files.'''
  def __init__(self, sourceDB, compiler, inputTag, outputTag = None, isSetwise = 0, updateType = 'immediate'):
    if not isinstance(inputTag, list): inputTag = [inputTag]
    if outputTag is None:
      outputTag = inputTag[0]+' object'
    Processor.__init__(self, sourceDB, compiler, inputTag, outputTag, isSetwise, updateType)
    self.includeDirs = []
    self.defines     = []
    return

  def includeDirsIter(self):
    '''Return an iterator for the include directories'''
    for dir in self.includeDirs:
      try:
        dir = str(dir)
      except TypeError:
        for d in dir.getPath():
          yield d
      else:
        yield dir
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
    for dirname in self.includeDirsIter():
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
      if isinstance(define, tuple):
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
    '''Return a list of the compiler optimization flags. The default is empty.'''
    return []

  def getWarningFlags(self, source = None):
    '''Return a list of the compiler warning flags. The default is empty.'''
    return []

  def getFlags(self, source):
    return self.getOptimizationFlags(source)+self.getWarningFlags(source)+self.getDefineFlags(source)+self.getIncludeFlags(source)+self.getOutputFlags(source)

  def processFile(self, source, set):
    '''Compile "source"'''
    return self.processFileSet(build.fileset.FileSet([source], tag = set.tag))

  def processFileSet(self, set):
    '''Compile all the files in "set"'''
    objs = map(self.getIntermediateFileName, set)
    self.debugPrint('Compiling '+str(set)+' into '+str(objs), 3, 'compile')
    command = ' '.join([self.processor]+self.getFlags(set)+set)
    output  = self.executeShellCommand(command, self.handleErrors)
    self.output.extend(objs)
    return self.output

  def processOldFile(self, f, set):
    '''Put "old" object in for old source'''
    self.oldOutput.append(self.getIntermediateFileName(f))
    return self.output

class Linker(Processor):
  '''A Linker processes any FileSet with intermediate object files, and outputs a FileSet of libraries.'''
  def __init__(self, sourceDB, using, linker, inputTag, outputTag = None, isSetwise = 0, updateType = 'immediate', library = None, libExt = None):
    Processor.__init__(self, sourceDB, linker, inputTag, outputTag, isSetwise, updateType)
    self.using          = using
    self.library        = library
    self.libExt         = libExt
    self.extraLibraries = []
    return

  def __str__(self):
    return 'Linker('+self.processor+') for '+str(self.inputTag)

  def getProcessor(self):
    '''Return the processor executable'''
    if self._processor is None:
      return self.using.linker
    return self._processor
  processor = property(getProcessor, Processor.setProcessor, doc = 'This is the executable which will process files')

  def getLibExt(self):
    return self._libExt

  def setLibExt(self, ext):
    self._libExt = ext
  libExt = property(getLibExt, setLibExt, doc = 'The library extension')

  def extraLibrariesIter(self):
    '''Return an iterator for the extra libraries
       - Empty library names are possible, and they are ignored'''
    for lib in self.extraLibraries:
      try:
        lib = str(lib)
      except TypeError:
        for l in lib.getPath():
          if l: yield l
      else:
        if lib: yield lib
    return

  def getLibrary(self, object):
    '''Get the library for "object", and ensures that a FileSet all has the same library'''
    if isinstance(object, build.fileset.FileSet):
      library = dict(zip(map(self.getLibrary, object), range(len(object))))
      if len(library) > 1: raise RuntimeError('Invalid setwise link due to incompatible libraries: '+str(library.keys()))
      return library.keys()[0]
    if not self.library is None:
      (library, ext) = os.path.splitext(str(self.library))
    else:
      source      = self.getSourceFileName(object)
      (dir, file) = os.path.split(source)
      (base, ext) = os.path.splitext(file)
      # Handle Python
      if base[-7:] == '_Module':
        library = os.path.join(dir, base[:-7]+'module')
      else:
        library = os.path.join(dir, base)
    # Ensure the directory exists
    dir = os.path.dirname(library)
    if dir and not os.path.exists(dir):
      os.makedirs(dir)
    if self.libExt:
      return library+'.'+self.libExt
    return library

  def getOptimizationFlags(self, source):
    '''Return a list of the linker optimization flags.'''
    return []

  def getLinkerFlags(self, source):
    '''Return a list of the linker specific flags. The default is gives the extraLibraries as arguments.'''
    flags = [self.using.getLinkerFlags()]
    for lib in self.extraLibrariesIter():
      # Options and object files are passed verbatim
      if lib[0] == '-' or lib.endswith('.o'):
        flags.append(lib)
      # Big Intel F90 hack (the shared library is broken)
      elif lib.endswith('intrins.a'):
        flags.append(lib)
      else:
        (dir, file) = os.path.split(lib)
        (base, ext) = os.path.splitext(file)
        if not base.startswith('lib'):
          flags.append(lib)
        else:
          if dir:
            if 'C_LINKER_SLFLAG' in self.argDB:
              flags.extend(['-L'+dir, self.argDB['C_LINKER_SLFLAG']+dir])
            else:
              flags.extend(['-L'+dir])
          flags.append('-l'+base[3:])
    return flags

  def getOutputFlags(self, source):
    '''Return a list of the linker flags specifying the library'''
    return ['-o '+self.getLibrary(source)]

  def getFlags(self, source):
    return self.getOptimizationFlags(source)+self.getLinkerFlags(source)+self.getOutputFlags(source)

  def processFile(self, source, set):
    '''Link "source"'''
    # Leave this set unchanged
    build.transform.Transform.handleFile(self, source, set)
    return self.processFileSet(build.fileset.FileSet([source], tag = set.tag))

  def processFileSet(self, set):
    '''Link all the files in "set"'''
    if len(set) == 0: return self.output
    # Leave this set unchanged
    for f in set:
      build.transform.Transform.handleFile(self, f, set)
    library = self.getLibrary(set)
    self.debugPrint('Linking '+str(set)+' into '+library, 3, 'compile')
    command = ' '.join([self.processor]+set+self.getFlags(set))
    output  = self.executeShellCommand(command, self.handleErrors)
    self.output.append(library)
    return self.output

  def processOldFile(self, f, set):
    '''Output old library'''
    self.oldOutput.append(self.getLibrary(f))
    return self.output

class DirectoryArchiver(Linker):
  '''A DirectoryArchiver processes any FileSet with intermediate object files, and outputs a FileSet of those files moved to a storage directory.'''
  def __init__(self, sourceDB, using, archiver, inputTag, outputTag = None, isSetwise = 0, updateType = 'none', library = None, libExt = 'dir'):
    if not isinstance(inputTag, list): inputTag = [inputTag]
    if outputTag is None:
      outputTag = inputTag[0]+' library'
    Linker.__init__(self, sourceDB, using, archiver, inputTag, outputTag, isSetwise, updateType, library, libExt)
    return

  def __str__(self):
    return 'DirectoryArchiver('+self.processor+') for '+str(self.inputTag)

  def getOptimizationFlags(self, source):
    '''Return a list of the archiver optimization flags. The default is empty.'''
    return []

  def getLinkerFlags(self, source):
    '''Return a list of the archiver specific flags. The default is empty.'''
    return []

  def getOutputFlags(self, source):
    '''Return a list of the archiver flags specifying the archive'''
    return [self.getLibrary(source)]

  def processFileSet(self, set):
    '''Link all the files in "set"'''
    if len(set) == 0: return self.output
    # Leave this set unchanged
    for f in set:
      build.transform.Transform.handleFile(self, f, set)
    library = self.getLibrary(set)
    # Ensure the directory exists
    if not os.path.exists(library):
      os.makedirs(library)
    self.debugPrint('Linking '+str(set)+' into '+library, 3, 'compile')
    command = ' '.join([self.processor]+set+self.getFlags(set))
    output  = self.executeShellCommand(command, self.handleErrors)
    self.output.extend(map(lambda f: os.path.join(library, os.path.basename(f)), set))
    return self.output

  def processOldFile(self, f, set):
    '''Convert old objects'''
    self.oldOutput.append(os.path.join(self.getLibrary(f), os.path.basename(f)))
    return self.output

class Archiver(Linker):
  '''An Archiver processes any FileSet with intermediate object files, and outputs a FileSet of static libraries.'''
  def __init__(self, sourceDB, using, archiver, inputTag, outputTag = None, isSetwise = 0, updateType = 'immediate', library = None, libExt = 'a'):
    if not isinstance(inputTag, list): inputTag = [inputTag]
    if outputTag is None:
      outputTag = inputTag[0]+' library'
    Linker.__init__(self, sourceDB, using, archiver, inputTag, outputTag, isSetwise, updateType, library, libExt)
    return

  def __str__(self):
    return 'Archiver('+self.processor+') for '+str(self.inputTag)

  def getOptimizationFlags(self, source):
    '''Return a list of the archiver optimization flags. The default is empty.'''
    return []

  def getLinkerFlags(self, source):
    '''Return a list of the archiver specific flags. The default is crv.'''
    return ['crv']

  def getOutputFlags(self, source):
    '''Return a list of the archiver flags specifying the archive'''
    return [self.getLibrary(source)]

  def processOldFile(self, f, set):
    '''An Archiver produces no "old" filesets'''
    return self.output

class SharedLinker(Linker):
  '''A SharedLinker processes any FileSet of libraries, and outputs a FileSet of shared libraries
     - This linker now works correctly with Cygwin'''
  def __init__(self, sourceDB, using, linker, inputTag, outputTag = None, isSetwise = 0, updateType = 'none', library = None, libExt = None):
    if not isinstance(inputTag, list): inputTag = [inputTag]
    if outputTag is None:
      outputTag = inputTag[0]+' shared library'
    Linker.__init__(self, sourceDB, using, linker, inputTag, outputTag, isSetwise, updateType, library, libExt)
    return

  def getLibExt(self):
    if self._libExt is None:
      return 'so'
    return self._libExt
  libExt = property(getLibExt, Linker.setLibExt, doc = 'The library extension')

  def __str__(self):
    if self.argDB['HAVE_CYGWIN']:
      return 'Cygwin Shared linker('+self.processor+') for '+str(self.inputTag)
    return 'Shared linker('+self.processor+') for '+str(self.inputTag)

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
    '''Return a list of the linker optimization flags. The default is empty.'''
    return []

  def getLinkerFlags(self, source):
    '''Return a list of the linker specific flags. The default is the flags for shared linking plus the base class flags.'''
    flags = []
    if self.argDB['SHARED_LIBRARY_FLAG']:
      flags.append(self.argDB['SHARED_LIBRARY_FLAG'])
    flags.extend(Linker.getLinkerFlags(self, source))
    return flags

class ImportSharedLinker(SharedLinker):
  '''An ImportSharedLinker processes any FileSet of libraries, and outputs a FileSet of import libraries'''
  def __init__(self, sourceDB, using, linker, inputTag, outputTag = None, isSetwise = 0, updateType = 'none', library = None, libExt = None):
    SharedLinker.__init__(self, sourceDB, using, linker, inputTag, outputTag, isSetwise, updateType, library, libExt)
    self.imports     = build.fileset.FileSet()
    self.imports.tag = outputTag+' import'
    return

  def __str__(self):
    return 'Import shared linker('+self.processor+') for '+str(self.inputTag)

  def getLibExt(self):
    if self._libExt is None:
      return 'dll.a'
    return self._libExt
  libExt = property(getLibExt, Linker.setLibExt, doc = 'The library extension')

  def getOutputFlags(self, source):
    '''Return a list of the linker flags specifying the library'''
    implibname = self.getLibrary(source)
    # This is a really ugly hack.
    # Since the dllname is a symbol in the implib, we should be generating the implibname from the dllname,
    # not by hoping the dllname is the implibname without the .a
    dllname    = implibname[:-2]
    return ['-o '+dllname+' -Wl,--out-implib='+implibname]

  def handleErrors(self, command, status, output):
    '''Ignore errors when trying to link libraries
       - This is the only way to get correct C++ mangling'''
    return

  def processFileSet(self, set):
    '''Link all the files in "set"'''
    if self.argDB['HAVE_CYGWIN']:
      super(SharedLinker, self).processFileSet(set)
    else:
      # Leave this set unchanged
      for f in set:
        build.transform.Transform.handleFile(self, f, set)
    return self.output

  def processOldFile(self, f, set):
    '''Output old library'''
    if self.argDB['HAVE_CYGWIN']:
      super(SharedLinker, self).processOldFile(f, set)
    return self.output

class LibraryAdder (build.transform.Transform):
  '''A LibraryAdder adds every library matching inputTag to the extraLibraries member of linker'''
  def __init__(self, inputTag, linker, prepend = 0):
    build.transform.Transform.__init__(self)
    self.inputTag = inputTag
    if not isinstance(self.inputTag, list):
      self.inputTag = [self.inputTag]
    self.linker  = linker
    self.prepend = prepend
    return

  def __str__(self):
    return 'Adding libraries from '+str(self.inputTag)+' to '+str(self.linker)

  def handleFile(self, f, set):
    '''Put all libraries matching inputTag in linker.extraLibraries'''
    if self.inputTag is None or set.tag in self.inputTag:
      if self.prepend:
        self.linker.extraLibraries.insert(0, f)
      else:
        self.linker.extraLibraries.append(f)
    return build.transform.Transform.handleFile(self, f, set)
