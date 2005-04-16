#!/usr/bin/env python
from __future__ import generators
import user
import config.base

import re
import os

class Configure(config.base.Configure):
  def __init__(self, framework):
    config.base.Configure.__init__(self, framework)
    self.headerPrefix = ''
    self.substPrefix  = ''
    self.found        = 0
    self.foundLib     = 0
    self.foundInclude = 0
    self.dir          = None
    self.petsc        = None
    self.compilers    = self.framework.require('config.compilers', self)
    self.libraries    = self.framework.require('config.libraries', self)
    return

  def __str__(self):
    if self.found:
      desc = ['PETSc:']	
      desc.append('  Type: '+self.name)
      desc.append('  Version: '+self.version)
      desc.append('  Includes: '+str(self.include))
      desc.append('  Library: '+str(self.lib))
      return '\n'.join(desc)+'\n'
    else:
      return ''

  def setupHelp(self, help):
    import nargs
    help.addArgument('PETSc', '-with-petsc=<bool>',                nargs.ArgBool(None, 1, 'Activate PETSc'))
    help.addArgument('PETSc', '-with-petsc-dir=<root dir>',        nargs.ArgDir(None, None, 'Specify the root directory of the PETSc installation'))
    help.addArgument('PETSc', '-with-petsc-shared=<bool>',         nargs.ArgBool(None, 1, 'Require that the PETSc library be shared'))
    help.addArgument('PETSc', '-with-petsc-arch=<arch>',           nargs.Arg(None, None, 'Specify PETSC_ARCH'))
    help.addArgument('PETSc', '-download-petsc=<no,yes,ifneeded>', nargs.ArgFuzzyBool(None, 0, 'Install PETSc'))
    return

  def loadPETScConfigure(self):
    '''Load the configure module from PETSc-AS'''
    import RDict
    import sys
    confPath = os.path.join(self.dir, 'bmake', self.arch)
    oldDir = os.getcwd()
    os.chdir(confPath)
    argDB = RDict.RDict()
    os.chdir(oldDir)
    sys.path.append(os.path.join(self.dir, 'python'))
    framework = self.loadConfigure(argDB)
    if framework is None:
      raise RuntimeError('PETSc-AS has no cached configuration in '+confPath)
    else:
      self.logPrint('Loaded PETSc-AS configuration from '+confPath)
    return framework.require('PETSc.Configure', None)

  def getOtherIncludes(self):
    if not hasattr(self, '_otherIncludes'):
      includes = []
      if not self.petsc is None:
        includes.extend(['-I'+include for include in self.petsc.mpi.include])
      return ' '.join(includes)
    return self._otherIncludes
  def setOtherIncludes(self, otherIncludes):
    self._otherIncludes = otherIncludes
  otherIncludes = property(getOtherIncludes, setOtherIncludes, doc = 'Includes needed to compile PETSc')

  def getOtherLibs(self):
    if not hasattr(self, '_otherLibs'):
      libs = self.compilers.flibs
      if not self.petsc is None:
        libs.extend(self.petsc.mpi.lib)
      return libs
    return self._otherLibs
  def setOtherLibs(self, otherLibs):
    self._otherLibs = otherLibs
  otherLibs = property(getOtherLibs, setOtherLibs, doc = 'Libraries needed to link PETSc')

  def configureArchitecture(self):
    '''Determine the PETSc architecture'''
    if 'with-petsc-arch' in self.framework.argDB:
      self.arch = self.framework.argDB['with-petsc-arch']
    elif 'PETSC_ARCH' in os.environ:
      self.arch = os.environ['PETSC_ARCH']
    else:
      #HACK
      self.arch = 'linux-gnu'
    return

  def checkLib(self, libraries):
    '''Check for PETSc creation functions in libraries, which can be a list of libraries or a single library
       - PetscInitialize from libpetsc
       - VecCreate from libpetscvec
       - MatCreate from libpetscmat
       - DADestroy from libpetscdm
       - KSPCreate from libpetscksp
       - SNESCreate from libpetscsnes
       - TSCreate from libpetscts
       '''
    if not isinstance(libraries, list): libraries = [libraries]
    oldLibs = self.framework.argDB['LIBS']
    found   = (self.libraries.check(libraries, 'PetscInitialize', otherLibs = self.otherLibs) and
               self.libraries.check(libraries, 'VecCreate', otherLibs = self.otherLibs) and
               self.libraries.check(libraries, 'MatCreate', otherLibs = self.otherLibs) and
               self.libraries.check(libraries, 'DADestroy', otherLibs = self.otherLibs) and
               self.libraries.check(libraries, 'KSPCreate', otherLibs = self.otherLibs) and
               self.libraries.check(libraries, 'SNESCreate', otherLibs = self.otherLibs) and
               self.libraries.check(libraries, 'TSCreate', otherLibs = self.otherLibs))
    self.framework.argDB['LIBS'] = oldLibs
    return found

  def checkInclude(self, includeDir):
    '''Check that petsc.h is present'''
    oldFlags = self.compilers.CPPFLAGS
    self.compilers.CPPFLAGS += ' '.join([self.libraries.getIncludeArgument(inc) for inc in includeDir])
    if self.otherIncludes:
      self.compilers.CPPFLAGS += ' '+self.otherIncludes
    found = self.checkPreprocess('#include <petsc.h>\n')
    self.compilers.CPPFLAGS = oldFlags
    return found

  def checkPETScLink(self, includes, body, cleanup = 1, codeBegin = None, codeEnd = None, shared = None):
    '''Analogous to checkLink(), but the PETSc includes and libraries are automatically provided'''
    success  = 0
    oldFlags = self.compilers.CPPFLAGS
    self.compilers.CPPFLAGS += ' '.join([self.libraries.getIncludeArgument(inc) for inc in self.include])
    if self.otherIncludes:
      self.compilers.CPPFLAGS += ' '+self.otherIncludes
    oldLibs  = self.framework.argDB['LIBS']
    self.framework.argDB['LIBS'] = ' '.join([self.libraries.getLibArgument(lib) for lib in self.lib+self.otherLibs])+' '+self.framework.argDB['LIBS']
    if self.checkLink(includes, body, cleanup, codeBegin, codeEnd, shared):
      success = 1
    self.compilers.CPPFLAGS = oldFlags
    self.framework.argDB['LIBS']     = oldLibs
    return success

  def checkWorkingLink(self):
    '''Checking that we can link a PETSc executable'''
    if not self.checkPETScLink('#include <petsc.h>\n', 'PetscLogDouble time;\nPetscErrorCode ierr;\n\nierr = PetscGetTime(&time); CHKERRQ(ierr);\n'):
      self.framework.log.write('PETSc cannot link, which indicates a problem with the PETSc installation\n')
      return 0
    self.framework.log.write('PETSc can link with C\n')
      
    if 'CXX' in self.framework.argDB:
      self.pushLanguage('C++')
      self.sourceExtension = '.C'
      if not self.checkPETScLink('#define PETSC_USE_EXTERN_CXX\n#include <petsc.h>\n', 'PetscLogDouble time;\nPetscErrorCode ierr;\n\nierr = PetscGetTime(&time); CHKERRQ(ierr);\n'):
        self.framework.log.write('PETSc cannot link C++ but can link C, which indicates a problem with the PETSc installation\n')
        self.popLanguage()
        return 0
      self.popLanguage()
      self.framework.log.write('PETSc can link with C++\n')
    
    if 'FC' in self.framework.argDB:
      self.pushLanguage('FC')
      self.sourceExtension = '.F'
      # HACK (?)
      self.lib.insert(0, os.path.join(self.dir,'lib',self.arch, 'libpetscfortran.a'))
      if not self.checkPETScLink('', '          integer ierr\n          real time\n          call PetscGetTime(time, ierr)\n'):
        self.framework.log.write('PETSc cannot link Fortran, but can link C, which indicates a problem with the PETSc installation\nRun with -with-fc=0 if you do not wish to use Fortran')
        self.popLanguage()
        return 0
      self.popLanguage()
      self.framework.log.write('PETSc can link with Fortran\n')
    return 1

  def checkSharedLibrary(self):
    '''Check that the libraries for PETSc are shared libraries'''
    return self.libraries.checkShared('#include <petsc.h>\n', 'PetscInitialize', 'PetscInitialized', 'PetscFinalize', checkLink = self.checkPETScLink, libraries = self.lib, initArgs = '&argc, &argv, 0, 0', boolType = 'PetscTruth')

  def configureVersion(self):
    '''Determine the PETSc version'''
    majorRE    = re.compile(r'^#define PETSC_VERSION_MAJOR([\s]+)(?P<versionNum>\d+)[\s]*$');
    minorRE    = re.compile(r'^#define PETSC_VERSION_MINOR([\s]+)(?P<versionNum>\d+)[\s]*$');
    subminorRE = re.compile(r'^#define PETSC_VERSION_SUBMINOR([\s]+)(?P<versionNum>\d+)[\s]*$');
    patchRE    = re.compile(r'^#define PETSC_VERSION_PATCH([\s]+)(?P<patchNum>\d+)[\s]*$');
    dateRE     = re.compile(r'^#define PETSC_VERSION_DATE([\s]+)"(?P<date>(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) \d\d?, \d\d\d\d)"[\s]*$');
    input   = file(os.path.join(self.dir, 'include', 'petscversion.h'))
    lines   = []
    majorNum = 'Unknown'
    minorNum = 'Unknown'
    subminorNum = 'Unknown'
    patchNum = 'Unknown'
    self.date = 'Unknown'
    for line in input.readlines():
      m1 = majorRE.match(line)
      m2 = minorRE.match(line)
      m3 = subminorRE.match(line)
      m4 = patchRE.match(line)
      m5 = dateRE.match(line)
      if m1:
        majorNum = int(m1.group('versionNum'))
      elif m2:
        minorNum = int(m2.group('versionNum'))
      elif m3:
        subminorNum = int(m3.group('versionNum'))

      if m4:
        patchNum = int(m4.group('patchNum'))+1
        lines.append('#define PETSC_VERSION_PATCH'+m4.group(1)+str(patchNum)+'\n')
      elif m5:
        self.date = time.strftime('%b %d, %Y', time.localtime(time.time()))
        lines.append('#define PETSC_VERSION_DATE'+m5.group(1)+'"'+self.date+'"\n')
      else:
        lines.append(line)
    input.close()
    # Update the version and patchNum in argDB
    self.logPrint('Found PETSc version (%s,%s,%s) patch %s on %s' % (majorNum, minorNum, subminorNum, patchNum, self.date))
    return '%d.%d.%d' % (majorNum, minorNum, subminorNum)

  def includeGuesses(self, path):
    '''Return all include directories present in path or its ancestors'''
    while path:
      dir = os.path.join(path, 'include')
      if os.path.isdir(dir):
        yield [dir, os.path.join(path, 'bmake', self.arch)]
      if path == '/':
        return
      path = os.path.dirname(path)
    return

  def libraryGuesses(self, root = None):
    '''Return standard library name guesses for a given installation root'''
    libs = ['ts', 'snes', 'ksp', 'dm', 'mat', 'vec', '']
    if root:
      dir = os.path.join(root, 'lib', self.arch)
      if not os.path.isdir(dir):
        self.logPrint('', 3, 'petsc')
        return
      yield [os.path.join(dir, 'libpetsc'+lib+'.a') for lib in libs]
    else:
      yield ['libpetsc'+lib+'.a' for lib in libs]
    return

  def generateGuesses(self):
    if self.framework.argDB['download-petsc'] == 1:
      (name, lib, include) = self.downloadPETSc()
      yield (name, lib, include,'\'downloaded\'')
      # Since the generator has been reinvoked, the request to download PETSc did not produce a valid version of PETSc.
      # We assume that the user insists upon using this version of PETSc, hence there is no legal way to proceed.
      raise RuntimeError('Downloaded PETSc could not be used.\n')
    # Try specified installation root
    dirs = []
    if 'PETSC_DIR' in os.environ:
      dirs.append(os.environ['PETSC_DIR'])
    if 'with-petsc-dir' in self.framework.argDB:
      dirs.append(self.framework.argDB['with-petsc-dir'])
    for dir in dirs:
      if dir is None:
        continue
      if not (len(dir) > 2 and dir[1] == ':'):
        dir = os.path.abspath(dir)
      self.dir = dir
      yield ('User specified installation root', self.libraryGuesses(dir), self.includeGuesses(dir),dir)
      # Since the generator has been reinvoked, the user specified installation root did not contain a valid PETSc.
      # We assume that the user insists upon using this version of PETSc, hence there is no legal way to proceed.
      raise RuntimeError('You set a value for the PETSc directory, but '+dir+' cannot be used.\n It could be the PETSc located is not working for all the languages, you can try running\n configure again with --with-fc=0 or --with-cxx=0\n')
    # May not need to list anything
    yield ('Default compiler locations', self.libraryGuesses(), [[]],'\'default\'')
    # Try configure package directories
    dirExp = re.compile(r'(PETSC|pets)c(-.*)?')
    for packageDir in self.framework.argDB['package-dirs']:
      packageDir = os.path.abspath(packageDir)
      if not os.path.isdir(packageDir):
        raise RuntimeError('Invalid package directory: '+packageDir)
      for f in os.listdir(packageDir):
        dir = os.path.join(packageDir, f)
        if not os.path.isdir(dir):
          continue
        if not dirExp.match(f):
          continue
        self.dir = dir
        yield ('Package directory installation root', self.libraryGuesses(dir), self.includeGuesses(dir),dir)
    # Try /usr/local
    dir = os.path.abspath(os.path.join('/usr', 'local'))
    self.dir = dir
    yield ('Frequent user install location (/usr/local)', self.libraryGuesses(dir), self.includeGuesses(dir),dir)
    # Try /usr/local/*petsc*
    ls = os.listdir(os.path.join('/usr','local'))
    for dir in ls:
      if dir.find('petsc') >= 0:
        dir = os.path.join('/usr','local',dir)
        if os.path.isdir(dir):
          self.dir = dir
          yield ('Frequent user install location (/usr/local/*petsc*)', self.libraryGuesses(dir), self.includeGuesses(dir),dir)
    # Try ~/petsc*
    ls = os.listdir(os.getenv('HOME'))
    for dir in ls:
      if dir.find('petsc') >= 0:
        dir = os.path.join(os.getenv('HOME'),dir)
        if os.path.isdir(dir):
          self.dir = dir
          yield ('Frequent user install location (~/*petsc*)', self.libraryGuesses(dir), self.includeGuesses(dir),dir)
    # If necessary, download PETSc
    if not self.found and self.framework.argDB['download-petsc'] == 2:
      (name, lib, include) = self.downloadPETSc()
      yield (name, lib, include,'\'downloaded\'')
      #raise RuntimeError('Downloaded PETSc could not be used. Please check in install in '+os.path.dirname(include)+'\n')
    return

  def getDir(self):
    '''Find the directory containing PETSc'''
    packages  = os.path.join(self.getRoot(), 'packages')
    if not os.path.isdir(packages):
      os.mkdir(packages)
      self.framework.actions.addArgument('PETSc', 'Directory creation', 'Created the packages directory: '+packages)
    petscDir = None
    for dir in os.listdir(packages):
      if dir.startswith('petsc') and os.path.isdir(os.path.join(packages, dir)):
        petscDir = dir
    if petscDir is None:
      self.framework.log.write('Could not locate already downloaded PETSc\n')
      raise RuntimeError('Error locating PETSc directory')
    return os.path.join(packages, petscDir)

  def configureLibrary(self):
    '''Find all working PETSc libraries and then choose one
       - Right now, C++ builds are required to use PETSC_USE_EXTERN_CXX'''
    functionalPETSc = []
    nonsharedPETSc  = []

    for (name, libraryGuesses, includeGuesses,location) in self.generateGuesses():
      self.framework.log.write('================================================================================\n')
      self.framework.log.write('Checking for a functional PETSc in '+name+', location/origin '+location+'\n')
      self.lib     = None
      self.include = None
      found        = 0
      for libraries in libraryGuesses:
        if self.checkLib(libraries):
          self.lib = libraries
          self.petsc = self.loadPETScConfigure()
          for includeDir in includeGuesses:
            if self.checkInclude(includeDir):
              self.include = includeDir
              if self.executeTest(self.checkWorkingLink):
                found = 1
                break
              else:
                self.framework.log.write('--------------------------------------------------------------------------------\n')
                self.framework.log.write('PETSc in '+name+', location/origin  '+location+' failed checkWorkingLink test\n')
            else:
              self.framework.log.write('--------------------------------------------------------------------------------\n')
              self.framework.log.write('PETSc in '+name+', location/origin '+location+' failed checkInclude test with includeDir: '+includeDir+'\n')

          if found:
            break
          self.petsc = None
        else:
          self.framework.log.write('--------------------------------------------------------------------------------\n')
          self.framework.log.write('PETSc in '+name+', location/origin '+location+' failed checkLib test with libraries: '+str(libraries)+'\n')
      if not found: continue

      version = self.executeTest(self.configureVersion)
      if self.framework.argDB['with-petsc-shared']:
        if not self.executeTest(self.checkSharedLibrary):
          nonsharedPETSc.append((name, self.lib, self.include, version))
          self.framework.log.write('--------------------------------------------------------------------------------\n')
          self.framework.log.write('PETSc in '+name+', location/origin '+location+' failed checkSharedLibrary test with libraries: '+str(libraries)+'\n')
          continue
      self.found = 1
      functionalPETSc.append((name, self.lib, self.include, version))
      if not self.framework.argDB['with-alternatives']:
        break
    # User chooses one or take first (sort by version)
    if self.found:
      self.name, self.lib, self.include, self.version = functionalPETSc[0]
      self.framework.log.write('Choose PETSc '+self.version+' in '+self.name+'\n')
    elif len(nonsharedPETSc):
      raise RuntimeError('Could not locate any PETSc with shared libraries')
    else:
      raise RuntimeError('Could not locate any functional PETSc')
    return

  def setOutput(self):
    '''Add defines and substitutions
       - HAVE_PETSC is defined if a working PETSc is found
       - PETSC_INCLUDE and PETSC_LIB are command line arguments for the compile and link
       - PETSC_INCLUDE_DIR is the directory containing petsc.h
       - PETSC_LIBRARY is the list of PETSc libraries'''
    if self.found:
      self.addDefine('HAVE_PETSC', 1)
      if self.include:
        self.addSubstitution('PETSC_INCLUDE',     ' '.join(['-I'+inc for inc in self.include]))
        self.addSubstitution('PETSC_INCLUDE_DIR', self.include[0])
      else:
        self.addSubstitution('PETSC_INCLUDE',     '')
        self.addSubstitution('PETSC_INCLUDE_DIR', '')
      if self.lib:
        self.addSubstitution('PETSC_LIB',     ' '.join(map(self.libraries.getLibArgument, self.lib)))
        self.addSubstitution('PETSC_LIBRARY', self.lib)
      else:
        self.addSubstitution('PETSC_LIB',     '')
        self.addSubstitution('PETSC_LIBRARY', '')
    return

  def configure(self):
    self.executeTest(self.configureArchitecture)
    self.executeTest(self.configureLibrary)
    self.setOutput()
    return

if __name__ == '__main__':
  import config.framework
  import sys
  framework = config.framework.Framework(sys.argv[1:])
  framework.setup()
  framework.addChild(Configure(framework))
  framework.configure()
  framework.dumpSubstitutions()
