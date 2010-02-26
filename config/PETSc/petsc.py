#!/usr/bin/env python
'''
  This is the first try for a hierarchically configured module. The idea is to
add the configure objects from a previously executed framework into the current
framework. However, this necessitates a reorganization of the activities in the
module.

  We must now have three distinct phases: location, construction, and testing.
This is very similar to the current compiler checks. The construction phase is
optional, and only necessary when the package has not been previously configured.
The phases will necessarily interact, as an installtion must be located before
testing, however anothe should be located if the testing fails.

  We will give each installation a unique key, which is returned by the location
method. This will allow us to identify working installations, as well as those
that failed testing.

  There is a wierd role reversal that can happen. If we look for PETSc, but
cannot find it, it is reasonable to ask to have it automatically downloaded.
However, in this case, rather than using the configure objects from the existing
PETSc, we contribute objects to the PETSc which will be built.

'''
from __future__ import generators
import user
import config.base

import re
import os

class InvalidPETScError(RuntimeError):
  pass

class Configure(config.base.Configure):
  def __init__(self, framework):
    config.base.Configure.__init__(self, framework)
    self.headerPrefix = ''
    self.substPrefix  = ''
    self.location     = None
    self.trial        = {}
    self.working      = {}
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
    # Location options
    help.addArgument('PETSc', '-with-petsc-dir=<root dir>',        nargs.ArgDir(None, None, 'Specify the root directory of the PETSc installation'))
    help.addArgument('PETSc', '-with-petsc-arch=<arch>',           nargs.Arg(None, None, 'Specify PETSC_ARCH'))
    # Construction options
    help.addArgument('PETSc', '-download-petsc=<no,yes,ifneeded>', nargs.ArgFuzzyBool(None, 0, 'Install PETSc'))
    # Testing options
    help.addArgument('PETSc', '-with-petsc-shared=<bool>',         nargs.ArgBool(None, 1, 'Require that the PETSc library be shared'))
    return

  def setupPackageDependencies(self, framework):
    import sys

    petscConf = None
    for (name, (petscDir, petscArch)) in self.getLocations():
      petscPythonDir = os.path.join(petscDir, 'config')
      sys.path.append(petscPythonDir)
      confPath = os.path.join(petscDir, petscArch,'conf')
      petscConf = framework.loadFramework(confPath)
      if petscConf:
        self.logPrint('Loaded PETSc-AS configuration ('+name+') from '+confPath)
        self.location = (petscDir, petscArch)
        self.trial[self.location] = name
        break
      else:
        self.logPrint('PETSc-AS has no cached configuration in '+confPath)
        sys.path.reverse()
        sys.path.remove(petscPythonDir)
        sys.path.reverse()
    if not petscConf:
      self.downloadPETSc()
    framework.addPackageDependency(petscConf, confPath)
    return

  def setupDependencies(self, framework):
    config.base.Configure.setupDependencies(self, framework)
    self.languages  = framework.require('PETSc.utilities.languages', self)
    self.compilers  = framework.require('config.compilers', self)
    self.headers    = framework.require('config.headers', self)
    self.libraries  = framework.require('config.libraries', self)
    self.blaslapack = framework.require('PETSc.packages.BlasLapack', self)
    self.mpi        = framework.require('PETSc.packages.MPI', self)
    return

  def getPETScArch(self, petscDir):
    '''Return the allowable PETSc architectures for a given root'''
    if 'with-petsc-arch' in self.framework.argDB:
      yield self.framework.argDB['with-petsc-arch']
    elif 'PETSC_ARCH' in os.environ:
      yield os.environ['PETSC_ARCH']
    else:
      raise InvalidPETScError('Must set PETSC_ARCH or use --with-petsc-arch')
    return

  def getLocations(self):
    '''Return all allowable locations for PETSc'''
    if hasattr(self, '_configured'):
      key =(self.dir, self.arch)
      yield (self.working[key], key)
      raise InvalidPETScError('Configured PETSc is not usable')
    if self.framework.argDB['download-petsc'] == 1:
      yield self.downloadPETSc()
      raise InvalidPETScError('Downloaded PETSc is not usable')
    if 'with-petsc-dir' in self.framework.argDB:
      petscDir = self.framework.argDB['with-petsc-dir']
      for petscArch in self.getPETScArch(petscDir):
        yield ('User specified installation root', (petscDir, petscArch))
      raise InvalidPETScError('No working architecitures in '+str(petscDir))
    elif 'PETSC_DIR' in os.environ:
      petscDir = os.environ['PETSC_DIR']
      for petscArch in self.getPETScArch(petscDir):
        yield ('User specified installation root', (petscDir, petscArch))
      raise InvalidPETScError('No working architecitures in '+str(petscDir))
    else:
      for petscArch in self.getPETScArch(petscDir):
        yield ('Default compiler locations', ('', petscArch))
      petscDirRE = re.compile(r'(PETSC|pets)c(-.*)?')
      trialDirs = []
      for packageDir in self.framework.argDB['package-dirs']:
        if os.path.isdir(packageDir):
          for d in os.listdir(packageDir):
            if petscDirRE.match(d):
              trialDirs.append(('Package directory installation root', os.path.join(packageDir, d)))
      usrLocal = os.path.join('/usr', 'local')
      if os.path.isdir(os.path.join('/usr', 'local')):
        trialDirs.append(('Frequent user install location (/usr/local)', usrLocal))
        for d in os.listdir(usrLocal):
          if petscDirRE.match(d):
            trialDirs.append(('Frequent user install location (/usr/local/'+d+')', os.path.join(usrLocal, d)))
      if 'HOME' in os.environ and os.path.isdir(os.environ['HOME']):
        for d in os.listdir(os.environ['HOME']):
          if petscDirRE.match(d):
            trialDirs.append(('Frequent user install location (~/'+d+')', os.path.join(os.environ['HOME'], d)))
    return

  def downloadPETSc(self):
    if self.framework.argDB['download-petsc'] == 0:
      raise RuntimeError('No functioning PETSc located')
    # Download and build PETSc
    #   Use only the already configured objects from this run
    raise RuntimeError('Not implemented')

  def getDir(self):
    if self.location:
      return self.location[0]
    return None
  dir = property(getDir, doc = 'The PETSc root directory')

  def getArch(self):
    if self.location:
      return self.location[1]
    return None
  arch = property(getArch, doc = 'The PETSc architecture')

  def getFound(self):
    return self.location and self.location in self.working
  found = property(getFound, doc = 'Did we find a valid PETSc installation')

  def getName(self):
    if self.location and self.location in self.working:
      return self.working[self.location][0]
    return None
  name = property(getName, doc = 'The PETSc installation type')

  def getInclude(self, useTrial = 0):
    if self.location and self.location in self.working:
      return self.working[self.location][1]
    elif useTrial and self.location and self.location in self.trial:
      return self.trial[self.location][1]
    return None
  include = property(getInclude, doc = 'The PETSc include directories')

  def getLib(self, useTrial = 0):
    if self.location and self.location in self.working:
      return self.working[self.location][2]
    elif useTrial and self.location and self.location in self.trial:
      return self.trial[self.location][2]
    return None
  lib = property(getLib, doc = 'The PETSc libraries')

  def getVersion(self):
    if self.location and self.location in self.working:
      return self.working[self.location][3]
    return None
  version = property(getVersion, doc = 'The PETSc version')

  def getOtherIncludes(self):
    if not hasattr(self, '_otherIncludes'):
      includes = []
      includes.extend([self.headers.getIncludeArgument(inc) for inc in self.mpi.include])
      return ' '.join(includes)
    return self._otherIncludes
  def setOtherIncludes(self, otherIncludes):
    self._otherIncludes = otherIncludes
  otherIncludes = property(getOtherIncludes, setOtherIncludes, doc = 'Includes needed to compile PETSc')

  def getOtherLibs(self):
    if not hasattr(self, '_otherLibs'):
      libs = self.compilers.flibs[:]
      libs.extend(self.mpi.lib)
      libs.extend(self.blaslapack.lib)
      return libs
    return self._otherLibs
  def setOtherLibs(self, otherLibs):
    self._otherLibs = otherLibs
  otherLibs = property(getOtherLibs, setOtherLibs, doc = 'Libraries needed to link PETSc')

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
    oldLibs = self.compilers.LIBS
    self.libraries.pushLanguage(self.languages.clanguage)
    found   = (self.libraries.check(libraries, 'PetscInitializeNoArguments', otherLibs = self.otherLibs, prototype = 'int PetscInitializeNoArguments(void);', cxxMangle = not self.languages.cSupport) and
               self.libraries.check(libraries, 'VecDestroy', otherLibs = self.otherLibs, prototype = 'typedef struct _p_Vec *Vec;int VecDestroy(Vec);', call = 'VecDestroy((Vec) 0)', cxxMangle = not self.languages.cSupport) and
               self.libraries.check(libraries, 'MatDestroy', otherLibs = self.otherLibs, prototype = 'typedef struct _p_Mat *Mat;int MatDestroy(Mat);', call = 'MatDestroy((Mat) 0)', cxxMangle = not self.languages.cSupport) and
               self.libraries.check(libraries, 'DADestroy', otherLibs = self.otherLibs, prototype = 'typedef struct _p_DA *DA;int DADestroy(DA);', call = 'DADestroy((DA) 0)', cxxMangle = not self.languages.cSupport) and
               self.libraries.check(libraries, 'KSPDestroy', otherLibs = self.otherLibs, prototype = 'typedef struct _p_KSP *KSP;int KSPDestroy(KSP);', call = 'KSPDestroy((KSP) 0)', cxxMangle = not self.languages.cSupport) and
               self.libraries.check(libraries, 'SNESDestroy', otherLibs = self.otherLibs, prototype = 'typedef struct _p_SNES *SNES;int SNESDestroy(SNES);', call = 'SNESDestroy((SNES) 0)', cxxMangle = not self.languages.cSupport) and
               self.libraries.check(libraries, 'TSDestroy', otherLibs = self.otherLibs, prototype = 'typedef struct _p_TS *TS;int TSDestroy(TS);', call = 'TSDestroy((TS) 0)', cxxMangle = not self.languages.cSupport))
    self.libraries.popLanguage()
    self.compilers.LIBS = oldLibs
    return found

  def checkInclude(self, includeDir):
    '''Check that petscsys.h is present'''
    oldFlags = self.compilers.CPPFLAGS
    self.compilers.CPPFLAGS += ' '.join([self.headers.getIncludeArgument(inc) for inc in includeDir])
    if self.otherIncludes:
      self.compilers.CPPFLAGS += ' '+self.otherIncludes
    self.pushLanguage(self.languages.clanguage)
    found = self.checkPreprocess('#include <petscsys.h>\n')
    self.popLanguage()
    self.compilers.CPPFLAGS = oldFlags
    return found

  def checkPETScLink(self, includes, body, cleanup = 1, codeBegin = None, codeEnd = None, shared = None):
    '''Analogous to checkLink(), but the PETSc includes and libraries are automatically provided'''
    success  = 0
    oldFlags = self.compilers.CPPFLAGS
    self.compilers.CPPFLAGS += ' '.join([self.headers.getIncludeArgument(inc) for inc in self.getInclude(useTrial = 1)])
    if self.otherIncludes:
      self.compilers.CPPFLAGS += ' '+self.otherIncludes
    oldLibs  = self.compilers.LIBS
    self.compilers.LIBS = ' '.join([self.libraries.getLibArgument(lib) for lib in self.getLib(useTrial = 1)+self.otherLibs])+' '+self.compilers.LIBS
    if self.checkLink(includes, body, cleanup, codeBegin, codeEnd, shared):
      success = 1
    self.compilers.CPPFLAGS = oldFlags
    self.compilers.LIBS     = oldLibs
    return success

  def checkWorkingLink(self):
    '''Checking that we can link a PETSc executable'''
    self.pushLanguage(self.languages.clanguage)
    if not self.checkPETScLink('#include <petsclog.h>\n', 'PetscLogDouble time;\nPetscErrorCode ierr;\n\nierr = PetscGetTime(&time); CHKERRQ(ierr);\n'):
      self.logPrint('PETSc cannot link, which indicates a problem with the PETSc installation')
      return 0
    self.logPrint('PETSc can link with '+self.languages.clanguage)
    self.popLanguage()

    if hasattr(self.compilers, 'CXX') and self.languages.clanguage == 'C':
      self.pushLanguage('C++')
      self.sourceExtension = '.C'
      if not self.checkPETScLink('#define PETSC_USE_EXTERN_CXX\n#include <petscsys.h>\n', 'PetscLogDouble time;\nPetscErrorCode ierr;\n\nierr = PetscGetTime(&time); CHKERRQ(ierr);\n'):
        self.logPrint('PETSc cannot link C++ but can link C, which indicates a problem with the PETSc installation')
        self.popLanguage()
        return 0
      self.popLanguage()
      self.logPrint('PETSc can link with C++')
    
    if hasattr(self.compilers, 'FC'):
      self.pushLanguage('FC')
      self.sourceExtension = '.F'
      if not self.checkPETScLink('', '          integer ierr\n          real time\n          call PetscGetTime(time, ierr)\n'):
        self.logPrint('PETSc cannot link Fortran, but can link C, which indicates a problem with the PETSc installation\nRun with -with-fc=0 if you do not wish to use Fortran')
        self.popLanguage()
        return 0
      self.popLanguage()
      self.logPrint('PETSc can link with Fortran')
    return 1

  def checkSharedLibrary(self, libraries):
    '''Check that the libraries for PETSc are shared libraries'''
    if config.setCompilers.Configure.isDarwin():
      # on Apple if you list the MPI libraries again you will generate multiply defined errors 
      # since they are already copied into the PETSc dynamic library.
      self.setOtherLibs([])
    self.pushLanguage(self.languages.clanguage)
    isShared = self.libraries.checkShared('#include <petscsys.h>\n', 'PetscInitialize', 'PetscInitialized', 'PetscFinalize', checkLink = self.checkPETScLink, libraries = libraries, initArgs = '&argc, &argv, 0, 0', boolType = 'PetscTruth', executor = self.mpi.mpiexec)
    self.popLanguage()
    return isShared

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
    self.logPrint('Found PETSc version (%s,%s,%s) patch %s on %s' % (majorNum, minorNum, subminorNum, patchNum, self.date))
    return '%d.%d.%d' % (majorNum, minorNum, subminorNum)

  def includeGuesses(self, path = None):
    '''Return all include directories present in path or its ancestors'''
    if not path:
      yield []
    while path:
      dir = os.path.join(path, 'include')
      if os.path.isdir(dir):
        yield [dir, os.path.join(path, self.arch,'include')]
      if path == '/':
        return
      path = os.path.dirname(path)
    return

  def libraryGuesses(self, root = None):
    '''Return standard library name guesses for a given installation root'''
    libs = ['ts', 'snes', 'ksp', 'dm', 'mat', 'vec', '']
    if root:
      d = os.path.join(root, 'lib', self.arch)
      if not os.path.isdir(d):
        self.logPrint('', 3, 'petsc')
        return
      yield [os.path.join(d, 'libpetsc'+lib+'.a') for lib in libs]
    else:
      yield ['libpetsc'+lib+'.a' for lib in libs]
    return

  def configureLibrary(self):
    '''Find a working PETSc
       - Right now, C++ builds are required to use PETSC_USE_EXTERN_CXX'''
    for location, name in self.trial.items():
      self.framework.logPrintDivider()
      self.framework.logPrint('Checking for a functional PETSc in '+name+', location/origin '+str(location))
      lib     = None
      include = None
      found   = 0
      for libraries in self.libraryGuesses(location[0]):
        if self.checkLib(libraries):
          lib = libraries
          for includeDir in self.includeGuesses(location[0]):
            if self.checkInclude(includeDir):
              include = includeDir
              self.trial[location] = (name, include, lib, 'Unknown')
              if self.executeTest(self.checkWorkingLink):
                found = 1
                break
              else:
                self.framework.logPrintDivider(single = 1)
                self.framework.logPrint('PETSc in '+name+', location/origin '+str(location)+' failed checkWorkingLink test')
            else:
              self.framework.logPrintDivider(single = 1)
              self.framework.logPrint('PETSc in '+name+', location/origin '+str(location)+' failed checkInclude test with includeDir: '+str(includeDir))
          if not found:
            self.framework.logPrintDivider(single = 1)
            self.framework.logPrint('PETSc in '+name+', location/origin '+str(location)+' failed checkIncludes test')
            continue
        else:
          self.framework.logPrintDivider(single = 1)
          self.framework.logPrint('PETSc in '+name+', location/origin '+str(location)+' failed checkLib test with libraries: '+str(libraries))
          continue
        if self.framework.argDB['with-petsc-shared']:
          if not self.executeTest(self.checkSharedLibrary, [libraries]):
            self.framework.logPrintDivider(single = 1)
            self.framework.logPrint('PETSc in '+name+', location/origin '+str(location)+' failed checkSharedLibrary test with libraries: '+str(libraries))
            found = 0
        if found:
          break
      if found:
        version = self.executeTest(self.configureVersion)
        self.working[location] = (name, include, lib, version)
        break
    if found:
      self.logPrint('Choose PETSc '+self.version+' in '+self.name)
    else:
      raise RuntimeError('Could not locate any functional PETSc')
    return

  def setOutput(self):
    '''Add defines and substitutions
       - HAVE_PETSC is defined if a working PETSc is found
       - PETSC_INCLUDE and PETSC_LIB are command line arguments for the compile and link'''
    if self.found:
      self.addDefine('HAVE_PETSC', 1)
      self.addSubstitution('PETSC_INCLUDE', ' '.join([self.headers.getIncludeArgument(inc) for inc in self.include]))
      self.addSubstitution('PETSC_LIB', ' '.join(map(self.libraries.getLibArgument, self.lib)))
    return

  def configure(self):
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
