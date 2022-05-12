import config.base

import re
import os
import shutil

def remove_xcode_verbose(buf):
  retbuf =[]
  for line in buf.splitlines():
    if not line.startswith('ld: warning: text-based stub file'): retbuf.append(line)
  return ('\n').join(retbuf)

class MissingProcessor(AttributeError):
  pass

class Configure(config.base.Configure):
  def __init__(self, framework):
    config.base.Configure.__init__(self, framework)
    self.headerPrefix = ''
    self.substPrefix  = ''
    return

  def setupHelp(self, help):
    import nargs
    help.addArgument('Compilers', '-with-fortran-type-initialize=<bool>',   nargs.ArgBool(None, 1, 'Initialize PETSc objects in Fortran'))
    return

  def setupDependencies(self, framework):
    config.base.Configure.setupDependencies(self, framework)
    self.setCompilers = framework.require('config.setCompilers', self)
    self.compilerFlags = framework.require('config.compilerFlags', self)
    self.libraries = framework.require('config.libraries', None)
    self.compilers = framework.require('config.compilers', self)
    return

  def __getattr__(self, name):
    if 'dispatchNames' in self.__dict__:
      if name in self.dispatchNames:
        if not hasattr(self.setCompilers, name):
          raise MissingProcessor(self.dispatchNames[name])
        return getattr(self.setCompilers, name)
      if name in ['CC_LINKER_FLAGS', 'FC_LINKER_FLAGS', 'CXX_LINKER_FLAGS', 'CUDAC_LINKER_FLAGS', 'HIPC_LINKER_FLAGS', 'SYCLC_LINKER_FLAGS', 'sharedLibraryFlags', 'dynamicLibraryFlags']:
        flags = getattr(self.setCompilers, name)
        if not isinstance(flags, list): flags = [flags]
        return ' '.join(flags)
    raise AttributeError('Configure attribute not found: '+name)

  def __setattr__(self, name, value):
    if 'dispatchNames' in self.__dict__:
      if name in self.dispatchNames:
        return setattr(self.setCompilers, name, value)
    config.base.Configure.__setattr__(self, name, value)
    return


  def checkFortranTypeSizes(self):
    '''Check whether real*8 is supported and suggest flags which will allow support'''
    self.pushLanguage('FC')
    # Check whether the compiler (ifc) bitches about real*8, if so try using -w90 -w to eliminate bitch
    (output, error, returnCode) = self.outputCompile('', '      real*8 variable', 1)
    if (output+error).find('Type size specifiers are an extension to standard Fortran 95') >= 0:
      oldFlags = self.setCompilers.FFLAGS
      self.setCompilers.FFLAGS += ' -w90 -w'
      (output, error, returnCode) = self.outputCompile('', '      real*8 variable', 1)
      if returnCode or (output+error).find('Type size specifiers are an extension to standard Fortran 95') >= 0:
        self.setCompilers.FFLAGS = oldFlags
      else:
        self.logPrint('Looks like ifc compiler, adding -w90 -w flags to avoid warnings about real*8 etc', 4, 'compilers')
    self.popLanguage()
    return


  def checkFortranPreprocessor(self):
    '''Determine if Fortran handles preprocessing properly'''
    self.setCompilers.pushLanguage('FC')
    # Does Fortran compiler need special flag for using CPP
    for flag in ['', '-cpp', '-xpp=cpp', '-F', '-Cpp', '-fpp', '-fpp:-m']:
      try:
        flagsArg = self.setCompilers.getCompilerFlagsArg()
        oldFlags = getattr(self.setCompilers, flagsArg)
        self.setCompilers.saveLog()
        self.setCompilers.addCompilerFlag(flag, body = '#define dummy \n           dummy\n#ifndef dummy\n       fooey\n#endif')
        self.logWrite(self.setCompilers.restoreLog())
        setattr(self.setCompilers, flagsArg, oldFlags+' '+flag)
        self.fortranPreprocess = 1
        self.setCompilers.popLanguage()
        self.logPrint('Fortran uses '+flag+' preprocessor', 3, 'compilers')
        return
      except RuntimeError:
        setattr(self.setCompilers, flagsArg, oldFlags)
    self.setCompilers.popLanguage()
    self.fortranPreprocess = 0
    self.logPrint('Fortran does NOT use preprocessor', 3, 'compilers')
    return

  def checkFortranDefineCompilerOption(self):
    '''Check if -WF,-Dfoobar or -Dfoobar is the compiler option to define a macro'''
    self.FortranDefineCompilerOption = ''
    if not self.fortranPreprocess:
      return
    self.setCompilers.saveLog()
    self.setCompilers.pushLanguage('FC')
    for flag in ['-D', '-WF,-D']:
      if self.setCompilers.checkCompilerFlag(flag+'Testing', body = '#define dummy \n           dummy\n#ifndef Testing\n       fooey\n#endif'):
        self.logWrite(self.setCompilers.restoreLog())
        self.FortranDefineCompilerOption = flag
        self.framework.addMakeMacro('FC_DEFINE_FLAG',self.FortranDefineCompilerOption)
        self.setCompilers.popLanguage()
        self.logPrint('Fortran uses '+flag+' for defining macro', 3, 'compilers')
        return
    self.logWrite(self.setCompilers.restoreLog())
    self.setCompilers.popLanguage()
    self.logPrint('Fortran does not support defining macro', 3, 'compilers')
    return

  def configureFortranFlush(self):
    self.pushLanguage('FC')
    for baseName in ['flush','flush_']:
      if self.checkLink(body='      call '+baseName+'(6)'):
        self.addDefine('HAVE_FORTRAN_'+baseName.upper(), 1)
        break
    self.popLanguage()
    return

  def checkFortranTypeInitialize(self):
    '''Determines if PETSc objects in Fortran are initialized by default (doesn't work with common blocks)'''
    if self.argDB['with-fortran-type-initialize']:
      self.addDefine('FORTRAN_TYPE_INITIALIZE', ' = -2') # If change -2, please also update PETSC_FORTRAN_OBJECT_F_DESTROYED_TO_C_NULL() etc.
      self.logPrint('Initializing Fortran objects')
    else:
      self.addDefine('FORTRAN_TYPE_INITIALIZE', ' ')
      self.logPrint('Not initializing Fortran objects')
    return

  def checkFortranTypeStar(self):
    '''Determine whether the Fortran compiler handles type(*)'''
    self.pushLanguage('FC')
    if self.checkCompile(body = '      interface\n      subroutine a(b)\n      type(*) :: b(:)\n      end subroutine\n      end interface\n'):
      self.addDefine('HAVE_FORTRAN_TYPE_STAR', 1)
      self.logPrint('Fortran compiler supports type(*)')
    else:
      self.logPrint('Fortran compiler does not support type(*)')
    self.popLanguage()
    return

  def checkFortran90(self):
    '''Determine whether the Fortran compiler handles F90'''
    self.pushLanguage('FC')
    if self.checkLink(body = '''
        REAL(KIND=SELECTED_REAL_KIND(10)) d
        INTEGER, PARAMETER :: int = SELECTED_INT_KIND(8)
        INTEGER (KIND=int) :: ierr
        ierr = 1'''):
      self.fortranIsF90 = 1
      self.logPrint('Fortran compiler supports F90')
    else:
      self.fortranIsF90 = 0
      self.logPrint('Fortran compiler does not support F90')
    self.popLanguage()
    return

  def checkFortran90LineLength(self):
    '''Determine whether the Fortran compiler has infinite line length'''
    self.pushLanguage('FC')
    if self.checkLink(body = '      INTEGER, PARAMETER ::        int = SELECTED_INT_KIND(8);              INTEGER (KIND=int) :: ierr,ierr2;       ierr                            =                                                                                                               1; ierr2 =                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      2'):
      self.addDefine('HAVE_FORTRAN_FREE_LINE_LENGTH_NONE', 1)
      self.logPrint('Fortran compiler has unlimited line length')
    else:
      self.logPrint('Fortran compiler does not have unlimited line length')
    self.popLanguage()
    return

  def checkFortran90FreeForm(self):
    '''Determine whether the Fortran compiler handles F90FreeForm
       We also require that the compiler handles lines longer than 132 characters'''
    self.pushLanguage('FC')
    if self.checkLink(body = '      INTEGER, PARAMETER ::        int = SELECTED_INT_KIND(8);              INTEGER (KIND=int) :: ierr;       ierr                            =                                                                                                               1'):
      self.addDefine('USING_F90FREEFORM', 1)
      self.fortranIsF90FreeForm = 1
      self.logPrint('Fortran compiler supports F90FreeForm')
    else:
      self.fortranIsF90FreeForm = 0
      self.logPrint('Fortran compiler does not support F90FreeForm')
    self.popLanguage()
    return

  def checkFortran2003(self):
    '''Determine whether the Fortran compiler handles F2003'''
    self.pushLanguage('FC')
    if self.fortranIsF90 and self.checkLink(codeBegin = '''
      module Base_module
        type, public :: base_type
           integer :: A
         contains
           procedure, public :: Print => BasePrint
        end type base_type
      contains
        subroutine BasePrint(this)
          class(base_type) :: this
        end subroutine BasePrint
      end module Base_module

      program main''',body = '''
      use,intrinsic :: iso_c_binding
      Type(C_Ptr),Dimension(:),Pointer :: CArray
      character(kind=c_char),pointer   :: nullc => null()
      character(kind=c_char,len=5),dimension(:),pointer::list1

      allocate(list1(5))
      CArray = (/(c_loc(list1(i)),i=1,5),c_loc(nullc)/)'''):
      self.addDefine('USING_F2003', 1)
      self.fortranIsF2003 = 1
      self.logPrint('Fortran compiler supports F2003')
    else:
      self.fortranIsF2003 = 0
      self.logPrint('Fortran compiler does not support F2003')
    self.popLanguage()
    for f in [os.path.abspath('base_module.mod'), os.path.abspath('BASE_MODULE.mod'), os.path.join(os.path.dirname(self.compilerObj),'base_module.mod'), os.path.join(os.path.dirname(self.compilerObj),'BASE_MODULE.mod')]:
      if os.path.isfile(f): os.remove(f)
    return

  def checkFortran90Array(self):
    '''Check for F90 array interfaces'''
    if not self.fortranIsF90:
      self.logPrint('Not a Fortran90 compiler - hence skipping f90-array test')
      return
    # do an apporximate test when batch mode is used, as we cannot run the proper test..
    if self.argDB['with-batch']:
      if config.setCompilers.Configure.isPGI(self.setCompilers.FC, self.log):
        self.addDefine('HAVE_F90_2PTR_ARG', 1)
        self.logPrint('PGI F90 compiler detected & using --with-batch, so use two arguments for array pointers', 3, 'compilers')
      else:
        self.logPrint('Using --with-batch, so guess that F90 uses a single argument for array pointers', 3, 'compilers')
      return
    # do not check on windows - as it pops up the annoying debugger
    if config.setCompilers.Configure.isCygwin(self.log):
      self.logPrint('Cygwin detected: ignoring HAVE_F90_2PTR_ARG test')
      return

    # Compile the C test object
    cinc  = '#include<stdio.h>\n#include <stdlib.h>\n'
    ccode = 'void '+self.compilers.mangleFortranFunction('f90arraytest')+'''(void* a1, void* a2,void* a3, void* i)
{
  printf("arrays [%p %p %p]\\n",a1,a2,a3);
  fflush(stdout);
  return;
}
''' + 'void '+self.compilers.mangleFortranFunction('f90ptrtest')+'''(void* a1, void* a2,void* a3, void* i, void* p1 ,void* p2, void* p3)
{
  printf("arrays [%p %p %p]\\n",a1,a2,a3);
  if ((p1 == p3) && (p1 != p2)) {
    printf("pointers match! [%p %p] [%p]\\n",p1,p3,p2);
    fflush(stdout);
  } else {
    printf("pointers do not match! [%p %p] [%p]\\n",p1,p3,p2);
    fflush(stdout);
    exit(111);
  }
  return;
}\n'''
    cobj = os.path.join(self.tmpDir, 'fooobj.o')
    self.pushLanguage('C')
    if not self.checkCompile(cinc+ccode, None, cleanup = 0):
      self.logPrint('Cannot compile C function: f90ptrtest', 3, 'compilers')
      raise RuntimeError('Could not check Fortran pointer arguments')
    if not os.path.isfile(self.compilerObj):
      self.logPrint('Cannot locate object file: '+os.path.abspath(self.compilerObj), 3, 'compilers')
      raise RuntimeError('Could not check Fortran pointer arguments')
    os.rename(self.compilerObj, cobj)
    self.popLanguage()
    # Link the test object against a Fortran driver
    self.pushLanguage('FC')
    oldLIBS = self.setCompilers.LIBS
    self.setCompilers.LIBS = cobj+' '+self.setCompilers.LIBS
    fcode = '''\
      Interface
         Subroutine f90ptrtest(p1,p2,p3,i)
         integer, pointer :: p1(:,:)
         integer, pointer :: p2(:,:)
         integer, pointer :: p3(:,:)
         integer i
         End Subroutine
      End Interface

      integer, pointer :: ptr1(:,:),ptr2(:,:)
      integer, target  :: array(6:8,9:21)
      integer  in

      in   = 25
      ptr1 => array
      ptr2 => array

      call f90arraytest(ptr1,ptr2,ptr1,in)
      call f90ptrtest(ptr1,ptr2,ptr1,in)\n'''

    found = self.checkRun(None, fcode, defaultArg = 'f90-2ptr-arg')
    self.setCompilers.LIBS = oldLIBS
    self.popLanguage()
    # Cleanup
    if os.path.isfile(cobj):
      os.remove(cobj)
    if found:
      self.addDefine('HAVE_F90_2PTR_ARG', 1)
      self.logPrint('F90 compiler uses two arguments for array pointers', 3, 'compilers')
    else:
      self.logPrint('F90 uses a single argument for array pointers', 3, 'compilers')
    return

  def checkFortran90AssumedType(self):
    if config.setCompilers.Configure.isIBM(self.setCompilers.FC, self.log):
      self.addDefine('HAVE_F90_ASSUMED_TYPE_NOT_PTR', 1)
      self.logPrint('IBM F90 compiler detected so using HAVE_F90_ASSUMED_TYPE_NOT_PTR', 3, 'compilers')

  def checkFortranModuleInclude(self):
    '''Figures out what flag is used to specify the include path for Fortran modules'''
    self.setCompilers.fortranModuleIncludeFlag = None
    if not self.fortranIsF90:
      self.logPrint('Not a Fortran90 compiler - hence skipping module include test')
      return
    found   = False
    testdir = os.path.join(self.tmpDir, 'confdir')
    modobj  = os.path.join(self.tmpDir, 'configtest.o')
    modcode = '''\
      module configtest
      integer testint
      parameter (testint = 42)
      end module configtest\n'''
    # Compile the Fortran test module
    self.pushLanguage('FC')
    if not self.checkCompile(modcode, None, cleanup = 0):
      self.logPrint('Cannot compile Fortran module', 3, 'compilers')
      self.popLanguage()
      raise RuntimeError('Cannot determine Fortran module include flag')
    if not os.path.isfile(self.compilerObj):
      self.logPrint('Cannot locate object file: '+os.path.abspath(self.compilerObj), 3, 'compilers')
      self.popLanguage()
      raise RuntimeError('Cannot determine Fortran module include flag')
    if not os.path.isdir(testdir):
      os.mkdir(testdir)
    os.rename(self.compilerObj, modobj)
    foundModule = 0
    for f in [os.path.abspath('configtest.mod'), os.path.abspath('CONFIGTEST.mod'), os.path.join(os.path.dirname(self.compilerObj),'configtest.mod'), os.path.join(os.path.dirname(self.compilerObj),'CONFIGTEST.mod')]:
      if os.path.isfile(f):
        modname     = f
        foundModule = 1
        break
    if not foundModule:
      d = os.path.dirname(os.path.abspath('configtest.mod'))
      self.logPrint('Directory '+d+' contents:\n'+str(os.listdir(d)))
      raise RuntimeError('Fortran module was not created during the compile. %s/CONFIGTEST.mod not found' % os.path.abspath('configtest.mod'))
    shutil.move(modname, os.path.join(testdir, os.path.basename(modname)))
    fcode = '''\
      use configtest

      write(*,*) testint\n'''
    self.pushLanguage('FC')
    oldFLAGS = self.setCompilers.FFLAGS
    oldLIBS  = self.setCompilers.LIBS
    for flag in ['-I', '-p', '-M']:
      self.setCompilers.FFLAGS = flag+testdir+' '+self.setCompilers.FFLAGS
      self.setCompilers.LIBS   = modobj+' '+self.setCompilers.LIBS
      if not self.checkLink(None, fcode):
        self.logPrint('Fortran module include flag '+flag+' failed', 3, 'compilers')
      else:
        self.logPrint('Fortran module include flag '+flag+' found', 3, 'compilers')
        self.setCompilers.fortranModuleIncludeFlag = flag
        found = 1
      self.setCompilers.LIBS   = oldLIBS
      self.setCompilers.FFLAGS = oldFLAGS
      if found: break
    self.popLanguage()
    if os.path.isfile(modobj):
      os.remove(modobj)
    os.remove(os.path.join(testdir, os.path.basename(modname)))
    os.rmdir(testdir)
    if not found:
      raise RuntimeError('Cannot determine Fortran module include flag')
    return

  def checkFortranModuleOutput(self):
    '''Figures out what flag is used to specify the output path for Fortran modules'''
    self.setCompilers.fortranModuleOutputFlag = None
    if not self.fortranIsF90:
      self.logPrint('Not a Fortran90 compiler - hence skipping module include test')
      return
    found   = False
    testdir = os.path.join(self.tmpDir, 'confdir')
    modobj  = os.path.join(self.tmpDir, 'configtest.o')
    modcode = '''\
      module configtest
      integer testint
      parameter (testint = 42)
      end module configtest\n'''
    modname = None
    # Compile the Fortran test module
    if not os.path.isdir(testdir):
      os.mkdir(testdir)
    self.pushLanguage('FC')
    oldFLAGS = self.setCompilers.FFLAGS
    oldLIBS  = self.setCompilers.LIBS
    for flag in ['-module ', '-module:', '-fmod=', '-J', '-M', '-p', '-qmoddir=', '-moddir=']:
      self.setCompilers.FFLAGS = flag+testdir+' '+self.setCompilers.FFLAGS
      self.setCompilers.LIBS   = modobj+' '+self.setCompilers.LIBS
      if not self.checkCompile(modcode, None, cleanup = 0):
        self.logPrint('Fortran module output flag '+flag+' compile failed', 3, 'compilers')
      elif os.path.isfile(os.path.join(testdir, 'configtest.mod')) or os.path.isfile(os.path.join(testdir, 'CONFIGTEST.mod')):
        if os.path.isfile(os.path.join(testdir, 'configtest.mod')): modname = 'configtest.mod'
        if os.path.isfile(os.path.join(testdir, 'CONFIGTEST.mod')): modname = 'CONFIGTEST.mod'
        self.logPrint('Fortran module output flag '+flag+' found', 3, 'compilers')
        self.setCompilers.fortranModuleOutputFlag = flag
        found = 1
      else:
        self.logPrint('Fortran module output flag '+flag+' failed', 3, 'compilers')
      self.setCompilers.LIBS   = oldLIBS
      self.setCompilers.FFLAGS = oldFLAGS
      if found: break
    self.popLanguage()
    if modname: os.remove(os.path.join(testdir, modname))
    os.rmdir(testdir)
    # Flag not used by PETSc - do do not flag a runtime error
    #if not found:
    #  raise RuntimeError('Cannot determine Fortran module output flag')
    return

  def checkDependencyGenerationFlag(self):
    '''Check if -MMD works for dependency generation, and add it if it does'''
    self.generateDependencies       = {}
    self.dependenciesGenerationFlag = {}
    if not self.argDB['with-dependencies'] :
      self.logPrint("Skip checking dependency compiler options on user request")
      return
    languages = ['FC']
    for language in languages:
      self.generateDependencies[language] = 0
      self.setCompilers.saveLog()
      self.setCompilers.pushLanguage(language)
      for testFlag in ['-MMD -MP', # GCC, Intel, Clang, Pathscale
                       '-MMD',     # PGI
                       '-xMMD',    # Sun
                       '-qmakedep=gcc', # xlc
                       '-MD',
                       # Cray only supports -M, which writes to stdout
                     ]:
        try:
          self.logPrint('Trying '+language+' compiler flag '+testFlag)
          if self.setCompilers.checkCompilerFlag(testFlag, compilerOnly = 1):
            depFilename = os.path.splitext(self.setCompilers.compilerObj)[0]+'.d'
            if os.path.isfile(depFilename):
              os.remove(depFilename)
              #self.setCompilers.insertCompilerFlag(testFlag, compilerOnly = 1)
              self.framework.addMakeMacro(language.upper()+'_DEPFLAGS',testFlag)
              self.dependenciesGenerationFlag[language] = testFlag
              self.generateDependencies[language]       = 1
              break
            else:
              self.logPrint('Rejected '+language+' compiler flag '+testFlag+' because no dependency file ('+depFilename+') was generated')
          else:
            self.logPrint('Rejected '+language+' compiler flag '+testFlag)
        except RuntimeError:
          self.logPrint('Rejected '+language+' compiler flag '+testFlag)
      self.setCompilers.popLanguage()
      self.logWrite(self.setCompilers.restoreLog())
    return

  def configure(self):
    import config.setCompilers
    if hasattr(self.setCompilers, 'FC'):
      self.executeTest(self.checkFortranTypeSizes)
      self.executeTest(self.checkFortranPreprocessor)
      self.executeTest(self.checkFortranDefineCompilerOption)
      self.executeTest(self.checkFortran90)
      self.executeTest(self.checkFortran90FreeForm)
      self.executeTest(self.checkFortran2003)
      self.executeTest(self.checkFortran90Array)
      self.executeTest(self.checkFortran90AssumedType)
      self.executeTest(self.checkFortranModuleInclude)
      self.executeTest(self.checkFortranModuleOutput)
      self.executeTest(self.checkFortranTypeStar)
      self.executeTest(self.checkFortranTypeInitialize)
      self.executeTest(self.configureFortranFlush)
      self.executeTest(self.checkDependencyGenerationFlag)
      self.executeTest(self.checkFortran90LineLength)
    return

