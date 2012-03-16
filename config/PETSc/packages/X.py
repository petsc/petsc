import PETSc.package
import config.autoconf

import os

class Configure(PETSc.package.NewPackage,config.autoconf.Configure):
  def __init__(self, framework):
    config.autoconf.Configure.__init__(self, framework)
    PETSc.package.NewPackage.__init__(self, framework)
    self.lookforbydefault=1
    return

  def setupHelp(self, help):
    import nargs
    PETSc.package.NewPackage.setupHelp(self, help)
    help.addArgument('X', '-with-xt=<bool>',               nargs.ArgBool(None, 0,   'Activate Xt'))
    return

  def setupDependencies(self, framework):
    PETSc.package.NewPackage.setupDependencies(self, framework)
    self.make = framework.require('config.programs', self)
    return

  def checkXMake(self):
    import shutil
    import time

    includeDir = ''
    libraryDir = ''
    # Create Imakefile
    testDir = os.path.join(self.tmpDir, 'Xtestdir')
    oldDir  = os.getcwd()
    if os.path.exists(testDir): shutil.rmtree(testDir)
    os.mkdir(testDir)
    os.chdir(testDir)
    f = file('Imakefile', 'w')
    f.write('''
acfindx:
	@echo \'X_INCLUDE_ROOT = ${INCROOT}\'
	@echo \'X_USR_LIB_DIR = ${USRLIBDIR}\'
	@echo \'X_LIB_DIR = ${LIBDIR}\'
''')
    f.close()
    # Compile makefile
    try:
      (output, error, status) = PETSc.package.NewPackage.executeShellCommand('xmkmf', log = self.framework.log)
      if not status and os.path.exists('Makefile'):
        (output, error, status) = PETSc.package.NewPackage.executeShellCommand(self.make.make+' acfindx', log = self.framework.log)
        results                 = self.parseShellOutput(output)
        if not ('X_INCLUDE_ROOT' in results and 'X_USR_LIB_DIR' in results and 'X_LIB_DIR' in results):
          raise RuntimeError('Invalid output: '+str(output))
        # Open Windows xmkmf reportedly sets LIBDIR instead of USRLIBDIR.
        if not os.path.isfile(os.path.join(results['X_USR_LIB_DIR'])) and os.path.isfile(os.path.join(results['X_LIB_DIR'])):
          results['X_USR_LIB_DIR'] = results['X_LIB_DIR']
        # Screen out bogus values from the imake configuration.  They are
        # bogus both because they are the default anyway, and because
        # using them would break gcc on systems where it needs fixed includes.
        if not results['X_INCLUDE_ROOT'] == '/usr/include' and os.path.isfile(os.path.join(results['X_INCLUDE_ROOT'], 'X11', 'Xos.h')):
          includeDir = results['X_INCLUDE_ROOT']
        if not (results['X_USR_LIB_DIR'] == '/lib' or results['X_USR_LIB_DIR'] == '/usr/lib' or results['X_USR_LIB_DIR'] == '/usr/lib64') and os.path.isdir(results['X_USR_LIB_DIR']):
          libraryDir = results['X_USR_LIB_DIR']
    except RuntimeError, e:
      self.framework.log.write('Error using Xmake: '+str(e)+'\n')
    # Cleanup
    os.chdir(oldDir)
    time.sleep(1)
    shutil.rmtree(testDir)
    return (includeDir, libraryDir)

  def generateGuesses(self):
    '''Generate list of possible locations of X11'''
    useXt        = self.framework.argDB['with-xt']
    includeDirs  = ['/Developer/SDKs/MacOSX10.5.sdk/usr/X11/include',
                    '/Developer/SDKs/MacOSX10.4u.sdk/usr/X11R6/include',
                    '/usr/X11/include',
                   '/usr/X11R6/include',
                   '/usr/X11R5/include',
                   '/usr/X11R4/include',
                   '/usr/include/X11',
                   '/usr/include/X11R6',
                   '/usr/include/X11R5',
                   '/usr/include/X11R4',
                   '/usr/local/X11/include',
                   '/usr/local/X11R6/include',
                   '/usr/local/X11R5/include',
                   '/usr/local/X11R4/include',
                   '/usr/local/include/X11',
                   '/usr/local/include/X11R6',
                   '/usr/local/include/X11R5',
                   '/usr/local/include/X11R4',
                   '/usr/X386/include',
                   '/usr/x386/include',
                   '/usr/XFree86/include/X11',
                   '/usr/include',
                   '/usr/local/include',
                   '/usr/unsupported/include',
                   '/usr/athena/include',
                   '/usr/local/x11r5/include',
                   '/usr/lpp/Xamples/include',
                   '/usr/openwin/include',
                   '/usr/openwin/share/include']
    if self.framework.argDB.has_key('with-x-include'):
      if not os.path.isdir(self.framework.argDB['with-x-include']):
        raise RuntimeError('Invalid X include directory specified by --with-x-include='+os.path.abspath(self.framework.argDB['with-x-include']))
      includeDir = self.framework.argDB['with-x-include']

    testLibraries = ['libX11.a'] # 'XSetWMName'
    if useXt:
      testLibraries.append('libXt.a') # 'XtMalloc'
    # Guess X location
    (includeDirGuess, libraryDirGuess) = self.checkXMake()
    yield ('Standard X Location', libraryDirGuess, testLibraries, includeDirGuess)
    return

  def configureLibrary(self):
    '''Checks for X windows, sets PETSC_HAVE_X if found, and defines X_CFLAGS, X_PRE_LIBS, X_LIBS, and X_EXTRA_LIBS'''
    # This needs to be rewritten to use generateGuesses()
    foundInclude = 0
    includeDirs  = ['/Developer/SDKs/MacOSX10.5.sdk/usr/X11/include',
                    '/Developer/SDKs/MacOSX10.4u.sdk/usr/X11R6/include',
                    '/usr/X11/include',
                   '/usr/X11R6/include',
                   '/usr/X11R5/include',
                   '/usr/X11R4/include',
                   '/usr/include/X11',
                   '/usr/include/X11R6',
                   '/usr/include/X11R5',
                   '/usr/include/X11R4',
                   '/usr/local/X11/include',
                   '/usr/local/X11R6/include',
                   '/usr/local/X11R5/include',
                   '/usr/local/X11R4/include',
                   '/usr/local/include/X11',
                   '/usr/local/include/X11R6',
                   '/usr/local/include/X11R5',
                   '/usr/local/include/X11R4',
                   '/usr/X386/include',
                   '/usr/x386/include',
                   '/usr/XFree86/include/X11',
                   '/usr/include',
                   '/usr/local/include',
                   '/usr/unsupported/include',
                   '/usr/athena/include',
                   '/usr/local/x11r5/include',
                   '/usr/lpp/Xamples/include',
                   '/usr/openwin/include',
                   '/usr/openwin/share/include']
    includeDir   = ''
    foundLibrary = 0
    libraryDirs  = map(lambda s: s.replace('include', 'lib'), includeDirs)
    libraryDir   = ''
    # Guess X location
    (includeDirGuess, libraryDirGuess) = self.checkXMake()
    # Check for X includes
    if self.framework.argDB.has_key('with-x-include'):
      if not os.path.isdir(self.framework.argDB['with-x-include']):
        raise RuntimeError('Invalid X include directory specified by --with-x-include='+os.path.abspath(self.framework.argDB['with-x-include']))
      includeDir = self.framework.argDB['with-x-include']
      foundInclude = 1
    else:
      includes  = ['X11/Xlib.h']
      if self.framework.argDB['with-xt']:
        includes.append('X11/Intrinsic.h')

      for testInclude in includes:
        # Check guess
        if includeDirGuess and os.path.isfile(os.path.join(includeDirGuess, testInclude)):
          foundInclude = 1
          includeDir   = includeDirGuess
          # Check default compiler paths
        if not foundInclude and self.checkPreprocess('#include <'+testInclude+'>\n'):
          foundInclude = 1
        # Check standard paths
        if not foundInclude:
          for dir in includeDirs:
            if os.path.isfile(os.path.join(dir, testInclude)):
              foundInclude = 1
              includeDir   = dir
        if not foundInclude:
          break
    # Check for X11 libraries
    if self.framework.argDB.has_key('with-x-lib'):
      if not os.path.isfile(self.framework.argDB['with-x-lib'][0]):
        raise RuntimeError('Invalid X library specified by --with-x-lib='+os.path.abspath(self.framework.argDB['with-x-lib'][0]))
      libraryDir = os.path.dirname(self.framework.argDB['with-x-lib'][0])
      foundLibrary = 1
    else:
      testLibraries = [('X11', 'XSetWMName')]
      if self.framework.argDB['with-xt']:
        testLibraries.append(('Xt', 'XtMalloc'))

      # Check guess
      for testLibrary, testFunction in testLibraries:
        if libraryDirGuess:
          for ext in ['.a', '.so', '.sl', '.dll.a','.dylib']:
            if os.path.isfile(os.path.join(libraryDirGuess, 'lib'+testLibrary+ext)):
              foundLibrary = 1
              libraryDir   = libraryDirGuess
              break
        # Check default compiler libraries
        if not foundLibrary:
          oldLibs = self.compilers.LIBS
          self.compilers.LIBS += ' -l'+testLibrary
          self.pushLanguage(self.language[-1])
          if self.checkLink('', testFunction+'();\n'):
            foundLibrary = 1
          self.compilers.LIBS = oldLibs
          self.popLanguage()
        # Check standard paths
        if not foundLibrary:
          for dir in libraryDirs:
            for ext in ['.a', '.so', '.sl', '.dll.a','.dylib']:
              if os.path.isfile(os.path.join(dir, 'lib'+testLibrary+ext)):
                foundLibrary = 1
                libraryDir   = dir
        if not foundLibrary:
          break
      # Verify that library can be linked with
      if foundLibrary:
        oldLibs = self.compilers.LIBS
        if libraryDir:
          self.compilers.LIBS += ' -L'+libraryDir
        self.compilers.LIBS += ' -l'+testLibrary
        self.pushLanguage(self.language[-1])
        if not self.checkLink('', testFunction+'();\n'):
          foundLibrary = 0
        self.compilers.LIBS = oldLibs
        self.popLanguage()
          
    if foundInclude and foundLibrary:
      self.logPrint('Found X includes and libraries')
      self.found     = 1
      self.include = includeDir
      if libraryDir:
        self.lib     = ['-L'+libraryDir,'-lX11']
      else:
        self.lib     = ['-lX11']

      self.addSubstitution('X_CFLAGS',     self.headers.getIncludeArgument(self.include))
      self.addSubstitution('X_LIBS',       self.lib)
      self.addSubstitution('X_PRE_LIBS',   '')
      self.addSubstitution('X_EXTRA_LIBS', '')
      if hasattr(self.framework, 'packages'):
        self.framework.packages.append(self)
    else:
      if self.framework.clArgDB.get('with-x'):
        raise RuntimeError("Could not locate X *development* package. Perhaps its not installed")
      if not foundInclude:
        self.logPrint('Could not find X includes')
      if not foundLibrary:
        self.logPrint('Could not find X libraries')
    self.dlib = self.lib
    return

  def configure(self):
    if self.framework.argDB['with-x']:
      if not self.libraryOptions.integerSize == 32:
        self.logPrintBox('Turning off X because integers are not 32 bit', debugSection = None)
        return
#      if not self.scalartypes.precision == 'double':
#        self.logPrintBox('Turning off X because scalars are not doubles', debugSection = None)
#        return
      self.executeTest(self.configureLibrary)
    return
