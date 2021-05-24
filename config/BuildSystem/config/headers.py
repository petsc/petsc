import config.base

class Configure(config.base.Configure):
  def __init__(self, framework, headers = []):
    config.base.Configure.__init__(self, framework)
    self.headerPrefix = ''
    self.substPrefix  = ''
    self.headers      = headers
    return

  def setupDependencies(self, framework):
    config.base.Configure.setupDependencies(self, framework)
    self.compilers    = self.framework.require('config.compilers', self)
    self.setCompilers = self.framework.require('config.setCompilers', self)
    return

  def getIncludeArgumentList(self, include):
    '''Return the proper include line argument for the given filename as a list
       - If the path is empty, return it unchanged
       - If starts with - then return unchanged
       - Otherwise return -I<include>'''
    if not include:
      return []
    include = include.replace('\\ ',' ').replace(' ', '\\ ')
    include = include.replace('\\(','(').replace('(', '\\(')
    include = include.replace('\\)',')').replace(')', '\\)')
    if include[0] == '-':
      return [include]
    return ['-I'+include]

  def getIncludeModulesArgumentList(self, include):
    '''Return the proper include line argument for the given filename as a list
       - If the path is empty, return it unchanged
       - If starts with - then return unchanged
       - Otherwise return -fortranmoduleflag includedirectory'''
    if not include:
      return []
    include = include.replace('\\ ',' ').replace(' ', '\\ ')
    include = include.replace('\\(','(').replace('(', '\\(')
    include = include.replace('\\)',')').replace(')', '\\)')
    if include[0] == '-':
      return [include]

    self.pushLanguage('FC')
    string = self.setCompilers.fortranModuleIncludeFlag+include
    self.popLanguage()
    return [string]

  def getIncludeArgument(self, include):
    '''Same as getIncludeArgumentList - except it returns a string instead of list.'''
    return  ' '.join(self.getIncludeArgumentList(include))

  def toString(self,includes):
    '''Converts a list of includes to a string suitable for a compiler'''
    return ' '.join([self.getIncludeArgument(include) for include in includes])

  def toStringNoDupes(self,includes,modincludes=[]):
    '''Converts a list of -Iincludes and -fmodule flags to a string suitable for a compiler, removes duplicates'''
    newincludes = []
    for include in includes:
      newincludes += self.getIncludeArgumentList(include)
    for modinclude in modincludes:
      newincludes += self.getIncludeModulesArgumentList(modinclude)
    includes = newincludes
    newincludes = []
    for j in includes:
      if j in newincludes: continue
      newincludes.append(j)
    return ' '.join(newincludes)

  def getDefineName(self, header):
    return 'HAVE_'+header.upper().replace('.', '_').replace('/', '_')

  def haveHeader(self, header):
    return self.getDefineName(header) in self.defines

  def check(self,header, adddefine = 1):
    '''Checks for "header", and defines HAVE_"header" if found'''
    self.log.write('Checking for header: '+header+'\n')
    found = 0
    if self.checkPreprocess('#include <'+header+'>\n'):
      found = 1
      if adddefine: self.addDefine(self.getDefineName(header), found)
    return found

  def checkInclude(self, incl, hfiles, otherIncludes = [], timeout = 600.0):
    '''Checks if a particular include file can be found along particular include paths'''
    if not isinstance(hfiles, list):
      hfiles = [hfiles]
    if not isinstance(incl, list):
      inclu = [incl]
    self.log.write('Checking for header files ' +str(hfiles)+ ' in '+str(incl)+'\n')
    for hfile in hfiles:
      flagsArg = self.getPreprocessorFlagsArg()
      self.logPrint('Checking include with compiler flags var '+flagsArg+' '+str(incl+otherIncludes))
      #oldFlags = self.compilers.CPPFLAGS
      oldFlags = getattr(self.compilers, flagsArg)
      #self.compilers.CPPFLAGS += ' '+' '.join([self.getIncludeArgument(inc) for inc in incl+otherIncludes])
      setattr(self.compilers, flagsArg, getattr(self.compilers, flagsArg)+' '+' '.join([self.getIncludeArgument(inc) for inc in incl+otherIncludes]))
      found = self.checkPreprocess('#include <' +hfile+ '>\n', timeout = timeout)
      #self.compilers.CPPFLAGS = oldFlags
      setattr(self.compilers, flagsArg, oldFlags)
      if not found: return 0
    self.log.write('Found header files ' +str(hfiles)+ ' in '+str(incl)+'\n')
    return 1

  def checkStdC(self):
    haveStdC = 0
    includes = '''
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <float.h>
'''
    haveStdC = self.checkCompile(includes)
    # SunOS 4.x string.h does not declare mem*, contrary to ANSI.
    if haveStdC and not self.outputPreprocess('#include <string.h>').find('memchr'): haveStdC = 0
    # ISC 2.0.2 stdlib.h does not declare free, contrary to ANSI.
    if haveStdC and not self.outputPreprocess('#include <stdlib.h>').find('free'): haveStdC = 0
    # /bin/cc in Irix-4.0.5 gets non-ANSI ctype macros unless using -ansi.
    if haveStdC and not self.argDB['with-batch']:
      includes = '''
#include <stdlib.h>
#include <ctype.h>
#define ISLOWER(c) (\'a\' <= (c) && (c) <= \'z\')
#define TOUPPER(c) (ISLOWER(c) ? \'A\' + ((c) - \'a\') : (c))
#define XOR(e, f) (((e) && !(f)) || (!(e) && (f)))
'''
      body = '''
        int i;

        for(i = 0; i < 256; i++) if (XOR(islower(i), ISLOWER(i)) || toupper(i) != TOUPPER(i)) exit(2);
        exit(0);
      '''
      if not self.checkRun(includes, body): haveStdC = 0
    if not haveStdC:
      raise RuntimeError("Cannot locate all the standard C header files needed by PETSc")
    return

  def checkStat(self):
    '''Checks whether stat file-mode macros are broken, and defines STAT_MACROS_BROKEN if they are'''
    code = '''
#include <sys/types.h>
#include <sys/stat.h>

#if defined(S_ISBLK) && defined(S_IFDIR)
# if S_ISBLK (S_IFDIR)
  You lose.
# endif
#endif

#if defined(S_ISBLK) && defined(S_IFCHR)
# if S_ISBLK (S_IFCHR)
  You lose.
# endif
#endif

#if defined(S_ISLNK) && defined(S_IFREG)
# if S_ISLNK (S_IFREG)
  You lose.
# endif
#endif

#if defined(S_ISSOCK) && defined(S_IFREG)
# if S_ISSOCK (S_IFREG)
  You lose.
# endif
#endif
'''
    if self.outputPreprocess(code).find('You lose') >= 0:
      self.addDefine('STAT_MACROS_BROKEN', 1)
      return 0
    return 1

  def checkSysWait(self):
    '''Check for POSIX.1 compatible sys/wait.h, and defines HAVE_SYS_WAIT_H if found'''
    includes = '''
#include <sys/types.h>
#include <sys/wait.h>
#ifndef WEXITSTATUS
#define WEXITSTATUS(stat_val) ((unsigned)(stat_val) >> 8)
#endif
#ifndef WIFEXITED
#define WIFEXITED(stat_val) (((stat_val) & 255) == 0)
#endif
'''
    body = '''
    int s;
    wait (&s);
    s = WIFEXITED (s) ? WEXITSTATUS (s) : 1;
    '''
    if self.checkCompile(includes, body):
      self.addDefine('HAVE_SYS_WAIT_H', 1)
      return 1
    return 0

  def checkTime(self):
    '''Checks if you can safely include both <sys/time.h> and <time.h>, and if so defines TIME_WITH_SYS_TIME'''
    self.check('time.h')
    self.check('sys/time.h')
    return

  def checkMath(self):
    '''Checks for the math headers and defines'''
    haveMath = self.check('math.h',adddefine=0)
    if haveMath:
      if self.checkCompile('#include <math.h>\n', 'double pi = M_PI;\n\nif (pi);\n'):
        self.logPrint('Found math #defines, like M_PI')
      elif self.checkCompile('#define _USE_MATH_DEFINES 1\n#include <math.h>\n', 'double pi = M_PI;\n\nif (pi);\n'):
        self.framework.addDefine('_USE_MATH_DEFINES', 1)
        self.logPrint('Activated Windows math #defines, like M_PI')
      else:
        self.logPrint('Missing math #defines, like M_PI')
    else:
      raise RuntimeError("PETSc requires math.h")
    return

  def checkRecursiveMacros(self):
    '''Checks that the preprocessor allows recursive macros, and if not defines HAVE_BROKEN_RECURSIVE_MACRO'''
    includes = 'void a(int i, int j) {}\n#define a(b) a(b,__LINE__)'
    body     = 'a(0);\n'
    if not self.checkCompile(includes, body):
      self.addDefine('HAVE_BROKEN_RECURSIVE_MACRO', 1)
      return 0
    return 1

  def configure(self):
    self.executeTest(self.checkStdC)
    self.executeTest(self.checkStat)
    self.executeTest(self.checkSysWait)
    self.executeTest(self.checkTime)
    self.executeTest(self.checkMath)
    for header in self.headers:
      self.executeTest(self.check, header)
    self.executeTest(self.checkRecursiveMacros)
    return
