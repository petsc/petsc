import config.base

class Configure(config.base.Configure):
  def __init__(self, framework, functions = []):
    config.base.Configure.__init__(self, framework)
    self.headerPrefix = ''
    self.substPrefix  = ''
    self.functions    = functions
    self.headers      = self.framework.require('config.headers', self)
    return

  def getDefineName(self, funcName):
    return 'HAVE_'+funcName.upper()

  def haveFunction(self, function):
    return self.getDefineName(function) in self.defines

  def check(self, funcName, libraries = None):
    '''Checks for the function "funcName", and if found defines HAVE_"funcName"'''
    self.framework.log.write('Checking for function '+funcName+'\n')
    # Don't include <ctype.h> because on OSF/1 3.0 it includes <sys/types.h>
    # which includes <sys/select.h> which contains a prototype for
    # select.  Similarly for bzero.
    includes  = '/* System header to define __stub macros and hopefully few prototypes, which can conflict with char '+funcName+'(); below. */\n'
    includes += '''
    #include <assert.h>
    /* Override any gcc2 internal prototype to avoid an error. */
    '''
    if self.language[-1] == 'C++':
      includes += '''
      #ifdef __cplusplus
      extern "C"
      #endif'''
    includes += '''
    /* We use char because int might match the return type of a gcc2
    builtin and then its argument prototype would still apply. */
    '''
    includes += 'char '+funcName+'();\n'
    body = '''
    /* The GNU C library defines this for functions which it implements
    to always fail with ENOSYS.  Some functions are actually named
    something starting with __ and the normal name is an alias.  */
    #if defined (__stub_'''+funcName+''') || defined (__stub___'''+funcName+''')
    choke me
    #else
    '''+funcName+'''();
    #endif
    '''
    if libraries:
      oldLibs = self.framework.argDB['LIBS']
      if not isinstance(libraries, list): libraries
      for library in libraries:
        self.framework.argDB['LIBS'] += ' -l'+library
    if self.checkLink(includes, body):
      self.addDefine(self.getDefineName(funcName), 1)
    if libraries:
      self.framework.argDB['LIBS'] = oldLibs
    return

  def checkMemcmp(self):
    '''Check for 8-bit clean memcmp'''
    if not self.checkRun('#include <string.h>\nvoid exit(int);\n\n', 'char c0 = 0x40, c1 = 0x80, c2 = 0x81;\nexit(memcmp(&c0, &c2, 1) < 0 && memcmp(&c1, &c2, 1) < 0 ? 0 : 1);\n'):
      raise RuntimeError('Failed to find 8-bit clean memcmp()')
    return

  def checkSysinfo(self):
    '''Check whether sysinfo takes three arguments, and if it does define HAVE_SYSINFO_3ARG'''
    self.check('sysinfo')
    if self.defines.has_key(self.getDefineName('sysinfo')):
      map(self.headers.check, ['linux/kernel.h', 'sys/sysinfo.h', 'sys/systeminfo.h'])
      includes = '''
      #ifdef HAVE_LINUX_KERNEL_H
      #  include <linux/kernel.h>
      #  include <linux/sys.h>
      #  ifdef HAVE_SYS_SYSINFO_H
      #    include <sys/sysinfo.h>
      #  endif
      #elif defined(HAVE_SYS_SYSTEMINFO_H)
      #  include <sys/systeminfo.h>
      #else
      #  error "Cannot check sysinfo without special headers"
      #endif
      '''
      body = 'char buf[10]; long count=10; sysinfo(1, buf, count);\n'
      if self.checkCompile(includes, body):
        self.addDefine('HAVE_SYSINFO_3ARG', 1)
    return

  def checkVPrintf(self):
    self.check('vprintf')
    '''Checks whether vprintf requires a char * last argument, and if it does defines HAVE_VPRINTF_CHAR'''
    if not self.checkCompile('#include <stdio.h>\n#include <stdarg.h>\n', 'va_list Argp;\nvprintf( "%d", Argp );\n'):
      self.addDefine('HAVE_VPRINTF_CHAR', 1)
    return

  def checkSignalHandlerType(self):
    '''Checks the type of C++ signals handlers, and defines SIGNAL_CAST to the correct value'''
    self.pushLanguage('C++')
    if not self.checkCompile('#include <signal.h>\nstatic void myhandler(int sig) {}\n', 'signal(SIGFPE,myhandler);\n'):
      self.addDefine('SIGNAL_CAST', '(void (*)(int))')
    else:
      self.addDefine('SIGNAL_CAST', ' ')
    self.popLanguage()
    return

  def checkFreeReturnType(self):
    '''Checks whether free returns void or int, and defines HAVE_FREE_RETURN_INT'''
    if self.checkCompile('#include <stdlib.h>\n', 'int free(void *); void *p; return free(p);\n'):
      self.addDefine('HAVE_FREE_RETURN_INT', 1)
    return

  def configure(self):
    self.executeTest(self.checkMemcmp)
    self.executeTest(self.checkSysinfo)
    self.executeTest(self.checkVPrintf)
    self.executeTest(self.checkSignalHandlerType)
    self.executeTest(self.checkFreeReturnType)
    map(lambda function: self.executeTest(self.check, function), self.functions)
    return
