#!/usr/bin/env python
from __future__ import generators
import user
import config.base

class Configure(config.base.Configure):
  def __init__(self, framework):
    config.base.Configure.__init__(self, framework)
    self.headerPrefix = ''
    self.substPrefix  = ''
    return

  def __str__(self):
    return ''
    
  def setupHelp(self, help):
    import nargs
    return

  def setupDependencies(self, framework):
    config.base.Configure.setupDependencies(self, framework)
    self.compilers = framework.require('config.compilers', self)
    self.functions = framework.require('config.functions', self)
    self.libraries = framework.require('config.libraries', self)
    return

  def checkPrototype(self, includes = '', body = '', cleanup = 1, codeBegin = None, codeEnd = None):
    (output, error, status) = self.outputCompile(includes, body, cleanup, codeBegin, codeEnd)
    output += error
    if output.find('implicit') >= 0 or output.find('Implicit') >= 0:
      return 0
    return 1

#-------------------------------------------------------
  def configureMissingDefines(self):
    '''Checks for limits'''
    if not self.checkCompile('#ifdef PETSC_HAVE_LIMITS_H\n  #include <limits.h>\n#endif\n', 'int i=INT_MAX;\n\nif (i);\n'):
      self.addDefine('INT_MIN', '(-INT_MAX - 1)')
      self.addDefine('INT_MAX', 2147483647)
    if not self.checkCompile('#ifdef PETSC_HAVE_FLOAT_H\n  #include <float.h>\n#endif\n', 'double d=DBL_MAX;\n\nif (d);\n'):
      self.addDefine('DBL_MIN', 2.2250738585072014e-308)
      self.addDefine('DBL_MAX', 1.7976931348623157e+308)
    return

  def configureMissingUtypeTypedefs(self):
    ''' Checks if u_short is undefined '''
    if not self.checkCompile('#include <sys/types.h>\n', 'u_short foo;\n'):
      self.addDefine('NEEDS_UTYPE_TYPEDEFS',1)
    return

  def configureMissingFunctions(self):
    '''Checks for SOCKETS'''
    if not self.functions.haveFunction('socket'):
      # solaris requires these two libraries for socket()
      if self.libraries.haveLib('socket') and self.libraries.haveLib('nsl'):
        # check if it can find the function
        if self.functions.check('socket',['-lsocket','-lnsl']):
          self.addDefine('HAVE_SOCKET', 1)
          self.compilers.LIBS += ' -lsocket -lnsl'
        
      # Windows requires Ws2_32.lib for socket(), uses stdcall, and declspec prototype decoration
      if self.libraries.add('Ws2_32.lib','socket',prototype='#include <Winsock2.h>',call='socket(0,0,0);'):
        self.addDefine('HAVE_WINSOCK2_H',1)
        self.addDefine('HAVE_SOCKET', 1)
        if self.checkLink('#include <Winsock2.h>','closesocket(0)'):
          self.addDefine('HAVE_CLOSESOCKET',1)
        if self.checkLink('#include <Winsock2.h>','WSAGetLastError()'):
          self.addDefine('HAVE_WSAGETLASTERROR',1)
    return

  def configureMissingSignals(self):
    '''Check for missing signals, and define MISSING_<signal name> if necessary'''
    for signal in ['ABRT', 'ALRM', 'BUS',  'CHLD', 'CONT', 'FPE',  'HUP',  'ILL', 'INT',  'KILL', 'PIPE', 'QUIT', 'SEGV',
                   'STOP', 'SYS',  'TERM', 'TRAP', 'TSTP', 'URG',  'USR1', 'USR2']:
      if not self.checkCompile('#include <signal.h>\n', 'int i=SIG'+signal+';\n\nif (i);\n'):
        self.addDefine('MISSING_SIG'+signal, 1)
    return


  def configureMissingErrnos(self):
    '''Check for missing errno values, and define MISSING_<errno value> if necessary'''
    for errnoval in ['EINTR']:
      if not self.checkCompile('#include <errno.h>','int i='+errnoval+';\n\nif (i);\n'):
        self.addDefine('MISSING_ERRNO_'+errnoval, 1)
    return
  

  def configureMissingGetdomainnamePrototype(self):
    head ='''
#ifdef PETSC_HAVE_UNISTD_H
#include <unistd.h>
#endif
#ifdef PETSC_HAVE_NETDB_H
#include <netdb.h>
#endif
'''
    code = '''
char test[10];
int err = getdomainname(test,10);
'''
    if not self.checkPrototype(head,code):
      self.addPrototype('int getdomainname(char *, int);', 'C')
    if hasattr(self.compilers, 'CXX'):
      self.pushLanguage('C++')
      if not self.checkLink(head,code):
        self.addPrototype('int getdomainname(char *, int);', 'extern C')
      self.popLanguage()  
    return

  def configureMissingSrandPrototype(self):
    head ='''
#ifdef PETSC_HAVE_STDLIB_H
#include <stdlib.h>
#endif
'''
    code = '''
double a;
long   b=10;
srand(b);
a=drand48();

'''
    if not self.checkPrototype(head,code):
      self.addPrototype('double drand48();', 'C')
      self.addPrototype('void   srand48(long);', 'C')
    if hasattr(self.compilers, 'CXX'):
      self.pushLanguage('C++')
      if not self.checkLink(head,code):
        self.addPrototype('double drand48();', 'extern C')
        self.addPrototype('void   srand48(long);', 'extern C')
      self.popLanguage()  
    return

  def configureMissingIntelFastPrototypes(self):
    if self.functions.haveFunction('_intel_fast_memcpy'):
      self.addPrototype('#include <stddef.h> \nvoid *_intel_fast_memcpy(void *,const void *,size_t);', 'C')
      if hasattr(self.compilers, 'CXX'):
        self.pushLanguage('C++')
        self.addPrototype('#include <stddef.h> \nvoid *_intel_fast_memcpy(void *,const void *,size_t);', 'extern C')
        self.popLanguage()
    if self.functions.haveFunction('_intel_fast_memset'):
      self.addPrototype('#include <stddef.h> \nvoid *_intel_fast_memset(void *, int, size_t);', 'C')
      if hasattr(self.compilers, 'CXX'):
        self.pushLanguage('C++')
        self.addPrototype('#include <stddef.h> \nvoid *_intel_fast_memset(void *, int, size_t);', 'extern C')
        self.popLanguage()
    return

  def configure(self):
    self.executeTest(self.configureMissingDefines)
    self.executeTest(self.configureMissingUtypeTypedefs)
    self.executeTest(self.configureMissingFunctions)
    self.executeTest(self.configureMissingSignals)
    self.executeTest(self.configureMissingGetdomainnamePrototype)
    self.executeTest(self.configureMissingSrandPrototype)
    self.executeTest(self.configureMissingIntelFastPrototypes)
    return
