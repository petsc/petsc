#!/usr/bin/env python
from __future__ import generators
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
    self.ftm = framework.require('config.utilities.featureTestMacros', self)
    return

  def featureTestMacros(self):
    features = ''
    if self.ftm.defines.get('_POSIX_C_SOURCE_200112L'):
      features += '''
#if !defined(_POSIX_C_SOURCE)
#define _POSIX_C_SOURCE 200112L
#endif
'''
    if self.ftm.defines.get('_BSD_SOURCE'):
      features += '''
#if !defined(_BSD_SOURCE)
#define _BSD_SOURCE
#endif
'''
    if self.ftm.defines.get('_DEFAULT_SOURCE'):
      features += '''
#if !defined(_DEFAULT_SOURCE)
#define _DEFAULT_SOURCE
#endif
'''
    if self.ftm.defines.get('_GNU_SOURCE'):
      features += '''
#if !defined(_GNU_SOURCE)
#define _GNU_SOURCE
#endif
'''
    return features

  def configureMissingUtypeTypedefs(self):
    ''' Checks if u_short is undefined '''
    if not self.checkCompile('#include <sys/types.h>\n', 'u_short foo;\n'):
      self.addDefine('NEEDS_UTYPE_TYPEDEFS',1)
    return

  def configureMissingFunctions(self):
    '''Checks for SOCKETS and getline'''
    if not self.functions.haveFunction('socket'):
      # solaris requires these two libraries for socket()
      if self.libraries.haveLib('socket') and self.libraries.haveLib('nsl'):
        # check if it can find the function
        if self.functions.check('socket',['-lsocket','-lnsl']):
          self.compilers.LIBS += ' -lsocket -lnsl'

      # Windows requires Ws2_32.lib for socket(), uses stdcall, and declspec prototype decoration
      if self.libraries.add('Ws2_32.lib','socket',prototype='#include <Winsock2.h>',call='socket(0,0,0);'):
        self.addDefine('HAVE_WINSOCK2_H',1)
        if self.checkLink('#include <Winsock2.h>','closesocket(0)'):
          self.addDefine('HAVE_CLOSESOCKET',1)
        if self.checkLink('#include <Winsock2.h>','WSAGetLastError()'):
          self.addDefine('HAVE_WSAGETLASTERROR',1)
    if not self.checkLink('#include <stdio.h>\nchar *lineptr;\nsize_t n;\nFILE *stream;\n', 'getline(&lineptr, &n, stream);\n'):
      self.addDefine('MISSING_GETLINE', 1)
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
    head = self.featureTestMacros() + '''
#ifdef PETSC_HAVE_UNISTD_H
#include <unistd.h>
#endif
#ifdef PETSC_HAVE_NETDB_H
#include <netdb.h>
#endif
'''
    def code(t):
      """The type of the len parameter is size_t on Linux and int on BSD, so we'll have to try both."""
      return '''
int (*getdomainname_ptr)(char*,%s) = getdomainname;
char test[10];
if (getdomainname_ptr(test,10)) return 1;
''' % (t,)
    if not (self.checkCompile(head,code('size_t')) or self.checkCompile(head,code('int'))):
      self.addPrototype('#include <stddef.h>\nint getdomainname(char *, size_t);', 'C')
    if hasattr(self.compilers, 'CXX'):
      self.pushLanguage('C++')
      if not (self.checkLink(head,code('size_t')) or self.checkLink(head,code('int'))):
        self.addPrototype('#include <stddef.h>\nint getdomainname(char *, size_t);', 'extern C')
      self.popLanguage()
    return

  def configureMissingSrandPrototype(self):
    head = self.featureTestMacros() + '''
#include <stdlib.h>
'''
    code = '''
double (*drand48_ptr)(void) = drand48;
void (*srand48_ptr)(long int) = srand48;
long int seed=10;
srand48_ptr(seed);
if (drand48_ptr() > 0.5) return 1;
'''
    if not self.checkCompile(head,code):
      self.addPrototype('double drand48(void);', 'C')
      self.addPrototype('void   srand48(long int);', 'C')
    if hasattr(self.compilers, 'CXX'):
      self.pushLanguage('C++')
      if not self.checkLink(head,code):
        self.addPrototype('double drand48(void);', 'extern C')
        self.addPrototype('void   srand48(long int);', 'extern C')
      self.popLanguage()
    return

  def configure(self):
    self.executeTest(self.configureMissingUtypeTypedefs)
    self.executeTest(self.configureMissingFunctions)
    self.executeTest(self.configureMissingSignals)
    self.executeTest(self.configureMissingGetdomainnamePrototype)
    self.executeTest(self.configureMissingSrandPrototype)
    return
