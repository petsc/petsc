from __future__ import generators
import config.base
import os
import re

class Configure(config.base.Configure):
  def __init__(self, framework):
    config.base.Configure.__init__(self, framework)
    self.headerPrefix = 'PETSC'
    self.substPrefix  = 'PETSC'
    self.updated      = 0
    self.strmsg       = ''
    return

  def __str__(self):
    return self.strmsg

  def setupHelp(self, help):
    import nargs
    help.addArgument('PETSc', '-with-proc-filesystem=<bool>', nargs.ArgBool(None, 1, 'Use the /proc filesystem for system statistics'))
    return

  def setupDependencies(self, framework):
    config.base.Configure.setupDependencies(self, framework)
    self.functions = framework.require('config.functions', self)
    self.functions.functions.append('getrusage')
    self.functions.functions.append('sbreak')
    self.functions.functions.append('getpagesize')
    return

  def configureMemorySize(self):
    '''Try to determine how to measure the memory usage'''

    # sbreak() is used on cray t3d/e
    if self.functions.haveFunction('sbreak'):
      self.addDefine('USE_SBREAK_FOR_SIZE', 1)
      return

    # /proc is used on Linux systems
    if self.argDB['with-proc-filesystem'] and not self.argDB['with-batch']:
      if os.path.isfile(os.path.join('/proc',str(os.getpid()),'statm')):
        self.addDefine('USE_PROC_FOR_SIZE', 1)
        try:
          fd = open(os.path.join('/proc',str(os.getpid()),'statm'))
          l  = fd.readline().split(' ')
          # make sure we can access the rss field
          if not l[1].isdigit():
            raise RuntimeError("/proc stat file has wrong format rss not integer:"+l[1])
          self.logPrint("Using /proc for PetscMemoryGetCurrentUsage()")
          return
        except:
          pass

    # getrusage() is still used on BSD systems
    if self.functions.haveFunction('getrusage') and not self.argDB['with-batch']:
      if self.functions.haveFunction('getpagesize'):
        (output,status) = self.outputRun('''#include <stdio.h>\n#include <ctype.h>\n#include <sys/times.h>\n#include <sys/types.h>\n
            #include <sys/stat.h>\n#include <sys/resource.h>\n#include <stdlib.h>''','''#define ARRAYSIZE 10000000
            int i,*m;
            struct   rusage temp1,temp2;
            double f0,f1,f2;

            if (getrusage(RUSAGE_SELF,&temp1)) {
              printf("Error calling getrusage()\\n");
              return -1;
            }
            m = malloc(ARRAYSIZE*sizeof(int));
            if (!m) {
              printf("Error calling malloc()\\n");
              return -3;
            }
            for (i=0; i<ARRAYSIZE; i++){
              m[i] = i+1;
            }

            if (getrusage(RUSAGE_SELF,&temp2)) {
              printf("Error calling getrusage()\\n");
              return -1;
            }

            f0 = ((double)(temp2.ru_maxrss-temp1.ru_maxrss))/(4.0*ARRAYSIZE);
            f1 = 1024.0 * ((double)(temp2.ru_maxrss-temp1.ru_maxrss))/(4.0*ARRAYSIZE);
            f2 = getpagesize() * ((double)(temp2.ru_maxrss-temp1.ru_maxrss))/(4.0*ARRAYSIZE);
            printf("Final value %g Initial value %g Increment %g 1K Scaled Increment %g pagesize scaled Increment %g\\n",(double)(temp2.ru_maxrss),(double)(temp1.ru_maxrss),f0,f1,f2);

            if (f1 == 0) {
              printf("getrusage() does not work\\n");
              return 0;
            }
            if (f0 > .90 && f0 < 1.1) {
              printf("uses bytes in getrusage()\\n");
              return 1;
            if (f1 > .90 && f1 < 1.1) {
              printf("uses 1024 size chunks in getrusage()\\n");
              return 2;
            } else if (f2 > .9 && f2 < 1.1) {
              printf("uses getpagesize() chunks in getrusage()\\n");
              return 3;
            }
            printf("unable to determine if uses bytes, 1024 or getpagesize() chunks in getrusage()\\n");
            return -2;''')
        if status > 0:
          if status == 2:
            self.addDefine('USE_KBYTES_FOR_SIZE',1)
          if status == 3:
            self.addDefine('USE_PAGES_FOR_SIZE',1)
        elif status == 0:
          self.delDefine('HAVE_GETRUSAGE')
          self.logPrint("getrusage() does not work (returns 0)")
        else:
          self.delDefine('HAVE_GETRUSAGE')
          self.logPrint("Unable to determine how to use getrusage() memory information")
        self.logPrint("output from getrusage()")
        self.logPrint(output)
        return

      # do not provide a way to get resident set size
      self.delDefine('HAVE_GETRUSAGE')
    return

  def configure(self):
    self.executeTest(self.configureMemorySize)
    return
