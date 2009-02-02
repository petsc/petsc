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
    help.addArgument('PETSc', '-with-proc-filesystem=<yes or no>', nargs.ArgBool(None, 1, 'Use the /proc filesystem for system statistics'))    
    return

  def setupDependencies(self, framework):
    config.base.Configure.setupDependencies(self, framework)
    self.functions = framework.require('config.functions', self)
    self.functions.functions.append('getrusage')
    self.functions.functions.append('sbreak')
    self.functions.functions.append('getpagesize')
    self.functions.functions.append('task_info')            
    return

  def configureMemorySize(self):
    '''Try to determine how to measure the memory usage'''

    # sbreak() is used on cray t3d/e
    if self.functions.haveFunction('sbreak'):
      self.addDefine('USE_SBREAK_FOR_SIZE', 1)
      return
    
    # /proc is used on Linux systems
    if self.argDB['with-proc-filesystem']:
      if os.path.isfile(os.path.join('/proc',str(os.getpid()),'statm')):
        self.addDefine('USE_PROC_FOR_SIZE', 1)
        try:
          fd = open(os.path.join('/proc',str(os.getpid()),'statm'))
          l  = fd.readline().split(' ')
          # make sure we can access the rss field
          if not l[1].isdigit():
            raise RuntimeError("/proc stat file has wrong format rss not integer:"+l[1])
          self.framework.logPrint("Using /proc for PetscMemoryGetCurrentUsage()")
          return
        except:
          pass
      
    # task_info() is  used on Mach and darwin (MacOS X) systems
    if self.functions.haveFunction('task_info') and not self.framework.argDB['with-batch']:
      (output,status) = self.outputRun('#include <mach/mach.h>\n#include <stdlib.h>\n#include <stdio.h>\n','''#define  ARRAYSIZE 10000000
          int *m,i;
          unsigned int count;
          task_basic_info_data_t ti1,ti2;
          double ratio;

          if (task_info(mach_task_self(), TASK_BASIC_INFO, (task_info_t)&ti1,&count) != KERN_SUCCESS) {
            printf("failed calling task_info()");
            return 1;
          }
          printf("original resident size %g \\n",(double)ti1.resident_size);
          m = malloc(ARRAYSIZE*sizeof(int));
          if (!m) {
            printf("Error calling malloc()\\n");
            return 2;
          }
          for (i=0; i<ARRAYSIZE; i++){
            m[i] = i+1;
          }
          if (task_info(mach_task_self(), TASK_BASIC_INFO, (task_info_t)&ti2,&count) != KERN_SUCCESS) {
            printf("failed calling task_info()");
            return 3;
          }
          printf("final resident size %g \\n",(double)ti2.resident_size);
          ratio = ((double)ti2.resident_size - ti1.resident_size)/(ARRAYSIZE*sizeof(int));
          printf("Should be near 1.0 %g \\n",ratio);
          if (ratio > .95 && ratio < 1.05) {
            printf("task_info() returned a reasonable resident size\\n");
            return 0;
          }
          return 4;''')      
      if status == 0:
        self.framework.logPrint("Using task_info() for PetscMemoryGetCurrentUsage()")
        return
      self.delDefine('HAVE_TASK_INFO')
      self.framework.logPrint("task_info() does not work\n"+output)
      
    # getrusage() is still used on BSD systems
    if self.functions.haveFunction('getrusage') and not self.framework.argDB['with-batch']:
      if self.functions.haveFunction('getpagesize'):
        (output,status) = self.outputRun('''#include <stdio.h>\n#include <ctype.h>\n#include <sys/times.h>\n#include <sys/types.h>\n
            #include <sys/stat.h>\n#include <sys/resource.h>\n#include <stdlib.h>''','''#define ARRAYSIZE 10000000
            int i,*m;
            struct   rusage temp1,temp2;
            double f1,f2;

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

            f1 = 1024.0 * ((double)(temp2.ru_maxrss-temp1.ru_maxrss))/(4.0*ARRAYSIZE);
            f2 = getpagesize() * ((double)(temp2.ru_maxrss-temp1.ru_maxrss))/(4.0*ARRAYSIZE);
            printf("Final value %g Initial value %g 1K Scaled Increment %g pagesize scaled Increment %g\\n",(double)(temp2.ru_maxrss),(double)(temp1.ru_maxrss),f1,f2);

            if (f1 == 0) {
              printf("getrusage() does not work\\n");
              return 0;
            }
            if (f1 > .90 && f1 < 1.1) {
              printf("uses 1024 size chunks in getrusage()\\n");
              return 1;
            } else if (f2 > .9 && f2 < 1.1) {
              printf("uses getpagesize() chunks in getrusage()\\n");
              return 2;
            }
            printf("unable to determine if uses 1024 or getpagesize() chunks in getrusage()\\n");
            return -2;''')
        if status > 0:
          if status == 1:
            self.addDefine('USE_KBYTES_FOR_SIZE',1)
        elif status == 0:
          self.delDefine('HAVE_GETRUSAGE')
          self.framework.logPrint("getrusage() does not work (returns 0)")
        else:
          self.delDefine('HAVE_GETRUSAGE')
          self.framework.logPrint("Unable to determine how to use getrusage() memory information")
        self.framework.logPrint("output from getrusage()")
        self.framework.logPrint(output)
        return
      
      # do not provide a way to get resident set size
      self.delDefine('HAVE_GETRUSAGE')
    return

  def configure(self):
    self.executeTest(self.configureMemorySize)
    return
