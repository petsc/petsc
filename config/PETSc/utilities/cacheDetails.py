import config.base
import os
import sys
import string

class CacheAttribute(object):
  def __init__(self, name, keyword, help, default=None, min=0, max=min(sys.maxint,2**31-1)):
    self.name = name
    self.help = help
    self.keyword = keyword
    self.default = default
    self.min = min
    self.max = max
  def valid(self,val):
    return self.min <= val <= self.max
  def sanitize(self,val):
    if self.valid(val):
      return val
    else:
      return self.default
  def enum(self):
    return self.name.upper().replace('-','_')

class Configure(config.base.Configure):
  def __init__(self, framework):
    config.base.Configure.__init__(self, framework)
    self.headerPrefix = 'PETSC'
    self.substPrefix  = 'PETSC'
    self.updated      = 0
    self.strmsg       = ''
    self.attrs        = [CacheAttribute('level1-dcache-size', 'int', 'Size in bytes of Level 1 data cache', 32768, 16),
                         CacheAttribute('level1-dcache-linesize', 'int', 'Size in bytes of each line of the Level 1 data cache', 32, 16),
                         CacheAttribute('level1-dcache-assoc', 'int', 'Associativity of the Level 1 data cache, 0 for full associative', 2, 0)]
    self.method       = None
    return

  def __str__(self):
    return self.strmsg

  def setupHelp(self, help):
    import nargs
    for a in self.attrs:
      help.addArgument('PETSc', '-known-'+a.name+'=<'+a.keyword+'>', nargs.ArgInt(None, None, a.help, min=a.min, max=a.max))
    return

  def setupDependencies(self, framework):
    config.base.Configure.setupDependencies(self, framework)
    self.compilers = framework.require('config.compilers', self)
    self.headers   = framework.require('config.headers', self)
    return

  def getconfFunction(self, a):
    VAR = a.enum()
    funcname = 'getconf_' + VAR
    sanitize = '('+str(a.min)+' <= val && val <= '+str(a.max)+') ? val : '+str(a.default)
    methods = [
      # On some systems, maybe just with glibc, sysconf can provide this stuff
      '#include <unistd.h>\nlong '+funcname+'() { long val = sysconf(_SC_'+VAR+'); return '+sanitize+'; }\n'
      ,
      # A total hack since this will compile with any C compiler, but only return useful results when the getconf program is available
      '#include <stdio.h>\nlong '+funcname+'() { long val=-1; FILE *f = popen("getconf '+VAR+'","r"); fscanf(f,"%ld",&val); pclose(f); return '+sanitize+'; }\n'
      ,
      # Fallback that just returns the default, guaranteed to compile
      'long '+funcname+'() { return '+str(a.default)+'; }\n'
      ]
    if self.method is None:         # Determine which method of finding configuration variables, only runs the first time around
      self.pushLanguage('C')
      for m in range(len(methods)):
        d = methods[m]
        if self.checkCompile(d,''):
          self.method = m
          break
      self.popLanguage()
    if self.method is None:
      raise RuntimeError("The C compiler does not work")
    return (funcname,methods[self.method])

  def configureCacheDetails(self):
    '''Try to determine the size and associativity of the cache.'''
    for a in self.attrs:
      arg = 'known-' + a.name
      (fname, source) = self.getconfFunction(a)
      if arg in self.framework.argDB:
        val = self.framework.argDB[arg]
      elif self.framework.argDB['with-batch']:
        body = 'freopen("/dev/null","w",stderr);\n' + 'fprintf(output,"  \'--'+arg+'=%ld\',\\n",'+fname+'());'
        self.framework.addBatchInclude(source)
        self.framework.addBatchBody(body)
        val = a.default
      else:
        filename = 'conftestval'
        includes = '#include <stdio.h>\n'
        body = 'FILE *output = fopen("'+filename+'","w"); if (!output) return 1; fprintf(output,"%ld",'+fname+'()); fclose(output);'
        self.pushLanguage('C')
        if self.checkRun(includes+source, body) and os.path.exists(filename):
          f = open(filename)
          val = int(f.read())
          if not a.valid(val):
            self.framework.log.write('Cannot use value returned for '+str(a.enum())+': '+str(val)+'\n')
          f.close()
        else:
          self.framework.log.write('Could not determine '+str(a.enum())+', using default '+str(a.default)+'\n')
          val = a.default
        self.popLanguage()
      self.addDefine(a.enum(), a.sanitize(val))

  def configure(self):
    self.executeTest(self.configureCacheDetails)
    return
