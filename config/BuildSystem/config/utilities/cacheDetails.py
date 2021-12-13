import config.base
import os
import sys
import collections

class CacheAttribute(object):
  def __init__(self, name, type_name, help_descr, default=None, min_value=0, max_value=min(sys.maxsize,2**31-1)):
    self.name      = str(name)
    self.type_name = str(type_name)
    self.help      = str(help_descr)
    self.default   = default
    self.min       = int(min_value)
    self.max       = int(max_value)

  def valid(self,val):
    return self.min <= val <= self.max

  def enum(self):
    return self.name.upper().replace('-','_')

class Configure(config.base.Configure):
  def __init__(self, framework):
    config.base.Configure.__init__(self, framework)
    self.headerPrefix = 'PETSC'
    self.substPrefix  = 'PETSC'
    self.updated      = 0
    self.strmsg       = ''
    self.attrs        = collections.OrderedDict(
      level1_dcache_linesize=CacheAttribute(
        'level1-dcache-linesize','int','Size in bytes of each line of the Level 1 data cache',
        default=32,min_value=16
      )
    )
# the next two are not currently used
#                         CacheAttribute('level1-dcache-size', 'int', 'Size in bytes of Level 1 data cache', 32768, 16),
#                         CacheAttribute('level1-dcache-assoc', 'int', 'Associativity of the Level 1 data cache, 0 for full associative', 2, 0)]
    return

  def __str__(self):
    return self.strmsg

  def setupHelp(self, help):
    import nargs
    for a in self.attrs.values():
      help.addArgument('PETSc', '-known-'+a.name+'=<'+a.type_name+'>', nargs.ArgInt(None, None, a.help, min=a.min, max=a.max))
    return

  def setupDependencies(self, framework):
    config.base.Configure.setupDependencies(self, framework)
    self.compilers    = framework.require('config.compilers', self)
    self.setCompilers = framework.require('config.setCompilers', self)
    self.headers      = framework.require('config.headers', self)
    return

  def L1CacheLineSizeMethods(self,default_val):
    var       = 'level1_dcache_linesize'
    func_name = 'getconf_{}'.format(var)
    if self.setCompilers.isDarwin(self.log):
      yield (
        func_name,
        '\n'.join([
          '#include <sys/sysctl.h>',
          'int64_t {}() {{'.format(func_name),
          '  int64_t linesize = {};'.format(default_val),
          '  size_t  size = sizeof(linesize);',
          '  int     ret = sysctlbyname("hw.cachelinesize", &linesize, &size, NULL, 0);',
          '  return  ret ? {} : linesize;'.format(default_val),
          '}'
        ])
      )
    # On some systems, maybe just with glibc, sysconf can provide this stuff
    yield (
      func_name,
      '\n'.join([
        '#include <unistd.h>',
        'long {}() {{'.format(func_name),
        '  long val = sysconf(_SC_{});'.format(var.upper()),
        '  return val >= 0 ? val : {};'.format(default_val),
        '}'
      ])
    )
    # A total hack since this will compile with any C compiler, but only return useful
    # results when the getconf program is available
    yield (
      func_name,
      '\n'.join([
        '#include <stdio.h>',
        'long {}() {{'.format(func_name),
        '  long val = -1;',
        '  FILE *f  = popen("getconf {}", "r");'.format(var.lower()),
        '  fscanf(f, "%ld", &val);',
        '  pclose(f);',
        '  return val >= 0 ? val : {};'.format(default_val),
        '}'
      ])
    )
    return

  def discoverL1CacheLineSize(self,attr):
    """
    Try to determine the L1CacheLineSize dynamically, if not possible returns the default value
    """
    filename       = 'conftestval'
    main_body_base = '\n'.join([
      '  FILE *output = fopen("{}", "w");'.format(filename),
      '  if (!output) return 1;',
      # note the '{func_name}', this is filled out below
      '  fprintf(output, "%ld", (long){func_name}());',
      '  fclose(output);'
    ])
    with self.Language('C'):
      for func_name,includes in self.L1CacheLineSizeMethods(attr.default):
        if not self.checkCompile(includes=includes,body=func_name+'();'):
          continue

        main_includes = '\n'.join(['#include <stdio.h>',includes])
        main_body     = main_body_base.format(func_name=func_name)
        if self.checkRun(includes=main_includes,body=main_body) and os.path.exists(filename):
          with open(filename,"r") as f:
            val = int(f.read())
          os.remove(filename)
          if attr.valid(val):
            return val
          self.log.write('Cannot use value returned for {}: {}, continuing\n'.format(attr.enum(),val))
    return attr.default

  def configureL1CacheLineSize(self):
    """
    Try to determine the size (in bytes) of an L1 cacheline. On success defines the
    variable PETSC_LEVEL1_DCACHE_LINESIZE to the determined value.
    """
    attr        = self.attrs['level1_dcache_linesize']
    argdb_val   = self.argDB.get('known-'+attr.name)
    if argdb_val is not None:
      val = int(argdb_val)
    elif self.argDB['with-batch']:
      self.log.write('Skipping determination of {} in batch mode, using default {}\n'.format(attr.enum(),attr.default))
      val = attr.default
    else:
      val = self.discoverL1CacheLineSize(attr)
    self.addDefine(attr.enum(),val)
    return

  def configure(self):
    self.executeTest(self.configureL1CacheLineSize)
    return
