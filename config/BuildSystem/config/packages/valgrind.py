import config.package
import os

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.functions = []
    self.includes  = ['valgrind/valgrind.h']
    self.required  = 1
    return

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    self.deps = []
    return

  def setup(self):
    config.package.Package.setup(self)
    if 'with-'+self.package+'-lib' in self.argDB:
      raise RuntimeError('It is incorrect to specify library for valgrind, please remove --with-valgrind-lib')
    return

  def getSearchDirectories(self):
    if 'with-'+self.package+'-dir' in self.argDB:
      yield self.argDB['with-valgrind-dir']
    else:
      yield('')
      yield os.path.join('/usr','local')
      yield os.path.join('/opt','local')
    return

  def configure(self):
    '''By default we look for valgrind, but do not stop if it is not found'''
    self.consistencyChecks()
    found = 0
    if self.argDB['with-'+self.package]:
      if self.cxx:
        self.libraries.pushLanguage('C++')
      else:
        self.libraries.pushLanguage(self.defaultLanguage)
      try:
        self.executeTest(self.configureLibrary)
        oldFlags = self.compilers.CPPFLAGS
        self.compilers.CPPFLAGS += ' '+self.headers.toString(self.include)
        if self.checkCompile('#include <valgrind/valgrind.h>', 'RUNNING_ON_VALGRIND;\n'):
          found = 1
          self.addDefine('HAVE_VALGRIND', 1)
        self.compilers.CPPFLAGS = oldFlags
      except:
        pass
      self.libraries.popLanguage()

    if found and config.setCompilers.Configure.isDarwin(self.log):
      self.addDefine('HAVE_VALGRIND_DARWIN', 1)
      langMap = {'C':'CC','FC':'FC','Cxx':'CXX','CUDA':'CUDAC','HIP':'HIPCC','SYCL':'SYCLCXX'}
      languages = ['C']
      if hasattr(self, 'CXX'):languages.append('Cxx')
      if hasattr(self, 'FC'):languages.append('FC')
      for language in languages:
        self.setCompilers.pushLanguage(language)
        for testFlag in ['-Wl,-bind_at_load']:
          if self.setCompilers.checkLinkerFlag(testFlag):
            # expand to CC_LINKER_FLAGS or CXX_LINKER_FLAGS or FC_LINKER_FLAGS
            linker_flag_var = langMap[language]+'_LINKER_FLAGS'
            val = getattr(self.setCompilers,linker_flag_var)
            val.append(testFlag)
            setattr(self.setCompilers,linker_flag_var,val)
        self.setCompilers.popLanguage()

    if not found and self.setCompilers.isLinux(self.log):
      self.logPrintBox('It appears you do not have valgrind installed on your system.\n\
We HIGHLY recommend you install it from www.valgrind.org\n\
Or install valgrind-devel or equivalent using your package manager.\n\
Then rerun ./configure')
    else:
      self.executeTest(self.alternateConfigureLibrary)
    return
