import config.base

class Configure(config.base.Configure):
    def __init__(self, framework):
        config.base.Configure.__init__(self, framework)

    def setupDependencies(self, framework):
        config.base.Configure.setupDependencies(self, framework)
        self.compilers = framework.require('config.compilers', self)
        self.functions = framework.require('config.functions', self)
        self.headers   = framework.require('config.headers', self)

    def configureFeatureTestMacros(self):
        '''Checks if certain feature test macros are support'''
        if self.checkCompile('#define _POSIX_C_SOURCE 200112L\n#include <sysctl.h>',''):
            self.addDefine('_POSIX_C_SOURCE_200112L', '1')
        if self.checkCompile('#define _BSD_SOURCE\n#include<stdlib.h>',''):
            self.addDefine('_BSD_SOURCE', '1')
        if self.checkCompile('#define _DEFAULT_SOURCE\n#include<stdlib.h>',''):
            self.addDefine('_DEFAULT_SOURCE', '1')
        if self.checkCompile('#define _GNU_SOURCE\n#include <sched.h>','cpu_set_t mset;\nCPU_ZERO(&mset)'):
            self.addDefine('_GNU_SOURCE', '1')
        if self.checkCompile('#define _GNU_SOURCE\n#include <stdlib.h>\n#include <dlfcn.h>','Dl_info info;\nif (dladdr(*(void **)&exit, &info) == 0) return 1;\n'):
            self.addDefine('_GNU_SOURCE', '1')

    def configure(self):
        self.executeTest(self.configureFeatureTestMacros)
