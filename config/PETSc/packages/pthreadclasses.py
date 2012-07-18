import PETSc.package
import os

class Configure(PETSc.package.NewPackage):
  def __init__(self, framework):
    PETSc.package.NewPackage.__init__(self, framework)
    self.functions         = 0
    self.includes          = 0
    self.liblist           = 0
    self.complex           = 1
    return

  def setupDependencies(self, framework):
    PETSc.package.NewPackage.setupDependencies(self, framework)
    self.pthread = framework.require('PETSc.packages.pthread',self)
    self.deps    = []
    return

  def configureLibrary(self):
    if not self.pthread.found:
       raise RuntimeError('Pthreads not found, pthread classes needs pthreads to run')
    self.found = 1
    self.framework.packages.append(self)
    if self.checkCompile('__thread int a;\n',''):
      self.addDefine('PTHREAD_LOCAL','__thread')
    elif self.checkCompile('__declspec(thread) int i;\n',''):
      self.addDefine('PTHREAD_LOCAL','__declspec(thread)')

    # Definition for cpu_relax()
    # From Linux documentation
    # cpu_relax() call can lower power consumption or yield to a hyperthreaded
    # twin processor; it also happens to serve as a compiler barrier
    # x86
    if self.checkCompile('', 'asm volatile("rep; nop" ::: "memory");'):
      self.addDefine('CPURelax()','asm volatile("rep; nop" ::: "memory")')
      return
    # PowerPC
    if self.checkCompile('','do { HMT_low; HMT_medium; barrier(); } while (0)'):
      self.addDefine('CPURelax()','do {HMT_low; HMT_medium; barrier(); } while(0)')
      return
    elif self.checkCompile('','barrier();'):
      self.addDefine('CPURelax()','barrier()')
      return
    self.addDefine('CPURelax()','')
    return


