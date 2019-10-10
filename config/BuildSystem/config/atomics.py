import config.base

import os

class Configure(config.base.Configure):
  def __init__(self, framework):
    config.base.Configure.__init__(self, framework)
    self.headerPrefix = 'PETSC'
    return

  def setupDependencies(self, framework):
    config.base.Configure.setupDependencies(self, framework)
    self.setCompilers = framework.require('config.setCompilers', self)
    self.libraries    = framework.require('config.libraries', self)
    return

  def configureCPURelax(self):
    ''' Definitions for cpu relax assembly instructions '''
    # Definition for cpu_relax()
    # From Linux documentation
    # cpu_relax() call can lower power consumption or yield to a hyperthreaded
    # twin processor; it also happens to serve as a compiler barrier

    # x86
    if self.checkCompile('', '__asm__ __volatile__("rep; nop" ::: "memory");'):
      self.addDefine('CPU_RELAX()','__asm__ __volatile__("rep; nop" ::: "memory")')
      return
    # PowerPC
    if self.checkCompile('','do { HMT_low; HMT_medium; __asm__ __volatile__ ("":::"memory"); } while (0)'):
      self.addDefine('CPU_RELAX()','do { HMT_low; HMT_medium; __asm__ __volatile__ ("":::"memory"); } while (0)')
      return
    elif self.checkCompile('','__asm__ __volatile__ ("":::"memory");'):
      self.addDefine('CPU_RELAX()','__asm__ __volatile__ ("":::"memory")')
      return

  # The list is not exhaustive and may not compile on all systems. If you use these macros,
  # you need to have a fallback when the macros are not defined.
  def configureMemoryBarriers(self):
    ''' Definitions for memory barrier instructions'''
    # ---- Definitions for x86_64 -----
    # General Memory Barrier
    if self.checkCompile('','__asm__ __volatile__ ("mfence":::"memory")'):
      self.addDefine('MEMORY_BARRIER()','__asm__ __volatile__ ("mfence":::"memory")')
    # Read Memory Barrier
    if self.checkCompile('','__asm__ __volatile__ ("lfence":::"memory")'):
      self.addDefine('READ_MEMORY_BARRIER()','__asm__ __volatile__ ("lfence":::"memory")')
    # Write Memory Barrier
    if self.checkCompile('','__asm__ __volatile__ ("sfence":::"memory")'):
      self.addDefine('WRITE_MEMORY_BARRIER()','__asm__ __volatile__ ("sfence":::"memory")')

    # ---- Definitions for PowerPC -----
    # General Memory Barrier
    if self.checkCompile('','__asm__ __volatile__ ("sync":::"memory")'):
      self.addDefine('MEMORY_BARRIER()','__asm__ __volatile__ ("sync":::"memory")')
    # Read Memory Barrier
    if self.checkCompile('','__asm__ __volatile__ ("lwsync":::"memory")'):
      self.addDefine('READ_MEMORY_BARRIER()','__asm__ __volatile__ ("lwsync":::"memory")')
    # Write Memory Barrier
    if self.checkCompile('','__asm__ __volatile__ ("eieio":::"memory")'):
      self.addDefine('WRITE_MEMORY_BARRIER()','__asm__ __volatile__ ("eieio":::"memory")')

    # ---- Definitions for ARM v7 -----
    # Use dmb for all memory barrier types
    if self.checkCompile('','__asm__ __volatile__ ("dmb":::"memory")'):
      self.addDefine('MEMORY_BARRIER()','__asm__ __volatile__ ("dmb":::"memory")')
      self.addDefine('READ_MEMORY_BARRIER()','__asm__ __volatile__ ("dmb":::"memory")')
      self.addDefine('WRITE_MEMORY_BARRIER()','__asm__ __volatile__ ("dmb":::"memory")')

    # ---- Definitions for ARM v8 -----
    # Ref: http://infocenter.arm.com/help/index.jsp?topic=/com.arm.doc.den0024a/CEGDBEJE.html
    # Ref: http://infocenter.arm.com/help/index.jsp?topic=/com.arm.doc.den0024a/CHDGACJD.html
    # Use "Inner shareable", which makes sense for PETSc.
    # General Memory Barrier
    if self.checkCompile('','__asm__ __volatile__ ("dmb ish":::"memory")'):
      self.addDefine('MEMORY_BARRIER()','__asm__ __volatile__ ("dmb ish":::"memory")')
    # Read Memory Barrier
    if self.checkCompile('','__asm__ __volatile__ ("dmb ishld":::"memory")'):
      self.addDefine('READ_MEMORY_BARRIER()','__asm__ __volatile__ ("dmb ishld":::"memory")')
    # Write Memory Barrier
    if self.checkCompile('','__asm__ __volatile__ ("dmb ishst":::"memory")'):
      self.addDefine('WRITE_MEMORY_BARRIER()','__asm__ __volatile__ ("dmb ishst":::"memory")')
    return

  def configure(self):
    # These are not currently used in PETSc
    # self.executeTest(self.configureCPURelax)
    # self.executeTest(self.configureMemoryBarriers)
    return
