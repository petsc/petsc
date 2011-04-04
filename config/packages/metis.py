#!/usr/bin/env python
import config.package

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.download          = ['http://glaros.dtc.umn.edu/gkhome/fetch/sw/metis/metis-4.0.3.tar.gz']
    self.functions         = ['METIS_mCPartGraphKway']
    self.includes          = ['metis.h']
    self.liblist           = [['libmetis.a']]
    self.needsMath         = 1
    self.complex           = 1
    self.worksonWindows    = 1
    self.downloadonWindows = 1
    return

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    return

  def Install(self):
    import os
    import sys

    makeinc              = os.path.join(self.packageDir, 'make.inc')
    installmakeinc       = os.path.join(self.confDir, 'Metis')
    metisconfigheader    = os.path.join(self.packageDir, 'METISLib', 'configureheader.h')

    # Configure Metis
    g = open(makeinc,'w')
    g.write('SHELL          = '+self.programs.SHELL+'\n')
    g.write('CP             = '+self.programs.cp+'\n')
    g.write('RM             = '+self.programs.RM+'\n')
    g.write('MKDIR          = '+self.programs.mkdir+'\n')

    g.write('AR             = '+self.setCompilers.AR+'\n')
    g.write('ARFLAGS        = '+self.setCompilers.AR_FLAGS+'\n')
    g.write('AR_LIB_SUFFIX  = '+self.setCompilers.AR_LIB_SUFFIX+'\n')
    g.write('RANLIB         = '+self.setCompilers.RANLIB+'\n')

    g.write('METIS_ROOT     = '+self.packageDir+'\n')
    g.write('PREFIX         = '+self.installDir+'\n')
    g.write('METISLIB       = $(METIS_ROOT)/libmetis.$(AR_LIB_SUFFIX)\n')

    self.setCompilers.pushLanguage('C')
    cflags = self.setCompilers.getCompilerFlags().replace('-Wall','').replace('-Wshadow','')
    cflags += ' ' + self.headers.toString(self.mpi.include)+' '+self.headers.toString('.')

    g.write('CC             = '+self.setCompilers.getCompiler()+'\n')
    g.write('OPTFLAGS       = '+cflags+'\n')
    # parmetis uses defaut 'make' targets, and this uses TARGET_ARCH var. If this var
    # is set incorrectly in user env - build breaks.
    g.write('TARGET_ARCH    = \n')

    self.setCompilers.popLanguage()
    g.close()

    if self.installNeeded('make.inc'):    # Now compile & install
      self.framework.outputHeader(metisconfigheader, prefix = 'METIS')
      try:
        self.logPrintBox('Compiling & installing Metis; this may take several minutes')
        output,err,ret  = config.package.Package.executeShellCommand('cd '+self.packageDir+' && make clean && make library && make minstall && make clean', timeout=2500, log = self.framework.log)
      except RuntimeError, e:
        raise RuntimeError('Error running make on Metis: '+str(e))
      self.postInstall(output+err, 'make.inc')
    return self.installDir
