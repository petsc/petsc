import PETSc.package

class Configure(PETSc.package.NewPackage):
  def __init__(self, framework):
    PETSc.package.NewPackage.__init__(self, framework)
    #'https://gforge.inria.fr/frs/download.php/28978/scotch_5.1.12b_esmumps.tar.gz'
    self.download     = ['http://ftp.mcs.anl.gov/pub/petsc/externalpackages/scotch_5.1.12b_esmumps-p1.tar.gz']
    self.downloadfilename = 'scotch'
    self.liblist      = [['libptesmumps.a', 'libptscotch.a','libptscotcherr.a']]
    self.functions    = ['SCOTCH_archBuild']
    self.includes     = ['ptscotch.h']
    self.requires32bitint = 0
    self.complex      = 1
    self.needsMath    = 1
    return

  def setupDependencies(self, framework):
    PETSc.package.NewPackage.setupDependencies(self, framework)
    self.deps = [self.mpi]
    return

  def Install(self):
    import os

    self.framework.log.write('Creating PTScotch '+os.path.join(os.path.join(self.packageDir,'src'),'Makefile.inc')+'\n')

    self.programs.getExecutable('bison',   getFullPath = 1)
    if not hasattr(self.programs, 'bison'): raise RuntimeError('PTScotch needs bison installed')
    self.programs.getExecutable('flex',   getFullPath = 1)
    if not hasattr(self.programs, 'flex'): raise RuntimeError('PTScotch needs flex installed')

    g = open(os.path.join(self.packageDir,'src','Makefile.inc'),'w')

    g.write('EXE	=\n')
    g.write('LIB        = .'+self.setCompilers.AR_LIB_SUFFIX+'\n')
    g.write('OBJ	= .o\n')
    g.write('\n')
    g.write('MAKE	= make\n')

    g.write('AR	        = '+self.setCompilers.AR+'\n')
    g.write('ARFLAGS	= '+self.setCompilers.AR_FLAGS+'\n')
    g.write('CAT	= cat\n')
    self.setCompilers.pushLanguage('C')
    g.write('CCS        = '+self.setCompilers.getCompiler()+'\n')
    g.write('CCP        = '+self.setCompilers.getCompiler()+'\n')
    g.write('CCD        = '+self.setCompilers.getCompiler()+'\n')

    # Building cflags
    self.cflags = self.setCompilers.getCompilerFlags()
    if self.libraries.add('-lz','gzwrite'):
      self.cflags = self.cflags + ' -DCOMMON_FILE_COMPRESS_GZ'
    self.cflags = self.cflags + ' -DCOMMON_PTHREAD -DCOMMON_RANDOM_FIXED_SEED'
    # do not use -DSCOTCH_PTHREAD because requires MPI built for threads.
    self.cflags = self.cflags + ' -DSCOTCH_RENAME -Drestrict="" '
    # this is needed on the Mac, because common2.c includes common.h which DOES NOT include mpi.h because
    # SCOTCH_PTSCOTCH is NOT defined above Mac does not know what clock_gettime() is!
    if self.setCompilers.isDarwin():
      self.cflags = self.cflags + ' -DCOMMON_TIMING_OLD'

    if self.libraryOptions.integerSize == 64:
      self.cflags = self.cflags + ' -DINTSIZE64'
    else:
      self.cflags = self.cflags + ' -DINTSIZE32'
    g.write('CFLAGS	= '+self.cflags+'\n')

    self.setCompilers.popLanguage()
    ldflags = ''
    if self.libraries.add('-lz','gzwrite'): ldflags += '-lz'
    if self.libraries.add('-lm','sin'): ldflags += ' -lm'
    if self.libraries.add('-lrt','timer_create'): ldflags += ' -lrt'
    g.write('LDFLAGS	= '+ldflags+'\n')
    g.write('CP         = '+self.programs.cp+'\n')
    g.write('LEX	= '+self.programs.flex+'\n')
    g.write('LN	        = ln\n')
    g.write('MKDIR      = '+self.programs.mkdir+'\n')
    g.write('MV         = '+self.programs.mv+'\n')
    g.write('RANLIB	= '+self.setCompilers.RANLIB+'\n')
    g.write('YACC	= '+self.programs.bison+' -y\n')
    g.close()

    if self.installNeeded(os.path.join('src','Makefile.inc')):
      try:
        self.logPrintBox('Compiling PTScotch; this may take several minutes')
#
#    If desired one can have this build Scotch as well as PTScoth as indicated here
#        output,err,ret  = PETSc.package.NewPackage.executeShellCommand('cd '+os.path.join(self.packageDir,'src')+' && make clean scotch ptscotch', timeout=2500, log = self.framework.log)
#
        if self.mpi.found:
          output,err,ret  = PETSc.package.NewPackage.executeShellCommand('cd '+os.path.join(self.packageDir,'src')+' && make clean ptscotch', timeout=2500, log = self.framework.log)
        else:
          output,err,ret  = PETSc.package.NewPackage.executeShellCommand('cd '+os.path.join(self.packageDir,'src')+' && make clean scotch', timeout=2500, log = self.framework.log)
      except RuntimeError, e:
        raise RuntimeError('Error running make on PTScotch: '+str(e))

      #Scotch has a file identical to one in ParMETIS, remove it so ParMETIS will not use it by mistake
      try: # PTScotch installs parmetis.h by default, we need to remove it so it does not conflict with the ParMETIS native copy
        os.unlink(os.path.join(self.packageDir,'include','parmetis.h'))
      except: pass
      try: # This would only be produced if "make scotch" was invoked, but try to remove it anyway in case someone was messing with it
        os.unlink(os.path.join(self.packageDir,'include','metis.h'))
      except: pass

      libDir     = os.path.join(self.installDir, self.libdir)
      includeDir = os.path.join(self.installDir, self.includedir)
      output,err,ret = PETSc.package.NewPackage.executeShellCommand('cd '+self.packageDir+' && cp -f lib/*.a '+libDir+'/. && cp -f include/*.h '+includeDir+'/.', timeout=2500, log = self.framework.log)
      self.postInstall(output+err,os.path.join('src','Makefile.inc'))
    return self.installDir

#  def consistencyChecks(self):
#    PETSc.package.NewPackage.consistencyChecks(self)
#    if self.framework.argDB['with-'+self.package]:
#     if self.libraries.rt is None:
#        raise RuntimeError('Scotch requires a realtime library (librt) with clock_gettime()')
