import config.package

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.version          = '6.1.0'
    self.versionname      = 'SCOTCH_VERSION.SCOTCH_RELEASE.SCOTCH_PATCHLEVEL'
    self.gitcommit        = 'v'+self.version
    self.download         = ['git://https://gitlab.inria.fr/scotch/scotch.git',
                             'https://gitlab.inria.fr/scotch/scotch/-/archive/'+self.gitcommit+'/scotch-'+self.gitcommit+'.tar.gz',
                             'http://ftp.mcs.anl.gov/pub/petsc/externalpackages/scotch-'+self.gitcommit+'.tar.gz']
    self.downloaddirnames = ['scotch','petsc-pkg-scotch']
    self.liblist          = [['libptesmumps.a','libptscotchparmetis.a','libptscotch.a','libptscotcherr.a','libesmumps.a','libscotch.a','libscotcherr.a']]
    self.functions        = ['SCOTCH_archBuild']
    self.functionsDefine  = ['SCOTCH_ParMETIS_V3_NodeND']
    self.includes         = ['ptscotch.h']
    self.hastests         = 1
    return

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    self.mpi            = framework.require('config.packages.MPI',self)
    self.mathlib        = framework.require('config.packages.mathlib',self)
    self.pthread        = framework.require('config.packages.pthread',self)
    self.zlib           = framework.require('config.packages.zlib',self)
    self.regex          = framework.require('config.packages.regex',self)
    self.deps           = [self.mpi,self.mathlib,self.regex]
    self.odeps          = [self.pthread,self.zlib]
    return

  def Install(self):
    import os

    self.log.write('Creating PTScotch '+os.path.join(os.path.join(self.packageDir,'src'),'Makefile.inc')+'\n')

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
    self.pushLanguage('C')
    g.write('CCS        = '+self.getCompiler()+'\n')
    g.write('CCP        = '+self.getCompiler()+'\n')
    g.write('CCD        = '+self.getCompiler()+'\n')

    # Building cflags/ldflags
    self.cflags = self.updatePackageCFlags(self.getCompilerFlags())+' '+self.headers.toString(self.mpi.include)
    functions = self.framework.require('config.functions', self)
    if not functions.haveFunction('FORK') and not functions.haveFunction('_PIPE'):
      raise RuntimeError('Error building PTScotch: no pipe function')
    ldflags = self.libraries.toString(self.dlib)
    if self.zlib.found:
      self.cflags = self.cflags + ' -DCOMMON_FILE_COMPRESS_GZ'
    # OSX does not have pthread_barrier_destroy
    if self.pthread.found and self.pthread.pthread_barrier:
      self.cflags = self.cflags + ' -DCOMMON_PTHREAD'
    if functions.haveFunction('_PIPE'):
      self.cflags = self.cflags + ' -D\'pipe(pfds)=_pipe(pfds,1024,0x8000)\''
    if self.libraries.add('-lrt','timer_create'): ldflags += ' -lrt'
    self.cflags = self.cflags + ' -DCOMMON_RANDOM_FIXED_SEED'
    # do not use -DSCOTCH_PTHREAD because requires MPI built for threads.
    self.cflags = self.cflags + ' -DSCOTCH_RENAME -Drestrict="'+self.compilers.cRestrict+'"'
    # this is needed on the Mac, because common2.c includes common.h which DOES NOT include mpi.h because
    # SCOTCH_PTSCOTCH is NOT defined above Mac does not know what clock_gettime() is!
    if self.setCompilers.isDarwin(self.log):
      self.cflags = self.cflags + ' -DCOMMON_TIMING_OLD'
    if self.getDefaultIndexSize() == 64:
      self.cflags = self.cflags + ' -DINTSIZE64'
    else:
      self.cflags = self.cflags + ' -DINTSIZE32'
    # Prepend SCOTCH_ for the compatibility layer with ParMETIS
    self.cflags = self.cflags + ' -DSCOTCH_METIS_PREFIX'

    g.write('CFLAGS	= '+self.cflags+'\n')
    if self.argDB['with-batch']:
      g.write('CCDFLAGS = '+self.cflags+' '+self.checkNoOptFlag()+'\n')
    g.write('LDFLAGS	= '+ldflags+'\n')
    g.write('CP         = '+self.programs.cp+'\n')
    g.write('LEX	= '+self.programs.flex+'\n')
    g.write('LN	        = ln\n')
    g.write('MKDIR      = '+self.programs.mkdir+'\n')
    g.write('MV         = '+self.programs.mv+'\n')
    g.write('RANLIB	= '+self.setCompilers.RANLIB+'\n')
    g.write('YACC	= '+self.programs.bison+' -y\n')
    g.close()

    self.popLanguage()

    if self.installNeeded(os.path.join('src','Makefile.inc')):
      try:
        self.logPrintBox('Compiling PTScotch; this may take several minutes')
        output,err,ret  = config.package.Package.executeShellCommand('cd '+os.path.join(self.packageDir,'src')+' && make clean ptesmumps esmumps', timeout=2500, log = self.log)
      except RuntimeError as e:
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
      self.logPrintBox('Installing PTScotch; this may take several minutes')
      self.installDirProvider.printSudoPasswordMessage()
      output,err,ret = config.package.Package.executeShellCommand(self.installSudo+'mkdir -p '+os.path.join(self.installDir,includeDir)+' && '+self.installSudo+'mkdir -p '+os.path.join(self.installDir,self.libdir)+' && cd '+self.packageDir+' && '+self.installSudo+'cp -f lib/*.a '+libDir+'/. && '+self.installSudo+' cp -f include/*.h '+includeDir+'/.', timeout=60, log = self.log)
      self.postInstall(output+err,os.path.join('src','Makefile.inc'))
    return self.installDir

#  def consistencyChecks(self):
#    config.package.Package.consistencyChecks(self)
#    if self.argDB['with-'+self.package]:
#     if self.libraries.rt is None:
#        raise RuntimeError('Scotch requires a realtime library (librt) with clock_gettime()')
