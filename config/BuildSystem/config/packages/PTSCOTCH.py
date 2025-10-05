import config.package

class Configure(config.package.CMakePackage):
  def __init__(self, framework):
    config.package.CMakePackage.__init__(self, framework)
    self.version          = '7.0.10'
    self.versionname      = 'SCOTCH_VERSION.SCOTCH_RELEASE.SCOTCH_PATCHLEVEL'
    self.gitcommit        = 'v'+self.version
    self.download         = ['git://https://gitlab.inria.fr/scotch/scotch.git',
                             'https://gitlab.inria.fr/scotch/scotch/-/archive/'+self.gitcommit+'/scotch-'+self.gitcommit+'.tar.gz',
                             'https://web.cels.anl.gov/projects/petsc/download/externalpackages/scotch-'+self.gitcommit+'.tar.gz']
    self.downloaddirnames = ['scotch','petsc-pkg-scotch']
    self.liblist          = [['libptesmumps.a','libptscotchparmetisv3.a','libptscotch.a','libptscotcherr.a','libesmumps.a','libscotch.a','libscotcherr.a'],
                             ['libptesmumps.a','libptscotchparmetis.a','libptscotch.a','libptscotcherr.a','libesmumps.a','libscotch.a','libscotcherr.a']]
    self.functions        = ['SCOTCH_archBuild']
    self.functionsDefine  = ['SCOTCH_ParMETIS_V3_NodeND']
    self.includes         = ['ptscotch.h']
    self.hastests         = 1
    return

  def setupDependencies(self, framework):
    config.package.CMakePackage.setupDependencies(self, framework)
    self.mpi            = framework.require('config.packages.MPI',self)
    self.mathlib        = framework.require('config.packages.mathlib',self)
    self.bison          = framework.require('config.packages.Bison',self)
    self.deps           = [self.mpi, self.mathlib]
    self.pthread        = framework.require('config.packages.pthread',self)
    self.zlib           = framework.require('config.packages.zlib',self)
    self.odeps          = [self.bison, self.pthread, self.zlib]
    return

  def formCMakeConfigureArgs(self):
    args = config.package.CMakePackage.formCMakeConfigureArgs(self)

    if not hasattr(self.programs, 'flex'): raise RuntimeError('PTScotch needs flex installed')
    args.append('-DFLEX_EXECUTABLE:STRING="'+self.programs.flex+'"')

    if not self.bison.found or not self.bison.haveBison3plus: raise RuntimeError('PTScotch needs Bison version 3.0 or above, use --download-bison')
    args.append('-DBISON_EXECUTABLE:STRING="'+self.bison.bison+'"')

    args = self.rmArgsStartsWith(args, '-DCMAKE_C_FLAGS')
    args.append('-DCMAKE_C_FLAGS:STRING="'+self.updatePackageCFlags(self.getCompilerFlags())+' -USCOTCH_PTHREAD"')

    args.append('-DINTSIZE:STRING='+ ('64' if self.getDefaultIndexSize() == 64 else '32'))

    args.append('-DINSTALL_METIS_HEADERS:BOOL=OFF')
    args.append('-DSCOTCH_METIS_PREFIX:BOOL=ON')

    if self.zlib.found:
      args.append('-DCOMMON_FILE_COMPRESS_GZ:BOOL=ON')
    else:
      args.append('-DCOMMON_FILE_COMPRESS_GZ:BOOL=OFF')

    if self.pthread.found:
      if self.pthread.pthread_barrier:
        args.append('-DCOMMON_PTHREAD_BARRIER:BOOL=ON')
      else:
        args.append('-DCOMMON_PTHREAD_BARRIER:BOOL=OFF') # macOS does not have pthread_barrier_destroy

    if self.setCompilers.isDarwin(self.log):
      args.append('-DCOMMON_TIMING_OLD:BOOL=ON')

    args.append('-DBUILD_FORTRAN:BOOL=OFF')
    args.append('-DENABLE_TESTS:BOOL=OFF')
    return args
