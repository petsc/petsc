

#!/usr/bin/env python
from __future__ import generators
import config.base
import config.package
import os
from stat import *

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.minversion         = '2'
    self.versionname        = 'MPI_VERSION'
    self.functions          = ['MPI_Init', 'MPI_Comm_create']
    self.includes           = ['mpi.h']
    liblist_mpich         = [['fmpich2.lib','fmpich2g.lib','fmpich2s.lib','mpi.lib'],
                             ['fmpich2.lib','fmpich2g.lib','mpi.lib'],['fmpich2.lib','mpich2.lib'],
                             ['libfmpich2g.a','libmpi.a'],['libfmpich.a','libmpich.a', 'libpmpich.a'],
                             ['libmpich.a', 'libpmpich.a'],
                             ['libfmpich.a','libmpich.a', 'libpmpich.a', 'libmpich.a', 'libpmpich.a', 'libpmpich.a'],
                             ['libmpich.a', 'libpmpich.a', 'libmpich.a', 'libpmpich.a', 'libpmpich.a'],
                             ['libmpich.a','librt.a','libaio.a','libsnl.a','libpthread.a'],
                             ['libmpich.a','libssl.a','libuuid.a','libpthread.a','librt.a','libdl.a'],
                             ['libmpich.a','libnsl.a','libsocket.a','librt.a','libnsl.a','libsocket.a'],
                             ['libmpich.a','libgm.a','libpthread.a']]
    liblist_lam           = [['liblamf77mpi.a','libmpi++.a','libmpi.a','liblam.a'],
                             ['liblammpi++.a','libmpi.a','liblam.a'],
                             ['liblammpio.a','libpmpi.a','liblamf77mpi.a','libmpi.a','liblam.a'],
                             ['liblammpio.a','libpmpi.a','liblamf90mpi.a','libmpi.a','liblam.a'],
                             ['liblammpio.a','libpmpi.a','libmpi.a','liblam.a'],
                             ['liblammpi++.a','libmpi.a','liblam.a'],
                             ['libmpi.a','liblam.a']]
    liblist_msmpi         = [[os.path.join('amd64','msmpifec.lib'),os.path.join('amd64','msmpi.lib')],
                             [os.path.join('i386','msmpifec.lib'),os.path.join('i386','msmpi.lib')]]
    liblist_other         = [['libmpich.a','libpthread.a'],['libmpi++.a','libmpi.a']]
    liblist_single        = [['libmpi.a'],['libmpich.a'],['mpi.lib'],['mpich2.lib'],['mpich.lib'],
                             [os.path.join('amd64','msmpi.lib')],[os.path.join('i386','msmpi.lib')]]
    self.liblist          = liblist_mpich + liblist_lam + liblist_msmpi + liblist_other + liblist_single
    # defaults to --with-mpi=yes
    self.required         = 1
    self.complex          = 1
    self.isPOE            = 0
    self.usingMPIUni      = 0
    self.shared           = 0
    # local state
    self.commf2c          = 0
    self.commc2f          = 0
    self.needBatchMPI     = 1
    self.alternativedownload = 'mpich'
    # support MPI-3 process shared memory
    self.support_mpi3_shm = 0
    # support MPI-3 non-blocking collectives
    self.support_mpi3_nbc = 0
    self.mpi_pkg_version  = ''
    return

  def setupHelp(self, help):
    config.package.Package.setupHelp(self,help)
    import nargs
    help.addArgument('MPI', '-with-mpiexec=<prog>',                              nargs.Arg(None, None, 'The utility used to launch MPI jobs'))
    help.addArgument('MPI', '-with-mpi-compilers=<bool>',                        nargs.ArgBool(None, 1, 'Try to use the MPI compilers, e.g. mpicc'))
    help.addArgument('MPI', '-known-mpi-shared-libraries=<bool>',                nargs.ArgBool(None, None, 'Indicates the MPI libraries are shared (the usual test will be skipped)'))
    return

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    self.mpich   = framework.require('config.packages.MPICH', self)
    self.openmpi = framework.require('config.packages.OpenMPI', self)
    return

  def __str__(self):
    output  = config.package.Package.__str__(self)
    return output+self.mpi_pkg_version

  def generateLibList(self, directory):
    if self.setCompilers.usedMPICompilers:
      self.liblist = []
      self.libdir  = []
    return config.package.Package.generateLibList(self,directory)

  # search many obscure locations for MPI
  def getSearchDirectories(self):
    import re
    if self.mpich.found:
      yield (self.mpich.installDir)
      raise RuntimeError('--download-mpich libraries cannot be used')
    if self.openmpi.found:
      yield (self.openmpi.installDir)
      raise RuntimeError('--download-openmpi libraries cannot be used')

    yield ''
    # Try configure package directories
    dirExp = re.compile(r'mpi(ch)?(-.*)?')
    for packageDir in self.argDB['with-packages-search-path']:
      packageDir = os.path.abspath(packageDir)
      if not os.path.isdir(packageDir):
        raise RuntimeError('Invalid package directory: '+packageDir)
      for f in os.listdir(packageDir):
        dir = os.path.join(packageDir, f)
        if not os.path.isdir(dir):
          continue
        if not dirExp.match(f):
          continue
        yield (dir)
    # Try SUSE location
    yield (os.path.abspath(os.path.join('/opt', 'mpich')))
    # Try IBM
    self.isPOE = 1
    dir = os.path.abspath(os.path.join('/usr', 'lpp', 'ppe.poe'))
    yield (os.path.abspath(os.path.join('/usr', 'lpp', 'ppe.poe')))
    self.isPOE = 0
    # Try /usr/local
    yield (os.path.abspath(os.path.join('/usr', 'local')))
    # Try /usr/local/*mpich*
    if os.path.isdir(dir):
      ls = os.listdir(dir)
      for dir in ls:
        if dir.find('mpich') >= 0:
          dir = os.path.join('/usr','local',dir)
          if os.path.isdir(dir):
            yield (dir)
    # Try ~/mpich*
    homedir = os.getenv('HOME')
    if homedir:
      ls = os.listdir(homedir)
      for dir in ls:
        if dir.find('mpich') >= 0:
          dir = os.path.join(homedir,dir)
          if os.path.isdir(dir):
            yield (dir)
    # Try MSMPI/MPICH install locations under Windows
    # ex: /cygdrive/c/Program Files/Microsoft HPC Pack 2008 SDK
    for root in ['/',os.path.join('/','cygdrive')]:
      for drive in ['c']:
        for programFiles in ['Program Files','Program Files (x86)']:
          for packageDir in ['Microsoft HPC Pack 2008 SDK','Microsoft Compute Cluster Pack','MPICH2','MPICH',os.path.join('MPICH','SDK.gcc'),os.path.join('MPICH','SDK')]:
            yield(os.path.join(root,drive,programFiles,packageDir))
    return

  def checkSharedLibrary(self):
    '''Sets flag indicating if MPI libraries are shared or not and
    determines if MPI libraries CANNOT be used by shared libraries'''
    self.executeTest(self.configureMPIEXEC)
    if self.argDB['with-batch']:
      if self.argDB['with-shared-libraries']:
        if not 'known-mpi-shared-libraries' in self.argDB:
          self.logPrintBox('***** WARNING: Cannot verify that MPI is a shared library - in\n\
batch-mode! If MPI is a static library but linked into multiple shared\n\
libraries that the application uses, sometimes compiles work -\n\
but one might get run-time errors. If you know that the MPI library is\n\
shared - run with --known-mpi-shared-libraries=1 option to remove this\n\
warning message *****')
        elif not self.argDB['known-mpi-shared-libraries']:
          raise RuntimeError('Provided MPI library is flagged as static library! If its linked\n\
into multipe shared libraries that an application uses, sometimes\n\
compiles go through - but one might get run-time errors.  Either\n\
reconfigure PETSc with --with-shared-libraries=0 or provide MPI with\n\
shared libraries and run with --known-mpi-shared-libraries=1')
      return
    try:
      self.shared = self.libraries.checkShared('#include <mpi.h>\n','MPI_Init','MPI_Initialized','MPI_Finalize',checkLink = self.checkPackageLink,libraries = self.lib, defaultArg = 'known-mpi-shared-libraries', executor = self.mpiexec)
    except RuntimeError as e:
      if self.argDB['with-shared-libraries']:
        raise RuntimeError('Shared libraries cannot be built using MPI provided.\nEither reconfigure with --with-shared-libraries=0 or rebuild MPI with shared library support')
      self.logPrint('MPI libraries cannot be used with shared libraries')
      self.shared = 0
    return

  def configureMPIEXEC(self):
    '''Checking for mpiexec'''
    if 'with-mpiexec' in self.argDB:
      self.argDB['with-mpiexec'] = os.path.expanduser(self.argDB['with-mpiexec'])
      if not self.getExecutable(self.argDB['with-mpiexec'], resultName = 'mpiexec'):
        raise RuntimeError('Invalid mpiexec specified: '+str(self.argDB['with-mpiexec']))
      return
    if self.isPOE:
      self.mpiexec = os.path.abspath(os.path.join('bin', 'mpiexec.poe'))
      return
    if self.argDB['with-batch']:
      self.mpiexec = 'Not_appropriate_for_batch_systems_You_must_use_your_batch_system_to_submit_MPI_jobs_speak_with_your_local_sys_admin'
      self.addMakeMacro('MPIEXEC',self.mpiexec)
      return
    mpiexecs = ['mpiexec -n 1', 'mpirun -n 1', 'mprun -n 1', 'mpiexec', 'mpirun', 'mprun']
    path    = []
    if 'with-mpi-dir' in self.argDB:
      path.append(os.path.join(os.path.abspath(self.argDB['with-mpi-dir']), 'bin'))
      # MPICH-NT-1.2.5 installs MPIRun.exe in mpich/mpd/bin
      path.append(os.path.join(os.path.abspath(self.argDB['with-mpi-dir']), 'mpd','bin'))
    for inc in self.include:
      path.append(os.path.join(os.path.dirname(inc), 'bin'))
      # MPICH-NT-1.2.5 installs MPIRun.exe in mpich/SDK/include/../../mpd/bin
      path.append(os.path.join(os.path.dirname(os.path.dirname(inc)),'mpd','bin'))
    for lib in self.lib:
      path.append(os.path.join(os.path.dirname(os.path.dirname(lib)), 'bin'))
    self.pushLanguage('C')
    if os.path.basename(self.getCompiler()) == 'mpicc' and os.path.dirname(self.getCompiler()):
      path.append(os.path.dirname(self.getCompiler()))
    self.popLanguage()
    if not self.getExecutable(mpiexecs, path = path, useDefaultPath = 1, resultName = 'mpiexec',setMakeMacro=0):
      if not self.getExecutable('/bin/false', path = [], useDefaultPath = 0, resultName = 'mpiexec',setMakeMacro=0):
        raise RuntimeError('Could not locate MPIEXEC - please specify --with-mpiexec option')
    self.mpiexec = self.mpiexec.replace(' -n 1','').replace(' ', '\\ ').replace('(', '\\(').replace(')', '\\)')
    if (hasattr(self, 'ompi_major_version') and int(self.ompi_major_version) >= 3):
      self.mpiexec = self.mpiexec + ' --oversubscribe'
    self.addMakeMacro('MPIEXEC', self.mpiexec)
    return

  def configureMPI2(self):
    '''Check for functions added to the interface in MPI-2'''
    oldFlags = self.compilers.CPPFLAGS
    oldLibs  = self.compilers.LIBS
    self.compilers.CPPFLAGS += ' '+self.headers.toString(self.include)
    self.compilers.LIBS = self.libraries.toString(self.lib)+' '+self.compilers.LIBS
    self.framework.saveLog()
    if self.checkLink('#include <mpi.h>\n', 'int flag;if (MPI_Finalized(&flag));\n'):
      self.haveFinalized = 1
      self.addDefine('HAVE_MPI_FINALIZED', 1)
    if self.checkLink('#include <mpi.h>\n', 'if (MPI_Allreduce(MPI_IN_PLACE,0, 1, MPI_INT, MPI_SUM, MPI_COMM_SELF));\n'):
      self.haveInPlace = 1
      self.addDefine('HAVE_MPI_IN_PLACE', 1)
    if self.checkLink('#include <mpi.h>\n', 'int count=2; int blocklens[2]={0,1}; MPI_Aint indices[2]={0,1}; MPI_Datatype old_types[2]={0,1}; MPI_Datatype *newtype = 0;\n \
                                             if (MPI_Type_create_struct(count, blocklens, indices, old_types, newtype));\n'):
      self.haveTypeCreateStruct = 1
    else:
      self.haveTypeCreateStruct = 0
      self.framework.addDefine('MPI_Type_create_struct(count,lens,displs,types,newtype)', 'MPI_Type_struct((count),(lens),(displs),(types),(newtype))')
    if self.checkLink('#include <mpi.h>\n', 'MPI_Comm_errhandler_fn * p_err_fun = 0; MPI_Errhandler * p_errhandler = 0; if (MPI_Comm_create_errhandler(p_err_fun,p_errhandler));\n'):
      self.haveCommCreateErrhandler = 1
    else:
      self.haveCommCreateErrhandler = 0
      self.framework.addDefine('MPI_Comm_create_errhandler(p_err_fun,p_errhandler)', 'MPI_Errhandler_create((p_err_fun),(p_errhandler))')
    if self.checkLink('#include <mpi.h>\n', 'if (MPI_Comm_set_errhandler(MPI_COMM_WORLD,MPI_ERRORS_RETURN));\n'):
      self.haveCommSetErrhandler = 1
    else:
      self.haveCommSetErrhandler = 0
      self.framework.addDefine('MPI_Comm_set_errhandler(comm,p_errhandler)', 'MPI_Errhandler_set((comm),(p_errhandler))')
    if self.checkLink('#include <mpi.h>\n', 'if (MPI_Reduce_local(0, 0, 0, MPI_INT, MPI_SUM));'):
      self.haveReduceLocal = 1
      self.addDefine('HAVE_MPI_REDUCE_LOCAL',1)
    if self.checkLink('#include <mpi.h>\n', 'char version[MPI_MAX_LIBRARY_VERSION_STRING];int verlen;if (MPI_Get_library_version(version,&verlen));\n'):
      self.addDefine('HAVE_MPI_GET_LIBRARY_VERSION', 1)
    self.compilers.CPPFLAGS = oldFlags
    self.compilers.LIBS = oldLibs
    self.logWrite(self.framework.restoreLog())
    return

  def configureMPI3(self):
    '''Check for functions added to the interface in MPI-3'''
    oldFlags = self.compilers.CPPFLAGS
    oldLibs  = self.compilers.LIBS
    self.compilers.CPPFLAGS += ' '+self.headers.toString(self.include)
    self.compilers.LIBS = self.libraries.toString(self.lib)+' '+self.compilers.LIBS
    self.framework.saveLog()
    if self.libraries.check(self.dlib, "MPI_Win_create"):
      self.addDefine('HAVE_MPI_WIN_CREATE',1)
    if self.checkLink('#include <mpi.h>\n', 'MPI_Comm scomm; if (MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &scomm));\n'):
      self.haveMPISharedComm = 1
      self.addDefine('HAVE_MPI_SHARED_COMM', 1)
    else: self.haveMPISharedComm = 0
    if 'HAVE_MPI_WIN_CREATE' in self.defines and 'HAVE_MPI_WIN_ALLOCATE_SHARED' in self.defines and 'HAVE_MPI_WIN_SHARED_QUERY' in self.defines:
      if (hasattr(self, 'mpich_numversion') and int(self.mpich_numversion) > 30004300) or not hasattr(self, 'mpich_numversion'):
        self.addDefine('HAVE_MPI_WIN_CREATE_FEATURE',1)
        self.addDefine('HAVE_MPI_PROCESS_SHARED_MEMORY',1)
        self.support_mpi3_shm = 1
    if self.checkLink('#include <mpi.h>\n',
                      'int send=0,recv,counts[2]={1,1},displs[2]={1,2}; MPI_Request req;\n\
                       if (MPI_Iscatter(&send,1,MPI_INT,&recv,1,MPI_INT,0,MPI_COMM_WORLD,&req));\n \
                       if (MPI_Iscatterv(&send,counts,displs,MPI_INT,&recv,1,MPI_INT,0,MPI_COMM_WORLD,&req));\n \
                       if (MPI_Igather(&send,1,MPI_INT,&recv,1,MPI_INT,0,MPI_COMM_WORLD,&req));\n \
                       if (MPI_Igatherv(&send,1,MPI_INT,&recv,counts,displs,MPI_INT,0,MPI_COMM_WORLD,&req));\n \
                       if (MPI_Iallgather(&send,1,MPI_INT,&recv,1,MPI_INT,MPI_COMM_WORLD,&req));\n \
                       if (MPI_Iallgatherv(&send,1,MPI_INT,&recv,counts,displs,MPI_INT,MPI_COMM_WORLD,&req));\n \
                       if (MPI_Ialltoall(&send,1,MPI_INT,&recv,1,MPI_INT,MPI_COMM_WORLD,&req));\n'):
      self.addDefine('HAVE_MPI_NONBLOCKING_COLLECTIVES', 1)
      self.support_mpi3_nbc = 1
    if self.checkLink('#include <mpi.h>\n',
                      'MPI_Comm distcomm; \n\
                       MPI_Request req; \n\
                       if (MPI_Dist_graph_create_adjacent(MPI_COMM_WORLD,0,0,MPI_WEIGHTS_EMPTY,0,0,MPI_WEIGHTS_EMPTY,MPI_INFO_NULL,0,&distcomm));\n\
                       if (MPI_Neighbor_alltoallv(0,0,0,MPI_INT,0,0,0,MPI_INT,distcomm));\n\
                       if (MPI_Ineighbor_alltoallv(0,0,0,MPI_INT,0,0,0,MPI_INT,distcomm,&req));\n'):
      self.addDefine('HAVE_MPI_NEIGHBORHOOD_COLLECTIVES',1)
    self.compilers.CPPFLAGS = oldFlags
    self.compilers.LIBS = oldLibs
    self.logWrite(self.framework.restoreLog())
    return

  def configureTypes(self):
    '''Checking for MPI types'''
    oldFlags = self.compilers.CPPFLAGS
    self.compilers.CPPFLAGS += ' '+self.headers.toString(self.include)
    self.framework.batchIncludeDirs.extend([self.headers.getIncludeArgument(inc) for inc in self.include])
    self.framework.addBatchLib(self.lib)
    self.compilers.CPPFLAGS = oldFlags
    return

  def configureMPITypes(self):
    '''Checking for MPI Datatype handles'''
    oldFlags = self.compilers.CPPFLAGS
    oldLibs  = self.compilers.LIBS
    self.compilers.CPPFLAGS += ' '+self.headers.toString(self.include)
    self.compilers.LIBS = self.libraries.toString(self.lib)+' '+self.compilers.LIBS
    mpitypes = [('MPI_LONG_DOUBLE', 'long-double'), ('MPI_INT64_T', 'int64_t')]
    if self.getDefaultLanguage() == 'C': mpitypes.extend([('MPI_C_DOUBLE_COMPLEX', 'c-double-complex')])
    for datatype, name in mpitypes:
      includes = '#include <stdlib.h>\n#include <mpi.h>\n'
      body     = 'int size;\nint ierr;\nMPI_Init(0,0);\nierr = MPI_Type_size('+datatype+', &size);\nif(ierr || (size == 0)) exit(1);\nMPI_Finalize();\n'
      if self.checkCompile(includes, body):
        if 'known-mpi-'+name in self.argDB:
          if int(self.argDB['known-mpi-'+name]):
            self.addDefine('HAVE_'+datatype, 1)
        elif not self.argDB['with-batch']:
          self.pushLanguage('C')
          if self.checkRun(includes, body, defaultArg = 'known-mpi-'+name):
            self.addDefine('HAVE_'+datatype, 1)
          self.popLanguage()
        else:
         self.logPrintBox('***** WARNING: Cannot determine if '+datatype+' works on your system\n\
in batch-mode! Assuming it does work. Run with --known-mpi-'+name+'=0\n\
if you know it does not work (very unlikely). Run with --known-mpi-'+name+'=1\n\
to remove this warning message *****')
         self.addDefine('HAVE_'+datatype, 1)
    self.compilers.CPPFLAGS = oldFlags
    self.compilers.LIBS = oldLibs
    return

  def alternateConfigureLibrary(self):
    '''Setup MPIUNI, our uniprocessor version of MPI'''
    self.addDefine('HAVE_MPIUNI', 1)
    self.addMakeMacro('MPI_IS_MPIUNI', 1)
    self.framework.packages.append(self)
    self.mpiexec = '${PETSC_DIR}/lib/petsc/bin/petsc-mpiexec.uni'
    self.addMakeMacro('MPIEXEC','${PETSC_DIR}/lib/petsc/bin/petsc-mpiexec.uni')
    self.addDefine('HAVE_MPI_IN_PLACE', 1)
    self.addDefine('HAVE_MPI_TYPE_DUP', 1)
    self.addDefine('HAVE_MPI_TYPE_GET_ENVELOPE', 1)
    self.framework.saveLog()
    self.framework.addDefine('MPI_Type_create_struct(count,lens,displs,types,newtype)', 'MPI_Type_struct((count),(lens),(displs),(types),(newtype))')
    self.framework.addDefine('MPI_Comm_create_errhandler(p_err_fun,p_errhandler)', 'MPI_Errhandler_create((p_err_fun),(p_errhandler))')
    self.framework.addDefine('MPI_Comm_set_errhandler(comm,p_errhandler)', 'MPI_Errhandler_set((comm),(p_errhandler))')
    self.logWrite(self.framework.restoreLog())
    if self.getDefaultLanguage == 'C': self.addDefine('HAVE_MPI_C_DOUBLE_COMPLEX', 1)
    self.commf2c = 1
    self.commc2f = 1
    self.usingMPIUni = 1
    self.version = 'PETSc MPIUNI uniprocessor MPI replacement'
    self.executeTest(self.PetscArchMPICheck)
    return

  def checkDownload(self):
    '''Check if we should download MPICH or OpenMPI'''
    if 'download-mpi' in self.argDB and self.argDB['download-mpi']:
      raise RuntimeError('Option --download-mpi does not exist! Use --download-mpich or --download-openmpi instead.')

    if self.argDB['download-mpich'] and self.argDB['download-openmpi']:
      raise RuntimeError('Cannot install more than one of OpenMPI or  MPICH-2 for a single configuration. \nUse different PETSC_ARCH if you want to be able to switch between two')
    return None

  def SGIMPICheck(self):
    '''Returns true if SGI MPI is used'''
    if self.libraries.check(self.lib, 'MPI_SGI_barrier') :
      self.logPrint('SGI MPI detected - defining MISSING_SIGTERM')
      self.addDefine('MISSING_SIGTERM', 1)
      return 1
    else:
      self.logPrint('SGI MPI test failure')
      return 0

  def CxxMPICheck(self):
    '''Make sure C++ can compile and link'''
    if not hasattr(self.compilers, 'CXX'):
      return 0
    self.libraries.pushLanguage('Cxx')
    oldFlags = self.compilers.CXXPPFLAGS
    self.compilers.CXXPPFLAGS += ' '+self.headers.toString(self.include)
    self.log.write('Checking for header mpi.h\n')
    if not self.libraries.checkCompile(includes = '#include <mpi.h>\n'):
      raise RuntimeError('C++ error! mpi.h could not be located at: '+str(self.include))
    # check if MPI_Finalize from c++ exists
    self.log.write('Checking for C++ MPI_Finalize()\n')
    if not self.libraries.check(self.lib, 'MPI_Finalize', prototype = '#include <mpi.h>', call = 'int ierr;\nierr = MPI_Finalize();', cxxMangle = 1):
      raise RuntimeError('C++ error! MPI_Finalize() could not be located!')
    self.compilers.CXXPPFLAGS = oldFlags
    self.libraries.popLanguage()
    return

  def FortranMPICheck(self):
    '''Make sure fortran include [mpif.h] and library symbols are found'''
    if not hasattr(self.compilers, 'FC'):
      return 0
    # Fortran compiler is being used - so make sure mpif.h exists
    self.libraries.pushLanguage('FC')
    oldFlags = self.compilers.FPPFLAGS
    self.compilers.FPPFLAGS += ' '+self.headers.toString(self.include)
    self.log.write('Checking for header mpif.h\n')
    if not self.libraries.checkCompile(body = '#include "mpif.h"'):
      raise RuntimeError('Fortran error! mpif.h could not be located at: '+str(self.include))
    # check if mpi_init form fortran works
    self.log.write('Checking for fortran mpi_init()\n')
    if not self.libraries.check(self.lib,'', call = '#include "mpif.h"\n       integer ierr\n       call mpi_init(ierr)'):
      raise RuntimeError('Fortran error! mpi_init() could not be located!')
    # check if mpi.mod exists
    if self.compilers.fortranIsF90:
      self.log.write('Checking for mpi.mod\n')
      if self.libraries.check(self.lib,'', call = '       use mpi\n       integer ierr,rank\n       call mpi_init(ierr)\n       call mpi_comm_rank(MPI_COMM_WORLD,rank,ierr)\n'):
        self.havef90module = 1
        self.addDefine('HAVE_MPI_F90MODULE', 1)
    self.compilers.FPPFLAGS = oldFlags
    self.libraries.popLanguage()
    return 0

  def configureIO(self):
    '''Check for the functions in MPI/IO
       - Define HAVE_MPIIO if they are present
       - Some older MPI 1 implementations are missing these'''
    oldFlags = self.compilers.CPPFLAGS
    oldLibs  = self.compilers.LIBS
    self.compilers.CPPFLAGS += ' '+self.headers.toString(self.include)
    self.compilers.LIBS = self.libraries.toString(self.lib)+' '+self.compilers.LIBS
    if not self.checkLink('#include <mpi.h>\n', 'MPI_Aint lb, extent;\nif (MPI_Type_get_extent(MPI_INT, &lb, &extent));\n'):
      self.compilers.CPPFLAGS = oldFlags
      self.compilers.LIBS = oldLibs
      return
    if not self.checkLink('#include <mpi.h>\n', 'MPI_File fh;\nvoid *buf;\nMPI_Status status;\nif (MPI_File_write_all(fh, buf, 1, MPI_INT, &status));\n'):
      self.compilers.CPPFLAGS = oldFlags
      self.compilers.LIBS = oldLibs
      return
    if not self.checkLink('#include <mpi.h>\n', 'MPI_File fh;\nvoid *buf;\nMPI_Status status;\nif (MPI_File_read_all(fh, buf, 1, MPI_INT, &status));\n'):
      self.compilers.CPPFLAGS = oldFlags
      self.compilers.LIBS = oldLibs
      return
    if not self.checkLink('#include <mpi.h>\n', 'MPI_File fh;\nMPI_Offset disp;\nMPI_Info info;\nif (MPI_File_set_view(fh, disp, MPI_INT, MPI_INT, "", info));\n'):
      self.compilers.CPPFLAGS = oldFlags
      self.compilers.LIBS = oldLibs
      return
    if not self.checkLink('#include <mpi.h>\n', 'MPI_File fh;\nMPI_Info info;\nif (MPI_File_open(MPI_COMM_SELF, "", 0, info, &fh));\n'):
      self.compilers.CPPFLAGS = oldFlags
      self.compilers.LIBS = oldLibs
      return
    if not self.checkLink('#include <mpi.h>\n', 'MPI_File fh;\nMPI_Info info;\nif (MPI_File_close(&fh));\n'):
      self.compilers.CPPFLAGS = oldFlags
      self.compilers.LIBS = oldLibs
      return
    # MSWIN has buggy MPI IO
    if 'HAVE_LIBMSMPI' in self.defines: return
    self.addDefine('HAVE_MPIIO', 1)
    self.compilers.CPPFLAGS = oldFlags
    self.compilers.LIBS = oldLibs
    return

  def checkMPICHorOpenMPI(self):
    '''Determine if MPICH_NUMVERSION or OMPI_MAJOR_VERSION exist in mpi.h
       Used for consistency checking of MPI installation at compile time'''
    import re
    HASHLINESPACE = ' *(?:\n#.*\n *)*'
    oldFlags = self.compilers.CPPFLAGS
    self.compilers.CPPFLAGS += ' '+self.headers.toString(self.include)
    for mpichpkg in ['i_mpi','mvapich2','mpich']:
      MPICHPKG = mpichpkg.upper()
      mpich_test = '#include <mpi.h>\nint mpich_ver = '+MPICHPKG+'_NUMVERSION;\n'
      if self.checkCompile(mpich_test):
        buf = self.outputPreprocess(mpich_test)
        try:
          mpich_numversion = re.compile('\nint mpich_ver ='+HASHLINESPACE+'([0-9]+)'+HASHLINESPACE+';').search(buf).group(1)
          self.addDefine('HAVE_'+MPICHPKG+'_NUMVERSION',mpich_numversion)
          self.mpi_pkg_version  = '  '+MPICHPKG+'_NUMVERSION: '+mpich_numversion+'\n'
          if mpichpkg == 'mpich': self.mpich_numversion = mpich_numversion
        except:
          self.logPrint('Unable to parse '+MPICHPKG+' version from header. Probably a buggy preprocessor')
        self.compilers.CPPFLAGS = oldFlags
        return
    openmpi_test = '#include <mpi.h>\nint ompi_major = OMPI_MAJOR_VERSION;\nint ompi_minor = OMPI_MINOR_VERSION;\nint ompi_release = OMPI_RELEASE_VERSION;\n'
    if self.checkCompile(openmpi_test):
      buf = self.outputPreprocess(openmpi_test)
      ompi_major_version = ompi_minor_version = ompi_release_version = 'unknown'
      try:
        ompi_major_version = re.compile('\nint ompi_major ='+HASHLINESPACE+'([0-9]+)'+HASHLINESPACE+';').search(buf).group(1)
        ompi_minor_version = re.compile('\nint ompi_minor ='+HASHLINESPACE+'([0-9]+)'+HASHLINESPACE+';').search(buf).group(1)
        ompi_release_version = re.compile('\nint ompi_release ='+HASHLINESPACE+'([0-9]+)'+HASHLINESPACE+';').search(buf).group(1)
        self.addDefine('HAVE_OMPI_MAJOR_VERSION',ompi_major_version)
        self.addDefine('HAVE_OMPI_MINOR_VERSION',ompi_minor_version)
        self.addDefine('HAVE_OMPI_RELEASE_VERSION',ompi_release_version)
        self.ompi_major_version = ompi_major_version
        self.mpi_pkg_version = '  OMPI_VERSION: '+ompi_major_version+'.'+ompi_minor_version+'.'+ompi_release_version+'\n'
      except:
        self.logPrint('Unable to parse OpenMPI version from header. Probably a buggy preprocessor')
    self.compilers.CPPFLAGS = oldFlags
    return

  def findMPIInc(self):
    '''Find MPI include paths from "mpicc -show" and use with CUDAC_FLAGS'''
    if not hasattr(self.compilers, 'CUDAC'): return
    import re
    output = ''
    try:
      output   = self.executeShellCommand(self.compilers.CC + ' -show', log = self.log)[0]
      compiler = output.split(' ')[0]
    except:
      pass
    argIter = iter(output.split())
    try:
      while 1:
        arg = next(argIter)
        self.logPrint( 'Checking arg '+arg, 4, 'compilers')
        m = re.match(r'^-I.*$', arg)
        if m:
          self.logPrint('Found include option: '+arg, 4, 'compilers')
          self.setCompilers.pushLanguage('CUDA')
          self.setCompilers.addCompilerFlag(arg)
          self.setCompilers.popLanguage()
          continue
    except StopIteration:
      pass
    return

  def log_print_mpi_h_line(self,buf):
    for line in buf.splitlines():
      if 'mpi.h' in line:
        self.log.write('mpi_h_line:\n'+line+'\n')
        return
    self.log.write('mpi.h not found in buf')
    return

  def PetscArchMPICheck(self):
    import os
    '''Makes sure incompatable mpi.h is not in the PETSC_ARCH/include directory'''
    build_mpi_h_dir = os.path.join(self.petscdir.dir,self.arch,'include')
    build_mpi_h = os.path.join(build_mpi_h_dir,'mpi.h')
    if os.path.isfile(build_mpi_h):
      self.log.write('mpi.h found in build dir! Checking if its a bad copy.\n')
      if self.usingMPIUni:
        raise RuntimeError('There is a copy of mpi.h in '+build_mpi_h_dir+' that will conflict with --with-mpi=0 build. do:\nrm -rf '+self.arch+' and run ./configure again\n')
      oldFlags = self.compilers.CPPFLAGS
      mpi_h_test = '#include <mpi.h>'
      # check self.include
      self.compilers.CPPFLAGS = oldFlags+' '+self.headers.toString(self.include)
      buf1 = self.outputPreprocess(mpi_h_test)
      self.log_print_mpi_h_line(buf1)
      # check build_mpi_h_dir and self.include
      self.compilers.CPPFLAGS = oldFlags+' '+self.headers.getIncludeArgument(build_mpi_h_dir)+' '+self.headers.toString(self.include)
      buf2 = self.outputPreprocess(mpi_h_test)
      self.log_print_mpi_h_line(buf2)
      if buf1 != buf2:
        raise RuntimeError('There is a copy of mpi.h in '+build_mpi_h_dir+' that is not compatible with your MPI, do:\nrm -rf '+self.arch+' and run ./configure again\n')
      self.compilers.CPPFLAGS = oldFlags
    return


  def configureLibrary(self):
    import platform
    '''Calls the regular package configureLibrary and then does an additional test needed by MPI'''
    if 'with-'+self.package+'-shared' in self.argDB:
      self.argDB['with-'+self.package] = 1
    config.package.Package.configureLibrary(self)
    self.executeTest(self.checkMPICHorOpenMPI)
    if any(x in platform.processor() for x in ['i386','x86','i86pc']) and config.setCompilers.Configure.isSolaris(self.log) and hasattr(self, 'mpich_numversion') and int(self.mpich_numversion) >= 30301300:
      # this is only needed if MPICH/HWLOC were compiled with optimization
      self.logWrite('Setting environmental variable to work around buggy HWLOC\nhttps://github.com/open-mpi/hwloc/issues/290\n')
      os.environ['HWLOC_COMPONENTS'] = '-x86'
      self.addDefine('HAVE_HWLOC_SOLARIS_BUG',1)
      self.logPrintBox('***** WARNING: This MPI implementation may have a bug in it that causes programs to hang.\n\
You may need to set the environmental variable HWLOC_COMPONENTS to -x86 to prevent such hangs. warning message *****')
    self.executeTest(self.configureMPI2)
    self.executeTest(self.configureMPI3) #depends on checkMPICHorOpenMPI for self.mpich_numversion
    self.executeTest(self.configureTypes)
    self.executeTest(self.configureMPITypes)
    self.executeTest(self.SGIMPICheck)
    self.executeTest(self.CxxMPICheck)
    self.executeTest(self.FortranMPICheck)
    self.executeTest(self.configureIO)
    self.executeTest(self.findMPIInc)
    self.executeTest(self.PetscArchMPICheck)
    funcs = '''MPI_Type_get_envelope  MPI_Type_dup MPI_Init_thread
      MPI_Iallreduce MPI_Ibarrier MPI_Finalized MPI_Exscan MPI_Reduce_scatter MPI_Reduce_scatter_block'''.split()
    found, missing = self.libraries.checkClassify(self.dlib, funcs)
    for f in found:
      self.addDefine('HAVE_' + f.upper(),1)
    for f in ['MPIX_Iallreduce', 'MPIX_Ibarrier']: # Unlikely to be found
      if self.libraries.check(self.dlib, f):
        self.addDefine('HAVE_' + f.upper(),1)

    oldFlags = self.compilers.CPPFLAGS # Disgusting save and restore
    self.compilers.CPPFLAGS += ' '+self.headers.toString(self.include)
    for combiner in ['MPI_COMBINER_DUP', 'MPI_COMBINER_CONTIGUOUS',
                     'MPI_COMBINER_NAMED']:
      if self.checkCompile('#include <mpi.h>', 'int combiner = %s;' % (combiner,)):
        self.addDefine('HAVE_' + combiner,1)
    self.compilers.CPPFLAGS = oldFlags

    if self.libraries.check(self.dlib, "MPIDI_CH3I_sock_set"):
      self.addDefine('HAVE_MPICH_CH3_SOCK', 1)
    if self.libraries.check(self.dlib, "MPIDI_CH3I_sock_fixed_nbc_progress"):
      # Indicates that this bug was fixed: http://trac.mpich.org/projects/mpich/ticket/1785
      self.addDefine('HAVE_MPICH_CH3_SOCK_FIXED_NBC_PROGRESS', 1)
