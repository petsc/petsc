/*

   This file defines part of the initialization of PETSc

  This file uses regular malloc and free because it cannot be known
  what malloc is being used until it has already processed the input.
*/

#include <petscsys.h>        /*I  "petscsys.h"   I*/
#include <petsc/private/petscimpl.h>
#include <petscvalgrind.h>
#include <petscviewer.h>
#if defined(PETSC_USE_LOG)
PETSC_INTERN PetscErrorCode PetscLogInitialize(void);
#endif

#if defined(PETSC_HAVE_SYS_SYSINFO_H)
#include <sys/sysinfo.h>
#endif
#if defined(PETSC_HAVE_UNISTD_H)
#include <unistd.h>
#endif
#if defined(PETSC_HAVE_CUDA)
#include <cuda_runtime.h>
#include <petsccublas.h>
#if defined(PETSC_HAVE_OMPI_MAJOR_VERSION)
#include "mpi-ext.h" /* Needed for OpenMPI CUDA-aware check */
#endif
#endif

#if defined(PETSC_HAVE_VIENNACL)
PETSC_EXTERN PetscErrorCode PetscViennaCLInit();
#endif

/* ------------------------Nasty global variables -------------------------------*/
/*
     Indicates if PETSc started up MPI, or it was
   already started before PETSc was initialized.
*/
PetscBool   PetscBeganMPI         = PETSC_FALSE;
PetscBool   PetscErrorHandlingInitialized = PETSC_FALSE;
PetscBool   PetscInitializeCalled = PETSC_FALSE;
PetscBool   PetscFinalizeCalled   = PETSC_FALSE;
PetscBool   PetscCUDAInitialized  = PETSC_FALSE;

PetscMPIInt PetscGlobalRank       = -1;
PetscMPIInt PetscGlobalSize       = -1;

PetscBool   use_gpu_aware_mpi     = PETSC_TRUE;

#if defined(PETSC_HAVE_COMPLEX)
#if defined(PETSC_COMPLEX_INSTANTIATE)
template <> class std::complex<double>; /* instantiate complex template class */
#endif
#if !defined(PETSC_HAVE_MPI_C_DOUBLE_COMPLEX)
MPI_Datatype MPIU_C_DOUBLE_COMPLEX;
MPI_Datatype MPIU_C_COMPLEX;
#endif

/*MC
   PETSC_i - the imaginary number i

   Synopsis:
   #include <petscsys.h>
   PetscComplex PETSC_i;

   Level: beginner

   Note:
   Complex numbers are automatically available if PETSc located a working complex implementation

.seealso: PetscRealPart(), PetscImaginaryPart(), PetscRealPartComplex(), PetscImaginaryPartComplex()
M*/
PetscComplex PETSC_i;
#endif
#if defined(PETSC_USE_REAL___FLOAT128)
MPI_Datatype MPIU___FLOAT128 = 0;
#if defined(PETSC_HAVE_COMPLEX)
MPI_Datatype MPIU___COMPLEX128 = 0;
#endif
#elif defined(PETSC_USE_REAL___FP16)
MPI_Datatype MPIU___FP16 = 0;
#endif
MPI_Datatype MPIU_2SCALAR = 0;
#if defined(PETSC_USE_64BIT_INDICES)
MPI_Datatype MPIU_2INT = 0;
#endif
MPI_Datatype MPIU_BOOL;
MPI_Datatype MPIU_ENUM;
MPI_Datatype MPIU_FORTRANADDR;
MPI_Datatype MPIU_SIZE_T;

/*
       Function that is called to display all error messages
*/
PetscErrorCode (*PetscErrorPrintf)(const char [],...)          = PetscErrorPrintfDefault;
PetscErrorCode (*PetscHelpPrintf)(MPI_Comm,const char [],...)  = PetscHelpPrintfDefault;
PetscErrorCode (*PetscVFPrintf)(FILE*,const char[],va_list)    = PetscVFPrintfDefault;
/*
  This is needed to turn on/off GPU synchronization
*/
PetscBool PetscViennaCLSynchronize = PETSC_FALSE;
PetscBool PetscCUDASynchronize = PETSC_FALSE;

/* ------------------------------------------------------------------------------*/
/*
   Optional file where all PETSc output from various prints is saved
*/
PETSC_INTERN FILE *petsc_history;
FILE *petsc_history = NULL;

PetscErrorCode  PetscOpenHistoryFile(const char filename[],FILE **fd)
{
  PetscErrorCode ierr;
  PetscMPIInt    rank,size;
  char           pfile[PETSC_MAX_PATH_LEN],pname[PETSC_MAX_PATH_LEN],fname[PETSC_MAX_PATH_LEN],date[64];
  char           version[256];

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  if (!rank) {
    char        arch[10];
    int         err;

    ierr = PetscGetArchType(arch,10);CHKERRQ(ierr);
    ierr = PetscGetDate(date,64);CHKERRQ(ierr);
    ierr = PetscGetVersion(version,256);CHKERRQ(ierr);
    ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
    if (filename) {
      ierr = PetscFixFilename(filename,fname);CHKERRQ(ierr);
    } else {
      ierr = PetscGetHomeDirectory(pfile,sizeof(pfile));CHKERRQ(ierr);
      ierr = PetscStrlcat(pfile,"/.petschistory",sizeof(pfile));CHKERRQ(ierr);
      ierr = PetscFixFilename(pfile,fname);CHKERRQ(ierr);
    }

    *fd = fopen(fname,"a");
    if (!fd) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Cannot open file: %s",fname);

    ierr = PetscFPrintf(PETSC_COMM_SELF,*fd,"----------------------------------------\n");CHKERRQ(ierr);
    ierr = PetscFPrintf(PETSC_COMM_SELF,*fd,"%s %s\n",version,date);CHKERRQ(ierr);
    ierr = PetscGetProgramName(pname,sizeof(pname));CHKERRQ(ierr);
    ierr = PetscFPrintf(PETSC_COMM_SELF,*fd,"%s on a %s, %d proc. with options:\n",pname,arch,size);CHKERRQ(ierr);
    ierr = PetscFPrintf(PETSC_COMM_SELF,*fd,"----------------------------------------\n");CHKERRQ(ierr);

    err = fflush(*fd);
    if (err) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SYS,"fflush() failed on file");
  }
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode PetscCloseHistoryFile(FILE **fd)
{
  PetscErrorCode ierr;
  PetscMPIInt    rank;
  char           date[64];
  int            err;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  if (!rank) {
    ierr = PetscGetDate(date,64);CHKERRQ(ierr);
    ierr = PetscFPrintf(PETSC_COMM_SELF,*fd,"----------------------------------------\n");CHKERRQ(ierr);
    ierr = PetscFPrintf(PETSC_COMM_SELF,*fd,"Finished at %s\n",date);CHKERRQ(ierr);
    ierr = PetscFPrintf(PETSC_COMM_SELF,*fd,"----------------------------------------\n");CHKERRQ(ierr);
    err  = fflush(*fd);
    if (err) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SYS,"fflush() failed on file");
    err = fclose(*fd);
    if (err) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SYS,"fclose() failed on file");
  }
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------------------*/

/*
   This is ugly and probably belongs somewhere else, but I want to
  be able to put a true MPI abort error handler with command line args.

    This is so MPI errors in the debugger will leave all the stack
  frames. The default MP_Abort() cleans up and exits thus providing no useful information
  in the debugger hence we call abort() instead of MPI_Abort().
*/

void Petsc_MPI_AbortOnError(MPI_Comm *comm,PetscMPIInt *flag,...)
{
  PetscFunctionBegin;
  (*PetscErrorPrintf)("MPI error %d\n",*flag);
  abort();
}

void Petsc_MPI_DebuggerOnError(MPI_Comm *comm,PetscMPIInt *flag,...)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  (*PetscErrorPrintf)("MPI error %d\n",*flag);
  ierr = PetscAttachDebugger();
  if (ierr) PETSCABORT(*comm,*flag); /* hopeless so get out */
}

#if defined(PETSC_HAVE_CUDA)
/* CUDA validation after it is lazily initialized */
static PetscErrorCode PetscCUDAValidate(void)
{
  PetscBool             mpi_gpu_awareness;

  PetscFunctionBegin;
  if (use_gpu_aware_mpi) {
#if defined(PETSC_HAVE_OMPI_MAJOR_VERSION) && defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
    /* Trust OpenMPI's compile time cuda query interface */
    mpi_gpu_awareness = PETSC_TRUE;
#else
    /* For other MPI implementations without cuda query API, we do a GPU MPI call to see if it segfaults.
      Note that Spectrum MPI sets OMPI_MAJOR_VERSION and is CUDA-aware, but does not have MPIX_CUDA_AWARE_SUPPORT.
    */
    mpi_gpu_awareness = PetscCheckMpiGpuAwareness();
#endif
    if (!mpi_gpu_awareness) {
      (*PetscErrorPrintf)("PETSc is configured with GPU support, but your MPI is not GPU-aware. For better performance, please use a GPU-aware MPI.\n");
      (*PetscErrorPrintf)("If you do not care, add option -use_gpu_aware_mpi 0. To not see the message again, add the option to your .petscrc, OR add it to the env var PETSC_OPTIONS.\n");
      (*PetscErrorPrintf)("If you do care, for IBM Spectrum MPI on OLCF Summit, you may need jsrun --smpiargs=-gpu.\n");
      (*PetscErrorPrintf)("For OpenMPI, you need to configure it --with-cuda (https://www.open-mpi.org/faq/?category=buildcuda)\n");
      (*PetscErrorPrintf)("For MVAPICH2-GDR, you need to set MV2_USE_CUDA=1 (http://mvapich.cse.ohio-state.edu/userguide/gdr/)\n");
      (*PetscErrorPrintf)("For Cray-MPICH, you need to set MPICH_RDMA_ENABLED_CUDA=1 (https://www.olcf.ornl.gov/tutorials/gpudirect-mpich-enabled-cuda/)\n");
      PETSCABORT(PETSC_COMM_SELF,PETSC_ERR_LIB);
    }
  }
  PetscFunctionReturn(0);
}

/* Initialize the CUDA device lazily just before creating the first CUDA object. */
static PetscBool PetscNotUseGpu = PETSC_FALSE; /* Assert the code will not use GPUs */
PetscErrorCode PetscCUDAInitializeLazily(void)
{
  PetscErrorCode        ierr;
  cudaError_t           cerr;
  int                   devId,devCount;
  PetscMPIInt           rank;
  static PetscBool      cudaValdidateChecked = PETSC_FALSE;

  PetscFunctionBegin;
  if (PetscNotUseGpu) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"You asserted the code won't use GPUs with -cuda_set_device none, but it is trying to create a cuda object. Remove this option or see manpage of PetscCUDAInitialize().");
  if (!PetscCUDAInitialized) {
    cerr = cudaGetDeviceCount(&devCount);CHKERRCUDA(cerr);
    if (devCount > 1) {
      cerr = cudaSetDeviceFlags(cudaDeviceMapHost);
      cudaGetLastError(); /* Reset the last error */
      if (cerr == cudaSuccess) { /* It implies cuda runtime has not been initialized? */
        ierr  = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
        devId = rank % devCount;
        cerr  = cudaSetDevice(devId);CHKERRCUDA(cerr);
      } else if (cerr == cudaErrorSetOnActiveProcess) {
        /* It implies user has initialized cuda runtime outside of petsc. We do nothing to respect the device choice. */
      }
    }
    PetscCUDAInitialized = PETSC_TRUE;
  }
  if (!cudaValdidateChecked) {
    ierr = PetscCUDAValidate();CHKERRQ(ierr);
    cudaValdidateChecked = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}

/*@C
     PetscCUDAInitialize - Initializes the CUDA device (eagerly in PetscInitialize()) and cuBLAS/cuSPARSE on the device

     Logically collective

  Input Parameter:
  comm   - the MPI communicator that will utilize the CUDA devices
  device - the device assigned to current MPI process. Special values like PETSC_DECIDE/DEFAULT have special meanings (see details below)

  Options Database:
+  -cuda_set_device <device> - the device assigned to current MPI rank. <device> is case-insensitive and can be:
       NONE (or none, or -3) : the code will not use GPUs, otherwise it will error out;
       PETSC_DEFAULT(or DEFAULT, or -2) : do not explicitly set device, i.e., use whatever device already set by user (probably before PetscInitialize()). Init cuda runtime etc;
       PETSC_DECIDE (or DECIDE, or -1) : assign MPI ranks in comm to available devices in round-robin, and init cuda runtime etc on the selected device;
       >= 0 integer  : assign the device with this id to current MPI process. Error out if <device> is invalid. Init cuda runtime etc on this device;
     With PETSC_{DECIDE, DEFAULT}, if there are actually no GPUs, the code can still run, but it will error out when trying to create cuda objects.
.  -cuda_view                - view information about the CUDA devices.
.  -cuda_synchronize         - wait at the end of asynchronize CUDA calls so that their time gets credited to the current event; default with -log_view.
.  -log_view                 - logging, however if alone or combined with `-cuda_set_device DEFAULT | DECIDE | >=0 int`, will int cuda; if combined with `-cuda_set_device none`, won't init cuda.
-  -use_gpu_aware_mpi        - assume the MPI is GPU-aware when communicating data on GPUs.

  Level: beginner

  Notes:
   Initializing cuBLAS takes about 1/2 second therefore it is done by default in PetscCUDAInitialize() before logging begins.

@*/
PetscErrorCode PetscCUDAInitialize(MPI_Comm comm,PetscInt device)
{
  PetscErrorCode        ierr;
  cudaError_t           cerr;
  int                   devId,devCount=0;
  const PetscInt        PETSC_NONE=-3; /* Unlike PETSC_DECIDE, we don't have a macro PETSC_NONE in petsc headers */
  PetscMPIInt           rank;

  PetscFunctionBegin;
  if (!PetscCUDAInitialized) {
    cerr = cudaGetDeviceCount(&devCount);
    cudaGetLastError(); /* Reset the last error */
    if (cerr != cudaSuccess) devCount = 0;
    if (device >= 0) { /* User wants to use this specific device */
      cerr = cudaSetDeviceFlags(cudaDeviceMapHost); /* Allow it to fail since user might have already initialized the device. */
      cudaGetLastError(); /* Reset the last error */
      cerr = cudaSetDevice((int)device);CHKERRCUDA(cerr);
    } else if (device == PETSC_DECIDE) { /* Assign MPI ranks to available devices in round-robin */
      if (devCount > 0) { /* Allow no-GPU as long as user does not use GPUs */
        /* Set the device flags so that it can map host memory */
        cerr  = cudaSetDeviceFlags(cudaDeviceMapHost);CHKERRCUDA(cerr);
        ierr  = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
        devId = rank % devCount;
        cerr  = cudaSetDevice(devId);CHKERRCUDA(cerr);
      }
    } else if (device == PETSC_DEFAULT) {
      /* Do nothing, i.e., use whatever device set by user before PetscInitialize() */
    } else if (device == PETSC_NONE) {
      PetscNotUseGpu = PETSC_TRUE; /* Assert the code won't use GPUs even there are */
    } else SETERRQ1(comm,PETSC_ERR_ARG_OUTOFRANGE,"Wrong device (%D) passed to -cuda_set_device <dev>. Must be NONE(-3),PETSC_DEFAULT(-2),PETSC_DECIDE(-1) or a non-negative integer.",device);

    if (devCount > 0 && device != PETSC_NONE) {
      /* Do costly CUDA handles initialization here to not to distort petsc logging later */
      ierr = PetscCUBLASInitializeHandle();CHKERRQ(ierr);
      ierr = PetscCUSOLVERDnInitializeHandle();CHKERRQ(ierr);
      PetscCUDAInitialized = PETSC_TRUE;
    }
  }
  PetscFunctionReturn(0);
}
#endif

/*@C
   PetscEnd - Calls PetscFinalize() and then ends the program. This is useful if one
     wishes a clean exit somewhere deep in the program.

   Collective on PETSC_COMM_WORLD

   Options Database Keys are the same as for PetscFinalize()

   Level: advanced

   Note:
   See PetscInitialize() for more general runtime options.

.seealso: PetscInitialize(), PetscOptionsView(), PetscMallocDump(), PetscMPIDump(), PetscFinalize()
@*/
PetscErrorCode  PetscEnd(void)
{
  PetscFunctionBegin;
  PetscFinalize();
  exit(0);
  return 0;
}

PetscBool PetscOptionsPublish = PETSC_FALSE;
PETSC_INTERN PetscErrorCode PetscSetUseHBWMalloc_Private(void);
PETSC_INTERN PetscBool      petscsetmallocvisited;
static       char           emacsmachinename[256];

PetscErrorCode (*PetscExternalVersionFunction)(MPI_Comm) = NULL;
PetscErrorCode (*PetscExternalHelpFunction)(MPI_Comm)    = NULL;

/*@C
   PetscSetHelpVersionFunctions - Sets functions that print help and version information
   before the PETSc help and version information is printed. Must call BEFORE PetscInitialize().
   This routine enables a "higher-level" package that uses PETSc to print its messages first.

   Input Parameter:
+  help - the help function (may be NULL)
-  version - the version function (may be NULL)

   Level: developer

@*/
PetscErrorCode  PetscSetHelpVersionFunctions(PetscErrorCode (*help)(MPI_Comm),PetscErrorCode (*version)(MPI_Comm))
{
  PetscFunctionBegin;
  PetscExternalHelpFunction    = help;
  PetscExternalVersionFunction = version;
  PetscFunctionReturn(0);
}

#if defined(PETSC_USE_LOG)
PETSC_INTERN PetscBool   PetscObjectsLog;
#endif

void PetscMPI_Comm_eh(MPI_Comm *comm, PetscMPIInt *err, ...)
{
  if (PetscUnlikely(*err)) {
    PetscMPIInt len;
    char        errstring[MPI_MAX_ERROR_STRING];

    MPI_Error_string(*err,errstring,&len);
    PetscError(MPI_COMM_SELF,__LINE__,PETSC_FUNCTION_NAME,__FILE__,PETSC_MPI_ERROR_CODE,PETSC_ERROR_INITIAL,"Internal error in MPI: %s",errstring);
  }
  return;
}

PETSC_INTERN PetscErrorCode  PetscOptionsCheckInitial_Private(const char help[])
{
  char              string[64];
  MPI_Comm          comm = PETSC_COMM_WORLD;
  PetscBool         flg1 = PETSC_FALSE,flg2 = PETSC_FALSE,flg3 = PETSC_FALSE,flag,hasHelp;
  PetscErrorCode    ierr;
  PetscReal         si;
  PetscInt          intensity;
  int               i;
  PetscMPIInt       rank;
  char              version[256];
#if defined(PETSC_USE_LOG)
  char              mname[PETSC_MAX_PATH_LEN];
  PetscViewerFormat format;
  PetscBool         flg4 = PETSC_FALSE;
#endif

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);

#if !defined(PETSC_HAVE_THREADSAFETY)
  if (!(PETSC_RUNNING_ON_VALGRIND)) {
    /*
      Setup the memory management; support for tracing malloc() usage
    */
    PetscBool         mdebug = PETSC_FALSE, eachcall = PETSC_FALSE, initializenan = PETSC_FALSE, mlog = PETSC_FALSE;

    if (PetscDefined(USE_DEBUG)) {
      mdebug        = PETSC_TRUE;
      initializenan = PETSC_TRUE;
      ierr   = PetscOptionsHasName(NULL,NULL,"-malloc_test",&flg1);CHKERRQ(ierr);
    } else {
      /* don't warn about unused option */
      ierr = PetscOptionsHasName(NULL,NULL,"-malloc_test",&flg1);CHKERRQ(ierr);
      flg1 = PETSC_FALSE;
    }
    ierr = PetscOptionsGetBool(NULL,NULL,"-malloc_debug",&flg2,&flg3);CHKERRQ(ierr);
    if (flg1 || flg2) {
      mdebug        = PETSC_TRUE;
      eachcall      = PETSC_TRUE;
      initializenan = PETSC_TRUE;
    } else if (flg3 && !flg2) {
      mdebug        = PETSC_FALSE;
      eachcall      = PETSC_FALSE;
      initializenan = PETSC_FALSE;
    }

    ierr = PetscOptionsHasName(NULL,NULL,"-malloc_view",&mlog);CHKERRQ(ierr);
    if (mlog) {
      mdebug = PETSC_TRUE;
    }
    /* the next line is deprecated */
    ierr = PetscOptionsGetBool(NULL,NULL,"-malloc",&mdebug,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetBool(NULL,NULL,"-malloc_dump",&mdebug,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetBool(NULL,NULL,"-log_view_memory",&mdebug,NULL);CHKERRQ(ierr);
    if (mdebug) {
      ierr = PetscMallocSetDebug(eachcall,initializenan);CHKERRQ(ierr);
    }
    if (mlog) {
      PetscReal logthreshold = 0;
      ierr = PetscOptionsGetReal(NULL,NULL,"-malloc_view_threshold",&logthreshold,NULL);CHKERRQ(ierr);
      ierr = PetscMallocViewSet(logthreshold);CHKERRQ(ierr);
    }
#if defined(PETSC_USE_LOG)
    ierr = PetscOptionsGetBool(NULL,NULL,"-log_view_memory",&PetscLogMemory,NULL);CHKERRQ(ierr);
#endif
  }

  ierr = PetscOptionsGetBool(NULL,NULL,"-malloc_coalesce",&flg1,&flg2);CHKERRQ(ierr);
  if (flg2) {ierr = PetscMallocSetCoalesce(flg1);CHKERRQ(ierr);}
  flg1 = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-malloc_hbw",&flg1,NULL);CHKERRQ(ierr);
  /* ignore this option if malloc is already set */
  if (flg1 && !petscsetmallocvisited) {ierr = PetscSetUseHBWMalloc_Private();CHKERRQ(ierr);}

  flg1 = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-malloc_info",&flg1,NULL);CHKERRQ(ierr);
  if (!flg1) {
    flg1 = PETSC_FALSE;
    ierr = PetscOptionsGetBool(NULL,NULL,"-memory_view",&flg1,NULL);CHKERRQ(ierr);
  }
  if (flg1) {
    ierr = PetscMemorySetGetMaximumUsage();CHKERRQ(ierr);
  }
#endif

#if defined(PETSC_USE_LOG)
  ierr = PetscOptionsHasName(NULL,NULL,"-objects_dump",&PetscObjectsLog);CHKERRQ(ierr);
#endif

  /*
      Set the display variable for graphics
  */
  ierr = PetscSetDisplay();CHKERRQ(ierr);

  /*
     Print main application help message
  */
  ierr = PetscOptionsHasHelp(NULL,&hasHelp);CHKERRQ(ierr);
  if (help && hasHelp) {
    ierr = PetscPrintf(comm,help);CHKERRQ(ierr);
    ierr = PetscPrintf(comm,"----------------------------------------\n");CHKERRQ(ierr);
  }

  /*
      Print the PETSc version information
  */
  ierr = PetscOptionsHasName(NULL,NULL,"-version",&flg1);CHKERRQ(ierr);
  if (flg1 || hasHelp) {
    /*
       Print "higher-level" package version message
    */
    if (PetscExternalVersionFunction) {
      ierr = (*PetscExternalVersionFunction)(comm);CHKERRQ(ierr);
    }

    ierr = PetscGetVersion(version,256);CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(comm,"%s\n",version);CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(comm,"%s",PETSC_AUTHOR_INFO);CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(comm,"See docs/changes/index.html for recent updates.\n");CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(comm,"See docs/faq.html for problems.\n");CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(comm,"See docs/manualpages/index.html for help. \n");CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(comm,"Libraries linked from %s\n",PETSC_LIB_DIR);CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(comm,"----------------------------------------\n");CHKERRQ(ierr);
  }

  /*
       Print "higher-level" package help message
  */
  if (hasHelp) {
    PetscBool hasHelpIntro;

    if (PetscExternalHelpFunction) {
      ierr = (*PetscExternalHelpFunction)(comm);CHKERRQ(ierr);
    }
    ierr = PetscOptionsHasHelpIntro_Internal(NULL,&hasHelpIntro);CHKERRQ(ierr);
    if (hasHelpIntro) {
      ierr = PetscOptionsDestroyDefault();CHKERRQ(ierr);
      ierr = PetscFreeMPIResources();CHKERRQ(ierr);
      ierr = MPI_Finalize();CHKERRQ(ierr);
      exit(0);
    }
  }

  /*
      Setup the error handling
  */
  flg1 = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-on_error_abort",&flg1,NULL);CHKERRQ(ierr);
  if (flg1) {
    ierr = MPI_Comm_set_errhandler(comm,MPI_ERRORS_ARE_FATAL);CHKERRQ(ierr);
    ierr = PetscPushErrorHandler(PetscAbortErrorHandler,NULL);CHKERRQ(ierr);
  }
  flg1 = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-on_error_mpiabort",&flg1,NULL);CHKERRQ(ierr);
  if (flg1) { ierr = PetscPushErrorHandler(PetscMPIAbortErrorHandler,NULL);CHKERRQ(ierr);}
  flg1 = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-mpi_return_on_error",&flg1,NULL);CHKERRQ(ierr);
  if (flg1) {
    ierr = MPI_Comm_set_errhandler(comm,MPI_ERRORS_RETURN);CHKERRQ(ierr);
  }
  /* experimental */
  flg1 = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-mpi_return_error_string",&flg1,NULL);CHKERRQ(ierr);
  if (flg1) {
    MPI_Errhandler eh;

    ierr = MPI_Comm_create_errhandler(PetscMPI_Comm_eh,&eh);CHKERRQ(ierr);
    ierr = MPI_Comm_set_errhandler(comm,eh);CHKERRQ(ierr);
    ierr = MPI_Errhandler_free(&eh);CHKERRQ(ierr);
  }
  flg1 = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-no_signal_handler",&flg1,NULL);CHKERRQ(ierr);
  if (!flg1) {ierr = PetscPushSignalHandler(PetscSignalHandlerDefault,(void*)0);CHKERRQ(ierr);}

  /*
      Setup debugger information
  */
  ierr = PetscSetDefaultDebugger();CHKERRQ(ierr);
  ierr = PetscOptionsGetString(NULL,NULL,"-on_error_attach_debugger",string,sizeof(string),&flg1);CHKERRQ(ierr);
  if (flg1) {
    MPI_Errhandler err_handler;

    ierr = PetscSetDebuggerFromString(string);CHKERRQ(ierr);
    ierr = MPI_Comm_create_errhandler(Petsc_MPI_DebuggerOnError,&err_handler);CHKERRQ(ierr);
    ierr = MPI_Comm_set_errhandler(comm,err_handler);CHKERRQ(ierr);
    ierr = PetscPushErrorHandler(PetscAttachDebuggerErrorHandler,NULL);CHKERRQ(ierr);
  }
  ierr = PetscOptionsGetString(NULL,NULL,"-debug_terminal",string,sizeof(string),&flg1);CHKERRQ(ierr);
  if (flg1) { ierr = PetscSetDebugTerminal(string);CHKERRQ(ierr); }
  ierr = PetscOptionsGetString(NULL,NULL,"-start_in_debugger",string,sizeof(string),&flg1);CHKERRQ(ierr);
  ierr = PetscOptionsGetString(NULL,NULL,"-stop_for_debugger",string,sizeof(string),&flg2);CHKERRQ(ierr);
  if (flg1 || flg2) {
    PetscMPIInt    size;
    PetscInt       lsize,*nodes;
    MPI_Errhandler err_handler;
    /*
       we have to make sure that all processors have opened
       connections to all other processors, otherwise once the
       debugger has stated it is likely to receive a SIGUSR1
       and kill the program.
    */
    ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
    if (size > 2) {
      PetscMPIInt dummy = 0;
      MPI_Status  status;
      for (i=0; i<size; i++) {
        if (rank != i) {
          ierr = MPI_Send(&dummy,1,MPI_INT,i,109,comm);CHKERRQ(ierr);
        }
      }
      for (i=0; i<size; i++) {
        if (rank != i) {
          ierr = MPI_Recv(&dummy,1,MPI_INT,i,109,comm,&status);CHKERRQ(ierr);
        }
      }
    }
    /* check if this processor node should be in debugger */
    ierr  = PetscMalloc1(size,&nodes);CHKERRQ(ierr);
    lsize = size;
    ierr  = PetscOptionsGetIntArray(NULL,NULL,"-debugger_nodes",nodes,&lsize,&flag);CHKERRQ(ierr);
    if (flag) {
      for (i=0; i<lsize; i++) {
        if (nodes[i] == rank) { flag = PETSC_FALSE; break; }
      }
    }
    if (!flag) {
      ierr = PetscSetDebuggerFromString(string);CHKERRQ(ierr);
      ierr = PetscPushErrorHandler(PetscAbortErrorHandler,NULL);CHKERRQ(ierr);
      if (flg1) {
        ierr = PetscAttachDebugger();CHKERRQ(ierr);
      } else {
        ierr = PetscStopForDebugger();CHKERRQ(ierr);
      }
      ierr = MPI_Comm_create_errhandler(Petsc_MPI_AbortOnError,&err_handler);CHKERRQ(ierr);
      ierr = MPI_Comm_set_errhandler(comm,err_handler);CHKERRQ(ierr);
    }
    ierr = PetscFree(nodes);CHKERRQ(ierr);
  }

  ierr = PetscOptionsGetString(NULL,NULL,"-on_error_emacs",emacsmachinename,sizeof(emacsmachinename),&flg1);CHKERRQ(ierr);
  if (flg1 && !rank) {ierr = PetscPushErrorHandler(PetscEmacsClientErrorHandler,emacsmachinename);CHKERRQ(ierr);}

  /*
        Setup profiling and logging
  */
#if defined(PETSC_USE_INFO)
  {
    ierr = PetscInfoSetFromOptions(NULL);CHKERRQ(ierr);
  }
#endif
  ierr = PetscDetermineInitialFPTrap();
  flg1 = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-fp_trap",&flg1,&flag);CHKERRQ(ierr);
  if (flag) {ierr = PetscSetFPTrap((PetscFPTrap)flg1);CHKERRQ(ierr);}
  ierr = PetscOptionsGetInt(NULL,NULL,"-check_pointer_intensity",&intensity,&flag);CHKERRQ(ierr);
  if (flag) {ierr = PetscCheckPointerSetIntensity(intensity);CHKERRQ(ierr);}
#if defined(PETSC_USE_LOG)
  mname[0] = 0;
  ierr = PetscOptionsGetString(NULL,NULL,"-history",mname,sizeof(mname),&flg1);CHKERRQ(ierr);
  if (flg1) {
    if (mname[0]) {
      ierr = PetscOpenHistoryFile(mname,&petsc_history);CHKERRQ(ierr);
    } else {
      ierr = PetscOpenHistoryFile(NULL,&petsc_history);CHKERRQ(ierr);
    }
  }

  ierr = PetscOptionsGetBool(NULL,NULL,"-log_sync",&PetscLogSyncOn,NULL);CHKERRQ(ierr);

#if defined(PETSC_HAVE_MPE)
  flg1 = PETSC_FALSE;
  ierr = PetscOptionsHasName(NULL,NULL,"-log_mpe",&flg1);CHKERRQ(ierr);
  if (flg1) {ierr = PetscLogMPEBegin();CHKERRQ(ierr);}
#endif
  flg1 = PETSC_FALSE;
  flg3 = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-log_all",&flg1,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(NULL,NULL,"-log_summary",&flg3);CHKERRQ(ierr);
  if (flg1)                      { ierr = PetscLogAllBegin();CHKERRQ(ierr); }
  else if (flg3)                 { ierr = PetscLogDefaultBegin();CHKERRQ(ierr);}

  ierr = PetscOptionsGetString(NULL,NULL,"-log_trace",mname,sizeof(mname),&flg1);CHKERRQ(ierr);
  if (flg1) {
    char name[PETSC_MAX_PATH_LEN],fname[PETSC_MAX_PATH_LEN];
    FILE *file;
    if (mname[0]) {
      PetscSNPrintf(name,PETSC_MAX_PATH_LEN,"%s.%d",mname,rank);
      ierr = PetscFixFilename(name,fname);CHKERRQ(ierr);
      file = fopen(fname,"w");
      if (!file) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Unable to open trace file: %s",fname);
    } else file = PETSC_STDOUT;
    ierr = PetscLogTraceBegin(file);CHKERRQ(ierr);
  }

  ierr = PetscOptionsGetViewer(comm,NULL,NULL,"-log_view",NULL,&format,&flg4);CHKERRQ(ierr);
  if (flg4) {
    if (format == PETSC_VIEWER_ASCII_XML) {
      ierr = PetscLogNestedBegin();CHKERRQ(ierr);
    } else {
      ierr = PetscLogDefaultBegin();CHKERRQ(ierr);
    }
  }
  if (flg4 && format == PETSC_VIEWER_ASCII_XML) {
    PetscReal threshold = PetscRealConstant(0.01);
    ierr = PetscOptionsGetReal(NULL,NULL,"-log_threshold",&threshold,&flg1);CHKERRQ(ierr);
    if (flg1) {ierr = PetscLogSetThreshold((PetscLogDouble)threshold,NULL);CHKERRQ(ierr);}
  }
#endif

  ierr = PetscOptionsGetBool(NULL,NULL,"-saws_options",&PetscOptionsPublish,NULL);CHKERRQ(ierr);

#if defined(PETSC_HAVE_CUDA)
  {
    cudaError_t           cerr;
    PetscBool             initCUDA = PETSC_FALSE,cudaView = PETSC_FALSE,logView = PETSC_FALSE,devNone = PETSC_FALSE;
    struct cudaDeviceProp prop;
    PetscInt              device;
    char                  devStr[16]={0};
    int                   devId,devCount;
    /*
      If collecting logging information, by default, wait for GPU to complete its operations
      before returning to the CPU in order to get accurate timings of each event
    */
    ierr = PetscOptionsHasName(NULL,NULL,"-log_summary",&logView);CHKERRQ(ierr);
    if (!logView) {
      ierr = PetscOptionsHasName(NULL,NULL,"-log_view",&logView);CHKERRQ(ierr);
    }
    PetscCUDASynchronize = logView;

    ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"CUDA options","Sys");CHKERRQ(ierr);
    ierr = PetscOptionsString("-cuda_set_device","Set which MPI ranks to use which CUDA devices","PetscCUDAInitialize",devStr,devStr,sizeof(devStr),&initCUDA);CHKERRQ(ierr);
    ierr = PetscStrcasecmp("none",devStr,&devNone);CHKERRQ(ierr);
    if (devNone) device = -3; /* -3 is the locally used PETSC_NONE in PetscCUDAInitialize() */
    else {ierr = PetscOptionsInt("-cuda_set_device","Set which MPI ranks to use which CUDA devices","PetscCUDAInitialize",device,&device,&initCUDA);CHKERRQ(ierr);}
    ierr = PetscOptionsBool("-cuda_synchronize","Wait for the GPU to complete operations before returning to the CPU (on by default with -log_summary or -log_view)",NULL,PetscCUDASynchronize,&PetscCUDASynchronize,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsDeprecated("-cuda_show_devices","-cuda_view","3.12",NULL);CHKERRQ(ierr);
    ierr = PetscOptionsName("-cuda_view","Display CUDA device information and assignments",NULL,&cudaView);CHKERRQ(ierr);
    /* Get use_gpu_aware_mpi ASAP since it might be accessed even before lazy CUDA initialization */
    ierr = PetscOptionsBool("-use_gpu_aware_mpi","Use GPU-aware MPI",NULL,use_gpu_aware_mpi,&use_gpu_aware_mpi,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsEnd();CHKERRQ(ierr);

    if (initCUDA) {ierr = PetscCUDAInitialize(PETSC_COMM_WORLD,device);CHKERRQ(ierr);}
    else if (logView) { /* With -log_view, we want to do costly cuda runtime initialization early so that not to distort the timing later. */
      devCount = 0;
      cerr = cudaGetDeviceCount(&devCount);
      cudaGetLastError(); /* Reset the last error */
      if (cerr == cudaSuccess && devCount >= 1) { /* There are GPU(s) */
        devId = 0;
        if (devCount > 1) { /* Decide which GPU to init when there are multiple GPUs */
          cerr = cudaSetDeviceFlags(cudaDeviceMapHost);
          cudaGetLastError(); /* Reset the last error */
          if (cerr == cudaSuccess) { /* It implies cuda runtime has not been initialized */
            ierr  = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
            devId = rank % devCount;
            cerr  = cudaSetDevice(devId);CHKERRCUDA(cerr);
          } else if (cerr == cudaErrorSetOnActiveProcess) {
            /* It means user initialized cuda runtime outside of petsc. We respect the device choice. */
            cerr = cudaGetDevice(&devId);CHKERRCUDA(cerr);
          }
        }
        ierr = PetscCUDAInitialize(PETSC_COMM_WORLD,(PetscInt)devId);CHKERRQ(ierr);
      }
    }

    if (cudaView) {
      ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
      cerr = cudaGetDeviceCount(&devCount);CHKERRCUDA(cerr);
      for (devId = 0; devId < devCount; ++devId) {
        cerr = cudaGetDeviceProperties(&prop,devId);CHKERRCUDA(cerr);
        ierr = PetscPrintf(PETSC_COMM_WORLD, "CUDA device %d: %s\n", devId, prop.name);CHKERRQ(ierr);
      }
      cerr = cudaGetDevice(&devId);CHKERRCUDA(cerr);
      ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d] Using CUDA device %d.\n",rank,devId);CHKERRQ(ierr);
      ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT);CHKERRQ(ierr);
    }
  }
#endif

  /*
       Print basic help message
  */
  if (hasHelp) {
    ierr = (*PetscHelpPrintf)(comm,"Options for all PETSc programs:\n");CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(comm," -version: prints PETSc version\n");CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(comm," -help intro: prints example description and PETSc version, and exits\n");CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(comm," -help: prints example description, PETSc version, and available options for used routines\n");CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(comm," -on_error_abort: cause an abort when an error is detected. Useful \n ");CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(comm,"       only when run in the debugger\n");CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(comm," -on_error_attach_debugger [gdb,dbx,xxgdb,ups,noxterm]\n");CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(comm,"       start the debugger in new xterm\n");CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(comm,"       unless noxterm is given\n");CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(comm," -start_in_debugger [gdb,dbx,xxgdb,ups,noxterm]\n");CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(comm,"       start all processes in the debugger\n");CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(comm," -on_error_emacs <machinename>\n");CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(comm,"    emacs jumps to error file\n");CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(comm," -debugger_nodes [n1,n2,..] Nodes to start in debugger\n");CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(comm," -debugger_pause [m] : delay (in seconds) to attach debugger\n");CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(comm," -stop_for_debugger : prints message on how to attach debugger manually\n");CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(comm,"                      waits the delay for you to attach\n");CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(comm," -display display: Location where X window graphics and debuggers are displayed\n");CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(comm," -no_signal_handler: do not trap error signals\n");CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(comm," -mpi_return_on_error: MPI returns error code, rather than abort on internal error\n");CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(comm," -fp_trap: stop on floating point exceptions\n");CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(comm,"           note on IBM RS6000 this slows run greatly\n");CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(comm," -malloc_dump <optional filename>: dump list of unfreed memory at conclusion\n");CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(comm," -malloc: use PETSc error checking malloc (deprecated, use -malloc_debug)\n");CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(comm," -malloc no: don't use PETSc error checking malloc (deprecated, use -malloc_debug no)\n");CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(comm," -malloc_info: prints total memory usage\n");CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(comm," -malloc_view <optional filename>: keeps log of all memory allocations, displays in PetscFinalize()\n");CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(comm," -malloc_debug <true or false>: enables or disables extended checking for memory corruption\n");CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(comm," -options_view: dump list of options inputted\n");CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(comm," -options_left: dump list of unused options\n");CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(comm," -options_left no: don't dump list of unused options\n");CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(comm," -tmp tmpdir: alternative /tmp directory\n");CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(comm," -shared_tmp: tmp directory is shared by all processors\n");CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(comm," -not_shared_tmp: each processor has separate tmp directory\n");CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(comm," -memory_view: print memory usage at end of run\n");CHKERRQ(ierr);
#if defined(PETSC_USE_LOG)
    ierr = (*PetscHelpPrintf)(comm," -get_total_flops: total flops over all processors\n");CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(comm," -log_view [:filename:[format]]: logging objects and events\n");CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(comm," -log_trace [filename]: prints trace of all PETSc calls\n");CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(comm," -log_exclude <list,of,classnames>: exclude given classes from logging\n");CHKERRQ(ierr);
#if defined(PETSC_HAVE_MPE)
    ierr = (*PetscHelpPrintf)(comm," -log_mpe: Also create logfile viewable through Jumpshot\n");CHKERRQ(ierr);
#endif
#endif
#if defined(PETSC_USE_INFO)
    ierr = (*PetscHelpPrintf)(comm," -info [filename][:[~]<list,of,classnames>[:[~]self]]: print verbose information\n");CHKERRQ(ierr);
#endif
    ierr = (*PetscHelpPrintf)(comm," -options_file <file>: reads options from file\n");CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(comm," -options_monitor: monitor options to standard output, including that set previously e.g. in option files\n");CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(comm," -options_monitor_cancel: cancels all hardwired option monitors\n");CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(comm," -petsc_sleep n: sleeps n seconds before running program\n");CHKERRQ(ierr);
  }

#if defined(PETSC_HAVE_POPEN)
  {
  char machine[128];
  ierr = PetscOptionsGetString(NULL,NULL,"-popen_machine",machine,sizeof(machine),&flg1);CHKERRQ(ierr);
  if (flg1) {
    ierr = PetscPOpenSetMachine(machine);CHKERRQ(ierr);
  }
  }
#endif

  ierr = PetscOptionsGetReal(NULL,NULL,"-petsc_sleep",&si,&flg1);CHKERRQ(ierr);
  if (flg1) {
    ierr = PetscSleep(si);CHKERRQ(ierr);
  }

#if defined(PETSC_HAVE_VIENNACL)
  ierr = PetscOptionsHasName(NULL,NULL,"-log_summary",&flg3);CHKERRQ(ierr);
  if (!flg3) {
    ierr = PetscOptionsHasName(NULL,NULL,"-log_view",&flg3);CHKERRQ(ierr);
  }
  ierr = PetscOptionsGetBool(NULL,NULL,"-viennacl_synchronize",&flg3,NULL);CHKERRQ(ierr);
  PetscViennaCLSynchronize = flg3;
  ierr = PetscViennaCLInit();CHKERRQ(ierr);
#endif

  /*
     Creates the logging data structures; this is enabled even if logging is not turned on
     This is the last thing we do before returning to the user code to prevent having the
     logging numbers contaminated by any startup time associated with MPI and the GPUs
  */
#if defined(PETSC_USE_LOG)
  ierr = PetscLogInitialize();CHKERRQ(ierr);
#endif

  PetscFunctionReturn(0);
}
