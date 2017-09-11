/*
  This file contains Fortran stubs for PetscInitialize and Finalize.
*/

/*
    This is to prevent the Cray T3D version of MPI (University of Edinburgh)
  from stupidly redefining MPI_INIT(). They put this in to detect errors
  in C code,but here I do want to be calling the Fortran version from a
  C subroutine.
*/
#define T3DMPI_FORTRAN
#define T3EMPI_FORTRAN

#include <petsc/private/fortranimpl.h>

#if defined(PETSC_HAVE_CUDA)
#include <cublas_v2.h>

extern cublasHandle_t cublasv2handle;
#endif

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define petscinitialize_              PETSCINITIALIZE
#define petscinitializenoarguments_   PETSCINITIALIZENOARGUMENTS
#define petscfinalize_                PETSCFINALIZE
#define petscend_                     PETSCEND
#define iargc_                        IARGC
#define getarg_                       GETARG
#define mpi_init_                     MPI_INIT
#define petscgetcommoncomm_           PETSCGETCOMMONCOMM
#define petsccommandargumentcount_    PETSCCOMMANDARGUMENTCOUNT
#define petscgetcommandargument_      PETSCGETCOMMANDARGUMENT
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petscinitialize_              petscinitialize
#define petscinitializenoarguments_   petscinitializenoarguments
#define petscfinalize_                petscfinalize
#define petscend_                     petscend
#define mpi_init_                     mpi_init
#define iargc_                        iargc
#define getarg_                       getarg
#define petscgetcommoncomm_           petscgetcommoncomm
#define petsccommandargumentcount_    petsccommandargumentcount
#define petscgetcommandargument_      petscgetcommandargument
#endif

#if defined(PETSC_HAVE_NAGF90)
#undef iargc_
#undef getarg_
#define iargc_  f90_unix_MP_iargc
#define getarg_ f90_unix_MP_getarg
#endif
#if defined(PETSC_USE_NARGS) /* Digital Fortran */
#undef iargc_
#undef getarg_
#define iargc_  NARGS
#define getarg_ GETARG
#elif defined(PETSC_HAVE_PXFGETARG_NEW)  /* cray x1 */
#undef iargc_
#undef getarg_
#define iargc_  ipxfargc_
#define getarg_ pxfgetarg_
#endif
#if defined(PETSC_HAVE_FORTRAN_IARGC_UNDERSCORE) /* HPUX + no underscore */
#undef iargc_
#undef getarg_
#define iargc_   iargc_
#define getarg_  getarg_
#endif

#if defined(PETSC_HAVE_FORTRAN_GET_COMMAND_ARGUMENT) /* Fortran 2003 */
#undef iargc_
#undef getarg_
#define iargc_ petsccommandargumentcount_
#define getarg_ petscgetcommandargument_
#elif defined(PETSC_HAVE_GFORTRAN_IARGC) /* gfortran from gcc4 */
#undef iargc_
#undef getarg_
#define iargc_  _gfortran_iargc
#define getarg_ _gfortran_getarg_i4
#elif defined(PETSC_HAVE_BGL_IARGC) /* bgl g77 has different external & internal name mangling */
#undef iargc_
#undef getarg_
#define iargc  iargc_
#define getarg getarg_
#endif

/*
    The extra _ is because the f2c compiler puts an
  extra _ at the end if the original routine name
  contained any _.
*/
#if defined(PETSC_HAVE_FORTRAN_UNDERSCORE_UNDERSCORE)
#define mpi_init_             mpi_init__
#endif

#if defined(PETSC_HAVE_MPIUNI)
#if defined(mpi_init_)
#undef mpi_init_
#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define mpi_init_             PETSC_MPI_INIT
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define mpi_init_             petsc_mpi_init
#elif defined(PETSC_HAVE_FORTRAN_UNDERSCORE_UNDERSCORE)
#define mpi_init_             petsc_mpi_init__
#endif
#else    /* mpi_init_ */
#define mpi_init_             petsc_mpi_init_
#endif   /* mpi_init_ */
#endif   /* PETSC_HAVE_MPIUNI */

PETSC_EXTERN void PETSC_STDCALL mpi_init_(int*);
PETSC_EXTERN void PETSC_STDCALL petscgetcommoncomm_(PetscMPIInt*);

/*
     Different Fortran compilers handle command lines in different ways
*/
#if defined(PETSC_HAVE_FORTRAN_GET_COMMAND_ARGUMENT) /* Fortran 2003  - same as 'else' case */
PETSC_EXTERN int iargc_(void);
PETSC_EXTERN void getarg_(int*,char*,int);
#elif defined(PETSC_USE_NARGS)
PETSC_EXTERN short __stdcall NARGS();
PETSC_EXTERN void __stdcall GETARG(short*,char*,int,short *);

#elif defined(PETSC_HAVE_FORTRAN_STDCALL)
PETSC_EXTERN int PETSC_STDCALL IARGC();
PETSC_EXTERN void PETSC_STDCALL GETARG(int *,char *,int);

#elif defined(PETSC_HAVE_PXFGETARG_NEW)
PETSC_EXTERN int iargc_();
PETSC_EXTERN void getarg_(int*,char*,int*,int*,int);

#else
PETSC_EXTERN int iargc_();
PETSC_EXTERN void getarg_(int*,char*,int);
/*
      The Cray T3D/T3E use the PXFGETARG() function
*/
#if defined(PETSC_HAVE_PXFGETARG)
PETSC_EXTERN void PXFGETARG(int*,_fcd,int*,int*);
#endif
#endif

#if (defined(PETSC_HAVE_COMPLEX) && !defined(PETSC_HAVE_MPI_C_DOUBLE_COMPLEX)) || defined(PETSC_USE_REAL___FLOAT128) || defined(PETSC_USE_REAL___FP16)
extern MPI_Op MPIU_SUM;

PETSC_EXTERN void MPIAPI PetscSum_Local(void*,void*,PetscMPIInt*,MPI_Datatype*);

#endif
#if defined(PETSC_USE_REAL___FLOAT128) || defined(PETSC_USE_REAL___FP16)

PETSC_EXTERN void MPIAPI PetscSum_Local(void*,void*,PetscMPIInt*,MPI_Datatype*);
PETSC_EXTERN void MPIAPI PetscMax_Local(void*,void*,PetscMPIInt*,MPI_Datatype*);
PETSC_EXTERN void MPIAPI PetscMin_Local(void*,void*,PetscMPIInt*,MPI_Datatype*);
#endif

PETSC_INTERN void MPIAPI MPIU_MaxSum_Local(void*,void*,PetscMPIInt*,MPI_Datatype*);
PETSC_EXTERN PetscMPIInt MPIAPI Petsc_DelCounter(MPI_Comm,PetscMPIInt,void*,void*);
PETSC_EXTERN PetscMPIInt MPIAPI Petsc_DelComm_Inner(MPI_Comm,PetscMPIInt,void*,void*);
PETSC_EXTERN PetscMPIInt MPIAPI Petsc_DelComm_Outer(MPI_Comm,PetscMPIInt,void*,void*);

extern PetscErrorCode  PetscOptionsCheckInitial_Private(void);
extern PetscErrorCode  PetscOptionsCheckInitial_Components(void);
extern PetscErrorCode  PetscInitialize_DynamicLibraries(void);
#if defined(PETSC_USE_LOG)
extern PetscErrorCode  PetscLogInitialize(void);
#endif
extern PetscErrorCode  PetscMallocAlign(size_t,int,const char[],const char[],void**);
extern PetscErrorCode  PetscFreeAlign(void*,int,const char[],const char[]);
extern int  PetscGlobalArgc;
extern char **PetscGlobalArgs;

/*
    Reads in Fortran command line argments and sends them to
  all processors.
*/

PetscErrorCode PETScParseFortranArgs_Private(int *argc,char ***argv)
{
#if defined(PETSC_USE_NARGS)
  short          i,flg;
#else
  int            i;
#endif
  PetscErrorCode ierr;
  int            warg = 256;
  PetscMPIInt    rank;
  char           *p;

  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  if (!rank) {
#if defined(PETSC_HAVE_IARG_COUNT_PROGNAME)
    *argc = iargc_();
#else
    /* most compilers do not count the program name for argv[0] */
    *argc = 1 + iargc_();
#endif
  }
  ierr = MPI_Bcast(argc,1,MPI_INT,0,PETSC_COMM_WORLD);CHKERRQ(ierr);

  /* PetscTrMalloc() not yet set, so don't use PetscMalloc() */
  ierr = PetscMallocAlign((*argc+1)*(warg*sizeof(char)+sizeof(char*)),0,0,0,(void**)argv);CHKERRQ(ierr);
  (*argv)[0] = (char*)(*argv + *argc + 1);

  if (!rank) {
    ierr = PetscMemzero((*argv)[0],(*argc)*warg*sizeof(char));CHKERRQ(ierr);
    for (i=0; i<*argc; i++) {
      (*argv)[i+1] = (*argv)[i] + warg;
#if defined (PETSC_HAVE_FORTRAN_GET_COMMAND_ARGUMENT) /* same as 'else' case */
      getarg_(&i,(*argv)[i],warg);
#elif defined(PETSC_HAVE_PXFGETARG_NEW)
      {char *tmp = (*argv)[i];
      int ilen;
      getarg_(&i,tmp,&ilen,&ierr,warg);CHKERRQ(ierr);
      tmp[ilen] = 0;}
#elif defined(PETSC_USE_NARGS)
      GETARG(&i,(*argv)[i],warg,&flg);
#else
      /*
      Because the stupid #defines above define all kinds of things to getarg_ we cannot do this test
      #elif defined(PETSC_HAVE_GETARG)
      getarg_(&i,(*argv)[i],warg);
      #else
         SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Cannot get Fortran command line arguments");
      */
      getarg_(&i,(*argv)[i],warg);
#endif
      /* zero out garbage at end of each argument */
      p = (*argv)[i] + warg-1;
      while (p > (*argv)[i]) {
        if (*p == ' ') *p = 0;
        p--;
      }
    }
  }
  ierr = MPI_Bcast((*argv)[0],*argc*warg,MPI_CHAR,0,PETSC_COMM_WORLD);CHKERRQ(ierr);
  if (rank) {
    for (i=0; i<*argc; i++) (*argv)[i+1] = (*argv)[i] + warg;
  }
  return 0;
}

#if defined(PETSC_SERIALIZE_FUNCTIONS)
extern PetscFPT PetscFPTData;
#endif

#if defined(PETSC_HAVE_THREADSAFETY)
extern PetscSpinlock PetscViewerASCIISpinLockOpen;
extern PetscSpinlock PetscViewerASCIISpinLockStdout;
extern PetscSpinlock PetscViewerASCIISpinLockStderr;
extern PetscSpinlock PetscCommSpinLock;
#endif

/* -----------------------------------------------------------------------------------------------*/

#if defined(PETSC_HAVE_SAWS)
#include <petscviewersaws.h>
extern PetscErrorCode  PetscInitializeSAWs(const char[]);
#endif

/*
    petscinitialize - Version called from Fortran.

    Notes:
      Since this is called from Fortran it does not return error codes

*/
static void petscinitialize_internal(char* filename, PetscInt len, PetscBool readarguments, PetscErrorCode *ierr)
{
  int            j,i;
#if defined (PETSC_USE_NARGS)
  short          flg;
#endif
#if defined(PETSC_HAVE_CUDA)
  PetscBool      flg2;
  cublasStatus_t cberr;
#endif
  int            flag;
  PetscMPIInt    size;
  char           *t1,name[256],hostname[64];
  PetscMPIInt    f_petsc_comm_world;

  *ierr = PetscMemzero(name,sizeof(name)); if (*ierr) return;
  if (PetscInitializeCalled) {*ierr = 0; return;}

  /* this must be initialized in a routine, not as a constant declaration*/
  PETSC_STDOUT = stdout;
  PETSC_STDERR = stderr;

  /* on Windows - set printf to default to printing 2 digit exponents */
#if defined(PETSC_HAVE__SET_OUTPUT_FORMAT)
  _set_output_format(_TWO_DIGIT_EXPONENT);
#endif

  *ierr = PetscOptionsCreateDefault();
  if (*ierr) return;
  i = 0;
#if defined (PETSC_HAVE_FORTRAN_GET_COMMAND_ARGUMENT) /* same as 'else' case */
  getarg_(&i,name,sizeof(name));
#elif defined (PETSC_HAVE_PXFGETARG_NEW)
  { int ilen,sierr;
    getarg_(&i,name,&ilen,&sierr,256);
    if (sierr) PetscStrncpy(name,"Unknown Name",256);
    else name[ilen] = 0;
  }
#elif defined(PETSC_USE_NARGS)
  GETARG(&i,name,256,&flg);
#else
  getarg_(&i,name,256);
#endif
  /* Eliminate spaces at the end of the string */
  for (j=sizeof(name)-2; j>=0; j--) {
    if (name[j] != ' ') {
      name[j+1] = 0;
      break;
    }
  }
  if (j<0) PetscStrncpy(name,"Unknown Name",256);
  *ierr = PetscSetProgramName(name);
  if (*ierr) {(*PetscErrorPrintf)("PetscInitialize: Calling PetscSetProgramName()\n");return;}

  /* check if PETSC_COMM_WORLD is initialized by the user in fortran */
  petscgetcommoncomm_(&f_petsc_comm_world);
  MPI_Initialized(&flag);
  if (!flag) {
    PetscMPIInt mierr;

    if (f_petsc_comm_world) {(*PetscErrorPrintf)("You cannot set PETSC_COMM_WORLD if you have not initialized MPI first\n");return;}
    mpi_init_(&mierr);
    if (mierr) {
      *ierr = mierr;
      (*PetscErrorPrintf)("PetscInitialize: Calling Fortran MPI_Init()\n");
      return;
    }
    PetscBeganMPI = PETSC_TRUE;
  }
  if (f_petsc_comm_world) PETSC_COMM_WORLD = MPI_Comm_f2c(*(MPI_Fint*)&f_petsc_comm_world); /* User called MPI_INITIALIZE() and changed PETSC_COMM_WORLD */
  else PETSC_COMM_WORLD = MPI_COMM_WORLD;
  PetscInitializeCalled = PETSC_TRUE;

  *ierr = PetscSpinlockCreate(&PetscViewerASCIISpinLockOpen);
  if (*ierr) {(*PetscErrorPrintf)("PetscInitialize: Creating global spin lock\n");return;}
  *ierr = PetscSpinlockCreate(&PetscViewerASCIISpinLockStdout);
  if (*ierr) {(*PetscErrorPrintf)("PetscInitialize: Creating global spin lock\n");return;}
  *ierr = PetscSpinlockCreate(&PetscViewerASCIISpinLockStderr);
  if (*ierr) {(*PetscErrorPrintf)("PetscInitialize: Creating global spin lock\n");return;}
  *ierr = PetscSpinlockCreate(&PetscCommSpinLock);
  if (*ierr) {(*PetscErrorPrintf)("PetscInitialize: Creating global spin lock\n");return;}

  *ierr = PetscErrorPrintfInitialize();
  if (*ierr) {(*PetscErrorPrintf)("PetscInitialize: Calling PetscErrorPrintfInitialize()\n");return;}
  *ierr = MPI_Comm_rank(MPI_COMM_WORLD,&PetscGlobalRank);
  if (*ierr) {(*PetscErrorPrintf)("PetscInitialize: Setting PetscGlobalRank\n");return;}
  *ierr = MPI_Comm_size(MPI_COMM_WORLD,&PetscGlobalSize);
  if (*ierr) {(*PetscErrorPrintf)("PetscInitialize: Setting PetscGlobalSize\n");return;}

  MPIU_BOOL = MPI_INT;
  MPIU_ENUM = MPI_INT;

#if defined(PETSC_HAVE_COMPLEX)
  /*
     Initialized the global variable; this is because with
     shared libraries the constructors for global variables
     are not called; at least on IRIX.
  */
  {
#if defined(PETSC_CLANGUAGE_CXX) && !defined(PETSC_USE_REAL___FLOAT128)
    PetscComplex ic(0.0,1.0);
    PETSC_i = ic;
#else
    PETSC_i = _Complex_I;
#endif
  }

#if !defined(PETSC_HAVE_MPI_C_DOUBLE_COMPLEX)
  *ierr = MPI_Type_contiguous(2,MPI_DOUBLE,&MPIU_C_DOUBLE_COMPLEX);
  if (*ierr) {(*PetscErrorPrintf)("PetscInitialize:Creating MPI types\n");return;}
  *ierr = MPI_Type_commit(&MPIU_C_DOUBLE_COMPLEX);
  if (*ierr) {(*PetscErrorPrintf)("PetscInitialize:Creating MPI types\n");return;}
  *ierr = MPI_Type_contiguous(2,MPI_FLOAT,&MPIU_C_COMPLEX);
  if (*ierr) {(*PetscErrorPrintf)("PetscInitialize:Creating MPI types\n");return;}
  *ierr = MPI_Type_commit(&MPIU_C_COMPLEX);
  if (*ierr) {(*PetscErrorPrintf)("PetscInitialize:Creating MPI types\n");return;}
  *ierr = MPI_Op_create(PetscSum_Local,1,&MPIU_SUM);
  if (*ierr) {(*PetscErrorPrintf)("PetscInitialize:Creating MPI ops\n");return;}
#endif

#endif

#if defined(PETSC_USE_REAL___FLOAT128)
  *ierr = MPI_Type_contiguous(2,MPI_DOUBLE,&MPIU___FLOAT128);
  if (*ierr) {(*PetscErrorPrintf)("PetscInitialize:Creating MPI types\n");return;}
  *ierr = MPI_Type_commit(&MPIU___FLOAT128);
  if (*ierr) {(*PetscErrorPrintf)("PetscInitialize:Creating MPI types\n");return;}
#if defined(PETSC_HAVE_COMPLEX)
  *ierr = MPI_Type_contiguous(4,MPI_DOUBLE,&MPIU___COMPLEX128);
  if (*ierr) {(*PetscErrorPrintf)("PetscInitialize:Creating MPI types\n");return;}
  *ierr = MPI_Type_commit(&MPIU___COMPLEX128);
  if (*ierr) {(*PetscErrorPrintf)("PetscInitialize:Creating MPI types\n");return;}
#endif
  *ierr = MPI_Op_create(PetscSum_Local,1,&MPIU_SUM);
  if (*ierr) {(*PetscErrorPrintf)("PetscInitialize:Creating MPI ops\n");return;}
  *ierr = MPI_Op_create(PetscMax_Local,1,&MPIU_MAX);
  if (*ierr) {(*PetscErrorPrintf)("PetscInitialize:Creating MPI ops\n");return;}
  *ierr = MPI_Op_create(PetscMin_Local,1,&MPIU_MIN);
  if (*ierr) {(*PetscErrorPrintf)("PetscInitialize:Creating MPI ops\n");return;}
#elif defined(PETSC_USE_REAL___FP16)
  *ierr = MPI_Type_contiguous(2,MPI_CHAR,&MPIU___FP16);
  if (*ierr) {(*PetscErrorPrintf)("PetscInitialize:Creating MPI types\n");return;}
  *ierr = MPI_Type_commit(&MPIU___FP16);
  if (*ierr) {(*PetscErrorPrintf)("PetscInitialize:Creating MPI types\n");return;}
  *ierr = MPI_Op_create(PetscSum_Local,1,&MPIU_SUM);
  if (*ierr) {(*PetscErrorPrintf)("PetscInitialize:Creating MPI ops\n");return;}
  *ierr = MPI_Op_create(PetscMax_Local,1,&MPIU_MAX);
  if (*ierr) {(*PetscErrorPrintf)("PetscInitialize:Creating MPI ops\n");return;}
  *ierr = MPI_Op_create(PetscMin_Local,1,&MPIU_MIN);
  if (*ierr) {(*PetscErrorPrintf)("PetscInitialize:Creating MPI ops\n");return;}
#endif

  /*
       Create the PETSc MPI reduction operator that sums of the first
     half of the entries and maxes the second half.
  */
  *ierr = MPI_Op_create(MPIU_MaxSum_Local,1,&MPIU_MAXSUM_OP);
  if (*ierr) {(*PetscErrorPrintf)("PetscInitialize:Creating MPI ops\n");return;}

  *ierr = MPI_Type_contiguous(2,MPIU_SCALAR,&MPIU_2SCALAR);
  if (*ierr) {(*PetscErrorPrintf)("PetscInitialize:Creating MPI types\n");return;}
  *ierr = MPI_Type_commit(&MPIU_2SCALAR);
  if (*ierr) {(*PetscErrorPrintf)("PetscInitialize:Creating MPI types\n");return;}
#if defined(PETSC_USE_64BIT_INDICES) || !defined(MPI_2INT)
  *ierr = MPI_Type_contiguous(2,MPIU_INT,&MPIU_2INT);
  if (*ierr) {(*PetscErrorPrintf)("PetscInitialize:Creating MPI types\n");return;}
  *ierr = MPI_Type_commit(&MPIU_2INT);
  if (*ierr) {(*PetscErrorPrintf)("PetscInitialize:Creating MPI types\n");return;}
#endif
  *ierr = MPI_Keyval_create(MPI_NULL_COPY_FN,Petsc_DelCounter,&Petsc_Counter_keyval,(void*)0);
  if (*ierr) {(*PetscErrorPrintf)("PetscInitialize:Creating MPI keyvals\n");return;}
  *ierr = MPI_Keyval_create(MPI_NULL_COPY_FN,Petsc_DelComm_Outer,&Petsc_InnerComm_keyval,(void*)0);
  if (*ierr) {(*PetscErrorPrintf)("PetscInitialize:Creating MPI keyvals\n");return;}
  *ierr = MPI_Keyval_create(MPI_NULL_COPY_FN,Petsc_DelComm_Inner,&Petsc_OuterComm_keyval,(void*)0);
  if (*ierr) {(*PetscErrorPrintf)("PetscInitialize:Creating MPI keyvals\n");return;}

  /*
     PetscInitializeFortran() is called twice. Here it initializes
     PETSC_NULL_CHARACTER_Fortran. Below it initializes the PETSC_VIEWERs.
     The PETSC_VIEWERs have not been created yet, so they must be initialized
     below.
  */
  PetscInitializeFortran();
  if(readarguments == PETSC_TRUE) {
    PETScParseFortranArgs_Private(&PetscGlobalArgc,&PetscGlobalArgs);
    FIXCHAR(filename,len,t1);
    *ierr = PetscOptionsInsert(NULL,&PetscGlobalArgc,&PetscGlobalArgs,t1);
    FREECHAR(filename,t1);
    if (*ierr) {(*PetscErrorPrintf)("PetscInitialize:Creating options database\n");return;}
  }
  *ierr = PetscOptionsCheckInitial_Private();
  if (*ierr) {(*PetscErrorPrintf)("PetscInitialize:Checking initial options\n");return;}
  *ierr = PetscCitationsInitialize();
  if (*ierr) {(*PetscErrorPrintf)("PetscInitialize:PetscCitationsInitialize()\n");return;}
#if defined(PETSC_HAVE_SAWS)
  *ierr = PetscInitializeSAWs(NULL);
  if (*ierr) {(*PetscErrorPrintf)("PetscInitialize:Initializing SAWs\n");return;}
#endif
#if defined(PETSC_USE_LOG)
  *ierr = PetscLogInitialize();
  if (*ierr) {(*PetscErrorPrintf)("PetscInitialize: intializing logging\n");return;}
#endif
  *ierr = PetscInitialize_DynamicLibraries();
  if (*ierr) {(*PetscErrorPrintf)("PetscInitialize:Initializing dynamic libraries\n");return;}

  *ierr = PetscInitializeFortran();
  if (*ierr) { (*PetscErrorPrintf)("PetscInitialize:Setting up common block\n");return;}

  *ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);
  if (*ierr) { (*PetscErrorPrintf)("PetscInitialize:Getting MPI_Comm_size()\n");return;}
  *ierr = PetscInfo1(0,"(Fortran):PETSc successfully started: procs %d\n",size);
  if (*ierr) { (*PetscErrorPrintf)("PetscInitialize:Calling PetscInfo()\n");return;}
  *ierr = PetscGetHostName(hostname,64);
  if (*ierr) { (*PetscErrorPrintf)("PetscInitialize:Getting hostname\n");return;}
  *ierr = PetscInfo1(0,"Running on machine: %s\n",hostname);
  if (*ierr) { (*PetscErrorPrintf)("PetscInitialize:Calling PetscInfo()\n");return;}
  *ierr = PetscOptionsCheckInitial_Components();
  if (*ierr) {(*PetscErrorPrintf)("PetscInitialize:Checking initial options\n");return;}

#if defined(PETSC_USE_DEBUG) && !defined(PETSC_HAVE_THREADSAFETY)
  *ierr = PetscStackCreate();
  if (*ierr) {(*PetscErrorPrintf)("PetscInitialize:PetscStackCreate()\n");return;}
#endif

#if defined(PETSC_SERIALIZE_FUNCTIONS)
  *ierr = PetscFPTCreate(10000);
  if (*ierr) {(*PetscErrorPrintf)("PetscInitialize:PetscFPTCreate()\n");return;}
#endif

#if defined(PETSC_HAVE_CUDA)
  flg2  = PETSC_TRUE;
  *ierr = PetscOptionsGetBool(NULL,NULL,"-cublas",&flg2,NULL);
  if (flg2) {
    cberr = cublasCreate(&cublasv2handle);
    if (((int)cberr) != (int)CUBLAS_STATUS_SUCCESS) {(*PetscErrorPrintf)("PetscInitialize:CUBLAS error %d\n",cberr);return;}
  }
#endif
}

PETSC_EXTERN void PETSC_STDCALL petscinitialize_(char* filename PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  petscinitialize_internal(filename, len, PETSC_TRUE, ierr);
}

PETSC_EXTERN void PETSC_STDCALL petscinitializenoarguments_(PetscErrorCode *ierr)
{
  petscinitialize_internal(NULL, (PetscInt) 0, PETSC_FALSE, ierr);
}


PETSC_EXTERN void PETSC_STDCALL petscfinalize_(PetscErrorCode *ierr)
{
#if defined(PETSC_HAVE_SUNMATHPRO)
  extern void standard_arithmetic();
  standard_arithmetic();
#endif
  /* was malloced with PetscMallocAlign() so free the same way */
  *ierr = PetscFreeAlign(PetscGlobalArgs,0,0,0);if (*ierr) {(*PetscErrorPrintf)("PetscFinalize:Freeing args\n");return;}

  *ierr = PetscFinalize();
}

PETSC_EXTERN void PETSC_STDCALL petscend_(PetscErrorCode *ierr)
{
#if defined(PETSC_HAVE_SUNMATHPRO)
  extern void standard_arithmetic();
  standard_arithmetic();
#endif

  *ierr = PetscEnd();
}

