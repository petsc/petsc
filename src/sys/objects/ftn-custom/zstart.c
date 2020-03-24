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

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define petscinitialize_              PETSCINITIALIZE
#define petscinitializenoarguments_   PETSCINITIALIZENOARGUMENTS
#define petscfinalize_                PETSCFINALIZE
#define petscend_                     PETSCEND
#define iargc_                        IARGC
#define getarg_                       GETARG
#define mpi_init_                     MPI_INIT
#define petscgetcomm_                 PETSCGETCOMM
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
#define petscgetcomm_                 petscgetcomm
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
#elif defined(PETSC_HAVE__GFORTRAN_IARGC) /* gfortran from gcc4 */
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

PETSC_EXTERN void mpi_init_(int*);
PETSC_EXTERN void petscgetcomm_(PetscMPIInt*);

/*
     Different Fortran compilers handle command lines in different ways
*/
#if defined(PETSC_HAVE_FORTRAN_GET_COMMAND_ARGUMENT) /* Fortran 2003  - same as 'else' case */
PETSC_EXTERN int iargc_(void);
PETSC_EXTERN void getarg_(int*,char*,int);
#elif defined(PETSC_USE_NARGS)
PETSC_EXTERN short __stdcall NARGS();
PETSC_EXTERN void __stdcall GETARG(short*,char*,int,short *);

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
PETSC_EXTERN MPI_Op MPIU_SUM;

PETSC_EXTERN void MPIAPI PetscSum_Local(void*,void*,PetscMPIInt*,MPI_Datatype*);

#endif
#if defined(PETSC_USE_REAL___FLOAT128) || defined(PETSC_USE_REAL___FP16)

PETSC_EXTERN void MPIAPI PetscSum_Local(void*,void*,PetscMPIInt*,MPI_Datatype*);
PETSC_EXTERN void MPIAPI PetscMax_Local(void*,void*,PetscMPIInt*,MPI_Datatype*);
PETSC_EXTERN void MPIAPI PetscMin_Local(void*,void*,PetscMPIInt*,MPI_Datatype*);
#endif

PETSC_INTERN void MPIAPI MPIU_MaxSum_Local(void*,void*,PetscMPIInt*,MPI_Datatype*);
PETSC_EXTERN PetscMPIInt MPIAPI Petsc_Counter_Attr_Delete_Fn(MPI_Comm,PetscMPIInt,void*,void*);
PETSC_EXTERN PetscMPIInt MPIAPI Petsc_InnerComm_Attr_Delete_Fn(MPI_Comm,PetscMPIInt,void*,void*);
PETSC_EXTERN PetscMPIInt MPIAPI Petsc_OuterComm_Attr_Delete_Fn(MPI_Comm,PetscMPIInt,void*,void*);

PETSC_INTERN PetscErrorCode PetscOptionsCheckInitial_Private(void);
PETSC_INTERN PetscErrorCode PetscOptionsCheckInitial_Components(void);
PETSC_INTERN PetscErrorCode PetscInitialize_DynamicLibraries(void);
#if defined(PETSC_USE_LOG)
PETSC_INTERN PetscErrorCode PetscLogInitialize(void);
#endif
PETSC_EXTERN PetscErrorCode PetscMallocAlign(size_t,PetscBool,int,const char[],const char[],void**);
PETSC_EXTERN PetscErrorCode PetscFreeAlign(void*,int,const char[],const char[]);
PETSC_INTERN int  PetscGlobalArgc;
PETSC_INTERN char **PetscGlobalArgs;

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
  ierr = PetscMallocAlign((*argc+1)*(warg*sizeof(char)+sizeof(char*)),PETSC_FALSE,0,0,0,(void**)argv);CHKERRQ(ierr);
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
PETSC_INTERN PetscFPT PetscFPTData;
#endif

#if defined(PETSC_HAVE_ADIOS)
#include <adios.h>
#include <adios_read.h>
#endif
/* -----------------------------------------------------------------------------------------------*/

#if defined(PETSC_HAVE_SAWS)
#include <petscviewersaws.h>
PETSC_INTERN PetscErrorCode  PetscInitializeSAWs(const char[]);
#endif

PETSC_EXTERN PetscMPIInt MPIAPI Petsc_ShmComm_Attr_Delete_Fn(MPI_Comm,PetscMPIInt,void *,void *);
PETSC_INTERN PetscErrorCode PetscPreMPIInit_Private();

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
  petscgetcomm_(&f_petsc_comm_world);
  MPI_Initialized(&flag);
  if (!flag) {
    PetscMPIInt mierr;

    if (f_petsc_comm_world) {(*PetscErrorPrintf)("You cannot set PETSC_COMM_WORLD if you have not initialized MPI first\n");return;}

    *ierr = PetscPreMPIInit_Private(); if(*ierr) return;
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
  *ierr = MPI_Comm_set_errhandler(PETSC_COMM_WORLD,MPI_ERRORS_RETURN);
  if (*ierr) {(*PetscErrorPrintf)("PetscInitialize: Setting MPI error handler\n");return;}
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
  MPIU_FORTRANADDR = (sizeof(void*) == sizeof(int)) ? MPI_INT : MPIU_INT64;
  if (sizeof(size_t) == sizeof(unsigned)) MPIU_SIZE_T = MPI_UNSIGNED;
  else if (sizeof(size_t) == sizeof(unsigned long)) MPIU_SIZE_T = MPI_UNSIGNED_LONG;
#if defined(PETSC_SIZEOF_LONG_LONG)
  else if (sizeof(size_t) == sizeof(unsigned long long)) MPIU_SIZE_T = MPI_UNSIGNED_LONG_LONG;
#endif
  else {(*PetscErrorPrintf)("PetscInitialize: Could not find MPI type for size_t\n"); return;}

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
#if defined(PETSC_USE_64BIT_INDICES)
  *ierr = MPI_Type_contiguous(2,MPIU_INT,&MPIU_2INT);
  if (*ierr) {(*PetscErrorPrintf)("PetscInitialize:Creating MPI types\n");return;}
  *ierr = MPI_Type_commit(&MPIU_2INT);
  if (*ierr) {(*PetscErrorPrintf)("PetscInitialize:Creating MPI types\n");return;}
#endif
  *ierr = MPI_Comm_create_keyval(MPI_COMM_NULL_COPY_FN,Petsc_Counter_Attr_Delete_Fn,&Petsc_Counter_keyval,(void*)0);
  if (*ierr) {(*PetscErrorPrintf)("PetscInitialize:Creating MPI keyvals\n");return;}
  *ierr = MPI_Comm_create_keyval(MPI_COMM_NULL_COPY_FN,Petsc_InnerComm_Attr_Delete_Fn,&Petsc_InnerComm_keyval,(void*)0);
  if (*ierr) {(*PetscErrorPrintf)("PetscInitialize:Creating MPI keyvals\n");return;}
  *ierr = MPI_Comm_create_keyval(MPI_COMM_NULL_COPY_FN,Petsc_OuterComm_Attr_Delete_Fn,&Petsc_OuterComm_keyval,(void*)0);
  if (*ierr) {(*PetscErrorPrintf)("PetscInitialize:Creating MPI keyvals\n");return;}
  *ierr = MPI_Comm_create_keyval(MPI_COMM_NULL_COPY_FN,Petsc_ShmComm_Attr_Delete_Fn,&Petsc_ShmComm_keyval,(void*)0);
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
    if (*ierr) {(*PetscErrorPrintf)("PetscInitialize:Creating options database\n");return;}
    FREECHAR(filename,t1);
    if (*ierr) {(*PetscErrorPrintf)("PetscInitialize:Freeing string in creating options database\n");return;}
  }
  *ierr = PetscOptionsCheckInitial_Private();
  if (*ierr) {(*PetscErrorPrintf)("PetscInitialize:Checking initial options\n");return;}
  /* call a second time to check options database */
  *ierr = PetscErrorPrintfInitialize();
  if (*ierr) {(*PetscErrorPrintf)("PetscInitialize: Calling PetscErrorPrintfInitialize()\n");return;}
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
#if defined(PETSC_HAVE_ADIOS)
  *ierr = adios_init_noxml(PETSC_COMM_WORLD);
  if (*ierr) {(*PetscErrorPrintf)("PetscInitialize:adios_init_noxml()\n");return;}
  *ierr = adios_declare_group(&Petsc_adios_group,"PETSc","",adios_stat_default);
  if (*ierr) {(*PetscErrorPrintf)("PetscInitialize:adios_declare_group()\n");return;}
  *ierr = adios_select_method(Petsc_adios_group,"MPI","","");
  if (*ierr) {(*PetscErrorPrintf)("PetscInitialize:adios_select_method()\n");return;}
  *ierr = adios_read_init_method(ADIOS_READ_METHOD_BP,PETSC_COMM_WORLD,"");
  if (*ierr) {(*PetscErrorPrintf)("PetscInitialize:adios_read_init_method()\n");return;}
#endif
}

PETSC_EXTERN void petscinitialize_(char* filename,PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T len)
{
  petscinitialize_internal(filename, len, PETSC_TRUE, ierr);
}

PETSC_EXTERN void petscinitializenoarguments_(PetscErrorCode *ierr)
{
  petscinitialize_internal(NULL, (PetscInt) 0, PETSC_FALSE, ierr);
}


PETSC_EXTERN void petscfinalize_(PetscErrorCode *ierr)
{
#if defined(PETSC_HAVE_SUNMATHPRO)
  extern void standard_arithmetic();
  standard_arithmetic();
#endif
  /* was malloced with PetscMallocAlign() so free the same way */
  *ierr = PetscFreeAlign(PetscGlobalArgs,0,0,0);if (*ierr) {(*PetscErrorPrintf)("PetscFinalize:Freeing args\n");return;}

  *ierr = PetscFinalize();
}

PETSC_EXTERN void petscend_(PetscErrorCode *ierr)
{
#if defined(PETSC_HAVE_SUNMATHPRO)
  extern void standard_arithmetic();
  standard_arithmetic();
#endif

  *ierr = PetscEnd();
}

