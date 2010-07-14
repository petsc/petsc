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

#include "private/fortranimpl.h" 

extern PETSC_DLL_IMPORT PetscTruth PetscBeganMPI;

#ifdef PETSC_HAVE_FORTRAN_CAPS
#define petscinitialize_              PETSCINITIALIZE
#define petscfinalize_                PETSCFINALIZE
#define petscend_                     PETSCEND
#define iargc_                        IARGC
#define getarg_                       GETARG
#define mpi_init_                     MPI_INIT
#define petscgetcommoncomm_           PETSCGETCOMMONCOMM
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petscinitialize_              petscinitialize
#define petscfinalize_                petscfinalize
#define petscend_                     petscend
#define mpi_init_                     mpi_init
#define iargc_                        iargc
#define getarg_                       getarg
#define petscgetcommoncomm_           petscgetcommoncomm
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
#elif defined (PETSC_HAVE_PXFGETARG_NEW) /* cray x1 */
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
#if defined(PETSC_HAVE_GFORTRAN_IARGC) /* gfortran from gcc4 */
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
#undef mpi_init_
#define mpi_init_             mpi_init__
#endif

EXTERN_C_BEGIN
extern void PETSC_STDCALL mpi_init_(int*);
extern void PETSC_STDCALL petscgetcommoncomm_(PetscMPIInt*);

/*
     Different Fortran compilers handle command lines in different ways
*/
#if defined(PETSC_USE_NARGS)
extern short __stdcall NARGS();
extern void  __stdcall GETARG(short*,char*,int,short *);

#elif defined(PETSC_HAVE_FORTRAN_STDCALL)
extern int  PETSC_STDCALL IARGC();
extern void PETSC_STDCALL GETARG(int *,char *,int);

#elif defined (PETSC_HAVE_PXFGETARG_NEW)
extern int  iargc_();
extern void getarg_(int*,char*,int*,int*,int);

#else
extern int  iargc_();
extern void getarg_(int*,char*,int);
/*
      The Cray T3D/T3E use the PXFGETARG() function
*/
#if defined(PETSC_HAVE_PXFGETARG)
extern void PXFGETARG(int*,_fcd,int*,int*);
#endif
#endif
EXTERN_C_END

#if defined(PETSC_USE_COMPLEX) && !defined(PETSC_HAVE_MPI_C_DOUBLE_COMPLEX)
extern MPI_Op MPIU_SUM;
EXTERN_C_BEGIN
extern void PETSC_DLLEXPORT MPIAPI PetscSum_Local(void*,void *,PetscMPIInt *,MPI_Datatype *);
EXTERN_C_END
#endif
extern PETSC_DLL_IMPORT MPI_Op PetscMaxSum_Op;

EXTERN_C_BEGIN
extern void PETSC_DLLEXPORT MPIAPI PetscMaxSum_Local(void*,void *,PetscMPIInt *,MPI_Datatype *);
extern PetscMPIInt PETSC_DLLEXPORT MPIAPI Petsc_DelCounter(MPI_Comm,PetscMPIInt,void*,void*);
extern PetscMPIInt PETSC_DLLEXPORT MPIAPI Petsc_DelComm(MPI_Comm,PetscMPIInt,void*,void*);
EXTERN_C_END

EXTERN PetscErrorCode PETSC_DLL_IMPORT PetscOptionsCheckInitial_Private(void);
EXTERN PetscErrorCode PETSC_DLL_IMPORT PetscOptionsCheckInitial_Components(void);
EXTERN PetscErrorCode PETSC_DLL_IMPORT PetscInitialize_DynamicLibraries(void);
#if defined(PETSC_USE_LOG)
EXTERN PetscErrorCode PETSC_DLL_IMPORT PetscLogBegin_Private(void);
#endif
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscMallocAlign(size_t,int,const char[],const char[],const char[],void**);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscFreeAlign(void*,int,const char[],const char[],const char[]);
extern int PetscGlobalArgc;
extern char **PetscGlobalArgs;

/*
    Reads in Fortran command line argments and sends them to 
  all processors and adds them to Options database.
*/

PetscErrorCode PETScParseFortranArgs_Private(int *argc,char ***argv)
{
#if defined (PETSC_USE_NARGS)
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
#if defined (PETSC_HAVE_IARG_COUNT_PROGNAME)
    *argc = iargc_();
#else
    /* most compilers do not count the program name for argv[0] */
    *argc = 1 + iargc_();
#endif
  }
  ierr = MPI_Bcast(argc,1,MPI_INT,0,PETSC_COMM_WORLD);CHKERRQ(ierr);

  /* PetscTrMalloc() not yet set, so don't use PetscMalloc() */
  ierr = PetscMallocAlign((*argc+1)*(warg*sizeof(char)+sizeof(char*)),0,0,0,0,(void**)argv);CHKERRQ(ierr);
  (*argv)[0] = (char*)(*argv + *argc + 1);

  if (!rank) {
    ierr = PetscMemzero((*argv)[0],(*argc)*warg*sizeof(char));CHKERRQ(ierr);
    for (i=0; i<*argc; i++) {
      (*argv)[i+1] = (*argv)[i] + warg;
#if defined (PETSC_HAVE_PXFGETARG_NEW)
      {char *tmp = (*argv)[i];
      int ilen;
      getarg_(&i,tmp,&ilen,&ierr,warg);CHKERRQ(ierr);
      tmp[ilen] = 0;
      }
#elif defined (PETSC_USE_NARGS)
      GETARG(&i,(*argv)[i],warg,&flg);
#else
      /*
      Because the stupid #defines above define all kinds of things to getarg_ we cannot do this test 
      #elif defined(PETSC_HAVE_GETARG) 
      getarg_(&i,(*argv)[i],warg);
      #else
         SETERRQ(PETSC_ERR_SUP,"Cannot get Fortran command line arguments");
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
    for (i=0; i<*argc; i++) {
      (*argv)[i+1] = (*argv)[i] + warg;
    }
  } 
  return 0;   
}

/* -----------------------------------------------------------------------------------------------*/

extern PETSC_DLL_IMPORT MPI_Op PetscADMax_Op;
extern PETSC_DLL_IMPORT MPI_Op PetscADMin_Op;
EXTERN_C_BEGIN
extern void PETSC_DLL_IMPORT MPIAPI PetscADMax_Local(void *,void *,PetscMPIInt *,MPI_Datatype *);
extern void PETSC_DLL_IMPORT MPIAPI PetscADMin_Local(void *,void *,PetscMPIInt *,MPI_Datatype *);
EXTERN_C_END


EXTERN_C_BEGIN
/*
    petscinitialize - Version called from Fortran.

    Notes:
      Since this is called from Fortran it does not return error codes
      
*/
void PETSC_STDCALL petscinitialize_(CHAR filename PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
#if defined (PETSC_USE_NARGS)
  short       flg,i;
#else
  int         i;
#if !defined(PETSC_HAVE_PXFGETARG_NEW) && !defined (PETSC_HAVE_PXFGETARG_NEW) 
  int         j;
#endif
#endif
  int         flag;
  PetscMPIInt size;
  char        *t1,name[256],hostname[64];
  PetscMPIInt f_petsc_comm_world;

  *ierr = PetscMemzero(name,256); if (*ierr) return;
  if (PetscInitializeCalled) {*ierr = 0; return;}

  /* this must be initialized in a routine, not as a constant declaration*/
  PETSC_STDOUT = stdout;
  PETSC_STDERR = stderr;
  
  *ierr = PetscOptionsCreate(); 
  if (*ierr) return;
  i = 0;
#if defined (PETSC_HAVE_PXFGETARG_NEW)
  { int ilen,sierr;
    getarg_(&i,name,&ilen,&sierr,256);
    if (sierr) {
      PetscStrncpy(name,"Unknown Name",256);
    } else {
      name[ilen] = 0;
    }
  }
#elif defined (PETSC_USE_NARGS)
  GETARG(&i,name,256,&flg);
#else
  getarg_(&i,name,256);
  /* Eliminate spaces at the end of the string */
  for (j=254; j>=0; j--) {
    if (name[j] != ' ') {
      name[j+1] = 0;
      break;
    }
  }
  if (j<0) {
    PetscStrncpy(name,"Unknown Name",256);
  }
#endif
  *ierr = PetscSetProgramName(name);
  if (*ierr) {(*PetscErrorPrintf)("PetscInitialize: Calling PetscSetProgramName()");return;}

  /* check if PETSC_COMM_WORLD is initialized by the user in fortran */
  petscgetcommoncomm_(&f_petsc_comm_world);
  MPI_Initialized(&flag);
  if (!flag) {
    PetscMPIInt mierr;

    if (f_petsc_comm_world) {(*PetscErrorPrintf)("You cannot set PETSC_COMM_WORLD if you have not initialized MPI first\n");return;}
    /* MPI requires calling Fortran mpi_init() if main program is Fortran */
    mpi_init_(&mierr);
    if (mierr) {
      *ierr = mierr;
      (*PetscErrorPrintf)("PetscInitialize: Calling Fortran MPI_Init()");
      return;
    }
    PetscBeganMPI    = PETSC_TRUE;
  } 
  if (f_petsc_comm_world) { /* User called MPI_INITIALIZE() and changed PETSC_COMM_WORLD */
    PETSC_COMM_WORLD = MPI_Comm_f2c(*(MPI_Fint *)&f_petsc_comm_world);
  } else {
    PETSC_COMM_WORLD = MPI_COMM_WORLD;
  }
  PetscInitializeCalled = PETSC_TRUE;

  *ierr = PetscErrorPrintfInitialize();
  if (*ierr) {(*PetscErrorPrintf)("PetscInitialize: Calling PetscErrorPrintfInitialize()");return;}
  *ierr = MPI_Comm_rank(MPI_COMM_WORLD,&PetscGlobalRank);
  if (*ierr) {(*PetscErrorPrintf)("PetscInitialize: Setting PetscGlobalRank");return;}
  *ierr = MPI_Comm_size(MPI_COMM_WORLD,&PetscGlobalSize);
  if (*ierr) {(*PetscErrorPrintf)("PetscInitialize: Setting PetscGlobalSize");return;}
#if defined(PETSC_USE_COMPLEX)
  /* 
     Initialized the global variable; this is because with 
     shared libraries the constructors for global variables
     are not called; at least on IRIX.
  */
  {
#if defined(PETSC_CLANGUAGE_CXX)
    PetscScalar ic(0.0,1.0);
    PETSC_i = ic;
#else
    PetscScalar ic;
    ic = 1.0*I;
    PETSC_i = ic;
#endif
  }

#if !defined(PETSC_HAVE_MPI_C_DOUBLE_COMPLEX)
  *ierr = MPI_Type_contiguous(2,MPIU_REAL,&MPI_C_DOUBLE_COMPLEX);
  if (*ierr) {(*PetscErrorPrintf)("PetscInitialize:Creating MPI types");return;}
  *ierr = MPI_Type_commit(&MPI_C_DOUBLE_COMPLEX);
  if (*ierr) {(*PetscErrorPrintf)("PetscInitialize:Creating MPI types");return;}
  *ierr = MPI_Type_contiguous(2,MPI_FLOAT,&MPI_C_COMPLEX);
  if (*ierr) {(*PetscErrorPrintf)("PetscInitialize:Creating MPI types");return;}
  *ierr = MPI_Type_commit(&MPI_C_COMPLEX);
  if (*ierr) {(*PetscErrorPrintf)("PetscInitialize:Creating MPI types");return;}
  *ierr = MPI_Op_create(PetscSum_Local,1,&MPIU_SUM);
  if (*ierr) {(*PetscErrorPrintf)("PetscInitialize:Creating MPI ops");return;}
#endif

#endif
  /*
       Create the PETSc MPI reduction operator that sums of the first
     half of the entries and maxes the second half.
  */
  *ierr = MPI_Op_create(PetscMaxSum_Local,1,&PetscMaxSum_Op);
  if (*ierr) {(*PetscErrorPrintf)("PetscInitialize:Creating MPI ops");return;}

  *ierr = MPI_Type_contiguous(2,MPIU_SCALAR,&MPIU_2SCALAR);
  if (*ierr) {(*PetscErrorPrintf)("PetscInitialize:Creating MPI types");return;}
  *ierr = MPI_Type_commit(&MPIU_2SCALAR);
  if (*ierr) {(*PetscErrorPrintf)("PetscInitialize:Creating MPI types");return;}
  *ierr = MPI_Type_contiguous(2,MPIU_INT,&MPIU_2INT);
  if (*ierr) {(*PetscErrorPrintf)("PetscInitialize:Creating MPI types");return;}
  *ierr = MPI_Type_commit(&MPIU_2INT);
  if (*ierr) {(*PetscErrorPrintf)("PetscInitialize:Creating MPI types");return;}
  *ierr = MPI_Op_create(PetscADMax_Local,1,&PetscADMax_Op);
  if (*ierr) {(*PetscErrorPrintf)("PetscInitialize:Creating MPI ops");return;}
  *ierr = MPI_Op_create(PetscADMin_Local,1,&PetscADMin_Op);
  if (*ierr) {(*PetscErrorPrintf)("PetscInitialize:Creating MPI ops");return;}
  *ierr = MPI_Keyval_create(MPI_NULL_COPY_FN,Petsc_DelCounter,&Petsc_Counter_keyval,(void*)0);
  if (*ierr) {(*PetscErrorPrintf)("PetscInitialize:Creating MPI keyvals");return;}
  *ierr = MPI_Keyval_create(MPI_NULL_COPY_FN,Petsc_DelComm,&Petsc_InnerComm_keyval,(void*)0);
  if (*ierr) {(*PetscErrorPrintf)("PetscInitialize:Creating MPI keyvals");return;}
  *ierr = MPI_Keyval_create(MPI_NULL_COPY_FN,Petsc_DelComm,&Petsc_OuterComm_keyval,(void*)0);
  if (*ierr) {(*PetscErrorPrintf)("PetscInitialize:Creating MPI keyvals");return;}

  /*
     PetscInitializeFortran() is called twice. Here it initializes
     PETSC_NULL_CHARACTER_Fortran. Below it initializes the PETSC_VIEWERs.
     The PETSC_VIEWERs have not been created yet, so they must be initialized
     below.
  */
  PetscInitializeFortran();
  PETScParseFortranArgs_Private(&PetscGlobalArgc,&PetscGlobalArgs);
  FIXCHAR(filename,len,t1);
  *ierr = PetscOptionsInsert(&PetscGlobalArgc,&PetscGlobalArgs,t1); 
  FREECHAR(filename,t1);
  if (*ierr) {(*PetscErrorPrintf)("PetscInitialize:Creating options database");return;}
  *ierr = PetscOptionsCheckInitial_Private(); 
  if (*ierr) {(*PetscErrorPrintf)("PetscInitialize:Checking initial options");return;}
#if defined (PETSC_USE_LOG)
  *ierr = PetscLogBegin_Private();
  if (*ierr) {(*PetscErrorPrintf)("PetscInitialize: intializing logging");return;}
#endif
  *ierr = PetscInitialize_DynamicLibraries(); 
  if (*ierr) {(*PetscErrorPrintf)("PetscInitialize:Initializing dynamic libraries");return;}

  *ierr = PetscInitializeFortran();
  if (*ierr) { (*PetscErrorPrintf)("PetscInitialize:Setting up common block");return;}

  *ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);
  if (*ierr) { (*PetscErrorPrintf)("PetscInitialize:Getting MPI_Comm_size()");return;}
  *ierr = PetscInfo1(0,"(Fortran):PETSc successfully started: procs %d\n",size);
  if (*ierr) { (*PetscErrorPrintf)("PetscInitialize:Calling PetscInfo()");return;}
  *ierr = PetscGetHostName(hostname,64);
  if (*ierr) { (*PetscErrorPrintf)("PetscInitialize:Getting hostname");return;}
  *ierr = PetscInfo1(0,"Running on machine: %s\n",hostname);
  if (*ierr) { (*PetscErrorPrintf)("PetscInitialize:Calling PetscInfo()");return;}  
  *ierr = PetscOptionsCheckInitial_Components(); 
  if (*ierr) {(*PetscErrorPrintf)("PetscInitialize:Checking initial options");return;}
}

void PETSC_STDCALL petscfinalize_(PetscErrorCode *ierr)
{
#if defined(PETSC_HAVE_SUNMATHPRO)
  extern void standard_arithmetic();
  standard_arithmetic();
#endif
  /* was malloced with PetscMallocAlign() so free the same way */
  *ierr = PetscFreeAlign(PetscGlobalArgs,0,0,0,0);if (*ierr) {(*PetscErrorPrintf)("PetscFinalize:Freeing args");return;}

  *ierr = PetscFinalize();
}

void PETSC_STDCALL petscend_(PetscErrorCode *ierr)
{
#if defined(PETSC_HAVE_SUNMATHPRO)
  extern void standard_arithmetic();
  standard_arithmetic();
#endif

  *ierr = PetscEnd();
}


EXTERN_C_END
