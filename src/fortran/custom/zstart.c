/*$Id: zstart.c,v 1.84 2001/08/10 03:35:41 bsmith Exp $*/

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

#include "src/fortran/custom/zpetsc.h" 
#include "petscsys.h"

extern PetscTruth PetscBeganMPI;

#ifdef PETSC_HAVE_FORTRAN_CAPS
#define petscinitialize_              PETSCINITIALIZE
#define petscfinalize_                PETSCFINALIZE
#define petscend_                     PETSCEND
#define petscsetcommworld_            PETSCSETCOMMWORLD
#define iargc_                        IARGC
#define getarg_                       GETARG
#define mpi_init_                     MPI_INIT
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petscinitialize_              petscinitialize
#define petscfinalize_                petscfinalize
#define petscend_                     petscend
#define petscsetcommworld_            petscsetcommworld
#define mpi_init_                     mpi_init
#define iargc_                        iargc
#define getarg_                       getarg
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
#endif
#if defined(PETSC_HAVE_FORTRAN_IARGC_UNDERSCORE) /* HPUX + no underscore */
#undef iargc_
#undef getarg_
#define iargc   iargc_
#define getarg  getarg_
#endif

/*
    The extra _ is because the f2c compiler puts an
  extra _ at the end if the original routine name 
  contained any _.
*/
#if defined(PETSC_HAVE_FORTRAN_UNDERSCORE_UNDERSCORE)
#define mpi_init_             mpi_init__
#endif

EXTERN_C_BEGIN
extern void PETSC_STDCALL mpi_init_(int*);

/*
     Different Fortran compilers handle command lines in different ways
*/
#if defined(PETSC_USE_NARGS)
extern short __stdcall NARGS();
extern void  __stdcall GETARG(short*,char*,int,short *);

#else
extern int  iargc_();
extern void getarg_(int*,char*,int);
/*
      The Cray T3D/T3E use the PXFGETARG() function
*/
#if defined(PETSC_HAVE_PXFGETARG)
extern void PXFGETARG(int *,_fcd,int*,int*);
#endif
#endif
EXTERN_C_END

#if defined(PETSC_USE_COMPLEX)
extern MPI_Op PetscSum_Op;

EXTERN_C_BEGIN
extern void PetscSum_Local(void *,void *,int *,MPI_Datatype *);
EXTERN_C_END
#endif
extern MPI_Op PetscMaxSum_Op;

EXTERN_C_BEGIN
extern void PetscMaxSum_Local(void *,void *,int *,MPI_Datatype *);
EXTERN_C_END

EXTERN int PetscOptionsCheckInitial(void);
EXTERN int PetscOptionsCheckInitial_Components(void);
EXTERN int PetscInitialize_DynamicLibraries(void);
EXTERN int PetscLogBegin_Private(void);

/*
    Reads in Fortran command line argments and sends them to 
  all processors and adds them to Options database.
*/

int PETScParseFortranArgs_Private(int *argc,char ***argv)
{
#if defined (PETSC_USE_NARGS)
  short i,flg;
#else
  int  i;
#endif
  int warg = 256,rank,ierr;
  char *p;

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

  ierr = PetscMalloc((*argc+1)*(warg*sizeof(char)+sizeof(char*)),argv);CHKERRQ(ierr);
  (*argv)[0] = (char*)(*argv + *argc + 1);

  if (!rank) {
    ierr = PetscMemzero((*argv)[0],(*argc)*warg*sizeof(char));CHKERRQ(ierr);
    for (i=0; i<*argc; i++) {
      (*argv)[i+1] = (*argv)[i] + warg;
#if defined(PETSC_HAVE_PXFGETARG)
      {char *tmp = (*argv)[i]; 
       int  ierr,ilen;
       PXFGETARG(&i,_cptofcd(tmp,warg),&ilen,&ierr);CHKERRQ(ierr);
       tmp[ilen] = 0;
      } 
#elif defined (PETSC_USE_NARGS)
      GETARG(&i,(*argv)[i],warg,&flg);
#else
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


EXTERN_C_BEGIN
/*
    petscinitialize - Version called from Fortran.

    Notes:
      Since this is called from Fortran it does not return error codes
      
*/
void PETSC_STDCALL petscinitialize_(CHAR filename PETSC_MIXED_LEN(len),int *ierr PETSC_END_LEN(len))
{
#if defined (PETSC_USE_NARGS)
  short flg,i;
#else
  int   i;
#endif
  int   j,flag,argc = 0,dummy_tag,size;
  char  **args = 0,*t1,name[256],hostname[64];
  
  *ierr = 1;
  *ierr = PetscMemzero(name,256); if (*ierr) return;
  if (PetscInitializeCalled) {*ierr = 0; return;}
  
  *ierr = PetscOptionsCreate(); 
  if (*ierr) return;
  i = 0;
#if defined(PETSC_HAVE_PXFGETARG)
  { int ilen;
    PXFGETARG(&i,_cptofcd(name,256),&ilen,ierr); 
    if (*ierr) return;
    name[ilen] = 0;
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
#endif
  *ierr = PetscSetProgramName(name);
  if (*ierr) {(*PetscErrorPrintf)("PETSC ERROR: PetscInitialize: Calling PetscSetProgramName()");return;}
  *ierr = PetscSetInitialDate();
  if (*ierr) {(*PetscErrorPrintf)("PETSC ERROR: PetscInitialize: Calling PetscSetInitialDate()");return;}

  MPI_Initialized(&flag);
  if (!flag) {
    mpi_init_(ierr);
    if (*ierr) {(*PetscErrorPrintf)("PetscInitialize:");return;}
    PetscBeganMPI    = PETSC_TRUE;
  }
  PetscInitializeCalled = PETSC_TRUE;

  if (!PETSC_COMM_WORLD) {
    PETSC_COMM_WORLD          = MPI_COMM_WORLD;
  }

  *ierr = MPI_Comm_rank(MPI_COMM_WORLD,&PetscGlobalRank);
  if (*ierr) {(*PetscErrorPrintf)("PETSC ERROR: PetscInitialize: Setting PetscGlobalRank");return;}
  *ierr = MPI_Comm_size(MPI_COMM_WORLD,&PetscGlobalSize);
  if (*ierr) {(*PetscErrorPrintf)("PETSC ERROR: PetscInitialize: Setting PetscGlobalSize");return;}

#if defined(PETSC_USE_COMPLEX)
  /* 
     Initialized the global variable; this is because with 
     shared libraries the constructors for global variables
     are not called; at least on IRIX.
  */
  {
    PetscScalar ic(0.0,1.0);
    PETSC_i = ic;
  }
  MPI_Type_contiguous(2,MPIU_REAL,&MPIU_COMPLEX);
  MPI_Type_commit(&MPIU_COMPLEX);
  *ierr = MPI_Op_create(PetscSum_Local,1,&PetscSum_Op);
  if (*ierr) {(*PetscErrorPrintf)("PETSC ERROR: PetscInitialize:Creating MPI ops");return;}
#endif

  /*
       Create the PETSc MPI reduction operator that sums of the first
     half of the entries and maxes the second half.
  */
  *ierr = MPI_Op_create(PetscMaxSum_Local,1,&PetscMaxSum_Op);
  if (*ierr) {(*PetscErrorPrintf)("PETSC ERROR: PetscInitialize:Creating MPI ops");return;}

  /*
     PetscInitializeFortran() is called twice. Here it initializes
     PETSC_NULLCHARACTER_Fortran. Below it initializes the PETSC_VIEWERs.
     The PETSC_VIEWERs have not been created yet, so they must be initialized
     below.
  */
  PetscInitializeFortran();

  PETScParseFortranArgs_Private(&argc,&args);
  FIXCHAR(filename,len,t1);
  *ierr = PetscOptionsInsert(&argc,&args,t1); 
  FREECHAR(filename,t1);
  if (*ierr) {(*PetscErrorPrintf)("PETSC ERROR: PetscInitialize:Creating options database");return;}
  *ierr = PetscFree(args);
  if (*ierr) {(*PetscErrorPrintf)("PETSC ERROR: PetscInitialize:Freeing args");return;}
  *ierr = PetscOptionsCheckInitial(); 
  if (*ierr) {(*PetscErrorPrintf)("PETSC ERROR: PetscInitialize:Checking initial options");return;}
  *ierr = PetscLogBegin_Private();
  if (*ierr) {(*PetscErrorPrintf)("PETSC ERROR: PetscInitialize: intializing logging");return;}
  /*
       Initialize PETSC_COMM_SELF as a MPI_Comm with the PETSc attribute.
  */
  *ierr = PetscCommDuplicate_Private(MPI_COMM_SELF,&PETSC_COMM_SELF,&dummy_tag);
  if (*ierr) { (*PetscErrorPrintf)("PETSC ERROR: PetscInitialize:Setting up PETSC_COMM_SELF");return;}
  *ierr = PetscCommDuplicate_Private(PETSC_COMM_WORLD,&PETSC_COMM_WORLD,&dummy_tag); 
  if (*ierr) { (*PetscErrorPrintf)("PETSC ERROR: PetscInitialize:Setting up PETSC_COMM_WORLD");return;}
  *ierr = PetscInitialize_DynamicLibraries(); 
  if (*ierr) {(*PetscErrorPrintf)("PETSC ERROR: PetscInitialize:Initializing dynamic libraries");return;}

  *ierr = PetscInitializeFortran();
  if (*ierr) { (*PetscErrorPrintf)("PETSC ERROR: PetscInitialize:Setting up common block");return;}

  *ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);
  if (*ierr) { (*PetscErrorPrintf)("PETSC ERROR: PetscInitialize:Getting MPI_Comm_size()");return;}
  PetscLogInfo(0,"PetscInitialize(Fortran):PETSc successfully started: procs %d\n",size);
  *ierr = PetscGetHostName(hostname,64);
  if (*ierr) { (*PetscErrorPrintf)("PETSC ERROR: PetscInitialize:Getting hostname");return;}
  PetscLogInfo(0,"Running on machine: %s\n",hostname);
  
  *ierr = PetscOptionsCheckInitial_Components(); 
  if (*ierr) {(*PetscErrorPrintf)("PETSC ERROR: PetscInitialize:Checking initial options");return;}

}

void PETSC_STDCALL petscfinalize_(int *ierr)
{
#if defined(PETSC_HAVE_SUNMATHPRO)
  extern void standard_arithmetic();
  standard_arithmetic();
#endif

  *ierr = PetscFinalize();
}

void PETSC_STDCALL petscend_(int *ierr)
{
#if defined(PETSC_HAVE_SUNMATHPRO)
  extern void standard_arithmetic();
  standard_arithmetic();
#endif

  *ierr = PetscEnd();
}

void PETSC_STDCALL petscsetcommworld_(MPI_Comm *comm,int *ierr)
{
  *ierr = PetscSetCommWorld((MPI_Comm)PetscToPointerComm(*comm));
}
EXTERN_C_END
