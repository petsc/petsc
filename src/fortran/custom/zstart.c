#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: zstart.c,v 1.38 1998/03/24 21:13:03 balay Exp balay $";
#endif

/*
  This file contains Fortran stubs for PetscInitialize and Finalize.
*/

/*
    This is to prevent the Cray T3D version of MPI (University of Edinburgh)
  from stupidly redefining MPI_INIT(). They put this in to detect errors
  in C code, but here I do want to be calling the Fortran version from a
  C subroutine. 
*/
#define T3DMPI_FORTRAN
#define T3EMPI_FORTRAN
#include "src/fortran/custom/zpetsc.h" 
#include "sys.h"
#include "pinclude/pviewer.h"
#include "pinclude/petscfix.h"
extern int          PetscBeganMPI;

#if defined(HAVE_NAGF90)
#define iargc_  f90_unix_MP_iargc
#define getarg_ f90_unix_MP_getarg
#endif

#ifdef HAVE_FORTRAN_CAPS
#define petscfinalize_                PETSCFINALIZE
#define petscinitialize_              PETSCINITIALIZE
#define iargc_                        IARGC
#define getarg_                       GETARG
#define mpi_init_                     MPI_INIT
#define petscinitializefortran_       PETSCINITIALIZEFORTRAN
#define petsc_null_function_          PETSC_NULL_FUNCTION
#if defined(PARCH_nt)
#define IARGC                        NARGS
#endif

#elif !defined(HAVE_FORTRAN_UNDERSCORE)
#define petscfinalize_                petscfinalize
#define petscinitialize_              petscinitialize
#define mpi_init_                     mpi_init
#define petscinitializefortran_       petscinitializefortran
#define petsc_null_function_          petsc_null_function
/*
    HP-UX does not have Fortran underscore but iargc and getarg 
  do have underscores????
*/
#if !defined(PARCH_hpux)
#define iargc_                        iargc
#define getarg_                       getarg
#endif

#endif

int OptionsCheckInitial_Private(),
    OptionsCreate_Private(int*,char***,char*),
    OptionsSetAlias_Private(char *,char *);

/*
    The extra _ is because the f2c compiler puts an
  extra _ at the end if the original routine name 
  contained any _.
*/
#if defined(HAVE_FORTRAN_UNDERSCORE_UNDERSCORE)
#define mpi_init_             mpi_init__
#define petsc_null_function_  petsc_null_function__
#endif

#if defined(__cplusplus)
extern "C" {
#endif
extern void mpi_init_(int*);
#if defined(PARCH_nt)
/*
extern short  __declspec(dllimport) __stdcall iargc_();
extern void __declspec(dllimport) __stdcall  getarg_(short*,char*,int,short *);
*/
extern short __stdcall iargc_();
extern void __stdcall  getarg_(short*,char*,int,short *);

#else
extern int  iargc_();
extern void getarg_(int*,char*,int);
#if defined(PARCH_t3d)
extern void PXFGETARG(int *,_fcd,int*,int*);
#endif
#endif
#if defined(__cplusplus)
}
#endif

/*
    Reads in Fortran command line argments and sends them to 
  all processors and adds them to Options database.
*/

int PETScParseFortranArgs_Private(int *argc,char ***argv)
{
#if defined (PARCH_nt)
  short i,flg;
#else
  int  i;
#endif
  int warg = 256,rank;
  char *p;

  MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
  if (!rank) {
    *argc = 1 + iargc_();
  }
  MPI_Bcast(argc,1,MPI_INT,0,PETSC_COMM_WORLD);

  *argv = (char **) PetscMalloc((*argc+1)*(warg*sizeof(char)+sizeof(char*)));CHKPTRQ(*argv);
  (*argv)[0] = (char*) (*argv + *argc + 1);

  if (!rank) {
    PetscMemzero((*argv)[0],(*argc)*warg*sizeof(char));
    for ( i=0; i<*argc; i++ ) {
      (*argv)[i+1] = (*argv)[i] + warg;
#if defined(PARCH_t3d)
      {char *tmp = (*argv)[i]; 
       int  ierr,ilen;
       PXFGETARG(&i, _cptofcd(tmp,warg),&ilen,&ierr); CHKERRQ(ierr);
       tmp[ilen] = 0;
      } 
#elif defined (PARCH_nt)
      getarg_( &i, (*argv)[i],warg,&flg );
#else
      getarg_( &i, (*argv)[i], warg );
#endif
      /* zero out garbage at end of each argument */
      p = (*argv)[i] + warg-1;
      while (p > (*argv)[i]) {
        if (*p == ' ') *p = 0; 
        p--;
      }
    }
  }
  MPI_Bcast((*argv)[0],*argc*warg,MPI_CHAR,0,PETSC_COMM_WORLD);  
  if (rank) {
    for ( i=0; i<*argc; i++ ) {
      (*argv)[i+1] = (*argv)[i] + warg;
    }
  } 
  return 0;   
}

extern int PetscInitializeOptions(void);
extern int PetscInitialize_DynamicLibraries(void);
extern int OptionsSetProgramName(char *);

#if defined(__cplusplus)
extern "C" {
#endif

void petscinitialize_(CHAR filename,int *__ierr,int len)
{
#if defined (PARCH_nt)
  short  flg,i;
#else
  int i;
#endif
  int  j,flag,argc = 0,dummy_tag, PETSC_COMM_WORLD_FromUser = 1;
  char **args = 0,*t1, name[256];

  *__ierr = 1;
  PetscMemzero(name,256);
  if (PetscInitializedCalled) {*__ierr = 0; return;}
  
  *__ierr = PetscInitializeOptions(); 
  if (*__ierr) return;
  i = 0;
#if defined(PARCH_t3d)
  { int ilen;
    PXFGETARG(&i, _cptofcd(name,256),&ilen,__ierr); 
    if (*__ierr) return;
    name[ilen] = 0;
  }
#elif defined (PARCH_nt)
  getarg_( &i, name, 256, &flg);
#else
  getarg_( &i, name, 256);
  /* Eliminate spaces at the end of the string */
  for ( j=254; j>=0; j-- ) {
    if (name[j] != ' ') {
      name[j+1] = 0;
      break;
    }
  }
#endif
  OptionsSetProgramName(name);

  MPI_Initialized(&flag);
  if (!flag) {
    mpi_init_(__ierr);
    if (*__ierr) {(*PetscErrorPrintf)("PetscInitialize:");return;}
    PetscBeganMPI    = 1;
  }
  PetscInitializedCalled = 1;

  if (!PETSC_COMM_WORLD) {
    PETSC_COMM_WORLD_FromUser = 0;
    PETSC_COMM_WORLD          = MPI_COMM_WORLD;
  }

#if defined(USE_PETSC_COMPLEX)
  /* 
     Initialized the global variable; this is because with 
     shared libraries the constructors for global variables
     are not called; at least on IRIX.
  */
  {
    Scalar ic(0.0,1.0);
    PETSC_i = ic;
  }
  MPI_Type_contiguous(2,MPI_DOUBLE,&MPIU_COMPLEX);
  MPI_Type_commit(&MPIU_COMPLEX);
#endif

  /*
     PetscInitializeFortran() is called twice. Here it initializes
     PETSC_NULLCHARACTOR_Fortran. Below it initializes the VIEWERs.
     The VIEWERs have not been created yet, so they must be initialized
     below.
  */
  PetscInitializeFortran();

  PETScParseFortranArgs_Private(&argc,&args);
  FIXCHAR(filename,len,t1);
  *__ierr = OptionsCreate_Private(&argc,&args,t1); 
  FREECHAR(filename,t1);
  if (*__ierr) { (*PetscErrorPrintf)("PETSC ERROR: PetscInitialize:Creating options database");return;}
  PetscFree(args);
  *__ierr = OptionsCheckInitial_Private(); 
  if (*__ierr) { (*PetscErrorPrintf)("PETSC ERROR: PetscInitialize:Checking initial options");return;}
  /*
       Initialize PETSC_COMM_SELF as a MPI_Comm with the PETSc 
     attribute.
  */
  PetscCommDup_Private(MPI_COMM_SELF,&PETSC_COMM_SELF,&dummy_tag);
   if (!PETSC_COMM_WORLD_FromUser) {
    *__ierr = PetscCommDup_Private(MPI_COMM_WORLD,&PETSC_COMM_WORLD,&dummy_tag); 
    if (*__ierr) { (*PetscErrorPrintf)("PETSC ERROR: PetscInitialize:Setting up PETSC_COMM_WORLD");return;}
  }
  *__ierr = ViewerInitialize_Private(); 
  if (*__ierr) { (*PetscErrorPrintf)("PETSC ERROR: PetscInitialize:Setting up default viewers");return;}
  PetscInitializeFortran();

  *__ierr = PetscInitialize_DynamicLibraries(); 
  if (*__ierr) return;

  if (PetscBeganMPI) {
    int size;

    MPI_Comm_size(PETSC_COMM_WORLD,&size);
    PLogInfo(0,"PetscInitialize(Fortran):PETSc successfully started: procs %d\n",size);
  }

  *__ierr = 0;
}

void petscfinalize_(int *__ierr)
{
#if defined(HAVE_SUNMATHPRO)
  extern void standard_arithmetic();
  standard_arithmetic();
#endif

  *__ierr = PetscFinalize();
}

void petscsetcommworld_(MPI_Comm *comm,int *__ierr)
{
  *__ierr = PetscSetCommWorld((MPI_Comm)PetscToPointerComm( *comm )  );
}

void petsc_null_function_(void)
{
  return;
}

#if defined(__cplusplus)
}
#endif



