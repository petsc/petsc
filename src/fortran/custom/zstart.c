#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: zstart.c,v 1.61 1999/10/04 22:51:03 balay Exp balay $";
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

extern int          PetscBeganMPI;

#if defined(PETSC_HAVE_NAGF90)
#define iargc_  f90_unix_MP_iargc
#define getarg_ f90_unix_MP_getarg
#endif

#ifdef PETSC_HAVE_FORTRAN_CAPS
#define petscinitialize_              PETSCINITIALIZE
#define petscfinalize_                PETSCFINALIZE
#define aliceinitialize_              ALICEINITIALIZE
#define alicefinalize_                ALICEFINALIZE
#define petscsetcommworld_            PETSCSETCOMMWORLD
#define iargc_                        IARGC
#define getarg_                       GETARG
#define mpi_init_                     MPI_INIT
#if defined(PARCH_win32)
#define IARGC                         NARGS
#endif

#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petscinitialize_              petscinitialize
#define petscfinalize_                petscfinalize
#define aliceinitialize_              aliceinitialize
#define alicefinalize_                alicefinalize
#define petscsetcommworld_            petscsetcommworld
#define mpi_init_                     mpi_init
/*
    HP-UX does not have Fortran underscore but iargc and getarg 
  do have underscores????
*/
#if !defined(PARCH_hpux)
#define iargc_                        iargc
#define getarg_                       getarg
#endif

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
#if defined(PARCH_win32)
/*
extern short  __declspec(dllimport) __stdcall iargc_();
extern void __declspec(dllimport) __stdcall  getarg_(short*,char*,int,short *);
*/
extern short __stdcall iargc_();
extern void __stdcall  getarg_(short*,char*,int,short *);

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


extern int OptionsCheckInitial_Alice(void);
extern int OptionsCheckInitial_Components(void);
extern int PetscInitialize_DynamicLibraries(void);

/*
    Reads in Fortran command line argments and sends them to 
  all processors and adds them to Options database.
*/

int PETScParseFortranArgs_Private(int *argc,char ***argv)
{
#if defined (PARCH_win32)
  short i,flg;
#else
  int  i;
#endif
  int warg = 256,rank,ierr;
  char *p;

  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  if (!rank) {
    *argc = 1 + iargc_();
  }
  ierr = MPI_Bcast(argc,1,MPI_INT,0,PETSC_COMM_WORLD); if (ierr) return ierr;

  *argv = (char **) PetscMalloc((*argc+1)*(warg*sizeof(char)+sizeof(char*)));CHKPTRQ(*argv);
  (*argv)[0] = (char*) (*argv + *argc + 1);

  if (!rank) {
    ierr = PetscMemzero((*argv)[0],(*argc)*warg*sizeof(char));CHKERRQ(ierr);
    for ( i=0; i<*argc; i++ ) {
      (*argv)[i+1] = (*argv)[i] + warg;
#if defined(PETSC_HAVE_PXFGETARG)
      {char *tmp = (*argv)[i]; 
       int  ierr,ilen;
       PXFGETARG(&i, _cptofcd(tmp,warg),&ilen,&ierr);CHKERRQ(ierr);
       tmp[ilen] = 0;
      } 
#elif defined (PARCH_win32)
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
  ierr = MPI_Bcast((*argv)[0],*argc*warg,MPI_CHAR,0,PETSC_COMM_WORLD); if (ierr) return ierr; 
  if (rank) {
    for ( i=0; i<*argc; i++ ) {
      (*argv)[i+1] = (*argv)[i] + warg;
    }
  } 
  return 0;   
}

EXTERN_C_BEGIN
/*
    aliceinitialize - Version called from Fortran.

    Notes:
      Since this is called from Fortran it does not return error codes
      
*/
void PETSC_STDCALL aliceinitialize_(CHAR filename PETSC_MIXED_LEN(len),int *__ierr PETSC_END_LEN(len) )
{
#if defined (PARCH_win32)
  short  flg,i;
#else
  int i;
#endif
  int  j,flag,argc = 0,dummy_tag,ierr;
  char **args = 0,*t1, name[256];

  *__ierr = 1;
  ierr = PetscMemzero(name,256); if (ierr) return;
  if (PetscInitializedCalled) {*__ierr = 0; return;}
  
  *__ierr = OptionsCreate(); 
  if (*__ierr) return;
  i = 0;
#if defined(PETSC_HAVE_PXFGETARG)
  { int ilen;
    PXFGETARG(&i, _cptofcd(name,256),&ilen,__ierr); 
    if (*__ierr) return;
    name[ilen] = 0;
  }
#elif defined (PARCH_win32)
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
  PetscSetProgramName(name);

  MPI_Initialized(&flag);
  if (!flag) {
    mpi_init_(__ierr);
    if (*__ierr) {(*PetscErrorPrintf)("PetscInitialize:");return;}
    PetscBeganMPI    = 1;
  }
  PetscInitializedCalled = 1;

  if (!PETSC_COMM_WORLD) {
    PETSC_COMM_WORLD          = MPI_COMM_WORLD;
  }

#if defined(PETSC_USE_COMPLEX)
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
  *__ierr = OptionsInsert(&argc,&args,t1); 
  FREECHAR(filename,t1);
  if (*__ierr) { (*PetscErrorPrintf)("PETSC ERROR: PetscInitialize:Creating options database");return;}
  PetscFree(args);
  *__ierr = OptionsCheckInitial_Alice(); 
  if (*__ierr) { (*PetscErrorPrintf)("PETSC ERROR: PetscInitialize:Checking initial options");return;}

  /*
       Initialize PETSC_COMM_SELF as a MPI_Comm with the PETSc 
     attribute.
  */
  *__ierr = PetscCommDuplicate_Private(MPI_COMM_SELF,&PETSC_COMM_SELF,&dummy_tag);
  if (*__ierr) { (*PetscErrorPrintf)("PETSC ERROR: PetscInitialize:Setting up PETSC_COMM_SELF");return;}
  *__ierr = PetscCommDuplicate_Private(PETSC_COMM_WORLD,&PETSC_COMM_WORLD,&dummy_tag); 
  if (*__ierr) { (*PetscErrorPrintf)("PETSC ERROR: PetscInitialize:Setting up PETSC_COMM_WORLD");return;}
  *__ierr = PetscInitialize_DynamicLibraries(); 
  if (*__ierr) {(*PetscErrorPrintf)("PETSC ERROR: PetscInitialize:Initializing dynamic libraries");return;}

  *__ierr = ViewerInitializeASCII_Private(); 
  if (*__ierr) { (*PetscErrorPrintf)("PETSC ERROR: PetscInitialize:Setting up default viewers");return;}
  PetscInitializeFortran();

  if (PetscBeganMPI) {
    int size;

    *__ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);
    if (*__ierr) { (*PetscErrorPrintf)("PETSC ERROR: PetscInitialize:Getting MPI_Comm_size()");return;}
    PLogInfo(0,"PetscInitialize(Fortran):PETSc successfully started: procs %d\n",size);
  }

  *__ierr = 0;
}

void PETSC_STDCALL alicefinalize_(int *__ierr)
{
#if defined(PETSC_HAVE_SUNMATHPRO)
  extern void standard_arithmetic();
  standard_arithmetic();
#endif

  *__ierr = AliceFinalize();
}
EXTERN_C_END

/* -----------------------------------------------------------------------------------------------*/


EXTERN_C_BEGIN
/*
    petscinitialize - Version called from Fortran.

    Notes:
      Since this is called from Fortran it does not return error codes
      
*/
void PETSC_STDCALL petscinitialize_(CHAR filename PETSC_MIXED_LEN(len),int *__ierr PETSC_END_LEN(len) )
{
#if defined(PETSC_USE_FORTRAN_MIXED_STR_ARG)
  aliceinitialize_(filename,len,__ierr); 
#else
  aliceinitialize_(filename,__ierr,len); 
#endif
  if (*__ierr) return;
  
  *__ierr = OptionsCheckInitial_Components(); 
  if (*__ierr) {(*PetscErrorPrintf)("PETSC ERROR: PetscInitialize:Checking initial options");return;}

}

void PETSC_STDCALL petscfinalize_(int *__ierr)
{
#if defined(PETSC_HAVE_SUNMATHPRO)
  extern void standard_arithmetic();
  standard_arithmetic();
#endif

  *__ierr = PetscFinalize();
}

void PETSC_STDCALL petscsetcommworld_(MPI_Comm *comm,int *__ierr)
{
  *__ierr = PetscSetCommWorld((MPI_Comm)PetscToPointerComm( *comm )  );
}
EXTERN_C_END
