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
#define petscinitializef_             PETSCINITIALIZEF
#define petscfinalize_                PETSCFINALIZE
#define petscend_                     PETSCEND
#define iargc_                        IARGC
#define getarg_                       GETARG
#define mpi_init_                     MPI_INIT
#define petscgetcomm_                 PETSCGETCOMM
#define petsccommandargumentcount_    PETSCCOMMANDARGUMENTCOUNT
#define petscgetcommandargument_      PETSCGETCOMMANDARGUMENT
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petscinitializef_             petscinitializef
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

PETSC_EXTERN PetscErrorCode PetscMallocAlign(size_t,PetscBool,int,const char[],const char[],void**);
PETSC_EXTERN PetscErrorCode PetscFreeAlign(void*,int,const char[],const char[]);
PETSC_INTERN int  PetscGlobalArgc;
PETSC_INTERN char **PetscGlobalArgs;

/*
    Reads in Fortran command line arguments and sends them to
  all processors.
*/

PetscErrorCode PETScParseFortranArgs_Private(int *argc,char ***argv)
{
#if defined(PETSC_USE_NARGS)
  short          i,flg;
#else
  int            i;
#endif
  int            warg = 256;
  PetscMPIInt    rank;
  char           *p;

  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  if (rank == 0) {
#if defined(PETSC_HAVE_IARG_COUNT_PROGNAME)
    *argc = iargc_();
#else
    /* most compilers do not count the program name for argv[0] */
    *argc = 1 + iargc_();
#endif
  }
  PetscCallMPI(MPI_Bcast(argc,1,MPI_INT,0,PETSC_COMM_WORLD));

  /* PetscTrMalloc() not yet set, so don't use PetscMalloc() */
  PetscCall(PetscMallocAlign((*argc+1)*(warg*sizeof(char)+sizeof(char*)),PETSC_FALSE,0,0,0,(void**)argv));
  (*argv)[0] = (char*)(*argv + *argc + 1);

  if (rank == 0) {
    PetscCall(PetscMemzero((*argv)[0],(*argc)*warg*sizeof(char)));
    for (i=0; i<*argc; i++) {
      (*argv)[i+1] = (*argv)[i] + warg;
#if defined (PETSC_HAVE_FORTRAN_GET_COMMAND_ARGUMENT) /* same as 'else' case */
      getarg_(&i,(*argv)[i],warg);
#elif defined(PETSC_HAVE_PXFGETARG_NEW)
      {char *tmp = (*argv)[i];
      int ilen;
      PetscCallFortranVoidFunction(getarg_(&i,tmp,&ilen,&ierr,warg));
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
  PetscCallMPI(MPI_Bcast((*argv)[0],*argc*warg,MPI_CHAR,0,PETSC_COMM_WORLD));
  if (rank) {
    for (i=0; i<*argc; i++) (*argv)[i+1] = (*argv)[i] + warg;
  }
  return 0;
}

/* -----------------------------------------------------------------------------------------------*/

PETSC_INTERN PetscErrorCode PetscPreMPIInit_Private();

PETSC_INTERN PetscErrorCode PetscInitFortran_Private(PetscBool readarguments,const char *filename,PetscInt len)
{
  char           *tmp = NULL;

  PetscFunctionBegin;
  PetscCall(PetscInitializeFortran());
  if (readarguments) {
    PetscCall(PETScParseFortranArgs_Private(&PetscGlobalArgc,&PetscGlobalArgs));
    if (filename != PETSC_NULL_CHARACTER_Fortran) {  /* FIXCHAR */
      while ((len > 0) && (filename[len-1] == ' ')) len--;
      PetscCall(PetscMalloc1(len+1,&tmp));
      PetscCall(PetscStrncpy(tmp,filename,len+1));
    }
    PetscCall(PetscOptionsInsert(NULL,&PetscGlobalArgc,&PetscGlobalArgs,tmp));
    PetscCall(PetscFree(tmp)); /* FREECHAR */
  }
  PetscFunctionReturn(0);
}

/*
    petscinitialize - Version called from Fortran.

    Notes:
      Since this is called from Fortran it does not return error codes

*/
PETSC_EXTERN void petscinitializef_(char* filename,char* help,PetscBool *readarguments,PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T len,PETSC_FORTRAN_CHARLEN_T helplen)
{
  int            j,i;
#if defined (PETSC_USE_NARGS)
  short          flg;
#endif
  int            flag;
  char           name[256] = {0};
  PetscMPIInt    f_petsc_comm_world;

  if (PetscInitializeCalled) {*ierr = 0; return;}
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

  /* check if PETSC_COMM_WORLD is initialized by the user in fortran */
  petscgetcomm_(&f_petsc_comm_world);
  MPI_Initialized(&flag);
  if (!flag) {
    PetscMPIInt mierr;

    if (f_petsc_comm_world) {(*PetscErrorPrintf)("You cannot set PETSC_COMM_WORLD if you have not initialized MPI first\n");return;}

    *ierr = PetscPreMPIInit_Private(); if (*ierr) return;
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

  *ierr = PetscInitialize_Common(name,filename,help,PETSC_TRUE,*readarguments,(PetscInt)len);
  if (*ierr) {(*PetscErrorPrintf)("PetscInitialize:PetscInitialize_Common\n");return;}
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
