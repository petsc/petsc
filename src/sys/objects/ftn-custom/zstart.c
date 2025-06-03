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

#include <petsc/private/ftnimpl.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define petscinitializef_          PETSCINITIALIZEF
  #define mpi_init_                  MPI_INIT
  #define petscgetcomm_              PETSCGETCOMM
  #define petsccommandargumentcount_ PETSCCOMMANDARGUMENTCOUNT
  #define petscgetcommandargument_   PETSCGETCOMMANDARGUMENT
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define petscinitializef_          petscinitializef
  #define mpi_init_                  mpi_init
  #define petscgetcomm_              petscgetcomm
  #define petsccommandargumentcount_ petsccommandargumentcount
  #define petscgetcommandargument_   petscgetcommandargument
#endif

/*
    The extra _ is because the f2c compiler puts an
  extra _ at the end if the original routine name
  contained any _.
*/
#if defined(PETSC_HAVE_FORTRAN_UNDERSCORE_UNDERSCORE)
  #define mpi_init_ mpi_init__
#endif

#if defined(PETSC_HAVE_MPIUNI)
  #if defined(mpi_init_)
    #undef mpi_init_
    #if defined(PETSC_HAVE_FORTRAN_CAPS)
      #define mpi_init_ PETSC_MPI_INIT
    #elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
      #define mpi_init_ petsc_mpi_init
    #elif defined(PETSC_HAVE_FORTRAN_UNDERSCORE_UNDERSCORE)
      #define mpi_init_ petsc_mpi_init__
    #endif
  #else /* mpi_init_ */
    #define mpi_init_ petsc_mpi_init_
  #endif /* mpi_init_ */
#endif   /* PETSC_HAVE_MPIUNI */

PETSC_EXTERN void mpi_init_(int *);
PETSC_EXTERN void petscgetcomm_(PetscMPIInt *);

/*
     Different Fortran compilers handle command lines in different ways
*/
PETSC_EXTERN int            petsccommandargumentcount_(void);
PETSC_EXTERN void           petscgetcommandargument_(int *, char *, PETSC_FORTRAN_CHARLEN_T);
PETSC_EXTERN PetscErrorCode PetscMallocAlign(size_t, PetscBool, int, const char[], const char[], void **);
PETSC_EXTERN PetscErrorCode PetscFreeAlign(void *, int, const char[], const char[]);
PETSC_INTERN int            PetscGlobalArgc;
PETSC_INTERN char         **PetscGlobalArgs, **PetscGlobalArgsFortran;

/*
    Reads in Fortran command line arguments and sends them to
  all processors.
*/

PetscErrorCode PETScParseFortranArgs_Private(int *argc, char ***argv)
{
  int         i;
  int         warg = 256;
  PetscMPIInt rank;
  char       *p;

  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  if (rank == 0) *argc = 1 + petsccommandargumentcount_();
  PetscCallMPI(MPI_Bcast(argc, 1, MPI_INT, 0, PETSC_COMM_WORLD));

  /* PetscTrMalloc() not yet set, so don't use PetscMalloc() */
  PetscCall(PetscMallocAlign((*argc + 1) * (warg * sizeof(char) + sizeof(char *)), PETSC_FALSE, 0, NULL, NULL, (void **)argv));
  (*argv)[0] = (char *)(*argv + *argc + 1);

  if (rank == 0) {
    PetscCall(PetscMemzero((*argv)[0], (*argc) * warg * sizeof(char)));
    for (i = 0; i < *argc; i++) {
      (*argv)[i + 1] = (*argv)[i] + warg;
      petscgetcommandargument_(&i, (*argv)[i], warg);
      /* zero out garbage at end of each argument */
      p = (*argv)[i] + warg - 1;
      while (p > (*argv)[i]) {
        if (*p == ' ') *p = 0;
        p--;
      }
    }
  }
  PetscCallMPI(MPI_Bcast((*argv)[0], *argc * warg, MPI_CHAR, 0, PETSC_COMM_WORLD));
  if (rank) {
    for (i = 0; i < *argc; i++) (*argv)[i + 1] = (*argv)[i] + warg;
  }
  return PETSC_SUCCESS;
}

/* -----------------------------------------------------------------------------------------------*/

PETSC_INTERN PetscErrorCode PetscPreMPIInit_Private(void);

PETSC_INTERN PetscErrorCode PetscInitFortran_Private(const char *filename, PetscInt len)
{
  char *tmp = NULL;

  PetscFunctionBegin;
  PetscCall(PetscInitializeFortran());
  PetscCall(PETScParseFortranArgs_Private(&PetscGlobalArgc, &PetscGlobalArgsFortran));
  PetscGlobalArgs = PetscGlobalArgsFortran;
  if (filename != PETSC_NULL_CHARACTER_Fortran) { /* filename comes from Fortran so may have blanking padding that needs removal */
    while ((len > 0) && (filename[len - 1] == ' ')) len--;
    PetscCall(PetscMalloc1(len + 1, &tmp));
    PetscCall(PetscStrncpy(tmp, filename, len + 1));
  }
  PetscCall(PetscOptionsInsert(NULL, &PetscGlobalArgc, &PetscGlobalArgsFortran, tmp));
  PetscCall(PetscFree(tmp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_EXTERN void petscinitializef_(char *filename, char *help, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T len, PETSC_FORTRAN_CHARLEN_T helplen)
{
  int         j, i;
  int         flag;
  char        name[256] = {0};
  PetscMPIInt f_petsc_comm_world;

  *ierr = PETSC_SUCCESS;
  if (PetscInitializeCalled) return;
  i = 0;
  petscgetcommandargument_(&i, name, sizeof(name));
  /* Eliminate spaces at the end of the string */
  for (j = sizeof(name) - 2; j >= 0; j--) {
    if (name[j] != ' ') {
      name[j + 1] = 0;
      break;
    }
  }
  if (j < 0) {
    *ierr = PetscStrncpy(name, "Unknown Name", 256);
    if (*ierr) return;
  }

  /* check if PETSC_COMM_WORLD is initialized by the user in Fortran */
  petscgetcomm_(&f_petsc_comm_world);
  MPI_Initialized(&flag);
  if (!flag) {
    PetscMPIInt mierr;

    if (f_petsc_comm_world) {
      *ierr = (*PetscErrorPrintf)("You cannot set PETSC_COMM_WORLD if you have not initialized MPI first\n");
      return;
    }

    *ierr = PetscPreMPIInit_Private();
    if (*ierr) return;
    mpi_init_(&mierr);
    if (mierr) {
      *ierr = (*PetscErrorPrintf)("PetscInitialize: Calling Fortran MPI_Init()\n");
      *ierr = (PetscErrorCode)mierr;
      return;
    }
    PetscBeganMPI = PETSC_TRUE;
  }
  if (f_petsc_comm_world) PETSC_COMM_WORLD = MPI_Comm_f2c(*(MPI_Fint *)&f_petsc_comm_world); /* User called MPI_INITIALIZE() and changed PETSC_COMM_WORLD */
  else PETSC_COMM_WORLD = MPI_COMM_WORLD;

  *ierr = PetscInitialize_Common(name, filename, help, PETSC_TRUE, (PetscInt)len);
  if (*ierr) {
    (void)(*PetscErrorPrintf)("PetscInitialize:PetscInitialize_Common\n");
    return;
  }
}
