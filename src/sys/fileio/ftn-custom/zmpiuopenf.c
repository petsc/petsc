#include <petsc/private/fortranimpl.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define petscfopen_  PETSCFOPEN
  #define petscfclose_ PETSCFCLOSE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define petscfopen_  petscfopen
  #define petscfclose_ petscfclose
#endif

#if defined(__cplusplus)
extern "C" {
#endif

PETSC_EXTERN void petscfopen_(MPI_Comm *comm, char *fname, char *fmode, FILE **file, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T len1, PETSC_FORTRAN_CHARLEN_T len2)
{
  char *c1, *c2;

  FIXCHAR(fname, len1, c1);
  FIXCHAR(fmode, len2, c2);
  *ierr = PetscFOpen(MPI_Comm_f2c(*(MPI_Fint *)&*comm), c1, c2, file);
  if (*ierr) return;
  FREECHAR(fname, c1);
  FREECHAR(fmode, c2);
}

PETSC_EXTERN void petscfclose_(MPI_Comm *comm, FILE **file, PetscErrorCode *ierr)
{
  *ierr = PetscFClose(MPI_Comm_f2c(*(MPI_Fint *)&*comm), *file);
}

#if defined(__cplusplus)
}
#endif
