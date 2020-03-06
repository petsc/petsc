/*
  This file contains Fortran stubs for Options routines.
  These are not generated automatically since they require passing strings
  between Fortran and C.
*/

#include <petsc/private/fortranimpl.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define petscoptionsinsertfileyaml_             PETSCOPTIONSINSERTFILEYAML
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petscoptionsinsertfileyaml_             petscoptionsinsertfileyaml
#endif

PETSC_EXTERN void petscoptionsinsertfileyaml_(MPI_Fint *comm,char* file,PetscBool  *require,PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T len)
{
  char *c1;

  FIXCHAR(file,len,c1);
  *ierr = PetscOptionsInsertFileYAML(MPI_Comm_f2c(*comm),c1,*require);if (*ierr) return;
  FREECHAR(file,c1);
}
