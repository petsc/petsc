/*
  This file contains Fortran stubs for Options routines. 
  These are not generated automatically since they require passing strings
  between Fortran and C.
*/

#include <petsc-private/fortranimpl.h> 

#ifdef PETSC_HAVE_FORTRAN_CAPS
#define petscoptionsinsertfileyaml_             PETSCOPTIONSINSERTFILEYAML
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petscoptionsinsertfileyaml_             petscoptionsinsertfileyaml
#endif

EXTERN_C_BEGIN

void PETSC_STDCALL petscoptionsinsertfileyaml_(MPI_Fint *comm,CHAR file PETSC_MIXED_LEN(len),PetscBool  *require,PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *c1;

  FIXCHAR(file,len,c1);
  *ierr = PetscOptionsInsertFileYAML(MPI_Comm_f2c(*comm),c1,*require);
  FREECHAR(file,c1);
}

EXTERN_C_END
