#include <petsc/private/fortranimpl.h>
#include <petscmatlab.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define petscmatlabenginecreate_      PETSCMATLABENGINECREATE
  #define petscmatlabengineevaluate_    PETSCMATLABENGINEEVALUATE
  #define petscmatlabenginegetoutput_   PETSCMATLABENGINEGETOUTPUT
  #define petscmatlabengineprintoutput_ PETSCMATLABENGINEPRINTOUTPUT
  #define petscmatlabengineputarray_    PETSCMATLABENGINEPUTARRAY
  #define petscmatlabenginegetarray_    PETSCMATLABENGINEGETARRAY
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define petscmatlabenginecreate_      petscmatlabenginecreate
  #define petscmatlabengineevaluate_    petscmatlabengineevaluate
  #define petscmatlabenginegetoutput_   petscmatlabenginegetoutput
  #define petscmatlabengineprintoutput_ petscmatlabengineprintoutput
  #define petscmatlabengineputarray_    petscmatlabengineputarray
  #define petscmatlabenginegetarray_    petscmatlabenginegetarray
#endif

PETSC_EXTERN void petscmatlabenginecreate_(MPI_Comm *comm, char *m, PetscMatlabEngine *e, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T len)
{
  char *ms;

  FIXCHAR(m, len, ms);
  *ierr = PetscMatlabEngineCreate(MPI_Comm_f2c(*(MPI_Fint *)&*comm), ms, e);
  if (*ierr) return;
  FREECHAR(m, ms);
}

PETSC_EXTERN void petscmatlabengineevaluate_(PetscMatlabEngine *e, char *m, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T len)
{
  char *ms;
  FIXCHAR(m, len, ms);
  *ierr = PetscMatlabEngineEvaluate(*e, ms);
  if (*ierr) return;
  FREECHAR(m, ms);
}

PETSC_EXTERN void petscmatlabengineputarray_(PetscMatlabEngine *e, PetscInt *m, PetscInt *n, PetscScalar *a, char *s, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T len)
{
  char *ms;
  FIXCHAR(s, len, ms);
  *ierr = PetscMatlabEnginePutArray(*e, *m, *n, a, ms);
  if (*ierr) return;
  FREECHAR(s, ms);
}

PETSC_EXTERN void petscmatlabenginegetarray_(PetscMatlabEngine *e, PetscInt *m, PetscInt *n, PetscScalar *a, char *s, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T len)
{
  char *ms;
  FIXCHAR(s, len, ms);
  *ierr = PetscMatlabEngineGetArray(*e, *m, *n, a, ms);
  if (*ierr) return;
  FREECHAR(s, ms);
}
