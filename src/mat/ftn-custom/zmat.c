
#include <petsc/private/fortranimpl.h>
#include <petscmat.h>


#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define matsetvalue_                     MATSETVALUE
#define matsetvaluelocal_                MATSETVALUELOCAL
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define matsetvalue_                     matsetvalue
#define matsetvaluelocal_                matsetvaluelocal
#endif

PETSC_EXTERN void PETSC_STDCALL matsetvalue_(Mat *mat,PetscInt *i,PetscInt *j,PetscScalar *va,InsertMode *mode,PetscErrorCode *ierr)
{
  /* cannot use MatSetValue() here since that uses CHKERRQ() which has a return in it */
  *ierr = MatSetValues(*mat,1,i,1,j,va,*mode);
}

PETSC_EXTERN void PETSC_STDCALL matsetvaluelocal_(Mat *mat,PetscInt *i,PetscInt *j,PetscScalar *va,InsertMode *mode,PetscErrorCode *ierr)
{
  /* cannot use MatSetValueLocal() here since that uses CHKERRQ() which has a return in it */
  *ierr = MatSetValuesLocal(*mat,1,i,1,j,va,*mode);
}

