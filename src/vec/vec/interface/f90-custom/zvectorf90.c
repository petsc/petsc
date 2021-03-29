
#include <petscvec.h>
#include <petsc/private/f90impl.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define vecgetarrayf90_            VECGETARRAYF90
#define vecrestorearrayf90_        VECRESTOREARRAYF90
#define vecgetarrayreadf90_        VECGETARRAYREADF90
#define vecrestorearrayreadf90_    VECRESTOREARRAYREADF90
#define vecduplicatevecsf90_       VECDUPLICATEVECSF90
#define vecdestroyvecsf90_         VECDESTROYVECSF90
#define vecdestroy_                VECDESTROY
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define vecgetarrayf90_            vecgetarrayf90
#define vecrestorearrayf90_        vecrestorearrayf90
#define vecgetarrayreadf90_        vecgetarrayreadf90
#define vecrestorearrayreadf90_    vecrestorearrayreadf90
#define vecduplicatevecsf90_       vecduplicatevecsf90
#define vecdestroyvecsf90_         vecdestroyvecsf90
#define vecdestroy_                vecdestroy
#endif

PETSC_EXTERN void vecgetarrayf90_(Vec *x,F90Array1d *ptr,int *__ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscScalar *fa;
  PetscInt    len;
  if (!ptr) {
    *__ierr = PetscError(((PetscObject)*x)->comm,__LINE__,PETSC_FUNCTION_NAME,__FILE__,PETSC_ERR_ARG_BADPTR,PETSC_ERROR_INITIAL,"ptr==NULL, maybe #include <petsc/finclude/petscvec.h> is missing?");
    return;
  }
  *__ierr = VecGetArray(*x,&fa);      if (*__ierr) return;
  *__ierr = VecGetLocalSize(*x,&len); if (*__ierr) return;
  *__ierr = F90Array1dCreate(fa,MPIU_SCALAR,1,len,ptr PETSC_F90_2PTR_PARAM(ptrd));
}
PETSC_EXTERN void vecrestorearrayf90_(Vec *x,F90Array1d *ptr,int *__ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscScalar *fa;
  *__ierr = F90Array1dAccess(ptr,MPIU_SCALAR,(void**)&fa PETSC_F90_2PTR_PARAM(ptrd));if (*__ierr) return;
  *__ierr = F90Array1dDestroy(ptr,MPIU_SCALAR PETSC_F90_2PTR_PARAM(ptrd));if (*__ierr) return;
  *__ierr = VecRestoreArray(*x,&fa);
}

PETSC_EXTERN void vecgetarrayreadf90_(Vec *x,F90Array1d *ptr,int *__ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  const PetscScalar *fa;
  PetscInt          len;
  if (!ptr) {
    *__ierr = PetscError(((PetscObject)*x)->comm,__LINE__,PETSC_FUNCTION_NAME,__FILE__,PETSC_ERR_ARG_BADPTR,PETSC_ERROR_INITIAL,"ptr==NULL, maybe #include <petsc/finclude/petscvec.h> is missing?");
    return;
  }
  *__ierr = VecGetArrayRead(*x,&fa);      if (*__ierr) return;
  *__ierr = VecGetLocalSize(*x,&len); if (*__ierr) return;
  *__ierr = F90Array1dCreate((PetscScalar*)fa,MPIU_SCALAR,1,len,ptr PETSC_F90_2PTR_PARAM(ptrd));
}
PETSC_EXTERN void vecrestorearrayreadf90_(Vec *x,F90Array1d *ptr,int *__ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  const PetscScalar *fa;
  *__ierr = F90Array1dAccess(ptr,MPIU_SCALAR,(void**)&fa PETSC_F90_2PTR_PARAM(ptrd));if (*__ierr) return;
  *__ierr = F90Array1dDestroy(ptr,MPIU_SCALAR PETSC_F90_2PTR_PARAM(ptrd));if (*__ierr) return;
  *__ierr = VecRestoreArrayRead(*x,&fa);
}

PETSC_EXTERN void vecduplicatevecsf90_(Vec *v,int *m,F90Array1d *ptr,int *__ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  Vec              *lV;
  PetscFortranAddr *newvint;
  int              i;
  *__ierr = VecDuplicateVecs(*v,*m,&lV); if (*__ierr) return;
  *__ierr = PetscMalloc1(*m,&newvint);  if (*__ierr) return;

  for (i=0; i<*m; i++) newvint[i] = (PetscFortranAddr)lV[i];
  *__ierr = PetscFree(lV); if (*__ierr) return;
  *__ierr = F90Array1dCreate(newvint,MPIU_FORTRANADDR,1,*m,ptr PETSC_F90_2PTR_PARAM(ptrd));
}

PETSC_EXTERN void vecdestroyvecsf90_(int *m,F90Array1d *ptr,int *__ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  Vec *vecs;
  int i;

  *__ierr = F90Array1dAccess(ptr,MPIU_FORTRANADDR,(void**)&vecs PETSC_F90_2PTR_PARAM(ptrd));if (*__ierr) return;
  for (i=0; i<*m; i++) {
    PETSC_FORTRAN_OBJECT_F_DESTROYED_TO_C_NULL(&vecs[i]);
    *__ierr = VecDestroy(&vecs[i]);
    if (*__ierr) return;
    PETSC_FORTRAN_OBJECT_C_NULL_TO_F_DESTROYED(&vecs[i]);
  }
  *__ierr = F90Array1dDestroy(ptr,MPIU_FORTRANADDR PETSC_F90_2PTR_PARAM(ptrd));if (*__ierr) return;
  *__ierr = PetscFree(vecs);
}

PETSC_EXTERN void vecdestroy_(Vec *x,int *ierr)
{
  PETSC_FORTRAN_OBJECT_F_DESTROYED_TO_C_NULL(x);
  *ierr = VecDestroy(x); if (*ierr) return;
  PETSC_FORTRAN_OBJECT_C_NULL_TO_F_DESTROYED(x);
}