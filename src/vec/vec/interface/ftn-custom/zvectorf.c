#include "zpetsc.h"
#include "petscvec.h"
#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define vecsetvalue_              VECSETVALUE
#define vecloadintovector_        VECLOADINTOVECTOR  
#define vecview_                  VECVIEW
#define vecgetarray_              VECGETARRAY
#define vecrestorearray_          VECRESTOREARRAY
#define vecduplicatevecs_         VECDUPLICATEVECS
#define vecdestroyvecs_           VECDESTROYVECS
#define vecmax_                   VECMAX
#define vecgetownershiprange_     VECGETOWNERSHIPRANGE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define vecsetvalue_              vecsetvalue
#define vecloadintovector_        vecloadintovector
#define vecview_                  vecview
#define vecgetarray_              vecgetarray
#define vecrestorearray_          vecrestorearray
#define vecduplicatevecs_         vecduplicatevecs
#define vecdestroyvecs_           vecdestroyvecs
#define vecmax_                   vecmax
#define vecgetownershiprange_     vecgetownershiprange
#endif

EXTERN_C_BEGIN

void PETSC_STDCALL vecsetvalue_(Vec *v,PetscInt *i,PetscScalar *va,InsertMode *mode,PetscErrorCode *ierr)
{
  /* cannot use VecSetValue() here since that uses CHKERRQ() which has a return in it */
  *ierr = VecSetValues(*v,1,i,va,*mode);
}

void PETSC_STDCALL vecloadintovector_(PetscViewer *viewer,Vec *vec,PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer,v);
  *ierr = VecLoadIntoVector(v,*vec);
}

void PETSC_STDCALL vecview_(Vec *x,PetscViewer *vin,PetscErrorCode *ierr)
{
  PetscViewer v;

  PetscPatchDefaultViewers_Fortran(vin,v);
  *ierr = VecView(*x,v);
}

void PETSC_STDCALL vecgetarray_(Vec *x,PetscScalar *fa,size_t *ia,PetscErrorCode *ierr)
{
  PetscScalar *lx;
  PetscInt    m;

  *ierr = VecGetArray(*x,&lx); if (*ierr) return;
  *ierr = VecGetLocalSize(*x,&m);if (*ierr) return;
  *ierr = PetscScalarAddressToFortran((PetscObject)*x,fa,lx,m,ia);
}

/* Be to keep vec/examples/ex21.F and snes/examples/ex12.F up to date */
void PETSC_STDCALL vecrestorearray_(Vec *x,PetscScalar *fa,size_t *ia,PetscErrorCode *ierr)
{
  PetscInt    m;
  PetscScalar *lx;

  *ierr = VecGetLocalSize(*x,&m);if (*ierr) return;
  *ierr = PetscScalarAddressFromFortran((PetscObject)*x,fa,*ia,m,&lx);if (*ierr) return;
  *ierr = VecRestoreArray(*x,&lx);if (*ierr) return;
}

/*
      vecduplicatevecs() and vecdestroyvecs() are slightly different from C since the 
    Fortran provides the array to hold the vector objects,while in C that 
    array is allocated by the VecDuplicateVecs()
*/
void PETSC_STDCALL vecduplicatevecs_(Vec *v,PetscInt *m,Vec *newv,PetscErrorCode *ierr)
{
  Vec *lV;
  PetscInt i;
  *ierr = VecDuplicateVecs(*v,*m,&lV); if (*ierr) return;
  for (i=0; i<*m; i++) {
    newv[i] = lV[i];
  }
  *ierr = PetscFree(lV); 
}

void PETSC_STDCALL vecdestroyvecs_(Vec *vecs,PetscInt *m,PetscErrorCode *ierr)
{
  PetscInt i;
  for (i=0; i<*m; i++) {
    *ierr = VecDestroy(vecs[i]);if (*ierr) return;
  }
}

void PETSC_STDCALL vecmax_(Vec *x,PetscInt *p,PetscReal *val,PetscErrorCode *ierr)
{
  CHKFORTRANNULLINTEGER(p);
  *ierr = VecMax(*x,p,val);
}

void PETSC_STDCALL vecgetownershiprange_(Vec *x,PetscInt *low,PetscInt *high, PetscErrorCode *ierr)
{
  CHKFORTRANNULLINTEGER(low);
  CHKFORTRANNULLINTEGER(high);
  *ierr = VecGetOwnershipRange(*x,low,high);
}


EXTERN_C_END
