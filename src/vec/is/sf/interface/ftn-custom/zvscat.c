#include <petsc/private/fortranimpl.h>
#include <petscvec.h>
#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define vecscatterremap_        VECSCATTERREMAP
  #define vecscatterview_         VECSCATTERVIEW
  #define vecscattercreatetoall_  VECSCATTERCREATETOALL
  #define vecscattercreatetozero_ VECSCATTERCREATETOZERO
  #define vecscatterdestroy_      VECSCATTERDESTROY
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define vecscatterremap_        vecscatterremap
  #define vecscatterview_         vecscatterview
  #define vecscattercreatetoall_  vecscattercreatetoall
  #define vecscattercreatetozero_ vecscattercreatetozero
  #define vecscatterdestroy_      vecscatterdestroy
#endif

PETSC_EXTERN void vecscattercreatetoall_(Vec *vin, VecScatter *ctx, Vec *vout, PetscErrorCode *ierr)
{
  CHKFORTRANNULLOBJECT(vout);
  *ierr = VecScatterCreateToAll(*vin, ctx, vout);
}

PETSC_EXTERN void vecscattercreatetozero_(Vec *vin, VecScatter *ctx, Vec *vout, PetscErrorCode *ierr)
{
  CHKFORTRANNULLOBJECT(vout);
  *ierr = VecScatterCreateToZero(*vin, ctx, vout);
}

PETSC_EXTERN void vecscatterview_(VecScatter *vecscatter, PetscViewer *viewer, PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer, v);
  *ierr = VecScatterView(*vecscatter, v);
}

PETSC_EXTERN void vecscatterremap_(VecScatter *scat, PetscInt *rto, PetscInt *rfrom, PetscErrorCode *ierr)
{
  CHKFORTRANNULLINTEGER(rto);
  CHKFORTRANNULLINTEGER(rfrom);
  *ierr = VecScatterRemap(*scat, rto, rfrom);
}

PETSC_EXTERN void vecscatterdestroy_(VecScatter *x, int *ierr)
{
  PETSC_FORTRAN_OBJECT_F_DESTROYED_TO_C_NULL(x);
  *ierr = VecScatterDestroy(x);
  if (*ierr) return;
  PETSC_FORTRAN_OBJECT_C_NULL_TO_F_DESTROYED(x);
}
