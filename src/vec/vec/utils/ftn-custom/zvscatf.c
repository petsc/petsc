#include <petsc/private/fortranimpl.h>
#include <petscvec.h>
#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define vecscattercreate_         VECSCATTERCREATE
#define vecscatterremap_          VECSCATTERREMAP
#define vecscatterdestroy_        VECSCATTERDESTROY
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define vecscattercreate_         vecscattercreate
#define vecscatterremap_          vecscatterremap
#define vecscatterdestroy_        vecscatterdestroy
#endif

PETSC_EXTERN void PETSC_STDCALL vecscattercreate_(Vec *xin,IS *ix,Vec *yin,IS *iy,VecScatter *newctx,PetscErrorCode *ierr)
{
  CHKFORTRANNULLOBJECTDEREFERENCE(ix);
  CHKFORTRANNULLOBJECTDEREFERENCE(iy);
  *ierr = VecScatterCreate(*xin,*ix,*yin,*iy,newctx);
}

PETSC_EXTERN void PETSC_STDCALL vecscatterremap_(VecScatter *scat,PetscInt *rto,PetscInt *rfrom, int *ierr)
{
  CHKFORTRANNULLINTEGER(rto);
  CHKFORTRANNULLINTEGER(rfrom);
  *ierr = VecScatterRemap(*scat,rto,rfrom);
}

PETSC_EXTERN void PETSC_STDCALL vecscatterdestroy_(VecScatter *ctx, int *__ierr)
{
  *__ierr = VecScatterDestroy(ctx);
}
