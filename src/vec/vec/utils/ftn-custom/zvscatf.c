#include <petsc/private/fortranimpl.h>
#include <petscvec.h>
#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define vecscattercreate_         VECSCATTERCREATE
#define vecscatterremap_          VECSCATTERREMAP
#define vecscatterdestroy_        VECSCATTERDESTROY
#define vecscatterview_           VECSCATTERVIEW
#define vecscattercreatetoall_    VECSCATTERCREATETOALL
#define vecscattercreatetozero_   VECSCATTERCREATETOZERO
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define vecscattercreate_         vecscattercreate
#define vecscatterremap_          vecscatterremap
#define vecscatterdestroy_        vecscatterdestroy
#define vecscatterview_           vecscatterview
#define vecscattercreatetoall_    vecscattercreatetoall
#define vecscattercreatetozero_   vecscattercreatetozero
#endif

PETSC_EXTERN void PETSC_STDCALL  vecscattercreatetoall_(Vec *vin,VecScatter *ctx,Vec *vout, int *ierr)
{
  CHKFORTRANNULLOBJECT(vout);
  *ierr = VecScatterCreateToAll(*vin,ctx,vout);
}

PETSC_EXTERN void PETSC_STDCALL  vecscattercreatetozero_(Vec *vin,VecScatter *ctx,Vec *vout, int *ierr)
{
  CHKFORTRANNULLOBJECT(vout);
  *ierr = VecScatterCreateToZero(*vin,ctx,vout);
}

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

PETSC_EXTERN void PETSC_STDCALL vecscatterdestroy_(VecScatter *ctx, int *ierr)
{
  *ierr = VecScatterDestroy(ctx);
}

PETSC_EXTERN void PETSC_STDCALL vecscatterview_(VecScatter *vecscatter,PetscViewer *viewer,PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer,v);
  *ierr = VecScatterView(*vecscatter,v);
}

