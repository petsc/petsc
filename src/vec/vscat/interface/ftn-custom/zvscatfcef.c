#include <petsc/private/fortranimpl.h>
#include <petscvec.h>
#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define vecscattercreate_         VECSCATTERCREATE
#define vecscatterremap_          VECSCATTERREMAP
#define vecscatterview_           VECSCATTERVIEW
#define vecscattercreatetoall_    VECSCATTERCREATETOALL
#define vecscattercreatetozero_   VECSCATTERCREATETOZERO
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define vecscattercreate_         vecscattercreate
#define vecscatterremap_          vecscatterremap
#define vecscatterview_           vecscatterview
#define vecscattercreatetoall_    vecscattercreatetoall
#define vecscattercreatetozero_   vecscattercreatetozero
#endif

PETSC_EXTERN void  vecscattercreatetoall_(Vec *vin,VecScatter *ctx,Vec *vout, PetscErrorCode *ierr)
{
  CHKFORTRANNULLOBJECT(vout);
  *ierr = VecScatterCreateToAll(*vin,ctx,vout);
}

PETSC_EXTERN void  vecscattercreatetozero_(Vec *vin,VecScatter *ctx,Vec *vout, PetscErrorCode *ierr)
{
  CHKFORTRANNULLOBJECT(vout);
  *ierr = VecScatterCreateToZero(*vin,ctx,vout);
}

PETSC_EXTERN void vecscatterview_(VecScatter *vecscatter,PetscViewer *viewer,PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer,v);
  *ierr = VecScatterView(*vecscatter,v);
}

PETSC_EXTERN void vecscatterremap_(VecScatter *scat,PetscInt *rto,PetscInt *rfrom, PetscErrorCode *ierr)
{
  CHKFORTRANNULLINTEGER(rto);
  CHKFORTRANNULLINTEGER(rfrom);
  *ierr = VecScatterRemap(*scat,rto,rfrom);
}

