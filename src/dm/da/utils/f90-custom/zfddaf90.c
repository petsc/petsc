#include "../src/sys/f90-src/f90impl.h"
#include "petscda.h"

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define dmcompositegetaccessvpvp_             DMCOMPOSITEGETACCESSVPVP
#define dmcompositerestoreaccessvpvp_         DMCOMPOSITERESTOREACCESSVPVP
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define dmcompositegetaccessvpvp_             dmcompositegetaccessvpvp
#define dmcompositerestoreaccessvpvp_         dmcompositerestoreaccessvpvp
#endif

EXTERN_C_BEGIN
void PETSC_STDCALL dmcompositegetaccessvpvp_(DMComposite *dm,Vec *v,Vec *v1,F90Array1d *p1,Vec *v2,F90Array1d *p2,PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(ptrd1) PETSC_F90_2PTR_PROTO(ptrd2))
{
  PetscScalar *pp1,*pp2;
  PetscInt    np1,np2;
  *ierr = DMCompositeGetEntries(*dm,0,&np1,0,&np2);
  *ierr = DMCompositeGetAccess(*dm,*v,v1,&pp1,v2,&pp2);
  *ierr = F90Array1dCreate(pp1,PETSC_SCALAR,0,np1-1,p1 PETSC_F90_2PTR_PARAM(ptrd1));
  *ierr = F90Array1dCreate(pp2,PETSC_SCALAR,0,np2-1,p2 PETSC_F90_2PTR_PARAM(ptrd2));
}

void PETSC_STDCALL dmcompositerestoreaccessvpvp_(DMComposite *dm,Vec *v,Vec *v1,F90Array1d *p1,Vec *v2,F90Array1d *p2,PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(ptrd1) PETSC_F90_2PTR_PROTO(ptrd2))
{
  *ierr = DMCompositeRestoreAccess(*dm,*v,v1,0,v2,0);
  *ierr = F90Array1dDestroy(p1,PETSC_SCALAR PETSC_F90_2PTR_PARAM(ptrd1));
  *ierr = F90Array1dDestroy(p2,PETSC_SCALAR PETSC_F90_2PTR_PARAM(ptrd2));
}

EXTERN_C_END
