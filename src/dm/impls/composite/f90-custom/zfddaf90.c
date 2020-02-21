#include <petsc/private/f90impl.h>
#include <petscdmcomposite.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define dmcompositegetaccessvpvp_             DMCOMPOSITEGETACCESSVPVP
#define dmcompositerestoreaccessvpvp_         DMCOMPOSITERESTOREACCESSVPVP
#define dmcompositegetentriesarray_           DMCOMPOSITEGETENTRIESARRAY
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define dmcompositegetaccessvpvp_             dmcompositegetaccessvpvp
#define dmcompositerestoreaccessvpvp_         dmcompositerestoreaccessvpvp
#define dmcompositegetentriesarray_           dmcompositegetentriesarray
#endif

PETSC_EXTERN void dmcompositegetaccessvpvp_(DM *dm,Vec *v,Vec *v1,F90Array1d *p1,Vec *v2,F90Array1d *p2,PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(ptrd1) PETSC_F90_2PTR_PROTO(ptrd2))
{
  PetscScalar *pp1,*pp2;
  PetscInt    np1,np2;
  *ierr = DMCompositeGetEntries(*dm,0,&np1,0,&np2);
  *ierr = DMCompositeGetAccess(*dm,*v,v1,&pp1,v2,&pp2);
  *ierr = F90Array1dCreate(pp1,MPIU_SCALAR,0,np1-1,p1 PETSC_F90_2PTR_PARAM(ptrd1));
  *ierr = F90Array1dCreate(pp2,MPIU_SCALAR,0,np2-1,p2 PETSC_F90_2PTR_PARAM(ptrd2));
}

PETSC_EXTERN void dmcompositerestoreaccessvpvp_(DM *dm,Vec *v,Vec *v1,F90Array1d *p1,Vec *v2,F90Array1d *p2,PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(ptrd1) PETSC_F90_2PTR_PROTO(ptrd2))
{
  *ierr = DMCompositeRestoreAccess(*dm,*v,v1,0,v2,0);
  *ierr = F90Array1dDestroy(p1,MPIU_SCALAR PETSC_F90_2PTR_PARAM(ptrd1));
  *ierr = F90Array1dDestroy(p2,MPIU_SCALAR PETSC_F90_2PTR_PARAM(ptrd2));
}

PETSC_EXTERN void dmcompositegetentriesarray_(DM *dm, DM *dmarray, PetscErrorCode *ierr)
{
  *ierr = DMCompositeGetEntriesArray(*dm, dmarray);
}

