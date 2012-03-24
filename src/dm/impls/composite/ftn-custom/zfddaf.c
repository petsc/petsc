
#include <petsc-private/fortranimpl.h>
#include <petscdmcomposite.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define dmcompositegetentries1_      DMCOMPOSITEGETENTRIES1
#define dmcompositegetentries2_      DMCOMPOSITEGETENTRIES2
#define dmcompositegetentries3_      DMCOMPOSITEGETENTRIES3
#define dmcompositegetentries4_      DMCOMPOSITEGETENTRIES4
#define dmcompositegetentries5_      DMCOMPOSITEGETENTRIES5
#define dmcompositecreate_           DMCOMPOSITECREATE
#define dmcompositeadddm_            DMCOMPOSITEADDDM
#define dmcompositedestroy_          DMCOMPOSITEDESTROY
#define dmcompositegetaccess4_       DMCOMPOSITEGETACCESS4
#define dmcompositescatter4_         DMCOMPOSITESCATTER4
#define dmcompositerestoreaccess4_   DMCOMPOSITERESTOREACCESS4
#define dmcompositegetlocalvectors4_ DMCOMPOSITEGETLOCALVECTORS4
#define dmcompositerestorelocalvectors4_ DMCOMPOSITERESTORELOCALVECTORS4
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define dmcompositegetentries1_      dmcompositegetentries1
#define dmcompositegetentries2_      dmcompositegetentries2
#define dmcompositegetentries3_      dmcompositegetentries3
#define dmcompositegetentries4_      dmcompositegetentries4
#define dmcompositegetentries5_      dmcompositegetentries5
#define dmcompositecreate_           dmcompositecreate
#define dmcompositeadddm_            dmcompositeadddm
#define dmcompositedestroy_          dmcompositedestroy
#define dmcompositegetaccess4_       dmcompositegetaccess4
#define dmcompositescatter4_         dmcompositescatter4
#define dmcompositerestoreaccess4_   dmcompositerestoreaccess4
#define dmcompositegetlocalvectors4_ dmcompositegetlocalvectors4
#define dmcompositerestorelocalvectors4_ dmcompositerestorelocalvectors4
#endif

EXTERN_C_BEGIN

void PETSC_STDCALL dmcompositegetentries1_(DM *dm,DM *da1,PetscErrorCode *ierr)
{
  *ierr = DMCompositeGetEntries(*dm,da1);
}

void PETSC_STDCALL dmcompositegetentries2_(DM *dm,DM *da1,DM *da2,PetscErrorCode *ierr)
{
  *ierr = DMCompositeGetEntries(*dm,da1,da2);
}

void PETSC_STDCALL dmcompositegetentries3_(DM *dm,DM *da1,DM *da2,DM *da3,PetscErrorCode *ierr)
{
  *ierr = DMCompositeGetEntries(*dm,da1,da2,da3);
}

void PETSC_STDCALL dmcompositegetentries4_(DM *dm,DM *da1,DM *da2,DM *da3,DM *da4,PetscErrorCode *ierr)
{
  *ierr = DMCompositeGetEntries(*dm,da1,da2,da3,da4);
}

void PETSC_STDCALL dmcompositegetentries5_(DM *dm,DM *da1,DM *da2,DM *da3,DM *da4,DM *da5,PetscErrorCode *ierr)
{
  *ierr = DMCompositeGetEntries(*dm,da1,da2,da3,da4,da5);
}

void PETSC_STDCALL  dmcompositecreate_(MPI_Fint * comm,DM *A, int *ierr ){
  *ierr = DMCompositeCreate(MPI_Comm_f2c( *(comm) ),A);
}

void PETSC_STDCALL dmcompositeadddm_(DM *dm,DM *da,PetscErrorCode *ierr)
{
  *ierr = DMCompositeAddDM(*dm,*da);
}

void PETSC_STDCALL dmcompositedestroy_(DM *dm,PetscErrorCode *ierr)
{
  *ierr = DMDestroy(dm);
}

void PETSC_STDCALL dmcompositegetaccess4_(DM *dm,Vec *v,void **v1,void **p1,void **v2,void **p2,PetscErrorCode *ierr)
{
  Vec *vv1 = (Vec*)v1,*vv2 = (Vec*)v2;
  *ierr = DMCompositeGetAccess(*dm,*v,vv1,(PetscScalar*)p1,vv2,(PetscScalar*)p2);
}

void PETSC_STDCALL dmcompositescatter4_(DM *dm,Vec *v,void *v1,void *p1,void *v2,void *p2,PetscErrorCode *ierr)
{
  Vec *vv1 = (Vec*)v1,*vv2 = (Vec*)v2;
  *ierr = DMCompositeScatter(*dm,*v,*vv1,(PetscScalar*)p1,*vv2,(PetscScalar*)p2);
}

void PETSC_STDCALL dmcompositerestoreaccess4_(DM *dm,Vec *v,void **v1,void **p1,void **v2,void **p2,PetscErrorCode *ierr)
{
  *ierr = DMCompositeRestoreAccess(*dm,*v,(Vec*)v1,0,(Vec*)v2,0);
}

void PETSC_STDCALL dmcompositegetlocalvectors4_(DM *dm,void **v1,void **p1,void **v2,void **p2,PetscErrorCode *ierr)
{
  Vec *vv1 = (Vec*)v1,*vv2 = (Vec*)v2;
  *ierr = DMCompositeGetLocalVectors(*dm,vv1,(PetscScalar*)p1,vv2,(PetscScalar*)p2);
}

void PETSC_STDCALL dmcompositerestorelocalvectors4_(DM *dm,void **v1,void **p1,void **v2,void **p2,PetscErrorCode *ierr)
{
  Vec *vv1 = (Vec*)v1,*vv2 = (Vec*)v2;
  *ierr = DMCompositeRestoreLocalVectors(*dm,vv1,(PetscScalar*)p1,vv2,(PetscScalar*)p2);
}

EXTERN_C_END

