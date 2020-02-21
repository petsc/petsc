
#include <petsc/private/fortranimpl.h>
#include <petscdmcomposite.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define dmcompositegetentries1_      DMCOMPOSITEGETENTRIES1
#define dmcompositegetentries2_      DMCOMPOSITEGETENTRIES2
#define dmcompositegetentries3_      DMCOMPOSITEGETENTRIES3
#define dmcompositegetentries4_      DMCOMPOSITEGETENTRIES4
#define dmcompositegetentries5_      DMCOMPOSITEGETENTRIES5
#define dmcompositeadddm_            DMCOMPOSITEADDDM
#define dmcompositedestroy_          DMCOMPOSITEDESTROY
#define dmcompositegetaccess4_       DMCOMPOSITEGETACCESS4
#define dmcompositescatter4_         DMCOMPOSITESCATTER4
#define dmcompositerestoreaccess4_   DMCOMPOSITERESTOREACCESS4
#define dmcompositegetlocalvectors4_ DMCOMPOSITEGETLOCALVECTORS4
#define dmcompositerestorelocalvectors4_  DMCOMPOSITERESTORELOCALVECTORS4
#define dmcompositegetglobaliss_     DMCOMPOSITEGETGLOBALISS
#define dmcompositegetlocaliss_      DMCOMPOSITEGETLOCALISS
#define dmcompositegetaccessarray_   DMCOMPOSITEGETACCESSARRAY
#define dmcompositerestoreaccessarray_  DMCOMPOSITERESTOREACCESSARRAY
#define dmcompositegetlocalaccessarray_ DMCOMPOSITEGETLOCALACCESSARRAY
#define dmcompositerestorelocalaccessarray_ DMCOMPOSITERESTORELOCALACCESSARRAY
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define dmcompositegetentries1_      dmcompositegetentries1
#define dmcompositegetentries2_      dmcompositegetentries2
#define dmcompositegetentries3_      dmcompositegetentries3
#define dmcompositegetentries4_      dmcompositegetentries4
#define dmcompositegetentries5_      dmcompositegetentries5
#define dmcompositeadddm_            dmcompositeadddm
#define dmcompositedestroy_          dmcompositedestroy
#define dmcompositegetaccess4_       dmcompositegetaccess4
#define dmcompositescatter4_         dmcompositescatter4
#define dmcompositerestoreaccess4_   dmcompositerestoreaccess4
#define dmcompositegetlocalvectors4_ dmcompositegetlocalvectors4
#define dmcompositerestorelocalvectors4_ dmcompositerestorelocalvectors4
#define dmcompositegetglobaliss_     dmcompositegetglobaliss
#define dmcompositegetlocaliss_      dmcompositegetlocaliss
#define dmcompositegetaccessarray_   dmcompositegetaccessarray
#define dmcompositerestoreaccessarray_  dmcompositerestoreaccessarray
#define dmcompositegetlocalaccessarray_ dmcompositegetlocalaccessarray
#define dmcompositerestorelocalaccessarray_ dmcompositerestorelocalaccessarray
#endif

PETSC_EXTERN void dmcompositegetentries1_(DM *dm,DM *da1,PetscErrorCode *ierr)
{
  *ierr = DMCompositeGetEntries(*dm,da1);
}

PETSC_EXTERN void dmcompositegetentries2_(DM *dm,DM *da1,DM *da2,PetscErrorCode *ierr)
{
  *ierr = DMCompositeGetEntries(*dm,da1,da2);
}

PETSC_EXTERN void dmcompositegetentries3_(DM *dm,DM *da1,DM *da2,DM *da3,PetscErrorCode *ierr)
{
  *ierr = DMCompositeGetEntries(*dm,da1,da2,da3);
}

PETSC_EXTERN void dmcompositegetentries4_(DM *dm,DM *da1,DM *da2,DM *da3,DM *da4,PetscErrorCode *ierr)
{
  *ierr = DMCompositeGetEntries(*dm,da1,da2,da3,da4);
}

PETSC_EXTERN void dmcompositegetentries5_(DM *dm,DM *da1,DM *da2,DM *da3,DM *da4,DM *da5,PetscErrorCode *ierr)
{
  *ierr = DMCompositeGetEntries(*dm,da1,da2,da3,da4,da5);
}

PETSC_EXTERN void dmcompositegetaccess4_(DM *dm,Vec *v,void **v1,void **p1,void **v2,void **p2,PetscErrorCode *ierr)
{
  Vec *vv1 = (Vec*)v1,*vv2 = (Vec*)v2;
  *ierr = DMCompositeGetAccess(*dm,*v,vv1,(PetscScalar*)p1,vv2,(PetscScalar*)p2);
}

PETSC_EXTERN void dmcompositescatter4_(DM *dm,Vec *v,void *v1,void *p1,void *v2,void *p2,PetscErrorCode *ierr)
{
  Vec *vv1 = (Vec*)v1,*vv2 = (Vec*)v2;
  *ierr = DMCompositeScatter(*dm,*v,*vv1,(PetscScalar*)p1,*vv2,(PetscScalar*)p2);
}

PETSC_EXTERN void dmcompositerestoreaccess4_(DM *dm,Vec *v,void **v1,void **p1,void **v2,void **p2,PetscErrorCode *ierr)
{
  *ierr = DMCompositeRestoreAccess(*dm,*v,(Vec*)v1,0,(Vec*)v2,0);
}

PETSC_EXTERN void dmcompositegetlocalvectors4_(DM *dm,void **v1,void **p1,void **v2,void **p2,PetscErrorCode *ierr)
{
  Vec *vv1 = (Vec*)v1,*vv2 = (Vec*)v2;
  *ierr = DMCompositeGetLocalVectors(*dm,vv1,(PetscScalar*)p1,vv2,(PetscScalar*)p2);
}

PETSC_EXTERN void dmcompositerestorelocalvectors4_(DM *dm,void **v1,void **p1,void **v2,void **p2,PetscErrorCode *ierr)
{
  Vec *vv1 = (Vec*)v1,*vv2 = (Vec*)v2;
  *ierr = DMCompositeRestoreLocalVectors(*dm,vv1,(PetscScalar*)p1,vv2,(PetscScalar*)p2);
}

PETSC_EXTERN void dmcompositegetglobaliss_(DM *dm,IS *iss,PetscErrorCode *ierr)
{
  IS      *ais;
  PetscInt i,ndm;
  *ierr = DMCompositeGetGlobalISs(*dm,&ais); if (*ierr) return;
  *ierr = DMCompositeGetNumberDM(*dm,&ndm); if (*ierr) return;
  for (i=0; i<ndm; i++) iss[i] = ais[i];
  *ierr = PetscFree(ais);
}

PETSC_EXTERN void dmcompositegetlocaliss_(DM *dm,IS *iss,PetscErrorCode *ierr)
{
  IS      *ais;
  PetscInt i,ndm;
  *ierr = DMCompositeGetLocalISs(*dm,&ais); if (*ierr) return;
  *ierr = DMCompositeGetNumberDM(*dm,&ndm); if (*ierr) return;
  for (i=0; i<ndm; i++) iss[i] = ais[i];
  *ierr = PetscFree(ais);
}

PETSC_EXTERN void dmcompositegetaccessarray_(DM *dm,Vec *gvec,PetscInt *n,const PetscInt *wanted,Vec *vecs,PetscErrorCode *ierr)
{
  CHKFORTRANNULLINTEGER(wanted);
  *ierr = DMCompositeGetAccessArray(*dm,*gvec,*n,wanted,vecs);
}

PETSC_EXTERN void dmcompositerestoreaccessarray_(DM *dm,Vec *gvec,PetscInt *n,const PetscInt *wanted,Vec *vecs,PetscErrorCode *ierr)
{
  CHKFORTRANNULLINTEGER(wanted);
  *ierr = DMCompositeRestoreAccessArray(*dm,*gvec,*n,wanted,vecs);
}

PETSC_EXTERN void dmcompositegetlocalaccessarray_(DM *dm,Vec *lvec,PetscInt *n,const PetscInt *wanted,Vec *vecs,PetscErrorCode *ierr)
{
  CHKFORTRANNULLINTEGER(wanted);
  *ierr = DMCompositeGetLocalAccessArray(*dm,*lvec,*n,wanted,vecs);
}

PETSC_EXTERN void dmcompositerestorelocalaccessarray_(DM *dm,Vec *lvec,PetscInt *n,const PetscInt *wanted,Vec *vecs,PetscErrorCode *ierr)
{
  CHKFORTRANNULLINTEGER(wanted);
  *ierr = DMCompositeRestoreLocalAccessArray(*dm,*lvec,*n,wanted,vecs);
}
