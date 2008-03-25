
#include "private/fortranimpl.h"
#include "petscda.h"

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define dmgetmatrix_                 DMGETMATRIX
#define dagetmatrix_                 DAGETMATRIX
#define dmcompositegetentries1_      DMCOMPOSITEGETENTRIES1
#define dmcompositegetentries2_      DMCOMPOSITEGETENTRIES2
#define dmcompositegetentries3_      DMCOMPOSITEGETENTRIES3
#define dmcompositegetentries4_      DMCOMPOSITEGETENTRIES4
#define dmcompositegetentries5_      DMCOMPOSITEGETENTRIES5
#define dmcompositecreate_           DMCOMPOSITECREATE
#define dmcompositeadddm_            DMCOMPOSITEADDDM
#define dmcompositeaddarray_         DMCOMPOSITEADDARRAY
#define dmcompositedestroy_          DMCOMPOSITEDESTROY
#define dmcompositegetaccess4_       DMCOMPOSITEGETACCESS4
#define dmcompositescatter4_         DMCOMPOSITESCATTER4
#define dmcompositerestoreaccess4_   DMCOMPOSITERESTOREACCESS4
#define dmcompositegetlocalvectors4_ DMCOMPOSITEGETLOCALVECTORS4
#define dmcompositerestorelocalvectors4_ DMCOMPOSITERESTORELOCALVECTORS4
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define dagetmatrix_                 dagetmatrix
#define dmgetmatrix_                 dmgetmatrix
#define dmcompositegetentries1_      dmcompositegetentries1
#define dmcompositegetentries2_      dmcompositegetentries2
#define dmcompositegetentries3_      dmcompositegetentries3
#define dmcompositegetentries4_      dmcompositegetentries4
#define dmcompositegetentries5_      dmcompositegetentries5
#define dmcompositecreate_           dmcompositecreate
#define dmcompositeadddm_            dmcompositeadddm
#define dmcompositedestroy_          dmcompositedestroy
#define dmcompositeaddarray_         dmcompositeaddarray
#define dmcompositegetaccess4_       dmcompositegetaccess4
#define dmcompositescatter4_         dmcompositescatter4
#define dmcompositerestoreaccess4_   dmcompositerestoreaccess4
#define dmcompositegetlocalvectors4_ dmcompositegetlocalvectors4
#define dmcompositerestorelocalvectors4_ dmcompositerestorelocalvectors4
#endif

EXTERN_C_BEGIN
void PETSC_STDCALL dagetmatrix_(DA *da,CHAR mat_type PETSC_MIXED_LEN(len),Mat *J,PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;
  FIXCHAR(mat_type,len,t);
  *ierr = DAGetMatrix(*da,t,J);
  FREECHAR(mat_type,t);
}

void PETSC_STDCALL dmgetmatrix_(DM *dm,CHAR mat_type PETSC_MIXED_LEN(len),Mat *J,PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;
  FIXCHAR(mat_type,len,t);
  *ierr = DMGetMatrix(*dm,t,J);
  FREECHAR(mat_type,t);
}

void PETSC_STDCALL dmcompositegetentries1_(DMComposite *dm,DA *da1,PetscErrorCode *ierr)
{
  *ierr = DMCompositeGetEntries(*dm,da1);
}

void PETSC_STDCALL dmcompositegetentries2_(DMComposite *dm,DA *da1,DA *da2,PetscErrorCode *ierr)
{
  *ierr = DMCompositeGetEntries(*dm,da1,da2);
}

void PETSC_STDCALL dmcompositegetentries3_(DMComposite *dm,DA *da1,DA *da2,DA *da3,PetscErrorCode *ierr)
{
  *ierr = DMCompositeGetEntries(*dm,da1,da2,da3);
}

void PETSC_STDCALL dmcompositegetentries4_(DMComposite *dm,DA *da1,DA *da2,DA *da3,DA *da4,PetscErrorCode *ierr)
{
  *ierr = DMCompositeGetEntries(*dm,da1,da2,da3,da4);
}

void PETSC_STDCALL dmcompositegetentries5_(DMComposite *dm,DA *da1,DA *da2,DA *da3,DA *da4,DA *da5,PetscErrorCode *ierr)
{
  *ierr = DMCompositeGetEntries(*dm,da1,da2,da3,da4,da5);
}

void PETSC_STDCALL  dmcompositecreate_(MPI_Fint * comm,DMComposite *A, int *ierr ){
  *ierr = DMCompositeCreate(MPI_Comm_f2c( *(comm) ),A);
}

void PETSC_STDCALL dmcompositeadddm_(DMComposite *dm,DM *da,PetscErrorCode *ierr)
{
  *ierr = DMCompositeAddDM(*dm,*da);
}

void PETSC_STDCALL dmcompositedestroy_(DMComposite *dm,PetscErrorCode *ierr)
{
  *ierr = DMCompositeDestroy(*dm);
}

void PETSC_STDCALL dmcompositeaddarray_(DMComposite *dm,PetscInt *r,PetscInt *n,PetscErrorCode *ierr)
{
  *ierr = DMCompositeAddArray(*dm,*r,*n);
}

void PETSC_STDCALL dmcompositegetaccess4_(DMComposite *dm,Vec *v,void **v1,void **p1,void **v2,void **p2,PetscErrorCode *ierr)
{
  Vec *vv1 = (Vec*)v1,*vv2 = (Vec*)v2;
  *ierr = DMCompositeGetAccess(*dm,*v,vv1,(PetscScalar*)p1,vv2,(PetscScalar*)p2);
}

void PETSC_STDCALL dmcompositescatter4_(DMComposite *dm,Vec *v,void *v1,void *p1,void *v2,void *p2,PetscErrorCode *ierr)
{
  Vec *vv1 = (Vec*)v1,*vv2 = (Vec*)v2;
  *ierr = DMCompositeScatter(*dm,*v,*vv1,(PetscScalar*)p1,*vv2,(PetscScalar*)p2);
}

void PETSC_STDCALL dmcompositerestoreaccess4_(DMComposite *dm,Vec *v,void **v1,void **p1,void **v2,void **p2,PetscErrorCode *ierr)
{
  *ierr = DMCompositeRestoreAccess(*dm,*v,(Vec*)v1,0,(Vec*)v2,0);
}

void PETSC_STDCALL dmcompositegetlocalvectors4_(DMComposite *dm,void **v1,void **p1,void **v2,void **p2,PetscErrorCode *ierr)
{
  Vec *vv1 = (Vec*)v1,*vv2 = (Vec*)v2;
  *ierr = DMCompositeGetLocalVectors(*dm,vv1,(PetscScalar*)p1,vv2,(PetscScalar*)p2);
}

void PETSC_STDCALL dmcompositerestorelocalvectors4_(DMComposite *dm,void **v1,void **p1,void **v2,void **p2,PetscErrorCode *ierr)
{
  Vec *vv1 = (Vec*)v1,*vv2 = (Vec*)v2;
  *ierr = DMCompositeRestoreLocalVectors(*dm,vv1,(PetscScalar*)p1,vv2,(PetscScalar*)p2);
}

EXTERN_C_END

