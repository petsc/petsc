
#include "zpetsc.h"
#include "petscda.h"

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define dagetmatrix_                 DAGETMATRIX
#define dmcompositegetentries1_      DMCOMPOSITEGETENTRIES1
#define dmcompositegetentries2_      DMCOMPOSITEGETENTRIES2
#define dmcompositegetentries3_      DMCOMPOSITEGETENTRIES3
#define dmcompositegetentries4_      DMCOMPOSITEGETENTRIES4
#define dmcompositegetentries5_      DMCOMPOSITEGETENTRIES5
#define dmcompositecreate_           DMCOMPOSITECREATE
#define dmcompositeaddda_            DMCOMPOSITEADDDA
#define dmcompositeaddarray_         DMCOMPOSITEADDARRAY
#define dmcompositedestroy_          DMCOMPOSITEDESTROY
#define dmcompositegetlocalvectors2_ DMCOMPOSITEGETLOCALVECTORS2
#define dmcompositerestorelocalvectors2_ DMCOMPOSITERESTORELOCALVECTORS2
#define dmcompositegetaccess4_           DMCOMPOSITEGETACCESS4
#define dmcompositescatter4_             DMCOMPOSITESCATTER4
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define dagetmatrix_                 dagetmatrix
#define dmcompositegetentries1_      dmcompositegetentries1
#define dmcompositegetentries2_      dmcompositegetentries2
#define dmcompositegetentries3_      dmcompositegetentries3
#define dmcompositegetentries4_      dmcompositegetentries4
#define dmcompositegetentries5_      dmcompositegetentries5
#define dmcompositecreate_           dmcompositecreate
#define dmcompositeaddda_            dmcompositeaddda
#define dmcompositedestroy_          dmcompositedestroy
#define dmcompositeaddarray_         dmcompositeaddarray
#define dmcompositegetlocalvectors2_ dmcompositegetlocalvectors2
#define dmcompositerestorelocalvectors2_ dmcompositerestorelocalvectors2
#define dmcompositegetaccess4_           dmcompositegetaccess4
#define dmcompositescatter4_             dmcompositescatter4
#endif

EXTERN_C_BEGIN
void PETSC_STDCALL dagetmatrix_(DA *da,CHAR mat_type PETSC_MIXED_LEN(len),Mat *J,PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;
  FIXCHAR(mat_type,len,t);
  *ierr = DAGetMatrix(*da,t,J);
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

void PETSC_STDCALL dmcompositeaddda_(DMComposite *dm,DA *da,PetscErrorCode *ierr)
{
  *ierr = DMCompositeAddDA(*dm,*da);
}

void PETSC_STDCALL dmcompositedestroy_(DMComposite *dm,PetscErrorCode *ierr)
{
  *ierr = DMCompositeDestroy(*dm);
}

void PETSC_STDCALL dmcompositeaddarray_(DMComposite *dm,PetscInt *r,PetscInt *n,PetscErrorCode *ierr)
{
  *ierr = DMCompositeAddArray(*dm,*r,*n);
}
EXTERN_C_END

#if defined(PETSC_HAVE_F90_C)
#include "src/sys/f90/f90impl.h"

EXTERN_C_BEGIN
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

#endif
