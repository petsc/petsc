
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
