#include "private/fortranimpl.h"
#include "petscksp.h"

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define pcasmgetsubksp_            PCASMGETSUBKSP
#define pcasmsetlocalsubdomains_   PCASMSETLOCALSUBDOMAINS
#define pcasmsetglobalsubdomains_  PCASMSETGLOBALSUBDOMAINS
#define pcasmgetlocalsubmatrices_  PCASMGETLOCALSUBMATRICES
#define pcasmgetlocalsubdomains_   PCASMGETLOCALSUBDOMAINS
#define pcasmcreatesubdomains_     PCASMCREATESUBDOMAINS
#define pcasmdestroysubdomains_    PCASMDESTROYSUBDOMAINS
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define pcasmgetsubksp_            pcasmgetsubksp
#define pcasmsetlocalsubdomains_   pcasmsetlocalsubdomains
#define pcasmsetglobalsubdomains_  pcasmsetglobalsubdomains
#define pcasmgetlocalsubmatrices_  pcasmgetlocalsubmatrices
#define pcasmgetlocalsubdomains_   pcasmgetlocalsubdomains
#define pcasmcreatesubdomains_     pcasmcreatesubdomains
#define pcasmdestroysubdomains_    pcasmdestroysubdomains
#endif

EXTERN_C_BEGIN
void PETSC_STDCALL pcasmcreatesubdomains_(Mat *mat,PetscInt *n,IS *subs,PetscErrorCode *ierr)
{
  PetscInt i;
  IS       *insubs;

  *ierr = PCASMCreateSubdomains(*mat,*n,&insubs);if (*ierr) return;
  for (i=0; i<*n; i++) {
    subs[i] = insubs[i];
  }
  *ierr = PetscFree(insubs); 
}


void PETSC_STDCALL pcasmdestroysubdomains_(Mat *mat,PetscInt *n,IS *subs,PetscErrorCode *ierr)
{
  PetscInt i;

  for (i=0; i<*n; i++) {
    *ierr = ISDestroy(subs[i]);if (*ierr) return;
  }
}

void PETSC_STDCALL pcasmgetsubksp_(PC *pc,PetscInt *n_local,PetscInt *first_local,KSP *ksp,PetscErrorCode *ierr)
{
  KSP *tksp;
  PetscInt  i,nloc;
  CHKFORTRANNULLINTEGER(n_local);
  CHKFORTRANNULLINTEGER(first_local);
  *ierr = PCASMGetSubKSP(*pc,&nloc,first_local,&tksp);
  if (n_local) *n_local = nloc;
  for (i=0; i<nloc; i++){
    ksp[i] = tksp[i];
  }
}

void PETSC_STDCALL pcasmsetlocalsubdomains_(PC *pc,PetscInt *n,IS *is,IS *is_local, PetscErrorCode *ierr)
{
  CHKFORTRANNULLOBJECT(is);
  CHKFORTRANNULLOBJECT(is_local);
  *ierr = PCASMSetLocalSubdomains(*pc,*n,is,is_local);
}

void PETSC_STDCALL pcasmsettotalsubdomains_(PC *pc,PetscInt *N,IS *is,IS *is_local, PetscErrorCode *ierr)
{
  CHKFORTRANNULLOBJECT(is);
  CHKFORTRANNULLOBJECT(is_local);
  *ierr = PCASMSetTotalSubdomains(*pc,*N,is,is_local);
}

void PETSC_STDCALL pcasmgetlocalsubmatrices_(PC *pc,PetscInt *n,Mat *mat, PetscErrorCode *ierr)
{
  PetscInt nloc,i;
  Mat  *tmat;
  CHKFORTRANNULLOBJECT(mat);
  CHKFORTRANNULLINTEGER(n);
  *ierr = PCASMGetLocalSubmatrices(*pc,&nloc,&tmat);
  if (n) *n = nloc;
  if (mat) {
    for (i=0; i<nloc; i++){
      mat[i] = tmat[i];
    }
  }
}
void PETSC_STDCALL pcasmgetlocalsubdomains_(PC *pc,PetscInt *n,IS *is,IS *is_local, PetscErrorCode *ierr)
{
  PetscInt nloc,i;
  IS  *tis, *tis_local;
  CHKFORTRANNULLOBJECT(is);
  CHKFORTRANNULLOBJECT(is_local);
  CHKFORTRANNULLINTEGER(n);
  *ierr = PCASMGetLocalSubdomains(*pc,&nloc,&tis,&tis_local);
  if (n) *n = nloc;
  if (is) {
    for (i=0; i<nloc; i++){
      is[i] = tis[i];
    }
  }
  if (is_local && tis_local) {
    for (i=0; i<nloc; i++){
      is[i] = tis_local[i];
    }
  }
}

EXTERN_C_END
