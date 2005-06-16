#include "zpetsc.h"
#include "petscksp.h"

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define pcasmgetsubksp_            PCASMGETSUBKSP
#define pcasmsetlocalsubdomains_   PCASMSETLOCALSUBDOMAINS
#define pcasmsetglobalsubdomains_  PCASMSETGLOBALSUBDOMAINS
#define pcasmgetlocalsubmatrices_  PCASMGETLOCALSUBMATRICES
#define pcasmgetlocalsubdomains_   PCASMGETLOCALSUBDOMAINS
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define pcasmgetsubksp_            pcasmgetsubksp
#define pcasmsetlocalsubdomains_   pcasmsetlocalsubdomains
#define pcasmsetglobalsubdomains_  pcasmsetglobalsubdomains
#define pcasmgetlocalsubmatrices_  pcasmgetlocalsubmatrices
#define pcasmgetlocalsubdomains_   pcasmgetlocalsubdomains
#endif

EXTERN_C_BEGIN
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

void PETSC_STDCALL pcasmsetlocalsubdomains_(PC *pc,PetscInt *n,IS *is, PetscErrorCode *ierr)
{
  CHKFORTRANNULLOBJECT(is);
  *ierr = PCASMSetLocalSubdomains(*pc,*n,is);
}

void PETSC_STDCALL pcasmsettotalsubdomains_(PC *pc,PetscInt *N,IS *is, PetscErrorCode *ierr)
{
  CHKFORTRANNULLOBJECT(is);
  *ierr = PCASMSetTotalSubdomains(*pc,*N,is);
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
void PETSC_STDCALL pcasmgetlocalsubdomains_(PC *pc,PetscInt *n,IS *is, PetscErrorCode *ierr)
{
  PetscInt nloc,i;
  IS  *tis;
  CHKFORTRANNULLOBJECT(is);
  CHKFORTRANNULLINTEGER(n);
  *ierr = PCASMGetLocalSubdomains(*pc,&nloc,&tis);
  if (n) *n = nloc;
  if (is) {
    for (i=0; i<nloc; i++){
      is[i] = tis[i];
    }
  }
}

EXTERN_C_END
