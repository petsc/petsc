
#include "private/fortranimpl.h"
#include "petscda.h"

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define dmdagetinfo_                   DMDAGETINFO
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define dmdagetinfo_                   dmdagetinfo
#endif

EXTERN_C_BEGIN
void PETSC_STDCALL dmdagetinfo_(DM *da,PetscInt *dim,PetscInt *M,PetscInt *N,PetscInt *P,PetscInt *m,PetscInt *n,PetscInt *p,PetscInt *w,PetscInt *s,
                DMDAPeriodicType *wrap,DMDAStencilType *st,PetscErrorCode *ierr)
{
  CHKFORTRANNULLINTEGER(dim);
  CHKFORTRANNULLINTEGER(M);
  CHKFORTRANNULLINTEGER(N);
  CHKFORTRANNULLINTEGER(P);
  CHKFORTRANNULLINTEGER(m);
  CHKFORTRANNULLINTEGER(n);
  CHKFORTRANNULLINTEGER(p);
  CHKFORTRANNULLINTEGER(w);
  CHKFORTRANNULLINTEGER(s);
  CHKFORTRANNULLINTEGER(wrap);
  CHKFORTRANNULLINTEGER(st);
  *ierr = DMDAGetInfo(*da,dim,M,N,P,m,n,p,w,s,wrap,st);
}
EXTERN_C_END
