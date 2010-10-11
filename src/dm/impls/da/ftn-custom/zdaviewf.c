
#include "private/fortranimpl.h"
#include "petscda.h"

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define daview_                      DAVIEW
#define dagetinfo_                   DAGETINFO
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define daview_                      daview
#define dagetinfo_                   dagetinfo
#endif

EXTERN_C_BEGIN
void PETSC_STDCALL daview_(DA *da,PetscViewer *vin,PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(vin,v);
  *ierr = DAView(*da,v);
}

void PETSC_STDCALL dagetinfo_(DA *da,PetscInt *dim,PetscInt *M,PetscInt *N,PetscInt *P,PetscInt *m,PetscInt *n,PetscInt *p,PetscInt *w,PetscInt *s,
                DAPeriodicType *wrap,DAStencilType *st,PetscErrorCode *ierr)
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
  *ierr = DAGetInfo(*da,dim,M,N,P,m,n,p,w,s,wrap,st);
}
EXTERN_C_END
