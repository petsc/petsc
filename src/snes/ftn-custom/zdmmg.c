
#include "private/fortranimpl.h"
#include "petscdmmg.h"

#ifdef PETSC_HAVE_FORTRAN_CAPS
#define dmmggetda_               DMMGGETDA
#define dmmggetx_                DMMGGETX
#define dmmggetj_                DMMGGETJ
#define dmmggetb_                DMMGGETB
#define dmmggetrhs_              DMMGGETRHS
#define dmmggetksp_              DMMGGETKSP
#define dmmggetlevels_           DMMGGETLEVELS
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define dmmggetrhs_              dmmggetrhs
#define dmmggetx_                dmmggetx
#define dmmggetj_                dmmggetj
#define dmmggetb_                dmmggetb
#define dmmggetksp_              dmmggetksp
#define dmmggetda_               dmmggetda
#define dmmggetlevels_           dmmggetlevels
#endif

EXTERN_C_BEGIN

void PETSC_STDCALL dmmggetx_(DMMG **dmmg,Vec *x,PetscErrorCode *ierr)
{
  *ierr = 0;
  *x    = DMMGGetx(*dmmg);
}

void PETSC_STDCALL dmmggetj_(DMMG **dmmg,Mat *x,PetscErrorCode *ierr)
{
  *ierr = 0;
  *x    = DMMGGetJ(*dmmg);
}

void PETSC_STDCALL dmmggetb_(DMMG **dmmg,Mat *x,PetscErrorCode *ierr)
{
  *ierr = 0;
  *x    = DMMGGetB(*dmmg);
}

void PETSC_STDCALL dmmggetrhs_(DMMG **dmmg,Vec *x,PetscErrorCode *ierr)
{
  *ierr = 0;
  *x    = DMMGGetRHS(*dmmg);
}

void PETSC_STDCALL dmmggetksp_(DMMG **dmmg,KSP *x,PetscErrorCode *ierr)
{
  *ierr = 0;
  *x    = DMMGGetKSP(*dmmg);
}

void PETSC_STDCALL dmmggetlevels_(DMMG **dmmg,PetscInt *x,PetscErrorCode *ierr)
{
  *ierr = 0;
  *x    = DMMGGetLevels(*dmmg);
}

/* ----------------------------------------------------------------------------------------------------------*/

void PETSC_STDCALL dmmggetda_(DMMG *dmmg,DA *da,PetscErrorCode *ierr)
{
  *da   = (DA)(*dmmg)->dm;
  *ierr = 0;
}

EXTERN_C_END


