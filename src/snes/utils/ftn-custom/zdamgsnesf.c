#include "private/fortranimpl.h"
#include "petscsnes.h"
#include "petscdmmg.h"

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define dmmgsetsnes_                     DMMGSETSNES
#define snesgetsolutionupdate_           SNESGETSOLUTIONUPDATE
#define dmmggetsnes_                     DMMGGETSNES
#define dmmgsetfromoptions_              DMMGSETFROMOPTIONS
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define dmmgsetsnes_                     dmmgsetsnes
#define snesgetsolutionupdate_           snesgetsolutionupdate
#define dmmggetsnes_                     dmmggetsnes
#define dmmgsetfromoptions_              dmmgsetfromoptions
#endif

EXTERN_C_BEGIN

static PetscErrorCode ourrhs(SNES snes,Vec vec,Vec vec2,void*ctx)
{
  PetscErrorCode ierr = 0;
  DMMG dmmg = (DMMG)ctx;
  (*(PetscErrorCode (PETSC_STDCALL *)(SNES*,Vec*,Vec*,void *,PetscErrorCode*))(((PetscObject)(dmmg)->dm)->fortran_func_pointers[0]))(&snes,&vec,&vec2,&ctx,&ierr);
  return ierr;
}

void PETSC_STDCALL dmmgsetsnes_(DMMG **dmmg,PetscErrorCode (PETSC_STDCALL *rhs)(SNES*,Vec*,Vec*,void*,PetscErrorCode*),PetscErrorCode (PETSC_STDCALL *mat)(DMMG*,Mat*,PetscErrorCode*),PetscErrorCode *ierr)
{
  PetscInt i;
  *ierr = DMMGSetSNES(*dmmg,ourrhs,PETSC_NULL); if (*ierr) return;
  /*
    Save the fortran rhs function in the DM on each level; ourrhs() pulls it out when needed
  */
  for (i=0; i<(**dmmg)->nlevels; i++) {
    ((PetscObject)(*dmmg)[i]->dm)->fortran_func_pointers[0] = (PetscVoidFunction)rhs;
    ((PetscObject)(*dmmg)[i]->dm)->fortran_func_pointers[1] = (PetscVoidFunction)mat;
  }
}

void PETSC_STDCALL dmmggetsnes_(DMMG **dmmg,SNES *snes,PetscErrorCode *ierr)
{
  *snes = DMMGGetSNES(*dmmg);
}

void PETSC_STDCALL dmmgsetfromoptions_(DMMG **dmmg,PetscErrorCode *ierr)
{
  *ierr = DMMGSetFromOptions(*dmmg);
}

EXTERN_C_END
