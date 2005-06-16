#include "zpetsc.h"
#include "petscsnes.h"
#include "petscda.h"

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define dmmgsetsnes_                     DMMGSETSNES
#define snesgetsolutionupdate_           SNESGETSOLUTIONUPDATE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define dmmgsetsnes_                     dmmgsetsnes
#define snesgetsolutionupdate_           snesgetsolutionupdate
#endif

EXTERN_C_BEGIN

#if defined(notused)
static PetscErrorCode ourrhs(SNES snes,Vec vec,Vec vec2,void*ctx)
{
  PetscErrorCode ierr = 0;
  DMMG *dmmg = (DMMG*)ctx;
  (*(PetscErrorCode (PETSC_STDCALL *)(SNES*,Vec*,Vec*,PetscErrorCode*))(((PetscObject)dmmg->dm)->fortran_func_pointers[0]))(&snes,&vec,&vec2,&ierr);
  return ierr;
}

static PetscErrorCode ourmat(DMMG dmmg,Mat mat)
{
  PetscErrorCode ierr = 0;
  (*(PetscErrorCode (PETSC_STDCALL *)(DMMG*,Vec*,PetscErrorCode*))(((PetscObject)dmmg->dm)->fortran_func_pointers[1]))(&dmmg,&vec,&ierr);
  return ierr;
}

void PETSC_STDCALL dmmgsetsnes_(DMMG **dmmg,PetscErrorCode (PETSC_STDCALL *rhs)(SNES*,Vec*,Vec*,PetscErrorCode*),PetscErrorCode (PETSC_STDCALL *mat)(DMMG*,Mat*,PetscErrorCode*),PetscErrorCode *ierr)
{
  PetscInt i;
  theirmat = mat;
  *ierr = DMMGSetSNES(*dmmg,ourrhs,ourmat,*dmmg);
  /*
    Save the fortran rhs function in the DM on each level; ourrhs() pulls it out when needed
  */
  for (i=0; i<(**dmmg)->nlevels; i++) {
    ((PetscObject)(*dmmg)[i]->dm)->fortran_func_pointers[0] = (FCNVOID)rhs;
    ((PetscObject)(*dmmg)[i]->dm)->fortran_func_pointers[1] = (FCNVOID)mat;
  }
}

#endif

EXTERN_C_END
