#include <petsc/private/dmpleximpl.h>   /*I      "petscdmplex.h"   I*/
#include <petscdmplextransform.h>

PetscBool DMPlexGenerateRegisterAllCalled = PETSC_FALSE;

#if defined(PETSC_HAVE_TRIANGLE)
PETSC_EXTERN PetscErrorCode DMPlexGenerate_Triangle(DM, PetscBool, DM*);
PETSC_EXTERN PetscErrorCode DMPlexRefine_Triangle(DM, double*, DM*);
#endif
#if defined(PETSC_HAVE_TETGEN)
PETSC_EXTERN PetscErrorCode DMPlexGenerate_Tetgen(DM, PetscBool, DM*);
PETSC_EXTERN PetscErrorCode DMPlexRefine_Tetgen(DM, double*, DM*);
#endif
#if defined(PETSC_HAVE_CTETGEN)
PETSC_EXTERN PetscErrorCode DMPlexGenerate_CTetgen(DM, PetscBool, DM*);
PETSC_EXTERN PetscErrorCode DMPlexRefine_CTetgen(DM, double*, DM*);
#endif

/*@C
  DMPlexGenerateRegisterAll - Registers all of the mesh generation methods in the DMPlexGenerate package.

  Not Collective

  Level: advanced

.seealso:  DMPlexGenerateRegisterDestroy()
@*/
PetscErrorCode  DMPlexGenerateRegisterAll(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (DMPlexGenerateRegisterAllCalled) PetscFunctionReturn(0);
  DMPlexGenerateRegisterAllCalled = PETSC_TRUE;
#if defined(PETSC_HAVE_TRIANGLE)
  ierr = DMPlexGenerateRegister("triangle",DMPlexGenerate_Triangle,DMPlexRefine_Triangle,NULL,1);CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_CTETGEN)
  ierr = DMPlexGenerateRegister("ctetgen",DMPlexGenerate_CTetgen,DMPlexRefine_CTetgen,NULL,2);CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_TETGEN)
  ierr = DMPlexGenerateRegister("tetgen",DMPlexGenerate_Tetgen,DMPlexRefine_Tetgen,NULL,2);CHKERRQ(ierr);
#endif
  ierr = DMPlexGenerateRegister("cellrefiner",NULL,NULL,DMPlexTransformAdaptLabel,-1);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
