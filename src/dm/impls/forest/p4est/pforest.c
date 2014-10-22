#include <petsc-private/dmforestimpl.h>

#if defined(PETSC_HAVE_P4EST)
#define _pforest_string(a) #a

#if !defined(P4_TO_P8)
#include <p4est.h>
#else
#include <p8est.h>
#endif

#define DMCreate_pforest _append_pforest(DMCreate)
#undef __FUNCT__
#define __FUNCT__ _pforest_string(DMCreate_pforest)
PETSC_EXTERN PetscErrorCode DMCreate_pforest(DM dm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  DMCreate_Forest(dm);
  ierr = DMSetDimension(dm,P4EST_DIM);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#endif /* defined(PETSC_HAVE_P4EST) */
