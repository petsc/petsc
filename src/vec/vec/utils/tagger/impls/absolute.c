
#include <petsc/private/vecimpl.h> /*I "petscvec.h" I*/
#include "../src/vec/vec/utils/tagger/impls/simple.h"

static PetscErrorCode VecTaggerComputeIntervals_Absolute(VecTagger tagger,Vec vec,PetscInt *numIntervals,PetscScalar (**intervals)[2])
{
  VecTagger_Simple *smpl = (VecTagger_Simple *)tagger->data;
  PetscInt       bs, i;
  PetscScalar    (*ints) [2];
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecTaggerGetBlockSize(tagger,&bs);CHKERRQ(ierr);
  *numIntervals = 1;
  ierr = PetscMalloc1(bs,&ints);CHKERRQ(ierr);
  for (i = 0; i < bs; i++) {
    ints[i][0] = smpl->interval[i][0];
    ints[i][1] = smpl->interval[i][1];
  }
  *intervals = ints;
  PetscFunctionReturn(0);
}

/*@C
  VecTaggerAbsoluteSetInterval - Set the interval (multi-dimensional box) defining the values to be tagged by the tagger.

  Logically Collective

  Input Arguments:
+ tagger - the VecTagger context
- interval - the interval: a blocksize list of [min,max] pairs

  Level: advanced

.seealso: VecTaggerAbsoluteGetInterval()
@*/
PetscErrorCode VecTaggerAbsoluteSetInterval(VecTagger tagger,PetscScalar (*interval)[2])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecTaggerSetInterval_Simple(tagger,interval);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  VecTaggerAbsoluteGetInterval - Get the interval (multi-dimensional box) defining the values to be tagged by the tagger.

  Logically Collective

  Input Arguments:
. tagger - the VecTagger context

  Output Arguments:
. interval - the interval: a blocksize list of [min,max] pairs

  Level: advanced

.seealso: VecTaggerAbsoluteSetInterval()
@*/
PetscErrorCode VecTaggerAbsoluteGetInterval(VecTagger tagger,const PetscScalar (**interval)[2])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecTaggerGetInterval_Simple(tagger,interval);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode VecTaggerCreate_Absolute(VecTagger tagger)
{
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = VecTaggerCreate_Simple(tagger);CHKERRQ(ierr);
  tagger->ops->computeintervals = VecTaggerComputeIntervals_Absolute;
  PetscFunctionReturn(0);
}
