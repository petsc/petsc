
#include <petsc/private/vecimpl.h> /*I "petscvec.h" I*/
#include "../src/vec/vec/utils/tagger/impls/simple.h"

static PetscErrorCode VecTaggerComputeBoxes_Absolute(VecTagger tagger,Vec vec,PetscInt *numBoxes,VecTaggerBox **boxes)
{
  VecTagger_Simple *smpl = (VecTagger_Simple *)tagger->data;
  PetscInt       bs, i;
  VecTaggerBox   *bxs;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecTaggerGetBlockSize(tagger,&bs);CHKERRQ(ierr);
  *numBoxes = 1;
  ierr = PetscMalloc1(bs,&bxs);CHKERRQ(ierr);
  for (i = 0; i < bs; i++) {
    bxs[i].min = smpl->box[i].min;
    bxs[i].max = smpl->box[i].max;
  }
  *boxes = bxs;
  PetscFunctionReturn(0);
}

/*@C
  VecTaggerAbsoluteSetBox - Set the box defining the values to be tagged by the tagger.

  Logically Collective

  Input Arguments:
+ tagger - the VecTagger context
- box - the box: a blocksize array of VecTaggerBox boxes

  Level: advanced

.seealso: VecTaggerAbsoluteGetBox()
@*/
PetscErrorCode VecTaggerAbsoluteSetBox(VecTagger tagger,VecTaggerBox *box)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecTaggerSetBox_Simple(tagger,box);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  VecTaggerAbsoluteGetBox - Get the box defining the values to be tagged by the tagger.

  Logically Collective

  Input Arguments:
. tagger - the VecTagger context

  Output Arguments:
. box - the box: a blocksize array of VecTaggerBox boxes

  Level: advanced

.seealso: VecTaggerAbsoluteSetBox()
@*/
PetscErrorCode VecTaggerAbsoluteGetBox(VecTagger tagger,const VecTaggerBox **box)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecTaggerGetBox_Simple(tagger,box);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode VecTaggerCreate_Absolute(VecTagger tagger)
{
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = VecTaggerCreate_Simple(tagger);CHKERRQ(ierr);
  tagger->ops->computeboxes = VecTaggerComputeBoxes_Absolute;
  PetscFunctionReturn(0);
}
