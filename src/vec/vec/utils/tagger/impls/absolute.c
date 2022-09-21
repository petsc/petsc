
#include <petsc/private/vecimpl.h> /*I "petscvec.h" I*/
#include "../src/vec/vec/utils/tagger/impls/simple.h"

static PetscErrorCode VecTaggerComputeBoxes_Absolute(VecTagger tagger, Vec vec, PetscInt *numBoxes, VecTaggerBox **boxes, PetscBool *listed)
{
  VecTagger_Simple *smpl = (VecTagger_Simple *)tagger->data;
  PetscInt          bs, i;
  VecTaggerBox     *bxs;

  PetscFunctionBegin;
  PetscCall(VecTaggerGetBlockSize(tagger, &bs));
  *numBoxes = 1;
  PetscCall(PetscMalloc1(bs, &bxs));
  for (i = 0; i < bs; i++) {
    bxs[i].min = smpl->box[i].min;
    bxs[i].max = smpl->box[i].max;
  }
  *boxes = bxs;
  if (listed) *listed = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*@C
  VecTaggerAbsoluteSetBox - Set the box defining the values to be tagged by the tagger.

  Logically Collective

  Input Parameters:
+ tagger - the VecTagger context
- box - the box: a blocksize array of VecTaggerBox boxes

  Level: advanced

.seealso: `VecTaggerAbsoluteGetBox()`
@*/
PetscErrorCode VecTaggerAbsoluteSetBox(VecTagger tagger, VecTaggerBox *box)
{
  PetscFunctionBegin;
  PetscCall(VecTaggerSetBox_Simple(tagger, box));
  PetscFunctionReturn(0);
}

/*@C
  VecTaggerAbsoluteGetBox - Get the box defining the values to be tagged by the tagger.

  Logically Collective

  Input Parameter:
. tagger - the VecTagger context

  Output Parameter:
. box - the box: a blocksize array of VecTaggerBox boxes

  Level: advanced

.seealso: `VecTaggerAbsoluteSetBox()`
@*/
PetscErrorCode VecTaggerAbsoluteGetBox(VecTagger tagger, const VecTaggerBox **box)
{
  PetscFunctionBegin;
  PetscCall(VecTaggerGetBox_Simple(tagger, box));
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode VecTaggerCreate_Absolute(VecTagger tagger)
{
  PetscFunctionBegin;
  PetscCall(VecTaggerCreate_Simple(tagger));
  tagger->ops->computeboxes = VecTaggerComputeBoxes_Absolute;
  PetscFunctionReturn(0);
}
