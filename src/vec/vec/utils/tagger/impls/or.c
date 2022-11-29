
#include <petsc/private/vecimpl.h> /*I "petscvec.h" I*/
#include "../src/vec/vec/utils/tagger/impls/andor.h"

/*@C
  VecTaggerOrGetSubs - Get the sub VecTaggers whose union defines the outer VecTagger

  Not collective

  Input Parameter:
. tagger - the VecTagger context

  Output Parameters:
+ nsubs - the number of sub VecTaggers
- subs - the sub VecTaggers

  Level: advanced

.seealso: `VecTaggerOrSetSubs()`
@*/
PetscErrorCode VecTaggerOrGetSubs(VecTagger tagger, PetscInt *nsubs, VecTagger **subs)
{
  PetscFunctionBegin;
  PetscCall(VecTaggerGetSubs_AndOr(tagger, nsubs, subs));
  PetscFunctionReturn(0);
}

/*@C
  VecTaggerOrSetSubs - Set the sub VecTaggers whose union defines the outer VecTagger

  Logically collective

  Input Parameters:
+ tagger - the VecTagger context
. nsubs - the number of sub VecTaggers
- subs - the sub VecTaggers

  Level: advanced

.seealso: `VecTaggerOrSetSubs()`
@*/
PetscErrorCode VecTaggerOrSetSubs(VecTagger tagger, PetscInt nsubs, VecTagger *subs, PetscCopyMode mode)
{
  PetscFunctionBegin;
  PetscCall(VecTaggerSetSubs_AndOr(tagger, nsubs, subs, mode));
  PetscFunctionReturn(0);
}

static PetscErrorCode VecTaggerComputeBoxes_Or(VecTagger tagger, Vec vec, PetscInt *numBoxes, VecTaggerBox **boxes, PetscBool *listed)
{
  PetscInt       i, bs, nsubs, *numSubBoxes, nboxes, total;
  VecTaggerBox **subBoxes;
  VecTagger     *subs;
  VecTaggerBox  *bxs;
  PetscBool      boxlisted;

  PetscFunctionBegin;
  PetscCall(VecTaggerGetBlockSize(tagger, &bs));
  PetscCall(VecTaggerOrGetSubs(tagger, &nsubs, &subs));
  PetscCall(PetscMalloc2(nsubs, &numSubBoxes, nsubs, &subBoxes));
  for (i = 0, total = 0; i < nsubs; i++) {
    PetscCall(VecTaggerComputeBoxes(subs[i], vec, &numSubBoxes[i], &subBoxes[i], &boxlisted));
    if (!boxlisted) { /* no support, clean up and exit */
      PetscInt j;

      for (j = 0; j < i; j++) PetscCall(PetscFree(subBoxes[j]));
      PetscCall(PetscFree2(numSubBoxes, subBoxes));
      if (listed) *listed = PETSC_FALSE;
    }
    total += numSubBoxes[i];
  }
  PetscCall(PetscMalloc1(bs * total, &bxs));
  for (i = 0, nboxes = 0; i < nsubs; i++) { /* stupid O(N^2) check to remove subboxes */
    PetscInt j;

    for (j = 0; j < numSubBoxes[i]; j++) {
      PetscInt      k;
      VecTaggerBox *subBox = &subBoxes[i][j * bs];

      for (k = 0; k < nboxes; k++) {
        PetscBool isSub = PETSC_FALSE;

        VecTaggerBox *prevBox = &bxs[bs * k];
        PetscCall(VecTaggerAndOrIsSubBox_Private(bs, prevBox, subBox, &isSub));
        if (isSub) break;
        PetscCall(VecTaggerAndOrIsSubBox_Private(bs, subBox, prevBox, &isSub));
        if (isSub) {
          PetscInt l;

          for (l = 0; l < bs; l++) prevBox[l] = subBox[l];
          break;
        }
      }
      if (k < nboxes) continue;
      for (k = 0; k < bs; k++) bxs[nboxes * bs + k] = subBox[k];
      nboxes++;
    }
    PetscCall(PetscFree(subBoxes[i]));
  }
  PetscCall(PetscFree2(numSubBoxes, subBoxes));
  *numBoxes = nboxes;
  *boxes    = bxs;
  if (listed) *listed = PETSC_TRUE;
  PetscFunctionReturn(0);
}

static PetscErrorCode VecTaggerComputeIS_Or(VecTagger tagger, Vec vec, IS *is, PetscBool *listed)
{
  PetscInt   nsubs, i;
  VecTagger *subs;
  IS         unionIS;
  PetscBool  boxlisted;

  PetscFunctionBegin;
  PetscCall(VecTaggerComputeIS_FromBoxes(tagger, vec, is, &boxlisted));
  if (boxlisted) {
    if (listed) *listed = PETSC_TRUE;
    PetscFunctionReturn(0);
  }
  PetscCall(VecTaggerOrGetSubs(tagger, &nsubs, &subs));
  PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)vec), 0, NULL, PETSC_OWN_POINTER, &unionIS));
  for (i = 0; i < nsubs; i++) {
    IS subIS, newUnionIS;

    PetscCall(VecTaggerComputeIS(subs[i], vec, &subIS, &boxlisted));
    PetscCheck(boxlisted, PetscObjectComm((PetscObject)tagger), PETSC_ERR_SUP, "Tagger cannot VecTaggerComputeIS()");
    PetscCall(ISExpand(unionIS, subIS, &newUnionIS));
    PetscCall(ISSort(newUnionIS));
    PetscCall(ISDestroy(&unionIS));
    unionIS = newUnionIS;
    PetscCall(ISDestroy(&subIS));
  }
  *is = unionIS;
  if (listed) *listed = PETSC_TRUE;
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode VecTaggerCreate_Or(VecTagger tagger)
{
  PetscFunctionBegin;
  PetscCall(VecTaggerCreate_AndOr(tagger));
  tagger->ops->computeboxes = VecTaggerComputeBoxes_Or;
  tagger->ops->computeis    = VecTaggerComputeIS_Or;
  PetscFunctionReturn(0);
}
