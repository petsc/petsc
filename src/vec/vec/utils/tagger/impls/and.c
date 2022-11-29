
#include <petsc/private/vecimpl.h> /*I "petscvec.h" I*/
#include "../src/vec/vec/utils/tagger/impls/andor.h"

/*@C
  VecTaggerAndGetSubs - Get the sub VecTaggers whose intersection defines the outer VecTagger

  Not collective

  Input Parameter:
. tagger - the VecTagger context

  Output Parameters:
+ nsubs - the number of sub VecTaggers
- subs - the sub VecTaggers

  Level: advanced

.seealso: `VecTaggerAndSetSubs()`
@*/
PetscErrorCode VecTaggerAndGetSubs(VecTagger tagger, PetscInt *nsubs, VecTagger **subs)
{
  PetscFunctionBegin;
  PetscCall(VecTaggerGetSubs_AndOr(tagger, nsubs, subs));
  PetscFunctionReturn(0);
}

/*@C
  VecTaggerAndSetSubs - Set the sub VecTaggers whose intersection defines the outer VecTagger

  Logically collective

  Input Parameters:
+ tagger - the VecTagger context
. nsubs - the number of sub VecTaggers
- subs - the sub VecTaggers

  Level: advanced

.seealso: `VecTaggerAndSetSubs()`
@*/
PetscErrorCode VecTaggerAndSetSubs(VecTagger tagger, PetscInt nsubs, VecTagger *subs, PetscCopyMode mode)
{
  PetscFunctionBegin;
  PetscCall(VecTaggerSetSubs_AndOr(tagger, nsubs, subs, mode));
  PetscFunctionReturn(0);
}

static PetscErrorCode VecTaggerComputeBoxes_And(VecTagger tagger, Vec vec, PetscInt *numBoxes, VecTaggerBox **boxes, PetscBool *listed)
{
  PetscInt       i, bs, nsubs, *numSubBoxes, nboxes;
  VecTaggerBox **subBoxes;
  VecTagger     *subs;
  VecTaggerBox  *bxs = NULL;
  PetscBool      sublisted;

  PetscFunctionBegin;
  PetscCall(VecTaggerGetBlockSize(tagger, &bs));
  PetscCall(VecTaggerOrGetSubs(tagger, &nsubs, &subs));
  PetscCall(PetscMalloc2(nsubs, &numSubBoxes, nsubs, &subBoxes));
  for (i = 0; i < nsubs; i++) {
    PetscCall(VecTaggerComputeBoxes(subs[i], vec, &numSubBoxes[i], &subBoxes[i], &sublisted));
    if (!sublisted) {
      PetscInt j;

      for (j = 0; j < i; j++) PetscCall(PetscFree(subBoxes[j]));
      PetscCall(PetscFree2(numSubBoxes, subBoxes));
      *listed = PETSC_FALSE;
      PetscFunctionReturn(0);
    }
  }
  for (i = 0, nboxes = 0; i < nsubs; i++) { /* stupid O(N^3) check to intersect boxes */
    VecTaggerBox *isect;
    PetscInt      j, k, l, m, n;

    n = numSubBoxes[i];
    if (!n) {
      nboxes = 0;
      PetscCall(PetscFree(bxs));
      break;
    }
    if (!i) {
      PetscCall(PetscMalloc1(n * bs, &bxs));
      for (j = 0; j < numSubBoxes[i] * bs; j++) bxs[j] = subBoxes[i][j];
      nboxes = n;
      PetscCall(PetscFree(subBoxes[i]));
      continue;
    }
    PetscCall(PetscMalloc1(n * nboxes * bs, &isect));
    for (j = 0, l = 0; j < n; j++) {
      VecTaggerBox *subBox = &subBoxes[i][j * bs];

      for (k = 0; k < nboxes; k++) {
        PetscBool     isEmpty;
        VecTaggerBox *prevBox = &bxs[bs * k];

        PetscCall(VecTaggerAndOrIntersect_Private(bs, prevBox, subBox, &isect[l * bs], &isEmpty));
        if (isEmpty) continue;
        for (m = 0; m < l; m++) {
          PetscBool isSub = PETSC_FALSE;

          PetscCall(VecTaggerAndOrIsSubBox_Private(bs, &isect[m * bs], &isect[l * bs], &isSub));
          if (isSub) break;
          PetscCall(VecTaggerAndOrIsSubBox_Private(bs, &isect[l * bs], &isect[m * bs], &isSub));
          if (isSub) {
            PetscInt r;

            for (r = 0; r < bs; r++) isect[m * bs + r] = isect[l * bs + r];
            break;
          }
        }
        if (m == l) l++;
      }
    }
    PetscCall(PetscFree(bxs));
    bxs    = isect;
    nboxes = l;
    PetscCall(PetscFree(subBoxes[i]));
  }
  PetscCall(PetscFree2(numSubBoxes, subBoxes));
  *numBoxes = nboxes;
  *boxes    = bxs;
  if (listed) *listed = PETSC_TRUE;
  PetscFunctionReturn(0);
}

static PetscErrorCode VecTaggerComputeIS_And(VecTagger tagger, Vec vec, IS *is, PetscBool *listed)
{
  PetscInt   nsubs, i;
  VecTagger *subs;
  IS         isectIS;
  PetscBool  boxlisted;

  PetscFunctionBegin;
  PetscCall(VecTaggerComputeIS_FromBoxes(tagger, vec, is, &boxlisted));
  if (boxlisted) {
    if (listed) *listed = PETSC_TRUE;
    PetscFunctionReturn(0);
  }
  PetscCall(VecTaggerOrGetSubs(tagger, &nsubs, &subs));
  if (!nsubs) {
    PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)vec), 0, NULL, PETSC_OWN_POINTER, is));
    PetscFunctionReturn(0);
  }
  PetscCall(VecTaggerComputeIS(subs[0], vec, &isectIS, &boxlisted));
  PetscCheck(boxlisted, PetscObjectComm((PetscObject)tagger), PETSC_ERR_SUP, "Does not support VecTaggerComputeIS()");
  for (i = 1; i < nsubs; i++) {
    IS subIS, newIsectIS;

    PetscCall(VecTaggerComputeIS(subs[i], vec, &subIS, &boxlisted));
    PetscCheck(boxlisted, PetscObjectComm((PetscObject)tagger), PETSC_ERR_SUP, "Does not support VecTaggerComputeIS()");
    PetscCall(ISIntersect(isectIS, subIS, &newIsectIS));
    PetscCall(ISDestroy(&isectIS));
    PetscCall(ISDestroy(&subIS));
    isectIS = newIsectIS;
  }
  *is = isectIS;
  if (listed) *listed = PETSC_TRUE;
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode VecTaggerCreate_And(VecTagger tagger)
{
  PetscFunctionBegin;
  PetscCall(VecTaggerCreate_AndOr(tagger));
  tagger->ops->computeboxes = VecTaggerComputeBoxes_And;
  tagger->ops->computeis    = VecTaggerComputeIS_And;
  PetscFunctionReturn(0);
}
