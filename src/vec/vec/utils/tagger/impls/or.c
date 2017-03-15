
#include <petsc/private/vecimpl.h> /*I "petscvec.h" I*/
#include "../src/vec/vec/utils/tagger/impls/andor.h"

/*@C
  VecTaggerOrGetSubs - Get the sub VecTaggers whose union defines the outer VecTagger

  Not collective

  Input Arguments:
. tagger - the VecTagger context

  Output Arguments:
+ nsubs - the number of sub VecTaggers
- subs - the sub VecTaggers

  Level: advanced

.seealso: VecTaggerOrSetSubs()
@*/
PetscErrorCode VecTaggerOrGetSubs(VecTagger tagger, PetscInt *nsubs, VecTagger **subs)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecTaggerGetSubs_AndOr(tagger,nsubs,subs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  VecTaggerOrSetSubs - Set the sub VecTaggers whose union defines the outer VecTagger

  Logically collective

  Input Arguments:
+ tagger - the VecTagger context
. nsubs - the number of sub VecTaggers
- subs - the sub VecTaggers

  Level: advanced

.seealso: VecTaggerOrSetSubs()
@*/
PetscErrorCode VecTaggerOrSetSubs(VecTagger tagger, PetscInt nsubs, VecTagger *subs, PetscCopyMode mode)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecTaggerSetSubs_AndOr(tagger,nsubs,subs,mode);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode VecTaggerComputeBoxes_Or(VecTagger tagger,Vec vec,PetscInt *numBoxes,VecTaggerBox **boxes)
{
  PetscInt        i, bs, nsubs, *numSubBoxes, nboxes, total;
  VecTaggerBox    **subBoxes;
  VecTagger       *subs;
  VecTaggerBox    *bxs;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = VecTaggerGetBlockSize(tagger,&bs);CHKERRQ(ierr);
  ierr = VecTaggerOrGetSubs(tagger,&nsubs,&subs);CHKERRQ(ierr);
  ierr = PetscMalloc2(nsubs,&numSubBoxes,nsubs,&subBoxes);CHKERRQ(ierr);
  for (i = 0, total = 0; i < nsubs; i++) {
    PetscErrorCode ierr2;

    ierr2 = VecTaggerComputeBoxes(subs[i],vec,&numSubBoxes[i],&subBoxes[i]);
    if (ierr2 == PETSC_ERR_SUP) { /* no support, clean up and exit */
      PetscInt j;

      for (j = 0; j < i; j++) {
        ierr = PetscFree(subBoxes[j]);CHKERRQ(ierr);
      }
      ierr = PetscFree2(numSubBoxes,subBoxes);CHKERRQ(ierr);
      SETERRQ(PetscObjectComm((PetscObject)tagger),PETSC_ERR_SUP,"Sub tagger does not support box computation");
    } else {
      CHKERRQ(ierr2);
    }
    total += numSubBoxes[i];
  }
  ierr = PetscMalloc1(bs * total, &bxs);CHKERRQ(ierr);
  for (i = 0, nboxes = 0; i < nsubs; i++) { /* stupid O(N^2) check to remove subboxes */
    PetscInt j;

    for (j = 0; j < numSubBoxes[i]; j++) {
      PetscInt     k;
      VecTaggerBox *subBox = &subBoxes[i][j*bs];

      for (k = 0; k < nboxes; k++) {
        PetscBool   isSub = PETSC_FALSE;

        VecTaggerBox *prevBox = &bxs[bs * k];
        ierr = VecTaggerAndOrIsSubBox_Private(bs,prevBox,subBox,&isSub);CHKERRQ(ierr);
        if (isSub) break;
        ierr = VecTaggerAndOrIsSubBox_Private(bs,subBox,prevBox,&isSub);CHKERRQ(ierr);
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
    ierr = PetscFree(subBoxes[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree2(numSubBoxes,subBoxes);CHKERRQ(ierr);
  *numBoxes = nboxes;
  *boxes = bxs;
  PetscFunctionReturn(0);
}

static PetscErrorCode VecTaggerComputeIS_Or(VecTagger tagger, Vec vec, IS *is)
{
  PetscInt       nsubs, i;
  VecTagger      *subs;
  IS             unionIS;
  PetscErrorCode ierr, ierr2;

  PetscFunctionBegin;
  ierr2 = VecTaggerComputeIS_FromBoxes(tagger,vec,is);
  if (ierr2 != PETSC_ERR_SUP) {
    CHKERRQ(ierr2);
    PetscFunctionReturn(0);
  }
  ierr = VecTaggerOrGetSubs(tagger,&nsubs,&subs);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PetscObjectComm((PetscObject)vec),0,NULL,PETSC_OWN_POINTER,&unionIS);CHKERRQ(ierr);
  for (i = 0; i < nsubs; i++) {
    IS subIS, newUnionIS;

    ierr = VecTaggerComputeIS(subs[i],vec,&subIS);CHKERRQ(ierr);
    ierr = ISExpand(unionIS,subIS,&newUnionIS);CHKERRQ(ierr);
    ierr = ISSort(newUnionIS);CHKERRQ(ierr);
    ierr = ISDestroy(&unionIS);CHKERRQ(ierr);
    unionIS = newUnionIS;
    ierr = ISDestroy(&subIS);CHKERRQ(ierr);
  }
  *is = unionIS;
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode VecTaggerCreate_Or(VecTagger tagger)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecTaggerCreate_AndOr(tagger);CHKERRQ(ierr);
  tagger->ops->computeboxes = VecTaggerComputeBoxes_Or;
  tagger->ops->computeis        = VecTaggerComputeIS_Or;
  PetscFunctionReturn(0);
}
