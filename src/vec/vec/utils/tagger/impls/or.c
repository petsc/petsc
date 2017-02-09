
#include <petsc/private/vecimpl.h> /*I "petscvec.h" I*/
#include "../src/vec/vec/utils/tagger/impls/andor.h"

PetscErrorCode VecTaggerOrGetSubs(VecTagger tagger, PetscInt *nsubs, VecTagger **subs)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecTaggerGetSubs_AndOr(tagger,nsubs,subs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecTaggerOrSetSubs(VecTagger tagger, PetscInt nsubs, VecTagger *subs, PetscCopyMode mode)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecTaggerSetSubs_AndOr(tagger,nsubs,subs,mode);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode VecTaggerComputeIntervals_Or(VecTagger tagger,Vec vec,PetscInt *numIntervals,PetscScalar (**intervals)[2])
{
  PetscInt        i, bs, nsubs, *numSubIntervals, nints, total;
  PetscScalar     (**subIntervals)[2];
  VecTagger       *subs;
  PetscScalar     (*ints) [2];
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = VecTaggerGetBlockSize(tagger,&bs);CHKERRQ(ierr);
  ierr = VecTaggerOrGetSubs(tagger,&nsubs,&subs);CHKERRQ(ierr);
  ierr = PetscMalloc2(nsubs,&numSubIntervals,nsubs,&subIntervals);CHKERRQ(ierr);
  for (i = 0, total = 0; i < nsubs; i++) {
    PetscErrorCode ierr2;

    ierr2 = VecTaggerComputeIntervals(subs[i],vec,&numSubIntervals[i],&subIntervals[i]);
    if (ierr2 == PETSC_ERR_SUP) { /* no support, clean up and exit */
      PetscInt j;

      for (j = 0; j < i; j++) {
        ierr = PetscFree(subIntervals[j]);CHKERRQ(ierr);
      }
      ierr = PetscFree2(numSubIntervals,subIntervals);CHKERRQ(ierr);
      SETERRQ(PetscObjectComm((PetscObject)tagger),PETSC_ERR_SUP,"Sub tagger does not support interval computation");
    } else {
      CHKERRQ(ierr2);
    }
    total += numSubIntervals[i];
  }
  ierr = PetscMalloc1(bs * total, &ints);CHKERRQ(ierr);
  for (i = 0, nints = 0; i < nsubs; i++) { /* stupid O(N^2) check to remove subintervals */
    PetscInt j;

    for (j = 0; j < numSubIntervals[i]; j++) {
      PetscInt    k;
      PetscScalar (*subInt)[2] = &subIntervals[i][j*bs];

      for (k = 0; k < nints; k++) {
        PetscBool   isSub = PETSC_FALSE;

        PetscScalar (*prevInt)[2] = &ints[bs * k];
        ierr = VecTaggerAndOrIsSubinterval_Private(bs,prevInt,subInt,&isSub);CHKERRQ(ierr);
        if (isSub) break;
        ierr = VecTaggerAndOrIsSubinterval_Private(bs,subInt,prevInt,&isSub);CHKERRQ(ierr);
        if (isSub) {
          PetscInt l;

          for (l = 0; l < bs; l++) {
            prevInt[l][0] = subInt[l][0];
            prevInt[l][1] = subInt[l][1];
          }
          break;
        }
      }
      if (k < nints) continue;
      for (k = 0; k < bs; k++) {
        ints[nints * bs + k][0] = subInt[k][0];
        ints[nints * bs + k][1] = subInt[k][1];
      }
      nints++;
    }
    ierr = PetscFree(subIntervals[i]);
  }
  ierr = PetscFree2(numSubIntervals,subIntervals);CHKERRQ(ierr);
  *numIntervals = nints;
  *intervals = ints;
  PetscFunctionReturn(0);
}

static PetscErrorCode VecTaggerComputeIS_Or(VecTagger tagger, Vec vec, IS *is)
{
  PetscInt       nsubs, i;
  VecTagger      *subs;
  IS             unionIS;
  PetscErrorCode ierr, ierr2;

  PetscFunctionBegin;
  ierr2 = VecTaggerComputeIS_FromIntervals(tagger,vec,is);
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

PetscErrorCode VecTaggerCreate_Or(VecTagger tagger)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecTaggerCreate_AndOr(tagger);CHKERRQ(ierr);
  tagger->ops->computeintervals = VecTaggerComputeIntervals_Or;
  tagger->ops->computeis        = VecTaggerComputeIS_Or;
  PetscFunctionReturn(0);
}
