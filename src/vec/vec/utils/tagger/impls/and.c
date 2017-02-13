
#include <petsc/private/vecimpl.h> /*I "petscvec.h" I*/
#include "../src/vec/vec/utils/tagger/impls/andor.h"

PetscErrorCode VecTaggerAndGetSubs(VecTagger tagger, PetscInt *nsubs, VecTagger **subs)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecTaggerGetSubs_AndOr(tagger,nsubs,subs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecTaggerAndSetSubs(VecTagger tagger, PetscInt nsubs, VecTagger *subs, PetscCopyMode mode)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecTaggerSetSubs_AndOr(tagger,nsubs,subs,mode);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode VecTaggerComputeIntervals_And(VecTagger tagger,Vec vec,PetscInt *numIntervals,PetscScalar (**intervals)[2])
{
  PetscInt        i, bs, nsubs, *numSubIntervals, nints;
  PetscScalar     (**subIntervals)[2];
  VecTagger       *subs;
  PetscScalar     (*ints) [2] = NULL;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = VecTaggerGetBlockSize(tagger,&bs);CHKERRQ(ierr);
  ierr = VecTaggerOrGetSubs(tagger,&nsubs,&subs);CHKERRQ(ierr);
  ierr = PetscMalloc2(nsubs,&numSubIntervals,nsubs,&subIntervals);CHKERRQ(ierr);
  for (i = 0; i < nsubs; i++) {
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
  }
  for (i = 0, nints = 0; i < nsubs; i++) { /* stupid O(N^3) check to intersect intervals */
    PetscScalar (*isect)[2];
    PetscInt j, k, l, m, n;

    n = numSubIntervals[i];
    if (!n) {
      nints = 0;
      ierr = PetscFree(ints);CHKERRQ(ierr);
      break;
    }
    if (!i) {
      ierr = PetscMalloc1(n * bs, &ints);CHKERRQ(ierr);
      for (j = 0; j < numSubIntervals[i] * bs; j++) {
        ints[j][0] = subIntervals[i][j][0];
        ints[j][1] = subIntervals[i][j][1];
      }
      nints = n;
      ierr = PetscFree(subIntervals[i]);CHKERRQ(ierr);
      continue;
    }
    ierr = PetscMalloc1(n * nints * bs,&isect);CHKERRQ(ierr);
    for (j = 0, l = 0; j < n; j++) {
      PetscScalar (*subInt)[2] = &subIntervals[i][j*bs];

      for (k = 0; k < nints; k++) {
        PetscBool   isEmpty;
        PetscScalar (*prevInt)[2] = &ints[bs * k];
        ierr = VecTaggerAndOrIntersect_Private(bs,prevInt,subInt,&isect[l * bs],&isEmpty);CHKERRQ(ierr);
        if (isEmpty) continue;
        for (m = 0; m < l; m++) {
          PetscBool isSub = PETSC_FALSE;

          ierr = VecTaggerAndOrIsSubinterval_Private(bs,&isect[m*bs],&isect[l*bs],&isSub);CHKERRQ(ierr);
          if (isSub) break;
          ierr = VecTaggerAndOrIsSubinterval_Private(bs,&isect[l*bs],&isect[m*bs],&isSub);CHKERRQ(ierr);
          if (isSub) {
            PetscInt r;

            for (r = 0; r < bs; r++) {
              isect[m*bs + r][0] = isect[l * bs + r][0];
              isect[m*bs + r][1] = isect[l * bs + r][1];
            }
            break;
          }
        }
        if (m == l) l++;
      }
    }
    ierr = PetscFree(ints);CHKERRQ(ierr);
    ints = isect;
    nints = l;
    ierr = PetscFree(subIntervals[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree2(numSubIntervals,subIntervals);CHKERRQ(ierr);
  *numIntervals = nints;
  *intervals = ints;
  PetscFunctionReturn(0);
}

static PetscErrorCode VecTaggerComputeIS_And(VecTagger tagger, Vec vec, IS *is)
{
  PetscInt       nsubs, i;
  VecTagger      *subs;
  IS             isectIS;
  PetscErrorCode ierr, ierr2;

  PetscFunctionBegin;
  ierr2 = VecTaggerComputeIS_FromIntervals(tagger,vec,is);
  if (ierr2 != PETSC_ERR_SUP) {
    CHKERRQ(ierr2);
    PetscFunctionReturn(0);
  }
  ierr = VecTaggerOrGetSubs(tagger,&nsubs,&subs);CHKERRQ(ierr);
  if (!nsubs) {
    ierr = ISCreateGeneral(PetscObjectComm((PetscObject)vec),0,NULL,PETSC_OWN_POINTER,is);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  ierr = VecTaggerComputeIS(subs[0],vec,&isectIS);CHKERRQ(ierr);
  for (i = 1; i < nsubs; i++) {
    IS subIS, newIsectIS;

    ierr = VecTaggerComputeIS(subs[i],vec,&subIS);CHKERRQ(ierr);
    ierr = ISIntersect(isectIS,subIS,&newIsectIS);CHKERRQ(ierr);
    ierr = ISDestroy(&isectIS);CHKERRQ(ierr);
    ierr = ISDestroy(&subIS);CHKERRQ(ierr);
    isectIS = newIsectIS;
  }
  *is = isectIS;
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode VecTaggerCreate_And(VecTagger tagger)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecTaggerCreate_AndOr(tagger);CHKERRQ(ierr);
  tagger->ops->computeintervals = VecTaggerComputeIntervals_And;
  tagger->ops->computeis        = VecTaggerComputeIS_And;
  PetscFunctionReturn(0);
}
