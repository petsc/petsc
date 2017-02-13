#include <petsc/private/vecimpl.h> /*I "petscvec.h" I*/
#include "../src/vec/vec/utils/tagger/impls/andor.h"

static PetscErrorCode VecTaggerDestroy_AndOr(VecTagger tagger)
{
  VecTagger_AndOr *andOr = (VecTagger_AndOr *) tagger->data;
  PetscInt        i;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  for (i = 0; i < andOr->nsubs; i++) {
    ierr = VecTaggerDestroy(&andOr->subs[i]);CHKERRQ(ierr);
  }
  if (andOr->mode == PETSC_OWN_POINTER) {
    ierr = PetscFree(andOr->subs);CHKERRQ(ierr);
  }
  ierr = PetscFree(tagger->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecTaggerGetSubs_AndOr(VecTagger tagger, PetscInt *nsubs, VecTagger **subs)
{
  VecTagger_AndOr *andOr = (VecTagger_AndOr *) tagger->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tagger,VEC_TAGGER_CLASSID,1);
  if (nsubs) {
    PetscValidIntPointer(nsubs,2);
    *nsubs = andOr->nsubs;
  }
  if (subs) {
    PetscValidPointer(subs,3);
    *subs = andOr->subs;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecTaggerSetSubs_AndOr(VecTagger tagger, PetscInt nsubs, VecTagger *subs, PetscCopyMode mode)
{
  PetscInt        i;
  VecTagger_AndOr *andOr = (VecTagger_AndOr *) tagger->data;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tagger,VEC_TAGGER_CLASSID,1);
  if (subs) PetscValidPointer(subs,3);
  if (nsubs == andOr->nsubs && subs == andOr->subs && mode != PETSC_COPY_VALUES) PetscFunctionReturn(0);
  if (subs) {
    for (i = 0; i < nsubs; i++) {
      ierr = PetscObjectReference((PetscObject)subs[i]);CHKERRQ(ierr);
    }
  }
  for (i = 0; i < andOr->nsubs; i++) {
    ierr = VecTaggerDestroy(&(andOr->subs[i]));CHKERRQ(ierr);
  }
  if (andOr->mode == PETSC_OWN_POINTER && andOr->subs != subs) {
    ierr = PetscFree(andOr->subs);CHKERRQ(ierr);
  }
  andOr->nsubs = nsubs;
  if (subs) {
    if (mode == PETSC_COPY_VALUES) {
      andOr->mode = PETSC_OWN_POINTER;
      ierr = PetscMalloc1(nsubs,&(andOr->subs));CHKERRQ(ierr);
      for (i = 0; i < nsubs; i++) {
        andOr->subs[i] = subs[i];
      }
    } else {
      andOr->subs = subs;
      andOr->mode = mode;
      for (i = 0; i < nsubs; i++) {
        ierr = PetscObjectDereference((PetscObject)subs[i]);CHKERRQ(ierr);
      }
    }
  } else {
    MPI_Comm   comm = PetscObjectComm((PetscObject)tagger);
    PetscInt   bs;
    const char *prefix;
    char       tprefix[128];

    ierr = VecTaggerGetBlockSize(tagger,&bs);CHKERRQ(ierr);
    ierr = PetscObjectGetOptionsPrefix((PetscObject)tagger,&prefix);CHKERRQ(ierr);
    andOr->mode = PETSC_OWN_POINTER;
    ierr = PetscMalloc1(nsubs,&(andOr->subs));CHKERRQ(ierr);
    for (i = 0; i < nsubs; i++) {
      VecTagger sub;

      ierr = PetscSNPrintf(tprefix,128,"sub_%D_",i);CHKERRQ(ierr);
      ierr = VecTaggerCreate(comm,&sub);CHKERRQ(ierr);
      ierr = VecTaggerSetBlockSize(sub,bs);CHKERRQ(ierr);
      ierr = PetscObjectSetOptionsPrefix((PetscObject)sub,prefix);CHKERRQ(ierr);
      ierr = PetscObjectAppendOptionsPrefix((PetscObject)sub,tprefix);CHKERRQ(ierr);
      andOr->subs[i] = sub;
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode VecTaggerSetFromOptions_AndOr(PetscOptionItems *PetscOptionsObject,VecTagger tagger)
{
  PetscInt       i, nsubs, nsubsOrig;
  const char     *name;
  char           headstring[BUFSIZ];
  char           funcstring[BUFSIZ];
  char           descstring[BUFSIZ];
  VecTagger      *subs;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetType((PetscObject)tagger,&name);CHKERRQ(ierr);
  ierr = VecTaggerGetSubs_AndOr(tagger,&nsubs,NULL);CHKERRQ(ierr);
  nsubsOrig = nsubs;
  ierr = PetscSNPrintf(headstring,BUFSIZ,"VecTagger %s options",name);CHKERRQ(ierr);
  ierr = PetscSNPrintf(funcstring,BUFSIZ,"VecTagger%sSetSubs()",name);CHKERRQ(ierr);
  ierr = PetscSNPrintf(descstring,BUFSIZ,"number of sub tags in %s tag",name);CHKERRQ(ierr);
  ierr = PetscOptionsHead(PetscOptionsObject,headstring);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-vec_tagger_num_subs",descstring,funcstring,nsubs,&nsubs,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  if (nsubs != nsubsOrig) {
    ierr = VecTaggerSetSubs_AndOr(tagger,nsubs,NULL,PETSC_OWN_POINTER);CHKERRQ(ierr);
    ierr = VecTaggerGetSubs_AndOr(tagger,NULL,&subs);CHKERRQ(ierr);
    for (i = 0; i < nsubs; i++) {
      ierr = VecTaggerSetFromOptions(subs[i]);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode VecTaggerSetUp_AndOr (VecTagger tagger)
{
  PetscInt        nsubs, i;
  VecTagger       *subs;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = VecTaggerGetSubs_AndOr(tagger,&nsubs,&subs);CHKERRQ(ierr);
  if (!nsubs) SETERRQ(PetscObjectComm((PetscObject)tagger),PETSC_ERR_ARG_WRONGSTATE,"Must set sub taggers before calling setup.");
  for (i = 0; i < nsubs; i++) {
    ierr = VecTaggerSetUp(subs[i]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode VecTaggerView_AndOr(VecTagger tagger, PetscViewer viewer)
{
  PetscBool       iascii;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    PetscInt i, nsubs;
    VecTagger *subs;
    const char *name;

    ierr = VecTaggerGetSubs_AndOr(tagger,&nsubs,&subs);CHKERRQ(ierr);
    ierr = PetscObjectGetType((PetscObject)tagger,&name);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer," %s of %D subtags:\n",name,nsubs);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    for (i = 0; i < nsubs; i++) {
      ierr = VecTaggerView(subs[i],viewer);CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecTaggerCreate_AndOr(VecTagger tagger)
{
  VecTagger_AndOr    *andOr;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  tagger->ops->destroy          = VecTaggerDestroy_AndOr;
  tagger->ops->setfromoptions   = VecTaggerSetFromOptions_AndOr;
  tagger->ops->setup            = VecTaggerSetUp_AndOr;
  tagger->ops->view             = VecTaggerView_AndOr;
  tagger->ops->computeis        = VecTaggerComputeIS_FromIntervals;
  ierr = PetscNewLog(tagger,&andOr);CHKERRQ(ierr);
  tagger->data = andOr;
  PetscFunctionReturn(0);
}

PetscErrorCode VecTaggerAndOrIsSubinterval_Private(PetscInt bs, PetscScalar (*superInt)[2], PetscScalar (*subInt)[2],PetscBool *isSub)
{
  PetscInt       i;

  PetscFunctionBegin;
  *isSub = PETSC_TRUE;
  for (i = 0; i < bs; i++) {
#if !defined(PETSC_USE_COMPLEX)
    if (superInt[i][0] > subInt[i][0] || superInt[i][1] < subInt[i][1]) {
      *isSub = PETSC_FALSE;
      break;
    }
#else
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Complex support not implemented yet.");
#endif
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecTaggerAndOrIntersect_Private(PetscInt bs, PetscScalar (*a)[2], PetscScalar (*b)[2],PetscScalar(*c)[2],PetscBool *empty)
{
  PetscInt       i;

  PetscFunctionBegin;
  *empty = PETSC_FALSE;
  for (i = 0; i < bs; i++) {
#if !defined(PETSC_USE_COMPLEX)
    c[i][0] = PetscMax(a[i][0],b[i][0]);
    c[i][1] = PetscMin(a[i][1],b[i][1]);
    if (c[i][1] < c[i][0]) {
      *empty = PETSC_TRUE;
      break;
    }
#else
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Complex support not implemented yet.");
#endif
  }
  PetscFunctionReturn(0);
}
