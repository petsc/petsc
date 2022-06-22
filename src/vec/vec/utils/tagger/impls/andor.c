#include <petsc/private/vecimpl.h> /*I "petscvec.h" I*/
#include "../src/vec/vec/utils/tagger/impls/andor.h"

static PetscErrorCode VecTaggerDestroy_AndOr(VecTagger tagger)
{
  VecTagger_AndOr *andOr = (VecTagger_AndOr *) tagger->data;
  PetscInt        i;

  PetscFunctionBegin;
  for (i = 0; i < andOr->nsubs; i++) {
    PetscCall(VecTaggerDestroy(&andOr->subs[i]));
  }
  if (andOr->mode == PETSC_OWN_POINTER) {
    PetscCall(PetscFree(andOr->subs));
  }
  PetscCall(PetscFree(tagger->data));
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

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tagger,VEC_TAGGER_CLASSID,1);
  if (subs) PetscValidPointer(subs,3);
  if (nsubs == andOr->nsubs && subs == andOr->subs && mode != PETSC_COPY_VALUES) PetscFunctionReturn(0);
  if (subs) {
    for (i = 0; i < nsubs; i++) {
      PetscCall(PetscObjectReference((PetscObject)subs[i]));
    }
  }
  for (i = 0; i < andOr->nsubs; i++) {
    PetscCall(VecTaggerDestroy(&(andOr->subs[i])));
  }
  if (andOr->mode == PETSC_OWN_POINTER && andOr->subs != subs) {
    PetscCall(PetscFree(andOr->subs));
  }
  andOr->nsubs = nsubs;
  if (subs) {
    if (mode == PETSC_COPY_VALUES) {
      andOr->mode = PETSC_OWN_POINTER;
      PetscCall(PetscMalloc1(nsubs,&(andOr->subs)));
      for (i = 0; i < nsubs; i++) {
        andOr->subs[i] = subs[i];
      }
    } else {
      andOr->subs = subs;
      andOr->mode = mode;
      for (i = 0; i < nsubs; i++) {
        PetscCall(PetscObjectDereference((PetscObject)subs[i]));
      }
    }
  } else {
    MPI_Comm   comm = PetscObjectComm((PetscObject)tagger);
    PetscInt   bs;
    const char *prefix;
    char       tprefix[128];

    PetscCall(VecTaggerGetBlockSize(tagger,&bs));
    PetscCall(PetscObjectGetOptionsPrefix((PetscObject)tagger,&prefix));
    andOr->mode = PETSC_OWN_POINTER;
    PetscCall(PetscMalloc1(nsubs,&(andOr->subs)));
    for (i = 0; i < nsubs; i++) {
      VecTagger sub;

      PetscCall(PetscSNPrintf(tprefix,128,"sub_%" PetscInt_FMT "_",i));
      PetscCall(VecTaggerCreate(comm,&sub));
      PetscCall(VecTaggerSetBlockSize(sub,bs));
      PetscCall(PetscObjectSetOptionsPrefix((PetscObject)sub,prefix));
      PetscCall(PetscObjectAppendOptionsPrefix((PetscObject)sub,tprefix));
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

  PetscFunctionBegin;
  PetscCall(PetscObjectGetType((PetscObject)tagger,&name));
  PetscCall(VecTaggerGetSubs_AndOr(tagger,&nsubs,NULL));
  nsubsOrig = nsubs;
  PetscCall(PetscSNPrintf(headstring,sizeof(headstring),"VecTagger %s options",name));
  PetscCall(PetscSNPrintf(funcstring,sizeof(funcstring),"VecTagger%sSetSubs()",name));
  PetscCall(PetscSNPrintf(descstring,sizeof(descstring),"number of sub tags in %s tag",name));
  PetscOptionsHeadBegin(PetscOptionsObject,headstring);
  PetscCall(PetscOptionsInt("-vec_tagger_num_subs",descstring,funcstring,nsubs,&nsubs,NULL));
  PetscOptionsHeadEnd();
  if (nsubs != nsubsOrig) {
    PetscCall(VecTaggerSetSubs_AndOr(tagger,nsubs,NULL,PETSC_OWN_POINTER));
    PetscCall(VecTaggerGetSubs_AndOr(tagger,NULL,&subs));
    for (i = 0; i < nsubs; i++) {
      PetscCall(VecTaggerSetFromOptions(subs[i]));
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode VecTaggerSetUp_AndOr (VecTagger tagger)
{
  PetscInt        nsubs, i;
  VecTagger       *subs;

  PetscFunctionBegin;
  PetscCall(VecTaggerGetSubs_AndOr(tagger,&nsubs,&subs));
  PetscCheck(nsubs,PetscObjectComm((PetscObject)tagger),PETSC_ERR_ARG_WRONGSTATE,"Must set sub taggers before calling setup.");
  for (i = 0; i < nsubs; i++) {
    PetscCall(VecTaggerSetUp(subs[i]));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode VecTaggerView_AndOr(VecTagger tagger, PetscViewer viewer)
{
  PetscBool       iascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii));
  if (iascii) {
    PetscInt i, nsubs;
    VecTagger *subs;
    const char *name;

    PetscCall(VecTaggerGetSubs_AndOr(tagger,&nsubs,&subs));
    PetscCall(PetscObjectGetType((PetscObject)tagger,&name));
    PetscCall(PetscViewerASCIIPrintf(viewer," %s of %" PetscInt_FMT " subtags:\n",name,nsubs));
    PetscCall(PetscViewerASCIIPushTab(viewer));
    for (i = 0; i < nsubs; i++) {
      PetscCall(VecTaggerView(subs[i],viewer));
    }
    PetscCall(PetscViewerASCIIPopTab(viewer));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecTaggerCreate_AndOr(VecTagger tagger)
{
  VecTagger_AndOr    *andOr;

  PetscFunctionBegin;
  tagger->ops->destroy          = VecTaggerDestroy_AndOr;
  tagger->ops->setfromoptions   = VecTaggerSetFromOptions_AndOr;
  tagger->ops->setup            = VecTaggerSetUp_AndOr;
  tagger->ops->view             = VecTaggerView_AndOr;
  tagger->ops->computeis        = VecTaggerComputeIS_FromBoxes;
  PetscCall(PetscNewLog(tagger,&andOr));
  tagger->data = andOr;
  PetscFunctionReturn(0);
}

PetscErrorCode VecTaggerAndOrIsSubBox_Private(PetscInt bs, const VecTaggerBox *superBox, const VecTaggerBox *subBox,PetscBool *isSub)
{
  PetscInt       i;

  PetscFunctionBegin;
  *isSub = PETSC_TRUE;
  for (i = 0; i < bs; i++) {
#if !defined(PETSC_USE_COMPLEX)
    if (superBox[i].min > subBox[i].min || superBox[i].max < subBox[i].max) {
      *isSub = PETSC_FALSE;
      break;
    }
#else
    if (PetscRealPart(superBox[i].min) > PetscRealPart(subBox[i].min) || PetscImaginaryPart(superBox[i].min) > PetscImaginaryPart(subBox[i].min) ||
        PetscRealPart(superBox[i].max) < PetscRealPart(subBox[i].max) || PetscImaginaryPart(superBox[i].max) < PetscImaginaryPart(subBox[i].max)) {
      *isSub = PETSC_FALSE;
      break;
    }
#endif
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecTaggerAndOrIntersect_Private(PetscInt bs, const VecTaggerBox *a, const VecTaggerBox *b,VecTaggerBox *c,PetscBool *empty)
{
  PetscInt       i;

  PetscFunctionBegin;
  *empty = PETSC_FALSE;
  for (i = 0; i < bs; i++) {
#if !defined(PETSC_USE_COMPLEX)
    c[i].min = PetscMax(a[i].min,b[i].min);
    c[i].max = PetscMin(a[i].max,b[i].max);
    if (c[i].max < c[i].min) {
      *empty = PETSC_TRUE;
      break;
    }
#else
    {
      PetscReal maxMinReal = PetscMax(PetscRealPart(a[i].min),PetscRealPart(b[i].min));
      PetscReal maxMinImag = PetscMax(PetscImaginaryPart(a[i].min),PetscImaginaryPart(b[i].min));
      PetscReal minMaxReal = PetscMin(PetscRealPart(a[i].max),PetscRealPart(b[i].max));
      PetscReal minMaxImag = PetscMin(PetscImaginaryPart(a[i].max),PetscImaginaryPart(b[i].max));

      c[i].min = PetscCMPLX(maxMinReal,maxMinImag);
      c[i].max = PetscCMPLX(minMaxReal,minMaxImag);
      if ((PetscRealPart(c[i].max - c[i].min) < 0.) || (PetscImaginaryPart(c[i].max - c[i].min) < 0.)) {
        *empty = PETSC_TRUE;
        break;
      }
    }
#endif
  }
  PetscFunctionReturn(0);
}
