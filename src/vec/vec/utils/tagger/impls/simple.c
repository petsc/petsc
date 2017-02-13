
#include <petsc/private/vecimpl.h> /*I "petscvec.h" I*/
#include "../src/vec/vec/utils/tagger/impls/simple.h"

PetscErrorCode VecTaggerDestroy_Simple(VecTagger tagger)
{
  VecTagger_Simple *smpl = (VecTagger_Simple *) tagger->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree (smpl->interval);CHKERRQ(ierr);
  ierr = PetscFree (tagger->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecTaggerSetFromOptions_Simple(PetscOptionItems *PetscOptionsObject,VecTagger tagger)
{
  PetscInt       nvals, bs;
  char           headstring[BUFSIZ];
  char           funcstring[BUFSIZ];
  const char     *name;
  PetscBool      set;
  PetscScalar    (*tmpInterval)[2];
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetType((PetscObject)tagger,&name);CHKERRQ(ierr);
  ierr = VecTaggerGetBlockSize(tagger,&bs);CHKERRQ(ierr);
  nvals = 2 * bs;
  ierr = PetscMalloc1(bs,&tmpInterval);CHKERRQ(ierr);
  ierr = PetscSNPrintf(headstring,BUFSIZ,"VecTagger %s options",name);CHKERRQ(ierr);
  ierr = PetscSNPrintf(funcstring,BUFSIZ,"VecTagger%sSetInterval()",name);CHKERRQ(ierr);
  ierr = PetscOptionsHead(PetscOptionsObject,headstring);CHKERRQ(ierr);
  ierr = PetscOptionsScalarArray("-vec_tagger_interval","lower and upper bounds of the interval",funcstring,(PetscScalar *) &tmpInterval[0][0],&nvals,&set);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  if (set) {
    if (nvals != 2 *bs) SETERRQ2(PetscObjectComm((PetscObject)tagger),PETSC_ERR_ARG_INCOMP,"Expect array of %D values for -vec_tagger_interval, got %D",2 * bs,nvals);
    ierr = VecTaggerSetInterval_Simple(tagger,tmpInterval);CHKERRQ(ierr);
  }
  ierr = PetscFree(tmpInterval);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecTaggerSetUp_Simple(VecTagger tagger)
{
  VecTagger_Simple *smpl = (VecTagger_Simple *) tagger->data;

  PetscFunctionBegin;
  if (!smpl->interval) SETERRQ(PetscObjectComm((PetscObject)tagger),PETSC_ERR_ARG_WRONGSTATE,"Must set an interval before calling setup.");
  PetscFunctionReturn(0);
}

PetscErrorCode VecTaggerView_Simple(VecTagger tagger, PetscViewer viewer)
{
  VecTagger_Simple *smpl = (VecTagger_Simple *) tagger->data;
  PetscBool        iascii;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    PetscInt bs, i;
    const char *name;

    ierr = PetscObjectGetType((PetscObject)tagger,&name);CHKERRQ(ierr);
    ierr = VecTaggerGetBlockSize(tagger,&bs);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer," %s interval=[",name);CHKERRQ(ierr);
    for (i = 0; i < bs; i++) {
      if (i) {ierr = PetscViewerASCIIPrintf(viewer,"; ");CHKERRQ(ierr);}
#if !defined(PETSC_USE_COMPLEX)
      ierr = PetscViewerASCIIPrintf(viewer,"%g,%g",smpl->interval[i][0],smpl->interval[i][1]);CHKERRQ(ierr);
#else
      ierr = PetscViewerASCIIPrintf(viewer,"%g+%gi,%g+%gi",PetscRealPart(smpl->interval[i][0]),PetscImaginaryPart(smpl->interval[i][0]),PetscRealPart(smpl->interval[i][1]),PetscImaginaryPart(smpl->interval[i][1]));CHKERRQ(ierr);
#endif
    }
    ierr = PetscViewerASCIIPrintf(viewer,"]\n");CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecTaggerSetInterval_Simple(VecTagger tagger,PetscScalar (*interval)[2])
{
  VecTagger_Simple *smpl = (VecTagger_Simple *) tagger->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tagger,VEC_TAGGER_CLASSID,1);
  PetscValidPointer(interval,2);
  if (interval != smpl->interval) {
    PetscInt bs, i;

    ierr = VecTaggerGetBlockSize(tagger,&bs);CHKERRQ(ierr);
    ierr = PetscFree(smpl->interval);CHKERRQ(ierr);
    ierr = PetscMalloc1(bs,&(smpl->interval));CHKERRQ(ierr);
    for (i = 0; i < bs; i++) {
      smpl->interval[i][0] = interval[i][0];
      smpl->interval[i][1] = interval[i][1];
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecTaggerGetInterval_Simple(VecTagger tagger,const PetscScalar (**interval)[2])
{
  VecTagger_Simple *smpl = (VecTagger_Simple *) tagger->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tagger,VEC_TAGGER_CLASSID,1);
  PetscValidPointer(interval,2);
  *interval = (const PetscScalar (*)[2]) smpl->interval;
  PetscFunctionReturn(0);
}

PetscErrorCode VecTaggerCreate_Simple(VecTagger tagger)
{
  VecTagger_Simple *smpl;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  tagger->ops->destroy          = VecTaggerDestroy_Simple;
  tagger->ops->setfromoptions   = VecTaggerSetFromOptions_Simple;
  tagger->ops->setup            = VecTaggerSetUp_Simple;
  tagger->ops->view             = VecTaggerView_Simple;
  tagger->ops->computeis        = VecTaggerComputeIS_FromIntervals;
  ierr = PetscNewLog(tagger,&smpl);CHKERRQ(ierr);
  tagger->data = smpl;
  PetscFunctionReturn(0);
}
