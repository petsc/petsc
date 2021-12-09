
#include <petsc/private/vecimpl.h> /*I "petscvec.h" I*/
#include "../src/vec/vec/utils/tagger/impls/simple.h"

PetscErrorCode VecTaggerDestroy_Simple(VecTagger tagger)
{
  VecTagger_Simple *smpl = (VecTagger_Simple *) tagger->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree (smpl->box);CHKERRQ(ierr);
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
  PetscScalar    *inBoxVals;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetType((PetscObject)tagger,&name);CHKERRQ(ierr);
  ierr = VecTaggerGetBlockSize(tagger,&bs);CHKERRQ(ierr);
  nvals = 2 * bs;
  ierr = PetscMalloc1(nvals,&inBoxVals);CHKERRQ(ierr);
  ierr = PetscSNPrintf(headstring,BUFSIZ,"VecTagger %s options",name);CHKERRQ(ierr);
  ierr = PetscSNPrintf(funcstring,BUFSIZ,"VecTagger%sSetBox()",name);CHKERRQ(ierr);
  ierr = PetscOptionsHead(PetscOptionsObject,headstring);CHKERRQ(ierr);
  ierr = PetscOptionsScalarArray("-vec_tagger_box","lower and upper bounds of the box",funcstring,inBoxVals,&nvals,&set);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  if (set) {
    if (nvals != 2 *bs) SETERRQ2(PetscObjectComm((PetscObject)tagger),PETSC_ERR_ARG_INCOMP,"Expect array of %" PetscInt_FMT " values for -vec_tagger_box, got %" PetscInt_FMT "",2 * bs,nvals);
    ierr = VecTaggerSetBox_Simple(tagger,(VecTaggerBox *)inBoxVals);CHKERRQ(ierr);
  }
  ierr = PetscFree(inBoxVals);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecTaggerSetUp_Simple(VecTagger tagger)
{
  VecTagger_Simple *smpl = (VecTagger_Simple *) tagger->data;

  PetscFunctionBegin;
  if (!smpl->box) SETERRQ(PetscObjectComm((PetscObject)tagger),PETSC_ERR_ARG_WRONGSTATE,"Must set a box before calling setup.");
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
    ierr = PetscViewerASCIIPrintf(viewer," %s box=[",name);CHKERRQ(ierr);
    for (i = 0; i < bs; i++) {
      if (i) {ierr = PetscViewerASCIIPrintf(viewer,"; ");CHKERRQ(ierr);}
#if !defined(PETSC_USE_COMPLEX)
      ierr = PetscViewerASCIIPrintf(viewer,"%g,%g",(double)smpl->box[i].min,(double)smpl->box[i].max);CHKERRQ(ierr);
#else
      ierr = PetscViewerASCIIPrintf(viewer,"%g+%gi,%g+%gi",(double)PetscRealPart(smpl->box[i].min),(double)PetscImaginaryPart(smpl->box[i].min),(double)PetscRealPart(smpl->box[i].max),(double)PetscImaginaryPart(smpl->box[i].max));CHKERRQ(ierr);
#endif
    }
    ierr = PetscViewerASCIIPrintf(viewer,"]\n");CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecTaggerSetBox_Simple(VecTagger tagger,VecTaggerBox *box)
{
  VecTagger_Simple *smpl = (VecTagger_Simple *) tagger->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tagger,VEC_TAGGER_CLASSID,1);
  PetscValidPointer(box,2);
  if (box != smpl->box) {
    PetscInt bs, i;

    ierr = VecTaggerGetBlockSize(tagger,&bs);CHKERRQ(ierr);
    ierr = PetscFree(smpl->box);CHKERRQ(ierr);
    ierr = PetscMalloc1(bs,&(smpl->box));CHKERRQ(ierr);
    for (i = 0; i < bs; i++) smpl->box[i] = box[i];
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecTaggerGetBox_Simple(VecTagger tagger,const VecTaggerBox **box)
{
  VecTagger_Simple *smpl = (VecTagger_Simple *) tagger->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tagger,VEC_TAGGER_CLASSID,1);
  PetscValidPointer(box,2);
  *box = smpl->box;
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
  tagger->ops->computeis        = VecTaggerComputeIS_FromBoxes;
  ierr = PetscNewLog(tagger,&smpl);CHKERRQ(ierr);
  tagger->data = smpl;
  PetscFunctionReturn(0);
}
