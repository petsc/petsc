
#include <petsc/private/vecimpl.h> /*I "petscvec.h" I*/
#include "../src/vec/vec/utils/tagger/impls/simple.h"

PetscErrorCode VecTaggerDestroy_Simple(VecTagger tagger)
{
  VecTagger_Simple *smpl = (VecTagger_Simple *) tagger->data;

  PetscFunctionBegin;
  CHKERRQ(PetscFree (smpl->box));
  CHKERRQ(PetscFree (tagger->data));
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

  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetType((PetscObject)tagger,&name));
  CHKERRQ(VecTaggerGetBlockSize(tagger,&bs));
  nvals = 2 * bs;
  CHKERRQ(PetscMalloc1(nvals,&inBoxVals));
  CHKERRQ(PetscSNPrintf(headstring,BUFSIZ,"VecTagger %s options",name));
  CHKERRQ(PetscSNPrintf(funcstring,BUFSIZ,"VecTagger%sSetBox()",name));
  CHKERRQ(PetscOptionsHead(PetscOptionsObject,headstring));
  CHKERRQ(PetscOptionsScalarArray("-vec_tagger_box","lower and upper bounds of the box",funcstring,inBoxVals,&nvals,&set));
  CHKERRQ(PetscOptionsTail());
  if (set) {
    PetscCheckFalse(nvals != 2 *bs,PetscObjectComm((PetscObject)tagger),PETSC_ERR_ARG_INCOMP,"Expect array of %" PetscInt_FMT " values for -vec_tagger_box, got %" PetscInt_FMT,2 * bs,nvals);
    CHKERRQ(VecTaggerSetBox_Simple(tagger,(VecTaggerBox *)inBoxVals));
  }
  CHKERRQ(PetscFree(inBoxVals));
  PetscFunctionReturn(0);
}

PetscErrorCode VecTaggerSetUp_Simple(VecTagger tagger)
{
  VecTagger_Simple *smpl = (VecTagger_Simple *) tagger->data;

  PetscFunctionBegin;
  PetscCheckFalse(!smpl->box,PetscObjectComm((PetscObject)tagger),PETSC_ERR_ARG_WRONGSTATE,"Must set a box before calling setup.");
  PetscFunctionReturn(0);
}

PetscErrorCode VecTaggerView_Simple(VecTagger tagger, PetscViewer viewer)
{
  VecTagger_Simple *smpl = (VecTagger_Simple *) tagger->data;
  PetscBool        iascii;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii));
  if (iascii) {
    PetscInt bs, i;
    const char *name;

    CHKERRQ(PetscObjectGetType((PetscObject)tagger,&name));
    CHKERRQ(VecTaggerGetBlockSize(tagger,&bs));
    CHKERRQ(PetscViewerASCIIPrintf(viewer," %s box=[",name));
    for (i = 0; i < bs; i++) {
      if (i) {CHKERRQ(PetscViewerASCIIPrintf(viewer,"; "));}
#if !defined(PETSC_USE_COMPLEX)
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"%g,%g",(double)smpl->box[i].min,(double)smpl->box[i].max));
#else
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"%g+%gi,%g+%gi",(double)PetscRealPart(smpl->box[i].min),(double)PetscImaginaryPart(smpl->box[i].min),(double)PetscRealPart(smpl->box[i].max),(double)PetscImaginaryPart(smpl->box[i].max)));
#endif
    }
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"]\n"));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecTaggerSetBox_Simple(VecTagger tagger,VecTaggerBox *box)
{
  VecTagger_Simple *smpl = (VecTagger_Simple *) tagger->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tagger,VEC_TAGGER_CLASSID,1);
  PetscValidPointer(box,2);
  if (box != smpl->box) {
    PetscInt bs, i;

    CHKERRQ(VecTaggerGetBlockSize(tagger,&bs));
    CHKERRQ(PetscFree(smpl->box));
    CHKERRQ(PetscMalloc1(bs,&(smpl->box)));
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

  PetscFunctionBegin;
  tagger->ops->destroy          = VecTaggerDestroy_Simple;
  tagger->ops->setfromoptions   = VecTaggerSetFromOptions_Simple;
  tagger->ops->setup            = VecTaggerSetUp_Simple;
  tagger->ops->view             = VecTaggerView_Simple;
  tagger->ops->computeis        = VecTaggerComputeIS_FromBoxes;
  CHKERRQ(PetscNewLog(tagger,&smpl));
  tagger->data = smpl;
  PetscFunctionReturn(0);
}
