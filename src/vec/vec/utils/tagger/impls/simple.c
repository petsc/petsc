
#include <petsc/private/vecimpl.h> /*I "petscvec.h" I*/
#include "../src/vec/vec/utils/tagger/impls/simple.h"

PetscErrorCode VecTaggerDestroy_Simple(VecTagger tagger)
{
  VecTagger_Simple *smpl = (VecTagger_Simple *) tagger->data;

  PetscFunctionBegin;
  PetscCall(PetscFree (smpl->box));
  PetscCall(PetscFree (tagger->data));
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
  PetscCall(PetscObjectGetType((PetscObject)tagger,&name));
  PetscCall(VecTaggerGetBlockSize(tagger,&bs));
  nvals = 2 * bs;
  PetscCall(PetscMalloc1(nvals,&inBoxVals));
  PetscCall(PetscSNPrintf(headstring,BUFSIZ,"VecTagger %s options",name));
  PetscCall(PetscSNPrintf(funcstring,BUFSIZ,"VecTagger%sSetBox()",name));
  PetscOptionsHeadBegin(PetscOptionsObject,headstring);
  PetscCall(PetscOptionsScalarArray("-vec_tagger_box","lower and upper bounds of the box",funcstring,inBoxVals,&nvals,&set));
  PetscOptionsHeadEnd();
  if (set) {
    PetscCheckFalse(nvals != 2 *bs,PetscObjectComm((PetscObject)tagger),PETSC_ERR_ARG_INCOMP,"Expect array of %" PetscInt_FMT " values for -vec_tagger_box, got %" PetscInt_FMT,2 * bs,nvals);
    PetscCall(VecTaggerSetBox_Simple(tagger,(VecTaggerBox *)inBoxVals));
  }
  PetscCall(PetscFree(inBoxVals));
  PetscFunctionReturn(0);
}

PetscErrorCode VecTaggerSetUp_Simple(VecTagger tagger)
{
  VecTagger_Simple *smpl = (VecTagger_Simple *) tagger->data;

  PetscFunctionBegin;
  PetscCheck(smpl->box,PetscObjectComm((PetscObject)tagger),PETSC_ERR_ARG_WRONGSTATE,"Must set a box before calling setup.");
  PetscFunctionReturn(0);
}

PetscErrorCode VecTaggerView_Simple(VecTagger tagger, PetscViewer viewer)
{
  VecTagger_Simple *smpl = (VecTagger_Simple *) tagger->data;
  PetscBool        iascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii));
  if (iascii) {
    PetscInt bs, i;
    const char *name;

    PetscCall(PetscObjectGetType((PetscObject)tagger,&name));
    PetscCall(VecTaggerGetBlockSize(tagger,&bs));
    PetscCall(PetscViewerASCIIPrintf(viewer," %s box=[",name));
    for (i = 0; i < bs; i++) {
      if (i) {PetscCall(PetscViewerASCIIPrintf(viewer,"; "));}
#if !defined(PETSC_USE_COMPLEX)
      PetscCall(PetscViewerASCIIPrintf(viewer,"%g,%g",(double)smpl->box[i].min,(double)smpl->box[i].max));
#else
      PetscCall(PetscViewerASCIIPrintf(viewer,"%g+%gi,%g+%gi",(double)PetscRealPart(smpl->box[i].min),(double)PetscImaginaryPart(smpl->box[i].min),(double)PetscRealPart(smpl->box[i].max),(double)PetscImaginaryPart(smpl->box[i].max)));
#endif
    }
    PetscCall(PetscViewerASCIIPrintf(viewer,"]\n"));
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

    PetscCall(VecTaggerGetBlockSize(tagger,&bs));
    PetscCall(PetscFree(smpl->box));
    PetscCall(PetscMalloc1(bs,&(smpl->box)));
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
  PetscCall(PetscNewLog(tagger,&smpl));
  tagger->data = smpl;
  PetscFunctionReturn(0);
}
