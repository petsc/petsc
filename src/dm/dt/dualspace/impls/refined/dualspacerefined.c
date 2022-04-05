#include <petsc/private/petscfeimpl.h> /*I "petscfe.h" I*/
#include <petscdmplex.h>

typedef struct {
  PetscInt        dummy;
} PetscDualSpace_Refined;

/*@
   PetscDualSpaceRefinedSetCellSpaces - Set the dual spaces for the closures of each of the cells
   in the multicell DM of a PetscDualSpace

   Collective on PetscDualSpace

   Input Parameters:
+  sp - a PetscDualSpace
-  cellSpaces - one PetscDualSpace for each of the cells.  The reference count of each cell space will be incremented,
                so the user is still responsible for these spaces afterwards

   Level: intermediate

.seealso: PetscFERefine()
@*/
PetscErrorCode PetscDualSpaceRefinedSetCellSpaces(PetscDualSpace sp, const PetscDualSpace cellSpaces[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 1);
  PetscValidPointer(cellSpaces,2);
  PetscCheck(!sp->setupcalled,PetscObjectComm((PetscObject)sp), PETSC_ERR_ARG_WRONGSTATE, "Cannot change cell spaces after setup is called");
  PetscTryMethod(sp, "PetscDualSpaceRefinedSetCellSpaces_C", (PetscDualSpace,const PetscDualSpace []),(sp,cellSpaces));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDualSpaceRefinedSetCellSpaces_Refined(PetscDualSpace sp, const PetscDualSpace cellSpaces[])
{
  DM dm;
  PetscInt pStart, pEnd;
  PetscInt cStart, cEnd, c;

  PetscFunctionBegin;
  dm = sp->dm;
  PetscCheck(dm,PetscObjectComm((PetscObject) sp), PETSC_ERR_ARG_WRONGSTATE, "PetscDualSpace must have a DM (PetscDualSpaceSetDM()) before calling PetscDualSpaceRefinedSetCellSpaces");
  PetscCall(DMPlexGetChart(dm, &pStart, &pEnd));
  if (!sp->pointSpaces) {

    PetscCall(PetscCalloc1(pEnd-pStart,&(sp->pointSpaces)));
  }
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  for (c = 0; c < cEnd - cStart; c++) {
    PetscCall(PetscObjectReference((PetscObject)cellSpaces[c]));
    PetscCall(PetscDualSpaceDestroy(&(sp->pointSpaces[c + cStart - pStart])));
    sp->pointSpaces[c+cStart-pStart] = cellSpaces[c];
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDualSpaceDestroy_Refined(PetscDualSpace sp)
{
  PetscDualSpace_Refined *ref = (PetscDualSpace_Refined *) sp->data;

  PetscFunctionBegin;
  PetscCall(PetscObjectComposeFunction((PetscObject) sp, "PetscDualSpaceRefinedSetCellSpaces_C", NULL));
  PetscCall(PetscFree(ref));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDualSpaceSetUp_Refined(PetscDualSpace sp)
{
  PetscInt pStart, pEnd, depth;
  PetscInt cStart, cEnd, c, spdim;
  PetscInt h;
  DM dm;
  PetscSection   section;

  PetscFunctionBegin;
  PetscCall(PetscDualSpaceGetDM(sp, &dm));
  PetscCall(DMPlexGetDepth(dm, &depth));
  PetscCall(DMPlexGetChart(dm, &pStart, &pEnd));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  for (c = cStart; c < cEnd; c++) {
    if (sp->pointSpaces[c-pStart]) {
      PetscInt ccStart, ccEnd;
      PetscCheck(sp->pointSpaces[c-pStart]->k == sp->k,PetscObjectComm((PetscObject)sp), PETSC_ERR_ARG_INCOMP, "All cell spaces must have the same form degree as the refined dual space");
      PetscCheck(sp->pointSpaces[c-pStart]->Nc == sp->Nc,PetscObjectComm((PetscObject)sp), PETSC_ERR_ARG_INCOMP, "All cell spaces must have the same number of components as the refined dual space");
      PetscCall(DMPlexGetHeightStratum(sp->pointSpaces[c-pStart]->dm, 0, &ccStart, &ccEnd));
      PetscCheckFalse(ccEnd - ccStart != 1,PetscObjectComm((PetscObject)sp), PETSC_ERR_ARG_INCOMP, "All cell spaces must have a single cell themselves");
    }
  }
  for (c = cStart; c < cEnd; c++) {
    if (sp->pointSpaces[c-pStart]) {
      PetscBool cUniform;

      PetscCall(PetscDualSpaceGetUniform(sp->pointSpaces[c-pStart],&cUniform));
      if (!cUniform) break;
    }
    if ((c > cStart) && sp->pointSpaces[c-pStart] != sp->pointSpaces[c-1-pStart]) break;
  }
  if (c < cEnd) sp->uniform = PETSC_FALSE;
  for (h = 0; h < depth; h++) {
    PetscInt hStart, hEnd;

    PetscCall(DMPlexGetHeightStratum(dm, h, &hStart, &hEnd));
    for (c = hStart; c < hEnd; c++) {
      PetscInt coneSize, e;
      PetscDualSpace cspace = sp->pointSpaces[c-pStart];
      const PetscInt *cone;
      const PetscInt *refCone;

      if (!cspace) continue;
      PetscCall(DMPlexGetConeSize(dm, c, &coneSize));
      PetscCall(DMPlexGetCone(dm, c, &cone));
      PetscCall(DMPlexGetCone(cspace->dm, 0, &refCone));
      for (e = 0; e < coneSize; e++) {
        PetscInt point = cone[e];
        PetscInt refpoint = refCone[e];
        PetscDualSpace espace;

        PetscCall(PetscDualSpaceGetPointSubspace(cspace,refpoint,&espace));
        if (sp->pointSpaces[point-pStart] == NULL) {
          PetscCall(PetscObjectReference((PetscObject)espace));
          sp->pointSpaces[point-pStart] = espace;
        }
      }
    }
  }
  PetscCall(PetscDualSpaceGetSection(sp, &section));
  PetscCall(PetscDualSpaceGetDimension(sp, &spdim));
  PetscCall(PetscMalloc1(spdim, &(sp->functional)));
  PetscCall(PetscDualSpacePushForwardSubspaces_Internal(sp, pStart, pEnd));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDualSpaceRefinedView_Ascii(PetscDualSpace sp, PetscViewer viewer)
{
  PetscFunctionBegin;
  if (sp->dm && sp->pointSpaces) {
    PetscInt pStart, pEnd;
    PetscInt cStart, cEnd, c;

    PetscCall(DMPlexGetChart(sp->dm, &pStart, &pEnd));
    PetscCall(DMPlexGetHeightStratum(sp->dm, 0, &cStart, &cEnd));
    PetscCall(PetscViewerASCIIPrintf(viewer, "Refined dual space:\n"));
    PetscCall(PetscViewerASCIIPushTab(viewer));
    for (c = cStart; c < cEnd; c++) {
      if (!sp->pointSpaces[c-pStart]) {
        PetscCall(PetscViewerASCIIPrintf(viewer, "Cell space %D not set yet\n", c));
      } else {
        PetscCall(PetscViewerASCIIPrintf(viewer, "Cell space %D:ot set yet\n", c));
        PetscCall(PetscViewerASCIIPushTab(viewer));
        PetscCall(PetscDualSpaceView(sp->pointSpaces[c-pStart],viewer));
        PetscCall(PetscViewerASCIIPopTab(viewer));
      }
    }
    PetscCall(PetscViewerASCIIPopTab(viewer));
  } else {
    PetscCall(PetscViewerASCIIPrintf(viewer, "Refined dual space: (cell spaces not set yet)\n"));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDualSpaceView_Refined(PetscDualSpace sp, PetscViewer viewer)
{
  PetscBool      iascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  PetscCall(PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &iascii));
  if (iascii) PetscCall(PetscDualSpaceRefinedView_Ascii(sp, viewer));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDualSpaceInitialize_Refined(PetscDualSpace sp)
{
  PetscFunctionBegin;
  sp->ops->destroy              = PetscDualSpaceDestroy_Refined;
  sp->ops->view                 = PetscDualSpaceView_Refined;
  sp->ops->setfromoptions       = NULL;
  sp->ops->duplicate            = NULL;
  sp->ops->setup                = PetscDualSpaceSetUp_Refined;
  sp->ops->createheightsubspace = NULL;
  sp->ops->createpointsubspace  = NULL;
  sp->ops->getsymmetries        = NULL;
  sp->ops->apply                = PetscDualSpaceApplyDefault;
  sp->ops->applyall             = PetscDualSpaceApplyAllDefault;
  sp->ops->applyint             = PetscDualSpaceApplyInteriorDefault;
  sp->ops->createalldata        = PetscDualSpaceCreateAllDataDefault;
  sp->ops->createintdata        = PetscDualSpaceCreateInteriorDataDefault;
  PetscFunctionReturn(0);
}

/*MC
  PETSCDUALSPACEREFINED = "refined" - A PetscDualSpace object that defines the joint dual space of a group of cells, usually refined from one larger cell

  Level: intermediate

.seealso: PetscDualSpaceType, PetscDualSpaceCreate(), PetscDualSpaceSetType()
M*/
PETSC_EXTERN PetscErrorCode PetscDualSpaceCreate_Refined(PetscDualSpace sp)
{
  PetscDualSpace_Refined *ref;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 1);
  PetscCall(PetscNewLog(sp,&ref));
  sp->data = ref;

  PetscCall(PetscDualSpaceInitialize_Refined(sp));
  PetscCall(PetscObjectComposeFunction((PetscObject) sp, "PetscDualSpaceRefinedSetCellSpaces_C", PetscDualSpaceRefinedSetCellSpaces_Refined));
  PetscFunctionReturn(0);
}
