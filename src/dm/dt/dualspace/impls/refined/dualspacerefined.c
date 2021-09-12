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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 1);
  PetscValidPointer(cellSpaces,2);
  if (sp->setupcalled) SETERRQ(PetscObjectComm((PetscObject)sp), PETSC_ERR_ARG_WRONGSTATE, "Cannot change cell spaces after setup is called");
  ierr = PetscTryMethod(sp, "PetscDualSpaceRefinedSetCellSpaces_C", (PetscDualSpace,const PetscDualSpace []),(sp,cellSpaces));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDualSpaceRefinedSetCellSpaces_Refined(PetscDualSpace sp, const PetscDualSpace cellSpaces[])
{
  DM dm;
  PetscInt pStart, pEnd;
  PetscInt cStart, cEnd, c;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  dm = sp->dm;
  if (!dm) SETERRQ(PetscObjectComm((PetscObject) sp), PETSC_ERR_ARG_WRONGSTATE, "PetscDualSpace must have a DM (PetscDualSpaceSetDM()) before calling PetscDualSpaceRefinedSetCellSpaces");
  ierr = DMPlexGetChart(dm, &pStart, &pEnd);CHKERRQ(ierr);
  if (!sp->pointSpaces) {

    ierr = PetscCalloc1(pEnd-pStart,&(sp->pointSpaces));CHKERRQ(ierr);
  }
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  for (c = 0; c < cEnd - cStart; c++) {
    ierr = PetscObjectReference((PetscObject)cellSpaces[c]);CHKERRQ(ierr);
    ierr = PetscDualSpaceDestroy(&(sp->pointSpaces[c + cStart - pStart]));CHKERRQ(ierr);
    sp->pointSpaces[c+cStart-pStart] = cellSpaces[c];
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDualSpaceDestroy_Refined(PetscDualSpace sp)
{
  PetscDualSpace_Refined *ref = (PetscDualSpace_Refined *) sp->data;
  PetscErrorCode          ierr;

  PetscFunctionBegin;
  ierr = PetscObjectComposeFunction((PetscObject) sp, "PetscDualSpaceRefinedSetCellSpaces_C", NULL);CHKERRQ(ierr);
  ierr = PetscFree(ref);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDualSpaceSetUp_Refined(PetscDualSpace sp)
{
  PetscInt pStart, pEnd, depth;
  PetscInt cStart, cEnd, c, spdim;
  PetscInt h;
  DM dm;
  PetscSection   section;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscDualSpaceGetDM(sp, &dm);CHKERRQ(ierr);
  ierr = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
  ierr = DMPlexGetChart(dm, &pStart, &pEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  for (c = cStart; c < cEnd; c++) {
    if (sp->pointSpaces[c-pStart]) {
      PetscInt ccStart, ccEnd;
      if (sp->pointSpaces[c-pStart]->k != sp->k) SETERRQ(PetscObjectComm((PetscObject)sp), PETSC_ERR_ARG_INCOMP, "All cell spaces must have the same form degree as the refined dual space");
      if (sp->pointSpaces[c-pStart]->Nc != sp->Nc) SETERRQ(PetscObjectComm((PetscObject)sp), PETSC_ERR_ARG_INCOMP, "All cell spaces must have the same number of components as the refined dual space");
      ierr = DMPlexGetHeightStratum(sp->pointSpaces[c-pStart]->dm, 0, &ccStart, &ccEnd);CHKERRQ(ierr);
      if (ccEnd - ccStart != 1) SETERRQ(PetscObjectComm((PetscObject)sp), PETSC_ERR_ARG_INCOMP, "All cell spaces must have a single cell themselves");
    }
  }
  for (c = cStart; c < cEnd; c++) {
    if (sp->pointSpaces[c-pStart]) {
      PetscBool cUniform;

      ierr = PetscDualSpaceGetUniform(sp->pointSpaces[c-pStart],&cUniform);CHKERRQ(ierr);
      if (!cUniform) break;
    }
    if ((c > cStart) && sp->pointSpaces[c-pStart] != sp->pointSpaces[c-1-pStart]) break;
  }
  if (c < cEnd) sp->uniform = PETSC_FALSE;
  for (h = 0; h < depth; h++) {
    PetscInt hStart, hEnd;

    ierr = DMPlexGetHeightStratum(dm, h, &hStart, &hEnd);CHKERRQ(ierr);
    for (c = hStart; c < hEnd; c++) {
      PetscInt coneSize, e;
      PetscDualSpace cspace = sp->pointSpaces[c-pStart];
      const PetscInt *cone;
      const PetscInt *refCone;

      if (!cspace) continue;
      ierr = DMPlexGetConeSize(dm, c, &coneSize);CHKERRQ(ierr);
      ierr = DMPlexGetCone(dm, c, &cone);CHKERRQ(ierr);
      ierr = DMPlexGetCone(cspace->dm, 0, &refCone);CHKERRQ(ierr);
      for (e = 0; e < coneSize; e++) {
        PetscInt point = cone[e];
        PetscInt refpoint = refCone[e];
        PetscDualSpace espace;

        ierr = PetscDualSpaceGetPointSubspace(cspace,refpoint,&espace);CHKERRQ(ierr);
        if (sp->pointSpaces[point-pStart] == NULL) {
          ierr = PetscObjectReference((PetscObject)espace);CHKERRQ(ierr);
          sp->pointSpaces[point-pStart] = espace;
        }
      }
    }
  }
  ierr = PetscDualSpaceGetSection(sp, &section);CHKERRQ(ierr);
  ierr = PetscDualSpaceGetDimension(sp, &spdim);CHKERRQ(ierr);
  ierr = PetscMalloc1(spdim, &(sp->functional));CHKERRQ(ierr);
  ierr = PetscDualSpacePushForwardSubspaces_Internal(sp, pStart, pEnd);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDualSpaceRefinedView_Ascii(PetscDualSpace sp, PetscViewer viewer)
{
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  if (sp->dm && sp->pointSpaces) {
    PetscInt pStart, pEnd;
    PetscInt cStart, cEnd, c;

    ierr = DMPlexGetChart(sp->dm, &pStart, &pEnd);CHKERRQ(ierr);
    ierr = DMPlexGetHeightStratum(sp->dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "Refined dual space:\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    for (c = cStart; c < cEnd; c++) {
      if (!sp->pointSpaces[c-pStart]) {
        ierr = PetscViewerASCIIPrintf(viewer, "Cell space %D not set yet\n", c);CHKERRQ(ierr);
      } else {
        ierr = PetscViewerASCIIPrintf(viewer, "Cell space %D:ot set yet\n", c);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
        ierr = PetscDualSpaceView(sp->pointSpaces[c-pStart],viewer);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
      }
    }
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  } else {
    ierr = PetscViewerASCIIPrintf(viewer, "Refined dual space: (cell spaces not set yet)\n");CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDualSpaceView_Refined(PetscDualSpace sp, PetscViewer viewer)
{
  PetscBool      iascii;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &iascii);CHKERRQ(ierr);
  if (iascii) {ierr = PetscDualSpaceRefinedView_Ascii(sp, viewer);CHKERRQ(ierr);}
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
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 1);
  ierr     = PetscNewLog(sp,&ref);CHKERRQ(ierr);
  sp->data = ref;

  ierr = PetscDualSpaceInitialize_Refined(sp);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject) sp, "PetscDualSpaceRefinedSetCellSpaces_C", PetscDualSpaceRefinedSetCellSpaces_Refined);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
