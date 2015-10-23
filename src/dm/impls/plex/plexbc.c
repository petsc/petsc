#include <petsc/private/dmpleximpl.h>   /*I      "petscdmplex.h"   I*/

#undef __FUNCT__
#define __FUNCT__ "BoundaryDuplicate"
static PetscErrorCode BoundaryDuplicate(DMBoundary bd, DMBoundary *boundary)
{
  DMBoundary     b = bd, b2, bold = NULL;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  *boundary = NULL;
  for (; b; b = b->next, bold = b2) {
    ierr = PetscNew(&b2);CHKERRQ(ierr);
    ierr = PetscStrallocpy(b->name, (char **) &b2->name);CHKERRQ(ierr);
    ierr = PetscStrallocpy(b->labelname, (char **) &b2->labelname);CHKERRQ(ierr);
    ierr = PetscMalloc1(b->numids, &b2->ids);CHKERRQ(ierr);
    ierr = PetscMemcpy(b2->ids, b->ids, b->numids*sizeof(PetscInt));CHKERRQ(ierr);
    ierr = PetscMalloc1(b->numcomps, &b2->comps);CHKERRQ(ierr);
    ierr = PetscMemcpy(b2->comps, b->comps, b->numcomps*sizeof(PetscInt));CHKERRQ(ierr);
    b2->label     = NULL;
    b2->essential = b->essential;
    b2->field     = b->field;
    b2->numcomps  = b->numcomps;
    b2->func      = b->func;
    b2->numids    = b->numids;
    b2->ctx       = b->ctx;
    b2->next      = NULL;
    if (!*boundary) *boundary   = b2;
    if (bold)        bold->next = b2;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexCopyBoundary"
PetscErrorCode DMPlexCopyBoundary(DM dm, DM dmNew)
{
  DM_Plex       *mesh    = (DM_Plex *) dm->data;
  DM_Plex       *meshNew = (DM_Plex *) dmNew->data;
  DMBoundary     b;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = BoundaryDuplicate(mesh->boundary, &meshNew->boundary);CHKERRQ(ierr);
  for (b = meshNew->boundary; b; b = b->next) {
    if (b->labelname) {
      ierr = DMPlexGetLabel(dmNew, b->labelname, &b->label);CHKERRQ(ierr);
      if (!b->label) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Label %s does not exist in this DM", b->labelname);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexAddBoundary"
/*@C
  DMPlexAddBoundary - Add a boundary condition to the model

  Input Parameters:
+ dm          - The mesh object
. isEssential - Flag for an essential (Dirichlet) condition, as opposed to a natural (Neumann) condition
. name        - The BC name
. labelname   - The label defining constrained points
. field       - The field to constrain
. numcomps    - The number of constrained field components
. comps       - An array of constrained component numbers
. bcFunc      - A pointwise function giving boundary values
. numids      - The number of DMLabel ids for constrained points
. ids         - An array of ids for constrained points
- ctx         - An optional user context for bcFunc

  Options Database Keys:
+ -bc_<boundary name> <num> - Overrides the boundary ids
- -bc_<boundary name>_comp <num> - Overrides the boundary components

  Level: developer

.seealso: DMPlexGetBoundary()
@*/
PetscErrorCode DMPlexAddBoundary(DM dm, PetscBool isEssential, const char name[], const char labelname[], PetscInt field, PetscInt numcomps, const PetscInt *comps, void (*bcFunc)(), PetscInt numids, const PetscInt *ids, void *ctx)
{
  DM_Plex       *mesh = (DM_Plex *) dm->data;
  DMBoundary     b;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = PetscNew(&b);CHKERRQ(ierr);
  ierr = PetscStrallocpy(name, (char **) &b->name);CHKERRQ(ierr);
  ierr = PetscStrallocpy(labelname, (char **) &b->labelname);CHKERRQ(ierr);
  ierr = PetscMalloc1(numcomps, &b->comps);CHKERRQ(ierr);
  if (numcomps) {ierr = PetscMemcpy(b->comps, comps, numcomps*sizeof(PetscInt));CHKERRQ(ierr);}
  ierr = PetscMalloc1(numids, &b->ids);CHKERRQ(ierr);
  if (numids) {ierr = PetscMemcpy(b->ids, ids, numids*sizeof(PetscInt));CHKERRQ(ierr);}
  if (b->labelname) {
    ierr = DMPlexGetLabel(dm, b->labelname, &b->label);CHKERRQ(ierr);
    if (!b->label) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Label %s does not exist in this DM", b->labelname);
  }
  b->essential   = isEssential;
  b->field       = field;
  b->numcomps    = numcomps;
  b->func        = bcFunc;
  b->numids      = numids;
  b->ctx         = ctx;
  b->next        = mesh->boundary;
  mesh->boundary = b;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexGetNumBoundary"
/*@
  DMPlexGetNumBoundary - Get the number of registered BC

  Input Parameters:
. dm - The mesh object

  Output Parameters:
. numBd - The number of BC

  Level: intermediate

.seealso: DMPlexAddBoundary(), DMPlexGetBoundary()
@*/
PetscErrorCode DMPlexGetNumBoundary(DM dm, PetscInt *numBd)
{
  DM_Plex   *mesh = (DM_Plex *) dm->data;
  DMBoundary b    = mesh->boundary;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(numBd, 2);
  *numBd = 0;
  while (b) {++(*numBd); b = b->next;}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexGetBoundary"
/*@C
  DMPlexGetBoundary - Add a boundary condition to the model

  Input Parameters:
+ dm          - The mesh object
- bd          - The BC number

  Output Parameters:
+ isEssential - Flag for an essential (Dirichlet) condition, as opposed to a natural (Neumann) condition
. name        - The BC name
. labelname   - The label defining constrained points
. field       - The field to constrain
. numcomps    - The number of constrained field components
. comps       - An array of constrained component numbers
. bcFunc      - A pointwise function giving boundary values
. numids      - The number of DMLabel ids for constrained points
. ids         - An array of ids for constrained points
- ctx         - An optional user context for bcFunc

  Options Database Keys:
+ -bc_<boundary name> <num> - Overrides the boundary ids
- -bc_<boundary name>_comp <num> - Overrides the boundary components

  Level: developer

.seealso: DMPlexAddBoundary()
@*/
PetscErrorCode DMPlexGetBoundary(DM dm, PetscInt bd, PetscBool *isEssential, const char **name, const char **labelname, PetscInt *field, PetscInt *numcomps, const PetscInt **comps, void (**func)(), PetscInt *numids, const PetscInt **ids, void **ctx)
{
  DM_Plex   *mesh = (DM_Plex *) dm->data;
  DMBoundary b    = mesh->boundary;
  PetscInt   n    = 0;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  while (b) {
    if (n == bd) break;
    b = b->next;
    ++n;
  }
  if (n != bd) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Boundary %d is not in [0, %d)", bd, n);
  if (isEssential) {
    PetscValidPointer(isEssential, 3);
    *isEssential = b->essential;
  }
  if (name) {
    PetscValidPointer(name, 4);
    *name = b->name;
  }
  if (labelname) {
    PetscValidPointer(labelname, 5);
    *labelname = b->labelname;
  }
  if (field) {
    PetscValidPointer(field, 6);
    *field = b->field;
  }
  if (numcomps) {
    PetscValidPointer(numcomps, 7);
    *numcomps = b->numcomps;
  }
  if (comps) {
    PetscValidPointer(comps, 8);
    *comps = b->comps;
  }
  if (func) {
    PetscValidPointer(func, 9);
    *func = b->func;
  }
  if (numids) {
    PetscValidPointer(numids, 10);
    *numids = b->numids;
  }
  if (ids) {
    PetscValidPointer(ids, 11);
    *ids = b->ids;
  }
  if (ctx) {
    PetscValidPointer(ctx, 12);
    *ctx = b->ctx;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexIsBoundaryPoint"
PetscErrorCode DMPlexIsBoundaryPoint(DM dm, PetscInt point, PetscBool *isBd)
{
  DM_Plex       *mesh = (DM_Plex *) dm->data;
  DMBoundary     b    = mesh->boundary;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(isBd, 3);
  *isBd = PETSC_FALSE;
  while (b && !(*isBd)) {
    if (b->label) {
      PetscInt i;

      for (i = 0; i < b->numids && !(*isBd); ++i) {
        ierr = DMLabelStratumHasPoint(b->label, b->ids[i], point, isBd);CHKERRQ(ierr);
      }
    }
    b = b->next;
  }
  PetscFunctionReturn(0);
}
