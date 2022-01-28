#include <petsc/private/dmplextransformimpl.h> /*I "petscdmplextransform.h" I*/

#include <petsc/private/petscfeimpl.h>  /* For PetscFEInterpolate_Static() */

PetscClassId DMPLEXTRANSFORM_CLASSID;

PetscFunctionList DMPlexTransformList = NULL;
PetscBool         DMPlexTransformRegisterAllCalled = PETSC_FALSE;

/* Construct cell type order since we must loop over cell types in depth order */
static PetscErrorCode DMPlexCreateCellTypeOrder_Internal(PetscInt dim, PetscInt *ctOrder[], PetscInt *ctOrderInv[])
{
  PetscInt      *ctO, *ctOInv;
  PetscInt       c, d, off = 0;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscCalloc2(DM_NUM_POLYTOPES+1, &ctO, DM_NUM_POLYTOPES+1, &ctOInv);CHKERRQ(ierr);
  for (d = 3; d >= dim; --d) {
    for (c = 0; c <= DM_NUM_POLYTOPES; ++c) {
      if (DMPolytopeTypeGetDim((DMPolytopeType) c) != d) continue;
      ctO[off++] = c;
    }
  }
  if (dim != 0) {
    for (c = 0; c <= DM_NUM_POLYTOPES; ++c) {
      if (DMPolytopeTypeGetDim((DMPolytopeType) c) != 0) continue;
      ctO[off++] = c;
    }
  }
  for (d = dim-1; d > 0; --d) {
    for (c = 0; c <= DM_NUM_POLYTOPES; ++c) {
      if (DMPolytopeTypeGetDim((DMPolytopeType) c) != d) continue;
      ctO[off++] = c;
    }
  }
  for (c = 0; c <= DM_NUM_POLYTOPES; ++c) {
    if (DMPolytopeTypeGetDim((DMPolytopeType) c) >= 0) continue;
    ctO[off++] = c;
  }
  PetscAssertFalse(off != DM_NUM_POLYTOPES+1,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Invalid offset %D for cell type order", off);
  for (c = 0; c <= DM_NUM_POLYTOPES; ++c) {
    ctOInv[ctO[c]] = c;
  }
  *ctOrder    = ctO;
  *ctOrderInv = ctOInv;
  PetscFunctionReturn(0);
}

/*@C
  DMPlexTransformRegister - Adds a new transform component implementation

  Not Collective

  Input Parameters:
+ name        - The name of a new user-defined creation routine
- create_func - The creation routine itself

  Notes:
  DMPlexTransformRegister() may be called multiple times to add several user-defined transforms

  Sample usage:
.vb
  DMPlexTransformRegister("my_transform", MyTransformCreate);
.ve

  Then, your transform type can be chosen with the procedural interface via
.vb
  DMPlexTransformCreate(MPI_Comm, DMPlexTransform *);
  DMPlexTransformSetType(DMPlexTransform, "my_transform");
.ve
  or at runtime via the option
.vb
  -dm_plex_transform_type my_transform
.ve

  Level: advanced

.seealso: DMPlexTransformRegisterAll(), DMPlexTransformRegisterDestroy()
@*/
PetscErrorCode DMPlexTransformRegister(const char name[], PetscErrorCode (*create_func)(DMPlexTransform))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMInitializePackage();CHKERRQ(ierr);
  ierr = PetscFunctionListAdd(&DMPlexTransformList, name, create_func);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode DMPlexTransformCreate_Filter(DMPlexTransform);
PETSC_EXTERN PetscErrorCode DMPlexTransformCreate_Regular(DMPlexTransform);
PETSC_EXTERN PetscErrorCode DMPlexTransformCreate_ToBox(DMPlexTransform);
PETSC_EXTERN PetscErrorCode DMPlexTransformCreate_Alfeld(DMPlexTransform);
PETSC_EXTERN PetscErrorCode DMPlexTransformCreate_SBR(DMPlexTransform);
PETSC_EXTERN PetscErrorCode DMPlexTransformCreate_BL(DMPlexTransform);
PETSC_EXTERN PetscErrorCode DMPlexTransformCreate_1D(DMPlexTransform);
PETSC_EXTERN PetscErrorCode DMPlexTransformCreate_Extrude(DMPlexTransform);

/*@C
  DMPlexTransformRegisterAll - Registers all of the transform components in the DM package.

  Not Collective

  Level: advanced

.seealso: DMRegisterAll(), DMPlexTransformRegisterDestroy()
@*/
PetscErrorCode DMPlexTransformRegisterAll(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (DMPlexTransformRegisterAllCalled) PetscFunctionReturn(0);
  DMPlexTransformRegisterAllCalled = PETSC_TRUE;

  ierr = DMPlexTransformRegister(DMPLEXTRANSFORMFILTER,     DMPlexTransformCreate_Filter);CHKERRQ(ierr);
  ierr = DMPlexTransformRegister(DMPLEXREFINEREGULAR,       DMPlexTransformCreate_Regular);CHKERRQ(ierr);
  ierr = DMPlexTransformRegister(DMPLEXREFINETOBOX,         DMPlexTransformCreate_ToBox);CHKERRQ(ierr);
  ierr = DMPlexTransformRegister(DMPLEXREFINEALFELD,        DMPlexTransformCreate_Alfeld);CHKERRQ(ierr);
  ierr = DMPlexTransformRegister(DMPLEXREFINEBOUNDARYLAYER, DMPlexTransformCreate_BL);CHKERRQ(ierr);
  ierr = DMPlexTransformRegister(DMPLEXREFINESBR,           DMPlexTransformCreate_SBR);CHKERRQ(ierr);
  ierr = DMPlexTransformRegister(DMPLEXREFINE1D,            DMPlexTransformCreate_1D);CHKERRQ(ierr);
  ierr = DMPlexTransformRegister(DMPLEXEXTRUDE,             DMPlexTransformCreate_Extrude);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  DMPlexTransformRegisterDestroy - This function destroys the . It is called from PetscFinalize().

  Level: developer

.seealso: PetscInitialize()
@*/
PetscErrorCode DMPlexTransformRegisterDestroy(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFunctionListDestroy(&DMPlexTransformList);CHKERRQ(ierr);
  DMPlexTransformRegisterAllCalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*@
  DMPlexTransformCreate - Creates an empty transform object. The type can then be set with DMPlexTransformSetType().

  Collective

  Input Parameter:
. comm - The communicator for the transform object

  Output Parameter:
. dm - The transform object

  Level: beginner

.seealso: DMPlexTransformSetType(), DMPLEXREFINEREGULAR, DMPLEXTRANSFORMFILTER
@*/
PetscErrorCode DMPlexTransformCreate(MPI_Comm comm, DMPlexTransform *tr)
{
  DMPlexTransform t;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidPointer(tr, 2);
  *tr = NULL;
  ierr = DMInitializePackage();CHKERRQ(ierr);

  ierr = PetscHeaderCreate(t, DMPLEXTRANSFORM_CLASSID, "DMPlexTransform", "Mesh Transform", "DMPlexTransform", comm, DMPlexTransformDestroy, DMPlexTransformView);CHKERRQ(ierr);
  t->setupcalled = PETSC_FALSE;
  ierr = PetscCalloc2(DM_NUM_POLYTOPES, &t->coordFE, DM_NUM_POLYTOPES, &t->refGeom);CHKERRQ(ierr);
  *tr = t;
  PetscFunctionReturn(0);
}

/*@C
  DMSetType - Sets the particular implementation for a transform.

  Collective on tr

  Input Parameters:
+ tr     - The transform
- method - The name of the transform type

  Options Database Key:
. -dm_plex_transform_type <type> - Sets the transform type; use -help for a list of available types

  Notes:
  See "petsc/include/petscdmplextransform.h" for available transform types

  Level: intermediate

.seealso: DMPlexTransformGetType(), DMPlexTransformCreate()
@*/
PetscErrorCode DMPlexTransformSetType(DMPlexTransform tr, DMPlexTransformType method)
{
  PetscErrorCode (*r)(DMPlexTransform);
  PetscBool      match;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tr, DMPLEXTRANSFORM_CLASSID, 1);
  ierr = PetscObjectTypeCompare((PetscObject) tr, method, &match);CHKERRQ(ierr);
  if (match) PetscFunctionReturn(0);

  ierr = DMPlexTransformRegisterAll();CHKERRQ(ierr);
  ierr = PetscFunctionListFind(DMPlexTransformList, method, &r);CHKERRQ(ierr);
  PetscAssertFalse(!r,PetscObjectComm((PetscObject) tr), PETSC_ERR_ARG_UNKNOWN_TYPE, "Unknown DMPlexTransform type: %s", method);

  if (tr->ops->destroy) {ierr = (*tr->ops->destroy)(tr);CHKERRQ(ierr);}
  ierr = PetscMemzero(tr->ops, sizeof(*tr->ops));CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject) tr, method);CHKERRQ(ierr);
  ierr = (*r)(tr);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  DMPlexTransformGetType - Gets the type name (as a string) from the transform.

  Not Collective

  Input Parameter:
. tr  - The DMPlexTransform

  Output Parameter:
. type - The DMPlexTransform type name

  Level: intermediate

.seealso: DMPlexTransformSetType(), DMPlexTransformCreate()
@*/
PetscErrorCode DMPlexTransformGetType(DMPlexTransform tr, DMPlexTransformType *type)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tr, DMPLEXTRANSFORM_CLASSID, 1);
  PetscValidPointer(type, 2);
  ierr = DMPlexTransformRegisterAll();CHKERRQ(ierr);
  *type = ((PetscObject) tr)->type_name;
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexTransformView_Ascii(DMPlexTransform tr, PetscViewer v)
{
  PetscViewerFormat format;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscViewerGetFormat(v, &format);CHKERRQ(ierr);
  if (format == PETSC_VIEWER_ASCII_INFO_DETAIL) {
    const PetscInt *trTypes = NULL;
    IS              trIS;
    PetscInt        cols = 8;
    PetscInt        Nrt = 8, f, g;

    ierr = PetscViewerASCIIPrintf(v, "Source Starts\n");CHKERRQ(ierr);
    for (g = 0; g <= cols; ++g) {ierr = PetscViewerASCIIPrintf(v, " % 14s", DMPolytopeTypes[g]);CHKERRQ(ierr);}
    ierr = PetscViewerASCIIPrintf(v, "\n");CHKERRQ(ierr);
    for (f = 0; f <= cols; ++f) {ierr = PetscViewerASCIIPrintf(v, " % 14d", tr->ctStart[f]);CHKERRQ(ierr);}
    ierr = PetscViewerASCIIPrintf(v, "\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(v, "Target Starts\n");CHKERRQ(ierr);
    for (g = 0; g <= cols; ++g) {ierr = PetscViewerASCIIPrintf(v, " % 14s", DMPolytopeTypes[g]);CHKERRQ(ierr);}
    ierr = PetscViewerASCIIPrintf(v, "\n");CHKERRQ(ierr);
    for (f = 0; f <= cols; ++f) {ierr = PetscViewerASCIIPrintf(v, " % 14d", tr->ctStartNew[f]);CHKERRQ(ierr);}
    ierr = PetscViewerASCIIPrintf(v, "\n");CHKERRQ(ierr);

    if (tr->trType) {
      ierr = DMLabelGetNumValues(tr->trType, &Nrt);CHKERRQ(ierr);
      ierr = DMLabelGetValueIS(tr->trType, &trIS);CHKERRQ(ierr);
      ierr = ISGetIndices(trIS, &trTypes);CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPrintf(v, "Offsets\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(v, "     ");CHKERRQ(ierr);
    for (g = 0; g < cols; ++g) {
      ierr = PetscViewerASCIIPrintf(v, " % 14s", DMPolytopeTypes[g]);CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPrintf(v, "\n");CHKERRQ(ierr);
    for (f = 0; f < Nrt; ++f) {
      ierr = PetscViewerASCIIPrintf(v, "%2d  |", trTypes ? trTypes[f] : f);CHKERRQ(ierr);
      for (g = 0; g < cols; ++g) {
        ierr = PetscViewerASCIIPrintf(v, " % 14D", tr->offset[f*DM_NUM_POLYTOPES+g]);CHKERRQ(ierr);
      }
      ierr = PetscViewerASCIIPrintf(v, " |\n");CHKERRQ(ierr);
    }
    if (trTypes) {
      ierr = ISGetIndices(trIS, &trTypes);CHKERRQ(ierr);
      ierr = ISDestroy(&trIS);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

/*@C
  DMPlexTransformView - Views a DMPlexTransform

  Collective on tr

  Input Parameter:
+ tr - the DMPlexTransform object to view
- v  - the viewer

  Level: beginner

.seealso DMPlexTransformDestroy(), DMPlexTransformCreate()
@*/
PetscErrorCode DMPlexTransformView(DMPlexTransform tr, PetscViewer v)
{
  PetscBool      isascii;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tr, DMPLEXTRANSFORM_CLASSID ,1);
  if (!v) {ierr = PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject) tr), &v);CHKERRQ(ierr);}
  PetscValidHeaderSpecific(v, PETSC_VIEWER_CLASSID, 2);
  PetscCheckSameComm(tr, 1, v, 2);
  ierr = PetscViewerCheckWritable(v);CHKERRQ(ierr);
  ierr = PetscObjectPrintClassNamePrefixType((PetscObject) tr, v);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject) v, PETSCVIEWERASCII, &isascii);CHKERRQ(ierr);
  if (isascii) {ierr = DMPlexTransformView_Ascii(tr, v);CHKERRQ(ierr);}
  if (tr->ops->view) {ierr = (*tr->ops->view)(tr, v);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

/*@
  DMPlexTransformSetFromOptions - Sets parameters in a transform from the options database

  Collective on tr

  Input Parameter:
. tr - the DMPlexTransform object to set options for

  Options Database:
. -dm_plex_transform_type - Set the transform type, e.g. refine_regular

  Level: intermediate

.seealso DMPlexTransformView(), DMPlexTransformCreate()
@*/
PetscErrorCode DMPlexTransformSetFromOptions(DMPlexTransform tr)
{
  char           typeName[1024];
  const char    *defName = DMPLEXREFINEREGULAR;
  PetscBool      flg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tr,DMPLEXTRANSFORM_CLASSID,1);
  ierr = PetscObjectOptionsBegin((PetscObject)tr);CHKERRQ(ierr);
  ierr = PetscOptionsFList("-dm_plex_transform_type", "DMPlexTransform", "DMPlexTransformSetType", DMPlexTransformList, defName, typeName, 1024, &flg);CHKERRQ(ierr);
  if (flg) {ierr = DMPlexTransformSetType(tr, typeName);CHKERRQ(ierr);}
  else if (!((PetscObject) tr)->type_name) {ierr = DMPlexTransformSetType(tr, defName);CHKERRQ(ierr);}
  if (tr->ops->setfromoptions) {ierr = (*tr->ops->setfromoptions)(PetscOptionsObject,tr);CHKERRQ(ierr);}
  /* process any options handlers added with PetscObjectAddOptionsHandler() */
  ierr = PetscObjectProcessOptionsHandlers(PetscOptionsObject,(PetscObject) tr);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  DMPlexTransformDestroy - Destroys a transform.

  Collective on tr

  Input Parameter:
. tr - the transform object to destroy

  Level: beginner

.seealso DMPlexTransformView(), DMPlexTransformCreate()
@*/
PetscErrorCode DMPlexTransformDestroy(DMPlexTransform *tr)
{
  PetscInt       c;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*tr) PetscFunctionReturn(0);
  PetscValidHeaderSpecific((*tr), DMPLEXTRANSFORM_CLASSID, 1);
  if (--((PetscObject) (*tr))->refct > 0) {*tr = NULL; PetscFunctionReturn(0);}

  if ((*tr)->ops->destroy) {
    ierr = (*(*tr)->ops->destroy)(*tr);CHKERRQ(ierr);
  }
  ierr = DMDestroy(&(*tr)->dm);CHKERRQ(ierr);
  ierr = DMLabelDestroy(&(*tr)->active);CHKERRQ(ierr);
  ierr = DMLabelDestroy(&(*tr)->trType);CHKERRQ(ierr);
  ierr = PetscFree2((*tr)->ctOrderOld, (*tr)->ctOrderInvOld);CHKERRQ(ierr);
  ierr = PetscFree2((*tr)->ctOrderNew, (*tr)->ctOrderInvNew);CHKERRQ(ierr);
  ierr = PetscFree2((*tr)->ctStart, (*tr)->ctStartNew);CHKERRQ(ierr);
  ierr = PetscFree((*tr)->offset);CHKERRQ(ierr);
  for (c = 0; c < DM_NUM_POLYTOPES; ++c) {
    ierr = PetscFEDestroy(&(*tr)->coordFE[c]);CHKERRQ(ierr);
    ierr = PetscFEGeomDestroy(&(*tr)->refGeom[c]);CHKERRQ(ierr);
  }
  if ((*tr)->trVerts) {
    for (c = 0; c < DM_NUM_POLYTOPES; ++c) {
      DMPolytopeType *rct;
      PetscInt       *rsize, *rcone, *rornt, Nct, n, r;

      if (DMPolytopeTypeGetDim((DMPolytopeType) c) > 0) {
        ierr = DMPlexTransformCellTransform((*tr), (DMPolytopeType) c, 0, NULL, &Nct, &rct, &rsize, &rcone, &rornt);CHKERRQ(ierr);
        for (n = 0; n < Nct; ++n) {

          if (rct[n] == DM_POLYTOPE_POINT) continue;
          for (r = 0; r < rsize[n]; ++r) {ierr = PetscFree((*tr)->trSubVerts[c][rct[n]][r]);CHKERRQ(ierr);}
          ierr = PetscFree((*tr)->trSubVerts[c][rct[n]]);CHKERRQ(ierr);
        }
      }
      ierr = PetscFree((*tr)->trSubVerts[c]);CHKERRQ(ierr);
      ierr = PetscFree((*tr)->trVerts[c]);CHKERRQ(ierr);
    }
  }
  ierr = PetscFree3((*tr)->trNv, (*tr)->trVerts, (*tr)->trSubVerts);CHKERRQ(ierr);
  ierr = PetscFree2((*tr)->coordFE, (*tr)->refGeom);CHKERRQ(ierr);
  /* We do not destroy (*dm)->data here so that we can reference count backend objects */
  ierr = PetscHeaderDestroy(tr);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexTransformCreateOffset_Internal(DMPlexTransform tr, PetscInt ctOrderOld[], PetscInt ctStart[], PetscInt **offset)
{
  DMLabel        trType = tr->trType;
  PetscInt       c, cN, *off;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (trType) {
    DM              dm;
    IS              rtIS;
    const PetscInt *reftypes;
    PetscInt        Nrt, r;

    ierr = DMPlexTransformGetDM(tr, &dm);CHKERRQ(ierr);
    ierr = DMLabelGetNumValues(trType, &Nrt);CHKERRQ(ierr);
    ierr = DMLabelGetValueIS(trType, &rtIS);CHKERRQ(ierr);
    ierr = ISGetIndices(rtIS, &reftypes);CHKERRQ(ierr);
    ierr = PetscCalloc1(Nrt*DM_NUM_POLYTOPES, &off);CHKERRQ(ierr);
    for (r = 0; r < Nrt; ++r) {
      const PetscInt  rt = reftypes[r];
      IS              rtIS;
      const PetscInt *points;
      DMPolytopeType  ct;
      PetscInt        p;

      ierr = DMLabelGetStratumIS(trType, rt, &rtIS);CHKERRQ(ierr);
      ierr = ISGetIndices(rtIS, &points);CHKERRQ(ierr);
      p    = points[0];
      ierr = ISRestoreIndices(rtIS, &points);CHKERRQ(ierr);
      ierr = ISDestroy(&rtIS);CHKERRQ(ierr);
      ierr = DMPlexGetCellType(dm, p, &ct);CHKERRQ(ierr);
      for (cN = DM_POLYTOPE_POINT; cN < DM_NUM_POLYTOPES; ++cN) {
        const DMPolytopeType ctNew = (DMPolytopeType) cN;
        DMPolytopeType      *rct;
        PetscInt            *rsize, *cone, *ornt;
        PetscInt             Nct, n, s;

        if (DMPolytopeTypeGetDim(ct) < 0 || DMPolytopeTypeGetDim(ctNew) < 0) {off[r*DM_NUM_POLYTOPES+ctNew] = -1; break;}
        off[r*DM_NUM_POLYTOPES+ctNew] = 0;
        for (s = 0; s <= r; ++s) {
          const PetscInt st = reftypes[s];
          DMPolytopeType sct;
          PetscInt       q, qrt;

          ierr = DMLabelGetStratumIS(trType, st, &rtIS);CHKERRQ(ierr);
          ierr = ISGetIndices(rtIS, &points);CHKERRQ(ierr);
          q    = points[0];
          ierr = ISRestoreIndices(rtIS, &points);CHKERRQ(ierr);
          ierr = ISDestroy(&rtIS);CHKERRQ(ierr);
          ierr = DMPlexGetCellType(dm, q, &sct);CHKERRQ(ierr);
          ierr = DMPlexTransformCellTransform(tr, sct, q, &qrt, &Nct, &rct, &rsize, &cone, &ornt);CHKERRQ(ierr);
          PetscAssertFalse(st != qrt,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Refine type %D of point %D does not match predicted type %D", qrt, q, st);
          if (st == rt) {
            for (n = 0; n < Nct; ++n) if (rct[n] == ctNew) break;
            if (n == Nct) off[r*DM_NUM_POLYTOPES+ctNew] = -1;
            break;
          }
          for (n = 0; n < Nct; ++n) {
            if (rct[n] == ctNew) {
              PetscInt sn;

              ierr = DMLabelGetStratumSize(trType, st, &sn);CHKERRQ(ierr);
              off[r*DM_NUM_POLYTOPES+ctNew] += sn * rsize[n];
            }
          }
        }
      }
    }
    ierr = ISRestoreIndices(rtIS, &reftypes);CHKERRQ(ierr);
    ierr = ISDestroy(&rtIS);CHKERRQ(ierr);
  } else {
    ierr = PetscCalloc1(DM_NUM_POLYTOPES*DM_NUM_POLYTOPES, &off);CHKERRQ(ierr);
    for (c = DM_POLYTOPE_POINT; c < DM_NUM_POLYTOPES; ++c) {
      const DMPolytopeType ct = (DMPolytopeType) c;
      for (cN = DM_POLYTOPE_POINT; cN < DM_NUM_POLYTOPES; ++cN) {
        const DMPolytopeType ctNew = (DMPolytopeType) cN;
        DMPolytopeType      *rct;
        PetscInt            *rsize, *cone, *ornt;
        PetscInt             Nct, n, i;

        if (DMPolytopeTypeGetDim(ct) < 0 || DMPolytopeTypeGetDim(ctNew) < 0) {off[ct*DM_NUM_POLYTOPES+ctNew] = -1; continue;}
        off[ct*DM_NUM_POLYTOPES+ctNew] = 0;
        for (i = DM_POLYTOPE_POINT; i < DM_NUM_POLYTOPES; ++i) {
          const DMPolytopeType ict  = (DMPolytopeType) ctOrderOld[i];
          const DMPolytopeType ictn = (DMPolytopeType) ctOrderOld[i+1];

          ierr = DMPlexTransformCellTransform(tr, ict, PETSC_DETERMINE, NULL, &Nct, &rct, &rsize, &cone, &ornt);CHKERRQ(ierr);
          if (ict == ct) {
            for (n = 0; n < Nct; ++n) if (rct[n] == ctNew) break;
            if (n == Nct) off[ct*DM_NUM_POLYTOPES+ctNew] = -1;
            break;
          }
          for (n = 0; n < Nct; ++n) if (rct[n] == ctNew) off[ct*DM_NUM_POLYTOPES+ctNew] += (ctStart[ictn]-ctStart[ict]) * rsize[n];
        }
      }
    }
  }
  *offset = off;
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexTransformSetUp(DMPlexTransform tr)
{
  DM             dm;
  DMPolytopeType ctCell;
  PetscInt       pStart, pEnd, p, c, celldim = 0;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tr, DMPLEXTRANSFORM_CLASSID, 1);
  if (tr->setupcalled) PetscFunctionReturn(0);
  if (tr->ops->setup) {ierr = (*tr->ops->setup)(tr);CHKERRQ(ierr);}
  ierr = DMPlexTransformGetDM(tr, &dm);CHKERRQ(ierr);
  ierr = DMPlexGetChart(dm, &pStart, &pEnd);CHKERRQ(ierr);
  if (pEnd > pStart) {
    ierr = DMPlexGetCellType(dm, 0, &ctCell);CHKERRQ(ierr);
  } else {
    PetscInt dim;

    ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
    switch (dim) {
      case 0: ctCell = DM_POLYTOPE_POINT;break;
      case 1: ctCell = DM_POLYTOPE_SEGMENT;break;
      case 2: ctCell = DM_POLYTOPE_TRIANGLE;break;
      case 3: ctCell = DM_POLYTOPE_TETRAHEDRON;break;
      default: ctCell = DM_POLYTOPE_UNKNOWN;
    }
  }
  ierr = DMPlexCreateCellTypeOrder_Internal(DMPolytopeTypeGetDim(ctCell), &tr->ctOrderOld, &tr->ctOrderInvOld);CHKERRQ(ierr);
  for (p = pStart; p < pEnd; ++p) {
    DMPolytopeType  ct;
    DMPolytopeType *rct;
    PetscInt       *rsize, *cone, *ornt;
    PetscInt        Nct, n;

    ierr = DMPlexGetCellType(dm, p, &ct);CHKERRQ(ierr);
    PetscAssertFalse(ct == DM_POLYTOPE_UNKNOWN,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "No cell type for point %D", p);
    ierr = DMPlexTransformCellTransform(tr, ct, p, NULL, &Nct, &rct, &rsize, &cone, &ornt);CHKERRQ(ierr);
    for (n = 0; n < Nct; ++n) celldim = PetscMax(celldim, DMPolytopeTypeGetDim(rct[n]));
  }
  ierr = DMPlexCreateCellTypeOrder_Internal(celldim, &tr->ctOrderNew, &tr->ctOrderInvNew);CHKERRQ(ierr);
  /* Construct sizes and offsets for each cell type */
  if (!tr->ctStart) {
    PetscInt *ctS, *ctSN, *ctC, *ctCN;

    ierr = PetscCalloc2(DM_NUM_POLYTOPES+1, &ctS, DM_NUM_POLYTOPES+1, &ctSN);CHKERRQ(ierr);
    ierr = PetscCalloc2(DM_NUM_POLYTOPES+1, &ctC, DM_NUM_POLYTOPES+1, &ctCN);CHKERRQ(ierr);
    for (p = pStart; p < pEnd; ++p) {
      DMPolytopeType  ct;
      DMPolytopeType *rct;
      PetscInt       *rsize, *cone, *ornt;
      PetscInt        Nct, n;

      ierr = DMPlexGetCellType(dm, p, &ct);CHKERRQ(ierr);
      PetscAssertFalse(ct == DM_POLYTOPE_UNKNOWN,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "No cell type for point %D", p);
      ++ctC[ct];
      ierr = DMPlexTransformCellTransform(tr, ct, p, NULL, &Nct, &rct, &rsize, &cone, &ornt);CHKERRQ(ierr);
      for (n = 0; n < Nct; ++n) ctCN[rct[n]] += rsize[n];
    }
    for (c = 0; c < DM_NUM_POLYTOPES; ++c) {
      const PetscInt cto  = tr->ctOrderOld[c];
      const PetscInt cton = tr->ctOrderOld[c+1];
      const PetscInt ctn  = tr->ctOrderNew[c];
      const PetscInt ctnn = tr->ctOrderNew[c+1];

      ctS[cton]  = ctS[cto]  + ctC[cto];
      ctSN[ctnn] = ctSN[ctn] + ctCN[ctn];
    }
    ierr = PetscFree2(ctC, ctCN);CHKERRQ(ierr);
    tr->ctStart    = ctS;
    tr->ctStartNew = ctSN;
  }
  ierr = DMPlexTransformCreateOffset_Internal(tr, tr->ctOrderOld, tr->ctStart, &tr->offset);CHKERRQ(ierr);
  tr->setupcalled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexTransformGetDM(DMPlexTransform tr, DM *dm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tr, DMPLEXTRANSFORM_CLASSID, 1);
  PetscValidPointer(dm, 2);
  *dm = tr->dm;
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexTransformSetDM(DMPlexTransform tr, DM dm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tr, DMPLEXTRANSFORM_CLASSID, 1);
  PetscValidHeaderSpecific(dm, DM_CLASSID, 2);
  ierr = PetscObjectReference((PetscObject) dm);CHKERRQ(ierr);
  ierr = DMDestroy(&tr->dm);CHKERRQ(ierr);
  tr->dm = dm;
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexTransformGetActive(DMPlexTransform tr, DMLabel *active)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tr, DMPLEXTRANSFORM_CLASSID, 1);
  PetscValidPointer(active, 2);
  *active = tr->active;
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexTransformSetActive(DMPlexTransform tr, DMLabel active)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tr, DMPLEXTRANSFORM_CLASSID, 1);
  PetscValidHeaderSpecific(active, DMLABEL_CLASSID, 2);
  ierr = PetscObjectReference((PetscObject) active);CHKERRQ(ierr);
  ierr = DMLabelDestroy(&tr->active);CHKERRQ(ierr);
  tr->active = active;
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexTransformGetCoordinateFE(DMPlexTransform tr, DMPolytopeType ct, PetscFE *fe)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!tr->coordFE[ct]) {
    PetscInt  dim, cdim;
    PetscBool isSimplex;

    switch (ct) {
      case DM_POLYTOPE_SEGMENT:       dim = 1; isSimplex = PETSC_TRUE;  break;
      case DM_POLYTOPE_TRIANGLE:      dim = 2; isSimplex = PETSC_TRUE;  break;
      case DM_POLYTOPE_QUADRILATERAL: dim = 2; isSimplex = PETSC_FALSE; break;
      case DM_POLYTOPE_TETRAHEDRON:   dim = 3; isSimplex = PETSC_TRUE;  break;
      case DM_POLYTOPE_HEXAHEDRON:    dim = 3; isSimplex = PETSC_FALSE; break;
      default: SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "No coordinate FE for cell type %s", DMPolytopeTypes[ct]);
    }
    ierr = DMGetCoordinateDim(tr->dm, &cdim);CHKERRQ(ierr);
    ierr = PetscFECreateLagrange(PETSC_COMM_SELF, dim, cdim, isSimplex, 1, PETSC_DETERMINE, &tr->coordFE[ct]);CHKERRQ(ierr);
    {
      PetscDualSpace  dsp;
      PetscQuadrature quad;
      DM              K;
      PetscFEGeom    *cg;
      PetscScalar    *Xq;
      PetscReal      *xq, *wq;
      PetscInt        Nq, q;

      ierr = DMPlexTransformGetCellVertices(tr, ct, &Nq, &Xq);CHKERRQ(ierr);
      ierr = PetscMalloc1(Nq*cdim, &xq);CHKERRQ(ierr);
      for (q = 0; q < Nq*cdim; ++q) xq[q] = PetscRealPart(Xq[q]);
      ierr = PetscMalloc1(Nq, &wq);CHKERRQ(ierr);
      for (q = 0; q < Nq; ++q) wq[q] = 1.0;
      ierr = PetscQuadratureCreate(PETSC_COMM_SELF, &quad);CHKERRQ(ierr);
      ierr = PetscQuadratureSetData(quad, dim, 1, Nq, xq, wq);CHKERRQ(ierr);
      ierr = PetscFESetQuadrature(tr->coordFE[ct], quad);CHKERRQ(ierr);

      ierr = PetscFEGetDualSpace(tr->coordFE[ct], &dsp);CHKERRQ(ierr);
      ierr = PetscDualSpaceGetDM(dsp, &K);CHKERRQ(ierr);
      ierr = PetscFEGeomCreate(quad, 1, cdim, PETSC_FALSE, &tr->refGeom[ct]);CHKERRQ(ierr);
      cg   = tr->refGeom[ct];
      ierr = DMPlexComputeCellGeometryFEM(K, 0, NULL, cg->v, cg->J, cg->invJ, cg->detJ);CHKERRQ(ierr);
      ierr = PetscQuadratureDestroy(&quad);CHKERRQ(ierr);
    }
  }
  *fe = tr->coordFE[ct];
  PetscFunctionReturn(0);
}

/*@
  DMPlexTransformSetDimensions - Set the dimensions for the transformed DM

  Input Parameters:
+ tr - The DMPlexTransform object
- dm - The original DM

  Output Parameter:
. tdm - The transformed DM

  Level: advanced

.seealso: DMPlexTransformApply(), DMPlexTransformCreate()
@*/
PetscErrorCode DMPlexTransformSetDimensions(DMPlexTransform tr, DM dm, DM tdm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (tr->ops->setdimensions) {
    ierr = (*tr->ops->setdimensions)(tr, dm, tdm);CHKERRQ(ierr);
  } else {
    PetscInt dim, cdim;

    ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
    ierr = DMSetDimension(tdm, dim);CHKERRQ(ierr);
    ierr = DMGetCoordinateDim(dm, &cdim);CHKERRQ(ierr);
    ierr = DMSetCoordinateDim(tdm, cdim);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@
  DMPlexTransformGetTargetPoint - Get the number of a point in the transformed mesh based on information from the original mesh.

  Not collective

  Input Parameters:
+ tr    - The DMPlexTransform
. ct    - The type of the original point which produces the new point
. ctNew - The type of the new point
. p     - The original point which produces the new point
- r     - The replica number of the new point, meaning it is the rth point of type ctNew produced from p

  Output Parameters:
. pNew  - The new point number

  Level: developer

.seealso: DMPlexTransformGetSourcePoint(), DMPlexTransformCellTransform()
@*/
PetscErrorCode DMPlexTransformGetTargetPoint(DMPlexTransform tr, DMPolytopeType ct, DMPolytopeType ctNew, PetscInt p, PetscInt r, PetscInt *pNew)
{
  DMPolytopeType *rct;
  PetscInt       *rsize, *cone, *ornt;
  PetscInt       rt, Nct, n, off, rp;
  DMLabel        trType = tr->trType;
  PetscInt       ctS  = tr->ctStart[ct],       ctE  = tr->ctStart[tr->ctOrderOld[tr->ctOrderInvOld[ct]+1]];
  PetscInt       ctSN = tr->ctStartNew[ctNew], ctEN = tr->ctStartNew[tr->ctOrderNew[tr->ctOrderInvNew[ctNew]+1]];
  PetscInt       newp = ctSN, cind;
  PetscErrorCode ierr;

  PetscFunctionBeginHot;
  PetscAssertFalse((p < ctS) || (p >= ctE),PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D is not a %s [%D, %D)", p, DMPolytopeTypes[ct], ctS, ctE);
  ierr = DMPlexTransformCellTransform(tr, ct, p, &rt, &Nct, &rct, &rsize, &cone, &ornt);CHKERRQ(ierr);
  if (trType) {
    ierr = DMLabelGetValueIndex(trType, rt, &cind);CHKERRQ(ierr);
    ierr = DMLabelGetStratumPointIndex(trType, rt, p, &rp);CHKERRQ(ierr);
    PetscAssertFalse(rp < 0,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cell type %s point %D does not have refine type %D", DMPolytopeTypes[ct], p, rt);
  } else {
    cind = ct;
    rp   = p - ctS;
  }
  off = tr->offset[cind*DM_NUM_POLYTOPES + ctNew];
  PetscAssertFalse(off < 0,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cell type %s (%D) of point %D does not produce type %s for transform %s", DMPolytopeTypes[ct], rt, p, DMPolytopeTypes[ctNew], tr->hdr.type_name);
  newp += off;
  for (n = 0; n < Nct; ++n) {
    if (rct[n] == ctNew) {
      if (rsize[n] && r >= rsize[n])
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Replica number %D should be in [0, %D) for subcell type %s in cell type %s", r, rsize[n], DMPolytopeTypes[rct[n]], DMPolytopeTypes[ct]);
      newp += rp * rsize[n] + r;
      break;
    }
  }

  PetscAssertFalse((newp < ctSN) || (newp >= ctEN),PETSC_COMM_SELF, PETSC_ERR_PLIB, "New point %D is not a %s [%D, %D)", newp, DMPolytopeTypes[ctNew], ctSN, ctEN);
  *pNew = newp;
  PetscFunctionReturn(0);
}

/*@
  DMPlexTransformGetSourcePoint - Get the number of a point in the original mesh based on information from the transformed mesh.

  Not collective

  Input Parameters:
+ tr    - The DMPlexTransform
- pNew  - The new point number

  Output Parameters:
+ ct    - The type of the original point which produces the new point
. ctNew - The type of the new point
. p     - The original point which produces the new point
- r     - The replica number of the new point, meaning it is the rth point of type ctNew produced from p

  Level: developer

.seealso: DMPlexTransformGetTargetPoint(), DMPlexTransformCellTransform()
@*/
PetscErrorCode DMPlexTransformGetSourcePoint(DMPlexTransform tr, PetscInt pNew, DMPolytopeType *ct, DMPolytopeType *ctNew, PetscInt *p, PetscInt *r)
{
  DMLabel         trType = tr->trType;
  DMPolytopeType *rct;
  PetscInt       *rsize, *cone, *ornt;
  PetscInt        rt, Nct, n, rp = 0, rO = 0, pO;
  PetscInt        offset = -1, ctS, ctE, ctO = 0, ctN, ctTmp;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  for (ctN = 0; ctN < DM_NUM_POLYTOPES; ++ctN) {
    PetscInt ctSN = tr->ctStartNew[ctN], ctEN = tr->ctStartNew[tr->ctOrderNew[tr->ctOrderInvNew[ctN]+1]];

    if ((pNew >= ctSN) && (pNew < ctEN)) break;
  }
  PetscAssertFalse(ctN == DM_NUM_POLYTOPES,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Cell type for target point %D could be not found", pNew);
  if (trType) {
    DM              dm;
    IS              rtIS;
    const PetscInt *reftypes;
    PetscInt        Nrt, r, rtStart;

    ierr = DMPlexTransformGetDM(tr, &dm);CHKERRQ(ierr);
    ierr = DMLabelGetNumValues(trType, &Nrt);CHKERRQ(ierr);
    ierr = DMLabelGetValueIS(trType, &rtIS);CHKERRQ(ierr);
    ierr = ISGetIndices(rtIS, &reftypes);CHKERRQ(ierr);
    for (r = 0; r < Nrt; ++r) {
      const PetscInt off = tr->offset[r*DM_NUM_POLYTOPES + ctN];

      if (tr->ctStartNew[ctN] + off > pNew) continue;
      /* Check that any of this refinement type exist */
      /* TODO Actually keep track of the number produced here instead */
      if (off > offset) {rt = reftypes[r]; offset = off;}
    }
    ierr = ISRestoreIndices(rtIS, &reftypes);CHKERRQ(ierr);
    ierr = ISDestroy(&rtIS);CHKERRQ(ierr);
    PetscAssertFalse(offset < 0,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Source cell type for target point %D could be not found", pNew);
    /* TODO Map refinement types to cell types */
    ierr = DMLabelGetStratumBounds(trType, rt, &rtStart, NULL);CHKERRQ(ierr);
    PetscAssertFalse(rtStart < 0,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Refinement type %D has no source points", rt);
    for (ctO = 0; ctO < DM_NUM_POLYTOPES; ++ctO) {
      PetscInt ctS = tr->ctStart[ctO], ctE = tr->ctStart[tr->ctOrderOld[tr->ctOrderInvOld[ctO]+1]];

      if ((rtStart >= ctS) && (rtStart < ctE)) break;
    }
    PetscAssertFalse(ctO == DM_NUM_POLYTOPES,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Could not determine a cell type for refinement type %D", rt);
  } else {
    for (ctTmp = 0; ctTmp < DM_NUM_POLYTOPES; ++ctTmp) {
      const PetscInt off = tr->offset[ctTmp*DM_NUM_POLYTOPES + ctN];

      if (tr->ctStartNew[ctN] + off > pNew) continue;
      if (tr->ctStart[tr->ctOrderOld[tr->ctOrderInvOld[ctTmp]+1]] <= tr->ctStart[ctTmp]) continue;
      /* TODO Actually keep track of the number produced here instead */
      if (off > offset) {ctO = ctTmp; offset = off;}
    }
    PetscAssertFalse(offset < 0,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Source cell type for target point %D could be not found", pNew);
  }
  ctS = tr->ctStart[ctO];
  ctE = tr->ctStart[tr->ctOrderOld[tr->ctOrderInvOld[ctO]+1]];
  ierr = DMPlexTransformCellTransform(tr, (DMPolytopeType) ctO, ctS, &rt, &Nct, &rct, &rsize, &cone, &ornt);CHKERRQ(ierr);
  for (n = 0; n < Nct; ++n) {
    if (rct[n] == ctN) {
      PetscInt tmp = pNew - tr->ctStartNew[ctN] - offset;

      rp = tmp / rsize[n];
      rO = tmp % rsize[n];
      break;
    }
  }
  PetscAssertFalse(n == Nct,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Replica number for target point %D could be not found", pNew);
  pO = rp + ctS;
  PetscAssertFalse((pO < ctS) || (pO >= ctE),PETSC_COMM_SELF, PETSC_ERR_PLIB, "Source point %D is not a %s [%D, %D)", pO, DMPolytopeTypes[ctO], ctS, ctE);
  if (ct)    *ct    = (DMPolytopeType) ctO;
  if (ctNew) *ctNew = (DMPolytopeType) ctN;
  if (p)     *p     = pO;
  if (r)     *r     = rO;
  PetscFunctionReturn(0);
}

/*@
  DMPlexTransformCellTransform - Describes the transform of a given source cell into a set of other target cells. These produced cells become the new mesh.

  Input Parameters:
+ tr     - The DMPlexTransform object
. source - The source cell type
- p      - The source point, which can also determine the refine type

  Output Parameters:
+ rt     - The refine type for this point
. Nt     - The number of types produced by this point
. target - An array of length Nt giving the types produced
. size   - An array of length Nt giving the number of cells of each type produced
. cone   - An array of length Nt*size[t]*coneSize[t] giving the cell type for each point in the cone of each produced point
- ornt   - An array of length Nt*size[t]*coneSize[t] giving the orientation for each point in the cone of each produced point

  Note: The cone array gives the cone of each subcell listed by the first three outputs. For each cone point, we
  need the cell type, point identifier, and orientation within the subcell. The orientation is with respect to the canonical
  division (described in these outputs) of the cell in the original mesh. The point identifier is given by
$   the number of cones to be taken, or 0 for the current cell
$   the cell cone point number at each level from which it is subdivided
$   the replica number r of the subdivision.
The orientation is with respect to the canonical cone orientation. For example, the prescription for edge division is
$   Nt     = 2
$   target = {DM_POLYTOPE_POINT, DM_POLYTOPE_SEGMENT}
$   size   = {1, 2}
$   cone   = {DM_POLYTOPE_POINT, 1, 0, 0, DM_POLYTOPE_POINT, 0, 0,  DM_POLYTOPE_POINT, 0, 0, DM_POLYTOPE_POINT, 1, 1, 0}
$   ornt   = {                         0,                       0,                        0,                          0}

  Level: advanced

.seealso: DMPlexTransformApply(), DMPlexTransformCreate()
@*/
PetscErrorCode DMPlexTransformCellTransform(DMPlexTransform tr, DMPolytopeType source, PetscInt p, PetscInt *rt, PetscInt *Nt, DMPolytopeType *target[], PetscInt *size[], PetscInt *cone[], PetscInt *ornt[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = (*tr->ops->celltransform)(tr, source, p, rt, Nt, target, size, cone, ornt);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexTransformGetSubcellOrientationIdentity(DMPlexTransform tr, DMPolytopeType sct, PetscInt sp, PetscInt so, DMPolytopeType tct, PetscInt r, PetscInt o, PetscInt *rnew, PetscInt *onew)
{
  PetscFunctionBegin;
  *rnew = r;
  *onew = DMPolytopeTypeComposeOrientation(tct, o, so);
  PetscFunctionReturn(0);
}

/* Returns the same thing */
PetscErrorCode DMPlexTransformCellTransformIdentity(DMPlexTransform tr, DMPolytopeType source, PetscInt p, PetscInt *rt, PetscInt *Nt, DMPolytopeType *target[], PetscInt *size[], PetscInt *cone[], PetscInt *ornt[])
{
  static DMPolytopeType vertexT[] = {DM_POLYTOPE_POINT};
  static PetscInt       vertexS[] = {1};
  static PetscInt       vertexC[] = {0};
  static PetscInt       vertexO[] = {0};
  static DMPolytopeType edgeT[]   = {DM_POLYTOPE_SEGMENT};
  static PetscInt       edgeS[]   = {1};
  static PetscInt       edgeC[]   = {DM_POLYTOPE_POINT, 1, 0, 0, DM_POLYTOPE_POINT, 1, 1, 0};
  static PetscInt       edgeO[]   = {0, 0};
  static DMPolytopeType tedgeT[]  = {DM_POLYTOPE_POINT_PRISM_TENSOR};
  static PetscInt       tedgeS[]  = {1};
  static PetscInt       tedgeC[]  = {DM_POLYTOPE_POINT, 1, 0, 0, DM_POLYTOPE_POINT, 1, 1, 0};
  static PetscInt       tedgeO[]  = {0, 0};
  static DMPolytopeType triT[]    = {DM_POLYTOPE_TRIANGLE};
  static PetscInt       triS[]    = {1};
  static PetscInt       triC[]    = {DM_POLYTOPE_SEGMENT, 1, 0, 0, DM_POLYTOPE_SEGMENT, 1, 1, 0, DM_POLYTOPE_SEGMENT, 1, 2, 0};
  static PetscInt       triO[]    = {0, 0, 0};
  static DMPolytopeType quadT[]   = {DM_POLYTOPE_QUADRILATERAL};
  static PetscInt       quadS[]   = {1};
  static PetscInt       quadC[]   = {DM_POLYTOPE_SEGMENT, 1, 0, 0, DM_POLYTOPE_SEGMENT, 1, 1, 0, DM_POLYTOPE_SEGMENT, 1, 2, 0, DM_POLYTOPE_SEGMENT, 1, 3, 0};
  static PetscInt       quadO[]   = {0, 0, 0, 0};
  static DMPolytopeType tquadT[]  = {DM_POLYTOPE_SEG_PRISM_TENSOR};
  static PetscInt       tquadS[]  = {1};
  static PetscInt       tquadC[]  = {DM_POLYTOPE_SEGMENT, 1, 0, 0, DM_POLYTOPE_SEGMENT, 1, 1, 0, DM_POLYTOPE_POINT_PRISM_TENSOR, 1, 2, 0, DM_POLYTOPE_POINT_PRISM_TENSOR, 1, 3, 0};
  static PetscInt       tquadO[]  = {0, 0, 0, 0};
  static DMPolytopeType tetT[]    = {DM_POLYTOPE_TETRAHEDRON};
  static PetscInt       tetS[]    = {1};
  static PetscInt       tetC[]    = {DM_POLYTOPE_TRIANGLE, 1, 0, 0, DM_POLYTOPE_TRIANGLE, 1, 1, 0, DM_POLYTOPE_TRIANGLE, 1, 2, 0, DM_POLYTOPE_TRIANGLE, 1, 3, 0};
  static PetscInt       tetO[]    = {0, 0, 0, 0};
  static DMPolytopeType hexT[]    = {DM_POLYTOPE_HEXAHEDRON};
  static PetscInt       hexS[]    = {1};
  static PetscInt       hexC[]    = {DM_POLYTOPE_QUADRILATERAL, 1, 0, 0, DM_POLYTOPE_QUADRILATERAL, 1, 1, 0, DM_POLYTOPE_QUADRILATERAL, 1, 2, 0,
                                     DM_POLYTOPE_QUADRILATERAL, 1, 3, 0, DM_POLYTOPE_QUADRILATERAL, 1, 4, 0, DM_POLYTOPE_QUADRILATERAL, 1, 5, 0};
  static PetscInt       hexO[]    = {0, 0, 0, 0, 0, 0};
  static DMPolytopeType tripT[]   = {DM_POLYTOPE_TRI_PRISM};
  static PetscInt       tripS[]   = {1};
  static PetscInt       tripC[]   = {DM_POLYTOPE_TRIANGLE, 1, 0, 0, DM_POLYTOPE_TRIANGLE, 1, 1, 0,
                                     DM_POLYTOPE_QUADRILATERAL, 1, 2, 0, DM_POLYTOPE_QUADRILATERAL, 1, 3, 0, DM_POLYTOPE_QUADRILATERAL, 1, 4, 0};
  static PetscInt       tripO[]   = {0, 0, 0, 0, 0};
  static DMPolytopeType ttripT[]  = {DM_POLYTOPE_TRI_PRISM_TENSOR};
  static PetscInt       ttripS[]  = {1};
  static PetscInt       ttripC[]  = {DM_POLYTOPE_TRIANGLE, 1, 0, 0, DM_POLYTOPE_TRIANGLE, 1, 1, 0,
                                     DM_POLYTOPE_SEG_PRISM_TENSOR, 1, 2, 0, DM_POLYTOPE_SEG_PRISM_TENSOR, 1, 3, 0, DM_POLYTOPE_SEG_PRISM_TENSOR, 1, 4, 0};
  static PetscInt       ttripO[]  = {0, 0, 0, 0, 0};
  static DMPolytopeType tquadpT[] = {DM_POLYTOPE_QUAD_PRISM_TENSOR};
  static PetscInt       tquadpS[] = {1};
  static PetscInt       tquadpC[] = {DM_POLYTOPE_QUADRILATERAL, 1, 0, 0, DM_POLYTOPE_QUADRILATERAL, 1, 1, 0,
                                     DM_POLYTOPE_SEG_PRISM_TENSOR, 1, 2, 0, DM_POLYTOPE_SEG_PRISM_TENSOR, 1, 3, 0, DM_POLYTOPE_SEG_PRISM_TENSOR, 1, 4, 0, DM_POLYTOPE_SEG_PRISM_TENSOR, 1, 5, 0};
  static PetscInt       tquadpO[] = {0, 0, 0, 0, 0, 0};
  static DMPolytopeType pyrT[]    = {DM_POLYTOPE_PYRAMID};
  static PetscInt       pyrS[]    = {1};
  static PetscInt       pyrC[]    = {DM_POLYTOPE_QUADRILATERAL, 1, 0, 0, DM_POLYTOPE_TRIANGLE, 1, 1, 0,
                                     DM_POLYTOPE_TRIANGLE, 1, 2, 0, DM_POLYTOPE_TRIANGLE, 1, 3, 0, DM_POLYTOPE_TRIANGLE, 1, 4, 0};
  static PetscInt       pyrO[]    = {0, 0, 0, 0, 0};

  PetscFunctionBegin;
  if (rt) *rt = 0;
  switch (source) {
    case DM_POLYTOPE_POINT:              *Nt = 1; *target = vertexT; *size = vertexS; *cone = vertexC; *ornt = vertexO; break;
    case DM_POLYTOPE_SEGMENT:            *Nt = 1; *target = edgeT;   *size = edgeS;   *cone = edgeC;   *ornt = edgeO;   break;
    case DM_POLYTOPE_POINT_PRISM_TENSOR: *Nt = 1; *target = tedgeT;  *size = tedgeS;  *cone = tedgeC;  *ornt = tedgeO;  break;
    case DM_POLYTOPE_TRIANGLE:           *Nt = 1; *target = triT;    *size = triS;    *cone = triC;    *ornt = triO;    break;
    case DM_POLYTOPE_QUADRILATERAL:      *Nt = 1; *target = quadT;   *size = quadS;   *cone = quadC;   *ornt = quadO;   break;
    case DM_POLYTOPE_SEG_PRISM_TENSOR:   *Nt = 1; *target = tquadT;  *size = tquadS;  *cone = tquadC;  *ornt = tquadO;  break;
    case DM_POLYTOPE_TETRAHEDRON:        *Nt = 1; *target = tetT;    *size = tetS;    *cone = tetC;    *ornt = tetO;    break;
    case DM_POLYTOPE_HEXAHEDRON:         *Nt = 1; *target = hexT;    *size = hexS;    *cone = hexC;    *ornt = hexO;    break;
    case DM_POLYTOPE_TRI_PRISM:          *Nt = 1; *target = tripT;   *size = tripS;   *cone = tripC;   *ornt = tripO;   break;
    case DM_POLYTOPE_TRI_PRISM_TENSOR:   *Nt = 1; *target = ttripT;  *size = ttripS;  *cone = ttripC;  *ornt = ttripO;  break;
    case DM_POLYTOPE_QUAD_PRISM_TENSOR:  *Nt = 1; *target = tquadpT; *size = tquadpS; *cone = tquadpC; *ornt = tquadpO; break;
    case DM_POLYTOPE_PYRAMID:            *Nt = 1; *target = pyrT;    *size = pyrS;    *cone = pyrC;    *ornt = pyrO;    break;
    default: SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "No refinement strategy for %s", DMPolytopeTypes[source]);
  }
  PetscFunctionReturn(0);
}

/*@
  DMPlexTransformGetSubcellOrientation - Transform the replica number and orientation for a target point according to the group action for the source point

  Not collective

  Input Parameters:
+ tr  - The DMPlexTransform
. sct - The source point cell type, from whom the new cell is being produced
. sp  - The source point
. so  - The orientation of the source point in its enclosing parent
. tct - The target point cell type
. r   - The replica number requested for the produced cell type
- o   - The orientation of the replica

  Output Parameters:
+ rnew - The replica number, given the orientation of the parent
- onew - The replica orientation, given the orientation of the parent

  Level: advanced

.seealso: DMPlexTransformCellTransform(), DMPlexTransformApply()
@*/
PetscErrorCode DMPlexTransformGetSubcellOrientation(DMPlexTransform tr, DMPolytopeType sct, PetscInt sp, PetscInt so, DMPolytopeType tct, PetscInt r, PetscInt o, PetscInt *rnew, PetscInt *onew)
{
  PetscErrorCode ierr;

  PetscFunctionBeginHot;
  ierr = (*tr->ops->getsubcellorientation)(tr, sct, sp, so, tct, r, o, rnew, onew);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexTransformSetConeSizes(DMPlexTransform tr, DM rdm)
{
  DM              dm;
  PetscInt        pStart, pEnd, p, pNew;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = DMPlexTransformGetDM(tr, &dm);CHKERRQ(ierr);
  /* Must create the celltype label here so that we do not automatically try to compute the types */
  ierr = DMCreateLabel(rdm, "celltype");CHKERRQ(ierr);
  ierr = DMPlexGetChart(dm, &pStart, &pEnd);CHKERRQ(ierr);
  for (p = pStart; p < pEnd; ++p) {
    DMPolytopeType  ct;
    DMPolytopeType *rct;
    PetscInt       *rsize, *rcone, *rornt;
    PetscInt        Nct, n, r;

    ierr = DMPlexGetCellType(dm, p, &ct);CHKERRQ(ierr);
    ierr = DMPlexTransformCellTransform(tr, ct, p, NULL, &Nct, &rct, &rsize, &rcone, &rornt);CHKERRQ(ierr);
    for (n = 0; n < Nct; ++n) {
      for (r = 0; r < rsize[n]; ++r) {
        ierr = DMPlexTransformGetTargetPoint(tr, ct, rct[n], p, r, &pNew);CHKERRQ(ierr);
        ierr = DMPlexSetConeSize(rdm, pNew, DMPolytopeTypeGetConeSize(rct[n]));CHKERRQ(ierr);
        ierr = DMPlexSetCellType(rdm, pNew, rct[n]);CHKERRQ(ierr);
      }
    }
  }
  /* Let the DM know we have set all the cell types */
  {
    DMLabel  ctLabel;
    DM_Plex *plex = (DM_Plex *) rdm->data;

    ierr = DMPlexGetCellTypeLabel(rdm, &ctLabel);CHKERRQ(ierr);
    ierr = PetscObjectStateGet((PetscObject) ctLabel, &plex->celltypeState);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#if 0
static PetscErrorCode DMPlexTransformGetConeSize(DMPlexTransform tr, PetscInt q, PetscInt *coneSize)
{
  PetscInt ctNew;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tr, DMPLEXTRANSFORM_CLASSID, 1);
  PetscValidPointer(coneSize, 3);
  /* TODO Can do bisection since everything is sorted */
  for (ctNew = DM_POLYTOPE_POINT; ctNew < DM_NUM_POLYTOPES; ++ctNew) {
    PetscInt ctSN = tr->ctStartNew[ctNew], ctEN = tr->ctStartNew[tr->ctOrderNew[tr->ctOrderInvNew[ctNew]+1]];

    if (q >= ctSN && q < ctEN) break;
  }
  PetscAssertFalse(ctNew >= DM_NUM_POLYTOPES,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Point %D cannot be located in the transformed mesh", q);
  *coneSize = DMPolytopeTypeGetConeSize((DMPolytopeType) ctNew);
  PetscFunctionReturn(0);
}
#endif

/* The orientation o is for the interior of the cell p */
static PetscErrorCode DMPlexTransformGetCone_Internal(DMPlexTransform tr, PetscInt p, PetscInt o, DMPolytopeType ct, DMPolytopeType ctNew,
                                                      const PetscInt rcone[], PetscInt *coneoff, const PetscInt rornt[], PetscInt *orntoff,
                                                      PetscInt coneNew[], PetscInt orntNew[])
{
  DM              dm;
  const PetscInt  csizeNew = DMPolytopeTypeGetConeSize(ctNew);
  const PetscInt *cone;
  PetscInt        c, coff = *coneoff, ooff = *orntoff;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = DMPlexTransformGetDM(tr, &dm);CHKERRQ(ierr);
  ierr = DMPlexGetCone(dm, p, &cone);CHKERRQ(ierr);
  for (c = 0; c < csizeNew; ++c) {
    PetscInt             ppp   = -1;                             /* Parent Parent point: Parent of point pp */
    PetscInt             pp    = p;                              /* Parent point: Point in the original mesh producing new cone point */
    PetscInt             po    = 0;                              /* Orientation of parent point pp in parent parent point ppp */
    DMPolytopeType       pct   = ct;                             /* Parent type: Cell type for parent of new cone point */
    const PetscInt      *pcone = cone;                           /* Parent cone: Cone of parent point pp */
    PetscInt             pr    = -1;                             /* Replica number of pp that produces new cone point  */
    const DMPolytopeType ft    = (DMPolytopeType) rcone[coff++]; /* Cell type for new cone point of pNew */
    const PetscInt       fn    = rcone[coff++];                  /* Number of cones of p that need to be taken when producing new cone point */
    PetscInt             fo    = rornt[ooff++];                  /* Orientation of new cone point in pNew */
    PetscInt             lc;

    /* Get the type (pct) and point number (pp) of the parent point in the original mesh which produces this cone point */
    for (lc = 0; lc < fn; ++lc) {
      const PetscInt *parr = DMPolytopeTypeGetArrangment(pct, po);
      const PetscInt  acp  = rcone[coff++];
      const PetscInt  pcp  = parr[acp*2];
      const PetscInt  pco  = parr[acp*2+1];
      const PetscInt *ppornt;

      ppp  = pp;
      pp   = pcone[pcp];
      ierr = DMPlexGetCellType(dm, pp, &pct);CHKERRQ(ierr);
      ierr = DMPlexGetCone(dm, pp, &pcone);CHKERRQ(ierr);
      ierr = DMPlexGetConeOrientation(dm, ppp, &ppornt);CHKERRQ(ierr);
      po   = DMPolytopeTypeComposeOrientation(pct, ppornt[pcp], pco);
    }
    pr = rcone[coff++];
    /* Orientation po of pp maps (pr, fo) -> (pr', fo') */
    ierr = DMPlexTransformGetSubcellOrientation(tr, pct, pp, fn ? po : o, ft, pr, fo, &pr, &fo);CHKERRQ(ierr);
    ierr = DMPlexTransformGetTargetPoint(tr, pct, ft, pp, pr, &coneNew[c]);CHKERRQ(ierr);
    orntNew[c] = fo;
  }
  *coneoff = coff;
  *orntoff = ooff;
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexTransformSetCones(DMPlexTransform tr, DM rdm)
{
  DM             dm;
  DMPolytopeType ct;
  PetscInt      *coneNew, *orntNew;
  PetscInt       maxConeSize = 0, pStart, pEnd, p, pNew;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMPlexTransformGetDM(tr, &dm);CHKERRQ(ierr);
  for (p = 0; p < DM_NUM_POLYTOPES; ++p) maxConeSize = PetscMax(maxConeSize, DMPolytopeTypeGetConeSize((DMPolytopeType) p));
  ierr = DMGetWorkArray(rdm, maxConeSize, MPIU_INT, &coneNew);CHKERRQ(ierr);
  ierr = DMGetWorkArray(rdm, maxConeSize, MPIU_INT, &orntNew);CHKERRQ(ierr);
  ierr = DMPlexGetChart(dm, &pStart, &pEnd);CHKERRQ(ierr);
  for (p = pStart; p < pEnd; ++p) {
    const PetscInt *cone, *ornt;
    PetscInt        coff, ooff;
    DMPolytopeType *rct;
    PetscInt       *rsize, *rcone, *rornt;
    PetscInt        Nct, n, r;

    ierr = DMPlexGetCellType(dm, p, &ct);CHKERRQ(ierr);
    ierr = DMPlexGetCone(dm, p, &cone);CHKERRQ(ierr);
    ierr = DMPlexGetConeOrientation(dm, p, &ornt);CHKERRQ(ierr);
    ierr = DMPlexTransformCellTransform(tr, ct, p, NULL, &Nct, &rct, &rsize, &rcone, &rornt);CHKERRQ(ierr);
    for (n = 0, coff = 0, ooff = 0; n < Nct; ++n) {
      const DMPolytopeType ctNew = rct[n];

      for (r = 0; r < rsize[n]; ++r) {
        ierr = DMPlexTransformGetTargetPoint(tr, ct, rct[n], p, r, &pNew);CHKERRQ(ierr);
        ierr = DMPlexTransformGetCone_Internal(tr, p, 0, ct, ctNew, rcone, &coff, rornt, &ooff, coneNew, orntNew);CHKERRQ(ierr);
        ierr = DMPlexSetCone(rdm, pNew, coneNew);CHKERRQ(ierr);
        ierr = DMPlexSetConeOrientation(rdm, pNew, orntNew);CHKERRQ(ierr);
      }
    }
  }
  ierr = DMRestoreWorkArray(rdm, maxConeSize, MPIU_INT, &coneNew);CHKERRQ(ierr);
  ierr = DMRestoreWorkArray(rdm, maxConeSize, MPIU_INT, &orntNew);CHKERRQ(ierr);
  ierr = DMViewFromOptions(rdm, NULL, "-rdm_view");CHKERRQ(ierr);
  ierr = DMPlexSymmetrize(rdm);CHKERRQ(ierr);
  ierr = DMPlexStratify(rdm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexTransformGetConeOriented(DMPlexTransform tr, PetscInt q, PetscInt po, const PetscInt *cone[], const PetscInt *ornt[])
{
  DM              dm;
  DMPolytopeType  ct, qct;
  DMPolytopeType *rct;
  PetscInt       *rsize, *rcone, *rornt, *qcone, *qornt;
  PetscInt        maxConeSize = 0, Nct, p, r, n, nr, coff = 0, ooff = 0;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tr, DMPLEXTRANSFORM_CLASSID, 1);
  PetscValidPointer(cone, 4);
  PetscValidPointer(ornt, 5);
  for (p = 0; p < DM_NUM_POLYTOPES; ++p) maxConeSize = PetscMax(maxConeSize, DMPolytopeTypeGetConeSize((DMPolytopeType) p));
  ierr = DMPlexTransformGetDM(tr, &dm);CHKERRQ(ierr);
  ierr = DMGetWorkArray(dm, maxConeSize, MPIU_INT, &qcone);CHKERRQ(ierr);
  ierr = DMGetWorkArray(dm, maxConeSize, MPIU_INT, &qornt);CHKERRQ(ierr);
  ierr = DMPlexTransformGetSourcePoint(tr, q, &ct, &qct, &p, &r);CHKERRQ(ierr);
  ierr = DMPlexTransformCellTransform(tr, ct, p, NULL, &Nct, &rct, &rsize, &rcone, &rornt);CHKERRQ(ierr);
  for (n = 0; n < Nct; ++n) {
    const DMPolytopeType ctNew    = rct[n];
    const PetscInt       csizeNew = DMPolytopeTypeGetConeSize(ctNew);
    PetscInt             Nr = rsize[n], fn, c;

    if (ctNew == qct) Nr = r;
    for (nr = 0; nr < Nr; ++nr) {
      for (c = 0; c < csizeNew; ++c) {
        ++coff;             /* Cell type of new cone point */
        fn = rcone[coff++]; /* Number of cones of p that need to be taken when producing new cone point */
        coff += fn;
        ++coff;             /* Replica number of new cone point */
        ++ooff;             /* Orientation of new cone point */
      }
    }
    if (ctNew == qct) break;
  }
  ierr = DMPlexTransformGetCone_Internal(tr, p, po, ct, qct, rcone, &coff, rornt, &ooff, qcone, qornt);CHKERRQ(ierr);
  *cone = qcone;
  *ornt = qornt;
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexTransformGetCone(DMPlexTransform tr, PetscInt q, const PetscInt *cone[], const PetscInt *ornt[])
{
  DM              dm;
  DMPolytopeType  ct, qct;
  DMPolytopeType *rct;
  PetscInt       *rsize, *rcone, *rornt, *qcone, *qornt;
  PetscInt        maxConeSize = 0, Nct, p, r, n, nr, coff = 0, ooff = 0;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tr, DMPLEXTRANSFORM_CLASSID, 1);
  PetscValidPointer(cone, 3);
  for (p = 0; p < DM_NUM_POLYTOPES; ++p) maxConeSize = PetscMax(maxConeSize, DMPolytopeTypeGetConeSize((DMPolytopeType) p));
  ierr = DMPlexTransformGetDM(tr, &dm);CHKERRQ(ierr);
  ierr = DMGetWorkArray(dm, maxConeSize, MPIU_INT, &qcone);CHKERRQ(ierr);
  ierr = DMGetWorkArray(dm, maxConeSize, MPIU_INT, &qornt);CHKERRQ(ierr);
  ierr = DMPlexTransformGetSourcePoint(tr, q, &ct, &qct, &p, &r);CHKERRQ(ierr);
  ierr = DMPlexTransformCellTransform(tr, ct, p, NULL, &Nct, &rct, &rsize, &rcone, &rornt);CHKERRQ(ierr);
  for (n = 0; n < Nct; ++n) {
    const DMPolytopeType ctNew    = rct[n];
    const PetscInt       csizeNew = DMPolytopeTypeGetConeSize(ctNew);
    PetscInt             Nr = rsize[n], fn, c;

    if (ctNew == qct) Nr = r;
    for (nr = 0; nr < Nr; ++nr) {
      for (c = 0; c < csizeNew; ++c) {
        ++coff;             /* Cell type of new cone point */
        fn = rcone[coff++]; /* Number of cones of p that need to be taken when producing new cone point */
        coff += fn;
        ++coff;             /* Replica number of new cone point */
        ++ooff;             /* Orientation of new cone point */
      }
    }
    if (ctNew == qct) break;
  }
  ierr = DMPlexTransformGetCone_Internal(tr, p, 0, ct, qct, rcone, &coff, rornt, &ooff, qcone, qornt);CHKERRQ(ierr);
  *cone = qcone;
  *ornt = qornt;
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexTransformRestoreCone(DMPlexTransform tr, PetscInt q, const PetscInt *cone[], const PetscInt *ornt[])
{
  DM             dm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tr, DMPLEXTRANSFORM_CLASSID, 1);
  PetscValidPointer(cone, 3);
  ierr = DMPlexTransformGetDM(tr, &dm);CHKERRQ(ierr);
  ierr = DMRestoreWorkArray(dm, 0, MPIU_INT, cone);CHKERRQ(ierr);
  ierr = DMRestoreWorkArray(dm, 0, MPIU_INT, ornt);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexTransformCreateCellVertices_Internal(DMPlexTransform tr)
{
  PetscInt       ict;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscCalloc3(DM_NUM_POLYTOPES, &tr->trNv, DM_NUM_POLYTOPES, &tr->trVerts, DM_NUM_POLYTOPES, &tr->trSubVerts);CHKERRQ(ierr);
  for (ict = DM_POLYTOPE_POINT; ict < DM_NUM_POLYTOPES; ++ict) {
    const DMPolytopeType ct = (DMPolytopeType) ict;
    DMPlexTransform    reftr;
    DM                 refdm, trdm;
    Vec                coordinates;
    const PetscScalar *coords;
    DMPolytopeType    *rct;
    PetscInt          *rsize, *rcone, *rornt;
    PetscInt           Nct, n, r, pNew;
    PetscInt           vStart, vEnd, Nc;
    const PetscInt     debug = 0;
    const char        *typeName;

    /* Since points are 0-dimensional, coordinates make no sense */
    if (DMPolytopeTypeGetDim(ct) <= 0) continue;
    ierr = DMPlexCreateReferenceCell(PETSC_COMM_SELF, ct, &refdm);CHKERRQ(ierr);
    ierr = DMPlexTransformCreate(PETSC_COMM_SELF, &reftr);CHKERRQ(ierr);
    ierr = DMPlexTransformSetDM(reftr, refdm);CHKERRQ(ierr);
    ierr = DMPlexTransformGetType(tr, &typeName);CHKERRQ(ierr);
    ierr = DMPlexTransformSetType(reftr, typeName);CHKERRQ(ierr);
    ierr = DMPlexTransformSetUp(reftr);CHKERRQ(ierr);
    ierr = DMPlexTransformApply(reftr, refdm, &trdm);CHKERRQ(ierr);

    ierr = DMPlexGetDepthStratum(trdm, 0, &vStart, &vEnd);CHKERRQ(ierr);
    tr->trNv[ct] = vEnd - vStart;
    ierr = DMGetCoordinatesLocal(trdm, &coordinates);CHKERRQ(ierr);
    ierr = VecGetLocalSize(coordinates, &Nc);CHKERRQ(ierr);
    PetscAssertFalse(tr->trNv[ct] * DMPolytopeTypeGetDim(ct) != Nc,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Cell type %s, transformed coordinate size %D != %D size of coordinate storage", DMPolytopeTypes[ct], tr->trNv[ct] * DMPolytopeTypeGetDim(ct), Nc);
    ierr = PetscCalloc1(Nc, &tr->trVerts[ct]);CHKERRQ(ierr);
    ierr = VecGetArrayRead(coordinates, &coords);CHKERRQ(ierr);
    ierr = PetscArraycpy(tr->trVerts[ct], coords, Nc);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(coordinates, &coords);CHKERRQ(ierr);

    ierr = PetscCalloc1(DM_NUM_POLYTOPES, &tr->trSubVerts[ct]);CHKERRQ(ierr);
    ierr = DMPlexTransformCellTransform(reftr, ct, 0, NULL, &Nct, &rct, &rsize, &rcone, &rornt);CHKERRQ(ierr);
    for (n = 0; n < Nct; ++n) {

      /* Since points are 0-dimensional, coordinates make no sense */
      if (rct[n] == DM_POLYTOPE_POINT) continue;
      ierr = PetscCalloc1(rsize[n], &tr->trSubVerts[ct][rct[n]]);CHKERRQ(ierr);
      for (r = 0; r < rsize[n]; ++r) {
        PetscInt *closure = NULL;
        PetscInt  clSize, cl, Nv = 0;

        ierr = PetscCalloc1(DMPolytopeTypeGetNumVertices(rct[n]), &tr->trSubVerts[ct][rct[n]][r]);CHKERRQ(ierr);
        ierr = DMPlexTransformGetTargetPoint(reftr, ct, rct[n], 0, r, &pNew);CHKERRQ(ierr);
        ierr = DMPlexGetTransitiveClosure(trdm, pNew, PETSC_TRUE, &clSize, &closure);CHKERRQ(ierr);
        for (cl = 0; cl < clSize*2; cl += 2) {
          const PetscInt sv = closure[cl];

          if ((sv >= vStart) && (sv < vEnd)) tr->trSubVerts[ct][rct[n]][r][Nv++] = sv - vStart;
        }
        ierr = DMPlexRestoreTransitiveClosure(trdm, pNew, PETSC_TRUE, &clSize, &closure);CHKERRQ(ierr);
        PetscAssertFalse(Nv != DMPolytopeTypeGetNumVertices(rct[n]),PETSC_COMM_SELF, PETSC_ERR_PLIB, "Number of vertices %D != %D for %s subcell %D from cell %s", Nv, DMPolytopeTypeGetNumVertices(rct[n]), DMPolytopeTypes[rct[n]], r, DMPolytopeTypes[ct]);
      }
    }
    if (debug) {
      DMPolytopeType *rct;
      PetscInt       *rsize, *rcone, *rornt;
      PetscInt        v, dE = DMPolytopeTypeGetDim(ct), d, off = 0;

      ierr = PetscPrintf(PETSC_COMM_SELF, "%s: %D vertices\n", DMPolytopeTypes[ct], tr->trNv[ct]);CHKERRQ(ierr);
      for (v = 0; v < tr->trNv[ct]; ++v) {
        ierr = PetscPrintf(PETSC_COMM_SELF, "  ");CHKERRQ(ierr);
        for (d = 0; d < dE; ++d) {ierr = PetscPrintf(PETSC_COMM_SELF, "%g ", tr->trVerts[ct][off++]);CHKERRQ(ierr);}
        ierr = PetscPrintf(PETSC_COMM_SELF, "\n");CHKERRQ(ierr);
      }

      ierr = DMPlexTransformCellTransform(reftr, ct, 0, NULL, &Nct, &rct, &rsize, &rcone, &rornt);CHKERRQ(ierr);
      for (n = 0; n < Nct; ++n) {
        if (rct[n] == DM_POLYTOPE_POINT) continue;
        ierr = PetscPrintf(PETSC_COMM_SELF, "%s: %s subvertices\n", DMPolytopeTypes[ct], DMPolytopeTypes[rct[n]], tr->trNv[ct]);CHKERRQ(ierr);
        for (r = 0; r < rsize[n]; ++r) {
          ierr = PetscPrintf(PETSC_COMM_SELF, "  ");CHKERRQ(ierr);
          for (v = 0; v < DMPolytopeTypeGetNumVertices(rct[n]); ++v) {
            ierr = PetscPrintf(PETSC_COMM_SELF, "%D ", tr->trSubVerts[ct][rct[n]][r][v]);CHKERRQ(ierr);
          }
          ierr = PetscPrintf(PETSC_COMM_SELF, "\n");CHKERRQ(ierr);
        }
      }
    }
    ierr = DMDestroy(&refdm);CHKERRQ(ierr);
    ierr = DMDestroy(&trdm);CHKERRQ(ierr);
    ierr = DMPlexTransformDestroy(&reftr);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*
  DMPlexTransformGetCellVertices - Get the set of transformed vertices lying in the closure of a reference cell of given type

  Input Parameters:
+ tr - The DMPlexTransform object
- ct - The cell type

  Output Parameters:
+ Nv      - The number of transformed vertices in the closure of the reference cell of given type
- trVerts - The coordinates of these vertices in the reference cell

  Level: developer

.seealso: DMPlexTransformGetSubcellVertices()
*/
PetscErrorCode DMPlexTransformGetCellVertices(DMPlexTransform tr, DMPolytopeType ct, PetscInt *Nv, PetscScalar *trVerts[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!tr->trNv) {ierr = DMPlexTransformCreateCellVertices_Internal(tr);CHKERRQ(ierr);}
  if (Nv)      *Nv      = tr->trNv[ct];
  if (trVerts) *trVerts = tr->trVerts[ct];
  PetscFunctionReturn(0);
}

/*
  DMPlexTransformGetSubcellVertices - Get the set of transformed vertices defining a subcell in the reference cell of given type

  Input Parameters:
+ tr  - The DMPlexTransform object
. ct  - The cell type
. rct - The subcell type
- r   - The subcell index

  Output Parameter:
. subVerts - The indices of these vertices in the set of vertices returned by DMPlexTransformGetCellVertices()

  Level: developer

.seealso: DMPlexTransformGetCellVertices()
*/
PetscErrorCode DMPlexTransformGetSubcellVertices(DMPlexTransform tr, DMPolytopeType ct, DMPolytopeType rct, PetscInt r, PetscInt *subVerts[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!tr->trNv) {ierr = DMPlexTransformCreateCellVertices_Internal(tr);CHKERRQ(ierr);}
  PetscAssertFalse(!tr->trSubVerts[ct][rct],PetscObjectComm((PetscObject) tr), PETSC_ERR_ARG_WRONG, "Cell type %s does not produce %s", DMPolytopeTypes[ct], DMPolytopeTypes[rct]);
  if (subVerts) *subVerts = tr->trSubVerts[ct][rct][r];
  PetscFunctionReturn(0);
}

/* Computes new vertex as the barycenter, or centroid */
PetscErrorCode DMPlexTransformMapCoordinatesBarycenter_Internal(DMPlexTransform tr, DMPolytopeType pct, DMPolytopeType ct, PetscInt p, PetscInt r, PetscInt Nv, PetscInt dE, const PetscScalar in[], PetscScalar out[])
{
  PetscInt v,d;

  PetscFunctionBeginHot;
  PetscAssertFalse(ct != DM_POLYTOPE_POINT,PETSC_COMM_SELF,PETSC_ERR_SUP,"Not for refined point type %s",DMPolytopeTypes[ct]);
  for (d = 0; d < dE; ++d) out[d] = 0.0;
  for (v = 0; v < Nv; ++v) for (d = 0; d < dE; ++d) out[d] += in[v*dE+d];
  for (d = 0; d < dE; ++d) out[d] /= Nv;
  PetscFunctionReturn(0);
}

/*@
  DMPlexTransformMapCoordinates -
  Input Parameters:
+ cr   - The DMPlexCellRefiner
. pct  - The cell type of the parent, from whom the new cell is being produced
. ct   - The type being produced
. p    - The original point
. r    - The replica number requested for the produced cell type
. Nv   - Number of vertices in the closure of the parent cell
. dE   - Spatial dimension
- in   - array of size Nv*dE, holding coordinates of the vertices in the closure of the parent cell

  Output Parameters:
. out - The coordinates of the new vertices

  Level: intermediate
@*/
PetscErrorCode DMPlexTransformMapCoordinates(DMPlexTransform tr, DMPolytopeType pct, DMPolytopeType ct, PetscInt p, PetscInt r, PetscInt Nv, PetscInt dE, const PetscScalar in[], PetscScalar out[])
{
  PetscErrorCode ierr;

  PetscFunctionBeginHot;
  ierr = (*tr->ops->mapcoordinates)(tr, pct, ct, p, r, Nv, dE, in, out);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode RefineLabel_Internal(DMPlexTransform tr, DMLabel label, DMLabel labelNew)
{
  DM              dm;
  IS              valueIS;
  const PetscInt *values;
  PetscInt        defVal, Nv, val;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = DMPlexTransformGetDM(tr, &dm);CHKERRQ(ierr);
  ierr = DMLabelGetDefaultValue(label, &defVal);CHKERRQ(ierr);
  ierr = DMLabelSetDefaultValue(labelNew, defVal);CHKERRQ(ierr);
  ierr = DMLabelGetValueIS(label, &valueIS);CHKERRQ(ierr);
  ierr = ISGetLocalSize(valueIS, &Nv);CHKERRQ(ierr);
  ierr = ISGetIndices(valueIS, &values);CHKERRQ(ierr);
  for (val = 0; val < Nv; ++val) {
    IS              pointIS;
    const PetscInt *points;
    PetscInt        numPoints, p;

    /* Ensure refined label is created with same number of strata as
     * original (even if no entries here). */
    ierr = DMLabelAddStratum(labelNew, values[val]);CHKERRQ(ierr);
    ierr = DMLabelGetStratumIS(label, values[val], &pointIS);CHKERRQ(ierr);
    ierr = ISGetLocalSize(pointIS, &numPoints);CHKERRQ(ierr);
    ierr = ISGetIndices(pointIS, &points);CHKERRQ(ierr);
    for (p = 0; p < numPoints; ++p) {
      const PetscInt  point = points[p];
      DMPolytopeType  ct;
      DMPolytopeType *rct;
      PetscInt       *rsize, *rcone, *rornt;
      PetscInt        Nct, n, r, pNew=0;

      ierr = DMPlexGetCellType(dm, point, &ct);CHKERRQ(ierr);
      ierr = DMPlexTransformCellTransform(tr, ct, point, NULL, &Nct, &rct, &rsize, &rcone, &rornt);CHKERRQ(ierr);
      for (n = 0; n < Nct; ++n) {
        for (r = 0; r < rsize[n]; ++r) {
          ierr = DMPlexTransformGetTargetPoint(tr, ct, rct[n], point, r, &pNew);CHKERRQ(ierr);
          ierr = DMLabelSetValue(labelNew, pNew, values[val]);CHKERRQ(ierr);
        }
      }
    }
    ierr = ISRestoreIndices(pointIS, &points);CHKERRQ(ierr);
    ierr = ISDestroy(&pointIS);CHKERRQ(ierr);
  }
  ierr = ISRestoreIndices(valueIS, &values);CHKERRQ(ierr);
  ierr = ISDestroy(&valueIS);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexTransformCreateLabels(DMPlexTransform tr, DM rdm)
{
  DM             dm;
  PetscInt       numLabels, l;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMPlexTransformGetDM(tr, &dm);CHKERRQ(ierr);
  ierr = DMGetNumLabels(dm, &numLabels);CHKERRQ(ierr);
  for (l = 0; l < numLabels; ++l) {
    DMLabel         label, labelNew;
    const char     *lname;
    PetscBool       isDepth, isCellType;

    ierr = DMGetLabelName(dm, l, &lname);CHKERRQ(ierr);
    ierr = PetscStrcmp(lname, "depth", &isDepth);CHKERRQ(ierr);
    if (isDepth) continue;
    ierr = PetscStrcmp(lname, "celltype", &isCellType);CHKERRQ(ierr);
    if (isCellType) continue;
    ierr = DMCreateLabel(rdm, lname);CHKERRQ(ierr);
    ierr = DMGetLabel(dm, lname, &label);CHKERRQ(ierr);
    ierr = DMGetLabel(rdm, lname, &labelNew);CHKERRQ(ierr);
    ierr = RefineLabel_Internal(tr, label, labelNew);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* This refines the labels which define regions for fields and DSes since they are not in the list of labels for the DM */
PetscErrorCode DMPlexTransformCreateDiscLabels(DMPlexTransform tr, DM rdm)
{
  DM             dm;
  PetscInt       Nf, f, Nds, s;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMPlexTransformGetDM(tr, &dm);CHKERRQ(ierr);
  ierr = DMGetNumFields(dm, &Nf);CHKERRQ(ierr);
  for (f = 0; f < Nf; ++f) {
    DMLabel     label, labelNew;
    PetscObject obj;
    const char *lname;

    ierr = DMGetField(rdm, f, &label, &obj);CHKERRQ(ierr);
    if (!label) continue;
    ierr = PetscObjectGetName((PetscObject) label, &lname);CHKERRQ(ierr);
    ierr = DMLabelCreate(PETSC_COMM_SELF, lname, &labelNew);CHKERRQ(ierr);
    ierr = RefineLabel_Internal(tr, label, labelNew);CHKERRQ(ierr);
    ierr = DMSetField_Internal(rdm, f, labelNew, obj);CHKERRQ(ierr);
    ierr = DMLabelDestroy(&labelNew);CHKERRQ(ierr);
  }
  ierr = DMGetNumDS(dm, &Nds);CHKERRQ(ierr);
  for (s = 0; s < Nds; ++s) {
    DMLabel     label, labelNew;
    const char *lname;

    ierr = DMGetRegionNumDS(rdm, s, &label, NULL, NULL);CHKERRQ(ierr);
    if (!label) continue;
    ierr = PetscObjectGetName((PetscObject) label, &lname);CHKERRQ(ierr);
    ierr = DMLabelCreate(PETSC_COMM_SELF, lname, &labelNew);CHKERRQ(ierr);
    ierr = RefineLabel_Internal(tr, label, labelNew);CHKERRQ(ierr);
    ierr = DMSetRegionNumDS(rdm, s, labelNew, NULL, NULL);CHKERRQ(ierr);
    ierr = DMLabelDestroy(&labelNew);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexTransformCreateSF(DMPlexTransform tr, DM rdm)
{
  DM                 dm;
  PetscSF            sf, sfNew;
  PetscInt           numRoots, numLeaves, numLeavesNew = 0, l, m;
  const PetscInt    *localPoints;
  const PetscSFNode *remotePoints;
  PetscInt          *localPointsNew;
  PetscSFNode       *remotePointsNew;
  PetscInt           pStartNew, pEndNew, pNew;
  /* Brute force algorithm */
  PetscSF            rsf;
  PetscSection       s;
  const PetscInt    *rootdegree;
  PetscInt          *rootPointsNew, *remoteOffsets;
  PetscInt           numPointsNew, pStart, pEnd, p;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = DMPlexTransformGetDM(tr, &dm);CHKERRQ(ierr);
  ierr = DMPlexGetChart(rdm, &pStartNew, &pEndNew);CHKERRQ(ierr);
  ierr = DMGetPointSF(dm, &sf);CHKERRQ(ierr);
  ierr = DMGetPointSF(rdm, &sfNew);CHKERRQ(ierr);
  /* Calculate size of new SF */
  ierr = PetscSFGetGraph(sf, &numRoots, &numLeaves, &localPoints, &remotePoints);CHKERRQ(ierr);
  if (numRoots < 0) PetscFunctionReturn(0);
  for (l = 0; l < numLeaves; ++l) {
    const PetscInt  p = localPoints[l];
    DMPolytopeType  ct;
    DMPolytopeType *rct;
    PetscInt       *rsize, *rcone, *rornt;
    PetscInt        Nct, n;

    ierr = DMPlexGetCellType(dm, p, &ct);CHKERRQ(ierr);
    ierr = DMPlexTransformCellTransform(tr, ct, p, NULL, &Nct, &rct, &rsize, &rcone, &rornt);CHKERRQ(ierr);
    for (n = 0; n < Nct; ++n) {
      numLeavesNew += rsize[n];
    }
  }
  /* Send new root point numbers
       It is possible to optimize for regular transforms by sending only the cell type offsets, but it seems a needless complication
  */
  ierr = DMPlexGetChart(dm, &pStart, &pEnd);CHKERRQ(ierr);
  ierr = PetscSectionCreate(PetscObjectComm((PetscObject) dm), &s);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(s, pStart, pEnd);CHKERRQ(ierr);
  for (p = pStart; p < pEnd; ++p) {
    DMPolytopeType  ct;
    DMPolytopeType *rct;
    PetscInt       *rsize, *rcone, *rornt;
    PetscInt        Nct, n;

    ierr = DMPlexGetCellType(dm, p, &ct);CHKERRQ(ierr);
    ierr = DMPlexTransformCellTransform(tr, ct, p, NULL, &Nct, &rct, &rsize, &rcone, &rornt);CHKERRQ(ierr);
    for (n = 0; n < Nct; ++n) {
      ierr = PetscSectionAddDof(s, p, rsize[n]);CHKERRQ(ierr);
    }
  }
  ierr = PetscSectionSetUp(s);CHKERRQ(ierr);
  ierr = PetscSectionGetStorageSize(s, &numPointsNew);CHKERRQ(ierr);
  ierr = PetscSFCreateRemoteOffsets(sf, s, s, &remoteOffsets);CHKERRQ(ierr);
  ierr = PetscSFCreateSectionSF(sf, s, remoteOffsets, s, &rsf);CHKERRQ(ierr);
  ierr = PetscFree(remoteOffsets);CHKERRQ(ierr);
  ierr = PetscSFComputeDegreeBegin(sf, &rootdegree);CHKERRQ(ierr);
  ierr = PetscSFComputeDegreeEnd(sf, &rootdegree);CHKERRQ(ierr);
  ierr = PetscMalloc1(numPointsNew, &rootPointsNew);CHKERRQ(ierr);
  for (p = 0; p < numPointsNew; ++p) rootPointsNew[p] = -1;
  for (p = pStart; p < pEnd; ++p) {
    DMPolytopeType  ct;
    DMPolytopeType *rct;
    PetscInt       *rsize, *rcone, *rornt;
    PetscInt        Nct, n, r, off;

    if (!rootdegree[p-pStart]) continue;
    ierr = PetscSectionGetOffset(s, p, &off);CHKERRQ(ierr);
    ierr = DMPlexGetCellType(dm, p, &ct);CHKERRQ(ierr);
    ierr = DMPlexTransformCellTransform(tr, ct, p, NULL, &Nct, &rct, &rsize, &rcone, &rornt);CHKERRQ(ierr);
    for (n = 0, m = 0; n < Nct; ++n) {
      for (r = 0; r < rsize[n]; ++r, ++m) {
        ierr = DMPlexTransformGetTargetPoint(tr, ct, rct[n], p, r, &pNew);CHKERRQ(ierr);
        rootPointsNew[off+m] = pNew;
      }
    }
  }
  ierr = PetscSFBcastBegin(rsf, MPIU_INT, rootPointsNew, rootPointsNew,MPI_REPLACE);CHKERRQ(ierr);
  ierr = PetscSFBcastEnd(rsf, MPIU_INT, rootPointsNew, rootPointsNew,MPI_REPLACE);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&rsf);CHKERRQ(ierr);
  ierr = PetscMalloc1(numLeavesNew, &localPointsNew);CHKERRQ(ierr);
  ierr = PetscMalloc1(numLeavesNew, &remotePointsNew);CHKERRQ(ierr);
  for (l = 0, m = 0; l < numLeaves; ++l) {
    const PetscInt  p = localPoints[l];
    DMPolytopeType  ct;
    DMPolytopeType *rct;
    PetscInt       *rsize, *rcone, *rornt;
    PetscInt        Nct, n, r, q, off;

    ierr = PetscSectionGetOffset(s, p, &off);CHKERRQ(ierr);
    ierr = DMPlexGetCellType(dm, p, &ct);CHKERRQ(ierr);
    ierr = DMPlexTransformCellTransform(tr, ct, p, NULL, &Nct, &rct, &rsize, &rcone, &rornt);CHKERRQ(ierr);
    for (n = 0, q = 0; n < Nct; ++n) {
      for (r = 0; r < rsize[n]; ++r, ++m, ++q) {
        ierr = DMPlexTransformGetTargetPoint(tr, ct, rct[n], p, r, &pNew);CHKERRQ(ierr);
        localPointsNew[m]        = pNew;
        remotePointsNew[m].index = rootPointsNew[off+q];
        remotePointsNew[m].rank  = remotePoints[l].rank;
      }
    }
  }
  ierr = PetscSectionDestroy(&s);CHKERRQ(ierr);
  ierr = PetscFree(rootPointsNew);CHKERRQ(ierr);
  /* SF needs sorted leaves to correctly calculate Gather */
  {
    PetscSFNode *rp, *rtmp;
    PetscInt    *lp, *idx, *ltmp, i;

    ierr = PetscMalloc1(numLeavesNew, &idx);CHKERRQ(ierr);
    ierr = PetscMalloc1(numLeavesNew, &lp);CHKERRQ(ierr);
    ierr = PetscMalloc1(numLeavesNew, &rp);CHKERRQ(ierr);
    for (i = 0; i < numLeavesNew; ++i) {
      PetscAssertFalse((localPointsNew[i] < pStartNew) || (localPointsNew[i] >= pEndNew),PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Local SF point %D (%D) not in [%D, %D)", localPointsNew[i], i, pStartNew, pEndNew);
      idx[i] = i;
    }
    ierr = PetscSortIntWithPermutation(numLeavesNew, localPointsNew, idx);CHKERRQ(ierr);
    for (i = 0; i < numLeavesNew; ++i) {
      lp[i] = localPointsNew[idx[i]];
      rp[i] = remotePointsNew[idx[i]];
    }
    ltmp            = localPointsNew;
    localPointsNew  = lp;
    rtmp            = remotePointsNew;
    remotePointsNew = rp;
    ierr = PetscFree(idx);CHKERRQ(ierr);
    ierr = PetscFree(ltmp);CHKERRQ(ierr);
    ierr = PetscFree(rtmp);CHKERRQ(ierr);
  }
  ierr = PetscSFSetGraph(sfNew, pEndNew-pStartNew, numLeavesNew, localPointsNew, PETSC_OWN_POINTER, remotePointsNew, PETSC_OWN_POINTER);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
  DMPlexCellRefinerMapLocalizedCoordinates - Given a cell of type ct with localized coordinates x, we generate localized coordinates xr for subcell r of type rct.

  Not collective

  Input Parameters:
+ cr  - The DMPlexCellRefiner
. ct  - The type of the parent cell
. rct - The type of the produced cell
. r   - The index of the produced cell
- x   - The localized coordinates for the parent cell

  Output Parameter:
. xr  - The localized coordinates for the produced cell

  Level: developer

.seealso: DMPlexCellRefinerSetCoordinates()
*/
static PetscErrorCode DMPlexTransformMapLocalizedCoordinates(DMPlexTransform tr, DMPolytopeType ct, DMPolytopeType rct, PetscInt r, const PetscScalar x[], PetscScalar xr[])
{
  PetscFE        fe = NULL;
  PetscInt       cdim, v, *subcellV;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMPlexTransformGetCoordinateFE(tr, ct, &fe);CHKERRQ(ierr);
  ierr = DMPlexTransformGetSubcellVertices(tr, ct, rct, r, &subcellV);CHKERRQ(ierr);
  ierr = PetscFEGetNumComponents(fe, &cdim);CHKERRQ(ierr);
  for (v = 0; v < DMPolytopeTypeGetNumVertices(rct); ++v) {
    ierr = PetscFEInterpolate_Static(fe, x, tr->refGeom[ct], subcellV[v], &xr[v*cdim]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexTransformSetCoordinates(DMPlexTransform tr, DM rdm)
{
  DM                    dm, cdm;
  PetscSection          coordSection, coordSectionNew;
  Vec                   coordsLocal, coordsLocalNew;
  const PetscScalar    *coords;
  PetscScalar          *coordsNew;
  const DMBoundaryType *bd;
  const PetscReal      *maxCell, *L;
  PetscBool             isperiodic, localizeVertices = PETSC_FALSE, localizeCells = PETSC_FALSE;
  PetscInt              dE, dEo, d, cStart, cEnd, c, vStartNew, vEndNew, v, pStart, pEnd, p, ocStart, ocEnd;
  PetscErrorCode        ierr;

  PetscFunctionBegin;
  ierr = DMPlexTransformGetDM(tr, &dm);CHKERRQ(ierr);
  ierr = DMGetCoordinateDM(dm, &cdm);CHKERRQ(ierr);
  ierr = DMGetPeriodicity(dm, &isperiodic, &maxCell, &L, &bd);CHKERRQ(ierr);
  /* Determine if we need to localize coordinates when generating them */
  if (isperiodic) {
    localizeVertices = PETSC_TRUE;
    if (!maxCell) {
      PetscBool localized;
      ierr = DMGetCoordinatesLocalized(dm, &localized);CHKERRQ(ierr);
      PetscAssertFalse(!localized,PetscObjectComm((PetscObject) dm), PETSC_ERR_USER, "Cannot refine a periodic mesh if coordinates have not been localized");
      localizeCells = PETSC_TRUE;
    }
  }

  ierr = DMGetCoordinateSection(dm, &coordSection);CHKERRQ(ierr);
  ierr = PetscSectionGetFieldComponents(coordSection, 0, &dEo);CHKERRQ(ierr);
  if (maxCell) {
    PetscReal maxCellNew[3];

    for (d = 0; d < dEo; ++d) maxCellNew[d] = maxCell[d]/2.0;
    ierr = DMSetPeriodicity(rdm, isperiodic, maxCellNew, L, bd);CHKERRQ(ierr);
  } else {
    ierr = DMSetPeriodicity(rdm, isperiodic, maxCell, L, bd);CHKERRQ(ierr);
  }
  ierr = DMGetCoordinateDim(rdm, &dE);CHKERRQ(ierr);
  ierr = PetscSectionCreate(PetscObjectComm((PetscObject) rdm), &coordSectionNew);CHKERRQ(ierr);
  ierr = PetscSectionSetNumFields(coordSectionNew, 1);CHKERRQ(ierr);
  ierr = PetscSectionSetFieldComponents(coordSectionNew, 0, dE);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(rdm, 0, &vStartNew, &vEndNew);CHKERRQ(ierr);
  if (localizeCells) {ierr = PetscSectionSetChart(coordSectionNew, 0,         vEndNew);CHKERRQ(ierr);}
  else               {ierr = PetscSectionSetChart(coordSectionNew, vStartNew, vEndNew);CHKERRQ(ierr);}

  /* Localization should be inherited */
  /*   Stefano calculates parent cells for each new cell for localization */
  /*   Localized cells need coordinates of closure */
  for (v = vStartNew; v < vEndNew; ++v) {
    ierr = PetscSectionSetDof(coordSectionNew, v, dE);CHKERRQ(ierr);
    ierr = PetscSectionSetFieldDof(coordSectionNew, v, 0, dE);CHKERRQ(ierr);
  }
  if (localizeCells) {
    ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
    for (c = cStart; c < cEnd; ++c) {
      PetscInt dof;

      ierr = PetscSectionGetDof(coordSection, c, &dof);CHKERRQ(ierr);
      if (dof) {
        DMPolytopeType  ct;
        DMPolytopeType *rct;
        PetscInt       *rsize, *rcone, *rornt;
        PetscInt        dim, cNew, Nct, n, r;

        ierr = DMPlexGetCellType(dm, c, &ct);CHKERRQ(ierr);
        dim  = DMPolytopeTypeGetDim(ct);
        ierr = DMPlexTransformCellTransform(tr, ct, c, NULL, &Nct, &rct, &rsize, &rcone, &rornt);CHKERRQ(ierr);
        /* This allows for different cell types */
        for (n = 0; n < Nct; ++n) {
          if (dim != DMPolytopeTypeGetDim(rct[n])) continue;
          for (r = 0; r < rsize[n]; ++r) {
            PetscInt *closure = NULL;
            PetscInt  clSize, cl, Nv = 0;

            ierr = DMPlexTransformGetTargetPoint(tr, ct, rct[n], c, r, &cNew);CHKERRQ(ierr);
            ierr = DMPlexGetTransitiveClosure(rdm, cNew, PETSC_TRUE, &clSize, &closure);CHKERRQ(ierr);
            for (cl = 0; cl < clSize*2; cl += 2) {if ((closure[cl] >= vStartNew) && (closure[cl] < vEndNew)) ++Nv;}
            ierr = DMPlexRestoreTransitiveClosure(rdm, cNew, PETSC_TRUE, &clSize, &closure);CHKERRQ(ierr);
            ierr = PetscSectionSetDof(coordSectionNew, cNew, Nv * dE);CHKERRQ(ierr);
            ierr = PetscSectionSetFieldDof(coordSectionNew, cNew, 0, Nv * dE);CHKERRQ(ierr);
          }
        }
      }
    }
  }
  ierr = PetscSectionSetUp(coordSectionNew);CHKERRQ(ierr);
  ierr = DMViewFromOptions(dm, NULL, "-coarse_dm_view");CHKERRQ(ierr);
  ierr = DMSetCoordinateSection(rdm, PETSC_DETERMINE, coordSectionNew);CHKERRQ(ierr);
  {
    VecType     vtype;
    PetscInt    coordSizeNew, bs;
    const char *name;

    ierr = DMGetCoordinatesLocal(dm, &coordsLocal);CHKERRQ(ierr);
    ierr = VecCreate(PETSC_COMM_SELF, &coordsLocalNew);CHKERRQ(ierr);
    ierr = PetscSectionGetStorageSize(coordSectionNew, &coordSizeNew);CHKERRQ(ierr);
    ierr = VecSetSizes(coordsLocalNew, coordSizeNew, PETSC_DETERMINE);CHKERRQ(ierr);
    ierr = PetscObjectGetName((PetscObject) coordsLocal, &name);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) coordsLocalNew, name);CHKERRQ(ierr);
    ierr = VecGetBlockSize(coordsLocal, &bs);CHKERRQ(ierr);
    ierr = VecSetBlockSize(coordsLocalNew, dEo == dE ? bs : dE);CHKERRQ(ierr);
    ierr = VecGetType(coordsLocal, &vtype);CHKERRQ(ierr);
    ierr = VecSetType(coordsLocalNew, vtype);CHKERRQ(ierr);
  }
  ierr = VecGetArrayRead(coordsLocal, &coords);CHKERRQ(ierr);
  ierr = VecGetArray(coordsLocalNew, &coordsNew);CHKERRQ(ierr);
  ierr = PetscSectionGetChart(coordSection, &ocStart, &ocEnd);CHKERRQ(ierr);
  ierr = DMPlexGetChart(dm, &pStart, &pEnd);CHKERRQ(ierr);
  /* First set coordinates for vertices*/
  for (p = pStart; p < pEnd; ++p) {
    DMPolytopeType  ct;
    DMPolytopeType *rct;
    PetscInt       *rsize, *rcone, *rornt;
    PetscInt        Nct, n, r;
    PetscBool       hasVertex = PETSC_FALSE, isLocalized = PETSC_FALSE;

    ierr = DMPlexGetCellType(dm, p, &ct);CHKERRQ(ierr);
    ierr = DMPlexTransformCellTransform(tr, ct, p, NULL, &Nct, &rct, &rsize, &rcone, &rornt);CHKERRQ(ierr);
    for (n = 0; n < Nct; ++n) {
      if (rct[n] == DM_POLYTOPE_POINT) {hasVertex = PETSC_TRUE; break;}
    }
    if (localizeVertices && ct != DM_POLYTOPE_POINT && (p >= ocStart) && (p < ocEnd)) {
      PetscInt dof;
      ierr = PetscSectionGetDof(coordSection, p, &dof);CHKERRQ(ierr);
      if (dof) isLocalized = PETSC_TRUE;
    }
    if (hasVertex) {
      const PetscScalar *icoords = NULL;
      PetscScalar       *pcoords = NULL;
      PetscInt          Nc, Nv, v, d;

      ierr = DMPlexVecGetClosure(dm, coordSection, coordsLocal, p, &Nc, &pcoords);CHKERRQ(ierr);

      icoords = pcoords;
      Nv      = Nc/dEo;
      if (ct != DM_POLYTOPE_POINT) {
        if (localizeVertices) {
          PetscScalar anchor[3];

          for (d = 0; d < dEo; ++d) anchor[d] = pcoords[d];
          if (!isLocalized) {
            for (v = 0; v < Nv; ++v) {ierr = DMLocalizeCoordinate_Internal(dm, dEo, anchor, &pcoords[v*dEo], &pcoords[v*dEo]);CHKERRQ(ierr);}
          } else {
            Nv = Nc/(2*dEo);
            for (v = Nv; v < Nv*2; ++v) {ierr = DMLocalizeCoordinate_Internal(dm, dEo, anchor, &pcoords[v*dEo], &pcoords[v*dEo]);CHKERRQ(ierr);}
          }
        }
      }
      for (n = 0; n < Nct; ++n) {
        if (rct[n] != DM_POLYTOPE_POINT) continue;
        for (r = 0; r < rsize[n]; ++r) {
          PetscScalar vcoords[3];
          PetscInt    vNew, off;

          ierr = DMPlexTransformGetTargetPoint(tr, ct, rct[n], p, r, &vNew);CHKERRQ(ierr);
          ierr = PetscSectionGetOffset(coordSectionNew, vNew, &off);CHKERRQ(ierr);
          ierr = DMPlexTransformMapCoordinates(tr, ct, rct[n], p, r, Nv, dEo, icoords, vcoords);CHKERRQ(ierr);
          ierr = DMPlexSnapToGeomModel(dm, p, dE, vcoords, &coordsNew[off]);CHKERRQ(ierr);
        }
      }
      ierr = DMPlexVecRestoreClosure(dm, coordSection, coordsLocal, p, &Nc, &pcoords);CHKERRQ(ierr);
    }
  }
  /* Then set coordinates for cells by localizing */
  for (p = pStart; p < pEnd; ++p) {
    DMPolytopeType  ct;
    DMPolytopeType *rct;
    PetscInt       *rsize, *rcone, *rornt;
    PetscInt        Nct, n, r;
    PetscBool       isLocalized = PETSC_FALSE;

    ierr = DMPlexGetCellType(dm, p, &ct);CHKERRQ(ierr);
    ierr = DMPlexTransformCellTransform(tr, ct, p, NULL, &Nct, &rct, &rsize, &rcone, &rornt);CHKERRQ(ierr);
    if (localizeCells && ct != DM_POLYTOPE_POINT && (p >= ocStart) && (p < ocEnd)) {
      PetscInt dof;
      ierr = PetscSectionGetDof(coordSection, p, &dof);CHKERRQ(ierr);
      if (dof) isLocalized = PETSC_TRUE;
    }
    if (isLocalized) {
      const PetscScalar *pcoords;

      ierr = DMPlexPointLocalRead(cdm, p, coords, &pcoords);CHKERRQ(ierr);
      for (n = 0; n < Nct; ++n) {
        const PetscInt Nr = rsize[n];

        if (DMPolytopeTypeGetDim(ct) != DMPolytopeTypeGetDim(rct[n])) continue;
        for (r = 0; r < Nr; ++r) {
          PetscInt pNew, offNew;

          /* It looks like Stefano and Lisandro are allowing localized coordinates without defining the periodic boundary, which means that
             DMLocalizeCoordinate_Internal() will not work. Localized coordinates will have to have obtained by the affine map of the larger
             cell to the ones it produces. */
          ierr = DMPlexTransformGetTargetPoint(tr, ct, rct[n], p, r, &pNew);CHKERRQ(ierr);
          ierr = PetscSectionGetOffset(coordSectionNew, pNew, &offNew);CHKERRQ(ierr);
          ierr = DMPlexTransformMapLocalizedCoordinates(tr, ct, rct[n], r, pcoords, &coordsNew[offNew]);CHKERRQ(ierr);
        }
      }
    }
  }
  ierr = VecRestoreArrayRead(coordsLocal, &coords);CHKERRQ(ierr);
  ierr = VecRestoreArray(coordsLocalNew, &coordsNew);CHKERRQ(ierr);
  ierr = DMSetCoordinatesLocal(rdm, coordsLocalNew);CHKERRQ(ierr);
  /* TODO Stefano has a final reduction if some hybrid coordinates cannot be found. (needcoords) Should not be needed. */
  ierr = VecDestroy(&coordsLocalNew);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&coordSectionNew);CHKERRQ(ierr);
  if (!localizeCells) {ierr = DMLocalizeCoordinates(rdm);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexTransformApply(DMPlexTransform tr, DM dm, DM *tdm)
{
  DM                     rdm;
  DMPlexInterpolatedFlag interp;
  PetscErrorCode         ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tr, DMPLEXTRANSFORM_CLASSID, 1);
  PetscValidHeaderSpecific(dm, DM_CLASSID, 2);
  PetscValidPointer(tdm, 3);
  ierr = DMPlexTransformSetDM(tr, dm);CHKERRQ(ierr);

  ierr = DMCreate(PetscObjectComm((PetscObject)dm), &rdm);CHKERRQ(ierr);
  ierr = DMSetType(rdm, DMPLEX);CHKERRQ(ierr);
  ierr = DMPlexTransformSetDimensions(tr, dm, rdm);CHKERRQ(ierr);
  /* Calculate number of new points of each depth */
  ierr = DMPlexIsInterpolated(dm, &interp);CHKERRQ(ierr);
  PetscAssertFalse(interp != DMPLEX_INTERPOLATED_FULL,PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_WRONG, "Mesh must be fully interpolated for regular refinement");
  /* Step 1: Set chart */
  ierr = DMPlexSetChart(rdm, 0, tr->ctStartNew[tr->ctOrderNew[DM_NUM_POLYTOPES]]);CHKERRQ(ierr);
  /* Step 2: Set cone/support sizes (automatically stratifies) */
  ierr = DMPlexTransformSetConeSizes(tr, rdm);CHKERRQ(ierr);
  /* Step 3: Setup refined DM */
  ierr = DMSetUp(rdm);CHKERRQ(ierr);
  /* Step 4: Set cones and supports (automatically symmetrizes) */
  ierr = DMPlexTransformSetCones(tr, rdm);CHKERRQ(ierr);
  /* Step 5: Create pointSF */
  ierr = DMPlexTransformCreateSF(tr, rdm);CHKERRQ(ierr);
  /* Step 6: Create labels */
  ierr = DMPlexTransformCreateLabels(tr, rdm);CHKERRQ(ierr);
  /* Step 7: Set coordinates */
  ierr = DMPlexTransformSetCoordinates(tr, rdm);CHKERRQ(ierr);
  *tdm = rdm;
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexTransformAdaptLabel(DM dm, PETSC_UNUSED Vec metric, DMLabel adaptLabel, PETSC_UNUSED DMLabel rgLabel, DM *rdm)
{
  DMPlexTransform tr;
  DM              cdm, rcdm;
  const char     *prefix;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = DMPlexTransformCreate(PetscObjectComm((PetscObject) dm), &tr);CHKERRQ(ierr);
  ierr = PetscObjectGetOptionsPrefix((PetscObject) dm, &prefix);CHKERRQ(ierr);
  ierr = PetscObjectSetOptionsPrefix((PetscObject) tr,  prefix);CHKERRQ(ierr);
  ierr = DMPlexTransformSetDM(tr, dm);CHKERRQ(ierr);
  ierr = DMPlexTransformSetFromOptions(tr);CHKERRQ(ierr);
  ierr = DMPlexTransformSetActive(tr, adaptLabel);CHKERRQ(ierr);
  ierr = DMPlexTransformSetUp(tr);CHKERRQ(ierr);
  ierr = PetscObjectViewFromOptions((PetscObject) tr, NULL, "-dm_plex_transform_view");CHKERRQ(ierr);
  ierr = DMPlexTransformApply(tr, dm, rdm);CHKERRQ(ierr);
  ierr = DMCopyDisc(dm, *rdm);CHKERRQ(ierr);
  ierr = DMGetCoordinateDM(dm, &cdm);CHKERRQ(ierr);
  ierr = DMGetCoordinateDM(*rdm, &rcdm);CHKERRQ(ierr);
  ierr = DMCopyDisc(cdm, rcdm);CHKERRQ(ierr);
  ierr = DMPlexTransformCreateDiscLabels(tr, *rdm);CHKERRQ(ierr);
  ierr = DMCopyDisc(dm, *rdm);CHKERRQ(ierr);
  ierr = DMPlexTransformDestroy(&tr);CHKERRQ(ierr);
  ((DM_Plex *) (*rdm)->data)->useHashLocation = ((DM_Plex *) dm->data)->useHashLocation;
  PetscFunctionReturn(0);
}
