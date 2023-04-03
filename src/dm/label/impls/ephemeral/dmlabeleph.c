#include <petsc/private/dmlabelimpl.h> /*I      "petscdmlabelephemeral.h"   I*/
#include <petscdmlabelephemeral.h>

/*
  An emphemeral label is read-only

  This initial implementation makes the assumption that the produced points have unique parents. If this is
  not satisfied, hash tables can be introduced to ensure produced points are unique.
*/

static PetscErrorCode DMLabelEphemeralComputeStratumSize_Private(DMLabel label, PetscInt value)
{
  DMPlexTransform tr;
  DM              dm;
  DMLabel         olabel;
  IS              opointIS;
  const PetscInt *opoints;
  PetscInt        Np = 0, Nop, op, v;

  PetscFunctionBegin;
  PetscCall(DMLabelEphemeralGetTransform(label, &tr));
  PetscCall(DMLabelEphemeralGetLabel(label, &olabel));
  PetscCall(DMPlexTransformGetDM(tr, &dm));

  PetscCall(DMLabelLookupStratum(olabel, value, &v));
  PetscCall(DMLabelGetStratumIS(olabel, value, &opointIS));
  PetscCall(ISGetLocalSize(opointIS, &Nop));
  PetscCall(ISGetIndices(opointIS, &opoints));
  for (op = 0; op < Nop; ++op) {
    const PetscInt  point = opoints[op];
    DMPolytopeType  ct;
    DMPolytopeType *rct;
    PetscInt       *rsize, *rcone, *rornt;
    PetscInt        Nct;

    PetscCall(DMPlexGetCellType(dm, point, &ct));
    PetscCall(DMPlexTransformCellTransform(tr, ct, point, NULL, &Nct, &rct, &rsize, &rcone, &rornt));
    for (PetscInt n = 0; n < Nct; ++n) Np += rsize[n];
  }
  PetscCall(ISRestoreIndices(opointIS, &opoints));
  PetscCall(ISDestroy(&opointIS));
  label->stratumSizes[v] = Np;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMLabelGetStratumIS_Ephemeral(DMLabel label, PetscInt v, IS *stratum)
{
  DMPlexTransform tr;
  DM              dm;
  DMLabel         olabel;
  IS              opointIS;
  const PetscInt *opoints;
  PetscInt       *points;
  PetscInt        Np, p, Nop, op;

  PetscFunctionBegin;
  PetscCall(DMLabelEphemeralGetTransform(label, &tr));
  PetscCall(DMLabelEphemeralGetLabel(label, &olabel));
  PetscCall(DMPlexTransformGetDM(tr, &dm));

  PetscCall(DMLabelGetStratumSize_Private(label, v, &Np));
  PetscCall(PetscMalloc1(Np, &points));
  PetscUseTypeMethod(olabel, getstratumis, v, &opointIS);
  PetscCall(ISGetLocalSize(opointIS, &Nop));
  PetscCall(ISGetIndices(opointIS, &opoints));
  for (op = 0, p = 0; op < Nop; ++op) {
    const PetscInt  point = opoints[op];
    DMPolytopeType  ct;
    DMPolytopeType *rct;
    PetscInt       *rsize, *rcone, *rornt;
    PetscInt        Nct, n, r, pNew = 0;

    PetscCall(DMPlexGetCellType(dm, point, &ct));
    PetscCall(DMPlexTransformCellTransform(tr, ct, point, NULL, &Nct, &rct, &rsize, &rcone, &rornt));
    for (n = 0; n < Nct; ++n) {
      for (r = 0; r < rsize[n]; ++r) {
        PetscCall(DMPlexTransformGetTargetPoint(tr, ct, rct[n], point, r, &pNew));
        points[p++] = pNew;
      }
    }
  }
  PetscCheck(p == Np, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Number of stratum points %" PetscInt_FMT " != %" PetscInt_FMT " precomputed size", p, Np);
  PetscCall(ISRestoreIndices(opointIS, &opoints));
  PetscCall(ISDestroy(&opointIS));
  PetscCall(ISCreateGeneral(PETSC_COMM_SELF, Np, points, PETSC_OWN_POINTER, stratum));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMLabelSetUp_Ephemeral(DMLabel label)
{
  DMLabel         olabel;
  IS              valueIS;
  const PetscInt *values;
  PetscInt        defValue, Nv;

  PetscFunctionBegin;
  PetscCall(DMLabelEphemeralGetLabel(label, &olabel));
  PetscCall(DMLabelGetDefaultValue(olabel, &defValue));
  PetscCall(DMLabelSetDefaultValue(label, defValue));
  PetscCall(DMLabelGetNumValues(olabel, &Nv));
  PetscCall(DMLabelGetValueIS(olabel, &valueIS));
  PetscCall(ISGetIndices(valueIS, &values));
  PetscCall(DMLabelAddStrataIS(label, valueIS));
  for (PetscInt v = 0; v < Nv; ++v) PetscCall(DMLabelEphemeralComputeStratumSize_Private(label, values[v]));
  PetscCall(ISRestoreIndices(valueIS, &values));
  PetscCall(ISDestroy(&valueIS));
  label->readonly = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMLabelView_Ephemeral_Ascii(DMLabel label, PetscViewer viewer)
{
  DMLabel     olabel;
  PetscMPIInt rank;

  PetscFunctionBegin;
  PetscCall(DMLabelEphemeralGetLabel(label, &olabel));
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)viewer), &rank));
  PetscCall(PetscViewerASCIIPushSynchronized(viewer));
  if (olabel) {
    IS              valueIS;
    const PetscInt *values;
    PetscInt        Nv, v;
    const char     *name;

    PetscCall(PetscObjectGetName((PetscObject)label, &name));
    PetscCall(PetscViewerASCIIPrintf(viewer, "Ephemeral Label '%s':\n", name));
    PetscCall(DMLabelGetNumValues(olabel, &Nv));
    PetscCall(DMLabelGetValueIS(olabel, &valueIS));
    PetscCall(ISGetIndices(valueIS, &values));
    for (v = 0; v < Nv; ++v) {
      IS              pointIS;
      const PetscInt  value = values[v];
      const PetscInt *points;
      PetscInt        n, p;

      PetscCall(DMLabelGetStratumIS(olabel, value, &pointIS));
      PetscCall(ISGetIndices(pointIS, &points));
      PetscCall(ISGetLocalSize(pointIS, &n));
      for (p = 0; p < n; ++p) PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "[%d]: %" PetscInt_FMT " (%" PetscInt_FMT ")\n", rank, points[p], value));
      PetscCall(ISRestoreIndices(pointIS, &points));
      PetscCall(ISDestroy(&pointIS));
    }
    PetscCall(ISRestoreIndices(valueIS, &values));
    PetscCall(ISDestroy(&valueIS));
  }
  PetscCall(PetscViewerFlush(viewer));
  PetscCall(PetscViewerASCIIPopSynchronized(viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMLabelView_Ephemeral(DMLabel label, PetscViewer viewer)
{
  PetscBool iascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));
  if (iascii) PetscCall(DMLabelView_Ephemeral_Ascii(label, viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMLabelDuplicate_Ephemeral(DMLabel label, DMLabel *labelnew)
{
  PetscObject olabel;

  PetscFunctionBegin;
  PetscCall(PetscObjectQuery((PetscObject)label, "__original_label__", &olabel));
  PetscCall(PetscObjectCompose((PetscObject)*labelnew, "__original_label__", olabel));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMLabelInitialize_Ephemeral(DMLabel label)
{
  PetscFunctionBegin;
  label->ops->view         = DMLabelView_Ephemeral;
  label->ops->setup        = DMLabelSetUp_Ephemeral;
  label->ops->duplicate    = DMLabelDuplicate_Ephemeral;
  label->ops->getstratumis = DMLabelGetStratumIS_Ephemeral;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
 DMLABELEPHEMERAL = "ephemeral" - This type of `DMLabel` is used to label ephemeral meshes.

 Ephemeral meshes are never concretely instantiated, but rather the answers to queries are created on the fly from a base mesh and a `DMPlexTransform` object.
 For example, we could integrate over a refined mesh without ever storing the entire thing, just producing each cell closure one at a time. The label consists
 of a base label and the same transform. Stratum are produced on demand, so that the time is in $O(max(M, N))$ where $M$ is the size of the original stratum,
 and $N$ is the size of the output stratum. Queries take the same time, since we cannot sort the output.

  Level: intermediate

.seealso: `DMLabel`, `DM`, `DMLabelType`, `DMLabelCreate()`, `DMLabelSetType()`
M*/

PETSC_EXTERN PetscErrorCode DMLabelCreate_Ephemeral(DMLabel label)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 1);
  PetscCall(DMLabelInitialize_Ephemeral(label));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMLabelEphemeralGetLabel - Get the base label for this ephemeral label

  Not Collective

  Input Parameter:
. label - the `DMLabel`

  Output Parameter:
. olabel - the base label for this ephemeral label

  Level: intermediate

  Note:
  Ephemeral labels are produced automatically from a base label and a `DMPlexTransform`.

.seealso: `DMLabelEphemeralSetLabel()`, `DMLabelEphemeralGetTransform()`, `DMLabelSetType()`
@*/
PetscErrorCode DMLabelEphemeralGetLabel(DMLabel label, DMLabel *olabel)
{
  PetscFunctionBegin;
  PetscCall(PetscObjectQuery((PetscObject)label, "__original_label__", (PetscObject *)olabel));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMLabelEphemeralSetLabel - Set the base label for this ephemeral label

  Not Collective

  Input Parameters:
+ label - the `DMLabel`
- olabel - the base label for this ephemeral label

  Level: intermediate

  Note:
  Ephemeral labels are produced automatically from a base label and a `DMPlexTransform`.

.seealso: `DMLabelEphemeralGetLabel()`, `DMLabelEphemeralSetTransform()`, `DMLabelSetType()`
@*/
PetscErrorCode DMLabelEphemeralSetLabel(DMLabel label, DMLabel olabel)
{
  PetscFunctionBegin;
  PetscCall(PetscObjectCompose((PetscObject)label, "__original_label__", (PetscObject)olabel));
  PetscFunctionReturn(PETSC_SUCCESS);
}
