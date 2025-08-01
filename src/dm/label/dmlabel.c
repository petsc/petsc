#include <petscdm.h>
#include <petsc/private/dmlabelimpl.h> /*I      "petscdmlabel.h"   I*/
#include <petsc/private/sectionimpl.h> /*I      "petscsection.h"   I*/
#include <petscsf.h>
#include <petscsection.h>

PetscFunctionList DMLabelList              = NULL;
PetscBool         DMLabelRegisterAllCalled = PETSC_FALSE;

/*@
  DMLabelCreate - Create a `DMLabel` object, which is a multimap

  Collective

  Input Parameters:
+ comm - The communicator, usually `PETSC_COMM_SELF`
- name - The label name

  Output Parameter:
. label - The `DMLabel`

  Level: beginner

  Notes:
  The label name is actually usually the `PetscObject` name.
  One can get/set it with `PetscObjectGetName()`/`PetscObjectSetName()`.

.seealso: `DMLabel`, `DM`, `DMLabelDestroy()`
@*/
PetscErrorCode DMLabelCreate(MPI_Comm comm, const char name[], DMLabel *label)
{
  PetscFunctionBegin;
  PetscAssertPointer(label, 3);
  PetscCall(DMInitializePackage());

  PetscCall(PetscHeaderCreate(*label, DMLABEL_CLASSID, "DMLabel", "DMLabel", "DM", comm, DMLabelDestroy, DMLabelView));
  (*label)->numStrata     = 0;
  (*label)->defaultValue  = -1;
  (*label)->stratumValues = NULL;
  (*label)->validIS       = NULL;
  (*label)->stratumSizes  = NULL;
  (*label)->points        = NULL;
  (*label)->ht            = NULL;
  (*label)->pStart        = -1;
  (*label)->pEnd          = -1;
  (*label)->bt            = NULL;
  PetscCall(PetscHMapICreate(&(*label)->hmap));
  PetscCall(PetscObjectSetName((PetscObject)*label, name));
  PetscCall(DMLabelSetType(*label, DMLABELCONCRETE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMLabelSetUp - SetUp a `DMLabel` object

  Collective

  Input Parameters:
. label - The `DMLabel`

  Level: intermediate

.seealso: `DMLabel`, `DM`, `DMLabelCreate()`, `DMLabelDestroy()`
@*/
PetscErrorCode DMLabelSetUp(DMLabel label)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 1);
  PetscTryTypeMethod(label, setup);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  DMLabelMakeValid_Private - Transfer stratum data from the hash format to the sorted list format

  Not collective

  Input parameter:
+ label - The `DMLabel`
- v - The stratum value

  Output parameter:
. label - The `DMLabel` with stratum in sorted list format

  Level: developer

.seealso: `DMLabel`, `DM`, `DMLabelCreate()`
*/
static PetscErrorCode DMLabelMakeValid_Private(DMLabel label, PetscInt v)
{
  IS       is;
  PetscInt off = 0, *pointArray, p;

  PetscFunctionBegin;
  if ((PetscLikely(v >= 0 && v < label->numStrata) && label->validIS[v]) || label->readonly) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCheck(v >= 0 && v < label->numStrata, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Trying to access invalid stratum %" PetscInt_FMT " in DMLabelMakeValid_Private", v);
  PetscCall(PetscHSetIGetSize(label->ht[v], &label->stratumSizes[v]));
  PetscCall(PetscMalloc1(label->stratumSizes[v], &pointArray));
  PetscCall(PetscHSetIGetElems(label->ht[v], &off, pointArray));
  PetscCall(PetscHSetIClear(label->ht[v]));
  PetscCall(PetscSortInt(label->stratumSizes[v], pointArray));
  if (label->bt) {
    for (p = 0; p < label->stratumSizes[v]; ++p) {
      const PetscInt point = pointArray[p];
      PetscCheck(!(point < label->pStart) && !(point >= label->pEnd), PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Label point %" PetscInt_FMT " is not in [%" PetscInt_FMT ", %" PetscInt_FMT ")", point, label->pStart, label->pEnd);
      PetscCall(PetscBTSet(label->bt, point - label->pStart));
    }
  }
  if (label->stratumSizes[v] > 0 && pointArray[label->stratumSizes[v] - 1] == pointArray[0] + label->stratumSizes[v] - 1) {
    PetscCall(ISCreateStride(PETSC_COMM_SELF, label->stratumSizes[v], pointArray[0], 1, &is));
    PetscCall(PetscFree(pointArray));
  } else {
    PetscCall(ISCreateGeneral(PETSC_COMM_SELF, label->stratumSizes[v], pointArray, PETSC_OWN_POINTER, &is));
  }
  PetscCall(ISSetInfo(is, IS_SORTED, IS_LOCAL, PETSC_TRUE, PETSC_TRUE));
  PetscCall(PetscObjectSetName((PetscObject)is, "indices"));
  label->points[v]  = is;
  label->validIS[v] = PETSC_TRUE;
  PetscCall(PetscObjectStateIncrease((PetscObject)label));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  DMLabelMakeAllValid_Private - Transfer all strata from the hash format to the sorted list format

  Not Collective

  Input parameter:
. label - The `DMLabel`

  Output parameter:
. label - The `DMLabel` with all strata in sorted list format

  Level: developer

.seealso: `DMLabel`, `DM`, `DMLabelCreate()`
*/
static PetscErrorCode DMLabelMakeAllValid_Private(DMLabel label)
{
  PetscInt v;

  PetscFunctionBegin;
  for (v = 0; v < label->numStrata; v++) PetscCall(DMLabelMakeValid_Private(label, v));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  DMLabelMakeInvalid_Private - Transfer stratum data from the sorted list format to the hash format

  Not Collective

  Input parameter:
+ label - The `DMLabel`
- v - The stratum value

  Output parameter:
. label - The `DMLabel` with stratum in hash format

  Level: developer

.seealso: `DMLabel`, `DM`, `DMLabelCreate()`
*/
static PetscErrorCode DMLabelMakeInvalid_Private(DMLabel label, PetscInt v)
{
  PetscInt        p;
  const PetscInt *points;

  PetscFunctionBegin;
  if ((PetscLikely(v >= 0 && v < label->numStrata) && !label->validIS[v]) || label->readonly) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCheck(v >= 0 && v < label->numStrata, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Trying to access invalid stratum %" PetscInt_FMT " in DMLabelMakeInvalid_Private", v);
  if (label->points[v]) {
    PetscCall(ISGetIndices(label->points[v], &points));
    for (p = 0; p < label->stratumSizes[v]; ++p) PetscCall(PetscHSetIAdd(label->ht[v], points[p]));
    PetscCall(ISRestoreIndices(label->points[v], &points));
    PetscCall(ISDestroy(&label->points[v]));
  }
  label->validIS[v] = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMLabelMakeAllInvalid_Internal(DMLabel label)
{
  PetscInt v;

  PetscFunctionBegin;
  for (v = 0; v < label->numStrata; v++) PetscCall(DMLabelMakeInvalid_Private(label, v));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#if !defined(DMLABEL_LOOKUP_THRESHOLD)
  #define DMLABEL_LOOKUP_THRESHOLD 16
#endif

PetscErrorCode DMLabelLookupStratum(DMLabel label, PetscInt value, PetscInt *index)
{
  PetscInt v;

  PetscFunctionBegin;
  *index = -1;
  if (label->numStrata <= DMLABEL_LOOKUP_THRESHOLD || label->readonly) {
    for (v = 0; v < label->numStrata; ++v)
      if (label->stratumValues[v] == value) {
        *index = v;
        break;
      }
  } else {
    PetscCall(PetscHMapIGet(label->hmap, value, index));
  }
  if (PetscDefined(USE_DEBUG) && !label->readonly) { /* Check strata hash map consistency */
    PetscInt len, loc = -1;
    PetscCall(PetscHMapIGetSize(label->hmap, &len));
    PetscCheck(len == label->numStrata, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Inconsistent strata hash map size");
    if (label->numStrata <= DMLABEL_LOOKUP_THRESHOLD) {
      PetscCall(PetscHMapIGet(label->hmap, value, &loc));
    } else {
      for (v = 0; v < label->numStrata; ++v)
        if (label->stratumValues[v] == value) {
          loc = v;
          break;
        }
    }
    PetscCheck(loc == *index, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Inconsistent strata hash map lookup");
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static inline PetscErrorCode DMLabelNewStratum(DMLabel label, PetscInt value, PetscInt *index)
{
  PetscInt    v;
  PetscInt   *tmpV;
  PetscInt   *tmpS;
  PetscHSetI *tmpH, ht;
  IS         *tmpP, is;
  PetscBool  *tmpB;
  PetscHMapI  hmap = label->hmap;

  PetscFunctionBegin;
  v    = label->numStrata;
  tmpV = label->stratumValues;
  tmpS = label->stratumSizes;
  tmpH = label->ht;
  tmpP = label->points;
  tmpB = label->validIS;
  { /* TODO: PetscRealloc() is broken, use malloc+memcpy+free  */
    PetscInt   *oldV = tmpV;
    PetscInt   *oldS = tmpS;
    PetscHSetI *oldH = tmpH;
    IS         *oldP = tmpP;
    PetscBool  *oldB = tmpB;
    PetscCall(PetscMalloc((v + 1) * sizeof(*tmpV), &tmpV));
    PetscCall(PetscMalloc((v + 1) * sizeof(*tmpS), &tmpS));
    PetscCall(PetscCalloc((v + 1) * sizeof(*tmpH), &tmpH));
    PetscCall(PetscCalloc((v + 1) * sizeof(*tmpP), &tmpP));
    PetscCall(PetscMalloc((v + 1) * sizeof(*tmpB), &tmpB));
    PetscCall(PetscArraycpy(tmpV, oldV, v));
    PetscCall(PetscArraycpy(tmpS, oldS, v));
    PetscCall(PetscArraycpy(tmpH, oldH, v));
    PetscCall(PetscArraycpy(tmpP, oldP, v));
    PetscCall(PetscArraycpy(tmpB, oldB, v));
    PetscCall(PetscFree(oldV));
    PetscCall(PetscFree(oldS));
    PetscCall(PetscFree(oldH));
    PetscCall(PetscFree(oldP));
    PetscCall(PetscFree(oldB));
  }
  label->numStrata     = v + 1;
  label->stratumValues = tmpV;
  label->stratumSizes  = tmpS;
  label->ht            = tmpH;
  label->points        = tmpP;
  label->validIS       = tmpB;
  PetscCall(PetscHSetICreate(&ht));
  PetscCall(ISCreateStride(PETSC_COMM_SELF, 0, 0, 1, &is));
  PetscCall(PetscHMapISet(hmap, value, v));
  tmpV[v] = value;
  tmpS[v] = 0;
  tmpH[v] = ht;
  tmpP[v] = is;
  tmpB[v] = PETSC_TRUE;
  PetscCall(PetscObjectStateIncrease((PetscObject)label));
  *index = v;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static inline PetscErrorCode DMLabelLookupAddStratum(DMLabel label, PetscInt value, PetscInt *index)
{
  PetscFunctionBegin;
  PetscCall(DMLabelLookupStratum(label, value, index));
  if (*index < 0) PetscCall(DMLabelNewStratum(label, value, index));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMLabelGetStratumSize_Private(DMLabel label, PetscInt v, PetscInt *size)
{
  PetscFunctionBegin;
  *size = 0;
  if (v < 0) PetscFunctionReturn(PETSC_SUCCESS);
  if (label->readonly || label->validIS[v]) {
    *size = label->stratumSizes[v];
  } else {
    PetscCall(PetscHSetIGetSize(label->ht[v], size));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMLabelAddStratum - Adds a new stratum value in a `DMLabel`

  Input Parameters:
+ label - The `DMLabel`
- value - The stratum value

  Level: beginner

.seealso: `DMLabel`, `DM`, `DMLabelCreate()`, `DMLabelDestroy()`
@*/
PetscErrorCode DMLabelAddStratum(DMLabel label, PetscInt value)
{
  PetscInt v;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 1);
  PetscCheck(!label->readonly, PetscObjectComm((PetscObject)label), PETSC_ERR_ARG_WRONG, "Read-only labels cannot be altered");
  PetscCall(DMLabelLookupAddStratum(label, value, &v));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMLabelAddStrata - Adds new stratum values in a `DMLabel`

  Not Collective

  Input Parameters:
+ label         - The `DMLabel`
. numStrata     - The number of stratum values
- stratumValues - The stratum values

  Level: beginner

.seealso: `DMLabel`, `DM`, `DMLabelCreate()`, `DMLabelDestroy()`
@*/
PetscErrorCode DMLabelAddStrata(DMLabel label, PetscInt numStrata, const PetscInt stratumValues[])
{
  PetscInt *values, v;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 1);
  if (numStrata) PetscAssertPointer(stratumValues, 3);
  PetscCheck(!label->readonly, PetscObjectComm((PetscObject)label), PETSC_ERR_ARG_WRONG, "Read-only labels cannot be altered");
  PetscCall(PetscMalloc1(numStrata, &values));
  PetscCall(PetscArraycpy(values, stratumValues, numStrata));
  PetscCall(PetscSortRemoveDupsInt(&numStrata, values));
  if (!label->numStrata) { /* Fast preallocation */
    PetscInt   *tmpV;
    PetscInt   *tmpS;
    PetscHSetI *tmpH, ht;
    IS         *tmpP, is;
    PetscBool  *tmpB;
    PetscHMapI  hmap = label->hmap;

    PetscCall(PetscMalloc1(numStrata, &tmpV));
    PetscCall(PetscMalloc1(numStrata, &tmpS));
    PetscCall(PetscCalloc1(numStrata, &tmpH));
    PetscCall(PetscCalloc1(numStrata, &tmpP));
    PetscCall(PetscMalloc1(numStrata, &tmpB));
    label->numStrata     = numStrata;
    label->stratumValues = tmpV;
    label->stratumSizes  = tmpS;
    label->ht            = tmpH;
    label->points        = tmpP;
    label->validIS       = tmpB;
    for (v = 0; v < numStrata; ++v) {
      PetscCall(PetscHSetICreate(&ht));
      PetscCall(ISCreateStride(PETSC_COMM_SELF, 0, 0, 1, &is));
      PetscCall(PetscHMapISet(hmap, values[v], v));
      tmpV[v] = values[v];
      tmpS[v] = 0;
      tmpH[v] = ht;
      tmpP[v] = is;
      tmpB[v] = PETSC_TRUE;
    }
    PetscCall(PetscObjectStateIncrease((PetscObject)label));
  } else {
    for (v = 0; v < numStrata; ++v) PetscCall(DMLabelAddStratum(label, values[v]));
  }
  PetscCall(PetscFree(values));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMLabelAddStrataIS - Adds new stratum values in a `DMLabel`

  Not Collective

  Input Parameters:
+ label   - The `DMLabel`
- valueIS - Index set with stratum values

  Level: beginner

.seealso: `DMLabel`, `DM`, `DMLabelCreate()`, `DMLabelDestroy()`
@*/
PetscErrorCode DMLabelAddStrataIS(DMLabel label, IS valueIS)
{
  PetscInt        numStrata;
  const PetscInt *stratumValues;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 1);
  PetscValidHeaderSpecific(valueIS, IS_CLASSID, 2);
  PetscCheck(!label->readonly, PetscObjectComm((PetscObject)label), PETSC_ERR_ARG_WRONG, "Read-only labels cannot be altered");
  PetscCall(ISGetLocalSize(valueIS, &numStrata));
  PetscCall(ISGetIndices(valueIS, &stratumValues));
  PetscCall(DMLabelAddStrata(label, numStrata, stratumValues));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMLabelView_Concrete_Ascii(DMLabel label, PetscViewer viewer)
{
  PetscInt    v;
  PetscMPIInt rank;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)viewer), &rank));
  PetscCall(PetscViewerASCIIPushSynchronized(viewer));
  if (label) {
    const char *name;

    PetscCall(PetscObjectGetName((PetscObject)label, &name));
    PetscCall(PetscViewerASCIIPrintf(viewer, "Label '%s':\n", name));
    if (label->bt) PetscCall(PetscViewerASCIIPrintf(viewer, "  Index has been calculated in [%" PetscInt_FMT ", %" PetscInt_FMT ")\n", label->pStart, label->pEnd));
    for (v = 0; v < label->numStrata; ++v) {
      const PetscInt  value = label->stratumValues[v];
      const PetscInt *points;
      PetscInt        p;

      PetscCall(ISGetIndices(label->points[v], &points));
      for (p = 0; p < label->stratumSizes[v]; ++p) PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "[%d]: %" PetscInt_FMT " (%" PetscInt_FMT ")\n", rank, points[p], value));
      PetscCall(ISRestoreIndices(label->points[v], &points));
    }
  }
  PetscCall(PetscViewerFlush(viewer));
  PetscCall(PetscViewerASCIIPopSynchronized(viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMLabelView_Concrete(DMLabel label, PetscViewer viewer)
{
  PetscBool iascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));
  if (iascii) PetscCall(DMLabelView_Concrete_Ascii(label, viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMLabelView - View the label

  Collective

  Input Parameters:
+ label  - The `DMLabel`
- viewer - The `PetscViewer`

  Level: intermediate

.seealso: `DMLabel`, `PetscViewer`, `DM`, `DMLabelCreate()`, `DMLabelDestroy()`
@*/
PetscErrorCode DMLabelView(DMLabel label, PetscViewer viewer)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 1);
  if (!viewer) PetscCall(PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)label), &viewer));
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  PetscCall(DMLabelMakeAllValid_Private(label));
  PetscUseTypeMethod(label, view, viewer);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMLabelReset - Destroys internal data structures in a `DMLabel`

  Not Collective

  Input Parameter:
. label - The `DMLabel`

  Level: beginner

.seealso: `DMLabel`, `DM`, `DMLabelDestroy()`, `DMLabelCreate()`
@*/
PetscErrorCode DMLabelReset(DMLabel label)
{
  PetscInt v;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 1);
  for (v = 0; v < label->numStrata; ++v) {
    if (label->ht[v]) PetscCall(PetscHSetIDestroy(&label->ht[v]));
    PetscCall(ISDestroy(&label->points[v]));
  }
  label->numStrata = 0;
  PetscCall(PetscFree(label->stratumValues));
  PetscCall(PetscFree(label->stratumSizes));
  PetscCall(PetscFree(label->ht));
  PetscCall(PetscFree(label->points));
  PetscCall(PetscFree(label->validIS));
  PetscCall(PetscHMapIReset(label->hmap));
  label->pStart = -1;
  label->pEnd   = -1;
  PetscCall(PetscBTDestroy(&label->bt));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMLabelDestroy - Destroys a `DMLabel`

  Collective

  Input Parameter:
. label - The `DMLabel`

  Level: beginner

.seealso: `DMLabel`, `DM`, `DMLabelReset()`, `DMLabelCreate()`
@*/
PetscErrorCode DMLabelDestroy(DMLabel *label)
{
  PetscFunctionBegin;
  if (!*label) PetscFunctionReturn(PETSC_SUCCESS);
  PetscValidHeaderSpecific(*label, DMLABEL_CLASSID, 1);
  if (--((PetscObject)*label)->refct > 0) {
    *label = NULL;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(DMLabelReset(*label));
  PetscCall(PetscHMapIDestroy(&(*label)->hmap));
  PetscCall(PetscHeaderDestroy(label));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMLabelDuplicate_Concrete(DMLabel label, DMLabel *labelnew)
{
  PetscFunctionBegin;
  for (PetscInt v = 0; v < label->numStrata; ++v) {
    PetscCall(PetscHSetICreate(&(*labelnew)->ht[v]));
    PetscCall(PetscObjectReference((PetscObject)label->points[v]));
    (*labelnew)->points[v] = label->points[v];
  }
  PetscCall(PetscHMapIDestroy(&(*labelnew)->hmap));
  PetscCall(PetscHMapIDuplicate(label->hmap, &(*labelnew)->hmap));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMLabelDuplicate - Duplicates a `DMLabel`

  Collective

  Input Parameter:
. label - The `DMLabel`

  Output Parameter:
. labelnew - new label

  Level: intermediate

.seealso: `DMLabel`, `DM`, `DMLabelCreate()`, `DMLabelDestroy()`
@*/
PetscErrorCode DMLabelDuplicate(DMLabel label, DMLabel *labelnew)
{
  const char *name;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 1);
  PetscCall(DMLabelMakeAllValid_Private(label));
  PetscCall(PetscObjectGetName((PetscObject)label, &name));
  PetscCall(DMLabelCreate(PetscObjectComm((PetscObject)label), name, labelnew));

  (*labelnew)->numStrata    = label->numStrata;
  (*labelnew)->defaultValue = label->defaultValue;
  (*labelnew)->readonly     = label->readonly;
  PetscCall(PetscMalloc1(label->numStrata, &(*labelnew)->stratumValues));
  PetscCall(PetscMalloc1(label->numStrata, &(*labelnew)->stratumSizes));
  PetscCall(PetscCalloc1(label->numStrata, &(*labelnew)->ht));
  PetscCall(PetscCalloc1(label->numStrata, &(*labelnew)->points));
  PetscCall(PetscMalloc1(label->numStrata, &(*labelnew)->validIS));
  for (PetscInt v = 0; v < label->numStrata; ++v) {
    (*labelnew)->stratumValues[v] = label->stratumValues[v];
    (*labelnew)->stratumSizes[v]  = label->stratumSizes[v];
    (*labelnew)->validIS[v]       = PETSC_TRUE;
  }
  (*labelnew)->pStart = -1;
  (*labelnew)->pEnd   = -1;
  (*labelnew)->bt     = NULL;
  PetscUseTypeMethod(label, duplicate, labelnew);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMLabelCompare - Compare two `DMLabel` objects

  Collective; No Fortran Support

  Input Parameters:
+ comm - Comm over which to compare labels
. l0   - First `DMLabel`
- l1   - Second `DMLabel`

  Output Parameters:
+ equal   - (Optional) Flag whether the two labels are equal
- message - (Optional) Message describing the difference

  Level: intermediate

  Notes:
  The output flag equal is the same on all processes.
  If it is passed as `NULL` and difference is found, an error is thrown on all processes.
  Make sure to pass `NULL` on all processes.

  The output message is set independently on each rank.
  It is set to `NULL` if no difference was found on the current rank. It must be freed by user.
  If message is passed as `NULL` and difference is found, the difference description is printed to stderr in synchronized manner.
  Make sure to pass `NULL` on all processes.

  For the comparison, we ignore the order of stratum values, and strata with no points.

  The communicator needs to be specified because currently `DMLabel` can live on `PETSC_COMM_SELF` even if the underlying `DM` is parallel.

  Developer Note:
  Fortran stub cannot be generated automatically because `message` must be freed with `PetscFree()`

.seealso: `DMLabel`, `DM`, `DMCompareLabels()`, `DMLabelGetNumValues()`, `DMLabelGetDefaultValue()`, `DMLabelGetNonEmptyStratumValuesIS()`, `DMLabelGetStratumIS()`
@*/
PetscErrorCode DMLabelCompare(MPI_Comm comm, DMLabel l0, DMLabel l1, PetscBool *equal, char **message)
{
  const char *name0, *name1;
  char        msg[PETSC_MAX_PATH_LEN] = "";
  PetscBool   eq;
  PetscMPIInt rank;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(l0, DMLABEL_CLASSID, 2);
  PetscValidHeaderSpecific(l1, DMLABEL_CLASSID, 3);
  if (equal) PetscAssertPointer(equal, 4);
  if (message) PetscAssertPointer(message, 5);
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCall(PetscObjectGetName((PetscObject)l0, &name0));
  PetscCall(PetscObjectGetName((PetscObject)l1, &name1));
  {
    PetscInt v0, v1;

    PetscCall(DMLabelGetDefaultValue(l0, &v0));
    PetscCall(DMLabelGetDefaultValue(l1, &v1));
    eq = (PetscBool)(v0 == v1);
    if (!eq) PetscCall(PetscSNPrintf(msg, sizeof(msg), "Default value of DMLabel l0 \"%s\" = %" PetscInt_FMT " != %" PetscInt_FMT " = Default value of DMLabel l1 \"%s\"", name0, v0, v1, name1));
    PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, &eq, 1, MPIU_BOOL, MPI_LAND, comm));
    if (!eq) goto finish;
  }
  {
    IS is0, is1;

    PetscCall(DMLabelGetNonEmptyStratumValuesIS(l0, &is0));
    PetscCall(DMLabelGetNonEmptyStratumValuesIS(l1, &is1));
    PetscCall(ISEqual(is0, is1, &eq));
    PetscCall(ISDestroy(&is0));
    PetscCall(ISDestroy(&is1));
    if (!eq) PetscCall(PetscSNPrintf(msg, sizeof(msg), "Stratum values in DMLabel l0 \"%s\" are different than in DMLabel l1 \"%s\"", name0, name1));
    PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, &eq, 1, MPIU_BOOL, MPI_LAND, comm));
    if (!eq) goto finish;
  }
  {
    PetscInt i, nValues;

    PetscCall(DMLabelGetNumValues(l0, &nValues));
    for (i = 0; i < nValues; i++) {
      const PetscInt v = l0->stratumValues[i];
      PetscInt       n;
      IS             is0, is1;

      PetscCall(DMLabelGetStratumSize_Private(l0, i, &n));
      if (!n) continue;
      PetscCall(DMLabelGetStratumIS(l0, v, &is0));
      PetscCall(DMLabelGetStratumIS(l1, v, &is1));
      PetscCall(ISEqualUnsorted(is0, is1, &eq));
      PetscCall(ISDestroy(&is0));
      PetscCall(ISDestroy(&is1));
      if (!eq) {
        PetscCall(PetscSNPrintf(msg, sizeof(msg), "Stratum #%" PetscInt_FMT " with value %" PetscInt_FMT " contains different points in DMLabel l0 \"%s\" and DMLabel l1 \"%s\"", i, v, name0, name1));
        break;
      }
    }
    PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, &eq, 1, MPIU_BOOL, MPI_LAND, comm));
  }
finish:
  /* If message output arg not set, print to stderr */
  if (message) {
    *message = NULL;
    if (msg[0]) PetscCall(PetscStrallocpy(msg, message));
  } else {
    if (msg[0]) PetscCall(PetscSynchronizedFPrintf(comm, PETSC_STDERR, "[%d] %s\n", rank, msg));
    PetscCall(PetscSynchronizedFlush(comm, PETSC_STDERR));
  }
  /* If same output arg not ser and labels are not equal, throw error */
  if (equal) *equal = eq;
  else PetscCheck(eq, comm, PETSC_ERR_ARG_INCOMP, "DMLabels l0 \"%s\" and l1 \"%s\" are not equal", name0, name1);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMLabelComputeIndex - Create an index structure for membership determination, automatically determining the bounds

  Not Collective

  Input Parameter:
. label - The `DMLabel`

  Level: intermediate

.seealso: `DMLabel`, `DM`, `DMLabelHasPoint()`, `DMLabelCreateIndex()`, `DMLabelDestroyIndex()`, `DMLabelGetValue()`, `DMLabelSetValue()`
@*/
PetscErrorCode DMLabelComputeIndex(DMLabel label)
{
  PetscInt pStart = PETSC_INT_MAX, pEnd = -1, v;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 1);
  PetscCall(DMLabelMakeAllValid_Private(label));
  for (v = 0; v < label->numStrata; ++v) {
    const PetscInt *points;
    PetscInt        i;

    PetscCall(ISGetIndices(label->points[v], &points));
    for (i = 0; i < label->stratumSizes[v]; ++i) {
      const PetscInt point = points[i];

      pStart = PetscMin(point, pStart);
      pEnd   = PetscMax(point + 1, pEnd);
    }
    PetscCall(ISRestoreIndices(label->points[v], &points));
  }
  label->pStart = pStart == PETSC_INT_MAX ? -1 : pStart;
  label->pEnd   = pEnd;
  PetscCall(DMLabelCreateIndex(label, label->pStart, label->pEnd));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMLabelCreateIndex - Create an index structure for membership determination

  Not Collective

  Input Parameters:
+ label  - The `DMLabel`
. pStart - The smallest point
- pEnd   - The largest point + 1

  Level: intermediate

.seealso: `DMLabel`, `DM`, `DMLabelHasPoint()`, `DMLabelComputeIndex()`, `DMLabelDestroyIndex()`, `DMLabelGetValue()`, `DMLabelSetValue()`
@*/
PetscErrorCode DMLabelCreateIndex(DMLabel label, PetscInt pStart, PetscInt pEnd)
{
  PetscInt v;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 1);
  PetscCall(DMLabelDestroyIndex(label));
  PetscCall(DMLabelMakeAllValid_Private(label));
  label->pStart = pStart;
  label->pEnd   = pEnd;
  /* This can be hooked into SetValue(),  ClearValue(), etc. for updating */
  PetscCall(PetscBTCreate(pEnd - pStart, &label->bt));
  for (v = 0; v < label->numStrata; ++v) {
    IS              pointIS;
    const PetscInt *points;
    PetscInt        i;

    PetscUseTypeMethod(label, getstratumis, v, &pointIS);
    PetscCall(ISGetIndices(pointIS, &points));
    for (i = 0; i < label->stratumSizes[v]; ++i) {
      const PetscInt point = points[i];

      PetscCheck(!(point < pStart) && !(point >= pEnd), PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Label point %" PetscInt_FMT " in stratum %" PetscInt_FMT " is not in [%" PetscInt_FMT ", %" PetscInt_FMT ")", point, label->stratumValues[v], pStart, pEnd);
      PetscCall(PetscBTSet(label->bt, point - pStart));
    }
    PetscCall(ISRestoreIndices(label->points[v], &points));
    PetscCall(ISDestroy(&pointIS));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMLabelDestroyIndex - Destroy the index structure

  Not Collective

  Input Parameter:
. label - the `DMLabel`

  Level: intermediate

.seealso: `DMLabel`, `DM`, `DMLabelHasPoint()`, `DMLabelCreateIndex()`, `DMLabelGetValue()`, `DMLabelSetValue()`
@*/
PetscErrorCode DMLabelDestroyIndex(DMLabel label)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 1);
  label->pStart = -1;
  label->pEnd   = -1;
  PetscCall(PetscBTDestroy(&label->bt));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMLabelGetBounds - Return the smallest and largest point in the label

  Not Collective

  Input Parameter:
. label - the `DMLabel`

  Output Parameters:
+ pStart - The smallest point
- pEnd   - The largest point + 1

  Level: intermediate

  Note:
  This will compute an index for the label if one does not exist.

.seealso: `DMLabel`, `DM`, `DMLabelHasPoint()`, `DMLabelCreateIndex()`, `DMLabelGetValue()`, `DMLabelSetValue()`
@*/
PetscErrorCode DMLabelGetBounds(DMLabel label, PetscInt *pStart, PetscInt *pEnd)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 1);
  if ((label->pStart == -1) && (label->pEnd == -1)) PetscCall(DMLabelComputeIndex(label));
  if (pStart) {
    PetscAssertPointer(pStart, 2);
    *pStart = label->pStart;
  }
  if (pEnd) {
    PetscAssertPointer(pEnd, 3);
    *pEnd = label->pEnd;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMLabelHasValue - Determine whether a label assigns the value to any point

  Not Collective

  Input Parameters:
+ label - the `DMLabel`
- value - the value

  Output Parameter:
. contains - Flag indicating whether the label maps this value to any point

  Level: developer

.seealso: `DMLabel`, `DM`, `DMLabelHasPoint()`, `DMLabelGetValue()`, `DMLabelSetValue()`
@*/
PetscErrorCode DMLabelHasValue(DMLabel label, PetscInt value, PetscBool *contains)
{
  PetscInt v;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 1);
  PetscAssertPointer(contains, 3);
  PetscCall(DMLabelLookupStratum(label, value, &v));
  *contains = v < 0 ? PETSC_FALSE : PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMLabelHasPoint - Determine whether a label assigns a value to a point

  Not Collective

  Input Parameters:
+ label - the `DMLabel`
- point - the point

  Output Parameter:
. contains - Flag indicating whether the label maps this point to a value

  Level: developer

  Note:
  The user must call `DMLabelCreateIndex()` before this function.

.seealso: `DMLabel`, `DM`, `DMLabelCreateIndex()`, `DMLabelGetValue()`, `DMLabelSetValue()`
@*/
PetscErrorCode DMLabelHasPoint(DMLabel label, PetscInt point, PetscBool *contains)
{
  PetscFunctionBeginHot;
  PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 1);
  PetscAssertPointer(contains, 3);
  PetscCall(DMLabelMakeAllValid_Private(label));
  if (PetscDefined(USE_DEBUG)) {
    PetscCheck(label->bt, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Must call DMLabelCreateIndex() before DMLabelHasPoint()");
    PetscCheck(!(point < label->pStart) && !(point >= label->pEnd), PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Label point %" PetscInt_FMT " is not in [%" PetscInt_FMT ", %" PetscInt_FMT ")", point, label->pStart, label->pEnd);
  }
  *contains = PetscBTLookup(label->bt, point - label->pStart) ? PETSC_TRUE : PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMLabelStratumHasPoint - Return true if the stratum contains a point

  Not Collective

  Input Parameters:
+ label - the `DMLabel`
. value - the stratum value
- point - the point

  Output Parameter:
. contains - true if the stratum contains the point

  Level: intermediate

.seealso: `DMLabel`, `DM`, `DMLabelCreate()`, `DMLabelSetValue()`, `DMLabelClearValue()`
@*/
PetscErrorCode DMLabelStratumHasPoint(DMLabel label, PetscInt value, PetscInt point, PetscBool *contains)
{
  PetscFunctionBeginHot;
  PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 1);
  PetscAssertPointer(contains, 4);
  if (value == label->defaultValue) {
    PetscInt pointVal;

    PetscCall(DMLabelGetValue(label, point, &pointVal));
    *contains = (PetscBool)(pointVal == value);
  } else {
    PetscInt v;

    PetscCall(DMLabelLookupStratum(label, value, &v));
    if (v >= 0) {
      if (label->validIS[v] || label->readonly) {
        IS       is;
        PetscInt i;

        PetscUseTypeMethod(label, getstratumis, v, &is);
        PetscCall(ISLocate(is, point, &i));
        PetscCall(ISDestroy(&is));
        *contains = (PetscBool)(i >= 0);
      } else {
        PetscCall(PetscHSetIHas(label->ht[v], point, contains));
      }
    } else { // value is not present
      *contains = PETSC_FALSE;
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMLabelGetDefaultValue - Get the default value returned by `DMLabelGetValue()` if a point has not been explicitly given a value.
  When a label is created, it is initialized to -1.

  Not Collective

  Input Parameter:
. label - a `DMLabel` object

  Output Parameter:
. defaultValue - the default value

  Level: beginner

.seealso: `DMLabel`, `DM`, `DMLabelSetDefaultValue()`, `DMLabelGetValue()`, `DMLabelSetValue()`
@*/
PetscErrorCode DMLabelGetDefaultValue(DMLabel label, PetscInt *defaultValue)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 1);
  *defaultValue = label->defaultValue;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMLabelSetDefaultValue - Set the default value returned by `DMLabelGetValue()` if a point has not been explicitly given a value.
  When a label is created, it is initialized to -1.

  Not Collective

  Input Parameter:
. label - a `DMLabel` object

  Output Parameter:
. defaultValue - the default value

  Level: beginner

.seealso: `DMLabel`, `DM`, `DMLabelGetDefaultValue()`, `DMLabelGetValue()`, `DMLabelSetValue()`
@*/
PetscErrorCode DMLabelSetDefaultValue(DMLabel label, PetscInt defaultValue)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 1);
  label->defaultValue = defaultValue;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMLabelGetValue - Return the value a label assigns to a point, or the label's default value (which is initially -1, and can be changed with
  `DMLabelSetDefaultValue()`)

  Not Collective

  Input Parameters:
+ label - the `DMLabel`
- point - the point

  Output Parameter:
. value - The point value, or the default value (-1 by default)

  Level: intermediate

  Note:
  A label may assign multiple values to a point.  No guarantees are made about which value is returned in that case.
  Use `DMLabelStratumHasPoint()` to check for inclusion in a specific value stratum.

.seealso: `DMLabel`, `DM`, `DMLabelCreate()`, `DMLabelSetValue()`, `DMLabelClearValue()`, `DMLabelGetDefaultValue()`, `DMLabelSetDefaultValue()`
@*/
PetscErrorCode DMLabelGetValue(DMLabel label, PetscInt point, PetscInt *value)
{
  PetscInt v;

  PetscFunctionBeginHot;
  PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 1);
  PetscAssertPointer(value, 3);
  *value = label->defaultValue;
  for (v = 0; v < label->numStrata; ++v) {
    if (label->validIS[v] || label->readonly) {
      IS       is;
      PetscInt i;

      PetscUseTypeMethod(label, getstratumis, v, &is);
      PetscCall(ISLocate(label->points[v], point, &i));
      PetscCall(ISDestroy(&is));
      if (i >= 0) {
        *value = label->stratumValues[v];
        break;
      }
    } else {
      PetscBool has;

      PetscCall(PetscHSetIHas(label->ht[v], point, &has));
      if (has) {
        *value = label->stratumValues[v];
        break;
      }
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMLabelSetValue - Set the value a label assigns to a point.  If the value is the same as the label's default value (which is initially -1, and can
  be changed with `DMLabelSetDefaultValue()` to something different), then this function will do nothing.

  Not Collective

  Input Parameters:
+ label - the `DMLabel`
. point - the point
- value - The point value

  Level: intermediate

.seealso: `DMLabel`, `DM`, `DMLabelCreate()`, `DMLabelGetValue()`, `DMLabelClearValue()`, `DMLabelGetDefaultValue()`, `DMLabelSetDefaultValue()`
@*/
PetscErrorCode DMLabelSetValue(DMLabel label, PetscInt point, PetscInt value)
{
  PetscInt v;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 1);
  /* Find label value, add new entry if needed */
  if (value == label->defaultValue) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCheck(!label->readonly, PetscObjectComm((PetscObject)label), PETSC_ERR_ARG_WRONG, "Read-only labels cannot be altered");
  PetscCall(DMLabelLookupAddStratum(label, value, &v));
  /* Set key */
  PetscCall(DMLabelMakeInvalid_Private(label, v));
  PetscCall(PetscHSetIAdd(label->ht[v], point));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMLabelClearValue - Clear the value a label assigns to a point

  Not Collective

  Input Parameters:
+ label - the `DMLabel`
. point - the point
- value - The point value

  Level: intermediate

.seealso: `DMLabel`, `DM`, `DMLabelCreate()`, `DMLabelGetValue()`, `DMLabelSetValue()`
@*/
PetscErrorCode DMLabelClearValue(DMLabel label, PetscInt point, PetscInt value)
{
  PetscInt v;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 1);
  PetscCheck(!label->readonly, PetscObjectComm((PetscObject)label), PETSC_ERR_ARG_WRONG, "Read-only labels cannot be altered");
  /* Find label value */
  PetscCall(DMLabelLookupStratum(label, value, &v));
  if (v < 0) PetscFunctionReturn(PETSC_SUCCESS);

  if (label->bt) {
    PetscCheck(!(point < label->pStart) && !(point >= label->pEnd), PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Label point %" PetscInt_FMT " is not in [%" PetscInt_FMT ", %" PetscInt_FMT ")", point, label->pStart, label->pEnd);
    PetscCall(PetscBTClear(label->bt, point - label->pStart));
  }

  /* Delete key */
  PetscCall(DMLabelMakeInvalid_Private(label, v));
  PetscCall(PetscHSetIDel(label->ht[v], point));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMLabelInsertIS - Set all points in the `IS` to a value

  Not Collective

  Input Parameters:
+ label - the `DMLabel`
. is    - the point `IS`
- value - The point value

  Level: intermediate

.seealso: `DMLabel`, `DM`, `DMLabelCreate()`, `DMLabelGetValue()`, `DMLabelSetValue()`, `DMLabelClearValue()`
@*/
PetscErrorCode DMLabelInsertIS(DMLabel label, IS is, PetscInt value)
{
  PetscInt        v, n, p;
  const PetscInt *points;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 1);
  PetscValidHeaderSpecific(is, IS_CLASSID, 2);
  /* Find label value, add new entry if needed */
  if (value == label->defaultValue) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCheck(!label->readonly, PetscObjectComm((PetscObject)label), PETSC_ERR_ARG_WRONG, "Read-only labels cannot be altered");
  PetscCall(DMLabelLookupAddStratum(label, value, &v));
  /* Set keys */
  PetscCall(DMLabelMakeInvalid_Private(label, v));
  PetscCall(ISGetLocalSize(is, &n));
  PetscCall(ISGetIndices(is, &points));
  for (p = 0; p < n; ++p) PetscCall(PetscHSetIAdd(label->ht[v], points[p]));
  PetscCall(ISRestoreIndices(is, &points));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMLabelGetNumValues - Get the number of values that the `DMLabel` takes

  Not Collective

  Input Parameter:
. label - the `DMLabel`

  Output Parameter:
. numValues - the number of values

  Level: intermediate

.seealso: `DMLabel`, `DM`, `DMLabelCreate()`, `DMLabelGetValue()`, `DMLabelSetValue()`, `DMLabelClearValue()`
@*/
PetscErrorCode DMLabelGetNumValues(DMLabel label, PetscInt *numValues)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 1);
  PetscAssertPointer(numValues, 2);
  *numValues = label->numStrata;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMLabelGetValueIS - Get an `IS` of all values that the `DMlabel` takes

  Not Collective

  Input Parameter:
. label - the `DMLabel`

  Output Parameter:
. values - the value `IS`

  Level: intermediate

  Notes:
  The output `IS` should be destroyed when no longer needed.
  Strata which are allocated but empty [`DMLabelGetStratumSize()` yields 0] are counted.
  If you need to count only nonempty strata, use `DMLabelGetNonEmptyStratumValuesIS()`.

.seealso: `DMLabel`, `DM`, `DMLabelGetNonEmptyStratumValuesIS()`, `DMLabelCreate()`, `DMLabelGetValue()`, `DMLabelSetValue()`, `DMLabelClearValue()`
@*/
PetscErrorCode DMLabelGetValueIS(DMLabel label, IS *values)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 1);
  PetscAssertPointer(values, 2);
  PetscCall(ISCreateGeneral(PETSC_COMM_SELF, label->numStrata, label->stratumValues, PETSC_USE_POINTER, values));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMLabelGetValueBounds - Return the smallest and largest value in the label

  Not Collective

  Input Parameter:
. label - the `DMLabel`

  Output Parameters:
+ minValue - The smallest value
- maxValue - The largest value

  Level: intermediate

.seealso: `DMLabel`, `DM`, `DMLabelGetBounds()`, `DMLabelGetValue()`, `DMLabelSetValue()`
@*/
PetscErrorCode DMLabelGetValueBounds(DMLabel label, PetscInt *minValue, PetscInt *maxValue)
{
  PetscInt min = PETSC_INT_MAX, max = PETSC_INT_MIN;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 1);
  for (PetscInt v = 0; v < label->numStrata; ++v) {
    min = PetscMin(min, label->stratumValues[v]);
    max = PetscMax(max, label->stratumValues[v]);
  }
  if (minValue) {
    PetscAssertPointer(minValue, 2);
    *minValue = min;
  }
  if (maxValue) {
    PetscAssertPointer(maxValue, 3);
    *maxValue = max;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMLabelGetNonEmptyStratumValuesIS - Get an `IS` of all values that the `DMlabel` takes

  Not Collective

  Input Parameter:
. label - the `DMLabel`

  Output Parameter:
. values - the value `IS`

  Level: intermediate

  Notes:
  The output `IS` should be destroyed when no longer needed.
  This is similar to `DMLabelGetValueIS()` but counts only nonempty strata.

.seealso: `DMLabel`, `DM`, `DMLabelGetValueIS()`, `DMLabelCreate()`, `DMLabelGetValue()`, `DMLabelSetValue()`, `DMLabelClearValue()`
@*/
PetscErrorCode DMLabelGetNonEmptyStratumValuesIS(DMLabel label, IS *values)
{
  PetscInt  i, j;
  PetscInt *valuesArr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 1);
  PetscAssertPointer(values, 2);
  PetscCall(PetscMalloc1(label->numStrata, &valuesArr));
  for (i = 0, j = 0; i < label->numStrata; i++) {
    PetscInt n;

    PetscCall(DMLabelGetStratumSize_Private(label, i, &n));
    if (n) valuesArr[j++] = label->stratumValues[i];
  }
  if (j == label->numStrata) {
    PetscCall(ISCreateGeneral(PETSC_COMM_SELF, label->numStrata, label->stratumValues, PETSC_USE_POINTER, values));
  } else {
    PetscCall(ISCreateGeneral(PETSC_COMM_SELF, j, valuesArr, PETSC_COPY_VALUES, values));
  }
  PetscCall(PetscFree(valuesArr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMLabelGetValueIndex - Get the index of a given value in the list of values for the `DMlabel`, or -1 if it is not present

  Not Collective

  Input Parameters:
+ label - the `DMLabel`
- value - the value

  Output Parameter:
. index - the index of value in the list of values

  Level: intermediate

.seealso: `DMLabel`, `DM`, `DMLabelGetValueIS()`, `DMLabelCreate()`, `DMLabelGetValue()`, `DMLabelSetValue()`, `DMLabelClearValue()`
@*/
PetscErrorCode DMLabelGetValueIndex(DMLabel label, PetscInt value, PetscInt *index)
{
  PetscInt v;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 1);
  PetscAssertPointer(index, 3);
  /* Do not assume they are sorted */
  for (v = 0; v < label->numStrata; ++v)
    if (label->stratumValues[v] == value) break;
  if (v >= label->numStrata) *index = -1;
  else *index = v;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMLabelHasStratum - Determine whether points exist with the given value

  Not Collective

  Input Parameters:
+ label - the `DMLabel`
- value - the stratum value

  Output Parameter:
. exists - Flag saying whether points exist

  Level: intermediate

.seealso: `DMLabel`, `DM`, `DMLabelCreate()`, `DMLabelGetValue()`, `DMLabelSetValue()`, `DMLabelClearValue()`
@*/
PetscErrorCode DMLabelHasStratum(DMLabel label, PetscInt value, PetscBool *exists)
{
  PetscInt v;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 1);
  PetscAssertPointer(exists, 3);
  PetscCall(DMLabelLookupStratum(label, value, &v));
  *exists = v < 0 ? PETSC_FALSE : PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMLabelGetStratumSize - Get the size of a stratum

  Not Collective

  Input Parameters:
+ label - the `DMLabel`
- value - the stratum value

  Output Parameter:
. size - The number of points in the stratum

  Level: intermediate

.seealso: `DMLabel`, `DM`, `DMLabelCreate()`, `DMLabelGetValue()`, `DMLabelSetValue()`, `DMLabelClearValue()`
@*/
PetscErrorCode DMLabelGetStratumSize(DMLabel label, PetscInt value, PetscInt *size)
{
  PetscInt v;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 1);
  PetscAssertPointer(size, 3);
  PetscCall(DMLabelLookupStratum(label, value, &v));
  PetscCall(DMLabelGetStratumSize_Private(label, v, size));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMLabelGetStratumBounds - Get the largest and smallest point of a stratum

  Not Collective

  Input Parameters:
+ label - the `DMLabel`
- value - the stratum value

  Output Parameters:
+ start - the smallest point in the stratum
- end   - the largest point in the stratum

  Level: intermediate

.seealso: `DMLabel`, `DM`, `DMLabelCreate()`, `DMLabelGetValue()`, `DMLabelSetValue()`, `DMLabelClearValue()`
@*/
PetscErrorCode DMLabelGetStratumBounds(DMLabel label, PetscInt value, PetscInt *start, PetscInt *end)
{
  IS       is;
  PetscInt v, min, max;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 1);
  if (start) {
    PetscAssertPointer(start, 3);
    *start = -1;
  }
  if (end) {
    PetscAssertPointer(end, 4);
    *end = -1;
  }
  PetscCall(DMLabelLookupStratum(label, value, &v));
  if (v < 0) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(DMLabelMakeValid_Private(label, v));
  if (label->stratumSizes[v] <= 0) PetscFunctionReturn(PETSC_SUCCESS);
  PetscUseTypeMethod(label, getstratumis, v, &is);
  PetscCall(ISGetMinMax(is, &min, &max));
  PetscCall(ISDestroy(&is));
  if (start) *start = min;
  if (end) *end = max + 1;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMLabelGetStratumIS_Concrete(DMLabel label, PetscInt v, IS *pointIS)
{
  PetscFunctionBegin;
  PetscCall(PetscObjectReference((PetscObject)label->points[v]));
  *pointIS = label->points[v];
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMLabelGetStratumIS - Get an `IS` with the stratum points

  Not Collective

  Input Parameters:
+ label - the `DMLabel`
- value - the stratum value

  Output Parameter:
. points - The stratum points

  Level: intermediate

  Notes:
  The output `IS` should be destroyed when no longer needed.
  Returns `NULL` if the stratum is empty.

.seealso: `DMLabel`, `DM`, `DMLabelCreate()`, `DMLabelGetValue()`, `DMLabelSetValue()`, `DMLabelClearValue()`
@*/
PetscErrorCode DMLabelGetStratumIS(DMLabel label, PetscInt value, IS *points)
{
  PetscInt v;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 1);
  PetscAssertPointer(points, 3);
  *points = NULL;
  PetscCall(DMLabelLookupStratum(label, value, &v));
  if (v < 0) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(DMLabelMakeValid_Private(label, v));
  PetscUseTypeMethod(label, getstratumis, v, points);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMLabelSetStratumIS - Set the stratum points using an `IS`

  Not Collective

  Input Parameters:
+ label - the `DMLabel`
. value - the stratum value
- is    - The stratum points

  Level: intermediate

.seealso: `DMLabel`, `DM`, `DMLabelCreate()`, `DMLabelGetValue()`, `DMLabelSetValue()`, `DMLabelClearValue()`
@*/
PetscErrorCode DMLabelSetStratumIS(DMLabel label, PetscInt value, IS is)
{
  PetscInt v;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 1);
  PetscValidHeaderSpecific(is, IS_CLASSID, 3);
  PetscCheck(!label->readonly, PetscObjectComm((PetscObject)label), PETSC_ERR_ARG_WRONG, "Read-only labels cannot be altered");
  PetscCall(DMLabelLookupAddStratum(label, value, &v));
  if (is == label->points[v]) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(DMLabelClearStratum(label, value));
  PetscCall(ISGetLocalSize(is, &label->stratumSizes[v]));
  PetscCall(PetscObjectReference((PetscObject)is));
  PetscCall(ISDestroy(&label->points[v]));
  label->points[v]  = is;
  label->validIS[v] = PETSC_TRUE;
  PetscCall(PetscObjectStateIncrease((PetscObject)label));
  if (label->bt) {
    const PetscInt *points;
    PetscInt        p;

    PetscCall(ISGetIndices(is, &points));
    for (p = 0; p < label->stratumSizes[v]; ++p) {
      const PetscInt point = points[p];

      PetscCheck(!(point < label->pStart) && !(point >= label->pEnd), PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Label point %" PetscInt_FMT " is not in [%" PetscInt_FMT ", %" PetscInt_FMT ")", point, label->pStart, label->pEnd);
      PetscCall(PetscBTSet(label->bt, point - label->pStart));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMLabelClearStratum - Remove a stratum

  Not Collective

  Input Parameters:
+ label - the `DMLabel`
- value - the stratum value

  Level: intermediate

.seealso: `DMLabel`, `DM`, `DMLabelCreate()`, `DMLabelGetValue()`, `DMLabelSetValue()`, `DMLabelClearValue()`
@*/
PetscErrorCode DMLabelClearStratum(DMLabel label, PetscInt value)
{
  PetscInt v;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 1);
  PetscCheck(!label->readonly, PetscObjectComm((PetscObject)label), PETSC_ERR_ARG_WRONG, "Read-only labels cannot be altered");
  PetscCall(DMLabelLookupStratum(label, value, &v));
  if (v < 0) PetscFunctionReturn(PETSC_SUCCESS);
  if (label->validIS[v]) {
    if (label->bt) {
      PetscInt        i;
      const PetscInt *points;

      PetscCall(ISGetIndices(label->points[v], &points));
      for (i = 0; i < label->stratumSizes[v]; ++i) {
        const PetscInt point = points[i];

        PetscCheck(!(point < label->pStart) && !(point >= label->pEnd), PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Label point %" PetscInt_FMT " is not in [%" PetscInt_FMT ", %" PetscInt_FMT ")", point, label->pStart, label->pEnd);
        PetscCall(PetscBTClear(label->bt, point - label->pStart));
      }
      PetscCall(ISRestoreIndices(label->points[v], &points));
    }
    label->stratumSizes[v] = 0;
    PetscCall(ISDestroy(&label->points[v]));
    PetscCall(ISCreateStride(PETSC_COMM_SELF, 0, 0, 1, &label->points[v]));
    PetscCall(PetscObjectSetName((PetscObject)label->points[v], "indices"));
    PetscCall(PetscObjectStateIncrease((PetscObject)label));
  } else {
    PetscCall(PetscHSetIClear(label->ht[v]));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMLabelSetStratumBounds - Efficiently give a contiguous set of points a given label value

  Not Collective

  Input Parameters:
+ label  - The `DMLabel`
. value  - The label value for all points
. pStart - The first point
- pEnd   - A point beyond all marked points

  Level: intermediate

  Note:
  The marks points are [`pStart`, `pEnd`), and only the bounds are stored.

.seealso: `DMLabel`, `DM`, `DMLabelCreate()`, `DMLabelSetStratumIS()`, `DMLabelGetStratumIS()`
@*/
PetscErrorCode DMLabelSetStratumBounds(DMLabel label, PetscInt value, PetscInt pStart, PetscInt pEnd)
{
  IS pIS;

  PetscFunctionBegin;
  PetscCall(ISCreateStride(PETSC_COMM_SELF, pEnd - pStart, pStart, 1, &pIS));
  PetscCall(DMLabelSetStratumIS(label, value, pIS));
  PetscCall(ISDestroy(&pIS));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMLabelGetStratumPointIndex - Get the index of a point in a given stratum

  Not Collective

  Input Parameters:
+ label - The `DMLabel`
. value - The label value
- p     - A point with this value

  Output Parameter:
. index - The index of this point in the stratum, or -1 if the point is not in the stratum or the stratum does not exist

  Level: intermediate

.seealso: `DMLabel`, `DM`, `DMLabelGetValueIndex()`, `DMLabelGetStratumIS()`, `DMLabelCreate()`
@*/
PetscErrorCode DMLabelGetStratumPointIndex(DMLabel label, PetscInt value, PetscInt p, PetscInt *index)
{
  IS       pointIS;
  PetscInt v;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 1);
  PetscAssertPointer(index, 4);
  *index = -1;
  PetscCall(DMLabelLookupStratum(label, value, &v));
  if (v < 0) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(DMLabelMakeValid_Private(label, v));
  PetscUseTypeMethod(label, getstratumis, v, &pointIS);
  PetscCall(ISLocate(pointIS, p, index));
  PetscCall(ISDestroy(&pointIS));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMLabelFilter - Remove all points outside of [`start`, `end`)

  Not Collective

  Input Parameters:
+ label - the `DMLabel`
. start - the first point kept
- end   - one more than the last point kept

  Level: intermediate

.seealso: `DMLabel`, `DM`, `DMLabelCreate()`, `DMLabelGetValue()`, `DMLabelSetValue()`, `DMLabelClearValue()`
@*/
PetscErrorCode DMLabelFilter(DMLabel label, PetscInt start, PetscInt end)
{
  PetscInt v;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 1);
  PetscCheck(!label->readonly, PetscObjectComm((PetscObject)label), PETSC_ERR_ARG_WRONG, "Read-only labels cannot be altered");
  PetscCall(DMLabelDestroyIndex(label));
  PetscCall(DMLabelMakeAllValid_Private(label));
  for (v = 0; v < label->numStrata; ++v) {
    PetscCall(ISGeneralFilter(label->points[v], start, end));
    PetscCall(ISGetLocalSize(label->points[v], &label->stratumSizes[v]));
  }
  PetscCall(DMLabelCreateIndex(label, start, end));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMLabelPermute - Create a new label with permuted points

  Not Collective

  Input Parameters:
+ label       - the `DMLabel`
- permutation - the point permutation

  Output Parameter:
. labelNew - the new label containing the permuted points

  Level: intermediate

.seealso: `DMLabel`, `DM`, `DMLabelCreate()`, `DMLabelGetValue()`, `DMLabelSetValue()`, `DMLabelClearValue()`
@*/
PetscErrorCode DMLabelPermute(DMLabel label, IS permutation, DMLabel *labelNew)
{
  const PetscInt *perm;
  PetscInt        numValues, numPoints, v, q;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 1);
  PetscValidHeaderSpecific(permutation, IS_CLASSID, 2);
  PetscCheck(!label->readonly, PetscObjectComm((PetscObject)label), PETSC_ERR_ARG_WRONG, "Read-only labels cannot be altered");
  PetscCall(DMLabelMakeAllValid_Private(label));
  PetscCall(DMLabelDuplicate(label, labelNew));
  PetscCall(DMLabelGetNumValues(*labelNew, &numValues));
  PetscCall(ISGetLocalSize(permutation, &numPoints));
  PetscCall(ISGetIndices(permutation, &perm));
  for (v = 0; v < numValues; ++v) {
    const PetscInt  size = (*labelNew)->stratumSizes[v];
    const PetscInt *points;
    PetscInt       *pointsNew;

    PetscCall(ISGetIndices((*labelNew)->points[v], &points));
    PetscCall(PetscCalloc1(size, &pointsNew));
    for (q = 0; q < size; ++q) {
      const PetscInt point = points[q];

      PetscCheck(!(point < 0) && !(point >= numPoints), PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Label point %" PetscInt_FMT " is not in [0, %" PetscInt_FMT ") for the remapping", point, numPoints);
      pointsNew[q] = perm[point];
    }
    PetscCall(ISRestoreIndices((*labelNew)->points[v], &points));
    PetscCall(PetscSortInt(size, pointsNew));
    PetscCall(ISDestroy(&(*labelNew)->points[v]));
    if (size > 0 && pointsNew[size - 1] == pointsNew[0] + size - 1) {
      PetscCall(ISCreateStride(PETSC_COMM_SELF, size, pointsNew[0], 1, &((*labelNew)->points[v])));
      PetscCall(PetscFree(pointsNew));
    } else {
      PetscCall(ISCreateGeneral(PETSC_COMM_SELF, size, pointsNew, PETSC_OWN_POINTER, &((*labelNew)->points[v])));
    }
    PetscCall(PetscObjectSetName((PetscObject)((*labelNew)->points[v]), "indices"));
  }
  PetscCall(ISRestoreIndices(permutation, &perm));
  if (label->bt) {
    PetscCall(PetscBTDestroy(&label->bt));
    PetscCall(DMLabelCreateIndex(label, label->pStart, label->pEnd));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMLabelPermuteValues - Permute the values in a label

  Not collective

  Input Parameters:
+ label       - the `DMLabel`
- permutation - the value permutation, permutation[old value] = new value

  Output Parameter:
. label - the `DMLabel` now with permuted values

  Note:
  The modification is done in-place

  Level: intermediate

.seealso: `DMLabelRewriteValues()`, `DMLabel`, `DM`, `DMLabelPermute()`, `DMLabelCreate()`, `DMLabelGetValue()`, `DMLabelSetValue()`, `DMLabelClearValue()`
@*/
PetscErrorCode DMLabelPermuteValues(DMLabel label, IS permutation)
{
  PetscInt Nv, Np;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 1);
  PetscValidHeaderSpecific(permutation, IS_CLASSID, 2);
  PetscCall(DMLabelGetNumValues(label, &Nv));
  PetscCall(ISGetLocalSize(permutation, &Np));
  PetscCheck(Np == Nv, PetscObjectComm((PetscObject)label), PETSC_ERR_ARG_SIZ, "Permutation has size %" PetscInt_FMT " != %" PetscInt_FMT " number of label values", Np, Nv);
  if (PetscDefined(USE_DEBUG)) {
    PetscBool flg;
    PetscCall(ISGetInfo(permutation, IS_PERMUTATION, IS_LOCAL, PETSC_TRUE, &flg));
    PetscCheck(flg, PetscObjectComm((PetscObject)label), PETSC_ERR_ARG_WRONG, "IS is not a permutation");
  }
  PetscCall(DMLabelRewriteValues(label, permutation));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMLabelRewriteValues - Permute the values in a label, but some may be omitted

  Not collective

  Input Parameters:
+ label       - the `DMLabel`
- permutation - the value permutation, permutation[old value] = new value, but some maybe omitted

  Output Parameter:
. label - the `DMLabel` now with permuted values

  Note:
  The modification is done in-place

  Level: intermediate

.seealso: `DMLabelPermuteValues()`, `DMLabel`, `DM`, `DMLabelPermute()`, `DMLabelCreate()`, `DMLabelGetValue()`, `DMLabelSetValue()`, `DMLabelClearValue()`
@*/
PetscErrorCode DMLabelRewriteValues(DMLabel label, IS permutation)
{
  const PetscInt *perm;
  PetscInt        Nv, Np;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 1);
  PetscValidHeaderSpecific(permutation, IS_CLASSID, 2);
  PetscCall(DMLabelMakeAllValid_Private(label));
  PetscCall(DMLabelGetNumValues(label, &Nv));
  PetscCall(ISGetLocalSize(permutation, &Np));
  PetscCheck(Np >= Nv, PetscObjectComm((PetscObject)label), PETSC_ERR_ARG_SIZ, "Permutation has size %" PetscInt_FMT " < %" PetscInt_FMT " number of label values", Np, Nv);
  PetscCall(ISGetIndices(permutation, &perm));
  for (PetscInt v = 0; v < Nv; ++v) label->stratumValues[v] = perm[label->stratumValues[v]];
  PetscCall(ISRestoreIndices(permutation, &perm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMLabelDistribute_Internal(DMLabel label, PetscSF sf, PetscSection *leafSection, PetscInt **leafStrata)
{
  MPI_Comm     comm;
  PetscInt     s, l, nroots, nleaves, offset, size;
  PetscInt    *remoteOffsets, *rootStrata, *rootIdx;
  PetscSection rootSection;
  PetscSF      labelSF;

  PetscFunctionBegin;
  if (label) PetscCall(DMLabelMakeAllValid_Private(label));
  PetscCall(PetscObjectGetComm((PetscObject)sf, &comm));
  /* Build a section of stratum values per point, generate the according SF
     and distribute point-wise stratum values to leaves. */
  PetscCall(PetscSFGetGraph(sf, &nroots, &nleaves, NULL, NULL));
  PetscCall(PetscSectionCreate(comm, &rootSection));
  PetscCall(PetscSectionSetChart(rootSection, 0, nroots));
  if (label) {
    for (s = 0; s < label->numStrata; ++s) {
      const PetscInt *points;

      PetscCall(ISGetIndices(label->points[s], &points));
      for (l = 0; l < label->stratumSizes[s]; l++) PetscCall(PetscSectionAddDof(rootSection, points[l], 1));
      PetscCall(ISRestoreIndices(label->points[s], &points));
    }
  }
  PetscCall(PetscSectionSetUp(rootSection));
  /* Create a point-wise array of stratum values */
  PetscCall(PetscSectionGetStorageSize(rootSection, &size));
  PetscCall(PetscMalloc1(size, &rootStrata));
  PetscCall(PetscCalloc1(nroots, &rootIdx));
  if (label) {
    for (s = 0; s < label->numStrata; ++s) {
      const PetscInt *points;

      PetscCall(ISGetIndices(label->points[s], &points));
      for (l = 0; l < label->stratumSizes[s]; l++) {
        const PetscInt p = points[l];
        PetscCall(PetscSectionGetOffset(rootSection, p, &offset));
        rootStrata[offset + rootIdx[p]++] = label->stratumValues[s];
      }
      PetscCall(ISRestoreIndices(label->points[s], &points));
    }
  }
  /* Build SF that maps label points to remote processes */
  PetscCall(PetscSectionCreate(comm, leafSection));
  PetscCall(PetscSFDistributeSection(sf, rootSection, &remoteOffsets, *leafSection));
  PetscCall(PetscSFCreateSectionSF(sf, rootSection, remoteOffsets, *leafSection, &labelSF));
  PetscCall(PetscFree(remoteOffsets));
  /* Send the strata for each point over the derived SF */
  PetscCall(PetscSectionGetStorageSize(*leafSection, &size));
  PetscCall(PetscMalloc1(size, leafStrata));
  PetscCall(PetscSFBcastBegin(labelSF, MPIU_INT, rootStrata, *leafStrata, MPI_REPLACE));
  PetscCall(PetscSFBcastEnd(labelSF, MPIU_INT, rootStrata, *leafStrata, MPI_REPLACE));
  /* Clean up */
  PetscCall(PetscFree(rootStrata));
  PetscCall(PetscFree(rootIdx));
  PetscCall(PetscSectionDestroy(&rootSection));
  PetscCall(PetscSFDestroy(&labelSF));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMLabelDistribute - Create a new label pushed forward over the `PetscSF`

  Collective

  Input Parameters:
+ label - the `DMLabel`
- sf    - the map from old to new distribution

  Output Parameter:
. labelNew - the new redistributed label

  Level: intermediate

.seealso: `DMLabel`, `DM`, `DMLabelCreate()`, `DMLabelGetValue()`, `DMLabelSetValue()`, `DMLabelClearValue()`
@*/
PetscErrorCode DMLabelDistribute(DMLabel label, PetscSF sf, DMLabel *labelNew)
{
  MPI_Comm     comm;
  PetscSection leafSection;
  PetscInt     p, pStart, pEnd, s, size, dof, offset, stratum;
  PetscInt    *leafStrata, *strataIdx;
  PetscInt   **points;
  const char  *lname = NULL;
  char        *name;
  PetscMPIInt  nameSize;
  PetscHSetI   stratumHash;
  size_t       len = 0;
  PetscMPIInt  rank;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf, PETSCSF_CLASSID, 2);
  if (label) {
    PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 1);
    PetscCheck(!label->readonly, PetscObjectComm((PetscObject)label), PETSC_ERR_ARG_WRONG, "Read-only labels cannot be altered");
    PetscCall(DMLabelMakeAllValid_Private(label));
  }
  PetscCall(PetscObjectGetComm((PetscObject)sf, &comm));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  /* Bcast name */
  if (rank == 0) {
    PetscCall(PetscObjectGetName((PetscObject)label, &lname));
    PetscCall(PetscStrlen(lname, &len));
  }
  PetscCall(PetscMPIIntCast(len, &nameSize));
  PetscCallMPI(MPI_Bcast(&nameSize, 1, MPI_INT, 0, comm));
  PetscCall(PetscMalloc1(nameSize + 1, &name));
  if (rank == 0) PetscCall(PetscArraycpy(name, lname, nameSize + 1));
  PetscCallMPI(MPI_Bcast(name, nameSize + 1, MPI_CHAR, 0, comm));
  PetscCall(DMLabelCreate(PETSC_COMM_SELF, name, labelNew));
  PetscCall(PetscFree(name));
  /* Bcast defaultValue */
  if (rank == 0) (*labelNew)->defaultValue = label->defaultValue;
  PetscCallMPI(MPI_Bcast(&(*labelNew)->defaultValue, 1, MPIU_INT, 0, comm));
  /* Distribute stratum values over the SF and get the point mapping on the receiver */
  PetscCall(DMLabelDistribute_Internal(label, sf, &leafSection, &leafStrata));
  /* Determine received stratum values and initialise new label*/
  PetscCall(PetscHSetICreate(&stratumHash));
  PetscCall(PetscSectionGetStorageSize(leafSection, &size));
  for (p = 0; p < size; ++p) PetscCall(PetscHSetIAdd(stratumHash, leafStrata[p]));
  PetscCall(PetscHSetIGetSize(stratumHash, &(*labelNew)->numStrata));
  PetscCall(PetscMalloc1((*labelNew)->numStrata, &(*labelNew)->validIS));
  for (s = 0; s < (*labelNew)->numStrata; ++s) (*labelNew)->validIS[s] = PETSC_TRUE;
  PetscCall(PetscMalloc1((*labelNew)->numStrata, &(*labelNew)->stratumValues));
  /* Turn leafStrata into indices rather than stratum values */
  offset = 0;
  PetscCall(PetscHSetIGetElems(stratumHash, &offset, (*labelNew)->stratumValues));
  PetscCall(PetscSortInt((*labelNew)->numStrata, (*labelNew)->stratumValues));
  for (s = 0; s < (*labelNew)->numStrata; ++s) PetscCall(PetscHMapISet((*labelNew)->hmap, (*labelNew)->stratumValues[s], s));
  for (p = 0; p < size; ++p) {
    for (s = 0; s < (*labelNew)->numStrata; ++s) {
      if (leafStrata[p] == (*labelNew)->stratumValues[s]) {
        leafStrata[p] = s;
        break;
      }
    }
  }
  /* Rebuild the point strata on the receiver */
  PetscCall(PetscCalloc1((*labelNew)->numStrata, &(*labelNew)->stratumSizes));
  PetscCall(PetscSectionGetChart(leafSection, &pStart, &pEnd));
  for (p = pStart; p < pEnd; p++) {
    PetscCall(PetscSectionGetDof(leafSection, p, &dof));
    PetscCall(PetscSectionGetOffset(leafSection, p, &offset));
    for (s = 0; s < dof; s++) (*labelNew)->stratumSizes[leafStrata[offset + s]]++;
  }
  PetscCall(PetscCalloc1((*labelNew)->numStrata, &(*labelNew)->ht));
  PetscCall(PetscCalloc1((*labelNew)->numStrata, &(*labelNew)->points));
  PetscCall(PetscCalloc1((*labelNew)->numStrata, &points));
  for (s = 0; s < (*labelNew)->numStrata; ++s) {
    PetscCall(PetscHSetICreate(&(*labelNew)->ht[s]));
    PetscCall(PetscMalloc1((*labelNew)->stratumSizes[s], &points[s]));
  }
  /* Insert points into new strata */
  PetscCall(PetscCalloc1((*labelNew)->numStrata, &strataIdx));
  PetscCall(PetscSectionGetChart(leafSection, &pStart, &pEnd));
  for (p = pStart; p < pEnd; p++) {
    PetscCall(PetscSectionGetDof(leafSection, p, &dof));
    PetscCall(PetscSectionGetOffset(leafSection, p, &offset));
    for (s = 0; s < dof; s++) {
      stratum                               = leafStrata[offset + s];
      points[stratum][strataIdx[stratum]++] = p;
    }
  }
  for (s = 0; s < (*labelNew)->numStrata; s++) {
    PetscCall(ISCreateGeneral(PETSC_COMM_SELF, (*labelNew)->stratumSizes[s], &points[s][0], PETSC_OWN_POINTER, &((*labelNew)->points[s])));
    PetscCall(PetscObjectSetName((PetscObject)((*labelNew)->points[s]), "indices"));
  }
  PetscCall(PetscFree(points));
  PetscCall(PetscHSetIDestroy(&stratumHash));
  PetscCall(PetscFree(leafStrata));
  PetscCall(PetscFree(strataIdx));
  PetscCall(PetscSectionDestroy(&leafSection));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMLabelGather - Gather all label values from leafs into roots

  Collective

  Input Parameters:
+ label - the `DMLabel`
- sf    - the `PetscSF` communication map

  Output Parameter:
. labelNew - the new `DMLabel` with localised leaf values

  Level: developer

  Note:
  This is the inverse operation to `DMLabelDistribute()`.

.seealso: `DMLabel`, `DM`, `DMLabelDistribute()`
@*/
PetscErrorCode DMLabelGather(DMLabel label, PetscSF sf, DMLabel *labelNew)
{
  MPI_Comm        comm;
  PetscSection    rootSection;
  PetscSF         sfLabel;
  PetscSFNode    *rootPoints, *leafPoints;
  PetscInt        p, s, d, nroots, nleaves, nmultiroots, idx, dof, offset;
  const PetscInt *rootDegree, *ilocal;
  PetscInt       *rootStrata;
  const char     *lname;
  char           *name;
  PetscMPIInt     nameSize;
  size_t          len = 0;
  PetscMPIInt     rank, size;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 1);
  PetscValidHeaderSpecific(sf, PETSCSF_CLASSID, 2);
  PetscCheck(!label->readonly, PetscObjectComm((PetscObject)label), PETSC_ERR_ARG_WRONG, "Read-only labels cannot be altered");
  PetscCall(PetscObjectGetComm((PetscObject)sf, &comm));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  /* Bcast name */
  if (rank == 0) {
    PetscCall(PetscObjectGetName((PetscObject)label, &lname));
    PetscCall(PetscStrlen(lname, &len));
  }
  PetscCall(PetscMPIIntCast(len, &nameSize));
  PetscCallMPI(MPI_Bcast(&nameSize, 1, MPI_INT, 0, comm));
  PetscCall(PetscMalloc1(nameSize + 1, &name));
  if (rank == 0) PetscCall(PetscArraycpy(name, lname, nameSize + 1));
  PetscCallMPI(MPI_Bcast(name, nameSize + 1, MPI_CHAR, 0, comm));
  PetscCall(DMLabelCreate(PETSC_COMM_SELF, name, labelNew));
  PetscCall(PetscFree(name));
  /* Gather rank/index pairs of leaves into local roots to build
     an inverse, multi-rooted SF. Note that this ignores local leaf
     indexing due to the use of the multiSF in PetscSFGather. */
  PetscCall(PetscSFGetGraph(sf, &nroots, &nleaves, &ilocal, NULL));
  PetscCall(PetscMalloc1(nroots, &leafPoints));
  for (p = 0; p < nroots; ++p) leafPoints[p].rank = leafPoints[p].index = -1;
  for (p = 0; p < nleaves; p++) {
    PetscInt ilp = ilocal ? ilocal[p] : p;

    leafPoints[ilp].index = ilp;
    leafPoints[ilp].rank  = rank;
  }
  PetscCall(PetscSFComputeDegreeBegin(sf, &rootDegree));
  PetscCall(PetscSFComputeDegreeEnd(sf, &rootDegree));
  for (p = 0, nmultiroots = 0; p < nroots; ++p) nmultiroots += rootDegree[p];
  PetscCall(PetscMalloc1(nmultiroots, &rootPoints));
  PetscCall(PetscSFGatherBegin(sf, MPIU_SF_NODE, leafPoints, rootPoints));
  PetscCall(PetscSFGatherEnd(sf, MPIU_SF_NODE, leafPoints, rootPoints));
  PetscCall(PetscSFCreate(comm, &sfLabel));
  PetscCall(PetscSFSetGraph(sfLabel, nroots, nmultiroots, NULL, PETSC_OWN_POINTER, rootPoints, PETSC_OWN_POINTER));
  /* Migrate label over inverted SF to pull stratum values at leaves into roots. */
  PetscCall(DMLabelDistribute_Internal(label, sfLabel, &rootSection, &rootStrata));
  /* Rebuild the point strata on the receiver */
  for (p = 0, idx = 0; p < nroots; p++) {
    for (d = 0; d < rootDegree[p]; d++) {
      PetscCall(PetscSectionGetDof(rootSection, idx + d, &dof));
      PetscCall(PetscSectionGetOffset(rootSection, idx + d, &offset));
      for (s = 0; s < dof; s++) PetscCall(DMLabelSetValue(*labelNew, p, rootStrata[offset + s]));
    }
    idx += rootDegree[p];
  }
  PetscCall(PetscFree(leafPoints));
  PetscCall(PetscFree(rootStrata));
  PetscCall(PetscSectionDestroy(&rootSection));
  PetscCall(PetscSFDestroy(&sfLabel));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMLabelPropagateInit_Internal(DMLabel label, PetscSF pointSF, PetscInt valArray[])
{
  const PetscInt *degree;
  const PetscInt *points;
  PetscInt        Nr, r, Nl, l, val, defVal;

  PetscFunctionBegin;
  PetscCall(DMLabelGetDefaultValue(label, &defVal));
  /* Add in leaves */
  PetscCall(PetscSFGetGraph(pointSF, &Nr, &Nl, &points, NULL));
  for (l = 0; l < Nl; ++l) {
    PetscCall(DMLabelGetValue(label, points[l], &val));
    if (val != defVal) valArray[points[l]] = val;
  }
  /* Add in shared roots */
  PetscCall(PetscSFComputeDegreeBegin(pointSF, &degree));
  PetscCall(PetscSFComputeDegreeEnd(pointSF, &degree));
  for (r = 0; r < Nr; ++r) {
    if (degree[r]) {
      PetscCall(DMLabelGetValue(label, r, &val));
      if (val != defVal) valArray[r] = val;
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMLabelPropagateFini_Internal(DMLabel label, PetscSF pointSF, PetscInt valArray[], PetscErrorCode (*markPoint)(DMLabel, PetscInt, PetscInt, void *), void *ctx)
{
  const PetscInt *degree;
  const PetscInt *points;
  PetscInt        Nr, r, Nl, l, val, defVal;

  PetscFunctionBegin;
  PetscCall(DMLabelGetDefaultValue(label, &defVal));
  /* Read out leaves */
  PetscCall(PetscSFGetGraph(pointSF, &Nr, &Nl, &points, NULL));
  for (l = 0; l < Nl; ++l) {
    const PetscInt p    = points[l];
    const PetscInt cval = valArray[p];

    if (cval != defVal) {
      PetscCall(DMLabelGetValue(label, p, &val));
      if (val == defVal) {
        PetscCall(DMLabelSetValue(label, p, cval));
        if (markPoint) PetscCall((*markPoint)(label, p, cval, ctx));
      }
    }
  }
  /* Read out shared roots */
  PetscCall(PetscSFComputeDegreeBegin(pointSF, &degree));
  PetscCall(PetscSFComputeDegreeEnd(pointSF, &degree));
  for (r = 0; r < Nr; ++r) {
    if (degree[r]) {
      const PetscInt cval = valArray[r];

      if (cval != defVal) {
        PetscCall(DMLabelGetValue(label, r, &val));
        if (val == defVal) {
          PetscCall(DMLabelSetValue(label, r, cval));
          if (markPoint) PetscCall((*markPoint)(label, r, cval, ctx));
        }
      }
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMLabelPropagateBegin - Setup a cycle of label propagation

  Collective

  Input Parameters:
+ label - The `DMLabel` to propagate across processes
- sf    - The `PetscSF` describing parallel layout of the label points

  Level: intermediate

.seealso: `DMLabel`, `DM`, `DMLabelPropagateEnd()`, `DMLabelPropagatePush()`
@*/
PetscErrorCode DMLabelPropagateBegin(DMLabel label, PetscSF sf)
{
  PetscInt    Nr, r, defVal;
  PetscMPIInt size;

  PetscFunctionBegin;
  PetscCheck(!label->readonly, PetscObjectComm((PetscObject)label), PETSC_ERR_ARG_WRONG, "Read-only labels cannot be altered");
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)sf), &size));
  if (size > 1) {
    PetscCall(DMLabelGetDefaultValue(label, &defVal));
    PetscCall(PetscSFGetGraph(sf, &Nr, NULL, NULL, NULL));
    if (Nr >= 0) PetscCall(PetscMalloc1(Nr, &label->propArray));
    for (r = 0; r < Nr; ++r) label->propArray[r] = defVal;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMLabelPropagateEnd - Tear down a cycle of label propagation

  Collective

  Input Parameters:
+ label   - The `DMLabel` to propagate across processes
- pointSF - The `PetscSF` describing parallel layout of the label points

  Level: intermediate

.seealso: `DMLabel`, `DM`, `DMLabelPropagateBegin()`, `DMLabelPropagatePush()`
@*/
PetscErrorCode DMLabelPropagateEnd(DMLabel label, PetscSF pointSF)
{
  PetscFunctionBegin;
  PetscCheck(!label->readonly, PetscObjectComm((PetscObject)label), PETSC_ERR_ARG_WRONG, "Read-only labels cannot be altered");
  PetscCall(PetscFree(label->propArray));
  label->propArray = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMLabelPropagatePush - Tear down a cycle of label propagation

  Collective

  Input Parameters:
+ label     - The `DMLabel` to propagate across processes
. pointSF   - The `PetscSF` describing parallel layout of the label points
. markPoint - An optional callback that is called when a point is marked, or `NULL`
- ctx       - An optional user context for the callback, or `NULL`

  Calling sequence of `markPoint`:
+ label - The `DMLabel`
. p     - The point being marked
. val   - The label value for `p`
- ctx   - An optional user context

  Level: intermediate

.seealso: `DMLabel`, `DM`, `DMLabelPropagateBegin()`, `DMLabelPropagateEnd()`
@*/
PetscErrorCode DMLabelPropagatePush(DMLabel label, PetscSF pointSF, PetscErrorCode (*markPoint)(DMLabel label, PetscInt p, PetscInt val, void *ctx), void *ctx)
{
  PetscInt   *valArray = label->propArray, Nr;
  PetscMPIInt size;

  PetscFunctionBegin;
  PetscCheck(!label->readonly, PetscObjectComm((PetscObject)label), PETSC_ERR_ARG_WRONG, "Read-only labels cannot be altered");
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)pointSF), &size));
  PetscCall(PetscSFGetGraph(pointSF, &Nr, NULL, NULL, NULL));
  if (size > 1 && Nr >= 0) {
    /* Communicate marked edges
       The current implementation allocates an array the size of the number of root. We put the label values into the
       array, and then call PetscSFReduce()+PetscSFBcast() to make the marks consistent.

       TODO: We could use in-place communication with a different SF
       We use MPI_SUM for the Reduce, and check the result against the rootdegree. If sum >= rootdegree+1, then the edge has
       already been marked. If not, it might have been handled on the process in this round, but we add it anyway.

       In order to update the queue with the new edges from the label communication, we use BcastAnOp(MPI_SUM), so that new
       values will have 1+0=1 and old values will have 1+1=2. Loop over these, resetting the values to 1, and adding any new
       edge to the queue.
    */
    PetscCall(DMLabelPropagateInit_Internal(label, pointSF, valArray));
    PetscCall(PetscSFReduceBegin(pointSF, MPIU_INT, valArray, valArray, MPI_MAX));
    PetscCall(PetscSFReduceEnd(pointSF, MPIU_INT, valArray, valArray, MPI_MAX));
    PetscCall(PetscSFBcastBegin(pointSF, MPIU_INT, valArray, valArray, MPI_REPLACE));
    PetscCall(PetscSFBcastEnd(pointSF, MPIU_INT, valArray, valArray, MPI_REPLACE));
    PetscCall(DMLabelPropagateFini_Internal(label, pointSF, valArray, markPoint, ctx));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMLabelConvertToSection - Make a `PetscSection`/`IS` pair that encodes the label

  Not Collective

  Input Parameter:
. label - the `DMLabel`

  Output Parameters:
+ section - the section giving offsets for each stratum
- is      - An `IS` containing all the label points

  Level: developer

.seealso: `DMLabel`, `DM`, `DMLabelDistribute()`
@*/
PetscErrorCode DMLabelConvertToSection(DMLabel label, PetscSection *section, IS *is)
{
  IS              vIS;
  const PetscInt *values;
  PetscInt       *points;
  PetscInt        nV, vS = 0, vE = 0, v, N;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 1);
  PetscCall(DMLabelGetNumValues(label, &nV));
  PetscCall(DMLabelGetValueIS(label, &vIS));
  PetscCall(ISGetIndices(vIS, &values));
  if (nV) {
    vS = values[0];
    vE = values[0] + 1;
  }
  for (v = 1; v < nV; ++v) {
    vS = PetscMin(vS, values[v]);
    vE = PetscMax(vE, values[v] + 1);
  }
  PetscCall(PetscSectionCreate(PETSC_COMM_SELF, section));
  PetscCall(PetscSectionSetChart(*section, vS, vE));
  for (v = 0; v < nV; ++v) {
    PetscInt n;

    PetscCall(DMLabelGetStratumSize(label, values[v], &n));
    PetscCall(PetscSectionSetDof(*section, values[v], n));
  }
  PetscCall(PetscSectionSetUp(*section));
  PetscCall(PetscSectionGetStorageSize(*section, &N));
  PetscCall(PetscMalloc1(N, &points));
  for (v = 0; v < nV; ++v) {
    IS              is;
    const PetscInt *spoints;
    PetscInt        dof, off, p;

    PetscCall(PetscSectionGetDof(*section, values[v], &dof));
    PetscCall(PetscSectionGetOffset(*section, values[v], &off));
    PetscCall(DMLabelGetStratumIS(label, values[v], &is));
    PetscCall(ISGetIndices(is, &spoints));
    for (p = 0; p < dof; ++p) points[off + p] = spoints[p];
    PetscCall(ISRestoreIndices(is, &spoints));
    PetscCall(ISDestroy(&is));
  }
  PetscCall(ISRestoreIndices(vIS, &values));
  PetscCall(ISDestroy(&vIS));
  PetscCall(ISCreateGeneral(PETSC_COMM_SELF, N, points, PETSC_OWN_POINTER, is));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMLabelRegister - Adds a new label component implementation

  Not Collective

  Input Parameters:
+ name        - The name of a new user-defined creation routine
- create_func - The creation routine itself

  Notes:
  `DMLabelRegister()` may be called multiple times to add several user-defined labels

  Example Usage:
.vb
  DMLabelRegister("my_label", MyLabelCreate);
.ve

  Then, your label type can be chosen with the procedural interface via
.vb
  DMLabelCreate(MPI_Comm, DMLabel *);
  DMLabelSetType(DMLabel, "my_label");
.ve
  or at runtime via the option
.vb
  -dm_label_type my_label
.ve

  Level: advanced

.seealso: `DMLabel`, `DM`, `DMLabelType`, `DMLabelRegisterAll()`, `DMLabelRegisterDestroy()`
@*/
PetscErrorCode DMLabelRegister(const char name[], PetscErrorCode (*create_func)(DMLabel))
{
  PetscFunctionBegin;
  PetscCall(DMInitializePackage());
  PetscCall(PetscFunctionListAdd(&DMLabelList, name, create_func));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_EXTERN PetscErrorCode DMLabelCreate_Concrete(DMLabel);
PETSC_EXTERN PetscErrorCode DMLabelCreate_Ephemeral(DMLabel);

/*@C
  DMLabelRegisterAll - Registers all of the `DMLabel` implementations in the `DM` package.

  Not Collective

  Level: advanced

.seealso: `DMLabel`, `DM`, `DMRegisterAll()`, `DMLabelRegisterDestroy()`
@*/
PetscErrorCode DMLabelRegisterAll(void)
{
  PetscFunctionBegin;
  if (DMLabelRegisterAllCalled) PetscFunctionReturn(PETSC_SUCCESS);
  DMLabelRegisterAllCalled = PETSC_TRUE;

  PetscCall(DMLabelRegister(DMLABELCONCRETE, DMLabelCreate_Concrete));
  PetscCall(DMLabelRegister(DMLABELEPHEMERAL, DMLabelCreate_Ephemeral));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMLabelRegisterDestroy - This function destroys the `DMLabel` registry. It is called from `PetscFinalize()`.

  Level: developer

.seealso: `DMLabel`, `DM`, `PetscInitialize()`
@*/
PetscErrorCode DMLabelRegisterDestroy(void)
{
  PetscFunctionBegin;
  PetscCall(PetscFunctionListDestroy(&DMLabelList));
  DMLabelRegisterAllCalled = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMLabelSetType - Sets the particular implementation for a label.

  Collective

  Input Parameters:
+ label  - The label
- method - The name of the label type

  Options Database Key:
. -dm_label_type <type> - Sets the label type; use -help for a list of available types or see `DMLabelType`

  Level: intermediate

.seealso: `DMLabel`, `DM`, `DMLabelGetType()`, `DMLabelCreate()`
@*/
PetscErrorCode DMLabelSetType(DMLabel label, DMLabelType method)
{
  PetscErrorCode (*r)(DMLabel);
  PetscBool match;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 1);
  PetscCall(PetscObjectTypeCompare((PetscObject)label, method, &match));
  if (match) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(DMLabelRegisterAll());
  PetscCall(PetscFunctionListFind(DMLabelList, method, &r));
  PetscCheck(r, PetscObjectComm((PetscObject)label), PETSC_ERR_ARG_UNKNOWN_TYPE, "Unknown DMLabel type: %s", method);

  PetscTryTypeMethod(label, destroy);
  PetscCall(PetscMemzero(label->ops, sizeof(*label->ops)));
  PetscCall(PetscObjectChangeTypeName((PetscObject)label, method));
  PetscCall((*r)(label));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMLabelGetType - Gets the type name (as a string) from the label.

  Not Collective

  Input Parameter:
. label - The `DMLabel`

  Output Parameter:
. type - The `DMLabel` type name

  Level: intermediate

.seealso: `DMLabel`, `DM`, `DMLabelSetType()`, `DMLabelCreate()`
@*/
PetscErrorCode DMLabelGetType(DMLabel label, DMLabelType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 1);
  PetscAssertPointer(type, 2);
  PetscCall(DMLabelRegisterAll());
  *type = ((PetscObject)label)->type_name;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMLabelInitialize_Concrete(DMLabel label)
{
  PetscFunctionBegin;
  label->ops->view         = DMLabelView_Concrete;
  label->ops->setup        = NULL;
  label->ops->duplicate    = DMLabelDuplicate_Concrete;
  label->ops->getstratumis = DMLabelGetStratumIS_Concrete;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_EXTERN PetscErrorCode DMLabelCreate_Concrete(DMLabel label)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 1);
  PetscCall(DMLabelInitialize_Concrete(label));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscSectionCreateGlobalSectionLabel - Create a section describing the global field layout using
  the local section and an `PetscSF` describing the section point overlap.

  Collective

  Input Parameters:
+ s                  - The `PetscSection` for the local field layout
. sf                 - The `PetscSF` describing parallel layout of the section points
. includeConstraints - By default this is `PETSC_FALSE`, meaning that the global field vector will not possess constrained dofs
. label              - The label specifying the points
- labelValue         - The label stratum specifying the points

  Output Parameter:
. gsection - The `PetscSection` for the global field layout

  Level: developer

  Note:
  This gives negative sizes and offsets to points not owned by this process

.seealso: `DMLabel`, `DM`, `PetscSectionCreate()`
@*/
PetscErrorCode PetscSectionCreateGlobalSectionLabel(PetscSection s, PetscSF sf, PetscBool includeConstraints, DMLabel label, PetscInt labelValue, PetscSection *gsection)
{
  PetscInt *neg = NULL, *tmpOff = NULL;
  PetscInt  pStart, pEnd, p, dof, cdof, off, globalOff = 0, nroots;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(s, PETSC_SECTION_CLASSID, 1);
  PetscValidHeaderSpecific(sf, PETSCSF_CLASSID, 2);
  PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 4);
  PetscCall(PetscSectionCreate(PetscObjectComm((PetscObject)s), gsection));
  PetscCall(PetscSectionGetChart(s, &pStart, &pEnd));
  PetscCall(PetscSectionSetChart(*gsection, pStart, pEnd));
  PetscCall(PetscSFGetGraph(sf, &nroots, NULL, NULL, NULL));
  if (nroots >= 0) {
    PetscCheck(nroots >= pEnd - pStart, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "PetscSF nroots %" PetscInt_FMT " < %" PetscInt_FMT " section size", nroots, pEnd - pStart);
    PetscCall(PetscCalloc1(nroots, &neg));
    if (nroots > pEnd - pStart) {
      PetscCall(PetscCalloc1(nroots, &tmpOff));
    } else {
      tmpOff = &(*gsection)->atlasDof[-pStart];
    }
  }
  /* Mark ghost points with negative dof */
  for (p = pStart; p < pEnd; ++p) {
    PetscInt value;

    PetscCall(DMLabelGetValue(label, p, &value));
    if (value != labelValue) continue;
    PetscCall(PetscSectionGetDof(s, p, &dof));
    PetscCall(PetscSectionSetDof(*gsection, p, dof));
    PetscCall(PetscSectionGetConstraintDof(s, p, &cdof));
    if (!includeConstraints && cdof > 0) PetscCall(PetscSectionSetConstraintDof(*gsection, p, cdof));
    if (neg) neg[p] = -(dof + 1);
  }
  PetscCall(PetscSectionSetUpBC(*gsection));
  if (nroots >= 0) {
    PetscCall(PetscSFBcastBegin(sf, MPIU_INT, neg, tmpOff, MPI_REPLACE));
    PetscCall(PetscSFBcastEnd(sf, MPIU_INT, neg, tmpOff, MPI_REPLACE));
    if (nroots > pEnd - pStart) {
      for (p = pStart; p < pEnd; ++p) {
        if (tmpOff[p] < 0) (*gsection)->atlasDof[p - pStart] = tmpOff[p];
      }
    }
  }
  /* Calculate new sizes, get process offset, and calculate point offsets */
  for (p = 0, off = 0; p < pEnd - pStart; ++p) {
    cdof                     = (!includeConstraints && s->bc) ? s->bc->atlasDof[p] : 0;
    (*gsection)->atlasOff[p] = off;
    off += (*gsection)->atlasDof[p] > 0 ? (*gsection)->atlasDof[p] - cdof : 0;
  }
  PetscCallMPI(MPI_Scan(&off, &globalOff, 1, MPIU_INT, MPI_SUM, PetscObjectComm((PetscObject)s)));
  globalOff -= off;
  for (p = 0, off = 0; p < pEnd - pStart; ++p) {
    (*gsection)->atlasOff[p] += globalOff;
    if (neg) neg[p] = -((*gsection)->atlasOff[p] + 1);
  }
  /* Put in negative offsets for ghost points */
  if (nroots >= 0) {
    PetscCall(PetscSFBcastBegin(sf, MPIU_INT, neg, tmpOff, MPI_REPLACE));
    PetscCall(PetscSFBcastEnd(sf, MPIU_INT, neg, tmpOff, MPI_REPLACE));
    if (nroots > pEnd - pStart) {
      for (p = pStart; p < pEnd; ++p) {
        if (tmpOff[p] < 0) (*gsection)->atlasOff[p - pStart] = tmpOff[p];
      }
    }
  }
  if (nroots >= 0 && nroots > pEnd - pStart) PetscCall(PetscFree(tmpOff));
  PetscCall(PetscFree(neg));
  PetscFunctionReturn(PETSC_SUCCESS);
}

typedef struct _n_PetscSectionSym_Label {
  DMLabel              label;
  PetscCopyMode       *modes;
  PetscInt            *sizes;
  const PetscInt    ***perms;
  const PetscScalar ***rots;
  PetscInt (*minMaxOrients)[2];
  PetscInt numStrata; /* numStrata is only increasing, functions as a state */
} PetscSectionSym_Label;

static PetscErrorCode PetscSectionSymLabelReset(PetscSectionSym sym)
{
  PetscInt               i, j;
  PetscSectionSym_Label *sl = (PetscSectionSym_Label *)sym->data;

  PetscFunctionBegin;
  for (i = 0; i <= sl->numStrata; i++) {
    if (sl->modes[i] == PETSC_OWN_POINTER || sl->modes[i] == PETSC_COPY_VALUES) {
      for (j = sl->minMaxOrients[i][0]; j < sl->minMaxOrients[i][1]; j++) {
        if (sl->perms[i]) PetscCall(PetscFree(sl->perms[i][j]));
        if (sl->rots[i]) PetscCall(PetscFree(sl->rots[i][j]));
      }
      if (sl->perms[i]) {
        const PetscInt **perms = &sl->perms[i][sl->minMaxOrients[i][0]];

        PetscCall(PetscFree(perms));
      }
      if (sl->rots[i]) {
        const PetscScalar **rots = &sl->rots[i][sl->minMaxOrients[i][0]];

        PetscCall(PetscFree(rots));
      }
    }
  }
  PetscCall(PetscFree5(sl->modes, sl->sizes, sl->perms, sl->rots, sl->minMaxOrients));
  PetscCall(DMLabelDestroy(&sl->label));
  sl->numStrata = 0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscSectionSymDestroy_Label(PetscSectionSym sym)
{
  PetscFunctionBegin;
  PetscCall(PetscSectionSymLabelReset(sym));
  PetscCall(PetscFree(sym->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscSectionSymView_Label(PetscSectionSym sym, PetscViewer viewer)
{
  PetscSectionSym_Label *sl = (PetscSectionSym_Label *)sym->data;
  PetscBool              isAscii;
  DMLabel                label = sl->label;
  const char            *name;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isAscii));
  if (isAscii) {
    PetscInt          i, j, k;
    PetscViewerFormat format;

    PetscCall(PetscViewerGetFormat(viewer, &format));
    if (label) {
      PetscCall(PetscViewerGetFormat(viewer, &format));
      if (format == PETSC_VIEWER_ASCII_INFO_DETAIL) {
        PetscCall(PetscViewerASCIIPushTab(viewer));
        PetscCall(DMLabelView(label, viewer));
        PetscCall(PetscViewerASCIIPopTab(viewer));
      } else {
        PetscCall(PetscObjectGetName((PetscObject)sl->label, &name));
        PetscCall(PetscViewerASCIIPrintf(viewer, "  Label '%s'\n", name));
      }
    } else {
      PetscCall(PetscViewerASCIIPrintf(viewer, "No label given\n"));
    }
    PetscCall(PetscViewerASCIIPushTab(viewer));
    for (i = 0; i <= sl->numStrata; i++) {
      PetscInt value = i < sl->numStrata ? label->stratumValues[i] : label->defaultValue;

      if (!(sl->perms[i] || sl->rots[i])) {
        PetscCall(PetscViewerASCIIPrintf(viewer, "Symmetry for stratum value %" PetscInt_FMT " (%" PetscInt_FMT " dofs per point): no symmetries\n", value, sl->sizes[i]));
      } else {
        PetscCall(PetscViewerASCIIPrintf(viewer, "Symmetry for stratum value %" PetscInt_FMT " (%" PetscInt_FMT " dofs per point):\n", value, sl->sizes[i]));
        PetscCall(PetscViewerASCIIPushTab(viewer));
        PetscCall(PetscViewerASCIIPrintf(viewer, "Orientation range: [%" PetscInt_FMT ", %" PetscInt_FMT ")\n", sl->minMaxOrients[i][0], sl->minMaxOrients[i][1]));
        if (format == PETSC_VIEWER_ASCII_INFO_DETAIL) {
          PetscCall(PetscViewerASCIIPushTab(viewer));
          for (j = sl->minMaxOrients[i][0]; j < sl->minMaxOrients[i][1]; j++) {
            if (!((sl->perms[i] && sl->perms[i][j]) || (sl->rots[i] && sl->rots[i][j]))) {
              PetscCall(PetscViewerASCIIPrintf(viewer, "Orientation %" PetscInt_FMT ": identity\n", j));
            } else {
              PetscInt tab;

              PetscCall(PetscViewerASCIIPrintf(viewer, "Orientation %" PetscInt_FMT ":\n", j));
              PetscCall(PetscViewerASCIIPushTab(viewer));
              PetscCall(PetscViewerASCIIGetTab(viewer, &tab));
              if (sl->perms[i] && sl->perms[i][j]) {
                PetscCall(PetscViewerASCIIPrintf(viewer, "Permutation:"));
                PetscCall(PetscViewerASCIISetTab(viewer, 0));
                for (k = 0; k < sl->sizes[i]; k++) PetscCall(PetscViewerASCIIPrintf(viewer, " %" PetscInt_FMT, sl->perms[i][j][k]));
                PetscCall(PetscViewerASCIIPrintf(viewer, "\n"));
                PetscCall(PetscViewerASCIISetTab(viewer, tab));
              }
              if (sl->rots[i] && sl->rots[i][j]) {
                PetscCall(PetscViewerASCIIPrintf(viewer, "Rotations:  "));
                PetscCall(PetscViewerASCIISetTab(viewer, 0));
#if defined(PETSC_USE_COMPLEX)
                for (k = 0; k < sl->sizes[i]; k++) PetscCall(PetscViewerASCIIPrintf(viewer, " %+g+i*%+g", (double)PetscRealPart(sl->rots[i][j][k]), (double)PetscImaginaryPart(sl->rots[i][j][k])));
#else
                for (k = 0; k < sl->sizes[i]; k++) PetscCall(PetscViewerASCIIPrintf(viewer, " %+g", (double)sl->rots[i][j][k]));
#endif
                PetscCall(PetscViewerASCIIPrintf(viewer, "\n"));
                PetscCall(PetscViewerASCIISetTab(viewer, tab));
              }
              PetscCall(PetscViewerASCIIPopTab(viewer));
            }
          }
          PetscCall(PetscViewerASCIIPopTab(viewer));
        }
        PetscCall(PetscViewerASCIIPopTab(viewer));
      }
    }
    PetscCall(PetscViewerASCIIPopTab(viewer));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscSectionSymLabelSetLabel - set the label whose strata will define the points that receive symmetries

  Logically

  Input Parameters:
+ sym   - the section symmetries
- label - the `DMLabel` describing the types of points

  Level: developer:

.seealso: `DMLabel`, `DM`, `PetscSectionSymLabelSetStratum()`, `PetscSectionSymCreateLabel()`, `PetscSectionGetPointSyms()`
@*/
PetscErrorCode PetscSectionSymLabelSetLabel(PetscSectionSym sym, DMLabel label)
{
  PetscSectionSym_Label *sl;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sym, PETSC_SECTION_SYM_CLASSID, 1);
  sl = (PetscSectionSym_Label *)sym->data;
  if (sl->label && sl->label != label) PetscCall(PetscSectionSymLabelReset(sym));
  if (label) {
    sl->label = label;
    PetscCall(PetscObjectReference((PetscObject)label));
    PetscCall(DMLabelGetNumValues(label, &sl->numStrata));
    PetscCall(PetscMalloc5(sl->numStrata + 1, &sl->modes, sl->numStrata + 1, &sl->sizes, sl->numStrata + 1, &sl->perms, sl->numStrata + 1, &sl->rots, sl->numStrata + 1, &sl->minMaxOrients));
    PetscCall(PetscMemzero((void *)sl->modes, (sl->numStrata + 1) * sizeof(PetscCopyMode)));
    PetscCall(PetscMemzero((void *)sl->sizes, (sl->numStrata + 1) * sizeof(PetscInt)));
    PetscCall(PetscMemzero((void *)sl->perms, (sl->numStrata + 1) * sizeof(const PetscInt **)));
    PetscCall(PetscMemzero((void *)sl->rots, (sl->numStrata + 1) * sizeof(const PetscScalar **)));
    PetscCall(PetscMemzero((void *)sl->minMaxOrients, (sl->numStrata + 1) * sizeof(PetscInt[2])));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscSectionSymLabelGetStratum - get the symmetries for the orientations of a stratum

  Logically Collective

  Input Parameters:
+ sym     - the section symmetries
- stratum - the stratum value in the label that we are assigning symmetries for

  Output Parameters:
+ size      - the number of dofs for points in the `stratum` of the label
. minOrient - the smallest orientation for a point in this `stratum`
. maxOrient - one greater than the largest orientation for a ppoint in this `stratum` (i.e., orientations are in the range [`minOrient`, `maxOrient`))
. perms     - `NULL` if there are no permutations, or (`maxOrient` - `minOrient`) permutations, one for each orientation.  A `NULL` permutation is the identity
- rots      - `NULL` if there are no rotations, or (`maxOrient` - `minOrient`) sets of rotations, one for each orientation.  A `NULL` set of orientations is the identity

  Level: developer

.seealso: `DMLabel`, `DM`, `PetscSectionSymLabelSetStratum()`, `PetscSectionSymCreate()`, `PetscSectionSetSym()`, `PetscSectionGetPointSyms()`, `PetscSectionSymCreateLabel()`
@*/
PetscErrorCode PetscSectionSymLabelGetStratum(PetscSectionSym sym, PetscInt stratum, PetscInt *size, PetscInt *minOrient, PetscInt *maxOrient, const PetscInt ***perms, const PetscScalar ***rots)
{
  PetscSectionSym_Label *sl;
  const char            *name;
  PetscInt               i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sym, PETSC_SECTION_SYM_CLASSID, 1);
  sl = (PetscSectionSym_Label *)sym->data;
  PetscCheck(sl->label, PetscObjectComm((PetscObject)sym), PETSC_ERR_ARG_WRONGSTATE, "No label set yet");
  for (i = 0; i <= sl->numStrata; i++) {
    PetscInt value = (i < sl->numStrata) ? sl->label->stratumValues[i] : sl->label->defaultValue;

    if (stratum == value) break;
  }
  PetscCall(PetscObjectGetName((PetscObject)sl->label, &name));
  PetscCheck(i <= sl->numStrata, PetscObjectComm((PetscObject)sym), PETSC_ERR_ARG_OUTOFRANGE, "Stratum %" PetscInt_FMT " not found in label %s", stratum, name);
  if (size) {
    PetscAssertPointer(size, 3);
    *size = sl->sizes[i];
  }
  if (minOrient) {
    PetscAssertPointer(minOrient, 4);
    *minOrient = sl->minMaxOrients[i][0];
  }
  if (maxOrient) {
    PetscAssertPointer(maxOrient, 5);
    *maxOrient = sl->minMaxOrients[i][1];
  }
  if (perms) {
    PetscAssertPointer(perms, 6);
    *perms = PetscSafePointerPlusOffset(sl->perms[i], sl->minMaxOrients[i][0]);
  }
  if (rots) {
    PetscAssertPointer(rots, 7);
    *rots = PetscSafePointerPlusOffset(sl->rots[i], sl->minMaxOrients[i][0]);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscSectionSymLabelSetStratum - set the symmetries for the orientations of a stratum

  Logically

  Input Parameters:
+ sym       - the section symmetries
. stratum   - the stratum value in the label that we are assigning symmetries for
. size      - the number of dofs for points in the `stratum` of the label
. minOrient - the smallest orientation for a point in this `stratum`
. maxOrient - one greater than the largest orientation for a point in this `stratum` (i.e., orientations are in the range [`minOrient`, `maxOrient`))
. mode      - how `sym` should copy the `perms` and `rots` arrays
. perms     - `NULL` if there are no permutations, or (`maxOrient` - `minOrient`) permutations, one for each orientation.  A `NULL` permutation is the identity
- rots      - `NULL` if there are no rotations, or (`maxOrient` - `minOrient`) sets of rotations, one for each orientation.  A `NULL` set of orientations is the identity

  Level: developer

.seealso: `DMLabel`, `DM`, `PetscSectionSymLabelGetStratum()`, `PetscSectionSymCreate()`, `PetscSectionSetSym()`, `PetscSectionGetPointSyms()`, `PetscSectionSymCreateLabel()`
@*/
PetscErrorCode PetscSectionSymLabelSetStratum(PetscSectionSym sym, PetscInt stratum, PetscInt size, PetscInt minOrient, PetscInt maxOrient, PetscCopyMode mode, const PetscInt **perms, const PetscScalar **rots)
{
  PetscSectionSym_Label *sl;
  const char            *name;
  PetscInt               i, j, k;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sym, PETSC_SECTION_SYM_CLASSID, 1);
  sl = (PetscSectionSym_Label *)sym->data;
  PetscCheck(sl->label, PetscObjectComm((PetscObject)sym), PETSC_ERR_ARG_WRONGSTATE, "No label set yet");
  for (i = 0; i <= sl->numStrata; i++) {
    PetscInt value = (i < sl->numStrata) ? sl->label->stratumValues[i] : sl->label->defaultValue;

    if (stratum == value) break;
  }
  PetscCall(PetscObjectGetName((PetscObject)sl->label, &name));
  PetscCheck(i <= sl->numStrata, PetscObjectComm((PetscObject)sym), PETSC_ERR_ARG_OUTOFRANGE, "Stratum %" PetscInt_FMT " not found in label %s", stratum, name);
  sl->sizes[i]            = size;
  sl->modes[i]            = mode;
  sl->minMaxOrients[i][0] = minOrient;
  sl->minMaxOrients[i][1] = maxOrient;
  if (mode == PETSC_COPY_VALUES) {
    if (perms) {
      PetscInt **ownPerms;

      PetscCall(PetscCalloc1(maxOrient - minOrient, &ownPerms));
      for (j = 0; j < maxOrient - minOrient; j++) {
        if (perms[j]) {
          PetscCall(PetscMalloc1(size, &ownPerms[j]));
          for (k = 0; k < size; k++) ownPerms[j][k] = perms[j][k];
        }
      }
      sl->perms[i] = (const PetscInt **)&ownPerms[-minOrient];
    }
    if (rots) {
      PetscScalar **ownRots;

      PetscCall(PetscCalloc1(maxOrient - minOrient, &ownRots));
      for (j = 0; j < maxOrient - minOrient; j++) {
        if (rots[j]) {
          PetscCall(PetscMalloc1(size, &ownRots[j]));
          for (k = 0; k < size; k++) ownRots[j][k] = rots[j][k];
        }
      }
      sl->rots[i] = (const PetscScalar **)&ownRots[-minOrient];
    }
  } else {
    sl->perms[i] = PetscSafePointerPlusOffset(perms, -minOrient);
    sl->rots[i]  = PetscSafePointerPlusOffset(rots, -minOrient);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscSectionSymGetPoints_Label(PetscSectionSym sym, PetscSection section, PetscInt numPoints, const PetscInt *points, const PetscInt **perms, const PetscScalar **rots)
{
  PetscInt               i, j, numStrata;
  PetscSectionSym_Label *sl;
  DMLabel                label;

  PetscFunctionBegin;
  sl        = (PetscSectionSym_Label *)sym->data;
  numStrata = sl->numStrata;
  label     = sl->label;
  for (i = 0; i < numPoints; i++) {
    PetscInt point = points[2 * i];
    PetscInt ornt  = points[2 * i + 1];

    for (j = 0; j < numStrata; j++) {
      if (label->validIS[j]) {
        PetscInt k;

        PetscCall(ISLocate(label->points[j], point, &k));
        if (k >= 0) break;
      } else {
        PetscBool has;

        PetscCall(PetscHSetIHas(label->ht[j], point, &has));
        if (has) break;
      }
    }
    PetscCheck(!(sl->minMaxOrients[j][1] > sl->minMaxOrients[j][0]) || !(ornt < sl->minMaxOrients[j][0] || ornt >= sl->minMaxOrients[j][1]), PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "point %" PetscInt_FMT " orientation %" PetscInt_FMT " not in range [%" PetscInt_FMT ", %" PetscInt_FMT ") for stratum %" PetscInt_FMT, point, ornt, sl->minMaxOrients[j][0], sl->minMaxOrients[j][1],
               j < numStrata ? label->stratumValues[j] : label->defaultValue);
    if (perms) perms[i] = sl->perms[j] ? sl->perms[j][ornt] : NULL;
    if (rots) rots[i] = sl->rots[j] ? sl->rots[j][ornt] : NULL;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscSectionSymCopy_Label(PetscSectionSym sym, PetscSectionSym nsym)
{
  PetscSectionSym_Label *sl = (PetscSectionSym_Label *)nsym->data;
  IS                     valIS;
  const PetscInt        *values;
  PetscInt               Nv, v;

  PetscFunctionBegin;
  PetscCall(DMLabelGetNumValues(sl->label, &Nv));
  PetscCall(DMLabelGetValueIS(sl->label, &valIS));
  PetscCall(ISGetIndices(valIS, &values));
  for (v = 0; v < Nv; ++v) {
    const PetscInt      val = values[v];
    PetscInt            size, minOrient, maxOrient;
    const PetscInt    **perms;
    const PetscScalar **rots;

    PetscCall(PetscSectionSymLabelGetStratum(sym, val, &size, &minOrient, &maxOrient, &perms, &rots));
    PetscCall(PetscSectionSymLabelSetStratum(nsym, val, size, minOrient, maxOrient, PETSC_COPY_VALUES, perms, rots));
  }
  PetscCall(ISDestroy(&valIS));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscSectionSymDistribute_Label(PetscSectionSym sym, PetscSF migrationSF, PetscSectionSym *dsym)
{
  PetscSectionSym_Label *sl = (PetscSectionSym_Label *)sym->data;
  DMLabel                dlabel;

  PetscFunctionBegin;
  PetscCall(DMLabelDistribute(sl->label, migrationSF, &dlabel));
  PetscCall(PetscSectionSymCreateLabel(PetscObjectComm((PetscObject)sym), dlabel, dsym));
  PetscCall(DMLabelDestroy(&dlabel));
  PetscCall(PetscSectionSymCopy(sym, *dsym));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscSectionSymCreate_Label(PetscSectionSym sym)
{
  PetscSectionSym_Label *sl;

  PetscFunctionBegin;
  PetscCall(PetscNew(&sl));
  sym->ops->getpoints  = PetscSectionSymGetPoints_Label;
  sym->ops->distribute = PetscSectionSymDistribute_Label;
  sym->ops->copy       = PetscSectionSymCopy_Label;
  sym->ops->view       = PetscSectionSymView_Label;
  sym->ops->destroy    = PetscSectionSymDestroy_Label;
  sym->data            = (void *)sl;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscSectionSymCreateLabel - Create a section symmetry that assigns one symmetry to each stratum of a label

  Collective

  Input Parameters:
+ comm  - the MPI communicator for the new symmetry
- label - the label defining the strata

  Output Parameter:
. sym - the section symmetries

  Level: developer

.seealso: `DMLabel`, `DM`, `PetscSectionSymCreate()`, `PetscSectionSetSym()`, `PetscSectionGetSym()`, `PetscSectionSymLabelSetStratum()`, `PetscSectionGetPointSyms()`
@*/
PetscErrorCode PetscSectionSymCreateLabel(MPI_Comm comm, DMLabel label, PetscSectionSym *sym)
{
  PetscFunctionBegin;
  PetscCall(DMInitializePackage());
  PetscCall(PetscSectionSymCreate(comm, sym));
  PetscCall(PetscSectionSymSetType(*sym, PETSCSECTIONSYMLABEL));
  PetscCall(PetscSectionSymLabelSetLabel(*sym, label));
  PetscFunctionReturn(PETSC_SUCCESS);
}
