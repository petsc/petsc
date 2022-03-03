#include <petscdm.h>
#include <petsc/private/dmlabelimpl.h>   /*I      "petscdmlabel.h"   I*/
#include <petsc/private/sectionimpl.h>   /*I      "petscsection.h"   I*/
#include <petscsf.h>
#include <petscsection.h>

/*@C
  DMLabelCreate - Create a DMLabel object, which is a multimap

  Collective

  Input parameters:
+ comm - The communicator, usually PETSC_COMM_SELF
- name - The label name

  Output parameter:
. label - The DMLabel

  Level: beginner

  Notes:
  The label name is actually usual PetscObject name.
  One can get/set it with PetscObjectGetName()/PetscObjectSetName().

.seealso: DMLabelDestroy()
@*/
PetscErrorCode DMLabelCreate(MPI_Comm comm, const char name[], DMLabel *label)
{
  PetscFunctionBegin;
  PetscValidPointer(label,3);
  CHKERRQ(DMInitializePackage());

  CHKERRQ(PetscHeaderCreate(*label,DMLABEL_CLASSID,"DMLabel","DMLabel","DM",comm,DMLabelDestroy,DMLabelView));

  (*label)->numStrata      = 0;
  (*label)->defaultValue   = -1;
  (*label)->stratumValues  = NULL;
  (*label)->validIS        = NULL;
  (*label)->stratumSizes   = NULL;
  (*label)->points         = NULL;
  (*label)->ht             = NULL;
  (*label)->pStart         = -1;
  (*label)->pEnd           = -1;
  (*label)->bt             = NULL;
  CHKERRQ(PetscHMapICreate(&(*label)->hmap));
  CHKERRQ(PetscObjectSetName((PetscObject) *label, name));
  PetscFunctionReturn(0);
}

/*
  DMLabelMakeValid_Private - Transfer stratum data from the hash format to the sorted list format

  Not collective

  Input parameter:
+ label - The DMLabel
- v - The stratum value

  Output parameter:
. label - The DMLabel with stratum in sorted list format

  Level: developer

.seealso: DMLabelCreate()
*/
static PetscErrorCode DMLabelMakeValid_Private(DMLabel label, PetscInt v)
{
  IS             is;
  PetscInt       off = 0, *pointArray, p;

  if (PetscLikely(v >= 0 && v < label->numStrata) && label->validIS[v]) return 0;
  PetscFunctionBegin;
  PetscCheckFalse(v < 0 || v >= label->numStrata,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Trying to access invalid stratum %D in DMLabelMakeValid_Private", v);
  CHKERRQ(PetscHSetIGetSize(label->ht[v], &label->stratumSizes[v]));
  CHKERRQ(PetscMalloc1(label->stratumSizes[v], &pointArray));
  CHKERRQ(PetscHSetIGetElems(label->ht[v], &off, pointArray));
  CHKERRQ(PetscHSetIClear(label->ht[v]));
  CHKERRQ(PetscSortInt(label->stratumSizes[v], pointArray));
  if (label->bt) {
    for (p = 0; p < label->stratumSizes[v]; ++p) {
      const PetscInt point = pointArray[p];
      PetscCheckFalse((point < label->pStart) || (point >= label->pEnd),PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Label point %D is not in [%D, %D)", point, label->pStart, label->pEnd);
      CHKERRQ(PetscBTSet(label->bt, point - label->pStart));
    }
  }
  if (label->stratumSizes[v] > 0 && pointArray[label->stratumSizes[v]-1] == pointArray[0] + label->stratumSizes[v]-1) {
    CHKERRQ(ISCreateStride(PETSC_COMM_SELF, label->stratumSizes[v], pointArray[0], 1, &is));
    CHKERRQ(PetscFree(pointArray));
  } else {
    CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF, label->stratumSizes[v], pointArray, PETSC_OWN_POINTER, &is));
  }
  CHKERRQ(PetscObjectSetName((PetscObject) is, "indices"));
  label->points[v]  = is;
  label->validIS[v] = PETSC_TRUE;
  CHKERRQ(PetscObjectStateIncrease((PetscObject) label));
  PetscFunctionReturn(0);
}

/*
  DMLabelMakeAllValid_Private - Transfer all strata from the hash format to the sorted list format

  Not collective

  Input parameter:
. label - The DMLabel

  Output parameter:
. label - The DMLabel with all strata in sorted list format

  Level: developer

.seealso: DMLabelCreate()
*/
static PetscErrorCode DMLabelMakeAllValid_Private(DMLabel label)
{
  PetscInt       v;

  PetscFunctionBegin;
  for (v = 0; v < label->numStrata; v++) {
    CHKERRQ(DMLabelMakeValid_Private(label, v));
  }
  PetscFunctionReturn(0);
}

/*
  DMLabelMakeInvalid_Private - Transfer stratum data from the sorted list format to the hash format

  Not collective

  Input parameter:
+ label - The DMLabel
- v - The stratum value

  Output parameter:
. label - The DMLabel with stratum in hash format

  Level: developer

.seealso: DMLabelCreate()
*/
static PetscErrorCode DMLabelMakeInvalid_Private(DMLabel label, PetscInt v)
{
  PetscInt       p;
  const PetscInt *points;

  if (PetscLikely(v >= 0 && v < label->numStrata) && !label->validIS[v]) return 0;
  PetscFunctionBegin;
  PetscCheckFalse(v < 0 || v >= label->numStrata,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Trying to access invalid stratum %D in DMLabelMakeInvalid_Private", v);
  if (label->points[v]) {
    CHKERRQ(ISGetIndices(label->points[v], &points));
    for (p = 0; p < label->stratumSizes[v]; ++p) {
      CHKERRQ(PetscHSetIAdd(label->ht[v], points[p]));
    }
    CHKERRQ(ISRestoreIndices(label->points[v],&points));
    CHKERRQ(ISDestroy(&(label->points[v])));
  }
  label->validIS[v] = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#if !defined(DMLABEL_LOOKUP_THRESHOLD)
#define DMLABEL_LOOKUP_THRESHOLD 16
#endif

static inline PetscErrorCode DMLabelLookupStratum(DMLabel label, PetscInt value, PetscInt *index)
{
  PetscInt       v;

  PetscFunctionBegin;
  *index = -1;
  if (label->numStrata <= DMLABEL_LOOKUP_THRESHOLD) {
    for (v = 0; v < label->numStrata; ++v)
      if (label->stratumValues[v] == value) {*index = v; break;}
  } else {
    CHKERRQ(PetscHMapIGet(label->hmap, value, index));
  }
  if (PetscDefined(USE_DEBUG)) { /* Check strata hash map consistency */
    PetscInt len, loc = -1;
    CHKERRQ(PetscHMapIGetSize(label->hmap, &len));
    PetscCheckFalse(len != label->numStrata,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Inconsistent strata hash map size");
    if (label->numStrata <= DMLABEL_LOOKUP_THRESHOLD) {
      CHKERRQ(PetscHMapIGet(label->hmap, value, &loc));
    } else {
      for (v = 0; v < label->numStrata; ++v)
        if (label->stratumValues[v] == value) {loc = v; break;}
    }
    PetscCheckFalse(loc != *index,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Inconsistent strata hash map lookup");
  }
  PetscFunctionReturn(0);
}

static inline PetscErrorCode DMLabelNewStratum(DMLabel label, PetscInt value, PetscInt *index)
{
  PetscInt       v;
  PetscInt      *tmpV;
  PetscInt      *tmpS;
  PetscHSetI    *tmpH, ht;
  IS            *tmpP, is;
  PetscBool     *tmpB;
  PetscHMapI     hmap = label->hmap;

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
    CHKERRQ(PetscMalloc((v+1)*sizeof(*tmpV), &tmpV));
    CHKERRQ(PetscMalloc((v+1)*sizeof(*tmpS), &tmpS));
    CHKERRQ(PetscMalloc((v+1)*sizeof(*tmpH), &tmpH));
    CHKERRQ(PetscMalloc((v+1)*sizeof(*tmpP), &tmpP));
    CHKERRQ(PetscMalloc((v+1)*sizeof(*tmpB), &tmpB));
    CHKERRQ(PetscArraycpy(tmpV, oldV, v));
    CHKERRQ(PetscArraycpy(tmpS, oldS, v));
    CHKERRQ(PetscArraycpy(tmpH, oldH, v));
    CHKERRQ(PetscArraycpy(tmpP, oldP, v));
    CHKERRQ(PetscArraycpy(tmpB, oldB, v));
    CHKERRQ(PetscFree(oldV));
    CHKERRQ(PetscFree(oldS));
    CHKERRQ(PetscFree(oldH));
    CHKERRQ(PetscFree(oldP));
    CHKERRQ(PetscFree(oldB));
  }
  label->numStrata     = v+1;
  label->stratumValues = tmpV;
  label->stratumSizes  = tmpS;
  label->ht            = tmpH;
  label->points        = tmpP;
  label->validIS       = tmpB;
  CHKERRQ(PetscHSetICreate(&ht));
  CHKERRQ(ISCreateStride(PETSC_COMM_SELF,0,0,1,&is));
  CHKERRQ(PetscHMapISet(hmap, value, v));
  tmpV[v] = value;
  tmpS[v] = 0;
  tmpH[v] = ht;
  tmpP[v] = is;
  tmpB[v] = PETSC_TRUE;
  CHKERRQ(PetscObjectStateIncrease((PetscObject) label));
  *index = v;
  PetscFunctionReturn(0);
}

static inline PetscErrorCode DMLabelLookupAddStratum(DMLabel label, PetscInt value, PetscInt *index)
{
  PetscFunctionBegin;
  CHKERRQ(DMLabelLookupStratum(label, value, index));
  if (*index < 0) CHKERRQ(DMLabelNewStratum(label, value, index));
  PetscFunctionReturn(0);
}

static inline PetscErrorCode DMLabelGetStratumSize_Private(DMLabel label, PetscInt v, PetscInt *size)
{
  PetscFunctionBegin;
  *size = 0;
  if (v < 0) PetscFunctionReturn(0);
  if (label->validIS[v]) {
    *size = label->stratumSizes[v];
  } else {
    CHKERRQ(PetscHSetIGetSize(label->ht[v], size));
  }
  PetscFunctionReturn(0);
}

/*@
  DMLabelAddStratum - Adds a new stratum value in a DMLabel

  Input Parameters:
+ label - The DMLabel
- value - The stratum value

  Level: beginner

.seealso:  DMLabelCreate(), DMLabelDestroy()
@*/
PetscErrorCode DMLabelAddStratum(DMLabel label, PetscInt value)
{
  PetscInt       v;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 1);
  CHKERRQ(DMLabelLookupAddStratum(label, value, &v));
  PetscFunctionReturn(0);
}

/*@
  DMLabelAddStrata - Adds new stratum values in a DMLabel

  Not collective

  Input Parameters:
+ label - The DMLabel
. numStrata - The number of stratum values
- stratumValues - The stratum values

  Level: beginner

.seealso:  DMLabelCreate(), DMLabelDestroy()
@*/
PetscErrorCode DMLabelAddStrata(DMLabel label, PetscInt numStrata, const PetscInt stratumValues[])
{
  PetscInt       *values, v;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 1);
  if (numStrata) PetscValidIntPointer(stratumValues, 3);
  CHKERRQ(PetscMalloc1(numStrata, &values));
  CHKERRQ(PetscArraycpy(values, stratumValues, numStrata));
  CHKERRQ(PetscSortRemoveDupsInt(&numStrata, values));
  if (!label->numStrata) { /* Fast preallocation */
    PetscInt   *tmpV;
    PetscInt   *tmpS;
    PetscHSetI *tmpH, ht;
    IS         *tmpP, is;
    PetscBool  *tmpB;
    PetscHMapI  hmap = label->hmap;

    CHKERRQ(PetscMalloc1(numStrata, &tmpV));
    CHKERRQ(PetscMalloc1(numStrata, &tmpS));
    CHKERRQ(PetscMalloc1(numStrata, &tmpH));
    CHKERRQ(PetscMalloc1(numStrata, &tmpP));
    CHKERRQ(PetscMalloc1(numStrata, &tmpB));
    label->numStrata     = numStrata;
    label->stratumValues = tmpV;
    label->stratumSizes  = tmpS;
    label->ht            = tmpH;
    label->points        = tmpP;
    label->validIS       = tmpB;
    for (v = 0; v < numStrata; ++v) {
      CHKERRQ(PetscHSetICreate(&ht));
      CHKERRQ(ISCreateStride(PETSC_COMM_SELF,0,0,1,&is));
      CHKERRQ(PetscHMapISet(hmap, values[v], v));
      tmpV[v] = values[v];
      tmpS[v] = 0;
      tmpH[v] = ht;
      tmpP[v] = is;
      tmpB[v] = PETSC_TRUE;
    }
    CHKERRQ(PetscObjectStateIncrease((PetscObject) label));
  } else {
    for (v = 0; v < numStrata; ++v) {
      CHKERRQ(DMLabelAddStratum(label, values[v]));
    }
  }
  CHKERRQ(PetscFree(values));
  PetscFunctionReturn(0);
}

/*@
  DMLabelAddStrataIS - Adds new stratum values in a DMLabel

  Not collective

  Input Parameters:
+ label - The DMLabel
- valueIS - Index set with stratum values

  Level: beginner

.seealso:  DMLabelCreate(), DMLabelDestroy()
@*/
PetscErrorCode DMLabelAddStrataIS(DMLabel label, IS valueIS)
{
  PetscInt       numStrata;
  const PetscInt *stratumValues;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 1);
  PetscValidHeaderSpecific(valueIS, IS_CLASSID, 2);
  CHKERRQ(ISGetLocalSize(valueIS, &numStrata));
  CHKERRQ(ISGetIndices(valueIS, &stratumValues));
  CHKERRQ(DMLabelAddStrata(label, numStrata, stratumValues));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMLabelView_Ascii(DMLabel label, PetscViewer viewer)
{
  PetscInt       v;
  PetscMPIInt    rank;

  PetscFunctionBegin;
  CHKERRMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)viewer), &rank));
  CHKERRQ(PetscViewerASCIIPushSynchronized(viewer));
  if (label) {
    const char *name;

    CHKERRQ(PetscObjectGetName((PetscObject) label, &name));
    CHKERRQ(PetscViewerASCIIPrintf(viewer, "Label '%s':\n", name));
    if (label->bt) CHKERRQ(PetscViewerASCIIPrintf(viewer, "  Index has been calculated in [%D, %D)\n", label->pStart, label->pEnd));
    for (v = 0; v < label->numStrata; ++v) {
      const PetscInt value = label->stratumValues[v];
      const PetscInt *points;
      PetscInt       p;

      CHKERRQ(ISGetIndices(label->points[v], &points));
      for (p = 0; p < label->stratumSizes[v]; ++p) {
        CHKERRQ(PetscViewerASCIISynchronizedPrintf(viewer, "[%d]: %D (%D)\n", rank, points[p], value));
      }
      CHKERRQ(ISRestoreIndices(label->points[v],&points));
    }
  }
  CHKERRQ(PetscViewerFlush(viewer));
  CHKERRQ(PetscViewerASCIIPopSynchronized(viewer));
  PetscFunctionReturn(0);
}

/*@C
  DMLabelView - View the label

  Collective on viewer

  Input Parameters:
+ label - The DMLabel
- viewer - The PetscViewer

  Level: intermediate

.seealso: DMLabelCreate(), DMLabelDestroy()
@*/
PetscErrorCode DMLabelView(DMLabel label, PetscViewer viewer)
{
  PetscBool      iascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 1);
  if (!viewer) CHKERRQ(PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)label), &viewer));
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  if (label) CHKERRQ(DMLabelMakeAllValid_Private(label));
  CHKERRQ(PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &iascii));
  if (iascii) {
    CHKERRQ(DMLabelView_Ascii(label, viewer));
  }
  PetscFunctionReturn(0);
}

/*@
  DMLabelReset - Destroys internal data structures in a DMLabel

  Not collective

  Input Parameter:
. label - The DMLabel

  Level: beginner

.seealso: DMLabelDestroy(), DMLabelCreate()
@*/
PetscErrorCode DMLabelReset(DMLabel label)
{
  PetscInt       v;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 1);
  for (v = 0; v < label->numStrata; ++v) {
    CHKERRQ(PetscHSetIDestroy(&label->ht[v]));
    CHKERRQ(ISDestroy(&label->points[v]));
  }
  label->numStrata = 0;
  CHKERRQ(PetscFree(label->stratumValues));
  CHKERRQ(PetscFree(label->stratumSizes));
  CHKERRQ(PetscFree(label->ht));
  CHKERRQ(PetscFree(label->points));
  CHKERRQ(PetscFree(label->validIS));
  label->stratumValues = NULL;
  label->stratumSizes  = NULL;
  label->ht            = NULL;
  label->points        = NULL;
  label->validIS       = NULL;
  CHKERRQ(PetscHMapIReset(label->hmap));
  label->pStart = -1;
  label->pEnd   = -1;
  CHKERRQ(PetscBTDestroy(&label->bt));
  PetscFunctionReturn(0);
}

/*@
  DMLabelDestroy - Destroys a DMLabel

  Collective on label

  Input Parameter:
. label - The DMLabel

  Level: beginner

.seealso: DMLabelReset(), DMLabelCreate()
@*/
PetscErrorCode DMLabelDestroy(DMLabel *label)
{
  PetscFunctionBegin;
  if (!*label) PetscFunctionReturn(0);
  PetscValidHeaderSpecific((*label),DMLABEL_CLASSID,1);
  if (--((PetscObject)(*label))->refct > 0) {*label = NULL; PetscFunctionReturn(0);}
  CHKERRQ(DMLabelReset(*label));
  CHKERRQ(PetscHMapIDestroy(&(*label)->hmap));
  CHKERRQ(PetscHeaderDestroy(label));
  PetscFunctionReturn(0);
}

/*@
  DMLabelDuplicate - Duplicates a DMLabel

  Collective on label

  Input Parameter:
. label - The DMLabel

  Output Parameter:
. labelnew - location to put new vector

  Level: intermediate

.seealso: DMLabelCreate(), DMLabelDestroy()
@*/
PetscErrorCode DMLabelDuplicate(DMLabel label, DMLabel *labelnew)
{
  const char    *name;
  PetscInt       v;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 1);
  CHKERRQ(DMLabelMakeAllValid_Private(label));
  CHKERRQ(PetscObjectGetName((PetscObject) label, &name));
  CHKERRQ(DMLabelCreate(PetscObjectComm((PetscObject) label), name, labelnew));

  (*labelnew)->numStrata    = label->numStrata;
  (*labelnew)->defaultValue = label->defaultValue;
  CHKERRQ(PetscMalloc1(label->numStrata, &(*labelnew)->stratumValues));
  CHKERRQ(PetscMalloc1(label->numStrata, &(*labelnew)->stratumSizes));
  CHKERRQ(PetscMalloc1(label->numStrata, &(*labelnew)->ht));
  CHKERRQ(PetscMalloc1(label->numStrata, &(*labelnew)->points));
  CHKERRQ(PetscMalloc1(label->numStrata, &(*labelnew)->validIS));
  for (v = 0; v < label->numStrata; ++v) {
    CHKERRQ(PetscHSetICreate(&(*labelnew)->ht[v]));
    (*labelnew)->stratumValues[v]  = label->stratumValues[v];
    (*labelnew)->stratumSizes[v]   = label->stratumSizes[v];
    CHKERRQ(PetscObjectReference((PetscObject) (label->points[v])));
    (*labelnew)->points[v]         = label->points[v];
    (*labelnew)->validIS[v]        = PETSC_TRUE;
  }
  CHKERRQ(PetscHMapIDestroy(&(*labelnew)->hmap));
  CHKERRQ(PetscHMapIDuplicate(label->hmap,&(*labelnew)->hmap));
  (*labelnew)->pStart = -1;
  (*labelnew)->pEnd   = -1;
  (*labelnew)->bt     = NULL;
  PetscFunctionReturn(0);
}

/*@C
  DMLabelCompare - Compare two DMLabel objects

  Collective on comm

  Input Parameters:
+ comm - Comm over which to compare labels
. l0 - First DMLabel
- l1 - Second DMLabel

  Output Parameters
+ equal   - (Optional) Flag whether the two labels are equal
- message - (Optional) Message describing the difference

  Level: intermediate

  Notes:
  The output flag equal is the same on all processes.
  If it is passed as NULL and difference is found, an error is thrown on all processes.
  Make sure to pass NULL on all processes.

  The output message is set independently on each rank.
  It is set to NULL if no difference was found on the current rank. It must be freed by user.
  If message is passed as NULL and difference is found, the difference description is printed to stderr in synchronized manner.
  Make sure to pass NULL on all processes.

  For the comparison, we ignore the order of stratum values, and strata with no points.

  The communicator needs to be specified because currently DMLabel can live on PETSC_COMM_SELF even if the underlying DM is parallel.

  Fortran Notes:
  This function is currently not available from Fortran.

.seealso: DMCompareLabels(), DMLabelGetNumValues(), DMLabelGetDefaultValue(), DMLabelGetNonEmptyStratumValuesIS(), DMLabelGetStratumIS()
@*/
PetscErrorCode DMLabelCompare(MPI_Comm comm, DMLabel l0, DMLabel l1, PetscBool *equal, char **message)
{
  const char     *name0, *name1;
  char            msg[PETSC_MAX_PATH_LEN] = "";
  PetscBool       eq;
  PetscMPIInt     rank;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(l0, DMLABEL_CLASSID, 2);
  PetscValidHeaderSpecific(l1, DMLABEL_CLASSID, 3);
  if (equal) PetscValidBoolPointer(equal, 4);
  if (message) PetscValidPointer(message, 5);
  CHKERRMPI(MPI_Comm_rank(comm, &rank));
  CHKERRQ(PetscObjectGetName((PetscObject)l0, &name0));
  CHKERRQ(PetscObjectGetName((PetscObject)l1, &name1));
  {
    PetscInt v0, v1;

    CHKERRQ(DMLabelGetDefaultValue(l0, &v0));
    CHKERRQ(DMLabelGetDefaultValue(l1, &v1));
    eq = (PetscBool) (v0 == v1);
    if (!eq) {
      CHKERRQ(PetscSNPrintf(msg, sizeof(msg), "Default value of DMLabel l0 \"%s\" = %D != %D = Default value of DMLabel l1 \"%s\"", name0, v0, v1, name1));
    }
    CHKERRMPI(MPI_Allreduce(MPI_IN_PLACE, &eq, 1, MPIU_BOOL, MPI_LAND, comm));
    if (!eq) goto finish;
  }
  {
    IS              is0, is1;

    CHKERRQ(DMLabelGetNonEmptyStratumValuesIS(l0, &is0));
    CHKERRQ(DMLabelGetNonEmptyStratumValuesIS(l1, &is1));
    CHKERRQ(ISEqual(is0, is1, &eq));
    CHKERRQ(ISDestroy(&is0));
    CHKERRQ(ISDestroy(&is1));
    if (!eq) {
      CHKERRQ(PetscSNPrintf(msg, sizeof(msg), "Stratum values in DMLabel l0 \"%s\" are different than in DMLabel l1 \"%s\"", name0, name1));
    }
    CHKERRMPI(MPI_Allreduce(MPI_IN_PLACE, &eq, 1, MPIU_BOOL, MPI_LAND, comm));
    if (!eq) goto finish;
  }
  {
    PetscInt i, nValues;

    CHKERRQ(DMLabelGetNumValues(l0, &nValues));
    for (i=0; i<nValues; i++) {
      const PetscInt  v = l0->stratumValues[i];
      PetscInt        n;
      IS              is0, is1;

      CHKERRQ(DMLabelGetStratumSize_Private(l0, i, &n));
      if (!n) continue;
      CHKERRQ(DMLabelGetStratumIS(l0, v, &is0));
      CHKERRQ(DMLabelGetStratumIS(l1, v, &is1));
      CHKERRQ(ISEqualUnsorted(is0, is1, &eq));
      CHKERRQ(ISDestroy(&is0));
      CHKERRQ(ISDestroy(&is1));
      if (!eq) {
        CHKERRQ(PetscSNPrintf(msg, sizeof(msg), "Stratum #%D with value %D contains different points in DMLabel l0 \"%s\" and DMLabel l1 \"%s\"", i, v, name0, name1));
        break;
      }
    }
    CHKERRMPI(MPI_Allreduce(MPI_IN_PLACE, &eq, 1, MPIU_BOOL, MPI_LAND, comm));
  }
finish:
  /* If message output arg not set, print to stderr */
  if (message) {
    *message = NULL;
    if (msg[0]) {
      CHKERRQ(PetscStrallocpy(msg, message));
    }
  } else {
    if (msg[0]) {
      CHKERRQ(PetscSynchronizedFPrintf(comm, PETSC_STDERR, "[%d] %s\n", rank, msg));
    }
    CHKERRQ(PetscSynchronizedFlush(comm, PETSC_STDERR));
  }
  /* If same output arg not ser and labels are not equal, throw error */
  if (equal) *equal = eq;
  else PetscCheck(eq,comm, PETSC_ERR_ARG_INCOMP, "DMLabels l0 \"%s\" and l1 \"%s\" are not equal");
  PetscFunctionReturn(0);
}

/*@
  DMLabelComputeIndex - Create an index structure for membership determination, automatically determining the bounds

  Not collective

  Input Parameter:
. label  - The DMLabel

  Level: intermediate

.seealso: DMLabelHasPoint(), DMLabelCreateIndex(), DMLabelDestroyIndex(), DMLabelGetValue(), DMLabelSetValue()
@*/
PetscErrorCode DMLabelComputeIndex(DMLabel label)
{
  PetscInt       pStart = PETSC_MAX_INT, pEnd = -1, v;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 1);
  CHKERRQ(DMLabelMakeAllValid_Private(label));
  for (v = 0; v < label->numStrata; ++v) {
    const PetscInt *points;
    PetscInt       i;

    CHKERRQ(ISGetIndices(label->points[v], &points));
    for (i = 0; i < label->stratumSizes[v]; ++i) {
      const PetscInt point = points[i];

      pStart = PetscMin(point,   pStart);
      pEnd   = PetscMax(point+1, pEnd);
    }
    CHKERRQ(ISRestoreIndices(label->points[v], &points));
  }
  label->pStart = pStart == PETSC_MAX_INT ? -1 : pStart;
  label->pEnd   = pEnd;
  CHKERRQ(DMLabelCreateIndex(label, label->pStart, label->pEnd));
  PetscFunctionReturn(0);
}

/*@
  DMLabelCreateIndex - Create an index structure for membership determination

  Not collective

  Input Parameters:
+ label  - The DMLabel
. pStart - The smallest point
- pEnd   - The largest point + 1

  Level: intermediate

.seealso: DMLabelHasPoint(), DMLabelComputeIndex(), DMLabelDestroyIndex(), DMLabelGetValue(), DMLabelSetValue()
@*/
PetscErrorCode DMLabelCreateIndex(DMLabel label, PetscInt pStart, PetscInt pEnd)
{
  PetscInt       v;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 1);
  CHKERRQ(DMLabelDestroyIndex(label));
  CHKERRQ(DMLabelMakeAllValid_Private(label));
  label->pStart = pStart;
  label->pEnd   = pEnd;
  /* This can be hooked into SetValue(),  ClearValue(), etc. for updating */
  CHKERRQ(PetscBTCreate(pEnd - pStart, &label->bt));
  for (v = 0; v < label->numStrata; ++v) {
    const PetscInt *points;
    PetscInt       i;

    CHKERRQ(ISGetIndices(label->points[v], &points));
    for (i = 0; i < label->stratumSizes[v]; ++i) {
      const PetscInt point = points[i];

      PetscCheckFalse((point < pStart) || (point >= pEnd),PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Label point %D is not in [%D, %D)", point, pStart, pEnd);
      CHKERRQ(PetscBTSet(label->bt, point - pStart));
    }
    CHKERRQ(ISRestoreIndices(label->points[v], &points));
  }
  PetscFunctionReturn(0);
}

/*@
  DMLabelDestroyIndex - Destroy the index structure

  Not collective

  Input Parameter:
. label - the DMLabel

  Level: intermediate

.seealso: DMLabelHasPoint(), DMLabelCreateIndex(), DMLabelGetValue(), DMLabelSetValue()
@*/
PetscErrorCode DMLabelDestroyIndex(DMLabel label)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 1);
  label->pStart = -1;
  label->pEnd   = -1;
  CHKERRQ(PetscBTDestroy(&label->bt));
  PetscFunctionReturn(0);
}

/*@
  DMLabelGetBounds - Return the smallest and largest point in the label

  Not collective

  Input Parameter:
. label - the DMLabel

  Output Parameters:
+ pStart - The smallest point
- pEnd   - The largest point + 1

  Note: This will compute an index for the label if one does not exist.

  Level: intermediate

.seealso: DMLabelHasPoint(), DMLabelCreateIndex(), DMLabelGetValue(), DMLabelSetValue()
@*/
PetscErrorCode DMLabelGetBounds(DMLabel label, PetscInt *pStart, PetscInt *pEnd)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 1);
  if ((label->pStart == -1) && (label->pEnd == -1)) CHKERRQ(DMLabelComputeIndex(label));
  if (pStart) {
    PetscValidIntPointer(pStart, 2);
    *pStart = label->pStart;
  }
  if (pEnd) {
    PetscValidIntPointer(pEnd, 3);
    *pEnd = label->pEnd;
  }
  PetscFunctionReturn(0);
}

/*@
  DMLabelHasValue - Determine whether a label assigns the value to any point

  Not collective

  Input Parameters:
+ label - the DMLabel
- value - the value

  Output Parameter:
. contains - Flag indicating whether the label maps this value to any point

  Level: developer

.seealso: DMLabelHasPoint(), DMLabelGetValue(), DMLabelSetValue()
@*/
PetscErrorCode DMLabelHasValue(DMLabel label, PetscInt value, PetscBool *contains)
{
  PetscInt v;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 1);
  PetscValidBoolPointer(contains, 3);
  CHKERRQ(DMLabelLookupStratum(label, value, &v));
  *contains = v < 0 ? PETSC_FALSE : PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*@
  DMLabelHasPoint - Determine whether a label assigns a value to a point

  Not collective

  Input Parameters:
+ label - the DMLabel
- point - the point

  Output Parameter:
. contains - Flag indicating whether the label maps this point to a value

  Note: The user must call DMLabelCreateIndex() before this function.

  Level: developer

.seealso: DMLabelCreateIndex(), DMLabelGetValue(), DMLabelSetValue()
@*/
PetscErrorCode DMLabelHasPoint(DMLabel label, PetscInt point, PetscBool *contains)
{
  PetscFunctionBeginHot;
  PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 1);
  PetscValidBoolPointer(contains, 3);
  CHKERRQ(DMLabelMakeAllValid_Private(label));
  if (PetscDefined(USE_DEBUG)) {
    PetscCheck(label->bt,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Must call DMLabelCreateIndex() before DMLabelHasPoint()");
    PetscCheckFalse((point < label->pStart) || (point >= label->pEnd),PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Label point %D is not in [%D, %D)", point, label->pStart, label->pEnd);
  }
  *contains = PetscBTLookup(label->bt, point - label->pStart) ? PETSC_TRUE : PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*@
  DMLabelStratumHasPoint - Return true if the stratum contains a point

  Not collective

  Input Parameters:
+ label - the DMLabel
. value - the stratum value
- point - the point

  Output Parameter:
. contains - true if the stratum contains the point

  Level: intermediate

.seealso: DMLabelCreate(), DMLabelSetValue(), DMLabelClearValue()
@*/
PetscErrorCode DMLabelStratumHasPoint(DMLabel label, PetscInt value, PetscInt point, PetscBool *contains)
{
  PetscInt       v;

  PetscFunctionBeginHot;
  PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 1);
  PetscValidBoolPointer(contains, 4);
  *contains = PETSC_FALSE;
  CHKERRQ(DMLabelLookupStratum(label, value, &v));
  if (v < 0) PetscFunctionReturn(0);

  if (label->validIS[v]) {
    PetscInt i;

    CHKERRQ(ISLocate(label->points[v], point, &i));
    if (i >= 0) *contains = PETSC_TRUE;
  } else {
    PetscBool has;

    CHKERRQ(PetscHSetIHas(label->ht[v], point, &has));
    if (has) *contains = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}

/*@
  DMLabelGetDefaultValue - Get the default value returned by DMLabelGetValue() if a point has not been explicitly given a value.
  When a label is created, it is initialized to -1.

  Not collective

  Input parameter:
. label - a DMLabel object

  Output parameter:
. defaultValue - the default value

  Level: beginner

.seealso: DMLabelSetDefaultValue(), DMLabelGetValue(), DMLabelSetValue()
@*/
PetscErrorCode DMLabelGetDefaultValue(DMLabel label, PetscInt *defaultValue)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 1);
  *defaultValue = label->defaultValue;
  PetscFunctionReturn(0);
}

/*@
  DMLabelSetDefaultValue - Set the default value returned by DMLabelGetValue() if a point has not been explicitly given a value.
  When a label is created, it is initialized to -1.

  Not collective

  Input parameter:
. label - a DMLabel object

  Output parameter:
. defaultValue - the default value

  Level: beginner

.seealso: DMLabelGetDefaultValue(), DMLabelGetValue(), DMLabelSetValue()
@*/
PetscErrorCode DMLabelSetDefaultValue(DMLabel label, PetscInt defaultValue)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 1);
  label->defaultValue = defaultValue;
  PetscFunctionReturn(0);
}

/*@
  DMLabelGetValue - Return the value a label assigns to a point, or the label's default value (which is initially -1, and can be changed with DMLabelSetDefaultValue())

  Not collective

  Input Parameters:
+ label - the DMLabel
- point - the point

  Output Parameter:
. value - The point value, or the default value (-1 by default)

  Level: intermediate

.seealso: DMLabelCreate(), DMLabelSetValue(), DMLabelClearValue(), DMLabelGetDefaultValue(), DMLabelSetDefaultValue()
@*/
PetscErrorCode DMLabelGetValue(DMLabel label, PetscInt point, PetscInt *value)
{
  PetscInt       v;

  PetscFunctionBeginHot;
  PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 1);
  PetscValidPointer(value, 3);
  *value = label->defaultValue;
  for (v = 0; v < label->numStrata; ++v) {
    if (label->validIS[v]) {
      PetscInt i;

      CHKERRQ(ISLocate(label->points[v], point, &i));
      if (i >= 0) {
        *value = label->stratumValues[v];
        break;
      }
    } else {
      PetscBool has;

      CHKERRQ(PetscHSetIHas(label->ht[v], point, &has));
      if (has) {
        *value = label->stratumValues[v];
        break;
      }
    }
  }
  PetscFunctionReturn(0);
}

/*@
  DMLabelSetValue - Set the value a label assigns to a point.  If the value is the same as the label's default value (which is initially -1, and can be changed with DMLabelSetDefaultValue() to something different), then this function will do nothing.

  Not collective

  Input Parameters:
+ label - the DMLabel
. point - the point
- value - The point value

  Level: intermediate

.seealso: DMLabelCreate(), DMLabelGetValue(), DMLabelClearValue(), DMLabelGetDefaultValue(), DMLabelSetDefaultValue()
@*/
PetscErrorCode DMLabelSetValue(DMLabel label, PetscInt point, PetscInt value)
{
  PetscInt       v;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 1);
  /* Find label value, add new entry if needed */
  if (value == label->defaultValue) PetscFunctionReturn(0);
  CHKERRQ(DMLabelLookupAddStratum(label, value, &v));
  /* Set key */
  CHKERRQ(DMLabelMakeInvalid_Private(label, v));
  CHKERRQ(PetscHSetIAdd(label->ht[v], point));
  PetscFunctionReturn(0);
}

/*@
  DMLabelClearValue - Clear the value a label assigns to a point

  Not collective

  Input Parameters:
+ label - the DMLabel
. point - the point
- value - The point value

  Level: intermediate

.seealso: DMLabelCreate(), DMLabelGetValue(), DMLabelSetValue()
@*/
PetscErrorCode DMLabelClearValue(DMLabel label, PetscInt point, PetscInt value)
{
  PetscInt       v;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 1);
  /* Find label value */
  CHKERRQ(DMLabelLookupStratum(label, value, &v));
  if (v < 0) PetscFunctionReturn(0);

  if (label->bt) {
    PetscCheckFalse((point < label->pStart) || (point >= label->pEnd),PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Label point %D is not in [%D, %D)", point, label->pStart, label->pEnd);
    CHKERRQ(PetscBTClear(label->bt, point - label->pStart));
  }

  /* Delete key */
  CHKERRQ(DMLabelMakeInvalid_Private(label, v));
  CHKERRQ(PetscHSetIDel(label->ht[v], point));
  PetscFunctionReturn(0);
}

/*@
  DMLabelInsertIS - Set all points in the IS to a value

  Not collective

  Input Parameters:
+ label - the DMLabel
. is    - the point IS
- value - The point value

  Level: intermediate

.seealso: DMLabelCreate(), DMLabelGetValue(), DMLabelSetValue(), DMLabelClearValue()
@*/
PetscErrorCode DMLabelInsertIS(DMLabel label, IS is, PetscInt value)
{
  PetscInt        v, n, p;
  const PetscInt *points;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 1);
  PetscValidHeaderSpecific(is, IS_CLASSID, 2);
  /* Find label value, add new entry if needed */
  if (value == label->defaultValue) PetscFunctionReturn(0);
  CHKERRQ(DMLabelLookupAddStratum(label, value, &v));
  /* Set keys */
  CHKERRQ(DMLabelMakeInvalid_Private(label, v));
  CHKERRQ(ISGetLocalSize(is, &n));
  CHKERRQ(ISGetIndices(is, &points));
  for (p = 0; p < n; ++p) CHKERRQ(PetscHSetIAdd(label->ht[v], points[p]));
  CHKERRQ(ISRestoreIndices(is, &points));
  PetscFunctionReturn(0);
}

/*@
  DMLabelGetNumValues - Get the number of values that the DMLabel takes

  Not collective

  Input Parameter:
. label - the DMLabel

  Output Parameter:
. numValues - the number of values

  Level: intermediate

.seealso: DMLabelCreate(), DMLabelGetValue(), DMLabelSetValue(), DMLabelClearValue()
@*/
PetscErrorCode DMLabelGetNumValues(DMLabel label, PetscInt *numValues)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 1);
  PetscValidPointer(numValues, 2);
  *numValues = label->numStrata;
  PetscFunctionReturn(0);
}

/*@
  DMLabelGetValueIS - Get an IS of all values that the DMlabel takes

  Not collective

  Input Parameter:
. label - the DMLabel

  Output Parameter:
. is    - the value IS

  Level: intermediate

  Notes:
  The output IS should be destroyed when no longer needed.
  Strata which are allocated but empty [DMLabelGetStratumSize() yields 0] are counted.
  If you need to count only nonempty strata, use DMLabelGetNonEmptyStratumValuesIS().

.seealso: DMLabelGetNonEmptyStratumValuesIS(), DMLabelCreate(), DMLabelGetValue(), DMLabelSetValue(), DMLabelClearValue()
@*/
PetscErrorCode DMLabelGetValueIS(DMLabel label, IS *values)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 1);
  PetscValidPointer(values, 2);
  CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF, label->numStrata, label->stratumValues, PETSC_USE_POINTER, values));
  PetscFunctionReturn(0);
}

/*@
  DMLabelGetNonEmptyStratumValuesIS - Get an IS of all values that the DMlabel takes

  Not collective

  Input Parameter:
. label - the DMLabel

  Output Paramater:
. is    - the value IS

  Level: intermediate

  Notes:
  The output IS should be destroyed when no longer needed.
  This is similar to DMLabelGetValueIS() but counts only nonempty strata.

.seealso: DMLabelGetValueIS(), DMLabelCreate(), DMLabelGetValue(), DMLabelSetValue(), DMLabelClearValue()
@*/
PetscErrorCode DMLabelGetNonEmptyStratumValuesIS(DMLabel label, IS *values)
{
  PetscInt        i, j;
  PetscInt       *valuesArr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 1);
  PetscValidPointer(values, 2);
  CHKERRQ(PetscMalloc1(label->numStrata, &valuesArr));
  for (i = 0, j = 0; i < label->numStrata; i++) {
    PetscInt        n;

    CHKERRQ(DMLabelGetStratumSize_Private(label, i, &n));
    if (n) valuesArr[j++] = label->stratumValues[i];
  }
  if (j == label->numStrata) {
    CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF, label->numStrata, label->stratumValues, PETSC_USE_POINTER, values));
  } else {
    CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF, j, valuesArr, PETSC_COPY_VALUES, values));
  }
  CHKERRQ(PetscFree(valuesArr));
  PetscFunctionReturn(0);
}

/*@
  DMLabelGetValueIndex - Get the index of a given value in the list of values for the DMlabel, or -1 if it is not present

  Not collective

  Input Parameters:
+ label - the DMLabel
- value - the value

  Output Parameter:
. index - the index of value in the list of values

  Level: intermediate

.seealso: DMLabelGetValueIS(), DMLabelCreate(), DMLabelGetValue(), DMLabelSetValue(), DMLabelClearValue()
@*/
PetscErrorCode DMLabelGetValueIndex(DMLabel label, PetscInt value, PetscInt *index)
{
  PetscInt v;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 1);
  PetscValidPointer(index, 3);
  /* Do not assume they are sorted */
  for (v = 0; v < label->numStrata; ++v) if (label->stratumValues[v] == value) break;
  if (v >= label->numStrata) *index = -1;
  else                       *index = v;
  PetscFunctionReturn(0);
}

/*@
  DMLabelHasStratum - Determine whether points exist with the given value

  Not collective

  Input Parameters:
+ label - the DMLabel
- value - the stratum value

  Output Parameter:
. exists - Flag saying whether points exist

  Level: intermediate

.seealso: DMLabelCreate(), DMLabelGetValue(), DMLabelSetValue(), DMLabelClearValue()
@*/
PetscErrorCode DMLabelHasStratum(DMLabel label, PetscInt value, PetscBool *exists)
{
  PetscInt       v;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 1);
  PetscValidPointer(exists, 3);
  CHKERRQ(DMLabelLookupStratum(label, value, &v));
  *exists = v < 0 ? PETSC_FALSE : PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*@
  DMLabelGetStratumSize - Get the size of a stratum

  Not collective

  Input Parameters:
+ label - the DMLabel
- value - the stratum value

  Output Parameter:
. size - The number of points in the stratum

  Level: intermediate

.seealso: DMLabelCreate(), DMLabelGetValue(), DMLabelSetValue(), DMLabelClearValue()
@*/
PetscErrorCode DMLabelGetStratumSize(DMLabel label, PetscInt value, PetscInt *size)
{
  PetscInt       v;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 1);
  PetscValidPointer(size, 3);
  CHKERRQ(DMLabelLookupStratum(label, value, &v));
  CHKERRQ(DMLabelGetStratumSize_Private(label, v, size));
  PetscFunctionReturn(0);
}

/*@
  DMLabelGetStratumBounds - Get the largest and smallest point of a stratum

  Not collective

  Input Parameters:
+ label - the DMLabel
- value - the stratum value

  Output Parameters:
+ start - the smallest point in the stratum
- end - the largest point in the stratum

  Level: intermediate

.seealso: DMLabelCreate(), DMLabelGetValue(), DMLabelSetValue(), DMLabelClearValue()
@*/
PetscErrorCode DMLabelGetStratumBounds(DMLabel label, PetscInt value, PetscInt *start, PetscInt *end)
{
  PetscInt       v, min, max;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 1);
  if (start) {PetscValidPointer(start, 3); *start = -1;}
  if (end)   {PetscValidPointer(end,   4); *end   = -1;}
  CHKERRQ(DMLabelLookupStratum(label, value, &v));
  if (v < 0) PetscFunctionReturn(0);
  CHKERRQ(DMLabelMakeValid_Private(label, v));
  if (label->stratumSizes[v] <= 0) PetscFunctionReturn(0);
  CHKERRQ(ISGetMinMax(label->points[v], &min, &max));
  if (start) *start = min;
  if (end)   *end   = max+1;
  PetscFunctionReturn(0);
}

/*@
  DMLabelGetStratumIS - Get an IS with the stratum points

  Not collective

  Input Parameters:
+ label - the DMLabel
- value - the stratum value

  Output Parameter:
. points - The stratum points

  Level: intermediate

  Notes:
  The output IS should be destroyed when no longer needed.
  Returns NULL if the stratum is empty.

.seealso: DMLabelCreate(), DMLabelGetValue(), DMLabelSetValue(), DMLabelClearValue()
@*/
PetscErrorCode DMLabelGetStratumIS(DMLabel label, PetscInt value, IS *points)
{
  PetscInt       v;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 1);
  PetscValidPointer(points, 3);
  *points = NULL;
  CHKERRQ(DMLabelLookupStratum(label, value, &v));
  if (v < 0) PetscFunctionReturn(0);
  CHKERRQ(DMLabelMakeValid_Private(label, v));
  CHKERRQ(PetscObjectReference((PetscObject) label->points[v]));
  *points = label->points[v];
  PetscFunctionReturn(0);
}

/*@
  DMLabelSetStratumIS - Set the stratum points using an IS

  Not collective

  Input Parameters:
+ label - the DMLabel
. value - the stratum value
- points - The stratum points

  Level: intermediate

.seealso: DMLabelCreate(), DMLabelGetValue(), DMLabelSetValue(), DMLabelClearValue()
@*/
PetscErrorCode DMLabelSetStratumIS(DMLabel label, PetscInt value, IS is)
{
  PetscInt       v;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 1);
  PetscValidHeaderSpecific(is, IS_CLASSID, 3);
  CHKERRQ(DMLabelLookupAddStratum(label, value, &v));
  if (is == label->points[v]) PetscFunctionReturn(0);
  CHKERRQ(DMLabelClearStratum(label, value));
  CHKERRQ(ISGetLocalSize(is, &(label->stratumSizes[v])));
  CHKERRQ(PetscObjectReference((PetscObject)is));
  CHKERRQ(ISDestroy(&(label->points[v])));
  label->points[v]  = is;
  label->validIS[v] = PETSC_TRUE;
  CHKERRQ(PetscObjectStateIncrease((PetscObject) label));
  if (label->bt) {
    const PetscInt *points;
    PetscInt p;

    CHKERRQ(ISGetIndices(is,&points));
    for (p = 0; p < label->stratumSizes[v]; ++p) {
      const PetscInt point = points[p];

      PetscCheckFalse((point < label->pStart) || (point >= label->pEnd),PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Label point %D is not in [%D, %D)", point, label->pStart, label->pEnd);
      CHKERRQ(PetscBTSet(label->bt, point - label->pStart));
    }
  }
  PetscFunctionReturn(0);
}

/*@
  DMLabelClearStratum - Remove a stratum

  Not collective

  Input Parameters:
+ label - the DMLabel
- value - the stratum value

  Level: intermediate

.seealso: DMLabelCreate(), DMLabelGetValue(), DMLabelSetValue(), DMLabelClearValue()
@*/
PetscErrorCode DMLabelClearStratum(DMLabel label, PetscInt value)
{
  PetscInt       v;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 1);
  CHKERRQ(DMLabelLookupStratum(label, value, &v));
  if (v < 0) PetscFunctionReturn(0);
  if (label->validIS[v]) {
    if (label->bt) {
      PetscInt       i;
      const PetscInt *points;

      CHKERRQ(ISGetIndices(label->points[v], &points));
      for (i = 0; i < label->stratumSizes[v]; ++i) {
        const PetscInt point = points[i];

        PetscCheckFalse((point < label->pStart) || (point >= label->pEnd),PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Label point %D is not in [%D, %D)", point, label->pStart, label->pEnd);
        CHKERRQ(PetscBTClear(label->bt, point - label->pStart));
      }
      CHKERRQ(ISRestoreIndices(label->points[v], &points));
    }
    label->stratumSizes[v] = 0;
    CHKERRQ(ISDestroy(&label->points[v]));
    CHKERRQ(ISCreateStride(PETSC_COMM_SELF, 0, 0, 1, &label->points[v]));
    CHKERRQ(PetscObjectSetName((PetscObject) label->points[v], "indices"));
    CHKERRQ(PetscObjectStateIncrease((PetscObject) label));
  } else {
    CHKERRQ(PetscHSetIClear(label->ht[v]));
  }
  PetscFunctionReturn(0);
}

/*@
  DMLabelSetStratumBounds - Efficiently give a contiguous set of points a given label value

  Not collective

  Input Parameters:
+ label  - The DMLabel
. value  - The label value for all points
. pStart - The first point
- pEnd   - A point beyond all marked points

  Note: The marks points are [pStart, pEnd), and only the bounds are stored.

  Level: intermediate

.seealso: DMLabelCreate(), DMLabelSetStratumIS(), DMLabelGetStratumIS()
@*/
PetscErrorCode DMLabelSetStratumBounds(DMLabel label, PetscInt value, PetscInt pStart, PetscInt pEnd)
{
  IS             pIS;

  PetscFunctionBegin;
  CHKERRQ(ISCreateStride(PETSC_COMM_SELF, pEnd - pStart, pStart, 1, &pIS));
  CHKERRQ(DMLabelSetStratumIS(label, value, pIS));
  CHKERRQ(ISDestroy(&pIS));
  PetscFunctionReturn(0);
}

/*@
  DMLabelGetStratumPointIndex - Get the index of a point in a given stratum

  Not collective

  Input Parameters:
+ label  - The DMLabel
. value  - The label value
- p      - A point with this value

  Output Parameter:
. index  - The index of this point in the stratum, or -1 if the point is not in the stratum or the stratum does not exist

  Level: intermediate

.seealso: DMLabelGetValueIndex(), DMLabelGetStratumIS(), DMLabelCreate()
@*/
PetscErrorCode DMLabelGetStratumPointIndex(DMLabel label, PetscInt value, PetscInt p, PetscInt *index)
{
  const PetscInt *indices;
  PetscInt        v;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 1);
  PetscValidPointer(index, 4);
  *index = -1;
  CHKERRQ(DMLabelLookupStratum(label, value, &v));
  if (v < 0) PetscFunctionReturn(0);
  CHKERRQ(DMLabelMakeValid_Private(label, v));
  CHKERRQ(ISGetIndices(label->points[v], &indices));
  CHKERRQ(PetscFindInt(p, label->stratumSizes[v], indices, index));
  CHKERRQ(ISRestoreIndices(label->points[v], &indices));
  PetscFunctionReturn(0);
}

/*@
  DMLabelFilter - Remove all points outside of [start, end)

  Not collective

  Input Parameters:
+ label - the DMLabel
. start - the first point kept
- end - one more than the last point kept

  Level: intermediate

.seealso: DMLabelCreate(), DMLabelGetValue(), DMLabelSetValue(), DMLabelClearValue()
@*/
PetscErrorCode DMLabelFilter(DMLabel label, PetscInt start, PetscInt end)
{
  PetscInt       v;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 1);
  CHKERRQ(DMLabelDestroyIndex(label));
  CHKERRQ(DMLabelMakeAllValid_Private(label));
  for (v = 0; v < label->numStrata; ++v) {
    CHKERRQ(ISGeneralFilter(label->points[v], start, end));
  }
  CHKERRQ(DMLabelCreateIndex(label, start, end));
  PetscFunctionReturn(0);
}

/*@
  DMLabelPermute - Create a new label with permuted points

  Not collective

  Input Parameters:
+ label - the DMLabel
- permutation - the point permutation

  Output Parameter:
. labelnew - the new label containing the permuted points

  Level: intermediate

.seealso: DMLabelCreate(), DMLabelGetValue(), DMLabelSetValue(), DMLabelClearValue()
@*/
PetscErrorCode DMLabelPermute(DMLabel label, IS permutation, DMLabel *labelNew)
{
  const PetscInt *perm;
  PetscInt        numValues, numPoints, v, q;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 1);
  PetscValidHeaderSpecific(permutation, IS_CLASSID, 2);
  CHKERRQ(DMLabelMakeAllValid_Private(label));
  CHKERRQ(DMLabelDuplicate(label, labelNew));
  CHKERRQ(DMLabelGetNumValues(*labelNew, &numValues));
  CHKERRQ(ISGetLocalSize(permutation, &numPoints));
  CHKERRQ(ISGetIndices(permutation, &perm));
  for (v = 0; v < numValues; ++v) {
    const PetscInt size   = (*labelNew)->stratumSizes[v];
    const PetscInt *points;
    PetscInt *pointsNew;

    CHKERRQ(ISGetIndices((*labelNew)->points[v],&points));
    CHKERRQ(PetscMalloc1(size,&pointsNew));
    for (q = 0; q < size; ++q) {
      const PetscInt point = points[q];

      PetscCheckFalse((point < 0) || (point >= numPoints),PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Label point %D is not in [0, %D) for the remapping", point, numPoints);
      pointsNew[q] = perm[point];
    }
    CHKERRQ(ISRestoreIndices((*labelNew)->points[v],&points));
    CHKERRQ(PetscSortInt(size, pointsNew));
    CHKERRQ(ISDestroy(&((*labelNew)->points[v])));
    if (size > 0 && pointsNew[size - 1] == pointsNew[0] + size - 1) {
      CHKERRQ(ISCreateStride(PETSC_COMM_SELF,size,pointsNew[0],1,&((*labelNew)->points[v])));
      CHKERRQ(PetscFree(pointsNew));
    } else {
      CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF,size,pointsNew,PETSC_OWN_POINTER,&((*labelNew)->points[v])));
    }
    CHKERRQ(PetscObjectSetName((PetscObject) ((*labelNew)->points[v]), "indices"));
  }
  CHKERRQ(ISRestoreIndices(permutation, &perm));
  if (label->bt) {
    CHKERRQ(PetscBTDestroy(&label->bt));
    CHKERRQ(DMLabelCreateIndex(label, label->pStart, label->pEnd));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DMLabelDistribute_Internal(DMLabel label, PetscSF sf, PetscSection *leafSection, PetscInt **leafStrata)
{
  MPI_Comm       comm;
  PetscInt       s, l, nroots, nleaves, dof, offset, size;
  PetscInt      *remoteOffsets, *rootStrata, *rootIdx;
  PetscSection   rootSection;
  PetscSF        labelSF;

  PetscFunctionBegin;
  if (label) CHKERRQ(DMLabelMakeAllValid_Private(label));
  CHKERRQ(PetscObjectGetComm((PetscObject)sf, &comm));
  /* Build a section of stratum values per point, generate the according SF
     and distribute point-wise stratum values to leaves. */
  CHKERRQ(PetscSFGetGraph(sf, &nroots, &nleaves, NULL, NULL));
  CHKERRQ(PetscSectionCreate(comm, &rootSection));
  CHKERRQ(PetscSectionSetChart(rootSection, 0, nroots));
  if (label) {
    for (s = 0; s < label->numStrata; ++s) {
      const PetscInt *points;

      CHKERRQ(ISGetIndices(label->points[s], &points));
      for (l = 0; l < label->stratumSizes[s]; l++) {
        CHKERRQ(PetscSectionGetDof(rootSection, points[l], &dof));
        CHKERRQ(PetscSectionSetDof(rootSection, points[l], dof+1));
      }
      CHKERRQ(ISRestoreIndices(label->points[s], &points));
    }
  }
  CHKERRQ(PetscSectionSetUp(rootSection));
  /* Create a point-wise array of stratum values */
  CHKERRQ(PetscSectionGetStorageSize(rootSection, &size));
  CHKERRQ(PetscMalloc1(size, &rootStrata));
  CHKERRQ(PetscCalloc1(nroots, &rootIdx));
  if (label) {
    for (s = 0; s < label->numStrata; ++s) {
      const PetscInt *points;

      CHKERRQ(ISGetIndices(label->points[s], &points));
      for (l = 0; l < label->stratumSizes[s]; l++) {
        const PetscInt p = points[l];
        CHKERRQ(PetscSectionGetOffset(rootSection, p, &offset));
        rootStrata[offset+rootIdx[p]++] = label->stratumValues[s];
      }
      CHKERRQ(ISRestoreIndices(label->points[s], &points));
    }
  }
  /* Build SF that maps label points to remote processes */
  CHKERRQ(PetscSectionCreate(comm, leafSection));
  CHKERRQ(PetscSFDistributeSection(sf, rootSection, &remoteOffsets, *leafSection));
  CHKERRQ(PetscSFCreateSectionSF(sf, rootSection, remoteOffsets, *leafSection, &labelSF));
  CHKERRQ(PetscFree(remoteOffsets));
  /* Send the strata for each point over the derived SF */
  CHKERRQ(PetscSectionGetStorageSize(*leafSection, &size));
  CHKERRQ(PetscMalloc1(size, leafStrata));
  CHKERRQ(PetscSFBcastBegin(labelSF, MPIU_INT, rootStrata, *leafStrata,MPI_REPLACE));
  CHKERRQ(PetscSFBcastEnd(labelSF, MPIU_INT, rootStrata, *leafStrata,MPI_REPLACE));
  /* Clean up */
  CHKERRQ(PetscFree(rootStrata));
  CHKERRQ(PetscFree(rootIdx));
  CHKERRQ(PetscSectionDestroy(&rootSection));
  CHKERRQ(PetscSFDestroy(&labelSF));
  PetscFunctionReturn(0);
}

/*@
  DMLabelDistribute - Create a new label pushed forward over the PetscSF

  Collective on sf

  Input Parameters:
+ label - the DMLabel
- sf    - the map from old to new distribution

  Output Parameter:
. labelnew - the new redistributed label

  Level: intermediate

.seealso: DMLabelCreate(), DMLabelGetValue(), DMLabelSetValue(), DMLabelClearValue()
@*/
PetscErrorCode DMLabelDistribute(DMLabel label, PetscSF sf, DMLabel *labelNew)
{
  MPI_Comm       comm;
  PetscSection   leafSection;
  PetscInt       p, pStart, pEnd, s, size, dof, offset, stratum;
  PetscInt      *leafStrata, *strataIdx;
  PetscInt     **points;
  const char    *lname = NULL;
  char          *name;
  PetscInt       nameSize;
  PetscHSetI     stratumHash;
  size_t         len = 0;
  PetscMPIInt    rank;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf, PETSCSF_CLASSID, 2);
  if (label) {
    PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 1);
    CHKERRQ(DMLabelMakeAllValid_Private(label));
  }
  CHKERRQ(PetscObjectGetComm((PetscObject)sf, &comm));
  CHKERRMPI(MPI_Comm_rank(comm, &rank));
  /* Bcast name */
  if (rank == 0) {
    CHKERRQ(PetscObjectGetName((PetscObject) label, &lname));
    CHKERRQ(PetscStrlen(lname, &len));
  }
  nameSize = len;
  CHKERRMPI(MPI_Bcast(&nameSize, 1, MPIU_INT, 0, comm));
  CHKERRQ(PetscMalloc1(nameSize+1, &name));
  if (rank == 0) CHKERRQ(PetscArraycpy(name, lname, nameSize+1));
  CHKERRMPI(MPI_Bcast(name, nameSize+1, MPI_CHAR, 0, comm));
  CHKERRQ(DMLabelCreate(PETSC_COMM_SELF, name, labelNew));
  CHKERRQ(PetscFree(name));
  /* Bcast defaultValue */
  if (rank == 0) (*labelNew)->defaultValue = label->defaultValue;
  CHKERRMPI(MPI_Bcast(&(*labelNew)->defaultValue, 1, MPIU_INT, 0, comm));
  /* Distribute stratum values over the SF and get the point mapping on the receiver */
  CHKERRQ(DMLabelDistribute_Internal(label, sf, &leafSection, &leafStrata));
  /* Determine received stratum values and initialise new label*/
  CHKERRQ(PetscHSetICreate(&stratumHash));
  CHKERRQ(PetscSectionGetStorageSize(leafSection, &size));
  for (p = 0; p < size; ++p) CHKERRQ(PetscHSetIAdd(stratumHash, leafStrata[p]));
  CHKERRQ(PetscHSetIGetSize(stratumHash, &(*labelNew)->numStrata));
  CHKERRQ(PetscMalloc1((*labelNew)->numStrata, &(*labelNew)->validIS));
  for (s = 0; s < (*labelNew)->numStrata; ++s) (*labelNew)->validIS[s] = PETSC_TRUE;
  CHKERRQ(PetscMalloc1((*labelNew)->numStrata, &(*labelNew)->stratumValues));
  /* Turn leafStrata into indices rather than stratum values */
  offset = 0;
  CHKERRQ(PetscHSetIGetElems(stratumHash, &offset, (*labelNew)->stratumValues));
  CHKERRQ(PetscSortInt((*labelNew)->numStrata,(*labelNew)->stratumValues));
  for (s = 0; s < (*labelNew)->numStrata; ++s) {
    CHKERRQ(PetscHMapISet((*labelNew)->hmap, (*labelNew)->stratumValues[s], s));
  }
  for (p = 0; p < size; ++p) {
    for (s = 0; s < (*labelNew)->numStrata; ++s) {
      if (leafStrata[p] == (*labelNew)->stratumValues[s]) {leafStrata[p] = s; break;}
    }
  }
  /* Rebuild the point strata on the receiver */
  CHKERRQ(PetscCalloc1((*labelNew)->numStrata,&(*labelNew)->stratumSizes));
  CHKERRQ(PetscSectionGetChart(leafSection, &pStart, &pEnd));
  for (p=pStart; p<pEnd; p++) {
    CHKERRQ(PetscSectionGetDof(leafSection, p, &dof));
    CHKERRQ(PetscSectionGetOffset(leafSection, p, &offset));
    for (s=0; s<dof; s++) {
      (*labelNew)->stratumSizes[leafStrata[offset+s]]++;
    }
  }
  CHKERRQ(PetscCalloc1((*labelNew)->numStrata,&(*labelNew)->ht));
  CHKERRQ(PetscMalloc1((*labelNew)->numStrata,&(*labelNew)->points));
  CHKERRQ(PetscMalloc1((*labelNew)->numStrata,&points));
  for (s = 0; s < (*labelNew)->numStrata; ++s) {
    CHKERRQ(PetscHSetICreate(&(*labelNew)->ht[s]));
    CHKERRQ(PetscMalloc1((*labelNew)->stratumSizes[s], &(points[s])));
  }
  /* Insert points into new strata */
  CHKERRQ(PetscCalloc1((*labelNew)->numStrata, &strataIdx));
  CHKERRQ(PetscSectionGetChart(leafSection, &pStart, &pEnd));
  for (p=pStart; p<pEnd; p++) {
    CHKERRQ(PetscSectionGetDof(leafSection, p, &dof));
    CHKERRQ(PetscSectionGetOffset(leafSection, p, &offset));
    for (s=0; s<dof; s++) {
      stratum = leafStrata[offset+s];
      points[stratum][strataIdx[stratum]++] = p;
    }
  }
  for (s = 0; s < (*labelNew)->numStrata; s++) {
    CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF,(*labelNew)->stratumSizes[s],&(points[s][0]),PETSC_OWN_POINTER,&((*labelNew)->points[s])));
    CHKERRQ(PetscObjectSetName((PetscObject)((*labelNew)->points[s]),"indices"));
  }
  CHKERRQ(PetscFree(points));
  CHKERRQ(PetscHSetIDestroy(&stratumHash));
  CHKERRQ(PetscFree(leafStrata));
  CHKERRQ(PetscFree(strataIdx));
  CHKERRQ(PetscSectionDestroy(&leafSection));
  PetscFunctionReturn(0);
}

/*@
  DMLabelGather - Gather all label values from leafs into roots

  Collective on sf

  Input Parameters:
+ label - the DMLabel
- sf - the Star Forest point communication map

  Output Parameters:
. labelNew - the new DMLabel with localised leaf values

  Level: developer

  Note: This is the inverse operation to DMLabelDistribute.

.seealso: DMLabelDistribute()
@*/
PetscErrorCode DMLabelGather(DMLabel label, PetscSF sf, DMLabel *labelNew)
{
  MPI_Comm       comm;
  PetscSection   rootSection;
  PetscSF        sfLabel;
  PetscSFNode   *rootPoints, *leafPoints;
  PetscInt       p, s, d, nroots, nleaves, nmultiroots, idx, dof, offset;
  const PetscInt *rootDegree, *ilocal;
  PetscInt       *rootStrata;
  const char    *lname;
  char          *name;
  PetscInt       nameSize;
  size_t         len = 0;
  PetscMPIInt    rank, size;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 1);
  PetscValidHeaderSpecific(sf, PETSCSF_CLASSID, 2);
  CHKERRQ(PetscObjectGetComm((PetscObject)sf, &comm));
  CHKERRMPI(MPI_Comm_rank(comm, &rank));
  CHKERRMPI(MPI_Comm_size(comm, &size));
  /* Bcast name */
  if (rank == 0) {
    CHKERRQ(PetscObjectGetName((PetscObject) label, &lname));
    CHKERRQ(PetscStrlen(lname, &len));
  }
  nameSize = len;
  CHKERRMPI(MPI_Bcast(&nameSize, 1, MPIU_INT, 0, comm));
  CHKERRQ(PetscMalloc1(nameSize+1, &name));
  if (rank == 0) CHKERRQ(PetscArraycpy(name, lname, nameSize+1));
  CHKERRMPI(MPI_Bcast(name, nameSize+1, MPI_CHAR, 0, comm));
  CHKERRQ(DMLabelCreate(PETSC_COMM_SELF, name, labelNew));
  CHKERRQ(PetscFree(name));
  /* Gather rank/index pairs of leaves into local roots to build
     an inverse, multi-rooted SF. Note that this ignores local leaf
     indexing due to the use of the multiSF in PetscSFGather. */
  CHKERRQ(PetscSFGetGraph(sf, &nroots, &nleaves, &ilocal, NULL));
  CHKERRQ(PetscMalloc1(nroots, &leafPoints));
  for (p = 0; p < nroots; ++p) leafPoints[p].rank = leafPoints[p].index = -1;
  for (p = 0; p < nleaves; p++) {
    PetscInt ilp = ilocal ? ilocal[p] : p;

    leafPoints[ilp].index = ilp;
    leafPoints[ilp].rank  = rank;
  }
  CHKERRQ(PetscSFComputeDegreeBegin(sf, &rootDegree));
  CHKERRQ(PetscSFComputeDegreeEnd(sf, &rootDegree));
  for (p = 0, nmultiroots = 0; p < nroots; ++p) nmultiroots += rootDegree[p];
  CHKERRQ(PetscMalloc1(nmultiroots, &rootPoints));
  CHKERRQ(PetscSFGatherBegin(sf, MPIU_2INT, leafPoints, rootPoints));
  CHKERRQ(PetscSFGatherEnd(sf, MPIU_2INT, leafPoints, rootPoints));
  CHKERRQ(PetscSFCreate(comm,& sfLabel));
  CHKERRQ(PetscSFSetGraph(sfLabel, nroots, nmultiroots, NULL, PETSC_OWN_POINTER, rootPoints, PETSC_OWN_POINTER));
  /* Migrate label over inverted SF to pull stratum values at leaves into roots. */
  CHKERRQ(DMLabelDistribute_Internal(label, sfLabel, &rootSection, &rootStrata));
  /* Rebuild the point strata on the receiver */
  for (p = 0, idx = 0; p < nroots; p++) {
    for (d = 0; d < rootDegree[p]; d++) {
      CHKERRQ(PetscSectionGetDof(rootSection, idx+d, &dof));
      CHKERRQ(PetscSectionGetOffset(rootSection, idx+d, &offset));
      for (s = 0; s < dof; s++) CHKERRQ(DMLabelSetValue(*labelNew, p, rootStrata[offset+s]));
    }
    idx += rootDegree[p];
  }
  CHKERRQ(PetscFree(leafPoints));
  CHKERRQ(PetscFree(rootStrata));
  CHKERRQ(PetscSectionDestroy(&rootSection));
  CHKERRQ(PetscSFDestroy(&sfLabel));
  PetscFunctionReturn(0);
}

/*@
  DMLabelConvertToSection - Make a PetscSection/IS pair that encodes the label

  Not collective

  Input Parameter:
. label - the DMLabel

  Output Parameters:
+ section - the section giving offsets for each stratum
- is - An IS containing all the label points

  Level: developer

.seealso: DMLabelDistribute()
@*/
PetscErrorCode DMLabelConvertToSection(DMLabel label, PetscSection *section, IS *is)
{
  IS              vIS;
  const PetscInt *values;
  PetscInt       *points;
  PetscInt        nV, vS = 0, vE = 0, v, N;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 1);
  CHKERRQ(DMLabelGetNumValues(label, &nV));
  CHKERRQ(DMLabelGetValueIS(label, &vIS));
  CHKERRQ(ISGetIndices(vIS, &values));
  if (nV) {vS = values[0]; vE = values[0]+1;}
  for (v = 1; v < nV; ++v) {
    vS = PetscMin(vS, values[v]);
    vE = PetscMax(vE, values[v]+1);
  }
  CHKERRQ(PetscSectionCreate(PETSC_COMM_SELF, section));
  CHKERRQ(PetscSectionSetChart(*section, vS, vE));
  for (v = 0; v < nV; ++v) {
    PetscInt n;

    CHKERRQ(DMLabelGetStratumSize(label, values[v], &n));
    CHKERRQ(PetscSectionSetDof(*section, values[v], n));
  }
  CHKERRQ(PetscSectionSetUp(*section));
  CHKERRQ(PetscSectionGetStorageSize(*section, &N));
  CHKERRQ(PetscMalloc1(N, &points));
  for (v = 0; v < nV; ++v) {
    IS              is;
    const PetscInt *spoints;
    PetscInt        dof, off, p;

    CHKERRQ(PetscSectionGetDof(*section, values[v], &dof));
    CHKERRQ(PetscSectionGetOffset(*section, values[v], &off));
    CHKERRQ(DMLabelGetStratumIS(label, values[v], &is));
    CHKERRQ(ISGetIndices(is, &spoints));
    for (p = 0; p < dof; ++p) points[off+p] = spoints[p];
    CHKERRQ(ISRestoreIndices(is, &spoints));
    CHKERRQ(ISDestroy(&is));
  }
  CHKERRQ(ISRestoreIndices(vIS, &values));
  CHKERRQ(ISDestroy(&vIS));
  CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF, N, points, PETSC_OWN_POINTER, is));
  PetscFunctionReturn(0);
}

/*@
  PetscSectionCreateGlobalSectionLabel - Create a section describing the global field layout using
  the local section and an SF describing the section point overlap.

  Collective on sf

  Input Parameters:
  + s - The PetscSection for the local field layout
  . sf - The SF describing parallel layout of the section points
  . includeConstraints - By default this is PETSC_FALSE, meaning that the global field vector will not possess constrained dofs
  . label - The label specifying the points
  - labelValue - The label stratum specifying the points

  Output Parameter:
  . gsection - The PetscSection for the global field layout

  Note: This gives negative sizes and offsets to points not owned by this process

  Level: developer

.seealso: PetscSectionCreate()
@*/
PetscErrorCode PetscSectionCreateGlobalSectionLabel(PetscSection s, PetscSF sf, PetscBool includeConstraints, DMLabel label, PetscInt labelValue, PetscSection *gsection)
{
  PetscInt      *neg = NULL, *tmpOff = NULL;
  PetscInt       pStart, pEnd, p, dof, cdof, off, globalOff = 0, nroots;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(s, PETSC_SECTION_CLASSID, 1);
  PetscValidHeaderSpecific(sf, PETSCSF_CLASSID, 2);
  PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 4);
  CHKERRQ(PetscSectionCreate(PetscObjectComm((PetscObject) s), gsection));
  CHKERRQ(PetscSectionGetChart(s, &pStart, &pEnd));
  CHKERRQ(PetscSectionSetChart(*gsection, pStart, pEnd));
  CHKERRQ(PetscSFGetGraph(sf, &nroots, NULL, NULL, NULL));
  if (nroots >= 0) {
    PetscCheckFalse(nroots < pEnd-pStart,PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "PetscSF nroots %d < %d section size", nroots, pEnd-pStart);
    CHKERRQ(PetscCalloc1(nroots, &neg));
    if (nroots > pEnd-pStart) {
      CHKERRQ(PetscCalloc1(nroots, &tmpOff));
    } else {
      tmpOff = &(*gsection)->atlasDof[-pStart];
    }
  }
  /* Mark ghost points with negative dof */
  for (p = pStart; p < pEnd; ++p) {
    PetscInt value;

    CHKERRQ(DMLabelGetValue(label, p, &value));
    if (value != labelValue) continue;
    CHKERRQ(PetscSectionGetDof(s, p, &dof));
    CHKERRQ(PetscSectionSetDof(*gsection, p, dof));
    CHKERRQ(PetscSectionGetConstraintDof(s, p, &cdof));
    if (!includeConstraints && cdof > 0) CHKERRQ(PetscSectionSetConstraintDof(*gsection, p, cdof));
    if (neg) neg[p] = -(dof+1);
  }
  CHKERRQ(PetscSectionSetUpBC(*gsection));
  if (nroots >= 0) {
    CHKERRQ(PetscSFBcastBegin(sf, MPIU_INT, neg, tmpOff,MPI_REPLACE));
    CHKERRQ(PetscSFBcastEnd(sf, MPIU_INT, neg, tmpOff,MPI_REPLACE));
    if (nroots > pEnd-pStart) {
      for (p = pStart; p < pEnd; ++p) {if (tmpOff[p] < 0) (*gsection)->atlasDof[p-pStart] = tmpOff[p];}
    }
  }
  /* Calculate new sizes, get proccess offset, and calculate point offsets */
  for (p = 0, off = 0; p < pEnd-pStart; ++p) {
    cdof = (!includeConstraints && s->bc) ? s->bc->atlasDof[p] : 0;
    (*gsection)->atlasOff[p] = off;
    off += (*gsection)->atlasDof[p] > 0 ? (*gsection)->atlasDof[p]-cdof : 0;
  }
  CHKERRMPI(MPI_Scan(&off, &globalOff, 1, MPIU_INT, MPI_SUM, PetscObjectComm((PetscObject) s)));
  globalOff -= off;
  for (p = 0, off = 0; p < pEnd-pStart; ++p) {
    (*gsection)->atlasOff[p] += globalOff;
    if (neg) neg[p] = -((*gsection)->atlasOff[p]+1);
  }
  /* Put in negative offsets for ghost points */
  if (nroots >= 0) {
    CHKERRQ(PetscSFBcastBegin(sf, MPIU_INT, neg, tmpOff,MPI_REPLACE));
    CHKERRQ(PetscSFBcastEnd(sf, MPIU_INT, neg, tmpOff,MPI_REPLACE));
    if (nroots > pEnd-pStart) {
      for (p = pStart; p < pEnd; ++p) {if (tmpOff[p] < 0) (*gsection)->atlasOff[p-pStart] = tmpOff[p];}
    }
  }
  if (nroots >= 0 && nroots > pEnd-pStart) CHKERRQ(PetscFree(tmpOff));
  CHKERRQ(PetscFree(neg));
  PetscFunctionReturn(0);
}

typedef struct _n_PetscSectionSym_Label
{
  DMLabel           label;
  PetscCopyMode     *modes;
  PetscInt          *sizes;
  const PetscInt    ***perms;
  const PetscScalar ***rots;
  PetscInt          (*minMaxOrients)[2];
  PetscInt          numStrata; /* numStrata is only increasing, functions as a state */
} PetscSectionSym_Label;

static PetscErrorCode PetscSectionSymLabelReset(PetscSectionSym sym)
{
  PetscInt              i, j;
  PetscSectionSym_Label *sl = (PetscSectionSym_Label *) sym->data;

  PetscFunctionBegin;
  for (i = 0; i <= sl->numStrata; i++) {
    if (sl->modes[i] == PETSC_OWN_POINTER || sl->modes[i] == PETSC_COPY_VALUES) {
      for (j = sl->minMaxOrients[i][0]; j < sl->minMaxOrients[i][1]; j++) {
        if (sl->perms[i]) CHKERRQ(PetscFree(sl->perms[i][j]));
        if (sl->rots[i]) CHKERRQ(PetscFree(sl->rots[i][j]));
      }
      if (sl->perms[i]) {
        const PetscInt **perms = &sl->perms[i][sl->minMaxOrients[i][0]];

        CHKERRQ(PetscFree(perms));
      }
      if (sl->rots[i]) {
        const PetscScalar **rots = &sl->rots[i][sl->minMaxOrients[i][0]];

        CHKERRQ(PetscFree(rots));
      }
    }
  }
  CHKERRQ(PetscFree5(sl->modes,sl->sizes,sl->perms,sl->rots,sl->minMaxOrients));
  CHKERRQ(DMLabelDestroy(&sl->label));
  sl->numStrata = 0;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSectionSymDestroy_Label(PetscSectionSym sym)
{
  PetscFunctionBegin;
  CHKERRQ(PetscSectionSymLabelReset(sym));
  CHKERRQ(PetscFree(sym->data));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSectionSymView_Label(PetscSectionSym sym, PetscViewer viewer)
{
  PetscSectionSym_Label *sl = (PetscSectionSym_Label *) sym->data;
  PetscBool             isAscii;
  DMLabel               label = sl->label;
  const char           *name;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &isAscii));
  if (isAscii) {
    PetscInt          i, j, k;
    PetscViewerFormat format;

    CHKERRQ(PetscViewerGetFormat(viewer,&format));
    if (label) {
      CHKERRQ(PetscViewerGetFormat(viewer,&format));
      if (format == PETSC_VIEWER_ASCII_INFO_DETAIL) {
        CHKERRQ(PetscViewerASCIIPushTab(viewer));
        CHKERRQ(DMLabelView(label, viewer));
        CHKERRQ(PetscViewerASCIIPopTab(viewer));
      } else {
        CHKERRQ(PetscObjectGetName((PetscObject) sl->label, &name));
        CHKERRQ(PetscViewerASCIIPrintf(viewer,"  Label '%s'\n",name));
      }
    } else {
      CHKERRQ(PetscViewerASCIIPrintf(viewer, "No label given\n"));
    }
    CHKERRQ(PetscViewerASCIIPushTab(viewer));
    for (i = 0; i <= sl->numStrata; i++) {
      PetscInt value = i < sl->numStrata ? label->stratumValues[i] : label->defaultValue;

      if (!(sl->perms[i] || sl->rots[i])) {
        CHKERRQ(PetscViewerASCIIPrintf(viewer, "Symmetry for stratum value %D (%D dofs per point): no symmetries\n", value, sl->sizes[i]));
      } else {
      CHKERRQ(PetscViewerASCIIPrintf(viewer, "Symmetry for stratum value %D (%D dofs per point):\n", value, sl->sizes[i]));
        CHKERRQ(PetscViewerASCIIPushTab(viewer));
        CHKERRQ(PetscViewerASCIIPrintf(viewer, "Orientation range: [%D, %D)\n", sl->minMaxOrients[i][0], sl->minMaxOrients[i][1]));
        if (format == PETSC_VIEWER_ASCII_INFO_DETAIL) {
          CHKERRQ(PetscViewerASCIIPushTab(viewer));
          for (j = sl->minMaxOrients[i][0]; j < sl->minMaxOrients[i][1]; j++) {
            if (!((sl->perms[i] && sl->perms[i][j]) || (sl->rots[i] && sl->rots[i][j]))) {
              CHKERRQ(PetscViewerASCIIPrintf(viewer, "Orientation %D: identity\n",j));
            } else {
              PetscInt tab;

              CHKERRQ(PetscViewerASCIIPrintf(viewer, "Orientation %D:\n",j));
              CHKERRQ(PetscViewerASCIIPushTab(viewer));
              CHKERRQ(PetscViewerASCIIGetTab(viewer,&tab));
              if (sl->perms[i] && sl->perms[i][j]) {
                CHKERRQ(PetscViewerASCIIPrintf(viewer,"Permutation:"));
                CHKERRQ(PetscViewerASCIISetTab(viewer,0));
                for (k = 0; k < sl->sizes[i]; k++) CHKERRQ(PetscViewerASCIIPrintf(viewer," %D",sl->perms[i][j][k]));
                CHKERRQ(PetscViewerASCIIPrintf(viewer,"\n"));
                CHKERRQ(PetscViewerASCIISetTab(viewer,tab));
              }
              if (sl->rots[i] && sl->rots[i][j]) {
                CHKERRQ(PetscViewerASCIIPrintf(viewer,"Rotations:  "));
                CHKERRQ(PetscViewerASCIISetTab(viewer,0));
#if defined(PETSC_USE_COMPLEX)
                for (k = 0; k < sl->sizes[i]; k++) CHKERRQ(PetscViewerASCIIPrintf(viewer," %+f+i*%+f",PetscRealPart(sl->rots[i][j][k]),PetscImaginaryPart(sl->rots[i][j][k])));
#else
                for (k = 0; k < sl->sizes[i]; k++) CHKERRQ(PetscViewerASCIIPrintf(viewer," %+f",sl->rots[i][j][k]));
#endif
                CHKERRQ(PetscViewerASCIIPrintf(viewer,"\n"));
                CHKERRQ(PetscViewerASCIISetTab(viewer,tab));
              }
              CHKERRQ(PetscViewerASCIIPopTab(viewer));
            }
          }
          CHKERRQ(PetscViewerASCIIPopTab(viewer));
        }
        CHKERRQ(PetscViewerASCIIPopTab(viewer));
      }
    }
    CHKERRQ(PetscViewerASCIIPopTab(viewer));
  }
  PetscFunctionReturn(0);
}

/*@
  PetscSectionSymLabelSetLabel - set the label whose strata will define the points that receive symmetries

  Logically collective on sym

  Input parameters:
+ sym - the section symmetries
- label - the DMLabel describing the types of points

  Level: developer:

.seealso: PetscSectionSymLabelSetStratum(), PetscSectionSymCreateLabel(), PetscSectionGetPointSyms()
@*/
PetscErrorCode PetscSectionSymLabelSetLabel(PetscSectionSym sym, DMLabel label)
{
  PetscSectionSym_Label *sl;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sym,PETSC_SECTION_SYM_CLASSID,1);
  sl = (PetscSectionSym_Label *) sym->data;
  if (sl->label && sl->label != label) CHKERRQ(PetscSectionSymLabelReset(sym));
  if (label) {
    sl->label = label;
    CHKERRQ(PetscObjectReference((PetscObject) label));
    CHKERRQ(DMLabelGetNumValues(label,&sl->numStrata));
    CHKERRQ(PetscMalloc5(sl->numStrata+1,&sl->modes,sl->numStrata+1,&sl->sizes,sl->numStrata+1,&sl->perms,sl->numStrata+1,&sl->rots,sl->numStrata+1,&sl->minMaxOrients));
    CHKERRQ(PetscMemzero((void *) sl->modes,(sl->numStrata+1)*sizeof(PetscCopyMode)));
    CHKERRQ(PetscMemzero((void *) sl->sizes,(sl->numStrata+1)*sizeof(PetscInt)));
    CHKERRQ(PetscMemzero((void *) sl->perms,(sl->numStrata+1)*sizeof(const PetscInt **)));
    CHKERRQ(PetscMemzero((void *) sl->rots,(sl->numStrata+1)*sizeof(const PetscScalar **)));
    CHKERRQ(PetscMemzero((void *) sl->minMaxOrients,(sl->numStrata+1)*sizeof(PetscInt[2])));
  }
  PetscFunctionReturn(0);
}

/*@C
  PetscSectionSymLabelGetStratum - get the symmetries for the orientations of a stratum

  Logically collective on sym

  Input Parameters:
+ sym       - the section symmetries
- stratum   - the stratum value in the label that we are assigning symmetries for

  Output Parameters:
+ size      - the number of dofs for points in the stratum of the label
. minOrient - the smallest orientation for a point in this stratum
. maxOrient - one greater than the largest orientation for a ppoint in this stratum (i.e., orientations are in the range [minOrient, maxOrient))
. perms     - NULL if there are no permutations, or (maxOrient - minOrient) permutations, one for each orientation.  A NULL permutation is the identity
- rots      - NULL if there are no rotations, or (maxOrient - minOrient) sets of rotations, one for each orientation.  A NULL set of orientations is the identity

  Level: developer

.seealso: PetscSectionSymLabelSetStratum(), PetscSectionSymCreate(), PetscSectionSetSym(), PetscSectionGetPointSyms(), PetscSectionSymCreateLabel()
@*/
PetscErrorCode PetscSectionSymLabelGetStratum(PetscSectionSym sym, PetscInt stratum, PetscInt *size, PetscInt *minOrient, PetscInt *maxOrient, const PetscInt ***perms, const PetscScalar ***rots)
{
  PetscSectionSym_Label *sl;
  const char            *name;
  PetscInt               i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sym,PETSC_SECTION_SYM_CLASSID,1);
  sl = (PetscSectionSym_Label *) sym->data;
  PetscCheck(sl->label, PetscObjectComm((PetscObject) sym), PETSC_ERR_ARG_WRONGSTATE, "No label set yet");
  for (i = 0; i <= sl->numStrata; i++) {
    PetscInt value = (i < sl->numStrata) ? sl->label->stratumValues[i] : sl->label->defaultValue;

    if (stratum == value) break;
  }
  CHKERRQ(PetscObjectGetName((PetscObject) sl->label, &name));
  PetscCheck(i <= sl->numStrata, PetscObjectComm((PetscObject) sym), PETSC_ERR_ARG_OUTOFRANGE, "Stratum %" PetscInt_FMT " not found in label %s", stratum, name);
  if (size)      {PetscValidIntPointer(size, 3);      *size      = sl->sizes[i];}
  if (minOrient) {PetscValidIntPointer(minOrient, 4); *minOrient = sl->minMaxOrients[i][0];}
  if (maxOrient) {PetscValidIntPointer(maxOrient, 5); *maxOrient = sl->minMaxOrients[i][1];}
  if (perms)     {PetscValidPointer(perms, 6);        *perms     = sl->perms[i] ? &sl->perms[i][sl->minMaxOrients[i][0]] : NULL;}
  if (rots)      {PetscValidPointer(rots, 7);         *rots      = sl->rots[i]  ? &sl->rots[i][sl->minMaxOrients[i][0]]  : NULL;}
  PetscFunctionReturn(0);
}

/*@C
  PetscSectionSymLabelSetStratum - set the symmetries for the orientations of a stratum

  Logically collective on sym

  InputParameters:
+ sym       - the section symmetries
. stratum   - the stratum value in the label that we are assigning symmetries for
. size      - the number of dofs for points in the stratum of the label
. minOrient - the smallest orientation for a point in this stratum
. maxOrient - one greater than the largest orientation for a ppoint in this stratum (i.e., orientations are in the range [minOrient, maxOrient))
. mode      - how sym should copy the perms and rots arrays
. perms     - NULL if there are no permutations, or (maxOrient - minOrient) permutations, one for each orientation.  A NULL permutation is the identity
- rots      - NULL if there are no rotations, or (maxOrient - minOrient) sets of rotations, one for each orientation.  A NULL set of orientations is the identity

  Level: developer

.seealso: PetscSectionSymLabelGetStratum(), PetscSectionSymCreate(), PetscSectionSetSym(), PetscSectionGetPointSyms(), PetscSectionSymCreateLabel()
@*/
PetscErrorCode PetscSectionSymLabelSetStratum(PetscSectionSym sym, PetscInt stratum, PetscInt size, PetscInt minOrient, PetscInt maxOrient, PetscCopyMode mode, const PetscInt **perms, const PetscScalar **rots)
{
  PetscSectionSym_Label *sl;
  const char            *name;
  PetscInt               i, j, k;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sym,PETSC_SECTION_SYM_CLASSID,1);
  sl = (PetscSectionSym_Label *) sym->data;
  PetscCheck(sl->label, PetscObjectComm((PetscObject) sym), PETSC_ERR_ARG_WRONGSTATE, "No label set yet");
  for (i = 0; i <= sl->numStrata; i++) {
    PetscInt value = (i < sl->numStrata) ? sl->label->stratumValues[i] : sl->label->defaultValue;

    if (stratum == value) break;
  }
  CHKERRQ(PetscObjectGetName((PetscObject) sl->label, &name));
  PetscCheck(i <= sl->numStrata, PetscObjectComm((PetscObject) sym), PETSC_ERR_ARG_OUTOFRANGE, "Stratum %D not found in label %s", stratum, name);
  sl->sizes[i] = size;
  sl->modes[i] = mode;
  sl->minMaxOrients[i][0] = minOrient;
  sl->minMaxOrients[i][1] = maxOrient;
  if (mode == PETSC_COPY_VALUES) {
    if (perms) {
      PetscInt    **ownPerms;

      CHKERRQ(PetscCalloc1(maxOrient - minOrient,&ownPerms));
      for (j = 0; j < maxOrient-minOrient; j++) {
        if (perms[j]) {
          CHKERRQ(PetscMalloc1(size,&ownPerms[j]));
          for (k = 0; k < size; k++) {ownPerms[j][k] = perms[j][k];}
        }
      }
      sl->perms[i] = (const PetscInt **) &ownPerms[-minOrient];
    }
    if (rots) {
      PetscScalar **ownRots;

      CHKERRQ(PetscCalloc1(maxOrient - minOrient,&ownRots));
      for (j = 0; j < maxOrient-minOrient; j++) {
        if (rots[j]) {
          CHKERRQ(PetscMalloc1(size,&ownRots[j]));
          for (k = 0; k < size; k++) {ownRots[j][k] = rots[j][k];}
        }
      }
      sl->rots[i] = (const PetscScalar **) &ownRots[-minOrient];
    }
  } else {
    sl->perms[i] = perms ? &perms[-minOrient] : NULL;
    sl->rots[i]  = rots ? &rots[-minOrient] : NULL;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSectionSymGetPoints_Label(PetscSectionSym sym, PetscSection section, PetscInt numPoints, const PetscInt *points, const PetscInt **perms, const PetscScalar **rots)
{
  PetscInt              i, j, numStrata;
  PetscSectionSym_Label *sl;
  DMLabel               label;

  PetscFunctionBegin;
  sl = (PetscSectionSym_Label *) sym->data;
  numStrata = sl->numStrata;
  label     = sl->label;
  for (i = 0; i < numPoints; i++) {
    PetscInt point = points[2*i];
    PetscInt ornt  = points[2*i+1];

    for (j = 0; j < numStrata; j++) {
      if (label->validIS[j]) {
        PetscInt k;

        CHKERRQ(ISLocate(label->points[j],point,&k));
        if (k >= 0) break;
      } else {
        PetscBool has;

        CHKERRQ(PetscHSetIHas(label->ht[j], point, &has));
        if (has) break;
      }
    }
    PetscCheckFalse((sl->minMaxOrients[j][1] > sl->minMaxOrients[j][0]) && (ornt < sl->minMaxOrients[j][0] || ornt >= sl->minMaxOrients[j][1]),PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"point %D orientation %D not in range [%D, %D) for stratum %D",point,ornt,sl->minMaxOrients[j][0],sl->minMaxOrients[j][1],j < numStrata ? label->stratumValues[j] : label->defaultValue);
    if (perms) {perms[i] = sl->perms[j] ? sl->perms[j][ornt] : NULL;}
    if (rots) {rots[i]  = sl->rots[j] ? sl->rots[j][ornt] : NULL;}
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSectionSymCopy_Label(PetscSectionSym sym, PetscSectionSym nsym)
{
  PetscSectionSym_Label *sl = (PetscSectionSym_Label *) nsym->data;
  IS                     valIS;
  const PetscInt        *values;
  PetscInt               Nv, v;

  PetscFunctionBegin;
  CHKERRQ(DMLabelGetNumValues(sl->label, &Nv));
  CHKERRQ(DMLabelGetValueIS(sl->label, &valIS));
  CHKERRQ(ISGetIndices(valIS, &values));
  for (v = 0; v < Nv; ++v) {
    const PetscInt      val = values[v];
    PetscInt            size, minOrient, maxOrient;
    const PetscInt    **perms;
    const PetscScalar **rots;

    CHKERRQ(PetscSectionSymLabelGetStratum(sym,  val, &size, &minOrient, &maxOrient, &perms, &rots));
    CHKERRQ(PetscSectionSymLabelSetStratum(nsym, val,  size,  minOrient,  maxOrient, PETSC_COPY_VALUES, perms, rots));
  }
  CHKERRQ(ISDestroy(&valIS));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSectionSymDistribute_Label(PetscSectionSym sym, PetscSF migrationSF, PetscSectionSym *dsym)
{
  PetscSectionSym_Label *sl = (PetscSectionSym_Label *) sym->data;
  DMLabel                dlabel;

  PetscFunctionBegin;
  CHKERRQ(DMLabelDistribute(sl->label, migrationSF, &dlabel));
  CHKERRQ(PetscSectionSymCreateLabel(PetscObjectComm((PetscObject) sym), dlabel, dsym));
  CHKERRQ(DMLabelDestroy(&dlabel));
  CHKERRQ(PetscSectionSymCopy(sym, *dsym));
  PetscFunctionReturn(0);
}

PetscErrorCode PetscSectionSymCreate_Label(PetscSectionSym sym)
{
  PetscSectionSym_Label *sl;

  PetscFunctionBegin;
  CHKERRQ(PetscNewLog(sym,&sl));
  sym->ops->getpoints  = PetscSectionSymGetPoints_Label;
  sym->ops->distribute = PetscSectionSymDistribute_Label;
  sym->ops->copy       = PetscSectionSymCopy_Label;
  sym->ops->view       = PetscSectionSymView_Label;
  sym->ops->destroy    = PetscSectionSymDestroy_Label;
  sym->data            = (void *) sl;
  PetscFunctionReturn(0);
}

/*@
  PetscSectionSymCreateLabel - Create a section symmetry that assigns one symmetry to each stratum of a label

  Collective

  Input Parameters:
+ comm - the MPI communicator for the new symmetry
- label - the label defining the strata

  Output Parameters:
. sym - the section symmetries

  Level: developer

.seealso: PetscSectionSymCreate(), PetscSectionSetSym(), PetscSectionGetSym(), PetscSectionSymLabelSetStratum(), PetscSectionGetPointSyms()
@*/
PetscErrorCode PetscSectionSymCreateLabel(MPI_Comm comm, DMLabel label, PetscSectionSym *sym)
{
  PetscFunctionBegin;
  CHKERRQ(DMInitializePackage());
  CHKERRQ(PetscSectionSymCreate(comm,sym));
  CHKERRQ(PetscSectionSymSetType(*sym,PETSCSECTIONSYMLABEL));
  CHKERRQ(PetscSectionSymLabelSetLabel(*sym,label));
  PetscFunctionReturn(0);
}
