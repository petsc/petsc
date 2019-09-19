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

.seealso: DMLabelDestroy()
@*/
PetscErrorCode DMLabelCreate(MPI_Comm comm, const char name[], DMLabel *label)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(label,2);
  ierr = DMInitializePackage();CHKERRQ(ierr);

  ierr = PetscHeaderCreate(*label,DMLABEL_CLASSID,"DMLabel","DMLabel","DM",comm,DMLabelDestroy,DMLabelView);CHKERRQ(ierr);

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
  ierr = PetscHMapICreate(&(*label)->hmap);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) *label, name);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  if (PetscLikely(v >= 0 && v < label->numStrata) && label->validIS[v]) return 0;
  PetscFunctionBegin;
  if (v < 0 || v >= label->numStrata) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Trying to access invalid stratum %D in DMLabelMakeValid_Private\n", v);
  ierr = PetscHSetIGetSize(label->ht[v], &label->stratumSizes[v]);CHKERRQ(ierr);
  ierr = PetscMalloc1(label->stratumSizes[v], &pointArray);CHKERRQ(ierr);
  ierr = PetscHSetIGetElems(label->ht[v], &off, pointArray);CHKERRQ(ierr);
  ierr = PetscHSetIClear(label->ht[v]);CHKERRQ(ierr);
  ierr = PetscSortInt(label->stratumSizes[v], pointArray);CHKERRQ(ierr);
  if (label->bt) {
    for (p = 0; p < label->stratumSizes[v]; ++p) {
      const PetscInt point = pointArray[p];
      if ((point < label->pStart) || (point >= label->pEnd)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Label point %D is not in [%D, %D)", point, label->pStart, label->pEnd);
      ierr = PetscBTSet(label->bt, point - label->pStart);CHKERRQ(ierr);
    }
  }
  ierr = ISCreateGeneral(PETSC_COMM_SELF, label->stratumSizes[v], pointArray, PETSC_OWN_POINTER, &is);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) is, "indices");CHKERRQ(ierr);
  label->points[v]  = is;
  label->validIS[v] = PETSC_TRUE;
  ierr = PetscObjectStateIncrease((PetscObject) label);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  for (v = 0; v < label->numStrata; v++){
    ierr = DMLabelMakeValid_Private(label, v);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  if (PetscLikely(v >= 0 && v < label->numStrata) && !label->validIS[v]) return 0;
  PetscFunctionBegin;
  if (v < 0 || v >= label->numStrata) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Trying to access invalid stratum %D in DMLabelMakeInvalid_Private\n", v);
  if (label->points[v]) {
    ierr = ISGetIndices(label->points[v], &points);CHKERRQ(ierr);
    for (p = 0; p < label->stratumSizes[v]; ++p) {
      ierr = PetscHSetIAdd(label->ht[v], points[p]);CHKERRQ(ierr);
    }
    ierr = ISRestoreIndices(label->points[v],&points);CHKERRQ(ierr);
    ierr = ISDestroy(&(label->points[v]));CHKERRQ(ierr);
  }
  label->validIS[v] = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#if !defined(DMLABEL_LOOKUP_THRESHOLD)
#define DMLABEL_LOOKUP_THRESHOLD 16
#endif

PETSC_STATIC_INLINE PetscErrorCode DMLabelLookupStratum(DMLabel label, PetscInt value, PetscInt *index)
{
  PetscInt       v;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  *index = -1;
  if (label->numStrata <= DMLABEL_LOOKUP_THRESHOLD) {
    for (v = 0; v < label->numStrata; ++v)
      if (label->stratumValues[v] == value) {*index = v; break;}
  } else {
    ierr = PetscHMapIGet(label->hmap, value, index);CHKERRQ(ierr);
  }
#if defined(PETSC_USE_DEBUG)
  { /* Check strata hash map consistency */
    PetscInt len, loc = -1;
    ierr = PetscHMapIGetSize(label->hmap, &len);CHKERRQ(ierr);
    if (len != label->numStrata) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Inconsistent strata hash map size");
    if (label->numStrata <= DMLABEL_LOOKUP_THRESHOLD) {
      ierr = PetscHMapIGet(label->hmap, value, &loc);CHKERRQ(ierr);
    } else {
      for (v = 0; v < label->numStrata; ++v)
        if (label->stratumValues[v] == value) {loc = v; break;}
    }
    if (loc != *index) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Inconsistent strata hash map lookup");
  }
#endif
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode DMLabelNewStratum(DMLabel label, PetscInt value, PetscInt *index)
{
  PetscInt       v;
  PetscInt      *tmpV;
  PetscInt      *tmpS;
  PetscHSetI    *tmpH, ht;
  IS            *tmpP, is;
  PetscBool     *tmpB;
  PetscHMapI     hmap = label->hmap;
  PetscErrorCode ierr;

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
    ierr = PetscMalloc((v+1)*sizeof(*tmpV), &tmpV);CHKERRQ(ierr);
    ierr = PetscMalloc((v+1)*sizeof(*tmpS), &tmpS);CHKERRQ(ierr);
    ierr = PetscMalloc((v+1)*sizeof(*tmpH), &tmpH);CHKERRQ(ierr);
    ierr = PetscMalloc((v+1)*sizeof(*tmpP), &tmpP);CHKERRQ(ierr);
    ierr = PetscMalloc((v+1)*sizeof(*tmpB), &tmpB);CHKERRQ(ierr);
    ierr = PetscArraycpy(tmpV, oldV, v);CHKERRQ(ierr);
    ierr = PetscArraycpy(tmpS, oldS, v);CHKERRQ(ierr);
    ierr = PetscArraycpy(tmpH, oldH, v);CHKERRQ(ierr);
    ierr = PetscArraycpy(tmpP, oldP, v);CHKERRQ(ierr);
    ierr = PetscArraycpy(tmpB, oldB, v);CHKERRQ(ierr);
    ierr = PetscFree(oldV);CHKERRQ(ierr);
    ierr = PetscFree(oldS);CHKERRQ(ierr);
    ierr = PetscFree(oldH);CHKERRQ(ierr);
    ierr = PetscFree(oldP);CHKERRQ(ierr);
    ierr = PetscFree(oldB);CHKERRQ(ierr);
  }
  label->numStrata     = v+1;
  label->stratumValues = tmpV;
  label->stratumSizes  = tmpS;
  label->ht            = tmpH;
  label->points        = tmpP;
  label->validIS       = tmpB;
  ierr = PetscHSetICreate(&ht);CHKERRQ(ierr);
  ierr = ISCreateStride(PETSC_COMM_SELF,0,0,1,&is);CHKERRQ(ierr);
  ierr = PetscHMapISet(hmap, value, v);CHKERRQ(ierr);
  tmpV[v] = value;
  tmpS[v] = 0;
  tmpH[v] = ht;
  tmpP[v] = is;
  tmpB[v] = PETSC_TRUE;
  ierr = PetscObjectStateIncrease((PetscObject) label);CHKERRQ(ierr);
  *index = v;
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode DMLabelLookupAddStratum(DMLabel label, PetscInt value, PetscInt *index)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = DMLabelLookupStratum(label, value, index);CHKERRQ(ierr);
  if (*index < 0) {ierr = DMLabelNewStratum(label, value, index);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

/*@
  DMLabelAddStratum - Adds a new stratum value in a DMLabel

  Input Parameter:
+ label - The DMLabel
- value - The stratum value

  Level: beginner

.seealso:  DMLabelCreate(), DMLabelDestroy()
@*/
PetscErrorCode DMLabelAddStratum(DMLabel label, PetscInt value)
{
  PetscInt       v;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 1);
  ierr = DMLabelLookupAddStratum(label, value, &v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  DMLabelAddStrata - Adds new stratum values in a DMLabel

  Not collective

  Input Parameter:
+ label - The DMLabel
. numStrata - The number of stratum values
- stratumValues - The stratum values

  Level: beginner

.seealso:  DMLabelCreate(), DMLabelDestroy()
@*/
PetscErrorCode DMLabelAddStrata(DMLabel label, PetscInt numStrata, const PetscInt stratumValues[])
{
  PetscInt       *values, v;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 1);
  if (numStrata) PetscValidIntPointer(stratumValues, 3);
  ierr = PetscMalloc1(numStrata, &values);CHKERRQ(ierr);
  ierr = PetscArraycpy(values, stratumValues, numStrata);CHKERRQ(ierr);
  ierr = PetscSortRemoveDupsInt(&numStrata, values);CHKERRQ(ierr);
  if (!label->numStrata) { /* Fast preallocation */
    PetscInt   *tmpV;
    PetscInt   *tmpS;
    PetscHSetI *tmpH, ht;
    IS         *tmpP, is;
    PetscBool  *tmpB;
    PetscHMapI  hmap = label->hmap;

    ierr = PetscMalloc1(numStrata, &tmpV);CHKERRQ(ierr);
    ierr = PetscMalloc1(numStrata, &tmpS);CHKERRQ(ierr);
    ierr = PetscMalloc1(numStrata, &tmpH);CHKERRQ(ierr);
    ierr = PetscMalloc1(numStrata, &tmpP);CHKERRQ(ierr);
    ierr = PetscMalloc1(numStrata, &tmpB);CHKERRQ(ierr);
    label->numStrata     = numStrata;
    label->stratumValues = tmpV;
    label->stratumSizes  = tmpS;
    label->ht            = tmpH;
    label->points        = tmpP;
    label->validIS       = tmpB;
    for (v = 0; v < numStrata; ++v) {
      ierr = PetscHSetICreate(&ht);CHKERRQ(ierr);
      ierr = ISCreateStride(PETSC_COMM_SELF,0,0,1,&is);CHKERRQ(ierr);
      ierr = PetscHMapISet(hmap, values[v], v);CHKERRQ(ierr);
      tmpV[v] = values[v];
      tmpS[v] = 0;
      tmpH[v] = ht;
      tmpP[v] = is;
      tmpB[v] = PETSC_TRUE;
    }
    ierr = PetscObjectStateIncrease((PetscObject) label);CHKERRQ(ierr);
  } else {
    for (v = 0; v < numStrata; ++v) {
      ierr = DMLabelAddStratum(label, values[v]);CHKERRQ(ierr);
    }
  }
  ierr = PetscFree(values);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  DMLabelAddStrataIS - Adds new stratum values in a DMLabel

  Not collective

  Input Parameter:
+ label - The DMLabel
- valueIS - Index set with stratum values

  Level: beginner

.seealso:  DMLabelCreate(), DMLabelDestroy()
@*/
PetscErrorCode DMLabelAddStrataIS(DMLabel label, IS valueIS)
{
  PetscInt       numStrata;
  const PetscInt *stratumValues;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 1);
  PetscValidHeaderSpecific(valueIS, IS_CLASSID, 2);
  ierr = ISGetLocalSize(valueIS, &numStrata);CHKERRQ(ierr);
  ierr = ISGetIndices(valueIS, &stratumValues);CHKERRQ(ierr);
  ierr = DMLabelAddStrata(label, numStrata, stratumValues);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMLabelView_Ascii(DMLabel label, PetscViewer viewer)
{
  PetscInt       v;
  PetscMPIInt    rank;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)viewer), &rank);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPushSynchronized(viewer);CHKERRQ(ierr);
  if (label) {
    const char *name;

    ierr = PetscObjectGetName((PetscObject) label, &name);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "Label '%s':\n", name);CHKERRQ(ierr);
    if (label->bt) {ierr = PetscViewerASCIIPrintf(viewer, "  Index has been calculated in [%D, %D)\n", label->pStart, label->pEnd);CHKERRQ(ierr);}
    for (v = 0; v < label->numStrata; ++v) {
      const PetscInt value = label->stratumValues[v];
      const PetscInt *points;
      PetscInt       p;

      ierr = ISGetIndices(label->points[v], &points);CHKERRQ(ierr);
      for (p = 0; p < label->stratumSizes[v]; ++p) {
        ierr = PetscViewerASCIISynchronizedPrintf(viewer, "[%d]: %D (%D)\n", rank, points[p], value);CHKERRQ(ierr);
      }
      ierr = ISRestoreIndices(label->points[v],&points);CHKERRQ(ierr);
    }
  }
  ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPopSynchronized(viewer);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 1);
  if (!viewer) {ierr = PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)label), &viewer);CHKERRQ(ierr);}
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  if (label) {ierr = DMLabelMakeAllValid_Private(label);CHKERRQ(ierr);}
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = DMLabelView_Ascii(label, viewer);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 1);
  for (v = 0; v < label->numStrata; ++v) {
    ierr = PetscHSetIDestroy(&label->ht[v]);CHKERRQ(ierr);
    ierr = ISDestroy(&label->points[v]);CHKERRQ(ierr);
  }
  label->numStrata = 0;
  ierr = PetscFree(label->stratumValues);CHKERRQ(ierr);
  ierr = PetscFree(label->stratumSizes);CHKERRQ(ierr);
  ierr = PetscFree(label->ht);CHKERRQ(ierr);
  ierr = PetscFree(label->points);CHKERRQ(ierr);
  ierr = PetscFree(label->validIS);CHKERRQ(ierr);
  ierr = PetscHMapIReset(label->hmap);CHKERRQ(ierr);
  label->pStart = -1;
  label->pEnd   = -1;
  ierr = PetscBTDestroy(&label->bt);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*label) PetscFunctionReturn(0);
  PetscValidHeaderSpecific((*label),DMLABEL_CLASSID,1);
  if (--((PetscObject)(*label))->refct > 0) {*label = NULL; PetscFunctionReturn(0);}
  ierr = DMLabelReset(*label);CHKERRQ(ierr);
  ierr = PetscHMapIDestroy(&(*label)->hmap);CHKERRQ(ierr);
  ierr = PetscHeaderDestroy(label);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 1);
  ierr = DMLabelMakeAllValid_Private(label);CHKERRQ(ierr);
  ierr = PetscObjectGetName((PetscObject) label, &name);CHKERRQ(ierr);
  ierr = DMLabelCreate(PetscObjectComm((PetscObject) label), name, labelnew);CHKERRQ(ierr);

  (*labelnew)->numStrata    = label->numStrata;
  (*labelnew)->defaultValue = label->defaultValue;
  ierr = PetscMalloc1(label->numStrata, &(*labelnew)->stratumValues);CHKERRQ(ierr);
  ierr = PetscMalloc1(label->numStrata, &(*labelnew)->stratumSizes);CHKERRQ(ierr);
  ierr = PetscMalloc1(label->numStrata, &(*labelnew)->ht);CHKERRQ(ierr);
  ierr = PetscMalloc1(label->numStrata, &(*labelnew)->points);CHKERRQ(ierr);
  ierr = PetscMalloc1(label->numStrata, &(*labelnew)->validIS);CHKERRQ(ierr);
  for (v = 0; v < label->numStrata; ++v) {
    ierr = PetscHSetICreate(&(*labelnew)->ht[v]);CHKERRQ(ierr);
    (*labelnew)->stratumValues[v]  = label->stratumValues[v];
    (*labelnew)->stratumSizes[v]   = label->stratumSizes[v];
    ierr = PetscObjectReference((PetscObject) (label->points[v]));CHKERRQ(ierr);
    (*labelnew)->points[v]         = label->points[v];
    (*labelnew)->validIS[v]        = PETSC_TRUE;
  }
  ierr = PetscHMapIDestroy(&(*labelnew)->hmap);CHKERRQ(ierr);
  ierr = PetscHMapIDuplicate(label->hmap,&(*labelnew)->hmap);CHKERRQ(ierr);
  (*labelnew)->pStart = -1;
  (*labelnew)->pEnd   = -1;
  (*labelnew)->bt     = NULL;
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 1);
  ierr = DMLabelMakeAllValid_Private(label);CHKERRQ(ierr);
  for (v = 0; v < label->numStrata; ++v) {
    const PetscInt *points;
    PetscInt       i;

    ierr = ISGetIndices(label->points[v], &points);CHKERRQ(ierr);
    for (i = 0; i < label->stratumSizes[v]; ++i) {
      const PetscInt point = points[i];

      pStart = PetscMin(point,   pStart);
      pEnd   = PetscMax(point+1, pEnd);
    }
    ierr = ISRestoreIndices(label->points[v], &points);CHKERRQ(ierr);
  }
  label->pStart = pStart == PETSC_MAX_INT ? -1 : pStart;
  label->pEnd   = pEnd;
  ierr = DMLabelCreateIndex(label, label->pStart, label->pEnd);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 1);
  ierr = DMLabelDestroyIndex(label);CHKERRQ(ierr);
  ierr = DMLabelMakeAllValid_Private(label);CHKERRQ(ierr);
  label->pStart = pStart;
  label->pEnd   = pEnd;
  /* This can be hooked into SetValue(),  ClearValue(), etc. for updating */
  ierr = PetscBTCreate(pEnd - pStart, &label->bt);CHKERRQ(ierr);
  for (v = 0; v < label->numStrata; ++v) {
    const PetscInt *points;
    PetscInt       i;

    ierr = ISGetIndices(label->points[v], &points);CHKERRQ(ierr);
    for (i = 0; i < label->stratumSizes[v]; ++i) {
      const PetscInt point = points[i];

      if ((point < pStart) || (point >= pEnd)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Label point %D is not in [%D, %D)", point, pStart, pEnd);
      ierr = PetscBTSet(label->bt, point - pStart);CHKERRQ(ierr);
    }
    ierr = ISRestoreIndices(label->points[v], &points);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 1);
  label->pStart = -1;
  label->pEnd   = -1;
  ierr = PetscBTDestroy(&label->bt);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 1);
  if ((label->pStart == -1) && (label->pEnd == -1)) {ierr = DMLabelComputeIndex(label);CHKERRQ(ierr);}
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 1);
  PetscValidBoolPointer(contains, 3);
  ierr = DMLabelLookupStratum(label, value, &v);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBeginHot;
  PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 1);
  PetscValidBoolPointer(contains, 3);
  ierr = DMLabelMakeAllValid_Private(label);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
  if (!label->bt) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Must call DMLabelCreateIndex() before DMLabelHasPoint()");
  if ((point < label->pStart) || (point >= label->pEnd)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Label point %D is not in [%D, %D)", point, label->pStart, label->pEnd);
#endif
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
  PetscErrorCode ierr;

  PetscFunctionBeginHot;
  PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 1);
  PetscValidBoolPointer(contains, 4);
  *contains = PETSC_FALSE;
  ierr = DMLabelLookupStratum(label, value, &v);CHKERRQ(ierr);
  if (v < 0) PetscFunctionReturn(0);

  if (label->validIS[v]) {
    PetscInt i;

    ierr = ISLocate(label->points[v], point, &i);CHKERRQ(ierr);
    if (i >= 0) *contains = PETSC_TRUE;
  } else {
    PetscBool has;

    ierr = PetscHSetIHas(label->ht[v], point, &has);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBeginHot;
  PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 1);
  PetscValidPointer(value, 3);
  *value = label->defaultValue;
  for (v = 0; v < label->numStrata; ++v) {
    if (label->validIS[v]) {
      PetscInt i;

      ierr = ISLocate(label->points[v], point, &i);CHKERRQ(ierr);
      if (i >= 0) {
        *value = label->stratumValues[v];
        break;
      }
    } else {
      PetscBool has;

      ierr = PetscHSetIHas(label->ht[v], point, &has);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 1);
  /* Find label value, add new entry if needed */
  if (value == label->defaultValue) PetscFunctionReturn(0);
  ierr = DMLabelLookupAddStratum(label, value, &v);CHKERRQ(ierr);
  /* Set key */
  ierr = DMLabelMakeInvalid_Private(label, v);CHKERRQ(ierr);
  ierr = PetscHSetIAdd(label->ht[v], point);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 1);
  /* Find label value */
  ierr = DMLabelLookupStratum(label, value, &v);CHKERRQ(ierr);
  if (v < 0) PetscFunctionReturn(0);

  if (label->bt) {
    if ((point < label->pStart) || (point >= label->pEnd)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Label point %D is not in [%D, %D)", point, label->pStart, label->pEnd);
    ierr = PetscBTClear(label->bt, point - label->pStart);CHKERRQ(ierr);
  }

  /* Delete key */
  ierr = DMLabelMakeInvalid_Private(label, v);CHKERRQ(ierr);
  ierr = PetscHSetIDel(label->ht[v], point);CHKERRQ(ierr);
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
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 1);
  PetscValidHeaderSpecific(is, IS_CLASSID, 2);
  /* Find label value, add new entry if needed */
  if (value == label->defaultValue) PetscFunctionReturn(0);
  ierr = DMLabelLookupAddStratum(label, value, &v);CHKERRQ(ierr);
  /* Set keys */
  ierr = DMLabelMakeInvalid_Private(label, v);CHKERRQ(ierr);
  ierr = ISGetLocalSize(is, &n);CHKERRQ(ierr);
  ierr = ISGetIndices(is, &points);CHKERRQ(ierr);
  for (p = 0; p < n; ++p) {ierr = PetscHSetIAdd(label->ht[v], points[p]);CHKERRQ(ierr);}
  ierr = ISRestoreIndices(is, &points);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  DMLabelGetNumValues - Get the number of values that the DMLabel takes

  Not collective

  Input Parameter:
. label - the DMLabel

  Output Paramater:
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

  Output Paramater:
. is    - the value IS

  Level: intermediate

.seealso: DMLabelCreate(), DMLabelGetValue(), DMLabelSetValue(), DMLabelClearValue()
@*/
PetscErrorCode DMLabelGetValueIS(DMLabel label, IS *values)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 1);
  PetscValidPointer(values, 2);
  ierr = ISCreateGeneral(PETSC_COMM_SELF, label->numStrata, label->stratumValues, PETSC_USE_POINTER, values);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  DMLabelHasStratum - Determine whether points exist with the given value

  Not collective

  Input Parameters:
+ label - the DMLabel
- value - the stratum value

  Output Paramater:
. exists - Flag saying whether points exist

  Level: intermediate

.seealso: DMLabelCreate(), DMLabelGetValue(), DMLabelSetValue(), DMLabelClearValue()
@*/
PetscErrorCode DMLabelHasStratum(DMLabel label, PetscInt value, PetscBool *exists)
{
  PetscInt       v;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 1);
  PetscValidPointer(exists, 3);
  ierr = DMLabelLookupStratum(label, value, &v);CHKERRQ(ierr);
  *exists = v < 0 ? PETSC_FALSE : PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*@
  DMLabelGetStratumSize - Get the size of a stratum

  Not collective

  Input Parameters:
+ label - the DMLabel
- value - the stratum value

  Output Paramater:
. size - The number of points in the stratum

  Level: intermediate

.seealso: DMLabelCreate(), DMLabelGetValue(), DMLabelSetValue(), DMLabelClearValue()
@*/
PetscErrorCode DMLabelGetStratumSize(DMLabel label, PetscInt value, PetscInt *size)
{
  PetscInt       v;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 1);
  PetscValidPointer(size, 3);
  *size = 0;
  ierr = DMLabelLookupStratum(label, value, &v);CHKERRQ(ierr);
  if (v < 0) PetscFunctionReturn(0);
  ierr = DMLabelMakeValid_Private(label, v);CHKERRQ(ierr);
  *size = label->stratumSizes[v];
  PetscFunctionReturn(0);
}

/*@
  DMLabelGetStratumBounds - Get the largest and smallest point of a stratum

  Not collective

  Input Parameters:
+ label - the DMLabel
- value - the stratum value

  Output Paramaters:
+ start - the smallest point in the stratum
- end - the largest point in the stratum

  Level: intermediate

.seealso: DMLabelCreate(), DMLabelGetValue(), DMLabelSetValue(), DMLabelClearValue()
@*/
PetscErrorCode DMLabelGetStratumBounds(DMLabel label, PetscInt value, PetscInt *start, PetscInt *end)
{
  PetscInt       v, min, max;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 1);
  if (start) {PetscValidPointer(start, 3); *start = 0;}
  if (end)   {PetscValidPointer(end,   4); *end   = 0;}
  ierr = DMLabelLookupStratum(label, value, &v);CHKERRQ(ierr);
  if (v < 0) PetscFunctionReturn(0);
  ierr = DMLabelMakeValid_Private(label, v);CHKERRQ(ierr);
  if (label->stratumSizes[v] <= 0) PetscFunctionReturn(0);
  ierr = ISGetMinMax(label->points[v], &min, &max);CHKERRQ(ierr);
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

  Output Paramater:
. points - The stratum points

  Level: intermediate

.seealso: DMLabelCreate(), DMLabelGetValue(), DMLabelSetValue(), DMLabelClearValue()
@*/
PetscErrorCode DMLabelGetStratumIS(DMLabel label, PetscInt value, IS *points)
{
  PetscInt       v;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 1);
  PetscValidPointer(points, 3);
  *points = NULL;
  ierr = DMLabelLookupStratum(label, value, &v);CHKERRQ(ierr);
  if (v < 0) PetscFunctionReturn(0);
  ierr = DMLabelMakeValid_Private(label, v);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject) label->points[v]);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 1);
  PetscValidHeaderSpecific(is, IS_CLASSID, 3);
  ierr = DMLabelLookupAddStratum(label, value, &v);CHKERRQ(ierr);
  if (is == label->points[v]) PetscFunctionReturn(0);
  ierr = DMLabelClearStratum(label, value);CHKERRQ(ierr);
  ierr = ISGetLocalSize(is, &(label->stratumSizes[v]));CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)is);CHKERRQ(ierr);
  ierr = ISDestroy(&(label->points[v]));CHKERRQ(ierr);
  label->points[v]  = is;
  label->validIS[v] = PETSC_TRUE;
  ierr = PetscObjectStateIncrease((PetscObject) label);CHKERRQ(ierr);
  if (label->bt) {
    const PetscInt *points;
    PetscInt p;

    ierr = ISGetIndices(is,&points);CHKERRQ(ierr);
    for (p = 0; p < label->stratumSizes[v]; ++p) {
      const PetscInt point = points[p];

      if ((point < label->pStart) || (point >= label->pEnd)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Label point %D is not in [%D, %D)", point, label->pStart, label->pEnd);
      ierr = PetscBTSet(label->bt, point - label->pStart);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 1);
  ierr = DMLabelLookupStratum(label, value, &v);CHKERRQ(ierr);
  if (v < 0) PetscFunctionReturn(0);
  if (label->validIS[v]) {
    if (label->bt) {
      PetscInt       i;
      const PetscInt *points;

      ierr = ISGetIndices(label->points[v], &points);CHKERRQ(ierr);
      for (i = 0; i < label->stratumSizes[v]; ++i) {
        const PetscInt point = points[i];

        if ((point < label->pStart) || (point >= label->pEnd)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Label point %D is not in [%D, %D)", point, label->pStart, label->pEnd);
        ierr = PetscBTClear(label->bt, point - label->pStart);CHKERRQ(ierr);
      }
      ierr = ISRestoreIndices(label->points[v], &points);CHKERRQ(ierr);
    }
    label->stratumSizes[v] = 0;
    ierr = ISDestroy(&label->points[v]);CHKERRQ(ierr);
    ierr = ISCreateStride(PETSC_COMM_SELF, 0, 0, 1, &label->points[v]);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) label->points[v], "indices");CHKERRQ(ierr);
    ierr = PetscObjectStateIncrease((PetscObject) label);CHKERRQ(ierr);
  } else {
    ierr = PetscHSetIClear(label->ht[v]);CHKERRQ(ierr);
  }
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 1);
  ierr = DMLabelDestroyIndex(label);CHKERRQ(ierr);
  ierr = DMLabelMakeAllValid_Private(label);CHKERRQ(ierr);
  for (v = 0; v < label->numStrata; ++v) {
    ierr = ISGeneralFilter(label->points[v], start, end);CHKERRQ(ierr);
  }
  ierr = DMLabelCreateIndex(label, start, end);CHKERRQ(ierr);
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
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 1);
  PetscValidHeaderSpecific(permutation, IS_CLASSID, 2);
  ierr = DMLabelMakeAllValid_Private(label);CHKERRQ(ierr);
  ierr = DMLabelDuplicate(label, labelNew);CHKERRQ(ierr);
  ierr = DMLabelGetNumValues(*labelNew, &numValues);CHKERRQ(ierr);
  ierr = ISGetLocalSize(permutation, &numPoints);CHKERRQ(ierr);
  ierr = ISGetIndices(permutation, &perm);CHKERRQ(ierr);
  for (v = 0; v < numValues; ++v) {
    const PetscInt size   = (*labelNew)->stratumSizes[v];
    const PetscInt *points;
    PetscInt *pointsNew;

    ierr = ISGetIndices((*labelNew)->points[v],&points);CHKERRQ(ierr);
    ierr = PetscMalloc1(size,&pointsNew);CHKERRQ(ierr);
    for (q = 0; q < size; ++q) {
      const PetscInt point = points[q];

      if ((point < 0) || (point >= numPoints)) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Label point %D is not in [0, %D) for the remapping", point, numPoints);
      pointsNew[q] = perm[point];
    }
    ierr = ISRestoreIndices((*labelNew)->points[v],&points);CHKERRQ(ierr);
    ierr = PetscSortInt(size, pointsNew);CHKERRQ(ierr);
    ierr = ISDestroy(&((*labelNew)->points[v]));CHKERRQ(ierr);
    if (size > 0 && pointsNew[size - 1] == pointsNew[0] + size - 1) {
      ierr = ISCreateStride(PETSC_COMM_SELF,size,pointsNew[0],1,&((*labelNew)->points[v]));CHKERRQ(ierr);
      ierr = PetscFree(pointsNew);CHKERRQ(ierr);
    } else {
      ierr = ISCreateGeneral(PETSC_COMM_SELF,size,pointsNew,PETSC_OWN_POINTER,&((*labelNew)->points[v]));CHKERRQ(ierr);
    }
    ierr = PetscObjectSetName((PetscObject) ((*labelNew)->points[v]), "indices");CHKERRQ(ierr);
  }
  ierr = ISRestoreIndices(permutation, &perm);CHKERRQ(ierr);
  if (label->bt) {
    ierr = PetscBTDestroy(&label->bt);CHKERRQ(ierr);
    ierr = DMLabelCreateIndex(label, label->pStart, label->pEnd);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (label) {ierr = DMLabelMakeAllValid_Private(label);CHKERRQ(ierr);}
  ierr = PetscObjectGetComm((PetscObject)sf, &comm);CHKERRQ(ierr);
  /* Build a section of stratum values per point, generate the according SF
     and distribute point-wise stratum values to leaves. */
  ierr = PetscSFGetGraph(sf, &nroots, &nleaves, NULL, NULL);CHKERRQ(ierr);
  ierr = PetscSectionCreate(comm, &rootSection);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(rootSection, 0, nroots);CHKERRQ(ierr);
  if (label) {
    for (s = 0; s < label->numStrata; ++s) {
      const PetscInt *points;

      ierr = ISGetIndices(label->points[s], &points);CHKERRQ(ierr);
      for (l = 0; l < label->stratumSizes[s]; l++) {
        ierr = PetscSectionGetDof(rootSection, points[l], &dof);CHKERRQ(ierr);
        ierr = PetscSectionSetDof(rootSection, points[l], dof+1);CHKERRQ(ierr);
      }
      ierr = ISRestoreIndices(label->points[s], &points);CHKERRQ(ierr);
    }
  }
  ierr = PetscSectionSetUp(rootSection);CHKERRQ(ierr);
  /* Create a point-wise array of stratum values */
  ierr = PetscSectionGetStorageSize(rootSection, &size);CHKERRQ(ierr);
  ierr = PetscMalloc1(size, &rootStrata);CHKERRQ(ierr);
  ierr = PetscCalloc1(nroots, &rootIdx);CHKERRQ(ierr);
  if (label) {
    for (s = 0; s < label->numStrata; ++s) {
      const PetscInt *points;

      ierr = ISGetIndices(label->points[s], &points);CHKERRQ(ierr);
      for (l = 0; l < label->stratumSizes[s]; l++) {
        const PetscInt p = points[l];
        ierr = PetscSectionGetOffset(rootSection, p, &offset);CHKERRQ(ierr);
        rootStrata[offset+rootIdx[p]++] = label->stratumValues[s];
      }
      ierr = ISRestoreIndices(label->points[s], &points);CHKERRQ(ierr);
    }
  }
  /* Build SF that maps label points to remote processes */
  ierr = PetscSectionCreate(comm, leafSection);CHKERRQ(ierr);
  ierr = PetscSFDistributeSection(sf, rootSection, &remoteOffsets, *leafSection);CHKERRQ(ierr);
  ierr = PetscSFCreateSectionSF(sf, rootSection, remoteOffsets, *leafSection, &labelSF);CHKERRQ(ierr);
  ierr = PetscFree(remoteOffsets);CHKERRQ(ierr);
  /* Send the strata for each point over the derived SF */
  ierr = PetscSectionGetStorageSize(*leafSection, &size);CHKERRQ(ierr);
  ierr = PetscMalloc1(size, leafStrata);CHKERRQ(ierr);
  ierr = PetscSFBcastBegin(labelSF, MPIU_INT, rootStrata, *leafStrata);CHKERRQ(ierr);
  ierr = PetscSFBcastEnd(labelSF, MPIU_INT, rootStrata, *leafStrata);CHKERRQ(ierr);
  /* Clean up */
  ierr = PetscFree(rootStrata);CHKERRQ(ierr);
  ierr = PetscFree(rootIdx);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&rootSection);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&labelSF);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf, PETSCSF_CLASSID, 2);
  if (label) {
    PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 1);
    ierr = DMLabelMakeAllValid_Private(label);CHKERRQ(ierr);
  }
  ierr = PetscObjectGetComm((PetscObject)sf, &comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  /* Bcast name */
  if (!rank) {
    ierr = PetscObjectGetName((PetscObject) label, &lname);CHKERRQ(ierr);
    ierr = PetscStrlen(lname, &len);CHKERRQ(ierr);
  }
  nameSize = len;
  ierr = MPI_Bcast(&nameSize, 1, MPIU_INT, 0, comm);CHKERRQ(ierr);
  ierr = PetscMalloc1(nameSize+1, &name);CHKERRQ(ierr);
  if (!rank) {ierr = PetscArraycpy(name, lname, nameSize+1);CHKERRQ(ierr);}
  ierr = MPI_Bcast(name, nameSize+1, MPI_CHAR, 0, comm);CHKERRQ(ierr);
  ierr = DMLabelCreate(PETSC_COMM_SELF, name, labelNew);CHKERRQ(ierr);
  ierr = PetscFree(name);CHKERRQ(ierr);
  /* Bcast defaultValue */
  if (!rank) (*labelNew)->defaultValue = label->defaultValue;
  ierr = MPI_Bcast(&(*labelNew)->defaultValue, 1, MPIU_INT, 0, comm);CHKERRQ(ierr);
  /* Distribute stratum values over the SF and get the point mapping on the receiver */
  ierr = DMLabelDistribute_Internal(label, sf, &leafSection, &leafStrata);CHKERRQ(ierr);
  /* Determine received stratum values and initialise new label*/
  ierr = PetscHSetICreate(&stratumHash);CHKERRQ(ierr);
  ierr = PetscSectionGetStorageSize(leafSection, &size);CHKERRQ(ierr);
  for (p = 0; p < size; ++p) {ierr = PetscHSetIAdd(stratumHash, leafStrata[p]);CHKERRQ(ierr);}
  ierr = PetscHSetIGetSize(stratumHash, &(*labelNew)->numStrata);CHKERRQ(ierr);
  ierr = PetscMalloc1((*labelNew)->numStrata, &(*labelNew)->validIS);CHKERRQ(ierr);
  for (s = 0; s < (*labelNew)->numStrata; ++s) (*labelNew)->validIS[s] = PETSC_TRUE;
  ierr = PetscMalloc1((*labelNew)->numStrata, &(*labelNew)->stratumValues);CHKERRQ(ierr);
  /* Turn leafStrata into indices rather than stratum values */
  offset = 0;
  ierr = PetscHSetIGetElems(stratumHash, &offset, (*labelNew)->stratumValues);CHKERRQ(ierr);
  ierr = PetscSortInt((*labelNew)->numStrata,(*labelNew)->stratumValues);CHKERRQ(ierr);
  for (s = 0; s < (*labelNew)->numStrata; ++s) {
    ierr = PetscHMapISet((*labelNew)->hmap, (*labelNew)->stratumValues[s], s);CHKERRQ(ierr);
  }
  for (p = 0; p < size; ++p) {
    for (s = 0; s < (*labelNew)->numStrata; ++s) {
      if (leafStrata[p] == (*labelNew)->stratumValues[s]) {leafStrata[p] = s; break;}
    }
  }
  /* Rebuild the point strata on the receiver */
  ierr = PetscCalloc1((*labelNew)->numStrata,&(*labelNew)->stratumSizes);CHKERRQ(ierr);
  ierr = PetscSectionGetChart(leafSection, &pStart, &pEnd);CHKERRQ(ierr);
  for (p=pStart; p<pEnd; p++) {
    ierr = PetscSectionGetDof(leafSection, p, &dof);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(leafSection, p, &offset);CHKERRQ(ierr);
    for (s=0; s<dof; s++) {
      (*labelNew)->stratumSizes[leafStrata[offset+s]]++;
    }
  }
  ierr = PetscCalloc1((*labelNew)->numStrata,&(*labelNew)->ht);CHKERRQ(ierr);
  ierr = PetscMalloc1((*labelNew)->numStrata,&(*labelNew)->points);CHKERRQ(ierr);
  ierr = PetscMalloc1((*labelNew)->numStrata,&points);CHKERRQ(ierr);
  for (s = 0; s < (*labelNew)->numStrata; ++s) {
    ierr = PetscHSetICreate(&(*labelNew)->ht[s]);CHKERRQ(ierr);
    ierr = PetscMalloc1((*labelNew)->stratumSizes[s], &(points[s]));CHKERRQ(ierr);
  }
  /* Insert points into new strata */
  ierr = PetscCalloc1((*labelNew)->numStrata, &strataIdx);CHKERRQ(ierr);
  ierr = PetscSectionGetChart(leafSection, &pStart, &pEnd);CHKERRQ(ierr);
  for (p=pStart; p<pEnd; p++) {
    ierr = PetscSectionGetDof(leafSection, p, &dof);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(leafSection, p, &offset);CHKERRQ(ierr);
    for (s=0; s<dof; s++) {
      stratum = leafStrata[offset+s];
      points[stratum][strataIdx[stratum]++] = p;
    }
  }
  for (s = 0; s < (*labelNew)->numStrata; s++) {
    ierr = ISCreateGeneral(PETSC_COMM_SELF,(*labelNew)->stratumSizes[s],&(points[s][0]),PETSC_OWN_POINTER,&((*labelNew)->points[s]));CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)((*labelNew)->points[s]),"indices");CHKERRQ(ierr);
  }
  ierr = PetscFree(points);CHKERRQ(ierr);
  ierr = PetscHSetIDestroy(&stratumHash);CHKERRQ(ierr);
  ierr = PetscFree(leafStrata);CHKERRQ(ierr);
  ierr = PetscFree(strataIdx);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&leafSection);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 1);
  PetscValidHeaderSpecific(sf, PETSCSF_CLASSID, 2);
  ierr = PetscObjectGetComm((PetscObject)sf, &comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &size);CHKERRQ(ierr);
  /* Bcast name */
  if (!rank) {
    ierr = PetscObjectGetName((PetscObject) label, &lname);CHKERRQ(ierr);
    ierr = PetscStrlen(lname, &len);CHKERRQ(ierr);
  }
  nameSize = len;
  ierr = MPI_Bcast(&nameSize, 1, MPIU_INT, 0, comm);CHKERRQ(ierr);
  ierr = PetscMalloc1(nameSize+1, &name);CHKERRQ(ierr);
  if (!rank) {ierr = PetscArraycpy(name, lname, nameSize+1);CHKERRQ(ierr);}
  ierr = MPI_Bcast(name, nameSize+1, MPI_CHAR, 0, comm);CHKERRQ(ierr);
  ierr = DMLabelCreate(PETSC_COMM_SELF, name, labelNew);CHKERRQ(ierr);
  ierr = PetscFree(name);CHKERRQ(ierr);
  /* Gather rank/index pairs of leaves into local roots to build
     an inverse, multi-rooted SF. Note that this ignores local leaf
     indexing due to the use of the multiSF in PetscSFGather. */
  ierr = PetscSFGetGraph(sf, &nroots, &nleaves, &ilocal, NULL);CHKERRQ(ierr);
  ierr = PetscMalloc1(nroots, &leafPoints);CHKERRQ(ierr);
  for (p = 0; p < nroots; ++p) leafPoints[p].rank = leafPoints[p].index = -1;
  for (p = 0; p < nleaves; p++) {
    PetscInt ilp = ilocal ? ilocal[p] : p;

    leafPoints[ilp].index = ilp;
    leafPoints[ilp].rank  = rank;
  }
  ierr = PetscSFComputeDegreeBegin(sf, &rootDegree);CHKERRQ(ierr);
  ierr = PetscSFComputeDegreeEnd(sf, &rootDegree);CHKERRQ(ierr);
  for (p = 0, nmultiroots = 0; p < nroots; ++p) nmultiroots += rootDegree[p];
  ierr = PetscMalloc1(nmultiroots, &rootPoints);CHKERRQ(ierr);
  ierr = PetscSFGatherBegin(sf, MPIU_2INT, leafPoints, rootPoints);CHKERRQ(ierr);
  ierr = PetscSFGatherEnd(sf, MPIU_2INT, leafPoints, rootPoints);CHKERRQ(ierr);
  ierr = PetscSFCreate(comm,& sfLabel);CHKERRQ(ierr);
  ierr = PetscSFSetGraph(sfLabel, nroots, nmultiroots, NULL, PETSC_OWN_POINTER, rootPoints, PETSC_OWN_POINTER);CHKERRQ(ierr);
  /* Migrate label over inverted SF to pull stratum values at leaves into roots. */
  ierr = DMLabelDistribute_Internal(label, sfLabel, &rootSection, &rootStrata);CHKERRQ(ierr);
  /* Rebuild the point strata on the receiver */
  for (p = 0, idx = 0; p < nroots; p++) {
    for (d = 0; d < rootDegree[p]; d++) {
      ierr = PetscSectionGetDof(rootSection, idx+d, &dof);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(rootSection, idx+d, &offset);CHKERRQ(ierr);
      for (s = 0; s < dof; s++) {ierr = DMLabelSetValue(*labelNew, p, rootStrata[offset+s]);CHKERRQ(ierr);}
    }
    idx += rootDegree[p];
  }
  ierr = PetscFree(leafPoints);CHKERRQ(ierr);
  ierr = PetscFree(rootStrata);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&rootSection);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&sfLabel);CHKERRQ(ierr);
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
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 1);
  ierr = DMLabelGetNumValues(label, &nV);CHKERRQ(ierr);
  ierr = DMLabelGetValueIS(label, &vIS);CHKERRQ(ierr);
  ierr = ISGetIndices(vIS, &values);CHKERRQ(ierr);
  if (nV) {vS = values[0]; vE = values[0]+1;}
  for (v = 1; v < nV; ++v) {
    vS = PetscMin(vS, values[v]);
    vE = PetscMax(vE, values[v]+1);
  }
  ierr = PetscSectionCreate(PETSC_COMM_SELF, section);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(*section, vS, vE);CHKERRQ(ierr);
  for (v = 0; v < nV; ++v) {
    PetscInt n;

    ierr = DMLabelGetStratumSize(label, values[v], &n);CHKERRQ(ierr);
    ierr = PetscSectionSetDof(*section, values[v], n);CHKERRQ(ierr);
  }
  ierr = PetscSectionSetUp(*section);CHKERRQ(ierr);
  ierr = PetscSectionGetStorageSize(*section, &N);CHKERRQ(ierr);
  ierr = PetscMalloc1(N, &points);CHKERRQ(ierr);
  for (v = 0; v < nV; ++v) {
    IS              is;
    const PetscInt *spoints;
    PetscInt        dof, off, p;

    ierr = PetscSectionGetDof(*section, values[v], &dof);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(*section, values[v], &off);CHKERRQ(ierr);
    ierr = DMLabelGetStratumIS(label, values[v], &is);CHKERRQ(ierr);
    ierr = ISGetIndices(is, &spoints);CHKERRQ(ierr);
    for (p = 0; p < dof; ++p) points[off+p] = spoints[p];
    ierr = ISRestoreIndices(is, &spoints);CHKERRQ(ierr);
    ierr = ISDestroy(&is);CHKERRQ(ierr);
  }
  ierr = ISRestoreIndices(vIS, &values);CHKERRQ(ierr);
  ierr = ISDestroy(&vIS);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_SELF, N, points, PETSC_OWN_POINTER, is);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(s, PETSC_SECTION_CLASSID, 1);
  PetscValidHeaderSpecific(sf, PETSCSF_CLASSID, 2);
  PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 4);
  ierr = PetscSectionCreate(PetscObjectComm((PetscObject) s), gsection);CHKERRQ(ierr);
  ierr = PetscSectionGetChart(s, &pStart, &pEnd);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(*gsection, pStart, pEnd);CHKERRQ(ierr);
  ierr = PetscSFGetGraph(sf, &nroots, NULL, NULL, NULL);CHKERRQ(ierr);
  if (nroots >= 0) {
    if (nroots < pEnd-pStart) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "PetscSF nroots %d < %d section size", nroots, pEnd-pStart);
    ierr = PetscCalloc1(nroots, &neg);CHKERRQ(ierr);
    if (nroots > pEnd-pStart) {
      ierr = PetscCalloc1(nroots, &tmpOff);CHKERRQ(ierr);
    } else {
      tmpOff = &(*gsection)->atlasDof[-pStart];
    }
  }
  /* Mark ghost points with negative dof */
  for (p = pStart; p < pEnd; ++p) {
    PetscInt value;

    ierr = DMLabelGetValue(label, p, &value);CHKERRQ(ierr);
    if (value != labelValue) continue;
    ierr = PetscSectionGetDof(s, p, &dof);CHKERRQ(ierr);
    ierr = PetscSectionSetDof(*gsection, p, dof);CHKERRQ(ierr);
    ierr = PetscSectionGetConstraintDof(s, p, &cdof);CHKERRQ(ierr);
    if (!includeConstraints && cdof > 0) {ierr = PetscSectionSetConstraintDof(*gsection, p, cdof);CHKERRQ(ierr);}
    if (neg) neg[p] = -(dof+1);
  }
  ierr = PetscSectionSetUpBC(*gsection);CHKERRQ(ierr);
  if (nroots >= 0) {
    ierr = PetscSFBcastBegin(sf, MPIU_INT, neg, tmpOff);CHKERRQ(ierr);
    ierr = PetscSFBcastEnd(sf, MPIU_INT, neg, tmpOff);CHKERRQ(ierr);
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
  ierr       = MPI_Scan(&off, &globalOff, 1, MPIU_INT, MPI_SUM, PetscObjectComm((PetscObject) s));CHKERRQ(ierr);
  globalOff -= off;
  for (p = 0, off = 0; p < pEnd-pStart; ++p) {
    (*gsection)->atlasOff[p] += globalOff;
    if (neg) neg[p] = -((*gsection)->atlasOff[p]+1);
  }
  /* Put in negative offsets for ghost points */
  if (nroots >= 0) {
    ierr = PetscSFBcastBegin(sf, MPIU_INT, neg, tmpOff);CHKERRQ(ierr);
    ierr = PetscSFBcastEnd(sf, MPIU_INT, neg, tmpOff);CHKERRQ(ierr);
    if (nroots > pEnd-pStart) {
      for (p = pStart; p < pEnd; ++p) {if (tmpOff[p] < 0) (*gsection)->atlasOff[p-pStart] = tmpOff[p];}
    }
  }
  if (nroots >= 0 && nroots > pEnd-pStart) {ierr = PetscFree(tmpOff);CHKERRQ(ierr);}
  ierr = PetscFree(neg);CHKERRQ(ierr);
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
  PetscErrorCode        ierr;

  PetscFunctionBegin;
  for (i = 0; i <= sl->numStrata; i++) {
    if (sl->modes[i] == PETSC_OWN_POINTER || sl->modes[i] == PETSC_COPY_VALUES) {
      for (j = sl->minMaxOrients[i][0]; j < sl->minMaxOrients[i][1]; j++) {
        if (sl->perms[i]) {ierr = PetscFree(sl->perms[i][j]);CHKERRQ(ierr);}
        if (sl->rots[i]) {ierr = PetscFree(sl->rots[i][j]);CHKERRQ(ierr);}
      }
      if (sl->perms[i]) {
        const PetscInt **perms = &sl->perms[i][sl->minMaxOrients[i][0]];

        ierr = PetscFree(perms);CHKERRQ(ierr);
      }
      if (sl->rots[i]) {
        const PetscScalar **rots = &sl->rots[i][sl->minMaxOrients[i][0]];

        ierr = PetscFree(rots);CHKERRQ(ierr);
      }
    }
  }
  ierr = PetscFree5(sl->modes,sl->sizes,sl->perms,sl->rots,sl->minMaxOrients);CHKERRQ(ierr);
  ierr = DMLabelDestroy(&sl->label);CHKERRQ(ierr);
  sl->numStrata = 0;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSectionSymDestroy_Label(PetscSectionSym sym)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscSectionSymLabelReset(sym);CHKERRQ(ierr);
  ierr = PetscFree(sym->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSectionSymView_Label(PetscSectionSym sym, PetscViewer viewer)
{
  PetscSectionSym_Label *sl = (PetscSectionSym_Label *) sym->data;
  PetscBool             isAscii;
  DMLabel               label = sl->label;
  const char           *name;
  PetscErrorCode        ierr;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &isAscii);CHKERRQ(ierr);
  if (isAscii) {
    PetscInt          i, j, k;
    PetscViewerFormat format;

    ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
    if (label) {
      ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
      if (format == PETSC_VIEWER_ASCII_INFO_DETAIL) {
        ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
        ierr = DMLabelView(label, viewer);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
      } else {
        ierr = PetscObjectGetName((PetscObject) sl->label, &name);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"  Label '%s'\n",name);CHKERRQ(ierr);
      }
    } else {
      ierr = PetscViewerASCIIPrintf(viewer, "No label given\n");CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    for (i = 0; i <= sl->numStrata; i++) {
      PetscInt value = i < sl->numStrata ? label->stratumValues[i] : label->defaultValue;

      if (!(sl->perms[i] || sl->rots[i])) {
        ierr = PetscViewerASCIIPrintf(viewer, "Symmetry for stratum value %D (%D dofs per point): no symmetries\n", value, sl->sizes[i]);CHKERRQ(ierr);
      } else {
      ierr = PetscViewerASCIIPrintf(viewer, "Symmetry for stratum value %D (%D dofs per point):\n", value, sl->sizes[i]);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer, "Orientation range: [%D, %D)\n", sl->minMaxOrients[i][0], sl->minMaxOrients[i][1]);CHKERRQ(ierr);
        if (format == PETSC_VIEWER_ASCII_INFO_DETAIL) {
          ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
          for (j = sl->minMaxOrients[i][0]; j < sl->minMaxOrients[i][1]; j++) {
            if (!((sl->perms[i] && sl->perms[i][j]) || (sl->rots[i] && sl->rots[i][j]))) {
              ierr = PetscViewerASCIIPrintf(viewer, "Orientation %D: identity\n",j);CHKERRQ(ierr);
            } else {
              PetscInt tab;

              ierr = PetscViewerASCIIPrintf(viewer, "Orientation %D:\n",j);CHKERRQ(ierr);
              ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
              ierr = PetscViewerASCIIGetTab(viewer,&tab);CHKERRQ(ierr);
              if (sl->perms[i] && sl->perms[i][j]) {
                ierr = PetscViewerASCIIPrintf(viewer,"Permutation:");CHKERRQ(ierr);
                ierr = PetscViewerASCIISetTab(viewer,0);CHKERRQ(ierr);
                for (k = 0; k < sl->sizes[i]; k++) {ierr = PetscViewerASCIIPrintf(viewer," %D",sl->perms[i][j][k]);CHKERRQ(ierr);}
                ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
                ierr = PetscViewerASCIISetTab(viewer,tab);CHKERRQ(ierr);
              }
              if (sl->rots[i] && sl->rots[i][j]) {
                ierr = PetscViewerASCIIPrintf(viewer,"Rotations:  ");CHKERRQ(ierr);
                ierr = PetscViewerASCIISetTab(viewer,0);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
                for (k = 0; k < sl->sizes[i]; k++) {ierr = PetscViewerASCIIPrintf(viewer," %+f+i*%+f",PetscRealPart(sl->rots[i][j][k]),PetscImaginaryPart(sl->rots[i][j][k]));CHKERRQ(ierr);}
#else
                for (k = 0; k < sl->sizes[i]; k++) {ierr = PetscViewerASCIIPrintf(viewer," %+f",sl->rots[i][j][k]);CHKERRQ(ierr);}
#endif
                ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
                ierr = PetscViewerASCIISetTab(viewer,tab);CHKERRQ(ierr);
              }
              ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
            }
          }
          ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
        }
        ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
      }
    }
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
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
  PetscErrorCode        ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sym,PETSC_SECTION_SYM_CLASSID,1);
  sl = (PetscSectionSym_Label *) sym->data;
  if (sl->label && sl->label != label) {ierr = PetscSectionSymLabelReset(sym);CHKERRQ(ierr);}
  if (label) {
    sl->label = label;
    ierr = PetscObjectReference((PetscObject) label);CHKERRQ(ierr);
    ierr = DMLabelGetNumValues(label,&sl->numStrata);CHKERRQ(ierr);
    ierr = PetscMalloc5(sl->numStrata+1,&sl->modes,sl->numStrata+1,&sl->sizes,sl->numStrata+1,&sl->perms,sl->numStrata+1,&sl->rots,sl->numStrata+1,&sl->minMaxOrients);CHKERRQ(ierr);
    ierr = PetscMemzero((void *) sl->modes,(sl->numStrata+1)*sizeof(PetscCopyMode));CHKERRQ(ierr);
    ierr = PetscMemzero((void *) sl->sizes,(sl->numStrata+1)*sizeof(PetscInt));CHKERRQ(ierr);
    ierr = PetscMemzero((void *) sl->perms,(sl->numStrata+1)*sizeof(const PetscInt **));CHKERRQ(ierr);
    ierr = PetscMemzero((void *) sl->rots,(sl->numStrata+1)*sizeof(const PetscScalar **));CHKERRQ(ierr);
    ierr = PetscMemzero((void *) sl->minMaxOrients,(sl->numStrata+1)*sizeof(PetscInt[2]));CHKERRQ(ierr);
  }
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

.seealso: PetscSectionSymCreate(), PetscSectionSetSym(), PetscSectionGetPointSyms(), PetscSectionSymCreateLabel()
@*/
PetscErrorCode PetscSectionSymLabelSetStratum(PetscSectionSym sym, PetscInt stratum, PetscInt size, PetscInt minOrient, PetscInt maxOrient, PetscCopyMode mode, const PetscInt **perms, const PetscScalar **rots)
{
  PetscSectionSym_Label *sl;
  const char            *name;
  PetscInt               i, j, k;
  PetscErrorCode         ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sym,PETSC_SECTION_SYM_CLASSID,1);
  sl = (PetscSectionSym_Label *) sym->data;
  if (!sl->label) SETERRQ(PetscObjectComm((PetscObject)sym),PETSC_ERR_ARG_WRONGSTATE,"No label set yet");
  for (i = 0; i <= sl->numStrata; i++) {
    PetscInt value = (i < sl->numStrata) ? sl->label->stratumValues[i] : sl->label->defaultValue;

    if (stratum == value) break;
  }
  ierr = PetscObjectGetName((PetscObject) sl->label, &name);CHKERRQ(ierr);
  if (i > sl->numStrata) SETERRQ2(PetscObjectComm((PetscObject)sym),PETSC_ERR_ARG_OUTOFRANGE,"Stratum %D not found in label %s\n",stratum,name);
  sl->sizes[i] = size;
  sl->modes[i] = mode;
  sl->minMaxOrients[i][0] = minOrient;
  sl->minMaxOrients[i][1] = maxOrient;
  if (mode == PETSC_COPY_VALUES) {
    if (perms) {
      PetscInt    **ownPerms;

      ierr = PetscCalloc1(maxOrient - minOrient,&ownPerms);CHKERRQ(ierr);
      for (j = 0; j < maxOrient-minOrient; j++) {
        if (perms[j]) {
          ierr = PetscMalloc1(size,&ownPerms[j]);CHKERRQ(ierr);
          for (k = 0; k < size; k++) {ownPerms[j][k] = perms[j][k];}
        }
      }
      sl->perms[i] = (const PetscInt **) &ownPerms[-minOrient];
    }
    if (rots) {
      PetscScalar **ownRots;

      ierr = PetscCalloc1(maxOrient - minOrient,&ownRots);CHKERRQ(ierr);
      for (j = 0; j < maxOrient-minOrient; j++) {
        if (rots[j]) {
          ierr = PetscMalloc1(size,&ownRots[j]);CHKERRQ(ierr);
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
  PetscErrorCode        ierr;

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

        ierr = ISLocate(label->points[j],point,&k);CHKERRQ(ierr);
        if (k >= 0) break;
      } else {
        PetscBool has;

        ierr = PetscHSetIHas(label->ht[j], point, &has);CHKERRQ(ierr);
        if (has) break;
      }
    }
    if ((sl->minMaxOrients[j][1] > sl->minMaxOrients[j][0]) && (ornt < sl->minMaxOrients[j][0] || ornt >= sl->minMaxOrients[j][1])) SETERRQ5(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"point %D orientation %D not in range [%D, %D) for stratum %D",point,ornt,sl->minMaxOrients[j][0],sl->minMaxOrients[j][1],j < numStrata ? label->stratumValues[j] : label->defaultValue);
    if (perms) {perms[i] = sl->perms[j] ? sl->perms[j][ornt] : NULL;}
    if (rots) {rots[i]  = sl->rots[j] ? sl->rots[j][ornt] : NULL;}
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscSectionSymCreate_Label(PetscSectionSym sym)
{
  PetscSectionSym_Label *sl;
  PetscErrorCode        ierr;

  PetscFunctionBegin;
  ierr = PetscNewLog(sym,&sl);CHKERRQ(ierr);
  sym->ops->getpoints = PetscSectionSymGetPoints_Label;
  sym->ops->view      = PetscSectionSymView_Label;
  sym->ops->destroy   = PetscSectionSymDestroy_Label;
  sym->data           = (void *) sl;
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
  PetscErrorCode        ierr;

  PetscFunctionBegin;
  ierr = DMInitializePackage();CHKERRQ(ierr);
  ierr = PetscSectionSymCreate(comm,sym);CHKERRQ(ierr);
  ierr = PetscSectionSymSetType(*sym,PETSCSECTIONSYMLABEL);CHKERRQ(ierr);
  ierr = PetscSectionSymLabelSetLabel(*sym,label);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
