#include <petsc/private/dmpleximpl.h>   /*I      "petscdmplex.h"   I*/
#include <petscsf.h>

#undef __FUNCT__
#define __FUNCT__ "DMLabelCreate"
PetscErrorCode DMLabelCreate(const char name[], DMLabel *label)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNew(label);CHKERRQ(ierr);
  ierr = PetscStrallocpy(name, &(*label)->name);CHKERRQ(ierr);

  (*label)->refct          = 1;
  (*label)->numStrata      = 0;
  (*label)->stratumValues  = NULL;
  (*label)->arrayValid     = NULL;
  (*label)->stratumSizes   = NULL;
  (*label)->points         = NULL;
  (*label)->ht             = NULL;
  (*label)->pStart         = -1;
  (*label)->pEnd           = -1;
  (*label)->bt             = NULL;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMLabelMakeValid_Private"
static PetscErrorCode DMLabelMakeValid_Private(DMLabel label, PetscInt v)
{
  PetscInt       off;
  PetscErrorCode ierr;

  if (label->arrayValid[v]) return 0;
  if (v >= label->numStrata) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Trying to access invalid stratum %D in DMLabelMakeValid_Private\n", v);
  PetscFunctionBegin;
  PetscHashISize(label->ht[v], label->stratumSizes[v]);

  ierr = PetscMalloc1(label->stratumSizes[v], &label->points[v]);CHKERRQ(ierr);
  off = 0;
  ierr = PetscHashIGetKeys(label->ht[v], &off, &(label->points[v][0]));CHKERRQ(ierr);
  if (off != label->stratumSizes[v]) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Invalid number of contributed points %D from value %D should be %D", off, label->stratumValues[v], label->stratumSizes[v]);
  PetscHashIClear(label->ht[v]);
  ierr = PetscSortInt(label->stratumSizes[v], label->points[v]);CHKERRQ(ierr);
  if (label->bt) {
    PetscInt p;

    for (p = 0; p < label->stratumSizes[v]; ++p) {
      const PetscInt point = label->points[v][p];

      if ((point < label->pStart) || (point >= label->pEnd)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Label point %D is not in [%D, %D)", point, label->pStart, label->pEnd);
      ierr = PetscBTSet(label->bt, point - label->pStart);CHKERRQ(ierr);
    }
  }
  label->arrayValid[v] = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMLabelMakeAllValid_Private"
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

#undef __FUNCT__
#define __FUNCT__ "DMLabelMakeInvalid_Private"
static PetscErrorCode DMLabelMakeInvalid_Private(DMLabel label, PetscInt v)
{
  PETSC_UNUSED PetscHashIIter ret, iter;
  PetscInt                    p;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!label->arrayValid[v]) PetscFunctionReturn(0);
  for (p = 0; p < label->stratumSizes[v]; ++p) PetscHashIPut(label->ht[v], label->points[v][p], ret, iter);
  ierr = PetscFree(label->points[v]);CHKERRQ(ierr);
  label->arrayValid[v] = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMLabelAddStratum"
PetscErrorCode DMLabelAddStratum(DMLabel label, PetscInt value)
{
  PetscInt    v, *tmpV, *tmpS, **tmpP;
  PetscHashI *tmpH;
  PetscBool  *tmpB;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  ierr = PetscMalloc1((label->numStrata+1), &tmpV);CHKERRQ(ierr);
  ierr = PetscMalloc1((label->numStrata+1), &tmpS);CHKERRQ(ierr);
  ierr = PetscMalloc1((label->numStrata+1), &tmpH);CHKERRQ(ierr);
  ierr = PetscMalloc1((label->numStrata+1), &tmpP);CHKERRQ(ierr);
  ierr = PetscMalloc1((label->numStrata+1), &tmpB);CHKERRQ(ierr);
  for (v = 0; v < label->numStrata; ++v) {
    tmpV[v] = label->stratumValues[v];
    tmpS[v] = label->stratumSizes[v];
    tmpH[v] = label->ht[v];
    tmpP[v] = label->points[v];
    tmpB[v] = label->arrayValid[v];
  }
  tmpV[v] = value;
  tmpS[v] = 0;
  PetscHashICreate(tmpH[v]);
  tmpP[v] = NULL;
  tmpB[v] = PETSC_TRUE;
  ++label->numStrata;
  ierr = PetscFree(label->stratumValues);CHKERRQ(ierr);
  ierr = PetscFree(label->stratumSizes);CHKERRQ(ierr);
  ierr = PetscFree(label->ht);CHKERRQ(ierr);
  ierr = PetscFree(label->points);CHKERRQ(ierr);
  ierr = PetscFree(label->arrayValid);CHKERRQ(ierr);
  label->stratumValues = tmpV;
  label->stratumSizes  = tmpS;
  label->ht            = tmpH;
  label->points        = tmpP;
  label->arrayValid    = tmpB;

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMLabelGetName"
PetscErrorCode DMLabelGetName(DMLabel label, const char **name)
{
  PetscFunctionBegin;
  PetscValidPointer(name, 2);
  *name = label->name;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMLabelView_Ascii"
static PetscErrorCode DMLabelView_Ascii(DMLabel label, PetscViewer viewer)
{
  PetscInt       v;
  PetscMPIInt    rank;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)viewer), &rank);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPushSynchronized(viewer);CHKERRQ(ierr);
  if (label) {
    ierr = PetscViewerASCIIPrintf(viewer, "Label '%s':\n", label->name);CHKERRQ(ierr);
    if (label->bt) {ierr = PetscViewerASCIIPrintf(viewer, "  Index has been calculated in [%D, %D)\n", label->pStart, label->pEnd);CHKERRQ(ierr);}
    for (v = 0; v < label->numStrata; ++v) {
      const PetscInt value = label->stratumValues[v];
      PetscInt       p;

      for (p = 0; p < label->stratumSizes[v]; ++p) {
        ierr = PetscViewerASCIISynchronizedPrintf(viewer, "[%d]: %D (%D)\n", rank, label->points[v][p], value);CHKERRQ(ierr);
      }
    }
  }
  ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPopSynchronized(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMLabelView"
/*@C
  DMLabelView - View the label

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
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  if (label) {ierr = DMLabelMakeAllValid_Private(label);CHKERRQ(ierr);}
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = DMLabelView_Ascii(label, viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMLabelDestroy"
PetscErrorCode DMLabelDestroy(DMLabel *label)
{
  PetscInt       v;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!(*label)) PetscFunctionReturn(0);
  if (--(*label)->refct > 0) PetscFunctionReturn(0);
  ierr = PetscFree((*label)->name);CHKERRQ(ierr);
  ierr = PetscFree((*label)->stratumValues);CHKERRQ(ierr);
  ierr = PetscFree((*label)->stratumSizes);CHKERRQ(ierr);
  for (v = 0; v < (*label)->numStrata; ++v) {ierr = PetscFree((*label)->points[v]);CHKERRQ(ierr);}
  ierr = PetscFree((*label)->points);CHKERRQ(ierr);
  ierr = PetscFree((*label)->arrayValid);CHKERRQ(ierr);
  if ((*label)->ht) {
    for (v = 0; v < (*label)->numStrata; ++v) {PetscHashIDestroy((*label)->ht[v]);}
    ierr = PetscFree((*label)->ht);CHKERRQ(ierr);
  }
  ierr = PetscBTDestroy(&(*label)->bt);CHKERRQ(ierr);
  ierr = PetscFree(*label);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMLabelDuplicate"
PetscErrorCode DMLabelDuplicate(DMLabel label, DMLabel *labelnew)
{
  PetscInt       v, q;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMLabelMakeAllValid_Private(label);CHKERRQ(ierr);
  ierr = PetscNew(labelnew);CHKERRQ(ierr);
  ierr = PetscStrallocpy(label->name, &(*labelnew)->name);CHKERRQ(ierr);

  (*labelnew)->refct      = 1;
  (*labelnew)->numStrata  = label->numStrata;
  if (label->numStrata) {
    ierr = PetscMalloc1(label->numStrata, &(*labelnew)->stratumValues);CHKERRQ(ierr);
    ierr = PetscMalloc1(label->numStrata, &(*labelnew)->stratumSizes);CHKERRQ(ierr);
    ierr = PetscMalloc1(label->numStrata, &(*labelnew)->ht);CHKERRQ(ierr);
    ierr = PetscMalloc1(label->numStrata, &(*labelnew)->points);CHKERRQ(ierr);
    ierr = PetscMalloc1(label->numStrata, &(*labelnew)->arrayValid);CHKERRQ(ierr);
    /* Could eliminate unused space here */
    for (v = 0; v < label->numStrata; ++v) {
      ierr = PetscMalloc1(label->stratumSizes[v], &(*labelnew)->points[v]);CHKERRQ(ierr);
      PetscHashICreate((*labelnew)->ht[v]);
      (*labelnew)->arrayValid[v]     = PETSC_TRUE;
      (*labelnew)->stratumValues[v]  = label->stratumValues[v];
      (*labelnew)->stratumSizes[v]   = label->stratumSizes[v];
      for (q = 0; q < label->stratumSizes[v]; ++q) {
        (*labelnew)->points[v][q] = label->points[v][q];
      }
    }
  }
  (*labelnew)->pStart = -1;
  (*labelnew)->pEnd   = -1;
  (*labelnew)->bt     = NULL;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMLabelCreateIndex"
/* This can be hooked into SetValue(),  ClearValue(), etc. for updating */
PetscErrorCode DMLabelCreateIndex(DMLabel label, PetscInt pStart, PetscInt pEnd)
{
  PetscInt       v;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMLabelMakeAllValid_Private(label);CHKERRQ(ierr);
  if (label->bt) {ierr = PetscBTDestroy(&label->bt);CHKERRQ(ierr);}
  label->pStart = pStart;
  label->pEnd   = pEnd;
  ierr = PetscBTCreate(pEnd - pStart, &label->bt);CHKERRQ(ierr);
  ierr = PetscBTMemzero(pEnd - pStart, label->bt);CHKERRQ(ierr);
  for (v = 0; v < label->numStrata; ++v) {
    PetscInt i;

    for (i = 0; i < label->stratumSizes[v]; ++i) {
      const PetscInt point = label->points[v][i];

      if ((point < pStart) || (point >= pEnd)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Label point %D is not in [%D, %D)", point, pStart, pEnd);
      ierr = PetscBTSet(label->bt, point - pStart);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMLabelDestroyIndex"
PetscErrorCode DMLabelDestroyIndex(DMLabel label)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  label->pStart = -1;
  label->pEnd   = -1;
  if (label->bt) {ierr = PetscBTDestroy(&label->bt);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMLabelHasValue"
/*@
  DMLabelHasValue - Determine whether a label assigns the value to any point

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
  PetscValidPointer(contains, 3);
  for (v = 0; v < label->numStrata; ++v) {
    if (value == label->stratumValues[v]) break;
  }
  *contains = (v < label->numStrata ? PETSC_TRUE : PETSC_FALSE);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMLabelHasPoint"
/*@
  DMLabelHasPoint - Determine whether a label assigns a value to a point

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
  PetscValidPointer(contains, 3);
  ierr = DMLabelMakeAllValid_Private(label);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
  if (!label->bt) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Must call DMLabelCreateIndex() before DMLabelHasPoint()");
  if ((point < label->pStart) || (point >= label->pEnd)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Label point %D is not in [%D, %D)", point, label->pStart, label->pEnd);
#endif
  *contains = PetscBTLookup(label->bt, point - label->pStart) ? PETSC_TRUE : PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMLabelStratumHasPoint"
/*@
  DMLabelStratumHasPoint - Return true if the stratum contains a point

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

  PetscFunctionBegin;
  PetscValidPointer(contains, 4);
  *contains = PETSC_FALSE;
  for (v = 0; v < label->numStrata; ++v) {
    if (label->stratumValues[v] == value) {
      if (label->arrayValid[v]) {
        PetscInt i;

        ierr = PetscFindInt(point, label->stratumSizes[v], &label->points[v][0], &i);CHKERRQ(ierr);
        if (i >= 0) {
          *contains = PETSC_TRUE;
          break;
        }
      } else {
        PetscBool has;

        PetscHashIHasKey(label->ht[v], point, has);
        if (has) {
          *contains = PETSC_TRUE;
          break;
        }
      }
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMLabelGetValue"
/*@
  DMLabelGetValue - Return the value a label assigns to a point, or -1

  Input Parameters:
+ label - the DMLabel
- point - the point

  Output Parameter:
. value - The point value, or -1

  Level: intermediate

.seealso: DMLabelCreate(), DMLabelSetValue(), DMLabelClearValue()
@*/
PetscErrorCode DMLabelGetValue(DMLabel label, PetscInt point, PetscInt *value)
{
  PetscInt       v;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(value, 3);
  *value = -1;
  for (v = 0; v < label->numStrata; ++v) {
    if (label->arrayValid[v]) {
      PetscInt i;

      ierr = PetscFindInt(point, label->stratumSizes[v], &label->points[v][0], &i);CHKERRQ(ierr);
      if (i >= 0) {
        *value = label->stratumValues[v];
        break;
      }
    } else {
      PetscBool has;

      PetscHashIHasKey(label->ht[v], point, has);
      if (has) {
        *value = label->stratumValues[v];
        break;
      }
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMLabelSetValue"
/*@
  DMLabelSetValue - Set the value a label assigns to a point

  Input Parameters:
+ label - the DMLabel
. point - the point
- value - The point value

  Level: intermediate

.seealso: DMLabelCreate(), DMLabelGetValue(), DMLabelClearValue()
@*/
PetscErrorCode DMLabelSetValue(DMLabel label, PetscInt point, PetscInt value)
{
  PETSC_UNUSED PetscHashIIter iter, ret;
  PetscInt                    v;
  PetscErrorCode              ierr;

  PetscFunctionBegin;
  /* Find, or add, label value */
  for (v = 0; v < label->numStrata; ++v) {
    if (label->stratumValues[v] == value) break;
  }
  /* Create new table */
  if (v >= label->numStrata) {ierr = DMLabelAddStratum(label, value);CHKERRQ(ierr);}
  ierr = DMLabelMakeInvalid_Private(label, v);CHKERRQ(ierr);
  /* Set key */
  PetscHashIPut(label->ht[v], point, ret, iter);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMLabelClearValue"
/*@
  DMLabelClearValue - Clear the value a label assigns to a point

  Input Parameters:
+ label - the DMLabel
. point - the point
- value - The point value

  Level: intermediate

.seealso: DMLabelCreate(), DMLabelGetValue(), DMLabelSetValue()
@*/
PetscErrorCode DMLabelClearValue(DMLabel label, PetscInt point, PetscInt value)
{
  PetscInt       v, p;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* Find label value */
  for (v = 0; v < label->numStrata; ++v) {
    if (label->stratumValues[v] == value) break;
  }
  if (v >= label->numStrata) PetscFunctionReturn(0);
  if (label->arrayValid[v]) {
    /* Check whether point exists */
    ierr = PetscFindInt(point, label->stratumSizes[v], &label->points[v][0], &p);CHKERRQ(ierr);
    if (p >= 0) {
      ierr = PetscMemmove(&label->points[v][p], &label->points[v][p+1], (label->stratumSizes[v]-p-1) * sizeof(PetscInt));CHKERRQ(ierr);
      --label->stratumSizes[v];
      if (label->bt) {
        if ((point < label->pStart) || (point >= label->pEnd)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Label point %D is not in [%D, %D)", point, label->pStart, label->pEnd);
        ierr = PetscBTClear(label->bt, point - label->pStart);CHKERRQ(ierr);
      }
    }
  } else {
    ierr = PetscHashIDelKey(label->ht[v], point);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMLabelInsertIS"
/*@
  DMLabelInsertIS - Set all points in the IS to a value

  Input Parameters:
+ label - the DMLabel
. is    - the point IS
- value - The point value

  Level: intermediate

.seealso: DMLabelCreate(), DMLabelGetValue(), DMLabelSetValue(), DMLabelClearValue()
@*/
PetscErrorCode DMLabelInsertIS(DMLabel label, IS is, PetscInt value)
{
  const PetscInt *points;
  PetscInt        n, p;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(is, IS_CLASSID, 2);
  ierr = ISGetLocalSize(is, &n);CHKERRQ(ierr);
  ierr = ISGetIndices(is, &points);CHKERRQ(ierr);
  for (p = 0; p < n; ++p) {ierr = DMLabelSetValue(label, points[p], value);CHKERRQ(ierr);}
  ierr = ISRestoreIndices(is, &points);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMLabelGetNumValues"
PetscErrorCode DMLabelGetNumValues(DMLabel label, PetscInt *numValues)
{
  PetscFunctionBegin;
  PetscValidPointer(numValues, 2);
  *numValues = label->numStrata;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMLabelGetValueIS"
PetscErrorCode DMLabelGetValueIS(DMLabel label, IS *values)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(values, 2);
  ierr = ISCreateGeneral(PETSC_COMM_SELF, label->numStrata, label->stratumValues, PETSC_USE_POINTER, values);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMLabelGetStratumSize"
PetscErrorCode DMLabelGetStratumSize(DMLabel label, PetscInt value, PetscInt *size)
{
  PetscInt       v;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(size, 3);
  *size = 0;
  for (v = 0; v < label->numStrata; ++v) {
    if (label->stratumValues[v] == value) {
      ierr = DMLabelMakeValid_Private(label, v);CHKERRQ(ierr);
      *size = label->stratumSizes[v];
      break;
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMLabelGetStratumBounds"
PetscErrorCode DMLabelGetStratumBounds(DMLabel label, PetscInt value, PetscInt *start, PetscInt *end)
{
  PetscInt       v;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (start) {PetscValidPointer(start, 3); *start = 0;}
  if (end)   {PetscValidPointer(end,   4); *end   = 0;}
  for (v = 0; v < label->numStrata; ++v) {
    if (label->stratumValues[v] != value) continue;
    ierr = DMLabelMakeValid_Private(label, v);CHKERRQ(ierr);
    if (label->stratumSizes[v]  <= 0)     break;
    if (start) *start = label->points[v][0];
    if (end)   *end   = label->points[v][label->stratumSizes[v]-1]+1;
    break;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMLabelGetStratumIS"
PetscErrorCode DMLabelGetStratumIS(DMLabel label, PetscInt value, IS *points)
{
  PetscInt       v;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(points, 3);
  *points = NULL;
  for (v = 0; v < label->numStrata; ++v) {
    if (label->stratumValues[v] == value) {
      ierr = DMLabelMakeValid_Private(label, v);CHKERRQ(ierr);
      if (label->arrayValid[v]) {
        ierr = ISCreateGeneral(PETSC_COMM_SELF, label->stratumSizes[v], &label->points[v][0], PETSC_COPY_VALUES, points);CHKERRQ(ierr);
        ierr = PetscObjectSetName((PetscObject) *points, "indices");CHKERRQ(ierr);
      } else {
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Need to implement this to speedup Stratify");
      }
      break;
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMLabelClearStratum"
PetscErrorCode DMLabelClearStratum(DMLabel label, PetscInt value)
{
  PetscInt       v;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  for (v = 0; v < label->numStrata; ++v) {
    if (label->stratumValues[v] == value) break;
  }
  if (v >= label->numStrata) PetscFunctionReturn(0);
  if (label->bt) {
    PetscInt i;

    for (i = 0; i < label->stratumSizes[v]; ++i) {
      const PetscInt point = label->points[v][i];

      if ((point < label->pStart) || (point >= label->pEnd)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Label point %D is not in [%D, %D)", point, label->pStart, label->pEnd);
      ierr = PetscBTClear(label->bt, point - label->pStart);CHKERRQ(ierr);
    }
  }
  if (label->arrayValid[v]) {
    label->stratumSizes[v] = 0;
  } else {
    PetscHashIClear(label->ht[v]);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMLabelFilter"
PetscErrorCode DMLabelFilter(DMLabel label, PetscInt start, PetscInt end)
{
  PetscInt       v;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMLabelMakeAllValid_Private(label);CHKERRQ(ierr);
  label->pStart = start;
  label->pEnd   = end;
  if (label->bt) {ierr = PetscBTDestroy(&label->bt);CHKERRQ(ierr);}
  /* Could squish offsets, but would only make sense if I reallocate the storage */
  for (v = 0; v < label->numStrata; ++v) {
    PetscInt off, q;

    for (off = 0, q = 0; q < label->stratumSizes[v]; ++q) {
      const PetscInt point = label->points[v][q];

      if ((point < start) || (point >= end)) continue;
      label->points[v][off++] = point;
    }
    label->stratumSizes[v] = off;
  }
  ierr = DMLabelCreateIndex(label, start, end);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMLabelPermute"
PetscErrorCode DMLabelPermute(DMLabel label, IS permutation, DMLabel *labelNew)
{
  const PetscInt *perm;
  PetscInt        numValues, numPoints, v, q;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = DMLabelMakeAllValid_Private(label);CHKERRQ(ierr);
  ierr = DMLabelDuplicate(label, labelNew);CHKERRQ(ierr);
  ierr = DMLabelGetNumValues(*labelNew, &numValues);CHKERRQ(ierr);
  ierr = ISGetLocalSize(permutation, &numPoints);CHKERRQ(ierr);
  ierr = ISGetIndices(permutation, &perm);CHKERRQ(ierr);
  for (v = 0; v < numValues; ++v) {
    const PetscInt size   = (*labelNew)->stratumSizes[v];

    for (q = 0; q < size; ++q) {
      const PetscInt point = (*labelNew)->points[v][q];

      if ((point < 0) || (point >= numPoints)) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Label point %D is not in [0, %D) for the remapping", point, numPoints);
      (*labelNew)->points[v][q] = perm[point];
    }
    ierr = PetscSortInt(size, &(*labelNew)->points[v][0]);CHKERRQ(ierr);
  }
  ierr = ISRestoreIndices(permutation, &perm);CHKERRQ(ierr);
  if (label->bt) {
    ierr = PetscBTDestroy(&label->bt);CHKERRQ(ierr);
    ierr = DMLabelCreateIndex(label, label->pStart, label->pEnd);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMLabelDistribute"
PetscErrorCode DMLabelDistribute(DMLabel label, PetscSF sf, DMLabel *labelNew)
{
  MPI_Comm       comm;
  PetscSection   rootSection, leafSection;
  PetscSF        labelSF;
  PetscInt       p, pStart, pEnd, l, lStart, lEnd, s, nroots, nleaves, size, dof, offset, stratum;
  PetscInt      *remoteOffsets, *rootStrata, *rootIdx, *leafStrata, *strataIdx;
  char          *name;
  PetscInt       nameSize;
  size_t         len = 0;
  PetscMPIInt    rank, numProcs;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (label) {ierr = DMLabelMakeAllValid_Private(label);CHKERRQ(ierr);}
  ierr = PetscObjectGetComm((PetscObject)sf, &comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &numProcs);CHKERRQ(ierr);
  /* Bcast name */
  if (!rank) {ierr = PetscStrlen(label->name, &len);CHKERRQ(ierr);}
  nameSize = len;
  ierr = MPI_Bcast(&nameSize, 1, MPIU_INT, 0, comm);CHKERRQ(ierr);
  ierr = PetscMalloc1(nameSize+1, &name);CHKERRQ(ierr);
  if (!rank) {ierr = PetscMemcpy(name, label->name, nameSize+1);CHKERRQ(ierr);}
  ierr = MPI_Bcast(name, nameSize+1, MPI_CHAR, 0, comm);CHKERRQ(ierr);
  ierr = DMLabelCreate(name, labelNew);CHKERRQ(ierr);
  ierr = PetscFree(name);CHKERRQ(ierr);
  /* Bcast numStrata */
  if (!rank) (*labelNew)->numStrata = label->numStrata;
  ierr = MPI_Bcast(&(*labelNew)->numStrata, 1, MPIU_INT, 0, comm);CHKERRQ(ierr);
  /* Bcast stratumValues */
  ierr = PetscMalloc1((*labelNew)->numStrata, &(*labelNew)->stratumValues);CHKERRQ(ierr);
  if (!rank) {ierr = PetscMemcpy((*labelNew)->stratumValues, label->stratumValues, label->numStrata * sizeof(PetscInt));CHKERRQ(ierr);}
  ierr = MPI_Bcast((*labelNew)->stratumValues, (*labelNew)->numStrata, MPIU_INT, 0, comm);CHKERRQ(ierr);
  ierr = PetscMalloc1((*labelNew)->numStrata, &(*labelNew)->arrayValid);CHKERRQ(ierr);
  for (s = 0; s < (*labelNew)->numStrata; ++s) (*labelNew)->arrayValid[s] = PETSC_TRUE;

  /* Build a section detailing strata-per-point, distribute and build SF
     from that and then send our points. */
  ierr = PetscSFGetGraph(sf, &nroots, &nleaves, NULL, NULL);CHKERRQ(ierr);
  ierr = PetscSectionCreate(comm, &rootSection);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(rootSection, 0, nroots);CHKERRQ(ierr);
  if (label) {
    for (s = 0; s < label->numStrata; ++s) {
      lStart = 0;
      lEnd = label->stratumSizes[s];
      for (l=lStart; l<lEnd; l++) {
        ierr = PetscSectionGetDof(rootSection, label->points[s][l], &dof);CHKERRQ(ierr);
        ierr = PetscSectionSetDof(rootSection, label->points[s][l], dof+1);CHKERRQ(ierr);
      }
    }
  }
  ierr = PetscSectionSetUp(rootSection);CHKERRQ(ierr);

  /* Create a point-wise array of point strata */
  ierr = PetscSectionGetStorageSize(rootSection, &size);CHKERRQ(ierr);
  ierr = PetscMalloc1(size, &rootStrata);CHKERRQ(ierr);
  ierr = PetscCalloc1(nroots, &rootIdx);CHKERRQ(ierr);
  if (label) {
    for (s = 0; s < label->numStrata; ++s) {
      lStart = 0;
      lEnd = label->stratumSizes[s];
      for (l=lStart; l<lEnd; l++) {
        p = label->points[s][l];
        ierr = PetscSectionGetOffset(rootSection, p, &offset);CHKERRQ(ierr);
        rootStrata[offset+rootIdx[p]++] = s;
      }
    }
  }

  /* Build SF that maps label points to remote processes */
  ierr = PetscSectionCreate(comm, &leafSection);CHKERRQ(ierr);
  ierr = PetscSFDistributeSection(sf, rootSection, &remoteOffsets, leafSection);CHKERRQ(ierr);
  ierr = PetscSFCreateSectionSF(sf, rootSection, remoteOffsets, leafSection, &labelSF);CHKERRQ(ierr);

  /* Send the strata for each point over the derived SF */
  ierr = PetscSectionGetStorageSize(leafSection, &size);CHKERRQ(ierr);
  ierr = PetscMalloc1(size, &leafStrata);CHKERRQ(ierr);
  ierr = PetscSFBcastBegin(labelSF, MPIU_INT, rootStrata, leafStrata);CHKERRQ(ierr);
  ierr = PetscSFBcastEnd(labelSF, MPIU_INT, rootStrata, leafStrata);CHKERRQ(ierr);

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
  for (s = 0; s < (*labelNew)->numStrata; ++s) {
    PetscHashICreate((*labelNew)->ht[s]);
    ierr = PetscMalloc1((*labelNew)->stratumSizes[s], &(*labelNew)->points[s]);CHKERRQ(ierr);
  }

  /* Insert points into new strata */
  ierr = PetscCalloc1((*labelNew)->numStrata, &strataIdx);CHKERRQ(ierr);
  ierr = PetscSectionGetChart(leafSection, &pStart, &pEnd);CHKERRQ(ierr);
  for (p=pStart; p<pEnd; p++) {
    ierr = PetscSectionGetDof(leafSection, p, &dof);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(leafSection, p, &offset);CHKERRQ(ierr);
    for (s=0; s<dof; s++) {
      stratum = leafStrata[offset+s];
      (*labelNew)->points[stratum][strataIdx[stratum]++] = p;
    }
  }
  ierr = PetscFree(rootStrata);CHKERRQ(ierr);
  ierr = PetscFree(leafStrata);CHKERRQ(ierr);
  ierr = PetscFree(rootIdx);CHKERRQ(ierr);
  ierr = PetscFree(strataIdx);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&rootSection);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&leafSection);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&labelSF);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMLabelConvertToSection"
PetscErrorCode DMLabelConvertToSection(DMLabel label, PetscSection *section, IS *is)
{
  IS              vIS;
  const PetscInt *values;
  PetscInt       *points;
  PetscInt        nV, vS = 0, vE = 0, v, N;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
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


#undef __FUNCT__
#define __FUNCT__ "DMPlexCreateLabel"
/*@C
  DMPlexCreateLabel - Create a label of the given name if it does not already exist

  Not Collective

  Input Parameters:
+ dm   - The DMPlex object
- name - The label name

  Level: intermediate

.keywords: mesh
.seealso: DMLabelCreate(), DMPlexHasLabel(), DMPlexGetLabelValue(), DMPlexSetLabelValue(), DMPlexGetStratumIS()
@*/
PetscErrorCode DMPlexCreateLabel(DM dm, const char name[])
{
  DM_Plex        *mesh = (DM_Plex*) dm->data;
  PlexLabel      next  = mesh->labels;
  PetscBool      flg   = PETSC_FALSE;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidCharPointer(name, 2);
  while (next) {
    ierr = PetscStrcmp(name, next->label->name, &flg);CHKERRQ(ierr);
    if (flg) break;
    next = next->next;
  }
  if (!flg) {
    PlexLabel tmpLabel;

    ierr = PetscCalloc1(1, &tmpLabel);CHKERRQ(ierr);
    ierr = DMLabelCreate(name, &tmpLabel->label);CHKERRQ(ierr);
    tmpLabel->output = PETSC_TRUE;
    tmpLabel->next   = mesh->labels;
    mesh->labels     = tmpLabel;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexGetLabelValue"
/*@C
  DMPlexGetLabelValue - Get the value in a Sieve Label for the given point, with 0 as the default

  Not Collective

  Input Parameters:
+ dm   - The DMPlex object
. name - The label name
- point - The mesh point

  Output Parameter:
. value - The label value for this point, or -1 if the point is not in the label

  Level: beginner

.keywords: mesh
.seealso: DMLabelGetValue(), DMPlexSetLabelValue(), DMPlexGetStratumIS()
@*/
PetscErrorCode DMPlexGetLabelValue(DM dm, const char name[], PetscInt point, PetscInt *value)
{
  DMLabel        label;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidCharPointer(name, 2);
  ierr = DMPlexGetLabel(dm, name, &label);CHKERRQ(ierr);
  if (!label) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "No label named %s was found", name);CHKERRQ(ierr);
  ierr = DMLabelGetValue(label, point, value);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexSetLabelValue"
/*@C
  DMPlexSetLabelValue - Add a point to a Sieve Label with given value

  Not Collective

  Input Parameters:
+ dm   - The DMPlex object
. name - The label name
. point - The mesh point
- value - The label value for this point

  Output Parameter:

  Level: beginner

.keywords: mesh
.seealso: DMLabelSetValue(), DMPlexGetStratumIS(), DMPlexClearLabelValue()
@*/
PetscErrorCode DMPlexSetLabelValue(DM dm, const char name[], PetscInt point, PetscInt value)
{
  DMLabel        label;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidCharPointer(name, 2);
  ierr = DMPlexGetLabel(dm, name, &label);CHKERRQ(ierr);
  if (!label) {
    ierr = DMPlexCreateLabel(dm, name);CHKERRQ(ierr);
    ierr = DMPlexGetLabel(dm, name, &label);CHKERRQ(ierr);
  }
  ierr = DMLabelSetValue(label, point, value);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexClearLabelValue"
/*@C
  DMPlexClearLabelValue - Remove a point from a Sieve Label with given value

  Not Collective

  Input Parameters:
+ dm   - The DMPlex object
. name - The label name
. point - The mesh point
- value - The label value for this point

  Output Parameter:

  Level: beginner

.keywords: mesh
.seealso: DMLabelClearValue(), DMPlexSetLabelValue(), DMPlexGetStratumIS()
@*/
PetscErrorCode DMPlexClearLabelValue(DM dm, const char name[], PetscInt point, PetscInt value)
{
  DMLabel        label;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidCharPointer(name, 2);
  ierr = DMPlexGetLabel(dm, name, &label);CHKERRQ(ierr);
  if (!label) PetscFunctionReturn(0);
  ierr = DMLabelClearValue(label, point, value);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexGetLabelSize"
/*@C
  DMPlexGetLabelSize - Get the number of different integer ids in a Label

  Not Collective

  Input Parameters:
+ dm   - The DMPlex object
- name - The label name

  Output Parameter:
. size - The number of different integer ids, or 0 if the label does not exist

  Level: beginner

.keywords: mesh
.seealso: DMLabeGetNumValues(), DMPlexSetLabelValue()
@*/
PetscErrorCode DMPlexGetLabelSize(DM dm, const char name[], PetscInt *size)
{
  DMLabel        label;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidCharPointer(name, 2);
  PetscValidPointer(size, 3);
  ierr  = DMPlexGetLabel(dm, name, &label);CHKERRQ(ierr);
  *size = 0;
  if (!label) PetscFunctionReturn(0);
  ierr = DMLabelGetNumValues(label, size);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexGetLabelIdIS"
/*@C
  DMPlexGetLabelIdIS - Get the integer ids in a label

  Not Collective

  Input Parameters:
+ mesh - The DMPlex object
- name - The label name

  Output Parameter:
. ids - The integer ids, or NULL if the label does not exist

  Level: beginner

.keywords: mesh
.seealso: DMLabelGetValueIS(), DMPlexGetLabelSize()
@*/
PetscErrorCode DMPlexGetLabelIdIS(DM dm, const char name[], IS *ids)
{
  DMLabel        label;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidCharPointer(name, 2);
  PetscValidPointer(ids, 3);
  ierr = DMPlexGetLabel(dm, name, &label);CHKERRQ(ierr);
  *ids = NULL;
  if (!label) PetscFunctionReturn(0);
  ierr = DMLabelGetValueIS(label, ids);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexGetStratumSize"
/*@C
  DMPlexGetStratumSize - Get the number of points in a label stratum

  Not Collective

  Input Parameters:
+ dm - The DMPlex object
. name - The label name
- value - The stratum value

  Output Parameter:
. size - The stratum size

  Level: beginner

.keywords: mesh
.seealso: DMLabelGetStratumSize(), DMPlexGetLabelSize(), DMPlexGetLabelIds()
@*/
PetscErrorCode DMPlexGetStratumSize(DM dm, const char name[], PetscInt value, PetscInt *size)
{
  DMLabel        label;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidCharPointer(name, 2);
  PetscValidPointer(size, 4);
  ierr  = DMPlexGetLabel(dm, name, &label);CHKERRQ(ierr);
  *size = 0;
  if (!label) PetscFunctionReturn(0);
  ierr = DMLabelGetStratumSize(label, value, size);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexGetStratumIS"
/*@C
  DMPlexGetStratumIS - Get the points in a label stratum

  Not Collective

  Input Parameters:
+ dm - The DMPlex object
. name - The label name
- value - The stratum value

  Output Parameter:
. points - The stratum points, or NULL if the label does not exist or does not have that value

  Level: beginner

.keywords: mesh
.seealso: DMLabelGetStratumIS(), DMPlexGetStratumSize()
@*/
PetscErrorCode DMPlexGetStratumIS(DM dm, const char name[], PetscInt value, IS *points)
{
  DMLabel        label;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidCharPointer(name, 2);
  PetscValidPointer(points, 4);
  ierr    = DMPlexGetLabel(dm, name, &label);CHKERRQ(ierr);
  *points = NULL;
  if (!label) PetscFunctionReturn(0);
  ierr = DMLabelGetStratumIS(label, value, points);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexClearLabelStratum"
/*@C
  DMPlexClearLabelStratum - Remove all points from a stratum from a Sieve Label

  Not Collective

  Input Parameters:
+ dm   - The DMPlex object
. name - The label name
- value - The label value for this point

  Output Parameter:

  Level: beginner

.keywords: mesh
.seealso: DMLabelClearStratum(), DMPlexSetLabelValue(), DMPlexGetStratumIS(), DMPlexClearLabelValue()
@*/
PetscErrorCode DMPlexClearLabelStratum(DM dm, const char name[], PetscInt value)
{
  DMLabel        label;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidCharPointer(name, 2);
  ierr = DMPlexGetLabel(dm, name, &label);CHKERRQ(ierr);
  if (!label) PetscFunctionReturn(0);
  ierr = DMLabelClearStratum(label, value);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexGetNumLabels"
/*@
  DMPlexGetNumLabels - Return the number of labels defined by the mesh

  Not Collective

  Input Parameter:
. dm   - The DMPlex object

  Output Parameter:
. numLabels - the number of Labels

  Level: intermediate

.keywords: mesh
.seealso: DMPlexGetLabelValue(), DMPlexSetLabelValue(), DMPlexGetStratumIS()
@*/
PetscErrorCode DMPlexGetNumLabels(DM dm, PetscInt *numLabels)
{
  DM_Plex  *mesh = (DM_Plex*) dm->data;
  PlexLabel next = mesh->labels;
  PetscInt  n    = 0;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(numLabels, 2);
  while (next) {++n; next = next->next;}
  *numLabels = n;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexGetLabelName"
/*@C
  DMPlexGetLabelName - Return the name of nth label

  Not Collective

  Input Parameters:
+ dm - The DMPlex object
- n  - the label number

  Output Parameter:
. name - the label name

  Level: intermediate

.keywords: mesh
.seealso: DMPlexGetLabelValue(), DMPlexSetLabelValue(), DMPlexGetStratumIS()
@*/
PetscErrorCode DMPlexGetLabelName(DM dm, PetscInt n, const char **name)
{
  DM_Plex  *mesh = (DM_Plex*) dm->data;
  PlexLabel next = mesh->labels;
  PetscInt  l    = 0;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(name, 3);
  while (next) {
    if (l == n) {
      *name = next->label->name;
      PetscFunctionReturn(0);
    }
    ++l;
    next = next->next;
  }
  SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Label %D does not exist in this DM", n);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexHasLabel"
/*@C
  DMPlexHasLabel - Determine whether the mesh has a label of a given name

  Not Collective

  Input Parameters:
+ dm   - The DMPlex object
- name - The label name

  Output Parameter:
. hasLabel - PETSC_TRUE if the label is present

  Level: intermediate

.keywords: mesh
.seealso: DMPlexCreateLabel(), DMPlexGetLabelValue(), DMPlexSetLabelValue(), DMPlexGetStratumIS()
@*/
PetscErrorCode DMPlexHasLabel(DM dm, const char name[], PetscBool *hasLabel)
{
  DM_Plex       *mesh = (DM_Plex*) dm->data;
  PlexLabel      next = mesh->labels;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidCharPointer(name, 2);
  PetscValidPointer(hasLabel, 3);
  *hasLabel = PETSC_FALSE;
  while (next) {
    ierr = PetscStrcmp(name, next->label->name, hasLabel);CHKERRQ(ierr);
    if (*hasLabel) break;
    next = next->next;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexGetLabel"
/*@C
  DMPlexGetLabel - Return the label of a given name, or NULL

  Not Collective

  Input Parameters:
+ dm   - The DMPlex object
- name - The label name

  Output Parameter:
. label - The DMLabel, or NULL if the label is absent

  Level: intermediate

.keywords: mesh
.seealso: DMPlexCreateLabel(), DMPlexHasLabel(), DMPlexGetLabelValue(), DMPlexSetLabelValue(), DMPlexGetStratumIS()
@*/
PetscErrorCode DMPlexGetLabel(DM dm, const char name[], DMLabel *label)
{
  DM_Plex       *mesh = (DM_Plex*) dm->data;
  PlexLabel      next = mesh->labels;
  PetscBool      hasLabel;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidCharPointer(name, 2);
  PetscValidPointer(label, 3);
  *label = NULL;
  while (next) {
    ierr = PetscStrcmp(name, next->label->name, &hasLabel);CHKERRQ(ierr);
    if (hasLabel) {
      *label = next->label;
      break;
    }
    next = next->next;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexGetLabelByNum"
/*@C
  DMPlexGetLabelByNum - Return the nth label

  Not Collective

  Input Parameters:
+ dm - The DMPlex object
- n  - the label number

  Output Parameter:
. label - the label

  Level: intermediate

.keywords: mesh
.seealso: DMPlexGetLabelValue(), DMPlexSetLabelValue(), DMPlexGetStratumIS()
@*/
PetscErrorCode DMPlexGetLabelByNum(DM dm, PetscInt n, DMLabel *label)
{
  DM_Plex  *mesh = (DM_Plex*) dm->data;
  PlexLabel next = mesh->labels;
  PetscInt  l    = 0;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(label, 3);
  while (next) {
    if (l == n) {
      *label = next->label;
      PetscFunctionReturn(0);
    }
    ++l;
    next = next->next;
  }
  SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Label %D does not exist in this DM", n);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexAddLabel"
/*@C
  DMPlexAddLabel - Add the label to this mesh

  Not Collective

  Input Parameters:
+ dm   - The DMPlex object
- label - The DMLabel

  Level: developer

.keywords: mesh
.seealso: DMPlexCreateLabel(), DMPlexHasLabel(), DMPlexGetLabelValue(), DMPlexSetLabelValue(), DMPlexGetStratumIS()
@*/
PetscErrorCode DMPlexAddLabel(DM dm, DMLabel label)
{
  DM_Plex       *mesh = (DM_Plex*) dm->data;
  PlexLabel      tmpLabel;
  PetscBool      hasLabel;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = DMPlexHasLabel(dm, label->name, &hasLabel);CHKERRQ(ierr);
  if (hasLabel) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Label %s already exists in this DM", label->name);
  ierr = PetscCalloc1(1, &tmpLabel);CHKERRQ(ierr);
  tmpLabel->label  = label;
  tmpLabel->output = PETSC_TRUE;
  tmpLabel->next   = mesh->labels;
  mesh->labels     = tmpLabel;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexRemoveLabel"
/*@C
  DMPlexRemoveLabel - Remove the label from this mesh

  Not Collective

  Input Parameters:
+ dm   - The DMPlex object
- name - The label name

  Output Parameter:
. label - The DMLabel, or NULL if the label is absent

  Level: developer

.keywords: mesh
.seealso: DMPlexCreateLabel(), DMPlexHasLabel(), DMPlexGetLabelValue(), DMPlexSetLabelValue(), DMPlexGetStratumIS()
@*/
PetscErrorCode DMPlexRemoveLabel(DM dm, const char name[], DMLabel *label)
{
  DM_Plex       *mesh = (DM_Plex*) dm->data;
  PlexLabel      next = mesh->labels;
  PlexLabel      last = NULL;
  PetscBool      hasLabel;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr   = DMPlexHasLabel(dm, name, &hasLabel);CHKERRQ(ierr);
  *label = NULL;
  if (!hasLabel) PetscFunctionReturn(0);
  while (next) {
    ierr = PetscStrcmp(name, next->label->name, &hasLabel);CHKERRQ(ierr);
    if (hasLabel) {
      if (last) last->next   = next->next;
      else      mesh->labels = next->next;
      next->next = NULL;
      *label     = next->label;
      ierr = PetscFree(next);CHKERRQ(ierr);
      ierr = PetscStrcmp(name, "depth", &hasLabel);CHKERRQ(ierr);
      if (hasLabel) mesh->depthLabel = NULL;
      break;
    }
    last = next;
    next = next->next;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexGetLabelOutput"
/*@C
  DMPlexGetLabelOutput - Get the output flag for a given label

  Not Collective

  Input Parameters:
+ dm   - The DMPlex object
- name - The label name

  Output Parameter:
. output - The flag for output

  Level: developer

.keywords: mesh
.seealso: DMPlexSetLabelOutput(), DMPlexCreateLabel(), DMPlexHasLabel(), DMPlexGetLabelValue(), DMPlexSetLabelValue(), DMPlexGetStratumIS()
@*/
PetscErrorCode DMPlexGetLabelOutput(DM dm, const char name[], PetscBool *output)
{
  DM_Plex       *mesh = (DM_Plex*) dm->data;
  PlexLabel      next = mesh->labels;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(name, 2);
  PetscValidPointer(output, 3);
  while (next) {
    PetscBool flg;

    ierr = PetscStrcmp(name, next->label->name, &flg);CHKERRQ(ierr);
    if (flg) {*output = next->output; PetscFunctionReturn(0);}
    next = next->next;
  }
  SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "No label named %s was present in this mesh", name);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexSetLabelOutput"
/*@C
  DMPlexSetLabelOutput - Set the output flag for a given label

  Not Collective

  Input Parameters:
+ dm     - The DMPlex object
. name   - The label name
- output - The flag for output

  Level: developer

.keywords: mesh
.seealso: DMPlexGetLabelOutput(), DMPlexCreateLabel(), DMPlexHasLabel(), DMPlexGetLabelValue(), DMPlexSetLabelValue(), DMPlexGetStratumIS()
@*/
PetscErrorCode DMPlexSetLabelOutput(DM dm, const char name[], PetscBool output)
{
  DM_Plex       *mesh = (DM_Plex*) dm->data;
  PlexLabel      next = mesh->labels;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(name, 2);
  while (next) {
    PetscBool flg;

    ierr = PetscStrcmp(name, next->label->name, &flg);CHKERRQ(ierr);
    if (flg) {next->output = output; PetscFunctionReturn(0);}
    next = next->next;
  }
  SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "No label named %s was present in this mesh", name);
}
