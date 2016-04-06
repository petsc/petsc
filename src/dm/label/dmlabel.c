#include <petsc/private/dmlabelimpl.h>   /*I      "petscdmlabel.h"   I*/
#include <petsc/private/isimpl.h>        /*I      "petscis.h"        I*/
#include <petscsf.h>

#undef __FUNCT__
#define __FUNCT__ "DMLabelCreate"
/*@C
  DMLabelCreate - Create a DMLabel object, which is a multimap

  Input parameter:
. name - The label name

  Output parameter:
. label - The DMLabel

  Level: beginner

.seealso: DMLabelDestroy()
@*/
PetscErrorCode DMLabelCreate(const char name[], DMLabel *label)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNew(label);CHKERRQ(ierr);
  ierr = PetscStrallocpy(name, &(*label)->name);CHKERRQ(ierr);

  (*label)->refct          = 1;
  (*label)->state          = -1;
  (*label)->numStrata      = 0;
  (*label)->defaultValue   = -1;
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
/*
  DMLabelMakeValid_Private - Transfer stratum data from the hash format to the sorted list format

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
  ++label->state;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMLabelMakeAllValid_Private"
/*
  DMLabelMakeAllValid_Private - Transfer all strata from the hash format to the sorted list format

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

#undef __FUNCT__
#define __FUNCT__ "DMLabelMakeInvalid_Private"
/*
  DMLabelMakeInvalid_Private - Transfer stratum data from the sorted list format to the hash format

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
#define __FUNCT__ "DMLabelGetState"
PetscErrorCode DMLabelGetState(DMLabel label, PetscObjectState *state)
{
  PetscFunctionBegin;
  PetscValidPointer(state, 2);
  *state = label->state;
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
/*@C
  DMLabelGetName - Return the name of a DMLabel object

  Input parameter:
. label - The DMLabel

  Output parameter:
. name - The label name

  Level: beginner

.seealso: DMLabelCreate()
@*/
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

  (*labelnew)->refct        = 1;
  (*labelnew)->numStrata    = label->numStrata;
  (*labelnew)->defaultValue = label->defaultValue;
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
#define __FUNCT__ "DMLabelGetDefaultValue"
/*
  DMLabelGetDefaultValue - Get the default value returned by DMLabelGetValue() if a point has not been explicitly given a value.
  When a label is created, it is initialized to -1.

  Input parameter:
. label - a DMLabel object

  Output parameter:
. defaultValue - the default value

  Level: beginner

.seealso: DMLabelSetDefaultValue(), DMLabelGetValue(), DMLabelSetValue()
*/
PetscErrorCode DMLabelGetDefaultValue(DMLabel label, PetscInt *defaultValue)
{
  PetscFunctionBegin;
  *defaultValue = label->defaultValue;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMLabelSetDefaultValue"
/*
  DMLabelSetDefaultValue - Set the default value returned by DMLabelGetValue() if a point has not been explicitly given a value.
  When a label is created, it is initialized to -1.

  Input parameter:
. label - a DMLabel object

  Output parameter:
. defaultValue - the default value

  Level: beginner

.seealso: DMLabelGetDefaultValue(), DMLabelGetValue(), DMLabelSetValue()
*/
PetscErrorCode DMLabelSetDefaultValue(DMLabel label, PetscInt defaultValue)
{
  PetscFunctionBegin;
  label->defaultValue = defaultValue;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMLabelGetValue"
/*@
  DMLabelGetValue - Return the value a label assigns to a point, or the label's default value (which is initially -1, and can be changed with DMLabelSetDefaultValue())

  Input Parameters:
+ label - the DMLabel
- point - the point

  Output Parameter:
. value - The point value, or -1

  Level: intermediate

.seealso: DMLabelCreate(), DMLabelSetValue(), DMLabelClearValue(), DMLabelGetDefaultValue(), DMLabelSetDefaultValue()
@*/
PetscErrorCode DMLabelGetValue(DMLabel label, PetscInt point, PetscInt *value)
{
  PetscInt       v;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(value, 3);
  *value = label->defaultValue;
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
  DMLabelSetValue - Set the value a label assigns to a point.  If the value is the same as the label's default value (which is initially -1, and can be changed with DMLabelSetDefaultValue() to somethingg different), then this function will do nothing.

  Input Parameters:
+ label - the DMLabel
. point - the point
- value - The point value

  Level: intermediate

.seealso: DMLabelCreate(), DMLabelGetValue(), DMLabelClearValue(), DMLabelGetDefaultValue(), DMLabelSetDefaultValue()
@*/
PetscErrorCode DMLabelSetValue(DMLabel label, PetscInt point, PetscInt value)
{
  PETSC_UNUSED PetscHashIIter iter, ret;
  PetscInt                    v;
  PetscErrorCode              ierr;

  PetscFunctionBegin;
  /* Find, or add, label value */
  if (value == label->defaultValue) PetscFunctionReturn(0);
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
#define __FUNCT__ "DMLabelHasStratum"
PetscErrorCode DMLabelHasStratum(DMLabel label, PetscInt value, PetscBool *exists)
{
  PetscInt v;

  PetscFunctionBegin;
  PetscValidPointer(exists, 3);
  *exists = PETSC_FALSE;
  for (v = 0; v < label->numStrata; ++v) {
    if (label->stratumValues[v] == value) {
      *exists = PETSC_TRUE;
      break;
    }
  }
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
      } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Need to implement this to speedup Stratify");
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
#define __FUNCT__ "DMLabelDistribute_Internal"
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
      for (l = 0; l < label->stratumSizes[s]; l++) {
        ierr = PetscSectionGetDof(rootSection, label->points[s][l], &dof);CHKERRQ(ierr);
        ierr = PetscSectionSetDof(rootSection, label->points[s][l], dof+1);CHKERRQ(ierr);
      }
    }
  }
  ierr = PetscSectionSetUp(rootSection);CHKERRQ(ierr);
  /* Create a point-wise array of stratum values */
  ierr = PetscSectionGetStorageSize(rootSection, &size);CHKERRQ(ierr);
  ierr = PetscMalloc1(size, &rootStrata);CHKERRQ(ierr);
  ierr = PetscCalloc1(nroots, &rootIdx);CHKERRQ(ierr);
  if (label) {
    for (s = 0; s < label->numStrata; ++s) {
      for (l = 0; l < label->stratumSizes[s]; l++) {
        const PetscInt p = label->points[s][l];
        ierr = PetscSectionGetOffset(rootSection, p, &offset);CHKERRQ(ierr);
        rootStrata[offset+rootIdx[p]++] = label->stratumValues[s];
      }
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

#undef __FUNCT__
#define __FUNCT__ "DMLabelDistribute"
PetscErrorCode DMLabelDistribute(DMLabel label, PetscSF sf, DMLabel *labelNew)
{
  MPI_Comm       comm;
  PetscSection   leafSection;
  PetscInt       p, pStart, pEnd, s, size, dof, offset, stratum;
  PetscInt      *leafStrata, *strataIdx;
  char          *name;
  PetscInt       nameSize;
  PetscHashI     stratumHash;
  PETSC_UNUSED   PetscHashIIter ret, iter;
  size_t         len = 0;
  PetscMPIInt    rank;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (label) {ierr = DMLabelMakeAllValid_Private(label);CHKERRQ(ierr);}
  ierr = PetscObjectGetComm((PetscObject)sf, &comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  /* Bcast name */
  if (!rank) {ierr = PetscStrlen(label->name, &len);CHKERRQ(ierr);}
  nameSize = len;
  ierr = MPI_Bcast(&nameSize, 1, MPIU_INT, 0, comm);CHKERRQ(ierr);
  ierr = PetscMalloc1(nameSize+1, &name);CHKERRQ(ierr);
  if (!rank) {ierr = PetscMemcpy(name, label->name, nameSize+1);CHKERRQ(ierr);}
  ierr = MPI_Bcast(name, nameSize+1, MPI_CHAR, 0, comm);CHKERRQ(ierr);
  ierr = DMLabelCreate(name, labelNew);CHKERRQ(ierr);
  ierr = PetscFree(name);CHKERRQ(ierr);
  /* Bcast defaultValue */
  if (!rank) (*labelNew)->defaultValue = label->defaultValue;
  ierr = MPI_Bcast(&(*labelNew)->defaultValue, 1, MPIU_INT, 0, comm);CHKERRQ(ierr);
  /* Distribute stratum values over the SF and get the point mapping on the receiver */
  ierr = DMLabelDistribute_Internal(label, sf, &leafSection, &leafStrata);CHKERRQ(ierr);
  /* Determine received stratum values and initialise new label*/
  PetscHashICreate(stratumHash);
  ierr = PetscSectionGetStorageSize(leafSection, &size);CHKERRQ(ierr);
  for (p = 0; p < size; ++p) PetscHashIPut(stratumHash, leafStrata[p], ret, iter);
  PetscHashISize(stratumHash, (*labelNew)->numStrata);
  ierr = PetscMalloc1((*labelNew)->numStrata, &(*labelNew)->arrayValid);CHKERRQ(ierr);
  for (s = 0; s < (*labelNew)->numStrata; ++s) (*labelNew)->arrayValid[s] = PETSC_TRUE;
  ierr = PetscMalloc1((*labelNew)->numStrata, &(*labelNew)->stratumValues);CHKERRQ(ierr);
  /* Turn leafStrata into indices rather than stratum values */
  offset = 0;
  ierr = PetscHashIGetKeys(stratumHash, &offset, (*labelNew)->stratumValues);CHKERRQ(ierr);
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
  PetscHashIDestroy(stratumHash);
  ierr = PetscFree(leafStrata);CHKERRQ(ierr);
  ierr = PetscFree(strataIdx);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&leafSection);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMLabelGather"
/*@
  DMLabelGather - Gather all label values from leafs into roots

  Input Parameters:
+ label - the DMLabel
. point - the Star Forest point communication map

  Input Parameters:
+ label - the new DMLabel with localised leaf values

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
  char          *name;
  PetscInt       nameSize;
  size_t         len = 0;
  PetscMPIInt    rank, numProcs;
  PetscErrorCode ierr;

  PetscFunctionBegin;
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
  /* Gather rank/index pairs of leaves into local roots to build
     an inverse, multi-rooted SF. Note that this ignores local leaf
     indexing due to the use of the multiSF in PetscSFGather. */
  ierr = PetscSFGetGraph(sf, &nroots, &nleaves, &ilocal, NULL);CHKERRQ(ierr);
  ierr = PetscMalloc1(nleaves, &leafPoints);CHKERRQ(ierr);
  for (p = 0; p < nleaves; p++) {
    leafPoints[p].index = ilocal[p];
    leafPoints[p].rank = rank;
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
#define __FUNCT__ "PetscSectionCreateGlobalSectionLabel"
/*@C
  PetscSectionCreateGlobalSectionLabel - Create a section describing the global field layout using
  the local section and an SF describing the section point overlap.

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

