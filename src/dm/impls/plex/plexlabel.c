#include <petsc-private/dmpleximpl.h>   /*I      "petscdmplex.h"   I*/

#undef __FUNCT__
#define __FUNCT__ "DMLabelCreate"
PetscErrorCode DMLabelCreate(const char name[], DMLabel *label)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNew(struct _n_DMLabel, label);CHKERRQ(ierr);
  ierr = PetscStrallocpy(name, &(*label)->name);CHKERRQ(ierr);

  (*label)->refct          = 1;
  (*label)->numStrata      = 0;
  (*label)->stratumValues  = NULL;
  (*label)->arrayValid     = PETSC_TRUE;
  (*label)->stratumOffsets = NULL;
  (*label)->stratumSizes   = NULL;
  (*label)->points         = NULL;
  (*label)->ht             = NULL;
  (*label)->pStart         = -1;
  (*label)->pEnd           = -1;
  (*label)->bt             = NULL;
  (*label)->next           = NULL;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMLabelMakeValid_Private"
static PetscErrorCode DMLabelMakeValid_Private(DMLabel label)
{
  PetscInt       off = 0, v;
  PetscErrorCode ierr;

  if (label->arrayValid) return 0;
  PetscFunctionBegin;
  ierr = PetscMalloc2(label->numStrata,PetscInt,&label->stratumSizes,label->numStrata+1,PetscInt,&label->stratumOffsets);CHKERRQ(ierr);
  for (v = 0; v < label->numStrata; ++v) {
    PetscInt size = 0;
    PetscHashISize(label->ht[v], size);
    label->stratumSizes[v]   = size;
    label->stratumOffsets[v] = off;
    off += size;
  }
  label->stratumOffsets[v] = off;
  ierr = PetscMalloc(off * sizeof(PetscInt), &label->points);CHKERRQ(ierr);
  off = 0;
  for (v = 0; v < label->numStrata; ++v) {
    PetscInt size = 0;
    PetscHashISize(label->ht[v], size);
    if (size) {PetscHashIGetKeys(label->ht[v], off, label->points);}
    PetscHashIDestroy(label->ht[v]);
    ierr = PetscSortInt(label->stratumSizes[v], &label->points[label->stratumOffsets[v]]);CHKERRQ(ierr);
    if (label->bt) {
      PetscInt p;

      for (p = label->stratumOffsets[v]; p < label->stratumOffsets[v]+label->stratumSizes[v]; ++p) {
        const PetscInt point = label->points[p];

        if ((point < label->pStart) || (point >= label->pEnd)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Label point %d is not in [%d, %d)", point, label->pStart, label->pEnd);
        ierr = PetscBTSet(label->bt, point - label->pStart);CHKERRQ(ierr);
      }
    }
  }
  ierr = PetscFree(label->ht);CHKERRQ(ierr);
  label->ht = NULL;
  if (off != label->stratumOffsets[label->numStrata]) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Invalid number of points %d should be %d", off, label->stratumOffsets[label->numStrata]);
  label->arrayValid = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMLabelMakeInvalid_Private"
static PetscErrorCode DMLabelMakeInvalid_Private(DMLabel label)
{
  PetscInt       v;
  PetscErrorCode ierr;

  if (!label->arrayValid) return 0;
  PetscFunctionBegin;
  ierr = PetscMalloc(label->numStrata * sizeof(PetscHashI), &label->ht);CHKERRQ(ierr);
  for (v = 0; v < label->numStrata; ++v) {
    PetscHashIIter ret, iter;
    PetscInt       p;

    PetscHashICreate(label->ht[v]);
    for (p = label->stratumOffsets[v]; p < label->stratumOffsets[v]+label->stratumSizes[v]; ++p) PetscHashIPut(label->ht[v], label->points[p], &ret, iter);
  }
  ierr = PetscFree2(label->stratumSizes,label->stratumOffsets);CHKERRQ(ierr);
  ierr = PetscFree(label->points);CHKERRQ(ierr);
  label->arrayValid = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMLabelGetName"
PetscErrorCode DMLabelGetName(DMLabel label, const char **name)
{
  PetscFunctionBegin;
  PetscValidCharPointer(name, 2);
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
  if (label) {
    ierr = PetscViewerASCIIPrintf(viewer, "Label '%s':\n", label->name);CHKERRQ(ierr);
    if (label->bt) {ierr = PetscViewerASCIIPrintf(viewer, "  Index has been calculated in [%d, %d)\n", label->pStart, label->pEnd);CHKERRQ(ierr);}
    for (v = 0; v < label->numStrata; ++v) {
      const PetscInt value = label->stratumValues[v];
      PetscInt       p;

      for (p = label->stratumOffsets[v]; p < label->stratumOffsets[v]+label->stratumSizes[v]; ++p) {
        ierr = PetscViewerASCIISynchronizedPrintf(viewer, "[%D]: %D (%D)\n", rank, label->points[p], value);CHKERRQ(ierr);
      }
    }
  }
  ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMLabelView"
PetscErrorCode DMLabelView(DMLabel label, PetscViewer viewer)
{
  PetscBool      iascii;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  if (label) {ierr = DMLabelMakeValid_Private(label);CHKERRQ(ierr);}
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = DMLabelView_Ascii(label, viewer);CHKERRQ(ierr);
  } else SETERRQ1(PetscObjectComm((PetscObject)viewer), PETSC_ERR_SUP, "Viewer type %s not supported by this mesh object", ((PetscObject) viewer)->type_name);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMLabelDestroy"
PetscErrorCode DMLabelDestroy(DMLabel *label)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!(*label)) PetscFunctionReturn(0);
  if (--(*label)->refct > 0) PetscFunctionReturn(0);
  ierr = PetscFree((*label)->name);CHKERRQ(ierr);
  ierr = PetscFree((*label)->stratumValues);CHKERRQ(ierr);
  ierr = PetscFree2((*label)->stratumSizes,(*label)->stratumOffsets);CHKERRQ(ierr);
  ierr = PetscFree((*label)->points);CHKERRQ(ierr);
  if ((*label)->ht) {
    PetscInt v;
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
  ierr = DMLabelMakeValid_Private(label);CHKERRQ(ierr);
  ierr = PetscNew(struct _n_DMLabel, labelnew);CHKERRQ(ierr);
  ierr = PetscStrallocpy(label->name, &(*labelnew)->name);CHKERRQ(ierr);

  (*labelnew)->refct      = 1;
  (*labelnew)->numStrata  = label->numStrata;
  (*labelnew)->arrayValid = PETSC_TRUE;
  if (label->numStrata) {
    ierr = PetscMalloc3(label->numStrata,PetscInt,&(*labelnew)->stratumValues,label->numStrata+1,PetscInt,&(*labelnew)->stratumOffsets,label->numStrata,PetscInt,&(*labelnew)->stratumSizes);CHKERRQ(ierr);
    ierr = PetscMalloc(label->stratumOffsets[label->numStrata] * sizeof(PetscInt), &(*labelnew)->points);CHKERRQ(ierr);
    /* Could eliminate unused space here */
    for (v = 0; v < label->numStrata; ++v) {
      (*labelnew)->stratumValues[v]  = label->stratumValues[v];
      (*labelnew)->stratumOffsets[v] = label->stratumOffsets[v];
      (*labelnew)->stratumSizes[v]   = label->stratumSizes[v];
      for (q = label->stratumOffsets[v]; q < label->stratumOffsets[v]+label->stratumSizes[v]; ++q) {
        (*labelnew)->points[q] = label->points[q];
      }
    }
    (*labelnew)->stratumOffsets[label->numStrata] = label->stratumOffsets[label->numStrata];
  }
  (*labelnew)->ht     = NULL;
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
  ierr = DMLabelMakeValid_Private(label);CHKERRQ(ierr);
  if (label->bt) {ierr = PetscBTDestroy(&label->bt);CHKERRQ(ierr);}
  label->pStart = pStart;
  label->pEnd   = pEnd;
  ierr = PetscBTCreate(pEnd - pStart, &label->bt);CHKERRQ(ierr);
  ierr = PetscBTMemzero(pEnd - pStart, label->bt);CHKERRQ(ierr);
  for (v = 0; v < label->numStrata; ++v) {
    PetscInt i;

    for (i = 0; i < label->stratumSizes[v]; ++i) {
      const PetscInt point = label->points[label->stratumOffsets[v]+i];

      if ((point < pStart) || (point >= pEnd)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Label point %d is not in [%d, %d)", point, pStart, pEnd);
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
  ierr = DMLabelMakeValid_Private(label);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
  if (!label->bt) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Must call DMLabelCreateIndex() before DMLabelHasPoint()");
  if ((point < label->pStart) || (point >= label->pEnd)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Label point %d is not in [%d, %d)", point, label->pStart, label->pEnd);
#endif
  *contains = PetscBTLookup(label->bt, point - label->pStart) ? PETSC_TRUE : PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMLabelGetValue"
PetscErrorCode DMLabelGetValue(DMLabel label, PetscInt point, PetscInt *value)
{
  PetscInt       v;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(value, 3);
  *value = -1;
  for (v = 0; v < label->numStrata; ++v) {
    if (label->arrayValid) {
      PetscInt i;

      ierr = PetscFindInt(point, label->stratumSizes[v], &label->points[label->stratumOffsets[v]], &i);CHKERRQ(ierr);
      if (i >= 0) {
        *value = label->stratumValues[v];
        break;
      }
    } else {
      PetscInt iter;

      PetscHashIMap(label->ht[v], point, iter);
      if (iter >= 0) {
        *value = label->stratumValues[v];
        break;
      }
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMLabelSetValue"
PetscErrorCode DMLabelSetValue(DMLabel label, PetscInt point, PetscInt value)
{
  PetscHashIIter iter, ret;
  PetscInt       v;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMLabelMakeInvalid_Private(label);CHKERRQ(ierr);
  /* Find, or add, label value */
  for (v = 0; v < label->numStrata; ++v) {
    if (label->stratumValues[v] == value) break;
  }
  /* Create new table */
  if (v >= label->numStrata) {
    PetscInt   *tmpV;
    PetscHashI *tmpH;

    ierr = PetscMalloc((label->numStrata+1) * sizeof(PetscInt), &tmpV);CHKERRQ(ierr);
    ierr = PetscMalloc((label->numStrata+1) * sizeof(PetscHashI), &tmpH);CHKERRQ(ierr);
    for (v = 0; v < label->numStrata; ++v) {
      tmpV[v] = label->stratumValues[v];
      tmpH[v] = label->ht[v];
    }
    tmpV[v] = value;
    PetscHashICreate(tmpH[v]);
    ++label->numStrata;
    ierr = PetscFree(label->stratumValues);CHKERRQ(ierr);
    ierr = PetscFree(label->ht);CHKERRQ(ierr);
    label->stratumValues = tmpV;
    label->ht            = tmpH;
  }
  /* Check whether point exists */
  PetscHashIPut(label->ht[v], point, &ret, iter);
#if 0
  /* Find, or add, label value */
  for (v = 0; v < label->numStrata; ++v) {
    if (label->stratumValues[v] == value) break;
  }
  if (v >= label->numStrata) {
    PetscInt *tmpV, *tmpO, *tmpS;

    ierr = PetscMalloc3(label->numStrata+1,PetscInt,&tmpV,label->numStrata+2,PetscInt,&tmpO,label->numStrata+1,PetscInt,&tmpS);CHKERRQ(ierr);
    for (v = 0; v < label->numStrata; ++v) {
      tmpV[v] = label->stratumValues[v];
      tmpO[v] = label->stratumOffsets[v];
      tmpS[v] = label->stratumSizes[v];
    }
    tmpV[v]   = value;
    tmpO[v]   = v == 0 ? 0 : label->stratumOffsets[v];
    tmpS[v]   = 0;
    tmpO[v+1] = tmpO[v];
    ++label->numStrata;
    ierr = PetscFree3(label->stratumValues,label->stratumOffsets,label->stratumSizes);CHKERRQ(ierr);

    label->stratumValues  = tmpV;
    label->stratumOffsets = tmpO;
    label->stratumSizes   = tmpS;
  }
  /* Check whether point exists */
  ierr = PetscFindInt(point,label->stratumSizes[v],label->points+label->stratumOffsets[v],&loc);CHKERRQ(ierr);
  if (loc < 0) {
    PetscInt off = label->stratumOffsets[v] - (loc+1); /* decode insert location */
    /* Check for reallocation */
    if (label->stratumSizes[v] >= label->stratumOffsets[v+1]-label->stratumOffsets[v]) {
      PetscInt oldSize   = label->stratumOffsets[v+1]-label->stratumOffsets[v];
      PetscInt newSize   = PetscMax(10, 2*oldSize);  /* Double the size, since 2 is the optimal base for this online algorithm */
      PetscInt shift     = newSize - oldSize;
      PetscInt allocSize = label->stratumOffsets[label->numStrata] + shift;
      PetscInt *newPoints;
      PetscInt w, q;

      ierr = PetscMalloc(allocSize * sizeof(PetscInt), &newPoints);CHKERRQ(ierr);
      for (q = 0; q < label->stratumOffsets[v]+label->stratumSizes[v]; ++q) {
        newPoints[q] = label->points[q];
      }
      for (w = v+1; w < label->numStrata; ++w) {
        for (q = label->stratumOffsets[w]; q < label->stratumOffsets[w]+label->stratumSizes[w]; ++q) {
          newPoints[q+shift] = label->points[q];
        }
        label->stratumOffsets[w] += shift;
      }
      label->stratumOffsets[label->numStrata] += shift;

      ierr          = PetscFree(label->points);CHKERRQ(ierr);
      label->points = newPoints;
    }
    ierr = PetscMemmove(&label->points[off+1], &label->points[off], (label->stratumSizes[v]+(loc+1)) * sizeof(PetscInt));CHKERRQ(ierr);

    label->points[off] = point;
    ++label->stratumSizes[v];
    if (label->bt) {
      if ((point < label->pStart) || (point >= label->pEnd)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Label point %d is not in [%d, %d)", point, label->pStart, label->pEnd);
      ierr = PetscBTSet(label->bt, point - label->pStart);CHKERRQ(ierr);
    }
  }
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMLabelClearValue"
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
  if (label->arrayValid) {
    /* Check whether point exists */
    ierr = PetscFindInt(point, label->stratumSizes[v], &label->points[label->stratumOffsets[v]], &p);CHKERRQ(ierr);
    if (p >= 0) {
      ierr = PetscMemmove(&label->points[p], &label->points[p+1], (label->stratumSizes[v]-p-1) * sizeof(PetscInt));CHKERRQ(ierr);
      --label->stratumSizes[v];
      if (label->bt) {
        if ((point < label->pStart) || (point >= label->pEnd)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Label point %d is not in [%d, %d)", point, label->pStart, label->pEnd);
        ierr = PetscBTClear(label->bt, point - label->pStart);CHKERRQ(ierr);
      }
    }
  } else {
    PetscInt iter;

    PetscHashIMap(label->ht[v], point, iter);
    if (iter >= 0) {PetscHashIDel(label->ht[v], point);}
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMLabelGetNumValues"
PetscErrorCode DMLabelGetNumValues(DMLabel label, PetscInt *numValues)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(numValues, 2);
  ierr = DMLabelMakeValid_Private(label);CHKERRQ(ierr);
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
  ierr = DMLabelMakeValid_Private(label);CHKERRQ(ierr);
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
  ierr = DMLabelMakeValid_Private(label);CHKERRQ(ierr);
  *size = 0;
  for (v = 0; v < label->numStrata; ++v) {
    if (label->stratumValues[v] == value) {
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
  ierr = DMLabelMakeValid_Private(label);CHKERRQ(ierr);
  for (v = 0; v < label->numStrata; ++v) {
    if (label->stratumValues[v] != value) continue;
    if (start) *start = label->points[label->stratumOffsets[v]];
    if (end)   *end   = label->points[label->stratumOffsets[v]+label->stratumSizes[v]-1]+1;
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
  ierr = DMLabelMakeValid_Private(label);CHKERRQ(ierr);
  *points = NULL;
  for (v = 0; v < label->numStrata; ++v) {
    if (label->stratumValues[v] == value) {
      ierr = ISCreateGeneral(PETSC_COMM_SELF, label->stratumSizes[v], &label->points[label->stratumOffsets[v]], PETSC_COPY_VALUES, points);CHKERRQ(ierr);
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
      const PetscInt point = label->points[label->stratumOffsets[v]+i];

      if ((point < label->pStart) || (point >= label->pEnd)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Label point %d is not in [%d, %d)", point, label->pStart, label->pEnd);
      ierr = PetscBTClear(label->bt, point - label->pStart);CHKERRQ(ierr);
    }
  }
  if (label->arrayValid) {
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
  ierr = DMLabelMakeValid_Private(label);CHKERRQ(ierr);
  label->pStart = start;
  label->pEnd   = end;
  if (label->bt) {ierr = PetscBTDestroy(&label->bt);CHKERRQ(ierr);}
  /* Could squish offsets, but would only make sense if I reallocate the storage */
  for (v = 0; v < label->numStrata; ++v) {
    const PetscInt offset = label->stratumOffsets[v];
    const PetscInt size   = label->stratumSizes[v];
    PetscInt       off    = offset, q;

    for (q = offset; q < offset+size; ++q) {
      const PetscInt point = label->points[q];

      if ((point < start) || (point >= end)) continue;
      label->points[off++] = point;
    }
    label->stratumSizes[v] = off-offset;
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
  ierr = DMLabelMakeValid_Private(label);CHKERRQ(ierr);
  ierr = DMLabelDuplicate(label, labelNew);CHKERRQ(ierr);
  ierr = DMLabelGetNumValues(*labelNew, &numValues);CHKERRQ(ierr);
  ierr = ISGetLocalSize(permutation, &numPoints);CHKERRQ(ierr);
  ierr = ISGetIndices(permutation, &perm);CHKERRQ(ierr);
  for (v = 0; v < numValues; ++v) {
    const PetscInt offset = (*labelNew)->stratumOffsets[v];
    const PetscInt size   = (*labelNew)->stratumSizes[v];

    for (q = offset; q < offset+size; ++q) {
      const PetscInt point = (*labelNew)->points[q];

      if ((point < 0) || (point >= numPoints)) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Label point %d is not in [0, %d) for the remapping", point, numPoints);
      (*labelNew)->points[q] = perm[point];
    }
    ierr = PetscSortInt(size, &(*labelNew)->points[offset]);CHKERRQ(ierr);
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
PetscErrorCode DMLabelDistribute(DMLabel label, PetscSection partSection, IS part, ISLocalToGlobalMapping renumbering, DMLabel *labelNew)
{
  MPI_Comm       comm = PetscObjectComm((PetscObject) partSection);
  PetscInt      *stratumSizes = NULL, *points = NULL, s, p;
  PetscMPIInt   *sendcnts = NULL, *offsets = NULL, *displs = NULL, proc;
  char          *name;
  PetscInt       nameSize;
  size_t         len = 0;
  PetscMPIInt    rank, numProcs;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (label) {ierr = DMLabelMakeValid_Private(label);CHKERRQ(ierr);}
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &numProcs);CHKERRQ(ierr);
  /* Bcast name */
  if (!rank) {ierr = PetscStrlen(label->name, &len);CHKERRQ(ierr);}
  nameSize = len;
  ierr = MPI_Bcast(&nameSize, 1, MPIU_INT, 0, comm);CHKERRQ(ierr);
  ierr = PetscMalloc(nameSize+1, &name);CHKERRQ(ierr);
  if (!rank) {ierr = PetscMemcpy(name, label->name, nameSize+1);CHKERRQ(ierr);}
  ierr = MPI_Bcast(name, nameSize+1, MPI_CHAR, 0, comm);CHKERRQ(ierr);
  ierr = DMLabelCreate(name, labelNew);CHKERRQ(ierr);
  ierr = PetscFree(name);CHKERRQ(ierr);
  /* Bcast numStrata */
  if (!rank) (*labelNew)->numStrata = label->numStrata;
  ierr = MPI_Bcast(&(*labelNew)->numStrata, 1, MPIU_INT, 0, comm);CHKERRQ(ierr);
  ierr = PetscMalloc((*labelNew)->numStrata * sizeof(PetscInt), &(*labelNew)->stratumValues);CHKERRQ(ierr);
  ierr = PetscMalloc2((*labelNew)->numStrata,PetscInt,&(*labelNew)->stratumSizes,(*labelNew)->numStrata+1,PetscInt,&(*labelNew)->stratumOffsets);CHKERRQ(ierr);
  /* Bcast stratumValues */
  if (!rank) {ierr = PetscMemcpy((*labelNew)->stratumValues, label->stratumValues, label->numStrata * sizeof(PetscInt));CHKERRQ(ierr);}
  ierr = MPI_Bcast((*labelNew)->stratumValues, (*labelNew)->numStrata, MPIU_INT, 0, comm);CHKERRQ(ierr);
  /* Find size on each process and Scatter: we use the fact that both the stratum points and partArray are sorted */
  if (!rank) {
    const PetscInt *partArray;
    PetscInt        proc;

    ierr = ISGetIndices(part, &partArray);CHKERRQ(ierr);
    ierr = PetscMalloc(numProcs*label->numStrata * sizeof(PetscInt), &stratumSizes);CHKERRQ(ierr);
    ierr = PetscMemzero(stratumSizes, numProcs*label->numStrata * sizeof(PetscInt));CHKERRQ(ierr);
    /* TODO We should switch to using binary search if the label is a lot smaller than partitions */
    for (proc = 0; proc < numProcs; ++proc) {
      PetscInt dof, off;

      ierr = PetscSectionGetDof(partSection, proc, &dof);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(partSection, proc, &off);CHKERRQ(ierr);
      for (s = 0; s < label->numStrata; ++s) {
        PetscInt lStart = label->stratumOffsets[s], lEnd = label->stratumOffsets[s]+label->stratumSizes[s];
        PetscInt pStart = off,                      pEnd = off+dof;

        while (pStart < pEnd && lStart < lEnd) {
          if (partArray[pStart] > label->points[lStart]) {
            ++lStart;
          } else if (label->points[lStart] > partArray[pStart]) {
            ++pStart;
          } else {
            ++stratumSizes[proc*label->numStrata+s];
            ++pStart; ++lStart;
          }
        }
      }
    }
    ierr = ISRestoreIndices(part, &partArray);CHKERRQ(ierr);
  }
  ierr = MPI_Scatter(stratumSizes, (*labelNew)->numStrata, MPIU_INT, (*labelNew)->stratumSizes, (*labelNew)->numStrata, MPIU_INT, 0, comm);CHKERRQ(ierr);
  /* Calculate stratumOffsets */
  (*labelNew)->stratumOffsets[0] = 0;
  for (s = 0; s < (*labelNew)->numStrata; ++s) {(*labelNew)->stratumOffsets[s+1] = (*labelNew)->stratumSizes[s] + (*labelNew)->stratumOffsets[s];}
  /* Pack points and Scatter */
  if (!rank) {
    const PetscInt *partArray;

    ierr = ISGetIndices(part, &partArray);CHKERRQ(ierr);
    ierr = PetscMalloc3(numProcs,PetscMPIInt,&sendcnts,numProcs,PetscMPIInt,&offsets,numProcs+1,PetscMPIInt,&displs);CHKERRQ(ierr);
    displs[0] = 0;
    for (p = 0; p < numProcs; ++p) {
      sendcnts[p] = 0;
      for (s = 0; s < label->numStrata; ++s) sendcnts[p] += stratumSizes[p*label->numStrata+s];
      offsets[p]  = displs[p];
      displs[p+1] = displs[p] + sendcnts[p];
    }
    ierr = PetscMalloc(displs[numProcs] * sizeof(PetscInt), &points);CHKERRQ(ierr);
    /* TODO We should switch to using binary search if the label is a lot smaller than partitions */
    for (proc = 0; proc < numProcs; ++proc) {
      PetscInt dof, off;

      ierr = PetscSectionGetDof(partSection, proc, &dof);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(partSection, proc, &off);CHKERRQ(ierr);
      for (s = 0; s < label->numStrata; ++s) {
        PetscInt lStart = label->stratumOffsets[s], lEnd = label->stratumOffsets[s]+label->stratumSizes[s];
        PetscInt pStart = off,                     pEnd = off+dof;

        while (pStart < pEnd && lStart < lEnd) {
          if (partArray[pStart] > label->points[lStart]) {
            ++lStart;
          } else if (label->points[lStart] > partArray[pStart]) {
            ++pStart;
          } else {
            points[offsets[proc]++] = label->points[lStart];
            ++pStart; ++lStart;
          }
        }
      }
    }
    ierr = ISRestoreIndices(part, &partArray);CHKERRQ(ierr);
  }
  ierr = PetscMalloc((*labelNew)->stratumOffsets[(*labelNew)->numStrata] * sizeof(PetscInt), &(*labelNew)->points);CHKERRQ(ierr);
  ierr = MPI_Scatterv(points, sendcnts, displs, MPIU_INT, (*labelNew)->points, (*labelNew)->stratumOffsets[(*labelNew)->numStrata], MPIU_INT, 0, comm);CHKERRQ(ierr);
  ierr = PetscFree(points);CHKERRQ(ierr);
  ierr = PetscFree3(sendcnts,offsets,displs);CHKERRQ(ierr);
  ierr = PetscFree(stratumSizes);CHKERRQ(ierr);
  /* Renumber points */
  ierr = ISGlobalToLocalMappingApply(renumbering, IS_GTOLM_MASK, (*labelNew)->stratumOffsets[(*labelNew)->numStrata], (*labelNew)->points, NULL, (*labelNew)->points);CHKERRQ(ierr);
  /* Sort points */
  for (s = 0; s < (*labelNew)->numStrata; ++s) {ierr = PetscSortInt((*labelNew)->stratumSizes[s], &(*labelNew)->points[(*labelNew)->stratumOffsets[s]]);CHKERRQ(ierr);}
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
  DMLabel        next  = mesh->labels;
  PetscBool      flg   = PETSC_FALSE;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidCharPointer(name, 2);
  while (next) {
    ierr = PetscStrcmp(name, next->name, &flg);CHKERRQ(ierr);
    if (flg) break;
    next = next->next;
  }
  if (!flg) {
    DMLabel tmpLabel = mesh->labels;

    ierr = DMLabelCreate(name, &mesh->labels);CHKERRQ(ierr);

    mesh->labels->next = tmpLabel;
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
  DMLabel  next  = mesh->labels;
  PetscInt n     = 0;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(numLabels, 2);
  while (next) {
    ++n;
    next = next->next;
  }
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
  DMLabel  next  = mesh->labels;
  PetscInt l     = 0;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidCharPointer(name, 3);
  while (next) {
    if (l == n) {
      *name = next->name;
      PetscFunctionReturn(0);
    }
    ++l;
    next = next->next;
  }
  SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Label %d does not exist in this DM", n);
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
  DM_Plex        *mesh = (DM_Plex*) dm->data;
  DMLabel        next  = mesh->labels;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidCharPointer(name, 2);
  PetscValidPointer(hasLabel, 3);
  *hasLabel = PETSC_FALSE;
  while (next) {
    ierr = PetscStrcmp(name, next->name, hasLabel);CHKERRQ(ierr);
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
  DM_Plex        *mesh = (DM_Plex*) dm->data;
  DMLabel        next  = mesh->labels;
  PetscBool      hasLabel;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidCharPointer(name, 2);
  PetscValidPointer(label, 3);
  *label = NULL;
  while (next) {
    ierr = PetscStrcmp(name, next->name, &hasLabel);CHKERRQ(ierr);
    if (hasLabel) {
      *label = next;
      break;
    }
    next = next->next;
  }
  PetscFunctionReturn(0);
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
  DM_Plex        *mesh = (DM_Plex*) dm->data;
  PetscBool      hasLabel;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = DMPlexHasLabel(dm, label->name, &hasLabel);CHKERRQ(ierr);
  if (hasLabel) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Label %s already exists in this DM", label->name);
  label->next  = mesh->labels;
  mesh->labels = label;
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
  DM_Plex        *mesh = (DM_Plex*) dm->data;
  DMLabel        next  = mesh->labels;
  DMLabel        last  = NULL;
  PetscBool      hasLabel;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr   = DMPlexHasLabel(dm, name, &hasLabel);CHKERRQ(ierr);
  *label = NULL;
  if (!hasLabel) PetscFunctionReturn(0);
  while (next) {
    ierr = PetscStrcmp(name, next->name, &hasLabel);CHKERRQ(ierr);
    if (hasLabel) {
      if (last) last->next   = next->next;
      else      mesh->labels = next->next;
      next->next = NULL;
      *label     = next;
      break;
    }
    last = next;
    next = next->next;
  }
  PetscFunctionReturn(0);
}
