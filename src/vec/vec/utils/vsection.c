/*
   This file contains routines for section object operations on Vecs
*/
#include <petsc/private/sectionimpl.h> /*I  "petscsection.h"   I*/
#include <petsc/private/vecimpl.h>     /*I  "petscvec.h"   I*/

static PetscErrorCode PetscSectionVecView_ASCII(PetscSection s, Vec v, PetscViewer viewer)
{
  PetscScalar *array;
  PetscInt     p, i;
  PetscMPIInt  rank;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)viewer), &rank));
  PetscCall(VecGetArray(v, &array));
  PetscCall(PetscViewerASCIIPushSynchronized(viewer));
  PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "Process %d:\n", rank));
  for (p = 0; p < s->pEnd - s->pStart; ++p) {
    if ((s->bc) && (s->bc->atlasDof[p] > 0)) {
      PetscInt b;

      PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "  (%4" PetscInt_FMT ") dim %2" PetscInt_FMT " offset %3" PetscInt_FMT, p + s->pStart, s->atlasDof[p], s->atlasOff[p]));
      for (i = s->atlasOff[p]; i < s->atlasOff[p] + s->atlasDof[p]; ++i) {
        PetscScalar v = array[i];
#if defined(PETSC_USE_COMPLEX)
        if (PetscImaginaryPart(v) > 0.0) {
          PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, " %g + %g i", (double)PetscRealPart(v), (double)PetscImaginaryPart(v)));
        } else if (PetscImaginaryPart(v) < 0.0) {
          PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, " %g - %g i", (double)PetscRealPart(v), (double)(-PetscImaginaryPart(v))));
        } else {
          PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, " %g", (double)PetscRealPart(v)));
        }
#else
        PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, " %g", (double)v));
#endif
      }
      PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, " constrained"));
      for (b = 0; b < s->bc->atlasDof[p]; ++b) PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, " %" PetscInt_FMT, s->bcIndices[s->bc->atlasOff[p] + b]));
      PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "\n"));
    } else {
      PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "  (%4" PetscInt_FMT ") dim %2" PetscInt_FMT " offset %3" PetscInt_FMT, p + s->pStart, s->atlasDof[p], s->atlasOff[p]));
      for (i = s->atlasOff[p]; i < s->atlasOff[p] + s->atlasDof[p]; ++i) {
        PetscScalar v = array[i];
#if defined(PETSC_USE_COMPLEX)
        if (PetscImaginaryPart(v) > 0.0) {
          PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, " %g + %g i", (double)PetscRealPart(v), (double)PetscImaginaryPart(v)));
        } else if (PetscImaginaryPart(v) < 0.0) {
          PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, " %g - %g i", (double)PetscRealPart(v), (double)(-PetscImaginaryPart(v))));
        } else {
          PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, " %g", (double)PetscRealPart(v)));
        }
#else
        PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, " %g", (double)v));
#endif
      }
      PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "\n"));
    }
  }
  PetscCall(PetscViewerFlush(viewer));
  PetscCall(PetscViewerASCIIPopSynchronized(viewer));
  PetscCall(VecRestoreArray(v, &array));
  PetscFunctionReturn(0);
}

/*@
  PetscSectionVecView - View a vector, using the section to structure the values

  Not collective

  Input Parameters:
+ s      - the organizing PetscSection
. v      - the Vec
- viewer - the PetscViewer

  Level: developer

.seealso: `PetscSection`, `PetscSectionCreate()`, `VecSetValuesSection()`
@*/
PetscErrorCode PetscSectionVecView(PetscSection s, Vec v, PetscViewer viewer)
{
  PetscBool isascii;
  PetscInt  f;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(s, PETSC_SECTION_CLASSID, 1);
  PetscValidHeaderSpecific(v, VEC_CLASSID, 2);
  if (!viewer) PetscCall(PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)v), &viewer));
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 3);
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  if (isascii) {
    const char *name;

    PetscCall(PetscObjectGetName((PetscObject)v, &name));
    if (s->numFields) {
      PetscCall(PetscViewerASCIIPrintf(viewer, "%s with %" PetscInt_FMT " fields\n", name, s->numFields));
      for (f = 0; f < s->numFields; ++f) {
        PetscCall(PetscViewerASCIIPrintf(viewer, "  field %" PetscInt_FMT " with %" PetscInt_FMT " components\n", f, s->numFieldComponents[f]));
        PetscCall(PetscSectionVecView_ASCII(s->field[f], v, viewer));
      }
    } else {
      PetscCall(PetscViewerASCIIPrintf(viewer, "%s\n", name));
      PetscCall(PetscSectionVecView_ASCII(s, v, viewer));
    }
  }
  PetscFunctionReturn(0);
}

/*@C
  VecGetValuesSection - Gets all the values associated with a given point, according to the section, in the given Vec

  Not collective

  Input Parameters:
+ v - the Vec
. s - the organizing PetscSection
- point - the point

  Output Parameter:
. values - the array of output values

  Level: developer

.seealso: `PetscSection`, `PetscSectionCreate()`, `VecSetValuesSection()`
@*/
PetscErrorCode VecGetValuesSection(Vec v, PetscSection s, PetscInt point, PetscScalar **values)
{
  PetscScalar   *baseArray;
  const PetscInt p = point - s->pStart;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v, VEC_CLASSID, 1);
  PetscValidHeaderSpecific(s, PETSC_SECTION_CLASSID, 2);
  PetscCall(VecGetArray(v, &baseArray));
  *values = &baseArray[s->atlasOff[p]];
  PetscCall(VecRestoreArray(v, &baseArray));
  PetscFunctionReturn(0);
}

/*@C
  VecSetValuesSection - Sets all the values associated with a given point, according to the section, in the given Vec

  Not collective

  Input Parameters:
+ v - the Vec
. s - the organizing PetscSection
. point - the point
. values - the array of input values
- mode - the insertion mode, either ADD_VALUES or INSERT_VALUES

  Level: developer

  Note: This is similar to MatSetValuesStencil(). The Fortran binding is
$
$   VecSetValuesSectionF90(vec, section, point, values, mode, ierr)
$

.seealso: `PetscSection`, `PetscSectionCreate()`, `VecGetValuesSection()`
@*/
PetscErrorCode VecSetValuesSection(Vec v, PetscSection s, PetscInt point, PetscScalar values[], InsertMode mode)
{
  PetscScalar    *baseArray, *array;
  const PetscBool doInsert    = mode == INSERT_VALUES || mode == INSERT_ALL_VALUES || mode == INSERT_BC_VALUES ? PETSC_TRUE : PETSC_FALSE;
  const PetscBool doInterior  = mode == INSERT_ALL_VALUES || mode == ADD_ALL_VALUES || mode == INSERT_VALUES || mode == ADD_VALUES ? PETSC_TRUE : PETSC_FALSE;
  const PetscBool doBC        = mode == INSERT_ALL_VALUES || mode == ADD_ALL_VALUES || mode == INSERT_BC_VALUES || mode == ADD_BC_VALUES ? PETSC_TRUE : PETSC_FALSE;
  const PetscInt  p           = point - s->pStart;
  const PetscInt  orientation = 0; /* Needs to be included for use in closure operations */
  PetscInt        cDim        = 0;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v, VEC_CLASSID, 1);
  PetscValidHeaderSpecific(s, PETSC_SECTION_CLASSID, 2);
  PetscCall(PetscSectionGetConstraintDof(s, point, &cDim));
  PetscCall(VecGetArray(v, &baseArray));
  array = &baseArray[s->atlasOff[p]];
  if (!cDim && doInterior) {
    if (orientation >= 0) {
      const PetscInt dim = s->atlasDof[p];
      PetscInt       i;

      if (doInsert) {
        for (i = 0; i < dim; ++i) array[i] = values[i];
      } else {
        for (i = 0; i < dim; ++i) array[i] += values[i];
      }
    } else {
      PetscInt offset = 0;
      PetscInt j      = -1, field, i;

      for (field = 0; field < s->numFields; ++field) {
        const PetscInt dim = s->field[field]->atlasDof[p]; /* PetscSectionGetFieldDof() */

        for (i = dim - 1; i >= 0; --i) array[++j] = values[i + offset];
        offset += dim;
      }
    }
  } else if (cDim) {
    if (orientation >= 0) {
      const PetscInt  dim  = s->atlasDof[p];
      PetscInt        cInd = 0, i;
      const PetscInt *cDof;

      PetscCall(PetscSectionGetConstraintIndices(s, point, &cDof));
      if (doInsert) {
        for (i = 0; i < dim; ++i) {
          if ((cInd < cDim) && (i == cDof[cInd])) {
            if (doBC) array[i] = values[i]; /* Constrained update */
            ++cInd;
            continue;
          }
          if (doInterior) array[i] = values[i]; /* Unconstrained update */
        }
      } else {
        for (i = 0; i < dim; ++i) {
          if ((cInd < cDim) && (i == cDof[cInd])) {
            if (doBC) array[i] += values[i]; /* Constrained update */
            ++cInd;
            continue;
          }
          if (doInterior) array[i] += values[i]; /* Unconstrained update */
        }
      }
    } else {
      /* TODO This is broken for add and constrained update */
      const PetscInt *cDof;
      PetscInt        offset  = 0;
      PetscInt        cOffset = 0;
      PetscInt        j       = 0, field;

      PetscCall(PetscSectionGetConstraintIndices(s, point, &cDof));
      for (field = 0; field < s->numFields; ++field) {
        const PetscInt dim  = s->field[field]->atlasDof[p];     /* PetscSectionGetFieldDof() */
        const PetscInt tDim = s->field[field]->bc->atlasDof[p]; /* PetscSectionGetFieldConstraintDof() */
        const PetscInt sDim = dim - tDim;
        PetscInt       cInd = 0, i, k;

        for (i = 0, k = dim + offset - 1; i < dim; ++i, ++j, --k) {
          if ((cInd < sDim) && (j == cDof[cInd + cOffset])) {
            ++cInd;
            continue;
          }
          if (doInterior) array[j] = values[k]; /* Unconstrained update */
        }
        offset += dim;
        cOffset += dim - tDim;
      }
    }
  }
  PetscCall(VecRestoreArray(v, &baseArray));
  PetscFunctionReturn(0);
}

PetscErrorCode PetscSectionGetField_Internal(PetscSection section, PetscSection sectionGlobal, Vec v, PetscInt field, PetscInt pStart, PetscInt pEnd, IS *is, Vec *subv)
{
  PetscInt *subIndices;
  PetscInt  Nc, subSize = 0, subOff = 0, p;

  PetscFunctionBegin;
  PetscCall(PetscSectionGetFieldComponents(section, field, &Nc));
  for (p = pStart; p < pEnd; ++p) {
    PetscInt gdof, fdof = 0;

    PetscCall(PetscSectionGetDof(sectionGlobal, p, &gdof));
    if (gdof > 0) PetscCall(PetscSectionGetFieldDof(section, p, field, &fdof));
    subSize += fdof;
  }
  PetscCall(PetscMalloc1(subSize, &subIndices));
  for (p = pStart; p < pEnd; ++p) {
    PetscInt gdof, goff;

    PetscCall(PetscSectionGetDof(sectionGlobal, p, &gdof));
    if (gdof > 0) {
      PetscInt fdof, fc, f2, poff = 0;

      PetscCall(PetscSectionGetOffset(sectionGlobal, p, &goff));
      /* Can get rid of this loop by storing field information in the global section */
      for (f2 = 0; f2 < field; ++f2) {
        PetscCall(PetscSectionGetFieldDof(section, p, f2, &fdof));
        poff += fdof;
      }
      PetscCall(PetscSectionGetFieldDof(section, p, field, &fdof));
      for (fc = 0; fc < fdof; ++fc, ++subOff) subIndices[subOff] = goff + poff + fc;
    }
  }
  PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)v), subSize, subIndices, PETSC_OWN_POINTER, is));
  PetscCall(VecGetSubVector(v, *is, subv));
  PetscCall(VecSetBlockSize(*subv, Nc));
  PetscFunctionReturn(0);
}

PetscErrorCode PetscSectionRestoreField_Internal(PetscSection section, PetscSection sectionGlobal, Vec v, PetscInt field, PetscInt pStart, PetscInt pEnd, IS *is, Vec *subv)
{
  PetscFunctionBegin;
  PetscCall(VecRestoreSubVector(v, *is, subv));
  PetscCall(ISDestroy(is));
  PetscFunctionReturn(0);
}

/*@C
  PetscSectionVecNorm - Computes the vector norm, separated into field components.

  Input Parameters:
+ s    - the local Section
. gs   - the global section
. x    - the vector
- type - one of NORM_1, NORM_2, NORM_INFINITY.

  Output Parameter:
. val  - the array of norms

  Level: intermediate

.seealso: `VecNorm()`, `PetscSectionCreate()`
@*/
PetscErrorCode PetscSectionVecNorm(PetscSection s, PetscSection gs, Vec x, NormType type, PetscReal val[])
{
  PetscInt Nf, f, pStart, pEnd;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(s, PETSC_SECTION_CLASSID, 1);
  PetscValidHeaderSpecific(gs, PETSC_SECTION_CLASSID, 2);
  PetscValidHeaderSpecific(x, VEC_CLASSID, 3);
  PetscValidRealPointer(val, 5);
  PetscCall(PetscSectionGetNumFields(s, &Nf));
  if (Nf < 2) PetscCall(VecNorm(x, type, val));
  else {
    PetscCall(PetscSectionGetChart(s, &pStart, &pEnd));
    for (f = 0; f < Nf; ++f) {
      Vec subv;
      IS  is;

      PetscCall(PetscSectionGetField_Internal(s, gs, x, f, pStart, pEnd, &is, &subv));
      PetscCall(VecNorm(subv, type, &val[f]));
      PetscCall(PetscSectionRestoreField_Internal(s, gs, x, f, pStart, pEnd, &is, &subv));
    }
  }
  PetscFunctionReturn(0);
}
