/*
   This file contains routines for section object operations on Vecs
*/
#include <petsc/private/sectionimpl.h> /*I  "petscsection.h"   I*/
#include <petsc/private/vecimpl.h>     /*I  "petscvec.h"   I*/

static PetscErrorCode PetscSectionVecView_ASCII(PetscSection s, Vec v, PetscViewer viewer)
{
  PetscScalar    *array;
  PetscInt       p, i;
  PetscMPIInt    rank;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)viewer), &rank);CHKERRQ(ierr);
  ierr = VecGetArray(v, &array);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPushSynchronized(viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIISynchronizedPrintf(viewer, "Process %d:\n", rank);CHKERRQ(ierr);
  for (p = 0; p < s->pEnd - s->pStart; ++p) {
    if ((s->bc) && (s->bc->atlasDof[p] > 0)) {
      PetscInt b;

      ierr = PetscViewerASCIISynchronizedPrintf(viewer, "  (%4D) dim %2D offset %3D", p+s->pStart, s->atlasDof[p], s->atlasOff[p]);CHKERRQ(ierr);
      for (i = s->atlasOff[p]; i < s->atlasOff[p]+s->atlasDof[p]; ++i) {
        PetscScalar v = array[i];
#if defined(PETSC_USE_COMPLEX)
        if (PetscImaginaryPart(v) > 0.0) {
          ierr = PetscViewerASCIISynchronizedPrintf(viewer," %g + %g i", (double)PetscRealPart(v), (double)PetscImaginaryPart(v));CHKERRQ(ierr);
        } else if (PetscImaginaryPart(v) < 0.0) {
          ierr = PetscViewerASCIISynchronizedPrintf(viewer," %g - %g i", (double)PetscRealPart(v),(double)(-PetscImaginaryPart(v)));CHKERRQ(ierr);
        } else {
          ierr = PetscViewerASCIISynchronizedPrintf(viewer, " %g", (double)PetscRealPart(v));CHKERRQ(ierr);
        }
#else
        ierr = PetscViewerASCIISynchronizedPrintf(viewer, " %g", (double)v);CHKERRQ(ierr);
#endif
      }
      ierr = PetscViewerASCIISynchronizedPrintf(viewer, " constrained");CHKERRQ(ierr);
      for (b = 0; b < s->bc->atlasDof[p]; ++b) {
        ierr = PetscViewerASCIISynchronizedPrintf(viewer, " %D", s->bcIndices[s->bc->atlasOff[p]+b]);CHKERRQ(ierr);
      }
      ierr = PetscViewerASCIISynchronizedPrintf(viewer, "\n");CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIISynchronizedPrintf(viewer, "  (%4D) dim %2D offset %3D", p+s->pStart, s->atlasDof[p], s->atlasOff[p]);CHKERRQ(ierr);
      for (i = s->atlasOff[p]; i < s->atlasOff[p]+s->atlasDof[p]; ++i) {
        PetscScalar v = array[i];
#if defined(PETSC_USE_COMPLEX)
        if (PetscImaginaryPart(v) > 0.0) {
          ierr = PetscViewerASCIISynchronizedPrintf(viewer," %g + %g i", (double)PetscRealPart(v), (double)PetscImaginaryPart(v));CHKERRQ(ierr);
        } else if (PetscImaginaryPart(v) < 0.0) {
          ierr = PetscViewerASCIISynchronizedPrintf(viewer," %g - %g i", (double)PetscRealPart(v),(double)(-PetscImaginaryPart(v)));CHKERRQ(ierr);
        } else {
          ierr = PetscViewerASCIISynchronizedPrintf(viewer, " %g", (double)PetscRealPart(v));CHKERRQ(ierr);
        }
#else
        ierr = PetscViewerASCIISynchronizedPrintf(viewer, " %g", (double)v);CHKERRQ(ierr);
#endif
      }
      ierr = PetscViewerASCIISynchronizedPrintf(viewer, "\n");CHKERRQ(ierr);
    }
  }
  ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPopSynchronized(viewer);CHKERRQ(ierr);
  ierr = VecRestoreArray(v, &array);CHKERRQ(ierr);
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

.seealso: PetscSection, PetscSectionCreate(), VecSetValuesSection()
@*/
PetscErrorCode PetscSectionVecView(PetscSection s, Vec v, PetscViewer viewer)
{
  PetscBool      isascii;
  PetscInt       f;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(s, PETSC_SECTION_CLASSID, 1);
  PetscValidHeaderSpecific(v, VEC_CLASSID, 2);
  if (!viewer) {ierr = PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)v), &viewer);CHKERRQ(ierr);}
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 3);
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &isascii);CHKERRQ(ierr);
  if (isascii) {
    const char *name;

    ierr = PetscObjectGetName((PetscObject) v, &name);CHKERRQ(ierr);
    if (s->numFields) {
      ierr = PetscViewerASCIIPrintf(viewer, "%s with %D fields\n", name, s->numFields);CHKERRQ(ierr);
      for (f = 0; f < s->numFields; ++f) {
        ierr = PetscViewerASCIIPrintf(viewer, "  field %D with %D components\n", f, s->numFieldComponents[f]);CHKERRQ(ierr);
        ierr = PetscSectionVecView_ASCII(s->field[f], v, viewer);CHKERRQ(ierr);
      }
    } else {
      ierr = PetscViewerASCIIPrintf(viewer, "%s\n", name);CHKERRQ(ierr);
      ierr = PetscSectionVecView_ASCII(s, v, viewer);CHKERRQ(ierr);
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

.seealso: PetscSection, PetscSectionCreate(), VecSetValuesSection()
@*/
PetscErrorCode VecGetValuesSection(Vec v, PetscSection s, PetscInt point, PetscScalar **values)
{
  PetscScalar    *baseArray;
  const PetscInt p = point - s->pStart;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v, VEC_CLASSID, 1);
  PetscValidHeaderSpecific(s, PETSC_SECTION_CLASSID, 2);
  ierr = VecGetArray(v, &baseArray);CHKERRQ(ierr);
  *values = &baseArray[s->atlasOff[p]];
  ierr = VecRestoreArray(v, &baseArray);CHKERRQ(ierr);
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

.seealso: PetscSection, PetscSectionCreate(), VecGetValuesSection()
@*/
PetscErrorCode VecSetValuesSection(Vec v, PetscSection s, PetscInt point, PetscScalar values[], InsertMode mode)
{
  PetscScalar     *baseArray, *array;
  const PetscBool doInsert    = mode == INSERT_VALUES     || mode == INSERT_ALL_VALUES || mode == INSERT_BC_VALUES                          ? PETSC_TRUE : PETSC_FALSE;
  const PetscBool doInterior  = mode == INSERT_ALL_VALUES || mode == ADD_ALL_VALUES    || mode == INSERT_VALUES    || mode == ADD_VALUES    ? PETSC_TRUE : PETSC_FALSE;
  const PetscBool doBC        = mode == INSERT_ALL_VALUES || mode == ADD_ALL_VALUES    || mode == INSERT_BC_VALUES || mode == ADD_BC_VALUES ? PETSC_TRUE : PETSC_FALSE;
  const PetscInt  p           = point - s->pStart;
  const PetscInt  orientation = 0; /* Needs to be included for use in closure operations */
  PetscInt        cDim        = 0;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v, VEC_CLASSID, 1);
  PetscValidHeaderSpecific(s, PETSC_SECTION_CLASSID, 2);
  ierr  = PetscSectionGetConstraintDof(s, point, &cDim);CHKERRQ(ierr);
  ierr  = VecGetArray(v, &baseArray);CHKERRQ(ierr);
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

        for (i = dim-1; i >= 0; --i) array[++j] = values[i+offset];
        offset += dim;
      }
    }
  } else if (cDim) {
    if (orientation >= 0) {
      const PetscInt dim  = s->atlasDof[p];
      PetscInt       cInd = 0, i;
      const PetscInt *cDof;

      ierr = PetscSectionGetConstraintIndices(s, point, &cDof);CHKERRQ(ierr);
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
      PetscInt       offset  = 0;
      PetscInt       cOffset = 0;
      PetscInt       j       = 0, field;

      ierr = PetscSectionGetConstraintIndices(s, point, &cDof);CHKERRQ(ierr);
      for (field = 0; field < s->numFields; ++field) {
        const PetscInt dim  = s->field[field]->atlasDof[p];     /* PetscSectionGetFieldDof() */
        const PetscInt tDim = s->field[field]->bc->atlasDof[p]; /* PetscSectionGetFieldConstraintDof() */
        const PetscInt sDim = dim - tDim;
        PetscInt       cInd = 0, i ,k;

        for (i = 0, k = dim+offset-1; i < dim; ++i, ++j, --k) {
          if ((cInd < sDim) && (j == cDof[cInd+cOffset])) {++cInd; continue;}
          if (doInterior) array[j] = values[k];   /* Unconstrained update */
        }
        offset  += dim;
        cOffset += dim - tDim;
      }
    }
  }
  ierr = VecRestoreArray(v, &baseArray);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscSectionGetField_Internal(PetscSection section, PetscSection sectionGlobal, Vec v, PetscInt field, PetscInt pStart, PetscInt pEnd, IS *is, Vec *subv)
{
  PetscInt      *subIndices;
  PetscInt       Nc, subSize = 0, subOff = 0, p;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscSectionGetFieldComponents(section, field, &Nc);CHKERRQ(ierr);
  for (p = pStart; p < pEnd; ++p) {
    PetscInt gdof, fdof = 0;

    ierr = PetscSectionGetDof(sectionGlobal, p, &gdof);CHKERRQ(ierr);
    if (gdof > 0) {ierr = PetscSectionGetFieldDof(section, p, field, &fdof);CHKERRQ(ierr);}
    subSize += fdof;
  }
  ierr = PetscMalloc1(subSize, &subIndices);CHKERRQ(ierr);
  for (p = pStart; p < pEnd; ++p) {
    PetscInt gdof, goff;

    ierr = PetscSectionGetDof(sectionGlobal, p, &gdof);CHKERRQ(ierr);
    if (gdof > 0) {
      PetscInt fdof, fc, f2, poff = 0;

      ierr = PetscSectionGetOffset(sectionGlobal, p, &goff);CHKERRQ(ierr);
      /* Can get rid of this loop by storing field information in the global section */
      for (f2 = 0; f2 < field; ++f2) {
        ierr  = PetscSectionGetFieldDof(section, p, f2, &fdof);CHKERRQ(ierr);
        poff += fdof;
      }
      ierr = PetscSectionGetFieldDof(section, p, field, &fdof);CHKERRQ(ierr);
      for (fc = 0; fc < fdof; ++fc, ++subOff) subIndices[subOff] = goff+poff+fc;
    }
  }
  ierr = ISCreateGeneral(PetscObjectComm((PetscObject) v), subSize, subIndices, PETSC_OWN_POINTER, is);CHKERRQ(ierr);
  ierr = VecGetSubVector(v, *is, subv);CHKERRQ(ierr);
  ierr = VecSetBlockSize(*subv, Nc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscSectionRestoreField_Internal(PetscSection section, PetscSection sectionGlobal, Vec v, PetscInt field, PetscInt pStart, PetscInt pEnd, IS *is, Vec *subv)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecRestoreSubVector(v, *is, subv);CHKERRQ(ierr);
  ierr = ISDestroy(is);CHKERRQ(ierr);
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

.seealso: VecNorm(), PetscSectionCreate()
@*/
PetscErrorCode PetscSectionVecNorm(PetscSection s, PetscSection gs, Vec x, NormType type, PetscReal val[])
{
  PetscInt       Nf, f, pStart, pEnd;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(s, PETSC_SECTION_CLASSID, 1);
  PetscValidHeaderSpecific(gs, PETSC_SECTION_CLASSID, 2);
  PetscValidHeaderSpecific(x, VEC_CLASSID, 3);
  PetscValidPointer(val, 5);
  ierr = PetscSectionGetNumFields(s, &Nf);CHKERRQ(ierr);
  if (Nf < 2) {ierr = VecNorm(x, type, val);CHKERRQ(ierr);}
  else {
    ierr = PetscSectionGetChart(s, &pStart, &pEnd);CHKERRQ(ierr);
    for (f = 0; f < Nf; ++f) {
      Vec subv;
      IS  is;

      ierr = PetscSectionGetField_Internal(s, gs, x, f, pStart, pEnd, &is, &subv);CHKERRQ(ierr);
      ierr = VecNorm(subv, type, &val[f]);CHKERRQ(ierr);
      ierr = PetscSectionRestoreField_Internal(s, gs, x, f, pStart, pEnd, &is, &subv);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}
