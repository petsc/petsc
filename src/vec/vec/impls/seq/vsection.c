/*
   This file contains routines for basic section object implementation.
*/

#include <private/vecimpl.h>   /*I  "petscvec.h"   I*/

#if 0
/* Should I protect these for C++? */
#undef __FUNCT__
#define __FUNCT__ "PetscSectionGetDof"
PetscErrorCode PetscSectionGetDof(PetscUniformSection s, PetscInt point, PetscInt *numDof)
{
  PetscFunctionBegin;
  if ((point < s->pStart) || (point >= s->pEnd)) {
    SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Section point %d should be in [%d, %d)", point, s->pStart, s->pEnd);
  }
  *numDof = s->numDof;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSectionGetOffset"
PetscErrorCode PetscSectionGetOffset(PetscUniformSection s, PetscInt point, PetscInt *offset)
{
  PetscFunctionBegin;
  if ((point < s->pStart) || (point >= s->pEnd)) {
    SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Section point %d should be in [%d, %d)", point, s->pStart, s->pEnd);
  }
  *offset = s->numDof*(point - s->pStart);
  PetscFunctionReturn(0);
}
#endif

#undef __FUNCT__
#define __FUNCT__ "PetscSectionCreate"
/*@C
  PetscSectionCreate - Allocates PetscSection space and sets the map contents to the default.

  Collective on MPI_Comm

  Input Parameters:
+ comm - the MPI communicator
- s    - pointer to the section

  Level: developer

  Notes: Typical calling sequence
       PetscSectionCreate(MPI_Comm,PetscSection *);
       PetscSectionSetChart(PetscSection,low,high);
       PetscSectionSetDof(PetscSection,point,numdof);
       PetscSectionSetUp(PetscSection);
       PetscSectionGetOffset(PetscSection,point,PetscInt *);
       PetscSectionDestroy(PetscSection);

       The PetscSection object and methods are intended to be used in the PETSc Vec and Mat implementions; it is
       recommended they not be used in user codes unless you really gain something in their use.

  Fortran Notes:
      Not available from Fortran

.seealso: PetscSection, PetscSectionDestroy()
@*/
PetscErrorCode PetscSectionCreate(MPI_Comm comm, PetscSection *s)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNew(struct _n_PetscSection, s);CHKERRQ(ierr);
  (*s)->atlasLayout.comm   = comm;
  (*s)->atlasLayout.pStart = -1;
  (*s)->atlasLayout.pEnd   = -1;
  (*s)->atlasLayout.numDof = 1;
  (*s)->atlasDof           = PETSC_NULL;
  (*s)->atlasOff           = PETSC_NULL;
  (*s)->bc                 = PETSC_NULL;
  (*s)->bcIndices          = PETSC_NULL;
  (*s)->numFields          = 0;
  (*s)->field              = PETSC_NULL;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSectionGetNumFields"
PetscErrorCode PetscSectionGetNumFields(PetscSection s, PetscInt *numFields)
{
  PetscFunctionBegin;
  PetscValidPointer(numFields,2);
  *numFields = s->numFields;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSectionSetNumFields"
PetscErrorCode PetscSectionSetNumFields(PetscSection s, PetscInt numFields)
{
  PetscInt       f;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (numFields <= 0) {
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "The number of fields %d must be positive", numFields);
  }
  s->numFields = numFields;
  ierr = PetscMalloc(s->numFields * sizeof(PetscInt), &s->numFieldComponents);CHKERRQ(ierr);
  ierr = PetscMalloc(s->numFields * sizeof(PetscSection), &s->field);CHKERRQ(ierr);
  for(f = 0; f < s->numFields; ++f) {
    s->numFieldComponents[f] = 1;
    ierr = PetscSectionCreate(s->atlasLayout.comm, &s->field[f]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSectionGetFieldComponents"
PetscErrorCode PetscSectionGetFieldComponents(PetscSection s, PetscInt field, PetscInt *numComp)
{
  PetscFunctionBegin;
  PetscValidPointer(numComp,2);
  if ((field < 0) || (field >= s->numFields)) {
    SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Section field %d should be in [%d, %d)", field, 0, s->numFields);
  }
  *numComp = s->numFieldComponents[field];
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSectionSetFieldComponents"
PetscErrorCode PetscSectionSetFieldComponents(PetscSection s, PetscInt field, PetscInt numComp)
{
  PetscFunctionBegin;
  if ((field < 0) || (field >= s->numFields)) {
    SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Section field %d should be in [%d, %d)", field, 0, s->numFields);
  }
  s->numFieldComponents[field] = numComp;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSectionCheckConstraints"
PetscErrorCode PetscSectionCheckConstraints(PetscSection s)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!s->bc) {
    ierr = PetscSectionCreate(s->atlasLayout.comm, &s->bc);CHKERRQ(ierr);
    ierr = PetscSectionSetChart(s->bc, s->atlasLayout.pStart, s->atlasLayout.pEnd);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSectionGetChart"
PetscErrorCode PetscSectionGetChart(PetscSection s, PetscInt *pStart, PetscInt *pEnd)
{
  PetscFunctionBegin;
  if (pStart) {*pStart = s->atlasLayout.pStart;}
  if (pEnd)   {*pEnd   = s->atlasLayout.pEnd;}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSectionSetChart"
PetscErrorCode PetscSectionSetChart(PetscSection s, PetscInt pStart, PetscInt pEnd)
{
  PetscInt       f;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  s->atlasLayout.pStart = pStart;
  s->atlasLayout.pEnd   = pEnd;
  ierr = PetscFree2(s->atlasDof, s->atlasOff);CHKERRQ(ierr);
  ierr = PetscMalloc2((pEnd - pStart), PetscInt, &s->atlasDof, (pEnd - pStart), PetscInt, &s->atlasOff);CHKERRQ(ierr);
  ierr = PetscMemzero(s->atlasDof, (pEnd - pStart)*sizeof(PetscInt));CHKERRQ(ierr);
  for(f = 0; f < s->numFields; ++f) {
    ierr = PetscSectionSetChart(s->field[f], pStart, pEnd);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSectionGetDof"
PetscErrorCode PetscSectionGetDof(PetscSection s, PetscInt point, PetscInt *numDof)
{
  PetscFunctionBegin;
  if ((point < s->atlasLayout.pStart) || (point >= s->atlasLayout.pEnd)) {
    SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Section point %d should be in [%d, %d)", point, s->atlasLayout.pStart, s->atlasLayout.pEnd);
  }
  *numDof = s->atlasDof[point - s->atlasLayout.pStart];
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSectionSetDof"
PetscErrorCode PetscSectionSetDof(PetscSection s, PetscInt point, PetscInt numDof)
{
  PetscFunctionBegin;
  if ((point < s->atlasLayout.pStart) || (point >= s->atlasLayout.pEnd)) {
    SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Section point %d should be in [%d, %d)", point, s->atlasLayout.pStart, s->atlasLayout.pEnd);
  }
  s->atlasDof[point - s->atlasLayout.pStart] = numDof;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSectionAddDof"
PetscErrorCode PetscSectionAddDof(PetscSection s, PetscInt point, PetscInt numDof)
{
  PetscFunctionBegin;
  if ((point < s->atlasLayout.pStart) || (point >= s->atlasLayout.pEnd)) {
    SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Section point %d should be in [%d, %d)", point, s->atlasLayout.pStart, s->atlasLayout.pEnd);
  }
  s->atlasDof[point - s->atlasLayout.pStart] += numDof;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSectionGetFieldDof"
PetscErrorCode PetscSectionGetFieldDof(PetscSection s, PetscInt point, PetscInt field, PetscInt *numDof)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if ((field < 0) || (field >= s->numFields)) {
    SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Section field %d should be in [%d, %d)", field, 0, s->numFields);
  }
  ierr = PetscSectionGetDof(s->field[field], point, numDof);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSectionSetFieldDof"
PetscErrorCode PetscSectionSetFieldDof(PetscSection s, PetscInt point, PetscInt field, PetscInt numDof)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if ((field < 0) || (field >= s->numFields)) {
    SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Section field %d should be in [%d, %d)", field, 0, s->numFields);
  }
  ierr = PetscSectionSetDof(s->field[field], point, numDof);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSectionGetConstraintDof"
PetscErrorCode PetscSectionGetConstraintDof(PetscSection s, PetscInt point, PetscInt *numDof)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (s->bc) {
    ierr = PetscSectionGetDof(s->bc, point, numDof);CHKERRQ(ierr);
  } else {
    *numDof = 0;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSectionSetConstraintDof"
PetscErrorCode PetscSectionSetConstraintDof(PetscSection s, PetscInt point, PetscInt numDof)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (numDof) {
    ierr = PetscSectionCheckConstraints(s);CHKERRQ(ierr);
    ierr = PetscSectionSetDof(s->bc, point, numDof);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSectionAddConstraintDof"
PetscErrorCode PetscSectionAddConstraintDof(PetscSection s, PetscInt point, PetscInt numDof)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (numDof) {
    ierr = PetscSectionCheckConstraints(s);CHKERRQ(ierr);
    ierr = PetscSectionAddDof(s->bc, point, numDof);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSectionGetFieldConstraintDof"
PetscErrorCode PetscSectionGetFieldConstraintDof(PetscSection s, PetscInt point, PetscInt field, PetscInt *numDof)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if ((field < 0) || (field >= s->numFields)) {
    SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Section field %d should be in [%d, %d)", field, 0, s->numFields);
  }
  ierr = PetscSectionGetConstraintDof(s->field[field], point, numDof);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSectionSetFieldConstraintDof"
PetscErrorCode PetscSectionSetFieldConstraintDof(PetscSection s, PetscInt point, PetscInt field, PetscInt numDof)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if ((field < 0) || (field >= s->numFields)) {
    SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Section field %d should be in [%d, %d)", field, 0, s->numFields);
  }
  ierr = PetscSectionSetConstraintDof(s->field[field], point, numDof);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSectionSetUpBC"
PetscErrorCode PetscSectionSetUpBC(PetscSection s)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (s->bc) {
    const PetscInt last = (s->bc->atlasLayout.pEnd-s->bc->atlasLayout.pStart) - 1;

    ierr = PetscSectionSetUp(s->bc);CHKERRQ(ierr);
    ierr = PetscMalloc((s->bc->atlasOff[last] + s->bc->atlasDof[last]) * sizeof(PetscInt), &s->bcIndices);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSectionSetUp"
PetscErrorCode PetscSectionSetUp(PetscSection s)
{
  PetscInt       offset = 0, p, f;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  for(p = 0; p < s->atlasLayout.pEnd - s->atlasLayout.pStart; ++p) {
    s->atlasOff[p] = offset;
    offset += s->atlasDof[p];
  }
  ierr = PetscSectionSetUpBC(s);CHKERRQ(ierr);
  /* Assume that all fields have the same chart */
  for(p = 0; p < s->atlasLayout.pEnd - s->atlasLayout.pStart; ++p) {
    PetscInt off = s->atlasOff[p];

    for(f = 0; f < s->numFields; ++f) {
      PetscSection sf = s->field[f];

      sf->atlasOff[p] = off;
      off += sf->atlasDof[p];
    }
  }
  for(f = 0; f < s->numFields; ++f) {
    ierr = PetscSectionSetUpBC(s->field[f]);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSectionGetStorageSize"
PetscErrorCode PetscSectionGetStorageSize(PetscSection s, PetscInt *size)
{
  PetscInt p, n = 0;

  PetscFunctionBegin;
  for(p = 0; p < s->atlasLayout.pEnd - s->atlasLayout.pStart; ++p) {
    n += s->atlasDof[p];
  }
  *size = n;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSectionGetOffset"
PetscErrorCode PetscSectionGetOffset(PetscSection s, PetscInt point, PetscInt *offset)
{
  PetscFunctionBegin;
  if ((point < s->atlasLayout.pStart) || (point >= s->atlasLayout.pEnd)) {
    SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Section point %d should be in [%d, %d)", point, s->atlasLayout.pStart, s->atlasLayout.pEnd);
  }
  *offset = s->atlasOff[point - s->atlasLayout.pStart];
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSectionGetFieldOffset"
PetscErrorCode PetscSectionGetFieldOffset(PetscSection s, PetscInt point, PetscInt field, PetscInt *offset)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if ((field < 0) || (field >= s->numFields)) {
    SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Section field %d should be in [%d, %d)", field, 0, s->numFields);
  }
  ierr = PetscSectionGetOffset(s->field[field], point, offset);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSectionView_ASCII"
PetscErrorCode  PetscSectionView_ASCII(PetscSection s, PetscViewer viewer)
{
  PetscInt       p;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (s->atlasLayout.numDof != 1) {SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_SUP, "Cannot handle %d dof in a uniform section", s->atlasLayout.numDof);}
  for(p = 0; p < s->atlasLayout.pEnd - s->atlasLayout.pStart; ++p) {
    if ((s->bc) && (s->bc->atlasDof[p] > 0)) {
      PetscInt b;

      ierr = PetscViewerASCIIPrintf(viewer, "  (%4d) dim %2d offset %3d constrained", p+s->atlasLayout.pStart, s->atlasDof[p], s->atlasOff[p]);CHKERRQ(ierr);
      for(b = 0; b < s->bc->atlasDof[p]; ++b) {
        ierr = PetscViewerASCIIPrintf(viewer, " %d", s->bcIndices[s->bc->atlasOff[p]+b]);CHKERRQ(ierr);
      }
      ierr = PetscViewerASCIIPrintf(viewer, "\n");CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIIPrintf(viewer, "  (%4d) dim %2d offset %3d\n", p+s->atlasLayout.pStart, s->atlasDof[p], s->atlasOff[p]);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSectionView"
PetscErrorCode  PetscSectionView(PetscSection s, PetscViewer viewer)
{
  PetscBool      isascii;
  PetscInt       f;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!viewer) {ierr = PetscViewerASCIIGetStdout(PETSC_COMM_SELF, &viewer);CHKERRQ(ierr);}
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  ierr = PetscTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &isascii);CHKERRQ(ierr);
  if (isascii) {
    if (s->numFields) {
      ierr = PetscViewerASCIIPrintf(viewer, "PetscSection with %d fields\n", s->numFields);CHKERRQ(ierr);
      for(f = 0; f < s->numFields; ++f) {
        ierr = PetscViewerASCIIPrintf(viewer, "  field %d with %d components\n", f, s->numFieldComponents[f]);CHKERRQ(ierr);
        ierr = PetscSectionView_ASCII(s->field[f], viewer);CHKERRQ(ierr);
      }
    } else {
      ierr = PetscViewerASCIIPrintf(viewer, "PetscSection\n");CHKERRQ(ierr);
      ierr = PetscSectionView_ASCII(s, viewer);CHKERRQ(ierr);
    }
  } else {
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_SUP, "Viewer type %s not supported by this section object", ((PetscObject) viewer)->type_name);
  }
  PetscFunctionReturn(0);
}

/*@C
  PetscSectionDestroy - Frees a section object and frees its range if that exists.

  Collective on MPI_Comm

  Input Parameters:
. s - the PetscSection

  Level: developer

    The PetscSection object and methods are intended to be used in the PETSc Vec and Mat implementions; it is
    recommended they not be used in user codes unless you really gain something in their use.

  Fortran Notes:
    Not available from Fortran

.seealso: PetscSection, PetscSectionCreate()
@*/
#undef __FUNCT__
#define __FUNCT__ "PetscSectionDestroy"
PetscErrorCode  PetscSectionDestroy(PetscSection *s)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*s) PetscFunctionReturn(0);
  if (!(*s)->refcnt--) {
    PetscInt f;

    ierr = PetscFree((*s)->numFieldComponents);CHKERRQ(ierr);
    for(f = 0; f < (*s)->numFields; ++f) {
      ierr = PetscSectionDestroy(&(*s)->field[f]);CHKERRQ(ierr);
    }
    ierr = PetscSectionDestroy(&(*s)->bc);CHKERRQ(ierr);
    ierr = PetscFree((*s)->bcIndices);CHKERRQ(ierr);
    ierr = PetscFree2((*s)->atlasDof, (*s)->atlasOff);CHKERRQ(ierr);
    ierr = PetscFree((*s));CHKERRQ(ierr);
  }
  *s = PETSC_NULL;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecGetValuesSection"
PetscErrorCode VecGetValuesSection(Vec v, PetscSection s, PetscInt point, PetscScalar **values)
{
  PetscScalar   *baseArray;
  const PetscInt p = point - s->atlasLayout.pStart;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecGetArray(v, &baseArray);CHKERRQ(ierr);
  *values = &baseArray[s->atlasOff[p]];
  ierr = VecRestoreArray(v, &baseArray);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecIntGetValuesSection"
PetscErrorCode VecIntGetValuesSection(PetscInt *baseArray, PetscSection s, PetscInt point, PetscInt **values)
{
  const PetscInt p = point - s->atlasLayout.pStart;

  PetscFunctionBegin;
  *values = &baseArray[s->atlasOff[p]];
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecSetValuesSection"
PetscErrorCode VecSetValuesSection(Vec v, PetscSection s, PetscInt point, PetscScalar values[], InsertMode mode)
{
  PetscScalar    *baseArray, *array;
  const PetscBool doInsert    = mode == INSERT_VALUES     || mode == INSERT_ALL_VALUES ? PETSC_TRUE : PETSC_FALSE;
  const PetscBool doBC        = mode == INSERT_ALL_VALUES || mode == ADD_ALL_VALUES    ? PETSC_TRUE : PETSC_FALSE;
  const PetscInt  p           = point - s->atlasLayout.pStart;
  const PetscInt  orientation = 0; /* Needs to be included for use in closure operations */
  PetscInt        cDim        = 0;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscSectionGetConstraintDof(s, p, &cDim);CHKERRQ(ierr);
  ierr = VecGetArray(v, &baseArray);CHKERRQ(ierr);
  array = &baseArray[s->atlasOff[p]];
  if (!cDim) {
    if (orientation >= 0) {
      const PetscInt dim = s->atlasDof[p];
      PetscInt       i;

      if (doInsert) {
        for(i = 0; i < dim; ++i) {
          array[i] = values[i];
        }
      } else {
        for(i = 0; i < dim; ++i) {
          array[i] += values[i];
        }
      }
    } else {
      PetscInt offset = 0;
      PetscInt j      = -1, field, i;

      for(field = 0; field < s->numFields; ++field) {
        const PetscInt dim = s->field[field]->atlasDof[p]; /* PetscSectionGetFieldDof() */

        for(i = dim-1; i >= 0; --i) {
          array[++j] = values[i+offset];
        }
        offset += dim;
      }
    }
  } else {
    if (orientation >= 0) {
      const PetscInt  dim  = s->atlasDof[p];
      PetscInt        cInd = 0, i;
      PetscInt       *cDof;

      ierr = PetscSectionGetConstraintIndices(s, point, &cDof);CHKERRQ(ierr);
      if (doInsert) {
        for(i = 0; i < dim; ++i) {
          if ((cInd < cDim) && (i == cDof[cInd])) {
            if (doBC) {array[i] = values[i];} /* Constrained update */
            ++cInd;
            continue;
          }
          array[i] = values[i]; /* Unconstrained update */
        }
      } else {
        for(i = 0; i < dim; ++i) {
          if ((cInd < cDim) && (i == cDof[cInd])) {
            if (doBC) {array[i] += values[i];} /* Constrained update */
            ++cInd;
            continue;
          }
          array[i] += values[i]; /* Unconstrained update */
        }
      }
    } else {
      PetscInt *cDof;
      PetscInt  offset  = 0;
      PetscInt  cOffset = 0;
      PetscInt  j       = 0, field;

      ierr = PetscSectionGetConstraintIndices(s, point, &cDof);CHKERRQ(ierr);
      for(field = 0; field < s->numFields; ++field) {
        const PetscInt  dim = s->field[field]->atlasDof[p];     /* PetscSectionGetFieldDof() */
        const PetscInt tDim = s->field[field]->bc->atlasDof[p]; /* PetscSectionGetFieldConstraintDof() */
        const PetscInt sDim = dim - tDim;
        PetscInt       cInd = 0, i ,k;

        for(i = 0, k = dim+offset-1; i < dim; ++i, ++j, --k) {
          if ((cInd < sDim) && (j == cDof[cInd+cOffset])) {++cInd; continue;}
          array[j] = values[k];
        }
        offset  += dim;
        cOffset += dim - tDim;
      }
    }
  }
  ierr = VecRestoreArray(v, &baseArray);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "VecIntSetValuesSection"
PetscErrorCode VecIntSetValuesSection(PetscInt *baseArray, PetscSection s, PetscInt point, PetscInt values[], InsertMode mode)
{
  PetscInt      *array;
  const PetscInt p           = point - s->atlasLayout.pStart;
  const PetscInt orientation = 0; /* Needs to be included for use in closure operations */
  PetscInt       cDim        = 0;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscSectionGetConstraintDof(s, p, &cDim);CHKERRQ(ierr);
  array = &baseArray[s->atlasOff[p]];
  if (!cDim) {
    if (orientation >= 0) {
      const PetscInt dim = s->atlasDof[p];
      PetscInt       i;

      if (mode == INSERT_VALUES) {
        for(i = 0; i < dim; ++i) {
          array[i] = values[i];
        }
      } else {
        for(i = 0; i < dim; ++i) {
          array[i] += values[i];
        }
      }
    } else {
      PetscInt offset = 0;
      PetscInt j      = -1, field, i;

      for(field = 0; field < s->numFields; ++field) {
        const PetscInt dim = s->field[field]->atlasDof[p];

        for(i = dim-1; i >= 0; --i) {
          array[++j] = values[i+offset];
        }
        offset += dim;
      }
    }
  } else {
    if (orientation >= 0) {
      const PetscInt dim  = s->atlasDof[p];
      PetscInt       cInd = 0, i;
      PetscInt      *cDof;

      ierr = PetscSectionGetConstraintIndices(s, point, &cDof);CHKERRQ(ierr);
      if (mode == INSERT_VALUES) {
        for(i = 0; i < dim; ++i) {
          if ((cInd < cDim) && (i == cDof[cInd])) {++cInd; continue;}
          array[i] = values[i];
        }
      } else {
        for(i = 0; i < dim; ++i) {
          if ((cInd < cDim) && (i == cDof[cInd])) {++cInd; continue;}
          array[i] += values[i];
        }
      }
    } else {
      PetscInt *cDof;
      PetscInt  offset  = 0;
      PetscInt  cOffset = 0;
      PetscInt  j       = 0, field;

      ierr = PetscSectionGetConstraintIndices(s, point, &cDof);CHKERRQ(ierr);
      for(field = 0; field < s->numFields; ++field) {
        const PetscInt  dim = s->field[field]->atlasDof[p];     /* PetscSectionGetFieldDof() */
        const PetscInt tDim = s->field[field]->bc->atlasDof[p]; /* PetscSectionGetFieldConstraintDof() */
        const PetscInt sDim = dim - tDim;
        PetscInt       cInd = 0, i ,k;

        for(i = 0, k = dim+offset-1; i < dim; ++i, ++j, --k) {
          if ((cInd < sDim) && (j == cDof[cInd+cOffset])) {++cInd; continue;}
          array[j] = values[k];
        }
        offset  += dim;
        cOffset += dim - tDim;
      }
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSectionGetConstraintIndices"
PetscErrorCode PetscSectionGetConstraintIndices(PetscSection s, PetscInt point, PetscInt **indices)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (s->bc) {
    ierr = VecIntGetValuesSection(s->bcIndices, s->bc, point, indices);CHKERRQ(ierr);
  } else {
    *indices = PETSC_NULL;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSectionSetConstraintIndices"
PetscErrorCode PetscSectionSetConstraintIndices(PetscSection s, PetscInt point, PetscInt indices[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (s->bc) {
    ierr = VecIntSetValuesSection(s->bcIndices, s->bc, point, indices, INSERT_VALUES);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSectionGetFieldConstraintIndices"
PetscErrorCode PetscSectionGetFieldConstraintIndices(PetscSection s, PetscInt point, PetscInt field, PetscInt **indices)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if ((field < 0) || (field >= s->numFields)) {
    SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Section field %d should be in [%d, %d)", field, 0, s->numFields);
  }
  ierr = PetscSectionGetConstraintIndices(s->field[field], point, indices);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSectionSetFieldConstraintIndices"
PetscErrorCode PetscSectionSetFieldConstraintIndices(PetscSection s, PetscInt point, PetscInt field, PetscInt indices[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if ((field < 0) || (field >= s->numFields)) {
    SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Section field %d should be in [%d, %d)", field, 0, s->numFields);
  }
  ierr = PetscSectionSetConstraintIndices(s->field[field], point, indices);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
