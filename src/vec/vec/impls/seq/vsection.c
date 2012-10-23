/*
   This file contains routines for basic section object implementation.
*/

#include <petsc-private/vecimpl.h>   /*I  "petscvec.h"   I*/

#undef __FUNCT__
#define __FUNCT__ "PetscSectionCreate"
/*@
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
  (*s)->maxDof             = 0;
  (*s)->atlasDof           = PETSC_NULL;
  (*s)->atlasOff           = PETSC_NULL;
  (*s)->bc                 = PETSC_NULL;
  (*s)->bcIndices          = PETSC_NULL;
  (*s)->setup              = PETSC_FALSE;
  (*s)->numFields          = 0;
  (*s)->fieldNames         = PETSC_NULL;
  (*s)->field              = PETSC_NULL;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSectionClone"
/*@
  PetscSectionClone - Creates a shallow (if possible) copy of the PetscSection

  Collective on MPI_Comm

  Input Parameter:
. section - the PetscSection

  Output Parameter:
. newSection - the copy

  Level: developer

.seealso: PetscSection, PetscSectionCreate(), PetscSectionDestroy()
@*/
PetscErrorCode PetscSectionClone(PetscSection section, PetscSection *newSection)
{
  PetscInt       numFields, f, pStart, pEnd, p;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscSectionCreate(section->atlasLayout.comm, newSection);CHKERRQ(ierr);
  ierr = PetscSectionGetNumFields(section, &numFields);CHKERRQ(ierr);
  if (numFields) {ierr = PetscSectionSetNumFields(*newSection, numFields);CHKERRQ(ierr);}
  for (f = 0; f < numFields; ++f) {
    const char *name;
    PetscInt    numComp;

    ierr = PetscSectionGetFieldName(section, f, &name);CHKERRQ(ierr);
    ierr = PetscSectionSetFieldName(*newSection, f, name);CHKERRQ(ierr);
    ierr = PetscSectionGetFieldComponents(section, f, &numComp);CHKERRQ(ierr);
    ierr = PetscSectionSetFieldComponents(*newSection, f, numComp);CHKERRQ(ierr);
  }
  ierr = PetscSectionGetChart(section, &pStart, &pEnd);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(*newSection, pStart, pEnd);CHKERRQ(ierr);
  for (p = pStart; p < pEnd; ++p) {
    PetscInt dof, cdof, fcdof;

    ierr = PetscSectionGetDof(section, p, &dof);CHKERRQ(ierr);
    ierr = PetscSectionSetDof(*newSection, p, dof);CHKERRQ(ierr);
    ierr = PetscSectionGetConstraintDof(section, p, &cdof);CHKERRQ(ierr);
    if (cdof) {ierr = PetscSectionSetConstraintDof(*newSection, p, cdof);CHKERRQ(ierr);}
    for (f = 0; f < numFields; ++f) {
      ierr = PetscSectionGetFieldDof(section, p, f, &dof);CHKERRQ(ierr);
      ierr = PetscSectionSetFieldDof(*newSection, p, f, dof);CHKERRQ(ierr);
      if (cdof) {
        ierr = PetscSectionGetFieldConstraintDof(section, p, f, &fcdof);CHKERRQ(ierr);
        if (fcdof) {ierr = PetscSectionSetFieldConstraintDof(*newSection, p, f, fcdof);CHKERRQ(ierr);}
      }
    }
  }
  ierr = PetscSectionSetUp(*newSection);CHKERRQ(ierr);
  for (p = pStart; p < pEnd; ++p) {
    PetscInt        cdof, fcdof;
    const PetscInt *cInd;

    ierr = PetscSectionGetConstraintDof(section, p, &cdof);CHKERRQ(ierr);
    if (cdof) {
      ierr = PetscSectionGetConstraintIndices(section, p, &cInd);CHKERRQ(ierr);
      ierr = PetscSectionSetConstraintIndices(*newSection, p, cInd);CHKERRQ(ierr);
      for (f = 0; f < numFields; ++f) {
        ierr = PetscSectionGetFieldConstraintDof(section, p, f, &fcdof);CHKERRQ(ierr);
        if (fcdof) {
          ierr = PetscSectionGetFieldConstraintIndices(section, p, f, &cInd);CHKERRQ(ierr);
          ierr = PetscSectionSetFieldConstraintIndices(*newSection, p, f, cInd);CHKERRQ(ierr);
        }
      }
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSectionGetNumFields"
/*@
  PetscSectionGetNumFields - Returns the number of fields, or 0 if no fields were defined.

  Not collective

  Input Parameter:
. s - the PetscSection

  Output Parameter:
. numFields - the number of fields defined, or 0 if none were defined

  Level: intermediate

.seealso: PetscSectionSetNumFields()
@*/
PetscErrorCode PetscSectionGetNumFields(PetscSection s, PetscInt *numFields)
{
  PetscFunctionBegin;
  PetscValidPointer(numFields,2);
  *numFields = s->numFields;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSectionSetNumFields"
/*@
  PetscSectionSetNumFields - Sets the number of fields.

  Not collective

  Input Parameters:
+ s - the PetscSection
- numFields - the number of fields

  Level: intermediate

.seealso: PetscSectionGetNumFields()
@*/
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
  ierr = PetscMalloc(s->numFields * sizeof(char *), &s->fieldNames);CHKERRQ(ierr);
  ierr = PetscMalloc(s->numFields * sizeof(PetscSection), &s->field);CHKERRQ(ierr);
  for (f = 0; f < s->numFields; ++f) {
    char name[64];

    s->numFieldComponents[f] = 1;
    ierr = PetscSectionCreate(s->atlasLayout.comm, &s->field[f]);CHKERRQ(ierr);
    ierr = PetscSNPrintf(name, 64, "Field_%D", f);CHKERRQ(ierr);
    ierr = PetscStrallocpy(name, (char **) &s->fieldNames[f]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSectionGetFieldName"
/*@C
  PetscSectionGetFieldName - Returns the name of a field in the PetscSection

  Not Collective

  Input Parameters:
+ s     - the PetscSection
- field - the field number

  Output Parameter:
. fieldName - the field name

  Level: developer

.seealso: PetscSectionSetFieldName()
@*/
PetscErrorCode PetscSectionGetFieldName(PetscSection s, PetscInt field, const char *fieldName[])
{
  PetscFunctionBegin;
  PetscValidPointer(fieldName,2);
  if ((field < 0) || (field >= s->numFields)) {
    SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Section field %d should be in [%d, %d)", field, 0, s->numFields);
  }
  *fieldName = s->fieldNames[field];
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSectionSetFieldName"
/*@C
  PetscSectionSetFieldName - Sets the name of a field in the PetscSection

  Not Collective

  Input Parameters:
+ s     - the PetscSection
. field - the field number
- fieldName - the field name

  Level: developer

.seealso: PetscSectionGetFieldName()
@*/
PetscErrorCode PetscSectionSetFieldName(PetscSection s, PetscInt field, const char fieldName[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidCharPointer(fieldName,3);
  if ((field < 0) || (field >= s->numFields)) {
    SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Section field %d should be in [%d, %d)", field, 0, s->numFields);
  }
  ierr = PetscFree(s->fieldNames[field]);CHKERRQ(ierr);
  ierr = PetscStrallocpy(fieldName, (char **) &s->fieldNames[field]);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSectionGetFieldComponents"
/*@
  PetscSectionGetFieldComponents - Returns the number of field components for the given field.

  Not collective

  Input Parameters:
+ s - the PetscSection
- field - the field number

  Output Parameter:
. numComp - the number of field components

  Level: intermediate

.seealso: PetscSectionSetNumFieldComponents(), PetscSectionGetNumFields()
@*/
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
/*@
  PetscSectionSetFieldComponents - Sets the number of field components for the given field.

  Not collective

  Input Parameters:
+ s - the PetscSection
. field - the field number
- numComp - the number of field components

  Level: intermediate

.seealso: PetscSectionGetNumFieldComponents(), PetscSectionGetNumFields()
@*/
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
/*@
  PetscSectionGetChart - Returns the range [pStart, pEnd) in which points in the lie.

  Not collective

  Input Parameter:
. s - the PetscSection

  Output Parameters:
+ pStart - the first point
- pEnd - one past the last point

  Level: intermediate

.seealso: PetscSectionSetChart(), PetscSectionCreate()
@*/
PetscErrorCode PetscSectionGetChart(PetscSection s, PetscInt *pStart, PetscInt *pEnd)
{
  PetscFunctionBegin;
  if (pStart) {*pStart = s->atlasLayout.pStart;}
  if (pEnd)   {*pEnd   = s->atlasLayout.pEnd;}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSectionSetChart"
/*@
  PetscSectionSetChart - Sets the range [pStart, pEnd) in which points in the lie.

  Not collective

  Input Parameters:
+ s - the PetscSection
. pStart - the first point
- pEnd - one past the last point

  Level: intermediate

.seealso: PetscSectionSetChart(), PetscSectionCreate()
@*/
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
  for (f = 0; f < s->numFields; ++f) {
    ierr = PetscSectionSetChart(s->field[f], pStart, pEnd);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSectionGetDof"
/*@
  PetscSectionGetDof - Return the number of degrees of freedom associated with a given point.

  Not collective

  Input Parameters:
+ s - the PetscSection
- point - the point

  Output Parameter:
. numDof - the number of dof

  Level: intermediate

.seealso: PetscSectionSetDof(), PetscSectionCreate()
@*/
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
/*@
  PetscSectionSetDof - Sets the number of degrees of freedom associated with a given point.

  Not collective

  Input Parameters:
+ s - the PetscSection
. point - the point
- numDof - the number of dof

  Level: intermediate

.seealso: PetscSectionGetDof(), PetscSectionAddDof(), PetscSectionCreate()
@*/
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
/*@
  PetscSectionAddDof - Adds to the number of degrees of freedom associated with a given point.

  Not collective

  Input Parameters:
+ s - the PetscSection
. point - the point
- numDof - the number of additional dof

  Level: intermediate

.seealso: PetscSectionGetDof(), PetscSectionSetDof(), PetscSectionCreate()
@*/
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
/*@
  PetscSectionGetFieldDof - Return the number of degrees of freedom associated with a field on a given point.

  Not collective

  Input Parameters:
+ s - the PetscSection
. point - the point
- field - the field

  Output Parameter:
. numDof - the number of dof

  Level: intermediate

.seealso: PetscSectionSetFieldDof(), PetscSectionCreate()
@*/
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
/*@
  PetscSectionSetFieldDof - Sets the number of degrees of freedom associated with a field on a given point.

  Not collective

  Input Parameters:
+ s - the PetscSection
. point - the point
. field - the field
- numDof - the number of dof

  Level: intermediate

.seealso: PetscSectionGetFieldDof(), PetscSectionCreate()
@*/
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
#define __FUNCT__ "PetscSectionAddFieldDof"
/*@
  PetscSectionAddFieldDof - Adds a number of degrees of freedom associated with a field on a given point.

  Not collective

  Input Parameters:
+ s - the PetscSection
. point - the point
. field - the field
- numDof - the number of dof

  Level: intermediate

.seealso: PetscSectionSetFieldDof(), PetscSectionGetFieldDof(), PetscSectionCreate()
@*/
PetscErrorCode PetscSectionAddFieldDof(PetscSection s, PetscInt point, PetscInt field, PetscInt numDof)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if ((field < 0) || (field >= s->numFields)) {
    SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Section field %d should be in [%d, %d)", field, 0, s->numFields);
  }
  ierr = PetscSectionAddDof(s->field[field], point, numDof);CHKERRQ(ierr);
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
#define __FUNCT__ "PetscSectionAddFieldConstraintDof"
PetscErrorCode PetscSectionAddFieldConstraintDof(PetscSection s, PetscInt point, PetscInt field, PetscInt numDof)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if ((field < 0) || (field >= s->numFields)) {
    SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Section field %d should be in [%d, %d)", field, 0, s->numFields);
  }
  ierr = PetscSectionAddConstraintDof(s->field[field], point, numDof);CHKERRQ(ierr);
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
/*@
  PetscSectionSetUp - Calculate offsets based upon the number of degrees of freedom for each point.

  Not collective

  Input Parameter:
. s - the PetscSection

  Level: intermediate

.seealso: PetscSectionCreate()
@*/
PetscErrorCode PetscSectionSetUp(PetscSection s)
{
  PetscInt       offset = 0, p, f;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (s->setup) {PetscFunctionReturn(0);}
  s->setup = PETSC_TRUE;
  for (p = 0; p < s->atlasLayout.pEnd - s->atlasLayout.pStart; ++p) {
    s->atlasOff[p] = offset;
    offset += s->atlasDof[p];
    s->maxDof = PetscMax(s->maxDof, s->atlasDof[p]);
  }
  ierr = PetscSectionSetUpBC(s);CHKERRQ(ierr);
  /* Assume that all fields have the same chart */
  for (p = 0; p < s->atlasLayout.pEnd - s->atlasLayout.pStart; ++p) {
    PetscInt off = s->atlasOff[p];

    for (f = 0; f < s->numFields; ++f) {
      PetscSection sf = s->field[f];

      sf->atlasOff[p] = off;
      off += sf->atlasDof[p];
    }
  }
  for (f = 0; f < s->numFields; ++f) {
    ierr = PetscSectionSetUpBC(s->field[f]);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSectionGetMaxDof"
/*@
  PetscSectionGetMaxDof - Return the maximum number of degrees of freedom on any point in the chart

  Not collective

  Input Parameters:
. s - the PetscSection

  Output Parameter:
. maxDof - the maximum dof

  Level: intermediate

.seealso: PetscSectionGetDof(), PetscSectionSetDof(), PetscSectionCreate()
@*/
PetscErrorCode PetscSectionGetMaxDof(PetscSection s, PetscInt *maxDof)
{
  PetscFunctionBegin;
  *maxDof = s->maxDof;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSectionGetStorageSize"
/*@
  PetscSectionGetStorageSize - Return the size of an arary or local Vec capable of holding all the degrees of freedom.

  Not collective

  Input Parameters:
+ s - the PetscSection
- point - the point

  Output Parameter:
. size - the size of an array which can hold all the dofs

  Level: intermediate

.seealso: PetscSectionGetOffset(), PetscSectionCreate()
@*/
PetscErrorCode PetscSectionGetStorageSize(PetscSection s, PetscInt *size)
{
  PetscInt p, n = 0;

  PetscFunctionBegin;
  for (p = 0; p < s->atlasLayout.pEnd - s->atlasLayout.pStart; ++p) {
    n += s->atlasDof[p] > 0 ? s->atlasDof[p] : 0;
  }
  *size = n;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSectionGetConstrainedStorageSize"
PetscErrorCode PetscSectionGetConstrainedStorageSize(PetscSection s, PetscInt *size)
{
  PetscInt p, n = 0;

  PetscFunctionBegin;
  for (p = 0; p < s->atlasLayout.pEnd - s->atlasLayout.pStart; ++p) {
    const PetscInt cdof = s->bc ? s->bc->atlasDof[p] : 0;
    n += s->atlasDof[p] > 0 ? s->atlasDof[p] - cdof : 0;
  }
  *size = n;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSectionCreateGlobalSection"
/*@
  PetscSectionCreateGlobalSection - Create a section describing the global field layout using
  the local section and an SF describing the section point overlap.

  Input Parameters:
  + lsection - The PetscSection for the global field layout
  . sf - The SF describing parallel layout of the section points
  - includeConstraints - By default this is PETSC_FALSE, meaning that the global field vector will not posses constrained dofs

  Output Parameter:
  . gsection - The PetscSection for the global field layout

  Note: This gives negative sizes and offsets to points not owned by this process

  Level: developer

.seealso: PetscSectionCreate()
@*/
PetscErrorCode PetscSectionCreateGlobalSection(PetscSection s, PetscSF sf, PetscBool includeConstraints, PetscSection *gsection)
{
  PetscInt      *neg;
  PetscInt       pStart, pEnd, p, dof, cdof, off, globalOff = 0, nroots;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscSectionCreate(s->atlasLayout.comm, gsection);CHKERRQ(ierr);
  ierr = PetscSectionGetChart(s, &pStart, &pEnd);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(*gsection, pStart, pEnd);CHKERRQ(ierr);
  ierr = PetscMalloc((pEnd - pStart) * sizeof(PetscInt), &neg);CHKERRQ(ierr);
  /* Mark ghost points with negative dof */
  for (p = pStart; p < pEnd; ++p) {
    ierr = PetscSectionGetDof(s, p, &dof);CHKERRQ(ierr);
    ierr = PetscSectionSetDof(*gsection, p, dof);CHKERRQ(ierr);
    ierr = PetscSectionGetConstraintDof(s, p, &cdof);CHKERRQ(ierr);
    if (!includeConstraints && cdof > 0) {ierr = PetscSectionSetConstraintDof(*gsection, p, cdof);CHKERRQ(ierr);}
    neg[p-pStart] = -(dof+1);
  }
  ierr = PetscSectionSetUpBC(*gsection);CHKERRQ(ierr);
  ierr = PetscSFGetGraph(sf, &nroots, PETSC_NULL, PETSC_NULL, PETSC_NULL);CHKERRQ(ierr);
  if (nroots >= 0) {
    if (nroots > pEnd - pStart) {
      PetscInt *tmpDof;
      /* Help Jed: HAVE TO MAKE A BUFFER HERE THE SIZE OF THE COMPLETE SPACE AND THEN COPY INTO THE atlasDof FOR THIS SECTION */
      ierr = PetscMalloc(nroots * sizeof(PetscInt), &tmpDof);CHKERRQ(ierr);
      ierr = PetscSFBcastBegin(sf, MPIU_INT, &neg[-pStart], tmpDof);CHKERRQ(ierr);
      ierr = PetscSFBcastEnd(sf, MPIU_INT, &neg[-pStart], tmpDof);CHKERRQ(ierr);
      for (p = pStart; p < pEnd; ++p) {
        if (tmpDof[p] < 0) {(*gsection)->atlasDof[p-pStart] = tmpDof[p];}
      }
      ierr = PetscFree(tmpDof);CHKERRQ(ierr);
    } else {
      ierr = PetscSFBcastBegin(sf, MPIU_INT, &neg[-pStart], &(*gsection)->atlasDof[-pStart]);CHKERRQ(ierr);
      ierr = PetscSFBcastEnd(sf, MPIU_INT, &neg[-pStart], &(*gsection)->atlasDof[-pStart]);CHKERRQ(ierr);
    }
  }
  /* Calculate new sizes, get proccess offset, and calculate point offsets */
  for (p = 0, off = 0; p < pEnd-pStart; ++p) {
    cdof = (!includeConstraints && s->bc) ? s->bc->atlasDof[p] : 0;
    (*gsection)->atlasOff[p] = off;
    off += (*gsection)->atlasDof[p] > 0 ? (*gsection)->atlasDof[p]-cdof : 0;
  }
  ierr = MPI_Scan(&off, &globalOff, 1, MPIU_INT, MPI_SUM, s->atlasLayout.comm);CHKERRQ(ierr);
  globalOff -= off;
  for (p = 0, off = 0; p < pEnd-pStart; ++p) {
    (*gsection)->atlasOff[p] += globalOff;
    neg[p] = -((*gsection)->atlasOff[p]+1);
  }
  /* Put in negative offsets for ghost points */
  if (nroots >= 0) {
    if (nroots > pEnd - pStart) {
      PetscInt *tmpOff;
      /* Help Jed: HAVE TO MAKE A BUFFER HERE THE SIZE OF THE COMPLETE SPACE AND THEN COPY INTO THE atlasDof FOR THIS SECTION */
      ierr = PetscMalloc(nroots * sizeof(PetscInt), &tmpOff);CHKERRQ(ierr);
      ierr = PetscSFBcastBegin(sf, MPIU_INT, &neg[-pStart], tmpOff);CHKERRQ(ierr);
      ierr = PetscSFBcastEnd(sf, MPIU_INT, &neg[-pStart], tmpOff);CHKERRQ(ierr);
      for (p = pStart; p < pEnd; ++p) {
        if (tmpOff[p] < 0) {(*gsection)->atlasOff[p-pStart] = tmpOff[p];}
      }
      ierr = PetscFree(tmpOff);CHKERRQ(ierr);
    } else {
      ierr = PetscSFBcastBegin(sf, MPIU_INT, &neg[-pStart], &(*gsection)->atlasOff[-pStart]);CHKERRQ(ierr);
      ierr = PetscSFBcastEnd(sf, MPIU_INT, &neg[-pStart], &(*gsection)->atlasOff[-pStart]);CHKERRQ(ierr);
    }
  }
  ierr = PetscFree(neg);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSectionCreateGlobalSectionCensored"
/*@
  PetscSectionCreateGlobalSectionCensored - Create a section describing the global field layout using
  the local section and an SF describing the section point overlap.

  Input Parameters:
  + lsection - The PetscSection for the global field layout
  . sf - The SF describing parallel layout of the section points
  . includeConstraints - By default this is PETSC_FALSE, meaning that the global field vector will not posses constrained dofs
  . numExcludes - The number of exclusion ranges
  - excludes - An array [start_0, end_0, start_1, end_1, ...] where there are numExcludes pairs

  Output Parameter:
  . gsection - The PetscSection for the global field layout

  Note: This gives negative sizes and offsets to points not owned by this process

  Level: developer

.seealso: PetscSectionCreate()
@*/
PetscErrorCode PetscSectionCreateGlobalSectionCensored(PetscSection s, PetscSF sf, PetscBool includeConstraints, PetscInt numExcludes, const PetscInt excludes[], PetscSection *gsection)
{
  PetscInt      *neg;
  PetscInt       pStart, pEnd, p, e, dof, cdof, off, globalOff = 0, nroots;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscSectionCreate(s->atlasLayout.comm, gsection);CHKERRQ(ierr);
  ierr = PetscSectionGetChart(s, &pStart, &pEnd);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(*gsection, pStart, pEnd);CHKERRQ(ierr);
  ierr = PetscMalloc((pEnd - pStart) * sizeof(PetscInt), &neg);CHKERRQ(ierr);
  /* Mark ghost points with negative dof */
  for (p = pStart; p < pEnd; ++p) {
    for(e = 0; e < numExcludes; ++e) {
      if ((p >= excludes[e*2+0]) && (p < excludes[e*2+1])) {
        ierr = PetscSectionSetDof(*gsection, p, 0);CHKERRQ(ierr);
        break;
      }
    }
    if (e < numExcludes) continue;
    ierr = PetscSectionGetDof(s, p, &dof);CHKERRQ(ierr);
    ierr = PetscSectionSetDof(*gsection, p, dof);CHKERRQ(ierr);
    ierr = PetscSectionGetConstraintDof(s, p, &cdof);CHKERRQ(ierr);
    if (!includeConstraints && cdof > 0) {ierr = PetscSectionSetConstraintDof(*gsection, p, cdof);CHKERRQ(ierr);}
    neg[p-pStart] = -(dof+1);
  }
  ierr = PetscSectionSetUpBC(*gsection);CHKERRQ(ierr);
  ierr = PetscSFGetGraph(sf, &nroots, PETSC_NULL, PETSC_NULL, PETSC_NULL);CHKERRQ(ierr);
  if (nroots >= 0) {
    if (nroots > pEnd - pStart) {
      PetscInt *tmpDof;
      /* Help Jed: HAVE TO MAKE A BUFFER HERE THE SIZE OF THE COMPLETE SPACE AND THEN COPY INTO THE atlasDof FOR THIS SECTION */
      ierr = PetscMalloc(nroots * sizeof(PetscInt), &tmpDof);CHKERRQ(ierr);
      ierr = PetscSFBcastBegin(sf, MPIU_INT, &neg[-pStart], tmpDof);CHKERRQ(ierr);
      ierr = PetscSFBcastEnd(sf, MPIU_INT, &neg[-pStart], tmpDof);CHKERRQ(ierr);
      for (p = pStart; p < pEnd; ++p) {
        if (tmpDof[p] < 0) {(*gsection)->atlasDof[p-pStart] = tmpDof[p];}
      }
      ierr = PetscFree(tmpDof);CHKERRQ(ierr);
    } else {
      ierr = PetscSFBcastBegin(sf, MPIU_INT, &neg[-pStart], &(*gsection)->atlasDof[-pStart]);CHKERRQ(ierr);
      ierr = PetscSFBcastEnd(sf, MPIU_INT, &neg[-pStart], &(*gsection)->atlasDof[-pStart]);CHKERRQ(ierr);
    }
  }
  /* Calculate new sizes, get proccess offset, and calculate point offsets */
  for (p = 0, off = 0; p < pEnd-pStart; ++p) {
    cdof = (!includeConstraints && s->bc) ? s->bc->atlasDof[p] : 0;
    (*gsection)->atlasOff[p] = off;
    off += (*gsection)->atlasDof[p] > 0 ? (*gsection)->atlasDof[p]-cdof : 0;
  }
  ierr = MPI_Scan(&off, &globalOff, 1, MPIU_INT, MPI_SUM, s->atlasLayout.comm);CHKERRQ(ierr);
  globalOff -= off;
  for (p = 0, off = 0; p < pEnd-pStart; ++p) {
    (*gsection)->atlasOff[p] += globalOff;
    neg[p] = -((*gsection)->atlasOff[p]+1);
  }
  /* Put in negative offsets for ghost points */
  if (nroots >= 0) {
    if (nroots > pEnd - pStart) {
      PetscInt *tmpOff;
      /* Help Jed: HAVE TO MAKE A BUFFER HERE THE SIZE OF THE COMPLETE SPACE AND THEN COPY INTO THE atlasDof FOR THIS SECTION */
      ierr = PetscMalloc(nroots * sizeof(PetscInt), &tmpOff);CHKERRQ(ierr);
      ierr = PetscSFBcastBegin(sf, MPIU_INT, &neg[-pStart], tmpOff);CHKERRQ(ierr);
      ierr = PetscSFBcastEnd(sf, MPIU_INT, &neg[-pStart], tmpOff);CHKERRQ(ierr);
      for (p = pStart; p < pEnd; ++p) {
        if (tmpOff[p] < 0) {(*gsection)->atlasOff[p-pStart] = tmpOff[p];}
      }
      ierr = PetscFree(tmpOff);CHKERRQ(ierr);
    } else {
      ierr = PetscSFBcastBegin(sf, MPIU_INT, &neg[-pStart], &(*gsection)->atlasOff[-pStart]);CHKERRQ(ierr);
      ierr = PetscSFBcastEnd(sf, MPIU_INT, &neg[-pStart], &(*gsection)->atlasOff[-pStart]);CHKERRQ(ierr);
    }
  }
  ierr = PetscFree(neg);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSectionGetPointLayout"
PetscErrorCode PetscSectionGetPointLayout(MPI_Comm comm, PetscSection s, PetscLayout *layout)
{
  PetscInt       pStart, pEnd, p, localSize = 0;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscSectionGetChart(s, &pStart, &pEnd);CHKERRQ(ierr);
  for (p = pStart; p < pEnd; ++p) {
    PetscInt dof;

    ierr = PetscSectionGetDof(s, p, &dof);CHKERRQ(ierr);
    if (dof > 0) {++localSize;}
  }
  ierr = PetscLayoutCreate(comm, layout);CHKERRQ(ierr);
  ierr = PetscLayoutSetLocalSize(*layout, localSize);CHKERRQ(ierr);
  ierr = PetscLayoutSetBlockSize(*layout, 1);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(*layout);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSectionGetValueLayout"
PetscErrorCode PetscSectionGetValueLayout(MPI_Comm comm, PetscSection s, PetscLayout *layout)
{
  PetscInt       pStart, pEnd, p, localSize = 0;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscSectionGetChart(s, &pStart, &pEnd);CHKERRQ(ierr);
  for (p = pStart; p < pEnd; ++p) {
    PetscInt dof;

    ierr = PetscSectionGetDof(s, p, &dof);CHKERRQ(ierr);
    if (dof > 0) {localSize += dof;}
  }
  ierr = PetscLayoutCreate(comm, layout);CHKERRQ(ierr);
  ierr = PetscLayoutSetLocalSize(*layout, localSize);CHKERRQ(ierr);
  ierr = PetscLayoutSetBlockSize(*layout, 1);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(*layout);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSectionGetOffset"
/*@
  PetscSectionGetOffset - Return the offset into an array or local Vec for the dof associated with the given point.

  Not collective

  Input Parameters:
+ s - the PetscSection
- point - the point

  Output Parameter:
. offset - the offset

  Level: intermediate

.seealso: PetscSectionGetFieldOffset(), PetscSectionCreate()
@*/
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
#define __FUNCT__ "PetscSectionSetOffset"
/*@
  PetscSectionSetOffset - Set the offset into an array or local Vec for the dof associated with the given point.

  Not collective

  Input Parameters:
+ s - the PetscSection
. point - the point
- offset - the offset

  Note: The user usually does not call this function, but uses PetscSectionSetUp()

  Level: intermediate

.seealso: PetscSectionGetFieldOffset(), PetscSectionCreate(), PetscSectionSetUp()
@*/
PetscErrorCode PetscSectionSetOffset(PetscSection s, PetscInt point, PetscInt offset)
{
  PetscFunctionBegin;
  if ((point < s->atlasLayout.pStart) || (point >= s->atlasLayout.pEnd)) {
    SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Section point %d should be in [%d, %d)", point, s->atlasLayout.pStart, s->atlasLayout.pEnd);
  }
  s->atlasOff[point - s->atlasLayout.pStart] = offset;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSectionGetFieldOffset"
/*@
  PetscSectionGetFieldOffset - Return the offset into an array or local Vec for the dof associated with the given point.

  Not collective

  Input Parameters:
+ s - the PetscSection
. point - the point
- field - the field

  Output Parameter:
. offset - the offset

  Level: intermediate

.seealso: PetscSectionGetOffset(), PetscSectionCreate()
@*/
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
#define __FUNCT__ "PetscSectionSetFieldOffset"
/*@
  PetscSectionSetFieldOffset - Set the offset into an array or local Vec for the dof associated with the given point.

  Not collective

  Input Parameters:
+ s - the PetscSection
. point - the point
. field - the field
- offset - the offset

  Note: The user usually does not call this function, but uses PetscSectionSetUp()

  Level: intermediate

.seealso: PetscSectionGetOffset(), PetscSectionCreate(), PetscSectionSetUp()
@*/
PetscErrorCode PetscSectionSetFieldOffset(PetscSection s, PetscInt point, PetscInt field, PetscInt offset)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if ((field < 0) || (field >= s->numFields)) {
    SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Section field %d should be in [%d, %d)", field, 0, s->numFields);
  }
  ierr = PetscSectionSetOffset(s->field[field], point, offset);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSectionGetFieldPointOffset"
/* This gives the offset on a point of the field, ignoring constraints */
PetscErrorCode PetscSectionGetFieldPointOffset(PetscSection s, PetscInt point, PetscInt field, PetscInt *offset)
{
  PetscInt       off, foff;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if ((field < 0) || (field >= s->numFields)) {
    SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Section field %d should be in [%d, %d)", field, 0, s->numFields);
  }
  ierr = PetscSectionGetOffset(s, point, &off);CHKERRQ(ierr);
  ierr = PetscSectionGetOffset(s->field[field], point, &foff);CHKERRQ(ierr);
  *offset = foff - off;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSectionGetOffsetRange"
/*@
  PetscSectionGetOffsetRange - Return the full range of offsets [start, end)

  Not collective

  Input Parameter:
. s - the PetscSection

  Output Parameters:
+ start - the minimum offset
- end   - one more than the maximum offset

  Level: intermediate

.seealso: PetscSectionGetOffset(), PetscSectionCreate()
@*/
PetscErrorCode PetscSectionGetOffsetRange(PetscSection s, PetscInt *start, PetscInt *end)
{
  PetscInt       os = s->atlasOff[0], oe = s->atlasOff[0], pStart, pEnd, p;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscSectionGetChart(s, &pStart, &pEnd);CHKERRQ(ierr);
  for(p = 0; p < pEnd-pStart; ++p) {
    PetscInt dof = s->atlasDof[p], off = s->atlasOff[p];

    if (off >= 0) {
      os = PetscMin(os, off);
      oe = PetscMax(oe, off+dof);
    }
  }
  if (start) *start = os;
  if (end)   *end   = oe;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSectionCreateSubsection"
PetscErrorCode PetscSectionCreateSubsection(PetscSection s, PetscInt numFields, PetscInt fields[], PetscSection *subs)
{
  PetscInt       nF, f, pStart, pEnd, p, maxCdof = 0;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!numFields) PetscFunctionReturn(0);
  ierr = PetscSectionGetNumFields(s, &nF);CHKERRQ(ierr);
  if (numFields > nF) SETERRQ2(s->atlasLayout.comm, PETSC_ERR_ARG_WRONG, "Number of requested fields %d greater than number of fields %d", numFields, nF);
  ierr = PetscSectionCreate(s->atlasLayout.comm, subs);CHKERRQ(ierr);
  ierr = PetscSectionSetNumFields(*subs, numFields);CHKERRQ(ierr);
  for (f = 0; f < numFields; ++f) {
    const char *name;
    PetscInt    numComp;

    ierr = PetscSectionGetFieldName(s, fields[f], &name);CHKERRQ(ierr);
    ierr = PetscSectionSetFieldName(*subs, f, name);CHKERRQ(ierr);
    ierr = PetscSectionGetFieldComponents(s, fields[f], &numComp);CHKERRQ(ierr);
    ierr = PetscSectionSetFieldComponents(*subs, f, numComp);CHKERRQ(ierr);
  }
  ierr = PetscSectionGetChart(s, &pStart, &pEnd);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(*subs, pStart, pEnd);CHKERRQ(ierr);
  for (p = pStart; p < pEnd; ++p) {
    PetscInt dof = 0, cdof = 0, fdof, cfdof;

    for (f = 0; f < numFields; ++f) {
      ierr = PetscSectionGetFieldDof(s, p, fields[f], &fdof);CHKERRQ(ierr);
      ierr = PetscSectionSetFieldDof(*subs, p, f, fdof);CHKERRQ(ierr);
      ierr = PetscSectionGetFieldConstraintDof(s, p, fields[f], &cfdof);CHKERRQ(ierr);
      if (cfdof) {ierr = PetscSectionSetFieldConstraintDof(*subs, p, f, cfdof);CHKERRQ(ierr);}
      dof  += fdof;
      cdof += cfdof;
    }
    ierr = PetscSectionSetDof(*subs, p, dof);CHKERRQ(ierr);
    if (cdof) {ierr = PetscSectionSetConstraintDof(*subs, p, cdof);CHKERRQ(ierr);}
    maxCdof = PetscMax(cdof, maxCdof);
  }
  ierr = PetscSectionSetUp(*subs);CHKERRQ(ierr);
  if (maxCdof) {
    PetscInt *indices;

    ierr = PetscMalloc(maxCdof * sizeof(PetscInt), &indices);CHKERRQ(ierr);
    for (p = pStart; p < pEnd; ++p) {
      PetscInt cdof;

      ierr = PetscSectionGetConstraintDof(*subs, p, &cdof);CHKERRQ(ierr);
      if (cdof) {
        const PetscInt *oldIndices;
        PetscInt        fdof, cfdof, fc, numConst = 0, fOff = 0;

        for (f = 0; f < numFields; ++f) {
          PetscInt oldFoff = 0, oldf;

          ierr = PetscSectionGetFieldDof(s, p, fields[f], &fdof);CHKERRQ(ierr);
          ierr = PetscSectionGetFieldConstraintDof(s, p, fields[f], &cfdof);CHKERRQ(ierr);
          ierr = PetscSectionGetFieldConstraintIndices(s, p, fields[f], &oldIndices);CHKERRQ(ierr);
          /* This can be sped up if we assume sorted fields */
          for (oldf = 0; oldf < fields[f]; ++oldf) {
            PetscInt oldfdof;
            ierr = PetscSectionGetFieldDof(s, p, oldf, &oldfdof);CHKERRQ(ierr);
            oldFoff += oldfdof;
          }
          for (fc = 0; fc < cfdof; ++fc) {
            indices[numConst+fc] = oldIndices[fc] + (fOff - oldFoff);
          }
          ierr = PetscSectionSetFieldConstraintIndices(*subs, p, f, &indices[numConst]);CHKERRQ(ierr);
          numConst += cfdof;
          fOff     += fdof;
        }
        ierr = PetscSectionSetConstraintIndices(*subs, p, indices);CHKERRQ(ierr);
      }
    }
    ierr = PetscFree(indices);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSectionCreateSubmeshSection"
PetscErrorCode PetscSectionCreateSubmeshSection(PetscSection s, IS subpointMap, PetscSection *subs)
{
  const PetscInt *points, *indices;
  PetscInt        numFields, f, numSubpoints, pStart, pEnd, p, subp;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscSectionGetNumFields(s, &numFields);CHKERRQ(ierr);
  ierr = PetscSectionCreate(s->atlasLayout.comm, subs);CHKERRQ(ierr);
  if (numFields) {ierr = PetscSectionSetNumFields(*subs, numFields);CHKERRQ(ierr);}
  for(f = 0; f < numFields; ++f) {
    const char *name;
    PetscInt    numComp;

    ierr = PetscSectionGetFieldName(s, f, &name);CHKERRQ(ierr);
    ierr = PetscSectionSetFieldName(*subs, f, name);CHKERRQ(ierr);
    ierr = PetscSectionGetFieldComponents(s, f, &numComp);CHKERRQ(ierr);
    ierr = PetscSectionSetFieldComponents(*subs, f, numComp);CHKERRQ(ierr);
  }
  /* For right now, we do not try to squeeze the subchart */
  ierr = ISGetSize(subpointMap, &numSubpoints);CHKERRQ(ierr);
  ierr = ISGetIndices(subpointMap, &points);CHKERRQ(ierr);
  ierr = PetscSectionGetChart(s, &pStart, &pEnd);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(*subs, 0, numSubpoints);CHKERRQ(ierr);
  for(p = pStart; p < pEnd; ++p) {
    PetscInt dof, cdof, fdof, cfdof;

    ierr = PetscFindInt(p, numSubpoints, points, &subp);CHKERRQ(ierr);
    if (subp < 0) continue;
    for(f = 0; f < numFields; ++f) {
      ierr = PetscSectionGetFieldDof(s, p, f, &fdof);CHKERRQ(ierr);
      ierr = PetscSectionSetFieldDof(*subs, subp, f, fdof);CHKERRQ(ierr);
      ierr = PetscSectionGetFieldConstraintDof(s, p, f, &cfdof);CHKERRQ(ierr);
      if (cfdof) {ierr = PetscSectionSetFieldConstraintDof(*subs, subp, f, cfdof);CHKERRQ(ierr);}
    }
    ierr = PetscSectionGetDof(s, p, &dof);CHKERRQ(ierr);
    ierr = PetscSectionSetDof(*subs, subp, dof);CHKERRQ(ierr);
    ierr = PetscSectionGetConstraintDof(s, p, &cdof);CHKERRQ(ierr);
    if (cdof) {ierr = PetscSectionSetConstraintDof(*subs, subp, cdof);CHKERRQ(ierr);}
  }
  ierr = PetscSectionSetUp(*subs);CHKERRQ(ierr);
  /* Change offsets to original offsets */
  for(p = pStart; p < pEnd; ++p) {
    PetscInt off, foff;

    ierr = PetscFindInt(p, numSubpoints, points, &subp);CHKERRQ(ierr);
    if (subp < 0) continue;
    for(f = 0; f < numFields; ++f) {
      ierr = PetscSectionGetFieldOffset(s, p, f, &foff);CHKERRQ(ierr);
      ierr = PetscSectionSetFieldOffset(*subs, subp, f, foff);CHKERRQ(ierr);
    }
    ierr = PetscSectionGetOffset(s, p, &off);CHKERRQ(ierr);
    ierr = PetscSectionSetOffset(*subs, subp, off);CHKERRQ(ierr);
  }
  /* Copy constraint indices */
  for(subp = 0; subp < numSubpoints; ++subp) {
    PetscInt cdof;

    ierr = PetscSectionGetConstraintDof(*subs, subp, &cdof);CHKERRQ(ierr);
    if (cdof) {
      for(f = 0; f < numFields; ++f) {
        ierr = PetscSectionGetFieldConstraintIndices(s, points[subp], f, &indices);CHKERRQ(ierr);
        ierr = PetscSectionSetFieldConstraintIndices(*subs, subp, f, indices);CHKERRQ(ierr);
      }
      ierr = PetscSectionGetConstraintIndices(s, points[subp], &indices);CHKERRQ(ierr);
      ierr = PetscSectionSetConstraintIndices(*subs, subp, indices);CHKERRQ(ierr);
    }
  }
  ierr = ISRestoreIndices(subpointMap, &points);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSectionView_ASCII"
PetscErrorCode  PetscSectionView_ASCII(PetscSection s, PetscViewer viewer)
{
  PetscInt       p;
  PetscMPIInt    rank;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (s->atlasLayout.numDof != 1) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_SUP, "Cannot handle %d dof in a uniform section", s->atlasLayout.numDof);
  ierr = MPI_Comm_rank(((PetscObject) viewer)->comm, &rank);CHKERRQ(ierr);
  ierr = PetscViewerASCIISynchronizedAllow(viewer, PETSC_TRUE);CHKERRQ(ierr);
  ierr = PetscViewerASCIISynchronizedPrintf(viewer, "Process %d:\n", rank);CHKERRQ(ierr);
  for (p = 0; p < s->atlasLayout.pEnd - s->atlasLayout.pStart; ++p) {
    if ((s->bc) && (s->bc->atlasDof[p] > 0)) {
      PetscInt b;

      ierr = PetscViewerASCIISynchronizedPrintf(viewer, "  (%4d) dim %2d offset %3d constrained", p+s->atlasLayout.pStart, s->atlasDof[p], s->atlasOff[p]);CHKERRQ(ierr);
      for (b = 0; b < s->bc->atlasDof[p]; ++b) {
        ierr = PetscViewerASCIISynchronizedPrintf(viewer, " %d", s->bcIndices[s->bc->atlasOff[p]+b]);CHKERRQ(ierr);
      }
      ierr = PetscViewerASCIISynchronizedPrintf(viewer, "\n");CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIISynchronizedPrintf(viewer, "  (%4d) dim %2d offset %3d\n", p+s->atlasLayout.pStart, s->atlasDof[p], s->atlasOff[p]);CHKERRQ(ierr);
    }
  }
  ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSectionView"
/*@C
  PetscSectionView - Views a PetscSection

  Collective on PetscSection

  Input Parameters:
+ s - the PetscSection object to view
- v - the viewer

  Level: developer

.seealso PetscSectionCreate(), PetscSectionDestroy()
@*/
PetscErrorCode PetscSectionView(PetscSection s, PetscViewer viewer)
{
  PetscBool      isascii;
  PetscInt       f;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!viewer) {ierr = PetscViewerASCIIGetStdout(PETSC_COMM_SELF, &viewer);CHKERRQ(ierr);}
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &isascii);CHKERRQ(ierr);
  if (isascii) {
    if (s->numFields) {
      ierr = PetscViewerASCIIPrintf(viewer, "PetscSection with %d fields\n", s->numFields);CHKERRQ(ierr);
      for (f = 0; f < s->numFields; ++f) {
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

#undef __FUNCT__
#define __FUNCT__ "PetscSectionVecView_ASCII"
PetscErrorCode PetscSectionVecView_ASCII(PetscSection s, Vec v, PetscViewer viewer)
{
  PetscScalar   *array;
  PetscInt       p, i;
  PetscMPIInt    rank;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (s->atlasLayout.numDof != 1) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_SUP, "Cannot handle %d dof in a uniform section", s->atlasLayout.numDof);
  ierr = MPI_Comm_rank(((PetscObject) viewer)->comm, &rank);CHKERRQ(ierr);
  ierr = VecGetArray(v, &array);CHKERRQ(ierr);
  ierr = PetscViewerASCIISynchronizedAllow(viewer, PETSC_TRUE);CHKERRQ(ierr);
  ierr = PetscViewerASCIISynchronizedPrintf(viewer, "Process %d:\n", rank);CHKERRQ(ierr);
  for (p = 0; p < s->atlasLayout.pEnd - s->atlasLayout.pStart; ++p) {
    if ((s->bc) && (s->bc->atlasDof[p] > 0)) {
      PetscInt b;

      ierr = PetscViewerASCIISynchronizedPrintf(viewer, "  (%4d) dim %2d offset %3d", p+s->atlasLayout.pStart, s->atlasDof[p], s->atlasOff[p]);CHKERRQ(ierr);
      for (i = s->atlasOff[p]; i < s->atlasOff[p]+s->atlasDof[p]; ++i) {
        PetscScalar v = array[i];
#if defined(PETSC_USE_COMPLEX)
        if (PetscImaginaryPart(v) > 0.0) {
          ierr = PetscViewerASCIISynchronizedPrintf(viewer," %G + %G i", PetscRealPart(v), PetscImaginaryPart(v));CHKERRQ(ierr);
        } else if (PetscImaginaryPart(v) < 0.0) {
          ierr = PetscViewerASCIISynchronizedPrintf(viewer," %G - %G i", PetscRealPart(v),-PetscImaginaryPart(v));CHKERRQ(ierr);
        } else {
          ierr = PetscViewerASCIISynchronizedPrintf(viewer, " %G", PetscRealPart(v));CHKERRQ(ierr);
        }
#else
        ierr = PetscViewerASCIISynchronizedPrintf(viewer, " %G", v);CHKERRQ(ierr);
#endif
      }
      ierr = PetscViewerASCIISynchronizedPrintf(viewer, " constrained");CHKERRQ(ierr);
      for (b = 0; b < s->bc->atlasDof[p]; ++b) {
        ierr = PetscViewerASCIISynchronizedPrintf(viewer, " %d", s->bcIndices[s->bc->atlasOff[p]+b]);CHKERRQ(ierr);
      }
      ierr = PetscViewerASCIISynchronizedPrintf(viewer, "\n");CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIISynchronizedPrintf(viewer, "  (%4d) dim %2d offset %3d", p+s->atlasLayout.pStart, s->atlasDof[p], s->atlasOff[p]);CHKERRQ(ierr);
      for (i = s->atlasOff[p]; i < s->atlasOff[p]+s->atlasDof[p]; ++i) {
        PetscScalar v = array[i];
#if defined(PETSC_USE_COMPLEX)
        if (PetscImaginaryPart(v) > 0.0) {
          ierr = PetscViewerASCIISynchronizedPrintf(viewer," %G + %G i", PetscRealPart(v), PetscImaginaryPart(v));CHKERRQ(ierr);
        } else if (PetscImaginaryPart(v) < 0.0) {
          ierr = PetscViewerASCIISynchronizedPrintf(viewer," %G - %G i", PetscRealPart(v),-PetscImaginaryPart(v));CHKERRQ(ierr);
        } else {
          ierr = PetscViewerASCIISynchronizedPrintf(viewer, " %G", PetscRealPart(v));CHKERRQ(ierr);
        }
#else
        ierr = PetscViewerASCIISynchronizedPrintf(viewer, " %G", v);CHKERRQ(ierr);
#endif
      }
      ierr = PetscViewerASCIISynchronizedPrintf(viewer, "\n");CHKERRQ(ierr);
    }
  }
  ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
  ierr = VecRestoreArray(v, &array);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSectionVecView"
PetscErrorCode PetscSectionVecView(PetscSection s, Vec v, PetscViewer viewer)
{
  PetscBool      isascii;
  PetscInt       f;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!viewer) {ierr = PetscViewerASCIIGetStdout(((PetscObject) v)->comm, &viewer);CHKERRQ(ierr);}
  PetscValidHeaderSpecific(v, VEC_CLASSID, 2);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 3);
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &isascii);CHKERRQ(ierr);
  if (isascii) {
    const char *name;

    ierr = PetscObjectGetName((PetscObject) v, &name);CHKERRQ(ierr);
    if (s->numFields) {
      ierr = PetscViewerASCIIPrintf(viewer, "%s with %d fields\n", name, s->numFields);CHKERRQ(ierr);
      for (f = 0; f < s->numFields; ++f) {
        ierr = PetscViewerASCIIPrintf(viewer, "  field %d with %d components\n", f, s->numFieldComponents[f]);CHKERRQ(ierr);
        ierr = PetscSectionVecView_ASCII(s->field[f], v, viewer);CHKERRQ(ierr);
      }
    } else {
      ierr = PetscViewerASCIIPrintf(viewer, "%s\n", name);CHKERRQ(ierr);
      ierr = PetscSectionVecView_ASCII(s, v, viewer);CHKERRQ(ierr);
    }
  } else {
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_SUP, "Viewer type %s not supported by this section object", ((PetscObject) viewer)->type_name);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSectionDestroy"
/*@
  PetscSectionDestroy - Frees a section object and frees its range if that exists.

  Not collective

  Input Parameters:
. s - the PetscSection

  Level: developer

    The PetscSection object and methods are intended to be used in the PETSc Vec and Mat implementions; it is
    recommended they not be used in user codes unless you really gain something in their use.

.seealso: PetscSection, PetscSectionCreate()
@*/
PetscErrorCode  PetscSectionDestroy(PetscSection *s)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*s) PetscFunctionReturn(0);
  if (!(*s)->refcnt--) {
    PetscInt f;

    ierr = PetscFree((*s)->numFieldComponents);CHKERRQ(ierr);
    for (f = 0; f < (*s)->numFields; ++f) {
      ierr = PetscSectionDestroy(&(*s)->field[f]);CHKERRQ(ierr);
      ierr = PetscFree((*s)->fieldNames[f]);CHKERRQ(ierr);
    }
    ierr = PetscFree((*s)->fieldNames);CHKERRQ(ierr);
    ierr = PetscFree((*s)->field);CHKERRQ(ierr);
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
PetscErrorCode VecIntGetValuesSection(PetscInt *baseArray, PetscSection s, PetscInt point, const PetscInt **values)
{
  const PetscInt p = point - s->atlasLayout.pStart;

  PetscFunctionBegin;
  *values = &baseArray[s->atlasOff[p]];
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecSetValuesSection"
/*@C
  VecSetValuesSection - Sets all the values associated with a given point, accoridng to the section, in the given Vec

  Not collective

  Input Parameters:
+ v - the Vec
. s - the organizing PetscSection
. point - the point
. values - the array of input values
- mode - the insertion mode, either ADD_VALUES or INSERT_VALUES

  Level: developer

  Note: This is similar to MatSetValuesStencil()

.seealso: PetscSection, PetscSectionCreate()
@*/
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
        for (i = 0; i < dim; ++i) {
          array[i] = values[i];
        }
      } else {
        for (i = 0; i < dim; ++i) {
          array[i] += values[i];
        }
      }
    } else {
      PetscInt offset = 0;
      PetscInt j      = -1, field, i;

      for (field = 0; field < s->numFields; ++field) {
        const PetscInt dim = s->field[field]->atlasDof[p]; /* PetscSectionGetFieldDof() */

        for (i = dim-1; i >= 0; --i) {
          array[++j] = values[i+offset];
        }
        offset += dim;
      }
    }
  } else {
    if (orientation >= 0) {
      const PetscInt  dim  = s->atlasDof[p];
      PetscInt        cInd = 0, i;
      const PetscInt *cDof;

      ierr = PetscSectionGetConstraintIndices(s, point, &cDof);CHKERRQ(ierr);
      if (doInsert) {
        for (i = 0; i < dim; ++i) {
          if ((cInd < cDim) && (i == cDof[cInd])) {
            if (doBC) {array[i] = values[i];} /* Constrained update */
            ++cInd;
            continue;
          }
          array[i] = values[i]; /* Unconstrained update */
        }
      } else {
        for (i = 0; i < dim; ++i) {
          if ((cInd < cDim) && (i == cDof[cInd])) {
            if (doBC) {array[i] += values[i];} /* Constrained update */
            ++cInd;
            continue;
          }
          array[i] += values[i]; /* Unconstrained update */
        }
      }
    } else {
      const PetscInt *cDof;
      PetscInt        offset  = 0;
      PetscInt        cOffset = 0;
      PetscInt        j       = 0, field;

      ierr = PetscSectionGetConstraintIndices(s, point, &cDof);CHKERRQ(ierr);
      for (field = 0; field < s->numFields; ++field) {
        const PetscInt  dim = s->field[field]->atlasDof[p];     /* PetscSectionGetFieldDof() */
        const PetscInt tDim = s->field[field]->bc->atlasDof[p]; /* PetscSectionGetFieldConstraintDof() */
        const PetscInt sDim = dim - tDim;
        PetscInt       cInd = 0, i ,k;

        for (i = 0, k = dim+offset-1; i < dim; ++i, ++j, --k) {
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
PetscErrorCode VecIntSetValuesSection(PetscInt *baseArray, PetscSection s, PetscInt point, const PetscInt values[], InsertMode mode)
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
        for (i = 0; i < dim; ++i) {
          array[i] = values[i];
        }
      } else {
        for (i = 0; i < dim; ++i) {
          array[i] += values[i];
        }
      }
    } else {
      PetscInt offset = 0;
      PetscInt j      = -1, field, i;

      for (field = 0; field < s->numFields; ++field) {
        const PetscInt dim = s->field[field]->atlasDof[p];

        for (i = dim-1; i >= 0; --i) {
          array[++j] = values[i+offset];
        }
        offset += dim;
      }
    }
  } else {
    if (orientation >= 0) {
      const PetscInt  dim  = s->atlasDof[p];
      PetscInt        cInd = 0, i;
      const PetscInt *cDof;

      ierr = PetscSectionGetConstraintIndices(s, point, &cDof);CHKERRQ(ierr);
      if (mode == INSERT_VALUES) {
        for (i = 0; i < dim; ++i) {
          if ((cInd < cDim) && (i == cDof[cInd])) {++cInd; continue;}
          array[i] = values[i];
        }
      } else {
        for (i = 0; i < dim; ++i) {
          if ((cInd < cDim) && (i == cDof[cInd])) {++cInd; continue;}
          array[i] += values[i];
        }
      }
    } else {
      const PetscInt *cDof;
      PetscInt        offset  = 0;
      PetscInt        cOffset = 0;
      PetscInt        j       = 0, field;

      ierr = PetscSectionGetConstraintIndices(s, point, &cDof);CHKERRQ(ierr);
      for (field = 0; field < s->numFields; ++field) {
        const PetscInt  dim = s->field[field]->atlasDof[p];     /* PetscSectionGetFieldDof() */
        const PetscInt tDim = s->field[field]->bc->atlasDof[p]; /* PetscSectionGetFieldConstraintDof() */
        const PetscInt sDim = dim - tDim;
        PetscInt       cInd = 0, i ,k;

        for (i = 0, k = dim+offset-1; i < dim; ++i, ++j, --k) {
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
PetscErrorCode PetscSectionGetConstraintIndices(PetscSection s, PetscInt point, const PetscInt **indices)
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
PetscErrorCode PetscSectionSetConstraintIndices(PetscSection s, PetscInt point, const PetscInt indices[])
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
PetscErrorCode PetscSectionGetFieldConstraintIndices(PetscSection s, PetscInt point, PetscInt field, const PetscInt **indices)
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
PetscErrorCode PetscSectionSetFieldConstraintIndices(PetscSection s, PetscInt point, PetscInt field, const PetscInt indices[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if ((field < 0) || (field >= s->numFields)) {
    SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Section field %d should be in [%d, %d)", field, 0, s->numFields);
  }
  ierr = PetscSectionSetConstraintIndices(s->field[field], point, indices);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
  I need extract and merge routines for section based on fields. This should be trivial except for updating the
  constraint indices.

  Then I need a new interface for DMCreateDecomposition that takes groups of fields and returns a real DMComplex
  that shares the mesh parts, and has the extracted section
*/

#undef __FUNCT__
#define __FUNCT__ "PetscSFConvertPartition"
PetscErrorCode PetscSFConvertPartition(PetscSF sfPart, PetscSection partSection, IS partition, ISLocalToGlobalMapping *renumbering, PetscSF *sf)
{
  MPI_Comm       comm = ((PetscObject)sfPart)->comm;
  PetscSF        sfPoints;
  PetscInt       *partSizes,*partOffsets,p,i,numParts,numMyPoints,numPoints,count;
  const PetscInt *partArray;
  PetscSFNode    *sendPoints;
  PetscMPIInt    rank;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);

  /* Get the number of parts and sizes that I have to distribute */
  ierr = PetscSFGetGraph(sfPart,PETSC_NULL,&numParts,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscMalloc2(numParts,PetscInt,&partSizes,numParts,PetscInt,&partOffsets);CHKERRQ(ierr);
  for (p=0,numPoints=0; p<numParts; p++) {
    ierr = PetscSectionGetDof(partSection, p, &partSizes[p]);CHKERRQ(ierr);
    numPoints += partSizes[p];
  }
  numMyPoints = 0;
  ierr = PetscSFFetchAndOpBegin(sfPart,MPIU_INT,&numMyPoints,partSizes,partOffsets,MPIU_SUM);CHKERRQ(ierr);
  ierr = PetscSFFetchAndOpEnd(sfPart,MPIU_INT,&numMyPoints,partSizes,partOffsets,MPIU_SUM);CHKERRQ(ierr);
  /* I will receive numMyPoints. I will send a total of numPoints, to be placed on remote procs at partOffsets */

  /* Create SF mapping locations (addressed through partition, as indexed leaves) to new owners (roots) */
  ierr = PetscMalloc(numPoints*sizeof(PetscSFNode),&sendPoints);CHKERRQ(ierr);
  for (p=0,count=0; p<numParts; p++) {
    for (i=0; i<partSizes[p]; i++) {
      sendPoints[count].rank = p;
      sendPoints[count].index = partOffsets[p]+i;
      count++;
    }
  }
  if (count != numPoints) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Count %D should equal numPoints=%D",count,numPoints);
  ierr = PetscFree2(partSizes,partOffsets);CHKERRQ(ierr);
  ierr = ISGetIndices(partition,&partArray);CHKERRQ(ierr);
  ierr = PetscSFCreate(comm,&sfPoints);CHKERRQ(ierr);
  ierr = PetscSFSetGraph(sfPoints,numMyPoints,numPoints,partArray,PETSC_USE_POINTER,sendPoints,PETSC_OWN_POINTER);CHKERRQ(ierr);

  /* Invert SF so that the new owners are leaves and the locations indexed through partition are the roots */
  ierr = PetscSFCreateInverseSF(sfPoints,sf);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&sfPoints);CHKERRQ(ierr);
  ierr = ISRestoreIndices(partition,&partArray);CHKERRQ(ierr);

  /* Create the new local-to-global mapping */
  ierr = ISLocalToGlobalMappingCreateSF(*sf,0,renumbering);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSFDistributeSection"
/*@C
  PetscSFDistributeSection - Create a new PetscSection reorganized, moving from the root to the leaves of the SF

  Collective

  Input Parameters:
+ sf - The SF
- rootSection - Section defined on root space

  Output Parameters:
+ remoteOffsets - root offsets in leaf storage, or PETSC_NULL
- leafSection - Section defined on the leaf space

  Level: intermediate

.seealso: PetscSFCreate()
@*/
PetscErrorCode PetscSFDistributeSection(PetscSF sf, PetscSection rootSection, PetscInt **remoteOffsets, PetscSection leafSection)
{
  PetscSF         embedSF;
  const PetscInt *ilocal, *indices;
  IS              selected;
  PetscInt        numFields, nleaves, rpStart, rpEnd, lpStart = PETSC_MAX_INT, lpEnd = -1, i, f;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscSectionGetNumFields(rootSection, &numFields);CHKERRQ(ierr);
  if (numFields) {ierr = PetscSectionSetNumFields(leafSection, numFields);CHKERRQ(ierr);}
  for (f = 0; f < numFields; ++f) {
    PetscInt numComp;
    ierr = PetscSectionGetFieldComponents(rootSection, f, &numComp);CHKERRQ(ierr);
    ierr = PetscSectionSetFieldComponents(leafSection, f, numComp);CHKERRQ(ierr);
  }
  ierr = PetscSectionGetChart(rootSection, &rpStart, &rpEnd);CHKERRQ(ierr);
  ierr = ISCreateStride(PETSC_COMM_SELF, rpEnd - rpStart, rpStart, 1, &selected);CHKERRQ(ierr);
  ierr = ISGetIndices(selected, &indices);CHKERRQ(ierr);
  ierr = PetscSFCreateEmbeddedSF(sf, rpEnd - rpStart, indices, &embedSF);CHKERRQ(ierr);
  ierr = ISRestoreIndices(selected, &indices);CHKERRQ(ierr);
  ierr = ISDestroy(&selected);CHKERRQ(ierr);
  ierr = PetscSFGetGraph(embedSF, PETSC_NULL, &nleaves, &ilocal, PETSC_NULL);CHKERRQ(ierr);
  if (ilocal) {
    for (i = 0; i < nleaves; ++i) {
      lpStart = PetscMin(lpStart, ilocal[i]);
      lpEnd   = PetscMax(lpEnd,   ilocal[i]);
    }
  } else {
    lpStart = 0;
    lpEnd   = nleaves;
  }
  ++lpEnd;
  ierr = PetscSectionSetChart(leafSection, lpStart, lpEnd);CHKERRQ(ierr);
  /* Could fuse these at the cost of a copy and extra allocation */
  ierr = PetscSFBcastBegin(embedSF, MPIU_INT, &rootSection->atlasDof[-rpStart], &leafSection->atlasDof[-lpStart]);CHKERRQ(ierr);
  ierr = PetscSFBcastEnd(embedSF, MPIU_INT, &rootSection->atlasDof[-rpStart], &leafSection->atlasDof[-lpStart]);CHKERRQ(ierr);
  for (f = 0; f < numFields; ++f) {
    ierr = PetscSFBcastBegin(embedSF, MPIU_INT, &rootSection->field[f]->atlasDof[-rpStart], &leafSection->field[f]->atlasDof[-lpStart]);CHKERRQ(ierr);
    ierr = PetscSFBcastEnd(embedSF, MPIU_INT, &rootSection->field[f]->atlasDof[-rpStart], &leafSection->field[f]->atlasDof[-lpStart]);CHKERRQ(ierr);
  }
  if (remoteOffsets) {
    ierr = PetscMalloc((lpEnd - lpStart) * sizeof(PetscInt), remoteOffsets);CHKERRQ(ierr);
    ierr = PetscSFBcastBegin(embedSF, MPIU_INT, &rootSection->atlasOff[-rpStart], &(*remoteOffsets)[-lpStart]);CHKERRQ(ierr);
    ierr = PetscSFBcastEnd(embedSF, MPIU_INT, &rootSection->atlasOff[-rpStart], &(*remoteOffsets)[-lpStart]);CHKERRQ(ierr);
  }
  ierr = PetscSFDestroy(&embedSF);CHKERRQ(ierr);
  ierr = PetscSectionSetUp(leafSection);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSFCreateSectionSF"
/*@C
  PetscSFCreateSectionSF - Create an expanded SF of dofs, assuming the input SF relates points

  Input Parameters:
+ sf - The SF
. rootSection - Data layout of remote points for outgoing data (this is usually the serial section), or PETSC_NULL
- remoteOffsets - Offsets for point data on remote processes (these are offsets from the root section), or PETSC_NULL

  Output Parameters:
+ leafSection - Data layout of local points for incoming data  (this is the distributed section)
- sectionSF - The new SF

  Note: Either rootSection or remoteOffsets can be specified

  Level: intermediate

.seealso: PetscSFCreate()
@*/
PetscErrorCode PetscSFCreateSectionSF(PetscSF sf, PetscSection rootSection, PetscInt remoteOffsets[], PetscSection leafSection, PetscSF *sectionSF)
{
  MPI_Comm           comm = ((PetscObject) sf)->comm;
  const PetscInt    *localPoints;
  const PetscSFNode *remotePoints;
  PetscInt           lpStart, lpEnd;
  PetscInt           numRoots, numSectionRoots, numPoints, numIndices = 0;
  PetscInt          *localIndices;
  PetscSFNode       *remoteIndices;
  PetscInt           i, ind;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PetscSFCreate(comm, sectionSF);CHKERRQ(ierr);
  ierr = PetscSectionGetChart(leafSection, &lpStart, &lpEnd);CHKERRQ(ierr);
  ierr = PetscSectionGetStorageSize(rootSection, &numSectionRoots);CHKERRQ(ierr);
  ierr = PetscSFGetGraph(sf, &numRoots, &numPoints, &localPoints, &remotePoints);CHKERRQ(ierr);
  if (numRoots < 0) {PetscFunctionReturn(0);}
  for (i = 0; i < numPoints; ++i) {
    PetscInt localPoint = localPoints ? localPoints[i] : i;
    PetscInt dof;

    if ((localPoint >= lpStart) && (localPoint < lpEnd)) {
      ierr = PetscSectionGetDof(leafSection, localPoint, &dof);CHKERRQ(ierr);
      numIndices += dof;
    }
  }
  ierr = PetscMalloc(numIndices * sizeof(PetscInt), &localIndices);CHKERRQ(ierr);
  ierr = PetscMalloc(numIndices * sizeof(PetscSFNode), &remoteIndices);CHKERRQ(ierr);
  /* Get offsets for remote data */
  if (!remoteOffsets) {
    PetscSF         embedSF;
    const PetscInt *indices;
    IS              selected;
    PetscInt        rpStart, rpEnd, isSize;

    ierr = PetscMalloc((lpEnd - lpStart) * sizeof(PetscInt), &remoteOffsets);CHKERRQ(ierr);
    ierr = PetscSectionGetChart(rootSection, &rpStart, &rpEnd);CHKERRQ(ierr);
    isSize = PetscMin(numRoots, rpEnd - rpStart);
    ierr = ISCreateStride(PETSC_COMM_SELF, isSize, rpStart, 1, &selected);CHKERRQ(ierr);
    ierr = ISGetIndices(selected, &indices);CHKERRQ(ierr);
#if 0
    ierr = PetscSFCreateEmbeddedSF(sf, isSize, indices, &embedSF);CHKERRQ(ierr);
#else
    embedSF = sf;
    ierr = PetscObjectReference((PetscObject) embedSF);CHKERRQ(ierr);
#endif
    ierr = ISRestoreIndices(selected, &indices);CHKERRQ(ierr);
    ierr = ISDestroy(&selected);CHKERRQ(ierr);
    ierr = PetscSFBcastBegin(embedSF, MPIU_INT, &rootSection->atlasOff[-rpStart], &remoteOffsets[-lpStart]);CHKERRQ(ierr);
    ierr = PetscSFBcastEnd(embedSF, MPIU_INT, &rootSection->atlasOff[-rpStart], &remoteOffsets[-lpStart]);CHKERRQ(ierr);
    ierr = PetscSFDestroy(&embedSF);CHKERRQ(ierr);
  }
  /* Create new index graph */
  for (i = 0, ind = 0; i < numPoints; ++i) {
    PetscInt localPoint = localPoints ? localPoints[i] : i;
    PetscInt rank       = remotePoints[i].rank;

    if ((localPoint >= lpStart) && (localPoint < lpEnd)) {
      PetscInt remoteOffset = remoteOffsets[localPoint-lpStart];
      PetscInt loff, dof, d;

      ierr = PetscSectionGetOffset(leafSection, localPoint, &loff);CHKERRQ(ierr);
      ierr = PetscSectionGetDof(leafSection, localPoint, &dof);CHKERRQ(ierr);
      for (d = 0; d < dof; ++d, ++ind) {
        localIndices[ind]        = loff+d;
        remoteIndices[ind].rank  = rank;
        remoteIndices[ind].index = remoteOffset+d;
      }
    }
  }
  ierr = PetscFree(remoteOffsets);CHKERRQ(ierr);
  if (numIndices != ind) SETERRQ2(comm, PETSC_ERR_PLIB, "Inconsistency in indices, %d should be %d", ind, numIndices);
  ierr = PetscSFSetGraph(*sectionSF, numSectionRoots, numIndices, localIndices, PETSC_OWN_POINTER, remoteIndices, PETSC_OWN_POINTER);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
