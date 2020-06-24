/*
   This file contains routines for basic section object implementation.
*/

#include <petsc/private/sectionimpl.h>   /*I  "petscsection.h"   I*/
#include <petscsection.h>
#include <petscsf.h>
#include <petscviewer.h>

PetscClassId PETSC_SECTION_CLASSID;

/*@
  PetscSectionCreate - Allocates PetscSection space and sets the map contents to the default.

  Collective

  Input Parameters:
+ comm - the MPI communicator
- s    - pointer to the section

  Level: beginner

  Notes:
  Typical calling sequence
$       PetscSectionCreate(MPI_Comm,PetscSection *);
$       PetscSectionSetNumFields(PetscSection, numFields);
$       PetscSectionSetChart(PetscSection,low,high);
$       PetscSectionSetDof(PetscSection,point,numdof);
$       PetscSectionSetUp(PetscSection);
$       PetscSectionGetOffset(PetscSection,point,PetscInt *);
$       PetscSectionDestroy(PetscSection);

  The PetscSection object and methods are intended to be used in the PETSc Vec and Mat implementions; it is
  recommended they not be used in user codes unless you really gain something in their use.

.seealso: PetscSection, PetscSectionDestroy()
@*/
PetscErrorCode PetscSectionCreate(MPI_Comm comm, PetscSection *s)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(s,2);
  ierr = ISInitializePackage();CHKERRQ(ierr);

  ierr = PetscHeaderCreate(*s,PETSC_SECTION_CLASSID,"PetscSection","Section","IS",comm,PetscSectionDestroy,PetscSectionView);CHKERRQ(ierr);

  (*s)->pStart             = -1;
  (*s)->pEnd               = -1;
  (*s)->perm               = NULL;
  (*s)->pointMajor         = PETSC_TRUE;
  (*s)->maxDof             = 0;
  (*s)->atlasDof           = NULL;
  (*s)->atlasOff           = NULL;
  (*s)->bc                 = NULL;
  (*s)->bcIndices          = NULL;
  (*s)->setup              = PETSC_FALSE;
  (*s)->numFields          = 0;
  (*s)->fieldNames         = NULL;
  (*s)->field              = NULL;
  (*s)->useFieldOff        = PETSC_FALSE;
  (*s)->compNames          = NULL;
  (*s)->clObj              = NULL;
  (*s)->clHash             = NULL;
  (*s)->clSection          = NULL;
  (*s)->clPoints           = NULL;
  PetscFunctionReturn(0);
}

/*@
  PetscSectionCopy - Creates a shallow (if possible) copy of the PetscSection

  Collective

  Input Parameter:
. section - the PetscSection

  Output Parameter:
. newSection - the copy

  Level: intermediate

.seealso: PetscSection, PetscSectionCreate(), PetscSectionDestroy()
@*/
PetscErrorCode PetscSectionCopy(PetscSection section, PetscSection newSection)
{
  PetscSectionSym sym;
  IS              perm;
  PetscInt        numFields, f, c, pStart, pEnd, p;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(section, PETSC_SECTION_CLASSID, 1);
  PetscValidHeaderSpecific(newSection, PETSC_SECTION_CLASSID, 2);
  ierr = PetscSectionReset(newSection);CHKERRQ(ierr);
  ierr = PetscSectionGetNumFields(section, &numFields);CHKERRQ(ierr);
  if (numFields) {ierr = PetscSectionSetNumFields(newSection, numFields);CHKERRQ(ierr);}
  for (f = 0; f < numFields; ++f) {
    const char *fieldName = NULL, *compName = NULL;
    PetscInt   numComp = 0;

    ierr = PetscSectionGetFieldName(section, f, &fieldName);CHKERRQ(ierr);
    ierr = PetscSectionSetFieldName(newSection, f, fieldName);CHKERRQ(ierr);
    ierr = PetscSectionGetFieldComponents(section, f, &numComp);CHKERRQ(ierr);
    ierr = PetscSectionSetFieldComponents(newSection, f, numComp);CHKERRQ(ierr);
    for (c = 0; c < numComp; ++c) {
      ierr = PetscSectionGetComponentName(section, f, c, &compName);CHKERRQ(ierr);
      ierr = PetscSectionSetComponentName(newSection, f, c, compName);CHKERRQ(ierr);
    }
    ierr = PetscSectionGetFieldSym(section, f, &sym);CHKERRQ(ierr);
    ierr = PetscSectionSetFieldSym(newSection, f, sym);CHKERRQ(ierr);
  }
  ierr = PetscSectionGetChart(section, &pStart, &pEnd);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(newSection, pStart, pEnd);CHKERRQ(ierr);
  ierr = PetscSectionGetPermutation(section, &perm);CHKERRQ(ierr);
  ierr = PetscSectionSetPermutation(newSection, perm);CHKERRQ(ierr);
  ierr = PetscSectionGetSym(section, &sym);CHKERRQ(ierr);
  ierr = PetscSectionSetSym(newSection, sym);CHKERRQ(ierr);
  for (p = pStart; p < pEnd; ++p) {
    PetscInt dof, cdof, fcdof = 0;

    ierr = PetscSectionGetDof(section, p, &dof);CHKERRQ(ierr);
    ierr = PetscSectionSetDof(newSection, p, dof);CHKERRQ(ierr);
    ierr = PetscSectionGetConstraintDof(section, p, &cdof);CHKERRQ(ierr);
    if (cdof) {ierr = PetscSectionSetConstraintDof(newSection, p, cdof);CHKERRQ(ierr);}
    for (f = 0; f < numFields; ++f) {
      ierr = PetscSectionGetFieldDof(section, p, f, &dof);CHKERRQ(ierr);
      ierr = PetscSectionSetFieldDof(newSection, p, f, dof);CHKERRQ(ierr);
      if (cdof) {
        ierr = PetscSectionGetFieldConstraintDof(section, p, f, &fcdof);CHKERRQ(ierr);
        if (fcdof) {ierr = PetscSectionSetFieldConstraintDof(newSection, p, f, fcdof);CHKERRQ(ierr);}
      }
    }
  }
  ierr = PetscSectionSetUp(newSection);CHKERRQ(ierr);
  for (p = pStart; p < pEnd; ++p) {
    PetscInt       off, cdof, fcdof = 0;
    const PetscInt *cInd;

    /* Must set offsets in case they do not agree with the prefix sums */
    ierr = PetscSectionGetOffset(section, p, &off);CHKERRQ(ierr);
    ierr = PetscSectionSetOffset(newSection, p, off);CHKERRQ(ierr);
    ierr = PetscSectionGetConstraintDof(section, p, &cdof);CHKERRQ(ierr);
    if (cdof) {
      ierr = PetscSectionGetConstraintIndices(section, p, &cInd);CHKERRQ(ierr);
      ierr = PetscSectionSetConstraintIndices(newSection, p, cInd);CHKERRQ(ierr);
      for (f = 0; f < numFields; ++f) {
        ierr = PetscSectionGetFieldConstraintDof(section, p, f, &fcdof);CHKERRQ(ierr);
        if (fcdof) {
          ierr = PetscSectionGetFieldConstraintIndices(section, p, f, &cInd);CHKERRQ(ierr);
          ierr = PetscSectionSetFieldConstraintIndices(newSection, p, f, cInd);CHKERRQ(ierr);
        }
      }
    }
  }
  PetscFunctionReturn(0);
}

/*@
  PetscSectionClone - Creates a shallow (if possible) copy of the PetscSection

  Collective

  Input Parameter:
. section - the PetscSection

  Output Parameter:
. newSection - the copy

  Level: beginner

.seealso: PetscSection, PetscSectionCreate(), PetscSectionDestroy()
@*/
PetscErrorCode PetscSectionClone(PetscSection section, PetscSection *newSection)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(section, PETSC_SECTION_CLASSID, 1);
  PetscValidPointer(newSection, 2);
  ierr = PetscSectionCreate(PetscObjectComm((PetscObject) section), newSection);CHKERRQ(ierr);
  ierr = PetscSectionCopy(section, *newSection);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  PetscSectionSetFromOptions - sets parameters in a PetscSection from the options database

  Collective on PetscSection

  Input Parameter:
. section - the PetscSection

  Options Database:
. -petscsection_point_major the dof order

  Level: intermediate

.seealso: PetscSection, PetscSectionCreate(), PetscSectionDestroy()
@*/
PetscErrorCode PetscSectionSetFromOptions(PetscSection s)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(s, PETSC_SECTION_CLASSID, 1);
  ierr = PetscObjectOptionsBegin((PetscObject) s);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-petscsection_point_major", "The for ordering, either point major or field major", "PetscSectionSetPointMajor", s->pointMajor, &s->pointMajor, NULL);CHKERRQ(ierr);
  /* process any options handlers added with PetscObjectAddOptionsHandler() */
  ierr = PetscObjectProcessOptionsHandlers(PetscOptionsObject,(PetscObject) s);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  ierr = PetscObjectViewFromOptions((PetscObject) s, NULL, "-petscsection_view");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  PetscSectionCompare - Compares two sections

  Collective on PetscSection

  Input Parameters:
+ s1 - the first PetscSection
- s2 - the second PetscSection

  Output Parameter:
. congruent - PETSC_TRUE if the two sections are congruent, PETSC_FALSE otherwise

  Level: intermediate

  Notes:
  Field names are disregarded.

.seealso: PetscSection, PetscSectionCreate(), PetscSectionCopy(), PetscSectionClone()
@*/
PetscErrorCode PetscSectionCompare(PetscSection s1, PetscSection s2, PetscBool *congruent)
{
  PetscInt pStart, pEnd, nfields, ncdof, nfcdof, p, f, n1, n2;
  const PetscInt *idx1, *idx2;
  IS perm1, perm2;
  PetscBool flg;
  PetscMPIInt mflg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(s1, PETSC_SECTION_CLASSID, 1);
  PetscValidHeaderSpecific(s2, PETSC_SECTION_CLASSID, 2);
  PetscValidIntPointer(congruent,3);
  flg = PETSC_FALSE;

  ierr = MPI_Comm_compare(PetscObjectComm((PetscObject)s1),PetscObjectComm((PetscObject)s2),&mflg);CHKERRQ(ierr);
  if (mflg != MPI_CONGRUENT && mflg != MPI_IDENT) {
    *congruent = PETSC_FALSE;
    PetscFunctionReturn(0);
  }

  ierr = PetscSectionGetChart(s1, &pStart, &pEnd);CHKERRQ(ierr);
  ierr = PetscSectionGetChart(s2, &n1, &n2);CHKERRQ(ierr);
  if (pStart != n1 || pEnd != n2) goto not_congruent;

  ierr = PetscSectionGetPermutation(s1, &perm1);CHKERRQ(ierr);
  ierr = PetscSectionGetPermutation(s2, &perm2);CHKERRQ(ierr);
  if (perm1 && perm2) {
    ierr = ISEqual(perm1, perm2, congruent);CHKERRQ(ierr);
    if (!(*congruent)) goto not_congruent;
  } else if (perm1 != perm2) goto not_congruent;

  for (p = pStart; p < pEnd; ++p) {
    ierr = PetscSectionGetOffset(s1, p, &n1);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(s2, p, &n2);CHKERRQ(ierr);
    if (n1 != n2) goto not_congruent;

    ierr = PetscSectionGetDof(s1, p, &n1);CHKERRQ(ierr);
    ierr = PetscSectionGetDof(s2, p, &n2);CHKERRQ(ierr);
    if (n1 != n2) goto not_congruent;

    ierr = PetscSectionGetConstraintDof(s1, p, &ncdof);CHKERRQ(ierr);
    ierr = PetscSectionGetConstraintDof(s2, p, &n2);CHKERRQ(ierr);
    if (ncdof != n2) goto not_congruent;

    ierr = PetscSectionGetConstraintIndices(s1, p, &idx1);CHKERRQ(ierr);
    ierr = PetscSectionGetConstraintIndices(s2, p, &idx2);CHKERRQ(ierr);
    ierr = PetscArraycmp(idx1, idx2, ncdof, congruent);CHKERRQ(ierr);
    if (!(*congruent)) goto not_congruent;
  }

  ierr = PetscSectionGetNumFields(s1, &nfields);CHKERRQ(ierr);
  ierr = PetscSectionGetNumFields(s2, &n2);CHKERRQ(ierr);
  if (nfields != n2) goto not_congruent;

  for (f = 0; f < nfields; ++f) {
    ierr = PetscSectionGetFieldComponents(s1, f, &n1);CHKERRQ(ierr);
    ierr = PetscSectionGetFieldComponents(s2, f, &n2);CHKERRQ(ierr);
    if (n1 != n2) goto not_congruent;

    for (p = pStart; p < pEnd; ++p) {
      ierr = PetscSectionGetFieldOffset(s1, p, f, &n1);CHKERRQ(ierr);
      ierr = PetscSectionGetFieldOffset(s2, p, f, &n2);CHKERRQ(ierr);
      if (n1 != n2) goto not_congruent;

      ierr = PetscSectionGetFieldDof(s1, p, f, &n1);CHKERRQ(ierr);
      ierr = PetscSectionGetFieldDof(s2, p, f, &n2);CHKERRQ(ierr);
      if (n1 != n2) goto not_congruent;

      ierr = PetscSectionGetFieldConstraintDof(s1, p, f, &nfcdof);CHKERRQ(ierr);
      ierr = PetscSectionGetFieldConstraintDof(s2, p, f, &n2);CHKERRQ(ierr);
      if (nfcdof != n2) goto not_congruent;

      ierr = PetscSectionGetFieldConstraintIndices(s1, p, f, &idx1);CHKERRQ(ierr);
      ierr = PetscSectionGetFieldConstraintIndices(s2, p, f, &idx2);CHKERRQ(ierr);
      ierr = PetscArraycmp(idx1, idx2, nfcdof, congruent);CHKERRQ(ierr);
      if (!(*congruent)) goto not_congruent;
    }
  }

  flg = PETSC_TRUE;
not_congruent:
  ierr = MPIU_Allreduce(&flg,congruent,1,MPIU_BOOL,MPI_LAND,PetscObjectComm((PetscObject)s1));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

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
  PetscValidHeaderSpecific(s, PETSC_SECTION_CLASSID, 1);
  PetscValidPointer(numFields,2);
  *numFields = s->numFields;
  PetscFunctionReturn(0);
}

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
  PetscValidHeaderSpecific(s, PETSC_SECTION_CLASSID, 1);
  if (numFields <= 0) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "The number of fields %D must be positive", numFields);
  ierr = PetscSectionReset(s);CHKERRQ(ierr);

  s->numFields = numFields;
  ierr = PetscMalloc1(s->numFields, &s->numFieldComponents);CHKERRQ(ierr);
  ierr = PetscMalloc1(s->numFields, &s->fieldNames);CHKERRQ(ierr);
  ierr = PetscMalloc1(s->numFields, &s->compNames);CHKERRQ(ierr);
  ierr = PetscMalloc1(s->numFields, &s->field);CHKERRQ(ierr);
  for (f = 0; f < s->numFields; ++f) {
    char name[64];

    s->numFieldComponents[f] = 1;

    ierr = PetscSectionCreate(PetscObjectComm((PetscObject) s), &s->field[f]);CHKERRQ(ierr);
    ierr = PetscSNPrintf(name, 64, "Field_%D", f);CHKERRQ(ierr);
    ierr = PetscStrallocpy(name, (char **) &s->fieldNames[f]);CHKERRQ(ierr);
    ierr = PetscSNPrintf(name, 64, "Component_0");CHKERRQ(ierr);
    ierr = PetscMalloc1(s->numFieldComponents[f], &s->compNames[f]);CHKERRQ(ierr);
    ierr = PetscStrallocpy(name, (char **) &s->compNames[f][0]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@C
  PetscSectionGetFieldName - Returns the name of a field in the PetscSection

  Not Collective

  Input Parameters:
+ s     - the PetscSection
- field - the field number

  Output Parameter:
. fieldName - the field name

  Level: intermediate

.seealso: PetscSectionSetFieldName()
@*/
PetscErrorCode PetscSectionGetFieldName(PetscSection s, PetscInt field, const char *fieldName[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(s, PETSC_SECTION_CLASSID, 1);
  PetscValidPointer(fieldName, 3);
  if ((field < 0) || (field >= s->numFields)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Section field %D should be in [%D, %D)", field, 0, s->numFields);
  *fieldName = s->fieldNames[field];
  PetscFunctionReturn(0);
}

/*@C
  PetscSectionSetFieldName - Sets the name of a field in the PetscSection

  Not Collective

  Input Parameters:
+ s     - the PetscSection
. field - the field number
- fieldName - the field name

  Level: intermediate

.seealso: PetscSectionGetFieldName()
@*/
PetscErrorCode PetscSectionSetFieldName(PetscSection s, PetscInt field, const char fieldName[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(s, PETSC_SECTION_CLASSID, 1);
  if (fieldName) PetscValidCharPointer(fieldName, 3);
  if ((field < 0) || (field >= s->numFields)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Section field %D should be in [%D, %D)", field, 0, s->numFields);
  ierr = PetscFree(s->fieldNames[field]);CHKERRQ(ierr);
  ierr = PetscStrallocpy(fieldName, (char**) &s->fieldNames[field]);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  PetscSectionGetComponentName - Gets the name of a field component in the PetscSection

  Not Collective

  Input Parameters:
+ s     - the PetscSection
. field - the field number
. comp  - the component number
- compName - the component name

  Level: intermediate

.seealso: PetscSectionSetComponentName()
@*/
PetscErrorCode PetscSectionGetComponentName(PetscSection s, PetscInt field, PetscInt comp, const char *compName[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(s, PETSC_SECTION_CLASSID, 1);
  PetscValidPointer(compName, 3);
  if ((field < 0) || (field >= s->numFields)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Section field %D should be in [%D, %D)", field, 0, s->numFields);
  if ((comp < 0) || (comp >= s->numFieldComponents[field])) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Section field component %D should be in [%D, %D)", comp, 0, s->numFieldComponents[field]);
  *compName = s->compNames[field][comp];
  PetscFunctionReturn(0);
}

/*@C
  PetscSectionSetComponentName - Sets the name of a field component in the PetscSection

  Not Collective

  Input Parameters:
+ s     - the PetscSection
. field - the field number
. comp  - the component number
- compName - the component name

  Level: intermediate

.seealso: PetscSectionGetComponentName()
@*/
PetscErrorCode PetscSectionSetComponentName(PetscSection s, PetscInt field, PetscInt comp, const char compName[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(s, PETSC_SECTION_CLASSID, 1);
  if (compName) PetscValidCharPointer(compName, 3);
  if ((field < 0) || (field >= s->numFields)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Section field %D should be in [%D, %D)", field, 0, s->numFields);
  if ((comp < 0) || (comp >= s->numFieldComponents[field])) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Section field component %D should be in [%D, %D)", comp, 0, s->numFieldComponents[field]);
  ierr = PetscFree(s->compNames[field][comp]);CHKERRQ(ierr);
  ierr = PetscStrallocpy(compName, (char**) &s->compNames[field][comp]);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  PetscSectionGetFieldComponents - Returns the number of field components for the given field.

  Not collective

  Input Parameters:
+ s - the PetscSection
- field - the field number

  Output Parameter:
. numComp - the number of field components

  Level: intermediate

.seealso: PetscSectionSetFieldComponents(), PetscSectionGetNumFields()
@*/
PetscErrorCode PetscSectionGetFieldComponents(PetscSection s, PetscInt field, PetscInt *numComp)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(s, PETSC_SECTION_CLASSID, 1);
  PetscValidPointer(numComp, 3);
  if ((field < 0) || (field >= s->numFields)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Section field %D should be in [%D, %D)", field, 0, s->numFields);
  *numComp = s->numFieldComponents[field];
  PetscFunctionReturn(0);
}

/*@
  PetscSectionSetFieldComponents - Sets the number of field components for the given field.

  Not collective

  Input Parameters:
+ s - the PetscSection
. field - the field number
- numComp - the number of field components

  Level: intermediate

.seealso: PetscSectionGetFieldComponents(), PetscSectionGetNumFields()
@*/
PetscErrorCode PetscSectionSetFieldComponents(PetscSection s, PetscInt field, PetscInt numComp)
{
  PetscErrorCode ierr;
  PetscInt c;
  char name[64];

  PetscFunctionBegin;
  PetscValidHeaderSpecific(s, PETSC_SECTION_CLASSID, 1);
  if ((field < 0) || (field >= s->numFields)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Section field %D should be in [%D, %D)", field, 0, s->numFields);
  if (s->compNames) {
    for (c = 0; c < s->numFieldComponents[field]; ++c) {
      ierr = PetscFree(s->compNames[field][c]);CHKERRQ(ierr);
    }
    ierr = PetscFree(s->compNames[field]);CHKERRQ(ierr);
  }

  s->numFieldComponents[field] = numComp;
  if (numComp) {
    ierr = PetscMalloc1(numComp, (char ***) &s->compNames[field]);CHKERRQ(ierr);
    for (c = 0; c < numComp; ++c) {
      ierr = PetscSNPrintf(name, 64, "%D", c);CHKERRQ(ierr);
      ierr = PetscStrallocpy(name, (char **) &s->compNames[field][c]);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSectionCheckConstraints_Static(PetscSection s)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!s->bc) {
    ierr = PetscSectionCreate(PETSC_COMM_SELF, &s->bc);CHKERRQ(ierr);
    ierr = PetscSectionSetChart(s->bc, s->pStart, s->pEnd);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@
  PetscSectionGetChart - Returns the range [pStart, pEnd) in which points lie.

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
  PetscValidHeaderSpecific(s, PETSC_SECTION_CLASSID, 1);
  if (pStart) *pStart = s->pStart;
  if (pEnd)   *pEnd   = s->pEnd;
  PetscFunctionReturn(0);
}

/*@
  PetscSectionSetChart - Sets the range [pStart, pEnd) in which points lie.

  Not collective

  Input Parameters:
+ s - the PetscSection
. pStart - the first point
- pEnd - one past the last point

  Level: intermediate

.seealso: PetscSectionGetChart(), PetscSectionCreate()
@*/
PetscErrorCode PetscSectionSetChart(PetscSection s, PetscInt pStart, PetscInt pEnd)
{
  PetscInt       f;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(s, PETSC_SECTION_CLASSID, 1);
  /* Cannot Reset() because it destroys field information */
  s->setup = PETSC_FALSE;
  ierr = PetscSectionDestroy(&s->bc);CHKERRQ(ierr);
  ierr = PetscFree(s->bcIndices);CHKERRQ(ierr);
  ierr = PetscFree2(s->atlasDof, s->atlasOff);CHKERRQ(ierr);

  s->pStart = pStart;
  s->pEnd   = pEnd;
  ierr = PetscMalloc2((pEnd - pStart), &s->atlasDof, (pEnd - pStart), &s->atlasOff);CHKERRQ(ierr);
  ierr = PetscArrayzero(s->atlasDof, pEnd - pStart);CHKERRQ(ierr);
  for (f = 0; f < s->numFields; ++f) {
    ierr = PetscSectionSetChart(s->field[f], pStart, pEnd);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@
  PetscSectionGetPermutation - Returns the permutation of [0, pEnd-pStart) or NULL

  Not collective

  Input Parameter:
. s - the PetscSection

  Output Parameters:
. perm - The permutation as an IS

  Level: intermediate

.seealso: PetscSectionSetPermutation(), PetscSectionCreate()
@*/
PetscErrorCode PetscSectionGetPermutation(PetscSection s, IS *perm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(s, PETSC_SECTION_CLASSID, 1);
  if (perm) {PetscValidPointer(perm, 2); *perm = s->perm;}
  PetscFunctionReturn(0);
}

/*@
  PetscSectionSetPermutation - Sets the permutation for [0, pEnd-pStart)

  Not collective

  Input Parameters:
+ s - the PetscSection
- perm - the permutation of points

  Level: intermediate

.seealso: PetscSectionGetPermutation(), PetscSectionCreate()
@*/
PetscErrorCode PetscSectionSetPermutation(PetscSection s, IS perm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(s, PETSC_SECTION_CLASSID, 1);
  if (perm) PetscValidHeaderSpecific(perm, IS_CLASSID, 2);
  if (s->setup) SETERRQ(PetscObjectComm((PetscObject) s), PETSC_ERR_ARG_WRONGSTATE, "Cannot set a permutation after the section is setup");
  if (s->perm != perm) {
    ierr = ISDestroy(&s->perm);CHKERRQ(ierr);
    if (perm) {
      s->perm = perm;
      ierr = PetscObjectReference((PetscObject) s->perm);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

/*@
  PetscSectionGetPointMajor - Returns the flag for dof ordering, true if it is point major, otherwise field major

  Not collective

  Input Parameter:
. s - the PetscSection

  Output Parameter:
. pm - the flag for point major ordering

  Level: intermediate

.seealso: PetscSectionSetPointMajor()
@*/
PetscErrorCode PetscSectionGetPointMajor(PetscSection s, PetscBool *pm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(s, PETSC_SECTION_CLASSID, 1);
  PetscValidPointer(pm,2);
  *pm = s->pointMajor;
  PetscFunctionReturn(0);
}

/*@
  PetscSectionSetPointMajor - Sets the flag for dof ordering, true if it is point major, otherwise field major

  Not collective

  Input Parameters:
+ s  - the PetscSection
- pm - the flag for point major ordering

  Not collective

  Level: intermediate

.seealso: PetscSectionGetPointMajor()
@*/
PetscErrorCode PetscSectionSetPointMajor(PetscSection s, PetscBool pm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(s, PETSC_SECTION_CLASSID, 1);
  if (s->setup) SETERRQ(PetscObjectComm((PetscObject) s), PETSC_ERR_ARG_WRONGSTATE, "Cannot set the dof ordering after the section is setup");
  s->pointMajor = pm;
  PetscFunctionReturn(0);
}

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
  PetscFunctionBeginHot;
  PetscValidHeaderSpecific(s, PETSC_SECTION_CLASSID, 1);
  PetscValidPointer(numDof, 3);
  if (PetscDefined(USE_DEBUG)) {
    if ((point < s->pStart) || (point >= s->pEnd)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Section point %D should be in [%D, %D)", point, s->pStart, s->pEnd);
  }
  *numDof = s->atlasDof[point - s->pStart];
  PetscFunctionReturn(0);
}

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
  PetscValidHeaderSpecific(s, PETSC_SECTION_CLASSID, 1);
  if ((point < s->pStart) || (point >= s->pEnd)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Section point %D should be in [%D, %D)", point, s->pStart, s->pEnd);
  s->atlasDof[point - s->pStart] = numDof;
  PetscFunctionReturn(0);
}

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
  PetscFunctionBeginHot;
  PetscValidHeaderSpecific(s, PETSC_SECTION_CLASSID, 1);
  if (PetscDefined(USE_DEBUG)) {
    if ((point < s->pStart) || (point >= s->pEnd)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Section point %D should be in [%D, %D)", point, s->pStart, s->pEnd);
  }
  s->atlasDof[point - s->pStart] += numDof;
  PetscFunctionReturn(0);
}

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
  PetscValidHeaderSpecific(s, PETSC_SECTION_CLASSID, 1);
  if ((field < 0) || (field >= s->numFields)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Section field %D should be in [%D, %D)", field, 0, s->numFields);
  ierr = PetscSectionGetDof(s->field[field], point, numDof);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

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
  PetscValidHeaderSpecific(s, PETSC_SECTION_CLASSID, 1);
  if ((field < 0) || (field >= s->numFields)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Section field %D should be in [%D, %D)", field, 0, s->numFields);
  ierr = PetscSectionSetDof(s->field[field], point, numDof);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

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
  PetscValidHeaderSpecific(s, PETSC_SECTION_CLASSID, 1);
  if ((field < 0) || (field >= s->numFields)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Section field %D should be in [%D, %D)", field, 0, s->numFields);
  ierr = PetscSectionAddDof(s->field[field], point, numDof);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  PetscSectionGetConstraintDof - Return the number of constrained degrees of freedom associated with a given point.

  Not collective

  Input Parameters:
+ s - the PetscSection
- point - the point

  Output Parameter:
. numDof - the number of dof which are fixed by constraints

  Level: intermediate

.seealso: PetscSectionGetDof(), PetscSectionSetConstraintDof(), PetscSectionCreate()
@*/
PetscErrorCode PetscSectionGetConstraintDof(PetscSection s, PetscInt point, PetscInt *numDof)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(s, PETSC_SECTION_CLASSID, 1);
  PetscValidPointer(numDof, 3);
  if (s->bc) {
    ierr = PetscSectionGetDof(s->bc, point, numDof);CHKERRQ(ierr);
  } else *numDof = 0;
  PetscFunctionReturn(0);
}

/*@
  PetscSectionSetConstraintDof - Set the number of constrained degrees of freedom associated with a given point.

  Not collective

  Input Parameters:
+ s - the PetscSection
. point - the point
- numDof - the number of dof which are fixed by constraints

  Level: intermediate

.seealso: PetscSectionSetDof(), PetscSectionGetConstraintDof(), PetscSectionCreate()
@*/
PetscErrorCode PetscSectionSetConstraintDof(PetscSection s, PetscInt point, PetscInt numDof)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(s, PETSC_SECTION_CLASSID, 1);
  if (numDof) {
    ierr = PetscSectionCheckConstraints_Static(s);CHKERRQ(ierr);
    ierr = PetscSectionSetDof(s->bc, point, numDof);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@
  PetscSectionAddConstraintDof - Increment the number of constrained degrees of freedom associated with a given point.

  Not collective

  Input Parameters:
+ s - the PetscSection
. point - the point
- numDof - the number of additional dof which are fixed by constraints

  Level: intermediate

.seealso: PetscSectionAddDof(), PetscSectionGetConstraintDof(), PetscSectionCreate()
@*/
PetscErrorCode PetscSectionAddConstraintDof(PetscSection s, PetscInt point, PetscInt numDof)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(s, PETSC_SECTION_CLASSID, 1);
  if (numDof) {
    ierr = PetscSectionCheckConstraints_Static(s);CHKERRQ(ierr);
    ierr = PetscSectionAddDof(s->bc, point, numDof);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@
  PetscSectionGetFieldConstraintDof - Return the number of constrained degrees of freedom associated with a given field on a point.

  Not collective

  Input Parameters:
+ s - the PetscSection
. point - the point
- field - the field

  Output Parameter:
. numDof - the number of dof which are fixed by constraints

  Level: intermediate

.seealso: PetscSectionGetDof(), PetscSectionSetFieldConstraintDof(), PetscSectionCreate()
@*/
PetscErrorCode PetscSectionGetFieldConstraintDof(PetscSection s, PetscInt point, PetscInt field, PetscInt *numDof)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(s, PETSC_SECTION_CLASSID, 1);
  PetscValidPointer(numDof, 4);
  if ((field < 0) || (field >= s->numFields)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Section field %D should be in [%D, %D)", field, 0, s->numFields);
  ierr = PetscSectionGetConstraintDof(s->field[field], point, numDof);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  PetscSectionSetFieldConstraintDof - Set the number of constrained degrees of freedom associated with a given field on a point.

  Not collective

  Input Parameters:
+ s - the PetscSection
. point - the point
. field - the field
- numDof - the number of dof which are fixed by constraints

  Level: intermediate

.seealso: PetscSectionSetDof(), PetscSectionGetFieldConstraintDof(), PetscSectionCreate()
@*/
PetscErrorCode PetscSectionSetFieldConstraintDof(PetscSection s, PetscInt point, PetscInt field, PetscInt numDof)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(s, PETSC_SECTION_CLASSID, 1);
  if ((field < 0) || (field >= s->numFields)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Section field %D should be in [%D, %D)", field, 0, s->numFields);
  ierr = PetscSectionSetConstraintDof(s->field[field], point, numDof);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  PetscSectionAddFieldConstraintDof - Increment the number of constrained degrees of freedom associated with a given field on a point.

  Not collective

  Input Parameters:
+ s - the PetscSection
. point - the point
. field - the field
- numDof - the number of additional dof which are fixed by constraints

  Level: intermediate

.seealso: PetscSectionAddDof(), PetscSectionGetFieldConstraintDof(), PetscSectionCreate()
@*/
PetscErrorCode PetscSectionAddFieldConstraintDof(PetscSection s, PetscInt point, PetscInt field, PetscInt numDof)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(s, PETSC_SECTION_CLASSID, 1);
  if ((field < 0) || (field >= s->numFields)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Section field %D should be in [%D, %D)", field, 0, s->numFields);
  ierr = PetscSectionAddConstraintDof(s->field[field], point, numDof);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  PetscSectionSetUpBC - Setup the subsections describing boundary conditions.

  Not collective

  Input Parameter:
. s - the PetscSection

  Level: advanced

.seealso: PetscSectionSetUp(), PetscSectionCreate()
@*/
PetscErrorCode PetscSectionSetUpBC(PetscSection s)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(s, PETSC_SECTION_CLASSID, 1);
  if (s->bc) {
    const PetscInt last = (s->bc->pEnd-s->bc->pStart) - 1;

    ierr = PetscSectionSetUp(s->bc);CHKERRQ(ierr);
    ierr = PetscMalloc1(last >= 0 ? s->bc->atlasOff[last] + s->bc->atlasDof[last] : 0, &s->bcIndices);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

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
  const PetscInt *pind   = NULL;
  PetscInt        offset = 0, foff, p, f;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(s, PETSC_SECTION_CLASSID, 1);
  if (s->setup) PetscFunctionReturn(0);
  s->setup = PETSC_TRUE;
  /* Set offsets and field offsets for all points */
  /*   Assume that all fields have the same chart */
  if (s->perm) {ierr = ISGetIndices(s->perm, &pind);CHKERRQ(ierr);}
  if (s->pointMajor) {
    for (p = 0; p < s->pEnd - s->pStart; ++p) {
      const PetscInt q = pind ? pind[p] : p;

      /* Set point offset */
      s->atlasOff[q] = offset;
      offset        += s->atlasDof[q];
      s->maxDof      = PetscMax(s->maxDof, s->atlasDof[q]);
      /* Set field offset */
      for (f = 0, foff = s->atlasOff[q]; f < s->numFields; ++f) {
        PetscSection sf = s->field[f];

        sf->atlasOff[q] = foff;
        foff += sf->atlasDof[q];
      }
    }
  } else {
    /* Set field offsets for all points */
    for (f = 0; f < s->numFields; ++f) {
      PetscSection sf = s->field[f];

      for (p = 0; p < s->pEnd - s->pStart; ++p) {
        const PetscInt q = pind ? pind[p] : p;

        sf->atlasOff[q] = offset;
        offset += sf->atlasDof[q];
      }
    }
    /* Disable point offsets since these are unused */
    for (p = 0; p < s->pEnd - s->pStart; ++p) {
      s->atlasOff[p] = -1;
      s->maxDof      = PetscMax(s->maxDof, s->atlasDof[p]);
    }
  }
  if (s->perm) {ierr = ISRestoreIndices(s->perm, &pind);CHKERRQ(ierr);}
  /* Setup BC sections */
  ierr = PetscSectionSetUpBC(s);CHKERRQ(ierr);
  for (f = 0; f < s->numFields; ++f) {ierr = PetscSectionSetUpBC(s->field[f]);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

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
  PetscValidHeaderSpecific(s, PETSC_SECTION_CLASSID, 1);
  PetscValidPointer(maxDof, 2);
  *maxDof = s->maxDof;
  PetscFunctionReturn(0);
}

/*@
  PetscSectionGetStorageSize - Return the size of an array or local Vec capable of holding all the degrees of freedom.

  Not collective

  Input Parameter:
. s - the PetscSection

  Output Parameter:
. size - the size of an array which can hold all the dofs

  Level: intermediate

.seealso: PetscSectionGetOffset(), PetscSectionGetConstrainedStorageSize(), PetscSectionCreate()
@*/
PetscErrorCode PetscSectionGetStorageSize(PetscSection s, PetscInt *size)
{
  PetscInt p, n = 0;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(s, PETSC_SECTION_CLASSID, 1);
  PetscValidPointer(size, 2);
  for (p = 0; p < s->pEnd - s->pStart; ++p) n += s->atlasDof[p] > 0 ? s->atlasDof[p] : 0;
  *size = n;
  PetscFunctionReturn(0);
}

/*@
  PetscSectionGetConstrainedStorageSize - Return the size of an array or local Vec capable of holding all unconstrained degrees of freedom.

  Not collective

  Input Parameter:
. s - the PetscSection

  Output Parameter:
. size - the size of an array which can hold all unconstrained dofs

  Level: intermediate

.seealso: PetscSectionGetStorageSize(), PetscSectionGetOffset(), PetscSectionCreate()
@*/
PetscErrorCode PetscSectionGetConstrainedStorageSize(PetscSection s, PetscInt *size)
{
  PetscInt p, n = 0;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(s, PETSC_SECTION_CLASSID, 1);
  PetscValidPointer(size, 2);
  for (p = 0; p < s->pEnd - s->pStart; ++p) {
    const PetscInt cdof = s->bc ? s->bc->atlasDof[p] : 0;
    n += s->atlasDof[p] > 0 ? s->atlasDof[p] - cdof : 0;
  }
  *size = n;
  PetscFunctionReturn(0);
}

/*@
  PetscSectionCreateGlobalSection - Create a section describing the global field layout using
  the local section and an SF describing the section point overlap.

  Input Parameters:
+ s - The PetscSection for the local field layout
. sf - The SF describing parallel layout of the section points (leaves are unowned local points)
. includeConstraints - By default this is PETSC_FALSE, meaning that the global field vector will not possess constrained dofs
- localOffsets - If PETSC_TRUE, use local rather than global offsets for the points

  Output Parameter:
. gsection - The PetscSection for the global field layout

  Note: This gives negative sizes and offsets to points not owned by this process

  Level: intermediate

.seealso: PetscSectionCreate()
@*/
PetscErrorCode PetscSectionCreateGlobalSection(PetscSection s, PetscSF sf, PetscBool includeConstraints, PetscBool localOffsets, PetscSection *gsection)
{
  PetscSection    gs;
  const PetscInt *pind = NULL;
  PetscInt       *recv = NULL, *neg = NULL;
  PetscInt        pStart, pEnd, p, dof, cdof, off, globalOff = 0, nroots, nlocal, maxleaf;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(s, PETSC_SECTION_CLASSID, 1);
  PetscValidHeaderSpecific(sf, PETSCSF_CLASSID, 2);
  PetscValidLogicalCollectiveBool(s, includeConstraints, 3);
  PetscValidLogicalCollectiveBool(s, localOffsets, 4);
  PetscValidPointer(gsection, 5);
  ierr = PetscSectionCreate(PetscObjectComm((PetscObject) s), &gs);CHKERRQ(ierr);
  ierr = PetscSectionGetChart(s, &pStart, &pEnd);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(gs, pStart, pEnd);CHKERRQ(ierr);
  ierr = PetscSFGetGraph(sf, &nroots, NULL, NULL, NULL);CHKERRQ(ierr);
  nlocal = nroots;              /* The local/leaf space matches global/root space */
  /* Must allocate for all points visible to SF, which may be more than this section */
  if (nroots >= 0) {             /* nroots < 0 means that the graph has not been set, only happens in serial */
    ierr = PetscSFGetLeafRange(sf, NULL, &maxleaf);CHKERRQ(ierr);
    if (nroots < pEnd) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "SF roots %D < pEnd %D", nroots, pEnd);
    if (maxleaf >= nroots) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Max local leaf %D >= nroots %D", maxleaf, nroots);
    ierr = PetscMalloc2(nroots,&neg,nlocal,&recv);CHKERRQ(ierr);
    ierr = PetscArrayzero(neg,nroots);CHKERRQ(ierr);
  }
  /* Mark all local points with negative dof */
  for (p = pStart; p < pEnd; ++p) {
    ierr = PetscSectionGetDof(s, p, &dof);CHKERRQ(ierr);
    ierr = PetscSectionSetDof(gs, p, dof);CHKERRQ(ierr);
    ierr = PetscSectionGetConstraintDof(s, p, &cdof);CHKERRQ(ierr);
    if (!includeConstraints && cdof > 0) {ierr = PetscSectionSetConstraintDof(gs, p, cdof);CHKERRQ(ierr);}
    if (neg) neg[p] = -(dof+1);
  }
  ierr = PetscSectionSetUpBC(gs);CHKERRQ(ierr);
  if (gs->bcIndices) {ierr = PetscArraycpy(gs->bcIndices, s->bcIndices,gs->bc->atlasOff[gs->bc->pEnd-gs->bc->pStart-1] + gs->bc->atlasDof[gs->bc->pEnd-gs->bc->pStart-1]);CHKERRQ(ierr);}
  if (nroots >= 0) {
    ierr = PetscArrayzero(recv,nlocal);CHKERRQ(ierr);
    ierr = PetscSFBcastBegin(sf, MPIU_INT, neg, recv);CHKERRQ(ierr);
    ierr = PetscSFBcastEnd(sf, MPIU_INT, neg, recv);CHKERRQ(ierr);
    for (p = pStart; p < pEnd; ++p) {
      if (recv[p] < 0) {
        gs->atlasDof[p-pStart] = recv[p];
        ierr = PetscSectionGetDof(s, p, &dof);CHKERRQ(ierr);
        if (-(recv[p]+1) != dof) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Global dof %D for point %D is not the unconstrained %D", -(recv[p]+1), p, dof);
      }
    }
  }
  /* Calculate new sizes, get process offset, and calculate point offsets */
  if (s->perm) {ierr = ISGetIndices(s->perm, &pind);CHKERRQ(ierr);}
  for (p = 0, off = 0; p < pEnd-pStart; ++p) {
    const PetscInt q = pind ? pind[p] : p;

    cdof = (!includeConstraints && s->bc) ? s->bc->atlasDof[q] : 0;
    gs->atlasOff[q] = off;
    off += gs->atlasDof[q] > 0 ? gs->atlasDof[q]-cdof : 0;
  }
  if (!localOffsets) {
    ierr = MPI_Scan(&off, &globalOff, 1, MPIU_INT, MPI_SUM, PetscObjectComm((PetscObject) s));CHKERRQ(ierr);
    globalOff -= off;
  }
  for (p = pStart, off = 0; p < pEnd; ++p) {
    gs->atlasOff[p-pStart] += globalOff;
    if (neg) neg[p] = -(gs->atlasOff[p-pStart]+1);
  }
  if (s->perm) {ierr = ISRestoreIndices(s->perm, &pind);CHKERRQ(ierr);}
  /* Put in negative offsets for ghost points */
  if (nroots >= 0) {
    ierr = PetscArrayzero(recv,nlocal);CHKERRQ(ierr);
    ierr = PetscSFBcastBegin(sf, MPIU_INT, neg, recv);CHKERRQ(ierr);
    ierr = PetscSFBcastEnd(sf, MPIU_INT, neg, recv);CHKERRQ(ierr);
    for (p = pStart; p < pEnd; ++p) {
      if (recv[p] < 0) gs->atlasOff[p-pStart] = recv[p];
    }
  }
  ierr = PetscFree2(neg,recv);CHKERRQ(ierr);
  ierr = PetscSectionViewFromOptions(gs, NULL, "-global_section_view");CHKERRQ(ierr);
  *gsection = gs;
  PetscFunctionReturn(0);
}

/*@
  PetscSectionCreateGlobalSectionCensored - Create a section describing the global field layout using
  the local section and an SF describing the section point overlap.

  Input Parameters:
  + s - The PetscSection for the local field layout
  . sf - The SF describing parallel layout of the section points
  . includeConstraints - By default this is PETSC_FALSE, meaning that the global field vector will not possess constrained dofs
  . numExcludes - The number of exclusion ranges
  - excludes - An array [start_0, end_0, start_1, end_1, ...] where there are numExcludes pairs

  Output Parameter:
  . gsection - The PetscSection for the global field layout

  Note: This gives negative sizes and offsets to points not owned by this process

  Level: advanced

.seealso: PetscSectionCreate()
@*/
PetscErrorCode PetscSectionCreateGlobalSectionCensored(PetscSection s, PetscSF sf, PetscBool includeConstraints, PetscInt numExcludes, const PetscInt excludes[], PetscSection *gsection)
{
  const PetscInt *pind = NULL;
  PetscInt       *neg  = NULL, *tmpOff = NULL;
  PetscInt        pStart, pEnd, p, e, dof, cdof, off, globalOff = 0, nroots;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(s, PETSC_SECTION_CLASSID, 1);
  PetscValidHeaderSpecific(sf, PETSCSF_CLASSID, 2);
  PetscValidPointer(gsection, 6);
  ierr = PetscSectionCreate(PetscObjectComm((PetscObject) s), gsection);CHKERRQ(ierr);
  ierr = PetscSectionGetChart(s, &pStart, &pEnd);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(*gsection, pStart, pEnd);CHKERRQ(ierr);
  ierr = PetscSFGetGraph(sf, &nroots, NULL, NULL, NULL);CHKERRQ(ierr);
  if (nroots >= 0) {
    if (nroots < pEnd-pStart) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "PetscSF nroots %D < %D section size", nroots, pEnd-pStart);
    ierr = PetscCalloc1(nroots, &neg);CHKERRQ(ierr);
    if (nroots > pEnd-pStart) {
      ierr = PetscCalloc1(nroots, &tmpOff);CHKERRQ(ierr);
    } else {
      tmpOff = &(*gsection)->atlasDof[-pStart];
    }
  }
  /* Mark ghost points with negative dof */
  for (p = pStart; p < pEnd; ++p) {
    for (e = 0; e < numExcludes; ++e) {
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
    if (neg) neg[p] = -(dof+1);
  }
  ierr = PetscSectionSetUpBC(*gsection);CHKERRQ(ierr);
  if (nroots >= 0) {
    ierr = PetscSFBcastBegin(sf, MPIU_INT, neg, tmpOff);CHKERRQ(ierr);
    ierr = PetscSFBcastEnd(sf, MPIU_INT, neg, tmpOff);CHKERRQ(ierr);
    if (nroots > pEnd - pStart) {
      for (p = pStart; p < pEnd; ++p) {if (tmpOff[p] < 0) (*gsection)->atlasDof[p-pStart] = tmpOff[p];}
    }
  }
  /* Calculate new sizes, get proccess offset, and calculate point offsets */
  if (s->perm) {ierr = ISGetIndices(s->perm, &pind);CHKERRQ(ierr);}
  for (p = 0, off = 0; p < pEnd-pStart; ++p) {
    const PetscInt q = pind ? pind[p] : p;

    cdof = (!includeConstraints && s->bc) ? s->bc->atlasDof[q] : 0;
    (*gsection)->atlasOff[q] = off;
    off += (*gsection)->atlasDof[q] > 0 ? (*gsection)->atlasDof[q]-cdof : 0;
  }
  ierr = MPI_Scan(&off, &globalOff, 1, MPIU_INT, MPI_SUM, PetscObjectComm((PetscObject) s));CHKERRQ(ierr);
  globalOff -= off;
  for (p = 0, off = 0; p < pEnd-pStart; ++p) {
    (*gsection)->atlasOff[p] += globalOff;
    if (neg) neg[p+pStart] = -((*gsection)->atlasOff[p]+1);
  }
  if (s->perm) {ierr = ISRestoreIndices(s->perm, &pind);CHKERRQ(ierr);}
  /* Put in negative offsets for ghost points */
  if (nroots >= 0) {
    if (nroots == pEnd-pStart) tmpOff = &(*gsection)->atlasOff[-pStart];
    ierr = PetscSFBcastBegin(sf, MPIU_INT, neg, tmpOff);CHKERRQ(ierr);
    ierr = PetscSFBcastEnd(sf, MPIU_INT, neg, tmpOff);CHKERRQ(ierr);
    if (nroots > pEnd - pStart) {
      for (p = pStart; p < pEnd; ++p) {if (tmpOff[p] < 0) (*gsection)->atlasOff[p-pStart] = tmpOff[p];}
    }
  }
  if (nroots >= 0 && nroots > pEnd-pStart) {ierr = PetscFree(tmpOff);CHKERRQ(ierr);}
  ierr = PetscFree(neg);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  PetscSectionGetPointLayout - Get the PetscLayout associated with the section points

  Collective on comm

  Input Parameters:
+ comm - The MPI_Comm
- s    - The PetscSection

  Output Parameter:
. layout - The point layout for the section

  Note: This is usually caleld for the default global section.

  Level: advanced

.seealso: PetscSectionGetValueLayout(), PetscSectionCreate()
@*/
PetscErrorCode PetscSectionGetPointLayout(MPI_Comm comm, PetscSection s, PetscLayout *layout)
{
  PetscInt       pStart, pEnd, p, localSize = 0;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscSectionGetChart(s, &pStart, &pEnd);CHKERRQ(ierr);
  for (p = pStart; p < pEnd; ++p) {
    PetscInt dof;

    ierr = PetscSectionGetDof(s, p, &dof);CHKERRQ(ierr);
    if (dof > 0) ++localSize;
  }
  ierr = PetscLayoutCreate(comm, layout);CHKERRQ(ierr);
  ierr = PetscLayoutSetLocalSize(*layout, localSize);CHKERRQ(ierr);
  ierr = PetscLayoutSetBlockSize(*layout, 1);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(*layout);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  PetscSectionGetValueLayout - Get the PetscLayout associated with the section dofs.

  Collective on comm

  Input Parameters:
+ comm - The MPI_Comm
- s    - The PetscSection

  Output Parameter:
. layout - The dof layout for the section

  Note: This is usually called for the default global section.

  Level: advanced

.seealso: PetscSectionGetPointLayout(), PetscSectionCreate()
@*/
PetscErrorCode PetscSectionGetValueLayout(MPI_Comm comm, PetscSection s, PetscLayout *layout)
{
  PetscInt       pStart, pEnd, p, localSize = 0;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(s, PETSC_SECTION_CLASSID, 2);
  PetscValidPointer(layout, 3);
  ierr = PetscSectionGetChart(s, &pStart, &pEnd);CHKERRQ(ierr);
  for (p = pStart; p < pEnd; ++p) {
    PetscInt dof,cdof;

    ierr = PetscSectionGetDof(s, p, &dof);CHKERRQ(ierr);
    ierr = PetscSectionGetConstraintDof(s, p, &cdof);CHKERRQ(ierr);
    if (dof-cdof > 0) localSize += dof-cdof;
  }
  ierr = PetscLayoutCreate(comm, layout);CHKERRQ(ierr);
  ierr = PetscLayoutSetLocalSize(*layout, localSize);CHKERRQ(ierr);
  ierr = PetscLayoutSetBlockSize(*layout, 1);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(*layout);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

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
  PetscValidHeaderSpecific(s, PETSC_SECTION_CLASSID, 1);
  PetscValidPointer(offset, 3);
  if (PetscDefined(USE_DEBUG)) {
    if ((point < s->pStart) || (point >= s->pEnd)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Section point %D should be in [%D, %D)", point, s->pStart, s->pEnd);
  }
  *offset = s->atlasOff[point - s->pStart];
  PetscFunctionReturn(0);
}

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
  PetscValidHeaderSpecific(s, PETSC_SECTION_CLASSID, 1);
  if ((point < s->pStart) || (point >= s->pEnd)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Section point %D should be in [%D, %D)", point, s->pStart, s->pEnd);
  s->atlasOff[point - s->pStart] = offset;
  PetscFunctionReturn(0);
}

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
  PetscValidHeaderSpecific(s, PETSC_SECTION_CLASSID, 1);
  PetscValidPointer(offset, 4);
  if ((field < 0) || (field >= s->numFields)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Section field %D should be in [%D, %D)", field, 0, s->numFields);
  ierr = PetscSectionGetOffset(s->field[field], point, offset);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

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
  PetscValidHeaderSpecific(s, PETSC_SECTION_CLASSID, 1);
  if ((field < 0) || (field >= s->numFields)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Section field %D should be in [%D, %D)", field, 0, s->numFields);
  ierr = PetscSectionSetOffset(s->field[field], point, offset);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  PetscSectionGetFieldPointOffset - Return the offset on the given point for the dof associated with the given point.

  Not collective

  Input Parameters:
+ s - the PetscSection
. point - the point
- field - the field

  Output Parameter:
. offset - the offset

  Note: This gives the offset on a point of the field, ignoring constraints, meaning starting at the first dof for
        this point, what number is the first dof with this field.

  Level: advanced

.seealso: PetscSectionGetOffset(), PetscSectionCreate()
@*/
PetscErrorCode PetscSectionGetFieldPointOffset(PetscSection s, PetscInt point, PetscInt field, PetscInt *offset)
{
  PetscInt       off, foff;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(s, PETSC_SECTION_CLASSID, 1);
  PetscValidPointer(offset, 4);
  if ((field < 0) || (field >= s->numFields)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Section field %D should be in [%D, %D)", field, 0, s->numFields);
  ierr = PetscSectionGetOffset(s, point, &off);CHKERRQ(ierr);
  ierr = PetscSectionGetOffset(s->field[field], point, &foff);CHKERRQ(ierr);
  *offset = foff - off;
  PetscFunctionReturn(0);
}

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
  PetscInt       os = 0, oe = 0, pStart, pEnd, p;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(s, PETSC_SECTION_CLASSID, 1);
  if (s->atlasOff) {os = s->atlasOff[0]; oe = s->atlasOff[0];}
  ierr = PetscSectionGetChart(s, &pStart, &pEnd);CHKERRQ(ierr);
  for (p = 0; p < pEnd-pStart; ++p) {
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

/*@
  PetscSectionCreateSubsection - Create a new, smaller section composed of only the selected fields

  Collective on s

  Input Parameter:
+ s      - the PetscSection
. len    - the number of subfields
- fields - the subfield numbers

  Output Parameter:
. subs   - the subsection

  Note: The section offsets now refer to a new, smaller vector.

  Level: advanced

.seealso: PetscSectionCreateSupersection(), PetscSectionCreate()
@*/
PetscErrorCode PetscSectionCreateSubsection(PetscSection s, PetscInt len, const PetscInt fields[], PetscSection *subs)
{
  PetscInt       nF, f, c, pStart, pEnd, p, maxCdof = 0;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!len) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(s, PETSC_SECTION_CLASSID, 1);
  PetscValidPointer(fields, 3);
  PetscValidPointer(subs, 4);
  ierr = PetscSectionGetNumFields(s, &nF);CHKERRQ(ierr);
  if (len > nF) SETERRQ2(PetscObjectComm((PetscObject) s), PETSC_ERR_ARG_WRONG, "Number of requested fields %D greater than number of fields %D", len, nF);
  ierr = PetscSectionCreate(PetscObjectComm((PetscObject) s), subs);CHKERRQ(ierr);
  ierr = PetscSectionSetNumFields(*subs, len);CHKERRQ(ierr);
  for (f = 0; f < len; ++f) {
    const char *name   = NULL;
    PetscInt   numComp = 0;

    ierr = PetscSectionGetFieldName(s, fields[f], &name);CHKERRQ(ierr);
    ierr = PetscSectionSetFieldName(*subs, f, name);CHKERRQ(ierr);
    ierr = PetscSectionGetFieldComponents(s, fields[f], &numComp);CHKERRQ(ierr);
    ierr = PetscSectionSetFieldComponents(*subs, f, numComp);CHKERRQ(ierr);
    for (c = 0; c < s->numFieldComponents[fields[f]]; ++c) {
      ierr = PetscSectionGetComponentName(s, fields[f], c, &name);CHKERRQ(ierr);
      ierr = PetscSectionSetComponentName(*subs, f, c, name);CHKERRQ(ierr);
    }
  }
  ierr = PetscSectionGetChart(s, &pStart, &pEnd);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(*subs, pStart, pEnd);CHKERRQ(ierr);
  for (p = pStart; p < pEnd; ++p) {
    PetscInt dof = 0, cdof = 0, fdof = 0, cfdof = 0;

    for (f = 0; f < len; ++f) {
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

    ierr = PetscMalloc1(maxCdof, &indices);CHKERRQ(ierr);
    for (p = pStart; p < pEnd; ++p) {
      PetscInt cdof;

      ierr = PetscSectionGetConstraintDof(*subs, p, &cdof);CHKERRQ(ierr);
      if (cdof) {
        const PetscInt *oldIndices = NULL;
        PetscInt       fdof = 0, cfdof = 0, fc, numConst = 0, fOff = 0;

        for (f = 0; f < len; ++f) {
          PetscInt oldFoff = 0, oldf;

          ierr = PetscSectionGetFieldDof(s, p, fields[f], &fdof);CHKERRQ(ierr);
          ierr = PetscSectionGetFieldConstraintDof(s, p, fields[f], &cfdof);CHKERRQ(ierr);
          ierr = PetscSectionGetFieldConstraintIndices(s, p, fields[f], &oldIndices);CHKERRQ(ierr);
          /* This can be sped up if we assume sorted fields */
          for (oldf = 0; oldf < fields[f]; ++oldf) {
            PetscInt oldfdof = 0;
            ierr = PetscSectionGetFieldDof(s, p, oldf, &oldfdof);CHKERRQ(ierr);
            oldFoff += oldfdof;
          }
          for (fc = 0; fc < cfdof; ++fc) indices[numConst+fc] = oldIndices[fc] + (fOff - oldFoff);
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

/*@
  PetscSectionCreateSupersection - Create a new, larger section composed of the input sections

  Collective on s

  Input Parameters:
+ s     - the input sections
- len   - the number of input sections

  Output Parameter:
. supers - the supersection

  Note: The section offsets now refer to a new, larger vector.

  Level: advanced

.seealso: PetscSectionCreateSubsection(), PetscSectionCreate()
@*/
PetscErrorCode PetscSectionCreateSupersection(PetscSection s[], PetscInt len, PetscSection *supers)
{
  PetscInt       Nf = 0, f, pStart = PETSC_MAX_INT, pEnd = 0, p, maxCdof = 0, i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!len) PetscFunctionReturn(0);
  for (i = 0; i < len; ++i) {
    PetscInt nf, pStarti, pEndi;

    ierr = PetscSectionGetNumFields(s[i], &nf);CHKERRQ(ierr);
    ierr = PetscSectionGetChart(s[i], &pStarti, &pEndi);CHKERRQ(ierr);
    pStart = PetscMin(pStart, pStarti);
    pEnd   = PetscMax(pEnd,   pEndi);
    Nf += nf;
  }
  ierr = PetscSectionCreate(PetscObjectComm((PetscObject) s[0]), supers);CHKERRQ(ierr);
  ierr = PetscSectionSetNumFields(*supers, Nf);CHKERRQ(ierr);
  for (i = 0, f = 0; i < len; ++i) {
    PetscInt nf, fi, ci;

    ierr = PetscSectionGetNumFields(s[i], &nf);CHKERRQ(ierr);
    for (fi = 0; fi < nf; ++fi, ++f) {
      const char *name   = NULL;
      PetscInt   numComp = 0;

      ierr = PetscSectionGetFieldName(s[i], fi, &name);CHKERRQ(ierr);
      ierr = PetscSectionSetFieldName(*supers, f, name);CHKERRQ(ierr);
      ierr = PetscSectionGetFieldComponents(s[i], fi, &numComp);CHKERRQ(ierr);
      ierr = PetscSectionSetFieldComponents(*supers, f, numComp);CHKERRQ(ierr);
      for (ci = 0; ci < s[i]->numFieldComponents[fi]; ++ci) {
        ierr = PetscSectionGetComponentName(s[i], fi, ci, &name);CHKERRQ(ierr);
        ierr = PetscSectionSetComponentName(*supers, f, ci, name);CHKERRQ(ierr);
      }
    }
  }
  ierr = PetscSectionSetChart(*supers, pStart, pEnd);CHKERRQ(ierr);
  for (p = pStart; p < pEnd; ++p) {
    PetscInt dof = 0, cdof = 0;

    for (i = 0, f = 0; i < len; ++i) {
      PetscInt nf, fi, pStarti, pEndi;
      PetscInt fdof = 0, cfdof = 0;

      ierr = PetscSectionGetNumFields(s[i], &nf);CHKERRQ(ierr);
      ierr = PetscSectionGetChart(s[i], &pStarti, &pEndi);CHKERRQ(ierr);
      if ((p < pStarti) || (p >= pEndi)) continue;
      for (fi = 0; fi < nf; ++fi, ++f) {
        ierr = PetscSectionGetFieldDof(s[i], p, fi, &fdof);CHKERRQ(ierr);
        ierr = PetscSectionAddFieldDof(*supers, p, f, fdof);CHKERRQ(ierr);
        ierr = PetscSectionGetFieldConstraintDof(s[i], p, fi, &cfdof);CHKERRQ(ierr);
        if (cfdof) {ierr = PetscSectionAddFieldConstraintDof(*supers, p, f, cfdof);CHKERRQ(ierr);}
        dof  += fdof;
        cdof += cfdof;
      }
    }
    ierr = PetscSectionSetDof(*supers, p, dof);CHKERRQ(ierr);
    if (cdof) {ierr = PetscSectionSetConstraintDof(*supers, p, cdof);CHKERRQ(ierr);}
    maxCdof = PetscMax(cdof, maxCdof);
  }
  ierr = PetscSectionSetUp(*supers);CHKERRQ(ierr);
  if (maxCdof) {
    PetscInt *indices;

    ierr = PetscMalloc1(maxCdof, &indices);CHKERRQ(ierr);
    for (p = pStart; p < pEnd; ++p) {
      PetscInt cdof;

      ierr = PetscSectionGetConstraintDof(*supers, p, &cdof);CHKERRQ(ierr);
      if (cdof) {
        PetscInt dof, numConst = 0, fOff = 0;

        for (i = 0, f = 0; i < len; ++i) {
          const PetscInt *oldIndices = NULL;
          PetscInt        nf, fi, pStarti, pEndi, fdof, cfdof, fc;

          ierr = PetscSectionGetNumFields(s[i], &nf);CHKERRQ(ierr);
          ierr = PetscSectionGetChart(s[i], &pStarti, &pEndi);CHKERRQ(ierr);
          if ((p < pStarti) || (p >= pEndi)) continue;
          for (fi = 0; fi < nf; ++fi, ++f) {
            ierr = PetscSectionGetFieldDof(s[i], p, fi, &fdof);CHKERRQ(ierr);
            ierr = PetscSectionGetFieldConstraintDof(s[i], p, fi, &cfdof);CHKERRQ(ierr);
            ierr = PetscSectionGetFieldConstraintIndices(s[i], p, fi, &oldIndices);CHKERRQ(ierr);
            for (fc = 0; fc < cfdof; ++fc) indices[numConst+fc] = oldIndices[fc] + fOff;
            ierr = PetscSectionSetFieldConstraintIndices(*supers, p, f, &indices[numConst]);CHKERRQ(ierr);
            numConst += cfdof;
          }
          ierr = PetscSectionGetDof(s[i], p, &dof);CHKERRQ(ierr);
          fOff += dof;
        }
        ierr = PetscSectionSetConstraintIndices(*supers, p, indices);CHKERRQ(ierr);
      }
    }
    ierr = PetscFree(indices);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@
  PetscSectionCreateSubmeshSection - Create a new, smaller section with support on the submesh

  Collective on s

  Input Parameters:
+ s           - the PetscSection
- subpointMap - a sorted list of points in the original mesh which are in the submesh

  Output Parameter:
. subs - the subsection

  Note: The section offsets now refer to a new, smaller vector.

  Level: advanced

.seealso: PetscSectionCreateSubsection(), DMPlexGetSubpointMap(), PetscSectionCreate()
@*/
PetscErrorCode PetscSectionCreateSubmeshSection(PetscSection s, IS subpointMap, PetscSection *subs)
{
  const PetscInt *points = NULL, *indices = NULL;
  PetscInt       numFields, f, c, numSubpoints = 0, pStart, pEnd, p, subp;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(s, PETSC_SECTION_CLASSID, 1);
  PetscValidHeaderSpecific(subpointMap, IS_CLASSID, 2);
  PetscValidPointer(subs, 3);
  ierr = PetscSectionGetNumFields(s, &numFields);CHKERRQ(ierr);
  ierr = PetscSectionCreate(PetscObjectComm((PetscObject) s), subs);CHKERRQ(ierr);
  if (numFields) {ierr = PetscSectionSetNumFields(*subs, numFields);CHKERRQ(ierr);}
  for (f = 0; f < numFields; ++f) {
    const char *name   = NULL;
    PetscInt   numComp = 0;

    ierr = PetscSectionGetFieldName(s, f, &name);CHKERRQ(ierr);
    ierr = PetscSectionSetFieldName(*subs, f, name);CHKERRQ(ierr);
    ierr = PetscSectionGetFieldComponents(s, f, &numComp);CHKERRQ(ierr);
    ierr = PetscSectionSetFieldComponents(*subs, f, numComp);CHKERRQ(ierr);
    for (c = 0; c < s->numFieldComponents[f]; ++c) {
      ierr = PetscSectionGetComponentName(s, f, c, &name);CHKERRQ(ierr);
      ierr = PetscSectionSetComponentName(*subs, f, c, name);CHKERRQ(ierr);
    }
  }
  /* For right now, we do not try to squeeze the subchart */
  if (subpointMap) {
    ierr = ISGetSize(subpointMap, &numSubpoints);CHKERRQ(ierr);
    ierr = ISGetIndices(subpointMap, &points);CHKERRQ(ierr);
  }
  ierr = PetscSectionGetChart(s, &pStart, &pEnd);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(*subs, 0, numSubpoints);CHKERRQ(ierr);
  for (p = pStart; p < pEnd; ++p) {
    PetscInt dof, cdof, fdof = 0, cfdof = 0;

    ierr = PetscFindInt(p, numSubpoints, points, &subp);CHKERRQ(ierr);
    if (subp < 0) continue;
    for (f = 0; f < numFields; ++f) {
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
  for (p = pStart; p < pEnd; ++p) {
    PetscInt off, foff = 0;

    ierr = PetscFindInt(p, numSubpoints, points, &subp);CHKERRQ(ierr);
    if (subp < 0) continue;
    for (f = 0; f < numFields; ++f) {
      ierr = PetscSectionGetFieldOffset(s, p, f, &foff);CHKERRQ(ierr);
      ierr = PetscSectionSetFieldOffset(*subs, subp, f, foff);CHKERRQ(ierr);
    }
    ierr = PetscSectionGetOffset(s, p, &off);CHKERRQ(ierr);
    ierr = PetscSectionSetOffset(*subs, subp, off);CHKERRQ(ierr);
  }
  /* Copy constraint indices */
  for (subp = 0; subp < numSubpoints; ++subp) {
    PetscInt cdof;

    ierr = PetscSectionGetConstraintDof(*subs, subp, &cdof);CHKERRQ(ierr);
    if (cdof) {
      for (f = 0; f < numFields; ++f) {
        ierr = PetscSectionGetFieldConstraintIndices(s, points[subp], f, &indices);CHKERRQ(ierr);
        ierr = PetscSectionSetFieldConstraintIndices(*subs, subp, f, indices);CHKERRQ(ierr);
      }
      ierr = PetscSectionGetConstraintIndices(s, points[subp], &indices);CHKERRQ(ierr);
      ierr = PetscSectionSetConstraintIndices(*subs, subp, indices);CHKERRQ(ierr);
    }
  }
  if (subpointMap) {ierr = ISRestoreIndices(subpointMap, &points);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSectionView_ASCII(PetscSection s, PetscViewer viewer)
{
  PetscInt       p;
  PetscMPIInt    rank;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)viewer), &rank);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPushSynchronized(viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIISynchronizedPrintf(viewer, "Process %d:\n", rank);CHKERRQ(ierr);
  for (p = 0; p < s->pEnd - s->pStart; ++p) {
    if ((s->bc) && (s->bc->atlasDof[p] > 0)) {
      PetscInt b;

      ierr = PetscViewerASCIISynchronizedPrintf(viewer, "  (%4D) dim %2D offset %3D constrained", p+s->pStart, s->atlasDof[p], s->atlasOff[p]);CHKERRQ(ierr);
      if (s->bcIndices) {
        for (b = 0; b < s->bc->atlasDof[p]; ++b) {
          ierr = PetscViewerASCIISynchronizedPrintf(viewer, " %D", s->bcIndices[s->bc->atlasOff[p]+b]);CHKERRQ(ierr);
        }
      }
      ierr = PetscViewerASCIISynchronizedPrintf(viewer, "\n");CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIISynchronizedPrintf(viewer, "  (%4D) dim %2D offset %3D\n", p+s->pStart, s->atlasDof[p], s->atlasOff[p]);CHKERRQ(ierr);
    }
  }
  ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPopSynchronized(viewer);CHKERRQ(ierr);
  if (s->sym) {
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    ierr = PetscSectionSymView(s->sym,viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@C
   PetscSectionViewFromOptions - View from Options

   Collective on PetscSection

   Input Parameters:
+  A - the PetscSection object to view
.  obj - Optional object
-  name - command line option

   Level: intermediate
.seealso:  PetscSection, PetscSectionView, PetscObjectViewFromOptions(), PetscSectionCreate()
@*/
PetscErrorCode  PetscSectionViewFromOptions(PetscSection A,PetscObject obj,const char name[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,PETSC_SECTION_CLASSID,1);
  ierr = PetscObjectViewFromOptions((PetscObject)A,obj,name);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  PetscSectionView - Views a PetscSection

  Collective on PetscSection

  Input Parameters:
+ s - the PetscSection object to view
- v - the viewer

  Level: beginner

.seealso PetscSectionCreate(), PetscSectionDestroy()
@*/
PetscErrorCode PetscSectionView(PetscSection s, PetscViewer viewer)
{
  PetscBool      isascii;
  PetscInt       f;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(s, PETSC_SECTION_CLASSID, 1);
  if (!viewer) {ierr = PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)s), &viewer);CHKERRQ(ierr);}
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = PetscObjectPrintClassNamePrefixType((PetscObject)s,viewer);CHKERRQ(ierr);
    if (s->numFields) {
      ierr = PetscViewerASCIIPrintf(viewer, "%D fields\n", s->numFields);CHKERRQ(ierr);
      for (f = 0; f < s->numFields; ++f) {
        ierr = PetscViewerASCIIPrintf(viewer, "  field %D with %D components\n", f, s->numFieldComponents[f]);CHKERRQ(ierr);
        ierr = PetscSectionView_ASCII(s->field[f], viewer);CHKERRQ(ierr);
      }
    } else {
      ierr = PetscSectionView_ASCII(s, viewer);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSectionResetClosurePermutation(PetscSection section)
{
  PetscErrorCode ierr;
  PetscSectionClosurePermVal clVal;

  PetscFunctionBegin;
  if (!section->clHash) PetscFunctionReturn(0);
  kh_foreach_value(section->clHash, clVal, {
      ierr = PetscFree(clVal.perm);CHKERRQ(ierr);
      ierr = PetscFree(clVal.invPerm);CHKERRQ(ierr);
    });
  kh_destroy(ClPerm, section->clHash);
  section->clHash = NULL;
  PetscFunctionReturn(0);
}

/*@
  PetscSectionReset - Frees all section data.

  Not collective

  Input Parameters:
. s - the PetscSection

  Level: beginner

.seealso: PetscSection, PetscSectionCreate()
@*/
PetscErrorCode PetscSectionReset(PetscSection s)
{
  PetscInt       f, c;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(s, PETSC_SECTION_CLASSID, 1);
  for (f = 0; f < s->numFields; ++f) {
    ierr = PetscSectionDestroy(&s->field[f]);CHKERRQ(ierr);
    ierr = PetscFree(s->fieldNames[f]);CHKERRQ(ierr);
    for (c = 0; c < s->numFieldComponents[f]; ++c)
      ierr = PetscFree(s->compNames[f][c]);CHKERRQ(ierr);
    ierr = PetscFree(s->compNames[f]);CHKERRQ(ierr);
  }
  ierr = PetscFree(s->numFieldComponents);CHKERRQ(ierr);
  ierr = PetscFree(s->fieldNames);CHKERRQ(ierr);
  ierr = PetscFree(s->compNames);CHKERRQ(ierr);
  ierr = PetscFree(s->field);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&s->bc);CHKERRQ(ierr);
  ierr = PetscFree(s->bcIndices);CHKERRQ(ierr);
  ierr = PetscFree2(s->atlasDof, s->atlasOff);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&s->clSection);CHKERRQ(ierr);
  ierr = ISDestroy(&s->clPoints);CHKERRQ(ierr);
  ierr = ISDestroy(&s->perm);CHKERRQ(ierr);
  ierr = PetscSectionResetClosurePermutation(s);CHKERRQ(ierr);
  ierr = PetscSectionSymDestroy(&s->sym);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&s->clSection);CHKERRQ(ierr);
  ierr = ISDestroy(&s->clPoints);CHKERRQ(ierr);

  s->pStart    = -1;
  s->pEnd      = -1;
  s->maxDof    = 0;
  s->setup     = PETSC_FALSE;
  s->numFields = 0;
  s->clObj     = NULL;
  PetscFunctionReturn(0);
}

/*@
  PetscSectionDestroy - Frees a section object and frees its range if that exists.

  Not collective

  Input Parameters:
. s - the PetscSection

  Level: beginner

.seealso: PetscSection, PetscSectionCreate()
@*/
PetscErrorCode PetscSectionDestroy(PetscSection *s)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*s) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(*s,PETSC_SECTION_CLASSID,1);
  if (--((PetscObject)(*s))->refct > 0) {
    *s = NULL;
    PetscFunctionReturn(0);
  }
  ierr = PetscSectionReset(*s);CHKERRQ(ierr);
  ierr = PetscHeaderDestroy(s);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecIntGetValuesSection(PetscInt *baseArray, PetscSection s, PetscInt point, const PetscInt **values)
{
  const PetscInt p = point - s->pStart;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(s, PETSC_SECTION_CLASSID, 2);
  *values = &baseArray[s->atlasOff[p]];
  PetscFunctionReturn(0);
}

PetscErrorCode VecIntSetValuesSection(PetscInt *baseArray, PetscSection s, PetscInt point, const PetscInt values[], InsertMode mode)
{
  PetscInt       *array;
  const PetscInt p           = point - s->pStart;
  const PetscInt orientation = 0; /* Needs to be included for use in closure operations */
  PetscInt       cDim        = 0;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(s, PETSC_SECTION_CLASSID, 2);
  ierr  = PetscSectionGetConstraintDof(s, p, &cDim);CHKERRQ(ierr);
  array = &baseArray[s->atlasOff[p]];
  if (!cDim) {
    if (orientation >= 0) {
      const PetscInt dim = s->atlasDof[p];
      PetscInt       i;

      if (mode == INSERT_VALUES) {
        for (i = 0; i < dim; ++i) array[i] = values[i];
      } else {
        for (i = 0; i < dim; ++i) array[i] += values[i];
      }
    } else {
      PetscInt offset = 0;
      PetscInt j      = -1, field, i;

      for (field = 0; field < s->numFields; ++field) {
        const PetscInt dim = s->field[field]->atlasDof[p];

        for (i = dim-1; i >= 0; --i) array[++j] = values[i+offset];
        offset += dim;
      }
    }
  } else {
    if (orientation >= 0) {
      const PetscInt dim  = s->atlasDof[p];
      PetscInt       cInd = 0, i;
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
      PetscInt       offset  = 0;
      PetscInt       cOffset = 0;
      PetscInt       j       = 0, field;

      ierr = PetscSectionGetConstraintIndices(s, point, &cDof);CHKERRQ(ierr);
      for (field = 0; field < s->numFields; ++field) {
        const PetscInt dim  = s->field[field]->atlasDof[p];     /* PetscSectionGetFieldDof() */
        const PetscInt tDim = s->field[field]->bc->atlasDof[p]; /* PetscSectionGetFieldConstraintDof() */
        const PetscInt sDim = dim - tDim;
        PetscInt       cInd = 0, i,k;

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

/*@C
  PetscSectionHasConstraints - Determine whether a section has constrained dofs

  Not collective

  Input Parameter:
. s - The PetscSection

  Output Parameter:
. hasConstraints - flag indicating that the section has constrained dofs

  Level: intermediate

.seealso: PetscSectionSetConstraintIndices(), PetscSectionGetConstraintDof(), PetscSection
@*/
PetscErrorCode PetscSectionHasConstraints(PetscSection s, PetscBool *hasConstraints)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(s, PETSC_SECTION_CLASSID, 1);
  PetscValidPointer(hasConstraints, 2);
  *hasConstraints = s->bc ? PETSC_TRUE : PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*@C
  PetscSectionGetConstraintIndices - Get the point dof numbers, in [0, dof), which are constrained

  Not collective

  Input Parameters:
+ s     - The PetscSection
- point - The point

  Output Parameter:
. indices - The constrained dofs

  Note: In Fortran, you call PetscSectionGetConstraintIndicesF90() and PetscSectionRestoreConstraintIndicesF90()

  Level: intermediate

.seealso: PetscSectionSetConstraintIndices(), PetscSectionGetConstraintDof(), PetscSection
@*/
PetscErrorCode PetscSectionGetConstraintIndices(PetscSection s, PetscInt point, const PetscInt **indices)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(s, PETSC_SECTION_CLASSID, 1);
  if (s->bc) {
    ierr = VecIntGetValuesSection(s->bcIndices, s->bc, point, indices);CHKERRQ(ierr);
  } else *indices = NULL;
  PetscFunctionReturn(0);
}

/*@C
  PetscSectionSetConstraintIndices - Set the point dof numbers, in [0, dof), which are constrained

  Not collective

  Input Parameters:
+ s     - The PetscSection
. point - The point
- indices - The constrained dofs

  Note: The Fortran is PetscSectionSetConstraintIndicesF90()

  Level: intermediate

.seealso: PetscSectionGetConstraintIndices(), PetscSectionGetConstraintDof(), PetscSection
@*/
PetscErrorCode PetscSectionSetConstraintIndices(PetscSection s, PetscInt point, const PetscInt indices[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(s, PETSC_SECTION_CLASSID, 1);
  if (s->bc) {
    ierr = VecIntSetValuesSection(s->bcIndices, s->bc, point, indices, INSERT_VALUES);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@C
  PetscSectionGetFieldConstraintIndices - Get the field dof numbers, in [0, fdof), which are constrained

  Not collective

  Input Parameters:
+ s     - The PetscSection
. field  - The field number
- point - The point

  Output Parameter:
. indices - The constrained dofs sorted in ascending order

  Notes:
  The indices array, which is provided by the caller, must have capacity to hold the number of constrained dofs, e.g., as returned by PetscSectionGetConstraintDof().

  Fortran Note:
  In Fortran, you call PetscSectionGetFieldConstraintIndicesF90() and PetscSectionRestoreFieldConstraintIndicesF90()

  Level: intermediate

.seealso: PetscSectionSetFieldConstraintIndices(), PetscSectionGetConstraintIndices(), PetscSectionGetConstraintDof(), PetscSection
@*/
PetscErrorCode PetscSectionGetFieldConstraintIndices(PetscSection s, PetscInt point, PetscInt field, const PetscInt **indices)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(s, PETSC_SECTION_CLASSID, 1);
  if ((field < 0) || (field >= s->numFields)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Section field %D should be in [%D, %D)", field, 0, s->numFields);
  ierr = PetscSectionGetConstraintIndices(s->field[field], point, indices);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  PetscSectionSetFieldConstraintIndices - Set the field dof numbers, in [0, fdof), which are constrained

  Not collective

  Input Parameters:
+ s       - The PetscSection
. point   - The point
. field   - The field number
- indices - The constrained dofs

  Note: The Fortran is PetscSectionSetFieldConstraintIndicesF90()

  Level: intermediate

.seealso: PetscSectionSetConstraintIndices(), PetscSectionGetFieldConstraintIndices(), PetscSectionGetConstraintDof(), PetscSection
@*/
PetscErrorCode PetscSectionSetFieldConstraintIndices(PetscSection s, PetscInt point, PetscInt field, const PetscInt indices[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(s, PETSC_SECTION_CLASSID, 1);
  if ((field < 0) || (field >= s->numFields)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Section field %D should be in [%D, %D)", field, 0, s->numFields);
  ierr = PetscSectionSetConstraintIndices(s->field[field], point, indices);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  PetscSectionPermute - Reorder the section according to the input point permutation

  Collective on PetscSection

  Input Parameter:
+ section - The PetscSection object
- perm - The point permutation, old point p becomes new point perm[p]

  Output Parameter:
. sectionNew - The permuted PetscSection

  Level: intermediate

.seealso: MatPermute()
@*/
PetscErrorCode PetscSectionPermute(PetscSection section, IS permutation, PetscSection *sectionNew)
{
  PetscSection    s = section, sNew;
  const PetscInt *perm;
  PetscInt        numFields, f, c, numPoints, pStart, pEnd, p;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(section, PETSC_SECTION_CLASSID, 1);
  PetscValidHeaderSpecific(permutation, IS_CLASSID, 2);
  PetscValidPointer(sectionNew, 3);
  ierr = PetscSectionCreate(PetscObjectComm((PetscObject) s), &sNew);CHKERRQ(ierr);
  ierr = PetscSectionGetNumFields(s, &numFields);CHKERRQ(ierr);
  if (numFields) {ierr = PetscSectionSetNumFields(sNew, numFields);CHKERRQ(ierr);}
  for (f = 0; f < numFields; ++f) {
    const char *name;
    PetscInt    numComp;

    ierr = PetscSectionGetFieldName(s, f, &name);CHKERRQ(ierr);
    ierr = PetscSectionSetFieldName(sNew, f, name);CHKERRQ(ierr);
    ierr = PetscSectionGetFieldComponents(s, f, &numComp);CHKERRQ(ierr);
    ierr = PetscSectionSetFieldComponents(sNew, f, numComp);CHKERRQ(ierr);
    for (c = 0; c < s->numFieldComponents[f]; ++c) {
      ierr = PetscSectionGetComponentName(s, f, c, &name);CHKERRQ(ierr);
      ierr = PetscSectionSetComponentName(sNew, f, c, name);CHKERRQ(ierr);
    }
  }
  ierr = ISGetLocalSize(permutation, &numPoints);CHKERRQ(ierr);
  ierr = ISGetIndices(permutation, &perm);CHKERRQ(ierr);
  ierr = PetscSectionGetChart(s, &pStart, &pEnd);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(sNew, pStart, pEnd);CHKERRQ(ierr);
  if (numPoints < pEnd) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Permutation size %D is less than largest Section point %D", numPoints, pEnd);
  for (p = pStart; p < pEnd; ++p) {
    PetscInt dof, cdof;

    ierr = PetscSectionGetDof(s, p, &dof);CHKERRQ(ierr);
    ierr = PetscSectionSetDof(sNew, perm[p], dof);CHKERRQ(ierr);
    ierr = PetscSectionGetConstraintDof(s, p, &cdof);CHKERRQ(ierr);
    if (cdof) {ierr = PetscSectionSetConstraintDof(sNew, perm[p], cdof);CHKERRQ(ierr);}
    for (f = 0; f < numFields; ++f) {
      ierr = PetscSectionGetFieldDof(s, p, f, &dof);CHKERRQ(ierr);
      ierr = PetscSectionSetFieldDof(sNew, perm[p], f, dof);CHKERRQ(ierr);
      ierr = PetscSectionGetFieldConstraintDof(s, p, f, &cdof);CHKERRQ(ierr);
      if (cdof) {ierr = PetscSectionSetFieldConstraintDof(sNew, perm[p], f, cdof);CHKERRQ(ierr);}
    }
  }
  ierr = PetscSectionSetUp(sNew);CHKERRQ(ierr);
  for (p = pStart; p < pEnd; ++p) {
    const PetscInt *cind;
    PetscInt        cdof;

    ierr = PetscSectionGetConstraintDof(s, p, &cdof);CHKERRQ(ierr);
    if (cdof) {
      ierr = PetscSectionGetConstraintIndices(s, p, &cind);CHKERRQ(ierr);
      ierr = PetscSectionSetConstraintIndices(sNew, perm[p], cind);CHKERRQ(ierr);
    }
    for (f = 0; f < numFields; ++f) {
      ierr = PetscSectionGetFieldConstraintDof(s, p, f, &cdof);CHKERRQ(ierr);
      if (cdof) {
        ierr = PetscSectionGetFieldConstraintIndices(s, p, f, &cind);CHKERRQ(ierr);
        ierr = PetscSectionSetFieldConstraintIndices(sNew, perm[p], f, cind);CHKERRQ(ierr);
      }
    }
  }
  ierr = ISRestoreIndices(permutation, &perm);CHKERRQ(ierr);
  *sectionNew = sNew;
  PetscFunctionReturn(0);
}

/* TODO: the next three functions should be moved to sf/utils */
#include <petsc/private/sfimpl.h>

/*@C
  PetscSFDistributeSection - Create a new PetscSection reorganized, moving from the root to the leaves of the SF

  Collective on sf

  Input Parameters:
+ sf - The SF
- rootSection - Section defined on root space

  Output Parameters:
+ remoteOffsets - root offsets in leaf storage, or NULL
- leafSection - Section defined on the leaf space

  Level: advanced

.seealso: PetscSFCreate()
@*/
PetscErrorCode PetscSFDistributeSection(PetscSF sf, PetscSection rootSection, PetscInt **remoteOffsets, PetscSection leafSection)
{
  PetscSF        embedSF;
  const PetscInt *indices;
  IS             selected;
  PetscInt       numFields, nroots, rpStart, rpEnd, lpStart = PETSC_MAX_INT, lpEnd = -1, f, c;
  PetscBool      *sub, hasc;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(PETSCSF_DistSect,sf,0,0,0);CHKERRQ(ierr);
  ierr = PetscSectionGetNumFields(rootSection, &numFields);CHKERRQ(ierr);
  if (numFields) {ierr = PetscSectionSetNumFields(leafSection, numFields);CHKERRQ(ierr);}
  ierr = PetscMalloc1(numFields+2, &sub);CHKERRQ(ierr);
  sub[1] = rootSection->bc ? PETSC_TRUE : PETSC_FALSE;
  for (f = 0; f < numFields; ++f) {
    PetscSectionSym sym;
    const char      *name   = NULL;
    PetscInt        numComp = 0;

    sub[2 + f] = rootSection->field[f]->bc ? PETSC_TRUE : PETSC_FALSE;
    ierr = PetscSectionGetFieldComponents(rootSection, f, &numComp);CHKERRQ(ierr);
    ierr = PetscSectionGetFieldName(rootSection, f, &name);CHKERRQ(ierr);
    ierr = PetscSectionGetFieldSym(rootSection, f, &sym);CHKERRQ(ierr);
    ierr = PetscSectionSetFieldComponents(leafSection, f, numComp);CHKERRQ(ierr);
    ierr = PetscSectionSetFieldName(leafSection, f, name);CHKERRQ(ierr);
    ierr = PetscSectionSetFieldSym(leafSection, f, sym);CHKERRQ(ierr);
    for (c = 0; c < rootSection->numFieldComponents[f]; ++c) {
      ierr = PetscSectionGetComponentName(rootSection, f, c, &name);CHKERRQ(ierr);
      ierr = PetscSectionSetComponentName(leafSection, f, c, name);CHKERRQ(ierr);
    }
  }
  ierr = PetscSectionGetChart(rootSection, &rpStart, &rpEnd);CHKERRQ(ierr);
  ierr = PetscSFGetGraph(sf,&nroots,NULL,NULL,NULL);CHKERRQ(ierr);
  rpEnd = PetscMin(rpEnd,nroots);
  rpEnd = PetscMax(rpStart,rpEnd);
  /* see if we can avoid creating the embedded SF, since it can cost more than an allreduce */
  sub[0] = (PetscBool)(nroots != rpEnd - rpStart);
  ierr = MPIU_Allreduce(MPI_IN_PLACE, sub, 2+numFields, MPIU_BOOL, MPI_LOR, PetscObjectComm((PetscObject)sf));CHKERRQ(ierr);
  if (sub[0]) {
    ierr = ISCreateStride(PETSC_COMM_SELF, rpEnd - rpStart, rpStart, 1, &selected);CHKERRQ(ierr);
    ierr = ISGetIndices(selected, &indices);CHKERRQ(ierr);
    ierr = PetscSFCreateEmbeddedSF(sf, rpEnd - rpStart, indices, &embedSF);CHKERRQ(ierr);
    ierr = ISRestoreIndices(selected, &indices);CHKERRQ(ierr);
    ierr = ISDestroy(&selected);CHKERRQ(ierr);
  } else {
    ierr = PetscObjectReference((PetscObject)sf);CHKERRQ(ierr);
    embedSF = sf;
  }
  ierr = PetscSFGetLeafRange(embedSF, &lpStart, &lpEnd);CHKERRQ(ierr);
  lpEnd++;

  ierr = PetscSectionSetChart(leafSection, lpStart, lpEnd);CHKERRQ(ierr);

  /* Constrained dof section */
  hasc = sub[1];
  for (f = 0; f < numFields; ++f) hasc = (PetscBool)(hasc || sub[2+f]);

  /* Could fuse these at the cost of copies and extra allocation */
  ierr = PetscSFBcastBegin(embedSF, MPIU_INT, &rootSection->atlasDof[-rpStart], &leafSection->atlasDof[-lpStart]);CHKERRQ(ierr);
  ierr = PetscSFBcastEnd(embedSF, MPIU_INT, &rootSection->atlasDof[-rpStart], &leafSection->atlasDof[-lpStart]);CHKERRQ(ierr);
  if (sub[1]) {
    ierr = PetscSectionCheckConstraints_Static(rootSection);CHKERRQ(ierr);
    ierr = PetscSectionCheckConstraints_Static(leafSection);CHKERRQ(ierr);
    ierr = PetscSFBcastBegin(embedSF, MPIU_INT, &rootSection->bc->atlasDof[-rpStart], &leafSection->bc->atlasDof[-lpStart]);CHKERRQ(ierr);
    ierr = PetscSFBcastEnd(embedSF, MPIU_INT, &rootSection->bc->atlasDof[-rpStart], &leafSection->bc->atlasDof[-lpStart]);CHKERRQ(ierr);
  }
  for (f = 0; f < numFields; ++f) {
    ierr = PetscSFBcastBegin(embedSF, MPIU_INT, &rootSection->field[f]->atlasDof[-rpStart], &leafSection->field[f]->atlasDof[-lpStart]);CHKERRQ(ierr);
    ierr = PetscSFBcastEnd(embedSF, MPIU_INT, &rootSection->field[f]->atlasDof[-rpStart], &leafSection->field[f]->atlasDof[-lpStart]);CHKERRQ(ierr);
    if (sub[2+f]) {
      ierr = PetscSectionCheckConstraints_Static(rootSection->field[f]);CHKERRQ(ierr);
      ierr = PetscSectionCheckConstraints_Static(leafSection->field[f]);CHKERRQ(ierr);
      ierr = PetscSFBcastBegin(embedSF, MPIU_INT, &rootSection->field[f]->bc->atlasDof[-rpStart], &leafSection->field[f]->bc->atlasDof[-lpStart]);CHKERRQ(ierr);
      ierr = PetscSFBcastEnd(embedSF, MPIU_INT, &rootSection->field[f]->bc->atlasDof[-rpStart], &leafSection->field[f]->bc->atlasDof[-lpStart]);CHKERRQ(ierr);
    }
  }
  if (remoteOffsets) {
    ierr = PetscMalloc1(lpEnd - lpStart, remoteOffsets);CHKERRQ(ierr);
    ierr = PetscSFBcastBegin(embedSF, MPIU_INT, &rootSection->atlasOff[-rpStart], &(*remoteOffsets)[-lpStart]);CHKERRQ(ierr);
    ierr = PetscSFBcastEnd(embedSF, MPIU_INT, &rootSection->atlasOff[-rpStart], &(*remoteOffsets)[-lpStart]);CHKERRQ(ierr);
  }
  ierr = PetscSectionSetUp(leafSection);CHKERRQ(ierr);
  if (hasc) { /* need to communicate bcIndices */
    PetscSF  bcSF;
    PetscInt *rOffBc;

    ierr = PetscMalloc1(lpEnd - lpStart, &rOffBc);CHKERRQ(ierr);
    if (sub[1]) {
      ierr = PetscSFBcastBegin(embedSF, MPIU_INT, &rootSection->bc->atlasOff[-rpStart], &rOffBc[-lpStart]);CHKERRQ(ierr);
      ierr = PetscSFBcastEnd(embedSF, MPIU_INT, &rootSection->bc->atlasOff[-rpStart], &rOffBc[-lpStart]);CHKERRQ(ierr);
      ierr = PetscSFCreateSectionSF(embedSF, rootSection->bc, rOffBc, leafSection->bc, &bcSF);CHKERRQ(ierr);
      ierr = PetscSFBcastBegin(bcSF, MPIU_INT, rootSection->bcIndices, leafSection->bcIndices);CHKERRQ(ierr);
      ierr = PetscSFBcastEnd(bcSF, MPIU_INT, rootSection->bcIndices, leafSection->bcIndices);CHKERRQ(ierr);
      ierr = PetscSFDestroy(&bcSF);CHKERRQ(ierr);
    }
    for (f = 0; f < numFields; ++f) {
      if (sub[2+f]) {
        ierr = PetscSFBcastBegin(embedSF, MPIU_INT, &rootSection->field[f]->bc->atlasOff[-rpStart], &rOffBc[-lpStart]);CHKERRQ(ierr);
        ierr = PetscSFBcastEnd(embedSF, MPIU_INT, &rootSection->field[f]->bc->atlasOff[-rpStart], &rOffBc[-lpStart]);CHKERRQ(ierr);
        ierr = PetscSFCreateSectionSF(embedSF, rootSection->field[f]->bc, rOffBc, leafSection->field[f]->bc, &bcSF);CHKERRQ(ierr);
        ierr = PetscSFBcastBegin(bcSF, MPIU_INT, rootSection->field[f]->bcIndices, leafSection->field[f]->bcIndices);CHKERRQ(ierr);
        ierr = PetscSFBcastEnd(bcSF, MPIU_INT, rootSection->field[f]->bcIndices, leafSection->field[f]->bcIndices);CHKERRQ(ierr);
        ierr = PetscSFDestroy(&bcSF);CHKERRQ(ierr);
      }
    }
    ierr = PetscFree(rOffBc);CHKERRQ(ierr);
  }
  ierr = PetscSFDestroy(&embedSF);CHKERRQ(ierr);
  ierr = PetscFree(sub);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(PETSCSF_DistSect,sf,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  PetscSFCreateRemoteOffsets - Create offsets for point data on remote processes

  Collective on sf

  Input Parameters:
+ sf          - The SF
. rootSection - Data layout of remote points for outgoing data (this is layout for SF roots)
- leafSection - Data layout of local points for incoming data  (this is layout for SF leaves)

  Output Parameter:
. remoteOffsets - Offsets for point data on remote processes (these are offsets from the root section), or NULL

  Level: developer

.seealso: PetscSFCreate()
@*/
PetscErrorCode PetscSFCreateRemoteOffsets(PetscSF sf, PetscSection rootSection, PetscSection leafSection, PetscInt **remoteOffsets)
{
  PetscSF         embedSF;
  const PetscInt *indices;
  IS              selected;
  PetscInt        numRoots, rpStart = 0, rpEnd = 0, lpStart = 0, lpEnd = 0;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  *remoteOffsets = NULL;
  ierr = PetscSFGetGraph(sf, &numRoots, NULL, NULL, NULL);CHKERRQ(ierr);
  if (numRoots < 0) PetscFunctionReturn(0);
  ierr = PetscLogEventBegin(PETSCSF_RemoteOff,sf,0,0,0);CHKERRQ(ierr);
  ierr = PetscSectionGetChart(rootSection, &rpStart, &rpEnd);CHKERRQ(ierr);
  ierr = PetscSectionGetChart(leafSection, &lpStart, &lpEnd);CHKERRQ(ierr);
  ierr = ISCreateStride(PETSC_COMM_SELF, rpEnd - rpStart, rpStart, 1, &selected);CHKERRQ(ierr);
  ierr = ISGetIndices(selected, &indices);CHKERRQ(ierr);
  ierr = PetscSFCreateEmbeddedSF(sf, rpEnd - rpStart, indices, &embedSF);CHKERRQ(ierr);
  ierr = ISRestoreIndices(selected, &indices);CHKERRQ(ierr);
  ierr = ISDestroy(&selected);CHKERRQ(ierr);
  ierr = PetscCalloc1(lpEnd - lpStart, remoteOffsets);CHKERRQ(ierr);
  ierr = PetscSFBcastBegin(embedSF, MPIU_INT, &rootSection->atlasOff[-rpStart], &(*remoteOffsets)[-lpStart]);CHKERRQ(ierr);
  ierr = PetscSFBcastEnd(embedSF, MPIU_INT, &rootSection->atlasOff[-rpStart], &(*remoteOffsets)[-lpStart]);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&embedSF);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(PETSCSF_RemoteOff,sf,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  PetscSFCreateSectionSF - Create an expanded SF of dofs, assuming the input SF relates points

  Collective on sf

  Input Parameters:
+ sf - The SF
. rootSection - Data layout of remote points for outgoing data (this is usually the serial section)
. remoteOffsets - Offsets for point data on remote processes (these are offsets from the root section), or NULL
- leafSection - Data layout of local points for incoming data  (this is the distributed section)

  Output Parameters:
- sectionSF - The new SF

  Note: Either rootSection or remoteOffsets can be specified

  Level: advanced

.seealso: PetscSFCreate()
@*/
PetscErrorCode PetscSFCreateSectionSF(PetscSF sf, PetscSection rootSection, PetscInt remoteOffsets[], PetscSection leafSection, PetscSF *sectionSF)
{
  MPI_Comm          comm;
  const PetscInt    *localPoints;
  const PetscSFNode *remotePoints;
  PetscInt          lpStart, lpEnd;
  PetscInt          numRoots, numSectionRoots, numPoints, numIndices = 0;
  PetscInt          *localIndices;
  PetscSFNode       *remoteIndices;
  PetscInt          i, ind;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf,PETSCSF_CLASSID,1);
  PetscValidPointer(rootSection,2);
  /* Cannot check PetscValidIntPointer(remoteOffsets,3) because it can be NULL if sf does not reference any points in leafSection */
  PetscValidPointer(leafSection,4);
  PetscValidPointer(sectionSF,5);
  ierr = PetscObjectGetComm((PetscObject)sf,&comm);CHKERRQ(ierr);
  ierr = PetscSFCreate(comm, sectionSF);CHKERRQ(ierr);
  ierr = PetscSectionGetChart(leafSection, &lpStart, &lpEnd);CHKERRQ(ierr);
  ierr = PetscSectionGetStorageSize(rootSection, &numSectionRoots);CHKERRQ(ierr);
  ierr = PetscSFGetGraph(sf, &numRoots, &numPoints, &localPoints, &remotePoints);CHKERRQ(ierr);
  if (numRoots < 0) PetscFunctionReturn(0);
  ierr = PetscLogEventBegin(PETSCSF_SectSF,sf,0,0,0);CHKERRQ(ierr);
  for (i = 0; i < numPoints; ++i) {
    PetscInt localPoint = localPoints ? localPoints[i] : i;
    PetscInt dof;

    if ((localPoint >= lpStart) && (localPoint < lpEnd)) {
      ierr = PetscSectionGetDof(leafSection, localPoint, &dof);CHKERRQ(ierr);
      numIndices += dof;
    }
  }
  ierr = PetscMalloc1(numIndices, &localIndices);CHKERRQ(ierr);
  ierr = PetscMalloc1(numIndices, &remoteIndices);CHKERRQ(ierr);
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
  if (numIndices != ind) SETERRQ2(comm, PETSC_ERR_PLIB, "Inconsistency in indices, %D should be %D", ind, numIndices);
  ierr = PetscSFSetGraph(*sectionSF, numSectionRoots, numIndices, localIndices, PETSC_OWN_POINTER, remoteIndices, PETSC_OWN_POINTER);CHKERRQ(ierr);
  ierr = PetscSFSetUp(*sectionSF);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(PETSCSF_SectSF,sf,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  PetscSectionSetClosureIndex - Set a cache of points in the closure of each point in the section

  Collective on section

  Input Parameters:
+ section   - The PetscSection
. obj       - A PetscObject which serves as the key for this index
. clSection - Section giving the size of the closure of each point
- clPoints  - IS giving the points in each closure

  Note: We compress out closure points with no dofs in this section

  Level: advanced

.seealso: PetscSectionGetClosureIndex(), DMPlexCreateClosureIndex()
@*/
PetscErrorCode PetscSectionSetClosureIndex(PetscSection section, PetscObject obj, PetscSection clSection, IS clPoints)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(section,PETSC_SECTION_CLASSID,1);
  PetscValidHeaderSpecific(clSection,PETSC_SECTION_CLASSID,3);
  PetscValidHeaderSpecific(clPoints,IS_CLASSID,4);
  if (section->clObj != obj) {ierr = PetscSectionResetClosurePermutation(section);CHKERRQ(ierr);}
  section->clObj     = obj;
  ierr = PetscObjectReference((PetscObject)clSection);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)clPoints);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&section->clSection);CHKERRQ(ierr);
  ierr = ISDestroy(&section->clPoints);CHKERRQ(ierr);
  section->clSection = clSection;
  section->clPoints  = clPoints;
  PetscFunctionReturn(0);
}

/*@
  PetscSectionGetClosureIndex - Get the cache of points in the closure of each point in the section

  Collective on section

  Input Parameters:
+ section   - The PetscSection
- obj       - A PetscObject which serves as the key for this index

  Output Parameters:
+ clSection - Section giving the size of the closure of each point
- clPoints  - IS giving the points in each closure

  Note: We compress out closure points with no dofs in this section

  Level: advanced

.seealso: PetscSectionSetClosureIndex(), DMPlexCreateClosureIndex()
@*/
PetscErrorCode PetscSectionGetClosureIndex(PetscSection section, PetscObject obj, PetscSection *clSection, IS *clPoints)
{
  PetscFunctionBegin;
  if (section->clObj == obj) {
    if (clSection) *clSection = section->clSection;
    if (clPoints)  *clPoints  = section->clPoints;
  } else {
    if (clSection) *clSection = NULL;
    if (clPoints)  *clPoints  = NULL;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscSectionSetClosurePermutation_Internal(PetscSection section, PetscObject obj, PetscInt depth, PetscInt clSize, PetscCopyMode mode, PetscInt *clPerm)
{
  PetscInt       i;
  khiter_t iter;
  int new_entry;
  PetscSectionClosurePermKey key = {depth, clSize};
  PetscSectionClosurePermVal *val;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (section->clObj != obj) {
    ierr = PetscSectionDestroy(&section->clSection);CHKERRQ(ierr);
    ierr = ISDestroy(&section->clPoints);CHKERRQ(ierr);
  }
  section->clObj = obj;
  if (!section->clHash) {ierr = PetscClPermCreate(&section->clHash);CHKERRQ(ierr);}
  iter = kh_put(ClPerm, section->clHash, key, &new_entry);
  val = &kh_val(section->clHash, iter);
  if (!new_entry) {
    ierr = PetscFree(val->perm);CHKERRQ(ierr);
    ierr = PetscFree(val->invPerm);CHKERRQ(ierr);
  }
  if (mode == PETSC_COPY_VALUES) {
    ierr = PetscMalloc1(clSize, &val->perm);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory((PetscObject) obj, clSize*sizeof(PetscInt));CHKERRQ(ierr);
    ierr = PetscArraycpy(val->perm, clPerm, clSize);CHKERRQ(ierr);
  } else if (mode == PETSC_OWN_POINTER) {
    val->perm = clPerm;
  } else SETERRQ(PetscObjectComm(obj), PETSC_ERR_SUP, "Do not support borrowed arrays");
  ierr = PetscMalloc1(clSize, &val->invPerm);CHKERRQ(ierr);
  for (i = 0; i < clSize; ++i) val->invPerm[clPerm[i]] = i;
  PetscFunctionReturn(0);
}

/*@
  PetscSectionSetClosurePermutation - Set the dof permutation for the closure of each cell in the section, meaning clPerm[newIndex] = oldIndex.

  Not Collective

  Input Parameters:
+ section - The PetscSection
. obj     - A PetscObject which serves as the key for this index (usually a DM)
. depth   - Depth of points on which to apply the given permutation
- perm    - Permutation of the cell dof closure

  Note:
  The specified permutation will only be applied to points at depth whose closure size matches the length of perm.  In a
  mixed-topology or variable-degree finite element space, this function can be called multiple times at each depth for
  each topology and degree.

  This approach assumes that (depth, len(perm)) uniquely identifies the desired permutation; this might not be true for
  exotic/enriched spaces on mixed topology meshes.

  Level: intermediate

.seealso: PetscSectionGetClosurePermutation(), PetscSectionGetClosureIndex(), DMPlexCreateClosureIndex(), PetscCopyMode
@*/
PetscErrorCode PetscSectionSetClosurePermutation(PetscSection section, PetscObject obj, PetscInt depth, IS perm)
{
  const PetscInt *clPerm = NULL;
  PetscInt        clSize = 0;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  if (perm) {
    ierr = ISGetLocalSize(perm, &clSize);CHKERRQ(ierr);
    ierr = ISGetIndices(perm, &clPerm);CHKERRQ(ierr);
  }
  ierr = PetscSectionSetClosurePermutation_Internal(section, obj, depth, clSize, PETSC_COPY_VALUES, (PetscInt *) clPerm);CHKERRQ(ierr);
  if (perm) {ierr = ISRestoreIndices(perm, &clPerm);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

PetscErrorCode PetscSectionGetClosurePermutation_Internal(PetscSection section, PetscObject obj, PetscInt depth, PetscInt size, const PetscInt *perm[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (section->clObj == obj) {
    PetscSectionClosurePermKey k = {depth, size};
    PetscSectionClosurePermVal v;
    ierr = PetscClPermGet(section->clHash, k, &v);CHKERRQ(ierr);
    if (perm) *perm = v.perm;
  } else {
    if (perm) *perm = NULL;
  }
  PetscFunctionReturn(0);
}

/*@
  PetscSectionGetClosurePermutation - Get the dof permutation for the closure of each cell in the section, meaning clPerm[newIndex] = oldIndex.

  Not collective

  Input Parameters:
+ section   - The PetscSection
. obj       - A PetscObject which serves as the key for this index (usually a DM)
. depth     - Depth stratum on which to obtain closure permutation
- clSize    - Closure size to be permuted (e.g., may vary with element topology and degree)

  Output Parameter:
. perm - The dof closure permutation

  Note:
  The user must destroy the IS that is returned.

  Level: intermediate

.seealso: PetscSectionSetClosurePermutation(), PetscSectionGetClosureInversePermutation(), PetscSectionGetClosureIndex(), PetscSectionSetClosureIndex(), DMPlexCreateClosureIndex()
@*/
PetscErrorCode PetscSectionGetClosurePermutation(PetscSection section, PetscObject obj, PetscInt depth, PetscInt clSize, IS *perm)
{
  const PetscInt *clPerm;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscSectionGetClosurePermutation_Internal(section, obj, depth, clSize, &clPerm);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_SELF, clSize, clPerm, PETSC_USE_POINTER, perm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscSectionGetClosureInversePermutation_Internal(PetscSection section, PetscObject obj, PetscInt depth, PetscInt size, const PetscInt *perm[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (section->clObj == obj && section->clHash) {
    PetscSectionClosurePermKey k = {depth, size};
    PetscSectionClosurePermVal v;
    ierr = PetscClPermGet(section->clHash, k, &v);CHKERRQ(ierr);
    if (perm) *perm = v.invPerm;
  } else {
    if (perm) *perm = NULL;
  }
  PetscFunctionReturn(0);
}

/*@
  PetscSectionGetClosureInversePermutation - Get the inverse dof permutation for the closure of each cell in the section, meaning clPerm[oldIndex] = newIndex.

  Not collective

  Input Parameters:
+ section   - The PetscSection
. obj       - A PetscObject which serves as the key for this index (usually a DM)
. depth     - Depth stratum on which to obtain closure permutation
- clSize    - Closure size to be permuted (e.g., may vary with element topology and degree)

  Output Parameters:
. perm - The dof closure permutation

  Note:
  The user must destroy the IS that is returned.

  Level: intermediate

.seealso: PetscSectionSetClosurePermutation(), PetscSectionGetClosureIndex(), PetscSectionSetClosureIndex(), DMPlexCreateClosureIndex()
@*/
PetscErrorCode PetscSectionGetClosureInversePermutation(PetscSection section, PetscObject obj, PetscInt depth, PetscInt clSize, IS *perm)
{
  const PetscInt *clPerm;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscSectionGetClosureInversePermutation_Internal(section, obj, depth, clSize, &clPerm);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_SELF, clSize, clPerm, PETSC_USE_POINTER, perm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  PetscSectionGetField - Get the subsection associated with a single field

  Input Parameters:
+ s     - The PetscSection
- field - The field number

  Output Parameter:
. subs  - The subsection for the given field

  Level: intermediate

.seealso: PetscSectionSetNumFields()
@*/
PetscErrorCode PetscSectionGetField(PetscSection s, PetscInt field, PetscSection *subs)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(s,PETSC_SECTION_CLASSID,1);
  PetscValidPointer(subs,3);
  if ((field < 0) || (field >= s->numFields)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Section field %D should be in [%D, %D)", field, 0, s->numFields);
  *subs = s->field[field];
  PetscFunctionReturn(0);
}

PetscClassId      PETSC_SECTION_SYM_CLASSID;
PetscFunctionList PetscSectionSymList = NULL;

/*@
  PetscSectionSymCreate - Creates an empty PetscSectionSym object.

  Collective

  Input Parameter:
. comm - the MPI communicator

  Output Parameter:
. sym - pointer to the new set of symmetries

  Level: developer

.seealso: PetscSectionSym, PetscSectionSymDestroy()
@*/
PetscErrorCode PetscSectionSymCreate(MPI_Comm comm, PetscSectionSym *sym)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(sym,2);
  ierr = ISInitializePackage();CHKERRQ(ierr);
  ierr = PetscHeaderCreate(*sym,PETSC_SECTION_SYM_CLASSID,"PetscSectionSym","Section Symmetry","IS",comm,PetscSectionSymDestroy,PetscSectionSymView);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  PetscSectionSymSetType - Builds a PetscSection symmetry, for a particular implementation.

  Collective on PetscSectionSym

  Input Parameters:
+ sym    - The section symmetry object
- method - The name of the section symmetry type

  Level: developer

.seealso: PetscSectionSymGetType(), PetscSectionSymCreate()
@*/
PetscErrorCode  PetscSectionSymSetType(PetscSectionSym sym, PetscSectionSymType method)
{
  PetscErrorCode (*r)(PetscSectionSym);
  PetscBool      match;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sym, PETSC_SECTION_SYM_CLASSID,1);
  ierr = PetscObjectTypeCompare((PetscObject) sym, method, &match);CHKERRQ(ierr);
  if (match) PetscFunctionReturn(0);

  ierr = PetscFunctionListFind(PetscSectionSymList,method,&r);CHKERRQ(ierr);
  if (!r) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE, "Unknown PetscSectionSym type: %s", method);
  if (sym->ops->destroy) {
    ierr = (*sym->ops->destroy)(sym);CHKERRQ(ierr);
    sym->ops->destroy = NULL;
  }
  ierr = (*r)(sym);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)sym,method);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


/*@C
  PetscSectionSymGetType - Gets the section symmetry type name (as a string) from the PetscSectionSym.

  Not Collective

  Input Parameter:
. sym  - The section symmetry

  Output Parameter:
. type - The index set type name

  Level: developer

.seealso: PetscSectionSymSetType(), PetscSectionSymCreate()
@*/
PetscErrorCode PetscSectionSymGetType(PetscSectionSym sym, PetscSectionSymType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sym, PETSC_SECTION_SYM_CLASSID,1);
  PetscValidCharPointer(type,2);
  *type = ((PetscObject)sym)->type_name;
  PetscFunctionReturn(0);
}

/*@C
  PetscSectionSymRegister - Adds a new section symmetry implementation

  Not Collective

  Input Parameters:
+ name        - The name of a new user-defined creation routine
- create_func - The creation routine itself

  Notes:
  PetscSectionSymRegister() may be called multiple times to add several user-defined vectors

  Level: developer

.seealso: PetscSectionSymCreate(), PetscSectionSymSetType()
@*/
PetscErrorCode PetscSectionSymRegister(const char sname[], PetscErrorCode (*function)(PetscSectionSym))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = ISInitializePackage();CHKERRQ(ierr);
  ierr = PetscFunctionListAdd(&PetscSectionSymList,sname,function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   PetscSectionSymDestroy - Destroys a section symmetry.

   Collective on PetscSectionSym

   Input Parameters:
.  sym - the section symmetry

   Level: developer

.seealso: PetscSectionSymCreate(), PetscSectionSymDestroy()
@*/
PetscErrorCode PetscSectionSymDestroy(PetscSectionSym *sym)
{
  SymWorkLink    link,next;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*sym) PetscFunctionReturn(0);
  PetscValidHeaderSpecific((*sym),PETSC_SECTION_SYM_CLASSID,1);
  if (--((PetscObject)(*sym))->refct > 0) {*sym = 0; PetscFunctionReturn(0);}
  if ((*sym)->ops->destroy) {
    ierr = (*(*sym)->ops->destroy)(*sym);CHKERRQ(ierr);
  }
  if ((*sym)->workout) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Work array still checked out");
  for (link=(*sym)->workin; link; link=next) {
    next = link->next;
    ierr = PetscFree2(*(PetscInt***)&link->perms,*(PetscScalar***)&link->rots);CHKERRQ(ierr);
    ierr = PetscFree(link);CHKERRQ(ierr);
  }
  (*sym)->workin = NULL;
  ierr = PetscHeaderDestroy(sym);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   PetscSectionSymView - Displays a section symmetry

   Collective on PetscSectionSym

   Input Parameters:
+  sym - the index set
-  viewer - viewer used to display the set, for example PETSC_VIEWER_STDOUT_SELF.

   Level: developer

.seealso: PetscViewerASCIIOpen()
@*/
PetscErrorCode PetscSectionSymView(PetscSectionSym sym,PetscViewer viewer)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sym,PETSC_SECTION_SYM_CLASSID,1);
  if (!viewer) {
    ierr = PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)sym),&viewer);CHKERRQ(ierr);
  }
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  PetscCheckSameComm(sym,1,viewer,2);
  ierr = PetscObjectPrintClassNamePrefixType((PetscObject)sym,viewer);CHKERRQ(ierr);
  if (sym->ops->view) {
    ierr = (*sym->ops->view)(sym,viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@
  PetscSectionSetSym - Set the symmetries for the data referred to by the section

  Collective on PetscSection

  Input Parameters:
+ section - the section describing data layout
- sym - the symmetry describing the affect of orientation on the access of the data

  Level: developer

.seealso: PetscSectionGetSym(), PetscSectionSymCreate()
@*/
PetscErrorCode PetscSectionSetSym(PetscSection section, PetscSectionSym sym)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(section,PETSC_SECTION_CLASSID,1);
  ierr = PetscSectionSymDestroy(&(section->sym));CHKERRQ(ierr);
  if (sym) {
    PetscValidHeaderSpecific(sym,PETSC_SECTION_SYM_CLASSID,2);
    PetscCheckSameComm(section,1,sym,2);
    ierr = PetscObjectReference((PetscObject) sym);CHKERRQ(ierr);
  }
  section->sym = sym;
  PetscFunctionReturn(0);
}

/*@
  PetscSectionGetSym - Get the symmetries for the data referred to by the section

  Not collective

  Input Parameters:
. section - the section describing data layout

  Output Parameters:
. sym - the symmetry describing the affect of orientation on the access of the data

  Level: developer

.seealso: PetscSectionSetSym(), PetscSectionSymCreate()
@*/
PetscErrorCode PetscSectionGetSym(PetscSection section, PetscSectionSym *sym)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(section,PETSC_SECTION_CLASSID,1);
  *sym = section->sym;
  PetscFunctionReturn(0);
}

/*@
  PetscSectionSetFieldSym - Set the symmetries for the data referred to by a field of the section

  Collective on PetscSection

  Input Parameters:
+ section - the section describing data layout
. field - the field number
- sym - the symmetry describing the affect of orientation on the access of the data

  Level: developer

.seealso: PetscSectionGetFieldSym(), PetscSectionSymCreate()
@*/
PetscErrorCode PetscSectionSetFieldSym(PetscSection section, PetscInt field, PetscSectionSym sym)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(section,PETSC_SECTION_CLASSID,1);
  if (field < 0 || field >= section->numFields) SETERRQ2(PetscObjectComm((PetscObject)section),PETSC_ERR_ARG_OUTOFRANGE,"Invalid field number %D (not in [0,%D)", field, section->numFields);
  ierr = PetscSectionSetSym(section->field[field],sym);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  PetscSectionGetFieldSym - Get the symmetries for the data referred to by a field of the section

  Collective on PetscSection

  Input Parameters:
+ section - the section describing data layout
- field - the field number

  Output Parameters:
. sym - the symmetry describing the affect of orientation on the access of the data

  Level: developer

.seealso: PetscSectionSetFieldSym(), PetscSectionSymCreate()
@*/
PetscErrorCode PetscSectionGetFieldSym(PetscSection section, PetscInt field, PetscSectionSym *sym)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(section,PETSC_SECTION_CLASSID,1);
  if (field < 0 || field >= section->numFields) SETERRQ2(PetscObjectComm((PetscObject)section),PETSC_ERR_ARG_OUTOFRANGE,"Invalid field number %D (not in [0,%D)", field, section->numFields);
  *sym = section->field[field]->sym;
  PetscFunctionReturn(0);
}

/*@C
  PetscSectionGetPointSyms - Get the symmetries for a set of points in a PetscSection under specific orientations.

  Not collective

  Input Parameters:
+ section - the section
. numPoints - the number of points
- points - an array of size 2 * numPoints, containing a list of (point, orientation) pairs. (An orientation is an
    arbitrary integer: its interpretation is up to sym.  Orientations are used by DM: for their interpretation in that
    context, see DMPlexGetConeOrientation()).

  Output Parameter:
+ perms - The permutations for the given orientations (or NULL if there is no symmetry or the permutation is the identity).
- rots - The field rotations symmetries for the given orientations (or NULL if there is no symmetry or the rotations are all
    identity).

  Example of usage, gathering dofs into a local array (lArray) from a section array (sArray):
.vb
     const PetscInt    **perms;
     const PetscScalar **rots;
     PetscInt            lOffset;

     PetscSectionGetPointSyms(section,numPoints,points,&perms,&rots);
     for (i = 0, lOffset = 0; i < numPoints; i++) {
       PetscInt           point = points[2*i], dof, sOffset;
       const PetscInt    *perm  = perms ? perms[i] : NULL;
       const PetscScalar *rot   = rots  ? rots[i]  : NULL;

       PetscSectionGetDof(section,point,&dof);
       PetscSectionGetOffset(section,point,&sOffset);

       if (perm) {for (j = 0; j < dof; j++) {lArray[lOffset + perm[j]]  = sArray[sOffset + j];}}
       else      {for (j = 0; j < dof; j++) {lArray[lOffset +      j ]  = sArray[sOffset + j];}}
       if (rot)  {for (j = 0; j < dof; j++) {lArray[lOffset +      j ] *= rot[j];             }}
       lOffset += dof;
     }
     PetscSectionRestorePointSyms(section,numPoints,points,&perms,&rots);
.ve

  Example of usage, adding dofs into a section array (sArray) from a local array (lArray):
.vb
     const PetscInt    **perms;
     const PetscScalar **rots;
     PetscInt            lOffset;

     PetscSectionGetPointSyms(section,numPoints,points,&perms,&rots);
     for (i = 0, lOffset = 0; i < numPoints; i++) {
       PetscInt           point = points[2*i], dof, sOffset;
       const PetscInt    *perm  = perms ? perms[i] : NULL;
       const PetscScalar *rot   = rots  ? rots[i]  : NULL;

       PetscSectionGetDof(section,point,&dof);
       PetscSectionGetOffset(section,point,&sOff);

       if (perm) {for (j = 0; j < dof; j++) {sArray[sOffset + j] += lArray[lOffset + perm[j]] * (rot ? PetscConj(rot[perm[j]]) : 1.);}}
       else      {for (j = 0; j < dof; j++) {sArray[sOffset + j] += lArray[lOffset +      j ] * (rot ? PetscConj(rot[     j ]) : 1.);}}
       offset += dof;
     }
     PetscSectionRestorePointSyms(section,numPoints,points,&perms,&rots);
.ve

  Level: developer

.seealso: PetscSectionRestorePointSyms(), PetscSectionSymCreate(), PetscSectionSetSym(), PetscSectionGetSym()
@*/
PetscErrorCode PetscSectionGetPointSyms(PetscSection section, PetscInt numPoints, const PetscInt *points, const PetscInt ***perms, const PetscScalar ***rots)
{
  PetscSectionSym sym;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(section,PETSC_SECTION_CLASSID,1);
  if (numPoints) PetscValidIntPointer(points,3);
  if (perms) *perms = NULL;
  if (rots)  *rots  = NULL;
  sym = section->sym;
  if (sym && (perms || rots)) {
    SymWorkLink link;

    if (sym->workin) {
      link        = sym->workin;
      sym->workin = sym->workin->next;
    } else {
      ierr = PetscNewLog(sym,&link);CHKERRQ(ierr);
    }
    if (numPoints > link->numPoints) {
      ierr = PetscFree2(*(PetscInt***)&link->perms,*(PetscInt***)&link->rots);CHKERRQ(ierr);
      ierr = PetscMalloc2(numPoints,(PetscInt***)&link->perms,numPoints,(PetscScalar***)&link->rots);CHKERRQ(ierr);
      link->numPoints = numPoints;
    }
    link->next   = sym->workout;
    sym->workout = link;
    ierr = PetscArrayzero((PetscInt**)link->perms,numPoints);CHKERRQ(ierr);
    ierr = PetscArrayzero((PetscInt**)link->rots,numPoints);CHKERRQ(ierr);
    ierr = (*sym->ops->getpoints) (sym, section, numPoints, points, link->perms, link->rots);CHKERRQ(ierr);
    if (perms) *perms = link->perms;
    if (rots)  *rots  = link->rots;
  }
  PetscFunctionReturn(0);
}

/*@C
  PetscSectionRestorePointSyms - Restore the symmetries returned by PetscSectionGetPointSyms()

  Not collective

  Input Parameters:
+ section - the section
. numPoints - the number of points
- points - an array of size 2 * numPoints, containing a list of (point, orientation) pairs. (An orientation is an
    arbitrary integer: its interpretation is up to sym.  Orientations are used by DM: for their interpretation in that
    context, see DMPlexGetConeOrientation()).

  Output Parameter:
+ perms - The permutations for the given orientations: set to NULL at conclusion
- rots - The field rotations symmetries for the given orientations: set to NULL at conclusion

  Level: developer

.seealso: PetscSectionGetPointSyms(), PetscSectionSymCreate(), PetscSectionSetSym(), PetscSectionGetSym()
@*/
PetscErrorCode PetscSectionRestorePointSyms(PetscSection section, PetscInt numPoints, const PetscInt *points, const PetscInt ***perms, const PetscScalar ***rots)
{
  PetscSectionSym sym;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(section,PETSC_SECTION_CLASSID,1);
  sym = section->sym;
  if (sym && (perms || rots)) {
    SymWorkLink *p,link;

    for (p=&sym->workout; (link=*p); p=&link->next) {
      if ((perms && link->perms == *perms) || (rots && link->rots == *rots)) {
        *p          = link->next;
        link->next  = sym->workin;
        sym->workin = link;
        if (perms) *perms = NULL;
        if (rots)  *rots  = NULL;
        PetscFunctionReturn(0);
      }
    }
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Array was not checked out");
  }
  PetscFunctionReturn(0);
}

/*@C
  PetscSectionGetFieldPointSyms - Get the symmetries for a set of points in a field of a PetscSection under specific orientations.

  Not collective

  Input Parameters:
+ section - the section
. field - the field of the section
. numPoints - the number of points
- points - an array of size 2 * numPoints, containing a list of (point, orientation) pairs. (An orientation is an
    arbitrary integer: its interpretation is up to sym.  Orientations are used by DM: for their interpretation in that
    context, see DMPlexGetConeOrientation()).

  Output Parameter:
+ perms - The permutations for the given orientations (or NULL if there is no symmetry or the permutation is the identity).
- rots - The field rotations symmetries for the given orientations (or NULL if there is no symmetry or the rotations are all
    identity).

  Level: developer

.seealso: PetscSectionGetPointSyms(), PetscSectionRestoreFieldPointSyms()
@*/
PetscErrorCode PetscSectionGetFieldPointSyms(PetscSection section, PetscInt field, PetscInt numPoints, const PetscInt *points, const PetscInt ***perms, const PetscScalar ***rots)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(section,PETSC_SECTION_CLASSID,1);
  if (field > section->numFields) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"field %D greater than number of fields (%D) in section",field,section->numFields);
  ierr = PetscSectionGetPointSyms(section->field[field],numPoints,points,perms,rots);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  PetscSectionRestoreFieldPointSyms - Restore the symmetries returned by PetscSectionGetFieldPointSyms()

  Not collective

  Input Parameters:
+ section - the section
. field - the field number
. numPoints - the number of points
- points - an array of size 2 * numPoints, containing a list of (point, orientation) pairs. (An orientation is an
    arbitrary integer: its interpretation is up to sym.  Orientations are used by DM: for their interpretation in that
    context, see DMPlexGetConeOrientation()).

  Output Parameter:
+ perms - The permutations for the given orientations: set to NULL at conclusion
- rots - The field rotations symmetries for the given orientations: set to NULL at conclusion

  Level: developer

.seealso: PetscSectionRestorePointSyms(), petscSectionGetFieldPointSyms(), PetscSectionSymCreate(), PetscSectionSetSym(), PetscSectionGetSym()
@*/
PetscErrorCode PetscSectionRestoreFieldPointSyms(PetscSection section, PetscInt field, PetscInt numPoints, const PetscInt *points, const PetscInt ***perms, const PetscScalar ***rots)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(section,PETSC_SECTION_CLASSID,1);
  if (field > section->numFields) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"field %D greater than number of fields (%D) in section",field,section->numFields);
  ierr = PetscSectionRestorePointSyms(section->field[field],numPoints,points,perms,rots);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  PetscSectionGetUseFieldOffsets - Get the flag to use field offsets directly in a global section, rather than just the point offset

  Not collective

  Input Parameter:
. s - the global PetscSection

  Output Parameters:
. flg - the flag

  Level: developer

.seealso: PetscSectionSetChart(), PetscSectionCreate()
@*/
PetscErrorCode PetscSectionGetUseFieldOffsets(PetscSection s, PetscBool *flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(s, PETSC_SECTION_CLASSID, 1);
  *flg = s->useFieldOff;
  PetscFunctionReturn(0);
}

/*@
  PetscSectionSetUseFieldOffsets - Set the flag to use field offsets directly in a global section, rather than just the point offset

  Not collective

  Input Parameters:
+ s   - the global PetscSection
- flg - the flag

  Level: developer

.seealso: PetscSectionGetUseFieldOffsets(), PetscSectionSetChart(), PetscSectionCreate()
@*/
PetscErrorCode PetscSectionSetUseFieldOffsets(PetscSection s, PetscBool flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(s, PETSC_SECTION_CLASSID, 1);
  s->useFieldOff = flg;
  PetscFunctionReturn(0);
}

#define PetscSectionExpandPoints_Loop(TYPE) \
{ \
  PetscInt i, n, o0, o1, size; \
  TYPE *a0 = (TYPE*)origArray, *a1; \
  ierr = PetscSectionGetStorageSize(s, &size);CHKERRQ(ierr); \
  ierr = PetscMalloc1(size, &a1);CHKERRQ(ierr); \
  for (i=0; i<npoints; i++) { \
    ierr = PetscSectionGetOffset(origSection, points_[i], &o0);CHKERRQ(ierr); \
    ierr = PetscSectionGetOffset(s, i, &o1);CHKERRQ(ierr); \
    ierr = PetscSectionGetDof(s, i, &n);CHKERRQ(ierr); \
    ierr = PetscMemcpy(&a1[o1], &a0[o0], n*unitsize);CHKERRQ(ierr); \
  } \
  *newArray = (void*)a1; \
}

/*@
  PetscSectionExtractDofsFromArray - Extracts elements of an array corresponding to DOFs of specified points.

  Not collective

  Input Parameters:
+ origSection - the PetscSection describing the layout of the array
. dataType - MPI_Datatype describing the data type of the array (currently only MPIU_INT, MPIU_SCALAR, MPIU_REAL)
. origArray - the array; its size must be equal to the storage size of origSection
- points - IS with points to extract; its indices must lie in the chart of origSection

  Output Parameters:
+ newSection - the new PetscSection desribing the layout of the new array (with points renumbered 0,1,... but preserving numbers of DOFs)
- newArray - the array of the extracted DOFs; its size is the storage size of newSection

  Level: developer

.seealso: PetscSectionGetChart(), PetscSectionGetDof(), PetscSectionGetStorageSize(), PetscSectionCreate()
@*/
PetscErrorCode PetscSectionExtractDofsFromArray(PetscSection origSection, MPI_Datatype dataType, const void *origArray, IS points, PetscSection *newSection, void *newArray[])
{
  PetscSection        s;
  const PetscInt      *points_;
  PetscInt            i, n, npoints, pStart, pEnd;
  PetscMPIInt         unitsize;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(origSection, PETSC_SECTION_CLASSID, 1);
  PetscValidPointer(origArray, 3);
  PetscValidHeaderSpecific(points, IS_CLASSID, 4);
  if (newSection) PetscValidPointer(newSection, 5);
  if (newArray) PetscValidPointer(newArray, 6);
  ierr = MPI_Type_size(dataType, &unitsize);CHKERRQ(ierr);
  ierr = ISGetLocalSize(points, &npoints);CHKERRQ(ierr);
  ierr = ISGetIndices(points, &points_);CHKERRQ(ierr);
  ierr = PetscSectionGetChart(origSection, &pStart, &pEnd);CHKERRQ(ierr);
  ierr = PetscSectionCreate(PETSC_COMM_SELF, &s);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(s, 0, npoints);CHKERRQ(ierr);
  for (i=0; i<npoints; i++) {
    if (PetscUnlikely(points_[i] < pStart || points_[i] >= pEnd)) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "point %d (index %d) in input IS out of input section's chart", points_[i], i);
    ierr = PetscSectionGetDof(origSection, points_[i], &n);CHKERRQ(ierr);
    ierr = PetscSectionSetDof(s, i, n);CHKERRQ(ierr);
  }
  ierr = PetscSectionSetUp(s);CHKERRQ(ierr);
  if (newArray) {
    if (dataType == MPIU_INT)           {PetscSectionExpandPoints_Loop(PetscInt);}
    else if (dataType == MPIU_SCALAR)   {PetscSectionExpandPoints_Loop(PetscScalar);}
    else if (dataType == MPIU_REAL)     {PetscSectionExpandPoints_Loop(PetscReal);}
    else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "not implemented for this MPI_Datatype");
  }
  if (newSection) {
    *newSection = s;
  } else {
    ierr = PetscSectionDestroy(&s);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
