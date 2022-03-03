/*
   This file contains routines for basic section object implementation.
*/

#include <petsc/private/sectionimpl.h>   /*I  "petscsection.h"   I*/
#include <petscsf.h>

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
  PetscFunctionBegin;
  PetscValidPointer(s,2);
  CHKERRQ(ISInitializePackage());

  CHKERRQ(PetscHeaderCreate(*s,PETSC_SECTION_CLASSID,"PetscSection","Section","IS",comm,PetscSectionDestroy,PetscSectionView));

  (*s)->pStart              = -1;
  (*s)->pEnd                = -1;
  (*s)->perm                = NULL;
  (*s)->pointMajor          = PETSC_TRUE;
  (*s)->includesConstraints = PETSC_TRUE;
  (*s)->maxDof              = 0;
  (*s)->atlasDof            = NULL;
  (*s)->atlasOff            = NULL;
  (*s)->bc                  = NULL;
  (*s)->bcIndices           = NULL;
  (*s)->setup               = PETSC_FALSE;
  (*s)->numFields           = 0;
  (*s)->fieldNames          = NULL;
  (*s)->field               = NULL;
  (*s)->useFieldOff         = PETSC_FALSE;
  (*s)->compNames           = NULL;
  (*s)->clObj               = NULL;
  (*s)->clHash              = NULL;
  (*s)->clSection           = NULL;
  (*s)->clPoints            = NULL;
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

  PetscFunctionBegin;
  PetscValidHeaderSpecific(section, PETSC_SECTION_CLASSID, 1);
  PetscValidHeaderSpecific(newSection, PETSC_SECTION_CLASSID, 2);
  CHKERRQ(PetscSectionReset(newSection));
  CHKERRQ(PetscSectionGetNumFields(section, &numFields));
  if (numFields) CHKERRQ(PetscSectionSetNumFields(newSection, numFields));
  for (f = 0; f < numFields; ++f) {
    const char *fieldName = NULL, *compName = NULL;
    PetscInt   numComp = 0;

    CHKERRQ(PetscSectionGetFieldName(section, f, &fieldName));
    CHKERRQ(PetscSectionSetFieldName(newSection, f, fieldName));
    CHKERRQ(PetscSectionGetFieldComponents(section, f, &numComp));
    CHKERRQ(PetscSectionSetFieldComponents(newSection, f, numComp));
    for (c = 0; c < numComp; ++c) {
      CHKERRQ(PetscSectionGetComponentName(section, f, c, &compName));
      CHKERRQ(PetscSectionSetComponentName(newSection, f, c, compName));
    }
    CHKERRQ(PetscSectionGetFieldSym(section, f, &sym));
    CHKERRQ(PetscSectionSetFieldSym(newSection, f, sym));
  }
  CHKERRQ(PetscSectionGetChart(section, &pStart, &pEnd));
  CHKERRQ(PetscSectionSetChart(newSection, pStart, pEnd));
  CHKERRQ(PetscSectionGetPermutation(section, &perm));
  CHKERRQ(PetscSectionSetPermutation(newSection, perm));
  CHKERRQ(PetscSectionGetSym(section, &sym));
  CHKERRQ(PetscSectionSetSym(newSection, sym));
  for (p = pStart; p < pEnd; ++p) {
    PetscInt dof, cdof, fcdof = 0;

    CHKERRQ(PetscSectionGetDof(section, p, &dof));
    CHKERRQ(PetscSectionSetDof(newSection, p, dof));
    CHKERRQ(PetscSectionGetConstraintDof(section, p, &cdof));
    if (cdof) CHKERRQ(PetscSectionSetConstraintDof(newSection, p, cdof));
    for (f = 0; f < numFields; ++f) {
      CHKERRQ(PetscSectionGetFieldDof(section, p, f, &dof));
      CHKERRQ(PetscSectionSetFieldDof(newSection, p, f, dof));
      if (cdof) {
        CHKERRQ(PetscSectionGetFieldConstraintDof(section, p, f, &fcdof));
        if (fcdof) CHKERRQ(PetscSectionSetFieldConstraintDof(newSection, p, f, fcdof));
      }
    }
  }
  CHKERRQ(PetscSectionSetUp(newSection));
  for (p = pStart; p < pEnd; ++p) {
    PetscInt       off, cdof, fcdof = 0;
    const PetscInt *cInd;

    /* Must set offsets in case they do not agree with the prefix sums */
    CHKERRQ(PetscSectionGetOffset(section, p, &off));
    CHKERRQ(PetscSectionSetOffset(newSection, p, off));
    CHKERRQ(PetscSectionGetConstraintDof(section, p, &cdof));
    if (cdof) {
      CHKERRQ(PetscSectionGetConstraintIndices(section, p, &cInd));
      CHKERRQ(PetscSectionSetConstraintIndices(newSection, p, cInd));
      for (f = 0; f < numFields; ++f) {
        CHKERRQ(PetscSectionGetFieldConstraintDof(section, p, f, &fcdof));
        if (fcdof) {
          CHKERRQ(PetscSectionGetFieldConstraintIndices(section, p, f, &cInd));
          CHKERRQ(PetscSectionSetFieldConstraintIndices(newSection, p, f, cInd));
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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(section, PETSC_SECTION_CLASSID, 1);
  PetscValidPointer(newSection, 2);
  CHKERRQ(PetscSectionCreate(PetscObjectComm((PetscObject) section), newSection));
  CHKERRQ(PetscSectionCopy(section, *newSection));
  PetscFunctionReturn(0);
}

/*@
  PetscSectionSetFromOptions - sets parameters in a PetscSection from the options database

  Collective on PetscSection

  Input Parameter:
. section - the PetscSection

  Options Database:
. -petscsection_point_major - PETSC_TRUE for point-major order

  Level: intermediate

.seealso: PetscSection, PetscSectionCreate(), PetscSectionDestroy()
@*/
PetscErrorCode PetscSectionSetFromOptions(PetscSection s)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(s, PETSC_SECTION_CLASSID, 1);
  ierr = PetscObjectOptionsBegin((PetscObject) s);CHKERRQ(ierr);
  CHKERRQ(PetscOptionsBool("-petscsection_point_major", "The for ordering, either point major or field major", "PetscSectionSetPointMajor", s->pointMajor, &s->pointMajor, NULL));
  /* process any options handlers added with PetscObjectAddOptionsHandler() */
  CHKERRQ(PetscObjectProcessOptionsHandlers(PetscOptionsObject,(PetscObject) s));
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  CHKERRQ(PetscObjectViewFromOptions((PetscObject) s, NULL, "-petscsection_view"));
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

  PetscFunctionBegin;
  PetscValidHeaderSpecific(s1, PETSC_SECTION_CLASSID, 1);
  PetscValidHeaderSpecific(s2, PETSC_SECTION_CLASSID, 2);
  PetscValidBoolPointer(congruent,3);
  flg = PETSC_FALSE;

  CHKERRMPI(MPI_Comm_compare(PetscObjectComm((PetscObject)s1),PetscObjectComm((PetscObject)s2),&mflg));
  if (mflg != MPI_CONGRUENT && mflg != MPI_IDENT) {
    *congruent = PETSC_FALSE;
    PetscFunctionReturn(0);
  }

  CHKERRQ(PetscSectionGetChart(s1, &pStart, &pEnd));
  CHKERRQ(PetscSectionGetChart(s2, &n1, &n2));
  if (pStart != n1 || pEnd != n2) goto not_congruent;

  CHKERRQ(PetscSectionGetPermutation(s1, &perm1));
  CHKERRQ(PetscSectionGetPermutation(s2, &perm2));
  if (perm1 && perm2) {
    CHKERRQ(ISEqual(perm1, perm2, congruent));
    if (!(*congruent)) goto not_congruent;
  } else if (perm1 != perm2) goto not_congruent;

  for (p = pStart; p < pEnd; ++p) {
    CHKERRQ(PetscSectionGetOffset(s1, p, &n1));
    CHKERRQ(PetscSectionGetOffset(s2, p, &n2));
    if (n1 != n2) goto not_congruent;

    CHKERRQ(PetscSectionGetDof(s1, p, &n1));
    CHKERRQ(PetscSectionGetDof(s2, p, &n2));
    if (n1 != n2) goto not_congruent;

    CHKERRQ(PetscSectionGetConstraintDof(s1, p, &ncdof));
    CHKERRQ(PetscSectionGetConstraintDof(s2, p, &n2));
    if (ncdof != n2) goto not_congruent;

    CHKERRQ(PetscSectionGetConstraintIndices(s1, p, &idx1));
    CHKERRQ(PetscSectionGetConstraintIndices(s2, p, &idx2));
    CHKERRQ(PetscArraycmp(idx1, idx2, ncdof, congruent));
    if (!(*congruent)) goto not_congruent;
  }

  CHKERRQ(PetscSectionGetNumFields(s1, &nfields));
  CHKERRQ(PetscSectionGetNumFields(s2, &n2));
  if (nfields != n2) goto not_congruent;

  for (f = 0; f < nfields; ++f) {
    CHKERRQ(PetscSectionGetFieldComponents(s1, f, &n1));
    CHKERRQ(PetscSectionGetFieldComponents(s2, f, &n2));
    if (n1 != n2) goto not_congruent;

    for (p = pStart; p < pEnd; ++p) {
      CHKERRQ(PetscSectionGetFieldOffset(s1, p, f, &n1));
      CHKERRQ(PetscSectionGetFieldOffset(s2, p, f, &n2));
      if (n1 != n2) goto not_congruent;

      CHKERRQ(PetscSectionGetFieldDof(s1, p, f, &n1));
      CHKERRQ(PetscSectionGetFieldDof(s2, p, f, &n2));
      if (n1 != n2) goto not_congruent;

      CHKERRQ(PetscSectionGetFieldConstraintDof(s1, p, f, &nfcdof));
      CHKERRQ(PetscSectionGetFieldConstraintDof(s2, p, f, &n2));
      if (nfcdof != n2) goto not_congruent;

      CHKERRQ(PetscSectionGetFieldConstraintIndices(s1, p, f, &idx1));
      CHKERRQ(PetscSectionGetFieldConstraintIndices(s2, p, f, &idx2));
      CHKERRQ(PetscArraycmp(idx1, idx2, nfcdof, congruent));
      if (!(*congruent)) goto not_congruent;
    }
  }

  flg = PETSC_TRUE;
not_congruent:
  CHKERRMPI(MPIU_Allreduce(&flg,congruent,1,MPIU_BOOL,MPI_LAND,PetscObjectComm((PetscObject)s1)));
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

  PetscFunctionBegin;
  PetscValidHeaderSpecific(s, PETSC_SECTION_CLASSID, 1);
  PetscCheckFalse(numFields <= 0,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "The number of fields %" PetscInt_FMT " must be positive", numFields);
  CHKERRQ(PetscSectionReset(s));

  s->numFields = numFields;
  CHKERRQ(PetscMalloc1(s->numFields, &s->numFieldComponents));
  CHKERRQ(PetscMalloc1(s->numFields, &s->fieldNames));
  CHKERRQ(PetscMalloc1(s->numFields, &s->compNames));
  CHKERRQ(PetscMalloc1(s->numFields, &s->field));
  for (f = 0; f < s->numFields; ++f) {
    char name[64];

    s->numFieldComponents[f] = 1;

    CHKERRQ(PetscSectionCreate(PetscObjectComm((PetscObject) s), &s->field[f]));
    CHKERRQ(PetscSNPrintf(name, 64, "Field_%" PetscInt_FMT, f));
    CHKERRQ(PetscStrallocpy(name, (char **) &s->fieldNames[f]));
    CHKERRQ(PetscSNPrintf(name, 64, "Component_0"));
    CHKERRQ(PetscMalloc1(s->numFieldComponents[f], &s->compNames[f]));
    CHKERRQ(PetscStrallocpy(name, (char **) &s->compNames[f][0]));
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
  PetscSectionCheckValidField(field,s->numFields);
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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(s, PETSC_SECTION_CLASSID, 1);
  if (fieldName) PetscValidCharPointer(fieldName, 3);
  PetscSectionCheckValidField(field,s->numFields);
  CHKERRQ(PetscFree(s->fieldNames[field]));
  CHKERRQ(PetscStrallocpy(fieldName, (char**) &s->fieldNames[field]));
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
  PetscValidPointer(compName, 4);
  PetscSectionCheckValidField(field,s->numFields);
  PetscSectionCheckValidFieldComponent(comp,s->numFieldComponents[field]);
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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(s, PETSC_SECTION_CLASSID, 1);
  if (compName) PetscValidCharPointer(compName, 4);
  PetscSectionCheckValidField(field,s->numFields);
  PetscSectionCheckValidFieldComponent(comp,s->numFieldComponents[field]);
  CHKERRQ(PetscFree(s->compNames[field][comp]));
  CHKERRQ(PetscStrallocpy(compName, (char**) &s->compNames[field][comp]));
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
  PetscValidIntPointer(numComp, 3);
  PetscSectionCheckValidField(field,s->numFields);
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
  PetscInt       c;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(s, PETSC_SECTION_CLASSID, 1);
  PetscSectionCheckValidField(field,s->numFields);
  if (s->compNames) {
    for (c = 0; c < s->numFieldComponents[field]; ++c) {
      CHKERRQ(PetscFree(s->compNames[field][c]));
    }
    CHKERRQ(PetscFree(s->compNames[field]));
  }

  s->numFieldComponents[field] = numComp;
  if (numComp) {
    CHKERRQ(PetscMalloc1(numComp, (char ***) &s->compNames[field]));
    for (c = 0; c < numComp; ++c) {
      char name[64];

      CHKERRQ(PetscSNPrintf(name, 64, "%" PetscInt_FMT, c));
      CHKERRQ(PetscStrallocpy(name, (char **) &s->compNames[field][c]));
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSectionCheckConstraints_Static(PetscSection s)
{
  PetscFunctionBegin;
  if (!s->bc) {
    CHKERRQ(PetscSectionCreate(PETSC_COMM_SELF, &s->bc));
    CHKERRQ(PetscSectionSetChart(s->bc, s->pStart, s->pEnd));
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

  PetscFunctionBegin;
  PetscValidHeaderSpecific(s, PETSC_SECTION_CLASSID, 1);
  if (pStart == s->pStart && pEnd == s->pEnd) PetscFunctionReturn(0);
  /* Cannot Reset() because it destroys field information */
  s->setup = PETSC_FALSE;
  CHKERRQ(PetscSectionDestroy(&s->bc));
  CHKERRQ(PetscFree(s->bcIndices));
  CHKERRQ(PetscFree2(s->atlasDof, s->atlasOff));

  s->pStart = pStart;
  s->pEnd   = pEnd;
  CHKERRQ(PetscMalloc2((pEnd - pStart), &s->atlasDof, (pEnd - pStart), &s->atlasOff));
  CHKERRQ(PetscArrayzero(s->atlasDof, pEnd - pStart));
  for (f = 0; f < s->numFields; ++f) {
    CHKERRQ(PetscSectionSetChart(s->field[f], pStart, pEnd));
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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(s, PETSC_SECTION_CLASSID, 1);
  if (perm) PetscValidHeaderSpecific(perm, IS_CLASSID, 2);
  PetscCheck(!s->setup,PetscObjectComm((PetscObject) s), PETSC_ERR_ARG_WRONGSTATE, "Cannot set a permutation after the section is setup");
  if (s->perm != perm) {
    CHKERRQ(ISDestroy(&s->perm));
    if (perm) {
      s->perm = perm;
      CHKERRQ(PetscObjectReference((PetscObject) s->perm));
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
  PetscCheck(!s->setup,PetscObjectComm((PetscObject) s), PETSC_ERR_ARG_WRONGSTATE, "Cannot set the dof ordering after the section is setup");
  s->pointMajor = pm;
  PetscFunctionReturn(0);
}

/*@
  PetscSectionGetIncludesConstraints - Returns the flag indicating if constrained dofs were included when computing offsets

  Not collective

  Input Parameter:
. s - the PetscSection

  Output Parameter:
. includesConstraints - the flag indicating if constrained dofs were included when computing offsets

  Level: intermediate

.seealso: PetscSectionSetIncludesConstraints()
@*/
PetscErrorCode PetscSectionGetIncludesConstraints(PetscSection s, PetscBool *includesConstraints)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(s, PETSC_SECTION_CLASSID, 1);
  PetscValidBoolPointer(includesConstraints,2);
  *includesConstraints = s->includesConstraints;
  PetscFunctionReturn(0);
}

/*@
  PetscSectionSetIncludesConstraints - Sets the flag indicating if constrained dofs are to be included when computing offsets

  Not collective

  Input Parameters:
+ s  - the PetscSection
- includesConstraints - the flag indicating if constrained dofs are to be included when computing offsets

  Not collective

  Level: intermediate

.seealso: PetscSectionGetIncludesConstraints()
@*/
PetscErrorCode PetscSectionSetIncludesConstraints(PetscSection s, PetscBool includesConstraints)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(s, PETSC_SECTION_CLASSID, 1);
  PetscCheck(!s->setup,PetscObjectComm((PetscObject) s), PETSC_ERR_ARG_WRONGSTATE, "Cannot set includesConstraints after the section is set up");
  s->includesConstraints = includesConstraints;
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
    PetscCheckFalse((point < s->pStart) || (point >= s->pEnd),PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Section point %" PetscInt_FMT " should be in [%" PetscInt_FMT ", %" PetscInt_FMT ")", point, s->pStart, s->pEnd);
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
  PetscCheckFalse((point < s->pStart) || (point >= s->pEnd),PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Section point %" PetscInt_FMT " should be in [%" PetscInt_FMT ", %" PetscInt_FMT ")", point, s->pStart, s->pEnd);
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
    PetscCheckFalse((point < s->pStart) || (point >= s->pEnd),PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Section point %" PetscInt_FMT " should be in [%" PetscInt_FMT ", %" PetscInt_FMT ")", point, s->pStart, s->pEnd);
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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(s, PETSC_SECTION_CLASSID, 1);
  PetscValidIntPointer(numDof,4);
  PetscSectionCheckValidField(field,s->numFields);
  CHKERRQ(PetscSectionGetDof(s->field[field], point, numDof));
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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(s, PETSC_SECTION_CLASSID, 1);
  PetscSectionCheckValidField(field,s->numFields);
  CHKERRQ(PetscSectionSetDof(s->field[field], point, numDof));
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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(s, PETSC_SECTION_CLASSID, 1);
  PetscSectionCheckValidField(field,s->numFields);
  CHKERRQ(PetscSectionAddDof(s->field[field], point, numDof));
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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(s, PETSC_SECTION_CLASSID, 1);
  PetscValidPointer(numDof, 3);
  if (s->bc) {
    CHKERRQ(PetscSectionGetDof(s->bc, point, numDof));
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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(s, PETSC_SECTION_CLASSID, 1);
  if (numDof) {
    CHKERRQ(PetscSectionCheckConstraints_Static(s));
    CHKERRQ(PetscSectionSetDof(s->bc, point, numDof));
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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(s, PETSC_SECTION_CLASSID, 1);
  if (numDof) {
    CHKERRQ(PetscSectionCheckConstraints_Static(s));
    CHKERRQ(PetscSectionAddDof(s->bc, point, numDof));
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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(s, PETSC_SECTION_CLASSID, 1);
  PetscValidIntPointer(numDof, 4);
  PetscSectionCheckValidField(field,s->numFields);
  CHKERRQ(PetscSectionGetConstraintDof(s->field[field], point, numDof));
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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(s, PETSC_SECTION_CLASSID, 1);
  PetscSectionCheckValidField(field,s->numFields);
  CHKERRQ(PetscSectionSetConstraintDof(s->field[field], point, numDof));
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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(s, PETSC_SECTION_CLASSID, 1);
  PetscSectionCheckValidField(field,s->numFields);
  CHKERRQ(PetscSectionAddConstraintDof(s->field[field], point, numDof));
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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(s, PETSC_SECTION_CLASSID, 1);
  if (s->bc) {
    const PetscInt last = (s->bc->pEnd-s->bc->pStart) - 1;

    CHKERRQ(PetscSectionSetUp(s->bc));
    CHKERRQ(PetscMalloc1(last >= 0 ? s->bc->atlasOff[last] + s->bc->atlasDof[last] : 0, &s->bcIndices));
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

  PetscFunctionBegin;
  PetscValidHeaderSpecific(s, PETSC_SECTION_CLASSID, 1);
  if (s->setup) PetscFunctionReturn(0);
  s->setup = PETSC_TRUE;
  /* Set offsets and field offsets for all points */
  /*   Assume that all fields have the same chart */
  PetscCheck(s->includesConstraints,PETSC_COMM_SELF,PETSC_ERR_SUP,"PetscSectionSetUp is currently unsupported for includesConstraints = PETSC_TRUE");
  if (s->perm) CHKERRQ(ISGetIndices(s->perm, &pind));
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
  if (s->perm) CHKERRQ(ISRestoreIndices(s->perm, &pind));
  /* Setup BC sections */
  CHKERRQ(PetscSectionSetUpBC(s));
  for (f = 0; f < s->numFields; ++f) CHKERRQ(PetscSectionSetUpBC(s->field[f]));
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
  PetscInt        pStart, pEnd, p, dof, cdof, off, foff, globalOff = 0, nroots, nlocal, maxleaf;
  PetscInt        numFields, f, numComponents;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(s, PETSC_SECTION_CLASSID, 1);
  PetscValidHeaderSpecific(sf, PETSCSF_CLASSID, 2);
  PetscValidLogicalCollectiveBool(s, includeConstraints, 3);
  PetscValidLogicalCollectiveBool(s, localOffsets, 4);
  PetscValidPointer(gsection, 5);
  PetscCheck(s->pointMajor,PETSC_COMM_SELF,PETSC_ERR_SUP, "No support for field major ordering");
  CHKERRQ(PetscSectionCreate(PetscObjectComm((PetscObject) s), &gs));
  CHKERRQ(PetscSectionGetNumFields(s, &numFields));
  if (numFields > 0) CHKERRQ(PetscSectionSetNumFields(gs, numFields));
  CHKERRQ(PetscSectionGetChart(s, &pStart, &pEnd));
  CHKERRQ(PetscSectionSetChart(gs, pStart, pEnd));
  gs->includesConstraints = includeConstraints;
  CHKERRQ(PetscSFGetGraph(sf, &nroots, NULL, NULL, NULL));
  nlocal = nroots;              /* The local/leaf space matches global/root space */
  /* Must allocate for all points visible to SF, which may be more than this section */
  if (nroots >= 0) {             /* nroots < 0 means that the graph has not been set, only happens in serial */
    CHKERRQ(PetscSFGetLeafRange(sf, NULL, &maxleaf));
    PetscCheckFalse(nroots < pEnd,PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "SF roots %" PetscInt_FMT " < pEnd %" PetscInt_FMT, nroots, pEnd);
    PetscCheckFalse(maxleaf >= nroots,PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Max local leaf %" PetscInt_FMT " >= nroots %" PetscInt_FMT, maxleaf, nroots);
    CHKERRQ(PetscMalloc2(nroots,&neg,nlocal,&recv));
    CHKERRQ(PetscArrayzero(neg,nroots));
  }
  /* Mark all local points with negative dof */
  for (p = pStart; p < pEnd; ++p) {
    CHKERRQ(PetscSectionGetDof(s, p, &dof));
    CHKERRQ(PetscSectionSetDof(gs, p, dof));
    CHKERRQ(PetscSectionGetConstraintDof(s, p, &cdof));
    if (!includeConstraints && cdof > 0) CHKERRQ(PetscSectionSetConstraintDof(gs, p, cdof));
    if (neg) neg[p] = -(dof+1);
  }
  CHKERRQ(PetscSectionSetUpBC(gs));
  if (gs->bcIndices) CHKERRQ(PetscArraycpy(gs->bcIndices, s->bcIndices, gs->bc->atlasOff[gs->bc->pEnd-gs->bc->pStart-1] + gs->bc->atlasDof[gs->bc->pEnd-gs->bc->pStart-1]));
  if (nroots >= 0) {
    CHKERRQ(PetscArrayzero(recv,nlocal));
    CHKERRQ(PetscSFBcastBegin(sf, MPIU_INT, neg, recv,MPI_REPLACE));
    CHKERRQ(PetscSFBcastEnd(sf, MPIU_INT, neg, recv,MPI_REPLACE));
    for (p = pStart; p < pEnd; ++p) {
      if (recv[p] < 0) {
        gs->atlasDof[p-pStart] = recv[p];
        CHKERRQ(PetscSectionGetDof(s, p, &dof));
        PetscCheckFalse(-(recv[p]+1) != dof,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Global dof %" PetscInt_FMT " for point %" PetscInt_FMT " is not the unconstrained %" PetscInt_FMT, -(recv[p]+1), p, dof);
      }
    }
  }
  /* Calculate new sizes, get process offset, and calculate point offsets */
  if (s->perm) CHKERRQ(ISGetIndices(s->perm, &pind));
  for (p = 0, off = 0; p < pEnd-pStart; ++p) {
    const PetscInt q = pind ? pind[p] : p;

    cdof = (!includeConstraints && s->bc) ? s->bc->atlasDof[q] : 0;
    gs->atlasOff[q] = off;
    off += gs->atlasDof[q] > 0 ? gs->atlasDof[q]-cdof : 0;
  }
  if (!localOffsets) {
    CHKERRMPI(MPI_Scan(&off, &globalOff, 1, MPIU_INT, MPI_SUM, PetscObjectComm((PetscObject) s)));
    globalOff -= off;
  }
  for (p = pStart, off = 0; p < pEnd; ++p) {
    gs->atlasOff[p-pStart] += globalOff;
    if (neg) neg[p] = -(gs->atlasOff[p-pStart]+1);
  }
  if (s->perm) CHKERRQ(ISRestoreIndices(s->perm, &pind));
  /* Put in negative offsets for ghost points */
  if (nroots >= 0) {
    CHKERRQ(PetscArrayzero(recv,nlocal));
    CHKERRQ(PetscSFBcastBegin(sf, MPIU_INT, neg, recv,MPI_REPLACE));
    CHKERRQ(PetscSFBcastEnd(sf, MPIU_INT, neg, recv,MPI_REPLACE));
    for (p = pStart; p < pEnd; ++p) {
      if (recv[p] < 0) gs->atlasOff[p-pStart] = recv[p];
    }
  }
  CHKERRQ(PetscFree2(neg,recv));
  /* Set field dofs/offsets/constraints */
  for (f = 0; f < numFields; ++f) {
    gs->field[f]->includesConstraints = includeConstraints;
    CHKERRQ(PetscSectionGetFieldComponents(s, f, &numComponents));
    CHKERRQ(PetscSectionSetFieldComponents(gs, f, numComponents));
  }
  for (p = pStart; p < pEnd; ++p) {
    CHKERRQ(PetscSectionGetOffset(gs, p, &off));
    for (f = 0, foff = off; f < numFields; ++f) {
      CHKERRQ(PetscSectionGetFieldConstraintDof(s, p, f, &cdof));
      if (!includeConstraints && cdof > 0) CHKERRQ(PetscSectionSetFieldConstraintDof(gs, p, f, cdof));
      CHKERRQ(PetscSectionGetFieldDof(s, p, f, &dof));
      CHKERRQ(PetscSectionSetFieldDof(gs, p, f, off < 0 ? -(dof + 1) : dof));
      CHKERRQ(PetscSectionSetFieldOffset(gs, p, f, foff));
      CHKERRQ(PetscSectionGetFieldConstraintDof(gs, p, f, &cdof));
      foff = off < 0 ? foff - (dof - cdof) : foff + (dof - cdof);
    }
  }
  for (f = 0; f < numFields; ++f) {
    PetscSection gfs = gs->field[f];

    CHKERRQ(PetscSectionSetUpBC(gfs));
    if (gfs->bcIndices) CHKERRQ(PetscArraycpy(gfs->bcIndices, s->field[f]->bcIndices, gfs->bc->atlasOff[gfs->bc->pEnd-gfs->bc->pStart-1] + gfs->bc->atlasDof[gfs->bc->pEnd-gfs->bc->pStart-1]));
  }
  gs->setup = PETSC_TRUE;
  CHKERRQ(PetscSectionViewFromOptions(gs, NULL, "-global_section_view"));
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

  PetscFunctionBegin;
  PetscValidHeaderSpecific(s, PETSC_SECTION_CLASSID, 1);
  PetscValidHeaderSpecific(sf, PETSCSF_CLASSID, 2);
  PetscValidPointer(gsection, 6);
  CHKERRQ(PetscSectionCreate(PetscObjectComm((PetscObject) s), gsection));
  CHKERRQ(PetscSectionGetChart(s, &pStart, &pEnd));
  CHKERRQ(PetscSectionSetChart(*gsection, pStart, pEnd));
  CHKERRQ(PetscSFGetGraph(sf, &nroots, NULL, NULL, NULL));
  if (nroots >= 0) {
    PetscCheckFalse(nroots < pEnd-pStart,PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "PetscSF nroots %" PetscInt_FMT " < %" PetscInt_FMT " section size", nroots, pEnd-pStart);
    CHKERRQ(PetscCalloc1(nroots, &neg));
    if (nroots > pEnd-pStart) {
      CHKERRQ(PetscCalloc1(nroots, &tmpOff));
    } else {
      tmpOff = &(*gsection)->atlasDof[-pStart];
    }
  }
  /* Mark ghost points with negative dof */
  for (p = pStart; p < pEnd; ++p) {
    for (e = 0; e < numExcludes; ++e) {
      if ((p >= excludes[e*2+0]) && (p < excludes[e*2+1])) {
        CHKERRQ(PetscSectionSetDof(*gsection, p, 0));
        break;
      }
    }
    if (e < numExcludes) continue;
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
    if (nroots > pEnd - pStart) {
      for (p = pStart; p < pEnd; ++p) {if (tmpOff[p] < 0) (*gsection)->atlasDof[p-pStart] = tmpOff[p];}
    }
  }
  /* Calculate new sizes, get proccess offset, and calculate point offsets */
  if (s->perm) CHKERRQ(ISGetIndices(s->perm, &pind));
  for (p = 0, off = 0; p < pEnd-pStart; ++p) {
    const PetscInt q = pind ? pind[p] : p;

    cdof = (!includeConstraints && s->bc) ? s->bc->atlasDof[q] : 0;
    (*gsection)->atlasOff[q] = off;
    off += (*gsection)->atlasDof[q] > 0 ? (*gsection)->atlasDof[q]-cdof : 0;
  }
  CHKERRMPI(MPI_Scan(&off, &globalOff, 1, MPIU_INT, MPI_SUM, PetscObjectComm((PetscObject) s)));
  globalOff -= off;
  for (p = 0, off = 0; p < pEnd-pStart; ++p) {
    (*gsection)->atlasOff[p] += globalOff;
    if (neg) neg[p+pStart] = -((*gsection)->atlasOff[p]+1);
  }
  if (s->perm) CHKERRQ(ISRestoreIndices(s->perm, &pind));
  /* Put in negative offsets for ghost points */
  if (nroots >= 0) {
    if (nroots == pEnd-pStart) tmpOff = &(*gsection)->atlasOff[-pStart];
    CHKERRQ(PetscSFBcastBegin(sf, MPIU_INT, neg, tmpOff,MPI_REPLACE));
    CHKERRQ(PetscSFBcastEnd(sf, MPIU_INT, neg, tmpOff,MPI_REPLACE));
    if (nroots > pEnd - pStart) {
      for (p = pStart; p < pEnd; ++p) {if (tmpOff[p] < 0) (*gsection)->atlasOff[p-pStart] = tmpOff[p];}
    }
  }
  if (nroots >= 0 && nroots > pEnd-pStart) CHKERRQ(PetscFree(tmpOff));
  CHKERRQ(PetscFree(neg));
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

  Note: This is usually called for the default global section.

  Level: advanced

.seealso: PetscSectionGetValueLayout(), PetscSectionCreate()
@*/
PetscErrorCode PetscSectionGetPointLayout(MPI_Comm comm, PetscSection s, PetscLayout *layout)
{
  PetscInt       pStart, pEnd, p, localSize = 0;

  PetscFunctionBegin;
  CHKERRQ(PetscSectionGetChart(s, &pStart, &pEnd));
  for (p = pStart; p < pEnd; ++p) {
    PetscInt dof;

    CHKERRQ(PetscSectionGetDof(s, p, &dof));
    if (dof >= 0) ++localSize;
  }
  CHKERRQ(PetscLayoutCreate(comm, layout));
  CHKERRQ(PetscLayoutSetLocalSize(*layout, localSize));
  CHKERRQ(PetscLayoutSetBlockSize(*layout, 1));
  CHKERRQ(PetscLayoutSetUp(*layout));
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

  PetscFunctionBegin;
  PetscValidHeaderSpecific(s, PETSC_SECTION_CLASSID, 2);
  PetscValidPointer(layout, 3);
  CHKERRQ(PetscSectionGetChart(s, &pStart, &pEnd));
  for (p = pStart; p < pEnd; ++p) {
    PetscInt dof,cdof;

    CHKERRQ(PetscSectionGetDof(s, p, &dof));
    CHKERRQ(PetscSectionGetConstraintDof(s, p, &cdof));
    if (dof-cdof > 0) localSize += dof-cdof;
  }
  CHKERRQ(PetscLayoutCreate(comm, layout));
  CHKERRQ(PetscLayoutSetLocalSize(*layout, localSize));
  CHKERRQ(PetscLayoutSetBlockSize(*layout, 1));
  CHKERRQ(PetscLayoutSetUp(*layout));
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
    PetscCheckFalse((point < s->pStart) || (point >= s->pEnd),PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Section point %" PetscInt_FMT " should be in [%" PetscInt_FMT ", %" PetscInt_FMT ")", point, s->pStart, s->pEnd);
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
  PetscCheckFalse((point < s->pStart) || (point >= s->pEnd),PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Section point %" PetscInt_FMT " should be in [%" PetscInt_FMT ", %" PetscInt_FMT ")", point, s->pStart, s->pEnd);
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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(s, PETSC_SECTION_CLASSID, 1);
  PetscValidIntPointer(offset, 4);
  PetscSectionCheckValidField(field,s->numFields);
  CHKERRQ(PetscSectionGetOffset(s->field[field], point, offset));
  PetscFunctionReturn(0);
}

/*@
  PetscSectionSetFieldOffset - Set the offset into an array or local Vec for the dof associated with the given field at a point.

  Not collective

  Input Parameters:
+ s - the PetscSection
. point - the point
. field - the field
- offset - the offset

  Note: The user usually does not call this function, but uses PetscSectionSetUp()

  Level: intermediate

.seealso: PetscSectionGetFieldOffset(), PetscSectionSetOffset(), PetscSectionCreate(), PetscSectionSetUp()
@*/
PetscErrorCode PetscSectionSetFieldOffset(PetscSection s, PetscInt point, PetscInt field, PetscInt offset)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(s, PETSC_SECTION_CLASSID, 1);
  PetscSectionCheckValidField(field,s->numFields);
  CHKERRQ(PetscSectionSetOffset(s->field[field], point, offset));
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

  PetscFunctionBegin;
  PetscValidHeaderSpecific(s, PETSC_SECTION_CLASSID, 1);
  PetscValidIntPointer(offset, 4);
  PetscSectionCheckValidField(field,s->numFields);
  CHKERRQ(PetscSectionGetOffset(s, point, &off));
  CHKERRQ(PetscSectionGetOffset(s->field[field], point, &foff));
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

  PetscFunctionBegin;
  PetscValidHeaderSpecific(s, PETSC_SECTION_CLASSID, 1);
  if (s->atlasOff) {os = s->atlasOff[0]; oe = s->atlasOff[0];}
  CHKERRQ(PetscSectionGetChart(s, &pStart, &pEnd));
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

  Input Parameters:
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

  PetscFunctionBegin;
  if (!len) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(s, PETSC_SECTION_CLASSID, 1);
  PetscValidPointer(fields, 3);
  PetscValidPointer(subs, 4);
  CHKERRQ(PetscSectionGetNumFields(s, &nF));
  PetscCheckFalse(len > nF,PetscObjectComm((PetscObject) s), PETSC_ERR_ARG_WRONG, "Number of requested fields %" PetscInt_FMT " greater than number of fields %" PetscInt_FMT, len, nF);
  CHKERRQ(PetscSectionCreate(PetscObjectComm((PetscObject) s), subs));
  CHKERRQ(PetscSectionSetNumFields(*subs, len));
  for (f = 0; f < len; ++f) {
    const char *name   = NULL;
    PetscInt   numComp = 0;

    CHKERRQ(PetscSectionGetFieldName(s, fields[f], &name));
    CHKERRQ(PetscSectionSetFieldName(*subs, f, name));
    CHKERRQ(PetscSectionGetFieldComponents(s, fields[f], &numComp));
    CHKERRQ(PetscSectionSetFieldComponents(*subs, f, numComp));
    for (c = 0; c < s->numFieldComponents[fields[f]]; ++c) {
      CHKERRQ(PetscSectionGetComponentName(s, fields[f], c, &name));
      CHKERRQ(PetscSectionSetComponentName(*subs, f, c, name));
    }
  }
  CHKERRQ(PetscSectionGetChart(s, &pStart, &pEnd));
  CHKERRQ(PetscSectionSetChart(*subs, pStart, pEnd));
  for (p = pStart; p < pEnd; ++p) {
    PetscInt dof = 0, cdof = 0, fdof = 0, cfdof = 0;

    for (f = 0; f < len; ++f) {
      CHKERRQ(PetscSectionGetFieldDof(s, p, fields[f], &fdof));
      CHKERRQ(PetscSectionSetFieldDof(*subs, p, f, fdof));
      CHKERRQ(PetscSectionGetFieldConstraintDof(s, p, fields[f], &cfdof));
      if (cfdof) CHKERRQ(PetscSectionSetFieldConstraintDof(*subs, p, f, cfdof));
      dof  += fdof;
      cdof += cfdof;
    }
    CHKERRQ(PetscSectionSetDof(*subs, p, dof));
    if (cdof) CHKERRQ(PetscSectionSetConstraintDof(*subs, p, cdof));
    maxCdof = PetscMax(cdof, maxCdof);
  }
  CHKERRQ(PetscSectionSetUp(*subs));
  if (maxCdof) {
    PetscInt *indices;

    CHKERRQ(PetscMalloc1(maxCdof, &indices));
    for (p = pStart; p < pEnd; ++p) {
      PetscInt cdof;

      CHKERRQ(PetscSectionGetConstraintDof(*subs, p, &cdof));
      if (cdof) {
        const PetscInt *oldIndices = NULL;
        PetscInt       fdof = 0, cfdof = 0, fc, numConst = 0, fOff = 0;

        for (f = 0; f < len; ++f) {
          CHKERRQ(PetscSectionGetFieldDof(s, p, fields[f], &fdof));
          CHKERRQ(PetscSectionGetFieldConstraintDof(s, p, fields[f], &cfdof));
          CHKERRQ(PetscSectionGetFieldConstraintIndices(s, p, fields[f], &oldIndices));
          CHKERRQ(PetscSectionSetFieldConstraintIndices(*subs, p, f, oldIndices));
          for (fc = 0; fc < cfdof; ++fc) indices[numConst+fc] = oldIndices[fc] + fOff;
          numConst += cfdof;
          fOff     += fdof;
        }
        CHKERRQ(PetscSectionSetConstraintIndices(*subs, p, indices));
      }
    }
    CHKERRQ(PetscFree(indices));
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

  PetscFunctionBegin;
  if (!len) PetscFunctionReturn(0);
  for (i = 0; i < len; ++i) {
    PetscInt nf, pStarti, pEndi;

    CHKERRQ(PetscSectionGetNumFields(s[i], &nf));
    CHKERRQ(PetscSectionGetChart(s[i], &pStarti, &pEndi));
    pStart = PetscMin(pStart, pStarti);
    pEnd   = PetscMax(pEnd,   pEndi);
    Nf += nf;
  }
  CHKERRQ(PetscSectionCreate(PetscObjectComm((PetscObject) s[0]), supers));
  CHKERRQ(PetscSectionSetNumFields(*supers, Nf));
  for (i = 0, f = 0; i < len; ++i) {
    PetscInt nf, fi, ci;

    CHKERRQ(PetscSectionGetNumFields(s[i], &nf));
    for (fi = 0; fi < nf; ++fi, ++f) {
      const char *name   = NULL;
      PetscInt   numComp = 0;

      CHKERRQ(PetscSectionGetFieldName(s[i], fi, &name));
      CHKERRQ(PetscSectionSetFieldName(*supers, f, name));
      CHKERRQ(PetscSectionGetFieldComponents(s[i], fi, &numComp));
      CHKERRQ(PetscSectionSetFieldComponents(*supers, f, numComp));
      for (ci = 0; ci < s[i]->numFieldComponents[fi]; ++ci) {
        CHKERRQ(PetscSectionGetComponentName(s[i], fi, ci, &name));
        CHKERRQ(PetscSectionSetComponentName(*supers, f, ci, name));
      }
    }
  }
  CHKERRQ(PetscSectionSetChart(*supers, pStart, pEnd));
  for (p = pStart; p < pEnd; ++p) {
    PetscInt dof = 0, cdof = 0;

    for (i = 0, f = 0; i < len; ++i) {
      PetscInt nf, fi, pStarti, pEndi;
      PetscInt fdof = 0, cfdof = 0;

      CHKERRQ(PetscSectionGetNumFields(s[i], &nf));
      CHKERRQ(PetscSectionGetChart(s[i], &pStarti, &pEndi));
      if ((p < pStarti) || (p >= pEndi)) continue;
      for (fi = 0; fi < nf; ++fi, ++f) {
        CHKERRQ(PetscSectionGetFieldDof(s[i], p, fi, &fdof));
        CHKERRQ(PetscSectionAddFieldDof(*supers, p, f, fdof));
        CHKERRQ(PetscSectionGetFieldConstraintDof(s[i], p, fi, &cfdof));
        if (cfdof) CHKERRQ(PetscSectionAddFieldConstraintDof(*supers, p, f, cfdof));
        dof  += fdof;
        cdof += cfdof;
      }
    }
    CHKERRQ(PetscSectionSetDof(*supers, p, dof));
    if (cdof) CHKERRQ(PetscSectionSetConstraintDof(*supers, p, cdof));
    maxCdof = PetscMax(cdof, maxCdof);
  }
  CHKERRQ(PetscSectionSetUp(*supers));
  if (maxCdof) {
    PetscInt *indices;

    CHKERRQ(PetscMalloc1(maxCdof, &indices));
    for (p = pStart; p < pEnd; ++p) {
      PetscInt cdof;

      CHKERRQ(PetscSectionGetConstraintDof(*supers, p, &cdof));
      if (cdof) {
        PetscInt dof, numConst = 0, fOff = 0;

        for (i = 0, f = 0; i < len; ++i) {
          const PetscInt *oldIndices = NULL;
          PetscInt        nf, fi, pStarti, pEndi, fdof, cfdof, fc;

          CHKERRQ(PetscSectionGetNumFields(s[i], &nf));
          CHKERRQ(PetscSectionGetChart(s[i], &pStarti, &pEndi));
          if ((p < pStarti) || (p >= pEndi)) continue;
          for (fi = 0; fi < nf; ++fi, ++f) {
            CHKERRQ(PetscSectionGetFieldDof(s[i], p, fi, &fdof));
            CHKERRQ(PetscSectionGetFieldConstraintDof(s[i], p, fi, &cfdof));
            CHKERRQ(PetscSectionGetFieldConstraintIndices(s[i], p, fi, &oldIndices));
            for (fc = 0; fc < cfdof; ++fc) indices[numConst+fc] = oldIndices[fc];
            CHKERRQ(PetscSectionSetFieldConstraintIndices(*supers, p, f, &indices[numConst]));
            for (fc = 0; fc < cfdof; ++fc) indices[numConst+fc] += fOff;
            numConst += cfdof;
          }
          CHKERRQ(PetscSectionGetDof(s[i], p, &dof));
          fOff += dof;
        }
        CHKERRQ(PetscSectionSetConstraintIndices(*supers, p, indices));
      }
    }
    CHKERRQ(PetscFree(indices));
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

  PetscFunctionBegin;
  PetscValidHeaderSpecific(s, PETSC_SECTION_CLASSID, 1);
  PetscValidHeaderSpecific(subpointMap, IS_CLASSID, 2);
  PetscValidPointer(subs, 3);
  CHKERRQ(PetscSectionGetNumFields(s, &numFields));
  CHKERRQ(PetscSectionCreate(PetscObjectComm((PetscObject) s), subs));
  if (numFields) CHKERRQ(PetscSectionSetNumFields(*subs, numFields));
  for (f = 0; f < numFields; ++f) {
    const char *name   = NULL;
    PetscInt   numComp = 0;

    CHKERRQ(PetscSectionGetFieldName(s, f, &name));
    CHKERRQ(PetscSectionSetFieldName(*subs, f, name));
    CHKERRQ(PetscSectionGetFieldComponents(s, f, &numComp));
    CHKERRQ(PetscSectionSetFieldComponents(*subs, f, numComp));
    for (c = 0; c < s->numFieldComponents[f]; ++c) {
      CHKERRQ(PetscSectionGetComponentName(s, f, c, &name));
      CHKERRQ(PetscSectionSetComponentName(*subs, f, c, name));
    }
  }
  /* For right now, we do not try to squeeze the subchart */
  if (subpointMap) {
    CHKERRQ(ISGetSize(subpointMap, &numSubpoints));
    CHKERRQ(ISGetIndices(subpointMap, &points));
  }
  CHKERRQ(PetscSectionGetChart(s, &pStart, &pEnd));
  CHKERRQ(PetscSectionSetChart(*subs, 0, numSubpoints));
  for (p = pStart; p < pEnd; ++p) {
    PetscInt dof, cdof, fdof = 0, cfdof = 0;

    CHKERRQ(PetscFindInt(p, numSubpoints, points, &subp));
    if (subp < 0) continue;
    for (f = 0; f < numFields; ++f) {
      CHKERRQ(PetscSectionGetFieldDof(s, p, f, &fdof));
      CHKERRQ(PetscSectionSetFieldDof(*subs, subp, f, fdof));
      CHKERRQ(PetscSectionGetFieldConstraintDof(s, p, f, &cfdof));
      if (cfdof) CHKERRQ(PetscSectionSetFieldConstraintDof(*subs, subp, f, cfdof));
    }
    CHKERRQ(PetscSectionGetDof(s, p, &dof));
    CHKERRQ(PetscSectionSetDof(*subs, subp, dof));
    CHKERRQ(PetscSectionGetConstraintDof(s, p, &cdof));
    if (cdof) CHKERRQ(PetscSectionSetConstraintDof(*subs, subp, cdof));
  }
  CHKERRQ(PetscSectionSetUp(*subs));
  /* Change offsets to original offsets */
  for (p = pStart; p < pEnd; ++p) {
    PetscInt off, foff = 0;

    CHKERRQ(PetscFindInt(p, numSubpoints, points, &subp));
    if (subp < 0) continue;
    for (f = 0; f < numFields; ++f) {
      CHKERRQ(PetscSectionGetFieldOffset(s, p, f, &foff));
      CHKERRQ(PetscSectionSetFieldOffset(*subs, subp, f, foff));
    }
    CHKERRQ(PetscSectionGetOffset(s, p, &off));
    CHKERRQ(PetscSectionSetOffset(*subs, subp, off));
  }
  /* Copy constraint indices */
  for (subp = 0; subp < numSubpoints; ++subp) {
    PetscInt cdof;

    CHKERRQ(PetscSectionGetConstraintDof(*subs, subp, &cdof));
    if (cdof) {
      for (f = 0; f < numFields; ++f) {
        CHKERRQ(PetscSectionGetFieldConstraintIndices(s, points[subp], f, &indices));
        CHKERRQ(PetscSectionSetFieldConstraintIndices(*subs, subp, f, indices));
      }
      CHKERRQ(PetscSectionGetConstraintIndices(s, points[subp], &indices));
      CHKERRQ(PetscSectionSetConstraintIndices(*subs, subp, indices));
    }
  }
  if (subpointMap) CHKERRQ(ISRestoreIndices(subpointMap, &points));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSectionView_ASCII(PetscSection s, PetscViewer viewer)
{
  PetscInt       p;
  PetscMPIInt    rank;

  PetscFunctionBegin;
  CHKERRMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)viewer), &rank));
  CHKERRQ(PetscViewerASCIIPushSynchronized(viewer));
  CHKERRQ(PetscViewerASCIISynchronizedPrintf(viewer, "Process %d:\n", rank));
  for (p = 0; p < s->pEnd - s->pStart; ++p) {
    if ((s->bc) && (s->bc->atlasDof[p] > 0)) {
      PetscInt b;

      CHKERRQ(PetscViewerASCIISynchronizedPrintf(viewer, "  (%4" PetscInt_FMT ") dim %2" PetscInt_FMT " offset %3" PetscInt_FMT " constrained", p+s->pStart, s->atlasDof[p], s->atlasOff[p]));
      if (s->bcIndices) {
        for (b = 0; b < s->bc->atlasDof[p]; ++b) {
          CHKERRQ(PetscViewerASCIISynchronizedPrintf(viewer, " %" PetscInt_FMT, s->bcIndices[s->bc->atlasOff[p]+b]));
        }
      }
      CHKERRQ(PetscViewerASCIISynchronizedPrintf(viewer, "\n"));
    } else {
      CHKERRQ(PetscViewerASCIISynchronizedPrintf(viewer, "  (%4" PetscInt_FMT ") dim %2" PetscInt_FMT " offset %3" PetscInt_FMT "\n", p+s->pStart, s->atlasDof[p], s->atlasOff[p]));
    }
  }
  CHKERRQ(PetscViewerFlush(viewer));
  CHKERRQ(PetscViewerASCIIPopSynchronized(viewer));
  if (s->sym) {
    CHKERRQ(PetscViewerASCIIPushTab(viewer));
    CHKERRQ(PetscSectionSymView(s->sym,viewer));
    CHKERRQ(PetscViewerASCIIPopTab(viewer));
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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,PETSC_SECTION_CLASSID,1);
  CHKERRQ(PetscObjectViewFromOptions((PetscObject)A,obj,name));
  PetscFunctionReturn(0);
}

/*@C
  PetscSectionView - Views a PetscSection

  Collective on PetscSection

  Input Parameters:
+ s - the PetscSection object to view
- v - the viewer

  Note:
  PetscSectionView(), when viewer is of type PETSCVIEWERHDF5, only saves
  distribution independent data, such as dofs, offsets, constraint dofs,
  and constraint indices. Points that have negative dofs, for instance,
  are not saved as they represent points owned by other processes.
  Point numbering and rank assignment is currently not stored.
  The saved section can be loaded with PetscSectionLoad().

  Level: beginner

.seealso PetscSectionCreate(), PetscSectionDestroy(), PetscSectionLoad()
@*/
PetscErrorCode PetscSectionView(PetscSection s, PetscViewer viewer)
{
  PetscBool      isascii, ishdf5;
  PetscInt       f;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(s, PETSC_SECTION_CLASSID, 1);
  if (!viewer) CHKERRQ(PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)s), &viewer));
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  CHKERRQ(PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &isascii));
  CHKERRQ(PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERHDF5, &ishdf5));
  if (isascii) {
    CHKERRQ(PetscObjectPrintClassNamePrefixType((PetscObject)s,viewer));
    if (s->numFields) {
      CHKERRQ(PetscViewerASCIIPrintf(viewer, "%" PetscInt_FMT " fields\n", s->numFields));
      for (f = 0; f < s->numFields; ++f) {
        CHKERRQ(PetscViewerASCIIPrintf(viewer, "  field %" PetscInt_FMT " with %" PetscInt_FMT " components\n", f, s->numFieldComponents[f]));
        CHKERRQ(PetscSectionView_ASCII(s->field[f], viewer));
      }
    } else {
      CHKERRQ(PetscSectionView_ASCII(s, viewer));
    }
  } else if (ishdf5) {
#if PetscDefined(HAVE_HDF5)
    CHKERRQ(PetscSectionView_HDF5_Internal(s, viewer));
#else
    SETERRQ(PetscObjectComm((PetscObject) s), PETSC_ERR_SUP, "HDF5 not supported in this build.\nPlease reconfigure using --download-hdf5");
#endif
  }
  PetscFunctionReturn(0);
}

/*@C
  PetscSectionLoad - Loads a PetscSection

  Collective on PetscSection

  Input Parameters:
+ s - the PetscSection object to load
- v - the viewer

  Note:
  PetscSectionLoad(), when viewer is of type PETSCVIEWERHDF5, loads
  a section saved with PetscSectionView(). The number of processes
  used here (N) does not need to be the same as that used when saving.
  After calling this function, the chart of s on rank i will be set
  to [0, E_i), where \sum_{i=0}^{N-1}E_i equals to the total number of
  saved section points.

  Level: beginner

.seealso PetscSectionCreate(), PetscSectionDestroy(), PetscSectionView()
@*/
PetscErrorCode PetscSectionLoad(PetscSection s, PetscViewer viewer)
{
  PetscBool      ishdf5;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(s, PETSC_SECTION_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  CHKERRQ(PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERHDF5, &ishdf5));
  if (ishdf5) {
#if PetscDefined(HAVE_HDF5)
    CHKERRQ(PetscSectionLoad_HDF5_Internal(s, viewer));
    PetscFunctionReturn(0);
#else
    SETERRQ(PetscObjectComm((PetscObject) s), PETSC_ERR_SUP, "HDF5 not supported in this build.\nPlease reconfigure using --download-hdf5");
#endif
  } else SETERRQ(PetscObjectComm((PetscObject) s), PETSC_ERR_SUP, "Viewer type %s not yet supported for PetscSection loading", ((PetscObject)viewer)->type_name);
}

static PetscErrorCode PetscSectionResetClosurePermutation(PetscSection section)
{
  PetscSectionClosurePermVal clVal;

  PetscFunctionBegin;
  if (!section->clHash) PetscFunctionReturn(0);
  kh_foreach_value(section->clHash, clVal, {
      CHKERRQ(PetscFree(clVal.perm));
      CHKERRQ(PetscFree(clVal.invPerm));
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

  PetscFunctionBegin;
  PetscValidHeaderSpecific(s, PETSC_SECTION_CLASSID, 1);
  for (f = 0; f < s->numFields; ++f) {
    CHKERRQ(PetscSectionDestroy(&s->field[f]));
    CHKERRQ(PetscFree(s->fieldNames[f]));
    for (c = 0; c < s->numFieldComponents[f]; ++c) {
      CHKERRQ(PetscFree(s->compNames[f][c]));
    }
    CHKERRQ(PetscFree(s->compNames[f]));
  }
  CHKERRQ(PetscFree(s->numFieldComponents));
  CHKERRQ(PetscFree(s->fieldNames));
  CHKERRQ(PetscFree(s->compNames));
  CHKERRQ(PetscFree(s->field));
  CHKERRQ(PetscSectionDestroy(&s->bc));
  CHKERRQ(PetscFree(s->bcIndices));
  CHKERRQ(PetscFree2(s->atlasDof, s->atlasOff));
  CHKERRQ(PetscSectionDestroy(&s->clSection));
  CHKERRQ(ISDestroy(&s->clPoints));
  CHKERRQ(ISDestroy(&s->perm));
  CHKERRQ(PetscSectionResetClosurePermutation(s));
  CHKERRQ(PetscSectionSymDestroy(&s->sym));
  CHKERRQ(PetscSectionDestroy(&s->clSection));
  CHKERRQ(ISDestroy(&s->clPoints));

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
  PetscFunctionBegin;
  if (!*s) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(*s,PETSC_SECTION_CLASSID,1);
  if (--((PetscObject)(*s))->refct > 0) {
    *s = NULL;
    PetscFunctionReturn(0);
  }
  CHKERRQ(PetscSectionReset(*s));
  CHKERRQ(PetscHeaderDestroy(s));
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

  PetscFunctionBegin;
  PetscValidHeaderSpecific(s, PETSC_SECTION_CLASSID, 2);
  CHKERRQ(PetscSectionGetConstraintDof(s, p, &cDim));
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

      CHKERRQ(PetscSectionGetConstraintIndices(s, point, &cDof));
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

      CHKERRQ(PetscSectionGetConstraintIndices(s, point, &cDof));
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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(s, PETSC_SECTION_CLASSID, 1);
  if (s->bc) {
    CHKERRQ(VecIntGetValuesSection(s->bcIndices, s->bc, point, indices));
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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(s, PETSC_SECTION_CLASSID, 1);
  if (s->bc) {
    const PetscInt dof  = s->atlasDof[point];
    const PetscInt cdof = s->bc->atlasDof[point];
    PetscInt       d;

    for (d = 0; d < cdof; ++d) {
      PetscCheck(indices[d] < dof,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Point %" PetscInt_FMT " dof %" PetscInt_FMT ", invalid constraint index[%" PetscInt_FMT "]: %" PetscInt_FMT, point, dof, d, indices[d]);
    }
    CHKERRQ(VecIntSetValuesSection(s->bcIndices, s->bc, point, indices, INSERT_VALUES));
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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(s, PETSC_SECTION_CLASSID, 1);
  PetscValidIntPointer(indices,4);
  PetscSectionCheckValidField(field,s->numFields);
  CHKERRQ(PetscSectionGetConstraintIndices(s->field[field], point, indices));
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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(s, PETSC_SECTION_CLASSID, 1);
  if (PetscDefined(USE_DEBUG)) {
    PetscInt nfdof;

    CHKERRQ(PetscSectionGetFieldConstraintDof(s, point, field, &nfdof));
    if (nfdof) PetscValidIntPointer(indices, 4);
  }
  PetscSectionCheckValidField(field,s->numFields);
  CHKERRQ(PetscSectionSetConstraintIndices(s->field[field], point, indices));
  PetscFunctionReturn(0);
}

/*@
  PetscSectionPermute - Reorder the section according to the input point permutation

  Collective on PetscSection

  Input Parameters:
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

  PetscFunctionBegin;
  PetscValidHeaderSpecific(section, PETSC_SECTION_CLASSID, 1);
  PetscValidHeaderSpecific(permutation, IS_CLASSID, 2);
  PetscValidPointer(sectionNew, 3);
  CHKERRQ(PetscSectionCreate(PetscObjectComm((PetscObject) s), &sNew));
  CHKERRQ(PetscSectionGetNumFields(s, &numFields));
  if (numFields) CHKERRQ(PetscSectionSetNumFields(sNew, numFields));
  for (f = 0; f < numFields; ++f) {
    const char *name;
    PetscInt    numComp;

    CHKERRQ(PetscSectionGetFieldName(s, f, &name));
    CHKERRQ(PetscSectionSetFieldName(sNew, f, name));
    CHKERRQ(PetscSectionGetFieldComponents(s, f, &numComp));
    CHKERRQ(PetscSectionSetFieldComponents(sNew, f, numComp));
    for (c = 0; c < s->numFieldComponents[f]; ++c) {
      CHKERRQ(PetscSectionGetComponentName(s, f, c, &name));
      CHKERRQ(PetscSectionSetComponentName(sNew, f, c, name));
    }
  }
  CHKERRQ(ISGetLocalSize(permutation, &numPoints));
  CHKERRQ(ISGetIndices(permutation, &perm));
  CHKERRQ(PetscSectionGetChart(s, &pStart, &pEnd));
  CHKERRQ(PetscSectionSetChart(sNew, pStart, pEnd));
  PetscCheckFalse(numPoints < pEnd,PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Permutation size %" PetscInt_FMT " is less than largest Section point %" PetscInt_FMT, numPoints, pEnd);
  for (p = pStart; p < pEnd; ++p) {
    PetscInt dof, cdof;

    CHKERRQ(PetscSectionGetDof(s, p, &dof));
    CHKERRQ(PetscSectionSetDof(sNew, perm[p], dof));
    CHKERRQ(PetscSectionGetConstraintDof(s, p, &cdof));
    if (cdof) CHKERRQ(PetscSectionSetConstraintDof(sNew, perm[p], cdof));
    for (f = 0; f < numFields; ++f) {
      CHKERRQ(PetscSectionGetFieldDof(s, p, f, &dof));
      CHKERRQ(PetscSectionSetFieldDof(sNew, perm[p], f, dof));
      CHKERRQ(PetscSectionGetFieldConstraintDof(s, p, f, &cdof));
      if (cdof) CHKERRQ(PetscSectionSetFieldConstraintDof(sNew, perm[p], f, cdof));
    }
  }
  CHKERRQ(PetscSectionSetUp(sNew));
  for (p = pStart; p < pEnd; ++p) {
    const PetscInt *cind;
    PetscInt        cdof;

    CHKERRQ(PetscSectionGetConstraintDof(s, p, &cdof));
    if (cdof) {
      CHKERRQ(PetscSectionGetConstraintIndices(s, p, &cind));
      CHKERRQ(PetscSectionSetConstraintIndices(sNew, perm[p], cind));
    }
    for (f = 0; f < numFields; ++f) {
      CHKERRQ(PetscSectionGetFieldConstraintDof(s, p, f, &cdof));
      if (cdof) {
        CHKERRQ(PetscSectionGetFieldConstraintIndices(s, p, f, &cind));
        CHKERRQ(PetscSectionSetFieldConstraintIndices(sNew, perm[p], f, cind));
      }
    }
  }
  CHKERRQ(ISRestoreIndices(permutation, &perm));
  *sectionNew = sNew;
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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(section,PETSC_SECTION_CLASSID,1);
  PetscValidHeaderSpecific(clSection,PETSC_SECTION_CLASSID,3);
  PetscValidHeaderSpecific(clPoints,IS_CLASSID,4);
  if (section->clObj != obj) CHKERRQ(PetscSectionResetClosurePermutation(section));
  section->clObj     = obj;
  CHKERRQ(PetscObjectReference((PetscObject)clSection));
  CHKERRQ(PetscObjectReference((PetscObject)clPoints));
  CHKERRQ(PetscSectionDestroy(&section->clSection));
  CHKERRQ(ISDestroy(&section->clPoints));
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

  PetscFunctionBegin;
  if (section->clObj != obj) {
    CHKERRQ(PetscSectionDestroy(&section->clSection));
    CHKERRQ(ISDestroy(&section->clPoints));
  }
  section->clObj = obj;
  if (!section->clHash) CHKERRQ(PetscClPermCreate(&section->clHash));
  iter = kh_put(ClPerm, section->clHash, key, &new_entry);
  val = &kh_val(section->clHash, iter);
  if (!new_entry) {
    CHKERRQ(PetscFree(val->perm));
    CHKERRQ(PetscFree(val->invPerm));
  }
  if (mode == PETSC_COPY_VALUES) {
    CHKERRQ(PetscMalloc1(clSize, &val->perm));
    CHKERRQ(PetscLogObjectMemory((PetscObject) obj, clSize*sizeof(PetscInt)));
    CHKERRQ(PetscArraycpy(val->perm, clPerm, clSize));
  } else if (mode == PETSC_OWN_POINTER) {
    val->perm = clPerm;
  } else SETERRQ(PetscObjectComm(obj), PETSC_ERR_SUP, "Do not support borrowed arrays");
  CHKERRQ(PetscMalloc1(clSize, &val->invPerm));
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

  PetscFunctionBegin;
  if (perm) {
    CHKERRQ(ISGetLocalSize(perm, &clSize));
    CHKERRQ(ISGetIndices(perm, &clPerm));
  }
  CHKERRQ(PetscSectionSetClosurePermutation_Internal(section, obj, depth, clSize, PETSC_COPY_VALUES, (PetscInt *) clPerm));
  if (perm) CHKERRQ(ISRestoreIndices(perm, &clPerm));
  PetscFunctionReturn(0);
}

PetscErrorCode PetscSectionGetClosurePermutation_Internal(PetscSection section, PetscObject obj, PetscInt depth, PetscInt size, const PetscInt *perm[])
{
  PetscFunctionBegin;
  if (section->clObj == obj) {
    PetscSectionClosurePermKey k = {depth, size};
    PetscSectionClosurePermVal v;
    CHKERRQ(PetscClPermGet(section->clHash, k, &v));
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

  PetscFunctionBegin;
  CHKERRQ(PetscSectionGetClosurePermutation_Internal(section, obj, depth, clSize, &clPerm));
  CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF, clSize, clPerm, PETSC_USE_POINTER, perm));
  PetscFunctionReturn(0);
}

PetscErrorCode PetscSectionGetClosureInversePermutation_Internal(PetscSection section, PetscObject obj, PetscInt depth, PetscInt size, const PetscInt *perm[])
{
  PetscFunctionBegin;
  if (section->clObj == obj && section->clHash) {
    PetscSectionClosurePermKey k = {depth, size};
    PetscSectionClosurePermVal v;
    CHKERRQ(PetscClPermGet(section->clHash, k, &v));
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

  PetscFunctionBegin;
  CHKERRQ(PetscSectionGetClosureInversePermutation_Internal(section, obj, depth, clSize, &clPerm));
  CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF, clSize, clPerm, PETSC_USE_POINTER, perm));
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
  PetscSectionCheckValidField(field,s->numFields);
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
  PetscFunctionBegin;
  PetscValidPointer(sym,2);
  CHKERRQ(ISInitializePackage());
  CHKERRQ(PetscHeaderCreate(*sym,PETSC_SECTION_SYM_CLASSID,"PetscSectionSym","Section Symmetry","IS",comm,PetscSectionSymDestroy,PetscSectionSymView));
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

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sym, PETSC_SECTION_SYM_CLASSID,1);
  CHKERRQ(PetscObjectTypeCompare((PetscObject) sym, method, &match));
  if (match) PetscFunctionReturn(0);

  CHKERRQ(PetscFunctionListFind(PetscSectionSymList,method,&r));
  PetscCheck(r,PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE, "Unknown PetscSectionSym type: %s", method);
  if (sym->ops->destroy) {
    CHKERRQ((*sym->ops->destroy)(sym));
    sym->ops->destroy = NULL;
  }
  CHKERRQ((*r)(sym));
  CHKERRQ(PetscObjectChangeTypeName((PetscObject)sym,method));
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
  PetscFunctionBegin;
  CHKERRQ(ISInitializePackage());
  CHKERRQ(PetscFunctionListAdd(&PetscSectionSymList,sname,function));
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

  PetscFunctionBegin;
  if (!*sym) PetscFunctionReturn(0);
  PetscValidHeaderSpecific((*sym),PETSC_SECTION_SYM_CLASSID,1);
  if (--((PetscObject)(*sym))->refct > 0) {*sym = NULL; PetscFunctionReturn(0);}
  if ((*sym)->ops->destroy) {
    CHKERRQ((*(*sym)->ops->destroy)(*sym));
  }
  PetscCheckFalse((*sym)->workout,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Work array still checked out");
  for (link=(*sym)->workin; link; link=next) {
    next = link->next;
    CHKERRQ(PetscFree2(link->perms,link->rots));
    CHKERRQ(PetscFree(link));
  }
  (*sym)->workin = NULL;
  CHKERRQ(PetscHeaderDestroy(sym));
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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sym,PETSC_SECTION_SYM_CLASSID,1);
  if (!viewer) {
    CHKERRQ(PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)sym),&viewer));
  }
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  PetscCheckSameComm(sym,1,viewer,2);
  CHKERRQ(PetscObjectPrintClassNamePrefixType((PetscObject)sym,viewer));
  if (sym->ops->view) {
    CHKERRQ((*sym->ops->view)(sym,viewer));
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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(section,PETSC_SECTION_CLASSID,1);
  CHKERRQ(PetscSectionSymDestroy(&(section->sym)));
  if (sym) {
    PetscValidHeaderSpecific(sym,PETSC_SECTION_SYM_CLASSID,2);
    PetscCheckSameComm(section,1,sym,2);
    CHKERRQ(PetscObjectReference((PetscObject) sym));
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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(section,PETSC_SECTION_CLASSID,1);
  PetscSectionCheckValidField(field,section->numFields);
  CHKERRQ(PetscSectionSetSym(section->field[field],sym));
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
  PetscSectionCheckValidField(field,section->numFields);
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

  Output Parameters:
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
      CHKERRQ(PetscNewLog(sym,&link));
    }
    if (numPoints > link->numPoints) {
      CHKERRQ(PetscFree2(link->perms,link->rots));
      CHKERRQ(PetscMalloc2(numPoints,&link->perms,numPoints,&link->rots));
      link->numPoints = numPoints;
    }
    link->next   = sym->workout;
    sym->workout = link;
    CHKERRQ(PetscArrayzero((PetscInt**)link->perms,numPoints));
    CHKERRQ(PetscArrayzero((PetscInt**)link->rots,numPoints));
    CHKERRQ((*sym->ops->getpoints) (sym, section, numPoints, points, link->perms, link->rots));
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

  Output Parameters:
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

  Output Parameters:
+ perms - The permutations for the given orientations (or NULL if there is no symmetry or the permutation is the identity).
- rots - The field rotations symmetries for the given orientations (or NULL if there is no symmetry or the rotations are all
    identity).

  Level: developer

.seealso: PetscSectionGetPointSyms(), PetscSectionRestoreFieldPointSyms()
@*/
PetscErrorCode PetscSectionGetFieldPointSyms(PetscSection section, PetscInt field, PetscInt numPoints, const PetscInt *points, const PetscInt ***perms, const PetscScalar ***rots)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(section,PETSC_SECTION_CLASSID,1);
  PetscCheckFalse(field > section->numFields,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"field %" PetscInt_FMT " greater than number of fields (%" PetscInt_FMT ") in section",field,section->numFields);
  CHKERRQ(PetscSectionGetPointSyms(section->field[field],numPoints,points,perms,rots));
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

  Output Parameters:
+ perms - The permutations for the given orientations: set to NULL at conclusion
- rots - The field rotations symmetries for the given orientations: set to NULL at conclusion

  Level: developer

.seealso: PetscSectionRestorePointSyms(), petscSectionGetFieldPointSyms(), PetscSectionSymCreate(), PetscSectionSetSym(), PetscSectionGetSym()
@*/
PetscErrorCode PetscSectionRestoreFieldPointSyms(PetscSection section, PetscInt field, PetscInt numPoints, const PetscInt *points, const PetscInt ***perms, const PetscScalar ***rots)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(section,PETSC_SECTION_CLASSID,1);
  PetscCheckFalse(field > section->numFields,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"field %" PetscInt_FMT " greater than number of fields (%" PetscInt_FMT ") in section",field,section->numFields);
  CHKERRQ(PetscSectionRestorePointSyms(section->field[field],numPoints,points,perms,rots));
  PetscFunctionReturn(0);
}

/*@
  PetscSectionSymCopy - Copy the symmetries, assuming that the point structure is compatible

  Not collective

  Input Parameter:
. sym - the PetscSectionSym

  Output Parameter:
. nsym - the equivalent symmetries

  Level: developer

.seealso: PetscSectionSymCreate(), PetscSectionSetSym(), PetscSectionGetSym(), PetscSectionSymLabelSetStratum(), PetscSectionGetPointSyms()
@*/
PetscErrorCode PetscSectionSymCopy(PetscSectionSym sym, PetscSectionSym nsym)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sym, PETSC_SECTION_SYM_CLASSID, 1);
  PetscValidHeaderSpecific(nsym, PETSC_SECTION_SYM_CLASSID, 2);
  if (sym->ops->copy) CHKERRQ((*sym->ops->copy)(sym, nsym));
  PetscFunctionReturn(0);
}

/*@
  PetscSectionSymDistribute - Distribute the symmetries in accordance with the input SF

  Collective

  Input Parameters:
+ sym - the PetscSectionSym
- migrationSF - the distribution map from roots to leaves

  Output Parameters:
. dsym - the redistributed symmetries

  Level: developer

.seealso: PetscSectionSymCreate(), PetscSectionSetSym(), PetscSectionGetSym(), PetscSectionSymLabelSetStratum(), PetscSectionGetPointSyms()
@*/
PetscErrorCode PetscSectionSymDistribute(PetscSectionSym sym, PetscSF migrationSF, PetscSectionSym *dsym)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sym, PETSC_SECTION_SYM_CLASSID, 1);
  PetscValidHeaderSpecific(migrationSF, PETSCSF_CLASSID, 2);
  PetscValidPointer(dsym, 3);
  if (sym->ops->distribute) CHKERRQ((*sym->ops->distribute)(sym, migrationSF, dsym));
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
  CHKERRQ(PetscSectionGetStorageSize(s, &size)); \
  CHKERRQ(PetscMalloc1(size, &a1)); \
  for (i=0; i<npoints; i++) { \
    CHKERRQ(PetscSectionGetOffset(origSection, points_[i], &o0)); \
    CHKERRQ(PetscSectionGetOffset(s, i, &o1)); \
    CHKERRQ(PetscSectionGetDof(s, i, &n)); \
    CHKERRQ(PetscMemcpy(&a1[o1], &a0[o0], n*unitsize)); \
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

  PetscFunctionBegin;
  PetscValidHeaderSpecific(origSection, PETSC_SECTION_CLASSID, 1);
  PetscValidPointer(origArray, 3);
  PetscValidHeaderSpecific(points, IS_CLASSID, 4);
  if (newSection) PetscValidPointer(newSection, 5);
  if (newArray) PetscValidPointer(newArray, 6);
  CHKERRMPI(MPI_Type_size(dataType, &unitsize));
  CHKERRQ(ISGetLocalSize(points, &npoints));
  CHKERRQ(ISGetIndices(points, &points_));
  CHKERRQ(PetscSectionGetChart(origSection, &pStart, &pEnd));
  CHKERRQ(PetscSectionCreate(PETSC_COMM_SELF, &s));
  CHKERRQ(PetscSectionSetChart(s, 0, npoints));
  for (i=0; i<npoints; i++) {
    PetscCheckFalse(points_[i] < pStart || points_[i] >= pEnd,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "point %" PetscInt_FMT " (index %" PetscInt_FMT ") in input IS out of input section's chart", points_[i], i);
    CHKERRQ(PetscSectionGetDof(origSection, points_[i], &n));
    CHKERRQ(PetscSectionSetDof(s, i, n));
  }
  CHKERRQ(PetscSectionSetUp(s));
  if (newArray) {
    if (dataType == MPIU_INT)           {PetscSectionExpandPoints_Loop(PetscInt);}
    else if (dataType == MPIU_SCALAR)   {PetscSectionExpandPoints_Loop(PetscScalar);}
    else if (dataType == MPIU_REAL)     {PetscSectionExpandPoints_Loop(PetscReal);}
    else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "not implemented for this MPI_Datatype");
  }
  if (newSection) {
    *newSection = s;
  } else {
    CHKERRQ(PetscSectionDestroy(&s));
  }
  CHKERRQ(ISRestoreIndices(points, &points_));
  PetscFunctionReturn(0);
}
