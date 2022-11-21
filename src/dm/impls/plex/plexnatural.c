#include <petsc/private/dmpleximpl.h> /*I      "petscdmplex.h"   I*/

/*@
  DMPlexSetMigrationSF - Sets the `PetscSF` for migrating from a parent `DM` into this `DM`

  Logically Collective on dm

  Input Parameters:
+ dm        - The `DM`
- naturalSF - The `PetscSF`

  Level: intermediate

  Note:
  It is necessary to call this in order to have `DMCreateSubDM()` or `DMCreateSuperDM()` build the Global-To-Natural map

.seealso: [](chapter_unstructured), `DM`, `DMPLEX`, `PetscSF`, `DMPlexDistribute()`, `DMPlexDistributeField()`, `DMPlexCreateMigrationSF()`, `DMPlexGetMigrationSF()`
@*/
PetscErrorCode DMPlexSetMigrationSF(DM dm, PetscSF migrationSF)
{
  PetscFunctionBegin;
  dm->sfMigration = migrationSF;
  PetscCall(PetscObjectReference((PetscObject)migrationSF));
  PetscFunctionReturn(0);
}

/*@
  DMPlexGetMigrationSF - Gets the `PetscSF` for migrating from a parent `DM` into this `DM`

  Note Collective

  Input Parameter:
. dm          - The `DM`

  Output Parameter:
. migrationSF - The `PetscSF`

  Level: intermediate

.seealso: [](chapter_unstructured), `DM`, `DMPLEX`, `PetscSF`, `DMPlexDistribute()`, `DMPlexDistributeField()`, `DMPlexCreateMigrationSF()`, `DMPlexSetMigrationSF`
@*/
PetscErrorCode DMPlexGetMigrationSF(DM dm, PetscSF *migrationSF)
{
  PetscFunctionBegin;
  *migrationSF = dm->sfMigration;
  PetscFunctionReturn(0);
}

/*@
  DMPlexSetGlobalToNaturalSF - Sets the `PetscSF` for mapping Global `Vec` to the Natural `Vec`

  Input Parameters:
+ dm          - The `DM`
- naturalSF   - The `PetscSF`

  Level: intermediate

.seealso: [](chapter_unstructured), `DM`, `DMPLEX`, `PetscSF`, `DMPlexDistribute()`, `DMPlexDistributeField()`, `DMPlexCreateGlobalToNaturalSF()`, `DMPlexGetGlobaltoNaturalSF()`
@*/
PetscErrorCode DMPlexSetGlobalToNaturalSF(DM dm, PetscSF naturalSF)
{
  PetscFunctionBegin;
  dm->sfNatural = naturalSF;
  PetscCall(PetscObjectReference((PetscObject)naturalSF));
  dm->useNatural = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*@
  DMPlexGetGlobalToNaturalSF - Gets the `PetscSF` for mapping Global `Vec` to the Natural `Vec`

  Input Parameter:
. dm          - The `DM`

  Output Parameter:
. naturalSF   - The `PetscSF`

  Level: intermediate

.seealso: [](chapter_unstructured), `DM`, `DMPLEX`, `PetscSF`, `DMPlexDistribute()`, `DMPlexDistributeField()`, `DMPlexCreateGlobalToNaturalSF()`, `DMPlexSetGlobaltoNaturalSF`
@*/
PetscErrorCode DMPlexGetGlobalToNaturalSF(DM dm, PetscSF *naturalSF)
{
  PetscFunctionBegin;
  *naturalSF = dm->sfNatural;
  PetscFunctionReturn(0);
}

/*@
  DMPlexCreateGlobalToNaturalSF - Creates the `PetscSF` for mapping Global `Vec` to the Natural `Vec`

  Input Parameters:
+ dm          - The redistributed `DM`
. section     - The local `PetscSection` describing the `Vec` before the mesh was distributed, or NULL if not available
- sfMigration - The `PetscSF` used to distribute the mesh, or NULL if it cannot be computed

  Output Parameter:
. sfNatural   - `PetscSF` for mapping the `Vec` in PETSc ordering to the canonical ordering

  Level: intermediate

  Note:
  This is not typically called by the user.

.seealso: [](chapter_unstructured), `DM`, `DMPLEX`, `PetscSF`, `PetscSection`, `DMPlexDistribute()`, `DMPlexDistributeField()`
 @*/
PetscErrorCode DMPlexCreateGlobalToNaturalSF(DM dm, PetscSection section, PetscSF sfMigration, PetscSF *sfNatural)
{
  MPI_Comm     comm;
  PetscSF      sf, sfEmbed, sfField;
  PetscSection gSection, sectionDist, gLocSection;
  PetscInt    *spoints, *remoteOffsets;
  PetscInt     ssize, pStart, pEnd, p, localSize, maxStorageSize;
  PetscBool    destroyFlag = PETSC_FALSE, debug = PETSC_FALSE;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)dm, &comm));
  if (!sfMigration) {
    /* If sfMigration is missing, sfNatural cannot be computed and is set to NULL */
    *sfNatural = NULL;
    PetscFunctionReturn(0);
  } else if (!section) {
    /* If the sequential section is not provided (NULL), it is reconstructed from the parallel section */
    PetscSF      sfMigrationInv;
    PetscSection localSection;

    PetscCall(DMGetLocalSection(dm, &localSection));
    PetscCall(PetscSFCreateInverseSF(sfMigration, &sfMigrationInv));
    PetscCall(PetscSectionCreate(PetscObjectComm((PetscObject)dm), &section));
    PetscCall(PetscSFDistributeSection(sfMigrationInv, localSection, NULL, section));
    PetscCall(PetscSFDestroy(&sfMigrationInv));
    destroyFlag = PETSC_TRUE;
  }
  if (debug) PetscCall(PetscSFView(sfMigration, NULL));
  /* Create a new section from distributing the original section */
  PetscCall(PetscSectionCreate(comm, &sectionDist));
  PetscCall(PetscSFDistributeSection(sfMigration, section, &remoteOffsets, sectionDist));
  PetscCall(PetscObjectSetName((PetscObject)sectionDist, "Migrated Section"));
  if (debug) PetscCall(PetscSectionView(sectionDist, NULL));
  PetscCall(DMSetLocalSection(dm, sectionDist));
  /* If a sequential section is provided but no dof is affected, sfNatural cannot be computed and is set to NULL */
  PetscCall(PetscSectionGetStorageSize(sectionDist, &localSize));
  PetscCallMPI(MPI_Allreduce(&localSize, &maxStorageSize, 1, MPI_INT, MPI_MAX, PetscObjectComm((PetscObject)dm)));
  if (maxStorageSize) {
    const PetscInt *leaves;
    PetscInt       *sortleaves, *indices;
    PetscInt        Nl;

    /* Get a pruned version of migration SF */
    PetscCall(DMGetGlobalSection(dm, &gSection));
    if (debug) PetscCall(PetscSectionView(gSection, NULL));
    PetscCall(PetscSFGetGraph(sfMigration, NULL, &Nl, &leaves, NULL));
    PetscCall(PetscSectionGetChart(gSection, &pStart, &pEnd));
    for (p = pStart, ssize = 0; p < pEnd; ++p) {
      PetscInt dof, off;

      PetscCall(PetscSectionGetDof(gSection, p, &dof));
      PetscCall(PetscSectionGetOffset(gSection, p, &off));
      if ((dof > 0) && (off >= 0)) ++ssize;
    }
    PetscCall(PetscMalloc3(ssize, &spoints, Nl, &sortleaves, Nl, &indices));
    for (p = 0; p < Nl; ++p) {
      sortleaves[p] = leaves ? leaves[p] : p;
      indices[p]    = p;
    }
    PetscCall(PetscSortIntWithArray(Nl, sortleaves, indices));
    for (p = pStart, ssize = 0; p < pEnd; ++p) {
      PetscInt dof, off, loc;

      PetscCall(PetscSectionGetDof(gSection, p, &dof));
      PetscCall(PetscSectionGetOffset(gSection, p, &off));
      if ((dof > 0) && (off >= 0)) {
        PetscCall(PetscFindInt(p, Nl, sortleaves, &loc));
        PetscCheck(loc >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Point %" PetscInt_FMT " with nonzero dof is not a leaf of the migration SF", p);
        spoints[ssize++] = indices[loc];
      }
    }
    PetscCall(PetscSFCreateEmbeddedLeafSF(sfMigration, ssize, spoints, &sfEmbed));
    PetscCall(PetscObjectSetName((PetscObject)sfEmbed, "Embedded SF"));
    PetscCall(PetscFree3(spoints, sortleaves, indices));
    if (debug) PetscCall(PetscSFView(sfEmbed, NULL));
    /* Create the SF associated with this section
         Roots are natural dofs, leaves are global dofs */
    PetscCall(DMGetPointSF(dm, &sf));
    PetscCall(PetscSectionCreateGlobalSection(sectionDist, sf, PETSC_TRUE, PETSC_TRUE, &gLocSection));
    PetscCall(PetscSFCreateSectionSF(sfEmbed, section, remoteOffsets, gLocSection, &sfField));
    PetscCall(PetscSFDestroy(&sfEmbed));
    PetscCall(PetscSectionDestroy(&gLocSection));
    PetscCall(PetscObjectSetName((PetscObject)sfField, "Natural-to-Global SF"));
    if (debug) PetscCall(PetscSFView(sfField, NULL));
    /* Invert the field SF
         Roots are global dofs, leaves are natural dofs */
    PetscCall(PetscSFCreateInverseSF(sfField, sfNatural));
    PetscCall(PetscObjectSetName((PetscObject)*sfNatural, "Global-to-Natural SF"));
    PetscCall(PetscObjectViewFromOptions((PetscObject)*sfNatural, NULL, "-globaltonatural_sf_view"));
    /* Clean up */
    PetscCall(PetscSFDestroy(&sfField));
  } else {
    *sfNatural = NULL;
  }
  PetscCall(PetscSectionDestroy(&sectionDist));
  PetscCall(PetscFree(remoteOffsets));
  if (destroyFlag) PetscCall(PetscSectionDestroy(&section));
  PetscFunctionReturn(0);
}

/*@
  DMPlexGlobalToNaturalBegin - Rearranges a global `Vec` in the natural order.

  Collective on dm

  Input Parameters:
+ dm - The distributed `DMPLEX`
- gv - The global `Vec`

  Output Parameters:
. nv - `Vec` in the canonical ordering distributed over all processors associated with gv

  Level: intermediate

  Note:
  The user must call `DMSetUseNatural`(dm, `PETSC_TRUE`) before `DMPlexDistribute()`.

.seealso: [](chapter_unstructured), `DM`, `DMPLEX`, `Vec`, `DMPlexDistribute()`, `DMPlexDistributeField()`, `DMPlexNaturalToGlobalBegin()`, `DMPlexGlobalToNaturalEnd()`
@*/
PetscErrorCode DMPlexGlobalToNaturalBegin(DM dm, Vec gv, Vec nv)
{
  const PetscScalar *inarray;
  PetscScalar       *outarray;
  PetscMPIInt        size;

  PetscFunctionBegin;
  PetscCall(PetscLogEventBegin(DMPLEX_GlobalToNaturalBegin, dm, 0, 0, 0));
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)dm), &size));
  if (dm->sfNatural) {
    PetscCall(VecGetArray(nv, &outarray));
    PetscCall(VecGetArrayRead(gv, &inarray));
    PetscCall(PetscSFBcastBegin(dm->sfNatural, MPIU_SCALAR, (PetscScalar *)inarray, outarray, MPI_REPLACE));
    PetscCall(VecRestoreArrayRead(gv, &inarray));
    PetscCall(VecRestoreArray(nv, &outarray));
  } else if (size == 1) {
    PetscCall(VecCopy(gv, nv));
  } else {
    PetscCheck(!dm->useNatural, PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "DM global to natural SF not present.\nIf DMPlexDistribute() was called and a section was defined, report to petsc-maint@mcs.anl.gov.");
    SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONGSTATE, "DM global to natural SF was not created.\nYou must call DMSetUseNatural() before DMPlexDistribute().");
  }
  PetscCall(PetscLogEventEnd(DMPLEX_GlobalToNaturalBegin, dm, 0, 0, 0));
  PetscFunctionReturn(0);
}

/*@
  DMPlexGlobalToNaturalEnd - Rearranges a global `Vec` in the natural order.

  Collective on dm

  Input Parameters:
+ dm - The distributed `DMPLEX`
- gv - The global `Vec`

  Output Parameter:
. nv - The natural `Vec`

  Level: intermediate

  Note:
  The user must call `DMSetUseNatural`(dm, `PETSC_TRUE`) before `DMPlexDistribute()`.

 .seealso: [](chapter_unstructured), `DM`, `DMPLEX`, `Vec`, `DMPlexDistribute()`, `DMPlexDistributeField()`, `DMPlexNaturalToGlobalBegin()`, `DMPlexGlobalToNaturalBegin()`
 @*/
PetscErrorCode DMPlexGlobalToNaturalEnd(DM dm, Vec gv, Vec nv)
{
  const PetscScalar *inarray;
  PetscScalar       *outarray;
  PetscMPIInt        size;

  PetscFunctionBegin;
  PetscCall(PetscLogEventBegin(DMPLEX_GlobalToNaturalEnd, dm, 0, 0, 0));
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)dm), &size));
  if (dm->sfNatural) {
    PetscCall(VecGetArrayRead(gv, &inarray));
    PetscCall(VecGetArray(nv, &outarray));
    PetscCall(PetscSFBcastEnd(dm->sfNatural, MPIU_SCALAR, (PetscScalar *)inarray, outarray, MPI_REPLACE));
    PetscCall(VecRestoreArrayRead(gv, &inarray));
    PetscCall(VecRestoreArray(nv, &outarray));
  } else if (size == 1) {
  } else {
    PetscCheck(!dm->useNatural, PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "DM global to natural SF not present.\nIf DMPlexDistribute() was called and a section was defined, report to petsc-maint@mcs.anl.gov.");
    SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONGSTATE, "DM global to natural SF was not created.\nYou must call DMSetUseNatural() before DMPlexDistribute().");
  }
  PetscCall(PetscLogEventEnd(DMPLEX_GlobalToNaturalEnd, dm, 0, 0, 0));
  PetscFunctionReturn(0);
}

/*@
  DMPlexNaturalToGlobalBegin - Rearranges a `Vec` in the natural order to the Global order.

  Collective on dm

  Input Parameters:
+ dm - The distributed `DMPLEX`
- nv - The natural `Vec`

  Output Parameters:
. gv - The global `Vec`

  Level: intermediate

  Note:
  The user must call `DMSetUseNatural`(dm, `PETSC_TRUE`) before `DMPlexDistribute()`.

.seealso: [](chapter_unstructured), `DM`, `DMPLEX`, `Vec`, `DMPlexDistribute()`, `DMPlexDistributeField()`, `DMPlexNaturalToGlobalBegin()`, `DMPlexGlobalToNaturalEnd()`
@*/
PetscErrorCode DMPlexNaturalToGlobalBegin(DM dm, Vec nv, Vec gv)
{
  const PetscScalar *inarray;
  PetscScalar       *outarray;
  PetscMPIInt        size;

  PetscFunctionBegin;
  PetscCall(PetscLogEventBegin(DMPLEX_NaturalToGlobalBegin, dm, 0, 0, 0));
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)dm), &size));
  if (dm->sfNatural) {
    /* We only have access to the SF that goes from Global to Natural.
       Instead of inverting dm->sfNatural, we can call PetscSFReduceBegin/End with MPI_Op MPI_SUM.
       Here the SUM really does nothing since sfNatural is one to one, as long as gV is set to zero first. */
    PetscCall(VecZeroEntries(gv));
    PetscCall(VecGetArray(gv, &outarray));
    PetscCall(VecGetArrayRead(nv, &inarray));
    PetscCall(PetscSFReduceBegin(dm->sfNatural, MPIU_SCALAR, (PetscScalar *)inarray, outarray, MPI_SUM));
    PetscCall(VecRestoreArrayRead(nv, &inarray));
    PetscCall(VecRestoreArray(gv, &outarray));
  } else if (size == 1) {
    PetscCall(VecCopy(nv, gv));
  } else {
    PetscCheck(!dm->useNatural, PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "DM global to natural SF not present.\nIf DMPlexDistribute() was called and a section was defined, report to petsc-maint@mcs.anl.gov.");
    SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONGSTATE, "DM global to natural SF was not created.\nYou must call DMSetUseNatural() before DMPlexDistribute().");
  }
  PetscCall(PetscLogEventEnd(DMPLEX_NaturalToGlobalBegin, dm, 0, 0, 0));
  PetscFunctionReturn(0);
}

/*@
  DMPlexNaturalToGlobalEnd - Rearranges a `Vec` in the natural order to the Global order.

  Collective on dm

  Input Parameters:
+ dm - The distributed `DMPLEX`
- nv - The natural `Vec`

  Output Parameters:
. gv - The global `Vec`

  Level: intermediate

  Note:
  The user must call `DMSetUseNatural`(dm, `PETSC_TRUE`) before `DMPlexDistribute()`.

.seealso: [](chapter_unstructured), `DM`, `DMPLEX`, `Vec`, `DMPlexDistribute()`, `DMPlexDistributeField()`, `DMPlexNaturalToGlobalBegin()`, `DMPlexGlobalToNaturalBegin()`
 @*/
PetscErrorCode DMPlexNaturalToGlobalEnd(DM dm, Vec nv, Vec gv)
{
  const PetscScalar *inarray;
  PetscScalar       *outarray;
  PetscMPIInt        size;

  PetscFunctionBegin;
  PetscCall(PetscLogEventBegin(DMPLEX_NaturalToGlobalEnd, dm, 0, 0, 0));
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)dm), &size));
  if (dm->sfNatural) {
    PetscCall(VecGetArrayRead(nv, &inarray));
    PetscCall(VecGetArray(gv, &outarray));
    PetscCall(PetscSFReduceEnd(dm->sfNatural, MPIU_SCALAR, (PetscScalar *)inarray, outarray, MPI_SUM));
    PetscCall(VecRestoreArrayRead(nv, &inarray));
    PetscCall(VecRestoreArray(gv, &outarray));
  } else if (size == 1) {
  } else {
    PetscCheck(!dm->useNatural, PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "DM global to natural SF not present.\nIf DMPlexDistribute() was called and a section was defined, report to petsc-maint@mcs.anl.gov.");
    SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONGSTATE, "DM global to natural SF was not created.\nYou must call DMSetUseNatural() before DMPlexDistribute().");
  }
  PetscCall(PetscLogEventEnd(DMPLEX_NaturalToGlobalEnd, dm, 0, 0, 0));
  PetscFunctionReturn(0);
}

/*@
  DMPlexCreateNaturalVector - Provide a `Vec` capable of holding the natural ordering and distribution.

  Collective on dm

  Input Parameter:
. dm - The distributed `DMPLEX`

  Output Parameter:
. nv - The natural `Vec`

  Level: intermediate

  Note:
  The user must call `DMSetUseNatural`(dm, `PETSC_TRUE`) before `DMPlexDistribute()`.

.seealso: [](chapter_unstructured), `DM`, `DMPLEX`, `Vec`, `DMPlexDistribute()`, `DMPlexNaturalToGlobalBegin()`, `DMPlexGlobalToNaturalBegin()`
 @*/
PetscErrorCode DMPlexCreateNaturalVector(DM dm, Vec *nv)
{
  PetscMPIInt size;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)dm), &size));
  if (dm->sfNatural) {
    PetscInt nleaves, bs;
    Vec      v;
    PetscCall(DMGetLocalVector(dm, &v));
    PetscCall(VecGetBlockSize(v, &bs));
    PetscCall(DMRestoreLocalVector(dm, &v));

    PetscCall(PetscSFGetGraph(dm->sfNatural, NULL, &nleaves, NULL, NULL));
    PetscCall(VecCreate(PetscObjectComm((PetscObject)dm), nv));
    PetscCall(VecSetSizes(*nv, nleaves, PETSC_DETERMINE));
    PetscCall(VecSetBlockSize(*nv, bs));
    PetscCall(VecSetType(*nv, dm->vectype));
    PetscCall(VecSetDM(*nv, dm));
  } else if (size == 1) {
    PetscCall(DMCreateLocalVector(dm, nv));
  } else {
    PetscCheck(!dm->useNatural, PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "DM global to natural SF not present.\nIf DMPlexDistribute() was called and a section was defined, report to petsc-maint@mcs.anl.gov.");
    SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONGSTATE, "DM global to natural SF was not created.\nYou must call DMSetUseNatural() before DMPlexDistribute().");
  }
  PetscFunctionReturn(0);
}
