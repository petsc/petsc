#include <petsc/private/dmpleximpl.h> /*I      "petscdmplex.h"   I*/

/*@
  DMPlexSetMigrationSF - Sets the `PetscSF` for migrating from a parent `DM` into this `DM`

  Logically Collective

  Input Parameters:
+ dm          - The `DM`
- migrationSF - The `PetscSF`

  Level: intermediate

  Note:
  It is necessary to call this in order to have `DMCreateSubDM()` or `DMCreateSuperDM()` build the Global-To-Natural map

.seealso: [](ch_unstructured), `DM`, `DMPLEX`, `PetscSF`, `DMPlexDistribute()`, `DMPlexDistributeField()`, `DMPlexCreateMigrationSF()`, `DMPlexGetMigrationSF()`
@*/
PetscErrorCode DMPlexSetMigrationSF(DM dm, PetscSF migrationSF)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (migrationSF) PetscValidHeaderSpecific(migrationSF, PETSCSF_CLASSID, 2);
  PetscCall(PetscObjectReference((PetscObject)migrationSF));
  PetscCall(PetscSFDestroy(&dm->sfMigration));
  dm->sfMigration = migrationSF;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMPlexGetMigrationSF - Gets the `PetscSF` for migrating from a parent `DM` into this `DM`

  Not Collective

  Input Parameter:
. dm - The `DM`

  Output Parameter:
. migrationSF - The `PetscSF`

  Level: intermediate

.seealso: [](ch_unstructured), `DM`, `DMPLEX`, `PetscSF`, `DMPlexDistribute()`, `DMPlexDistributeField()`, `DMPlexCreateMigrationSF()`, `DMPlexSetMigrationSF`
@*/
PetscErrorCode DMPlexGetMigrationSF(DM dm, PetscSF *migrationSF)
{
  PetscFunctionBegin;
  *migrationSF = dm->sfMigration;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMPlexCreateGlobalToNaturalSF - Creates the `PetscSF` for mapping Global `Vec` to the Natural `Vec`

  Input Parameters:
+ dm          - The redistributed `DM`
. section     - The local `PetscSection` describing the `Vec` before the mesh was distributed, or `NULL` if not available
- sfMigration - The `PetscSF` used to distribute the mesh, or `NULL` if it cannot be computed

  Output Parameter:
. sfNatural - `PetscSF` for mapping the `Vec` in PETSc ordering to the canonical ordering

  Level: intermediate

  Note:
  This is not typically called by the user.

.seealso: [](ch_unstructured), `DM`, `DMPLEX`, `PetscSF`, `PetscSection`, `DMPlexDistribute()`, `DMPlexDistributeField()`
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
    PetscFunctionReturn(PETSC_SUCCESS);
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
  PetscCallMPI(MPIU_Allreduce(&localSize, &maxStorageSize, 1, MPIU_INT, MPI_MAX, PetscObjectComm((PetscObject)dm)));
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
    PetscCall(PetscSectionCreateGlobalSection(sectionDist, sf, PETSC_TRUE, PETSC_TRUE, PETSC_TRUE, &gLocSection));
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
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMPlexMigrateGlobalToNaturalSF - Migrates the input `sfNatural` based on sfMigration

  Input Parameters:
+ dmOld        - The original `DM`
. dmNew        - The `DM` to be migrated to
. sfNaturalOld - The sfNatural for the `dmOld`
- sfMigration  - The `PetscSF` used to distribute the mesh, or `NULL` if it cannot be computed

  Output Parameter:
. sfNaturalNew - `PetscSF` for mapping the `Vec` in PETSc ordering to the canonical ordering

  Level: intermediate

  Notes:
  `sfNaturalOld` maps from the old Global section (roots) to the natural Vec layout (leaves, may or may not be described by a PetscSection).
  `DMPlexMigrateGlobalToNaturalSF` creates an SF to map from the old global section to the new global section (generated from `sfMigration`).
  That SF is then composed with the `sfNaturalOld` to generate `sfNaturalNew`.
  This also distributes and sets the local section for `dmNew`.

  This is not typically called by the user.

.seealso: [](ch_unstructured), `DM`, `DMPLEX`, `PetscSF`, `PetscSection`, `DMPlexDistribute()`, `DMPlexDistributeField()`
 @*/
PetscErrorCode DMPlexMigrateGlobalToNaturalSF(DM dmOld, DM dmNew, PetscSF sfNaturalOld, PetscSF sfMigration, PetscSF *sfNaturalNew)
{
  MPI_Comm     comm;
  PetscSection oldGlobalSection, newGlobalSection;
  PetscInt    *remoteOffsets;
  PetscBool    debug = PETSC_FALSE;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)dmNew, &comm));
  if (!sfMigration) {
    /* If sfMigration is missing, sfNatural cannot be computed and is set to NULL */
    *sfNaturalNew = NULL;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  if (debug) PetscCall(PetscSFView(sfMigration, NULL));

  { // Create oldGlobalSection and newGlobalSection *with* localOffsets
    PetscSection oldLocalSection, newLocalSection;
    PetscSF      pointSF;

    PetscCall(DMGetLocalSection(dmOld, &oldLocalSection));
    PetscCall(DMGetPointSF(dmOld, &pointSF));
    PetscCall(PetscSectionCreateGlobalSection(oldLocalSection, pointSF, PETSC_TRUE, PETSC_TRUE, PETSC_TRUE, &oldGlobalSection));

    PetscCall(PetscSectionCreate(comm, &newLocalSection));
    PetscCall(PetscSFDistributeSection(sfMigration, oldLocalSection, NULL, newLocalSection));
    PetscCall(DMSetLocalSection(dmNew, newLocalSection));

    PetscCall(DMGetPointSF(dmNew, &pointSF));
    PetscCall(PetscSectionCreateGlobalSection(newLocalSection, pointSF, PETSC_TRUE, PETSC_TRUE, PETSC_TRUE, &newGlobalSection));

    PetscCall(PetscObjectSetName((PetscObject)oldLocalSection, "Old Local Section"));
    if (debug) PetscCall(PetscSectionView(oldLocalSection, NULL));
    PetscCall(PetscObjectSetName((PetscObject)oldGlobalSection, "Old Global Section"));
    if (debug) PetscCall(PetscSectionView(oldGlobalSection, NULL));
    PetscCall(PetscObjectSetName((PetscObject)newLocalSection, "New Local Section"));
    if (debug) PetscCall(PetscSectionView(newLocalSection, NULL));
    PetscCall(PetscObjectSetName((PetscObject)newGlobalSection, "New Global Section"));
    if (debug) PetscCall(PetscSectionView(newGlobalSection, NULL));
    PetscCall(PetscSectionDestroy(&newLocalSection));
  }

  { // Create remoteOffsets array, mapping the oldGlobalSection offsets to the local points (according to sfMigration)
    PetscInt lpStart, lpEnd, rpStart, rpEnd;

    PetscCall(PetscSectionGetChart(oldGlobalSection, &rpStart, &rpEnd));
    PetscCall(PetscSectionGetChart(newGlobalSection, &lpStart, &lpEnd));

    // in `PetscSFDistributeSection` (where this is taken from), it possibly makes a new embedded SF. Should possibly do that here?
    PetscCall(PetscMalloc1(lpEnd - lpStart, &remoteOffsets));
    PetscCall(PetscSFBcastBegin(sfMigration, MPIU_INT, PetscSafePointerPlusOffset(oldGlobalSection->atlasOff, -rpStart), PetscSafePointerPlusOffset(remoteOffsets, -lpStart), MPI_REPLACE));
    PetscCall(PetscSFBcastEnd(sfMigration, MPIU_INT, PetscSafePointerPlusOffset(oldGlobalSection->atlasOff, -rpStart), PetscSafePointerPlusOffset(remoteOffsets, -lpStart), MPI_REPLACE));
    if (debug) {
      PetscViewer viewer;

      PetscCall(PetscPrintf(comm, "Remote Offsets:\n"));
      PetscCall(PetscViewerASCIIGetStdout(comm, &viewer));
      PetscCall(PetscIntView(lpEnd - lpStart, remoteOffsets, viewer));
    }
  }

  { // Create SF from oldGlobalSection to newGlobalSection and compose with sfNaturalOld
    PetscSF oldglob_to_newglob_sf, newglob_to_oldglob_sf;

    PetscCall(PetscSFCreateSectionSF(sfMigration, oldGlobalSection, remoteOffsets, newGlobalSection, &oldglob_to_newglob_sf));
    PetscCall(PetscObjectSetName((PetscObject)oldglob_to_newglob_sf, "OldGlobal-to-NewGlobal SF"));
    if (debug) PetscCall(PetscSFView(oldglob_to_newglob_sf, NULL));

    PetscCall(PetscSFCreateInverseSF(oldglob_to_newglob_sf, &newglob_to_oldglob_sf));
    PetscCall(PetscObjectSetName((PetscObject)newglob_to_oldglob_sf, "NewGlobal-to-OldGlobal SF"));
    PetscCall(PetscObjectViewFromOptions((PetscObject)newglob_to_oldglob_sf, (PetscObject)dmOld, "-natural_migrate_sf_view"));
    PetscCall(PetscSFCompose(newglob_to_oldglob_sf, sfNaturalOld, sfNaturalNew));

    PetscCall(PetscSFDestroy(&oldglob_to_newglob_sf));
    PetscCall(PetscSFDestroy(&newglob_to_oldglob_sf));
  }

  PetscCall(PetscSectionDestroy(&oldGlobalSection));
  PetscCall(PetscSectionDestroy(&newGlobalSection));
  PetscCall(PetscFree(remoteOffsets));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMPlexGlobalToNaturalBegin - Rearranges a global `Vec` in the natural order.

  Collective

  Input Parameters:
+ dm - The distributed `DMPLEX`
- gv - The global `Vec`

  Output Parameter:
. nv - `Vec` in the canonical ordering distributed over all processors associated with `gv`

  Level: intermediate

  Note:
  The user must call `DMSetUseNatural`(dm, `PETSC_TRUE`) before `DMPlexDistribute()`.

.seealso: [](ch_unstructured), `DM`, `DMPLEX`, `Vec`, `DMPlexDistribute()`, `DMPlexDistributeField()`, `DMPlexNaturalToGlobalBegin()`, `DMPlexGlobalToNaturalEnd()`
@*/
PetscErrorCode DMPlexGlobalToNaturalBegin(DM dm, Vec gv, Vec nv)
{
  const PetscScalar *inarray;
  PetscScalar       *outarray;
  MPI_Comm           comm;
  PetscMPIInt        size;

  PetscFunctionBegin;
  PetscCall(PetscLogEventBegin(DMPLEX_GlobalToNaturalBegin, dm, 0, 0, 0));
  PetscCall(PetscObjectGetComm((PetscObject)dm, &comm));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  if (dm->sfNatural) {
    if (PetscDefined(USE_DEBUG)) {
      PetscSection gs;
      PetscInt     Nl, n;

      PetscCall(PetscSFGetGraph(dm->sfNatural, NULL, &Nl, NULL, NULL));
      PetscCall(VecGetLocalSize(nv, &n));
      PetscCheck(n == Nl, comm, PETSC_ERR_ARG_INCOMP, "Natural vector local size %" PetscInt_FMT " != %" PetscInt_FMT " local size of natural section", n, Nl);

      PetscCall(DMGetGlobalSection(dm, &gs));
      PetscCall(PetscSectionGetConstrainedStorageSize(gs, &Nl));
      PetscCall(VecGetLocalSize(gv, &n));
      PetscCheck(n == Nl, comm, PETSC_ERR_ARG_INCOMP, "Global vector local size %" PetscInt_FMT " != %" PetscInt_FMT " local size of global section", n, Nl);
    }
    PetscCall(VecGetArray(nv, &outarray));
    PetscCall(VecGetArrayRead(gv, &inarray));
    PetscCall(PetscSFBcastBegin(dm->sfNatural, MPIU_SCALAR, (PetscScalar *)inarray, outarray, MPI_REPLACE));
    PetscCall(VecRestoreArrayRead(gv, &inarray));
    PetscCall(VecRestoreArray(nv, &outarray));
  } else if (size == 1) {
    PetscCall(VecCopy(gv, nv));
  } else {
    PetscCheck(!dm->useNatural, PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "DM global to natural SF not present. If DMPlexDistribute() was called and a section was defined, report to petsc-maint@mcs.anl.gov.");
    SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONGSTATE, "DM global to natural SF was not created. You must call DMSetUseNatural() before DMPlexDistribute().");
  }
  PetscCall(PetscLogEventEnd(DMPLEX_GlobalToNaturalBegin, dm, 0, 0, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMPlexGlobalToNaturalEnd - Rearranges a global `Vec` in the natural order.

  Collective

  Input Parameters:
+ dm - The distributed `DMPLEX`
- gv - The global `Vec`

  Output Parameter:
. nv - The natural `Vec`

  Level: intermediate

  Note:
  The user must call `DMSetUseNatural`(dm, `PETSC_TRUE`) before `DMPlexDistribute()`.

.seealso: [](ch_unstructured), `DM`, `DMPLEX`, `Vec`, `DMPlexDistribute()`, `DMPlexDistributeField()`, `DMPlexNaturalToGlobalBegin()`, `DMPlexGlobalToNaturalBegin()`
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
    PetscCheck(!dm->useNatural, PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "DM global to natural SF not present. If DMPlexDistribute() was called and a section was defined, report to petsc-maint@mcs.anl.gov.");
    SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONGSTATE, "DM global to natural SF was not created. You must call DMSetUseNatural() before DMPlexDistribute().");
  }
  PetscCall(PetscLogEventEnd(DMPLEX_GlobalToNaturalEnd, dm, 0, 0, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMPlexNaturalToGlobalBegin - Rearranges a `Vec` in the natural order to the Global order.

  Collective

  Input Parameters:
+ dm - The distributed `DMPLEX`
- nv - The natural `Vec`

  Output Parameter:
. gv - The global `Vec`

  Level: intermediate

  Note:
  The user must call `DMSetUseNatural`(dm, `PETSC_TRUE`) before `DMPlexDistribute()`.

.seealso: [](ch_unstructured), `DM`, `DMPLEX`, `Vec`, `DMPlexDistribute()`, `DMPlexDistributeField()`, `DMPlexGlobalToNaturalEnd()`
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
    PetscCheck(!dm->useNatural, PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "DM global to natural SF not present. If DMPlexDistribute() was called and a section was defined, report to petsc-maint@mcs.anl.gov.");
    SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONGSTATE, "DM global to natural SF was not created. You must call DMSetUseNatural() before DMPlexDistribute().");
  }
  PetscCall(PetscLogEventEnd(DMPLEX_NaturalToGlobalBegin, dm, 0, 0, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMPlexNaturalToGlobalEnd - Rearranges a `Vec` in the natural order to the Global order.

  Collective

  Input Parameters:
+ dm - The distributed `DMPLEX`
- nv - The natural `Vec`

  Output Parameter:
. gv - The global `Vec`

  Level: intermediate

  Note:
  The user must call `DMSetUseNatural`(dm, `PETSC_TRUE`) before `DMPlexDistribute()`.

.seealso: [](ch_unstructured), `DM`, `DMPLEX`, `Vec`, `DMPlexDistribute()`, `DMPlexDistributeField()`, `DMPlexNaturalToGlobalBegin()`, `DMPlexGlobalToNaturalBegin()`
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
    PetscCheck(!dm->useNatural, PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "DM global to natural SF not present. If DMPlexDistribute() was called and a section was defined, report to petsc-maint@mcs.anl.gov.");
    SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONGSTATE, "DM global to natural SF was not created. You must call DMSetUseNatural() before DMPlexDistribute().");
  }
  PetscCall(PetscLogEventEnd(DMPLEX_NaturalToGlobalEnd, dm, 0, 0, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMPlexCreateNaturalVector - Provide a `Vec` capable of holding the natural ordering and distribution.

  Collective

  Input Parameter:
. dm - The distributed `DMPLEX`

  Output Parameter:
. nv - The natural `Vec`

  Level: intermediate

  Note:
  The user must call `DMSetUseNatural`(dm, `PETSC_TRUE`) before `DMPlexDistribute()`.

.seealso: [](ch_unstructured), `DM`, `DMPLEX`, `Vec`, `DMPlexDistribute()`, `DMPlexNaturalToGlobalBegin()`, `DMPlexGlobalToNaturalBegin()`
 @*/
PetscErrorCode DMPlexCreateNaturalVector(DM dm, Vec *nv)
{
  PetscMPIInt size;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)dm), &size));
  if (dm->sfNatural) {
    PetscInt nleaves, bs, maxbs;
    Vec      v;

    /*
      Setting the natural vector block size.
      We can't get it from a global vector because of constraints, and the block size in the local vector
      may be inconsistent across processes, typically when some local vectors have size 0, their block size is set to 1
    */
    PetscCall(DMGetLocalVector(dm, &v));
    PetscCall(VecGetBlockSize(v, &bs));
    PetscCallMPI(MPIU_Allreduce(&bs, &maxbs, 1, MPIU_INT, MPI_MAX, PetscObjectComm((PetscObject)dm)));
    if (bs == 1 && maxbs > 1) bs = maxbs;
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
    PetscCheck(!dm->useNatural, PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "DM global to natural SF not present. If DMPlexDistribute() was called and a section was defined, report to petsc-maint@mcs.anl.gov.");
    SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONGSTATE, "DM global to natural SF was not created. You must call DMSetUseNatural() before DMPlexDistribute().");
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
