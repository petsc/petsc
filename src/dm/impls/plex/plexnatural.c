#include <petsc/private/dmpleximpl.h>   /*I      "petscdmplex.h"   I*/

/*@
  DMPlexSetMigrationSF - Sets the SF for migrating from a parent DM into this DM

  Input Parameters:
+ dm        - The DM
- naturalSF - The PetscSF

  Note: It is necessary to call this in order to have DMCreateSubDM() or DMCreateSuperDM() build the Global-To-Natural map

  Level: intermediate

.seealso: `DMPlexDistribute()`, `DMPlexDistributeField()`, `DMPlexCreateMigrationSF()`, `DMPlexGetMigrationSF()`
@*/
PetscErrorCode DMPlexSetMigrationSF(DM dm, PetscSF migrationSF)
{
  PetscFunctionBegin;
  dm->sfMigration = migrationSF;
  PetscCall(PetscObjectReference((PetscObject) migrationSF));
  PetscFunctionReturn(0);
}

/*@
  DMPlexGetMigrationSF - Gets the SF for migrating from a parent DM into this DM

  Input Parameter:
. dm          - The DM

  Output Parameter:
. migrationSF - The PetscSF

  Level: intermediate

.seealso: `DMPlexDistribute()`, `DMPlexDistributeField()`, `DMPlexCreateMigrationSF()`, `DMPlexSetMigrationSF`
@*/
PetscErrorCode DMPlexGetMigrationSF(DM dm, PetscSF *migrationSF)
{
  PetscFunctionBegin;
  *migrationSF = dm->sfMigration;
  PetscFunctionReturn(0);
}

/*@
  DMPlexSetGlobalToNaturalSF - Sets the SF for mapping Global Vec to the Natural Vec

  Input Parameters:
+ dm          - The DM
- naturalSF   - The PetscSF

  Level: intermediate

.seealso: `DMPlexDistribute()`, `DMPlexDistributeField()`, `DMPlexCreateGlobalToNaturalSF()`, `DMPlexGetGlobaltoNaturalSF()`
@*/
PetscErrorCode DMPlexSetGlobalToNaturalSF(DM dm, PetscSF naturalSF)
{
  PetscFunctionBegin;
  dm->sfNatural = naturalSF;
  PetscCall(PetscObjectReference((PetscObject) naturalSF));
  dm->useNatural = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*@
  DMPlexGetGlobalToNaturalSF - Gets the SF for mapping Global Vec to the Natural Vec

  Input Parameter:
. dm          - The DM

  Output Parameter:
. naturalSF   - The PetscSF

  Level: intermediate

.seealso: `DMPlexDistribute()`, `DMPlexDistributeField()`, `DMPlexCreateGlobalToNaturalSF()`, `DMPlexSetGlobaltoNaturalSF`
@*/
PetscErrorCode DMPlexGetGlobalToNaturalSF(DM dm, PetscSF *naturalSF)
{
  PetscFunctionBegin;
  *naturalSF = dm->sfNatural;
  PetscFunctionReturn(0);
}

/*@
  DMPlexCreateGlobalToNaturalSF - Creates the SF for mapping Global Vec to the Natural Vec

  Input Parameters:
+ dm          - The DM
. section     - The PetscSection describing the Vec before the mesh was distributed,
                or NULL if not available
- sfMigration - The PetscSF used to distribute the mesh, or NULL if it cannot be computed

  Output Parameter:
. sfNatural   - PetscSF for mapping the Vec in PETSc ordering to the canonical ordering

  Note: This is not typically called by the user.

  Level: intermediate

.seealso: `DMPlexDistribute()`, `DMPlexDistributeField()`
 @*/
PetscErrorCode DMPlexCreateGlobalToNaturalSF(DM dm, PetscSection section, PetscSF sfMigration, PetscSF *sfNatural)
{
  MPI_Comm       comm;
  Vec            gv, tmpVec;
  PetscSF        sf, sfEmbed, sfSeqToNatural, sfField, sfFieldInv;
  PetscSection   gSection, sectionDist, gLocSection;
  PetscInt      *spoints, *remoteOffsets;
  PetscInt       ssize, pStart, pEnd, p, globalSize;
  PetscLayout    map;
  PetscBool      destroyFlag = PETSC_FALSE;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject) dm, &comm));
  if (!sfMigration) {
    /* If sfMigration is missing,
    sfNatural cannot be computed and is set to NULL */
    *sfNatural = NULL;
    PetscFunctionReturn(0);
  } else if (!section) {
    /* If the sequential section is not provided (NULL),
    it is reconstructed from the parallel section */
    PetscSF sfMigrationInv;
    PetscSection localSection;

    PetscCall(DMGetLocalSection(dm, &localSection));
    PetscCall(PetscSFCreateInverseSF(sfMigration, &sfMigrationInv));
    PetscCall(PetscSectionCreate(PetscObjectComm((PetscObject) dm), &section));
    PetscCall(PetscSFDistributeSection(sfMigrationInv, localSection, NULL, section));
    PetscCall(PetscSFDestroy(&sfMigrationInv));
    destroyFlag = PETSC_TRUE;
  }
  /* PetscCall(PetscPrintf(comm, "Point migration SF\n"));
   PetscCall(PetscSFView(sfMigration, 0)); */
  /* Create a new section from distributing the original section */
  PetscCall(PetscSectionCreate(comm, &sectionDist));
  PetscCall(PetscSFDistributeSection(sfMigration, section, &remoteOffsets, sectionDist));
  /* PetscCall(PetscPrintf(comm, "Distributed Section\n"));
   PetscCall(PetscSectionView(sectionDist, PETSC_VIEWER_STDOUT_WORLD)); */
  PetscCall(DMSetLocalSection(dm, sectionDist));
  /* If a sequential section is provided but no dof is affected,
  sfNatural cannot be computed and is set to NULL */
  PetscCall(DMCreateGlobalVector(dm, &tmpVec));
  PetscCall(VecGetSize(tmpVec, &globalSize));
  PetscCall(DMRestoreGlobalVector(dm, &tmpVec));
  if (globalSize) {
  /* Get a pruned version of migration SF */
    PetscCall(DMGetGlobalSection(dm, &gSection));
    PetscCall(PetscSectionGetChart(gSection, &pStart, &pEnd));
    for (p = pStart, ssize = 0; p < pEnd; ++p) {
      PetscInt dof, off;

      PetscCall(PetscSectionGetDof(gSection, p, &dof));
      PetscCall(PetscSectionGetOffset(gSection, p, &off));
      if ((dof > 0) && (off >= 0)) ++ssize;
    }
    PetscCall(PetscMalloc1(ssize, &spoints));
    for (p = pStart, ssize = 0; p < pEnd; ++p) {
      PetscInt dof, off;

      PetscCall(PetscSectionGetDof(gSection, p, &dof));
      PetscCall(PetscSectionGetOffset(gSection, p, &off));
      if ((dof > 0) && (off >= 0)) spoints[ssize++] = p;
    }
    PetscCall(PetscSFCreateEmbeddedLeafSF(sfMigration, ssize, spoints, &sfEmbed));
    PetscCall(PetscFree(spoints));
    /* PetscCall(PetscPrintf(comm, "Embedded SF\n"));
    PetscCall(PetscSFView(sfEmbed, 0)); */
    /* Create the SF for seq to natural */
    PetscCall(DMGetGlobalVector(dm, &gv));
    PetscCall(VecGetLayout(gv,&map));
    /* Note that entries of gv are leaves in sfSeqToNatural, entries of the seq vec are roots */
    PetscCall(PetscSFCreate(comm, &sfSeqToNatural));
    PetscCall(PetscSFSetGraphWithPattern(sfSeqToNatural, map, PETSCSF_PATTERN_GATHER));
    PetscCall(DMRestoreGlobalVector(dm, &gv));
    /* PetscCall(PetscPrintf(comm, "Seq-to-Natural SF\n"));
    PetscCall(PetscSFView(sfSeqToNatural, 0)); */
    /* Create the SF associated with this section */
    PetscCall(DMGetPointSF(dm, &sf));
    PetscCall(PetscSectionCreateGlobalSection(sectionDist, sf, PETSC_FALSE, PETSC_TRUE, &gLocSection));
    PetscCall(PetscSFCreateSectionSF(sfEmbed, section, remoteOffsets, gLocSection, &sfField));
    PetscCall(PetscSFDestroy(&sfEmbed));
    PetscCall(PetscSectionDestroy(&gLocSection));
    /* PetscCall(PetscPrintf(comm, "Field SF\n"));
    PetscCall(PetscSFView(sfField, 0)); */
    /* Invert the field SF so it's now from distributed to sequential */
    PetscCall(PetscSFCreateInverseSF(sfField, &sfFieldInv));
    PetscCall(PetscSFDestroy(&sfField));
    /* PetscCall(PetscPrintf(comm, "Inverse Field SF\n"));
    PetscCall(PetscSFView(sfFieldInv, 0)); */
    /* Multiply the sfFieldInv with the */
    PetscCall(PetscSFComposeInverse(sfFieldInv, sfSeqToNatural, sfNatural));
    PetscCall(PetscObjectViewFromOptions((PetscObject) *sfNatural, NULL, "-globaltonatural_sf_view"));
    /* Clean up */
    PetscCall(PetscSFDestroy(&sfFieldInv));
    PetscCall(PetscSFDestroy(&sfSeqToNatural));
  } else {
    *sfNatural = NULL;
  }
  PetscCall(PetscSectionDestroy(&sectionDist));
  PetscCall(PetscFree(remoteOffsets));
  if (destroyFlag) PetscCall(PetscSectionDestroy(&section));
  PetscFunctionReturn(0);
}

/*@
  DMPlexGlobalToNaturalBegin - Rearranges a global Vector in the natural order.

  Collective on dm

  Input Parameters:
+ dm - The distributed DMPlex
- gv - The global Vec

  Output Parameters:
. nv - Vec in the canonical ordering distributed over all processors associated with gv

  Note: The user must call DMSetUseNatural(dm, PETSC_TRUE) before DMPlexDistribute().

  Level: intermediate

.seealso: `DMPlexDistribute()`, `DMPlexDistributeField()`, `DMPlexNaturalToGlobalBegin()`, `DMPlexGlobalToNaturalEnd()`
@*/
PetscErrorCode DMPlexGlobalToNaturalBegin(DM dm, Vec gv, Vec nv)
{
  const PetscScalar *inarray;
  PetscScalar       *outarray;
  PetscMPIInt        size;

  PetscFunctionBegin;
  PetscCall(PetscLogEventBegin(DMPLEX_GlobalToNaturalBegin,dm,0,0,0));
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject) dm), &size));
  if (dm->sfNatural) {
    PetscCall(VecGetArray(nv, &outarray));
    PetscCall(VecGetArrayRead(gv, &inarray));
    PetscCall(PetscSFBcastBegin(dm->sfNatural, MPIU_SCALAR, (PetscScalar *) inarray, outarray,MPI_REPLACE));
    PetscCall(VecRestoreArrayRead(gv, &inarray));
    PetscCall(VecRestoreArray(nv, &outarray));
  } else if (size == 1) {
    PetscCall(VecCopy(gv, nv));
  } else PetscCheck(!dm->useNatural,PetscObjectComm((PetscObject) dm), PETSC_ERR_PLIB, "DM global to natural SF not present.\nIf DMPlexDistribute() was called and a section was defined, report to petsc-maint@mcs.anl.gov.");
  else SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_WRONGSTATE, "DM global to natural SF was not created.\nYou must call DMSetUseNatural() before DMPlexDistribute().");
  PetscCall(PetscLogEventEnd(DMPLEX_GlobalToNaturalBegin,dm,0,0,0));
  PetscFunctionReturn(0);
}

/*@
  DMPlexGlobalToNaturalEnd - Rearranges a global Vector in the natural order.

  Collective on dm

  Input Parameters:
+ dm - The distributed DMPlex
- gv - The global Vec

  Output Parameter:
. nv - The natural Vec

  Note: The user must call DMSetUseNatural(dm, PETSC_TRUE) before DMPlexDistribute().

  Level: intermediate

 .seealso: `DMPlexDistribute()`, `DMPlexDistributeField()`, `DMPlexNaturalToGlobalBegin()`, `DMPlexGlobalToNaturalBegin()`
 @*/
PetscErrorCode DMPlexGlobalToNaturalEnd(DM dm, Vec gv, Vec nv)
{
  const PetscScalar *inarray;
  PetscScalar       *outarray;
  PetscMPIInt        size;

  PetscFunctionBegin;
  PetscCall(PetscLogEventBegin(DMPLEX_GlobalToNaturalEnd,dm,0,0,0));
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject) dm), &size));
  if (dm->sfNatural) {
    PetscCall(VecGetArrayRead(gv, &inarray));
    PetscCall(VecGetArray(nv, &outarray));
    PetscCall(PetscSFBcastEnd(dm->sfNatural, MPIU_SCALAR, (PetscScalar *) inarray, outarray,MPI_REPLACE));
    PetscCall(VecRestoreArrayRead(gv, &inarray));
    PetscCall(VecRestoreArray(nv, &outarray));
  } else if (size == 1) {
  } else PetscCheck(!dm->useNatural,PetscObjectComm((PetscObject) dm), PETSC_ERR_PLIB, "DM global to natural SF not present.\nIf DMPlexDistribute() was called and a section was defined, report to petsc-maint@mcs.anl.gov.");
  else SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_WRONGSTATE, "DM global to natural SF was not created.\nYou must call DMSetUseNatural() before DMPlexDistribute().");
  PetscCall(PetscLogEventEnd(DMPLEX_GlobalToNaturalEnd,dm,0,0,0));
  PetscFunctionReturn(0);
}

/*@
  DMPlexNaturalToGlobalBegin - Rearranges a Vector in the natural order to the Global order.

  Collective on dm

  Input Parameters:
+ dm - The distributed DMPlex
- nv - The natural Vec

  Output Parameters:
. gv - The global Vec

  Note: The user must call DMSetUseNatural(dm, PETSC_TRUE) before DMPlexDistribute().

  Level: intermediate

.seealso: `DMPlexDistribute()`, `DMPlexDistributeField()`, `DMPlexNaturalToGlobalBegin()`, `DMPlexGlobalToNaturalEnd()`
@*/
PetscErrorCode DMPlexNaturalToGlobalBegin(DM dm, Vec nv, Vec gv)
{
  const PetscScalar *inarray;
  PetscScalar       *outarray;
  PetscMPIInt        size;

  PetscFunctionBegin;
  PetscCall(PetscLogEventBegin(DMPLEX_NaturalToGlobalBegin,dm,0,0,0));
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject) dm), &size));
  if (dm->sfNatural) {
    /* We only have access to the SF that goes from Global to Natural.
       Instead of inverting dm->sfNatural, we can call PetscSFReduceBegin/End with MPI_Op MPI_SUM.
       Here the SUM really does nothing since sfNatural is one to one, as long as gV is set to zero first. */
    PetscCall(VecZeroEntries(gv));
    PetscCall(VecGetArray(gv, &outarray));
    PetscCall(VecGetArrayRead(nv, &inarray));
    PetscCall(PetscSFReduceBegin(dm->sfNatural, MPIU_SCALAR, (PetscScalar *) inarray, outarray, MPI_SUM));
    PetscCall(VecRestoreArrayRead(nv, &inarray));
    PetscCall(VecRestoreArray(gv, &outarray));
  } else if (size == 1) {
    PetscCall(VecCopy(nv, gv));
  } else PetscCheck(!dm->useNatural,PetscObjectComm((PetscObject) dm), PETSC_ERR_PLIB, "DM global to natural SF not present.\nIf DMPlexDistribute() was called and a section was defined, report to petsc-maint@mcs.anl.gov.");
  else SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_WRONGSTATE, "DM global to natural SF was not created.\nYou must call DMSetUseNatural() before DMPlexDistribute().");
  PetscCall(PetscLogEventEnd(DMPLEX_NaturalToGlobalBegin,dm,0,0,0));
  PetscFunctionReturn(0);
}

/*@
  DMPlexNaturalToGlobalEnd - Rearranges a Vector in the natural order to the Global order.

  Collective on dm

  Input Parameters:
+ dm - The distributed DMPlex
- nv - The natural Vec

  Output Parameters:
. gv - The global Vec

  Note: The user must call DMSetUseNatural(dm, PETSC_TRUE) before DMPlexDistribute().

  Level: intermediate

.seealso: `DMPlexDistribute()`, `DMPlexDistributeField()`, `DMPlexNaturalToGlobalBegin()`, `DMPlexGlobalToNaturalBegin()`
 @*/
PetscErrorCode DMPlexNaturalToGlobalEnd(DM dm, Vec nv, Vec gv)
{
  const PetscScalar *inarray;
  PetscScalar       *outarray;
  PetscMPIInt        size;

  PetscFunctionBegin;
  PetscCall(PetscLogEventBegin(DMPLEX_NaturalToGlobalEnd,dm,0,0,0));
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject) dm), &size));
  if (dm->sfNatural) {
    PetscCall(VecGetArrayRead(nv, &inarray));
    PetscCall(VecGetArray(gv, &outarray));
    PetscCall(PetscSFReduceEnd(dm->sfNatural, MPIU_SCALAR, (PetscScalar *) inarray, outarray, MPI_SUM));
    PetscCall(VecRestoreArrayRead(nv, &inarray));
    PetscCall(VecRestoreArray(gv, &outarray));
  } else if (size == 1) {
  } else PetscCheck(!dm->useNatural,PetscObjectComm((PetscObject) dm), PETSC_ERR_PLIB, "DM global to natural SF not present.\nIf DMPlexDistribute() was called and a section was defined, report to petsc-maint@mcs.anl.gov.");
  else SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_WRONGSTATE, "DM global to natural SF was not created.\nYou must call DMSetUseNatural() before DMPlexDistribute().");
  PetscCall(PetscLogEventEnd(DMPLEX_NaturalToGlobalEnd,dm,0,0,0));
  PetscFunctionReturn(0);
}
