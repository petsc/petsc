#include <petsc/private/dmpleximpl.h>   /*I      "petscdmplex.h"   I*/

/*@
  DMPlexSetMigrationSF - Sets the SF for migrating from a parent DM into this DM

  Input Parameters:
+ dm        - The DM
- naturalSF - The PetscSF

  Note: It is necessary to call this in order to have DMCreateSubDM() or DMCreateSuperDM() build the Global-To-Natural map

  Level: intermediate

.seealso: DMPlexDistribute(), DMPlexDistributeField(), DMPlexCreateMigrationSF(), DMPlexGetMigrationSF()
@*/
PetscErrorCode DMPlexSetMigrationSF(DM dm, PetscSF migrationSF)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  dm->sfMigration = migrationSF;
  ierr = PetscObjectReference((PetscObject) migrationSF);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  DMPlexGetMigrationSF - Gets the SF for migrating from a parent DM into this DM

  Input Parameter:
. dm          - The DM

  Output Parameter:
. migrationSF - The PetscSF

  Level: intermediate

.seealso: DMPlexDistribute(), DMPlexDistributeField(), DMPlexCreateMigrationSF(), DMPlexSetMigrationSF
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

.seealso: DMPlexDistribute(), DMPlexDistributeField(), DMPlexCreateGlobalToNaturalSF(), DMPlexGetGlobaltoNaturalSF()
@*/
PetscErrorCode DMPlexSetGlobalToNaturalSF(DM dm, PetscSF naturalSF)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  dm->sfNatural = naturalSF;
  ierr = PetscObjectReference((PetscObject) naturalSF);CHKERRQ(ierr);
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

.seealso: DMPlexDistribute(), DMPlexDistributeField(), DMPlexCreateGlobalToNaturalSF(), DMPlexSetGlobaltoNaturalSF
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

.seealso: DMPlexDistribute(), DMPlexDistributeField()
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject) dm, &comm);CHKERRQ(ierr);
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

    ierr = DMGetLocalSection(dm, &localSection);CHKERRQ(ierr);
    ierr = PetscSFCreateInverseSF(sfMigration, &sfMigrationInv);CHKERRQ(ierr);
    ierr = PetscSectionCreate(PetscObjectComm((PetscObject) dm), &section);CHKERRQ(ierr);
    ierr = PetscSFDistributeSection(sfMigrationInv, localSection, NULL, section);CHKERRQ(ierr);
    ierr = PetscSFDestroy(&sfMigrationInv);CHKERRQ(ierr);
    destroyFlag = PETSC_TRUE;
  }
  /* ierr = PetscPrintf(comm, "Point migration SF\n");CHKERRQ(ierr);
   ierr = PetscSFView(sfMigration, 0);CHKERRQ(ierr); */
  /* Create a new section from distributing the original section */
  ierr = PetscSectionCreate(comm, &sectionDist);CHKERRQ(ierr);
  ierr = PetscSFDistributeSection(sfMigration, section, &remoteOffsets, sectionDist);CHKERRQ(ierr);
  /* ierr = PetscPrintf(comm, "Distributed Section\n");CHKERRQ(ierr);
   ierr = PetscSectionView(sectionDist, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr); */
  ierr = DMSetLocalSection(dm, sectionDist);CHKERRQ(ierr);
  /* If a sequential section is provided but no dof is affected,
  sfNatural cannot be computed and is set to NULL */
  ierr = DMCreateGlobalVector(dm, &tmpVec);CHKERRQ(ierr);
  ierr = VecGetSize(tmpVec, &globalSize);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dm, &tmpVec);CHKERRQ(ierr);
  if (!globalSize) {
    *sfNatural = NULL;
    if (destroyFlag) {ierr = PetscSectionDestroy(&section);CHKERRQ(ierr);}
    PetscFunctionReturn(0);
  }
  /* Get a pruned version of migration SF */
  ierr = DMGetGlobalSection(dm, &gSection);CHKERRQ(ierr);
  ierr = PetscSectionGetChart(gSection, &pStart, &pEnd);CHKERRQ(ierr);
  for (p = pStart, ssize = 0; p < pEnd; ++p) {
    PetscInt dof, off;

    ierr = PetscSectionGetDof(gSection, p, &dof);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(gSection, p, &off);CHKERRQ(ierr);
    if ((dof > 0) && (off >= 0)) ++ssize;
  }
  ierr = PetscMalloc1(ssize, &spoints);CHKERRQ(ierr);
  for (p = pStart, ssize = 0; p < pEnd; ++p) {
    PetscInt dof, off;

    ierr = PetscSectionGetDof(gSection, p, &dof);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(gSection, p, &off);CHKERRQ(ierr);
    if ((dof > 0) && (off >= 0)) spoints[ssize++] = p;
  }
  ierr = PetscSFCreateEmbeddedLeafSF(sfMigration, ssize, spoints, &sfEmbed);CHKERRQ(ierr);
  ierr = PetscFree(spoints);CHKERRQ(ierr);
  /* ierr = PetscPrintf(comm, "Embedded SF\n");CHKERRQ(ierr);
   ierr = PetscSFView(sfEmbed, 0);CHKERRQ(ierr); */
  /* Create the SF for seq to natural */
  ierr = DMGetGlobalVector(dm, &gv);CHKERRQ(ierr);
  ierr = VecGetLayout(gv,&map);CHKERRQ(ierr);
  /* Note that entries of gv are leaves in sfSeqToNatural, entries of the seq vec are roots */
  ierr = PetscSFCreate(comm, &sfSeqToNatural);CHKERRQ(ierr);
  ierr = PetscSFSetGraphWithPattern(sfSeqToNatural, map, PETSCSF_PATTERN_GATHER);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dm, &gv);CHKERRQ(ierr);
  /* ierr = PetscPrintf(comm, "Seq-to-Natural SF\n");CHKERRQ(ierr);
   ierr = PetscSFView(sfSeqToNatural, 0);CHKERRQ(ierr); */
  /* Create the SF associated with this section */
  ierr = DMGetPointSF(dm, &sf);CHKERRQ(ierr);
  ierr = PetscSectionCreateGlobalSection(sectionDist, sf, PETSC_FALSE, PETSC_TRUE, &gLocSection);CHKERRQ(ierr);
  ierr = PetscSFCreateSectionSF(sfEmbed, section, remoteOffsets, gLocSection, &sfField);CHKERRQ(ierr);
  ierr = PetscFree(remoteOffsets);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&sfEmbed);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&gLocSection);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&sectionDist);CHKERRQ(ierr);
  /* ierr = PetscPrintf(comm, "Field SF\n");CHKERRQ(ierr);
   ierr = PetscSFView(sfField, 0);CHKERRQ(ierr); */
  /* Invert the field SF so it's now from distributed to sequential */
  ierr = PetscSFCreateInverseSF(sfField, &sfFieldInv);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&sfField);CHKERRQ(ierr);
  /* ierr = PetscPrintf(comm, "Inverse Field SF\n");CHKERRQ(ierr);
   ierr = PetscSFView(sfFieldInv, 0);CHKERRQ(ierr); */
  /* Multiply the sfFieldInv with the */
  ierr = PetscSFComposeInverse(sfFieldInv, sfSeqToNatural, sfNatural);CHKERRQ(ierr);
  ierr = PetscObjectViewFromOptions((PetscObject) *sfNatural, NULL, "-globaltonatural_sf_view");CHKERRQ(ierr);
  /* Clean up */
  ierr = PetscSFDestroy(&sfFieldInv);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&sfSeqToNatural);CHKERRQ(ierr);
  if (destroyFlag) {ierr = PetscSectionDestroy(&section);CHKERRQ(ierr);}
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

.seealso: DMPlexDistribute(), DMPlexDistributeField(), DMPlexNaturalToGlobalBegin(), DMPlexGlobalToNaturalEnd()
@*/
PetscErrorCode DMPlexGlobalToNaturalBegin(DM dm, Vec gv, Vec nv)
{
  const PetscScalar *inarray;
  PetscScalar       *outarray;
  PetscMPIInt        size;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(DMPLEX_GlobalToNaturalBegin,dm,0,0,0);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject) dm), &size);CHKERRMPI(ierr);
  if (dm->sfNatural) {
    ierr = VecGetArray(nv, &outarray);CHKERRQ(ierr);
    ierr = VecGetArrayRead(gv, &inarray);CHKERRQ(ierr);
    ierr = PetscSFBcastBegin(dm->sfNatural, MPIU_SCALAR, (PetscScalar *) inarray, outarray);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(gv, &inarray);CHKERRQ(ierr);
    ierr = VecRestoreArray(nv, &outarray);CHKERRQ(ierr);
  } else if (size == 1) {
    ierr = VecCopy(gv, nv);CHKERRQ(ierr);
  } else if (dm->useNatural) SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_PLIB, "DM global to natural SF not present.\nIf DMPlexDistribute() was called and a section was defined, report to petsc-maint@mcs.anl.gov.\n");
  else SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_WRONGSTATE, "DM global to natural SF was not created.\nYou must call DMSetUseNatural() before DMPlexDistribute().\n");
  ierr = PetscLogEventEnd(DMPLEX_GlobalToNaturalBegin,dm,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  DMPlexGlobalToNaturalEnd - Rearranges a global Vector in the natural order.

  Collective on dm

  Input Parameters:
+ dm - The distributed DMPlex
- gv - The global Vec

  Output Parameters:
. nv - The natural Vec

  Note: The user must call DMSetUseNatural(dm, PETSC_TRUE) before DMPlexDistribute().

  Level: intermediate

 .seealso: DMPlexDistribute(), DMPlexDistributeField(), DMPlexNaturalToGlobalBegin(), DMPlexGlobalToNaturalBegin()
 @*/
PetscErrorCode DMPlexGlobalToNaturalEnd(DM dm, Vec gv, Vec nv)
{
  const PetscScalar *inarray;
  PetscScalar       *outarray;
  PetscMPIInt        size;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(DMPLEX_GlobalToNaturalEnd,dm,0,0,0);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject) dm), &size);CHKERRMPI(ierr);
  if (dm->sfNatural) {
    ierr = VecGetArrayRead(gv, &inarray);CHKERRQ(ierr);
    ierr = VecGetArray(nv, &outarray);CHKERRQ(ierr);
    ierr = PetscSFBcastEnd(dm->sfNatural, MPIU_SCALAR, (PetscScalar *) inarray, outarray);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(gv, &inarray);CHKERRQ(ierr);
    ierr = VecRestoreArray(nv, &outarray);CHKERRQ(ierr);
  } else if (size == 1) {
  } else if (dm->useNatural) SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_PLIB, "DM global to natural SF not present.\nIf DMPlexDistribute() was called and a section was defined, report to petsc-maint@mcs.anl.gov.\n");
  else SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_WRONGSTATE, "DM global to natural SF was not created.\nYou must call DMSetUseNatural() before DMPlexDistribute().\n");
  ierr = PetscLogEventEnd(DMPLEX_GlobalToNaturalEnd,dm,0,0,0);CHKERRQ(ierr);
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

.seealso: DMPlexDistribute(), DMPlexDistributeField(), DMPlexNaturalToGlobalBegin(),DMPlexGlobalToNaturalEnd()
@*/
PetscErrorCode DMPlexNaturalToGlobalBegin(DM dm, Vec nv, Vec gv)
{
  const PetscScalar *inarray;
  PetscScalar       *outarray;
  PetscMPIInt        size;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(DMPLEX_NaturalToGlobalBegin,dm,0,0,0);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject) dm), &size);CHKERRMPI(ierr);
  if (dm->sfNatural) {
    /* We only have acces to the SF that goes from Global to Natural.
       Instead of inverting dm->sfNatural, we can call PetscSFReduceBegin/End with MPI_Op MPI_SUM.
       Here the SUM really does nothing since sfNatural is one to one, as long as gV is set to zero first. */
    ierr = VecZeroEntries(gv);CHKERRQ(ierr);
    ierr = VecGetArray(gv, &outarray);CHKERRQ(ierr);
    ierr = VecGetArrayRead(nv, &inarray);CHKERRQ(ierr);
    ierr = PetscSFReduceBegin(dm->sfNatural, MPIU_SCALAR, (PetscScalar *) inarray, outarray, MPI_SUM);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(nv, &inarray);CHKERRQ(ierr);
    ierr = VecRestoreArray(gv, &outarray);CHKERRQ(ierr);
  } else if (size == 1) {
    ierr = VecCopy(nv, gv);CHKERRQ(ierr);
  } else if (dm->useNatural) SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_PLIB, "DM global to natural SF not present.\nIf DMPlexDistribute() was called and a section was defined, report to petsc-maint@mcs.anl.gov.\n");
  else SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_WRONGSTATE, "DM global to natural SF was not created.\nYou must call DMSetUseNatural() before DMPlexDistribute().\n");
  ierr = PetscLogEventEnd(DMPLEX_NaturalToGlobalBegin,dm,0,0,0);CHKERRQ(ierr);
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

.seealso: DMPlexDistribute(), DMPlexDistributeField(), DMPlexNaturalToGlobalBegin(), DMPlexGlobalToNaturalBegin()
 @*/
PetscErrorCode DMPlexNaturalToGlobalEnd(DM dm, Vec nv, Vec gv)
{
  const PetscScalar *inarray;
  PetscScalar       *outarray;
  PetscMPIInt        size;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(DMPLEX_NaturalToGlobalEnd,dm,0,0,0);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject) dm), &size);CHKERRMPI(ierr);
  if (dm->sfNatural) {
    ierr = VecGetArrayRead(nv, &inarray);CHKERRQ(ierr);
    ierr = VecGetArray(gv, &outarray);CHKERRQ(ierr);
    ierr = PetscSFReduceEnd(dm->sfNatural, MPIU_SCALAR, (PetscScalar *) inarray, outarray, MPI_SUM);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(nv, &inarray);CHKERRQ(ierr);
    ierr = VecRestoreArray(gv, &outarray);CHKERRQ(ierr);
  } else if (size == 1) {
  } else if (dm->useNatural) SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_PLIB, "DM global to natural SF not present.\nIf DMPlexDistribute() was called and a section was defined, report to petsc-maint@mcs.anl.gov.\n");
  else SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_WRONGSTATE, "DM global to natural SF was not created.\nYou must call DMSetUseNatural() before DMPlexDistribute().\n");
  ierr = PetscLogEventEnd(DMPLEX_NaturalToGlobalEnd,dm,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
