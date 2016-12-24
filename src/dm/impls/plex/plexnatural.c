#include <petsc/private/dmpleximpl.h>   /*I      "petscdmplex.h"   I*/

/*@
  DMPlexCreateGlobalToNaturalSF - Creates the SF for mapping Global Vec to the Natural Vec

  Input Parameters:
+ dm          - The DM
. section     - The PetscSection before the mesh was distributed
- sfMigration - The PetscSF used to distribute the mesh

  Output Parameters:
. sfNatural - PetscSF for mapping the Vec in PETSc ordering to the canonical ordering

  Level: intermediate

.seealso: DMPlexDistribute(), DMPlexDistributeField()
 @*/
PetscErrorCode DMPlexCreateGlobalToNaturalSF(DM dm, PetscSection section, PetscSF sfMigration, PetscSF *sfNatural)
{
  MPI_Comm       comm;
  Vec            gv;
  PetscSF        sf, sfEmbed, sfSeqToNatural, sfField, sfFieldInv;
  PetscSection   gSection, sectionDist, gLocSection;
  PetscInt      *spoints, *remoteOffsets;
  PetscInt       ssize, pStart, pEnd, p;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject) dm, &comm);CHKERRQ(ierr);
  /* ierr = PetscPrintf(comm, "Point migration SF\n");CHKERRQ(ierr);
   ierr = PetscSFView(sfMigration, 0);CHKERRQ(ierr); */
  /* Create a new section from distributing the original section */
  ierr = PetscSectionCreate(comm, &sectionDist);CHKERRQ(ierr);
  ierr = PetscSFDistributeSection(sfMigration, section, &remoteOffsets, sectionDist);CHKERRQ(ierr);
  /* ierr = PetscPrintf(comm, "Distributed Section\n");CHKERRQ(ierr);
   ierr = PetscSectionView(sectionDist, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr); */
  ierr = DMSetDefaultSection(dm, sectionDist);CHKERRQ(ierr);
  /* Get a pruned version of migration SF */
  ierr = DMGetDefaultGlobalSection(dm, &gSection);CHKERRQ(ierr);
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
  ierr = PetscSFCreateFromZero(comm, gv, &sfSeqToNatural);CHKERRQ(ierr);
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
  ierr = PetscSFCompose(sfFieldInv, sfSeqToNatural, sfNatural);CHKERRQ(ierr);
  ierr = PetscObjectViewFromOptions((PetscObject) *sfNatural, NULL, "-globaltonatural_sf_view");CHKERRQ(ierr);
  /* Clean up */
  ierr = PetscSFDestroy(&sfFieldInv);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&sfSeqToNatural);CHKERRQ(ierr);
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

  Note: The user must call DMPlexSetUseNaturalSF(dm, PETSC_TRUE) before DMPlexDistribute().

  Level: intermediate

.seealso: DMPlexDistribute(), DMPlexDistributeField(), DMPlexNaturalToGlobalBegin(), DMPlexGlobalToNaturalEnd()
@*/
PetscErrorCode DMPlexGlobalToNaturalBegin(DM dm, Vec gv, Vec nv)
{
  const PetscScalar *inarray;
  PetscScalar       *outarray;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(DMPLEX_GlobalToNaturalBegin,dm,0,0,0);CHKERRQ(ierr);
  if (dm->sfNatural) {
    ierr = VecGetArray(nv, &outarray);CHKERRQ(ierr);
    ierr = VecGetArrayRead(gv, &inarray);CHKERRQ(ierr);
    ierr = PetscSFBcastBegin(dm->sfNatural, MPIU_SCALAR, (PetscScalar *) inarray, outarray);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(gv, &inarray);CHKERRQ(ierr);
    ierr = VecRestoreArray(nv, &outarray);CHKERRQ(ierr);
  } else SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_WRONGSTATE, "DM global to natural SF was not created.\nYou must call DMSetUseNatural() before DMPlexDistribute().\n");
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

  Note: The user must call DMPlexSetUseNaturalSF(dm, PETSC_TRUE) before DMPlexDistribute().

  Level: intermediate

 .seealso: DMPlexDistribute(), DMPlexDistributeField(), DMPlexNaturalToGlobalBegin(), DMPlexGlobalToNaturalBegin()
 @*/
PetscErrorCode DMPlexGlobalToNaturalEnd(DM dm, Vec gv, Vec nv)
{
  const PetscScalar *inarray;
  PetscScalar       *outarray;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(DMPLEX_GlobalToNaturalEnd,dm,0,0,0);CHKERRQ(ierr);
  if (dm->sfNatural) {
    ierr = VecGetArrayRead(gv, &inarray);CHKERRQ(ierr);
    ierr = VecGetArray(nv, &outarray);CHKERRQ(ierr);CHKERRQ(ierr);
    ierr = PetscSFBcastEnd(dm->sfNatural, MPIU_SCALAR, (PetscScalar *) inarray, outarray);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(gv, &inarray);CHKERRQ(ierr);
    ierr = VecRestoreArray(nv, &outarray);CHKERRQ(ierr);
  }
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

  Note: The user must call DMPlexSetUseNaturalSF(dm, PETSC_TRUE) before DMPlexDistribute().

  Level: intermediate

.seealso: DMPlexDistribute(), DMPlexDistributeField(), DMPlexNaturalToGlobalBegin(),DMPlexGlobalToNaturalEnd()
@*/
PetscErrorCode DMPlexNaturalToGlobalBegin(DM dm, Vec nv, Vec gv)
{
  const PetscScalar *inarray;
  PetscScalar       *outarray;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(DMPLEX_NaturalToGlobalBegin,dm,0,0,0);CHKERRQ(ierr);
  if (dm->sfNatural) {
    ierr = VecGetArray(gv, &outarray);CHKERRQ(ierr);
    ierr = VecGetArrayRead(nv, &inarray);CHKERRQ(ierr);
    ierr = PetscSFReduceBegin(dm->sfNatural, MPIU_SCALAR, (PetscScalar *) inarray, outarray, MPI_SUM);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(nv, &inarray);CHKERRQ(ierr);
    ierr = VecRestoreArray(gv, &outarray);CHKERRQ(ierr);
  } else SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_WRONGSTATE, "DM global to natural SF was not created.\nYou must call DMPlexSetUseNaturalSF() before DMPlexDistribute().\n");
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

  Note: The user must call DMPlexSetUseNaturalSF(dm, PETSC_TRUE) before DMPlexDistribute().

  Level: intermediate

.seealso: DMPlexDistribute(), DMPlexDistributeField(), DMPlexNaturalToGlobalBegin(), DMPlexGlobalToNaturalBegin()
 @*/
PetscErrorCode DMPlexNaturalToGlobalEnd(DM dm, Vec nv, Vec gv)
{
  const PetscScalar *inarray;
  PetscScalar       *outarray;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(DMPLEX_NaturalToGlobalEnd,dm,0,0,0);CHKERRQ(ierr);
  if (dm->sfNatural) {
    ierr = VecGetArrayRead(nv, &inarray);CHKERRQ(ierr);
    ierr = VecGetArray(gv, &outarray);CHKERRQ(ierr);CHKERRQ(ierr);
    ierr = PetscSFReduceEnd(dm->sfNatural, MPIU_SCALAR, (PetscScalar *) inarray, outarray, MPI_SUM);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(nv, &inarray);CHKERRQ(ierr);
    ierr = VecRestoreArray(gv, &outarray);CHKERRQ(ierr);
  }
  ierr = PetscLogEventEnd(DMPLEX_NaturalToGlobalEnd,dm,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
