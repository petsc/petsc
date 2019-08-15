#include <petsc/private/dmpleximpl.h>   /*I      "petscdmplex.h"   I*/

/*@
  DMPlexCreateClosureIndex - Calculate an index for the given PetscSection for the closure operation on the DM

  Not collective

  Input Parameters:
+ dm - The DM
- section - The section describing the layout in v, or NULL to use the default section

  Note:
  This should greatly improve the performance of the closure operations, at the cost of additional memory.

  Level: intermediate

.seealso DMPlexVecGetClosure(), DMPlexVecRestoreClosure(), DMPlexVecSetClosure(), DMPlexMatSetClosure()
@*/
PetscErrorCode DMPlexCreateClosureIndex(DM dm, PetscSection section)
{
  PetscSection   closureSection;
  IS             closureIS;
  PetscInt      *clPoints;
  PetscInt       pStart, pEnd, sStart, sEnd, point, clSize;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (!section) {ierr = DMGetLocalSection(dm, &section);CHKERRQ(ierr);}
  PetscValidHeaderSpecific(section, PETSC_SECTION_CLASSID, 2);
  ierr = PetscSectionGetChart(section, &sStart, &sEnd);CHKERRQ(ierr);
  ierr = DMPlexGetChart(dm, &pStart, &pEnd);CHKERRQ(ierr);
  ierr = PetscSectionCreate(PetscObjectComm((PetscObject) section), &closureSection);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(closureSection, pStart, pEnd);CHKERRQ(ierr);
  for (point = pStart; point < pEnd; ++point) {
    PetscInt *points = NULL, numPoints, p, dof, cldof = 0;

    ierr = DMPlexGetTransitiveClosure(dm, point, PETSC_TRUE, &numPoints, &points);CHKERRQ(ierr);
    for (p = 0; p < numPoints*2; p += 2) {
      if ((points[p] >= sStart) && (points[p] < sEnd)) {
        ierr = PetscSectionGetDof(section, points[p], &dof);CHKERRQ(ierr);
        if (dof) cldof += 2;
      }
    }
    ierr = DMPlexRestoreTransitiveClosure(dm, point, PETSC_TRUE, &numPoints, &points);CHKERRQ(ierr);
    ierr = PetscSectionSetDof(closureSection, point, cldof);CHKERRQ(ierr);
  }
  ierr = PetscSectionSetUp(closureSection);CHKERRQ(ierr);
  ierr = PetscSectionGetStorageSize(closureSection, &clSize);CHKERRQ(ierr);
  ierr = PetscMalloc1(clSize, &clPoints);CHKERRQ(ierr);
  for (point = pStart; point < pEnd; ++point) {
    PetscInt *points = NULL, numPoints, p, q, dof, cldof, cloff;

    ierr = PetscSectionGetDof(closureSection, point, &cldof);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(closureSection, point, &cloff);CHKERRQ(ierr);
    ierr = DMPlexGetTransitiveClosure(dm, point, PETSC_TRUE, &numPoints, &points);CHKERRQ(ierr);
    for (p = 0, q = 0; p < numPoints*2; p += 2) {
      if ((points[p] >= sStart) && (points[p] < sEnd)) {
        ierr = PetscSectionGetDof(section, points[p], &dof);CHKERRQ(ierr);
        if (dof) {
          clPoints[cloff+q*2]   = points[p];
          clPoints[cloff+q*2+1] = points[p+1];
          ++q;
        }
      }
    }
    ierr = DMPlexRestoreTransitiveClosure(dm, point, PETSC_TRUE, &numPoints, &points);CHKERRQ(ierr);
    if (q*2 != cldof) SETERRQ2(PetscObjectComm((PetscObject) dm), PETSC_ERR_PLIB, "Invalid size for closure %d should be %d", q*2, cldof);
  }
  ierr = ISCreateGeneral(PETSC_COMM_SELF, clSize, clPoints, PETSC_OWN_POINTER, &closureIS);CHKERRQ(ierr);
  ierr = PetscSectionSetClosureIndex(section, (PetscObject) dm, closureSection, closureIS);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
