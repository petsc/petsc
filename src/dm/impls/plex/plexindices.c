#include <petsc/private/dmpleximpl.h>   /*I      "petscdmplex.h"   I*/

/*@
  DMPlexCreateClosureIndex - Calculate an index for the given PetscSection for the closure operation on the DM

  Not collective

  Input Parameters:
+ dm - The DM
- section - The section describing the layout in the local vector, or NULL to use the default section

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

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (!section) CHKERRQ(DMGetLocalSection(dm, &section));
  PetscValidHeaderSpecific(section, PETSC_SECTION_CLASSID, 2);
  CHKERRQ(PetscSectionGetChart(section, &sStart, &sEnd));
  CHKERRQ(DMPlexGetChart(dm, &pStart, &pEnd));
  CHKERRQ(PetscSectionCreate(PetscObjectComm((PetscObject) section), &closureSection));
  CHKERRQ(PetscSectionSetChart(closureSection, pStart, pEnd));
  for (point = pStart; point < pEnd; ++point) {
    PetscInt *points = NULL, numPoints, p, dof, cldof = 0;

    CHKERRQ(DMPlexGetTransitiveClosure(dm, point, PETSC_TRUE, &numPoints, &points));
    for (p = 0; p < numPoints*2; p += 2) {
      if ((points[p] >= sStart) && (points[p] < sEnd)) {
        CHKERRQ(PetscSectionGetDof(section, points[p], &dof));
        if (dof) cldof += 2;
      }
    }
    CHKERRQ(DMPlexRestoreTransitiveClosure(dm, point, PETSC_TRUE, &numPoints, &points));
    CHKERRQ(PetscSectionSetDof(closureSection, point, cldof));
  }
  CHKERRQ(PetscSectionSetUp(closureSection));
  CHKERRQ(PetscSectionGetStorageSize(closureSection, &clSize));
  CHKERRQ(PetscMalloc1(clSize, &clPoints));
  for (point = pStart; point < pEnd; ++point) {
    PetscInt *points = NULL, numPoints, p, q, dof, cldof, cloff;

    CHKERRQ(PetscSectionGetDof(closureSection, point, &cldof));
    CHKERRQ(PetscSectionGetOffset(closureSection, point, &cloff));
    CHKERRQ(DMPlexGetTransitiveClosure(dm, point, PETSC_TRUE, &numPoints, &points));
    for (p = 0, q = 0; p < numPoints*2; p += 2) {
      if ((points[p] >= sStart) && (points[p] < sEnd)) {
        CHKERRQ(PetscSectionGetDof(section, points[p], &dof));
        if (dof) {
          clPoints[cloff+q*2]   = points[p];
          clPoints[cloff+q*2+1] = points[p+1];
          ++q;
        }
      }
    }
    CHKERRQ(DMPlexRestoreTransitiveClosure(dm, point, PETSC_TRUE, &numPoints, &points));
    PetscCheckFalse(q*2 != cldof,PetscObjectComm((PetscObject) dm), PETSC_ERR_PLIB, "Invalid size for closure %D should be %D", q*2, cldof);
  }
  CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF, clSize, clPoints, PETSC_OWN_POINTER, &closureIS));
  CHKERRQ(PetscSectionSetClosureIndex(section, (PetscObject) dm, closureSection, closureIS));
  CHKERRQ(PetscSectionDestroy(&closureSection));
  CHKERRQ(ISDestroy(&closureIS));
  PetscFunctionReturn(0);
}
