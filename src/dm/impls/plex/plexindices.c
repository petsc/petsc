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
  if (!section) PetscCall(DMGetLocalSection(dm, &section));
  PetscValidHeaderSpecific(section, PETSC_SECTION_CLASSID, 2);
  PetscCall(PetscSectionGetChart(section, &sStart, &sEnd));
  PetscCall(DMPlexGetChart(dm, &pStart, &pEnd));
  PetscCall(PetscSectionCreate(PetscObjectComm((PetscObject) section), &closureSection));
  PetscCall(PetscSectionSetChart(closureSection, pStart, pEnd));
  for (point = pStart; point < pEnd; ++point) {
    PetscInt *points = NULL, numPoints, p, dof, cldof = 0;

    PetscCall(DMPlexGetTransitiveClosure(dm, point, PETSC_TRUE, &numPoints, &points));
    for (p = 0; p < numPoints*2; p += 2) {
      if ((points[p] >= sStart) && (points[p] < sEnd)) {
        PetscCall(PetscSectionGetDof(section, points[p], &dof));
        if (dof) cldof += 2;
      }
    }
    PetscCall(DMPlexRestoreTransitiveClosure(dm, point, PETSC_TRUE, &numPoints, &points));
    PetscCall(PetscSectionSetDof(closureSection, point, cldof));
  }
  PetscCall(PetscSectionSetUp(closureSection));
  PetscCall(PetscSectionGetStorageSize(closureSection, &clSize));
  PetscCall(PetscMalloc1(clSize, &clPoints));
  for (point = pStart; point < pEnd; ++point) {
    PetscInt *points = NULL, numPoints, p, q, dof, cldof, cloff;

    PetscCall(PetscSectionGetDof(closureSection, point, &cldof));
    PetscCall(PetscSectionGetOffset(closureSection, point, &cloff));
    PetscCall(DMPlexGetTransitiveClosure(dm, point, PETSC_TRUE, &numPoints, &points));
    for (p = 0, q = 0; p < numPoints*2; p += 2) {
      if ((points[p] >= sStart) && (points[p] < sEnd)) {
        PetscCall(PetscSectionGetDof(section, points[p], &dof));
        if (dof) {
          clPoints[cloff+q*2]   = points[p];
          clPoints[cloff+q*2+1] = points[p+1];
          ++q;
        }
      }
    }
    PetscCall(DMPlexRestoreTransitiveClosure(dm, point, PETSC_TRUE, &numPoints, &points));
    PetscCheck(q*2 == cldof,PetscObjectComm((PetscObject) dm), PETSC_ERR_PLIB, "Invalid size for closure %D should be %D", q*2, cldof);
  }
  PetscCall(ISCreateGeneral(PETSC_COMM_SELF, clSize, clPoints, PETSC_OWN_POINTER, &closureIS));
  PetscCall(PetscSectionSetClosureIndex(section, (PetscObject) dm, closureSection, closureIS));
  PetscCall(PetscSectionDestroy(&closureSection));
  PetscCall(ISDestroy(&closureIS));
  PetscFunctionReturn(0);
}
