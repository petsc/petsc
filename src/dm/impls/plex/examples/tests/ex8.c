static char help[] = "Tests for cell geometry\n\n";

/* TODO
*/

#include <petscdmplex.h>

#undef __FUNCT__
#define __FUNCT__ "TestTriangle"
PetscErrorCode TestTriangle(MPI_Comm comm)
{
  DM             dm;
  PetscReal      v0[3], J[9], invJ[9], detJ;
  PetscInt       dim, i, j;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* Create reference triangle */
  dim  = 2;
  ierr = DMCreate(comm, &dm);CHKERRQ(ierr);
  ierr = DMSetType(dm, DMPLEX);CHKERRQ(ierr);
  ierr = DMPlexSetDimension(dm, dim);CHKERRQ(ierr);
  {
    PetscInt    numPoints[2]        = {3, 1};
    PetscInt    coneSize[4]         = {3, 0, 0, 0};
    PetscInt    cones[3]            = {1, 2, 3};
    PetscInt    coneOrientations[3] = {0, 0, 0};
    PetscScalar vertexCoords[6]     = {-1.0, -1.0, 1.0, -1.0, -1.0, 1.0};

    ierr = DMPlexCreateFromDAG(dm, 1, numPoints, coneSize, cones, coneOrientations, vertexCoords);CHKERRQ(ierr);
    ierr = DMSetFromOptions(dm);CHKERRQ(ierr);
  }
  /* Check reference geometry: determinant is scaled by reference volume (2.0) */
  ierr = DMPlexComputeCellGeometry(dm, 0, v0, J, invJ, &detJ);CHKERRQ(ierr);
  if ((v0[0] != -1.0) || (v0[1] != -1.0)) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Invalid v0 (%g, %g)", v0[0], v0[1]);
  for (i = 0; i < dim; ++i) {
    for (j = 0; j < dim; ++j) {
      if (fabs(J[i*dim+j] - (i == j ? 1.0 : 0.0)) > 1.0e-9) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Invalid J");
      if (fabs(invJ[i*dim+j] - (i == j ? 1.0 : 0.0)) > 1.0e-9) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Invalid invJ");
    }
  }
  if (fabs(detJ - 1.0) > 1.0e-9) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Invalid |J| = %g should be 1.0", detJ);
  /* Check random triangles: rotate and translate */

  /* Move to 3D */
  {
    PetscSection coordSection;
    Vec          coordinates;
    PetscScalar  vertexCoords[9] = {-1.0, -1.0, 0.0, 1.0, -1.0, 0.0, -1.0, 1.0, 0.0};
    PetscScalar *coords;
    PetscInt     vStart, vEnd, v, d, coordSize, spaceDim = 3;

    ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
    ierr = DMPlexGetCoordinateSection(dm, &coordSection);CHKERRQ(ierr);
    ierr = PetscSectionSetChart(coordSection, vStart, vEnd);CHKERRQ(ierr);
    for (v = vStart; v < vEnd; ++v) {
      ierr = PetscSectionSetDof(coordSection, v, spaceDim);CHKERRQ(ierr);
      ierr = PetscSectionSetFieldDof(coordSection, v, 0, spaceDim);CHKERRQ(ierr);
    }
    ierr = PetscSectionSetUp(coordSection);CHKERRQ(ierr);
    ierr = PetscSectionGetStorageSize(coordSection, &coordSize);CHKERRQ(ierr);
    ierr = VecCreate(PetscObjectComm((PetscObject) dm), &coordinates);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) coordinates, "coordinates");CHKERRQ(ierr);
    ierr = VecSetSizes(coordinates, coordSize, PETSC_DETERMINE);CHKERRQ(ierr);
    ierr = VecSetFromOptions(coordinates);CHKERRQ(ierr);
    ierr = VecGetArray(coordinates, &coords);CHKERRQ(ierr);
    for (v = vStart; v < vEnd; ++v) {
      PetscInt off;

      ierr = PetscSectionGetOffset(coordSection, v, &off);CHKERRQ(ierr);
      for (d = 0; d < spaceDim; ++d) {
        coords[off+d] = vertexCoords[(v-vStart)*spaceDim+d];
      }
    }
    ierr = VecRestoreArray(coordinates, &coords);CHKERRQ(ierr);
    ierr = DMSetCoordinatesLocal(dm, coordinates);CHKERRQ(ierr);
    ierr = VecDestroy(&coordinates);CHKERRQ(ierr);
    ierr = DMSetFromOptions(dm);CHKERRQ(ierr);
  }
  /* Check reference geometry: determinant is scaled by reference volume (2.0) */
  ierr = DMPlexComputeCellGeometry(dm, 0, v0, J, invJ, &detJ);CHKERRQ(ierr);
  /* if ((v0[0] != -1.0) || (v0[1] != -1.0)) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Invalid v0 (%g, %g)", v0[0], v0[1]); */
  for (i = 0; i < dim; ++i) {
    for (j = 0; j < dim; ++j) {
      if (fabs(J[i*dim+j] - (i == j ? 1.0 : 0.0)) > 1.0e-9) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Invalid J[%d,%d]", i, j);
      if (fabs(invJ[i*dim+j] - (i == j ? 1.0 : 0.0)) > 1.0e-9) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Invalid invJ[%d,%d]", i, j);
    }
  }
  if (fabs(detJ - 1.0) > 1.0e-9) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Invalid |J| = %g should be 1.0", detJ);
  /* Rotated reference element */
  {
    PetscSection coordSection;
    Vec          coordinates;
    PetscScalar  vertexCoords[9] = {0.0, -1.0, -1.0, 0.0, 1.0, -1.0, 0.0, -1.0, 1.0};
    PetscScalar *coords;
    PetscInt     vStart, vEnd, v, d, spaceDim = 3;

    ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
    ierr = DMPlexGetCoordinateSection(dm, &coordSection);CHKERRQ(ierr);
    ierr = DMGetCoordinatesLocal(dm, &coordinates);CHKERRQ(ierr);
    ierr = VecGetArray(coordinates, &coords);CHKERRQ(ierr);
    for (v = vStart; v < vEnd; ++v) {
      PetscInt off;

      ierr = PetscSectionGetOffset(coordSection, v, &off);CHKERRQ(ierr);
      for (d = 0; d < spaceDim; ++d) {
        coords[off+d] = vertexCoords[(v-vStart)*spaceDim+d];
      }
    }
    ierr = VecRestoreArray(coordinates, &coords);CHKERRQ(ierr);
    ierr = DMSetFromOptions(dm);CHKERRQ(ierr);
  }
  ierr = DMPlexComputeCellGeometry(dm, 0, v0, J, invJ, &detJ);CHKERRQ(ierr);
  for (i = 0; i < dim; ++i) {
    for (j = 0; j < dim; ++j) {
      if (fabs(J[i*dim+j] - (i == j ? 0.0 : i == 0 ? -1.0 : 1.0)) > 1.0e-9) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Invalid J");
      if (fabs(invJ[i*dim+j] - (i == j ? 0.0 : i == 0 ? 1.0 : -1.0)) > 1.0e-9) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Invalid invJ");
    }
  }
  if (fabs(detJ - 1.0) > 1.0e-9) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Invalid |J| = %g should be 1.0", detJ);

  /* Cleanup */
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char **argv)
{
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help);CHKERRQ(ierr);
  ierr = TestTriangle(PETSC_COMM_SELF);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}
