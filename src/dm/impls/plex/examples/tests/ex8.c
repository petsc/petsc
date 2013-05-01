static char help[] = "Tests for cell geometry\n\n";

/* TODO
*/

#include <petscdmplex.h>

#undef __FUNCT__
#define __FUNCT__ "ChangeCoordinates"
PetscErrorCode ChangeCoordinates(DM dm, PetscInt spaceDim, PetscScalar vertexCoords[])
{
  PetscSection   coordSection;
  Vec            coordinates;
  PetscScalar   *coords;
  PetscInt       vStart, vEnd, v, d, coordSize;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
  ierr = DMPlexGetCoordinateSection(dm, &coordSection);CHKERRQ(ierr);
  ierr = PetscSectionSetNumFields(coordSection, 1);CHKERRQ(ierr);
  ierr = PetscSectionSetFieldComponents(coordSection, 0, spaceDim);CHKERRQ(ierr);
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
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CheckFEMGeometry"
PetscErrorCode CheckFEMGeometry(DM dm, PetscInt cell, PetscInt spaceDim, PetscReal v0Ex[], PetscReal JEx[], PetscReal invJEx[], PetscReal detJEx)
{
  PetscReal      v0[3], J[9], invJ[9], detJ;
  PetscInt       d, i, j;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMPlexComputeCellGeometry(dm, cell, v0, J, invJ, &detJ);CHKERRQ(ierr);
  for (d = 0; d < spaceDim; ++d) {
    if (v0[d] != v0Ex[d]) {
      switch (spaceDim) {
      case 2: SETERRQ4(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Invalid v0 (%g, %g) != (%g, %g)", v0[0], v0[1], v0Ex[0], v0Ex[1]);break;
      case 3: SETERRQ6(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Invalid v0 (%g, %g, %g) != (%g, %g, %g)", v0[0], v0[1], v0[2], v0Ex[0], v0Ex[1], v0Ex[2]);break;
      default: SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Invalid space dimension %d", spaceDim);
      }
    }
  }
  for (i = 0; i < spaceDim; ++i) {
    for (j = 0; j < spaceDim; ++j) {
      if (fabs(J[i*spaceDim+j]    - JEx[i*spaceDim+j])    > 1.0e-9) SETERRQ4(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Invalid J[%d,%d]: %g != %g", i, j, J[i*spaceDim+j], JEx[i*spaceDim+j]);
      if (fabs(invJ[i*spaceDim+j] - invJEx[i*spaceDim+j]) > 1.0e-9) SETERRQ4(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Invalid invJ[%d,%d]: %g != %g", i, j, invJ[i*spaceDim+j], invJEx[i*spaceDim+j]);
    }
  }
  if (fabs(detJ - detJEx) > 1.0e-9) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Invalid |J| = %g != %g", detJ, detJEx);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CheckFVMGeometry"
PetscErrorCode CheckFVMGeometry(DM dm, PetscInt cell, PetscInt spaceDim, PetscReal centroidEx[], PetscReal normalEx[], PetscReal volEx)
{
  PetscReal      centroid[3], normal[3], vol;
  PetscInt       d;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMPlexComputeCellGeometryFVM(dm, cell, &vol, centroid, normal);CHKERRQ(ierr);
  for (d = 0; d < spaceDim; ++d) {
    if (fabs(centroid[d] - centroidEx[d]) > 1.0e-9) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Invalid centroid[%d]: %g != %g", d, centroid[d], centroidEx[d]);
    if (fabs(normal[d] - normalEx[d]) > 1.0e-9) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Invalid normal[%d]: %g != %g", d, normal[d], normalEx[d]);
  }
  if (fabs(volEx - vol) > 1.0e-9) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Invalid volume = %g != %g", vol, volEx);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TestTriangle"
PetscErrorCode TestTriangle(MPI_Comm comm, PetscBool interpolate)
{
  DM             dm;
  PetscInt       dim;
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
    if (interpolate) {
      DM idm;

      ierr = DMPlexInterpolate(dm, &idm);CHKERRQ(ierr);
      ierr = DMPlexCopyCoordinates(dm, idm);CHKERRQ(ierr);
      ierr = DMDestroy(&dm);CHKERRQ(ierr);
      dm   = idm;
    }
    ierr = DMSetFromOptions(dm);CHKERRQ(ierr);
  }
  /* Check reference geometry: determinant is scaled by reference volume (2.0) */
  {
    PetscReal v0Ex[2]       = {-1.0, -1.0};
    PetscReal JEx[4]        = {1.0, 0.0, 0.0, 1.0};
    PetscReal invJEx[4]     = {1.0, 0.0, 0.0, 1.0};
    PetscReal detJEx        = 1.0;
    PetscReal centroidEx[2] = {-0.333333333333, -0.333333333333};
    PetscReal normalEx[2]   = {0.0, 0.0};
    PetscReal volEx         = 2.0;

    ierr = CheckFEMGeometry(dm, 0, dim, v0Ex, JEx, invJEx, detJEx);CHKERRQ(ierr);
    if (interpolate) {ierr = CheckFVMGeometry(dm, 0, dim, centroidEx, normalEx, volEx);CHKERRQ(ierr);}
  }
  /* Check random triangles: rotate and translate */

  /* Move to 3D: Check reference geometry: determinant is scaled by reference volume (2.0) */
  dim = 3;
  {
    PetscScalar vertexCoords[9] = {-1.0, -1.0, 0.0, 1.0, -1.0, 0.0, -1.0, 1.0, 0.0};
    PetscReal v0Ex[3]       = {-1.0, -1.0, 0.0};
    PetscReal JEx[9]        = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    PetscReal invJEx[9]     = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    PetscReal detJEx        = 1.0;
    PetscReal centroidEx[3] = {-0.333333333333, -0.333333333333, 0.0};
    PetscReal normalEx[3]   = {0.0, 0.0, 1.0};
    PetscReal volEx         = 2.0;

    ierr = ChangeCoordinates(dm, dim, vertexCoords);CHKERRQ(ierr);
    ierr = CheckFEMGeometry(dm, 0, dim, v0Ex, JEx, invJEx, detJEx);CHKERRQ(ierr);
    if (interpolate) {ierr = CheckFVMGeometry(dm, 0, dim, centroidEx, normalEx, volEx);CHKERRQ(ierr);}
  }
  /* Rotated reference element */
  {
    PetscScalar vertexCoords[9] = {0.0, -1.0, -1.0, 0.0, 1.0, -1.0, 0.0, -1.0, 1.0};
    PetscReal v0Ex[3]   = {0.0, -1.0, -1.0};
    PetscReal JEx[9]    = {0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0};
    PetscReal invJEx[9] = {0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0};
    PetscReal detJEx    = 1.0;
    PetscReal centroidEx[3] = {0.0, -0.333333333333, -0.333333333333};
    PetscReal normalEx[3]   = {1.0, 0.0, 0.0};
    PetscReal volEx         = 2.0;

    ierr = ChangeCoordinates(dm, dim, vertexCoords);CHKERRQ(ierr);
    ierr = CheckFEMGeometry(dm, 0, dim, v0Ex, JEx, invJEx, detJEx);CHKERRQ(ierr);
    if (interpolate) {ierr = CheckFVMGeometry(dm, 0, dim, centroidEx, normalEx, volEx);CHKERRQ(ierr);}
  }
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
  ierr = TestTriangle(PETSC_COMM_SELF, PETSC_FALSE);CHKERRQ(ierr);
  ierr = TestTriangle(PETSC_COMM_SELF, PETSC_TRUE);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}
