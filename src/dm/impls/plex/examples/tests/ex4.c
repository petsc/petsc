static char help[] = "Tests for uniform refinement of hybrid meshes\n\n";

#include <petscdmplex.h>

typedef struct {
  DM        dm;
  PetscInt  debug;             /* The debugging level */
  PetscInt  dim;               /* The topological mesh dimension */
  PetscBool refinementUniform; /* Uniformly refine the mesh */
  PetscBool cellSimplex;       /* Use simplices or hexes */
} AppCtx;

#undef __FUNCT__
#define __FUNCT__ "ProcessOptions"
PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  options->debug             = 0;
  options->dim               = 2;
  options->refinementUniform = PETSC_FALSE;
  options->cellSimplex       = PETSC_TRUE;

  ierr = PetscOptionsBegin(comm, "", "Meshing Problem Options", "DMPLEX");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-debug", "The debugging level", "ex4.c", options->debug, &options->debug, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dim", "The topological mesh dimension", "ex4.c", options->dim, &options->dim, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-refinement_uniform", "Uniformly refine the mesh", "ex4.c", options->refinementUniform, &options->refinementUniform, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-cell_simplex", "Use simplices if true, otherwise hexes", "ex4.c", options->cellSimplex, &options->cellSimplex, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();
  PetscFunctionReturn(0);
};

#undef __FUNCT__
#define __FUNCT__ "CreateSimplexHybrid_2D"
/* Two triangles separated by a zero-volume cell with 6 vertices
        5--16--8
      / |      | \
    11  |      |  12
    /   |      |   \
   3  0 10  2 14 1  6
    \   |      |   /
     9  |      |  13
      \ |      | /
        4--15--7
*/
PetscErrorCode CreateSimplexHybrid_2D(MPI_Comm comm, DM dm)
{
  Vec            coordinates;
  PetscSection   coordSection;
  PetscScalar    *coords;
  PetscInt       numVertices = 0, numEdges = 0, numCells = 0, cMax = PETSC_DETERMINE, fMax = PETSC_DETERMINE;
  PetscInt       firstVertex, firstEdge, coordSize;
  PetscInt       v, e;
  PetscMPIInt    rank;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  if (!rank) {
    numVertices = 3 + 3;
    numEdges    = 6 + 2;
    numCells    = 3;
    cMax        = 2;
    fMax        = 15;
  }
  firstVertex = numCells;
  firstEdge   = numCells + numVertices;
  ierr        = DMPlexSetChart(dm, 0, numCells+numEdges+numVertices);CHKERRQ(ierr);
  if (numCells) {
    ierr = DMPlexSetConeSize(dm, 0, 3);CHKERRQ(ierr);
    ierr = DMPlexSetConeSize(dm, 1, 3);CHKERRQ(ierr);
    ierr = DMPlexSetConeSize(dm, 2, 4);CHKERRQ(ierr);
  }
  for (e = firstEdge; e < firstEdge+numEdges; ++e) {
    ierr = DMPlexSetConeSize(dm, e, 2);CHKERRQ(ierr);
  }
  ierr = DMSetUp(dm);CHKERRQ(ierr); /* Allocate space for cones */
  ierr = DMPlexSetHybridBounds(dm, cMax, PETSC_DETERMINE, fMax, PETSC_DETERMINE);CHKERRQ(ierr); /* Indicate a hybrid mesh */
  /* Build cells */
  if (numCells > 0) {
    const PetscInt cone[3] = {9, 10, 11};
    const PetscInt ornt[3] = {0,  0,  0};
    const PetscInt cell    = 0;

    ierr = DMPlexSetCone(dm, cell, cone);CHKERRQ(ierr);
    ierr = DMPlexSetConeOrientation(dm, cell, ornt);CHKERRQ(ierr);
  }
  if (numCells > 1) {
    const PetscInt cone[3] = {12, 14, 13};
    const PetscInt ornt[3] = { 0, -2,  0};
    const PetscInt cell    = 1;

    ierr = DMPlexSetCone(dm, cell, cone);CHKERRQ(ierr);
    ierr = DMPlexSetConeOrientation(dm, cell, ornt);CHKERRQ(ierr);
  }
  if (numCells > 2) {
    const PetscInt cone[4] = {10, 14, 15, 16};
    const PetscInt ornt[4] = { 0,  0,  0,  0};
    const PetscInt cell    = 2;

    ierr = DMPlexSetCone(dm, cell, cone);CHKERRQ(ierr);
    ierr = DMPlexSetConeOrientation(dm, cell, ornt);CHKERRQ(ierr);
  }
  /* Build edges*/
  if (numEdges > 0) {
    const PetscInt cone[2] = {3, 4};
    const PetscInt edge    = 9;

    ierr = DMPlexSetCone(dm, edge, cone);CHKERRQ(ierr);
    ierr = DMPlexSetLabelValue(dm, "marker", edge, 1);CHKERRQ(ierr);
  }
  if (numEdges > 1) {
    const PetscInt cone[2] = {4, 5};
    const PetscInt edge    = 10;

    ierr = DMPlexSetCone(dm, edge, cone);CHKERRQ(ierr);
  }
  if (numEdges > 2) {
    const PetscInt cone[2] = {5, 3};
    const PetscInt edge    = 11;

    ierr = DMPlexSetCone(dm, edge, cone);CHKERRQ(ierr);
    ierr = DMPlexSetLabelValue(dm, "marker", edge, 1);CHKERRQ(ierr);
  }
  if (numEdges > 3) {
    const PetscInt cone[2] = {6, 8};
    const PetscInt edge    = 12;

    ierr = DMPlexSetCone(dm, edge, cone);CHKERRQ(ierr);
    ierr = DMPlexSetLabelValue(dm, "marker", edge, 1);CHKERRQ(ierr);
  }
  if (numEdges > 4) {
    const PetscInt cone[2] = {7, 6};
    const PetscInt edge    = 13;

    ierr = DMPlexSetCone(dm, edge, cone);CHKERRQ(ierr);
    ierr = DMPlexSetLabelValue(dm, "marker", edge, 1);CHKERRQ(ierr);
  }
  if (numEdges > 5) {
    const PetscInt cone[2] = {7, 8};
    const PetscInt edge    = 14;

    ierr = DMPlexSetCone(dm, edge, cone);CHKERRQ(ierr);
  }
  if (numEdges > 6) {
    const PetscInt cone[2] = {4, 7};
    const PetscInt edge    = 15;

    ierr = DMPlexSetCone(dm, edge, cone);CHKERRQ(ierr);
  }
  if (numEdges > 7) {
    const PetscInt cone[2] = {5, 8};
    const PetscInt edge    = 16;

    ierr = DMPlexSetCone(dm, edge, cone);CHKERRQ(ierr);
  }
  ierr = DMPlexSymmetrize(dm);CHKERRQ(ierr);
  ierr = DMPlexStratify(dm);CHKERRQ(ierr);
  /* Build coordinates */
  ierr = DMPlexGetCoordinateSection(dm, &coordSection);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(coordSection, firstVertex, firstVertex+numVertices);CHKERRQ(ierr);
  for (v = firstVertex; v < firstVertex+numVertices; ++v) {
    ierr = PetscSectionSetDof(coordSection, v, 2);CHKERRQ(ierr);
  }
  ierr = PetscSectionSetUp(coordSection);CHKERRQ(ierr);
  ierr = PetscSectionGetStorageSize(coordSection, &coordSize);CHKERRQ(ierr);
  ierr = VecCreate(PetscObjectComm((PetscObject)dm), &coordinates);CHKERRQ(ierr);
  ierr = VecSetSizes(coordinates, coordSize, PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = VecSetFromOptions(coordinates);CHKERRQ(ierr);
  ierr = VecGetArray(coordinates, &coords);CHKERRQ(ierr);
  if (numVertices) {
    coords[0]  = -0.5; coords[1]  = 0.5;
    coords[2]  = -0.2; coords[3]  = 0.0;
    coords[4]  = -0.2; coords[5]  = 1.0;
    coords[6]  =  0.5; coords[7]  = 0.5;
    coords[8]  =  0.2; coords[9]  = 0.0;
    coords[10] =  0.2; coords[11] = 1.0;
  }
  ierr = VecRestoreArray(coordinates, &coords);CHKERRQ(ierr);
  ierr = DMSetCoordinatesLocal(dm, coordinates);CHKERRQ(ierr);
  ierr = VecDestroy(&coordinates);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CreateMesh"
PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscInt       dim               = user->dim;
  PetscBool      refinementUniform = user->refinementUniform;
  PetscBool      cellSimplex       = user->cellSimplex;
  const char     *partitioner      = "chaco";
  PetscMPIInt    rank;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = DMCreate(comm, dm);CHKERRQ(ierr);
  ierr = DMSetType(*dm, DMPLEX);CHKERRQ(ierr);
  ierr = DMPlexSetDimension(*dm, dim);CHKERRQ(ierr);
  switch (dim) {
  case 2:
    if (cellSimplex) {
      ierr = CreateSimplexHybrid_2D(comm, *dm);CHKERRQ(ierr);
    } else SETERRQ(comm, PETSC_ERR_ARG_OUTOFRANGE, "Cannot make hybrid meshes for quadrilaterals");
    break;
  default:
    SETERRQ1(comm, PETSC_ERR_ARG_OUTOFRANGE, "Cannot make hybrid meshes for dimension %d", dim);
  }
  {
    DM refinedMesh     = NULL;
    DM distributedMesh = NULL;

    /* Distribute mesh over processes */
    ierr = DMPlexDistribute(*dm, partitioner, 0, NULL, &distributedMesh);CHKERRQ(ierr);
    if (distributedMesh) {
      PetscInt cMax = PETSC_DETERMINE, fMax = PETSC_DETERMINE;

      /* Do not know how to preserve this after distribution */
      if (rank) {
        cMax = 1;
        fMax = 11;
      }
      ierr = DMPlexSetHybridBounds(distributedMesh, cMax, PETSC_DETERMINE, fMax, PETSC_DETERMINE);CHKERRQ(ierr);
      ierr = DMDestroy(dm);CHKERRQ(ierr);
      *dm  = distributedMesh;
    }
    if (refinementUniform) {
      ierr = DMPlexSetRefinementUniform(*dm, refinementUniform);CHKERRQ(ierr);
      ierr = DMRefine(*dm, comm, &refinedMesh);CHKERRQ(ierr);
      if (refinedMesh) {
        ierr = DMDestroy(dm);CHKERRQ(ierr);
        *dm  = refinedMesh;
      }
    }
  }
  ierr     = PetscObjectSetName((PetscObject) *dm, "Hybrid Mesh");CHKERRQ(ierr);
  ierr     = DMSetFromOptions(*dm);CHKERRQ(ierr);
  user->dm = *dm;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char **argv)
{
  AppCtx         user;                 /* user-defined work context */
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help);CHKERRQ(ierr);
  ierr = ProcessOptions(PETSC_COMM_WORLD, &user);CHKERRQ(ierr);
  ierr = CreateMesh(PETSC_COMM_WORLD, &user, &user.dm);CHKERRQ(ierr);
  ierr = DMDestroy(&user.dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}
