static char help[] = "Tests for creation of hybrid meshes\n\n";

/* TODO
 - Propagate hybridSize with distribution
 - Test with multiple fault segments
 - Test with embedded fault
 - Test with multiple faults
 - Move over all PyLith tests?
*/

#include <petscdmplex.h>
/* List of test meshes

Simplex
-------
Test 0:
Two triangles sharing a face

        4
      / | \
     8  |  9
    /   |   \
   2  0 7 1  5
    \   |   /
     6  |  10
      \ | /
        3

should become two triangles separated by a zero-volume cell with 6 vertices

        5--16--8              4--12--6 3
      / |      | \          / |      | | \
    11  |      |  12       9  |      | |  4
    /   |      |   \      /   |      | |   \
   3  0 10  2 14 1  6    2  0 8  1  10 6 0  1
    \   |      |   /      \   |      | |   /
     9  |      |  13       7  |      | |  5
      \ |      | /          \ |      | | /
        4--15--7              3--11--5 2

Hex
------
Test 0:
Two quads sharing a face

   5--10---4--14---7
   |       |       |
  11   0   9   1  13
   |       |       |
   2---8---3--12---6

should become two quads separated by a zero-volume cell with 4 vertices

   6--13---5-20-10--17---8    5--10---4-14--7  4---7---2
   |       |     |       |    |       |     |  |       |
  14   0  12  2 18   1  16   11   0   9  1 12  8   0   6
   |       |     |       |    |       |     |  |       |
   3--11---4-19--9--15---7    2---8---3-13--6  3---5---1
*/

typedef struct {
  DM        dm;
  PetscInt  debug;       /* The debugging level */
  PetscInt  dim;         /* The topological mesh dimension */
  PetscBool cellSimplex; /* Use simplices or hexes */
} AppCtx;

#undef __FUNCT__
#define __FUNCT__ "ProcessOptions"
PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  options->debug       = 0;
  options->dim         = 2;
  options->cellSimplex = PETSC_TRUE;

  ierr = PetscOptionsBegin(comm, "", "Meshing Problem Options", "DMPLEX");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-debug", "The debugging level", "ex4.c", options->debug, &options->debug, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dim", "The topological mesh dimension", "ex4.c", options->dim, &options->dim, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-cell_simplex", "Use simplices if true, otherwise hexes", "ex4.c", options->cellSimplex, &options->cellSimplex, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();
  PetscFunctionReturn(0);
};

#undef __FUNCT__
#define __FUNCT__ "CreateTopology"
PetscErrorCode CreateTopology(DM dm, PetscInt depth, const PetscInt numPoints[], const PetscInt coneSize[], const PetscInt cones[], const PetscInt coneOrientations[], const PetscScalar vertexCoords[])
{
  Vec            coordinates;
  PetscSection   coordSection;
  PetscScalar   *coords;
  PetscInt       coordSize, firstVertex = numPoints[depth], pStart = 0, pEnd = 0, p, v, dim, d, off;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMPlexGetDimension(dm, &dim);CHKERRQ(ierr);
  for(d = 0; d <= depth; ++d) {
    pEnd += numPoints[d];
  }
  ierr = DMPlexSetChart(dm, pStart, pEnd);CHKERRQ(ierr);
  for(p = pStart; p < pEnd; ++p) {
    ierr = DMPlexSetConeSize(dm, p, coneSize[p-pStart]);CHKERRQ(ierr);
  }
  ierr = DMSetUp(dm);CHKERRQ(ierr); /* Allocate space for cones */
  for(p = pStart, off = 0; p < pEnd; off += coneSize[p-pStart], ++p) {
    ierr = DMPlexSetCone(dm, p, &cones[off]);CHKERRQ(ierr);
    ierr = DMPlexSetConeOrientation(dm, p, &coneOrientations[off]);CHKERRQ(ierr);
  }
  ierr = DMPlexSymmetrize(dm);CHKERRQ(ierr);
  ierr = DMPlexStratify(dm);CHKERRQ(ierr);
  /* Build coordinates */
  ierr = DMPlexGetCoordinateSection(dm, &coordSection);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(coordSection, firstVertex, firstVertex+numPoints[0]);CHKERRQ(ierr);
  for(v = firstVertex; v < firstVertex+numPoints[0]; ++v) {
    ierr = PetscSectionSetDof(coordSection, v, dim);CHKERRQ(ierr);
  }
  ierr = PetscSectionSetUp(coordSection);CHKERRQ(ierr);
  ierr = PetscSectionGetStorageSize(coordSection, &coordSize);CHKERRQ(ierr);
  ierr = VecCreate(((PetscObject) dm)->comm, &coordinates);CHKERRQ(ierr);
  ierr = VecSetSizes(coordinates, coordSize, PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = VecSetFromOptions(coordinates);CHKERRQ(ierr);
  ierr = VecGetArray(coordinates, &coords);CHKERRQ(ierr);
  for(v = 0; v < numPoints[0]; ++v) {
    PetscInt off;

    ierr = PetscSectionGetOffset(coordSection, v+firstVertex, &off);CHKERRQ(ierr);
    for(d = 0; d < dim; ++d) {
      coords[off+d] = vertexCoords[v*dim+d];
    }
  }
  ierr = VecRestoreArray(coordinates, &coords);CHKERRQ(ierr);
  ierr = DMSetCoordinatesLocal(dm, coordinates);CHKERRQ(ierr);
  ierr = VecDestroy(&coordinates);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CreateSimplex_2D"
PetscErrorCode CreateSimplex_2D(MPI_Comm comm, DM dm)
{
  PetscInt       depth = 2, testNum  = 0, p;
  PetscMPIInt    rank;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  if (!rank) {
    switch(testNum) {
    case 0:
    {
      PetscInt    numPoints[3]         = {4, 5, 2};
      PetscInt    coneSize[11]         = {3, 3, 0, 0, 0, 0, 2, 2, 2, 2, 2};
      PetscInt    cones[16]            = {6, 7, 8,  9, 7, 10,  2, 3,  3, 4,  4, 2,  5, 4,  3, 5};
      PetscInt    coneOrientations[16] = {0, 0, 0,  0, -2, 0,  0, 0,  0, 0,  0, 0,  0, 0,  0, 0};
      PetscScalar vertexCoords[8]      = {-0.5, 0.5, 0.0, 0.0, 0.0, 1.0, 0.5, 0.5};
      PetscInt    markerPoints[16]     = {2, 1, 3, 1, 4, 1, 5, 1, 6, 1, 8, 1, 9, 1, 10, 1};
      PetscInt    faultPoints[6]       = {7, 1, 3, 0, 4, 0};

      ierr = CreateTopology(dm, depth, numPoints, coneSize, cones, coneOrientations, vertexCoords);CHKERRQ(ierr);
      for(p = 0; p < 8; ++p) {
        ierr = DMPlexSetLabelValue(dm, "marker", markerPoints[p*2], markerPoints[p*2+1]);CHKERRQ(ierr);
      }
      for(p = 0; p < 3; ++p) {
        ierr = DMPlexSetLabelValue(dm, "fault", faultPoints[p*2], faultPoints[p*2+1]);CHKERRQ(ierr);
      }
    }
    break;
    default:
      SETERRQ1(comm, PETSC_ERR_ARG_OUTOFRANGE, "No test mesh %d", testNum);
    }
  } else {
    PetscInt numPoints[3] = {0, 0, 0};

    ierr = CreateTopology(dm, depth, numPoints, PETSC_NULL, PETSC_NULL, PETSC_NULL, PETSC_NULL);CHKERRQ(ierr);
    ierr = DMPlexCreateLabel(dm, "fault");CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CreateHex_2D"
PetscErrorCode CreateHex_2D(MPI_Comm comm, DM dm)
{
  PetscInt       depth = 2, testNum  = 0, p;
  PetscMPIInt    rank;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  if (!rank) {
    switch(testNum) {
    case 0:
    {
      PetscInt    numPoints[3]         = {6, 7, 2};
      PetscInt    coneSize[15]         = {4, 4, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2};
      PetscInt    cones[22]            = {8, 9, 10, 11,  12, 13, 14,  9,  2, 3,  3, 4,  4, 5,  5, 2,  3, 6,  6, 7,  7, 4};
      PetscInt    coneOrientations[22] = {0, 0,  0 , 0,   0,  0,  0, -2,  0, 0,  0, 0,  0, 0,  0, 0,  0, 0,  0, 0,  0, 0};
      PetscScalar vertexCoords[12]     = {-0.5, 0.0, 0.0, 0.0, 0.0, 1.0, -0.5, 1.0, 0.5, 0.0, 0.5, 1.0};
      PetscInt    markerPoints[24]     = {2, 1, 3, 1, 4, 1, 5, 1, 6, 1, 7, 1, 8, 1, 10, 1, 11, 1, 12, 1, 13, 1, 14, 1};
      PetscInt    faultPoints[6]       = {9, 1, 3, 0, 4, 0};

      ierr = CreateTopology(dm, depth, numPoints, coneSize, cones, coneOrientations, vertexCoords);CHKERRQ(ierr);
      for(p = 0; p < 12; ++p) {
        ierr = DMPlexSetLabelValue(dm, "marker", markerPoints[p*2], markerPoints[p*2+1]);CHKERRQ(ierr);
      }
      for(p = 0; p < 3; ++p) {
        ierr = DMPlexSetLabelValue(dm, "fault", faultPoints[p*2], faultPoints[p*2+1]);CHKERRQ(ierr);
      }
    }
    break;
    default:
      SETERRQ1(comm, PETSC_ERR_ARG_OUTOFRANGE, "No test mesh %d", testNum);
    }
  } else {
    PetscInt numPoints[3] = {0, 0, 0};

    ierr = CreateTopology(dm, depth, numPoints, PETSC_NULL, PETSC_NULL, PETSC_NULL, PETSC_NULL);CHKERRQ(ierr);
    ierr = DMPlexCreateLabel(dm, "fault");CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CreateMesh"
PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscInt       dim         = user->dim;
  PetscBool      cellSimplex = user->cellSimplex;
  const char    *partitioner = "chaco";
  PetscMPIInt    rank;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = DMCreate(comm, dm);CHKERRQ(ierr);
  ierr = DMSetType(*dm, DMPLEX);CHKERRQ(ierr);
  ierr = DMPlexSetDimension(*dm, dim);CHKERRQ(ierr);
  switch(dim) {
  case 2:
    if (cellSimplex) {
      ierr = CreateSimplex_2D(comm, *dm);CHKERRQ(ierr);
    } else {
      ierr = CreateHex_2D(comm, *dm);CHKERRQ(ierr);
    }
    break;
  default:
    SETERRQ1(comm, PETSC_ERR_ARG_OUTOFRANGE, "Cannot make hybrid meshes for dimension %d", dim);
  }
  {
    DM      hybridMesh = PETSC_NULL;
    DMLabel label;

    ierr = DMPlexGetLabel(*dm, "fault", &label);CHKERRQ(ierr);
    ierr = DMLabelCohesiveComplete(*dm, label);CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedAllow(PETSC_VIEWER_STDOUT_WORLD, PETSC_TRUE);CHKERRQ(ierr);
    ierr = DMLabelView(label, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = DMPlexConstructCohesiveCells(*dm, "fault", &hybridMesh);CHKERRQ(ierr);
    ierr = DMDestroy(dm);CHKERRQ(ierr);
    *dm  = hybridMesh;
  }
  {
    DM distributedMesh = PETSC_NULL;

    /* Distribute mesh over processes */
    ierr = DMPlexDistribute(*dm, partitioner, 0, &distributedMesh);CHKERRQ(ierr);
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
  }
  ierr = PetscObjectSetName((PetscObject) *dm, "Hybrid Mesh");CHKERRQ(ierr);
  ierr = DMSetFromOptions(*dm);CHKERRQ(ierr);
  user->dm = *dm;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char **argv)
{
  AppCtx         user;                 /* user-defined work context */
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, PETSC_NULL, help);CHKERRQ(ierr);
  ierr = ProcessOptions(PETSC_COMM_WORLD, &user);CHKERRQ(ierr);
  ierr = CreateMesh(PETSC_COMM_WORLD, &user, &user.dm);CHKERRQ(ierr);
  ierr = DMDestroy(&user.dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}
