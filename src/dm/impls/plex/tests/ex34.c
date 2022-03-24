static char help[] = "Tests interpolation and output of hybrid meshes\n\n";

#include <petscdmplex.h>

typedef struct {
  char      filename[PETSC_MAX_PATH_LEN]; /* Import mesh from file */
  PetscBool interpolate;                  /* Interpolate the mesh */
  PetscInt  meshNum;                      /* Which mesh we should construct */
} AppCtx;

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  options->filename[0] = '\0';
  options->interpolate = PETSC_FALSE;
  options->meshNum     = 0;

  ierr = PetscOptionsBegin(comm, "", "Hybrid Output Test Options", "DMPLEX");CHKERRQ(ierr);
  CHKERRQ(PetscOptionsString("-filename", "The mesh file", "ex8.c", options->filename, options->filename, sizeof(options->filename), NULL));
  CHKERRQ(PetscOptionsBool("-interpolate", "Interpolate the mesh", "ex8.c", options->interpolate, &options->interpolate, NULL));
  CHKERRQ(PetscOptionsBoundedInt("-mesh_num", "The mesh we should construct", "ex8.c", options->meshNum, &options->meshNum, NULL,0));
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

static PetscErrorCode CreateHybridMesh(MPI_Comm comm, PetscBool interpolate, DM *dm)
{
  PetscInt       dim;

  PetscFunctionBegin;
  dim  = 3;
  CHKERRQ(DMCreate(comm, dm));
  CHKERRQ(PetscObjectSetName((PetscObject) *dm, "Simple Hybrid Mesh"));
  CHKERRQ(DMSetType(*dm, DMPLEX));
  CHKERRQ(DMSetDimension(*dm, dim));
  {
    /* Simple mesh with 2 tets and 1 wedge */
    PetscInt    numPoints[2]         = {8, 3};
    PetscInt    coneSize[11]         = {4, 4, 6,  0, 0, 0, 0, 0, 0, 0, 0};
    PetscInt    cones[14]            = {4, 5, 6, 3,  7, 9, 8, 10,  4, 5, 6, 7, 8, 9};
    PetscInt    coneOrientations[14] = {0, 0, 0, 0,  0, 0, 0,  0,  0, 0, 0, 0, 0, 0};
    PetscScalar vertexCoords[48]     = {-1.0, 1.0, 0.0,
                                         0.0, 0.0, 0.0,  0.0, 1.0, -1.0,  0.0, 1.0, 1.0,
                                         1.0, 0.0, 0.0,  1.0, 1.0, -1.0,  1.0, 1.0, 1.0,
                                         2.0, 1.0, 0.0};

    CHKERRQ(DMPlexCreateFromDAG(*dm, 1, numPoints, coneSize, cones, coneOrientations, vertexCoords));
    if (interpolate) {
      DM idm;

      CHKERRQ(DMPlexInterpolate(*dm, &idm));
      CHKERRQ(DMDestroy(dm));
      *dm  = idm;
    }
    CHKERRQ(DMViewFromOptions(*dm, NULL, "-dm_view"));
  }
  PetscFunctionReturn(0);
}

/*
   This is not a valid mesh. We need to either change to tensor quad prisms or regular triangular prisms.

           10-------16--------20
           /|        |
          / |        |
         /  |        |
        9---|---15   |
       /|   7    |  13--------18
      / |  /     |  /    ____/
     /  | /      | /____/
    8   |/  14---|//---19
    |   6    |  12
    |  /     |  / \
    | /      | /   \__
    |/       |/       \
    5--------11--------17
*/
static PetscErrorCode CreateReverseHybridMesh(MPI_Comm comm, PetscBool interpolate, DM *dm)
{
  PetscInt       dim;

  PetscFunctionBegin;
  dim  = 3;
  CHKERRQ(DMCreate(comm, dm));
  CHKERRQ(PetscObjectSetName((PetscObject) *dm, "Reverse Hybrid Mesh"));
  CHKERRQ(DMSetType(*dm, DMPLEX));
  CHKERRQ(DMSetDimension(*dm, dim));
  {
    /* Simple mesh with 2 hexes and 3 wedges */
    PetscInt    numPoints[2]         = {16, 5};
    PetscInt    coneSize[21]         = {8, 8, 6, 6, 6,
                                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    PetscInt    cones[34]            = { 5,  6, 12, 11, 8, 14, 15, 9,
                                         6,  7, 13, 12, 9, 15, 16, 10,
                                        11, 17, 12, 14, 19, 15,
                                        12, 18, 13, 15, 20, 16,
                                        12, 17, 18, 15, 19, 20};
    PetscInt    coneOrientations[34] = {0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0,
                                        0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0};
    PetscScalar vertexCoords[48]     = {-1.0, -1.0, 0.0,  -1.0,  0.0, 0.0,  -1.0, 1.0, 0.0,
                                        -1.0, -1.0, 1.0,  -1.0,  0.0, 1.0,  -1.0, 1.0, 1.0,
                                         0.0, -1.0, 0.0,   0.0,  0.0, 0.0,   0.0, 1.0, 0.0,
                                         0.0, -1.0, 1.0,   0.0,  0.0, 1.0,   0.0, 1.0, 1.0,
                                         1.0, -1.0, 0.0,                     1.0, 1.0, 0.0,
                                         1.0, -1.0, 1.0,                     1.0, 1.0, 1.0};

    CHKERRQ(DMPlexCreateFromDAG(*dm, 1, numPoints, coneSize, cones, coneOrientations, vertexCoords));
    if (interpolate) {
      DM idm;

      CHKERRQ(DMPlexInterpolate(*dm, &idm));
      CHKERRQ(DMDestroy(dm));
      *dm  = idm;
    }
    CHKERRQ(DMViewFromOptions(*dm, NULL, "-dm_view"));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode OrderHybridMesh(DM *dm)
{
  DM             pdm;
  IS             perm;
  PetscInt      *ind;
  PetscInt       dim, pStart, pEnd, p, cStart, cEnd, c, Nhyb = 0, off[2];

  PetscFunctionBegin;
  CHKERRQ(DMGetDimension(*dm, &dim));
  PetscCheckFalse(dim != 3,PetscObjectComm((PetscObject) *dm), PETSC_ERR_SUP, "No support for dimension %D", dim);
  CHKERRQ(DMPlexGetChart(*dm, &pStart, &pEnd));
  CHKERRQ(PetscMalloc1(pEnd-pStart, &ind));
  for (p = 0; p < pEnd-pStart; ++p) ind[p] = p;
  CHKERRQ(DMPlexGetHeightStratum(*dm, 0, &cStart, &cEnd));
  for (c = cStart; c < cEnd; ++c) {
    PetscInt coneSize;

    CHKERRQ(DMPlexGetConeSize(*dm, c, &coneSize));
    if (coneSize == 6) ++Nhyb;
  }
  off[0] = 0;
  off[1] = cEnd - Nhyb;
  for (c = cStart; c < cEnd; ++c) {
    PetscInt coneSize;

    CHKERRQ(DMPlexGetConeSize(*dm, c, &coneSize));
    if (coneSize == 6) ind[c] = off[1]++;
    else               ind[c] = off[0]++;
  }
  PetscCheckFalse(off[0] != cEnd - Nhyb,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Number of normal cells %D should be %D", off[0], cEnd - Nhyb);
  PetscCheckFalse(off[1] != cEnd,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Number of hybrid cells %D should be %D", off[1] - off[0], Nhyb);
  CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF, pEnd-pStart, ind, PETSC_OWN_POINTER, &perm));
  CHKERRQ(DMPlexPermute(*dm, perm, &pdm));
  CHKERRQ(ISDestroy(&perm));
  CHKERRQ(DMDestroy(dm));
  *dm  = pdm;
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  const char    *filename    = user->filename;
  PetscBool      interpolate = user->interpolate;
  PetscInt       meshNum     = user->meshNum;
  size_t         len;

  PetscFunctionBegin;
  CHKERRQ(PetscStrlen(filename, &len));
  if (len) {
    CHKERRQ(DMPlexCreateFromFile(comm, filename, "ex34_plex", PETSC_FALSE, dm));
    CHKERRQ(OrderHybridMesh(dm));
    if (interpolate) {
      DM idm;

      CHKERRQ(DMPlexInterpolate(*dm, &idm));
      CHKERRQ(DMDestroy(dm));
      *dm  = idm;
    }
    CHKERRQ(PetscObjectSetName((PetscObject) *dm, "Input Mesh"));
    CHKERRQ(DMViewFromOptions(*dm, NULL, "-dm_view"));
  } else {
    switch (meshNum) {
    case 0:
      CHKERRQ(CreateHybridMesh(comm, interpolate, dm));break;
    case 1:
      CHKERRQ(CreateReverseHybridMesh(comm, interpolate, dm));break;
    default: SETERRQ(comm, PETSC_ERR_ARG_WRONG, "Unknown mesh number %D", user->meshNum);
    }
  }
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM             dm;
  AppCtx         user;

  CHKERRQ(PetscInitialize(&argc, &argv, NULL,help));
  CHKERRQ(ProcessOptions(PETSC_COMM_WORLD, &user));
  CHKERRQ(CreateMesh(PETSC_COMM_WORLD, &user, &dm));
  CHKERRQ(DMDestroy(&dm));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: 0
    args: -interpolate -dm_view ascii::ascii_info_detail

  # Test needs to be reworked
  test:
    requires: BROKEN
    suffix: 1
    args: -mesh_num 1 -interpolate -dm_view ascii::ascii_info_detail

TEST*/
