static char help[] = "Tests interpolation and output of hybrid meshes\n\n";

#include <petscdmplex.h>

typedef struct {
  char      filename[PETSC_MAX_PATH_LEN]; /* Import mesh from file */
  PetscBool interpolate;                  /* Interpolate the mesh */
  PetscInt  meshNum;                      /* Which mesh we should construct */
} AppCtx;

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscFunctionBeginUser;
  options->filename[0] = '\0';
  options->interpolate = PETSC_FALSE;
  options->meshNum     = 0;

  PetscOptionsBegin(comm, "", "Hybrid Output Test Options", "DMPLEX");
  PetscCall(PetscOptionsString("-filename", "The mesh file", "ex8.c", options->filename, options->filename, sizeof(options->filename), NULL));
  PetscCall(PetscOptionsBool("-interpolate", "Interpolate the mesh", "ex8.c", options->interpolate, &options->interpolate, NULL));
  PetscCall(PetscOptionsBoundedInt("-mesh_num", "The mesh we should construct", "ex8.c", options->meshNum, &options->meshNum, NULL, 0));
  PetscOptionsEnd();

  PetscFunctionReturn(0);
}

static PetscErrorCode CreateHybridMesh(MPI_Comm comm, PetscBool interpolate, DM *dm)
{
  PetscInt dim;

  PetscFunctionBegin;
  dim = 3;
  PetscCall(DMCreate(comm, dm));
  PetscCall(PetscObjectSetName((PetscObject)*dm, "Simple Hybrid Mesh"));
  PetscCall(DMSetType(*dm, DMPLEX));
  PetscCall(DMSetDimension(*dm, dim));
  {
    /* Simple mesh with 2 tets and 1 wedge */
    PetscInt    numPoints[2]         = {8, 3};
    PetscInt    coneSize[11]         = {4, 4, 6, 0, 0, 0, 0, 0, 0, 0, 0};
    PetscInt    cones[14]            = {4, 5, 6, 3, 7, 9, 8, 10, 4, 5, 6, 7, 8, 9};
    PetscInt    coneOrientations[14] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    PetscScalar vertexCoords[48]     = {-1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 0.0};

    PetscCall(DMPlexCreateFromDAG(*dm, 1, numPoints, coneSize, cones, coneOrientations, vertexCoords));
    if (interpolate) {
      DM idm;

      PetscCall(DMPlexInterpolate(*dm, &idm));
      PetscCall(DMDestroy(dm));
      *dm = idm;
    }
    PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));
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
  PetscInt dim;

  PetscFunctionBegin;
  dim = 3;
  PetscCall(DMCreate(comm, dm));
  PetscCall(PetscObjectSetName((PetscObject)*dm, "Reverse Hybrid Mesh"));
  PetscCall(DMSetType(*dm, DMPLEX));
  PetscCall(DMSetDimension(*dm, dim));
  {
    /* Simple mesh with 2 hexes and 3 wedges */
    PetscInt    numPoints[2]         = {16, 5};
    PetscInt    coneSize[21]         = {8, 8, 6, 6, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    PetscInt    cones[34]            = {5, 6, 12, 11, 8, 14, 15, 9, 6, 7, 13, 12, 9, 15, 16, 10, 11, 17, 12, 14, 19, 15, 12, 18, 13, 15, 20, 16, 12, 17, 18, 15, 19, 20};
    PetscInt    coneOrientations[34] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    PetscScalar vertexCoords[48]     = {-1.0, -1.0, 0.0, -1.0, 0.0,  0.0, -1.0, 1.0, 0.0, -1.0, -1.0, 1.0, -1.0, 0.0,  1.0, -1.0, 1.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0,
                                        0.0,  1.0,  0.0, 0.0,  -1.0, 1.0, 0.0,  0.0, 1.0, 0.0,  1.0,  1.0, 1.0,  -1.0, 0.0, 1.0,  1.0, 0.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0};

    PetscCall(DMPlexCreateFromDAG(*dm, 1, numPoints, coneSize, cones, coneOrientations, vertexCoords));
    if (interpolate) {
      DM idm;

      PetscCall(DMPlexInterpolate(*dm, &idm));
      PetscCall(DMDestroy(dm));
      *dm = idm;
    }
    PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode OrderHybridMesh(DM *dm)
{
  DM        pdm;
  IS        perm;
  PetscInt *ind;
  PetscInt  dim, pStart, pEnd, p, cStart, cEnd, c, Nhyb = 0, off[2];

  PetscFunctionBegin;
  PetscCall(DMGetDimension(*dm, &dim));
  PetscCheck(dim == 3, PetscObjectComm((PetscObject)*dm), PETSC_ERR_SUP, "No support for dimension %" PetscInt_FMT, dim);
  PetscCall(DMPlexGetChart(*dm, &pStart, &pEnd));
  PetscCall(PetscMalloc1(pEnd - pStart, &ind));
  for (p = 0; p < pEnd - pStart; ++p) ind[p] = p;
  PetscCall(DMPlexGetHeightStratum(*dm, 0, &cStart, &cEnd));
  for (c = cStart; c < cEnd; ++c) {
    PetscInt coneSize;

    PetscCall(DMPlexGetConeSize(*dm, c, &coneSize));
    if (coneSize == 6) ++Nhyb;
  }
  off[0] = 0;
  off[1] = cEnd - Nhyb;
  for (c = cStart; c < cEnd; ++c) {
    PetscInt coneSize;

    PetscCall(DMPlexGetConeSize(*dm, c, &coneSize));
    if (coneSize == 6) ind[c] = off[1]++;
    else ind[c] = off[0]++;
  }
  PetscCheck(off[0] == cEnd - Nhyb, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Number of normal cells %" PetscInt_FMT " should be %" PetscInt_FMT, off[0], cEnd - Nhyb);
  PetscCheck(off[1] == cEnd, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Number of hybrid cells %" PetscInt_FMT " should be %" PetscInt_FMT, off[1] - off[0], Nhyb);
  PetscCall(ISCreateGeneral(PETSC_COMM_SELF, pEnd - pStart, ind, PETSC_OWN_POINTER, &perm));
  PetscCall(DMPlexPermute(*dm, perm, &pdm));
  PetscCall(ISDestroy(&perm));
  PetscCall(DMDestroy(dm));
  *dm = pdm;
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  const char *filename    = user->filename;
  PetscBool   interpolate = user->interpolate;
  PetscInt    meshNum     = user->meshNum;
  size_t      len;

  PetscFunctionBegin;
  PetscCall(PetscStrlen(filename, &len));
  if (len) {
    PetscCall(DMPlexCreateFromFile(comm, filename, "ex34_plex", PETSC_FALSE, dm));
    PetscCall(OrderHybridMesh(dm));
    if (interpolate) {
      DM idm;

      PetscCall(DMPlexInterpolate(*dm, &idm));
      PetscCall(DMDestroy(dm));
      *dm = idm;
    }
    PetscCall(PetscObjectSetName((PetscObject)*dm, "Input Mesh"));
    PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));
  } else {
    switch (meshNum) {
    case 0:
      PetscCall(CreateHybridMesh(comm, interpolate, dm));
      break;
    case 1:
      PetscCall(CreateReverseHybridMesh(comm, interpolate, dm));
      break;
    default:
      SETERRQ(comm, PETSC_ERR_ARG_WRONG, "Unknown mesh number %" PetscInt_FMT, user->meshNum);
    }
  }
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM     dm;
  AppCtx user;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(ProcessOptions(PETSC_COMM_WORLD, &user));
  PetscCall(CreateMesh(PETSC_COMM_WORLD, &user, &dm));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
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
