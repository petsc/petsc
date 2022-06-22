static char help[] = "Tests for DMLabel\n\n";

#include <petscdmplex.h>
#include <petsc/private/dmimpl.h>

static PetscErrorCode TestInsertion()
{
  DMLabel        label, label2;
  const PetscInt values[5] = {0, 3, 4, -1, 176}, N = 10000;
  PetscInt       i, v;

  PetscFunctionBegin;
  PetscCall(DMLabelCreate(PETSC_COMM_SELF, "Test Label", &label));
  PetscCall(DMLabelSetDefaultValue(label, -100));
  for (i = 0; i < N; ++i) {
    PetscCall(DMLabelSetValue(label, i, values[i%5]));
  }
  /* Test get in hash mode */
  for (i = 0; i < N; ++i) {
    PetscInt val;

    PetscCall(DMLabelGetValue(label, i, &val));
    PetscCheck(val == values[i%5],PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Value %" PetscInt_FMT " for point %" PetscInt_FMT " should be %" PetscInt_FMT, val, i, values[i%5]);
  }
  /* Test stratum */
  for (v = 0; v < 5; ++v) {
    IS              stratum;
    const PetscInt *points;
    PetscInt        n;

    PetscCall(DMLabelGetStratumIS(label, values[v], &stratum));
    PetscCheck(stratum,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Stratum %" PetscInt_FMT " is empty!", v);
    PetscCall(ISGetIndices(stratum, &points));
    PetscCall(ISGetLocalSize(stratum, &n));
    for (i = 0; i < n; ++i) {
      PetscCheck(points[i] == i*5+v,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Point %" PetscInt_FMT " should be %" PetscInt_FMT, points[i], i*5+v);
    }
    PetscCall(ISRestoreIndices(stratum, &points));
    PetscCall(ISDestroy(&stratum));
  }
  /* Test get in array mode */
  for (i = 0; i < N; ++i) {
    PetscInt val;

    PetscCall(DMLabelGetValue(label, i, &val));
    PetscCheck(val == values[i%5],PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Value %" PetscInt_FMT " should be %" PetscInt_FMT, val, values[i%5]);
  }
  /* Test Duplicate */
  PetscCall(DMLabelDuplicate(label, &label2));
  for (i = 0; i < N; ++i) {
    PetscInt val;

    PetscCall(DMLabelGetValue(label2, i, &val));
    PetscCheck(val == values[i%5],PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Value %" PetscInt_FMT " should be %" PetscInt_FMT, val, values[i%5]);
  }
  PetscCall(DMLabelDestroy(&label2));
  PetscCall(DMLabelDestroy(&label));
  PetscFunctionReturn(0);
}

static PetscErrorCode TestEmptyStrata(MPI_Comm comm)
{
  DM               dm, dmDist;
  PetscPartitioner part;
  PetscInt         c0[6]  = {2,3,6,7,9,11};
  PetscInt         c1[6]  = {4,5,7,8,10,12};
  PetscInt         c2[4]  = {13,15,19,21};
  PetscInt         c3[4]  = {14,16,20,22};
  PetscInt         c4[4]  = {15,17,21,23};
  PetscInt         c5[4]  = {16,18,22,24};
  PetscInt         c6[4]  = {13,14,19,20};
  PetscInt         c7[4]  = {15,16,21,22};
  PetscInt         c8[4]  = {17,18,23,24};
  PetscInt         c9[4]  = {13,14,15,16};
  PetscInt         c10[4] = {15,16,17,18};
  PetscInt         c11[4] = {19,20,21,22};
  PetscInt         c12[4] = {21,22,23,24};
  PetscInt         dim    = 3;
  PetscMPIInt      rank;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  /* A 3D box with two adjacent cells, sharing one face and four vertices */
  PetscCall(DMCreate(comm, &dm));
  PetscCall(DMSetType(dm, DMPLEX));
  PetscCall(DMSetDimension(dm, dim));
  if (rank == 0) {
    PetscCall(DMPlexSetChart(dm, 0, 25));
    PetscCall(DMPlexSetConeSize(dm, 0, 6));
    PetscCall(DMPlexSetConeSize(dm, 1, 6));
    PetscCall(DMPlexSetConeSize(dm, 2, 4));
    PetscCall(DMPlexSetConeSize(dm, 3, 4));
    PetscCall(DMPlexSetConeSize(dm, 4, 4));
    PetscCall(DMPlexSetConeSize(dm, 5, 4));
    PetscCall(DMPlexSetConeSize(dm, 6, 4));
    PetscCall(DMPlexSetConeSize(dm, 7, 4));
    PetscCall(DMPlexSetConeSize(dm, 8, 4));
    PetscCall(DMPlexSetConeSize(dm, 9, 4));
    PetscCall(DMPlexSetConeSize(dm, 10, 4));
    PetscCall(DMPlexSetConeSize(dm, 11, 4));
    PetscCall(DMPlexSetConeSize(dm, 12, 4));
  }
  PetscCall(DMSetUp(dm));
  if (rank == 0) {
    PetscCall(DMPlexSetCone(dm, 0, c0));
    PetscCall(DMPlexSetCone(dm, 1, c1));
    PetscCall(DMPlexSetCone(dm, 2, c2));
    PetscCall(DMPlexSetCone(dm, 3, c3));
    PetscCall(DMPlexSetCone(dm, 4, c4));
    PetscCall(DMPlexSetCone(dm, 5, c5));
    PetscCall(DMPlexSetCone(dm, 6, c6));
    PetscCall(DMPlexSetCone(dm, 7, c7));
    PetscCall(DMPlexSetCone(dm, 8, c8));
    PetscCall(DMPlexSetCone(dm, 9, c9));
    PetscCall(DMPlexSetCone(dm, 10, c10));
    PetscCall(DMPlexSetCone(dm, 11, c11));
    PetscCall(DMPlexSetCone(dm, 12, c12));
  }
  PetscCall(DMPlexSymmetrize(dm));
  /* Create a user managed depth label, so that we can leave out edges */
  {
    DMLabel label;
    PetscInt numValues, maxValues = 0, v;

    PetscCall(DMCreateLabel(dm, "depth"));
    PetscCall(DMPlexGetDepthLabel(dm, &label));
    if (rank == 0) {
      PetscInt i;

      for (i = 0; i < 25; ++i) {
        if (i < 2)       PetscCall(DMLabelSetValue(label, i, 3));
        else if (i < 13) PetscCall(DMLabelSetValue(label, i, 2));
        else             {
          if (i==13) PetscCall(DMLabelAddStratum(label, 1));
          PetscCall(DMLabelSetValue(label, i, 0));
        }
      }
    }
    PetscCall(DMLabelGetNumValues(label, &numValues));
    PetscCallMPI(MPI_Allreduce(&numValues, &maxValues, 1, MPIU_INT, MPI_MAX, PetscObjectComm((PetscObject) dm)));
    for (v = numValues; v < maxValues; ++v) PetscCall(DMLabelAddStratum(label,v));
  }
  {
    DMLabel label;
    PetscCall(DMPlexGetDepthLabel(dm, &label));
    PetscCall(DMLabelView(label, PETSC_VIEWER_STDOUT_(comm)));
  }
  PetscCall(DMPlexGetPartitioner(dm,&part));
  PetscCall(PetscPartitionerSetFromOptions(part));
  PetscCall(DMPlexDistribute(dm, 1, NULL, &dmDist));
  if (dmDist) {
    PetscCall(DMDestroy(&dm));
    dm   = dmDist;
  }
  {
    DMLabel label;
    PetscCall(DMPlexGetDepthLabel(dm, &label));
    PetscCall(DMLabelView(label, PETSC_VIEWER_STDOUT_(comm)));
  }
  /* Create a cell vector */
  {
    Vec          v;
    PetscSection s;
    PetscInt     numComp[] = {1};
    PetscInt     dof[]     = {0,0,0,1};
    PetscInt     N;

    PetscCall(DMSetNumFields(dm, 1));
    PetscCall(DMPlexCreateSection(dm, NULL, numComp, dof, 0, NULL, NULL, NULL, NULL, &s));
    PetscCall(DMSetLocalSection(dm, s));
    PetscCall(PetscSectionDestroy(&s));
    PetscCall(DMCreateGlobalVector(dm, &v));
    PetscCall(VecGetSize(v, &N));
    if (N != 2) {
      PetscCall(DMView(dm, PETSC_VIEWER_STDOUT_(comm)));
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "FAIL: Vector size %" PetscInt_FMT " != 2", N);
    }
    PetscCall(VecDestroy(&v));
  }
  PetscCall(DMDestroy(&dm));
  PetscFunctionReturn(0);
}

static PetscErrorCode TestDistribution(MPI_Comm comm)
{
  DM               dm, dmDist;
  PetscPartitioner part;
  DMLabel          label;
  char             filename[PETSC_MAX_PATH_LEN];
  const char      *name    = "test label";
  PetscInt         overlap = 0, cStart, cEnd, c;
  PetscMPIInt      rank;
  PetscBool        flg;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCall(PetscOptionsGetString(NULL, NULL, "-filename", filename, sizeof(filename), &flg));
  if (!flg) PetscFunctionReturn(0);
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-overlap", &overlap, NULL));
  PetscCall(DMPlexCreateFromFile(comm, filename, "ex11_plex", PETSC_TRUE, &dm));
  PetscCall(DMSetBasicAdjacency(dm, PETSC_TRUE, PETSC_FALSE));
  PetscCall(DMCreateLabel(dm, name));
  PetscCall(DMGetLabel(dm, name, &label));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  for (c = cStart; c < cEnd; ++c) {
    PetscCall(DMLabelSetValue(label, c, c));
  }
  PetscCall(DMLabelView(label, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(DMPlexGetPartitioner(dm,&part));
  PetscCall(PetscPartitionerSetFromOptions(part));
  PetscCall(DMPlexDistribute(dm, overlap, NULL, &dmDist));
  if (dmDist) {
    PetscCall(DMDestroy(&dm));
    dm   = dmDist;
  }
  PetscCall(PetscObjectSetName((PetscObject) dm, "Mesh"));
  PetscCall(DMViewFromOptions(dm, NULL, "-dm_view"));
  PetscCall(DMGetLabel(dm, name, &label));
  PetscCall(DMLabelView(label, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(DMDestroy(&dm));
  PetscFunctionReturn(0);
}

static PetscErrorCode TestUniversalLabel(MPI_Comm comm)
{
  DM               dm1, dm2;
  DMLabel          bd1, bd2, ulabel;
  DMUniversalLabel universal;
  PetscInt         pStart, pEnd, p;
  PetscBool        run = PETSC_FALSE, notFile;

  PetscFunctionBeginUser;
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-universal", &run, NULL));
  if (!run) PetscFunctionReturn(0);

  char filename[PETSC_MAX_PATH_LEN];
  PetscBool flg;

  PetscCall(PetscOptionsGetString(NULL, NULL, "-filename", filename, sizeof(filename), &flg));
  if (flg) {
    PetscCall(DMPlexCreateFromFile(comm, filename, "ex11_plex", PETSC_TRUE, &dm1));
  } else {
    PetscCall(DMCreate(comm, &dm1));
    PetscCall(DMSetType(dm1, DMPLEX));
    PetscCall(DMSetFromOptions(dm1));
  }
  PetscCall(DMHasLabel(dm1, "marker", &notFile));
  if (notFile) {
    PetscCall(DMCreateLabel(dm1, "Boundary Faces"));
    PetscCall(DMGetLabel(dm1, "Boundary Faces", &bd1));
    PetscCall(DMPlexMarkBoundaryFaces(dm1, 13, bd1));
    PetscCall(DMCreateLabel(dm1, "Boundary"));
    PetscCall(DMGetLabel(dm1, "Boundary", &bd2));
    PetscCall(DMPlexMarkBoundaryFaces(dm1, 121, bd2));
    PetscCall(DMPlexLabelComplete(dm1, bd2));
  }
  PetscCall(PetscObjectSetName((PetscObject) dm1, "First Mesh"));
  PetscCall(DMViewFromOptions(dm1, NULL, "-dm_view"));

  PetscCall(DMUniversalLabelCreate(dm1, &universal));
  PetscCall(DMUniversalLabelGetLabel(universal, &ulabel));
  PetscCall(PetscObjectViewFromOptions((PetscObject) ulabel, NULL, "-universal_view"));

  if (!notFile) {
    PetscInt Nl, l;

    PetscCall(DMClone(dm1, &dm2));
    PetscCall(DMGetNumLabels(dm2, &Nl));
    for (l = Nl-1; l >= 0; --l) {
      PetscBool   isdepth, iscelltype;
      const char *name;

      PetscCall(DMGetLabelName(dm2, l, &name));
      PetscCall(PetscStrncmp(name, "depth", 6, &isdepth));
      PetscCall(PetscStrncmp(name, "celltype", 9, &iscelltype));
      if (!isdepth && !iscelltype) PetscCall(DMRemoveLabel(dm2, name, NULL));
    }
  } else {
    PetscCall(DMCreate(comm, &dm2));
    PetscCall(DMSetType(dm2, DMPLEX));
    PetscCall(DMSetFromOptions(dm2));
  }
  PetscCall(PetscObjectSetName((PetscObject) dm2, "Second Mesh"));
  PetscCall(DMUniversalLabelCreateLabels(universal, PETSC_TRUE, dm2));
  PetscCall(DMPlexGetChart(dm2, &pStart, &pEnd));
  for (p = pStart; p < pEnd; ++p) {
    PetscInt val;

    PetscCall(DMLabelGetValue(ulabel, p, &val));
    if (val < 0) continue;
    PetscCall(DMUniversalLabelSetLabelValue(universal, dm2, PETSC_TRUE, p, val));
  }
  PetscCall(DMViewFromOptions(dm2, NULL, "-dm_view"));

  PetscCall(DMUniversalLabelDestroy(&universal));
  PetscCall(DMDestroy(&dm1));
  PetscCall(DMDestroy(&dm2));
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  /*PetscCall(ProcessOptions(PETSC_COMM_WORLD, &user));*/
  PetscCall(TestInsertion());
  PetscCall(TestEmptyStrata(PETSC_COMM_WORLD));
  PetscCall(TestDistribution(PETSC_COMM_WORLD));
  PetscCall(TestUniversalLabel(PETSC_COMM_WORLD));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: 0
    requires: triangle
  test:
    suffix: 1
    requires: triangle
    nsize: 2
    args: -petscpartitioner_type simple

  testset:
    suffix: gmsh
    args: -filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/square.msh -petscpartitioner_type simple
    test:
      suffix: 1
      nsize: 1
    test:
      suffix: 2
      nsize: 2

  testset:
    suffix: exodusii
    requires: exodusii
    args: -filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/2Dgrd.exo -petscpartitioner_type simple
    test:
      suffix: 1
      nsize: 1
    test:
      suffix: 2
      nsize: 2

  test:
    suffix: univ
    requires: triangle
    args: -universal -dm_view -universal_view

  test:
    # Note that the labels differ because we have multiply-marked some points during EGADS creation
    suffix: univ_egads_sphere
    requires: egads
    args: -universal -dm_plex_filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/unit_sphere.egadslite -dm_view -universal_view

  test:
    # Note that the labels differ because we have multiply-marked some points during EGADS creation
    suffix: univ_egads_ball
    requires: egads ctetgen
    args: -universal -dm_plex_boundary_filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/unit_sphere.egadslite -dm_view -universal_view

TEST*/
