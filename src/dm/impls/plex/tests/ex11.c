static char help[] = "Tests for DMLabel\n\n";

#include <petscdmplex.h>
#include <petsc/private/dmimpl.h>

static PetscErrorCode TestInsertion()
{
  DMLabel        label, label2;
  const PetscInt values[5] = {0, 3, 4, -1, 176}, N = 10000;
  PetscInt       i, v;

  PetscFunctionBegin;
  CHKERRQ(DMLabelCreate(PETSC_COMM_SELF, "Test Label", &label));
  CHKERRQ(DMLabelSetDefaultValue(label, -100));
  for (i = 0; i < N; ++i) {
    CHKERRQ(DMLabelSetValue(label, i, values[i%5]));
  }
  /* Test get in hash mode */
  for (i = 0; i < N; ++i) {
    PetscInt val;

    CHKERRQ(DMLabelGetValue(label, i, &val));
    PetscCheckFalse(val != values[i%5],PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Value %d for point %d should be %d", val, i, values[i%5]);
  }
  /* Test stratum */
  for (v = 0; v < 5; ++v) {
    IS              stratum;
    const PetscInt *points;
    PetscInt        n;

    CHKERRQ(DMLabelGetStratumIS(label, values[v], &stratum));
    PetscCheck(stratum,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Stratum %d is empty!", v);
    CHKERRQ(ISGetIndices(stratum, &points));
    CHKERRQ(ISGetLocalSize(stratum, &n));
    for (i = 0; i < n; ++i) {
      PetscCheckFalse(points[i] != i*5+v,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Point %d should be %d", points[i], i*5+v);
    }
    CHKERRQ(ISRestoreIndices(stratum, &points));
    CHKERRQ(ISDestroy(&stratum));
  }
  /* Test get in array mode */
  for (i = 0; i < N; ++i) {
    PetscInt val;

    CHKERRQ(DMLabelGetValue(label, i, &val));
    PetscCheckFalse(val != values[i%5],PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Value %d should be %d", val, values[i%5]);
  }
  /* Test Duplicate */
  CHKERRQ(DMLabelDuplicate(label, &label2));
  for (i = 0; i < N; ++i) {
    PetscInt val;

    CHKERRQ(DMLabelGetValue(label2, i, &val));
    PetscCheckFalse(val != values[i%5],PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Value %d should be %d", val, values[i%5]);
  }
  CHKERRQ(DMLabelDestroy(&label2));
  CHKERRQ(DMLabelDestroy(&label));
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
  CHKERRMPI(MPI_Comm_rank(comm, &rank));
  /* A 3D box with two adjacent cells, sharing one face and four vertices */
  CHKERRQ(DMCreate(comm, &dm));
  CHKERRQ(DMSetType(dm, DMPLEX));
  CHKERRQ(DMSetDimension(dm, dim));
  if (rank == 0) {
    CHKERRQ(DMPlexSetChart(dm, 0, 25));
    CHKERRQ(DMPlexSetConeSize(dm, 0, 6));
    CHKERRQ(DMPlexSetConeSize(dm, 1, 6));
    CHKERRQ(DMPlexSetConeSize(dm, 2, 4));
    CHKERRQ(DMPlexSetConeSize(dm, 3, 4));
    CHKERRQ(DMPlexSetConeSize(dm, 4, 4));
    CHKERRQ(DMPlexSetConeSize(dm, 5, 4));
    CHKERRQ(DMPlexSetConeSize(dm, 6, 4));
    CHKERRQ(DMPlexSetConeSize(dm, 7, 4));
    CHKERRQ(DMPlexSetConeSize(dm, 8, 4));
    CHKERRQ(DMPlexSetConeSize(dm, 9, 4));
    CHKERRQ(DMPlexSetConeSize(dm, 10, 4));
    CHKERRQ(DMPlexSetConeSize(dm, 11, 4));
    CHKERRQ(DMPlexSetConeSize(dm, 12, 4));
  }
  CHKERRQ(DMSetUp(dm));
  if (rank == 0) {
    CHKERRQ(DMPlexSetCone(dm, 0, c0));
    CHKERRQ(DMPlexSetCone(dm, 1, c1));
    CHKERRQ(DMPlexSetCone(dm, 2, c2));
    CHKERRQ(DMPlexSetCone(dm, 3, c3));
    CHKERRQ(DMPlexSetCone(dm, 4, c4));
    CHKERRQ(DMPlexSetCone(dm, 5, c5));
    CHKERRQ(DMPlexSetCone(dm, 6, c6));
    CHKERRQ(DMPlexSetCone(dm, 7, c7));
    CHKERRQ(DMPlexSetCone(dm, 8, c8));
    CHKERRQ(DMPlexSetCone(dm, 9, c9));
    CHKERRQ(DMPlexSetCone(dm, 10, c10));
    CHKERRQ(DMPlexSetCone(dm, 11, c11));
    CHKERRQ(DMPlexSetCone(dm, 12, c12));
  }
  CHKERRQ(DMPlexSymmetrize(dm));
  /* Create a user managed depth label, so that we can leave out edges */
  {
    DMLabel label;
    PetscInt numValues, maxValues = 0, v;

    CHKERRQ(DMCreateLabel(dm, "depth"));
    CHKERRQ(DMPlexGetDepthLabel(dm, &label));
    if (rank == 0) {
      PetscInt i;

      for (i = 0; i < 25; ++i) {
        if (i < 2)       CHKERRQ(DMLabelSetValue(label, i, 3));
        else if (i < 13) CHKERRQ(DMLabelSetValue(label, i, 2));
        else             {
          if (i==13) CHKERRQ(DMLabelAddStratum(label, 1));
          CHKERRQ(DMLabelSetValue(label, i, 0));
        }
      }
    }
    CHKERRQ(DMLabelGetNumValues(label, &numValues));
    CHKERRMPI(MPI_Allreduce(&numValues, &maxValues, 1, MPIU_INT, MPI_MAX, PetscObjectComm((PetscObject) dm)));
    for (v = numValues; v < maxValues; ++v) CHKERRQ(DMLabelAddStratum(label,v));
  }
  {
    DMLabel label;
    CHKERRQ(DMPlexGetDepthLabel(dm, &label));
    CHKERRQ(DMLabelView(label, PETSC_VIEWER_STDOUT_(comm)));
  }
  CHKERRQ(DMPlexGetPartitioner(dm,&part));
  CHKERRQ(PetscPartitionerSetFromOptions(part));
  CHKERRQ(DMPlexDistribute(dm, 1, NULL, &dmDist));
  if (dmDist) {
    CHKERRQ(DMDestroy(&dm));
    dm   = dmDist;
  }
  {
    DMLabel label;
    CHKERRQ(DMPlexGetDepthLabel(dm, &label));
    CHKERRQ(DMLabelView(label, PETSC_VIEWER_STDOUT_(comm)));
  }
  /* Create a cell vector */
  {
    Vec          v;
    PetscSection s;
    PetscInt     numComp[] = {1};
    PetscInt     dof[]     = {0,0,0,1};
    PetscInt     N;

    CHKERRQ(DMSetNumFields(dm, 1));
    CHKERRQ(DMPlexCreateSection(dm, NULL, numComp, dof, 0, NULL, NULL, NULL, NULL, &s));
    CHKERRQ(DMSetLocalSection(dm, s));
    CHKERRQ(PetscSectionDestroy(&s));
    CHKERRQ(DMCreateGlobalVector(dm, &v));
    CHKERRQ(VecGetSize(v, &N));
    if (N != 2) {
      CHKERRQ(DMView(dm, PETSC_VIEWER_STDOUT_(comm)));
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "FAIL: Vector size %d != 2", N);
    }
    CHKERRQ(VecDestroy(&v));
  }
  CHKERRQ(DMDestroy(&dm));
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
  CHKERRMPI(MPI_Comm_rank(comm, &rank));
  CHKERRQ(PetscOptionsGetString(NULL, NULL, "-filename", filename, sizeof(filename), &flg));
  if (!flg) PetscFunctionReturn(0);
  CHKERRQ(PetscOptionsGetInt(NULL, NULL, "-overlap", &overlap, NULL));
  CHKERRQ(DMPlexCreateFromFile(comm, filename, "ex11_plex", PETSC_TRUE, &dm));
  CHKERRQ(DMSetBasicAdjacency(dm, PETSC_TRUE, PETSC_FALSE));
  CHKERRQ(DMCreateLabel(dm, name));
  CHKERRQ(DMGetLabel(dm, name, &label));
  CHKERRQ(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  for (c = cStart; c < cEnd; ++c) {
    CHKERRQ(DMLabelSetValue(label, c, c));
  }
  CHKERRQ(DMLabelView(label, PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(DMPlexGetPartitioner(dm,&part));
  CHKERRQ(PetscPartitionerSetFromOptions(part));
  CHKERRQ(DMPlexDistribute(dm, overlap, NULL, &dmDist));
  if (dmDist) {
    CHKERRQ(DMDestroy(&dm));
    dm   = dmDist;
  }
  CHKERRQ(PetscObjectSetName((PetscObject) dm, "Mesh"));
  CHKERRQ(DMViewFromOptions(dm, NULL, "-dm_view"));
  CHKERRQ(DMGetLabel(dm, name, &label));
  CHKERRQ(DMLabelView(label, PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(DMDestroy(&dm));
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
  CHKERRQ(PetscOptionsGetBool(NULL, NULL, "-universal", &run, NULL));
  if (!run) PetscFunctionReturn(0);

  char filename[PETSC_MAX_PATH_LEN];
  PetscBool flg;

  CHKERRQ(PetscOptionsGetString(NULL, NULL, "-filename", filename, sizeof(filename), &flg));
  if (flg) {
    CHKERRQ(DMPlexCreateFromFile(comm, filename, "ex11_plex", PETSC_TRUE, &dm1));
  } else {
    CHKERRQ(DMCreate(comm, &dm1));
    CHKERRQ(DMSetType(dm1, DMPLEX));
    CHKERRQ(DMSetFromOptions(dm1));
  }
  CHKERRQ(DMHasLabel(dm1, "marker", &notFile));
  if (notFile) {
    CHKERRQ(DMCreateLabel(dm1, "Boundary Faces"));
    CHKERRQ(DMGetLabel(dm1, "Boundary Faces", &bd1));
    CHKERRQ(DMPlexMarkBoundaryFaces(dm1, 13, bd1));
    CHKERRQ(DMCreateLabel(dm1, "Boundary"));
    CHKERRQ(DMGetLabel(dm1, "Boundary", &bd2));
    CHKERRQ(DMPlexMarkBoundaryFaces(dm1, 121, bd2));
    CHKERRQ(DMPlexLabelComplete(dm1, bd2));
  }
  CHKERRQ(PetscObjectSetName((PetscObject) dm1, "First Mesh"));
  CHKERRQ(DMViewFromOptions(dm1, NULL, "-dm_view"));

  CHKERRQ(DMUniversalLabelCreate(dm1, &universal));
  CHKERRQ(DMUniversalLabelGetLabel(universal, &ulabel));
  CHKERRQ(PetscObjectViewFromOptions((PetscObject) ulabel, NULL, "-universal_view"));

  if (!notFile) {
    PetscInt Nl, l;

    CHKERRQ(DMClone(dm1, &dm2));
    CHKERRQ(DMGetNumLabels(dm2, &Nl));
    for (l = Nl-1; l >= 0; --l) {
      PetscBool   isdepth, iscelltype;
      const char *name;

      CHKERRQ(DMGetLabelName(dm2, l, &name));
      CHKERRQ(PetscStrncmp(name, "depth", 6, &isdepth));
      CHKERRQ(PetscStrncmp(name, "celltype", 9, &iscelltype));
      if (!isdepth && !iscelltype) CHKERRQ(DMRemoveLabel(dm2, name, NULL));
    }
  } else {
    CHKERRQ(DMCreate(comm, &dm2));
    CHKERRQ(DMSetType(dm2, DMPLEX));
    CHKERRQ(DMSetFromOptions(dm2));
  }
  CHKERRQ(PetscObjectSetName((PetscObject) dm2, "Second Mesh"));
  CHKERRQ(DMUniversalLabelCreateLabels(universal, PETSC_TRUE, dm2));
  CHKERRQ(DMPlexGetChart(dm2, &pStart, &pEnd));
  for (p = pStart; p < pEnd; ++p) {
    PetscInt val;

    CHKERRQ(DMLabelGetValue(ulabel, p, &val));
    if (val < 0) continue;
    CHKERRQ(DMUniversalLabelSetLabelValue(universal, dm2, PETSC_TRUE, p, val));
  }
  CHKERRQ(DMViewFromOptions(dm2, NULL, "-dm_view"));

  CHKERRQ(DMUniversalLabelDestroy(&universal));
  CHKERRQ(DMDestroy(&dm1));
  CHKERRQ(DMDestroy(&dm2));
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{

  CHKERRQ(PetscInitialize(&argc, &argv, NULL, help));
  /*CHKERRQ(ProcessOptions(PETSC_COMM_WORLD, &user));*/
  CHKERRQ(TestInsertion());
  CHKERRQ(TestEmptyStrata(PETSC_COMM_WORLD));
  CHKERRQ(TestDistribution(PETSC_COMM_WORLD));
  CHKERRQ(TestUniversalLabel(PETSC_COMM_WORLD));
  CHKERRQ(PetscFinalize());
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
