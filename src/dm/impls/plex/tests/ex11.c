static char help[] = "Tests for DMLabel\n\n";

#include <petscdmplex.h>
#include <petsc/private/dmimpl.h>

static PetscErrorCode TestInsertion()
{
  DMLabel        label, label2;
  const PetscInt values[5] = {0, 3, 4, -1, 176}, N = 10000;
  PetscInt       i, v;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMLabelCreate(PETSC_COMM_SELF, "Test Label", &label);CHKERRQ(ierr);
  ierr = DMLabelSetDefaultValue(label, -100);CHKERRQ(ierr);
  for (i = 0; i < N; ++i) {
    ierr = DMLabelSetValue(label, i, values[i%5]);CHKERRQ(ierr);
  }
  /* Test get in hash mode */
  for (i = 0; i < N; ++i) {
    PetscInt val;

    ierr = DMLabelGetValue(label, i, &val);CHKERRQ(ierr);
    PetscAssertFalse(val != values[i%5],PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Value %d for point %d should be %d", val, i, values[i%5]);
  }
  /* Test stratum */
  for (v = 0; v < 5; ++v) {
    IS              stratum;
    const PetscInt *points;
    PetscInt        n;

    ierr = DMLabelGetStratumIS(label, values[v], &stratum);CHKERRQ(ierr);
    PetscAssertFalse(!stratum,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Stratum %d is empty!", v);
    ierr = ISGetIndices(stratum, &points);CHKERRQ(ierr);
    ierr = ISGetLocalSize(stratum, &n);CHKERRQ(ierr);
    for (i = 0; i < n; ++i) {
      PetscAssertFalse(points[i] != i*5+v,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Point %d should be %d", points[i], i*5+v);
    }
    ierr = ISRestoreIndices(stratum, &points);CHKERRQ(ierr);
    ierr = ISDestroy(&stratum);CHKERRQ(ierr);
  }
  /* Test get in array mode */
  for (i = 0; i < N; ++i) {
    PetscInt val;

    ierr = DMLabelGetValue(label, i, &val);CHKERRQ(ierr);
    PetscAssertFalse(val != values[i%5],PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Value %d should be %d", val, values[i%5]);
  }
  /* Test Duplicate */
  ierr = DMLabelDuplicate(label, &label2);CHKERRQ(ierr);
  for (i = 0; i < N; ++i) {
    PetscInt val;

    ierr = DMLabelGetValue(label2, i, &val);CHKERRQ(ierr);
    PetscAssertFalse(val != values[i%5],PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Value %d should be %d", val, values[i%5]);
  }
  ierr = DMLabelDestroy(&label2);CHKERRQ(ierr);
  ierr = DMLabelDestroy(&label);CHKERRQ(ierr);
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
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm, &rank);CHKERRMPI(ierr);
  /* A 3D box with two adjacent cells, sharing one face and four vertices */
  ierr = DMCreate(comm, &dm);CHKERRQ(ierr);
  ierr = DMSetType(dm, DMPLEX);CHKERRQ(ierr);
  ierr = DMSetDimension(dm, dim);CHKERRQ(ierr);
  if (rank == 0) {
    ierr = DMPlexSetChart(dm, 0, 25);CHKERRQ(ierr);
    ierr = DMPlexSetConeSize(dm, 0, 6);CHKERRQ(ierr);
    ierr = DMPlexSetConeSize(dm, 1, 6);CHKERRQ(ierr);
    ierr = DMPlexSetConeSize(dm, 2, 4);CHKERRQ(ierr);
    ierr = DMPlexSetConeSize(dm, 3, 4);CHKERRQ(ierr);
    ierr = DMPlexSetConeSize(dm, 4, 4);CHKERRQ(ierr);
    ierr = DMPlexSetConeSize(dm, 5, 4);CHKERRQ(ierr);
    ierr = DMPlexSetConeSize(dm, 6, 4);CHKERRQ(ierr);
    ierr = DMPlexSetConeSize(dm, 7, 4);CHKERRQ(ierr);
    ierr = DMPlexSetConeSize(dm, 8, 4);CHKERRQ(ierr);
    ierr = DMPlexSetConeSize(dm, 9, 4);CHKERRQ(ierr);
    ierr = DMPlexSetConeSize(dm, 10, 4);CHKERRQ(ierr);
    ierr = DMPlexSetConeSize(dm, 11, 4);CHKERRQ(ierr);
    ierr = DMPlexSetConeSize(dm, 12, 4);CHKERRQ(ierr);
  }
  ierr = DMSetUp(dm);CHKERRQ(ierr);
  if (rank == 0) {
    ierr = DMPlexSetCone(dm, 0, c0);CHKERRQ(ierr);
    ierr = DMPlexSetCone(dm, 1, c1);CHKERRQ(ierr);
    ierr = DMPlexSetCone(dm, 2, c2);CHKERRQ(ierr);
    ierr = DMPlexSetCone(dm, 3, c3);CHKERRQ(ierr);
    ierr = DMPlexSetCone(dm, 4, c4);CHKERRQ(ierr);
    ierr = DMPlexSetCone(dm, 5, c5);CHKERRQ(ierr);
    ierr = DMPlexSetCone(dm, 6, c6);CHKERRQ(ierr);
    ierr = DMPlexSetCone(dm, 7, c7);CHKERRQ(ierr);
    ierr = DMPlexSetCone(dm, 8, c8);CHKERRQ(ierr);
    ierr = DMPlexSetCone(dm, 9, c9);CHKERRQ(ierr);
    ierr = DMPlexSetCone(dm, 10, c10);CHKERRQ(ierr);
    ierr = DMPlexSetCone(dm, 11, c11);CHKERRQ(ierr);
    ierr = DMPlexSetCone(dm, 12, c12);CHKERRQ(ierr);
  }
  ierr = DMPlexSymmetrize(dm);CHKERRQ(ierr);
  /* Create a user managed depth label, so that we can leave out edges */
  {
    DMLabel label;
    PetscInt numValues, maxValues = 0, v;

    ierr = DMCreateLabel(dm, "depth");CHKERRQ(ierr);
    ierr = DMPlexGetDepthLabel(dm, &label);CHKERRQ(ierr);
    if (rank == 0) {
      PetscInt i;

      for (i = 0; i < 25; ++i) {
        if (i < 2)       {ierr = DMLabelSetValue(label, i, 3);CHKERRQ(ierr);}
        else if (i < 13) {ierr = DMLabelSetValue(label, i, 2);CHKERRQ(ierr);}
        else             {
          if (i==13) {ierr = DMLabelAddStratum(label, 1);CHKERRQ(ierr);}
          ierr = DMLabelSetValue(label, i, 0);CHKERRQ(ierr);
        }
      }
    }
    ierr = DMLabelGetNumValues(label, &numValues);CHKERRQ(ierr);
    ierr = MPI_Allreduce(&numValues, &maxValues, 1, MPIU_INT, MPI_MAX, PetscObjectComm((PetscObject) dm));CHKERRMPI(ierr);
    for (v = numValues; v < maxValues; ++v) {ierr = DMLabelAddStratum(label,v);CHKERRQ(ierr);}
  }
  {
    DMLabel label;
    ierr = DMPlexGetDepthLabel(dm, &label);CHKERRQ(ierr);
    ierr = DMLabelView(label, PETSC_VIEWER_STDOUT_(comm));CHKERRQ(ierr);
  }
  ierr = DMPlexGetPartitioner(dm,&part);CHKERRQ(ierr);
  ierr = PetscPartitionerSetFromOptions(part);CHKERRQ(ierr);
  ierr = DMPlexDistribute(dm, 1, NULL, &dmDist);CHKERRQ(ierr);
  if (dmDist) {
    ierr = DMDestroy(&dm);CHKERRQ(ierr);
    dm   = dmDist;
  }
  {
    DMLabel label;
    ierr = DMPlexGetDepthLabel(dm, &label);CHKERRQ(ierr);
    ierr = DMLabelView(label, PETSC_VIEWER_STDOUT_(comm));CHKERRQ(ierr);
  }
  /* Create a cell vector */
  {
    Vec          v;
    PetscSection s;
    PetscInt     numComp[] = {1};
    PetscInt     dof[]     = {0,0,0,1};
    PetscInt     N;

    ierr = DMSetNumFields(dm, 1);CHKERRQ(ierr);
    ierr = DMPlexCreateSection(dm, NULL, numComp, dof, 0, NULL, NULL, NULL, NULL, &s);CHKERRQ(ierr);
    ierr = DMSetLocalSection(dm, s);CHKERRQ(ierr);
    ierr = PetscSectionDestroy(&s);CHKERRQ(ierr);
    ierr = DMCreateGlobalVector(dm, &v);CHKERRQ(ierr);
    ierr = VecGetSize(v, &N);CHKERRQ(ierr);
    if (N != 2) {
      ierr = DMView(dm, PETSC_VIEWER_STDOUT_(comm));CHKERRQ(ierr);
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "FAIL: Vector size %d != 2", N);
    }
    ierr = VecDestroy(&v);CHKERRQ(ierr);
  }
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
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
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm, &rank);CHKERRMPI(ierr);
  ierr = PetscOptionsGetString(NULL, NULL, "-filename", filename, sizeof(filename), &flg);CHKERRQ(ierr);
  if (!flg) PetscFunctionReturn(0);
  ierr = PetscOptionsGetInt(NULL, NULL, "-overlap", &overlap, NULL);CHKERRQ(ierr);
  ierr = DMPlexCreateFromFile(comm, filename, "ex11_plex", PETSC_TRUE, &dm);CHKERRQ(ierr);
  ierr = DMSetBasicAdjacency(dm, PETSC_TRUE, PETSC_FALSE);CHKERRQ(ierr);
  ierr = DMCreateLabel(dm, name);CHKERRQ(ierr);
  ierr = DMGetLabel(dm, name, &label);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  for (c = cStart; c < cEnd; ++c) {
    ierr = DMLabelSetValue(label, c, c);CHKERRQ(ierr);
  }
  ierr = DMLabelView(label, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = DMPlexGetPartitioner(dm,&part);CHKERRQ(ierr);
  ierr = PetscPartitionerSetFromOptions(part);CHKERRQ(ierr);
  ierr = DMPlexDistribute(dm, overlap, NULL, &dmDist);CHKERRQ(ierr);
  if (dmDist) {
    ierr = DMDestroy(&dm);CHKERRQ(ierr);
    dm   = dmDist;
  }
  ierr = PetscObjectSetName((PetscObject) dm, "Mesh");CHKERRQ(ierr);
  ierr = DMViewFromOptions(dm, NULL, "-dm_view");CHKERRQ(ierr);
  ierr = DMGetLabel(dm, name, &label);CHKERRQ(ierr);
  ierr = DMLabelView(label, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TestUniversalLabel(MPI_Comm comm)
{
  DM               dm1, dm2;
  DMLabel          bd1, bd2, ulabel;
  DMUniversalLabel universal;
  PetscInt         pStart, pEnd, p;
  PetscBool        run = PETSC_FALSE, notFile;
  PetscErrorCode   ierr;

  PetscFunctionBeginUser;
  ierr = PetscOptionsGetBool(NULL, NULL, "-universal", &run, NULL);CHKERRQ(ierr);
  if (!run) PetscFunctionReturn(0);

  char filename[PETSC_MAX_PATH_LEN];
  PetscBool flg;

  ierr = PetscOptionsGetString(NULL, NULL, "-filename", filename, sizeof(filename), &flg);CHKERRQ(ierr);
  if (flg) {
    ierr = DMPlexCreateFromFile(comm, filename, "ex11_plex", PETSC_TRUE, &dm1);CHKERRQ(ierr);
  } else {
    ierr = DMCreate(comm, &dm1);CHKERRQ(ierr);
    ierr = DMSetType(dm1, DMPLEX);CHKERRQ(ierr);
    ierr = DMSetFromOptions(dm1);CHKERRQ(ierr);
  }
  ierr = DMHasLabel(dm1, "marker", &notFile);CHKERRQ(ierr);
  if (notFile) {
    ierr = DMCreateLabel(dm1, "Boundary Faces");CHKERRQ(ierr);
    ierr = DMGetLabel(dm1, "Boundary Faces", &bd1);CHKERRQ(ierr);
    ierr = DMPlexMarkBoundaryFaces(dm1, 13, bd1);CHKERRQ(ierr);
    ierr = DMCreateLabel(dm1, "Boundary");CHKERRQ(ierr);
    ierr = DMGetLabel(dm1, "Boundary", &bd2);CHKERRQ(ierr);
    ierr = DMPlexMarkBoundaryFaces(dm1, 121, bd2);CHKERRQ(ierr);
    ierr = DMPlexLabelComplete(dm1, bd2);CHKERRQ(ierr);
  }
  ierr = PetscObjectSetName((PetscObject) dm1, "First Mesh");CHKERRQ(ierr);
  ierr = DMViewFromOptions(dm1, NULL, "-dm_view");CHKERRQ(ierr);

  ierr = DMUniversalLabelCreate(dm1, &universal);CHKERRQ(ierr);
  ierr = DMUniversalLabelGetLabel(universal, &ulabel);CHKERRQ(ierr);
  ierr = PetscObjectViewFromOptions((PetscObject) ulabel, NULL, "-universal_view");CHKERRQ(ierr);

  if (!notFile) {
    PetscInt Nl, l;

    ierr = DMClone(dm1, &dm2);CHKERRQ(ierr);
    ierr = DMGetNumLabels(dm2, &Nl);CHKERRQ(ierr);
    for (l = Nl-1; l >= 0; --l) {
      PetscBool   isdepth, iscelltype;
      const char *name;

      ierr = DMGetLabelName(dm2, l, &name);CHKERRQ(ierr);
      ierr = PetscStrncmp(name, "depth", 6, &isdepth);CHKERRQ(ierr);
      ierr = PetscStrncmp(name, "celltype", 9, &iscelltype);CHKERRQ(ierr);
      if (!isdepth && !iscelltype) {ierr = DMRemoveLabel(dm2, name, NULL);CHKERRQ(ierr);}
    }
  } else {
    ierr = DMCreate(comm, &dm2);CHKERRQ(ierr);
    ierr = DMSetType(dm2, DMPLEX);CHKERRQ(ierr);
    ierr = DMSetFromOptions(dm2);CHKERRQ(ierr);
  }
  ierr = PetscObjectSetName((PetscObject) dm2, "Second Mesh");CHKERRQ(ierr);
  ierr = DMUniversalLabelCreateLabels(universal, PETSC_TRUE, dm2);CHKERRQ(ierr);
  ierr = DMPlexGetChart(dm2, &pStart, &pEnd);CHKERRQ(ierr);
  for (p = pStart; p < pEnd; ++p) {
    PetscInt val;

    ierr = DMLabelGetValue(ulabel, p, &val);CHKERRQ(ierr);
    if (val < 0) continue;
    ierr = DMUniversalLabelSetLabelValue(universal, dm2, PETSC_TRUE, p, val);CHKERRQ(ierr);
  }
  ierr = DMViewFromOptions(dm2, NULL, "-dm_view");CHKERRQ(ierr);

  ierr = DMUniversalLabelDestroy(&universal);CHKERRQ(ierr);
  ierr = DMDestroy(&dm1);CHKERRQ(ierr);
  ierr = DMDestroy(&dm2);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help);if (ierr) return ierr;
  /*ierr = ProcessOptions(PETSC_COMM_WORLD, &user);CHKERRQ(ierr);*/
  ierr = TestInsertion();CHKERRQ(ierr);
  ierr = TestEmptyStrata(PETSC_COMM_WORLD);CHKERRQ(ierr);
  ierr = TestDistribution(PETSC_COMM_WORLD);CHKERRQ(ierr);
  ierr = TestUniversalLabel(PETSC_COMM_WORLD);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
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
