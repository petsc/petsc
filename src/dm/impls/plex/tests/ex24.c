static char help[]     = "Test that MatPartitioning and PetscPartitioner interfaces are equivalent when using PETSCPARTITIONERMATPARTITIONING\n\n";
static char FILENAME[] = "ex24.c";

#include <petscdmplex.h>
#include <petscviewerhdf5.h>

#if defined(PETSC_HAVE_PTSCOTCH)
EXTERN_C_BEGIN
  #include <ptscotch.h>
EXTERN_C_END
#endif

typedef struct {
  PetscBool compare_is; /* Compare ISs and PetscSections */
  PetscBool compare_dm; /* Compare DM */
  PetscBool tpw;        /* Use target partition weights */
  char      partitioning[64];
  char      repartitioning[64];
} AppCtx;

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscBool repartition = PETSC_TRUE;

  PetscFunctionBegin;
  options->compare_is = PETSC_FALSE;
  options->compare_dm = PETSC_FALSE;

  PetscOptionsBegin(comm, "", "Meshing Interpolation Test Options", "DMPLEX");
  PetscCall(PetscOptionsBool("-compare_is", "Compare ISs and PetscSections?", FILENAME, options->compare_is, &options->compare_is, NULL));
  PetscCall(PetscOptionsBool("-compare_dm", "Compare DMs?", FILENAME, options->compare_dm, &options->compare_dm, NULL));
  PetscCall(PetscStrncpy(options->partitioning, MATPARTITIONINGPARMETIS, sizeof(options->partitioning)));
  PetscCall(PetscOptionsString("-partitioning", "The mat partitioning type to test", "None", options->partitioning, options->partitioning, sizeof(options->partitioning), NULL));
  PetscCall(PetscOptionsBool("-repartition", "Partition again after the first partition?", FILENAME, repartition, &repartition, NULL));
  if (repartition) {
    PetscCall(PetscStrncpy(options->repartitioning, MATPARTITIONINGPARMETIS, 64));
    PetscCall(PetscOptionsString("-repartitioning", "The mat partitioning type to test (second partitioning)", "None", options->repartitioning, options->repartitioning, sizeof(options->repartitioning), NULL));
  } else {
    options->repartitioning[0] = '\0';
  }
  PetscCall(PetscOptionsBool("-tpweight", "Use target partition weights", FILENAME, options->tpw, &options->tpw, NULL));
  PetscOptionsEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ScotchResetRandomSeed()
{
  PetscFunctionBegin;
#if defined(PETSC_HAVE_PTSCOTCH)
  SCOTCH_randomReset();
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscFunctionBegin;
  PetscCall(DMCreate(comm, dm));
  PetscCall(DMSetType(*dm, DMPLEX));
  PetscCall(DMPlexDistributeSetDefault(*dm, PETSC_FALSE));
  PetscCall(DMSetFromOptions(*dm));
  PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  MPI_Comm               comm;
  DM                     dm1, dm2, dmdist1, dmdist2;
  DMPlexInterpolatedFlag interp;
  MatPartitioning        mp;
  PetscPartitioner       part1, part2;
  AppCtx                 user;
  IS                     is1 = NULL, is2 = NULL;
  IS                     is1g, is2g;
  PetscSection           s1 = NULL, s2 = NULL, tpws = NULL;
  PetscInt               i;
  PetscBool              flg;
  PetscMPIInt            size;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  comm = PETSC_COMM_WORLD;
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCall(ProcessOptions(comm, &user));
  PetscCall(CreateMesh(comm, &user, &dm1));
  PetscCall(CreateMesh(comm, &user, &dm2));

  if (user.tpw) {
    PetscCall(PetscSectionCreate(comm, &tpws));
    PetscCall(PetscSectionSetChart(tpws, 0, size));
    for (i = 0; i < size; i++) {
      PetscInt tdof = i % 2 ? 2 * i - 1 : i + 2;
      PetscCall(PetscSectionSetDof(tpws, i, tdof));
    }
    if (size > 1) { /* test zero tpw entry */
      PetscCall(PetscSectionSetDof(tpws, 0, 0));
    }
    PetscCall(PetscSectionSetUp(tpws));
  }

  /* partition dm1 using PETSCPARTITIONERPARMETIS */
  PetscCall(ScotchResetRandomSeed());
  PetscCall(DMPlexGetPartitioner(dm1, &part1));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)part1, "p1_"));
  PetscCall(PetscPartitionerSetType(part1, user.partitioning));
  PetscCall(PetscPartitionerSetFromOptions(part1));
  PetscCall(PetscSectionCreate(comm, &s1));
  PetscCall(PetscPartitionerDMPlexPartition(part1, dm1, tpws, s1, &is1));

  /* partition dm2 using PETSCPARTITIONERMATPARTITIONING with MATPARTITIONINGPARMETIS */
  PetscCall(ScotchResetRandomSeed());
  PetscCall(DMPlexGetPartitioner(dm2, &part2));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)part2, "p2_"));
  PetscCall(PetscPartitionerSetType(part2, PETSCPARTITIONERMATPARTITIONING));
  PetscCall(PetscPartitionerMatPartitioningGetMatPartitioning(part2, &mp));
  PetscCall(MatPartitioningSetType(mp, user.partitioning));
  PetscCall(PetscPartitionerSetFromOptions(part2));
  PetscCall(PetscSectionCreate(comm, &s2));
  PetscCall(PetscPartitionerDMPlexPartition(part2, dm2, tpws, s2, &is2));

  PetscCall(ISOnComm(is1, comm, PETSC_USE_POINTER, &is1g));
  PetscCall(ISOnComm(is2, comm, PETSC_USE_POINTER, &is2g));
  PetscCall(ISViewFromOptions(is1g, NULL, "-seq_is1_view"));
  PetscCall(ISViewFromOptions(is2g, NULL, "-seq_is2_view"));
  /* compare the two ISs */
  if (user.compare_is) {
    PetscCall(ISEqualUnsorted(is1g, is2g, &flg));
    if (!flg) PetscCall(PetscPrintf(comm, "ISs are not equal with type %s with size %d.\n", user.partitioning, size));
  }
  PetscCall(ISDestroy(&is1g));
  PetscCall(ISDestroy(&is2g));

  /* compare the two PetscSections */
  PetscCall(PetscSectionViewFromOptions(s1, NULL, "-seq_s1_view"));
  PetscCall(PetscSectionViewFromOptions(s2, NULL, "-seq_s2_view"));
  if (user.compare_is) {
    PetscCall(PetscSectionCompare(s1, s2, &flg));
    if (!flg) PetscCall(PetscPrintf(comm, "PetscSections are not equal with %s with size %d.\n", user.partitioning, size));
  }

  /* distribute both DMs */
  PetscCall(ScotchResetRandomSeed());
  PetscCall(DMPlexDistribute(dm1, 0, NULL, &dmdist1));
  PetscCall(ScotchResetRandomSeed());
  PetscCall(DMPlexDistribute(dm2, 0, NULL, &dmdist2));

  /* cleanup */
  PetscCall(PetscSectionDestroy(&tpws));
  PetscCall(PetscSectionDestroy(&s1));
  PetscCall(PetscSectionDestroy(&s2));
  PetscCall(ISDestroy(&is1));
  PetscCall(ISDestroy(&is2));
  PetscCall(DMDestroy(&dm1));
  PetscCall(DMDestroy(&dm2));

  /* if distributed DMs are NULL (sequential case), then quit */
  if (!dmdist1 && !dmdist2) return 0;

  PetscCall(DMViewFromOptions(dmdist1, NULL, "-dm_dist1_view"));
  PetscCall(DMViewFromOptions(dmdist2, NULL, "-dm_dist2_view"));

  /* compare the two distributed DMs */
  if (user.compare_dm) {
    PetscCall(DMPlexEqual(dmdist1, dmdist2, &flg));
    if (!flg) PetscCall(PetscPrintf(comm, "Distributed DMs are not equal %s with size %d.\n", user.partitioning, size));
  }

  /* if repartitioning is disabled, then quit */
  if (user.repartitioning[0] == '\0') return 0;

  if (user.tpw) {
    PetscCall(PetscSectionCreate(comm, &tpws));
    PetscCall(PetscSectionSetChart(tpws, 0, size));
    for (i = 0; i < size; i++) {
      PetscInt tdof = i % 2 ? i + 1 : size - i;
      PetscCall(PetscSectionSetDof(tpws, i, tdof));
    }
    PetscCall(PetscSectionSetUp(tpws));
  }

  /* repartition distributed DM dmdist1 */
  PetscCall(ScotchResetRandomSeed());
  PetscCall(DMPlexGetPartitioner(dmdist1, &part1));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)part1, "dp1_"));
  PetscCall(PetscPartitionerSetType(part1, user.repartitioning));
  PetscCall(PetscPartitionerSetFromOptions(part1));
  PetscCall(PetscSectionCreate(comm, &s1));
  PetscCall(PetscPartitionerDMPlexPartition(part1, dmdist1, tpws, s1, &is1));

  /* repartition distributed DM dmdist2 */
  PetscCall(ScotchResetRandomSeed());
  PetscCall(DMPlexGetPartitioner(dmdist2, &part2));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)part2, "dp2_"));
  PetscCall(PetscPartitionerSetType(part2, PETSCPARTITIONERMATPARTITIONING));
  PetscCall(PetscPartitionerMatPartitioningGetMatPartitioning(part2, &mp));
  PetscCall(MatPartitioningSetType(mp, user.repartitioning));
  PetscCall(PetscPartitionerSetFromOptions(part2));
  PetscCall(PetscSectionCreate(comm, &s2));
  PetscCall(PetscPartitionerDMPlexPartition(part2, dmdist2, tpws, s2, &is2));

  /* compare the two ISs */
  PetscCall(ISOnComm(is1, comm, PETSC_USE_POINTER, &is1g));
  PetscCall(ISOnComm(is2, comm, PETSC_USE_POINTER, &is2g));
  PetscCall(ISViewFromOptions(is1g, NULL, "-dist_is1_view"));
  PetscCall(ISViewFromOptions(is2g, NULL, "-dist_is2_view"));
  if (user.compare_is) {
    PetscCall(ISEqualUnsorted(is1g, is2g, &flg));
    if (!flg) PetscCall(PetscPrintf(comm, "Distributed ISs are not equal, with %s with size %d.\n", user.repartitioning, size));
  }
  PetscCall(ISDestroy(&is1g));
  PetscCall(ISDestroy(&is2g));

  /* compare the two PetscSections */
  PetscCall(PetscSectionViewFromOptions(s1, NULL, "-dist_s1_view"));
  PetscCall(PetscSectionViewFromOptions(s2, NULL, "-dist_s2_view"));
  if (user.compare_is) {
    PetscCall(PetscSectionCompare(s1, s2, &flg));
    if (!flg) PetscCall(PetscPrintf(comm, "Distributed PetscSections are not equal, with %s with size %d.\n", user.repartitioning, size));
  }

  /* redistribute both distributed DMs */
  PetscCall(ScotchResetRandomSeed());
  PetscCall(DMPlexDistribute(dmdist1, 0, NULL, &dm1));
  PetscCall(ScotchResetRandomSeed());
  PetscCall(DMPlexDistribute(dmdist2, 0, NULL, &dm2));

  /* compare the two distributed DMs */
  PetscCall(DMPlexIsInterpolated(dm1, &interp));
  if (interp == DMPLEX_INTERPOLATED_NONE) {
    PetscCall(DMPlexEqual(dm1, dm2, &flg));
    if (!flg) PetscCall(PetscPrintf(comm, "Redistributed DMs are not equal, with %s with size %d.\n", user.repartitioning, size));
  }

  /* cleanup */
  PetscCall(PetscSectionDestroy(&tpws));
  PetscCall(PetscSectionDestroy(&s1));
  PetscCall(PetscSectionDestroy(&s2));
  PetscCall(ISDestroy(&is1));
  PetscCall(ISDestroy(&is2));
  PetscCall(DMDestroy(&dm1));
  PetscCall(DMDestroy(&dm2));
  PetscCall(DMDestroy(&dmdist1));
  PetscCall(DMDestroy(&dmdist2));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    # partition sequential mesh loaded from Exodus file
    suffix: 0
    nsize: {{1 2 3 4 8}}
    requires: chaco parmetis ptscotch exodusii
    args: -dm_plex_filename ${PETSC_DIR}/share/petsc/datafiles/meshes/blockcylinder-50.exo
    args: -partitioning {{chaco parmetis ptscotch}} -repartitioning {{parmetis ptscotch}} -tpweight {{0 1}}
  test:
    # repartition mesh already partitioned naively by MED loader
    suffix: 1
    nsize: {{1 2 3 4 8}}
    TODO: MED
    requires: parmetis ptscotch med
    args: -dm_plex_filename ${PETSC_DIR}/share/petsc/datafiles/meshes/cylinder.med
    args: -repartition 0 -partitioning {{parmetis ptscotch}}
  test:
    # partition mesh generated by ctetgen using scotch, then repartition with scotch, diff view
    suffix: 3
    nsize: 4
    requires: ptscotch ctetgen
    args: -dm_plex_dim 3 -dm_plex_box_faces 2,3,2 -partitioning ptscotch -repartitioning ptscotch
    args: -p1_petscpartitioner_view -p2_petscpartitioner_view -dp1_petscpartitioner_view -dp2_petscpartitioner_view -tpweight {{0 1}}
  test:
    # partition mesh generated by ctetgen using partitioners supported both by MatPartitioning and PetscPartitioner
    suffix: 4
    nsize: {{1 2 3 4 8}}
    requires: chaco parmetis ptscotch ctetgen
    args: -dm_plex_dim 3 -dm_plex_box_faces {{2,3,4  5,4,3  7,11,5}} -partitioning {{chaco parmetis ptscotch}} -repartitioning {{parmetis ptscotch}} -tpweight {{0 1}}

TEST*/
