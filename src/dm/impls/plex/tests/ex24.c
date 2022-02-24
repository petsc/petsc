static char help[] = "Test that MatPartitioning and PetscPartitioner interfaces are equivalent when using PETSCPARTITIONERMATPARTITIONING\n\n";
static char FILENAME[] = "ex24.c";

#include <petscdmplex.h>
#include <petscviewerhdf5.h>

#if defined(PETSC_HAVE_PTSCOTCH)
EXTERN_C_BEGIN
#include <ptscotch.h>
EXTERN_C_END
#endif

typedef struct {
  PetscBool compare_is;                   /* Compare ISs and PetscSections */
  PetscBool compare_dm;                   /* Compare DM */
  PetscBool tpw;                          /* Use target partition weights */
  char      partitioning[64];
  char      repartitioning[64];
} AppCtx;

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscBool      repartition = PETSC_TRUE;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  options->compare_is = PETSC_FALSE;
  options->compare_dm = PETSC_FALSE;

  ierr = PetscOptionsBegin(comm, "", "Meshing Interpolation Test Options", "DMPLEX");CHKERRQ(ierr);
  CHKERRQ(PetscOptionsBool("-compare_is", "Compare ISs and PetscSections?", FILENAME, options->compare_is, &options->compare_is, NULL));
  CHKERRQ(PetscOptionsBool("-compare_dm", "Compare DMs?", FILENAME, options->compare_dm, &options->compare_dm, NULL));
  CHKERRQ(PetscStrncpy(options->partitioning,MATPARTITIONINGPARMETIS,sizeof(options->partitioning)));
  CHKERRQ(PetscOptionsString("-partitioning","The mat partitioning type to test","None",options->partitioning, options->partitioning,sizeof(options->partitioning),NULL));
  CHKERRQ(PetscOptionsBool("-repartition", "Partition again after the first partition?", FILENAME, repartition, &repartition, NULL));
  if (repartition) {
    CHKERRQ(PetscStrncpy(options->repartitioning,MATPARTITIONINGPARMETIS,64));
    CHKERRQ(PetscOptionsString("-repartitioning","The mat partitioning type to test (second partitioning)","None", options->repartitioning, options->repartitioning,sizeof(options->repartitioning),NULL));
  } else {
    options->repartitioning[0] = '\0';
  }
  CHKERRQ(PetscOptionsBool("-tpweight", "Use target partition weights", FILENAME, options->tpw, &options->tpw, NULL));
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode ScotchResetRandomSeed()
{
  PetscFunctionBegin;
#if defined(PETSC_HAVE_PTSCOTCH)
  SCOTCH_randomReset();
#endif
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscFunctionBegin;
  CHKERRQ(DMCreate(comm, dm));
  CHKERRQ(DMSetType(*dm, DMPLEX));
  CHKERRQ(DMPlexDistributeSetDefault(*dm, PETSC_FALSE));
  CHKERRQ(DMSetFromOptions(*dm));
  CHKERRQ(DMViewFromOptions(*dm, NULL, "-dm_view"));
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  MPI_Comm       comm;
  DM             dm1, dm2, dmdist1, dmdist2;
  DMPlexInterpolatedFlag interp;
  MatPartitioning mp;
  PetscPartitioner part1, part2;
  AppCtx         user;
  IS             is1=NULL, is2=NULL;
  IS             is1g, is2g;
  PetscSection   s1=NULL, s2=NULL, tpws = NULL;
  PetscInt       i;
  PetscBool      flg;
  PetscErrorCode ierr;
  PetscMPIInt    size;

  ierr = PetscInitialize(&argc, &argv, NULL,help);if (ierr) return ierr;
  comm = PETSC_COMM_WORLD;
  CHKERRMPI(MPI_Comm_size(comm,&size));
  CHKERRQ(ProcessOptions(comm, &user));
  CHKERRQ(CreateMesh(comm, &user, &dm1));
  CHKERRQ(CreateMesh(comm, &user, &dm2));

  if (user.tpw) {
    CHKERRQ(PetscSectionCreate(comm, &tpws));
    CHKERRQ(PetscSectionSetChart(tpws, 0, size));
    for (i=0;i<size;i++) {
      PetscInt tdof = i%2 ? 2*i -1 : i+2;
      CHKERRQ(PetscSectionSetDof(tpws, i, tdof));
    }
    if (size > 1) { /* test zero tpw entry */
      CHKERRQ(PetscSectionSetDof(tpws, 0, 0));
    }
    CHKERRQ(PetscSectionSetUp(tpws));
  }

  /* partition dm1 using PETSCPARTITIONERPARMETIS */
  CHKERRQ(ScotchResetRandomSeed());
  CHKERRQ(DMPlexGetPartitioner(dm1, &part1));
  CHKERRQ(PetscObjectSetOptionsPrefix((PetscObject)part1,"p1_"));
  CHKERRQ(PetscPartitionerSetType(part1, user.partitioning));
  CHKERRQ(PetscPartitionerSetFromOptions(part1));
  CHKERRQ(PetscSectionCreate(comm, &s1));
  CHKERRQ(PetscPartitionerDMPlexPartition(part1, dm1, tpws, s1, &is1));

  /* partition dm2 using PETSCPARTITIONERMATPARTITIONING with MATPARTITIONINGPARMETIS */
  CHKERRQ(ScotchResetRandomSeed());
  CHKERRQ(DMPlexGetPartitioner(dm2, &part2));
  CHKERRQ(PetscObjectSetOptionsPrefix((PetscObject)part2,"p2_"));
  CHKERRQ(PetscPartitionerSetType(part2, PETSCPARTITIONERMATPARTITIONING));
  CHKERRQ(PetscPartitionerMatPartitioningGetMatPartitioning(part2, &mp));
  CHKERRQ(MatPartitioningSetType(mp, user.partitioning));
  CHKERRQ(PetscPartitionerSetFromOptions(part2));
  CHKERRQ(PetscSectionCreate(comm, &s2));
  CHKERRQ(PetscPartitionerDMPlexPartition(part2, dm2, tpws, s2, &is2));

  CHKERRQ(ISOnComm(is1, comm, PETSC_USE_POINTER, &is1g));
  CHKERRQ(ISOnComm(is2, comm, PETSC_USE_POINTER, &is2g));
  CHKERRQ(ISViewFromOptions(is1g, NULL, "-seq_is1_view"));
  CHKERRQ(ISViewFromOptions(is2g, NULL, "-seq_is2_view"));
  /* compare the two ISs */
  if (user.compare_is) {
    CHKERRQ(ISEqualUnsorted(is1g, is2g, &flg));
    if (!flg) CHKERRQ(PetscPrintf(comm, "ISs are not equal with type %s with size %d.\n",user.partitioning,size));
  }
  CHKERRQ(ISDestroy(&is1g));
  CHKERRQ(ISDestroy(&is2g));

  /* compare the two PetscSections */
  CHKERRQ(PetscSectionViewFromOptions(s1, NULL, "-seq_s1_view"));
  CHKERRQ(PetscSectionViewFromOptions(s2, NULL, "-seq_s2_view"));
  if (user.compare_is) {
    CHKERRQ(PetscSectionCompare(s1, s2, &flg));
    if (!flg) CHKERRQ(PetscPrintf(comm, "PetscSections are not equal with %s with size %d.\n",user.partitioning,size));
  }

  /* distribute both DMs */
  CHKERRQ(ScotchResetRandomSeed());
  CHKERRQ(DMPlexDistribute(dm1, 0, NULL, &dmdist1));
  CHKERRQ(ScotchResetRandomSeed());
  CHKERRQ(DMPlexDistribute(dm2, 0, NULL, &dmdist2));

  /* cleanup */
  CHKERRQ(PetscSectionDestroy(&tpws));
  CHKERRQ(PetscSectionDestroy(&s1));
  CHKERRQ(PetscSectionDestroy(&s2));
  CHKERRQ(ISDestroy(&is1));
  CHKERRQ(ISDestroy(&is2));
  CHKERRQ(DMDestroy(&dm1));
  CHKERRQ(DMDestroy(&dm2));

  /* if distributed DMs are NULL (sequential case), then quit */
  if (!dmdist1 && !dmdist2) return ierr;

  CHKERRQ(DMViewFromOptions(dmdist1, NULL, "-dm_dist1_view"));
  CHKERRQ(DMViewFromOptions(dmdist2, NULL, "-dm_dist2_view"));

  /* compare the two distributed DMs */
  if (user.compare_dm) {
    CHKERRQ(DMPlexEqual(dmdist1, dmdist2, &flg));
    if (!flg) CHKERRQ(PetscPrintf(comm, "Distributed DMs are not equal %s with size %d.\n",user.partitioning,size));
  }

  /* if repartitioning is disabled, then quit */
  if (user.repartitioning[0] == '\0') return ierr;

  if (user.tpw) {
    CHKERRQ(PetscSectionCreate(comm, &tpws));
    CHKERRQ(PetscSectionSetChart(tpws, 0, size));
    for (i=0;i<size;i++) {
      PetscInt tdof = i%2 ? i+1 : size - i;
      CHKERRQ(PetscSectionSetDof(tpws, i, tdof));
    }
    CHKERRQ(PetscSectionSetUp(tpws));
  }

  /* repartition distributed DM dmdist1 */
  CHKERRQ(ScotchResetRandomSeed());
  CHKERRQ(DMPlexGetPartitioner(dmdist1, &part1));
  CHKERRQ(PetscObjectSetOptionsPrefix((PetscObject)part1,"dp1_"));
  CHKERRQ(PetscPartitionerSetType(part1, user.repartitioning));
  CHKERRQ(PetscPartitionerSetFromOptions(part1));
  CHKERRQ(PetscSectionCreate(comm, &s1));
  CHKERRQ(PetscPartitionerDMPlexPartition(part1, dmdist1, tpws, s1, &is1));

  /* repartition distributed DM dmdist2 */
  CHKERRQ(ScotchResetRandomSeed());
  CHKERRQ(DMPlexGetPartitioner(dmdist2, &part2));
  CHKERRQ(PetscObjectSetOptionsPrefix((PetscObject)part2,"dp2_"));
  CHKERRQ(PetscPartitionerSetType(part2, PETSCPARTITIONERMATPARTITIONING));
  CHKERRQ(PetscPartitionerMatPartitioningGetMatPartitioning(part2, &mp));
  CHKERRQ(MatPartitioningSetType(mp, user.repartitioning));
  CHKERRQ(PetscPartitionerSetFromOptions(part2));
  CHKERRQ(PetscSectionCreate(comm, &s2));
  CHKERRQ(PetscPartitionerDMPlexPartition(part2, dmdist2, tpws, s2, &is2));

  /* compare the two ISs */
  CHKERRQ(ISOnComm(is1, comm, PETSC_USE_POINTER, &is1g));
  CHKERRQ(ISOnComm(is2, comm, PETSC_USE_POINTER, &is2g));
  CHKERRQ(ISViewFromOptions(is1g, NULL, "-dist_is1_view"));
  CHKERRQ(ISViewFromOptions(is2g, NULL, "-dist_is2_view"));
  if (user.compare_is) {
    CHKERRQ(ISEqualUnsorted(is1g, is2g, &flg));
    if (!flg) CHKERRQ(PetscPrintf(comm, "Distributed ISs are not equal, with %s with size %d.\n",user.repartitioning,size));
  }
  CHKERRQ(ISDestroy(&is1g));
  CHKERRQ(ISDestroy(&is2g));

  /* compare the two PetscSections */
  CHKERRQ(PetscSectionViewFromOptions(s1, NULL, "-dist_s1_view"));
  CHKERRQ(PetscSectionViewFromOptions(s2, NULL, "-dist_s2_view"));
  if (user.compare_is) {
    CHKERRQ(PetscSectionCompare(s1, s2, &flg));
    if (!flg) CHKERRQ(PetscPrintf(comm, "Distributed PetscSections are not equal, with %s with size %d.\n",user.repartitioning,size));
  }

  /* redistribute both distributed DMs */
  CHKERRQ(ScotchResetRandomSeed());
  CHKERRQ(DMPlexDistribute(dmdist1, 0, NULL, &dm1));
  CHKERRQ(ScotchResetRandomSeed());
  CHKERRQ(DMPlexDistribute(dmdist2, 0, NULL, &dm2));

  /* compare the two distributed DMs */
  CHKERRQ(DMPlexIsInterpolated(dm1, &interp));
  if (interp == DMPLEX_INTERPOLATED_NONE) {
    CHKERRQ(DMPlexEqual(dm1, dm2, &flg));
    if (!flg) CHKERRQ(PetscPrintf(comm, "Redistributed DMs are not equal, with %s with size %d.\n",user.repartitioning,size));
  }

  /* cleanup */
  CHKERRQ(PetscSectionDestroy(&tpws));
  CHKERRQ(PetscSectionDestroy(&s1));
  CHKERRQ(PetscSectionDestroy(&s2));
  CHKERRQ(ISDestroy(&is1));
  CHKERRQ(ISDestroy(&is2));
  CHKERRQ(DMDestroy(&dm1));
  CHKERRQ(DMDestroy(&dm2));
  CHKERRQ(DMDestroy(&dmdist1));
  CHKERRQ(DMDestroy(&dmdist2));
  ierr = PetscFinalize();
  return ierr;
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
