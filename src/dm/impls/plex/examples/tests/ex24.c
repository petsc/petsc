static char help[] = "Test that MatPartitioning and PetscPartitioner interfaces to parmetis are equivalent - using PETSCPARTITIONERMATPARTITIONING\n\n";
static char FILENAME[] = "ex24.c";

#include <petscdmplex.h>
#include <petscviewerhdf5.h>

#if defined(PETSC_HAVE_PTSCOTCH)
EXTERN_C_BEGIN
#include <ptscotch.h>
EXTERN_C_END
#endif

typedef struct {
  PetscInt  dim;                          /* The topological mesh dimension */
  PetscInt  faces[3];                     /* Number of faces per dimension */
  PetscBool simplex;                      /* Use simplices or hexes */
  PetscBool interpolate;                  /* Interpolate mesh */
  PetscBool compare_is;                   /* Compare ISs and PetscSections */
  char      filename[PETSC_MAX_PATH_LEN]; /* Import mesh from file */
  char      partitioning[64];
  char      repartitioning[64];
} AppCtx;

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscInt dim;
  PetscBool repartition = PETSC_TRUE;
  PetscBool flg = PETSC_TRUE;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  options->compare_is   = PETSC_FALSE;
  options->dim          = 3;
  options->simplex      = PETSC_TRUE;
  options->interpolate  = PETSC_FALSE;
  options->filename[0]  = '\0';
  ierr = PetscOptionsBegin(comm, "", "Meshing Interpolation Test Options", "DMPLEX");CHKERRQ(ierr);
  ierr = PetscOptionsBool("-compare_is", "Compare ISs and PetscSections?", FILENAME, options->compare_is, &options->compare_is, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsRangeInt("-dim", "The topological mesh dimension", FILENAME, options->dim, &options->dim, NULL,1,3);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-simplex", "Use simplices if true, otherwise hexes", FILENAME, options->simplex, &options->simplex, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-interpolate", "Interpolate the mesh", FILENAME, options->interpolate, &options->interpolate, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsString("-filename", "The mesh file", FILENAME, options->filename, options->filename, PETSC_MAX_PATH_LEN, NULL);CHKERRQ(ierr);
  options->faces[0] = 1; options->faces[1] = 1; options->faces[2] = 1;
  dim = options->dim;
  ierr = PetscOptionsIntArray("-faces", "Number of faces per dimension", FILENAME, options->faces, &dim, NULL);CHKERRQ(ierr);
  ierr = PetscStrncpy(options->partitioning,MATPARTITIONINGPARMETIS,64);CHKERRQ(ierr);
  ierr = PetscOptionsString("-partitioning","The mat partitioning type to test","None",options->partitioning, options->partitioning,64,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-repartition", "Partition again after the first partition?", FILENAME, repartition, &repartition, NULL);CHKERRQ(ierr);
  if (repartition) {
    ierr = PetscStrncpy(options->repartitioning,MATPARTITIONINGPARMETIS,64);CHKERRQ(ierr);
    ierr = PetscOptionsString("-repartitioning","The mat partitioning type to test (second partitioning)","None", options->repartitioning, options->repartitioning,64,NULL);CHKERRQ(ierr);
  } else {
    options->repartitioning[0] = '\0';
  }
  if (dim) options->dim = dim;
  /* Currently this is here only to always consume this option. It is actually queried in PTScotch_PartGraph_{Seq,MPI} functions. */
  ierr = PetscOptionsBool("-petscpartititoner_ptscotch_vertex_weight", "Should PetscPartitionerPTScotch specify the vertex  weights to Scotch (0 to yield the same results as MatPartitioningPTScotch)?",  FILENAME, flg, &flg, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-mat_partitioning_ptscotch_proc_weight",    "Should MatPartitioningPTScotch  specify the process weights to Scotch (0 to yield the same results as PetscPartitionerPTScotch)?", FILENAME, flg, &flg, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();
  PetscFunctionReturn(0);
}

static PetscErrorCode ScotchResetRandomSeed()
{
#if defined(PETSC_HAVE_PTSCOTCH)
  SCOTCH_randomReset();
#endif
  PetscFunctionReturn(0);
}


static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscInt       dim          = user->dim;
  PetscInt      *faces        = user->faces;
  PetscBool      simplex      = user->simplex;
  PetscBool      interpolate  = user->interpolate;
  const char    *filename     = user->filename;
  size_t         len;
  PetscMPIInt    rank;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = PetscStrlen(filename, &len);CHKERRQ(ierr);
  if (len) {
    ierr = DMPlexCreateFromFile(comm, filename, interpolate, dm);CHKERRQ(ierr);
    ierr = DMGetDimension(*dm, &user->dim);CHKERRQ(ierr);
  } else {
    ierr = DMPlexCreateBoxMesh(comm, dim, simplex, faces, NULL, NULL, NULL, interpolate, dm);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  MPI_Comm       comm;
  DM             dm1, dm2, dmdist1, dmdist2;
  MatPartitioning mp;
  PetscPartitioner part1, part2;
  AppCtx         user;
  IS             is1=NULL, is2=NULL;
  PetscSection   s1=NULL, s2=NULL;
  PetscBool      flg;
  PetscErrorCode ierr;
  PetscMPIInt    size;

  ierr = PetscInitialize(&argc, &argv, NULL,help);if (ierr) return ierr;
  comm = PETSC_COMM_WORLD;
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = ProcessOptions(comm, &user);CHKERRQ(ierr);
  ierr = CreateMesh(comm, &user, &dm1);CHKERRQ(ierr);
  ierr = CreateMesh(comm, &user, &dm2);CHKERRQ(ierr);

  /* partition dm1 using PETSCPARTITIONERPARMETIS */
  ierr = ScotchResetRandomSeed();CHKERRQ(ierr);
  ierr = DMPlexGetPartitioner(dm1, &part1);CHKERRQ(ierr);
  ierr = PetscObjectSetOptionsPrefix((PetscObject)part1,"p1_");CHKERRQ(ierr);
  ierr = PetscPartitionerSetType(part1, user.partitioning);CHKERRQ(ierr);
  ierr = PetscPartitionerSetFromOptions(part1);CHKERRQ(ierr);
  ierr = PetscSectionCreate(comm, &s1);CHKERRQ(ierr);
  ierr = PetscPartitionerPartition(part1, dm1, s1, &is1);CHKERRQ(ierr);

  /* partition dm2 using PETSCPARTITIONERMATPARTITIONING with MATPARTITIONINGPARMETIS */
  ierr = ScotchResetRandomSeed();CHKERRQ(ierr);
  ierr = DMPlexGetPartitioner(dm2, &part2);CHKERRQ(ierr);
  ierr = PetscObjectSetOptionsPrefix((PetscObject)part2,"p2_");CHKERRQ(ierr);
  ierr = PetscPartitionerSetType(part2, PETSCPARTITIONERMATPARTITIONING);CHKERRQ(ierr);
  ierr = PetscPartitionerMatPartitioningGetMatPartitioning(part2, &mp);CHKERRQ(ierr);
  ierr = MatPartitioningSetType(mp, user.partitioning);CHKERRQ(ierr);
  ierr = PetscPartitionerSetFromOptions(part2);CHKERRQ(ierr);
  ierr = PetscSectionCreate(comm, &s2);CHKERRQ(ierr);
  ierr = PetscPartitionerPartition(part2, dm2, s2, &is2);CHKERRQ(ierr);

  /* compare the two ISs */
  if (user.compare_is) {
    IS is1g, is2g;
    ierr = ISOnComm(is1, comm, PETSC_USE_POINTER, &is1g);CHKERRQ(ierr);
    ierr = ISOnComm(is2, comm, PETSC_USE_POINTER, &is2g);CHKERRQ(ierr);
    ierr = ISEqualUnsorted(is1g, is2g, &flg);CHKERRQ(ierr);
    if (!flg) PetscPrintf(comm, "ISs are not equal with type %s with size %d.\n",user.partitioning,size);
    ierr = ISDestroy(&is1g);CHKERRQ(ierr);
    ierr = ISDestroy(&is2g);CHKERRQ(ierr);
  }

  /* compare the two PetscSections */
  if (user.compare_is) {
    ierr = PetscSectionCompare(s1, s2, &flg);CHKERRQ(ierr);
    if (!flg) PetscPrintf(comm, "PetscSections are not equal with %s with size %d.\n",user.partitioning,size);
  }

  /* distribute both DMs */
  ierr = ScotchResetRandomSeed();CHKERRQ(ierr);
  ierr = DMPlexDistribute(dm1, 0, NULL, &dmdist1);CHKERRQ(ierr);
  ierr = ScotchResetRandomSeed();CHKERRQ(ierr);
  ierr = DMPlexDistribute(dm2, 0, NULL, &dmdist2);CHKERRQ(ierr);

  /* cleanup */
  ierr = PetscSectionDestroy(&s1);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&s2);CHKERRQ(ierr);
  ierr = ISDestroy(&is1);CHKERRQ(ierr);
  ierr = ISDestroy(&is2);CHKERRQ(ierr);
  ierr = DMDestroy(&dm1);CHKERRQ(ierr);
  ierr = DMDestroy(&dm2);CHKERRQ(ierr);

  /* if distributed DMs are NULL (sequential case), then quit */
  if (!dmdist1 && !dmdist2) return ierr;

  /* compare the two distributed DMs */
  if (!user.interpolate) {
    ierr = DMPlexEqual(dmdist1, dmdist2, &flg);CHKERRQ(ierr);
    if (!flg) PetscPrintf(comm, "Distributed DMs are not equal %s with size %d.\n",user.partitioning,size);
  }

  /* if repartitioning is disabled, then quit */
  if (user.repartitioning[0] == '\0') return ierr;

  /* repartition distributed DM dmdist1 using PETSCPARTITIONERPARMETIS */
  ierr = ScotchResetRandomSeed();CHKERRQ(ierr);
  ierr = DMPlexGetPartitioner(dmdist1, &part1);CHKERRQ(ierr);
  ierr = PetscObjectSetOptionsPrefix((PetscObject)part1,"dp1_");CHKERRQ(ierr);
  ierr = PetscPartitionerSetType(part1, user.repartitioning);CHKERRQ(ierr);
  ierr = PetscPartitionerSetFromOptions(part1);CHKERRQ(ierr);
  ierr = PetscSectionCreate(comm, &s1);CHKERRQ(ierr);
  ierr = PetscPartitionerPartition(part1, dmdist1, s1, &is1);CHKERRQ(ierr);

  /* repartition distributed DM dmdist2 using PETSCPARTITIONERMATPARTITIONING with MATPARTITIONINGPARMETIS */
  ierr = ScotchResetRandomSeed();CHKERRQ(ierr);
  ierr = DMPlexGetPartitioner(dmdist2, &part2);CHKERRQ(ierr);
  ierr = PetscObjectSetOptionsPrefix((PetscObject)part2,"dp2_");CHKERRQ(ierr);
  ierr = PetscPartitionerSetType(part2, PETSCPARTITIONERMATPARTITIONING);CHKERRQ(ierr);
  ierr = PetscPartitionerMatPartitioningGetMatPartitioning(part2, &mp);CHKERRQ(ierr);
  ierr = MatPartitioningSetType(mp, user.repartitioning);CHKERRQ(ierr);
  ierr = PetscPartitionerSetFromOptions(part2);CHKERRQ(ierr);
  ierr = PetscSectionCreate(comm, &s2);CHKERRQ(ierr);
  ierr = PetscPartitionerPartition(part2, dmdist2, s2, &is2);CHKERRQ(ierr);

  /* compare the two ISs */
  if (user.compare_is) {
    IS is1g, is2g;
    ierr = ISOnComm(is1, comm, PETSC_USE_POINTER, &is1g);CHKERRQ(ierr);
    ierr = ISOnComm(is2, comm, PETSC_USE_POINTER, &is2g);CHKERRQ(ierr);
    ierr = ISEqualUnsorted(is1g, is2g, &flg);CHKERRQ(ierr);
    if (!flg) PetscPrintf(comm, "Distributed ISs are not equal, with %s with size %d.\n",user.repartitioning,size);
    ierr = ISDestroy(&is1g);CHKERRQ(ierr);
    ierr = ISDestroy(&is2g);CHKERRQ(ierr);
  }

  /* compare the two PetscSections */
  if (user.compare_is) {
    ierr = PetscSectionCompare(s1, s2, &flg);CHKERRQ(ierr);
    if (!flg) PetscPrintf(comm, "Distributed PetscSections are not equal, with %s with size %d.\n",user.repartitioning,size);
  }

  /* redistribute both distributed DMs */
  ierr = ScotchResetRandomSeed();CHKERRQ(ierr);
  ierr = DMPlexDistribute(dmdist1, 0, NULL, &dm1);CHKERRQ(ierr);
  ierr = ScotchResetRandomSeed();CHKERRQ(ierr);
  ierr = DMPlexDistribute(dmdist2, 0, NULL, &dm2);CHKERRQ(ierr);

  /* compare the two distributed DMs */
  if (!user.interpolate) {
    ierr = DMPlexEqual(dm1, dm2, &flg);CHKERRQ(ierr);
    if (!flg) PetscPrintf(comm, "Redistributed DMs are not equal, with %s with size %d.\n",user.repartitioning,size);
  }

  /* cleanup */
  ierr = PetscSectionDestroy(&s1);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&s2);CHKERRQ(ierr);
  ierr = ISDestroy(&is1);CHKERRQ(ierr);
  ierr = ISDestroy(&is2);CHKERRQ(ierr);
  ierr = DMDestroy(&dm1);CHKERRQ(ierr);
  ierr = DMDestroy(&dm2);CHKERRQ(ierr);
  ierr = DMDestroy(&dmdist1);CHKERRQ(ierr);
  ierr = DMDestroy(&dmdist2);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  test:
    # partition sequential mesh loaded from Exodus file
    suffix: 0
    nsize: {{1 2 3 4 8}}
    requires: chaco parmetis ptscotch exodusii
    args: -filename ${PETSC_DIR}/share/petsc/datafiles/meshes/blockcylinder-50.exo -interpolate
    args: -partitioning {{chaco parmetis ptscotch}} -repartitioning {{parmetis ptscotch}}
    args: -petscpartititoner_ptscotch_vertex_weight 0 -mat_partitioning_ptscotch_proc_weight 0
  test:
    # repartition mesh already partitioned naively by MED loader
    suffix: 1
    nsize: {{1 2 3 4 8}}
    TODO: MED
    requires: parmetis ptscotch med
    args: -filename ${PETSC_DIR}/share/petsc/datafiles/meshes/cylinder.med -interpolate
    args: -repartition 0 -partitioning {{parmetis ptscotch}}
    args: -petscpartititoner_ptscotch_vertex_weight 0 -mat_partitioning_ptscotch_proc_weight 0
  test:
    # partition mesh generated by ctetgen using scotch, then repartition with scotch, diff view
    # This fails if p1 uses SCOTCH_archCmplt and p2 uses SCOTCH_archCmpltw.
    suffix: 3
    nsize: 4
    requires: ptscotch ctetgen
    args: -faces 2,3,2 -partitioning ptscotch -repartitioning ptscotch -interpolate
    args: -p1_petscpartitioner_view -p2_petscpartitioner_view -dp1_petscpartitioner_view -dp2_petscpartitioner_view
    args: -petscpartititoner_ptscotch_vertex_weight 0 -mat_partitioning_ptscotch_proc_weight 0
  test:
    # partition mesh generated by ctetgen using partitioners supported both by MatPartitioning and PetscPartitioner
    suffix: 4
    nsize: {{1 2 3 4 8}}
    requires: chaco parmetis ptscotch ctetgen
    args: -faces {{2,3,4  5,4,3  7,11,5}} -partitioning {{chaco parmetis ptscotch}} -repartitioning {{parmetis ptscotch}} -interpolate
    args: -petscpartititoner_ptscotch_vertex_weight 0 -mat_partitioning_ptscotch_proc_weight 0

TEST*/

