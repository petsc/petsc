static char help[] = "Test that MatPartitioning and PetscPartitioner interfaces to parmetis are equivalent - using PETSCPARTITIONERMATPARTITIONING\n\n";
static char FILENAME[] = "ex24.c";

#include <petscdmplex.h>
#include <petscviewerhdf5.h>

typedef struct {
  PetscInt  dim;                          /* The topological mesh dimension */
  PetscInt  faces[3];                     /* Number of faces per dimension */
  PetscBool simplex;                      /* Use simplices or hexes */
  PetscBool interpolate;                  /* Interpolate mesh */
  char      filename[PETSC_MAX_PATH_LEN]; /* Import mesh from file */
  char      partitioning[64];
} AppCtx;

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscInt dim;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  options->dim          = 3;
  options->simplex      = PETSC_TRUE;
  options->interpolate  = PETSC_FALSE;
  options->filename[0]  = '\0';
  ierr = PetscOptionsBegin(comm, "", "Meshing Interpolation Test Options", "DMPLEX");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dim", "The topological mesh dimension", FILENAME, options->dim, &options->dim, NULL);CHKERRQ(ierr);
  if (options->dim > 3) SETERRQ1(comm, PETSC_ERR_ARG_OUTOFRANGE, "dimension set to %d, must be <= 3", options->dim);
  ierr = PetscOptionsBool("-simplex", "Use simplices if true, otherwise hexes", FILENAME, options->simplex, &options->simplex, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-interpolate", "Interpolate the mesh", FILENAME, options->interpolate, &options->interpolate, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsString("-filename", "The mesh file", FILENAME, options->filename, options->filename, PETSC_MAX_PATH_LEN, NULL);CHKERRQ(ierr);
  options->faces[0] = 1; options->faces[1] = 1; options->faces[2] = 1;
  dim = options->dim;
  ierr = PetscOptionsIntArray("-faces", "Number of faces per dimension", FILENAME, options->faces, &dim, NULL);CHKERRQ(ierr);
  ierr = PetscStrncpy(options->partitioning,MATPARTITIONINGCURRENT,64);CHKERRQ(ierr);
  ierr = PetscOptionsString("-partitioning","The mat partitioning type to test","None",options->partitioning, options->partitioning,64,NULL);CHKERRQ(ierr);
  if (dim) options->dim = dim;
  ierr = PetscOptionsEnd();
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
  IS             is1, is2;
  PetscSection   s1, s2;
  PetscBool      flg;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL,help);if (ierr) return ierr;
  comm = PETSC_COMM_WORLD;
  ierr = ProcessOptions(comm, &user);CHKERRQ(ierr);
  ierr = CreateMesh(comm, &user, &dm1);CHKERRQ(ierr);
  ierr = CreateMesh(comm, &user, &dm2);CHKERRQ(ierr);

  /* partition dm1 using PETSCPARTITIONERPARMETIS */
  ierr = DMPlexGetPartitioner(dm1, &part1);CHKERRQ(ierr);
  ierr = PetscObjectSetOptionsPrefix((PetscObject)part1,"p1_");CHKERRQ(ierr);
  ierr = PetscPartitionerSetType(part1, user.partitioning);CHKERRQ(ierr);
  ierr = PetscPartitionerSetFromOptions(part1);CHKERRQ(ierr);
  ierr = PetscSectionCreate(comm, &s1);CHKERRQ(ierr);
  ierr = PetscPartitionerPartition(part1, dm1, s1, &is1);CHKERRQ(ierr);

  /* partition dm2 using PETSCPARTITIONERMATPARTITIONING with MATPARTITIONINGPARMETIS */
  ierr = DMPlexGetPartitioner(dm2, &part2);CHKERRQ(ierr);
  ierr = PetscObjectSetOptionsPrefix((PetscObject)part2,"p2_");CHKERRQ(ierr);
  ierr = PetscPartitionerSetType(part2, PETSCPARTITIONERMATPARTITIONING);CHKERRQ(ierr);
  ierr = PetscPartitionerMatPartitioningGetMatPartitioning(part2, &mp);CHKERRQ(ierr);
  ierr = MatPartitioningSetType(mp, user.partitioning);CHKERRQ(ierr);
  ierr = PetscPartitionerSetFromOptions(part2);CHKERRQ(ierr);
  ierr = PetscSectionCreate(comm, &s2);CHKERRQ(ierr);
  ierr = PetscPartitionerPartition(part2, dm2, s2, &is2);CHKERRQ(ierr);

  /* compare the two ISs */
  {
    IS is1g, is2g;
    ierr = ISOnComm(is1, comm, PETSC_USE_POINTER, &is1g);CHKERRQ(ierr);
    ierr = ISOnComm(is2, comm, PETSC_USE_POINTER, &is2g);CHKERRQ(ierr);
    ierr = ISEqualUnsorted(is1g, is2g, &flg);CHKERRQ(ierr);
    if (!flg) PetscPrintf(comm, "ISs are not equal with type %s.\n",user.partitioning);
    ierr = ISDestroy(&is1g);CHKERRQ(ierr);
    ierr = ISDestroy(&is2g);CHKERRQ(ierr);
  }

  /* compare the two PetscSections */
  ierr = PetscSectionCompare(s1, s2, &flg);CHKERRQ(ierr);
  if (!flg) PetscPrintf(comm, "PetscSections are not equal with %.\n",user.partitioning);

  /* distribute both DMs */
  ierr = DMPlexDistribute(dm1, 0, NULL, &dmdist1);CHKERRQ(ierr);
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
  ierr = DMPlexEqual(dmdist1, dmdist2, &flg);CHKERRQ(ierr);
  if (!flg) PetscPrintf(comm, "Distributed DMs are not equal %s.\n",user.partitioning);

  /* repartition distributed DM dmdist1 using PETSCPARTITIONERPARMETIS */
  ierr = DMPlexGetPartitioner(dmdist1, &part1);CHKERRQ(ierr);
  ierr = PetscObjectSetOptionsPrefix((PetscObject)part1,"dp1_");CHKERRQ(ierr);
  ierr = PetscPartitionerSetType(part1, PETSCPARTITIONERPARMETIS);CHKERRQ(ierr);
  ierr = PetscPartitionerSetFromOptions(part1);CHKERRQ(ierr);
  ierr = PetscSectionCreate(comm, &s1);CHKERRQ(ierr);
  ierr = PetscPartitionerPartition(part1, dmdist1, s1, &is1);CHKERRQ(ierr);

  /* repartition distributed DM dmdist2 using PETSCPARTITIONERMATPARTITIONING with MATPARTITIONINGPARMETIS */
  ierr = DMPlexGetPartitioner(dmdist2, &part2);CHKERRQ(ierr);
  ierr = PetscObjectSetOptionsPrefix((PetscObject)part2,"dp2_");CHKERRQ(ierr);
  ierr = PetscPartitionerSetType(part2, PETSCPARTITIONERMATPARTITIONING);CHKERRQ(ierr);
  ierr = PetscPartitionerMatPartitioningGetMatPartitioning(part2, &mp);CHKERRQ(ierr);
  ierr = MatPartitioningSetType(mp, MATPARTITIONINGPARMETIS);CHKERRQ(ierr);
  ierr = PetscPartitionerSetFromOptions(part2);CHKERRQ(ierr);
  ierr = PetscSectionCreate(comm, &s2);CHKERRQ(ierr);
  ierr = PetscPartitionerPartition(part2, dmdist2, s2, &is2);CHKERRQ(ierr);

  /* compare the two ISs */
  {
    IS is1g, is2g;
    ierr = ISOnComm(is1, comm, PETSC_USE_POINTER, &is1g);CHKERRQ(ierr);
    ierr = ISOnComm(is2, comm, PETSC_USE_POINTER, &is2g);CHKERRQ(ierr);
    ierr = ISEqualUnsorted(is1g, is2g, &flg);CHKERRQ(ierr);
    if (!flg) PetscPrintf(comm, "Distributed ISs are not equal, with %s.\n",user.partitioning);
    ierr = ISDestroy(&is1g);CHKERRQ(ierr);
    ierr = ISDestroy(&is2g);CHKERRQ(ierr);
  }

  /* compare the two PetscSections */
  ierr = PetscSectionCompare(s1, s2, &flg);CHKERRQ(ierr);
  if (!flg) PetscPrintf(comm, "Distributed PetscSections are not equal, with %s.\n",user.partitioning);

  /* redistribute both distributed DMs */
  ierr = DMPlexDistribute(dmdist1, 0, NULL, &dm1);CHKERRQ(ierr);
  ierr = DMPlexDistribute(dmdist2, 0, NULL, &dm2);CHKERRQ(ierr);

  /* compare the two distributed DMs */
  ierr = DMPlexEqual(dm1, dm2, &flg);CHKERRQ(ierr);
  if (!flg) PetscPrintf(comm, "Redistributed DMs are not equal, with %s.\n",user.partitioning);

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
    nsize: 2
    requires: parmetis exodusii
    args: -filename ${PETSC_DIR}/share/petsc/datafiles/meshes/blockcylinder-50.exo
  test:
    # repartition mesh already partitioned naively by MED loader
    suffix: 1
    nsize: 2
    requires: parmetis med
    args: -filename ${PETSC_DIR}/share/petsc/datafiles/meshes/cylinder.med
  test:
    # the same for 4 processors
    suffix: 1_1
    nsize: 4
    requires: parmetis med
    args: -filename ${PETSC_DIR}/share/petsc/datafiles/meshes/cylinder.med
  test:
    # partition mesh generated by ctetgen using parmetis, then repartition using parmetis again
    suffix: 2
    nsize: 2
    requires: parmetis ctetgen
    args: -faces 2,3,4
  test:
    # the same for different number of elements
    suffix: 2_1
    nsize: 2
    requires: parmetis ctetgen
    args: -faces 5,4,3
  test:
    # the same for different number of elements, 4 processors
    suffix: 2_2
    nsize: 4
    requires: parmetis ctetgen
    args: -faces 7,11,5
  test:
    # partition mesh generated by ctetgen using parmetis, then repartition with scotch
    TODO: This fails due to inconsistent results of scotch. Its inputs are identical.
    suffix: 3
    nsize: 4
    requires: parmetis ptscotch ctetgen
    args: -options_left -faces 2,3,2 -p1_petscpartitioner_type parmetis -p2_petscpartitioner_type parmetis -dp1_petscpartitioner_type ptscotch -dp1_petscpartitioner_ptscotch_strategy BALANCE -dp1_petscpartitioner_ptscotch_imbalance 0.001 -dp1_petscpartitioner_view -dp2_petscpartitioner_type matpartitioning -dp2_mat_partitioning_type ptscotch -dp2_mat_partitioning_ptscotch_strategy BALANCE -dp2_mat_partitioning_ptscotch_imbalance 0.001 -dp2_petscpartitioner_view
  test:

    # partition mesh generated by ctetgen using chaco
    suffix: 4
    nsize: 2
    requires: chaco parmetis ptscotch ctetgen
    args: -faces 2,3,2 -partitioning {{chaco parmetis ptscotch}}}

TEST*/

