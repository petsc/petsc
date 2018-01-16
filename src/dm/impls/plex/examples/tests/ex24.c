static char help[] = "Test that MatPartitioning and PetscPartitioner interfaces to parmetis are equivalent - using PETSCPARTITIONERMATPARTITIONING\n\n";

#include <petscdmplex.h>
#include <petscviewerhdf5.h>

typedef struct {
  PetscInt  dim;                          /* The topological mesh dimension */
  PetscInt  faces[3];                     /* Number of faces per dimension */
  PetscBool simplex;                      /* Use simplices or hexes */
  PetscBool interpolate;                  /* Interpolate mesh */
  char      filename[PETSC_MAX_PATH_LEN]; /* Import mesh from file */
} AppCtx;

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscInt nfaces;
  PetscInt  faces[3] = {1,1,1};
  PetscErrorCode ierr;

  PetscFunctionBegin;
  options->dim          = 3;
  options->simplex      = PETSC_TRUE;
  options->interpolate  = PETSC_FALSE;
  options->filename[0]  = '\0';
  ierr = PetscOptionsBegin(comm, "", "Meshing Interpolation Test Options", "DMPLEX");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dim", "The topological mesh dimension", "ex23.c", options->dim, &options->dim, NULL);CHKERRQ(ierr);
  if (options->dim > 3) SETERRQ1(comm, PETSC_ERR_ARG_OUTOFRANGE, "dimension set to %d, must be <= 3", options->dim);
  ierr = PetscOptionsBool("-simplex", "Use simplices if true, otherwise hexes", "ex23.c", options->simplex, &options->simplex, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-interpolate", "Interpolate the mesh", "ex23.c", options->interpolate, &options->interpolate, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsString("-filename", "The mesh file", "ex23.c", options->filename, options->filename, PETSC_MAX_PATH_LEN, NULL);CHKERRQ(ierr);
  nfaces = options->dim;
  ierr = PetscOptionsIntArray("-faces", "Number of faces per dimension", "ex23.c", faces, &nfaces, NULL);CHKERRQ(ierr);
  if (nfaces) options->dim = nfaces;
  ierr = PetscMemcpy(options->faces, faces, 3*sizeof(PetscInt));CHKERRQ(ierr);
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
  ierr = PetscPartitionerSetType(part1, PETSCPARTITIONERPARMETIS);CHKERRQ(ierr);
  ierr = PetscPartitionerSetFromOptions(part1);CHKERRQ(ierr);
  ierr = PetscSectionCreate(comm, &s1);CHKERRQ(ierr);
  ierr = PetscPartitionerPartition(part1, dm1, s1, &is1);CHKERRQ(ierr);

  /* partition dm2 using PETSCPARTITIONERMATPARTITIONING with MATPARTITIONINGPARMETIS */
  ierr = DMPlexGetPartitioner(dm2, &part2);CHKERRQ(ierr);
  ierr = PetscObjectSetOptionsPrefix((PetscObject)part2,"p2_");CHKERRQ(ierr);
  ierr = PetscPartitionerSetType(part2, PETSCPARTITIONERMATPARTITIONING);CHKERRQ(ierr);
  ierr = PetscPartitionerMatPartitioningGetMatPartitioning(part2, &mp);CHKERRQ(ierr);
  ierr = MatPartitioningSetType(mp, MATPARTITIONINGPARMETIS);CHKERRQ(ierr);
  ierr = PetscPartitionerSetFromOptions(part2);CHKERRQ(ierr);
  ierr = PetscSectionCreate(comm, &s2);CHKERRQ(ierr);
  ierr = PetscPartitionerPartition(part2, dm2, s2, &is2);CHKERRQ(ierr);

  /* compare the two ISs */
  {
    IS is1g, is2g;
    ierr = ISOnComm(is1, comm, PETSC_USE_POINTER, &is1g);CHKERRQ(ierr);
    ierr = ISOnComm(is2, comm, PETSC_USE_POINTER, &is2g);CHKERRQ(ierr);
    ierr = ISEqualUnsorted(is1g, is2g, &flg);CHKERRQ(ierr);
    if (!flg) PetscPrintf(comm, "ISs are not equal.\n");
    ierr = ISDestroy(&is1g);CHKERRQ(ierr);
    ierr = ISDestroy(&is2g);CHKERRQ(ierr);
  }

  /* compare the two PetscSections */
  ierr = PetscSectionCompare(s1, s2, &flg);CHKERRQ(ierr);
  if (!flg) PetscPrintf(comm, "PetscSections are not equal.\n");

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
  if (!flg) PetscPrintf(comm, "Distributed DMs are not equal.\n");

  /* repartition distributed DM dmdist1 using PETSCPARTITIONERPARMETIS */
  ierr = DMPlexGetPartitioner(dmdist1, &part1);CHKERRQ(ierr);
  ierr = PetscObjectSetOptionsPrefix((PetscObject)part1,"dp1_");CHKERRQ(ierr);
  ierr = PetscPartitionerSetType(part1, PETSCPARTITIONERPARMETIS);CHKERRQ(ierr);
  ierr = PetscPartitionerSetFromOptions(part1);CHKERRQ(ierr);
  ierr = PetscSectionCreate(comm, &s1);CHKERRQ(ierr);
  ierr = PetscPartitionerPartition(part1, dmdist1, s1, &is1);CHKERRQ(ierr);

  /* repartition distributed DM dmdist2 using PETSCPARTITIONERMATPARTITIONING with MATPARTITIONINGPARMETIS */
  ierr = DMPlexGetPartitioner(dmdist2, &part2);CHKERRQ(ierr);
  ierr = PetscObjectSetOptionsPrefix((PetscObject)part1,"dp2_");CHKERRQ(ierr);
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
    if (!flg) PetscPrintf(comm, "Distributed ISs are not equal.\n");
    ierr = ISDestroy(&is1g);CHKERRQ(ierr);
    ierr = ISDestroy(&is2g);CHKERRQ(ierr);
  }

  /* compare the two PetscSections */
  ierr = PetscSectionCompare(s1, s2, &flg);CHKERRQ(ierr);
  if (!flg) PetscPrintf(comm, "Distributed PetscSections are not equal.\n");

  /* redistribute both distributed DMs */
  ierr = DMPlexDistribute(dmdist1, 0, NULL, &dm1);CHKERRQ(ierr);
  ierr = DMPlexDistribute(dmdist2, 0, NULL, &dm2);CHKERRQ(ierr);

  /* compare the two distributed DMs */
  ierr = DMPlexEqual(dm1, dm2, &flg);CHKERRQ(ierr);
  if (!flg) PetscPrintf(comm, "Redistributed DMs are not equal.\n");

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
    suffix: 0
    requires: parmetis exodusii
    args: -filename ${PETSC_DIR}/share/petsc/datafiles/meshes/blockcylinder-50.exo
  test:
    suffix: 1
    nsize: 2
    requires: parmetis med
    args: -filename ${PETSC_DIR}/share/petsc/datafiles/meshes/cylinder.med
  test:
    suffix: 2
    nsize: 4
    requires: parmetis med
    args: -filename ${PETSC_DIR}/share/petsc/datafiles/meshes/cylinder.med
  test:
    suffix: 3
    requires: parmetis ctetgen
    args: -faces 2,3,4
  test:
    suffix: 4
    nsize: 2
    requires: parmetis ctetgen
    args: -faces 5,4,3
  test:
    suffix: 5
    nsize: 4
    requires: parmetis ctetgen
    args: -faces 7,11,5
  test:
    TODO: why this fails?
    suffix: 6
    nsize: 4
    requires: ptscotch ctetgen
    args: -options_left -faces 7,11,5 -p1_petscpartitioner_type ptscotch -p1_petscpartitioner_ptscotch_strategy BALANCE -p1_petscpartitioner_ptscotch_imbalance 0.001 -p1_petscpartitioner_view -p2_petscpartitioner_type matpartitioning -p2_mat_partitioning_type ptscotch -p2_mat_partitioning_ptscotch_strategy BALANCE -p2_mat_partitioning_ptscotch_imbalance 0.001 -p2_petscpartitioner_view

TEST*/

