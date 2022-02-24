static char help[] = "Tests interpolation and output of hybrid meshes\n\n";

#include <petscdmplex.h>
#include <petscsf.h>

/* Much can be learned using
     -rd_dm_view -rd2_dm_view -rd_section_view -rd_vec_view -rd2_section_view */

static PetscErrorCode redistribute_vec(DM dist_dm, PetscSF sf, Vec *v)
{
    DM             dm, dist_v_dm;
    PetscSection   section, dist_section;
    Vec            dist_v;
    PetscMPIInt    rank, size, p;

    PetscFunctionBegin;
    CHKERRQ(VecGetDM(*v, &dm));
    CHKERRQ(DMGetLocalSection(dm, &section));
    CHKERRQ(DMViewFromOptions(dm, NULL, "-rd_dm_view"));
    CHKERRQ(DMViewFromOptions(dist_dm, NULL, "-rd2_dm_view"));

    CHKERRQ(DMClone(dm, &dist_v_dm));
    CHKERRQ(VecCreate(PetscObjectComm((PetscObject) *v), &dist_v));
    CHKERRQ(VecSetDM(dist_v, dist_v_dm));
    CHKERRQ(PetscSectionCreate(PetscObjectComm((PetscObject) *v), &dist_section));
    CHKERRQ(DMSetLocalSection(dist_v_dm, dist_section));

    CHKERRQ(PetscObjectViewFromOptions((PetscObject) section, NULL, "-rd_section_view"));
    CHKERRMPI(MPI_Comm_rank(PetscObjectComm((PetscObject) dm), &rank));
    CHKERRMPI(MPI_Comm_size(PetscObjectComm((PetscObject) dm), &size));
    for (p = 0; p < size; ++p) {
      if (p == rank) {
        CHKERRQ(PetscObjectViewFromOptions((PetscObject) *v, NULL, "-rd_vec_view"));}
      CHKERRQ(PetscBarrier((PetscObject) dm));
      CHKERRQ(PetscViewerFlush(PETSC_VIEWER_STDOUT_WORLD));
    }
    CHKERRQ(DMPlexDistributeField(dm, sf, section, *v, dist_section, dist_v));
    for (p = 0; p < size; ++p) {
      if (p == rank) {
        CHKERRQ(PetscObjectViewFromOptions((PetscObject) dist_section, NULL, "-rd2_section_view"));
        CHKERRQ(PetscObjectViewFromOptions((PetscObject) dist_v, NULL, "-rd2_vec_view"));
      }
      CHKERRQ(PetscBarrier((PetscObject) dm));
      CHKERRQ(PetscViewerFlush(PETSC_VIEWER_STDOUT_WORLD));
    }

    CHKERRQ(PetscSectionDestroy(&dist_section));
    CHKERRQ(DMDestroy(&dist_v_dm));

    CHKERRQ(VecDestroy(v));
    *v   = dist_v;
    PetscFunctionReturn(0);
}

static PetscErrorCode dm_view_geometry(DM dm, Vec cell_geom, Vec face_geom)
{
    DM                 cell_dm, face_dm;
    PetscSection       cell_section, face_section;
    const PetscScalar *cell_array, *face_array;
    const PetscInt    *cells;
    PetscInt           c, start_cell, end_cell;
    PetscInt           f, start_face, end_face;
    PetscInt           supportSize, offset;
    PetscMPIInt        rank;

    PetscFunctionBegin;
    CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));

    /* cells */
    CHKERRQ(DMPlexGetHeightStratum(dm, 0, &start_cell, &end_cell));
    CHKERRQ(VecGetDM(cell_geom, &cell_dm));
    CHKERRQ(DMGetLocalSection(cell_dm, &cell_section));
    CHKERRQ(VecGetArrayRead(cell_geom, &cell_array));

    for (c = start_cell; c < end_cell; ++c) {
      const PetscFVCellGeom *geom;
      CHKERRQ(PetscSectionGetOffset(cell_section, c, &offset));
      geom = (PetscFVCellGeom*)&cell_array[offset];
      CHKERRQ(PetscSynchronizedPrintf(PETSC_COMM_WORLD, "rank %d c %D centroid %g,%g,%g vol %g\n", rank, c, (double)geom->centroid[0], (double)geom->centroid[1], (double)geom->centroid[2], (double)geom->volume));
    }
    CHKERRQ(PetscSynchronizedFlush(PETSC_COMM_WORLD, NULL));
    CHKERRQ(VecRestoreArrayRead(cell_geom, &cell_array));

    /* faces */
    CHKERRQ(DMPlexGetHeightStratum(dm, 1, &start_face, &end_face));
    CHKERRQ(VecGetDM(face_geom, &face_dm));
    CHKERRQ(DMGetLocalSection(face_dm, &face_section));
    CHKERRQ(VecGetArrayRead(face_geom, &face_array));
    for (f = start_face; f < end_face; ++f) {
       CHKERRQ(DMPlexGetSupport(dm, f, &cells));
       CHKERRQ(DMPlexGetSupportSize(dm, f, &supportSize));
       if (supportSize > 1) {
          CHKERRQ(PetscSectionGetOffset(face_section, f, &offset));
          CHKERRQ(PetscSynchronizedPrintf(PETSC_COMM_WORLD, "rank %d f %D cells %D,%D normal %g,%g,%g centroid %g,%g,%g\n", rank, f, cells[0], cells[1], (double) PetscRealPart(face_array[offset+0]), (double) PetscRealPart(face_array[offset+1]), (double) PetscRealPart(face_array[offset+2]), (double) PetscRealPart(face_array[offset+3]), (double) PetscRealPart(face_array[offset+4]), (double) PetscRealPart(face_array[offset+5])));
       }
    }
    CHKERRQ(PetscSynchronizedFlush(PETSC_COMM_WORLD, NULL));
    CHKERRQ(VecRestoreArrayRead(cell_geom, &cell_array));
    PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM               dm, redist_dm;
  PetscPartitioner part;
  PetscSF          redist_sf;
  Vec              cell_geom, face_geom;
  PetscInt         overlap2 = 1;
  PetscErrorCode   ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help); if (ierr) return ierr;
  CHKERRQ(DMCreate(PETSC_COMM_WORLD, &dm));
  CHKERRQ(DMSetType(dm, DMPLEX));
  CHKERRQ(DMSetFromOptions(dm));
  CHKERRQ(DMViewFromOptions(dm, NULL, "-dm_view"));

  CHKERRQ(DMPlexComputeGeometryFVM(dm, &cell_geom, &face_geom));
  CHKERRQ(dm_view_geometry(dm, cell_geom, face_geom));

  /* redistribute */
  CHKERRQ(DMPlexGetPartitioner(dm, &part));
  CHKERRQ(PetscPartitionerSetFromOptions(part));
  CHKERRQ(PetscOptionsGetInt(NULL, NULL, "-overlap2", &overlap2, NULL));
  CHKERRQ(DMPlexDistribute(dm, overlap2, &redist_sf, &redist_dm));
  if (redist_dm) {
    CHKERRQ(redistribute_vec(redist_dm, redist_sf, &cell_geom));
    CHKERRQ(redistribute_vec(redist_dm, redist_sf, &face_geom));
    CHKERRQ(PetscObjectViewFromOptions((PetscObject) redist_sf, NULL, "-rd2_sf_view"));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "redistributed:\n"));
    CHKERRQ(dm_view_geometry(redist_dm, cell_geom, face_geom));
  }

  CHKERRQ(VecDestroy(&cell_geom));
  CHKERRQ(VecDestroy(&face_geom));
  CHKERRQ(PetscSFDestroy(&redist_sf));
  CHKERRQ(DMDestroy(&redist_dm));
  CHKERRQ(DMDestroy(&dm));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  test:
    suffix: 0
    nsize: 3
    args: -dm_plex_dim 3 -dm_plex_box_faces 8,1,1 -dm_plex_simplex 0 -dm_plex_adj_cone 1 -dm_plex_adj_closure 0 -petscpartitioner_type simple -dm_distribute_overlap 1 -overlap2 1

TEST*/
