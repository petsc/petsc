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
    PetscErrorCode ierr;

    PetscFunctionBegin;
    ierr = VecGetDM(*v, &dm);CHKERRQ(ierr);
    ierr = DMGetLocalSection(dm, &section);CHKERRQ(ierr);
    ierr = DMViewFromOptions(dm, NULL, "-rd_dm_view");CHKERRQ(ierr);
    ierr = DMViewFromOptions(dist_dm, NULL, "-rd2_dm_view");CHKERRQ(ierr);

    ierr = DMClone(dm, &dist_v_dm);CHKERRQ(ierr);
    ierr = VecCreate(PetscObjectComm((PetscObject) *v), &dist_v);CHKERRQ(ierr);
    ierr = VecSetDM(dist_v, dist_v_dm);CHKERRQ(ierr);
    ierr = PetscSectionCreate(PetscObjectComm((PetscObject) *v), &dist_section);CHKERRQ(ierr);
    ierr = DMSetLocalSection(dist_v_dm, dist_section);CHKERRQ(ierr);

    ierr = PetscObjectViewFromOptions((PetscObject) section, NULL, "-rd_section_view");CHKERRQ(ierr);
    ierr = MPI_Comm_rank(PetscObjectComm((PetscObject) dm), &rank);CHKERRQ(ierr);
    ierr = MPI_Comm_size(PetscObjectComm((PetscObject) dm), &size);CHKERRQ(ierr);
    for (p = 0; p < size; ++p) {
      if (p == rank) {
        ierr = PetscObjectViewFromOptions((PetscObject) *v, NULL, "-rd_vec_view");CHKERRQ(ierr);}
      ierr = PetscBarrier((PetscObject) dm);CHKERRQ(ierr);
      ierr = PetscViewerFlush(PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    }
    ierr = DMPlexDistributeField(dm, sf, section, *v, dist_section, dist_v);CHKERRQ(ierr);
    for (p = 0; p < size; ++p) {
      if (p == rank) {
        ierr = PetscObjectViewFromOptions((PetscObject) dist_section, NULL, "-rd2_section_view");CHKERRQ(ierr);
        ierr = PetscObjectViewFromOptions((PetscObject) dist_v, NULL, "-rd2_vec_view");CHKERRQ(ierr);
      }
      ierr = PetscBarrier((PetscObject) dm);CHKERRQ(ierr);
      ierr = PetscViewerFlush(PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    }

    ierr = PetscSectionDestroy(&dist_section);CHKERRQ(ierr);
    ierr = DMDestroy(&dist_v_dm);CHKERRQ(ierr);

    ierr = VecDestroy(v);CHKERRQ(ierr);
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
    PetscErrorCode     ierr;

    PetscFunctionBegin;
    ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank);CHKERRQ(ierr);

    /* cells */
    ierr = DMPlexGetHeightStratum(dm, 0, &start_cell, &end_cell);CHKERRQ(ierr);
    ierr = VecGetDM(cell_geom, &cell_dm);CHKERRQ(ierr);
    ierr = DMGetLocalSection(cell_dm, &cell_section);CHKERRQ(ierr);
    ierr = VecGetArrayRead(cell_geom, &cell_array);CHKERRQ(ierr);

    for (c = start_cell; c < end_cell; ++c) {
      const PetscFVCellGeom *geom;
      ierr = PetscSectionGetOffset(cell_section, c, &offset);CHKERRQ(ierr);
      geom = (PetscFVCellGeom*)&cell_array[offset];
      ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, "rank %d c %D centroid %g,%g,%g vol %g\n", rank, c, (double)geom->centroid[0], (double)geom->centroid[1], (double)geom->centroid[2], (double)geom->volume);CHKERRQ(ierr);
    }
    ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD, NULL);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(cell_geom, &cell_array);CHKERRQ(ierr);

    /* faces */
    ierr = DMPlexGetHeightStratum(dm, 1, &start_face, &end_face);CHKERRQ(ierr);
    ierr = VecGetDM(face_geom, &face_dm);CHKERRQ(ierr);
    ierr = DMGetLocalSection(face_dm, &face_section);CHKERRQ(ierr);
    ierr = VecGetArrayRead(face_geom, &face_array);CHKERRQ(ierr);
    for (f = start_face; f < end_face; ++f) {
       ierr = DMPlexGetSupport(dm, f, &cells);CHKERRQ(ierr);
       ierr = DMPlexGetSupportSize(dm, f, &supportSize);CHKERRQ(ierr);
       if (supportSize > 1) {
          ierr = PetscSectionGetOffset(face_section, f, &offset);CHKERRQ(ierr);
          ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, "rank %d f %D cells %D,%D normal %g,%g,%g centroid %g,%g,%g\n", rank, f, cells[0], cells[1], (double) PetscRealPart(face_array[offset+0]), (double) PetscRealPart(face_array[offset+1]), (double) PetscRealPart(face_array[offset+2]), (double) PetscRealPart(face_array[offset+3]), (double) PetscRealPart(face_array[offset+4]), (double) PetscRealPart(face_array[offset+5]));CHKERRQ(ierr);
       }
    }
    ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD, NULL);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(cell_geom, &cell_array);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM               dm, dist_dm, redist_dm;
  PetscPartitioner part;
  PetscSF          dist_sf, redist_sf;
  Vec              cell_geom, face_geom;
  PetscInt         overlap = 1, overlap2 = 1;
  PetscMPIInt      rank;
  const char      *filename = "gminc_1d.exo";
  PetscErrorCode   ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help); if (ierr) return ierr;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank);CHKERRQ(ierr);
  if (0) {
    ierr = DMPlexCreateFromFile(PETSC_COMM_WORLD, filename, PETSC_TRUE, &dm);CHKERRQ(ierr);
  } else {
    ierr = DMPlexCreateBoxMesh(PETSC_COMM_WORLD, 3, PETSC_FALSE, NULL, NULL, NULL, NULL, PETSC_TRUE, &dm);CHKERRQ(ierr);
  }
  ierr = DMSetBasicAdjacency(dm, PETSC_TRUE, PETSC_FALSE);CHKERRQ(ierr);

  ierr = DMPlexGetPartitioner(dm, &part);CHKERRQ(ierr);
  ierr = PetscPartitionerSetFromOptions(part);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL, NULL, "-overlap", &overlap, NULL);CHKERRQ(ierr);
  ierr = DMPlexDistribute(dm, overlap, &dist_sf, &dist_dm);CHKERRQ(ierr);
  if (dist_dm) {
     ierr = DMDestroy(&dm);CHKERRQ(ierr);
     dm = dist_dm;
  }

  ierr = DMPlexComputeGeometryFVM(dm, &cell_geom, &face_geom);CHKERRQ(ierr);

  ierr = dm_view_geometry(dm, cell_geom, face_geom);CHKERRQ(ierr);

  /* redistribute */
  ierr = DMPlexGetPartitioner(dm, &part);CHKERRQ(ierr);
  ierr = PetscPartitionerSetFromOptions(part);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL, NULL, "-overlap2", &overlap2, NULL);CHKERRQ(ierr);
  ierr = DMPlexDistribute(dm, overlap2, &redist_sf, &redist_dm);CHKERRQ(ierr);
  if (redist_dm) {
    ierr = redistribute_vec(redist_dm, redist_sf, &cell_geom);CHKERRQ(ierr);
    ierr = redistribute_vec(redist_dm, redist_sf, &face_geom);CHKERRQ(ierr);
    ierr = PetscObjectViewFromOptions((PetscObject) redist_sf, NULL, "-rd2_sf_view");CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "redistributed:\n");CHKERRQ(ierr);
    ierr = dm_view_geometry(redist_dm, cell_geom, face_geom);CHKERRQ(ierr);
  }

  ierr = VecDestroy(&cell_geom);CHKERRQ(ierr);
  ierr = VecDestroy(&face_geom);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&dist_sf);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&redist_sf);CHKERRQ(ierr);
  ierr = DMDestroy(&redist_dm);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  test:
    suffix: 0
    nsize: 3
    args: -overlap 1 -overlap2 1 -dm_plex_box_faces 8,1,1 -petscpartitioner_type simple

TEST*/
