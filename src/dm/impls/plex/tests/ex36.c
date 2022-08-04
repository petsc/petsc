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
    PetscCall(VecGetDM(*v, &dm));
    PetscCall(DMGetLocalSection(dm, &section));
    PetscCall(DMViewFromOptions(dm, NULL, "-rd_dm_view"));
    PetscCall(DMViewFromOptions(dist_dm, NULL, "-rd2_dm_view"));

    PetscCall(DMClone(dm, &dist_v_dm));
    PetscCall(VecCreate(PetscObjectComm((PetscObject) *v), &dist_v));
    PetscCall(VecSetDM(dist_v, dist_v_dm));
    PetscCall(PetscSectionCreate(PetscObjectComm((PetscObject) *v), &dist_section));
    PetscCall(DMSetLocalSection(dist_v_dm, dist_section));

    PetscCall(PetscObjectViewFromOptions((PetscObject) section, NULL, "-rd_section_view"));
    PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject) dm), &rank));
    PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject) dm), &size));
    for (p = 0; p < size; ++p) {
      if (p == rank) {
        PetscCall(PetscObjectViewFromOptions((PetscObject) *v, NULL, "-rd_vec_view"));}
      PetscCall(PetscBarrier((PetscObject) dm));
      PetscCall(PetscViewerFlush(PETSC_VIEWER_STDOUT_WORLD));
    }
    PetscCall(DMPlexDistributeField(dm, sf, section, *v, dist_section, dist_v));
    for (p = 0; p < size; ++p) {
      if (p == rank) {
        PetscCall(PetscObjectViewFromOptions((PetscObject) dist_section, NULL, "-rd2_section_view"));
        PetscCall(PetscObjectViewFromOptions((PetscObject) dist_v, NULL, "-rd2_vec_view"));
      }
      PetscCall(PetscBarrier((PetscObject) dm));
      PetscCall(PetscViewerFlush(PETSC_VIEWER_STDOUT_WORLD));
    }

    PetscCall(PetscSectionDestroy(&dist_section));
    PetscCall(DMDestroy(&dist_v_dm));

    PetscCall(VecDestroy(v));
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
    PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));

    /* cells */
    PetscCall(DMPlexGetHeightStratum(dm, 0, &start_cell, &end_cell));
    PetscCall(VecGetDM(cell_geom, &cell_dm));
    PetscCall(DMGetLocalSection(cell_dm, &cell_section));
    PetscCall(VecGetArrayRead(cell_geom, &cell_array));

    for (c = start_cell; c < end_cell; ++c) {
      const PetscFVCellGeom *geom;
      PetscCall(PetscSectionGetOffset(cell_section, c, &offset));
      geom = (PetscFVCellGeom*)&cell_array[offset];
      PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD, "rank %d c %" PetscInt_FMT " centroid %g,%g,%g vol %g\n", rank, c, (double)geom->centroid[0], (double)geom->centroid[1], (double)geom->centroid[2], (double)geom->volume));
    }
    PetscCall(PetscSynchronizedFlush(PETSC_COMM_WORLD, NULL));
    PetscCall(VecRestoreArrayRead(cell_geom, &cell_array));

    /* faces */
    PetscCall(DMPlexGetHeightStratum(dm, 1, &start_face, &end_face));
    PetscCall(VecGetDM(face_geom, &face_dm));
    PetscCall(DMGetLocalSection(face_dm, &face_section));
    PetscCall(VecGetArrayRead(face_geom, &face_array));
    for (f = start_face; f < end_face; ++f) {
       PetscCall(DMPlexGetSupport(dm, f, &cells));
       PetscCall(DMPlexGetSupportSize(dm, f, &supportSize));
       if (supportSize > 1) {
          PetscCall(PetscSectionGetOffset(face_section, f, &offset));
          PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD, "rank %d f %" PetscInt_FMT " cells %" PetscInt_FMT ",%" PetscInt_FMT " normal %g,%g,%g centroid %g,%g,%g\n", rank, f, cells[0], cells[1], (double) PetscRealPart(face_array[offset+0]), (double) PetscRealPart(face_array[offset+1]), (double) PetscRealPart(face_array[offset+2]), (double) PetscRealPart(face_array[offset+3]), (double) PetscRealPart(face_array[offset+4]), (double) PetscRealPart(face_array[offset+5])));
       }
    }
    PetscCall(PetscSynchronizedFlush(PETSC_COMM_WORLD, NULL));
    PetscCall(VecRestoreArrayRead(cell_geom, &cell_array));
    PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM               dm, redist_dm;
  PetscPartitioner part;
  PetscSF          redist_sf;
  Vec              cell_geom, face_geom;
  PetscInt         overlap2 = 1;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(DMCreate(PETSC_COMM_WORLD, &dm));
  PetscCall(DMSetType(dm, DMPLEX));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMViewFromOptions(dm, NULL, "-dm_view"));

  PetscCall(DMPlexComputeGeometryFVM(dm, &cell_geom, &face_geom));
  PetscCall(dm_view_geometry(dm, cell_geom, face_geom));

  /* redistribute */
  PetscCall(DMPlexGetPartitioner(dm, &part));
  PetscCall(PetscPartitionerSetFromOptions(part));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-overlap2", &overlap2, NULL));
  PetscCall(DMPlexDistribute(dm, overlap2, &redist_sf, &redist_dm));
  if (redist_dm) {
    PetscCall(redistribute_vec(redist_dm, redist_sf, &cell_geom));
    PetscCall(redistribute_vec(redist_dm, redist_sf, &face_geom));
    PetscCall(PetscObjectViewFromOptions((PetscObject) redist_sf, NULL, "-rd2_sf_view"));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "redistributed:\n"));
    PetscCall(dm_view_geometry(redist_dm, cell_geom, face_geom));
  }

  PetscCall(VecDestroy(&cell_geom));
  PetscCall(VecDestroy(&face_geom));
  PetscCall(PetscSFDestroy(&redist_sf));
  PetscCall(DMDestroy(&redist_dm));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: 0
    nsize: 3
    args: -dm_plex_dim 3 -dm_plex_box_faces 8,1,1 -dm_plex_simplex 0 -dm_plex_adj_cone 1 -dm_plex_adj_closure 0 -petscpartitioner_type simple -dm_distribute_overlap 1 -overlap2 1

TEST*/
