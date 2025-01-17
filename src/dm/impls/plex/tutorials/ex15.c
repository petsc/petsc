#include "petscsf.h"
static char help[] = "Demonstrate CGNS parallel load-save-reload cycle, including data\n\n";

#include <petscdmplex.h>
#include <petscviewerhdf5.h>
#define EX "ex15.c"

typedef struct {
  char      infile[PETSC_MAX_PATH_LEN];  /* Input mesh filename */
  char      outfile[PETSC_MAX_PATH_LEN]; /* Dump/reload mesh filename */
  PetscBool heterogeneous;               /* Test save on N / load on M */
  PetscInt  ntimes;                      /* How many times do the cycle */
} AppCtx;

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscBool flg;

  PetscFunctionBeginUser;
  options->infile[0]     = '\0';
  options->outfile[0]    = '\0';
  options->heterogeneous = PETSC_FALSE;
  options->ntimes        = 2;
  PetscOptionsBegin(comm, "", "Meshing Problem Options", "DMPLEX");
  PetscCall(PetscOptionsString("-infile", "The input CGNS file", EX, options->infile, options->infile, sizeof(options->infile), &flg));
  PetscCall(PetscOptionsString("-outfile", "The output CGNS file", EX, options->outfile, options->outfile, sizeof(options->outfile), &flg));
  PetscCall(PetscOptionsBool("-heterogeneous", "Test save on N / load on M", EX, options->heterogeneous, &options->heterogeneous, NULL));
  PetscCall(PetscOptionsInt("-ntimes", "How many times do the cycle", EX, options->ntimes, &options->ntimes, NULL));
  PetscOptionsEnd();
  PetscCheck(flg, comm, PETSC_ERR_USER_INPUT, "-infile needs to be specified");
  PetscCheck(flg, comm, PETSC_ERR_USER_INPUT, "-outfile needs to be specified");
  PetscFunctionReturn(PETSC_SUCCESS);
}

// @brief Create DM from CGNS file and setup PetscFE to VecLoad solution from that file
PetscErrorCode ReadCGNSDM(MPI_Comm comm, const char filename[], DM *dm)
{
  PetscInt degree;

  PetscFunctionBeginUser;
  PetscCall(DMPlexCreateFromFile(comm, filename, "ex15_plex", PETSC_TRUE, dm));
  PetscCall(DMPlexDistributeSetDefault(*dm, PETSC_FALSE));
  PetscCall(DMSetFromOptions(*dm));
  PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));

  /* Redistribute */
  PetscCall(DMSetOptionsPrefix(*dm, "redistributed_"));
  PetscCall(DMSetFromOptions(*dm));
  PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));

  { // Get degree of the natural section
    PetscFE        fe_natural;
    PetscDualSpace dual_space_natural;

    PetscCall(DMGetField(*dm, 0, NULL, (PetscObject *)&fe_natural));
    PetscCall(PetscFEGetDualSpace(fe_natural, &dual_space_natural));
    PetscCall(PetscDualSpaceGetOrder(dual_space_natural, &degree));
    PetscCall(DMClearFields(*dm));
    PetscCall(DMSetLocalSection(*dm, NULL));
  }

  { // Setup fe to load in the initial condition data
    PetscFE  fe;
    PetscInt dim;

    PetscCall(DMGetDimension(*dm, &dim));
    PetscCall(PetscFECreateLagrange(PETSC_COMM_SELF, dim, 5, PETSC_FALSE, degree, PETSC_DETERMINE, &fe));
    PetscCall(PetscObjectSetName((PetscObject)fe, "FE for VecLoad"));
    PetscCall(DMAddField(*dm, NULL, (PetscObject)fe));
    PetscCall(DMCreateDS(*dm));
    PetscCall(PetscFEDestroy(&fe));
  }

  // Set section component names, used when writing out CGNS files
  PetscSection section;
  PetscCall(DMGetLocalSection(*dm, &section));
  PetscCall(PetscSectionSetFieldName(section, 0, ""));
  PetscCall(PetscSectionSetComponentName(section, 0, 0, "Pressure"));
  PetscCall(PetscSectionSetComponentName(section, 0, 1, "VelocityX"));
  PetscCall(PetscSectionSetComponentName(section, 0, 2, "VelocityY"));
  PetscCall(PetscSectionSetComponentName(section, 0, 3, "VelocityZ"));
  PetscCall(PetscSectionSetComponentName(section, 0, 4, "Temperature"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Verify that V_load is equivalent to V_serial, even if V_load is distributed
PetscErrorCode VerifyLoadedSolution(DM dm_serial, Vec V_serial, DM dm_load, Vec V_load, PetscScalar tol)
{
  MPI_Comm     comm = PetscObjectComm((PetscObject)dm_load);
  PetscSF      load_to_serial_sf;
  PetscScalar *array_load_bcast = NULL;
  PetscInt     num_comps        = 5;

  PetscFunctionBeginUser;
  { // Create SF to broadcast loaded vec nodes to serial vec nodes
    PetscInt           dim, num_local_serial = 0, num_local_load;
    Vec                coord_Vec_serial, coord_Vec_load;
    const PetscScalar *coord_serial = NULL, *coord_load;

    PetscCall(DMGetCoordinateDim(dm_load, &dim));
    PetscCall(DMGetCoordinates(dm_load, &coord_Vec_load));
    PetscCall(VecGetLocalSize(coord_Vec_load, &num_local_load));
    num_local_load /= dim;

    PetscCall(VecGetArrayRead(coord_Vec_load, &coord_load));

    if (dm_serial) {
      PetscCall(DMGetCoordinates(dm_serial, &coord_Vec_serial));
      PetscCall(VecGetLocalSize(coord_Vec_serial, &num_local_serial));
      num_local_serial /= dim;
      PetscCall(VecGetArrayRead(coord_Vec_serial, &coord_serial));
    }

    PetscCall(PetscSFCreate(comm, &load_to_serial_sf));
    PetscCall(PetscSFSetGraphFromCoordinates(load_to_serial_sf, num_local_load, num_local_serial, dim, 100 * PETSC_MACHINE_EPSILON, coord_load, coord_serial));
    PetscCall(PetscSFViewFromOptions(load_to_serial_sf, NULL, "-verify_sf_view"));

    PetscCall(VecRestoreArrayRead(coord_Vec_load, &coord_load));
    if (dm_serial) PetscCall(VecRestoreArrayRead(coord_Vec_serial, &coord_serial));
  }

  { // Broadcast the loaded vector data into a serial-sized array
    PetscInt           size_local_serial = 0;
    const PetscScalar *array_load;
    MPI_Datatype       unit;

    PetscCall(VecGetArrayRead(V_load, &array_load));
    if (V_serial) {
      PetscCall(VecGetLocalSize(V_serial, &size_local_serial));
      PetscCall(PetscMalloc1(size_local_serial, &array_load_bcast));
    }

    PetscCallMPI(MPI_Type_contiguous(num_comps, MPIU_REAL, &unit));
    PetscCallMPI(MPI_Type_commit(&unit));
    PetscCall(PetscSFBcastBegin(load_to_serial_sf, unit, array_load, array_load_bcast, MPI_REPLACE));
    PetscCall(PetscSFBcastEnd(load_to_serial_sf, unit, array_load, array_load_bcast, MPI_REPLACE));
    PetscCallMPI(MPI_Type_free(&unit));
    PetscCall(VecRestoreArrayRead(V_load, &array_load));
  }

  if (V_serial) {
    const PetscScalar *array_serial;
    PetscInt           size_local_serial;

    PetscCall(VecGetArrayRead(V_serial, &array_serial));
    PetscCall(VecGetLocalSize(V_serial, &size_local_serial));
    for (PetscInt i = 0; i < size_local_serial; i++) {
      if (PetscAbs(array_serial[i] - array_load_bcast[i]) > tol) PetscCall(PetscPrintf(comm, "DoF %" PetscInt_FMT " is inconsistent. Found %g, expected %g\n", i, array_load_bcast[i], array_serial[i]));
    }
    PetscCall(VecRestoreArrayRead(V_serial, &array_serial));
  }

  PetscCall(PetscFree(array_load_bcast));
  PetscCall(PetscSFDestroy(&load_to_serial_sf));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  AppCtx      user;
  MPI_Comm    comm;
  PetscMPIInt gsize, grank, mycolor;
  PetscBool   flg;
  const char *infilename;
  DM          dm_serial = NULL;
  Vec         V_serial  = NULL;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(ProcessOptions(PETSC_COMM_WORLD, &user));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &gsize));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &grank));

  { // Read infile in serial
    PetscViewer viewer;
    PetscMPIInt gsize_serial;

    mycolor = grank == 0 ? 0 : 1;
    PetscCallMPI(MPI_Comm_split(PETSC_COMM_WORLD, mycolor, grank, &comm));

    if (grank == 0) {
      PetscCallMPI(MPI_Comm_size(comm, &gsize_serial));

      PetscCall(ReadCGNSDM(comm, user.infile, &dm_serial));
      PetscCall(DMSetOptionsPrefix(dm_serial, "serial_"));

      /* We just test/demonstrate DM is indeed distributed - unneeded in the application code */
      PetscCall(DMPlexIsDistributed(dm_serial, &flg));
      PetscCall(PetscPrintf(comm, "Loaded mesh distributed? %s\n", PetscBools[flg]));

      PetscCall(DMViewFromOptions(dm_serial, NULL, "-dm_view"));
      PetscCall(PetscViewerCGNSOpen(comm, user.infile, FILE_MODE_READ, &viewer));
      PetscCall(DMGetGlobalVector(dm_serial, &V_serial));
      PetscCall(VecLoad(V_serial, viewer));
      PetscCall(PetscViewerDestroy(&viewer));
      PetscCallMPI(MPI_Comm_free(&comm));
    }
    PetscCallMPI(MPI_Barrier(PETSC_COMM_WORLD));
  }

  for (PetscInt i = 0; i < user.ntimes; i++) {
    if (i == 0) {
      /* Use infile for the initial load */
      infilename = user.infile;
    } else {
      /* Use outfile for all I/O except the very initial load */
      infilename = user.outfile;
    }

    if (user.heterogeneous) {
      mycolor = (PetscMPIInt)(grank > user.ntimes - i);
    } else {
      mycolor = (PetscMPIInt)0;
    }
    PetscCallMPI(MPI_Comm_split(PETSC_COMM_WORLD, mycolor, grank, &comm));

    if (mycolor == 0) {
      /* Load/Save only on processes with mycolor == 0 */
      DM          dm;
      Vec         V;
      PetscViewer viewer;
      PetscMPIInt comm_size;
      const char *name;
      PetscReal   time;
      PetscBool   set;

      PetscCallMPI(MPI_Comm_size(comm, &comm_size));
      PetscCall(PetscPrintf(comm, "Begin cycle %" PetscInt_FMT ", comm size %d\n", i, comm_size));

      // Load DM from CGNS file
      PetscCall(ReadCGNSDM(comm, infilename, &dm));
      PetscCall(DMSetOptionsPrefix(dm, "loaded_"));
      PetscCall(DMViewFromOptions(dm, NULL, "-dm_view"));

      // Load solution from CGNS file
      PetscCall(PetscViewerCGNSOpen(comm, infilename, FILE_MODE_READ, &viewer));
      PetscCall(DMGetGlobalVector(dm, &V));
      PetscCall(PetscViewerCGNSSetSolutionIndex(viewer, 1));
      { // Test GetSolutionIndex, not needed in application code
        PetscInt solution_index;
        PetscCall(PetscViewerCGNSGetSolutionIndex(viewer, &solution_index));
        PetscCheck(solution_index == 1, comm, PETSC_ERR_ARG_INCOMP, "Returned solution index wrong.");
      }
      PetscCall(PetscViewerCGNSGetSolutionName(viewer, &name));
      PetscCall(PetscViewerCGNSGetSolutionTime(viewer, &time, &set));
      PetscCheck(set, comm, PETSC_ERR_RETURN, "Time data wasn't set!");
      PetscCall(PetscPrintf(comm, "Solution Name: %s, and time %g\n", name, time));
      PetscCall(VecLoad(V, viewer));
      PetscCall(PetscViewerDestroy(&viewer));

      // Verify loaded solution against serial solution
      PetscCall(VerifyLoadedSolution(dm_serial, V_serial, dm, V, 100 * PETSC_MACHINE_EPSILON));

      // Write loaded solution to CGNS file
      PetscCall(PetscViewerCGNSOpen(comm, user.outfile, FILE_MODE_WRITE, &viewer));
      PetscCall(VecView(V, viewer));
      PetscCall(PetscViewerDestroy(&viewer));

      PetscCall(DMRestoreGlobalVector(dm, &V));
      PetscCall(DMDestroy(&dm));
      PetscCall(PetscPrintf(comm, "End   cycle %" PetscInt_FMT "\n--------\n", i));
    }
    PetscCallMPI(MPI_Comm_free(&comm));
    PetscCallMPI(MPI_Barrier(PETSC_COMM_WORLD));
  }

  if (V_serial && dm_serial) PetscCall(DMRestoreGlobalVector(dm_serial, &V_serial));
  if (dm_serial) PetscCall(DMDestroy(&dm_serial));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST
  build:
    requires: cgns
  testset:
    suffix: cgns
    requires: !complex
    nsize: 4
    args: -infile ${wPETSC_DIR}/share/petsc/datafiles/meshes/2x2x2_Q3_wave.cgns -outfile 2x2x2_Q3_wave_output.cgns
    args: -dm_plex_cgns_parallel -ntimes 3 -heterogeneous -serial_dm_view -loaded_dm_view -redistributed_dm_distribute
    test:
      # this partitioner should not shuffle anything, it should yield the same partitioning as the XDMF reader - added just for testing
      suffix: simple
      args: -petscpartitioner_type simple
    test:
      suffix: parmetis
      requires: parmetis
      args: -petscpartitioner_type parmetis
    test:
      suffix: ptscotch
      requires: ptscotch
      args: -petscpartitioner_type ptscotch

TEST*/
