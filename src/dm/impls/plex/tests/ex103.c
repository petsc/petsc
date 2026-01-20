static char help[] = "Test CGNS parallel load-save-reload cycle, including data and DMLabels\n\n";
// This is a modification of src/dm/impls/plex/tutorials/ex15.c, but with additional tests that don't make sense for a tutorial problem (such as verify FaceLabels)

#include <petscdmlabel.h>
#include <petscdmplex.h>
#include <petscsf.h>
#include <petscviewerhdf5.h>
#define EX "ex103.c"

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
  PetscSF      load_to_serial_sf_;
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

    PetscCall(PetscSFCreate(comm, &load_to_serial_sf_));
    PetscCall(PetscSFSetGraphFromCoordinates(load_to_serial_sf_, num_local_load, num_local_serial, dim, 100 * PETSC_MACHINE_EPSILON, coord_load, coord_serial));
    PetscCall(PetscSFViewFromOptions(load_to_serial_sf_, NULL, "-verify_sf_view"));

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
    PetscCall(PetscSFBcastBegin(load_to_serial_sf_, unit, array_load, array_load_bcast, MPI_REPLACE));
    PetscCall(PetscSFBcastEnd(load_to_serial_sf_, unit, array_load, array_load_bcast, MPI_REPLACE));
    PetscCallMPI(MPI_Type_free(&unit));
    PetscCall(VecRestoreArrayRead(V_load, &array_load));
  }

  if (V_serial) {
    const PetscScalar *array_serial;
    PetscInt           size_local_serial;

    PetscCall(VecGetArrayRead(V_serial, &array_serial));
    PetscCall(VecGetLocalSize(V_serial, &size_local_serial));
    for (PetscInt i = 0; i < size_local_serial; i++) {
      if (PetscAbs(array_serial[i] - array_load_bcast[i]) > tol) PetscCall(PetscPrintf(comm, "DoF %" PetscInt_FMT " is inconsistent. Found %g, expected %g\n", i, (double)array_load_bcast[i], (double)array_serial[i]));
    }
    PetscCall(VecRestoreArrayRead(V_serial, &array_serial));
  }

  PetscCall(PetscFree(array_load_bcast));
  PetscCall(PetscSFDestroy(&load_to_serial_sf_));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Get centroids associated with every Plex point
PetscErrorCode DMPlexGetPointsCentroids(DM dm, PetscReal *points_centroids[])
{
  PetscInt     coords_dim, pStart, pEnd, depth = 0;
  PetscSection coord_section;
  Vec          coords_vec;
  PetscReal   *points_centroids_;

  PetscFunctionBeginUser;
  PetscCall(DMGetCoordinateDim(dm, &coords_dim));
  PetscCall(DMPlexGetChart(dm, &pStart, &pEnd));
  PetscCall(DMGetCoordinateSection(dm, &coord_section));
  PetscCall(DMGetCoordinatesLocal(dm, &coords_vec));

  PetscCall(PetscCalloc1((pEnd - pStart) * coords_dim, &points_centroids_));
  for (PetscInt p = pStart; p < pEnd; p++) {
    PetscScalar *coords = NULL;
    PetscInt     coords_size, num_coords;

    PetscCall(DMPlexVecGetClosureAtDepth(dm, coord_section, coords_vec, p, depth, &coords_size, &coords));
    num_coords = coords_size / coords_dim;
    for (PetscInt c = 0; c < num_coords; c++) {
      for (PetscInt d = 0; d < coords_dim; d++) points_centroids_[p * coords_dim + d] += PetscRealPart(coords[c * coords_dim + d]);
    }
    for (PetscInt d = 0; d < coords_dim; d++) points_centroids_[p * coords_dim + d] /= num_coords;
    PetscCall(DMPlexVecRestoreClosure(dm, coord_section, coords_vec, p, &coords_size, &coords));
  }
  *points_centroids = points_centroids_;
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Verify that the DMLabel in dm_serial is identical to that in dm_load
PetscErrorCode VerifyDMLabels(DM dm_serial, DM dm_load, const char label_name[], PetscSF *serial2loadPointSF)
{
  PetscMPIInt rank;
  MPI_Comm    comm              = PetscObjectComm((PetscObject)dm_load);
  PetscInt    num_values_serial = 0, dim;
  PetscInt   *values_serial     = NULL;
  DMLabel     label_serial      = NULL, label_load;

  PetscFunctionBeginUser;
  PetscCall(DMGetCoordinateDim(dm_load, &dim));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  if (dm_serial) { // Communicate valid label values to all ranks
    IS              serialValuesIS;
    const PetscInt *values_serial_is;

    PetscCheck(rank == 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Rank with serial DM not rank 0");
    PetscCall(DMGetLabel(dm_serial, label_name, &label_serial));
    PetscCall(DMLabelGetNonEmptyStratumValuesIS(label_serial, &serialValuesIS));
    PetscCall(ISGetLocalSize(serialValuesIS, &num_values_serial));

    PetscCall(PetscMalloc1(num_values_serial, &values_serial));
    PetscCall(ISGetIndices(serialValuesIS, &values_serial_is));
    PetscCall(PetscArraycpy(values_serial, values_serial_is, num_values_serial));
    PetscCall(PetscSortInt(num_values_serial, values_serial));
    PetscCall(ISRestoreIndices(serialValuesIS, &values_serial_is));
    PetscCall(ISDestroy(&serialValuesIS));
  }
  PetscCallMPI(MPI_Bcast(&num_values_serial, 1, MPIU_INT, 0, comm));
  if (values_serial == NULL) PetscCall(PetscMalloc1(num_values_serial, &values_serial));
  PetscCallMPI(MPI_Bcast(values_serial, num_values_serial, MPIU_INT, 0, comm));

  IS              loadValuesIS;
  PetscInt        num_values_global;
  const PetscInt *values_global;
  PetscBool       are_values_same = PETSC_TRUE;

  PetscCall(DMGetLabel(dm_load, label_name, &label_load));
  PetscCall(DMLabelGetValueISGlobal(comm, label_load, PETSC_TRUE, &loadValuesIS));
  PetscCall(ISSort(loadValuesIS));
  PetscCall(ISGetLocalSize(loadValuesIS, &num_values_global));
  PetscCall(ISGetIndices(loadValuesIS, &values_global));
  if (num_values_serial != num_values_global) {
    PetscCall(PetscPrintf(comm, "DMLabel '%s': Number of nonempty values in serial DM (%" PetscInt_FMT ") not equal to number of values in global DM (%" PetscInt_FMT ")\n", label_name, num_values_serial, num_values_global));
    are_values_same = PETSC_FALSE;
  }
  PetscCall(PetscPrintf(comm, "DMLabel '%s': serial values:\n", label_name));
  PetscCall(PetscIntView(num_values_serial, values_serial, PETSC_VIEWER_STDOUT_(comm)));
  PetscCall(PetscPrintf(comm, "DMLabel '%s': global values:\n", label_name));
  PetscCall(PetscIntView(num_values_global, values_global, PETSC_VIEWER_STDOUT_(comm)));
  for (PetscInt i = 0; i < num_values_serial; i++) {
    PetscInt loc;
    PetscCall(PetscFindInt(values_serial[i], num_values_global, values_global, &loc));
    if (loc < 0) {
      PetscCall(PetscPrintf(comm, "DMLabel '%s': Label value %" PetscInt_FMT " in serial DM not found in global DM\n", label_name, values_serial[i]));
      are_values_same = PETSC_FALSE;
    }
  }
  PetscCheck(are_values_same, comm, PETSC_ERR_PLIB, "The values in DMLabel are not the same");
  PetscCall(PetscFree(values_serial));

  PetscSF  serial2loadPointSF_;
  PetscInt pStart, pEnd, pStartSerial = -1, pEndSerial = -2;
  PetscInt num_points_load, num_points_serial = 0;
  { // Create SF mapping serialDM points to loadDM points
    PetscReal *points_centroid_load, *points_centroid_serial = NULL;

    if (rank == 0) {
      PetscCall(DMPlexGetChart(dm_serial, &pStartSerial, &pEndSerial));
      num_points_serial = pEndSerial - pStartSerial;
      PetscCall(DMPlexGetPointsCentroids(dm_serial, &points_centroid_serial));
    }
    PetscCall(DMPlexGetChart(dm_load, &pStart, &pEnd));
    num_points_load = pEnd - pStart;
    PetscCall(DMPlexGetPointsCentroids(dm_load, &points_centroid_load));

    PetscCall(PetscSFCreate(comm, &serial2loadPointSF_));
    PetscCall(PetscSFSetGraphFromCoordinates(serial2loadPointSF_, num_points_serial, num_points_load, dim, 100 * PETSC_MACHINE_EPSILON, points_centroid_serial, points_centroid_load));
    PetscCall(PetscObjectSetName((PetscObject)serial2loadPointSF_, "Serial To Loaded DM Points SF"));
    PetscCall(PetscSFViewFromOptions(serial2loadPointSF_, NULL, "-verify_points_sf_view"));
    PetscCall(PetscFree(points_centroid_load));
    PetscCall(PetscFree(points_centroid_serial));
  }

  PetscSection pointSerialSection = NULL;
  PetscInt     npointMaskSerial   = 0;
  PetscBool   *pointMask, *pointMaskSerial = NULL;

  if (rank == 0) {
    const PetscInt *root_degree;
    PetscCall(PetscSectionCreate(PETSC_COMM_SELF, &pointSerialSection));
    PetscCall(PetscSectionSetChart(pointSerialSection, pStartSerial, pEndSerial));
    PetscCall(PetscSFComputeDegreeBegin(serial2loadPointSF_, &root_degree));
    PetscCall(PetscSFComputeDegreeEnd(serial2loadPointSF_, &root_degree));
    for (PetscInt p = 0; p < num_points_serial; p++) PetscCall(PetscSectionSetDof(pointSerialSection, p, root_degree[p]));
    PetscCall(PetscSectionSetUp(pointSerialSection));
    PetscCall(PetscSectionGetStorageSize(pointSerialSection, &npointMaskSerial));

    PetscCall(PetscMalloc1(npointMaskSerial, &pointMaskSerial));
  }
  PetscCall(PetscMalloc1(num_points_load, &pointMask));

  for (PetscInt v = 0; v < num_values_global; v++) {
    PetscInt value     = values_global[v];
    IS       stratumIS = NULL;

    if (pointMaskSerial) PetscCall(PetscArrayzero(pointMaskSerial, npointMaskSerial));
    PetscCall(PetscArrayzero(pointMask, num_points_load));
    PetscCall(DMLabelGetStratumIS(label_load, value, &stratumIS));
    if (stratumIS) {
      PetscInt        num_points;
      const PetscInt *points;

      PetscCall(ISGetLocalSize(stratumIS, &num_points));
      PetscCall(ISGetIndices(stratumIS, &points));
      for (PetscInt p = 0; p < num_points; p++) pointMask[points[p]] = PETSC_TRUE;
      PetscCall(ISRestoreIndices(stratumIS, &points));
      PetscCall(ISDestroy(&stratumIS));
    }
    PetscCall(PetscSFGatherBegin(serial2loadPointSF_, MPI_C_BOOL, pointMask, pointMaskSerial));
    PetscCall(PetscSFGatherEnd(serial2loadPointSF_, MPI_C_BOOL, pointMask, pointMaskSerial));

    if (rank == 0) {
      IS stratumIS = NULL;

      PetscCall(DMLabelGetStratumIS(label_serial, value, &stratumIS));
      if (stratumIS) {
        PetscInt        num_points;
        const PetscInt *points;

        PetscCall(ISSort(stratumIS));
        PetscCall(ISGetLocalSize(stratumIS, &num_points));
        PetscCall(ISGetIndices(stratumIS, &points));
        for (PetscInt p = 0; p < num_points_serial; p++) {
          PetscInt ndof, offset, loc;

          PetscCall(PetscSectionGetDof(pointSerialSection, p, &ndof));
          PetscCall(PetscSectionGetOffset(pointSerialSection, p, &offset));
          PetscCall(PetscFindInt(p, num_points, points, &loc));
          PetscBool serial_has_point = loc >= 0;

          for (PetscInt d = 0; d < ndof; d++) {
            if (serial_has_point != pointMaskSerial[offset + d]) PetscCall(PetscPrintf(comm, "DMLabel '%s': Serial and global DM disagree on point %" PetscInt_FMT " valid for label value %" PetscInt_FMT "\n", label_name, p, value));
          }
        }
        PetscCall(ISRestoreIndices(stratumIS, &points));
        PetscCall(ISDestroy(&stratumIS));
      }
    }
  }
  if (serial2loadPointSF && !*serial2loadPointSF) *serial2loadPointSF = serial2loadPointSF_;
  else PetscCall(PetscSFDestroy(&serial2loadPointSF_));

  PetscCall(ISRestoreIndices(loadValuesIS, &values_global));
  PetscCall(ISDestroy(&loadValuesIS));
  PetscCall(PetscSectionDestroy(&pointSerialSection));
  PetscCall(PetscFree(pointMaskSerial));
  PetscCall(PetscFree(pointMask));
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
      PetscCall(DMSetFromOptions(dm_serial));

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
      mycolor = (PetscMPIInt)(grank < (gsize - i) ? 0 : 1);
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

      // Verify DMLabel values against the serial DM
      PetscCall(VerifyDMLabels(dm_serial, dm, "Face Sets", NULL));

      { // Complete the label so that the writer must sort through non-face points
        DMLabel label;
        PetscCall(DMGetLabel(dm, "Face Sets", &label));
        PetscCall(DMPlexLabelComplete(dm, label));
      }

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
    suffix: cgns_3x3
    requires: !complex
    nsize: 4
    args: -infile ${DATAFILESPATH}/meshes/3x3_Q1.cgns -outfile 3x3_Q1_output.cgns -dm_plex_cgns_parallel -ntimes 3 -heterogeneous -serial_dm_view -loaded_dm_view -redistributed_dm_distribute
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

  testset:
    suffix: cgns_2x2x2
    requires: !complex
    nsize: 4
    args: -infile ${DATAFILESPATH}/meshes/2x2x2_Q3_wave.cgns -outfile 2x2x2_Q3_wave_output.cgns -dm_plex_cgns_parallel -ntimes 3 -heterogeneous -serial_dm_view -loaded_dm_view -redistributed_dm_distribute
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

  test:
    # This file is meant to explicitly have a special case where a partition completely surrounds a boundary element, but does not own it
    requires: !complex
    suffix: cgns_3x3_2
    args: -infile ${DATAFILESPATH}/meshes/3x3_Q1.cgns -outfile 3x3_Q1_output.cgns -dm_plex_cgns_parallel -serial_dm_view -loaded_dm_view -redistributed_dm_distribute -petscpartitioner_type simple
TEST*/
