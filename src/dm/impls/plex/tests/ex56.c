#include <petscdmplex.h>
#include <petscviewerhdf5.h>
#include <petscsf.h>

static const char help[] = "Load and save the mesh to the native HDF5 format\n\n";
static const char EX[] = "ex56.c";
static const char LABEL_NAME[] = "BoundaryVertices";
static const PetscInt LABEL_VALUE = 12345;
typedef struct {
  MPI_Comm    comm;
  const char *meshname;                     /* Mesh name */
  PetscBool   compare;                      /* Compare the meshes using DMPlexEqual() and DMCompareLabels() */
  PetscBool   compare_labels;               /* Compare labels in the meshes using DMCompareLabels() */
  PetscBool   compare_boundary;             /* Check label I/O via boundary vertex coordinates */
  PetscBool   compare_pre_post;             /* Compare labels loaded before distribution with those loaded after distribution */
  char        outfile[PETSC_MAX_PATH_LEN];  /* Output file */
  PetscBool   use_low_level_functions;      /* Use low level functions for viewing and loading */
  //TODO This is meant as temporary option; can be removed once we have full parallel loading in place
  PetscBool   distribute_after_topo_load;   /* Distribute topology right after DMPlexTopologyLoad(), if use_low_level_functions=true */
  PetscBool   verbose;
} AppCtx;

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  options->comm                       = comm;
  options->compare                    = PETSC_FALSE;
  options->compare_labels             = PETSC_FALSE;
  options->compare_boundary           = PETSC_FALSE;
  options->compare_pre_post           = PETSC_FALSE;
  options->outfile[0]                 = '\0';
  options->use_low_level_functions    = PETSC_FALSE;
  options->distribute_after_topo_load = PETSC_FALSE;
  options->verbose                    = PETSC_FALSE;

  ierr = PetscOptionsBegin(comm, "", "Meshing Problem Options", "DMPLEX");PetscCall(ierr);
  PetscCall(PetscOptionsBool("-compare", "Compare the meshes using DMPlexEqual() and DMCompareLabels()", EX, options->compare, &options->compare, NULL));
  PetscCall(PetscOptionsBool("-compare_labels", "Compare labels in the meshes using DMCompareLabels()", "ex55.c", options->compare_labels, &options->compare_labels, NULL));
  PetscCall(PetscOptionsBool("-compare_boundary", "Check label I/O via boundary vertex coordinates", "ex55.c", options->compare_boundary, &options->compare_boundary, NULL));
  PetscCall(PetscOptionsBool("-compare_pre_post", "Compare labels loaded before distribution with those loaded after distribution", "ex55.c", options->compare_pre_post, &options->compare_pre_post, NULL));
  PetscCall(PetscOptionsString("-outfile", "Output mesh file", EX, options->outfile, options->outfile, sizeof(options->outfile), NULL));
  PetscCall(PetscOptionsBool("-use_low_level_functions", "Use low level functions for viewing and loading", EX, options->use_low_level_functions, &options->use_low_level_functions, NULL));
  PetscCall(PetscOptionsBool("-distribute_after_topo_load", "Distribute topology right after DMPlexTopologyLoad(), if use_low_level_functions=true", EX, options->distribute_after_topo_load, &options->distribute_after_topo_load, NULL));
  PetscCall(PetscOptionsBool("-verbose", "Verbose printing", EX, options->verbose, &options->verbose, NULL));
  ierr = PetscOptionsEnd();PetscCall(ierr);
  PetscFunctionReturn(0);
};

static PetscErrorCode CreateMesh(AppCtx *options, DM *newdm)
{
  DM             dm;

  PetscFunctionBeginUser;
  PetscCall(DMCreate(options->comm, &dm));
  PetscCall(DMSetType(dm, DMPLEX));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(PetscObjectGetName((PetscObject)dm, &options->meshname));
  PetscCall(DMViewFromOptions(dm, NULL, "-dm_view"));
  *newdm = dm;
  PetscFunctionReturn(0);
}

static PetscErrorCode SaveMesh(AppCtx *options, DM dm)
{
  PetscViewer    v;

  PetscFunctionBeginUser;
  PetscCall(PetscViewerHDF5Open(PetscObjectComm((PetscObject) dm), options->outfile, FILE_MODE_WRITE, &v));
  if (options->use_low_level_functions) {
    PetscCall(DMPlexTopologyView(dm, v));
    PetscCall(DMPlexCoordinatesView(dm, v));
    PetscCall(DMPlexLabelsView(dm, v));
  } else {
    PetscCall(DMView(dm, v));
  }
  PetscCall(PetscViewerDestroy(&v));
  PetscFunctionReturn(0);
}

typedef enum {NONE=0, PRE_DIST=1, POST_DIST=2} AuxObjLoadMode;

static PetscErrorCode LoadMeshLowLevel(AppCtx *options, PetscViewer v, PetscBool explicitDistribute, AuxObjLoadMode mode, DM *newdm)
{
  DM              dm;
  PetscSF         sfXC;

  PetscFunctionBeginUser;
  PetscCall(DMCreate(options->comm, &dm));
  PetscCall(DMSetType(dm, DMPLEX));
  PetscCall(PetscObjectSetName((PetscObject) dm, options->meshname));
  PetscCall(DMPlexTopologyLoad(dm, v, &sfXC));
  if (mode == PRE_DIST) {
    PetscCall(DMPlexCoordinatesLoad(dm, v, sfXC));
    PetscCall(DMPlexLabelsLoad(dm, v, sfXC));
  }
  if (explicitDistribute) {
    DM      dmdist;
    PetscSF sfXB = sfXC, sfBC;

    PetscCall(DMPlexDistribute(dm, 0, &sfBC, &dmdist));
    if (dmdist) {
      const char *name;

      PetscCall(PetscObjectGetName((PetscObject) dm, &name));
      PetscCall(PetscObjectSetName((PetscObject) dmdist, name));
      PetscCall(PetscSFCompose(sfXB, sfBC, &sfXC));
      PetscCall(PetscSFDestroy(&sfXB));
      PetscCall(PetscSFDestroy(&sfBC));
      PetscCall(DMDestroy(&dm));
      dm   = dmdist;
    }
  }
  if (mode == POST_DIST) {
    PetscCall(DMPlexCoordinatesLoad(dm, v, sfXC));
    PetscCall(DMPlexLabelsLoad(dm, v, sfXC));
  }
  PetscCall(PetscSFDestroy(&sfXC));
  *newdm = dm;
  PetscFunctionReturn(0);
}

static PetscErrorCode LoadMesh(AppCtx *options, DM *dmnew)
{
  DM             dm;
  PetscViewer    v;

  PetscFunctionBeginUser;
  PetscCall(PetscViewerHDF5Open(options->comm, options->outfile, FILE_MODE_READ, &v));
  if (options->use_low_level_functions) {
    if (options->compare_pre_post) {
      DM dm0;

      PetscCall(LoadMeshLowLevel(options, v, PETSC_TRUE, PRE_DIST, &dm0));
      PetscCall(LoadMeshLowLevel(options, v, PETSC_TRUE, POST_DIST, &dm));
      PetscCall(DMCompareLabels(dm0, dm, NULL, NULL));
      PetscCall(DMDestroy(&dm0));
    } else {
      PetscCall(LoadMeshLowLevel(options, v, options->distribute_after_topo_load, POST_DIST, &dm));
    }
  } else {
    PetscCall(DMCreate(options->comm, &dm));
    PetscCall(DMSetType(dm, DMPLEX));
    PetscCall(PetscObjectSetName((PetscObject) dm, options->meshname));
    PetscCall(DMLoad(dm, v));
  }
  PetscCall(PetscViewerDestroy(&v));

  PetscCall(DMSetOptionsPrefix(dm, "load_"));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMViewFromOptions(dm, NULL, "-dm_view"));
  *dmnew = dm;
  PetscFunctionReturn(0);
}

static PetscErrorCode CompareMeshes(AppCtx *options, DM dm0, DM dm1)
{
  PetscBool       flg;

  PetscFunctionBeginUser;
  if (options->compare) {
    PetscCall(DMPlexEqual(dm0, dm1, &flg));
    PetscCheck(flg,options->comm, PETSC_ERR_ARG_INCOMP, "DMs are not equal");
    PetscCall(PetscPrintf(options->comm,"DMs equal\n"));
  }
  if (options->compare_labels) {
    PetscCall(DMCompareLabels(dm0, dm1, NULL, NULL));
    PetscCall(PetscPrintf(options->comm,"DMLabels equal\n"));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MarkBoundaryVertices(DM dm, PetscInt value, DMLabel *label)
{
  DMLabel         l;
  IS              points;
  const PetscInt *idx;
  PetscInt        i, n;

  PetscFunctionBeginUser;
  PetscCall(DMLabelCreate(PetscObjectComm((PetscObject)dm), LABEL_NAME, &l));
  PetscCall(DMPlexMarkBoundaryFaces(dm, value, l));
  PetscCall(DMPlexLabelComplete(dm, l));
  PetscCall(DMLabelGetStratumIS(l, value, &points));

  PetscCall(ISGetLocalSize(points, &n));
  PetscCall(ISGetIndices(points, &idx));
  for (i=0; i<n; i++) {
    const PetscInt p = idx[i];
    PetscInt       d;

    PetscCall(DMPlexGetPointDepth(dm, p, &d));
    if (d != 0) {
      PetscCall(DMLabelClearValue(l, p, value));
    }
  }
  PetscCall(ISRestoreIndices(points, &idx));
  PetscCall(ISDestroy(&points));
  *label = l;
  PetscFunctionReturn(0);
}

static PetscErrorCode VertexCoordinatesToAll(DM dm, IS vertices, Vec *allCoords)
{
  Vec             coords, allCoords_;
  VecScatter      sc;
  MPI_Comm        comm;

  PetscFunctionBeginUser;
  PetscCall(PetscObjectGetComm((PetscObject)dm, &comm));
  PetscCall(DMGetCoordinatesLocalSetUp(dm));
  if (vertices) {
    PetscCall(DMGetCoordinatesLocalTuple(dm, vertices, NULL, &coords));
  } else {
    PetscCall(VecCreateSeq(PETSC_COMM_SELF, 0, &coords));
  }
  {
    PetscInt  n;
    Vec       mpivec;

    PetscCall(VecGetLocalSize(coords, &n));
    PetscCall(VecCreateMPI(comm, n, PETSC_DECIDE, &mpivec));
    PetscCall(VecCopy(coords, mpivec));
    PetscCall(VecDestroy(&coords));
    coords = mpivec;
  }

  PetscCall(VecScatterCreateToAll(coords, &sc, &allCoords_));
  PetscCall(VecScatterBegin(sc,coords,allCoords_,INSERT_VALUES,SCATTER_FORWARD));
  PetscCall(VecScatterEnd(sc,coords,allCoords_,INSERT_VALUES,SCATTER_FORWARD));
  PetscCall(VecScatterDestroy(&sc));
  PetscCall(VecDestroy(&coords));
  *allCoords = allCoords_;
  PetscFunctionReturn(0);
}

/* Compute boundary label, remember boundary vertices using coordinates, save and load label, check it is defined on the original boundary vertices */
static PetscErrorCode DMAddBoundaryLabel_GetCoordinateRepresentation(DM dm, Vec *allCoords)
{
  DMLabel         label;
  IS              vertices;

  PetscFunctionBeginUser;
  PetscCall(MarkBoundaryVertices(dm, LABEL_VALUE, &label));
  PetscCall(DMLabelGetStratumIS(label, LABEL_VALUE, &vertices));
  PetscCall(VertexCoordinatesToAll(dm, vertices, allCoords));
  PetscCall(DMAddLabel(dm, label));
  PetscCall(DMLabelDestroy(&label));
  PetscCall(ISDestroy(&vertices));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMGetBoundaryLabel_CompareWithCoordinateRepresentation(AppCtx *user, DM dm, Vec allCoords)
{
  DMLabel         label;
  IS              pointsIS;
  const PetscInt *points;
  PetscInt        i, n;
  PetscBool       fail = PETSC_FALSE;
  MPI_Comm        comm;
  PetscMPIInt     rank;

  PetscFunctionBeginUser;
  PetscCall(PetscObjectGetComm((PetscObject)dm, &comm));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCall(DMGetLabel(dm, LABEL_NAME, &label));
  PetscCheck(label,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Label \"%s\" was not loaded", LABEL_NAME);
  {
    PetscInt pStart, pEnd;

    PetscCall(DMPlexGetChart(dm, &pStart, &pEnd));
    PetscCall(DMLabelCreateIndex(label, pStart, pEnd));
  }
  PetscCall(DMPlexFindVertices(dm, allCoords, 0.0, &pointsIS));
  PetscCall(ISGetIndices(pointsIS, &points));
  PetscCall(ISGetLocalSize(pointsIS, &n));
  if (user->verbose) PetscCall(DMLabelView(label, PETSC_VIEWER_STDOUT_(comm)));
  for (i=0; i<n; i++) {
    const PetscInt  p = points[i];
    PetscBool       has;
    PetscInt        v;

    if (p < 0) continue;
    PetscCall(DMLabelHasPoint(label, p, &has));
    if (!has) {
      PetscCall(PetscSynchronizedFPrintf(comm, PETSC_STDERR, "[%d] Label does not have point %D\n", rank, p));
      fail = PETSC_TRUE;
      continue;
    }
    PetscCall(DMLabelGetValue(label, p, &v));
    if (v != LABEL_VALUE) {
      PetscCall(PetscSynchronizedFPrintf(comm, PETSC_STDERR, "Point %D has bad value %D", p, v));
      fail = PETSC_TRUE;
      continue;
    }
    if (user->verbose) PetscCall(PetscSynchronizedPrintf(comm, "[%d] OK point %D\n", rank, p));
  }
  PetscCall(PetscSynchronizedFlush(comm, PETSC_STDOUT));
  PetscCall(PetscSynchronizedFlush(comm, PETSC_STDERR));
  PetscCall(ISRestoreIndices(pointsIS, &points));
  PetscCall(ISDestroy(&pointsIS));
  PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE, &fail, 1, MPIU_BOOL, MPI_LOR, comm));
  PetscCheck(!fail,comm, PETSC_ERR_PLIB, "Label \"%s\" was not loaded correctly - see details above", LABEL_NAME);
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM             dm, dmnew;
  AppCtx         user;
  Vec            allCoords = NULL;

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(ProcessOptions(PETSC_COMM_WORLD, &user));
  PetscCall(CreateMesh(&user, &dm));
  if (user.compare_boundary) {
    PetscCall(DMAddBoundaryLabel_GetCoordinateRepresentation(dm, &allCoords));
  }
  PetscCall(SaveMesh(&user, dm));
  PetscCall(LoadMesh(&user, &dmnew));
  PetscCall(CompareMeshes(&user, dm, dmnew));
  if (user.compare_boundary) {
    PetscCall(DMGetBoundaryLabel_CompareWithCoordinateRepresentation(&user, dmnew, allCoords));
  }
  PetscCall(DMDestroy(&dm));
  PetscCall(DMDestroy(&dmnew));
  PetscCall(VecDestroy(&allCoords));
  PetscCall(PetscFinalize());
  return 0;
}

//TODO we can -compare once the new parallel topology format is in place
/*TEST
  build:
    requires: hdf5

  # load old format, save in new format, reload, distribute
  testset:
    suffix: 1
    requires: !complex datafilespath
    args: -dm_plex_name plex
    args: -dm_plex_check_all -dm_plex_view_hdf5_storage_version 2.0.0
    args: -dm_plex_interpolate
    args: -load_dm_plex_check_all
    args: -use_low_level_functions {{0 1}} -compare_boundary
    args: -outfile ex56_1.h5
    nsize: {{1 3}}
    test:
      suffix: a
      args: -dm_plex_filename ${DATAFILESPATH}/meshes/hdf5-petsc/petsc-v3.16.0/v1.0.0/annulus-20.h5
    test:
      suffix: b
      TODO: broken
      args: -dm_plex_filename ${DATAFILESPATH}/meshes/hdf5-petsc/petsc-v3.16.0/v1.0.0/barycentricallyrefinedcube.h5
    test:
      suffix: c
      args: -dm_plex_filename ${DATAFILESPATH}/meshes/hdf5-petsc/petsc-v3.16.0/v1.0.0/blockcylinder-50.h5
    test:
      suffix: d
      args: -dm_plex_filename ${DATAFILESPATH}/meshes/hdf5-petsc/petsc-v3.16.0/v1.0.0/cube-hexahedra-refined.h5
    test:
      suffix: e
      args: -dm_plex_filename ${DATAFILESPATH}/meshes/hdf5-petsc/petsc-v3.16.0/v1.0.0/hybrid_hexwedge.h5
    test:
      suffix: f
      args: -dm_plex_filename ${DATAFILESPATH}/meshes/hdf5-petsc/petsc-v3.16.0/v1.0.0/square.h5

  # load old format, save in new format, reload topology, distribute, load geometry and labels
  testset:
    suffix: 2
    requires: !complex datafilespath
    args: -dm_plex_name plex
    args: -dm_plex_check_all -dm_plex_view_hdf5_storage_version 2.0.0
    args: -dm_plex_interpolate
    args: -load_dm_plex_check_all
    args: -use_low_level_functions -load_dm_distribute 0 -distribute_after_topo_load -compare_boundary
    args: -outfile ex56_2.h5
    nsize: 3
    test:
      suffix: a
      args: -dm_plex_filename ${DATAFILESPATH}/meshes/hdf5-petsc/petsc-v3.16.0/v1.0.0/annulus-20.h5
    test:
      suffix: b
      TODO: broken
      args: -dm_plex_filename ${DATAFILESPATH}/meshes/hdf5-petsc/petsc-v3.16.0/v1.0.0/barycentricallyrefinedcube.h5
    test:
      suffix: c
      args: -dm_plex_filename ${DATAFILESPATH}/meshes/hdf5-petsc/petsc-v3.16.0/v1.0.0/blockcylinder-50.h5
    test:
      suffix: d
      args: -dm_plex_filename ${DATAFILESPATH}/meshes/hdf5-petsc/petsc-v3.16.0/v1.0.0/cube-hexahedra-refined.h5
    test:
      suffix: e
      args: -dm_plex_filename ${DATAFILESPATH}/meshes/hdf5-petsc/petsc-v3.16.0/v1.0.0/hybrid_hexwedge.h5
    test:
      suffix: f
      args: -dm_plex_filename ${DATAFILESPATH}/meshes/hdf5-petsc/petsc-v3.16.0/v1.0.0/square.h5

  # load old format, save in new format, reload topology, distribute, load geometry and labels
  testset:
    suffix: 3
    requires: !complex datafilespath
    args: -dm_plex_name plex
    args: -dm_plex_view_hdf5_storage_version 2.0.0
    args: -dm_plex_interpolate -load_dm_distribute 0
    args: -use_low_level_functions -compare_pre_post
    args: -outfile ex56_3.h5
    nsize: 3
    test:
      suffix: a
      args: -dm_plex_filename ${DATAFILESPATH}/meshes/hdf5-petsc/petsc-v3.16.0/v1.0.0/annulus-20.h5
    test:
      suffix: b
      TODO: broken
      args: -dm_plex_filename ${DATAFILESPATH}/meshes/hdf5-petsc/petsc-v3.16.0/v1.0.0/barycentricallyrefinedcube.h5
    test:
      suffix: c
      args: -dm_plex_filename ${DATAFILESPATH}/meshes/hdf5-petsc/petsc-v3.16.0/v1.0.0/blockcylinder-50.h5
    test:
      suffix: d
      args: -dm_plex_filename ${DATAFILESPATH}/meshes/hdf5-petsc/petsc-v3.16.0/v1.0.0/cube-hexahedra-refined.h5
    test:
      suffix: e
      args: -dm_plex_filename ${DATAFILESPATH}/meshes/hdf5-petsc/petsc-v3.16.0/v1.0.0/hybrid_hexwedge.h5
    test:
      suffix: f
      args: -dm_plex_filename ${DATAFILESPATH}/meshes/hdf5-petsc/petsc-v3.16.0/v1.0.0/square.h5
TEST*/
