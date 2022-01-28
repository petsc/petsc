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

  ierr = PetscOptionsBegin(comm, "", "Meshing Problem Options", "DMPLEX");CHKERRQ(ierr);
  ierr = PetscOptionsBool("-compare", "Compare the meshes using DMPlexEqual() and DMCompareLabels()", EX, options->compare, &options->compare, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-compare_labels", "Compare labels in the meshes using DMCompareLabels()", "ex55.c", options->compare_labels, &options->compare_labels, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-compare_boundary", "Check label I/O via boundary vertex coordinates", "ex55.c", options->compare_boundary, &options->compare_boundary, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-compare_pre_post", "Compare labels loaded before distribution with those loaded after distribution", "ex55.c", options->compare_pre_post, &options->compare_pre_post, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsString("-outfile", "Output mesh file", EX, options->outfile, options->outfile, sizeof(options->outfile), NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-use_low_level_functions", "Use low level functions for viewing and loading", EX, options->use_low_level_functions, &options->use_low_level_functions, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-distribute_after_topo_load", "Distribute topology right after DMPlexTopologyLoad(), if use_low_level_functions=true", EX, options->distribute_after_topo_load, &options->distribute_after_topo_load, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-verbose", "Verbose printing", EX, options->verbose, &options->verbose, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
};

static PetscErrorCode CreateMesh(AppCtx *options, DM *newdm)
{
  DM             dm;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMCreate(options->comm, &dm);CHKERRQ(ierr);
  ierr = DMSetType(dm, DMPLEX);CHKERRQ(ierr);
  ierr = DMSetFromOptions(dm);CHKERRQ(ierr);
  ierr = PetscObjectGetName((PetscObject)dm, &options->meshname);CHKERRQ(ierr);
  ierr = DMViewFromOptions(dm, NULL, "-dm_view");CHKERRQ(ierr);
  *newdm = dm;
  PetscFunctionReturn(0);
}

static PetscErrorCode SaveMesh(AppCtx *options, DM dm)
{
  PetscViewer    v;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscViewerHDF5Open(PetscObjectComm((PetscObject) dm), options->outfile, FILE_MODE_WRITE, &v);CHKERRQ(ierr);
  if (options->use_low_level_functions) {
    ierr = DMPlexTopologyView(dm, v);CHKERRQ(ierr);
    ierr = DMPlexCoordinatesView(dm, v);CHKERRQ(ierr);
    ierr = DMPlexLabelsView(dm, v);CHKERRQ(ierr);
  } else {
    ierr = DMView(dm, v);CHKERRQ(ierr);
  }
  ierr = PetscViewerDestroy(&v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

typedef enum {NONE=0, PRE_DIST=1, POST_DIST=2} AuxObjLoadMode;

static PetscErrorCode LoadMeshLowLevel(AppCtx *options, PetscViewer v, PetscBool explicitDistribute, AuxObjLoadMode mode, DM *newdm)
{
  DM              dm;
  PetscSF         sfXC;
  PetscErrorCode  ierr;

  PetscFunctionBeginUser;
  ierr = DMCreate(options->comm, &dm);CHKERRQ(ierr);
  ierr = DMSetType(dm, DMPLEX);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) dm, options->meshname);CHKERRQ(ierr);
  ierr = DMPlexTopologyLoad(dm, v, &sfXC);CHKERRQ(ierr);
  if (mode == PRE_DIST) {
    ierr = DMPlexCoordinatesLoad(dm, v, sfXC);CHKERRQ(ierr);
    ierr = DMPlexLabelsLoad(dm, v, sfXC);CHKERRQ(ierr);
  }
  if (explicitDistribute) {
    DM      dmdist;
    PetscSF sfXB = sfXC, sfBC;

    ierr = DMPlexDistribute(dm, 0, &sfBC, &dmdist);CHKERRQ(ierr);
    if (dmdist) {
      const char *name;

      ierr = PetscObjectGetName((PetscObject) dm, &name);CHKERRQ(ierr);
      ierr = PetscObjectSetName((PetscObject) dmdist, name);CHKERRQ(ierr);
      ierr = PetscSFCompose(sfXB, sfBC, &sfXC);CHKERRQ(ierr);
      ierr = PetscSFDestroy(&sfXB);CHKERRQ(ierr);
      ierr = PetscSFDestroy(&sfBC);CHKERRQ(ierr);
      ierr = DMDestroy(&dm);CHKERRQ(ierr);
      dm   = dmdist;
    }
  }
  if (mode == POST_DIST) {
    ierr = DMPlexCoordinatesLoad(dm, v, sfXC);CHKERRQ(ierr);
    ierr = DMPlexLabelsLoad(dm, v, sfXC);CHKERRQ(ierr);
  }
  ierr = PetscSFDestroy(&sfXC);CHKERRQ(ierr);
  *newdm = dm;
  PetscFunctionReturn(0);
}

static PetscErrorCode LoadMesh(AppCtx *options, DM *dmnew)
{
  DM             dm;
  PetscViewer    v;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscViewerHDF5Open(options->comm, options->outfile, FILE_MODE_READ, &v);CHKERRQ(ierr);
  if (options->use_low_level_functions) {
    if (options->compare_pre_post) {
      DM dm0;

      ierr = LoadMeshLowLevel(options, v, PETSC_TRUE, PRE_DIST, &dm0);CHKERRQ(ierr);
      ierr = LoadMeshLowLevel(options, v, PETSC_TRUE, POST_DIST, &dm);CHKERRQ(ierr);
      ierr = DMCompareLabels(dm0, dm, NULL, NULL);CHKERRQ(ierr);
      ierr = DMDestroy(&dm0);CHKERRQ(ierr);
    } else {
      ierr = LoadMeshLowLevel(options, v, options->distribute_after_topo_load, POST_DIST, &dm);CHKERRQ(ierr);
    }
  } else {
    ierr = DMCreate(options->comm, &dm);CHKERRQ(ierr);
    ierr = DMSetType(dm, DMPLEX);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) dm, options->meshname);CHKERRQ(ierr);
    ierr = DMLoad(dm, v);CHKERRQ(ierr);
  }
  ierr = PetscViewerDestroy(&v);CHKERRQ(ierr);

  ierr = DMSetOptionsPrefix(dm, "load_");CHKERRQ(ierr);
  ierr = DMSetFromOptions(dm);CHKERRQ(ierr);
  ierr = DMViewFromOptions(dm, NULL, "-dm_view");CHKERRQ(ierr);
  *dmnew = dm;
  PetscFunctionReturn(0);
}

static PetscErrorCode CompareMeshes(AppCtx *options, DM dm0, DM dm1)
{
  PetscBool       flg;
  PetscErrorCode  ierr;

  PetscFunctionBeginUser;
  if (options->compare) {
    ierr = DMPlexEqual(dm0, dm1, &flg);CHKERRQ(ierr);
    if (!flg) SETERRQ(options->comm, PETSC_ERR_ARG_INCOMP, "DMs are not equal");
    ierr = PetscPrintf(options->comm,"DMs equal\n");CHKERRQ(ierr);
  }
  if (options->compare_labels) {
    ierr = DMCompareLabels(dm0, dm1, NULL, NULL);CHKERRQ(ierr);
    ierr = PetscPrintf(options->comm,"DMLabels equal\n");CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MarkBoundaryVertices(DM dm, PetscInt value, DMLabel *label)
{
  DMLabel         l;
  IS              points;
  const PetscInt *idx;
  PetscInt        i, n;
  PetscErrorCode  ierr;

  PetscFunctionBeginUser;
  ierr = DMLabelCreate(PetscObjectComm((PetscObject)dm), LABEL_NAME, &l);CHKERRQ(ierr);
  ierr = DMPlexMarkBoundaryFaces(dm, value, l);CHKERRQ(ierr);
  ierr = DMPlexLabelComplete(dm, l);CHKERRQ(ierr);
  ierr = DMLabelGetStratumIS(l, value, &points);CHKERRQ(ierr);

  ierr = ISGetLocalSize(points, &n);CHKERRQ(ierr);
  ierr = ISGetIndices(points, &idx);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    const PetscInt p = idx[i];
    PetscInt       d;

    ierr = DMPlexGetPointDepth(dm, p, &d);CHKERRQ(ierr);
    if (d != 0) {
      ierr = DMLabelClearValue(l, p, value);CHKERRQ(ierr);
    }
  }
  ierr = ISRestoreIndices(points, &idx);CHKERRQ(ierr);
  ierr = ISDestroy(&points);CHKERRQ(ierr);
  *label = l;
  PetscFunctionReturn(0);
}

static PetscErrorCode VertexCoordinatesToAll(DM dm, IS vertices, Vec *allCoords)
{
  Vec             coords, allCoords_;
  VecScatter      sc;
  MPI_Comm        comm;
  PetscErrorCode  ierr;

  PetscFunctionBeginUser;
  ierr = PetscObjectGetComm((PetscObject)dm, &comm);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocalSetUp(dm);CHKERRQ(ierr);
  if (vertices) {
    ierr = DMGetCoordinatesLocalTuple(dm, vertices, NULL, &coords);CHKERRQ(ierr);
  } else {
    ierr = VecCreateSeq(PETSC_COMM_SELF, 0, &coords);CHKERRQ(ierr);
  }
  {
    PetscInt  n;
    Vec       mpivec;

    ierr = VecGetLocalSize(coords, &n);CHKERRQ(ierr);
    ierr = VecCreateMPI(comm, n, PETSC_DECIDE, &mpivec);CHKERRQ(ierr);
    ierr = VecCopy(coords, mpivec);CHKERRQ(ierr);
    ierr = VecDestroy(&coords);CHKERRQ(ierr);
    coords = mpivec;
  }

  ierr = VecScatterCreateToAll(coords, &sc, &allCoords_);CHKERRQ(ierr);
  ierr = VecScatterBegin(sc,coords,allCoords_,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(sc,coords,allCoords_,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&sc);CHKERRQ(ierr);
  ierr = VecDestroy(&coords);CHKERRQ(ierr);
  *allCoords = allCoords_;
  PetscFunctionReturn(0);
}

/* Compute boundary label, remember boundary vertices using coordinates, save and load label, check it is defined on the original boundary vertices */
static PetscErrorCode DMAddBoundaryLabel_GetCoordinateRepresentation(DM dm, Vec *allCoords)
{
  DMLabel         label;
  IS              vertices;
  PetscErrorCode  ierr;

  PetscFunctionBeginUser;
  ierr = MarkBoundaryVertices(dm, LABEL_VALUE, &label);CHKERRQ(ierr);
  ierr = DMLabelGetStratumIS(label, LABEL_VALUE, &vertices);CHKERRQ(ierr);
  ierr = VertexCoordinatesToAll(dm, vertices, allCoords);CHKERRQ(ierr);
  ierr = DMAddLabel(dm, label);CHKERRQ(ierr);
  ierr = DMLabelDestroy(&label);CHKERRQ(ierr);
  ierr = ISDestroy(&vertices);CHKERRQ(ierr);
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
  PetscErrorCode  ierr;

  PetscFunctionBeginUser;
  ierr = PetscObjectGetComm((PetscObject)dm, &comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRMPI(ierr);
  ierr = DMGetLabel(dm, LABEL_NAME, &label);CHKERRQ(ierr);
  if (!label) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Label \"%s\" was not loaded", LABEL_NAME);
  {
    PetscInt pStart, pEnd;

    ierr = DMPlexGetChart(dm, &pStart, &pEnd);CHKERRQ(ierr);
    ierr = DMLabelCreateIndex(label, pStart, pEnd);CHKERRQ(ierr);
  }
  ierr = DMPlexFindVertices(dm, allCoords, 0.0, &pointsIS);CHKERRQ(ierr);
  ierr = ISGetIndices(pointsIS, &points);CHKERRQ(ierr);
  ierr = ISGetLocalSize(pointsIS, &n);CHKERRQ(ierr);
  if (user->verbose) {ierr = DMLabelView(label, PETSC_VIEWER_STDOUT_(comm));CHKERRQ(ierr);}
  for (i=0; i<n; i++) {
    const PetscInt  p = points[i];
    PetscBool       has;
    PetscInt        v;

    if (p < 0) continue;
    ierr = DMLabelHasPoint(label, p, &has);CHKERRQ(ierr);
    if (!has) {
      ierr = PetscSynchronizedFPrintf(comm, PETSC_STDERR, "[%d] Label does not have point %D\n", rank, p);CHKERRQ(ierr);
      fail = PETSC_TRUE;
      continue;
    }
    ierr = DMLabelGetValue(label, p, &v);CHKERRQ(ierr);
    if (v != LABEL_VALUE) {
      ierr = PetscSynchronizedFPrintf(comm, PETSC_STDERR, "Point %D has bad value %D", p, v);CHKERRQ(ierr);
      fail = PETSC_TRUE;
      continue;
    }
    if (user->verbose) {ierr = PetscSynchronizedPrintf(comm, "[%d] OK point %D\n", rank, p);CHKERRQ(ierr);}
  }
  ierr = PetscSynchronizedFlush(comm, PETSC_STDOUT);CHKERRQ(ierr);
  ierr = PetscSynchronizedFlush(comm, PETSC_STDERR);CHKERRQ(ierr);
  ierr = ISRestoreIndices(pointsIS, &points);CHKERRQ(ierr);
  ierr = ISDestroy(&pointsIS);CHKERRQ(ierr);
  ierr = MPI_Allreduce(MPI_IN_PLACE, &fail, 1, MPIU_BOOL, MPI_LOR, comm);CHKERRMPI(ierr);
  if (fail) SETERRQ(comm, PETSC_ERR_PLIB, "Label \"%s\" was not loaded correctly - see details above", LABEL_NAME);
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM             dm, dmnew;
  AppCtx         user;
  Vec            allCoords = NULL;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help);if (ierr) return ierr;
  ierr = ProcessOptions(PETSC_COMM_WORLD, &user);CHKERRQ(ierr);
  ierr = CreateMesh(&user, &dm);CHKERRQ(ierr);
  if (user.compare_boundary) {
    ierr = DMAddBoundaryLabel_GetCoordinateRepresentation(dm, &allCoords);CHKERRQ(ierr);
  }
  ierr = SaveMesh(&user, dm);CHKERRQ(ierr);
  ierr = LoadMesh(&user, &dmnew);CHKERRQ(ierr);
  ierr = CompareMeshes(&user, dm, dmnew);CHKERRQ(ierr);
  if (user.compare_boundary) {
    ierr = DMGetBoundaryLabel_CompareWithCoordinateRepresentation(&user, dmnew, allCoords);CHKERRQ(ierr);
  }
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = DMDestroy(&dmnew);CHKERRQ(ierr);
  ierr = VecDestroy(&allCoords);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
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
    args: -dm_distribute -dm_plex_interpolate
    args: -load_dm_plex_check_all
    args: -use_low_level_functions {{0 1}} -load_dm_distribute -compare_boundary
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
    args: -dm_distribute -dm_plex_interpolate
    args: -load_dm_plex_check_all
    args: -use_low_level_functions -distribute_after_topo_load -compare_boundary
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
    args: -dm_distribute -dm_plex_interpolate
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
