static char help[] = "Run C version of TetGen to construct and refine a mesh\n\n";

#include <petscdmplex.h>

typedef enum {BOX, CYLINDER} DomainShape;
enum {STAGE_LOAD, STAGE_DISTRIBUTE, STAGE_REFINE, STAGE_OVERLAP};

typedef struct {
  DM            dm;                /* REQUIRED in order to use SNES evaluation functions */
  PetscInt      debug;             /* The debugging level */
  PetscLogEvent createMeshEvent;
  PetscLogStage stages[4];
  /* Domain and mesh definition */
  PetscInt      dim;                          /* The topological mesh dimension */
  PetscBool     interpolate;                  /* Generate intermediate mesh elements */
  PetscReal     refinementLimit;              /* The largest allowable cell volume */
  PetscBool     cellSimplex;                  /* Use simplices or hexes */
  PetscBool     cellWedge;                    /* Use wedges */
  PetscBool     simplex2tensor;               /* Refine simplicials in hexes */
  DomainShape   domainShape;                  /* Shep of the region to be meshed */
  DMBoundaryType periodicity[3];              /* The domain periodicity */
  char          filename[PETSC_MAX_PATH_LEN]; /* Import mesh from file */
  char          bdfilename[PETSC_MAX_PATH_LEN]; /* Import mesh boundary from file */
  PetscBool     testPartition;                /* Use a fixed partitioning for testing */
  PetscInt      overlap;                      /* The cell overlap to use during partitioning */
  PetscBool     testShape;                    /* Test the cell shape quality */
} AppCtx;

PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  const char    *dShapes[2] = {"box", "cylinder"};
  PetscInt       shape, bd;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  options->debug             = 0;
  options->dim               = 2;
  options->interpolate       = PETSC_FALSE;
  options->refinementLimit   = 0.0;
  options->cellSimplex       = PETSC_TRUE;
  options->cellWedge         = PETSC_FALSE;
  options->domainShape       = BOX;
  options->periodicity[0]    = DM_BOUNDARY_NONE;
  options->periodicity[1]    = DM_BOUNDARY_NONE;
  options->periodicity[2]    = DM_BOUNDARY_NONE;
  options->filename[0]       = '\0';
  options->bdfilename[0]     = '\0';
  options->testPartition     = PETSC_FALSE;
  options->overlap           = PETSC_FALSE;
  options->testShape         = PETSC_FALSE;
  options->simplex2tensor    = PETSC_FALSE;

  ierr = PetscOptionsBegin(comm, "", "Meshing Problem Options", "DMPLEX");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-debug", "The debugging level", "ex1.c", options->debug, &options->debug, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dim", "The topological mesh dimension", "ex1.c", options->dim, &options->dim, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-interpolate", "Generate intermediate mesh elements", "ex1.c", options->interpolate, &options->interpolate, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-refinement_limit", "The largest allowable cell volume", "ex1.c", options->refinementLimit, &options->refinementLimit, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-cell_simplex", "Use simplices if true, otherwise hexes", "ex1.c", options->cellSimplex, &options->cellSimplex, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-cell_wedge", "Use wedges if true", "ex1.c", options->cellWedge, &options->cellWedge, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-simplex2tensor", "Refine simplicial cells in tensor product cells", "ex1.c", options->simplex2tensor, &options->simplex2tensor, NULL);CHKERRQ(ierr);
  if (options->simplex2tensor) options->interpolate = PETSC_TRUE;
  shape = options->domainShape;
  ierr = PetscOptionsEList("-domain_shape","The shape of the domain","ex1.c", dShapes, 2, dShapes[options->domainShape], &shape, NULL);CHKERRQ(ierr);
  options->domainShape = (DomainShape) shape;
  bd = options->periodicity[0];
  ierr = PetscOptionsEList("-x_periodicity", "The x-boundary periodicity", "ex1.c", DMBoundaryTypes, 5, DMBoundaryTypes[options->periodicity[0]], &bd, NULL);CHKERRQ(ierr);
  options->periodicity[0] = (DMBoundaryType) bd;
  bd = options->periodicity[1];
  ierr = PetscOptionsEList("-y_periodicity", "The y-boundary periodicity", "ex1.c", DMBoundaryTypes, 5, DMBoundaryTypes[options->periodicity[1]], &bd, NULL);CHKERRQ(ierr);
  options->periodicity[1] = (DMBoundaryType) bd;
  bd = options->periodicity[2];
  ierr = PetscOptionsEList("-z_periodicity", "The z-boundary periodicity", "ex1.c", DMBoundaryTypes, 5, DMBoundaryTypes[options->periodicity[2]], &bd, NULL);CHKERRQ(ierr);
  options->periodicity[2] = (DMBoundaryType) bd;
  ierr = PetscOptionsString("-filename", "The mesh file", "ex1.c", options->filename, options->filename, PETSC_MAX_PATH_LEN, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsString("-bd_filename", "The mesh boundary file", "ex1.c", options->bdfilename, options->bdfilename, PETSC_MAX_PATH_LEN, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-test_partition", "Use a fixed partition for testing", "ex1.c", options->testPartition, &options->testPartition, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-overlap", "The cell overlap for partitioning", "ex1.c", options->overlap, &options->overlap, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-test_shape", "Report cell shape qualities (Jacobian condition numbers)", "ex1.c", options->testShape, &options->testShape, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();

  ierr = PetscLogEventRegister("CreateMesh", DM_CLASSID, &options->createMeshEvent);CHKERRQ(ierr);
  ierr = PetscLogStageRegister("MeshLoad",       &options->stages[STAGE_LOAD]);CHKERRQ(ierr);
  ierr = PetscLogStageRegister("MeshDistribute", &options->stages[STAGE_DISTRIBUTE]);CHKERRQ(ierr);
  ierr = PetscLogStageRegister("MeshRefine",     &options->stages[STAGE_REFINE]);CHKERRQ(ierr);
  ierr = PetscLogStageRegister("MeshOverlap",    &options->stages[STAGE_OVERLAP]);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscInt       dim                  = user->dim;
  PetscBool      interpolate          = user->interpolate;
  PetscReal      refinementLimit      = user->refinementLimit;
  PetscBool      cellSimplex          = user->cellSimplex;
  PetscBool      cellWedge            = user->cellWedge;
  PetscBool      simplex2tensor       = user->simplex2tensor;
  const char    *filename             = user->filename;
  const char    *bdfilename           = user->bdfilename;
  PetscInt       triSizes_n2[2]       = {4, 4};
  PetscInt       triPoints_n2[8]      = {3, 5, 6, 7, 0, 1, 2, 4};
  PetscInt       triSizes_n8[8]       = {1, 1, 1, 1, 1, 1, 1, 1};
  PetscInt       triPoints_n8[8]      = {0, 1, 2, 3, 4, 5, 6, 7};
  PetscInt       quadSizes[2]         = {2, 2};
  PetscInt       quadPoints[4]        = {2, 3, 0, 1};
  PetscInt       gmshSizes_n3[3]      = {14, 14, 14};
  PetscInt       gmshPoints_n3[42]    = {1, 2,  4,  5,  9, 10, 11, 15, 16, 20, 21, 27, 28, 29,
                                         3, 8, 12, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                                         0, 6,  7, 13, 14, 17, 18, 19, 22, 23, 24, 25, 26, 41};
  PetscInt       fluentSizes_n3[3]    = {50, 50, 50};
  PetscInt       fluentPoints_n3[150] = { 5,  6,  7,  8, 12, 14, 16,  34,  36,  37,  38,  39,  40,  41,  42,  43,  44,  45,  46,  48,  50,  51,  80,  81,  89,
                                         91, 93, 94, 95, 96, 97, 98,  99, 100, 101, 104, 121, 122, 124, 125, 126, 127, 128, 129, 131, 133, 143, 144, 145, 147,
                                          1,  3,  4,  9, 10, 17, 18,  19,  24,  25,  26,  27,  28,  29,  30,  31,  32,  33,  35,  47,  61,  71,  72,  73,  74,
                                         75, 76, 77, 78, 79, 86, 87,  88,  90,  92, 113, 115, 116, 117, 118, 119, 120, 123, 138, 140, 141, 142, 146, 148, 149,
                                          0,  2, 11, 13, 15, 20, 21,  22,  23,  49,  52,  53,  54,  55,  56,  57,  58,  59,  60,  62,  63,  64,  65,  66,  67,
                                         68, 69, 70, 82, 83, 84, 85, 102, 103, 105, 106, 107, 108, 109, 110, 111, 112, 114, 130, 132, 134, 135, 136, 137, 139};
  const PetscInt cells[3]             = {2, 2, 2};
  size_t         len, bdlen;
  PetscMPIInt    rank, size;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(user->createMeshEvent,0,0,0,0);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &size);CHKERRQ(ierr);
  ierr = PetscStrlen(filename, &len);CHKERRQ(ierr);
  ierr = PetscStrlen(bdfilename, &bdlen);CHKERRQ(ierr);
  ierr = PetscLogStagePush(user->stages[STAGE_LOAD]);CHKERRQ(ierr);
  if (len) {
    ierr = DMPlexCreateFromFile(comm, filename, interpolate, dm);CHKERRQ(ierr);
  } else if (bdlen) {
    DM boundary;

    ierr = DMPlexCreateFromFile(comm, bdfilename, interpolate, &boundary);CHKERRQ(ierr);
    ierr = DMPlexGenerate(boundary, NULL, interpolate, dm);CHKERRQ(ierr);
    ierr = DMDestroy(&boundary);CHKERRQ(ierr);
  } else {
    switch (user->domainShape) {
    case BOX:
      if (cellSimplex) {ierr = DMPlexCreateBoxMesh(comm, dim, dim == 2 ? 2 : 1, interpolate, dm);CHKERRQ(ierr);}
      else             {ierr = DMPlexCreateHexBoxMesh(comm, dim, cells, user->periodicity[0], user->periodicity[1], user->periodicity[2], dm);CHKERRQ(ierr);}
      break;
    case CYLINDER:
      if (cellSimplex) SETERRQ(comm, PETSC_ERR_ARG_WRONG, "Cannot mesh a cylinder with simplices");
      if (dim != 3)    SETERRQ1(comm, PETSC_ERR_ARG_WRONG, "Dimension must be 3 for a cylinder mesh, not %D", dim);
      if (cellWedge) {
        ierr = DMPlexCreateWedgeCylinderMesh(comm, 6, PETSC_FALSE, dm);CHKERRQ(ierr);
      } else {
        ierr = DMPlexCreateHexCylinderMesh(comm, 3, user->periodicity[2], dm);CHKERRQ(ierr);
        ierr = DMLocalizeCoordinates(*dm);CHKERRQ(ierr);
      }
      break;
    default: SETERRQ1(comm, PETSC_ERR_ARG_WRONG, "Unknown domain shape %D", user->domainShape);
    }
  }
  ierr = PetscLogStagePop();CHKERRQ(ierr);
  {
    DM refinedMesh     = NULL;
    DM distributedMesh = NULL;

    if (user->testPartition) {
      const PetscInt  *sizes = NULL;
      const PetscInt  *points = NULL;
      PetscPartitioner part;

      if (!rank) {
        if (dim == 2 && cellSimplex && size == 2) {
           sizes = triSizes_n2; points = triPoints_n2;
        } else if (dim == 2 && cellSimplex && size == 8) {
          sizes = triSizes_n8; points = triPoints_n8;
        } else if (dim == 2 && !cellSimplex && size == 2) {
          sizes = quadSizes; points = quadPoints;
        } else if (dim == 2 && size == 3) {
          PetscInt Nc;

          ierr = DMPlexGetHeightStratum(*dm, 0, NULL, &Nc);CHKERRQ(ierr);
          if (Nc == 42) { /* Gmsh 3 & 4 */
            sizes = gmshSizes_n3; points = gmshPoints_n3;
          } else if (Nc == 150) { /* Fluent 1 */
            sizes = fluentSizes_n3; points = fluentPoints_n3;
          } else if (Nc == 42) { /* Med 1 */
          } else if (Nc == 161) { /* Med 3 */
          }
        }
      }
      ierr = DMPlexGetPartitioner(*dm, &part);CHKERRQ(ierr);
      ierr = PetscPartitionerSetType(part, PETSCPARTITIONERSHELL);CHKERRQ(ierr);
      ierr = PetscPartitionerShellSetPartition(part, size, sizes, points);CHKERRQ(ierr);
    } else {
      PetscPartitioner part;

      ierr = DMPlexGetPartitioner(*dm,&part);CHKERRQ(ierr);
      ierr = PetscPartitionerSetFromOptions(part);CHKERRQ(ierr);
    }
    /* Distribute mesh over processes */
    ierr = PetscLogStagePush(user->stages[STAGE_DISTRIBUTE]);CHKERRQ(ierr);
    ierr = DMPlexDistribute(*dm, 0, NULL, &distributedMesh);CHKERRQ(ierr);
    if (distributedMesh) {
      ierr = DMDestroy(dm);CHKERRQ(ierr);
      *dm  = distributedMesh;
    }
    ierr = PetscLogStagePop();CHKERRQ(ierr);
    /* Refine mesh using a volume constraint */
    ierr = PetscLogStagePush(user->stages[STAGE_REFINE]);CHKERRQ(ierr);
    ierr = DMPlexSetRefinementUniform(*dm, PETSC_FALSE);CHKERRQ(ierr);
    ierr = DMPlexSetRefinementLimit(*dm, refinementLimit);CHKERRQ(ierr);
    ierr = DMRefine(*dm, comm, &refinedMesh);CHKERRQ(ierr);
    if (refinedMesh) {
      ierr = DMDestroy(dm);CHKERRQ(ierr);
      *dm  = refinedMesh;
    }
    ierr = PetscLogStagePop();CHKERRQ(ierr);
  }
  ierr = PetscLogStagePush(user->stages[STAGE_REFINE]);CHKERRQ(ierr);
  ierr = DMSetFromOptions(*dm);CHKERRQ(ierr);
  ierr = PetscLogStagePop();CHKERRQ(ierr);
  if (user->overlap) {
    DM overlapMesh = NULL;
    /* Add the level-1 overlap to refined mesh */
    ierr = PetscLogStagePush(user->stages[STAGE_OVERLAP]);CHKERRQ(ierr);
    ierr = DMPlexDistributeOverlap(*dm, 1, NULL, &overlapMesh);CHKERRQ(ierr);
    if (overlapMesh) {
      ierr = DMView(overlapMesh, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
      ierr = DMDestroy(dm);CHKERRQ(ierr);
      *dm = overlapMesh;
    }
    ierr = PetscLogStagePop();CHKERRQ(ierr);
  }
  if (simplex2tensor) {
    DM rdm = NULL;
    ierr = DMPlexSetRefinementUniform(*dm, PETSC_TRUE);CHKERRQ(ierr);
    ierr = DMPlexRefineSimplexToTensor(*dm, &rdm);CHKERRQ(ierr);
    if (rdm) {
      ierr = DMDestroy(dm);CHKERRQ(ierr);
      *dm  = rdm;
    }
  }
  ierr = PetscObjectSetName((PetscObject) *dm, "Simplicial Mesh");CHKERRQ(ierr);
  ierr = DMViewFromOptions(*dm, NULL, "-dm_view");CHKERRQ(ierr);
  ierr = PetscLogEventEnd(user->createMeshEvent,0,0,0,0);CHKERRQ(ierr);
  user->dm = *dm;
  PetscFunctionReturn(0);
}

typedef struct ex1_stats
{
  PetscReal min, max, sum, squaresum;
  PetscInt  count;
}
ex1_stats_t;

static void ex1_stats_reduce(void *a, void *b, int * len, MPI_Datatype *datatype)
{
  PetscInt i, N = *len;

  for (i = 0; i < N; i++) {
    ex1_stats_t *A = (ex1_stats_t *) a;
    ex1_stats_t *B = (ex1_stats_t *) b;

    B->min = PetscMin(A->min,B->min);
    B->max = PetscMax(A->max,B->max);
    B->sum += A->sum;
    B->squaresum += A->squaresum;
    B->count += A->count;
  }
}

static PetscErrorCode TestCellShape(DM dm)
{
  PetscMPIInt    rank,size;
  PetscInt       dim, c, cStart, cEnd, count = 0;
  ex1_stats_t    stats, globalStats;
  PetscReal      *J, *invJ, min = 0, max = 0, mean = 0, stdev = 0;
  MPI_Comm       comm = PetscObjectComm((PetscObject)dm);
  DM             dmCoarse;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  stats.min = PETSC_MAX_REAL;
  stats.max = PETSC_MIN_REAL;
  stats.sum = stats.squaresum = 0.;
  stats.count = 0;

  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);

  ierr = PetscMalloc2(dim * dim, &J, dim * dim, &invJ);CHKERRQ(ierr);

  ierr = DMPlexGetHeightStratum(dm,0,&cStart,&cEnd);CHKERRQ(ierr);
  for (c = cStart; c < cEnd; c++) {
    PetscInt  i;
    PetscReal frobJ = 0., frobInvJ = 0., cond2, cond, detJ;

    ierr = DMPlexComputeCellGeometryAffineFEM(dm,c,NULL,J,invJ,&detJ);CHKERRQ(ierr);
    if (detJ < 0.0) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Mesh cell %D is inverted", c);

    for (i = 0; i < dim * dim; i++) {
      frobJ += J[i] * J[i];
      frobInvJ += invJ[i] * invJ[i];
    }
    cond2 = frobJ * frobInvJ;
    cond  = PetscSqrtReal(cond2);

    stats.min = PetscMin(stats.min,cond);
    stats.max = PetscMax(stats.max,cond);
    stats.sum += cond;
    stats.squaresum += cond2;
    stats.count++;
  }

  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  if (size > 1) {
    PetscMPIInt    blockLengths[2] = {4,1};
    MPI_Aint       blockOffsets[2] = {offsetof(ex1_stats_t,min),offsetof(ex1_stats_t,count)};
    MPI_Datatype   blockTypes[2]   = {MPIU_REAL,MPIU_INT}, statType;
    MPI_Op         statReduce;

    ierr = MPI_Type_create_struct(2,blockLengths,blockOffsets,blockTypes,&statType);CHKERRQ(ierr);
    ierr = MPI_Type_commit(&statType);CHKERRQ(ierr);
    ierr = MPI_Op_create(ex1_stats_reduce, PETSC_TRUE, &statReduce);CHKERRQ(ierr);
    ierr = MPI_Reduce(&stats,&globalStats,1,statType,statReduce,0,comm);CHKERRQ(ierr);
    ierr = MPI_Op_free(&statReduce);CHKERRQ(ierr);
    ierr = MPI_Type_free(&statType);CHKERRQ(ierr);
  } else {
    ierr = PetscMemcpy(&globalStats,&stats,sizeof(stats));CHKERRQ(ierr);
  }

  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  if (!rank) {
    count = globalStats.count;
    min = globalStats.min;
    max = globalStats.max;
    mean = globalStats.sum / globalStats.count;
    stdev = PetscSqrtReal(globalStats.squaresum / globalStats.count - mean * mean);
  }
  ierr = PetscPrintf(comm,"Mesh with %d cells, shape condition numbers: min = %g, max = %g, mean = %g, stddev = %g\n", count, (double) min, (double) max, (double) mean, (double) stdev);CHKERRQ(ierr);

  ierr = PetscFree2(J,invJ);CHKERRQ(ierr);

  ierr = DMGetCoarseDM(dm,&dmCoarse);CHKERRQ(ierr);
  if (dmCoarse) {
    ierr = TestCellShape(dmCoarse);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  AppCtx         user;                 /* user-defined work context */
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help);if (ierr) return ierr;
  ierr = ProcessOptions(PETSC_COMM_WORLD, &user);CHKERRQ(ierr);
  ierr = CreateMesh(PETSC_COMM_WORLD, &user, &user.dm);CHKERRQ(ierr);
  if (user.testShape) {
    ierr = TestCellShape(user.dm);CHKERRQ(ierr);
  }
  ierr = DMDestroy(&user.dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  # CTetGen 0-1
  test:
    suffix: 0
    requires: ctetgen
    args: -dim 3 -ctetgen_verbose 4 -dm_view ascii::ascii_info_detail -info -info_exclude null
  test:
    suffix: 1
    requires: ctetgen
    args: -dim 3 -ctetgen_verbose 4 -refinement_limit 0.0625 -dm_view ascii::ascii_info_detail -info -info_exclude null

  # 2D LaTex and ASCII output 2-9
  test:
    suffix: 2
    requires: triangle
    args: -dim 2 -dm_view ascii::ascii_latex
  test:
    suffix: 3
    requires: triangle
    args: -dim 2 -dm_refine 1 -interpolate 1 -dm_view ascii::ascii_info_detail
  test:
    suffix: 4
    requires: triangle
    nsize: 2
    args: -dim 2 -dm_refine 1 -interpolate 1 -test_partition -dm_view ascii::ascii_info_detail
  test:
    suffix: 5
    requires: triangle
    nsize: 2
    args: -dim 2 -dm_refine 1 -interpolate 1 -test_partition -dm_view ascii::ascii_latex
  test:
    suffix: 6
    args: -dim 2 -cell_simplex 0 -dm_view ascii::ascii_info_detail
  test:
    suffix: 7
    args: -dim 2 -cell_simplex 0 -dm_refine 1 -dm_view ascii::ascii_info_detail
  test:
    suffix: 8
    nsize: 2
    args: -dim 2 -cell_simplex 0 -dm_refine 1 -interpolate 1 -test_partition -dm_view ascii::ascii_latex

  # Parallel refinement tests with overlap
  test:
    suffix: refine_overlap_0
    requires: triangle
    nsize: 2
    requires: triangle
    args: -dim 2 -cell_simplex 1 -dm_refine 1 -interpolate 1 -test_partition -overlap 1 -dm_view ascii::ascii_info_detail
  test:
    suffix: refine_overlap_1
    requires: triangle
    nsize: 8
    args: -dim 2 -cell_simplex 1 -dm_refine 1 -interpolate 1 -test_partition -overlap 1 -dm_view ascii::ascii_info_detail

  # Parallel simple partitioner tests
  test:
    suffix: part_simple_0
    requires: triangle
    nsize: 2
    args: -dim 2 -cell_simplex 1 -dm_refine 0 -interpolate 0 -petscpartitioner_type simple -partition_view -dm_view ascii::ascii_info_detail
  test:
    suffix: part_simple_1
    requires: triangle
    nsize: 8
    args: -dim 2 -cell_simplex 1 -dm_refine 1 -interpolate 1 -petscpartitioner_type simple -partition_view -dm_view ascii::ascii_info_detail

  # CGNS reader tests 10-11 (need to find smaller test meshes)
  test:
    suffix: cgns_0
    requires: cgns
    args: -filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/tut21.cgns -interpolate 1 -dm_view

  # Gmsh mesh reader tests
  test:
    suffix: gmsh_0
    requires: !single
    args: -filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/doublet-tet.msh -interpolate 1 -dm_view
  test:
    suffix: gmsh_1
    requires: !single
    args: -filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/square.msh -interpolate 1 -dm_view
  test:
    suffix: gmsh_2
    requires: !single
    args: -filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/square_bin.msh -interpolate 1 -dm_view
  test:
    suffix: gmsh_3
    nsize: 3
    requires: !single
    args: -filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/square.msh -test_partition -interpolate 1 -dm_view
  test:
    suffix: gmsh_4
    nsize: 3
    requires: !single
    args: -filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/square_bin.msh -test_partition -interpolate 1 -dm_view
  test:
    suffix: gmsh_5
    requires: !single
    args: -filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/square_quad.msh -interpolate 1 -dm_view
  test:
    suffix: gmsh_6
    requires: !single
    args: -filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/square_bin_physnames.msh -interpolate 1 -dm_view

  # Fluent mesh reader tests
  test:
    suffix: fluent_0
    requires: !complex
    args: -filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/square.cas -interpolate 1 -dm_view
  test:
    suffix: fluent_1
    nsize: 3
    requires: !complex
    args: -filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/square.cas -interpolate 1 -test_partition -dm_view
  test:
    suffix: fluent_2
    requires: !complex
    args: -filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/cube_5tets_ascii.cas -interpolate 1 -dm_view
  test:
    suffix: fluent_3
    requires: !complex
    TODO: broken
    args: -filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/cube_5tets.cas -interpolate 1 -dm_view

  # Med mesh reader tests, including parallel file reads
  test:
    suffix: med_0
    requires: med
    args: -filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/square.med -interpolate 1 -dm_view
  test:
    suffix: med_1
    requires: med
    nsize: 3
    args: -filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/square.med -interpolate 1 -petscpartitioner_type simple -dm_view
  test:
    suffix: med_2
    requires: med
    args: -filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/cylinder.med -interpolate 1 -dm_view
  test:
    suffix: med_3
    requires: med
    nsize: 3
    args: -filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/cylinder.med -interpolate 1 -petscpartitioner_type simple -dm_view

  # Test shape quality
  test:
    suffix: test_shape
    requires: ctetgen
    args: -dim 3 -interpolate -dm_refine_hierarchy 3 -test_shape

  # Test simplex to tensor conversion
  test:
    suffix: s2t2
    requires: triangle
    args: -dim 2 -simplex2tensor -refinement_limit 0.0625 -dm_view ascii::ascii_info_detail

  test:
    suffix: s2t3
    requires: ctetgen
    args: -dim 3 -simplex2tensor -refinement_limit 0.0625 -dm_view ascii::ascii_info_detail

  # Test domain shapes
  test:
    suffix: cylinder
    args: -dim 3 -cell_simplex 0 -domain_shape cylinder -test_shape -dm_view

  test:
    suffix: cylinder_per
    args: -dim 3 -cell_simplex 0 -domain_shape cylinder -z_periodicity periodic -test_shape -dm_view

  test:
    suffix: cylinder_wedge
    args: -dim 3 -cell_simplex 0 -cell_wedge -domain_shape cylinder -dm_view

  test:
    suffix: box_2d
    args: -dim 2 -cell_simplex 0 -domain_shape box -dm_refine 2 -test_shape -dm_view

  test:
    suffix: box_2d_per
    args: -dim 2 -cell_simplex 0 -domain_shape box -dm_refine 2 -test_shape -dm_view

  test:
    suffix: box_3d
    args: -dim 3 -cell_simplex 0 -domain_shape box -dm_refine 2 -test_shape -dm_view

TEST*/
