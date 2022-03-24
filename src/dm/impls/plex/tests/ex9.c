static char help[] = "Performance tests for DMPlex query operations\n\n";

#include <petscdmplex.h>

typedef struct {
  PetscInt  dim;             /* The topological mesh dimension */
  PetscBool cellSimplex;     /* Flag for simplices */
  PetscBool spectral;        /* Flag for spectral element layout */
  PetscBool interpolate;     /* Flag for mesh interpolation */
  PetscReal refinementLimit; /* Maximum volume of a refined cell */
  PetscInt  numFields;       /* The number of section fields */
  PetscInt *numComponents;   /* The number of field components */
  PetscInt *numDof;          /* The dof signature for the section */
  PetscBool reuseArray;      /* Pass in user allocated array to VecGetClosure() */
  /* Test data */
  PetscBool errors;            /* Treat failures as errors */
  PetscInt  iterations;        /* The number of iterations for a query */
  PetscReal maxConeTime;       /* Max time per run for DMPlexGetCone() */
  PetscReal maxClosureTime;    /* Max time per run for DMPlexGetTransitiveClosure() */
  PetscReal maxVecClosureTime; /* Max time per run for DMPlexVecGetClosure() */
  PetscBool printTimes;        /* Print total times, do not check limits */
} AppCtx;

static PetscErrorCode ProcessOptions(AppCtx *options)
{
  PetscInt       len;
  PetscBool      flg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  options->dim               = 2;
  options->cellSimplex       = PETSC_TRUE;
  options->spectral          = PETSC_FALSE;
  options->interpolate       = PETSC_FALSE;
  options->refinementLimit   = 0.0;
  options->numFields         = 0;
  options->numComponents     = NULL;
  options->numDof            = NULL;
  options->reuseArray        = PETSC_FALSE;
  options->errors            = PETSC_FALSE;
  options->iterations        = 1;
  options->maxConeTime       = 0.0;
  options->maxClosureTime    = 0.0;
  options->maxVecClosureTime = 0.0;
  options->printTimes        = PETSC_FALSE;

  ierr = PetscOptionsBegin(PETSC_COMM_SELF, "", "Meshing Problem Options", "DMPLEX");CHKERRQ(ierr);
  CHKERRQ(PetscOptionsRangeInt("-dim", "The topological mesh dimension", "ex9.c", options->dim, &options->dim, NULL,1,3));
  CHKERRQ(PetscOptionsBool("-cellSimplex", "Flag for simplices", "ex9.c", options->cellSimplex, &options->cellSimplex, NULL));
  CHKERRQ(PetscOptionsBool("-spectral", "Flag for spectral element layout", "ex9.c", options->spectral, &options->spectral, NULL));
  CHKERRQ(PetscOptionsBool("-interpolate", "Flag for mesh interpolation", "ex9.c", options->interpolate, &options->interpolate, NULL));
  CHKERRQ(PetscOptionsReal("-refinement_limit", "The maximum volume of a refined cell", "ex9.c", options->refinementLimit, &options->refinementLimit, NULL));
  CHKERRQ(PetscOptionsBoundedInt("-num_fields", "The number of section fields", "ex9.c", options->numFields, &options->numFields, NULL, 0));
  if (options->numFields) {
    len  = options->numFields;
    CHKERRQ(PetscMalloc1(len, &options->numComponents));
    CHKERRQ(PetscOptionsIntArray("-num_components", "The number of components per field", "ex9.c", options->numComponents, &len, &flg));
    PetscCheckFalse(flg && (len != options->numFields),PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Length of components array is %d should be %d", len, options->numFields);
  }
  len  = (options->dim+1) * PetscMax(1, options->numFields);
  CHKERRQ(PetscMalloc1(len, &options->numDof));
  CHKERRQ(PetscOptionsIntArray("-num_dof", "The dof signature for the section", "ex9.c", options->numDof, &len, &flg));
  PetscCheckFalse(flg && (len != (options->dim+1) * PetscMax(1, options->numFields)),PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Length of dof array is %d should be %d", len, (options->dim+1) * PetscMax(1, options->numFields));

  /* We are specifying the scalar dof, so augment it for multiple components */
  {
    PetscInt f, d;

    for (f = 0; f < options->numFields; ++f) {
      for (d = 0; d <= options->dim; ++d) options->numDof[f*(options->dim+1)+d] *= options->numComponents[f];
    }
  }

  CHKERRQ(PetscOptionsBool("-reuse_array", "Pass in user allocated array to VecGetClosure()", "ex9.c", options->reuseArray, &options->reuseArray, NULL));
  CHKERRQ(PetscOptionsBool("-errors", "Treat failures as errors", "ex9.c", options->errors, &options->errors, NULL));
  CHKERRQ(PetscOptionsBoundedInt("-iterations", "The number of iterations for a query", "ex9.c", options->iterations, &options->iterations, NULL,0));
  CHKERRQ(PetscOptionsReal("-max_cone_time", "The maximum time per run for DMPlexGetCone()", "ex9.c", options->maxConeTime, &options->maxConeTime, NULL));
  CHKERRQ(PetscOptionsReal("-max_closure_time", "The maximum time per run for DMPlexGetTransitiveClosure()", "ex9.c", options->maxClosureTime, &options->maxClosureTime, NULL));
  CHKERRQ(PetscOptionsReal("-max_vec_closure_time", "The maximum time per run for DMPlexVecGetClosure()", "ex9.c", options->maxVecClosureTime, &options->maxVecClosureTime, NULL));
  CHKERRQ(PetscOptionsBool("-print_times", "Print total times, do not check limits", "ex9.c", options->printTimes, &options->printTimes, NULL));
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateSimplex_2D(MPI_Comm comm, DM *newdm)
{
  DM             dm;
  PetscInt       numPoints[2]        = {4, 2};
  PetscInt       coneSize[6]         = {3, 3, 0, 0, 0, 0};
  PetscInt       cones[6]            = {2, 3, 4,  5, 4, 3};
  PetscInt       coneOrientations[6] = {0, 0, 0,  0, 0, 0};
  PetscScalar    vertexCoords[8]     = {-0.5, 0.5, 0.0, 0.0, 0.0, 1.0, 0.5, 0.5};
  PetscInt       markerPoints[8]     = {2, 1, 3, 1, 4, 1, 5, 1};
  PetscInt       dim = 2, depth = 1, p;

  PetscFunctionBegin;
  CHKERRQ(DMCreate(comm, &dm));
  CHKERRQ(PetscObjectSetName((PetscObject) dm, "triangular"));
  CHKERRQ(DMSetType(dm, DMPLEX));
  CHKERRQ(DMSetDimension(dm, dim));
  CHKERRQ(DMPlexCreateFromDAG(dm, depth, numPoints, coneSize, cones, coneOrientations, vertexCoords));
  for (p = 0; p < 4; ++p) {
    CHKERRQ(DMSetLabelValue(dm, "marker", markerPoints[p*2], markerPoints[p*2+1]));
  }
  *newdm = dm;
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateSimplex_3D(MPI_Comm comm, DM *newdm)
{
  DM             dm;
  PetscInt       numPoints[2]        = {5, 2};
  PetscInt       coneSize[23]        = {4, 4, 0, 0, 0, 0, 0};
  PetscInt       cones[8]            = {2, 4, 3, 5,  3, 4, 6, 5};
  PetscInt       coneOrientations[8] = {0, 0, 0, 0,  0, 0, 0, 0};
  PetscScalar    vertexCoords[15]    = {0.0, 0.0, -0.5,  0.0, -0.5, 0.0,  1.0, 0.0, 0.0,  0.0, 0.5, 0.0,  0.0, 0.0, 0.5};
  PetscInt       markerPoints[10]    = {2, 1, 3, 1, 4, 1, 5, 1, 6, 1};
  PetscInt       dim = 3, depth = 1, p;

  PetscFunctionBegin;
  CHKERRQ(DMCreate(comm, &dm));
  CHKERRQ(PetscObjectSetName((PetscObject) dm, "tetrahedral"));
  CHKERRQ(DMSetType(dm, DMPLEX));
  CHKERRQ(DMSetDimension(dm, dim));
  CHKERRQ(DMPlexCreateFromDAG(dm, depth, numPoints, coneSize, cones, coneOrientations, vertexCoords));
  for (p = 0; p < 5; ++p) {
    CHKERRQ(DMSetLabelValue(dm, "marker", markerPoints[p*2], markerPoints[p*2+1]));
  }
  *newdm = dm;
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateQuad_2D(MPI_Comm comm, DM *newdm)
{
  DM             dm;
  PetscInt       numPoints[2]        = {6, 2};
  PetscInt       coneSize[8]         = {4, 4, 0, 0, 0, 0, 0, 0};
  PetscInt       cones[8]            = {2, 3, 4, 5,  3, 6, 7, 4};
  PetscInt       coneOrientations[8] = {0, 0, 0, 0,  0, 0, 0, 0};
  PetscScalar    vertexCoords[12]    = {-0.5, 0.0, 0.0, 0.0, 0.0, 1.0, -0.5, 1.0, 0.5, 0.0, 0.5, 1.0};
  PetscInt       markerPoints[12]    = {2, 1, 3, 1, 4, 1, 5, 1, 6, 1, 7, 1};
  PetscInt       dim = 2, depth = 1, p;

  PetscFunctionBegin;
  CHKERRQ(DMCreate(comm, &dm));
  CHKERRQ(PetscObjectSetName((PetscObject) dm, "quadrilateral"));
  CHKERRQ(DMSetType(dm, DMPLEX));
  CHKERRQ(DMSetDimension(dm, dim));
  CHKERRQ(DMPlexCreateFromDAG(dm, depth, numPoints, coneSize, cones, coneOrientations, vertexCoords));
  for (p = 0; p < 6; ++p) {
    CHKERRQ(DMSetLabelValue(dm, "marker", markerPoints[p*2], markerPoints[p*2+1]));
  }
  *newdm = dm;
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateHex_3D(MPI_Comm comm, DM *newdm)
{
  DM             dm;
  PetscInt       numPoints[2]         = {12, 2};
  PetscInt       coneSize[14]         = {8, 8, 0,0,0,0,0,0,0,0,0,0,0,0};
  PetscInt       cones[16]            = {2,5,4,3,6,7,8,9,  3,4,11,10,7,12,13,8};
  PetscInt       coneOrientations[16] = {0,0,0,0,0,0,0,0,  0,0, 0, 0,0, 0, 0,0};
  PetscScalar    vertexCoords[36]     = {-0.5,0.0,0.0, 0.0,0.0,0.0, 0.0,1.0,0.0, -0.5,1.0,0.0,
                                         -0.5,0.0,1.0, 0.0,0.0,1.0, 0.0,1.0,1.0, -0.5,1.0,1.0,
                                          0.5,0.0,0.0, 0.5,1.0,0.0, 0.5,0.0,1.0,  0.5,1.0,1.0};
  PetscInt       markerPoints[24]     = {2,1,3,1,4,1,5,1,6,1,7,1,8,1,9,1,10,1,11,1,12,1,13,1};
  PetscInt       dim = 3, depth = 1, p;

  PetscFunctionBegin;
  CHKERRQ(DMCreate(comm, &dm));
  CHKERRQ(PetscObjectSetName((PetscObject) dm, "hexahedral"));
  CHKERRQ(DMSetType(dm, DMPLEX));
  CHKERRQ(DMSetDimension(dm, dim));
  CHKERRQ(DMPlexCreateFromDAG(dm, depth, numPoints, coneSize, cones, coneOrientations, vertexCoords));
  for (p = 0; p < 12; ++p) {
    CHKERRQ(DMSetLabelValue(dm, "marker", markerPoints[p*2], markerPoints[p*2+1]));
  }
  *newdm = dm;
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *newdm)
{
  PetscInt       dim         = user->dim;
  PetscBool      cellSimplex = user->cellSimplex;

  PetscFunctionBegin;
  switch (dim) {
  case 2:
    if (cellSimplex) {
      CHKERRQ(CreateSimplex_2D(comm, newdm));
    } else {
      CHKERRQ(CreateQuad_2D(comm, newdm));
    }
    break;
  case 3:
    if (cellSimplex) {
      CHKERRQ(CreateSimplex_3D(comm, newdm));
    } else {
      CHKERRQ(CreateHex_3D(comm, newdm));
    }
    break;
  default:
    SETERRQ(comm, PETSC_ERR_ARG_OUTOFRANGE, "Cannot make meshes for dimension %d", dim);
  }
  if (user->refinementLimit > 0.0) {
    DM rdm;
    const char *name;

    CHKERRQ(DMPlexSetRefinementUniform(*newdm, PETSC_FALSE));
    CHKERRQ(DMPlexSetRefinementLimit(*newdm, user->refinementLimit));
    CHKERRQ(DMRefine(*newdm, PETSC_COMM_SELF, &rdm));
    CHKERRQ(PetscObjectGetName((PetscObject) *newdm, &name));
    CHKERRQ(PetscObjectSetName((PetscObject)    rdm,  name));
    CHKERRQ(DMDestroy(newdm));
    *newdm = rdm;
  }
  if (user->interpolate) {
    DM idm;

    CHKERRQ(DMPlexInterpolate(*newdm, &idm));
    CHKERRQ(DMDestroy(newdm));
    *newdm = idm;
  }
  CHKERRQ(DMSetFromOptions(*newdm));
  PetscFunctionReturn(0);
}

static PetscErrorCode TestCone(DM dm, AppCtx *user)
{
  PetscInt           numRuns, cStart, cEnd, c, i;
  PetscReal          maxTimePerRun = user->maxConeTime;
  PetscLogStage      stage;
  PetscLogEvent      event;
  PetscEventPerfInfo eventInfo;
  MPI_Comm           comm;
  PetscMPIInt        rank;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetComm((PetscObject)dm, &comm));
  CHKERRMPI(MPI_Comm_rank(comm, &rank));
  CHKERRQ(PetscLogStageRegister("DMPlex Cone Test", &stage));
  CHKERRQ(PetscLogEventRegister("Cone", PETSC_OBJECT_CLASSID, &event));
  CHKERRQ(PetscLogStagePush(stage));
  CHKERRQ(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  CHKERRQ(PetscLogEventBegin(event,0,0,0,0));
  for (i = 0; i < user->iterations; ++i) {
    for (c = cStart; c < cEnd; ++c) {
      const PetscInt *cone;

      CHKERRQ(DMPlexGetCone(dm, c, &cone));
    }
  }
  CHKERRQ(PetscLogEventEnd(event,0,0,0,0));
  CHKERRQ(PetscLogStagePop());

  CHKERRQ(PetscLogEventGetPerfInfo(stage, event, &eventInfo));
  numRuns = (cEnd-cStart) * user->iterations;
  PetscCheckFalse(eventInfo.count != 1,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Number of event calls %d should be %d", eventInfo.count, 1);
  PetscCheckFalse((PetscInt) eventInfo.flops != 0,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Number of event flops %d should be %d", (PetscInt) eventInfo.flops, 0);
  if (user->printTimes) {
    CHKERRQ(PetscSynchronizedPrintf(comm, "[%d] Cones: %d Total time: %.3es Average time per cone: %.3es\n", rank, numRuns, eventInfo.time, eventInfo.time/numRuns));
    CHKERRQ(PetscSynchronizedFlush(comm, PETSC_STDOUT));
  } else if (eventInfo.time > maxTimePerRun * numRuns) {
    CHKERRQ(PetscSynchronizedPrintf(comm, "[%d] Cones: %d Average time per cone: %gs standard: %gs\n", rank, numRuns, eventInfo.time/numRuns, maxTimePerRun));
    CHKERRQ(PetscSynchronizedFlush(comm, PETSC_STDOUT));
    PetscCheck(!user->errors,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Average time for cone %g > standard %g", eventInfo.time/numRuns, maxTimePerRun);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TestTransitiveClosure(DM dm, AppCtx *user)
{
  PetscInt           numRuns, cStart, cEnd, c, i;
  PetscReal          maxTimePerRun = user->maxClosureTime;
  PetscLogStage      stage;
  PetscLogEvent      event;
  PetscEventPerfInfo eventInfo;
  MPI_Comm           comm;
  PetscMPIInt        rank;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetComm((PetscObject)dm, &comm));
  CHKERRMPI(MPI_Comm_rank(comm, &rank));
  CHKERRQ(PetscLogStageRegister("DMPlex Transitive Closure Test", &stage));
  CHKERRQ(PetscLogEventRegister("TransitiveClosure", PETSC_OBJECT_CLASSID, &event));
  CHKERRQ(PetscLogStagePush(stage));
  CHKERRQ(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  CHKERRQ(PetscLogEventBegin(event,0,0,0,0));
  for (i = 0; i < user->iterations; ++i) {
    for (c = cStart; c < cEnd; ++c) {
      PetscInt *closure = NULL;
      PetscInt  closureSize;

      CHKERRQ(DMPlexGetTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure));
      CHKERRQ(DMPlexRestoreTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure));
    }
  }
  CHKERRQ(PetscLogEventEnd(event,0,0,0,0));
  CHKERRQ(PetscLogStagePop());

  CHKERRQ(PetscLogEventGetPerfInfo(stage, event, &eventInfo));
  numRuns = (cEnd-cStart) * user->iterations;
  PetscCheckFalse(eventInfo.count != 1,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Number of event calls %d should be %d", eventInfo.count, 1);
  PetscCheckFalse((PetscInt) eventInfo.flops != 0,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Number of event flops %d should be %d", (PetscInt) eventInfo.flops, 0);
  if (user->printTimes) {
    CHKERRQ(PetscSynchronizedPrintf(comm, "[%d] Closures: %d Total time: %.3es Average time per cone: %.3es\n", rank, numRuns, eventInfo.time, eventInfo.time/numRuns));
    CHKERRQ(PetscSynchronizedFlush(comm, PETSC_STDOUT));
  } else if (eventInfo.time > maxTimePerRun * numRuns) {
    CHKERRQ(PetscSynchronizedPrintf(comm, "[%d] Closures: %d Average time per cone: %gs standard: %gs\n", rank, numRuns, eventInfo.time/numRuns, maxTimePerRun));
    CHKERRQ(PetscSynchronizedFlush(comm, PETSC_STDOUT));
    PetscCheck(!user->errors,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Average time for closure %g > standard %g", eventInfo.time/numRuns, maxTimePerRun);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TestVecClosure(DM dm, PetscBool useIndex, PetscBool useSpectral, AppCtx *user)
{
  PetscSection       s;
  Vec                v;
  PetscInt           numRuns, cStart, cEnd, c, i;
  PetscScalar        tmpArray[64];
  PetscScalar       *userArray     = user->reuseArray ? tmpArray : NULL;
  PetscReal          maxTimePerRun = user->maxVecClosureTime;
  PetscLogStage      stage;
  PetscLogEvent      event;
  PetscEventPerfInfo eventInfo;
  MPI_Comm           comm;
  PetscMPIInt        rank;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetComm((PetscObject)dm, &comm));
  CHKERRMPI(MPI_Comm_rank(comm, &rank));
  if (useIndex) {
    if (useSpectral) {
      CHKERRQ(PetscLogStageRegister("DMPlex Vector Closure with Index Test", &stage));
      CHKERRQ(PetscLogEventRegister("VecClosureInd", PETSC_OBJECT_CLASSID, &event));
    } else {
      CHKERRQ(PetscLogStageRegister("DMPlex Vector Spectral Closure with Index Test", &stage));
      CHKERRQ(PetscLogEventRegister("VecClosureSpecInd", PETSC_OBJECT_CLASSID, &event));
    }
  } else {
    if (useSpectral) {
      CHKERRQ(PetscLogStageRegister("DMPlex Vector Spectral Closure Test", &stage));
      CHKERRQ(PetscLogEventRegister("VecClosureSpec", PETSC_OBJECT_CLASSID, &event));
    } else {
      CHKERRQ(PetscLogStageRegister("DMPlex Vector Closure Test", &stage));
      CHKERRQ(PetscLogEventRegister("VecClosure", PETSC_OBJECT_CLASSID, &event));
    }
  }
  CHKERRQ(PetscLogStagePush(stage));
  CHKERRQ(DMSetNumFields(dm, user->numFields));
  CHKERRQ(DMPlexCreateSection(dm, NULL, user->numComponents, user->numDof, 0, NULL, NULL, NULL, NULL, &s));
  CHKERRQ(DMSetLocalSection(dm, s));
  if (useIndex) CHKERRQ(DMPlexCreateClosureIndex(dm, s));
  if (useSpectral) CHKERRQ(DMPlexSetClosurePermutationTensor(dm, PETSC_DETERMINE, s));
  CHKERRQ(PetscSectionDestroy(&s));
  CHKERRQ(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  CHKERRQ(DMGetLocalVector(dm, &v));
  CHKERRQ(PetscLogEventBegin(event,0,0,0,0));
  for (i = 0; i < user->iterations; ++i) {
    for (c = cStart; c < cEnd; ++c) {
      PetscScalar *closure     = userArray;
      PetscInt     closureSize = 64;

      CHKERRQ(DMPlexVecGetClosure(dm, s, v, c, &closureSize, &closure));
      if (!user->reuseArray) CHKERRQ(DMPlexVecRestoreClosure(dm, s, v, c, &closureSize, &closure));
    }
  }
  CHKERRQ(PetscLogEventEnd(event,0,0,0,0));
  CHKERRQ(DMRestoreLocalVector(dm, &v));
  CHKERRQ(PetscLogStagePop());

  CHKERRQ(PetscLogEventGetPerfInfo(stage, event, &eventInfo));
  numRuns = (cEnd-cStart) * user->iterations;
  PetscCheckFalse(eventInfo.count != 1,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Number of event calls %d should be %d", eventInfo.count, 1);
  PetscCheckFalse((PetscInt) eventInfo.flops != 0,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Number of event flops %d should be %d", (PetscInt) eventInfo.flops, 0);
  if (user->printTimes || eventInfo.time > maxTimePerRun * numRuns) {
    const char *title = "VecClosures";
    const char *titleIndex = "VecClosures with Index";
    const char *titleSpec = "VecClosures Spectral";
    const char *titleSpecIndex = "VecClosures Spectral with Index";

    if (user->printTimes) {
      CHKERRQ(PetscSynchronizedPrintf(comm, "[%d] %s: %d Total time: %.3es Average time per vector closure: %.3es\n", rank, useIndex ? (useSpectral ? titleSpecIndex : titleIndex) : (useSpectral ? titleSpec : title), numRuns, eventInfo.time, eventInfo.time/numRuns));
      CHKERRQ(PetscSynchronizedFlush(comm, PETSC_STDOUT));
    } else {
      CHKERRQ(PetscSynchronizedPrintf(comm, "[%d] %s: %d Average time per vector closure: %gs standard: %gs\n", rank, useIndex ? (useSpectral ? titleSpecIndex : titleIndex) : (useSpectral ? titleSpec : title), numRuns, eventInfo.time/numRuns, maxTimePerRun));
      CHKERRQ(PetscSynchronizedFlush(comm, PETSC_STDOUT));
      PetscCheck(!user->errors,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Average time for vector closure %g > standard %g", eventInfo.time/numRuns, maxTimePerRun);
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode CleanupContext(AppCtx *user)
{
  PetscFunctionBegin;
  CHKERRQ(PetscFree(user->numComponents));
  CHKERRQ(PetscFree(user->numDof));
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM             dm;
  AppCtx         user;

  CHKERRQ(PetscInitialize(&argc, &argv, NULL,help));
  CHKERRQ(ProcessOptions(&user));
  CHKERRQ(PetscLogDefaultBegin());
  CHKERRQ(CreateMesh(PETSC_COMM_WORLD, &user, &dm));
  CHKERRQ(TestCone(dm, &user));
  CHKERRQ(TestTransitiveClosure(dm, &user));
  CHKERRQ(TestVecClosure(dm, PETSC_FALSE, PETSC_FALSE, &user));
  CHKERRQ(TestVecClosure(dm, PETSC_TRUE,  PETSC_FALSE, &user));
  if (!user.cellSimplex && user.spectral) {
    CHKERRQ(TestVecClosure(dm, PETSC_FALSE, PETSC_TRUE,  &user));
    CHKERRQ(TestVecClosure(dm, PETSC_TRUE,  PETSC_TRUE,  &user));
  }
  CHKERRQ(DMDestroy(&dm));
  CHKERRQ(CleanupContext(&user));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

  build:
    requires: defined(PETSC_USE_LOG)

  # 2D Simplex P_1 scalar tests
  testset:
    args: -num_dof 1,0,0 -iterations 2 -print_times
    test:
      suffix: correctness_0
    test:
      suffix: correctness_1
      args: -interpolate -dm_refine 2
    test:
      suffix: correctness_2
      requires: triangle
      args: -interpolate -refinement_limit 1.0e-5
  test:
    suffix: 0
    TODO: Only for performance testing
    args: -num_dof 1,0,0 -iterations 10000 -max_cone_time 1.1e-8 -max_closure_time 1.3e-7 -max_vec_closure_time 3.6e-7
  test:
    suffix: 1
    requires: triangle
    TODO: Only for performance testing
    args: -refinement_limit 1.0e-5 -num_dof 1,0,0 -iterations 2 -max_cone_time 2.1e-8 -max_closure_time 1.5e-7 -max_vec_closure_time 3.6e-7
  test:
    suffix: 2
    TODO: Only for performance testing
    args: -num_fields 1 -num_components 1 -num_dof 1,0,0 -iterations 10000 -max_cone_time 1.1e-8 -max_closure_time 1.3e-7 -max_vec_closure_time 4.5e-7
  test:
    suffix: 3
    requires: triangle
    TODO: Only for performance testing
    args: -refinement_limit 1.0e-5 -num_fields 1 -num_components 1 -num_dof 1,0,0 -iterations 2 -max_cone_time 2.1e-8 -max_closure_time 1.5e-7 -max_vec_closure_time 4.7e-7
  test:
    suffix: 4
    TODO: Only for performance testing
    args: -interpolate -num_dof 1,0,0 -iterations 10000 -max_cone_time 1.1e-8 -max_closure_time 6.5e-7 -max_vec_closure_time 1.0e-6
  test:
    suffix: 5
    requires: triangle
    TODO: Only for performance testing
    args: -interpolate -refinement_limit 1.0e-4 -num_dof 1,0,0 -iterations 2 -max_cone_time 2.1e-8 -max_closure_time 6.5e-7 -max_vec_closure_time 1.0e-6
  test:
    suffix: 6
    TODO: Only for performance testing
    args: -interpolate -num_fields 1 -num_components 1 -num_dof 1,0,0 -iterations 10000 -max_cone_time 1.1e-8 -max_closure_time 6.5e-7 -max_vec_closure_time 1.1e-6
  test:
    suffix: 7
    requires: triangle
    TODO: Only for performance testing
    args: -interpolate -refinement_limit 1.0e-4 -num_fields 1 -num_components 1 -num_dof 1,0,0 -iterations 2 -max_cone_time 2.1e-8 -max_closure_time 6.5e-7 -max_vec_closure_time 1.2e-6

  # 2D Simplex P_1 vector tests
  # 2D Simplex P_2 scalar tests
  # 2D Simplex P_2 vector tests
  # 2D Simplex P_2/P_1 vector/scalar tests
  # 2D Quad P_1 scalar tests
  # 2D Quad P_1 vector tests
  # 2D Quad P_2 scalar tests
  # 2D Quad P_2 vector tests
  # 3D Simplex P_1 scalar tests
  # 3D Simplex P_1 vector tests
  # 3D Simplex P_2 scalar tests
  # 3D Simplex P_2 vector tests
  # 3D Hex P_1 scalar tests
  # 3D Hex P_1 vector tests
  # 3D Hex P_2 scalar tests
  # 3D Hex P_2 vector tests

TEST*/
