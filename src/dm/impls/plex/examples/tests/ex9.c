static char help[] = "Performance tests for DMPlex query operations\n\n";

#include <petscdmplex.h>

typedef struct {
  PetscInt  dim;             /* The topological mesh dimension */
  PetscBool cellSimplex;     /* Flag for simplices */
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
} AppCtx;

#undef __FUNCT__
#define __FUNCT__ "ProcessOptions"
static PetscErrorCode ProcessOptions(AppCtx *options)
{
  PetscInt       len;
  PetscBool      flg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  options->dim               = 2;
  options->cellSimplex       = PETSC_TRUE;
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

  ierr = PetscOptionsBegin(PETSC_COMM_SELF, "", "Meshing Problem Options", "DMPLEX");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dim", "The topological mesh dimension", "ex9.c", options->dim, &options->dim, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-cellSimplex", "Flag for simplices", "ex9.c", options->cellSimplex, &options->cellSimplex, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-interpolate", "Flag for mesh interpolation", "ex9.c", options->interpolate, &options->interpolate, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-refinement_limit", "The maximum volume of a refined cell", "ex9.c", options->refinementLimit, &options->refinementLimit, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-num_fields", "The number of section fields", "ex9.c", options->numFields, &options->numFields, NULL);CHKERRQ(ierr);
  if (options->numFields) {
    len  = options->numFields;
    ierr = PetscMalloc1(len, &options->numComponents);CHKERRQ(ierr);
    ierr = PetscOptionsIntArray("-num_components", "The number of components per field", "ex9.c", options->numComponents, &len, &flg);CHKERRQ(ierr);
    if (flg && (len != options->numFields)) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Length of components array is %d should be %d", len, options->numFields);
  }
  len  = (options->dim+1) * PetscMax(1, options->numFields);
  ierr = PetscMalloc1(len, &options->numDof);CHKERRQ(ierr);
  ierr = PetscOptionsIntArray("-num_dof", "The dof signature for the section", "ex9.c", options->numDof, &len, &flg);CHKERRQ(ierr);
  if (flg && (len != (options->dim+1) * PetscMax(1, options->numFields))) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Length of dof array is %d should be %d", len, (options->dim+1) * PetscMax(1, options->numFields));

  ierr = PetscOptionsBool("-reuse_array", "Pass in user allocated array to VecGetClosure()", "ex9.c", options->reuseArray, &options->reuseArray, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-errors", "Treat failures as errors", "ex9.c", options->errors, &options->errors, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-iterations", "The number of iterations for a query", "ex9.c", options->iterations, &options->iterations, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-max_cone_time", "The maximum time per run for DMPlexGetCone()", "ex9.c", options->maxConeTime, &options->maxConeTime, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-max_closure_time", "The maximum time per run for DMPlexGetTransitiveClosure()", "ex9.c", options->maxClosureTime, &options->maxClosureTime, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-max_vec_closure_time", "The maximum time per run for DMPlexVecGetClosure()", "ex9.c", options->maxVecClosureTime, &options->maxVecClosureTime, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
};

#undef __FUNCT__
#define __FUNCT__ "CreateSimplex_2D"
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMCreate(comm, &dm);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) dm, "triangular");CHKERRQ(ierr);
  ierr = DMSetType(dm, DMPLEX);CHKERRQ(ierr);
  ierr = DMSetDimension(dm, dim);CHKERRQ(ierr);
  ierr = DMPlexCreateFromDAG(dm, depth, numPoints, coneSize, cones, coneOrientations, vertexCoords);CHKERRQ(ierr);
  for (p = 0; p < 4; ++p) {
    ierr = DMSetLabelValue(dm, "marker", markerPoints[p*2], markerPoints[p*2+1]);CHKERRQ(ierr);
  }
  *newdm = dm;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CreateSimplex_3D"
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMCreate(comm, &dm);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) dm, "tetrahedral");CHKERRQ(ierr);
  ierr = DMSetType(dm, DMPLEX);CHKERRQ(ierr);
  ierr = DMSetDimension(dm, dim);CHKERRQ(ierr);
  ierr = DMPlexCreateFromDAG(dm, depth, numPoints, coneSize, cones, coneOrientations, vertexCoords);CHKERRQ(ierr);
  for (p = 0; p < 5; ++p) {
    ierr = DMSetLabelValue(dm, "marker", markerPoints[p*2], markerPoints[p*2+1]);CHKERRQ(ierr);
  }
  *newdm = dm;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CreateQuad_2D"
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMCreate(comm, &dm);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) dm, "quadrilateral");CHKERRQ(ierr);
  ierr = DMSetType(dm, DMPLEX);CHKERRQ(ierr);
  ierr = DMSetDimension(dm, dim);CHKERRQ(ierr);
  ierr = DMPlexCreateFromDAG(dm, depth, numPoints, coneSize, cones, coneOrientations, vertexCoords);CHKERRQ(ierr);
  for (p = 0; p < 6; ++p) {
    ierr = DMSetLabelValue(dm, "marker", markerPoints[p*2], markerPoints[p*2+1]);CHKERRQ(ierr);
  }
  *newdm = dm;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CreateHex_3D"
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMCreate(comm, &dm);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) dm, "hexahedral");CHKERRQ(ierr);
  ierr = DMSetType(dm, DMPLEX);CHKERRQ(ierr);
  ierr = DMSetDimension(dm, dim);CHKERRQ(ierr);
  ierr = DMPlexCreateFromDAG(dm, depth, numPoints, coneSize, cones, coneOrientations, vertexCoords);CHKERRQ(ierr);
  for(p = 0; p < 12; ++p) {
    ierr = DMSetLabelValue(dm, "marker", markerPoints[p*2], markerPoints[p*2+1]);CHKERRQ(ierr);
  }
  *newdm = dm;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CreateMesh"
static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *newdm)
{
  PetscInt       dim         = user->dim;
  PetscBool      cellSimplex = user->cellSimplex;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  switch (dim) {
  case 2:
    if (cellSimplex) {
      ierr = CreateSimplex_2D(comm, newdm);CHKERRQ(ierr);
    } else {
      ierr = CreateQuad_2D(comm, newdm);CHKERRQ(ierr);
    }
    break;
  case 3:
    if (cellSimplex) {
      ierr = CreateSimplex_3D(comm, newdm);CHKERRQ(ierr);
    } else {
      ierr = CreateHex_3D(comm, newdm);CHKERRQ(ierr);
    }
    break;
  default:
    SETERRQ1(comm, PETSC_ERR_ARG_OUTOFRANGE, "Cannot make meshes for dimension %d", dim);
  }
  if (user->refinementLimit > 0.0) {
    DM rdm;
    const char *name;

    ierr = DMPlexSetRefinementUniform(*newdm, PETSC_FALSE);CHKERRQ(ierr);
    ierr = DMPlexSetRefinementLimit(*newdm, user->refinementLimit);CHKERRQ(ierr);
    ierr = DMRefine(*newdm, PETSC_COMM_SELF, &rdm);CHKERRQ(ierr);
    ierr = PetscObjectGetName((PetscObject) *newdm, &name);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)    rdm,  name);CHKERRQ(ierr);
    ierr = DMDestroy(newdm);CHKERRQ(ierr);
    *newdm = rdm;
  }
  if (user->interpolate) {
    DM idm = NULL;
    const char *name;

    ierr = DMPlexInterpolate(*newdm, &idm);CHKERRQ(ierr);
    ierr = PetscObjectGetName((PetscObject) *newdm, &name);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)    idm,  name);CHKERRQ(ierr);
    ierr = DMPlexCopyCoordinates(*newdm, idm);CHKERRQ(ierr);
    ierr = DMDestroy(newdm);CHKERRQ(ierr);
    *newdm = idm;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TestCone"
static PetscErrorCode TestCone(DM dm, AppCtx *user)
{
  PetscInt           numRuns, cStart, cEnd, c, i;
  PetscReal          maxTimePerRun = user->maxConeTime;
  PetscLogStage      stage;
  PetscLogEvent      event;
  PetscEventPerfInfo eventInfo;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PetscLogStageRegister("DMPlex Cone Test", &stage);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("Cone", PETSC_OBJECT_CLASSID, &event);CHKERRQ(ierr);
  ierr = PetscLogStagePush(stage);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(event,0,0,0,0);CHKERRQ(ierr);
  for (i = 0; i < user->iterations; ++i) {
    for (c = cStart; c < cEnd; ++c) {
      const PetscInt *cone;

      ierr = DMPlexGetCone(dm, c, &cone);CHKERRQ(ierr);
    }
  }
  ierr = PetscLogEventEnd(event,0,0,0,0);CHKERRQ(ierr);
  ierr = PetscLogStagePop();CHKERRQ(ierr);

  ierr = PetscLogEventGetPerfInfo(stage, event, &eventInfo);CHKERRQ(ierr);
  numRuns = (cEnd-cStart) * user->iterations;
  if (eventInfo.count != 1) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Number of event calls %d should be %d", eventInfo.count, 1);
  if ((PetscInt) eventInfo.flops != 0) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Number of event flops %d should be %d", (PetscInt) eventInfo.flops, 0);
  if (eventInfo.time > maxTimePerRun * numRuns) {
    ierr = PetscPrintf(PETSC_COMM_SELF, "Cones: %d Average time per cone: %gs standard: %gs\n", numRuns, eventInfo.time/numRuns, maxTimePerRun);
    if (user->errors) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Average time for cone %g > standard %g", eventInfo.time/numRuns, maxTimePerRun);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TestTransitiveClosure"
static PetscErrorCode TestTransitiveClosure(DM dm, AppCtx *user)
{
  PetscInt           numRuns, cStart, cEnd, c, i;
  PetscReal          maxTimePerRun = user->maxClosureTime;
  PetscLogStage      stage;
  PetscLogEvent      event;
  PetscEventPerfInfo eventInfo;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PetscLogStageRegister("DMPlex Transitive Closure Test", &stage);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("TransitiveClosure", PETSC_OBJECT_CLASSID, &event);CHKERRQ(ierr);
  ierr = PetscLogStagePush(stage);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(event,0,0,0,0);CHKERRQ(ierr);
  for (i = 0; i < user->iterations; ++i) {
    for (c = cStart; c < cEnd; ++c) {
      PetscInt *closure = NULL;
      PetscInt  closureSize;

      ierr = DMPlexGetTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
      ierr = DMPlexRestoreTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
    }
  }
  ierr = PetscLogEventEnd(event,0,0,0,0);CHKERRQ(ierr);
  ierr = PetscLogStagePop();CHKERRQ(ierr);

  ierr = PetscLogEventGetPerfInfo(stage, event, &eventInfo);CHKERRQ(ierr);
  numRuns = (cEnd-cStart) * user->iterations;
  if (eventInfo.count != 1) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Number of event calls %d should be %d", eventInfo.count, 1);
  if ((PetscInt) eventInfo.flops != 0) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Number of event flops %d should be %d", (PetscInt) eventInfo.flops, 0);
  if (eventInfo.time > maxTimePerRun * numRuns) {
    ierr = PetscPrintf(PETSC_COMM_SELF, "Closures: %d Average time per cone: %gs standard: %gs\n", numRuns, eventInfo.time/numRuns, maxTimePerRun);
    if (user->errors) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Average time for closure %g > standard %g", eventInfo.time/numRuns, maxTimePerRun);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TestVecClosure"
static PetscErrorCode TestVecClosure(DM dm, PetscBool useIndex, AppCtx *user)
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
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  if (useIndex) {
    ierr = PetscLogStageRegister("DMPlex Vector Closure with Index Test", &stage);CHKERRQ(ierr);
    ierr = PetscLogEventRegister("VecClosureInd", PETSC_OBJECT_CLASSID, &event);CHKERRQ(ierr);
  } else {
    ierr = PetscLogStageRegister("DMPlex Vector Closure Test", &stage);CHKERRQ(ierr);
    ierr = PetscLogEventRegister("VecClosure", PETSC_OBJECT_CLASSID, &event);CHKERRQ(ierr);
  }
  ierr = PetscLogStagePush(stage);CHKERRQ(ierr);
  ierr = DMPlexCreateSection(dm, user->dim, user->numFields, user->numComponents, user->numDof, 0, NULL, NULL, NULL, NULL, &s);CHKERRQ(ierr);
  ierr = DMSetDefaultSection(dm, s);CHKERRQ(ierr);
  if (useIndex) {ierr = DMPlexCreateClosureIndex(dm, s);CHKERRQ(ierr);}
  ierr = PetscSectionDestroy(&s);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dm, &v);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(event,0,0,0,0);CHKERRQ(ierr);
  for (i = 0; i < user->iterations; ++i) {
    for (c = cStart; c < cEnd; ++c) {
      PetscScalar *closure     = userArray;
      PetscInt     closureSize = 64;;

      ierr = DMPlexVecGetClosure(dm, s, v, c, &closureSize, &closure);CHKERRQ(ierr);
      if (!user->reuseArray) {ierr = DMPlexVecRestoreClosure(dm, s, v, c, &closureSize, &closure);CHKERRQ(ierr);}
    }
  }
  ierr = PetscLogEventEnd(event,0,0,0,0);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm, &v);CHKERRQ(ierr);
  ierr = PetscLogStagePop();CHKERRQ(ierr);

  ierr = PetscLogEventGetPerfInfo(stage, event, &eventInfo);CHKERRQ(ierr);
  numRuns = (cEnd-cStart) * user->iterations;
  if (eventInfo.count != 1) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Number of event calls %d should be %d", eventInfo.count, 1);
  if ((PetscInt) eventInfo.flops != 0) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Number of event flops %d should be %d", (PetscInt) eventInfo.flops, 0);
  if (eventInfo.time > maxTimePerRun * numRuns) {
    ierr = PetscPrintf(PETSC_COMM_SELF, "VecClosures: %d Average time per cone: %gs standard: %gs\n", numRuns, eventInfo.time/numRuns, maxTimePerRun);
    if (user->errors) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Average time for vector closure %g > standard %g", eventInfo.time/numRuns, maxTimePerRun);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CleanupContext"
static PetscErrorCode CleanupContext(AppCtx *user)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(user->numComponents);CHKERRQ(ierr);
  ierr = PetscFree(user->numDof);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char **argv)
{
  DM             dm;
  AppCtx         user;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help);CHKERRQ(ierr);
  ierr = ProcessOptions(&user);CHKERRQ(ierr);
  ierr = PetscLogDefaultBegin();CHKERRQ(ierr);
  ierr = CreateMesh(PETSC_COMM_SELF, &user, &dm);CHKERRQ(ierr);
  ierr = TestCone(dm, &user);CHKERRQ(ierr);
  ierr = TestTransitiveClosure(dm, &user);CHKERRQ(ierr);
  ierr = TestVecClosure(dm, PETSC_FALSE, &user);CHKERRQ(ierr);
  ierr = TestVecClosure(dm, PETSC_TRUE,  &user);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = CleanupContext(&user);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}
