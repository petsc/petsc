
#include <petsc/private/tsimpl.h> /*I "petscts.h"  I*/

typedef struct {
  PetscViewer viewer;
} TSTrajectory_Basic;
/*
  For n-th time step, TSTrajectorySet_Basic always saves the solution X(t_n) and the current time t_n,
  and optionally saves the stage values Y[] between t_{n-1} and t_n, the previous time t_{n-1}, and
  forward stage sensitivities S[] = dY[]/dp.
*/
static PetscErrorCode TSTrajectorySet_Basic(TSTrajectory tj, TS ts, PetscInt stepnum, PetscReal time, Vec X)
{
  TSTrajectory_Basic *tjbasic = (TSTrajectory_Basic *)tj->data;
  char                filename[PETSC_MAX_PATH_LEN];
  PetscInt            ns, i;

  PetscFunctionBegin;
  PetscCall(PetscSNPrintf(filename, sizeof(filename), tj->dirfiletemplate, stepnum));
  PetscCall(PetscViewerFileSetName(tjbasic->viewer, filename)); /* this triggers PetscViewer to be set up again */
  PetscCall(PetscViewerSetUp(tjbasic->viewer));
  PetscCall(VecView(X, tjbasic->viewer));
  PetscCall(PetscViewerBinaryWrite(tjbasic->viewer, &time, 1, PETSC_REAL));
  if (stepnum && !tj->solution_only) {
    Vec      *Y;
    PetscReal tprev;
    PetscCall(TSGetStages(ts, &ns, &Y));
    for (i = 0; i < ns; i++) {
      /* For stiffly accurate TS methods, the last stage Y[ns-1] is the same as the solution X, thus does not need to be saved again. */
      if (ts->stifflyaccurate && i == ns - 1) continue;
      PetscCall(VecView(Y[i], tjbasic->viewer));
    }
    PetscCall(TSGetPrevTime(ts, &tprev));
    PetscCall(PetscViewerBinaryWrite(tjbasic->viewer, &tprev, 1, PETSC_REAL));
  }
  /* Tangent linear sensitivities needed by second-order adjoint */
  if (ts->forward_solve) {
    Mat A, *S;

    PetscCall(TSForwardGetSensitivities(ts, NULL, &A));
    PetscCall(MatView(A, tjbasic->viewer));
    if (stepnum) {
      PetscCall(TSForwardGetStages(ts, &ns, &S));
      for (i = 0; i < ns; i++) PetscCall(MatView(S[i], tjbasic->viewer));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSTrajectorySetFromOptions_Basic(TSTrajectory tj, PetscOptionItems *PetscOptionsObject)
{
  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "TS trajectory options for Basic type");
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSTrajectoryGet_Basic(TSTrajectory tj, TS ts, PetscInt stepnum, PetscReal *t)
{
  PetscViewer viewer;
  char        filename[PETSC_MAX_PATH_LEN];
  Vec         Sol;
  PetscInt    ns, i;

  PetscFunctionBegin;
  PetscCall(PetscSNPrintf(filename, sizeof(filename), tj->dirfiletemplate, stepnum));
  PetscCall(PetscViewerBinaryOpen(PetscObjectComm((PetscObject)tj), filename, FILE_MODE_READ, &viewer));
  PetscCall(PetscViewerPushFormat(viewer, PETSC_VIEWER_NATIVE));
  PetscCall(TSGetSolution(ts, &Sol));
  PetscCall(VecLoad(Sol, viewer));
  PetscCall(PetscViewerBinaryRead(viewer, t, 1, NULL, PETSC_REAL));
  if (stepnum && !tj->solution_only) {
    Vec      *Y;
    PetscReal timepre;
    PetscCall(TSGetStages(ts, &ns, &Y));
    for (i = 0; i < ns; i++) {
      /* For stiffly accurate TS methods, the last stage Y[ns-1] is the same as the solution X, thus does not need to be loaded again. */
      if (ts->stifflyaccurate && i == ns - 1) continue;
      PetscCall(VecLoad(Y[i], viewer));
    }
    PetscCall(PetscViewerBinaryRead(viewer, &timepre, 1, NULL, PETSC_REAL));
    if (tj->adjoint_solve_mode) PetscCall(TSSetTimeStep(ts, -(*t) + timepre));
  }
  /* Tangent linear sensitivities needed by second-order adjoint */
  if (ts->forward_solve) {
    if (!ts->stifflyaccurate) {
      Mat A;
      PetscCall(TSForwardGetSensitivities(ts, NULL, &A));
      PetscCall(MatLoad(A, viewer));
    }
    if (stepnum) {
      Mat *S;
      PetscCall(TSForwardGetStages(ts, &ns, &S));
      for (i = 0; i < ns; i++) PetscCall(MatLoad(S[i], viewer));
    }
  }
  PetscCall(PetscViewerDestroy(&viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TSTrajectorySetUp_Basic(TSTrajectory tj, TS ts)
{
  MPI_Comm    comm;
  PetscMPIInt rank;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)tj, &comm));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  if (rank == 0) {
    char     *dir = tj->dirname;
    PetscBool flg;

    if (!dir) {
      char dtempname[16] = "TS-data-XXXXXX";
      PetscCall(PetscMkdtemp(dtempname));
      PetscCall(PetscStrallocpy(dtempname, &tj->dirname));
    } else {
      PetscCall(PetscTestDirectory(dir, 'w', &flg));
      if (!flg) {
        PetscCall(PetscTestFile(dir, 'r', &flg));
        PetscCheck(!flg, PETSC_COMM_SELF, PETSC_ERR_USER, "Specified path is a file - not a dir: %s", dir);
        PetscCall(PetscMkdir(dir));
      } else SETERRQ(comm, PETSC_ERR_SUP, "Directory %s not empty", tj->dirname);
    }
  }
  PetscCall(PetscBarrier((PetscObject)tj));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSTrajectoryDestroy_Basic(TSTrajectory tj)
{
  TSTrajectory_Basic *tjbasic = (TSTrajectory_Basic *)tj->data;

  PetscFunctionBegin;
  PetscCall(PetscViewerDestroy(&tjbasic->viewer));
  PetscCall(PetscFree(tjbasic));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
      TSTRAJECTORYBASIC - Stores each solution of the ODE/DAE in a file

      Saves each timestep into a separate file named TS-data-XXXXXX/TS-%06d.bin. The file name can be changed.

      This version saves the solutions at all the stages

      $PETSC_DIR/share/petsc/matlab/PetscReadBinaryTrajectory.m can read in files created with this format

  Level: intermediate

.seealso: [](chapter_ts), `TSTrajectoryCreate()`, `TS`, `TSTrajectory`, `TSTrajectorySetType()`, `TSTrajectorySetDirname()`, `TSTrajectorySetFile()`,
          `TSTrajectoryType`
M*/
PETSC_EXTERN PetscErrorCode TSTrajectoryCreate_Basic(TSTrajectory tj, TS ts)
{
  TSTrajectory_Basic *tjbasic;

  PetscFunctionBegin;
  PetscCall(PetscNew(&tjbasic));

  PetscCall(PetscViewerCreate(PetscObjectComm((PetscObject)tj), &tjbasic->viewer));
  PetscCall(PetscViewerSetType(tjbasic->viewer, PETSCVIEWERBINARY));
  PetscCall(PetscViewerPushFormat(tjbasic->viewer, PETSC_VIEWER_NATIVE));
  PetscCall(PetscViewerFileSetMode(tjbasic->viewer, FILE_MODE_WRITE));
  tj->data = tjbasic;

  tj->ops->set            = TSTrajectorySet_Basic;
  tj->ops->get            = TSTrajectoryGet_Basic;
  tj->ops->setup          = TSTrajectorySetUp_Basic;
  tj->ops->destroy        = TSTrajectoryDestroy_Basic;
  tj->ops->setfromoptions = TSTrajectorySetFromOptions_Basic;
  PetscFunctionReturn(PETSC_SUCCESS);
}
