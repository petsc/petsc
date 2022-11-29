
#include <petsc/private/tsimpl.h> /*I "petscts.h"  I*/

static PetscErrorCode OutputBIN(MPI_Comm comm, const char *filename, PetscViewer *viewer)
{
  PetscFunctionBegin;
  PetscCall(PetscViewerCreate(comm, viewer));
  PetscCall(PetscViewerSetType(*viewer, PETSCVIEWERBINARY));
  PetscCall(PetscViewerFileSetMode(*viewer, FILE_MODE_WRITE));
  PetscCall(PetscViewerFileSetName(*viewer, filename));
  PetscFunctionReturn(0);
}

static PetscErrorCode TSTrajectorySet_Visualization(TSTrajectory tj, TS ts, PetscInt stepnum, PetscReal time, Vec X)
{
  PetscViewer viewer;
  char        filename[PETSC_MAX_PATH_LEN];
  PetscReal   tprev;
  MPI_Comm    comm;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)ts, &comm));
  if (stepnum == 0) {
    PetscMPIInt rank;
    PetscCallMPI(MPI_Comm_rank(comm, &rank));
    if (rank == 0) {
      PetscCall(PetscRMTree("Visualization-data"));
      PetscCall(PetscMkdir("Visualization-data"));
    }
    if (tj->names) {
      PetscViewer bnames;
      PetscCall(PetscViewerBinaryOpen(comm, "Visualization-data/variablenames", FILE_MODE_WRITE, &bnames));
      PetscCall(PetscViewerBinaryWriteStringArray(bnames, (const char *const *)tj->names));
      PetscCall(PetscViewerDestroy(&bnames));
    }
    PetscCall(PetscSNPrintf(filename, sizeof(filename), "Visualization-data/SA-%06" PetscInt_FMT ".bin", stepnum));
    PetscCall(OutputBIN(comm, filename, &viewer));
    if (!tj->transform) {
      PetscCall(VecView(X, viewer));
    } else {
      Vec XX;
      PetscCall((*tj->transform)(tj->transformctx, X, &XX));
      PetscCall(VecView(XX, viewer));
      PetscCall(VecDestroy(&XX));
    }
    PetscCall(PetscViewerBinaryWrite(viewer, &time, 1, PETSC_REAL));
    PetscCall(PetscViewerDestroy(&viewer));
    PetscFunctionReturn(0);
  }
  PetscCall(PetscSNPrintf(filename, sizeof(filename), "Visualization-data/SA-%06" PetscInt_FMT ".bin", stepnum));
  PetscCall(OutputBIN(comm, filename, &viewer));
  if (!tj->transform) {
    PetscCall(VecView(X, viewer));
  } else {
    Vec XX;
    PetscCall((*tj->transform)(tj->transformctx, X, &XX));
    PetscCall(VecView(XX, viewer));
    PetscCall(VecDestroy(&XX));
  }
  PetscCall(PetscViewerBinaryWrite(viewer, &time, 1, PETSC_REAL));

  PetscCall(TSGetPrevTime(ts, &tprev));
  PetscCall(PetscViewerBinaryWrite(viewer, &tprev, 1, PETSC_REAL));

  PetscCall(PetscViewerDestroy(&viewer));
  PetscFunctionReturn(0);
}

/*MC
      TSTRAJECTORYVISUALIZATION - Stores each solution of the ODE/DAE in a file

      Saves each timestep into a separate file in Visualization-data/SA-%06d.bin

      This version saves only the solutions at each timestep, it does not save the solution at each stage,
      see `TSTRAJECTORYBASIC` that saves all stages

      $PETSC_DIR/share/petsc/matlab/PetscReadBinaryTrajectory.m and $PETSC_DIR/lib/petsc/bin/PetscBinaryIOTrajectory.py
      can read in files created with this format into MATLAB and Python.

  Level: intermediate

.seealso: [](chapter_ts), `TSTrajectoryCreate()`, `TS`, `TSTrajectorySetType()`, `TSTrajectoryType`, `TSTrajectorySetVariableNames()`,
          `TSTrajectoryType`, `TSTrajectory`
M*/
PETSC_EXTERN PetscErrorCode TSTrajectoryCreate_Visualization(TSTrajectory tj, TS ts)
{
  PetscFunctionBegin;
  tj->ops->set    = TSTrajectorySet_Visualization;
  tj->setupcalled = PETSC_TRUE;
  PetscFunctionReturn(0);
}
