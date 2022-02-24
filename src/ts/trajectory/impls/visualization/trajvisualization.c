
#include <petsc/private/tsimpl.h>        /*I "petscts.h"  I*/

static PetscErrorCode OutputBIN(MPI_Comm comm,const char *filename,PetscViewer *viewer)
{
  PetscFunctionBegin;
  CHKERRQ(PetscViewerCreate(comm,viewer));
  CHKERRQ(PetscViewerSetType(*viewer,PETSCVIEWERBINARY));
  CHKERRQ(PetscViewerFileSetMode(*viewer,FILE_MODE_WRITE));
  CHKERRQ(PetscViewerFileSetName(*viewer,filename));
  PetscFunctionReturn(0);
}

static PetscErrorCode TSTrajectorySet_Visualization(TSTrajectory tj,TS ts,PetscInt stepnum,PetscReal time,Vec X)
{
  PetscViewer    viewer;
  char           filename[PETSC_MAX_PATH_LEN];
  PetscReal      tprev;
  MPI_Comm       comm;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetComm((PetscObject)ts,&comm));
  if (stepnum == 0) {
    PetscMPIInt rank;
    CHKERRMPI(MPI_Comm_rank(comm,&rank));
    if (rank == 0) {
      CHKERRQ(PetscRMTree("Visualization-data"));
      CHKERRQ(PetscMkdir("Visualization-data"));
    }
    if (tj->names) {
      PetscViewer bnames;
      CHKERRQ(PetscViewerBinaryOpen(comm,"Visualization-data/variablenames",FILE_MODE_WRITE,&bnames));
      CHKERRQ(PetscViewerBinaryWriteStringArray(bnames,(const char *const *)tj->names));
      CHKERRQ(PetscViewerDestroy(&bnames));
    }
    CHKERRQ(PetscSNPrintf(filename,sizeof(filename),"Visualization-data/SA-%06d.bin",stepnum));
    CHKERRQ(OutputBIN(comm,filename,&viewer));
    if (!tj->transform) {
      CHKERRQ(VecView(X,viewer));
    } else {
      Vec XX;
      CHKERRQ((*tj->transform)(tj->transformctx,X,&XX));
      CHKERRQ(VecView(XX,viewer));
      CHKERRQ(VecDestroy(&XX));
    }
    CHKERRQ(PetscViewerBinaryWrite(viewer,&time,1,PETSC_REAL));
    CHKERRQ(PetscViewerDestroy(&viewer));
    PetscFunctionReturn(0);
  }
  CHKERRQ(PetscSNPrintf(filename,sizeof(filename),"Visualization-data/SA-%06d.bin",stepnum));
  CHKERRQ(OutputBIN(comm,filename,&viewer));
  if (!tj->transform) {
    CHKERRQ(VecView(X,viewer));
  } else {
    Vec XX;
    CHKERRQ((*tj->transform)(tj->transformctx,X,&XX));
    CHKERRQ(VecView(XX,viewer));
    CHKERRQ(VecDestroy(&XX));
  }
  CHKERRQ(PetscViewerBinaryWrite(viewer,&time,1,PETSC_REAL));

  CHKERRQ(TSGetPrevTime(ts,&tprev));
  CHKERRQ(PetscViewerBinaryWrite(viewer,&tprev,1,PETSC_REAL));

  CHKERRQ(PetscViewerDestroy(&viewer));
  PetscFunctionReturn(0);
}

/*MC
      TSTRAJECTORYVISUALIZATION - Stores each solution of the ODE/DAE in a file

      Saves each timestep into a separate file in Visualization-data/SA-%06d.bin

      This version saves only the solutions at each timestep, it does not save the solution at each stage,
      see TSTRAJECTORYBASIC that saves all stages

      $PETSC_DIR/share/petsc/matlab/PetscReadBinaryTrajectory.m and $PETSC_DIR/lib/petsc/bin/PetscBinaryIOTrajectory.py
      can read in files created with this format into MATLAB and Python.

  Level: intermediate

.seealso:  TSTrajectoryCreate(), TS, TSTrajectorySetType(), TSTrajectoryType, TSTrajectorySetVariableNames()

M*/
PETSC_EXTERN PetscErrorCode TSTrajectoryCreate_Visualization(TSTrajectory tj,TS ts)
{
  PetscFunctionBegin;
  tj->ops->set    = TSTrajectorySet_Visualization;
  tj->setupcalled = PETSC_TRUE;
  PetscFunctionReturn(0);
}
