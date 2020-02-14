
#include <petsc/private/tsimpl.h>        /*I "petscts.h"  I*/

static PetscErrorCode OutputBIN(MPI_Comm comm,const char *filename,PetscViewer *viewer)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscViewerCreate(PETSC_COMM_WORLD,viewer);CHKERRQ(ierr);
  ierr = PetscViewerSetType(*viewer,PETSCVIEWERBINARY);CHKERRQ(ierr);
  ierr = PetscViewerFileSetMode(*viewer,FILE_MODE_WRITE);CHKERRQ(ierr);
  ierr = PetscViewerFileSetName(*viewer,filename);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSTrajectorySet_Visualization(TSTrajectory tj,TS ts,PetscInt stepnum,PetscReal time,Vec X)
{
  PetscViewer    viewer;
  char           filename[PETSC_MAX_PATH_LEN];
  PetscReal      tprev;
  PetscErrorCode ierr;
  MPI_Comm       comm;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)ts,&comm);CHKERRQ(ierr);
  if (stepnum == 0) {
    PetscMPIInt rank;
    ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
    if (!rank) {
      ierr = PetscRMTree("Visualization-data");CHKERRQ(ierr);
      ierr = PetscMkdir("Visualization-data");CHKERRQ(ierr);
    }
    if (tj->names) {
      PetscViewer bnames;
      ierr = PetscViewerBinaryOpen(comm,"Visualization-data/variablenames",FILE_MODE_WRITE,&bnames);CHKERRQ(ierr);
      ierr = PetscViewerBinaryWriteStringArray(bnames,(const char *const *)tj->names);CHKERRQ(ierr);
      ierr = PetscViewerDestroy(&bnames);CHKERRQ(ierr);
    }
    ierr = PetscSNPrintf(filename,sizeof(filename),"Visualization-data/SA-%06d.bin",stepnum);CHKERRQ(ierr);
    ierr = OutputBIN(comm,filename,&viewer);CHKERRQ(ierr);
    if (!tj->transform) {
      ierr = VecView(X,viewer);CHKERRQ(ierr);
    } else {
      Vec XX;
      ierr = (*tj->transform)(tj->transformctx,X,&XX);CHKERRQ(ierr);
      ierr = VecView(XX,viewer);CHKERRQ(ierr);
      ierr = VecDestroy(&XX);CHKERRQ(ierr);
    }
    ierr = PetscViewerBinaryWrite(viewer,&time,1,PETSC_REAL,PETSC_FALSE);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  ierr = PetscSNPrintf(filename,sizeof(filename),"Visualization-data/SA-%06d.bin",stepnum);CHKERRQ(ierr);
  ierr = OutputBIN(comm,filename,&viewer);CHKERRQ(ierr);
  if (!tj->transform) {
    ierr = VecView(X,viewer);CHKERRQ(ierr);
  } else {
    Vec XX;
    ierr = (*tj->transform)(tj->transformctx,X,&XX);CHKERRQ(ierr);
    ierr = VecView(XX,viewer);CHKERRQ(ierr);
    ierr = VecDestroy(&XX);CHKERRQ(ierr);
  }
  ierr = PetscViewerBinaryWrite(viewer,&time,1,PETSC_REAL,PETSC_FALSE);CHKERRQ(ierr);

  ierr = TSGetPrevTime(ts,&tprev);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(viewer,&tprev,1,PETSC_REAL,PETSC_FALSE);CHKERRQ(ierr);

  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
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
