
#include <petsc/private/tsimpl.h>        /*I "petscts.h"  I*/

#undef __FUNCT__
#define __FUNCT__ "OutputBIN"
static PetscErrorCode OutputBIN(const char *filename,PetscViewer *viewer)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscViewerCreate(PETSC_COMM_WORLD,viewer);CHKERRQ(ierr);
  ierr = PetscViewerSetType(*viewer,PETSCVIEWERBINARY);CHKERRQ(ierr);
  ierr = PetscViewerFileSetMode(*viewer,FILE_MODE_WRITE);CHKERRQ(ierr);
  ierr = PetscViewerFileSetName(*viewer,filename);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSTrajectorySet_Visualization"
static PetscErrorCode TSTrajectorySet_Visualization(TSTrajectory tj,TS ts,PetscInt stepnum,PetscReal time,Vec X)
{
  PetscViewer    viewer;
  char           filename[PETSC_MAX_PATH_LEN];
  PetscReal      tprev;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSGetTotalSteps(ts,&stepnum);CHKERRQ(ierr);
  if (stepnum == 0) {
    PetscMPIInt rank;
    ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)ts),&rank);CHKERRQ(ierr);
    if (!rank) {
      ierr = PetscRMTree("Visualization-data");CHKERRQ(ierr);
      ierr = PetscMkdir("Visualization-data");CHKERRQ(ierr);
    }
    ierr = PetscSNPrintf(filename,sizeof(filename),"Visualization-data/SA-%06d.bin",stepnum);CHKERRQ(ierr);
    ierr = OutputBIN(filename,&viewer);CHKERRQ(ierr);
    ierr = VecView(X,viewer);CHKERRQ(ierr);
    ierr = PetscViewerBinaryWrite(viewer,&time,1,PETSC_REAL,PETSC_FALSE);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  ierr = PetscSNPrintf(filename,sizeof(filename),"Visualization-data/SA-%06d.bin",stepnum);CHKERRQ(ierr);
  ierr = OutputBIN(filename,&viewer);CHKERRQ(ierr);
  ierr = VecView(X,viewer);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(viewer,&time,1,PETSC_REAL,PETSC_FALSE);CHKERRQ(ierr);

  ierr = TSGetPrevTime(ts,&tprev);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(viewer,&tprev,1,PETSC_REAL,PETSC_FALSE);CHKERRQ(ierr);

  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
      TSTRAJECTORYVISUALIZATION - Stores each solution of the ODE/DAE in a file

  Level: intermediate

.seealso:  TSTrajectoryCreate(), TS, TSTrajectorySetType()

M*/
#undef __FUNCT__
#define __FUNCT__ "TSTrajectoryCreate_Visualization"
PETSC_EXTERN PetscErrorCode TSTrajectoryCreate_Visualization(TSTrajectory tj,TS ts)
{
  PetscFunctionBegin;
  tj->ops->set  = TSTrajectorySet_Visualization;
  PetscFunctionReturn(0);
}
