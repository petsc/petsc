
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
#define __FUNCT__ "TSTrajectorySet_Basic"
static PetscErrorCode TSTrajectorySet_Basic(TSTrajectory tj,TS ts,PetscInt stepnum,PetscReal time,Vec X)
{
  PetscViewer    viewer;
  PetscInt       ns,i;
  Vec            *Y;
  char           filename[PETSC_MAX_PATH_LEN];
  PetscReal      tprev;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSGetTotalSteps(ts,&stepnum);CHKERRQ(ierr);
  if (stepnum == 0) {
    PetscMPIInt rank;
    ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)ts),&rank);CHKERRQ(ierr);
    if (!rank) {
      ierr = PetscRMTree("SA-data");CHKERRQ(ierr);
      ierr = PetscMkdir("SA-data");CHKERRQ(ierr);
    }
    ierr = PetscSNPrintf(filename,sizeof(filename),"SA-data/SA-%06d.bin",stepnum);CHKERRQ(ierr);
    ierr = OutputBIN(filename,&viewer);CHKERRQ(ierr);
    ierr = VecView(X,viewer);CHKERRQ(ierr);
    ierr = PetscViewerBinaryWrite(viewer,&time,1,PETSC_REAL,PETSC_FALSE);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  ierr = PetscSNPrintf(filename,sizeof(filename),"SA-data/SA-%06d.bin",stepnum);CHKERRQ(ierr);
  ierr = OutputBIN(filename,&viewer);CHKERRQ(ierr);
  ierr = VecView(X,viewer);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(viewer,&time,1,PETSC_REAL,PETSC_FALSE);CHKERRQ(ierr);

  ierr = TSGetStages(ts,&ns,&Y);CHKERRQ(ierr);
  for (i=0;i<ns;i++) {
    ierr = VecView(Y[i],viewer);CHKERRQ(ierr);
  }

  ierr = TSGetPrevTime(ts,&tprev);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(viewer,&tprev,1,PETSC_REAL,PETSC_FALSE);CHKERRQ(ierr);

  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSTrajectoryGet_Basic"
static PetscErrorCode TSTrajectoryGet_Basic(TSTrajectory tj,TS ts,PetscInt stepnum,PetscReal *t)
{
  Vec            Sol,*Y;
  PetscInt       Nr,i;
  PetscViewer    viewer;
  PetscReal      timepre;
  char           filename[PETSC_MAX_PATH_LEN];
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSGetTotalSteps(ts,&stepnum);CHKERRQ(ierr);
  ierr = PetscSNPrintf(filename,sizeof filename,"SA-data/SA-%06d.bin",stepnum);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_READ,&viewer);CHKERRQ(ierr);

  ierr = TSGetSolution(ts,&Sol);CHKERRQ(ierr);
  ierr = VecLoad(Sol,viewer);CHKERRQ(ierr);

  ierr = PetscViewerBinaryRead(viewer,t,1,NULL,PETSC_REAL);CHKERRQ(ierr);

  if (stepnum != 0) {
    ierr = TSGetStages(ts,&Nr,&Y);CHKERRQ(ierr);
    for (i=0;i<Nr ;i++) {
      ierr = VecLoad(Y[i],viewer);CHKERRQ(ierr);
    }
    ierr = PetscViewerBinaryRead(viewer,&timepre,1,NULL,PETSC_REAL);CHKERRQ(ierr);
    ierr = TSSetTimeStep(ts,-(*t)+timepre);CHKERRQ(ierr);
  }

  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
      TSTRAJECTORYBASIC - Stores each solution of the ODE/ADE in a file

  Level: intermediate

.seealso:  TSTrajectoryCreate(), TS, TSTrajectorySetType()

M*/
#undef __FUNCT__
#define __FUNCT__ "TSTrajectoryCreate_Basic"
PETSC_EXTERN PetscErrorCode TSTrajectoryCreate_Basic(TSTrajectory tj,TS ts)
{
  PetscFunctionBegin;
  tj->ops->set  = TSTrajectorySet_Basic;
  tj->ops->get  = TSTrajectoryGet_Basic;
  PetscFunctionReturn(0);
}
