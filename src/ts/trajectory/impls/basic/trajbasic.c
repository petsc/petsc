
#include <petsc-private/tsimpl.h>        /*I "petscts.h"  I*/

#undef __FUNCT__
#define __FUNCT__ "OutputBIN"
static PetscErrorCode OutputBIN(const char *filename, PetscViewer *viewer)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscViewerCreate(PETSC_COMM_WORLD, viewer);CHKERRQ(ierr);
  ierr = PetscViewerSetType(*viewer, PETSCVIEWERBINARY);CHKERRQ(ierr);
  ierr = PetscViewerFileSetMode(*viewer,FILE_MODE_WRITE);CHKERRQ(ierr);
  ierr = PetscViewerFileSetName(*viewer, filename);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "TSTrajectorySet_Basic"
PetscErrorCode TSTrajectorySet_Basic(TSTrajectory jac,TS ts,PetscInt stepnum,PetscReal time,Vec X)
{
  PetscViewer    viewer;
  PetscInt       ns,i;
  Vec            *Y;
  char           filename[PETSC_MAX_PATH_LEN];
  PetscReal      tprev;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  if (stepnum == 0) {
#if defined(PETSC_HAVE_POPEN)
    ierr = TSGetTotalSteps(ts,&stepnum);CHKERRQ(ierr);
    if (stepnum == 0) {
      PetscMPIInt rank;
      ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)ts),&rank);CHKERRQ(ierr);
      if (!rank) {
        char command[PETSC_MAX_PATH_LEN];
        FILE *fd;
        int  err;

        ierr = PetscMemzero(command,sizeof(command));CHKERRQ(ierr);
        ierr = PetscSNPrintf(command,PETSC_MAX_PATH_LEN,"rm -fr %s","SA-data");CHKERRQ(ierr);
        ierr = PetscPOpen(PETSC_COMM_SELF,NULL,command,"r",&fd);CHKERRQ(ierr);
        ierr = PetscPClose(PETSC_COMM_SELF,fd,&err);CHKERRQ(ierr);
        ierr = PetscSNPrintf(command,PETSC_MAX_PATH_LEN,"mkdir %s","SA-data");CHKERRQ(ierr);
        ierr = PetscPOpen(PETSC_COMM_SELF,NULL,command,"r",&fd);CHKERRQ(ierr);
        ierr = PetscPClose(PETSC_COMM_SELF,fd,&err);CHKERRQ(ierr);
      }
    }
#endif
    PetscFunctionReturn(0);
  }
  ierr = TSGetPrevTime(ts,&tprev);CHKERRQ(ierr);
  ierr = TSGetTotalSteps(ts,&stepnum);CHKERRQ(ierr);
  ierr = PetscSNPrintf(filename,sizeof(filename),"SA-data/SA-%06d.bin",stepnum);CHKERRQ(ierr);
  ierr = OutputBIN(filename,&viewer);CHKERRQ(ierr);
  ierr = VecView(X,viewer);CHKERRQ(ierr);
  /* ierr = PetscRealView(1,&time,viewer);CHKERRQ(ierr); */
  ierr = PetscViewerBinaryWrite(viewer,&tprev,1,PETSC_REAL,PETSC_FALSE);CHKERRQ(ierr);
  /* ierr = PetscViewerBinaryWrite(viewer,&h ,1,PETSC_REAL,PETSC_FALSE);CHKERRQ(ierr); */
  ierr = TSGetStages(ts,&ns,&Y);CHKERRQ(ierr);

  for (i=0;i<ns;i++) {
    ierr = VecView(Y[i],viewer);CHKERRQ(ierr);
  }

  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSTrajectoryGet_Basic"
PetscErrorCode TSTrajectoryGet_Basic(TSTrajectory jac,TS ts,PetscInt step,PetscReal t)
{
  PetscReal      ptime;
  Vec            Sol,*Y;
  PetscInt       Nr,i,num = 1;
  PetscViewer    viewer;
  PetscReal      timepre;
  char           filename[PETSC_MAX_PATH_LEN];
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = TSGetTotalSteps(ts,&step);CHKERRQ(ierr);
  ierr = PetscSNPrintf(filename,sizeof filename,"SA-data/SA-%06d.bin",step);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_READ,&viewer);CHKERRQ(ierr);

  ierr = TSGetSolution(ts,&Sol);CHKERRQ(ierr);
  ierr = VecLoad(Sol,viewer);CHKERRQ(ierr);

  Nr   = 1;
  /* ierr = PetscRealLoad(Nr,&Nr,&timepre,viewer);CHKERRQ(ierr); */
  ierr = PetscViewerBinaryRead(viewer,&timepre,&num,PETSC_REAL);CHKERRQ(ierr);

  ierr = TSGetStages(ts,&Nr,&Y);CHKERRQ(ierr);
  for (i=0;i<Nr ;i++) {
    ierr = VecLoad(Y[i],viewer);CHKERRQ(ierr);
  }

  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  ierr = TSGetTime(ts,&ptime);CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts,-ptime+timepre);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/*MC
      TSTRAJECTORYBASIC - Stores each solution of the ODE/ADE in a file

  Level: intermediate

.seealso:  TSTrajectoryCreate(), TS, TSTrajectorySetType()

M*/
#undef __FUNCT__
#define __FUNCT__ "TSTrajectoryCreate_Basic"
PETSC_EXTERN PetscErrorCode TSTrajectoryCreate_Basic(TSTrajectory ts)
{
  PetscFunctionBegin;
  ts->ops->set  = TSTrajectorySet_Basic;
  ts->ops->get  = TSTrajectoryGet_Basic;
  PetscFunctionReturn(0);
}
