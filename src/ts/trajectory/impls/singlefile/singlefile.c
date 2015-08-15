
#include <petsc/private/tsimpl.h>        /*I "petscts.h"  I*/

typedef struct {
  PetscViewer viewer;
} TSTrajectory_Singlefile;

#undef __FUNCT__
#define __FUNCT__ "TSTrajectorySet_Singlefile"
PetscErrorCode TSTrajectorySet_Singlefile(TSTrajectory jac,TS ts,PetscInt stepnum,PetscReal time,Vec X)
{
  TSTrajectory_Singlefile *sf = (TSTrajectory_Singlefile*)jac->data;
  PetscInt                ns,i;
  Vec                     *Y;
  /* tprev is only needed for the adjoint run */
  /*
  PetscReal               tprev; 
   */
  PetscErrorCode          ierr;
  const char              *filename;

  PetscFunctionBeginUser;
  if (stepnum == 0) {
    ierr = PetscViewerCreate(PETSC_COMM_WORLD, &sf->viewer);CHKERRQ(ierr);
    ierr = PetscViewerSetType(sf->viewer, PETSCVIEWERBINARY);CHKERRQ(ierr);
    ierr = PetscViewerFileSetMode(sf->viewer,FILE_MODE_WRITE);CHKERRQ(ierr);
    ierr = PetscObjectGetName((PetscObject)jac,&filename);CHKERRQ(ierr);
    ierr = PetscViewerFileSetName(sf->viewer, filename);CHKERRQ(ierr);
  }
  ierr = TSGetTotalSteps(ts,&stepnum);CHKERRQ(ierr);

  ierr = VecView(X,sf->viewer);CHKERRQ(ierr);

  ierr = PetscViewerBinaryWrite(sf->viewer,&time,1,PETSC_REAL,PETSC_FALSE);CHKERRQ(ierr);

  ierr = TSGetStages(ts,&ns,&Y);CHKERRQ(ierr);
  for (i=0;i<ns;i++) {
    ierr = VecView(Y[i],sf->viewer);CHKERRQ(ierr);
  }

  /* tprev is only needed for the adjoint run */
  /*
  ierr = TSGetPrevTime(ts,&tprev);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(sf->viewer,&tprev,1,PETSC_REAL,PETSC_FALSE);CHKERRQ(ierr);
  */
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSTrajectoryDestroy_Singlefile"
PetscErrorCode TSTrajectoryDestroy_Singlefile(TSTrajectory jac)
{
  PetscErrorCode          ierr;
  TSTrajectory_Singlefile *sf = (TSTrajectory_Singlefile*)jac->data;

  PetscFunctionBegin;
  ierr = PetscViewerDestroy(&sf->viewer);CHKERRQ(ierr);
  ierr = PetscFree(sf);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
      TSTRAJECTORYSINGLEFILE - Stores all solutions of the ODE/ADE into a single file

  Level: intermediate

.seealso:  TSTrajectoryCreate(), TS, TSTrajectorySetType()

M*/
#undef __FUNCT__
#define __FUNCT__ "TSTrajectoryCreate_Singlefile"
PETSC_EXTERN PetscErrorCode TSTrajectoryCreate_Singlefile(TSTrajectory ts)
{
  PetscErrorCode          ierr;
  TSTrajectory_Singlefile *sf;

  PetscFunctionBegin;
  ierr = PetscNew(&sf);CHKERRQ(ierr);
  ts->data         = sf;
  ts->ops->set     = TSTrajectorySet_Singlefile;
  ts->ops->get     = NULL;
  ts->ops->destroy = TSTrajectoryDestroy_Singlefile;
  PetscFunctionReturn(0);
}
