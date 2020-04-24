
#include <petsc/private/tsimpl.h>        /*I "petscts.h"  I*/

typedef struct {
  PetscViewer viewer;
} TSTrajectory_Singlefile;

static PetscErrorCode TSTrajectorySet_Singlefile(TSTrajectory tj,TS ts,PetscInt stepnum,PetscReal time,Vec X)
{
  TSTrajectory_Singlefile *sf = (TSTrajectory_Singlefile*)tj->data;
  PetscErrorCode          ierr;
  const char              *filename;

  PetscFunctionBegin;
  if (stepnum == 0) {
    ierr = PetscViewerCreate(PETSC_COMM_WORLD,&sf->viewer);CHKERRQ(ierr);
    ierr = PetscViewerSetType(sf->viewer,PETSCVIEWERBINARY);CHKERRQ(ierr);
    ierr = PetscViewerFileSetMode(sf->viewer,FILE_MODE_WRITE);CHKERRQ(ierr);
    ierr = PetscObjectGetName((PetscObject)tj,&filename);CHKERRQ(ierr);
    ierr = PetscViewerFileSetName(sf->viewer,filename);CHKERRQ(ierr);
  }
  ierr = VecView(X,sf->viewer);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(sf->viewer,&time,1,PETSC_REAL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSTrajectoryDestroy_Singlefile(TSTrajectory tj)
{
  PetscErrorCode          ierr;
  TSTrajectory_Singlefile *sf = (TSTrajectory_Singlefile*)tj->data;

  PetscFunctionBegin;
  ierr = PetscViewerDestroy(&sf->viewer);CHKERRQ(ierr);
  ierr = PetscFree(sf);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
      TSTRAJECTORYSINGLEFILE - Stores all solutions of the ODE/ADE into a single file followed by each timestep. Does not save the intermediate stages in a multistage method

  Level: intermediate

.seealso:  TSTrajectoryCreate(), TS, TSTrajectorySetType()

M*/
PETSC_EXTERN PetscErrorCode TSTrajectoryCreate_Singlefile(TSTrajectory tj,TS ts)
{
  PetscErrorCode          ierr;
  TSTrajectory_Singlefile *sf;

  PetscFunctionBegin;
  ierr = PetscNew(&sf);CHKERRQ(ierr);
  tj->data         = sf;
  tj->ops->set     = TSTrajectorySet_Singlefile;
  tj->ops->get     = NULL;
  tj->ops->destroy = TSTrajectoryDestroy_Singlefile;
  ts->setupcalled  = PETSC_TRUE;
  PetscFunctionReturn(0);
}
