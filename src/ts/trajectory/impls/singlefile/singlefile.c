
#include <petsc/private/tsimpl.h>        /*I "petscts.h"  I*/

typedef struct {
  PetscViewer viewer;
} TSTrajectory_Singlefile;

static PetscErrorCode TSTrajectorySet_Singlefile(TSTrajectory tj,TS ts,PetscInt stepnum,PetscReal time,Vec X)
{
  TSTrajectory_Singlefile *sf = (TSTrajectory_Singlefile*)tj->data;
  const char              *filename;

  PetscFunctionBegin;
  if (stepnum == 0) {
    CHKERRQ(PetscViewerCreate(PetscObjectComm((PetscObject)X),&sf->viewer));
    CHKERRQ(PetscViewerSetType(sf->viewer,PETSCVIEWERBINARY));
    CHKERRQ(PetscViewerFileSetMode(sf->viewer,FILE_MODE_WRITE));
    CHKERRQ(PetscObjectGetName((PetscObject)tj,&filename));
    CHKERRQ(PetscViewerFileSetName(sf->viewer,filename));
  }
  CHKERRQ(VecView(X,sf->viewer));
  CHKERRQ(PetscViewerBinaryWrite(sf->viewer,&time,1,PETSC_REAL));
  PetscFunctionReturn(0);
}

static PetscErrorCode TSTrajectoryDestroy_Singlefile(TSTrajectory tj)
{
  TSTrajectory_Singlefile *sf = (TSTrajectory_Singlefile*)tj->data;

  PetscFunctionBegin;
  CHKERRQ(PetscViewerDestroy(&sf->viewer));
  CHKERRQ(PetscFree(sf));
  PetscFunctionReturn(0);
}

/*MC
      TSTRAJECTORYSINGLEFILE - Stores all solutions of the ODE/ADE into a single file followed by each timestep. Does not save the intermediate stages in a multistage method

  Level: intermediate

.seealso:  TSTrajectoryCreate(), TS, TSTrajectorySetType()

M*/
PETSC_EXTERN PetscErrorCode TSTrajectoryCreate_Singlefile(TSTrajectory tj,TS ts)
{
  TSTrajectory_Singlefile *sf;

  PetscFunctionBegin;
  CHKERRQ(PetscNew(&sf));
  tj->data         = sf;
  tj->ops->set     = TSTrajectorySet_Singlefile;
  tj->ops->get     = NULL;
  tj->ops->destroy = TSTrajectoryDestroy_Singlefile;
  ts->setupcalled  = PETSC_TRUE;
  PetscFunctionReturn(0);
}
