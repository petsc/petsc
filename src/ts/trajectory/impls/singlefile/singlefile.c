
#include <petsc/private/tsimpl.h> /*I "petscts.h"  I*/

typedef struct {
  PetscViewer viewer;
} TSTrajectory_Singlefile;

static PetscErrorCode TSTrajectorySet_Singlefile(TSTrajectory tj, TS ts, PetscInt stepnum, PetscReal time, Vec X)
{
  TSTrajectory_Singlefile *sf = (TSTrajectory_Singlefile *)tj->data;
  const char              *filename;

  PetscFunctionBegin;
  if (stepnum == 0) {
    PetscCall(PetscViewerCreate(PetscObjectComm((PetscObject)X), &sf->viewer));
    PetscCall(PetscViewerSetType(sf->viewer, PETSCVIEWERBINARY));
    PetscCall(PetscViewerFileSetMode(sf->viewer, FILE_MODE_WRITE));
    PetscCall(PetscObjectGetName((PetscObject)tj, &filename));
    PetscCall(PetscViewerFileSetName(sf->viewer, filename));
  }
  PetscCall(VecView(X, sf->viewer));
  PetscCall(PetscViewerBinaryWrite(sf->viewer, &time, 1, PETSC_REAL));
  PetscFunctionReturn(0);
}

static PetscErrorCode TSTrajectoryDestroy_Singlefile(TSTrajectory tj)
{
  TSTrajectory_Singlefile *sf = (TSTrajectory_Singlefile *)tj->data;

  PetscFunctionBegin;
  PetscCall(PetscViewerDestroy(&sf->viewer));
  PetscCall(PetscFree(sf));
  PetscFunctionReturn(0);
}

/*MC
      TSTRAJECTORYSINGLEFILE - Stores all solutions of the ODE/ADE into a single file followed by each timestep.
      Does not save the intermediate stages in a multistage method

  Level: intermediate

.seealso: [](chapter_ts), `TSTrajectoryCreate()`, `TS`, `TSTrajectorySetType()`, `TSTrajectoryType`, `TSTrajectory`
M*/
PETSC_EXTERN PetscErrorCode TSTrajectoryCreate_Singlefile(TSTrajectory tj, TS ts)
{
  TSTrajectory_Singlefile *sf;

  PetscFunctionBegin;
  PetscCall(PetscNew(&sf));
  tj->data         = sf;
  tj->ops->set     = TSTrajectorySet_Singlefile;
  tj->ops->get     = NULL;
  tj->ops->destroy = TSTrajectoryDestroy_Singlefile;
  ts->setupcalled  = PETSC_TRUE;
  PetscFunctionReturn(0);
}
