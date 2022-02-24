
#include <petsc/private/tsimpl.h>        /*I "petscts.h"  I*/

typedef struct {
  PetscViewer viewer;
} TSTrajectory_Basic;
/*
  For n-th time step, TSTrajectorySet_Basic always saves the solution X(t_n) and the current time t_n,
  and optionally saves the stage values Y[] between t_{n-1} and t_n, the previous time t_{n-1}, and
  forward stage sensitivities S[] = dY[]/dp.
*/
static PetscErrorCode TSTrajectorySet_Basic(TSTrajectory tj,TS ts,PetscInt stepnum,PetscReal time,Vec X)
{
  TSTrajectory_Basic *tjbasic = (TSTrajectory_Basic*)tj->data;
  char               filename[PETSC_MAX_PATH_LEN];
  PetscInt           ns,i;

  PetscFunctionBegin;
  CHKERRQ(PetscSNPrintf(filename,sizeof(filename),tj->dirfiletemplate,stepnum));
  CHKERRQ(PetscViewerFileSetName(tjbasic->viewer,filename)); /* this triggers PetscViewer to be set up again */
  CHKERRQ(PetscViewerSetUp(tjbasic->viewer));
  CHKERRQ(VecView(X,tjbasic->viewer));
  CHKERRQ(PetscViewerBinaryWrite(tjbasic->viewer,&time,1,PETSC_REAL));
  if (stepnum && !tj->solution_only) {
    Vec       *Y;
    PetscReal tprev;
    CHKERRQ(TSGetStages(ts,&ns,&Y));
    for (i=0; i<ns; i++) {
      /* For stiffly accurate TS methods, the last stage Y[ns-1] is the same as the solution X, thus does not need to be saved again. */
      if (ts->stifflyaccurate && i == ns-1) continue;
      CHKERRQ(VecView(Y[i],tjbasic->viewer));
    }
    CHKERRQ(TSGetPrevTime(ts,&tprev));
    CHKERRQ(PetscViewerBinaryWrite(tjbasic->viewer,&tprev,1,PETSC_REAL));
  }
  /* Tangent linear sensitivities needed by second-order adjoint */
  if (ts->forward_solve) {
    Mat A,*S;

    CHKERRQ(TSForwardGetSensitivities(ts,NULL,&A));
    CHKERRQ(MatView(A,tjbasic->viewer));
    if (stepnum) {
      CHKERRQ(TSForwardGetStages(ts,&ns,&S));
      for (i=0; i<ns; i++) {
        CHKERRQ(MatView(S[i],tjbasic->viewer));
      }
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TSTrajectorySetFromOptions_Basic(PetscOptionItems *PetscOptionsObject,TSTrajectory tj)
{
  PetscFunctionBegin;
  CHKERRQ(PetscOptionsHead(PetscOptionsObject,"TS trajectory options for Basic type"));
  CHKERRQ(PetscOptionsTail());
  PetscFunctionReturn(0);
}

static PetscErrorCode TSTrajectoryGet_Basic(TSTrajectory tj,TS ts,PetscInt stepnum,PetscReal *t)
{
  PetscViewer    viewer;
  char           filename[PETSC_MAX_PATH_LEN];
  Vec            Sol;
  PetscInt       ns,i;

  PetscFunctionBegin;
  CHKERRQ(PetscSNPrintf(filename,sizeof(filename),tj->dirfiletemplate,stepnum));
  CHKERRQ(PetscViewerBinaryOpen(PetscObjectComm((PetscObject)tj),filename,FILE_MODE_READ,&viewer));
  CHKERRQ(PetscViewerPushFormat(viewer,PETSC_VIEWER_NATIVE));
  CHKERRQ(TSGetSolution(ts,&Sol));
  CHKERRQ(VecLoad(Sol,viewer));
  CHKERRQ(PetscViewerBinaryRead(viewer,t,1,NULL,PETSC_REAL));
  if (stepnum && !tj->solution_only) {
    Vec       *Y;
    PetscReal timepre;
    CHKERRQ(TSGetStages(ts,&ns,&Y));
    for (i=0; i<ns; i++) {
      /* For stiffly accurate TS methods, the last stage Y[ns-1] is the same as the solution X, thus does not need to be loaded again. */
      if (ts->stifflyaccurate && i == ns-1) continue;
      CHKERRQ(VecLoad(Y[i],viewer));
    }
    CHKERRQ(PetscViewerBinaryRead(viewer,&timepre,1,NULL,PETSC_REAL));
    if (tj->adjoint_solve_mode) {
      CHKERRQ(TSSetTimeStep(ts,-(*t)+timepre));
    }
  }
  /* Tangent linear sensitivities needed by second-order adjoint */
  if (ts->forward_solve) {

    if (!ts->stifflyaccurate) {
      Mat A;
      CHKERRQ(TSForwardGetSensitivities(ts,NULL,&A));
      CHKERRQ(MatLoad(A,viewer));
    }
    if (stepnum) {
      Mat *S;
      CHKERRQ(TSForwardGetStages(ts,&ns,&S));
      for (i=0; i<ns; i++) {
        CHKERRQ(MatLoad(S[i],viewer));
      }
    }
  }
  CHKERRQ(PetscViewerDestroy(&viewer));
  PetscFunctionReturn(0);
}

PetscErrorCode TSTrajectorySetUp_Basic(TSTrajectory tj,TS ts)
{
  MPI_Comm       comm;
  PetscMPIInt    rank;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetComm((PetscObject)tj,&comm));
  CHKERRMPI(MPI_Comm_rank(comm,&rank));
  if (rank == 0) {
    char      *dir = tj->dirname;
    PetscBool flg;

    if (!dir) {
      char dtempname[16] = "TS-data-XXXXXX";
      CHKERRQ(PetscMkdtemp(dtempname));
      CHKERRQ(PetscStrallocpy(dtempname,&tj->dirname));
    } else {
      CHKERRQ(PetscTestDirectory(dir,'w',&flg));
      if (!flg) {
        CHKERRQ(PetscTestFile(dir,'r',&flg));
        PetscCheck(!flg,PETSC_COMM_SELF,PETSC_ERR_USER,"Specified path is a file - not a dir: %s",dir);
        CHKERRQ(PetscMkdir(dir));
      } else SETERRQ(comm,PETSC_ERR_SUP,"Directory %s not empty",tj->dirname);
    }
  }
  CHKERRQ(PetscBarrier((PetscObject)tj));
  PetscFunctionReturn(0);
}

static PetscErrorCode TSTrajectoryDestroy_Basic(TSTrajectory tj)
{
  TSTrajectory_Basic *tjbasic = (TSTrajectory_Basic*)tj->data;

  PetscFunctionBegin;
  CHKERRQ(PetscViewerDestroy(&tjbasic->viewer));
  CHKERRQ(PetscFree(tjbasic));
  PetscFunctionReturn(0);
}

/*MC
      TSTRAJECTORYBASIC - Stores each solution of the ODE/DAE in a file

      Saves each timestep into a separate file named TS-data-XXXXXX/TS-%06d.bin. The file name can be changed.

      This version saves the solutions at all the stages

      $PETSC_DIR/share/petsc/matlab/PetscReadBinaryTrajectory.m can read in files created with this format

  Level: intermediate

.seealso:  TSTrajectoryCreate(), TS, TSTrajectorySetType(), TSTrajectorySetDirname(), TSTrajectorySetFile()

M*/
PETSC_EXTERN PetscErrorCode TSTrajectoryCreate_Basic(TSTrajectory tj,TS ts)
{
  TSTrajectory_Basic *tjbasic;

  PetscFunctionBegin;
  CHKERRQ(PetscNew(&tjbasic));

  CHKERRQ(PetscViewerCreate(PetscObjectComm((PetscObject)tj),&tjbasic->viewer));
  CHKERRQ(PetscViewerSetType(tjbasic->viewer,PETSCVIEWERBINARY));
  CHKERRQ(PetscViewerPushFormat(tjbasic->viewer,PETSC_VIEWER_NATIVE));
  CHKERRQ(PetscViewerFileSetMode(tjbasic->viewer,FILE_MODE_WRITE));
  tj->data = tjbasic;

  tj->ops->set            = TSTrajectorySet_Basic;
  tj->ops->get            = TSTrajectoryGet_Basic;
  tj->ops->setup          = TSTrajectorySetUp_Basic;
  tj->ops->destroy        = TSTrajectoryDestroy_Basic;
  tj->ops->setfromoptions = TSTrajectorySetFromOptions_Basic;
  PetscFunctionReturn(0);
}
