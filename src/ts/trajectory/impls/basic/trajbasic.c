
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
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PetscSNPrintf(filename,sizeof(filename),tj->dirfiletemplate,stepnum);CHKERRQ(ierr);
  ierr = PetscViewerFileSetName(tjbasic->viewer,filename);CHKERRQ(ierr); /* this triggers PetscViewer to be set up again */
  ierr = PetscViewerSetUp(tjbasic->viewer);CHKERRQ(ierr);
  ierr = VecView(X,tjbasic->viewer);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(tjbasic->viewer,&time,1,PETSC_REAL);CHKERRQ(ierr);
  if (stepnum && !tj->solution_only) {
    Vec       *Y;
    PetscReal tprev;
    ierr = TSGetStages(ts,&ns,&Y);CHKERRQ(ierr);
    for (i=0; i<ns; i++) {
      /* For stiffly accurate TS methods, the last stage Y[ns-1] is the same as the solution X, thus does not need to be saved again. */
      if (ts->stifflyaccurate && i == ns-1) continue;
      ierr = VecView(Y[i],tjbasic->viewer);CHKERRQ(ierr);
    }
    ierr = TSGetPrevTime(ts,&tprev);CHKERRQ(ierr);
    ierr = PetscViewerBinaryWrite(tjbasic->viewer,&tprev,1,PETSC_REAL);CHKERRQ(ierr);
  }
  /* Tangent linear sensitivities needed by second-order adjoint */
  if (ts->forward_solve) {
    Mat A,*S;

    ierr = TSForwardGetSensitivities(ts,NULL,&A);CHKERRQ(ierr);
    ierr = MatView(A,tjbasic->viewer);CHKERRQ(ierr);
    if (stepnum) {
      ierr = TSForwardGetStages(ts,&ns,&S);CHKERRQ(ierr);
      for (i=0; i<ns; i++) {
        ierr = MatView(S[i],tjbasic->viewer);CHKERRQ(ierr);
      }
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TSTrajectorySetFromOptions_Basic(PetscOptionItems *PetscOptionsObject,TSTrajectory tj)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"TS trajectory options for Basic type");CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSTrajectoryGet_Basic(TSTrajectory tj,TS ts,PetscInt stepnum,PetscReal *t)
{
  PetscViewer    viewer;
  char           filename[PETSC_MAX_PATH_LEN];
  PetscErrorCode ierr;
  Vec            Sol;
  PetscInt       ns,i;

  PetscFunctionBegin;
  ierr = PetscSNPrintf(filename,sizeof(filename),tj->dirfiletemplate,stepnum);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PetscObjectComm((PetscObject)tj),filename,FILE_MODE_READ,&viewer);CHKERRQ(ierr);
  ierr = PetscViewerPushFormat(viewer,PETSC_VIEWER_NATIVE);CHKERRQ(ierr);
  ierr = TSGetSolution(ts,&Sol);CHKERRQ(ierr);
  ierr = VecLoad(Sol,viewer);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,t,1,NULL,PETSC_REAL);CHKERRQ(ierr);
  if (stepnum && !tj->solution_only) {
    Vec       *Y;
    PetscReal timepre;
    ierr = TSGetStages(ts,&ns,&Y);CHKERRQ(ierr);
    for (i=0; i<ns; i++) {
      /* For stiffly accurate TS methods, the last stage Y[ns-1] is the same as the solution X, thus does not need to be loaded again. */
      if (ts->stifflyaccurate && i == ns-1) continue;
      ierr = VecLoad(Y[i],viewer);CHKERRQ(ierr);
    }
    ierr = PetscViewerBinaryRead(viewer,&timepre,1,NULL,PETSC_REAL);CHKERRQ(ierr);
    if (tj->adjoint_solve_mode) {
      ierr = TSSetTimeStep(ts,-(*t)+timepre);CHKERRQ(ierr);
    }
  }
  /* Tangent linear sensitivities needed by second-order adjoint */
  if (ts->forward_solve) {

    if (!ts->stifflyaccurate) {
      Mat A;
      ierr = TSForwardGetSensitivities(ts,NULL,&A);CHKERRQ(ierr);
      ierr = MatLoad(A,viewer);CHKERRQ(ierr);
    }
    if (stepnum) {
      Mat *S;
      ierr = TSForwardGetStages(ts,&ns,&S);CHKERRQ(ierr);
      for (i=0; i<ns; i++) {
        ierr = MatLoad(S[i],viewer);CHKERRQ(ierr);
      }
    }
  }
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode TSTrajectorySetUp_Basic(TSTrajectory tj,TS ts)
{
  MPI_Comm       comm;
  PetscMPIInt    rank;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)tj,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRMPI(ierr);
  if (!rank) {
    char      *dir = tj->dirname;
    PetscBool flg;

    if (!dir) {
      char dtempname[16] = "TS-data-XXXXXX";
      ierr = PetscMkdtemp(dtempname);CHKERRQ(ierr);
      ierr = PetscStrallocpy(dtempname,&tj->dirname);CHKERRQ(ierr);
    } else {
      ierr = PetscTestDirectory(dir,'w',&flg);CHKERRQ(ierr);
      if (!flg) {
        ierr = PetscTestFile(dir,'r',&flg);CHKERRQ(ierr);
        if (flg) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"Specified path is a file - not a dir: %s",dir);
        ierr = PetscMkdir(dir);CHKERRQ(ierr);
      } else SETERRQ1(comm,PETSC_ERR_SUP,"Directory %s not empty",tj->dirname);
    }
  }
  ierr = PetscBarrier((PetscObject)tj);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSTrajectoryDestroy_Basic(TSTrajectory tj)
{
  TSTrajectory_Basic *tjbasic = (TSTrajectory_Basic*)tj->data;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PetscViewerDestroy(&tjbasic->viewer);CHKERRQ(ierr);
  ierr = PetscFree(tjbasic);CHKERRQ(ierr);
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
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PetscNew(&tjbasic);CHKERRQ(ierr);

  ierr = PetscViewerCreate(PetscObjectComm((PetscObject)tj),&tjbasic->viewer);CHKERRQ(ierr);
  ierr = PetscViewerSetType(tjbasic->viewer,PETSCVIEWERBINARY);CHKERRQ(ierr);
  ierr = PetscViewerPushFormat(tjbasic->viewer,PETSC_VIEWER_NATIVE);CHKERRQ(ierr);
  ierr = PetscViewerFileSetMode(tjbasic->viewer,FILE_MODE_WRITE);CHKERRQ(ierr);
  tj->data = tjbasic;

  tj->ops->set            = TSTrajectorySet_Basic;
  tj->ops->get            = TSTrajectoryGet_Basic;
  tj->ops->setup          = TSTrajectorySetUp_Basic;
  tj->ops->destroy        = TSTrajectoryDestroy_Basic;
  tj->ops->setfromoptions = TSTrajectorySetFromOptions_Basic;
  PetscFunctionReturn(0);
}
