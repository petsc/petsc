#define TJ_VERBOSE
#include <petsc/private/tsimpl.h>        /*I "petscts.h"  I*/
#include <petscsys.h>

extern int wrap_revolve(int* check,int* capo,int* fine,int *snaps_in,int* info,int* rank);

typedef struct _StackElement {
  PetscInt  stepnum;
  Vec       X;
  Vec       *Y;
  PetscReal time;
  PetscReal timeprev;
} *StackElement;

typedef struct _RevolveCTX {
  PetscBool    reverseonestep;
  PetscInt     snaps_in;
  PetscInt     stepsleft;
  PetscInt     check;
  PetscInt     oldcapo;
  PetscInt     capo;
  PetscInt     fine;
  PetscInt     info;
} RevolveCTX;

typedef struct _Stack {
  PetscBool    userevolve;
  RevolveCTX   *rctx;
  PetscInt     top;         /* top of the stack */
  PetscInt     max_cps;     /* maximum stack size */
  PetscInt     numY;
  PetscInt     stride;
  PetscInt     total_steps; /* total number of steps */
  MPI_Comm     comm;
  StackElement *stack;      /* container */
} Stack;

static PetscErrorCode StackCreate(MPI_Comm,Stack *,PetscInt,PetscInt);
static PetscErrorCode StackDestroy(Stack*);
static PetscErrorCode StackPush(Stack*,StackElement);
static PetscErrorCode StackPop(Stack*,StackElement*);
static PetscErrorCode StackTop(Stack*,StackElement*);
static PetscErrorCode StackDumpAll(Stack*,PetscInt);
static PetscErrorCode StackLoadAll(TS,Stack*,PetscInt);

#ifdef TJ_VERBOSE
static void printwhattodo(PetscInt whattodo,RevolveCTX *rctx)
{
  switch(whattodo) {
    case 1:
      PetscPrintf(PETSC_COMM_WORLD,"Advance from %D to %D.\n",rctx->oldcapo,rctx->capo);
      break;
    case 2:
      PetscPrintf(PETSC_COMM_WORLD,"Store in checkpoint number %D\n",rctx->check);
      break;
    case 3:
      PetscPrintf(PETSC_COMM_WORLD,"First turn: Initialize adjoints and reverse first step.\n");
      break;
    case 4:
      PetscPrintf(PETSC_COMM_WORLD,"Forward and reverse one step.\n");
      break;
    case 5:
      PetscPrintf(PETSC_COMM_WORLD,"Restore in checkpoint number %D\n",rctx->check);
      break;
    case -1:
      PetscPrintf(PETSC_COMM_WORLD,"Error!");
      break;
  }
}
#endif

#undef __FUNCT__
#define __FUNCT__ "StackCreate"
static PetscErrorCode StackCreate(MPI_Comm comm,Stack *s,PetscInt size,PetscInt ny)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  s->top         = -1;
  s->max_cps     = size;
  s->comm        = comm;
  s->numY        = ny;

  ierr = PetscMalloc1(s->max_cps*sizeof(StackElement),&s->stack);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "StackDestroy"
static PetscErrorCode StackDestroy(Stack *s)
{
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (s->top>-1) {
    for (i=0;i<=s->top;i++) {
      ierr = VecDestroy(&s->stack[i]->X);CHKERRQ(ierr);
      ierr = VecDestroyVecs(s->numY,&s->stack[i]->Y);CHKERRQ(ierr);
      ierr = PetscFree(s->stack[i]);CHKERRQ(ierr);
    }
  }
  ierr = PetscFree(s->stack);CHKERRQ(ierr);
  if (s->userevolve) {
    ierr = PetscFree(s->rctx);CHKERRQ(ierr);
  }
  ierr = PetscFree(s);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "StackPush"
static PetscErrorCode StackPush(Stack *s,StackElement e)
{
  PetscFunctionBegin;
  if (s->top+1 >= s->max_cps) SETERRQ1(s->comm,PETSC_ERR_MEMC,"Maximum stack size (%D) exceeded",s->max_cps);
  s->stack[++s->top] = e;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "StackPop"
static PetscErrorCode StackPop(Stack *s,StackElement *e)
{
  PetscFunctionBegin;
  if (s->top == -1) SETERRQ(s->comm,PETSC_ERR_MEMC,"Emptry stack");
  *e = s->stack[s->top--];
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "StackTop"
static PetscErrorCode StackTop(Stack *s,StackElement *e)
{
  PetscFunctionBegin;
  *e = s->stack[s->top];
  PetscFunctionReturn(0);
}

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
#define __FUNCT__ "StackDumpAll"
static PetscErrorCode StackDumpAll(Stack *s,PetscInt id)
{
  PetscInt       i,j;
  StackElement   e;
  PetscViewer    viewer;
  char           filename[PETSC_MAX_PATH_LEN];
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (id == 1) {
#if defined(PETSC_HAVE_POPEN)
    PetscMPIInt rank;
    ierr = MPI_Comm_rank(s->comm,&rank);CHKERRQ(ierr);
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
#endif
  }
  ierr = PetscSNPrintf(filename,sizeof(filename),"SA-data/SA-STACK%06d.bin",id);CHKERRQ(ierr);
  ierr = OutputBIN(filename,&viewer);CHKERRQ(ierr);
  for (i=0;i<s->stride;i++) {
    e = s->stack[i];
    ierr = PetscViewerBinaryWrite(viewer,&e->stepnum,1,PETSC_INT,PETSC_FALSE);CHKERRQ(ierr);
    ierr = VecView(e->X,viewer);CHKERRQ(ierr);
    for (j=0;j<s->numY;j++) {
      ierr = VecView(e->Y[j],viewer);CHKERRQ(ierr);
    }
    ierr = PetscViewerBinaryWrite(viewer,&e->time,1,PETSC_REAL,PETSC_FALSE);CHKERRQ(ierr);
    ierr = PetscViewerBinaryWrite(viewer,&e->timeprev,1,PETSC_REAL,PETSC_FALSE);CHKERRQ(ierr);
  }
  for (i=0;i<s->stride;i++) {
    ierr = StackPop(s,&e);CHKERRQ(ierr);
    ierr = VecDestroy(&e->X);CHKERRQ(ierr);
    ierr = VecDestroyVecs(s->numY,&e->Y);CHKERRQ(ierr);
    ierr = PetscFree(e);CHKERRQ(ierr);
  }
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "StackLoadAll"
static PetscErrorCode StackLoadAll(TS ts,Stack *s,PetscInt id)
{
  Vec            *Y;
  PetscInt       i,j;
  StackElement   e;
  PetscViewer    viewer;
  char           filename[PETSC_MAX_PATH_LEN];
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscSNPrintf(filename,sizeof filename,"SA-data/SA-STACK%06d.bin",id);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_READ,&viewer);CHKERRQ(ierr);

  for (i=0;i<s->stride;i++) {
    ierr = PetscCalloc1(1,&e);
    ierr = TSGetStages(ts,&s->numY,&Y);CHKERRQ(ierr);
    ierr = VecDuplicate(Y[0],&e->X);CHKERRQ(ierr);
    ierr = VecDuplicateVecs(Y[0],s->numY,&e->Y);CHKERRQ(ierr);
    ierr = StackPush(s,e);CHKERRQ(ierr);
  }
  for (i=0;i<s->stride;i++) {
    e = s->stack[i];
    ierr = PetscViewerBinaryRead(viewer,&e->stepnum,1,NULL,PETSC_INT);CHKERRQ(ierr);
    ierr = VecLoad(e->X,viewer);CHKERRQ(ierr);
    for (j=0;j<s->numY;j++) {
      ierr = VecLoad(e->Y[j],viewer);CHKERRQ(ierr);
    }
    ierr = PetscViewerBinaryRead(viewer,&e->time,1,NULL,PETSC_REAL);CHKERRQ(ierr);
    ierr = PetscViewerBinaryRead(viewer,&e->timeprev,1,NULL,PETSC_REAL);CHKERRQ(ierr);
  }
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSTrajectorySetStride_Memory"
PetscErrorCode TSTrajectorySetStride_Memory(TSTrajectory tj,TS ts,PetscInt stride)
{
  Stack    *s = (Stack*)tj->data;

  PetscFunctionBegin;
  s->stride = stride;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSTrajectorySetMaxCheckpoints_Memory"
PetscErrorCode TSTrajectorySetMaxCheckpoints_Memory(TSTrajectory tj,TS ts,PetscInt max_cps)
{
  Stack    *s = (Stack*)tj->data;

  PetscFunctionBegin;
  s->max_cps = max_cps;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSTrajectorySetFromOptions_Memory"
PetscErrorCode TSTrajectorySetFromOptions_Memory(PetscOptions *PetscOptionsObject,TSTrajectory tj)
{
  Stack     *s = (Stack*)tj->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"Memory based TS trajectory options");CHKERRQ(ierr);
  {
    ierr = PetscOptionsInt("-tstrajectory_max_cps","Maximum number of checkpoints","TSTrajectorySetMaxCheckpoints_Memory",s->max_cps,&s->max_cps,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-tstrajectory_stride","Stride to save checkpoints to file","TSTrajectorySetStride_Memory",s->stride,&s->stride,NULL);CHKERRQ(ierr);
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSTrajectorySetUp_Memory"
PetscErrorCode TSTrajectorySetUp_Memory(TSTrajectory tj,TS ts)
{
  Stack          *s = (Stack*)tj->data;
  RevolveCTX     *rctx;
  PetscInt       numY;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  ierr = TSGetStages(ts,&numY,NULL);CHKERRQ(ierr);
  s->total_steps = PetscMin(ts->max_steps,(PetscInt)(ceil(ts->max_time/ts->time_step)));

  if ((s->stride>1 && s->max_cps>1 && s->max_cps-1<s->stride)||(s->stride<=1 && s->max_cps>1 && s->max_cps-1<s->total_steps)) {
    s->userevolve  = PETSC_TRUE;
    ierr = PetscCalloc1(1,&rctx);CHKERRQ(ierr);
    s->rctx = rctx;
    rctx->snaps_in       = s->max_cps; /* for theta methods snaps_in=2*max_cps */
    rctx->reverseonestep = PETSC_FALSE;
    rctx->check          = -1;
    rctx->oldcapo        = 0;
    rctx->capo           = 0;
    rctx->info           = 2;
    if (s->stride>1) rctx->fine = s->stride;
    else rctx->fine = s->total_steps;
    ierr = StackCreate(PetscObjectComm((PetscObject)ts),s,s->max_cps,numY);CHKERRQ(ierr);
  } else {
    s->userevolve = PETSC_FALSE;
    if (s->stride>1) {
      ierr = StackCreate(PetscObjectComm((PetscObject)ts),s,s->stride+1,numY);CHKERRQ(ierr);
    } else {
      ierr = StackCreate(PetscObjectComm((PetscObject)ts),s,ts->max_steps+1,numY);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSTrajectorySet_Memory"
PetscErrorCode TSTrajectorySet_Memory(TSTrajectory tj,TS ts,PetscInt stepnum,PetscReal time,Vec X)
{
  PetscInt       i;
  Vec            *Y;
  PetscReal      timeprev;
  StackElement   e;
  Stack          *s = (Stack*)tj->data;
  RevolveCTX     *rctx;
  PetscInt       whattodo,rank,localstepnum,id;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (stepnum<s->top) SETERRQ(s->comm,PETSC_ERR_MEMC,"Illegal modification of a non-top stack element");

  if (s->stride>1) {
    localstepnum = stepnum%s->stride;
    if (s->userevolve && stepnum!=0 && localstepnum==0 && stepnum!=s->total_steps) { /* first turn point  */
      id     = stepnum/s->stride;
      ierr   = StackDumpAll(s,id);CHKERRQ(ierr);
      s->top = -1; /* reset top */
      rctx = s->rctx;
      rctx->check = -1;
      rctx->capo  = 0;
      rctx->fine  = s->stride;
    }
  } else {
    localstepnum = stepnum;
  }

  if (s->userevolve) {
    rctx = s->rctx;
    if (rctx->reverseonestep) {
      rctx->reverseonestep = PETSC_FALSE;
      PetscFunctionReturn(0);
    }
    if (rctx->stepsleft==0) { /* let the controller determine what to do next */
      rctx->capo = localstepnum;
      rctx->oldcapo = rctx->capo;
      whattodo = wrap_revolve(&rctx->check,&rctx->capo,&rctx->fine,&rctx->snaps_in,&rctx->info,&rank);
#ifdef TJ_VERBOSE
      printwhattodo(whattodo,rctx);
#endif
      if (whattodo==-1) SETERRQ(s->comm,PETSC_ERR_MEMC,"Error in the controller");
      if (whattodo==1) {
        rctx->stepsleft = rctx->capo-rctx->oldcapo-1;
        PetscFunctionReturn(0); /* do not need to checkpoint */
      }
      if (whattodo==3 || whattodo==4) {
        rctx->reverseonestep = PETSC_TRUE;
        PetscFunctionReturn(0);
      }
      if (whattodo==5) {
        rctx->oldcapo = rctx->capo;
        whattodo = wrap_revolve(&rctx->check,&rctx->capo,&rctx->fine,&rctx->snaps_in,&rctx->info,&rank); /* must return 1*/
#ifdef TJ_VERBOSE
        printwhattodo(whattodo,rctx);
#endif
        rctx->stepsleft = rctx->capo-rctx->oldcapo;
        PetscFunctionReturn(0);
      }
      if (whattodo==2) {
        rctx->oldcapo = rctx->capo;
        whattodo = wrap_revolve(&rctx->check,&rctx->capo,&rctx->fine,&rctx->snaps_in,&rctx->info,&rank); /* must return 1*/
#ifdef TJ_VERBOSE
        printwhattodo(whattodo,rctx);
#endif
        rctx->stepsleft = rctx->capo-rctx->oldcapo-1;
      }
    } else { /* advance s->stepsleft time steps without checkpointing */
      rctx->stepsleft--;
      PetscFunctionReturn(0);
    }
  }

  if (!s->userevolve && stepnum==0) PetscFunctionReturn(0);

  /* checkpoint to memmory */
  if (localstepnum==s->top) { /* overwrite the top checkpoint */
    ierr = StackTop(s,&e);
    ierr = VecCopy(X,e->X);CHKERRQ(ierr);
    ierr = TSGetStages(ts,&s->numY,&Y);CHKERRQ(ierr);
    for (i=0;i<s->numY;i++) {
      ierr = VecCopy(Y[i],e->Y[i]);CHKERRQ(ierr);
    }
    e->stepnum  = stepnum;
    e->time     = time;
    ierr        = TSGetPrevTime(ts,&timeprev);CHKERRQ(ierr);
    e->timeprev = timeprev;
  } else {
    ierr = PetscCalloc1(1,&e);
    ierr = VecDuplicate(X,&e->X);CHKERRQ(ierr);
    ierr = VecCopy(X,e->X);CHKERRQ(ierr);
    ierr = TSGetStages(ts,&s->numY,&Y);CHKERRQ(ierr);
    ierr = VecDuplicateVecs(Y[0],s->numY,&e->Y);CHKERRQ(ierr);
    for (i=0;i<s->numY;i++) {
      ierr = VecCopy(Y[i],e->Y[i]);CHKERRQ(ierr);
    }
    e->stepnum  = stepnum;
    e->time     = time;
    if (stepnum == 0) {
      e->timeprev = e->time - ts->time_step; /* for consistency */
    } else {
      ierr        = TSGetPrevTime(ts,&timeprev);CHKERRQ(ierr);
      e->timeprev = timeprev;
    }
    ierr        = StackPush(s,e);CHKERRQ(ierr);
  }
  if (!s->userevolve && stepnum!=0 && localstepnum==0 && stepnum!=s->total_steps) {
    id     = stepnum/s->stride;
    ierr   = StackDumpAll(s,id);CHKERRQ(ierr);
    s->top = -1; /* reset top */
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSTrajectoryGet_Memory"
PetscErrorCode TSTrajectoryGet_Memory(TSTrajectory tj,TS ts,PetscInt stepnum,PetscReal *t)
{
  Vec            *Y;
  PetscInt       i;
  StackElement   e;
  Stack          *s = (Stack*)tj->data;
  RevolveCTX     *rctx;
  PetscReal      stepsize;
  PetscInt       whattodo,rank,localstepnum,id;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (s->stride>1) {
    localstepnum = stepnum%s->stride;
    if (localstepnum==0 && stepnum!=0 && stepnum!=s->total_steps) {
      id     = stepnum/s->stride;
      ierr   = StackLoadAll(ts,s,id);CHKERRQ(ierr);
      ierr   = StackPop(s,&e);CHKERRQ(ierr); /* pop out stepnum 0 */
      s->top = s->stride-1;
      if (s->userevolve) {
        rctx = s->rctx;
        rctx->reverseonestep = PETSC_TRUE;
        rctx->check = s->top;
      }
    }
  }
  if (s->userevolve) rctx = s->rctx;
  if (s->userevolve && rctx->reverseonestep) {
    ierr = TSGetTimeStep(ts,&stepsize);CHKERRQ(ierr);
    ierr = TSSetTimeStep(ts,-stepsize);CHKERRQ(ierr); /* go backward */
    rctx->reverseonestep = PETSC_FALSE;
    PetscFunctionReturn(0);
  }

  /* restore a checkpoint */
  ierr = StackTop(s,&e);CHKERRQ(ierr);
  ierr = VecCopy(e->X,ts->vec_sol);CHKERRQ(ierr);
  ierr = TSGetStages(ts,&s->numY,&Y);CHKERRQ(ierr);
  for (i=0;i<s->numY;i++) {
    ierr = VecCopy(e->Y[i],Y[i]);CHKERRQ(ierr);
  }
  *t = e->time;

  if (e->stepnum < stepnum) { /* need recomputation */
    rctx->capo = stepnum;
    whattodo = wrap_revolve(&rctx->check,&rctx->capo,&rctx->fine,&rctx->snaps_in,&rctx->info,&rank);
#ifdef TJ_VERBOSE
    printwhattodo(whattodo,rctx);
#endif
    ierr = TSSetTimeStep(ts,(*t)-e->timeprev);CHKERRQ(ierr);
    /* reset ts context */
    PetscInt steps = ts->steps;
    ts->steps      = e->stepnum;
    ts->ptime      = e->time;
    ts->ptime_prev = e->timeprev;
    for (i=e->stepnum;i<stepnum;i++) { /* assume fixed step size */
      ierr = TSTrajectorySet(ts->trajectory,ts,ts->steps,ts->ptime,ts->vec_sol);CHKERRQ(ierr);
      ierr = TSMonitor(ts,ts->steps,ts->ptime,ts->vec_sol);CHKERRQ(ierr);
      ierr = TSStep(ts);CHKERRQ(ierr);
      if (ts->event) {
        ierr = TSEventMonitor(ts);CHKERRQ(ierr);
      }
      if (!ts->steprollback) {
        ierr = TSPostStep(ts);CHKERRQ(ierr);
      }
    }
    /* reverseonestep must be true after the for loop */
    ts->steps = steps;
    ts->total_steps = stepnum;
    ierr = TSGetTimeStep(ts,&stepsize);CHKERRQ(ierr);
    ierr = TSSetTimeStep(ts,-stepsize);CHKERRQ(ierr); /* go backward */
    if (stepnum-e->stepnum==1) {
      ierr = StackPop(s,&e);CHKERRQ(ierr);
      ierr = VecDestroy(&e->X);CHKERRQ(ierr);
      ierr = VecDestroyVecs(s->numY,&e->Y);CHKERRQ(ierr);
      ierr = PetscFree(e);CHKERRQ(ierr);
    }
    rctx->reverseonestep = PETSC_FALSE;
  } else if (e->stepnum == stepnum) {
    ierr = TSSetTimeStep(ts,-(*t)+e->timeprev);CHKERRQ(ierr); /* go backward */
    ierr = StackPop(s,&e);CHKERRQ(ierr);
    ierr = VecDestroy(&e->X);CHKERRQ(ierr);
    ierr = VecDestroyVecs(s->numY,&e->Y);CHKERRQ(ierr);
    ierr = PetscFree(e);CHKERRQ(ierr);
  } else {
    SETERRQ2(s->comm,PETSC_ERR_ARG_OUTOFRANGE,"The current step no. is %D, but the step number at top of the stack is %D",stepnum,e->stepnum);
  }

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSTrajectoryDestroy_Memory"
PETSC_EXTERN PetscErrorCode TSTrajectoryDestroy_Memory(TSTrajectory tj)
{
  Stack          *s = (Stack*)tj->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = StackDestroy(s);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
      TSTRAJECTORYMEMORY - Stores each solution of the ODE/ADE in memory

  Level: intermediate

.seealso:  TSTrajectoryCreate(), TS, TSTrajectorySetType()

M*/
#undef __FUNCT__
#define __FUNCT__ "TSTrajectoryCreate_Memory"
PETSC_EXTERN PetscErrorCode TSTrajectoryCreate_Memory(TSTrajectory tj,TS ts)
{
  Stack          *s;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  tj->ops->set            = TSTrajectorySet_Memory;
  tj->ops->get            = TSTrajectoryGet_Memory;
  tj->ops->setup          = TSTrajectorySetUp_Memory;
  tj->ops->destroy        = TSTrajectoryDestroy_Memory;
  tj->ops->setfromoptions = TSTrajectorySetFromOptions_Memory;

  ierr = PetscCalloc1(1,&s);CHKERRQ(ierr);
  s->max_cps = -1; /* -1 indicates that it is not set */
  s->stride  = 0; /* if not zero, two-level checkpointing will be used */
  tj->data   = s;
  PetscFunctionReturn(0);
}
