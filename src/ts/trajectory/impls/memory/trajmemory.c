#define PRINTWHATTODO
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
  PetscInt     top;         /* The top of the stack */
  PetscInt     max_cps; /* The maximum stack size */
  PetscInt     numY;
  MPI_Comm     comm;
  StackElement *stack;      /* The storage */
} Stack;

static PetscErrorCode StackCreate(MPI_Comm,Stack *,PetscInt,PetscInt);
static PetscErrorCode StackDestroy(Stack*);
static PetscErrorCode StackPush(Stack*,StackElement);
static PetscErrorCode StackPop(Stack*,StackElement*);
static PetscErrorCode StackTop(Stack*,StackElement*);

#ifdef PRINTWHATTODO
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
  s->max_cps = size;
  s->comm        = comm;
  s->numY        = ny;

  ierr = PetscMalloc1(s->max_cps*sizeof(StackElement),&s->stack);CHKERRQ(ierr);
  ierr = PetscMemzero(s->stack,s->max_cps*sizeof(StackElement));CHKERRQ(ierr);
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
#define __FUNCT__ "TSTrajectorySetMaxCheckpoints_Memory"
PetscErrorCode TSTrajectorySetMaxCheckpoints_Memory(TSTrajectory tj,PetscInt max_cps)
{
  Stack      *s = (Stack*)tj->data;
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
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSTrajectorySet_Memory"
PetscErrorCode TSTrajectorySet_Memory(TSTrajectory tj,TS ts,PetscInt stepnum,PetscReal time,Vec X)
{
  PetscInt       ns,i;
  Vec            *Y;
  PetscReal      timeprev;
  StackElement   e;
  Stack          *s = (Stack*)tj->data;
  RevolveCTX     *rctx;
  PetscInt       whattodo,rank;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (stepnum<s->top) SETERRQ(s->comm,PETSC_ERR_MEMC,"Illegal modification of a non-top stack element");

  if (s->userevolve) {
    rctx = s->rctx;
    if (rctx->reverseonestep) PetscFunctionReturn(0);
    if (rctx->stepsleft==0) { /* let the controller determine what to do next */
      rctx->capo = stepnum;
      rctx->oldcapo = rctx->capo;
      whattodo = wrap_revolve(&rctx->check,&rctx->capo,&rctx->fine,&rctx->snaps_in,&rctx->info,&rank);
#ifdef PRINTWHATTODO
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
#ifdef PRINTWHATTODO
        printwhattodo(whattodo,rctx);
#endif
        rctx->stepsleft = rctx->capo-rctx->oldcapo;
        PetscFunctionReturn(0);
      }
      if (whattodo==2) {
        rctx->oldcapo = rctx->capo;
        whattodo = wrap_revolve(&rctx->check,&rctx->capo,&rctx->fine,&rctx->snaps_in,&rctx->info,&rank); /* must return 1*/
#ifdef PRINTWHATTODO
        printwhattodo(whattodo,rctx);
#endif
        rctx->stepsleft = rctx->capo-rctx->oldcapo-1;
      }
    } else { /* advance s->stepsleft time steps without checkpointing */
      rctx->stepsleft--;
      PetscFunctionReturn(0);
    }
  }

  /* checkpoint to memmory */
  if (stepnum==s->top) { /* overwrite the top checkpoint */
    ierr = StackTop(s,&e);
    ierr = VecCopy(X,e->X);CHKERRQ(ierr);
    ierr = TSGetStages(ts,&ns,&Y);CHKERRQ(ierr);
    for (i=0;i<ns;i++) {
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
    ierr = TSGetStages(ts,&ns,&Y);CHKERRQ(ierr);
    ierr = VecDuplicateVecs(Y[0],ns,&e->Y);CHKERRQ(ierr);
    for (i=0;i<ns;i++) {
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
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSTrajectoryGet_Memory"
PetscErrorCode TSTrajectoryGet_Memory(TSTrajectory tj,TS ts,PetscInt stepnum,PetscReal *t)
{
  Vec            *Y;
  PetscInt       nr,i;
  StackElement   e;
  Stack          *s = (Stack*)tj->data;
  RevolveCTX     *rctx;
  PetscReal      stepsize;
  PetscInt       whattodo,rank;
  PetscErrorCode ierr;

  PetscFunctionBegin;
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
  ierr = TSGetStages(ts,&nr,&Y);CHKERRQ(ierr);
  for (i=0;i<nr ;i++) {
    ierr = VecCopy(e->Y[i],Y[i]);CHKERRQ(ierr);
  }
  *t = e->time;

  if (e->stepnum < stepnum) { /* need recomputation */
    rctx->capo = stepnum;
    whattodo = wrap_revolve(&rctx->check,&rctx->capo,&rctx->fine,&rctx->snaps_in,&rctx->info,&rank);
#ifdef PRINTWHATTODO
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
  PetscInt       nr,maxsteps;
  Stack          *s;
  RevolveCTX     *rctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  tj->ops->set            = TSTrajectorySet_Memory;
  tj->ops->get            = TSTrajectoryGet_Memory;
  tj->ops->destroy        = TSTrajectoryDestroy_Memory;
  tj->ops->setfromoptions = TSTrajectorySetFromOptions_Memory;

  ierr = PetscCalloc1(1,&s);CHKERRQ(ierr);
  s->max_cps = 3; /* will be provided by users */
  ierr = TSGetStages(ts,&nr,PETSC_IGNORE);CHKERRQ(ierr);

  maxsteps = PetscMin(ts->max_steps,(PetscInt)(ceil(ts->max_time/ts->time_step)));
  if (s->max_cps-1<maxsteps) { /* Need to use revolve */
    s->userevolve  = PETSC_TRUE;
    ierr = PetscCalloc1(1,&rctx);CHKERRQ(ierr);
    s->rctx = rctx;
    rctx->snaps_in       = s->max_cps; /* for theta methods snaps_in=2*max_cps */
    rctx->reverseonestep = PETSC_FALSE;
    rctx->check          = -1;
    rctx->oldcapo        = 0;
    rctx->capo           = 0;
    rctx->fine           = maxsteps;
    rctx->info           = 2;
    ierr = StackCreate(PetscObjectComm((PetscObject)ts),s,s->max_cps,nr);CHKERRQ(ierr);
  } else { /* Enough space for checkpointing all time steps */
    s->userevolve = PETSC_FALSE;
    ierr = StackCreate(PetscObjectComm((PetscObject)ts),s,ts->max_steps+1,nr);CHKERRQ(ierr);
  }
  tj->data = s;
  PetscFunctionReturn(0);
}
