
#include <petsc/private/tsimpl.h>        /*I "petscts.h"  I*/
#include <petscsys.h>

typedef struct _StackElement {
  PetscInt  stepnum;
  Vec       X;
  Vec       *Y;
  PetscReal time;
  PetscReal timeprev;
} *StackElement; 

typedef struct _Stack {
   PetscInt     top;         /* The top of the stack */
   PetscInt     maxelements; /* The maximum stack size */
   PetscInt     numY;
   MPI_Comm     comm;
   StackElement *stack;      /* The storage */
 } Stack;

static PetscErrorCode StackCreate(MPI_Comm,Stack *,PetscInt,PetscInt);
static PetscErrorCode StackDestroy(Stack*);
static PetscErrorCode StackPush(Stack*,StackElement);
static PetscErrorCode StackPop(Stack*,StackElement*);
static PetscErrorCode StackTop(Stack*,StackElement*);

#undef __FUNCT__
#define __FUNCT__ "StackCreate"
static PetscErrorCode StackCreate(MPI_Comm comm,Stack *s,PetscInt size,PetscInt ny)
{
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  s->top         = -1;
  s->maxelements = size;
  s->comm        = comm;
  s->numY        = ny;

  ierr = PetscMalloc1(s->maxelements*sizeof(StackElement),&s->stack);CHKERRQ(ierr);
  ierr = PetscMemzero(s->stack,s->maxelements*sizeof(StackElement));CHKERRQ(ierr);
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
  ierr = PetscFree(s);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "StackPush"
static PetscErrorCode StackPush(Stack *s,StackElement e)
{
  PetscFunctionBegin;
  if (s->top+1 >= s->maxelements) SETERRQ1(s->comm,PETSC_ERR_MEMC,"Maximum stack size (%D) exceeded",s->maxelements);
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
#define __FUNCT__ "TSTrajectorySet_Memory"
PetscErrorCode TSTrajectorySet_Memory(TSTrajectory tj,TS ts,PetscInt stepnum,PetscReal time,Vec X)
{
  PetscInt       ns,i;
  Vec            *Y;
  PetscReal      timeprev;
  StackElement   e;
  Stack          *s = (Stack*)tj->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (stepnum<s->top) SETERRQ(s->comm,PETSC_ERR_MEMC,"Illegal modification of a non-top stack element");
  
  if (stepnum > (rand()%ts->max_steps)) PetscFunctionReturn(0);/* choose check points randomly */
  
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  
  ierr = StackTop(s,&e);CHKERRQ(ierr);
  ierr = VecCopy(e->X,ts->vec_sol);CHKERRQ(ierr);
  ierr = TSGetStages(ts,&nr,&Y);CHKERRQ(ierr);
  for (i=0;i<nr ;i++) {
    ierr = VecCopy(e->Y[i],Y[i]);CHKERRQ(ierr);
  }
  *t = e->time;

  if (e->stepnum < stepnum) { /* go forward */
    ierr = TSSetTimeStep(ts,(*t)-e->timeprev);CHKERRQ(ierr);
    /* reset ts context */
    PetscInt steps = ts->steps;
    ts->steps      = e->stepnum; 
    ts->ptime      = e->time;
    ts->ptime_prev = e->timeprev; 

    /* recompute to the specified step */
    for (i=e->stepnum;i<stepnum;i++) { /* assume fixed step size */
      ierr = TSMonitor(ts,ts->steps,ts->ptime,ts->vec_sol);CHKERRQ(ierr);
      ierr = TSStep(ts);CHKERRQ(ierr);
      if (ts->event) {
        ierr = TSEventMonitor(ts);CHKERRQ(ierr);
      }
      if (!ts->steprollback) {
        if (i+1<stepnum) { /* skip checkpointing at the last step */
          ierr = TSTrajectorySet(ts->trajectory,ts,ts->steps,ts->ptime,ts->vec_sol);CHKERRQ(ierr);
        }
        ierr = TSPostStep(ts);CHKERRQ(ierr);
      }
    }
    ts->steps = steps;
    ts->total_steps = stepnum;
    PetscReal stepsize;
    ierr = TSGetTimeStep(ts,&stepsize);CHKERRQ(ierr);
    ierr = TSSetTimeStep(ts,-stepsize);CHKERRQ(ierr); /* go backward */
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
  PetscInt       nr;
  Stack          *s;
  PetscErrorCode ierr;
 
  PetscFunctionBegin;
  tj->ops->set     = TSTrajectorySet_Memory;
  tj->ops->get     = TSTrajectoryGet_Memory;
  tj->ops->destroy = TSTrajectoryDestroy_Memory;
 
  ierr = PetscCalloc1(1,&s);
  ierr = TSGetStages(ts,&nr,PETSC_IGNORE);CHKERRQ(ierr);
  ierr = StackCreate(PetscObjectComm((PetscObject)ts),s,ts->max_steps+1,nr);CHKERRQ(ierr);

  tj->data = s;
  PetscFunctionReturn(0);
}
