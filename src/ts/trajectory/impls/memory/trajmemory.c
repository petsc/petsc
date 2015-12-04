#include <petsc/private/tsimpl.h>        /*I "petscts.h"  I*/
#include <petscsys.h>
#ifdef PETSC_HAVE_REVOLVE
#include <revolve_c.h>
#endif

PetscLogEvent Disk_Write, Disk_Read;

typedef enum {NONE,TWO_LEVEL_NOREVOLVE,TWO_LEVEL_REVOLVE,REVOLVE_OFFLINE,REVOLVE_ONLINE,REVOLVE_MULTISTAGE} SchedulerType;

typedef struct _StackElement {
  PetscInt  stepnum;
  Vec       X;
  Vec       *Y;
  PetscReal time;
  PetscReal timeprev;
} *StackElement;

#ifdef PETSC_HAVE_REVOLVE
typedef struct _RevolveCTX {
  PetscBool reverseonestep;
  PetscInt  where;
  PetscInt  snaps_in;
  PetscInt  stepsleft;
  PetscInt  check;
  PetscInt  oldcapo;
  PetscInt  capo;
  PetscInt  fine;
  PetscInt  info;
} RevolveCTX;
#endif

typedef struct _Stack {
  PetscInt      stacksize;
  PetscInt      top;
  StackElement  *container;
  PetscInt      numY;
  PetscBool     solution_only;
} Stack;

typedef struct _DiskStack {
  PetscInt  stacksize;
  PetscInt  top;
  PetscInt  *container;
} DiskStack;

typedef struct _TJScheduler {
  SchedulerType stype;
#ifdef PETSC_HAVE_REVOLVE
  RevolveCTX    *rctx,*rctx2;
  PetscBool     use_online;
#endif
  PetscBool     recompute;
  PetscBool     skip_trajectory;
  PetscBool     save_stack;
  MPI_Comm      comm;
  PetscInt      max_cps_ram;  /* maximum checkpoints in RAM */
  PetscInt      max_cps_disk; /* maximum checkpoints on disk */
  PetscInt      stride;
  PetscInt      total_steps;  /* total number of steps */
  Stack         stack;
  DiskStack     diskstack;
} TJScheduler;

#undef __FUNCT__
#define __FUNCT__ "StackCreate"
static PetscErrorCode StackCreate(Stack *stack,PetscInt size,PetscInt ny)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  stack->top  = -1;
  stack->numY = ny;

  ierr = PetscMalloc1(size*sizeof(StackElement),&stack->container);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "StackDestroy"
static PetscErrorCode StackDestroy(Stack *stack)
{
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (stack->top>-1) {
    for (i=0;i<=stack->top;i++) {
      ierr = VecDestroy(&stack->container[i]->X);CHKERRQ(ierr);
      if (!stack->solution_only) {
        ierr = VecDestroyVecs(stack->numY,&stack->container[i]->Y);CHKERRQ(ierr);
      }
      ierr = PetscFree(stack->container[i]);CHKERRQ(ierr);
    }
  }
  ierr = PetscFree(stack->container);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "StackPush"
static PetscErrorCode StackPush(Stack *stack,StackElement e)
{
  PetscFunctionBegin;
  if (stack->top+1 >= stack->stacksize) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_MEMC,"Maximum stack size (%D) exceeded",stack->stacksize);
  stack->container[++stack->top] = e;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "StackPop"
static PetscErrorCode StackPop(Stack *stack,StackElement *e)
{
  PetscFunctionBegin;
  if (stack->top == -1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEMC,"Empty stack");
  *e = stack->container[stack->top--];
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "StackTop"
static PetscErrorCode StackTop(Stack *stack,StackElement *e)
{
  PetscFunctionBegin;
  *e = stack->container[stack->top];
  PetscFunctionReturn(0);
}

#ifdef PETSC_HAVE_REVOLVE
#undef __FUNCT__
#define __FUNCT__ "StackFind"
static PetscErrorCode StackFind(Stack *stack,StackElement *e,PetscInt index)
{
  PetscFunctionBegin;
  *e = stack->container[index];
  PetscFunctionReturn(0);
}
#endif

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
#define __FUNCT__ "WriteToDisk"
static PetscErrorCode WriteToDisk(PetscInt stepnum,PetscReal time,PetscReal timeprev,Vec X,Vec *Y,PetscInt numY,PetscBool solution_only,PetscViewer viewer)
{
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscViewerBinaryWrite(viewer,&stepnum,1,PETSC_INT,PETSC_FALSE);CHKERRQ(ierr);
  ierr = VecView(X,viewer);CHKERRQ(ierr);
  for (i=0;!solution_only && i<numY;i++) {
    ierr = VecView(Y[i],viewer);CHKERRQ(ierr);
  }
  ierr = PetscViewerBinaryWrite(viewer,&time,1,PETSC_REAL,PETSC_FALSE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(viewer,&timeprev,1,PETSC_REAL,PETSC_FALSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ReadFromDisk"
static PetscErrorCode ReadFromDisk(PetscInt *stepnum,PetscReal *time,PetscReal *timeprev,Vec X,Vec *Y,PetscInt numY,PetscBool solution_only,PetscViewer viewer)
{
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscViewerBinaryRead(viewer,stepnum,1,NULL,PETSC_INT);CHKERRQ(ierr);
  ierr = VecLoad(X,viewer);CHKERRQ(ierr);
  for (i=0;!solution_only && i<numY;i++) {
    ierr = VecLoad(Y[i],viewer);CHKERRQ(ierr);
  }
  ierr = PetscViewerBinaryRead(viewer,time,1,NULL,PETSC_REAL);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,timeprev,1,NULL,PETSC_REAL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "StackDumpAll"
static PetscErrorCode StackDumpAll(TS ts,Stack *stack,PetscInt id)
{
  Vec            *Y;
  PetscInt       i;
  StackElement   e;
  PetscViewer    viewer;
  char           filename[PETSC_MAX_PATH_LEN];
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (id == 1) {
    PetscMPIInt rank;
    ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)ts),&rank);CHKERRQ(ierr);
    if (!rank) {
      ierr = PetscRMTree("SA-data");CHKERRQ(ierr);
      ierr = PetscMkdir("SA-data");CHKERRQ(ierr);
    }
  }
  ierr = PetscSNPrintf(filename,sizeof(filename),"SA-data/SA-STACK%06d.bin",id);CHKERRQ(ierr);
  ierr = OutputBIN(filename,&viewer);CHKERRQ(ierr);
  for (i=0;i<stack->stacksize;i++) {
    e = stack->container[i];
    ierr = WriteToDisk(e->stepnum,e->time,e->timeprev,e->X,e->Y,stack->numY,stack->solution_only,viewer);CHKERRQ(ierr);
  }
  /* save the last step for restart, the last step is in memory when using single level schemes, but not necessarily the case for multi level schemes */
  ierr = TSGetStages(ts,&stack->numY,&Y);CHKERRQ(ierr);
  ierr = WriteToDisk(ts->total_steps,ts->ptime,ts->ptime_prev,ts->vec_sol,Y,stack->numY,stack->solution_only,viewer);CHKERRQ(ierr);
  for (i=0;i<stack->stacksize;i++) {
    ierr = StackPop(stack,&e);CHKERRQ(ierr);
    ierr = VecDestroy(&e->X);CHKERRQ(ierr);
    if (!stack->solution_only) {
      ierr = VecDestroyVecs(stack->numY,&e->Y);CHKERRQ(ierr);
    }
    ierr = PetscFree(e);CHKERRQ(ierr);
  }
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "StackLoadAll"
static PetscErrorCode StackLoadAll(TS ts,Stack *stack,PetscInt id)
{
  Vec            *Y;
  PetscInt       i;
  StackElement   e;
  PetscViewer    viewer;
  char           filename[PETSC_MAX_PATH_LEN];
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscSNPrintf(filename,sizeof filename,"SA-data/SA-STACK%06d.bin",id);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_READ,&viewer);CHKERRQ(ierr);
  for (i=0;i<stack->stacksize;i++) {
    ierr = PetscCalloc1(1,&e);
    ierr = TSGetStages(ts,&stack->numY,&Y);CHKERRQ(ierr);
    ierr = VecDuplicate(Y[0],&e->X);CHKERRQ(ierr);
    if (!stack->solution_only && stack->numY>0) {
      ierr = VecDuplicateVecs(Y[0],stack->numY,&e->Y);CHKERRQ(ierr);
    }
    ierr = StackPush(stack,e);CHKERRQ(ierr);
  }
  for (i=0;i<stack->stacksize;i++) {
    e = stack->container[i];
    ierr = ReadFromDisk(&e->stepnum,&e->time,&e->timeprev,e->X,e->Y,stack->numY,stack->solution_only,viewer);CHKERRQ(ierr);
  }
  /* load the last step into TS */
  ierr = TSGetStages(ts,&stack->numY,&Y);CHKERRQ(ierr);
  ierr = ReadFromDisk(&ts->total_steps,&ts->ptime,&ts->ptime_prev,ts->vec_sol,Y,stack->numY,stack->solution_only,viewer);CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts,ts->ptime_prev-ts->ptime);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DumpSingle"
static PetscErrorCode DumpSingle(TS ts,Stack *stack,PetscInt id)
{
  Vec            *Y;
  PetscInt       stepnum;
  PetscViewer    viewer;
  char           filename[PETSC_MAX_PATH_LEN];
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSGetTotalSteps(ts,&stepnum);CHKERRQ(ierr);
  if (id == 0) {
    PetscMPIInt rank;
    ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)ts),&rank);CHKERRQ(ierr);
    if (!rank) {
      ierr = PetscRMTree("SA-data");CHKERRQ(ierr);
      ierr = PetscMkdir("SA-data");CHKERRQ(ierr);
    }
  }
  ierr = PetscSNPrintf(filename,sizeof(filename),"SA-data/SA-CPS%06d.bin",id);CHKERRQ(ierr);
  ierr = OutputBIN(filename,&viewer);CHKERRQ(ierr);

  ierr = PetscLogEventBegin(Disk_Write,ts,0,0,0);CHKERRQ(ierr);
  ierr = TSGetStages(ts,&stack->numY,&Y);CHKERRQ(ierr);
  ierr = WriteToDisk(stepnum,ts->ptime,ts->ptime_prev,ts->vec_sol,Y,stack->numY,stack->solution_only,viewer);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(Disk_Write,ts,0,0,0);CHKERRQ(ierr);

  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "LoadSingle"
static PetscErrorCode LoadSingle(TS ts,Stack *stack,PetscInt id)
{
  Vec            *Y;
  PetscViewer    viewer;
  char           filename[PETSC_MAX_PATH_LEN];
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscSNPrintf(filename,sizeof filename,"SA-data/SA-CPS%06d.bin",id);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_READ,&viewer);CHKERRQ(ierr);

  ierr = PetscLogEventBegin(Disk_Read,ts,0,0,0);CHKERRQ(ierr);
  ierr = TSGetStages(ts,&stack->numY,&Y);CHKERRQ(ierr);
  ierr = ReadFromDisk(&ts->total_steps,&ts->ptime,&ts->ptime_prev,ts->vec_sol,Y,stack->numY,stack->solution_only,viewer);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(Disk_Read,ts,0,0,0);CHKERRQ(ierr);

  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ElementCreate"
static PetscErrorCode ElementCreate(TS ts,Stack *stack,StackElement *e,PetscInt stepnum,PetscReal time,Vec X)
{
  Vec            *Y;
  PetscInt       i;
  PetscReal      timeprev;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscCalloc1(1,e);CHKERRQ(ierr);
  ierr = VecDuplicate(X,&(*e)->X);CHKERRQ(ierr);
  ierr = VecCopy(X,(*e)->X);CHKERRQ(ierr);
  if (stack->numY > 0 && !stack->solution_only) {
    ierr = TSGetStages(ts,&stack->numY,&Y);CHKERRQ(ierr);
    ierr = VecDuplicateVecs(Y[0],stack->numY,&(*e)->Y);CHKERRQ(ierr);
    for (i=0;i<stack->numY;i++) {
      ierr = VecCopy(Y[i],(*e)->Y[i]);CHKERRQ(ierr);
    }
  }
  (*e)->stepnum = stepnum;
  (*e)->time    = time;
  /* for consistency */
  if (stepnum == 0) {
    (*e)->timeprev = (*e)->time - ts->time_step;
  } else {
    ierr = TSGetPrevTime(ts,&timeprev);CHKERRQ(ierr);
    (*e)->timeprev = timeprev;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ElementDestroy"
static PetscErrorCode ElementDestroy(Stack *stack,StackElement e)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDestroy(&e->X);CHKERRQ(ierr);
  if (!stack->solution_only) {
    ierr = VecDestroyVecs(stack->numY,&e->Y);CHKERRQ(ierr);
  }
  ierr = PetscFree(e);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "UpdateTS"
static PetscErrorCode UpdateTS(TS ts,Stack *stack,StackElement e)
{
  Vec            *Y;
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCopy(e->X,ts->vec_sol);CHKERRQ(ierr);
  if (!stack->solution_only) {
    ierr = TSGetStages(ts,&stack->numY,&Y);CHKERRQ(ierr);
    for (i=0;i<stack->numY;i++) {
      ierr = VecCopy(e->Y[i],Y[i]);CHKERRQ(ierr);
    }
  }
  ierr = TSSetTimeStep(ts,e->timeprev-e->time);CHKERRQ(ierr); /* stepsize will be negative */
  ts->ptime      = e->time;
  ts->ptime_prev = e->timeprev;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ReCompute"
static PetscErrorCode ReCompute(TS ts,TJScheduler *tjsch,PetscInt stepnumbegin,PetscInt stepnumend)
{
  Stack          *stack = &tjsch->stack;
  PetscInt       i,adjsteps;
  PetscReal      stepsize;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  adjsteps = ts->steps;
  /* reset ts context */
  ierr = TSGetTimeStep(ts,&stepsize);CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts,-stepsize);CHKERRQ(ierr);
  ts->steps = stepnumbegin; /* global step number */
  for (i=ts->steps;i<stepnumend;i++) { /* assume fixed step size */
    if (stack->solution_only && !tjsch->skip_trajectory) { /* revolve online need this */
      ierr = TSTrajectorySet(ts->trajectory,ts,ts->steps,ts->ptime,ts->vec_sol);CHKERRQ(ierr);
    }
    ierr = TSMonitor(ts,ts->steps,ts->ptime,ts->vec_sol);CHKERRQ(ierr);
    ierr = TSStep(ts);CHKERRQ(ierr);
    if (!stack->solution_only && !tjsch->skip_trajectory) {
      ierr = TSTrajectorySet(ts->trajectory,ts,ts->steps,ts->ptime,ts->vec_sol);CHKERRQ(ierr);
    }
    if (ts->event) {
      ierr = TSEventMonitor(ts);CHKERRQ(ierr);
    }
    if (!ts->steprollback) {
      ierr = TSPostStep(ts);CHKERRQ(ierr);
    }
  }
  ierr = TSGetTimeStep(ts,&stepsize);CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts,-stepsize);CHKERRQ(ierr);
  ts->steps = adjsteps;
  ts->total_steps = stepnumend;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SetTrajN"
static PetscErrorCode SetTrajN(TS ts,TJScheduler *tjsch,PetscInt stepnum,PetscReal time,Vec X)
{
  Stack          *stack = &tjsch->stack;
  StackElement   e;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* skip the last two steps of each stride or the whole interval */
  if (stack->solution_only && (stepnum >= tjsch->total_steps-1 || tjsch->recompute)) PetscFunctionReturn(0); //?
  /* skip the first and the last steps of each stride or the whole interval */
  if (!stack->solution_only && (stepnum == 0 || stepnum == tjsch->total_steps)) PetscFunctionReturn(0);

  if (stepnum < stack->top) SETERRQ(tjsch->comm,PETSC_ERR_MEMC,"Illegal modification of a non-top stack element");
  ierr = ElementCreate(ts,stack,&e,stepnum,time,X);CHKERRQ(ierr);
  ierr = StackPush(stack,e);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "GetTrajN"
static PetscErrorCode GetTrajN(TS ts,TJScheduler *tjsch,PetscInt stepnum)
{
  Stack          *stack = &tjsch->stack;
  PetscReal      stepsize;
  StackElement   e;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (stepnum == tjsch->total_steps) {
    ierr = TSGetTimeStep(ts,&stepsize);CHKERRQ(ierr);
    ierr = TSSetTimeStep(ts,-stepsize);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  /* restore a checkpoint */
  ierr = StackTop(stack,&e);CHKERRQ(ierr);
  ierr = UpdateTS(ts,stack,e);CHKERRQ(ierr);
  if (stack->solution_only) {/* recompute one step */
    tjsch->recompute = PETSC_TRUE;
    ierr = ReCompute(ts,tjsch,e->stepnum,stepnum);CHKERRQ(ierr);
  }
  ierr = StackPop(stack,&e);CHKERRQ(ierr);
  ierr = ElementDestroy(stack,e);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SetTrajTLNR"
static PetscErrorCode SetTrajTLNR(TS ts,TJScheduler *tjsch,PetscInt stepnum,PetscReal time,Vec X)
{
  Stack          *stack = &tjsch->stack;
  PetscInt       localstepnum,id,laststridesize;
  StackElement   e;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  localstepnum = stepnum%tjsch->stride;
  id           = stepnum/tjsch->stride;
  if (stepnum == tjsch->total_steps) PetscFunctionReturn(0);

  /* (stride size-1) checkpoints are saved in each stride */
  laststridesize = tjsch->total_steps%tjsch->stride;
  if (!laststridesize) laststridesize = tjsch->stride;
  if (stack->solution_only) {
    if (tjsch->save_stack) {
      if (tjsch->recompute) PetscFunctionReturn(0);
      if (localstepnum == tjsch->stride-1 && stepnum < tjsch->total_steps-laststridesize) { /* dump when stack is full */
        ierr = StackDumpAll(ts,stack,id+1);CHKERRQ(ierr);
        PetscFunctionReturn(0);
      }
      if (stepnum == tjsch->total_steps-1) PetscFunctionReturn(0); /* do not checkpoint tjsch->total_steps-1 */
    } else {
      if (localstepnum == tjsch->stride-1) PetscFunctionReturn(0);
      if (!tjsch->recompute && localstepnum == 0 && stepnum < tjsch->total_steps-laststridesize ) {
        ierr = DumpSingle(ts,stack,id+1);CHKERRQ(ierr);
      }
      if (stepnum < tjsch->total_steps-laststridesize && !tjsch->recompute) PetscFunctionReturn(0);
    }
  } else {
    if (stepnum == 0) PetscFunctionReturn(0);
    if (tjsch->save_stack) {
      if (tjsch->recompute) PetscFunctionReturn(0);
      if (localstepnum == 0 && stepnum != 0) { /* no stack at point 0 */
        ierr = StackDumpAll(ts,stack,id);CHKERRQ(ierr);
        PetscFunctionReturn(0);
      }
    } else {
      if (localstepnum == 0) PetscFunctionReturn(0); /* skip last point in each stride */
      if (!tjsch->recompute && localstepnum == 1 && stepnum < tjsch->total_steps-laststridesize ) { /* skip last stride */
        ierr = DumpSingle(ts,stack,id);CHKERRQ(ierr);
      }
      if (stepnum <= tjsch->total_steps-laststridesize && !tjsch->recompute) PetscFunctionReturn(0);
    }
  }
  ierr = ElementCreate(ts,stack,&e,stepnum,time,X);CHKERRQ(ierr);
  ierr = StackPush(stack,e);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "GetTrajTLNR"
static PetscErrorCode GetTrajTLNR(TS ts,TJScheduler *tjsch,PetscInt stepnum)
{
  Stack          *stack = &tjsch->stack;
  PetscInt       id,localstepnum,laststridesize;
  PetscReal      stepsize;
  StackElement   e;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  localstepnum = stepnum%tjsch->stride;
  if (stepnum == tjsch->total_steps) {
    ierr = TSGetTimeStep(ts,&stepsize);CHKERRQ(ierr);
    ierr = TSSetTimeStep(ts,-stepsize);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  laststridesize = tjsch->total_steps%tjsch->stride;
  if (!laststridesize) laststridesize = tjsch->stride;
  if (stack->solution_only) {
    /* fill stack with info */
    if (localstepnum == 0 && tjsch->total_steps-stepnum >= laststridesize) {
      id = stepnum/tjsch->stride;
      if (tjsch->save_stack) {
        ierr = StackLoadAll(ts,stack,id);CHKERRQ(ierr);
        tjsch->recompute = PETSC_TRUE;
        tjsch->skip_trajectory = PETSC_TRUE;
        ierr = ReCompute(ts,tjsch,id*tjsch->stride-1,id*tjsch->stride);CHKERRQ(ierr);
        tjsch->skip_trajectory = PETSC_FALSE;
      } else {
        ierr = LoadSingle(ts,stack,id);CHKERRQ(ierr);
        tjsch->recompute = PETSC_TRUE;
        ierr = ReCompute(ts,tjsch,(id-1)*tjsch->stride,id*tjsch->stride);CHKERRQ(ierr);
      }
      PetscFunctionReturn(0);
    }
    /* restore a checkpoint */
    ierr = StackPop(stack,&e);CHKERRQ(ierr);
    ierr = UpdateTS(ts,stack,e);CHKERRQ(ierr);
    tjsch->recompute = PETSC_TRUE;
    tjsch->skip_trajectory = PETSC_TRUE;
    ierr = ReCompute(ts,tjsch,e->stepnum,stepnum);CHKERRQ(ierr);
    tjsch->skip_trajectory = PETSC_FALSE;
    ierr = ElementDestroy(stack,e);CHKERRQ(ierr);
  } else {
    /* fill stack with info */
    if (localstepnum == 0 && tjsch->total_steps-stepnum >= laststridesize) {
      id = stepnum/tjsch->stride;
      if (tjsch->save_stack) {
        ierr = StackLoadAll(ts,stack,id);CHKERRQ(ierr);
      } else {
        ierr = LoadSingle(ts,stack,id-1);CHKERRQ(ierr);
        ierr = ElementCreate(ts,stack,&e,(id-1)*tjsch->stride+1,ts->ptime,ts->vec_sol);CHKERRQ(ierr);
        ierr = StackPush(stack,e);CHKERRQ(ierr);
        tjsch->recompute = PETSC_TRUE;
        ierr = ReCompute(ts,tjsch,e->stepnum,id*tjsch->stride);CHKERRQ(ierr);
      }
      PetscFunctionReturn(0);
    }
    /* restore a checkpoint */
    ierr = StackPop(stack,&e);CHKERRQ(ierr);
    ierr = UpdateTS(ts,stack,e);CHKERRQ(ierr);
    ierr = ElementDestroy(stack,e);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#ifdef PETSC_HAVE_REVOLVE
static void printwhattodo(PetscInt whattodo,RevolveCTX *rctx,PetscInt shift)
{
  switch(whattodo) {
    case 1:
      PetscPrintf(PETSC_COMM_WORLD,"\x1B[35mAdvance from %D to %D\033[0m\n",rctx->oldcapo+shift,rctx->capo+shift);
      break;
    case 2:
      PetscPrintf(PETSC_COMM_WORLD,"\x1B[35mStore in checkpoint number %D (located in RAM)\033[0m\n",rctx->check);
      break;
    case 3:
      PetscPrintf(PETSC_COMM_WORLD,"\x1B[35mFirst turn: Initialize adjoints and reverse first step\033[0m\n");
      break;
    case 4:
      PetscPrintf(PETSC_COMM_WORLD,"\x1B[35mForward and reverse one step\033[0m\n");
      break;
    case 5:
      PetscPrintf(PETSC_COMM_WORLD,"\x1B[35mRestore in checkpoint number %D (located in RAM)\033[0m\n",rctx->check);
      break;
    case 7:
      PetscPrintf(PETSC_COMM_WORLD,"\x1B[35mStore in checkpoint number %D (located on disk)\033[0m\n",rctx->check);
      break;
    case 8:
      PetscPrintf(PETSC_COMM_WORLD,"\x1B[35mRestore in checkpoint number %D (located on disk)\033[0m\n",rctx->check);
      break;
    case -1:
      PetscPrintf(PETSC_COMM_WORLD,"\x1B[35mError!");
      break;
  }
}

#undef __FUNCT__
#define __FUNCT__ "ApplyRevolve"
static PetscErrorCode ApplyRevolve(TJScheduler *tjsch,PetscInt stepnum,PetscInt localstepnum,PetscInt *store)
{
  PetscInt       shift,whattodo;
  RevolveCTX     *rctx = tjsch->rctx;

  PetscFunctionBegin;
  *store = 0;
  if (rctx->reverseonestep && stepnum == tjsch->total_steps) { /* intermediate information is ready inside TS, this happens at last time step */
    rctx->reverseonestep = PETSC_FALSE;
    PetscFunctionReturn(0);
  }
  if (rctx->stepsleft > 0) { /* advance the solution without checkpointing anything as Revolve requires */
    rctx->stepsleft--;
    PetscFunctionReturn(0);
  }
  /* let Revolve determine what to do next */
  shift         = stepnum-localstepnum;
  rctx->oldcapo = rctx->capo;
  rctx->capo    = localstepnum;
  whattodo = revolve_action(&rctx->check,&rctx->capo,&rctx->fine,rctx->snaps_in,&rctx->info,&rctx->where);
  if (tjsch->stype == REVOLVE_ONLINE && whattodo == 8) whattodo = 5;
  if (tjsch->stype == REVOLVE_ONLINE && whattodo == 7) whattodo = 2;
  printwhattodo(whattodo,rctx,shift);
  if (whattodo == -1) SETERRQ(tjsch->comm,PETSC_ERR_LIB,"Error in the Revolve library");
  if (whattodo == 1) { /* advance some time steps */
    if (tjsch->stype == REVOLVE_ONLINE && rctx->capo >= tjsch->total_steps-1) {
      revolve_turn(tjsch->total_steps,&rctx->capo,&rctx->fine);
      printwhattodo(whattodo,rctx,shift);
    }
    rctx->stepsleft = rctx->capo-rctx->oldcapo-1;
  }
  if (whattodo == 3 || whattodo == 4) { /* ready for a reverse step */
    rctx->reverseonestep = PETSC_TRUE;
  }
  if (whattodo == 5) { /* restore a checkpoint and ask Revolve what to do next */
    rctx->oldcapo = rctx->capo;
    whattodo = revolve_action(&rctx->check,&rctx->capo,&rctx->fine,rctx->snaps_in,&rctx->info,&rctx->where); /* must return 1 or 3 or 4*/
    printwhattodo(whattodo,rctx,shift);
    if (whattodo == 3 || whattodo == 4) rctx->reverseonestep = PETSC_TRUE;
    if (whattodo == 1) rctx->stepsleft = rctx->capo-rctx->oldcapo;
  }
  if (whattodo == 7) { /* save the checkpoint to disk */
    *store = 2;
    rctx->oldcapo = rctx->capo;
    whattodo = revolve_action(&rctx->check,&rctx->capo,&rctx->fine,rctx->snaps_in,&rctx->info,&rctx->where); /* must return 1*/
    printwhattodo(whattodo,rctx,shift);
    rctx->stepsleft = rctx->capo-rctx->oldcapo-1;
  }
  if (whattodo == 2) { /* store a checkpoint to RAM and ask Revolve how many time steps to advance next */
    *store = 1;
    rctx->oldcapo = rctx->capo;
    whattodo = revolve_action(&rctx->check,&rctx->capo,&rctx->fine,rctx->snaps_in,&rctx->info,&rctx->where); /* must return 1*/
    printwhattodo(whattodo,rctx,shift);
    if (tjsch->stype == REVOLVE_ONLINE && rctx->capo >= tjsch->total_steps-1) {
      revolve_turn(tjsch->total_steps,&rctx->capo,&rctx->fine);
      printwhattodo(whattodo,rctx,shift);
    }
    rctx->stepsleft = rctx->capo-rctx->oldcapo-1;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SetTrajROF"
static PetscErrorCode SetTrajROF(TS ts,TJScheduler *tjsch,PetscInt stepnum,PetscReal time,Vec X)
{
  Stack          *stack = &tjsch->stack;
  PetscInt       store;
  StackElement   e;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!stack->solution_only && stepnum == 0) PetscFunctionReturn(0);
  ierr = ApplyRevolve(tjsch,stepnum,stepnum,&store);CHKERRQ(ierr);
  if (store == 1) {
    if (stepnum < stack->top) SETERRQ(tjsch->comm,PETSC_ERR_MEMC,"Illegal modification of a non-top stack element");
    ierr = ElementCreate(ts,stack,&e,stepnum,time,X);CHKERRQ(ierr);
    ierr = StackPush(stack,e);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "GetTrajROF"
static PetscErrorCode GetTrajROF(TS ts,TJScheduler *tjsch,PetscInt stepnum)
{
  Stack          *stack = &tjsch->stack;
  PetscInt       whattodo,shift,store;
  PetscReal      stepsize;
  StackElement   e;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (stepnum == 0 || stepnum == tjsch->total_steps) {
    ierr = TSGetTimeStep(ts,&stepsize);CHKERRQ(ierr);
    ierr = TSSetTimeStep(ts,-stepsize);CHKERRQ(ierr);
    if (tjsch->rctx->reverseonestep) tjsch->rctx->reverseonestep = PETSC_FALSE;
    PetscFunctionReturn(0);
  }
  /* restore a checkpoint */
  ierr = StackTop(stack,&e);CHKERRQ(ierr);
  ierr = UpdateTS(ts,stack,e);CHKERRQ(ierr);
  if (stack->solution_only) { /* start with restoring a checkpoint */
    tjsch->rctx->capo = stepnum;
    tjsch->rctx->oldcapo = tjsch->rctx->capo;
    shift = 0;
    whattodo = revolve_action(&tjsch->rctx->check,&tjsch->rctx->capo,&tjsch->rctx->fine,tjsch->rctx->snaps_in,&tjsch->rctx->info,&tjsch->rctx->where);
    printwhattodo(whattodo,tjsch->rctx,shift);
  } else { /* 2 revolve actions: restore a checkpoint and then advance */
    ierr = ApplyRevolve(tjsch,stepnum,stepnum,&store);CHKERRQ(ierr);
    if (!tjsch->rctx->reverseonestep && tjsch->rctx->stepsleft > 0) {
      tjsch->rctx->stepsleft--;
      PetscPrintf(PETSC_COMM_WORLD,"\x1B[35mSkip the step from %D to %D (already checkpointed)\033[0m\n",tjsch->rctx->oldcapo,tjsch->rctx->oldcapo+1);
    }
  }
  if (stack->solution_only || (!stack->solution_only && e->stepnum < stepnum)) {
    tjsch->recompute = PETSC_TRUE;
    ierr = ReCompute(ts,tjsch,e->stepnum,stepnum);CHKERRQ(ierr);
  }
  if ((stack->solution_only && e->stepnum+1 == stepnum) || (!stack->solution_only && e->stepnum == stepnum)) {
    ierr = StackPop(stack,&e);CHKERRQ(ierr);
    ierr = ElementDestroy(stack,e);CHKERRQ(ierr);
  }
  if (tjsch->rctx->reverseonestep) tjsch->rctx->reverseonestep = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SetTrajRON"
static PetscErrorCode SetTrajRON(TS ts,TJScheduler *tjsch,PetscInt stepnum,PetscReal time,Vec X)
{
  Stack          *stack = &tjsch->stack;
  Vec            *Y;
  PetscInt       i,store;
  PetscReal      timeprev;
  StackElement   e;
  RevolveCTX     *rctx = tjsch->rctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!stack->solution_only && stepnum == 0) PetscFunctionReturn(0);
  ierr = ApplyRevolve(tjsch,stepnum,stepnum,&store);CHKERRQ(ierr);
  if (store == 1) {
    if (rctx->check != stack->top+1) { /* overwrite some non-top checkpoint in the stack */
      ierr = StackFind(stack,&e,rctx->check);CHKERRQ(ierr);
      ierr = VecCopy(X,e->X);CHKERRQ(ierr);
      if (stack->numY > 0 && !stack->solution_only) {
        ierr = TSGetStages(ts,&stack->numY,&Y);CHKERRQ(ierr);
        for (i=0;i<stack->numY;i++) {
          ierr = VecCopy(Y[i],e->Y[i]);CHKERRQ(ierr);
        }
      }
      e->stepnum  = stepnum;
      e->time     = time;
      ierr        = TSGetPrevTime(ts,&timeprev);CHKERRQ(ierr);
      e->timeprev = timeprev;
    } else {
      if (stepnum < stack->top) SETERRQ(tjsch->comm,PETSC_ERR_MEMC,"Illegal modification of a non-top stack element");
      ierr = ElementCreate(ts,stack,&e,stepnum,time,X);CHKERRQ(ierr);
      ierr = StackPush(stack,e);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "GetTrajRON"
static PetscErrorCode GetTrajRON(TS ts,TJScheduler *tjsch,PetscInt stepnum)
{
  Stack          *stack = &tjsch->stack;
  PetscInt       whattodo,shift;
  PetscReal      stepsize;
  StackElement   e;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (stepnum == 0 || stepnum == tjsch->total_steps) {
    ierr = TSGetTimeStep(ts,&stepsize);CHKERRQ(ierr);
    ierr = TSSetTimeStep(ts,-stepsize);CHKERRQ(ierr);
    if (tjsch->rctx->reverseonestep) tjsch->rctx->reverseonestep = PETSC_FALSE;
    PetscFunctionReturn(0);
  }
  tjsch->rctx->capo = stepnum;
  tjsch->rctx->oldcapo = tjsch->rctx->capo;
  shift = 0;
  whattodo = revolve_action(&tjsch->rctx->check,&tjsch->rctx->capo,&tjsch->rctx->fine,tjsch->rctx->snaps_in,&tjsch->rctx->info,&tjsch->rctx->where); /* whattodo=restore */
  if (whattodo == 8) whattodo = 5;
  printwhattodo(whattodo,tjsch->rctx,shift);
  /* restore a checkpoint */
  ierr = StackFind(stack,&e,tjsch->rctx->check);CHKERRQ(ierr);
  ierr = UpdateTS(ts,stack,e);CHKERRQ(ierr);
  if (!stack->solution_only) { /* whattodo must be 5 */
    /* ask Revolve what to do next */
    tjsch->rctx->oldcapo = tjsch->rctx->capo;
    whattodo = revolve_action(&tjsch->rctx->check,&tjsch->rctx->capo,&tjsch->rctx->fine,tjsch->rctx->snaps_in,&tjsch->rctx->info,&tjsch->rctx->where); /* must return 1 or 3 or 4*/
    printwhattodo(whattodo,tjsch->rctx,shift);
    if (whattodo == 3 || whattodo == 4) tjsch->rctx->reverseonestep = PETSC_TRUE;
    if (whattodo == 1) tjsch->rctx->stepsleft = tjsch->rctx->capo-tjsch->rctx->oldcapo;
    if (!tjsch->rctx->reverseonestep && tjsch->rctx->stepsleft > 0) {
      tjsch->rctx->stepsleft--;
      PetscPrintf(PETSC_COMM_WORLD,"\x1B[35mSkip the step from %D to %D (already checkpointed)\033[0m\n",tjsch->rctx->oldcapo,tjsch->rctx->oldcapo+1);
    }
  }
  if (stack->solution_only || (!stack->solution_only && e->stepnum < stepnum)) {
    tjsch->recompute = PETSC_TRUE;
    ierr = ReCompute(ts,tjsch,e->stepnum,stepnum);CHKERRQ(ierr);
  }
  if (tjsch->rctx->reverseonestep) tjsch->rctx->reverseonestep = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SetTrajTLR"
static PetscErrorCode SetTrajTLR(TS ts,TJScheduler *tjsch,PetscInt stepnum,PetscReal time,Vec X)
{
  Stack          *stack = &tjsch->stack;
  PetscInt       store,localstepnum,id,laststridesize;
  StackElement   e;
  RevolveCTX     *rctx = tjsch->rctx;
  PetscBool      resetrevolve = PETSC_FALSE;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  localstepnum = stepnum%tjsch->stride;
  id           = stepnum/tjsch->stride; /* stride index */

  /* (stride size-1) checkpoints are saved in each stride */
  laststridesize = tjsch->total_steps%tjsch->stride;
  if (!laststridesize) laststridesize = tjsch->stride;
  if (stack->solution_only) {
    if (stepnum == tjsch->total_steps) PetscFunctionReturn(0);
    if (tjsch->save_stack) {
      if (!tjsch->recompute && localstepnum == tjsch->stride-1 && stepnum < tjsch->total_steps-laststridesize) {
        PetscPrintf(PETSC_COMM_WORLD,"\x1B[33mDump stack to file\033[0m\n");
        resetrevolve = PETSC_TRUE;
        ierr = StackDumpAll(ts,stack,id+1);CHKERRQ(ierr);
      }
    } else {
      if (!tjsch->recompute && localstepnum == 0 && stepnum < tjsch->total_steps-laststridesize ) {
        PetscPrintf(PETSC_COMM_WORLD,"\x1B[33mDump a single point to file\033[0m\n");
        ierr = DumpSingle(ts,stack,id+1);CHKERRQ(ierr);
      }
      if (stepnum < tjsch->total_steps-laststridesize && !tjsch->recompute) PetscFunctionReturn(0); /* no need to checkpoint except last stride in the first sweep */
    }
  } else {
    if (stepnum == 0) PetscFunctionReturn(0);
    if (tjsch->save_stack) {
      if (!tjsch->recompute && localstepnum == 0 && stepnum != tjsch->total_steps) { /* do not dump stack for last stride */
        PetscPrintf(PETSC_COMM_WORLD,"\x1B[33mDump stack to file\033[0m\n");
        resetrevolve = PETSC_TRUE;
        ierr = StackDumpAll(ts,stack,id);CHKERRQ(ierr);
      }
    } else {
      if (!tjsch->recompute && localstepnum == 1 && stepnum <  tjsch->total_steps-laststridesize ) { /* skip last stride */
        PetscPrintf(PETSC_COMM_WORLD,"\x1B[33mDump a single point to file\033[0m\n");
        ierr = DumpSingle(ts,stack,id);CHKERRQ(ierr);
      }
      if (stepnum <= tjsch->total_steps-laststridesize && !tjsch->recompute) PetscFunctionReturn(0);
    }
  }

  ierr = ApplyRevolve(tjsch,stepnum,localstepnum,&store);CHKERRQ(ierr);
  if (store == 1) {
    if (localstepnum < stack->top) SETERRQ(tjsch->comm,PETSC_ERR_MEMC,"Illegal modification of a non-top stack element");
    ierr = ElementCreate(ts,stack,&e,stepnum,time,X);CHKERRQ(ierr);
    ierr = StackPush(stack,e);CHKERRQ(ierr);
  }
  if (resetrevolve) {
    revolve_reset();
    revolve_create_offline(tjsch->stride,tjsch->max_cps_ram);
    rctx = tjsch->rctx;
    rctx->check = 0;
    rctx->capo  = 0;
    rctx->fine  = tjsch->stride;
    if (tjsch->rctx->reverseonestep) tjsch->rctx->reverseonestep = PETSC_FALSE;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "GetTrajTLR"
static PetscErrorCode GetTrajTLR(TS ts,TJScheduler *tjsch,PetscInt stepnum)
{
  Stack          *stack = &tjsch->stack;
  PetscInt       whattodo,shift;
  PetscInt       localstepnum,id,laststridesize,store;
  PetscReal      stepsize;
  StackElement   e;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  localstepnum = stepnum%tjsch->stride;
  id           = stepnum/tjsch->stride;
  if (stepnum == tjsch->total_steps) {
    ierr = TSGetTimeStep(ts,&stepsize);CHKERRQ(ierr);
    ierr = TSSetTimeStep(ts,-stepsize);CHKERRQ(ierr);
    if ( tjsch->rctx->reverseonestep) tjsch->rctx->reverseonestep = PETSC_FALSE;
    PetscFunctionReturn(0);
  }
  laststridesize = tjsch->total_steps%tjsch->stride;
  if (!laststridesize) laststridesize = tjsch->stride;
  if (stack->solution_only) {
    /* fill stack */
    if (localstepnum == 0 && stepnum <= tjsch->total_steps-laststridesize) {
      if (tjsch->save_stack) {
        PetscPrintf(PETSC_COMM_WORLD,"\x1B[33mLoad stack from file\033[0m\n");
        ierr = StackLoadAll(ts,stack,id);CHKERRQ(ierr);
        revolve_reset();
        revolve_create_offline(tjsch->stride,tjsch->max_cps_ram);
        tjsch->rctx->check = 0;
        tjsch->rctx->capo  = 0;
        tjsch->rctx->fine  = tjsch->stride;
        whattodo = 0;
        while(whattodo!=3) { /* stupid revolve */
          whattodo = revolve_action(&tjsch->rctx->check,&tjsch->rctx->capo,&tjsch->rctx->fine,tjsch->rctx->snaps_in,&tjsch->rctx->info,&tjsch->rctx->where);
        }
        tjsch->recompute = PETSC_TRUE;
        tjsch->skip_trajectory = PETSC_TRUE;
        ierr = ReCompute(ts,tjsch,id*tjsch->stride-1,id*tjsch->stride);CHKERRQ(ierr);
        tjsch->skip_trajectory = PETSC_FALSE;
      } else {
        PetscPrintf(PETSC_COMM_WORLD,"\x1B[33mLoad a single point from file\033[0m\n");
        ierr = LoadSingle(ts,stack,id);CHKERRQ(ierr);
        revolve_reset();
        revolve_create_offline(tjsch->stride,tjsch->max_cps_ram);
        tjsch->rctx->check = 0;
        tjsch->rctx->capo  = 0;
        tjsch->rctx->fine  = tjsch->stride;
        tjsch->recompute = PETSC_TRUE;
        ierr = ReCompute(ts,tjsch,(id-1)*tjsch->stride,id*tjsch->stride);CHKERRQ(ierr);
      }
      PetscFunctionReturn(0);
    }
    /* restore a checkpoint */
    ierr = StackTop(stack,&e);CHKERRQ(ierr);
    ierr = UpdateTS(ts,stack,e);CHKERRQ(ierr);
    /* start with restoring a checkpoint */
    tjsch->rctx->capo = stepnum;
    tjsch->rctx->oldcapo = tjsch->rctx->capo;
    shift = stepnum-localstepnum;
    whattodo = revolve_action(&tjsch->rctx->check,&tjsch->rctx->capo,&tjsch->rctx->fine,tjsch->rctx->snaps_in,&tjsch->rctx->info,&tjsch->rctx->where);
    printwhattodo(whattodo,tjsch->rctx,shift);
    tjsch->recompute = PETSC_TRUE;
    ierr = ReCompute(ts,tjsch,e->stepnum,stepnum);CHKERRQ(ierr);
    if (e->stepnum+1 == stepnum) {
      ierr = StackPop(stack,&e);CHKERRQ(ierr);
      ierr = ElementDestroy(stack,e);CHKERRQ(ierr);
    }
  } else {
    /* fill stack with info */
    if (localstepnum == 0 && tjsch->total_steps-stepnum >= laststridesize) {
      if (tjsch->save_stack) {
        PetscPrintf(PETSC_COMM_WORLD,"\x1B[33mLoad stack from file\033[0m\n");
        ierr = StackLoadAll(ts,stack,id);CHKERRQ(ierr);
        revolve_reset();
        revolve_create_offline(tjsch->stride,tjsch->max_cps_ram);
        tjsch->rctx->check = 0;
        tjsch->rctx->capo  = 0;
        tjsch->rctx->fine  = tjsch->stride;
        whattodo = 0;
        while(whattodo!=3) { /* stupid revolve */
          whattodo = revolve_action(&tjsch->rctx->check,&tjsch->rctx->capo,&tjsch->rctx->fine,tjsch->rctx->snaps_in,&tjsch->rctx->info,&tjsch->rctx->where);
        }
      } else {
        PetscPrintf(PETSC_COMM_WORLD,"\x1B[33mLoad a single point from file\033[0m\n");
        ierr = LoadSingle(ts,stack,id-1);CHKERRQ(ierr);
        revolve_reset();
        revolve_create_offline(tjsch->stride,tjsch->max_cps_ram);
        tjsch->rctx->check = 0;
        tjsch->rctx->capo  = 0;
        tjsch->rctx->fine  = tjsch->stride;
        shift = stepnum-localstepnum;
        whattodo = revolve_action(&tjsch->rctx->check,&tjsch->rctx->capo,&tjsch->rctx->fine,tjsch->rctx->snaps_in,&tjsch->rctx->info,&tjsch->rctx->where);
        printwhattodo(whattodo,tjsch->rctx,shift);
        ierr = ElementCreate(ts,stack,&e,(id-1)*tjsch->stride+1,ts->ptime,ts->vec_sol);CHKERRQ(ierr);
        ierr = StackPush(stack,e);CHKERRQ(ierr);
        tjsch->recompute = PETSC_TRUE;
        ierr = ReCompute(ts,tjsch,e->stepnum,id*tjsch->stride);CHKERRQ(ierr);
        if ( tjsch->rctx->reverseonestep) { /* ready for the reverse step */
          ierr = TSGetTimeStep(ts,&stepsize);CHKERRQ(ierr);
          ierr = TSSetTimeStep(ts,-stepsize);CHKERRQ(ierr);
          tjsch->rctx->reverseonestep = PETSC_FALSE;
        }
      }
      PetscFunctionReturn(0);
    }
    /* restore a checkpoint */
    ierr = StackTop(stack,&e);CHKERRQ(ierr);
    ierr = UpdateTS(ts,stack,e);CHKERRQ(ierr);
    /* 2 revolve actions: restore a checkpoint and then advance */
    ierr = ApplyRevolve(tjsch,stepnum,localstepnum,&store);CHKERRQ(ierr);
    if (!tjsch->rctx->reverseonestep && tjsch->rctx->stepsleft > 0) {
      tjsch->rctx->stepsleft--;
      PetscPrintf(PETSC_COMM_WORLD,"\x1B[35mSkip the step from %D to %D (already checkpointed)\033[0m\n",stepnum-localstepnum+tjsch->rctx->oldcapo,stepnum-localstepnum+tjsch->rctx->oldcapo+1);
    }
    if (e->stepnum < stepnum) {
      tjsch->recompute = PETSC_TRUE;
      ierr = ReCompute(ts,tjsch,e->stepnum,stepnum);CHKERRQ(ierr);
    }
    if (e->stepnum == stepnum) {
      ierr = StackPop(stack,&e);CHKERRQ(ierr);
      ierr = ElementDestroy(stack,e);CHKERRQ(ierr);
    }
  }
  if (tjsch->rctx->reverseonestep) tjsch->rctx->reverseonestep = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SetTrajRMS"
static PetscErrorCode SetTrajRMS(TS ts,TJScheduler *tjsch,PetscInt stepnum,PetscReal time,Vec X)
{
  Stack          *stack = &tjsch->stack;
  PetscInt       store;
  StackElement   e;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!stack->solution_only && stepnum == 0) PetscFunctionReturn(0);
  ierr = ApplyRevolve(tjsch,stepnum,stepnum,&store);CHKERRQ(ierr);
  if (store == 1){
    if (stepnum < stack->top) SETERRQ(tjsch->comm,PETSC_ERR_MEMC,"Illegal modification of a non-top stack element");
    ierr = ElementCreate(ts,stack,&e,stepnum,time,X);CHKERRQ(ierr);
    ierr = StackPush(stack,e);CHKERRQ(ierr);
  } else if (store == 2) {
    ierr = DumpSingle(ts,stack,tjsch->rctx->check);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "GetTrajRMS"
static PetscErrorCode GetTrajRMS(TS ts,TJScheduler *tjsch,PetscInt stepnum)
{
  Stack          *stack = &tjsch->stack;
  PetscInt       whattodo,shift;
  PetscInt       restart;
  PetscBool      ondisk;
  PetscReal      stepsize;
  StackElement   e;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (stepnum == 0 || stepnum == tjsch->total_steps) {
    ierr = TSGetTimeStep(ts,&stepsize);CHKERRQ(ierr);
    ierr = TSSetTimeStep(ts,-stepsize);CHKERRQ(ierr);
    if (tjsch->rctx->reverseonestep) tjsch->rctx->reverseonestep = PETSC_FALSE;
    PetscFunctionReturn(0);
  }
  tjsch->rctx->capo = stepnum;
  tjsch->rctx->oldcapo = tjsch->rctx->capo;
  shift = 0;
  whattodo = revolve_action(&tjsch->rctx->check,&tjsch->rctx->capo,&tjsch->rctx->fine,tjsch->rctx->snaps_in,&tjsch->rctx->info,&tjsch->rctx->where); /* whattodo=restore */
  printwhattodo(whattodo,tjsch->rctx,shift);
  /* restore a checkpoint */
  restart = tjsch->rctx->capo;
  if (!tjsch->rctx->where) {
    ondisk = PETSC_TRUE;
    ierr = LoadSingle(ts,stack,tjsch->rctx->check);CHKERRQ(ierr);
    ierr = TSSetTimeStep(ts,ts->ptime_prev-ts->ptime);CHKERRQ(ierr);
  } else {
    ondisk = PETSC_FALSE;
    ierr = StackTop(stack,&e);CHKERRQ(ierr);
    ierr = UpdateTS(ts,stack,e);CHKERRQ(ierr);
  }
  if (!stack->solution_only) { /* whattodo must be 5 or 8 */
    /* ask Revolve what to do next */
    tjsch->rctx->oldcapo = tjsch->rctx->capo;
    whattodo = revolve_action(&tjsch->rctx->check,&tjsch->rctx->capo,&tjsch->rctx->fine,tjsch->rctx->snaps_in,&tjsch->rctx->info,&tjsch->rctx->where); /* must return 1 or 3 or 4*/
    printwhattodo(whattodo,tjsch->rctx,shift);
    if (whattodo == 3 || whattodo == 4) tjsch->rctx->reverseonestep = PETSC_TRUE;
    if (whattodo == 1) tjsch->rctx->stepsleft = tjsch->rctx->capo-tjsch->rctx->oldcapo;
    if (!tjsch->rctx->reverseonestep && tjsch->rctx->stepsleft > 0) {
      tjsch->rctx->stepsleft--;
      PetscPrintf(PETSC_COMM_WORLD,"\x1B[35mSkip the step from %D to %D (already checkpointed)\033[0m\n",tjsch->rctx->oldcapo,tjsch->rctx->oldcapo+1);
    }
    restart++; /* skip one step */
  }
  if (stack->solution_only || (!stack->solution_only && restart < stepnum)) {
    tjsch->recompute = PETSC_TRUE;
    ierr = ReCompute(ts,tjsch,restart,stepnum);CHKERRQ(ierr);
  }
  if (!ondisk && ( (stack->solution_only && e->stepnum+1 == stepnum) || (!stack->solution_only && e->stepnum == stepnum) )) {
    ierr = StackPop(stack,&e);CHKERRQ(ierr);
    ierr = ElementDestroy(stack,e);CHKERRQ(ierr);
  }
  if (tjsch->rctx->reverseonestep) tjsch->rctx->reverseonestep = PETSC_FALSE;
  PetscFunctionReturn(0);
}
#endif

#undef __FUNCT__
#define __FUNCT__ "TSTrajectorySet_Memory"
PetscErrorCode TSTrajectorySet_Memory(TSTrajectory tj,TS ts,PetscInt stepnum,PetscReal time,Vec X)
{
  TJScheduler *tjsch = (TJScheduler*)tj->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!tjsch->recompute) { /* use global stepnum in the forward sweep */
    ierr = TSGetTotalSteps(ts,&stepnum);CHKERRQ(ierr);
  }
  /* for consistency */
  if (!tjsch->recompute && stepnum == 0) ts->ptime_prev = ts->ptime-ts->time_step;
  switch (tjsch->stype) {
    case NONE:
      ierr = SetTrajN(ts,tjsch,stepnum,time,X);CHKERRQ(ierr);
      break;
    case TWO_LEVEL_NOREVOLVE:
      ierr = SetTrajTLNR(ts,tjsch,stepnum,time,X);CHKERRQ(ierr);
      break;
#ifdef PETSC_HAVE_REVOLVE
    case TWO_LEVEL_REVOLVE:
      ierr = SetTrajTLR(ts,tjsch,stepnum,time,X);CHKERRQ(ierr);
      break;
    case REVOLVE_OFFLINE:
      ierr = SetTrajROF(ts,tjsch,stepnum,time,X);CHKERRQ(ierr);
      break;
    case REVOLVE_ONLINE:
      ierr = SetTrajRON(ts,tjsch,stepnum,time,X);CHKERRQ(ierr);
      break;
    case REVOLVE_MULTISTAGE:
      ierr = SetTrajRMS(ts,tjsch,stepnum,time,X);CHKERRQ(ierr);
      break;
#endif
    default:
      break;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSTrajectoryGet_Memory"
PetscErrorCode TSTrajectoryGet_Memory(TSTrajectory tj,TS ts,PetscInt stepnum,PetscReal *t)
{
  TJScheduler *tjsch = (TJScheduler*)tj->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSGetTotalSteps(ts,&stepnum);CHKERRQ(ierr);
  if (stepnum == 0) PetscFunctionReturn(0);
  switch (tjsch->stype) {
    case NONE:
      ierr = GetTrajN(ts,tjsch,stepnum);CHKERRQ(ierr);
      break;
    case TWO_LEVEL_NOREVOLVE:
      ierr = GetTrajTLNR(ts,tjsch,stepnum);CHKERRQ(ierr);
      break;
#ifdef PETSC_HAVE_REVOLVE
    case TWO_LEVEL_REVOLVE:
      ierr = GetTrajTLR(ts,tjsch,stepnum);CHKERRQ(ierr);
      break;
    case REVOLVE_OFFLINE:
      ierr = GetTrajROF(ts,tjsch,stepnum);CHKERRQ(ierr);
      break;
    case REVOLVE_ONLINE:
      ierr = GetTrajRON(ts,tjsch,stepnum);CHKERRQ(ierr);
      break;
    case REVOLVE_MULTISTAGE:
      ierr = GetTrajRMS(ts,tjsch,stepnum);CHKERRQ(ierr);
      break;
#endif
    default:
      break;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSTrajectorySetStride_Memory"
PetscErrorCode TSTrajectorySetStride_Memory(TSTrajectory tj,TS ts,PetscInt stride)
{
  TJScheduler *tjsch = (TJScheduler*)tj->data;

  PetscFunctionBegin;
  tjsch->stride = stride;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSTrajectorySetMaxCpsRAM_Memory"
PetscErrorCode TSTrajectorySetMaxCpsRAM_Memory(TSTrajectory tj,TS ts,PetscInt max_cps_ram)
{
  TJScheduler *tjsch = (TJScheduler*)tj->data;

  PetscFunctionBegin;
  tjsch->max_cps_ram = max_cps_ram;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSTrajectorySetMaxCpsDisk_Memory"
PetscErrorCode TSTrajectorySetMaxCpsDisk_Memory(TSTrajectory tj,TS ts,PetscInt max_cps_disk)
{
  TJScheduler *tjsch = (TJScheduler*)tj->data;

  PetscFunctionBegin;
  tjsch->max_cps_disk = max_cps_disk;
  PetscFunctionReturn(0);
}

#ifdef PETSC_HAVE_REVOLVE
#undef __FUNCT__
#define __FUNCT__ "TSTrajectorySetRevolveOnline"
PetscErrorCode TSTrajectorySetRevolveOnline(TSTrajectory tj,PetscBool use_online)
{
  TJScheduler *tjsch = (TJScheduler*)tj->data;

  PetscFunctionBegin;
  tjsch->use_online = use_online;
  PetscFunctionReturn(0);
}
#endif

#undef __FUNCT__
#define __FUNCT__ "TSTrajectorySetSaveStack"
PetscErrorCode TSTrajectorySetSaveStack(TSTrajectory tj,PetscBool save_stack)
{
  TJScheduler *tjsch = (TJScheduler*)tj->data;

  PetscFunctionBegin;
  tjsch->save_stack = save_stack;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSTrajectorySetSolutionOnly"
PetscErrorCode TSTrajectorySetSolutionOnly(TSTrajectory tj,PetscBool solution_only)
{
  TJScheduler *tjsch = (TJScheduler*)tj->data;
  Stack       *stack = &tjsch->stack;

  PetscFunctionBegin;
  stack->solution_only = solution_only;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSTrajectorySetFromOptions_Memory"
PetscErrorCode TSTrajectorySetFromOptions_Memory(PetscOptions *PetscOptionsObject,TSTrajectory tj)
{
  TJScheduler    *tjsch = (TJScheduler*)tj->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"Memory based TS trajectory options");CHKERRQ(ierr);
  {
    ierr = PetscOptionsInt("-tstrajectory_max_cps_ram","Maximum number of checkpoints in RAM","TSTrajectorySetMaxCpsRAM_Memory",tjsch->max_cps_ram,&tjsch->max_cps_ram,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-tstrajectory_max_cps_disk","Maximum number of checkpoints on disk","TSTrajectorySetMaxCpsDisk_Memory",tjsch->max_cps_disk,&tjsch->max_cps_disk,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-tstrajectory_stride","Stride to save checkpoints to file","TSTrajectorySetStride_Memory",tjsch->stride,&tjsch->stride,NULL);CHKERRQ(ierr);
#ifdef PETSC_HAVE_REVOLVE
    ierr = PetscOptionsBool("-tstrajectory_revolve_online","Trick TS trajectory into using online mode of revolve","TSTrajectorySetRevolveOnline",tjsch->use_online,&tjsch->use_online,NULL);CHKERRQ(ierr);
#endif
    ierr = PetscOptionsBool("-tstrajectory_save_stack","Save all stack to disk","TSTrajectorySetSaveStack",tjsch->save_stack,&tjsch->save_stack,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-tstrajectory_solution_only","Checkpoint solution only","TSTrajectorySetSolutionOnly",tjsch->stack.solution_only,&tjsch->stack.solution_only,NULL);CHKERRQ(ierr);
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSTrajectorySetUp_Memory"
PetscErrorCode TSTrajectorySetUp_Memory(TSTrajectory tj,TS ts)
{
  TJScheduler    *tjsch = (TJScheduler*)tj->data;
  Stack          *stack = &tjsch->stack;
#ifdef PETSC_HAVE_REVOLVE
  RevolveCTX     *rctx;
#endif
  PetscInt       numY;
  PetscBool      flg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscStrcmp(((PetscObject)ts->adapt)->type_name,TSADAPTNONE,&flg);
  if (flg) { /* fixed time step */
    tjsch->total_steps = PetscMin(ts->max_steps,(PetscInt)(ceil(ts->max_time/ts->time_step)));
  }
  if (tjsch->max_cps_ram > 1) stack->stacksize = tjsch->max_cps_ram;
  if (tjsch->stride > 1) { /* two level mode works for both fixed time step and adaptive time step */
    if (tjsch->max_cps_ram > 1 && tjsch->max_cps_ram < tjsch->stride-1) { /* use revolve_offline for each stride */
      tjsch->stype = TWO_LEVEL_REVOLVE;
    }else { /* checkpoint all for each stride */
      tjsch->stype = TWO_LEVEL_NOREVOLVE;
      stack->stacksize = tjsch->stride-1;
    }
  } else {
    if (flg) { /* fixed time step */
      if (tjsch->max_cps_ram >= tjsch->total_steps-1 || tjsch->max_cps_ram < 1) { /* checkpoint all */
        tjsch->stype = NONE;
        stack->stacksize = stack->solution_only ? tjsch->total_steps : tjsch->total_steps-1;
      } else {
        if (tjsch->max_cps_disk > 1) { /* disk can be used */
          tjsch->stype = REVOLVE_MULTISTAGE;
        } else { /* memory only */
          tjsch->stype = REVOLVE_OFFLINE;
        }
      }
    } else { /* adaptive time step */
      tjsch->stype = REVOLVE_ONLINE;
    }
#ifdef PETSC_HAVE_REVOLVE
    if (tjsch->use_online) { /* trick into online */
      tjsch->stype = REVOLVE_ONLINE;
      stack->stacksize = tjsch->max_cps_ram;
    }
#endif
  }

  if (tjsch->stype > TWO_LEVEL_NOREVOLVE) {
#ifndef PETSC_HAVE_REVOLVE
    SETERRQ(tjsch->comm,PETSC_ERR_SUP,"revolve is needed when there is not enough memory to checkpoint all time steps according to the user's settings, please reconfigure with the additional option --download-revolve.");
#else
    if (tjsch->stype == TWO_LEVEL_REVOLVE) revolve_create_offline(tjsch->stride,tjsch->max_cps_ram);
    else if (tjsch->stype == REVOLVE_OFFLINE) revolve_create_offline(tjsch->total_steps,tjsch->max_cps_ram);
    else if (tjsch->stype == REVOLVE_ONLINE) revolve_create_online(tjsch->max_cps_ram);
    else if (tjsch->stype == REVOLVE_MULTISTAGE) revolve_create_multistage(tjsch->total_steps,tjsch->max_cps_ram+tjsch->max_cps_disk,tjsch->max_cps_ram);

    ierr = PetscCalloc1(1,&rctx);CHKERRQ(ierr);
    rctx->snaps_in       = tjsch->max_cps_ram; /* for theta methods snaps_in=2*max_cps_ram */
    rctx->reverseonestep = PETSC_FALSE;
    rctx->check          = 0;
    rctx->oldcapo        = 0;
    rctx->capo           = 0;
    rctx->info           = 2;
    rctx->fine           = (tjsch->stride > 1) ? tjsch->stride : tjsch->total_steps;

    tjsch->rctx      = rctx;
    if (tjsch->stype == REVOLVE_ONLINE) rctx->fine = -1;
#endif
  }

  tjsch->recompute = PETSC_FALSE;
  tjsch->comm      = PetscObjectComm((PetscObject)ts);
  ierr = TSGetStages(ts,&numY,PETSC_IGNORE);CHKERRQ(ierr);
  ierr = StackCreate(stack,stack->stacksize,numY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSTrajectoryDestroy_Memory"
PETSC_EXTERN PetscErrorCode TSTrajectoryDestroy_Memory(TSTrajectory tj)
{
  TJScheduler    *tjsch = (TJScheduler*)tj->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (tjsch->stype > TWO_LEVEL_NOREVOLVE) {
#ifdef PETSC_HAVE_REVOLVE
    revolve_reset();
#endif
  }
  ierr = StackDestroy(&tjsch->stack);CHKERRQ(ierr);
#ifdef PETSC_HAVE_REVOLVE
  if (tjsch->stype > TWO_LEVEL_NOREVOLVE) {
    ierr = PetscFree(tjsch->rctx);CHKERRQ(ierr);
  }
#endif
  ierr = PetscFree(tjsch);CHKERRQ(ierr);
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
  TJScheduler    *tjsch;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  tj->ops->set            = TSTrajectorySet_Memory;
  tj->ops->get            = TSTrajectoryGet_Memory;
  tj->ops->setup          = TSTrajectorySetUp_Memory;
  tj->ops->destroy        = TSTrajectoryDestroy_Memory;
  tj->ops->setfromoptions = TSTrajectorySetFromOptions_Memory;

  ierr = PetscCalloc1(1,&tjsch);CHKERRQ(ierr);
  tjsch->stype        = NONE;
  tjsch->max_cps_ram  = -1; /* -1 indicates that it is not set */
  tjsch->max_cps_disk = -1; /* -1 indicates that it is not set */
  tjsch->stride       = 0; /* if not zero, two-level checkpointing will be used */
#ifdef PETSC_HAVE_REVOLVE
  tjsch->use_online   = PETSC_FALSE;
#endif
  tjsch->save_stack   = PETSC_TRUE;

  tjsch->stack.solution_only = PETSC_TRUE;

  tj->data = tjsch;

  PetscFunctionReturn(0);
}
