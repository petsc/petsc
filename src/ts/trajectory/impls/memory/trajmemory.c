#include <petsc/private/tsimpl.h>        /*I "petscts.h"  I*/
#include <petscsys.h>
#ifdef PETSC_HAVE_REVOLVE
#include <revolve_c.h>
#endif

PetscLogEvent TSTrajectory_DiskWrite, TSTrajectory_DiskRead;

typedef enum {NONE,TWO_LEVEL_NOREVOLVE,TWO_LEVEL_REVOLVE,TWO_LEVEL_TWO_REVOLVE,REVOLVE_OFFLINE,REVOLVE_ONLINE,REVOLVE_MULTISTAGE} SchedulerType;

typedef struct _StackElement {
  PetscInt  stepnum;
  Vec       X;
  Vec       *Y;
  PetscReal time;
  PetscReal timeprev; /* for no solution_only mode */
  PetscReal timenext; /* for solution_only mode */
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
  PetscInt      store_stride;
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
#define __FUNCT__ "TurnForwardWithStepsize"
static PetscErrorCode TurnForwardWithStepsize(TS ts,PetscReal nextstepsize)
{
    PetscReal      stepsize;
    PetscErrorCode ierr;

    PetscFunctionBegin;
    /* reverse the direction */
    ierr = TSGetTimeStep(ts,&stepsize);CHKERRQ(ierr);
    stepsize = nextstepsize;
    ierr = TSSetTimeStep(ts,stepsize);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TurnForward"
static PetscErrorCode TurnForward(TS ts)
{
  PetscReal      stepsize;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* reverse the direction */
  ierr = TSGetTimeStep(ts,&stepsize);CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts,-stepsize);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TurnBackward"
static PetscErrorCode TurnBackward(TS ts)
{
  PetscReal      stepsize;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* reverse the direction */
  stepsize = ts->ptime_prev-ts->ptime;
  ierr = TSSetTimeStep(ts,stepsize);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

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
  if (stack->top > -1) {
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
#define __FUNCT__ "StackResize"
static PetscErrorCode StackResize(Stack *stack,PetscInt newsize)
{
  StackElement   *newcontainer;
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscMalloc1(newsize*sizeof(StackElement),&newcontainer);CHKERRQ(ierr);
  for (i=0;i<stack->stacksize;i++) {
    newcontainer[i] = stack->container[i];
  }
  ierr = PetscFree(stack->container);CHKERRQ(ierr);
  stack->container = newcontainer;
  stack->stacksize = newsize;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "StackPush"
static PetscErrorCode StackPush(Stack *stack,StackElement e)
{
  PetscFunctionBegin;
  if (stack->top+1 >= stack->stacksize) SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_MEMC,"Maximum stack size (%D) exceeded",stack->stacksize);
  stack->container[++stack->top] = e;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "StackPop"
static PetscErrorCode StackPop(Stack *stack,StackElement *e)
{
  PetscFunctionBegin;
  if (stack->top == -1) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_MEMC,"Empty stack");
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
    ierr = PetscLogEventBegin(TSTrajectory_DiskWrite,ts,0,0,0);CHKERRQ(ierr);
    ierr = WriteToDisk(e->stepnum,e->time,e->timeprev,e->X,e->Y,stack->numY,stack->solution_only,viewer);CHKERRQ(ierr);
    ierr = PetscLogEventEnd(TSTrajectory_DiskWrite,ts,0,0,0);CHKERRQ(ierr);
    ts->trajectory->diskwrites++;
  }
  /* save the last step for restart, the last step is in memory when using single level schemes, but not necessarily the case for multi level schemes */
  ierr = TSGetStages(ts,&stack->numY,&Y);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(TSTrajectory_DiskWrite,ts,0,0,0);CHKERRQ(ierr);
  ierr = WriteToDisk(ts->total_steps,ts->ptime,ts->ptime_prev,ts->vec_sol,Y,stack->numY,stack->solution_only,viewer);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(TSTrajectory_DiskWrite,ts,0,0,0);CHKERRQ(ierr);
  ts->trajectory->diskwrites++;
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
  PetscPrintf(PETSC_COMM_WORLD,"\x1B[33mLoad stack from file\033[0m\n");
  ierr = PetscSNPrintf(filename,sizeof filename,"SA-data/SA-STACK%06d.bin",id);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_READ,&viewer);CHKERRQ(ierr);
  for (i=0;i<stack->stacksize;i++) {
    ierr = PetscCalloc1(1,&e);CHKERRQ(ierr);
    ierr = TSGetStages(ts,&stack->numY,&Y);CHKERRQ(ierr);
    ierr = VecDuplicate(Y[0],&e->X);CHKERRQ(ierr);
    if (!stack->solution_only && stack->numY>0) {
      ierr = VecDuplicateVecs(Y[0],stack->numY,&e->Y);CHKERRQ(ierr);
    }
    ierr = StackPush(stack,e);CHKERRQ(ierr);
  }
  for (i=0;i<stack->stacksize;i++) {
    e = stack->container[i];
    ierr = PetscLogEventBegin(TSTrajectory_DiskRead,ts,0,0,0);CHKERRQ(ierr);
    ierr = ReadFromDisk(&e->stepnum,&e->time,&e->timeprev,e->X,e->Y,stack->numY,stack->solution_only,viewer);CHKERRQ(ierr);
    ierr = PetscLogEventEnd(TSTrajectory_DiskRead,ts,0,0,0);CHKERRQ(ierr);
    ts->trajectory->diskreads++;
  }
  /* load the last step into TS */
  ierr = TSGetStages(ts,&stack->numY,&Y);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(TSTrajectory_DiskRead,ts,0,0,0);CHKERRQ(ierr);
  ierr = ReadFromDisk(&ts->total_steps,&ts->ptime,&ts->ptime_prev,ts->vec_sol,Y,stack->numY,stack->solution_only,viewer);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(TSTrajectory_DiskRead,ts,0,0,0);CHKERRQ(ierr);
  ts->trajectory->diskreads++;
  ierr = TurnBackward(ts);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#ifdef PETSC_HAVE_REVOLVE
#undef __FUNCT__
#define __FUNCT__ "StackLoadLast"
static PetscErrorCode StackLoadLast(TS ts,Stack *stack,PetscInt id)
{
  Vec            *Y;
  PetscInt       size;
  PetscViewer    viewer;
  char           filename[PETSC_MAX_PATH_LEN];
#if defined(PETSC_HAVE_MPIIO)
  PetscBool      usempiio;
#endif
  int            fd;
  off_t          off,offset;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"\x1B[33mLoad last stack element from file\033[0m\n");
  ierr = TSGetStages(ts,&stack->numY,&Y);CHKERRQ(ierr);
  ierr = VecGetSize(Y[0],&size);CHKERRQ(ierr);
  /* VecView writes to file two extra int's for class id and number of rows */
  off  = -((stack->solution_only?0:stack->numY)+1)*(size*PETSC_BINARY_SCALAR_SIZE+2*PETSC_BINARY_INT_SIZE)-PETSC_BINARY_INT_SIZE-2*PETSC_BINARY_SCALAR_SIZE;

  ierr = PetscSNPrintf(filename,sizeof filename,"SA-data/SA-STACK%06d.bin",id);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_READ,&viewer);CHKERRQ(ierr);
#if defined(PETSC_HAVE_MPIIO)
  ierr = PetscViewerBinaryGetUseMPIIO(viewer,&usempiio);
  if (usempiio) {
    ierr = PetscViewerBinaryGetMPIIODescriptor(viewer,(MPI_File*)&fd);CHKERRQ(ierr);
    ierr = PetscBinarySynchronizedSeek(PETSC_COMM_WORLD,fd,off,PETSC_BINARY_SEEK_END,&offset);CHKERRQ(ierr);
  } else {
#endif
    ierr = PetscViewerBinaryGetDescriptor(viewer,&fd);CHKERRQ(ierr);
    ierr = PetscBinarySeek(fd,off,PETSC_BINARY_SEEK_END,&offset);CHKERRQ(ierr);
#if defined(PETSC_HAVE_MPIIO)
  }
#endif
  /* load the last step into TS */
  ierr = PetscLogEventBegin(TSTrajectory_DiskRead,ts,0,0,0);CHKERRQ(ierr);
  ierr = ReadFromDisk(&ts->total_steps,&ts->ptime,&ts->ptime_prev,ts->vec_sol,Y,stack->numY,stack->solution_only,viewer);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(TSTrajectory_DiskRead,ts,0,0,0);CHKERRQ(ierr);
  ts->trajectory->diskreads++;
  ierr = TurnBackward(ts);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif

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
  if (id == 1) {
    PetscMPIInt rank;
    ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)ts),&rank);CHKERRQ(ierr);
    if (!rank) {
      ierr = PetscRMTree("SA-data");CHKERRQ(ierr);
      ierr = PetscMkdir("SA-data");CHKERRQ(ierr);
    }
  }
  ierr = PetscSNPrintf(filename,sizeof(filename),"SA-data/SA-CPS%06d.bin",id);CHKERRQ(ierr);
  ierr = OutputBIN(filename,&viewer);CHKERRQ(ierr);

  ierr = TSGetStages(ts,&stack->numY,&Y);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(TSTrajectory_DiskWrite,ts,0,0,0);CHKERRQ(ierr);
  ierr = WriteToDisk(stepnum,ts->ptime,ts->ptime_prev,ts->vec_sol,Y,stack->numY,stack->solution_only,viewer);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(TSTrajectory_DiskWrite,ts,0,0,0);CHKERRQ(ierr);
  ts->trajectory->diskwrites++;

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
  PetscPrintf(PETSC_COMM_WORLD,"\x1B[33mLoad a single point from file\033[0m\n");
  ierr = PetscSNPrintf(filename,sizeof filename,"SA-data/SA-CPS%06d.bin",id);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_READ,&viewer);CHKERRQ(ierr);

  ierr = TSGetStages(ts,&stack->numY,&Y);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(TSTrajectory_DiskRead,ts,0,0,0);CHKERRQ(ierr);
  ierr = ReadFromDisk(&ts->total_steps,&ts->ptime,&ts->ptime_prev,ts->vec_sol,Y,stack->numY,stack->solution_only,viewer);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(TSTrajectory_DiskRead,ts,0,0,0);CHKERRQ(ierr);
  ts->trajectory->diskreads++;

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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  adjsteps = ts->steps;
  ts->steps = stepnumbegin; /* global step number */
  for (i=stepnumbegin;i<stepnumend;i++) { /* assume fixed step size */
    if (stack->solution_only && !tjsch->skip_trajectory) { /* revolve online need this */
      ierr = TSTrajectorySet(ts->trajectory,ts,ts->steps,ts->ptime,ts->vec_sol);CHKERRQ(ierr);
    }
    ierr = TSMonitor(ts,ts->steps,ts->ptime,ts->vec_sol);CHKERRQ(ierr);
    ierr = TSStep(ts);CHKERRQ(ierr);
    if (!stack->solution_only && !tjsch->skip_trajectory) {
      ierr = TSTrajectorySet(ts->trajectory,ts,ts->steps,ts->ptime,ts->vec_sol);CHKERRQ(ierr);
    }
    ierr = TSEventHandler(ts);CHKERRQ(ierr);
    if (!ts->steprollback) {
      ierr = TSPostStep(ts);CHKERRQ(ierr);
    }
  }
  ierr = TurnBackward(ts);CHKERRQ(ierr);
  ts->trajectory->recomps += stepnumend-stepnumbegin; /* recomputation counter */
  ts->steps = adjsteps;
  ts->total_steps = stepnumend;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TopLevelStore"
static PetscErrorCode TopLevelStore(TS ts,TJScheduler *tjsch,PetscInt stepnum,PetscInt localstepnum,PetscInt laststridesize,PetscBool *done)
{
  Stack          *stack = &tjsch->stack;
  DiskStack      *diskstack = &tjsch->diskstack;
  PetscInt       stridenum;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  *done = PETSC_FALSE;
  stridenum    = stepnum/tjsch->stride;
  /* make sure saved checkpoint id starts from 1
     skip last stride when using stridenum+1
     skip first stride when using stridenum */
  if (stack->solution_only) {
    if (tjsch->save_stack) {
      if (localstepnum == tjsch->stride-1 && stepnum < tjsch->total_steps-laststridesize) { /* current step will be saved without going through stack */
        ierr = StackDumpAll(ts,stack,stridenum+1);CHKERRQ(ierr);
        if (tjsch->stype == TWO_LEVEL_TWO_REVOLVE) diskstack->container[++diskstack->top] = stridenum+1;
        PetscPrintf(PETSC_COMM_WORLD,"\x1B[33mDump stack to file\033[0m\n");
        *done = PETSC_TRUE;
      }
    } else {
      if (localstepnum == 0 && stepnum < tjsch->total_steps-laststridesize) {
        ierr = DumpSingle(ts,stack,stridenum+1);CHKERRQ(ierr);
        if (tjsch->stype == TWO_LEVEL_TWO_REVOLVE) diskstack->container[++diskstack->top] = stridenum+1;
        PetscPrintf(PETSC_COMM_WORLD,"\x1B[33mDump a single point (solution) to file\033[0m\n");
        *done = PETSC_TRUE;
      }
    }
  } else {
    if (tjsch->save_stack) {
      if (localstepnum == 0 && stepnum < tjsch->total_steps && stepnum != 0) { /* skip the first stride */
        ierr = StackDumpAll(ts,stack,stridenum);CHKERRQ(ierr);
        if (tjsch->stype == TWO_LEVEL_TWO_REVOLVE) diskstack->container[++diskstack->top] = stridenum;
        PetscPrintf(PETSC_COMM_WORLD,"\x1B[33mDump stack to file\033[0m\n");
        *done = PETSC_TRUE;
      }
    } else {
      if (localstepnum == 1 && stepnum < tjsch->total_steps-laststridesize) {
        ierr = DumpSingle(ts,stack,stridenum+1);CHKERRQ(ierr);
        if (tjsch->stype == TWO_LEVEL_TWO_REVOLVE) diskstack->container[++diskstack->top] = stridenum+1;
        PetscPrintf(PETSC_COMM_WORLD,"\x1B[33mDump a single point (solution+stages) to file\033[0m\n");
        *done = PETSC_TRUE;
      }
    }
  }
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
  /* skip the last step */
  if (ts->reason) { /* only affect the forward run */
    /* update total_steps in the end of forward run */
    if (stepnum != tjsch->total_steps) tjsch->total_steps = stepnum;
    if (stack->solution_only) {
      /* get rid of the solution at second last step */
      ierr = StackPop(stack,&e);CHKERRQ(ierr);
      ierr = ElementDestroy(stack,e);CHKERRQ(ierr);
    }
    PetscFunctionReturn(0);
  }
  /*  do not save trajectory at the recompute stage for solution_only mode */
  if (stack->solution_only && tjsch->recompute) PetscFunctionReturn(0);
  /* skip the first step for no_solution_only mode */
  if (!stack->solution_only && stepnum == 0) PetscFunctionReturn(0);

  /* resize the stack */
  if (stack->top+1 == stack->stacksize) {
    ierr = StackResize(stack,2*stack->stacksize);CHKERRQ(ierr);
  }
  /* update timenext for the previous step; necessary for step adaptivity */
  if (stack->top > -1) {
    ierr = StackTop(stack,&e);CHKERRQ(ierr);
    e->timenext = ts->ptime;
  }
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
  StackElement   e;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (stepnum == tjsch->total_steps) {
    ierr = TurnBackward(ts);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  /* restore a checkpoint */
  ierr = StackTop(stack,&e);CHKERRQ(ierr);
  ierr = UpdateTS(ts,stack,e);CHKERRQ(ierr);
  if (stack->solution_only) {/* recompute one step */
    tjsch->recompute = PETSC_TRUE;
    ierr = TurnForwardWithStepsize(ts,e->timenext-e->time);CHKERRQ(ierr);
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
  PetscInt       localstepnum,laststridesize;
  StackElement   e;
  PetscBool      done;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!stack->solution_only && stepnum == 0) PetscFunctionReturn(0);
  if (stack->solution_only && stepnum == tjsch->total_steps) PetscFunctionReturn(0);
  if (tjsch->save_stack && tjsch->recompute) PetscFunctionReturn(0);

  localstepnum = stepnum%tjsch->stride;
  /* (stride size-1) checkpoints are saved in each stride; an extra point is added by StackDumpAll() */
  laststridesize = tjsch->total_steps%tjsch->stride;
  if (!laststridesize) laststridesize = tjsch->stride;

  if (!tjsch->recompute) {
    ierr = TopLevelStore(ts,tjsch,stepnum,localstepnum,laststridesize,&done);CHKERRQ(ierr);
    if (!tjsch->save_stack && stepnum < tjsch->total_steps-laststridesize) PetscFunctionReturn(0);
  }
  if (!stack->solution_only && localstepnum == 0) PetscFunctionReturn(0); /* skip last point in each stride at recompute stage or last stride */
  if (stack->solution_only && localstepnum == tjsch->stride-1) PetscFunctionReturn(0); /* skip last step in each stride at recompute stage or last stride */

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
  StackElement   e;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (stepnum == tjsch->total_steps) {
    ierr = TurnBackward(ts);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  localstepnum = stepnum%tjsch->stride;
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
        ierr = TurnForward(ts);CHKERRQ(ierr);
        ierr = ReCompute(ts,tjsch,id*tjsch->stride-1,id*tjsch->stride);CHKERRQ(ierr);
        tjsch->skip_trajectory = PETSC_FALSE;
      } else {
        ierr = LoadSingle(ts,stack,id);CHKERRQ(ierr);
        tjsch->recompute = PETSC_TRUE;
        ierr = TurnForward(ts);CHKERRQ(ierr);
        ierr = ReCompute(ts,tjsch,(id-1)*tjsch->stride,id*tjsch->stride);CHKERRQ(ierr);
      }
      PetscFunctionReturn(0);
    }
    /* restore a checkpoint */
    ierr = StackPop(stack,&e);CHKERRQ(ierr);
    ierr = UpdateTS(ts,stack,e);CHKERRQ(ierr);
    tjsch->recompute = PETSC_TRUE;
    tjsch->skip_trajectory = PETSC_TRUE;
    ierr = TurnForward(ts);CHKERRQ(ierr);
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
        ierr = LoadSingle(ts,stack,id);CHKERRQ(ierr);
        ierr = ElementCreate(ts,stack,&e,(id-1)*tjsch->stride+1,ts->ptime,ts->vec_sol);CHKERRQ(ierr);
        ierr = StackPush(stack,e);CHKERRQ(ierr);
        tjsch->recompute = PETSC_TRUE;
        ierr = TurnForward(ts);CHKERRQ(ierr);
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

static void printwhattodo2(PetscInt whattodo,RevolveCTX *rctx,PetscInt shift)
{
  switch(whattodo) {
    case 1:
      PetscPrintf(PETSC_COMM_WORLD,"\x1B[35m[Top Level] Advance from stride %D to stride %D\033[0m\n",rctx->oldcapo+shift,rctx->capo+shift);
      break;
    case 2:
      PetscPrintf(PETSC_COMM_WORLD,"\x1B[35m[Top Level] Store in checkpoint number %D\033[0m\n",rctx->check);
      break;
    case 3:
      PetscPrintf(PETSC_COMM_WORLD,"\x1B[35m[Top Level] First turn: Initialize adjoints and reverse first stride\033[0m\n");
      break;
    case 4:
      PetscPrintf(PETSC_COMM_WORLD,"\x1B[35m[Top Level] Forward and reverse one stride\033[0m\n");
      break;
    case 5:
      PetscPrintf(PETSC_COMM_WORLD,"\x1B[35m[Top Level] Restore in checkpoint number %D\033[0m\n",rctx->check);
      break;
    case 7:
      PetscPrintf(PETSC_COMM_WORLD,"\x1B[35m[Top Level] Store in top-level checkpoint number %D\033[0m\n",rctx->check);
      break;
    case 8:
      PetscPrintf(PETSC_COMM_WORLD,"\x1B[35m[Top Level] Restore in top-level checkpoint number %D\033[0m\n",rctx->check);
      break;
    case -1:
      PetscPrintf(PETSC_COMM_WORLD,"\x1B[35m[Top Level] Error!");
      break;
  }
}

#undef __FUNCT__
#define __FUNCT__ "InitRevolve"
static PetscErrorCode InitRevolve(PetscInt fine,PetscInt snaps,RevolveCTX *rctx)
{
  PetscFunctionBegin;
  revolve_reset();
  revolve_create_offline(fine,snaps);
  rctx->snaps_in       = snaps;
  rctx->fine           = fine;
  rctx->check          = 0;
  rctx->capo           = 0;
  rctx->reverseonestep = PETSC_FALSE;
  /* check stepsleft? */
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FastForwardRevolve"
static PetscErrorCode FastForwardRevolve(RevolveCTX *rctx)
{
  PetscInt whattodo;

  PetscFunctionBegin;
  whattodo = 0;
  while(whattodo!=3) { /* we have to fast forward revolve to the beginning of the backward sweep due to unfriendly revolve interface */
    whattodo = revolve_action(&rctx->check,&rctx->capo,&rctx->fine,rctx->snaps_in,&rctx->info,&rctx->where);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ApplyRevolve"
static PetscErrorCode ApplyRevolve(SchedulerType stype,RevolveCTX *rctx,PetscInt total_steps,PetscInt stepnum,PetscInt localstepnum,PetscBool toplevel,PetscInt *store)
{
  PetscInt       shift,whattodo;

  PetscFunctionBegin;
  *store = 0;
  if (rctx->stepsleft > 0) { /* advance the solution without checkpointing anything as Revolve requires */
    rctx->stepsleft--;
    PetscFunctionReturn(0);
  }
  /* let Revolve determine what to do next */
  shift         = stepnum-localstepnum;
  rctx->oldcapo = rctx->capo;
  rctx->capo    = localstepnum;

  if (!toplevel) whattodo = revolve_action(&rctx->check,&rctx->capo,&rctx->fine,rctx->snaps_in,&rctx->info,&rctx->where);
  else whattodo = revolve2_action(&rctx->check,&rctx->capo,&rctx->fine,rctx->snaps_in,&rctx->info,&rctx->where);
  if (stype == REVOLVE_ONLINE && whattodo == 8) whattodo = 5;
  if (stype == REVOLVE_ONLINE && whattodo == 7) whattodo = 2;
  if (!toplevel) printwhattodo(whattodo,rctx,shift);
  else printwhattodo2(whattodo,rctx,shift);
  if (whattodo == -1) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_LIB,"Error in the Revolve library");
  if (whattodo == 1) { /* advance some time steps */
    if (stype == REVOLVE_ONLINE && rctx->capo >= total_steps-1) {
      revolve_turn(total_steps,&rctx->capo,&rctx->fine);
      if (!toplevel) printwhattodo(whattodo,rctx,shift);
      else printwhattodo2(whattodo,rctx,shift);
    }
    rctx->stepsleft = rctx->capo-rctx->oldcapo-1;
  }
  if (whattodo == 3 || whattodo == 4) { /* ready for a reverse step */
    rctx->reverseonestep = PETSC_TRUE;
  }
  if (whattodo == 5) { /* restore a checkpoint and ask Revolve what to do next */
    rctx->oldcapo = rctx->capo;
    if (!toplevel) whattodo = revolve_action(&rctx->check,&rctx->capo,&rctx->fine,rctx->snaps_in,&rctx->info,&rctx->where); /* must return 1 or 3 or 4*/
    else whattodo = revolve2_action(&rctx->check,&rctx->capo,&rctx->fine,rctx->snaps_in,&rctx->info,&rctx->where);
    if (!toplevel) printwhattodo(whattodo,rctx,shift);
    else printwhattodo2(whattodo,rctx,shift);
    if (whattodo == 3 || whattodo == 4) rctx->reverseonestep = PETSC_TRUE;
    if (whattodo == 1) rctx->stepsleft = rctx->capo-rctx->oldcapo;
  }
  if (whattodo == 7) { /* save the checkpoint to disk */
    *store = 2;
    rctx->oldcapo = rctx->capo;
    whattodo = revolve_action(&rctx->check,&rctx->capo,&rctx->fine,rctx->snaps_in,&rctx->info,&rctx->where); /* must return 1 */
    printwhattodo(whattodo,rctx,shift);
    rctx->stepsleft = rctx->capo-rctx->oldcapo-1;
  }
  if (whattodo == 2) { /* store a checkpoint to RAM and ask Revolve how many time steps to advance next */
    *store = 1;
    rctx->oldcapo = rctx->capo;
    if (!toplevel) whattodo = revolve_action(&rctx->check,&rctx->capo,&rctx->fine,rctx->snaps_in,&rctx->info,&rctx->where); /* must return 1 */
    else whattodo = revolve2_action(&rctx->check,&rctx->capo,&rctx->fine,rctx->snaps_in,&rctx->info,&rctx->where);
    if (!toplevel) printwhattodo(whattodo,rctx,shift);
    else printwhattodo2(whattodo,rctx,shift);
    if (stype == REVOLVE_ONLINE && rctx->capo >= total_steps-1) {
      revolve_turn(total_steps,&rctx->capo,&rctx->fine);
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
  if (stack->solution_only && stepnum == tjsch->total_steps) PetscFunctionReturn(0);
  ierr = ApplyRevolve(tjsch->stype,tjsch->rctx,tjsch->total_steps,stepnum,stepnum,PETSC_FALSE,&store);CHKERRQ(ierr);
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
  StackElement   e;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (stepnum == 0 || stepnum == tjsch->total_steps) {
    ierr = TurnBackward(ts);CHKERRQ(ierr);
    tjsch->rctx->reverseonestep = PETSC_FALSE;
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
    ierr = ApplyRevolve(tjsch->stype,tjsch->rctx,tjsch->total_steps,stepnum,stepnum,PETSC_FALSE,&store);CHKERRQ(ierr);
    PetscPrintf(PETSC_COMM_WORLD,"\x1B[35mSkip the step from %D to %D (stage values already checkpointed)\033[0m\n",tjsch->rctx->oldcapo,tjsch->rctx->oldcapo+1);
    if (!tjsch->rctx->reverseonestep && tjsch->rctx->stepsleft > 0) tjsch->rctx->stepsleft--;
  }
  if (stack->solution_only || (!stack->solution_only && e->stepnum < stepnum)) {
    tjsch->recompute = PETSC_TRUE;
    ierr = TurnForward(ts);CHKERRQ(ierr);
    ierr = ReCompute(ts,tjsch,e->stepnum,stepnum);CHKERRQ(ierr);
  }
  if ((stack->solution_only && e->stepnum+1 == stepnum) || (!stack->solution_only && e->stepnum == stepnum)) {
    ierr = StackPop(stack,&e);CHKERRQ(ierr);
    ierr = ElementDestroy(stack,e);CHKERRQ(ierr);
  }
  tjsch->rctx->reverseonestep = PETSC_FALSE;
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
  if (stack->solution_only && stepnum == tjsch->total_steps) PetscFunctionReturn(0);
  ierr = ApplyRevolve(tjsch->stype,rctx,tjsch->total_steps,stepnum,stepnum,PETSC_FALSE,&store);CHKERRQ(ierr);
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
  StackElement   e;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (stepnum == 0 || stepnum == tjsch->total_steps) {
    ierr = TurnBackward(ts);CHKERRQ(ierr);
    tjsch->rctx->reverseonestep = PETSC_FALSE;
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
    PetscPrintf(PETSC_COMM_WORLD,"\x1B[35mSkip the step from %D to %D (stage values already checkpointed)\033[0m\n",tjsch->rctx->oldcapo,tjsch->rctx->oldcapo+1);
    if (!tjsch->rctx->reverseonestep && tjsch->rctx->stepsleft > 0) tjsch->rctx->stepsleft--;
  }
  if (stack->solution_only || (!stack->solution_only && e->stepnum < stepnum)) {
    tjsch->recompute = PETSC_TRUE;
    ierr = TurnForward(ts);CHKERRQ(ierr);
    ierr = ReCompute(ts,tjsch,e->stepnum,stepnum);CHKERRQ(ierr);
  }
  tjsch->rctx->reverseonestep = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SetTrajTLR"
static PetscErrorCode SetTrajTLR(TS ts,TJScheduler *tjsch,PetscInt stepnum,PetscReal time,Vec X)
{
  Stack          *stack = &tjsch->stack;
  PetscInt       store,localstepnum,laststridesize;
  StackElement   e;
  PetscBool      done = PETSC_FALSE;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!stack->solution_only && stepnum == 0) PetscFunctionReturn(0);
  if (stack->solution_only && stepnum == tjsch->total_steps) PetscFunctionReturn(0);

  localstepnum = stepnum%tjsch->stride;
  laststridesize = tjsch->total_steps%tjsch->stride;
  if (!laststridesize) laststridesize = tjsch->stride;

  if (!tjsch->recompute) {
    ierr = TopLevelStore(ts,tjsch,stepnum,localstepnum,laststridesize,&done);CHKERRQ(ierr);
    /* revolve is needed for the last stride; different starting points for last stride between solutin_only and !solutin_only */
    if (!stack->solution_only && !tjsch->save_stack && stepnum <= tjsch->total_steps-laststridesize) PetscFunctionReturn(0);
    if (stack->solution_only && !tjsch->save_stack && stepnum < tjsch->total_steps-laststridesize) PetscFunctionReturn(0);
  }
  if (tjsch->save_stack && done) {
    ierr = InitRevolve(tjsch->stride,tjsch->max_cps_ram,tjsch->rctx);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  if (laststridesize < tjsch->stride) {
    if (stack->solution_only && stepnum == tjsch->total_steps-laststridesize && !tjsch->recompute) { /* step tjsch->total_steps-laststridesize-1 is skipped, but the next step is not */
      ierr = InitRevolve(laststridesize,tjsch->max_cps_ram,tjsch->rctx);CHKERRQ(ierr);
    }
    if (!stack->solution_only && stepnum == tjsch->total_steps-laststridesize+1 && !tjsch->recompute) { /* step tjsch->total_steps-laststridesize is skipped, but the next step is not */
      ierr = InitRevolve(laststridesize,tjsch->max_cps_ram,tjsch->rctx);CHKERRQ(ierr);
    }
  }
  ierr = ApplyRevolve(tjsch->stype,tjsch->rctx,tjsch->total_steps,stepnum,localstepnum,PETSC_FALSE,&store);CHKERRQ(ierr);
  if (store == 1) {
    if (localstepnum < stack->top) SETERRQ(tjsch->comm,PETSC_ERR_MEMC,"Illegal modification of a non-top stack element");
    ierr = ElementCreate(ts,stack,&e,stepnum,time,X);CHKERRQ(ierr);
    ierr = StackPush(stack,e);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "GetTrajTLR"
static PetscErrorCode GetTrajTLR(TS ts,TJScheduler *tjsch,PetscInt stepnum)
{
  Stack          *stack = &tjsch->stack;
  PetscInt       whattodo,shift;
  PetscInt       localstepnum,stridenum,laststridesize,store;
  StackElement   e;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  localstepnum = stepnum%tjsch->stride;
  stridenum    = stepnum/tjsch->stride;
  if (stepnum == tjsch->total_steps) {
    ierr = TurnBackward(ts);CHKERRQ(ierr);
    tjsch->rctx->reverseonestep = PETSC_FALSE;
    PetscFunctionReturn(0);
  }
  laststridesize = tjsch->total_steps%tjsch->stride;
  if (!laststridesize) laststridesize = tjsch->stride;
  if (stack->solution_only) {
    /* fill stack */
    if (localstepnum == 0 && stepnum <= tjsch->total_steps-laststridesize) {
      if (tjsch->save_stack) {
        ierr = StackLoadAll(ts,stack,stridenum);CHKERRQ(ierr);
        ierr = InitRevolve(tjsch->stride,tjsch->max_cps_ram,tjsch->rctx);CHKERRQ(ierr);
        ierr = FastForwardRevolve(tjsch->rctx);CHKERRQ(ierr);
        tjsch->recompute = PETSC_TRUE;
        tjsch->skip_trajectory = PETSC_TRUE;
        ierr = TurnForward(ts);CHKERRQ(ierr);
        ierr = ReCompute(ts,tjsch,stridenum*tjsch->stride-1,stridenum*tjsch->stride);CHKERRQ(ierr);
        tjsch->skip_trajectory = PETSC_FALSE;
      } else {
        ierr = LoadSingle(ts,stack,stridenum);CHKERRQ(ierr);
        ierr = InitRevolve(tjsch->stride,tjsch->max_cps_ram,tjsch->rctx);CHKERRQ(ierr);
        tjsch->recompute = PETSC_TRUE;
        ierr = TurnForward(ts);CHKERRQ(ierr);
        ierr = ReCompute(ts,tjsch,(stridenum-1)*tjsch->stride,stridenum*tjsch->stride);CHKERRQ(ierr);
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
    ierr = TurnForward(ts);CHKERRQ(ierr);
    ierr = ReCompute(ts,tjsch,e->stepnum,stepnum);CHKERRQ(ierr);
    if (e->stepnum+1 == stepnum) {
      ierr = StackPop(stack,&e);CHKERRQ(ierr);
      ierr = ElementDestroy(stack,e);CHKERRQ(ierr);
    }
  } else {
    /* fill stack with info */
    if (localstepnum == 0 && tjsch->total_steps-stepnum >= laststridesize) {
      if (tjsch->save_stack) {
        ierr = StackLoadAll(ts,stack,stridenum);CHKERRQ(ierr);
        ierr = InitRevolve(tjsch->stride,tjsch->max_cps_ram,tjsch->rctx);CHKERRQ(ierr);
        ierr = FastForwardRevolve(tjsch->rctx);CHKERRQ(ierr);
      } else {
        ierr = LoadSingle(ts,stack,stridenum);CHKERRQ(ierr);
        ierr = InitRevolve(tjsch->stride,tjsch->max_cps_ram,tjsch->rctx);CHKERRQ(ierr);
        ierr = ApplyRevolve(tjsch->stype,tjsch->rctx,tjsch->total_steps,(stridenum-1)*tjsch->stride+1,1,PETSC_FALSE,&store);CHKERRQ(ierr);
        PetscPrintf(PETSC_COMM_WORLD,"\x1B[35mSkip the step from %D to %D (stage values already checkpointed)\033[0m\n",(stridenum-1)*tjsch->stride+tjsch->rctx->oldcapo,(stridenum-1)*tjsch->stride+tjsch->rctx->oldcapo+1);
        ierr = ElementCreate(ts,stack,&e,(stridenum-1)*tjsch->stride+1,ts->ptime,ts->vec_sol);CHKERRQ(ierr);
        ierr = StackPush(stack,e);CHKERRQ(ierr);
        tjsch->recompute = PETSC_TRUE;
        ierr = TurnForward(ts);CHKERRQ(ierr);
        ierr = ReCompute(ts,tjsch,e->stepnum,stridenum*tjsch->stride);CHKERRQ(ierr);
      }
      PetscFunctionReturn(0);
    }
    /* restore a checkpoint */
    ierr = StackTop(stack,&e);CHKERRQ(ierr);
    ierr = UpdateTS(ts,stack,e);CHKERRQ(ierr);
    /* 2 revolve actions: restore a checkpoint and then advance */
    ierr = ApplyRevolve(tjsch->stype,tjsch->rctx,tjsch->total_steps,stepnum,localstepnum,PETSC_FALSE,&store);CHKERRQ(ierr);
    PetscPrintf(PETSC_COMM_WORLD,"\x1B[35mSkip the step from %D to %D (stage values already checkpointed)\033[0m\n",stepnum-localstepnum+tjsch->rctx->oldcapo,stepnum-localstepnum+tjsch->rctx->oldcapo+1);
    if (!tjsch->rctx->reverseonestep && tjsch->rctx->stepsleft > 0) tjsch->rctx->stepsleft--;
    if (e->stepnum < stepnum) {
      tjsch->recompute = PETSC_TRUE;
      ierr = TurnForward(ts);CHKERRQ(ierr);
      ierr = ReCompute(ts,tjsch,e->stepnum,stepnum);CHKERRQ(ierr);
    }
    if (e->stepnum == stepnum) {
      ierr = StackPop(stack,&e);CHKERRQ(ierr);
      ierr = ElementDestroy(stack,e);CHKERRQ(ierr);
    }
  }
  tjsch->rctx->reverseonestep = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SetTrajTLTR"
static PetscErrorCode SetTrajTLTR(TS ts,TJScheduler *tjsch,PetscInt stepnum,PetscReal time,Vec X)
{
  Stack          *stack = &tjsch->stack;
  PetscInt       store,localstepnum,stridenum,laststridesize;
  StackElement   e;
  PetscBool      done = PETSC_FALSE;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!stack->solution_only && stepnum == 0) PetscFunctionReturn(0);
  if (stack->solution_only && stepnum == tjsch->total_steps) PetscFunctionReturn(0);

  localstepnum = stepnum%tjsch->stride; /* index at the bottom level (inside a stride) */
  stridenum    = stepnum/tjsch->stride; /* index at the top level */
  laststridesize = tjsch->total_steps%tjsch->stride;
  if (!laststridesize) laststridesize = tjsch->stride;
  if (stack->solution_only && localstepnum == 0 && !tjsch->rctx2->reverseonestep) {
    ierr = ApplyRevolve(tjsch->stype,tjsch->rctx2,(tjsch->total_steps+tjsch->stride-1)/tjsch->stride,stridenum,stridenum,PETSC_TRUE,&tjsch->store_stride);CHKERRQ(ierr);
    if (laststridesize < tjsch->stride && stepnum == tjsch->total_steps-laststridesize) {
      ierr = InitRevolve(laststridesize,tjsch->max_cps_ram,tjsch->rctx);CHKERRQ(ierr);
    }
  }
  if (!stack->solution_only && localstepnum == 1 && !tjsch->rctx2->reverseonestep) {
    ierr = ApplyRevolve(tjsch->stype,tjsch->rctx2,(tjsch->total_steps+tjsch->stride-1)/tjsch->stride,stridenum,stridenum,PETSC_TRUE,&tjsch->store_stride);CHKERRQ(ierr);
    if (laststridesize < tjsch->stride && stepnum == tjsch->total_steps-laststridesize+1) {
      ierr = InitRevolve(laststridesize,tjsch->max_cps_ram,tjsch->rctx);CHKERRQ(ierr);
    }
  }
  if (tjsch->store_stride) {
    ierr = TopLevelStore(ts,tjsch,stepnum,localstepnum,laststridesize,&done);CHKERRQ(ierr);
    if (done) {
      ierr = InitRevolve(tjsch->stride,tjsch->max_cps_ram,tjsch->rctx);CHKERRQ(ierr);
      PetscFunctionReturn(0);
    }
  }
  if (stepnum < tjsch->total_steps-laststridesize) {
    if (tjsch->save_stack && !tjsch->store_stride && !tjsch->rctx2->reverseonestep) PetscFunctionReturn(0); /* store or forward-and-reverse at top level trigger revolve at bottom level */
    if (!tjsch->save_stack && !tjsch->rctx2->reverseonestep) PetscFunctionReturn(0); /* store operation does not require revolve be called at bottom level */
  }
  /* Skipping stepnum=0 for !stack->only is enough for TLR, but not for TLTR. Here we skip the first step for each stride so that the top-level revolve is applied (always at localstepnum=1) ahead of the bottom-level revolve */
  if (!stack->solution_only && localstepnum == 0 && stepnum != tjsch->total_steps && !tjsch->recompute) PetscFunctionReturn(0);
  ierr = ApplyRevolve(tjsch->stype,tjsch->rctx,tjsch->total_steps,stepnum,localstepnum,PETSC_FALSE,&store);CHKERRQ(ierr);
  if (store == 1) {
    if (localstepnum < stack->top) SETERRQ(tjsch->comm,PETSC_ERR_MEMC,"Illegal modification of a non-top stack element");
    ierr = ElementCreate(ts,stack,&e,stepnum,time,X);CHKERRQ(ierr);
    ierr = StackPush(stack,e);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "GetTrajTLTR"
static PetscErrorCode GetTrajTLTR(TS ts,TJScheduler *tjsch,PetscInt stepnum)
{
  Stack          *stack = &tjsch->stack;
  DiskStack      *diskstack = &tjsch->diskstack;
  PetscInt       whattodo,shift;
  PetscInt       localstepnum,stridenum,restoredstridenum,laststridesize,store;
  StackElement   e;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  localstepnum = stepnum%tjsch->stride;
  stridenum    = stepnum/tjsch->stride;
  if (stepnum == tjsch->total_steps) {
    ierr = TurnBackward(ts);CHKERRQ(ierr);
    tjsch->rctx->reverseonestep = PETSC_FALSE;
    PetscFunctionReturn(0);
  }
  laststridesize = tjsch->total_steps%tjsch->stride;
  if (!laststridesize) laststridesize = tjsch->stride;
  /*
   Last stride can be adjoined directly. All the other strides require that the stack in memory be ready before an adjoint step is taken (at the end of each stride). The following two cases need to be addressed differently:
     Case 1 (save_stack)
       Restore a disk checkpoint; update TS with the last element in the restored data; recompute to the current point.
     Case 2 (!save_stack)
       Restore a disk checkpoint; update TS with the restored point; recompute to the current point.
  */
  if (localstepnum == 0 && stepnum <= tjsch->total_steps-laststridesize) {
    /* restore the top element in the stack for disk checkpoints */
    restoredstridenum = diskstack->container[diskstack->top];
    tjsch->rctx2->reverseonestep = PETSC_FALSE;
    /* top-level revolve must be applied before current step, just like the solution_only mode for single-level revolve */
    if (!tjsch->save_stack && stack->solution_only) { /* start with restoring a checkpoint */
      tjsch->rctx2->capo = stridenum;
      tjsch->rctx2->oldcapo = tjsch->rctx2->capo;
      shift = 0;
      whattodo = revolve2_action(&tjsch->rctx2->check,&tjsch->rctx2->capo,&tjsch->rctx2->fine,tjsch->rctx2->snaps_in,&tjsch->rctx2->info,&tjsch->rctx2->where);
      printwhattodo2(whattodo,tjsch->rctx2,shift);
    } else { /* 2 revolve actions: restore a checkpoint and then advance */
      ierr = ApplyRevolve(tjsch->stype,tjsch->rctx2,(tjsch->total_steps+tjsch->stride-1)/tjsch->stride,stridenum,stridenum,PETSC_TRUE,&tjsch->store_stride);CHKERRQ(ierr);
      PetscPrintf(PETSC_COMM_WORLD,"\x1B[35m[Top Level] Skip the stride from %D to %D (stage values already checkpointed)\033[0m\n",tjsch->rctx2->oldcapo,tjsch->rctx2->oldcapo+1);
      if (!tjsch->rctx2->reverseonestep && tjsch->rctx2->stepsleft > 0) tjsch->rctx2->stepsleft--;
    }
    /* fill stack */
    if (stack->solution_only) {
      if (tjsch->save_stack) {
        if (restoredstridenum < stridenum) {
          ierr = StackLoadLast(ts,stack,restoredstridenum);CHKERRQ(ierr);
        } else {
          ierr = StackLoadAll(ts,stack,restoredstridenum);CHKERRQ(ierr);
        }
        /* recompute one step ahead */
        tjsch->recompute = PETSC_TRUE;
        tjsch->skip_trajectory = PETSC_TRUE;
        ierr = TurnForward(ts);CHKERRQ(ierr);
        ierr = ReCompute(ts,tjsch,stridenum*tjsch->stride-1,stridenum*tjsch->stride);CHKERRQ(ierr);
        tjsch->skip_trajectory = PETSC_FALSE;
        if (restoredstridenum < stridenum) {
          ierr = InitRevolve(tjsch->stride,tjsch->max_cps_ram,tjsch->rctx);CHKERRQ(ierr);
          tjsch->recompute = PETSC_TRUE;
          ierr = TurnForward(ts);CHKERRQ(ierr);
          ierr = ReCompute(ts,tjsch,restoredstridenum*tjsch->stride,stepnum);CHKERRQ(ierr);
        } else { /* stack ready, fast forward revolve status */
          ierr = InitRevolve(tjsch->stride,tjsch->max_cps_ram,tjsch->rctx);CHKERRQ(ierr);
          ierr = FastForwardRevolve(tjsch->rctx);CHKERRQ(ierr);
        }
      } else {
        ierr = LoadSingle(ts,stack,restoredstridenum);CHKERRQ(ierr);
        ierr = InitRevolve(tjsch->stride,tjsch->max_cps_ram,tjsch->rctx);CHKERRQ(ierr);
        tjsch->recompute = PETSC_TRUE;
        ierr = TurnForward(ts);CHKERRQ(ierr);
        ierr = ReCompute(ts,tjsch,(restoredstridenum-1)*tjsch->stride,stepnum);CHKERRQ(ierr);
      }
    } else {
      if (tjsch->save_stack) {
        if (restoredstridenum < stridenum) {
          ierr = StackLoadLast(ts,stack,restoredstridenum);CHKERRQ(ierr);
          /* reset revolve */
          ierr = InitRevolve(tjsch->stride,tjsch->max_cps_ram,tjsch->rctx);CHKERRQ(ierr);
          tjsch->recompute = PETSC_TRUE;
          ierr = TurnForward(ts);CHKERRQ(ierr);
          ierr = ReCompute(ts,tjsch,restoredstridenum*tjsch->stride,stepnum);CHKERRQ(ierr);
        } else { /* stack ready, fast forward revolve status */
          ierr = StackLoadAll(ts,stack,restoredstridenum);CHKERRQ(ierr);
          ierr = InitRevolve(tjsch->stride,tjsch->max_cps_ram,tjsch->rctx);CHKERRQ(ierr);
          ierr = FastForwardRevolve(tjsch->rctx);CHKERRQ(ierr);
        }
      } else {
        ierr = LoadSingle(ts,stack,restoredstridenum);CHKERRQ(ierr);
        ierr = InitRevolve(tjsch->stride,tjsch->max_cps_ram,tjsch->rctx);CHKERRQ(ierr);
        /* push first element to stack */
        if (tjsch->store_stride || tjsch->rctx2->reverseonestep) {
          shift = (restoredstridenum-1)*tjsch->stride-localstepnum;
          ierr = ApplyRevolve(tjsch->stype,tjsch->rctx,tjsch->total_steps,(restoredstridenum-1)*tjsch->stride+1,1,PETSC_FALSE,&store);CHKERRQ(ierr);
          PetscPrintf(PETSC_COMM_WORLD,"\x1B[35mSkip the step from %D to %D (stage values already checkpointed)\033[0m\n",(restoredstridenum-1)*tjsch->stride,(restoredstridenum-1)*tjsch->stride+1);
          ierr = ElementCreate(ts,stack,&e,(restoredstridenum-1)*tjsch->stride+1,ts->ptime,ts->vec_sol);CHKERRQ(ierr);
          ierr = StackPush(stack,e);CHKERRQ(ierr);
        }
        tjsch->recompute = PETSC_TRUE;
        ierr = TurnForward(ts);CHKERRQ(ierr);
        ierr = ReCompute(ts,tjsch,(restoredstridenum-1)*tjsch->stride+1,stepnum);CHKERRQ(ierr);
      }
    }
    if (restoredstridenum == stridenum) diskstack->top--;
    tjsch->rctx->reverseonestep = PETSC_FALSE;
    PetscFunctionReturn(0);
  }

  if (stack->solution_only) {
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
    ierr = TurnForward(ts);CHKERRQ(ierr);
    ierr = ReCompute(ts,tjsch,e->stepnum,stepnum);CHKERRQ(ierr);
    if (e->stepnum+1 == stepnum) {
      ierr = StackPop(stack,&e);CHKERRQ(ierr);
      ierr = ElementDestroy(stack,e);CHKERRQ(ierr);
    }
  } else {
    /* restore a checkpoint */
    ierr = StackTop(stack,&e);CHKERRQ(ierr);
    ierr = UpdateTS(ts,stack,e);CHKERRQ(ierr);
    /* 2 revolve actions: restore a checkpoint and then advance */
    ierr = ApplyRevolve(tjsch->stype,tjsch->rctx,tjsch->total_steps,stepnum,localstepnum,PETSC_FALSE,&store);CHKERRQ(ierr);
    PetscPrintf(PETSC_COMM_WORLD,"\x1B[35mSkip the step from %D to %D (stage values already checkpointed)\033[0m\n",stepnum-localstepnum+tjsch->rctx->oldcapo,stepnum-localstepnum+tjsch->rctx->oldcapo+1);
    if (!tjsch->rctx->reverseonestep && tjsch->rctx->stepsleft > 0) tjsch->rctx->stepsleft--;
    if (e->stepnum < stepnum) {
      tjsch->recompute = PETSC_TRUE;
      ierr = TurnForward(ts);CHKERRQ(ierr);
      ierr = ReCompute(ts,tjsch,e->stepnum,stepnum);CHKERRQ(ierr);
    }
    if (e->stepnum == stepnum) {
      ierr = StackPop(stack,&e);CHKERRQ(ierr);
      ierr = ElementDestroy(stack,e);CHKERRQ(ierr);
    }
  }
  tjsch->rctx->reverseonestep = PETSC_FALSE;
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
  if (stack->solution_only && stepnum == tjsch->total_steps) PetscFunctionReturn(0);
  ierr = ApplyRevolve(tjsch->stype,tjsch->rctx,tjsch->total_steps,stepnum,stepnum,PETSC_FALSE,&store);CHKERRQ(ierr);
  if (store == 1){
    if (stepnum < stack->top) SETERRQ(tjsch->comm,PETSC_ERR_MEMC,"Illegal modification of a non-top stack element");
    ierr = ElementCreate(ts,stack,&e,stepnum,time,X);CHKERRQ(ierr);
    ierr = StackPush(stack,e);CHKERRQ(ierr);
  } else if (store == 2) {
    ierr = DumpSingle(ts,stack,tjsch->rctx->check+1);CHKERRQ(ierr);
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
  StackElement   e;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (stepnum == 0 || stepnum == tjsch->total_steps) {
    ierr = TurnBackward(ts);CHKERRQ(ierr);
    tjsch->rctx->reverseonestep = PETSC_FALSE;
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
    ierr = LoadSingle(ts,stack,tjsch->rctx->check+1);CHKERRQ(ierr);
    ierr = TurnBackward(ts);CHKERRQ(ierr);
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
    PetscPrintf(PETSC_COMM_WORLD,"\x1B[35mSkip the step from %D to %D (stage values already checkpointed)\033[0m\n",tjsch->rctx->oldcapo,tjsch->rctx->oldcapo+1);
    if (!tjsch->rctx->reverseonestep && tjsch->rctx->stepsleft > 0) tjsch->rctx->stepsleft--;
    restart++; /* skip one step */
  }
  if (stack->solution_only || (!stack->solution_only && restart < stepnum)) {
    tjsch->recompute = PETSC_TRUE;
    ierr = TurnForward(ts);CHKERRQ(ierr);
    ierr = ReCompute(ts,tjsch,restart,stepnum);CHKERRQ(ierr);
  }
  if (!ondisk && ( (stack->solution_only && e->stepnum+1 == stepnum) || (!stack->solution_only && e->stepnum == stepnum) )) {
    ierr = StackPop(stack,&e);CHKERRQ(ierr);
    ierr = ElementDestroy(stack,e);CHKERRQ(ierr);
  }
  tjsch->rctx->reverseonestep = PETSC_FALSE;
  PetscFunctionReturn(0);
}
#endif

#undef __FUNCT__
#define __FUNCT__ "TSTrajectorySet_Memory"
static PetscErrorCode TSTrajectorySet_Memory(TSTrajectory tj,TS ts,PetscInt stepnum,PetscReal time,Vec X)
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
    case TWO_LEVEL_TWO_REVOLVE:
      ierr = SetTrajTLTR(ts,tjsch,stepnum,time,X);CHKERRQ(ierr);
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
static PetscErrorCode TSTrajectoryGet_Memory(TSTrajectory tj,TS ts,PetscInt stepnum,PetscReal *t)
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
    case TWO_LEVEL_TWO_REVOLVE:
      ierr = GetTrajTLTR(ts,tjsch,stepnum);CHKERRQ(ierr);
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
PETSC_UNUSED static PetscErrorCode TSTrajectorySetStride_Memory(TSTrajectory tj,TS ts,PetscInt stride)
{
  TJScheduler *tjsch = (TJScheduler*)tj->data;

  PetscFunctionBegin;
  tjsch->stride = stride;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSTrajectorySetMaxCpsRAM_Memory"
PETSC_UNUSED static PetscErrorCode TSTrajectorySetMaxCpsRAM_Memory(TSTrajectory tj,TS ts,PetscInt max_cps_ram)
{
  TJScheduler *tjsch = (TJScheduler*)tj->data;

  PetscFunctionBegin;
  tjsch->max_cps_ram = max_cps_ram;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSTrajectorySetMaxCpsDisk_Memory"
PETSC_UNUSED static PetscErrorCode TSTrajectorySetMaxCpsDisk_Memory(TSTrajectory tj,TS ts,PetscInt max_cps_disk)
{
  TJScheduler *tjsch = (TJScheduler*)tj->data;

  PetscFunctionBegin;
  tjsch->max_cps_disk = max_cps_disk;
  PetscFunctionReturn(0);
}

#ifdef PETSC_HAVE_REVOLVE
#undef __FUNCT__
#define __FUNCT__ "TSTrajectorySetRevolveOnline"
PETSC_UNUSED static PetscErrorCode TSTrajectorySetRevolveOnline(TSTrajectory tj,PetscBool use_online)
{
  TJScheduler *tjsch = (TJScheduler*)tj->data;

  PetscFunctionBegin;
  tjsch->use_online = use_online;
  PetscFunctionReturn(0);
}
#endif

#undef __FUNCT__
#define __FUNCT__ "TSTrajectorySetSaveStack"
PETSC_UNUSED static PetscErrorCode TSTrajectorySetSaveStack(TSTrajectory tj,PetscBool save_stack)
{
  TJScheduler *tjsch = (TJScheduler*)tj->data;

  PetscFunctionBegin;
  tjsch->save_stack = save_stack;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSTrajectorySetSolutionOnly"
PETSC_UNUSED static PetscErrorCode TSTrajectorySetSolutionOnly(TSTrajectory tj,PetscBool solution_only)
{
  TJScheduler *tjsch = (TJScheduler*)tj->data;
  Stack       *stack = &tjsch->stack;

  PetscFunctionBegin;
  stack->solution_only = solution_only;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSTrajectorySetFromOptions_Memory"
static PetscErrorCode TSTrajectorySetFromOptions_Memory(PetscOptionItems *PetscOptionsObject,TSTrajectory tj)
{
  TJScheduler    *tjsch = (TJScheduler*)tj->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"Memory based TS trajectory options");CHKERRQ(ierr);
  {
    ierr = PetscOptionsInt("-ts_trajectory_max_cps_ram","Maximum number of checkpoints in RAM","TSTrajectorySetMaxCpsRAM_Memory",tjsch->max_cps_ram,&tjsch->max_cps_ram,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-ts_trajectory_max_cps_disk","Maximum number of checkpoints on disk","TSTrajectorySetMaxCpsDisk_Memory",tjsch->max_cps_disk,&tjsch->max_cps_disk,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-ts_trajectory_stride","Stride to save checkpoints to file","TSTrajectorySetStride_Memory",tjsch->stride,&tjsch->stride,NULL);CHKERRQ(ierr);
#ifdef PETSC_HAVE_REVOLVE
    ierr = PetscOptionsBool("-ts_trajectory_revolve_online","Trick TS trajectory into using online mode of revolve","TSTrajectorySetRevolveOnline",tjsch->use_online,&tjsch->use_online,NULL);CHKERRQ(ierr);
#endif
    ierr = PetscOptionsBool("-ts_trajectory_save_stack","Save all stack to disk","TSTrajectorySetSaveStack",tjsch->save_stack,&tjsch->save_stack,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-ts_trajectory_solution_only","Checkpoint solution only","TSTrajectorySetSolutionOnly",tjsch->stack.solution_only,&tjsch->stack.solution_only,NULL);CHKERRQ(ierr);
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSTrajectorySetUp_Memory"
static PetscErrorCode TSTrajectorySetUp_Memory(TSTrajectory tj,TS ts)
{
  TJScheduler    *tjsch = (TJScheduler*)tj->data;
  Stack          *stack = &tjsch->stack;
#ifdef PETSC_HAVE_REVOLVE
  RevolveCTX     *rctx,*rctx2;
  DiskStack      *diskstack = &tjsch->diskstack;
  PetscInt       diskblocks;
#endif
  PetscInt       numY;
  PetscBool      flg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscStrcmp(((PetscObject)ts->adapt)->type_name,TSADAPTNONE,&flg);
  if (flg) tjsch->total_steps = PetscMin(ts->max_steps,(PetscInt)(ceil(ts->max_time/ts->time_step))); /* fixed time step */
  if (tjsch->max_cps_ram > 0) stack->stacksize = tjsch->max_cps_ram;

  if (tjsch->stride > 1) { /* two level mode */
    if (tjsch->save_stack && tjsch->max_cps_disk > 1 && tjsch->max_cps_disk <= tjsch->max_cps_ram) SETERRQ(tjsch->comm,PETSC_ERR_ARG_INCOMP,"The specified disk capacity is not enough to store a full stack of RAM checkpoints. You might want to change the disk capacity or use single level checkpointing instead.");
    if (tjsch->max_cps_disk <= 1 && tjsch->max_cps_ram > 1 && tjsch->max_cps_ram <= tjsch->stride-1) tjsch->stype = TWO_LEVEL_REVOLVE; /* use revolve_offline for each stride */
    if (tjsch->max_cps_disk > 1 && tjsch->max_cps_ram > 1 && tjsch->max_cps_ram <= tjsch->stride-1) tjsch->stype = TWO_LEVEL_TWO_REVOLVE;  /* use revolve_offline for each stride */
    if (tjsch->max_cps_disk <= 1 && (tjsch->max_cps_ram >= tjsch->stride || tjsch->max_cps_ram == -1)) tjsch->stype = TWO_LEVEL_NOREVOLVE; /* can also be handled by TWO_LEVEL_REVOLVE */
  } else { /* single level mode */
    if (flg) { /* fixed time step */
      if (tjsch->max_cps_ram >= tjsch->total_steps-1 || tjsch->max_cps_ram < 1) tjsch->stype = NONE; /* checkpoint all */
      else tjsch->stype = (tjsch->max_cps_disk>1) ? REVOLVE_MULTISTAGE : REVOLVE_OFFLINE;
    } else tjsch->stype = NONE; /* checkpoint all for adaptive time step */
#ifdef PETSC_HAVE_REVOLVE
    if (tjsch->use_online) tjsch->stype = REVOLVE_ONLINE; /* trick into online (for testing purpose only) */
#endif
  }

  if (tjsch->stype > TWO_LEVEL_NOREVOLVE) {
#ifndef PETSC_HAVE_REVOLVE
    SETERRQ(tjsch->comm,PETSC_ERR_SUP,"revolve is needed when there is not enough memory to checkpoint all time steps according to the user's settings, please reconfigure with the additional option --download-revolve.");
#else
    switch (tjsch->stype) {
      case TWO_LEVEL_REVOLVE:
        revolve_create_offline(tjsch->stride,tjsch->max_cps_ram);
        break;
      case TWO_LEVEL_TWO_REVOLVE:
        diskblocks = tjsch->save_stack ? tjsch->max_cps_disk/(tjsch->max_cps_ram+1) : tjsch->max_cps_disk; /* The block size depends on whether the stack is saved. */
        diskstack->stacksize = diskblocks;
        revolve_create_offline(tjsch->stride,tjsch->max_cps_ram);
        revolve2_create_offline((tjsch->total_steps+tjsch->stride-1)/tjsch->stride,diskblocks);
        ierr = PetscCalloc1(1,&rctx2);CHKERRQ(ierr);
        rctx2->snaps_in       = diskblocks;
        rctx2->reverseonestep = PETSC_FALSE;
        rctx2->check          = 0;
        rctx2->oldcapo        = 0;
        rctx2->capo           = 0;
        rctx2->info           = 2;
        rctx2->fine           = (tjsch->total_steps+tjsch->stride-1)/tjsch->stride;
        tjsch->rctx2          = rctx2;
        diskstack->top        = -1;
        ierr = PetscMalloc1(diskstack->stacksize*sizeof(PetscInt),&diskstack->container);CHKERRQ(ierr);
        break;
      case REVOLVE_OFFLINE:
        revolve_create_offline(tjsch->total_steps,tjsch->max_cps_ram);
        break;
      case REVOLVE_ONLINE:
        stack->stacksize = tjsch->max_cps_ram;
        revolve_create_online(tjsch->max_cps_ram);
        break;
      case REVOLVE_MULTISTAGE:
        revolve_create_multistage(tjsch->total_steps,tjsch->max_cps_ram+tjsch->max_cps_disk,tjsch->max_cps_ram);
        break;
      default:
        break;
    }
    ierr = PetscCalloc1(1,&rctx);CHKERRQ(ierr);
    rctx->snaps_in       = tjsch->max_cps_ram; /* for theta methods snaps_in=2*max_cps_ram */
    rctx->reverseonestep = PETSC_FALSE;
    rctx->check          = 0;
    rctx->oldcapo        = 0;
    rctx->capo           = 0;
    rctx->info           = 2;
    rctx->fine           = (tjsch->stride > 1) ? tjsch->stride : tjsch->total_steps;
    tjsch->rctx          = rctx;
    if (tjsch->stype == REVOLVE_ONLINE) rctx->fine = -1;
#endif
  } else {
    if (tjsch->stype == TWO_LEVEL_NOREVOLVE) stack->stacksize = tjsch->stride-1; /* need tjsch->stride-1 at most */
    if (tjsch->stype == NONE) {
      if (flg) stack->stacksize = stack->solution_only ? tjsch->total_steps : tjsch->total_steps-1; /* fix time step */
      else { /* adaptive time step */
        if(tjsch->max_cps_ram == -1) stack->stacksize = ts->max_steps; /* if max_cps_ram is not specified, use maximal allowed number of steps for stack size */
        tjsch->total_steps = stack->solution_only ? stack->stacksize:stack->stacksize+1; /* will be updated as time integration advances */
      }
    }
  }

  tjsch->recompute = PETSC_FALSE;
  tjsch->comm      = PetscObjectComm((PetscObject)ts);
  ierr = TSGetStages(ts,&numY,PETSC_IGNORE);CHKERRQ(ierr);
  ierr = StackCreate(stack,stack->stacksize,numY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSTrajectoryDestroy_Memory"
static PetscErrorCode TSTrajectoryDestroy_Memory(TSTrajectory tj)
{
  TJScheduler    *tjsch = (TJScheduler*)tj->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (tjsch->stype > TWO_LEVEL_NOREVOLVE) {
#ifdef PETSC_HAVE_REVOLVE
    revolve_reset();
    if (tjsch->stype == TWO_LEVEL_TWO_REVOLVE) {
      revolve2_reset();
      ierr = PetscFree(tjsch->diskstack.container);CHKERRQ(ierr);
    }
#endif
  }
  ierr = StackDestroy(&tjsch->stack);CHKERRQ(ierr);
#ifdef PETSC_HAVE_REVOLVE
  if (tjsch->stype > TWO_LEVEL_NOREVOLVE) {
    ierr = PetscFree(tjsch->rctx);CHKERRQ(ierr);
    ierr = PetscFree(tjsch->rctx2);CHKERRQ(ierr);
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
