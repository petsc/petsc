#include <petsc/private/tsimpl.h>        /*I "petscts.h"  I*/
#include <petscsys.h>
#if defined(PETSC_HAVE_REVOLVE)
#include <revolve_c.h>

/* Limit Revolve to 32-bits */
#define PETSC_REVOLVE_INT_MAX  2147483647

typedef int PetscRevolveInt;

PETSC_STATIC_INLINE PetscErrorCode PetscRevolveIntCast(PetscInt a,PetscRevolveInt *b)
{
  PetscFunctionBegin;
#if defined(PETSC_USE_64BIT_INDICES)
  *b = 0;
  if ((a) > PETSC_REVOLVE_INT_MAX) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Parameter is too large for Revolve, which is restricted to 32 bit integers");
#endif
  *b = (PetscRevolveInt)(a);
  PetscFunctionReturn(0);
}
#endif

PetscLogEvent TSTrajectory_DiskWrite, TSTrajectory_DiskRead;
static PetscErrorCode TSTrajectorySet_Memory(TSTrajectory,TS,PetscInt,PetscReal,Vec);

typedef enum {NONE,TWO_LEVEL_NOREVOLVE,TWO_LEVEL_REVOLVE,TWO_LEVEL_TWO_REVOLVE,REVOLVE_OFFLINE,REVOLVE_ONLINE,REVOLVE_MULTISTAGE} SchedulerType;

typedef struct _StackElement {
  PetscInt  stepnum;
  Vec       X;
  Vec       *Y;
  PetscReal time;
  PetscReal timeprev; /* for no solution_only mode */
  PetscReal timenext; /* for solution_only mode */
} *StackElement;

#if defined(PETSC_HAVE_REVOLVE)
typedef struct _RevolveCTX {
  PetscBool       reverseonestep;
  PetscRevolveInt where;
  PetscRevolveInt snaps_in;
  PetscRevolveInt stepsleft;
  PetscRevolveInt check;
  PetscRevolveInt oldcapo;
  PetscRevolveInt capo;
  PetscRevolveInt fine;
  PetscRevolveInt info;
} RevolveCTX;
#endif

typedef struct _Stack {
  PetscInt      stacksize;
  PetscInt      top;
  StackElement  *container;
  PetscInt      nallocated;
  PetscInt      numY;
  PetscBool     solution_only;
  PetscBool     use_dram;
} Stack;

typedef struct _DiskStack {
  PetscInt  stacksize;
  PetscInt  top;
  PetscInt  *container;
} DiskStack;

typedef struct _TJScheduler {
  SchedulerType stype;
#if defined(PETSC_HAVE_REVOLVE)
  RevolveCTX    *rctx,*rctx2;
  PetscBool     use_online;
  PetscInt      store_stride;
#endif
  PetscBool     recompute;
  PetscBool     skip_trajectory;
  PetscBool     save_stack;
  PetscInt      max_cps_ram;  /* maximum checkpoints in RAM */
  PetscInt      max_cps_disk; /* maximum checkpoints on disk */
  PetscInt      stride;
  PetscInt      total_steps;  /* total number of steps */
  Stack         stack;
  DiskStack     diskstack;
  PetscViewer   viewer;
} TJScheduler;

static PetscErrorCode TurnForwardWithStepsize(TS ts,PetscReal nextstepsize)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* reverse the direction */
  ierr = TSSetTimeStep(ts,nextstepsize);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

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

static PetscErrorCode TurnBackward(TS ts)
{
  PetscReal      stepsize;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!ts->trajectory->adjoint_solve_mode) PetscFunctionReturn(0);
  /* reverse the direction */
  stepsize = ts->ptime_prev-ts->ptime;
  ierr = TSSetTimeStep(ts,stepsize);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode ElementCreate(TS ts,Stack *stack,StackElement *e)
{
  Vec            X;
  Vec            *Y;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (stack->top < stack->stacksize-1 && stack->container[stack->top+1]) {
    *e = stack->container[stack->top+1];
    PetscFunctionReturn(0);
  }
  if (stack->use_dram) {
    ierr = PetscMallocSetDRAM();CHKERRQ(ierr);
  }
  ierr = PetscNew(e);CHKERRQ(ierr);
  ierr = TSGetSolution(ts,&X);CHKERRQ(ierr);
  ierr = VecDuplicate(X,&(*e)->X);CHKERRQ(ierr);
  if (stack->numY > 0 && !stack->solution_only) {
    ierr = TSGetStages(ts,&stack->numY,&Y);CHKERRQ(ierr);
    ierr = VecDuplicateVecs(Y[0],stack->numY,&(*e)->Y);CHKERRQ(ierr);
  }
  if (stack->use_dram) {
    ierr = PetscMallocResetDRAM();CHKERRQ(ierr);
  }
  stack->nallocated++;
  PetscFunctionReturn(0);
}

static PetscErrorCode ElementSet(TS ts,Stack *stack,StackElement *e,PetscInt stepnum,PetscReal time,Vec X)
{
  Vec            *Y;
  PetscInt       i;
  PetscReal      timeprev;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCopy(X,(*e)->X);CHKERRQ(ierr);
  if (stack->numY > 0 && !stack->solution_only) {
    ierr = TSGetStages(ts,&stack->numY,&Y);CHKERRQ(ierr);
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

static PetscErrorCode ElementDestroy(Stack *stack,StackElement e)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (stack->use_dram) {
    ierr = PetscMallocSetDRAM();CHKERRQ(ierr);
  }
  ierr = VecDestroy(&e->X);CHKERRQ(ierr);
  if (stack->numY > 0 && !stack->solution_only) {
    ierr = VecDestroyVecs(stack->numY,&e->Y);CHKERRQ(ierr);
  }
  ierr = PetscFree(e);CHKERRQ(ierr);
  if (stack->use_dram) {
    ierr = PetscMallocResetDRAM();CHKERRQ(ierr);
  }
  stack->nallocated--;
  PetscFunctionReturn(0);
}

static PetscErrorCode StackResize(Stack *stack,PetscInt newsize)
{
  StackElement   *newcontainer;
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscCalloc1(newsize*sizeof(StackElement),&newcontainer);CHKERRQ(ierr);
  for (i=0;i<stack->stacksize;i++) {
    newcontainer[i] = stack->container[i];
  }
  ierr = PetscFree(stack->container);CHKERRQ(ierr);
  stack->container = newcontainer;
  stack->stacksize = newsize;
  PetscFunctionReturn(0);
}

static PetscErrorCode StackPush(Stack *stack,StackElement e)
{
  PetscFunctionBegin;
  if (stack->top+1 >= stack->stacksize) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_MEMC,"Maximum stack size (%D) exceeded",stack->stacksize);
  stack->container[++stack->top] = e;
  PetscFunctionReturn(0);
}

static PetscErrorCode StackPop(Stack *stack,StackElement *e)
{
  PetscFunctionBegin;
  *e = NULL;
  if (stack->top == -1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEMC,"Empty stack");
  *e = stack->container[stack->top--];
  PetscFunctionReturn(0);
}

static PetscErrorCode StackTop(Stack *stack,StackElement *e)
{
  PetscFunctionBegin;
  *e = stack->container[stack->top];
  PetscFunctionReturn(0);
}

static PetscErrorCode StackInit(Stack *stack,PetscInt size,PetscInt ny)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  stack->top  = -1;
  stack->numY = ny;

  if (!stack->container) {
    ierr = PetscCalloc1(size,&stack->container);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode StackDestroy(Stack *stack)
{
  PetscInt       i,n = stack->nallocated;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!stack->container) PetscFunctionReturn(0);
  if (stack->top+1 > stack->nallocated) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Stack size does not match element counter %D",stack->nallocated);
  for (i=0; i<n; i++) {
    ierr = ElementDestroy(stack,stack->container[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree(stack->container);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode StackFind(Stack *stack,StackElement *e,PetscInt index)
{
  PetscFunctionBegin;
  *e = NULL;
  if (index < 0 || index > stack->top) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Invalid index %D",index);
  *e = stack->container[index];
  PetscFunctionReturn(0);
}

static PetscErrorCode WriteToDisk(PetscInt stepnum,PetscReal time,PetscReal timeprev,Vec X,Vec *Y,PetscInt numY,PetscBool solution_only,PetscViewer viewer)
{
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscViewerBinaryWrite(viewer,&stepnum,1,PETSC_INT);CHKERRQ(ierr);
  ierr = VecView(X,viewer);CHKERRQ(ierr);
  for (i=0;!solution_only && i<numY;i++) {
    ierr = VecView(Y[i],viewer);CHKERRQ(ierr);
  }
  ierr = PetscViewerBinaryWrite(viewer,&time,1,PETSC_REAL);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(viewer,&timeprev,1,PETSC_REAL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

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

static PetscErrorCode StackDumpAll(TSTrajectory tj,TS ts,Stack *stack,PetscInt id)
{
  Vec            *Y;
  PetscInt       i;
  StackElement   e = NULL;
  TJScheduler    *tjsch = (TJScheduler*)tj->data;
  char           filename[PETSC_MAX_PATH_LEN];
  PetscErrorCode ierr;
  MPI_Comm       comm;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)ts,&comm);CHKERRQ(ierr);
  if (tj->monitor) {
    ierr = PetscViewerASCIIPushTab(tj->monitor);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(tj->monitor,"Dump stack id %D to file\n",id);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopTab(tj->monitor);CHKERRQ(ierr);
  }
  ierr = PetscSNPrintf(filename,sizeof(filename),"%s/TS-STACK%06d.bin",tj->dirname,id);CHKERRQ(ierr);
  ierr = PetscViewerFileSetName(tjsch->viewer,filename);CHKERRQ(ierr);
  ierr = PetscViewerSetUp(tjsch->viewer);CHKERRQ(ierr);
  for (i=0;i<stack->stacksize;i++) {
    e = stack->container[i];
    ierr = PetscLogEventBegin(TSTrajectory_DiskWrite,tj,ts,0,0);CHKERRQ(ierr);
    ierr = WriteToDisk(e->stepnum,e->time,e->timeprev,e->X,e->Y,stack->numY,stack->solution_only,tjsch->viewer);CHKERRQ(ierr);
    ierr = PetscLogEventEnd(TSTrajectory_DiskWrite,tj,ts,0,0);CHKERRQ(ierr);
    ts->trajectory->diskwrites++;
  }
  /* save the last step for restart, the last step is in memory when using single level schemes, but not necessarily the case for multi level schemes */
  ierr = TSGetStages(ts,&stack->numY,&Y);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(TSTrajectory_DiskWrite,tj,ts,0,0);CHKERRQ(ierr);
  ierr = WriteToDisk(ts->steps,ts->ptime,ts->ptime_prev,ts->vec_sol,Y,stack->numY,stack->solution_only,tjsch->viewer);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(TSTrajectory_DiskWrite,tj,ts,0,0);CHKERRQ(ierr);
  ts->trajectory->diskwrites++;
  for (i=0;i<stack->stacksize;i++) {
    ierr = StackPop(stack,&e);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode StackLoadAll(TSTrajectory tj,TS ts,Stack *stack,PetscInt id)
{
  Vec            *Y;
  PetscInt       i;
  StackElement   e;
  PetscViewer    viewer;
  char           filename[PETSC_MAX_PATH_LEN];
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (tj->monitor) {
    ierr = PetscViewerASCIIAddTab(tj->monitor,((PetscObject)tj)->tablevel);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(tj->monitor,"Load stack from file\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIISubtractTab(tj->monitor,((PetscObject)tj)->tablevel);CHKERRQ(ierr);
  }
  ierr = PetscSNPrintf(filename,sizeof filename,"%s/TS-STACK%06d.bin",tj->dirname,id);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PetscObjectComm((PetscObject)tj),filename,FILE_MODE_READ,&viewer);CHKERRQ(ierr);
  ierr = PetscViewerBinarySetSkipInfo(viewer,PETSC_TRUE);CHKERRQ(ierr);
  ierr = PetscViewerPushFormat(viewer,PETSC_VIEWER_NATIVE);CHKERRQ(ierr);
  for (i=0;i<stack->stacksize;i++) {
    ierr = ElementCreate(ts,stack,&e);CHKERRQ(ierr);
    ierr = StackPush(stack,e);CHKERRQ(ierr);
    ierr = PetscLogEventBegin(TSTrajectory_DiskRead,tj,ts,0,0);CHKERRQ(ierr);
    ierr = ReadFromDisk(&e->stepnum,&e->time,&e->timeprev,e->X,e->Y,stack->numY,stack->solution_only,viewer);CHKERRQ(ierr);
    ierr = PetscLogEventEnd(TSTrajectory_DiskRead,tj,ts,0,0);CHKERRQ(ierr);
    ts->trajectory->diskreads++;
  }
  /* load the last step into TS */
  ierr = TSGetStages(ts,&stack->numY,&Y);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(TSTrajectory_DiskRead,tj,ts,0,0);CHKERRQ(ierr);
  ierr = ReadFromDisk(&ts->steps,&ts->ptime,&ts->ptime_prev,ts->vec_sol,Y,stack->numY,stack->solution_only,viewer);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(TSTrajectory_DiskRead,tj,ts,0,0);CHKERRQ(ierr);
  ts->trajectory->diskreads++;
  ierr = TurnBackward(ts);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_REVOLVE)
static PetscErrorCode StackLoadLast(TSTrajectory tj,TS ts,Stack *stack,PetscInt id)
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
  if (tj->monitor) {
    ierr = PetscViewerASCIIAddTab(tj->monitor,((PetscObject)tj)->tablevel);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(tj->monitor,"Load last stack element from file\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIISubtractTab(tj->monitor,((PetscObject)tj)->tablevel);CHKERRQ(ierr);
  }
  ierr = TSGetStages(ts,&stack->numY,&Y);CHKERRQ(ierr);
  ierr = VecGetSize(Y[0],&size);CHKERRQ(ierr);
  /* VecView writes to file two extra int's for class id and number of rows */
  off  = -((stack->solution_only?0:stack->numY)+1)*(size*PETSC_BINARY_SCALAR_SIZE+2*PETSC_BINARY_INT_SIZE)-PETSC_BINARY_INT_SIZE-2*PETSC_BINARY_SCALAR_SIZE;

  ierr = PetscSNPrintf(filename,sizeof filename,"%s/TS-STACK%06d.bin",tj->dirname,id);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PetscObjectComm((PetscObject)tj),filename,FILE_MODE_READ,&viewer);CHKERRQ(ierr);
  ierr = PetscViewerBinarySetSkipInfo(viewer,PETSC_TRUE);CHKERRQ(ierr);
  ierr = PetscViewerPushFormat(viewer,PETSC_VIEWER_NATIVE);CHKERRQ(ierr);
#if defined(PETSC_HAVE_MPIIO)
  ierr = PetscViewerBinaryGetUseMPIIO(viewer,&usempiio);CHKERRQ(ierr);
  if (usempiio) {
    ierr = PetscViewerBinaryGetMPIIODescriptor(viewer,(MPI_File*)&fd);CHKERRQ(ierr);
    ierr = PetscBinarySynchronizedSeek(PetscObjectComm((PetscObject)tj),fd,off,PETSC_BINARY_SEEK_END,&offset);CHKERRQ(ierr);
  } else {
#endif
    ierr = PetscViewerBinaryGetDescriptor(viewer,&fd);CHKERRQ(ierr);
    ierr = PetscBinarySeek(fd,off,PETSC_BINARY_SEEK_END,&offset);CHKERRQ(ierr);
#if defined(PETSC_HAVE_MPIIO)
  }
#endif
  /* load the last step into TS */
  ierr = PetscLogEventBegin(TSTrajectory_DiskRead,tj,ts,0,0);CHKERRQ(ierr);
  ierr = ReadFromDisk(&ts->steps,&ts->ptime,&ts->ptime_prev,ts->vec_sol,Y,stack->numY,stack->solution_only,viewer);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(TSTrajectory_DiskRead,tj,ts,0,0);CHKERRQ(ierr);
  ts->trajectory->diskreads++;
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  ierr = TurnBackward(ts);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif

static PetscErrorCode DumpSingle(TSTrajectory tj,TS ts,Stack *stack,PetscInt id)
{
  Vec            *Y;
  PetscInt       stepnum;
  TJScheduler    *tjsch = (TJScheduler*)tj->data;
  char           filename[PETSC_MAX_PATH_LEN];
  PetscErrorCode ierr;
  MPI_Comm       comm;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)ts,&comm);CHKERRQ(ierr);
  if (tj->monitor) {
    ierr = PetscViewerASCIIAddTab(tj->monitor,((PetscObject)tj)->tablevel);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(tj->monitor,"Dump a single point from file\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIISubtractTab(tj->monitor,((PetscObject)tj)->tablevel);CHKERRQ(ierr);
  }
  ierr = TSGetStepNumber(ts,&stepnum);CHKERRQ(ierr);
  ierr = PetscSNPrintf(filename,sizeof(filename),"%s/TS-CPS%06d.bin",tj->dirname,id);CHKERRQ(ierr);
  ierr = PetscViewerFileSetName(tjsch->viewer,filename);CHKERRQ(ierr);
  ierr = PetscViewerSetUp(tjsch->viewer);CHKERRQ(ierr);

  ierr = TSGetStages(ts,&stack->numY,&Y);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(TSTrajectory_DiskWrite,tj,ts,0,0);CHKERRQ(ierr);
  ierr = WriteToDisk(stepnum,ts->ptime,ts->ptime_prev,ts->vec_sol,Y,stack->numY,stack->solution_only,tjsch->viewer);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(TSTrajectory_DiskWrite,tj,ts,0,0);CHKERRQ(ierr);
  ts->trajectory->diskwrites++;
  PetscFunctionReturn(0);
}

static PetscErrorCode LoadSingle(TSTrajectory tj,TS ts,Stack *stack,PetscInt id)
{
  Vec            *Y;
  PetscViewer    viewer;
  char           filename[PETSC_MAX_PATH_LEN];
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (tj->monitor) {
    ierr = PetscViewerASCIIAddTab(tj->monitor,((PetscObject)tj)->tablevel);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(tj->monitor,"Load a single point from file\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIISubtractTab(tj->monitor,((PetscObject)tj)->tablevel);CHKERRQ(ierr);
  }
  ierr = PetscSNPrintf(filename,sizeof filename,"%s/TS-CPS%06d.bin",tj->dirname,id);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PetscObjectComm((PetscObject)tj),filename,FILE_MODE_READ,&viewer);CHKERRQ(ierr);
  ierr = PetscViewerBinarySetSkipInfo(viewer,PETSC_TRUE);CHKERRQ(ierr);
  ierr = PetscViewerPushFormat(viewer,PETSC_VIEWER_NATIVE);CHKERRQ(ierr);
  ierr = TSGetStages(ts,&stack->numY,&Y);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(TSTrajectory_DiskRead,tj,ts,0,0);CHKERRQ(ierr);
  ierr = ReadFromDisk(&ts->steps,&ts->ptime,&ts->ptime_prev,ts->vec_sol,Y,stack->numY,stack->solution_only,viewer);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(TSTrajectory_DiskRead,tj,ts,0,0);CHKERRQ(ierr);
  ts->trajectory->diskreads++;
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode UpdateTS(TS ts,Stack *stack,StackElement e, PetscBool adjoint_mode)
{
  Vec            *Y;
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCopy(e->X,ts->vec_sol);CHKERRQ(ierr);
  if (!stack->solution_only && e->stepnum) {
    ierr = TSGetStages(ts,&stack->numY,&Y);CHKERRQ(ierr);
    for (i=0;i<stack->numY;i++) {
      ierr = VecCopy(e->Y[i],Y[i]);CHKERRQ(ierr);
    }
  }
  if (adjoint_mode) {
    ierr = TSSetTimeStep(ts,e->timeprev-e->time);CHKERRQ(ierr); /* stepsize will be negative */
  } else {
    ierr = TSSetTimeStep(ts,e->time-e->timeprev);CHKERRQ(ierr); /* stepsize will be positive */
  }
  ts->ptime      = e->time;
  ts->ptime_prev = e->timeprev;
  PetscFunctionReturn(0);
}

static PetscErrorCode ReCompute(TS ts,TJScheduler *tjsch,PetscInt stepnumbegin,PetscInt stepnumend)
{
  Stack          *stack = &tjsch->stack;
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  tjsch->recompute = PETSC_TRUE; /* hints TSTrajectorySet() that it is in recompute mode */
  ierr = TSSetStepNumber(ts,stepnumbegin);CHKERRQ(ierr);/* global step number */
  for (i=stepnumbegin;i<stepnumend;i++) { /* assume fixed step size */
    if (stack->solution_only && !tjsch->skip_trajectory) { /* revolve online need this */
      /* don't use the public interface as it will update the TSHistory: this need a better fix */
      ierr = TSTrajectorySet_Memory(ts->trajectory,ts,ts->steps,ts->ptime,ts->vec_sol);CHKERRQ(ierr);
    }
    ierr = TSMonitor(ts,ts->steps,ts->ptime,ts->vec_sol);CHKERRQ(ierr);
    ierr = TSStep(ts);CHKERRQ(ierr);
    if (!stack->solution_only && !tjsch->skip_trajectory) {
      /* don't use the public interface as it will update the TSHistory: this need a better fix */
      ierr = TSTrajectorySet_Memory(ts->trajectory,ts,ts->steps,ts->ptime,ts->vec_sol);CHKERRQ(ierr);
    }
    ierr = TSEventHandler(ts);CHKERRQ(ierr);
    if (!ts->steprollback) {
      ierr = TSPostStep(ts);CHKERRQ(ierr);
    }
  }
  ierr = TurnBackward(ts);CHKERRQ(ierr);
  ts->trajectory->recomps += stepnumend-stepnumbegin; /* recomputation counter */
  ierr = TSSetStepNumber(ts,stepnumend);CHKERRQ(ierr);
  tjsch->recompute = PETSC_FALSE; /* reset the flag for recompute mode */
  PetscFunctionReturn(0);
}

static PetscErrorCode TopLevelStore(TSTrajectory tj,TS ts,TJScheduler *tjsch,PetscInt stepnum,PetscInt localstepnum,PetscInt laststridesize,PetscBool *done)
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
        ierr = StackDumpAll(tj,ts,stack,stridenum+1);CHKERRQ(ierr);
        if (tjsch->stype == TWO_LEVEL_TWO_REVOLVE) diskstack->container[++diskstack->top] = stridenum+1;
        *done = PETSC_TRUE;
      }
    } else {
      if (localstepnum == 0 && stepnum < tjsch->total_steps-laststridesize) {
        ierr = DumpSingle(tj,ts,stack,stridenum+1);CHKERRQ(ierr);
        if (tjsch->stype == TWO_LEVEL_TWO_REVOLVE) diskstack->container[++diskstack->top] = stridenum+1;
        *done = PETSC_TRUE;
      }
    }
  } else {
    if (tjsch->save_stack) {
      if (localstepnum == 0 && stepnum < tjsch->total_steps && stepnum != 0) { /* skip the first stride */
        ierr = StackDumpAll(tj,ts,stack,stridenum);CHKERRQ(ierr);
        if (tjsch->stype == TWO_LEVEL_TWO_REVOLVE) diskstack->container[++diskstack->top] = stridenum;
        *done = PETSC_TRUE;
      }
    } else {
      if (localstepnum == 1 && stepnum < tjsch->total_steps-laststridesize) {
        ierr = DumpSingle(tj,ts,stack,stridenum+1);CHKERRQ(ierr);
        if (tjsch->stype == TWO_LEVEL_TWO_REVOLVE) diskstack->container[++diskstack->top] = stridenum+1;
        *done = PETSC_TRUE;
      }
    }
  }
  PetscFunctionReturn(0);
}

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
  if (stepnum < stack->top) {
    SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_MEMC,"Illegal modification of a non-top stack element");
  }
  ierr = ElementCreate(ts,stack,&e);CHKERRQ(ierr);
  ierr = ElementSet(ts,stack,&e,stepnum,time,X);CHKERRQ(ierr);
  ierr = StackPush(stack,e);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SetTrajN_2(TS ts,TJScheduler *tjsch,PetscInt stepnum,PetscReal time,Vec X)
{
  Stack          *stack = &tjsch->stack;
  StackElement   e;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (stack->top+1 == stack->stacksize) {
    ierr = StackResize(stack,2*stack->stacksize);CHKERRQ(ierr);
  }
  /* update timenext for the previous step; necessary for step adaptivity */
  if (stack->top > -1) {
    ierr = StackTop(stack,&e);CHKERRQ(ierr);
    e->timenext = ts->ptime;
  }
  if (stepnum < stack->top) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_MEMC,"Illegal modification of a non-top stack element");
  ierr = ElementCreate(ts,stack,&e);CHKERRQ(ierr);
  ierr = ElementSet(ts,stack,&e,stepnum,time,X);CHKERRQ(ierr);
  ierr = StackPush(stack,e);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode GetTrajN(TS ts,TJScheduler *tjsch,PetscInt stepnum)
{
  Stack          *stack = &tjsch->stack;
  StackElement   e;
  PetscInt       ns;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* If TSTrajectoryGet() is called after TSAdjointSolve() converges (e.g. outside the while loop in TSAdjointSolve()), skip getting the checkpoint. */
  if (ts->reason) PetscFunctionReturn(0);
  if (stepnum == tjsch->total_steps) {
    ierr = TurnBackward(ts);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  /* restore a checkpoint */
  ierr = StackTop(stack,&e);CHKERRQ(ierr);
  ierr = UpdateTS(ts,stack,e,PETSC_TRUE);CHKERRQ(ierr);
  ierr = TSGetStages(ts,&ns,NULL);CHKERRQ(ierr);
  if (stack->solution_only && ns) { /* recompute one step */
    ierr = TurnForwardWithStepsize(ts,e->timenext-e->time);CHKERRQ(ierr);
    ierr = ReCompute(ts,tjsch,e->stepnum,stepnum);CHKERRQ(ierr);
  }
  ierr = StackPop(stack,&e);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode GetTrajN_2(TS ts,TJScheduler *tjsch,PetscInt stepnum)
{
  Stack          *stack = &tjsch->stack;
  StackElement   e = NULL;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = StackFind(stack,&e,stepnum);CHKERRQ(ierr);
  if (stepnum != e->stepnum) SETERRQ2(PetscObjectComm((PetscObject)ts),PETSC_ERR_PLIB,"Inconsistent steps! %D != %D",stepnum,e->stepnum);
  ierr = UpdateTS(ts,stack,e,PETSC_FALSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SetTrajTLNR(TSTrajectory tj,TS ts,TJScheduler *tjsch,PetscInt stepnum,PetscReal time,Vec X)
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
  /* (stridesize-1) checkpoints are saved in each stride; an extra point is added by StackDumpAll() */
  laststridesize = tjsch->total_steps%tjsch->stride;
  if (!laststridesize) laststridesize = tjsch->stride;

  if (!tjsch->recompute) {
    ierr = TopLevelStore(tj,ts,tjsch,stepnum,localstepnum,laststridesize,&done);CHKERRQ(ierr);
    if (!tjsch->save_stack && stepnum < tjsch->total_steps-laststridesize) PetscFunctionReturn(0);
  }
  if (!stack->solution_only && localstepnum == 0) PetscFunctionReturn(0); /* skip last point in each stride at recompute stage or last stride */
  if (stack->solution_only && localstepnum == tjsch->stride-1) PetscFunctionReturn(0); /* skip last step in each stride at recompute stage or last stride */

  ierr = ElementCreate(ts,stack,&e);CHKERRQ(ierr);
  ierr = ElementSet(ts,stack,&e,stepnum,time,X);CHKERRQ(ierr);
  ierr = StackPush(stack,e);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode GetTrajTLNR(TSTrajectory tj,TS ts,TJScheduler *tjsch,PetscInt stepnum)
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
        ierr = StackLoadAll(tj,ts,stack,id);CHKERRQ(ierr);
        tjsch->skip_trajectory = PETSC_TRUE;
        ierr = TurnForward(ts);CHKERRQ(ierr);
        ierr = ReCompute(ts,tjsch,id*tjsch->stride-1,id*tjsch->stride);CHKERRQ(ierr);
        tjsch->skip_trajectory = PETSC_FALSE;
      } else {
        ierr = LoadSingle(tj,ts,stack,id);CHKERRQ(ierr);
        ierr = TurnForward(ts);CHKERRQ(ierr);
        ierr = ReCompute(ts,tjsch,(id-1)*tjsch->stride,id*tjsch->stride);CHKERRQ(ierr);
      }
      PetscFunctionReturn(0);
    }
    /* restore a checkpoint */
    ierr = StackPop(stack,&e);CHKERRQ(ierr);
    ierr = UpdateTS(ts,stack,e,PETSC_TRUE);CHKERRQ(ierr);
    tjsch->skip_trajectory = PETSC_TRUE;
    ierr = TurnForward(ts);CHKERRQ(ierr);
    ierr = ReCompute(ts,tjsch,e->stepnum,stepnum);CHKERRQ(ierr);
    tjsch->skip_trajectory = PETSC_FALSE;
  } else {
    /* fill stack with info */
    if (localstepnum == 0 && tjsch->total_steps-stepnum >= laststridesize) {
      id = stepnum/tjsch->stride;
      if (tjsch->save_stack) {
        ierr = StackLoadAll(tj,ts,stack,id);CHKERRQ(ierr);
      } else {
        ierr = LoadSingle(tj,ts,stack,id);CHKERRQ(ierr);
        ierr = ElementCreate(ts,stack,&e);CHKERRQ(ierr);
        ierr = ElementSet(ts,stack,&e,(id-1)*tjsch->stride+1,ts->ptime,ts->vec_sol);CHKERRQ(ierr);
        ierr = StackPush(stack,e);CHKERRQ(ierr);
        ierr = TurnForward(ts);CHKERRQ(ierr);
        ierr = ReCompute(ts,tjsch,e->stepnum,id*tjsch->stride);CHKERRQ(ierr);
      }
      PetscFunctionReturn(0);
    }
    /* restore a checkpoint */
    ierr = StackPop(stack,&e);CHKERRQ(ierr);
    ierr = UpdateTS(ts,stack,e,PETSC_TRUE);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_REVOLVE)
static PetscErrorCode printwhattodo(PetscViewer viewer,PetscRevolveInt whattodo,RevolveCTX *rctx,PetscRevolveInt shift)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!viewer) PetscFunctionReturn(0);

  switch(whattodo) {
    case 1:
      ierr = PetscViewerASCIIPrintf(viewer,"Advance from %D to %D\n",rctx->oldcapo+shift,rctx->capo+shift);CHKERRQ(ierr);
      break;
    case 2:
      ierr = PetscViewerASCIIPrintf(viewer,"Store in checkpoint number %D (located in RAM)\n",rctx->check);CHKERRQ(ierr);
      break;
    case 3:
      ierr = PetscViewerASCIIPrintf(viewer,"First turn: Initialize adjoints and reverse first step\n");CHKERRQ(ierr);
      break;
    case 4:
      ierr = PetscViewerASCIIPrintf(viewer,"Forward and reverse one step\n");CHKERRQ(ierr);
      break;
    case 5:
      ierr = PetscViewerASCIIPrintf(viewer,"Restore in checkpoint number %D (located in RAM)\n",rctx->check);CHKERRQ(ierr);
      break;
    case 7:
      ierr = PetscViewerASCIIPrintf(viewer,"Store in checkpoint number %D (located on disk)\n",rctx->check);CHKERRQ(ierr);
      break;
    case 8:
      ierr = PetscViewerASCIIPrintf(viewer,"Restore in checkpoint number %D (located on disk)\n",rctx->check);CHKERRQ(ierr);
      break;
    case -1:
      ierr = PetscViewerASCIIPrintf(viewer,"Error!");CHKERRQ(ierr);
      break;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode printwhattodo2(PetscViewer viewer,PetscRevolveInt whattodo,RevolveCTX *rctx,PetscRevolveInt shift)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!viewer) PetscFunctionReturn(0);

  switch(whattodo) {
    case 1:
      ierr = PetscViewerASCIIPrintf(viewer,"[Top Level] Advance from stride %D to stride %D\n",rctx->oldcapo+shift,rctx->capo+shift);CHKERRQ(ierr);
      break;
    case 2:
      ierr = PetscViewerASCIIPrintf(viewer,"[Top Level] Store in checkpoint number %D\n",rctx->check);CHKERRQ(ierr);
      break;
    case 3:
      ierr = PetscViewerASCIIPrintf(viewer,"[Top Level] First turn: Initialize adjoints and reverse first stride\n");CHKERRQ(ierr);
      break;
    case 4:
      ierr = PetscViewerASCIIPrintf(viewer,"[Top Level] Forward and reverse one stride\n");CHKERRQ(ierr);
      break;
    case 5:
      ierr = PetscViewerASCIIPrintf(viewer,"[Top Level] Restore in checkpoint number %D\n",rctx->check);CHKERRQ(ierr);
      break;
    case 7:
      ierr = PetscViewerASCIIPrintf(viewer,"[Top Level] Store in top-level checkpoint number %D\n",rctx->check);CHKERRQ(ierr);
      break;
    case 8:
      ierr = PetscViewerASCIIPrintf(viewer,"[Top Level] Restore in top-level checkpoint number %D\n",rctx->check);CHKERRQ(ierr);
      break;
    case -1:
      ierr = PetscViewerASCIIPrintf(viewer,"[Top Level] Error!");CHKERRQ(ierr);
      break;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode InitRevolve(PetscInt fine,PetscInt snaps,RevolveCTX *rctx)
{
  PetscRevolveInt rsnaps,rfine;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscRevolveIntCast(snaps,&rsnaps);CHKERRQ(ierr);
  ierr = PetscRevolveIntCast(fine,&rfine);CHKERRQ(ierr);
  revolve_reset();
  revolve_create_offline(rfine,rsnaps);
  rctx->snaps_in       = rsnaps;
  rctx->fine           = rfine;
  rctx->check          = 0;
  rctx->capo           = 0;
  rctx->reverseonestep = PETSC_FALSE;
  /* check stepsleft? */
  PetscFunctionReturn(0);
}

static PetscErrorCode FastForwardRevolve(RevolveCTX *rctx)
{
  PetscRevolveInt whattodo;

  PetscFunctionBegin;
  whattodo = 0;
  while (whattodo!=3) { /* we have to fast forward revolve to the beginning of the backward sweep due to unfriendly revolve interface */
    whattodo = revolve_action(&rctx->check,&rctx->capo,&rctx->fine,rctx->snaps_in,&rctx->info,&rctx->where);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode ApplyRevolve(PetscViewer viewer,SchedulerType stype,RevolveCTX *rctx,PetscRevolveInt total_steps,PetscRevolveInt stepnum,PetscRevolveInt localstepnum,PetscBool toplevel,PetscInt *store)
{
  PetscErrorCode  ierr;
  PetscRevolveInt shift,whattodo;

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
  if (!toplevel) {ierr = printwhattodo(viewer,whattodo,rctx,shift);CHKERRQ(ierr);}
  else {ierr = printwhattodo2(viewer,whattodo,rctx,shift);CHKERRQ(ierr);}
  if (whattodo == -1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in the Revolve library");
  if (whattodo == 1) { /* advance some time steps */
    if (stype == REVOLVE_ONLINE && rctx->capo >= total_steps-1) {
      revolve_turn(total_steps,&rctx->capo,&rctx->fine);
      if (!toplevel) {ierr = printwhattodo(viewer,whattodo,rctx,shift);CHKERRQ(ierr);}
      else {ierr = printwhattodo2(viewer,whattodo,rctx,shift);CHKERRQ(ierr);}
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
    if (!toplevel) {ierr = printwhattodo(viewer,whattodo,rctx,shift);CHKERRQ(ierr);}
    else {ierr = printwhattodo2(viewer,whattodo,rctx,shift);CHKERRQ(ierr);}
    if (whattodo == 3 || whattodo == 4) rctx->reverseonestep = PETSC_TRUE;
    if (whattodo == 1) rctx->stepsleft = rctx->capo-rctx->oldcapo;
  }
  if (whattodo == 7) { /* save the checkpoint to disk */
    *store = 2;
    rctx->oldcapo = rctx->capo;
    whattodo = revolve_action(&rctx->check,&rctx->capo,&rctx->fine,rctx->snaps_in,&rctx->info,&rctx->where); /* must return 1 */
    ierr = printwhattodo(viewer,whattodo,rctx,shift);CHKERRQ(ierr);
    rctx->stepsleft = rctx->capo-rctx->oldcapo-1;
  }
  if (whattodo == 2) { /* store a checkpoint to RAM and ask Revolve how many time steps to advance next */
    *store = 1;
    rctx->oldcapo = rctx->capo;
    if (!toplevel) whattodo = revolve_action(&rctx->check,&rctx->capo,&rctx->fine,rctx->snaps_in,&rctx->info,&rctx->where); /* must return 1 */
    else whattodo = revolve2_action(&rctx->check,&rctx->capo,&rctx->fine,rctx->snaps_in,&rctx->info,&rctx->where);
    if (!toplevel) {ierr = printwhattodo(viewer,whattodo,rctx,shift);CHKERRQ(ierr);}
    else {ierr = printwhattodo2(viewer,whattodo,rctx,shift);CHKERRQ(ierr);}
    if (stype == REVOLVE_ONLINE && rctx->capo >= total_steps-1) {
      revolve_turn(total_steps,&rctx->capo,&rctx->fine);
      ierr = printwhattodo(viewer,whattodo,rctx,shift);CHKERRQ(ierr);
    }
    rctx->stepsleft = rctx->capo-rctx->oldcapo-1;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode SetTrajROF(TSTrajectory tj,TS ts,TJScheduler *tjsch,PetscInt stepnum,PetscReal time,Vec X)
{
  Stack           *stack = &tjsch->stack;
  PetscInt        store;
  StackElement    e;
  PetscRevolveInt rtotal_steps,rstepnum;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  if (!stack->solution_only && stepnum == 0) PetscFunctionReturn(0);
  if (stack->solution_only && stepnum == tjsch->total_steps) PetscFunctionReturn(0);
  ierr = PetscRevolveIntCast(tjsch->total_steps,&rtotal_steps);CHKERRQ(ierr);
  ierr = PetscRevolveIntCast(stepnum,&rstepnum);CHKERRQ(ierr);
  ierr = ApplyRevolve(tj->monitor,tjsch->stype,tjsch->rctx,rtotal_steps,rstepnum,rstepnum,PETSC_FALSE,&store);CHKERRQ(ierr);
  if (store == 1) {
    if (stepnum < stack->top) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_MEMC,"Illegal modification of a non-top stack element");
    ierr = ElementCreate(ts,stack,&e);CHKERRQ(ierr);
    ierr = ElementSet(ts,stack,&e,stepnum,time,X);CHKERRQ(ierr);
    ierr = StackPush(stack,e);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode GetTrajROF(TSTrajectory tj,TS ts,TJScheduler *tjsch,PetscInt stepnum)
{
  Stack           *stack = &tjsch->stack;
  PetscInt        store;
  PetscRevolveInt whattodo,shift,rtotal_steps,rstepnum;
  StackElement    e;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  if (stepnum == 0 || stepnum == tjsch->total_steps) {
    ierr = TurnBackward(ts);CHKERRQ(ierr);
    tjsch->rctx->reverseonestep = PETSC_FALSE;
    PetscFunctionReturn(0);
  }
  /* restore a checkpoint */
  ierr = StackTop(stack,&e);CHKERRQ(ierr);
  ierr = UpdateTS(ts,stack,e,PETSC_TRUE);CHKERRQ(ierr);
  ierr = PetscRevolveIntCast(tjsch->total_steps,&rtotal_steps);CHKERRQ(ierr);
  ierr = PetscRevolveIntCast(stepnum,&rstepnum);CHKERRQ(ierr);
  if (stack->solution_only) { /* start with restoring a checkpoint */
    tjsch->rctx->capo = rstepnum;
    tjsch->rctx->oldcapo = tjsch->rctx->capo;
    shift = 0;
    whattodo = revolve_action(&tjsch->rctx->check,&tjsch->rctx->capo,&tjsch->rctx->fine,tjsch->rctx->snaps_in,&tjsch->rctx->info,&tjsch->rctx->where);
    ierr = printwhattodo(tj->monitor,whattodo,tjsch->rctx,shift);CHKERRQ(ierr);
  } else { /* 2 revolve actions: restore a checkpoint and then advance */
    ierr = ApplyRevolve(tj->monitor,tjsch->stype,tjsch->rctx,rtotal_steps,rstepnum,rstepnum,PETSC_FALSE,&store);CHKERRQ(ierr);
    if (tj->monitor) {
      ierr = PetscViewerASCIIAddTab(tj->monitor,((PetscObject)tj)->tablevel);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(tj->monitor,"Skip the step from %D to %D (stage values already checkpointed)\n",tjsch->rctx->oldcapo,tjsch->rctx->oldcapo+1);CHKERRQ(ierr);
      ierr = PetscViewerASCIISubtractTab(tj->monitor,((PetscObject)tj)->tablevel);CHKERRQ(ierr);
    }
    if (!tjsch->rctx->reverseonestep && tjsch->rctx->stepsleft > 0) tjsch->rctx->stepsleft--;
  }
  if (stack->solution_only || (!stack->solution_only && e->stepnum < stepnum)) {
    ierr = TurnForward(ts);CHKERRQ(ierr);
    ierr = ReCompute(ts,tjsch,e->stepnum,stepnum);CHKERRQ(ierr);
  }
  if ((stack->solution_only && e->stepnum+1 == stepnum) || (!stack->solution_only && e->stepnum == stepnum)) {
    ierr = StackPop(stack,&e);CHKERRQ(ierr);
  }
  tjsch->rctx->reverseonestep = PETSC_FALSE;
  PetscFunctionReturn(0);
}

static PetscErrorCode SetTrajRON(TSTrajectory tj,TS ts,TJScheduler *tjsch,PetscInt stepnum,PetscReal time,Vec X)
{
  Stack           *stack = &tjsch->stack;
  Vec             *Y;
  PetscInt        i,store;
  PetscReal       timeprev;
  StackElement    e;
  RevolveCTX      *rctx = tjsch->rctx;
  PetscRevolveInt rtotal_steps,rstepnum;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  if (!stack->solution_only && stepnum == 0) PetscFunctionReturn(0);
  if (stack->solution_only && stepnum == tjsch->total_steps) PetscFunctionReturn(0);
  ierr = PetscRevolveIntCast(tjsch->total_steps,&rtotal_steps);CHKERRQ(ierr);
  ierr = PetscRevolveIntCast(stepnum,&rstepnum);CHKERRQ(ierr);
  ierr = ApplyRevolve(tj->monitor,tjsch->stype,rctx,rtotal_steps,rstepnum,rstepnum,PETSC_FALSE,&store);CHKERRQ(ierr);
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
      if (stepnum < stack->top) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_MEMC,"Illegal modification of a non-top stack element");
      ierr = ElementCreate(ts,stack,&e);CHKERRQ(ierr);
      ierr = ElementSet(ts,stack,&e,stepnum,time,X);CHKERRQ(ierr);
      ierr = StackPush(stack,e);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode GetTrajRON(TSTrajectory tj,TS ts,TJScheduler *tjsch,PetscInt stepnum)
{
  Stack           *stack = &tjsch->stack;
  PetscRevolveInt whattodo,shift,rstepnum;
  StackElement    e;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  if (stepnum == 0 || stepnum == tjsch->total_steps) {
    ierr = TurnBackward(ts);CHKERRQ(ierr);
    tjsch->rctx->reverseonestep = PETSC_FALSE;
    PetscFunctionReturn(0);
  }
  ierr = PetscRevolveIntCast(stepnum,&rstepnum);CHKERRQ(ierr);
  tjsch->rctx->capo = rstepnum;
  tjsch->rctx->oldcapo = tjsch->rctx->capo;
  shift = 0;
  whattodo = revolve_action(&tjsch->rctx->check,&tjsch->rctx->capo,&tjsch->rctx->fine,tjsch->rctx->snaps_in,&tjsch->rctx->info,&tjsch->rctx->where); /* whattodo=restore */
  if (whattodo == 8) whattodo = 5;
  ierr = printwhattodo(tj->monitor,whattodo,tjsch->rctx,shift);CHKERRQ(ierr);
  /* restore a checkpoint */
  ierr = StackFind(stack,&e,tjsch->rctx->check);CHKERRQ(ierr);
  ierr = UpdateTS(ts,stack,e,PETSC_TRUE);CHKERRQ(ierr);
  if (!stack->solution_only) { /* whattodo must be 5 */
    /* ask Revolve what to do next */
    tjsch->rctx->oldcapo = tjsch->rctx->capo;
    whattodo = revolve_action(&tjsch->rctx->check,&tjsch->rctx->capo,&tjsch->rctx->fine,tjsch->rctx->snaps_in,&tjsch->rctx->info,&tjsch->rctx->where); /* must return 1 or 3 or 4*/
    ierr = printwhattodo(tj->monitor,whattodo,tjsch->rctx,shift);CHKERRQ(ierr);
    if (whattodo == 3 || whattodo == 4) tjsch->rctx->reverseonestep = PETSC_TRUE;
    if (whattodo == 1) tjsch->rctx->stepsleft = tjsch->rctx->capo-tjsch->rctx->oldcapo;
    if (tj->monitor) {
      ierr = PetscViewerASCIIAddTab(tj->monitor,((PetscObject)tj)->tablevel);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(tj->monitor,"Skip the step from %D to %D (stage values already checkpointed)\n",tjsch->rctx->oldcapo,tjsch->rctx->oldcapo+1);CHKERRQ(ierr);
      ierr = PetscViewerASCIISubtractTab(tj->monitor,((PetscObject)tj)->tablevel);CHKERRQ(ierr);
    }
    if (!tjsch->rctx->reverseonestep && tjsch->rctx->stepsleft > 0) tjsch->rctx->stepsleft--;
  }
  if (stack->solution_only || (!stack->solution_only && e->stepnum < stepnum)) {
    ierr = TurnForward(ts);CHKERRQ(ierr);
    ierr = ReCompute(ts,tjsch,e->stepnum,stepnum);CHKERRQ(ierr);
  }
  tjsch->rctx->reverseonestep = PETSC_FALSE;
  PetscFunctionReturn(0);
}

static PetscErrorCode SetTrajTLR(TSTrajectory tj,TS ts,TJScheduler *tjsch,PetscInt stepnum,PetscReal time,Vec X)
{
  Stack           *stack = &tjsch->stack;
  PetscInt        store,localstepnum,laststridesize;
  StackElement    e;
  PetscBool       done = PETSC_FALSE;
  PetscRevolveInt rtotal_steps,rstepnum,rlocalstepnum;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!stack->solution_only && stepnum == 0) PetscFunctionReturn(0);
  if (stack->solution_only && stepnum == tjsch->total_steps) PetscFunctionReturn(0);

  localstepnum = stepnum%tjsch->stride;
  laststridesize = tjsch->total_steps%tjsch->stride;
  if (!laststridesize) laststridesize = tjsch->stride;

  if (!tjsch->recompute) {
    ierr = TopLevelStore(tj,ts,tjsch,stepnum,localstepnum,laststridesize,&done);CHKERRQ(ierr);
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
  ierr = PetscRevolveIntCast(tjsch->total_steps,&rtotal_steps);CHKERRQ(ierr);
  ierr = PetscRevolveIntCast(stepnum,&rstepnum);CHKERRQ(ierr);
  ierr = PetscRevolveIntCast(localstepnum,&rlocalstepnum);CHKERRQ(ierr);
  ierr = ApplyRevolve(tj->monitor,tjsch->stype,tjsch->rctx,rtotal_steps,rstepnum,rlocalstepnum,PETSC_FALSE,&store);CHKERRQ(ierr);
  if (store == 1) {
    if (localstepnum < stack->top) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_MEMC,"Illegal modification of a non-top stack element");
    ierr = ElementCreate(ts,stack,&e);CHKERRQ(ierr);
    ierr = ElementSet(ts,stack,&e,stepnum,time,X);CHKERRQ(ierr);
    ierr = StackPush(stack,e);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode GetTrajTLR(TSTrajectory tj,TS ts,TJScheduler *tjsch,PetscInt stepnum)
{
  Stack           *stack = &tjsch->stack;
  PetscRevolveInt whattodo,shift,rstepnum,rlocalstepnum,rtotal_steps;
  PetscInt        localstepnum,stridenum,laststridesize,store;
  StackElement    e;
  PetscErrorCode  ierr;

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
  ierr = PetscRevolveIntCast(tjsch->total_steps,&rtotal_steps);CHKERRQ(ierr);
  ierr = PetscRevolveIntCast(stepnum,&rstepnum);CHKERRQ(ierr);
  ierr = PetscRevolveIntCast(localstepnum,&rlocalstepnum);CHKERRQ(ierr);
  if (stack->solution_only) {
    /* fill stack */
    if (localstepnum == 0 && stepnum <= tjsch->total_steps-laststridesize) {
      if (tjsch->save_stack) {
        ierr = StackLoadAll(tj,ts,stack,stridenum);CHKERRQ(ierr);
        ierr = InitRevolve(tjsch->stride,tjsch->max_cps_ram,tjsch->rctx);CHKERRQ(ierr);
        ierr = FastForwardRevolve(tjsch->rctx);CHKERRQ(ierr);
        tjsch->skip_trajectory = PETSC_TRUE;
        ierr = TurnForward(ts);CHKERRQ(ierr);
        ierr = ReCompute(ts,tjsch,stridenum*tjsch->stride-1,stridenum*tjsch->stride);CHKERRQ(ierr);
        tjsch->skip_trajectory = PETSC_FALSE;
      } else {
        ierr = LoadSingle(tj,ts,stack,stridenum);CHKERRQ(ierr);
        ierr = InitRevolve(tjsch->stride,tjsch->max_cps_ram,tjsch->rctx);CHKERRQ(ierr);
        ierr = TurnForward(ts);CHKERRQ(ierr);
        ierr = ReCompute(ts,tjsch,(stridenum-1)*tjsch->stride,stridenum*tjsch->stride);CHKERRQ(ierr);
      }
      PetscFunctionReturn(0);
    }
    /* restore a checkpoint */
    ierr = StackTop(stack,&e);CHKERRQ(ierr);
    ierr = UpdateTS(ts,stack,e,PETSC_TRUE);CHKERRQ(ierr);
    /* start with restoring a checkpoint */
    tjsch->rctx->capo = rstepnum;
    tjsch->rctx->oldcapo = tjsch->rctx->capo;
    shift = rstepnum-rlocalstepnum;
    whattodo = revolve_action(&tjsch->rctx->check,&tjsch->rctx->capo,&tjsch->rctx->fine,tjsch->rctx->snaps_in,&tjsch->rctx->info,&tjsch->rctx->where);
    ierr = printwhattodo(tj->monitor,whattodo,tjsch->rctx,shift);CHKERRQ(ierr);
    ierr = TurnForward(ts);CHKERRQ(ierr);
    ierr = ReCompute(ts,tjsch,e->stepnum,stepnum);CHKERRQ(ierr);
    if (e->stepnum+1 == stepnum) {
      ierr = StackPop(stack,&e);CHKERRQ(ierr);
    }
  } else {
    /* fill stack with info */
    if (localstepnum == 0 && tjsch->total_steps-stepnum >= laststridesize) {
      if (tjsch->save_stack) {
        ierr = StackLoadAll(tj,ts,stack,stridenum);CHKERRQ(ierr);
        ierr = InitRevolve(tjsch->stride,tjsch->max_cps_ram,tjsch->rctx);CHKERRQ(ierr);
        ierr = FastForwardRevolve(tjsch->rctx);CHKERRQ(ierr);
      } else {
        PetscRevolveInt rnum;
        ierr = LoadSingle(tj,ts,stack,stridenum);CHKERRQ(ierr);
        ierr = InitRevolve(tjsch->stride,tjsch->max_cps_ram,tjsch->rctx);CHKERRQ(ierr);
        ierr = PetscRevolveIntCast((stridenum-1)*tjsch->stride+1,&rnum);CHKERRQ(ierr);
        ierr = ApplyRevolve(tj->monitor,tjsch->stype,tjsch->rctx,rtotal_steps,rnum,1,PETSC_FALSE,&store);CHKERRQ(ierr);
        if (tj->monitor) {
          ierr = PetscViewerASCIIAddTab(tj->monitor,((PetscObject)tj)->tablevel);CHKERRQ(ierr);
          ierr = PetscViewerASCIIPrintf(tj->monitor,"Skip the step from %D to %D (stage values already checkpointed)\n",(stridenum-1)*tjsch->stride+tjsch->rctx->oldcapo,(stridenum-1)*tjsch->stride+tjsch->rctx->oldcapo+1);CHKERRQ(ierr);
          ierr = PetscViewerASCIISubtractTab(tj->monitor,((PetscObject)tj)->tablevel);CHKERRQ(ierr);
        }
        ierr = ElementCreate(ts,stack,&e);CHKERRQ(ierr);
        ierr = ElementSet(ts,stack,&e,(stridenum-1)*tjsch->stride+1,ts->ptime,ts->vec_sol);CHKERRQ(ierr);
        ierr = StackPush(stack,e);CHKERRQ(ierr);
        ierr = TurnForward(ts);CHKERRQ(ierr);
        ierr = ReCompute(ts,tjsch,e->stepnum,stridenum*tjsch->stride);CHKERRQ(ierr);
      }
      PetscFunctionReturn(0);
    }
    /* restore a checkpoint */
    ierr = StackTop(stack,&e);CHKERRQ(ierr);
    ierr = UpdateTS(ts,stack,e,PETSC_TRUE);CHKERRQ(ierr);
    /* 2 revolve actions: restore a checkpoint and then advance */
    ierr = ApplyRevolve(tj->monitor,tjsch->stype,tjsch->rctx,rtotal_steps,rstepnum,rlocalstepnum,PETSC_FALSE,&store);CHKERRQ(ierr);
    if (tj->monitor) {
      ierr = PetscViewerASCIIAddTab(tj->monitor,((PetscObject)tj)->tablevel);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(tj->monitor,"Skip the step from %D to %D (stage values already checkpointed)\n",stepnum-localstepnum+tjsch->rctx->oldcapo,stepnum-localstepnum+tjsch->rctx->oldcapo+1);CHKERRQ(ierr);
      ierr = PetscViewerASCIISubtractTab(tj->monitor,((PetscObject)tj)->tablevel);CHKERRQ(ierr);
    }
    if (!tjsch->rctx->reverseonestep && tjsch->rctx->stepsleft > 0) tjsch->rctx->stepsleft--;
    if (e->stepnum < stepnum) {
      ierr = TurnForward(ts);CHKERRQ(ierr);
      ierr = ReCompute(ts,tjsch,e->stepnum,stepnum);CHKERRQ(ierr);
    }
    if (e->stepnum == stepnum) {
      ierr = StackPop(stack,&e);CHKERRQ(ierr);
    }
  }
  tjsch->rctx->reverseonestep = PETSC_FALSE;
  PetscFunctionReturn(0);
}

static PetscErrorCode SetTrajTLTR(TSTrajectory tj,TS ts,TJScheduler *tjsch,PetscInt stepnum,PetscReal time,Vec X)
{
  Stack           *stack = &tjsch->stack;
  PetscInt        store,localstepnum,stridenum,laststridesize;
  StackElement    e;
  PetscBool       done = PETSC_FALSE;
  PetscRevolveInt rlocalstepnum,rstepnum,rtotal_steps;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  if (!stack->solution_only && stepnum == 0) PetscFunctionReturn(0);
  if (stack->solution_only && stepnum == tjsch->total_steps) PetscFunctionReturn(0);

  localstepnum = stepnum%tjsch->stride; /* index at the bottom level (inside a stride) */
  stridenum    = stepnum/tjsch->stride; /* index at the top level */
  laststridesize = tjsch->total_steps%tjsch->stride;
  if (!laststridesize) laststridesize = tjsch->stride;
  if (stack->solution_only && localstepnum == 0 && !tjsch->rctx2->reverseonestep) {
    ierr = PetscRevolveIntCast((tjsch->total_steps+tjsch->stride-1)/tjsch->stride,&rtotal_steps);CHKERRQ(ierr);
    ierr = PetscRevolveIntCast(stridenum,&rstepnum);CHKERRQ(ierr);
    ierr = ApplyRevolve(tj->monitor,tjsch->stype,tjsch->rctx2,rtotal_steps,rstepnum,rstepnum,PETSC_TRUE,&tjsch->store_stride);CHKERRQ(ierr);
    if (laststridesize < tjsch->stride && stepnum == tjsch->total_steps-laststridesize) {
      ierr = InitRevolve(laststridesize,tjsch->max_cps_ram,tjsch->rctx);CHKERRQ(ierr);
    }
  }
  if (!stack->solution_only && localstepnum == 1 && !tjsch->rctx2->reverseonestep) {
    ierr = PetscRevolveIntCast((tjsch->total_steps+tjsch->stride-1)/tjsch->stride,&rtotal_steps);CHKERRQ(ierr);
    ierr = PetscRevolveIntCast(stridenum,&rstepnum);CHKERRQ(ierr);
    ierr = ApplyRevolve(tj->monitor,tjsch->stype,tjsch->rctx2,rtotal_steps,rstepnum,rstepnum,PETSC_TRUE,&tjsch->store_stride);CHKERRQ(ierr);
    if (laststridesize < tjsch->stride && stepnum == tjsch->total_steps-laststridesize+1) {
      ierr = InitRevolve(laststridesize,tjsch->max_cps_ram,tjsch->rctx);CHKERRQ(ierr);
    }
  }
  if (tjsch->store_stride) {
    ierr = TopLevelStore(tj,ts,tjsch,stepnum,localstepnum,laststridesize,&done);CHKERRQ(ierr);
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
  ierr = PetscRevolveIntCast(tjsch->total_steps,&rtotal_steps);CHKERRQ(ierr);
  ierr = PetscRevolveIntCast(stepnum,&rstepnum);CHKERRQ(ierr);
  ierr = PetscRevolveIntCast(localstepnum,&rlocalstepnum);CHKERRQ(ierr);
  ierr = ApplyRevolve(tj->monitor,tjsch->stype,tjsch->rctx,rtotal_steps,rstepnum,rlocalstepnum,PETSC_FALSE,&store);CHKERRQ(ierr);
  if (store == 1) {
    if (localstepnum < stack->top) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_MEMC,"Illegal modification of a non-top stack element");
    ierr = ElementCreate(ts,stack,&e);CHKERRQ(ierr);
    ierr = ElementSet(ts,stack,&e,stepnum,time,X);CHKERRQ(ierr);
    ierr = StackPush(stack,e);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode GetTrajTLTR(TSTrajectory tj,TS ts,TJScheduler *tjsch,PetscInt stepnum)
{
  Stack           *stack = &tjsch->stack;
  DiskStack       *diskstack = &tjsch->diskstack;
  PetscInt        localstepnum,stridenum,restoredstridenum,laststridesize,store;
  StackElement    e;
  PetscRevolveInt whattodo,shift;
  PetscRevolveInt rtotal_steps,rstepnum,rlocalstepnum;
  PetscErrorCode  ierr;

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
      ierr = PetscRevolveIntCast(stridenum,&rstepnum);CHKERRQ(ierr);
      tjsch->rctx2->capo = rstepnum;
      tjsch->rctx2->oldcapo = tjsch->rctx2->capo;
      shift = 0;
      whattodo = revolve2_action(&tjsch->rctx2->check,&tjsch->rctx2->capo,&tjsch->rctx2->fine,tjsch->rctx2->snaps_in,&tjsch->rctx2->info,&tjsch->rctx2->where);
      ierr = printwhattodo2(tj->monitor,whattodo,tjsch->rctx2,shift);CHKERRQ(ierr);
    } else { /* 2 revolve actions: restore a checkpoint and then advance */
      ierr = PetscRevolveIntCast((tjsch->total_steps+tjsch->stride-1)/tjsch->stride,&rtotal_steps);CHKERRQ(ierr);
      ierr = PetscRevolveIntCast(stridenum,&rstepnum);CHKERRQ(ierr);
      ierr = ApplyRevolve(tj->monitor,tjsch->stype,tjsch->rctx2,rtotal_steps,rstepnum,rstepnum,PETSC_TRUE,&tjsch->store_stride);CHKERRQ(ierr);
      if (tj->monitor) {
        ierr = PetscViewerASCIIAddTab(tj->monitor,((PetscObject)tj)->tablevel);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(tj->monitor,"[Top Level] Skip the stride from %D to %D (stage values already checkpointed)\n",tjsch->rctx2->oldcapo,tjsch->rctx2->oldcapo+1);CHKERRQ(ierr);
        ierr = PetscViewerASCIISubtractTab(tj->monitor,((PetscObject)tj)->tablevel);CHKERRQ(ierr);
      }
      if (!tjsch->rctx2->reverseonestep && tjsch->rctx2->stepsleft > 0) tjsch->rctx2->stepsleft--;
    }
    /* fill stack */
    if (stack->solution_only) {
      if (tjsch->save_stack) {
        if (restoredstridenum < stridenum) {
          ierr = StackLoadLast(tj,ts,stack,restoredstridenum);CHKERRQ(ierr);
        } else {
          ierr = StackLoadAll(tj,ts,stack,restoredstridenum);CHKERRQ(ierr);
        }
        /* recompute one step ahead */
        tjsch->skip_trajectory = PETSC_TRUE;
        ierr = TurnForward(ts);CHKERRQ(ierr);
        ierr = ReCompute(ts,tjsch,stridenum*tjsch->stride-1,stridenum*tjsch->stride);CHKERRQ(ierr);
        tjsch->skip_trajectory = PETSC_FALSE;
        if (restoredstridenum < stridenum) {
          ierr = InitRevolve(tjsch->stride,tjsch->max_cps_ram,tjsch->rctx);CHKERRQ(ierr);
          ierr = TurnForward(ts);CHKERRQ(ierr);
          ierr = ReCompute(ts,tjsch,restoredstridenum*tjsch->stride,stepnum);CHKERRQ(ierr);
        } else { /* stack ready, fast forward revolve status */
          ierr = InitRevolve(tjsch->stride,tjsch->max_cps_ram,tjsch->rctx);CHKERRQ(ierr);
          ierr = FastForwardRevolve(tjsch->rctx);CHKERRQ(ierr);
        }
      } else {
        ierr = LoadSingle(tj,ts,stack,restoredstridenum);CHKERRQ(ierr);
        ierr = InitRevolve(tjsch->stride,tjsch->max_cps_ram,tjsch->rctx);CHKERRQ(ierr);
        ierr = TurnForward(ts);CHKERRQ(ierr);
        ierr = ReCompute(ts,tjsch,(restoredstridenum-1)*tjsch->stride,stepnum);CHKERRQ(ierr);
      }
    } else {
      if (tjsch->save_stack) {
        if (restoredstridenum < stridenum) {
          ierr = StackLoadLast(tj,ts,stack,restoredstridenum);CHKERRQ(ierr);
          /* reset revolve */
          ierr = InitRevolve(tjsch->stride,tjsch->max_cps_ram,tjsch->rctx);CHKERRQ(ierr);
          ierr = TurnForward(ts);CHKERRQ(ierr);
          ierr = ReCompute(ts,tjsch,restoredstridenum*tjsch->stride,stepnum);CHKERRQ(ierr);
        } else { /* stack ready, fast forward revolve status */
          ierr = StackLoadAll(tj,ts,stack,restoredstridenum);CHKERRQ(ierr);
          ierr = InitRevolve(tjsch->stride,tjsch->max_cps_ram,tjsch->rctx);CHKERRQ(ierr);
          ierr = FastForwardRevolve(tjsch->rctx);CHKERRQ(ierr);
        }
      } else {
        ierr = LoadSingle(tj,ts,stack,restoredstridenum);CHKERRQ(ierr);
        ierr = InitRevolve(tjsch->stride,tjsch->max_cps_ram,tjsch->rctx);CHKERRQ(ierr);
        /* push first element to stack */
        if (tjsch->store_stride || tjsch->rctx2->reverseonestep) {
          ierr = PetscRevolveIntCast(tjsch->total_steps,&rtotal_steps);CHKERRQ(ierr);
          ierr = PetscRevolveIntCast((restoredstridenum-1)*tjsch->stride+1,&rstepnum);CHKERRQ(ierr);
          ierr = ApplyRevolve(tj->monitor,tjsch->stype,tjsch->rctx,rtotal_steps,rstepnum,1,PETSC_FALSE,&store);CHKERRQ(ierr);
          if (tj->monitor) {
            ierr = PetscViewerASCIIAddTab(tj->monitor,((PetscObject)tj)->tablevel);CHKERRQ(ierr);
            ierr = PetscViewerASCIIPrintf(tj->monitor,"Skip the step from %D to %D (stage values already checkpointed)\n",(restoredstridenum-1)*tjsch->stride,(restoredstridenum-1)*tjsch->stride+1);CHKERRQ(ierr);
            ierr = PetscViewerASCIISubtractTab(tj->monitor,((PetscObject)tj)->tablevel);CHKERRQ(ierr);
          }
          ierr = ElementCreate(ts,stack,&e);CHKERRQ(ierr);
          ierr = ElementSet(ts,stack,&e,(restoredstridenum-1)*tjsch->stride+1,ts->ptime,ts->vec_sol);CHKERRQ(ierr);
          ierr = StackPush(stack,e);CHKERRQ(ierr);
        }
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
    ierr = UpdateTS(ts,stack,e,PETSC_TRUE);CHKERRQ(ierr);
    /* start with restoring a checkpoint */
    ierr = PetscRevolveIntCast(stepnum,&rstepnum);CHKERRQ(ierr);
    ierr = PetscRevolveIntCast(localstepnum,&rlocalstepnum);CHKERRQ(ierr);
    tjsch->rctx->capo = rstepnum;
    tjsch->rctx->oldcapo = tjsch->rctx->capo;
    shift = rstepnum-rlocalstepnum;
    whattodo = revolve_action(&tjsch->rctx->check,&tjsch->rctx->capo,&tjsch->rctx->fine,tjsch->rctx->snaps_in,&tjsch->rctx->info,&tjsch->rctx->where);
    ierr = printwhattodo(tj->monitor,whattodo,tjsch->rctx,shift);CHKERRQ(ierr);
    ierr = TurnForward(ts);CHKERRQ(ierr);
    ierr = ReCompute(ts,tjsch,e->stepnum,stepnum);CHKERRQ(ierr);
    if (e->stepnum+1 == stepnum) {
      ierr = StackPop(stack,&e);CHKERRQ(ierr);
    }
  } else {
    PetscRevolveInt rlocalstepnum;
    /* restore a checkpoint */
    ierr = StackTop(stack,&e);CHKERRQ(ierr);
    ierr = UpdateTS(ts,stack,e,PETSC_TRUE);CHKERRQ(ierr);
    /* 2 revolve actions: restore a checkpoint and then advance */
    ierr = PetscRevolveIntCast(tjsch->total_steps,&rtotal_steps);CHKERRQ(ierr);
    ierr = PetscRevolveIntCast(stridenum,&rstepnum);CHKERRQ(ierr);
    ierr = PetscRevolveIntCast(localstepnum,&rlocalstepnum);CHKERRQ(ierr);
    ierr = ApplyRevolve(tj->monitor,tjsch->stype,tjsch->rctx,rtotal_steps,rstepnum,rlocalstepnum,PETSC_FALSE,&store);CHKERRQ(ierr);
    if (tj->monitor) {
      ierr = PetscViewerASCIIAddTab(tj->monitor,((PetscObject)tj)->tablevel);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(tj->monitor,"Skip the step from %D to %D (stage values already checkpointed)\n",stepnum-localstepnum+tjsch->rctx->oldcapo,stepnum-localstepnum+tjsch->rctx->oldcapo+1);CHKERRQ(ierr);
      ierr = PetscViewerASCIISubtractTab(tj->monitor,((PetscObject)tj)->tablevel);CHKERRQ(ierr);
    }
    if (!tjsch->rctx->reverseonestep && tjsch->rctx->stepsleft > 0) tjsch->rctx->stepsleft--;
    if (e->stepnum < stepnum) {
      ierr = TurnForward(ts);CHKERRQ(ierr);
      ierr = ReCompute(ts,tjsch,e->stepnum,stepnum);CHKERRQ(ierr);
    }
    if (e->stepnum == stepnum) {
      ierr = StackPop(stack,&e);CHKERRQ(ierr);
    }
  }
  tjsch->rctx->reverseonestep = PETSC_FALSE;
  PetscFunctionReturn(0);
}

static PetscErrorCode SetTrajRMS(TSTrajectory tj,TS ts,TJScheduler *tjsch,PetscInt stepnum,PetscReal time,Vec X)
{
  Stack           *stack = &tjsch->stack;
  PetscInt        store;
  StackElement    e;
  PetscRevolveInt rtotal_steps,rstepnum;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  if (!stack->solution_only && stepnum == 0) PetscFunctionReturn(0);
  if (stack->solution_only && stepnum == tjsch->total_steps) PetscFunctionReturn(0);
  ierr = PetscRevolveIntCast(tjsch->total_steps,&rtotal_steps);CHKERRQ(ierr);
  ierr = PetscRevolveIntCast(stepnum,&rstepnum);CHKERRQ(ierr);
  ierr = ApplyRevolve(tj->monitor,tjsch->stype,tjsch->rctx,rtotal_steps,rstepnum,rstepnum,PETSC_FALSE,&store);CHKERRQ(ierr);
  if (store == 1) {
    if (stepnum < stack->top) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_MEMC,"Illegal modification of a non-top stack element");
    ierr = ElementCreate(ts,stack,&e);CHKERRQ(ierr);
    ierr = ElementSet(ts,stack,&e,stepnum,time,X);CHKERRQ(ierr);
    ierr = StackPush(stack,e);CHKERRQ(ierr);
  } else if (store == 2) {
    ierr = DumpSingle(tj,ts,stack,tjsch->rctx->check+1);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode GetTrajRMS(TSTrajectory tj,TS ts,TJScheduler *tjsch,PetscInt stepnum)
{
  Stack           *stack = &tjsch->stack;
  PetscRevolveInt whattodo,shift,rstepnum;
  PetscInt        restart;
  PetscBool       ondisk;
  StackElement    e;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  if (stepnum == 0 || stepnum == tjsch->total_steps) {
    ierr = TurnBackward(ts);CHKERRQ(ierr);
    tjsch->rctx->reverseonestep = PETSC_FALSE;
    PetscFunctionReturn(0);
  }
  ierr = PetscRevolveIntCast(stepnum,&rstepnum);CHKERRQ(ierr);
  tjsch->rctx->capo = rstepnum;
  tjsch->rctx->oldcapo = tjsch->rctx->capo;
  shift = 0;
  whattodo = revolve_action(&tjsch->rctx->check,&tjsch->rctx->capo,&tjsch->rctx->fine,tjsch->rctx->snaps_in,&tjsch->rctx->info,&tjsch->rctx->where); /* whattodo=restore */
  ierr = printwhattodo(tj->monitor,whattodo,tjsch->rctx,shift);CHKERRQ(ierr);
  /* restore a checkpoint */
  restart = tjsch->rctx->capo;
  if (!tjsch->rctx->where) {
    ondisk = PETSC_TRUE;
    ierr = LoadSingle(tj,ts,stack,tjsch->rctx->check+1);CHKERRQ(ierr);
    ierr = TurnBackward(ts);CHKERRQ(ierr);
  } else {
    ondisk = PETSC_FALSE;
    ierr = StackTop(stack,&e);CHKERRQ(ierr);
    ierr = UpdateTS(ts,stack,e,PETSC_TRUE);CHKERRQ(ierr);
  }
  if (!stack->solution_only) { /* whattodo must be 5 or 8 */
    /* ask Revolve what to do next */
    tjsch->rctx->oldcapo = tjsch->rctx->capo;
    whattodo = revolve_action(&tjsch->rctx->check,&tjsch->rctx->capo,&tjsch->rctx->fine,tjsch->rctx->snaps_in,&tjsch->rctx->info,&tjsch->rctx->where); /* must return 1 or 3 or 4*/
    ierr = printwhattodo(tj->monitor,whattodo,tjsch->rctx,shift);CHKERRQ(ierr);
    if (whattodo == 3 || whattodo == 4) tjsch->rctx->reverseonestep = PETSC_TRUE;
    if (whattodo == 1) tjsch->rctx->stepsleft = tjsch->rctx->capo-tjsch->rctx->oldcapo;
    if (tj->monitor) {
      ierr = PetscViewerASCIIAddTab(tj->monitor,((PetscObject)tj)->tablevel);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(tj->monitor,"Skip the step from %D to %D (stage values already checkpointed)\n",tjsch->rctx->oldcapo,tjsch->rctx->oldcapo+1);CHKERRQ(ierr);
      ierr = PetscViewerASCIISubtractTab(tj->monitor,((PetscObject)tj)->tablevel);CHKERRQ(ierr);
    }
    if (!tjsch->rctx->reverseonestep && tjsch->rctx->stepsleft > 0) tjsch->rctx->stepsleft--;
    restart++; /* skip one step */
  }
  if (stack->solution_only || (!stack->solution_only && restart < stepnum)) {
    ierr = TurnForward(ts);CHKERRQ(ierr);
    ierr = ReCompute(ts,tjsch,restart,stepnum);CHKERRQ(ierr);
  }
  if (!ondisk && ( (stack->solution_only && e->stepnum+1 == stepnum) || (!stack->solution_only && e->stepnum == stepnum))) {
    ierr = StackPop(stack,&e);CHKERRQ(ierr);
  }
  tjsch->rctx->reverseonestep = PETSC_FALSE;
  PetscFunctionReturn(0);
}
#endif

static PetscErrorCode TSTrajectorySet_Memory(TSTrajectory tj,TS ts,PetscInt stepnum,PetscReal time,Vec X)
{
  TJScheduler *tjsch = (TJScheduler*)tj->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!tjsch->recompute) { /* use global stepnum in the forward sweep */
    ierr = TSGetStepNumber(ts,&stepnum);CHKERRQ(ierr);
  }
  /* for consistency */
  if (!tjsch->recompute && stepnum == 0) ts->ptime_prev = ts->ptime-ts->time_step;
  switch (tjsch->stype) {
    case NONE:
      if (tj->adjoint_solve_mode) {
        ierr = SetTrajN(ts,tjsch,stepnum,time,X);CHKERRQ(ierr);
      } else {
        ierr = SetTrajN_2(ts,tjsch,stepnum,time,X);CHKERRQ(ierr);
      }
      break;
    case TWO_LEVEL_NOREVOLVE:
      if (!tj->adjoint_solve_mode) SETERRQ(PetscObjectComm((PetscObject)tj),PETSC_ERR_SUP,"Not implemented");
      ierr = SetTrajTLNR(tj,ts,tjsch,stepnum,time,X);CHKERRQ(ierr);
      break;
#if defined(PETSC_HAVE_REVOLVE)
    case TWO_LEVEL_REVOLVE:
      if (!tj->adjoint_solve_mode) SETERRQ(PetscObjectComm((PetscObject)tj),PETSC_ERR_SUP,"Not implemented");
      ierr = SetTrajTLR(tj,ts,tjsch,stepnum,time,X);CHKERRQ(ierr);
      break;
    case TWO_LEVEL_TWO_REVOLVE:
      if (!tj->adjoint_solve_mode) SETERRQ(PetscObjectComm((PetscObject)tj),PETSC_ERR_SUP,"Not implemented");
      ierr = SetTrajTLTR(tj,ts,tjsch,stepnum,time,X);CHKERRQ(ierr);
      break;
    case REVOLVE_OFFLINE:
      if (!tj->adjoint_solve_mode) SETERRQ(PetscObjectComm((PetscObject)tj),PETSC_ERR_SUP,"Not implemented");
      ierr = SetTrajROF(tj,ts,tjsch,stepnum,time,X);CHKERRQ(ierr);
      break;
    case REVOLVE_ONLINE:
      if (!tj->adjoint_solve_mode) SETERRQ(PetscObjectComm((PetscObject)tj),PETSC_ERR_SUP,"Not implemented");
      ierr = SetTrajRON(tj,ts,tjsch,stepnum,time,X);CHKERRQ(ierr);
      break;
    case REVOLVE_MULTISTAGE:
      if (!tj->adjoint_solve_mode) SETERRQ(PetscObjectComm((PetscObject)tj),PETSC_ERR_SUP,"Not implemented");
      ierr = SetTrajRMS(tj,ts,tjsch,stepnum,time,X);CHKERRQ(ierr);
      break;
#endif
    default:
      break;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TSTrajectoryGet_Memory(TSTrajectory tj,TS ts,PetscInt stepnum,PetscReal *t)
{
  TJScheduler *tjsch = (TJScheduler*)tj->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (tj->adjoint_solve_mode && stepnum == 0) {
    ierr = TSTrajectoryReset(tj);CHKERRQ(ierr); /* reset TSTrajectory so users do not need to reset TSTrajectory */
    PetscFunctionReturn(0);
  }
  switch (tjsch->stype) {
    case NONE:
      if (tj->adjoint_solve_mode) {
        ierr = GetTrajN(ts,tjsch,stepnum);CHKERRQ(ierr);
      } else {
        ierr = GetTrajN_2(ts,tjsch,stepnum);CHKERRQ(ierr);
      }
      break;
    case TWO_LEVEL_NOREVOLVE:
      if (!tj->adjoint_solve_mode) SETERRQ(PetscObjectComm((PetscObject)tj),PETSC_ERR_SUP,"Not implemented");
      ierr = GetTrajTLNR(tj,ts,tjsch,stepnum);CHKERRQ(ierr);
      break;
#if defined(PETSC_HAVE_REVOLVE)
    case TWO_LEVEL_REVOLVE:
      if (!tj->adjoint_solve_mode) SETERRQ(PetscObjectComm((PetscObject)tj),PETSC_ERR_SUP,"Not implemented");
      ierr = GetTrajTLR(tj,ts,tjsch,stepnum);CHKERRQ(ierr);
      break;
    case TWO_LEVEL_TWO_REVOLVE:
      if (!tj->adjoint_solve_mode) SETERRQ(PetscObjectComm((PetscObject)tj),PETSC_ERR_SUP,"Not implemented");
      ierr = GetTrajTLTR(tj,ts,tjsch,stepnum);CHKERRQ(ierr);
      break;
    case REVOLVE_OFFLINE:
      if (!tj->adjoint_solve_mode) SETERRQ(PetscObjectComm((PetscObject)tj),PETSC_ERR_SUP,"Not implemented");
      ierr = GetTrajROF(tj,ts,tjsch,stepnum);CHKERRQ(ierr);
      break;
    case REVOLVE_ONLINE:
      if (!tj->adjoint_solve_mode) SETERRQ(PetscObjectComm((PetscObject)tj),PETSC_ERR_SUP,"Not implemented");
      ierr = GetTrajRON(tj,ts,tjsch,stepnum);CHKERRQ(ierr);
      break;
    case REVOLVE_MULTISTAGE:
      if (!tj->adjoint_solve_mode) SETERRQ(PetscObjectComm((PetscObject)tj),PETSC_ERR_SUP,"Not implemented");
      ierr = GetTrajRMS(tj,ts,tjsch,stepnum);CHKERRQ(ierr);
      break;
#endif
    default:
      break;
  }
  PetscFunctionReturn(0);
}

PETSC_UNUSED static PetscErrorCode TSTrajectorySetStride_Memory(TSTrajectory tj,PetscInt stride)
{
  TJScheduler *tjsch = (TJScheduler*)tj->data;

  PetscFunctionBegin;
  tjsch->stride = stride;
  PetscFunctionReturn(0);
}

PETSC_UNUSED static PetscErrorCode TSTrajectorySetMaxCpsRAM_Memory(TSTrajectory tj,PetscInt max_cps_ram)
{
  TJScheduler *tjsch = (TJScheduler*)tj->data;

  PetscFunctionBegin;
  tjsch->max_cps_ram = max_cps_ram;
  PetscFunctionReturn(0);
}

PETSC_UNUSED static PetscErrorCode TSTrajectorySetMaxCpsDisk_Memory(TSTrajectory tj,PetscInt max_cps_disk)
{
  TJScheduler *tjsch = (TJScheduler*)tj->data;

  PetscFunctionBegin;
  tjsch->max_cps_disk = max_cps_disk;
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_REVOLVE)
PETSC_UNUSED static PetscErrorCode TSTrajectorySetRevolveOnline(TSTrajectory tj,PetscBool use_online)
{
  TJScheduler *tjsch = (TJScheduler*)tj->data;

  PetscFunctionBegin;
  tjsch->use_online = use_online;
  PetscFunctionReturn(0);
}
#endif

PETSC_UNUSED static PetscErrorCode TSTrajectorySetSaveStack(TSTrajectory tj,PetscBool save_stack)
{
  TJScheduler *tjsch = (TJScheduler*)tj->data;

  PetscFunctionBegin;
  tjsch->save_stack = save_stack;
  PetscFunctionReturn(0);
}

PETSC_UNUSED static PetscErrorCode TSTrajectorySetUseDRAM(TSTrajectory tj,PetscBool use_dram)
{
  TJScheduler *tjsch = (TJScheduler*)tj->data;

  PetscFunctionBegin;
  tjsch->stack.use_dram = use_dram;
  PetscFunctionReturn(0);
}

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
#if defined(PETSC_HAVE_REVOLVE)
    ierr = PetscOptionsBool("-ts_trajectory_revolve_online","Trick TS trajectory into using online mode of revolve","TSTrajectorySetRevolveOnline",tjsch->use_online,&tjsch->use_online,NULL);CHKERRQ(ierr);
#endif
    ierr = PetscOptionsBool("-ts_trajectory_save_stack","Save all stack to disk","TSTrajectorySetSaveStack",tjsch->save_stack,&tjsch->save_stack,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-ts_trajectory_use_dram","Use DRAM for checkpointing","TSTrajectorySetUseDRAM",tjsch->stack.use_dram,&tjsch->stack.use_dram,NULL);CHKERRQ(ierr);
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  tjsch->stack.solution_only = tj->solution_only;
  PetscFunctionReturn(0);
}

static PetscErrorCode TSTrajectorySetUp_Memory(TSTrajectory tj,TS ts)
{
  TJScheduler    *tjsch = (TJScheduler*)tj->data;
  Stack          *stack = &tjsch->stack;
#if defined(PETSC_HAVE_REVOLVE)
  RevolveCTX     *rctx,*rctx2;
  DiskStack      *diskstack = &tjsch->diskstack;
  PetscInt       diskblocks;
#endif
  PetscInt       numY,total_steps;
  PetscBool      fixedtimestep;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (ts->adapt) {
    ierr = PetscObjectTypeCompare((PetscObject)ts->adapt,TSADAPTNONE,&fixedtimestep);CHKERRQ(ierr);
  } else fixedtimestep = PETSC_TRUE;
  total_steps = (PetscInt)(PetscCeilReal((ts->max_time-ts->ptime)/ts->time_step));
  total_steps = total_steps < 0 ? PETSC_MAX_INT : total_steps;
  if (fixedtimestep) tjsch->total_steps = PetscMin(ts->max_steps,total_steps);
  if (tjsch->max_cps_ram > 0) stack->stacksize = tjsch->max_cps_ram;

  if (tjsch->stride > 1) { /* two level mode */
    if (tjsch->save_stack && tjsch->max_cps_disk > 1 && tjsch->max_cps_disk <= tjsch->max_cps_ram) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_ARG_INCOMP,"The specified disk capacity is not enough to store a full stack of RAM checkpoints. You might want to change the disk capacity or use single level checkpointing instead.");
    if (tjsch->max_cps_disk <= 1 && tjsch->max_cps_ram > 1 && tjsch->max_cps_ram <= tjsch->stride-1) tjsch->stype = TWO_LEVEL_REVOLVE; /* use revolve_offline for each stride */
    if (tjsch->max_cps_disk > 1 && tjsch->max_cps_ram > 1 && tjsch->max_cps_ram <= tjsch->stride-1) tjsch->stype = TWO_LEVEL_TWO_REVOLVE;  /* use revolve_offline for each stride */
    if (tjsch->max_cps_disk <= 1 && (tjsch->max_cps_ram >= tjsch->stride || tjsch->max_cps_ram == -1)) tjsch->stype = TWO_LEVEL_NOREVOLVE; /* can also be handled by TWO_LEVEL_REVOLVE */
  } else { /* single level mode */
    if (fixedtimestep) {
      if (tjsch->max_cps_ram >= tjsch->total_steps-1 || tjsch->max_cps_ram < 1) tjsch->stype = NONE; /* checkpoint all */
      else tjsch->stype = (tjsch->max_cps_disk>1) ? REVOLVE_MULTISTAGE : REVOLVE_OFFLINE;
    } else tjsch->stype = NONE; /* checkpoint all for adaptive time step */
#if defined(PETSC_HAVE_REVOLVE)
    if (tjsch->use_online) tjsch->stype = REVOLVE_ONLINE; /* trick into online (for testing purpose only) */
#endif
  }

  if (tjsch->stype > TWO_LEVEL_NOREVOLVE) {
#ifndef PETSC_HAVE_REVOLVE
    SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_SUP,"revolve is needed when there is not enough memory to checkpoint all time steps according to the user's settings, please reconfigure with the additional option --download-revolve.");
#else
    PetscRevolveInt rfine,rsnaps,rsnaps2;
    switch (tjsch->stype) {
      case TWO_LEVEL_REVOLVE:
        ierr = PetscRevolveIntCast(tjsch->stride,&rfine);CHKERRQ(ierr);
        ierr = PetscRevolveIntCast(tjsch->max_cps_ram,&rsnaps);CHKERRQ(ierr);
        revolve_create_offline(rfine,rsnaps);
        break;
      case TWO_LEVEL_TWO_REVOLVE:
        diskblocks = tjsch->save_stack ? tjsch->max_cps_disk/(tjsch->max_cps_ram+1) : tjsch->max_cps_disk; /* The block size depends on whether the stack is saved. */
        diskstack->stacksize = diskblocks;
        ierr = PetscRevolveIntCast(tjsch->stride,&rfine);CHKERRQ(ierr);
        ierr = PetscRevolveIntCast(tjsch->max_cps_ram,&rsnaps);CHKERRQ(ierr);
        revolve_create_offline(rfine,rsnaps);
        ierr = PetscRevolveIntCast((tjsch->total_steps+tjsch->stride-1)/tjsch->stride,&rfine);CHKERRQ(ierr);
        ierr = PetscRevolveIntCast(diskblocks,&rsnaps);CHKERRQ(ierr);
        revolve2_create_offline(rfine,rsnaps);
        ierr = PetscNew(&rctx2);CHKERRQ(ierr);
        rctx2->snaps_in       = rsnaps;
        rctx2->reverseonestep = PETSC_FALSE;
        rctx2->check          = 0;
        rctx2->oldcapo        = 0;
        rctx2->capo           = 0;
        rctx2->info           = 2;
        rctx2->fine           = rfine;
        tjsch->rctx2          = rctx2;
        diskstack->top        = -1;
        ierr = PetscMalloc1(diskstack->stacksize*sizeof(PetscInt),&diskstack->container);CHKERRQ(ierr);
        break;
      case REVOLVE_OFFLINE:
        ierr = PetscRevolveIntCast(tjsch->total_steps,&rfine);CHKERRQ(ierr);
        ierr = PetscRevolveIntCast(tjsch->max_cps_ram,&rsnaps);CHKERRQ(ierr);
        revolve_create_offline(rfine,rsnaps);
        break;
      case REVOLVE_ONLINE:
        stack->stacksize = tjsch->max_cps_ram;
        ierr = PetscRevolveIntCast(tjsch->max_cps_ram,&rsnaps);CHKERRQ(ierr);
        revolve_create_online(rsnaps);
        break;
      case REVOLVE_MULTISTAGE:
        ierr = PetscRevolveIntCast(tjsch->total_steps,&rfine);CHKERRQ(ierr);
        ierr = PetscRevolveIntCast(tjsch->max_cps_ram,&rsnaps);CHKERRQ(ierr);
        ierr = PetscRevolveIntCast(tjsch->max_cps_ram+tjsch->max_cps_disk,&rsnaps2);CHKERRQ(ierr);
        revolve_create_multistage(rfine,rsnaps2,rsnaps);
        break;
      default:
        break;
    }
    ierr = PetscNew(&rctx);CHKERRQ(ierr);
    ierr = PetscRevolveIntCast(tjsch->max_cps_ram,&rsnaps);CHKERRQ(ierr);
    rctx->snaps_in       = rsnaps; /* for theta methods snaps_in=2*max_cps_ram */
    rctx->reverseonestep = PETSC_FALSE;
    rctx->check          = 0;
    rctx->oldcapo        = 0;
    rctx->capo           = 0;
    rctx->info           = 2;
    if (tjsch->stride > 1) {
      ierr = PetscRevolveIntCast(tjsch->stride,&rfine);CHKERRQ(ierr);
    } else {
      ierr = PetscRevolveIntCast(tjsch->total_steps,&rfine);CHKERRQ(ierr);
    }
    rctx->fine           = rfine;
    tjsch->rctx          = rctx;
    if (tjsch->stype == REVOLVE_ONLINE) rctx->fine = -1;
#endif
  } else {
    if (tjsch->stype == TWO_LEVEL_NOREVOLVE) stack->stacksize = tjsch->stride-1; /* need tjsch->stride-1 at most */
    if (tjsch->stype == NONE) {
      if (fixedtimestep) stack->stacksize = stack->solution_only ? tjsch->total_steps : tjsch->total_steps-1;
      else { /* adaptive time step */
        /* if max_cps_ram is not specified, use maximal allowed number of steps for stack size */
        if (tjsch->max_cps_ram == -1) stack->stacksize = ts->max_steps < PETSC_MAX_INT ? ts->max_steps : 10000;
        tjsch->total_steps = stack->solution_only ? stack->stacksize : stack->stacksize+1; /* will be updated as time integration advances */
      }
    }
  }

  if ((tjsch->stype >= TWO_LEVEL_NOREVOLVE && tjsch->stype < REVOLVE_OFFLINE) || tjsch->stype == REVOLVE_MULTISTAGE) { /* these types need to use disk */
    ierr = TSTrajectorySetUp_Basic(tj,ts);CHKERRQ(ierr);
  }

  stack->stacksize = PetscMax(stack->stacksize,1);
  tjsch->recompute = PETSC_FALSE;
  ierr = TSGetStages(ts,&numY,PETSC_IGNORE);CHKERRQ(ierr);
  ierr = StackInit(stack,stack->stacksize,numY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSTrajectoryReset_Memory(TSTrajectory tj)
{
#if defined(PETSC_HAVE_REVOLVE)
  TJScheduler    *tjsch = (TJScheduler*)tj->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (tjsch->stype > TWO_LEVEL_NOREVOLVE) {
    revolve_reset();
    if (tjsch->stype == TWO_LEVEL_TWO_REVOLVE) {
      revolve2_reset();
      ierr = PetscFree(tjsch->diskstack.container);CHKERRQ(ierr);
    }
  }
  if (tjsch->stype > TWO_LEVEL_NOREVOLVE) {
    ierr = PetscFree(tjsch->rctx);CHKERRQ(ierr);
    ierr = PetscFree(tjsch->rctx2);CHKERRQ(ierr);
  }
#endif
  PetscFunctionReturn(0);
}

static PetscErrorCode TSTrajectoryDestroy_Memory(TSTrajectory tj)
{
  TJScheduler    *tjsch = (TJScheduler*)tj->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = StackDestroy(&tjsch->stack);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&tjsch->viewer);CHKERRQ(ierr);
  ierr = PetscFree(tjsch);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
      TSTRAJECTORYMEMORY - Stores each solution of the ODE/ADE in memory

  Level: intermediate

.seealso:  TSTrajectoryCreate(), TS, TSTrajectorySetType()

M*/
PETSC_EXTERN PetscErrorCode TSTrajectoryCreate_Memory(TSTrajectory tj,TS ts)
{
  TJScheduler    *tjsch;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  tj->ops->set            = TSTrajectorySet_Memory;
  tj->ops->get            = TSTrajectoryGet_Memory;
  tj->ops->setup          = TSTrajectorySetUp_Memory;
  tj->ops->setfromoptions = TSTrajectorySetFromOptions_Memory;
  tj->ops->reset          = TSTrajectoryReset_Memory;
  tj->ops->destroy        = TSTrajectoryDestroy_Memory;

  ierr = PetscNew(&tjsch);CHKERRQ(ierr);
  tjsch->stype        = NONE;
  tjsch->max_cps_ram  = -1; /* -1 indicates that it is not set */
  tjsch->max_cps_disk = -1; /* -1 indicates that it is not set */
  tjsch->stride       = 0; /* if not zero, two-level checkpointing will be used */
#if defined(PETSC_HAVE_REVOLVE)
  tjsch->use_online   = PETSC_FALSE;
#endif
  tjsch->save_stack   = PETSC_TRUE;

  tjsch->stack.solution_only = tj->solution_only;
  ierr = PetscViewerCreate(PetscObjectComm((PetscObject)tj),&tjsch->viewer);CHKERRQ(ierr);
  ierr = PetscViewerSetType(tjsch->viewer,PETSCVIEWERBINARY);CHKERRQ(ierr);
  ierr = PetscViewerPushFormat(tjsch->viewer,PETSC_VIEWER_NATIVE);CHKERRQ(ierr);
  ierr = PetscViewerFileSetMode(tjsch->viewer,FILE_MODE_WRITE);CHKERRQ(ierr);

  tj->data = tjsch;
  PetscFunctionReturn(0);
}
