#include <petsc/private/tsimpl.h> /*I "petscts.h"  I*/
#include <petscsys.h>
#if defined(PETSC_HAVE_REVOLVE)
  #include <revolve_c.h>

  /* Limit Revolve to 32-bits */
  #define PETSC_REVOLVE_INT_MAX 2147483647

typedef int PetscRevolveInt;

static inline PetscErrorCode PetscRevolveIntCast(PetscInt a, PetscRevolveInt *b)
{
  PetscFunctionBegin;
  #if defined(PETSC_USE_64BIT_INDICES)
  *b = 0;
  PetscCheck(a <= PETSC_REVOLVE_INT_MAX, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Parameter is too large for Revolve, which is restricted to 32 bit integers");
  #endif
  *b = (PetscRevolveInt)(a);
  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif
#if defined(PETSC_HAVE_CAMS)
  #include <offline_schedule.h>
#endif

PetscLogEvent         TSTrajectory_DiskWrite, TSTrajectory_DiskRead;
static PetscErrorCode TSTrajectorySet_Memory(TSTrajectory, TS, PetscInt, PetscReal, Vec);

typedef enum {
  NONE,
  TWO_LEVEL_NOREVOLVE,
  TWO_LEVEL_REVOLVE,
  TWO_LEVEL_TWO_REVOLVE,
  REVOLVE_OFFLINE,
  REVOLVE_ONLINE,
  REVOLVE_MULTISTAGE,
  CAMS_OFFLINE
} SchedulerType;

typedef enum {
  UNSET           = -1,
  SOLUTIONONLY    = 0,
  STAGESONLY      = 1,
  SOLUTION_STAGES = 2
} CheckpointType;

typedef enum {
  TJ_REVOLVE,
  TJ_CAMS,
  TJ_PETSC
} TSTrajectoryMemoryType;
static const char *const TSTrajectoryMemoryTypes[] = {"REVOLVE", "CAMS", "PETSC", "TSTrajectoryMemoryType", "TJ_", NULL};

#define HaveSolution(m) ((m) == SOLUTIONONLY || (m) == SOLUTION_STAGES)
#define HaveStages(m)   ((m) == STAGESONLY || (m) == SOLUTION_STAGES)

typedef struct _StackElement {
  PetscInt       stepnum;
  Vec            X;
  Vec           *Y;
  PetscReal      time;
  PetscReal      timeprev; /* for no solution_only mode */
  PetscReal      timenext; /* for solution_only mode */
  CheckpointType cptype;
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

#if defined(PETSC_HAVE_CAMS)
typedef struct _CAMSCTX {
  PetscInt lastcheckpointstep;
  PetscInt lastcheckpointtype;
  PetscInt num_units_avail;
  PetscInt endstep;
  PetscInt num_stages;
  PetscInt nextcheckpointstep;
  PetscInt nextcheckpointtype; /* (0) solution only (1) stages (2) solution+stages */
  PetscInt info;
} CAMSCTX;
#endif

typedef struct _Stack {
  PetscInt      stacksize;
  PetscInt      top;
  StackElement *container;
  PetscInt      nallocated;
  PetscInt      numY;
  PetscBool     solution_only;
  PetscBool     use_dram;
} Stack;

typedef struct _DiskStack {
  PetscInt  stacksize;
  PetscInt  top;
  PetscInt *container;
} DiskStack;

typedef struct _TJScheduler {
  SchedulerType          stype;
  TSTrajectoryMemoryType tj_memory_type;
#if defined(PETSC_HAVE_REVOLVE)
  RevolveCTX *rctx, *rctx2;
  PetscBool   use_online;
  PetscInt    store_stride;
#endif
#if defined(PETSC_HAVE_CAMS)
  CAMSCTX *actx;
#endif
  PetscBool   recompute;
  PetscBool   skip_trajectory;
  PetscBool   save_stack;
  PetscInt    max_units_ram;  /* maximum checkpointing units in RAM */
  PetscInt    max_units_disk; /* maximum checkpointing units on disk */
  PetscInt    max_cps_ram;    /* maximum checkpoints in RAM */
  PetscInt    max_cps_disk;   /* maximum checkpoints on disk */
  PetscInt    stride;
  PetscInt    total_steps; /* total number of steps */
  Stack       stack;
  DiskStack   diskstack;
  PetscViewer viewer;
} TJScheduler;

static PetscErrorCode TurnForwardWithStepsize(TS ts, PetscReal nextstepsize)
{
  PetscFunctionBegin;
  /* reverse the direction */
  PetscCall(TSSetTimeStep(ts, nextstepsize));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TurnForward(TS ts)
{
  PetscReal stepsize;

  PetscFunctionBegin;
  /* reverse the direction */
  PetscCall(TSGetTimeStep(ts, &stepsize));
  PetscCall(TSSetTimeStep(ts, -stepsize));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TurnBackward(TS ts)
{
  PetscReal stepsize;

  PetscFunctionBegin;
  if (!ts->trajectory->adjoint_solve_mode) PetscFunctionReturn(PETSC_SUCCESS);
  /* reverse the direction */
  stepsize = ts->ptime_prev - ts->ptime;
  PetscCall(TSSetTimeStep(ts, stepsize));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ElementCreate(TS ts, CheckpointType cptype, Stack *stack, StackElement *e)
{
  Vec  X;
  Vec *Y;

  PetscFunctionBegin;
  if (stack->top < stack->stacksize - 1 && stack->container[stack->top + 1]) {
    *e = stack->container[stack->top + 1];
    if (HaveSolution(cptype) && !(*e)->X) {
      PetscCall(TSGetSolution(ts, &X));
      PetscCall(VecDuplicate(X, &(*e)->X));
    }
    if (cptype == 1 && (*e)->X) PetscCall(VecDestroy(&(*e)->X));
    if (HaveStages(cptype) && !(*e)->Y) {
      PetscCall(TSGetStages(ts, &stack->numY, &Y));
      if (stack->numY) PetscCall(VecDuplicateVecs(Y[0], stack->numY, &(*e)->Y));
    }
    if (cptype == 0 && (*e)->Y) PetscCall(VecDestroyVecs(stack->numY, &(*e)->Y));
    (*e)->cptype = cptype;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  if (stack->use_dram) PetscCall(PetscMallocSetDRAM());
  PetscCall(PetscNew(e));
  if (HaveSolution(cptype)) {
    PetscCall(TSGetSolution(ts, &X));
    PetscCall(VecDuplicate(X, &(*e)->X));
  }
  if (HaveStages(cptype)) {
    PetscCall(TSGetStages(ts, &stack->numY, &Y));
    if (stack->numY) PetscCall(VecDuplicateVecs(Y[0], stack->numY, &(*e)->Y));
  }
  if (stack->use_dram) PetscCall(PetscMallocResetDRAM());
  stack->nallocated++;
  (*e)->cptype = cptype;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ElementSet(TS ts, Stack *stack, StackElement *e, PetscInt stepnum, PetscReal time, Vec X)
{
  Vec      *Y;
  PetscInt  i;
  PetscReal timeprev;

  PetscFunctionBegin;
  if (HaveSolution((*e)->cptype)) PetscCall(VecCopy(X, (*e)->X));
  if (HaveStages((*e)->cptype)) {
    PetscCall(TSGetStages(ts, &stack->numY, &Y));
    for (i = 0; i < stack->numY; i++) PetscCall(VecCopy(Y[i], (*e)->Y[i]));
  }
  (*e)->stepnum = stepnum;
  (*e)->time    = time;
  /* for consistency */
  if (stepnum == 0) {
    (*e)->timeprev = (*e)->time - ts->time_step;
  } else {
    PetscCall(TSGetPrevTime(ts, &timeprev));
    (*e)->timeprev = timeprev;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ElementDestroy(Stack *stack, StackElement e)
{
  PetscFunctionBegin;
  if (stack->use_dram) PetscCall(PetscMallocSetDRAM());
  PetscCall(VecDestroy(&e->X));
  if (e->Y) PetscCall(VecDestroyVecs(stack->numY, &e->Y));
  PetscCall(PetscFree(e));
  if (stack->use_dram) PetscCall(PetscMallocResetDRAM());
  stack->nallocated--;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode StackResize(Stack *stack, PetscInt newsize)
{
  StackElement *newcontainer;
  PetscInt      i;

  PetscFunctionBegin;
  PetscCall(PetscCalloc1(newsize * sizeof(StackElement), &newcontainer));
  for (i = 0; i < stack->stacksize; i++) newcontainer[i] = stack->container[i];
  PetscCall(PetscFree(stack->container));
  stack->container = newcontainer;
  stack->stacksize = newsize;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode StackPush(Stack *stack, StackElement e)
{
  PetscFunctionBegin;
  PetscCheck(stack->top + 1 < stack->stacksize, PETSC_COMM_SELF, PETSC_ERR_MEMC, "Maximum stack size (%" PetscInt_FMT ") exceeded", stack->stacksize);
  stack->container[++stack->top] = e;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode StackPop(Stack *stack, StackElement *e)
{
  PetscFunctionBegin;
  *e = NULL;
  PetscCheck(stack->top != -1, PETSC_COMM_SELF, PETSC_ERR_MEMC, "Empty stack");
  *e = stack->container[stack->top--];
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode StackTop(Stack *stack, StackElement *e)
{
  PetscFunctionBegin;
  *e = stack->container[stack->top];
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode StackInit(Stack *stack, PetscInt size, PetscInt ny)
{
  PetscFunctionBegin;
  stack->top  = -1;
  stack->numY = ny;

  if (!stack->container) PetscCall(PetscCalloc1(size, &stack->container));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode StackDestroy(Stack *stack)
{
  const PetscInt n = stack->nallocated;

  PetscFunctionBegin;
  if (!stack->container) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCheck(stack->top + 1 <= n, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Stack size does not match element counter %" PetscInt_FMT, n);
  for (PetscInt i = 0; i < n; i++) PetscCall(ElementDestroy(stack, stack->container[i]));
  PetscCall(PetscFree(stack->container));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode StackFind(Stack *stack, StackElement *e, PetscInt index)
{
  PetscFunctionBegin;
  *e = NULL;
  PetscCheck(index >= 0 && index <= stack->top, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Invalid index %" PetscInt_FMT, index);
  *e = stack->container[index];
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode WriteToDisk(PetscBool stifflyaccurate, PetscInt stepnum, PetscReal time, PetscReal timeprev, Vec X, Vec *Y, PetscInt numY, CheckpointType cptype, PetscViewer viewer)
{
  PetscFunctionBegin;
  PetscCall(PetscViewerBinaryWrite(viewer, &stepnum, 1, PETSC_INT));
  if (HaveSolution(cptype)) PetscCall(VecView(X, viewer));
  if (HaveStages(cptype)) {
    for (PetscInt i = 0; i < numY; i++) {
      /* For stiffly accurate TS methods, the last stage Y[ns-1] is the same as the solution X, thus does not need to be saved again. */
      if (stifflyaccurate && i == numY - 1 && HaveSolution(cptype)) continue;
      PetscCall(VecView(Y[i], viewer));
    }
  }
  PetscCall(PetscViewerBinaryWrite(viewer, &time, 1, PETSC_REAL));
  PetscCall(PetscViewerBinaryWrite(viewer, &timeprev, 1, PETSC_REAL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ReadFromDisk(PetscBool stifflyaccurate, PetscInt *stepnum, PetscReal *time, PetscReal *timeprev, Vec X, Vec *Y, PetscInt numY, CheckpointType cptype, PetscViewer viewer)
{
  PetscFunctionBegin;
  PetscCall(PetscViewerBinaryRead(viewer, stepnum, 1, NULL, PETSC_INT));
  if (HaveSolution(cptype)) PetscCall(VecLoad(X, viewer));
  if (HaveStages(cptype)) {
    for (PetscInt i = 0; i < numY; i++) {
      /* For stiffly accurate TS methods, the last stage Y[ns-1] is the same as the solution X, thus does not need to be loaded again. */
      if (stifflyaccurate && i == numY - 1 && HaveSolution(cptype)) continue;
      PetscCall(VecLoad(Y[i], viewer));
    }
  }
  PetscCall(PetscViewerBinaryRead(viewer, time, 1, NULL, PETSC_REAL));
  PetscCall(PetscViewerBinaryRead(viewer, timeprev, 1, NULL, PETSC_REAL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode StackDumpAll(TSTrajectory tj, TS ts, Stack *stack, PetscInt id)
{
  Vec         *Y;
  PetscInt     ndumped, cptype_int;
  StackElement e     = NULL;
  TJScheduler *tjsch = (TJScheduler *)tj->data;
  char         filename[PETSC_MAX_PATH_LEN];
  MPI_Comm     comm;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)ts, &comm));
  if (tj->monitor) {
    PetscCall(PetscViewerASCIIPushTab(tj->monitor));
    PetscCall(PetscViewerASCIIPrintf(tj->monitor, "Dump stack id %" PetscInt_FMT " to file\n", id));
    PetscCall(PetscViewerASCIIPopTab(tj->monitor));
  }
  PetscCall(PetscSNPrintf(filename, sizeof(filename), "%s/TS-STACK%06" PetscInt_FMT ".bin", tj->dirname, id));
  PetscCall(PetscViewerFileSetName(tjsch->viewer, filename));
  PetscCall(PetscViewerSetUp(tjsch->viewer));
  ndumped = stack->top + 1;
  PetscCall(PetscViewerBinaryWrite(tjsch->viewer, &ndumped, 1, PETSC_INT));
  for (PetscInt i = 0; i < ndumped; i++) {
    e          = stack->container[i];
    cptype_int = (PetscInt)e->cptype;
    PetscCall(PetscViewerBinaryWrite(tjsch->viewer, &cptype_int, 1, PETSC_INT));
    PetscCall(PetscLogEventBegin(TSTrajectory_DiskWrite, tj, ts, 0, 0));
    PetscCall(WriteToDisk(ts->stifflyaccurate, e->stepnum, e->time, e->timeprev, e->X, e->Y, stack->numY, e->cptype, tjsch->viewer));
    PetscCall(PetscLogEventEnd(TSTrajectory_DiskWrite, tj, ts, 0, 0));
    ts->trajectory->diskwrites++;
    PetscCall(StackPop(stack, &e));
  }
  /* save the last step for restart, the last step is in memory when using single level schemes, but not necessarily the case for multi level schemes */
  PetscCall(TSGetStages(ts, &stack->numY, &Y));
  PetscCall(PetscLogEventBegin(TSTrajectory_DiskWrite, tj, ts, 0, 0));
  PetscCall(WriteToDisk(ts->stifflyaccurate, ts->steps, ts->ptime, ts->ptime_prev, ts->vec_sol, Y, stack->numY, SOLUTION_STAGES, tjsch->viewer));
  PetscCall(PetscLogEventEnd(TSTrajectory_DiskWrite, tj, ts, 0, 0));
  ts->trajectory->diskwrites++;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode StackLoadAll(TSTrajectory tj, TS ts, Stack *stack, PetscInt id)
{
  Vec         *Y;
  PetscInt     i, nloaded, cptype_int;
  StackElement e;
  PetscViewer  viewer;
  char         filename[PETSC_MAX_PATH_LEN];

  PetscFunctionBegin;
  if (tj->monitor) {
    PetscCall(PetscViewerASCIIAddTab(tj->monitor, ((PetscObject)tj)->tablevel));
    PetscCall(PetscViewerASCIIPrintf(tj->monitor, "Load stack from file\n"));
    PetscCall(PetscViewerASCIISubtractTab(tj->monitor, ((PetscObject)tj)->tablevel));
  }
  PetscCall(PetscSNPrintf(filename, sizeof filename, "%s/TS-STACK%06" PetscInt_FMT ".bin", tj->dirname, id));
  PetscCall(PetscViewerBinaryOpen(PetscObjectComm((PetscObject)tj), filename, FILE_MODE_READ, &viewer));
  PetscCall(PetscViewerBinarySetSkipInfo(viewer, PETSC_TRUE));
  PetscCall(PetscViewerPushFormat(viewer, PETSC_VIEWER_NATIVE));
  PetscCall(PetscViewerBinaryRead(viewer, &nloaded, 1, NULL, PETSC_INT));
  for (i = 0; i < nloaded; i++) {
    PetscCall(PetscViewerBinaryRead(viewer, &cptype_int, 1, NULL, PETSC_INT));
    PetscCall(ElementCreate(ts, (CheckpointType)cptype_int, stack, &e));
    PetscCall(StackPush(stack, e));
    PetscCall(PetscLogEventBegin(TSTrajectory_DiskRead, tj, ts, 0, 0));
    PetscCall(ReadFromDisk(ts->stifflyaccurate, &e->stepnum, &e->time, &e->timeprev, e->X, e->Y, stack->numY, e->cptype, viewer));
    PetscCall(PetscLogEventEnd(TSTrajectory_DiskRead, tj, ts, 0, 0));
    ts->trajectory->diskreads++;
  }
  /* load the last step into TS */
  PetscCall(TSGetStages(ts, &stack->numY, &Y));
  PetscCall(PetscLogEventBegin(TSTrajectory_DiskRead, tj, ts, 0, 0));
  PetscCall(ReadFromDisk(ts->stifflyaccurate, &ts->steps, &ts->ptime, &ts->ptime_prev, ts->vec_sol, Y, stack->numY, SOLUTION_STAGES, viewer));
  PetscCall(PetscLogEventEnd(TSTrajectory_DiskRead, tj, ts, 0, 0));
  ts->trajectory->diskreads++;
  PetscCall(TurnBackward(ts));
  PetscCall(PetscViewerDestroy(&viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#if defined(PETSC_HAVE_REVOLVE)
static PetscErrorCode StackLoadLast(TSTrajectory tj, TS ts, Stack *stack, PetscInt id)
{
  Vec        *Y;
  PetscInt    size;
  PetscViewer viewer;
  char        filename[PETSC_MAX_PATH_LEN];
  #if defined(PETSC_HAVE_MPIIO)
  PetscBool usempiio;
  #endif
  int   fd;
  off_t off, offset;

  PetscFunctionBegin;
  if (tj->monitor) {
    PetscCall(PetscViewerASCIIAddTab(tj->monitor, ((PetscObject)tj)->tablevel));
    PetscCall(PetscViewerASCIIPrintf(tj->monitor, "Load last stack element from file\n"));
    PetscCall(PetscViewerASCIISubtractTab(tj->monitor, ((PetscObject)tj)->tablevel));
  }
  PetscCall(TSGetStages(ts, &stack->numY, &Y));
  PetscCall(VecGetSize(Y[0], &size));
  /* VecView writes to file two extra int's for class id and number of rows */
  off = -((stack->solution_only ? 0 : stack->numY) + 1) * (size * PETSC_BINARY_SCALAR_SIZE + 2 * PETSC_BINARY_INT_SIZE) - PETSC_BINARY_INT_SIZE - 2 * PETSC_BINARY_SCALAR_SIZE;

  PetscCall(PetscSNPrintf(filename, sizeof filename, "%s/TS-STACK%06" PetscInt_FMT ".bin", tj->dirname, id));
  PetscCall(PetscViewerBinaryOpen(PetscObjectComm((PetscObject)tj), filename, FILE_MODE_READ, &viewer));
  PetscCall(PetscViewerBinarySetSkipInfo(viewer, PETSC_TRUE));
  PetscCall(PetscViewerPushFormat(viewer, PETSC_VIEWER_NATIVE));
  #if defined(PETSC_HAVE_MPIIO)
  PetscCall(PetscViewerBinaryGetUseMPIIO(viewer, &usempiio));
  if (usempiio) {
    PetscCall(PetscViewerBinaryGetMPIIODescriptor(viewer, (MPI_File *)&fd));
    PetscCall(PetscBinarySynchronizedSeek(PetscObjectComm((PetscObject)tj), fd, off, PETSC_BINARY_SEEK_END, &offset));
  } else {
  #endif
    PetscCall(PetscViewerBinaryGetDescriptor(viewer, &fd));
    PetscCall(PetscBinarySeek(fd, off, PETSC_BINARY_SEEK_END, &offset));
  #if defined(PETSC_HAVE_MPIIO)
  }
  #endif
  /* load the last step into TS */
  PetscCall(PetscLogEventBegin(TSTrajectory_DiskRead, tj, ts, 0, 0));
  PetscCall(ReadFromDisk(ts->stifflyaccurate, &ts->steps, &ts->ptime, &ts->ptime_prev, ts->vec_sol, Y, stack->numY, SOLUTION_STAGES, viewer));
  PetscCall(PetscLogEventEnd(TSTrajectory_DiskRead, tj, ts, 0, 0));
  ts->trajectory->diskreads++;
  PetscCall(PetscViewerDestroy(&viewer));
  PetscCall(TurnBackward(ts));
  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif

static PetscErrorCode DumpSingle(TSTrajectory tj, TS ts, Stack *stack, PetscInt id)
{
  Vec         *Y;
  PetscInt     stepnum;
  TJScheduler *tjsch = (TJScheduler *)tj->data;
  char         filename[PETSC_MAX_PATH_LEN];
  MPI_Comm     comm;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)ts, &comm));
  if (tj->monitor) {
    PetscCall(PetscViewerASCIIAddTab(tj->monitor, ((PetscObject)tj)->tablevel));
    PetscCall(PetscViewerASCIIPrintf(tj->monitor, "Dump a single point from file\n"));
    PetscCall(PetscViewerASCIISubtractTab(tj->monitor, ((PetscObject)tj)->tablevel));
  }
  PetscCall(TSGetStepNumber(ts, &stepnum));
  PetscCall(PetscSNPrintf(filename, sizeof(filename), "%s/TS-CPS%06" PetscInt_FMT ".bin", tj->dirname, id));
  PetscCall(PetscViewerFileSetName(tjsch->viewer, filename));
  PetscCall(PetscViewerSetUp(tjsch->viewer));

  PetscCall(TSGetStages(ts, &stack->numY, &Y));
  PetscCall(PetscLogEventBegin(TSTrajectory_DiskWrite, tj, ts, 0, 0));
  PetscCall(WriteToDisk(ts->stifflyaccurate, stepnum, ts->ptime, ts->ptime_prev, ts->vec_sol, Y, stack->numY, SOLUTION_STAGES, tjsch->viewer));
  PetscCall(PetscLogEventEnd(TSTrajectory_DiskWrite, tj, ts, 0, 0));
  ts->trajectory->diskwrites++;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode LoadSingle(TSTrajectory tj, TS ts, Stack *stack, PetscInt id)
{
  Vec        *Y;
  PetscViewer viewer;
  char        filename[PETSC_MAX_PATH_LEN];

  PetscFunctionBegin;
  if (tj->monitor) {
    PetscCall(PetscViewerASCIIAddTab(tj->monitor, ((PetscObject)tj)->tablevel));
    PetscCall(PetscViewerASCIIPrintf(tj->monitor, "Load a single point from file\n"));
    PetscCall(PetscViewerASCIISubtractTab(tj->monitor, ((PetscObject)tj)->tablevel));
  }
  PetscCall(PetscSNPrintf(filename, sizeof filename, "%s/TS-CPS%06" PetscInt_FMT ".bin", tj->dirname, id));
  PetscCall(PetscViewerBinaryOpen(PetscObjectComm((PetscObject)tj), filename, FILE_MODE_READ, &viewer));
  PetscCall(PetscViewerBinarySetSkipInfo(viewer, PETSC_TRUE));
  PetscCall(PetscViewerPushFormat(viewer, PETSC_VIEWER_NATIVE));
  PetscCall(TSGetStages(ts, &stack->numY, &Y));
  PetscCall(PetscLogEventBegin(TSTrajectory_DiskRead, tj, ts, 0, 0));
  PetscCall(ReadFromDisk(ts->stifflyaccurate, &ts->steps, &ts->ptime, &ts->ptime_prev, ts->vec_sol, Y, stack->numY, SOLUTION_STAGES, viewer));
  PetscCall(PetscLogEventEnd(TSTrajectory_DiskRead, tj, ts, 0, 0));
  ts->trajectory->diskreads++;
  PetscCall(PetscViewerDestroy(&viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode UpdateTS(TS ts, Stack *stack, StackElement e, PetscInt stepnum, PetscBool adjoint_mode)
{
  Vec     *Y;
  PetscInt i;

  PetscFunctionBegin;
  /* In adjoint mode we do not need to copy solution if the stepnum is the same */
  if (!adjoint_mode || (HaveSolution(e->cptype) && e->stepnum != stepnum)) PetscCall(VecCopy(e->X, ts->vec_sol));
  if (HaveStages(e->cptype)) {
    PetscCall(TSGetStages(ts, &stack->numY, &Y));
    if (e->stepnum && e->stepnum == stepnum) {
      for (i = 0; i < stack->numY; i++) PetscCall(VecCopy(e->Y[i], Y[i]));
    } else if (ts->stifflyaccurate) {
      PetscCall(VecCopy(e->Y[stack->numY - 1], ts->vec_sol));
    }
  }
  if (adjoint_mode) {
    PetscCall(TSSetTimeStep(ts, e->timeprev - e->time)); /* stepsize will be negative */
  } else {
    PetscCall(TSSetTimeStep(ts, e->time - e->timeprev)); /* stepsize will be positive */
  }
  ts->ptime      = e->time;
  ts->ptime_prev = e->timeprev;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ReCompute(TS ts, TJScheduler *tjsch, PetscInt stepnumbegin, PetscInt stepnumend)
{
  Stack   *stack = &tjsch->stack;
  PetscInt i;

  PetscFunctionBegin;
  tjsch->recompute = PETSC_TRUE;                           /* hints TSTrajectorySet() that it is in recompute mode */
  PetscCall(TSSetStepNumber(ts, stepnumbegin));            /* global step number */
  for (i = stepnumbegin; i < stepnumend; i++) {            /* assume fixed step size */
    if (stack->solution_only && !tjsch->skip_trajectory) { /* revolve online need this */
      /* don't use the public interface as it will update the TSHistory: this need a better fix */
      PetscCall(TSTrajectorySet_Memory(ts->trajectory, ts, ts->steps, ts->ptime, ts->vec_sol));
    }
    PetscCall(TSMonitor(ts, ts->steps, ts->ptime, ts->vec_sol));
    PetscCall(TSStep(ts));
    if (!stack->solution_only && !tjsch->skip_trajectory) {
      /* don't use the public interface as it will update the TSHistory: this need a better fix */
      PetscCall(TSTrajectorySet_Memory(ts->trajectory, ts, ts->steps, ts->ptime, ts->vec_sol));
    }
    PetscCall(TSEventHandler(ts));
    if (!ts->steprollback) PetscCall(TSPostStep(ts));
  }
  PetscCall(TurnBackward(ts));
  ts->trajectory->recomps += stepnumend - stepnumbegin; /* recomputation counter */
  PetscCall(TSSetStepNumber(ts, stepnumend));
  tjsch->recompute = PETSC_FALSE; /* reset the flag for recompute mode */
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TopLevelStore(TSTrajectory tj, TS ts, TJScheduler *tjsch, PetscInt stepnum, PetscInt localstepnum, PetscInt laststridesize, PetscBool *done)
{
  Stack     *stack     = &tjsch->stack;
  DiskStack *diskstack = &tjsch->diskstack;
  PetscInt   stridenum;

  PetscFunctionBegin;
  *done     = PETSC_FALSE;
  stridenum = stepnum / tjsch->stride;
  /* make sure saved checkpoint id starts from 1
     skip last stride when using stridenum+1
     skip first stride when using stridenum */
  if (stack->solution_only) {
    if (tjsch->save_stack) {
      if (localstepnum == tjsch->stride - 1 && stepnum < tjsch->total_steps - laststridesize) { /* current step will be saved without going through stack */
        PetscCall(StackDumpAll(tj, ts, stack, stridenum + 1));
        if (tjsch->stype == TWO_LEVEL_TWO_REVOLVE) diskstack->container[++diskstack->top] = stridenum + 1;
        *done = PETSC_TRUE;
      }
    } else {
      if (localstepnum == 0 && stepnum < tjsch->total_steps - laststridesize) {
        PetscCall(DumpSingle(tj, ts, stack, stridenum + 1));
        if (tjsch->stype == TWO_LEVEL_TWO_REVOLVE) diskstack->container[++diskstack->top] = stridenum + 1;
        *done = PETSC_TRUE;
      }
    }
  } else {
    if (tjsch->save_stack) {
      if (localstepnum == 0 && stepnum < tjsch->total_steps && stepnum != 0) { /* skip the first stride */
        PetscCall(StackDumpAll(tj, ts, stack, stridenum));
        if (tjsch->stype == TWO_LEVEL_TWO_REVOLVE) diskstack->container[++diskstack->top] = stridenum;
        *done = PETSC_TRUE;
      }
    } else {
      if (localstepnum == 1 && stepnum < tjsch->total_steps - laststridesize) {
        PetscCall(DumpSingle(tj, ts, stack, stridenum + 1));
        if (tjsch->stype == TWO_LEVEL_TWO_REVOLVE) diskstack->container[++diskstack->top] = stridenum + 1;
        *done = PETSC_TRUE;
      }
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSTrajectoryMemorySet_N(TS ts, TJScheduler *tjsch, PetscInt stepnum, PetscReal time, Vec X)
{
  Stack         *stack = &tjsch->stack;
  StackElement   e;
  CheckpointType cptype;

  PetscFunctionBegin;
  /* skip the last step */
  if (ts->reason) { /* only affect the forward run */
    /* update total_steps in the end of forward run */
    if (stepnum != tjsch->total_steps) tjsch->total_steps = stepnum;
    if (stack->solution_only) {
      /* get rid of the solution at second last step */
      PetscCall(StackPop(stack, &e));
    }
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  /*  do not save trajectory at the recompute stage for solution_only mode */
  if (stack->solution_only && tjsch->recompute) PetscFunctionReturn(PETSC_SUCCESS);
  /* skip the first step for no_solution_only mode */
  if (!stack->solution_only && stepnum == 0) PetscFunctionReturn(PETSC_SUCCESS);

  /* resize the stack */
  if (stack->top + 1 == stack->stacksize) PetscCall(StackResize(stack, 2 * stack->stacksize));
  /* update timenext for the previous step; necessary for step adaptivity */
  if (stack->top > -1) {
    PetscCall(StackTop(stack, &e));
    e->timenext = ts->ptime;
  }
  PetscCheck(stepnum >= stack->top, PetscObjectComm((PetscObject)ts), PETSC_ERR_MEMC, "Illegal modification of a non-top stack element");
  cptype = stack->solution_only ? SOLUTIONONLY : STAGESONLY;
  PetscCall(ElementCreate(ts, cptype, stack, &e));
  PetscCall(ElementSet(ts, stack, &e, stepnum, time, X));
  PetscCall(StackPush(stack, e));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSTrajectoryMemorySet_N_2(TS ts, TJScheduler *tjsch, PetscInt stepnum, PetscReal time, Vec X)
{
  Stack         *stack = &tjsch->stack;
  StackElement   e;
  CheckpointType cptype;

  PetscFunctionBegin;
  if (stack->top + 1 == stack->stacksize) PetscCall(StackResize(stack, 2 * stack->stacksize));
  /* update timenext for the previous step; necessary for step adaptivity */
  if (stack->top > -1) {
    PetscCall(StackTop(stack, &e));
    e->timenext = ts->ptime;
  }
  PetscCheck(stepnum >= stack->top, PetscObjectComm((PetscObject)ts), PETSC_ERR_MEMC, "Illegal modification of a non-top stack element");
  cptype = stack->solution_only ? SOLUTIONONLY : SOLUTION_STAGES; /* Always include solution in a checkpoint in non-adjoint mode */
  PetscCall(ElementCreate(ts, cptype, stack, &e));
  PetscCall(ElementSet(ts, stack, &e, stepnum, time, X));
  PetscCall(StackPush(stack, e));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSTrajectoryMemoryGet_N(TS ts, TJScheduler *tjsch, PetscInt stepnum)
{
  Stack       *stack = &tjsch->stack;
  StackElement e;
  PetscInt     ns;

  PetscFunctionBegin;
  /* If TSTrajectoryGet() is called after TSAdjointSolve() converges (e.g. outside the while loop in TSAdjointSolve()), skip getting the checkpoint. */
  if (ts->reason) PetscFunctionReturn(PETSC_SUCCESS);
  if (stepnum == tjsch->total_steps) {
    PetscCall(TurnBackward(ts));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  /* restore a checkpoint */
  PetscCall(StackTop(stack, &e));
  PetscCall(UpdateTS(ts, stack, e, stepnum, PETSC_TRUE));
  PetscCall(TSGetStages(ts, &ns, PETSC_IGNORE));
  if (stack->solution_only && ns) { /* recompute one step */
    PetscCall(TurnForwardWithStepsize(ts, e->timenext - e->time));
    PetscCall(ReCompute(ts, tjsch, e->stepnum, stepnum));
  }
  PetscCall(StackPop(stack, &e));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSTrajectoryMemoryGet_N_2(TS ts, TJScheduler *tjsch, PetscInt stepnum)
{
  Stack       *stack = &tjsch->stack;
  StackElement e     = NULL;

  PetscFunctionBegin;
  PetscCall(StackFind(stack, &e, stepnum));
  PetscCheck(stepnum == e->stepnum, PetscObjectComm((PetscObject)ts), PETSC_ERR_PLIB, "Inconsistent steps! %" PetscInt_FMT " != %" PetscInt_FMT, stepnum, e->stepnum);
  PetscCall(UpdateTS(ts, stack, e, stepnum, PETSC_FALSE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSTrajectoryMemorySet_TLNR(TSTrajectory tj, TS ts, TJScheduler *tjsch, PetscInt stepnum, PetscReal time, Vec X)
{
  Stack         *stack = &tjsch->stack;
  PetscInt       localstepnum, laststridesize;
  StackElement   e;
  PetscBool      done;
  CheckpointType cptype;

  PetscFunctionBegin;
  if (!stack->solution_only && stepnum == 0) PetscFunctionReturn(PETSC_SUCCESS);
  if (stack->solution_only && stepnum == tjsch->total_steps) PetscFunctionReturn(PETSC_SUCCESS);
  if (tjsch->save_stack && tjsch->recompute) PetscFunctionReturn(PETSC_SUCCESS);

  localstepnum = stepnum % tjsch->stride;
  /* (stridesize-1) checkpoints are saved in each stride; an extra point is added by StackDumpAll() */
  laststridesize = tjsch->total_steps % tjsch->stride;
  if (!laststridesize) laststridesize = tjsch->stride;

  if (!tjsch->recompute) {
    PetscCall(TopLevelStore(tj, ts, tjsch, stepnum, localstepnum, laststridesize, &done));
    if (!tjsch->save_stack && stepnum < tjsch->total_steps - laststridesize) PetscFunctionReturn(PETSC_SUCCESS);
  }
  if (!stack->solution_only && localstepnum == 0) PetscFunctionReturn(PETSC_SUCCESS);                /* skip last point in each stride at recompute stage or last stride */
  if (stack->solution_only && localstepnum == tjsch->stride - 1) PetscFunctionReturn(PETSC_SUCCESS); /* skip last step in each stride at recompute stage or last stride */

  cptype = stack->solution_only ? SOLUTIONONLY : STAGESONLY;
  PetscCall(ElementCreate(ts, cptype, stack, &e));
  PetscCall(ElementSet(ts, stack, &e, stepnum, time, X));
  PetscCall(StackPush(stack, e));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSTrajectoryMemoryGet_TLNR(TSTrajectory tj, TS ts, TJScheduler *tjsch, PetscInt stepnum)
{
  Stack       *stack = &tjsch->stack;
  PetscInt     id, localstepnum, laststridesize;
  StackElement e;

  PetscFunctionBegin;
  if (stepnum == tjsch->total_steps) {
    PetscCall(TurnBackward(ts));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  localstepnum   = stepnum % tjsch->stride;
  laststridesize = tjsch->total_steps % tjsch->stride;
  if (!laststridesize) laststridesize = tjsch->stride;
  if (stack->solution_only) {
    /* fill stack with info */
    if (localstepnum == 0 && tjsch->total_steps - stepnum >= laststridesize) {
      id = stepnum / tjsch->stride;
      if (tjsch->save_stack) {
        PetscCall(StackLoadAll(tj, ts, stack, id));
        tjsch->skip_trajectory = PETSC_TRUE;
        PetscCall(TurnForward(ts));
        PetscCall(ReCompute(ts, tjsch, id * tjsch->stride - 1, id * tjsch->stride));
        tjsch->skip_trajectory = PETSC_FALSE;
      } else {
        PetscCall(LoadSingle(tj, ts, stack, id));
        PetscCall(TurnForward(ts));
        PetscCall(ReCompute(ts, tjsch, (id - 1) * tjsch->stride, id * tjsch->stride));
      }
      PetscFunctionReturn(PETSC_SUCCESS);
    }
    /* restore a checkpoint */
    PetscCall(StackPop(stack, &e));
    PetscCall(UpdateTS(ts, stack, e, stepnum, PETSC_TRUE));
    tjsch->skip_trajectory = PETSC_TRUE;
    PetscCall(TurnForward(ts));
    PetscCall(ReCompute(ts, tjsch, e->stepnum, stepnum));
    tjsch->skip_trajectory = PETSC_FALSE;
  } else {
    CheckpointType cptype = STAGESONLY;
    /* fill stack with info */
    if (localstepnum == 0 && tjsch->total_steps - stepnum >= laststridesize) {
      id = stepnum / tjsch->stride;
      if (tjsch->save_stack) {
        PetscCall(StackLoadAll(tj, ts, stack, id));
      } else {
        PetscCall(LoadSingle(tj, ts, stack, id));
        PetscCall(ElementCreate(ts, cptype, stack, &e));
        PetscCall(ElementSet(ts, stack, &e, (id - 1) * tjsch->stride + 1, ts->ptime, ts->vec_sol));
        PetscCall(StackPush(stack, e));
        PetscCall(TurnForward(ts));
        PetscCall(ReCompute(ts, tjsch, e->stepnum, id * tjsch->stride));
      }
      PetscFunctionReturn(PETSC_SUCCESS);
    }
    /* restore a checkpoint */
    PetscCall(StackPop(stack, &e));
    PetscCall(UpdateTS(ts, stack, e, stepnum, PETSC_TRUE));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

#if defined(PETSC_HAVE_REVOLVE)
static PetscErrorCode printwhattodo(PetscViewer viewer, PetscRevolveInt whattodo, RevolveCTX *rctx, PetscRevolveInt shift)
{
  PetscFunctionBegin;
  if (!viewer) PetscFunctionReturn(PETSC_SUCCESS);

  switch (whattodo) {
  case 1:
    PetscCall(PetscViewerASCIIPrintf(viewer, "Advance from %d to %d\n", rctx->oldcapo + shift, rctx->capo + shift));
    break;
  case 2:
    PetscCall(PetscViewerASCIIPrintf(viewer, "Store in checkpoint number %d (located in RAM)\n", rctx->check));
    break;
  case 3:
    PetscCall(PetscViewerASCIIPrintf(viewer, "First turn: Initialize adjoints and reverse first step\n"));
    break;
  case 4:
    PetscCall(PetscViewerASCIIPrintf(viewer, "Forward and reverse one step\n"));
    break;
  case 5:
    PetscCall(PetscViewerASCIIPrintf(viewer, "Restore in checkpoint number %d (located in RAM)\n", rctx->check));
    break;
  case 7:
    PetscCall(PetscViewerASCIIPrintf(viewer, "Store in checkpoint number %d (located on disk)\n", rctx->check));
    break;
  case 8:
    PetscCall(PetscViewerASCIIPrintf(viewer, "Restore in checkpoint number %d (located on disk)\n", rctx->check));
    break;
  case -1:
    PetscCall(PetscViewerASCIIPrintf(viewer, "Error!"));
    break;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode printwhattodo2(PetscViewer viewer, PetscRevolveInt whattodo, RevolveCTX *rctx, PetscRevolveInt shift)
{
  PetscFunctionBegin;
  if (!viewer) PetscFunctionReturn(PETSC_SUCCESS);

  switch (whattodo) {
  case 1:
    PetscCall(PetscViewerASCIIPrintf(viewer, "[Top Level] Advance from stride %d to stride %d\n", rctx->oldcapo + shift, rctx->capo + shift));
    break;
  case 2:
    PetscCall(PetscViewerASCIIPrintf(viewer, "[Top Level] Store in checkpoint number %d\n", rctx->check));
    break;
  case 3:
    PetscCall(PetscViewerASCIIPrintf(viewer, "[Top Level] First turn: Initialize adjoints and reverse first stride\n"));
    break;
  case 4:
    PetscCall(PetscViewerASCIIPrintf(viewer, "[Top Level] Forward and reverse one stride\n"));
    break;
  case 5:
    PetscCall(PetscViewerASCIIPrintf(viewer, "[Top Level] Restore in checkpoint number %d\n", rctx->check));
    break;
  case 7:
    PetscCall(PetscViewerASCIIPrintf(viewer, "[Top Level] Store in top-level checkpoint number %d\n", rctx->check));
    break;
  case 8:
    PetscCall(PetscViewerASCIIPrintf(viewer, "[Top Level] Restore in top-level checkpoint number %d\n", rctx->check));
    break;
  case -1:
    PetscCall(PetscViewerASCIIPrintf(viewer, "[Top Level] Error!"));
    break;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode InitRevolve(PetscInt fine, PetscInt snaps, RevolveCTX *rctx)
{
  PetscRevolveInt rsnaps, rfine;

  PetscFunctionBegin;
  PetscCall(PetscRevolveIntCast(snaps, &rsnaps));
  PetscCall(PetscRevolveIntCast(fine, &rfine));
  revolve_reset();
  revolve_create_offline(rfine, rsnaps);
  rctx->snaps_in       = rsnaps;
  rctx->fine           = rfine;
  rctx->check          = 0;
  rctx->capo           = 0;
  rctx->reverseonestep = PETSC_FALSE;
  /* check stepsleft? */
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FastForwardRevolve(RevolveCTX *rctx)
{
  PetscRevolveInt whattodo;

  PetscFunctionBegin;
  whattodo = 0;
  while (whattodo != 3) { /* we have to fast forward revolve to the beginning of the backward sweep due to unfriendly revolve interface */
    whattodo = revolve_action(&rctx->check, &rctx->capo, &rctx->fine, rctx->snaps_in, &rctx->info, &rctx->where);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ApplyRevolve(PetscViewer viewer, SchedulerType stype, RevolveCTX *rctx, PetscRevolveInt total_steps, PetscRevolveInt stepnum, PetscRevolveInt localstepnum, PetscBool toplevel, PetscInt *store)
{
  PetscRevolveInt shift, whattodo;

  PetscFunctionBegin;
  *store = 0;
  if (rctx->stepsleft > 0) { /* advance the solution without checkpointing anything as Revolve requires */
    rctx->stepsleft--;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  /* let Revolve determine what to do next */
  shift         = stepnum - localstepnum;
  rctx->oldcapo = rctx->capo;
  rctx->capo    = localstepnum;

  if (!toplevel) whattodo = revolve_action(&rctx->check, &rctx->capo, &rctx->fine, rctx->snaps_in, &rctx->info, &rctx->where);
  else whattodo = revolve2_action(&rctx->check, &rctx->capo, &rctx->fine, rctx->snaps_in, &rctx->info, &rctx->where);
  if (stype == REVOLVE_ONLINE && whattodo == 8) whattodo = 5;
  if (stype == REVOLVE_ONLINE && whattodo == 7) whattodo = 2;
  if (!toplevel) PetscCall(printwhattodo(viewer, whattodo, rctx, shift));
  else PetscCall(printwhattodo2(viewer, whattodo, rctx, shift));
  PetscCheck(whattodo != -1, PETSC_COMM_SELF, PETSC_ERR_LIB, "Error in the Revolve library");
  if (whattodo == 1) { /* advance some time steps */
    if (stype == REVOLVE_ONLINE && rctx->capo >= total_steps - 1) {
      revolve_turn(total_steps, &rctx->capo, &rctx->fine);
      if (!toplevel) PetscCall(printwhattodo(viewer, whattodo, rctx, shift));
      else PetscCall(printwhattodo2(viewer, whattodo, rctx, shift));
    }
    rctx->stepsleft = rctx->capo - rctx->oldcapo - 1;
  }
  if (whattodo == 3 || whattodo == 4) { /* ready for a reverse step */
    rctx->reverseonestep = PETSC_TRUE;
  }
  if (whattodo == 5) { /* restore a checkpoint and ask Revolve what to do next */
    rctx->oldcapo = rctx->capo;
    if (!toplevel) whattodo = revolve_action(&rctx->check, &rctx->capo, &rctx->fine, rctx->snaps_in, &rctx->info, &rctx->where); /* must return 1 or 3 or 4*/
    else whattodo = revolve2_action(&rctx->check, &rctx->capo, &rctx->fine, rctx->snaps_in, &rctx->info, &rctx->where);
    if (!toplevel) PetscCall(printwhattodo(viewer, whattodo, rctx, shift));
    else PetscCall(printwhattodo2(viewer, whattodo, rctx, shift));
    if (whattodo == 3 || whattodo == 4) rctx->reverseonestep = PETSC_TRUE;
    if (whattodo == 1) rctx->stepsleft = rctx->capo - rctx->oldcapo;
  }
  if (whattodo == 7) { /* save the checkpoint to disk */
    *store        = 2;
    rctx->oldcapo = rctx->capo;
    whattodo      = revolve_action(&rctx->check, &rctx->capo, &rctx->fine, rctx->snaps_in, &rctx->info, &rctx->where); /* must return 1 */
    PetscCall(printwhattodo(viewer, whattodo, rctx, shift));
    rctx->stepsleft = rctx->capo - rctx->oldcapo - 1;
  }
  if (whattodo == 2) { /* store a checkpoint to RAM and ask Revolve how many time steps to advance next */
    *store        = 1;
    rctx->oldcapo = rctx->capo;
    if (!toplevel) whattodo = revolve_action(&rctx->check, &rctx->capo, &rctx->fine, rctx->snaps_in, &rctx->info, &rctx->where); /* must return 1 */
    else whattodo = revolve2_action(&rctx->check, &rctx->capo, &rctx->fine, rctx->snaps_in, &rctx->info, &rctx->where);
    if (!toplevel) PetscCall(printwhattodo(viewer, whattodo, rctx, shift));
    else PetscCall(printwhattodo2(viewer, whattodo, rctx, shift));
    if (stype == REVOLVE_ONLINE && rctx->capo >= total_steps - 1) {
      revolve_turn(total_steps, &rctx->capo, &rctx->fine);
      PetscCall(printwhattodo(viewer, whattodo, rctx, shift));
    }
    rctx->stepsleft = rctx->capo - rctx->oldcapo - 1;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSTrajectoryMemorySet_ROF(TSTrajectory tj, TS ts, TJScheduler *tjsch, PetscInt stepnum, PetscReal time, Vec X)
{
  Stack          *stack = &tjsch->stack;
  PetscInt        store;
  StackElement    e;
  PetscRevolveInt rtotal_steps, rstepnum;
  CheckpointType  cptype;

  PetscFunctionBegin;
  if (!stack->solution_only && stepnum == 0) PetscFunctionReturn(PETSC_SUCCESS);
  if (stack->solution_only && stepnum == tjsch->total_steps) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscRevolveIntCast(tjsch->total_steps, &rtotal_steps));
  PetscCall(PetscRevolveIntCast(stepnum, &rstepnum));
  PetscCall(ApplyRevolve(tj->monitor, tjsch->stype, tjsch->rctx, rtotal_steps, rstepnum, rstepnum, PETSC_FALSE, &store));
  if (store == 1) {
    PetscCheck(stepnum >= stack->top, PetscObjectComm((PetscObject)ts), PETSC_ERR_MEMC, "Illegal modification of a non-top stack element");
    cptype = stack->solution_only ? SOLUTIONONLY : SOLUTION_STAGES;
    PetscCall(ElementCreate(ts, cptype, stack, &e));
    PetscCall(ElementSet(ts, stack, &e, stepnum, time, X));
    PetscCall(StackPush(stack, e));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSTrajectoryMemoryGet_ROF(TSTrajectory tj, TS ts, TJScheduler *tjsch, PetscInt stepnum)
{
  Stack          *stack = &tjsch->stack;
  PetscInt        store;
  PetscRevolveInt whattodo, shift, rtotal_steps, rstepnum;
  StackElement    e;

  PetscFunctionBegin;
  if (stepnum == 0 || stepnum == tjsch->total_steps) {
    PetscCall(TurnBackward(ts));
    tjsch->rctx->reverseonestep = PETSC_FALSE;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  /* restore a checkpoint */
  PetscCall(StackTop(stack, &e));
  PetscCall(UpdateTS(ts, stack, e, stepnum, PETSC_TRUE));
  PetscCall(PetscRevolveIntCast(tjsch->total_steps, &rtotal_steps));
  PetscCall(PetscRevolveIntCast(stepnum, &rstepnum));
  if (stack->solution_only) { /* start with restoring a checkpoint */
    tjsch->rctx->capo    = rstepnum;
    tjsch->rctx->oldcapo = tjsch->rctx->capo;
    shift                = 0;
    whattodo             = revolve_action(&tjsch->rctx->check, &tjsch->rctx->capo, &tjsch->rctx->fine, tjsch->rctx->snaps_in, &tjsch->rctx->info, &tjsch->rctx->where);
    PetscCall(printwhattodo(tj->monitor, whattodo, tjsch->rctx, shift));
  } else { /* 2 revolve actions: restore a checkpoint and then advance */
    PetscCall(ApplyRevolve(tj->monitor, tjsch->stype, tjsch->rctx, rtotal_steps, rstepnum, rstepnum, PETSC_FALSE, &store));
    if (tj->monitor) {
      PetscCall(PetscViewerASCIIAddTab(tj->monitor, ((PetscObject)tj)->tablevel));
      PetscCall(PetscViewerASCIIPrintf(tj->monitor, "Skip the step from %d to %d (stage values already checkpointed)\n", tjsch->rctx->oldcapo, tjsch->rctx->oldcapo + 1));
      PetscCall(PetscViewerASCIISubtractTab(tj->monitor, ((PetscObject)tj)->tablevel));
    }
    if (!tjsch->rctx->reverseonestep && tjsch->rctx->stepsleft > 0) tjsch->rctx->stepsleft--;
  }
  if (stack->solution_only || (!stack->solution_only && e->stepnum < stepnum)) {
    PetscCall(TurnForward(ts));
    PetscCall(ReCompute(ts, tjsch, e->stepnum, stepnum));
  }
  if ((stack->solution_only && e->stepnum + 1 == stepnum) || (!stack->solution_only && e->stepnum == stepnum)) PetscCall(StackPop(stack, &e));
  tjsch->rctx->reverseonestep = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSTrajectoryMemorySet_RON(TSTrajectory tj, TS ts, TJScheduler *tjsch, PetscInt stepnum, PetscReal time, Vec X)
{
  Stack          *stack = &tjsch->stack;
  Vec            *Y;
  PetscInt        i, store;
  PetscReal       timeprev;
  StackElement    e;
  RevolveCTX     *rctx = tjsch->rctx;
  PetscRevolveInt rtotal_steps, rstepnum;
  CheckpointType  cptype;

  PetscFunctionBegin;
  if (!stack->solution_only && stepnum == 0) PetscFunctionReturn(PETSC_SUCCESS);
  if (stack->solution_only && stepnum == tjsch->total_steps) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscRevolveIntCast(tjsch->total_steps, &rtotal_steps));
  PetscCall(PetscRevolveIntCast(stepnum, &rstepnum));
  PetscCall(ApplyRevolve(tj->monitor, tjsch->stype, rctx, rtotal_steps, rstepnum, rstepnum, PETSC_FALSE, &store));
  if (store == 1) {
    if (rctx->check != stack->top + 1) { /* overwrite some non-top checkpoint in the stack */
      PetscCall(StackFind(stack, &e, rctx->check));
      if (HaveSolution(e->cptype)) PetscCall(VecCopy(X, e->X));
      if (HaveStages(e->cptype)) {
        PetscCall(TSGetStages(ts, &stack->numY, &Y));
        for (i = 0; i < stack->numY; i++) PetscCall(VecCopy(Y[i], e->Y[i]));
      }
      e->stepnum = stepnum;
      e->time    = time;
      PetscCall(TSGetPrevTime(ts, &timeprev));
      e->timeprev = timeprev;
    } else {
      PetscCheck(stepnum >= stack->top, PetscObjectComm((PetscObject)ts), PETSC_ERR_MEMC, "Illegal modification of a non-top stack element");
      cptype = stack->solution_only ? SOLUTIONONLY : SOLUTION_STAGES;
      PetscCall(ElementCreate(ts, cptype, stack, &e));
      PetscCall(ElementSet(ts, stack, &e, stepnum, time, X));
      PetscCall(StackPush(stack, e));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSTrajectoryMemoryGet_RON(TSTrajectory tj, TS ts, TJScheduler *tjsch, PetscInt stepnum)
{
  Stack          *stack = &tjsch->stack;
  PetscRevolveInt whattodo, shift, rstepnum;
  StackElement    e;

  PetscFunctionBegin;
  if (stepnum == 0 || stepnum == tjsch->total_steps) {
    PetscCall(TurnBackward(ts));
    tjsch->rctx->reverseonestep = PETSC_FALSE;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(PetscRevolveIntCast(stepnum, &rstepnum));
  tjsch->rctx->capo    = rstepnum;
  tjsch->rctx->oldcapo = tjsch->rctx->capo;
  shift                = 0;
  whattodo             = revolve_action(&tjsch->rctx->check, &tjsch->rctx->capo, &tjsch->rctx->fine, tjsch->rctx->snaps_in, &tjsch->rctx->info, &tjsch->rctx->where); /* whattodo=restore */
  if (whattodo == 8) whattodo = 5;
  PetscCall(printwhattodo(tj->monitor, whattodo, tjsch->rctx, shift));
  /* restore a checkpoint */
  PetscCall(StackFind(stack, &e, tjsch->rctx->check));
  PetscCall(UpdateTS(ts, stack, e, stepnum, PETSC_TRUE));
  if (!stack->solution_only) { /* whattodo must be 5 */
    /* ask Revolve what to do next */
    tjsch->rctx->oldcapo = tjsch->rctx->capo;
    whattodo             = revolve_action(&tjsch->rctx->check, &tjsch->rctx->capo, &tjsch->rctx->fine, tjsch->rctx->snaps_in, &tjsch->rctx->info, &tjsch->rctx->where); /* must return 1 or 3 or 4*/
    PetscCall(printwhattodo(tj->monitor, whattodo, tjsch->rctx, shift));
    if (whattodo == 3 || whattodo == 4) tjsch->rctx->reverseonestep = PETSC_TRUE;
    if (whattodo == 1) tjsch->rctx->stepsleft = tjsch->rctx->capo - tjsch->rctx->oldcapo;
    if (tj->monitor) {
      PetscCall(PetscViewerASCIIAddTab(tj->monitor, ((PetscObject)tj)->tablevel));
      PetscCall(PetscViewerASCIIPrintf(tj->monitor, "Skip the step from %d to %d (stage values already checkpointed)\n", tjsch->rctx->oldcapo, tjsch->rctx->oldcapo + 1));
      PetscCall(PetscViewerASCIISubtractTab(tj->monitor, ((PetscObject)tj)->tablevel));
    }
    if (!tjsch->rctx->reverseonestep && tjsch->rctx->stepsleft > 0) tjsch->rctx->stepsleft--;
  }
  if (stack->solution_only || (!stack->solution_only && e->stepnum < stepnum)) {
    PetscCall(TurnForward(ts));
    PetscCall(ReCompute(ts, tjsch, e->stepnum, stepnum));
  }
  tjsch->rctx->reverseonestep = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSTrajectoryMemorySet_TLR(TSTrajectory tj, TS ts, TJScheduler *tjsch, PetscInt stepnum, PetscReal time, Vec X)
{
  Stack          *stack = &tjsch->stack;
  PetscInt        store, localstepnum, laststridesize;
  StackElement    e;
  PetscBool       done = PETSC_FALSE;
  PetscRevolveInt rtotal_steps, rstepnum, rlocalstepnum;
  CheckpointType  cptype;

  PetscFunctionBegin;
  if (!stack->solution_only && stepnum == 0) PetscFunctionReturn(PETSC_SUCCESS);
  if (stack->solution_only && stepnum == tjsch->total_steps) PetscFunctionReturn(PETSC_SUCCESS);

  localstepnum   = stepnum % tjsch->stride;
  laststridesize = tjsch->total_steps % tjsch->stride;
  if (!laststridesize) laststridesize = tjsch->stride;

  if (!tjsch->recompute) {
    PetscCall(TopLevelStore(tj, ts, tjsch, stepnum, localstepnum, laststridesize, &done));
    /* revolve is needed for the last stride; different starting points for last stride between solutin_only and !solutin_only */
    if (!stack->solution_only && !tjsch->save_stack && stepnum <= tjsch->total_steps - laststridesize) PetscFunctionReturn(PETSC_SUCCESS);
    if (stack->solution_only && !tjsch->save_stack && stepnum < tjsch->total_steps - laststridesize) PetscFunctionReturn(PETSC_SUCCESS);
  }
  if (tjsch->save_stack && done) {
    PetscCall(InitRevolve(tjsch->stride, tjsch->max_cps_ram, tjsch->rctx));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  if (laststridesize < tjsch->stride) {
    if (stack->solution_only && stepnum == tjsch->total_steps - laststridesize && !tjsch->recompute) { /* step tjsch->total_steps-laststridesize-1 is skipped, but the next step is not */
      PetscCall(InitRevolve(laststridesize, tjsch->max_cps_ram, tjsch->rctx));
    }
    if (!stack->solution_only && stepnum == tjsch->total_steps - laststridesize + 1 && !tjsch->recompute) { /* step tjsch->total_steps-laststridesize is skipped, but the next step is not */
      PetscCall(InitRevolve(laststridesize, tjsch->max_cps_ram, tjsch->rctx));
    }
  }
  PetscCall(PetscRevolveIntCast(tjsch->total_steps, &rtotal_steps));
  PetscCall(PetscRevolveIntCast(stepnum, &rstepnum));
  PetscCall(PetscRevolveIntCast(localstepnum, &rlocalstepnum));
  PetscCall(ApplyRevolve(tj->monitor, tjsch->stype, tjsch->rctx, rtotal_steps, rstepnum, rlocalstepnum, PETSC_FALSE, &store));
  if (store == 1) {
    PetscCheck(localstepnum >= stack->top, PetscObjectComm((PetscObject)ts), PETSC_ERR_MEMC, "Illegal modification of a non-top stack element");
    cptype = stack->solution_only ? SOLUTIONONLY : SOLUTION_STAGES;
    PetscCall(ElementCreate(ts, cptype, stack, &e));
    PetscCall(ElementSet(ts, stack, &e, stepnum, time, X));
    PetscCall(StackPush(stack, e));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSTrajectoryMemoryGet_TLR(TSTrajectory tj, TS ts, TJScheduler *tjsch, PetscInt stepnum)
{
  Stack          *stack = &tjsch->stack;
  PetscRevolveInt whattodo, shift, rstepnum, rlocalstepnum, rtotal_steps;
  PetscInt        localstepnum, stridenum, laststridesize, store;
  StackElement    e;
  CheckpointType  cptype;

  PetscFunctionBegin;
  localstepnum = stepnum % tjsch->stride;
  stridenum    = stepnum / tjsch->stride;
  if (stepnum == tjsch->total_steps) {
    PetscCall(TurnBackward(ts));
    tjsch->rctx->reverseonestep = PETSC_FALSE;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  laststridesize = tjsch->total_steps % tjsch->stride;
  if (!laststridesize) laststridesize = tjsch->stride;
  PetscCall(PetscRevolveIntCast(tjsch->total_steps, &rtotal_steps));
  PetscCall(PetscRevolveIntCast(stepnum, &rstepnum));
  PetscCall(PetscRevolveIntCast(localstepnum, &rlocalstepnum));
  if (stack->solution_only) {
    /* fill stack */
    if (localstepnum == 0 && stepnum <= tjsch->total_steps - laststridesize) {
      if (tjsch->save_stack) {
        PetscCall(StackLoadAll(tj, ts, stack, stridenum));
        PetscCall(InitRevolve(tjsch->stride, tjsch->max_cps_ram, tjsch->rctx));
        PetscCall(FastForwardRevolve(tjsch->rctx));
        tjsch->skip_trajectory = PETSC_TRUE;
        PetscCall(TurnForward(ts));
        PetscCall(ReCompute(ts, tjsch, stridenum * tjsch->stride - 1, stridenum * tjsch->stride));
        tjsch->skip_trajectory = PETSC_FALSE;
      } else {
        PetscCall(LoadSingle(tj, ts, stack, stridenum));
        PetscCall(InitRevolve(tjsch->stride, tjsch->max_cps_ram, tjsch->rctx));
        PetscCall(TurnForward(ts));
        PetscCall(ReCompute(ts, tjsch, (stridenum - 1) * tjsch->stride, stridenum * tjsch->stride));
      }
      PetscFunctionReturn(PETSC_SUCCESS);
    }
    /* restore a checkpoint */
    PetscCall(StackTop(stack, &e));
    PetscCall(UpdateTS(ts, stack, e, stepnum, PETSC_TRUE));
    /* start with restoring a checkpoint */
    tjsch->rctx->capo    = rstepnum;
    tjsch->rctx->oldcapo = tjsch->rctx->capo;
    shift                = rstepnum - rlocalstepnum;
    whattodo             = revolve_action(&tjsch->rctx->check, &tjsch->rctx->capo, &tjsch->rctx->fine, tjsch->rctx->snaps_in, &tjsch->rctx->info, &tjsch->rctx->where);
    PetscCall(printwhattodo(tj->monitor, whattodo, tjsch->rctx, shift));
    PetscCall(TurnForward(ts));
    PetscCall(ReCompute(ts, tjsch, e->stepnum, stepnum));
    if (e->stepnum + 1 == stepnum) PetscCall(StackPop(stack, &e));
  } else {
    /* fill stack with info */
    if (localstepnum == 0 && tjsch->total_steps - stepnum >= laststridesize) {
      if (tjsch->save_stack) {
        PetscCall(StackLoadAll(tj, ts, stack, stridenum));
        PetscCall(InitRevolve(tjsch->stride, tjsch->max_cps_ram, tjsch->rctx));
        PetscCall(FastForwardRevolve(tjsch->rctx));
      } else {
        PetscRevolveInt rnum;
        PetscCall(LoadSingle(tj, ts, stack, stridenum));
        PetscCall(InitRevolve(tjsch->stride, tjsch->max_cps_ram, tjsch->rctx));
        PetscCall(PetscRevolveIntCast((stridenum - 1) * tjsch->stride + 1, &rnum));
        PetscCall(ApplyRevolve(tj->monitor, tjsch->stype, tjsch->rctx, rtotal_steps, rnum, 1, PETSC_FALSE, &store));
        if (tj->monitor) {
          PetscCall(PetscViewerASCIIAddTab(tj->monitor, ((PetscObject)tj)->tablevel));
          PetscCall(
            PetscViewerASCIIPrintf(tj->monitor, "Skip the step from %" PetscInt_FMT " to %" PetscInt_FMT " (stage values already checkpointed)\n", (stridenum - 1) * tjsch->stride + tjsch->rctx->oldcapo, (stridenum - 1) * tjsch->stride + tjsch->rctx->oldcapo + 1));
          PetscCall(PetscViewerASCIISubtractTab(tj->monitor, ((PetscObject)tj)->tablevel));
        }
        cptype = SOLUTION_STAGES;
        PetscCall(ElementCreate(ts, cptype, stack, &e));
        PetscCall(ElementSet(ts, stack, &e, (stridenum - 1) * tjsch->stride + 1, ts->ptime, ts->vec_sol));
        PetscCall(StackPush(stack, e));
        PetscCall(TurnForward(ts));
        PetscCall(ReCompute(ts, tjsch, e->stepnum, stridenum * tjsch->stride));
      }
      PetscFunctionReturn(PETSC_SUCCESS);
    }
    /* restore a checkpoint */
    PetscCall(StackTop(stack, &e));
    PetscCall(UpdateTS(ts, stack, e, stepnum, PETSC_TRUE));
    /* 2 revolve actions: restore a checkpoint and then advance */
    PetscCall(ApplyRevolve(tj->monitor, tjsch->stype, tjsch->rctx, rtotal_steps, rstepnum, rlocalstepnum, PETSC_FALSE, &store));
    if (tj->monitor) {
      PetscCall(PetscViewerASCIIAddTab(tj->monitor, ((PetscObject)tj)->tablevel));
      PetscCall(PetscViewerASCIIPrintf(tj->monitor, "Skip the step from %" PetscInt_FMT " to %" PetscInt_FMT " (stage values already checkpointed)\n", stepnum - localstepnum + tjsch->rctx->oldcapo, stepnum - localstepnum + tjsch->rctx->oldcapo + 1));
      PetscCall(PetscViewerASCIISubtractTab(tj->monitor, ((PetscObject)tj)->tablevel));
    }
    if (!tjsch->rctx->reverseonestep && tjsch->rctx->stepsleft > 0) tjsch->rctx->stepsleft--;
    if (e->stepnum < stepnum) {
      PetscCall(TurnForward(ts));
      PetscCall(ReCompute(ts, tjsch, e->stepnum, stepnum));
    }
    if (e->stepnum == stepnum) PetscCall(StackPop(stack, &e));
  }
  tjsch->rctx->reverseonestep = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSTrajectoryMemorySet_TLTR(TSTrajectory tj, TS ts, TJScheduler *tjsch, PetscInt stepnum, PetscReal time, Vec X)
{
  Stack          *stack = &tjsch->stack;
  PetscInt        store, localstepnum, stridenum, laststridesize;
  StackElement    e;
  PetscBool       done = PETSC_FALSE;
  PetscRevolveInt rlocalstepnum, rstepnum, rtotal_steps;

  PetscFunctionBegin;
  if (!stack->solution_only && stepnum == 0) PetscFunctionReturn(PETSC_SUCCESS);
  if (stack->solution_only && stepnum == tjsch->total_steps) PetscFunctionReturn(PETSC_SUCCESS);

  localstepnum   = stepnum % tjsch->stride; /* index at the bottom level (inside a stride) */
  stridenum      = stepnum / tjsch->stride; /* index at the top level */
  laststridesize = tjsch->total_steps % tjsch->stride;
  if (!laststridesize) laststridesize = tjsch->stride;
  if (stack->solution_only && localstepnum == 0 && !tjsch->rctx2->reverseonestep) {
    PetscCall(PetscRevolveIntCast((tjsch->total_steps + tjsch->stride - 1) / tjsch->stride, &rtotal_steps));
    PetscCall(PetscRevolveIntCast(stridenum, &rstepnum));
    PetscCall(ApplyRevolve(tj->monitor, tjsch->stype, tjsch->rctx2, rtotal_steps, rstepnum, rstepnum, PETSC_TRUE, &tjsch->store_stride));
    if (laststridesize < tjsch->stride && stepnum == tjsch->total_steps - laststridesize) PetscCall(InitRevolve(laststridesize, tjsch->max_cps_ram, tjsch->rctx));
  }
  if (!stack->solution_only && localstepnum == 1 && !tjsch->rctx2->reverseonestep) {
    PetscCall(PetscRevolveIntCast((tjsch->total_steps + tjsch->stride - 1) / tjsch->stride, &rtotal_steps));
    PetscCall(PetscRevolveIntCast(stridenum, &rstepnum));
    PetscCall(ApplyRevolve(tj->monitor, tjsch->stype, tjsch->rctx2, rtotal_steps, rstepnum, rstepnum, PETSC_TRUE, &tjsch->store_stride));
    if (laststridesize < tjsch->stride && stepnum == tjsch->total_steps - laststridesize + 1) PetscCall(InitRevolve(laststridesize, tjsch->max_cps_ram, tjsch->rctx));
  }
  if (tjsch->store_stride) {
    PetscCall(TopLevelStore(tj, ts, tjsch, stepnum, localstepnum, laststridesize, &done));
    if (done) {
      PetscCall(InitRevolve(tjsch->stride, tjsch->max_cps_ram, tjsch->rctx));
      PetscFunctionReturn(PETSC_SUCCESS);
    }
  }
  if (stepnum < tjsch->total_steps - laststridesize) {
    if (tjsch->save_stack && !tjsch->store_stride && !tjsch->rctx2->reverseonestep) PetscFunctionReturn(PETSC_SUCCESS); /* store or forward-and-reverse at top level trigger revolve at bottom level */
    if (!tjsch->save_stack && !tjsch->rctx2->reverseonestep) PetscFunctionReturn(PETSC_SUCCESS);                        /* store operation does not require revolve be called at bottom level */
  }
  /* Skipping stepnum=0 for !stack->only is enough for TLR, but not for TLTR. Here we skip the first step for each stride so that the top-level revolve is applied (always at localstepnum=1) ahead of the bottom-level revolve */
  if (!stack->solution_only && localstepnum == 0 && stepnum != tjsch->total_steps && !tjsch->recompute) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscRevolveIntCast(tjsch->total_steps, &rtotal_steps));
  PetscCall(PetscRevolveIntCast(stepnum, &rstepnum));
  PetscCall(PetscRevolveIntCast(localstepnum, &rlocalstepnum));
  PetscCall(ApplyRevolve(tj->monitor, tjsch->stype, tjsch->rctx, rtotal_steps, rstepnum, rlocalstepnum, PETSC_FALSE, &store));
  if (store == 1) {
    CheckpointType cptype;
    PetscCheck(localstepnum >= stack->top, PetscObjectComm((PetscObject)ts), PETSC_ERR_MEMC, "Illegal modification of a non-top stack element");
    cptype = stack->solution_only ? SOLUTIONONLY : SOLUTION_STAGES;
    PetscCall(ElementCreate(ts, cptype, stack, &e));
    PetscCall(ElementSet(ts, stack, &e, stepnum, time, X));
    PetscCall(StackPush(stack, e));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSTrajectoryMemoryGet_TLTR(TSTrajectory tj, TS ts, TJScheduler *tjsch, PetscInt stepnum)
{
  Stack          *stack     = &tjsch->stack;
  DiskStack      *diskstack = &tjsch->diskstack;
  PetscInt        localstepnum, stridenum, restoredstridenum, laststridesize, store;
  StackElement    e;
  PetscRevolveInt whattodo, shift;
  PetscRevolveInt rtotal_steps, rstepnum, rlocalstepnum;

  PetscFunctionBegin;
  localstepnum = stepnum % tjsch->stride;
  stridenum    = stepnum / tjsch->stride;
  if (stepnum == tjsch->total_steps) {
    PetscCall(TurnBackward(ts));
    tjsch->rctx->reverseonestep = PETSC_FALSE;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  laststridesize = tjsch->total_steps % tjsch->stride;
  if (!laststridesize) laststridesize = tjsch->stride;
  /*
   Last stride can be adjoined directly. All the other strides require that the stack in memory be ready before an adjoint step is taken (at the end of each stride). The following two cases need to be addressed differently:
     Case 1 (save_stack)
       Restore a disk checkpoint; update TS with the last element in the restored data; recompute to the current point.
     Case 2 (!save_stack)
       Restore a disk checkpoint; update TS with the restored point; recompute to the current point.
  */
  if (localstepnum == 0 && stepnum <= tjsch->total_steps - laststridesize) {
    /* restore the top element in the stack for disk checkpoints */
    restoredstridenum            = diskstack->container[diskstack->top];
    tjsch->rctx2->reverseonestep = PETSC_FALSE;
    /* top-level revolve must be applied before current step, just like the solution_only mode for single-level revolve */
    if (!tjsch->save_stack && stack->solution_only) { /* start with restoring a checkpoint */
      PetscCall(PetscRevolveIntCast(stridenum, &rstepnum));
      tjsch->rctx2->capo    = rstepnum;
      tjsch->rctx2->oldcapo = tjsch->rctx2->capo;
      shift                 = 0;
      whattodo              = revolve2_action(&tjsch->rctx2->check, &tjsch->rctx2->capo, &tjsch->rctx2->fine, tjsch->rctx2->snaps_in, &tjsch->rctx2->info, &tjsch->rctx2->where);
      PetscCall(printwhattodo2(tj->monitor, whattodo, tjsch->rctx2, shift));
    } else { /* 2 revolve actions: restore a checkpoint and then advance */
      PetscCall(PetscRevolveIntCast((tjsch->total_steps + tjsch->stride - 1) / tjsch->stride, &rtotal_steps));
      PetscCall(PetscRevolveIntCast(stridenum, &rstepnum));
      PetscCall(ApplyRevolve(tj->monitor, tjsch->stype, tjsch->rctx2, rtotal_steps, rstepnum, rstepnum, PETSC_TRUE, &tjsch->store_stride));
      if (tj->monitor) {
        PetscCall(PetscViewerASCIIAddTab(tj->monitor, ((PetscObject)tj)->tablevel));
        PetscCall(PetscViewerASCIIPrintf(tj->monitor, "[Top Level] Skip the stride from %d to %d (stage values already checkpointed)\n", tjsch->rctx2->oldcapo, tjsch->rctx2->oldcapo + 1));
        PetscCall(PetscViewerASCIISubtractTab(tj->monitor, ((PetscObject)tj)->tablevel));
      }
      if (!tjsch->rctx2->reverseonestep && tjsch->rctx2->stepsleft > 0) tjsch->rctx2->stepsleft--;
    }
    /* fill stack */
    if (stack->solution_only) {
      if (tjsch->save_stack) {
        if (restoredstridenum < stridenum) {
          PetscCall(StackLoadLast(tj, ts, stack, restoredstridenum));
        } else {
          PetscCall(StackLoadAll(tj, ts, stack, restoredstridenum));
        }
        /* recompute one step ahead */
        tjsch->skip_trajectory = PETSC_TRUE;
        PetscCall(TurnForward(ts));
        PetscCall(ReCompute(ts, tjsch, stridenum * tjsch->stride - 1, stridenum * tjsch->stride));
        tjsch->skip_trajectory = PETSC_FALSE;
        if (restoredstridenum < stridenum) {
          PetscCall(InitRevolve(tjsch->stride, tjsch->max_cps_ram, tjsch->rctx));
          PetscCall(TurnForward(ts));
          PetscCall(ReCompute(ts, tjsch, restoredstridenum * tjsch->stride, stepnum));
        } else { /* stack ready, fast forward revolve status */
          PetscCall(InitRevolve(tjsch->stride, tjsch->max_cps_ram, tjsch->rctx));
          PetscCall(FastForwardRevolve(tjsch->rctx));
        }
      } else {
        PetscCall(LoadSingle(tj, ts, stack, restoredstridenum));
        PetscCall(InitRevolve(tjsch->stride, tjsch->max_cps_ram, tjsch->rctx));
        PetscCall(TurnForward(ts));
        PetscCall(ReCompute(ts, tjsch, (restoredstridenum - 1) * tjsch->stride, stepnum));
      }
    } else {
      if (tjsch->save_stack) {
        if (restoredstridenum < stridenum) {
          PetscCall(StackLoadLast(tj, ts, stack, restoredstridenum));
          /* reset revolve */
          PetscCall(InitRevolve(tjsch->stride, tjsch->max_cps_ram, tjsch->rctx));
          PetscCall(TurnForward(ts));
          PetscCall(ReCompute(ts, tjsch, restoredstridenum * tjsch->stride, stepnum));
        } else { /* stack ready, fast forward revolve status */
          PetscCall(StackLoadAll(tj, ts, stack, restoredstridenum));
          PetscCall(InitRevolve(tjsch->stride, tjsch->max_cps_ram, tjsch->rctx));
          PetscCall(FastForwardRevolve(tjsch->rctx));
        }
      } else {
        PetscCall(LoadSingle(tj, ts, stack, restoredstridenum));
        PetscCall(InitRevolve(tjsch->stride, tjsch->max_cps_ram, tjsch->rctx));
        /* push first element to stack */
        if (tjsch->store_stride || tjsch->rctx2->reverseonestep) {
          CheckpointType cptype = SOLUTION_STAGES;
          shift                 = (restoredstridenum - 1) * tjsch->stride - localstepnum;
          PetscCall(PetscRevolveIntCast(tjsch->total_steps, &rtotal_steps));
          PetscCall(PetscRevolveIntCast((restoredstridenum - 1) * tjsch->stride + 1, &rstepnum));
          PetscCall(ApplyRevolve(tj->monitor, tjsch->stype, tjsch->rctx, rtotal_steps, rstepnum, 1, PETSC_FALSE, &store));
          if (tj->monitor) {
            PetscCall(PetscViewerASCIIAddTab(tj->monitor, ((PetscObject)tj)->tablevel));
            PetscCall(PetscViewerASCIIPrintf(tj->monitor, "Skip the step from %" PetscInt_FMT " to %" PetscInt_FMT " (stage values already checkpointed)\n", (restoredstridenum - 1) * tjsch->stride, (restoredstridenum - 1) * tjsch->stride + 1));
            PetscCall(PetscViewerASCIISubtractTab(tj->monitor, ((PetscObject)tj)->tablevel));
          }
          PetscCall(ElementCreate(ts, cptype, stack, &e));
          PetscCall(ElementSet(ts, stack, &e, (restoredstridenum - 1) * tjsch->stride + 1, ts->ptime, ts->vec_sol));
          PetscCall(StackPush(stack, e));
        }
        PetscCall(TurnForward(ts));
        PetscCall(ReCompute(ts, tjsch, (restoredstridenum - 1) * tjsch->stride + 1, stepnum));
      }
    }
    if (restoredstridenum == stridenum) diskstack->top--;
    tjsch->rctx->reverseonestep = PETSC_FALSE;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  if (stack->solution_only) {
    /* restore a checkpoint */
    PetscCall(StackTop(stack, &e));
    PetscCall(UpdateTS(ts, stack, e, stepnum, PETSC_TRUE));
    /* start with restoring a checkpoint */
    PetscCall(PetscRevolveIntCast(stepnum, &rstepnum));
    PetscCall(PetscRevolveIntCast(localstepnum, &rlocalstepnum));
    tjsch->rctx->capo    = rstepnum;
    tjsch->rctx->oldcapo = tjsch->rctx->capo;
    shift                = rstepnum - rlocalstepnum;
    whattodo             = revolve_action(&tjsch->rctx->check, &tjsch->rctx->capo, &tjsch->rctx->fine, tjsch->rctx->snaps_in, &tjsch->rctx->info, &tjsch->rctx->where);
    PetscCall(printwhattodo(tj->monitor, whattodo, tjsch->rctx, shift));
    PetscCall(TurnForward(ts));
    PetscCall(ReCompute(ts, tjsch, e->stepnum, stepnum));
    if (e->stepnum + 1 == stepnum) PetscCall(StackPop(stack, &e));
  } else {
    PetscRevolveInt rlocalstepnum;
    /* restore a checkpoint */
    PetscCall(StackTop(stack, &e));
    PetscCall(UpdateTS(ts, stack, e, stepnum, PETSC_TRUE));
    /* 2 revolve actions: restore a checkpoint and then advance */
    PetscCall(PetscRevolveIntCast(tjsch->total_steps, &rtotal_steps));
    PetscCall(PetscRevolveIntCast(stridenum, &rstepnum));
    PetscCall(PetscRevolveIntCast(localstepnum, &rlocalstepnum));
    PetscCall(ApplyRevolve(tj->monitor, tjsch->stype, tjsch->rctx, rtotal_steps, rstepnum, rlocalstepnum, PETSC_FALSE, &store));
    if (tj->monitor) {
      PetscCall(PetscViewerASCIIAddTab(tj->monitor, ((PetscObject)tj)->tablevel));
      PetscCall(PetscViewerASCIIPrintf(tj->monitor, "Skip the step from %" PetscInt_FMT " to %" PetscInt_FMT " (stage values already checkpointed)\n", stepnum - localstepnum + tjsch->rctx->oldcapo, stepnum - localstepnum + tjsch->rctx->oldcapo + 1));
      PetscCall(PetscViewerASCIISubtractTab(tj->monitor, ((PetscObject)tj)->tablevel));
    }
    if (!tjsch->rctx->reverseonestep && tjsch->rctx->stepsleft > 0) tjsch->rctx->stepsleft--;
    if (e->stepnum < stepnum) {
      PetscCall(TurnForward(ts));
      PetscCall(ReCompute(ts, tjsch, e->stepnum, stepnum));
    }
    if (e->stepnum == stepnum) PetscCall(StackPop(stack, &e));
  }
  tjsch->rctx->reverseonestep = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSTrajectoryMemorySet_RMS(TSTrajectory tj, TS ts, TJScheduler *tjsch, PetscInt stepnum, PetscReal time, Vec X)
{
  Stack          *stack = &tjsch->stack;
  PetscInt        store;
  StackElement    e;
  PetscRevolveInt rtotal_steps, rstepnum;

  PetscFunctionBegin;
  if (!stack->solution_only && stepnum == 0) PetscFunctionReturn(PETSC_SUCCESS);
  if (stack->solution_only && stepnum == tjsch->total_steps) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscRevolveIntCast(tjsch->total_steps, &rtotal_steps));
  PetscCall(PetscRevolveIntCast(stepnum, &rstepnum));
  PetscCall(ApplyRevolve(tj->monitor, tjsch->stype, tjsch->rctx, rtotal_steps, rstepnum, rstepnum, PETSC_FALSE, &store));
  if (store == 1) {
    CheckpointType cptype;
    PetscCheck(stepnum >= stack->top, PetscObjectComm((PetscObject)ts), PETSC_ERR_MEMC, "Illegal modification of a non-top stack element");
    cptype = stack->solution_only ? SOLUTIONONLY : SOLUTION_STAGES;
    PetscCall(ElementCreate(ts, cptype, stack, &e));
    PetscCall(ElementSet(ts, stack, &e, stepnum, time, X));
    PetscCall(StackPush(stack, e));
  } else if (store == 2) {
    PetscCall(DumpSingle(tj, ts, stack, tjsch->rctx->check + 1));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSTrajectoryMemoryGet_RMS(TSTrajectory tj, TS ts, TJScheduler *tjsch, PetscInt stepnum)
{
  Stack          *stack = &tjsch->stack;
  PetscRevolveInt whattodo, shift, rstepnum;
  PetscInt        restart;
  PetscBool       ondisk;
  StackElement    e;

  PetscFunctionBegin;
  if (stepnum == 0 || stepnum == tjsch->total_steps) {
    PetscCall(TurnBackward(ts));
    tjsch->rctx->reverseonestep = PETSC_FALSE;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(PetscRevolveIntCast(stepnum, &rstepnum));
  tjsch->rctx->capo    = rstepnum;
  tjsch->rctx->oldcapo = tjsch->rctx->capo;
  shift                = 0;
  whattodo             = revolve_action(&tjsch->rctx->check, &tjsch->rctx->capo, &tjsch->rctx->fine, tjsch->rctx->snaps_in, &tjsch->rctx->info, &tjsch->rctx->where); /* whattodo=restore */
  PetscCall(printwhattodo(tj->monitor, whattodo, tjsch->rctx, shift));
  /* restore a checkpoint */
  restart = tjsch->rctx->capo;
  if (!tjsch->rctx->where) {
    ondisk = PETSC_TRUE;
    PetscCall(LoadSingle(tj, ts, stack, tjsch->rctx->check + 1));
    PetscCall(TurnBackward(ts));
  } else {
    ondisk = PETSC_FALSE;
    PetscCall(StackTop(stack, &e));
    PetscCall(UpdateTS(ts, stack, e, stepnum, PETSC_TRUE));
  }
  if (!stack->solution_only) { /* whattodo must be 5 or 8 */
    /* ask Revolve what to do next */
    tjsch->rctx->oldcapo = tjsch->rctx->capo;
    whattodo             = revolve_action(&tjsch->rctx->check, &tjsch->rctx->capo, &tjsch->rctx->fine, tjsch->rctx->snaps_in, &tjsch->rctx->info, &tjsch->rctx->where); /* must return 1 or 3 or 4*/
    PetscCall(printwhattodo(tj->monitor, whattodo, tjsch->rctx, shift));
    if (whattodo == 3 || whattodo == 4) tjsch->rctx->reverseonestep = PETSC_TRUE;
    if (whattodo == 1) tjsch->rctx->stepsleft = tjsch->rctx->capo - tjsch->rctx->oldcapo;
    if (tj->monitor) {
      PetscCall(PetscViewerASCIIAddTab(tj->monitor, ((PetscObject)tj)->tablevel));
      PetscCall(PetscViewerASCIIPrintf(tj->monitor, "Skip the step from %d to %d (stage values already checkpointed)\n", tjsch->rctx->oldcapo, tjsch->rctx->oldcapo + 1));
      PetscCall(PetscViewerASCIISubtractTab(tj->monitor, ((PetscObject)tj)->tablevel));
    }
    if (!tjsch->rctx->reverseonestep && tjsch->rctx->stepsleft > 0) tjsch->rctx->stepsleft--;
    restart++; /* skip one step */
  }
  if (stack->solution_only || (!stack->solution_only && restart < stepnum)) {
    PetscCall(TurnForward(ts));
    PetscCall(ReCompute(ts, tjsch, restart, stepnum));
  }
  if (!ondisk && ((stack->solution_only && e->stepnum + 1 == stepnum) || (!stack->solution_only && e->stepnum == stepnum))) PetscCall(StackPop(stack, &e));
  tjsch->rctx->reverseonestep = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif

#if defined(PETSC_HAVE_CAMS)
/* Optimal offline adjoint checkpointing for multistage time integration methods */
static PetscErrorCode TSTrajectoryMemorySet_AOF(TSTrajectory tj, TS ts, TJScheduler *tjsch, PetscInt stepnum, PetscReal time, Vec X)
{
  Stack       *stack = &tjsch->stack;
  StackElement e;

  PetscFunctionBegin;
  /* skip if no checkpoint to use. This also avoids an error when num_units_avail=0  */
  if (tjsch->actx->nextcheckpointstep == -1) PetscFunctionReturn(PETSC_SUCCESS);
  if (stepnum == 0) { /* When placing the first checkpoint, no need to change the units available */
    if (stack->solution_only) {
      PetscCallExternal(offline_ca, tjsch->actx->lastcheckpointstep, tjsch->actx->num_units_avail, tjsch->actx->endstep, &tjsch->actx->nextcheckpointstep);
    } else {
      /* First two arguments must be -1 when first time calling cams */
      PetscCallExternal(offline_cams, tjsch->actx->lastcheckpointstep, tjsch->actx->lastcheckpointtype, tjsch->actx->num_units_avail, tjsch->actx->endstep, tjsch->actx->num_stages, &tjsch->actx->nextcheckpointstep, &tjsch->actx->nextcheckpointtype);
    }
  }

  if (stack->solution_only && stepnum == tjsch->total_steps) PetscFunctionReturn(PETSC_SUCCESS);

  if (tjsch->actx->nextcheckpointstep == stepnum) {
    PetscCheck(stepnum >= stack->top, PetscObjectComm((PetscObject)ts), PETSC_ERR_MEMC, "Illegal modification of a non-top stack element");

    if (tjsch->actx->nextcheckpointtype == 2) { /* solution + stage values */
      if (tj->monitor) PetscCall(PetscViewerASCIIPrintf(tj->monitor, "Store in checkpoint number %" PetscInt_FMT " with stage values and solution (located in RAM)\n", stepnum));
      PetscCall(ElementCreate(ts, SOLUTION_STAGES, stack, &e));
      PetscCall(ElementSet(ts, stack, &e, stepnum, time, X));
    }
    if (tjsch->actx->nextcheckpointtype == 1) {
      if (tj->monitor) PetscCall(PetscViewerASCIIPrintf(tj->monitor, "Store in checkpoint number %" PetscInt_FMT " with stage values (located in RAM)\n", stepnum));
      PetscCall(ElementCreate(ts, STAGESONLY, stack, &e));
      PetscCall(ElementSet(ts, stack, &e, stepnum, time, X));
    }
    if (tjsch->actx->nextcheckpointtype == 0) { /* solution only */
      if (tj->monitor) PetscCall(PetscViewerASCIIPrintf(tj->monitor, "Store in checkpoint number %" PetscInt_FMT " (located in RAM)\n", stepnum));
      PetscCall(ElementCreate(ts, SOLUTIONONLY, stack, &e));
      PetscCall(ElementSet(ts, stack, &e, stepnum, time, X));
    }
    PetscCall(StackPush(stack, e));

    tjsch->actx->lastcheckpointstep = stepnum;
    if (stack->solution_only) {
      PetscCallExternal(offline_ca, tjsch->actx->lastcheckpointstep, tjsch->actx->num_units_avail, tjsch->actx->endstep, &tjsch->actx->nextcheckpointstep);
      tjsch->actx->num_units_avail--;
    } else {
      tjsch->actx->lastcheckpointtype = tjsch->actx->nextcheckpointtype;
      PetscCallExternal(offline_cams, tjsch->actx->lastcheckpointstep, tjsch->actx->lastcheckpointtype, tjsch->actx->num_units_avail, tjsch->actx->endstep, tjsch->actx->num_stages, &tjsch->actx->nextcheckpointstep, &tjsch->actx->nextcheckpointtype);
      if (tjsch->actx->lastcheckpointtype == 2) tjsch->actx->num_units_avail -= tjsch->actx->num_stages + 1;
      if (tjsch->actx->lastcheckpointtype == 1) tjsch->actx->num_units_avail -= tjsch->actx->num_stages;
      if (tjsch->actx->lastcheckpointtype == 0) tjsch->actx->num_units_avail--;
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSTrajectoryMemoryGet_AOF(TSTrajectory tj, TS ts, TJScheduler *tjsch, PetscInt stepnum)
{
  Stack       *stack = &tjsch->stack;
  StackElement e;
  PetscInt     estepnum;

  PetscFunctionBegin;
  if (stepnum == 0 || stepnum == tjsch->total_steps) {
    PetscCall(TurnBackward(ts));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  /* Restore a checkpoint */
  PetscCall(StackTop(stack, &e));
  estepnum = e->stepnum;
  if (estepnum == stepnum && e->cptype == SOLUTIONONLY) { /* discard the checkpoint if not useful (corner case) */
    PetscCall(StackPop(stack, &e));
    tjsch->actx->num_units_avail++;
    PetscCall(StackTop(stack, &e));
    estepnum = e->stepnum;
  }
  /* Update TS with stage values if an adjoint step can be taken immediately */
  if (HaveStages(e->cptype)) {
    if (tj->monitor) PetscCall(PetscViewerASCIIPrintf(tj->monitor, "Restore in checkpoint number %" PetscInt_FMT " with stage values (located in RAM)\n", e->stepnum));
    if (e->cptype == STAGESONLY) tjsch->actx->num_units_avail += tjsch->actx->num_stages;
    if (e->cptype == SOLUTION_STAGES) tjsch->actx->num_units_avail += tjsch->actx->num_stages + 1;
  } else {
    if (tj->monitor) PetscCall(PetscViewerASCIIPrintf(tj->monitor, "Restore in checkpoint number %" PetscInt_FMT " (located in RAM)\n", e->stepnum));
    tjsch->actx->num_units_avail++;
  }
  PetscCall(UpdateTS(ts, stack, e, stepnum, PETSC_TRUE));
  /* Query the scheduler */
  tjsch->actx->lastcheckpointstep = estepnum;
  tjsch->actx->endstep            = stepnum;
  if (stack->solution_only) { /* start with restoring a checkpoint */
    PetscCallExternal(offline_ca, tjsch->actx->lastcheckpointstep, tjsch->actx->num_units_avail, tjsch->actx->endstep, &tjsch->actx->nextcheckpointstep);
  } else { /* 2 revolve actions: restore a checkpoint and then advance */
    tjsch->actx->lastcheckpointtype = e->cptype;
    PetscCallExternal(offline_cams, tjsch->actx->lastcheckpointstep, tjsch->actx->lastcheckpointtype, tjsch->actx->num_units_avail, tjsch->actx->endstep, tjsch->actx->num_stages, &tjsch->actx->nextcheckpointstep, &tjsch->actx->nextcheckpointtype);
  }
  /* Discard the checkpoint if not needed, decrease the number of available checkpoints if it still stays in stack */
  if (HaveStages(e->cptype)) {
    if (estepnum == stepnum) {
      PetscCall(StackPop(stack, &e));
    } else {
      if (e->cptype == STAGESONLY) tjsch->actx->num_units_avail -= tjsch->actx->num_stages;
      if (e->cptype == SOLUTION_STAGES) tjsch->actx->num_units_avail -= tjsch->actx->num_stages + 1;
    }
  } else {
    if (estepnum + 1 == stepnum) {
      PetscCall(StackPop(stack, &e));
    } else {
      tjsch->actx->num_units_avail--;
    }
  }
  /* Recompute from the restored checkpoint */
  if (stack->solution_only || (!stack->solution_only && estepnum < stepnum)) {
    PetscCall(TurnForward(ts));
    PetscCall(ReCompute(ts, tjsch, estepnum, stepnum));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif

static PetscErrorCode TSTrajectorySet_Memory(TSTrajectory tj, TS ts, PetscInt stepnum, PetscReal time, Vec X)
{
  TJScheduler *tjsch = (TJScheduler *)tj->data;

  PetscFunctionBegin;
  if (!tjsch->recompute) { /* use global stepnum in the forward sweep */
    PetscCall(TSGetStepNumber(ts, &stepnum));
  }
  /* for consistency */
  if (!tjsch->recompute && stepnum == 0) ts->ptime_prev = ts->ptime - ts->time_step;
  switch (tjsch->stype) {
  case NONE:
    if (tj->adjoint_solve_mode) {
      PetscCall(TSTrajectoryMemorySet_N(ts, tjsch, stepnum, time, X));
    } else {
      PetscCall(TSTrajectoryMemorySet_N_2(ts, tjsch, stepnum, time, X));
    }
    break;
  case TWO_LEVEL_NOREVOLVE:
    PetscCheck(tj->adjoint_solve_mode, PetscObjectComm((PetscObject)tj), PETSC_ERR_SUP, "Not implemented");
    PetscCall(TSTrajectoryMemorySet_TLNR(tj, ts, tjsch, stepnum, time, X));
    break;
#if defined(PETSC_HAVE_REVOLVE)
  case TWO_LEVEL_REVOLVE:
    PetscCheck(tj->adjoint_solve_mode, PetscObjectComm((PetscObject)tj), PETSC_ERR_SUP, "Not implemented");
    PetscCall(TSTrajectoryMemorySet_TLR(tj, ts, tjsch, stepnum, time, X));
    break;
  case TWO_LEVEL_TWO_REVOLVE:
    PetscCheck(tj->adjoint_solve_mode, PetscObjectComm((PetscObject)tj), PETSC_ERR_SUP, "Not implemented");
    PetscCall(TSTrajectoryMemorySet_TLTR(tj, ts, tjsch, stepnum, time, X));
    break;
  case REVOLVE_OFFLINE:
    PetscCheck(tj->adjoint_solve_mode, PetscObjectComm((PetscObject)tj), PETSC_ERR_SUP, "Not implemented");
    PetscCall(TSTrajectoryMemorySet_ROF(tj, ts, tjsch, stepnum, time, X));
    break;
  case REVOLVE_ONLINE:
    PetscCheck(tj->adjoint_solve_mode, PetscObjectComm((PetscObject)tj), PETSC_ERR_SUP, "Not implemented");
    PetscCall(TSTrajectoryMemorySet_RON(tj, ts, tjsch, stepnum, time, X));
    break;
  case REVOLVE_MULTISTAGE:
    PetscCheck(tj->adjoint_solve_mode, PetscObjectComm((PetscObject)tj), PETSC_ERR_SUP, "Not implemented");
    PetscCall(TSTrajectoryMemorySet_RMS(tj, ts, tjsch, stepnum, time, X));
    break;
#endif
#if defined(PETSC_HAVE_CAMS)
  case CAMS_OFFLINE:
    PetscCheck(tj->adjoint_solve_mode, PetscObjectComm((PetscObject)tj), PETSC_ERR_SUP, "Not implemented");
    PetscCall(TSTrajectoryMemorySet_AOF(tj, ts, tjsch, stepnum, time, X));
    break;
#endif
  default:
    break;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSTrajectoryGet_Memory(TSTrajectory tj, TS ts, PetscInt stepnum, PetscReal *t)
{
  TJScheduler *tjsch = (TJScheduler *)tj->data;

  PetscFunctionBegin;
  if (tj->adjoint_solve_mode && stepnum == 0) {
    PetscCall(TSTrajectoryReset(tj)); /* reset TSTrajectory so users do not need to reset TSTrajectory */
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  switch (tjsch->stype) {
  case NONE:
    if (tj->adjoint_solve_mode) {
      PetscCall(TSTrajectoryMemoryGet_N(ts, tjsch, stepnum));
    } else {
      PetscCall(TSTrajectoryMemoryGet_N_2(ts, tjsch, stepnum));
    }
    break;
  case TWO_LEVEL_NOREVOLVE:
    PetscCheck(tj->adjoint_solve_mode, PetscObjectComm((PetscObject)tj), PETSC_ERR_SUP, "Not implemented");
    PetscCall(TSTrajectoryMemoryGet_TLNR(tj, ts, tjsch, stepnum));
    break;
#if defined(PETSC_HAVE_REVOLVE)
  case TWO_LEVEL_REVOLVE:
    PetscCheck(tj->adjoint_solve_mode, PetscObjectComm((PetscObject)tj), PETSC_ERR_SUP, "Not implemented");
    PetscCall(TSTrajectoryMemoryGet_TLR(tj, ts, tjsch, stepnum));
    break;
  case TWO_LEVEL_TWO_REVOLVE:
    PetscCheck(tj->adjoint_solve_mode, PetscObjectComm((PetscObject)tj), PETSC_ERR_SUP, "Not implemented");
    PetscCall(TSTrajectoryMemoryGet_TLTR(tj, ts, tjsch, stepnum));
    break;
  case REVOLVE_OFFLINE:
    PetscCheck(tj->adjoint_solve_mode, PetscObjectComm((PetscObject)tj), PETSC_ERR_SUP, "Not implemented");
    PetscCall(TSTrajectoryMemoryGet_ROF(tj, ts, tjsch, stepnum));
    break;
  case REVOLVE_ONLINE:
    PetscCheck(tj->adjoint_solve_mode, PetscObjectComm((PetscObject)tj), PETSC_ERR_SUP, "Not implemented");
    PetscCall(TSTrajectoryMemoryGet_RON(tj, ts, tjsch, stepnum));
    break;
  case REVOLVE_MULTISTAGE:
    PetscCheck(tj->adjoint_solve_mode, PetscObjectComm((PetscObject)tj), PETSC_ERR_SUP, "Not implemented");
    PetscCall(TSTrajectoryMemoryGet_RMS(tj, ts, tjsch, stepnum));
    break;
#endif
#if defined(PETSC_HAVE_CAMS)
  case CAMS_OFFLINE:
    PetscCheck(tj->adjoint_solve_mode, PetscObjectComm((PetscObject)tj), PETSC_ERR_SUP, "Not implemented");
    PetscCall(TSTrajectoryMemoryGet_AOF(tj, ts, tjsch, stepnum));
    break;
#endif
  default:
    break;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_UNUSED static PetscErrorCode TSTrajectorySetStride_Memory(TSTrajectory tj, PetscInt stride)
{
  TJScheduler *tjsch = (TJScheduler *)tj->data;

  PetscFunctionBegin;
  tjsch->stride = stride;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSTrajectorySetMaxCpsRAM_Memory(TSTrajectory tj, PetscInt max_cps_ram)
{
  TJScheduler *tjsch = (TJScheduler *)tj->data;

  PetscFunctionBegin;
  tjsch->max_cps_ram = max_cps_ram;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSTrajectorySetMaxCpsDisk_Memory(TSTrajectory tj, PetscInt max_cps_disk)
{
  TJScheduler *tjsch = (TJScheduler *)tj->data;

  PetscFunctionBegin;
  tjsch->max_cps_disk = max_cps_disk;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSTrajectorySetMaxUnitsRAM_Memory(TSTrajectory tj, PetscInt max_units_ram)
{
  TJScheduler *tjsch = (TJScheduler *)tj->data;

  PetscFunctionBegin;
  PetscCheck(tjsch->max_cps_ram, PetscObjectComm((PetscObject)tj), PETSC_ERR_ARG_INCOMP, "Conflict with -ts_trjaectory_max_cps_ram or TSTrajectorySetMaxCpsRAM. You can set max_cps_ram or max_units_ram, but not both at the same time.");
  tjsch->max_units_ram = max_units_ram;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSTrajectorySetMaxUnitsDisk_Memory(TSTrajectory tj, PetscInt max_units_disk)
{
  TJScheduler *tjsch = (TJScheduler *)tj->data;

  PetscFunctionBegin;
  PetscCheck(tjsch->max_cps_disk, PetscObjectComm((PetscObject)tj), PETSC_ERR_ARG_INCOMP, "Conflict with -ts_trjaectory_max_cps_disk or TSTrajectorySetMaxCpsDisk. You can set max_cps_disk or max_units_disk, but not both at the same time.");
  tjsch->max_units_ram = max_units_disk;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSTrajectoryMemorySetType_Memory(TSTrajectory tj, TSTrajectoryMemoryType tj_memory_type)
{
  TJScheduler *tjsch = (TJScheduler *)tj->data;

  PetscFunctionBegin;
  PetscCheck(!tj->setupcalled, PetscObjectComm((PetscObject)tj), PETSC_ERR_ARG_WRONGSTATE, "Cannot change schedule software after TSTrajectory has been setup or used");
  tjsch->tj_memory_type = tj_memory_type;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#if defined(PETSC_HAVE_REVOLVE)
PETSC_UNUSED static PetscErrorCode TSTrajectorySetRevolveOnline(TSTrajectory tj, PetscBool use_online)
{
  TJScheduler *tjsch = (TJScheduler *)tj->data;

  PetscFunctionBegin;
  tjsch->use_online = use_online;
  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif

PETSC_UNUSED static PetscErrorCode TSTrajectorySetSaveStack(TSTrajectory tj, PetscBool save_stack)
{
  TJScheduler *tjsch = (TJScheduler *)tj->data;

  PetscFunctionBegin;
  tjsch->save_stack = save_stack;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_UNUSED static PetscErrorCode TSTrajectorySetUseDRAM(TSTrajectory tj, PetscBool use_dram)
{
  TJScheduler *tjsch = (TJScheduler *)tj->data;

  PetscFunctionBegin;
  tjsch->stack.use_dram = use_dram;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   TSTrajectoryMemorySetType - sets the software that is used to generate the checkpointing schedule.

   Logically Collective

   Input Parameters:
+  tj - the `TSTrajectory` context
-  tj_memory_type - Revolve or CAMS

   Options Database Key:
.  -ts_trajectory_memory_type <tj_memory_type> - petsc, revolve, cams

   Level: intermediate

.seealso: [](chapter_ts), `TSTrajectory`, `TSTrajectorySetMaxUnitsRAM()`, `TSTrajectoryMemoryType`
@*/
PetscErrorCode TSTrajectoryMemorySetType(TSTrajectory tj, TSTrajectoryMemoryType tj_memory_type)
{
  PetscFunctionBegin;
  PetscTryMethod(tj, "TSTrajectoryMemorySetType_C", (TSTrajectory, TSTrajectoryMemoryType), (tj, tj_memory_type));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  TSTrajectorySetMaxCpsRAM - Set maximum number of checkpoints in RAM

  Logically Collective

  Input Parameter:
.  tj - tstrajectory context

  Output Parameter:
.  max_cps_ram - maximum number of checkpoints in RAM

  Level: intermediate

.seealso: [](chapter_ts), `TSTrajectory`, `TSTrajectorySetMaxUnitsRAM()`
@*/
PetscErrorCode TSTrajectorySetMaxCpsRAM(TSTrajectory tj, PetscInt max_cps_ram)
{
  PetscFunctionBegin;
  PetscUseMethod(tj, "TSTrajectorySetMaxCpsRAM_C", (TSTrajectory, PetscInt), (tj, max_cps_ram));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  TSTrajectorySetMaxCpsDisk - Set maximum number of checkpoints on disk

  Logically Collective

  Input Parameter:
.  tj - tstrajectory context

  Output Parameter:
.  max_cps_disk - maximum number of checkpoints on disk

  Level: intermediate

.seealso: [](chapter_ts), `TSTrajectory`, `TSTrajectorySetMaxUnitsDisk()`, `TSTrajectorySetMaxUnitsRAM()`
@*/
PetscErrorCode TSTrajectorySetMaxCpsDisk(TSTrajectory tj, PetscInt max_cps_disk)
{
  PetscFunctionBegin;
  PetscUseMethod(tj, "TSTrajectorySetMaxCpsDisk_C", (TSTrajectory, PetscInt), (tj, max_cps_disk));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  TSTrajectorySetMaxUnitsRAM - Set maximum number of checkpointing units in RAM

  Logically Collective

  Input Parameter:
.  tj - tstrajectory context

  Output Parameter:
.  max_units_ram - maximum number of checkpointing units in RAM

  Level: intermediate

.seealso: [](chapter_ts), `TSTrajectory`, `TSTrajectorySetMaxCpsRAM()`
@*/
PetscErrorCode TSTrajectorySetMaxUnitsRAM(TSTrajectory tj, PetscInt max_units_ram)
{
  PetscFunctionBegin;
  PetscUseMethod(tj, "TSTrajectorySetMaxUnitsRAM_C", (TSTrajectory, PetscInt), (tj, max_units_ram));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  TSTrajectorySetMaxUnitsDisk - Set maximum number of checkpointing units on disk

  Logically Collective

  Input Parameter:
.  tj - tstrajectory context

  Output Parameter:
.  max_units_disk - maximum number of checkpointing units on disk

  Level: intermediate

.seealso: [](chapter_ts), `TSTrajectory`, `TSTrajectorySetMaxCpsDisk()`
@*/
PetscErrorCode TSTrajectorySetMaxUnitsDisk(TSTrajectory tj, PetscInt max_units_disk)
{
  PetscFunctionBegin;
  PetscUseMethod(tj, "TSTrajectorySetMaxUnitsDisk_C", (TSTrajectory, PetscInt), (tj, max_units_disk));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSTrajectorySetFromOptions_Memory(TSTrajectory tj, PetscOptionItems *PetscOptionsObject)
{
  TJScheduler *tjsch = (TJScheduler *)tj->data;
  PetscEnum    etmp;
  PetscInt     max_cps_ram, max_cps_disk, max_units_ram, max_units_disk;
  PetscBool    flg;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "Memory based TS trajectory options");
  {
    PetscCall(PetscOptionsInt("-ts_trajectory_max_cps_ram", "Maximum number of checkpoints in RAM", "TSTrajectorySetMaxCpsRAM", tjsch->max_cps_ram, &max_cps_ram, &flg));
    if (flg) PetscCall(TSTrajectorySetMaxCpsRAM(tj, max_cps_ram));
    PetscCall(PetscOptionsInt("-ts_trajectory_max_cps_disk", "Maximum number of checkpoints on disk", "TSTrajectorySetMaxCpsDisk", tjsch->max_cps_disk, &max_cps_disk, &flg));
    if (flg) PetscCall(TSTrajectorySetMaxCpsDisk(tj, max_cps_disk));
    PetscCall(PetscOptionsInt("-ts_trajectory_max_units_ram", "Maximum number of checkpointing units in RAM", "TSTrajectorySetMaxUnitsRAM", tjsch->max_units_ram, &max_units_ram, &flg));
    if (flg) PetscCall(TSTrajectorySetMaxUnitsRAM(tj, max_units_ram));
    PetscCall(PetscOptionsInt("-ts_trajectory_max_units_disk", "Maximum number of checkpointing units on disk", "TSTrajectorySetMaxUnitsDisk", tjsch->max_units_disk, &max_units_disk, &flg));
    if (flg) PetscCall(TSTrajectorySetMaxUnitsDisk(tj, max_units_disk));
    PetscCall(PetscOptionsInt("-ts_trajectory_stride", "Stride to save checkpoints to file", "TSTrajectorySetStride", tjsch->stride, &tjsch->stride, NULL));
#if defined(PETSC_HAVE_REVOLVE)
    PetscCall(PetscOptionsBool("-ts_trajectory_revolve_online", "Trick TS trajectory into using online mode of revolve", "TSTrajectorySetRevolveOnline", tjsch->use_online, &tjsch->use_online, NULL));
#endif
    PetscCall(PetscOptionsBool("-ts_trajectory_save_stack", "Save all stack to disk", "TSTrajectorySetSaveStack", tjsch->save_stack, &tjsch->save_stack, NULL));
    PetscCall(PetscOptionsBool("-ts_trajectory_use_dram", "Use DRAM for checkpointing", "TSTrajectorySetUseDRAM", tjsch->stack.use_dram, &tjsch->stack.use_dram, NULL));
    PetscCall(PetscOptionsEnum("-ts_trajectory_memory_type", "Checkpointing scchedule software to use", "TSTrajectoryMemorySetType", TSTrajectoryMemoryTypes, (PetscEnum)(int)(tjsch->tj_memory_type), &etmp, &flg));
    if (flg) PetscCall(TSTrajectoryMemorySetType(tj, (TSTrajectoryMemoryType)etmp));
  }
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSTrajectorySetUp_Memory(TSTrajectory tj, TS ts)
{
  TJScheduler *tjsch = (TJScheduler *)tj->data;
  Stack       *stack = &tjsch->stack;
#if defined(PETSC_HAVE_REVOLVE)
  RevolveCTX *rctx, *rctx2;
  DiskStack  *diskstack = &tjsch->diskstack;
  PetscInt    diskblocks;
#endif
  PetscInt  numY, total_steps;
  PetscBool fixedtimestep;

  PetscFunctionBegin;
  if (ts->adapt) {
    PetscCall(PetscObjectTypeCompare((PetscObject)ts->adapt, TSADAPTNONE, &fixedtimestep));
  } else {
    fixedtimestep = PETSC_TRUE;
  }
  total_steps = (PetscInt)(PetscCeilReal((ts->max_time - ts->ptime) / ts->time_step));
  total_steps = total_steps < 0 ? PETSC_MAX_INT : total_steps;
  if (fixedtimestep) tjsch->total_steps = PetscMin(ts->max_steps, total_steps);

  tjsch->stack.solution_only = tj->solution_only;
  PetscCall(TSGetStages(ts, &numY, PETSC_IGNORE));
  if (stack->solution_only) {
    if (tjsch->max_units_ram) tjsch->max_cps_ram = tjsch->max_units_ram;
    else tjsch->max_units_ram = tjsch->max_cps_ram;
    if (tjsch->max_units_disk) tjsch->max_cps_disk = tjsch->max_units_disk;
  } else {
    if (tjsch->max_units_ram) tjsch->max_cps_ram = (ts->stifflyaccurate) ? tjsch->max_units_ram / numY : tjsch->max_units_ram / (numY + 1);
    else tjsch->max_units_ram = (ts->stifflyaccurate) ? numY * tjsch->max_cps_ram : (numY + 1) * tjsch->max_cps_ram;
    if (tjsch->max_units_disk) tjsch->max_cps_disk = (ts->stifflyaccurate) ? tjsch->max_units_disk / numY : tjsch->max_units_disk / (numY + 1);
    else tjsch->max_units_disk = (ts->stifflyaccurate) ? numY * tjsch->max_cps_disk : (numY + 1) * tjsch->max_cps_disk;
  }
  if (tjsch->max_cps_ram > 0) stack->stacksize = tjsch->max_units_ram; /* maximum stack size. Could be overallocated. */

  /* Determine the scheduler type */
  if (tjsch->stride > 1) { /* two level mode */
    PetscCheck(!tjsch->save_stack || tjsch->max_cps_disk <= 1 || tjsch->max_cps_disk > tjsch->max_cps_ram, PetscObjectComm((PetscObject)ts), PETSC_ERR_ARG_INCOMP, "The specified disk capacity is not enough to store a full stack of RAM checkpoints. You might want to change the disk capacity or use single level checkpointing instead.");
    if (tjsch->max_cps_disk <= 1 && tjsch->max_cps_ram > 1 && tjsch->max_cps_ram <= tjsch->stride - 1) tjsch->stype = TWO_LEVEL_REVOLVE;    /* use revolve_offline for each stride */
    if (tjsch->max_cps_disk > 1 && tjsch->max_cps_ram > 1 && tjsch->max_cps_ram <= tjsch->stride - 1) tjsch->stype = TWO_LEVEL_TWO_REVOLVE; /* use revolve_offline for each stride */
    if (tjsch->max_cps_disk <= 1 && (tjsch->max_cps_ram >= tjsch->stride || tjsch->max_cps_ram == -1)) tjsch->stype = TWO_LEVEL_NOREVOLVE;  /* can also be handled by TWO_LEVEL_REVOLVE */
  } else {                                                                                                                                  /* single level mode */
    if (fixedtimestep) {
      if (tjsch->max_cps_ram >= tjsch->total_steps - 1 || tjsch->max_cps_ram == -1) tjsch->stype = NONE; /* checkpoint all */
      else { /* choose the schedule software for offline checkpointing */ switch (tjsch->tj_memory_type) {
        case TJ_PETSC:
          tjsch->stype = NONE;
          break;
        case TJ_CAMS:
          tjsch->stype = CAMS_OFFLINE;
          break;
        case TJ_REVOLVE:
          tjsch->stype = (tjsch->max_cps_disk > 1) ? REVOLVE_MULTISTAGE : REVOLVE_OFFLINE;
          break;
        default:
          break;
        }
      }
    } else tjsch->stype = NONE; /* checkpoint all for adaptive time step */
#if defined(PETSC_HAVE_REVOLVE)
    if (tjsch->use_online) tjsch->stype = REVOLVE_ONLINE; /* trick into online (for testing purpose only) */
#endif
    PetscCheck(tjsch->stype == NONE || tjsch->max_cps_ram >= 1 || tjsch->max_cps_disk >= 1, PetscObjectComm((PetscObject)ts), PETSC_ERR_ARG_INCOMP, "The specified storage capacity is insufficient for one checkpoint, which is the minimum");
  }
  if (tjsch->stype >= CAMS_OFFLINE) {
#ifndef PETSC_HAVE_CAMS
    SETERRQ(PetscObjectComm((PetscObject)ts), PETSC_ERR_SUP, "CAMS is needed when there is not enough memory to checkpoint all time steps according to the user's settings, please reconfigure with the additional option --download-cams.");
#else
    CAMSCTX *actx;
    PetscInt ns = 0;
    if (stack->solution_only) {
      offline_ca_create(tjsch->total_steps, tjsch->max_cps_ram);
    } else {
      PetscCall(TSGetStages(ts, &ns, PETSC_IGNORE));
      offline_cams_create(tjsch->total_steps, tjsch->max_units_ram, ns, ts->stifflyaccurate);
    }
    PetscCall(PetscNew(&actx));
    actx->lastcheckpointstep = -1; /* -1 can trigger the initialization of CAMS */
    actx->lastcheckpointtype = -1; /* -1 can trigger the initialization of CAMS */
    actx->endstep            = tjsch->total_steps;
    actx->num_units_avail    = tjsch->max_units_ram;
    actx->num_stages         = ns;
    tjsch->actx              = actx;
#endif
  } else if (tjsch->stype > TWO_LEVEL_NOREVOLVE) {
#ifndef PETSC_HAVE_REVOLVE
    SETERRQ(PetscObjectComm((PetscObject)ts), PETSC_ERR_SUP, "revolve is needed when there is not enough memory to checkpoint all time steps according to the user's settings, please reconfigure with the additional option --download-revolve.");
#else
    PetscRevolveInt rfine, rsnaps, rsnaps2;

    switch (tjsch->stype) {
    case TWO_LEVEL_REVOLVE:
      PetscCall(PetscRevolveIntCast(tjsch->stride, &rfine));
      PetscCall(PetscRevolveIntCast(tjsch->max_cps_ram, &rsnaps));
      revolve_create_offline(rfine, rsnaps);
      break;
    case TWO_LEVEL_TWO_REVOLVE:
      diskblocks           = tjsch->save_stack ? tjsch->max_cps_disk / (tjsch->max_cps_ram + 1) : tjsch->max_cps_disk; /* The block size depends on whether the stack is saved. */
      diskstack->stacksize = diskblocks;
      PetscCall(PetscRevolveIntCast(tjsch->stride, &rfine));
      PetscCall(PetscRevolveIntCast(tjsch->max_cps_ram, &rsnaps));
      revolve_create_offline(rfine, rsnaps);
      PetscCall(PetscRevolveIntCast((tjsch->total_steps + tjsch->stride - 1) / tjsch->stride, &rfine));
      PetscCall(PetscRevolveIntCast(diskblocks, &rsnaps));
      revolve2_create_offline(rfine, rsnaps);
      PetscCall(PetscNew(&rctx2));
      rctx2->snaps_in       = rsnaps;
      rctx2->reverseonestep = PETSC_FALSE;
      rctx2->check          = 0;
      rctx2->oldcapo        = 0;
      rctx2->capo           = 0;
      rctx2->info           = 2;
      rctx2->fine           = rfine;
      tjsch->rctx2          = rctx2;
      diskstack->top        = -1;
      PetscCall(PetscMalloc1(diskstack->stacksize, &diskstack->container));
      break;
    case REVOLVE_OFFLINE:
      PetscCall(PetscRevolveIntCast(tjsch->total_steps, &rfine));
      PetscCall(PetscRevolveIntCast(tjsch->max_cps_ram, &rsnaps));
      revolve_create_offline(rfine, rsnaps);
      break;
    case REVOLVE_ONLINE:
      stack->stacksize = tjsch->max_cps_ram;
      PetscCall(PetscRevolveIntCast(tjsch->max_cps_ram, &rsnaps));
      revolve_create_online(rsnaps);
      break;
    case REVOLVE_MULTISTAGE:
      PetscCall(PetscRevolveIntCast(tjsch->total_steps, &rfine));
      PetscCall(PetscRevolveIntCast(tjsch->max_cps_ram, &rsnaps));
      PetscCall(PetscRevolveIntCast(tjsch->max_cps_ram + tjsch->max_cps_disk, &rsnaps2));
      revolve_create_multistage(rfine, rsnaps2, rsnaps);
      break;
    default:
      break;
    }
    PetscCall(PetscNew(&rctx));
    PetscCall(PetscRevolveIntCast(tjsch->max_cps_ram, &rsnaps));
    rctx->snaps_in       = rsnaps; /* for theta methods snaps_in=2*max_cps_ram */
    rctx->reverseonestep = PETSC_FALSE;
    rctx->check          = 0;
    rctx->oldcapo        = 0;
    rctx->capo           = 0;
    rctx->info           = 2;
    if (tjsch->stride > 1) {
      PetscCall(PetscRevolveIntCast(tjsch->stride, &rfine));
    } else {
      PetscCall(PetscRevolveIntCast(tjsch->total_steps, &rfine));
    }
    rctx->fine  = rfine;
    tjsch->rctx = rctx;
    if (tjsch->stype == REVOLVE_ONLINE) rctx->fine = -1;
#endif
  } else {
    if (tjsch->stype == TWO_LEVEL_NOREVOLVE) stack->stacksize = tjsch->stride - 1; /* need tjsch->stride-1 at most */
    if (tjsch->stype == NONE) {
      if (fixedtimestep) stack->stacksize = stack->solution_only ? tjsch->total_steps : tjsch->total_steps - 1;
      else { /* adaptive time step */ /* if max_cps_ram is not specified, use maximal allowed number of steps for stack size */
        if (tjsch->max_cps_ram == -1) stack->stacksize = ts->max_steps < PETSC_MAX_INT ? ts->max_steps : 10000;
        tjsch->total_steps = stack->solution_only ? stack->stacksize : stack->stacksize + 1; /* will be updated as time integration advances */
      }
    }
  }

  if ((tjsch->stype >= TWO_LEVEL_NOREVOLVE && tjsch->stype < REVOLVE_OFFLINE) || tjsch->stype == REVOLVE_MULTISTAGE) { /* these types need to use disk */
    PetscCall(TSTrajectorySetUp_Basic(tj, ts));
  }

  stack->stacksize = PetscMax(stack->stacksize, 1);
  tjsch->recompute = PETSC_FALSE;
  PetscCall(StackInit(stack, stack->stacksize, numY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSTrajectoryReset_Memory(TSTrajectory tj)
{
#if defined(PETSC_HAVE_REVOLVE) || defined(PETSC_HAVE_CAMS)
  TJScheduler *tjsch = (TJScheduler *)tj->data;
#endif

  PetscFunctionBegin;
#if defined(PETSC_HAVE_REVOLVE)
  if (tjsch->stype > TWO_LEVEL_NOREVOLVE) {
    revolve_reset();
    if (tjsch->stype == TWO_LEVEL_TWO_REVOLVE) {
      revolve2_reset();
      PetscCall(PetscFree(tjsch->diskstack.container));
    }
  }
  if (tjsch->stype > TWO_LEVEL_NOREVOLVE) {
    PetscCall(PetscFree(tjsch->rctx));
    PetscCall(PetscFree(tjsch->rctx2));
  }
#endif
#if defined(PETSC_HAVE_CAMS)
  if (tjsch->stype == CAMS_OFFLINE) {
    if (tjsch->stack.solution_only) offline_ca_destroy();
    else offline_ca_destroy();
    PetscCall(PetscFree(tjsch->actx));
  }
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSTrajectoryDestroy_Memory(TSTrajectory tj)
{
  TJScheduler *tjsch = (TJScheduler *)tj->data;

  PetscFunctionBegin;
  PetscCall(StackDestroy(&tjsch->stack));
  PetscCall(PetscViewerDestroy(&tjsch->viewer));
  PetscCall(PetscObjectComposeFunction((PetscObject)tj, "TSTrajectorySetMaxCpsRAM_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)tj, "TSTrajectorySetMaxCpsDisk_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)tj, "TSTrajectorySetMaxUnitsRAM_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)tj, "TSTrajectorySetMaxUnitsDisk_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)tj, "TSTrajectoryMemorySetType_C", NULL));
  PetscCall(PetscFree(tjsch));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
      TSTRAJECTORYMEMORY - Stores each solution of the ODE/ADE in memory

  Level: intermediate

.seealso: [](chapter_ts), `TSTrajectoryCreate()`, `TS`, `TSTrajectorySetType()`, `TSTrajectoryType`, `TSTrajectory`
M*/
PETSC_EXTERN PetscErrorCode TSTrajectoryCreate_Memory(TSTrajectory tj, TS ts)
{
  TJScheduler *tjsch;

  PetscFunctionBegin;
  tj->ops->set            = TSTrajectorySet_Memory;
  tj->ops->get            = TSTrajectoryGet_Memory;
  tj->ops->setup          = TSTrajectorySetUp_Memory;
  tj->ops->setfromoptions = TSTrajectorySetFromOptions_Memory;
  tj->ops->reset          = TSTrajectoryReset_Memory;
  tj->ops->destroy        = TSTrajectoryDestroy_Memory;

  PetscCall(PetscNew(&tjsch));
  tjsch->stype        = NONE;
  tjsch->max_cps_ram  = -1; /* -1 indicates that it is not set */
  tjsch->max_cps_disk = -1; /* -1 indicates that it is not set */
  tjsch->stride       = 0;  /* if not zero, two-level checkpointing will be used */
#if defined(PETSC_HAVE_REVOLVE)
  tjsch->use_online = PETSC_FALSE;
#endif
  tjsch->save_stack = PETSC_TRUE;

  tjsch->stack.solution_only = tj->solution_only;
  PetscCall(PetscViewerCreate(PetscObjectComm((PetscObject)tj), &tjsch->viewer));
  PetscCall(PetscViewerSetType(tjsch->viewer, PETSCVIEWERBINARY));
  PetscCall(PetscViewerPushFormat(tjsch->viewer, PETSC_VIEWER_NATIVE));
  PetscCall(PetscViewerFileSetMode(tjsch->viewer, FILE_MODE_WRITE));

  PetscCall(PetscObjectComposeFunction((PetscObject)tj, "TSTrajectorySetMaxCpsRAM_C", TSTrajectorySetMaxCpsRAM_Memory));
  PetscCall(PetscObjectComposeFunction((PetscObject)tj, "TSTrajectorySetMaxCpsDisk_C", TSTrajectorySetMaxCpsDisk_Memory));
  PetscCall(PetscObjectComposeFunction((PetscObject)tj, "TSTrajectorySetMaxUnitsRAM_C", TSTrajectorySetMaxUnitsRAM_Memory));
  PetscCall(PetscObjectComposeFunction((PetscObject)tj, "TSTrajectorySetMaxUnitsDisk_C", TSTrajectorySetMaxUnitsDisk_Memory));
  PetscCall(PetscObjectComposeFunction((PetscObject)tj, "TSTrajectoryMemorySetType_C", TSTrajectoryMemorySetType_Memory));
  tj->data = tjsch;
  PetscFunctionReturn(PETSC_SUCCESS);
}
