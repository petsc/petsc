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

typedef struct _Stack {
  SchedulerType stype;
  PetscBool     use_online;
  PetscBool     recompute;
  PetscBool     solution_only;
  PetscBool     save_stack;
  MPI_Comm      comm;
  RevolveCTX    *rctx;
  PetscInt      max_cps_ram;  /* maximum checkpoints in RAM */
  PetscInt      max_cps_disk; /* maximum checkpoints on disk */
  PetscInt      stride;
  PetscInt      total_steps;  /* total number of steps */
  PetscInt      numY;
  PetscInt      stacksize;
  PetscInt      top;          /* top of the stack */
  StackElement  *stack;       /* container */
} Stack;

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
#endif

#undef __FUNCT__
#define __FUNCT__ "StackCreate"
static PetscErrorCode StackCreate(MPI_Comm comm,Stack *s,PetscInt size,PetscInt ny)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  s->top  = -1;
  s->comm = comm;
  s->numY = ny;

  ierr = PetscMalloc1(size*sizeof(StackElement),&s->stack);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "StackDestroy"
static PetscErrorCode StackDestroy(Stack **stack)
{
  PetscInt       i;
  Stack          *s = (*stack);
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (s->top>-1) {
    for (i=0;i<=s->top;i++) {
      ierr = VecDestroy(&s->stack[i]->X);CHKERRQ(ierr);
      if (!s->solution_only) {
        ierr = VecDestroyVecs(s->numY,&s->stack[i]->Y);CHKERRQ(ierr);
      }
      ierr = PetscFree(s->stack[i]);CHKERRQ(ierr);
    }
  }
  ierr = PetscFree(s->stack);CHKERRQ(ierr);
  if (s->stype) {
    ierr = PetscFree(s->rctx);CHKERRQ(ierr);
  }
  ierr = PetscFree(*stack);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "StackPush"
static PetscErrorCode StackPush(Stack *s,StackElement e)
{
  PetscFunctionBegin;
  if (s->top+1 >= s->stacksize) SETERRQ1(s->comm,PETSC_ERR_MEMC,"Maximum stack size (%D) exceeded",s->stacksize);
  s->stack[++s->top] = e;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "StackPop"
static PetscErrorCode StackPop(Stack *s,StackElement *e)
{
  PetscFunctionBegin;
  if (s->top == -1) SETERRQ(s->comm,PETSC_ERR_MEMC,"Empty stack");
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
#define __FUNCT__ "StackFind"
static PetscErrorCode StackFind(Stack *s,StackElement *e,PetscInt index)
{
  PetscFunctionBegin;
  *e = s->stack[index];
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
static PetscErrorCode StackDumpAll(TS ts,Stack *s,PetscInt id)
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
  for (i=0;i<s->stacksize;i++) {
    e = s->stack[i];
    ierr = WriteToDisk(e->stepnum,e->time,e->timeprev,e->X,e->Y,s->numY,s->solution_only,viewer);CHKERRQ(ierr);
  }
  /* save the last step for restart, the last step is in memory when using single level schemes, but not necessarily the case for multi level schemes */
  ierr = TSGetStages(ts,&s->numY,&Y);CHKERRQ(ierr);
  ierr = WriteToDisk(ts->total_steps,ts->ptime,ts->ptime_prev,ts->vec_sol,Y,s->numY,s->solution_only,viewer);CHKERRQ(ierr);
  for (i=0;i<s->stacksize;i++) {
    ierr = StackPop(s,&e);CHKERRQ(ierr);
    ierr = VecDestroy(&e->X);CHKERRQ(ierr);
    if (!s->solution_only) {
      ierr = VecDestroyVecs(s->numY,&e->Y);CHKERRQ(ierr);
    }
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
  PetscInt       i;
  StackElement   e;
  PetscViewer    viewer;
  char           filename[PETSC_MAX_PATH_LEN];
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscSNPrintf(filename,sizeof filename,"SA-data/SA-STACK%06d.bin",id);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_READ,&viewer);CHKERRQ(ierr);
  for (i=0;i<s->stacksize;i++) {
    ierr = PetscCalloc1(1,&e);
    ierr = TSGetStages(ts,&s->numY,&Y);CHKERRQ(ierr);
    ierr = VecDuplicate(Y[0],&e->X);CHKERRQ(ierr);
    if (!s->solution_only && s->numY>0) {
      ierr = VecDuplicateVecs(Y[0],s->numY,&e->Y);CHKERRQ(ierr);
    }
    ierr = StackPush(s,e);CHKERRQ(ierr);
  }
  for (i=0;i<s->stacksize;i++) {
    e = s->stack[i];
    ierr = ReadFromDisk(&e->stepnum,&e->time,&e->timeprev,e->X,e->Y,s->numY,s->solution_only,viewer);CHKERRQ(ierr);
  }
  /* load the last step into TS */
  ierr = TSGetStages(ts,&s->numY,&Y);CHKERRQ(ierr);
  ierr = ReadFromDisk(&ts->total_steps,&ts->ptime,&ts->ptime_prev,ts->vec_sol,Y,s->numY,s->solution_only,viewer);CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts,ts->ptime-ts->ptime_prev);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DumpSingle"
static PetscErrorCode DumpSingle(TS ts,Stack *s,PetscInt id)
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
  ierr = TSGetStages(ts,&s->numY,&Y);CHKERRQ(ierr);
  ierr = WriteToDisk(stepnum,ts->ptime,ts->ptime_prev,ts->vec_sol,Y,s->numY,s->solution_only,viewer);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(Disk_Write,ts,0,0,0);CHKERRQ(ierr);

  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "LoadSingle"
static PetscErrorCode LoadSingle(TS ts,Stack *s,PetscInt id)
{
  Vec            *Y;
  PetscViewer    viewer;
  char           filename[PETSC_MAX_PATH_LEN];
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscSNPrintf(filename,sizeof filename,"SA-data/SA-CPS%06d.bin",id);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_READ,&viewer);CHKERRQ(ierr);

  ierr = PetscLogEventBegin(Disk_Read,ts,0,0,0);CHKERRQ(ierr);
  ierr = TSGetStages(ts,&s->numY,&Y);CHKERRQ(ierr);
  ierr = ReadFromDisk(&ts->total_steps,&ts->ptime,&ts->ptime_prev,ts->vec_sol,Y,s->numY,s->solution_only,viewer);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(Disk_Read,ts,0,0,0);CHKERRQ(ierr);

  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSTrajectorySetStride_Memory"
PetscErrorCode TSTrajectorySetStride_Memory(TSTrajectory tj,TS ts,PetscInt stride)
{
  Stack *s = (Stack*)tj->data;

  PetscFunctionBegin;
  s->stride = stride;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSTrajectorySetMaxCpsRAM_Memory"
PetscErrorCode TSTrajectorySetMaxCpsRAM_Memory(TSTrajectory tj,TS ts,PetscInt max_cps_ram)
{
  Stack *s = (Stack*)tj->data;

  PetscFunctionBegin;
  s->max_cps_ram = max_cps_ram;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSTrajectorySetMaxCpsDisk_Memory"
PetscErrorCode TSTrajectorySetMaxCpsDisk_Memory(TSTrajectory tj,TS ts,PetscInt max_cps_disk)
{
  Stack *s = (Stack*)tj->data;

  PetscFunctionBegin;
  s->max_cps_disk = max_cps_disk;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSTrajectorySetRevolveOnline"
PetscErrorCode TSTrajectorySetRevolveOnline(TSTrajectory tj,PetscBool use_online)
{
  Stack *s = (Stack*)tj->data;

  PetscFunctionBegin;
  s->use_online = use_online;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSTrajectorySetSaveStack"
PetscErrorCode TSTrajectorySetSaveStack(TSTrajectory tj,PetscBool save_stack)
{
  Stack *s = (Stack*)tj->data;

  PetscFunctionBegin;
  s->save_stack = save_stack;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSTrajectorySetSolutionOnly"
PetscErrorCode TSTrajectorySetSolutionOnly(TSTrajectory tj,PetscBool solution_only)
{
  Stack *s = (Stack*)tj->data;

  PetscFunctionBegin;
  s->solution_only = solution_only;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSTrajectorySetFromOptions_Memory"
PetscErrorCode TSTrajectorySetFromOptions_Memory(PetscOptions *PetscOptionsObject,TSTrajectory tj)
{
  Stack          *s = (Stack*)tj->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"Memory based TS trajectory options");CHKERRQ(ierr);
  {
    ierr = PetscOptionsInt("-tstrajectory_max_cps_ram","Maximum number of checkpoints in RAM","TSTrajectorySetMaxCpsRAM_Memory",s->max_cps_ram,&s->max_cps_ram,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-tstrajectory_max_cps_disk","Maximum number of checkpoints on disk","TSTrajectorySetMaxCpsDisk_Memory",s->max_cps_disk,&s->max_cps_disk,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-tstrajectory_stride","Stride to save checkpoints to file","TSTrajectorySetStride_Memory",s->stride,&s->stride,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-tstrajectory_revolve_online","Trick TS trajectory into using online mode of revolve","TSTrajectorySetRevolveOnline",s->use_online,&s->use_online,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-tstrajectory_save_stack","Save all stack to disk","TSTrajectorySetSaveStack",s->save_stack,&s->save_stack,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-tstrajectory_solution_only","Checkpoint solution only","TSTrajectorySetSolutionOnly",s->solution_only,&s->solution_only,NULL);CHKERRQ(ierr);
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
  PetscBool      flg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscStrcmp(((PetscObject)ts->adapt)->type_name,TSADAPTNONE,&flg);
  if (flg) { /* fixed time step */
    s->total_steps = PetscMin(ts->max_steps,(PetscInt)(ceil(ts->max_time/ts->time_step)));
  }
  if (s->max_cps_ram > 1) s->stacksize = s->max_cps_ram;
  if (s->stride > 1) { /* two level mode works for both fixed time step and adaptive time step */
    if (s->max_cps_ram > 1 && s->max_cps_ram < s->stride-1) { /* use revolve_offline for each stride */
      s->stype = TWO_LEVEL_REVOLVE;
    }else { /* checkpoint all for each stride */
      s->stype     = TWO_LEVEL_NOREVOLVE;
      s->stacksize = s->solution_only ? s->stride : s->stride-1;
    }
  } else {
    if (flg) { /* fixed time step */
      if (s->max_cps_ram >= s->total_steps-1 || s->max_cps_ram < 1) { /* checkpoint all */
        s->stype     = NONE;
        s->stacksize = s->solution_only ? s->total_steps : s->total_steps-1;
      } else {
        if (s->max_cps_disk > 1) { /* disk can be used */
          s->stype = REVOLVE_MULTISTAGE;
        } else { /* memory only */
          s->stype = REVOLVE_OFFLINE;
        }
      }
    } else { /* adaptive time step */
      s->stype = REVOLVE_ONLINE;
    }
    if (s->use_online) { /* trick into online */
      s->stype     = REVOLVE_ONLINE;
      s->stacksize = s->max_cps_ram;
    }
  }

  if (s->stype > TWO_LEVEL_NOREVOLVE) {
#ifndef PETSC_HAVE_REVOLVE
    SETERRQ(s->comm,PETSC_ERR_SUP,"revolve is needed when there is not enough memory to checkpoint all time steps according to the user's settings, please reconfigure with the additional option --download-revolve.");
#else
    if (s->stype == TWO_LEVEL_REVOLVE) revolve_create_offline(s->stride,s->max_cps_ram);
    else if (s->stype == REVOLVE_OFFLINE) revolve_create_offline(s->total_steps,s->max_cps_ram);
    else if (s->stype == REVOLVE_ONLINE) revolve_create_online(s->max_cps_ram);
    else if (s->stype ==REVOLVE_MULTISTAGE) revolve_create_multistage(s->total_steps,s->max_cps_ram+s->max_cps_disk,s->max_cps_ram);

    ierr = PetscCalloc1(1,&rctx);CHKERRQ(ierr);
    rctx->snaps_in       = s->max_cps_ram; /* for theta methods snaps_in=2*max_cps_ram */
    rctx->reverseonestep = PETSC_FALSE;
    rctx->check          = 0;
    rctx->oldcapo        = 0;
    rctx->capo           = 0;
    rctx->info           = 2;
    rctx->fine           = (s->stride > 1) ? s->stride : s->total_steps;

    s->rctx      = rctx;
    if (s->stype == REVOLVE_ONLINE) rctx->fine = -1;
#endif
  }

  s->recompute = PETSC_FALSE;

  ierr = TSGetStages(ts,&numY,PETSC_IGNORE);CHKERRQ(ierr);
  ierr = StackCreate(PetscObjectComm((PetscObject)ts),s,s->stacksize,numY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ApplyRevolve"
static PetscErrorCode ApplyRevolve(TS ts,Stack *s,PetscInt stepnum,PetscInt localstepnum,PetscInt *whattodo)
{
#ifdef PETSC_HAVE_REVOLVE
  PetscInt       shift;
#endif
  RevolveCTX     *rctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
#ifdef PETSC_HAVE_REVOLVE
    rctx = s->rctx;
    if (rctx->reverseonestep && stepnum==s->total_steps) { /* intermediate information is ready inside TS, this happens at last time step */
      //rctx->reverseonestep = PETSC_FALSE;
      PetscFunctionReturn(0);
    }
    if (rctx->stepsleft != 0) { /* advance the solution without checkpointing anything as Revolve requires */
      rctx->stepsleft--;
      PetscFunctionReturn(0);
    }

    /* let Revolve determine what to do next */
    shift         = stepnum-localstepnum;
    rctx->capo    = localstepnum;
    rctx->oldcapo = rctx->capo;
    *whattodo = revolve_action(&rctx->check,&rctx->capo,&rctx->fine,rctx->snaps_in,&rctx->info,&rctx->where);
    if (s->stype == REVOLVE_ONLINE && *whattodo ==7) *whattodo = 2;
    printwhattodo(*whattodo,rctx,shift);
    if (*whattodo == -1) SETERRQ(s->comm,PETSC_ERR_LIB,"Error in the Revolve library");
    if (*whattodo == 1) { /* advance some time steps */
      if (s->stype == REVOLVE_ONLINE && rctx->capo >= s->total_steps-1) {
        revolve_turn(s->total_steps,&rctx->capo,&rctx->fine);
        printwhattodo(*whattodo,rctx,shift);
      }
      rctx->stepsleft = rctx->capo-rctx->oldcapo-1;
    }
    if (*whattodo == 3 || *whattodo == 4) { /* ready for a reverse step */
      rctx->reverseonestep = PETSC_TRUE;
    }
    if (*whattodo == 5) { /* restore a checkpoint and ask Revolve what to do next */
      rctx->oldcapo = rctx->capo;
      *whattodo = revolve_action(&rctx->check,&rctx->capo,&rctx->fine,rctx->snaps_in,&rctx->info,&rctx->where); /* must return 1*/
      printwhattodo(*whattodo,rctx,shift);
      rctx->stepsleft = rctx->capo-rctx->oldcapo;
    }
    if (*whattodo == 7) { /* save the checkpoint to disk */
      ierr = DumpSingle(ts,s,rctx->check);CHKERRQ(ierr);
      rctx->oldcapo = rctx->capo;
      *whattodo = revolve_action(&rctx->check,&rctx->capo,&rctx->fine,rctx->snaps_in,&rctx->info,&rctx->where); /* must return 1*/
      printwhattodo(*whattodo,rctx,shift);
      rctx->stepsleft = rctx->capo-rctx->oldcapo-1;
    }
    if (*whattodo != 2) {
      PetscFunctionReturn(0);
    } else { /* store a checkpoint to RAM and ask Revolve how many time steps to advance next */
      rctx->oldcapo = rctx->capo;
      *whattodo = revolve_action(&rctx->check,&rctx->capo,&rctx->fine,rctx->snaps_in,&rctx->info,&rctx->where); /* must return 1*/
      printwhattodo(*whattodo,rctx,shift);
      if (s->stype == REVOLVE_ONLINE && rctx->capo >= s->total_steps-1) {
        revolve_turn(s->total_steps,&rctx->capo,&rctx->fine);
        printwhattodo(*whattodo,rctx,shift);
      }
      rctx->stepsleft = rctx->capo-rctx->oldcapo-1;
    }
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ElementCreate"
static PetscErrorCode ElementCreate(TS ts,Stack *s,StackElement *e,PetscInt stepnum,PetscReal time,Vec X)
{
  Vec            *Y;
  PetscInt       i;
  PetscReal      timeprev;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscCalloc1(1,e);CHKERRQ(ierr);
  ierr = VecDuplicate(X,&(*e)->X);CHKERRQ(ierr);
  ierr = VecCopy(X,(*e)->X);CHKERRQ(ierr);
  if (s->numY > 0 && !s->solution_only) {
    ierr = TSGetStages(ts,&s->numY,&Y);CHKERRQ(ierr);
    ierr = VecDuplicateVecs(Y[0],s->numY,&(*e)->Y);CHKERRQ(ierr);
    for (i=0;i<s->numY;i++) {
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
static PetscErrorCode ElementDestroy(Stack *s,StackElement e)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDestroy(&e->X);CHKERRQ(ierr);
  if (!s->solution_only) {
    ierr = VecDestroyVecs(s->numY,&e->Y);CHKERRQ(ierr);
  }
  ierr = PetscFree(e);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "UpdateTS"
static PetscErrorCode UpdateTS(TS ts,Stack *s,StackElement e)
{
  Vec            *Y;
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCopy(e->X,ts->vec_sol);CHKERRQ(ierr);
  if (!s->solution_only) {
    ierr = TSGetStages(ts,&s->numY,&Y);CHKERRQ(ierr);
    for (i=0;i<s->numY;i++) {
      ierr = VecCopy(e->Y[i],Y[i]);CHKERRQ(ierr);
    }
  }
  ierr = TSSetTimeStep(ts,e->timeprev-e->time);CHKERRQ(ierr); /* stepsize will be negative */
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ReCompute"
static PetscErrorCode ReCompute(TS ts,Stack *s,StackElement e,PetscInt stepnum)
{
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* reset ts context */
  ts->steps      = e->stepnum; /* global stepnum */
  ts->ptime      = e->time;
  ts->ptime_prev = e->timeprev;
  for (i=ts->steps;i<stepnum;i++) { /* assume fixed step size */
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
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SetTrajN"
static PetscErrorCode SetTrajN(TS ts,Stack *s,PetscInt stepnum,PetscReal time,Vec X)
{
  StackElement   e;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (s->solution_only) {
    /* skip the last two steps of each stride or the whole interval */
    if (stepnum >= s->total_steps-1 || s->recompute) PetscFunctionReturn(0); //?
  } else {
    /* skip the first and the last steps of each stride or the whole interval */
    if (stepnum == 0 || stepnum == s->total_steps) PetscFunctionReturn(0);
  }
  if (stepnum < s->top) SETERRQ(s->comm,PETSC_ERR_MEMC,"Illegal modification of a non-top stack element");
  ierr = ElementCreate(ts,s,&e,stepnum,time,X);CHKERRQ(ierr);
  ierr = StackPush(s,e);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "GetTrajN"
static PetscErrorCode GetTrajN(TS ts,Stack *s,PetscInt stepnum)
{
  PetscInt       steps;
  PetscReal      stepsize;
  StackElement   e;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (stepnum == s->total_steps) {
    ierr = TSGetTimeStep(ts,&stepsize);CHKERRQ(ierr);
    ierr = TSSetTimeStep(ts,-stepsize);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  /* restore a checkpoint */
  ierr = StackTop(s,&e);CHKERRQ(ierr);
  ierr = UpdateTS(ts,s,e);CHKERRQ(ierr);
  if (s->solution_only) {/* recompute one step */
    steps = ts->steps;
    s->recompute = PETSC_TRUE;
    ierr = TSGetTimeStep(ts,&stepsize);CHKERRQ(ierr);
    ierr = TSSetTimeStep(ts,-stepsize);CHKERRQ(ierr);
    ierr = ReCompute(ts,s,e,stepnum);CHKERRQ(ierr);
    ts->steps = steps;
    ts->total_steps = stepnum;
    ierr = TSGetTimeStep(ts,&stepsize);CHKERRQ(ierr);
    ierr = TSSetTimeStep(ts,-stepsize);CHKERRQ(ierr);
  }
  ierr = StackPop(s,&e);CHKERRQ(ierr);
  ierr = ElementDestroy(s,e);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SetTrajROF"
static PetscErrorCode SetTrajROF(TS ts,Stack *s,PetscInt stepnum,PetscReal time,Vec X)
{
  PetscInt       whattodo;
  StackElement   e;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = ApplyRevolve(ts,s,stepnum,stepnum,&whattodo);CHKERRQ(ierr);
  if (whattodo == 2){
    if (stepnum < s->top) SETERRQ(s->comm,PETSC_ERR_MEMC,"Illegal modification of a non-top stack element");
    ierr = ElementCreate(ts,s,&e,stepnum,time,X);CHKERRQ(ierr);
    ierr = StackPush(s,e);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "GetTrajROF"
static PetscErrorCode GetTrajROF(TS ts,Stack *s,PetscInt stepnum)
{
#ifdef PETSC_HAVE_REVOLVE
  PetscInt       whattodo,shift;
#endif
  PetscInt       steps;
  PetscReal      stepsize;
  StackElement   e;
  PetscErrorCode ierr;

  PetscFunctionBegin;
#ifdef PETSC_HAVE_REVOLVE
  if ( s->rctx->reverseonestep) { /* ready for the reverse step */
    ierr = TSGetTimeStep(ts,&stepsize);CHKERRQ(ierr);
    ierr = TSSetTimeStep(ts,-stepsize);CHKERRQ(ierr);
    s->rctx->reverseonestep = PETSC_FALSE;
    PetscFunctionReturn(0);
  }
#endif
  if ((!s->solution_only && stepnum == 0) || stepnum == s->total_steps) {
    ierr = TSGetTimeStep(ts,&stepsize);CHKERRQ(ierr);
    ierr = TSSetTimeStep(ts,-stepsize);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  if (s->solution_only) {
#ifdef PETSC_HAVE_REVOLVE
    s->rctx->capo = stepnum;
    shift = 0;
    whattodo = revolve_action(&s->rctx->check,&s->rctx->capo,&s->rctx->fine,s->rctx->snaps_in,&s->rctx->info,&s->rctx->where);
    printwhattodo(whattodo,s->rctx,shift);
#endif
  }
  /* restore a checkpoint */
  ierr = StackTop(s,&e);CHKERRQ(ierr);
  if (e && e->stepnum >= stepnum) {
    SETERRQ2(s->comm,PETSC_ERR_ARG_OUTOFRANGE,"The current step no. is %D, but the step number at top of the stack is %D",stepnum,e->stepnum);
  }
  ierr = UpdateTS(ts,s,e);CHKERRQ(ierr);
  if (s->solution_only || (!s->solution_only && e->stepnum<stepnum)) { /* must recompute */
    steps = ts->steps;
    s->recompute = PETSC_TRUE;
    ierr = TSGetTimeStep(ts,&stepsize);CHKERRQ(ierr);
    ierr = TSSetTimeStep(ts,-stepsize);CHKERRQ(ierr);
    ierr = ReCompute(ts,s,e,stepnum);CHKERRQ(ierr);
    ts->steps = steps;
    //ts->total_steps = stepnum;
    ierr = TSGetTimeStep(ts,&stepsize);CHKERRQ(ierr);
    ierr = TSSetTimeStep(ts,-stepsize);CHKERRQ(ierr);
    #ifdef PETSC_HAVE_REVOLVE
      s->rctx->reverseonestep = PETSC_FALSE;
    #endif
  } else {
    ierr = ElementDestroy(s,e);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SetTrajRON"
static PetscErrorCode SetTrajRON(TS ts,Stack *s,PetscInt stepnum,PetscReal time,Vec X)
{
  PetscInt       whattodo;
  PetscReal      timeprev;
  StackElement   e;
  RevolveCTX     *rctx = s->rctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = ApplyRevolve(ts,s,stepnum,stepnum,&whattodo);CHKERRQ(ierr);
  if (whattodo == 2){
    if (rctx->check != s->top+1) { /* overwrite some non-top checkpoint in the stack*/
      ierr = StackFind(s,&e,rctx->check);CHKERRQ(ierr);
      ierr = VecCopy(X,e->X);CHKERRQ(ierr);
      e->stepnum  = stepnum;
      e->time     = time;
      ierr        = TSGetPrevTime(ts,&timeprev);CHKERRQ(ierr);
      e->timeprev = timeprev;
    } else {
      if (stepnum < s->top) SETERRQ(s->comm,PETSC_ERR_MEMC,"Illegal modification of a non-top stack element");
      ierr = ElementCreate(ts,s,&e,stepnum,time,X);CHKERRQ(ierr);
      ierr = StackPush(s,e);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "GetTrajRON"
static PetscErrorCode GetTrajRON(TS ts,Stack *s,PetscInt stepnum)
{
#ifdef PETSC_HAVE_REVOLVE
  PetscInt       whattodo,shift;
#endif
  PetscInt       steps;
  PetscReal      stepsize;
  StackElement   e;
  PetscErrorCode ierr;

  PetscFunctionBegin;
#ifdef PETSC_HAVE_REVOLVE
  if ( s->rctx->reverseonestep) { /* ready for the reverse step */
    ierr = TSGetTimeStep(ts,&stepsize);CHKERRQ(ierr);
    ierr = TSSetTimeStep(ts,-stepsize);CHKERRQ(ierr);
    s->rctx->reverseonestep = PETSC_FALSE;
    PetscFunctionReturn(0);
  }
#endif
  if ((!s->solution_only && stepnum == 0) || stepnum == s->total_steps) {
    ierr = TSGetTimeStep(ts,&stepsize);CHKERRQ(ierr);
    ierr = TSSetTimeStep(ts,-stepsize);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  if (s->solution_only) {
#ifdef PETSC_HAVE_REVOLVE
    s->rctx->capo = stepnum;
    shift = 0;
    whattodo = revolve_action(&s->rctx->check,&s->rctx->capo,&s->rctx->fine,s->rctx->snaps_in,&s->rctx->info,&s->rctx->where);
    printwhattodo(whattodo,s->rctx,shift);
#endif
  }
  /* restore a checkpoint */
  ierr = StackTop(s,&e);CHKERRQ(ierr);
  if (e && e->stepnum >= stepnum) {
    SETERRQ2(s->comm,PETSC_ERR_ARG_OUTOFRANGE,"The current step no. is %D, but the step number at top of the stack is %D",stepnum,e->stepnum);
  }
  ierr = UpdateTS(ts,s,e);CHKERRQ(ierr);
  if (s->solution_only || (!s->solution_only && e->stepnum<stepnum)) { /* must recompute */
    steps = ts->steps;
    s->recompute = PETSC_TRUE;
    ierr = TSGetTimeStep(ts,&stepsize);CHKERRQ(ierr);
    ierr = TSSetTimeStep(ts,-stepsize);CHKERRQ(ierr);
    ierr = ReCompute(ts,s,e,stepnum);CHKERRQ(ierr);
    ts->steps = steps;
    ts->total_steps = stepnum;
    ierr = TSGetTimeStep(ts,&stepsize);CHKERRQ(ierr);
    ierr = TSSetTimeStep(ts,-stepsize);CHKERRQ(ierr);
    #ifdef PETSC_HAVE_REVOLVE
      s->rctx->reverseonestep = PETSC_FALSE;
    #endif
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SetTrajTLR"
static PetscErrorCode SetTrajTLR(TS ts,Stack *s,PetscInt stepnum,PetscReal time,Vec X)
{
  PetscInt       whattodo,localstepnum,id;
  StackElement   e;
  RevolveCTX     *rctx = s->rctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  localstepnum = stepnum%s->stride;
  if (localstepnum == 0 && stepnum != s->total_steps && !s->recompute) { /* save to disk */
    id = stepnum/s->stride;
    if (s->save_stack) {
      if (stepnum) { /* skip step 0 */
#ifdef PETSC_HAVE_REVOLVE
        PetscPrintf(PETSC_COMM_WORLD,"\x1B[33mDump stack to file\033[0m\n");
#endif
        ierr = StackDumpAll(ts,s,id);CHKERRQ(ierr);
        s->top = -1; /* reset top */
#ifdef PETSC_HAVE_REVOLVE
        revolve_reset();
        revolve_create_offline(s->stride,s->max_cps_ram);
        rctx = s->rctx;
        rctx->check = 0;
        rctx->capo  = 0;
        rctx->fine  = s->stride;
      }
#endif
    } else {
      ierr = DumpSingle(ts,s,id);CHKERRQ(ierr);
      PetscPrintf(PETSC_COMM_WORLD,"\x1B[33mDump a single point to file\033[0m\n");
    }
  }
  /* first forward sweep only checkpoints once in each stride */
  if (!s->recompute && !s->save_stack) PetscFunctionReturn(0);

  ierr = ApplyRevolve(ts,s,stepnum,localstepnum,&whattodo);CHKERRQ(ierr);
  if (whattodo == 2){
    if (localstepnum < s->top) SETERRQ(s->comm,PETSC_ERR_MEMC,"Illegal modification of a non-top stack element");
    ierr = ElementCreate(ts,s,&e,localstepnum,time,X);CHKERRQ(ierr);
    ierr = StackPush(s,e);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "GetTrajTLR"
static PetscErrorCode GetTrajTLR(TS ts,Stack *s,PetscInt stepnum)
{
#ifdef PETSC_HAVE_REVOLVE
  PetscInt       whattodo,shift;
#endif
  PetscInt       i,steps,localstepnum,id;
  PetscReal      stepsize;
  StackElement   e;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  localstepnum = stepnum%s->stride;
  if (localstepnum == 0 && stepnum != s->total_steps) { /* load from disk */
#ifdef PETSC_HAVE_REVOLVE
    PetscPrintf(PETSC_COMM_WORLD,"\x1B[33mLoad stack from file\033[0m\n");
#endif
    if (s->save_stack) {
      id = stepnum/s->stride;
      ierr = StackLoadAll(ts,s,id);CHKERRQ(ierr);
      s->top = s->stacksize-1;
    } else {
      id = stepnum/s->stride-1;
      ierr = LoadSingle(ts,s,id);CHKERRQ(ierr);
    }
#ifdef PETSC_HAVE_REVOLVE
    revolve_reset();
    revolve_create_offline(s->stride,s->max_cps_ram);
    s->rctx->check = 0;
    s->rctx->capo  = 0;
    s->rctx->fine  = s->stride;
#endif
    if (s->save_stack) {
#ifdef PETSC_HAVE_REVOLVE
      whattodo = 0;
      while(whattodo!=3) { /* stupid revolve */
        whattodo = revolve_action(&s->rctx->check,&s->rctx->capo,&s->rctx->fine,s->rctx->snaps_in,&s->rctx->info,&s->rctx->where);
      }
#endif
    } else {
      /* save ts context */
      steps = ts->steps;
      ts->steps = ts->total_steps;
      s->recompute = PETSC_TRUE;
      for (i=ts->steps;i<stepnum;i++) { /* assume fixed step size */
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
      ts->steps = steps;
      ts->total_steps = stepnum;
    }
  }
#ifdef PETSC_HAVE_REVOLVE
  if ( s->rctx->reverseonestep) { /* ready for the reverse step */
    ierr = TSGetTimeStep(ts,&stepsize);CHKERRQ(ierr);
    ierr = TSSetTimeStep(ts,-stepsize);CHKERRQ(ierr);
    s->rctx->reverseonestep = PETSC_FALSE;
    PetscFunctionReturn(0);
  }
#endif
  if ((!s->solution_only && stepnum == 0) || stepnum == s->total_steps) {
    ierr = TSGetTimeStep(ts,&stepsize);CHKERRQ(ierr);
    ierr = TSSetTimeStep(ts,-stepsize);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  if (s->solution_only) {
#ifdef PETSC_HAVE_REVOLVE
    s->rctx->capo = stepnum;
    shift = 0;
    whattodo = revolve_action(&s->rctx->check,&s->rctx->capo,&s->rctx->fine,s->rctx->snaps_in,&s->rctx->info,&s->rctx->where);
    printwhattodo(whattodo,s->rctx,shift);
#endif
  }
  /* restore a checkpoint */
  ierr = StackTop(s,&e);CHKERRQ(ierr);
  if (e && e->stepnum >= stepnum) {
    SETERRQ2(s->comm,PETSC_ERR_ARG_OUTOFRANGE,"The current step no. is %D, but the step number at top of the stack is %D",stepnum,e->stepnum);
  }
  ierr = UpdateTS(ts,s,e);CHKERRQ(ierr);
  if (s->solution_only || (!s->solution_only && e->stepnum<stepnum)) { /* must recompute */
    steps = ts->steps;
    s->recompute = PETSC_TRUE;
    ierr = TSGetTimeStep(ts,&stepsize);CHKERRQ(ierr);
    ierr = TSSetTimeStep(ts,-stepsize);CHKERRQ(ierr);
    ierr = ReCompute(ts,s,e,stepnum);CHKERRQ(ierr);
    ts->steps = steps;
    ts->total_steps = stepnum;
    ierr = TSGetTimeStep(ts,&stepsize);CHKERRQ(ierr);
    ierr = TSSetTimeStep(ts,-stepsize);CHKERRQ(ierr);
    #ifdef PETSC_HAVE_REVOLVE
      s->rctx->reverseonestep = PETSC_FALSE;
    #endif
  } else {
    ierr = ElementDestroy(s,e);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SetTrajTLNR"
static PetscErrorCode SetTrajTLNR(TS ts,Stack *s,PetscInt stepnum,PetscReal time,Vec X)
{
  PetscInt       whattodo,localstepnum,id;
  StackElement   e;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  localstepnum = stepnum%s->stride;
  if (localstepnum == 0 && stepnum != s->total_steps && stepnum != 0  && !s->recompute) {
    id = stepnum/s->stride;
    if (s->save_stack) {
      ierr = StackDumpAll(ts,s,id);CHKERRQ(ierr);
      s->top = -1; /* reset top */
    } else {
      ierr = DumpSingle(ts,s,id);CHKERRQ(ierr);
    }
  }

  ierr = ApplyRevolve(ts,s,stepnum,localstepnum,&whattodo);CHKERRQ(ierr);
  if (whattodo == 2){
    if (localstepnum < s->top) SETERRQ(s->comm,PETSC_ERR_MEMC,"Illegal modification of a non-top stack element");
    ierr = ElementCreate(ts,s,&e,localstepnum,time,X);CHKERRQ(ierr);
    ierr = StackPush(s,e);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "GetTrajTLNR"
static PetscErrorCode GetTrajTLNR(TS ts,Stack *s,PetscInt stepnum)
{
  PetscInt       steps,id,localstepnum;
  PetscReal      stepsize;
  StackElement   e;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  localstepnum = stepnum%s->stride;
  if (stepnum != s->total_steps && localstepnum==0) {
    id = stepnum/s->stride;
    if (s->save_stack) {
      ierr = StackLoadAll(ts,s,id);CHKERRQ(ierr);
    } else {
      ierr = LoadSingle(ts,s,id);CHKERRQ(ierr);
    }
  }
  if ((!s->solution_only && stepnum == 0) || stepnum == s->total_steps) {
    ierr = TSGetTimeStep(ts,&stepsize);CHKERRQ(ierr);
    ierr = TSSetTimeStep(ts,-stepsize);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  /* restore a checkpoint */
  ierr = StackTop(s,&e);CHKERRQ(ierr);
  if (e && e->stepnum >= stepnum) {
    SETERRQ2(s->comm,PETSC_ERR_ARG_OUTOFRANGE,"The current step no. is %D, but the step number at top of the stack is %D",stepnum,e->stepnum);
  }
  ierr = UpdateTS(ts,s,e);CHKERRQ(ierr);
  if (s->solution_only || (!s->solution_only && e->stepnum<stepnum)) { /* must recompute */
    steps = ts->steps;
    s->recompute = PETSC_TRUE;
    ierr = TSGetTimeStep(ts,&stepsize);CHKERRQ(ierr);
    ierr = TSSetTimeStep(ts,-stepsize);CHKERRQ(ierr);
    ierr = ReCompute(ts,s,e,stepnum);CHKERRQ(ierr);
    ts->steps = steps;
    ts->total_steps = stepnum;
    ierr = TSGetTimeStep(ts,&stepsize);CHKERRQ(ierr);
    ierr = TSSetTimeStep(ts,-stepsize);CHKERRQ(ierr);
  } else {
    ierr = ElementDestroy(s,e);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SetTrajRMS"
static PetscErrorCode SetTrajRMS(TS ts,Stack *s,PetscInt stepnum,PetscReal time,Vec X)
{
  PetscInt       whattodo;
  StackElement   e;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = ApplyRevolve(ts,s,stepnum,stepnum,&whattodo);CHKERRQ(ierr);
  if (whattodo == 2){
    if (stepnum < s->top) SETERRQ(s->comm,PETSC_ERR_MEMC,"Illegal modification of a non-top stack element");
    ierr = ElementCreate(ts,s,&e,stepnum,time,X);CHKERRQ(ierr);
    ierr = StackPush(s,e);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "GetTrajRMS"
static PetscErrorCode GetTrajRMS(TS ts,Stack *s,PetscInt stepnum)
{
#ifdef PETSC_HAVE_REVOLVE
  PetscInt       whattodo,shift;
#endif
  PetscInt       steps;
  PetscReal      stepsize;
  StackElement   e;
  PetscErrorCode ierr;

  PetscFunctionBegin;
#ifdef PETSC_HAVE_REVOLVE
  if ( s->rctx->reverseonestep) { /* ready for the reverse step */
    ierr = TSGetTimeStep(ts,&stepsize);CHKERRQ(ierr);
    ierr = TSSetTimeStep(ts,-stepsize);CHKERRQ(ierr);
    s->rctx->reverseonestep = PETSC_FALSE;
    PetscFunctionReturn(0);
  }
#endif
  if ((!s->solution_only && stepnum == 0) || stepnum == s->total_steps) {
    ierr = TSGetTimeStep(ts,&stepsize);CHKERRQ(ierr);
    ierr = TSSetTimeStep(ts,-stepsize);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  if (s->solution_only) {
#ifdef PETSC_HAVE_REVOLVE
    s->rctx->capo = stepnum;
    shift = 0;
    whattodo = revolve_action(&s->rctx->check,&s->rctx->capo,&s->rctx->fine,s->rctx->snaps_in,&s->rctx->info,&s->rctx->where);
    printwhattodo(whattodo,s->rctx,shift);
#endif
  }
  /* restore a checkpoint */
  if (!s->rctx->where) {
    ierr = LoadSingle(ts,s,stepnum);CHKERRQ(ierr);
  } else {
    ierr = StackTop(s,&e);CHKERRQ(ierr);
    if (e && e->stepnum >= stepnum) {
      SETERRQ2(s->comm,PETSC_ERR_ARG_OUTOFRANGE,"The current step no. is %D, but the step number at top of the stack is %D",stepnum,e->stepnum);
    }
    ierr = UpdateTS(ts,s,e);CHKERRQ(ierr);
  }

  if (s->solution_only || (!s->solution_only && e->stepnum<stepnum)) { /* must recompute */
    steps = ts->steps;
    s->recompute = PETSC_TRUE;
    ierr = TSGetTimeStep(ts,&stepsize);CHKERRQ(ierr);
    ierr = TSSetTimeStep(ts,-stepsize);CHKERRQ(ierr);
    ierr = ReCompute(ts,s,e,stepnum);CHKERRQ(ierr);
    ts->steps = steps;
    ts->total_steps = stepnum;
    ierr = TSGetTimeStep(ts,&stepsize);CHKERRQ(ierr);
    ierr = TSSetTimeStep(ts,-stepsize);CHKERRQ(ierr);
    #ifdef PETSC_HAVE_REVOLVE
      s->rctx->reverseonestep = PETSC_FALSE;
    #endif
  } else {
    ierr = ElementDestroy(s,e);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSTrajectorySet_Memory"
PetscErrorCode TSTrajectorySet_Memory(TSTrajectory tj,TS ts,PetscInt stepnum,PetscReal time,Vec X)
{
  Stack          *s = (Stack*)tj->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!s->recompute) { /* use global stepnum in the forward sweep */
    ierr = TSGetTotalSteps(ts,&stepnum);CHKERRQ(ierr);
  }
  /* for consistency */
  if (!s->recompute && stepnum == 0) ts->ptime_prev = ts->ptime-ts->time_step;
  switch (s->stype) {
    case NONE:
      ierr = SetTrajN(ts,s,stepnum,time,X);CHKERRQ(ierr);
      break;
    case TWO_LEVEL_NOREVOLVE:
      ierr = SetTrajTLNR(ts,s,stepnum,time,X);CHKERRQ(ierr);
      break;
    case TWO_LEVEL_REVOLVE:
      ierr = SetTrajTLR(ts,s,stepnum,time,X);CHKERRQ(ierr);
      break;
    case REVOLVE_OFFLINE:
      ierr = SetTrajROF(ts,s,stepnum,time,X);CHKERRQ(ierr);
      break;
    case REVOLVE_ONLINE:
      ierr = SetTrajRON(ts,s,stepnum,time,X);CHKERRQ(ierr);
      break;
    case REVOLVE_MULTISTAGE:
      ierr = SetTrajRMS(ts,s,stepnum,time,X);CHKERRQ(ierr);
      break;
    default:
      break;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSTrajectoryGet_Memory"
PetscErrorCode TSTrajectoryGet_Memory(TSTrajectory tj,TS ts,PetscInt stepnum,PetscReal *t)
{
  Stack          *s = (Stack*)tj->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSGetTotalSteps(ts,&stepnum);CHKERRQ(ierr);
  if (stepnum == 0) PetscFunctionReturn(0);
  switch (s->stype) {
    case NONE:
      ierr = GetTrajN(ts,s,stepnum);CHKERRQ(ierr);
      break;
    case TWO_LEVEL_NOREVOLVE:
      ierr = GetTrajTLNR(ts,s,stepnum);CHKERRQ(ierr);
      break;
    case TWO_LEVEL_REVOLVE:
      ierr = GetTrajTLR(ts,s,stepnum);CHKERRQ(ierr);
      break;
    case REVOLVE_OFFLINE:
      ierr = GetTrajROF(ts,s,stepnum);CHKERRQ(ierr);
      break;
    case REVOLVE_ONLINE:
      ierr = GetTrajRON(ts,s,stepnum);CHKERRQ(ierr);
      break;
    case REVOLVE_MULTISTAGE:
      ierr = GetTrajRMS(ts,s,stepnum);CHKERRQ(ierr);
      break;
    default:
      break;
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
  if (s->stype > TWO_LEVEL_NOREVOLVE) {
#ifdef PETSC_HAVE_REVOLVE
    revolve_reset();
#endif
  }
  ierr = StackDestroy(&s);CHKERRQ(ierr);
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
  s->stype        = NONE;
  s->max_cps_ram  = -1; /* -1 indicates that it is not set */
  s->max_cps_disk = -1; /* -1 indicates that it is not set */
  s->stride       = 0; /* if not zero, two-level checkpointing will be used */
  s->use_online   = PETSC_FALSE;
  s->save_stack   = PETSC_TRUE;
  s->solution_only= PETSC_TRUE;
  tj->data        = s;
  PetscFunctionReturn(0);
}
