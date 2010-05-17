#define PETSC_DLL
/*
      PETSc code to log object creation and destruction and PETSc events.
*/
#include "petscsys.h"        /*I    "petscsys.h"   I*/
#include "petsctime.h"
#if defined(PETSC_HAVE_MPE)
#include "mpe.h"
#endif
#include <stdarg.h>
#include <sys/types.h>
#if defined(PETSC_HAVE_STDLIB_H)
#include <stdlib.h>
#endif
#if defined(PETSC_HAVE_MALLOC_H)
#include <malloc.h>
#endif
#include "../src/sys/plog/plog.h"

PetscLogEvent  PETSC_LARGEST_EVENT  = PETSC_EVENT;

#if defined(PETSC_CLANGUAGE_CXX) && !defined(PETSC_USE_EXTERN_CXX)
std::map<std::string,PETSc::LogEvent> PETSc::Log::event_registry;
std::map<std::string,PETSc::LogStage> PETSc::Log::stage_registry;
#endif

#if defined(PETSC_USE_LOG)
#include "petscmachineinfo.h"
#include "petscconfiginfo.h"

/* used in the MPI_XXX() count macros in petsclog.h */

/* Action and object logging variables */
Action    *actions    = PETSC_NULL;
Object    *objects    = PETSC_NULL;
PetscTruth logActions = PETSC_FALSE;
PetscTruth logObjects = PETSC_FALSE;
int        numActions = 0, maxActions = 100;
int        numObjects = 0, maxObjects = 100;
int        numObjectsDestroyed = 0;

/* Global counters */
PetscLogDouble PETSC_DLLEXPORT BaseTime        = 0.0;
PetscLogDouble PETSC_DLLEXPORT _TotalFlops     = 0.0; /* The number of flops */
PetscLogDouble PETSC_DLLEXPORT petsc_tmp_flops = 0.0; /* The incremental number of flops */
PetscLogDouble PETSC_DLLEXPORT send_ct         = 0.0; /* The number of sends */
PetscLogDouble PETSC_DLLEXPORT recv_ct         = 0.0; /* The number of receives */
PetscLogDouble PETSC_DLLEXPORT send_len        = 0.0; /* The total length of all sent messages */
PetscLogDouble PETSC_DLLEXPORT recv_len        = 0.0; /* The total length of all received messages */
PetscLogDouble PETSC_DLLEXPORT isend_ct        = 0.0; /* The number of immediate sends */
PetscLogDouble PETSC_DLLEXPORT irecv_ct        = 0.0; /* The number of immediate receives */
PetscLogDouble PETSC_DLLEXPORT isend_len       = 0.0; /* The total length of all immediate send messages */
PetscLogDouble PETSC_DLLEXPORT irecv_len       = 0.0; /* The total length of all immediate receive messages */
PetscLogDouble PETSC_DLLEXPORT wait_ct         = 0.0; /* The number of waits */
PetscLogDouble PETSC_DLLEXPORT wait_any_ct     = 0.0; /* The number of anywaits */
PetscLogDouble PETSC_DLLEXPORT wait_all_ct     = 0.0; /* The number of waitalls */
PetscLogDouble PETSC_DLLEXPORT sum_of_waits_ct = 0.0; /* The total number of waits */
PetscLogDouble PETSC_DLLEXPORT allreduce_ct    = 0.0; /* The number of reductions */
PetscLogDouble PETSC_DLLEXPORT gather_ct       = 0.0; /* The number of gathers and gathervs */
PetscLogDouble PETSC_DLLEXPORT scatter_ct      = 0.0; /* The number of scatters and scattervs */

/* Logging functions */
PetscErrorCode PETSC_DLLEXPORT (*_PetscLogPHC)(PetscObject) = PETSC_NULL;
PetscErrorCode PETSC_DLLEXPORT (*_PetscLogPHD)(PetscObject) = PETSC_NULL;
PetscErrorCode PETSC_DLLEXPORT (*_PetscLogPLB)(PetscLogEvent, int, PetscObject, PetscObject, PetscObject, PetscObject) = PETSC_NULL;
PetscErrorCode PETSC_DLLEXPORT (*_PetscLogPLE)(PetscLogEvent, int, PetscObject, PetscObject, PetscObject, PetscObject) = PETSC_NULL;

/* Tracing event logging variables */
FILE          *tracefile       = PETSC_NULL;
int            tracelevel      = 0;
const char    *traceblanks     = "                                                                                                    ";
char           tracespace[128] = " ";
PetscLogDouble tracetime       = 0.0;
PetscTruth PetscLogBegin_PrivateCalled = PETSC_FALSE;

/*---------------------------------------------- General Functions --------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "PetscLogDestroy"
/*@C
  PetscLogDestroy - Destroys the object and event logging data and resets the global counters. 

  Not Collective

  Notes:
  This routine should not usually be used by programmers. Instead employ 
  PetscLogStagePush() and PetscLogStagePop().

  Level: developer

.keywords: log, destroy
.seealso: PetscLogDump(), PetscLogAllBegin(), PetscLogPrintSummary(), PetscLogStagePush(), PlogStagePop()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscLogDestroy(void) 
{
  StageLog       stageLog;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(actions);CHKERRQ(ierr);
  actions = PETSC_NULL;
  ierr = PetscFree(objects);CHKERRQ(ierr);
  objects =  PETSC_NULL;
  ierr = PetscLogSet(PETSC_NULL, PETSC_NULL);CHKERRQ(ierr);

  /* Resetting phase */
  ierr = PetscLogGetStageLog(&stageLog);CHKERRQ(ierr);
  ierr = StageLogDestroy(stageLog);CHKERRQ(ierr);
  _TotalFlops         = 0.0;
  numActions          = 0;
  numObjects          = 0;
  numObjectsDestroyed = 0;
  maxActions          = 100;
  maxObjects          = 100;
  actions    = PETSC_NULL;
  objects    = PETSC_NULL;
  logActions = PETSC_FALSE;
  logObjects = PETSC_FALSE;
  BaseTime        = 0.0;
  _TotalFlops     = 0.0; 
  petsc_tmp_flops = 0.0; 
  send_ct         = 0.0; 
  recv_ct         = 0.0; 
  send_len        = 0.0; 
  recv_len        = 0.0; 
  isend_ct        = 0.0; 
  irecv_ct        = 0.0; 
  isend_len       = 0.0; 
  irecv_len       = 0.0; 
  wait_ct         = 0.0; 
  wait_any_ct     = 0.0; 
  wait_all_ct     = 0.0; 
  sum_of_waits_ct = 0.0; 
  allreduce_ct    = 0.0; 
  gather_ct       = 0.0; 
  scatter_ct      = 0.0; 
  PETSC_LARGEST_EVENT  = PETSC_EVENT;
  _PetscLogPHC = PETSC_NULL;
  _PetscLogPHD = PETSC_NULL;
  tracefile       = PETSC_NULL;
  tracelevel      = 0;
  traceblanks     = "                                                                                                    ";
  tracespace[0] = ' '; tracespace[1] = 0;
  tracetime       = 0.0;
  PETSC_LARGEST_COOKIE = PETSC_SMALLEST_COOKIE;
  PETSC_OBJECT_COOKIE  = 0;
  _stageLog = 0;
  PetscLogBegin_PrivateCalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscLogSet"
/*@C
  PetscLogSet - Sets the logging functions called at the beginning and ending of every event.

  Not Collective

  Input Parameters:
+ b - The function called at beginning of event
- e - The function called at end of event

  Level: developer

.seealso: PetscLogDump(), PetscLogBegin(), PetscLogAllBegin(), PetscLogTraceBegin()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscLogSet(PetscErrorCode (*b)(PetscLogEvent, int, PetscObject, PetscObject, PetscObject, PetscObject),
            PetscErrorCode (*e)(PetscLogEvent, int, PetscObject, PetscObject, PetscObject, PetscObject))
{
  PetscFunctionBegin;
  _PetscLogPLB = b;
  _PetscLogPLE = e;
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_CHUD)
#include <CHUD/CHUD.h>
#endif
#if defined(PETSC_HAVE_PAPI)
#include "papi.h"
int PAPIEventSet = PAPI_NULL;
#endif

/*------------------------------------------- Initialization Functions ----------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "PetscLogBegin_Private"
PetscErrorCode PETSC_DLLEXPORT PetscLogBegin_Private(void) 
{
  int               stage;
  PetscTruth        opt;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (PetscLogBegin_PrivateCalled) PetscFunctionReturn(0);
  PetscLogBegin_PrivateCalled = PETSC_TRUE;

  ierr = PetscOptionsHasName(PETSC_NULL, "-log_exclude_actions", &opt);CHKERRQ(ierr);
  if (opt) {
    logActions = PETSC_FALSE;
  }
  ierr = PetscOptionsHasName(PETSC_NULL, "-log_exclude_objects", &opt);CHKERRQ(ierr);
  if (opt) {
    logObjects = PETSC_FALSE;
  }
  if (logActions) {
    ierr = PetscMalloc(maxActions * sizeof(Action), &actions);CHKERRQ(ierr);
  }
  if (logObjects) {
    ierr = PetscMalloc(maxObjects * sizeof(Object), &objects);CHKERRQ(ierr);
  }
  _PetscLogPHC = PetscLogObjCreateDefault;
  _PetscLogPHD = PetscLogObjDestroyDefault;
  /* Setup default logging structures */
  ierr = StageLogCreate(&_stageLog);CHKERRQ(ierr);
  ierr = StageLogRegister(_stageLog, "Main Stage", &stage);CHKERRQ(ierr);
#if defined(PETSC_HAVE_CHUD)
  ierr = chudInitialize();CHKERRQ(ierr);
  ierr = chudAcquireSamplingFacility(CHUD_BLOCKING);CHKERRQ(ierr);
  ierr = chudSetSamplingDevice(chudCPU1Dev);CHKERRQ(ierr);
  ierr = chudSetStartDelay(0,chudNanoSeconds);CHKERRQ(ierr);
  ierr = chudClearPMCMode(chudCPU1Dev,chudUnused);CHKERRQ(ierr);
  ierr = chudClearPMCs();CHKERRQ(ierr);
  /* ierr = chudSetPMCMuxPosition(chudCPU1Dev,0,0);CHKERRQ(ierr); */
  printf("%s\n",chudGetEventName(chudCPU1Dev,PMC_1,193));
  printf("%s\n",chudGetEventDescription(chudCPU1Dev,PMC_1,193));
  printf("%s\n",chudGetEventNotes(chudCPU1Dev,PMC_1,193));
  ierr = chudSetPMCEvent(chudCPU1Dev,PMC_1,193);CHKERRQ(ierr);
  ierr = chudSetPMCMode(chudCPU1Dev,PMC_1,chudCounter);CHKERRQ(ierr);
  ierr = chudSetPrivilegeFilter(chudCPU1Dev,PMC_1,chudCountUserEvents);CHKERRQ(ierr);
  ierr = chudSetPMCEventMask(chudCPU1Dev,PMC_1,0xFE);CHKERRQ(ierr);
  if (!chudIsEventValid(chudCPU1Dev,PMC_1,193)) SETERRQ1(PETSC_ERR_SUP,"Event is not valid %d",193);
  ierr = chudStartPMCs();CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_PAPI)
  ierr = PAPI_library_init(PAPI_VER_CURRENT);
  if (ierr != PAPI_VER_CURRENT) SETERRQ(PETSC_ERR_LIB,"Cannot initialize PAPI");
  ierr = PAPI_query_event(PAPI_FP_INS);CHKERRQ(ierr);
  ierr = PAPI_create_eventset(&PAPIEventSet);CHKERRQ(ierr);
  ierr = PAPI_add_event(PAPIEventSet,PAPI_FP_INS);CHKERRQ(ierr);
  ierr = PAPI_start(PAPIEventSet);CHKERRQ(ierr);
#endif

  /* All processors sync here for more consistent logging */
  ierr = MPI_Barrier(PETSC_COMM_WORLD);CHKERRQ(ierr);
  PetscTime(BaseTime);
  ierr = PetscLogStagePush(stage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscLogBegin"
/*@C
  PetscLogBegin - Turns on logging of objects and events. This logs flop
  rates and object creation and should not slow programs down too much.
  This routine may be called more than once.

  Collective over PETSC_COMM_WORLD

  Options Database Keys:
+ -log_summary - Prints summary of flop and timing information to the 
                  screen (for code compiled with PETSC_USE_LOG)
- -log - Prints detailed log information (for code compiled with PETSC_USE_LOG)

  Usage:
.vb
      PetscInitialize(...);
      PetscLogBegin();
       ... code ...
      PetscLogPrintSummary(MPI_Comm,filename); or PetscLogDump(); 
      PetscFinalize();
.ve

  Notes:
  PetscLogPrintSummary(MPI_Comm,filename) or PetscLogDump() actually cause the printing of 
  the logging information.

  Level: advanced

.keywords: log, begin
.seealso: PetscLogDump(), PetscLogAllBegin(), PetscLogPrintSummary(), PetscLogTraceBegin()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscLogBegin(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLogSet(PetscLogEventBeginDefault, PetscLogEventEndDefault);CHKERRQ(ierr);
  ierr = PetscLogBegin_Private();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscLogAllBegin"
/*@C
  PetscLogAllBegin - Turns on extensive logging of objects and events. Logs 
  all events. This creates large log files and slows the program down.

  Collective on PETSC_COMM_WORLD

  Options Database Keys:
. -log_all - Prints extensive log information (for code compiled with PETSC_USE_LOG)

  Usage:
.vb
     PetscInitialize(...);
     PetscLogAllBegin();
     ... code ...
     PetscLogDump(filename);
     PetscFinalize();
.ve

  Notes:
  A related routine is PetscLogBegin (with the options key -log), which is 
  intended for production runs since it logs only flop rates and object
  creation (and shouldn't significantly slow the programs).

  Level: advanced

.keywords: log, all, begin
.seealso: PetscLogDump(), PetscLogBegin(), PetscLogTraceBegin()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscLogAllBegin(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLogSet(PetscLogEventBeginComplete, PetscLogEventEndComplete);CHKERRQ(ierr);
  ierr = PetscLogBegin_Private();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscLogTraceBegin"
/*@
  PetscLogTraceBegin - Activates trace logging.  Every time a PETSc event
  begins or ends, the event name is printed.

  Collective on PETSC_COMM_WORLD

  Input Parameter:
. file - The file to print trace in (e.g. stdout)

  Options Database Key:
. -log_trace [filename] - Activates PetscLogTraceBegin()

  Notes:
  PetscLogTraceBegin() prints the processor number, the execution time (sec),
  then "Event begin:" or "Event end:" followed by the event name.

  PetscLogTraceBegin() allows tracing of all PETSc calls, which is useful
  to determine where a program is hanging without running in the 
  debugger.  Can be used in conjunction with the -info option. 

  Level: intermediate

.seealso: PetscLogDump(), PetscLogAllBegin(), PetscLogPrintSummary(), PetscLogBegin()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscLogTraceBegin(FILE *file)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  tracefile = file;
  ierr = PetscLogSet(PetscLogEventBeginTrace, PetscLogEventEndTrace);CHKERRQ(ierr);
  ierr = PetscLogBegin_Private();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscLogActions"
/*@
  PetscLogActions - Determines whether actions are logged for the graphical viewer.

  Not Collective

  Input Parameter:
. flag - PETSC_TRUE if actions are to be logged

  Level: intermediate

  Note: Logging of actions continues to consume more memory as the program
  runs. Long running programs should consider turning this feature off.

  Options Database Keys:
. -log_exclude_actions - Turns off actions logging

.keywords: log, stage, register
.seealso: PetscLogStagePush(), PetscLogStagePop()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscLogActions(PetscTruth flag) 
{
  PetscFunctionBegin;
  logActions = flag;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscLogObjects"
/*@
  PetscLogObjects - Determines whether objects are logged for the graphical viewer.

  Not Collective

  Input Parameter:
. flag - PETSC_TRUE if objects are to be logged

  Level: intermediate

  Note: Logging of objects continues to consume more memory as the program
  runs. Long running programs should consider turning this feature off.

  Options Database Keys:
. -log_exclude_objects - Turns off objects logging

.keywords: log, stage, register
.seealso: PetscLogStagePush(), PetscLogStagePop()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscLogObjects(PetscTruth flag) 
{
  PetscFunctionBegin;
  logObjects = flag;
  PetscFunctionReturn(0);
}

/*------------------------------------------------ Stage Functions --------------------------------------------------*/
#undef __FUNCT__
#define __FUNCT__ "PetscLogStageRegister"
/*@C
  PetscLogStageRegister - Attaches a charactor string name to a logging stage.

  Not Collective

  Input Parameter:
. sname - The name to associate with that stage

  Output Parameter:
. stage - The stage number

  Level: intermediate

.keywords: log, stage, register
.seealso: PetscLogStagePush(), PetscLogStagePop()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscLogStageRegister(const char sname[],PetscLogStage *stage) 
{
  StageLog       stageLog;
  PetscLogEvent  event;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLogGetStageLog(&stageLog);CHKERRQ(ierr);
  ierr = StageLogRegister(stageLog, sname, stage);CHKERRQ(ierr);
  /* Copy events already changed in the main stage, this sucks */
  ierr = EventPerfLogEnsureSize(stageLog->stageInfo[*stage].eventLog, stageLog->eventLog->numEvents);CHKERRQ(ierr);
  for(event = 0; event < stageLog->eventLog->numEvents; event++) {
    ierr = EventPerfInfoCopy(&stageLog->stageInfo[0].eventLog->eventInfo[event],
                             &stageLog->stageInfo[*stage].eventLog->eventInfo[event]);CHKERRQ(ierr);
  }
  ierr = ClassPerfLogEnsureSize(stageLog->stageInfo[*stage].classLog, stageLog->classLog->numClasses);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscLogStagePush"
/*@C
  PetscLogStagePush - This function pushes a stage on the stack.

  Not Collective

  Input Parameter:
. stage - The stage on which to log

  Usage:
  If the option -log_sumary is used to run the program containing the 
  following code, then 2 sets of summary data will be printed during
  PetscFinalize().
.vb
      PetscInitialize(int *argc,char ***args,0,0);
      [stage 0 of code]   
      PetscLogStagePush(1);
      [stage 1 of code]
      PetscLogStagePop();
      PetscBarrier(...);
      [more stage 0 of code]   
      PetscFinalize();
.ve
 
  Notes:
  Use PetscLogStageRegister() to register a stage.

  Level: intermediate

.keywords: log, push, stage
.seealso: PetscLogStagePop(), PetscLogStageRegister(), PetscBarrier()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscLogStagePush(PetscLogStage stage)
{
  StageLog       stageLog;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLogGetStageLog(&stageLog);CHKERRQ(ierr);
  ierr = StageLogPush(stageLog, stage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscLogStagePop"
/*@C
  PetscLogStagePop - This function pops a stage from the stack.

  Not Collective

  Usage:
  If the option -log_sumary is used to run the program containing the 
  following code, then 2 sets of summary data will be printed during
  PetscFinalize().
.vb
      PetscInitialize(int *argc,char ***args,0,0);
      [stage 0 of code]   
      PetscLogStagePush(1);
      [stage 1 of code]
      PetscLogStagePop();
      PetscBarrier(...);
      [more stage 0 of code]   
      PetscFinalize();
.ve

  Notes:  
  Use PetscLogStageRegister() to register a stage.

  Level: intermediate

.keywords: log, pop, stage
.seealso: PetscLogStagePush(), PetscLogStageRegister(), PetscBarrier()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscLogStagePop(void)
{
  StageLog       stageLog;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLogGetStageLog(&stageLog);CHKERRQ(ierr);
  ierr = StageLogPop(stageLog);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscLogStageSetActive"
/*@
  PetscLogStageSetActive - Determines stage activity for PetscLogEventBegin() and PetscLogEventEnd().

  Not Collective 

  Input Parameters:
+ stage    - The stage
- isActive - The activity flag, PETSC_TRUE for logging, else PETSC_FALSE (defaults to PETSC_TRUE)

  Level: intermediate

.seealso: PetscLogStagePush(), PetscLogStagePop(), PetscLogEventBegin(), PetscLogEventEnd(), PreLoadBegin(), PreLoadEnd(), PreLoadStage()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscLogStageSetActive(PetscLogStage stage, PetscTruth isActive) 
{
  StageLog       stageLog;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLogGetStageLog(&stageLog);CHKERRQ(ierr);
  ierr = StageLogSetActive(stageLog, stage, isActive);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscLogStageGetActive"
/*@
  PetscLogStageGetActive - Returns stage activity for PetscLogEventBegin() and PetscLogEventEnd().

  Not Collective 

  Input Parameter:
. stage    - The stage

  Output Parameter:
. isActive - The activity flag, PETSC_TRUE for logging, else PETSC_FALSE (defaults to PETSC_TRUE)

  Level: intermediate

.seealso: PetscLogStagePush(), PetscLogStagePop(), PetscLogEventBegin(), PetscLogEventEnd(), PreLoadBegin(), PreLoadEnd(), PreLoadStage()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscLogStageGetActive(PetscLogStage stage, PetscTruth *isActive)
{
  StageLog       stageLog;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLogGetStageLog(&stageLog);CHKERRQ(ierr);
  ierr = StageLogGetActive(stageLog, stage, isActive);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscLogStageSetVisible"
/*@
  PetscLogStageSetVisible - Determines stage visibility in PetscLogPrintSummary()

  Not Collective 

  Input Parameters:
+ stage     - The stage
- isVisible - The visibility flag, PETSC_TRUE to print, else PETSC_FALSE (defaults to PETSC_TRUE)

  Level: intermediate

.seealso: PetscLogStagePush(), PetscLogStagePop(), PetscLogPrintSummary()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscLogStageSetVisible(PetscLogStage stage, PetscTruth isVisible)
{
  StageLog       stageLog;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLogGetStageLog(&stageLog);CHKERRQ(ierr);
  ierr = StageLogSetVisible(stageLog, stage, isVisible);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscLogStageGetVisible"
/*@
  PetscLogStageGetVisible - Returns stage visibility in PetscLogPrintSummary()

  Not Collective 

  Input Parameter:
. stage     - The stage

  Output Parameter:
. isVisible - The visibility flag, PETSC_TRUE to print, else PETSC_FALSE (defaults to PETSC_TRUE)

  Level: intermediate

.seealso: PetscLogStagePush(), PetscLogStagePop(), PetscLogPrintSummary()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscLogStageGetVisible(PetscLogStage stage, PetscTruth *isVisible)
{
  StageLog       stageLog;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLogGetStageLog(&stageLog);CHKERRQ(ierr);
  ierr = StageLogGetVisible(stageLog, stage, isVisible);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscLogStageGetId"
/*@C
  PetscLogStageGetId - Returns the stage id when given the stage name.

  Not Collective 

  Input Parameter:
. name  - The stage name

  Output Parameter:
. stage - The stage

  Level: intermediate

.seealso: PetscLogStagePush(), PetscLogStagePop(), PreLoadBegin(), PreLoadEnd(), PreLoadStage()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscLogStageGetId(const char name[], PetscLogStage *stage)
{
  StageLog       stageLog;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLogGetStageLog(&stageLog);CHKERRQ(ierr);
  ierr = StageLogGetStage(stageLog, name, stage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*------------------------------------------------ Event Functions --------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "PetscLogEventRegister"
/*@C
  PetscLogEventRegister - Registers an event name for logging operations in an application code. 

  Not Collective

  Input Parameter:
+ name   - The name associated with the event
- cookie - The cookie associated to the class for this event, obtain either with
           PetscCookieRegister() or use a predefined one such as KSP_COOKIE, SNES_COOKIE
            
  Output Parameter:
. event - The event id for use with PetscLogEventBegin() and PetscLogEventEnd().

  Example of Usage:
.vb
      PetscLogEvent USER_EVENT;
      PetscCookie cookie;
      PetscLogDouble user_event_flops;
      PetscCookieRegister("class name",&cookie);
      PetscLogEventRegister("User event name",cookie,&USER_EVENT);
      PetscLogEventBegin(USER_EVENT,0,0,0,0);
         [code segment to monitor]
         PetscLogFlops(user_event_flops);
      PetscLogEventEnd(USER_EVENT,0,0,0,0);
.ve

  Notes: 
  PETSc automatically logs library events if the code has been
  compiled with -DPETSC_USE_LOG (which is the default) and -log,
  -log_summary, or -log_all are specified.  PetscLogEventRegister() is
  intended for logging user events to supplement this PETSc
  information. 

  PETSc can gather data for use with the utilities Upshot/Nupshot
  (part of the MPICH distribution).  If PETSc has been compiled
  with flag -DPETSC_HAVE_MPE (MPE is an additional utility within
  MPICH), the user can employ another command line option, -log_mpe,
  to create a logfile, "mpe.log", which can be visualized
  Upshot/Nupshot. 

  The cookie is associated with each event so that classes of events
  can be disabled simultaneously, such as all matrix events. The user
  can either use an existing cookie, such as MAT_COOKIE, or create
  their own as shown in the example.

  Level: intermediate

.keywords: log, event, register
.seealso: PetscLogEventBegin(), PetscLogEventEnd(), PetscLogFlops(),
          PetscLogEventMPEActivate(), PetscLogEventMPEDeactivate(),
          PetscLogEventActivate(), PetscLogEventDeactivate(), PetscCookieRegister()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscLogEventRegister(const char name[],PetscCookie cookie,PetscLogEvent *event) 
{
  StageLog       stageLog;
  int            stage;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  *event = PETSC_DECIDE;
  ierr = PetscLogGetStageLog(&stageLog);CHKERRQ(ierr);
  ierr = EventRegLogRegister(stageLog->eventLog, name, cookie, event);CHKERRQ(ierr);
  for(stage = 0; stage < stageLog->numStages; stage++) {
    ierr = EventPerfLogEnsureSize(stageLog->stageInfo[stage].eventLog, stageLog->eventLog->numEvents);CHKERRQ(ierr);
    ierr = ClassPerfLogEnsureSize(stageLog->stageInfo[stage].classLog, stageLog->classLog->numClasses);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscLogEventActivate"
/*@
  PetscLogEventActivate - Indicates that a particular event should be logged.

  Not Collective

  Input Parameter:
. event - The event id

  Usage:
.vb
      PetscLogEventDeactivate(VEC_SetValues);
        [code where you do not want to log VecSetValues()]
      PetscLogEventActivate(VEC_SetValues);
        [code where you do want to log VecSetValues()]
.ve 

  Note:
  The event may be either a pre-defined PETSc event (found in include/petsclog.h)
  or an event number obtained with PetscLogEventRegister().

  Level: advanced

.keywords: log, event, activate
.seealso: PetscLogEventMPEDeactivate(),PetscLogEventMPEActivate(),PlogEventDeactivate()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscLogEventActivate(PetscLogEvent event)
{
  StageLog       stageLog;
  int            stage;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLogGetStageLog(&stageLog);CHKERRQ(ierr);
  ierr = StageLogGetCurrent(stageLog, &stage);CHKERRQ(ierr);
  ierr = EventPerfLogActivate(stageLog->stageInfo[stage].eventLog, event);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscLogEventDeactivate"
/*@
  PetscLogEventDeactivate - Indicates that a particular event should not be logged. 

  Not Collective

  Input Parameter:
. event - The event id

  Usage:
.vb
      PetscLogEventDeactivate(VEC_SetValues);
        [code where you do not want to log VecSetValues()]
      PetscLogEventActivate(VEC_SetValues);
        [code where you do want to log VecSetValues()]
.ve 

  Note: 
  The event may be either a pre-defined PETSc event (found in
  include/petsclog.h) or an event number obtained with PetscLogEventRegister()).

  Level: advanced

.keywords: log, event, deactivate
.seealso: PetscLogEventMPEDeactivate(),PetscLogEventMPEActivate(),PlogEventActivate()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscLogEventDeactivate(PetscLogEvent event)
{
  StageLog       stageLog;
  int            stage;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLogGetStageLog(&stageLog);CHKERRQ(ierr);
  ierr = StageLogGetCurrent(stageLog, &stage);CHKERRQ(ierr);
  ierr = EventPerfLogDeactivate(stageLog->stageInfo[stage].eventLog, event);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscLogEventSetActiveAll"
/*@
  PetscLogEventSetActiveAll - Sets the event activity in every stage.

  Not Collective

  Input Parameters:
+ event    - The event id
- isActive - The activity flag determining whether the event is logged

  Level: advanced

.keywords: log, event, activate
.seealso: PetscLogEventMPEDeactivate(),PetscLogEventMPEActivate(),PlogEventActivate(),PlogEventDeactivate()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscLogEventSetActiveAll(PetscLogEvent event, PetscTruth isActive)
{
  StageLog       stageLog;
  int            stage;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLogGetStageLog(&stageLog);CHKERRQ(ierr);
  for(stage = 0; stage < stageLog->numStages; stage++) {
    if (isActive) {
      ierr = EventPerfLogActivate(stageLog->stageInfo[stage].eventLog, event);CHKERRQ(ierr);
    } else {
      ierr = EventPerfLogDeactivate(stageLog->stageInfo[stage].eventLog, event);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscLogEventActivateClass"
/*@
  PetscLogEventActivateClass - Activates event logging for a PETSc object class.

  Not Collective

  Input Parameter:
. cookie - The event class, for example MAT_COOKIE, SNES_COOKIE, etc.

  Level: developer

.keywords: log, event, activate, class
.seealso: PetscInfoActivate(),PetscInfo(),PetscInfoAllow(),PetscLogEventDeactivateClass(), PetscLogEventActivate(),PetscLogEventDeactivate()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscLogEventActivateClass(PetscCookie cookie) 
{
  StageLog       stageLog;
  int            stage;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLogGetStageLog(&stageLog);CHKERRQ(ierr);
  ierr = StageLogGetCurrent(stageLog, &stage);CHKERRQ(ierr);
  ierr = EventPerfLogActivateClass(stageLog->stageInfo[stage].eventLog, stageLog->eventLog, cookie);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscLogEventDeactivateClass"
/*@
  PetscLogEventDeactivateClass - Deactivates event logging for a PETSc object class.

  Not Collective

  Input Parameter:
. cookie - The event class, for example MAT_COOKIE, SNES_COOKIE, etc.

  Level: developer

.keywords: log, event, deactivate, class
.seealso: PetscInfoActivate(),PetscInfo(),PetscInfoAllow(),PetscLogEventActivateClass(), PetscLogEventActivate(),PetscLogEventDeactivate()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscLogEventDeactivateClass(PetscCookie cookie)
{
  StageLog       stageLog;
  int            stage;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLogGetStageLog(&stageLog);CHKERRQ(ierr);
  ierr = StageLogGetCurrent(stageLog, &stage);CHKERRQ(ierr);
  ierr = EventPerfLogDeactivateClass(stageLog->stageInfo[stage].eventLog, stageLog->eventLog, cookie);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
   PetscLogEventBegin - Logs the beginning of a user event. 

   Synopsis:
   PetscErrorCode PetscLogEventBegin(int e,PetscObject o1,PetscObject o2,PetscObject o3,
                       PetscObject o4)

   Not Collective

   Input Parameters:
+  e - integer associated with the event obtained from PetscLogEventRegister()
-  o1,o2,o3,o4 - objects associated with the event, or 0


   Fortran Synopsis:
   void PetscLogEventBegin(int e,PetscErrorCode ierr)

   Usage:
.vb
     PetscLogEvent USER_EVENT;
     PetscLogDouble user_event_flops;
     PetscLogEventRegister("User event",0,&USER_EVENT);
     PetscLogEventBegin(USER_EVENT,0,0,0,0);
        [code segment to monitor]
        PetscLogFlops(user_event_flops);
     PetscLogEventEnd(USER_EVENT,0,0,0,0);
.ve

   Notes:
   You need to register each integer event with the command 
   PetscLogEventRegister().  The source code must be compiled with 
   -DPETSC_USE_LOG, which is the default.

   PETSc automatically logs library events if the code has been
   compiled with -DPETSC_USE_LOG, and -log, -log_summary, or -log_all are
   specified.  PetscLogEventBegin() is intended for logging user events
   to supplement this PETSc information.

   Level: intermediate

.seealso: PetscLogEventRegister(), PetscLogEventEnd(), PetscLogFlops()

.keywords: log, event, begin
M*/

/*MC
   PetscLogEventEnd - Log the end of a user event.

   Synopsis:
   PetscErrorCode PetscLogEventEnd(int e,PetscObject o1,PetscObject o2,PetscObject o3,
                     PetscObject o4)

   Not Collective

   Input Parameters:
+  e - integer associated with the event obtained with PetscLogEventRegister()
-  o1,o2,o3,o4 - objects associated with the event, or 0


   Fortran Synopsis:
   void PetscLogEventEnd(int e,PetscErrorCode ierr)

   Usage:
.vb
     PetscLogEvent USER_EVENT;
     PetscLogDouble user_event_flops;
     PetscLogEventRegister("User event",0,&USER_EVENT,);
     PetscLogEventBegin(USER_EVENT,0,0,0,0);
        [code segment to monitor]
        PetscLogFlops(user_event_flops);
     PetscLogEventEnd(USER_EVENT,0,0,0,0);
.ve

   Notes:
   You should also register each additional integer event with the command 
   PetscLogEventRegister(). Source code must be compiled with 
   -DPETSC_USE_LOG, which is the default.

   PETSc automatically logs library events if the code has been
   compiled with -DPETSC_USE_LOG, and -log, -log_summary, or -log_all are
   specified.  PetscLogEventEnd() is intended for logging user events
   to supplement this PETSc information.

   Level: intermediate

.seealso: PetscLogEventRegister(), PetscLogEventBegin(), PetscLogFlops()

.keywords: log, event, end
M*/

/*MC
   PetscLogEventBarrierBegin - Logs the time in a barrier before an event.

   Synopsis:
   PetscErrorCode PetscLogEventBarrierBegin(int e,PetscObject o1,PetscObject o2,PetscObject o3,
                  PetscObject o4,MPI_Comm comm)

   Not Collective

   Input Parameters:
.  e - integer associated with the event obtained from PetscLogEventRegister()
.  o1,o2,o3,o4 - objects associated with the event, or 0
.  comm - communicator the barrier takes place over


   Usage:
.vb
     PetscLogEventBarrierBegin(VEC_NormBarrier,0,0,0,0,comm);
       MPI_Allreduce()
     PetscLogEventBarrierEnd(VEC_NormBarrier,0,0,0,0,comm);
.ve

   Notes:
   This is for logging the amount of time spent in a barrier for an event
   that requires synchronization. 

   Additional Notes:
   Synchronization events always come in pairs; for example, VEC_NormBarrier and 
   VEC_NormComm = VEC_NormBarrier + 1

   Level: advanced

.seealso: PetscLogEventRegister(), PetscLogEventEnd(), PetscLogFlops(), PetscLogEventBegin(),
          PetscLogEventBarrierEnd()

.keywords: log, event, begin, barrier
M*/

/*MC
   PetscLogEventBarrierEnd - Logs the time in a barrier before an event.

   Synopsis:
   PetscErrorCode PetscLogEventBarrierEnd(int e,PetscObject o1,PetscObject o2,PetscObject o3,
                  PetscObject o4,MPI_Comm comm)

   Collective on MPI_Comm

   Input Parameters:
.  e - integer associated with the event obtained from PetscLogEventRegister()
.  o1,o2,o3,o4 - objects associated with the event, or 0
.  comm - communicator the barrier takes place over


    Usage:
.vb
     PetscLogEventBarrierBegin(VEC_NormBarrier,0,0,0,0,comm);
       MPI_Allreduce()
     PetscLogEventBarrierEnd(VEC_NormBarrier,0,0,0,0,comm);
.ve

   Notes:
   This is for logging the amount of time spent in a barrier for an event
   that requires synchronization. 

   Additional Notes:
   Synchronization events always come in pairs; for example, VEC_NormBarrier and 
   VEC_NormComm = VEC_NormBarrier + 1

   Level: advanced

.seealso: PetscLogEventRegister(), PetscLogEventEnd(), PetscLogFlops(), PetscLogEventBegin(),
          PetscLogEventBarrierBegin()

.keywords: log, event, begin, barrier
M*/

#undef __FUNCT__  
#define __FUNCT__ "PetscLogEventGetId"
/*@C
  PetscLogEventGetId - Returns the event id when given the event name.

  Not Collective 

  Input Parameter:
. name  - The event name

  Output Parameter:
. event - The event

  Level: intermediate

.seealso: PetscLogEventBegin(), PetscLogEventEnd(), PetscLogStageGetId()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscLogEventGetId(const char name[], PetscLogEvent *event)
{
  StageLog       stageLog;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLogGetStageLog(&stageLog);CHKERRQ(ierr);
  ierr = EventRegLogGetEvent(stageLog->eventLog, name, event);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


/*------------------------------------------------ Output Functions -------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "PetscLogDump"
/*@C
  PetscLogDump - Dumps logs of objects to a file. This file is intended to 
  be read by bin/petscview. This program no longer exists.

  Collective on PETSC_COMM_WORLD

  Input Parameter:
. name - an optional file name

  Options Database Keys:
+ -log     - Prints basic log information (for code compiled with PETSC_USE_LOG)
- -log_all - Prints extensive log information (for code compiled with PETSC_USE_LOG)
   
  Usage:
.vb
     PetscInitialize(...);
     PetscLogBegin(); or PetscLogAllBegin(); 
     ... code ...
     PetscLogDump(filename);
     PetscFinalize();
.ve

  Notes:
  The default file name is 
$    Log.<rank>
  where <rank> is the processor number. If no name is specified, 
  this file will be used.

  Level: advanced

.keywords: log, dump
.seealso: PetscLogBegin(), PetscLogAllBegin(), PetscLogPrintSummary()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscLogDump(const char sname[]) 
{
  StageLog       stageLog;
  EventPerfInfo *eventInfo;
  FILE          *fd;
  char           file[PETSC_MAX_PATH_LEN], fname[PETSC_MAX_PATH_LEN];
  PetscLogDouble flops, _TotalTime;
  PetscMPIInt    rank;
  int            action, object, curStage;
  PetscLogEvent  event;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  /* Calculate the total elapsed time */
  PetscTime(_TotalTime);
  _TotalTime -= BaseTime;
  /* Open log file */
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank);CHKERRQ(ierr);
  if (sname) {
    sprintf(file, "%s.%d", sname, rank);
  } else {
    sprintf(file, "Log.%d", rank);
  }
  ierr = PetscFixFilename(file, fname);CHKERRQ(ierr);
  ierr = PetscFOpen(PETSC_COMM_WORLD, fname, "w", &fd);CHKERRQ(ierr);
  if ((!rank) && (!fd)) SETERRQ1(PETSC_ERR_FILE_OPEN, "Cannot open file: %s", fname);
  /* Output totals */
  ierr = PetscFPrintf(PETSC_COMM_WORLD, fd, "Total Flops %14e %16.8e\n", _TotalFlops, _TotalTime);
  ierr = PetscFPrintf(PETSC_COMM_WORLD, fd, "Clock Resolution %g\n", 0.0);
  /* Output actions */
  if (logActions) {
    ierr = PetscFPrintf(PETSC_COMM_WORLD, fd, "Actions accomplished %d\n", numActions);
    for(action = 0; action < numActions; action++) {
      ierr = PetscFPrintf(PETSC_COMM_WORLD, fd, "%g %d %d %d %d %d %d %g %g %g\n",
                          actions[action].time, actions[action].action, (int)actions[action].event, (int)actions[action].cookie, actions[action].id1,
                          actions[action].id2, actions[action].id3, actions[action].flops, actions[action].mem, actions[action].maxmem);
    }
  }
  /* Output objects */
  if (logObjects) {
    ierr = PetscFPrintf(PETSC_COMM_WORLD, fd, "Objects created %d destroyed %d\n", numObjects, numObjectsDestroyed);
    for(object = 0; object < numObjects; object++) {
      ierr = PetscFPrintf(PETSC_COMM_WORLD, fd, "Parent ID: %d Memory: %d\n", objects[object].parent, (int) objects[object].mem);
      if (!objects[object].name[0]) {
        ierr = PetscFPrintf(PETSC_COMM_WORLD, fd,"No Name\n");
      } else {
        ierr = PetscFPrintf(PETSC_COMM_WORLD, fd, "Name: %s\n", objects[object].name);
      }
      if (objects[object].info[0] != 0) {
        ierr = PetscFPrintf(PETSC_COMM_WORLD, fd, "No Info\n");
      } else {
        ierr = PetscFPrintf(PETSC_COMM_WORLD, fd, "Info: %s\n", objects[object].info);
      }
    }
  }
  /* Output events */
  ierr = PetscFPrintf(PETSC_COMM_WORLD, fd, "Event log:\n");
  ierr = PetscLogGetStageLog(&stageLog);CHKERRQ(ierr);
  ierr = StackTop(stageLog->stack, &curStage);CHKERRQ(ierr);
  eventInfo = stageLog->stageInfo[curStage].eventLog->eventInfo;
  for(event = 0; event < stageLog->stageInfo[curStage].eventLog->numEvents; event++) {
    if (eventInfo[event].time != 0.0) {
      flops = eventInfo[event].flops/eventInfo[event].time;
    } else {
      flops = 0.0;
    }
    ierr = PetscFPrintf(PETSC_COMM_WORLD, fd, "%d %16d %16g %16g %16g\n", event, eventInfo[event].count,
                        eventInfo[event].flops, eventInfo[event].time, flops);
  }
  ierr = PetscFClose(PETSC_COMM_WORLD, fd);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscLogPrintSummary"
/*@C
  PetscLogPrintSummary - Prints a summary of the logging.

  Collective over MPI_Comm

  Input Parameter:
+ comm - The MPI communicator (only one processor prints output)
- file - [Optional] The output file name

  Options Database Keys:
. -log_summary - Prints summary of log information (for code compiled with PETSC_USE_LOG)

  Usage:
.vb
     PetscInitialize(...);
     PetscLogBegin();
     ... code ...
     PetscLogPrintSummary(MPI_Comm,filename);
     PetscFinalize(...);
.ve

  Notes:
  By default the summary is printed to stdout.

  Level: beginner
   
.keywords: log, dump, print
.seealso: PetscLogBegin(), PetscLogDump()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscLogPrintSummary(MPI_Comm comm, const char filename[]) 
{
  FILE           *fd = PETSC_STDOUT;
  PetscLogDouble zero = 0.0;
  StageLog       stageLog;
  StageInfo     *stageInfo = PETSC_NULL;
  EventPerfInfo *eventInfo = PETSC_NULL;
  ClassPerfInfo *classInfo;
  char           arch[10], hostname[64], username[16], pname[PETSC_MAX_PATH_LEN], date[64];
  const char    *name;
  PetscLogDouble locTotalTime, TotalTime, TotalFlops;
  PetscLogDouble numMessages, messageLength, avgMessLen, numReductions;
  PetscLogDouble stageTime, flops, flopr, mem, mess, messLen, red;
  PetscLogDouble fracTime, fracFlops, fracMessages, fracLength, fracReductions, fracMess, fracMessLen, fracRed;
  PetscLogDouble fracStageTime, fracStageFlops, fracStageMess, fracStageMessLen, fracStageRed;
  PetscLogDouble min, max, tot, ratio, avg, x, y;
  PetscLogDouble minf, maxf, totf, ratf, mint, maxt, tott, ratt, ratCt, totm, totml, totr;
  PetscMPIInt    minCt, maxCt;
  PetscMPIInt    size, rank;
  PetscTruth    *localStageUsed,    *stageUsed;
  PetscTruth    *localStageVisible, *stageVisible;
  int            numStages, localNumEvents, numEvents;
  int            stage, lastStage, oclass;
  PetscLogEvent  event;
  PetscErrorCode ierr;
  char           version[256];

  PetscFunctionBegin;
  ierr = MPI_Comm_size(comm, &size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  /* Pop off any stages the user forgot to remove */
  lastStage = 0;
  ierr = PetscLogGetStageLog(&stageLog);CHKERRQ(ierr);
  ierr = StageLogGetCurrent(stageLog, &stage);CHKERRQ(ierr);
  while (stage >= 0) {
    lastStage = stage;
    ierr = StageLogPop(stageLog);CHKERRQ(ierr);
    ierr = StageLogGetCurrent(stageLog, &stage);CHKERRQ(ierr);
  }
  /* Get the total elapsed time */
  PetscTime(locTotalTime);  locTotalTime -= BaseTime;
  /* Open the summary file */
  if (filename) {
    ierr = PetscFOpen(comm, filename, "w", &fd);CHKERRQ(ierr);
  }

  ierr = PetscFPrintf(comm, fd, "************************************************************************************************************************\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(comm, fd, "***             WIDEN YOUR WINDOW TO 120 CHARACTERS.  Use 'enscript -r -fCourier9' to print this document            ***\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(comm, fd, "************************************************************************************************************************\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(comm, fd, "\n---------------------------------------------- PETSc Performance Summary: ----------------------------------------------\n\n");CHKERRQ(ierr);
  ierr = PetscGetArchType(arch, 10);CHKERRQ(ierr);
  ierr = PetscGetHostName(hostname, 64);CHKERRQ(ierr);
  ierr = PetscGetUserName(username, 16);CHKERRQ(ierr);
  ierr = PetscGetProgramName(pname, PETSC_MAX_PATH_LEN);CHKERRQ(ierr);
  ierr = PetscGetDate(date, 64);CHKERRQ(ierr);
  ierr = PetscGetVersion(version,256);CHKERRQ(ierr);
  if (size == 1) {
    ierr = PetscFPrintf(comm,fd,"%s on a %s named %s with %d processor, by %s %s\n", pname, arch, hostname, size, username, date);CHKERRQ(ierr);
  } else {
    ierr = PetscFPrintf(comm,fd,"%s on a %s named %s with %d processors, by %s %s\n", pname, arch, hostname, size, username, date);CHKERRQ(ierr);
  }
  ierr = PetscFPrintf(comm, fd, "Using %s\n", version);CHKERRQ(ierr);

  /* Must preserve reduction count before we go on */
  red  = allreduce_ct + gather_ct + scatter_ct;

  /* Calculate summary information */
  ierr = PetscFPrintf(comm, fd, "\n                         Max       Max/Min        Avg      Total \n");CHKERRQ(ierr);
  /*   Time */
  ierr = MPI_Allreduce(&locTotalTime, &min, 1, MPIU_PETSCLOGDOUBLE, MPI_MIN, comm);CHKERRQ(ierr);
  ierr = MPI_Allreduce(&locTotalTime, &max, 1, MPIU_PETSCLOGDOUBLE, MPI_MAX, comm);CHKERRQ(ierr);
  ierr = MPI_Allreduce(&locTotalTime, &tot, 1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm);CHKERRQ(ierr);
  avg  = (tot)/((PetscLogDouble) size);
  if (min != 0.0) ratio = max/min; else ratio = 0.0;
  ierr = PetscFPrintf(comm, fd, "Time (sec):           %5.3e   %10.5f   %5.3e\n", max, ratio, avg);CHKERRQ(ierr);
  TotalTime = tot;
  /*   Objects */
  avg  = (PetscLogDouble) numObjects;
  ierr = MPI_Allreduce(&avg,          &min, 1, MPIU_PETSCLOGDOUBLE, MPI_MIN, comm);CHKERRQ(ierr);
  ierr = MPI_Allreduce(&avg,          &max, 1, MPIU_PETSCLOGDOUBLE, MPI_MAX, comm);CHKERRQ(ierr);
  ierr = MPI_Allreduce(&avg,          &tot, 1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm);CHKERRQ(ierr);
  avg  = (tot)/((PetscLogDouble) size);
  if (min != 0.0) ratio = max/min; else ratio = 0.0;
  ierr = PetscFPrintf(comm, fd, "Objects:              %5.3e   %10.5f   %5.3e\n", max, ratio, avg);CHKERRQ(ierr);
  /*   Flops */
  ierr = MPI_Allreduce(&_TotalFlops,  &min, 1, MPIU_PETSCLOGDOUBLE, MPI_MIN, comm);CHKERRQ(ierr);
  ierr = MPI_Allreduce(&_TotalFlops,  &max, 1, MPIU_PETSCLOGDOUBLE, MPI_MAX, comm);CHKERRQ(ierr);
  ierr = MPI_Allreduce(&_TotalFlops,  &tot, 1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm);CHKERRQ(ierr);
  avg  = (tot)/((PetscLogDouble) size);
  if (min != 0.0) ratio = max/min; else ratio = 0.0;
  ierr = PetscFPrintf(comm, fd, "Flops:                %5.3e   %10.5f   %5.3e  %5.3e\n", max, ratio, avg, tot);CHKERRQ(ierr);
  TotalFlops = tot;
  /*   Flops/sec -- Must talk to Barry here */
  if (locTotalTime != 0.0) flops = _TotalFlops/locTotalTime; else flops = 0.0;
  ierr = MPI_Allreduce(&flops,        &min, 1, MPIU_PETSCLOGDOUBLE, MPI_MIN, comm);CHKERRQ(ierr);
  ierr = MPI_Allreduce(&flops,        &max, 1, MPIU_PETSCLOGDOUBLE, MPI_MAX, comm);CHKERRQ(ierr);
  ierr = MPI_Allreduce(&flops,        &tot, 1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm);CHKERRQ(ierr);
  avg  = (tot)/((PetscLogDouble) size);
  if (min != 0.0) ratio = max/min; else ratio = 0.0;
  ierr = PetscFPrintf(comm, fd, "Flops/sec:            %5.3e   %10.5f   %5.3e  %5.3e\n", max, ratio, avg, tot);CHKERRQ(ierr);
  /*   Memory */
  ierr = PetscMallocGetMaximumUsage(&mem);CHKERRQ(ierr);
  if (mem > 0.0) {
    ierr = MPI_Allreduce(&mem,          &max, 1, MPIU_PETSCLOGDOUBLE, MPI_MAX, comm);CHKERRQ(ierr);
    ierr = MPI_Allreduce(&mem,          &min, 1, MPIU_PETSCLOGDOUBLE, MPI_MIN, comm);CHKERRQ(ierr);
    ierr = MPI_Allreduce(&mem,          &tot, 1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm);CHKERRQ(ierr);
    avg  = (tot)/((PetscLogDouble) size);
    if (min != 0.0) ratio = max/min; else ratio = 0.0;
    ierr = PetscFPrintf(comm, fd, "Memory:               %5.3e   %10.5f              %5.3e\n", max, ratio, tot);CHKERRQ(ierr);
  }
  /*   Messages */
  mess = 0.5*(irecv_ct + isend_ct + recv_ct + send_ct);
  ierr = MPI_Allreduce(&mess,         &min, 1, MPIU_PETSCLOGDOUBLE, MPI_MIN, comm);CHKERRQ(ierr);
  ierr = MPI_Allreduce(&mess,         &max, 1, MPIU_PETSCLOGDOUBLE, MPI_MAX, comm);CHKERRQ(ierr);
  ierr = MPI_Allreduce(&mess,         &tot, 1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm);CHKERRQ(ierr);
  avg  = (tot)/((PetscLogDouble) size);
  if (min != 0.0) ratio = max/min; else ratio = 0.0;
  ierr = PetscFPrintf(comm, fd, "MPI Messages:         %5.3e   %10.5f   %5.3e  %5.3e\n", max, ratio, avg, tot);CHKERRQ(ierr);
  numMessages = tot;
  /*   Message Lengths */
  mess = 0.5*(irecv_len + isend_len + recv_len + send_len);
  ierr = MPI_Allreduce(&mess,         &min, 1, MPIU_PETSCLOGDOUBLE, MPI_MIN, comm);CHKERRQ(ierr);
  ierr = MPI_Allreduce(&mess,         &max, 1, MPIU_PETSCLOGDOUBLE, MPI_MAX, comm);CHKERRQ(ierr);
  ierr = MPI_Allreduce(&mess,         &tot, 1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm);CHKERRQ(ierr);
  if (numMessages != 0) avg = (tot)/(numMessages); else avg = 0.0;
  if (min != 0.0) ratio = max/min; else ratio = 0.0;
  ierr = PetscFPrintf(comm, fd, "MPI Message Lengths:  %5.3e   %10.5f   %5.3e  %5.3e\n", max, ratio, avg, tot);CHKERRQ(ierr);
  messageLength = tot;
  /*   Reductions */
  ierr = MPI_Allreduce(&red,          &min, 1, MPIU_PETSCLOGDOUBLE, MPI_MIN, comm);CHKERRQ(ierr);
  ierr = MPI_Allreduce(&red,          &max, 1, MPIU_PETSCLOGDOUBLE, MPI_MAX, comm);CHKERRQ(ierr);
  ierr = MPI_Allreduce(&red,          &tot, 1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm);CHKERRQ(ierr);
  if (min != 0.0) ratio = max/min; else ratio = 0.0;
  ierr = PetscFPrintf(comm, fd, "MPI Reductions:       %5.3e   %10.5f\n", max, ratio);CHKERRQ(ierr);
  numReductions = red; /* wrong because uses count from process zero */
  ierr = PetscFPrintf(comm, fd, "\nFlop counting convention: 1 flop = 1 real number operation of type (multiply/divide/add/subtract)\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(comm, fd, "                            e.g., VecAXPY() for real vectors of length N --> 2N flops\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(comm, fd, "                            and VecAXPY() for complex vectors of length N --> 8N flops\n");CHKERRQ(ierr);

  /* Get total number of stages --
       Currently, a single processor can register more stages than another, but stages must all be registered in order.
       We can removed this requirement if necessary by having a global stage numbering and indirection on the stage ID.
       This seems best accomplished by assoicating a communicator with each stage.
  */
  ierr = MPI_Allreduce(&stageLog->numStages, &numStages, 1, MPI_INT, MPI_MAX, comm);CHKERRQ(ierr);
  ierr = PetscMalloc(numStages * sizeof(PetscTruth), &localStageUsed);CHKERRQ(ierr);
  ierr = PetscMalloc(numStages * sizeof(PetscTruth), &stageUsed);CHKERRQ(ierr);
  ierr = PetscMalloc(numStages * sizeof(PetscTruth), &localStageVisible);CHKERRQ(ierr);
  ierr = PetscMalloc(numStages * sizeof(PetscTruth), &stageVisible);CHKERRQ(ierr);
  if (numStages > 0) {
    stageInfo = stageLog->stageInfo;
    for(stage = 0; stage < numStages; stage++) {
      if (stage < stageLog->numStages) {
        localStageUsed[stage]    = stageInfo[stage].used;
        localStageVisible[stage] = stageInfo[stage].perfInfo.visible;
      } else {
        localStageUsed[stage]    = PETSC_FALSE;
        localStageVisible[stage] = PETSC_TRUE;
      }
    }
    ierr = MPI_Allreduce(localStageUsed,    stageUsed,    numStages, MPI_INT, MPI_LOR,  comm);CHKERRQ(ierr);
    ierr = MPI_Allreduce(localStageVisible, stageVisible, numStages, MPI_INT, MPI_LAND, comm);CHKERRQ(ierr);
    for(stage = 0; stage < numStages; stage++) {
      if (stageUsed[stage]) {
        ierr = PetscFPrintf(comm, fd, "\nSummary of Stages:   ----- Time ------  ----- Flops -----  --- Messages ---  -- Message Lengths --  -- Reductions --\n");CHKERRQ(ierr);
        ierr = PetscFPrintf(comm, fd, "                        Avg     %%Total     Avg     %%Total   counts   %%Total     Avg         %%Total   counts   %%Total \n");CHKERRQ(ierr);
        break;
      }
    }
    for(stage = 0; stage < numStages; stage++) {
      if (!stageUsed[stage]) continue;
      if (localStageUsed[stage]) {
        ierr = MPI_Allreduce(&stageInfo[stage].perfInfo.time,          &stageTime, 1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm);CHKERRQ(ierr);
        ierr = MPI_Allreduce(&stageInfo[stage].perfInfo.flops,         &flops,     1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm);CHKERRQ(ierr);
        ierr = MPI_Allreduce(&stageInfo[stage].perfInfo.numMessages,   &mess,      1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm);CHKERRQ(ierr);
        ierr = MPI_Allreduce(&stageInfo[stage].perfInfo.messageLength, &messLen,   1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm);CHKERRQ(ierr);
        ierr = MPI_Allreduce(&stageInfo[stage].perfInfo.numReductions, &red,       1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm);CHKERRQ(ierr);
        name = stageInfo[stage].name;
      } else {
        ierr = MPI_Allreduce(&zero,                           &stageTime, 1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm);CHKERRQ(ierr);
        ierr = MPI_Allreduce(&zero,                           &flops,     1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm);CHKERRQ(ierr);
        ierr = MPI_Allreduce(&zero,                           &mess,      1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm);CHKERRQ(ierr);
        ierr = MPI_Allreduce(&zero,                           &messLen,   1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm);CHKERRQ(ierr);
        ierr = MPI_Allreduce(&zero,                           &red,       1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm);CHKERRQ(ierr);
        name = "";
      }
      mess *= 0.5; messLen *= 0.5; red /= size;
      if (TotalTime     != 0.0) fracTime       = stageTime/TotalTime;    else fracTime       = 0.0;
      if (TotalFlops    != 0.0) fracFlops      = flops/TotalFlops;       else fracFlops      = 0.0;
      /* Talk to Barry if (stageTime     != 0.0) flops          = (size*flops)/stageTime; else flops          = 0.0; */
      if (numMessages   != 0.0) fracMessages   = mess/numMessages;       else fracMessages   = 0.0;
      if (numMessages   != 0.0) avgMessLen     = messLen/numMessages;    else avgMessLen     = 0.0;
      if (messageLength != 0.0) fracLength     = messLen/messageLength;  else fracLength     = 0.0;
      if (numReductions != 0.0) fracReductions = red/numReductions;      else fracReductions = 0.0;
      ierr = PetscFPrintf(comm, fd, "%2d: %15s: %6.4e %5.1f%%  %6.4e %5.1f%%  %5.3e %5.1f%%  %5.3e      %5.1f%%  %5.3e %5.1f%% \n",
                          stage, name, stageTime/size, 100.0*fracTime, flops, 100.0*fracFlops,
                          mess, 100.0*fracMessages, avgMessLen, 100.0*fracLength, red, 100.0*fracReductions);CHKERRQ(ierr);
    }
  }

  ierr = PetscFPrintf(comm, fd,
    "\n------------------------------------------------------------------------------------------------------------------------\n");
                                                                                                          CHKERRQ(ierr);  
  ierr = PetscFPrintf(comm, fd, "See the 'Profiling' chapter of the users' manual for details on interpreting output.\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(comm, fd, "Phase summary info:\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(comm, fd, "   Count: number of times phase was executed\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(comm, fd, "   Time and Flops: Max - maximum over all processors\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(comm, fd, "                   Ratio - ratio of maximum to minimum over all processors\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(comm, fd, "   Mess: number of messages sent\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(comm, fd, "   Avg. len: average message length\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(comm, fd, "   Reduct: number of global reductions\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(comm, fd, "   Global: entire computation\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(comm, fd, "   Stage: stages of a computation. Set stages with PetscLogStagePush() and PetscLogStagePop().\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(comm, fd, "      %%T - percent time in this phase         %%F - percent flops in this phase\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(comm, fd, "      %%M - percent messages in this phase     %%L - percent message lengths in this phase\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(comm, fd, "      %%R - percent reductions in this phase\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(comm, fd, "   Total Mflop/s: 10e-6 * (sum of flops over all processors)/(max time over all processors)\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(comm, fd,
    "------------------------------------------------------------------------------------------------------------------------\n");
                                                                                                          CHKERRQ(ierr);

#if defined(PETSC_USE_DEBUG)
  ierr = PetscFPrintf(comm, fd, "\n\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(comm, fd, "      ##########################################################\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(comm, fd, "      #                                                        #\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(comm, fd, "      #                          WARNING!!!                    #\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(comm, fd, "      #                                                        #\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(comm, fd, "      #   This code was compiled with a debugging option,      #\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(comm, fd, "      #   To get timing results run config/configure.py        #\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(comm, fd, "      #   using --with-debugging=no, the performance will      #\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(comm, fd, "      #   be generally two or three times faster.              #\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(comm, fd, "      #                                                        #\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(comm, fd, "      ##########################################################\n\n\n");CHKERRQ(ierr);
#endif
#if defined(PETSC_USE_COMPLEX) && !defined(PETSC_USE_FORTRAN_KERNELS)
  ierr = PetscFPrintf(comm, fd, "\n\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(comm, fd, "      ##########################################################\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(comm, fd, "      #                                                        #\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(comm, fd, "      #                          WARNING!!!                    #\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(comm, fd, "      #                                                        #\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(comm, fd, "      #   The code for various complex numbers numerical       #\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(comm, fd, "      #   kernels uses C++, which generally is not well        #\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(comm, fd, "      #   optimized.  For performance that is about 4-5 times  #\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(comm, fd, "      #   faster, specify --with-fortran-kernels=1             #\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(comm, fd, "      #   when running config/configure.py.                    #\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(comm, fd, "      #                                                        #\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(comm, fd, "      ##########################################################\n\n\n");CHKERRQ(ierr);
#endif

  /* Report events */
  ierr = PetscFPrintf(comm, fd,
    "Event                Count      Time (sec)     Flops                             --- Global ---  --- Stage ---   Total\n");
                                                                                                          CHKERRQ(ierr);
  ierr = PetscFPrintf(comm, fd,
    "                   Max Ratio  Max     Ratio   Max  Ratio  Mess   Avg len Reduct  %%T %%F %%M %%L %%R  %%T %%F %%M %%L %%R Mflop/s\n");
                                                                                                          CHKERRQ(ierr);
  ierr = PetscFPrintf(comm,fd,
    "------------------------------------------------------------------------------------------------------------------------\n");

                                                                                                          CHKERRQ(ierr); 
  /* Problem: The stage name will not show up unless the stage executed on proc 1 */
  for(stage = 0; stage < numStages; stage++) {
    if (!stageVisible[stage]) continue;
    if (localStageUsed[stage]) {
      ierr = PetscFPrintf(comm, fd, "\n--- Event Stage %d: %s\n\n", stage, stageInfo[stage].name);CHKERRQ(ierr);
      ierr = MPI_Allreduce(&stageInfo[stage].perfInfo.time,          &stageTime, 1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm);CHKERRQ(ierr);
      ierr = MPI_Allreduce(&stageInfo[stage].perfInfo.flops,         &flops,     1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm);CHKERRQ(ierr);
      ierr = MPI_Allreduce(&stageInfo[stage].perfInfo.numMessages,   &mess,      1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm);CHKERRQ(ierr);
      ierr = MPI_Allreduce(&stageInfo[stage].perfInfo.messageLength, &messLen,   1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm);CHKERRQ(ierr);
      ierr = MPI_Allreduce(&stageInfo[stage].perfInfo.numReductions, &red,       1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm);CHKERRQ(ierr);
    } else {
      ierr = PetscFPrintf(comm, fd, "\n--- Event Stage %d: Unknown\n\n", stage);CHKERRQ(ierr);
      ierr = MPI_Allreduce(&zero,                           &stageTime, 1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm);CHKERRQ(ierr);
      ierr = MPI_Allreduce(&zero,                           &flops,     1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm);CHKERRQ(ierr);
      ierr = MPI_Allreduce(&zero,                           &mess,      1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm);CHKERRQ(ierr);
      ierr = MPI_Allreduce(&zero,                           &messLen,   1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm);CHKERRQ(ierr);
      ierr = MPI_Allreduce(&zero,                           &red,       1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm);CHKERRQ(ierr);
    }
    mess *= 0.5; messLen *= 0.5; red /= size;

    /* Get total number of events in this stage --
       Currently, a single processor can register more events than another, but events must all be registered in order,
       just like stages. We can removed this requirement if necessary by having a global event numbering and indirection
       on the event ID. This seems best accomplished by assoicating a communicator with each stage.

       Problem: If the event did not happen on proc 1, its name will not be available.
       Problem: Event visibility is not implemented
    */
    if (localStageUsed[stage]) {
      eventInfo      = stageLog->stageInfo[stage].eventLog->eventInfo;
      localNumEvents = stageLog->stageInfo[stage].eventLog->numEvents;
    } else {
      localNumEvents = 0;
    }
    ierr = MPI_Allreduce(&localNumEvents, &numEvents, 1, MPI_INT, MPI_MAX, comm);CHKERRQ(ierr);
    for(event = 0; event < numEvents; event++) {
      if (localStageUsed[stage] && (event < stageLog->stageInfo[stage].eventLog->numEvents) && (eventInfo[event].depth == 0)) {
        if ((eventInfo[event].count > 0) && (eventInfo[event].time > 0.0)) {
          flopr = eventInfo[event].flops;
        } else {
          flopr = 0.0;
        }
        ierr = MPI_Allreduce(&flopr,                          &minf,  1, MPIU_PETSCLOGDOUBLE, MPI_MIN, comm);CHKERRQ(ierr);
        ierr = MPI_Allreduce(&flopr,                          &maxf,  1, MPIU_PETSCLOGDOUBLE, MPI_MAX, comm);CHKERRQ(ierr);
        ierr = MPI_Allreduce(&eventInfo[event].flops,         &totf,  1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm);CHKERRQ(ierr);
        ierr = MPI_Allreduce(&eventInfo[event].time,          &mint,  1, MPIU_PETSCLOGDOUBLE, MPI_MIN, comm);CHKERRQ(ierr);
        ierr = MPI_Allreduce(&eventInfo[event].time,          &maxt,  1, MPIU_PETSCLOGDOUBLE, MPI_MAX, comm);CHKERRQ(ierr);
        ierr = MPI_Allreduce(&eventInfo[event].time,          &tott,  1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm);CHKERRQ(ierr);
        ierr = MPI_Allreduce(&eventInfo[event].numMessages,   &totm,  1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm);CHKERRQ(ierr);
        ierr = MPI_Allreduce(&eventInfo[event].messageLength, &totml, 1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm);CHKERRQ(ierr);
        ierr = MPI_Allreduce(&eventInfo[event].numReductions, &totr,  1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm);CHKERRQ(ierr);
        ierr = MPI_Allreduce(&eventInfo[event].count,         &minCt, 1, MPI_INT,             MPI_MIN, comm);CHKERRQ(ierr);
        ierr = MPI_Allreduce(&eventInfo[event].count,         &maxCt, 1, MPI_INT,             MPI_MAX, comm);CHKERRQ(ierr);
        name = stageLog->eventLog->eventInfo[event].name;
      } else {
        flopr = 0.0;
        ierr = MPI_Allreduce(&flopr,                          &minf,  1, MPIU_PETSCLOGDOUBLE, MPI_MIN, comm);CHKERRQ(ierr);
        ierr = MPI_Allreduce(&flopr,                          &maxf,  1, MPIU_PETSCLOGDOUBLE, MPI_MAX, comm);CHKERRQ(ierr);
        ierr = MPI_Allreduce(&zero,                           &totf,  1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm);CHKERRQ(ierr);
        ierr = MPI_Allreduce(&zero,                           &mint,  1, MPIU_PETSCLOGDOUBLE, MPI_MIN, comm);CHKERRQ(ierr);
        ierr = MPI_Allreduce(&zero,                           &maxt,  1, MPIU_PETSCLOGDOUBLE, MPI_MAX, comm);CHKERRQ(ierr);
        ierr = MPI_Allreduce(&zero,                           &tott,  1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm);CHKERRQ(ierr);
        ierr = MPI_Allreduce(&zero,                           &totm,  1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm);CHKERRQ(ierr);
        ierr = MPI_Allreduce(&zero,                           &totml, 1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm);CHKERRQ(ierr);
        ierr = MPI_Allreduce(&zero,                           &totr,  1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm);CHKERRQ(ierr);
        ierr = MPI_Allreduce(&ierr,                           &minCt, 1, MPI_INT,             MPI_MIN, comm);CHKERRQ(ierr);
        ierr = MPI_Allreduce(&ierr,                           &maxCt, 1, MPI_INT,             MPI_MAX, comm);CHKERRQ(ierr);
        name = "";
      }
      if (mint < 0.0) {
        ierr = PetscFPrintf(comm, fd, "WARNING!!! Minimum time %g over all processors for %s is negative! This happens\n on some machines whose times cannot handle too rapid calls.!\n artificially changing minimum to zero.\n",mint,name);
        mint = 0;
      }
      if (minf < 0.0) SETERRQ2(PETSC_ERR_PLIB,"Minimum flops %g over all processors for %s is negative! Not possible!",minf,name);
      totm *= 0.5; totml *= 0.5; totr /= size;
     
      if (maxCt != 0) {
        if (minCt         != 0)   ratCt            = ((PetscLogDouble) maxCt)/minCt; else ratCt            = 0.0;
        if (mint          != 0.0) ratt             = maxt/mint;                  else ratt             = 0.0;
        if (minf          != 0.0) ratf             = maxf/minf;                  else ratf             = 0.0;
        if (TotalTime     != 0.0) fracTime         = tott/TotalTime;             else fracTime         = 0.0;
        if (TotalFlops    != 0.0) fracFlops        = totf/TotalFlops;            else fracFlops        = 0.0;
        if (stageTime     != 0.0) fracStageTime    = tott/stageTime;             else fracStageTime    = 0.0;
        if (flops         != 0.0) fracStageFlops   = totf/flops;                 else fracStageFlops   = 0.0;
        if (numMessages   != 0.0) fracMess         = totm/numMessages;           else fracMess         = 0.0;
        if (messageLength != 0.0) fracMessLen      = totml/messageLength;        else fracMessLen      = 0.0;
        if (numReductions != 0.0) fracRed          = totr/numReductions;         else fracRed          = 0.0;
        if (mess          != 0.0) fracStageMess    = totm/mess;                  else fracStageMess    = 0.0;
        if (messLen       != 0.0) fracStageMessLen = totml/messLen;              else fracStageMessLen = 0.0;
        if (red           != 0.0) fracStageRed     = totr/red;                   else fracStageRed     = 0.0;
        if (totm          != 0.0) totml           /= totm;                       else totml            = 0.0;
        if (maxt          != 0.0) flopr            = totf/maxt;                  else flopr            = 0.0;
        ierr = PetscFPrintf(comm, fd,
          "%-16s %7d%4.1f %5.4e%4.1f %3.2e%4.1f %2.1e %2.1e %2.1e%3.0f%3.0f%3.0f%3.0f%3.0f %3.0f%3.0f%3.0f%3.0f%3.0f %5.0f\n",
                            name, maxCt, ratCt, maxt, ratt, maxf, ratf, totm, totml, totr,
                            100.0*fracTime, 100.0*fracFlops, 100.0*fracMess, 100.0*fracMessLen, 100.0*fracRed,
                            100.0*fracStageTime, 100.0*fracStageFlops, 100.0*fracStageMess, 100.0*fracStageMessLen, 100.0*fracStageRed,
                            flopr/1.0e6);CHKERRQ(ierr);
      }
    }
  }

  /* Memory usage and object creation */
  ierr = PetscFPrintf(comm, fd,
    "------------------------------------------------------------------------------------------------------------------------\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(comm, fd, "\n");CHKERRQ(ierr); 
  ierr = PetscFPrintf(comm, fd, "Memory usage is given in bytes:\n\n");CHKERRQ(ierr);

  /* Right now, only stages on the first processor are reported here, meaning only objects associated with
     the global communicator, or MPI_COMM_SELF for proc 1. We really should report global stats and then
     stats for stages local to processor sets.
  */
  /* We should figure out the longest object name here (now 20 characters) */
  ierr = PetscFPrintf(comm, fd, "Object Type          Creations   Destructions     Memory  Descendants' Mem.\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(comm, fd, "Reports information only for process 0.\n");CHKERRQ(ierr);
  for(stage = 0; stage < numStages; stage++) {
    if (localStageUsed[stage]) {
      classInfo = stageLog->stageInfo[stage].classLog->classInfo;
      ierr = PetscFPrintf(comm, fd, "\n--- Event Stage %d: %s\n\n", stage, stageInfo[stage].name);CHKERRQ(ierr);
      for(oclass = 0; oclass < stageLog->stageInfo[stage].classLog->numClasses; oclass++) {
        if ((classInfo[oclass].creations > 0) || (classInfo[oclass].destructions > 0)) {
          ierr = PetscFPrintf(comm, fd, "%20s %5d          %5d  %11.0f     %g\n", stageLog->classLog->classInfo[oclass].name,
                              classInfo[oclass].creations, classInfo[oclass].destructions, classInfo[oclass].mem,
                              classInfo[oclass].descMem);CHKERRQ(ierr);
        }
      }
    } else {
      ierr = PetscFPrintf(comm, fd, "\n--- Event Stage %d: Unknown\n\n", stage);CHKERRQ(ierr);
    }
  }

  ierr = PetscFree(localStageUsed);CHKERRQ(ierr);
  ierr = PetscFree(stageUsed);CHKERRQ(ierr);
  ierr = PetscFree(localStageVisible);CHKERRQ(ierr);
  ierr = PetscFree(stageVisible);CHKERRQ(ierr);

  /* Information unrelated to this particular run */
  ierr = PetscFPrintf(comm, fd,
    "========================================================================================================================\n");CHKERRQ(ierr);
  PetscTime(y); 
  PetscTime(x);
  PetscTime(y); PetscTime(y); PetscTime(y); PetscTime(y); PetscTime(y);
  PetscTime(y); PetscTime(y); PetscTime(y); PetscTime(y); PetscTime(y);
  ierr = PetscFPrintf(comm,fd,"Average time to get PetscTime(): %g\n", (y-x)/10.0);CHKERRQ(ierr);
  /* MPI information */
  if (size > 1) {
    MPI_Status  status;
    PetscMPIInt tag;
    MPI_Comm    newcomm;

    ierr = MPI_Barrier(comm);CHKERRQ(ierr);
    PetscTime(x);
    ierr = MPI_Barrier(comm);CHKERRQ(ierr);
    ierr = MPI_Barrier(comm);CHKERRQ(ierr);
    ierr = MPI_Barrier(comm);CHKERRQ(ierr);
    ierr = MPI_Barrier(comm);CHKERRQ(ierr);
    ierr = MPI_Barrier(comm);CHKERRQ(ierr);
    PetscTime(y);
    ierr = PetscFPrintf(comm, fd, "Average time for MPI_Barrier(): %g\n", (y-x)/5.0);CHKERRQ(ierr);
    ierr = PetscCommDuplicate(comm,&newcomm, &tag);CHKERRQ(ierr);
    ierr = MPI_Barrier(comm);CHKERRQ(ierr);
    if (rank) {
      ierr = MPI_Recv(0, 0, MPI_INT, rank-1,            tag, newcomm, &status);CHKERRQ(ierr);
      ierr = MPI_Send(0, 0, MPI_INT, (rank+1)%size, tag, newcomm);CHKERRQ(ierr);
    } else {
      PetscTime(x);
      ierr = MPI_Send(0, 0, MPI_INT, 1,          tag, newcomm);CHKERRQ(ierr);
      ierr = MPI_Recv(0, 0, MPI_INT, size-1, tag, newcomm, &status);CHKERRQ(ierr);
      PetscTime(y);
      ierr = PetscFPrintf(comm,fd,"Average time for zero size MPI_Send(): %g\n", (y-x)/size);CHKERRQ(ierr);
    }
    ierr = PetscCommDestroy(&newcomm);CHKERRQ(ierr);
  }
  if (!rank) {
    ierr = PetscOptionsPrint(fd);CHKERRQ(ierr);
  }
  /* Machine and compile information */
#if defined(PETSC_USE_FORTRAN_KERNELS)
  ierr = PetscFPrintf(comm, fd, "Compiled with FORTRAN kernels\n");CHKERRQ(ierr);
#else
  ierr = PetscFPrintf(comm, fd, "Compiled without FORTRAN kernels\n");CHKERRQ(ierr);
#endif
#if defined(PETSC_USE_SCALAR_SINGLE)
  ierr = PetscFPrintf(comm, fd, "Compiled with single precision PetscScalar and PetscReal\n");CHKERRQ(ierr);
#elif defined(PETSC_USE_LONGDOUBLE)
  ierr = PetscFPrintf(comm, fd, "Compiled with long double precision PetscScalar and PetscReal\n");CHKERRQ(ierr);
#elif defined(PETSC_USE_SCALAR_INT)
  ierr = PetscFPrintf(comm, fd, "Compiled with int PetscScalar and PetscReal\n");CHKERRQ(ierr);
#endif

#if defined(PETSC_USE_SCALAR_MAT_SINGLE)
  ierr = PetscFPrintf(comm, fd, "Compiled with single precision matrices\n");CHKERRQ(ierr);
#else
  ierr = PetscFPrintf(comm, fd, "Compiled with full precision matrices (default)\n");CHKERRQ(ierr);
#endif
  ierr = PetscFPrintf(comm, fd, "sizeof(short) %d sizeof(int) %d sizeof(long) %d sizeof(void*) %d sizeof(PetscScalar) %d\n",
                      (int) sizeof(short), (int) sizeof(int), (int) sizeof(long), (int) sizeof(void*),(int) sizeof(PetscScalar));CHKERRQ(ierr);

  ierr = PetscFPrintf(comm, fd, "Configure run at: %s\n",petscconfigureruntime);CHKERRQ(ierr);
  ierr = PetscFPrintf(comm, fd, "Configure options: %s",petscconfigureoptions);CHKERRQ(ierr);
  ierr = PetscFPrintf(comm, fd, "%s", petscmachineinfo);CHKERRQ(ierr);
  ierr = PetscFPrintf(comm, fd, "%s", petsccompilerinfo);CHKERRQ(ierr);
  ierr = PetscFPrintf(comm, fd, "%s", petsccompilerflagsinfo);CHKERRQ(ierr);
  ierr = PetscFPrintf(comm, fd, "%s", petsclinkerinfo);CHKERRQ(ierr);

  /* Cleanup */
  ierr = PetscFPrintf(comm, fd, "\n");CHKERRQ(ierr);
  ierr = PetscFClose(comm, fd);CHKERRQ(ierr);
  ierr = StageLogPush(stageLog, lastStage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscLogPrintDetailed"
/*@C
  PetscLogPrintDetailed - Each process prints the times for its own events

  Collective over MPI_Comm

  Input Parameter:
+ comm - The MPI communicator (only one processor prints output)
- file - [Optional] The output file name

  Options Database Keys:
. -log_summary_detailed - Prints summary of log information (for code compiled with PETSC_USE_LOG)

  Usage:
.vb
     PetscInitialize(...);
     PetscLogBegin();
     ... code ...
     PetscLogPrintDetailed(MPI_Comm,filename);
     PetscFinalize(...);
.ve

  Notes:
  By default the summary is printed to stdout.

  Level: beginner
   
.keywords: log, dump, print
.seealso: PetscLogBegin(), PetscLogDump(), PetscLogPrintSummary()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscLogPrintDetailed(MPI_Comm comm, const char filename[]) 
{
  FILE          *fd = PETSC_STDOUT;
  StageLog       stageLog;
  StageInfo     *stageInfo = PETSC_NULL;
  EventPerfInfo *eventInfo = PETSC_NULL;
  const char    *name = PETSC_NULL;
  PetscLogDouble TotalTime;
  PetscLogDouble stageTime, flops, flopr, mess, messLen, red;
  PetscLogDouble maxf, totf, maxt, tott, totm, totml, totr = 0.0;
  PetscMPIInt    maxCt;
  PetscMPIInt    size, rank;
  PetscTruth     *stageUsed;
  PetscTruth     *stageVisible;
  int            numStages, numEvents;
  int            stage;
  PetscLogEvent  event;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(comm, &size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  /* Pop off any stages the user forgot to remove */
  ierr = PetscLogGetStageLog(&stageLog);CHKERRQ(ierr);
  ierr = StageLogGetCurrent(stageLog, &stage);CHKERRQ(ierr);
  while (stage >= 0) {
    ierr = StageLogPop(stageLog);CHKERRQ(ierr);
    ierr = StageLogGetCurrent(stageLog, &stage);CHKERRQ(ierr);
  }
  /* Get the total elapsed time */
  PetscTime(TotalTime);  TotalTime -= BaseTime;
  /* Open the summary file */
  if (filename) {
    ierr = PetscFOpen(comm, filename, "w", &fd);CHKERRQ(ierr);
  }

  ierr = PetscFPrintf(comm, fd, "************************************************************************************************************************\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(comm, fd, "***             WIDEN YOUR WINDOW TO 120 CHARACTERS.  Use 'enscript -r -fCourier9' to print this document            ***\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(comm, fd, "************************************************************************************************************************\n");CHKERRQ(ierr);


  numStages = stageLog->numStages;
  ierr = PetscMalloc(numStages * sizeof(PetscTruth), &stageUsed);CHKERRQ(ierr);
  ierr = PetscMalloc(numStages * sizeof(PetscTruth), &stageVisible);CHKERRQ(ierr);
  if (numStages > 0) {
    stageInfo = stageLog->stageInfo;
    for(stage = 0; stage < numStages; stage++) {
      if (stage < stageLog->numStages) {
        stageUsed[stage]    = stageInfo[stage].used;
        stageVisible[stage] = stageInfo[stage].perfInfo.visible;
      } else {
        stageUsed[stage]    = PETSC_FALSE;
        stageVisible[stage] = PETSC_TRUE;
      }
    }
  }

  /* Report events */
  ierr = PetscFPrintf(comm, fd,"Event                Count      Time (sec)     Flops/sec                          \n");CHKERRQ(ierr);
  ierr = PetscFPrintf(comm, fd,"                                                            Mess   Avg len Reduct \n");CHKERRQ(ierr);
  ierr = PetscFPrintf(comm,fd,"-----------------------------------------------------------------------------------\n");CHKERRQ(ierr); 
  /* Problem: The stage name will not show up unless the stage executed on proc 1 */
  for(stage = 0; stage < numStages; stage++) {
    if (!stageVisible[stage]) continue;
    if (stageUsed[stage]) {
      ierr = PetscSynchronizedFPrintf(comm, fd, "\n--- Event Stage %d: %s\n\n", stage, stageInfo[stage].name);CHKERRQ(ierr);
      ierr = MPI_Allreduce(&stageInfo[stage].perfInfo.time,          &stageTime, 1, MPIU_PETSCLOGDOUBLE, MPI_SUM, PETSC_COMM_SELF);CHKERRQ(ierr);
      ierr = MPI_Allreduce(&stageInfo[stage].perfInfo.flops,         &flops,     1, MPIU_PETSCLOGDOUBLE, MPI_SUM, PETSC_COMM_SELF);CHKERRQ(ierr);
      ierr = MPI_Allreduce(&stageInfo[stage].perfInfo.numMessages,   &mess,      1, MPIU_PETSCLOGDOUBLE, MPI_SUM, PETSC_COMM_SELF);CHKERRQ(ierr);
      ierr = MPI_Allreduce(&stageInfo[stage].perfInfo.messageLength, &messLen,   1, MPIU_PETSCLOGDOUBLE, MPI_SUM, PETSC_COMM_SELF);CHKERRQ(ierr);
      ierr = MPI_Allreduce(&stageInfo[stage].perfInfo.numReductions, &red,       1, MPIU_PETSCLOGDOUBLE, MPI_SUM, PETSC_COMM_SELF);CHKERRQ(ierr);
    } 
    mess *= 0.5; messLen *= 0.5;

    /* Get total number of events in this stage --
    */
    if (stageUsed[stage]) {
      eventInfo      = stageLog->stageInfo[stage].eventLog->eventInfo;
      numEvents = stageLog->stageInfo[stage].eventLog->numEvents;
    } else {
      numEvents = 0;
    }
    for(event = 0; event < numEvents; event++) {
      if (stageUsed[stage] && (event < stageLog->stageInfo[stage].eventLog->numEvents)) {
        if ((eventInfo[event].count > 0) && (eventInfo[event].time > 0.0)) {
          flopr = eventInfo[event].flops/eventInfo[event].time;
        } else {
          flopr = 0.0;
        }
        ierr = MPI_Allreduce(&flopr,                          &maxf,  1, MPIU_PETSCLOGDOUBLE, MPI_MAX, PETSC_COMM_SELF);CHKERRQ(ierr);
        ierr = MPI_Allreduce(&eventInfo[event].flops,         &totf,  1, MPIU_PETSCLOGDOUBLE, MPI_SUM, PETSC_COMM_SELF);CHKERRQ(ierr);
        ierr = MPI_Allreduce(&eventInfo[event].time,          &maxt,  1, MPIU_PETSCLOGDOUBLE, MPI_MAX, PETSC_COMM_SELF);CHKERRQ(ierr);
        ierr = MPI_Allreduce(&eventInfo[event].time,          &tott,  1, MPIU_PETSCLOGDOUBLE, MPI_SUM, PETSC_COMM_SELF);CHKERRQ(ierr);
        ierr = MPI_Allreduce(&eventInfo[event].numMessages,   &totm,  1, MPIU_PETSCLOGDOUBLE, MPI_SUM, PETSC_COMM_SELF);CHKERRQ(ierr);
        ierr = MPI_Allreduce(&eventInfo[event].messageLength, &totml, 1, MPIU_PETSCLOGDOUBLE, MPI_SUM, PETSC_COMM_SELF);CHKERRQ(ierr);
        totr = eventInfo[event].numReductions;
        ierr = MPI_Allreduce(&eventInfo[event].count,         &maxCt, 1, MPI_INT,             MPI_MAX, PETSC_COMM_SELF);CHKERRQ(ierr);
        name = stageLog->eventLog->eventInfo[event].name;
        totm *= 0.5; totml *= 0.5; 
      }
     
      if (maxCt != 0) {
        if (totm          != 0.0) totml           /= totm;                       else totml            = 0.0;
        ierr = PetscSynchronizedFPrintf(comm, fd,"%-16s %7d      %5.4e      %3.2e      %2.1e %2.1e %2.1e\n",name, maxCt,  maxt,  maxf, totm, totml, totr);CHKERRQ(ierr);
      }
    }
  }
  ierr = PetscSynchronizedFlush(comm);CHKERRQ(ierr);

  ierr = PetscFree(stageUsed);CHKERRQ(ierr);
  ierr = PetscFree(stageVisible);CHKERRQ(ierr);

  ierr = PetscFClose(comm, fd);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*----------------------------------------------- Counter Functions -------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "PetscGetFlops"
/*@C
   PetscGetFlops - Returns the number of flops used on this processor 
   since the program began. 

   Not Collective

   Output Parameter:
   flops - number of floating point operations 

   Notes:
   A global counter logs all PETSc flop counts.  The user can use
   PetscLogFlops() to increment this counter to include flops for the 
   application code.  

   PETSc automatically logs library events if the code has been
   compiled with -DPETSC_USE_LOG (which is the default), and -log,
   -log_summary, or -log_all are specified.  PetscLogFlops() is
   intended for logging user flops to supplement this PETSc
   information.

   Level: intermediate

.keywords: log, flops, floating point operations

.seealso: PetscGetTime(), PetscLogFlops()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscGetFlops(PetscLogDouble *flops)
{
  PetscFunctionBegin;
  *flops = _TotalFlops;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscLogObjectState"
PetscErrorCode PETSC_DLLEXPORT PetscLogObjectState(PetscObject obj, const char format[], ...)
{
  PetscErrorCode ierr;
  int            fullLength;
  va_list        Argp;

  PetscFunctionBegin;
  if (!logObjects) PetscFunctionReturn(0);
  va_start(Argp, format);
  ierr = PetscVSNPrintf(objects[obj->id].info, 64,format,&fullLength, Argp);CHKERRQ(ierr);
  va_end(Argp);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscLogGetStageLog"
/*@
  PetscLogGetStageLog - This function returns the default stage logging object.

  Not collective

  Output Parameter:
. stageLog - The default StageLog

  Level: beginner

.keywords: log, stage
.seealso: StageLogCreate()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscLogGetStageLog(StageLog *stageLog)
{
  PetscFunctionBegin;
  PetscValidPointer(stageLog,1);
  if (_stageLog == PETSC_NULL) {
    fprintf(stderr, "Logging has not been enabled.\nYou might have forgotten to call PetscInitialize().\n");
    MPI_Abort(MPI_COMM_WORLD, PETSC_ERR_SUP);
  }
  *stageLog = _stageLog;
  PetscFunctionReturn(0);
}

/*MC
   PetscLogFlops - Adds floating point operations to the global counter.

   Synopsis:
   PetscErrorCode PetscLogFlops(PetscLogDouble f)

   Not Collective

   Input Parameter:
.  f - flop counter


   Usage:
.vb
     PetscLogEvent USER_EVENT;
     PetscLogEventRegister("User event",0,&USER_EVENT);
     PetscLogEventBegin(USER_EVENT,0,0,0,0);
        [code segment to monitor]
        PetscLogFlops(user_flops)
     PetscLogEventEnd(USER_EVENT,0,0,0,0);
.ve

   Notes:
   A global counter logs all PETSc flop counts.  The user can use
   PetscLogFlops() to increment this counter to include flops for the 
   application code.  

   PETSc automatically logs library events if the code has been
   compiled with -DPETSC_USE_LOG (which is the default), and -log,
   -log_summary, or -log_all are specified.  PetscLogFlops() is
   intended for logging user flops to supplement this PETSc
   information.

   Level: intermediate

.seealso: PetscLogEventRegister(), PetscLogEventBegin(), PetscLogEventEnd(), PetscGetFlops()

.keywords: log, flops, floating point operations
M*/

/*MC
   PreLoadBegin - Begin a segment of code that may be preloaded (run twice)
    to get accurate timings

   Synopsis:
   void PreLoadBegin(PetscTruth flag,char *name);

   Not Collective

   Input Parameter:
+   flag - PETSC_TRUE to run twice, PETSC_FALSE to run once, may be overridden
           with command line option -preload true or -preload false
-   name - name of first stage (lines of code timed separately with -log_summary) to
           be preloaded

   Usage:
.vb
     PreLoadBegin(PETSC_TRUE,"first stage);
       lines of code
       PreLoadStage("second stage");
       lines of code
     PreLoadEnd();
.ve

   Notes: Only works in C/C++, not Fortran

     Flags available within the macro. 
+    PetscPreLoadingUsed - true if we are or have done preloading 
.    PetscPreLoadingOn - true if it is CURRENTLY doing preload
.    PreLoadIt - 0 for the first computation (with preloading turned off it is only 0) 1 for the second
-    PreLoadMax - number of times it will do the computation, only one when preloading is turned on
     The first two variables are available throughout the program, the second two only between the PreLoadBegin()
     and PreLoadEnd()

   Level: intermediate

.seealso: PetscLogEventRegister(), PetscLogEventBegin(), PetscLogEventEnd(), PreLoadEnd(), PreLoadStage()

   Concepts: preloading
   Concepts: timing^accurate
   Concepts: paging^eliminating effects of


M*/

/*MC
   PreLoadEnd - End a segment of code that may be preloaded (run twice)
    to get accurate timings

   Synopsis:
   void PreLoadEnd(void);

   Not Collective

   Usage:
.vb
     PreLoadBegin(PETSC_TRUE,"first stage);
       lines of code
       PreLoadStage("second stage");
       lines of code
     PreLoadEnd();
.ve

   Notes: only works in C/C++ not fortran

   Level: intermediate

.seealso: PetscLogEventRegister(), PetscLogEventBegin(), PetscLogEventEnd(), PreLoadBegin(), PreLoadStage()

M*/

/*MC
   PreLoadStage - Start a new segment of code to be timed separately.
    to get accurate timings

   Synopsis:
   void PreLoadStage(char *name);

   Not Collective

   Usage:
.vb
     PreLoadBegin(PETSC_TRUE,"first stage);
       lines of code
       PreLoadStage("second stage");
       lines of code
     PreLoadEnd();
.ve

   Notes: only works in C/C++ not fortran

   Level: intermediate

.seealso: PetscLogEventRegister(), PetscLogEventBegin(), PetscLogEventEnd(), PreLoadBegin(), PreLoadEnd()

M*/

/*----------------------------------------------- Stack Functions ---------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "StackDestroy"
/*@C
  StackDestroy - This function destroys a stack.

  Not Collective

  Input Parameter:
. stack - The stack

  Level: beginner

.keywords: log, stack, destroy
.seealso: StackCreate(), StackEmpty(), StackPush(), StackPop(), StackTop()
@*/
PetscErrorCode StackDestroy(IntStack stack)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(stack->stack);CHKERRQ(ierr);
  ierr = PetscFree(stack);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "StackEmpty"
/*@C
  StackEmpty - This function determines whether any items have been pushed.

  Not Collective

  Input Parameter:
. stack - The stack

  Output Parameter:
. empty - PETSC_TRUE if the stack is empty

  Level: intermediate

.keywords: log, stack, empty
.seealso: StackCreate(), StackDestroy(), StackPush(), StackPop(), StackTop()
@*/
PetscErrorCode StackEmpty(IntStack stack, PetscTruth *empty)
{
  PetscFunctionBegin;
  PetscValidIntPointer(empty,2);
  if (stack->top == -1) {
    *empty = PETSC_TRUE;
  } else {
    *empty = PETSC_FALSE;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "StackTop"
/*@C
  StackTop - This function returns the top of the stack.

  Not Collective

  Input Parameter:
. stack - The stack

  Output Parameter:
. top - The integer on top of the stack

  Level: intermediate

.keywords: log, stack, top
.seealso: StackCreate(), StackDestroy(), StackEmpty(), StackPush(), StackPop()
@*/
PetscErrorCode StackTop(IntStack stack, int *top)
{
  PetscFunctionBegin;
  PetscValidIntPointer(top,2);
  *top = stack->stack[stack->top];
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "StackPush"
/*@C
  StackPush - This function pushes an integer on the stack.

  Not Collective

  Input Parameters:
+ stack - The stack
- item  - The integer to push

  Level: intermediate

.keywords: log, stack, push
.seealso: StackCreate(), StackDestroy(), StackEmpty(), StackPop(), StackTop()
@*/
PetscErrorCode StackPush(IntStack stack, int item)
{
  int            *array;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  stack->top++;
  if (stack->top >= stack->max) {
    ierr = PetscMalloc(stack->max*2 * sizeof(int), &array);CHKERRQ(ierr);
    ierr = PetscMemcpy(array, stack->stack, stack->max * sizeof(int));CHKERRQ(ierr);
    ierr = PetscFree(stack->stack);CHKERRQ(ierr);
    stack->stack = array;
    stack->max  *= 2;
  }
  stack->stack[stack->top] = item;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "StackPop"
/*@C
  StackPop - This function pops an integer from the stack.

  Not Collective

  Input Parameter:
. stack - The stack

  Output Parameter:
. item  - The integer popped

  Level: intermediate

.keywords: log, stack, pop
.seealso: StackCreate(), StackDestroy(), StackEmpty(), StackPush(), StackTop()
@*/
PetscErrorCode StackPop(IntStack stack, int *item)
{
  PetscFunctionBegin;
  PetscValidPointer(item,2);
  if (stack->top == -1) SETERRQ(PETSC_ERR_ARG_WRONGSTATE, "Stack is empty");
  *item = stack->stack[stack->top--];
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "StackCreate"
/*@C
  StackCreate - This function creates a stack.

  Not Collective

  Output Parameter:
. stack - The stack

  Level: beginner

.keywords: log, stack, pop
.seealso: StackDestroy(), StackEmpty(), StackPush(), StackPop(), StackTop()
@*/
PetscErrorCode StackCreate(IntStack *stack)
{
  IntStack       s;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(stack,1);
  ierr = PetscNew(struct _n_IntStack, &s);CHKERRQ(ierr);
  s->top = -1;
  s->max = 128;
  ierr = PetscMalloc(s->max * sizeof(int), &s->stack);CHKERRQ(ierr);
  ierr = PetscMemzero(s->stack, s->max * sizeof(int));CHKERRQ(ierr);
  *stack = s;
  PetscFunctionReturn(0);
}

#else /* end of -DPETSC_USE_LOG section */

#undef __FUNCT__  
#define __FUNCT__ "PetscLogObjectState"
PetscErrorCode PETSC_DLLEXPORT PetscLogObjectState(PetscObject obj, const char format[], ...)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#endif /* PETSC_USE_LOG*/


PetscCookie PETSC_LARGEST_COOKIE = PETSC_SMALLEST_COOKIE;
PetscCookie PETSC_OBJECT_COOKIE  = 0;

#undef __FUNCT__  
#define __FUNCT__ "PetscCookieRegister"
/*@C
  PetscCookieRegister - Registers a new class name for objects and logging operations in an application code. 

  Not Collective

  Input Parameter:
. name   - The class name
            
  Output Parameter:
. oclass - The class id or cookie

  Level: developer

.keywords: log, class, register

@*/
PetscErrorCode PETSC_DLLEXPORT PetscCookieRegister(const char name[],PetscCookie *oclass )
{
#if defined(PETSC_USE_LOG)
  StageLog       stageLog;
  PetscInt       stage;
  PetscErrorCode ierr;
#endif

  PetscFunctionBegin;
  *oclass = ++PETSC_LARGEST_COOKIE;
#if defined(PETSC_USE_LOG)
  ierr = PetscLogGetStageLog(&stageLog);CHKERRQ(ierr);
  ierr = ClassRegLogRegister(stageLog->classLog, name, *oclass);CHKERRQ(ierr);
  for(stage = 0; stage < stageLog->numStages; stage++) {
    ierr = ClassPerfLogEnsureSize(stageLog->stageInfo[stage].classLog, stageLog->classLog->numClasses);CHKERRQ(ierr);
  }
#endif
  PetscFunctionReturn(0);
}
