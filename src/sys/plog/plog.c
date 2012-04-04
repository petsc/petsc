
/*
      PETSc code to log object creation and destruction and PETSc events.

      This provides the public API used by the rest of PETSc and by users.

      These routines use a private API that is not used elsewhere in PETSc and is not 
      accessible to users. The private API is defined in logimpl.h and the utils directory.

*/
#include <petscsys.h>        /*I    "petscsys.h"   I*/
#include <petsctime.h>
#if defined(PETSC_HAVE_MPE)
#include <mpe.h>
#endif
#include <stdarg.h>
#include <sys/types.h>
#if defined(PETSC_HAVE_STDLIB_H)
#include <stdlib.h>
#endif
#if defined(PETSC_HAVE_MALLOC_H)
#include <malloc.h>
#endif
#include <petsc-private/logimpl.h>

PetscLogEvent  PETSC_LARGEST_EVENT  = PETSC_EVENT;

#if defined(PETSC_CLANGUAGE_CXX) && !defined(PETSC_USE_EXTERN_CXX)
std::map<std::string,PETSc::LogEvent> PETSc::Log::event_registry;
std::map<std::string,PETSc::LogStage> PETSc::Log::stage_registry;
#endif

#if defined(PETSC_USE_LOG)
#include <petscmachineinfo.h>
#include <petscconfiginfo.h>

/* used in the MPI_XXX() count macros in petsclog.h */

/* Action and object logging variables */
Action    *petsc_actions    = PETSC_NULL;
Object    *petsc_objects    = PETSC_NULL;
PetscBool  petsc_logActions = PETSC_FALSE;
PetscBool  petsc_logObjects = PETSC_FALSE;
int        petsc_numActions = 0, petsc_maxActions = 100;
int        petsc_numObjects = 0, petsc_maxObjects = 100;
int        petsc_numObjectsDestroyed = 0;

/* Global counters */
PetscLogDouble  petsc_BaseTime        = 0.0;
PetscLogDouble  petsc_TotalFlops     = 0.0; /* The number of flops */
PetscLogDouble  petsc_tmp_flops = 0.0; /* The incremental number of flops */
PetscLogDouble  petsc_send_ct         = 0.0; /* The number of sends */
PetscLogDouble  petsc_recv_ct         = 0.0; /* The number of receives */
PetscLogDouble  petsc_send_len        = 0.0; /* The total length of all sent messages */
PetscLogDouble  petsc_recv_len        = 0.0; /* The total length of all received messages */
PetscLogDouble  petsc_isend_ct        = 0.0; /* The number of immediate sends */
PetscLogDouble  petsc_irecv_ct        = 0.0; /* The number of immediate receives */
PetscLogDouble  petsc_isend_len       = 0.0; /* The total length of all immediate send messages */
PetscLogDouble  petsc_irecv_len       = 0.0; /* The total length of all immediate receive messages */
PetscLogDouble  petsc_wait_ct         = 0.0; /* The number of waits */
PetscLogDouble  petsc_wait_any_ct     = 0.0; /* The number of anywaits */
PetscLogDouble  petsc_wait_all_ct     = 0.0; /* The number of waitalls */
PetscLogDouble  petsc_sum_of_waits_ct = 0.0; /* The total number of waits */
PetscLogDouble  petsc_allreduce_ct    = 0.0; /* The number of reductions */
PetscLogDouble  petsc_gather_ct       = 0.0; /* The number of gathers and gathervs */
PetscLogDouble  petsc_scatter_ct      = 0.0; /* The number of scatters and scattervs */

/* Logging functions */
PetscErrorCode  (*PetscLogPHC)(PetscObject) = PETSC_NULL;
PetscErrorCode  (*PetscLogPHD)(PetscObject) = PETSC_NULL;
PetscErrorCode  (*PetscLogPLB)(PetscLogEvent, int, PetscObject, PetscObject, PetscObject, PetscObject) = PETSC_NULL;
PetscErrorCode  (*PetscLogPLE)(PetscLogEvent, int, PetscObject, PetscObject, PetscObject, PetscObject) = PETSC_NULL;

/* Tracing event logging variables */
FILE          *petsc_tracefile       = PETSC_NULL;
int            petsc_tracelevel      = 0;
const char    *petsc_traceblanks     = "                                                                                                    ";
char           petsc_tracespace[128] = " ";
PetscLogDouble petsc_tracetime       = 0.0;
static PetscBool  PetscLogBegin_PrivateCalled = PETSC_FALSE;

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
.seealso: PetscLogDump(), PetscLogAllBegin(), PetscLogView(), PetscLogStagePush(), PlogStagePop()
@*/
PetscErrorCode  PetscLogDestroy(void)
{
  PetscStageLog       stageLog;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(petsc_actions);CHKERRQ(ierr);
  ierr = PetscFree(petsc_objects);CHKERRQ(ierr);
  ierr = PetscLogSet(PETSC_NULL, PETSC_NULL);CHKERRQ(ierr);

  /* Resetting phase */
  ierr = PetscLogGetStageLog(&stageLog);CHKERRQ(ierr);
  ierr = PetscStageLogDestroy(stageLog);CHKERRQ(ierr);
  petsc_TotalFlops         = 0.0;
  petsc_numActions          = 0;
  petsc_numObjects          = 0;
  petsc_numObjectsDestroyed = 0;
  petsc_maxActions          = 100;
  petsc_maxObjects          = 100;
  petsc_actions    = PETSC_NULL;
  petsc_objects    = PETSC_NULL;
  petsc_logActions = PETSC_FALSE;
  petsc_logObjects = PETSC_FALSE;
  petsc_BaseTime        = 0.0;
  petsc_TotalFlops     = 0.0; 
  petsc_tmp_flops = 0.0; 
  petsc_send_ct         = 0.0; 
  petsc_recv_ct         = 0.0; 
  petsc_send_len        = 0.0; 
  petsc_recv_len        = 0.0; 
  petsc_isend_ct        = 0.0; 
  petsc_irecv_ct        = 0.0; 
  petsc_isend_len       = 0.0; 
  petsc_irecv_len       = 0.0; 
  petsc_wait_ct         = 0.0; 
  petsc_wait_any_ct     = 0.0; 
  petsc_wait_all_ct     = 0.0; 
  petsc_sum_of_waits_ct = 0.0; 
  petsc_allreduce_ct    = 0.0; 
  petsc_gather_ct       = 0.0; 
  petsc_scatter_ct      = 0.0; 
  PETSC_LARGEST_EVENT  = PETSC_EVENT;
  PetscLogPHC = PETSC_NULL;
  PetscLogPHD = PETSC_NULL;
  petsc_tracefile       = PETSC_NULL;
  petsc_tracelevel      = 0;
  petsc_traceblanks     = "                                                                                                    ";
  petsc_tracespace[0] = ' '; petsc_tracespace[1] = 0;
  petsc_tracetime       = 0.0;
  PETSC_LARGEST_CLASSID = PETSC_SMALLEST_CLASSID;
  PETSC_OBJECT_CLASSID  = 0;
  petsc_stageLog = 0;
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
PetscErrorCode  PetscLogSet(PetscErrorCode (*b)(PetscLogEvent, int, PetscObject, PetscObject, PetscObject, PetscObject),
            PetscErrorCode (*e)(PetscLogEvent, int, PetscObject, PetscObject, PetscObject, PetscObject))
{
  PetscFunctionBegin;
  PetscLogPLB = b;
  PetscLogPLE = e;
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_CHUD)
#include <CHUD/CHUD.h>
#endif
#if defined(PETSC_HAVE_PAPI)
#include <papi.h>
int PAPIEventSet = PAPI_NULL;
#endif

/*------------------------------------------- Initialization Functions ----------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "PetscLogBegin_Private"
PetscErrorCode  PetscLogBegin_Private(void)
{
  int               stage;
  PetscBool         opt;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (PetscLogBegin_PrivateCalled) PetscFunctionReturn(0);
  PetscLogBegin_PrivateCalled = PETSC_TRUE;

  ierr = PetscOptionsHasName(PETSC_NULL, "-log_exclude_actions", &opt);CHKERRQ(ierr);
  if (opt) {
    petsc_logActions = PETSC_FALSE;
  }
  ierr = PetscOptionsHasName(PETSC_NULL, "-log_exclude_objects", &opt);CHKERRQ(ierr);
  if (opt) {
    petsc_logObjects = PETSC_FALSE;
  }
  if (petsc_logActions) {
    ierr = PetscMalloc(petsc_maxActions * sizeof(Action), &petsc_actions);CHKERRQ(ierr);
  }
  if (petsc_logObjects) {
    ierr = PetscMalloc(petsc_maxObjects * sizeof(Object), &petsc_objects);CHKERRQ(ierr);
  }
  PetscLogPHC = PetscLogObjCreateDefault;
  PetscLogPHD = PetscLogObjDestroyDefault;
  /* Setup default logging structures */
  ierr = PetscStageLogCreate(&petsc_stageLog);CHKERRQ(ierr);
  ierr = PetscStageLogRegister(petsc_stageLog, "Main Stage", &stage);CHKERRQ(ierr);
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
  if (!chudIsEventValid(chudCPU1Dev,PMC_1,193)) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Event is not valid %d",193);
  ierr = chudStartPMCs();CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_PAPI)
  ierr = PAPI_library_init(PAPI_VER_CURRENT);
  if (ierr != PAPI_VER_CURRENT) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Cannot initialize PAPI");
  ierr = PAPI_query_event(PAPI_FP_INS);CHKERRQ(ierr);
  ierr = PAPI_create_eventset(&PAPIEventSet);CHKERRQ(ierr);
  ierr = PAPI_add_event(PAPIEventSet,PAPI_FP_INS);CHKERRQ(ierr);
  ierr = PAPI_start(PAPIEventSet);CHKERRQ(ierr);
#endif

  /* All processors sync here for more consistent logging */
  ierr = MPI_Barrier(PETSC_COMM_WORLD);CHKERRQ(ierr);
  PetscTime(petsc_BaseTime);
  ierr = PetscLogStagePush(stage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscLogBegin"
/*@C
  PetscLogBegin - Turns on logging of objects and events. This logs flop
  rates and object creation and should not slow programs down too much.
  This routine may be called more than once.

  Logically Collective over PETSC_COMM_WORLD

  Options Database Keys:
+ -log_summary - Prints summary of flop and timing information to the 
                  screen (for code compiled with PETSC_USE_LOG)
- -log - Prints detailed log information (for code compiled with PETSC_USE_LOG)

  Usage:
.vb
      PetscInitialize(...);
      PetscLogBegin();
       ... code ...
      PetscLogView(viewer); or PetscLogDump(); 
      PetscFinalize();
.ve

  Notes:
  PetscLogView(viewer) or PetscLogDump() actually cause the printing of 
  the logging information.

  Level: advanced

.keywords: log, begin
.seealso: PetscLogDump(), PetscLogAllBegin(), PetscLogView(), PetscLogTraceBegin()
@*/
PetscErrorCode  PetscLogBegin(void)
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

  Logically Collective on PETSC_COMM_WORLD

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
PetscErrorCode  PetscLogAllBegin(void)
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

  Logically Collective on PETSC_COMM_WORLD

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

.seealso: PetscLogDump(), PetscLogAllBegin(), PetscLogView(), PetscLogBegin()
@*/
PetscErrorCode  PetscLogTraceBegin(FILE *file)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  petsc_tracefile = file;
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
PetscErrorCode  PetscLogActions(PetscBool  flag)
{
  PetscFunctionBegin;
  petsc_logActions = flag;
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
PetscErrorCode  PetscLogObjects(PetscBool  flag)
{
  PetscFunctionBegin;
  petsc_logObjects = flag;
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
PetscErrorCode  PetscLogStageRegister(const char sname[],PetscLogStage *stage)
{
  PetscStageLog  stageLog;
  PetscLogEvent  event;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLogGetStageLog(&stageLog);CHKERRQ(ierr);
  ierr = PetscStageLogRegister(stageLog, sname, stage);CHKERRQ(ierr);
  /* Copy events already changed in the main stage, this sucks */
  ierr = EventPerfLogEnsureSize(stageLog->stageInfo[*stage].eventLog, stageLog->eventLog->numEvents);CHKERRQ(ierr);
  for (event = 0; event < stageLog->eventLog->numEvents; event++) {
    ierr = EventPerfInfoCopy(&stageLog->stageInfo[0].eventLog->eventInfo[event],&stageLog->stageInfo[*stage].eventLog->eventInfo[event]);CHKERRQ(ierr);
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
PetscErrorCode  PetscLogStagePush(PetscLogStage stage)
{
  PetscStageLog  stageLog;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLogGetStageLog(&stageLog);CHKERRQ(ierr);
  ierr = PetscStageLogPush(stageLog, stage);CHKERRQ(ierr);
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
PetscErrorCode  PetscLogStagePop(void)
{
  PetscStageLog  stageLog;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLogGetStageLog(&stageLog);CHKERRQ(ierr);
  ierr = PetscStageLogPop(stageLog);CHKERRQ(ierr);
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

.seealso: PetscLogStagePush(), PetscLogStagePop(), PetscLogEventBegin(), PetscLogEventEnd(), PetscPreLoadBegin(), PetscPreLoadEnd(), PetscPreLoadStage()
@*/
PetscErrorCode  PetscLogStageSetActive(PetscLogStage stage, PetscBool  isActive)
{
  PetscStageLog  stageLog;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLogGetStageLog(&stageLog);CHKERRQ(ierr);
  ierr = PetscStageLogSetActive(stageLog, stage, isActive);CHKERRQ(ierr);
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

.seealso: PetscLogStagePush(), PetscLogStagePop(), PetscLogEventBegin(), PetscLogEventEnd(), PetscPreLoadBegin(), PetscPreLoadEnd(), PetscPreLoadStage()
@*/
PetscErrorCode  PetscLogStageGetActive(PetscLogStage stage, PetscBool  *isActive)
{
  PetscStageLog  stageLog;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLogGetStageLog(&stageLog);CHKERRQ(ierr);
  ierr = PetscStageLogGetActive(stageLog, stage, isActive);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscLogStageSetVisible"
/*@
  PetscLogStageSetVisible - Determines stage visibility in PetscLogView()

  Not Collective 

  Input Parameters:
+ stage     - The stage
- isVisible - The visibility flag, PETSC_TRUE to print, else PETSC_FALSE (defaults to PETSC_TRUE)

  Level: intermediate

.seealso: PetscLogStagePush(), PetscLogStagePop(), PetscLogView()
@*/
PetscErrorCode  PetscLogStageSetVisible(PetscLogStage stage, PetscBool  isVisible)
{
  PetscStageLog  stageLog;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLogGetStageLog(&stageLog);CHKERRQ(ierr);
  ierr = PetscStageLogSetVisible(stageLog, stage, isVisible);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscLogStageGetVisible"
/*@
  PetscLogStageGetVisible - Returns stage visibility in PetscLogView()

  Not Collective 

  Input Parameter:
. stage     - The stage

  Output Parameter:
. isVisible - The visibility flag, PETSC_TRUE to print, else PETSC_FALSE (defaults to PETSC_TRUE)

  Level: intermediate

.seealso: PetscLogStagePush(), PetscLogStagePop(), PetscLogView()
@*/
PetscErrorCode  PetscLogStageGetVisible(PetscLogStage stage, PetscBool  *isVisible)
{
  PetscStageLog  stageLog;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLogGetStageLog(&stageLog);CHKERRQ(ierr);
  ierr = PetscStageLogGetVisible(stageLog, stage, isVisible);CHKERRQ(ierr);
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

.seealso: PetscLogStagePush(), PetscLogStagePop(), PetscPreLoadBegin(), PetscPreLoadEnd(), PetscPreLoadStage()
@*/
PetscErrorCode  PetscLogStageGetId(const char name[], PetscLogStage *stage)
{
  PetscStageLog  stageLog;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLogGetStageLog(&stageLog);CHKERRQ(ierr);
  ierr = PetscStageLogGetStage(stageLog, name, stage);CHKERRQ(ierr);
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
- classid - The classid associated to the class for this event, obtain either with
           PetscClassIdRegister() or use a predefined one such as KSP_CLASSID, SNES_CLASSID, the predefined ones 
           are only available in C code
            
  Output Parameter:
. event - The event id for use with PetscLogEventBegin() and PetscLogEventEnd().

  Example of Usage:
.vb
      PetscLogEvent USER_EVENT;
      PetscClassId classid;
      PetscLogDouble user_event_flops;
      PetscClassIdRegister("class name",&classid);
      PetscLogEventRegister("User event name",classid,&USER_EVENT);
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

  The classid is associated with each event so that classes of events
  can be disabled simultaneously, such as all matrix events. The user
  can either use an existing classid, such as MAT_CLASSID, or create
  their own as shown in the example.

  Level: intermediate

.keywords: log, event, register
.seealso: PetscLogEventBegin(), PetscLogEventEnd(), PetscLogFlops(),
          PetscLogEventMPEActivate(), PetscLogEventMPEDeactivate(),
          PetscLogEventActivate(), PetscLogEventDeactivate(), PetscClassIdRegister()
@*/
PetscErrorCode  PetscLogEventRegister(const char name[],PetscClassId classid,PetscLogEvent *event)
{
  PetscStageLog  stageLog;
  int            stage;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  *event = PETSC_DECIDE;
  ierr = PetscLogGetStageLog(&stageLog);CHKERRQ(ierr);
  ierr = EventRegLogRegister(stageLog->eventLog, name, classid, event);CHKERRQ(ierr);
  for (stage = 0; stage < stageLog->numStages; stage++) {
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
PetscErrorCode  PetscLogEventActivate(PetscLogEvent event)
{
  PetscStageLog  stageLog;
  int            stage;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLogGetStageLog(&stageLog);CHKERRQ(ierr);
  ierr = PetscStageLogGetCurrent(stageLog, &stage);CHKERRQ(ierr);
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
PetscErrorCode  PetscLogEventDeactivate(PetscLogEvent event)
{
  PetscStageLog  stageLog;
  int            stage;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLogGetStageLog(&stageLog);CHKERRQ(ierr);
  ierr = PetscStageLogGetCurrent(stageLog, &stage);CHKERRQ(ierr);
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
PetscErrorCode  PetscLogEventSetActiveAll(PetscLogEvent event, PetscBool  isActive)
{
  PetscStageLog  stageLog;
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
. classid - The event class, for example MAT_CLASSID, SNES_CLASSID, etc.

  Level: developer

.keywords: log, event, activate, class
.seealso: PetscInfoActivate(),PetscInfo(),PetscInfoAllow(),PetscLogEventDeactivateClass(), PetscLogEventActivate(),PetscLogEventDeactivate()
@*/
PetscErrorCode  PetscLogEventActivateClass(PetscClassId classid)
{
  PetscStageLog  stageLog;
  int            stage;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLogGetStageLog(&stageLog);CHKERRQ(ierr);
  ierr = PetscStageLogGetCurrent(stageLog, &stage);CHKERRQ(ierr);
  ierr = EventPerfLogActivateClass(stageLog->stageInfo[stage].eventLog, stageLog->eventLog, classid);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscLogEventDeactivateClass"
/*@
  PetscLogEventDeactivateClass - Deactivates event logging for a PETSc object class.

  Not Collective

  Input Parameter:
. classid - The event class, for example MAT_CLASSID, SNES_CLASSID, etc.

  Level: developer

.keywords: log, event, deactivate, class
.seealso: PetscInfoActivate(),PetscInfo(),PetscInfoAllow(),PetscLogEventActivateClass(), PetscLogEventActivate(),PetscLogEventDeactivate()
@*/
PetscErrorCode  PetscLogEventDeactivateClass(PetscClassId classid)
{
  PetscStageLog  stageLog;
  int            stage;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLogGetStageLog(&stageLog);CHKERRQ(ierr);
  ierr = PetscStageLogGetCurrent(stageLog, &stage);CHKERRQ(ierr);
  ierr = EventPerfLogDeactivateClass(stageLog->stageInfo[stage].eventLog, stageLog->eventLog, classid);CHKERRQ(ierr);
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

   Logically Collective on MPI_Comm

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
PetscErrorCode  PetscLogEventGetId(const char name[], PetscLogEvent *event)
{
  PetscStageLog  stageLog;
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
.seealso: PetscLogBegin(), PetscLogAllBegin(), PetscLogView()
@*/
PetscErrorCode  PetscLogDump(const char sname[])
{
  PetscStageLog      stageLog;
  PetscEventPerfInfo *eventInfo;
  FILE               *fd;
  char               file[PETSC_MAX_PATH_LEN], fname[PETSC_MAX_PATH_LEN];
  PetscLogDouble     flops, _TotalTime;
  PetscMPIInt        rank;
  int                action, object, curStage;
  PetscLogEvent      event;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  /* Calculate the total elapsed time */
  PetscTime(_TotalTime);
  _TotalTime -= petsc_BaseTime;
  /* Open log file */
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank);CHKERRQ(ierr);
  if (sname) {
    sprintf(file, "%s.%d", sname, rank);
  } else {
    sprintf(file, "Log.%d", rank);
  }
  ierr = PetscFixFilename(file, fname);CHKERRQ(ierr);
  ierr = PetscFOpen(PETSC_COMM_WORLD, fname, "w", &fd);CHKERRQ(ierr);
  if ((!rank) && (!fd)) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN, "Cannot open file: %s", fname);
  /* Output totals */
  ierr = PetscFPrintf(PETSC_COMM_WORLD, fd, "Total Flops %14e %16.8e\n", petsc_TotalFlops, _TotalTime);
  ierr = PetscFPrintf(PETSC_COMM_WORLD, fd, "Clock Resolution %g\n", 0.0);
  /* Output actions */
  if (petsc_logActions) {
    ierr = PetscFPrintf(PETSC_COMM_WORLD, fd, "Actions accomplished %d\n", petsc_numActions);
    for(action = 0; action < petsc_numActions; action++) {
      ierr = PetscFPrintf(PETSC_COMM_WORLD, fd, "%g %d %d %d %d %d %d %g %g %g\n",
                          petsc_actions[action].time, petsc_actions[action].action, (int)petsc_actions[action].event, (int)petsc_actions[action].classid, petsc_actions[action].id1,
                          petsc_actions[action].id2, petsc_actions[action].id3, petsc_actions[action].flops, petsc_actions[action].mem, petsc_actions[action].maxmem);
    }
  }
  /* Output objects */
  if (petsc_logObjects) {
    ierr = PetscFPrintf(PETSC_COMM_WORLD, fd, "Objects created %d destroyed %d\n", petsc_numObjects, petsc_numObjectsDestroyed);
    for(object = 0; object < petsc_numObjects; object++) {
      ierr = PetscFPrintf(PETSC_COMM_WORLD, fd, "Parent ID: %d Memory: %d\n", petsc_objects[object].parent, (int) petsc_objects[object].mem);
      if (!petsc_objects[object].name[0]) {
        ierr = PetscFPrintf(PETSC_COMM_WORLD, fd,"No Name\n");
      } else {
        ierr = PetscFPrintf(PETSC_COMM_WORLD, fd, "Name: %s\n", petsc_objects[object].name);
      }
      if (petsc_objects[object].info[0] != 0) {
        ierr = PetscFPrintf(PETSC_COMM_WORLD, fd, "No Info\n");
      } else {
        ierr = PetscFPrintf(PETSC_COMM_WORLD, fd, "Info: %s\n", petsc_objects[object].info);
      }
    }
  }
  /* Output events */
  ierr = PetscFPrintf(PETSC_COMM_WORLD, fd, "Event log:\n");
  ierr = PetscLogGetStageLog(&stageLog);CHKERRQ(ierr);
  ierr = PetscIntStackTop(stageLog->stack, &curStage);CHKERRQ(ierr);
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
#define __FUNCT__ "PetscLogView"
/*@C
  PetscLogViewer - Prints a summary of the logging.

  Collective over MPI_Comm

  Input Parameter:
.  viewer - an ASCII viewer

  Options Database Keys:
. -log_summary - Prints summary of log information (for code compiled with PETSC_USE_LOG)

  Usage:
.vb
     PetscInitialize(...);
     PetscLogBegin();
     ... code ...
     PetscLogView(PetscViewer);
     PetscFinalize(...);
.ve

  Notes:
  By default the summary is printed to stdout.

  Level: beginner
   
.keywords: log, dump, print
.seealso: PetscLogBegin(), PetscLogDump()
@*/
PetscErrorCode  PetscLogView(PetscViewer viewer)
{
  FILE               *fd;
  PetscLogDouble     zero       = 0.0;
  PetscStageLog      stageLog;
  PetscStageInfo     *stageInfo = PETSC_NULL;
  PetscEventPerfInfo *eventInfo = PETSC_NULL;
  PetscClassPerfInfo *classInfo;
  char               arch[10], hostname[64], username[16], pname[PETSC_MAX_PATH_LEN], date[64];
  const char         *name;
  PetscLogDouble     locTotalTime, TotalTime, TotalFlops;
  PetscLogDouble     numMessages, messageLength, avgMessLen, numReductions;
  PetscLogDouble     stageTime, flops, flopr, mem, mess, messLen, red;
  PetscLogDouble     fracTime, fracFlops, fracMessages, fracLength, fracReductions, fracMess, fracMessLen, fracRed;
  PetscLogDouble     fracStageTime, fracStageFlops, fracStageMess, fracStageMessLen, fracStageRed;
  PetscLogDouble     min, max, tot, ratio, avg, x, y;
  PetscLogDouble     minf, maxf, totf, ratf, mint, maxt, tott, ratt, ratCt, totm, totml, totr;
  PetscMPIInt        minCt, maxCt;
  PetscMPIInt        size, rank;
  PetscBool          *localStageUsed,    *stageUsed;
  PetscBool          *localStageVisible, *stageVisible;
  int                numStages, localNumEvents, numEvents;
  int                stage, lastStage, oclass;
  PetscLogEvent      event;
  PetscErrorCode     ierr;
  char               version[256];
  MPI_Comm           comm;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)viewer,&comm);CHKERRQ(ierr);
  ierr = PetscViewerASCIIGetPointer(viewer,&fd);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  /* Pop off any stages the user forgot to remove */
  lastStage = 0;
  ierr = PetscLogGetStageLog(&stageLog);CHKERRQ(ierr);
  ierr = PetscStageLogGetCurrent(stageLog, &stage);CHKERRQ(ierr);
  while (stage >= 0) {
    lastStage = stage;
    ierr = PetscStageLogPop(stageLog);CHKERRQ(ierr);
    ierr = PetscStageLogGetCurrent(stageLog, &stage);CHKERRQ(ierr);
  }
  /* Get the total elapsed time */
  PetscTime(locTotalTime);  locTotalTime -= petsc_BaseTime;

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
  red  = petsc_allreduce_ct + petsc_gather_ct + petsc_scatter_ct;

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
  avg  = (PetscLogDouble) petsc_numObjects;
  ierr = MPI_Allreduce(&avg,          &min, 1, MPIU_PETSCLOGDOUBLE, MPI_MIN, comm);CHKERRQ(ierr);
  ierr = MPI_Allreduce(&avg,          &max, 1, MPIU_PETSCLOGDOUBLE, MPI_MAX, comm);CHKERRQ(ierr);
  ierr = MPI_Allreduce(&avg,          &tot, 1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm);CHKERRQ(ierr);
  avg  = (tot)/((PetscLogDouble) size);
  if (min != 0.0) ratio = max/min; else ratio = 0.0;
  ierr = PetscFPrintf(comm, fd, "Objects:              %5.3e   %10.5f   %5.3e\n", max, ratio, avg);CHKERRQ(ierr);
  /*   Flops */
  ierr = MPI_Allreduce(&petsc_TotalFlops,  &min, 1, MPIU_PETSCLOGDOUBLE, MPI_MIN, comm);CHKERRQ(ierr);
  ierr = MPI_Allreduce(&petsc_TotalFlops,  &max, 1, MPIU_PETSCLOGDOUBLE, MPI_MAX, comm);CHKERRQ(ierr);
  ierr = MPI_Allreduce(&petsc_TotalFlops,  &tot, 1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm);CHKERRQ(ierr);
  avg  = (tot)/((PetscLogDouble) size);
  if (min != 0.0) ratio = max/min; else ratio = 0.0;
  ierr = PetscFPrintf(comm, fd, "Flops:                %5.3e   %10.5f   %5.3e  %5.3e\n", max, ratio, avg, tot);CHKERRQ(ierr);
  TotalFlops = tot;
  /*   Flops/sec -- Must talk to Barry here */
  if (locTotalTime != 0.0) flops = petsc_TotalFlops/locTotalTime; else flops = 0.0;
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
  mess = 0.5*(petsc_irecv_ct + petsc_isend_ct + petsc_recv_ct + petsc_send_ct);
  ierr = MPI_Allreduce(&mess,         &min, 1, MPIU_PETSCLOGDOUBLE, MPI_MIN, comm);CHKERRQ(ierr);
  ierr = MPI_Allreduce(&mess,         &max, 1, MPIU_PETSCLOGDOUBLE, MPI_MAX, comm);CHKERRQ(ierr);
  ierr = MPI_Allreduce(&mess,         &tot, 1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm);CHKERRQ(ierr);
  avg  = (tot)/((PetscLogDouble) size);
  if (min != 0.0) ratio = max/min; else ratio = 0.0;
  ierr = PetscFPrintf(comm, fd, "MPI Messages:         %5.3e   %10.5f   %5.3e  %5.3e\n", max, ratio, avg, tot);CHKERRQ(ierr);
  numMessages = tot;
  /*   Message Lengths */
  mess = 0.5*(petsc_irecv_len + petsc_isend_len + petsc_recv_len + petsc_send_len);
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
  ierr = PetscMalloc(numStages * sizeof(PetscBool), &localStageUsed);CHKERRQ(ierr);
  ierr = PetscMalloc(numStages * sizeof(PetscBool), &stageUsed);CHKERRQ(ierr);
  ierr = PetscMalloc(numStages * sizeof(PetscBool), &localStageVisible);CHKERRQ(ierr);
  ierr = PetscMalloc(numStages * sizeof(PetscBool), &stageVisible);CHKERRQ(ierr);
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
  ierr = PetscFPrintf(comm, fd, "      #   To get timing results run ./configure                #\n");CHKERRQ(ierr);
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
  ierr = PetscFPrintf(comm, fd, "      #   when running ./configure.py.                         #\n");CHKERRQ(ierr);
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
      if (minf < 0.0) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Minimum flops %g over all processors for %s is negative! Not possible!",minf,name);
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
  ierr = PetscOptionsView(viewer);CHKERRQ(ierr);

  /* Machine and compile information */
#if defined(PETSC_USE_FORTRAN_KERNELS)
  ierr = PetscFPrintf(comm, fd, "Compiled with FORTRAN kernels\n");CHKERRQ(ierr);
#else
  ierr = PetscFPrintf(comm, fd, "Compiled without FORTRAN kernels\n");CHKERRQ(ierr);
#endif
#if defined(PETSC_USE_REAL_SINGLE)
  ierr = PetscFPrintf(comm, fd, "Compiled with single precision PetscScalar and PetscReal\n");CHKERRQ(ierr);
#elif defined(PETSC_USE_LONGDOUBLE)
  ierr = PetscFPrintf(comm, fd, "Compiled with long double precision PetscScalar and PetscReal\n");CHKERRQ(ierr);
#endif

#if defined(PETSC_USE_REAL_MAT_SINGLE)
  ierr = PetscFPrintf(comm, fd, "Compiled with single precision matrices\n");CHKERRQ(ierr);
#else
  ierr = PetscFPrintf(comm, fd, "Compiled with full precision matrices (default)\n");CHKERRQ(ierr);
#endif
  ierr = PetscFPrintf(comm, fd, "sizeof(short) %d sizeof(int) %d sizeof(long) %d sizeof(void*) %d sizeof(PetscScalar) %d sizeof(PetscInt) %d\n",
                      (int) sizeof(short), (int) sizeof(int), (int) sizeof(long), (int) sizeof(void*),(int) sizeof(PetscScalar),(int) sizeof(PetscInt));CHKERRQ(ierr);

  ierr = PetscFPrintf(comm, fd, "Configure run at: %s\n",petscconfigureruntime);CHKERRQ(ierr);
  ierr = PetscFPrintf(comm, fd, "Configure options: %s",petscconfigureoptions);CHKERRQ(ierr);
  ierr = PetscFPrintf(comm, fd, "%s", petscmachineinfo);CHKERRQ(ierr);
  ierr = PetscFPrintf(comm, fd, "%s", petsccompilerinfo);CHKERRQ(ierr);
  ierr = PetscFPrintf(comm, fd, "%s", petsccompilerflagsinfo);CHKERRQ(ierr);
  ierr = PetscFPrintf(comm, fd, "%s", petsclinkerinfo);CHKERRQ(ierr);

  /* Cleanup */
  ierr = PetscFPrintf(comm, fd, "\n");CHKERRQ(ierr);
  ierr = PetscStageLogPush(stageLog, lastStage);CHKERRQ(ierr);
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
.seealso: PetscLogBegin(), PetscLogDump(), PetscLogView()
@*/
PetscErrorCode  PetscLogPrintDetailed(MPI_Comm comm, const char filename[])
{
  FILE               *fd                                       = PETSC_STDOUT;
  PetscStageLog      stageLog;
  PetscStageInfo     *stageInfo                                = PETSC_NULL;
  PetscEventPerfInfo *eventInfo                                = PETSC_NULL;
  const char         *name                                     = PETSC_NULL;
  PetscLogDouble     TotalTime;
  PetscLogDouble     stageTime, flops, flopr, mess, messLen, red;
  PetscLogDouble     maxf, totf, maxt, tott, totm, totml, totr = 0.0;
  PetscMPIInt        maxCt;
  PetscBool          *stageUsed;
  PetscBool          *stageVisible;
  int                numStages, numEvents;
  int                stage;
  PetscLogEvent      event;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  /* Pop off any stages the user forgot to remove */
  ierr = PetscLogGetStageLog(&stageLog);CHKERRQ(ierr);
  ierr = PetscStageLogGetCurrent(stageLog, &stage);CHKERRQ(ierr);
  while (stage >= 0) {
    ierr = PetscStageLogPop(stageLog);CHKERRQ(ierr);
    ierr = PetscStageLogGetCurrent(stageLog, &stage);CHKERRQ(ierr);
  }
  /* Get the total elapsed time */
  PetscTime(TotalTime);  TotalTime -= petsc_BaseTime;
  /* Open the summary file */
  if (filename) {
    ierr = PetscFOpen(comm, filename, "w", &fd);CHKERRQ(ierr);
  }

  ierr = PetscFPrintf(comm, fd, "************************************************************************************************************************\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(comm, fd, "***             WIDEN YOUR WINDOW TO 120 CHARACTERS.  Use 'enscript -r -fCourier9' to print this document            ***\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(comm, fd, "************************************************************************************************************************\n");CHKERRQ(ierr);


  numStages = stageLog->numStages;
  ierr = PetscMalloc(numStages * sizeof(PetscBool), &stageUsed);CHKERRQ(ierr);
  ierr = PetscMalloc(numStages * sizeof(PetscBool), &stageVisible);CHKERRQ(ierr);
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
PetscErrorCode  PetscGetFlops(PetscLogDouble *flops)
{
  PetscFunctionBegin;
  *flops = petsc_TotalFlops;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscLogObjectState"
PetscErrorCode  PetscLogObjectState(PetscObject obj, const char format[], ...)
{
  PetscErrorCode ierr;
  size_t         fullLength;
  va_list        Argp;

  PetscFunctionBegin;
  if (!petsc_logObjects) PetscFunctionReturn(0);
  va_start(Argp, format);
  ierr = PetscVSNPrintf(petsc_objects[obj->id].info, 64,format,&fullLength, Argp);CHKERRQ(ierr);
  va_end(Argp);
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
   PetscPreLoadBegin - Begin a segment of code that may be preloaded (run twice)
    to get accurate timings

   Synopsis:
   void PetscPreLoadBegin(PetscBool  flag,char *name);

   Not Collective

   Input Parameter:
+   flag - PETSC_TRUE to run twice, PETSC_FALSE to run once, may be overridden
           with command line option -preload true or -preload false
-   name - name of first stage (lines of code timed separately with -log_summary) to
           be preloaded

   Usage:
.vb
     PetscPreLoadBegin(PETSC_TRUE,"first stage);
       lines of code
       PetscPreLoadStage("second stage");
       lines of code
     PetscPreLoadEnd();
.ve

   Notes: Only works in C/C++, not Fortran

     Flags available within the macro. 
+    PetscPreLoadingUsed - true if we are or have done preloading 
.    PetscPreLoadingOn - true if it is CURRENTLY doing preload
.    PetscPreLoadIt - 0 for the first computation (with preloading turned off it is only 0) 1 for the second
-    PetscPreLoadMax - number of times it will do the computation, only one when preloading is turned on
     The first two variables are available throughout the program, the second two only between the PetscPreLoadBegin()
     and PetscPreLoadEnd()

   Level: intermediate

.seealso: PetscLogEventRegister(), PetscLogEventBegin(), PetscLogEventEnd(), PetscPreLoadEnd(), PetscPreLoadStage()

   Concepts: preloading
   Concepts: timing^accurate
   Concepts: paging^eliminating effects of


M*/

/*MC
   PetscPreLoadEnd - End a segment of code that may be preloaded (run twice)
    to get accurate timings

   Synopsis:
   void PetscPreLoadEnd(void);

   Not Collective

   Usage:
.vb
     PetscPreLoadBegin(PETSC_TRUE,"first stage);
       lines of code
       PetscPreLoadStage("second stage");
       lines of code
     PetscPreLoadEnd();
.ve

   Notes: only works in C/C++ not fortran

   Level: intermediate

.seealso: PetscLogEventRegister(), PetscLogEventBegin(), PetscLogEventEnd(), PetscPreLoadBegin(), PetscPreLoadStage()

M*/

/*MC
   PetscPreLoadStage - Start a new segment of code to be timed separately.
    to get accurate timings

   Synopsis:
   void PetscPreLoadStage(char *name);

   Not Collective

   Usage:
.vb
     PetscPreLoadBegin(PETSC_TRUE,"first stage);
       lines of code
       PetscPreLoadStage("second stage");
       lines of code
     PetscPreLoadEnd();
.ve

   Notes: only works in C/C++ not fortran

   Level: intermediate

.seealso: PetscLogEventRegister(), PetscLogEventBegin(), PetscLogEventEnd(), PetscPreLoadBegin(), PetscPreLoadEnd()

M*/

#include <../src/sys/viewer/impls/ascii/asciiimpl.h>  /*I     "petscsys.h"   I*/
#undef __FUNCT__  
#define __FUNCT__ "PetscLogViewPython"
/*@ 
   PetscLogViewPython - Saves logging information in a Python format.

   Collective on PetscViewer

   Input Paramter:
.   viewer - viewer to save Python data

  Level: intermediate 

@*/
PetscErrorCode  PetscLogViewPython(PetscViewer viewer)
{
  PetscViewer_ASCII  *ascii                  = (PetscViewer_ASCII*)viewer->data;
  FILE               *fd                     = ascii->fd;
  PetscLogDouble     zero                    = 0.0;
  PetscStageLog      stageLog;
  PetscStageInfo     *stageInfo              = PETSC_NULL;
  PetscEventPerfInfo *eventInfo              = PETSC_NULL;
  const char         *name;
  char               stageName[2048];
  char               eventName[2048];
  PetscLogDouble     locTotalTime, TotalTime = 0, TotalFlops = 0;
  PetscLogDouble     numMessages             = 0, messageLength = 0, avgMessLen, numReductions = 0;
  PetscLogDouble     stageTime, flops, mem, mess, messLen, red;
  PetscLogDouble     fracTime, fracFlops, fracMessages, fracLength;
  PetscLogDouble     fracReductions;
  PetscLogDouble     tot,avg,x,y,*mydata;
  PetscMPIInt        maxCt;
  PetscMPIInt        size, rank, *mycount;
  PetscBool          *localStageUsed,    *stageUsed;
  PetscBool          *localStageVisible, *stageVisible;
  int                numStages, localNumEvents, numEvents;
  int                stage, lastStage;
  PetscLogEvent      event;
  PetscErrorCode     ierr;
  PetscInt           i;
  MPI_Comm           comm;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)viewer,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = PetscMalloc(size*sizeof(PetscLogDouble), &mydata);CHKERRQ(ierr);
  ierr = PetscMalloc(size*sizeof(PetscMPIInt), &mycount);CHKERRQ(ierr);

  /* Pop off any stages the user forgot to remove */
  lastStage = 0;
  ierr = PetscLogGetStageLog(&stageLog);CHKERRQ(ierr);
  ierr = PetscStageLogGetCurrent(stageLog, &stage);CHKERRQ(ierr);
  while (stage >= 0) {
    lastStage = stage;
    ierr = PetscStageLogPop(stageLog);CHKERRQ(ierr);
    ierr = PetscStageLogGetCurrent(stageLog, &stage);CHKERRQ(ierr);
  }
  /* Get the total elapsed time */
  PetscTime(locTotalTime);  locTotalTime -= petsc_BaseTime;

  ierr = PetscFPrintf(comm, fd, "\n#------ PETSc Performance Summary ----------\n\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(comm, fd, "Nproc = %d\n",size);CHKERRQ(ierr);

  /* Must preserve reduction count before we go on */
  red  = (petsc_allreduce_ct + petsc_gather_ct + petsc_scatter_ct)/((PetscLogDouble) size);

  /* Calculate summary information */

  /*   Time */
  ierr = MPI_Gather(&locTotalTime,1,MPIU_PETSCLOGDOUBLE,mydata,1,MPIU_PETSCLOGDOUBLE,0,comm);CHKERRQ(ierr);
  if (!rank){
    ierr = PetscFPrintf(comm, fd, "Time = [ " );CHKERRQ(ierr);
    tot  = 0.0;
    for (i=0; i<size; i++){
      tot += mydata[i];
      ierr = PetscFPrintf(comm, fd, "  %5.3e,",mydata[i] );CHKERRQ(ierr);
    }
    ierr = PetscFPrintf(comm, fd, "]\n" );CHKERRQ(ierr);
    avg  = (tot)/((PetscLogDouble) size);
    TotalTime = tot;
  }

  /*   Objects */
  avg  = (PetscLogDouble) petsc_numObjects;
  ierr = MPI_Gather(&avg,1,MPIU_PETSCLOGDOUBLE,mydata,1,MPIU_PETSCLOGDOUBLE,0,comm);CHKERRQ(ierr);
  if (!rank){
    ierr = PetscFPrintf(comm, fd, "Objects = [ " );CHKERRQ(ierr);
    for (i=0; i<size; i++){
      ierr = PetscFPrintf(comm, fd, "  %5.3e,",mydata[i] );CHKERRQ(ierr);
    }
    ierr = PetscFPrintf(comm, fd, "]\n" );CHKERRQ(ierr);
  }

  /*   Flops */
  ierr = MPI_Gather(&petsc_TotalFlops,1,MPIU_PETSCLOGDOUBLE,mydata,1,MPIU_PETSCLOGDOUBLE,0,comm);CHKERRQ(ierr);
  if (!rank){
    ierr = PetscFPrintf(comm, fd, "Flops = [ " );CHKERRQ(ierr);
    tot  = 0.0;
    for (i=0; i<size; i++){
      tot += mydata[i];
      ierr = PetscFPrintf(comm, fd, "  %5.3e,",mydata[i] );CHKERRQ(ierr);
    }
    ierr = PetscFPrintf(comm, fd, "]\n");CHKERRQ(ierr);
    TotalFlops = tot;
  }

  /*   Memory */
  ierr = PetscMallocGetMaximumUsage(&mem);CHKERRQ(ierr);
  ierr = MPI_Gather(&mem,1,MPIU_PETSCLOGDOUBLE,mydata,1,MPIU_PETSCLOGDOUBLE,0,comm);CHKERRQ(ierr);
  if (!rank){
    ierr = PetscFPrintf(comm, fd, "Memory = [ " );CHKERRQ(ierr);
    for (i=0; i<size; i++){
      ierr = PetscFPrintf(comm, fd, "  %5.3e,",mydata[i] );CHKERRQ(ierr);
    }
    ierr = PetscFPrintf(comm, fd, "]\n" );CHKERRQ(ierr);
  }

  /*   Messages */
  mess = 0.5*(petsc_irecv_ct + petsc_isend_ct + petsc_recv_ct + petsc_send_ct);
  ierr = MPI_Gather(&mess,1,MPIU_PETSCLOGDOUBLE,mydata,1,MPIU_PETSCLOGDOUBLE,0,comm);CHKERRQ(ierr);
  if (!rank){
    ierr = PetscFPrintf(comm, fd, "MPIMessages = [ " );CHKERRQ(ierr);
    tot  = 0.0;
    for (i=0; i<size; i++){
      tot += mydata[i];
      ierr = PetscFPrintf(comm, fd, "  %5.3e,",mydata[i] );CHKERRQ(ierr);
    }
    ierr = PetscFPrintf(comm, fd, "]\n" );CHKERRQ(ierr);
    numMessages = tot;
  }

  /*   Message Lengths */
  mess = 0.5*(petsc_irecv_len + petsc_isend_len + petsc_recv_len + petsc_send_len);
  ierr = MPI_Gather(&mess,1,MPIU_PETSCLOGDOUBLE,mydata,1,MPIU_PETSCLOGDOUBLE,0,comm);CHKERRQ(ierr);
  if (!rank){
    ierr = PetscFPrintf(comm, fd, "MPIMessageLengths = [ " );CHKERRQ(ierr);
    tot  = 0.0;
    for (i=0; i<size; i++){
      tot += mydata[i];
      ierr = PetscFPrintf(comm, fd, "  %5.3e,",mydata[i] );CHKERRQ(ierr);
    }
    ierr = PetscFPrintf(comm, fd, "]\n" );CHKERRQ(ierr);
    messageLength = tot;
  }

  /*   Reductions */
  ierr = MPI_Gather(&red,1,MPIU_PETSCLOGDOUBLE,mydata,1,MPIU_PETSCLOGDOUBLE,0,comm);CHKERRQ(ierr);
  if (!rank){
    ierr = PetscFPrintf(comm, fd, "MPIReductions = [ " );CHKERRQ(ierr);
    tot  = 0.0;
    for (i=0; i<size; i++){
      tot += mydata[i];
      ierr = PetscFPrintf(comm, fd, "  %5.3e,",mydata[i] );CHKERRQ(ierr);
    }
    ierr = PetscFPrintf(comm, fd, "]\n" );CHKERRQ(ierr);
    numReductions = tot;
  }

  /* Get total number of stages --
       Currently, a single processor can register more stages than another, but stages must all be registered in order.
       We can removed this requirement if necessary by having a global stage numbering and indirection on the stage ID.
       This seems best accomplished by assoicating a communicator with each stage.
  */
  ierr = MPI_Allreduce(&stageLog->numStages, &numStages, 1, MPI_INT, MPI_MAX, comm);CHKERRQ(ierr);
  ierr = PetscMalloc(numStages * sizeof(PetscBool), &localStageUsed);CHKERRQ(ierr);
  ierr = PetscMalloc(numStages * sizeof(PetscBool), &stageUsed);CHKERRQ(ierr);
  ierr = PetscMalloc(numStages * sizeof(PetscBool), &localStageVisible);CHKERRQ(ierr);
  ierr = PetscMalloc(numStages * sizeof(PetscBool), &stageVisible);CHKERRQ(ierr);
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
        ierr = PetscFPrintf(comm, fd, "\n#Summary of Stages:   ----- Time ------  ----- Flops -----  --- Messages ---  -- Message Lengths --  -- Reductions --\n");CHKERRQ(ierr);
        ierr = PetscFPrintf(comm, fd, "#                       Avg     %%Total     Avg     %%Total   counts   %%Total     Avg         %%Total   counts   %%Total \n");CHKERRQ(ierr);
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
      ierr = PetscFPrintf(comm, fd, "# ");
      ierr = PetscFPrintf(comm, fd, "%2d: %15s: %6.4e %5.1f%%  %6.4e %5.1f%%  %5.3e %5.1f%%  %5.3e      %5.1f%%  %5.3e %5.1f%% \n",
                          stage, name, stageTime/size, 100.0*fracTime, flops, 100.0*fracFlops,
                          mess, 100.0*fracMessages, avgMessLen, 100.0*fracLength, red, 100.0*fracReductions);CHKERRQ(ierr);
    }
  }

  /* Report events */
  ierr = PetscFPrintf(comm,fd,"\n# Event\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(comm,fd,"# ------------------------------------------------------\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(comm,fd,"class Stage(object):\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(comm,fd,"    def __init__(self, name, time, flops, numMessages, messageLength, numReductions):\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(comm,fd,"        # The time and flops represent totals across processes, whereas reductions are only counted once\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(comm,fd,"        self.name          = name\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(comm,fd,"        self.time          = time\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(comm,fd,"        self.flops         = flops\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(comm,fd,"        self.numMessages   = numMessages\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(comm,fd,"        self.messageLength = messageLength\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(comm,fd,"        self.numReductions = numReductions\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(comm,fd,"        self.event         = {}\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(comm,fd, "class Dummy(object):\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(comm,fd, "    pass\n");CHKERRQ(ierr);
  /* Problem: The stage name will not show up unless the stage executed on proc 1 */
  for(stage = 0; stage < numStages; stage++) {
    if (!stageVisible[stage]) continue;
    if (localStageUsed[stage]) {
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

    {
      size_t len, c;

      ierr = PetscStrcpy(stageName, stageInfo[stage].name);CHKERRQ(ierr);
      ierr = PetscStrlen(stageName, &len);CHKERRQ(ierr);
      for(c = 0; c < len; ++c) {
        if (stageName[c] == ' ') stageName[c] = '_';
      }
    }
    if (!rank){
      ierr = PetscFPrintf(comm, fd, "%s = Stage('%s', %g, %g, %g, %g, %g)\n", stageName, stageName, stageTime, flops, mess, messLen, red);CHKERRQ(ierr);
    }

    if (localStageUsed[stage]) {
      eventInfo      = stageLog->stageInfo[stage].eventLog->eventInfo;
      localNumEvents = stageLog->stageInfo[stage].eventLog->numEvents;
    } else {
      localNumEvents = 0;
    }
    ierr = MPI_Allreduce(&localNumEvents, &numEvents, 1, MPI_INT, MPI_MAX, comm);CHKERRQ(ierr);
    for(event = 0; event < numEvents; event++) {
      PetscBool      hasEvent = PETSC_TRUE;
      PetscMPIInt    tmpI;
      PetscLogDouble tmpR;

      if (localStageUsed[stage] && (event < stageLog->stageInfo[stage].eventLog->numEvents) && (eventInfo[event].depth == 0)) {
        size_t len, c;

        ierr = MPI_Allreduce(&eventInfo[event].count, &maxCt, 1, MPI_INT, MPI_MAX, comm);CHKERRQ(ierr);
        ierr = PetscStrcpy(eventName, stageLog->eventLog->eventInfo[event].name);CHKERRQ(ierr);
        ierr = PetscStrlen(eventName, &len);CHKERRQ(ierr);
        for(c = 0; c < len; ++c) {
          if (eventName[c] == ' ') eventName[c] = '_';
        }
      } else {
        ierr = MPI_Allreduce(&ierr, &maxCt, 1, MPI_INT, MPI_MAX, comm);CHKERRQ(ierr);
        eventName[0] = 0;
        hasEvent     = PETSC_FALSE;
      }

      if (maxCt != 0) {
        ierr = PetscFPrintf(comm, fd,"#\n");CHKERRQ(ierr);
        if (!rank){
          ierr = PetscFPrintf(comm, fd, "%s = Dummy()\n",eventName);CHKERRQ(ierr);
          ierr = PetscFPrintf(comm, fd, "%s.event['%s'] = %s\n",stageName,eventName,eventName);CHKERRQ(ierr);
        }
        /* Count */
        if (hasEvent) {tmpI = eventInfo[event].count;}
        else          {tmpI = 0;}
        ierr = MPI_Gather(&tmpI,1, MPI_INT, mycount, 1, MPI_INT, 0, comm);CHKERRQ(ierr);
        ierr = PetscFPrintf(comm, fd, "%s.Count = [ ", eventName);CHKERRQ(ierr);
        if (!rank){
          for (i=0; i<size; i++){
            ierr = PetscFPrintf(comm, fd, "  %7d,",mycount[i] );CHKERRQ(ierr);
          }
          ierr = PetscFPrintf(comm, fd, "]\n" );CHKERRQ(ierr);
        }
        /* Time */
        if (hasEvent) {tmpR = eventInfo[event].time;}
        else          {tmpR = 0.0;}
        ierr = MPI_Gather(&tmpR, 1, MPIU_PETSCLOGDOUBLE, mydata, 1, MPIU_PETSCLOGDOUBLE, 0, comm);CHKERRQ(ierr);
        if (!rank){
          ierr = PetscFPrintf(comm, fd, "%s.Time  = [ ", eventName);CHKERRQ(ierr);
          for (i=0; i<size; i++){
            ierr = PetscFPrintf(comm, fd, "  %5.3e,",mydata[i] );CHKERRQ(ierr);
          }
          ierr = PetscFPrintf(comm, fd, "]\n" );CHKERRQ(ierr);
        }
        /* Flops */
        if (hasEvent) {tmpR = eventInfo[event].flops;}
        else          {tmpR = 0.0;}
        ierr = MPI_Gather(&tmpR, 1, MPIU_PETSCLOGDOUBLE, mydata, 1, MPIU_PETSCLOGDOUBLE, 0, comm);CHKERRQ(ierr);
        if (!rank){
          ierr = PetscFPrintf(comm, fd, "%s.Flops = [ ", eventName);CHKERRQ(ierr);
          for (i=0; i<size; i++){
            ierr = PetscFPrintf(comm, fd, "  %5.3e,",mydata[i] );CHKERRQ(ierr);
          }
          ierr = PetscFPrintf(comm, fd, "]\n" );CHKERRQ(ierr);
        }
        /* Num Messages */
        if (hasEvent) {tmpR = eventInfo[event].numMessages;}
        else          {tmpR = 0.0;}
        ierr = MPI_Gather(&tmpR, 1, MPIU_PETSCLOGDOUBLE, mydata, 1, MPIU_PETSCLOGDOUBLE, 0, comm);CHKERRQ(ierr);
        ierr = PetscFPrintf(comm, fd, "%s.NumMessages = [ ", eventName);CHKERRQ(ierr);
        if (!rank){
          for (i=0; i<size; i++){
            ierr = PetscFPrintf(comm, fd, "  %7.1e,",mydata[i] );CHKERRQ(ierr);
          }
          ierr = PetscFPrintf(comm, fd, "]\n" );CHKERRQ(ierr);
        }
        /* Message Length */
        if (hasEvent) {tmpR = eventInfo[event].messageLength;}
        else          {tmpR = 0.0;}
        ierr = MPI_Gather(&tmpR, 1, MPIU_PETSCLOGDOUBLE, mydata, 1, MPIU_PETSCLOGDOUBLE, 0, comm);CHKERRQ(ierr);
        if (!rank){
          ierr = PetscFPrintf(comm, fd, "%s.MessageLength = [ ", eventName);CHKERRQ(ierr);
          for (i=0; i<size; i++){
            ierr = PetscFPrintf(comm, fd, "  %5.3e,",mydata[i] );CHKERRQ(ierr);
          }
          ierr = PetscFPrintf(comm, fd, "]\n" );CHKERRQ(ierr);
        }
        /* Num Reductions */
        if (hasEvent) {tmpR = eventInfo[event].numReductions;}
        else          {tmpR = 0.0;}
        ierr = MPI_Gather(&tmpR, 1, MPIU_PETSCLOGDOUBLE, mydata, 1, MPIU_PETSCLOGDOUBLE, 0, comm);CHKERRQ(ierr);
        ierr = PetscFPrintf(comm, fd, "%s.NumReductions = [ ", eventName);CHKERRQ(ierr);
        if (!rank){
          for (i=0; i<size; i++){
            ierr = PetscFPrintf(comm, fd, "  %7.1e,",mydata[i] );CHKERRQ(ierr);
          }
          ierr = PetscFPrintf(comm, fd, "]\n" );CHKERRQ(ierr);
        }
      }
    }
  }

  /* Right now, only stages on the first processor are reported here, meaning only objects associated with
     the global communicator, or MPI_COMM_SELF for proc 1. We really should report global stats and then
     stats for stages local to processor sets.
  */
  for(stage = 0; stage < numStages; stage++) {
    if (!localStageUsed[stage]) {
      ierr = PetscFPrintf(comm, fd, "\n--- Event Stage %d: Unknown\n\n", stage);CHKERRQ(ierr);
    }
  }

  ierr = PetscFree(localStageUsed);CHKERRQ(ierr);
  ierr = PetscFree(stageUsed);CHKERRQ(ierr);
  ierr = PetscFree(localStageVisible);CHKERRQ(ierr);
  ierr = PetscFree(stageVisible);CHKERRQ(ierr);
  ierr = PetscFree(mydata);CHKERRQ(ierr);
  ierr = PetscFree(mycount);CHKERRQ(ierr);

  /* Information unrelated to this particular run */
  ierr = PetscFPrintf(comm, fd,
    "# ========================================================================================================================\n");CHKERRQ(ierr);
  PetscTime(y);
  PetscTime(x);
  PetscTime(y); PetscTime(y); PetscTime(y); PetscTime(y); PetscTime(y);
  PetscTime(y); PetscTime(y); PetscTime(y); PetscTime(y); PetscTime(y);
  ierr = PetscFPrintf(comm,fd,"AveragetimetogetPetscTime = %g\n", (y-x)/10.0);CHKERRQ(ierr);
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
    ierr = PetscFPrintf(comm, fd, "AveragetimeforMPI_Barrier = %g\n", (y-x)/5.0);CHKERRQ(ierr);
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
      ierr = PetscFPrintf(comm,fd,"AveragetimforzerosizeMPI_Send = %g\n", (y-x)/size);CHKERRQ(ierr);
    }
    ierr = PetscCommDestroy(&newcomm);CHKERRQ(ierr);
  }

  /* Cleanup */
  ierr = PetscFPrintf(comm, fd, "\n");CHKERRQ(ierr);
  ierr = PetscStageLogPush(stageLog, lastStage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#else /* end of -DPETSC_USE_LOG section */

#undef __FUNCT__  
#define __FUNCT__ "PetscLogObjectState"
PetscErrorCode  PetscLogObjectState(PetscObject obj, const char format[], ...)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#endif /* PETSC_USE_LOG*/


PetscClassId PETSC_LARGEST_CLASSID = PETSC_SMALLEST_CLASSID;
PetscClassId PETSC_OBJECT_CLASSID  = 0;

#undef __FUNCT__  
#define __FUNCT__ "PetscClassIdRegister"
/*@C
  PetscClassIdRegister - Registers a new class name for objects and logging operations in an application code. 

  Not Collective

  Input Parameter:
. name   - The class name
            
  Output Parameter:
. oclass - The class id or classid

  Level: developer

.keywords: log, class, register

@*/
PetscErrorCode  PetscClassIdRegister(const char name[],PetscClassId *oclass )
{
#if defined(PETSC_USE_LOG)
  PetscStageLog  stageLog;
  PetscInt       stage;
  PetscErrorCode ierr;
#endif

  PetscFunctionBegin;
  *oclass = ++PETSC_LARGEST_CLASSID;
#if defined(PETSC_USE_LOG)
  ierr = PetscLogGetStageLog(&stageLog);CHKERRQ(ierr);
  ierr = PetscClassRegLogRegister(stageLog->classLog, name, *oclass);CHKERRQ(ierr);
  for(stage = 0; stage < stageLog->numStages; stage++) {
    ierr = ClassPerfLogEnsureSize(stageLog->stageInfo[stage].classLog, stageLog->classLog->numClasses);CHKERRQ(ierr);
  }
#endif
  PetscFunctionReturn(0);
}
