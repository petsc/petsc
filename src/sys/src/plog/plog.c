/*$Id: plog.c,v 1.242 2000/08/16 16:37:20 bsmith Exp balay $*/
/*
      PETSc code to log object creation and destruction and PETSc events.
*/
#include "petsc.h"        /*I    "petsc.h"   I*/
#include "petscts.h"      /* This include is to define all the PETSc cookies */
#include "petscmachineinfo.h"
#if defined(PETSC_HAVE_MPE)
#include "mpe.h"
#endif
#include <stdarg.h>
#include <sys/types.h>
#include "petscsys.h"
#if defined(PETSC_HAVE_STDLIB_H)
#include <stdlib.h>
#endif
#if defined(PETSC_HAVE_MALLOC_H) && !defined(__cplusplus)
#include <malloc.h>
#endif
#include "petscfix.h"
#include "src/sys/src/plog/ptime.h"

/*
    The next three variables determine which, if any, PLogInfo() calls are used.
  If PLogPrintInfo is zero, no info messages are printed. 
  IF PLogPrintInfoNull is zero, no info messages associated with a null object are printed.

  If PLogInfoFlags[OBJECT_COOKIE - PETSC_COOKIE] is zero, no messages related
  to that object are printed. OBJECT_COOKIE is, for example, MAT_COOKIE.
*/
int  PLogPrintInfo = 0,PLogPrintInfoNull = 0;
int  PLogInfoFlags[] = {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                              1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                              1,1,1,1,1,1,1,1,1,1,1,1};
FILE *PLogInfoFile;

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"PLogInfoAllow"
/*@C
    PLogInfoAllow - Causes PLogInfo() messages to be printed to standard output.

    Not Collective, each processor may call this seperately, but printing is only
    turned on if the lowest processor number associated with the PetscObject associated
    with the call to PLogInfo() has called this routine.

    Input Parameter:
+   flag - PETSC_TRUE or PETSC_FALSE
-   filename - optional name of file to write output to (defaults to stdout)

    Options Database Key:
.   -log_info - Activates PLogInfoAllow()

    Level: advanced

.keywords: allow, information, printing, monitoring

.seealso: PLogInfo()
@*/
int PLogInfoAllow(PetscTruth flag,char *filename)
{
  char fname[256],tname[5];
  int  ierr,rank;

  PetscFunctionBegin;
  PLogPrintInfo     = (int)flag;
  PLogPrintInfoNull = (int)flag;
  if (flag && filename) {
    ierr = PetscFixFilename(filename,fname);CHKERRQ(ierr);
    ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
    sprintf(tname,".%d",rank);
    ierr = PetscStrcat(fname,tname);CHKERRQ(ierr);
    PLogInfoFile = fopen(fname,"w");
    if (!PLogInfoFile) SETERRQ1(1,1,"Cannot open requested file for writing: %s",fname);
  } else if (flag) {
    PLogInfoFile = stdout;
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"PLogInfoDeactivateClass"
/*@
    PLogInfoDeactivateClass - Deactivates PlogInfo() messages for a PETSc object class.

    Not Collective

    Input Parameter:
.   objclass - object class,  e.g., MAT_COOKIE, SNES_COOKIE, etc.

    Notes:
    One can pass PETSC_NULL to deactive all messages that are not associated
    with an object.

    Level: developer

.seealso: PLogInfoActivateClass(), PLogInfo(), PLogInfoAllow()
@*/
int PLogInfoDeactivateClass(int objclass)
{
  PetscFunctionBegin;
  
  if (!objclass) {
    PLogPrintInfoNull = 0;
    PetscFunctionReturn(0); 
  } 
  PLogInfoFlags[objclass - PETSC_COOKIE - 1] = 0;
  
  if (objclass == SLES_COOKIE) {
    PLogInfoFlags[PC_COOKIE - PETSC_COOKIE - 1]  = 0;
    PLogInfoFlags[KSP_COOKIE - PETSC_COOKIE - 1] = 0;
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"PLogInfoActivateClass"
/*@
    PLogInfoActivateClass - Activates PlogInfo() messages for a PETSc 
                            object class.

    Not Collective

    Input Parameter:
.   objclass - object class, e.g., MAT_COOKIE, SNES_COOKIE, etc.

    Level: developer

.seealso: PLogInfoDeactivateClass(), PLogInfo(), PLogInfoAllow()
@*/
int PLogInfoActivateClass(int objclass)
{
  PetscFunctionBegin;
  if (!objclass) {
    PLogPrintInfoNull = 1;
  } else {
    PLogInfoFlags[objclass - PETSC_COOKIE - 1] = 1;
    if (objclass == SLES_COOKIE) {
      PLogInfoFlags[PC_COOKIE - PETSC_COOKIE - 1]  = 1;
      PLogInfoFlags[KSP_COOKIE - PETSC_COOKIE - 1] = 1;
    }
  }
  PetscFunctionReturn(0);
}


/* -------------------------------------------------------------------*/
#if defined(PETSC_USE_LOG)
static int PLOG_USER_EVENT_LOW = PLOG_USER_EVENT_LOW_STATIC;

/* 
   Make sure that all events used by PETSc have the
   corresponding flags set here: 
     1 - activated for PETSc logging
     0 - not activated for PETSc logging
 */
int        PLogEventDepth[200];
PetscTruth PLogEventFlags[] = {PETSC_TRUE,PETSC_TRUE,PETSC_TRUE,PETSC_TRUE,PETSC_TRUE,  /* 0 - 24*/
                        PETSC_TRUE,PETSC_TRUE,PETSC_TRUE,PETSC_TRUE,PETSC_TRUE,
                        PETSC_TRUE,PETSC_TRUE,PETSC_TRUE,PETSC_TRUE,PETSC_TRUE,
                        PETSC_TRUE,PETSC_TRUE,PETSC_TRUE,PETSC_TRUE,PETSC_TRUE,
                        PETSC_TRUE,PETSC_TRUE,PETSC_TRUE,PETSC_TRUE,PETSC_TRUE,
                        PETSC_FALSE,PETSC_TRUE,PETSC_TRUE,PETSC_TRUE,PETSC_TRUE,  /* 25 -49 */
                        PETSC_TRUE,PETSC_TRUE,PETSC_TRUE,PETSC_TRUE,PETSC_TRUE,
                        PETSC_TRUE,PETSC_TRUE,PETSC_TRUE,PETSC_TRUE,PETSC_FALSE,
                        PETSC_TRUE,PETSC_TRUE,PETSC_TRUE,PETSC_TRUE,PETSC_TRUE,
                        PETSC_TRUE,PETSC_TRUE,PETSC_TRUE,PETSC_TRUE,PETSC_TRUE,
                        PETSC_TRUE,PETSC_TRUE,PETSC_TRUE,PETSC_TRUE,PETSC_TRUE, /* 50 - 74 */
                        PETSC_TRUE,PETSC_TRUE,PETSC_TRUE,PETSC_FALSE,PETSC_TRUE,
                        PETSC_FALSE,PETSC_TRUE,PETSC_TRUE,PETSC_TRUE,PETSC_FALSE,
                        PETSC_TRUE,PETSC_FALSE,PETSC_TRUE,PETSC_FALSE,PETSC_TRUE,
                        PETSC_TRUE,PETSC_TRUE,PETSC_TRUE,PETSC_TRUE,PETSC_TRUE,
                        PETSC_TRUE,PETSC_TRUE,PETSC_TRUE,PETSC_TRUE,PETSC_TRUE, /* 75 - 99 */
                        PETSC_TRUE,PETSC_TRUE,PETSC_TRUE,PETSC_TRUE,PETSC_TRUE,
                        PETSC_TRUE,PETSC_TRUE,PETSC_FALSE,PETSC_TRUE,PETSC_TRUE,
                        PETSC_TRUE,PETSC_TRUE,PETSC_TRUE,PETSC_TRUE,PETSC_TRUE,
                        PETSC_TRUE,PETSC_TRUE,PETSC_TRUE,PETSC_TRUE,PETSC_TRUE,
                        PETSC_TRUE,PETSC_FALSE,PETSC_FALSE,PETSC_TRUE,PETSC_TRUE, /* 100 - 124 */ 
                        PETSC_TRUE,PETSC_TRUE,PETSC_TRUE,PETSC_TRUE,PETSC_TRUE,
                        PETSC_TRUE,PETSC_TRUE,PETSC_TRUE,PETSC_TRUE,PETSC_TRUE,
                        PETSC_TRUE,PETSC_TRUE,PETSC_TRUE,PETSC_TRUE,PETSC_TRUE,
                        PETSC_TRUE,PETSC_TRUE,PETSC_TRUE,PETSC_TRUE,PETSC_TRUE,
                        PETSC_TRUE,PETSC_TRUE,PETSC_TRUE,PETSC_TRUE,PETSC_TRUE, /* 125 - 149 */
                        PETSC_TRUE,PETSC_TRUE,PETSC_TRUE,PETSC_TRUE,PETSC_TRUE,
                        PETSC_TRUE,PETSC_TRUE,PETSC_TRUE,PETSC_TRUE,PETSC_TRUE,
                        PETSC_TRUE,PETSC_TRUE,PETSC_TRUE,PETSC_TRUE,PETSC_TRUE,
                        PETSC_TRUE,PETSC_TRUE,PETSC_TRUE,PETSC_TRUE,PETSC_TRUE,
                        PETSC_TRUE,PETSC_TRUE,PETSC_TRUE,PETSC_TRUE,PETSC_TRUE, /* 150 - 174 */
                        PETSC_TRUE,PETSC_TRUE,PETSC_TRUE,PETSC_TRUE,PETSC_TRUE,
                        PETSC_TRUE,PETSC_TRUE,PETSC_TRUE,PETSC_TRUE,PETSC_TRUE,
                        PETSC_TRUE,PETSC_TRUE,PETSC_TRUE,PETSC_TRUE,PETSC_TRUE,
                        PETSC_TRUE,PETSC_TRUE,PETSC_TRUE,PETSC_TRUE,PETSC_TRUE,
                        PETSC_TRUE,PETSC_TRUE,PETSC_TRUE,PETSC_TRUE,PETSC_TRUE,
                        PETSC_TRUE,PETSC_TRUE,PETSC_TRUE,PETSC_TRUE,PETSC_TRUE, /* 175 - 199 */
                        PETSC_TRUE,PETSC_TRUE,PETSC_TRUE,PETSC_TRUE,PETSC_TRUE,
                        PETSC_TRUE,PETSC_TRUE,PETSC_TRUE,PETSC_TRUE,PETSC_TRUE,
                        PETSC_TRUE,PETSC_TRUE,PETSC_TRUE,PETSC_TRUE,PETSC_TRUE,
                        PETSC_TRUE,PETSC_TRUE,PETSC_TRUE,PETSC_TRUE,PETSC_TRUE};

static char *(oname[]) = {"Viewer           ",
                          "Index set        ",
                          "Vector           ",
                          "Vector Scatter   ",
                          "Matrix           ",
                          "Draw             ",
                          "Line graph       ",
                          "Krylov Solver    ",
                          "Preconditioner   ",
                          "SLES             ",  /* 10 */
                          "EC               ",  /* 11 */
                          "                 ",
                          "SNES             ",
                          "Distributed array",
                          "DF               ", /* 15 */
                          "Axis             ", /* 16 */
                          "Null Space       ",
                          "TS               ",
                          "Random           ",
                          "AO               ", /* 20 */
                          "DC               ",
                          "FD Coloring      ",
                          "Grid             ", /* 23 */
                          "                 ",
                          "                 ",
                          "                 ",
                          "Draw SP          ", /* 27 */                          
                          "                 ",                          
                          "                 ",                          
                          "                 ",                          
                          "                 ",                          
                          "                 ",                          
                          "                 ",                          
                          "                 ",                          
                          "                 ",                          
                          "                 ",                          
                          "                 ",                          
                          "                 ",                          
                          "                 ",                          
                          "                 ",                          
                          "                 ",                          
                          "                 ",                          
                          "                 ",                          
                          "                 ",                          
                          "                 ",                          
                          "                 ",                          
                          "                 ",                          
                          "                 ",                          
                          "                 ",                          
			  "                 "};

char *(PLogEventName[]) = {"MatMult         ",
                         "MatMatFreeMult  ",
                         "MatAssemblyBegin",
                         "MatAssemblyEnd  ",
                         "MatGetOrdering  ",
                         "MatMultTranspose",
                         "MatMultAdd      ",
                         "MatMultTransAdd ",
                         "MatLUFactor     ",
                         "MatCholeskyFctr ",
                         "MatLUFctrSymbol ",
                         "MatILUFctrSymbol",
                         "MatCholeskyFctr ",
                         "MatIncompleteCho",
                         "MatLUFactorNumer",
                         "MatCholeskyFact ",
                         "MatRelax        ",
                         "MatCopy         ",
                         "MatConvert      ",
                         "MatScale        ",
                         "MatZeroEntries  ",
                         "MatSolve        ",
                         "MatSolveAdd     ",
                         "MatSolveTranspos",
                         "MatSolveTransAdd",
                         "MatSetValues    ",
                         "MatForwardSolve ",
                         "MatBackwardSolve",
                         "MatLoad         ",
                         "MatView         ",
                         "MatILUFactor    ",
                         "MatGetColoring  ",
                         "MatGetSubMatrice",
                         "MatGetValues    ",
                         "MatIncreaseOvlap",
                         "MatGetRow       ",
                         "MatGetPartitioni",
                         "VecReduceArith. ",
                         "MatFDColoringApp",
                         "VecView         ",
                         "MatFDColoringCre",
                         "                ",
                         "VecMax          ",
                         "VecMin          ",
                         "VecTDot         ",
                         "VecScale        ",
                         "VecCopy         ",
                         "VecSet          ",
                         "VecAXPY         ",
                         "VecAYPX         ",
                         "VecSwap         ",
                         "VecWAXPY        ",
                         "VecAssemblyBegin",
                         "VecAssemblyEnd  ",
                         "VecMTDot        ",
                         "                ",
                         "VecMAXPY        ",
                         "VecPointwiseMult",
                         "VecSetValues    ",
                         "VecLoad         ",
                         "VecScatterBarrie",
                         "VecScatterBegin ",
                         "VecScatterEnd   ",
                         "VecSetRandom    ",
                         "VecNormBarrier  ",
                         "VecNorm         ",
                         "VecDotBarrier   ",
                         "VecDot          ",
                         "VecMDotBarrier  ",
                         "VecMDot         ",
                         "SLESSolve       ",
                         "SLESSetUp       ",
                         "KSPGMRESOrthog  ",
                         "PCApplyCoarse   ",
                         "PCModifySubMat  ",
                         "PCSetUp         ",
                         "PCSetUpOnBlocks ",
                         "PCApply         ",
                         "PCApplySymmLeft ",
                         "PCApplySymmRight",
                         "SNESSolve       ",
                         "SNESLineSearch  ",
                         "SNESFunctionEval",
                         "SNESJacobianEval",
                         "SNESMinFunctnEvl",
                         "SNESGradientEval",
                         "SNESHessianEval ",
                         "VecReduceBarrier",
                         "VecReduceCommuni",
                         "               ",
                         "TSStep          ",
                         "TSPseudoCmptTStp",
                         "TSFunctionEval  ",
                         "TSJacobianEval  ",
                         " ",
                         " ",
                         " ",
                         " ",
                         " ",
                         " ",
                         "PetscBarrier    ", /* 100 */
                         "                ",
                         "                ",
                         " ",
                         " ",
                         "ECSetUp         ",
                         "ECSolve         ",
                         " ",
                         " ",
                         " ",
                         "                ",
                         "                ",
                         "                ",
                         "                ",
                         " "," "," "," "," ",
                         " "," "," "," "," ",
                         " "," "," "," "," ",
                         " "," "," "," "," ",
                         " "," "," "," "," ",
                         " "," "," "," "," ",
                         " "," "," "," "," ",
                         " "," "," "," "," ",
                         " "," "," "," "," ",
                         " "," "," "," "," ",
                         " "," "," "," "," ",
                         " "," "," "," "," ",
                         " "," "," "," "," ",
                         " "," "," "," "," ",
                         " "," "," "," "," ",
                         " "," "," "," "," ",
                         " "," "," "," "," ",
                         " "," "," "," "," ",
                         " "," "," "," "," ",
                         " "," "," "," "," "};

#define CHUNCK       1000
#define CREATE       0
#define DESTROY      1
#define ACTIONBEGIN  2
#define ACTIONEND    3

/*
    flops contains cumulative flops 
    mem contains current memory usage
    memmax contains maximum memory usage so far
*/
typedef struct {
  PLogDouble      time,flops,mem,maxmem;
  int             cookie,type,event,id1,id2,id3;
} Events;

typedef struct {
  int         parent;
  PLogDouble  mem;
  char        string[64];
  char        name[32];
  PetscObject obj;
} Objects;

/* 
    Global counters 
*/
PLogDouble _TotalFlops = 0.0;
PLogDouble irecv_ct = 0.0,isend_ct = 0.0,wait_ct = 0.0,wait_any_ct = 0.0;
PLogDouble irecv_len = 0.0,isend_len = 0.0,recv_len = 0.0,send_len = 0.0;
PLogDouble send_ct = 0.0,recv_ct = 0.0;
PLogDouble wait_all_ct = 0.0,allreduce_ct = 0.0,sum_of_waits_ct = 0.0;

/* used in the MPI_XXX() count macros in petsclog.h */
int PETSC_DUMMY,PETSC_DUMMY_SIZE;

/*
    Log counters in this file only 
*/
static PLogDouble  BaseTime;
static Events      *events = 0;
static Objects     *objects = 0;

static int         nobjects = 0,nevents = 0,objectsspace = CHUNCK;
static int         ObjectsDestroyed = 0,eventsspace = CHUNCK;
static PLogDouble  ObjectsType[10][PETSC_MAX_COOKIES][4];

static int         EventsStage = 0;    /* which log sessions are we using */
static int         EventsStageMax = 0; /* highest event log used */ 
static int         EventsStagePushed = 0;
static int         EventsStageStack[100];
static char        *(EventsStageName[]) = {0,0,0,0,0,0,0,0,0,0};
static PLogDouble  EventsStageFlops[] = {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0};
static PLogDouble  EventsStageTime[] = {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0};
static PLogDouble  EventsStageMessageCounts[] = {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0};
static PLogDouble  EventsStageMessageLengths[] = {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0};
static PLogDouble  EventsStageReductions[] = {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0};
       PetscTruth  PLogStagePrintFlag[] = {PETSC_TRUE,PETSC_TRUE,PETSC_TRUE,PETSC_TRUE,PETSC_TRUE,
                                       PETSC_TRUE,PETSC_TRUE,PETSC_TRUE,PETSC_TRUE,PETSC_TRUE};
#define COUNT      0
#define FLOPS      1
#define TIME       2
#define MESSAGES   3
#define LENGTHS    4
#define REDUCTIONS 5
static PLogDouble  EventsType[10][PLOG_USER_EVENT_HIGH][6];
static int         EventsStagePrevious = 0;

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"PLogStageRegister"
/*@C
    PLogStageRegister - Attaches a charactor string name to a logging stage.

    Not Collective

    Input Parameters:
+   stage - the stage from 0 to 9 inclusive (use PETSC_DETERMINE for current stage)
-   sname - the name to associate with that stage

    Notes:
    The string information (for stage names) is not copied, so the user
    should NOT change any strings specified here.

    Level: intermediate

.seealso: PLogStagePush(), PLogStagePop(), PreLoadBegin(), PreLoadEnd(), PreLoadStage()
@*/
int PLogStageRegister(int stage,const char sname[])
{
  int ierr;

  PetscFunctionBegin;
  if (stage == PETSC_DETERMINE) stage = EventsStage;
  if (stage < 0 || stage > 10) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,0,"Stage must be >= 0 and < 10: Instead %d",stage);
  ierr = PetscStrallocpy(sname,&EventsStageName[stage]);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"PLogStagePrint"
/*@C
    PLogStagePrint - Tells PLogPrintSummary() whether to print this stage or not

    Collective on PETSC_COMM_WORLD 

    Input Parameters:
+   stage - the stage from 0 to 9 inclusive (use PETSC_DETERMINE for current stage)
-   flg - PETSC_TRUE to print, else PETSC_FALSE (defaults to PETSC_TRUE)

    Level: intermediate

.seealso: PLogStagePush(), PLogStagePop(), PreLoadBegin(), PreLoadEnd(), PreLoadStage()
@*/
int PLogStagePrint(int stage,PetscTruth flg)
{
  PetscFunctionBegin;
  if (stage == PETSC_DETERMINE) stage = EventsStage;
  if (stage < 0 || stage > 10) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,0,"Stage must be >= 0 and < 10: Instead %d",stage);
  PLogStagePrintFlag[stage] = flg;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"PLogStagePush"
/*@C
   PLogStagePush - Users can log up to 10 stages within a code by using
   -log_summary in conjunction with PLogStagePush() and PLogStagePop().

   Not Collective

   Input Parameter:
.  stage - stage on which to log (0 <= stage <= 9)

   Usage:
   If the option -log_sumary is used to run the program containing the 
   following code, then 3 sets of summary data will be printed during
   PetscFinalize().
.vb
      PetscInitialize(int *argc,char ***args,0,0);
      [stage 0 of code]   
      PLogStagePush(1);
      [stage 1 of code]
      PLogStagePop();
      PetscBarrier(...);
      [more stage 0 of code]   
      PetscFinalize();
.ve
 
   Notes:  
   Use PETSC_DETERMINE to increase the previous stage number (which was poped) by one
   Use PLogStageRegister() to register a name with a stage.

   Level: intermediate

.keywords: log, push, stage

.seealso: PLogStagePop(), PLogStageRegister(), PetscBarrier(), PreLoadBegin(), PreLoadEnd(),
          PreLoadStage()
@*/
int PLogStagePush(int stage)
{
  PetscFunctionBegin;
  if (stage == PETSC_DETERMINE) {
    stage = EventsStagePrevious + 1;
  }

  if (stage < 0 || stage > 10) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,0,"Stage must be >= 0 < 10: Instead %d",stage);
  /* record flops/time of previous stage */
  if (EventsStagePushed) {
    PetscTimeAdd(EventsStageTime[EventsStage]);
    EventsStageFlops[EventsStage]          += _TotalFlops;
    EventsStageMessageCounts[EventsStage]  += irecv_ct + isend_ct + recv_ct + send_ct;
    EventsStageMessageLengths[EventsStage] += irecv_len + isend_len + recv_len + send_len;
    EventsStageReductions[EventsStage]     += allreduce_ct;
  }
  EventsStageStack[EventsStagePushed] = EventsStage;
  if (EventsStagePushed++ > 99) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Too many pushes");
  EventsStage = stage;
  if (stage > EventsStageMax) EventsStageMax = stage;
  PetscTimeSubtract(EventsStageTime[EventsStage]);
  EventsStageFlops[EventsStage]          -= _TotalFlops;
  EventsStageMessageCounts[EventsStage]  -= irecv_ct + isend_ct + recv_ct + send_ct;
  EventsStageMessageLengths[EventsStage] -= irecv_len + isend_len + recv_len + send_len;
  EventsStageReductions[EventsStage]     -= allreduce_ct;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"PLogStagePop"
/*@C
   PLogStagePop - Users can log up to 10 stages within a code by using
   -log_summary in conjunction with PLogStagePush() and PLogStagePop().

   Not Collective

   Usage:
   If the option -log_sumary is used to run the program containing the 
   following code, then 2 sets of summary data will be printed during
   PetscFinalize().
.vb
      PetscInitialize(int *argc,char ***args,0,0);
      [stage 0 of code]   
      PLogStagePush(1);
      [stage 1 of code]
      PLogStagePop();
      PetscBarrier(...);
      [more stage 0 of code]   
      PetscFinalize();
.ve

   Notes:  
   Use PLogStageRegister() to register a stage.

   Level: intermediate

.keywords: log, pop, stage

.seealso: PLogStagePush(), PLogStageRegister(), PetscBarrier()
@*/
int PLogStagePop(void)
{
  PetscFunctionBegin;
  EventsStagePrevious = EventsStage; /* keep a record of too be poped stage */
  PetscTimeAdd(EventsStageTime[EventsStage]);
  EventsStageFlops[EventsStage]          += _TotalFlops;
  EventsStageMessageCounts[EventsStage]  += irecv_ct + isend_ct + recv_ct + send_ct;
  EventsStageMessageLengths[EventsStage] += irecv_len + isend_len + recv_len + send_len;
  EventsStageReductions[EventsStage]     += allreduce_ct;
  if (EventsStagePushed < 1) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Too many pops\n");
  EventsStage = EventsStageStack[--EventsStagePushed];
  if (EventsStagePushed) {
    PetscTimeSubtract(EventsStageTime[EventsStage]);
    EventsStageFlops[EventsStage]          -= _TotalFlops;
    EventsStageMessageCounts[EventsStage]  -= irecv_ct + isend_ct + recv_ct + send_ct;
    EventsStageMessageLengths[EventsStage] -= irecv_len + isend_len + recv_len + send_len;
    EventsStageReductions[EventsStage]     -= allreduce_ct;
  }
  PetscFunctionReturn(0);
}
#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"PLogStageDestroy_Private"
/*
   PLogStageDestroy_Private - Destroy the memory allocated during calls to 
        PLogStateRegister().

*/

int PLogStageDestroy_Private(void)
{
  int i,ierr;
  PetscFunctionBegin;
  for (i=0; i<10; i++) {
    if (EventsStageName[i]) {
      ierr = PetscFree(EventsStageName[i]);CHKERRQ(ierr);
      EventsStageName[i] =0;
    }
  }
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------------------*/

int (*_PLogPHC)(PetscObject)= 0;
int (*_PLogPHD)(PetscObject)= 0;
int (*_PLogPLB)(int,int,PetscObject,PetscObject,PetscObject,PetscObject) = 0;
int (*_PLogPLE)(int,int,PetscObject,PetscObject,PetscObject,PetscObject) = 0;

/*
      Default object create logger 
*/
#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"PLogDefaultPHC"
int PLogDefaultPHC(PetscObject obj)
{
  int ierr;

  PetscFunctionBegin;
  if (nevents >= eventsspace) {
    Events *tmp;
    PLogDouble end,start;
    PetscTime(start);
    tmp = (Events*)malloc((eventsspace+CHUNCK)*sizeof(Events));CHKPTRQ(tmp);
    ierr = PetscMemcpy(tmp,events,eventsspace*sizeof(Events));CHKERRQ(ierr);
    free(events);
    events = tmp; eventsspace += CHUNCK;
    PetscTime(end); BaseTime += (end - start);
  }
  if (nobjects >= objectsspace) {
    Objects *tmp;
    PLogDouble end,start;
    PetscTime(start);
    tmp = (Objects*)malloc((objectsspace+CHUNCK)*sizeof(Objects));CHKPTRQ(tmp);
    ierr = PetscMemcpy(tmp,objects,objectsspace*sizeof(Objects));CHKERRQ(ierr);
    free(objects);
    objects = tmp; objectsspace += CHUNCK;
    PetscTime(end); BaseTime += (end - start);
  }
  PetscTime(events[nevents].time); events[nevents].time -= BaseTime;
  events[nevents].cookie  = obj->cookie - PETSC_COOKIE - 1;
  events[nevents].type    = obj->type;
  events[nevents].id1     = nobjects;
  events[nevents].id2     = -1;
  events[nevents].id3     = -1;
  events[nevents].flops   = _TotalFlops;
  PetscTrSpace(&events[nevents].mem,PETSC_NULL,&events[nevents].maxmem);
  events[nevents++].event = CREATE;
  objects[nobjects].parent= -1;
  objects[nobjects].obj   = obj;
  ierr = PetscMemzero(objects[nobjects].string,64*sizeof(char));CHKERRQ(ierr);
  ierr = PetscMemzero(objects[nobjects].name,16*sizeof(char));CHKERRQ(ierr);
  obj->id = nobjects++;
  ObjectsType[EventsStage][obj->cookie - PETSC_COOKIE-1][0]++;
  PetscFunctionReturn(0);
}
/*
      Default object destroy logger 
*/
#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"PLogDefaultPHD"
int PLogDefaultPHD(PetscObject obj)
{
  PetscObject parent;
  PetscTruth  exists;
  int         ierr;

  PetscFunctionBegin;
  if (nevents >= eventsspace) {
    Events *tmp;
    PLogDouble end,start;
    PetscTime(start);
    tmp = (Events*)malloc((eventsspace+CHUNCK)*sizeof(Events));CHKPTRQ(tmp);
    ierr = PetscMemcpy(tmp,events,eventsspace*sizeof(Events));CHKERRQ(ierr);
    free(events);
    events = tmp; eventsspace += CHUNCK;
    PetscTime(end); BaseTime += (end - start);
  }
  PetscTime(events[nevents].time); events[nevents].time -= BaseTime;
  events[nevents].event     = DESTROY;
  events[nevents].cookie    = obj->cookie - PETSC_COOKIE - 1;
  events[nevents].type      = obj->type;
  events[nevents].id1       = obj->id;
  events[nevents].id2       = -1;
  events[nevents].flops   = _TotalFlops;
  PetscTrSpace(&events[nevents].mem,PETSC_NULL,&events[nevents].maxmem);
  events[nevents++].id3     = -1;
  if (obj->parent) {
    ierr = PetscObjectExists(obj->parent,&exists);CHKERRQ(ierr);
    if (exists) {
      objects[obj->id].parent   = obj->parent->id;
    } else {
      objects[obj->id].parent   = -1;
    }
  } else {
    objects[obj->id].parent   = -1;
  }
  if (obj->name) { ierr = PetscStrncpy(objects[obj->id].name,obj->name,16);CHKERRQ(ierr);}
  objects[obj->id].obj      = 0;
  objects[obj->id].mem      = obj->mem;
  ObjectsType[EventsStage][obj->cookie - PETSC_COOKIE-1][1]++;
  ObjectsType[EventsStage][obj->cookie - PETSC_COOKIE-1][2] += obj->mem;
  /*
     Credit all ancestors with your memory 
  */
  parent = obj->parent;
  while (parent) {
    ierr = PetscObjectExists(parent,&exists);CHKERRQ(ierr);
    if (!exists) break;
    ObjectsType[EventsStage][parent->cookie - PETSC_COOKIE-1][3] += obj->mem;   
    parent = parent->parent;
  } 
  ObjectsDestroyed++;
  PetscFunctionReturn(0);
}
/*
    Event begin logger with complete logging
*/
#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"PLogDefaultPLBAll"
int PLogDefaultPLBAll(int event,int t,PetscObject o1,PetscObject o2,PetscObject o3,PetscObject o4)
{
  PLogDouble ltime;
  int        ierr;

  PetscFunctionBegin;
  if (nevents >= eventsspace) {
    Events *tmp;
    PLogDouble end,start;
    PetscTime(start);
    tmp  = (Events*)malloc((eventsspace+CHUNCK)*sizeof(Events));CHKPTRQ(tmp);
    ierr = PetscMemcpy(tmp,events,eventsspace*sizeof(Events));CHKERRQ(ierr);
    free(events);
    events = tmp; eventsspace += CHUNCK;
    PetscTime(end); BaseTime += (end - start);
  }
  PetscTime(ltime);
  events[nevents].time = ltime - BaseTime;
  if (o1) events[nevents].id1     = o1->id; else events[nevents].id1 = -1;
  if (o2) events[nevents].id2     = o2->id; else events[nevents].id2 = -1;
  if (o3) events[nevents].id3     = o3->id; else events[nevents].id3 = -1;
  events[nevents].type   = event;
  events[nevents].cookie = 0;
  events[nevents].flops   = _TotalFlops;
  PetscTrSpace(&events[nevents].mem,PETSC_NULL,&events[nevents].maxmem);
  events[nevents++].event= ACTIONBEGIN;
  EventsType[EventsStage][event][COUNT]++;
  EventsType[EventsStage][event][TIME]        -= ltime;
  EventsType[EventsStage][event][FLOPS]       -= _TotalFlops;
  EventsType[EventsStage][event][MESSAGES]    -= irecv_ct + isend_ct + recv_ct + send_ct;
  EventsType[EventsStage][event][LENGTHS]     -= irecv_len + isend_len + recv_len + send_len;
  EventsType[EventsStage][event][REDUCTIONS]  -= allreduce_ct;
  PetscFunctionReturn(0);
}
/*
     Event end logger with complete logging
*/
#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"PLogDefaultPLEAll"
int PLogDefaultPLEAll(int event,int t,PetscObject o1,PetscObject o2,PetscObject o3,PetscObject o4)
{
  PLogDouble ltime;
  int        ierr;

  PetscFunctionBegin;
  if (nevents >= eventsspace) {
    Events *tmp;
    PLogDouble end,start;
    PetscTime(start);
    tmp  = (Events*)malloc((eventsspace+CHUNCK)*sizeof(Events));CHKPTRQ(tmp);
    ierr = PetscMemcpy(tmp,events,eventsspace*sizeof(Events));CHKERRQ(ierr);
    free(events);
    events = tmp; eventsspace += CHUNCK;
    PetscTime(end); BaseTime += (end - start);
  }
  PetscTime(ltime);
  events[nevents].time   = ltime - BaseTime;
  if (o1) events[nevents].id1    = o1->id; else events[nevents].id1 = -1;
  if (o2) events[nevents].id2    = o2->id; else events[nevents].id2 = -1;
  if (o3) events[nevents].id3    = o3->id; else events[nevents].id3 = -1;
  events[nevents].type   = event;
  events[nevents].cookie = 0;
  events[nevents].flops   = _TotalFlops;
  PetscTrSpace(&events[nevents].mem,PETSC_NULL,&events[nevents].maxmem);
  events[nevents++].event= ACTIONEND;
  EventsType[EventsStage][event][TIME]        += ltime;
  EventsType[EventsStage][event][FLOPS]       += _TotalFlops;
  EventsType[EventsStage][event][MESSAGES]    += irecv_ct + isend_ct + recv_ct + send_ct;
  EventsType[EventsStage][event][LENGTHS]     += irecv_len + isend_len + recv_len + send_len;
  EventsType[EventsStage][event][REDUCTIONS]  += allreduce_ct;
  PetscFunctionReturn(0);
}
/*
     Default event begin logger
*/
#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"PLogDefaultPLB"
int PLogDefaultPLB(int event,int t,PetscObject o1,PetscObject o2,PetscObject o3,PetscObject o4)
{
  PetscFunctionBegin;
  EventsType[EventsStage][event][COUNT]++;
  PetscTimeSubtract(EventsType[EventsStage][event][TIME]);
  EventsType[EventsStage][event][FLOPS]       -= _TotalFlops;
  EventsType[EventsStage][event][MESSAGES]    -= irecv_ct + isend_ct + recv_ct + send_ct;
  EventsType[EventsStage][event][LENGTHS]     -= irecv_len + isend_len + recv_len + send_len;
  EventsType[EventsStage][event][REDUCTIONS]  -= allreduce_ct;
  PetscFunctionReturn(0);
}

/*
     Default event end logger
*/
#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"PLogDefaultPLE"
int PLogDefaultPLE(int event,int t,PetscObject o1,PetscObject o2,PetscObject o3,PetscObject o4)
{
  PetscFunctionBegin;
  PetscTimeAdd(EventsType[EventsStage][event][TIME]);
  EventsType[EventsStage][event][FLOPS]       += _TotalFlops;
  EventsType[EventsStage][event][MESSAGES]    += irecv_ct + isend_ct + recv_ct + send_ct;
  EventsType[EventsStage][event][LENGTHS]     += irecv_len + isend_len + recv_len + send_len;
  EventsType[EventsStage][event][REDUCTIONS]  += allreduce_ct;
  PetscFunctionReturn(0);
}

/*
     Default trace event logging routines
*/
FILE   *tracefile = 0;
int    tracelevel = 0;
char   *traceblanks = "                                                                    ";
char   tracespace[72];
PLogDouble tracetime = 0.0;

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"PLogDefaultPLBTrace"
int PLogDefaultPLBTrace(int event,int t,PetscObject o1,PetscObject o2,PetscObject o3,PetscObject o4)
{
  int        rank,ierr;
  PLogDouble cur_time;

  PetscFunctionBegin;
  if (!tracetime) { PetscTime(tracetime);}

  ierr = MPI_Comm_rank(MPI_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = PetscStrncpy(tracespace,traceblanks,2*tracelevel);CHKERRQ(ierr);
  tracespace[2*tracelevel] = 0;
  PetscTime(cur_time);
  fprintf(tracefile,"%s[%d] %g Event begin: %s\n",tracespace,rank,cur_time-tracetime,PLogEventName[event]);
  fflush(tracefile);
  tracelevel++;

  PetscFunctionReturn(0);
}

/*
     Default trace event logging
*/
#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"PLogDefaultPLETrace"
int PLogDefaultPLETrace(int event,int t,PetscObject o1,PetscObject o2,PetscObject o3,PetscObject o4)
{
  int        ierr,rank;
  PLogDouble cur_time;

  PetscFunctionBegin;
  tracelevel--;
  ierr = MPI_Comm_rank(MPI_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = PetscStrncpy(tracespace,traceblanks,2*tracelevel);CHKERRQ(ierr);
  tracespace[2*tracelevel] = 0;
  PetscTime(cur_time);
  fprintf(tracefile,"%s[%d] %g Event end: %s\n",tracespace,rank,cur_time-tracetime,PLogEventName[event]);
  fflush(tracefile);
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------------*/
#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"PLogObjectState"
int PLogObjectState(PetscObject obj,const char format[],...)
{
  va_list Argp;

  PetscFunctionBegin;
  if (!objects) PetscFunctionReturn(0);
  va_start(Argp,format);
#if defined(PETSC_HAVE_VPRINTF_CHAR)
  vsprintf(objects[obj->id].string,format,(char *)Argp);
#else
  vsprintf(objects[obj->id].string,format,Argp);
#endif
  va_end(Argp);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"PLogSet"
/*@C
   PLogSet - Sets the logging functions called at the beginning and ending 
              of every event.

   Not Collective

   Input Parameters:
+  b - function called at beginning of event
-  e - function called at end of event

   Level: developer

.seealso: PLogDump(), PLogBegin(), PLogAllBegin(), PLogTraceBegin()

@*/
int PLogSet(int (*b)(int,int,PetscObject,PetscObject,PetscObject,PetscObject),
            int (*e)(int,int,PetscObject,PetscObject,PetscObject,PetscObject))
{
  PetscFunctionBegin;
  _PLogPLB    = b;
  _PLogPLE    = e;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"PLogAllBegin"
/*@C
   PLogAllBegin - Turns on extensive logging of objects and events. Logs 
   all events. This creates large log files and slows the program down.

   Not Collective

   Options Database Keys:
.  -log_all - Prints extensive log information (for code compiled with PETSC_USE_LOG)

   Usage:
.vb
     PetscInitialize(...);
     PLogAllBegin();
     ... code ...
     PLogDump(filename);
     PetscFinalize();
.ve

   Notes:
   A related routine is PLogBegin (with the options key -log), which is 
   intended for production runs since it logs only flop rates and object
   creation (and should not significantly slow the programs).

   Level: advanced

.keywords: log, all, begin

.seealso: PLogDump(), PLogBegin(), PLogTraceBegin()
@*/
int PLogAllBegin(void)
{
  int ierr;

  PetscFunctionBegin;
  objects  = (Objects*)malloc(CHUNCK*sizeof(Objects));CHKPTRQ(objects);
  events   = (Events*)malloc(CHUNCK*sizeof(Events));CHKPTRQ(events);
  _PLogPHC = PLogDefaultPHC;
  _PLogPHD = PLogDefaultPHD;
  ierr     = PLogSet(PLogDefaultPLBAll,PLogDefaultPLEAll);CHKERRQ(ierr);
  /* all processors sync here for more consistent logging */
  ierr     = MPI_Barrier(PETSC_COMM_WORLD);CHKERRQ(ierr);
  PetscTime(BaseTime);
  PLogStagePush(0);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"PLogDestroy"
/*@C
   PLogDestroy - Destroys the object and event logging data and resets the 
   global counters. 

   Not Collective

   Notes:
   This routine should not usually be used by programmers. Instead employ 
   PLogStagePush() and PLogStagePop().

   Level: developer

.keywords: log, destroy

.seealso: PLogDump(), PLogAllBegin(), PLogPrintSummary(), PLogStagePush(), PlogStagePop()
@*/
int PLogDestroy(void)
{
  int ierr;

  PetscFunctionBegin;
  if (objects) {free(objects); objects = 0;}
  if (events)  {free(events); events = 0;}
  ierr    = PLogSet(0,0);CHKERRQ(ierr);

  /* Resetting phase */
  ierr = PetscMemzero(EventsType,sizeof(EventsType));CHKERRQ(ierr);
  ierr = PetscMemzero(ObjectsType,sizeof(ObjectsType));CHKERRQ(ierr);
  _TotalFlops      = 0.0;
  nobjects         = 0;
  nevents          = 0;
  ObjectsDestroyed = 0;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"PLogBegin"
/*@C
    PLogBegin - Turns on logging of objects and events. This logs flop
       rates and object creation and should not slow programs down too much.
       This routine may be called more than once.

    Collective over PETSC_COMM_WORLD

    Options Database Keys:
+   -log_summary - Prints summary of flop and timing information to the 
                   screen (for code compiled with PETSC_USE_LOG)
-   -log - Prints detailed log information (for code compiled with PETSC_USE_LOG)

    Usage:
.vb
      PetscInitialize(...);
      PLogBegin();
       ... code ...
      PLogPrintSummary(MPI_Comm,filename); or PLogDump(); 
      PetscFinalize();
.ve

    Notes:
      PLogPrintSummary(MPI_Comm,filename) or PLogDump() actually cause the printing of 
    the logging information.

    Level: advanced

.keywords: log, begin

.seealso: PLogDump(), PLogAllBegin(), PLogPrintSummary(), PLogTraceBegin()
@*/
int PLogBegin(void)
{
  int ierr;

  PetscFunctionBegin;
  objects  = (Objects*)malloc(CHUNCK*sizeof(Objects));CHKPTRQ(objects);
  events   = (Events*)malloc(CHUNCK*sizeof(Events));CHKPTRQ(events);
  _PLogPHC = PLogDefaultPHC;
  _PLogPHD = PLogDefaultPHD;
  ierr     = PLogSet(PLogDefaultPLB,PLogDefaultPLE);CHKERRQ(ierr);
  /* all processors sync here for more consistent logging */
  ierr     = MPI_Barrier(PETSC_COMM_WORLD);CHKERRQ(ierr);
  PetscTime(BaseTime);
  PLogStagePush(0);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"PLogTraceBegin"
/*@
    PLogTraceBegin - Activates trace logging.  Every time a PETSc event
    begins or ends, the event name is printed.

    Not Collective

    Input Parameter:
.   file - file to print trace in (e.g. stdout)

    Options Database Key:
.   -log_trace [filename] - Activates PLogTraceBegin()

    Notes:
    PLogTraceBegin() prints the processor number, the execution time (sec),
    then "Event begin:" or "Event end:" followed by the event name.

    PLogTraceBegin() allows tracing of all PETSc calls, which is useful
    to determine where a program is hanging without running in the 
    debugger.  Can be used in conjunction with the -log_info option. 

    Level: intermediate

.seealso: PLogDump(), PLogAllBegin(), PLogPrintSummary(), PLogBegin()
@*/
int PLogTraceBegin(FILE *file)
{
  int ierr;

  PetscFunctionBegin;
  ierr      = PLogSet(PLogDefaultPLBTrace,PLogDefaultPLETrace);CHKERRQ(ierr);
  tracefile = file;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"PLogDump"
/*@C
   PLogDump - Dumps logs of objects to a file. This file is intended to 
   be read by petsc/bin/petscview.

   Collective on PETSC_COMM_WORLD

   Input Parameter:
.  name - an optional file name

   Options Database Keys:
+  -log - Prints basic log information (for code compiled with PETSC_USE_LOG)
-  -log_all - Prints extensive log information (for code compiled with PETSC_USE_LOG)
   
   Usage:
.vb
     PetscInitialize(...);
     PLogBegin(); or PLogAllBegin(); 
     ... code ...
     PLogDump(filename);
     PetscFinalize();
.ve

   Notes:
   The default file name is 
$      Log.<rank>
   where <rank> is the processor number. If no name is specified, 
   this file will be used.

   Level: advanced

.keywords: log, dump

.seealso: PLogBegin(), PLogAllBegin(), PLogPrintSummary()
@*/
int PLogDump(const char sname[])
{
  int        i,rank,ierr;
  FILE       *fd;
  char       file[64],fname[64];
  PLogDouble flops,_TotalTime;
  
  PetscFunctionBegin;
  PetscTime(_TotalTime);
  _TotalTime -= BaseTime;

  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  if (sname) sprintf(file,"%s.%d",sname,rank);
  else  sprintf(file,"Log.%d",rank);
  ierr = PetscFixFilename(file,fname);CHKERRQ(ierr);
  fd   = fopen(fname,"w"); if (!fd) SETERRQ1(PETSC_ERR_FILE_OPEN,0,"cannot open file: %s",fname);

  ierr = PetscFPrintf(PETSC_COMM_WORLD,fd,"Objects created %d Destroyed %d\n",nobjects,ObjectsDestroyed);
  ierr = PetscFPrintf(PETSC_COMM_WORLD,fd,"Clock Resolution %g\n",0.0);
  ierr = PetscFPrintf(PETSC_COMM_WORLD,fd,"Events %d\n",nevents);
  for (i=0; i<nevents; i++) {
    ierr = PetscFPrintf(PETSC_COMM_WORLD,fd,"%g %d %d %d %d %d %d %g %g %g\n",events[i].time,
                              events[i].event,
                              events[i].cookie,events[i].type,events[i].id1,
                              events[i].id2,events[i].id3,
                              events[i].flops,events[i].mem,
                              events[i].maxmem);
  }
  for (i=0; i<nobjects; i++) {
    ierr = PetscFPrintf(PETSC_COMM_WORLD,fd,"%d %d\n",objects[i].parent,(int)objects[i].mem);
    if (!objects[i].string[0]) {ierr = PetscFPrintf(PETSC_COMM_WORLD,fd,"No Info\n");}
    else ierr = PetscFPrintf(PETSC_COMM_WORLD,fd,"%s\n",objects[i].string);
    if (!objects[i].name[0]) {ierr = PetscFPrintf(PETSC_COMM_WORLD,fd,"No Name\n");}
    else ierr = PetscFPrintf(PETSC_COMM_WORLD,fd,"%s\n",objects[i].name);
  }
  for (i=0; i<PLOG_USER_EVENT_HIGH; i++) {
    flops = 0.0;
    if (EventsType[0][i][TIME]){flops = EventsType[0][i][FLOPS]/EventsType[0][i][TIME];}
    ierr = PetscFPrintf(PETSC_COMM_WORLD,fd,"%d %16g %16g %16g %16g\n",i,EventsType[0][i][COUNT],
                      EventsType[0][i][FLOPS],EventsType[0][i][TIME],flops);
  }
  ierr = PetscFPrintf(PETSC_COMM_WORLD,fd,"Total Flops %14e %16.8e\n",_TotalFlops,_TotalTime);
  fclose(fd);
  PetscFunctionReturn(0);
}

extern char *PLogEventColor[];
extern int  PLogEventColorMalloced[];

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"PLogEventRegister"
/*@C
    PLogEventRegister - Registers an event name for logging operations in 
    an application code. 

    Not Collective

    Input Parameter:
+   string - name associated with the event
-   color - (optional) string specifying a color and display pattern
            for viewing an event, used by Upshot/Nupshot to view logs
            generated by -log_mpe (e.g., - "red:", "green:vlines3");
            use PETSC_NULL to let PETSc assign a color.
            
    Output Parameter:
.   e -  event id for use with PLogEventBegin() and PLogEventEnd().

    Example of Usage:
.vb
      int USER_EVENT;
      int user_event_flops;
      PLogEventRegister(&USER_EVENT,"User event name","EventColor");
      PLogEventBegin(USER_EVENT,0,0,0,0);
         [code segment to monitor]
         PLogFlops(user_event_flops);
      PLogEventEnd(USER_EVENT,0,0,0,0);
.ve

    Notes: 
    PETSc automatically logs library events if the code has been
    compiled with -DPETSC_USE_LOG (which is the default) and -log,
    -log_summary, or -log_all are specified.  PLogEventRegister() is
    intended for logging user events to supplement this PETSc
    information. 

    PETSc can gather data for use with the utilities Upshot/Nupshot
    (part of the MPICH distribution).  If PETSc has been compiled
    with flag -DPETSC_HAVE_MPE (MPE is an additional utility within
    MPICH), the user can employ another command line option, -log_mpe,
    to create a logfile, "mpe.log", which can be visualized
    Upshot/Nupshot. The color argument is used by this utility
    in forming the display of this event; the standard X-windows
    color names should be used.

    Level: intermediate

.keywords: log, event, register

.seealso: PLogEventBegin(), PLogEventEnd(), PLogFlops(),
          PLogEventMPEActivate(), PLogEventMPEDeactivate(),
          PLogEventActivate(), PLogEventDeactivate()
@*/
int PLogEventRegister(int *e,const char string[],const char color[])
{
  char *cstring;
  int  ierr;

  PetscFunctionBegin;
  *e = PLOG_USER_EVENT_LOW++;
  if (*e > PLOG_USER_EVENT_HIGH) { 
    *e = 0;
    SETERRQ(PETSC_ERR_PLIB,0,"Out of event IDs");
  }
  ierr = PetscStrallocpy(string,&cstring);CHKERRQ(ierr);
  PLogEventName[*e] = cstring;
#if defined(PETSC_HAVE_MPE)
  if (UseMPE) {
    int   rank;
    char* ccolor;

    PLogEventMPEFlags[*e]       = 1;
    if (color) {
     ierr = PetscStrallocpy(color,&ccolor);CHKERRQ(ierr);
      PLogEventColor[*e]         = ccolor;
      PLogEventColorMalloced[*e] = 1;
    }
    ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
    if (!rank) {
      MPE_Describe_state(MPEBEGIN+2*(*e),MPEBEGIN+2*(*e)+1,cstring,PLogEventColor[*e]);
    }
  }
#endif
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"PLogEventRegisterDestroy_Private"
/*
   PLogEventRegisterDestroy_Private - Destroy the memory allocated during calls to 
        PLogEventRegister().

*/
int PLogEventRegisterDestroy_Private(void)
{
  int i,ierr;
  
  PetscFunctionBegin;
  for (i=PLOG_USER_EVENT_LOW-1; i>=PLOG_USER_EVENT_LOW_STATIC; i--) {
    ierr = PetscFree(PLogEventName[i]);CHKERRQ(ierr);
#if defined(PETSC_HAVE_MPE)
    if (PLogEventColorMalloced[i]) {ierr = PetscFree(PLogEventColor[i]);CHKERRQ(ierr);}
#endif
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"PLogEventDeactivate"
/*@
   PLogEventDeactivate - Indicates that a particular event should not be
   logged. 

   Not Collective

   Input Parameter:
.  event - integer indicating event

   Usage:
.vb
      PLogEventDeactivate(VEC_SetValues);
        [code where you do not want to log VecSetValues()]
      PLogEventActivate(VEC_SetValues);
        [code where you do want to log VecSetValues()]
.ve 

    Note: 
    The event may be either a pre-defined PETSc event (found in
    include/petsclog.h) or an event number obtained with PLogEventRegister()).

    Level: advanced

.seealso: PLogEventMPEDeactivate(),PLogEventMPEActivate(),PlogEventActivate()
@*/
int PLogEventDeactivate(int event)
{
  PetscFunctionBegin;
  PLogEventFlags[event] = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"PLogEventActivate"
/*@
   PLogEventActivate - Indicates that a particular event should be logged.
   The event may be either a pre-defined PETSc event (found in 
   include/petsclog.h) or an event number obtained with PLogEventRegister().

   Not Collective

   Input Parameter:
.  event - integer indicating event

   Usage:
.vb
      PLogEventDeactivate(VEC_SetValues);
        [code where you do not want to log VecSetValues()]
      PLogEventActivate(VEC_SetValues);
        [code where you do want to log VecSetValues()]
.ve 

    Level: advanced

.seealso: PLogEventMPEDeactivate(),PLogEventMPEActivate(),PlogEventDeactivate()
@*/
int PLogEventActivate(int event)
{
  PetscFunctionBegin;
  PLogEventFlags[event] = PETSC_TRUE;
  PetscFunctionReturn(0);
}

PetscTruth PetscPreLoadingUsed = PETSC_FALSE;

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"PLogPrintSummary"
/*@C
   PLogPrintSummary - Prints a summary of the logging.

   Collective over MPI_Comm

   Input Parameter:
+  comm - MPI communicator (only one processor prints output)
-  file - an optional file name

   Options Database Keys:
.  -log_summary - Prints summary of log information (for code
   compiled with PETSC_USE_LOG)

   Usage:
.vb
     PetscInitialize(...);
     PLogBegin();
     ... code ...
     PLogPrintSummary(MPI_Comm,filename);
     PetscFinalize(...);
.ve

   Notes:
   By default the summary is printed to stdout.
   More extensive examination of the log information can be done with 
   PLogDump(), which is activated by the option -log or -log_all.

   Level: beginner
   
.keywords: log, dump, print

.seealso: PLogBegin(), PLogDump()
@*/
int PLogPrintSummary(MPI_Comm comm,const char filename[])
{
  PLogDouble maxo,mino,aveo,mem,totmem,maxmem,minmem,mlensmcounts;
  PLogDouble maxf,minf,avef,totf,_TotalTime,maxt,mint,avet,tott,ratio;
  PLogDouble fmin,fmax,ftot,wdou,totts,totff,rat,sstime,sflops,ratf;
  PLogDouble ptotts,ptotff,ptotts_stime,ptotff_sflops,rat1,rat2,rat3;
  PLogDouble minm,maxm,avem,totm,minr,maxr,maxml,minml,totml,aveml,totr;
  PLogDouble rp,mp,lp,rpg,mpg,lpg,totms,totmls,totrs,mps,lps,rps,lpmp;
  PLogDouble pstime,psflops1,psflops,flopr,mict,mact,rct,x,y;
  int        size,rank,i,j,ierr,lEventsStageMax;
  char       arch[10],hostname[64],username[16],pname[256],date[64];
  FILE       *fd = stdout;

  PetscFunctionBegin;
  /* pop off any stages the user forgot to remove */
  while (EventsStagePushed) PLogStagePop();

  PetscTime(_TotalTime);  _TotalTime -= BaseTime;
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);

  /* Open the summary file */
  ierr = PetscFOpen(comm,filename,"w",&fd);CHKERRQ(ierr);

  ierr = PetscFPrintf(comm,fd,"************************************************************************************************************************\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(comm,fd,"***             WIDEN YOUR WINDOW TO 120 CHARACTERS.  Use 'enscript -r -fCourier9' to print this document            ***\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(comm,fd,"************************************************************************************************************************\n");CHKERRQ(ierr);

  ierr = PetscFPrintf(comm,fd,"\n---------------------------------------------- PETSc Performance Summary: ----------------------------------------------\n\n");CHKERRQ(ierr);
  ierr = PetscGetArchType(arch,10);CHKERRQ(ierr);
  ierr = PetscGetHostName(hostname,64);CHKERRQ(ierr);
  ierr = PetscGetUserName(username,16);CHKERRQ(ierr);
  ierr = PetscGetProgramName(pname,256);CHKERRQ(ierr);
  ierr = PetscGetDate(date,64);CHKERRQ(ierr);
  if (size == 1) {
    ierr = PetscFPrintf(comm,fd,"%s on a %s named %s with %d processor, by %s %s\n",
                 pname,arch,hostname,size,username,date);CHKERRQ(ierr);
  } else {
    ierr = PetscFPrintf(comm,fd,"%s on a %s named %s with %d processors, by %s %s\n",
                 pname,arch,hostname,size,username,date);CHKERRQ(ierr);
  }
  ierr = PetscFPrintf(comm,fd,"Using %s\n",PETSC_VERSION_NUMBER);CHKERRQ(ierr);


  wdou = _TotalFlops; 
  ierr = MPI_Allreduce(&wdou,&minf,1,MPIU_PLOGDOUBLE,MPI_MIN,comm);CHKERRQ(ierr);
  ierr = MPI_Allreduce(&wdou,&maxf,1,MPIU_PLOGDOUBLE,MPI_MAX,comm);CHKERRQ(ierr);
  ierr = MPI_Allreduce(&wdou,&totf,1,MPIU_PLOGDOUBLE,MPI_SUM,comm);CHKERRQ(ierr);
  avef = (totf)/((PLogDouble) size);
  wdou = nobjects;
  ierr = MPI_Allreduce(&wdou,&mino,1,MPIU_PLOGDOUBLE,MPI_MIN,comm);CHKERRQ(ierr);
  ierr = MPI_Allreduce(&wdou,&maxo,1,MPIU_PLOGDOUBLE,MPI_MAX,comm);CHKERRQ(ierr);
  ierr = MPI_Allreduce(&wdou,&aveo,1,MPIU_PLOGDOUBLE,MPI_SUM,comm);CHKERRQ(ierr);
  aveo = (aveo)/((PLogDouble) size);
  wdou = _TotalTime;
  ierr = MPI_Allreduce(&wdou,&mint,1,MPIU_PLOGDOUBLE,MPI_MIN,comm);CHKERRQ(ierr);
  ierr = MPI_Allreduce(&wdou,&maxt,1,MPIU_PLOGDOUBLE,MPI_MAX,comm);CHKERRQ(ierr);
  ierr = MPI_Allreduce(&wdou,&tott,1,MPIU_PLOGDOUBLE,MPI_SUM,comm);CHKERRQ(ierr);
  avet = (tott)/((PLogDouble) size);

  ierr = PetscFPrintf(comm,fd,"\n                         Max       Max/Min      Avg      Total \n");CHKERRQ(ierr);
  if (mint) ratio = maxt/mint; else ratio = 0.0;
  ierr = PetscFPrintf(comm,fd,"Time (sec):           %5.3e   %10.5f   %5.3e\n",maxt,ratio,avet);CHKERRQ(ierr);
  if (mino) ratio = maxo/mino; else ratio = 0.0;
  ierr = PetscFPrintf(comm,fd,"Objects:              %5.3e   %10.5f   %5.3e\n",maxo,ratio,aveo);CHKERRQ(ierr);
  if (minf) ratio = maxf/minf; else ratio = 0.0;
  ierr = PetscFPrintf(comm,fd,"Flops:                %5.3e   %10.5f   %5.3e  %5.3e\n",maxf,ratio,avef,totf);CHKERRQ(ierr);

  if (mint) fmin = minf/mint; else fmin = 0;
  if (maxt) fmax = maxf/maxt; else fmax = 0;
  if (maxt) ftot = totf/maxt; else ftot = 0;
  if (fmin) ratio = fmax/fmin; else ratio = 0.0;
  ierr = PetscFPrintf(comm,fd,"Flops/sec:            %5.3e   %10.5f              %5.3e\n",fmax,ratio,ftot);CHKERRQ(ierr);
  ierr = PetscTrSpace(PETSC_NULL,PETSC_NULL,&mem);CHKERRQ(ierr);
  if (mem > 0.0) {
    ierr = MPI_Allreduce(&mem,&maxmem,1,MPIU_PLOGDOUBLE,MPI_MAX,comm);CHKERRQ(ierr);
    ierr = MPI_Allreduce(&mem,&minmem,1,MPIU_PLOGDOUBLE,MPI_MIN,comm);CHKERRQ(ierr);
    ierr = MPI_Allreduce(&mem,&totmem,1,MPIU_PLOGDOUBLE,MPI_SUM,comm);CHKERRQ(ierr);
    if (minmem) ratio = maxmem/minmem; else ratio = 0.0;
    ierr = PetscFPrintf(comm,fd,"Memory:               %5.3e   %8.3f              %5.3e\n",maxmem,ratio,totmem);CHKERRQ(ierr);
  }
  wdou = .5*(irecv_ct + isend_ct + recv_ct + send_ct);
  ierr = MPI_Allreduce(&wdou,&minm,1,MPIU_PLOGDOUBLE,MPI_MIN,comm);CHKERRQ(ierr);
  ierr = MPI_Allreduce(&wdou,&maxm,1,MPIU_PLOGDOUBLE,MPI_MAX,comm);CHKERRQ(ierr);
  ierr = MPI_Allreduce(&wdou,&totm,1,MPIU_PLOGDOUBLE,MPI_SUM,comm);CHKERRQ(ierr);
  avem = (totm)/((PLogDouble) size);
  wdou = .5*(irecv_len + isend_len + recv_len + send_len);
  ierr = MPI_Allreduce(&wdou,&minml,1,MPIU_PLOGDOUBLE,MPI_MIN,comm);CHKERRQ(ierr);
  ierr = MPI_Allreduce(&wdou,&maxml,1,MPIU_PLOGDOUBLE,MPI_MAX,comm);CHKERRQ(ierr);
  ierr = MPI_Allreduce(&wdou,&totml,1,MPIU_PLOGDOUBLE,MPI_SUM,comm);CHKERRQ(ierr);
  if (totm) aveml = (totml)/(totm); else aveml = 0;
  if (minm) ratio = maxm/minm; else ratio = 0.0;
  ierr = PetscFPrintf(comm,fd,"MPI Messages:         %5.3e   %8.3f   %5.3e  %5.3e\n",maxm,ratio,avem,totm);CHKERRQ(ierr);
  if (minml) ratio = maxml/minml; else ratio = 0.0;
  ierr = PetscFPrintf(comm,fd,"MPI Message Lengths:  %5.3e   %8.3f   %5.3e  %5.3e\n",maxml,ratio,aveml,totml);CHKERRQ(ierr);
  ierr = MPI_Allreduce(&allreduce_ct,&minr,1,MPIU_PLOGDOUBLE,MPI_MIN,comm);CHKERRQ(ierr);
  ierr = MPI_Allreduce(&allreduce_ct,&maxr,1,MPIU_PLOGDOUBLE,MPI_MAX,comm);CHKERRQ(ierr);
  ierr = MPI_Allreduce(&allreduce_ct,&totr,1,MPIU_PLOGDOUBLE,MPI_SUM,comm);CHKERRQ(ierr);
  if (minr) ratio = maxr/minr; else ratio = 0.0;
  ierr = PetscFPrintf(comm,fd,"MPI Reductions:       %5.3e   %8.3f\n",maxr,ratio);CHKERRQ(ierr);
  ierr = PetscFPrintf(comm,fd,"\nFlop counting convention: 1 flop = 1 real number operation of type (multiply/divide/add/subtract)\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(comm,fd,"                            e.g., VecAXPY() for real vectors of length N --> 2N flops\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(comm,fd,"                            and VecAXPY() for complex vectors of length N --> 8N flops\n");CHKERRQ(ierr);

  ierr = MPI_Allreduce(&EventsStageMax,&lEventsStageMax,1,MPI_INT,MPI_MAX,comm);CHKERRQ(ierr);
  if (lEventsStageMax) {
    PLogDouble mcounts,mlens,rcounts;

    ierr = PetscFPrintf(comm,fd,"\nSummary of Stages:  ---- Time ------     ----- Flops -------    -- Messages -- -- Message-lengths -- Reductions --\n");CHKERRQ(ierr);
    ierr = PetscFPrintf(comm,fd,"                      Avg      %%Total        Avg       %%Total   counts   %%Total    avg      %%Total   counts  %%Total \n");CHKERRQ(ierr);
    for (j=0; j<=lEventsStageMax; j++) {
      if (!PLogStagePrintFlag[j]) continue;

      ierr = MPI_Allreduce(&EventsStageFlops[j],&sflops,1,MPIU_PLOGDOUBLE,MPI_SUM,comm);CHKERRQ(ierr);
      ierr = MPI_Allreduce(&EventsStageTime[j],&sstime,1,MPIU_PLOGDOUBLE,MPI_SUM,comm);CHKERRQ(ierr);
      if (tott)   pstime = 100.0*sstime/tott; else pstime = 0.0;if (pstime >= 99.9) pstime = 99.9;
      if (totf)   psflops = 100.*sflops/totf; else psflops = 0.0; 
      if (sstime) psflops1 = (size*sflops)/sstime; else psflops1 = 0.0;

      ierr = MPI_Allreduce(&EventsStageMessageCounts[j],&mcounts,1,MPIU_PLOGDOUBLE,MPI_SUM,comm);CHKERRQ(ierr);
      ierr = MPI_Allreduce(&EventsStageMessageLengths[j],&mlens,1,MPIU_PLOGDOUBLE,MPI_SUM,comm);CHKERRQ(ierr);
      ierr = MPI_Allreduce(&EventsStageReductions[j],&rcounts,1,MPIU_PLOGDOUBLE,MPI_SUM,comm);CHKERRQ(ierr);
      mcounts = .5*mcounts; mlens = .5*mlens; rcounts = rcounts/size;
      if (totm)  rat1 = 100.*mcounts/totm; else rat1 = 0.0; if (rat1 >= 99.9) rat1 = 99.9;
      if (totml) rat2 = 100.*mlens/totml; else rat2 = 0.0;  if (rat2 >= 99.9) rat2 = 99.9;
      if (totr)  rat3 = 100.*size*rcounts/totr; else rat3 = 0.0;if (rat3 >= 99.9) rat3 = 99.9;
      if (mcounts) mlensmcounts = mlens/mcounts; else mlensmcounts = 0.0; 
      if (EventsStageName[j]) {
        ierr = PetscFPrintf(comm,fd," %d: %15s: %6.4e    %4.1f%%     %6.4e      %4.1f%%  %5.3e   %4.1f%%  %5.3e  %4.1f%%  %5.3e  %4.1f%% \n",
                j,EventsStageName[j],sstime/size,pstime,psflops1,psflops,mcounts,rat1,mlensmcounts,rat2,
                rcounts,rat3);CHKERRQ(ierr);
      } else {
        ierr = PetscFPrintf(comm,fd," %d:                 %6.4e    %4.1f%%     %6.4e      %4.1f%%  %5.3e   %4.1f%%  %5.3e  %4.1f%%  %5.3e  %4.1f%% \n",
                j,sstime/size,pstime,psflops1,psflops,mcounts,rat1,mlensmcounts,rat2,rcounts,rat3);CHKERRQ(ierr);
      }
    }
  }

  ierr = PetscFPrintf(comm,fd, 
    "\n------------------------------------------------------------------------------------------------------------------------\n");CHKERRQ(ierr);  
  ierr = PetscFPrintf(comm,fd,"See the 'Profiling' chapter of the users' manual for details on interpreting output.\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(comm,fd,"Phase summary info:\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(comm,fd,"   Count: number of times phase was executed\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(comm,fd,"   Time and Flops/sec: Max - maximum over all processors\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(comm,fd,"                       Ratio - ratio of maximum to minimum over all processors\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(comm,fd,"   Mess: number of messages sent over all processors\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(comm,fd,"   Avg. len: average message length over all processors\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(comm,fd,"   Reduct: number of global reductions\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(comm,fd,"   Global: entire computation\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(comm,fd,"   Stage: optional user-defined stages of a computation. Set stages with PLogStagePush() and PLogStagePop().\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(comm,fd,"      %%T - percent time in this phase         %%F - percent flops in this phase\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(comm,fd,"      %%M - percent messages in this phase     %%L - percent message lengths in this phase\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(comm,fd,"      %%R - percent reductions in this phase\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(comm,fd,"   Total Mflop/s: 10e-6 * (sum of flops over all processors)/(max time over all processors)\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(comm,fd,
    "------------------------------------------------------------------------------------------------------------------------\n");CHKERRQ(ierr);

#if defined(PETSC_USE_BOPT_g)
  ierr = PetscFPrintf(comm,fd,"\n\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(comm,fd,"      ##########################################################\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(comm,fd,"      #                                                        #\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(comm,fd,"      #                          WARNING!!!                    #\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(comm,fd,"      #                                                        #\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(comm,fd,"      #   This code was compiled with a debugging option,      #\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(comm,fd,"      #   BOPT=<g,g_c++,g_complex>.   To get timing results    #\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(comm,fd,"      #   ALWAYS compile your code with an optimized version,  #\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(comm,fd,"      #   BOPT=<O,O_c++,O_complex>;  the performance will      #\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(comm,fd,"      #   be generally two or three times faster.              #\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(comm,fd,"      #                                                        #\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(comm,fd,"      ##########################################################\n\n\n");CHKERRQ(ierr);
#endif
#if defined(PETSC_USE_COMPLEX) && !defined(PETSC_USE_FORTRAN_KERNELS)
  ierr = PetscFPrintf(comm,fd,"\n\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(comm,fd,"      ##########################################################\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(comm,fd,"      #                                                        #\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(comm,fd,"      #                          WARNING!!!                    #\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(comm,fd,"      #                                                        #\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(comm,fd,"      #   The code for various complex numbers numerical       #\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(comm,fd,"      #   kernels uses C++, which generally is not well        #\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(comm,fd,"      #   optimized.  For performance that is about 4-5 times  #\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(comm,fd,"      #   faster, specify the flag -DPETSC_USE_FORTRAN_KERNELS in    #\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(comm,fd,"      #   base_variables and recompile the PETSc libraries.    #\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(comm,fd,"      #                                                        #\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(comm,fd,"      ##########################################################\n\n\n");CHKERRQ(ierr);
#endif

  if (!PetscPreLoadingUsed) {
    ierr = PetscFPrintf(comm,fd,"\n\n");CHKERRQ(ierr);
    ierr = PetscFPrintf(comm,fd,"      ##########################################################\n");CHKERRQ(ierr);
    ierr = PetscFPrintf(comm,fd,"      #                                                        #\n");CHKERRQ(ierr);
    ierr = PetscFPrintf(comm,fd,"      #                          WARNING!!!                    #\n");CHKERRQ(ierr);
    ierr = PetscFPrintf(comm,fd,"      #                                                        #\n");CHKERRQ(ierr);
    ierr = PetscFPrintf(comm,fd,"      #   This code was run with the PreLoadinBegin() macros   #\n");CHKERRQ(ierr);
    ierr = PetscFPrintf(comm,fd,"      #   To get timing results we always recommend preloading #\n");CHKERRQ(ierr);
    ierr = PetscFPrintf(comm,fd,"      #   otherwise timing numbers may be meaningless.         #\n");CHKERRQ(ierr);
    ierr = PetscFPrintf(comm,fd,"      ##########################################################\n\n\n");CHKERRQ(ierr);
  }

  /* loop over operations looking for interesting ones */
  ierr = PetscFPrintf(comm,fd,"Phase                  Count      Time (sec)        Flops/sec \
                          --- Global ---  --- Stage ---   Total\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(comm,fd,"                    Max  Ratio  Max     Ratio      Max     Ratio\
  Mess  Avg len  Reduct %%T %%F %%M %%L %%R  %%T %%F %%M %%L %%R Mflop/s\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(comm,fd,
    "------------------------------------------------------------------------------------------------------------------------\n");CHKERRQ(ierr); 
  for (j=0; j<=EventsStageMax; j++) {
    if (!PLogStagePrintFlag[j]) continue;
    ierr = MPI_Allreduce(&EventsStageFlops[j],&sflops,1,MPIU_PLOGDOUBLE,MPI_SUM,comm);CHKERRQ(ierr);
    ierr = MPI_Allreduce(&EventsStageTime[j],&sstime,1,MPIU_PLOGDOUBLE,MPI_SUM,comm);CHKERRQ(ierr);
    if (EventsStageMax) {
      if (EventsStageName[j]) {
        ierr = PetscFPrintf(comm,fd,"\n--- Event Stage %d: %s\n\n",j,EventsStageName[j]);CHKERRQ(ierr);
      } else {
        ierr = PetscFPrintf(comm,fd,"\n--- Event Stage %d:\n\n",j);CHKERRQ(ierr);
      }
    }
    ierr = MPI_Allreduce(&EventsStageMessageCounts[j],&totms,1,MPIU_PLOGDOUBLE,MPI_SUM,comm);CHKERRQ(ierr);
    ierr = MPI_Allreduce(&EventsStageMessageLengths[j],&totmls,1,MPIU_PLOGDOUBLE,MPI_SUM,comm);CHKERRQ(ierr);
    ierr = MPI_Allreduce(&EventsStageReductions[j],&totrs,1,MPIU_PLOGDOUBLE,MPI_SUM,comm);CHKERRQ(ierr);
    /* This loop assumes that PLOG_USER_EVENT_HIGH is the max event number */
    for (i=0; i<PLOG_USER_EVENT_HIGH; i++) {  
      if (EventsType[j][i][TIME]) {
        wdou = EventsType[j][i][FLOPS]/EventsType[j][i][TIME];
      }
      else wdou = 0.0;
      ierr = MPI_Allreduce(&wdou,&minf,1,MPIU_PLOGDOUBLE,MPI_MIN,comm);CHKERRQ(ierr);
      ierr = MPI_Allreduce(&wdou,&maxf,1,MPIU_PLOGDOUBLE,MPI_MAX,comm);CHKERRQ(ierr);
      ierr = MPI_Allreduce(&EventsType[j][i][FLOPS],&totff,1,MPIU_PLOGDOUBLE,MPI_SUM,comm);CHKERRQ(ierr);
      ierr = MPI_Allreduce(&EventsType[j][i][TIME],&mint,1,MPIU_PLOGDOUBLE,MPI_MIN,comm);CHKERRQ(ierr);
      ierr = MPI_Allreduce(&EventsType[j][i][TIME],&maxt,1,MPIU_PLOGDOUBLE,MPI_MAX,comm);CHKERRQ(ierr);
      ierr = MPI_Allreduce(&EventsType[j][i][TIME],&totts,1,MPIU_PLOGDOUBLE,MPI_SUM,comm);CHKERRQ(ierr);
      ierr = MPI_Allreduce(&EventsType[j][i][MESSAGES],&mp,1,MPIU_PLOGDOUBLE,MPI_SUM,comm);CHKERRQ(ierr);
      ierr = MPI_Allreduce(&EventsType[j][i][LENGTHS],&lp,1,MPIU_PLOGDOUBLE,MPI_SUM,comm);CHKERRQ(ierr);
      ierr = MPI_Allreduce(&EventsType[j][i][REDUCTIONS],&rp,1,MPIU_PLOGDOUBLE,MPI_SUM,comm);CHKERRQ(ierr);

      ierr = MPI_Allreduce(&EventsType[j][i][COUNT],&mict,1,MPIU_PLOGDOUBLE,MPI_MIN,comm);CHKERRQ(ierr);
      ierr = MPI_Allreduce(&EventsType[j][i][COUNT],&mact,1,MPIU_PLOGDOUBLE,MPI_MIN,comm);CHKERRQ(ierr);
     
      if (maxt) flopr = totff/maxt; else flopr = 0.0;

      if (EventsType[j][i][COUNT]) {
        if (mint)   rat = maxt/mint; else rat = 0.0;
        if (minf)   ratf = maxf/minf; else ratf = 0.0;
        if (tott)   ptotts = 100.*totts/tott; else ptotts = 0.0;if (ptotts >= 99.) ptotts = 99.;
        if (totf)   ptotff = 100.*totff/totf; else ptotff = 0.0;if (ptotff >= 99.) ptotff = 99.;
        if (sstime) ptotts_stime = 100.*totts/sstime; else  ptotts_stime = 0.0;if (ptotts_stime >= 99.) ptotts_stime = 99.;
        if (sflops) ptotff_sflops = 100.*totff/sflops; else ptotff_sflops = 0.0;if (ptotff_sflops >= 99.) ptotff_sflops = 99.;
        if (totm)   mpg  = 100.*mp/totm; else mpg = 0.0; if (mpg >= 99.) mpg = 99.;
        if (totml)  lpg  = 100.*lp/totml; else lpg = 0.0; if (lpg >= 99.) lpg = 99.;
        if (totr)   rpg  = 100.*rp/totr; else rpg = 0.0;if (rpg >= 99.) rpg = 99.;
        if (totms)  mps  = 100.*mp/totms; else mps = 0.0; if (mps >= 99.) mps = 99.;
        if (totmls) lps  = 100.*lp/totmls; else lps = 0.0; if (lps >= 99.) lps = 99.;
        if (totrs)  rps  = 100.*rp/totrs; else rps = 0.0;if (rps >= 99.) rps = 99.;
        if (mp)     lpmp = lp/mp; else lpmp = 0.0;
        if (mict)   rct  = mact/mict; else rct = 0.0;
        mp = mp/2.0;
        rp = rp/((PLogDouble) size);
        ierr = PetscFPrintf(comm,fd,"%-16s %7d %3.1f  %5.4e %5.1f  %3.2e %6.1f %2.1e %2.1e %2.1e %2.0f %2.0f %2.0f %2.0f %2.0f  %2.0f %2.0f %2.0f %2.0f %2.0f %5.0f\n",
                    PLogEventName[i],(int)mact,rct,maxt,rat,maxf,ratf,
                    mp,lpmp,rp,ptotts,ptotff,mpg,lpg,rpg,ptotts_stime,ptotff_sflops,mps,lps,rps,flopr/1.e6);CHKERRQ(ierr);
      }
    }
    /* print effective bandwidth in vector scatters */
    if (EventsType[j][VEC_ScatterBarrier][COUNT]) {
      ierr = MPI_Allreduce(&EventsType[j][i][MESSAGES],&mp,1,MPIU_PLOGDOUBLE,MPI_SUM,comm);CHKERRQ(ierr);
      ierr = MPI_Allreduce(&EventsType[j][i][LENGTHS],&lp,1,MPIU_PLOGDOUBLE,MPI_SUM,comm);CHKERRQ(ierr);
    }

  }

  ierr = PetscFPrintf(comm,fd,
    "------------------------------------------------------------------------------------------------------------------------\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(comm,fd,"\n");CHKERRQ(ierr); 
  ierr = PetscFPrintf(comm,fd,"Memory usage is given in bytes:\n\n");CHKERRQ(ierr);

  /* loop over objects looking for interesting ones */
  ierr = PetscFPrintf(comm,fd,"Object Type      Creations   Destructions   Memory  Descendants' Mem.\n");CHKERRQ(ierr);
  for (j=0; j<=EventsStageMax; j++) {
    if (EventsStageMax) {
      if (EventsStageName[j]) {
        ierr = PetscFPrintf(comm,fd,"\n--- Event Stage %d: %s\n\n",j,EventsStageName[j]);CHKERRQ(ierr);
      } else {
        ierr = PetscFPrintf(comm,fd,"\n--- Event Stage %d:\n\n",j);CHKERRQ(ierr);
      }
    }
    for (i=0; i<PETSC_MAX_COOKIES; i++) {
      if (ObjectsType[j][i][0]) {
        ierr = PetscFPrintf(comm,fd,"%s %5d          %5d  %9d     %g\n",oname[i],(int)
            ObjectsType[j][i][0],(int)ObjectsType[j][i][1],(int)ObjectsType[j][i][2],
            ObjectsType[j][i][3]);CHKERRQ(ierr);
      }
    }
  }

  ierr = PetscFPrintf(comm,fd,"===========================================================\n");CHKERRQ(ierr);
  PetscTime(y); 
  PetscTime(x);
  PetscTime(y); 
  PetscTime(y);
  PetscTime(y);
  PetscTime(y);
  PetscTime(y);
  PetscTime(y); 
  PetscTime(y);
  PetscTime(y);
  PetscTime(y);
  PetscTime(y);
  ierr = PetscFPrintf(comm,fd,"Average time to get PetscTime(): %g\n",(y-x)/10.0);CHKERRQ(ierr);
  if (size > 1) {
    int        tag;
    MPI_Status status;

    ierr = MPI_Barrier(comm);CHKERRQ(ierr);
    PetscTime(x);
    ierr = MPI_Barrier(comm);CHKERRQ(ierr);
    ierr = MPI_Barrier(comm);CHKERRQ(ierr);
    ierr = MPI_Barrier(comm);CHKERRQ(ierr);
    ierr = MPI_Barrier(comm);CHKERRQ(ierr);
    ierr = MPI_Barrier(comm);CHKERRQ(ierr);
    PetscTime(y);
    ierr = PetscFPrintf(comm,fd,"Average time for MPI_Barrier(): %g\n",(y-x)/5.0);CHKERRQ(ierr);
    ierr = PetscCommGetNewTag(comm,&tag);CHKERRQ(ierr);
    ierr = MPI_Barrier(comm);CHKERRQ(ierr);
    if (rank) {
      ierr = MPI_Recv(0,0,MPI_INT,rank-1,tag,comm,&status);CHKERRQ(ierr);
      ierr = MPI_Send(0,0,MPI_INT,(rank+1)%size,tag,comm);CHKERRQ(ierr);
    } else {
      PetscTime(x);
      ierr = MPI_Send(0,0,MPI_INT,1,tag,comm);CHKERRQ(ierr);
      ierr = MPI_Recv(0,0,MPI_INT,size-1,tag,comm,&status);CHKERRQ(ierr);
      PetscTime(y);
      ierr = PetscFPrintf(comm,fd,"Average time for zero size MPI_Send(): %g\n",(y-x)/size);CHKERRQ(ierr);
    }
  }

#if defined(PETSC_USE_FORTRAN_KERNELS)
  ierr = PetscFPrintf(comm,fd,"Compiled without FORTRAN kernels\n");CHKERRQ(ierr);
#else
  ierr = PetscFPrintf(comm,fd,"Compiled without FORTRAN kernels\n");CHKERRQ(ierr);
#endif
#if defined(PETSC_USE_MAT_SINGLE)
  ierr = PetscFPrintf(comm,fd,"Compiled with single precision matrices\n");CHKERRQ(ierr);
#else
  ierr = PetscFPrintf(comm,fd,"Compiled with double precision matrices (default)\n");CHKERRQ(ierr);
#endif
  ierr = PetscFPrintf(comm,fd,"sizeof(short) %d sizeof(int) %d sizeof(long) %d sizeof(void*)%d",sizeof(short),sizeof(int),sizeof(long),sizeof(void*));CHKERRQ(ierr);
  ierr = PetscFPrintf(comm,fd,"%s",petscmachineinfo);CHKERRQ(ierr);


  ierr = PetscFPrintf(comm,fd,"\n");CHKERRQ(ierr);
  ierr = PetscFClose(comm,fd);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"PetscGetFlops"
/*@C
   PetscGetFlops - Returns the number of flops used on this processor 
   since the program began. 

   Not Collective

   Output Parameter:
   flops - number of floating point operations 

   Notes:
   A global counter logs all PETSc flop counts.  The user can use
   PLogFlops() to increment this counter to include flops for the 
   application code.  

   PETSc automatically logs library events if the code has been
   compiled with -DPETSC_USE_LOG (which is the default), and -log,
   -log_summary, or -log_all are specified.  PLogFlops() is
   intended for logging user flops to supplement this PETSc
   information.

   Level: intermediate

.keywords: log, flops, floating point operations

.seealso: PetscGetTime(), PLogFlops()
@*/
int PetscGetFlops(PLogDouble *flops)
{
  PetscFunctionBegin;
  *flops = _TotalFlops;
  PetscFunctionReturn(0);
}

/* --------- Activate version -------------  */

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"PLogEventActivateClass"
/*@
   PLogEventActivateClass - Activates event logging for a PETSc object class.

   Not Collective

   Input Parameter:
.  cookie - for example MAT_COOKIE, SNES_COOKIE,

   Level: developer

.seealso: PLogInfoActivate(),PLogInfo(),PLogInfoAllow(),PLogEventDeactivateClass(),
          PLogEventActivate(),PLogEventDeactivate()
@*/
int PLogEventActivateClass(int cookie)
{
  PetscFunctionBegin;
  if (cookie == SNES_COOKIE) {
    PLogEventActivate(SNES_Solve);
    PLogEventActivate(SNES_LineSearch);
    PLogEventActivate(SNES_FunctionEval);
    PLogEventActivate(SNES_JacobianEval);
    PLogEventActivate(SNES_MinimizationFunctionEval);
    PLogEventActivate(SNES_GradientEval);
    PLogEventActivate(SNES_HessianEval);
  } else if (cookie == SLES_COOKIE || cookie == PC_COOKIE || cookie == KSP_COOKIE) {
    PLogEventActivate(SLES_Solve);
    PLogEventActivate(SLES_SetUp);
    PLogEventActivate(KSP_GMRESOrthogonalization);
    PLogEventActivate(PC_ModifySubMatrices);
    PLogEventActivate(PC_SetUp);
    PLogEventActivate(PC_SetUpOnBlocks);
    PLogEventActivate(PC_Apply);
    PLogEventActivate(PC_ApplySymmetricLeft);
    PLogEventActivate(PC_ApplySymmetricRight);
  } else if (cookie == MAT_COOKIE) {
    PLogEventActivate(MAT_Mult);
    PLogEventActivate(MAT_MatrixFreeMult);
    PLogEventActivate(MAT_AssemblyBegin);
    PLogEventActivate(MAT_AssemblyEnd);
    PLogEventActivate(MAT_GetOrdering);
    PLogEventActivate(MAT_MultTranspose);
    PLogEventActivate(MAT_MultAdd);
    PLogEventActivate(MAT_MultTransposeAdd);
    PLogEventActivate(MAT_LUFactor);
    PLogEventActivate(MAT_CholeskyFactor);
    PLogEventActivate(MAT_LUFactorSymbolic);
    PLogEventActivate(MAT_ILUFactorSymbolic);
    PLogEventActivate(MAT_CholeskyFactorSymbolic);
    PLogEventActivate(MAT_IncompleteCholeskyFactorSymbolic);
    PLogEventActivate(MAT_LUFactorNumeric);
    PLogEventActivate(MAT_CholeskyFactorNumeric);
    PLogEventActivate(MAT_CholeskyFactorNumeric);
    PLogEventActivate(MAT_Relax);
    PLogEventActivate(MAT_Copy);
    PLogEventActivate(MAT_Convert);
    PLogEventActivate(MAT_Scale);
    PLogEventActivate(MAT_ZeroEntries);
    PLogEventActivate(MAT_Solve);
    PLogEventActivate(MAT_SolveAdd);
    PLogEventActivate(MAT_SolveTranspose);
    PLogEventActivate(MAT_SolveTransposeAdd);
    PLogEventActivate(MAT_SetValues);
    PLogEventActivate(MAT_ForwardSolve);
    PLogEventActivate(MAT_BackwardSolve);
    PLogEventActivate(MAT_Load);
    PLogEventActivate(MAT_View);
    PLogEventActivate(MAT_ILUFactor);

    PLogEventActivate(MAT_GetValues);
    PLogEventActivate(MAT_IncreaseOverlap);
    PLogEventActivate(MAT_GetRow);
  } else if (cookie == VEC_COOKIE) {
    PLogEventActivate(VEC_Dot);
    PLogEventActivate(VEC_Norm);
    PLogEventActivate(VEC_Max);
    PLogEventActivate(VEC_Min);
    PLogEventActivate(VEC_TDot);
    PLogEventActivate(VEC_Scale);
    PLogEventActivate(VEC_Copy);
    PLogEventActivate(VEC_Set);
    PLogEventActivate(VEC_AXPY);
    PLogEventActivate(VEC_AYPX);
    PLogEventActivate(VEC_Swap);
    PLogEventActivate(VEC_WAXPY);
    PLogEventActivate(VEC_AssemblyBegin);
    PLogEventActivate(VEC_AssemblyEnd);
    PLogEventActivate(VEC_MTDot);
    PLogEventActivate(VEC_MDot);
    PLogEventActivate(VEC_MAXPY);
    PLogEventActivate(VEC_PMult);
    PLogEventActivate(VEC_SetValues);
    PLogEventActivate(VEC_Load);
    PLogEventActivate(VEC_View);
    PLogEventActivate(VEC_ScatterBegin);
    PLogEventActivate(VEC_ScatterEnd);
    PLogEventActivate(VEC_SetRandom);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"PLogEventDeactivateClass"
/*@
   PLogEventDeactivateClass - Deactivates event logging for a PETSc object class.

   Not Collective

   Input Parameter:
.  cookie - for example MAT_COOKIE, SNES_COOKIE,

   Level: developer

.seealso: PLogInfoActivate(),PLogInfo(),PLogInfoAllow(),PLogEventActivateClass(),
          PLogEventActivate(),PLogEventDeactivate()
@*/
int PLogEventDeactivateClass(int cookie)
{
  PetscFunctionBegin;
  if (cookie == SNES_COOKIE) {
    PLogEventDeactivate(SNES_Solve);
    PLogEventDeactivate(SNES_LineSearch);
    PLogEventDeactivate(SNES_FunctionEval);
    PLogEventDeactivate(SNES_JacobianEval);
    PLogEventDeactivate(SNES_MinimizationFunctionEval);
    PLogEventDeactivate(SNES_GradientEval);
    PLogEventDeactivate(SNES_HessianEval);
  } else if (cookie == SLES_COOKIE || cookie == PC_COOKIE || cookie == KSP_COOKIE) {
    PLogEventDeactivate(SLES_Solve);
    PLogEventDeactivate(SLES_SetUp);
    PLogEventDeactivate(KSP_GMRESOrthogonalization);
    PLogEventDeactivate(PC_ModifySubMatrices);
    PLogEventDeactivate(PC_SetUp);
    PLogEventDeactivate(PC_SetUpOnBlocks);
    PLogEventDeactivate(PC_Apply);
    PLogEventDeactivate(PC_ApplySymmetricLeft);
    PLogEventDeactivate(PC_ApplySymmetricRight);
  } else if (cookie == MAT_COOKIE) {
    PLogEventDeactivate(MAT_Mult);
    PLogEventDeactivate(MAT_MatrixFreeMult);
    PLogEventDeactivate(MAT_AssemblyBegin);
    PLogEventDeactivate(MAT_AssemblyEnd);
    PLogEventDeactivate(MAT_GetOrdering);
    PLogEventDeactivate(MAT_MultTranspose);
    PLogEventDeactivate(MAT_MultAdd);
    PLogEventDeactivate(MAT_MultTransposeAdd);
    PLogEventDeactivate(MAT_LUFactor);
    PLogEventDeactivate(MAT_CholeskyFactor);
    PLogEventDeactivate(MAT_LUFactorSymbolic);
    PLogEventDeactivate(MAT_ILUFactorSymbolic);
    PLogEventDeactivate(MAT_CholeskyFactorSymbolic);
    PLogEventDeactivate(MAT_IncompleteCholeskyFactorSymbolic);
    PLogEventDeactivate(MAT_LUFactorNumeric);
    PLogEventDeactivate(MAT_CholeskyFactorNumeric);
    PLogEventDeactivate(MAT_CholeskyFactorNumeric);
    PLogEventDeactivate(MAT_Relax);
    PLogEventDeactivate(MAT_Copy);
    PLogEventDeactivate(MAT_Convert);
    PLogEventDeactivate(MAT_Scale);
    PLogEventDeactivate(MAT_ZeroEntries);
    PLogEventDeactivate(MAT_Solve);
    PLogEventDeactivate(MAT_SolveAdd);
    PLogEventDeactivate(MAT_SolveTranspose);
    PLogEventDeactivate(MAT_SolveTransposeAdd);
    PLogEventDeactivate(MAT_SetValues);
    PLogEventDeactivate(MAT_ForwardSolve);
    PLogEventDeactivate(MAT_BackwardSolve);
    PLogEventDeactivate(MAT_Load);
    PLogEventDeactivate(MAT_View);
    PLogEventDeactivate(MAT_ILUFactor);

    PLogEventDeactivate(MAT_GetValues);
    PLogEventDeactivate(MAT_IncreaseOverlap);
    PLogEventDeactivate(MAT_GetRow);
  } else if (cookie == VEC_COOKIE) {
    PLogEventDeactivate(VEC_Dot);
    PLogEventDeactivate(VEC_Norm);
    PLogEventDeactivate(VEC_Max);
    PLogEventDeactivate(VEC_Min);
    PLogEventDeactivate(VEC_TDot);
    PLogEventDeactivate(VEC_Scale);
    PLogEventDeactivate(VEC_Copy);
    PLogEventDeactivate(VEC_Set);
    PLogEventDeactivate(VEC_AXPY);
    PLogEventDeactivate(VEC_AYPX);
    PLogEventDeactivate(VEC_Swap);
    PLogEventDeactivate(VEC_WAXPY);
    PLogEventDeactivate(VEC_AssemblyBegin);
    PLogEventDeactivate(VEC_AssemblyEnd);
    PLogEventDeactivate(VEC_MTDot);
    PLogEventDeactivate(VEC_MDot);
    PLogEventDeactivate(VEC_MAXPY);
    PLogEventDeactivate(VEC_PMult);
    PLogEventDeactivate(VEC_SetValues);
    PLogEventDeactivate(VEC_Load);
    PLogEventDeactivate(VEC_View);
    PLogEventDeactivate(VEC_ScatterBegin);
    PLogEventDeactivate(VEC_ScatterEnd);
    PLogEventDeactivate(VEC_SetRandom);
  }
  PetscFunctionReturn(0);
}



/* end of -DPETSC_USE_LOG section */
#else  /* -------------------------------------------------------------*/

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"PLogObjectState"
int PLogObjectState(PetscObject obj,const char format[],...)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#endif

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"PetscGetTime"
/*@
   PetscGetTime - Returns the current time of day in seconds. This 
   returns wall-clock time.  

   Not Collective

   Output Parameter:
.  v - time counter

   Usage: 
.vb
      PLogDouble v1,v2,elapsed_time;
      ierr = PetscGetTime(&v1);CHKERR(ierr);
      .... perform some calculation ...
      ierr = PetscGetTime(&v2);CHKERR(ierr);
      elapsed_time = v2 - v1;   
.ve

   Notes:
   Since the PETSc libraries incorporate timing of phases and operations, 
   PetscGetTime() is intended only for timing of application codes.  
   The options database commands -log, -log_summary, and -log_all activate
   PETSc library timing.  See the users manual for further details.

   Level: intermediate

.seealso: PLogEventRegister(), PLogEventBegin(), PLogEventEnd(),  PLogStagePush(), 
          PLogStagePop(), PLogStageRegister(), PetscGetFlops()

.keywords:  get, time
@*/
int PetscGetTime(PLogDouble *t)
{
  PetscFunctionBegin;
  PetscTime(*t);
  PetscFunctionReturn(0);
}

/*MC
   PLogFlops - Adds floating point operations to the global counter.

   Input Parameter:
.  f - flop counter

   Synopsis:
   void PLogFlops(int f)

   Usage:
.vb
     int USER_EVENT;
     PLogEventRegister(&USER_EVENT,"User event","Color:");
     PLogEventBegin(USER_EVENT,0,0,0,0);
        [code segment to monitor]
        PLogFlops(user_flops)
     PLogEventEnd(USER_EVENT,0,0,0,0);
.ve

   Notes:
   A global counter logs all PETSc flop counts.  The user can use
   PLogFlops() to increment this counter to include flops for the 
   application code.  

   PETSc automatically logs library events if the code has been
   compiled with -DPETSC_USE_LOG (which is the default), and -log,
   -log_summary, or -log_all are specified.  PLogFlops() is
   intended for logging user flops to supplement this PETSc
   information.

   Level: intermediate

.seealso: PLogEventRegister(), PLogEventBegin(), PLogEventEnd(), PetscGetFlops()

.keywords: log, flops, floating point operations
M*/


/*MC
   PLogEventBegin - Logs the beginning of a user event. 

   Input Parameters:
+  e - integer associated with the event obtained from PLogEventRegister()
-  o1,o2,o3,o4 - objects associated with the event, or 0

   Synopsis:
   void PLogEventBegin(int e,PetscObject o1,PetscObject o2,PetscObject o3,
                       PetscObject o4)

   Usage:
.vb
     int USER_EVENT;
     int user_event_flops;
     PLogEventRegister(&USER_EVENT,"User event","Color:");
     PLogEventBegin(USER_EVENT,0,0,0,0);
        [code segment to monitor]
        PLogFlops(user_event_flops);
     PLogEventEnd(USER_EVENT,0,0,0,0);
.ve

   Notes:
   You should also register each integer event with the command 
   PLogEventRegister().  The source code must be compiled with 
   -DPETSC_USE_LOG, which is the default.

   PETSc automatically logs library events if the code has been
   compiled with -DPETSC_USE_LOG, and -log, -log_summary, or -log_all are
   specified.  PLogEventBegin() is intended for logging user events
   to supplement this PETSc information.

   Level: intermediate

.seealso: PLogEventRegister(), PLogEventEnd(), PLogFlops()

.keywords: log, event, begin
M*/

/*MC
   PLogEventEnd - Log the end of a user event.

   Input Parameters:
+  e - integer associated with the event obtained with PLogEventRegister()
-  o1,o2,o3,o4 - objects associated with the event, or 0

   Synopsis:
   void PLogEventEnd(int e,PetscObject o1,PetscObject o2,PetscObject o3,
                     PetscObject o4)

   Usage:
.vb
     int USER_EVENT;
     int user_event_flops;
     PLogEventRegister(&USER_EVENT,"User event","Color:");
     PLogEventBegin(USER_EVENT,0,0,0,0);
        [code segment to monitor]
        PLogFlops(user_event_flops);
     PLogEventEnd(USER_EVENT,0,0,0,0);
.ve

   Notes:
   You should also register each additional integer event with the command 
   PLogEventRegister(). Source code must be compiled with 
   -DPETSC_USE_LOG, which is the default.

   PETSc automatically logs library events if the code has been
   compiled with -DPETSC_USE_LOG, and -log, -log_summary, or -log_all are
   specified.  PLogEventEnd() is intended for logging user events
   to supplement this PETSc information.

   Level: intermediate

.seealso: PLogEventRegister(), PLogEventBegin(), PLogFlops()

.keywords: log, event, end
M*/

/*MC
   PLogEventBarrierBegin - Logs the time in a barrier before an event.

   Input Parameters:
.  e - integer associated with the event obtained from PLogEventRegister()
.  o1,o2,o3,o4 - objects associated with the event, or 0
.  comm - communicator the barrier takes place over

   Synopsis:
   void PLogEventBarrierBegin(int e,PetscObject o1,PetscObject o2,PetscObject o3,
                  PetscObject o4,MPI_Comm comm)

   Usage:
.vb
     PLogEventBarrierBegin(VEC_NormBarrier,0,0,0,0,comm);
       Code
     PLogEventBarrierEnd(VEC_NormBarrier,0,0,0,0,comm);
.ve

   Notes:
   This is for logging the amount of time spent in a barrier for an event
   that requires synchronization. 

   Additional Notes:
   Synchronization events always come in pairs; for example, VEC_NormBarrier and 
   VEC_Norm = VEC_NormBarrier + 1

   Level: developer

.seealso: PLogEventRegister(), PLogEventEnd(), PLogFlops(), PLogEventBegin(),
          PLogEventBarrierEnd()

.keywords: log, event, begin, barrier
M*/

/*MC
   PLogEventBarrierEnd - Logs the time in a barrier before an event.

   Input Parameters:
.  e - integer associated with the event obtained from PLogEventRegister()
.  o1,o2,o3,o4 - objects associated with the event, or 0
.  comm - communicator the barrier takes place over

   Synopsis:
   void PLogEventBarrierEnd(int e,PetscObject o1,PetscObject o2,PetscObject o3,
                  PetscObject o4,MPI_Comm comm)

    Usage:
.vb
     PLogEventBarrierBegin(VEC_NormBarrier,0,0,0,0,comm);
       Code
     PLogEventBarrierEnd(VEC_NormBarrier,0,0,0,0,comm);
.ve

   Notes:
   This is for logging the amount of time spent in a barrier for an event
   that requires synchronization. 

   Additional Notes:
   Synchronization events always come in pairs; for example, VEC_NormBarrier and 
   VEC_Norm = VEC_NormBarrier + 1

   Level: developer

.seealso: PLogEventRegister(), PLogEventEnd(), PLogFlops(), PLogEventBegin(),
          PLogEventBarrierBegin()

.keywords: log, event, begin, barrier
M*/

/*MC
   PreLoadBegin - Begin a segment of code that may be preloaded (run twice)
    to get accurate timings

   Input Parameter:
+   flag - PETSC_TRUE to run twice, PETSC_FALSE to run once, may be overridden
           with command line option -preload true or -preload false
-   name - name of first stage (lines of code timed seperately with -log_summary) to
           be preloaded

   Synopsis:
   void PreLoadBegin(PetscTruth flag,char *name);

   Usage:
.vb
     PreLoadBegin(PETSC_TRUE,"first stage);
       lines of code
       PreLoadStage("second stage");
       lines of code
     PreLoadEnd();
.ve

   Level: intermediate

.seealso: PLogEventRegister(), PLogEventBegin(), PLogEventEnd(), PreLoadEnd(), PreLoadStage()

.keywords: timing, preloading
M*/

/*MC
   PreLoadEnd - End a segment of code that may be preloaded (run twice)
    to get accurate timings

   Synopsis:
   void PreLoadEnd(void);

   Usage:
.vb
     PreLoadBegin(PETSC_TRUE,"first stage);
       lines of code
       PreLoadStage("second stage");
       lines of code
     PreLoadEnd();
.ve

   Level: intermediate

.seealso: PLogEventRegister(), PLogEventBegin(), PLogEventEnd(), PreLoadBegin(), PreLoadStage()

.keywords: timing, preloading
M*/

/*MC
   PreLoadStage - Start a new segment of code to be timed seperately.
    to get accurate timings

   Synopsis:
   void PreLoadStage(char *name);

   Usage:
.vb
     PreLoadBegin(PETSC_TRUE,"first stage);
       lines of code
       PreLoadStage("second stage");
       lines of code
     PreLoadEnd();
.ve

   Level: intermediate

.seealso: PLogEventRegister(), PLogEventBegin(), PLogEventEnd(), PreLoadBegin(), PreLoadEnd()

.keywords: timing, preloading
M*/

