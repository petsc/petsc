
#ifndef lint
static char vcid[] = "$Id: plog.c,v 1.122 1996/08/18 19:46:46 curfman Exp curfman $";
#endif
/*
      PETSc code to log object creation and destruction and PETSc events.
*/
#include "petsc.h"        /*I    "petsc.h"   I*/
#include "snes.h"      /* This include is to define all the PETSc cookies */
#if defined(HAVE_MPE)
#include "mpe.h"
#endif
#include <stdio.h>
#include <stdarg.h>
#include <sys/types.h>
#include "sys.h"
#if defined(HAVE_STDLIB_H)
#include <stdlib.h>
#endif
#if defined(HAVE_MALLOC_H) && !defined(__cplusplus)
#include <malloc.h>
#endif
#include "pinclude/pviewer.h"
#include "pinclude/petscfix.h"
#include "pinclude/ptime.h"

static int PrintInfo = 0;
static int PLogInfoFlags[] = {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                              1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                              1,1,1,1,1,1,1,1,1,1,1,1};
/*@C
    PLogInfoAllow - Causes PLogInfo() messages to be printed to standard output.

    Input Parameter:
.   flag - PETSC_TRUE or PETSC_FALSE

    Options Database Key:
$    -log_info 

.keywords: allow, information, printing, monitoring

.seealso: PLogInfo()
@*/
int PLogInfoAllow(PetscTruth flag)
{
  PrintInfo = (int) flag;
  return 0;
}

extern FILE *petsc_history;

/*@
    PLogInfoDeactivateClass - Deactivates PlogInfo() messages for a PETSc 
                              object class.

    Input Parameter:
.   objclass - for example, MAT_COOKIE, SNES_COOKIE, etc.

.seealso: PLogInfoActivateClass(), PLogInfo(), PLogInfoAllow()
@*/
int PLogInfoDeactivateClass(int objclass)
{
  PLogInfoFlags[objclass - PETSC_COOKIE - 1] = 0;
  if (objclass == SLES_COOKIE) {
    PLogInfoFlags[PC_COOKIE - PETSC_COOKIE - 1]  = 0;
    PLogInfoFlags[PC_COOKIE - PETSC_COOKIE - 1] = 0;
  }
  return 0;
}

/*@
    PLogInfoActivateClass - Activates PlogInfo() messages for a PETSc 
                            object class.

    Input Parameter:
.   objclass - for example, MAT_COOKIE, SNES_COOKIE, etc.

.seealso: PLogInfoDeactivateClass(), PLogInfo(), PLogInfoAllow()
@*/
int PLogInfoActivateClass(int objclass)
{
  PLogInfoFlags[objclass - PETSC_COOKIE - 1] = 1;
  if (objclass == SLES_COOKIE) {
    PLogInfoFlags[PC_COOKIE - PETSC_COOKIE - 1]  = 1;
    PLogInfoFlags[KSP_COOKIE - PETSC_COOKIE - 1] = 1;
  }
  return 0;
}

/*@C
    PLogInfo - Logs informative data, which is printed to standard output
    when the option -log_info is specified.

    Input Parameter:
.   vobj - object most closely associated with the logging statement
.   message - logging message, using standard "printf" format

    Options Database Key:
$    -log_info : activates printing of PLogInfo() messages 

    Fortran Note:
    This routine is not supported in Fortran.

    Example of Usage:
$
$     Mat A
$     double alpha
$     PLogInfo(A,"Matrix uses parameter alpha=%g\n",alpha);
$

.keywords: information, printing, monitoring

.seealso: PLogInfoAllow()
@*/
int PLogInfo(void *vobj,char *message,...)
{
  va_list     Argp;
  int         rank,urank,len;
  PetscObject obj = (PetscObject) vobj;
  char        string[256];

  if (obj) PetscValidHeader(obj);
  if (!PrintInfo) return 0;
  if (obj && !PLogInfoFlags[obj->cookie - PETSC_COOKIE - 1]) return 0;
  if (!obj) rank = 0;
  else      {MPI_Comm_rank(obj->comm,&rank);} 
  if (rank) return 0;

  MPI_Comm_rank(MPI_COMM_WORLD,&urank);
  va_start( Argp, message );
  sprintf(string,"[%d]",urank); len = PetscStrlen(string);
  vsprintf(string+len,message,Argp);
  fprintf(stdout,"%s",string);
  fflush(stdout);
  if (petsc_history) {
    vfprintf(petsc_history,message,Argp);
  }
  va_end( Argp );
  return 0;
}

/* -------------------------------------------------------------------*/
#if defined(PETSC_LOG)
static int PLOG_USER_EVENT_LOW = PLOG_USER_EVENT_LOW_STATIC;

/* 
   Make sure that all events used by PETSc have the
   corresponding flags set here: 
     1 - activated for PETSc logging
     0 - not activated for PETSc logging
 */
int PLogEventFlags[] = {      1,1,1,1,1,  /* 0 - 24*/
                        1,1,1,1,1,
                        1,1,1,1,1,
                        1,1,1,1,1,
                        1,1,1,1,1,
                        1,1,1,1,1,  /* 25 -49 */
                        1,1,1,1,1,
                        1,1,1,1,1,
                        1,1,1,1,1,
                        1,1,1,1,1,
                        1,1,1,1,1, /* 50 - 74 */
                        1,1,1,1,1,
                        1,1,1,1,1,
                        1,1,1,1,1,
                        1,1,1,1,1,
                        1,1,1,1,1, /* 75 - 99 */
                        1,1,1,1,1,
                        1,1,1,1,1,
                        1,1,1,1,1,
                        1,1,1,1,1,
                        1,1,1,1,1, /* 100 - 124 */ 
                        1,1,1,1,1,
                        1,1,1,1,1,
                        1,1,1,1,1,
                        1,1,1,1,1,
                        1,1,1,1,1, /* 125 - 149 */
                        1,1,1,1,1,
                        1,1,1,1,1,
                        1,1,1,1,1,
                        1,1,1,1,1,
                        1,1,1,1,1, /* 150 - 174 */
                        1,1,1,1,1,
                        1,1,1,1,1,
                        1,1,1,1,1,
                        1,1,1,1,1,
                        1,1,1,1,1,
                        1,1,1,1,1, /* 175 - 199 */
                        1,1,1,1,1,
                        1,1,1,1,1,
                        1,1,1,1,1,
                        1,1,1,1,1};

static char *(oname[]) = {"Viewer           ",
                          "Index set        ",
                          "Vector           ",
                          "Vector Scatter   ",
                          "Matrix           ",
                          "Graphic          ",
                          "Line graph       ",
                          "Krylov Solver    ",
                          "Preconditioner   ",
                          "SLES             ",
                          "Grid             ",
                          "Stencil          ",
                          "SNES             ",
                          "Distributed array",
                          "Matrix scatter   ",
                          "                 ",
                          "                 ",
			  "                 "};

char *(PLogEventName[]) = {"MatMult         ",
                         "MatMatFreeMult  ",
                         "MatAssemblyBegin",
                         "MatAssemblyEnd  ",
                         "MatGetReordering",
                         "MatMultTrans    ",
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
                         "MatSolveTrans   ",
                         "MatSolveTransAdd",
                         "MatSetValues    ",
                         "MatForwardSolve ",
                         "MatBackwardSolve",
                         "MatLoad         ",
                         "MatView         ",
                         "MatILUFactor    ",
                         "                ",
                         "MatGetSubMatrice",
                         "MatGetValues    ",
                         "MatIncreaseOvlap",
                         "MatGetRow       ",
                         "                ",
                         "                ",
                         "                ",
                         "                ",
                         "VecDot          ",
                         "VecNorm         ",
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
                         "VecMDot         ",
                         "VecMAXPY        ",
                         "VecPointwiseMult",
                         "VecSetValues    ",
                         "VecLoad         ",
                         "VecView         ",
                         "VecScatterBegin ",
                         "VecScatterEnd   ",
                         "VecSetRandom    ",
                         " ",
                         " ",
                         " ",
                         " ",
                         " ",
                         " ",
                         "SLESSolve       ",
                         "SLESSetUp       ",
                         "KSPGMRESOrthog  ",
                         " ",
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
                         " ",
                         " ",
                         " ",
                         "TS_Step         ",
                         " ",
                         " ",
                         " ",
                         " ",
                         " ",
                         " ",
                         " ",
                         " ",
                         " ",
                         "PetscBarrier    ",
                         " "," "," "," ",
                         " "," "," "," "," ",
                         "DFVecRefineVecto",
                         "DFVec_AssembleFu",
                         "DFVec_GetCompone",
                         "DFVec_DrawContou",
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
  double      time,flops,mem,maxmem;
  int         cookie,type,event,id1,id2,id3;
} Events;

typedef struct {
  int         parent;
  double      mem;
  char        string[64];
  char        name[32];
  PetscObject obj;
} Objects;

/* 
    Global counters 
*/
double _TotalFlops = 0.0;
double irecv_ct = 0.0,isend_ct = 0.0,wait_ct = 0.0,wait_any_ct = 0.0;
double irecv_len = 0.0,isend_len = 0.0,recv_len = 0.0, send_len = 0.0;
double send_ct = 0.0,recv_ct = 0.0;
double wait_all_ct = 0.0,allreduce_ct = 0.0,sum_of_waits_ct = 0.0;

/*
    Log counters in this file only 
*/
static double  BaseTime;
static Events  *events = 0;
static Objects *objects = 0;

static int     nobjects = 0, nevents = 0, objectsspace = CHUNCK;
static int     ObjectsDestroyed = 0, eventsspace = CHUNCK;
static double  ObjectsType[20][4];

static int     EventsStage = 0;    /* which log sessions are we using */
static int     EventsStageMax = 0; /* highest event log used */ 
static int     EventsStagePushed = 0;
static int     EventsStageStack[100];
static char    *(EventsStageName[]) = {0,0,0,0,0,0,0,0,0,0};
static double  EventsStageFlops[] = {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0};
static double  EventsStageTime[] = {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0};
static double  EventsStageMessageCounts[] = {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0};
static double  EventsStageMessageLengths[] = {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0};
static double  EventsStageReductions[] = {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0};
#define COUNT      0
#define FLOPS      1
#define TIME       2
#define MESSAGES   3
#define LENGTHS    4
#define REDUCTIONS 5
static double  EventsType[10][PLOG_USER_EVENT_HIGH][6];


/*@C
    PLogStageRegister - Attaches a charactor string name to a logging stage.

    Input Parameters:
.   stage - the stage from 0 to 9 inclusive
.   sname - the name to associate with that stage

    Notes:
    The string information (for stage names) is not copied, so the user
    should NOT change any strings specified here.

.seealso: PLogStagePush(), PLogStagePop()
@*/
int PLogStageRegister(int stage, char *sname)
{
  if (stage < 0 || stage > 10) SETERRQ(1,"PLogStageRegister:Out of range");
  EventsStageName[stage] = sname;
  return 0;
}

/*@C
   PLogStagePush - Users can log up to 10 stages within a code by using
   -log_summary in conjunction with PLogStagePush() and PLogStagePop().

   Input Parameters:
.  stage - stage on which to log (0 <= stage <= 9)

   Example of Usage:
   If the option -log_sumary is used to run the program containing the 
   following code, then 3 sets of summary data will be printed during
   PetscFinalize().
$
$     PetscInitialize(int *argc,char ***args,0,0);
$     [stage 0 of code]   
$     for (i=0; i<ntimes; i++) {
$        PLogStagePush(1);
$        [stage 1 of code]
$        PLogStagePop()
$        PLogStagePush(2);
$        [stage 2 of code]
$        PLogStagePop()
$     }
$     PetscFinalize();
$

   Notes:  
   Use PLogStageRegister() to register a stage.

.keywords: log, push, stage

.seealso: PLogStagePop(), PLogStageRegister()
@*/
int PLogStagePush(int stage)
{
  if (stage < 0 || stage > 10) SETERRQ(1,"PLogStagePush:Out of range");
  /* record flops/time of previous stage */
  if (EventsStagePushed) {
    PetscTimeAdd(EventsStageTime[EventsStage]);
    EventsStageFlops[EventsStage]          += _TotalFlops;
    EventsStageMessageCounts[EventsStage]  += irecv_ct + isend_ct + recv_ct + send_ct;
    EventsStageMessageLengths[EventsStage] += irecv_len + isend_len + recv_len + send_len;
    EventsStageReductions[EventsStage]     += allreduce_ct;
  }
  EventsStageStack[EventsStagePushed] = EventsStage;
  if (EventsStagePushed++ > 99) SETERRQ(1,"PLogStagePush:Too many pushes");
  EventsStage = stage;
  if (stage > EventsStageMax) EventsStageMax = stage;
  PetscTimeSubtract(EventsStageTime[EventsStage]);
  EventsStageFlops[EventsStage]          -= _TotalFlops;
  EventsStageMessageCounts[EventsStage]  -= irecv_ct + isend_ct + recv_ct + send_ct;
  EventsStageMessageLengths[EventsStage] -= irecv_len + isend_len + recv_len + send_len;
  EventsStageReductions[EventsStage]     -= allreduce_ct;
  return 0;
}

/*@C
   PLogStagePop - Users can log up to 10 stages within a code by using
   -log_summary in conjunction with PLogStagePush() and PLogStagePop().

   Example of Usage:
   If the option -log_sumary is used to run the program containing the 
   following code, then 2 sets of summary data will be printed during
   PetscFinalize().
$
$     PetscInitialize(int *argc,char ***args,0,0);
$     [stage 0 of code]   
$     PLogStagePush(1);
$     [stage 1 of code]
$     [some code (stage 1)]
$     PLogStagePop();
$     [more stage 0 of code]   
$     PetscFinalize();
$

   Notes:  
   Use PLogStageRegister() to register a stage.

.keywords: log, pop, stage

.seealso: PLogStagePush(), PLogStageRegister()
@*/
int PLogStagePop()
{
  PetscTimeAdd(EventsStageTime[EventsStage]);
  EventsStageFlops[EventsStage]          += _TotalFlops;
  EventsStageMessageCounts[EventsStage]  += irecv_ct + isend_ct + recv_ct + send_ct;
  EventsStageMessageLengths[EventsStage] += irecv_len + isend_len + recv_len + send_len;
  EventsStageReductions[EventsStage]     += allreduce_ct;
  if (EventsStagePushed < 1) SETERRQ(1,"PLogStagePop:Too many pops\n");
  EventsStage = EventsStageStack[--EventsStagePushed];
  if (EventsStagePushed) {
    PetscTimeSubtract(EventsStageTime[EventsStage]);
    EventsStageFlops[EventsStage]          -= _TotalFlops;
    EventsStageMessageCounts[EventsStage]  -= irecv_ct + isend_ct + recv_ct + send_ct;
    EventsStageMessageLengths[EventsStage] -= irecv_len + isend_len + recv_len + send_len;
    EventsStageReductions[EventsStage]     -= allreduce_ct;
  }
  return 0;
}

/* --------------------------------------------------------------------------------*/

int (*_PHC)(PetscObject) = 0;
int (*_PHD)(PetscObject) = 0;
int (*_PLB)(int,int,PetscObject,PetscObject,PetscObject,PetscObject) = 0;
int (*_PLE)(int,int,PetscObject,PetscObject,PetscObject,PetscObject) = 0;

/*
      Default object create logger 
*/
int phc(PetscObject obj)
{
  if (nevents >= eventsspace) {
    Events *tmp;
    double end,start;
    PetscTime(start);
    tmp = (Events *) malloc((eventsspace+CHUNCK)*sizeof(Events));CHKPTRQ(tmp);
    PetscMemcpy(tmp,events,eventsspace*sizeof(Events));
    free(events);
    events = tmp; eventsspace += CHUNCK;
    PetscTime(end); BaseTime += (end - start);
  }
  if (nobjects >= objectsspace) {
    Objects *tmp;
    double end,start;
    PetscTime(start);
    tmp = (Objects *) malloc((objectsspace+CHUNCK)*sizeof(Objects));CHKPTRQ(tmp);
    PetscMemcpy(tmp,objects,objectsspace*sizeof(Objects));
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
  PetscMemzero(objects[nobjects].string,64*sizeof(char));
  PetscMemzero(objects[nobjects].name,16*sizeof(char));
  obj->id = nobjects++;
  ObjectsType[obj->cookie - PETSC_COOKIE-1][0]++;
  return 0;
}
/*
      Default object destroy logger 
*/
int phd(PetscObject obj)
{
  PetscObject parent;
  if (nevents >= eventsspace) {
    Events *tmp;
    double end,start;
    PetscTime(start);
    tmp = (Events *) malloc((eventsspace+CHUNCK)*sizeof(Events));CHKPTRQ(tmp);
    PetscMemcpy(tmp,events,eventsspace*sizeof(Events));
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
  if (obj->parent) {objects[obj->id].parent   = obj->parent->id;}
  else {objects[obj->id].parent   = -1;}
  if (obj->name) { PetscStrncpy(objects[obj->id].name,obj->name,16);}
  objects[obj->id].obj      = 0;
  objects[obj->id].mem      = obj->mem;
  ObjectsType[obj->cookie - PETSC_COOKIE-1][1]++;
  ObjectsType[obj->cookie - PETSC_COOKIE-1][2] += obj->mem;
  /*
     Credit all ancestors with your memory 
  */
  parent = obj->parent;
  while (parent) {
    int exists;
    PetscObjectExists(parent,&exists);
    if (!exists) break;
    ObjectsType[parent->cookie - PETSC_COOKIE-1][3] += obj->mem;   
    parent = parent->parent;
  } 
  ObjectsDestroyed++;
  return 0;
}
/*
    Event begin logger with complete logging
*/
int plball(int event,int t,PetscObject o1,PetscObject o2,PetscObject o3,PetscObject o4)
{
 double ltime;
 if (nevents >= eventsspace) {
    Events *tmp;
    double end,start;
    PetscTime(start);
    tmp = (Events *) malloc((eventsspace+CHUNCK)*sizeof(Events));CHKPTRQ(tmp);
    PetscMemcpy(tmp,events,eventsspace*sizeof(Events));
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
  if (t != 1) return 0;
  EventsType[EventsStage][event][COUNT]++;
  EventsType[EventsStage][event][TIME]        -= ltime;
  EventsType[EventsStage][event][FLOPS]       -= _TotalFlops;
  EventsType[EventsStage][event][MESSAGES]    -= irecv_ct + isend_ct + recv_ct + send_ct;
  EventsType[EventsStage][event][LENGTHS]     -= irecv_len + isend_len + recv_len + send_len;
  EventsType[EventsStage][event][REDUCTIONS]  -= allreduce_ct;
  return 0;
}
/*
     Event end logger with complete logging
*/
int pleall(int event,int t,PetscObject o1,PetscObject o2,PetscObject o3,PetscObject o4)
{
 double ltime;
 if (nevents >= eventsspace) {
    Events *tmp;
    double end,start;
    PetscTime(start);
    tmp = (Events *) malloc((eventsspace+CHUNCK)*sizeof(Events));CHKPTRQ(tmp);
    PetscMemcpy(tmp,events,eventsspace*sizeof(Events));
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
  if (t != 1) return 0;
  EventsType[EventsStage][event][TIME]        += ltime;
  EventsType[EventsStage][event][FLOPS]       += _TotalFlops;
  EventsType[EventsStage][event][MESSAGES]    += irecv_ct + isend_ct + recv_ct + send_ct;
  EventsType[EventsStage][event][LENGTHS]     += irecv_len + isend_len + recv_len + send_len;
  EventsType[EventsStage][event][REDUCTIONS]  += allreduce_ct;
  return 0;
}
/*
     Default event begin logger
*/
int plb(int event,int t,PetscObject o1,PetscObject o2,PetscObject o3,PetscObject o4)
{
  if (t != 1) return 0;
  EventsType[EventsStage][event][COUNT]++;
  PetscTimeSubtract(EventsType[EventsStage][event][TIME]);
  EventsType[EventsStage][event][FLOPS]       -= _TotalFlops;
  EventsType[EventsStage][event][MESSAGES]    -= irecv_ct + isend_ct + recv_ct + send_ct;
  EventsType[EventsStage][event][LENGTHS]     -= irecv_len + isend_len + recv_len + send_len;
  EventsType[EventsStage][event][REDUCTIONS]  -= allreduce_ct;
  return 0;
}

/*
     Default event end logger
*/
int ple(int event,int t,PetscObject o1,PetscObject o2,PetscObject o3,PetscObject o4)
{
  if (t != 1) return 0;
  PetscTimeAdd(EventsType[EventsStage][event][TIME]);
  EventsType[EventsStage][event][FLOPS]       += _TotalFlops;
  EventsType[EventsStage][event][MESSAGES]    += irecv_ct + isend_ct + recv_ct + send_ct;
  EventsType[EventsStage][event][LENGTHS]     += irecv_len + isend_len + recv_len + send_len;
  EventsType[EventsStage][event][REDUCTIONS]  += allreduce_ct;
  return 0;
}

int PLogObjectState(PetscObject obj,char *format,...)
{
  va_list Argp;
  if (!objects) return 0;
  va_start( Argp, format );
  vsprintf(objects[obj->id].string,format,Argp);
  va_end( Argp );
  return 0;
}

/*@C
   PLogAllBegin - Turns on extensive logging of objects and events. Logs 
   all events. This creates large log files and slows the program down.

   Options Database Keys:
$  -log_all : Prints extensive log information (for code compiled
$      with PETSC_LOG)

   Notes:
   A related routine is PLogBegin (with the options key -log), which is 
   intended for production runs since it logs only flop rates and object
   creation (and shouldn't significantly slow the programs).

.keywords: log, all, begin

.seealso: PLogDump(), PLogBegin()
@*/
int PLogAllBegin()
{
  objects = (Objects*) malloc(CHUNCK*sizeof(Objects));CHKPTRQ(objects);
  events  = (Events*) malloc(CHUNCK*sizeof(Events));CHKPTRQ(events);
  _PHC    = phc;
  _PHD    = phd;
  _PLB    = plball;
  _PLE    = pleall;
  /* all processors sync here for more consistent logging */
  MPI_Barrier(MPI_COMM_WORLD);
  PetscTime(BaseTime);
  PLogStagePush(0);
  return 0;
}

/*@C
   PLogDestroy - Destroys the object and event logging data and resets the 
   global counters. 

   Notes:
   This routine should not usually be used by programmers. Instead employ 
   PLogStagePush() and PLogStagePop().

.keywords: log, destroy

.seealso: PLogDump(), PLogAllBegin(), PLogPrintSummary(), PLogStagePush(), PlogStagePop()
@*/
int PLogDestroy()
{
  /* Destroying phase */
  if (objects) {free(objects); objects = 0;}
  if (events)  {free(events); events = 0;}
  _PHC             = 0;
  _PHD             = 0;

  /* Resetting phase */
  PetscMemzero(EventsType,sizeof(EventsType));
  PetscMemzero(ObjectsType,sizeof(ObjectsType));
  _TotalFlops      = 0.0;
  nobjects         = 0;
  nevents          = 0;
  ObjectsDestroyed = 0;
  return 0;
}

/*@C
    PLogBegin - Turns on logging of objects and events. This logs flop
    rates and object creation and should not slow programs down too much.
    This routine may be called more than once.

   Options Database Keys:
$  -log : Prints basic log information (for code compiled 
$      with PETSC_LOG)
$  -log_summary : Prints summary of flop and timing information 
$      to screen (for code compiled with PETSC_LOG)

.keywords: log, begin

.seealso: PLogDump(), PLogAllBegin(), PLogPrintSummary()
@*/
int PLogBegin()
{
  objects = (Objects*) malloc(CHUNCK*sizeof(Objects));CHKPTRQ(objects);
  events  = (Events*) malloc(CHUNCK*sizeof(Events));CHKPTRQ(events);
  _PHC    = phc;
  _PHD    = phd;
  _PLB    = plb;
  _PLE    = ple;
  /* all processors sync here for more consistent logging */
  MPI_Barrier(MPI_COMM_WORLD);
  PetscTime(BaseTime);
  PLogStagePush(0);
  return 0;
}

/*@C
   PLogDump - Dumps logs of objects to a file. This file is intended to 
   be read by petsc/bin/petscview.

   Input Parameter:
.  name - an optional file name

   Options Database Keys:
$  -log : Prints basic log information (for code compiled 
$      with PETSC_LOG)
$  -log_all : Prints extensive log information (for code compiled
$      with PETSC_LOG)
   
   Notes:
   The default file name is 
$      Log.<rank>
   where <rank> is the processor number. If no name is specified, 
   this file will be used.

.keywords: log, dump

.seealso: PLogBegin(), PLogPrintSummary()
@*/
int PLogDump(char* sname)
{
  int    i,rank;
  FILE   *fd;
  char   file[64];
  double flops,_TotalTime;
  
  PetscTime(_TotalTime);
  _TotalTime -= BaseTime;

  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  if (sname) sprintf(file,"%s.%d",sname,rank);
  else  sprintf(file,"Log.%d",rank);
  fd = fopen(file,"w"); if (!fd) SETERRQ(1,"PlogDump:cannot open file");

  fprintf(fd,"Objects created %d Destroyed %d\n",nobjects,
                                                 ObjectsDestroyed);
  fprintf(fd,"Clock Resolution %g\n",0.0);
  fprintf(fd,"Events %d\n",nevents);
  for ( i=0; i<nevents; i++ ) {
    fprintf(fd,"%g %d %d %d %d %d %d %g %g %g\n",events[i].time,
                              events[i].event,
                              events[i].cookie,events[i].type,events[i].id1,
                              events[i].id2,events[i].id3,
                              events[i].flops,events[i].mem,
                              events[i].maxmem);
  }
  for ( i=0; i<nobjects; i++ ) {
    fprintf(fd,"%d %d\n",objects[i].parent,(int)objects[i].mem);
    if (!objects[i].string[0]) {fprintf(fd,"No Info\n");}
    else fprintf(fd,"%s\n",objects[i].string);
    if (!objects[i].name[0]) {fprintf(fd,"No Name\n");}
    else fprintf(fd,"%s\n",objects[i].name);
  }
  for ( i=0; i<PLOG_USER_EVENT_HIGH; i++ ) {
    flops = 0.0;
    if (EventsType[0][i][TIME]){flops = EventsType[0][i][FLOPS]/EventsType[0][i][TIME];}
    fprintf(fd,"%d %16g %16g %16g %16g\n",i,EventsType[0][i][COUNT],
                      EventsType[0][i][FLOPS],EventsType[0][i][TIME],flops);
  }
  fprintf(fd,"Total Flops %14e %16.8e\n",_TotalFlops,_TotalTime);
  fclose(fd);
  return 0;
}

extern char *PLogEventColor[];


/*@C
    PLogEventRegister - Registers an event name for logging operations in 
    an application code. 

    Input Parameter:
.   string - name associated with the event
.   color - string giving a color for displaying event, and the
            display pattern (used by Upshot/Nupshot), 
            e.g., - "red:", "green:vlines3"

    Output Parameter:
.   e -  event id for use with PLogEventBegin() and PLogEventEnd().

    Notes: 
    PETSc automatically logs library events if the code has been
    compiled with -DPETSC_LOG (which is the default) and -log,
    -log_summary, or -log_all are specified.  PLogEventRegister() is
    intended for logging user events to supplement this PETSc
    information. 

    PETSc can gather data for use with the utilities Upshot/Nupshot
    (part of the MPICH distribution).  If PETSc has been compiled
    with flag -DHAVE_MPE (MPE is an additional utility within
    MPICH), the user can employ another command line option, -log_mpe,
    to create a logfile, "mpe.log", which can be visualized
    Upshot/Nupshot. The color argument is used by this utility
    in forming the display of this event.

    Example of Usage:
$     #include "plog.h"
$     int USER_EVENT;
$     int user_event_flops;
$     PLogEventRegister(&USER_EVENT,"User event name","EventColor");
$     PLogEventBegin(USER_EVENT,0,0,0,0);
$        [code segment to monitor]
$        PLogFlops(user_event_flops);
$     PLogEventEnd(USER_EVENT,0,0,0,0);

.keywords: log, event, register

.seealso: PLogEventBegin(), PLogEventEnd(), PLogFlops(),
          PLogEventMPEActivate(), PLogEventMPEDeactivate(),
          PLogEventActivate(), PLogEventDeactivate()
@*/
int PLogEventRegister(int *e,char *string,char *color)
{
  *e = PLOG_USER_EVENT_LOW++;
  if (*e > PLOG_USER_EVENT_HIGH) { 
    *e = 0;
    SETERRQ(1,"PLogEventRegister:Out of event IDs");
  }
  PLogEventName[*e] = string;
#if defined(HAVE_MPE)
  if (UseMPE) {
    int rank;

    PLogEventMPEFlags[*e]       = 1;
    PLogEventColor[*e] = color;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    if (!rank) {
      MPE_Describe_state(MPEBEGIN+2*(*e),MPEBEGIN+2*(*e)+1,string,color);
    }
  }
#endif
  return 0;
}
  
/*@
    PLogEventDeactivate - Indicates that a particular event should not be
       logged. Note: the event may be either a pre-defined
       PETSc event (found in include/plog.h) or an event number obtained
       with PLogEventRegister().

  Input Parameter:
.   event - integer indicating event

   Example of Usage:
$
$     PetscInitialize(int *argc,char ***args,0,0);
$     PLogEventDeactivate(VEC_SetValues);
$      code where you do not want to log VecSetValues() 
$     PLogEventActivate(VEC_SetValues);
$      code where you do want to log VecSetValues() 
$     .......
$     PetscFinalize();
$

.seealso: PLogEventMPEDeactivate(),PLogEventMPEActivate(),PlogEventActivate()
@*/
int PLogEventDeactivate(int event)
{
  PLogEventFlags[event] = 0;
  return 0;
}
/*@
    PLogEventActivate - Indicates that a particular event should be
       logged. Note: the event may be either a pre-defined
       PETSc event (found in include/plog.h) or an event number obtained
       with PLogEventRegister().

  Input Parameter:
.   event - integer indicating event

   Example of Usage:
$
$     PetscInitialize(int *argc,char ***args,0,0);
$     PLogEventDeactivate(VEC_SetValues);
$      code where you do not want to log VecSetValues() 
$     PLogEventActivate(VEC_SetValues);
$      code where you do want to log VecSetValues() 
$     .......
$     PetscFinalize();
$

.seealso: PLogEventMPEDeactivate(),PLogEventMPEActivate(),PlogEventDeactivate()
@*/
int PLogEventActivate(int event)
{
  PLogEventFlags[event] = 1;
  return 0;
}


/*@C
   PLogPrintSummary - Prints a summary of the logging.

   Input Parameter:
.  file - a file pointer
.  comm - MPI communicator (one processor prints)

   Options Database Keys:
$  -log_summary : Prints summary of log information (for code
   compiled with PETSC_LOG)

   Notes:
   More extensive examination of the log information can be done with 
   PLogDump(), which is activated by the option -log or -log_all, in 
   combination with petsc/bin/petscview.
   
.keywords: log, dump, print

.seealso: PLogBegin(), PLogDump()
@*/
int PLogPrintSummary(MPI_Comm comm,FILE *fd)
{
  double maxo,mino,aveo,mem,totmem,maxmem,minmem,mlensmcounts;
  double maxf,minf,avef,totf,_TotalTime,maxt,mint,avet,tott,ratio;
  double fmin,fmax,ftot,wdou,totts,totff,rat,sstime,sflops,ratf;
  double ptotts,ptotff,ptotts_stime,ptotff_sflops,rat1,rat2,rat3;
  double minm,maxm,avem,totm,minr,maxr,maxml,minml,totml,aveml,totr;
  double rp,mp,lp,rpg,mpg,lpg,totms,totmls,totrs,mps,lps,rps,lpmp;
  double pstime,psflops1,psflops,flopr;
  int    size,i,j;
  char   arch[10],hostname[64],username[16];

  /* pop off any stages the user forgot to remove */
  while (EventsStagePushed) PLogStagePop();

  PetscTime(_TotalTime);  _TotalTime -= BaseTime;
  MPI_Comm_size(comm,&size);

  PetscFPrintf(comm,fd,"************************************************************************************************************************\n");
  PetscFPrintf(comm,fd,"***                   WIDEN YOUR WINDOW TO 120 CHARACTERS.  Use 'enscript -r' to print this document                 ***\n");
  PetscFPrintf(comm,fd,"************************************************************************************************************************\n");

  PetscFPrintf(comm,fd,"\n---------------------------------------------- PETSc Performance Summary: ----------------------------------------------\n\n");
  PetscGetArchType(arch,10);
  PetscGetHostName(hostname,64);
  PetscGetUserName(username,16);
  if (size == 1)
    PetscFPrintf(comm,fd,"%s on a %s named %s with %d processor, by %s %s",
                 OptionsGetProgramName(),arch,hostname,size,username,PetscGetDate());
  else
    PetscFPrintf(comm,fd,"%s on a %s named %s with %d processors, by %s %s",
                 OptionsGetProgramName(),arch,hostname,size,username, PetscGetDate());

  wdou = _TotalFlops; 
  MPI_Allreduce(&wdou,&minf,1,MPI_DOUBLE,MPI_MIN,comm);
  MPI_Allreduce(&wdou,&maxf,1,MPI_DOUBLE,MPI_MAX,comm);
  MPI_Allreduce(&wdou,&totf,1,MPI_DOUBLE,MPI_SUM,comm);
  avef = (totf)/((double) size);
  wdou = nobjects;
  MPI_Allreduce(&wdou,&mino,1,MPI_DOUBLE,MPI_MIN,comm);
  MPI_Allreduce(&wdou,&maxo,1,MPI_DOUBLE,MPI_MAX,comm);
  MPI_Allreduce(&wdou,&aveo,1,MPI_DOUBLE,MPI_SUM,comm);
  aveo = (aveo)/((double) size);
  wdou = _TotalTime;
  MPI_Allreduce(&wdou,&mint,1,MPI_DOUBLE,MPI_MIN,comm);
  MPI_Allreduce(&wdou,&maxt,1,MPI_DOUBLE,MPI_MAX,comm);
  MPI_Allreduce(&wdou,&tott,1,MPI_DOUBLE,MPI_SUM,comm);
  avet = (tott)/((double) size);

  PetscFPrintf(comm,fd,"\n                         Max         Min        Avg        Total \n");
  if (mint) ratio = maxt/mint; else ratio = 0.0;
  PetscFPrintf(comm,fd,"Time (sec):           %5.3e   %6.1f   %5.3e\n",maxt,ratio,avet);
  if (mino) ratio = maxo/mino; else ratio = 0.0;
  PetscFPrintf(comm,fd,"Objects:              %5.3e   %6.1f   %5.3e\n",maxo,ratio,aveo);
  if (minf) ratio = maxf/minf; else ratio = 0.0;
  PetscFPrintf(comm,fd,"Flops:                %5.3e   %6.1f   %5.3e  %5.3e\n",maxf,ratio,avef,totf);

  if (mint) fmin = minf/mint; else fmin = 0;
  if (maxt) fmax = maxf/maxt; else fmax = 0;
  if (maxt) ftot = totf/maxt; else ftot = 0;
  if (fmin) ratio = fmax/fmin; else ratio = 0.0;
  PetscFPrintf(comm,fd,"Flops/sec:            %5.3e   %6.1f              %5.3e\n",fmax,ratio,ftot);
  PetscTrSpace(PETSC_NULL,PETSC_NULL,&mem);
  if (mem > 0.0) {
    MPI_Allreduce(&mem,&maxmem,1,MPI_DOUBLE,MPI_MAX,comm);
    MPI_Allreduce(&mem,&minmem,1,MPI_DOUBLE,MPI_MIN,comm);
    MPI_Allreduce(&mem,&totmem,1,MPI_DOUBLE,MPI_SUM,comm);
    if (minmem) ratio = maxmem/minmem; else ratio = 0.0;
    PetscFPrintf(comm,fd,"Memory:               %5.3e   %6.1f              %5.3e\n",maxmem,ratio,totmem);
  }
  wdou = .5*(irecv_ct + isend_ct + recv_ct + send_ct);
  MPI_Allreduce(&wdou,&minm,1,MPI_DOUBLE,MPI_MIN,comm);
  MPI_Allreduce(&wdou,&maxm,1,MPI_DOUBLE,MPI_MAX,comm);
  MPI_Allreduce(&wdou,&totm,1,MPI_DOUBLE,MPI_SUM,comm);
  avem = (totm)/((double) size);
  wdou = .5*(irecv_len + isend_len + recv_len + send_len);
  MPI_Allreduce(&wdou,&minml,1,MPI_DOUBLE,MPI_MIN,comm);
  MPI_Allreduce(&wdou,&maxml,1,MPI_DOUBLE,MPI_MAX,comm);
  MPI_Allreduce(&wdou,&totml,1,MPI_DOUBLE,MPI_SUM,comm);
  if (totm) aveml = (totml)/(totm); else aveml = 0;
  if (minm) ratio = maxm/minm; else ratio = 0.0;
  PetscFPrintf(comm,fd,"MPI Messages:         %5.3e   %6.1f   %5.3e  %5.3e\n",maxm,ratio,avem,totm);
  if (minml) ratio = maxml/minml; else ratio = 0.0;
  PetscFPrintf(comm,fd,"MPI Message Lengths:  %5.3e   %6.1f   %5.3e  %5.3e\n",maxml,ratio,aveml,totml);
  MPI_Allreduce(&allreduce_ct,&minr,1,MPI_DOUBLE,MPI_MIN,comm);
  MPI_Allreduce(&allreduce_ct,&maxr,1,MPI_DOUBLE,MPI_MAX,comm);
  MPI_Allreduce(&allreduce_ct,&totr,1,MPI_DOUBLE,MPI_SUM,comm);
  if (minr) ratio = maxr/minr; else ratio = 0.0;
  PetscFPrintf(comm,fd,"MPI Reductions:       %5.3e   %6.1f\n",maxr,ratio);


  if (EventsStageMax) {
    double mcounts,mlens,rcounts;

    PetscFPrintf(comm,fd,"\nSummary of Stages:  ---- Time ------     ----- Flops -------    -- Messages -- -- Message-lengths -- Reductions --\n");
    PetscFPrintf(comm,fd,"                      Avg      %%Total        Avg       %%Total   counts   %%Total    avg      %%Total   counts  %%Total \n");
    for ( j=0; j<=EventsStageMax; j++ ) {
      MPI_Allreduce(&EventsStageFlops[j],&sflops,1,MPI_DOUBLE,MPI_SUM,comm);
      MPI_Allreduce(&EventsStageTime[j],&sstime,1,MPI_DOUBLE,MPI_SUM,comm);
      if (tott > 0.) pstime = 100.0*sstime/tott; else pstime = 0.0;
      if (pstime >= 99.9 ) pstime = 99.9;
      if (totf > 0.) psflops = 100.*sflops/totf; else psflops = 0.0; 
      if (sstime > 0) psflops1 = (size*sflops)/sstime; else psflops1 = 0.0;

      MPI_Allreduce(&EventsStageMessageCounts[j],&mcounts,1,MPI_DOUBLE,MPI_SUM,comm);
      MPI_Allreduce(&EventsStageMessageLengths[j],&mlens,1,MPI_DOUBLE,MPI_SUM,comm);
      MPI_Allreduce(&EventsStageReductions[j],&rcounts,1,MPI_DOUBLE,MPI_SUM,comm);
      mcounts = .5*mcounts; mlens = .5*mlens; rcounts = rcounts/size;
      if (totm)  rat1 = 100.*mcounts/totm; else rat1 = 0.0; if (rat1 >= 99.9) rat1 = 99.9;
      if (totml) rat2 = 100.*mlens/totml; else rat2 = 0.0;  if (rat2 >= 99.9) rat2 = 99.9;
      if (totr)  rat3 = 100.*size*rcounts/totr; else rat3 = 0.0;if (rat2 >= 100.0) rat2 = 99.9;
      if (mcounts) mlensmcounts = mlens/mcounts; else mlensmcounts = 0.0; 
      if (EventsStageName[j]) {
        PetscFPrintf(comm,fd," %d: %15s: %5.3e    %4.1f%%     %5.3e      %4.1f%%  %5.3e   %4.1f%%  %5.3e  %4.1f%%  %5.3e  %4.1f%% \n",
                j,EventsStageName[j],sstime/size,pstime,psflops1,psflops,mcounts,rat1,mlensmcounts,rat2,
                rcounts,rat3);
      } else {
        PetscFPrintf(comm,fd," %d:                 %5.3e    %4.1f%%     %5.3e      %4.1f%%  %5.3e   %4.1f%%  %5.3e  %4.1f%%  %5.3e  %4.1f%% \n",
                j,sstime/size,pstime,psflops1,psflops,mcounts,rat1,mlensmcounts,rat2,rcounts,rat3);
      }
    }
  }


  PetscFPrintf(comm,fd,  
    "\n------------------------------------------------------------------------------------------------------------------------\n"); 
  PetscFPrintf(comm,fd,"Phase summary info:\n");
  PetscFPrintf(comm,fd,"   Count: number of times phase was executed\n");
  PetscFPrintf(comm,fd,"   Time and Flops/sec: Max - maximum over all processors\n");
  PetscFPrintf(comm,fd,"                       Ratio - ratio of maximum to minimum over all processors\n");
  PetscFPrintf(comm,fd,"   Mess: number of messages sent\n");
  PetscFPrintf(comm,fd,"   Avg. len: average message length\n");
  PetscFPrintf(comm,fd,"   Reduct: number of global reductions\n");
  PetscFPrintf(comm,fd,"   Global: entire computation\n");
  PetscFPrintf(comm,fd,"   Stage: optional user-defined stages of a computation. Set stages with PLogStagePush() and PLogStagePop().\n");
  PetscFPrintf(comm,fd,"      %%T - percent time in this phase         %%F - percent flops in this phase\n");
  PetscFPrintf(comm,fd,"      %%M - percent messages in this phase     %%L - percent message lengths in this phase\n");
  PetscFPrintf(comm,fd,"      %%R - percent reductions in this phase\n");
  PetscFPrintf(comm,fd,"   Total Mflop/s: 10e-6 * (sum of flops over all processors)/(max time over all processors)\n");
  PetscFPrintf(comm,fd,
    "------------------------------------------------------------------------------------------------------------------------\n"); 

  /* loop over operations looking for interesting ones */
  PetscFPrintf(comm,fd,"Phase              Count      Time (sec)       Flops/sec\
                          --- Global ---  --- Stage ---   Total\n");
  PetscFPrintf(comm,fd,"                            Max    Ratio      Max    Ratio\
  Mess  Avg len  Reduct %%T %%F %%M %%L %%R  %%T %%F %%M %%L %%R Mflop/s\n");
  PetscFPrintf(comm,fd,
    "------------------------------------------------------------------------------------------------------------------------\n"); 
  for ( j=0; j<=EventsStageMax; j++ ) {
    MPI_Allreduce(&EventsStageFlops[j],&sflops,1,MPI_DOUBLE,MPI_SUM,comm);
    MPI_Allreduce(&EventsStageTime[j],&sstime,1,MPI_DOUBLE,MPI_SUM,comm);
    if (EventsStageMax) {
      if (EventsStageName[j]) {
        PetscFPrintf(comm,fd,"\n--- Event Stage %d: %s\n\n",j,EventsStageName[j]);
      } else {
        PetscFPrintf(comm,fd,"\n--- Event Stage %d:\n\n",j);
      }
    }
    /* This loop assumes that PLOG_USER_EVENT_HIGH is the max event number */
    for ( i=0; i<PLOG_USER_EVENT_HIGH; i++ ) {  
      if (EventsType[j][i][TIME]) {
        wdou = EventsType[j][i][FLOPS]/EventsType[j][i][TIME];
      }
      else wdou = 0.0;
      MPI_Allreduce(&wdou,&minf,1,MPI_DOUBLE,MPI_MIN,comm);
      MPI_Allreduce(&wdou,&maxf,1,MPI_DOUBLE,MPI_MAX,comm);
      MPI_Allreduce(&EventsType[j][i][FLOPS],&totff,1,MPI_DOUBLE,MPI_SUM,comm);
      MPI_Allreduce(&EventsType[j][i][TIME],&mint,1,MPI_DOUBLE,MPI_MIN,comm);
      MPI_Allreduce(&EventsType[j][i][TIME],&maxt,1,MPI_DOUBLE,MPI_MAX,comm);
      MPI_Allreduce(&EventsType[j][i][TIME],&totts,1,MPI_DOUBLE,MPI_SUM,comm);
      MPI_Allreduce(&EventsType[j][i][MESSAGES],&mp,1,MPI_DOUBLE,MPI_SUM,comm);
      MPI_Allreduce(&EventsType[j][i][LENGTHS],&lp,1,MPI_DOUBLE,MPI_SUM,comm);
      MPI_Allreduce(&EventsType[j][i][REDUCTIONS],&rp,1,MPI_DOUBLE,MPI_SUM,comm);

      if (maxt) flopr = totff/maxt; else flopr = 0.0;
      MPI_Allreduce(&EventsStageMessageCounts[j],&totms,1,MPI_DOUBLE,MPI_SUM,comm);
      MPI_Allreduce(&EventsStageMessageLengths[j],&totmls,1,MPI_DOUBLE,MPI_SUM,comm);
      MPI_Allreduce(&EventsStageReductions[j],&totrs,1,MPI_DOUBLE,MPI_SUM,comm);

      if (EventsType[j][i][COUNT]) {
        if (mint > 0.0) rat = maxt/mint; else rat = 0.0;
        if (minf > 0.0) ratf = maxf/minf; else ratf = 0.0;
        if (tott > 0.0) ptotts = 100.*totts/tott; else ptotts = 0.0;
        if (ptotts >= 99. ) ptotts = 99.;
        if (totf > 0.0) ptotff = 100.*totff/totf; else ptotff = 0.0;
        if (ptotff >= 99. ) ptotff = 99.;
        if (sstime > 0.0) ptotts_stime = 100.*totts/sstime; else  ptotts_stime = 0.0;
        if (ptotts_stime >= 99. ) ptotts_stime = 99.;
        if (sflops > 0.0) ptotff_sflops = 100.*totff/sflops; else ptotff_sflops = 0.0;
        if (ptotff_sflops >= 99. ) ptotff_sflops = 99.;
        if (totm) mpg = 100.*mp/totm; else mpg = 0.0; if (mpg >= 99.) mpg = 99.;
        if (totml)  lpg = 100.*lp/totml; else lpg = 0.0; if (lpg >= 99.) lpg = 99.;
        if (totr)  rpg = 100.*rp/totr; else rpg = 0.0;if (rpg >= 99.) rpg = 99.;
        if (totms) mps = 100.*mp/totms; else mps = 0.0; if (mps >= 99.) mps = 99.;
        if (totmls)  lps = 100.*lp/totmls; else lps = 0.0; if (lps >= 99.) lps = 99.;
        if (totrs)  rps = 100.*rp/totrs; else rps = 0.0;if (rps >= 99.) rps = 99.;
        if (mp) lpmp = lp/mp; else lpmp = 0.0;
        mp = mp/2.0;
        rp = rp/((double) size);
        PetscFPrintf(comm,fd,"%s %7d %4.3e %6.1f  %2.1e %6.1f %2.1e %2.1e %2.1e %2.0f %2.0f %2.0f %2.0f %2.0f  %2.0f %2.0f %2.0f %2.0f %2.0f %5.0f\n",
                    PLogEventName[i],(int)EventsType[j][i][COUNT],maxt,rat,maxf,ratf,
                    mp,lpmp,rp,ptotts,ptotff,mpg,lpg,rpg,ptotts_stime,ptotff_sflops,mps,lps,rps,flopr/1.e6);
      }
    }
  }

  PetscFPrintf(comm,fd,
    "------------------------------------------------------------------------------------------------------------------------\n"); 
  PetscFPrintf(comm,fd,"\n"); 
  PetscFPrintf(comm,fd,"Memory usage is given in bytes:\n\n");

  /* loop over objects looking for interesting ones */
  PetscFPrintf(comm,fd,"Object Type      Creations   Destructions   Memory  Descendants' Mem.\n");
  for ( i=0; i<18; i++ ) {
    if (ObjectsType[i][0]) {
      PetscFPrintf(comm,fd,"%s %5d          %5d  %9d     %g\n",oname[i],(int) 
          ObjectsType[i][0],(int)ObjectsType[i][1],(int)ObjectsType[i][2],
          ObjectsType[i][3]);
    }

  }
  PetscFPrintf(comm,fd,"\n");

  return 0;
}

/*@C
   PetscGetFlops - Returns the number of flops used on this processor 
   since the program began. 

   Output Parameter:
   returns the number of flops as a double.

   Notes:
   A global counter logs all PETSc flop counts.  The user can use
   PLogFlops() to increment this counter to include flops for the 
   application code.  

   PETSc automatically logs library events if the code has been
   compiled with -DPETSC_LOG (which is the default), and -log,
   -log_summary, or -log_all are specified.  PLogFlops() is
   intended for logging user flops to supplement this PETSc
   information.

.keywords: log, flops, floating point operations

.seealso: PetscGetTime(), PLogFlops()
@*/
double PetscGetFlops()
{
  return _TotalFlops;
}

/* --------- Activate version -------------  */

/*@
    PLogEventActivateClass - Activates event logging for a PETSc object
        class.

  Input Parameter:
.    cookie - for example MAT_COOKIE, SNES_COOKIE,

.seealso: PLogInfoActivate(),PLogInfo(),PLogInfoAllow(),PLogEventDeactivateClass(),
          PLogEventActivate(),PLogEventDeactivate()
@*/
int PLogEventActivateClass(int cookie)
{
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
    PLogEventActivate(MAT_GetReordering);
    PLogEventActivate(MAT_MultTrans);
    PLogEventActivate(MAT_MultAdd);
    PLogEventActivate(MAT_MultTransAdd);
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
    PLogEventActivate(MAT_SolveTrans);
    PLogEventActivate(MAT_SolveTransAdd);
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
  return 0;
}

/*@
    PLogEventDeactivateClass - Deactivates event logging for a PETSc object
        class.

  Input Parameter:
.    cookie - for example MAT_COOKIE, SNES_COOKIE,

.seealso: PLogInfoActivate(),PLogInfo(),PLogInfoAllow(),PLogEventActivateClass(),
          PLogEventActivate(),PLogEventDeactivate()
@*/
int PLogEventDeactivateClass(int cookie)
{
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
    PLogEventDeactivate(MAT_GetReordering);
    PLogEventDeactivate(MAT_MultTrans);
    PLogEventDeactivate(MAT_MultAdd);
    PLogEventDeactivate(MAT_MultTransAdd);
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
    PLogEventDeactivate(MAT_SolveTrans);
    PLogEventDeactivate(MAT_SolveTransAdd);
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
  return 0;
}



/* end of -DPETSC_LOG section */
#else  /* -------------------------------------------------------------*/

int PLogObjectState(PetscObject obj,char *format,...)
{
  return 0;
}

#endif

/*@C
   PetscGetTime - Returns the current time of day in seconds. This 
     returns wall-clock time.  

   Output Parameter:
.  v - time counter

   Synopsis:
   double PetscGetTime()

   Usage: 
$     double v;
$     v = PetscGetTime();
$     .... perform some calculation ...
$     v = PetscGetTime() -v;

   Notes:
   Since the PETSc libraries incorporate timing of phases and operations, 
   PetscGetTime() is intended only for timing of application codes.  
   The options database commands -log, -log_summary, and -log_all activate
   PETSc library timing.  See the users manual for further details.

.seealso: PLogEventRegister(), PLogEventBegin(), PLogEventEnd(),  PLogStagePush(), 
          PLogStagePop(), PLogStageRegister(), PetscGetFlops()

.keywords:  get, time
@*/
double PetscGetTime()
{
  double t;
  PetscTime(t);
  return t;
}
