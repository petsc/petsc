
#ifndef lint
static char vcid[] = "$Id: plog.c,v 1.57 1996/01/09 03:32:09 curfman Exp curfman $";
#endif
/*
      PETSc code to log object creation and destruction and PETSc events.
*/
#include "petsc.h"        /*I    "petsc.h"   I*/
#include <stdio.h>
#include <stdarg.h>
#include <sys/types.h>
#include "sys.h"
#include "pinclude/petscfix.h"
#include "pinclude/ptime.h"

/*@C 
   PetscObjectSetName - Sets a string name associated with a PETSc object.

   Input Parameters:
.  obj - the Petsc variable
.  name - the name to give obj

.keywords: object, set, name

.seealso: PetscObjectGetName()
@*/
int PetscObjectSetName(PetscObject obj,char *name)
{
  if (!obj) SETERRQ(1,"PetscObjectSetName:Null object");
  obj->name = name;
  return 0;
}

/*@C
   PetscObjectGetName - Gets a string name associated with a PETSc object.

   Input Parameters:
.  obj - the Petsc variable
.  name - the name associated with obj

.keywords: object, get, name

.seealso: PetscObjectSetName()
@*/
int PetscObjectGetName(PetscObject obj,char **name)
{
  if (!obj) SETERRQ(1,"PetscObjectGetName:Null object");
  if (!name) SETERRQ(1,"PetscObjectGetName:Void location for name");
  *name = obj->name;
  return 0;
}

static int PrintInfo = 0;

/*@C
    PLogAllowInfo - Causes PLogInfo messages to be printed to standard output.

    Input Parameter:
.   flag - PETSC_TRUE or PETSC_FALSE

    Options Database Key:
$  -info 

.keywords: allow, information, printing, monitoring
@*/
int PLogAllowInfo(PetscTruth flag)
{
  PrintInfo = (int) flag;
  return 0;
}

extern FILE *petsc_history;

/* This is a temporary shell until we devise a complete version */
int PLogInfo(PetscObject obj,char *format,...)
{
  va_list Argp;
  int     rank;
  if (!PrintInfo) return 0;
  if (!obj) rank = 0;
  else      {MPI_Comm_rank(obj->comm,&rank);} 
  if (rank) return 0;
  va_start( Argp, format );
  vfprintf(stdout,format,Argp);
  if (petsc_history) {
    vfprintf(petsc_history,format,Argp);
  }
  va_end( Argp );
  return 0;
}

/* -------------------------------------------------------------------*/
#if defined(PETSC_LOG)

#define CHUNCK       1000
#define CREATE       0
#define DESTROY      1
#define ACTIONBEGIN  2
#define ACTIONEND    3

typedef struct {
  double      time;
  int         cookie,type,event,id1,id2,id3;
} Events;

typedef struct {
  int         parent;
  double      mem;
  char        string[64];
  char        name[32];
  PetscObject obj;
} Objects;

static double  BaseTime;
       double _TotalFlops = 0;
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
static double  EventsStageFlops[] = {0,0,0,0,0,0,0,0,0,0};
static double  EventsStageTime[] = {0,0,0,0,0,0,0,0,0,0};
#define COUNT 0
#define FLOPS 1
#define TIME  2
static double  EventsType[10][PLOG_USER_EVENT_HIGH][3];

/*@
    PLogStageRegister - Attach a charactor string name to a logging stage.

 Input Parameters:
.  stage - the stage from 0 to 9 inclusive
.  name - the name to associate with that stage

.seealso: PLogStagePush(), PLogStagePop()
@*/
int PLogStageRegister(int stage, char *name)
{
  if (stage < 0 || stage > 10) SETERRQ(1,"PLogStageRegister:Out of range");
  EventsStageName[stage] = name;
  return 0;
}

/*@
   PLogStagePush - Users can log up to 10 stages within a code by using
   -log_summary in conjunction with PLogStagePush() and PLogStagePop().

   Input Parameters:
.  stage - stage on which to log (0 <= stage <= 9)

   Example of Usage:
   If the option -log_sumary is used to run the program containing the 
   following code, then 3 sets of summary data will be printed during
   PetscFinalize().
$
$     PetscInitialize(int *argc,char ***args,0,0,0);
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

.keywords: log, push, stage

.seealso: PLogStagePop()
@*/
int PLogStagePush(int stage)
{
  if (stage < 0 || stage > 10) SETERRQ(1,"PLogStagePush:Out of range");
  /* record flops/time of previous stage */
  if (EventsStagePushed) {
    PetscTimeAdd(EventsStageTime[EventsStage]);
    EventsStageFlops[EventsStage] += _TotalFlops;
  }
  EventsStageStack[EventsStagePushed++] = EventsStage;
  if (EventsStagePushed++ > 99) SETERRQ(1,"PLogStagePush:Too many pushes");
  EventsStage = stage;
  if (stage > EventsStageMax) EventsStageMax = stage;
  PetscTimeSubtract(EventsStageTime[EventsStage]);
  EventsStageFlops[EventsStage] -= _TotalFlops;
  return 0;
}

/*@
   PLogStagePop - Users can log up to 10 stages within a code by using
   -log_summary in conjunction with PLogStagePush() and PLogStagePop().

   Example of Usage:
   If the option -log_sumary is used to run the program containing the 
   following code, then 2 sets of summary data will be printed during
   PetscFinalize().
$
$     PetscInitialize(int *argc,char ***args,0,0,0);
$     [stage 0 of code]   
$     PLogStagePush(1);
$     [stage 1 of code]
$     [some code (stage 1)]
$     PLogStagePop();
$     [more stage 0 of code]   
$     PetscFinalize();
$

.keywords: log, pop, stage

.seealso: PLogStagePush()
@*/
int PLogStagePop()
{
  PetscTimeAdd(EventsStageTime[EventsStage]);
  EventsStageFlops[EventsStage] += _TotalFlops;
  if (EventsStagePushed < 1) SETERRQ(1,"PLogStagePop:Too many pops\n");
  EventsStage = EventsStageStack[--EventsStagePushed];
  if (EventsStagePushed) {
    PetscTimeSubtract(EventsStageTime[EventsStage]);
    EventsStageFlops[EventsStage] -= _TotalFlops;
  }
  return 0;
}

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
    tmp = (Events *) PetscMalloc((eventsspace+CHUNCK)*sizeof(Events));CHKPTRQ(tmp);
    PetscMemcpy(tmp,events,eventsspace*sizeof(Events));
    PetscFree(events);
    events = tmp; eventsspace += CHUNCK;
  }
  if (nobjects >= objectsspace) {
    Objects *tmp;
    tmp = (Objects *) PetscMalloc((objectsspace+CHUNCK)*sizeof(Objects));CHKPTRQ(tmp);
    PetscMemcpy(tmp,objects,objectsspace*sizeof(Objects));
    PetscFree(objects);
    objects = tmp; objectsspace += CHUNCK;
  }
  PetscTime(events[nevents].time); events[nevents].time -= BaseTime;
  events[nevents].cookie  = obj->cookie - PETSC_COOKIE - 1;
  events[nevents].type    = obj->type;
  events[nevents].id1     = nobjects;
  events[nevents].id2     = -1;
  events[nevents].id3     = -1;
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
    tmp = (Events *) PetscMalloc((eventsspace+CHUNCK)*sizeof(Events));CHKPTRQ(tmp);
    PetscMemcpy(tmp,events,eventsspace*sizeof(Events));
    PetscFree(events);
    events = tmp; eventsspace += CHUNCK;
  }
  PetscTime(events[nevents].time); events[nevents].time -= BaseTime;
  events[nevents].event     = DESTROY;
  events[nevents].cookie    = obj->cookie - PETSC_COOKIE - 1;
  events[nevents].type      = obj->type;
  events[nevents].id1       = obj->id;
  events[nevents].id2       = -1;
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
    tmp = (Events *) PetscMalloc((eventsspace+CHUNCK)*sizeof(Events));CHKPTRQ(tmp);
    PetscMemcpy(tmp,events,eventsspace*sizeof(Events));
    PetscFree(events);
    events = tmp; eventsspace += CHUNCK;
  }
  PetscTime(ltime);
  events[nevents].time = ltime - BaseTime;
  events[nevents].id1     = o1->id;
  if (o2) events[nevents].id2     = o2->id; else events[nevents].id2 = -1;
  if (o3) events[nevents].id3     = o3->id; else events[nevents].id3 = -1;
  events[nevents].type   = event;
  events[nevents].cookie = 0;
  events[nevents++].event= ACTIONBEGIN;
  if (t != 1) return 0;
  EventsType[EventsStage][event][COUNT]++;
  EventsType[EventsStage][event][TIME]  -= ltime;
  EventsType[EventsStage][event][FLOPS] -= _TotalFlops;
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
    tmp = (Events *) PetscMalloc((eventsspace+CHUNCK)*sizeof(Events));CHKPTRQ(tmp);
    PetscMemcpy(tmp,events,eventsspace*sizeof(Events));
    PetscFree(events);
    events = tmp; eventsspace += CHUNCK;
  }
  PetscTime(ltime);
  events[nevents].time   = ltime - BaseTime;
  events[nevents].id1    = o1->id;
  if (o2) events[nevents].id2    = o2->id; else events[nevents].id2 = -1;
  if (o3) events[nevents].id3    = o3->id; else events[nevents].id3 = -1;
  events[nevents].type   = event;
  events[nevents].cookie = 0;
  events[nevents++].event= ACTIONEND;
  if (t != 1) return 0;
  EventsType[EventsStage][event][TIME] += ltime;
  EventsType[EventsStage][event][FLOPS] += _TotalFlops;
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
  EventsType[EventsStage][event][FLOPS] -= _TotalFlops;
  return 0;
}
/*
     Default event end logger
*/
int ple(int event,int t,PetscObject o1,PetscObject o2,PetscObject o3,PetscObject o4)
{
  if (t != 1) return 0;
  PetscTimeAdd(EventsType[EventsStage][event][TIME]);
  EventsType[EventsStage][event][FLOPS] += _TotalFlops;
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
  objects = (Objects*) PetscMalloc(CHUNCK*sizeof(Objects));CHKPTRQ(objects);
  events  = (Events*) PetscMalloc(CHUNCK*sizeof(Events));CHKPTRQ(events);
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
   This routine should be employed when the user wants to dump summary 
   information for multiple segments of code within one PETSc program.  

   Example of Usage:
   If the option -log_sumary is used to run the program containing the 
   following code, then 2 sets of summary data will be printed (one during 
   PLogPrint and one during PetscFinalize, which in turn calls PLogPrint).
$
$     PetscInitialize(int *argc,char ***args,0,0,0);
$     [section 1 of code]
$     PLogPrint(MPI_COMM_WORLD,stdout);
$     PLogDestroy();
$     PLogBegin();
$     [section 2 of code]
$     PetscFinalize();

.keywords: log, destroy

.seealso: PLogDump(), PLogAllBegin(), PLogPrint()
@*/
int PLogDestroy()
{
  /* Destroying phase */
  if (objects) {PetscFree(objects); objects = 0;}
  if (events)  {PetscFree(events); events = 0;}
  _PHC             = 0;
  _PHD             = 0;

  /* Resetting phase */
  PetscMemzero(EventsType,sizeof(EventsType));
  PetscMemzero(ObjectsType,sizeof(ObjectsType));
  _TotalFlops      = 0;
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

.seealso: PLogDump(), PLogAllBegin(), PLogPrint()
@*/
int PLogBegin()
{
  objects = (Objects*) PetscMalloc(CHUNCK*sizeof(Objects));CHKPTRQ(objects);
  events  = (Events*) PetscMalloc(CHUNCK*sizeof(Events));CHKPTRQ(events);
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
   be read by petsc/bin/petscview; it is not user friendly.

   Input Parameter:
.  name - an optional file name

   Notes:
   The default file name is 
$      Log.<rank>
   where <rank> is the processor number. If no name is specified, 
   this file will be used.

   Options Database Keys:
$  -log : Prints basic log information (for code compiled 
$      with PETSC_LOG)
$  -log_all : Prints extensive log information (for code compiled
$      with PETSC_LOG)
   
.keywords: log, dump

.seealso: PLogBegin(), PLogPrint()
@*/
int PLogDump(char* name)
{
  int    i,rank;
  FILE   *fd;
  char   file[64];
  double flops,_TotalTime;
  
  PetscTime(_TotalTime);
  _TotalTime -= BaseTime;

  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  if (name) sprintf(file,"%s.%d",name,rank);
  else  sprintf(file,"Log.%d",rank);
  fd = fopen(file,"w"); if (!fd) SETERRQ(1,"PlogDump:cannot open file");

  fprintf(fd,"Objects created %d Destroyed %d\n",nobjects,
                                                 ObjectsDestroyed);
  fprintf(fd,"Clock Resolution %g\n",0.0);
  fprintf(fd,"Events %d\n",nevents);
  for ( i=0; i<nevents; i++ ) {
    fprintf(fd,"%g %d %d %d %d %d %d\n",events[i].time,
                              events[i].event,
                              events[i].cookie,events[i].type,events[i].id1,
                              events[i].id2,events[i].id3);
  }
  for ( i=0; i<nobjects; i++ ) {
    fprintf(fd,"%d %d\n",objects[i].parent,(int)objects[i].mem);
    if (!objects[i].string[0]) {fprintf(fd,"No Info\n");}
    else fprintf(fd,"%s\n",objects[i].string);
    if (!objects[i].name[0]) {fprintf(fd,"No Name\n");}
    else fprintf(fd,"%s\n",objects[i].name);
  }
  for ( i=0; i<100; i++ ) {
    flops = 0.0;
    if (EventsType[0][i][TIME]){flops = EventsType[0][i][FLOPS]/EventsType[0][i][TIME];}
    fprintf(fd,"%d %16g %16g %16g %16g\n",i,EventsType[0][i][COUNT],
                      EventsType[0][i][FLOPS],EventsType[0][i][TIME],flops);
  }
  fprintf(fd,"Total Flops %14e %16.8e\n",_TotalFlops,_TotalTime);
  fclose(fd);
  return 0;
}

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
static char *(name[]) = {"MatMult         ",
                         "MatAssemblyBegin",
                         "MatAssemblyEnd  ",
                         "MatGetReordering",
                         "MatMultTrans    ",
                         "MatMultAdd      ",
                         "MatMltTrnsAdd   ",
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
                         "MatGetSubMatrix ",
                         "MatGetSubMatrice",
                         "MatGetValues    ",
                         "                ",
                         "                ",
                         "                ",
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
                         "VecPMult        ",
                         "VecSetValues    ",
                         "VecLoad         ",
                         "VecView         ",
                         "VecScatterBegin ",
                         "VecScatterEnd   ",
                         " ",
                         " ",
                         " ",
                         " ",
                         " ",
                         " ",
                         " ",
                         "SLESSolve       ",
                         "SLESSetUp       ",
                         "KSPGMRESOrthog  ",
                         "KSPSolve        ",
                         " ",
                         "PCSetUp         ",
                         "PCApply         ",
                         " ",
                         " ",
                         " ",
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
                         " "," "," "," "," ",
                         " "," "," "," "," ",
                         " "," "," "," "," ",
                         " "," "," "," "," ",
                         " "," "," "," "," "};

/*@C
    PLogEventRegister - Registers an event name for logging operations in 
    an application code. 

    Input Parameters:
.   e - integer associated with the event (PLOG_USER_EVENT_LOW <= e < PLOG_USER_EVENT_HIGH) 
.   string - name associated with the event

    Notes: 
    PETSc automatically logs library events if the code has been
    compiled with -DPETSC_LOG (which is the default) and -log,
    -log_summary, or -log_all are specified.  PLogEventRegister() is
    intended for logging user events to supplement this PETSc
    information.

    Example of Usage:
$     #define USER_EVENT PLOG_USER_EVENT_LOW
$     int user_event_flops;
$     PLogEventRegister(USER_EVENT,"User event");
$     PLogEventBegin(USER_EVENT,0,0,0,0);
$        [code segment to monitor]
$        PLogFlops(user_event_flops);
$     PLogEventEnd(USER_EVENT,0,0,0,0);

.keywords: log, event, register

.seealso: PLogEventBegin(), PLogEventEnd(), PLogFlops()
@*/
int PLogEventRegister(int e,char *string)
{
  if (e < PLOG_USER_EVENT_LOW) 
    SETERRQ(1,"PLogEventRegister:user events must be >= PLOG_USER_EVENT_LOW");
  if (e > PLOG_USER_EVENT_HIGH) 
    SETERRQ(1,"PLogEventRegister:user events must be < PLOG_USER_EVENT_HIGH ");
  name[e] = string;
  return 0;
}
  
/*@C
   PLogPrint - Prints a summary of the logging.

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
int PLogPrint(MPI_Comm comm,FILE *fd)
{
  double maxo,mino,aveo,mem,totmem,maxmem,minmem;
  double maxf,minf,avef,totf,_TotalTime,maxt,mint,avet,tott;
  double fmin,fmax,ftot,wdou,totts,totff,rat,stime,sflops,ratf;
  int    size,i,j;
  char   arch[10];

  /* pop off any stages the user forgot to remove */
  while (EventsStagePushed) PLogStagePop();

  PetscTime(_TotalTime);  _TotalTime -= BaseTime;

  MPI_Comm_size(comm,&size);

  wdou = _TotalFlops; 
  MPI_Reduce(&wdou,&minf,1,MPI_DOUBLE,MPI_MIN,0,comm);
  MPI_Reduce(&wdou,&maxf,1,MPI_DOUBLE,MPI_MAX,0,comm);
  MPI_Reduce(&wdou,&totf,1,MPI_DOUBLE,MPI_SUM,0,comm);
  avef = (totf)/((double) size);
  wdou = nobjects;
  MPI_Reduce(&wdou,&mino,1,MPI_DOUBLE,MPI_MIN,0,comm);
  MPI_Reduce(&wdou,&maxo,1,MPI_DOUBLE,MPI_MAX,0,comm);
  MPI_Reduce(&wdou,&aveo,1,MPI_DOUBLE,MPI_SUM,0,comm);
  aveo = (aveo)/((double) size);
  wdou = _TotalTime;
  MPI_Reduce(&wdou,&mint,1,MPI_DOUBLE,MPI_MIN,0,comm);
  MPI_Reduce(&wdou,&maxt,1,MPI_DOUBLE,MPI_MAX,0,comm);
  MPI_Reduce(&wdou,&tott,1,MPI_DOUBLE,MPI_SUM,0,comm);
  avet = (tott)/((double) size);

  MPIU_fprintf(comm,fd,
   "\nPerformance Summary:----------------------------------------------------------\n");
  SYGetArchType(arch,10);
  if (size == 1)
    MPIU_fprintf(comm,fd,"Machine: %s with %d processor, run on %s",arch,size,SYGetDate());
  else
    MPIU_fprintf(comm,fd,"Machine: %s with %d processors, run on %s",arch,size,SYGetDate());

  MPIU_fprintf(comm,fd,"\n                Max         Min        Avg        Total \n");
  MPIU_fprintf(comm,fd,"Time (sec):  %5.3e   %5.3e   %5.3e\n",maxt,mint,avet);
  MPIU_fprintf(comm,fd,"Objects:     %5.3e   %5.3e   %5.3e\n",maxo,mino,aveo);
  MPIU_fprintf(comm,fd,"Flops:       %5.3e   %5.3e   %5.3e  %5.3e\n",
                                                 maxf,minf,avef,totf);

  if (mint) fmin = minf/mint; else fmin = 0;
  if (maxt) fmax = maxf/maxt; else fmax = 0;
  if (maxt) ftot = totf/maxt; else ftot = 0;
  MPIU_fprintf(comm,fd,"Flops/sec:   %5.3e   %5.3e              %5.3e\n",
                                               fmin,fmax,ftot);
  TrGetMaximumAllocated(&mem);
  if (mem > 0.0) {
    MPI_Reduce(&mem,&maxmem,1,MPI_DOUBLE,MPI_MAX,0,comm);
    MPI_Reduce(&mem,&minmem,1,MPI_DOUBLE,MPI_MIN,0,comm);
    MPI_Reduce(&mem,&totmem,1,MPI_DOUBLE,MPI_SUM,0,comm);
    MPIU_fprintf(comm,fd,"Memory:      %5.3e   %5.3e              %5.3e\n",
                                               minmem,maxmem,totmem);
 
  }

  if (!tott) tott = 1.e-5;

  if (EventsStageMax) {
    MPIU_fprintf(comm,fd,"\nSummary of Stages:     Avg Time  %%Total  Avg Flops/sec  %%Total\n");
    for ( j=0; j<=EventsStageMax; j++ ) {
      MPI_Reduce(&EventsStageFlops[j],&sflops,1,MPI_DOUBLE,MPI_SUM,0,comm);
      MPI_Reduce(&EventsStageTime[j],&stime,1,MPI_DOUBLE,MPI_SUM,0,comm);
      if (EventsStageName[j]) {
        MPIU_fprintf(comm,fd," %d: %15s:  %5.3e   %4.1f%%    %5.3e     %4.1f%% \n",
                     j,EventsStageName[j],stime/size,100.0*stime/tott,sflops/size,
                     100.*sflops/totf);
      } else {
        MPIU_fprintf(comm,fd," %d:          %5.3e   %4.1f%%    %5.3e     %4.1f%% \n",
                    j,stime/size,100.0*stime/tott,sflops/size,100.*sflops/totf);
      }
    }
  }

  MPIU_fprintf(comm,fd,  
    "\n------------------------------------------------------------------------------\n"); 
  MPIU_fprintf(comm,fd,"Phase summary info:\n");
  MPIU_fprintf(comm,fd,"   Count: number of times phase was executed\n");
  MPIU_fprintf(comm,fd,"   Time and Flops/sec:\n");
  MPIU_fprintf(comm,fd,"      Max - maximum over all processors\n");
  MPIU_fprintf(comm,fd,"      Ratio - ratio of maximum to minimum over all processors\n");
  MPIU_fprintf(comm,fd,"   Global: entire computation\n");
  MPIU_fprintf(comm,fd,"   Stage: optional user-defined stages of a computation\n");
  MPIU_fprintf(comm,fd,"          Set stages with PLogStagePush() and PLogStagePop().\n");
  MPIU_fprintf(comm,fd,"      %%T - percent time in this phase\n");
  MPIU_fprintf(comm,fd,"      %%F - percent flops in this phase\n");
  MPIU_fprintf(comm,fd,
    "------------------------------------------------------------------------------\n"); 

  /* loop over operations looking for interesting ones */
  MPIU_fprintf(comm,fd,"Phase            Count    Time (sec)      Flops/sec\
      Global       Stage\n");
  MPIU_fprintf(comm,fd,"                        Max    Ratio     Max    Ratio\
     %%T %%F       %%T %%F\n");
  MPIU_fprintf(comm,fd,
    "------------------------------------------------------------------------------\n"); 
  for ( j=0; j<=EventsStageMax; j++ ) {
    MPI_Reduce(&EventsStageFlops[j],&sflops,1,MPI_DOUBLE,MPI_SUM,0,comm);
    MPI_Reduce(&EventsStageTime[j],&stime,1,MPI_DOUBLE,MPI_SUM,0,comm);
    if (EventsStageMax) {
      if (EventsStageName[j]) {
        MPIU_fprintf(comm,fd,"\n--- Event Stage %d: %s\n\n",j,EventsStageName[j]);
      } else {
        MPIU_fprintf(comm,fd,"\n--- Event Stage %d:\n\n",j);
      }
    }
    /* This loop assumes that PLOG_USER_EVENT_HIGH is the max event number */
    for ( i=0; i<PLOG_USER_EVENT_HIGH; i++ ) {  
      if (EventsType[j][i][TIME]) {
        wdou = EventsType[j][i][FLOPS]/EventsType[j][i][TIME];
      }
      else wdou = 0.0;
      MPI_Reduce(&wdou,&minf,1,MPI_DOUBLE,MPI_MIN,0,comm);
      MPI_Reduce(&wdou,&maxf,1,MPI_DOUBLE,MPI_MAX,0,comm);
      wdou = EventsType[j][i][FLOPS];
      MPI_Reduce(&wdou,&totff,1,MPI_DOUBLE,MPI_SUM,0,comm);
      wdou = EventsType[j][i][TIME];
      MPI_Reduce(&wdou,&mint,1,MPI_DOUBLE,MPI_MIN,0,comm);
      MPI_Reduce(&wdou,&maxt,1,MPI_DOUBLE,MPI_MAX,0,comm);
      MPI_Reduce(&wdou,&totts,1,MPI_DOUBLE,MPI_SUM,0,comm);
      if (EventsType[j][i][COUNT]) {
        if (mint > 0.0) rat = maxt/mint; else rat = 0.0;
        if (minf > 0.0) ratf = maxf/minf; else ratf = 0.0;
        MPIU_fprintf(comm,fd,"%s %4d  %2.1e %6.1f  %2.1e %6.1f   %4.1f %4.1f   %4.1f %4.1f\n",
                     name[i],(int)EventsType[j][i][COUNT],maxt,rat,maxf,ratf,
                    100.*totts/tott,100.*totff/totf,100.*totts/stime,100.*totff/sflops);
      }
    }
  }

  MPIU_fprintf(comm,fd,
    "------------------------------------------------------------------------------\n"); 
  MPIU_fprintf(comm,fd,"\n"); 
  MPIU_fprintf(comm,fd,"Memory usage is given in bytes:\n\n");

  /* loop over objects looking for interesting ones */
  MPIU_fprintf(comm,fd,"Object Type      Creations   Destructions   Memory  Descendants' Mem.\n");
  for ( i=0; i<18; i++ ) {
    if (ObjectsType[i][0]) {
      MPIU_fprintf(comm,fd,"%s %5d          %5d  %9d     %g\n",oname[i],(int) 
          ObjectsType[i][0],(int)ObjectsType[i][1],(int)ObjectsType[i][2],
          ObjectsType[i][3]);
    }

  }
  MPIU_fprintf(comm,fd,"\n");
  return 0;
}

#endif

/*@C 
   PetscGetTime - Returns the current time of day in seconds.  

   Output Parameter:
.  v - time counter

   Synopsis:
   double PetscGetTime()

   Usage: 
     double v;
     v = PetscGetTime();
     .... perform some calculation ...
     v = PetscGetTime() -v;

   Notes:
   Since the PETSc libraries incorporate timing of phases and operations, 
   PetscGetTime() is intended only for timing of application codes.  
   The options database commands -log, -log_summary, and -log_all activate
   PETSc library timing.  See the users manual for further details.

.seealso: PLogEventRegister(), PLogEventBegin(), PLogEventEnd(),  PLogStagePush(), 
          PLogStagePop(), PLogStageRegister().

.keywords:  Petsc, time
@*/
double PetscGetTime()
{
  double t;
  PetscTime(t);
  return t;
}
  




