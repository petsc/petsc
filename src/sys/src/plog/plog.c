
#ifndef lint
static char vcid[] = "$Id: plog.c,v 1.49 1995/11/20 04:46:40 bsmith Exp bsmith $";
#endif
/*
      PETSc code to log object creation and destruction and PETSc events.
*/
#include "petsc.h"        /*I    "petsc.h"   I*/
#include <stdio.h>
#include <stdarg.h>
#include <sys/types.h>
#include "pinclude/petscfix.h"
#include "ptime.h"

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

static int     EventsStage = 0; /* which log sessions are we putting in. */
static int     EventsStageMax = 0; /* highest event log used */ 
static int     EventsStagePushed = 0;
static int     EventsStageStack[100];
#define COUNT 0
#define FLOPS 1
#define TIME  2
static double  EventsType[10][100][3];

/*@
     PLogPushStage - You can log up to 10 stages of your code using PLogBegin() or
                    -log_summary. One switches  stages with this command.

  Input Parameters:
.   stage - stage to log on, 0 to 9

@*/
int PLogPushStage(int stage)
{
  if (stage < 0 || stage > 10) SETERRQ(1,"PLogPushStage:Out of range");
  EventsStageStack[EventsStagePushed++] = EventsStage;
  if (EventsStagePushed++ > 99) SETERRQ(1,"PLogPushStage:Too many pushes");
  EventsStage = stage;
  if (stage > EventsStageMax) EventsStageMax = stage;
  return 0;
}

int PLogPopStage()
{
  if (EventsStagePushed < 1) return 0;
  EventsStage = EventsStageStack[--EventsStagePushed];
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
  return 0;
}

/*@C
   PLogDump - Dumps logs of objects to a file. This file is intended to 
   be read by petsc/bin/petscsim; it is not user friendly.

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
                         "MatMultTransAdd ",
                         "MatLUFactor     ",
                         "MatCholeskyFacto",
                         "MatLUFactorSymbo",
                         "MatILUFactorSymb",
                         "MatCholeskyFacto",
                         "MatIncompleteCho",
                         "MatLUFactorNumer",
                         "MatCholeskyFacto",
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
                         "SNESMinFunctEval",
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
   combination with petsc/bin/petscsim.
   
.keywords: log, dump, print

.seealso: PLogBegin(), PLogDump()
@*/
int PLogPrint(MPI_Comm comm,FILE *fd)
{
  double maxo,mino,aveo,mem,totmem,maxmem,minmem;
  double maxf,minf,avef,totf,_TotalTime,maxt,mint,avet,tott;
  double fmin,fmax,ftot,wdou,totts,totff;
  int    size,i,j;

  MPI_Comm_size(comm,&size);

  PetscTime(_TotalTime);  _TotalTime -= BaseTime;

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

  MPIU_fprintf(comm,fd,"\nPerformance Summary:\n");

  MPIU_fprintf(comm,fd,"\n                Max         Min        Avg        Total \n");
  MPIU_fprintf(comm,fd,"Time:        %5.3e   %5.3e   %5.3e\n",maxt,mint,avet);
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

  MPIU_fprintf(comm,fd,
    "\n------------------------------------------------------------------------------\n"); 

  /* loop over operations looking for interesting ones */
  MPIU_fprintf(comm,fd,"\nPhase               Count       Time (sec)        \
   Flops/sec     %%Time %%Flop\n");
  MPIU_fprintf(comm,fd,"                             Min       Max      \
  Min       Max\n");
  for ( j=0; j<=EventsStageMax; j++ ) {
    for ( i=0; i<100; i++ ) {  
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
        if (!tott) tott = 1.e-5;
        MPIU_fprintf(comm,fd,"%s %8d  %3.2e  %3.2e   %3.2e  %3.2e %5.1f %5.1f\n",
                     name[i],(int)EventsType[j][i][COUNT],mint,maxt,minf,maxf,
                    100.*totts/tott,100.*totff/totf);
      }
    }
    MPIU_fprintf(comm,fd,
    "------------------------------------------------------------------------------\n"); 
  }

  MPIU_fprintf(comm,fd,"\n"); 
  MPIU_fprintf(comm,fd,"Memory usage is given in bytes:\n\n");

  /* loop over objects looking for interesting ones */
  MPIU_fprintf(comm,fd,"Object Type      Creations   Destructions   Memory  Descendants' Mem.\n");
  for ( i=0; i<15; i++ ) {
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






