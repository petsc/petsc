#ifndef lint
static char vcid[] = "$Id: plog.c,v 1.18 1995/07/12 16:17:49 curfman Exp curfman $";
#endif

#include "petsc.h"
#include "ptscimpl.h"
#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include <sys/types.h>
#include "petscfix.h"
#include "ptime.h"

/*@ 
   PetscObjectSetName - Sets a string name associated with a PETSc object.

   Input Parameters:
.  obj - the Petsc variable
.  name - the name to give obj

.keywords: object, set, name

.seealso: PetscObjectGetName()
@*/
int PetscObjectSetName(PetscObject obj,char *name)
{
  if (!obj) SETERRQ(1,"PetscObjectSetName: Null object");
  obj->name = name;
  return 0;
}

/*@ 
   PetscObjectGetName - Gets a string name associated with a PETSc object.

   Input Parameters:
.  obj - the Petsc variable
.  name - the name associated with obj

.keywords: object, get, name

.seealso: PetscObjectSetName()
@*/
int PetscObjectGetName(PetscObject obj,char **name)
{
  if (!obj) SETERRQ(1,"PetscObjectGetName: Null object");
  if (!name) SETERRQ(1,"PetscObjectGetName: Void location for name");
  *name = obj->name;
  return 0;
}

static int PrintInfo = 0;

/*@
    PLogAllowInfo - Causes PLogInfo messages to be printed to standard output.

    Input Parameter:
.   flag - PETSC_TRUE or PETSC_FALSE
@*/
int PLogAllowInfo(PetscTruth flag)
{
  PrintInfo = (int) flag;
  return 0;
}

/* This is a temporary shell until we devise a complete version */
int PLogInfo(PetscObject obj,char *format,...)
{
  va_list Argp;
  if (!PrintInfo) return 0;
  va_start( Argp, format );
  vfprintf(stdout,format,Argp);
  va_end( Argp );
  return 0;
}

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
  char        string[64];
  char        name[32];
  PetscObject obj;
} Objects;

static double  BaseTime;
static Events  *events = 0;
static Objects *objects = 0;
static int nobjects = 0, nevents = 0, objectsspace = CHUNCK;
static int ObjectsDestroyed = 0, eventsspace = CHUNCK;
#define COUNT 0
#define FLOPS 1
#define TIME  2
static double EventsType[100][3];
double _TotalFlops = 0;
int (*_PHC)(PetscObject) = 0;
int (*_PHD)(PetscObject) = 0;
int (*_PLB)(int,int,PetscObject,PetscObject,PetscObject,PetscObject);
int (*_PLE)(int,int,PetscObject,PetscObject,PetscObject,PetscObject);


/*
      Default object create logger 
*/
int phc(PetscObject obj)
{
  if (nevents >= eventsspace) {
    Events *tmp;
    tmp = (Events *) PETSCMALLOC( (eventsspace+CHUNCK)*sizeof(Events) );
    CHKPTRQ(tmp);
    PETSCMEMCPY(tmp,events,eventsspace*sizeof(Events));
    PETSCFREE(events);
    events = tmp; eventsspace += CHUNCK;
  }
  if (nobjects >= objectsspace) {
    Objects *tmp;
    tmp = (Objects *) PETSCMALLOC( (objectsspace+CHUNCK)*sizeof(Objects) );
    CHKPTRQ(tmp);
    PETSCMEMCPY(tmp,objects,objectsspace*sizeof(Objects));
    PETSCFREE(objects);
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
  PETSCMEMSET(objects[nobjects].string,0,64*sizeof(char));
  PETSCMEMSET(objects[nobjects].name,0,16*sizeof(char));
  obj->id = nobjects++;
  return 0;
}
/*
      Default object destroy logger 
*/
int phd(PetscObject obj)
{
  if (nevents >= eventsspace) {
    Events *tmp;
    tmp = (Events *) PETSCMALLOC( (eventsspace+CHUNCK)*sizeof(Events) );
    CHKPTRQ(tmp);
    PETSCMEMCPY(tmp,events,eventsspace*sizeof(Events));
    PETSCFREE(events);
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
  if (obj->name) { strncpy(objects[obj->id].name,obj->name,16);}
  objects[obj->id].obj      = 0;
  ObjectsDestroyed++;
  return 0;
}
/*
    Event begin logger with complete logging
*/
int plball(int event,int t,PetscObject o1,PetscObject o2,PetscObject o3,
                                                      PetscObject o4)
{
 double time;
 if (nevents >= eventsspace) {
    Events *tmp;
    tmp = (Events *) PETSCMALLOC( (eventsspace+CHUNCK)*sizeof(Events) );
    CHKPTRQ(tmp);
    PETSCMEMCPY(tmp,events,eventsspace*sizeof(Events));
    PETSCFREE(events);
    events = tmp; eventsspace += CHUNCK;
  }
  PetscTime(time);
  events[nevents].time = time - BaseTime;
  events[nevents].id1     = o1->id;
  if (o2) events[nevents].id2     = o2->id; else events[nevents].id2 = -1;
  if (o3) events[nevents].id3     = o3->id; else events[nevents].id3 = -1;
  events[nevents].type   = event;
  events[nevents].cookie = 0;
  events[nevents++].event= ACTIONBEGIN;
  if (t != 1) return 0;
  EventsType[event][COUNT]++;
  EventsType[event][TIME]  -= time;
  EventsType[event][FLOPS] -= _TotalFlops;
  return 0;
}
/*
     Event end logger with complete logging
*/
int pleall(int event,int t,PetscObject o1,PetscObject o2,PetscObject o3,
                                                         PetscObject o4)
{
 double time;
 if (nevents >= eventsspace) {
    Events *tmp;
    tmp = (Events *) PETSCMALLOC( (eventsspace+CHUNCK)*sizeof(Events) );
    CHKPTRQ(tmp);
    PETSCMEMCPY(tmp,events,eventsspace*sizeof(Events));
    PETSCFREE(events);
    events = tmp; eventsspace += CHUNCK;
  }
  PetscTime(time);
  events[nevents].time   = time - BaseTime;
  events[nevents].id1    = o1->id;
  if (o2) events[nevents].id2    = o2->id; else events[nevents].id2 = -1;
  if (o3) events[nevents].id3    = o3->id; else events[nevents].id3 = -1;
  events[nevents].type   = event;
  events[nevents].cookie = 0;
  events[nevents++].event= ACTIONEND;
  if (t != 1) return 0;
  EventsType[event][TIME] += time;
  EventsType[event][FLOPS] += _TotalFlops;
  return 0;
}
/*
     Default event begin logger
*/
int plb(int event,int t,PetscObject o1,PetscObject o2,PetscObject o3,
        PetscObject o4)
{
  if (t != 1) return 0;
  EventsType[event][COUNT]++;
  PetscTimeSubtract(EventsType[event][TIME]);
  EventsType[event][FLOPS] -= _TotalFlops;
  return 0;
}
/*
     Default event end logger
*/
int ple(int event,int t,PetscObject o1,PetscObject o2,PetscObject o3,
        PetscObject o4)
{
  if (t != 1) return 0;
  PetscTimeAdd(EventsType[event][TIME]);
  EventsType[event][FLOPS] += _TotalFlops;
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

/*@
   PLogAllBegin - Turns on extensive logging of objects and events. Logs 
   all events. This creates large log files and slows the program down.

   Options Database Keys:
$  -log_all : Prints extensive log information (for code compiled
$      with PETSC_LOG)

   Notes:
   A related routine is PLogBegin(), which is intended for production
   runs since it logs only flop rates and object creation (and shouldn't
   significantly slow the programs).

.keywords: log, all, begin

.seealso: PLogDump(), PLogBegin()
@*/
int PLogAllBegin()
{
  objects = (Objects*) PETSCMALLOC(CHUNCK*sizeof(Objects));CHKPTRQ(objects);
  events = (Events*) PETSCMALLOC(CHUNCK*sizeof(Events));CHKPTRQ(events);
  _PHC = phc;
  _PHD = phd;
  _PLB = plball;
  _PLE = pleall;
  /* all processors sync here for more consistent logging */
  MPI_Barrier(MPI_COMM_WORLD);
  PetscTime(BaseTime);
  return 0;
}

/*@
    PLogBegin - Turns on logging of objects and events. This logs flop
    rates and object creation and should not slow programs down too much.
    This may be called more than once.

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
  objects = (Objects*) PETSCMALLOC(CHUNCK*sizeof(Objects));CHKPTRQ(objects);
  events = (Events*) PETSCMALLOC(CHUNCK*sizeof(Events));CHKPTRQ(events);
  _PHC = phc;
  _PHD = phd;
  _PLB = plb;
  _PLE = ple;
  /* all processors sync here for more consistent logging */
  MPI_Barrier(MPI_COMM_WORLD);
  PetscTime(BaseTime);
  return 0;
}

/*@
   PLogDump - Dumps logs of objects to a file. This file is intended to 
   be read by petsc/bin/tkreview; it is not user friendly.

   Input Parameter:
.  name - an optional file name

   Notes:
   The default file name is 
$      Log.<mytid>
   where <mytid> is the processor number. If no name is specified, 
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
  int    i,mytid;
  FILE   *fd;
  char   file[64];
  double flops,_TotalTime;
  
  PetscTime(_TotalTime);
  _TotalTime -= BaseTime;

  MPI_Comm_rank(MPI_COMM_WORLD,&mytid);
  if (name) sprintf(file,"%s.%d",name,mytid);
  else  sprintf(file,"Log.%d",mytid);
  fd = fopen(file,"w"); if (!fd) SETERRQ(1,"PlogDump: cannot open file");

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
    fprintf(fd,"%d \n",objects[i].parent);
    if (!objects[i].string[0]) {fprintf(fd,"No Info\n");}
    else fprintf(fd,"%s\n",objects[i].string);
    if (!objects[i].name[0]) {fprintf(fd,"No Name\n");}
    else fprintf(fd,"%s\n",objects[i].name);
  }
  for ( i=0; i<100; i++ ) {
    flops = 0.0;
    if (EventsType[i][TIME]){flops = EventsType[i][FLOPS]/EventsType[i][TIME];}
    fprintf(fd,"%d %16g %16g %16g %16g\n",i,EventsType[i][COUNT],
                      EventsType[i][FLOPS],EventsType[i][TIME],flops);
  }
  fprintf(fd,"Total Flops %14e %16.8e\n",_TotalFlops,_TotalTime);
  fclose(fd);
  return 0;
}

static char *(name[]) = {"MatMult         ",
                         "MatBeginAssembly",
                         "MatEndAssembly  ",
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
                         " "," "," "," "," ",
                         "VecDot          ",
                         "VecNorm         ",
                         "VecASum         ",
                         "VecAMax         ",
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
                         "VecBeginAssembly",
                         "VecEndAssembly  ",
                         "VecMTDot        ",
                         "VecMDot         ",
                         "VecMAXPY        ",
                         "VecPMult        ",
                         " "," "," "," "," ",
                         "SLESSolve       ",
                         "PCSetUp         ",
                         "PCApply         ",
                         "SLESSetUp       ",
                         " ",
                         "SNESSolve       ",
                         "SNESLineSearch  ",
                         "SNESFunctionEval",
                         "SNESJacobianEval",
                         " "," "," "," "," ",
                         " "," "," "," "," ",
                         " "," "," "," "," ",
                         " "," "," "," "," "};

/*@
    PLogEventRegister - Registers an event name for logging operations in 
    an application code.  Note that petsc/include/plog.h MUST be included
    in the user's code to employ this function.

    Input Parameters:
.   e - integer associated with the event (69 < e < 89) 
.   string - name associated with the event

    Notes: 
    PETSc automatically logs library events if the code has been
    compiled with -DPETSC_LOG (which is the default) and -log,
    -log_summary, or -log_all are specified.  PLogEventRegister() is
    intended for logging user events to supplement this PETSc
    information.

    Example of Usage:
$     #define USER_EVENT 75
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
  if (e < 70) SETERRQ(1,"PLogRegisterEvent: user events must be > 69");
  if (e > 89) SETERRQ(1,"PLogRegisterEvent: user events must be < 89");
  name[e] = string;
  return 0;
}
  
/*@
   PLogPrint - Prints a summary of the logging.

   Input Parameter:
.  file - a file pointer
.  comm - communicator, only the first processor prints

   Options Database Keys:
$  -log_summary : Prints summary of log information (for code
   compiled with PETSC_LOG)

   Notes:
   More extensive examination of the log information can be done with 
   PLogDump (activated by -log or -log_all) in combination with
   tkreview. 
   
.keywords: log, dump, print

.seealso: PLogBegin(), PLogDump()
@*/
int PLogPrint(MPI_Comm comm,FILE *fd)
{
  double maxo,mino,aveo,mem,totmem,maxmem,minmem;
  int    numtid,i;
  double maxf,minf,avef,totf,_TotalTime,maxt,mint,avet,tott;
  double fmin,fmax,ftot,wdou,totts,totff;

  MPI_Comm_size(comm,&numtid);

  PetscTime(_TotalTime);  _TotalTime -= BaseTime;

  wdou = _TotalFlops; 
  MPI_Reduce(&wdou,&minf,1,MPI_DOUBLE,MPI_MIN,0,comm);
  MPI_Reduce(&wdou,&maxf,1,MPI_DOUBLE,MPI_MAX,0,comm);
  MPI_Reduce(&wdou,&totf,1,MPI_DOUBLE,MPI_SUM,0,comm);
  avef = (totf)/((double) numtid);
  wdou = nobjects;
  MPI_Reduce(&wdou,&mino,1,MPI_DOUBLE,MPI_MIN,0,comm);
  MPI_Reduce(&wdou,&maxo,1,MPI_DOUBLE,MPI_MAX,0,comm);
  MPI_Reduce(&wdou,&aveo,1,MPI_DOUBLE,MPI_SUM,0,comm);
  aveo = (aveo)/((double) numtid);
  wdou = _TotalTime;
  MPI_Reduce(&wdou,&mint,1,MPI_DOUBLE,MPI_MIN,0,comm);
  MPI_Reduce(&wdou,&maxt,1,MPI_DOUBLE,MPI_MAX,0,comm);
  MPI_Reduce(&wdou,&tott,1,MPI_DOUBLE,MPI_SUM,0,comm);
  avet = (tott)/((double) numtid);

  MPIU_fprintf(comm,fd,"\nPerformance Summary:\n");

  MPIU_fprintf(comm,fd,"\n                Max         Min        Avg        Total \n");
  MPIU_fprintf(comm,fd,"Time:        %5.3e   %5.3e   %5.3e\n",maxt,mint,avet);
  MPIU_fprintf(comm,fd,"Objects:     %5.3e   %5.3e   %5.3e\n",maxo,mino,aveo);
  MPIU_fprintf(comm,fd,"Flops:       %5.3e   %5.3e   %5.3e  %5.3e\n",
                                                 maxf,minf,avef,totf);

  fmin = minf/mint; fmax = maxf/maxt; ftot = totf/maxt;
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
    "\n-----------------------------------------------------------------\
-------------\n"); 

  /* loop over operations looking for interesting ones */
  MPIU_fprintf(comm,fd,"\nPhase               Count       Time (sec)        \
   Flops/sec     %%Time %%Flop\n");
  MPIU_fprintf(comm,fd,"                             Min       Max      \
  Min       Max\n");
  for ( i=0; i<100; i++ ) {
    if (EventsType[i][TIME]) {
      wdou = EventsType[i][FLOPS]/EventsType[i][TIME];
    }
    else wdou = 0.0;
    MPI_Reduce(&wdou,&minf,1,MPI_DOUBLE,MPI_MIN,0,comm);
    MPI_Reduce(&wdou,&maxf,1,MPI_DOUBLE,MPI_MAX,0,comm);
    wdou = EventsType[i][FLOPS];
    MPI_Reduce(&wdou,&totff,1,MPI_DOUBLE,MPI_SUM,0,comm);
    wdou = EventsType[i][TIME];
    MPI_Reduce(&wdou,&mint,1,MPI_DOUBLE,MPI_MIN,0,comm);
    MPI_Reduce(&wdou,&maxt,1,MPI_DOUBLE,MPI_MAX,0,comm);
    MPI_Reduce(&wdou,&totts,1,MPI_DOUBLE,MPI_SUM,0,comm);
    if (EventsType[i][COUNT]) {
    MPIU_fprintf(comm,fd,"%s %8d  %3.2e  %3.2e   %3.2e  %3.2e %5.1f %5.1f\n",
                   name[i],(int)EventsType[i][COUNT],mint,maxt,minf,maxf,
                   100.*totts/tott,100.*totff/totf);
    }
  }
  return 0;
}

#endif
