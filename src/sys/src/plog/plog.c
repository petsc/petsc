#ifndef lint
static char vcid[] = "$Id: header.c,v 1.11 1995/05/05 03:51:26 bsmith Exp bsmith $";
#endif

#include "petsc.h"
#include "ptscimpl.h"
#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include <sys/types.h>
#include <sys/time.h>
#include "petscfix.h"

/*
     For timing!
*/


#define PetscTime(v) {struct timeval _tp; \
        gettimeofday(&_tp,(struct timezone *)0);\
        (v) = ((double)_tp.tv_sec) + (1.0e-6)*(_tp.tv_usec);}
#define PetscTimeSubtract(v) {struct timeval _tp; \
        gettimeofday(&_tp,(struct timezone *)0);\
        (v) -= ((double)_tp.tv_sec) + (1.0e-6)*(_tp.tv_usec);}
#define PetscTimeAdd(v) {struct timeval _tp; \
        gettimeofday(&_tp,(struct timezone *)0);\
        (v) += ((double)_tp.tv_sec) + (1.0e-6)*(_tp.tv_usec);}

/*@ 
   PetscObjectSetName - Sets a string name associated with a Petsc object.

   Input Parameters:
.  obj - the Petsc variable
.  name - the name to give obj

.keywords: object, set, name

.seealso: PetscObjectGetName()
@*/
int PetscObjectSetName(PetscObject obj,char *name)
{
  if (!obj) SETERR(1,"Null object");
  obj->name = name;
  return 0;
}

/*@ 
   PetscObjectGetName - Gets a string name associated with a Petsc object.

   Input Parameters:
.  obj - the Petsc variable
.  name - the name associated with obj

.keywords: object, get, name

.seealso: PetscObjectSetName()
@*/
int PetscObjectGetName(PetscObject obj,char **name)
{
  if (!obj) SETERR(1,"Null object");
  if (!name) SETERR(1,"Void location for name");
  *name = obj->name;
  return 0;
}

static int PrintInfo = 0;

/*@
     PLogAllowInfo - Causes PLogInfo messages to be printed  to stdout.

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
int (*_PLB)(int,PetscObject,PetscObject,PetscObject,PetscObject);
int (*_PLE)(int,PetscObject,PetscObject,PetscObject,PetscObject);


/*
      Default object create logger 
*/
int phc(PetscObject obj)
{
  if (nevents >= eventsspace) {
    Events *tmp;
    tmp = (Events *) MALLOC( (eventsspace+CHUNCK)*sizeof(Events) );
    CHKPTR(tmp);
    MEMCPY(tmp,events,eventsspace*sizeof(Events));
    FREE(events);
    events = tmp; eventsspace += CHUNCK;
  }
  if (nobjects >= objectsspace) {
    Objects *tmp;
    tmp = (Objects *) MALLOC( (objectsspace+CHUNCK)*sizeof(Objects) );
    CHKPTR(tmp);
    MEMCPY(tmp,objects,objectsspace*sizeof(Objects));
    FREE(objects);
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
  MEMSET(objects[nobjects].string,0,64*sizeof(char));
  MEMSET(objects[nobjects].name,0,16*sizeof(char));
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
    tmp = (Events *) MALLOC( (eventsspace+CHUNCK)*sizeof(Events) );
    CHKPTR(tmp);
    MEMCPY(tmp,events,eventsspace*sizeof(Events));
    FREE(events);
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
  ObjectsDestroyed++;
  return 0;
}
/*
    Default event begin logger
*/
int plball(int event,PetscObject o1,PetscObject o2,PetscObject o3,
                                                               PetscObject o4)
{
 double time;
 if (nevents >= eventsspace) {
    Events *tmp;
    tmp = (Events *) MALLOC( (eventsspace+CHUNCK)*sizeof(Events) );
    CHKPTR(tmp);
    MEMCPY(tmp,events,eventsspace*sizeof(Events));
    FREE(events);
    events = tmp; eventsspace += CHUNCK;
  }
  PetscTime(time);
  events[nevents].time = time - BaseTime;
  events[nevents].id1     = o1->id;
  if (o2) events[nevents].id2     = o2->id; else events[nevents].id2     = -1;
  if (o3) events[nevents].id3     = o3->id; else events[nevents].id3     = -1;
  events[nevents].type   = event;
  events[nevents].cookie = 0;
  events[nevents++].event= ACTIONBEGIN;
  EventsType[event][COUNT]++;
  EventsType[event][TIME]  -= time;
  EventsType[event][FLOPS] -= _TotalFlops;
  return 0;
}
/*
     Default event end logger
*/
int pleall(int event,PetscObject o1,PetscObject o2,PetscObject o3,
                                                         PetscObject o4)
{
 double time;
 if (nevents >= eventsspace) {
    Events *tmp;
    tmp = (Events *) MALLOC( (eventsspace+CHUNCK)*sizeof(Events) );
    CHKPTR(tmp);
    MEMCPY(tmp,events,eventsspace*sizeof(Events));
    FREE(events);
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
  EventsType[event][TIME] += time;
  EventsType[event][FLOPS] += _TotalFlops;
  return 0;
}

int plb(int event,PetscObject o1,PetscObject o2,PetscObject o3,PetscObject o4)
{
  EventsType[event][COUNT]++;
  PetscTimeSubtract(EventsType[event][TIME]);
  EventsType[event][FLOPS] -= _TotalFlops;
  return 0;
}
/*
     Default event end logger
*/
int ple(int event,PetscObject o1,PetscObject o2,PetscObject o3,PetscObject o4)
{
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
    PLogAllBegin - Turns on logging of objects and events.

.keywords: log, begin

.seealso: PLogDump(), PLogBegin()
@*/
int PLogAllBegin()
{
  PetscTime(BaseTime);
  events = (Events*) MALLOC( CHUNCK*sizeof(Events) ); CHKPTR(events);
  objects = (Objects*) MALLOC( CHUNCK*sizeof(Objects) ); CHKPTR(objects);
  _PHC = phc;
  _PHD = phd;
  _PLB = plball;
  _PLE = pleall;
  return 0;
}

/*@
    PLogBegin - Turns on logging of objects and events.

.keywords: log, begin

.seealso: PLogDump(), PLogAllBegin()
@*/
int PLogBegin()
{
  PetscTime(BaseTime);
  events = (Events*) MALLOC( CHUNCK*sizeof(Events) ); CHKPTR(events);
  objects = (Objects*) MALLOC( CHUNCK*sizeof(Objects) ); CHKPTR(objects);
  _PHC = phc;
  _PHD = phd;
  _PLB = plb;
  _PLE = ple;
  return 0;
}

/*@
   PLogDump - Dumps logs of objects to a file.

   Input Parameter:
.  name - an optional file name

   Notes:
   The default file name is 
$      Log.<mytid>
   where <mytid> is the processor number. If no name is specified, 
   this file will be used.
   
.keywords: log, dump

.seealso: PLogBegin()
@*/
int PLogDump(char* name)
{
  int    i,mytid;
  FILE   *fd;
  char   file[64];
  double res,flops,_TotalTime;
  
  PetscTime(_TotalTime);
  _TotalTime -= BaseTime;

  MPI_Comm_rank(MPI_COMM_WORLD,&mytid);
  if (name) sprintf(file,"%s.%d",name,mytid);
  else  sprintf(file,"Log.%d",mytid);
  fd = fopen(file,"w"); if (!fd) SETERR(1,0);

  fprintf(fd,"Objects created %d Destroyed %d\n",nobjects,
                                                 ObjectsDestroyed);
  res = MPI_Wtick();
  fprintf(fd,"Clock Resolution %g\n",res);
  fprintf(fd,"Events %d\n",nevents);
  for ( i=0; i<nevents; i++ ) {
    fprintf(fd,"%d %d %d %d %d %d %d\n",(int) (events[i].time/res),events[i].event,
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

#endif
