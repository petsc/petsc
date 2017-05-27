/*************************************************************************************
 *    M A R I T I M E  R E S E A R C H  I N S T I T U T E  N E T H E R L A N D S     *
 *************************************************************************************
 *    authors: Bas van 't Hof, Koos Huijssen, Christiaan M. Klaij                    *
 *************************************************************************************
 *    content: Support for nested PetscTimers                                        *
 *************************************************************************************/
#include <petsclog.h>
#include <petsc/private/logimpl.h>
#include <petsctime.h>
#include <petscviewer.h>
#include "../src/sys/logging/xmllogevent.h"
#include "../src/sys/logging/xmlviewer.h"

#if defined(PETSC_USE_LOG)

/*
 * Support for nested PetscTimers
 *
 * PetscTimers keep track of a lot of useful information: Wall clock times, 
 * message passing statistics, flop counts.  Information about the nested structure 
 * of the timers is lost. Example: 
 *
 * 7:30   Start: awake
 * 7:30      Start: morning routine
 * 7:40         Start: eat
 * 7:49         Done:  eat
 * 7:43      Done:  morning routine
 * 8:15      Start: work
 * 12:15        Start: eat
 * 12:45        Done:  eat
 * 16:00     Done:  work
 * 16:30     Start: evening routine
 * 18:30        Start: eat
 * 19:15        Done:  eat
 * 22:00     Done:  evening routine
 * 22:00  Done:  awake
 *
 * Petsc timers provide the following timer results:
 *
 *    awake:              1 call    14:30 hours
 *    morning routine:    1 call     0:13 hours
 *    eat:                3 calls    1:24 hours
 *    work:               1 call     7:45 hours
 *    evening routine     1 call     5:30 hours
 *
 * Nested timers can be used to get the following table:
 *
 *   [1 call]: awake                14:30 hours
 *   [1 call]:    morning routine         0:13 hours         ( 2 % of awake)
 *   [1 call]:       eat                       0:09 hours         (69 % of morning routine)
 *                   rest (morning routine)    0:04 hours         (31 % of morning routine)
 *   [1 call]:    work                    7:45 hours         (53 % of awake)
 *   [1 call]:       eat                       0:30 hours         ( 6 % of work)
 *                   rest (work)               7:15 hours         (94 % of work)
 *   [1 call]:    evening routine         5:30 hours         (38 % of awake)
 *   [1 call]:       eat                       0:45 hours         (14 % of evening routine)
 *                   rest (evening routine)    4:45 hours         (86 % of morning routine)
 *
 * We ignore the concept of 'stages', because these seem to be conflicting notions, or at least, 
 * the nested timers make the stages unnecessary.
 *
 */

/*
 * Data structures for keeping track of nested timers:
 *
 *   nestedEvents: information about the timers that have actually been activated
 *   dftParentActive: if a timer is started now, it is part of (nested inside) the dftParentActive
 *
 * The Default-timers are used to time the nested timers. Every nested timer corresponds to 
 * (one or more) default timers, where one of the default timers has the same event-id as the 
 * nested one.
 *
 * Because of the risk of confusion between nested timer ids and default timer ids, we 
 * introduce a typedef for nested events (NestedEventId) and use the existing type PetscLogEvent 
 * only for default events. Also, all nested event variables are prepended with 'nst', and
 * default timers with 'dft'.
 */

#define DFT_ID_AWAKE -1

typedef PetscLogEvent NestedEventId;
typedef struct {
  NestedEventId   nstEvent;         /* event-code for this nested event, argument 'event' in PetscLogEventStartNested */
  int             nParents;         /* number of 'dftParents': the default timer which was the dftParentActive when this nested timer was activated */
  PetscLogEvent  *dftParentsSorted; /* The default timers which were the dftParentActive when this nested event was started */
  PetscLogEvent  *dftEvents;        /* The default timers which represent the different 'instances' of this nested event */

  PetscLogEvent  *dftParents;       /* The default timers which were the dftParentActive when this nested event was started */
  PetscLogEvent  *dftEventsSorted;  /* The default timers which represent the different 'instances' of this nested event */
} PetscNestedEvent;

static PetscLogEvent    dftParentActive  = DFT_ID_AWAKE;
static int              nNestedEvents           = 0;
static int              nNestedEventsAllocated  = 0;
static PetscNestedEvent *nestedEvents = NULL;
static PetscLogDouble   threshTime      = 0.01; /* initial value was .1 */

static PetscErrorCode PetscLogEventBeginNested(NestedEventId nstEvent, int t, PetscObject o1, PetscObject o2, PetscObject o3, PetscObject o4);
static PetscErrorCode PetscLogEventEndNested(NestedEventId nstEvent, int t, PetscObject o1, PetscObject o2, PetscObject o3, PetscObject o4);

PetscErrorCode PetscLogNestedBegin(void)
{
  PetscErrorCode    ierr;
  PetscFunctionBegin;
  if (nestedEvents) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_COR,"nestedEvents already allocated");

  nNestedEventsAllocated=10;
  ierr = PetscMalloc1(nNestedEventsAllocated,&nestedEvents);CHKERRQ(ierr);


  dftParentActive = DFT_ID_AWAKE;
  nNestedEvents =1;

  /* 'Awake' is nested event 0. It has no parents */
  nestedEvents[0].nstEvent          = 0;
  nestedEvents[0].nParents          = 0;
  nestedEvents[0].dftParentsSorted  = NULL;
  nestedEvents[0].dftEvents         = NULL;
  nestedEvents[0].dftParents        = NULL;
  nestedEvents[0].dftEventsSorted   = NULL;

  ierr = PetscLogSet(PetscLogEventBeginNested, PetscLogEventEndNested);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Delete the data structures for the nested timers */
PetscErrorCode PetscLogNestedEnd(void)
{
  PetscErrorCode    ierr;
  int               i;
  PetscFunctionBegin;

  if (!nestedEvents) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_COR,"nestedEvents does not exist");

  for (i=0; i<nNestedEvents; i++) {
    ierr = PetscFree4(nestedEvents[i].dftParentsSorted,nestedEvents[i].dftEventsSorted,nestedEvents[i].dftParents,nestedEvents[i].dftEvents);CHKERRQ(ierr);
  }
  ierr = PetscFree(nestedEvents);CHKERRQ(ierr);
  nestedEvents           = NULL;
  nNestedEvents          = 0;
  nNestedEventsAllocated = 0;
  PetscFunctionReturn(0);
}


/*
 * UTILITIES: FIND STUFF IN SORTED ARRAYS
 *
 * Utility: find a default timer in a sorted array */
static PetscErrorCode PetscLogEventFindDefaultTimer(PetscLogEvent dftIndex,    /* index to be found */
                                                    const PetscLogEvent *dftArray,  /* sorted array of PetscLogEvent-ids */
                                                    int narray,   /* dimension of dftArray */
                                                    int *entry)         /* entry in the array where dftIndex may be found; 
                                                                            *   if dftArray[entry] != dftIndex, then dftIndex is not part of dftArray  
                                                                            *   In that case, the dftIndex can be inserted at this entry. */
{
  PetscFunctionBegin;
  if (narray==0 || dftIndex <= dftArray[0]) {
    *entry = 0;
  } else if (dftIndex > dftArray[narray-1]) {
    *entry = narray;
  } else {
    int ihigh=narray-1,  ilow=0;
    while (ihigh>ilow) {
      const int imiddle = (ihigh+ilow)/2;
      if (dftArray[imiddle] > dftIndex) {
        ihigh=imiddle;
      } else if (dftArray[imiddle]<dftIndex) {
        ilow =imiddle+1;
      } else {
        ihigh=imiddle;
        ilow =imiddle;
      }
    }
    *entry = ihigh;
  }
  PetscFunctionReturn(0);
}

/* Utility: find the nested event with given identification */
static PetscErrorCode PetscLogEventFindNestedTimer(NestedEventId nstEvent, /* Nested event to be found */
                                                   int *entry)          /* entry in the nestedEvents where nstEvent may be found;
                                                                              if nestedEvents[entry].nstEvent != nstEvent, then index is not part of iarray  */
{
  PetscFunctionBegin;

  if (nNestedEvents==0 || nstEvent <= nestedEvents[0].nstEvent) { 
    *entry = 0;
  } else if (nstEvent > nestedEvents[nNestedEvents-1].nstEvent) {
    *entry = nNestedEvents;
  } else {
    int ihigh=nNestedEvents-1,  ilow=0;
    while (ihigh>ilow) {
      const int imiddle = (ihigh+ilow)/2;
      if (nestedEvents[imiddle].nstEvent > nstEvent) {
        ihigh=imiddle;
      } else if (nestedEvents[imiddle].nstEvent<nstEvent) {
        ilow =imiddle+1;
      } else {
        ihigh=imiddle;
        ilow =imiddle;
      }
    }
    *entry = ihigh;
  }
  PetscFunctionReturn(0);
}

/******************************************************************************************/
/* Start a nested event */
static PetscErrorCode PetscLogEventBeginNested(NestedEventId nstEvent, int t, PetscObject o1, PetscObject o2, PetscObject o3, PetscObject o4)
{
  PetscErrorCode  ierr;
  int             entry, pentry, tentry,i;
  PetscLogEvent   dftEvent;

  PetscFunctionBegin;
  ierr = PetscLogEventFindNestedTimer(nstEvent, &entry);CHKERRQ(ierr); 
  if (entry>=nNestedEvents || nestedEvents[entry].nstEvent != nstEvent) {
    /* Nested event doesn't exist yet: create it */

    if (nNestedEvents==nNestedEventsAllocated) {
      /* Enlarge and re-allocate nestedEvents if needed */
      PetscNestedEvent *tmp = nestedEvents;
      ierr = PetscMalloc1(2*nNestedEvents,&nestedEvents);CHKERRQ(ierr);
      nNestedEventsAllocated*=2;
      ierr = PetscMemcpy(nestedEvents, tmp, nNestedEvents*sizeof(PetscNestedEvent));CHKERRQ(ierr);
      ierr = PetscFree(tmp);CHKERRQ(ierr);
    }

    /* Clear space in nestedEvents for new nested event */
    nNestedEvents++;
    for (i = nNestedEvents-1; i>entry; i--) {
      nestedEvents[i] = nestedEvents[i-1];
    }

    /* Create event in nestedEvents */
    nestedEvents[entry].nstEvent = nstEvent;
    nestedEvents[entry].nParents=1;
    ierr = PetscMalloc4(1,&nestedEvents[entry].dftParentsSorted,1,&nestedEvents[entry].dftEventsSorted,1,&nestedEvents[entry].dftParents,1,&nestedEvents[entry].dftEvents);CHKERRQ(ierr);

    /* Fill in new event */
    pentry = 0;
    dftEvent = (PetscLogEvent) nstEvent;

    nestedEvents[entry].nstEvent                 = nstEvent;
    nestedEvents[entry].dftParents[pentry]       = dftParentActive;
    nestedEvents[entry].dftEvents[pentry]        = dftEvent;
    nestedEvents[entry].dftParentsSorted[pentry] = dftParentActive;
    nestedEvents[entry].dftEventsSorted[pentry]  = dftEvent;

  } else {
    /* Nested event exists: find current dftParentActive among parents */
    PetscLogEvent *dftParentsSorted = nestedEvents[entry].dftParentsSorted; 
    PetscLogEvent *dftEvents        = nestedEvents[entry].dftEvents; 
    int           nParents          = nestedEvents[entry].nParents;

    ierr = PetscLogEventFindDefaultTimer( dftParentActive, dftParentsSorted, nParents, &pentry);CHKERRQ(ierr); 

    if (pentry>=nParents || dftParentActive != dftParentsSorted[pentry]) {
      /* dftParentActive not in the list: add it to the list */
      int           i;
      PetscLogEvent *dftParents      = nestedEvents[entry].dftParents; 
      PetscLogEvent *dftEventsSorted = nestedEvents[entry].dftEventsSorted; 
      char          name[100]; 

      /* Register a new default timer */
      sprintf(name, "%d -> %d", (int) dftParentActive, (int) nstEvent);
      ierr = PetscLogEventRegister(name, 0, &dftEvent);CHKERRQ(ierr);
      ierr = PetscLogEventFindDefaultTimer( dftEvent, dftEventsSorted, nParents, &tentry);CHKERRQ(ierr); 

      /* Reallocate parents and dftEvents to make space for new parent */
      ierr = PetscMalloc4(1+nParents,&nestedEvents[entry].dftParentsSorted,1+nParents,&nestedEvents[entry].dftEventsSorted,1+nParents,&nestedEvents[entry].dftParents,1+nParents,&nestedEvents[entry].dftEvents);CHKERRQ(ierr);
      ierr = PetscMemcpy(nestedEvents[entry].dftParentsSorted, dftParentsSorted, nParents*sizeof(PetscLogEvent));CHKERRQ(ierr);
      ierr = PetscMemcpy(nestedEvents[entry].dftEventsSorted,  dftEventsSorted,  nParents*sizeof(PetscLogEvent));CHKERRQ(ierr);
      ierr = PetscMemcpy(nestedEvents[entry].dftParents,       dftParents,       nParents*sizeof(PetscLogEvent));CHKERRQ(ierr);
      ierr = PetscMemcpy(nestedEvents[entry].dftEvents,        dftEvents,        nParents*sizeof(PetscLogEvent));CHKERRQ(ierr);
      ierr = PetscFree4(dftParentsSorted,dftEventsSorted,dftParents,dftEvents);CHKERRQ(ierr);

      dftParents       = nestedEvents[entry].dftParents;
      dftEvents        = nestedEvents[entry].dftEvents;
      dftParentsSorted = nestedEvents[entry].dftParentsSorted;
      dftEventsSorted  = nestedEvents[entry].dftEventsSorted;

      nestedEvents[entry].nParents++;
      nParents++;

      for (i = nParents-1; i>pentry; i--) {
        dftParentsSorted[i] = dftParentsSorted[i-1];
        dftEvents[i]        = dftEvents[i-1];
      }
      for (i = nParents-1; i>tentry; i--) {
        dftParents[i]      = dftParents[i-1];
        dftEventsSorted[i] = dftEventsSorted[i-1];
      }

      /* Fill in the new default timer */
      dftParentsSorted[pentry] = dftParentActive;
      dftEvents[pentry]        = dftEvent;
      dftParents[tentry]       = dftParentActive;
      dftEventsSorted[tentry]  = dftEvent;

    } else {
      /* dftParentActive was found: find the corresponding default 'dftEvent'-timer */
      dftEvent = nestedEvents[entry].dftEvents[pentry];
    }
  }

  /* Start the default 'dftEvent'-timer and update the dftParentActive */
  ierr = PetscLogEventBeginDefault(dftEvent,t,o1,o2,o3,o4);CHKERRQ(ierr); 
  dftParentActive = dftEvent;
  PetscFunctionReturn(0);
}

/* End a nested event */
static PetscErrorCode PetscLogEventEndNested(NestedEventId nstEvent, int t, PetscObject o1, PetscObject o2, PetscObject o3, PetscObject o4)
{
  PetscErrorCode  ierr;
  int             entry, pentry, nParents;
  PetscLogEvent  *dftEventsSorted;

  PetscFunctionBegin;
  /* Find the nested event */
  ierr = PetscLogEventFindNestedTimer(nstEvent, &entry);CHKERRQ(ierr);
  if (entry>=nNestedEvents) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE, "Logging event %d larger than number of events %d",entry,nNestedEvents);
  if (nestedEvents[entry].nstEvent != nstEvent) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE, "Logging event %d had unbalanced begin/end pairs does not match %d",entry,nstEvent);
  dftEventsSorted = nestedEvents[entry].dftEventsSorted;
  nParents        = nestedEvents[entry].nParents;

  /* Find the current default timer among the 'dftEvents' of this event */
  ierr = PetscLogEventFindDefaultTimer( dftParentActive, dftEventsSorted, nParents, &pentry);CHKERRQ(ierr); 

  if (pentry>=nParents) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE, "Entry %d is larger than number of parents %d",pentry,nParents);
  if (dftEventsSorted[pentry] != dftParentActive) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE, "Active parent is %d, but we seem to be closing %d",dftParentActive,dftEventsSorted[pentry]);

  /* Stop the default timer and update the dftParentActive */
  ierr = PetscLogEventEndDefault(dftParentActive,t,o1,o2,o3,o4);CHKERRQ(ierr);
  dftParentActive = nestedEvents[entry].dftParents[pentry]; 
  PetscFunctionReturn(0);
}

/* Set the threshold time for logging the events 
 */
PetscErrorCode PetscLogSetThreshold(PetscLogDouble newThresh, PetscLogDouble *oldThresh)
{
  PetscFunctionBegin;
  *oldThresh = threshTime;
  threshTime = newThresh;
  PetscFunctionReturn(0);
}

static PetscErrorCode  PetscPrintExeSpecs(PetscViewer viewer)
{
  PetscErrorCode     ierr;
  char               arch[128],hostname[128],username[128],pname[PETSC_MAX_PATH_LEN],date[128];
  char               version[256], buildoptions[128];
  PetscMPIInt        size;
  MPI_Comm           comm;
  size_t             len;
  
  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)viewer,&comm);CHKERRQ(ierr);
  ierr = PetscGetArchType(arch,sizeof(arch));CHKERRQ(ierr);
  ierr = PetscGetHostName(hostname,sizeof(hostname));CHKERRQ(ierr);
  ierr = PetscGetUserName(username,sizeof(username));CHKERRQ(ierr);
  ierr = PetscGetProgramName(pname,sizeof(pname));CHKERRQ(ierr);
  ierr = PetscGetDate(date,sizeof(date));CHKERRQ(ierr);
  ierr = PetscGetVersion(version,sizeof(version));CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &size);CHKERRQ(ierr);

  ierr = PetscViewerXMLStartSection(viewer, "runspecification", "Run Specification");CHKERRQ(ierr);
  ierr = PetscViewerXMLPutString(   viewer, "executable"  , "Executable"   , pname );CHKERRQ(ierr);
  ierr = PetscViewerXMLPutString(   viewer, "architecture", "Architecture" , arch );CHKERRQ(ierr);
  ierr = PetscViewerXMLPutString(   viewer, "hostname"    , "Host"         , hostname);CHKERRQ(ierr);
  ierr = PetscViewerXMLPutInt(      viewer, "nprocesses"  , "Number of processes", size );CHKERRQ(ierr);
  ierr = PetscViewerXMLPutString(   viewer, "user"        , "Run by user"  , username);CHKERRQ(ierr);
  ierr = PetscViewerXMLPutString(   viewer, "date"        , "Started at"   , date);CHKERRQ(ierr);
  ierr = PetscViewerXMLPutString(   viewer, "petscrelease", "Petsc Release", version);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
  sprintf(buildoptions, "Debug");
#else
  buildoptions[0] = 0;
#endif
  ierr = PetscStrlen(buildoptions,&len);CHKERRQ(ierr);
  if (len) {
    ierr = PetscViewerXMLPutString(viewer, "petscbuildoptions", "Petsc build options", buildoptions);CHKERRQ(ierr);
  }
  ierr = PetscViewerXMLEndSection(viewer, "runspecification");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Print the global performance: max, max/min, average and total of 
 *      time, objects, flops, flops/sec, memory, MPI messages, MPI message lengths, MPI reductions.
 */
static PetscErrorCode  PetscPrintXMLGlobalPerformanceElement(PetscViewer viewer, const char *name, const char *desc, PetscLogDouble max, PetscLogDouble ratio, PetscLogDouble avg, PetscLogDouble tot)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscViewerXMLStartSection(viewer, name, desc);CHKERRQ(ierr);
  ierr = PetscViewerXMLPutDouble(viewer, "max", NULL, max, "%e");CHKERRQ(ierr);
  ierr = PetscViewerXMLPutDouble(viewer, "ratio", NULL, ratio, "%f");CHKERRQ(ierr);
  if (avg>-1.0) {
    ierr = PetscViewerXMLPutDouble(viewer, "average", NULL, avg, "%e");CHKERRQ(ierr);
  }
  if (tot>-1.0) {
    ierr = PetscViewerXMLPutDouble(viewer, "total", NULL, tot, "%e");CHKERRQ(ierr);
  }
  ierr = PetscViewerXMLEndSection(viewer, name);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Print the global performance: max, max/min, average and total of 
 *      time, objects, flops, flops/sec, memory, MPI messages, MPI message lengths, MPI reductions.
 */
static PetscErrorCode  PetscPrintGlobalPerformance(PetscViewer viewer, PetscLogDouble locTotalTime)
{
  PetscErrorCode     ierr;
  PetscLogDouble     min, max, tot, ratio, avg;
  PetscLogDouble     flops, mem, red, mess;
  PetscMPIInt        size;
  MPI_Comm           comm;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)viewer,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &size);CHKERRQ(ierr);

  /* Must preserve reduction count before we go on */
  red = petsc_allreduce_ct + petsc_gather_ct + petsc_scatter_ct;

  /* Calculate summary information */
  ierr = PetscViewerXMLStartSection(viewer, "globalperformance", "Global performance");CHKERRQ(ierr);

  /*   Time */
  ierr = MPIU_Allreduce(&locTotalTime, &min, 1, MPIU_PETSCLOGDOUBLE, MPI_MIN, comm);CHKERRQ(ierr);
  ierr = MPIU_Allreduce(&locTotalTime, &max, 1, MPIU_PETSCLOGDOUBLE, MPI_MAX, comm);CHKERRQ(ierr);
  ierr = MPIU_Allreduce(&locTotalTime, &tot, 1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm);CHKERRQ(ierr);
  avg  = (tot)/((PetscLogDouble) size);
  if (min != 0.0) ratio = max/min;
  else ratio = 0.0;
  ierr = PetscPrintXMLGlobalPerformanceElement(viewer, "time", "Time (sec)", max, ratio, avg, -1.0);CHKERRQ(ierr);

  /*   Objects */
  avg  = (PetscLogDouble) petsc_numObjects;
  ierr = MPIU_Allreduce(&avg,          &min, 1, MPIU_PETSCLOGDOUBLE, MPI_MIN, comm);CHKERRQ(ierr);
  ierr = MPIU_Allreduce(&avg,          &max, 1, MPIU_PETSCLOGDOUBLE, MPI_MAX, comm);CHKERRQ(ierr);
  ierr = MPIU_Allreduce(&avg,          &tot, 1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm);CHKERRQ(ierr);
  avg  = (tot)/((PetscLogDouble) size);
  if (min != 0.0) ratio = max/min;
  else ratio = 0.0;
  ierr = PetscPrintXMLGlobalPerformanceElement(viewer, "objects", "Objects", max, ratio, avg, -1.0);CHKERRQ(ierr);

  /*   Flop */
  ierr = MPIU_Allreduce(&petsc_TotalFlops,  &min, 1, MPIU_PETSCLOGDOUBLE, MPI_MIN, comm);CHKERRQ(ierr);
  ierr = MPIU_Allreduce(&petsc_TotalFlops,  &max, 1, MPIU_PETSCLOGDOUBLE, MPI_MAX, comm);CHKERRQ(ierr);
  ierr = MPIU_Allreduce(&petsc_TotalFlops,  &tot, 1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm);CHKERRQ(ierr);
  avg  = (tot)/((PetscLogDouble) size);
  if (min != 0.0) ratio = max/min;
  else ratio = 0.0;
  ierr = PetscPrintXMLGlobalPerformanceElement(viewer, "mflop", "MFlop", max/1.0E6, ratio, avg/1.0E6, tot/1.0E6);CHKERRQ(ierr);

  /*   Flop/sec -- Must talk to Barry here */
  if (locTotalTime != 0.0) flops = petsc_TotalFlops/locTotalTime;
  else flops = 0.0;
  ierr = MPIU_Allreduce(&flops,        &min, 1, MPIU_PETSCLOGDOUBLE, MPI_MIN, comm);CHKERRQ(ierr);
  ierr = MPIU_Allreduce(&flops,        &max, 1, MPIU_PETSCLOGDOUBLE, MPI_MAX, comm);CHKERRQ(ierr);
  ierr = MPIU_Allreduce(&flops,        &tot, 1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm);CHKERRQ(ierr);
  avg  = (tot)/((PetscLogDouble) size);
  if (min != 0.0) ratio = max/min;
  else ratio = 0.0;
  ierr = PetscPrintXMLGlobalPerformanceElement(viewer, "mflops", "MFlop/sec", max/1.0E6, ratio, avg/1.0E6, tot/1.0E6);CHKERRQ(ierr);

  /*   Memory */
  ierr = PetscMallocGetMaximumUsage(&mem);CHKERRQ(ierr);
  if (mem > 0.0) {
    ierr = MPIU_Allreduce(&mem,          &max, 1, MPIU_PETSCLOGDOUBLE, MPI_MAX, comm);CHKERRQ(ierr);
    ierr = MPIU_Allreduce(&mem,          &min, 1, MPIU_PETSCLOGDOUBLE, MPI_MIN, comm);CHKERRQ(ierr);
    ierr = MPIU_Allreduce(&mem,          &tot, 1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm);CHKERRQ(ierr);
    avg  = (tot)/((PetscLogDouble) size);
    if (min != 0.0) ratio = max/min;
    else ratio = 0.0;
    ierr = PetscPrintXMLGlobalPerformanceElement(viewer, "memory", "Memory (MiB)", max/1024.0/1024.0, ratio, avg/1024.0/1024.0, tot/1024.0/1024.0);CHKERRQ(ierr);
  }
  /*   Messages */
  mess = 0.5*(petsc_irecv_ct + petsc_isend_ct + petsc_recv_ct + petsc_send_ct);
  ierr = MPIU_Allreduce(&mess,         &min, 1, MPIU_PETSCLOGDOUBLE, MPI_MIN, comm);CHKERRQ(ierr);
  ierr = MPIU_Allreduce(&mess,         &max, 1, MPIU_PETSCLOGDOUBLE, MPI_MAX, comm);CHKERRQ(ierr);
  ierr = MPIU_Allreduce(&mess,         &tot, 1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm);CHKERRQ(ierr);
  avg  = (tot)/((PetscLogDouble) size);
  if (min != 0.0) ratio = max/min;
  else ratio = 0.0;
  ierr = PetscPrintXMLGlobalPerformanceElement(viewer, "messagetransfers", "MPI Message Transfers", max, ratio, avg, tot);CHKERRQ(ierr);

  /*   Message Volume */
  mess = 0.5*(petsc_irecv_len + petsc_isend_len + petsc_recv_len + petsc_send_len);
  ierr = MPIU_Allreduce(&mess,         &min, 1, MPIU_PETSCLOGDOUBLE, MPI_MIN, comm);CHKERRQ(ierr);
  ierr = MPIU_Allreduce(&mess,         &max, 1, MPIU_PETSCLOGDOUBLE, MPI_MAX, comm);CHKERRQ(ierr);
  ierr = MPIU_Allreduce(&mess,         &tot, 1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm);CHKERRQ(ierr);
  avg = (tot)/((PetscLogDouble) size);
  if (min != 0.0) ratio = max/min;
  else ratio = 0.0;
  ierr = PetscPrintXMLGlobalPerformanceElement(viewer, "messagevolume", "MPI Message Volume (MiB)", max/1024.0/1024.0, ratio, avg/1024.0/1024.0, tot/1024.0/1024.0);CHKERRQ(ierr);

  /*   Reductions */
  ierr = MPIU_Allreduce(&red,          &min, 1, MPIU_PETSCLOGDOUBLE, MPI_MIN, comm);CHKERRQ(ierr);
  ierr = MPIU_Allreduce(&red,          &max, 1, MPIU_PETSCLOGDOUBLE, MPI_MAX, comm);CHKERRQ(ierr);
  ierr = MPIU_Allreduce(&red,          &tot, 1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm);CHKERRQ(ierr);
  if (min != 0.0) ratio = max/min;
  else ratio = 0.0;
  ierr = PetscPrintXMLGlobalPerformanceElement(viewer, "reductions", "MPI Reductions", max, ratio, -1, -1);CHKERRQ(ierr);
  ierr = PetscViewerXMLEndSection(viewer, "globalperformance");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

typedef struct {
  PetscLogEvent  dftEvent;
  NestedEventId  nstEvent;
  PetscLogEvent  dftParent;
  NestedEventId  nstParent;
  PetscBool      own;
  int            depth;
  NestedEventId* nstPath;
} PetscNestedEventTree;

/* Compare timers to sort them in the tree */
static int compareTreeItems(const void *item1_, const void *item2_)
{
  int                  i;
  PetscNestedEventTree *item1 = (PetscNestedEventTree *) item1_;
  PetscNestedEventTree *item2 = (PetscNestedEventTree *) item2_;
  for (i=0; i<PetscMin(item1->depth,item2->depth); i++) {
    if (item1->nstPath[i]<item2->nstPath[i]) return -1;
    if (item1->nstPath[i]>item2->nstPath[i]) return +1;
  }
  if (item1->depth < item2->depth) return -1;
  if (item1->depth > item2->depth) return 1;
  return 0;
}
/*
 * Do MPI communication to get the complete, nested calling tree for all processes: there may be
 * calls that happen in some processes, but not in others.
 *
 * The output, tree[nTimers] is an array of PetscNestedEventTree-structs.
 * The tree is sorted so that the timers can be printed in the order of appearance.
 *
 * For tree-items which appear in the trees of multiple processes (which will be most items), the 
 * following rule is followed:
 * + if information from my own process is available, then that is the information stored in tree.
 *   otherwise it is some other process's information.
 */
static PetscErrorCode  PetscCreateLogTreeNested(PetscViewer viewer, PetscNestedEventTree **p_tree, int *p_nTimers)
{
  PetscNestedEventTree *tree = NULL, *newTree;
  int                  *treeIndices;
  int                  nTimers, totalNTimers, i, j, iTimer0, maxDefaultTimer;
  int                  yesno;
  PetscBool            done;
  PetscErrorCode       ierr;
  int                  maxdepth;
  int                  depth;
  int                  illegalEvent;
  int                  iextra;
  PetscStageLog        stageLog;
  NestedEventId        *nstPath, *nstMyPath;
  MPI_Comm             comm;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)viewer,&comm);CHKERRQ(ierr);
  ierr = PetscLogGetStageLog(&stageLog);CHKERRQ(ierr);

  /* Calculate memory needed to store everybody's information and allocate tree */
  nTimers = 0;
  for (i=0; i<nNestedEvents; i++) nTimers+=nestedEvents[i].nParents;

  ierr = PetscMalloc1(nTimers,&tree);CHKERRQ(ierr);

  /* Fill tree with readily available information */ 
  iTimer0 = 0;
  maxDefaultTimer =0;
  for (i=0; i<nNestedEvents; i++) {
    int           nParents          = nestedEvents[i].nParents;
    NestedEventId nstEvent          = nestedEvents[i].nstEvent;
    PetscLogEvent *dftParentsSorted = nestedEvents[i].dftParentsSorted;
    PetscLogEvent *dftEvents        = nestedEvents[i].dftEvents;
    for (j=0; j<nParents; j++) {
      maxDefaultTimer = PetscMax(dftEvents[j],maxDefaultTimer);

      tree[iTimer0+j].dftEvent   = dftEvents[j];
      tree[iTimer0+j].nstEvent   = nstEvent;
      tree[iTimer0+j].dftParent  = dftParentsSorted[j];
      tree[iTimer0+j].own        = PETSC_TRUE;

      tree[iTimer0+j].nstParent  = 0;
      tree[iTimer0+j].depth      = 0;
      tree[iTimer0+j].nstPath    = NULL;
    }
    iTimer0 += nParents;
  }

  /* Calculate the global maximum for the default timer index, so array treeIndices can 
   * be allocated only once */
  ierr = MPIU_Allreduce(&maxDefaultTimer, &j, 1, MPI_INT, MPI_MAX, comm);CHKERRQ(ierr);
  maxDefaultTimer = j;

  /* Find default timer's place in the tree */
  ierr = PetscCalloc1(maxDefaultTimer+1,&treeIndices);CHKERRQ(ierr);
  treeIndices[0] = 0;
  for (i=0; i<nTimers; i++) {
    PetscLogEvent dftEvent = tree[i].dftEvent;
    treeIndices[dftEvent] = i;
  }

  /* Find each dftParent's nested identification */
  for (i=0; i<nTimers; i++) {
    PetscLogEvent dftParent = tree[i].dftParent;
    if (dftParent!= DFT_ID_AWAKE) {
      int j = treeIndices[dftParent];
      tree[i].nstParent  = tree[j].nstEvent;
    }
  }

  /* Find depths for each timer path */
  done = PETSC_FALSE;
  maxdepth = 0;
  while (!done) {
    done = PETSC_TRUE;
    for (i=0; i<nTimers; i++) {
      if (tree[i].dftParent == DFT_ID_AWAKE) {
        tree[i].depth = 1;
        maxdepth = PetscMax(1,maxdepth);
      } else {
        int j = treeIndices[tree[i].dftParent];
        depth = 1+tree[j].depth;
        if (depth>tree[i].depth) {
          done          = PETSC_FALSE;
          tree[i].depth = depth;
          maxdepth      = PetscMax(depth,maxdepth);
        }
      }
    }
  }

  /* Allocate the paths in the entire tree */
  for (i=0; i<nTimers; i++) {
    depth = tree[i].depth;
    ierr = PetscCalloc1(depth,&tree[i].nstPath);CHKERRQ(ierr);
  }

  /* Calculate the paths for all timers */
  for (depth=1; depth<=maxdepth; depth++) {
    for (i=0; i<nTimers; i++) {
      if (tree[i].depth==depth) {
        if (depth>1) {
          int    j = treeIndices[tree[i].dftParent];
          ierr = PetscMemcpy(tree[i].nstPath,tree[j].nstPath,(depth-1)*sizeof(NestedEventId));CHKERRQ(ierr);
        }
        tree[i].nstPath[depth-1] = tree[i].nstEvent;
      }
    }
  }
  ierr = PetscFree(treeIndices);CHKERRQ(ierr);

  /* Sort the tree on basis of the paths */
  qsort(tree, nTimers, sizeof(PetscNestedEventTree), compareTreeItems);

  /* Allocate an array to store paths */
  depth = maxdepth;
  ierr = MPIU_Allreduce(&depth, &maxdepth, 1, MPI_INT, MPI_MAX, comm);CHKERRQ(ierr);
  ierr = PetscMalloc1(maxdepth+1, &nstPath);CHKERRQ(ierr);
  ierr = PetscMalloc1(maxdepth+1, &nstMyPath);CHKERRQ(ierr);

  /* Find an illegal nested event index (1+largest nested event index) */
  illegalEvent = 1+nestedEvents[nNestedEvents-1].nstEvent;
  i = illegalEvent;
  ierr = MPIU_Allreduce(&i, &illegalEvent, 1, MPI_INT, MPI_MAX, comm);CHKERRQ(ierr);

  /* First, detect timers which are not available in this process, but are available in others
   *        Allocate a new tree, that can contain all timers
   * Then,  fill the new tree with all (own and not-own) timers */
  newTree= NULL;
  for (yesno=0; yesno<=1; yesno++) {
    depth  = 1;
    i      = 0;
    iextra = 0;
    while (depth>0) {
      int       j;
      PetscBool same;

      /* Construct the next path in this process's tree:
       * if necessary, supplement with invalid path entries */
      depth++;
      if (depth > maxdepth + 1) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Depth %d > maxdepth+1 %d",depth,maxdepth+1);
      if (i<nTimers) {
        for (j=0;             j<tree[i].depth; j++) nstMyPath[j] = tree[i].nstPath[j];
        for (j=tree[i].depth; j<depth;         j++) nstMyPath[j] = illegalEvent;
      } else {
        for (j=0;             j<depth;         j++) nstMyPath[j] = illegalEvent;
      }
 
      /* Communicate with other processes to obtain the next path and its depth */
      ierr = MPIU_Allreduce(nstMyPath, nstPath, depth, MPI_INT, MPI_MIN, comm);CHKERRQ(ierr);
      for (j=depth-1; (int) j>=0; j--) { 
        if (nstPath[j]==illegalEvent) depth=j;
      }
 
      if (depth>0) {
        /* If the path exists */
        
        /* check whether the next path is the same as this process's next path */
        same = PETSC_TRUE;     
        for (j=0; same && j<depth; j++) { same = (same &&  nstMyPath[j] == nstPath[j]) ? PETSC_TRUE : PETSC_FALSE;}
        
        if (same) { 
          /* Register 'own path' */
          if (newTree) newTree[i+iextra] = tree[i];
          i++;  
        } else { 
          /* Register 'not an own path' */
          if (newTree) { 
            newTree[i+iextra].nstEvent   = nstPath[depth-1];
            newTree[i+iextra].own        = PETSC_FALSE;
            newTree[i+iextra].depth      = depth;
            ierr = PetscMalloc1(depth, &newTree[i+iextra].nstPath);CHKERRQ(ierr);
            for (j=0; j<depth; j++) {newTree[i+iextra].nstPath[j] = nstPath[j];}

            newTree[i+iextra].dftEvent  = 0;
            newTree[i+iextra].dftParent = 0;
            newTree[i+iextra].nstParent = 0;
          }
          iextra++; 
        }

      }
    }

    /* Determine the size of the complete tree (with own and not-own timers) and allocate the new tree */
    totalNTimers = nTimers + iextra;
    if (!newTree) {
      ierr = PetscMalloc1(totalNTimers, &newTree);CHKERRQ(ierr);
    }
  }
  ierr = PetscFree(nstPath);CHKERRQ(ierr);
  ierr = PetscFree(nstMyPath);CHKERRQ(ierr);
  ierr = PetscFree(tree);CHKERRQ(ierr);
  tree = newTree;
  newTree = NULL;

  /* Set return value and return */
  *p_tree    = tree;
  *p_nTimers = totalNTimers;
  PetscFunctionReturn(0);
}

/*
 * Delete the nested timer tree 
 */
static PetscErrorCode  PetscLogFreeNestedTree(PetscNestedEventTree *tree, int nTimers)
{
  int             i;
  PetscErrorCode  ierr;
  
  PetscFunctionBegin;
  for (i=0; i<nTimers; i++) {
    ierr = PetscFree(tree[i].nstPath);CHKERRQ(ierr);
  }
  ierr = PetscFree(tree);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Print the global performance: max, max/min, average and total of 
 *      time, objects, flops, flops/sec, memory, MPI messages, MPI message lengths, MPI reductions.
 */
static PetscErrorCode  PetscPrintXMLNestedLinePerfResults(PetscViewer viewer, const char *name, PetscLogDouble minvalue, PetscLogDouble maxvalue, PetscLogDouble minmaxtreshold)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscViewerXMLStartSection(viewer, name, NULL);CHKERRQ(ierr);
  if (maxvalue>minvalue*minmaxtreshold) {
    ierr = PetscViewerXMLPutDouble(viewer, "minvalue", NULL, minvalue, "%f");CHKERRQ(ierr);
    ierr = PetscViewerXMLPutDouble(viewer, "maxvalue", NULL, maxvalue, "%f");CHKERRQ(ierr);
  } else {
    ierr = PetscViewerXMLPutDouble(viewer, "value", NULL, (minvalue+maxvalue)/2.0, "%g");CHKERRQ(ierr);
  };
  ierr = PetscViewerXMLEndSection(viewer, name);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#define N_COMM 8
static PetscErrorCode  PetscLogPrintNestedLine(PetscViewer viewer,PetscEventPerfInfo perfInfo,PetscLogDouble countsPerCall,int parentCount,int depth,const char *name,PetscLogDouble totalTime,PetscBool *isPrinted)
{
  PetscLogDouble time = perfInfo.time;
  PetscLogDouble timeMx,          timeMn;
  PetscLogDouble countsPerCallMx, countsPerCallMn;
  PetscLogDouble reductSpeedMx,   reductSpeedMn;
  PetscLogDouble flopSpeedMx,     flopSpeedMn;
  PetscLogDouble msgSpeedMx,      msgSpeedMn;
  PetscLogDouble commarr_in[N_COMM], commarr_out[N_COMM];
  PetscErrorCode ierr;
  MPI_Comm       comm;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)viewer,&comm);CHKERRQ(ierr);

  commarr_in[0] =  time; 
  commarr_in[1] = -time; 
  ierr = MPIU_Allreduce(commarr_in, commarr_out,    2, MPIU_PETSCLOGDOUBLE, MPI_MAX, comm);CHKERRQ(ierr);
  timeMx =  commarr_out[0];
  timeMn = -commarr_out[1];

  commarr_in[0] = time>=timeMx*0.001 ?  perfInfo.flops/time         : 0;
  commarr_in[1] = time>=timeMx*0.001 ?  perfInfo.numReductions/time : 0;
  commarr_in[2] = time>=timeMx*0.001 ?  perfInfo.messageLength/time : 0;
  commarr_in[3] = parentCount>0    ?  countsPerCall      : 0;

  commarr_in[4] = time>=timeMx*0.001 ? -perfInfo.flops/time         : -1e30;
  commarr_in[5] = time>=timeMx*0.001 ? -perfInfo.numReductions/time : -1e30;
  commarr_in[6] = time>=timeMx*0.001 ? -perfInfo.messageLength/time : -1e30;
  commarr_in[7] = parentCount>0    ? -countsPerCall      : -1e30;

  ierr = MPIU_Allreduce(commarr_in, commarr_out,  N_COMM, MPIU_PETSCLOGDOUBLE, MPI_MAX, comm);CHKERRQ(ierr);

  flopSpeedMx     =  commarr_out[0];
  reductSpeedMx   =  commarr_out[1];
  msgSpeedMx      =  commarr_out[2];
  countsPerCallMx =  commarr_out[3];

  flopSpeedMn     = -commarr_out[4];
  reductSpeedMn   = -commarr_out[5];
  msgSpeedMn      = -commarr_out[6];
  countsPerCallMn = -commarr_out[7];

  *isPrinted = ((timeMx/totalTime) > (threshTime/100.0)) ? PETSC_TRUE : PETSC_FALSE;
  if (isPrinted) {
    ierr = PetscViewerXMLStartSection(viewer, "event", NULL);CHKERRQ(ierr);
    ierr = PetscViewerXMLPutString(viewer, "name", NULL, name);CHKERRQ(ierr);
    ierr = PetscPrintXMLNestedLinePerfResults(viewer, "time", timeMn/totalTime*100.0, timeMx/totalTime*100.0, 1.02);CHKERRQ(ierr);


    if (countsPerCallMx<1.01 && countsPerCallMn>0.99) {
      /* One call per parent */
    } else {
      ierr = PetscPrintXMLNestedLinePerfResults(viewer, "ncalls", countsPerCallMn, countsPerCallMx, 1.02);CHKERRQ(ierr);
    }
 
    if (flopSpeedMx<0.01) {
      /* NO flops: don't print */
    } else {
      ierr = PetscPrintXMLNestedLinePerfResults(viewer, "mflops", flopSpeedMn/1e6, flopSpeedMx/1e6, 1.05);CHKERRQ(ierr);
    }
 
    if (msgSpeedMx<0.01) {
      /* NO msgs: don't print */
    } else {
      ierr = PetscPrintXMLNestedLinePerfResults(viewer, "mbps", msgSpeedMn/1024.0/1024.0, msgSpeedMx/1024.0/1024.0, 1.05);CHKERRQ(ierr);
    }
 
    if (reductSpeedMx<0.01) {
      /* NO reductions: don't print */
    } else {
      ierr = PetscPrintXMLNestedLinePerfResults(viewer, "nreductsps", reductSpeedMn, reductSpeedMx, 1.05);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

/* Count the number of times the parent event was called */

static int countParents( const PetscNestedEventTree *tree, PetscEventPerfInfo *eventPerfInfo, int i)
{
   if (tree[i].depth<=1) {
     return 1;  /* Main event: only once */
   } else if (!tree[i].own) {
     return 1;  /* This event didn't happen in this process, but did in another */
   } else {
     int iParent;
     for (iParent=i-1; iParent>=0; iParent--) {
       if (tree[iParent].depth == tree[i].depth-1) break;
     }
     if (tree[iParent].depth != tree[i].depth-1) {
       printf("\n\n   *****  Internal error: cannot find parent ****\n\n");
       return -2;
     } else {
       PetscLogEvent dftEvent  = tree[iParent].dftEvent;
       return eventPerfInfo[dftEvent].count;
     }
   }
}

typedef struct {
  int             id;
  PetscLogDouble  val;
} PetscSortItem;

static int compareSortItems(const void *item1_, const void *item2_)
{
  PetscSortItem *item1 = (PetscSortItem *) item1_;
  PetscSortItem *item2 = (PetscSortItem *) item2_;
  if (item1->val > item2->val) return -1;
  if (item1->val < item2->val) return +1;
  return 0;
}

static PetscErrorCode  PetscLogNestedPrint(PetscViewer viewer, PetscNestedEventTree *tree,int nTimers, int iStart, PetscLogDouble totalTime)
{
  int                depth   = tree[iStart].depth;
  const char         *name;
  int                parentCount, nChildren;
  PetscSortItem      *children;
  PetscErrorCode     ierr;
  PetscEventPerfInfo *eventPerfInfo;
  PetscEventPerfInfo myPerfInfo,  otherPerfInfo, selfPerfInfo;
  PetscLogDouble     countsPerCall;
  PetscBool          wasPrinted;
  PetscBool          childWasPrinted;
  MPI_Comm           comm;

  {
  /* Look up the name of the event and its PerfInfo */
     const int          stage=0;
     PetscStageLog      stageLog;
     PetscEventRegInfo  *eventRegInfo;
     ierr          = PetscLogGetStageLog(&stageLog);CHKERRQ(ierr);
     eventRegInfo  = stageLog->eventLog->eventInfo;
     eventPerfInfo = stageLog->stageInfo[stage].eventLog->eventInfo;
     name          = eventRegInfo[(PetscLogEvent) tree[iStart].nstEvent].name;
  }

  ierr = PetscObjectGetComm((PetscObject)viewer,&comm);CHKERRQ(ierr);

  /* Count the number of child processes */
  nChildren = 0;
  {
    int i;
    for (i=iStart+1; i<nTimers; i++) {
      if (tree[i].depth<=depth) break;
      if (tree[i].depth == depth + 1) nChildren++;
    }
  }

  if (nChildren>0) {
    /* Create an array for the id-s and maxTimes of the children,
     *  leaving 2 spaces for self-time and other-time */
    int            i;
    PetscLogDouble *times, *maxTimes;

    ierr = PetscMalloc1(nChildren+2,&children);CHKERRQ(ierr);
    nChildren = 0;
    for (i=iStart+1; i<nTimers; i++) {
      if (tree[i].depth<=depth) break;
      if (tree[i].depth == depth + 1) {
        children[nChildren].id  = i;
        children[nChildren].val = eventPerfInfo[tree[i].dftEvent].time ;
        nChildren++;
      }
    }

    /* Calculate the children's maximum times, to see whether children will be ignored or printed */
    ierr = PetscMalloc1(nChildren,&times);CHKERRQ(ierr);
    for (i=0; i<nChildren; i++) { times[i] = children[i].val; }

    ierr = PetscMalloc1(nChildren,&maxTimes);CHKERRQ(ierr);
    ierr = MPIU_Allreduce(times, maxTimes, nChildren, MPIU_PETSCLOGDOUBLE, MPI_MAX, comm);CHKERRQ(ierr);
    ierr = PetscFree(times);CHKERRQ(ierr);

    for (i=0; i<nChildren; i++) { children[i].val = maxTimes[i]; }
    ierr = PetscFree(maxTimes);CHKERRQ(ierr);
  }

  if (!tree[iStart].own) {
  /* Set values for a timer that was not activated in this process 
   * (but was, in other processes of this run) */
    ierr = PetscMemzero(&myPerfInfo,sizeof(myPerfInfo));CHKERRQ(ierr);

    selfPerfInfo  = myPerfInfo;
    otherPerfInfo = myPerfInfo;

    parentCount   = 1;
    countsPerCall = 0;
  } else {
  /* Set the values for a timer that was activated in this process */
    int           i;
    PetscLogEvent dftEvent   = tree[iStart].dftEvent;

    parentCount    = countParents( tree, eventPerfInfo, iStart);
    myPerfInfo     = eventPerfInfo[dftEvent];
    countsPerCall  = (PetscLogDouble) myPerfInfo.count / (PetscLogDouble) parentCount;

    selfPerfInfo                = myPerfInfo;
    otherPerfInfo.time          = 0; 
    otherPerfInfo.flops         = 0;
    otherPerfInfo.numMessages   = 0;
    otherPerfInfo.messageLength = 0;
    otherPerfInfo.numReductions = 0;

    for (i=0; i<nChildren; i++) {
      /* For all child counters: subtract the child values from self-timers */

      PetscLogEvent      dftChild  = tree[children[i].id].dftEvent;
      PetscEventPerfInfo childPerfInfo = eventPerfInfo[dftChild];

      selfPerfInfo.time          -= childPerfInfo.time;
      selfPerfInfo.flops         -= childPerfInfo.flops;
      selfPerfInfo.numMessages   -= childPerfInfo.numMessages;
      selfPerfInfo.messageLength -= childPerfInfo.messageLength;
      selfPerfInfo.numReductions -= childPerfInfo.numReductions;

      if ((children[i].val/totalTime) < (threshTime/100.0)) {
        /* Add them to 'other' if the time is ignored in the output */
        otherPerfInfo.time          += childPerfInfo.time;
        otherPerfInfo.flops         += childPerfInfo.flops;
        otherPerfInfo.numMessages   += childPerfInfo.numMessages;
        otherPerfInfo.messageLength += childPerfInfo.messageLength;
        otherPerfInfo.numReductions += childPerfInfo.numReductions;
      }
    }
  }

  /* Main output for this timer */
  ierr = PetscLogPrintNestedLine(viewer, myPerfInfo, countsPerCall, parentCount, depth, name, totalTime, &wasPrinted);CHKERRQ(ierr);

  /* Now print the lines for the children */
  if (nChildren>0) {
    /* Calculate max-times for 'self' and 'other' */
    int            i;
    PetscLogDouble times[2], maxTimes[2];
    times[0] = selfPerfInfo.time;   times[1] = otherPerfInfo.time;
    ierr = MPIU_Allreduce(times, maxTimes, 2, MPIU_PETSCLOGDOUBLE, MPI_MAX, comm);CHKERRQ(ierr);
    children[nChildren+0].id = -1;
    children[nChildren+0].val = maxTimes[0];
    children[nChildren+1].id = -2;
    children[nChildren+1].val = maxTimes[1];

    /* Now sort the children (including 'self' and 'other') on total time */
    qsort(children, nChildren+2, sizeof(PetscSortItem), compareSortItems);

    /* Print (or ignore) the children in ascending order of total time */
    ierr = PetscViewerXMLStartSection(viewer,"events", NULL);CHKERRQ(ierr);
    for (i=0; i<nChildren+2; i++) {
      if ((children[i].val/totalTime) < (threshTime/100.0)) {
        /* ignored: no output */
      } else if (children[i].id==-1) {
        ierr = PetscLogPrintNestedLine(viewer, selfPerfInfo, 1, parentCount, depth+1, "self", totalTime, &childWasPrinted);CHKERRQ(ierr);
        if (childWasPrinted) {
          ierr = PetscViewerXMLEndSection(viewer,"event");CHKERRQ(ierr);
        }
      } else if (children[i].id==-2) {
        size_t  len;
        char    *otherName;

        ierr = PetscStrlen(name,&len);CHKERRQ(ierr);
        ierr = PetscMalloc1(16+len,&otherName);CHKERRQ(ierr);
        sprintf(otherName,"%s: other-timed",name);
        ierr = PetscLogPrintNestedLine(viewer, otherPerfInfo, 1, 1, depth+1, otherName, totalTime, &childWasPrinted);CHKERRQ(ierr); 
        ierr = PetscFree(otherName);CHKERRQ(ierr);
        if (childWasPrinted) {
          ierr = PetscViewerXMLEndSection(viewer,"event");CHKERRQ(ierr);
        }
      } else {
        /* Print the child with a recursive call to this function */
        ierr = PetscLogNestedPrint(viewer, tree, nTimers, children[i].id, totalTime);CHKERRQ(ierr);
      }
    }
    ierr = PetscViewerXMLEndSection(viewer,"events");CHKERRQ(ierr);
    ierr = PetscFree(children);CHKERRQ(ierr);
  }

  if (wasPrinted) {
    ierr = PetscViewerXMLEndSection(viewer, "event");CHKERRQ(ierr);
  }
  return 0;
}

static PetscErrorCode  PetscLogNestedPrintTop(PetscViewer viewer, PetscNestedEventTree *tree,int nTimers, PetscLogDouble totalTime)
{
  int                nChildren;
  PetscSortItem      *children;
  PetscErrorCode     ierr;
  PetscEventPerfInfo *eventPerfInfo;
  MPI_Comm           comm;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)viewer,&comm);CHKERRQ(ierr);
  {
  /* Look up the PerfInfo */
     const int          stage=0;
     PetscStageLog      stageLog;
     ierr          = PetscLogGetStageLog(&stageLog);CHKERRQ(ierr);
     eventPerfInfo = stageLog->stageInfo[stage].eventLog->eventInfo;
  }

  /* Count the number of child processes, and count total time */
  nChildren = 0;
  { 
    int i;
    for (i=0; i<nTimers; i++) {
      if (tree[i].depth==1) nChildren++;
    }
  }

  if (nChildren>0) {
    /* Create an array for the id-s and maxTimes of the children,
     *  leaving 2 spaces for self-time and other-time */
    int            i;
    PetscLogDouble *times, *maxTimes;

    ierr = PetscMalloc1(nChildren,&children);CHKERRQ(ierr);
    nChildren = 0;
    for (i=0; i<nTimers; i++) {
      if (tree[i].depth == 1) {
        children[nChildren].id  = i;
        children[nChildren].val = eventPerfInfo[tree[i].dftEvent].time ;
        nChildren++;
      }
    }
 
    /* Calculate the children's maximum times, to sort them */
    ierr = PetscMalloc1(nChildren,&times);CHKERRQ(ierr);
    for (i=0; i<nChildren; i++) { times[i] = children[i].val; }

    ierr = PetscMalloc1(nChildren,&maxTimes);CHKERRQ(ierr);
    ierr = MPIU_Allreduce(times, maxTimes, nChildren, MPIU_PETSCLOGDOUBLE, MPI_MAX, comm);CHKERRQ(ierr);
    ierr = PetscFree(times);CHKERRQ(ierr);

    for (i=0; i<nChildren; i++) { children[i].val = maxTimes[i]; }
    ierr = PetscFree(maxTimes);CHKERRQ(ierr);

    /* Now sort the children on total time */
    qsort(children, nChildren, sizeof(PetscSortItem), compareSortItems);
    /* Print (or ignore) the children in ascending order of total time */
    ierr = PetscViewerXMLStartSection(viewer, "timertree", "Timings tree");CHKERRQ(ierr);
    ierr = PetscViewerXMLPutDouble(viewer, "totaltime", NULL, totalTime, "%f");CHKERRQ(ierr);
    ierr = PetscViewerXMLPutDouble(viewer, "timethreshold", NULL, threshTime, "%f");CHKERRQ(ierr);

    for (i=0; i<nChildren; i++) {
      if ((children[i].val/totalTime) < (threshTime/100.0)) {
        /* ignored: no output */
      } else {
        /* Print the child with a recursive call to this function */
        ierr = PetscLogNestedPrint(viewer, tree, nTimers, children[i].id, totalTime);CHKERRQ(ierr);
      }
    }
    ierr = PetscViewerXMLEndSection(viewer, "timertree");CHKERRQ(ierr);
    ierr = PetscFree(children);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

typedef struct {
  char           *name;
  PetscLogDouble time;
  PetscLogDouble flops;
  PetscLogDouble numMessages;
  PetscLogDouble messageLength;
  PetscLogDouble numReductions;
} PetscSelfTimer;

static PetscErrorCode  PetscCalcSelfTime(PetscViewer viewer, PetscSelfTimer **p_self, int *p_nstMax)
{
  PetscErrorCode     ierr;
  PetscEventPerfInfo *eventPerfInfo;
  PetscEventRegInfo  *eventRegInfo; 
  PetscSelfTimer     *selftimes;
  PetscSelfTimer     *totaltimes;
  NestedEventId      *nstEvents;
  int                i, maxDefaultTimer;
  NestedEventId      nst;
  PetscLogEvent      dft;
  int                nstMax, nstMax_local;
  MPI_Comm           comm;
  
  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)viewer,&comm);CHKERRQ(ierr);
  {
    const int          stage=0;
    PetscStageLog      stageLog;
    ierr          = PetscLogGetStageLog(&stageLog);CHKERRQ(ierr);
    eventRegInfo  = stageLog->eventLog->eventInfo;
    eventPerfInfo = stageLog->stageInfo[stage].eventLog->eventInfo;
  }

  /* For each default timer, calculate the (one) nested timer that it corresponds to. */
  maxDefaultTimer =0;
  for (i=0; i<nNestedEvents; i++) {
    int           nParents         = nestedEvents[i].nParents;
    PetscLogEvent *dftEvents       = nestedEvents[i].dftEvents;
     int j;
     for (j=0; j<nParents; j++) {
       maxDefaultTimer = PetscMax(dftEvents[j],maxDefaultTimer);
     }
  }
  ierr = PetscMalloc1(maxDefaultTimer+1,&nstEvents);CHKERRQ(ierr);
  for (dft=0; dft<maxDefaultTimer; dft++) {nstEvents[dft] = 0;}
  for (i=0; i<nNestedEvents; i++) {
    int           nParents          = nestedEvents[i].nParents;
    NestedEventId nstEvent          = nestedEvents[i].nstEvent;
    PetscLogEvent *dftEvents        = nestedEvents[i].dftEvents;
    int           j;
    for (j=0; j<nParents; j++) { nstEvents[dftEvents[j]] = nstEvent; }
  }

  /* Calculate largest nested event-ID */
  nstMax_local = 0;
  for (i=0; i<nNestedEvents; i++) { if (nestedEvents[i].nstEvent>nstMax_local) {nstMax_local = nestedEvents[i].nstEvent;} }
  ierr = MPIU_Allreduce(&nstMax_local, &nstMax, 1, MPI_INT, MPI_MAX, comm);CHKERRQ(ierr);


  /* Initialize all total-times with zero */
  ierr = PetscMalloc1(nstMax+1,&selftimes);CHKERRQ(ierr);
  ierr = PetscMalloc1(nstMax+1,&totaltimes);CHKERRQ(ierr);
  for (nst=0; nst<=nstMax; nst++) { 
    totaltimes[nst].time          = 0;
    totaltimes[nst].flops         = 0;
    totaltimes[nst].numMessages   = 0;
    totaltimes[nst].messageLength = 0;
    totaltimes[nst].numReductions = 0;
    totaltimes[nst].name          = NULL;
  }

  /* Calculate total-times */
  for (i=0; i<nNestedEvents; i++) { 
    const int            nParents  = nestedEvents[i].nParents;
    const NestedEventId  nstEvent  = nestedEvents[i].nstEvent;
    const PetscLogEvent *dftEvents = nestedEvents[i].dftEvents;
    int                  j;
    for (j=0; j<nParents; j++) {
      const PetscLogEvent dftEvent = dftEvents[j];
      totaltimes[nstEvent].time          += eventPerfInfo[dftEvent].time;
      totaltimes[nstEvent].flops         += eventPerfInfo[dftEvent].flops;
      totaltimes[nstEvent].numMessages   += eventPerfInfo[dftEvent].numMessages;
      totaltimes[nstEvent].messageLength += eventPerfInfo[dftEvent].messageLength;
      totaltimes[nstEvent].numReductions += eventPerfInfo[dftEvent].numReductions;
    }
    totaltimes[nstEvent].name    = eventRegInfo[(PetscLogEvent) nstEvent].name;
  }

  /* Initialize: self-times := totaltimes */
  for (nst=0; nst<=nstMax; nst++) { selftimes[nst] = totaltimes[nst]; }

  /* Subtract timed subprocesses from self-times */
  for (i=0; i<nNestedEvents; i++) { 
    const int           nParents          = nestedEvents[i].nParents;
    const PetscLogEvent *dftEvents        = nestedEvents[i].dftEvents;
    const NestedEventId *dftParentsSorted = nestedEvents[i].dftParentsSorted;
    int                 j;
    for (j=0; j<nParents; j++) {
      if (dftParentsSorted[j] != DFT_ID_AWAKE) {
        const PetscLogEvent dftEvent  = dftEvents[j];
        const NestedEventId nstParent = nstEvents[dftParentsSorted[j]];
        selftimes[nstParent].time          -= eventPerfInfo[dftEvent].time;
        selftimes[nstParent].flops         -= eventPerfInfo[dftEvent].flops;
        selftimes[nstParent].numMessages   -= eventPerfInfo[dftEvent].numMessages;
        selftimes[nstParent].messageLength -= eventPerfInfo[dftEvent].messageLength;
        selftimes[nstParent].numReductions -= eventPerfInfo[dftEvent].numReductions;
      }
    }
  }

  ierr = PetscFree(nstEvents);CHKERRQ(ierr);
  ierr = PetscFree(totaltimes);CHKERRQ(ierr);

  /* Set outputs */
  *p_self  = selftimes;
  *p_nstMax = nstMax;
  PetscFunctionReturn(0);
}

static PetscErrorCode  PetscPrintSelfTime(PetscViewer viewer, const PetscSelfTimer *selftimes, int nstMax, PetscLogDouble totalTime)
{
  PetscErrorCode     ierr;
  int                i;
  NestedEventId      nst;
  PetscSortItem      *sortSelfTimes;
  PetscLogDouble     *times, *maxTimes;
  PetscEventRegInfo  *eventRegInfo; 
  const int          dum_depth = 1, dum_count=1, dum_parentcount=1;
  PetscBool          wasPrinted;
  MPI_Comm           comm;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)viewer,&comm);CHKERRQ(ierr);
  {
    PetscStageLog      stageLog;
    ierr          = PetscLogGetStageLog(&stageLog);CHKERRQ(ierr);
    eventRegInfo  = stageLog->eventLog->eventInfo;
  }

  ierr = PetscMalloc1(nstMax+1,&times);CHKERRQ(ierr);
  ierr = PetscMalloc1(nstMax+1,&maxTimes);CHKERRQ(ierr);
  for (nst=0; nst<=nstMax; nst++) { times[nst] = selftimes[nst].time;}
  ierr = MPIU_Allreduce(times, maxTimes, nstMax+1, MPIU_PETSCLOGDOUBLE, MPI_MAX, comm);CHKERRQ(ierr);
  ierr = PetscFree(times);CHKERRQ(ierr);

  ierr = PetscMalloc1(nstMax+1,&sortSelfTimes);CHKERRQ(ierr);

  /* Sort the self-timers on basis of the largest time needed */
  for (nst=0; nst<=nstMax; nst++) {
    sortSelfTimes[nst].id  = nst;
    sortSelfTimes[nst].val = maxTimes[nst];
  }
  ierr = PetscFree(maxTimes);CHKERRQ(ierr);
  qsort(sortSelfTimes, nstMax+1, sizeof(PetscSortItem), compareSortItems);

  ierr = PetscViewerXMLStartSection(viewer, "selftimertable", "Self-timings");CHKERRQ(ierr);
  ierr = PetscViewerXMLPutDouble(viewer, "totaltime", NULL, totalTime, "%f");CHKERRQ(ierr);

  for (i=0; i<=nstMax; i++) {
    if ((sortSelfTimes[i].val/totalTime) >= (threshTime/100.0)) {
      NestedEventId      nstEvent = sortSelfTimes[i].id;
      PetscEventPerfInfo selfPerfInfo;
      const char         *name     = eventRegInfo[(PetscLogEvent) nstEvent].name;

      selfPerfInfo.time          = selftimes[nstEvent].time ;
      selfPerfInfo.flops         = selftimes[nstEvent].flops;
      selfPerfInfo.numMessages   = selftimes[nstEvent].numMessages;
      selfPerfInfo.messageLength = selftimes[nstEvent].messageLength;
      selfPerfInfo.numReductions = selftimes[nstEvent].numReductions;
      
      ierr = PetscLogPrintNestedLine(viewer, selfPerfInfo, dum_count, dum_parentcount, dum_depth, name, totalTime, &wasPrinted);CHKERRQ(ierr);
      if (wasPrinted){
        ierr = PetscViewerXMLEndSection(viewer, "event");CHKERRQ(ierr);
      }
    }
  }
  ierr = PetscViewerXMLEndSection(viewer, "selftimertable");CHKERRQ(ierr);
  ierr = PetscFree(sortSelfTimes);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode  PetscLogView_Nested(PetscViewer viewer)
{
  MPI_Comm           comm;
  PetscErrorCode     ierr;
  PetscLogDouble     locTotalTime, globTotalTime;
  PetscNestedEventTree *tree = NULL;
  PetscSelfTimer     *selftimers = NULL;
  int                nTimers = 0, nstMax = 0;
  PetscViewerType    vType;

  PetscFunctionBegin;
  ierr = PetscViewerGetType(viewer,&vType);CHKERRQ(ierr);

  /* Set useXMLFormat that controls the format in all local PetscPrint.. functions */
  ierr = PetscViewerInitASCII_XML(viewer);CHKERRQ(ierr);

  ierr = PetscObjectGetComm((PetscObject)viewer,&comm);CHKERRQ(ierr);

  ierr = PetscViewerASCIIPrintf(viewer, "<!-- PETSc Performance Summary: -->\n");CHKERRQ(ierr);
  ierr = PetscViewerXMLStartSection(viewer, "petscroot", NULL);CHKERRQ(ierr);

  /* Get the total elapsed time, local and global maximum */
  ierr = PetscTime(&locTotalTime);CHKERRQ(ierr);  locTotalTime -= petsc_BaseTime;
  ierr = MPIU_Allreduce(&locTotalTime, &globTotalTime, 1, MPIU_PETSCLOGDOUBLE, MPI_MAX, comm);CHKERRQ(ierr);

  /* Print global information about this run */
  ierr = PetscPrintExeSpecs(viewer);CHKERRQ(ierr);
  ierr = PetscPrintGlobalPerformance(viewer, locTotalTime);CHKERRQ(ierr);
  
  /* Collect nested timer tree info from all processes */
  ierr = PetscCreateLogTreeNested(viewer, &tree, &nTimers);CHKERRQ(ierr);
  ierr = PetscLogNestedPrintTop(viewer, tree, nTimers, globTotalTime);CHKERRQ(ierr);
  ierr = PetscLogFreeNestedTree(tree, nTimers);CHKERRQ(ierr);

  /* Calculate self-time for all (not-nested) events */
  ierr = PetscCalcSelfTime(viewer, &selftimers, &nstMax);CHKERRQ(ierr);
  ierr = PetscPrintSelfTime(viewer, selftimers, nstMax, globTotalTime);CHKERRQ(ierr);
  ierr = PetscFree(selftimers);CHKERRQ(ierr);

  ierr = PetscViewerXMLEndSection(viewer, "petscroot");CHKERRQ(ierr);
  ierr = PetscViewerFinalASCII_XML(viewer);CHKERRQ(ierr);
  ierr = PetscLogNestedEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#endif
