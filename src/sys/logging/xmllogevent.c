/*************************************************************************************
 *    M A R I T I M E  R E S E A R C H  I N S T I T U T E  N E T H E R L A N D S     *
 *************************************************************************************
 *    authors: Bas van 't Hof, Koos Huijssen, Christiaan M. Klaij                    *
 *************************************************************************************
 *    content: Support for nested PetscTimers                                        *
 *************************************************************************************/
#include <petsclog.h>                  /*I "petsclog.h" I*/
#include <petsc/private/logimpl.h>
#include <petsctime.h>
#include <petscviewer.h>
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

static PetscLogEvent    dftParentActive        = DFT_ID_AWAKE;
static int              nNestedEvents          = 0;
static int              nNestedEventsAllocated = 0;
static PetscNestedEvent *nestedEvents          = NULL;
static PetscLogDouble   thresholdTime          = 0.01; /* initial value was 0.1 */

#define THRESHOLD (thresholdTime/100.0+1e-12)

static PetscErrorCode PetscLogEventBeginNested(NestedEventId nstEvent, int t, PetscObject o1, PetscObject o2, PetscObject o3, PetscObject o4);
static PetscErrorCode PetscLogEventEndNested(NestedEventId nstEvent, int t, PetscObject o1, PetscObject o2, PetscObject o3, PetscObject o4);
PETSC_INTERN PetscErrorCode PetscLogView_Nested(PetscViewer);
PETSC_INTERN PetscErrorCode PetscLogView_Flamegraph(PetscViewer);

/*@C
  PetscLogNestedBegin - Turns on nested logging of objects and events. This logs flop
  rates and object creation and should not slow programs down too much.

  Logically Collective over PETSC_COMM_WORLD

  Options Database Keys:
. -log_view :filename.xml:ascii_xml - Prints an XML summary of flop and timing information to the file

  Usage:
.vb
      PetscInitialize(...);
      PetscLogNestedBegin();
       ... code ...
      PetscLogView(viewer);
      PetscFinalize();
.ve

  Level: advanced

.seealso: PetscLogDump(), PetscLogAllBegin(), PetscLogView(), PetscLogTraceBegin(), PetscLogDefaultBegin()
@*/
PetscErrorCode PetscLogNestedBegin(void)
{
  PetscFunctionBegin;
  PetscCheck(!nestedEvents,PETSC_COMM_SELF,PETSC_ERR_COR,"nestedEvents already allocated");

  nNestedEventsAllocated = 10;
  CHKERRQ(PetscMalloc1(nNestedEventsAllocated,&nestedEvents));
  dftParentActive = DFT_ID_AWAKE;
  nNestedEvents =1;

  /* 'Awake' is nested event 0. It has no parents */
  nestedEvents[0].nstEvent          = 0;
  nestedEvents[0].nParents          = 0;
  nestedEvents[0].dftParentsSorted  = NULL;
  nestedEvents[0].dftEvents         = NULL;
  nestedEvents[0].dftParents        = NULL;
  nestedEvents[0].dftEventsSorted   = NULL;

  CHKERRQ(PetscLogSet(PetscLogEventBeginNested,PetscLogEventEndNested));
  PetscFunctionReturn(0);
}

/* Delete the data structures for the nested timers */
PetscErrorCode PetscLogNestedEnd(void)
{
  int            i;

  PetscFunctionBegin;
  if (!nestedEvents) PetscFunctionReturn(0);
  for (i=0; i<nNestedEvents; i++) {
    CHKERRQ(PetscFree4(nestedEvents[i].dftParentsSorted,nestedEvents[i].dftEventsSorted,nestedEvents[i].dftParents,nestedEvents[i].dftEvents));
  }
  CHKERRQ(PetscFree(nestedEvents));
  nestedEvents           = NULL;
  nNestedEvents          = 0;
  nNestedEventsAllocated = 0;
  PetscFunctionReturn(0);
}

/*
 UTILITIES: FIND STUFF IN SORTED ARRAYS

    dftIndex - index to be found
    dftArray - sorted array of PetscLogEvent-ids
    narray - dimension of dftArray
    entry - entry in the array where dftIndex may be found;

     if dftArray[entry] != dftIndex, then dftIndex is not part of dftArray
     In that case, the dftIndex can be inserted at this entry.
*/
static PetscErrorCode PetscLogEventFindDefaultTimer(PetscLogEvent dftIndex,const PetscLogEvent *dftArray,int narray,int *entry)
{
  PetscFunctionBegin;
  if (narray==0 || dftIndex <= dftArray[0]) {
    *entry = 0;
  } else if (dftIndex > dftArray[narray-1]) {
    *entry = narray;
  } else {
    int ihigh = narray-1, ilow=0;
    while (ihigh>ilow) {
      const int imiddle = (ihigh+ilow)/2;
      if (dftArray[imiddle] > dftIndex) {
        ihigh = imiddle;
      } else if (dftArray[imiddle]<dftIndex) {
        ilow = imiddle+1;
      } else {
        ihigh = imiddle;
        ilow  = imiddle;
      }
    }
    *entry = ihigh;
  }
  PetscFunctionReturn(0);
}

/*
    Utility: find the nested event with given identification

    nstEvent - Nested event to be found
    entry - entry in the nestedEvents where nstEvent may be found;

    if nestedEvents[entry].nstEvent != nstEvent, then index is not part of iarray
*/
static PetscErrorCode PetscLogEventFindNestedTimer(NestedEventId nstEvent,int *entry)
{
  PetscFunctionBegin;
  if (nNestedEvents==0 || nstEvent <= nestedEvents[0].nstEvent) {
    *entry = 0;
  } else if (nstEvent > nestedEvents[nNestedEvents-1].nstEvent) {
    *entry = nNestedEvents;
  } else {
    int ihigh = nNestedEvents-1,  ilow = 0;
    while (ihigh>ilow) {
      const int imiddle = (ihigh+ilow)/2;
      if (nestedEvents[imiddle].nstEvent > nstEvent) {
        ihigh = imiddle;
      } else if (nestedEvents[imiddle].nstEvent<nstEvent) {
        ilow = imiddle+1;
      } else {
        ihigh = imiddle;
        ilow  = imiddle;
      }
    }
    *entry = ihigh;
  }
  PetscFunctionReturn(0);
}

/*
 Nested logging is not prepared yet to support user-defined logging stages, so for now we force logging on the main stage.
 Using PetscLogStage{Push/Pop}() would be more appropriate, but these two calls do extra bookkeeping work we don't need.
*/

#define MAINSTAGE 0

static PetscLogStage savedStage = 0;

static inline PetscErrorCode PetscLogStageOverride(void)
{
  PetscStageLog  stageLog = petsc_stageLog;

  PetscFunctionBegin;
  if (stageLog->curStage == MAINSTAGE) PetscFunctionReturn(0);
  savedStage = stageLog->curStage;
  stageLog->curStage = MAINSTAGE;
  CHKERRQ(PetscIntStackPush(stageLog->stack, MAINSTAGE));
  PetscFunctionReturn(0);
}

static inline PetscErrorCode PetscLogStageRestore(void)
{
  PetscStageLog  stageLog = petsc_stageLog;

  PetscFunctionBegin;
  if (savedStage == MAINSTAGE) PetscFunctionReturn(0);
  stageLog->curStage = savedStage;
  CHKERRQ(PetscIntStackPop(stageLog->stack, &savedStage));
  PetscFunctionReturn(0);
}

/******************************************************************************************/
/* Start a nested event */
static PetscErrorCode PetscLogEventBeginNested(NestedEventId nstEvent, int t, PetscObject o1, PetscObject o2, PetscObject o3, PetscObject o4)
{
  int             entry, pentry, tentry,i;
  PetscLogEvent   dftEvent;

  PetscFunctionBegin;
  CHKERRQ(PetscLogEventFindNestedTimer(nstEvent, &entry));
  if (entry>=nNestedEvents || nestedEvents[entry].nstEvent != nstEvent) {
    /* Nested event doesn't exist yet: create it */

    if (nNestedEvents==nNestedEventsAllocated) {
      /* Enlarge and re-allocate nestedEvents if needed */
      PetscNestedEvent *tmp = nestedEvents;
      CHKERRQ(PetscMalloc1(2*nNestedEvents,&nestedEvents));
      nNestedEventsAllocated*=2;
      CHKERRQ(PetscArraycpy(nestedEvents, tmp, nNestedEvents));
      CHKERRQ(PetscFree(tmp));
    }

    /* Clear space in nestedEvents for new nested event */
    nNestedEvents++;
    for (i = nNestedEvents-1; i>entry; i--) {
      nestedEvents[i] = nestedEvents[i-1];
    }

    /* Create event in nestedEvents */
    nestedEvents[entry].nstEvent = nstEvent;
    nestedEvents[entry].nParents=1;
    CHKERRQ(PetscMalloc4(1,&nestedEvents[entry].dftParentsSorted,1,&nestedEvents[entry].dftEventsSorted,1,&nestedEvents[entry].dftParents,1,&nestedEvents[entry].dftEvents));

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

    CHKERRQ(PetscLogEventFindDefaultTimer( dftParentActive, dftParentsSorted, nParents, &pentry));

    if (pentry>=nParents || dftParentActive != dftParentsSorted[pentry]) {
      /* dftParentActive not in the list: add it to the list */
      int           i;
      PetscLogEvent *dftParents      = nestedEvents[entry].dftParents;
      PetscLogEvent *dftEventsSorted = nestedEvents[entry].dftEventsSorted;
      char          name[100];

      /* Register a new default timer */
      sprintf(name, "%d -> %d", (int) dftParentActive, (int) nstEvent);
      CHKERRQ(PetscLogEventRegister(name, 0, &dftEvent));
      CHKERRQ(PetscLogEventFindDefaultTimer( dftEvent, dftEventsSorted, nParents, &tentry));

      /* Reallocate parents and dftEvents to make space for new parent */
      CHKERRQ(PetscMalloc4(1+nParents,&nestedEvents[entry].dftParentsSorted,1+nParents,&nestedEvents[entry].dftEventsSorted,1+nParents,&nestedEvents[entry].dftParents,1+nParents,&nestedEvents[entry].dftEvents));
      CHKERRQ(PetscArraycpy(nestedEvents[entry].dftParentsSorted, dftParentsSorted, nParents));
      CHKERRQ(PetscArraycpy(nestedEvents[entry].dftEventsSorted,  dftEventsSorted,  nParents));
      CHKERRQ(PetscArraycpy(nestedEvents[entry].dftParents,       dftParents,       nParents));
      CHKERRQ(PetscArraycpy(nestedEvents[entry].dftEvents,        dftEvents,        nParents));
      CHKERRQ(PetscFree4(dftParentsSorted,dftEventsSorted,dftParents,dftEvents));

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
  CHKERRQ(PetscLogStageOverride());
  CHKERRQ(PetscLogEventBeginDefault(dftEvent,t,o1,o2,o3,o4));
  CHKERRQ(PetscLogStageRestore());
  dftParentActive = dftEvent;
  PetscFunctionReturn(0);
}

/* End a nested event */
static PetscErrorCode PetscLogEventEndNested(NestedEventId nstEvent, int t, PetscObject o1, PetscObject o2, PetscObject o3, PetscObject o4)
{
  int             entry, pentry, nParents;
  PetscLogEvent  *dftEventsSorted;

  PetscFunctionBegin;
  /* Find the nested event */
  CHKERRQ(PetscLogEventFindNestedTimer(nstEvent, &entry));
  PetscCheckFalse(entry>=nNestedEvents,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE, "Logging event %d larger than number of events %d",entry,nNestedEvents);
  PetscCheckFalse(nestedEvents[entry].nstEvent != nstEvent,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE, "Logging event %d had unbalanced begin/end pairs does not match %d",entry,nstEvent);
  dftEventsSorted = nestedEvents[entry].dftEventsSorted;
  nParents        = nestedEvents[entry].nParents;

  /* Find the current default timer among the 'dftEvents' of this event */
  CHKERRQ(PetscLogEventFindDefaultTimer( dftParentActive, dftEventsSorted, nParents, &pentry));

  PetscCheckFalse(pentry>=nParents,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE, "Entry %d is larger than number of parents %d",pentry,nParents);
  PetscCheckFalse(dftEventsSorted[pentry] != dftParentActive,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE, "Active parent is %d, but we seem to be closing %d",dftParentActive,dftEventsSorted[pentry]);

  /* Stop the default timer and update the dftParentActive */
  CHKERRQ(PetscLogStageOverride());
  CHKERRQ(PetscLogEventEndDefault(dftParentActive,t,o1,o2,o3,o4));
  CHKERRQ(PetscLogStageRestore());
  dftParentActive = nestedEvents[entry].dftParents[pentry];
  PetscFunctionReturn(0);
}

/*@
   PetscLogSetThreshold - Set the threshold time for logging the events; this is a percentage out of 100, so 1. means any event
          that takes 1 or more percent of the time.

  Logically Collective over PETSC_COMM_WORLD

  Input Parameter:
.   newThresh - the threshold to use

  Output Parameter:
.   oldThresh - the previously set threshold value

  Options Database Keys:
. -log_view :filename.xml:ascii_xml - Prints an XML summary of flop and timing information to the file

  Usage:
.vb
      PetscInitialize(...);
      PetscLogNestedBegin();
      PetscLogSetThreshold(0.1,&oldthresh);
       ... code ...
      PetscLogView(viewer);
      PetscFinalize();
.ve

  Level: advanced

.seealso: PetscLogDump(), PetscLogAllBegin(), PetscLogView(), PetscLogTraceBegin(), PetscLogDefaultBegin(),
          PetscLogNestedBegin()
@*/
PetscErrorCode PetscLogSetThreshold(PetscLogDouble newThresh, PetscLogDouble *oldThresh)
{
  PetscFunctionBegin;
  if (oldThresh) *oldThresh = thresholdTime;
  if (newThresh == PETSC_DECIDE)  newThresh = 0.01;
  if (newThresh == PETSC_DEFAULT) newThresh = 0.01;
  thresholdTime = PetscMax(newThresh, 0.0);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscPrintExeSpecs(PetscViewer viewer)
{
  char               arch[128],hostname[128],username[128],pname[PETSC_MAX_PATH_LEN],date[128];
  char               version[256], buildoptions[128] = "";
  PetscMPIInt        size;
  size_t             len;

  PetscFunctionBegin;
  CHKERRMPI(MPI_Comm_size(PetscObjectComm((PetscObject)viewer),&size));
  CHKERRQ(PetscGetArchType(arch,sizeof(arch)));
  CHKERRQ(PetscGetHostName(hostname,sizeof(hostname)));
  CHKERRQ(PetscGetUserName(username,sizeof(username)));
  CHKERRQ(PetscGetProgramName(pname,sizeof(pname)));
  CHKERRQ(PetscGetDate(date,sizeof(date)));
  CHKERRQ(PetscGetVersion(version,sizeof(version)));

  CHKERRQ(PetscViewerXMLStartSection(viewer, "runspecification", "Run Specification"));
  CHKERRQ(PetscViewerXMLPutString(   viewer, "executable"  , "Executable"   , pname));
  CHKERRQ(PetscViewerXMLPutString(   viewer, "architecture", "Architecture" , arch));
  CHKERRQ(PetscViewerXMLPutString(   viewer, "hostname"    , "Host"         , hostname));
  CHKERRQ(PetscViewerXMLPutInt(      viewer, "nprocesses"  , "Number of processes", size));
  CHKERRQ(PetscViewerXMLPutString(   viewer, "user"        , "Run by user"  , username));
  CHKERRQ(PetscViewerXMLPutString(   viewer, "date"        , "Started at"   , date));
  CHKERRQ(PetscViewerXMLPutString(   viewer, "petscrelease", "Petsc Release", version));

  if (PetscDefined(USE_DEBUG)) {
    CHKERRQ(PetscStrlcat(buildoptions, "Debug ", sizeof(buildoptions)));
  }
  if (PetscDefined(USE_COMPLEX)) {
    CHKERRQ(PetscStrlcat(buildoptions, "Complex ", sizeof(buildoptions)));
  }
  if (PetscDefined(USE_REAL_SINGLE)) {
    CHKERRQ(PetscStrlcat(buildoptions, "Single ", sizeof(buildoptions)));
  } else if (PetscDefined(USE_REAL___FLOAT128)) {
    CHKERRQ(PetscStrlcat(buildoptions, "Quadruple ", sizeof(buildoptions)));
  } else if (PetscDefined(USE_REAL___FP16)) {
    CHKERRQ(PetscStrlcat(buildoptions, "Half ", sizeof(buildoptions)));
  }
  if (PetscDefined(USE_64BIT_INDICES)) {
    CHKERRQ(PetscStrlcat(buildoptions, "Int64 ", sizeof(buildoptions)));
  }
#if defined(__cplusplus)
  CHKERRQ(PetscStrlcat(buildoptions, "C++ ", sizeof(buildoptions)));
#endif
  CHKERRQ(PetscStrlen(buildoptions,&len));
  if (len) {
    CHKERRQ(PetscViewerXMLPutString(viewer, "petscbuildoptions", "Petsc build options", buildoptions));
  }
  CHKERRQ(PetscViewerXMLEndSection(viewer, "runspecification"));
  PetscFunctionReturn(0);
}

/* Print the global performance: max, max/min, average and total of
 *      time, objects, flops, flops/sec, memory, MPI messages, MPI message lengths, MPI reductions.
 */
static PetscErrorCode PetscPrintXMLGlobalPerformanceElement(PetscViewer viewer, const char *name, const char *desc, PetscLogDouble local_val, const PetscBool print_average, const PetscBool print_total)
{
  PetscLogDouble  min, tot, ratio, avg;
  MPI_Comm        comm;
  PetscMPIInt     rank, size;
  PetscLogDouble  valrank[2], max[2];

  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetComm((PetscObject)viewer,&comm));
  CHKERRMPI(MPI_Comm_size(PetscObjectComm((PetscObject)viewer),&size));
  CHKERRMPI(MPI_Comm_rank(comm, &rank));

  valrank[0] = local_val;
  valrank[1] = (PetscLogDouble) rank;
  CHKERRMPI(MPIU_Allreduce(&local_val, &min, 1, MPIU_PETSCLOGDOUBLE,  MPI_MIN,    comm));
  CHKERRMPI(MPIU_Allreduce(valrank,    &max, 1, MPIU_2PETSCLOGDOUBLE, MPI_MAXLOC, comm));
  CHKERRMPI(MPIU_Allreduce(&local_val, &tot, 1, MPIU_PETSCLOGDOUBLE,  MPI_SUM,    comm));
  avg  = tot/((PetscLogDouble) size);
  if (min != 0.0) ratio = max[0]/min;
  else ratio = 0.0;

  CHKERRQ(PetscViewerXMLStartSection(viewer, name, desc));
  CHKERRQ(PetscViewerXMLPutDouble(viewer, "max", NULL, max[0], "%e"));
  CHKERRQ(PetscViewerXMLPutInt(   viewer, "maxrank"  , "rank at which max was found" , (PetscMPIInt) max[1]));
  CHKERRQ(PetscViewerXMLPutDouble(viewer, "ratio", NULL, ratio, "%f"));
  if (print_average) {
    CHKERRQ(PetscViewerXMLPutDouble(viewer, "average", NULL, avg, "%e"));
  }
  if (print_total) {
    CHKERRQ(PetscViewerXMLPutDouble(viewer, "total", NULL, tot, "%e"));
  }
  CHKERRQ(PetscViewerXMLEndSection(viewer, name));
  PetscFunctionReturn(0);
}

/* Print the global performance: max, max/min, average and total of
 *      time, objects, flops, flops/sec, memory, MPI messages, MPI message lengths, MPI reductions.
 */
static PetscErrorCode PetscPrintGlobalPerformance(PetscViewer viewer, PetscLogDouble locTotalTime)
{
  PetscLogDouble  flops, mem, red, mess;
  const PetscBool print_total_yes   = PETSC_TRUE,
                  print_total_no    = PETSC_FALSE,
                  print_average_no  = PETSC_FALSE,
                  print_average_yes = PETSC_TRUE;

  PetscFunctionBegin;
  /* Must preserve reduction count before we go on */
  red = petsc_allreduce_ct + petsc_gather_ct + petsc_scatter_ct;

  /* Calculate summary information */
  CHKERRQ(PetscViewerXMLStartSection(viewer, "globalperformance", "Global performance"));

  /*   Time */
  CHKERRQ(PetscPrintXMLGlobalPerformanceElement(viewer, "time", "Time (sec)", locTotalTime, print_average_yes, print_total_no));

  /*   Objects */
  CHKERRQ(PetscPrintXMLGlobalPerformanceElement(viewer, "objects", "Objects", (PetscLogDouble) petsc_numObjects, print_average_yes, print_total_no));

  /*   Flop */
  CHKERRQ(PetscPrintXMLGlobalPerformanceElement(viewer, "mflop", "MFlop", petsc_TotalFlops/1.0E6, print_average_yes, print_total_yes));

  /*   Flop/sec -- Must talk to Barry here */
  if (locTotalTime != 0.0) flops = petsc_TotalFlops/locTotalTime;
  else flops = 0.0;
  CHKERRQ(PetscPrintXMLGlobalPerformanceElement(viewer, "mflops", "MFlop/sec", flops/1.0E6, print_average_yes, print_total_yes));

  /*   Memory */
  CHKERRQ(PetscMallocGetMaximumUsage(&mem));
  if (mem > 0.0) {
    CHKERRQ(PetscPrintXMLGlobalPerformanceElement(viewer, "memory", "Memory (MiB)", mem/1024.0/1024.0, print_average_yes, print_total_yes));
  }
  /*   Messages */
  mess = 0.5*(petsc_irecv_ct + petsc_isend_ct + petsc_recv_ct + petsc_send_ct);
  CHKERRQ(PetscPrintXMLGlobalPerformanceElement(viewer, "messagetransfers", "MPI Message Transfers", mess, print_average_yes, print_total_yes));

  /*   Message Volume */
  mess = 0.5*(petsc_irecv_len + petsc_isend_len + petsc_recv_len + petsc_send_len);
  CHKERRQ(PetscPrintXMLGlobalPerformanceElement(viewer, "messagevolume", "MPI Message Volume (MiB)", mess/1024.0/1024.0, print_average_yes, print_total_yes));

  /*   Reductions */
  CHKERRQ(PetscPrintXMLGlobalPerformanceElement(viewer, "reductions", "MPI Reductions", red , print_average_no, print_total_no));
  CHKERRQ(PetscViewerXMLEndSection(viewer, "globalperformance"));
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
static PetscErrorCode PetscLogNestedTreeCreate(PetscViewer viewer, PetscNestedEventTree **p_tree, int *p_nTimers)
{
  PetscNestedEventTree *tree = NULL, *newTree;
  int                  *treeIndices;
  int                  nTimers, totalNTimers, i, j, iTimer0, maxDefaultTimer;
  int                  yesno;
  PetscBool            done;
  int                  maxdepth;
  int                  depth;
  int                  illegalEvent;
  int                  iextra;
  NestedEventId        *nstPath, *nstMyPath;
  MPI_Comm             comm;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetComm((PetscObject)viewer,&comm));

  /* Calculate memory needed to store everybody's information and allocate tree */
  nTimers = 0;
  for (i=0; i<nNestedEvents; i++) nTimers += nestedEvents[i].nParents;

  CHKERRQ(PetscMalloc1(nTimers,&tree));

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
  CHKERRMPI(MPIU_Allreduce(&maxDefaultTimer, &j, 1, MPI_INT, MPI_MAX, comm));
  maxDefaultTimer = j;

  /* Find default timer's place in the tree */
  CHKERRQ(PetscCalloc1(maxDefaultTimer+1,&treeIndices));
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
      tree[i].nstParent = tree[j].nstEvent;
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
    CHKERRQ(PetscCalloc1(depth,&tree[i].nstPath));
  }

  /* Calculate the paths for all timers */
  for (depth=1; depth<=maxdepth; depth++) {
    for (i=0; i<nTimers; i++) {
      if (tree[i].depth==depth) {
        if (depth>1) {
          int    j = treeIndices[tree[i].dftParent];
          CHKERRQ(PetscArraycpy(tree[i].nstPath,tree[j].nstPath,depth-1));
        }
        tree[i].nstPath[depth-1] = tree[i].nstEvent;
      }
    }
  }
  CHKERRQ(PetscFree(treeIndices));

  /* Sort the tree on basis of the paths */
  qsort(tree, nTimers, sizeof(PetscNestedEventTree), compareTreeItems);

  /* Allocate an array to store paths */
  depth = maxdepth;
  CHKERRMPI(MPIU_Allreduce(&depth, &maxdepth, 1, MPI_INT, MPI_MAX, comm));
  CHKERRQ(PetscMalloc1(maxdepth+1, &nstPath));
  CHKERRQ(PetscMalloc1(maxdepth+1, &nstMyPath));

  /* Find an illegal nested event index (1+largest nested event index) */
  illegalEvent = 1+nestedEvents[nNestedEvents-1].nstEvent;
  i = illegalEvent;
  CHKERRMPI(MPIU_Allreduce(&i, &illegalEvent, 1, MPI_INT, MPI_MAX, comm));

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
      PetscCheckFalse(depth > maxdepth + 1,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Depth %d > maxdepth+1 %d",depth,maxdepth+1);
      if (i<nTimers) {
        for (j=0;             j<tree[i].depth; j++) nstMyPath[j] = tree[i].nstPath[j];
        for (j=tree[i].depth; j<depth;         j++) nstMyPath[j] = illegalEvent;
      } else {
        for (j=0;             j<depth;         j++) nstMyPath[j] = illegalEvent;
      }

      /* Communicate with other processes to obtain the next path and its depth */
      CHKERRMPI(MPIU_Allreduce(nstMyPath, nstPath, depth, MPI_INT, MPI_MIN, comm));
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
            CHKERRQ(PetscMalloc1(depth, &newTree[i+iextra].nstPath));
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
      CHKERRQ(PetscMalloc1(totalNTimers, &newTree));
    }
  }
  CHKERRQ(PetscFree(nstPath));
  CHKERRQ(PetscFree(nstMyPath));
  CHKERRQ(PetscFree(tree));
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
static PetscErrorCode PetscLogNestedTreeDestroy(PetscNestedEventTree *tree, int nTimers)
{
  int             i;

  PetscFunctionBegin;
  for (i=0; i<nTimers; i++) {
    CHKERRQ(PetscFree(tree[i].nstPath));
  }
  CHKERRQ(PetscFree(tree));
  PetscFunctionReturn(0);
}

/* Print the global performance: max, max/min, average and total of
 *      time, objects, flops, flops/sec, memory, MPI messages, MPI message lengths, MPI reductions.
 */
static PetscErrorCode PetscPrintXMLNestedLinePerfResults(PetscViewer viewer,const char *name,PetscLogDouble value,PetscLogDouble minthreshold,PetscLogDouble maxthreshold,PetscLogDouble minmaxtreshold)
{
  MPI_Comm       comm;                          /* MPI communicator in reduction */
  PetscMPIInt    rank;                          /* rank of this process */
  PetscLogDouble val_in[2], max[2], min[2];
  PetscLogDouble minvalue, maxvalue, tot;
  PetscMPIInt    size;
  PetscMPIInt    minLoc, maxLoc;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetComm((PetscObject)viewer,&comm));
  CHKERRMPI(MPI_Comm_size(comm, &size));
  CHKERRMPI(MPI_Comm_rank(comm, &rank));
  val_in[0] = value;
  val_in[1] = (PetscLogDouble) rank;
  CHKERRMPI(MPIU_Allreduce(val_in, max,  1, MPIU_2PETSCLOGDOUBLE, MPI_MAXLOC, comm));
  CHKERRMPI(MPIU_Allreduce(val_in, min,  1, MPIU_2PETSCLOGDOUBLE, MPI_MINLOC, comm));
  maxvalue = max[0];
  maxLoc   = (PetscMPIInt) max[1];
  minvalue = min[0];
  minLoc   = (PetscMPIInt) min[1];
  CHKERRMPI(MPIU_Allreduce(&value, &tot, 1, MPIU_PETSCLOGDOUBLE,  MPI_SUM,    comm));

  if (maxvalue<maxthreshold && minvalue>=minthreshold) {
    /* One call per parent or NO value: don't print */
  } else {
     CHKERRQ(PetscViewerXMLStartSection(viewer, name, NULL));
     if (maxvalue>minvalue*minmaxtreshold) {
       CHKERRQ(PetscViewerXMLPutDouble(viewer, "avgvalue", NULL, tot/size, "%g"));
       CHKERRQ(PetscViewerXMLPutDouble(viewer, "minvalue", NULL, minvalue, "%g"));
       CHKERRQ(PetscViewerXMLPutDouble(viewer, "maxvalue", NULL, maxvalue, "%g"));
       CHKERRQ(PetscViewerXMLPutInt(   viewer, "minloc"  , NULL, minLoc));
       CHKERRQ(PetscViewerXMLPutInt(   viewer, "maxloc"  , NULL, maxLoc));
     } else {
       CHKERRQ(PetscViewerXMLPutDouble(viewer, "value", NULL, tot/size, "%g"));
     }
     CHKERRQ(PetscViewerXMLEndSection(viewer, name));
  }
  PetscFunctionReturn(0);
}

#define N_COMM 8
static PetscErrorCode PetscLogNestedTreePrintLine(PetscViewer viewer,PetscEventPerfInfo perfInfo,PetscLogDouble countsPerCall,int parentCount,int depth,const char *name,PetscLogDouble totalTime,PetscBool *isPrinted)
{
  PetscLogDouble time = perfInfo.time;
  PetscLogDouble timeMx;
  MPI_Comm       comm;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetComm((PetscObject)viewer,&comm));
  CHKERRMPI(MPIU_Allreduce(&time, &timeMx, 1, MPIU_PETSCLOGDOUBLE, MPI_MAX, comm));
  *isPrinted = ((timeMx/totalTime) >= THRESHOLD) ? PETSC_TRUE : PETSC_FALSE;
  if (*isPrinted) {
    CHKERRQ(PetscViewerXMLStartSection(viewer, "event", NULL));
    CHKERRQ(PetscViewerXMLPutString(viewer, "name", NULL, name));
    CHKERRQ(PetscPrintXMLNestedLinePerfResults(viewer, "time", time/totalTime*100.0, 0, 0, 1.02));
    CHKERRQ(PetscPrintXMLNestedLinePerfResults(viewer, "ncalls", parentCount>0 ? countsPerCall : 0, 0.99, 1.01, 1.02));
    CHKERRQ(PetscPrintXMLNestedLinePerfResults(viewer, "mflops", time>=timeMx*0.001 ? 1e-6*perfInfo.flops/time : 0, 0, 0.01, 1.05));
    CHKERRQ(PetscPrintXMLNestedLinePerfResults(viewer, "mbps",time>=timeMx*0.001 ? perfInfo.messageLength/(1024*1024*time) : 0, 0, 0.01, 1.05));
    CHKERRQ(PetscPrintXMLNestedLinePerfResults(viewer, "nreductsps", time>=timeMx*0.001 ? perfInfo.numReductions/time : 0, 0, 0.01, 1.05));
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
      /* *****  Internal error: cannot find parent */
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

/*
 * Find the number of child events.
 */
static PetscErrorCode PetscLogNestedTreeGetChildrenCount(const PetscNestedEventTree *tree,int nTimers,int iStart,int depth,int *nChildren)
{
  int n=0;

  PetscFunctionBegin;
  for (int i=iStart+1; i<nTimers; i++) {
    if (tree[i].depth <= depth) break;
    if (tree[i].depth == depth + 1) n++;
  }
  *nChildren = n;
  PetscFunctionReturn(0);
}

/*
 * Initialize child event sort items with ID and times.
 */
static PetscErrorCode PetscLogNestedTreeSetChildrenSortItems(const PetscViewer viewer,const PetscNestedEventTree *tree,int nTimers,int iStart,int depth,int nChildren,PetscSortItem **children)
{
  MPI_Comm        comm;
  PetscLogDouble  *times, *maxTimes;
  PetscStageLog   stageLog;
  PetscEventPerfInfo *eventPerfInfo;
  const int          stage = MAINSTAGE;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetComm((PetscObject)viewer,&comm));
  CHKERRQ(PetscLogGetStageLog(&stageLog));
  eventPerfInfo = stageLog->stageInfo[stage].eventLog->eventInfo;

  if (nChildren>0) {
    /* Create an array for the id-s and maxTimes of the children,
     *  leaving 2 spaces for self-time and other-time */

    CHKERRQ(PetscMalloc1(nChildren+2,children));
    nChildren = 0;
    for (int i=iStart+1; i<nTimers; i++) {
      if (tree[i].depth<=depth) break;
      if (tree[i].depth == depth + 1) {
        (*children)[nChildren].id  = i;
        (*children)[nChildren].val = eventPerfInfo[tree[i].dftEvent].time ;
        nChildren++;
      }
    }

    /* Calculate the children's maximum times, to see whether children will be ignored or printed */
    CHKERRQ(PetscMalloc1(nChildren,&times));
    for (int i=0; i<nChildren; i++) { times[i] = (*children)[i].val; }

    CHKERRQ(PetscMalloc1(nChildren,&maxTimes));
    CHKERRMPI(MPIU_Allreduce(times, maxTimes, nChildren, MPIU_PETSCLOGDOUBLE, MPI_MAX, comm));
    CHKERRQ(PetscFree(times));

    for (int i=0; i<nChildren; i++) { (*children)[i].val = maxTimes[i]; }
    CHKERRQ(PetscFree(maxTimes));
  }
  PetscFunctionReturn(0);
}

/*
 * Set 'self' and 'other' performance info.
 */
static PetscErrorCode PetscLogNestedTreeSetSelfOtherPerfInfo(const PetscNestedEventTree *tree,int iStart,PetscLogDouble totalTime,const PetscSortItem *children,int nChildren,
                                                             PetscEventPerfInfo *myPerfInfo,PetscEventPerfInfo *selfPerfInfo,PetscEventPerfInfo *otherPerfInfo,int *parentCount,PetscLogDouble *countsPerCall)
{
  const int          stage = MAINSTAGE;
  PetscStageLog      stageLog;
  PetscEventPerfInfo *eventPerfInfo;

  PetscFunctionBegin;
  CHKERRQ(PetscLogGetStageLog(&stageLog));
  eventPerfInfo = stageLog->stageInfo[stage].eventLog->eventInfo;
  if (!tree[iStart].own) {
  /* Set values for a timer that was not activated in this process
   * (but was, in other processes of this run) */
    CHKERRQ(PetscMemzero(myPerfInfo,sizeof(*myPerfInfo)));

    *selfPerfInfo  = *myPerfInfo;
    *otherPerfInfo = *myPerfInfo;

    *parentCount   = 1;
    *countsPerCall = 0;
  } else {
  /* Set the values for a timer that was activated in this process */
    PetscLogEvent dftEvent   = tree[iStart].dftEvent;

    *parentCount    = countParents(tree,eventPerfInfo,iStart);
    *myPerfInfo     = eventPerfInfo[dftEvent];
    *countsPerCall  = (PetscLogDouble) myPerfInfo->count / (PetscLogDouble) *parentCount;

    *selfPerfInfo                = *myPerfInfo;
    otherPerfInfo->time          = 0;
    otherPerfInfo->flops         = 0;
    otherPerfInfo->numMessages   = 0;
    otherPerfInfo->messageLength = 0;
    otherPerfInfo->numReductions = 0;

    for (int i=0; i<nChildren; i++) {
      /* For all child counters: subtract the child values from self-timers */

      PetscLogEvent      dftChild      = tree[children[i].id].dftEvent;
      PetscEventPerfInfo childPerfInfo = eventPerfInfo[dftChild];

      selfPerfInfo->time          -= childPerfInfo.time;
      selfPerfInfo->flops         -= childPerfInfo.flops;
      selfPerfInfo->numMessages   -= childPerfInfo.numMessages;
      selfPerfInfo->messageLength -= childPerfInfo.messageLength;
      selfPerfInfo->numReductions -= childPerfInfo.numReductions;

      if ((children[i].val/totalTime) < THRESHOLD) {
        /* Add them to 'other' if the time is ignored in the output */
        otherPerfInfo->time          += childPerfInfo.time;
        otherPerfInfo->flops         += childPerfInfo.flops;
        otherPerfInfo->numMessages   += childPerfInfo.numMessages;
        otherPerfInfo->messageLength += childPerfInfo.messageLength;
        otherPerfInfo->numReductions += childPerfInfo.numReductions;
      }
    }
  }
  PetscFunctionReturn(0);
}

/*
 * Set max times across ranks for 'self' and 'other'.
 */
static PetscErrorCode PetscLogNestedTreeSetMaxTimes(MPI_Comm comm,int nChildren,const PetscEventPerfInfo selfPerfInfo,const PetscEventPerfInfo otherPerfInfo,PetscSortItem *children)
{
  PetscLogDouble times[2], maxTimes[2];

  PetscFunctionBegin;
  times[0] = selfPerfInfo.time;
  times[1] = otherPerfInfo.time;

  CHKERRMPI(MPIU_Allreduce(times,maxTimes,2,MPIU_PETSCLOGDOUBLE,MPI_MAX,comm));
  children[nChildren+0].id = -1;
  children[nChildren+0].val = maxTimes[0];
  children[nChildren+1].id = -2;
  children[nChildren+1].val = maxTimes[1];
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscLogNestedTreePrint(PetscViewer viewer, PetscNestedEventTree *tree, int nTimers, int iStart, PetscLogDouble totalTime)
{
  int                depth = tree[iStart].depth;
  const char         *name;
  int                parentCount=1, nChildren;
  PetscSortItem      *children;
  PetscStageLog      stageLog;
  PetscEventRegInfo  *eventRegInfo;
  PetscEventPerfInfo myPerfInfo={0},selfPerfInfo={0},otherPerfInfo={0};
  PetscLogDouble     countsPerCall=0;
  PetscBool          wasPrinted;
  PetscBool          childWasPrinted;
  MPI_Comm           comm;

  PetscFunctionBegin;
  /* Look up the name of the event and its PerfInfo */
  CHKERRQ(PetscLogGetStageLog(&stageLog));
  eventRegInfo  = stageLog->eventLog->eventInfo;
  name = eventRegInfo[(PetscLogEvent)tree[iStart].nstEvent].name;
  CHKERRQ(PetscObjectGetComm((PetscObject)viewer,&comm));

  CHKERRQ(PetscLogNestedTreeGetChildrenCount(tree,nTimers,iStart,depth,&nChildren));
  CHKERRQ(PetscLogNestedTreeSetChildrenSortItems(viewer,tree,nTimers,iStart,depth,nChildren,&children));
  CHKERRQ(PetscLogNestedTreeSetSelfOtherPerfInfo(tree,iStart,totalTime,children,nChildren,&myPerfInfo,&selfPerfInfo,&otherPerfInfo,&parentCount,&countsPerCall));

  /* Main output for this timer */
  CHKERRQ(PetscLogNestedTreePrintLine(viewer, myPerfInfo, countsPerCall, parentCount, depth, name, totalTime, &wasPrinted));

  /* Now print the lines for the children */
  if (nChildren > 0) {
    int            i;

    /* Calculate max-times for 'self' and 'other' */
    CHKERRQ(PetscLogNestedTreeSetMaxTimes(comm,nChildren,selfPerfInfo,otherPerfInfo,children));

    /* Now sort the children (including 'self' and 'other') on total time */
    qsort(children, nChildren+2, sizeof(PetscSortItem), compareSortItems);

    /* Print (or ignore) the children in ascending order of total time */
    CHKERRQ(PetscViewerXMLStartSection(viewer,"events", NULL));
    for (i=0; i<nChildren+2; i++) {
      if ((children[i].val/totalTime) < THRESHOLD) {
        /* ignored: no output */
      } else if (children[i].id==-1) {
        CHKERRQ(PetscLogNestedTreePrintLine(viewer, selfPerfInfo, 1, parentCount, depth+1, "self", totalTime, &childWasPrinted));
        if (childWasPrinted) {
          CHKERRQ(PetscViewerXMLEndSection(viewer,"event"));
        }
      } else if (children[i].id==-2) {
        size_t  len;
        char    *otherName;

        CHKERRQ(PetscStrlen(name,&len));
        CHKERRQ(PetscMalloc1(len+16,&otherName));
        CHKERRQ(PetscSNPrintf(otherName,len+16,"%s: other-timed",name));
        CHKERRQ(PetscLogNestedTreePrintLine(viewer, otherPerfInfo, 1, 1, depth+1, otherName, totalTime, &childWasPrinted));
        CHKERRQ(PetscFree(otherName));
        if (childWasPrinted) {
          CHKERRQ(PetscViewerXMLEndSection(viewer,"event"));
        }
      } else {
        /* Print the child with a recursive call to this function */
        CHKERRQ(PetscLogNestedTreePrint(viewer, tree, nTimers, children[i].id, totalTime));
      }
    }
    CHKERRQ(PetscViewerXMLEndSection(viewer,"events"));
    CHKERRQ(PetscFree(children));
  }

  if (wasPrinted) {
    CHKERRQ(PetscViewerXMLEndSection(viewer, "event"));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscLogNestedTreePrintTop(PetscViewer viewer, PetscNestedEventTree *tree, int nTimers, PetscLogDouble totalTime)
{
  int                i, nChildren;
  PetscSortItem      *children;
  MPI_Comm           comm;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetComm((PetscObject)viewer,&comm));

  CHKERRQ(PetscLogNestedTreeGetChildrenCount(tree,nTimers,-1,0,&nChildren));
  CHKERRQ(PetscLogNestedTreeSetChildrenSortItems(viewer,tree,nTimers,-1,0,nChildren,&children));

  if (nChildren>0) {
    /* Now sort the children on total time */
    qsort(children, nChildren, sizeof(PetscSortItem), compareSortItems);
    /* Print (or ignore) the children in ascending order of total time */
    CHKERRQ(PetscViewerXMLStartSection(viewer, "timertree", "Timings tree"));
    CHKERRQ(PetscViewerXMLPutDouble(viewer, "totaltime", NULL, totalTime, "%f"));
    CHKERRQ(PetscViewerXMLPutDouble(viewer, "timethreshold", NULL, thresholdTime, "%f"));

    for (i=0; i<nChildren; i++) {
      if ((children[i].val/totalTime) < THRESHOLD) {
        /* ignored: no output */
      } else {
        /* Print the child with a recursive call to this function */
        CHKERRQ(PetscLogNestedTreePrint(viewer, tree, nTimers, children[i].id, totalTime));
      }
    }
    CHKERRQ(PetscViewerXMLEndSection(viewer, "timertree"));
    CHKERRQ(PetscFree(children));
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

static PetscErrorCode PetscCalcSelfTime(PetscViewer viewer, PetscSelfTimer **p_self, int *p_nstMax)
{
  const int          stage = MAINSTAGE;
  PetscStageLog      stageLog;
  PetscEventRegInfo  *eventRegInfo;
  PetscEventPerfInfo *eventPerfInfo;
  PetscSelfTimer     *selftimes;
  PetscSelfTimer     *totaltimes;
  NestedEventId      *nstEvents;
  int                i, j, maxDefaultTimer;
  NestedEventId      nst;
  PetscLogEvent      dft;
  int                nstMax, nstMax_local;
  MPI_Comm           comm;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetComm((PetscObject)viewer,&comm));
  CHKERRQ(PetscLogGetStageLog(&stageLog));
  eventRegInfo  = stageLog->eventLog->eventInfo;
  eventPerfInfo = stageLog->stageInfo[stage].eventLog->eventInfo;

  /* For each default timer, calculate the (one) nested timer that it corresponds to. */
  maxDefaultTimer =0;
  for (i=0; i<nNestedEvents; i++) {
    int           nParents   = nestedEvents[i].nParents;
    PetscLogEvent *dftEvents = nestedEvents[i].dftEvents;
    for (j=0; j<nParents; j++) maxDefaultTimer = PetscMax(dftEvents[j],maxDefaultTimer);
  }
  CHKERRQ(PetscMalloc1(maxDefaultTimer+1,&nstEvents));
  for (dft=0; dft<maxDefaultTimer; dft++) {nstEvents[dft] = 0;}
  for (i=0; i<nNestedEvents; i++) {
    int           nParents   = nestedEvents[i].nParents;
    NestedEventId nstEvent   = nestedEvents[i].nstEvent;
    PetscLogEvent *dftEvents = nestedEvents[i].dftEvents;
    for (j=0; j<nParents; j++) nstEvents[dftEvents[j]] = nstEvent;
  }

  /* Calculate largest nested event-ID */
  nstMax_local = 0;
  for (i=0; i<nNestedEvents; i++) nstMax_local = PetscMax(nestedEvents[i].nstEvent,nstMax_local);
  CHKERRMPI(MPIU_Allreduce(&nstMax_local, &nstMax, 1, MPI_INT, MPI_MAX, comm));

  /* Initialize all total-times with zero */
  CHKERRQ(PetscMalloc1(nstMax+1,&selftimes));
  CHKERRQ(PetscMalloc1(nstMax+1,&totaltimes));
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
    for (j=0; j<nParents; j++) {
      const PetscLogEvent dftEvent = dftEvents[j];
      totaltimes[nstEvent].time          += eventPerfInfo[dftEvent].time;
      totaltimes[nstEvent].flops         += eventPerfInfo[dftEvent].flops;
      totaltimes[nstEvent].numMessages   += eventPerfInfo[dftEvent].numMessages;
      totaltimes[nstEvent].messageLength += eventPerfInfo[dftEvent].messageLength;
      totaltimes[nstEvent].numReductions += eventPerfInfo[dftEvent].numReductions;
    }
    totaltimes[nstEvent].name = eventRegInfo[(PetscLogEvent)nstEvent].name;
  }

  /* Initialize: self-times := totaltimes */
  for (nst=0; nst<=nstMax; nst++) { selftimes[nst] = totaltimes[nst]; }

  /* Subtract timed subprocesses from self-times */
  for (i=0; i<nNestedEvents; i++) {
    const int           nParents          = nestedEvents[i].nParents;
    const PetscLogEvent *dftEvents        = nestedEvents[i].dftEvents;
    const NestedEventId *dftParentsSorted = nestedEvents[i].dftParentsSorted;
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

  CHKERRQ(PetscFree(nstEvents));
  CHKERRQ(PetscFree(totaltimes));

  /* Set outputs */
  *p_self  = selftimes;
  *p_nstMax = nstMax;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscPrintSelfTime(PetscViewer viewer, const PetscSelfTimer *selftimes, int nstMax, PetscLogDouble totalTime)
{
  int                i;
  NestedEventId      nst;
  PetscSortItem      *sortSelfTimes;
  PetscLogDouble     *times, *maxTimes;
  PetscStageLog      stageLog;
  PetscEventRegInfo  *eventRegInfo;
  const int          dum_depth = 1, dum_count=1, dum_parentcount=1;
  PetscBool          wasPrinted;
  MPI_Comm           comm;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetComm((PetscObject)viewer,&comm));
  CHKERRQ(PetscLogGetStageLog(&stageLog));
  eventRegInfo  = stageLog->eventLog->eventInfo;

  CHKERRQ(PetscMalloc1(nstMax+1,&times));
  CHKERRQ(PetscMalloc1(nstMax+1,&maxTimes));
  for (nst=0; nst<=nstMax; nst++) { times[nst] = selftimes[nst].time;}
  CHKERRMPI(MPIU_Allreduce(times, maxTimes, nstMax+1, MPIU_PETSCLOGDOUBLE, MPI_MAX, comm));
  CHKERRQ(PetscFree(times));

  CHKERRQ(PetscMalloc1(nstMax+1,&sortSelfTimes));

  /* Sort the self-timers on basis of the largest time needed */
  for (nst=0; nst<=nstMax; nst++) {
    sortSelfTimes[nst].id  = nst;
    sortSelfTimes[nst].val = maxTimes[nst];
  }
  CHKERRQ(PetscFree(maxTimes));
  qsort(sortSelfTimes, nstMax+1, sizeof(PetscSortItem), compareSortItems);

  CHKERRQ(PetscViewerXMLStartSection(viewer, "selftimertable", "Self-timings"));
  CHKERRQ(PetscViewerXMLPutDouble(viewer, "totaltime", NULL, totalTime, "%f"));

  for (i=0; i<=nstMax; i++) {
    if ((sortSelfTimes[i].val/totalTime) >= THRESHOLD) {
      NestedEventId      nstEvent = sortSelfTimes[i].id;
      const char         *name    = eventRegInfo[(PetscLogEvent)nstEvent].name;
      PetscEventPerfInfo selfPerfInfo;

      selfPerfInfo.time          = selftimes[nstEvent].time ;
      selfPerfInfo.flops         = selftimes[nstEvent].flops;
      selfPerfInfo.numMessages   = selftimes[nstEvent].numMessages;
      selfPerfInfo.messageLength = selftimes[nstEvent].messageLength;
      selfPerfInfo.numReductions = selftimes[nstEvent].numReductions;

      CHKERRQ(PetscLogNestedTreePrintLine(viewer, selfPerfInfo, dum_count, dum_parentcount, dum_depth, name, totalTime, &wasPrinted));
      if (wasPrinted) {
        CHKERRQ(PetscViewerXMLEndSection(viewer, "event"));
      }
    }
  }
  CHKERRQ(PetscViewerXMLEndSection(viewer, "selftimertable"));
  CHKERRQ(PetscFree(sortSelfTimes));
  PetscFunctionReturn(0);
}

PetscErrorCode PetscLogView_Nested(PetscViewer viewer)
{
  PetscLogDouble       locTotalTime, globTotalTime;
  PetscNestedEventTree *tree = NULL;
  PetscSelfTimer       *selftimers = NULL;
  int                  nTimers = 0, nstMax = 0;
  MPI_Comm             comm;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetComm((PetscObject)viewer,&comm));
  CHKERRQ(PetscViewerInitASCII_XML(viewer));
  CHKERRQ(PetscViewerASCIIPrintf(viewer, "<!-- PETSc Performance Summary: -->\n"));
  CHKERRQ(PetscViewerXMLStartSection(viewer, "petscroot", NULL));

  /* Get the total elapsed time, local and global maximum */
  CHKERRQ(PetscTime(&locTotalTime));  locTotalTime -= petsc_BaseTime;
  CHKERRMPI(MPIU_Allreduce(&locTotalTime, &globTotalTime, 1, MPIU_PETSCLOGDOUBLE, MPI_MAX, comm));

  /* Print global information about this run */
  CHKERRQ(PetscPrintExeSpecs(viewer));
  CHKERRQ(PetscPrintGlobalPerformance(viewer, locTotalTime));
  /* Collect nested timer tree info from all processes */
  CHKERRQ(PetscLogNestedTreeCreate(viewer, &tree, &nTimers));
  CHKERRQ(PetscLogNestedTreePrintTop(viewer, tree, nTimers, globTotalTime));
  CHKERRQ(PetscLogNestedTreeDestroy(tree, nTimers));

  /* Calculate self-time for all (not-nested) events */
  CHKERRQ(PetscCalcSelfTime(viewer, &selftimers, &nstMax));
  CHKERRQ(PetscPrintSelfTime(viewer, selftimers, nstMax, globTotalTime));
  CHKERRQ(PetscFree(selftimers));

  CHKERRQ(PetscViewerXMLEndSection(viewer, "petscroot"));
  CHKERRQ(PetscViewerFinalASCII_XML(viewer));
  PetscFunctionReturn(0);
}

/*
 * Get the name of a nested event.
 */
static PetscErrorCode PetscGetNestedEventName(const PetscNestedEventTree *tree,int id,char **name)
{
  PetscStageLog   stageLog;

  PetscFunctionBegin;
  CHKERRQ(PetscLogGetStageLog(&stageLog));
  *name = stageLog->eventLog->eventInfo[(PetscLogEvent)tree[id].nstEvent].name;
  PetscFunctionReturn(0);
}

/*
 * Get the total time elapsed.
 */
static PetscErrorCode PetscGetTotalTime(const PetscViewer viewer,PetscLogDouble *totalTime)
{
  PetscLogDouble  locTotalTime;
  MPI_Comm        comm;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetComm((PetscObject)viewer,&comm));
  CHKERRQ(PetscTime(&locTotalTime));
  locTotalTime -= petsc_BaseTime;
  CHKERRMPI(MPIU_Allreduce(&locTotalTime,totalTime,1,MPIU_PETSCLOGDOUBLE,MPI_MAX,comm));
  PetscFunctionReturn(0);
}

/*
 * Write a line to the flame graph output and then recurse into child events.
 */
static PetscErrorCode PetscLogNestedTreePrintFlamegraph(PetscViewer viewer,PetscNestedEventTree *tree,int nTimers,int iStart,PetscLogDouble totalTime,PetscIntStack eventStack)
{
  int                 depth=tree[iStart].depth,parentCount=1,i,nChildren;
  char                *name=NULL;
  PetscEventPerfInfo  myPerfInfo={0},selfPerfInfo={0},otherPerfInfo={0};
  PetscLogDouble      countsPerCall=0,locTime,globTime;
  PetscSortItem       *children;
  PetscStageLog       stageLog;
  MPI_Comm            comm;

  PetscFunctionBegin;
  CHKERRQ(PetscLogGetStageLog(&stageLog));
  CHKERRQ(PetscObjectGetComm((PetscObject)viewer,&comm));

  /* Determine information about the child events as well as 'self' and 'other' */
  CHKERRQ(PetscLogNestedTreeGetChildrenCount(tree,nTimers,iStart,depth,&nChildren));
  CHKERRQ(PetscLogNestedTreeSetChildrenSortItems(viewer,tree,nTimers,iStart,depth,nChildren,&children));
  CHKERRQ(PetscLogNestedTreeSetSelfOtherPerfInfo(tree,iStart,totalTime,children,nChildren,&myPerfInfo,&selfPerfInfo,&otherPerfInfo,&parentCount,&countsPerCall));

  /* Write line to the file. The time shown is 'self' + 'other' because each entry in the output
   * is the total time spent in the event minus the amount spent in child events. */
  locTime = selfPerfInfo.time + otherPerfInfo.time;
  CHKERRMPI(MPIU_Allreduce(&locTime,&globTime,1,MPIU_PETSCLOGDOUBLE,MPI_MAX,comm));
  if (globTime/totalTime > THRESHOLD && tree[iStart].own) {
    /* Iterate over parent events in the stack and write them */
    for (i=0; i<=eventStack->top; i++) {
      CHKERRQ(PetscGetNestedEventName(tree,eventStack->stack[i],&name));
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"%s;",name));
    }
    CHKERRQ(PetscGetNestedEventName(tree,iStart,&name));
    /* The output is given as an integer in microseconds because otherwise the file cannot be read
     * by apps such as speedscope (https://speedscope.app/). */
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"%s %" PetscInt64_FMT "\n",name,(PetscInt64)(globTime*1e6)));
  }

  /* Add the current event to the parent stack and write the child events */
  PetscIntStackPush(eventStack, iStart);
  for (i=0; i<nChildren; i++) {
    CHKERRQ(PetscLogNestedTreePrintFlamegraph(viewer,tree,nTimers,children[i].id,totalTime,eventStack));
  }
  /* Pop the top item from the stack and immediately discard it */
  {
    int tmp;
    PetscIntStackPop(eventStack, &tmp);
  }
  PetscFunctionReturn(0);
}

/*
 * Print nested logging information to a file suitable for reading into a Flame Graph.
 *
 * The format consists of a semicolon-separated list of events and the event duration in microseconds (which must be an integer).
 * An example output would look like:
 *   MatAssemblyBegin 1
 *   MatAssemblyEnd 10
 *   MatView 302
 *   KSPSetUp 98
 *   KSPSetUp;VecSet 5
 *   KSPSolve 150
 *
 * This option may be requested from the command line by passing in the flag `-log_view :<somefile>.txt:ascii_flamegraph`.
 */
PetscErrorCode PetscLogView_Flamegraph(PetscViewer viewer)
{
  int                   nTimers=0,i,nChildren;
  PetscIntStack         eventStack;
  PetscLogDouble        totalTime;
  PetscNestedEventTree  *tree=NULL;
  PetscSortItem         *children;

  PetscFunctionBegin;
  CHKERRQ(PetscGetTotalTime(viewer,&totalTime));
  CHKERRQ(PetscLogNestedTreeCreate(viewer, &tree, &nTimers));
  /* We use an integer stack to keep track of parent event IDs */
  CHKERRQ(PetscIntStackCreate(&eventStack));

  /* Initialize the child events and write them recursively */
  CHKERRQ(PetscLogNestedTreeGetChildrenCount(tree,nTimers,-1,0,&nChildren));
  CHKERRQ(PetscLogNestedTreeSetChildrenSortItems(viewer,tree,nTimers,-1,0,nChildren,&children));
  for (i=0; i<nChildren; i++) {
    CHKERRQ(PetscLogNestedTreePrintFlamegraph(viewer,tree,nTimers,children[i].id,totalTime,eventStack));
  }

  CHKERRQ(PetscLogNestedTreeDestroy(tree, nTimers));
  CHKERRQ(PetscIntStackDestroy(eventStack));
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode PetscASend(int count, int datatype)
{
#if !defined(MPIUNI_H) && !defined(PETSC_HAVE_BROKEN_RECURSIVE_MACRO) && !defined(PETSC_HAVE_MPI_MISSING_TYPESIZE)
#endif

  PetscFunctionBegin;
  petsc_send_ct++;
#if !defined(MPIUNI_H) && !defined(PETSC_HAVE_BROKEN_RECURSIVE_MACRO) && !defined(PETSC_HAVE_MPI_MISSING_TYPESIZE)
  CHKERRQ(PetscMPITypeSize(count,MPI_Type_f2c((MPI_Fint) datatype),&petsc_send_len));
#endif
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode PetscARecv(int count, int datatype)
{
#if !defined(MPIUNI_H) && !defined(PETSC_HAVE_BROKEN_RECURSIVE_MACRO) && !defined(PETSC_HAVE_MPI_MISSING_TYPESIZE)
#endif

  PetscFunctionBegin;
  petsc_recv_ct++;
#if !defined(MPIUNI_H) && !defined(PETSC_HAVE_BROKEN_RECURSIVE_MACRO) && !defined(PETSC_HAVE_MPI_MISSING_TYPESIZE)
  CHKERRQ(PetscMPITypeSize(count,MPI_Type_f2c((MPI_Fint) datatype),&petsc_recv_len));
#endif
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode PetscAReduce()
{
  PetscFunctionBegin;
  petsc_allreduce_ct++;
  PetscFunctionReturn(0);
}

#endif
