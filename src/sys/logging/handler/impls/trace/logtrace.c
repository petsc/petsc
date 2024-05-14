#include <petsc/private/logimpl.h> /*I "petscsys.h" I*/
#include <petsc/private/loghandlerimpl.h>

typedef struct _n_PetscLogHandler_Trace *PetscLogHandler_Trace;
struct _n_PetscLogHandler_Trace {
  FILE          *petsc_tracefile;
  size_t         petsc_tracelevel;
  char           petsc_tracespace[128];
  PetscLogDouble petsc_tracetime;
};

static PetscErrorCode PetscLogHandlerEventBegin_Trace(PetscLogHandler h, PetscLogEvent event, PetscObject o1, PetscObject o2, PetscObject o3, PetscObject o4)
{
  PetscLogHandler_Trace tr = (PetscLogHandler_Trace)h->data;
  PetscLogEventInfo     event_info;
  PetscLogDouble        cur_time;
  PetscMPIInt           rank;
  PetscLogState         state;
  PetscLogStage         stage;

  PetscFunctionBegin;
  if (!tr->petsc_tracetime) PetscCall(PetscTime(&tr->petsc_tracetime));
  tr->petsc_tracelevel++;
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)h), &rank));
  PetscCall(PetscLogHandlerGetState(h, &state));
  PetscCall(PetscLogStateGetCurrentStage(state, &stage));
  /* Log performance info */
  PetscCall(PetscTime(&cur_time));
  PetscCall(PetscLogStateEventGetInfo(state, event, &event_info));
  PetscCall(PetscFPrintf(PETSC_COMM_SELF, tr->petsc_tracefile, "%s[%d] %g Event begin: %s\n", tr->petsc_tracespace, rank, cur_time - tr->petsc_tracetime, event_info.name));
  for (size_t i = 0; i < PetscMin(sizeof(tr->petsc_tracespace), 2 * tr->petsc_tracelevel); i++) tr->petsc_tracespace[i] = ' ';
  tr->petsc_tracespace[PetscMin(127, 2 * tr->petsc_tracelevel)] = '\0';
  PetscCall(PetscFFlush(tr->petsc_tracefile));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogHandlerEventEnd_Trace(PetscLogHandler h, PetscLogEvent event, PetscObject o1, PetscObject o2, PetscObject o3, PetscObject o4)
{
  PetscLogHandler_Trace tr = (PetscLogHandler_Trace)h->data;
  PetscLogEventInfo     event_info;
  PetscLogDouble        cur_time;
  PetscLogState         state;
  PetscLogStage         stage;
  PetscMPIInt           rank;

  PetscFunctionBegin;
  tr->petsc_tracelevel--;
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)h), &rank));
  PetscCall(PetscLogHandlerGetState(h, &state));
  PetscCall(PetscLogStateGetCurrentStage(state, &stage));
  /* Log performance info */
  for (size_t i = 0; i < PetscMin(sizeof(tr->petsc_tracespace), 2 * tr->petsc_tracelevel); i++) tr->petsc_tracespace[i] = ' ';
  tr->petsc_tracespace[PetscMin(127, 2 * tr->petsc_tracelevel)] = '\0';
  PetscCall(PetscTime(&cur_time));
  PetscCall(PetscLogStateEventGetInfo(state, event, &event_info));
  PetscCall(PetscFPrintf(PETSC_COMM_SELF, tr->petsc_tracefile, "%s[%d] %g Event end: %s\n", tr->petsc_tracespace, rank, cur_time - tr->petsc_tracetime, event_info.name));
  PetscCall(PetscFFlush(tr->petsc_tracefile));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogHandlerDestroy_Trace(PetscLogHandler h)
{
  PetscFunctionBegin;
  PetscCall(PetscFree(h->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
  PETSCLOGHANDLERTRACE - PETSCLOGHANDLERTRACE = "trace" -  A
  `PetscLogHandler` that collects data for PETSc's tracing log viewer.
  A log handler of this type is created and started by `PetscLogTraceBegin()`.

  Level: developer

.seealso: [](ch_profiling), `PetscLogHandler`, `PetscLogHandlerCreateTrace()`
M*/

PETSC_INTERN PetscErrorCode PetscLogHandlerCreate_Trace(PetscLogHandler handler)
{
  PetscLogHandler_Trace tr;

  PetscFunctionBegin;
  PetscCall(PetscNew(&tr));
  handler->data            = (void *)tr;
  handler->ops->eventbegin = PetscLogHandlerEventBegin_Trace;
  handler->ops->eventend   = PetscLogHandlerEventEnd_Trace;
  handler->ops->destroy    = PetscLogHandlerDestroy_Trace;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscLogHandlerCreateTrace - Create a logger that traces events and stages to a given file descriptor

  Collective, No Fortran Support

  Input Parameters:
+ comm - an MPI communicator
- file - a file descriptor

  Output Parameters:
. handler - a `PetscLogHandler of type `PETSCLOGHANDLERTRACE`

  Level: developer

  Notes:
  Most users can just use `PetscLogTraceBegin()` to create and immediately start (`PetscLogHandlerStart()`) a tracing log handler

.seealso: [](ch_profiling), `PetscLogHandler`, `PetscLogHandlerTraceBegin()`
@*/
PetscErrorCode PetscLogHandlerCreateTrace(MPI_Comm comm, FILE *file, PetscLogHandler *handler)
{
  PetscLogHandler       h;
  PetscLogHandler_Trace tr;

  PetscFunctionBegin;
  PetscCall(PetscLogHandlerCreate(comm, handler));
  h = *handler;
  PetscCall(PetscLogHandlerSetType(h, PETSCLOGHANDLERTRACE));
  tr                  = (PetscLogHandler_Trace)h->data;
  tr->petsc_tracefile = file;
  PetscFunctionReturn(PETSC_SUCCESS);
}
