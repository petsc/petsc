#include <petsc/private/logimpl.h> /*I "petscsys.h" I*/
#include <petsc/private/loghandlerimpl.h>
#include <mpe.h>

typedef struct _n_PetscEventMPE {
  int start;
  int final;
} PetscEventMPE;

PETSC_LOG_RESIZABLE_ARRAY(MPEArray, PetscEventMPE, void *, NULL, NULL, NULL);

typedef struct _n_PetscLogHandler_MPE *PetscLogHandler_MPE;

struct _n_PetscLogHandler_MPE {
  PetscLogMPEArray events;
};

static PetscErrorCode PetscLogHandlerContextCreate_MPE(PetscLogHandler_MPE *mpe_p)
{
  PetscLogHandler_MPE mpe;

  PetscFunctionBegin;
  PetscCall(PetscNew(mpe_p));
  mpe = *mpe_p;
  PetscCall(PetscLogMPEArrayCreate(128, &mpe->events));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogHandlerDestroy_MPE(PetscLogHandler h)
{
  PetscLogHandler_MPE mpe = (PetscLogHandler_MPE)h->data;

  PetscFunctionBegin;
  PetscCall(PetscLogMPEArrayDestroy(&mpe->events));
  PetscCall(PetscFree(mpe));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#define PETSC_RGB_COLORS_MAX 39
static const char *PetscLogMPERGBColors[PETSC_RGB_COLORS_MAX] = {"OliveDrab:      ", "BlueViolet:     ", "CadetBlue:      ", "CornflowerBlue: ", "DarkGoldenrod:  ", "DarkGreen:      ", "DarkKhaki:      ", "DarkOliveGreen: ",
                                                                 "DarkOrange:     ", "DarkOrchid:     ", "DarkSeaGreen:   ", "DarkSlateGray:  ", "DarkTurquoise:  ", "DeepPink:       ", "DarkKhaki:      ", "DimGray:        ",
                                                                 "DodgerBlue:     ", "GreenYellow:    ", "HotPink:        ", "IndianRed:      ", "LavenderBlush:  ", "LawnGreen:      ", "LemonChiffon:   ", "LightCoral:     ",
                                                                 "LightCyan:      ", "LightPink:      ", "LightSalmon:    ", "LightSlateGray: ", "LightYellow:    ", "LimeGreen:      ", "MediumPurple:   ", "MediumSeaGreen: ",
                                                                 "MediumSlateBlue:", "MidnightBlue:   ", "MintCream:      ", "MistyRose:      ", "NavajoWhite:    ", "NavyBlue:       ", "OliveDrab:      "};

static PetscErrorCode PetscLogMPEGetRGBColor_Internal(const char *str[])
{
  static int idx = 0;

  PetscFunctionBegin;
  *str = PetscLogMPERGBColors[idx];
  idx  = (idx + 1) % PETSC_RGB_COLORS_MAX;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogHandlerMPECreateEvent(const char name[], PetscLogMPEArray array)
{
  PetscEventMPE mpe_event;
  PetscMPIInt   rank;

  PetscFunctionBegin;
  MPE_Log_get_state_eventIDs(&mpe_event.start, &mpe_event.final);
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  if (rank == 0) {
    const char *color;

    PetscCall(PetscLogMPEGetRGBColor_Internal(&color));
    MPE_Describe_state(mpe_event.start, mpe_event.final, name, (char *)color);
  }
  PetscCall(PetscLogMPEArrayPush(array, mpe_event));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogHandlerMPEUpdate(PetscLogHandler h)
{
  PetscLogHandler_MPE mpe = (PetscLogHandler_MPE)h->data;
  PetscLogState       state;
  PetscInt            num_events, num_events_old;

  PetscFunctionBegin;
  PetscCall(PetscLogHandlerGetState(h, &state));
  PetscCall(PetscLogStateGetNumEvents(state, &num_events));
  PetscCall(PetscLogMPEArrayGetSize(mpe->events, &num_events_old, NULL));
  for (PetscInt i = num_events_old; i < num_events; i++) {
    PetscLogEventInfo event_info;

    PetscCall(PetscLogStateEventGetInfo(state, (PetscLogEvent)i, &event_info));
    PetscCall(PetscLogHandlerMPECreateEvent(event_info.name, mpe->events));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogHandlerEventBegin_MPE(PetscLogHandler handler, PetscLogEvent event, PetscObject o1, PetscObject o2, PetscObject o3, PetscObject o4)
{
  PetscLogHandler_MPE mpe = (PetscLogHandler_MPE)handler->data;
  PetscEventMPE       mpe_event;

  PetscFunctionBegin;
  PetscCall(PetscLogHandlerMPEUpdate(handler));
  PetscCall(PetscLogMPEArrayGet(mpe->events, event, &mpe_event));
  PetscCall(MPE_Log_event(mpe_event.start, 0, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogHandlerEventEnd_MPE(PetscLogHandler handler, PetscLogEvent event, PetscObject o1, PetscObject o2, PetscObject o3, PetscObject o4)
{
  PetscLogHandler_MPE mpe = (PetscLogHandler_MPE)handler->data;
  PetscEventMPE       mpe_event;

  PetscFunctionBegin;
  PetscCall(PetscLogHandlerMPEUpdate(handler));
  PetscCall(PetscLogMPEArrayGet(mpe->events, event, &mpe_event));
  PetscCall(MPE_Log_event(mpe_event.final, 0, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
  PETSCLOGHANDLERMPE - PETSCLOGHANDLERMPE = "mpe" -  A
  `PetscLogHandler` that collects data for MPE, the MPI Parallel Environment for
  performance visualization.  A log handler of this type is created and started
  by `PetscLogMPEBegin()`.

  Level: developer

.seealso: [](ch_profiling), `PetscLogHandler`
M*/

PETSC_INTERN PetscErrorCode PetscLogHandlerCreate_MPE(PetscLogHandler handler)
{
  PetscFunctionBegin;
  PetscCall(PetscLogHandlerContextCreate_MPE((PetscLogHandler_MPE *)&handler->data));
  handler->ops->destroy    = PetscLogHandlerDestroy_MPE;
  handler->ops->eventbegin = PetscLogHandlerEventBegin_MPE;
  handler->ops->eventend   = PetscLogHandlerEventEnd_MPE;
  PetscFunctionReturn(PETSC_SUCCESS);
}
