#include <petsc/private/loghandlerimpl.h>

typedef struct _n_PetscLogHandler_Legacy *PetscLogHandler_Legacy;
struct _n_PetscLogHandler_Legacy {
  PetscErrorCode (*PetscLogPLB)(PetscLogEvent, int, PetscObject, PetscObject, PetscObject, PetscObject);
  PetscErrorCode (*PetscLogPLE)(PetscLogEvent, int, PetscObject, PetscObject, PetscObject, PetscObject);
  PetscErrorCode (*PetscLogPHC)(PetscObject);
  PetscErrorCode (*PetscLogPHD)(PetscObject);
};

static PetscErrorCode PetscLogHandlerEventBegin_Legacy(PetscLogHandler handler, PetscLogEvent e, PetscObject o1, PetscObject o2, PetscObject o3, PetscObject o4)
{
  PetscLogHandler_Legacy legacy = (PetscLogHandler_Legacy)handler->data;

  return (*legacy->PetscLogPLB)(e, 0, o1, o2, o3, o4);
}

static PetscErrorCode PetscLogHandlerEventEnd_Legacy(PetscLogHandler handler, PetscLogEvent e, PetscObject o1, PetscObject o2, PetscObject o3, PetscObject o4)
{
  PetscLogHandler_Legacy legacy = (PetscLogHandler_Legacy)handler->data;

  return (*legacy->PetscLogPLE)(e, 0, o1, o2, o3, o4);
}

static PetscErrorCode PetscLogHandlerObjectCreate_Legacy(PetscLogHandler handler, PetscObject o)
{
  PetscLogHandler_Legacy legacy = (PetscLogHandler_Legacy)handler->data;

  return (*legacy->PetscLogPHC)(o);
}

static PetscErrorCode PetscLogHandlerObjectDestroy_Legacy(PetscLogHandler handler, PetscObject o)
{
  PetscLogHandler_Legacy legacy = (PetscLogHandler_Legacy)handler->data;

  return (*legacy->PetscLogPHD)(o);
}

static PetscErrorCode PetscLogHandlerDestroy_Legacy(PetscLogHandler handler)
{
  PetscFunctionBegin;
  PetscCall(PetscFree(handler->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
  PETSCLOGHANDLERLEGACY - PETSCLOGHANDLERLEGACY = "legacy" -  A
  `PetscLogHandler` that can be constructed from the callbacks used in
  `PetscLogSet()`.  A log handler of this type is created and started by
  `PetscLogLegacyCallbacksBegin()`.

  Level: developer

.seealso: [](ch_profiling), `PetscLogHandler`, `PetscLogHandlerCreateLegacy()`
M*/

PETSC_INTERN PetscErrorCode PetscLogHandlerCreate_Legacy(PetscLogHandler handler)
{
  PetscLogHandler_Legacy legacy;

  PetscFunctionBegin;
  PetscCall(PetscNew(&legacy));
  handler->data         = (void *)legacy;
  handler->ops->destroy = PetscLogHandlerDestroy_Legacy;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscLogHandlerCreateLegacy - Create a `PetscLogHandler` from callbacks matching PETSc's legacy log handler callbacks

  Collective

  Input Parameters:
+ comm        - an MPI communicator
. PetscLogPLB - a function to call during `PetscLogHandlerEventBegin()` (or `NULL`)
. PetscLogPLE - a function to call during `PetscLogHandlerEventEnd()` (or `NULL`)
. PetscLogPHC - a function to call during `PetscLogHandlerObjectCreate()` (or `NULL`)
- PetscLogPHD - a function to call during `PetscLogHandlerObjectDestroy()` (or `NULL`)

  Output Parameter:
. handler - a `PetscLogHandler`

  Calling sequence of `PetscLogPLB`:
+ e  - a `PetscLogEvent` that is beginning
. _i - deprecated, unused
. o1 - a `PetscObject` associated with `e` (or `NULL`)
. o2 - a `PetscObject` associated with `e` (or `NULL`)
. o3 - a `PetscObject` associated with `e` (or `NULL`)
- o4 - a `PetscObject` associated with `e` (or `NULL`)

  Calling sequence of `PetscLogPLE`:
+ e  - a `PetscLogEvent` that is beginning
. _i - deprecated, unused
. o1 - a `PetscObject` associated with `e` (or `NULL`)
. o2 - a `PetscObject` associated with `e` (or `NULL`)
. o3 - a `PetscObject` associated with `e` (or `NULL`)
- o4 - a `PetscObject` associated with `e` (or `NULL`)

  Calling sequence of `PetscLogPHC`:
. o - a `PetscObject` that has just been created

  Calling sequence of `PetscLogPHD`:
. o - a `PetscObject` that is about to be destroyed

  Level: developer

  Notes:
  This is for transitioning from the deprecated function `PetscLogSet()` and should not be used in new code.

  `PetscLogLegacyCallbacksBegin()`, which calls this function, creates and starts (`PetscLogHandlerStart()`) a log handler,
  should be used in almost all cases.

.seealso: [](ch_profiling)
@*/
PetscErrorCode PetscLogHandlerCreateLegacy(MPI_Comm comm, PetscErrorCode (*PetscLogPLB)(PetscLogEvent e, int _i, PetscObject o1, PetscObject o2, PetscObject o3, PetscObject o4), PetscErrorCode (*PetscLogPLE)(PetscLogEvent e, int _i, PetscObject o1, PetscObject o2, PetscObject o3, PetscObject o4), PetscErrorCode (*PetscLogPHC)(PetscObject o), PetscErrorCode (*PetscLogPHD)(PetscObject o), PetscLogHandler *handler)
{
  PetscLogHandler_Legacy legacy;
  PetscLogHandler        h;

  PetscFunctionBegin;
  PetscCall(PetscLogHandlerCreate(comm, handler));
  h = *handler;
  PetscCall(PetscLogHandlerSetType(h, PETSCLOGHANDLERLEGACY));
  legacy = (PetscLogHandler_Legacy)h->data;

  legacy->PetscLogPLB = PetscLogPLB;
  legacy->PetscLogPLE = PetscLogPLE;
  legacy->PetscLogPHC = PetscLogPHC;
  legacy->PetscLogPHD = PetscLogPHD;

  h->ops->eventbegin    = PetscLogPLB ? PetscLogHandlerEventBegin_Legacy : NULL;
  h->ops->eventend      = PetscLogPLE ? PetscLogHandlerEventEnd_Legacy : NULL;
  h->ops->objectcreate  = PetscLogPHC ? PetscLogHandlerObjectCreate_Legacy : NULL;
  h->ops->objectdestroy = PetscLogPHD ? PetscLogHandlerObjectDestroy_Legacy : NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}
