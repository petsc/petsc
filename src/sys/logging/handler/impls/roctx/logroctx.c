#include <petsc/private/logimpl.h> /*I "petscsys.h" I*/
#include <petsc/private/loghandlerimpl.h>
#include <petscdevice_hip.h>

#if PetscDefined(HAVE_ROCTX)

  #if PETSC_PKG_HIP_VERSION_GE(6, 4, 0)
    #include <rocprofiler-sdk-roctx/roctx.h>

static PetscErrorCode PetscLogHandlerEventsPause_ROCTX(PetscLogHandler h)
{
  PetscFunctionBegin;
  /* Pause all profiling */
  PetscInt err = roctxProfilerPause(0);
  PetscCheck(err == 0, PETSC_COMM_SELF, PETSC_ERR_GPU, "Failed to pause ROCTX profiler");
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogHandlerEventsResume_ROCTX(PetscLogHandler h)
{
  PetscFunctionBegin;
  /* Resume all profiling */
  PetscInt err = roctxProfilerResume(0);
  PetscCheck(err == 0, PETSC_COMM_SELF, PETSC_ERR_GPU, "Failed to resume ROCTX profiler");
  PetscFunctionReturn(PETSC_SUCCESS);
}

  #elif PETSC_PKG_HIP_VERSION_GE(6, 0, 0)
    #include <roctracer/roctx.h>
  #elif PETSC_PKG_HIP_VERSION_GE(5, 0, 0)
    #include <roctx.h>
  #endif

#endif

#if PetscDefined(HAVE_ROCTX)

static PetscErrorCode PetscLogHandlerEventBegin_ROCTX(PetscLogHandler handler, PetscLogEvent event, PetscObject o1, PetscObject o2, PetscObject o3, PetscObject o4)
{
  PetscLogState     state;
  PetscLogEventInfo info;

  PetscFunctionBegin;
  if (PetscDeviceInitialized(PETSC_DEVICE_HIP)) {
    PetscCall(PetscLogHandlerGetState(handler, &state));
    PetscCall(PetscLogStateEventGetInfo(state, event, &info));
    (void)roctxRangePush(info.name);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogHandlerEventEnd_ROCTX(PetscLogHandler handler, PetscLogEvent event, PetscObject o1, PetscObject o2, PetscObject o3, PetscObject o4)
{
  PetscFunctionBegin;
  if (PetscDeviceInitialized(PETSC_DEVICE_HIP)) {
    (void)roctxRangePush("StreamSync0");
    /* Sync the default stream to ensure proper timing within event*/
    PetscCallHIP(hipDeviceSynchronize());
    (void)roctxRangePop();
    (void)roctxRangePop();
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

#else

static PetscErrorCode PetscLogHandlerEventBegin_ROCTX(PetscLogHandler handler, PetscLogEvent event, PetscObject o1, PetscObject o2, PetscObject o3, PetscObject o4)
{
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogHandlerEventEnd_ROCTX(PetscLogHandler handler, PetscLogEvent event, PetscObject o1, PetscObject o2, PetscObject o3, PetscObject o4)
{
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#endif

/*MC
  PETSCLOGHANDLERROCTX - PETSCLOGHANDLERROCTX = "roctx" -  A `PetscLogHandler` that creates an ROCTX range (which appears in rocprof profiling) for each PETSc event.

  Options Database Keys:
+ -log_roctx   - start an roctx log handler manually
- -log_roctx 0 - stop the roctx log handler from starting automatically in `PetscInitialize()` in a program run within a rocprof profiling session

  Level: developer

.seealso: [](ch_profiling), `PetscLogHandler`
M*/

PETSC_INTERN PetscErrorCode PetscLogHandlerCreate_ROCTX(PetscLogHandler handler)
{
  PetscFunctionBegin;
  handler->ops->eventbegin = PetscLogHandlerEventBegin_ROCTX;
  handler->ops->eventend   = PetscLogHandlerEventEnd_ROCTX;
#if PetscDefined(HAVE_ROCTX)
  #if PETSC_PKG_HIP_VERSION_GE(6, 4, 0)
  PetscCall(PetscObjectComposeFunction((PetscObject)handler, "PetscLogHandlerEventsPause_C", PetscLogHandlerEventsPause_ROCTX));
  PetscCall(PetscObjectComposeFunction((PetscObject)handler, "PetscLogHandlerEventsResume_C", PetscLogHandlerEventsResume_ROCTX));
  #endif
#endif
  PetscCall(PetscInfo(handler, "roctx log handler created\n"));
  PetscFunctionReturn(PETSC_SUCCESS);
}
