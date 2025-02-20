#include <petsc/private/logimpl.h> /*I "petscsys.h" I*/
#include <petsc/private/loghandlerimpl.h>
#include <petscdevice.h>
#if PETSC_PKG_CUDA_VERSION_GE(10, 0, 0)
  #include <nvtx3/nvToolsExt.h>
#else
  #include <nvToolsExt.h>
#endif

static PetscErrorCode PetscLogHandlerEventBegin_NVTX(PetscLogHandler handler, PetscLogEvent event, PetscObject o1, PetscObject o2, PetscObject o3, PetscObject o4)
{
  PetscLogState     state;
  PetscLogEventInfo info;

  PetscFunctionBegin;
  if (PetscDeviceInitialized(PETSC_DEVICE_CUDA)) {
    PetscCall(PetscLogHandlerGetState(handler, &state));
    PetscCall(PetscLogStateEventGetInfo(state, event, &info));
    (void)nvtxRangePushA(info.name);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogHandlerEventEnd_NVTX(PetscLogHandler handler, PetscLogEvent event, PetscObject o1, PetscObject o2, PetscObject o3, PetscObject o4)
{
  PetscFunctionBegin;
  if (PetscDeviceInitialized(PETSC_DEVICE_CUDA)) (void)nvtxRangePop();
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
  PETSCLOGHANDLERNVTX - PETSCLOGHANDLERNVTX = "nvtx" -  A
  `PetscLogHandler` that creates an NVTX range (which appears in Nvidia Nsight
  profiling) for each PETSc event.

  Options Database Keys:
+ -log_nvtx   - start an nvtx log handler manually
- -log_nvtx 0 - stop the nvtx log handler from starting automatically in `PetscInitialize()` in a program run within an nsys profiling session (see Note)

  Level: developer

  Note:
  If `PetscInitialize()` detects the environment variable `NSYS_PROFILING_SESSION_ID` (which is defined by `nsys
  profile`) or `NVPROF_ID` (which is defined by `nvprof`) an instance of this log handler will automatically be
  started.

.seealso: [](ch_profiling), `PetscLogHandler`
M*/

PETSC_INTERN PetscErrorCode PetscLogHandlerCreate_NVTX(PetscLogHandler handler)
{
  PetscFunctionBegin;
  handler->ops->eventbegin = PetscLogHandlerEventBegin_NVTX;
  handler->ops->eventend   = PetscLogHandlerEventEnd_NVTX;
  PetscCall(PetscInfo(handler, "nvtx log handler created\n"));
  PetscFunctionReturn(PETSC_SUCCESS);
}
