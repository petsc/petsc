static char help[] = "Tests several PetscLogHandler implementations.\n\n";

#include <petscsys.h>

/* Create a phony perfstubs implementation for testing.

   The dynamic loading in perfstubs is only enabled with the following flags,
   so we only try to export these functions if they are present */
#if defined(__linux__) && PetscDefined(HAVE_DLFCN_H)

PETSC_EXTERN void ps_tool_initialize(void)
{
  PetscFunctionBegin;
  PetscCallContinue(PetscPrintf(PETSC_COMM_SELF, "ps_tool_initialize()\n"));
  PetscFunctionReturnVoid();
}

PETSC_EXTERN void ps_tool_finalize(void)
{
  PetscFunctionBegin;
  PetscCallContinue(PetscPrintf(PETSC_COMM_SELF, "ps_tool_finalize()\n"));
  PetscFunctionReturnVoid();
}

PETSC_EXTERN void *ps_tool_timer_create(const char name[])
{
  PetscFunctionBegin;
  PetscCallContinue(PetscPrintf(PETSC_COMM_SELF, "ps_tool_timer_create(\"%s\")\n", name));
  PetscFunctionReturn((void *)name);
}

PETSC_EXTERN void *ps_tool_timer_start(void *arg)
{
  const char *name = (const char *)arg;

  PetscFunctionBegin;
  PetscCallContinue(PetscPrintf(PETSC_COMM_SELF, "ps_tool_timer_start() [%s]\n", name));
  PetscFunctionReturn(NULL);
}

PETSC_EXTERN void *ps_tool_timer_stop(void *arg)
{
  const char *name = (const char *)arg;

  PetscFunctionBegin;
  PetscCallContinue(PetscPrintf(PETSC_COMM_SELF, "ps_tool_timer_stop() [%s]\n", name));
  PetscFunctionReturn(NULL);
}
#endif

static PetscErrorCode CallEvents(PetscLogEvent event1, PetscLogEvent event2, PetscLogEvent event3)
{
  char *data;

  PetscFunctionBegin;
  PetscCall(PetscLogEventBegin(event1, NULL, NULL, NULL, NULL));
  PetscCall(PetscSleep(0.05));
  PetscCall(PetscLogEventBegin(event2, NULL, NULL, NULL, NULL));
  PetscCall(PetscSleep(0.1));
  PetscCall(PetscLogEventBegin(event3, NULL, NULL, NULL, NULL));
  PetscCall(PetscCalloc1(1048576, &data));
  PetscCall(PetscFree(data));
  PetscCall(PetscSleep(0.15));
  PetscCall(PetscLogEventEnd(event3, NULL, NULL, NULL, NULL));
  PetscCall(PetscLogEventEnd(event2, NULL, NULL, NULL, NULL));
  PetscCall(PetscLogEventEnd(event1, NULL, NULL, NULL, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  PetscLogStage  stage1, stage2, stage3 = -1;
  PetscLogEvent  event1, event2, event3;
  PetscMPIInt    rank;
  PetscContainer container1, container2;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  if (rank) {
    PetscCall(PetscLogEventRegister("Event3", 0, &event3));
    PetscCall(PetscLogEventRegister("Event2", 0, &event2));
    PetscCall(PetscLogEventRegister("Event1", PETSC_CONTAINER_CLASSID, &event1));
    PetscCall(PetscLogStageRegister("Stage2", &stage2));
    PetscCall(PetscLogStageRegister("Stage1", &stage1));
    PetscCall(PetscLogStageRegister("Stage3", &stage3));
    (void)stage3; // stage3 intentionally not used
  } else {
    PetscCall(PetscLogEventRegister("Event2", 0, &event2));
    PetscCall(PetscLogEventRegister("Event1", PETSC_CONTAINER_CLASSID, &event1));
    PetscCall(PetscLogEventRegister("Event3", 0, &event3));
    PetscCall(PetscLogStageRegister("Stage1", &stage1));
    PetscCall(PetscLogStageRegister("Stage2", &stage2));
  }

  for (PetscInt i = 0; i < 8; i++) {
    PetscCall(PetscLogEventSetDof(event3, i, (PetscLogDouble)i));
    PetscCall(PetscLogEventSetError(event3, i, (PetscLogDouble)i + 8));
  }

  PetscCall(CallEvents(event1, event2, event3));

  PetscCall(PetscLogStagePush(stage1));
  {
    PetscCall(PetscSleep(0.1));
    PetscCall(CallEvents(event1, event2, event3));
  }
  PetscCall(PetscLogStagePop());

  PetscCall(PetscContainerCreate(PETSC_COMM_WORLD, &container1));
  PetscCall(PetscContainerCreate(PETSC_COMM_WORLD, &container2));
  PetscCall(PetscObjectSetName((PetscObject)container2, "Container 2"));
  PetscCall(PetscLogObjectState((PetscObject)container1, "Setting object state for testing purposes with %d self-referential format argument", 1));

  PetscCall(PetscLogStagePush(stage2));
  {
    PetscCall(PetscSleep(0.1));
    PetscCall(CallEvents(event1, event2, event3));

    PetscCall(PetscLogStagePush(stage1));
    {
      PetscCall(PetscSleep(0.1));
      PetscCall(CallEvents(event1, event2, event3));
    }
    PetscCall(PetscLogStagePop());

    PetscCall(PetscLogEventSync(event1, PETSC_COMM_WORLD));
    PetscCall(PetscLogEventBegin(event1, container1, container2, NULL, NULL));
    {
      PetscCall(PetscSleep(0.1));
      PetscCall(PetscLogStagePush(stage1));
      {
        PetscCall(PetscSleep(0.1));
        PetscCall(CallEvents(event1, event2, event3));
      }
      PetscCall(PetscLogStagePop());
    }
    PetscCall(PetscLogEventEnd(event1, container1, container2, NULL, NULL));
  }
  PetscCall(PetscLogStagePop());

  PetscCall(PetscContainerDestroy(&container2));
  PetscCall(PetscContainerDestroy(&container1));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  # smoke test: does this program run with / without PETSC_USE_LOG?
  test:
    suffix: 0
    nsize: {{1 2}}

  # flamegraph: times of PetscSleep() are designed so the flamegraph should have reproducible entries
  test:
    suffix: 1
    nsize: {{1 2}}
    requires: defined(PETSC_USE_LOG)
    args: -log_view ::ascii_flamegraph
    filter: sed -E "s/ [0-9]+/ time_removed/g"

  test:
    suffix: 2
    requires: defined(PETSC_USE_LOG)
    nsize: 1
    args: -log_trace

  # test PetscLogDump() with action and object logging
  test:
    suffix: 3
    nsize: 1
    requires: defined(PETSC_USE_LOG)
    args: -log_include_actions -log_include_objects -log_all
    temporaries: Log.0
    filter: cat Log.0 | grep "\\(Actions accomplished\\|Objects created\\|Name\\|Info\\)"

  # -log_sync is not necessary for csv output, this is just a convenient test to add sync testing to
  test:
    suffix: 4
    nsize: 2
    requires: defined(PETSC_USE_LOG)
    args: -log_view ::ascii_csv -log_sync
    filter: grep "Event[123]" | grep -v "PCMPI"

  # we don't guarantee clog2print is available, so we just verify that our events are in the output file
  test:
    suffix: 5
    nsize: 1
    requires: defined(PETSC_USE_LOG) defined(PETSC_HAVE_MPE)
    args: -log_mpe ex30_mpe
    temporaries: ex30_mpe.clog2
    filter: strings ex30_mpe.clog2 | grep "Event[123]"

  # we don't have tau as a dependency, so we test a dummy perfstubs tool
  test:
    suffix: 6
    nsize: 1
    requires: tau_perfstubs linux dlfcn_h defined(PETSC_USE_LOG) defined(PETSC_USE_SHARED_LIBRARIES)
    args: -log_perfstubs
    filter: grep "\\(Main Stage\\|Event1\\|Event2\\|Event3\\|Stage1\\|Stage2\\)"

  test:
    suffix: 7
    nsize: 1
    requires: defined(PETSC_USE_LOG)
    args: -log_view ::ascii_info_detail -log_handler_default_use_threadsafe_events
    filter: grep "Event[123]" | grep "\\(Main Stage\\|Stage[123]\\)"

  # test the sync warning
  test:
    suffix: 8
    nsize: 2
    requires: defined(PETSC_USE_LOG)
    args: -log_view -log_sync
    filter: grep "This program was run with logging synchronization"

  # test -log_trace with an output file
  test:
    suffix: 9
    requires: defined(PETSC_USE_LOG)
    nsize: 1
    output_file: output/ex30_2.out
    args: -log_trace trace.log
    temporaries: trace.log
    filter: cat trace.log.0

  # test -log_nvtx
  test:
    suffix: 10
    requires: cuda defined(PETSC_USE_LOG)
    args: -device_enable eager -log_nvtx -info :loghandler

 TEST*/
