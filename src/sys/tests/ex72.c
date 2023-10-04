const char help[] = "Test dropping PetscLogEventEnd()";

#include <petsc.h>

int main(int argc, char **argv)
{
  PetscLogEvent e1, e2;
  PetscLogStage s;

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(PetscLogEventRegister("Event-1", PETSC_OBJECT_CLASSID, &e1));
  PetscCall(PetscLogEventRegister("Event-2", PETSC_OBJECT_CLASSID, &e2));
  PetscCall(PetscLogStageRegister("User Stage", &s));
  PetscCall(PetscLogStagePush(s));
  PetscCall(PetscSleep(0.1));
  PetscCall(PetscLogEventBegin(e1, NULL, NULL, NULL, NULL));
  PetscCall(PetscSleep(0.1));
  PetscCall(PetscLogEventBegin(e2, NULL, NULL, NULL, NULL));
  PetscCall(PetscSleep(0.1));
  PetscCall(PetscLogStagePop());
  PetscCall(PetscLogEventBegin(e1, NULL, NULL, NULL, NULL));
  PetscCall(PetscSleep(0.1));
  PetscCall(PetscLogEventBegin(e2, NULL, NULL, NULL, NULL));
  PetscCall(PetscSleep(0.1));
  PetscCall(PetscLogEventEnd(e1, NULL, NULL, NULL, NULL));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: 0
    requires: defined(PETSC_USE_LOG)
    args: -log_view ::ascii_flamegraph -info :loghandler
    filter: sed -E "s/ [0-9]+/ time_removed/g"

TEST*/
