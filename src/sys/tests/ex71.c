const char help[] = "Test getting performance info when the default log handler is not running";

#include <petscsys.h>
#include <petscviewer.h>

int main(int argc, char **argv)
{
  PetscLogEvent      event_id;
  PetscLogStage      stage_id;
  PetscEventPerfInfo stage_info;
  PetscEventPerfInfo event_info;

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(PetscLogEventRegister("My event", PETSC_VIEWER_CLASSID, &event_id));
  PetscCall(PetscLogStageRegister("My stage", &stage_id));
  PetscCall(PetscLogStagePush(stage_id));
  PetscCall(PetscLogEventBegin(event_id, NULL, NULL, NULL, NULL));
  PetscCall(PetscSleep(0.1));
  PetscCall(PetscLogEventEnd(event_id, NULL, NULL, NULL, NULL));
  PetscCall(PetscLogStagePop());
  PetscCall(PetscLogEventGetPerfInfo(stage_id, event_id, &event_info));
  PetscCall(PetscLogStageGetPerfInfo(stage_id, &stage_info));
  PetscCheck(event_info.time == 0.0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Stats should be zero");
  PetscCheck(stage_info.time == 0.0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Stats should be zero");
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: 0

TEST*/
