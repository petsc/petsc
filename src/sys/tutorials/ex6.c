const char help[] = "How to create a log handler using legacy callbacks";

#include <petscsys.h>

/* Log handlers that use the legacy callbacks have no context pointer,
   but they can access global logging information.  If your log handler only
   needs to interact with the arguments to the callback functions and global
   data structures, the legacy callbacks can be used. */

#define PrintData(format_string, ...) \
  do { \
    PetscMPIInt    rank; \
    PetscLogDouble time; \
    PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank)); \
    PetscCall(PetscTime(&time)); \
    PetscCall(PetscPrintf(PETSC_COMM_SELF, "[%d:%g:%-22s] " format_string, rank, time, PETSC_FUNCTION_NAME, __VA_ARGS__)); \
  } while (0)

static PetscErrorCode MyEventBeginHandler(PetscLogEvent event, int _unused, PetscObject o1, PetscObject o2, PetscObject o3, PetscObject o4)
{
  const char *name;
  PetscObject objs[] = {o1, o2, o3, o4};

  PetscFunctionBegin;
  PetscCall(PetscLogEventGetName(event, &name));
  PrintData("event name: %s\n", name);
  for (int i = 0; i < 4; i++) {
    if (objs[i]) {
      const char *obj_name;
      PetscCall(PetscObjectGetName(objs[i], &obj_name));
      PrintData("  associated object name: %s\n", obj_name);
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MyEventEndHandler(PetscLogEvent event, int _unused, PetscObject o1, PetscObject o2, PetscObject o3, PetscObject o4)
{
  const char *name;

  PetscFunctionBegin;
  PetscCall(PetscLogEventGetName(event, &name));
  PrintData("event name: %s\n", name);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MyObjectCreateHandler(PetscObject o)
{
  const char *obj_class;

  PetscCall(PetscObjectGetClassName(o, &obj_class));
  PetscFunctionBegin;
  PrintData("object class: %s\n", obj_class);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MyObjectDestroyHandler(PetscObject o)
{
  const char *obj_class;
  const char *name;

  PetscCall(PetscObjectGetClassName(o, &obj_class));
  PetscCall(PetscObjectGetName(o, &name));
  PetscFunctionBegin;
  PrintData("object type: %s, name: %s\n", obj_class, name);
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  PetscLogEvent  event;
  PetscLogStage  stage;
  PetscContainer o1, o2, o3, o4;

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(PetscLogLegacyCallbacksBegin(MyEventBeginHandler, MyEventEndHandler, MyObjectCreateHandler, MyObjectDestroyHandler));
  PetscCall(PetscLogStageRegister("User stage", &stage));
  PetscCall(PetscLogEventRegister("User class", PETSC_CONTAINER_CLASSID, &event));
  PetscCall(PetscLogStagePush(stage));
  PetscCall(PetscContainerCreate(PETSC_COMM_WORLD, &o1));
  PetscCall(PetscObjectSetName((PetscObject)o1, "Container 1"));
  PetscCall(PetscContainerCreate(PETSC_COMM_WORLD, &o2));
  PetscCall(PetscObjectSetName((PetscObject)o2, "Container 2"));
  PetscCall(PetscContainerCreate(PETSC_COMM_WORLD, &o3));
  PetscCall(PetscObjectSetName((PetscObject)o3, "Container 3"));
  PetscCall(PetscContainerCreate(PETSC_COMM_WORLD, &o4));
  PetscCall(PetscObjectSetName((PetscObject)o4, "Container 4"));
  PetscCall(PetscLogEventBegin(event, o1, o2, o3, o4));
  PetscCall(PetscLogEventEnd(event, o1, o2, o3, o4));
  PetscCall(PetscContainerDestroy(&o1));
  PetscCall(PetscContainerDestroy(&o2));
  PetscCall(PetscContainerDestroy(&o3));
  PetscCall(PetscContainerDestroy(&o4));
  PetscCall(PetscLogStagePop());
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: 0
    requires: defined(PETSC_USE_LOG)
    filter: sed -E "s/:[^:]+:/:time_removed:/g"

TEST*/
