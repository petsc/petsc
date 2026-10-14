static const char help[] = "Tests vector lock locations with deep call and lock stacks.\n\n";

#include <petscvec.h>

static PetscErrorCode CheckVectorLocks(Vec x, int remaining)
{
  const char *file, *function;
  int         line;

  PetscFunctionBeginUser;
  PetscCall(VecLockReadPush(x));
  for (int i = 0; i < 2; ++i) {
    PetscCall(VecLockGetLocation(x, &file, &function, &line));
    PetscCheck(file && function, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Missing lock location");
    if (petscstack.currentsize > PETSCSTACKSIZE) {
      char *match;

      PetscCall(PetscStrstr(file, "stack depth exceeds PETSCSTACKSIZE=", &match));
      PetscCheck(match && !line, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Missing lock stack truncation diagnostic");
    } else {
      PetscBool filematch, functionmatch;

      PetscCall(PetscStrcmp(file, __FILE__, &filematch));
      PetscCall(PetscStrcmp(function, PETSC_FUNCTION_NAME, &functionmatch));
      PetscCheck(filematch && functionmatch && line > 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Incorrect lock location");
    }
    if (!i && remaining) PetscCall(CheckVectorLocks(x, remaining - 1));
  }
  PetscCall(VecLockReadPop(x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  Vec         x;
  const char *file, *function;
  int         line;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, 1, &x));
  PetscCall(CheckVectorLocks(x, 2 * PETSCSTACKSIZE));
  PetscCall(VecLockGetLocation(x, &file, &function, &line));
  PetscCheck(!file && !function && !line, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Lock stack was not restored");
  PetscCall(VecDestroy(&x));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  build:
    requires: debug !threadsafety

  test:
    args: -checkstack {{0 1}}
    output_file: output/empty.out

TEST*/
