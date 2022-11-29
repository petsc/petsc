
static char help[] = "Tests signal handling.\n\n";

#include <petscsys.h>
#include <signal.h>

typedef struct _handlerCtx {
  int exitHandler;
  int signum;
} HandlerCtx;

int handleSignal(int signum, void *ctx)
{
  HandlerCtx *user = (HandlerCtx *)ctx;

  user->signum = signum;
  if (signum == SIGHUP) user->exitHandler = 1;
  return 0;
}

int main(int argc, char *args[])
{
  HandlerCtx user;

  user.exitHandler = 0;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));
  PetscCall(PetscPushSignalHandler(handleSignal, &user));
  while (!user.exitHandler) {
    if (user.signum > 0) {
      PetscCall(PetscPrintf(PETSC_COMM_SELF, "Caught signal %d\n", user.signum));
      user.signum = -1;
    }
  }
  PetscCall(PetscPopSignalHandler());
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   build:
     requires: !defined(PETSC_MISSING_SIGHUP)

   test:
     TODO: need to send a signal to the process to kill it from the test harness

TEST*/
