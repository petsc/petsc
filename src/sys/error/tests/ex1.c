
static char help[] = "Tests signal handling.\n\n";

#include <petscsys.h>
#include <signal.h>

typedef struct _handlerCtx {
  int exitHandler;
  int signum;
} HandlerCtx;

int handleSignal(int signum, void *ctx)
{
  HandlerCtx *user = (HandlerCtx*) ctx;

  user->signum = signum;
  if (signum == SIGHUP) user->exitHandler = 1;
  return 0;
}

int main(int argc, char *args[])
{
  HandlerCtx     user;
  PetscErrorCode ierr;

  user.exitHandler = 0;

  ierr = PetscInitialize(&argc, &args, (char*) 0, help);if (ierr) return ierr;
  ierr = PetscPushSignalHandler(handleSignal, &user);CHKERRQ(ierr);
  while (!user.exitHandler) {
    if (user.signum > 0) {
      ierr        = PetscPrintf(PETSC_COMM_SELF, "Caught signal %d\n", user.signum);CHKERRQ(ierr);
      user.signum = -1;
    }
  }
  ierr = PetscPopSignalHandler();CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   build:
     requires: !define(PETSC_MISSING_SIGHUP)

   test:
     TODO: need to send a signal to the process to kill it from the test harness

TEST*/
