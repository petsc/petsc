/*$Id: ex1.c,v 1.18 2003/08/08 21:30:50 knepley Exp $*/

static char help[] = "Tests signal handling.\n\n";

#include "petscsys.h"
#include <signal.h>

typedef struct _handlerCtx {
  int exitHandler;
  int signum;
} HandlerCtx;

#undef __FUNCT__
#define __FUNCT__ "handleSignal"
int handleSignal(int signum, void *ctx)
{
  HandlerCtx *user = (HandlerCtx *) ctx;

  user->signum = signum;
  if (signum == SIGHUP) {
    user->exitHandler = 1;
  }
  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char *args[])
{
  HandlerCtx     user;
  PetscErrorCode ierr;

  user.exitHandler = 0;

  ierr = PetscInitialize(&argc, &args, (char *) 0, help);CHKERRQ(ierr);
  ierr = PetscPushSignalHandler(handleSignal, &user);CHKERRQ(ierr);
  while(!user.exitHandler) {
    if (user.signum > 0) {
      ierr = PetscPrintf(PETSC_COMM_SELF, "Caught signal %d\n", user.signum);CHKERRQ(ierr);
      user.signum = -1;
    }
  }
  ierr = PetscPopSignalHandler();CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
