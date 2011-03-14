
#include <petscsys.h>                 /*I   "petscsys.h"    I*/
#if defined (PETSC_HAVE_UNISTD_H)
#include <unistd.h>
#endif
#if defined (PETSC_HAVE_STDLIB_H)
#include <stdlib.h>
#endif
#if defined (PETSC_HAVE_DOS_H)   /* borland */
#include <dos.h>
#endif
#if defined (PETSC_HAVE_TIME_H)
#include <time.h>
#endif

#undef __FUNCT__  
#define __FUNCT__ "PetscSleep"
/*@
   PetscSleep - Sleeps some number of seconds.

   Not Collective

   Input Parameters:
.  s - number of seconds to sleep

   Notes:
      If s is negative waits for keyboard input

   Level: intermediate

   Concepts: sleeping
   Concepts: pause
   Concepts: waiting

@*/
PetscErrorCode  PetscSleep(PetscReal s)
{
  PetscFunctionBegin;
  if (s < 0) getc(stdin);

  /* Some systems consider it an error to call nanosleep or usleep for more than one second so we only use them for subsecond sleeps. */
#if defined (PETSC_HAVE_NANOSLEEP)
  else if (s < 1) {
    struct timespec rq;
    rq.tv_sec = 0;
    rq.tv_nsec = (long)(s*1e9);
    nanosleep(&rq,0);
  }
#elif defined (PETSC_HAVE_USLEEP)
  /* POSIX.1-2001 deprecates this in favor of nanosleep because nanosleep defines interaction with signals */
  else if (s < 1) usleep((unsigned int)(s*1e6));
#endif

#if defined (PETSC_HAVE_SLEEP)
  else       sleep((int)s);
#elif defined (PETSC_HAVE__SLEEP) && defined(PETSC_HAVE__SLEEP_MILISEC)
  else       _sleep((int)(s*1000));
#elif defined (PETSC_HAVE__SLEEP)
  else       _sleep((int)s);
#else
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP_SYS,"No support for sleep() on this machine");
#endif
  PetscFunctionReturn(0);
}

