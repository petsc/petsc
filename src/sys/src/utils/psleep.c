#define PETSC_DLL

#include "petsc.h"                 /*I   "petsc.h"    I*/
#if defined (PETSC_HAVE_UNISTD_H)
#include <unistd.h>
#endif
#if defined (PETSC_HAVE_STDLIB_H)
#include <stdlib.h>
#endif
#if defined (PETSC_HAVE_DOS_H)   /* borland */
#include <dos.h>
#endif
#include "petscfix.h"

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
PetscErrorCode PETSC_DLLEXPORT PetscSleep(int s)
{
  PetscFunctionBegin;
  if (s < 0) getc(stdin);
#if defined (PETSC_HAVE_SLEEP)
  else       sleep(s);
#elif defined (PETSC_HAVE__SLEEP) && defined(PETSC_HAVE__SLEEP_MILISEC)
  else       _sleep(s*1000);
#elif defined (PETSC_HAVE__SLEEP)
  else       _sleep(s);
#else
  #error No sleep function located!
#endif
  PetscFunctionReturn(0);
}

