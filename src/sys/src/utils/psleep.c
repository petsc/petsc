/*$Id: psleep.c,v 1.30 2001/03/23 23:20:45 balay Exp $*/

#include "petscconfig.h"
#include "petsc.h"                 /*I   "petsc.h"    I*/
#if defined (HAVE_UNISTD_H)
#include <unistd.h>
#endif
#if defined (HAVE_STDLIB_H)
#include <stdlib.h>
#endif
#if defined (HAVE_DOS_H)   /* borland */
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
int PetscSleep(int s)
{
  PetscFunctionBegin;
  if (s < 0) getc(stdin);
#if defined (HAVE_SLEEP)
  else       sleep(s);
#elif defined (HAVE__SLEEP)
  else       _sleep(s*1000);
#else
  #error No sleep function located!
#endif
  PetscFunctionReturn(0);
}

