#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: psleep.c,v 1.19 1999/03/17 23:21:54 bsmith Exp bsmith $";
#endif

#include "petsc.h"                 /*I   "petsc.h"    I*/
#if defined (PETSC_HAVE_UNISTD_H)
#include <unistd.h>
#endif
#include "pinclude/petscfix.h"

#if defined (PARCH_win32)
#include <stdlib.h>
#endif

#undef __FUNC__  
#define __FUNC__ "PetscSleep"
/*@
   PetscSleep - Sleeps some number of seconds.

   Not Collective

   Input Parameters:
.  s - number of seconds to sleep

   Level: intermediate

.keywords: sleep, wait
@*/
int PetscSleep(int s)
{
  PetscFunctionBegin;
  if (s < 0) getc(stdin);
#if defined (PARCH_win32)
  else       _sleep(s*1000);
#else
  else       sleep(s);
#endif
  PetscFunctionReturn(0);
}

