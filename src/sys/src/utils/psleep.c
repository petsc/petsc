#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: psleep.c,v 1.16 1998/05/18 19:08:05 bsmith Exp balay $";
#endif

#include "petsc.h"                 /*I   "petsc.h"    I*/
#if defined (HAVE_UNISTD_H)
#include <unistd.h>
#endif
#include "pinclude/petscfix.h"

#if defined (PARCH_nt)
#include <stdlib.h>
#endif

#undef __FUNC__  
#define __FUNC__ "PetscSleep"
/*@
   PetscSleep - Sleeps some number of seconds.

   Not Collective

   Input Parameters:
.  s - number of seconds to sleep

.keywords: sleep, wait
@*/
int PetscSleep(int s)
{
  PetscFunctionBegin;
  if (s < 0) getc(stdin);
#if defined (PARCH_nt)
  else       _sleep(s*1000);
#else
  else       sleep(s);
#endif
  PetscFunctionReturn(0);
}

