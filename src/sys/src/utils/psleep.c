#ifndef lint
static char vcid[] = "$Id: psleep.c,v 1.6 1997/03/03 17:53:50 balay Exp balay $";
#endif
/*
     Provides utility routines for manulating any type of PETSc object.
*/
#include "petsc.h"  /*I   "petsc.h"    I*/
#include "src/sys/src/files.h"

#if defined(__cplusplus)
extern "C" {
#endif
extern void sleep(int);
#if defined(__cplusplus)
}
#endif

#undef __FUNC__  
#define __FUNC__ "PetscSleep" /* ADIC Ignore */
/*@
   PetscSleep - Sleeps some number of seconds.

   Input Parameters:
.  s - number of seconds to sleep

.keywords: sleep, wait
@*/
void PetscSleep(int s)
{
  if (s < 0) getc(stdin);
#if defined (PARCH_nt)
  else       _sleep(s*1000);
#else
  else       sleep(s);
#endif
}

