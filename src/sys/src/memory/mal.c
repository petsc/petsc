#ifndef lint
static char vcid[] = "$Id: mal.c,v 1.7 1995/06/08 03:08:02 bsmith Exp curfman $";
#endif

#include "petsc.h"
#if defined(HAVE_STDLIB_H)
#include <stdlib.h>
#endif
#if defined(HAVE_MALLOC_H) && !defined(__cplusplus)
#include <malloc.h>
#endif
#include "petscfix.h"

void *(*PetscMalloc)(unsigned int,int,char*) = 
                            (void*(*)(unsigned int,int,char*))malloc;
int  (*PetscFree)(void *,int,char*) = (int (*)(void*,int,char*))free;

/*@
   PetscSetMalloc - Sets the routines used to do mallocs and frees.
   This routine MUST be called before PetscInitialize() and may be
   called only once.

   Input Parameters:
.  malloc - the malloc routine
.  free - the free routine

.keywords: Petsc, set, malloc, free, memory allocation
@*/
int PetscSetMalloc(void *(*imalloc)(unsigned int,int,char*),
                   int (*ifree)(void*,int,char*))
{
  static int visited = 0;
  if (visited) SETERRQ(1,"PetscSetMalloc: cannot call multiple times");
  PetscMalloc = imalloc;
  PetscFree   = ifree;
  visited     = 1;
  return 0;
}

