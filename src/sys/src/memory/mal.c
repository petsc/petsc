#ifndef lint
static char vcid[] = "$Id: mal.c,v 1.2 1995/05/23 23:09:53 bsmith Exp bsmith $";
#endif

#include "petsc.h"
#include <malloc.h>
#include "petscfix.h"

void *(*PetscMalloc)(unsigned int,int,char*) = 
                            (void*(*)(unsigned int,int,char*))malloc;
int  (*PetscFree)(void *,int,char*) = (int (*)(void*,int,char*))free;

/*@
      PetscSetMalloc - Sets the routines used to do mallocs and frees.
         This MUST be called before PetscInitialize() and may be
         called only once.

  Input Parameters:
.   malloc - the malloc routine
.   free - the free routine

@*/
int PetscSetMalloc(void *(*malloc)(unsigned int,int,char*),
                   int (*free)(void*,int,char*))
{
  static int visited = 0;
  if (visited) SETERR(1,"PetscSetMalloc: cannot call multiple times");
  PetscMalloc = malloc;
  PetscFree   = free;
  visited     = 1;
  return 0;
}

