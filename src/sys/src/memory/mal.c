#ifndef lint
static char vcid[] = "$Id: mal.c,v 1.1 1995/05/23 21:52:04 bsmith Exp bsmith $";
#endif

#include "petsc.h"
#include <malloc.h>
#include "petscfix.h"

void *(*PetscMalloc)(int,...) = (void*(*)(int,...))malloc;
int  (*PetscFree)(void *,...) = (int (*)(void*,...))free;

/*

      PetscSetMalloc_Private - Sets the routines used to do mallocs and frees.

*/
int PetscSetMalloc_Private(void *(*malloc)(int,...),int (*free)(void*,...))
{
  static int visited = 0;
  if (visited) SETERR(1,"You cannot reset malloc");
  PetscMalloc = malloc;
  PetscFree   = free;
  visited     = 1;
  return 0;
}

