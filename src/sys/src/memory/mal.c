#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: mal.c,v 1.25 1997/07/09 20:51:14 balay Exp bsmith $";
#endif
/*
    Code that allows a user to dictate what malloc() PETSc uses.
*/
#include "petsc.h"             /*I   "petsc.h"   I*/
#if defined(HAVE_STDLIB_H)
#include <stdlib.h>
#endif
#if defined(HAVE_MALLOC_H) && !defined(__cplusplus)
#include <malloc.h>
#endif
#include "pinclude/petscfix.h"

/*
    Set the default malloc and free to be the usual system versions
*/
void *(*PetscTrMalloc)(unsigned int,int,char*,char*,char*)=(void*(*)(unsigned int,int,char*,char*,char*))malloc;
int  (*PetscTrFree)(void *,int,char*,char *,char*)        = (int (*)(void*,int,char*,char*,char*))free;

#undef __FUNC__  
#define __FUNC__ "PetscSetMalloc"
/*@C
   PetscSetMalloc - Sets the routines used to do mallocs and frees.
   This routine MUST be called before PetscInitialize() and may be
   called only once.

   Input Parameters:
.  malloc - the malloc routine
.  free - the free routine

.keywords: Petsc, set, malloc, free, memory allocation
@*/
int PetscSetMalloc(void *(*imalloc)(unsigned int,int,char*,char*,char*),
                   int (*ifree)(void*,int,char*,char*,char*))
{
  static int visited = 0;
  if (visited) SETERRQ(PETSC_ERR_SUP,0,"cannot call multiple times");
  PetscTrMalloc = imalloc;
  PetscTrFree   = ifree;
  visited       = 1;
  return 0;
}

