#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: mal.c,v 1.27 1997/10/19 03:23:45 bsmith Exp bsmith $";
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

static int petscsetmallocvisited = 0;

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
  PetscFunctionBegin;
  if (petscsetmallocvisited) SETERRQ(PETSC_ERR_SUP,0,"cannot call multiple times");
  PetscTrMalloc               = imalloc;
  PetscTrFree                 = ifree;
  petscsetmallocvisited       = 1;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PetscClearMalloc"
/*@C
   PetscClearMalloc - Clears the routines used to do mallocs and frees.

.keywords: Petsc, set, malloc, free, memory allocation
@*/
int PetscClearMalloc()
{
  PetscFunctionBegin;
  PetscTrMalloc               = (void*(*)(unsigned int,int,char*,char*,char*))malloc;
  PetscTrFree                 = (int (*)(void*,int,char*,char*,char*))free;
  petscsetmallocvisited       = 0;
  PetscFunctionReturn(0);
}
