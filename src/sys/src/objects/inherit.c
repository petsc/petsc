#ifndef lint
static char vcid[] = "$Id: inherit.c,v 1.1 1996/01/30 19:21:58 bsmith Exp bsmith $";
#endif
/*
     Provides utility routines for manulating any type of PETSc object.
*/
#include "petsc.h"  /*I   "petsc.h"    I*/

static int PetscObjectInherit_DefaultCopy(void *in, void **out)
{
  *out = in;
  return 0;
}

/*@C
  PetscObjectInherit - Associate another object with a given PETSc object. 
      This is to provide a limited support for inheritence when using 
      PETSc from C++.

  Input Parameters:
.   obj - the PETSc object
.   ptr - the other object to associate with the PETSc object
.   copy - a function used to copy the other object when the PETSc object 
           is copyed, or PETSC_NULL to indicate the pointer is copied.

@*/
int PetscObjectInherit(PetscObject obj,void *ptr, int (*copy)(void *,void **))
{
  obj->child = ptr;
  if (copy == PETSC_NULL) copy = PetscObjectInherit_DefaultCopy;
  obj->childcopy = copy;
  return 0;
}

