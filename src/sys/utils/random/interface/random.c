#define PETSC_DLL
/*
    This file contains routines for interfacing to random number generators.
    This provides more than just an interface to some system random number
    generator:

    Numbers can be shuffled for use as random tuples

    Multiple random number generators may be used

    We are still not sure what interface we want here.  There should be
    one to reinitialize and set the seed.
 */

#include "src/sys/utils/random/randomimpl.h"                              /*I "petscsys.h" I*/
#if defined (PETSC_HAVE_STDLIB_H)
#include <stdlib.h>
#endif

/* Logging support */
PetscCookie PETSC_DLLEXPORT PETSC_RANDOM_COOKIE = 0;

#undef __FUNCT__  
#define __FUNCT__ "PetscRandomDestroy" 
/*@
   PetscRandomDestroy - Destroys a context that has been formed by 
   PetscRandomCreate().

   Collective on PetscRandom

   Intput Parameter:
.  r  - the random number generator context

   Level: intermediate

.seealso: PetscRandomGetValue(), PetscRandomCreate(), VecSetRandom()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscRandomDestroy(PetscRandom r)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(r,PETSC_RANDOM_COOKIE,1);
  if (--r->refct > 0) PetscFunctionReturn(0);
  ierr = PetscHeaderDestroy(r);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscRandomGetInterval"
/*@C
   PetscRandomGetInterval - Gets the interval over which the random numbers
   will be randomly distributed.  By default, this interval is [0,1).

   Not collective

   Input Parameters:
.  r  - the random number generator context

   Output Parameters:
+  low - The lower bound of the interval
-  high - The upper bound of the interval

   Level: intermediate

   Concepts: random numbers^range

.seealso: PetscRandomCreate(), PetscRandomSetInterval()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscRandomGetInterval(PetscRandom r,PetscScalar *low,PetscScalar *high)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(r,PETSC_RANDOM_COOKIE,1);
  if (low) {
    PetscValidScalarPointer(low,2);
    *low = r->low;
  }
  if (high) {
    PetscValidScalarPointer(high,3);
    *high = r->low+r->width;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscRandomSetInterval"
/*@C
   PetscRandomSetInterval - Sets the interval over which the random numbers
   will be randomly distributed.  By default, this interval is [0,1).

   Not collective

   Input Parameters:
+  r  - the random number generator context
.  low - The lower bound of the interval
-  high - The upper bound of the interval

   Level: intermediate

   Concepts: random numbers^range

.seealso: PetscRandomCreate(), PetscRandomGetInterval()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscRandomSetInterval(PetscRandom r,PetscScalar low,PetscScalar high)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(r,PETSC_RANDOM_COOKIE,1);
#if defined(PETSC_USE_COMPLEX)
  if (PetscRealPart(low) >= PetscRealPart(high))           SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"only low < high");
  if (PetscImaginaryPart(low) >= PetscImaginaryPart(high)) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"only low < high");
#else
  if (low >= high) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"only low < high: Instead %G %G",low,high);
#endif
  r->low   = low;
  r->width = high-low;
  r->iset  = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscRandomGetSeed"
/*@C
   PetscRandomGetSeed - Gets the random seed.

   Not collective

   Input Parameters:
.  r - The random number generator context

   Output Parameter:
.  seed - The random seed

   Level: intermediate

   Concepts: random numbers^seed

.seealso: PetscRandomCreate(), PetscRandomSetSeed(), PetscRandomSeed()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscRandomGetSeed(PetscRandom r,unsigned long *seed)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(r,PETSC_RANDOM_COOKIE,1);
  if (seed) {
    PetscValidPointer(seed,2);
    *seed = r->seed;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscRandomSetSeed"
/*@C
   PetscRandomSetSeed - Sets the random seed.

   Not collective

   Input Parameters:
+  r  - The random number generator context
-  seed - The random seed

   Level: intermediate

   Concepts: random numbers^seed

.seealso: PetscRandomCreate(), PetscRandomGetSeed(), PetscRandomSeed()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscRandomSetSeed(PetscRandom r,unsigned long seed)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(r,PETSC_RANDOM_COOKIE,1);
  r->seed = seed;
  PetscFunctionReturn(0);
}



#undef __FUNCT__  
#define __FUNCT__ "PetscRandomCreate" 
/*@
   PetscRandomCreate - Creates a context for generating random numbers,
   and initializes the random-number generator.

   Collective on MPI_Comm

   Input Parameters:
+  comm - MPI communicator
-  type - the type of random numbers to be generated, usually RANDOM_DEFAULT

   Output Parameter:
.  r  - the random number generator context

   Level: intermediate

   Notes:
   By default, we generate random numbers via srand48()/drand48() that
   are uniformly distributed over [0,1).  The user can shift and stretch
   this interval by calling PetscRandomSetInterval().
  
   Currently three types of random numbers are supported. These types
   are equivalent when working with real numbers.
.     RANDOM_DEFAULT - both real and imaginary components are random
.     RANDOM_DEFAULT_REAL - real component is random; imaginary component is 0
.     RANDOM_DEFAULT_IMAGINARY - imaginary component is random; real component is 0

   Use VecSetRandom() to set the elements of a vector to random numbers.

   Example of Usage:
.vb
      PetscRandomCreate(PETSC_COMM_SELF,RANDOM_DEFAULT,&r);
      PetscRandomGetValue(r,&value1);
      PetscRandomGetValue(r,&value2);
      PetscRandomGetValue(r,&value3);
      PetscRandomDestroy(r);
.ve

   Concepts: random numbers^creating

.seealso: PetscRandomGetValue(), PetscRandomSetInterval(), PetscRandomDestroy(), VecSetRandom()
@*/

PetscErrorCode PETSC_DLLEXPORT PetscRandomCreate(MPI_Comm comm,PetscRandom *r)
{
  PetscRandom    rr;
  PetscErrorCode ierr;
  PetscMPIInt    rank;

  PetscFunctionBegin;
  PetscValidPointer(r,3);
  *r = PETSC_NULL;
#ifndef PETSC_USE_DYNAMIC_LIBRARIES
  ierr = PetscRandomInitializePackage(PETSC_NULL);CHKERRQ(ierr);
#endif

  ierr = PetscHeaderCreate(rr,_p_PetscRandom,struct _PetscRandomOps,PETSC_RANDOM_COOKIE,-1,"PetscRandom",comm,PetscRandomDestroy,0);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory(rr, sizeof(struct _p_PetscRandom));CHKERRQ(ierr);
  ierr = PetscMemzero(rr->ops, sizeof(struct _PetscRandomOps));CHKERRQ(ierr);
  rr->bops->publish = PETSC_NULL;
  rr->type_name     = PETSC_NULL;

  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  rr->data  = PETSC_NULL;
  rr->low   = 0.0;
  rr->width = 1.0;
  rr->iset  = PETSC_FALSE;
  rr->seed  = 0x12345678+rank;
  *r = rr;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscRandomSeed"
/*@C
   PetscRandomSeed - Seed the generator.

   Not collective

   Input Parameters:
.  r - The random number generator context

   Level: intermediate

   Concepts: random numbers^seed

.seealso: PetscRandomCreate(), PetscRandomGetSeed(), PetscRandomSetSeed()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscRandomSeed(PetscRandom r)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(r,PETSC_RANDOM_COOKIE,1);
  PetscValidType(r,1);

  ierr = (*r->ops->seed)(r);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)r);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscRandomGetValue"
/*@
   PetscRandomGetValue - Generates a random number.  Call this after first calling
   PetscRandomCreate().

   Not Collective

   Intput Parameter:
.  r  - the random number generator context

   Output Parameter:
.  val - the value

   Level: intermediate

   Notes:
   Use VecSetRandom() to set the elements of a vector to random numbers.

   Example of Usage:
.vb
      PetscRandomCreate(PETSC_COMM_WORLD,RANDOM_DEFAULT,&r);
      PetscRandomGetValue(r,&value1);
      PetscRandomGetValue(r,&value2);
      PetscRandomGetValue(r,&value3);
      PetscRandomDestroy(r);
.ve

   Concepts: random numbers^getting

.seealso: PetscRandomCreate(), PetscRandomDestroy(), VecSetRandom()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscRandomGetValue(PetscRandom r,PetscScalar *val)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(r,PETSC_RANDOM_COOKIE,1);
  PetscValidIntPointer(val,2);
  PetscValidType(r,1);

  ierr = (*r->ops->getvalue)(r,val);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)r);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscRandomGetValueReal"
/*@
   PetscRandomGetValueReal - Generates a random number.  Call this after first calling
   PetscRandomCreate().

   Not Collective

   Intput Parameter:
.  r  - the random number generator context

   Output Parameter:
.  val - the value

   Level: intermediate

   Notes:
   Use VecSetRandom() to set the elements of a vector to random numbers.

   Example of Usage:
.vb
      PetscRandomCreate(PETSC_COMM_WORLD,RANDOM_DEFAULT,&r);
      PetscRandomGetValueReal(r,&value1);
      PetscRandomGetValueReal(r,&value2);
      PetscRandomGetValueReal(r,&value3);
      PetscRandomDestroy(r);
.ve

   Concepts: random numbers^getting

.seealso: PetscRandomCreate(), PetscRandomDestroy(), VecSetRandom()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscRandomGetValueReal(PetscRandom r,PetscReal *val)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(r,PETSC_RANDOM_COOKIE,1);
  PetscValidIntPointer(val,2);
  PetscValidType(r,1);

  ierr = (*r->ops->getvaluereal)(r,val);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)r);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscRandomGetValueImaginary"
/*@
   PetscRandomGetValueImaginary - Generates a random number.  Call this after first calling
   PetscRandomCreate().

   Not Collective

   Intput Parameter:
.  r  - the random number generator context

   Output Parameter:
.  val - the value

   Level: intermediate

   Notes:
   Use VecSetRandom() to set the elements of a vector to random numbers.

   Example of Usage:
.vb
      PetscRandomCreate(PETSC_COMM_WORLD,RANDOM_DEFAULT,&r);
      PetscRandomGetValueImaginary(r,&value1);
      PetscRandomGetValueImaginary(r,&value2);
      PetscRandomGetValueImaginary(r,&value3);
      PetscRandomDestroy(r);
.ve

   Concepts: random numbers^getting

.seealso: PetscRandomCreate(), PetscRandomDestroy(), VecSetRandom()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscRandomGetValueImaginary(PetscRandom r,PetscScalar *val)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(r,PETSC_RANDOM_COOKIE,1);
  PetscValidIntPointer(val,2);
  PetscValidType(r,1);

  ierr = (*r->ops->getvalueimaginary)(r,val);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)r);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
