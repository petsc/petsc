#ifndef lint
static char vcid[] = "$Id: random.c,v 1.2 1996/01/22 00:41:57 curfman Exp curfman $";
#endif

/*
    This file contains routines for interfacing to random number generators.
    This provides more than just an interface to some system random number
    generator:

    Numbers can be shuffled for use as random tuples

    Multiple random number generators may be used

    I'm still not sure what interface I want here.  There should be
    one to reinitialize and set the seed.
 */

#include "petsc.h"
#include "sys.h"
#include "stdlib.h"

/* Private data */
struct _SYRandom {
  PETSCHEADER                         /* general PETSc header */
  unsigned long seed;
  /* array for shuffling ??? */
};

#if defined(PARCH_sun4)
#if defined(__cplusplus)
extern "C" {
extern double drand48();
extern void   srand48();
}
#else
extern double drand48();
extern void   srand48();
#endif

/*@
   SYRandomCreate - Creates a context for generating random numbers.

   Input Parameter:
.  type - the type of random numbers to be generated

   Output Parameter:
.  r  - the random number generator context

   Notes:
   Currently, the only type of random numbers supported is
   RANDOM_DEFAULT.

   Use VecSetRandom() to set the elements of a vector to random numbers.

   Example of Usage:
$    SYRandomCreate(RANDOM_DEFAULT,&r);
$    SYRandomGetValue(r,&value1);
$    SYRandomGetValue(r,&value2);
$    SYRandomGetValue(r,&value3);
$    SYRandomDestroy(r);

.keywords: system, random, create

.seealso: SYRandomGetValue(), SYRandomDestroy(), VecSetRandom()
@*/
int SYRandomCreate(SYRandomType type,SYRandom *r)
{
  SYRandom rr;
  *r = 0;
  if (type != RANDOM_DEFAULT)
    SETERRQ(PETSC_ERR_SUP,"SyRandomCreate:Not for this random number type");
  PetscHeaderCreate(rr,_SYRandom,SYRANDOM_COOKIE,type,MPI_COMM_SELF);
  PLogObjectCreate(rr);
  srand48(0x12345678);
  *r = rr;
  return 0;
}

/*@
   SYRandomDestroy - Destroys a context that has been formed by 
   SYRandomCreate().

   Intput Parameter:
.  r  - the random number generator context

.keywords: system, random, destroy

.seealso: SYRandomGetValue(), SYRandomCreate(), VecSetRandom()
@*/
int SYRandomDestroy(SYRandom r)
{
  PETSCVALIDHEADERSPECIFIC(r,SYRANDOM_COOKIE);
  PLogObjectDestroy((PetscObject)r);
  PetscHeaderDestroy((PetscObject)r);
  return 0;
}

/*@
   SYRandomGetValue - Generates a random number.  You must first call
   SYRandomCreate().

   Intput Parameter:
.  r  - the random number generator context

   Notes:
   Use VecSetRandom() to set the elements of a vector to random numbers.

   Example of Usage:
$    SYRandomCreate(RANDOM_DEFAULT,&r);
$    SYRandomGetValue(r,&value1);
$    SYRandomGetValue(r,&value2);
$    SYRandomGetValue(r,&value3);
$    SYRandomDestroy(r);

.keywords: system, random, get, value

.seealso: SYRandomCreate(), SYRandomDestroy(), VecSetRandom()
@*/
int SYRandomGetValue(SYRandom r,Scalar *val)
{
  PETSCVALIDHEADERSPECIFIC(r,SYRANDOM_COOKIE);
  *val = drand48();
  /* what should we do for complex numbers? */
  return 0;
}

#else
/* Should put a simple, portable random number generator here */

extern double drand48();
int SYRandomCreate(SYRandomType type,SYRandom *r)
{
  SYRandom rr;
  *r = 0;
  if (type != RANDOM_DEFAULT)
    SETERRQ(PETSC_ERR_SUP,"SyRandomCreate:Not for this random number type");
  PetscHeaderCreate(rr,_SYRandom,SYRANDOM_COOKIE,type,MPI_COMM_SELF);
  PLogObjectCreate(rr);
  *r = rr;
  return 0;
}

int SYRandomDestroy(SYRandom r)
{
  PETSCVALIDHEADERSPECIFIC(r,SYRANDOM_COOKIE);
  PLogObjectDestroy((PetscObject)r);
  PetscHeaderDestroy((PetscObject)r);
  return 0;
}

int SYRandomGetValue(SYRandom r,Scalar *val)
{
  PETSCVALIDHEADERSPECIFIC(r,SYRANDOM_COOKIE);
  /* what should we do for complex numbers? */
  *val = 0.5;
  return 0;
}
#endif
