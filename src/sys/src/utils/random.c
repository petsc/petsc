#ifndef lint
static char vcid[] = "$Id: random.c,v 1.6 1996/02/01 16:33:33 balay Exp bsmith $";
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
#include "sys.h"        /*I "sys.h" I*/
#include "stdlib.h"

/* Private data */
struct _SYRandom {
  PETSCHEADER                         /* general PETSc header */
  unsigned long seed;
  Scalar        low, high;
  /* array for shuffling ??? */
};

/* For now we've set up only the default sun4 random number generator.  
   We need to deal with other machines as well as other variants of
   random number generators. We should also add a routine to enable
   restarts [seed48()] */
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

/*@C
   SYRandomCreate - Creates a context for generating random numbers and
   initializes the random-number generator.

   Input Parameters:
.  comm - MPI communicator
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
int SYRandomCreate(MPI_Comm comm,SYRandomType type,SYRandom *r)
{
  SYRandom rr;
  int      rank;
  *r = 0;
  if (type != RANDOM_DEFAULT)
    SETERRQ(PETSC_ERR_SUP,"SyRandomCreate:Not for this random number type");
  PetscHeaderCreate(rr,_SYRandom,SYRANDOM_COOKIE,type,comm);
  PLogObjectCreate(rr);
  MPI_Comm_rank(comm,&rank);
  srand48(0x12345678+rank);
  *r = rr;
  return 0;
}

/*@C
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
   SYRandomGetValue - Generates a random number.  Call this after first calling
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
#if defined(PETSC_COMPLEX)
  complex tmp(drand48(),drand48());
  *val = tmp;
#else
  *val = drand48();
#endif
  return 0;
}

#else
/* Should put a simple, portable random number generator here */

extern double drand48();
int SYRandomCreate(MPI_Comm comm,SYRandomType type,SYRandom *r)
{
  SYRandom rr;
  *r = 0;
  if (type != RANDOM_DEFAULT)
    SETERRQ(PETSC_ERR_SUP,"SyRandomCreate:Not for this random number type");
  PetscHeaderCreate(rr,_SYRandom,SYRANDOM_COOKIE,type,comm);
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
#if defined(PETSC_COMPLEX)
  *val = (0.5,0.5);
#else
  *val = 0.5;
#endif
  return 0;
}
#endif
