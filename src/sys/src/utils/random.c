
#ifndef lint
static char vcid[] = "$Id: random.c,v 1.10 1996/03/20 22:54:44 bsmith Exp curfman $";
#endif

/*
    This file contains routines for interfacing to random number generators.
    This provides more than just an interface to some system random number
    generator:

    Numbers can be shuffled for use as random tuples

    Multiple random number generators may be used

    We're still not sure what interface we want here.  There should be
    one to reinitialize and set the seed.
 */

#include "petsc.h"
#include "sys.h"        /*I "sys.h" I*/
#include "stdlib.h"

/* Private data */
struct _PetscRandom {
  PETSCHEADER                         /* general PETSc header */
  unsigned long seed;
  Scalar        low, high;
  /* array for shuffling ??? */
};


/*@C
   PetscRandomDestroy - Destroys a context that has been formed by 
   PetscRandomCreate().

   Intput Parameter:
.  r  - the random number generator context

.keywords: system, random, destroy

.seealso: PetscRandomGetValue(), PetscRandomCreate(), VecSetRandom()
@*/
int PetscRandomDestroy(PetscRandom r)
{
  PetscValidHeaderSpecific(r,RANDOM_COOKIE);
  PLogObjectDestroy((PetscObject)r);
  PetscHeaderDestroy((PetscObject)r);
  return 0;
}

/*
   For now we've set up using the DRAND48() generater. We need to deal 
   with other variants of random number generators. We should also add
   a routine to enable restarts [seed48()] 
*/
#if defined(HAVE_DRAND48)
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
   PetscRandomCreate - Creates a context for generating random numbers and
   initializes the random-number generator.

   Input Parameters:
.  comm - MPI communicator
.  type - the type of random numbers to be generated, usually
          RANDOM_DEFAULT

   Output Parameter:
.  r  - the random number generator context

   Notes:
   Currently three types of random numbers are supported. These types
   are equivalent when working with real numbers.
$     RANDOM_DEFAULT - both real and imaginary components are random
$     RANDOM_DEFAULT_REAL - real component is random; imaginary component is 0
$     RANDOM_DEFAULT_IMAGINARY - imaginary component is random; real 
$                                component is 0

   Use VecSetRandom() to set the elements of a vector to random numbers.

   Example of Usage:
$    PetscRandomCreate(RANDOM_DEFAULT,&r);
$    PetscRandomGetValue(r,&value1);
$    PetscRandomGetValue(r,&value2);
$    PetscRandomGetValue(r,&value3);
$    PetscRandomDestroy(r);

.keywords: system, random, create

.seealso: PetscRandomGetValue(), PetscRandomDestroy(), VecSetRandom()
@*/
int PetscRandomCreate(MPI_Comm comm,PetscRandomType type,PetscRandom *r)
{
  PetscRandom rr;
  int      rank;
  *r = 0;
  if (type != RANDOM_DEFAULT && type != RANDOM_DEFAULT_REAL 
                             && type != RANDOM_DEFAULT_IMAGINARY)
    SETERRQ(PETSC_ERR_SUP,"PetscRandomCreate:Not for this random number type");
  PetscHeaderCreate(rr,_PetscRandom,RANDOM_COOKIE,type,comm);
  PLogObjectCreate(rr);
  MPI_Comm_rank(comm,&rank);
  srand48(0x12345678+rank);
  *r = rr;
  return 0;
}

/*@
   PetscRandomGetValue - Generates a random number.  Call this after first calling
   PetscRandomCreate().

   Intput Parameter:
.  r  - the random number generator context

   Notes:
   Use VecSetRandom() to set the elements of a vector to random numbers.

   Example of Usage:
$    PetscRandomCreate(RANDOM_DEFAULT,&r);
$    PetscRandomGetValue(r,&value1);
$    PetscRandomGetValue(r,&value2);
$    PetscRandomGetValue(r,&value3);
$    PetscRandomDestroy(r);

.keywords: system, random, get, value

.seealso: PetscRandomCreate(), PetscRandomDestroy(), VecSetRandom()
@*/
int PetscRandomGetValue(PetscRandom r,Scalar *val)
{
#if defined(PETSC_COMPLEX)
  double zero = 0.0;
  PetscValidHeaderSpecific(r,RANDOM_COOKIE);
  if (r->type == RANDOM_DEFAULT) *val = complex(drand48(),drand48());
  else if (r->type == RANDOM_DEFAULT_REAL) *val = complex(drand48(),zero);
  else if (r->type == RANDOM_DEFAULT_IMAGINARY) *val = complex(zero,drand48());
  else SETERRQ(1,"PetscRandomGetValue:Invalid random number type");
#else
  PetscValidHeaderSpecific(r,RANDOM_COOKIE);
  *val = drand48();
#endif
  return 0;
}

#else
/* Should put a simple, portable random number generator here? */

extern double drand48();

int PetscRandomCreate(MPI_Comm comm,PetscRandomType type,PetscRandom *r)
{
  PetscRandom rr;
  char   arch[10];

  *r = 0;
  if (type != RANDOM_DEFAULT)
    SETERRQ(PETSC_ERR_SUP,"PetscRandomCreate:Not for this random number type");
  PetscHeaderCreate(rr,_PetscRandom,RANDOM_COOKIE,type,comm);
  PLogObjectCreate(rr);
  *r = rr;
  PetscGetArchType(arch,10);
  PetscPrintf(comm,"PetscRandomCreate: Warning: Random number generator not set for machine %s; using fake random numbers.\n",arch);
  return 0;
}

int PetscRandomGetValue(PetscRandom r,Scalar *val)
{
  PetscValidHeaderSpecific(r,RANDOM_COOKIE);
#if defined(PETSC_COMPLEX)
  *val = (0.5,0.5);
#else
  *val = 0.5;
#endif
  return 0;
}
#endif


