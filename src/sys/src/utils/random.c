#ifndef lint
static char vcid[] = "$Id: options.c,v 1.65 1996/01/13 13:42:38 balay Exp bsmith $";
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

#define SYRANDOM_COOKIE PETSC_COOKIE+19

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

int SYRandomCreate(SYRandomType type,SYRandom *r)
{
  SYRandom rr;
  *r = 0;
  PetscHeaderCreate(rr,_SYRandom,SYRANDOM_COOKIE,type,MPI_COMM_SELF);
  srand48(0x12345678);
  *r = rr;
  return 0;
}

int SYRandomDestroy(SYRandom r)
{
  PETSCVALIDHEADERSPECIFIC(r,SYRANDOM_COOKIE);
  PetscFree(r);
  return 0;
}

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
  PetscHeaderCreate(rr,_SYRandom,SYRANDOM_COOKIE,type,MPI_COMM_SELF);
  *r = rr;
  return 0;
}

int SYRandomDestroy(SYRandom r)
{
  PETSCVALIDHEADERSPECIFIC(r,SYRANDOM_COOKIE);
  PetscFree(r);
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
