/*
    This file contains routines for interfacing to random number generators.
    This provides more than just an interface to some system random number
    generator:

    Numbers can be shuffled for use as random tuples

    Multiple random number generators may be used

    We are still not sure what interface we want here.  There should be
    one to reinitialize and set the seed.
 */

#include "petsc.h"
#include "petscsys.h"        /*I "petscsys.h" I*/
#include <stdlib.h>

PetscCookie PETSC_RANDOM_COOKIE = 0;

/* Private data */
struct _p_PetscRandom {
  PETSCHEADER(int);
  unsigned    long seed;
  PetscScalar low,width;       /* lower bound and width of the interval over
                                  which the random numbers are distributed */
  PetscTruth  iset;             /* if true, indicates that the user has set the interval */
  /* array for shuffling ??? */
};

#undef __FUNCT__  
#define __FUNCT__ "PetscRandomDestroy" 
/*@C
   PetscRandomDestroy - Destroys a context that has been formed by 
   PetscRandomCreate().

   Collective on PetscRandom

   Intput Parameter:
.  r  - the random number generator context

   Level: intermediate

.seealso: PetscRandomGetValue(), PetscRandomCreate(), VecSetRandom()
@*/
PetscErrorCode PetscRandomDestroy(PetscRandom r)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(r,PETSC_RANDOM_COOKIE,1);
  if (--r->refct > 0) PetscFunctionReturn(0);
  ierr = PetscHeaderDestroy(r);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscRandomSetInterval"
/*@C
   PetscRandomSetInterval - Sets the interval over which the random numbers
   will be randomly distributed.  By default, this interval is [0,1).

   Collective on PetscRandom

   Input Parameters:
.  r  - the random number generator context

   Example of Usage:
.vb
      PetscRandomCreate(PETSC_COMM_WORLD,RANDOM_DEFAULT,&r);
      PetscRandomSetInterval(RANDOM_DEFAULT,&r);
      PetscRandomGetValue(r,&value1);
      PetscRandomGetValue(r,&value2);
      PetscRandomDestroy(r);
.ve

   Level: intermediate

   Concepts: random numbers^range

.seealso: PetscRandomCreate()
@*/
PetscErrorCode PetscRandomSetInterval(PetscRandom r,PetscScalar low,PetscScalar high)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(r,PETSC_RANDOM_COOKIE,1);
#if defined(PETSC_USE_COMPLEX)
  if (PetscRealPart(low) >= PetscRealPart(high))           SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"only low < high");
  if (PetscImaginaryPart(low) >= PetscImaginaryPart(high)) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"only low < high");
#else
  if (low >= high) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"only low < high: Instead %g %g",low,high);
#endif
  r->low   = low;
  r->width = high-low;
  r->iset  = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*
   For now we have set up using the DRAND48() generater. We need to deal 
   with other variants of random number generators. We should also add
   a routine to enable restarts [seed48()] 
*/
#if defined(PETSC_HAVE_DRAND48)
EXTERN_C_BEGIN
extern double drand48();
extern void   srand48(long);
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "PetscRandomCreate" 
/*@C
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
PetscErrorCode PetscRandomCreate(MPI_Comm comm,PetscRandomType type,PetscRandom *r)
{
  PetscRandom    rr;
  PetscErrorCode ierr;
  PetscMPIInt    rank;

  PetscFunctionBegin;
  *r = 0;
  if (type != RANDOM_DEFAULT && type != RANDOM_DEFAULT_REAL && type != RANDOM_DEFAULT_IMAGINARY){
    SETERRQ(PETSC_ERR_SUP,"Not for this random number type");
  }
  ierr = PetscHeaderCreate(rr,_p_PetscRandom,int,PETSC_RANDOM_COOKIE,type,"random",comm,PetscRandomDestroy,0);CHKERRQ(ierr);
  rr->low   = 0.0;
  rr->width = 1.0;
  rr->iset  = PETSC_FALSE;
  rr->seed  = 0;
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  srand48(0x12345678+rank);
  *r = rr;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscRandomGetValue"
/*@C
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
PetscErrorCode PetscRandomGetValue(PetscRandom r,PetscScalar *val)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(r,PETSC_RANDOM_COOKIE,1);
  PetscValidIntPointer(val,2);
#if defined(PETSC_USE_COMPLEX)
  if (r->type == RANDOM_DEFAULT) {
    if (r->iset) {
         *val = PetscRealPart(r->width)*drand48() + PetscRealPart(r->low) +
                (PetscImaginaryPart(r->width)*drand48() + PetscImaginaryPart(r->low)) * PETSC_i;
    }
    else *val = drand48() + drand48()*PETSC_i;
  } else if (r->type == RANDOM_DEFAULT_REAL) {
    if (r->iset) *val = PetscRealPart(r->width)*drand48() + PetscRealPart(r->low);
    else                       *val = drand48();
  } else if (r->type == RANDOM_DEFAULT_IMAGINARY) {
    if (r->iset) *val = (PetscImaginaryPart(r->width)*drand48()+PetscImaginaryPart(r->low))*PETSC_i;
    else         *val = drand48()*PETSC_i;
  } else {
    SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Invalid random number type");
  }
#else
  if (r->iset) *val = r->width * drand48() + r->low;
  else         *val = drand48();
#endif
  PetscFunctionReturn(0);
}

#elif defined(PETSC_HAVE_RAND)

#undef __FUNCT__  
#define __FUNCT__ "PetscRandomCreate" 
PetscErrorCode PetscRandomCreate(MPI_Comm comm,PetscRandomType type,PetscRandom *r)
{
  PetscRandom    rr;
  PetscErrorCode ierr;
  PetscMPIInt    rank;

  PetscFunctionBegin;
  ierr = PetscLogInfo((0,"PetscRandomCreate: using rand(). not as efficinet as dran48\n"));CHKERRQ(ierr);
  *r = 0;
  if (type != RANDOM_DEFAULT && type != RANDOM_DEFAULT_REAL && type != RANDOM_DEFAULT_IMAGINARY) {
    SETERRQ(PETSC_ERR_SUP,"Not for this random number type");
  }
  ierr = PetscHeaderCreate(rr,_p_PetscRandom,int,PETSC_RANDOM_COOKIE,type,"random",comm,PetscRandomDestroy,0);CHKERRQ(ierr);
  rr->low   = 0.0;
  rr->width = 1.0;
  rr->iset  = PETSC_FALSE;
  rr->seed  = 0;
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  srand(0x12345678+rank);
  *r = rr;
  PetscFunctionReturn(0);
}

#define RAND_WRAP() (rand()/(double)((unsigned int)RAND_MAX+1))
#undef __FUNCT__  
#define __FUNCT__ "PetscRandomGetValue"
PetscErrorCode PetscRandomGetValue(PetscRandom r,PetscScalar *val)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(r,PETSC_RANDOM_COOKIE,1);
  PetscValidScalarPointer(val,2);
#if defined(PETSC_USE_COMPLEX)
  if (r->type == RANDOM_DEFAULT) {
    if (r->iset)
         *val = PetscRealPart(r->width)*RAND_WRAP() + PetscRealPart(r->low) +
                (PetscImaginaryPart(r->width)*RAND_WRAP() + PetscImaginaryPart(r->low)) * PETSC_i;
    else *val = RAND_WRAP() + RAND_WRAP()*PETSC_i;
  } else if (r->type == RANDOM_DEFAULT_REAL) {
    if (r->iset) *val = PetscRealPart(r->width)*RAND_WRAP() + PetscRealPart(r->low);
    else         *val = RAND_WRAP();
  } else if (r->type == RANDOM_DEFAULT_IMAGINARY) {
    if (r->iset) *val = (PetscImaginaryPart(r->width)*RAND_WRAP()+PetscImaginaryPart(r->low))*PETSC_i;
    else         *val = RAND_WRAP()*PETSC_i;
  } else SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Invalid random number type");
#else
  if (r->iset) *val = r->width * RAND_WRAP() + r->low;
  else         *val = RAND_WRAP();
#endif
  PetscFunctionReturn(0);
}

#else
/* Should put a simple, portable random number generator here? */

extern double drand48();

#undef __FUNCT__  
#define __FUNCT__ "PetscRandomCreate" 
PetscErrorCode PetscRandomCreate(MPI_Comm comm,PetscRandomType type,PetscRandom *r)
{
  PetscRandom    rr;
  char           arch[10];
  PetscErrorCode ierr;

  PetscFunctionBegin;
  *r = 0;
  if (type != RANDOM_DEFAULT) SETERRQ(PETSC_ERR_SUP,"Not for this random number type");
  ierr = PetscHeaderCreate(rr,_p_PetscRandom,int,PETSC_RANDOM_COOKIE,type,"random",comm,PetscRandomDestroy,0);CHKERRQ(ierr);
  *r = rr;
  ierr = PetscGetArchType(arch,10);CHKERRQ(ierr);
  ierr = PetscPrintf(comm,"PetscRandomCreate: Warning: Random number generator not set for machine %s; using fake random numbers.\n",arch);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscRandomGetValue"
PetscErrorCode PetscRandomGetValue(PetscRandom r,PetscScalar *val)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(r,PETSC_RANDOM_COOKIE,1);
  PetscValidScalarPointer(val,2);
#if defined(PETSC_USE_COMPLEX)
  *val = (0.5,0.5);
#else
  *val = 0.5;
#endif
  PetscFunctionReturn(0);
}
#endif


