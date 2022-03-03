
/*
    This file contains routines for interfacing to random number generators.
    This provides more than just an interface to some system random number
    generator:

    Numbers can be shuffled for use as random tuples

    Multiple random number generators may be used

    We are still not sure what interface we want here.  There should be
    one to reinitialize and set the seed.
 */

#include <petsc/private/randomimpl.h>                              /*I "petscsys.h" I*/

/*@
   PetscRandomGetValue - Generates a random number.  Call this after first calling
   PetscRandomCreate().

   Not Collective

   Input Parameter:
.  r  - the random number generator context

   Output Parameter:
.  val - the value

   Level: intermediate

   Notes:
   Use VecSetRandom() to set the elements of a vector to random numbers.

   When PETSc is compiled for complex numbers this returns a complex number with random real and complex parts.
   Use PetscRandomGetValueReal() to get a random real number.

   To get a complex number with only a random real part, first call PetscRandomSetInterval() with a equal
   low and high imaginary part. Similarly to get a complex number with only a random imaginary part call
   PetscRandomSetInterval() with a equal low and high real part.

   Example of Usage:
.vb
      PetscRandomCreate(PETSC_COMM_WORLD,&r);
      PetscRandomGetValue(r,&value1);
      PetscRandomGetValue(r,&value2);
      PetscRandomGetValue(r,&value3);
      PetscRandomDestroy(&r);
.ve

.seealso: PetscRandomCreate(), PetscRandomDestroy(), VecSetRandom(), PetscRandomGetValueReal()
@*/
PetscErrorCode  PetscRandomGetValue(PetscRandom r,PetscScalar *val)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(r,PETSC_RANDOM_CLASSID,1);
  PetscValidType(r,1);
  if (!r->ops->getvalue) {
    PetscCheck(r->ops->getvalues,PetscObjectComm((PetscObject)r),PETSC_ERR_SUP,"Random type %s cannot generate PetscScalar",((PetscObject)r)->type_name);
    CHKERRQ((*r->ops->getvalues)(r,1,val));
  } else {
    CHKERRQ((*r->ops->getvalue)(r,val));
  }
  CHKERRQ(PetscObjectStateIncrease((PetscObject)r));
  PetscFunctionReturn(0);
}

/*@
   PetscRandomGetValueReal - Generates a real random number.  Call this after first calling
   PetscRandomCreate().

   Not Collective

   Input Parameter:
.  r  - the random number generator context

   Output Parameter:
.  val - the value

   Level: intermediate

   Notes:
   Use VecSetRandom() to set the elements of a vector to random numbers.

   Example of Usage:
.vb
      PetscRandomCreate(PETSC_COMM_WORLD,&r);
      PetscRandomGetValueReal(r,&value1);
      PetscRandomGetValueReal(r,&value2);
      PetscRandomGetValueReal(r,&value3);
      PetscRandomDestroy(&r);
.ve

.seealso: PetscRandomCreate(), PetscRandomDestroy(), VecSetRandom(), PetscRandomGetValue()
@*/
PetscErrorCode  PetscRandomGetValueReal(PetscRandom r,PetscReal *val)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(r,PETSC_RANDOM_CLASSID,1);
  PetscValidType(r,1);
  if (!r->ops->getvaluereal) {
    PetscCheck(r->ops->getvaluesreal,PetscObjectComm((PetscObject)r),PETSC_ERR_SUP,"Random type %s cannot generate PetscReal",((PetscObject)r)->type_name);
    CHKERRQ((*r->ops->getvaluesreal)(r,1,val));
  } else {
    CHKERRQ((*r->ops->getvaluereal)(r,val));
  }
  CHKERRQ(PetscObjectStateIncrease((PetscObject)r));
  PetscFunctionReturn(0);
}

/*@
   PetscRandomGetValues - Generates a sequence of random numbers.  Call this after first calling
   PetscRandomCreate().

   Not Collective

   Input Parameters:
+  r  - the random number generator context
-  n  - number of random numbers to generate

   Output Parameter:
.  val - the array to hold the values

   Level: intermediate

   Notes:
   Use VecSetRandom() to set the elements of a vector to random numbers.

   When PETSc is compiled for complex numbers this returns an array of complex numbers with random real and complex parts.
   Use PetscRandomGetValuesReal() to get an array of random real numbers.

.seealso: PetscRandomCreate(), PetscRandomDestroy(), VecSetRandom(), PetscRandomGetValue()
@*/
PetscErrorCode  PetscRandomGetValues(PetscRandom r, PetscInt n, PetscScalar *val)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(r,PETSC_RANDOM_CLASSID,1);
  PetscValidType(r,1);
  if (!r->ops->getvalues) {
    PetscInt i;
    PetscCheck(r->ops->getvalue,PetscObjectComm((PetscObject)r),PETSC_ERR_SUP,"Random type %s cannot generate PetscScalar",((PetscObject)r)->type_name);
    for (i = 0; i < n; i++) {
      CHKERRQ((*r->ops->getvalue)(r,val+i));
    }
  } else {
    CHKERRQ((*r->ops->getvalues)(r,n,val));
  }
  CHKERRQ(PetscObjectStateIncrease((PetscObject)r));
  PetscFunctionReturn(0);
}

/*@
   PetscRandomGetValuesReal - Generates a sequence of real random numbers.  Call this after first calling
   PetscRandomCreate().

   Not Collective

   Input Parameters:
+  r  - the random number generator context
-  n  - number of random numbers to generate

   Output Parameter:
.  val - the array to hold the values

   Level: intermediate

   Notes:
   Use VecSetRandom() to set the elements of a vector to random numbers.

.seealso: PetscRandomCreate(), PetscRandomDestroy(), VecSetRandom(), PetscRandomGetValues()
@*/
PetscErrorCode  PetscRandomGetValuesReal(PetscRandom r, PetscInt n, PetscReal *val)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(r,PETSC_RANDOM_CLASSID,1);
  PetscValidType(r,1);
  if (!r->ops->getvaluesreal) {
    PetscInt i;
    PetscCheck(r->ops->getvaluereal,PetscObjectComm((PetscObject)r),PETSC_ERR_SUP,"Random type %s cannot generate PetscReal",((PetscObject)r)->type_name);
    for (i = 0; i < n; i++) {
      CHKERRQ((*r->ops->getvaluereal)(r,val+i));
    }
  } else {
    CHKERRQ((*r->ops->getvaluesreal)(r,n,val));
  }
  CHKERRQ(PetscObjectStateIncrease((PetscObject)r));
  PetscFunctionReturn(0);
}

/*@
   PetscRandomGetInterval - Gets the interval over which the random numbers
   will be randomly distributed.  By default, this interval is [0,1).

   Not collective

   Input Parameter:
.  r  - the random number generator context

   Output Parameters:
+  low - The lower bound of the interval
-  high - The upper bound of the interval

   Level: intermediate

.seealso: PetscRandomCreate(), PetscRandomSetInterval()
@*/
PetscErrorCode  PetscRandomGetInterval(PetscRandom r,PetscScalar *low,PetscScalar *high)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(r,PETSC_RANDOM_CLASSID,1);
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

/*@
   PetscRandomSetInterval - Sets the interval over which the random numbers
   will be randomly distributed.  By default, this interval is [0,1).

   Not collective

   Input Parameters:
+  r  - the random number generator context
.  low - The lower bound of the interval
-  high - The upper bound of the interval

   Level: intermediate

   Notes:
    for complex numbers either the real part or the imaginary part of high must be greater than its low part; or both of them can be greater.
    If the real or imaginary part of low and high are the same then that value is always returned in the real or imaginary part.

.seealso: PetscRandomCreate(), PetscRandomGetInterval()
@*/
PetscErrorCode  PetscRandomSetInterval(PetscRandom r,PetscScalar low,PetscScalar high)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(r,PETSC_RANDOM_CLASSID,1);
#if defined(PETSC_USE_COMPLEX)
  PetscCheckFalse(PetscRealPart(low) > PetscRealPart(high),PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"only low <= high");
  PetscCheckFalse(PetscImaginaryPart(low) > PetscImaginaryPart(high),PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"only low <= high");
#else
  PetscCheckFalse(low >= high,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"only low <= high: Instead %g %g",(double)low,(double)high);
#endif
  r->low   = low;
  r->width = high-low;
  r->iset  = PETSC_TRUE;
  PetscFunctionReturn(0);
}
