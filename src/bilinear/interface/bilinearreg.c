#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: bilinearreg.c,v 1.3 2000/01/10 03:12:47 knepley Exp $";
#endif

#include "src/bilinear/bilinearimpl.h"    /*I "bilinear.h"  I*/

PetscFList BilinearSerializeList              = 0;

#undef __FUNC__  
#define __FUNC__ "BilinearSetSerializeType"
/*@C
  BilinearSetSerializeType - Sets the serialization method for the bilienar operator.

  Collective on Bilinear

  Input Parameters:
+ B      - The Bilinear context
- method - A known method

  Options Database Command:
. -bilinear_serialize_type <method> - Sets the method; use -help for a list
                                      of available methods (for instance, seqdense_binary)

   Notes:
   See "petsc/include/bilinear.h" for available methods (for instance)
.  BILINEAR_SER_SEQDENSE_BINARY - Sequential dense bilienar operator to binary file

   Normally, it is best to use the BilinearSetFromOptions() command and
   then set the Bilinear type from the options database rather than by using
   this routine.  Using the options database provides the user with
   maximum flexibility in evaluating the many different solvers.
   The BilinearSetSerializeType() routine is provided for those situations
   where it is necessary to set the application ordering independently of the
   command line or options database.  This might be the case, for example,
   when the choice of solver changes during the execution of the
   program, and the user's application is taking responsibility for
   choosing the appropriate method.  In other words, this routine is
   not for beginners.

   Level: intermediate

.keywords: Bilinear, set, type, serialization
@*/
int BilinearSetSerializeType(Bilinear B, BilinearSerializeType method)
{
  int      (*r)(MPI_Comm, Bilinear *, PetscViewer, PetscTruth);
  PetscTruth match;
  int        ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(B, BILINEAR_COOKIE);
  ierr = PetscSerializeCompare((PetscObject) B, method, &match);                                          CHKERRQ(ierr);
  if (match == PETSC_TRUE) PetscFunctionReturn(0);

  /* Get the function pointers for the method requested but do not call */
  if (!BilinearSerializeRegisterAllCalled) {
    ierr = BilinearSerializeRegisterAll(PETSC_NULL);                                                      CHKERRQ(ierr);
  }
  ierr = PetscFListFind(B->comm, BilinearSerializeList, method, (void (**)(void)) &r);                    CHKERRQ(ierr);
  if (!r) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE, "Unknown serialization method: %s", method);

  ierr = PetscObjectChangeSerializeName((PetscObject) B, method);                                         CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
