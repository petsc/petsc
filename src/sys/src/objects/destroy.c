#ifndef lint
static char vcid[] = "$Id: destroy.c,v 1.6 1995/06/20 01:46:42 bsmith Exp curfman $";
#endif
#include "ptscimpl.h"

/*@
   PetscObjectDestroy - Destroys any PetscObject, regardless of the type. 
   This routine should seldom be needed.

   Input Parameters:
.  obj - any PETSc object, for example a Vec, Mat or KSP.

.keywords: object, destroy
@*/
int PetscObjectDestroy(PetscObject obj)
{
  if (!obj) SETERRQ(1,"PetscObjectDestroy: Null PETSc object");
  if (obj->destroy) return (*obj->destroy)(obj);
  return 0;
}

/*@
   PetscObjectGetComm - Gets the MPI communicator for any PetscObject, 
   regardless of the type.

   Input Parameter:
.  obj - any PETSc object, for example a Vec, Mat or KSP.

   Output Parameter:
.  comm - the MPI communicator

.keywords: object, get, communicator, MPI
@*/
int PetscObjectGetComm(PetscObject obj,MPI_Comm *comm)
{
  if (!obj) SETERRQ(1,"PetscObjectGetComm: Null PETSc object");
  *comm = obj->comm;
  return 0;
}

/*@
   PetscObjectExists - Determines whether a PETSc object has been destroyed.

   Input Parameter:
.  obj - any PETSc object, for example a Vec, Mat or KSP.

   Output Parameter:
.  exists - 0 if object does not exist; 1 if object does exist.

.keywords: object, exists
@*/
int PetscObjectExists(PetscObject obj,int *exists)
{
  *exists = 0;
  if (!obj) return 0;
  if (obj->cookie != FREEDHEADER) *exists = 1;
  return 0;
}
