#ifndef lint
static char vcid[] = "$Id: destroy.c,v 1.5 1995/06/18 16:23:36 bsmith Exp bsmith $";
#endif
#include "ptscimpl.h"

/*@
     PetscObjectDestroy - Destroys any PETSc object. This should be
        little used, but can destroy a PETSc object of which you 
        don't know the type.

  Input Parameters:
.  obj - any PETSc object, for example a Vec, Mat or KSP.

@*/
int PetscObjectDestroy(PetscObject obj)
{
  if (!obj) SETERRQ(1,"PetscObjectDestroy: Null PETSc object");
  if (obj->destroy) return (*obj->destroy)(obj);
  return 0;
}

/*@
     PetscObjectGetComm - Gets MPI Communicator for any PetscObject.

  Input Parameter:
.   obj - the Petsc Object

  Output Parameter:
.  comm - the MPI communicator
@*/
int PetscObjectGetComm(PetscObject obj,MPI_Comm *comm)
{
  if (!obj) SETERRQ(1,"PetscObjectGetComm: Null PETSc object");
  *comm = obj->comm;
  return 0;
}
