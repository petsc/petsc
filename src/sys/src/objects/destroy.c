#ifndef lint
static char vcid[] = "$Id: destroy.c,v 1.4 1995/06/08 03:08:02 bsmith Exp bsmith $";
#endif
#include "ptscimpl.h"

int PetscDestroy(PetscObject obj)
{
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
