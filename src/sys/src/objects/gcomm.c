#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: gcomm.c,v 1.12 1998/03/20 22:47:23 bsmith Exp bsmith $";
#endif
/*
     Provides utility routines for manulating any type of PETSc object.
*/
#include "petsc.h"  /*I   "petsc.h"    I*/

#undef __FUNC__  
#define __FUNC__ "PetscObjectGetComm"
/*@C
   PetscObjectGetComm - Gets the MPI communicator for any PetscObject, 
   regardless of the type.

   Input Parameter:
.  obj - any PETSc object, for example a Vec, Mat or KSP. Thus must be
         cast with a (PetscObject), for example, 
         PetscObjectGetComm((PetscObject) mat,&comm);

   Output Parameter:
.  comm - the MPI communicator

   Not Collective

.keywords: object, get, communicator, MPI
@*/
int PetscObjectGetComm(PetscObject obj,MPI_Comm *comm)
{
  PetscFunctionBegin;
  if (!obj) SETERRQ(PETSC_ERR_ARG_CORRUPT,0,"Null object");
  *comm = obj->comm;
  PetscFunctionReturn(0);
}


