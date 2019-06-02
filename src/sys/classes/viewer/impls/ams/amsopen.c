
#include <petsc/private/viewerimpl.h>   /*I  "petscsys.h"  */
#include <petscviewersaws.h>

/*@C
    PetscViewerSAWsOpen - Opens an SAWs PetscViewer.

    Collective

    Input Parameters:
.   comm - the MPI communicator

    Output Parameter:
.   lab - the PetscViewer

    Options Database Keys:
+   -saws_port <port number> - port number where you are running SAWs client
.   -xxx_view saws - publish the object xxx
-   -xxx_saws_block - blocks the program at the end of a critical point (for KSP and SNES it is the end of a solve) until
                    the user unblocks the problem with an external tool that access the object with SAWS

    Level: advanced

    Fortran Note:
    This routine is not supported in Fortran.


    Notes:
    Unlike other viewers that only access the object being viewed on the call to XXXView(object,viewer) the SAWs viewer allows
    one to view the object asynchronously as the program continues to run. One can remove SAWs access to the object with a call to
    PetscObjectSAWsViewOff().

    Information about the SAWs is available via https://bitbucket.org/saws/saws

.seealso: PetscViewerDestroy(), PetscViewerStringSPrintf(), PETSC_VIEWER_SAWS_(), PetscObjectSAWsBlock(),
          PetscObjectSAWsViewOff(), PetscObjectSAWsTakeAccess(), PetscObjectSAWsGrantAccess()

@*/
PetscErrorCode PetscViewerSAWsOpen(MPI_Comm comm,PetscViewer *lab)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscViewerCreate(comm,lab);CHKERRQ(ierr);
  ierr = PetscViewerSetType(*lab,PETSCVIEWERSAWS);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   PetscObjectViewSAWs - View the base portion of any object with an SAWs viewer

   Collective on PetscObject

   Input Parameters:
+  obj - the Petsc variable
         Thus must be cast with a (PetscObject), for example,
         PetscObjectSetName((PetscObject)mat,name);
-  viewer - the SAWs viewer

   Level: advanced

   Developer Note: Currently this is called only on rank zero of PETSC_COMM_WORLD

   The object must have already been named before calling this routine since naming an
   object can be collective.


.seealso: PetscObjectSetName(), PetscObjectSAWsViewOff()

@*/
PetscErrorCode  PetscObjectViewSAWs(PetscObject obj,PetscViewer viewer)
{
  PetscErrorCode ierr;
  char           dir[1024];
  PetscMPIInt    rank;

  PetscFunctionBegin;
  PetscValidHeader(obj,1);
  if (obj->amsmem) PetscFunctionReturn(0);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  if (rank) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Should only be being called on rank zero");
  if (!obj->name) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Object must already have been named");

  obj->amsmem = PETSC_TRUE;
  ierr = PetscSNPrintf(dir,1024,"/PETSc/Objects/%s/Class",obj->name);CHKERRQ(ierr);
  PetscStackCallSAWs(SAWs_Register,(dir,&obj->class_name,1,SAWs_READ,SAWs_STRING));
  ierr = PetscSNPrintf(dir,1024,"/PETSc/Objects/%s/Type",obj->name);CHKERRQ(ierr);
  PetscStackCallSAWs(SAWs_Register,(dir,&obj->type_name,1,SAWs_READ,SAWs_STRING));
  ierr = PetscSNPrintf(dir,1024,"/PETSc/Objects/%s/__Id",obj->name);CHKERRQ(ierr);
  PetscStackCallSAWs(SAWs_Register,(dir,&obj->id,1,SAWs_READ,SAWs_INT));
  ierr = PetscSNPrintf(dir,1024,"/PETSc/Objects/%s/__ParentID",obj->name);CHKERRQ(ierr);
  PetscStackCallSAWs(SAWs_Register,(dir,&obj->parentid,1,SAWs_READ,SAWs_INT));
  PetscFunctionReturn(0);
}
