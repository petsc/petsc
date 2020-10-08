
#include <petsc/private/petscimpl.h>        /*I    "petscsys.h"   I*/
#include <petscviewer.h>

/*@C
   PetscObjectSetName - Sets a string name associated with a PETSc object.

   Not Collective

   Input Parameters:
+  obj - the Petsc variable
         Thus must be cast with a (PetscObject), for example,
         PetscObjectSetName((PetscObject)mat,name);
-  name - the name to give obj

   Notes:
    If this routine is not called then the object may end up being name by PetscObjectName().
   Level: advanced

.seealso: PetscObjectGetName(), PetscObjectName()
@*/
PetscErrorCode  PetscObjectSetName(PetscObject obj,const char name[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeader(obj,1);
  ierr = PetscFree(obj->name);CHKERRQ(ierr);
  ierr = PetscStrallocpy(name,&obj->name);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
      PetscObjectPrintClassNamePrefixType - used in the XXXView() methods to display information about the class, name, prefix and type of an object

   Input Parameters:
+     obj - the PETSc object
-     viewer - ASCII viewer where the information is printed, function does nothing if the viewer is not PETSCVIEWERASCII type

   Level: developer

   Notes:
    If the viewer format is PETSC_VIEWER_ASCII_MATLAB then the information is printed after a % symbol
          so that MATLAB will treat it as a comment.

          If the viewer format is PETSC_VIEWER_ASCII_VTK*, PETSC_VIEWER_ASCII_LATEX, or
          PETSC_VIEWER_ASCII_MATRIXMARKET then don't print header information
          as these formats can't process it.

   Developer Note: The flag donotPetscObjectPrintClassNamePrefixType is useful to prevent double printing of the information when recursion is used
                   to actually print the object.

.seealso: PetscObjectSetName(), PetscObjectName()

@*/
PetscErrorCode PetscObjectPrintClassNamePrefixType(PetscObject obj,PetscViewer viewer)
{
  PetscErrorCode    ierr;
  MPI_Comm          comm;
  PetscMPIInt       size;
  PetscViewerFormat format;
  PetscBool         flg;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&flg);CHKERRQ(ierr);
  if (obj->donotPetscObjectPrintClassNamePrefixType) PetscFunctionReturn(0);
  if (!flg) PetscFunctionReturn(0);

  ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
  if (format == PETSC_VIEWER_ASCII_VTK_DEPRECATED || format == PETSC_VIEWER_ASCII_VTK_CELL_DEPRECATED || format == PETSC_VIEWER_ASCII_VTK_COORDS_DEPRECATED || format == PETSC_VIEWER_ASCII_MATRIXMARKET || format == PETSC_VIEWER_ASCII_LATEX || format == PETSC_VIEWER_ASCII_GLVIS) PetscFunctionReturn(0);

  if (format == PETSC_VIEWER_ASCII_MATLAB) {ierr = PetscViewerASCIIPrintf(viewer,"%%");CHKERRQ(ierr);}
  ierr = PetscViewerASCIIPrintf(viewer,"%s Object:",obj->class_name);CHKERRQ(ierr);
  ierr = PetscViewerASCIIUseTabs(viewer,PETSC_FALSE);CHKERRQ(ierr);
  if (obj->name) {
    ierr = PetscViewerASCIIPrintf(viewer," %s",obj->name);CHKERRQ(ierr);
  }
  if (obj->prefix) {
    ierr = PetscViewerASCIIPrintf(viewer," (%s)",obj->prefix);CHKERRQ(ierr);
  }
  ierr = PetscObjectGetComm(obj,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRMPI(ierr);
  ierr = PetscViewerASCIIPrintf(viewer," %d MPI processes\n",size);CHKERRQ(ierr);
  ierr = PetscViewerASCIIUseTabs(viewer,PETSC_TRUE);CHKERRQ(ierr);
  if (format == PETSC_VIEWER_ASCII_MATLAB) {ierr = PetscViewerASCIIPrintf(viewer,"%%");CHKERRQ(ierr);}
  if (obj->type_name) {
    ierr = PetscViewerASCIIPrintf(viewer,"  type: %s\n",obj->type_name);CHKERRQ(ierr);
  } else {
    ierr = PetscViewerASCIIPrintf(viewer,"  type not yet set\n");CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@C
   PetscObjectName - Gives an object a name if it does not have one

   Collective

   Input Parameters:
.  obj - the Petsc variable
         Thus must be cast with a (PetscObject), for example,
         PetscObjectName((PetscObject)mat,name);

   Level: developer

   Notes:
    This is used in a small number of places when an object NEEDS a name, for example when it is saved to MATLAB with that variable name.
          Use PetscObjectSetName() to set the name of an object to what you want. The SAWs viewer requires that no two published objects
          share the same name.

   Developer Note: this needs to generate the exact same string on all ranks that share the object. The current algorithm may not always work.



.seealso: PetscObjectGetName(), PetscObjectSetName()
@*/
PetscErrorCode  PetscObjectName(PetscObject obj)
{
  PetscErrorCode   ierr;
  PetscCommCounter *counter;
  PetscMPIInt      flg;
  char             name[64];

  PetscFunctionBegin;
  PetscValidHeader(obj,1);
  if (!obj->name) {
    union {MPI_Comm comm; void *ptr; char raw[sizeof(MPI_Comm)]; } ucomm;
    ierr = MPI_Comm_get_attr(obj->comm,Petsc_Counter_keyval,(void*)&counter,&flg);CHKERRMPI(ierr);
    if (!flg) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_CORRUPT,"Bad MPI communicator supplied; must be a PETSc communicator");
    ucomm.ptr = NULL;
    ucomm.comm = obj->comm;
    ierr = MPI_Bcast(ucomm.raw,sizeof(MPI_Comm),MPI_BYTE,0,obj->comm);CHKERRMPI(ierr);
    /* If the union has extra bytes, their value is implementation-dependent, but they will normally be what we set last
     * in 'ucomm.ptr = NULL'.  This output is always implementation-defined (and varies from run to run) so the union
     * abuse acceptable. */
    ierr = PetscSNPrintf(name,64,"%s_%p_%D",obj->class_name,ucomm.ptr,counter->namecount++);CHKERRQ(ierr);
    ierr = PetscStrallocpy(name,&obj->name);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}



PetscErrorCode  PetscObjectChangeTypeName(PetscObject obj,const char type_name[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeader(obj,1);
  ierr = PetscFree(obj->type_name);CHKERRQ(ierr);
  ierr = PetscStrallocpy(type_name,&obj->type_name);CHKERRQ(ierr);
  /* Clear all the old subtype callbacks so they can't accidentally be called (shouldn't happen anyway) */
  ierr = PetscMemzero(obj->fortrancallback[PETSC_FORTRAN_CALLBACK_SUBTYPE],obj->num_fortrancallback[PETSC_FORTRAN_CALLBACK_SUBTYPE]*sizeof(PetscFortranCallback));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

