#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: amsopen.c,v 1.4 1999/03/17 23:21:05 bsmith Exp bsmith $";
#endif

#include "src/sys/src/viewer/viewerimpl.h"   /*I  "petsc.h"  */
#if defined(HAVE_AMS)

#undef __FUNC__  
#define __FUNC__ "ViewerAMSOpen"
/*@C
    ViewerAMSOpen - Opens an AMS memory snooper viewer. 

    Collective on MPI_Comm

    Input Parameters:
+   comm - the MPI communicator
-   name - name of AMS communicator being created

    Output Parameter:
.   lab - the viewer

    Options Database Key:
+   -ams_port <port number>
.   -ams_publish_objects - publish all PETSc objects to be visible to the AMS memory snooper,
                           use PetscObjectPublish() to publish individual objects
-   -ams_publish_stack - publish the PETSc stack frames to the snooper

    Level: advanced

    Fortran Note:
    This routine is not supported in Fortran.

    See the matlab/petsc directory in the AMS installation for one example of external
    tools that can monitor PETSc objects that have been published.

    Notes:
    This viewer can be destroyed with ViewerDestroy().

.keywords: Viewer, open, AMS memory snooper

.seealso: PetscObjectPublish(), ViewerDestroy(), ViewerStringSPrintf()

@*/
int ViewerAMSOpen(MPI_Comm comm,const char name[],Viewer *lab)
{
  int ierr;
  
  PetscFunctionBegin;
  ierr = ViewerCreate(comm,lab);CHKERRQ(ierr);
  ierr = ViewerSetType(*lab,AMS_VIEWER);CHKERRQ(ierr);
  ierr = ViewerAMSSetCommName(*lab,name);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#else

int dummy6(void)
{ 
  return 0;
}

#endif
