#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ams.c,v 1.3 1998/08/26 21:55:14 balay Exp bsmith $";
#endif

#include "petsc.h"
#include "pinclude/pviewer.h"
#include <stdarg.h>
#if defined(HAVE_STDLIB_H)
#include <stdlib.h>
#endif

#if defined(HAVE_AMS)

#include "ams.h"
struct _p_Viewer {
  VIEWERHEADER
  char       *ams_name;
  AMS_Comm   ams_comm;
};

Viewer VIEWER_AMS_WORLD_PRIVATE = 0;

#undef __FUNC__  
#define __FUNC__ "ViewerInitializeAMSWorld_Private"
int ViewerInitializeAMSWorld_Private(void)
{
  int  ierr,port = -1;

  PetscFunctionBegin;
  if (VIEWER_AMS_WORLD_PRIVATE) PetscFunctionReturn(0);
  ierr = ViewerAMSOpen(PETSC_COMM_WORLD,"PETSc",&VIEWER_AMS_WORLD_PRIVATE); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "ViewerDestroy_AMS"
static int ViewerDestroy_AMS(Viewer viewer)
{
  int      ierr,rank;
  MPI_Comm comm;

  PetscFunctionBegin;
  PLogObjectDestroy((PetscObject)viewer);

  /* currently only first processor publishes */  
  ierr = PetscObjectGetComm((PetscObject) viewer,&comm);CHKERRQ(ierr);
  MPI_Comm_rank(comm,&rank);
  if (!rank) {
    ierr = AMS_Comm_destroy(viewer->ams_comm);
    if (ierr) {
      char *err;
      AMS_Explain_error(ierr,&err);
      printf("%s\n",err);
      SETERRQ(ierr,0," ");
    }
  }
  PetscHeaderDestroy((PetscObject)viewer);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "ViewerAMSOpen"
/*@C
    ViewerAMSOpen - Opens an AMS memory snooper viewer. 

    Cllective on MPI_Comm

    Input Parameters:
+   comm - the MPI communicator
-   name - name of AMS communicator being created

    Output Parameter:
.   lab - the viewer

    Options Database:
.   -ams_port port

    Fortran Note:
    This routine is not supported in Fortran.

.keywords: Viewer, open, AMS memory snooper

.seealso: ViewerDestroy(), ViewerStringSPrintf()
@*/
int ViewerAMSOpen(MPI_Comm comm,const char name[],Viewer *lab)
{
  Viewer v;
  int    ierr,port = -1;

  PetscFunctionBegin;
  PetscHeaderCreate(v,_p_Viewer,int,VIEWER_COOKIE,AMS_VIEWER,comm,ViewerDestroy,0);
  PLogObjectCreate(v);
  v->destroy     = ViewerDestroy_AMS;

  ierr = OptionsGetInt(PETSC_NULL,"-ams_port",&port,PETSC_NULL);CHKERRQ(ierr);
  ierr = AMS_Comm_publish((char *)name,&v->ams_comm,MPI_TYPE,comm,&port);CHKERRQ(ierr);

  *lab           = v;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "ViewerAMSGetAMSComm"
/*@C
    ViewerAMSGetAMSComm - Gets the AMS communicator associated with 
         the viewer.

    Collective on MPI_Comm

    Input Parameters:
.   lab - the viewer

    Output Parameter:
.    ams_comm - the AMS communicator

    Fortran Note:
    This routine is not supported in Fortran.

.keywords: Viewer, open, AMS memory snooper

.seealso: ViewerDestroy(), ViewerAMSOpen()
@*/
int ViewerAMSGetAMSComm(Viewer lab,AMS_Comm *ams_comm)
{
  PetscFunctionBegin;
  if (lab->type != AMS_VIEWER) SETERRQ(1,1,"Not an AMS viewer");

  *ams_comm = lab->ams_comm;
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------------*/
/*
    The variable Petsc_Viewer_Ams_keyval is used to indicate an MPI attribute that
  is attached to a communicator, in this case the attribute is a Viewer.
*/
static int Petsc_Viewer_Ams_keyval = MPI_KEYVAL_INVALID;

#undef __FUNC__  
#define __FUNC__ "VIEWER_AMS_" 
/*@C
     VIEWER_AMS_ - Creates a AMS memory snooper viewer shared by all processors 
                     in a communicator.

     Collective on MPI_Comm

     Input Parameters:
.    comm - the MPI communicator to share the viewer

     Notes:
     Unlike almost all other PETSc routines, VIEWER_AMS_ does not return 
     an error code.  The window viewer is usually used in the form
$       XXXView(XXX object,VIEWER_AMS_(comm));

.seealso: VIEWER_AMS_WORLD, VIEWER_AMS_SELF, ViewerAMSOpenX(), 
@*/
Viewer VIEWER_AMS_(MPI_Comm comm)
{
  int           ierr,flag,size,csize,rank;
  Viewer        viewer;
  char          name[128];

  PetscFunctionBegin;
  /*
     If communicator is across all processors then we store the AMS_Comm not 
     in the communicator but instead in a static variable. This is because this
     routine is called before the PETSC_COMM_WORLD communictor may have been 
     duplicated, thus if we stored it there we could not access it the next
     time we called this routin.
  */
  MPI_Comm_size(PETSC_COMM_WORLD,&size);
  MPI_Comm_size(comm,&csize);
  if (size == csize) {
    viewer = VIEWER_AMS_WORLD;
    PetscFunctionReturn(viewer);
  }

  if (Petsc_Viewer_Ams_keyval == MPI_KEYVAL_INVALID) {
    ierr = MPI_Keyval_create(MPI_NULL_COPY_FN,MPI_NULL_DELETE_FN,&Petsc_Viewer_Ams_keyval,0);
    if (ierr) {PetscError(__LINE__,"VIEWER_AMS_",__FILE__,__SDIR__,1,1,0); viewer = 0;}
  }
  ierr = MPI_Attr_get( comm, Petsc_Viewer_Ams_keyval, (void **)&viewer, &flag );
  if (ierr) {PetscError(__LINE__,"VIEWER_AMS_",__FILE__,__SDIR__,1,1,0); viewer = 0;}
  if (!flag) { /* viewer not yet created */

    if (csize == 1) {
      MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
      sprintf(name,"PETSc_%d",rank);
    } else {
      PetscError(__LINE__,"VIEWER_AMS_",__FILE__,__SDIR__,1,1,0); viewer = 0;
    }
    ierr = ViewerAMSOpen(comm,name,&viewer); 
    if (ierr) {PetscError(__LINE__,"VIEWER_AMS_",__FILE__,__SDIR__,1,1,0); viewer = 0;}
    ierr = MPI_Attr_put( comm, Petsc_Viewer_Ams_keyval, (void *) viewer );
    if (ierr) {PetscError(__LINE__,"VIEWER_AMS_",__FILE__,__SDIR__,1,1,0); viewer = 0;}
  } 
  PetscFunctionReturn(viewer);
}

/*
       If there is a Viewer associated with this communicator it is destroyed.
*/
int VIEWER_AMS_Destroy(MPI_Comm comm)
{
  int    ierr,flag;
  Viewer viewer;

  PetscFunctionBegin;
  if (Petsc_Viewer_Ams_keyval == MPI_KEYVAL_INVALID) {
    PetscFunctionReturn(0);
  }
  ierr = MPI_Attr_get( comm, Petsc_Viewer_Ams_keyval, (void **)&viewer, &flag );CHKERRQ(ierr);
  if (flag) { 
    ierr = ViewerDestroy(viewer); CHKERRQ(ierr);
    ierr = MPI_Attr_delete(comm,Petsc_Viewer_Ams_keyval);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "ViewerDestroyAMS_Private" 
int ViewerDestroyAMS_Private(void)
{
  int ierr;

  PetscFunctionBegin;
  ierr = VIEWER_AMS_Destroy(PETSC_COMM_SELF);CHKERRQ(ierr);
  ierr = ViewerDestroy(VIEWER_AMS_WORLD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#else

int dummy_()
{
  return 0;
}

#endif

