/*$Id: ams.c,v 1.26 1999/11/05 14:43:43 bsmith Exp bsmith $*/

#include "sys.h"
#include "src/sys/src/viewer/viewerimpl.h"
#if defined(PETSC_HAVE_STDLIB_H)
#include <stdlib.h>
#endif

#include "ams.h"
typedef struct {
  char       *ams_name;
  AMS_Comm   ams_comm;
} Viewer_AMS;

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ "ViewerAMSSetCommName_AMS"
int ViewerAMSSetCommName_AMS(Viewer v,const char name[])
{
  Viewer_AMS *vams = (Viewer_AMS*)v->data;
  int        ierr,port = -1;
  PetscTruth flg;

  PetscFunctionBegin;
  ierr = OptionsGetInt(PETSC_NULL,"-ams_port",&port,PETSC_NULL);CHKERRQ(ierr);
  ierr = AMS_Comm_publish((char *)name,&vams->ams_comm,MPI_TYPE,v->comm,&port);CHKERRQ(ierr);

  ierr = OptionsHasName(PETSC_NULL,"-ams_printf",&flg);CHKERRQ(ierr);
  if (!flg) {
    ierr = AMS_Set_output_file("/dev/null");CHKERRQ(ierr);
  }

  ierr = OptionsHasName(PETSC_NULL,"-ams_matlab",&flg);CHKERRQ(ierr);
  if (flg) {
    FILE *fp;
    char m[16],*machine = 0;

    ierr = OptionsGetString(PETSC_NULL,"-ams_matlab",m,16,&flg);CHKERRQ(ierr);
    if (m[0]) machine = m;
    ierr = PetscStartMatlab(v->comm,machine,"petscview",&fp);CHKERRQ(ierr);
  }

  ierr = OptionsHasName(PETSC_NULL,"-ams_java",&flg);CHKERRQ(ierr);
  if (flg) {
    FILE *fp;
    char dir[256];
    char m[16],*machine = 0;

    ierr = OptionsGetString(PETSC_NULL,"-ams_java",m,16,&flg);CHKERRQ(ierr);
    if (m[0]) machine = m;

    ierr = OptionsGetenv(v->comm,"AMS_HOME",dir,256,&flg);CHKERRQ(ierr);
    if (!flg) {
      ierr = PetscStrncpy(dir,AMS_DIR,256);CHKERRQ(ierr);
    }
    /* ierr = PetscStrcat(dir,"/java/client/jacc -display $DISPLAY");CHKERRQ(ierr); */
    ierr = PetscStrcat(dir,"/java/client/jacc");CHKERRQ(ierr);
    ierr = PetscPOpen(v->comm,machine,dir,"r",&fp);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ "ViewerAMSGetAMSComm_AMS"
int ViewerAMSGetAMSComm_AMS(Viewer lab,AMS_Comm *ams_comm)
{
  Viewer_AMS *vams = (Viewer_AMS *)lab->data;

  PetscFunctionBegin;
  if (vams->ams_comm == -1) SETERRQ(1,1,"AMS communicator name not yet set with ViewerAMSSetCommName()");
  *ams_comm = vams->ams_comm;
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNC__  
#define __FUNC__ "ViewerAMSSetCommName"
int ViewerAMSSetCommName(Viewer v,const char name[])
{
  int ierr,(*f)(Viewer,const char[]);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VIEWER_COOKIE);
  ierr = PetscObjectQueryFunction((PetscObject)v,"ViewerAMSSetCommName_C",(void **)&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(v,name);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "ViewerAMSGetAMSComm"
/*@C
    ViewerAMSGetAMSComm - Gets the AMS communicator associated with the viewer.

    Collective on MPI_Comm

    Input Parameters:
.   lab - the viewer

    Output Parameter:
.   ams_comm - the AMS communicator

    Level: developer

    Fortran Note:
    This routine is not supported in Fortran.

.keywords: Viewer, open, AMS memory snooper

.seealso: ViewerDestroy(), ViewerAMSOpen(), VIEWER_AMS_, VIEWER_AMS_WORLD, VIEWER_AMS_SELF

@*/
int ViewerAMSGetAMSComm(Viewer v,AMS_Comm *ams_comm)
{
  int ierr,(*f)(Viewer,AMS_Comm *);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VIEWER_COOKIE);
  ierr = PetscObjectQueryFunction((PetscObject)v,"ViewerAMSGetAMSComm_C",(void **)&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(v,ams_comm);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------------*/
Viewer VIEWER_AMS_WORLD_PRIVATE = 0;

#undef __FUNC__  
#define __FUNC__ "ViewerInitializeAMSWorld_Private"
int ViewerInitializeAMSWorld_Private(void)
{
  int  ierr;

  PetscFunctionBegin;
  if (VIEWER_AMS_WORLD_PRIVATE) PetscFunctionReturn(0);
  ierr = ViewerAMSOpen(PETSC_COMM_WORLD,"PETSc",&VIEWER_AMS_WORLD_PRIVATE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
    The variable Petsc_Viewer_Ams_keyval is used to indicate an MPI attribute that
  is attached to a communicator, in this case the attribute is a Viewer.
*/
static int Petsc_Viewer_Ams_keyval = MPI_KEYVAL_INVALID;

#undef __FUNC__  
#define __FUNC__ "VIEWER_AMS_" 
/*@C
     VIEWER_AMS_ - Creates an AMS memory snooper viewer shared by all processors 
                   in a communicator.

     Collective on MPI_Comm

     Input Parameters:
.    comm - the MPI communicator to share the viewer

     Level: developer

     Notes:
     Unlike almost all other PETSc routines, VIEWER_AMS_ does not return 
     an error code.  The window viewer is usually used in the form
$       XXXView(XXX object,VIEWER_AMS_(comm));

.seealso: VIEWER_AMS_WORLD, VIEWER_AMS_SELF, ViewerAMSOpen(), 
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
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);
  if (ierr) {PetscError(__LINE__,"VIEWER_AMS_",__FILE__,__SDIR__,1,1,0); viewer = 0;}
  ierr = MPI_Comm_size(comm,&csize);
  if (ierr) {PetscError(__LINE__,"VIEWER_AMS_",__FILE__,__SDIR__,1,1,0); viewer = 0;}
  if (size == csize) {
    viewer = VIEWER_AMS_WORLD;
    PetscFunctionReturn(viewer);
  }

  if (Petsc_Viewer_Ams_keyval == MPI_KEYVAL_INVALID) {
    ierr = MPI_Keyval_create(MPI_NULL_COPY_FN,MPI_NULL_DELETE_FN,&Petsc_Viewer_Ams_keyval,0);
    if (ierr) {PetscError(__LINE__,"VIEWER_AMS_",__FILE__,__SDIR__,1,1,0); viewer = 0;}
  }
  ierr = MPI_Attr_get(comm,Petsc_Viewer_Ams_keyval,(void **)&viewer,&flag);
  if (ierr) {PetscError(__LINE__,"VIEWER_AMS_",__FILE__,__SDIR__,1,1,0); viewer = 0;}
  if (!flag) { /* viewer not yet created */
    if (csize == 1) {
      ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);if (ierr) PetscFunctionReturn(0);
      sprintf(name,"PETSc_%d",rank);
    } else {
      PetscError(__LINE__,"VIEWER_AMS_",__FILE__,__SDIR__,1,1,0); viewer = 0;
    }
    ierr = ViewerAMSOpen(comm,name,&viewer); 
    if (ierr) {PetscError(__LINE__,"VIEWER_AMS_",__FILE__,__SDIR__,1,1,0); viewer = 0;}
    ierr = MPI_Attr_put(comm,Petsc_Viewer_Ams_keyval,(void*)viewer);
    if (ierr) {PetscError(__LINE__,"VIEWER_AMS_",__FILE__,__SDIR__,1,1,0); viewer = 0;}
  } 
  PetscFunctionReturn(viewer);
}

/*
       If there is a Viewer associated with this communicator, it is destroyed.
*/
#undef __FUNC__  
#define __FUNC__ "VIEWER_AMS_Destroy"
int VIEWER_AMS_Destroy(MPI_Comm comm)
{
  int    ierr,flag;
  Viewer viewer;

  PetscFunctionBegin;
  if (Petsc_Viewer_Ams_keyval == MPI_KEYVAL_INVALID) {
    PetscFunctionReturn(0);
  }
  ierr = MPI_Attr_get(comm,Petsc_Viewer_Ams_keyval,(void **)&viewer,&flag);CHKERRQ(ierr);
  if (flag) { 
    ierr = ViewerDestroy(viewer);CHKERRQ(ierr);
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
  if (VIEWER_AMS_WORLD_PRIVATE) {
    ierr = ViewerDestroy(VIEWER_AMS_WORLD_PRIVATE);CHKERRQ(ierr);
  }
  ierr = VIEWER_AMS_Destroy(PETSC_COMM_SELF);CHKERRQ(ierr);
  ierr = VIEWER_AMS_Destroy(PETSC_COMM_WORLD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "ViewerDestroy_AMS"
static int ViewerDestroy_AMS(Viewer viewer)
{
  Viewer_AMS *vams = (Viewer_AMS*)viewer->data;
  int        ierr;

  PetscFunctionBegin;

  ierr = AMS_Comm_destroy(vams->ams_comm);
  if (ierr) {
    char *err;
    AMS_Explain_error(ierr,&err);
    SETERRQ(ierr,0,err);
  }
  ierr = PetscFree(vams);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ "ViewerCreate_AMS"
int ViewerCreate_AMS(Viewer v)
{
  Viewer_AMS *vams;
  int        ierr;

  PetscFunctionBegin;
  v->ops->destroy = ViewerDestroy_AMS;
  ierr            = PetscStrallocpy(AMS_VIEWER,&v->type_name);CHKERRQ(ierr);
  vams            = PetscNew(Viewer_AMS);CHKPTRQ(vams);
  v->data         = (void*)vams;
  vams->ams_comm  = -1;
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)v,"ViewerAMSSetCommName_C",
                                    "ViewerAMSSetCommName_AMS",
                                     (void*)ViewerAMSSetCommName_AMS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)v,"ViewerAMSGetAMSComm_C",
                                    "ViewerAMSGetAMSComm_AMS",
                                     (void*)ViewerAMSGetAMSComm_AMS);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END


