
#include "src/sys/src/viewer/viewerimpl.h"
#include "petscsys.h"
#if defined(PETSC_HAVE_STDLIB_H)
#include <stdlib.h>
#endif

#include "ams.h"
typedef struct {
  char       *ams_name;
  AMS_Comm   ams_comm;
} PetscViewer_AMS;

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PetscViewerAMSSetCommName_AMS" 
int PetscViewerAMSSetCommName_AMS(PetscViewer v,const char name[])
{
  PetscViewer_AMS *vams = (PetscViewer_AMS*)v->data;
  int             ierr,port = -1;
  PetscTruth      flg,flg2;
  char            m[16],*pdir;

  PetscFunctionBegin;
  ierr = PetscOptionsGetInt(PETSC_NULL,"-ams_port",&port,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscLogInfo(v,"Publishing with the AMS on port %d\n",port);CHKERRQ(ierr);
  ierr = AMS_Comm_publish((char *)name,&vams->ams_comm,MPI_TYPE,v->comm,&port);CHKERRQ(ierr);

  ierr = PetscOptionsHasName(PETSC_NULL,"-ams_printf",&flg);CHKERRQ(ierr);
  if (!flg) {
#if !defined(PETSC_MISSING_DEV_NULL)
    ierr = AMS_Set_output_file("/dev/null");CHKERRQ(ierr);
#endif
  }

  ierr = PetscOptionsGetString(PETSC_NULL,"-ams_matlab",m,16,&flg);CHKERRQ(ierr);
  if (flg) {
    FILE *fp;
    ierr = PetscStartMatlab(v->comm,m,"petscview",&fp);CHKERRQ(ierr);
  }

  ierr = PetscOptionsGetString(PETSC_NULL,"-ams_java",m,16,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscOptionsHasName(PETSC_NULL,"-ams_publish_options",&flg2);CHKERRQ(ierr);
    if (flg2) {
      char cmd[PETSC_MAX_PATH_LEN];
      ierr = PetscStrcpy(cmd,"cd ");CHKERRQ(ierr);
      ierr = PetscGetPetscDir(&pdir);CHKERRQ(ierr);
      ierr = PetscStrcat(cmd,pdir);CHKERRQ(ierr);
      ierr = PetscStrcat(cmd,"/src/sys/src/objects/ams/java;make runamsoptions AMS_OPTIONS=\"-ams_server ${HOSTNAME}\"");CHKERRQ(ierr);
      ierr = PetscPOpen(v->comm,m,cmd,"r",PETSC_NULL);CHKERRQ(ierr);
    }

    ierr = PetscOptionsHasName(PETSC_NULL,"-ams_publish_objects",&flg2);CHKERRQ(ierr);
    if (flg2) {
      char dir[PETSC_MAX_PATH_LEN];
#if defined(PETSC_HAVE_UCBPS)
      char buf[PETSC_MAX_PATH_LEN],*found;
      FILE *fp;

      /* check if jacc is not already running */
      ierr  = PetscPOpen(v->comm,m,"/usr/ucb/ps -ugxww | grep jacc | grep -v grep","r",&fp);CHKERRQ(ierr);
      found = fgets(buf,1024,fp);
      ierr  = PetscFClose(v->comm,fp);CHKERRQ(ierr);
      if (found) PetscFunctionReturn(0);
#endif
      ierr = PetscOptionsGetenv(v->comm,"AMS_HOME",dir,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
      if (!flg) {
        ierr = PetscStrncpy(dir,AMS_HOME,PETSC_MAX_PATH_LEN);CHKERRQ(ierr);
      }
      /* ierr = PetscStrcat(dir,"/java/client/jacc -display ${DISPLAY}");CHKERRQ(ierr); */
      ierr = PetscStrcat(dir,"/java/client/jacc");CHKERRQ(ierr);
      ierr = PetscPOpen(v->comm,m,dir,"r",PETSC_NULL);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PetscViewerAMSGetAMSComm_AMS" 
int PetscViewerAMSGetAMSComm_AMS(PetscViewer lab,AMS_Comm *ams_comm)
{
  PetscViewer_AMS *vams = (PetscViewer_AMS *)lab->data;

  PetscFunctionBegin;
  if (vams->ams_comm == -1) SETERRQ(PETSC_ERR_ORDER,"AMS communicator name not yet set with PetscViewerAMSSetCommName()");
  *ams_comm = vams->ams_comm;
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "PetscViewerAMSSetCommName" 
int PetscViewerAMSSetCommName(PetscViewer v,const char name[])
{
  int ierr,(*f)(PetscViewer,const char[]);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,PETSC_VIEWER_COOKIE,1);
  PetscValidCharPointer(string,2);
  ierr = PetscObjectQueryFunction((PetscObject)v,"PetscViewerAMSSetCommName_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(v,name);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscViewerAMSGetAMSComm" 
/*@C
    PetscViewerAMSGetAMSComm - Gets the AMS communicator associated with the PetscViewer.

    Collective on MPI_Comm

    Input Parameters:
.   lab - the PetscViewer

    Output Parameter:
.   ams_comm - the AMS communicator

    Level: developer

    Fortran Note:
    This routine is not supported in Fortran.

  Concepts: publishing variables
  Concepts: AMS^getting communicator
  Concepts: communicator^accessing AMS communicator

.seealso: PetscViewerDestroy(), PetscViewerAMSOpen(), PetscViewer_AMS_, PetscViewer_AMS_WORLD, PetscViewer_AMS_SELF

@*/
int PetscViewerAMSGetAMSComm(PetscViewer v,AMS_Comm *ams_comm)
{
  int ierr,(*f)(PetscViewer,AMS_Comm *);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,PETSC_VIEWER_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)v,"PetscViewerAMSGetAMSComm_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(v,ams_comm);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*
    The variable Petsc_Viewer_Ams_keyval is used to indicate an MPI attribute that
  is attached to a communicator, in this case the attribute is a PetscViewer.
*/
static int Petsc_Viewer_Ams_keyval = MPI_KEYVAL_INVALID;

#undef __FUNCT__  
#define __FUNCT__ "PETSC_VIEWER_AMS_" 
/*@C
     PetscViewer_AMS_ - Creates an AMS memory snooper PetscViewer shared by all processors 
                   in a communicator.

     Collective on MPI_Comm

     Input Parameters:
.    comm - the MPI communicator to share the PetscViewer

     Level: developer

     Notes:
     Unlike almost all other PETSc routines, PetscViewer_AMS_ does not return 
     an error code.  The window PetscViewer is usually used in the form
$       XXXView(XXX object,PETSC_VIEWER_AMS_(comm));

.seealso: PetscViewer_AMS_WORLD, PetscViewer_AMS_SELF, PetscViewerAMSOpen(), 
@*/
PetscViewer PETSC_VIEWER_AMS_(MPI_Comm comm)
{
  int         ierr,flag,size,rank;
  PetscViewer viewer;
  char        name[256];

  PetscFunctionBegin;
  if (Petsc_Viewer_Ams_keyval == MPI_KEYVAL_INVALID) {
    ierr = MPI_Keyval_create(MPI_NULL_COPY_FN,MPI_NULL_DELETE_FN,&Petsc_Viewer_Ams_keyval,0);
    if (ierr) {PetscError(__LINE__,"PETSC_VIEWER_AMS_",__FILE__,__SDIR__,1,1," "); viewer = 0;}
  }
  ierr = MPI_Attr_get(comm,Petsc_Viewer_Ams_keyval,(void **)&viewer,&flag);
  if (ierr) {PetscError(__LINE__,"PETSC_VIEWER_AMS_",__FILE__,__SDIR__,1,1," "); viewer = 0;}
  if (!flag) { /* PetscViewer not yet created */
    if (comm == PETSC_COMM_WORLD) {
      ierr = PetscStrcpy(name,"PETSc");
      if (ierr) {PetscError(__LINE__,"PETSC_VIEWER_AMS_",__FILE__,__SDIR__,1,1," "); viewer = 0;}
    } else {
      ierr = MPI_Comm_size(comm,&size);
      if (ierr) {PetscError(__LINE__,"PETSC_VIEWER_AMS_",__FILE__,__SDIR__,1,1," "); viewer = 0;}
      if (size == 1) {
        ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);if (ierr) PetscFunctionReturn(0);
        sprintf(name,"PETSc_%d",rank);
      } else {
        PetscError(__LINE__,"PETSC_VIEWER_AMS_",__FILE__,__SDIR__,1,1," "); viewer = 0;
      } 
    }
    ierr = PetscViewerAMSOpen(comm,name,&viewer); 
    if (ierr) {PetscError(__LINE__,"PETSC_VIEWER_AMS_",__FILE__,__SDIR__,1,1," "); viewer = 0;}
    ierr = PetscObjectRegisterDestroy((PetscObject)viewer);
    if (ierr) {PetscError(__LINE__,"PETSC_VIEWER_STDOUT_",__FILE__,__SDIR__,1,1," "); viewer = 0;}
    ierr = MPI_Attr_put(comm,Petsc_Viewer_Ams_keyval,(void*)viewer);
    if (ierr) {PetscError(__LINE__,"PETSC_VIEWER_AMS_",__FILE__,__SDIR__,1,1," "); viewer = 0;}
  } 
  PetscFunctionReturn(viewer);
}

/*
       If there is a PetscViewer associated with this communicator, it is destroyed.
*/
#undef __FUNCT__  
#define __FUNCT__ "PETSC_VIEWER_AMS_Destroy" 
int PetscViewer_AMS_Destroy(MPI_Comm comm)
{
  int         ierr,flag;
  PetscViewer viewer;

  PetscFunctionBegin;
  if (Petsc_Viewer_Ams_keyval == MPI_KEYVAL_INVALID) {
    PetscFunctionReturn(0);
  }
  ierr = MPI_Attr_get(comm,Petsc_Viewer_Ams_keyval,(void **)&viewer,&flag);CHKERRQ(ierr);
  if (flag) { 
    ierr = PetscViewerDestroy(viewer);CHKERRQ(ierr);
    ierr = MPI_Attr_delete(comm,Petsc_Viewer_Ams_keyval);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscViewerDestroy_AMS" 
static int PetscViewerDestroy_AMS(PetscViewer viewer)
{
  PetscViewer_AMS *vams = (PetscViewer_AMS*)viewer->data;
  int             ierr;

  PetscFunctionBegin;

  /*
     Make sure that we mark that the stack is no longer published
  */
  if (viewer->comm == PETSC_COMM_WORLD) {
    ierr = PetscStackDepublish();CHKERRQ(ierr);
  }

  ierr = AMS_Comm_destroy(vams->ams_comm);
  if (ierr) {
    char *err;
    AMS_Explain_error(ierr,&err);
    SETERRQ(ierr,err);
  }
  ierr = PetscFree(vams);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PetscViewerCreate_AMS" 
int PetscViewerCreate_AMS(PetscViewer v)
{
  PetscViewer_AMS *vams;
  int             ierr;

  PetscFunctionBegin;
  v->ops->destroy = PetscViewerDestroy_AMS;
  ierr            = PetscStrallocpy(PETSC_VIEWER_AMS,&v->type_name);CHKERRQ(ierr);
  ierr            = PetscNew(PetscViewer_AMS,&vams);CHKERRQ(ierr);
  v->data         = (void*)vams;
  vams->ams_comm  = -1;
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)v,"PetscViewerAMSSetCommName_C",
                                    "PetscViewerAMSSetCommName_AMS",
                                     PetscViewerAMSSetCommName_AMS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)v,"PetscViewerAMSGetAMSComm_C",
                                    "PetscViewerAMSGetAMSComm_AMS",
                                     PetscViewerAMSGetAMSComm_AMS);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END


