
#include <petsc-private/viewerimpl.h>  /*I "petscsys.h" I*/

PetscFList PetscViewerList              = 0;

PetscErrorCode PetscOptionsFindPair_Private(const char[],const char[],char *[],PetscBool*);
#undef __FUNCT__
#define __FUNCT__ "PetscOptionsGetViewer"
/*@C
   PetscOptionsGetViewer - Gets a viewer appropriate for the type indicated by the user

   Collective on MPI_Comm

   Input Parameters:
+  comm - the communicator to own the viewer
.  pre - the string to prepend to the name or PETSC_NULL
-  name - the option one is seeking

   Output Parameter:
+  viewer - the viewer
-  set - PETSC_TRUE if found, else PETSC_FALSE

   Level: intermediate

   Notes: If no value is provided ascii:stdout is used
$       ascii[:[filename][:format]]   defaults to stdout - format can be one of info, info_detailed, or matlab, for example ascii::info prints just the info
$                                     about the object to standard out
$       binary[:filename]   defaults to binaryoutput
$       draw
$       socket[:port]    defaults to the standard output port

   Use PetscOptionsRestoreViewer() after using the viewer, otherwise a memory leak may occur

.seealso: PetscOptionsGetReal(), PetscOptionsHasName(), PetscOptionsGetString(),
          PetscOptionsGetIntArray(), PetscOptionsGetRealArray(), PetscOptionsBool()
          PetscOptionsInt(), PetscOptionsString(), PetscOptionsReal(), PetscOptionsBool(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsBoolGroupBegin(), PetscOptionsBoolGroup(), PetscOptionsBoolGroupEnd(),
          PetscOptionsList(), PetscOptionsEList(), PetscOptionsRestoreViewer()
@*/
PetscErrorCode  PetscOptionsGetViewer(MPI_Comm comm,const char pre[],const char name[],PetscViewer *viewer,PetscBool  *set)
{
  char           *value;
  PetscErrorCode ierr;
  PetscBool      flag;

  PetscFunctionBegin;
  PetscValidCharPointer(name,3);

  if (set) *set = PETSC_FALSE;
  ierr = PetscOptionsFindPair_Private(pre,name,&value,&flag);CHKERRQ(ierr);
  if (flag) {
    if (set) *set = PETSC_TRUE;
    if (!value) {
      ierr = PetscViewerASCIIGetStdout(comm,viewer);CHKERRQ(ierr);
    } else {
      char        *cvalue,*loc,*loc2 = PETSC_NULL;
      PetscInt    cnt;
      const char  *viewers[] = {PETSCVIEWERASCII,PETSCVIEWERBINARY,PETSCVIEWERDRAW,PETSCVIEWERSOCKET,0};

      ierr = PetscStrallocpy(value,&cvalue);CHKERRQ(ierr);
      ierr = PetscStrchr(cvalue,':',&loc);CHKERRQ(ierr);
      if (loc) {*loc = 0; loc++;}
      ierr = PetscStrendswithwhich(*cvalue ? cvalue : "ascii",viewers,&cnt);CHKERRQ(ierr);
      if (cnt == 4) SETERRQ1(comm,PETSC_ERR_ARG_OUTOFRANGE,"Unknown viewer type: %s",cvalue);
      if (!loc) {
        if (cnt == 0) {
          ierr = PetscViewerASCIIGetStdout(comm,viewer);CHKERRQ(ierr);
        }
        if (cnt == 1) {
          *viewer = PETSC_VIEWER_BINARY_(comm);CHKERRQ(ierr);
        }
        if (cnt == 2) {
          *viewer = PETSC_VIEWER_DRAW_(comm);CHKERRQ(ierr);
        }
        if (cnt == 3) {
          *viewer = PETSC_VIEWER_SOCKET_(comm);CHKERRQ(ierr);
        }
      } else {
        ierr = PetscStrchr(loc,':',&loc2);CHKERRQ(ierr);
        if (loc2) {*loc2 = 0; loc2++;}
        if (loc2 && !*loc) { /* ASCII format without file name */
          ierr = PetscViewerASCIIGetStdout(comm,viewer);CHKERRQ(ierr);
        } else {
          ierr = PetscViewerCreate(comm,viewer);CHKERRQ(ierr);
          ierr = PetscViewerSetType(*viewer,*cvalue ? cvalue : "ascii");CHKERRQ(ierr);
          ierr = PetscViewerFileSetMode(*viewer,FILE_MODE_WRITE);CHKERRQ(ierr);
          ierr = PetscViewerFileSetName(*viewer,loc);CHKERRQ(ierr);
          ierr = PetscObjectComposeFunction((PetscObject)*viewer,"PetscOptionsDestroyViewer","PetscViewerDestroy",(void (*)(void))PetscViewerDestroy);CHKERRQ(ierr);
        }
      }
      ierr = PetscFree(cvalue);CHKERRQ(ierr);
      ierr = PetscViewerSetUp(*viewer);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscOptionsRestoreViewer"
/*@C
   PetscOptionsRestoresViewer - Restores a viewer obtained with PetscOptionsGetViewer()

   Collective on PetscViewer

   Input Parameters:
.  viewer - the viewer

   Level: intermediate

.seealso: PetscOptionsGetReal(), PetscOptionsHasName(), PetscOptionsGetString(),
          PetscOptionsGetIntArray(), PetscOptionsGetRealArray(), PetscOptionsBool()
          PetscOptionsInt(), PetscOptionsString(), PetscOptionsReal(), PetscOptionsBool(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsBoolGroupBegin(), PetscOptionsBoolGroup(), PetscOptionsBoolGroupEnd(),
          PetscOptionsList(), PetscOptionsEList(), PetscOptionsRestoreViewer()
@*/
PetscErrorCode  PetscOptionsRestoreViewer(PetscViewer viewer)
{
  PetscErrorCode ierr,(*f)(PetscViewer*);

  PetscFunctionBegin;
  ierr = PetscObjectQueryFunction((PetscObject)viewer,"PetscOptionsDestroyViewer",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(&viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscViewerCreate"
/*@
   PetscViewerCreate - Creates a viewing context

   Collective on MPI_Comm

   Input Parameter:
.  comm - MPI communicator

   Output Parameter:
.  inviewer - location to put the PetscViewer context

   Level: advanced

   Concepts: graphics^creating PetscViewer
   Concepts: file input/output^creating PetscViewer
   Concepts: sockets^creating PetscViewer

.seealso: PetscViewerDestroy(), PetscViewerSetType(), PetscViewerType

@*/
PetscErrorCode  PetscViewerCreate(MPI_Comm comm,PetscViewer *inviewer)
{
  PetscViewer    viewer;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  *inviewer = 0;
#if !defined(PETSC_USE_DYNAMIC_LIBRARIES)
  ierr = PetscViewerInitializePackage(PETSC_NULL);CHKERRQ(ierr);
#endif
  ierr = PetscHeaderCreate(viewer,_p_PetscViewer,struct _PetscViewerOps,PETSC_VIEWER_CLASSID,-1,"PetscViewer","PetscViewer","Viewer",comm,PetscViewerDestroy,0);CHKERRQ(ierr);
  *inviewer           = viewer;
  viewer->data        = 0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscViewerSetType"
/*@C
   PetscViewerSetType - Builds PetscViewer for a particular implementation.

   Collective on PetscViewer

   Input Parameter:
+  viewer      - the PetscViewer context
-  type        - for example, "ASCII"

   Options Database Command:
.  -draw_type  <type> - Sets the type; use -help for a list
    of available methods (for instance, ascii)

   Level: advanced

   Notes:
   See "include/petscviewer.h" for available methods (for instance,
   PETSC_VIEWER_SOCKET)

.seealso: PetscViewerCreate(), PetscViewerGetType(), PetscViewerType
@*/
PetscErrorCode  PetscViewerSetType(PetscViewer viewer,PetscViewerType type)
{
  PetscErrorCode ierr,(*r)(PetscViewer);
  PetscBool      match;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,1);
  PetscValidCharPointer(type,2);
  CHKMEMQ;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,type,&match);CHKERRQ(ierr);
  if (match) PetscFunctionReturn(0);

  /* cleanup any old type that may be there */
  if (viewer->data) {
    ierr         = (*viewer->ops->destroy)(viewer);CHKERRQ(ierr);
    viewer->ops->destroy = PETSC_NULL;
    viewer->data = 0;
  }
  ierr = PetscMemzero(viewer->ops,sizeof(struct _PetscViewerOps));CHKERRQ(ierr);

  ierr =  PetscFListFind(PetscViewerList,((PetscObject)viewer)->comm,type,PETSC_TRUE,(void (**)(void)) &r);CHKERRQ(ierr);
  if (!r) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"Unknown PetscViewer type given: %s",type);

  ierr = PetscObjectChangeTypeName((PetscObject)viewer,type);CHKERRQ(ierr);
  ierr = (*r)(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscViewerRegisterDestroy"
/*@C
   PetscViewerRegisterDestroy - Frees the list of PetscViewer methods that were
   registered by PetscViewerRegisterDynamic().

   Not Collective

   Level: developer

.seealso: PetscViewerRegisterDynamic(), PetscViewerRegisterAll()
@*/
PetscErrorCode  PetscViewerRegisterDestroy(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFListDestroy(&PetscViewerList);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscViewerRegister"
PetscErrorCode  PetscViewerRegister(const char *sname,const char *path,const char *name,PetscErrorCode (*function)(PetscViewer))
{
  PetscErrorCode ierr;
  char fullname[PETSC_MAX_PATH_LEN];

  PetscFunctionBegin;
  ierr = PetscFListConcat(path,name,fullname);CHKERRQ(ierr);
  ierr = PetscFListAdd(&PetscViewerList,sname,fullname,(void (*)(void))function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscViewerSetFromOptions"
/*@C
   PetscViewerSetFromOptions - Sets the graphics type from the options database.
      Defaults to a PETSc X windows graphics.

   Collective on PetscViewer

   Input Parameter:
.     PetscViewer - the graphics context

   Level: intermediate

   Notes:
    Must be called after PetscViewerCreate() before the PetscViewer is used.

  Concepts: PetscViewer^setting options

.seealso: PetscViewerCreate(), PetscViewerSetType(), PetscViewerType

@*/
PetscErrorCode  PetscViewerSetFromOptions(PetscViewer viewer)
{
  PetscErrorCode ierr;
  char       vtype[256];
  PetscBool  flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,1);

  if (!PetscViewerList) {
    ierr = PetscViewerRegisterAll(PETSC_NULL);CHKERRQ(ierr);
  }
  ierr = PetscObjectOptionsBegin((PetscObject)viewer);CHKERRQ(ierr);
    ierr = PetscOptionsList("-viewer_type","Type of PetscViewer","None",PetscViewerList,(char *)(((PetscObject)viewer)->type_name?((PetscObject)viewer)->type_name:PETSCVIEWERASCII),vtype,256,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = PetscViewerSetType(viewer,vtype);CHKERRQ(ierr);
    }
    /* type has not been set? */
    if (!((PetscObject)viewer)->type_name) {
      ierr = PetscViewerSetType(viewer,PETSCVIEWERASCII);CHKERRQ(ierr);
    }
    if (viewer->ops->setfromoptions) {
      ierr = (*viewer->ops->setfromoptions)(viewer);CHKERRQ(ierr);
    }

    /* process any options handlers added with PetscObjectAddOptionsHandler() */
    ierr = PetscObjectProcessOptionsHandlers((PetscObject)viewer);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
