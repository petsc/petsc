
#include <petsc-private/viewerimpl.h>  /*I "petscsys.h" I*/  

PetscFList PetscViewerList              = 0;

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

.seealso: PetscViewerDestroy(), PetscViewerSetType()

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

.seealso: PetscViewerCreate(), PetscViewerGetType()
@*/
PetscErrorCode  PetscViewerSetType(PetscViewer viewer,const PetscViewerType type)
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

.seealso: PetscViewerCreate(), PetscViewerSetType()

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
