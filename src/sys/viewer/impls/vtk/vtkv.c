#include "../src/sys/viewer/impls/vtk/vtkvimpl.h" /*I "petscviewer.h" I*/

#undef __FUNCT__
#define __FUNCT__ "PetscViewerVTKAddField"
/*@C
   PetscViewerVTKAddField - Add a field to the viewer

   Collective

   Input Arguments:

   Output Arguments:

   Level: developer

.seealso:
@*/
PetscErrorCode PetscViewerVTKAddField(PetscViewer viewer,PetscObject dm,PetscViewerVTKWriteFunction func,PetscObject vec)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,1);
  PetscValidHeader(dm,2);
  PetscValidHeader(vec,4);
  ierr = PetscUseMethod(viewer,"PetscViewerVTKAddField_C",(PetscViewer,PetscObject,PetscViewerVTKWriteFunction,PetscObject),(viewer,dm,func,vec));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscViewerDestroy_VTK"
static PetscErrorCode PetscViewerDestroy_VTK(PetscViewer viewer)
{
 PetscViewer_VTK *vtk = (PetscViewer_VTK*)viewer->data;
 PetscErrorCode  ierr;

 PetscFunctionBegin;
 ierr = PetscFree(vtk->filename);CHKERRQ(ierr);
 ierr = PetscFree(vtk);CHKERRQ(ierr);
 ierr = PetscObjectComposeFunctionDynamic((PetscObject)viewer,"PetscViewerFileSetName_C","",PETSC_NULL);CHKERRQ(ierr);
 ierr = PetscObjectComposeFunctionDynamic((PetscObject)viewer,"PetscViewerFileSetMode_C","",PETSC_NULL);CHKERRQ(ierr);
 ierr = PetscObjectComposeFunctionDynamic((PetscObject)viewer,"PetscViewerVTKAddField_C","",PETSC_NULL);CHKERRQ(ierr);
 PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscViewerFlush_VTK"
static PetscErrorCode PetscViewerFlush_VTK(PetscViewer viewer)
{
  PetscViewer_VTK *vtk = (PetscViewer_VTK*)viewer->data;
  PetscErrorCode ierr;
  PetscViewerVTKObjectLink   link,next;

  PetscFunctionBegin;
  if (vtk->link && (!vtk->dm || !vtk->dmwriteall)) SETERRQ(((PetscObject)viewer)->comm,PETSC_ERR_ARG_WRONGSTATE,"No fields or no grid");
  ierr = (*vtk->dmwriteall)(vtk->dm,viewer);CHKERRQ(ierr);
  for (link=vtk->link; link; link=next) {
    next = link->next;
    ierr = PetscObjectDestroy(&link->vec);CHKERRQ(ierr);
    ierr = PetscFree(link);CHKERRQ(ierr);
  }
  ierr = PetscObjectDestroy(&vtk->dm);CHKERRQ(ierr);
  vtk->dmwriteall = PETSC_NULL;
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "PetscViewerFileSetName_VTK"
PetscErrorCode  PetscViewerFileSetName_VTK(PetscViewer viewer,const char name[])
{
  PetscViewer_VTK *vtk = (PetscViewer_VTK*)viewer->data;
  PetscErrorCode  ierr;
  PetscBool       isvts;
  size_t          len;

  PetscFunctionBegin;
  ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
  ierr = PetscFree(vtk->filename);CHKERRQ(ierr);
  ierr = PetscStrlen(name,&len);CHKERRQ(ierr);
  ierr = PetscStrcasecmp(name+len-4,".vts",&isvts);CHKERRQ(ierr);
  if (isvts) {
    if (viewer->format == PETSC_VIEWER_DEFAULT) {ierr = PetscViewerSetFormat(viewer,PETSC_VIEWER_VTK_VTS);CHKERRQ(ierr);}
    if (viewer->format != PETSC_VIEWER_VTK_VTS) SETERRQ2(((PetscObject)viewer)->comm,PETSC_ERR_ARG_INCOMP,"Cannot use file '%s' with format %s, should have '.vts' extension",name,PetscViewerFormats[viewer->format]);
  } else SETERRQ1(((PetscObject)viewer)->comm,PETSC_ERR_ARG_UNKNOWN_TYPE,"File '%s' has unrecognized extension",name);
  ierr = PetscStrallocpy(name,&vtk->filename);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "PetscViewerFileSetMode_VTK"
PetscErrorCode  PetscViewerFileSetMode_VTK(PetscViewer viewer,PetscFileMode type)
{
  PetscViewer_VTK *vtk = (PetscViewer_VTK*)viewer->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,1);
  vtk->btype = type;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "PetscViewerVTKAddField_VTK"
PetscErrorCode  PetscViewerVTKAddField_VTK(PetscViewer viewer,PetscObject dm,PetscViewerVTKWriteFunction dmwriteall,PetscObject vec)
{
  PetscViewer_VTK *vtk = (PetscViewer_VTK*)viewer->data;
  PetscViewerVTKObjectLink link;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (vtk->dm) {
    if (dm != vtk->dm) SETERRQ(((PetscObject)viewer)->comm,PETSC_ERR_ARG_INCOMP,"Cannot write a field from more than one grid to the same VTK file");
  }
  vtk->dm = dm;
  vtk->dmwriteall = dmwriteall;
  ierr = PetscMalloc(sizeof(struct _n_PetscViewerVTKObjectLink),&link);CHKERRQ(ierr);
  link->vec = vec;
  /* Prepend to list */
  link->next = vtk->link;
  vtk->link = link;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "PetscViewerCreate_VTK"
PetscErrorCode  PetscViewerCreate_VTK(PetscViewer v)
{
  PetscViewer_VTK *vtk;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscNewLog(v,PetscViewer_VTK,&vtk);CHKERRQ(ierr);

  v->data         = (void*)vtk;
  v->ops->destroy = PetscViewerDestroy_VTK;
  v->ops->flush   = PetscViewerFlush_VTK;
  v->iformat      = 0;
  vtk->btype     = (PetscFileMode) -1;
  vtk->filename  = 0;

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)v,"PetscViewerFileSetName_C","PetscViewerFileSetName_VTK",
                                           PetscViewerFileSetName_VTK);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)v,"PetscViewerFileSetMode_C","PetscViewerFileSetMode_VTK",
                                           PetscViewerFileSetMode_VTK);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)v,"PetscViewerVTKAddField_C","PetscViewerVTKAddField_C",
                                           PetscViewerVTKAddField_VTK);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__
#define __FUNCT__ "PetscViewerVTKOpen"
/*@C
   PetscViewerVTKOpen - Opens a file for VTK output.

   Collective on MPI_Comm

   Input Parameters:
+  comm - MPI communicator
.  name - name of file
-  type - type of file
$    FILE_MODE_WRITE - create new file for binary output
$    FILE_MODE_READ - open existing file for binary input (not currently supported)
$    FILE_MODE_APPEND - open existing file for binary output (not currently supported)

   Output Parameter:
.  vtk - PetscViewer for VTK input/output to use with the specified file

   Level: beginner

   Note:
   This PetscViewer should be destroyed with PetscViewerDestroy().

   Concepts: VTK files
   Concepts: PetscViewer^creating

.seealso: PetscViewerASCIIOpen(), PetscViewerSetFormat(), PetscViewerDestroy(),
          VecView(), MatView(), VecLoad(), MatLoad(),
          PetscFileMode, PetscViewer
@*/
PetscErrorCode PetscViewerVTKOpen(MPI_Comm comm,const char name[],PetscFileMode type,PetscViewer *vtk)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscViewerCreate(comm,vtk);CHKERRQ(ierr);
  ierr = PetscViewerSetType(*vtk,PETSCVIEWERVTK);CHKERRQ(ierr);
  ierr = PetscViewerFileSetMode(*vtk,type);CHKERRQ(ierr);
  ierr = PetscViewerFileSetName(*vtk,name);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
