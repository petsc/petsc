#include <../src/sys/classes/viewer/impls/vtk/vtkvimpl.h> /*I "petscviewer.h" I*/

/*MC
    PetscViewerVTKWriteFunction - functional form used to provide writer to the PetscViewerVTK

     Synopsis:
     #include <petscviewer.h>
     PetscViewerVTKWriteFunction(PetscObject object,PetscViewer viewer)

     Input Parameters:
+      object - the PETSc object to be written
-      viewer - viewer it is to be written to

   Level: developer

.seealso:   PetscViewerVTKAddField()
M*/

/*@C
   PetscViewerVTKAddField - Add a field to the viewer

   Collective

   Input Parameters:
+ viewer - VTK viewer
. dm - DM on which Vec lives
. PetscViewerVTKWriteFunction - function to write this Vec
. fieldnum - which field of the DM to write (PETSC_DEFAULT if the whle vector should be written)
. fieldtype - Either PETSC_VTK_POINT_FIELD or PETSC_VTK_CELL_FIELD
. checkdm - whether to check for identical dm arguments as fields are added
- vec - Vec from which to write

   Note:
   This routine keeps exclusive ownership of the Vec. The caller should not use or destroy the Vec after adding it.

   Level: developer

.seealso: PetscViewerVTKOpen(), DMDAVTKWriteAll(), PetscViewerVTKWriteFunction, PetscViewerVTKGetDM()
@*/
PetscErrorCode PetscViewerVTKAddField(PetscViewer viewer,PetscObject dm,PetscErrorCode (*PetscViewerVTKWriteFunction)(PetscObject,PetscViewer),PetscInt fieldnum,PetscViewerVTKFieldType fieldtype,PetscBool checkdm,PetscObject vec)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,1);
  PetscValidHeader(dm,2);
  PetscValidHeader(vec,7);
  CHKERRQ(PetscUseMethod(viewer,"PetscViewerVTKAddField_C",(PetscViewer,PetscObject,PetscErrorCode (*)(PetscObject,PetscViewer),PetscInt,PetscViewerVTKFieldType,PetscBool,PetscObject),(viewer,dm,PetscViewerVTKWriteFunction,fieldnum,fieldtype,checkdm,vec)));
  PetscFunctionReturn(0);
}

/*@C
   PetscViewerVTKGetDM - get the DM associated with the viewer

   Collective

   Input Parameters:
+ viewer - VTK viewer
- dm - DM associated with the viewer (as PetscObject)

   Level: developer

.seealso: PetscViewerVTKOpen(), DMDAVTKWriteAll(), PetscViewerVTKWriteFunction, PetscViewerVTKAddField()
@*/
PetscErrorCode PetscViewerVTKGetDM(PetscViewer viewer,PetscObject *dm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,1);
  CHKERRQ(PetscUseMethod(viewer,"PetscViewerVTKGetDM_C",(PetscViewer,PetscObject*),(viewer,dm)));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscViewerDestroy_VTK(PetscViewer viewer)
{
  PetscViewer_VTK *vtk = (PetscViewer_VTK*)viewer->data;

  PetscFunctionBegin;
  CHKERRQ(PetscFree(vtk->filename));
  CHKERRQ(PetscFree(vtk));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)viewer,"PetscViewerFileSetName_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)viewer,"PetscViewerFileGetName_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)viewer,"PetscViewerFileSetMode_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)viewer,"PetscViewerFileGetMode_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)viewer,"PetscViewerVTKAddField_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)viewer,"PetscViewerVTKGetDM_C",NULL));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscViewerFlush_VTK(PetscViewer viewer)
{
  PetscViewer_VTK          *vtk = (PetscViewer_VTK*)viewer->data;
  PetscViewerVTKObjectLink link,next;

  PetscFunctionBegin;
  PetscCheckFalse(vtk->link && (!vtk->dm || !vtk->write),PetscObjectComm((PetscObject)viewer),PETSC_ERR_ARG_WRONGSTATE,"No fields or no grid");
  if (vtk->write) CHKERRQ((*vtk->write)(vtk->dm,viewer));
  for (link=vtk->link; link; link=next) {
    next = link->next;
    CHKERRQ(PetscObjectDestroy(&link->vec));
    CHKERRQ(PetscFree(link));
  }
  CHKERRQ(PetscObjectDestroy(&vtk->dm));
  vtk->write = NULL;
  vtk->link  = NULL;
  PetscFunctionReturn(0);
}

PetscErrorCode  PetscViewerFileSetName_VTK(PetscViewer viewer,const char name[])
{
  PetscViewer_VTK *vtk = (PetscViewer_VTK*)viewer->data;
  PetscBool       isvtk,isvts,isvtu,isvtr;
  size_t          len;

  PetscFunctionBegin;
  CHKERRQ(PetscViewerFlush(viewer));
  CHKERRQ(PetscFree(vtk->filename));
  CHKERRQ(PetscStrlen(name,&len));
  if (!len) {
    isvtk = PETSC_TRUE;
  } else {
    CHKERRQ(PetscStrcasecmp(name+len-4,".vtk",&isvtk));
    CHKERRQ(PetscStrcasecmp(name+len-4,".vts",&isvts));
    CHKERRQ(PetscStrcasecmp(name+len-4,".vtu",&isvtu));
    CHKERRQ(PetscStrcasecmp(name+len-4,".vtr",&isvtr));
  }
  if (isvtk) {
    if (viewer->format == PETSC_VIEWER_DEFAULT) viewer->format = PETSC_VIEWER_ASCII_VTK_DEPRECATED;
    PetscCheckFalse(viewer->format != PETSC_VIEWER_ASCII_VTK_DEPRECATED,PetscObjectComm((PetscObject)viewer),PETSC_ERR_ARG_INCOMP,"Cannot use file '%s' with format %s, should have '.vtk' extension",name,PetscViewerFormats[viewer->format]);
  } else if (isvts) {
    if (viewer->format == PETSC_VIEWER_DEFAULT) viewer->format = PETSC_VIEWER_VTK_VTS;
    PetscCheckFalse(viewer->format != PETSC_VIEWER_VTK_VTS,PetscObjectComm((PetscObject)viewer),PETSC_ERR_ARG_INCOMP,"Cannot use file '%s' with format %s, should have '.vts' extension",name,PetscViewerFormats[viewer->format]);
  } else if (isvtu) {
    if (viewer->format == PETSC_VIEWER_DEFAULT) viewer->format = PETSC_VIEWER_VTK_VTU;
    PetscCheckFalse(viewer->format != PETSC_VIEWER_VTK_VTU,PetscObjectComm((PetscObject)viewer),PETSC_ERR_ARG_INCOMP,"Cannot use file '%s' with format %s, should have '.vtu' extension",name,PetscViewerFormats[viewer->format]);
  } else if (isvtr) {
    if (viewer->format == PETSC_VIEWER_DEFAULT) viewer->format = PETSC_VIEWER_VTK_VTR;
    PetscCheckFalse(viewer->format != PETSC_VIEWER_VTK_VTR,PetscObjectComm((PetscObject)viewer),PETSC_ERR_ARG_INCOMP,"Cannot use file '%s' with format %s, should have '.vtr' extension",name,PetscViewerFormats[viewer->format]);
  } else SETERRQ(PetscObjectComm((PetscObject)viewer),PETSC_ERR_ARG_UNKNOWN_TYPE,"File '%s' has unrecognized extension",name);
  CHKERRQ(PetscStrallocpy(len ? name : "stdout",&vtk->filename));
  PetscFunctionReturn(0);
}

PetscErrorCode  PetscViewerFileGetName_VTK(PetscViewer viewer,const char **name)
{
  PetscViewer_VTK *vtk = (PetscViewer_VTK*)viewer->data;
  PetscFunctionBegin;
  *name = vtk->filename;
  PetscFunctionReturn(0);
}

PetscErrorCode  PetscViewerFileSetMode_VTK(PetscViewer viewer,PetscFileMode type)
{
  PetscViewer_VTK *vtk = (PetscViewer_VTK*)viewer->data;

  PetscFunctionBegin;
  vtk->btype = type;
  PetscFunctionReturn(0);
}

PetscErrorCode  PetscViewerFileGetMode_VTK(PetscViewer viewer,PetscFileMode *type)
{
  PetscViewer_VTK *vtk = (PetscViewer_VTK*)viewer->data;

  PetscFunctionBegin;
  *type = vtk->btype;
  PetscFunctionReturn(0);
}

PetscErrorCode  PetscViewerVTKAddField_VTK(PetscViewer viewer,PetscObject dm,PetscErrorCode (*PetscViewerVTKWriteFunction)(PetscObject,PetscViewer),PetscInt fieldnum,PetscViewerVTKFieldType fieldtype,PetscBool checkdm,PetscObject vec)
{
  PetscViewer_VTK          *vtk = (PetscViewer_VTK*)viewer->data;
  PetscViewerVTKObjectLink link, tail = vtk->link;

  PetscFunctionBegin;
  if (vtk->dm) {
    PetscCheckFalse(checkdm && dm != vtk->dm,PetscObjectComm((PetscObject)viewer),PETSC_ERR_ARG_INCOMP,"Refusing to write a field from more than one grid to the same VTK file. Set checkdm = PETSC_FALSE to skip this check.");
  } else {
    CHKERRQ(PetscObjectReference(dm));
    vtk->dm = dm;
  }
  vtk->write  = PetscViewerVTKWriteFunction;
  CHKERRQ(PetscNew(&link));
  link->ft    = fieldtype;
  link->vec   = vec;
  link->field = fieldnum;
  link->next  = NULL;
  /* Append to list */
  if (tail) {
    while (tail->next) tail = tail->next;
    tail->next = link;
  } else vtk->link = link;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscViewerVTKGetDM_VTK(PetscViewer viewer,PetscObject *dm)
{
  PetscViewer_VTK *vtk = (PetscViewer_VTK*)viewer->data;

  PetscFunctionBegin;
  *dm = vtk->dm;
  PetscFunctionReturn(0);
}

/*MC
   PETSCVIEWERVTK - A viewer that writes to a VTK file

.seealso:  PetscViewerVTKOpen(), PetscViewerHDF5Open(), PetscViewerStringSPrintf(), PetscViewerSocketOpen(), PetscViewerDrawOpen(), PETSCVIEWERSOCKET,
           PetscViewerCreate(), PetscViewerASCIIOpen(), PetscViewerBinaryOpen(), PETSCVIEWERBINARY, PETSCVIEWERDRAW, PETSCVIEWERSTRING,
           PetscViewerMatlabOpen(), VecView(), DMView(), PetscViewerMatlabPutArray(), PETSCVIEWERASCII, PETSCVIEWERMATLAB,
           PetscViewerFileSetName(), PetscViewerFileSetMode(), PetscViewerFormat, PetscViewerType, PetscViewerSetType()

  Level: beginner
M*/

PETSC_EXTERN PetscErrorCode PetscViewerCreate_VTK(PetscViewer v)
{
  PetscViewer_VTK *vtk;

  PetscFunctionBegin;
  CHKERRQ(PetscNewLog(v,&vtk));

  v->data         = (void*)vtk;
  v->ops->destroy = PetscViewerDestroy_VTK;
  v->ops->flush   = PetscViewerFlush_VTK;
  vtk->btype      = FILE_MODE_UNDEFINED;
  vtk->filename   = NULL;

  CHKERRQ(PetscObjectComposeFunction((PetscObject)v,"PetscViewerFileSetName_C",PetscViewerFileSetName_VTK));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)v,"PetscViewerFileGetName_C",PetscViewerFileGetName_VTK));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)v,"PetscViewerFileSetMode_C",PetscViewerFileSetMode_VTK));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)v,"PetscViewerFileGetMode_C",PetscViewerFileGetMode_VTK));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)v,"PetscViewerVTKAddField_C",PetscViewerVTKAddField_VTK));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)v,"PetscViewerVTKGetDM_C",PetscViewerVTKGetDM_VTK));
  PetscFunctionReturn(0);
}

/*@C
   PetscViewerVTKOpen - Opens a file for VTK output.

   Collective

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

.seealso: PetscViewerASCIIOpen(), PetscViewerPushFormat(), PetscViewerDestroy(),
          VecView(), MatView(), VecLoad(), MatLoad(),
          PetscFileMode, PetscViewer
@*/
PetscErrorCode PetscViewerVTKOpen(MPI_Comm comm,const char name[],PetscFileMode type,PetscViewer *vtk)
{
  PetscFunctionBegin;
  CHKERRQ(PetscViewerCreate(comm,vtk));
  CHKERRQ(PetscViewerSetType(*vtk,PETSCVIEWERVTK));
  CHKERRQ(PetscViewerFileSetMode(*vtk,type));
  CHKERRQ(PetscViewerFileSetName(*vtk,name));
  PetscFunctionReturn(0);
}

/*@C
   PetscViewerVTKFWrite - write binary data preceded by 32-bit int length (in bytes), does not do byte swapping.

   Logically collective on PetscViewer

   Input Parameters:
+  viewer - logically collective viewer, data written from rank 0
.  fp - file pointer valid on rank 0
.  data - data pointer valid on rank 0
.  n - number of data items
-  dtype - data type

   Level: developer

   Notes:
    If PetscScalar is __float128 then the binary files are written in double precision

.seealso: DMDAVTKWriteAll(), DMPlexVTKWriteAll(), PetscViewerPushFormat(), PetscViewerVTKOpen(), PetscBinaryWrite()
@*/
PetscErrorCode PetscViewerVTKFWrite(PetscViewer viewer,FILE *fp,const void *data,PetscInt n,MPI_Datatype dtype)
{
  PetscMPIInt    rank;
  MPI_Datatype   vdtype=dtype;
#if defined(PETSC_USE_REAL___FLOAT128)
  double         *tmp;
  PetscInt       i;
  PetscReal      *ttmp = (PetscReal*)data;
#endif

  PetscFunctionBegin;
  PetscCheckFalse(n < 0,PetscObjectComm((PetscObject)viewer),PETSC_ERR_ARG_OUTOFRANGE,"Trying to write a negative amount of data %" PetscInt_FMT,n);
  if (!n) PetscFunctionReturn(0);
  CHKERRMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)viewer),&rank));
  if (rank == 0) {
    size_t      count;
    PetscMPIInt dsize;
    PetscVTKInt bytes;

#if defined(PETSC_USE_REAL___FLOAT128)
    if (dtype == MPIU___FLOAT128) {
      CHKERRQ(PetscMalloc1(n,&tmp));
      for (i=0; i<n; i++) tmp[i] = ttmp[i];
      data  = (void*) tmp;
      vdtype = MPI_DOUBLE;
    }
#endif
    CHKERRMPI(MPI_Type_size(vdtype,&dsize));
    bytes = PetscVTKIntCast(dsize*n);

    count = fwrite(&bytes,sizeof(int),1,fp);
    PetscCheckFalse(count != 1,PETSC_COMM_SELF,PETSC_ERR_FILE_WRITE,"Error writing byte count");
    count = fwrite(data,dsize,(size_t)n,fp);
    PetscCheckFalse((PetscInt)count != n,PETSC_COMM_SELF,PETSC_ERR_FILE_WRITE,"Wrote %" PetscInt_FMT "/%" PetscInt_FMT " array members of size %d",(PetscInt)count,n,dsize);
#if defined(PETSC_USE_REAL___FLOAT128)
    if (dtype == MPIU___FLOAT128) {
      CHKERRQ(PetscFree(tmp));
    }
#endif
  }
  PetscFunctionReturn(0);
}
