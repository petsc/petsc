#include <petscdm.h>
#include <petscdmda.h>
#include <petscdmswarm.h>
#include <petsc/private/dmswarmimpl.h>
#include "../src/dm/impls/swarm/data_bucket.h"

PetscErrorCode private_PetscViewerCreate_XDMF(MPI_Comm comm,const char filename[],PetscViewer *v)
{
  long int       *bytes;
  PetscContainer container;
  PetscViewer    viewer;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  ierr = PetscViewerCreate(comm,&viewer);CHKERRQ(ierr);
  ierr = PetscViewerSetType(viewer,PETSCVIEWERASCII);CHKERRQ(ierr);
  ierr = PetscViewerFileSetMode(viewer,FILE_MODE_WRITE);CHKERRQ(ierr);
  ierr = PetscViewerFileSetName(viewer,filename);CHKERRQ(ierr);
  
  ierr = PetscMalloc1(1,&bytes);CHKERRQ(ierr);
  bytes[0] = 0;
  ierr = PetscContainerCreate(comm,&container);CHKERRQ(ierr);
  ierr = PetscContainerSetPointer(container,(void*)bytes);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)viewer,"XDMFViewerContext",(PetscObject)container);CHKERRQ(ierr);
  
  /* write xdmf header */
  ierr = PetscViewerASCIIPrintf(viewer,"<?xml version=\"1.0\" encoding=\"utf-8\"?>\n");CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"<Xdmf xmlns:xi=\"http://www.w3.org/2001/XInclude/\" Version=\"2.99\">\n");CHKERRQ(ierr);
  /* write xdmf domain */
  ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"<Domain>\n");CHKERRQ(ierr);
  *v = viewer;
  PetscFunctionReturn(0);
}

PetscErrorCode private_PetscViewerDestroy_XDMF(PetscViewer *v)
{
  PetscViewer    viewer;
  DM             dm = NULL;
  long int       *bytes;
  PetscContainer container = NULL;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  if (!v) PetscFunctionReturn(0);
  viewer = *v;
  
  ierr = PetscObjectQuery((PetscObject)viewer,"DMSwarm",(PetscObject*)&dm);CHKERRQ(ierr);
  if (dm) {
    ierr = PetscViewerASCIIPrintf(viewer,"</Grid>\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  }
  
  /* close xdmf header */
  ierr = PetscViewerASCIIPrintf(viewer,"</Domain>\n");CHKERRQ(ierr);
  ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"</Xdmf>\n");CHKERRQ(ierr);
  
  ierr = PetscObjectQuery((PetscObject)viewer,"XDMFViewerContext",(PetscObject*)&container);CHKERRQ(ierr);
  if (container) {
    ierr = PetscContainerGetPointer(container,(void**)&bytes);CHKERRQ(ierr);
    ierr = PetscFree(bytes);CHKERRQ(ierr);
    ierr = PetscContainerDestroy(&container);CHKERRQ(ierr);
  }
  
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  *v = NULL;
  PetscFunctionReturn(0);
}

PetscErrorCode private_CreateDataFileNameXDMF(const char filename[],char dfilename[])
{
  char           *ext;
  PetscBool      flg;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  ierr = PetscStrrchr(filename,'.',&ext);CHKERRQ(ierr);
  ierr = PetscStrcmp("xmf",ext,&flg);CHKERRQ(ierr);
  if (flg) {
    size_t len;
    char    viewername_minus_ext[PETSC_MAX_PATH_LEN];
    
    ierr = PetscStrlen(filename,&len);CHKERRQ(ierr);
    ierr = PetscStrncpy(viewername_minus_ext,filename,len-2);CHKERRQ(ierr);
    ierr = PetscSNPrintf(dfilename,PETSC_MAX_PATH_LEN-1,"%s_swarm_fields.pbin",viewername_minus_ext);CHKERRQ(ierr);
  } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"File extension must by .xmf");

  PetscFunctionReturn(0);
}

PetscErrorCode private_DMSwarmView_XDMF(DM dm,PetscViewer viewer)
{
  PetscBool      isswarm = PETSC_FALSE;
  const char     *viewername;
  char           datafile[PETSC_MAX_PATH_LEN];
  PetscViewer    fviewer;
  PetscInt       k,ng,dim;
  Vec            dvec;
  long int       *bytes = NULL;
  PetscContainer container = NULL;
  const char     *dmname;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  ierr = PetscObjectQuery((PetscObject)viewer,"XDMFViewerContext",(PetscObject*)&container);CHKERRQ(ierr);
  if (container) {
    ierr = PetscContainerGetPointer(container,(void**)&bytes);CHKERRQ(ierr);
  } else SETERRQ(PetscObjectComm((PetscObject)viewer),PETSC_ERR_SUP,"Valid to find attached data XDMFViewerContext");
  
  ierr = PetscObjectTypeCompare((PetscObject)dm,DMSWARM,&isswarm);CHKERRQ(ierr);
  if (!isswarm) SETERRQ(PetscObjectComm((PetscObject)viewer),PETSC_ERR_SUP,"Only valid for DMSwarm");
  
  ierr = PetscObjectCompose((PetscObject)viewer,"DMSwarm",(PetscObject)dm);CHKERRQ(ierr);
  
  ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
  ierr = PetscObjectGetName((PetscObject)dm,&dmname);CHKERRQ(ierr);
  if (!dmname) {
    ierr = DMGetOptionsPrefix(dm,&dmname);CHKERRQ(ierr);
  }
  if (!dmname) {
    ierr = PetscViewerASCIIPrintf(viewer,"<Grid Name=\"DMSwarm\" GridType=\"Uniform\">\n");CHKERRQ(ierr);
  } else {
    ierr = PetscViewerASCIIPrintf(viewer,"<Grid Name=\"DMSwarm[%s]\" GridType=\"Uniform\">\n",dmname);CHKERRQ(ierr);
  }
  
  /* create a sub-viewer for topology, geometry and all data fields */
  /* name is viewer.name + "_swarm_fields.pbin" */
  ierr = PetscViewerCreate(PetscObjectComm((PetscObject)viewer),&fviewer);CHKERRQ(ierr);
  ierr = PetscViewerSetType(fviewer,PETSCVIEWERBINARY);CHKERRQ(ierr);
  ierr = PetscViewerBinarySetSkipHeader(fviewer,PETSC_TRUE);CHKERRQ(ierr);
  ierr = PetscViewerBinarySetSkipInfo(fviewer,PETSC_TRUE);CHKERRQ(ierr);
  ierr = PetscViewerFileSetMode(fviewer,FILE_MODE_WRITE);CHKERRQ(ierr);
  
  ierr = PetscViewerFileGetName(viewer,&viewername);CHKERRQ(ierr);
  ierr = private_CreateDataFileNameXDMF(viewername,datafile);CHKERRQ(ierr);
  ierr = PetscViewerFileSetName(fviewer,datafile);CHKERRQ(ierr);
  
  ierr = DMSwarmGetSize(dm,&ng);CHKERRQ(ierr);
  
  /* write topology header */
  ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"<Topology Dimensions=\"%D\" TopologyType=\"Mixed\">\n",ng);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"<DataItem Format=\"Binary\" Endian=\"Big\" DataType=\"Int\" Dimensions=\"%D\" Seek=\"%D\">\n",ng*3,bytes[0]);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"%s\n",datafile);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"</DataItem>\n");CHKERRQ(ierr);
  ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"</Topology>\n");CHKERRQ(ierr);
  ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  
  /* write topology data */
  for (k=0; k<ng; k++) {
    PetscInt pvertex[3];
    
    pvertex[0] = 1;
    pvertex[1] = 1;
    pvertex[2] = k;
    ierr = PetscViewerBinaryWrite(fviewer,pvertex,3,PETSC_INT);CHKERRQ(ierr);
  }
  bytes[0] += sizeof(PetscInt) * ng * 3;
  
  /* write geometry header */
  ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
  switch (dim) {
    case 1:
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"No support for 1D");
      break;
    case 2:
      ierr = PetscViewerASCIIPrintf(viewer,"<Geometry Type=\"XY\">\n");CHKERRQ(ierr);
      break;
    case 3:
      ierr = PetscViewerASCIIPrintf(viewer,"<Geometry Type=\"XYZ\">\n");CHKERRQ(ierr);
      break;
  }
  ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"<DataItem Format=\"Binary\" Endian=\"Big\" DataType=\"Float\" Precision=\"8\" Dimensions=\"%D %D\" Seek=\"%D\">\n",ng,dim,bytes[0]);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"%s\n",datafile);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"</DataItem>\n");CHKERRQ(ierr);
  ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"</Geometry>\n");CHKERRQ(ierr);
  ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  
  /* write geometry data */
  ierr = DMSwarmCreateGlobalVectorFromField(dm,DMSwarmPICField_coor,&dvec);CHKERRQ(ierr);
  ierr = VecView(dvec,fviewer);CHKERRQ(ierr);
  ierr = DMSwarmDestroyGlobalVectorFromField(dm,DMSwarmPICField_coor,&dvec);CHKERRQ(ierr);
  bytes[0] += sizeof(PetscReal) * ng * dim;
  
  ierr = PetscViewerDestroy(&fviewer);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

PetscErrorCode private_VecView_Swarm_XDMF(Vec x,PetscViewer viewer)
{
  long int       *bytes = NULL;
  PetscContainer container = NULL;
  const char     *viewername;
  char           datafile[PETSC_MAX_PATH_LEN];
  PetscViewer    fviewer;
  PetscInt       N,bs;
  const char     *vecname;
  char           fieldname[PETSC_MAX_PATH_LEN];
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  ierr = PetscObjectQuery((PetscObject)viewer,"XDMFViewerContext",(PetscObject*)&container);CHKERRQ(ierr);
  if (container) {
    ierr = PetscContainerGetPointer(container,(void**)&bytes);CHKERRQ(ierr);
  } else SETERRQ(PetscObjectComm((PetscObject)viewer),PETSC_ERR_SUP,"Valid to find attached data XDMFViewerContext");
  
  ierr = PetscViewerFileGetName(viewer,&viewername);CHKERRQ(ierr);
  ierr = private_CreateDataFileNameXDMF(viewername,datafile);CHKERRQ(ierr);
  
  /* re-open a sub-viewer for all data fields */
  /* name is viewer.name + "_swarm_fields.pbin" */
  ierr = PetscViewerCreate(PetscObjectComm((PetscObject)viewer),&fviewer);CHKERRQ(ierr);
  ierr = PetscViewerSetType(fviewer,PETSCVIEWERBINARY);CHKERRQ(ierr);
  ierr = PetscViewerBinarySetSkipHeader(fviewer,PETSC_TRUE);CHKERRQ(ierr);
  ierr = PetscViewerBinarySetSkipInfo(fviewer,PETSC_TRUE);CHKERRQ(ierr);
  ierr = PetscViewerFileSetMode(fviewer,FILE_MODE_APPEND);CHKERRQ(ierr);
  ierr = PetscViewerFileSetName(fviewer,datafile);CHKERRQ(ierr);
  
  ierr = VecGetSize(x,&N);CHKERRQ(ierr);
  ierr = VecGetBlockSize(x,&bs);CHKERRQ(ierr);
  N = N/bs;
  ierr = PetscObjectGetName((PetscObject)x,&vecname);CHKERRQ(ierr);
  if (!vecname) {
    ierr = PetscSNPrintf(fieldname,PETSC_MAX_PATH_LEN-1,"swarmfield_%D",((PetscObject)x)->tag);CHKERRQ(ierr);
  } else {
    ierr = PetscSNPrintf(fieldname,PETSC_MAX_PATH_LEN-1,"%s",vecname);CHKERRQ(ierr);
  }
  
  /* write data header */
  ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"<Attribute Center=\"Node\" Name=\"%s\" Type=\"None\">\n",fieldname);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
  if (bs == 1) {
    ierr = PetscViewerASCIIPrintf(viewer,"<DataItem Format=\"Binary\" Endian=\"Big\" DataType=\"Float\" Precision=\"8\" Dimensions=\"%D\" Seek=\"%D\">\n",N,bytes[0]);CHKERRQ(ierr);
  } else {
    ierr = PetscViewerASCIIPrintf(viewer,"<DataItem Format=\"Binary\" Endian=\"Big\" DataType=\"Float\" Precision=\"8\" Dimensions=\"%D %D\" Seek=\"%D\">\n",N,bs,bytes[0]);CHKERRQ(ierr);
  }
  ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"%s\n",datafile);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"</DataItem>\n");CHKERRQ(ierr);
  ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"</Attribute>\n");CHKERRQ(ierr);
  ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  
  /* write data */
  ierr = VecView(x,fviewer);CHKERRQ(ierr);
  bytes[0] += sizeof(PetscReal) * N * bs;
  
  ierr = PetscViewerDestroy(&fviewer);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

PetscErrorCode private_ISView_Swarm_XDMF(IS is,PetscViewer viewer)
{
  long int       *bytes = NULL;
  PetscContainer container = NULL;
  const char     *viewername;
  char           datafile[PETSC_MAX_PATH_LEN];
  PetscViewer    fviewer;
  PetscInt       N,bs;
  const char     *vecname;
  char           fieldname[PETSC_MAX_PATH_LEN];
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  ierr = PetscObjectQuery((PetscObject)viewer,"XDMFViewerContext",(PetscObject*)&container);CHKERRQ(ierr);
  if (container) {
    ierr = PetscContainerGetPointer(container,(void**)&bytes);CHKERRQ(ierr);
  } else SETERRQ(PetscObjectComm((PetscObject)viewer),PETSC_ERR_SUP,"Valid to find attached data XDMFViewerContext");
  
  ierr = PetscViewerFileGetName(viewer,&viewername);CHKERRQ(ierr);
  ierr = private_CreateDataFileNameXDMF(viewername,datafile);CHKERRQ(ierr);
  
  /* re-open a sub-viewer for all data fields */
  /* name is viewer.name + "_swarm_fields.pbin" */
  ierr = PetscViewerCreate(PetscObjectComm((PetscObject)viewer),&fviewer);CHKERRQ(ierr);
  ierr = PetscViewerSetType(fviewer,PETSCVIEWERBINARY);CHKERRQ(ierr);
  ierr = PetscViewerBinarySetSkipHeader(fviewer,PETSC_TRUE);CHKERRQ(ierr);
  ierr = PetscViewerBinarySetSkipInfo(fviewer,PETSC_TRUE);CHKERRQ(ierr);
  ierr = PetscViewerFileSetMode(fviewer,FILE_MODE_APPEND);CHKERRQ(ierr);
  ierr = PetscViewerFileSetName(fviewer,datafile);CHKERRQ(ierr);
  
  ierr = ISGetSize(is,&N);CHKERRQ(ierr);
  ierr = ISGetBlockSize(is,&bs);CHKERRQ(ierr);
  N = N/bs;
  ierr = PetscObjectGetName((PetscObject)is,&vecname);CHKERRQ(ierr);
  if (!vecname) {
    ierr = PetscSNPrintf(fieldname,PETSC_MAX_PATH_LEN-1,"swarmfield_%D",((PetscObject)is)->tag);CHKERRQ(ierr);
  } else {
    ierr = PetscSNPrintf(fieldname,PETSC_MAX_PATH_LEN-1,"%s",vecname);CHKERRQ(ierr);
  }
  
  /* write data header */
  ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"<Attribute Center=\"Node\" Name=\"%s\" Type=\"None\">\n",fieldname);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
  if (bs == 1) {
    ierr = PetscViewerASCIIPrintf(viewer,"<DataItem Format=\"Binary\" Endian=\"Big\" DataType=\"Int\" Precision=\"4\" Dimensions=\"%D\" Seek=\"%D\">\n",N,bytes[0]);CHKERRQ(ierr);
  } else {
    ierr = PetscViewerASCIIPrintf(viewer,"<DataItem Format=\"Binary\" Endian=\"Big\" DataType=\"Int\" Precision=\"4\" Dimensions=\"%D %D\" Seek=\"%D\">\n",N,bs,bytes[0]);CHKERRQ(ierr);
  }
  ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"%s\n",datafile);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"</DataItem>\n");CHKERRQ(ierr);
  ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"</Attribute>\n");CHKERRQ(ierr);
  ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  
  /* write data */
  ierr = ISView(is,fviewer);CHKERRQ(ierr);
  bytes[0] += sizeof(PetscInt) * N * bs;
  
  ierr = PetscViewerDestroy(&fviewer);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

/*@C
   DMSwarmViewFieldsXDMF - Write a selection of DMSwarm fields to an XDMF3 file
 
   Collective on dm
 
   Input parameters:
+  dm - the DMSwarm
.  filename - the file name of the XDMF file (must have the extension .xmf)
.  nfields - the number of fields to write into the XDMF file
-  field_name_list - array of length nfields containing the textual name of fields to write
 
   Level: beginner

   Notes:
   Only fields registered with data type PETSC_DOUBLE or PETSC_INT can be written into the file
 
.seealso: DMSwarmViewXDMF()
@*/
PETSC_EXTERN PetscErrorCode DMSwarmViewFieldsXDMF(DM dm,const char filename[],PetscInt nfields,const char *field_name_list[])
{
  PetscErrorCode ierr;
  Vec            dvec;
  PetscInt       f,N;
  PetscViewer    viewer;
  
  PetscFunctionBegin;
  ierr = private_PetscViewerCreate_XDMF(PetscObjectComm((PetscObject)dm),filename,&viewer);CHKERRQ(ierr);
  ierr = private_DMSwarmView_XDMF(dm,viewer);CHKERRQ(ierr);
  ierr = DMSwarmGetLocalSize(dm,&N);CHKERRQ(ierr);
  for (f=0; f<nfields; f++) {
    void          *data;
    PetscDataType type;

    ierr = DMSwarmGetField(dm,field_name_list[f],NULL,&type,&data);CHKERRQ(ierr);
    ierr = DMSwarmRestoreField(dm,field_name_list[f],NULL,&type,&data);CHKERRQ(ierr);

    if (type == PETSC_DOUBLE) {
      ierr = DMSwarmCreateGlobalVectorFromField(dm,field_name_list[f],&dvec);CHKERRQ(ierr);
      ierr = PetscObjectSetName((PetscObject)dvec,field_name_list[f]);CHKERRQ(ierr);
      ierr = private_VecView_Swarm_XDMF(dvec,viewer);CHKERRQ(ierr);
      ierr = DMSwarmDestroyGlobalVectorFromField(dm,field_name_list[f],&dvec);CHKERRQ(ierr);
    } else if (type == PETSC_INT) {
      IS is;
      const PetscInt *idx;

      ierr = DMSwarmGetField(dm,field_name_list[f],NULL,&type,&data);CHKERRQ(ierr);
      idx = (const PetscInt*)data;
      
      ierr = ISCreateGeneral(PetscObjectComm((PetscObject)dm),N,idx,PETSC_USE_POINTER,&is);CHKERRQ(ierr);
      ierr = PetscObjectSetName((PetscObject)is,field_name_list[f]);CHKERRQ(ierr);
      ierr = private_ISView_Swarm_XDMF(is,viewer);CHKERRQ(ierr);
      ierr = ISDestroy(&is);CHKERRQ(ierr);
      ierr = DMSwarmRestoreField(dm,field_name_list[f],NULL,&type,&data);CHKERRQ(ierr);
    } else SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Can only write PETSC_INT and PETSC_DOUBLE");

  }
  ierr = private_PetscViewerDestroy_XDMF(&viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   DMSwarmViewXDMF - Write DMSwarm fields to an XDMF3 file
 
   Collective on dm
 
   Input parameters:
+  dm - the DMSwarm
-  filename - the file name of the XDMF file (must have the extension .xmf)
 
   Level: beginner

   Notes:
   Only fields user registered with data type PETSC_DOUBLE or PETSC_INT will be written into the file

.seealso: DMSwarmViewFieldsXDMF()
@*/
PETSC_EXTERN PetscErrorCode DMSwarmViewXDMF(DM dm,const char filename[])
{
  DM_Swarm       *swarm = (DM_Swarm*)dm->data;
  PetscErrorCode ierr;
  Vec            dvec;
  PetscInt       f;
  PetscViewer    viewer;
  
  PetscFunctionBegin;
  ierr = private_PetscViewerCreate_XDMF(PetscObjectComm((PetscObject)dm),filename,&viewer);CHKERRQ(ierr);
  ierr = private_DMSwarmView_XDMF(dm,viewer);CHKERRQ(ierr);
  for (f=4; f<swarm->db->nfields; f++) { /* only examine user defined fields - the first 4 are internally created by DMSwarmPIC */
    DMSwarmDataField field;
    
    /* query field type - accept all those of type PETSC_DOUBLE */
    field = swarm->db->field[f];
    if (field->petsc_type == PETSC_DOUBLE) {
      ierr = DMSwarmCreateGlobalVectorFromField(dm,field->name,&dvec);CHKERRQ(ierr);
      ierr = PetscObjectSetName((PetscObject)dvec,field->name);CHKERRQ(ierr);
      ierr = private_VecView_Swarm_XDMF(dvec,viewer);CHKERRQ(ierr);
      ierr = DMSwarmDestroyGlobalVectorFromField(dm,field->name,&dvec);CHKERRQ(ierr);
    } else if (field->petsc_type == PETSC_INT) {
      IS             is;
      PetscInt       N;
      const PetscInt *idx;
      void           *data;
      
      ierr = DMSwarmGetLocalSize(dm,&N);CHKERRQ(ierr);
      ierr = DMSwarmGetField(dm,field->name,NULL,NULL,&data);CHKERRQ(ierr);
      idx = (const PetscInt*)data;
      
      ierr = ISCreateGeneral(PetscObjectComm((PetscObject)dm),N,idx,PETSC_USE_POINTER,&is);CHKERRQ(ierr);
      ierr = PetscObjectSetName((PetscObject)is,field->name);CHKERRQ(ierr);
      ierr = private_ISView_Swarm_XDMF(is,viewer);CHKERRQ(ierr);
      ierr = ISDestroy(&is);CHKERRQ(ierr);
      ierr = DMSwarmRestoreField(dm,field->name,NULL,NULL,&data);CHKERRQ(ierr);
    }
  }
  ierr = private_PetscViewerDestroy_XDMF(&viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
