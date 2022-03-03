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

  PetscFunctionBegin;
  CHKERRQ(PetscViewerCreate(comm,&viewer));
  CHKERRQ(PetscViewerSetType(viewer,PETSCVIEWERASCII));
  CHKERRQ(PetscViewerFileSetMode(viewer,FILE_MODE_WRITE));
  CHKERRQ(PetscViewerFileSetName(viewer,filename));

  CHKERRQ(PetscMalloc1(1,&bytes));
  bytes[0] = 0;
  CHKERRQ(PetscContainerCreate(comm,&container));
  CHKERRQ(PetscContainerSetPointer(container,(void*)bytes));
  CHKERRQ(PetscObjectCompose((PetscObject)viewer,"XDMFViewerContext",(PetscObject)container));

  /* write xdmf header */
  CHKERRQ(PetscViewerASCIIPrintf(viewer,"<?xml version=\"1.0\" encoding=\"utf-8\"?>\n"));
  CHKERRQ(PetscViewerASCIIPrintf(viewer,"<Xdmf xmlns:xi=\"http://www.w3.org/2001/XInclude/\" Version=\"2.99\">\n"));
  /* write xdmf domain */
  CHKERRQ(PetscViewerASCIIPushTab(viewer));
  CHKERRQ(PetscViewerASCIIPrintf(viewer,"<Domain>\n"));
  *v = viewer;
  PetscFunctionReturn(0);
}

PetscErrorCode private_PetscViewerDestroy_XDMF(PetscViewer *v)
{
  PetscViewer    viewer;
  DM             dm = NULL;
  long int       *bytes;
  PetscContainer container = NULL;

  PetscFunctionBegin;
  if (!v) PetscFunctionReturn(0);
  viewer = *v;

  CHKERRQ(PetscObjectQuery((PetscObject)viewer,"DMSwarm",(PetscObject*)&dm));
  if (dm) {
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"</Grid>\n"));
    CHKERRQ(PetscViewerASCIIPopTab(viewer));
  }

  /* close xdmf header */
  CHKERRQ(PetscViewerASCIIPrintf(viewer,"</Domain>\n"));
  CHKERRQ(PetscViewerASCIIPopTab(viewer));
  CHKERRQ(PetscViewerASCIIPrintf(viewer,"</Xdmf>\n"));

  CHKERRQ(PetscObjectQuery((PetscObject)viewer,"XDMFViewerContext",(PetscObject*)&container));
  if (container) {
    CHKERRQ(PetscContainerGetPointer(container,(void**)&bytes));
    CHKERRQ(PetscFree(bytes));
    CHKERRQ(PetscContainerDestroy(&container));
  }
  CHKERRQ(PetscViewerDestroy(&viewer));
  *v = NULL;
  PetscFunctionReturn(0);
}

PetscErrorCode private_CreateDataFileNameXDMF(const char filename[],char dfilename[])
{
  char           *ext;
  PetscBool      flg;

  PetscFunctionBegin;
  CHKERRQ(PetscStrrchr(filename,'.',&ext));
  CHKERRQ(PetscStrcmp("xmf",ext,&flg));
  if (flg) {
    size_t len;
    char    viewername_minus_ext[PETSC_MAX_PATH_LEN];

    CHKERRQ(PetscStrlen(filename,&len));
    CHKERRQ(PetscStrncpy(viewername_minus_ext,filename,len-2));
    CHKERRQ(PetscSNPrintf(dfilename,PETSC_MAX_PATH_LEN-1,"%s_swarm_fields.pbin",viewername_minus_ext));
  } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"File extension must by .xmf");
  PetscFunctionReturn(0);
}

PetscErrorCode private_DMSwarmView_XDMF(DM dm,PetscViewer viewer)
{
  PetscBool      isswarm = PETSC_FALSE;
  const char     *viewername;
  char           datafile[PETSC_MAX_PATH_LEN];
  char           *datafilename;
  PetscViewer    fviewer;
  PetscInt       k,ng,dim;
  Vec            dvec;
  long int       *bytes = NULL;
  PetscContainer container = NULL;
  const char     *dmname;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectQuery((PetscObject)viewer,"XDMFViewerContext",(PetscObject*)&container));
  if (container) {
    CHKERRQ(PetscContainerGetPointer(container,(void**)&bytes));
  } else SETERRQ(PetscObjectComm((PetscObject)viewer),PETSC_ERR_SUP,"Valid to find attached data XDMFViewerContext");

  CHKERRQ(PetscObjectTypeCompare((PetscObject)dm,DMSWARM,&isswarm));
  PetscCheck(isswarm,PetscObjectComm((PetscObject)viewer),PETSC_ERR_SUP,"Only valid for DMSwarm");

  CHKERRQ(PetscObjectCompose((PetscObject)viewer,"DMSwarm",(PetscObject)dm));

  CHKERRQ(PetscViewerASCIIPushTab(viewer));
  CHKERRQ(PetscObjectGetName((PetscObject)dm,&dmname));
  if (!dmname) {
    CHKERRQ(DMGetOptionsPrefix(dm,&dmname));
  }
  if (!dmname) {
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"<Grid Name=\"DMSwarm\" GridType=\"Uniform\">\n"));
  } else {
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"<Grid Name=\"DMSwarm[%s]\" GridType=\"Uniform\">\n",dmname));
  }

  /* create a sub-viewer for topology, geometry and all data fields */
  /* name is viewer.name + "_swarm_fields.pbin" */
  CHKERRQ(PetscViewerCreate(PetscObjectComm((PetscObject)viewer),&fviewer));
  CHKERRQ(PetscViewerSetType(fviewer,PETSCVIEWERBINARY));
  CHKERRQ(PetscViewerBinarySetSkipHeader(fviewer,PETSC_TRUE));
  CHKERRQ(PetscViewerBinarySetSkipInfo(fviewer,PETSC_TRUE));
  CHKERRQ(PetscViewerFileSetMode(fviewer,FILE_MODE_WRITE));

  CHKERRQ(PetscViewerFileGetName(viewer,&viewername));
  CHKERRQ(private_CreateDataFileNameXDMF(viewername,datafile));
  CHKERRQ(PetscViewerFileSetName(fviewer,datafile));
  CHKERRQ(PetscStrrchr(datafile,'/',&datafilename));

  CHKERRQ(DMSwarmGetSize(dm,&ng));

  /* write topology header */
  CHKERRQ(PetscViewerASCIIPushTab(viewer));
  CHKERRQ(PetscViewerASCIIPrintf(viewer,"<Topology Dimensions=\"%D\" TopologyType=\"Mixed\">\n",ng));
  CHKERRQ(PetscViewerASCIIPushTab(viewer));
  CHKERRQ(PetscViewerASCIIPrintf(viewer,"<DataItem Format=\"Binary\" Endian=\"Big\" DataType=\"Int\" Dimensions=\"%D\" Seek=\"%D\">\n",ng*3,bytes[0]));
  CHKERRQ(PetscViewerASCIIPushTab(viewer));
  CHKERRQ(PetscViewerASCIIPrintf(viewer,"%s\n",datafilename));
  CHKERRQ(PetscViewerASCIIPopTab(viewer));
  CHKERRQ(PetscViewerASCIIPrintf(viewer,"</DataItem>\n"));
  CHKERRQ(PetscViewerASCIIPopTab(viewer));
  CHKERRQ(PetscViewerASCIIPrintf(viewer,"</Topology>\n"));
  CHKERRQ(PetscViewerASCIIPopTab(viewer));

  /* write topology data */
  for (k=0; k<ng; k++) {
    PetscInt pvertex[3];

    pvertex[0] = 1;
    pvertex[1] = 1;
    pvertex[2] = k;
    CHKERRQ(PetscViewerBinaryWrite(fviewer,pvertex,3,PETSC_INT));
  }
  bytes[0] += sizeof(PetscInt) * ng * 3;

  /* write geometry header */
  CHKERRQ(PetscViewerASCIIPushTab(viewer));
  CHKERRQ(DMGetDimension(dm,&dim));
  switch (dim) {
    case 1:
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"No support for 1D");
    case 2:
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"<Geometry Type=\"XY\">\n"));
      break;
    case 3:
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"<Geometry Type=\"XYZ\">\n"));
      break;
  }
  CHKERRQ(PetscViewerASCIIPushTab(viewer));
  CHKERRQ(PetscViewerASCIIPrintf(viewer,"<DataItem Format=\"Binary\" Endian=\"Big\" DataType=\"Float\" Precision=\"8\" Dimensions=\"%D %D\" Seek=\"%D\">\n",ng,dim,bytes[0]));
  CHKERRQ(PetscViewerASCIIPushTab(viewer));
  CHKERRQ(PetscViewerASCIIPrintf(viewer,"%s\n",datafilename));
  CHKERRQ(PetscViewerASCIIPopTab(viewer));
  CHKERRQ(PetscViewerASCIIPrintf(viewer,"</DataItem>\n"));
  CHKERRQ(PetscViewerASCIIPopTab(viewer));
  CHKERRQ(PetscViewerASCIIPrintf(viewer,"</Geometry>\n"));
  CHKERRQ(PetscViewerASCIIPopTab(viewer));

  /* write geometry data */
  CHKERRQ(DMSwarmCreateGlobalVectorFromField(dm,DMSwarmPICField_coor,&dvec));
  CHKERRQ(VecView(dvec,fviewer));
  CHKERRQ(DMSwarmDestroyGlobalVectorFromField(dm,DMSwarmPICField_coor,&dvec));
  bytes[0] += sizeof(PetscReal) * ng * dim;

  CHKERRQ(PetscViewerDestroy(&fviewer));
  PetscFunctionReturn(0);
}

PetscErrorCode private_VecView_Swarm_XDMF(Vec x,PetscViewer viewer)
{
  long int       *bytes = NULL;
  PetscContainer container = NULL;
  const char     *viewername;
  char           datafile[PETSC_MAX_PATH_LEN];
  char           *datafilename;
  PetscViewer    fviewer;
  PetscInt       N,bs;
  const char     *vecname;
  char           fieldname[PETSC_MAX_PATH_LEN];

  PetscFunctionBegin;
  CHKERRQ(PetscObjectQuery((PetscObject)viewer,"XDMFViewerContext",(PetscObject*)&container));
  PetscCheck(container,PetscObjectComm((PetscObject)viewer),PETSC_ERR_SUP,"Unable to find attached data XDMFViewerContext");
  CHKERRQ(PetscContainerGetPointer(container,(void**)&bytes));
  CHKERRQ(PetscViewerFileGetName(viewer,&viewername));
  CHKERRQ(private_CreateDataFileNameXDMF(viewername,datafile));

  /* re-open a sub-viewer for all data fields */
  /* name is viewer.name + "_swarm_fields.pbin" */
  CHKERRQ(PetscViewerCreate(PetscObjectComm((PetscObject)viewer),&fviewer));
  CHKERRQ(PetscViewerSetType(fviewer,PETSCVIEWERBINARY));
  CHKERRQ(PetscViewerBinarySetSkipHeader(fviewer,PETSC_TRUE));
  CHKERRQ(PetscViewerBinarySetSkipInfo(fviewer,PETSC_TRUE));
  CHKERRQ(PetscViewerFileSetMode(fviewer,FILE_MODE_APPEND));
  CHKERRQ(PetscViewerFileSetName(fviewer,datafile));
  CHKERRQ(PetscStrrchr(datafile,'/',&datafilename));

  CHKERRQ(VecGetSize(x,&N));
  CHKERRQ(VecGetBlockSize(x,&bs));
  N = N/bs;
  CHKERRQ(PetscObjectGetName((PetscObject)x,&vecname));
  if (!vecname) {
    CHKERRQ(PetscSNPrintf(fieldname,PETSC_MAX_PATH_LEN-1,"swarmfield_%D",((PetscObject)x)->tag));
  } else {
    CHKERRQ(PetscSNPrintf(fieldname,PETSC_MAX_PATH_LEN-1,"%s",vecname));
  }

  /* write data header */
  CHKERRQ(PetscViewerASCIIPushTab(viewer));
  CHKERRQ(PetscViewerASCIIPrintf(viewer,"<Attribute Center=\"Node\" Name=\"%s\" Type=\"None\">\n",fieldname));
  CHKERRQ(PetscViewerASCIIPushTab(viewer));
  if (bs == 1) {
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"<DataItem Format=\"Binary\" Endian=\"Big\" DataType=\"Float\" Precision=\"8\" Dimensions=\"%D\" Seek=\"%D\">\n",N,bytes[0]));
  } else {
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"<DataItem Format=\"Binary\" Endian=\"Big\" DataType=\"Float\" Precision=\"8\" Dimensions=\"%D %D\" Seek=\"%D\">\n",N,bs,bytes[0]));
  }
  CHKERRQ(PetscViewerASCIIPushTab(viewer));
  CHKERRQ(PetscViewerASCIIPrintf(viewer,"%s\n",datafilename));
  CHKERRQ(PetscViewerASCIIPopTab(viewer));
  CHKERRQ(PetscViewerASCIIPrintf(viewer,"</DataItem>\n"));
  CHKERRQ(PetscViewerASCIIPopTab(viewer));
  CHKERRQ(PetscViewerASCIIPrintf(viewer,"</Attribute>\n"));
  CHKERRQ(PetscViewerASCIIPopTab(viewer));

  /* write data */
  CHKERRQ(VecView(x,fviewer));
  bytes[0] += sizeof(PetscReal) * N * bs;

  CHKERRQ(PetscViewerDestroy(&fviewer));
  PetscFunctionReturn(0);
}

PetscErrorCode private_ISView_Swarm_XDMF(IS is,PetscViewer viewer)
{
  long int       *bytes = NULL;
  PetscContainer container = NULL;
  const char     *viewername;
  char           datafile[PETSC_MAX_PATH_LEN];
  char           *datafilename;
  PetscViewer    fviewer;
  PetscInt       N,bs;
  const char     *vecname;
  char           fieldname[PETSC_MAX_PATH_LEN];

  PetscFunctionBegin;
  CHKERRQ(PetscObjectQuery((PetscObject)viewer,"XDMFViewerContext",(PetscObject*)&container));
  PetscCheck(container,PetscObjectComm((PetscObject)viewer),PETSC_ERR_SUP,"Unable to find attached data XDMFViewerContext");
  CHKERRQ(PetscContainerGetPointer(container,(void**)&bytes));
  CHKERRQ(PetscViewerFileGetName(viewer,&viewername));
  CHKERRQ(private_CreateDataFileNameXDMF(viewername,datafile));

  /* re-open a sub-viewer for all data fields */
  /* name is viewer.name + "_swarm_fields.pbin" */
  CHKERRQ(PetscViewerCreate(PetscObjectComm((PetscObject)viewer),&fviewer));
  CHKERRQ(PetscViewerSetType(fviewer,PETSCVIEWERBINARY));
  CHKERRQ(PetscViewerBinarySetSkipHeader(fviewer,PETSC_TRUE));
  CHKERRQ(PetscViewerBinarySetSkipInfo(fviewer,PETSC_TRUE));
  CHKERRQ(PetscViewerFileSetMode(fviewer,FILE_MODE_APPEND));
  CHKERRQ(PetscViewerFileSetName(fviewer,datafile));
  CHKERRQ(PetscStrrchr(datafile,'/',&datafilename));

  CHKERRQ(ISGetSize(is,&N));
  CHKERRQ(ISGetBlockSize(is,&bs));
  N = N/bs;
  CHKERRQ(PetscObjectGetName((PetscObject)is,&vecname));
  if (!vecname) {
    CHKERRQ(PetscSNPrintf(fieldname,PETSC_MAX_PATH_LEN-1,"swarmfield_%D",((PetscObject)is)->tag));
  } else {
    CHKERRQ(PetscSNPrintf(fieldname,PETSC_MAX_PATH_LEN-1,"%s",vecname));
  }

  /* write data header */
  CHKERRQ(PetscViewerASCIIPushTab(viewer));
  CHKERRQ(PetscViewerASCIIPrintf(viewer,"<Attribute Center=\"Node\" Name=\"%s\" Type=\"None\">\n",fieldname));
  CHKERRQ(PetscViewerASCIIPushTab(viewer));
  if (bs == 1) {
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"<DataItem Format=\"Binary\" Endian=\"Big\" DataType=\"Int\" Precision=\"4\" Dimensions=\"%D\" Seek=\"%D\">\n",N,bytes[0]));
  } else {
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"<DataItem Format=\"Binary\" Endian=\"Big\" DataType=\"Int\" Precision=\"4\" Dimensions=\"%D %D\" Seek=\"%D\">\n",N,bs,bytes[0]));
  }
  CHKERRQ(PetscViewerASCIIPushTab(viewer));
  CHKERRQ(PetscViewerASCIIPrintf(viewer,"%s\n",datafilename));
  CHKERRQ(PetscViewerASCIIPopTab(viewer));
  CHKERRQ(PetscViewerASCIIPrintf(viewer,"</DataItem>\n"));
  CHKERRQ(PetscViewerASCIIPopTab(viewer));
  CHKERRQ(PetscViewerASCIIPrintf(viewer,"</Attribute>\n"));
  CHKERRQ(PetscViewerASCIIPopTab(viewer));

  /* write data */
  CHKERRQ(ISView(is,fviewer));
  bytes[0] += sizeof(PetscInt) * N * bs;

  CHKERRQ(PetscViewerDestroy(&fviewer));
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
  Vec            dvec;
  PetscInt       f,N;
  PetscViewer    viewer;

  PetscFunctionBegin;
  CHKERRQ(private_PetscViewerCreate_XDMF(PetscObjectComm((PetscObject)dm),filename,&viewer));
  CHKERRQ(private_DMSwarmView_XDMF(dm,viewer));
  CHKERRQ(DMSwarmGetLocalSize(dm,&N));
  for (f=0; f<nfields; f++) {
    void          *data;
    PetscDataType type;

    CHKERRQ(DMSwarmGetField(dm,field_name_list[f],NULL,&type,&data));
    CHKERRQ(DMSwarmRestoreField(dm,field_name_list[f],NULL,&type,&data));
    if (type == PETSC_DOUBLE) {
      CHKERRQ(DMSwarmCreateGlobalVectorFromField(dm,field_name_list[f],&dvec));
      CHKERRQ(PetscObjectSetName((PetscObject)dvec,field_name_list[f]));
      CHKERRQ(private_VecView_Swarm_XDMF(dvec,viewer));
      CHKERRQ(DMSwarmDestroyGlobalVectorFromField(dm,field_name_list[f],&dvec));
    } else if (type == PETSC_INT) {
      IS is;
      const PetscInt *idx;

      CHKERRQ(DMSwarmGetField(dm,field_name_list[f],NULL,&type,&data));
      idx = (const PetscInt*)data;

      CHKERRQ(ISCreateGeneral(PetscObjectComm((PetscObject)dm),N,idx,PETSC_USE_POINTER,&is));
      CHKERRQ(PetscObjectSetName((PetscObject)is,field_name_list[f]));
      CHKERRQ(private_ISView_Swarm_XDMF(is,viewer));
      CHKERRQ(ISDestroy(&is));
      CHKERRQ(DMSwarmRestoreField(dm,field_name_list[f],NULL,&type,&data));
    } else SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Can only write PETSC_INT and PETSC_DOUBLE");

  }
  CHKERRQ(private_PetscViewerDestroy_XDMF(&viewer));
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

   Developer Notes:
     This should be removed and replaced with the standard use of PetscViewer

.seealso: DMSwarmViewFieldsXDMF()
@*/
PETSC_EXTERN PetscErrorCode DMSwarmViewXDMF(DM dm,const char filename[])
{
  DM_Swarm       *swarm = (DM_Swarm*)dm->data;
  Vec            dvec;
  PetscInt       f;
  PetscViewer    viewer;

  PetscFunctionBegin;
  CHKERRQ(private_PetscViewerCreate_XDMF(PetscObjectComm((PetscObject)dm),filename,&viewer));
  CHKERRQ(private_DMSwarmView_XDMF(dm,viewer));
  for (f=4; f<swarm->db->nfields; f++) { /* only examine user defined fields - the first 4 are internally created by DMSwarmPIC */
    DMSwarmDataField field;

    /* query field type - accept all those of type PETSC_DOUBLE */
    field = swarm->db->field[f];
    if (field->petsc_type == PETSC_DOUBLE) {
      CHKERRQ(DMSwarmCreateGlobalVectorFromField(dm,field->name,&dvec));
      CHKERRQ(PetscObjectSetName((PetscObject)dvec,field->name));
      CHKERRQ(private_VecView_Swarm_XDMF(dvec,viewer));
      CHKERRQ(DMSwarmDestroyGlobalVectorFromField(dm,field->name,&dvec));
    } else if (field->petsc_type == PETSC_INT) {
      IS             is;
      PetscInt       N;
      const PetscInt *idx;
      void           *data;

      CHKERRQ(DMSwarmGetLocalSize(dm,&N));
      CHKERRQ(DMSwarmGetField(dm,field->name,NULL,NULL,&data));
      idx = (const PetscInt*)data;

      CHKERRQ(ISCreateGeneral(PetscObjectComm((PetscObject)dm),N,idx,PETSC_USE_POINTER,&is));
      CHKERRQ(PetscObjectSetName((PetscObject)is,field->name));
      CHKERRQ(private_ISView_Swarm_XDMF(is,viewer));
      CHKERRQ(ISDestroy(&is));
      CHKERRQ(DMSwarmRestoreField(dm,field->name,NULL,NULL,&data));
    }
  }
  CHKERRQ(private_PetscViewerDestroy_XDMF(&viewer));
  PetscFunctionReturn(0);
}
