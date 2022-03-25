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
  PetscCall(PetscViewerCreate(comm,&viewer));
  PetscCall(PetscViewerSetType(viewer,PETSCVIEWERASCII));
  PetscCall(PetscViewerFileSetMode(viewer,FILE_MODE_WRITE));
  PetscCall(PetscViewerFileSetName(viewer,filename));

  PetscCall(PetscMalloc1(1,&bytes));
  bytes[0] = 0;
  PetscCall(PetscContainerCreate(comm,&container));
  PetscCall(PetscContainerSetPointer(container,(void*)bytes));
  PetscCall(PetscObjectCompose((PetscObject)viewer,"XDMFViewerContext",(PetscObject)container));

  /* write xdmf header */
  PetscCall(PetscViewerASCIIPrintf(viewer,"<?xml version=\"1.0\" encoding=\"utf-8\"?>\n"));
  PetscCall(PetscViewerASCIIPrintf(viewer,"<Xdmf xmlns:xi=\"http://www.w3.org/2001/XInclude/\" Version=\"2.99\">\n"));
  /* write xdmf domain */
  PetscCall(PetscViewerASCIIPushTab(viewer));
  PetscCall(PetscViewerASCIIPrintf(viewer,"<Domain>\n"));
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

  PetscCall(PetscObjectQuery((PetscObject)viewer,"DMSwarm",(PetscObject*)&dm));
  if (dm) {
    PetscCall(PetscViewerASCIIPrintf(viewer,"</Grid>\n"));
    PetscCall(PetscViewerASCIIPopTab(viewer));
  }

  /* close xdmf header */
  PetscCall(PetscViewerASCIIPrintf(viewer,"</Domain>\n"));
  PetscCall(PetscViewerASCIIPopTab(viewer));
  PetscCall(PetscViewerASCIIPrintf(viewer,"</Xdmf>\n"));

  PetscCall(PetscObjectQuery((PetscObject)viewer,"XDMFViewerContext",(PetscObject*)&container));
  if (container) {
    PetscCall(PetscContainerGetPointer(container,(void**)&bytes));
    PetscCall(PetscFree(bytes));
    PetscCall(PetscContainerDestroy(&container));
  }
  PetscCall(PetscViewerDestroy(&viewer));
  *v = NULL;
  PetscFunctionReturn(0);
}

PetscErrorCode private_CreateDataFileNameXDMF(const char filename[],char dfilename[])
{
  char           *ext;
  PetscBool      flg;

  PetscFunctionBegin;
  PetscCall(PetscStrrchr(filename,'.',&ext));
  PetscCall(PetscStrcmp("xmf",ext,&flg));
  if (flg) {
    size_t len;
    char    viewername_minus_ext[PETSC_MAX_PATH_LEN];

    PetscCall(PetscStrlen(filename,&len));
    PetscCall(PetscStrncpy(viewername_minus_ext,filename,len-2));
    PetscCall(PetscSNPrintf(dfilename,PETSC_MAX_PATH_LEN-1,"%s_swarm_fields.pbin",viewername_minus_ext));
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
  PetscCall(PetscObjectQuery((PetscObject)viewer,"XDMFViewerContext",(PetscObject*)&container));
  if (container) {
    PetscCall(PetscContainerGetPointer(container,(void**)&bytes));
  } else SETERRQ(PetscObjectComm((PetscObject)viewer),PETSC_ERR_SUP,"Valid to find attached data XDMFViewerContext");

  PetscCall(PetscObjectTypeCompare((PetscObject)dm,DMSWARM,&isswarm));
  PetscCheck(isswarm,PetscObjectComm((PetscObject)viewer),PETSC_ERR_SUP,"Only valid for DMSwarm");

  PetscCall(PetscObjectCompose((PetscObject)viewer,"DMSwarm",(PetscObject)dm));

  PetscCall(PetscViewerASCIIPushTab(viewer));
  PetscCall(PetscObjectGetName((PetscObject)dm,&dmname));
  if (!dmname) {
    PetscCall(DMGetOptionsPrefix(dm,&dmname));
  }
  if (!dmname) {
    PetscCall(PetscViewerASCIIPrintf(viewer,"<Grid Name=\"DMSwarm\" GridType=\"Uniform\">\n"));
  } else {
    PetscCall(PetscViewerASCIIPrintf(viewer,"<Grid Name=\"DMSwarm[%s]\" GridType=\"Uniform\">\n",dmname));
  }

  /* create a sub-viewer for topology, geometry and all data fields */
  /* name is viewer.name + "_swarm_fields.pbin" */
  PetscCall(PetscViewerCreate(PetscObjectComm((PetscObject)viewer),&fviewer));
  PetscCall(PetscViewerSetType(fviewer,PETSCVIEWERBINARY));
  PetscCall(PetscViewerBinarySetSkipHeader(fviewer,PETSC_TRUE));
  PetscCall(PetscViewerBinarySetSkipInfo(fviewer,PETSC_TRUE));
  PetscCall(PetscViewerFileSetMode(fviewer,FILE_MODE_WRITE));

  PetscCall(PetscViewerFileGetName(viewer,&viewername));
  PetscCall(private_CreateDataFileNameXDMF(viewername,datafile));
  PetscCall(PetscViewerFileSetName(fviewer,datafile));
  PetscCall(PetscStrrchr(datafile,'/',&datafilename));

  PetscCall(DMSwarmGetSize(dm,&ng));

  /* write topology header */
  PetscCall(PetscViewerASCIIPushTab(viewer));
  PetscCall(PetscViewerASCIIPrintf(viewer,"<Topology Dimensions=\"%D\" TopologyType=\"Mixed\">\n",ng));
  PetscCall(PetscViewerASCIIPushTab(viewer));
  PetscCall(PetscViewerASCIIPrintf(viewer,"<DataItem Format=\"Binary\" Endian=\"Big\" DataType=\"Int\" Dimensions=\"%D\" Seek=\"%D\">\n",ng*3,bytes[0]));
  PetscCall(PetscViewerASCIIPushTab(viewer));
  PetscCall(PetscViewerASCIIPrintf(viewer,"%s\n",datafilename));
  PetscCall(PetscViewerASCIIPopTab(viewer));
  PetscCall(PetscViewerASCIIPrintf(viewer,"</DataItem>\n"));
  PetscCall(PetscViewerASCIIPopTab(viewer));
  PetscCall(PetscViewerASCIIPrintf(viewer,"</Topology>\n"));
  PetscCall(PetscViewerASCIIPopTab(viewer));

  /* write topology data */
  for (k=0; k<ng; k++) {
    PetscInt pvertex[3];

    pvertex[0] = 1;
    pvertex[1] = 1;
    pvertex[2] = k;
    PetscCall(PetscViewerBinaryWrite(fviewer,pvertex,3,PETSC_INT));
  }
  bytes[0] += sizeof(PetscInt) * ng * 3;

  /* write geometry header */
  PetscCall(PetscViewerASCIIPushTab(viewer));
  PetscCall(DMGetDimension(dm,&dim));
  switch (dim) {
    case 1:
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"No support for 1D");
    case 2:
      PetscCall(PetscViewerASCIIPrintf(viewer,"<Geometry Type=\"XY\">\n"));
      break;
    case 3:
      PetscCall(PetscViewerASCIIPrintf(viewer,"<Geometry Type=\"XYZ\">\n"));
      break;
  }
  PetscCall(PetscViewerASCIIPushTab(viewer));
  PetscCall(PetscViewerASCIIPrintf(viewer,"<DataItem Format=\"Binary\" Endian=\"Big\" DataType=\"Float\" Precision=\"8\" Dimensions=\"%D %D\" Seek=\"%D\">\n",ng,dim,bytes[0]));
  PetscCall(PetscViewerASCIIPushTab(viewer));
  PetscCall(PetscViewerASCIIPrintf(viewer,"%s\n",datafilename));
  PetscCall(PetscViewerASCIIPopTab(viewer));
  PetscCall(PetscViewerASCIIPrintf(viewer,"</DataItem>\n"));
  PetscCall(PetscViewerASCIIPopTab(viewer));
  PetscCall(PetscViewerASCIIPrintf(viewer,"</Geometry>\n"));
  PetscCall(PetscViewerASCIIPopTab(viewer));

  /* write geometry data */
  PetscCall(DMSwarmCreateGlobalVectorFromField(dm,DMSwarmPICField_coor,&dvec));
  PetscCall(VecView(dvec,fviewer));
  PetscCall(DMSwarmDestroyGlobalVectorFromField(dm,DMSwarmPICField_coor,&dvec));
  bytes[0] += sizeof(PetscReal) * ng * dim;

  PetscCall(PetscViewerDestroy(&fviewer));
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
  PetscCall(PetscObjectQuery((PetscObject)viewer,"XDMFViewerContext",(PetscObject*)&container));
  PetscCheck(container,PetscObjectComm((PetscObject)viewer),PETSC_ERR_SUP,"Unable to find attached data XDMFViewerContext");
  PetscCall(PetscContainerGetPointer(container,(void**)&bytes));
  PetscCall(PetscViewerFileGetName(viewer,&viewername));
  PetscCall(private_CreateDataFileNameXDMF(viewername,datafile));

  /* re-open a sub-viewer for all data fields */
  /* name is viewer.name + "_swarm_fields.pbin" */
  PetscCall(PetscViewerCreate(PetscObjectComm((PetscObject)viewer),&fviewer));
  PetscCall(PetscViewerSetType(fviewer,PETSCVIEWERBINARY));
  PetscCall(PetscViewerBinarySetSkipHeader(fviewer,PETSC_TRUE));
  PetscCall(PetscViewerBinarySetSkipInfo(fviewer,PETSC_TRUE));
  PetscCall(PetscViewerFileSetMode(fviewer,FILE_MODE_APPEND));
  PetscCall(PetscViewerFileSetName(fviewer,datafile));
  PetscCall(PetscStrrchr(datafile,'/',&datafilename));

  PetscCall(VecGetSize(x,&N));
  PetscCall(VecGetBlockSize(x,&bs));
  N = N/bs;
  PetscCall(PetscObjectGetName((PetscObject)x,&vecname));
  if (!vecname) {
    PetscCall(PetscSNPrintf(fieldname,PETSC_MAX_PATH_LEN-1,"swarmfield_%D",((PetscObject)x)->tag));
  } else {
    PetscCall(PetscSNPrintf(fieldname,PETSC_MAX_PATH_LEN-1,"%s",vecname));
  }

  /* write data header */
  PetscCall(PetscViewerASCIIPushTab(viewer));
  PetscCall(PetscViewerASCIIPrintf(viewer,"<Attribute Center=\"Node\" Name=\"%s\" Type=\"None\">\n",fieldname));
  PetscCall(PetscViewerASCIIPushTab(viewer));
  if (bs == 1) {
    PetscCall(PetscViewerASCIIPrintf(viewer,"<DataItem Format=\"Binary\" Endian=\"Big\" DataType=\"Float\" Precision=\"8\" Dimensions=\"%D\" Seek=\"%D\">\n",N,bytes[0]));
  } else {
    PetscCall(PetscViewerASCIIPrintf(viewer,"<DataItem Format=\"Binary\" Endian=\"Big\" DataType=\"Float\" Precision=\"8\" Dimensions=\"%D %D\" Seek=\"%D\">\n",N,bs,bytes[0]));
  }
  PetscCall(PetscViewerASCIIPushTab(viewer));
  PetscCall(PetscViewerASCIIPrintf(viewer,"%s\n",datafilename));
  PetscCall(PetscViewerASCIIPopTab(viewer));
  PetscCall(PetscViewerASCIIPrintf(viewer,"</DataItem>\n"));
  PetscCall(PetscViewerASCIIPopTab(viewer));
  PetscCall(PetscViewerASCIIPrintf(viewer,"</Attribute>\n"));
  PetscCall(PetscViewerASCIIPopTab(viewer));

  /* write data */
  PetscCall(VecView(x,fviewer));
  bytes[0] += sizeof(PetscReal) * N * bs;

  PetscCall(PetscViewerDestroy(&fviewer));
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
  PetscCall(PetscObjectQuery((PetscObject)viewer,"XDMFViewerContext",(PetscObject*)&container));
  PetscCheck(container,PetscObjectComm((PetscObject)viewer),PETSC_ERR_SUP,"Unable to find attached data XDMFViewerContext");
  PetscCall(PetscContainerGetPointer(container,(void**)&bytes));
  PetscCall(PetscViewerFileGetName(viewer,&viewername));
  PetscCall(private_CreateDataFileNameXDMF(viewername,datafile));

  /* re-open a sub-viewer for all data fields */
  /* name is viewer.name + "_swarm_fields.pbin" */
  PetscCall(PetscViewerCreate(PetscObjectComm((PetscObject)viewer),&fviewer));
  PetscCall(PetscViewerSetType(fviewer,PETSCVIEWERBINARY));
  PetscCall(PetscViewerBinarySetSkipHeader(fviewer,PETSC_TRUE));
  PetscCall(PetscViewerBinarySetSkipInfo(fviewer,PETSC_TRUE));
  PetscCall(PetscViewerFileSetMode(fviewer,FILE_MODE_APPEND));
  PetscCall(PetscViewerFileSetName(fviewer,datafile));
  PetscCall(PetscStrrchr(datafile,'/',&datafilename));

  PetscCall(ISGetSize(is,&N));
  PetscCall(ISGetBlockSize(is,&bs));
  N = N/bs;
  PetscCall(PetscObjectGetName((PetscObject)is,&vecname));
  if (!vecname) {
    PetscCall(PetscSNPrintf(fieldname,PETSC_MAX_PATH_LEN-1,"swarmfield_%D",((PetscObject)is)->tag));
  } else {
    PetscCall(PetscSNPrintf(fieldname,PETSC_MAX_PATH_LEN-1,"%s",vecname));
  }

  /* write data header */
  PetscCall(PetscViewerASCIIPushTab(viewer));
  PetscCall(PetscViewerASCIIPrintf(viewer,"<Attribute Center=\"Node\" Name=\"%s\" Type=\"None\">\n",fieldname));
  PetscCall(PetscViewerASCIIPushTab(viewer));
  if (bs == 1) {
    PetscCall(PetscViewerASCIIPrintf(viewer,"<DataItem Format=\"Binary\" Endian=\"Big\" DataType=\"Int\" Precision=\"4\" Dimensions=\"%D\" Seek=\"%D\">\n",N,bytes[0]));
  } else {
    PetscCall(PetscViewerASCIIPrintf(viewer,"<DataItem Format=\"Binary\" Endian=\"Big\" DataType=\"Int\" Precision=\"4\" Dimensions=\"%D %D\" Seek=\"%D\">\n",N,bs,bytes[0]));
  }
  PetscCall(PetscViewerASCIIPushTab(viewer));
  PetscCall(PetscViewerASCIIPrintf(viewer,"%s\n",datafilename));
  PetscCall(PetscViewerASCIIPopTab(viewer));
  PetscCall(PetscViewerASCIIPrintf(viewer,"</DataItem>\n"));
  PetscCall(PetscViewerASCIIPopTab(viewer));
  PetscCall(PetscViewerASCIIPrintf(viewer,"</Attribute>\n"));
  PetscCall(PetscViewerASCIIPopTab(viewer));

  /* write data */
  PetscCall(ISView(is,fviewer));
  bytes[0] += sizeof(PetscInt) * N * bs;

  PetscCall(PetscViewerDestroy(&fviewer));
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
  PetscCall(private_PetscViewerCreate_XDMF(PetscObjectComm((PetscObject)dm),filename,&viewer));
  PetscCall(private_DMSwarmView_XDMF(dm,viewer));
  PetscCall(DMSwarmGetLocalSize(dm,&N));
  for (f=0; f<nfields; f++) {
    void          *data;
    PetscDataType type;

    PetscCall(DMSwarmGetField(dm,field_name_list[f],NULL,&type,&data));
    PetscCall(DMSwarmRestoreField(dm,field_name_list[f],NULL,&type,&data));
    if (type == PETSC_DOUBLE) {
      PetscCall(DMSwarmCreateGlobalVectorFromField(dm,field_name_list[f],&dvec));
      PetscCall(PetscObjectSetName((PetscObject)dvec,field_name_list[f]));
      PetscCall(private_VecView_Swarm_XDMF(dvec,viewer));
      PetscCall(DMSwarmDestroyGlobalVectorFromField(dm,field_name_list[f],&dvec));
    } else if (type == PETSC_INT) {
      IS is;
      const PetscInt *idx;

      PetscCall(DMSwarmGetField(dm,field_name_list[f],NULL,&type,&data));
      idx = (const PetscInt*)data;

      PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)dm),N,idx,PETSC_USE_POINTER,&is));
      PetscCall(PetscObjectSetName((PetscObject)is,field_name_list[f]));
      PetscCall(private_ISView_Swarm_XDMF(is,viewer));
      PetscCall(ISDestroy(&is));
      PetscCall(DMSwarmRestoreField(dm,field_name_list[f],NULL,&type,&data));
    } else SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Can only write PETSC_INT and PETSC_DOUBLE");

  }
  PetscCall(private_PetscViewerDestroy_XDMF(&viewer));
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
  PetscCall(private_PetscViewerCreate_XDMF(PetscObjectComm((PetscObject)dm),filename,&viewer));
  PetscCall(private_DMSwarmView_XDMF(dm,viewer));
  for (f=4; f<swarm->db->nfields; f++) { /* only examine user defined fields - the first 4 are internally created by DMSwarmPIC */
    DMSwarmDataField field;

    /* query field type - accept all those of type PETSC_DOUBLE */
    field = swarm->db->field[f];
    if (field->petsc_type == PETSC_DOUBLE) {
      PetscCall(DMSwarmCreateGlobalVectorFromField(dm,field->name,&dvec));
      PetscCall(PetscObjectSetName((PetscObject)dvec,field->name));
      PetscCall(private_VecView_Swarm_XDMF(dvec,viewer));
      PetscCall(DMSwarmDestroyGlobalVectorFromField(dm,field->name,&dvec));
    } else if (field->petsc_type == PETSC_INT) {
      IS             is;
      PetscInt       N;
      const PetscInt *idx;
      void           *data;

      PetscCall(DMSwarmGetLocalSize(dm,&N));
      PetscCall(DMSwarmGetField(dm,field->name,NULL,NULL,&data));
      idx = (const PetscInt*)data;

      PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)dm),N,idx,PETSC_USE_POINTER,&is));
      PetscCall(PetscObjectSetName((PetscObject)is,field->name));
      PetscCall(private_ISView_Swarm_XDMF(is,viewer));
      PetscCall(ISDestroy(&is));
      PetscCall(DMSwarmRestoreField(dm,field->name,NULL,NULL,&data));
    }
  }
  PetscCall(private_PetscViewerDestroy_XDMF(&viewer));
  PetscFunctionReturn(0);
}
