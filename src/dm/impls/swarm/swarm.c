
#define PETSCDM_DLL
#include <petsc/private/dmswarmimpl.h>    /*I   "petscdmswarm.h"   I*/
#include "data_bucket.h"

PetscErrorCode DMSwarmMigrate_Push_Basic(DM dm,PetscBool remove_sent_points);


//typedef PetscErrorCode (*swarm_project)(DM,DM,Vec) DMSwarmProjectMethod; /* swarm, geometry, result */

//typedef enum { PROJECT_DMDA_AQ1=0, PROJECT_DMDA_P0 } DMSwarmDMDAProjectionType;

#if 0

/* Defines what the local space will be */
PetscErrorCode DMSwarmSetOverlap(void)
{
  
  PetscFunctionReturn(0);
}


/* coordinates */
/*
DMGetCoordinateDM returns self
DMGetCoordinates and DMGetCoordinatesLocal return same thing
Local view could be used to define overlapping information
*/

#endif

#undef __FUNCT__
#define __FUNCT__ "DMSwarmVectorDefineField"
PETSC_EXTERN PetscErrorCode DMSwarmVectorDefineField(DM dm,const char fieldname[])
{
  DM_Swarm *swarm = (DM_Swarm*)dm->data;
  PetscErrorCode ierr;
  PetscInt bs,n;
  PetscScalar *array;
  PetscDataType type;

  if (!swarm->issetup) { ierr = DMSetUp(dm);CHKERRQ(ierr); }
  ierr = DataBucketGetSizes(swarm->db,&n,NULL,NULL);CHKERRQ(ierr);
  ierr = DMSwarmGetField(dm,fieldname,&bs,&type,(void**)&array);CHKERRQ(ierr);

  /* Check all fields are of type PETSC_REAL or PETSC_SCALAR */
  if (type != PETSC_REAL) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Only valid for PETSC_REAL");
  
  PetscSNPrintf(swarm->vec_field_name,PETSC_MAX_PATH_LEN-1,"%s",fieldname);
  swarm->vec_field_set = PETSC_TRUE;
  swarm->vec_field_bs = bs;
  swarm->vec_field_nlocal = n;
  ierr = DMSwarmRestoreField(dm,fieldname,&bs,&type,(void**)&array);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

/* requires DMSwarmDefineFieldVector has been called */
#undef __FUNCT__
#define __FUNCT__ "DMCreateGlobalVector_Swarm"
PetscErrorCode DMCreateGlobalVector_Swarm(DM dm,Vec *vec)
{
  DM_Swarm *swarm = (DM_Swarm*)dm->data;
  PetscErrorCode ierr;
  Vec x;
  char name[PETSC_MAX_PATH_LEN];

  if (!swarm->issetup) { ierr = DMSetUp(dm);CHKERRQ(ierr); }
  if (!swarm->vec_field_set) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_USER,"Must call DMSwarmVectorDefineField first");
  PetscSNPrintf(name,PETSC_MAX_PATH_LEN-1,"DMSwarmField_%s",swarm->vec_field_name);
  ierr = VecCreate(PetscObjectComm((PetscObject)dm),&x);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)x,name);CHKERRQ(ierr);
  ierr = VecSetSizes(x,swarm->db->L*swarm->vec_field_bs,PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = VecSetBlockSize(x,swarm->vec_field_bs);CHKERRQ(ierr);
  ierr = VecSetFromOptions(x);CHKERRQ(ierr);
  *vec = x;
  
  PetscFunctionReturn(0);
}

/* requires DMSwarmDefineFieldVector has been called */
#undef __FUNCT__
#define __FUNCT__ "DMCreateLocalVector_Swarm"
PetscErrorCode DMCreateLocalVector_Swarm(DM dm,Vec *vec)
{
  DM_Swarm *swarm = (DM_Swarm*)dm->data;
  PetscErrorCode ierr;
  Vec x;
  char name[PETSC_MAX_PATH_LEN];
  
  if (!swarm->issetup) { ierr = DMSetUp(dm);CHKERRQ(ierr); }
  if (!swarm->vec_field_set) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_USER,"Must call DMSwarmVectorDefineField first");
  PetscSNPrintf(name,PETSC_MAX_PATH_LEN-1,"DMSwarmField_%s",swarm->vec_field_name);
  ierr = VecCreate(PETSC_COMM_SELF,&x);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)x,name);CHKERRQ(ierr);
  ierr = VecSetSizes(x,swarm->db->L*swarm->vec_field_bs,swarm->db->L);CHKERRQ(ierr);
  ierr = VecSetBlockSize(x,swarm->vec_field_bs);CHKERRQ(ierr);
  ierr = VecSetFromOptions(x);CHKERRQ(ierr);
  *vec = x;
  
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMSwarmCreateGlobalVectorFromField"
PETSC_EXTERN PetscErrorCode DMSwarmCreateGlobalVectorFromField(DM dm,const char fieldname[],Vec *vec)
{
  DM_Swarm *swarm = (DM_Swarm*)dm->data;
  PetscErrorCode ierr;
  PetscInt bs,n;
  PetscScalar *array;
  Vec x;
  PetscDataType type;
  char name[PETSC_MAX_PATH_LEN];
  PetscMPIInt commsize;
  
  if (!swarm->issetup) { ierr = DMSetUp(dm);CHKERRQ(ierr); }
  ierr = DataBucketGetSizes(swarm->db,&n,NULL,NULL);CHKERRQ(ierr);
  ierr = DMSwarmGetField(dm,fieldname,&bs,&type,(void**)&array);CHKERRQ(ierr);

  /* Check all fields are of type PETSC_REAL or PETSC_SCALAR */
  if (type != PETSC_REAL) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Only valid for PETSC_REAL");
  
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)dm),&commsize);CHKERRQ(ierr);
  if (commsize == 1) {
    ierr = VecCreateSeqWithArray(PetscObjectComm((PetscObject)dm),bs,n*bs,array,&x);CHKERRQ(ierr);
  } else {
    ierr = VecCreateMPIWithArray(PetscObjectComm((PetscObject)dm),bs,n*bs,PETSC_DETERMINE,array,&x);CHKERRQ(ierr);
  }
  PetscSNPrintf(name,PETSC_MAX_PATH_LEN-1,"DMSwarmSharedField_%s",fieldname);
  ierr = PetscObjectSetName((PetscObject)x,name);CHKERRQ(ierr);

  /* Set guard */
  PetscSNPrintf(name,PETSC_MAX_PATH_LEN-1,"DMSwarm_VecFieldInPlace_%s",fieldname);
  ierr = PetscObjectComposeFunction((PetscObject)x,name,DMSwarmDestroyGlobalVectorFromField);CHKERRQ(ierr);
  
  *vec = x;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMSwarmDestroyGlobalVectorFromField"
PETSC_EXTERN PetscErrorCode DMSwarmDestroyGlobalVectorFromField(DM dm,const char fieldname[],Vec *vec)
{
  DM_Swarm *swarm = (DM_Swarm*)dm->data;
  PetscErrorCode ierr;
  DataField gfield;
  char name[PETSC_MAX_PATH_LEN];
  void (*fptr)(void);
  
  /* get data field */
  ierr = DataBucketGetDataFieldByName(swarm->db,fieldname,&gfield);CHKERRQ(ierr);
  
  /* check vector is an inplace array */
  PetscSNPrintf(name,PETSC_MAX_PATH_LEN-1,"DMSwarm_VecFieldInPlace_%s",fieldname);
  ierr = PetscObjectQueryFunction((PetscObject)(*vec),name,&fptr);CHKERRQ(ierr);
  if (!fptr) SETERRQ1(PetscObjectComm((PetscObject)dm),PETSC_ERR_USER,"Vector being destroyed was not created from DMSwarm field(%s)",fieldname);
  
  /* restore data field */
  ierr = DataFieldRestoreAccess(gfield);CHKERRQ(ierr);
  
  ierr = VecDestroy(vec);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

/*
PETSC_EXTERN PetscErrorCode DMSwarmCreateGlobalVectorFromFields(DM dm,const PetscInt nf,const char *fieldnames[],Vec *vec)
{
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode DMSwarmRestoreGlobalVectorFromFields(DM dm,Vec *vec)
{
  PetscFunctionReturn(0);
}
*/
 
#undef __FUNCT__
#define __FUNCT__ "DMSwarmInitializeFieldRegister"
PETSC_EXTERN PetscErrorCode DMSwarmInitializeFieldRegister(DM dm)
{
  DM_Swarm *swarm = (DM_Swarm*)dm->data;
  PetscErrorCode ierr;

  swarm->field_registration_initialized = PETSC_TRUE;

  ierr = DMSwarmRegisterPetscDatatypeField(dm,"DMSwarm_pid",1,PETSC_LONG);CHKERRQ(ierr); /* unique identifer */
  ierr = DMSwarmRegisterPetscDatatypeField(dm,"DMSwarm_rank",1,PETSC_INT);CHKERRQ(ierr); /* used for communication */

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMSwarmFinalizeFieldRegister"
PETSC_EXTERN PetscErrorCode DMSwarmFinalizeFieldRegister(DM dm)
{
  DM_Swarm *swarm = (DM_Swarm*)dm->data;
  PetscErrorCode ierr;
  
  swarm->field_registration_finalized = PETSC_TRUE;
  ierr = DataBucketFinalize(swarm->db);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMSwarmSetLocalSizes"
PETSC_EXTERN PetscErrorCode DMSwarmSetLocalSizes(DM dm,PetscInt nlocal,PetscInt buffer)
{
  DM_Swarm *swarm = (DM_Swarm*)dm->data;
  PetscErrorCode ierr;
  
  ierr = DataBucketSetSizes(swarm->db,nlocal,buffer);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMSwarmGetLocalSize"
PETSC_EXTERN PetscErrorCode DMSwarmGetLocalSize(DM dm,PetscInt *nlocal)
{
  DM_Swarm *swarm = (DM_Swarm*)dm->data;
  PetscErrorCode ierr;
  
  if (nlocal) {
    ierr = DataBucketGetSizes(swarm->db,nlocal,NULL,NULL);CHKERRQ(ierr);
  }
  
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMSwarmGetSize"
PETSC_EXTERN PetscErrorCode DMSwarmGetSize(DM dm,PetscInt *n)
{
  DM_Swarm *swarm = (DM_Swarm*)dm->data;
  PetscErrorCode ierr;
  PetscInt nlocal,ng;
  
  ierr = DataBucketGetSizes(swarm->db,&nlocal,NULL,NULL);CHKERRQ(ierr);
  ierr = MPI_Allreduce(&nlocal,&ng,1,MPIU_INT,MPI_SUM,PetscObjectComm((PetscObject)dm));CHKERRQ(ierr);
  if (n) { *n = ng; }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMSwarmRegisterPetscDatatypeField"
PETSC_EXTERN PetscErrorCode DMSwarmRegisterPetscDatatypeField(DM dm,const char fieldname[],PetscInt blocksize,PetscDataType type)
{
  PetscErrorCode ierr;
  DM_Swarm *swarm = (DM_Swarm*)dm->data;
  size_t size;
  
  if (!swarm->field_registration_initialized) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_USER,"Must call DMSwarmInitializeFieldRegister() first");
  if (swarm->field_registration_finalized) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_USER,"Cannot register additional fields after calling DMSwarmFinalizeFieldRegister() first");
  
  if (type == PETSC_OBJECT) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Valid for {char,short,int,long,float,double}");
  if (type == PETSC_FUNCTION) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Valid for {char,short,int,long,float,double}");
  if (type == PETSC_STRING) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Valid for {char,short,int,long,float,double}");
  if (type == PETSC_STRUCT) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Valid for {char,short,int,long,float,double}");
  if (type == PETSC_DATATYPE_UNKNOWN) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Valid for {char,short,int,long,float,double}");
  
  switch (type) {
    case PETSC_CHAR:
      size = sizeof(PetscChar);
      break;
    case PETSC_SHORT:
      size = sizeof(PetscShort);
      break;
    case PETSC_INT:
      size = sizeof(PetscInt);
      break;
    case PETSC_LONG:
      size = sizeof(Petsc64bitInt);
      break;
    case PETSC_FLOAT:
      size = sizeof(PetscFloat);
      break;
    case PETSC_DOUBLE:
      size = sizeof(PetscReal);
      break;
      
    default:
      SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Valid for {char,short,int,long,float,double}");
      break;
  }
  
  /* Load a specific data type into data bucket, specifying textual name and its size in bytes */
	ierr = DataBucketRegisterField(swarm->db,"DMSwarmRegisterPetscDatatypeField",fieldname,blocksize*size,NULL);CHKERRQ(ierr);
  {
    DataField gfield;
    
    ierr = DataBucketGetDataFieldByName(swarm->db,fieldname,&gfield);CHKERRQ(ierr);
    ierr = DataFieldSetBlockSize(gfield,blocksize);CHKERRQ(ierr);
  }
  swarm->db->field[swarm->db->nfields-1]->petsc_type = type;
  
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMSwarmRegisterUserStructField"
PETSC_EXTERN PetscErrorCode DMSwarmRegisterUserStructField(DM dm,const char fieldname[],size_t size)
{
  PetscErrorCode ierr;
  DM_Swarm *swarm = (DM_Swarm*)dm->data;
  
	ierr = DataBucketRegisterField(swarm->db,"DMSwarmRegisterUserStructField",fieldname,size,NULL);CHKERRQ(ierr);
  swarm->db->field[swarm->db->nfields-1]->petsc_type = PETSC_STRUCT ;
  
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMSwarmRegisterUserDatatypeField"
PETSC_EXTERN PetscErrorCode DMSwarmRegisterUserDatatypeField(DM dm,const char fieldname[],size_t size,PetscInt blocksize)
{
  DM_Swarm *swarm = (DM_Swarm*)dm->data;
  PetscErrorCode ierr;

	ierr = DataBucketRegisterField(swarm->db,"DMSwarmRegisterUserDatatypeField",fieldname,blocksize*size,NULL);CHKERRQ(ierr);
  {
    DataField gfield;
    
    ierr = DataBucketGetDataFieldByName(swarm->db,fieldname,&gfield);CHKERRQ(ierr);
    ierr = DataFieldSetBlockSize(gfield,blocksize);CHKERRQ(ierr);
  }
  swarm->db->field[swarm->db->nfields-1]->petsc_type = PETSC_DATATYPE_UNKNOWN;
  
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMSwarmGetField"
PETSC_EXTERN PetscErrorCode DMSwarmGetField(DM dm,const char fieldname[],PetscInt *blocksize,PetscDataType *type,void **data)
{
  DM_Swarm *swarm = (DM_Swarm*)dm->data;
  DataField gfield;
  PetscErrorCode ierr;
  
  if (!swarm->issetup) { ierr = DMSetUp(dm);CHKERRQ(ierr); }
  
  ierr = DataBucketGetDataFieldByName(swarm->db,fieldname,&gfield);CHKERRQ(ierr);
  ierr = DataFieldGetAccess(gfield);CHKERRQ(ierr);
  ierr = DataFieldGetEntries(gfield,data);CHKERRQ(ierr);
  if (blocksize) {*blocksize = gfield->bs; }
  if (type) { *type = gfield->petsc_type; }
  
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMSwarmRestoreField"
PETSC_EXTERN PetscErrorCode DMSwarmRestoreField(DM dm,const char fieldname[],PetscInt *blocksize,PetscDataType *type,void **data)
{
  DM_Swarm *swarm = (DM_Swarm*)dm->data;
  DataField gfield;
  PetscErrorCode ierr;
  
  ierr = DataBucketGetDataFieldByName(swarm->db,fieldname,&gfield);CHKERRQ(ierr);
  ierr = DataFieldRestoreAccess(gfield);CHKERRQ(ierr);
  if (data) *data = NULL;
  
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMSwarmAddPoint"
PETSC_EXTERN PetscErrorCode DMSwarmAddPoint(DM dm)
{
  DM_Swarm *swarm = (DM_Swarm*)dm->data;
  PetscErrorCode ierr;
  
  if (!swarm->issetup) { ierr = DMSetUp(dm);CHKERRQ(ierr); }
  ierr = DataBucketAddPoint(swarm->db);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMSwarmAddNPoints"
PETSC_EXTERN PetscErrorCode DMSwarmAddNPoints(DM dm,PetscInt npoints)
{
  DM_Swarm *swarm = (DM_Swarm*)dm->data;
  PetscErrorCode ierr;
  PetscInt nlocal;

  ierr = DataBucketGetSizes(swarm->db,&nlocal,NULL,NULL);CHKERRQ(ierr);
  nlocal = nlocal + npoints;
  ierr = DataBucketSetSizes(swarm->db,nlocal,-1);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMSwarmRemovePoint"
PETSC_EXTERN PetscErrorCode DMSwarmRemovePoint(DM dm)
{
  DM_Swarm *swarm = (DM_Swarm*)dm->data;
  PetscErrorCode ierr;

  ierr = DataBucketRemovePoint(swarm->db);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMSwarmRemovePointAtIndex"
PETSC_EXTERN PetscErrorCode DMSwarmRemovePointAtIndex(DM dm,PetscInt idx)
{
  DM_Swarm *swarm = (DM_Swarm*)dm->data;
  PetscErrorCode ierr;

  ierr = DataBucketRemovePointAtIndex(swarm->db,idx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMSwarmMigrate_Basic"
PetscErrorCode DMSwarmMigrate_Basic(DM dm,PetscBool remove_sent_points)
{
  PetscErrorCode ierr;
  ierr = DMSwarmMigrate_Push_Basic(dm,remove_sent_points);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMSwarmMigrate"
PETSC_EXTERN PetscErrorCode DMSwarmMigrate(DM dm,PetscBool remove_sent_points)
{
  PetscErrorCode ierr;
  ierr = DMSwarmMigrate_Basic(dm,remove_sent_points);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMSwarmGlobalToLocalViewCreate"
PETSC_EXTERN PetscErrorCode DMSwarmGlobalToLocalViewCreate(DM dm,InsertMode mode)
{
  PetscErrorCode ierr;
  DM_Swarm *swarm = (DM_Swarm*)dm->data;
  PetscInt ng;
  
  if (mode != INSERT_VALUES) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Only mode INSERT_VALUES is supported");

  ierr = DMSwarmMigrate_GlobalToLocal_Basic(dm,&ng);CHKERRQ(ierr);
  swarm->view_ng = ng;
  
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMSwarmGlobalToLocalViewDestroy"
PETSC_EXTERN PetscErrorCode DMSwarmGlobalToLocalViewDestroy(DM dm,InsertMode mode)
{
  PetscErrorCode ierr;
  DM_Swarm *swarm = (DM_Swarm*)dm->data;
  
  if (mode != INSERT_VALUES) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Only mode INSERT_VALUES is supported");
  
  ierr = DMSwarmSetLocalSizes(dm,swarm->view_ng,-1);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMSetup_Swarm"
PetscErrorCode DMSetup_Swarm(DM dm)
{
  DM_Swarm *swarm = (DM_Swarm*)dm->data;
  PetscErrorCode ierr;
  PetscMPIInt rank;
  PetscInt p,npoints,*rankval;
  
  if (swarm->issetup) PetscFunctionReturn(0);
  
  PetscPrintf(PETSC_COMM_SELF,"Swarm setup \n");
  swarm->issetup = PETSC_TRUE;

  /* check some fields were registered */
  if (swarm->db->nfields <= 2) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_USER,"At least one field user must be registered via DMSwarmRegisterXXX()");

  /* check local sizes were set */
  if (swarm->db->L == -1) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_USER,"Local sizes must be set via DMSwarmSetLocalSizes()");

  /* initialize values in pid and rank placeholders */
  /* TODO: [pid - use MPI_Scan] */
  
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)dm),&rank);CHKERRQ(ierr);
  ierr = DataBucketGetSizes(swarm->db,&npoints,NULL,NULL);CHKERRQ(ierr);
  ierr = DMSwarmGetField(dm,"DMSwarm_rank",NULL,NULL,(void**)&rankval);CHKERRQ(ierr);
  for (p=0; p<npoints; p++) {
    rankval[p] = (PetscInt)rank;
  }
  ierr = DMSwarmRestoreField(dm,"DMSwarm_rank",NULL,NULL,(void**)&rankval);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMDestroy_Swarm"
PetscErrorCode DMDestroy_Swarm(DM dm)
{
  DM_Swarm *swarm = (DM_Swarm*)dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DataBucketDestroy(&swarm->db);CHKERRQ(ierr);
  ierr = PetscFree(swarm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMView_Swarm"
PetscErrorCode DMView_Swarm(DM dm, PetscViewer viewer)
{
  DM_Swarm *swarm = (DM_Swarm*)dm->data;
  PetscBool      iascii,ibinary,ishdf5,isvtk;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  ierr = PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERBINARY,&ibinary);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERVTK,   &isvtk);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERHDF5,  &ishdf5);CHKERRQ(ierr);
  if (iascii) {
    ierr = DataBucketView(PetscObjectComm((PetscObject)dm),swarm->db,NULL,DATABUCKET_VIEW_STDOUT);CHKERRQ(ierr);
  } else if (ibinary) {
    SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"NO VTK support");
  } else if (ishdf5) {
#if defined(PETSC_HAVE_HDF5)
    SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"NO HDF5 support");
#else
    SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"HDF5 not supported. Please reconfigure using --download-hdf5");
#endif
  } else if (isvtk) {
    SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"NO VTK support");
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMCreate_Swarm"
PETSC_EXTERN PetscErrorCode DMCreate_Swarm(DM dm)
{
  DM_Swarm      *swarm;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr     = PetscNewLog(dm,&swarm);CHKERRQ(ierr);
  
  ierr = DataBucketCreate(&swarm->db);CHKERRQ(ierr);
  swarm->vec_field_set = PETSC_FALSE;
  swarm->issetup = PETSC_FALSE;
  
  dm->dim  = 0;
  dm->data = swarm;
  
  dm->ops->view                            = DMView_Swarm;
  dm->ops->load                            = NULL;
  dm->ops->setfromoptions                  = NULL;
  dm->ops->clone                           = NULL;
  dm->ops->setup                           = DMSetup_Swarm;
  dm->ops->createdefaultsection            = NULL;
  dm->ops->createdefaultconstraints        = NULL;
  dm->ops->createglobalvector              = DMCreateGlobalVector_Swarm;
  dm->ops->createlocalvector               = DMCreateLocalVector_Swarm;
  dm->ops->getlocaltoglobalmapping         = NULL;
  dm->ops->createfieldis                   = NULL;
  dm->ops->createcoordinatedm              = NULL;
  dm->ops->getcoloring                     = NULL;
  dm->ops->creatematrix                    = NULL;
  dm->ops->createinterpolation             = NULL;
  dm->ops->getaggregates                   = NULL;
  dm->ops->getinjection                    = NULL;
  dm->ops->refine                          = NULL;
  dm->ops->coarsen                         = NULL;
  dm->ops->refinehierarchy                 = NULL;
  dm->ops->coarsenhierarchy                = NULL;
  dm->ops->globaltolocalbegin              = NULL;
  dm->ops->globaltolocalend                = NULL;
  dm->ops->localtoglobalbegin              = NULL;
  dm->ops->localtoglobalend                = NULL;
  dm->ops->destroy                         = DMDestroy_Swarm;
  dm->ops->createsubdm                     = NULL;
  dm->ops->getdimpoints                    = NULL;
  dm->ops->locatepoints                    = NULL;
  
  PetscFunctionReturn(0);
}