

#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: aodatabasic.c,v 1.7 1997/09/26 14:59:43 balay Exp bsmith $";
#endif

/*
    The most basic AOData routines. These store the 
  entire database on each processor. These are very simple, not that
  we do not even use a private data structure for AOData and the 
  private datastructure for AODataSegment is just used as a simple array.

*/

#include "src/ao/aoimpl.h"
#include "pinclude/pviewer.h"
#include "sys.h"

#undef __FUNC__  
#define __FUNC__ "AODataDestroy_Basic"
int AODataDestroy_Basic(PetscObject obj)
{
  int    i;
  AOData ao = (AOData) obj;

  for (i=0; i<ao->nc; i++ ) {
    PetscFree(ao->segments[i].data);
    PetscFree(ao->segments[i].name);
  }
  PetscFree(ao->segments);
  
  PLogObjectDestroy(ao);
  PetscHeaderDestroy(ao);
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "AODataView_Basic_Binary"
int AODataView_Basic_Binary(PetscObject obj,Viewer viewer)
{
  AOData          ao = (AOData) obj;
  int             ierr,i;
  int             fd;
  AODataSegment   *segment;
  char            paddedname[256];

  if (ao->nsegments != ao->nc) SETERRQ(1,1,"Not all segments set in AOData");

  ierr  = ViewerBinaryGetDescriptor(viewer,&fd); CHKERRQ(ierr);

  /* write out number of segments */
  ierr = PetscBinaryWrite(fd,&ao->nsegments,1,PETSC_INT,0);CHKERRQ(ierr);

  for ( i=0; i<ao->nc; i++ ) {
    segment = ao->segments + i;
    /* 
       Write out name of segment - use a fixed length for the name in the binary 
       file to make seeking easier
    */
    PetscMemzero(paddedname,256*sizeof(char));
    PetscStrncpy(paddedname,segment->name,255);
    ierr = PetscBinaryWrite(fd,paddedname,256,PETSC_CHAR,0);CHKERRQ(ierr);
    /* write out the number of segments, blocksize and data type */
    ierr = PetscBinaryWrite(fd,&segment->bs,1,PETSC_INT,0);CHKERRQ(ierr);
    ierr = PetscBinaryWrite(fd,&segment->N,1,PETSC_INT,0);CHKERRQ(ierr);
    ierr = PetscBinaryWrite(fd,&segment->datatype,1,PETSC_INT,0);CHKERRQ(ierr);

    /* write out the data */
    ierr = PetscBinaryWrite(fd,segment->data,segment->N*segment->bs,segment->datatype,0);CHKERRQ(ierr);
  }

  return 0;
}

#undef __FUNC__  
#define __FUNC__ "AODataView_Basic_ASCII"
int AODataView_Basic_ASCII(PetscObject obj,Viewer viewer)
{
  AOData          ao = (AOData) obj;
  int             ierr,format,i,j,k;
  FILE            *fd;
  char            *dt;
  AODataSegment      *segment;
  

  ierr = ViewerASCIIGetPointer(viewer,&fd); CHKERRQ(ierr);
  ierr = ViewerGetFormat(viewer,&format);
  if (format == VIEWER_FORMAT_ASCII_INFO) {
    for ( i=0; i<ao->nc; i++ ) {
      segment = ao->segments + i;
      ierr = PetscDataTypeGetName(segment->datatype,&dt); CHKERRQ(ierr);    
      fprintf(fd,"AOData Segment: %s\n",segment->name);
      fprintf(fd,"Number of elements in database segment %d blocksize %d datatype %s\n",
              segment->N,segment->bs,dt);
    }
    return 0;
  } 

  for ( i=0; i<ao->nc; i++ ) {
    segment = ao->segments + i;
    ierr = PetscDataTypeGetName(segment->datatype,&dt); CHKERRQ(ierr);    
    fprintf(fd,"AOData Segment: %s\n",segment->name);
    fprintf(fd,"Number of elements in database segment %d blocksize %d datatype %s\n",
            segment->N,segment->bs,dt);
    if (segment->datatype == PETSC_INT) {
      int *mdata = (int *) segment->data;
      for ( k=0; k<segment->N; k++ ) {
        fprintf(fd," %d: ",k);
        for ( j=0; j<segment->bs; j++ ) {
          fprintf(fd," %d ",mdata[k*segment->bs + j]);
        }
        fprintf(fd,"\n");
      }
    } else if (segment->datatype == PETSC_DOUBLE) {
      double *mdata = (double *) segment->data;
      for ( k=0; k<segment->N; k++ ) {
        fprintf(fd," %d: ",i);
        for ( j=0; j<segment->bs; j++ ) {
          fprintf(fd," %18.16e ",mdata[k*segment->bs + j]);
        }
        fprintf(fd,"\n");
      }
    } else if (segment->datatype == PETSC_SCALAR) {
      Scalar *mdata = (Scalar *) segment->data;
      for ( k=0; k<segment->N; k++ ) {
        fprintf(fd," %d: ",k);
        for ( j=0; j<segment->bs; j++ ) {
#if !defined(PETSC_COMPLEX)
          fprintf(fd," %18.16e ",mdata[k*segment->bs + j]);
#else
          Scalar x = mdata[k*segment->bs + j];
          if (imag(x) > 0.0) {
            fprintf(fd," %18.16e + %18.16e i \n",real(x),imag(x));
          } else if (imag(x) < 0.0) {
            fprintf(fd," %18.16e - %18.16e i \n",real(x),-imag(x));
          } else {
            fprintf(fd," %18.16e \n",real(x));
          }
#endif
        }
        fprintf(fd,"\n");
      }
    } else {
      SETERRQ(1,1,"Unknown PETSc data format");
    }
  }
  return 0;
}


#undef __FUNC__  
#define __FUNC__ "AODataView_Basic"
int AODataView_Basic(PetscObject obj,Viewer viewer)
{
  AOData          ao = (AOData) obj;
  int             rank,ierr;
  ViewerType      vtype;

  MPI_Comm_rank(ao->comm,&rank); if (rank) return 0;

  if (!viewer) {
    viewer = VIEWER_STDOUT_SELF; 
  }

  ierr = ViewerGetType(viewer,&vtype); CHKERRQ(ierr);
  if (vtype  == ASCII_FILE_VIEWER || vtype == ASCII_FILES_VIEWER) { 
    ierr = AODataView_Basic_ASCII(obj,viewer); CHKERRQ(ierr);
  } else if (vtype == BINARY_FILE_VIEWER) {
    ierr = AODataView_Basic_Binary(obj,viewer); CHKERRQ(ierr);
  }

  return 0;
}

int AODataAdd_Basic(AOData aodata,char *name,int bs,int n,int *keys,void *data,PetscDataType dtype)
{
  AODataSegment    *segment;
  int              N,size,rank,flg1,ierr,*lens,i,*disp,*akeys,datasize,*fkeys,len;
  MPI_Datatype     mtype;
  char             *adata,*fdata;
  MPI_Comm         comm = aodata->comm;

  segment = aodata->segments + aodata->nc++; 

  segment->bs       = bs;
  segment->datatype = dtype;

  /* transmit all lengths to all processors */
  MPI_Comm_size(comm,&size);
  MPI_Comm_rank(comm,&rank);
  lens = (int *) PetscMalloc( 2*size*sizeof(int) ); CHKPTRQ(lens);
  disp = lens + size;
  MPI_Allgather(&n,1,MPI_INT,lens,1,MPI_INT,comm);
  N =  0;
  for ( i=0; i<size; i++ ) {
    disp[i]  = N;
    N       += lens[i];
  }
  segment->N = N;

  ierr = PetscDataTypeToMPIDataType(dtype,&mtype);CHKERRQ(ierr);
  ierr = PetscDataTypeGetSize(dtype,&datasize); CHKERRQ(ierr);

  /*
    Allocate space for all keys and all data 
  */
  akeys = (int *) PetscMalloc((segment->N+1)*sizeof(int)); CHKPTRQ(akeys);
  adata = (char *) PetscMalloc((N*bs+1)*datasize); CHKPTRQ(adata);


  MPI_Allgatherv(keys,n,MPI_INT,akeys,lens,disp,MPI_INT,comm);
  for ( i=0; i<size; i++ ) {
    disp[i] *= bs;
    lens[i] *= bs;
  }
  MPI_Allgatherv(data,n*bs,mtype,adata,lens,disp,mtype,comm);
  PetscFree(lens);

  /*
    Now we have all the keys and data we need to put it in order
  */
  fkeys = (int *) PetscMalloc((segment->N+1)*sizeof(int)); CHKPTRQ(fkeys);
  PetscMemzero(fkeys, (segment->N+1)*sizeof(int));
  fdata = (char *) PetscMalloc((N*bs+1)*datasize); CHKPTRQ(fdata);

  for ( i=0; i<N; i++ ) {
    if (fkeys[akeys[i]] != 0) {
      SETERRQ(1,1,"Duplicate key");
    }
    if (fkeys[akeys[i]] >= N) {
      SETERRQ(1,1,"Key out of range");
    }
    fkeys[akeys[i]] = 1;
    PetscMemcpy(fdata+i*bs*datasize,adata+i*bs*datasize,bs*datasize);
  }
  for ( i=0; i<N; i++ ) {
    if (!fkeys[i]) {
      SETERRQ(1,1,"Missing key");
    }
  }
  PetscFree(akeys);
  PetscFree(adata);
  PetscFree(fkeys);

  segment->data = (void *) fdata;

  len        = PetscStrlen(name);
  segment->name = (char *) PetscMalloc((len+1)*sizeof(char));CHKPTRQ(segment->name);
  PetscStrcpy(segment->name,name);

  ierr = OptionsHasName(PETSC_NULL,"-ao_data_view",&flg1); CHKERRQ(ierr);
  if (flg1 && aodata->nc == aodata->nsegments) {
    ierr = AODataView(aodata,VIEWER_STDOUT_(comm)); CHKERRQ(ierr);
  }
  ierr = OptionsHasName(PETSC_NULL,"-ao_data_view_info",&flg1); CHKERRQ(ierr);
  if (flg1 && aodata->nc == aodata->nsegments) {
    ierr = ViewerPushFormat(VIEWER_STDOUT_(comm),VIEWER_FORMAT_ASCII_INFO,0);CHKERRQ(ierr);
    ierr = AODataView(aodata,VIEWER_STDOUT_(comm)); CHKERRQ(ierr);
    ierr = ViewerPopFormat(VIEWER_STDOUT_(comm));CHKERRQ(ierr);
  }

  return 0;
}

int AODataGet_Basic(AOData aodata,char *name,int n,int *keys,void **data)
{
  /*  AODataSegment    *segment; */

  /* find the correct segment */

  return 0;
}

int AODataRestore_Basic(AOData aodata,char *name,int n,int *keys,void **data)
{
  PetscFree(*data);
  return 0;
}

static struct _AODataOps myops = {AODataAdd_Basic,AODataGet_Basic,AODataRestore_Basic};

#undef __FUNC__  
#define __FUNC__ "AODataCreateBasic" 
/*@C
   AODataCreateBasic - Creates a 

   Input Parameters:
.  comm  - MPI communicator that is to share AO
.  n - total number of segments that will be added

   Output Parameter:
.  aoout - the new database

   Options Database Key:
$   -ao_data_view : call AODataView() at the conclusion of AODataAdd()
$   -ao_data_view_info : call AODataView() at the conclusion of AODataAdd()

.keywords: AOData, create

.seealso: AODataAdd(), AODataDestroy()
@*/
int AODataCreateBasic(MPI_Comm comm,int n,AOData *aoout)
{
  AOData        ao;

  *aoout = 0;
  PetscHeaderCreate(ao, _p_AOData,AODATA_COOKIE,AODATA_BASIC,comm,AODataDestroy,AODataView); 
  PLogObjectCreate(ao);
  PLogObjectMemory(ao,sizeof(struct _p_AOData) + n*sizeof(void *));

  PetscMemcpy(&ao->ops,&myops,sizeof(myops));
  ao->destroy  = AODataDestroy_Basic;
  ao->view     = AODataView_Basic;

  ao->nsegments          = n;
  ao->segments           = (AODataSegment*) PetscMalloc((n+1)*sizeof(AODataSegment));CHKPTRQ(ao->segments);
  ao->nc                 = 0;
  
  *aoout = ao; return 0;
}

#undef __FUNC__  
#define __FUNC__ "AODataLoadBasic" 
/*@C
   AODataLoadBasic - Loads a AO database from a file.

   Input Parameters:
.  viewer - the binary file containing the data

   Output Parameter:
.  aoout - the new database

   Options Database Key:
$   -ao_data_view : call AODataView() at the conclusion of AODataLoadBasic()
$   -ao_data_view_info : call AODataView() at the conclusion of AODataLoadBasic()

.keywords: AOData, create, load

.seealso: AODataAdd(), AODataDestroy(), AODataCreateBasic()
@*/
int AODataLoadBasic(Viewer viewer,AOData *aoout)
{
  AOData        ao;
  int           fd,nsegments,i,len,flg1,ierr,dsize;
  char          paddedname[256];
  AODataSegment *segment;
  MPI_Comm      comm;
  ViewerType    vtype;

  *aoout = 0;
  ierr = ViewerGetType(viewer,&vtype); CHKERRQ(ierr);
  if (vtype  != BINARY_FILE_VIEWER) {
    SETERRQ(1,1,"Viewer must be obtained from ViewerFileOpenBinary()");
  }

  ierr = PetscObjectGetComm((PetscObject)viewer,&comm); CHKERRQ(ierr);
  ierr = ViewerBinaryGetDescriptor(viewer,&fd); CHKERRQ(ierr);

  /* read in number of segments */
  ierr = PetscBinaryRead(fd,&nsegments,1,PETSC_INT);CHKERRQ(ierr);

  PetscHeaderCreate(ao, _p_AOData,AODATA_COOKIE,AODATA_BASIC,comm,AODataDestroy,AODataView); 
  PLogObjectCreate(ao);
  PLogObjectMemory(ao,sizeof(struct _p_AOData) + nsegments*sizeof(void *));

  PetscMemcpy(&ao->ops,&myops,sizeof(myops));
  ao->destroy  = AODataDestroy_Basic;
  ao->view     = AODataView_Basic;

  ao->nsegments          = nsegments;
  ao->segments           = (AODataSegment*) PetscMalloc((nsegments+1)*sizeof(AODataSegment));
    CHKPTRQ(ao->segments);
  ao->nc                 = nsegments;
  
  for ( i=0; i<nsegments; i++ ) {
    segment = ao->segments + i;

    /* read in segment name */
    ierr = PetscBinaryRead(fd,paddedname,256,PETSC_CHAR); CHKERRQ(ierr);
    len  = PetscStrlen(paddedname);
    segment->name = (char *) PetscMalloc((len+1)*sizeof(char));CHKPTRQ(segment->name);
    PetscStrcpy(segment->name,paddedname);

    /* read in segment blocksize, size and datatype */
    ierr = PetscBinaryRead(fd,&segment->bs,1,PETSC_INT);CHKERRQ(ierr);
    ierr = PetscBinaryRead(fd,&segment->N,1,PETSC_INT);CHKERRQ(ierr);
    ierr = PetscBinaryRead(fd,&segment->datatype,1,PETSC_INT);CHKERRQ(ierr);

    /* allocate the space for the data */
    ierr = PetscDataTypeGetSize(segment->datatype,&dsize); CHKERRQ(ierr);
    segment->data = (void *) PetscMalloc(segment->N*segment->bs*dsize);CHKPTRQ(segment->data);
    /* read in the data */
    ierr = PetscBinaryRead(fd,segment->data,segment->N*segment->bs,segment->datatype);CHKERRQ(ierr);
  }
  *aoout = ao; 

  ierr = OptionsHasName(PETSC_NULL,"-ao_data_view",&flg1); CHKERRQ(ierr);
  if (flg1) {
    ierr = AODataView(ao,VIEWER_STDOUT_(comm)); CHKERRQ(ierr);
  }
  ierr = OptionsHasName(PETSC_NULL,"-ao_data_view_info",&flg1); CHKERRQ(ierr);
  if (flg1) {
    ierr = ViewerPushFormat(VIEWER_STDOUT_(comm),VIEWER_FORMAT_ASCII_INFO,0);CHKERRQ(ierr);
    ierr = AODataView(ao,VIEWER_STDOUT_(comm)); CHKERRQ(ierr);
    ierr = ViewerPopFormat(VIEWER_STDOUT_(comm));CHKERRQ(ierr);
  }
  return 0;
}

