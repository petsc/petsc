
#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: aodatabasic.c,v 1.3 1997/09/21 03:34:24 bsmith Exp bsmith $";
#endif

/*
    The most basic AOData routines. These store the 
  entire database on each processor.
*/

#include "src/ao/aoimpl.h"
#include "pinclude/pviewer.h"
#include "sys.h"

typedef struct {
  void *alldata;
} AOData_Basic;

#undef __FUNC__  
#define __FUNC__ "AODataDestroy_Basic"
int AODataDestroy_Basic(PetscObject obj)
{
  AOData       ao = (AOData) obj;
  AOData_Basic *aobasic = (AOData_Basic *) ao->data; 

  PetscFree(aobasic->alldata);
  PetscFree(ao->data); 
  PLogObjectDestroy(ao);
  PetscHeaderDestroy(ao);
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "AODataView_Basic_Binary"
int AODataView_Basic_Binary(PetscObject obj,Viewer viewer)
{
  AOData          ao = (AOData) obj;
  int             rank,ierr,format,i,j;
  int             fd;
  char            *dt;
  AOData_Basic    *aobasic = (AOData_Basic*) ao->data; 

  ierr  = ViewerBinaryGetDescriptor(viewer,&fd); CHKERRQ(ierr);

  /* write out the number of items, blocksize and data type */
  ierr = PetscBinaryWrite(fd,&ao->N,1,PETSC_INT,0);CHKERRQ(ierr);
  ierr = PetscBinaryWrite(fd,&ao->bs,1,PETSC_INT,0);CHKERRQ(ierr);
  ierr = PetscBinaryWrite(fd,&ao->datatype,1,PETSC_INT,0);CHKERRQ(ierr);

  /* write out the data */
  ierr = PetscBinaryWrite(fd,&aobasic->alldata,ao->N*ao->bs,PETSC_INT,0);CHKERRQ(ierr);

  return 0;
}

#undef __FUNC__  
#define __FUNC__ "AODataView_Basic_ASCII"
int AODataView_Basic_ASCII(PetscObject obj,Viewer viewer)
{
  AOData          ao = (AOData) obj;
  int             ierr,format,i,j;
  FILE            *fd;
  char            *dt;
  AOData_Basic    *aobasic = (AOData_Basic*) ao->data; 


  ierr = ViewerASCIIGetPointer(viewer,&fd); CHKERRQ(ierr);
  ierr = PetscDataTypeGetName(ao->datatype,&dt); CHKERRQ(ierr);
  fprintf(fd,"Number of elements in database %d blocksize %d datatype %s\n",ao->N,ao->bs,dt);
  ierr = ViewerGetFormat(viewer,&format);
  if (format == VIEWER_FORMAT_ASCII_INFO) {
    return 0;
  } 
  if (ao->datatype == PETSC_INT) {
    int *mdata = (int *) aobasic->alldata;
    for ( i=0; i<ao->N; i++ ) {
      fprintf(fd," %d: ",i);
      for ( j=0; j<ao->bs; j++ ) {
        fprintf(fd," %d ",mdata[i*ao->bs + j]);
      }
      fprintf(fd,"\n");
    }
  } else if (ao->datatype == PETSC_DOUBLE) {
    double *mdata = (double *) aobasic->alldata;
    for ( i=0; i<ao->N; i++ ) {
      fprintf(fd," %d: ",i);
      for ( j=0; j<ao->bs; j++ ) {
        fprintf(fd," %18.16e ",mdata[i*ao->bs + j]);
      }
      fprintf(fd,"\n");
    }
  } else if (ao->datatype == PETSC_SCALAR) {
    Scalar *mdata = (Scalar *) aobasic->alldata;
    for ( i=0; i<ao->N; i++ ) {
      fprintf(fd," %d: ",i);
      for ( j=0; j<ao->bs; j++ ) {
#if !defined(PETSC_COMPLEX)
        fprintf(fd," %18.16e ",mdata[i*ao->bs + j]);
#else
        Scalar x = mdata[i*ao->bs + j];
        if (imag(x > 0.0) {
          fprintf(fd," %18.16e + %18.16e i \n",real(x,imag(x);
        } else if (imag(x) < 0.0) {
          fprintf(fd," %18.16e - %18.16e i \n",real(x),-imag(x));
        } else {
          fprintf(fd," %18.16e \n",real(x);
        }
#endif
      }
      fprintf(fd,"\n");
    }
  } else {
    SETERRQ(1,1,"Unknown PETSc data format");
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


static struct _AODataOps myops = {0};

#undef __FUNC__  
#define __FUNC__ "AODataCreateBasic" 
/*@C
   AODataCreateBasic - Creates a 

   Input Parameters:
.  comm  - MPI communicator that is to share AO
.  bs    - number of items in basic data item
.  n     - number of local keys
.  keys  - integer keys of local data
.  data  - actual data
.  dtype - one of PETSC_INT, PETSC_DOUBLE, PETSC_SCALAR

   Output Parameter:
.  aoout - the new database

   Options Database Key:
$   -aodata_view : call AODataView() at the conclusion of AODataCreateBasic()

.keywords: AOData, create

.seealso: AODataCreateBasic(), AODataDestroy()
@*/
int AODataCreateBasic(MPI_Comm comm,int bs,int n,int *keys,void *data,PetscDataType dtype,AOData *aoout)
{
  AOData_Basic  *aobasic;
  AOData        ao;
  int           N,size,rank,flg1,ierr,*lens,i,*disp,*akeys,datasize,*fkeys;
  MPI_Datatype  mtype;
  char          *adata,*fdata;

  *aoout = 0;
  PetscHeaderCreate(ao, _p_AOData,AODATA_COOKIE,AODATA_BASIC,comm,AODataDestroy,AODataView); 
  PLogObjectCreate(ao);
  aobasic            = PetscNew(AOData_Basic);
  PLogObjectMemory(ao,sizeof(struct _p_AOData) + sizeof(AOData_Basic));

  PetscMemcpy(&ao->ops,&myops,sizeof(myops));
  ao->destroy  = AODataDestroy_Basic;
  ao->view     = AODataView_Basic;
  ao->data     = (void *)aobasic;
  ao->bs       = bs;
  ao->datatype = dtype;

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
  ao->N = N;

  ierr = PetscDataTypeToMPIDataType(dtype,&mtype);CHKERRQ(ierr);
  ierr = PetscDataTypeGetSize(dtype,&datasize); CHKERRQ(ierr);

  /*
    Allocate space for all keys and all data 
  */
  akeys = (int *) PetscMalloc((ao->N+1)*sizeof(int)); CHKPTRQ(akeys);
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
  fkeys = (int *) PetscMalloc((ao->N+1)*sizeof(int)); CHKPTRQ(fkeys);
  PetscMemzero(fkeys, (ao->N+1)*sizeof(int));
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

  aobasic->alldata = (void *) fdata;

  ierr = OptionsHasName(PETSC_NULL,"-ao_data_view",&flg1); CHKERRQ(ierr);
  if (flg1) {ierr = AODataView(ao,VIEWER_STDOUT_(comm)); CHKERRQ(ierr);}
  ierr = OptionsHasName(PETSC_NULL,"-ao_data_view_info",&flg1); CHKERRQ(ierr);
  if (flg1) {
    ierr = ViewerPushFormat(VIEWER_STDOUT_(comm),VIEWER_FORMAT_ASCII_INFO,0);CHKERRQ(ierr);
    ierr = AODataView(ao,VIEWER_STDOUT_(comm)); CHKERRQ(ierr);
    ierr = ViewerPopFormat(VIEWER_STDOUT_(comm));CHKERRQ(ierr);
  }

  *aoout = ao; return 0;
}

