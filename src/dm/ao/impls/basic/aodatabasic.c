
#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: aodatabasic.c,v 1.5 1997/09/26 14:55:35 balay Exp balay $";
#endif

/*
    The most basic AOData routines. These store the 
  entire database on each processor. These are very simple, not that
  we do not even use a private data structure for AOData and the 
  private datastructure for AODataItem is just used as a simple array.

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
    PetscFree(ao->items[i].data);
  }
  PetscFree(ao->items);
  
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
  AODataItem      *item;

  ierr  = ViewerBinaryGetDescriptor(viewer,&fd); CHKERRQ(ierr);

  for ( i=0; i<ao->nc; i++ ) {
    item = ao->items + i;
    /* write out the number of items, blocksize and data type */
    ierr = PetscBinaryWrite(fd,&item->N,1,PETSC_INT,0);CHKERRQ(ierr);
    ierr = PetscBinaryWrite(fd,&item->bs,1,PETSC_INT,0);CHKERRQ(ierr);
    ierr = PetscBinaryWrite(fd,&item->datatype,1,PETSC_INT,0);CHKERRQ(ierr);

    /* write out the data */
    ierr = PetscBinaryWrite(fd,&item->data,item->N*item->bs,PETSC_INT,0);CHKERRQ(ierr);
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
  AODataItem      *item;
  

  ierr = ViewerASCIIGetPointer(viewer,&fd); CHKERRQ(ierr);
  ierr = ViewerGetFormat(viewer,&format);
  if (format == VIEWER_FORMAT_ASCII_INFO) {
    return 0;
  } 

  for ( i=0; i<ao->nc; i++ ) {
    item = ao->items + i;
    ierr = PetscDataTypeGetName(item->datatype,&dt); CHKERRQ(ierr);    
    fprintf(fd,"Number of elements in database %d blocksize %d datatype %s\n",item->N,item->bs,dt);
    if (item->datatype == PETSC_INT) {
      int *mdata = (int *) item->data;
      for ( k=0; k<item->N; k++ ) {
        fprintf(fd," %d: ",k);
        for ( j=0; j<item->bs; j++ ) {
          fprintf(fd," %d ",mdata[k*item->bs + j]);
        }
        fprintf(fd,"\n");
      }
    } else if (item->datatype == PETSC_DOUBLE) {
      double *mdata = (double *) item->data;
      for ( k=0; k<item->N; k++ ) {
        fprintf(fd," %d: ",i);
        for ( j=0; j<item->bs; j++ ) {
          fprintf(fd," %18.16e ",mdata[k*item->bs + j]);
        }
        fprintf(fd,"\n");
      }
    } else if (item->datatype == PETSC_SCALAR) {
      Scalar *mdata = (Scalar *) item->data;
      for ( k=0; k<item->N; k++ ) {
        fprintf(fd," %d: ",k);
        for ( j=0; j<item->bs; j++ ) {
#if !defined(PETSC_COMPLEX)
          fprintf(fd," %18.16e ",mdata[k*item->bs + j]);
#else
          Scalar x = mdata[k*item->bs + j];
          if (imag(x) > 0.0) {
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

int AODataAdd_Basic(AOData aodata,int bs,int n,int *keys,void *data,PetscDataType dtype)
{
  AODataItem       *item;
  int              N,size,rank,flg1,ierr,*lens,i,*disp,*akeys,datasize,*fkeys;
  MPI_Datatype     mtype;
  char             *adata,*fdata;
  MPI_Comm         comm = aodata->comm;

  item = aodata->items + aodata->nc++; 

  item->bs       = bs;
  item->datatype = dtype;

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
  item->N = N;

  ierr = PetscDataTypeToMPIDataType(dtype,&mtype);CHKERRQ(ierr);
  ierr = PetscDataTypeGetSize(dtype,&datasize); CHKERRQ(ierr);

  /*
    Allocate space for all keys and all data 
  */
  akeys = (int *) PetscMalloc((item->N+1)*sizeof(int)); CHKPTRQ(akeys);
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
  fkeys = (int *) PetscMalloc((item->N+1)*sizeof(int)); CHKPTRQ(fkeys);
  PetscMemzero(fkeys, (item->N+1)*sizeof(int));
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

  item->data = (void *) fdata;

  ierr = OptionsHasName(PETSC_NULL,"-ao_data_view",&flg1); CHKERRQ(ierr);
  if (flg1 && aodata->nc == aodata->nitems) {ierr = AODataView(aodata,VIEWER_STDOUT_(comm)); CHKERRQ(ierr);}
  ierr = OptionsHasName(PETSC_NULL,"-ao_data_view_info",&flg1); CHKERRQ(ierr);
  if (flg1 && aodata->nc == aodata->nitems) {
    ierr = ViewerPushFormat(VIEWER_STDOUT_(comm),VIEWER_FORMAT_ASCII_INFO,0);CHKERRQ(ierr);
    ierr = AODataView(aodata,VIEWER_STDOUT_(comm)); CHKERRQ(ierr);
    ierr = ViewerPopFormat(VIEWER_STDOUT_(comm));CHKERRQ(ierr);
  }

  return 0;
}

static struct _AODataOps myops = {AODataAdd_Basic};

#undef __FUNC__  
#define __FUNC__ "AODataCreateBasic" 
/*@C
   AODataCreateBasic - Creates a 

   Input Parameters:
.  comm  - MPI communicator that is to share AO
.  n - total number of items that will be added

   Output Parameter:
.  aoout - the new database

   Options Database Key:
$   -aodata_view : call AODataView() at the conclusion of AODataAdd()

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

  ao->nitems          = n;
  ao->items           = (AODataItem*) PetscMalloc( (n+1)*sizeof(AODataItem) ); CHKPTRQ(ao->items);
  ao->items->datatype = (PetscDataType) -1;
  ao->nc              = 0;
  
  *aoout = ao; return 0;
}

