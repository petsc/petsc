
#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: aodatabasic.c,v 1.2 1997/09/20 23:57:08 bsmith Exp bsmith $";
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
  AOData_Basic *aodebug = (AOData_Basic *) ao->data;
  PetscFree(ao->data); 
  PLogObjectDestroy(ao);
  PetscHeaderDestroy(ao);
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "AODataView_Basic"
int AODataView_Basic(PetscObject obj,Viewer viewer)
{
  AOData          ao = (AOData) obj;
  int             rank,ierr;
  ViewerType      vtype;
  FILE            *fd;
  AOData_Basic    *aodebug = (AOData_Basic*) ao->data;

  MPI_Comm_rank(ao->comm,&rank); if (rank) return 0;

  if (!viewer) {
    viewer = VIEWER_STDOUT_SELF; 
  }

  ierr = ViewerGetType(viewer,&vtype); CHKERRQ(ierr);
  if (vtype  == ASCII_FILE_VIEWER || vtype == ASCII_FILES_VIEWER) { 
    ierr = ViewerASCIIGetPointer(viewer,&fd); CHKERRQ(ierr);
    fprintf(fd,"Number of elements in database %d blocksize %d\n",ao->N,ao->bs);
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
  AOData_Basic  *aodebug;
  AOData        ao;
  int           N,size,rank,flg1,ierr,*lens,i,*disp;

  *aoout = 0;
  PetscHeaderCreate(ao, _p_AOData,AODATA_COOKIE,AODATA_BASIC,comm,AODataDestroy,AODataView); 
  PLogObjectCreate(ao);
  aodebug            = PetscNew(AOData_Basic);
  PLogObjectMemory(ao,sizeof(struct _p_AOData) + sizeof(AOData_Basic));

  PetscMemcpy(&ao->ops,&myops,sizeof(myops));
  ao->destroy  = AODataDestroy_Basic;
  ao->view     = AODataView_Basic;
  ao->data     = (void *)aodebug;
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
    disp[i] = N;
    N       += lens[i];
  }
  ao->N = N;

  ierr = OptionsHasName(PETSC_NULL,"-ao_data_view",&flg1); CHKERRQ(ierr);
  if (flg1) {ierr = AODataView(ao,VIEWER_STDOUT_SELF); CHKERRQ(ierr);}

  *aoout = ao; return 0;
}

