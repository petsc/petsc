#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex3.c,v 1.10 1998/12/03 04:06:51 bsmith Exp bsmith $";
#endif

static char help[] = "Tests AOData \n\n";

#include "ao.h"
#include "bitarray.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **argv)
{
  int         n = 2,nglobal, bs = 2, *keys, *data,ierr,flg,rank,size,i,start;
  double      *gd;
  AOData      aodata;
  Viewer      binary;
  BT          ld;

  PetscInitialize(&argc,&argv,(char*)0,help);
  OptionsGetInt(PETSC_NULL,"-n",&n,&flg);

  MPI_Comm_rank(PETSC_COMM_WORLD,&rank); n = n + rank;
  MPI_Allreduce(&n,&nglobal,1,MPI_INT,MPI_SUM,PETSC_COMM_WORLD);
  MPI_Comm_size(PETSC_COMM_WORLD,&size);

  /*
       Create a database with two sets of keys 
  */
  ierr = AODataCreateBasic(PETSC_COMM_WORLD,&aodata);CHKERRA(ierr);
  ierr = AODataKeyAdd(aodata,"key1",PETSC_DECIDE,nglobal); CHKERRA(ierr);
  ierr = AODataKeyAdd(aodata,"key2",PETSC_DECIDE,nglobal); CHKERRA(ierr);

  /* allocate space for the keys each processor will provide */
  keys = (int *) PetscMalloc( n*sizeof(int) );CHKPTRA(keys);

  /*
     We assign the first set of keys (0 to 2) to processor 0, etc.
     This computes the first local key on each processor
  */
  MPI_Scan(&n,&start,1,MPI_INT,MPI_SUM,PETSC_COMM_WORLD);
  start -= n;

  for ( i=0; i<n; i++ ) {
    keys[i]     = start + i;
  }

  /* 
      Allocate data for the first key and first segment 
  */
  data = (int *) PetscMalloc( bs*n*sizeof(int) );CHKPTRA(data);
  for ( i=0; i<n; i++ ) {
    data[2*i]   = -(start + i);
    data[2*i+1] = -(start + i) - 10000;
  }
  ierr = AODataSegmentAdd(aodata,"key1","seg1",bs,n,keys,data,PETSC_INT);CHKERRA(ierr); 
  PetscFree(data);

  /*
      Allocate data for first key and second segment 
  */
  bs   = 3;
  gd   = (double *) PetscMalloc( bs*n*sizeof(double) );CHKPTRA(gd);
  for ( i=0; i<n; i++ ) {
    gd[3*i]   = -(start + i);
    gd[3*i+1] = -(start + i) - 10000;
    gd[3*i+2] = -(start + i) - 100000;
  }
  ierr = AODataSegmentAdd(aodata,"key1","seg2",bs,n,keys,gd,PETSC_DOUBLE);CHKERRA(ierr); 

  /*
      Allocate data for first key and third segment 
  */
  bs   = 1;
  ierr = BTCreate(n,ld);CHKERRA(ierr);
  for ( i=0; i<n; i++ ) {
    if (i % 2) BTSet(ld,i);
  }
  ierr = AODataSegmentAdd(aodata,"key1","seg3",bs,n,keys,ld,PETSC_LOGICAL);CHKERRA(ierr); 
  BTDestroy(ld);

  /*
       Use same data for second key and first segment 
  */
  bs   = 3;
  ierr = AODataSegmentAdd(aodata,"key2","seg1",bs,n,keys,gd,PETSC_DOUBLE);CHKERRA(ierr); 
  PetscFree(gd);
  PetscFree(keys);

  ierr = AODataView(aodata,VIEWER_STDOUT_WORLD);CHKERRA(ierr);

  /*
        Save the database to a file
  */
  ierr = ViewerBinaryOpen(PETSC_COMM_WORLD,"dataoutput",BINARY_CREATE,&binary);CHKERRA(ierr);
  ierr = AODataView(aodata,binary);CHKERRA(ierr);
  ierr = ViewerDestroy(binary); CHKERRA(ierr);
 
  ierr = AODataDestroy(aodata); CHKERRA(ierr);

  PetscFinalize();
  return 0;
}
 


