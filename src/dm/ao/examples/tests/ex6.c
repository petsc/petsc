/*$Id: ex6.c,v 1.8 1999/10/24 14:04:01 bsmith Exp bsmith $*/

static char help[] = "Tests removing entries from an AOData \n\n";

#include "ao.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **argv)
{
  int         n,nglobal, bs = 2, *keys, *data,ierr,rank,size,i,start;
  double      *gd;
  AOData      aodata;

  PetscInitialize(&argc,&argv,(char*)0,help);
  ierr = OptionsGetInt(PETSC_NULL,"-n",&n,PETSC_NULL);CHKERRA(ierr);

  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRA(ierr); n = rank + 2;
  ierr = MPI_Allreduce(&n,&nglobal,1,MPI_INT,MPI_SUM,PETSC_COMM_WORLD);CHKERRA(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRA(ierr);

  /*
       Create a database with two sets of keys 
  */
  ierr = AODataCreateBasic(PETSC_COMM_WORLD,&aodata);CHKERRA(ierr);

  /*
       Put two segments in the first key and one in the second
  */
  ierr = AODataKeyAdd(aodata,"key1",PETSC_DECIDE,nglobal);CHKERRA(ierr);
  ierr = AODataKeyAdd(aodata,"key2",PETSC_DECIDE,nglobal);CHKERRA(ierr);

  /* allocate space for the keys each processor will provide */
  keys = (int *) PetscMalloc( n*sizeof(int) );CHKPTRA(keys);

  /*
     We assign the first set of keys (0 to 2) to processor 0, etc.
     This computes the first local key on each processor
  */
  ierr = MPI_Scan(&n,&start,1,MPI_INT,MPI_SUM,PETSC_COMM_WORLD);CHKERRA(ierr);
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
  ierr = PetscFree(data);CHKERRA(ierr);

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
       Use same data for second key and first segment 
  */
  ierr = AODataSegmentAdd(aodata,"key2","seg1",bs,n,keys,gd,PETSC_DOUBLE);CHKERRA(ierr); 
  ierr = PetscFree(gd);CHKERRA(ierr);
  ierr = PetscFree(keys);CHKERRA(ierr);

  /*
     View the database
  */
  ierr = AODataView(aodata,VIEWER_STDOUT_WORLD);CHKERRA(ierr);

  /*
       Remove a key and a single segment from the database
  */ 
  ierr = AODataKeyRemove(aodata,"key2");CHKERRA(ierr); 
  ierr = AODataSegmentRemove(aodata,"key1","seg1");CHKERRA(ierr);

  ierr = AODataView(aodata,VIEWER_STDOUT_WORLD);CHKERRA(ierr);

  ierr = AODataDestroy(aodata);CHKERRA(ierr);

  PetscFinalize();
  return 0;
}
 


