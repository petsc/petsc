#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex5.c,v 1.1 1997/10/23 01:39:53 bsmith Exp bsmith $";
#endif

static char help[] = "Tests AODataRemap \n\n";

#include "petsc.h"
#include "ao.h"
#include <math.h>

int main(int argc,char **argv)
{
  int         n,nglobal, bs = 1, *keys, *data,ierr,flg,rank,size,i,start;
  AOData      aodata;

  PetscInitialize(&argc,&argv,(char*)0,help);
  OptionsGetInt(PETSC_NULL,"-n",&n,&flg);

  MPI_Comm_rank(PETSC_COMM_WORLD,&rank); n = rank + 2;
  MPI_Allreduce(&n,&nglobal,1,MPI_INT,MPI_SUM,PETSC_COMM_WORLD);
  MPI_Comm_size(PETSC_COMM_WORLD,&size);

  /*
       Create a database with one  key and one segment
  */
  ierr = AODataCreateBasic(PETSC_COMM_WORLD,1,&aodata);CHKERRA(ierr);

  /*
       Put one segment in the key
  */
  ierr = AODataKeyAdd(aodata,"key1",PETSC_DECIDE,nglobal,1); CHKERRA(ierr);

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
    data[i]   = start + i +1; /* the data is the neighbor to the right */
  }
  data[n-1] = 0; /* make it periodic */
  ierr = AODataSegmentAdd(aodata,"key1","key1",bs,n,keys,data,PETSC_INT);CHKERRA(ierr); 
  PetscFree(data);
  PetscFree(keys);

  /*
        View the database
  */
  ierr = AODataView(aodata,VIEWER_STDOUT_WORLD);CHKERRA(ierr);
 
  /*
         Remap the database so that i -> nglobal - i - 1
  */
  ierr = AODataView(aodata,VIEWER_STDOUT_WORLD);CHKERRA(ierr);
  ierr = AODataDestroy(aodata); CHKERRA(ierr);

  PetscFinalize();
  return 0;
}
 


