#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex3.c,v 1.1 1997/09/20 03:33:58 bsmith Exp bsmith $";
#endif

static char help[] = "Tests AOData \n\n";

#include "petsc.h"
#include "ao.h"
#include <math.h>

int main(int argc,char **argv)
{
  int         n, bs = 2, *keys, *data,ierr,flg,rank,size,i,start;
  AOData      aodata;

  PetscInitialize(&argc,&argv,(char*)0,help);
  OptionsGetInt(PETSC_NULL,"-n",&n,&flg);
  MPI_Comm_rank(PETSC_COMM_WORLD,&rank); n = rank + 2;
  MPI_Comm_size(PETSC_COMM_WORLD,&size);

  keys = (int *) PetscMalloc( n*sizeof(int) );CHKPTRA(keys);
  data = (int *) PetscMalloc( 2*n*sizeof(int) );CHKPTRA(data);

  /*
     We assign the first set of keys (0 to 2) to processor 0, etc.
     This computes the first local key on each processor
  */
  MPI_Scan(&n,&start,1,MPI_INT,MPI_SUM,PETSC_COMM_WORLD);
  start -= n;

  for ( i=0; i<n; i++ ) {
    keys[i]     = start + i;
    data[2*i]   = -(start + i);
    data[2*i+1] = -(start + i) - 10000;
  }

  ierr = AODataCreateBasic(PETSC_COMM_WORLD,bs,n,keys,data,PETSC_INT,&aodata);CHKERRA(ierr);

  ierr = AODataView(aodata,VIEWER_STDOUT_WORLD); CHKERRA(ierr);

  ierr = AODataDestroy(aodata); CHKERRA(ierr);
  PetscFinalize();
  return 0;
}
 


