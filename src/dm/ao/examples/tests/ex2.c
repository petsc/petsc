#ifndef lint
static char vcid[] = "$Id: ex2.c,v 1.1 1996/06/26 15:33:32 bsmith Exp bsmith $";
#endif

static char help[] = "Tests application ordering\n\n";

#include "petsc.h"
#include "ao.h"
#include <math.h>

int main(int argc,char **argv)
{
  int         n, ierr,flg,rank,size,*ispetsc,*isapp,start,N,i;
  AO          ao;

  PetscInitialize(&argc,&argv,(char*)0,help);
  OptionsGetInt(PETSC_NULL,"-n",&n,&flg);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank); n = rank + 2;
  MPI_Comm_size(MPI_COMM_WORLD,&size);

  /* create the orderings */
  ispetsc = (int *) PetscMalloc( 2*n*sizeof(int) ); CHKPTRA(ispetsc);
  isapp   = ispetsc + n;

  MPI_Scan(&n,&start,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
  MPI_Allreduce(&n,&N,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
  start -= n;

  for ( i=0; i<n; i++ ) {  
    ispetsc[i] = start + i;
    isapp[i]   = N - start - i - 1;
  }

  /* create the application ordering */
  ierr = AOCreateDebug(MPI_COMM_WORLD,n,isapp,ispetsc,&ao); CHKERRA(ierr);

  ierr = AOView(ao,STDOUT_VIEWER_WORLD); CHKERRA(ierr);

  /* check the mapping */
  ierr = AOPetscToApplication(ao,n,ispetsc); CHKERRA(ierr);
  for ( i=0; i<n; i++ ) {
    if (ispetsc[i] != isapp[i]) {
      fprintf(stdout,"[%d] Problem with mapping %d to %d\n",rank,i,ispetsc[i]);
    }
  }

  PetscFree(ispetsc);

  ierr = AODestroy(ao); CHKERRA(ierr);
  PetscFinalize();
  return 0;
}
 


