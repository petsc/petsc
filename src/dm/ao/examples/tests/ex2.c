#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex2.c,v 1.8 1999/05/04 20:37:15 balay Exp balay $";
#endif

static char help[] = "Tests application ordering\n\n";

#include "petsc.h"
#include "ao.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **argv)
{
  int         n, ierr,flg,rank,size,*ispetsc,*isapp,start,N,i;
  AO          ao;

  PetscInitialize(&argc,&argv,(char*)0,help);
  OptionsGetInt(PETSC_NULL,"-n",&n,&flg);
  MPI_Comm_rank(PETSC_COMM_WORLD,&rank); n = rank + 2;
  MPI_Comm_size(PETSC_COMM_WORLD,&size);

  /* create the orderings */
  ispetsc = (int *) PetscMalloc( 2*n*sizeof(int) );CHKPTRA(ispetsc);
  isapp   = ispetsc + n;

  MPI_Scan(&n,&start,1,MPI_INT,MPI_SUM,PETSC_COMM_WORLD);
  MPI_Allreduce(&n,&N,1,MPI_INT,MPI_SUM,PETSC_COMM_WORLD);
  start -= n;

  for ( i=0; i<n; i++ ) {  
    ispetsc[i] = start + i;
    isapp[i]   = N - start - i - 1;
  }

  /* create the application ordering */
  ierr = AOCreateBasic(PETSC_COMM_WORLD,n,isapp,ispetsc,&ao);CHKERRA(ierr);

  ierr = AOView(ao,VIEWER_STDOUT_WORLD);CHKERRA(ierr);

  /* check the mapping */
  ierr = AOPetscToApplication(ao,n,ispetsc);CHKERRA(ierr);
  for ( i=0; i<n; i++ ) {
    if (ispetsc[i] != isapp[i]) {
      fprintf(stdout,"[%d] Problem with mapping %d to %d\n",rank,i,ispetsc[i]);
    }
  }

  ierr = PetscFree(ispetsc);CHKERRA(ierr);

  ierr = AODestroy(ao);CHKERRA(ierr);
  PetscFinalize();
  return 0;
}
 


