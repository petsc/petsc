/*$Id: MPI_Wtime.c,v 1.16 2001/03/23 23:25:45 balay Exp balay $*/

#include "petsc.h"

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  double x,y;
  int    ierr;
  
  PetscInitialize(&argc,&argv,0,0);
 /* To take care of paging effects */
  y = MPI_Wtime();

  x = MPI_Wtime();
  y = MPI_Wtime();
  y = MPI_Wtime();
  y = MPI_Wtime();
  y = MPI_Wtime();
  y = MPI_Wtime();
  y = MPI_Wtime();
  y = MPI_Wtime();
  y = MPI_Wtime();
  y = MPI_Wtime();
  y = MPI_Wtime();

  fprintf(stdout,"%-15s : %e sec\n","MPI_Wtime",(y-x)/10.0);
  y = MPI_Wtick();
  fprintf(stderr,"%-15s : %e sec\n","MPI_Wtick",y);

  x = MPI_Wtime();
  ierr = PetscSleep(10);CHKERRQ(ierr);
  y = MPI_Wtime();
  fprintf(stdout,"%-15s : %e sec - Slept for 10 sec \n","MPI_Wtime",(y-x));

  ierr = PetscFinalize();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
