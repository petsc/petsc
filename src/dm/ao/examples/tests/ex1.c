#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex1.c,v 1.10 1999/03/19 21:23:59 bsmith Exp balay $";
#endif

static char help[] = "Demonstrates constructing an application ordering\n\n";

#include "petsc.h"
#include "ao.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **argv)
{
  int      n = 5, ierr,flg,rank,size,getpetsc[] = {0,3,4};
  int      getapp[] = {2,1,3,4};
  IS       ispetsc,isapp;
  AO       ao;

  PetscInitialize(&argc,&argv,(char*)0,help);
  OptionsGetInt(PETSC_NULL,"-n",&n,&flg);
  MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
  MPI_Comm_size(PETSC_COMM_WORLD,&size);

  /* create the index sets */
  ierr = ISCreateStride(PETSC_COMM_WORLD,n,rank,size,&ispetsc);CHKERRA(ierr);
  ierr = ISCreateStride(PETSC_COMM_WORLD,n,n*rank,1,&isapp);CHKERRA(ierr);

  /* create the application ordering */
  ierr = AOCreateBasicIS(isapp,ispetsc,&ao);CHKERRA(ierr);

  ierr = ISDestroy(ispetsc);CHKERRA(ierr);
  ierr = ISDestroy(isapp);CHKERRA(ierr);

  ierr = AOView(ao,VIEWER_STDOUT_WORLD);CHKERRA(ierr);

  ierr = AOPetscToApplication(ao,4,getapp);CHKERRA(ierr);
  printf("[%d] 2,1,3,4 PetscToApplication %d %d %d %d\n",rank,getapp[0],
          getapp[1],getapp[2],getapp[3]);

  ierr = AOApplicationToPetsc(ao,3,getpetsc);CHKERRA(ierr);
  printf("[%d] 0,3,4 ApplicationToPetsc %d %d %d\n",rank,getpetsc[0],
          getpetsc[1],getpetsc[2]);

  fflush(stdout);
  ierr = AODestroy(ao);CHKERRA(ierr);
  PetscFinalize();
  return 0;
}
 


