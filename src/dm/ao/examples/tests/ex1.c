/* $Id: ex1.c,v 1.16 2000/05/05 22:19:15 balay Exp bsmith $ */

static char help[] = "Demonstrates constructing an application ordering\n\n";

#include "petsc.h"
#include "petscao.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **argv)
{
  int      n = 5,ierr,rank,size,getpetsc[] = {0,3,4};
  int      getapp[] = {2,1,3,4};
  IS       ispetsc,isapp;
  AO       ao;

  PetscInitialize(&argc,&argv,(char*)0,help);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-n",&n,PETSC_NULL);CHKERRA(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRA(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRA(ierr);

  /* create the index sets */
  ierr = ISCreateStride(PETSC_COMM_WORLD,n,rank,size,&ispetsc);CHKERRA(ierr);
  ierr = ISCreateStride(PETSC_COMM_WORLD,n,n*rank,1,&isapp);CHKERRA(ierr);

  /* create the application ordering */
  ierr = AOCreateBasicIS(isapp,ispetsc,&ao);CHKERRA(ierr);

  ierr = ISDestroy(ispetsc);CHKERRA(ierr);
  ierr = ISDestroy(isapp);CHKERRA(ierr);

  ierr = AOView(ao,PETSC_VIEWER_STDOUT_WORLD);CHKERRA(ierr);

  ierr = AOPetscToApplication(ao,4,getapp);CHKERRA(ierr);
  ierr = AOApplicationToPetsc(ao,3,getpetsc);CHKERRA(ierr);
  ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d] 2,1,3,4 PetscToApplication %d %d %d %d\n",
          rank,getapp[0],getapp[1],getapp[2],getapp[3]);CHKERRA(ierr);
  ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d] 0,3,4 ApplicationToPetsc %d %d %d\n",
          rank,getpetsc[0],getpetsc[1],getpetsc[2]);CHKERRA(ierr);
  ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD);CHKERRA(ierr);

  ierr = AODestroy(ao);CHKERRA(ierr);
  PetscFinalize();
  return 0;
}
 


