#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex7.c,v 1.5 1999/05/04 20:37:15 balay Exp bsmith $";
#endif

static char help[] = "Demonstrates constructing an application ordering\n\n";

#include "ao.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **argv)
{
  int      n = 5, ierr,flg,rank,size;
  IS       ispetsc,isapp;
  AO       ao;

  PetscInitialize(&argc,&argv,(char*)0,help);
  ierr = OptionsGetInt(PETSC_NULL,"-n",&n,&flg);CHKERRA(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRA(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRA(ierr);

  /* create the index sets */
  ierr = ISCreateStride(PETSC_COMM_WORLD,n,rank,size,&ispetsc);CHKERRA(ierr);
  ierr = ISCreateStride(PETSC_COMM_WORLD,n,n*rank,1,&isapp);CHKERRA(ierr);

  /* create the application ordering */
  ierr = AOCreateBasicIS(isapp,ispetsc,&ao);CHKERRA(ierr);


  ierr = AOView(ao,VIEWER_STDOUT_WORLD);CHKERRA(ierr);

  ierr = ISView(ispetsc,VIEWER_STDOUT_WORLD);CHKERRA(ierr);
  ierr = ISView(isapp,VIEWER_STDOUT_WORLD);CHKERRA(ierr);
  ierr = AOPetscToApplicationIS(ao,ispetsc);CHKERRA(ierr);
  ierr = ISView(isapp,VIEWER_STDOUT_WORLD);CHKERRA(ierr);
  ierr = ISView(ispetsc,VIEWER_STDOUT_WORLD);CHKERRA(ierr);


  ierr = ISDestroy(ispetsc);CHKERRA(ierr);
  ierr = ISDestroy(isapp);CHKERRA(ierr);

  fflush(stdout);
  ierr = AODestroy(ao);CHKERRA(ierr);
  PetscFinalize();
  return 0;
}
 


