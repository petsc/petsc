/*$Id: ex16.c,v 1.1 2000/06/09 21:02:00 bsmith Exp bsmith $*/

static char help[] = "Tests VecPack routines.\n\n";

#include "petscda.h"
#include "petscpf.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **argv)
{
  int     ierr,nredundant1 = 5,nredundant2 = 2,rank,i;
  Scalar  *redundant1,*redundant2;
  VecPack packer;
  Vec     global;
  PF      pf;

  PetscInitialize(&argc,&argv,(char*)0,help);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  redundant1 = (Scalar*)PetscMalloc(nredundant1*sizeof(Scalar));CHKPTRQ(redundant1);
  redundant2 = (Scalar*)PetscMalloc(nredundant2*sizeof(Scalar));CHKPTRQ(redundant2);

  ierr = VecPackCreate(PETSC_COMM_WORLD,&packer);CHKERRQ(ierr);
  ierr = VecPackAddArray(packer,nredundant1);CHKERRQ(ierr);
  ierr = VecPackAddArray(packer,nredundant2);CHKERRQ(ierr);

  ierr = VecPackCreateGlobalVector(packer,&global);CHKERRQ(ierr);
  ierr = PFCreate(PETSC_COMM_WORLD,1,1,&pf);CHKERRQ(ierr);
  ierr = PFSetType(pf,PFIDENTITY,PETSC_NULL);CHKERRQ(ierr);
  ierr = PFApplyVec(pf,PETSC_NULL,global);CHKERRQ(ierr);
  ierr = PFDestroy(pf);CHKERRQ(ierr);
  ierr = VecView(global,VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = VecPackScatter(packer,global,redundant1,redundant2);CHKERRQ(ierr);
  ierr = ViewerASCIISynchronizedPrintf(VIEWER_STDOUT_WORLD,"[%d] My part of redundant1 array\n",rank);CHKERRQ(ierr);
  ierr = PetscScalarView(nredundant1,redundant1,VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = ViewerASCIISynchronizedPrintf(VIEWER_STDOUT_WORLD,"[%d] My part of redundant2 array\n",rank);CHKERRQ(ierr);
  ierr = PetscScalarView(nredundant2,redundant2,VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  for (i=0; i<nredundant1; i++) redundant1[i] = (rank+2)*i;
  for (i=0; i<nredundant2; i++) redundant2[i] = (rank+10)*i;

  ierr = VecPackGather(packer,global,redundant1,redundant2);CHKERRQ(ierr);
  ierr = VecView(global,VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = VecDestroy(global);CHKERRQ(ierr);
  ierr = VecPackDestroy(packer);CHKERRQ(ierr);
  ierr = PetscFree(redundant1);CHKERRQ(ierr);
  ierr = PetscFree(redundant2);CHKERRQ(ierr);
  PetscFinalize();
  return 0;
}
 
