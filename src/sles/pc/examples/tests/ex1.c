/*$Id: ex1.c,v 1.36 2001/03/23 23:23:25 balay Exp $*/

static char help[] = "Tests the creation of a PC context.\n\n";

#include "petscpc.h"

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  PC  pc;
  int ierr,n = 5;
  Vec u;
  Mat mat;

  PetscInitialize(&argc,&args,(char *)0,help);
  ierr = PCCreate(PETSC_COMM_WORLD,&pc);CHKERRQ(ierr);
  ierr = PCSetType(pc,PCNONE);CHKERRQ(ierr);

  /* Vector and matrix must be set before calling PCSetUp */
  ierr = VecCreateSeq(PETSC_COMM_SELF,n,&u);CHKERRQ(ierr);
  ierr = PCSetVector(pc,u);CHKERRQ(ierr);
  ierr = MatCreateSeqAIJ(PETSC_COMM_SELF,n,n,3,PETSC_NULL,&mat);CHKERRQ(ierr);
  ierr = PCSetOperators(pc,mat,mat,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = PCSetUp(pc);CHKERRQ(ierr);

  ierr = VecDestroy(u);	CHKERRQ(ierr);
  ierr = MatDestroy(mat);CHKERRQ(ierr);
  ierr = PCDestroy(pc);	CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
    


