
static char help[] = "Demonstrates a scatter with a stride and general index set.\n\n";

#include <petscvec.h>

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscInt       n   = 6,idx1[3] = {0,1,2},loc[6] = {0,1,2,3,4,5};
  PetscScalar    two = 2.0,vals[6] = {10,11,12,13,14,15};
  Vec            x,y;
  IS             is1,is2;
  VecScatter     ctx = 0;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;

  /* create two vector */
  ierr = VecCreateSeq(PETSC_COMM_SELF,n,&x);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&y);CHKERRQ(ierr);

  /* create two index sets */
  ierr = ISCreateGeneral(PETSC_COMM_SELF,3,idx1,PETSC_COPY_VALUES,&is1);CHKERRQ(ierr);
  ierr = ISCreateStride(PETSC_COMM_SELF,3,0,2,&is2);CHKERRQ(ierr);

  ierr = VecSetValues(x,6,loc,vals,INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecView(x,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF,"----\n");CHKERRQ(ierr);
  ierr = VecSet(y,two);CHKERRQ(ierr);
  ierr = VecScatterCreate(x,is1,y,is2,&ctx);CHKERRQ(ierr);
  ierr = VecScatterBegin(ctx,x,y,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(ctx,x,y,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&ctx);CHKERRQ(ierr);

  ierr = VecView(y,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);

  ierr = ISDestroy(&is1);CHKERRQ(ierr);
  ierr = ISDestroy(&is2);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&y);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}



/*TEST

   test:


TEST*/
