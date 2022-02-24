
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
  CHKERRQ(VecCreateSeq(PETSC_COMM_SELF,n,&x));
  CHKERRQ(VecDuplicate(x,&y));

  /* create two index sets */
  CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF,3,idx1,PETSC_COPY_VALUES,&is1));
  CHKERRQ(ISCreateStride(PETSC_COMM_SELF,3,0,2,&is2));

  CHKERRQ(VecSetValues(x,6,loc,vals,INSERT_VALUES));
  CHKERRQ(VecView(x,PETSC_VIEWER_STDOUT_SELF));
  CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"----\n"));
  CHKERRQ(VecSet(y,two));
  CHKERRQ(VecScatterCreate(x,is1,y,is2,&ctx));
  CHKERRQ(VecScatterBegin(ctx,x,y,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterEnd(ctx,x,y,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterDestroy(&ctx));

  CHKERRQ(VecView(y,PETSC_VIEWER_STDOUT_SELF));

  CHKERRQ(ISDestroy(&is1));
  CHKERRQ(ISDestroy(&is2));
  CHKERRQ(VecDestroy(&x));
  CHKERRQ(VecDestroy(&y));

  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:

TEST*/
