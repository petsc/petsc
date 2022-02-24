
static char help[] = "Tests vector scatter-gather operations.  Input arguments are\n\
  -n <length> : vector length\n\n";

#include <petscvec.h>

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscInt       n   = 5,idx1[2] = {0,3},idx2[2] = {1,4};
  PetscScalar    one = 1.0,two = 2.0;
  Vec            x,y;
  IS             is1,is2;
  VecScatter     ctx = 0;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));

  /* create two vector */
  CHKERRQ(VecCreateSeq(PETSC_COMM_SELF,n,&x));
  CHKERRQ(VecDuplicate(x,&y));

  /* create two index sets */
  CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF,2,idx1,PETSC_COPY_VALUES,&is1));
  CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF,2,idx2,PETSC_COPY_VALUES,&is2));

  CHKERRQ(VecSet(x,one));
  CHKERRQ(VecSet(y,two));
  CHKERRQ(VecScatterCreate(x,is1,y,is2,&ctx));
  CHKERRQ(VecScatterBegin(ctx,x,y,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterEnd(ctx,x,y,INSERT_VALUES,SCATTER_FORWARD));

  CHKERRQ(VecView(y,PETSC_VIEWER_STDOUT_SELF));

  CHKERRQ(VecScatterBegin(ctx,y,x,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterEnd(ctx,y,x,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterDestroy(&ctx));

  CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"-------\n"));
  CHKERRQ(VecView(x,PETSC_VIEWER_STDOUT_SELF));

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
