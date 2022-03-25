
static char help[] = "Tests vector scatter-gather operations.  Input arguments are\n\
  -n <length> : vector length\n\n";

#include <petscvec.h>

int main(int argc,char **argv)
{
  PetscInt       n   = 5,idx1[2] = {0,3},idx2[2] = {1,4};
  PetscScalar    one = 1.0,two = 2.0;
  Vec            x,y;
  IS             is1,is2;
  VecScatter     ctx = 0;

  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));

  /* create two vector */
  PetscCall(VecCreateSeq(PETSC_COMM_SELF,n,&x));
  PetscCall(VecDuplicate(x,&y));

  /* create two index sets */
  PetscCall(ISCreateGeneral(PETSC_COMM_SELF,2,idx1,PETSC_COPY_VALUES,&is1));
  PetscCall(ISCreateGeneral(PETSC_COMM_SELF,2,idx2,PETSC_COPY_VALUES,&is2));

  PetscCall(VecSet(x,one));
  PetscCall(VecSet(y,two));
  PetscCall(VecScatterCreate(x,is1,y,is2,&ctx));
  PetscCall(VecScatterBegin(ctx,x,y,INSERT_VALUES,SCATTER_FORWARD));
  PetscCall(VecScatterEnd(ctx,x,y,INSERT_VALUES,SCATTER_FORWARD));

  PetscCall(VecView(y,PETSC_VIEWER_STDOUT_SELF));

  PetscCall(VecScatterBegin(ctx,y,x,INSERT_VALUES,SCATTER_FORWARD));
  PetscCall(VecScatterEnd(ctx,y,x,INSERT_VALUES,SCATTER_FORWARD));
  PetscCall(VecScatterDestroy(&ctx));

  PetscCall(PetscPrintf(PETSC_COMM_SELF,"-------\n"));
  PetscCall(VecView(x,PETSC_VIEWER_STDOUT_SELF));

  PetscCall(ISDestroy(&is1));
  PetscCall(ISDestroy(&is2));

  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&y));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

     test:

TEST*/
