static char help[] = "Test MATMFFD for the rectangular case\n\n";

#include <petscmat.h>

static PetscErrorCode myF(void* ctx,Vec x,Vec y)
{
  const PetscScalar *ax;
  PetscScalar       *ay;
  PetscInt          i,j,m,n;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(x,&ax));
  PetscCall(VecGetArray(y,&ay));
  PetscCall(VecGetLocalSize(y,&m));
  PetscCall(VecGetLocalSize(x,&n));
  for (i=0;i<m;i++) {
    PetscScalar xx,yy;

    yy = 0.0;
    for (j=0;j<n;j++) {
      xx = PetscPowScalarInt(ax[j],i+1);
      yy += xx;
    }
    ay[i] = yy;
  }
  PetscCall(VecRestoreArray(y,&ay));
  PetscCall(VecRestoreArrayRead(x,&ax));
  PetscFunctionReturn(0);
}

int main(int argc,char **args)
{
  Mat            A,B;
  Vec            base;
  PetscInt       m = 3 ,n = 2;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL));
  PetscCall(MatCreateMFFD(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,m,n,&A));
  PetscCall(MatCreateVecs(A,&base,NULL));
  PetscCall(VecSet(base,2.0));
  PetscCall(MatMFFDSetFunction(A,myF,NULL));
  PetscCall(MatMFFDSetBase(A,base,NULL));
  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatComputeOperator(A,NULL,&B));
  PetscCall(VecDestroy(&base));
  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&B));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

    test:
      nsize: {{1 2 3 4}}

TEST*/
