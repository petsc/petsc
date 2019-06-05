static char help[] = "Test MATMFFD for the rectangular case\n\n";

#include <petscmat.h>

static PetscErrorCode myF(void* ctx,Vec x,Vec y)
{
  const PetscScalar *ax;
  PetscScalar       *ay;
  PetscInt          i,j,m,n;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(x,&ax);CHKERRQ(ierr);
  ierr = VecGetArray(y,&ay);CHKERRQ(ierr);
  ierr = VecGetLocalSize(y,&m);CHKERRQ(ierr);
  ierr = VecGetLocalSize(x,&n);CHKERRQ(ierr);
  for (i=0;i<m;i++) {
    PetscScalar xx,yy;

    yy = 0.0;
    for (j=0;j<n;j++) {
      xx = PetscPowScalarInt(ax[j],i+1);
      yy += xx;
    }
    ay[i] = yy;
  }
  ierr = VecRestoreArray(y,&ay);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(x,&ax);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc,char **args)
{
  Mat            A,B;
  Vec            base;
  PetscInt       m = 3 ,n = 2;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL);CHKERRQ(ierr);
  ierr = MatCreateMFFD(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,m,n,&A);CHKERRQ(ierr);
  ierr = MatCreateVecs(A,&base,NULL);CHKERRQ(ierr);
  ierr = VecSet(base,2.0);CHKERRQ(ierr);
  ierr = MatMFFDSetFunction(A,myF,NULL);CHKERRQ(ierr);
  ierr = MatMFFDSetBase(A,base,NULL);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatComputeOperator(A,NULL,&B);CHKERRQ(ierr);
  ierr = VecDestroy(&base);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

    test:
      nsize: {{1 2 3 4}}

TEST*/
