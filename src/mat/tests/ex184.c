static char help[] = "Example of inverting a block diagonal matrix.\n"
"\n";

#include <petscmat.h>

/*T
    Concepts: Mat
T*/

int main(int argc, char **args)
{
    Mat            A,A_inv;
    PetscMPIInt    rank,size;
    PetscInt       M,m,bs,rstart,rend,j,x,y;
    PetscInt*      dnnz;
    PetscErrorCode ierr;
    PetscScalar    *v;
    Vec            X, Y;
    PetscReal      norm;

    ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
    ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
    ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);

    ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"ex184","Mat");CHKERRQ(ierr);
    M=8;
    ierr = PetscOptionsGetInt(NULL,NULL,"-mat_size",&M,NULL);CHKERRQ(ierr);
    bs=3;
    ierr = PetscOptionsGetInt(NULL,NULL,"-mat_block_size",&bs,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsEnd();CHKERRQ(ierr);

    ierr = MatCreate(PETSC_COMM_WORLD, &A);CHKERRQ(ierr);
    ierr = MatSetFromOptions(A);CHKERRQ(ierr);
    ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,M*bs,M*bs);CHKERRQ(ierr);
    ierr = MatSetBlockSize(A,bs);CHKERRQ(ierr);
    ierr = MatSetUp(A);CHKERRQ(ierr);
    ierr = MatGetLocalSize(A,&m,NULL);CHKERRQ(ierr);
    ierr = PetscMalloc1(m/bs,&dnnz);CHKERRQ(ierr);
    for (j = 0; j < m/bs; j++) {
        dnnz[j] = 1;
    }
    ierr = MatXAIJSetPreallocation(A,bs,dnnz,NULL,NULL,NULL);CHKERRQ(ierr);
    ierr = PetscFree(dnnz);CHKERRQ(ierr);

    ierr = PetscMalloc1(bs*bs,&v);CHKERRQ(ierr);
    ierr = MatGetOwnershipRange(A,&rstart,&rend);CHKERRQ(ierr);
    for (j = rstart/bs; j < rend/bs; j++) {
        for (x = 0; x < bs; x++) {
            for (y = 0; y < bs; y++) {
                if (x == y) {
                    v[y+bs*x] = 2*bs;
                } else {
                    v[y+bs*x] = -1 * (x < y) - 2 * (x > y);
                }
            }
        }
        ierr = MatSetValuesBlocked(A,1,&j,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
    }
    ierr = PetscFree(v);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

    /* check that A  = inv(inv(A)) */
    ierr = MatCreate(PETSC_COMM_WORLD,&A_inv);CHKERRQ(ierr);
    ierr = MatSetFromOptions(A_inv);CHKERRQ(ierr);
    ierr = MatInvertBlockDiagonalMat(A,A_inv);CHKERRQ(ierr);

    /* Test A_inv * A on a random vector */
    ierr = MatCreateVecs(A, &X, &Y);CHKERRQ(ierr);
    ierr = VecSetRandom(X, NULL);CHKERRQ(ierr);
    ierr = MatMult(A, X, Y);CHKERRQ(ierr);
    ierr = VecScale(X, -1);CHKERRQ(ierr);
    ierr = MatMultAdd(A_inv, Y, X, X);CHKERRQ(ierr);
    ierr = VecNorm(X, NORM_MAX, &norm);CHKERRQ(ierr);
    if (norm > PETSC_SMALL) {
        ierr = PetscPrintf(PETSC_COMM_WORLD,"Norm of error exceeds tolerance.\nInverse of block diagonal A\n");CHKERRQ(ierr);
        ierr = MatView(A_inv,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    }

    ierr = MatDestroy(&A);CHKERRQ(ierr);
    ierr = MatDestroy(&A_inv);CHKERRQ(ierr);
    ierr = VecDestroy(&X);CHKERRQ(ierr);
    ierr = VecDestroy(&Y);CHKERRQ(ierr);

    ierr = PetscFinalize();
    return ierr;
}

/*TEST
  test:
    suffix: seqaij
    args: -mat_type seqaij -mat_size 12 -mat_block_size 3
    nsize: 1
  test:
    suffix: seqbaij
    args: -mat_type seqbaij -mat_size 12 -mat_block_size 3
    nsize: 1
  test:
    suffix: mpiaij
    args: -mat_type mpiaij -mat_size 12 -mat_block_size 3
    nsize: 2
  test:
    suffix: mpibaij
    args: -mat_type mpibaij -mat_size 12 -mat_block_size 3
    nsize: 2
TEST*/
