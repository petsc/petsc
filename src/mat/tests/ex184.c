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

    CHKERRQ(PetscInitialize(&argc,&args,(char*)0,help));
    CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
    CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));

    ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"ex184","Mat");CHKERRQ(ierr);
    M=8;
    CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-mat_size",&M,NULL));
    bs=3;
    CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-mat_block_size",&bs,NULL));
    ierr = PetscOptionsEnd();CHKERRQ(ierr);

    CHKERRQ(MatCreate(PETSC_COMM_WORLD, &A));
    CHKERRQ(MatSetFromOptions(A));
    CHKERRQ(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,M*bs,M*bs));
    CHKERRQ(MatSetBlockSize(A,bs));
    CHKERRQ(MatSetUp(A));
    CHKERRQ(MatGetLocalSize(A,&m,NULL));
    CHKERRQ(PetscMalloc1(m/bs,&dnnz));
    for (j = 0; j < m/bs; j++) {
        dnnz[j] = 1;
    }
    CHKERRQ(MatXAIJSetPreallocation(A,bs,dnnz,NULL,NULL,NULL));
    CHKERRQ(PetscFree(dnnz));

    CHKERRQ(PetscMalloc1(bs*bs,&v));
    CHKERRQ(MatGetOwnershipRange(A,&rstart,&rend));
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
        CHKERRQ(MatSetValuesBlocked(A,1,&j,1,&j,v,INSERT_VALUES));
    }
    CHKERRQ(PetscFree(v));
    CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

    /* check that A  = inv(inv(A)) */
    CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A_inv));
    CHKERRQ(MatSetFromOptions(A_inv));
    CHKERRQ(MatInvertBlockDiagonalMat(A,A_inv));

    /* Test A_inv * A on a random vector */
    CHKERRQ(MatCreateVecs(A, &X, &Y));
    CHKERRQ(VecSetRandom(X, NULL));
    CHKERRQ(MatMult(A, X, Y));
    CHKERRQ(VecScale(X, -1));
    CHKERRQ(MatMultAdd(A_inv, Y, X, X));
    CHKERRQ(VecNorm(X, NORM_MAX, &norm));
    if (norm > PETSC_SMALL) {
        CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Norm of error exceeds tolerance.\nInverse of block diagonal A\n"));
        CHKERRQ(MatView(A_inv,PETSC_VIEWER_STDOUT_WORLD));
    }

    CHKERRQ(MatDestroy(&A));
    CHKERRQ(MatDestroy(&A_inv));
    CHKERRQ(VecDestroy(&X));
    CHKERRQ(VecDestroy(&Y));

    CHKERRQ(PetscFinalize());
    return 0;
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
