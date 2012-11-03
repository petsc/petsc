
static char help[] ="Tests sequential and parallel MatMatMatMult() and MatPtAP(). Modified from ex96.c \n\
  -Mx <xg>, where <xg> = number of coarse grid points in the x-direction\n\
  -My <yg>, where <yg> = number of coarse grid points in the y-direction\n\
  -Mz <zg>, where <zg> = number of coarse grid points in the z-direction\n\
  -Npx <npx>, where <npx> = number of processors in the x-direction\n\
  -Npy <npy>, where <npy> = number of processors in the y-direction\n\
  -Npz <npz>, where <npz> = number of processors in the z-direction\n\n";

/*
    Example of usage: mpiexec -n 3 ./ex41 -Mx 10 -My 10 -Mz 10
*/

#include <petscdmda.h>
#include <../src/mat/impls/aij/seq/aij.h>
#include <../src/mat/impls/aij/mpi/mpiaij.h>

/* User-defined application contexts */
typedef struct {
   PetscInt   mx,my,mz;         /* number grid points in x, y and z direction */
   Vec        localX,localF;    /* local vectors with ghost region */
   DM         da;
   Vec        x,b,r;            /* global vectors */
   Mat        J;                /* Jacobian on grid */
} GridCtx;
typedef struct {
   GridCtx     fine;
   GridCtx     coarse;
   PetscInt    ratio;
   Mat         Ii;              /* interpolation from coarse to fine */
} AppCtx;

#define COARSE_LEVEL 0
#define FINE_LEVEL   1

/*
      Mm_ratio - ration of grid lines between fine and coarse grids.
*/
#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  AppCtx         user;
  PetscMPIInt    size,rank;
  PetscInt       m,n,M,N,i,nrows;
  PetscScalar    one = 1.0;
  PetscReal      fill=2.0;
  Mat            A,P,R,C;
  PetscScalar    *array;
  PetscRandom   rdm;
  PetscBool     Test_MatMatMatMult=PETSC_TRUE,Test_MatPtAP=PETSC_FALSE,Test_3D=PETSC_FALSE,flg;

  PetscInitialize(&argc,&argv,PETSC_NULL,help);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);

  /* Get size of fine grids and coarse grids */
  user.ratio = 2;
  user.coarse.mx = 2; user.coarse.my = 2; user.coarse.mz = 0;
  ierr = PetscOptionsGetInt(PETSC_NULL,"-Mx",&user.coarse.mx,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-My",&user.coarse.my,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-Mz",&user.coarse.mz,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-ratio",&user.ratio,PETSC_NULL);CHKERRQ(ierr);
  if (user.coarse.mz) Test_3D = PETSC_TRUE;

  user.fine.mx = user.ratio*(user.coarse.mx-1)+1;
  user.fine.my = user.ratio*(user.coarse.my-1)+1;
  user.fine.mz = user.ratio*(user.coarse.mz-1)+1;
 
  if (!rank){
    if (!Test_3D){
      ierr = PetscPrintf(PETSC_COMM_SELF,"coarse grids: %d %d; fine grids: %d %d\n",user.coarse.mx,user.coarse.my,user.fine.mx,user.fine.my);CHKERRQ(ierr);
    } else {
      ierr = PetscPrintf(PETSC_COMM_SELF,"coarse grids: %d %d %d; fine grids: %d %d %d\n",user.coarse.mx,user.coarse.my,user.coarse.mz,user.fine.mx,user.fine.my,user.fine.mz);CHKERRQ(ierr);
    }
  }

  /* Set up distributed array for fine grid */
  if (!Test_3D){
    ierr = DMDACreate2d(PETSC_COMM_WORLD, DMDA_BOUNDARY_NONE, DMDA_BOUNDARY_NONE,DMDA_STENCIL_STAR,user.fine.mx,
                    user.fine.my,PETSC_DECIDE,PETSC_DECIDE,1,1,PETSC_NULL,PETSC_NULL,&user.fine.da);CHKERRQ(ierr);
  } else {
    ierr = DMDACreate3d(PETSC_COMM_WORLD,DMDA_BOUNDARY_NONE,DMDA_BOUNDARY_NONE,DMDA_BOUNDARY_NONE,DMDA_STENCIL_STAR,
                    user.fine.mx,user.fine.my,user.fine.mz,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,
                    1,1,PETSC_NULL,PETSC_NULL,PETSC_NULL,&user.fine.da);CHKERRQ(ierr);
  }

  /* Create and set A at fine grids */
  ierr = DMCreateMatrix(user.fine.da,MATAIJ,&A);CHKERRQ(ierr);
  ierr = MatGetLocalSize(A,&m,&n);CHKERRQ(ierr);
  ierr = MatGetSize(A,&M,&N);CHKERRQ(ierr);

  /* set val=one to A ( replace with random values! */
  ierr = PetscRandomCreate(PETSC_COMM_WORLD,&rdm);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rdm);CHKERRQ(ierr);
  if (size == 1) {
    const PetscInt *ia,*ja;
    ierr = MatGetRowIJ(A,0,PETSC_FALSE,PETSC_FALSE,&nrows,&ia,&ja,&flg);
    if (flg){
      ierr = MatSeqAIJGetArray(A,&array);CHKERRQ(ierr);
      for (i=0; i<ia[nrows]; i++) array[i] = one;
      ierr = MatSeqAIJRestoreArray(A,&array);CHKERRQ(ierr);
    }
    ierr = MatRestoreRowIJ(A,0,PETSC_FALSE,PETSC_FALSE,&nrows,&ia,&ja,&flg);
  } else {
    Mat_MPIAIJ *aij = (Mat_MPIAIJ*)A->data;
    Mat_SeqAIJ *a=(Mat_SeqAIJ*)(aij->A)->data, *b=(Mat_SeqAIJ*)(aij->B)->data;
    /* A_part */
    for (i=0; i<a->i[m]; i++) a->a[i] = one;
    /* B_part */
    for (i=0; i<b->i[m]; i++) b->a[i] = one;

  }
  if (!rank) printf("A:\n");
  ierr = MatView(A, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr); 

  /* Set up distributed array for coarse grid */
  if (!Test_3D){
    ierr = DMDACreate2d(PETSC_COMM_WORLD, DMDA_BOUNDARY_NONE, DMDA_BOUNDARY_NONE,DMDA_STENCIL_STAR,user.coarse.mx,
                    user.coarse.my,PETSC_DECIDE,PETSC_DECIDE,1,1,PETSC_NULL,PETSC_NULL,&user.coarse.da);CHKERRQ(ierr);
  } else {
    ierr = DMDACreate3d(PETSC_COMM_WORLD,DMDA_BOUNDARY_NONE,DMDA_BOUNDARY_NONE,DMDA_BOUNDARY_NONE,DMDA_STENCIL_STAR,
                    user.coarse.mx,user.coarse.my,user.coarse.mz,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,
                    1,1,PETSC_NULL,PETSC_NULL,PETSC_NULL,&user.coarse.da);CHKERRQ(ierr);
  }

  /* Create interpolation between the fine and coarse grids */
  ierr = DMCreateInterpolation(user.coarse.da,user.fine.da,&P,PETSC_NULL);CHKERRQ(ierr);
  if (!rank) printf("P:\n");
  ierr = MatView(P, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr); 

  /* Get R = P^T */
  ierr = MatTranspose(P,MAT_INITIAL_MATRIX,&R);CHKERRQ(ierr);
  if (!rank) printf("R:\n");
  ierr = MatView(R, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  /* D = R*A*P */
  ierr = MatMatMatMult(R,A,P,MAT_INITIAL_MATRIX,fill,&C);CHKERRQ(ierr);

  /* Clean up */
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&rdm);CHKERRQ(ierr);
  ierr = DMDestroy(&user.fine.da);CHKERRQ(ierr);
  ierr = DMDestroy(&user.coarse.da);CHKERRQ(ierr);
  ierr = MatDestroy(&P);CHKERRQ(ierr);
  ierr = MatDestroy(&R);CHKERRQ(ierr);
  /* ierr = MatDestroy(&C);CHKERRQ(ierr); */
  ierr = PetscFinalize();
  return 0;
}
