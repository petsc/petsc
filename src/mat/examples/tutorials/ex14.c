
static char help[] = "Basic test of MatFwkAIJ: a block matrix with an AIJ-like datastructure keeping track of nonzero blocks.\
Each block is a matrix of (generally) any type.\n\n";

/* 
  Include "petscmat.h" so that we can use matrices.
  automatically includes:
     petscsys.h    - base PETSc routines   petscvec.h    - vectors
     petscmat.h    - matrices
     petscis.h     - index sets            petscviewer.h - viewers               
*/
#include "petscmat.h"

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  Mat M; /* Matrix framekwork */
  Mat B; /* Framework block */
  PetscInt e,n,N;
  PetscReal h;
  PetscInt llow, lhigh;
  PetscTruth flag;
  Vec v,V, W;
  IS isloc, isglob;
  PetscInt size;
  PetscInt id, *idx;
  VecScatter scatter, gather;
  Mat scattermat, gathermat;
  PetscInt i, *blocks;
  PetscReal third = 1.0/3.0, sixth = 1.0/6.0;
  PetscScalar values[4] = {third, sixth, sixth,third};
  PetscInt ii[2] = {0,1}, jj[2] = {0,1};
  PetscErrorCode ierr;

  PetscInitialize(&argc,&args,(char *)0,help);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD, &size); CHKERRQ(ierr);

  /* Construct a 1D mesh with e local elements.  Use the P1 function space on this mesh. */
  e = 5;
  ierr = PetscOptionsGetInt(PETSC_NULL, "-e", &e, &flag); CHKERRQ(ierr);
  h = 1.0;
  ierr = PetscOptionsGetReal(PETSC_NULL, "-h", &h, &flag); CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "Using %d 1D elements per process with mesh spacing %g\n", e,h); CHKERRQ(ierr);

  /* Construct vectors with the global and locale elementwise layouts of degrees of freedom */
  /* Total number of degrees of freedom: size*e+1 -- the global number of elements + 1 */
  N = size*e+1;
  ierr = VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, N, &V); CHKERRQ(ierr);
  ierr = VecGetLocalSize(V, &n); CHKERRQ(ierr);

  /* Local number of elementwise degrees of freedom is twice the local number of elements e*/
  ierr = VecCreateMPI(PETSC_COMM_WORLD, e*2, PETSC_DECIDE, &v); CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(v, &llow, &lhigh); CHKERRQ(ierr);
  
  /* 
     Construct a map from local degrees of freedom (2 per element) to the global numbering of degrees of freedom 
  */
  /* Create the set of local dof indices: 
     the endpoints of the locally-held elements, each interior endpoint appearing twice with difference indices; 
     these duplicated points are still numbered globally */
  ierr = ISCreateStride(PETSC_COMM_WORLD, lhigh-llow, llow, 1, &isloc); CHKERRQ(ierr);
  /* Create the set of corresponding global dof indices -- unduplicated element endpoints numbered globally and consecutively;
     the global range of locally-held elements is from llow/2 to lhigh/2-1; the corresponding range of element endpoints is from llow/2 to lhigh/2;
     the number of locally-held elements is e = (lhigh-llow)/2;
  */
  ierr = PetscMalloc(sizeof(PetscInt)*2*e, &idx); CHKERRQ(ierr);  
  idx[0] = llow/2;
  for(i = 1,id=idx[0]+1; i < 2*e-1; i+=2,++id) {
    idx[i] = idx[i+1] = id;
  }
  idx[2*e-1]=lhigh/2;
  ierr = ISCreateGeneral(PETSC_COMM_WORLD, 2*e, idx, &isglob); CHKERRQ(ierr);

  /* Create a global to local scatter */
  ierr = VecScatterCreate(V, isglob, v, isloc, &scatter); CHKERRQ(ierr);
  ierr = MatCreateScatter(PETSC_COMM_WORLD, scatter, &scattermat); CHKERRQ(ierr);

  /* Create a local to global gather (transpose of the scatter, in this case) */
  ierr = VecScatterCreate(v,isloc, V, isglob, &gather); CHKERRQ(ierr);
  ierr = MatCreateScatter(PETSC_COMM_WORLD, gather, &gathermat); CHKERRQ(ierr);

  /* Now create the matrix framework */
  ierr = MatCreate(PETSC_COMM_WORLD, &M); CHKERRQ(ierr);
  ierr = MatSetSizes(M, n, n, N, N); CHKERRQ(ierr);
  ierr = MatSetType(M, MATFWK); CHKERRQ(ierr);

  /* Set scatter, gather and define the local block structure: e blocks with 2 vec elements per block */
  ierr = PetscMalloc(e*sizeof(PetscInt), &blocks); CHKERRQ(ierr);
  for(i = 0; i < e; ++i) {
    blocks[i] = 2;
  }
  ierr = MatFwkSetScatter(M, scattermat, e, blocks); CHKERRQ(ierr);
  ierr = MatFwkSetGather(M, gathermat, e, blocks); CHKERRQ(ierr);

  /* Now set up the blocks */
  for(i = 0; i < 4; ++i) {
    values[i] *= h;
  }
  for(i = 0; i < e; ++i) {
    ierr = MatFwkAddBlock(M, i, i, MATDENSE, &B); CHKERRQ(ierr); /* Acquire reference to B */
    ierr = MatSetValues(B, 2,ii, 2,jj, values, INSERT_VALUES); CHKERRQ(ierr);
    ierr = MatDestroy(B); CHKERRQ(ierr); /* Drop reference to B */
  }

  /* Assemble MatFwk */
  ierr = MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

  /* Now we apply the matrix */
  ierr = VecSet(V, 1.0);      CHKERRQ(ierr);
  ierr = VecDuplicate(V, &W); CHKERRQ(ierr);
  ierr = MatMult(M,V,W);      CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD, "Initial vector V:\n"); CHKERRQ(ierr);
  ierr = VecView(V, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD, "Final vector W = M*V:\n"); CHKERRQ(ierr);
  ierr = VecView(W, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);

  /* Now we apply the transpose matrix */
  ierr = VecSet(V, 1.0);      CHKERRQ(ierr);
  ierr = MatMultTranspose(M,V,W);      CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD, "Initial vector V:\n"); CHKERRQ(ierr);
  ierr = VecView(V, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD, "Final vector W = M'*V:\n"); CHKERRQ(ierr);
  ierr = VecView(W, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);

  /* Clean up */
  ierr = VecDestroy(v); CHKERRQ(ierr);
  ierr = VecDestroy(V); CHKERRQ(ierr);
  ierr = VecDestroy(W); CHKERRQ(ierr);
  
  ierr = VecScatterDestroy(scatter); CHKERRQ(ierr);
  ierr = ISDestroy(isloc);           CHKERRQ(ierr);
  ierr = ISDestroy(isglob);          CHKERRQ(ierr);
  ierr = PetscFree(idx);             CHKERRQ(ierr);

  ierr = PetscFree(blocks);          CHKERRQ(ierr);
  ierr = MatDestroy(scattermat);     CHKERRQ(ierr);
  ierr = MatDestroy(M);              CHKERRQ(ierr);

  ierr = PetscFinalize(); CHKERRQ(ierr);
  return 0;
}
