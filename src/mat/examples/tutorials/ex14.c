
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
  PetscInt llow, lhigh, glow, ghigh;
  PetscTruth flag;
  Vec v,V, W;
  IS isloc, isglob;
  PetscInt size;
  VecScatter scatter;
  Mat scattermat;
  PetscInt i, *blocks;
  PetscScalar values[4];
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

  /* Construct vectors with the global and localized layouts of degrees of freedom */
  /* Total number of degrees of freedom: size*e+1 -- the global number of elements + 1 */
  N = size*2*e+1;
  
  ierr = VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, N, &V); CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(V, &glow, &ghigh); CHKERRQ(ierr);
  n = glow-ghigh+1;

  ierr = VecCreateMPI(PETSC_COMM_WORLD, e*2, PETSC_DECIDE, &v); CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(v, &llow, &lhigh); CHKERRQ(ierr);
  
  /* Construct a map from local degrees of freedom (2 per element) to the global numbering of degrees of freedom */
  /* Create the set of local dof indices */
  ierr = ISCreateStride(PETSC_COMM_SELF, lhigh-llow+1, llow, 1, &isloc); CHKERRQ(ierr);
  /* Create the set of corresponding global dof indices */
  ierr = ISCreateStride(PETSC_COMM_SELF, ghigh-glow+1, glow, 1, &isglob); CHKERRQ(ierr);
  /* Create a global to local scatter */
  ierr = VecScatterCreate(V, isglob, v, isloc, &scatter); CHKERRQ(ierr);
  ierr = MatCreateScatter(PETSC_COMM_WORLD, scatter, &scattermat); CHKERRQ(ierr);

  /* Now create the matrix framework */
  ierr = MatCreate(PETSC_COMM_WORLD, &M); CHKERRQ(ierr);
  ierr = MatSetSizes(M, n, N, n, N); CHKERRQ(ierr);
  ierr = MatSetType(M, MATFWK); CHKERRQ(ierr);

  /* Set scatter, gather and define the local block structure: e blocks with 2 vec elements per block */
  ierr = PetscMalloc(e*sizeof(PetscInt), &blocks); CHKERRQ(ierr);
  blocks[0] = 0;
  for(i = 1; i < e; ++i) {
    blocks[i] = blocks[i-1]+2;
  }
  ierr = MatFwkSetScatter(M, scattermat, e, blocks); CHKERRQ(ierr);
  ierr = MatFwkSetGather(M, scattermat, e, blocks); CHKERRQ(ierr);

  /* Now set up the blocks */
  for(i = 0; i < e; ++i) {
    ierr = MatFwkAddBlock(M, i, i, MATDENSE, &B); CHKERRQ(ierr);
    ierr = MatSetValues(B, 2,ii, 2,jj, values, INSERT_VALUES); CHKERRQ(ierr);
    ierr = MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY);   CHKERRQ(ierr);
  }

  /* Now we can apply the matrix */
  ierr = VecSet(V, 1.0);      CHKERRQ(ierr);
  ierr = VecDuplicate(V, &W); CHKERRQ(ierr);
  ierr = MatMult(M,V,W);      CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD, "Initial vector V:\n"); CHKERRQ(ierr);
  ierr = VecView(V, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD, "Final vector W = M*V:\n"); CHKERRQ(ierr);
  ierr = VecView(W, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);

  ierr = PetscFinalize(); CHKERRQ(ierr);
  return 0;
}
