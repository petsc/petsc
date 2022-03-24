#include <petscksp.h>

static char help[] = "Solves a linear system using PCHPDDM and MATHTOOL.\n\n";

static PetscErrorCode GenEntries(PetscInt sdim,PetscInt M,PetscInt N,const PetscInt *J,const PetscInt *K,PetscScalar *ptr,void *ctx)
{
  PetscInt  d,j,k;
  PetscReal diff = 0.0,*coords = (PetscReal*)(ctx);

  PetscFunctionBeginUser;
  for (j = 0; j < M; j++) {
    for (k = 0; k < N; k++) {
      diff = 0.0;
      for (d = 0; d < sdim; d++) diff += (coords[J[j]*sdim+d] - coords[K[k]*sdim+d]) * (coords[J[j]*sdim+d] - coords[K[k]*sdim+d]);
      ptr[j+M*k] = 1.0/(1.0e-2 + PetscSqrtReal(diff));
    }
  }
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  KSP            ksp;
  PC             pc;
  Vec            b,x;
  Mat            A;
  PetscInt       m = 100,dim = 3,M,begin = 0,n = 0,overlap = 1;
  PetscMPIInt    size;
  PetscReal      *coords,*gcoords;
  MatHtoolKernel kernel = GenEntries;
  PetscBool      flg,sym = PETSC_FALSE;
  PetscRandom    rdm;

  CHKERRQ(PetscInitialize(&argc,&argv,(char*)NULL,help));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-m_local",&m,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-symmetric",&sym,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-dim",&dim,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-overlap",&overlap,NULL));
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  M = size*m;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-M",&M,NULL));
  CHKERRQ(PetscMalloc1(m*dim,&coords));
  CHKERRQ(PetscRandomCreate(PETSC_COMM_WORLD,&rdm));
  CHKERRQ(PetscRandomGetValuesReal(rdm,m*dim,coords));
  CHKERRQ(PetscCalloc1(M*dim,&gcoords));
  CHKERRMPI(MPI_Exscan(&m,&begin,1,MPIU_INT,MPI_SUM,PETSC_COMM_WORLD));
  CHKERRQ(PetscArraycpy(gcoords+begin*dim,coords,m*dim));
  CHKERRMPI(MPIU_Allreduce(MPI_IN_PLACE,gcoords,M*dim,MPIU_REAL,MPI_SUM,PETSC_COMM_WORLD));
  CHKERRQ(MatCreateHtoolFromKernel(PETSC_COMM_WORLD,m,m,M,M,dim,coords,coords,kernel,gcoords,&A));
  CHKERRQ(MatSetOption(A,MAT_SYMMETRIC,sym));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatCreateVecs(A,&b,&x));
  CHKERRQ(VecSetRandom(b,rdm));
  CHKERRQ(KSPCreate(PETSC_COMM_WORLD,&ksp));
  CHKERRQ(KSPSetOperators(ksp,A,A));
  CHKERRQ(KSPSetFromOptions(ksp));
  CHKERRQ(KSPGetPC(ksp,&pc));
  CHKERRQ(PetscObjectTypeCompare((PetscObject)pc,PCHPDDM,&flg));
  if (flg) {
#if defined(PETSC_HAVE_HPDDM) && defined(PETSC_HAVE_DYNAMIC_LIBRARIES) && defined(PETSC_USE_SHARED_LIBRARIES)
    Mat aux;
    IS  is;
    CHKERRQ(MatGetOwnershipRange(A,&begin,&n));
    n -= begin;
    CHKERRQ(ISCreateStride(PETSC_COMM_SELF,n,begin,1,&is));
    CHKERRQ(MatIncreaseOverlap(A,1,&is,overlap));
    CHKERRQ(ISGetLocalSize(is,&n));
    CHKERRQ(MatCreateDense(PETSC_COMM_SELF,n,n,n,n,NULL,&aux));
    CHKERRQ(MatSetOption(aux,MAT_SYMMETRIC,sym));
    CHKERRQ(MatAssemblyBegin(aux,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(aux,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatShift(aux,1.0)); /* just the local identity matrix, not very meaningful numerically, but just testing that the necessary plumbing is there */
    CHKERRQ(PCHPDDMSetAuxiliaryMat(pc,is,aux,NULL,NULL));
    CHKERRQ(ISDestroy(&is));
    CHKERRQ(MatDestroy(&aux));
#endif
  }
  CHKERRQ(KSPSolve(ksp,b,x));
  CHKERRQ(KSPDestroy(&ksp));
  CHKERRQ(PetscRandomDestroy(&rdm));
  CHKERRQ(VecDestroy(&b));
  CHKERRQ(VecDestroy(&x));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(PetscFree(gcoords));
  CHKERRQ(PetscFree(coords));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

   build:
      requires: htool hpddm

   test:
      requires: htool hpddm slepc defined(PETSC_HAVE_DYNAMIC_LIBRARIES) defined(PETSC_USE_SHARED_LIBRARIES)
      nsize: 4
      # different numbers of iterations depending on PetscScalar type
      filter: sed -e "s/symmetry: S/symmetry: N/g" -e "/number of dense/d" -e "s/Linear solve converged due to CONVERGED_RTOL iterations 13/Linear solve converged due to CONVERGED_RTOL iterations 18/g"
      args: -ksp_view -ksp_converged_reason -mat_htool_epsilon 1e-2 -m_local 200 -pc_type hpddm -pc_hpddm_define_subdomains -pc_hpddm_levels_1_sub_pc_type lu -pc_hpddm_levels_1_eps_nev 1 -pc_hpddm_coarse_pc_type lu -pc_hpddm_levels_1_eps_gen_non_hermitian -symmetric {{false true}shared output} -overlap 2
      output_file: output/ex82_1.out

TEST*/
