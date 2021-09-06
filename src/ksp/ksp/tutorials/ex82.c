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
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&argv,(char*)NULL,help);if (ierr) return ierr;
  ierr = PetscOptionsGetInt(NULL,NULL,"-m_local",&m,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-symmetric",&sym,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-dim",&dim,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-overlap",&overlap,NULL);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);
  M = size*m;
  ierr = PetscOptionsGetInt(NULL,NULL,"-M",&M,NULL);CHKERRQ(ierr);
  ierr = PetscMalloc1(m*dim,&coords);CHKERRQ(ierr);
  ierr = PetscRandomCreate(PETSC_COMM_WORLD,&rdm);CHKERRQ(ierr);
  ierr = PetscRandomGetValuesReal(rdm,m*dim,coords);CHKERRQ(ierr);
  ierr = PetscCalloc1(M*dim,&gcoords);CHKERRQ(ierr);
  ierr = MPI_Exscan(&m,&begin,1,MPIU_INT,MPI_SUM,PETSC_COMM_WORLD);CHKERRMPI(ierr);
  ierr = PetscArraycpy(gcoords+begin*dim,coords,m*dim);CHKERRQ(ierr);
  ierr = MPIU_Allreduce(MPI_IN_PLACE,gcoords,M*dim,MPIU_REAL,MPI_SUM,PETSC_COMM_WORLD);CHKERRMPI(ierr);
  ierr = MatCreateHtoolFromKernel(PETSC_COMM_WORLD,m,m,M,M,dim,coords,coords,kernel,gcoords,&A);CHKERRQ(ierr);
  ierr = MatSetOption(A,MAT_SYMMETRIC,sym);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatCreateVecs(A,&b,&x);CHKERRQ(ierr);
  ierr = VecSetRandom(b,rdm);CHKERRQ(ierr);
  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)pc,PCHPDDM,&flg);CHKERRQ(ierr);
  if (flg) {
#if defined(PETSC_HAVE_HPDDM) && defined(PETSC_HAVE_DYNAMIC_LIBRARIES) && defined(PETSC_USE_SHARED_LIBRARIES)
    Mat aux;
    IS  is;
    ierr = MatGetOwnershipRange(A,&begin,&n);CHKERRQ(ierr);
    n -= begin;
    ierr = ISCreateStride(PETSC_COMM_SELF,n,begin,1,&is);CHKERRQ(ierr);
    ierr = MatIncreaseOverlap(A,1,&is,overlap);CHKERRQ(ierr);
    ierr = ISGetLocalSize(is,&n);CHKERRQ(ierr);
    ierr = MatCreateDense(PETSC_COMM_SELF,n,n,n,n,NULL,&aux);CHKERRQ(ierr);
    ierr = MatSetOption(aux,MAT_SYMMETRIC,sym);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(aux,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(aux,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatShift(aux,1.0);CHKERRQ(ierr); /* just the local identity matrix, not very meaningful numerically, but just testing that the necessary plumbing is there */
    ierr = PCHPDDMSetAuxiliaryMat(pc,is,aux,NULL,NULL);CHKERRQ(ierr);
    ierr = ISDestroy(&is);CHKERRQ(ierr);
    ierr = MatDestroy(&aux);CHKERRQ(ierr);
#endif
  }
  ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&rdm);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = PetscFree(gcoords);CHKERRQ(ierr);
  ierr = PetscFree(coords);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
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
