static const char help[] = "Solves a Q1-P0 Stokes problem from Underworld.\n\
\n\
You can obtain a sample matrix from http://ftp.mcs.anl.gov/pub/petsc/Datafiles/matrices/underworld32.gz\n\
and run with -f underworld32.gz\n\n";

#include <petscksp.h>
#include <petscdmda.h>

static PetscErrorCode replace_submats(Mat A, IS isu, IS isp)
{
  PetscErrorCode ierr;
  Mat            A11,A22,A12,A21;
  Mat            nA11,nA22,nA12,nA21;
  const char     *prefix;

  PetscFunctionBeginUser;
  ierr = MatCreateSubMatrix(A,isu,isu,MAT_INITIAL_MATRIX,&A11);CHKERRQ(ierr);
  ierr = MatCreateSubMatrix(A,isu,isp,MAT_INITIAL_MATRIX,&A12);CHKERRQ(ierr);
  ierr = MatCreateSubMatrix(A,isp,isu,MAT_INITIAL_MATRIX,&A21);CHKERRQ(ierr);
  ierr = MatCreateSubMatrix(A,isp,isp,MAT_INITIAL_MATRIX,&A22);CHKERRQ(ierr);
  ierr = MatDuplicate(A11,MAT_COPY_VALUES,&nA11);CHKERRQ(ierr);
  ierr = MatDuplicate(A12,MAT_COPY_VALUES,&nA12);CHKERRQ(ierr);
  ierr = MatDuplicate(A21,MAT_COPY_VALUES,&nA21);CHKERRQ(ierr);
  ierr = MatDuplicate(A22,MAT_COPY_VALUES,&nA22);CHKERRQ(ierr);
  ierr = MatGetOptionsPrefix(A11,&prefix);CHKERRQ(ierr);
  ierr = MatSetOptionsPrefix(nA11,prefix);CHKERRQ(ierr);
  ierr = MatGetOptionsPrefix(A22,&prefix);CHKERRQ(ierr);
  ierr = MatSetOptionsPrefix(nA22,prefix);CHKERRQ(ierr);
  ierr = MatNestSetSubMat(A,0,0,nA11);CHKERRQ(ierr);
  ierr = MatNestSetSubMat(A,0,1,nA12);CHKERRQ(ierr);
  ierr = MatNestSetSubMat(A,1,0,nA21);CHKERRQ(ierr);
  ierr = MatNestSetSubMat(A,1,1,nA22);CHKERRQ(ierr);
  ierr = MatDestroy(&A11);CHKERRQ(ierr);
  ierr = MatDestroy(&A12);CHKERRQ(ierr);
  ierr = MatDestroy(&A21);CHKERRQ(ierr);
  ierr = MatDestroy(&A22);CHKERRQ(ierr);
  ierr = MatDestroy(&nA11);CHKERRQ(ierr);
  ierr = MatDestroy(&nA12);CHKERRQ(ierr);
  ierr = MatDestroy(&nA21);CHKERRQ(ierr);
  ierr = MatDestroy(&nA22);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode LSCLoadTestOperators(Mat *A11,Mat *A12,Mat *A21,Mat *A22,Vec *b1,Vec *b2)
{
  PetscViewer    viewer;
  PetscErrorCode ierr;
  char           filename[PETSC_MAX_PATH_LEN];
  PetscBool      flg;

  PetscFunctionBeginUser;
  ierr = MatCreate(PETSC_COMM_WORLD,A11);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,A12);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,A21);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,A22);CHKERRQ(ierr);
  ierr = MatSetOptionsPrefix(*A11,"a11_");CHKERRQ(ierr);
  ierr = MatSetOptionsPrefix(*A22,"a22_");CHKERRQ(ierr);
  ierr = MatSetFromOptions(*A11);CHKERRQ(ierr);
  ierr = MatSetFromOptions(*A22);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_WORLD,b1);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_WORLD,b2);CHKERRQ(ierr);
  /* Load matrices from a Q1-P0 discretisation of variable viscosity Stokes. The matrix blocks are packed into one file. */
  ierr = PetscOptionsGetString(NULL,NULL,"-f",filename,sizeof(filename),&flg);CHKERRQ(ierr);
  PetscCheckFalse(!flg,PETSC_COMM_WORLD,PETSC_ERR_USER,"Must provide a matrix file with -f");
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_READ,&viewer);CHKERRQ(ierr);
  ierr = MatLoad(*A11,viewer);CHKERRQ(ierr);
  ierr = MatLoad(*A12,viewer);CHKERRQ(ierr);
  ierr = MatLoad(*A21,viewer);CHKERRQ(ierr);
  ierr = MatLoad(*A22,viewer);CHKERRQ(ierr);
  ierr = VecLoad(*b1,viewer);CHKERRQ(ierr);
  ierr = VecLoad(*b2,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode LoadTestMatrices(Mat *_A,Vec *_x,Vec *_b,IS *_isu,IS *_isp)
{
  Vec            f,h,x,b,bX[2];
  Mat            A,Auu,Aup,Apu,App,bA[2][2];
  IS             is_u,is_p,bis[2];
  PetscInt       lnu,lnp,nu,np,i,start_u,end_u,start_p,end_p;
  VecScatter     *vscat;
  PetscMPIInt    rank;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  /* fetch test matrices and vectors */
  ierr = LSCLoadTestOperators(&Auu,&Aup,&Apu,&App,&f,&h);CHKERRQ(ierr);

  /* build the mat-nest */
  ierr = VecGetSize(f,&nu);CHKERRQ(ierr);
  ierr = VecGetSize(h,&np);CHKERRQ(ierr);

  ierr = VecGetLocalSize(f,&lnu);CHKERRQ(ierr);
  ierr = VecGetLocalSize(h,&lnp);CHKERRQ(ierr);

  ierr = VecGetOwnershipRange(f,&start_u,&end_u);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(h,&start_p,&end_p);CHKERRQ(ierr);

  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRMPI(ierr);
  ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d] lnu = %D | lnp = %D \n", rank, lnu, lnp);CHKERRQ(ierr);
  ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d] s_u = %D | e_u = %D \n", rank, start_u, end_u);CHKERRQ(ierr);
  ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d] s_p = %D | e_p = %D \n", rank, start_p, end_p);CHKERRQ(ierr);
  ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d] is_u (offset) = %D \n", rank, start_u+start_p);CHKERRQ(ierr);
  ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d] is_p (offset) = %D \n", rank, start_u+start_p+lnu);CHKERRQ(ierr);
  ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT);CHKERRQ(ierr);

  ierr = ISCreateStride(PETSC_COMM_WORLD,lnu,start_u+start_p,1,&is_u);CHKERRQ(ierr);
  ierr = ISCreateStride(PETSC_COMM_WORLD,lnp,start_u+start_p+lnu,1,&is_p);CHKERRQ(ierr);

  bis[0]   = is_u; bis[1]   = is_p;
  bA[0][0] = Auu;  bA[0][1] = Aup;
  bA[1][0] = Apu;  bA[1][1] = App;
  ierr     = MatCreateNest(PETSC_COMM_WORLD,2,bis,2,bis,&bA[0][0],&A);CHKERRQ(ierr);
  ierr     = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr     = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* Pull f,h into b */
  ierr  = MatCreateVecs(A,&b,&x);CHKERRQ(ierr);
  bX[0] = f;  bX[1] = h;
  ierr  = PetscMalloc1(2,&vscat);CHKERRQ(ierr);
  for (i=0; i<2; i++) {
    ierr = VecScatterCreate(b,bis[i],bX[i],NULL,&vscat[i]);CHKERRQ(ierr);
    ierr = VecScatterBegin(vscat[i],bX[i],b,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterEnd(vscat[i],bX[i],b,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterDestroy(&vscat[i]);CHKERRQ(ierr);
  }

  ierr = PetscFree(vscat);CHKERRQ(ierr);
  ierr = MatDestroy(&Auu);CHKERRQ(ierr);
  ierr = MatDestroy(&Aup);CHKERRQ(ierr);
  ierr = MatDestroy(&Apu);CHKERRQ(ierr);
  ierr = MatDestroy(&App);CHKERRQ(ierr);
  ierr = VecDestroy(&f);CHKERRQ(ierr);
  ierr = VecDestroy(&h);CHKERRQ(ierr);

  *_isu = is_u;
  *_isp = is_p;
  *_A   = A;
  *_x   = x;
  *_b   = b;
  PetscFunctionReturn(0);
}

PetscErrorCode port_lsd_bfbt(void)
{
  Mat            A,P;
  Vec            x,b;
  KSP            ksp_A;
  PC             pc_A;
  IS             isu,isp;
  PetscBool      test_fs = PETSC_FALSE;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = LoadTestMatrices(&A,&x,&b,&isu,&isp);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-test_fs",&test_fs,NULL);CHKERRQ(ierr);
  if (!test_fs) {
    ierr = PetscObjectReference((PetscObject)A);CHKERRQ(ierr);
    P = A;
  } else {
    ierr = MatDuplicate(A,MAT_COPY_VALUES,&P);CHKERRQ(ierr);
  }
  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp_A);CHKERRQ(ierr);
  ierr = KSPSetOptionsPrefix(ksp_A,"fc_");CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp_A,A,P);CHKERRQ(ierr);

  ierr = KSPSetFromOptions(ksp_A);CHKERRQ(ierr);
  ierr = KSPGetPC(ksp_A,&pc_A);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)pc_A,PCLU,&test_fs);CHKERRQ(ierr);
  if (test_fs) {
    ierr = MatDestroy(&P);CHKERRQ(ierr);
    ierr = MatConvert(A,MATAIJ,MAT_INITIAL_MATRIX,&P);CHKERRQ(ierr);
    ierr = KSPSetOperators(ksp_A,A,P);CHKERRQ(ierr);
    ierr = KSPSetFromOptions(ksp_A);CHKERRQ(ierr);
    ierr = KSPSolve(ksp_A,b,x);CHKERRQ(ierr);
  } else {
    ierr = PCFieldSplitSetBlockSize(pc_A,2);CHKERRQ(ierr);
    ierr = PCFieldSplitSetIS(pc_A,"velocity",isu);CHKERRQ(ierr);
    ierr = PCFieldSplitSetIS(pc_A,"pressure",isp);CHKERRQ(ierr);
    ierr = KSPSolve(ksp_A,b,x);CHKERRQ(ierr);

    /* Pull u,p out of x */
    {
      PetscInt    loc;
      PetscReal   max,norm;
      PetscScalar sum;
      Vec         uvec,pvec;
      VecScatter  uscat,pscat;
      Mat         A11,A22;

      /* grab matrices and create the compatable u,p vectors */
      ierr = MatCreateSubMatrix(A,isu,isu,MAT_INITIAL_MATRIX,&A11);CHKERRQ(ierr);
      ierr = MatCreateSubMatrix(A,isp,isp,MAT_INITIAL_MATRIX,&A22);CHKERRQ(ierr);

      ierr = MatCreateVecs(A11,&uvec,NULL);CHKERRQ(ierr);
      ierr = MatCreateVecs(A22,&pvec,NULL);CHKERRQ(ierr);

      /* perform the scatter from x -> (u,p) */
      ierr = VecScatterCreate(x,isu,uvec,NULL,&uscat);CHKERRQ(ierr);
      ierr = VecScatterBegin(uscat,x,uvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecScatterEnd(uscat,x,uvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);

      ierr = VecScatterCreate(x,isp,pvec,NULL,&pscat);CHKERRQ(ierr);
      ierr = VecScatterBegin(pscat,x,pvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecScatterEnd(pscat,x,pvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);

      ierr = PetscPrintf(PETSC_COMM_WORLD,"-- vector vector values --\n");CHKERRQ(ierr);
      ierr = VecMin(uvec,&loc,&max);CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD,"  Min(u)  = %1.6f [loc=%D]\n",(double)max,loc);CHKERRQ(ierr);
      ierr = VecMax(uvec,&loc,&max);CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD,"  Max(u)  = %1.6f [loc=%D]\n",(double)max,loc);CHKERRQ(ierr);
      ierr = VecNorm(uvec,NORM_2,&norm);CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD,"  Norm(u) = %1.6f \n",(double)norm);CHKERRQ(ierr);
      ierr = VecSum(uvec,&sum);CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD,"  Sum(u)  = %1.6f \n",(double)PetscRealPart(sum));CHKERRQ(ierr);

      ierr = PetscPrintf(PETSC_COMM_WORLD,"-- pressure vector values --\n");CHKERRQ(ierr);
      ierr = VecMin(pvec,&loc,&max);CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD,"  Min(p)  = %1.6f [loc=%D]\n",(double)max,loc);CHKERRQ(ierr);
      ierr = VecMax(pvec,&loc,&max);CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD,"  Max(p)  = %1.6f [loc=%D]\n",(double)max,loc);CHKERRQ(ierr);
      ierr = VecNorm(pvec,NORM_2,&norm);CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD,"  Norm(p) = %1.6f \n",(double)norm);CHKERRQ(ierr);
      ierr = VecSum(pvec,&sum);CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD,"  Sum(p)  = %1.6f \n",(double)PetscRealPart(sum));CHKERRQ(ierr);

      ierr = PetscPrintf(PETSC_COMM_WORLD,"-- Full vector values --\n");CHKERRQ(ierr);
      ierr = VecMin(x,&loc,&max);CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD,"  Min(u,p)  = %1.6f [loc=%D]\n",(double)max,loc);CHKERRQ(ierr);
      ierr = VecMax(x,&loc,&max);CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD,"  Max(u,p)  = %1.6f [loc=%D]\n",(double)max,loc);CHKERRQ(ierr);
      ierr = VecNorm(x,NORM_2,&norm);CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD,"  Norm(u,p) = %1.6f \n",(double)norm);CHKERRQ(ierr);
      ierr = VecSum(x,&sum);CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD,"  Sum(u,p)  = %1.6f \n",(double)PetscRealPart(sum));CHKERRQ(ierr);

      ierr = VecScatterDestroy(&uscat);CHKERRQ(ierr);
      ierr = VecScatterDestroy(&pscat);CHKERRQ(ierr);
      ierr = VecDestroy(&uvec);CHKERRQ(ierr);
      ierr = VecDestroy(&pvec);CHKERRQ(ierr);
      ierr = MatDestroy(&A11);CHKERRQ(ierr);
      ierr = MatDestroy(&A22);CHKERRQ(ierr);
    }

    /* test second solve by changing the mat associated to the MATNEST blocks */
    {
      ierr = replace_submats(A,isu,isp);CHKERRQ(ierr);
      ierr = replace_submats(P,isu,isp);CHKERRQ(ierr);
      ierr = KSPSolve(ksp_A,b,x);CHKERRQ(ierr);
    }
  }

  ierr = KSPDestroy(&ksp_A);CHKERRQ(ierr);
  ierr = MatDestroy(&P);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = ISDestroy(&isu);CHKERRQ(ierr);
  ierr = ISDestroy(&isp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&argv,0,help);if (ierr) return ierr;
  ierr = port_lsd_bfbt();CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

    test:
      args: -f ${DATAFILESPATH}/matrices/underworld32.gz -fc_ksp_view -fc_ksp_monitor_short -fc_ksp_type fgmres -fc_ksp_max_it 4000 -fc_ksp_diagonal_scale -fc_pc_type fieldsplit -fc_pc_fieldsplit_type SCHUR -fc_pc_fieldsplit_schur_fact_type UPPER -fc_fieldsplit_velocity_ksp_type cg -fc_fieldsplit_velocity_pc_type cholesky -fc_fieldsplit_velocity_pc_factor_mat_ordering_type nd -fc_fieldsplit_pressure_ksp_max_it 100 -fc_fieldsplit_pressure_ksp_constant_null_space -fc_fieldsplit_pressure_ksp_monitor_short -fc_fieldsplit_pressure_pc_type lsc -fc_fieldsplit_pressure_lsc_ksp_type cg -fc_fieldsplit_pressure_lsc_ksp_max_it 100 -fc_fieldsplit_pressure_lsc_ksp_constant_null_space -fc_fieldsplit_pressure_lsc_ksp_converged_reason -fc_fieldsplit_pressure_lsc_pc_type icc -test_fs {{0 1}separate output} -fc_pc_fieldsplit_off_diag_use_amat {{0 1}separate output} -fc_pc_fieldsplit_diag_use_amat {{0 1}separate output}
      requires: datafilespath double !complex !defined(PETSC_USE_64BIT_INDICES)

    test:
      suffix: 2
      nsize: 4
      args: -f ${DATAFILESPATH}/matrices/underworld32.gz -fc_ksp_view -fc_ksp_monitor_short -fc_ksp_type fgmres -fc_ksp_max_it 4000 -fc_ksp_diagonal_scale -fc_pc_type fieldsplit -fc_pc_fieldsplit_type SCHUR -fc_pc_fieldsplit_schur_fact_type UPPER -fc_fieldsplit_velocity_ksp_type cg -fc_fieldsplit_velocity_ksp_rtol 1.0e-6 -fc_fieldsplit_velocity_pc_type bjacobi -fc_fieldsplit_velocity_sub_pc_type cholesky -fc_fieldsplit_velocity_sub_pc_factor_mat_ordering_type nd -fc_fieldsplit_pressure_ksp_type fgmres -fc_fieldsplit_pressure_ksp_constant_null_space -fc_fieldsplit_pressure_ksp_monitor_short -fc_fieldsplit_pressure_pc_type lsc -fc_fieldsplit_pressure_lsc_ksp_type cg -fc_fieldsplit_pressure_lsc_ksp_rtol 1.0e-2 -fc_fieldsplit_pressure_lsc_ksp_constant_null_space -fc_fieldsplit_pressure_lsc_ksp_converged_reason -fc_fieldsplit_pressure_lsc_pc_type bjacobi -fc_fieldsplit_pressure_lsc_sub_pc_type icc -test_fs {{0 1}separate output} -fc_pc_fieldsplit_off_diag_use_amat {{0 1}separate output} -fc_pc_fieldsplit_diag_use_amat {{0 1}separate output}
      requires: datafilespath double !complex !defined(PETSC_USE_64BIT_INDICES)

    test:
      suffix: 3
      nsize: 2
      args: -f ${DATAFILESPATH}/matrices/underworld32.gz -fc_ksp_view_pre -fc_pc_type lu
      requires: datafilespath mumps double !complex !defined(PETSC_USE_64BIT_INDICES)

TEST*/
