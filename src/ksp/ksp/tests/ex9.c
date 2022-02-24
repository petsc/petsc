
static char help[] = "Tests repeated setups and solves of PCFIELDSPLIT.\n\n";
#include <petscksp.h>

static PetscErrorCode replace_submats(Mat A)
{
  IS             *r,*c;
  PetscInt       i,j,nr,nc;

  PetscFunctionBeginUser;
  CHKERRQ(MatNestGetSubMats(A,&nr,&nc,NULL));
  CHKERRQ(PetscMalloc1(nr,&r));
  CHKERRQ(PetscMalloc1(nc,&c));
  CHKERRQ(MatNestGetISs(A,r,c));
  for (i=0;i<nr;i++) {
    for (j=0;j<nc;j++) {
      Mat        sA,nA;
      const char *prefix;

      CHKERRQ(MatCreateSubMatrix(A,r[i],c[j],MAT_INITIAL_MATRIX,&sA));
      CHKERRQ(MatDuplicate(sA,MAT_COPY_VALUES,&nA));
      CHKERRQ(MatGetOptionsPrefix(sA,&prefix));
      CHKERRQ(MatSetOptionsPrefix(nA,prefix));
      CHKERRQ(MatNestSetSubMat(A,i,j,nA));
      CHKERRQ(MatDestroy(&nA));
      CHKERRQ(MatDestroy(&sA));
    }
  }
  CHKERRQ(PetscFree(r));
  CHKERRQ(PetscFree(c));
  PetscFunctionReturn(0);
}

int main(int argc, char *argv[])
{
   KSP            ksp;
   PC             pc;
   Mat            M,A,P,sA[2][2],sP[2][2];
   Vec            x,b;
   IS             f[2];
   PetscInt       i,j,rstart,rend;
   PetscBool      missA,missM;
   PetscErrorCode ierr;

   ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
   CHKERRQ(MatCreateAIJ(PETSC_COMM_WORLD,10,10,PETSC_DECIDE,PETSC_DECIDE,1,NULL,0,NULL,&M));
   CHKERRQ(MatAssemblyBegin(M,MAT_FINAL_ASSEMBLY));
   CHKERRQ(MatAssemblyEnd(M,MAT_FINAL_ASSEMBLY));
   CHKERRQ(MatShift(M,1.));
   CHKERRQ(MatGetOwnershipRange(M,&rstart,&rend));
   CHKERRQ(ISCreateStride(PetscObjectComm((PetscObject)M),7,rstart,1,&f[0]));
   CHKERRQ(ISComplement(f[0],rstart,rend,&f[1]));
   for (i=0;i<2;i++) {
     for (j=0;j<2;j++) {
       CHKERRQ(MatCreateSubMatrix(M,f[i],f[j],MAT_INITIAL_MATRIX,&sA[i][j]));
       CHKERRQ(MatCreateSubMatrix(M,f[i],f[j],MAT_INITIAL_MATRIX,&sP[i][j]));
     }
   }
   CHKERRQ(MatCreateNest(PetscObjectComm((PetscObject)M),2,f,2,f,&sA[0][0],&A));
   CHKERRQ(MatCreateNest(PetscObjectComm((PetscObject)M),2,f,2,f,&sP[0][0],&P));

   /* Tests MatMissingDiagonal_Nest */
   CHKERRQ(MatMissingDiagonal(M,&missM,NULL));
   CHKERRQ(MatMissingDiagonal(A,&missA,NULL));
   if (missM != missA) {
     CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Unexpected %s != %s\n",missM ? "true": "false",missA ? "true" : "false"));
   }

   CHKERRQ(MatDestroy(&M));

   CHKERRQ(KSPCreate(PetscObjectComm((PetscObject)A),&ksp));
   CHKERRQ(KSPSetOperators(ksp,A,P));
   CHKERRQ(KSPGetPC(ksp,&pc));
   CHKERRQ(PCSetType(pc,PCFIELDSPLIT));
   CHKERRQ(KSPSetFromOptions(ksp));
   CHKERRQ(MatCreateVecs(A,&x,&b));
   CHKERRQ(VecSetRandom(b,NULL));
   CHKERRQ(KSPSolve(ksp,b,x));
   CHKERRQ(replace_submats(A));
   CHKERRQ(replace_submats(P));
   CHKERRQ(KSPSolve(ksp,b,x));

   CHKERRQ(KSPDestroy(&ksp));
   CHKERRQ(VecDestroy(&x));
   CHKERRQ(VecDestroy(&b));
   CHKERRQ(MatDestroy(&A));
   CHKERRQ(MatDestroy(&P));
   for (i=0;i<2;i++) {
     CHKERRQ(ISDestroy(&f[i]));
     for (j=0;j<2;j++) {
       CHKERRQ(MatDestroy(&sA[i][j]));
       CHKERRQ(MatDestroy(&sP[i][j]));
     }
   }
   ierr = PetscFinalize();
   return ierr;
}

/*TEST

   test:
     nsize: 1
     filter: sed -e "s/CONVERGED_RTOL/CONVERGED_ATOL/g"
     args: -pc_fieldsplit_diag_use_amat {{0 1}} -pc_fieldsplit_diag_use_amat {{0 1}} -pc_fieldsplit_type {{additive multiplicative}} -ksp_converged_reason -ksp_error_if_not_converged

   test:
     suffix: schur
     nsize: 1
     filter: sed -e "s/CONVERGED_RTOL/CONVERGED_ATOL/g"
     args: -pc_fieldsplit_diag_use_amat {{0 1}} -pc_fieldsplit_diag_use_amat {{0 1}} -pc_fieldsplit_type schur -pc_fieldsplit_schur_scale 1.0 -pc_fieldsplit_schur_fact_type {{diag lower upper full}} -ksp_converged_reason -ksp_error_if_not_converged

TEST*/
