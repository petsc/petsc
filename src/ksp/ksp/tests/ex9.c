
static char help[] = "Tests repeated setups and solves of PCFIELDSPLIT.\n\n";
#include <petscksp.h>

static PetscErrorCode replace_submats(Mat A)
{
  IS             *r,*c;
  PetscInt       i,j,nr,nc;

  PetscFunctionBeginUser;
  PetscCall(MatNestGetSubMats(A,&nr,&nc,NULL));
  PetscCall(PetscMalloc1(nr,&r));
  PetscCall(PetscMalloc1(nc,&c));
  PetscCall(MatNestGetISs(A,r,c));
  for (i=0;i<nr;i++) {
    for (j=0;j<nc;j++) {
      Mat        sA,nA;
      const char *prefix;

      PetscCall(MatCreateSubMatrix(A,r[i],c[j],MAT_INITIAL_MATRIX,&sA));
      PetscCall(MatDuplicate(sA,MAT_COPY_VALUES,&nA));
      PetscCall(MatGetOptionsPrefix(sA,&prefix));
      PetscCall(MatSetOptionsPrefix(nA,prefix));
      PetscCall(MatNestSetSubMat(A,i,j,nA));
      PetscCall(MatDestroy(&nA));
      PetscCall(MatDestroy(&sA));
    }
  }
  PetscCall(PetscFree(r));
  PetscCall(PetscFree(c));
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

   PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
   PetscCall(MatCreateAIJ(PETSC_COMM_WORLD,10,10,PETSC_DECIDE,PETSC_DECIDE,1,NULL,0,NULL,&M));
   PetscCall(MatAssemblyBegin(M,MAT_FINAL_ASSEMBLY));
   PetscCall(MatAssemblyEnd(M,MAT_FINAL_ASSEMBLY));
   PetscCall(MatShift(M,1.));
   PetscCall(MatGetOwnershipRange(M,&rstart,&rend));
   PetscCall(ISCreateStride(PetscObjectComm((PetscObject)M),7,rstart,1,&f[0]));
   PetscCall(ISComplement(f[0],rstart,rend,&f[1]));
   for (i=0;i<2;i++) {
     for (j=0;j<2;j++) {
       PetscCall(MatCreateSubMatrix(M,f[i],f[j],MAT_INITIAL_MATRIX,&sA[i][j]));
       PetscCall(MatCreateSubMatrix(M,f[i],f[j],MAT_INITIAL_MATRIX,&sP[i][j]));
     }
   }
   PetscCall(MatCreateNest(PetscObjectComm((PetscObject)M),2,f,2,f,&sA[0][0],&A));
   PetscCall(MatCreateNest(PetscObjectComm((PetscObject)M),2,f,2,f,&sP[0][0],&P));

   /* Tests MatMissingDiagonal_Nest */
   PetscCall(MatMissingDiagonal(M,&missM,NULL));
   PetscCall(MatMissingDiagonal(A,&missA,NULL));
   if (missM != missA) {
     PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Unexpected %s != %s\n",missM ? "true": "false",missA ? "true" : "false"));
   }

   PetscCall(MatDestroy(&M));

   PetscCall(KSPCreate(PetscObjectComm((PetscObject)A),&ksp));
   PetscCall(KSPSetOperators(ksp,A,P));
   PetscCall(KSPGetPC(ksp,&pc));
   PetscCall(PCSetType(pc,PCFIELDSPLIT));
   PetscCall(KSPSetFromOptions(ksp));
   PetscCall(MatCreateVecs(A,&x,&b));
   PetscCall(VecSetRandom(b,NULL));
   PetscCall(KSPSolve(ksp,b,x));
   PetscCall(replace_submats(A));
   PetscCall(replace_submats(P));
   PetscCall(KSPSolve(ksp,b,x));

   PetscCall(KSPDestroy(&ksp));
   PetscCall(VecDestroy(&x));
   PetscCall(VecDestroy(&b));
   PetscCall(MatDestroy(&A));
   PetscCall(MatDestroy(&P));
   for (i=0;i<2;i++) {
     PetscCall(ISDestroy(&f[i]));
     for (j=0;j<2;j++) {
       PetscCall(MatDestroy(&sA[i][j]));
       PetscCall(MatDestroy(&sP[i][j]));
     }
   }
   PetscCall(PetscFinalize());
   return 0;
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
