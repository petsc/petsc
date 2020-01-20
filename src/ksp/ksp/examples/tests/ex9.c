
static char help[] = "Tests repeated setups and solves of PCFIELDSPLIT.\n\n";
#include <petscksp.h>

static PetscErrorCode replace_submats(Mat A)
{
  PetscErrorCode ierr;
  IS             *r,*c;
  PetscInt       i,j,nr,nc;

  PetscFunctionBeginUser;
  ierr = MatNestGetSubMats(A,&nr,&nc,NULL);CHKERRQ(ierr);
  ierr = PetscMalloc1(nr,&r);CHKERRQ(ierr);
  ierr = PetscMalloc1(nc,&c);CHKERRQ(ierr);
  ierr = MatNestGetISs(A,r,c);CHKERRQ(ierr);
  for (i=0;i<nr;i++) {
    for (j=0;j<nc;j++) {
      Mat        sA,nA;
      const char *prefix;

      ierr = MatCreateSubMatrix(A,r[i],c[j],MAT_INITIAL_MATRIX,&sA);CHKERRQ(ierr);
      ierr = MatDuplicate(sA,MAT_COPY_VALUES,&nA);CHKERRQ(ierr);
      ierr = MatGetOptionsPrefix(sA,&prefix);CHKERRQ(ierr);
      ierr = MatSetOptionsPrefix(nA,prefix);CHKERRQ(ierr);
      ierr = MatNestSetSubMat(A,i,j,nA);CHKERRQ(ierr);
      ierr = MatDestroy(&nA);CHKERRQ(ierr);
      ierr = MatDestroy(&sA);CHKERRQ(ierr);
    }
  }
  ierr = PetscFree(r);CHKERRQ(ierr);
  ierr = PetscFree(c);CHKERRQ(ierr);
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
   ierr = MatCreateAIJ(PETSC_COMM_WORLD,10,10,PETSC_DECIDE,PETSC_DECIDE,1,NULL,0,NULL,&M);CHKERRQ(ierr);
   ierr = MatAssemblyBegin(M,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
   ierr = MatAssemblyEnd(M,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
   ierr = MatShift(M,1.);CHKERRQ(ierr);
   ierr = MatGetOwnershipRange(M,&rstart,&rend);CHKERRQ(ierr);
   ierr = ISCreateStride(PetscObjectComm((PetscObject)M),7,rstart,1,&f[0]);CHKERRQ(ierr);
   ierr = ISComplement(f[0],rstart,rend,&f[1]);CHKERRQ(ierr);
   for (i=0;i<2;i++) {
     for (j=0;j<2;j++) {
       ierr = MatCreateSubMatrix(M,f[i],f[j],MAT_INITIAL_MATRIX,&sA[i][j]);CHKERRQ(ierr);
       ierr = MatCreateSubMatrix(M,f[i],f[j],MAT_INITIAL_MATRIX,&sP[i][j]);CHKERRQ(ierr);
     }
   }
   ierr = MatCreateNest(PetscObjectComm((PetscObject)M),2,f,2,f,&sA[0][0],&A);CHKERRQ(ierr);
   ierr = MatCreateNest(PetscObjectComm((PetscObject)M),2,f,2,f,&sP[0][0],&P);CHKERRQ(ierr);

   /* Tests MatMissingDiagonal_Nest */
   ierr = MatMissingDiagonal(M,&missM,NULL);CHKERRQ(ierr);
   ierr = MatMissingDiagonal(A,&missA,NULL);CHKERRQ(ierr);
   if (missM != missA) {
     ierr = PetscPrintf(PETSC_COMM_WORLD,"Unexpected %s != %s\n",missM ? "true": "false",missA ? "true" : "false");CHKERRQ(ierr);
   }

   ierr = MatDestroy(&M);CHKERRQ(ierr);

   ierr = KSPCreate(PetscObjectComm((PetscObject)A),&ksp);CHKERRQ(ierr);
   ierr = KSPSetOperators(ksp,A,P);CHKERRQ(ierr);
   ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
   ierr = PCSetType(pc,PCFIELDSPLIT);CHKERRQ(ierr);
   ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
   ierr = MatCreateVecs(A,&x,&b);CHKERRQ(ierr);
   ierr = VecSetRandom(b,NULL);CHKERRQ(ierr);
   ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);
   ierr = replace_submats(A);CHKERRQ(ierr);
   ierr = replace_submats(P);CHKERRQ(ierr);
   ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);

   ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
   ierr = VecDestroy(&x);CHKERRQ(ierr);
   ierr = VecDestroy(&b);CHKERRQ(ierr);
   ierr = MatDestroy(&A);CHKERRQ(ierr);
   ierr = MatDestroy(&P);CHKERRQ(ierr);
   for (i=0;i<2;i++) {
     ierr = ISDestroy(&f[i]);CHKERRQ(ierr);
     for (j=0;j<2;j++) {
       ierr = MatDestroy(&sA[i][j]);CHKERRQ(ierr);
       ierr = MatDestroy(&sP[i][j]);CHKERRQ(ierr);
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
