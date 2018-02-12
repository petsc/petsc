static char help[] =  "This example illustrates the use of PCBDDC/FETI-DP with 3D DMDA.\n\n";
/* Contributed by Wim Vanroose <wim@vanroo.se> */

#include <petscksp.h>
#include <petscpc.h>
#include <petscdm.h>
#include <petscdmda.h>

int main(int argc,char **args)
{
  KSP                    ksp;
  PC                     pc;
  Mat                    A;
  DM                     da;
  Vec                    x,b;
  PetscErrorCode         ierr;
  ISLocalToGlobalMapping map;
  MatNullSpace           nullsp;
  PetscBool              useglobal = PETSC_FALSE;
  PetscInt               i;
  PetscInt               nel,nen;        /* Number of elements & element nodes */
  const PetscInt         *e_loc;         /* Local indices of element nodes (in local element order) */
  PetscInt               *e_glo = NULL;  /* Global indices of element nodes (in local element order) */
  PetscScalar            elemMat[64] = { 0.4444, 0.0556,-0.1111,-0.1389, 0.0556,-0.0556,-0.1389,-0.1111,
                                         0.0556, 0.4444,-0.1389,-0.1111,-0.0556, 0.0556,-0.1111,-0.1389,
			                -0.1111,-0.1389, 0.4444, 0.0556,-0.1389,-0.1111, 0.0556,-0.0556,
			                -0.1389,-0.1111, 0.0556, 0.4444,-0.1111,-0.1389,-0.0556, 0.0556,
			                 0.0556,-0.0556,-0.1389,-0.1111, 0.4444, 0.0556,-0.1111,-0.1389,
			                -0.0556, 0.0556,-0.1111,-0.1389, 0.0556, 0.4444,-0.1389,-0.1111,
			                -0.1389,-0.1111, 0.0556,-0.0556,-0.1111,-0.1389, 0.4444, 0.0556,
			                -0.1111,-0.1389,-0.0556, 0.0556,-0.1389,-0.1111, 0.0556, 0.4444};

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;

  ierr = DMDACreate3d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_BOX,9,7,5,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,1,1,PETSC_NULL,PETSC_NULL,PETSC_NULL,&da);CHKERRQ(ierr);
  ierr = DMSetMatType(da,MATIS);CHKERRQ(ierr);
  ierr = DMSetFromOptions(da);CHKERRQ(ierr);
  ierr = DMDASetElementType(da, DMDA_ELEMENT_Q1);CHKERRQ(ierr);
  ierr = DMSetUp(da);CHKERRQ(ierr);

  ierr = DMCreateMatrix(da,&A);CHKERRQ(ierr);
  ierr = DMGetLocalToGlobalMapping(da,&map);CHKERRQ(ierr);
  ierr = DMDAGetElements(da,&nel,&nen,&e_loc);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-use_global",&useglobal,NULL);CHKERRQ(ierr);
  if (useglobal) {
    ierr = PetscMalloc1(nel*nen,&e_glo);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingApply(map,nen*nel,e_loc,e_glo);CHKERRQ(ierr);
  }
  for (i = 0; i < nel; ++i) {
    if (!e_glo) {
      ierr = MatSetValuesLocal(A,nen,e_loc+i*nen,nen,e_loc+i*nen,elemMat,ADD_VALUES);CHKERRQ(ierr);
    } else {
      ierr = MatSetValues(A,nen,e_glo+i*nen,nen,e_glo+i*nen,elemMat,ADD_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = DMDARestoreElements(da,&nel,&nen,&e_loc);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = MatNullSpaceCreate(PETSC_COMM_WORLD,PETSC_TRUE,0,NULL,&nullsp);CHKERRQ(ierr);
  ierr = MatSetNullSpace(A,nullsp);CHKERRQ(ierr);

  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);
  ierr = KSPSetType(ksp,KSPCG);CHKERRQ(ierr);
  ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
  ierr = PCSetType(pc,PCBDDC);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  ierr = KSPSetUp(ksp);CHKERRQ(ierr);

  ierr = DMGetGlobalVector(da,&x);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(da,&b);CHKERRQ(ierr);
  ierr = VecSetRandom(b,NULL);CHKERRQ(ierr);
  ierr = MatNullSpaceRemove(nullsp,b);CHKERRQ(ierr);
  ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(da,&x);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(da,&b);CHKERRQ(ierr);

  /* cleanup */
  ierr = PetscFree(e_glo);CHKERRQ(ierr);
  ierr = MatNullSpaceDestroy(&nullsp);CHKERRQ(ierr);
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

 test:
   nsize: 8
   filter: grep -v "variant HERMITIAN"
   suffix: bddc_1
   args: -ksp_view -pc_bddc_coarse_redundant_pc_type svd -ksp_error_if_not_converged
 test:
   nsize: 8
   filter: grep -v "variant HERMITIAN"
   suffix: bddc_2
   args: -ksp_view -use_global -pc_bddc_coarse_redundant_pc_type svd -ksp_error_if_not_converged
 test:
   nsize: 8
   filter: grep -v "variant HERMITIAN"
   suffix: fetidp_1
   args: -ksp_view -ksp_type fetidp -fetidp_ksp_type cg -fetidp_bddc_pc_bddc_coarse_redundant_pc_type svd -ksp_fetidp_fullyredundant -ksp_error_if_not_converged
 test:
   nsize: 8
   filter: grep -v "variant HERMITIAN"
   suffix: fetidp_2
   args: -ksp_view -use_global -ksp_type fetidp -fetidp_ksp_type cg -fetidp_bddc_pc_bddc_coarse_redundant_pc_type svd -ksp_fetidp_fullyredundant -ksp_error_if_not_converged

TEST*/
