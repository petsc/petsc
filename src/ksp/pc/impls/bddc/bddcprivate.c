#include "bddc.h"
#include "bddcprivate.h"
#include <petscblaslapack.h>

#undef __FUNCT__
#define __FUNCT__ "PCBDDCSetUpSolvers"
PetscErrorCode PCBDDCSetUpSolvers(PC pc)
{
  PC_BDDC*          pcbddc = (PC_BDDC*)pc->data;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  /* Compute matrix after change of basis and extract local submatrices */
  ierr = PCBDDCSetUpLocalMatrices(pc);CHKERRQ(ierr);

  /* Allocate needed vectors */
  ierr = PCBDDCCreateWorkVectors(pc);CHKERRQ(ierr);

  /* Setup local scatters R_to_B and (optionally) R_to_D : PCBDDCCreateWorkVectors should be called first! */
  ierr = PCBDDCSetUpLocalScatters(pc);CHKERRQ(ierr);

  /* Setup local solvers ksp_D and ksp_R */
  ierr = PCBDDCSetUpLocalSolvers(pc);CHKERRQ(ierr);

  /* Change global null space passed in by the user if change of basis has been requested */
  if (pcbddc->NullSpace && pcbddc->use_change_of_basis) {
    ierr = PCBDDCNullSpaceAdaptGlobal(pc);CHKERRQ(ierr);
  }

  /* setup local correction and local part of coarse basis */
  ierr = PCBDDCSetUpCoarseLocal(pc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCBDDCSetLevel"
PetscErrorCode PCBDDCSetLevel(PC pc,PetscInt level)
{
  PC_BDDC  *pcbddc = (PC_BDDC*)pc->data;

  PetscFunctionBegin;
  pcbddc->current_level=level;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCBDDCResetCustomization"
PetscErrorCode PCBDDCResetCustomization(PC pc)
{
  PC_BDDC        *pcbddc = (PC_BDDC*)pc->data;
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PCBDDCGraphResetCSR(pcbddc->mat_graph);CHKERRQ(ierr);
  ierr = ISDestroy(&pcbddc->user_primal_vertices);CHKERRQ(ierr);
  ierr = MatNullSpaceDestroy(&pcbddc->NullSpace);CHKERRQ(ierr);
  ierr = ISDestroy(&pcbddc->NeumannBoundaries);CHKERRQ(ierr);
  ierr = ISDestroy(&pcbddc->DirichletBoundaries);CHKERRQ(ierr);
  for (i=0;i<pcbddc->n_ISForDofs;i++) {
    ierr = ISDestroy(&pcbddc->ISForDofs[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree(pcbddc->ISForDofs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCBDDCResetTopography"
PetscErrorCode PCBDDCResetTopography(PC pc)
{
  PC_BDDC        *pcbddc = (PC_BDDC*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatDestroy(&pcbddc->ChangeOfBasisMatrix);CHKERRQ(ierr);
  ierr = MatDestroy(&pcbddc->ConstraintMatrix);CHKERRQ(ierr);
  ierr = PCBDDCGraphReset(pcbddc->mat_graph);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCBDDCResetSolvers"
PetscErrorCode PCBDDCResetSolvers(PC pc)
{
  PC_BDDC        *pcbddc = (PC_BDDC*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDestroy(&pcbddc->coarse_vec);CHKERRQ(ierr);
  ierr = VecDestroy(&pcbddc->coarse_rhs);CHKERRQ(ierr);
  ierr = KSPDestroy(&pcbddc->coarse_ksp);CHKERRQ(ierr);
  ierr = MatDestroy(&pcbddc->coarse_mat);CHKERRQ(ierr);
  ierr = MatDestroy(&pcbddc->coarse_phi_B);CHKERRQ(ierr);
  ierr = MatDestroy(&pcbddc->coarse_phi_D);CHKERRQ(ierr);
  ierr = MatDestroy(&pcbddc->coarse_psi_B);CHKERRQ(ierr);
  ierr = MatDestroy(&pcbddc->coarse_psi_D);CHKERRQ(ierr);
  ierr = VecDestroy(&pcbddc->vec1_P);CHKERRQ(ierr);
  ierr = VecDestroy(&pcbddc->vec1_C);CHKERRQ(ierr);
  ierr = MatDestroy(&pcbddc->local_auxmat1);CHKERRQ(ierr);
  ierr = MatDestroy(&pcbddc->local_auxmat2);CHKERRQ(ierr);
  ierr = VecDestroy(&pcbddc->vec1_R);CHKERRQ(ierr);
  ierr = VecDestroy(&pcbddc->vec2_R);CHKERRQ(ierr);
  ierr = VecDestroy(&pcbddc->vec4_D);CHKERRQ(ierr);
  ierr = ISDestroy(&pcbddc->is_R_local);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&pcbddc->R_to_B);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&pcbddc->R_to_D);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&pcbddc->coarse_loc_to_glob);CHKERRQ(ierr);
  ierr = PetscFree(pcbddc->local_primal_indices);CHKERRQ(ierr);
  ierr = PetscFree(pcbddc->replicated_local_primal_indices);CHKERRQ(ierr);
  ierr = PetscFree(pcbddc->replicated_local_primal_values);CHKERRQ(ierr);
  ierr = PetscFree(pcbddc->local_primal_displacements);CHKERRQ(ierr);
  ierr = PetscFree(pcbddc->local_primal_sizes);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCBDDCCreateWorkVectors"
PetscErrorCode PCBDDCCreateWorkVectors(PC pc)
{
  PC_BDDC        *pcbddc = (PC_BDDC*)pc->data;
  PC_IS          *pcis = (PC_IS*)pc->data;
  VecType        impVecType;
  PetscInt       n_vertices,n_constraints,local_primal_size,n_R;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PCBDDCGetPrimalVerticesLocalIdx(pc,&n_vertices,NULL);CHKERRQ(ierr);
  ierr = PCBDDCGetPrimalConstraintsLocalIdx(pc,&n_constraints,NULL);CHKERRQ(ierr);
  local_primal_size = n_constraints+n_vertices;
  n_R = pcis->n-n_vertices;
  /* local work vectors */
  ierr = VecGetType(pcis->vec1_N,&impVecType);CHKERRQ(ierr);
  ierr = VecDuplicate(pcis->vec1_D,&pcbddc->vec4_D);CHKERRQ(ierr);
  ierr = VecCreate(PetscObjectComm((PetscObject)pcis->vec1_N),&pcbddc->vec1_R);CHKERRQ(ierr);
  ierr = VecSetSizes(pcbddc->vec1_R,PETSC_DECIDE,n_R);CHKERRQ(ierr);
  ierr = VecSetType(pcbddc->vec1_R,impVecType);CHKERRQ(ierr);
  ierr = VecDuplicate(pcbddc->vec1_R,&pcbddc->vec2_R);CHKERRQ(ierr);
  ierr = VecCreate(PetscObjectComm((PetscObject)pcis->vec1_N),&pcbddc->vec1_P);CHKERRQ(ierr);
  ierr = VecSetSizes(pcbddc->vec1_P,PETSC_DECIDE,local_primal_size);CHKERRQ(ierr);
  ierr = VecSetType(pcbddc->vec1_P,impVecType);CHKERRQ(ierr);
  if (n_constraints) {
    ierr = VecCreate(PetscObjectComm((PetscObject)pcis->vec1_N),&pcbddc->vec1_C);CHKERRQ(ierr);
    ierr = VecSetSizes(pcbddc->vec1_C,PETSC_DECIDE,n_constraints);CHKERRQ(ierr);
    ierr = VecSetType(pcbddc->vec1_C,impVecType);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCBDDCSetUpCoarseLocal"
PetscErrorCode PCBDDCSetUpCoarseLocal(PC pc)
{
  PetscErrorCode         ierr;
  /* pointers to pcis and pcbddc */
  PC_IS*                 pcis = (PC_IS*)pc->data;
  PC_BDDC*               pcbddc = (PC_BDDC*)pc->data;
  /* submatrices of local problem */
  Mat                    A_RV,A_VR,A_VV;
  /* working matrices */
  Mat                    M1,M2,M3,C_CR;
  /* working vectors */
  Vec                    vec1_C,vec2_C,vec1_V,vec2_V;
  /* additional working stuff */
  IS                     is_aux;
  ISLocalToGlobalMapping BtoNmap;
  PetscScalar            *coarse_submat_vals; /* TODO: use a PETSc matrix */
  const PetscScalar      *array,*row_cmat_values;
  const PetscInt         *row_cmat_indices,*idx_R_local;
  PetscInt               *vertices,*idx_V_B,*auxindices;
  PetscInt               n_vertices,n_constraints,size_of_constraint;
  PetscInt               i,j,n_R,n_D,n_B;
  PetscBool              setsym=PETSC_FALSE,issym=PETSC_FALSE;
  /* Vector and matrix types */
  VecType                impVecType;
  MatType                impMatType;
  /* some shortcuts to scalars */
  PetscScalar            zero=0.0,one=1.0,m_one=-1.0;
  /* for debugging purposes */
  PetscReal              *coarsefunctions_errors,*constraints_errors;

  PetscFunctionBegin;
  /* get number of vertices and their local indices */
  ierr = PCBDDCGetPrimalVerticesLocalIdx(pc,&n_vertices,&vertices);CHKERRQ(ierr);
  n_constraints = pcbddc->local_primal_size-n_vertices;
  /* Set Non-overlapping dimensions */
  n_B = pcis->n_B; n_D = pcis->n - n_B;
  n_R = pcis->n-n_vertices;

  /* Set types for local objects needed by BDDC precondtioner */
  impMatType = MATSEQDENSE;
  ierr = VecGetType(pcis->vec1_N,&impVecType);CHKERRQ(ierr);

  /* Allocating some extra storage just to be safe */
  ierr = PetscMalloc (pcis->n*sizeof(PetscInt),&auxindices);CHKERRQ(ierr);
  for (i=0;i<pcis->n;i++) auxindices[i]=i;

  /* vertices in boundary numbering */
  ierr = PetscMalloc(n_vertices*sizeof(PetscInt),&idx_V_B);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingCreateIS(pcis->is_B_local,&BtoNmap);CHKERRQ(ierr);
  ierr = ISGlobalToLocalMappingApply(BtoNmap,IS_GTOLM_DROP,n_vertices,vertices,&i,idx_V_B);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingDestroy(&BtoNmap);CHKERRQ(ierr);
  if (i != n_vertices) {
    SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_SUP,"Error in boundary numbering for BDDC vertices! %d != %d\n",n_vertices,i);
  }

  /* some work vectors on vertices and/or constraints */
  if (n_vertices) {
    ierr = VecCreate(PETSC_COMM_SELF,&vec1_V);CHKERRQ(ierr);
    ierr = VecSetSizes(vec1_V,n_vertices,n_vertices);CHKERRQ(ierr);
    ierr = VecSetType(vec1_V,impVecType);CHKERRQ(ierr);
    ierr = VecDuplicate(vec1_V,&vec2_V);CHKERRQ(ierr);
  }
  if (n_constraints) {
    ierr = VecDuplicate(pcbddc->vec1_C,&vec1_C);CHKERRQ(ierr);
    ierr = VecDuplicate(pcbddc->vec1_C,&vec2_C);CHKERRQ(ierr);
  }

  /* Precompute stuffs needed for preprocessing and application of BDDC*/
  if (n_constraints) {
    ierr = MatCreate(PETSC_COMM_SELF,&pcbddc->local_auxmat2);CHKERRQ(ierr);
    ierr = MatSetSizes(pcbddc->local_auxmat2,n_R,n_constraints,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
    ierr = MatSetType(pcbddc->local_auxmat2,impMatType);CHKERRQ(ierr);
    ierr = MatSetUp(pcbddc->local_auxmat2);CHKERRQ(ierr);

    /* Extract constraints on R nodes: C_{CR}  */
    ierr = ISCreateStride(PETSC_COMM_SELF,n_constraints,n_vertices,1,&is_aux);CHKERRQ(ierr);
    ierr = MatGetSubMatrix(pcbddc->ConstraintMatrix,is_aux,pcbddc->is_R_local,MAT_INITIAL_MATRIX,&C_CR);CHKERRQ(ierr);
    ierr = ISDestroy(&is_aux);CHKERRQ(ierr);

    /* Assemble local_auxmat2 = - A_{RR}^{-1} C^T_{CR} needed by BDDC application */
    for (i=0;i<n_constraints;i++) {
      ierr = VecSet(pcbddc->vec1_R,zero);CHKERRQ(ierr);
      /* Get row of constraint matrix in R numbering */
      ierr = MatGetRow(C_CR,i,&size_of_constraint,&row_cmat_indices,&row_cmat_values);CHKERRQ(ierr);
      ierr = VecSetValues(pcbddc->vec1_R,size_of_constraint,row_cmat_indices,row_cmat_values,INSERT_VALUES);CHKERRQ(ierr);
      ierr = MatRestoreRow(C_CR,i,&size_of_constraint,&row_cmat_indices,&row_cmat_values);CHKERRQ(ierr);
      ierr = VecAssemblyBegin(pcbddc->vec1_R);CHKERRQ(ierr);
      ierr = VecAssemblyEnd(pcbddc->vec1_R);CHKERRQ(ierr);
      /* Solve for row of constraint matrix in R numbering */
      ierr = KSPSolve(pcbddc->ksp_R,pcbddc->vec1_R,pcbddc->vec2_R);CHKERRQ(ierr);
      /* Set values in local_auxmat2 */
      ierr = VecGetArrayRead(pcbddc->vec2_R,&array);CHKERRQ(ierr);
      ierr = MatSetValues(pcbddc->local_auxmat2,n_R,auxindices,1,&i,array,INSERT_VALUES);CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(pcbddc->vec2_R,&array);CHKERRQ(ierr);
    }
    ierr = MatAssemblyBegin(pcbddc->local_auxmat2,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(pcbddc->local_auxmat2,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatScale(pcbddc->local_auxmat2,m_one);CHKERRQ(ierr);

    /* Assemble explicitly M1 = ( C_{CR} A_{RR}^{-1} C^T_{CR} )^{-1} needed in preproc  */
    ierr = MatMatMult(C_CR,pcbddc->local_auxmat2,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&M3);CHKERRQ(ierr);
    ierr = MatLUFactor(M3,NULL,NULL,NULL);CHKERRQ(ierr);
    ierr = MatCreate(PETSC_COMM_SELF,&M1);CHKERRQ(ierr);
    ierr = MatSetSizes(M1,n_constraints,n_constraints,n_constraints,n_constraints);CHKERRQ(ierr);
    ierr = MatSetType(M1,impMatType);CHKERRQ(ierr);
    ierr = MatSetUp(M1);CHKERRQ(ierr);
    ierr = MatDuplicate(M1,MAT_DO_NOT_COPY_VALUES,&M2);CHKERRQ(ierr);
    ierr = MatZeroEntries(M2);CHKERRQ(ierr);
    ierr = VecSet(vec1_C,m_one);CHKERRQ(ierr);
    ierr = MatDiagonalSet(M2,vec1_C,INSERT_VALUES);CHKERRQ(ierr); 
    ierr = MatMatSolve(M3,M2,M1);CHKERRQ(ierr);
    ierr = MatDestroy(&M2);CHKERRQ(ierr);
    ierr = MatDestroy(&M3);CHKERRQ(ierr);
    /* Assemble local_auxmat1 = M1*C_{CR} needed by BDDC application in KSP and in preproc */
    ierr = MatMatMult(M1,C_CR,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&pcbddc->local_auxmat1);CHKERRQ(ierr);
  }

  /* Get submatrices from subdomain matrix */
  if (n_vertices) {
    ierr = ISCreateGeneral(PETSC_COMM_SELF,n_vertices,vertices,PETSC_COPY_VALUES,&is_aux);CHKERRQ(ierr);
    ierr = MatGetSubMatrix(pcbddc->local_mat,pcbddc->is_R_local,is_aux,MAT_INITIAL_MATRIX,&A_RV);CHKERRQ(ierr);
    ierr = MatGetSubMatrix(pcbddc->local_mat,is_aux,pcbddc->is_R_local,MAT_INITIAL_MATRIX,&A_VR);CHKERRQ(ierr);
    ierr = MatGetSubMatrix(pcbddc->local_mat,is_aux,is_aux,MAT_INITIAL_MATRIX,&A_VV);CHKERRQ(ierr);
    ierr = ISDestroy(&is_aux);CHKERRQ(ierr);
  }

  /* Matrix of coarse basis functions (local) */
  ierr = MatCreate(PETSC_COMM_SELF,&pcbddc->coarse_phi_B);CHKERRQ(ierr);
  ierr = MatSetSizes(pcbddc->coarse_phi_B,n_B,pcbddc->local_primal_size,n_B,pcbddc->local_primal_size);CHKERRQ(ierr);
  ierr = MatSetType(pcbddc->coarse_phi_B,impMatType);CHKERRQ(ierr);
  ierr = MatSetUp(pcbddc->coarse_phi_B);CHKERRQ(ierr);
  if (pcbddc->inexact_prec_type || pcbddc->dbg_flag) {
    ierr = MatCreate(PETSC_COMM_SELF,&pcbddc->coarse_phi_D);CHKERRQ(ierr);
    ierr = MatSetSizes(pcbddc->coarse_phi_D,n_D,pcbddc->local_primal_size,n_D,pcbddc->local_primal_size);CHKERRQ(ierr);
    ierr = MatSetType(pcbddc->coarse_phi_D,impMatType);CHKERRQ(ierr);
    ierr = MatSetUp(pcbddc->coarse_phi_D);CHKERRQ(ierr);
  }

  if (pcbddc->dbg_flag) {
    ierr = ISGetIndices(pcbddc->is_R_local,&idx_R_local);CHKERRQ(ierr);
    ierr = PetscMalloc(2*pcbddc->local_primal_size*sizeof(*coarsefunctions_errors),&coarsefunctions_errors);CHKERRQ(ierr);
    ierr = PetscMalloc(2*pcbddc->local_primal_size*sizeof(*constraints_errors),&constraints_errors);CHKERRQ(ierr);
  }
  /* Subdomain contribution (Non-overlapping) to coarse matrix  */
  ierr = PetscMalloc((pcbddc->local_primal_size)*(pcbddc->local_primal_size)*sizeof(PetscScalar),&coarse_submat_vals);CHKERRQ(ierr);

  /* We are now ready to evaluate coarse basis functions and subdomain contribution to coarse problem */

  /* vertices */
  for (i=0;i<n_vertices;i++) {
    ierr = VecSet(vec1_V,zero);CHKERRQ(ierr);
    ierr = VecSetValue(vec1_V,i,one,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecAssemblyBegin(vec1_V);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(vec1_V);CHKERRQ(ierr);
    /* simplified solution of saddle point problem with null rhs on constraints multipliers */
    ierr = MatMult(A_RV,vec1_V,pcbddc->vec1_R);CHKERRQ(ierr);
    ierr = KSPSolve(pcbddc->ksp_R,pcbddc->vec1_R,pcbddc->vec1_R);CHKERRQ(ierr);
    ierr = VecScale(pcbddc->vec1_R,m_one);CHKERRQ(ierr);
    if (n_constraints) {
      ierr = MatMult(pcbddc->local_auxmat1,pcbddc->vec1_R,vec1_C);CHKERRQ(ierr);
      ierr = MatMultAdd(pcbddc->local_auxmat2,vec1_C,pcbddc->vec1_R,pcbddc->vec1_R);CHKERRQ(ierr);
      ierr = VecScale(vec1_C,m_one);CHKERRQ(ierr);
    }
    ierr = MatMult(A_VR,pcbddc->vec1_R,vec2_V);CHKERRQ(ierr);
    ierr = MatMultAdd(A_VV,vec1_V,vec2_V,vec2_V);CHKERRQ(ierr);

    /* Set values in coarse basis function and subdomain part of coarse_mat */
    /* coarse basis functions */
    ierr = VecSet(pcis->vec1_B,zero);CHKERRQ(ierr);
    ierr = VecScatterBegin(pcbddc->R_to_B,pcbddc->vec1_R,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(pcbddc->R_to_B,pcbddc->vec1_R,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecGetArrayRead(pcis->vec1_B,&array);CHKERRQ(ierr);
    ierr = MatSetValues(pcbddc->coarse_phi_B,n_B,auxindices,1,&i,array,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(pcis->vec1_B,&array);CHKERRQ(ierr);
    ierr = MatSetValue(pcbddc->coarse_phi_B,idx_V_B[i],i,one,INSERT_VALUES);CHKERRQ(ierr);
    if (pcbddc->inexact_prec_type || pcbddc->dbg_flag) {
      ierr = VecScatterBegin(pcbddc->R_to_D,pcbddc->vec1_R,pcis->vec1_D,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecScatterEnd(pcbddc->R_to_D,pcbddc->vec1_R,pcis->vec1_D,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecGetArrayRead(pcis->vec1_D,&array);CHKERRQ(ierr);
      ierr = MatSetValues(pcbddc->coarse_phi_D,n_D,auxindices,1,&i,array,INSERT_VALUES);CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(pcis->vec1_D,&array);CHKERRQ(ierr);
    }
    /* subdomain contribution to coarse matrix. WARNING -> column major ordering */
    ierr = VecGetArrayRead(vec2_V,&array);CHKERRQ(ierr);
    ierr = PetscMemcpy(&coarse_submat_vals[i*pcbddc->local_primal_size],array,n_vertices*sizeof(PetscScalar));CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(vec2_V,&array);CHKERRQ(ierr);
    if (n_constraints) {
      ierr = VecGetArrayRead(vec1_C,&array);CHKERRQ(ierr);
      ierr = PetscMemcpy(&coarse_submat_vals[i*pcbddc->local_primal_size+n_vertices],array,n_constraints*sizeof(PetscScalar));CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(vec1_C,&array);CHKERRQ(ierr);
    }

    /* check */
    if (pcbddc->dbg_flag) {
      /* assemble subdomain vector on local nodes */
      ierr = VecSet(pcis->vec1_N,zero);CHKERRQ(ierr);
      ierr = VecGetArrayRead(pcbddc->vec1_R,&array);CHKERRQ(ierr);
      ierr = VecSetValues(pcis->vec1_N,n_R,idx_R_local,array,INSERT_VALUES);CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(pcbddc->vec1_R,&array);CHKERRQ(ierr);
      ierr = VecSetValue(pcis->vec1_N,vertices[i],one,INSERT_VALUES);CHKERRQ(ierr);
      ierr = VecAssemblyBegin(pcis->vec1_N);CHKERRQ(ierr);
      ierr = VecAssemblyEnd(pcis->vec1_N);CHKERRQ(ierr);
      /* assemble subdomain vector of lagrange multipliers (i.e. primal nodes) */
      ierr = VecSet(pcbddc->vec1_P,zero);CHKERRQ(ierr);
      ierr = VecGetArrayRead(vec2_V,&array);CHKERRQ(ierr);
      ierr = VecSetValues(pcbddc->vec1_P,n_vertices,auxindices,array,INSERT_VALUES);CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(vec2_V,&array);CHKERRQ(ierr);
      if (n_constraints) {
        ierr = VecGetArrayRead(vec1_C,&array);CHKERRQ(ierr);
        ierr = VecSetValues(pcbddc->vec1_P,n_constraints,&auxindices[n_vertices],array,INSERT_VALUES);CHKERRQ(ierr);
        ierr = VecRestoreArrayRead(vec1_C,&array);CHKERRQ(ierr);
      }
      ierr = VecAssemblyBegin(pcbddc->vec1_P);CHKERRQ(ierr);
      ierr = VecAssemblyEnd(pcbddc->vec1_P);CHKERRQ(ierr);
      ierr = VecScale(pcbddc->vec1_P,m_one);CHKERRQ(ierr);
      /* check saddle point solution */
      ierr = MatMult(pcbddc->local_mat,pcis->vec1_N,pcis->vec2_N);CHKERRQ(ierr);
      ierr = MatMultTransposeAdd(pcbddc->ConstraintMatrix,pcbddc->vec1_P,pcis->vec2_N,pcis->vec2_N);CHKERRQ(ierr);
      ierr = VecNorm(pcis->vec2_N,NORM_INFINITY,&coarsefunctions_errors[i]);CHKERRQ(ierr);
      ierr = MatMult(pcbddc->ConstraintMatrix,pcis->vec1_N,pcbddc->vec1_P);CHKERRQ(ierr);
      /* shift by the identity matrix */
      ierr = VecSetValue(pcbddc->vec1_P,i,m_one,ADD_VALUES);CHKERRQ(ierr);
      ierr = VecAssemblyBegin(pcbddc->vec1_P);CHKERRQ(ierr);
      ierr = VecAssemblyEnd(pcbddc->vec1_P);CHKERRQ(ierr);
      ierr = VecNorm(pcbddc->vec1_P,NORM_INFINITY,&constraints_errors[i]);CHKERRQ(ierr);
    }
  }

  /* constraints */
  for (i=0;i<n_constraints;i++) {
    ierr = VecSet(vec2_C,zero);CHKERRQ(ierr);
    ierr = VecSetValue(vec2_C,i,m_one,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecAssemblyBegin(vec2_C);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(vec2_C);CHKERRQ(ierr);
    /* simplified solution of saddle point problem with null rhs on vertices multipliers */
    ierr = MatMult(M1,vec2_C,vec1_C);CHKERRQ(ierr);
    ierr = MatMult(pcbddc->local_auxmat2,vec1_C,pcbddc->vec1_R);CHKERRQ(ierr);
    ierr = VecScale(vec1_C,m_one);CHKERRQ(ierr);
    if (n_vertices) {
      ierr = MatMult(A_VR,pcbddc->vec1_R,vec2_V);CHKERRQ(ierr);
    }
    /* Set values in coarse basis function and subdomain part of coarse_mat */
    /* coarse basis functions */
    j = i+n_vertices; /* don't touch this! */
    ierr = VecSet(pcis->vec1_B,zero);CHKERRQ(ierr);
    ierr = VecScatterBegin(pcbddc->R_to_B,pcbddc->vec1_R,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(pcbddc->R_to_B,pcbddc->vec1_R,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecGetArrayRead(pcis->vec1_B,&array);CHKERRQ(ierr);
    ierr = MatSetValues(pcbddc->coarse_phi_B,n_B,auxindices,1,&j,array,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(pcis->vec1_B,&array);CHKERRQ(ierr);
    if (pcbddc->inexact_prec_type || pcbddc->dbg_flag) {
      ierr = VecScatterBegin(pcbddc->R_to_D,pcbddc->vec1_R,pcis->vec1_D,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecScatterEnd(pcbddc->R_to_D,pcbddc->vec1_R,pcis->vec1_D,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecGetArrayRead(pcis->vec1_D,&array);CHKERRQ(ierr);
      ierr = MatSetValues(pcbddc->coarse_phi_D,n_D,auxindices,1,&j,array,INSERT_VALUES);CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(pcis->vec1_D,&array);CHKERRQ(ierr);
    }
    /* subdomain contribution to coarse matrix. WARNING -> column major ordering */
    if (n_vertices) {
      ierr = VecGetArrayRead(vec2_V,&array);CHKERRQ(ierr);
      ierr = PetscMemcpy(&coarse_submat_vals[j*pcbddc->local_primal_size],array,n_vertices*sizeof(PetscScalar));CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(vec2_V,&array);CHKERRQ(ierr);
    }
    ierr = VecGetArrayRead(vec1_C,&array);CHKERRQ(ierr);
    ierr = PetscMemcpy(&coarse_submat_vals[j*pcbddc->local_primal_size+n_vertices],array,n_constraints*sizeof(PetscScalar));CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(vec1_C,&array);CHKERRQ(ierr);

    if (pcbddc->dbg_flag) {
      /* assemble subdomain vector on nodes */
      ierr = VecSet(pcis->vec1_N,zero);CHKERRQ(ierr);
      ierr = VecGetArrayRead(pcbddc->vec1_R,&array);CHKERRQ(ierr);
      ierr = VecSetValues(pcis->vec1_N,n_R,idx_R_local,array,INSERT_VALUES);CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(pcbddc->vec1_R,&array);CHKERRQ(ierr);
      ierr = VecAssemblyBegin(pcis->vec1_N);CHKERRQ(ierr);
      ierr = VecAssemblyEnd(pcis->vec1_N);CHKERRQ(ierr);
      /* assemble subdomain vector of lagrange multipliers */
      ierr = VecSet(pcbddc->vec1_P,zero);CHKERRQ(ierr);
      if (n_vertices) {
        ierr = VecGetArrayRead(vec2_V,&array);CHKERRQ(ierr);
        ierr = VecSetValues(pcbddc->vec1_P,n_vertices,auxindices,array,INSERT_VALUES);CHKERRQ(ierr);
        ierr = VecRestoreArrayRead(vec2_V,&array);CHKERRQ(ierr);
      }
      ierr = VecGetArrayRead(vec1_C,&array);CHKERRQ(ierr);
      ierr = VecSetValues(pcbddc->vec1_P,n_constraints,&auxindices[n_vertices],array,INSERT_VALUES);CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(vec1_C,&array);CHKERRQ(ierr);
      ierr = VecAssemblyBegin(pcbddc->vec1_P);CHKERRQ(ierr);
      ierr = VecAssemblyEnd(pcbddc->vec1_P);CHKERRQ(ierr);
      ierr = VecScale(pcbddc->vec1_P,m_one);CHKERRQ(ierr);
      /* check saddle point solution */
      ierr = MatMult(pcbddc->local_mat,pcis->vec1_N,pcis->vec2_N);CHKERRQ(ierr);
      ierr = MatMultTransposeAdd(pcbddc->ConstraintMatrix,pcbddc->vec1_P,pcis->vec2_N,pcis->vec2_N);CHKERRQ(ierr);
      ierr = VecNorm(pcis->vec2_N,NORM_INFINITY,&coarsefunctions_errors[j]);CHKERRQ(ierr);
      ierr = MatMult(pcbddc->ConstraintMatrix,pcis->vec1_N,pcbddc->vec1_P);CHKERRQ(ierr);
      /* shift by the identity matrix */
      ierr = VecSetValue(pcbddc->vec1_P,j,m_one,ADD_VALUES);CHKERRQ(ierr);
      ierr = VecAssemblyBegin(pcbddc->vec1_P);CHKERRQ(ierr);
      ierr = VecAssemblyEnd(pcbddc->vec1_P);CHKERRQ(ierr);
      ierr = VecNorm(pcbddc->vec1_P,NORM_INFINITY,&constraints_errors[j]);CHKERRQ(ierr);
    }
  }
  /* call assembling routines for local coarse basis */
  ierr = MatAssemblyBegin(pcbddc->coarse_phi_B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(pcbddc->coarse_phi_B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (pcbddc->inexact_prec_type || pcbddc->dbg_flag) {
    ierr = MatAssemblyBegin(pcbddc->coarse_phi_D,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(pcbddc->coarse_phi_D,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }

  /* compute other basis functions for non-symmetric problems */
  ierr = MatIsSymmetricKnown(pc->pmat,&setsym,&issym);CHKERRQ(ierr);
  if (!setsym || (setsym && !issym)) {
    ierr = MatCreate(PETSC_COMM_SELF,&pcbddc->coarse_psi_B);CHKERRQ(ierr);
    ierr = MatSetSizes(pcbddc->coarse_psi_B,n_B,pcbddc->local_primal_size,n_B,pcbddc->local_primal_size);CHKERRQ(ierr);
    ierr = MatSetType(pcbddc->coarse_psi_B,impMatType);CHKERRQ(ierr);
    ierr = MatSetUp(pcbddc->coarse_psi_B);CHKERRQ(ierr);
    if (pcbddc->inexact_prec_type || pcbddc->dbg_flag ) {
      ierr = MatCreate(PETSC_COMM_SELF,&pcbddc->coarse_psi_D);CHKERRQ(ierr);
      ierr = MatSetSizes(pcbddc->coarse_psi_D,n_D,pcbddc->local_primal_size,n_D,pcbddc->local_primal_size);CHKERRQ(ierr);
      ierr = MatSetType(pcbddc->coarse_psi_D,impMatType);CHKERRQ(ierr);
      ierr = MatSetUp(pcbddc->coarse_psi_D);CHKERRQ(ierr);
    }
    for (i=0;i<pcbddc->local_primal_size;i++) {
      if (n_constraints) {
        ierr = VecSet(vec1_C,zero);CHKERRQ(ierr);
        for (j=0;j<n_constraints;j++) {
          ierr = VecSetValue(vec1_C,j,coarse_submat_vals[(j+n_vertices)*pcbddc->local_primal_size+i],INSERT_VALUES);CHKERRQ(ierr);
        } 
        ierr = VecAssemblyBegin(vec1_C);CHKERRQ(ierr);
        ierr = VecAssemblyEnd(vec1_C);CHKERRQ(ierr);
      }
      if (i<n_vertices) {
        ierr = VecSet(vec1_V,zero);CHKERRQ(ierr);
        ierr = VecSetValue(vec1_V,i,m_one,INSERT_VALUES);CHKERRQ(ierr);
        ierr = VecAssemblyBegin(vec1_V);CHKERRQ(ierr);
        ierr = VecAssemblyEnd(vec1_V);CHKERRQ(ierr);
        ierr = MatMultTranspose(A_VR,vec1_V,pcbddc->vec1_R);CHKERRQ(ierr);
        if (n_constraints) {
          ierr = MatMultTransposeAdd(C_CR,vec1_C,pcbddc->vec1_R,pcbddc->vec1_R);CHKERRQ(ierr);
        }
      } else {
        ierr = MatMultTranspose(C_CR,vec1_C,pcbddc->vec1_R);CHKERRQ(ierr);
      }
      ierr = KSPSolveTranspose(pcbddc->ksp_R,pcbddc->vec1_R,pcbddc->vec1_R);CHKERRQ(ierr);
      ierr = VecSet(pcis->vec1_B,zero);CHKERRQ(ierr);
      ierr = VecScatterBegin(pcbddc->R_to_B,pcbddc->vec1_R,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecScatterEnd(pcbddc->R_to_B,pcbddc->vec1_R,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecGetArrayRead(pcis->vec1_B,&array);CHKERRQ(ierr);
      ierr = MatSetValues(pcbddc->coarse_psi_B,n_B,auxindices,1,&i,array,INSERT_VALUES);CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(pcis->vec1_B,&array);CHKERRQ(ierr);
      if (i<n_vertices) {
        ierr = MatSetValue(pcbddc->coarse_psi_B,idx_V_B[i],i,one,INSERT_VALUES);CHKERRQ(ierr);
      }
      if (pcbddc->inexact_prec_type || pcbddc->dbg_flag) {
        ierr = VecScatterBegin(pcbddc->R_to_D,pcbddc->vec1_R,pcis->vec1_D,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
        ierr = VecScatterEnd(pcbddc->R_to_D,pcbddc->vec1_R,pcis->vec1_D,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
        ierr = VecGetArrayRead(pcis->vec1_D,&array);CHKERRQ(ierr);
        ierr = MatSetValues(pcbddc->coarse_psi_D,n_D,auxindices,1,&i,array,INSERT_VALUES);CHKERRQ(ierr);
        ierr = VecRestoreArrayRead(pcis->vec1_D,&array);CHKERRQ(ierr);
      }

      if (pcbddc->dbg_flag) {
        /* assemble subdomain vector on nodes */
        ierr = VecSet(pcis->vec1_N,zero);CHKERRQ(ierr);
        ierr = VecGetArrayRead(pcbddc->vec1_R,&array);CHKERRQ(ierr);
        ierr = VecSetValues(pcis->vec1_N,n_R,idx_R_local,array,INSERT_VALUES);CHKERRQ(ierr);
        ierr = VecRestoreArrayRead(pcbddc->vec1_R,&array);CHKERRQ(ierr);
        if (i<n_vertices) {
          ierr = VecSetValue(pcis->vec1_N,vertices[i],one,INSERT_VALUES);CHKERRQ(ierr);
        }
        ierr = VecAssemblyBegin(pcis->vec1_N);CHKERRQ(ierr);
        ierr = VecAssemblyEnd(pcis->vec1_N);CHKERRQ(ierr);
        /* assemble subdomain vector of lagrange multipliers */
        for (j=0;j<pcbddc->local_primal_size;j++) {
          ierr = VecSetValue(pcbddc->vec1_P,j,-coarse_submat_vals[j*pcbddc->local_primal_size+i],INSERT_VALUES);CHKERRQ(ierr);
        } 
        ierr = VecAssemblyBegin(pcbddc->vec1_P);CHKERRQ(ierr);
        ierr = VecAssemblyEnd(pcbddc->vec1_P);CHKERRQ(ierr);
        /* check saddle point solution */
        ierr = MatMultTranspose(pcbddc->local_mat,pcis->vec1_N,pcis->vec2_N);CHKERRQ(ierr);
        ierr = MatMultTransposeAdd(pcbddc->ConstraintMatrix,pcbddc->vec1_P,pcis->vec2_N,pcis->vec2_N);CHKERRQ(ierr);
        ierr = VecNorm(pcis->vec2_N,NORM_INFINITY,&coarsefunctions_errors[i+pcbddc->local_primal_size]);CHKERRQ(ierr);
        ierr = MatMult(pcbddc->ConstraintMatrix,pcis->vec1_N,pcbddc->vec1_P);CHKERRQ(ierr);
        /* shift by the identity matrix */
        ierr = VecSetValue(pcbddc->vec1_P,i,m_one,ADD_VALUES);CHKERRQ(ierr);
        ierr = VecAssemblyBegin(pcbddc->vec1_P);CHKERRQ(ierr);
        ierr = VecAssemblyEnd(pcbddc->vec1_P);CHKERRQ(ierr);
        ierr = VecNorm(pcbddc->vec1_P,NORM_INFINITY,&constraints_errors[i+pcbddc->local_primal_size]);CHKERRQ(ierr);
      }
    }
    ierr = MatAssemblyBegin(pcbddc->coarse_psi_B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(pcbddc->coarse_psi_B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    if ( pcbddc->inexact_prec_type || pcbddc->dbg_flag ) {
      ierr = MatAssemblyBegin(pcbddc->coarse_psi_D,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      ierr = MatAssemblyEnd(pcbddc->coarse_psi_D,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    }
  }
  ierr = PetscFree(idx_V_B);CHKERRQ(ierr);
  /* Checking coarse_sub_mat and coarse basis functios */
  /* Symmetric case     : It should be \Phi^{(j)^T} A^{(j)} \Phi^{(j)}=coarse_sub_mat */
  /* Non-symmetric case : It should be \Psi^{(j)^T} A^{(j)} \Phi^{(j)}=coarse_sub_mat */
  if (pcbddc->dbg_flag) {
    Mat         coarse_sub_mat;
    Mat         AUXMAT,TM1,TM2,TM3,TM4;
    Mat         coarse_phi_D,coarse_phi_B;
    Mat         coarse_psi_D,coarse_psi_B;
    Mat         A_II,A_BB,A_IB,A_BI;
    MatType     checkmattype=MATSEQAIJ;
    PetscReal   real_value;

    ierr = MatConvert(pcis->A_II,checkmattype,MAT_INITIAL_MATRIX,&A_II);CHKERRQ(ierr);
    ierr = MatConvert(pcis->A_IB,checkmattype,MAT_INITIAL_MATRIX,&A_IB);CHKERRQ(ierr);
    ierr = MatConvert(pcis->A_BI,checkmattype,MAT_INITIAL_MATRIX,&A_BI);CHKERRQ(ierr);
    ierr = MatConvert(pcis->A_BB,checkmattype,MAT_INITIAL_MATRIX,&A_BB);CHKERRQ(ierr);
    ierr = MatConvert(pcbddc->coarse_phi_D,checkmattype,MAT_INITIAL_MATRIX,&coarse_phi_D);CHKERRQ(ierr);
    ierr = MatConvert(pcbddc->coarse_phi_B,checkmattype,MAT_INITIAL_MATRIX,&coarse_phi_B);CHKERRQ(ierr);
    if (pcbddc->coarse_psi_B) {
      ierr = MatConvert(pcbddc->coarse_psi_D,checkmattype,MAT_INITIAL_MATRIX,&coarse_psi_D);CHKERRQ(ierr);
      ierr = MatConvert(pcbddc->coarse_psi_B,checkmattype,MAT_INITIAL_MATRIX,&coarse_psi_B);CHKERRQ(ierr);
    }
    ierr = MatCreateSeqDense(PETSC_COMM_SELF,pcbddc->local_primal_size,pcbddc->local_primal_size,coarse_submat_vals,&coarse_sub_mat);CHKERRQ(ierr);

    ierr = PetscViewerASCIIPrintf(pcbddc->dbg_viewer,"--------------------------------------------------\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(pcbddc->dbg_viewer,"Check coarse sub mat and local basis functions\n");CHKERRQ(ierr);
    ierr = PetscViewerFlush(pcbddc->dbg_viewer);CHKERRQ(ierr);
    if (pcbddc->coarse_psi_B) {
      ierr = MatMatMult(A_II,coarse_phi_D,MAT_INITIAL_MATRIX,1.0,&AUXMAT);CHKERRQ(ierr);
      ierr = MatTransposeMatMult(coarse_psi_D,AUXMAT,MAT_INITIAL_MATRIX,1.0,&TM1);CHKERRQ(ierr);
      ierr = MatDestroy(&AUXMAT);CHKERRQ(ierr);
      ierr = MatMatMult(A_BB,coarse_phi_B,MAT_INITIAL_MATRIX,1.0,&AUXMAT);CHKERRQ(ierr);
      ierr = MatTransposeMatMult(coarse_psi_B,AUXMAT,MAT_INITIAL_MATRIX,1.0,&TM2);CHKERRQ(ierr);
      ierr = MatDestroy(&AUXMAT);CHKERRQ(ierr);
      ierr = MatMatMult(A_IB,coarse_phi_B,MAT_INITIAL_MATRIX,1.0,&AUXMAT);CHKERRQ(ierr);
      ierr = MatTransposeMatMult(coarse_psi_D,AUXMAT,MAT_INITIAL_MATRIX,1.0,&TM3);CHKERRQ(ierr);
      ierr = MatDestroy(&AUXMAT);CHKERRQ(ierr);
      ierr = MatMatMult(A_BI,coarse_phi_D,MAT_INITIAL_MATRIX,1.0,&AUXMAT);CHKERRQ(ierr);
      ierr = MatTransposeMatMult(coarse_psi_B,AUXMAT,MAT_INITIAL_MATRIX,1.0,&TM4);CHKERRQ(ierr);
      ierr = MatDestroy(&AUXMAT);CHKERRQ(ierr);
    } else {
      ierr = MatPtAP(A_II,coarse_phi_D,MAT_INITIAL_MATRIX,1.0,&TM1);CHKERRQ(ierr);
      ierr = MatPtAP(A_BB,coarse_phi_B,MAT_INITIAL_MATRIX,1.0,&TM2);CHKERRQ(ierr);
      ierr = MatMatMult(A_IB,coarse_phi_B,MAT_INITIAL_MATRIX,1.0,&AUXMAT);CHKERRQ(ierr);
      ierr = MatTransposeMatMult(coarse_phi_D,AUXMAT,MAT_INITIAL_MATRIX,1.0,&TM3);CHKERRQ(ierr);
      ierr = MatDestroy(&AUXMAT);CHKERRQ(ierr);
      ierr = MatMatMult(A_BI,coarse_phi_D,MAT_INITIAL_MATRIX,1.0,&AUXMAT);CHKERRQ(ierr);
      ierr = MatTransposeMatMult(coarse_phi_B,AUXMAT,MAT_INITIAL_MATRIX,1.0,&TM4);CHKERRQ(ierr);
      ierr = MatDestroy(&AUXMAT);CHKERRQ(ierr);
    }
    ierr = MatAXPY(TM1,one,TM2,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = MatAXPY(TM1,one,TM3,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = MatAXPY(TM1,one,TM4,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = MatConvert(TM1,MATSEQDENSE,MAT_REUSE_MATRIX,&TM1);CHKERRQ(ierr);
    ierr = MatAXPY(TM1,m_one,coarse_sub_mat,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = MatNorm(TM1,NORM_INFINITY,&real_value);CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"----------------------------------\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"Subdomain %04d \n",PetscGlobalRank);CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"matrix error = % 1.14e\n",real_value);CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"coarse functions (phi) errors\n");CHKERRQ(ierr);
    for (i=0;i<pcbddc->local_primal_size;i++) {
      ierr = PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"local %02d-th function error = % 1.14e\n",i,coarsefunctions_errors[i]);CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"constraints (phi) errors\n");CHKERRQ(ierr);
    for (i=0;i<pcbddc->local_primal_size;i++) {
      ierr = PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"local %02d-th function error = % 1.14e\n",i,constraints_errors[i]);CHKERRQ(ierr);
    }
    if (pcbddc->coarse_psi_B) {
      ierr = PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"coarse functions (psi) errors\n");CHKERRQ(ierr);
      for (i=pcbddc->local_primal_size;i<2*pcbddc->local_primal_size;i++) {
        ierr = PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"local %02d-th function error = % 1.14e\n",i-pcbddc->local_primal_size,coarsefunctions_errors[i]);CHKERRQ(ierr);
      }
      ierr = PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"constraints (psi) errors\n");CHKERRQ(ierr);
      for (i=pcbddc->local_primal_size;i<2*pcbddc->local_primal_size;i++) {
        ierr = PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"local %02d-th function error = % 1.14e\n",i-pcbddc->local_primal_size,constraints_errors[i]);CHKERRQ(ierr);
      }
    }
    ierr = PetscViewerFlush(pcbddc->dbg_viewer);CHKERRQ(ierr);
    ierr = MatDestroy(&A_II);CHKERRQ(ierr);
    ierr = MatDestroy(&A_BB);CHKERRQ(ierr);
    ierr = MatDestroy(&A_IB);CHKERRQ(ierr);
    ierr = MatDestroy(&A_BI);CHKERRQ(ierr);
    ierr = MatDestroy(&TM1);CHKERRQ(ierr);
    ierr = MatDestroy(&TM2);CHKERRQ(ierr);
    ierr = MatDestroy(&TM3);CHKERRQ(ierr);
    ierr = MatDestroy(&TM4);CHKERRQ(ierr);
    ierr = MatDestroy(&coarse_phi_D);CHKERRQ(ierr);
    ierr = MatDestroy(&coarse_phi_B);CHKERRQ(ierr);
    if (pcbddc->coarse_psi_B) {
      ierr = MatDestroy(&coarse_psi_D);CHKERRQ(ierr);
      ierr = MatDestroy(&coarse_psi_B);CHKERRQ(ierr);
    }
    ierr = MatDestroy(&coarse_sub_mat);CHKERRQ(ierr);
    ierr = ISRestoreIndices(pcbddc->is_R_local,&idx_R_local);CHKERRQ(ierr);
    ierr = PetscFree(coarsefunctions_errors);CHKERRQ(ierr);
    ierr = PetscFree(constraints_errors);CHKERRQ(ierr);
  }
  /* free memory */
  if (n_vertices) {
    ierr = PetscFree(vertices);CHKERRQ(ierr);
    ierr = VecDestroy(&vec1_V);CHKERRQ(ierr);
    ierr = VecDestroy(&vec2_V);CHKERRQ(ierr);
    ierr = MatDestroy(&A_RV);CHKERRQ(ierr);
    ierr = MatDestroy(&A_VR);CHKERRQ(ierr);
    ierr = MatDestroy(&A_VV);CHKERRQ(ierr);
  }
  if (n_constraints) {
    ierr = VecDestroy(&vec1_C);CHKERRQ(ierr);
    ierr = VecDestroy(&vec2_C);CHKERRQ(ierr);
    ierr = MatDestroy(&M1);CHKERRQ(ierr);
    ierr = MatDestroy(&C_CR);CHKERRQ(ierr);
  }
  ierr = PetscFree(auxindices);CHKERRQ(ierr);
  /* create coarse matrix and data structures for message passing associated actual choice of coarse problem type */
  ierr = PCBDDCSetUpCoarseEnvironment(pc,coarse_submat_vals);CHKERRQ(ierr);
  ierr = PetscFree(coarse_submat_vals);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCBDDCSetUpLocalMatrices"
PetscErrorCode PCBDDCSetUpLocalMatrices(PC pc)
{
  PC_IS*            pcis = (PC_IS*)(pc->data);
  PC_BDDC*          pcbddc = (PC_BDDC*)pc->data;
  Mat_IS*           matis = (Mat_IS*)pc->pmat->data;
  /* manage repeated solves */
  MatReuse          reuse;
  MatStructure      matstruct;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  /* get mat flags */
  ierr = PCGetOperators(pc,NULL,NULL,&matstruct);CHKERRQ(ierr);
  reuse = MAT_INITIAL_MATRIX;
  if (pc->setupcalled) {
    /* when matstruct is SAME_PRECONDITIONER, we shouldn't be here */
    if (matstruct == SAME_PRECONDITIONER) SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_PLIB,"This should not happen");
    if (matstruct == SAME_NONZERO_PATTERN) {
      reuse = MAT_REUSE_MATRIX;
    } else {
      reuse = MAT_INITIAL_MATRIX;
    }
  }
  if (reuse == MAT_INITIAL_MATRIX) {
    ierr = MatDestroy(&pcis->A_II);CHKERRQ(ierr);
    ierr = MatDestroy(&pcis->A_IB);CHKERRQ(ierr);
    ierr = MatDestroy(&pcis->A_BI);CHKERRQ(ierr);
    ierr = MatDestroy(&pcis->A_BB);CHKERRQ(ierr);
    ierr = MatDestroy(&pcbddc->local_mat);CHKERRQ(ierr);
  }

  /* transform local matrices if needed */
  if (pcbddc->use_change_of_basis) {
    Mat         change_mat_all;
    PetscScalar *row_cmat_values;
    PetscInt    *row_cmat_indices;
    PetscInt    *nnz,*is_indices,*temp_indices;
    PetscInt    i,j,k,n_D,n_B;
    
    /* Get Non-overlapping dimensions */
    n_B = pcis->n_B;
    n_D = pcis->n-n_B;

    /* compute nonzero structure of change of basis on all local nodes */
    ierr = PetscMalloc(pcis->n*sizeof(PetscInt),&nnz);CHKERRQ(ierr);
    ierr = ISGetIndices(pcis->is_I_local,(const PetscInt**)&is_indices);CHKERRQ(ierr);
    for (i=0;i<n_D;i++) nnz[is_indices[i]] = 1;
    ierr = ISRestoreIndices(pcis->is_I_local,(const PetscInt**)&is_indices);CHKERRQ(ierr);
    ierr = ISGetIndices(pcis->is_B_local,(const PetscInt**)&is_indices);CHKERRQ(ierr);
    k=1;
    for (i=0;i<n_B;i++) {
      ierr = MatGetRow(pcbddc->ChangeOfBasisMatrix,i,&j,NULL,NULL);CHKERRQ(ierr);
      nnz[is_indices[i]]=j;
      if (k < j) k = j;
      ierr = MatRestoreRow(pcbddc->ChangeOfBasisMatrix,i,&j,NULL,NULL);CHKERRQ(ierr);
    }
    ierr = ISRestoreIndices(pcis->is_B_local,(const PetscInt**)&is_indices);CHKERRQ(ierr);
    /* assemble change of basis matrix on the whole set of local dofs */
    ierr = PetscMalloc(k*sizeof(PetscInt),&temp_indices);CHKERRQ(ierr);
    ierr = MatCreate(PETSC_COMM_SELF,&change_mat_all);CHKERRQ(ierr);
    ierr = MatSetSizes(change_mat_all,pcis->n,pcis->n,pcis->n,pcis->n);CHKERRQ(ierr);
    ierr = MatSetType(change_mat_all,MATSEQAIJ);CHKERRQ(ierr);
    ierr = MatSeqAIJSetPreallocation(change_mat_all,0,nnz);CHKERRQ(ierr);
    ierr = ISGetIndices(pcis->is_I_local,(const PetscInt**)&is_indices);CHKERRQ(ierr);
    for (i=0;i<n_D;i++) {
      ierr = MatSetValue(change_mat_all,is_indices[i],is_indices[i],1.0,INSERT_VALUES);CHKERRQ(ierr);
    }
    ierr = ISRestoreIndices(pcis->is_I_local,(const PetscInt**)&is_indices);CHKERRQ(ierr);
    ierr = ISGetIndices(pcis->is_B_local,(const PetscInt**)&is_indices);CHKERRQ(ierr);
    for (i=0;i<n_B;i++) {
      ierr = MatGetRow(pcbddc->ChangeOfBasisMatrix,i,&j,(const PetscInt**)&row_cmat_indices,(const PetscScalar**)&row_cmat_values);CHKERRQ(ierr);
      for (k=0; k<j; k++) temp_indices[k]=is_indices[row_cmat_indices[k]];
      ierr = MatSetValues(change_mat_all,1,&is_indices[i],j,temp_indices,row_cmat_values,INSERT_VALUES);CHKERRQ(ierr);
      ierr = MatRestoreRow(pcbddc->ChangeOfBasisMatrix,i,&j,(const PetscInt**)&row_cmat_indices,(const PetscScalar**)&row_cmat_values);CHKERRQ(ierr);
    }
    ierr = MatAssemblyBegin(change_mat_all,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(change_mat_all,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    /* TODO: HOW TO WORK WITH BAIJ? PtAP not provided */
    ierr = MatGetBlockSize(matis->A,&i);CHKERRQ(ierr);
    if (i==1) {
      ierr = MatPtAP(matis->A,change_mat_all,reuse,2.0,&pcbddc->local_mat);CHKERRQ(ierr);
    } else {
      Mat work_mat;
      ierr = MatConvert(matis->A,MATSEQAIJ,MAT_INITIAL_MATRIX,&work_mat);CHKERRQ(ierr);
      ierr = MatPtAP(work_mat,change_mat_all,reuse,2.0,&pcbddc->local_mat);CHKERRQ(ierr);
      ierr = MatDestroy(&work_mat);CHKERRQ(ierr);
    }
    ierr = MatDestroy(&change_mat_all);CHKERRQ(ierr);
    ierr = PetscFree(nnz);CHKERRQ(ierr);
    ierr = PetscFree(temp_indices);CHKERRQ(ierr);
  } else {
    /* without change of basis, the local matrix is unchanged */
    if (!pcbddc->local_mat) {
      ierr = PetscObjectReference((PetscObject)matis->A);CHKERRQ(ierr);
      pcbddc->local_mat = matis->A;
    }
  }

  /* get submatrices */
  ierr = MatGetSubMatrix(pcbddc->local_mat,pcis->is_I_local,pcis->is_I_local,reuse,&pcis->A_II);CHKERRQ(ierr);
  ierr = MatGetSubMatrix(pcbddc->local_mat,pcis->is_I_local,pcis->is_B_local,reuse,&pcis->A_IB);CHKERRQ(ierr);
  ierr = MatGetSubMatrix(pcbddc->local_mat,pcis->is_B_local,pcis->is_I_local,reuse,&pcis->A_BI);CHKERRQ(ierr);
  ierr = MatGetSubMatrix(pcbddc->local_mat,pcis->is_B_local,pcis->is_B_local,reuse,&pcis->A_BB);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCBDDCSetUpLocalScatters"
PetscErrorCode PCBDDCSetUpLocalScatters(PC pc)
{
  PC_IS*         pcis = (PC_IS*)(pc->data);
  PC_BDDC*       pcbddc = (PC_BDDC*)pc->data;
  IS             is_aux1,is_aux2;
  PetscInt       *vertices,*aux_array1,*aux_array2,*is_indices,*idx_R_local;
  PetscInt       n_vertices,n_constraints,i,j,n_R,n_D,n_B;
  PetscBool      *array_bool;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* Set Non-overlapping dimensions */
  n_B = pcis->n_B; n_D = pcis->n - n_B;
  /* get vertex indices from constraint matrix */
  ierr = PCBDDCGetPrimalVerticesLocalIdx(pc,&n_vertices,&vertices);CHKERRQ(ierr);
  /* Set number of constraints */
  n_constraints = pcbddc->local_primal_size-n_vertices;
  /* Dohrmann's notation: dofs splitted in R (Remaining: all dofs but the vertices) and V (Vertices) */
  ierr = PetscMalloc(pcis->n*sizeof(PetscInt),&array_bool);CHKERRQ(ierr);
  for (i=0;i<pcis->n;i++) array_bool[i] = PETSC_TRUE;
  for (i=0;i<n_vertices;i++) array_bool[vertices[i]] = PETSC_FALSE;
  ierr = PetscMalloc((pcis->n-n_vertices)*sizeof(PetscInt),&idx_R_local);CHKERRQ(ierr);
  for (i=0, n_R=0; i<pcis->n; i++) {
    if (array_bool[i]) {
      idx_R_local[n_R] = i;
      n_R++;
    }
  }
  ierr = PetscFree(vertices);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_SELF,n_R,idx_R_local,PETSC_OWN_POINTER,&pcbddc->is_R_local);CHKERRQ(ierr);

  /* print some info if requested */
  if (pcbddc->dbg_flag) {
    ierr = PetscViewerASCIIPrintf(pcbddc->dbg_viewer,"--------------------------------------------------\n");CHKERRQ(ierr);
    ierr = PetscViewerFlush(pcbddc->dbg_viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"Subdomain %04d local dimensions\n",PetscGlobalRank);CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"local_size = %d, dirichlet_size = %d, boundary_size = %d\n",pcis->n,n_D,n_B);CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"r_size = %d, v_size = %d, constraints = %d, local_primal_size = %d\n",n_R,n_vertices,n_constraints,pcbddc->local_primal_size);CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"pcbddc->n_vertices = %d, pcbddc->n_constraints = %d\n",pcbddc->n_vertices,pcbddc->n_constraints);CHKERRQ(ierr);
    ierr = PetscViewerFlush(pcbddc->dbg_viewer);CHKERRQ(ierr);
  }

  /* VecScatters pcbddc->R_to_B and (optionally) pcbddc->R_to_D */
  ierr = PetscMalloc((pcis->n_B-n_vertices)*sizeof(PetscInt),&aux_array1);CHKERRQ(ierr);
  ierr = PetscMalloc((pcis->n_B-n_vertices)*sizeof(PetscInt),&aux_array2);CHKERRQ(ierr);
  ierr = ISGetIndices(pcis->is_I_local,(const PetscInt**)&is_indices);CHKERRQ(ierr);
  for (i=0; i<n_D; i++) array_bool[is_indices[i]] = PETSC_FALSE;
  ierr = ISRestoreIndices(pcis->is_I_local,(const PetscInt**)&is_indices);CHKERRQ(ierr);
  for (i=0, j=0; i<n_R; i++) {
    if (array_bool[idx_R_local[i]]) {
      aux_array1[j] = i;
      j++;
    }
  }
  ierr = ISCreateGeneral(PETSC_COMM_SELF,j,aux_array1,PETSC_OWN_POINTER,&is_aux1);CHKERRQ(ierr);
  ierr = ISGetIndices(pcis->is_B_local,(const PetscInt**)&is_indices);CHKERRQ(ierr);
  for (i=0, j=0; i<n_B; i++) {
    if (array_bool[is_indices[i]]) {
      aux_array2[j] = i; j++;
    }
  }
  ierr = ISRestoreIndices(pcis->is_B_local,(const PetscInt**)&is_indices);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_SELF,j,aux_array2,PETSC_OWN_POINTER,&is_aux2);CHKERRQ(ierr);
  ierr = VecScatterCreate(pcbddc->vec1_R,is_aux1,pcis->vec1_B,is_aux2,&pcbddc->R_to_B);CHKERRQ(ierr);
  ierr = ISDestroy(&is_aux1);CHKERRQ(ierr);
  ierr = ISDestroy(&is_aux2);CHKERRQ(ierr);

  if (pcbddc->inexact_prec_type || pcbddc->dbg_flag ) {
    ierr = PetscMalloc(n_D*sizeof(PetscInt),&aux_array1);CHKERRQ(ierr);
    for (i=0, j=0; i<n_R; i++) {
      if (!array_bool[idx_R_local[i]]) {
        aux_array1[j] = i;
        j++;
      }
    }
    ierr = ISCreateGeneral(PETSC_COMM_SELF,j,aux_array1,PETSC_OWN_POINTER,&is_aux1);CHKERRQ(ierr);
    ierr = VecScatterCreate(pcbddc->vec1_R,is_aux1,pcis->vec1_D,(IS)0,&pcbddc->R_to_D);CHKERRQ(ierr);
    ierr = ISDestroy(&is_aux1);CHKERRQ(ierr);
  }
  ierr = PetscFree(array_bool);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCBDDCSetUseExactDirichlet"
PetscErrorCode PCBDDCSetUseExactDirichlet(PC pc,PetscBool use)
{
  PC_BDDC  *pcbddc = (PC_BDDC*)pc->data;

  PetscFunctionBegin;
  pcbddc->use_exact_dirichlet=use;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCBDDCSetUpLocalSolvers"
PetscErrorCode PCBDDCSetUpLocalSolvers(PC pc)
{
  PC_BDDC        *pcbddc = (PC_BDDC*)pc->data;
  PC_IS          *pcis = (PC_IS*)pc->data;
  PC             pc_temp;
  Mat            A_RR;
  Vec            vec1,vec2,vec3;
  MatStructure   matstruct;
  PetscScalar    m_one = -1.0;
  PetscReal      value;
  PetscInt       n_D,n_R,use_exact,use_exact_reduced;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* Creating PC contexts for local Dirichlet and Neumann problems */
  ierr = PCGetOperators(pc,NULL,NULL,&matstruct);CHKERRQ(ierr);

  /* DIRICHLET PROBLEM */
  /* Matrix for Dirichlet problem is pcis->A_II */
  ierr = ISGetSize(pcis->is_I_local,&n_D);CHKERRQ(ierr);
  if (!pcbddc->ksp_D) { /* create object if not yet build */
    ierr = KSPCreate(PETSC_COMM_SELF,&pcbddc->ksp_D);CHKERRQ(ierr);
    ierr = PetscObjectIncrementTabLevel((PetscObject)pcbddc->ksp_D,(PetscObject)pc,1);CHKERRQ(ierr);
    /* default */
    ierr = KSPSetType(pcbddc->ksp_D,KSPPREONLY);CHKERRQ(ierr);
    ierr = KSPSetOptionsPrefix(pcbddc->ksp_D,"dirichlet_");CHKERRQ(ierr);
    ierr = KSPGetPC(pcbddc->ksp_D,&pc_temp);CHKERRQ(ierr);
    ierr = PCSetType(pc_temp,PCLU);CHKERRQ(ierr);
    ierr = PCFactorSetReuseFill(pc_temp,PETSC_TRUE);CHKERRQ(ierr);
  }
  ierr = KSPSetOperators(pcbddc->ksp_D,pcis->A_II,pcis->A_II,matstruct);CHKERRQ(ierr);
  /* Allow user's customization */
  ierr = KSPSetFromOptions(pcbddc->ksp_D);CHKERRQ(ierr);
  /* umfpack interface has a bug when matrix dimension is zero. TODO solve from umfpack interface */
  if (!n_D) {
    ierr = KSPGetPC(pcbddc->ksp_D,&pc_temp);CHKERRQ(ierr);
    ierr = PCSetType(pc_temp,PCNONE);CHKERRQ(ierr);
  }
  /* Set Up KSP for Dirichlet problem of BDDC */
  ierr = KSPSetUp(pcbddc->ksp_D);CHKERRQ(ierr);
  /* set ksp_D into pcis data */
  ierr = KSPDestroy(&pcis->ksp_D);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)pcbddc->ksp_D);CHKERRQ(ierr);
  pcis->ksp_D = pcbddc->ksp_D;

  /* NEUMANN PROBLEM */
  /* Matrix for Neumann problem is A_RR -> we need to create it */
  ierr = ISGetSize(pcbddc->is_R_local,&n_R);CHKERRQ(ierr);
  ierr = MatGetSubMatrix(pcbddc->local_mat,pcbddc->is_R_local,pcbddc->is_R_local,MAT_INITIAL_MATRIX,&A_RR);CHKERRQ(ierr);
  if (!pcbddc->ksp_R) { /* create object if not yet build */
    ierr = KSPCreate(PETSC_COMM_SELF,&pcbddc->ksp_R);CHKERRQ(ierr);
    ierr = PetscObjectIncrementTabLevel((PetscObject)pcbddc->ksp_R,(PetscObject)pc,1);CHKERRQ(ierr);
    /* default */
    ierr = KSPSetType(pcbddc->ksp_R,KSPPREONLY);CHKERRQ(ierr);
    ierr = KSPSetOptionsPrefix(pcbddc->ksp_R,"neumann_");CHKERRQ(ierr);
    ierr = KSPGetPC(pcbddc->ksp_R,&pc_temp);CHKERRQ(ierr);
    ierr = PCSetType(pc_temp,PCLU);CHKERRQ(ierr);
    ierr = PCFactorSetReuseFill(pc_temp,PETSC_TRUE);CHKERRQ(ierr);
  }
  ierr = KSPSetOperators(pcbddc->ksp_R,A_RR,A_RR,matstruct);CHKERRQ(ierr);
  /* Allow user's customization */
  ierr = KSPSetFromOptions(pcbddc->ksp_R);CHKERRQ(ierr);
  /* umfpack interface has a bug when matrix dimension is zero. TODO solve from umfpack interface */
  if (!n_R) {
    ierr = KSPGetPC(pcbddc->ksp_R,&pc_temp);CHKERRQ(ierr);
    ierr = PCSetType(pc_temp,PCNONE);CHKERRQ(ierr);
  }
  /* Set Up KSP for Neumann problem of BDDC */
  ierr = KSPSetUp(pcbddc->ksp_R);CHKERRQ(ierr);

  /* check Dirichlet and Neumann solvers and adapt them if a nullspace correction is needed */
  
  /* Dirichlet */
  ierr = MatGetVecs(pcis->A_II,&vec1,&vec2);CHKERRQ(ierr);
  ierr = VecDuplicate(vec1,&vec3);CHKERRQ(ierr);
  ierr = VecSetRandom(vec1,NULL);CHKERRQ(ierr);
  ierr = MatMult(pcis->A_II,vec1,vec2);CHKERRQ(ierr);
  ierr = KSPSolve(pcbddc->ksp_D,vec2,vec3);CHKERRQ(ierr);
  ierr = VecAXPY(vec3,m_one,vec1);CHKERRQ(ierr);
  ierr = VecNorm(vec3,NORM_INFINITY,&value);CHKERRQ(ierr);
  ierr = VecDestroy(&vec1);CHKERRQ(ierr);
  ierr = VecDestroy(&vec2);CHKERRQ(ierr);
  ierr = VecDestroy(&vec3);CHKERRQ(ierr);
  /* need to be adapted? */
  use_exact = (PetscAbsReal(value) > 1.e-4 ? 0 : 1);
  ierr = MPI_Allreduce(&use_exact,&use_exact_reduced,1,MPIU_INT,MPI_LAND,PetscObjectComm((PetscObject)pc));CHKERRQ(ierr);
  ierr = PCBDDCSetUseExactDirichlet(pc,(PetscBool)use_exact_reduced);CHKERRQ(ierr);
  /* print info */
  if (pcbddc->dbg_flag) {
    ierr = PetscViewerFlush(pcbddc->dbg_viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(pcbddc->dbg_viewer,"--------------------------------------------------\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(pcbddc->dbg_viewer,"Checking solution of Dirichlet and Neumann problems\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"Subdomain %04d infinity error for Dirichlet solve = % 1.14e \n",PetscGlobalRank,value);CHKERRQ(ierr);
    ierr = PetscViewerFlush(pcbddc->dbg_viewer);CHKERRQ(ierr);
  }
  if (n_D && pcbddc->NullSpace && !use_exact_reduced && !pcbddc->inexact_prec_type) {
    ierr = PCBDDCNullSpaceAssembleCorrection(pc,pcis->is_I_local);CHKERRQ(ierr);
  }

  /* Neumann */
  ierr = MatGetVecs(A_RR,&vec1,&vec2);CHKERRQ(ierr);
  ierr = VecDuplicate(vec1,&vec3);CHKERRQ(ierr);
  ierr = VecSetRandom(vec1,NULL);CHKERRQ(ierr);
  ierr = MatMult(A_RR,vec1,vec2);CHKERRQ(ierr);
  ierr = KSPSolve(pcbddc->ksp_R,vec2,vec3);CHKERRQ(ierr);
  ierr = VecAXPY(vec3,m_one,vec1);CHKERRQ(ierr);
  ierr = VecNorm(vec3,NORM_INFINITY,&value);CHKERRQ(ierr);
  ierr = VecDestroy(&vec1);CHKERRQ(ierr);
  ierr = VecDestroy(&vec2);CHKERRQ(ierr);
  ierr = VecDestroy(&vec3);CHKERRQ(ierr);
  /* need to be adapted? */
  use_exact = (PetscAbsReal(value) > 1.e-4 ? 0 : 1);
  if (PetscAbsReal(value) > 1.e-4) use_exact = 0;
  ierr = MPI_Allreduce(&use_exact,&use_exact_reduced,1,MPIU_INT,MPI_LAND,PetscObjectComm((PetscObject)pc));CHKERRQ(ierr);
  /* print info */
  if (pcbddc->dbg_flag) {
    ierr = PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"Subdomain %04d infinity error for  Neumann  solve = % 1.14e \n",PetscGlobalRank,value);CHKERRQ(ierr);
    ierr = PetscViewerFlush(pcbddc->dbg_viewer);CHKERRQ(ierr);
  }
  if (n_R && pcbddc->NullSpace && !use_exact_reduced) { /* is it the right logic? */
    ierr = PCBDDCNullSpaceAssembleCorrection(pc,pcbddc->is_R_local);CHKERRQ(ierr);
  }

  /* free Neumann problem's matrix */
  ierr = MatDestroy(&A_RR);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCBDDCSolveSaddlePoint"
static PetscErrorCode  PCBDDCSolveSaddlePoint(PC pc)
{
  PetscErrorCode ierr;
  PC_BDDC*       pcbddc = (PC_BDDC*)(pc->data);

  PetscFunctionBegin;
  ierr = KSPSolve(pcbddc->ksp_R,pcbddc->vec1_R,pcbddc->vec2_R);CHKERRQ(ierr);
  if (pcbddc->local_auxmat1) {
    ierr = MatMult(pcbddc->local_auxmat1,pcbddc->vec2_R,pcbddc->vec1_C);CHKERRQ(ierr);
    ierr = MatMultAdd(pcbddc->local_auxmat2,pcbddc->vec1_C,pcbddc->vec2_R,pcbddc->vec2_R);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCBDDCApplyInterfacePreconditioner"
PetscErrorCode  PCBDDCApplyInterfacePreconditioner(PC pc)
{
  PetscErrorCode ierr;
  PC_BDDC*        pcbddc = (PC_BDDC*)(pc->data);
  PC_IS*            pcis = (PC_IS*)  (pc->data);
  const PetscScalar zero = 0.0;

  PetscFunctionBegin;
  /* Application of PHI^T (or PSI^T)  */
  if (pcbddc->coarse_psi_B) {
    ierr = MatMultTranspose(pcbddc->coarse_psi_B,pcis->vec1_B,pcbddc->vec1_P);CHKERRQ(ierr);
    if (pcbddc->inexact_prec_type) { ierr = MatMultTransposeAdd(pcbddc->coarse_psi_D,pcis->vec1_D,pcbddc->vec1_P,pcbddc->vec1_P);CHKERRQ(ierr); }
  } else {
    ierr = MatMultTranspose(pcbddc->coarse_phi_B,pcis->vec1_B,pcbddc->vec1_P);CHKERRQ(ierr);
    if (pcbddc->inexact_prec_type) { ierr = MatMultTransposeAdd(pcbddc->coarse_phi_D,pcis->vec1_D,pcbddc->vec1_P,pcbddc->vec1_P);CHKERRQ(ierr); }
  }
  /* Scatter data of coarse_rhs */
  if (pcbddc->coarse_rhs) { ierr = VecSet(pcbddc->coarse_rhs,zero);CHKERRQ(ierr); }
  ierr = PCBDDCScatterCoarseDataBegin(pc,pcbddc->vec1_P,pcbddc->coarse_rhs,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);

  /* Local solution on R nodes */
  ierr = VecSet(pcbddc->vec1_R,zero);CHKERRQ(ierr);
  ierr = VecScatterBegin(pcbddc->R_to_B,pcis->vec1_B,pcbddc->vec1_R,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd  (pcbddc->R_to_B,pcis->vec1_B,pcbddc->vec1_R,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  if (pcbddc->inexact_prec_type) {
    ierr = VecScatterBegin(pcbddc->R_to_D,pcis->vec1_D,pcbddc->vec1_R,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterEnd  (pcbddc->R_to_D,pcis->vec1_D,pcbddc->vec1_R,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  }
  ierr = PCBDDCSolveSaddlePoint(pc);CHKERRQ(ierr);
  ierr = VecSet(pcis->vec1_B,zero);CHKERRQ(ierr);
  ierr = VecScatterBegin(pcbddc->R_to_B,pcbddc->vec2_R,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd  (pcbddc->R_to_B,pcbddc->vec2_R,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  if (pcbddc->inexact_prec_type) {
    ierr = VecScatterBegin(pcbddc->R_to_D,pcbddc->vec2_R,pcis->vec1_D,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd  (pcbddc->R_to_D,pcbddc->vec2_R,pcis->vec1_D,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  }

  /* Coarse solution */
  ierr = PCBDDCScatterCoarseDataEnd(pc,pcbddc->vec1_P,pcbddc->coarse_rhs,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  if (pcbddc->coarse_rhs) { /* TODO remove null space when doing multilevel */
    ierr = KSPSolve(pcbddc->coarse_ksp,pcbddc->coarse_rhs,pcbddc->coarse_vec);CHKERRQ(ierr);
  }
  ierr = PCBDDCScatterCoarseDataBegin(pc,pcbddc->coarse_vec,pcbddc->vec1_P,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = PCBDDCScatterCoarseDataEnd  (pc,pcbddc->coarse_vec,pcbddc->vec1_P,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);

  /* Sum contributions from two levels */
  ierr = MatMultAdd(pcbddc->coarse_phi_B,pcbddc->vec1_P,pcis->vec1_B,pcis->vec1_B);CHKERRQ(ierr);
  if (pcbddc->inexact_prec_type) { ierr = MatMultAdd(pcbddc->coarse_phi_D,pcbddc->vec1_P,pcis->vec1_D,pcis->vec1_D);CHKERRQ(ierr); }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCBDDCScatterCoarseDataBegin"
PetscErrorCode PCBDDCScatterCoarseDataBegin(PC pc,Vec vec_from, Vec vec_to, InsertMode imode, ScatterMode smode)
{
  PetscErrorCode ierr;
  PC_BDDC*       pcbddc = (PC_BDDC*)(pc->data);

  PetscFunctionBegin;
  switch (pcbddc->coarse_communications_type) {
    case SCATTERS_BDDC:
      ierr = VecScatterBegin(pcbddc->coarse_loc_to_glob,vec_from,vec_to,imode,smode);CHKERRQ(ierr);
      break;
    case GATHERS_BDDC:
      break;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCBDDCScatterCoarseDataEnd"
PetscErrorCode PCBDDCScatterCoarseDataEnd(PC pc,Vec vec_from, Vec vec_to, InsertMode imode, ScatterMode smode)
{
  PetscErrorCode ierr;
  PC_BDDC*       pcbddc = (PC_BDDC*)(pc->data);
  PetscScalar*   array_to;
  PetscScalar*   array_from;
  MPI_Comm       comm;
  PetscInt       i;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)pc,&comm);CHKERRQ(ierr);
  switch (pcbddc->coarse_communications_type) {
    case SCATTERS_BDDC:
      ierr = VecScatterEnd(pcbddc->coarse_loc_to_glob,vec_from,vec_to,imode,smode);CHKERRQ(ierr);
      break;
    case GATHERS_BDDC:
      if (vec_from) {
        ierr = VecGetArray(vec_from,&array_from);CHKERRQ(ierr);
      }
      if (vec_to) {
        ierr = VecGetArray(vec_to,&array_to);CHKERRQ(ierr);
      }
      switch(pcbddc->coarse_problem_type){
        case SEQUENTIAL_BDDC:
          if (smode == SCATTER_FORWARD) {
            ierr = MPI_Gatherv(&array_from[0],pcbddc->local_primal_size,MPIU_SCALAR,&pcbddc->replicated_local_primal_values[0],pcbddc->local_primal_sizes,pcbddc->local_primal_displacements,MPIU_SCALAR,0,comm);CHKERRQ(ierr);
            if (vec_to) {
              if (imode == ADD_VALUES) {
                for (i=0;i<pcbddc->replicated_primal_size;i++) {
                  array_to[pcbddc->replicated_local_primal_indices[i]]+=pcbddc->replicated_local_primal_values[i];
                }
              } else {
                for (i=0;i<pcbddc->replicated_primal_size;i++) {
                  array_to[pcbddc->replicated_local_primal_indices[i]]=pcbddc->replicated_local_primal_values[i];
                }
              }
            }
          } else {
            if (vec_from) {
              if (imode == ADD_VALUES) {
                MPI_Comm vec_from_comm;
                ierr = PetscObjectGetComm((PetscObject)(vec_from),&vec_from_comm);CHKERRQ(ierr);
                SETERRQ2(vec_from_comm,PETSC_ERR_SUP,"Unsupported insert mode ADD_VALUES for SCATTER_REVERSE in %s for case %d\n",__FUNCT__,pcbddc->coarse_problem_type);
              }
              for (i=0;i<pcbddc->replicated_primal_size;i++) {
                pcbddc->replicated_local_primal_values[i]=array_from[pcbddc->replicated_local_primal_indices[i]];
              }
            }
            ierr = MPI_Scatterv(&pcbddc->replicated_local_primal_values[0],pcbddc->local_primal_sizes,pcbddc->local_primal_displacements,MPIU_SCALAR,&array_to[0],pcbddc->local_primal_size,MPIU_SCALAR,0,comm);CHKERRQ(ierr);
          }
          break;
        case REPLICATED_BDDC:
          if (smode == SCATTER_FORWARD) {
            ierr = MPI_Allgatherv(&array_from[0],pcbddc->local_primal_size,MPIU_SCALAR,&pcbddc->replicated_local_primal_values[0],pcbddc->local_primal_sizes,pcbddc->local_primal_displacements,MPIU_SCALAR,comm);CHKERRQ(ierr);
            if (imode == ADD_VALUES) {
              for (i=0;i<pcbddc->replicated_primal_size;i++) {
                array_to[pcbddc->replicated_local_primal_indices[i]]+=pcbddc->replicated_local_primal_values[i];
              }
            } else {
              for (i=0;i<pcbddc->replicated_primal_size;i++) {
                array_to[pcbddc->replicated_local_primal_indices[i]]=pcbddc->replicated_local_primal_values[i];
              }
            }
          } else { /* no communications needed for SCATTER_REVERSE since needed data is already present */
            if (imode == ADD_VALUES) {
              for (i=0;i<pcbddc->local_primal_size;i++) {
                array_to[i]+=array_from[pcbddc->local_primal_indices[i]];
              }
            } else {
              for (i=0;i<pcbddc->local_primal_size;i++) {
                array_to[i]=array_from[pcbddc->local_primal_indices[i]];
              }
            }
          }
          break;
        case MULTILEVEL_BDDC:
          break;
        case PARALLEL_BDDC:
          break;
      }
      if (vec_from) {
        ierr = VecRestoreArray(vec_from,&array_from);CHKERRQ(ierr);
      }
      if (vec_to) {
        ierr = VecRestoreArray(vec_to,&array_to);CHKERRQ(ierr);
      }
      break;
  }
  PetscFunctionReturn(0);
}

/* uncomment for testing purposes */
/* #define PETSC_MISSING_LAPACK_GESVD 1 */
#undef __FUNCT__
#define __FUNCT__ "PCBDDCConstraintsSetUp"
PetscErrorCode PCBDDCConstraintsSetUp(PC pc)
{
  PetscErrorCode    ierr;
  PC_IS*            pcis = (PC_IS*)(pc->data);
  PC_BDDC*          pcbddc = (PC_BDDC*)pc->data;
  Mat_IS*           matis = (Mat_IS*)pc->pmat->data;
  /* constraint and (optionally) change of basis matrix implemented as SeqAIJ */ 
  MatType           impMatType=MATSEQAIJ;
  /* one and zero */
  PetscScalar       one=1.0,zero=0.0;
  /* space to store constraints and their local indices */
  PetscScalar       *temp_quadrature_constraint;
  PetscInt          *temp_indices,*temp_indices_to_constraint,*temp_indices_to_constraint_B;
  /* iterators */
  PetscInt          i,j,k,total_counts,temp_start_ptr;
  /* stuff to store connected components stored in pcbddc->mat_graph */
  IS                ISForVertices,*ISForFaces,*ISForEdges,*used_IS;
  PetscInt          n_ISForFaces,n_ISForEdges;
  PetscBool         get_faces,get_edges,get_vertices;
  /* near null space stuff */
  MatNullSpace      nearnullsp;
  const Vec         *nearnullvecs;
  Vec               *localnearnullsp;
  PetscBool         nnsp_has_cnst;
  PetscInt          nnsp_size;
  PetscScalar       *array;
  /* BLAS integers */
  PetscBLASInt      lwork,lierr;
  PetscBLASInt      Blas_N,Blas_M,Blas_K,Blas_one=1;
  PetscBLASInt      Blas_LDA,Blas_LDB,Blas_LDC;
  /* LAPACK working arrays for SVD or POD */
  PetscBool         skip_lapack;
  PetscScalar       *work;
  PetscReal         *singular_vals;
#if defined(PETSC_USE_COMPLEX)
  PetscReal         *rwork;
#endif
#if defined(PETSC_MISSING_LAPACK_GESVD)
  PetscBLASInt      Blas_one_2=1;
  PetscScalar       *temp_basis,*correlation_mat;
#else
  PetscBLASInt      dummy_int_1=1,dummy_int_2=1;
  PetscScalar       dummy_scalar_1=0.0,dummy_scalar_2=0.0;
#endif
  /* change of basis */
  PetscInt          *aux_primal_numbering,*aux_primal_minloc,*global_indices;
  PetscBool         boolforchange,*change_basis,*touched;
  /* auxiliary stuff */
  PetscInt          *nnz,*is_indices,*local_to_B;
  /* some quantities */
  PetscInt          n_vertices,total_primal_vertices;
  PetscInt          size_of_constraint,max_size_of_constraint,max_constraints,temp_constraints;


  PetscFunctionBegin;
  /* Get index sets for faces, edges and vertices from graph */
  get_faces = PETSC_TRUE;
  get_edges = PETSC_TRUE;
  get_vertices = PETSC_TRUE;
  if (pcbddc->vertices_flag) {
    get_faces = PETSC_FALSE;
    get_edges = PETSC_FALSE;
  }
  if (pcbddc->constraints_flag) {
    get_vertices = PETSC_FALSE;
  }
  if (pcbddc->faces_flag) {
    get_edges = PETSC_FALSE;
  }
  if (pcbddc->edges_flag) {
    get_faces = PETSC_FALSE;
  }
  /* default */
  if (!get_faces && !get_edges && !get_vertices) {
    get_vertices = PETSC_TRUE;
  }
  ierr = PCBDDCGraphGetCandidatesIS(pcbddc->mat_graph,get_faces,get_edges,get_vertices,&n_ISForFaces,&ISForFaces,&n_ISForEdges,&ISForEdges,&ISForVertices);
  /* print some info */
  if (pcbddc->dbg_flag) {
    ierr = PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"--------------------------------------------------------------\n");CHKERRQ(ierr);
    i = 0;
    if (ISForVertices) {
      ierr = ISGetSize(ISForVertices,&i);CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"Subdomain %04d got %02d local candidate vertices\n",PetscGlobalRank,i);CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"Subdomain %04d got %02d local candidate edges\n",PetscGlobalRank,n_ISForEdges);CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"Subdomain %04d got %02d local candidate faces\n",PetscGlobalRank,n_ISForFaces);CHKERRQ(ierr);
    ierr = PetscViewerFlush(pcbddc->dbg_viewer);CHKERRQ(ierr);
  }
  /* check if near null space is attached to global mat */
  ierr = MatGetNearNullSpace(pc->pmat,&nearnullsp);CHKERRQ(ierr);
  if (nearnullsp) {
    ierr = MatNullSpaceGetVecs(nearnullsp,&nnsp_has_cnst,&nnsp_size,&nearnullvecs);CHKERRQ(ierr);
  } else { /* if near null space is not provided BDDC uses constants by default */
    nnsp_size = 0;
    nnsp_has_cnst = PETSC_TRUE;
  }
  /* get max number of constraints on a single cc */
  max_constraints = nnsp_size; 
  if (nnsp_has_cnst) max_constraints++;

  /*
       Evaluate maximum storage size needed by the procedure
       - temp_indices will contain start index of each constraint stored as follows
       - temp_indices_to_constraint  [temp_indices[i],...,temp[indices[i+1]-1] will contain the indices (in local numbering) on which the constraint acts
       - temp_indices_to_constraint_B[temp_indices[i],...,temp[indices[i+1]-1] will contain the indices (in boundary numbering) on which the constraint acts
       - temp_quadrature_constraint  [temp_indices[i],...,temp[indices[i+1]-1] will contain the scalars representing the constraint itself
                                                                                                                                                         */
  total_counts = n_ISForFaces+n_ISForEdges;
  total_counts *= max_constraints;
  n_vertices = 0;
  if (ISForVertices) {
    ierr = ISGetSize(ISForVertices,&n_vertices);CHKERRQ(ierr);
  }
  total_counts += n_vertices;
  ierr = PetscMalloc((total_counts+1)*sizeof(PetscInt),&temp_indices);CHKERRQ(ierr);
  ierr = PetscMalloc((total_counts+1)*sizeof(PetscBool),&change_basis);CHKERRQ(ierr);
  total_counts = 0;
  max_size_of_constraint = 0;
  for (i=0;i<n_ISForEdges+n_ISForFaces;i++) {
    if (i<n_ISForEdges) {
      used_IS = &ISForEdges[i];
    } else {
      used_IS = &ISForFaces[i-n_ISForEdges];
    }
    ierr = ISGetSize(*used_IS,&j);CHKERRQ(ierr);
    total_counts += j;
    max_size_of_constraint = PetscMax(j,max_size_of_constraint);
  }
  total_counts *= max_constraints;
  total_counts += n_vertices;
  ierr = PetscMalloc(total_counts*sizeof(PetscScalar),&temp_quadrature_constraint);CHKERRQ(ierr);
  ierr = PetscMalloc(total_counts*sizeof(PetscInt),&temp_indices_to_constraint);CHKERRQ(ierr);
  ierr = PetscMalloc(total_counts*sizeof(PetscInt),&temp_indices_to_constraint_B);CHKERRQ(ierr);
  /* local to boundary numbering */
  ierr = PetscMalloc(pcis->n*sizeof(PetscInt),&local_to_B);CHKERRQ(ierr);
  ierr = ISGetIndices(pcis->is_B_local,(const PetscInt**)&is_indices);CHKERRQ(ierr);
  for (i=0;i<pcis->n;i++) local_to_B[i]=-1;
  for (i=0;i<pcis->n_B;i++) local_to_B[is_indices[i]]=i;
  ierr = ISRestoreIndices(pcis->is_B_local,(const PetscInt**)&is_indices);CHKERRQ(ierr);
  /* get local part of global near null space vectors */
  ierr = PetscMalloc(nnsp_size*sizeof(Vec),&localnearnullsp);CHKERRQ(ierr);
  for (k=0;k<nnsp_size;k++) {
    ierr = VecDuplicate(pcis->vec1_N,&localnearnullsp[k]);CHKERRQ(ierr);
    ierr = VecScatterBegin(matis->ctx,nearnullvecs[k],localnearnullsp[k],INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(matis->ctx,nearnullvecs[k],localnearnullsp[k],INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  }

  /* whether or not to skip lapack calls */
  skip_lapack = PETSC_TRUE;
  if (n_ISForFaces+n_ISForEdges) skip_lapack = PETSC_FALSE;

  /* First we issue queries to allocate optimal workspace for LAPACKgesvd (or LAPACKsyev if SVD is missing) */
  if (!pcbddc->use_nnsp_true && !skip_lapack) {
    PetscScalar temp_work;
#if defined(PETSC_MISSING_LAPACK_GESVD)
    /* Proper Orthogonal Decomposition (POD) using the snapshot method */
    ierr = PetscMalloc(max_constraints*max_constraints*sizeof(PetscScalar),&correlation_mat);CHKERRQ(ierr);
    ierr = PetscMalloc(max_constraints*sizeof(PetscReal),&singular_vals);CHKERRQ(ierr);
    ierr = PetscMalloc(max_size_of_constraint*max_constraints*sizeof(PetscScalar),&temp_basis);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
    ierr = PetscMalloc(3*max_constraints*sizeof(PetscReal),&rwork);CHKERRQ(ierr);
#endif
    /* now we evaluate the optimal workspace using query with lwork=-1 */
    ierr = PetscBLASIntCast(max_constraints,&Blas_N);CHKERRQ(ierr);
    ierr = PetscBLASIntCast(max_constraints,&Blas_LDA);CHKERRQ(ierr);
    lwork = -1;
    ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
    PetscStackCallBLAS("LAPACKsyev",LAPACKsyev_("V","U",&Blas_N,correlation_mat,&Blas_LDA,singular_vals,&temp_work,&lwork,&lierr));
#else
    PetscStackCallBLAS("LAPACKsyev",LAPACKsyev_("V","U",&Blas_N,correlation_mat,&Blas_LDA,singular_vals,&temp_work,&lwork,rwork,&lierr));
#endif
    ierr = PetscFPTrapPop();CHKERRQ(ierr);
    if (lierr) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in query to SYEV Lapack routine %d",(int)lierr);
#else /* on missing GESVD */
    /* SVD */
    PetscInt max_n,min_n;
    max_n = max_size_of_constraint;
    min_n = max_constraints;
    if (max_size_of_constraint < max_constraints) {
      min_n = max_size_of_constraint;
      max_n = max_constraints;
    }
    ierr = PetscMalloc(min_n*sizeof(PetscReal),&singular_vals);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
    ierr = PetscMalloc(5*min_n*sizeof(PetscReal),&rwork);CHKERRQ(ierr);
#endif
    /* now we evaluate the optimal workspace using query with lwork=-1 */
    lwork = -1;
    ierr = PetscBLASIntCast(max_n,&Blas_M);CHKERRQ(ierr);
    ierr = PetscBLASIntCast(min_n,&Blas_N);CHKERRQ(ierr);
    ierr = PetscBLASIntCast(max_n,&Blas_LDA);CHKERRQ(ierr);
    ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
    PetscStackCallBLAS("LAPACKgesvd",LAPACKgesvd_("O","N",&Blas_M,&Blas_N,&temp_quadrature_constraint[0],&Blas_LDA,singular_vals,&dummy_scalar_1,&dummy_int_1,&dummy_scalar_2,&dummy_int_2,&temp_work,&lwork,&lierr));
#else
    PetscStackCallBLAS("LAPACKgesvd",LAPACKgesvd_("O","N",&Blas_M,&Blas_N,&temp_quadrature_constraint[0],&Blas_LDA,singular_vals,&dummy_scalar_1,&dummy_int_1,&dummy_scalar_2,&dummy_int_2,&temp_work,&lwork,rwork,&lierr));
#endif
    ierr = PetscFPTrapPop();CHKERRQ(ierr);
    if (lierr) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in query to GESVD Lapack routine %d",(int)lierr);
#endif /* on missing GESVD */
    /* Allocate optimal workspace */
    ierr = PetscBLASIntCast((PetscInt)PetscRealPart(temp_work),&lwork);CHKERRQ(ierr);
    ierr = PetscMalloc((PetscInt)lwork*sizeof(PetscScalar),&work);CHKERRQ(ierr);
  }
  /* Now we can loop on constraining sets */
  total_counts = 0;
  temp_indices[0] = 0;
  /* vertices */
  if (ISForVertices) {
    ierr = ISGetIndices(ISForVertices,(const PetscInt**)&is_indices);CHKERRQ(ierr);
    if (nnsp_has_cnst) { /* consider all vertices */
      for (i=0;i<n_vertices;i++) {
        temp_indices_to_constraint[temp_indices[total_counts]]=is_indices[i];
        temp_indices_to_constraint_B[temp_indices[total_counts]]=local_to_B[is_indices[i]];
        temp_quadrature_constraint[temp_indices[total_counts]]=1.0;
        temp_indices[total_counts+1]=temp_indices[total_counts]+1;
        change_basis[total_counts]=PETSC_FALSE;
        total_counts++;
      }
    } else { /* consider vertices for which exist at least a localnearnullsp which is not null there */
      PetscBool used_vertex;
      for (i=0;i<n_vertices;i++) {
        used_vertex = PETSC_FALSE;
        k = 0;
        while (!used_vertex && k<nnsp_size) {
          ierr = VecGetArrayRead(localnearnullsp[k],(const PetscScalar**)&array);CHKERRQ(ierr);
          if (PetscAbsScalar(array[is_indices[i]])>0.0) {
            temp_indices_to_constraint[temp_indices[total_counts]]=is_indices[i];
            temp_indices_to_constraint_B[temp_indices[total_counts]]=local_to_B[is_indices[i]];
            temp_quadrature_constraint[temp_indices[total_counts]]=1.0;
            temp_indices[total_counts+1]=temp_indices[total_counts]+1;
            change_basis[total_counts]=PETSC_FALSE;
            total_counts++;
            used_vertex = PETSC_TRUE;
          }
          ierr = VecRestoreArrayRead(localnearnullsp[k],(const PetscScalar**)&array);CHKERRQ(ierr);
          k++;
        }
      }
    }
    ierr = ISRestoreIndices(ISForVertices,(const PetscInt**)&is_indices);CHKERRQ(ierr);
    n_vertices = total_counts;
  }

  /* edges and faces */
  for (i=0;i<n_ISForEdges+n_ISForFaces;i++) {
    if (i<n_ISForEdges) {
      used_IS = &ISForEdges[i];
      boolforchange = pcbddc->use_change_of_basis; /* change or not the basis on the edge */
    } else {
      used_IS = &ISForFaces[i-n_ISForEdges];
      boolforchange = (PetscBool)(pcbddc->use_change_of_basis && pcbddc->use_change_on_faces); /* change or not the basis on the face */
    }
    temp_constraints = 0;          /* zero the number of constraints I have on this conn comp */
    temp_start_ptr = total_counts; /* need to know the starting index of constraints stored */
    ierr = ISGetSize(*used_IS,&size_of_constraint);CHKERRQ(ierr);
    ierr = ISGetIndices(*used_IS,(const PetscInt**)&is_indices);CHKERRQ(ierr);
    /* change of basis should not be performed on local periodic nodes */
    if (pcbddc->mat_graph->mirrors && pcbddc->mat_graph->mirrors[is_indices[0]]) boolforchange = PETSC_FALSE;
    if (nnsp_has_cnst) {
      PetscScalar quad_value;
      temp_constraints++;
      quad_value = (PetscScalar)(1.0/PetscSqrtReal((PetscReal)size_of_constraint));
      for (j=0;j<size_of_constraint;j++) {
        temp_indices_to_constraint[temp_indices[total_counts]+j]=is_indices[j];
        temp_indices_to_constraint_B[temp_indices[total_counts]+j]=local_to_B[is_indices[j]];
        temp_quadrature_constraint[temp_indices[total_counts]+j]=quad_value;
      }
      temp_indices[total_counts+1]=temp_indices[total_counts]+size_of_constraint;  /* store new starting point */
      change_basis[total_counts]=boolforchange;
      total_counts++;
    }
    for (k=0;k<nnsp_size;k++) {
      PetscReal real_value;
      ierr = VecGetArrayRead(localnearnullsp[k],(const PetscScalar**)&array);CHKERRQ(ierr);
      for (j=0;j<size_of_constraint;j++) {
        temp_indices_to_constraint[temp_indices[total_counts]+j]=is_indices[j];
        temp_indices_to_constraint_B[temp_indices[total_counts]+j]=local_to_B[is_indices[j]];
        temp_quadrature_constraint[temp_indices[total_counts]+j]=array[is_indices[j]];
      }
      ierr = VecRestoreArrayRead(localnearnullsp[k],(const PetscScalar**)&array);CHKERRQ(ierr);
      /* check if array is null on the connected component */
      ierr = PetscBLASIntCast(size_of_constraint,&Blas_N);CHKERRQ(ierr);
      PetscStackCallBLAS("BLASasum",real_value = BLASasum_(&Blas_N,&temp_quadrature_constraint[temp_indices[total_counts]],&Blas_one));
      if (real_value > 0.0) { /* keep indices and values */
        temp_constraints++;
        temp_indices[total_counts+1]=temp_indices[total_counts]+size_of_constraint;  /* store new starting point */
        change_basis[total_counts]=boolforchange;
        total_counts++;
      }
    }
    ierr = ISRestoreIndices(*used_IS,(const PetscInt**)&is_indices);CHKERRQ(ierr);
    /* perform SVD on the constraints if use_nnsp_true has not be requested by the user */
    if (!pcbddc->use_nnsp_true) {
      PetscReal tol = 1.0e-8; /* tolerance for retaining eigenmodes */

#if defined(PETSC_MISSING_LAPACK_GESVD)
      /* SVD: Y = U*S*V^H                -> U (eigenvectors of Y*Y^H) = Y*V*(S)^\dag
         POD: Y^H*Y = V*D*V^H, D = S^H*S -> U = Y*V*D^(-1/2)
         -> When PETSC_USE_COMPLEX and PETSC_MISSING_LAPACK_GESVD are defined
            the constraints basis will differ (by a complex factor with absolute value equal to 1)
            from that computed using LAPACKgesvd
         -> This is due to a different computation of eigenvectors in LAPACKheev 
         -> The quality of the POD-computed basis will be the same */
      ierr = PetscMemzero(correlation_mat,temp_constraints*temp_constraints*sizeof(PetscScalar));CHKERRQ(ierr);
      /* Store upper triangular part of correlation matrix */
      ierr = PetscBLASIntCast(size_of_constraint,&Blas_N);CHKERRQ(ierr);
      ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
      for (j=0;j<temp_constraints;j++) {
        for (k=0;k<j+1;k++) {
          PetscStackCallBLAS("BLASdot",correlation_mat[j*temp_constraints+k]=BLASdot_(&Blas_N,&temp_quadrature_constraint[temp_indices[temp_start_ptr+k]],&Blas_one,&temp_quadrature_constraint[temp_indices[temp_start_ptr+j]],&Blas_one_2));
        }
      }
      /* compute eigenvalues and eigenvectors of correlation matrix */
      ierr = PetscBLASIntCast(temp_constraints,&Blas_N);CHKERRQ(ierr);
      ierr = PetscBLASIntCast(temp_constraints,&Blas_LDA);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
      PetscStackCallBLAS("LAPACKsyev",LAPACKsyev_("V","U",&Blas_N,correlation_mat,&Blas_LDA,singular_vals,work,&lwork,&lierr));
#else
      PetscStackCallBLAS("LAPACKsyev",LAPACKsyev_("V","U",&Blas_N,correlation_mat,&Blas_LDA,singular_vals,work,&lwork,rwork,&lierr));
#endif
      ierr = PetscFPTrapPop();CHKERRQ(ierr);
      if (lierr) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in SYEV Lapack routine %d",(int)lierr);
      /* retain eigenvalues greater than tol: note that LAPACKsyev gives eigs in ascending order */
      j=0;
      while (j < temp_constraints && singular_vals[j] < tol) j++;
      total_counts=total_counts-j;
      /* scale and copy POD basis into used quadrature memory */
      ierr = PetscBLASIntCast(size_of_constraint,&Blas_M);CHKERRQ(ierr);
      ierr = PetscBLASIntCast(temp_constraints,&Blas_N);CHKERRQ(ierr);
      ierr = PetscBLASIntCast(temp_constraints,&Blas_K);CHKERRQ(ierr);
      ierr = PetscBLASIntCast(size_of_constraint,&Blas_LDA);CHKERRQ(ierr);
      ierr = PetscBLASIntCast(temp_constraints,&Blas_LDB);CHKERRQ(ierr);
      ierr = PetscBLASIntCast(size_of_constraint,&Blas_LDC);CHKERRQ(ierr);
      if (j<temp_constraints) {
        PetscInt ii;
        for (k=j;k<temp_constraints;k++) singular_vals[k]=1.0/PetscSqrtReal(singular_vals[k]);
        ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
        PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&Blas_M,&Blas_N,&Blas_K,&one,&temp_quadrature_constraint[temp_indices[temp_start_ptr]],&Blas_LDA,correlation_mat,&Blas_LDB,&zero,temp_basis,&Blas_LDC));
        ierr = PetscFPTrapPop();CHKERRQ(ierr);
        for (k=0;k<temp_constraints-j;k++) {
          for (ii=0;ii<size_of_constraint;ii++) {
            temp_quadrature_constraint[temp_indices[temp_start_ptr+k]+ii]=singular_vals[temp_constraints-1-k]*temp_basis[(temp_constraints-1-k)*size_of_constraint+ii];
          }
        }
      }
#else  /* on missing GESVD */
      ierr = PetscBLASIntCast(size_of_constraint,&Blas_M);CHKERRQ(ierr);
      ierr = PetscBLASIntCast(temp_constraints,&Blas_N);CHKERRQ(ierr);
      ierr = PetscBLASIntCast(size_of_constraint,&Blas_LDA);CHKERRQ(ierr);
      ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
      PetscStackCallBLAS("LAPACKgesvd",LAPACKgesvd_("O","N",&Blas_M,&Blas_N,&temp_quadrature_constraint[temp_indices[temp_start_ptr]],&Blas_LDA,singular_vals,&dummy_scalar_1,&dummy_int_1,&dummy_scalar_2,&dummy_int_2,work,&lwork,&lierr));
#else
      PetscStackCallBLAS("LAPACKgesvd",LAPACKgesvd_("O","N",&Blas_M,&Blas_N,&temp_quadrature_constraint[temp_indices[temp_start_ptr]],&Blas_LDA,singular_vals,&dummy_scalar_1,&dummy_int_1,&dummy_scalar_2,&dummy_int_2,work,&lwork,rwork,&lierr));
#endif
      if (lierr) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in GESVD Lapack routine %d",(int)lierr);
      ierr = PetscFPTrapPop();CHKERRQ(ierr);
      /* retain eigenvalues greater than tol: note that LAPACKgesvd gives eigs in descending order */
      k = temp_constraints;
      if (k > size_of_constraint) k = size_of_constraint;
      j = 0;
      while (j < k && singular_vals[k-j-1] < tol) j++;
      total_counts = total_counts-temp_constraints+k-j;
#endif /* on missing GESVD */
    }
  }
  /* free index sets of faces, edges and vertices */
  for (i=0;i<n_ISForFaces;i++) {
    ierr = ISDestroy(&ISForFaces[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree(ISForFaces);CHKERRQ(ierr);
  for (i=0;i<n_ISForEdges;i++) {
    ierr = ISDestroy(&ISForEdges[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree(ISForEdges);CHKERRQ(ierr);
  ierr = ISDestroy(&ISForVertices);CHKERRQ(ierr);

  /* free workspace */
  if (!pcbddc->use_nnsp_true && !skip_lapack) {
    ierr = PetscFree(work);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
    ierr = PetscFree(rwork);CHKERRQ(ierr);
#endif
    ierr = PetscFree(singular_vals);CHKERRQ(ierr);
#if defined(PETSC_MISSING_LAPACK_GESVD)
    ierr = PetscFree(correlation_mat);CHKERRQ(ierr);
    ierr = PetscFree(temp_basis);CHKERRQ(ierr);
#endif
  }
  for (k=0;k<nnsp_size;k++) {
    ierr = VecDestroy(&localnearnullsp[k]);CHKERRQ(ierr);
  }
  ierr = PetscFree(localnearnullsp);CHKERRQ(ierr);

  /* set quantities in pcbddc data structure */
  /* n_vertices defines the number of subdomain corners in the primal space */
  /* n_constraints defines the number of averages (they can be point primal dofs if change of basis is requested) */
  pcbddc->local_primal_size = total_counts;
  pcbddc->n_vertices = n_vertices;
  pcbddc->n_constraints = pcbddc->local_primal_size-pcbddc->n_vertices;

  /* Create constraint matrix */
  /* The constraint matrix is used to compute the l2g map of primal dofs */
  /* so we need to set it up properly either with or without change of basis */
  ierr = MatCreate(PETSC_COMM_SELF,&pcbddc->ConstraintMatrix);CHKERRQ(ierr);
  ierr = MatSetType(pcbddc->ConstraintMatrix,impMatType);CHKERRQ(ierr);
  ierr = MatSetSizes(pcbddc->ConstraintMatrix,pcbddc->local_primal_size,pcis->n,pcbddc->local_primal_size,pcis->n);CHKERRQ(ierr);
  /* array to compute a local numbering of constraints : vertices first then constraints */
  ierr = PetscMalloc(pcbddc->local_primal_size*sizeof(PetscInt),&aux_primal_numbering);CHKERRQ(ierr);
  /* array to select the proper local node (of minimum index with respect to global ordering) when changing the basis */
  /* note: it should not be needed since IS for faces and edges are already sorted by global ordering when analyzing the graph but... just in case */
  ierr = PetscMalloc(pcbddc->local_primal_size*sizeof(PetscInt),&aux_primal_minloc);CHKERRQ(ierr);
  /* auxiliary stuff for basis change */
  ierr = PetscMalloc(max_size_of_constraint*sizeof(PetscInt),&global_indices);CHKERRQ(ierr);
  ierr = PetscMalloc(pcis->n_B*sizeof(PetscBool),&touched);CHKERRQ(ierr);
  ierr = PetscMemzero(touched,pcis->n_B*sizeof(PetscBool));CHKERRQ(ierr);

  /* find primal_dofs: subdomain corners plus dofs selected as primal after change of basis */
  total_primal_vertices=0;
  for (i=0;i<pcbddc->local_primal_size;i++) {
    size_of_constraint=temp_indices[i+1]-temp_indices[i];
    if (size_of_constraint == 1) {
      touched[temp_indices_to_constraint_B[temp_indices[i]]]=PETSC_TRUE;
      aux_primal_numbering[total_primal_vertices]=temp_indices_to_constraint[temp_indices[i]];
      aux_primal_minloc[total_primal_vertices]=0;
      total_primal_vertices++;
    } else if (change_basis[i]) { /* Same procedure used in PCBDDCGetPrimalConstraintsLocalIdx */
      PetscInt min_loc,min_index;
      ierr = ISLocalToGlobalMappingApply(pcbddc->mat_graph->l2gmap,size_of_constraint,&temp_indices_to_constraint[temp_indices[i]],global_indices);CHKERRQ(ierr);
      /* find first untouched local node */
      k = 0;
      while (touched[temp_indices_to_constraint_B[temp_indices[i]+k]]) k++;
      min_index = global_indices[k];
      min_loc = k;
      /* search the minimum among global nodes already untouched on the cc */
      for (k=1;k<size_of_constraint;k++) {
        /* there can be more than one constraint on a single connected component */
        if (min_index > global_indices[k] && !touched[temp_indices_to_constraint_B[temp_indices[i]+k]]) {
          min_index = global_indices[k];
          min_loc = k;
        }
      }
      touched[temp_indices_to_constraint_B[temp_indices[i]+min_loc]] = PETSC_TRUE;
      aux_primal_numbering[total_primal_vertices]=temp_indices_to_constraint[temp_indices[i]+min_loc];
      aux_primal_minloc[total_primal_vertices]=min_loc;
      total_primal_vertices++;
    }
  }
  /* free workspace */
  ierr = PetscFree(global_indices);CHKERRQ(ierr);
  ierr = PetscFree(touched);CHKERRQ(ierr);
  /* permute indices in order to have a sorted set of vertices */
  ierr = PetscSortInt(total_primal_vertices,aux_primal_numbering);

  /* nonzero structure of constraint matrix */
  ierr = PetscMalloc(pcbddc->local_primal_size*sizeof(PetscInt),&nnz);CHKERRQ(ierr);
  for (i=0;i<total_primal_vertices;i++) nnz[i]=1;
  j=total_primal_vertices;
  for (i=pcbddc->n_vertices;i<pcbddc->local_primal_size;i++) {
    if (!change_basis[i]) {
      nnz[j]=temp_indices[i+1]-temp_indices[i];
      j++;
    }
  }
  ierr = MatSeqAIJSetPreallocation(pcbddc->ConstraintMatrix,0,nnz);CHKERRQ(ierr);
  ierr = PetscFree(nnz);CHKERRQ(ierr);
  /* set values in constraint matrix */
  for (i=0;i<total_primal_vertices;i++) {
    ierr = MatSetValue(pcbddc->ConstraintMatrix,i,aux_primal_numbering[i],1.0,INSERT_VALUES);CHKERRQ(ierr);
  }
  total_counts = total_primal_vertices;
  for (i=pcbddc->n_vertices;i<pcbddc->local_primal_size;i++) {
    if (!change_basis[i]) {
      size_of_constraint=temp_indices[i+1]-temp_indices[i];
      ierr = MatSetValues(pcbddc->ConstraintMatrix,1,&total_counts,size_of_constraint,&temp_indices_to_constraint[temp_indices[i]],&temp_quadrature_constraint[temp_indices[i]],INSERT_VALUES);CHKERRQ(ierr);
      total_counts++;
    }
  }
  /* assembling */
  ierr = MatAssemblyBegin(pcbddc->ConstraintMatrix,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(pcbddc->ConstraintMatrix,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  /*
  ierr = MatView(pcbddc->ConstraintMatrix,(PetscViewer)0);CHKERRQ(ierr);
  */
  /* Create matrix for change of basis. We don't need it in case pcbddc->use_change_of_basis is FALSE */
  if (pcbddc->use_change_of_basis) {
    PetscBool qr_needed = PETSC_FALSE;
    /* change of basis acts on local interfaces -> dimension is n_B x n_B */
    ierr = MatCreate(PETSC_COMM_SELF,&pcbddc->ChangeOfBasisMatrix);CHKERRQ(ierr);
    ierr = MatSetType(pcbddc->ChangeOfBasisMatrix,impMatType);CHKERRQ(ierr);
    ierr = MatSetSizes(pcbddc->ChangeOfBasisMatrix,pcis->n_B,pcis->n_B,pcis->n_B,pcis->n_B);CHKERRQ(ierr);
    /* work arrays */
    ierr = PetscMalloc(pcis->n_B*sizeof(PetscInt),&nnz);CHKERRQ(ierr);
    for (i=0;i<pcis->n_B;i++) nnz[i]=1;
    /* nonzeros per row */
    for (i=pcbddc->n_vertices;i<pcbddc->local_primal_size;i++) {
      if (change_basis[i]) {
        qr_needed = PETSC_TRUE;
        size_of_constraint = temp_indices[i+1]-temp_indices[i];
        for (j=0;j<size_of_constraint;j++) nnz[temp_indices_to_constraint_B[temp_indices[i]+j]] = size_of_constraint;
      }
    }
    ierr = MatSeqAIJSetPreallocation(pcbddc->ChangeOfBasisMatrix,0,nnz);CHKERRQ(ierr);
    ierr = PetscFree(nnz);CHKERRQ(ierr);
    /* Set initial identity in the matrix */
    for (i=0;i<pcis->n_B;i++) {
      ierr = MatSetValue(pcbddc->ChangeOfBasisMatrix,i,i,1.0,INSERT_VALUES);CHKERRQ(ierr);
    }

    /* Now we loop on the constraints which need a change of basis */
    /* Change of basis matrix is evaluated as the FIRST APPROACH in */
    /* Klawonn and Widlund, Dual-primal FETI-DP methods for linear elasticity, (see Sect 6.2.1) */
    /* Change of basis matrix T computed via QR decomposition of constraints */
    if (qr_needed) {
      /* dual and primal dofs on a single cc */
      PetscInt     dual_dofs,primal_dofs;
      /* iterator on aux_primal_minloc (ordered as read from nearnullspace: vertices, edges and then constraints) */
      PetscInt     primal_counter;
      /* working stuff for GEQRF */
      PetscScalar  *qr_basis,*qr_tau,*qr_work,lqr_work_t;
      PetscBLASInt lqr_work;
      /* working stuff for UNGQR */
      PetscScalar  *gqr_work,lgqr_work_t;
      PetscBLASInt lgqr_work;
      /* working stuff for TRTRS */
      PetscScalar  *trs_rhs;
      PetscBLASInt Blas_NRHS;
      /* pointers for values insertion into change of basis matrix */
      PetscInt     *start_rows,*start_cols;
      PetscScalar  *start_vals;
      /* working stuff for values insertion */
      PetscBool    *is_primal;
 
      /* space to store Q */ 
      ierr = PetscMalloc((max_size_of_constraint)*(max_size_of_constraint)*sizeof(PetscScalar),&qr_basis);CHKERRQ(ierr);
      /* first we issue queries for optimal work */
      ierr = PetscBLASIntCast(max_size_of_constraint,&Blas_M);CHKERRQ(ierr);
      ierr = PetscBLASIntCast(max_constraints,&Blas_N);CHKERRQ(ierr);
      ierr = PetscBLASIntCast(max_size_of_constraint,&Blas_LDA);CHKERRQ(ierr);
      lqr_work = -1;
      PetscStackCallBLAS("LAPACKgeqrf",LAPACKgeqrf_(&Blas_M,&Blas_N,qr_basis,&Blas_LDA,qr_tau,&lqr_work_t,&lqr_work,&lierr));
      if (lierr) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in query to GEQRF Lapack routine %d",(int)lierr);
      ierr = PetscBLASIntCast((PetscInt)PetscRealPart(lqr_work_t),&lqr_work);CHKERRQ(ierr);
      ierr = PetscMalloc((PetscInt)PetscRealPart(lqr_work_t)*sizeof(*qr_work),&qr_work);CHKERRQ(ierr);
      lgqr_work = -1;
      ierr = PetscBLASIntCast(max_size_of_constraint,&Blas_M);CHKERRQ(ierr);
      ierr = PetscBLASIntCast(max_size_of_constraint,&Blas_N);CHKERRQ(ierr);
      ierr = PetscBLASIntCast(max_constraints,&Blas_K);CHKERRQ(ierr);
      ierr = PetscBLASIntCast(max_size_of_constraint,&Blas_LDA);CHKERRQ(ierr);
      if (Blas_K>Blas_M) Blas_K=Blas_M; /* adjust just for computing optimal work */
      PetscStackCallBLAS("LAPACKungqr",LAPACKungqr_(&Blas_M,&Blas_N,&Blas_K,qr_basis,&Blas_LDA,qr_tau,&lgqr_work_t,&lgqr_work,&lierr));
      if (lierr) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in query to UNGQR Lapack routine %d",(int)lierr);
      ierr = PetscBLASIntCast((PetscInt)PetscRealPart(lgqr_work_t),&lgqr_work);CHKERRQ(ierr);
      ierr = PetscMalloc((PetscInt)PetscRealPart(lgqr_work_t)*sizeof(*gqr_work),&gqr_work);CHKERRQ(ierr);
      /* array to store scaling factors for reflectors */
      ierr = PetscMalloc(max_constraints*sizeof(*qr_tau),&qr_tau);CHKERRQ(ierr);
      /* array to store rhs and solution of triangular solver */
      ierr = PetscMalloc(max_constraints*max_constraints*sizeof(*trs_rhs),&trs_rhs);CHKERRQ(ierr);
      /* array to store whether a node is primal or not */
      ierr = PetscMalloc(pcis->n_B*sizeof(*is_primal),&is_primal);CHKERRQ(ierr);
      ierr = PetscMemzero(is_primal,pcis->n_B*sizeof(*is_primal));CHKERRQ(ierr);
      for (i=0;i<total_primal_vertices;i++) is_primal[local_to_B[aux_primal_numbering[i]]] = PETSC_TRUE;

      /* allocating workspace for check */
      if (pcbddc->dbg_flag) {
        ierr = PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"--------------------------------------------------------------\n");CHKERRQ(ierr);
        ierr = PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"Checking change of basis computation for subdomain %04d\n",PetscGlobalRank);CHKERRQ(ierr);
        ierr = PetscMalloc(max_size_of_constraint*(max_constraints+max_size_of_constraint)*sizeof(*work),&work);CHKERRQ(ierr);
      }

      /* loop on constraints and see whether or not they need a change of basis */
      /* -> using implicit ordering contained in temp_indices data */
      total_counts = pcbddc->n_vertices;
      primal_counter = total_counts;
      while (total_counts<pcbddc->local_primal_size) {
        primal_dofs = 1;
        if (change_basis[total_counts]) {
          /* get all constraints with same support: if more then one constraint is present on the cc then surely indices are stored contiguosly */
          while (total_counts+primal_dofs < pcbddc->local_primal_size && temp_indices_to_constraint_B[temp_indices[total_counts]] == temp_indices_to_constraint_B[temp_indices[total_counts+primal_dofs]]) {
            primal_dofs++;
          }
          /* get constraint info */
          size_of_constraint = temp_indices[total_counts+1]-temp_indices[total_counts];
          dual_dofs = size_of_constraint-primal_dofs;

          /* copy quadrature constraints for change of basis check */
          if (pcbddc->dbg_flag) {
            ierr = PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"Constraint %d to %d need a change of basis (size %d)\n",total_counts,total_counts+primal_dofs,size_of_constraint);CHKERRQ(ierr);
            ierr = PetscMemcpy(work,&temp_quadrature_constraint[temp_indices[total_counts]],size_of_constraint*primal_dofs*sizeof(PetscScalar));CHKERRQ(ierr);
          } 
 
          /* copy temporary constraints into larger work vector (in order to store all columns of Q) */
          ierr = PetscMemcpy(qr_basis,&temp_quadrature_constraint[temp_indices[total_counts]],size_of_constraint*primal_dofs*sizeof(PetscScalar));CHKERRQ(ierr);
     
          /* compute QR decomposition of constraints */
          ierr = PetscBLASIntCast(size_of_constraint,&Blas_M);CHKERRQ(ierr);
          ierr = PetscBLASIntCast(primal_dofs,&Blas_N);CHKERRQ(ierr);
          ierr = PetscBLASIntCast(size_of_constraint,&Blas_LDA);CHKERRQ(ierr);
          ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
          PetscStackCallBLAS("LAPACKgeqrf",LAPACKgeqrf_(&Blas_M,&Blas_N,qr_basis,&Blas_LDA,qr_tau,qr_work,&lqr_work,&lierr));
          if (lierr) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in GEQRF Lapack routine %d",(int)lierr);
          ierr = PetscFPTrapPop();CHKERRQ(ierr);
  
          /* explictly compute R^-T */
          ierr = PetscMemzero(trs_rhs,primal_dofs*primal_dofs*sizeof(*trs_rhs));CHKERRQ(ierr);
          for (j=0;j<primal_dofs;j++) trs_rhs[j*(primal_dofs+1)] = 1.0;
          ierr = PetscBLASIntCast(primal_dofs,&Blas_N);CHKERRQ(ierr);
          ierr = PetscBLASIntCast(primal_dofs,&Blas_NRHS);CHKERRQ(ierr);
          ierr = PetscBLASIntCast(size_of_constraint,&Blas_LDA);CHKERRQ(ierr);
          ierr = PetscBLASIntCast(primal_dofs,&Blas_LDB);CHKERRQ(ierr);
          ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
          PetscStackCallBLAS("LAPACKtrtrs",LAPACKtrtrs_("U","T","N",&Blas_N,&Blas_NRHS,qr_basis,&Blas_LDA,trs_rhs,&Blas_LDB,&lierr)); 
          if (lierr) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in TRTRS Lapack routine %d",(int)lierr);
          ierr = PetscFPTrapPop();CHKERRQ(ierr);
  
          /* explcitly compute all columns of Q (Q = [Q1 | Q2] ) overwriting QR factorization in qr_basis */
          ierr = PetscBLASIntCast(size_of_constraint,&Blas_M);CHKERRQ(ierr);
          ierr = PetscBLASIntCast(size_of_constraint,&Blas_N);CHKERRQ(ierr);
          ierr = PetscBLASIntCast(primal_dofs,&Blas_K);CHKERRQ(ierr);
          ierr = PetscBLASIntCast(size_of_constraint,&Blas_LDA);CHKERRQ(ierr);
          ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
          PetscStackCallBLAS("LAPACKungqr",LAPACKungqr_(&Blas_M,&Blas_N,&Blas_K,qr_basis,&Blas_LDA,qr_tau,gqr_work,&lgqr_work,&lierr));
          if (lierr) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in UNGQR Lapack routine %d",(int)lierr);
          ierr = PetscFPTrapPop();CHKERRQ(ierr);
  
          /* first primal_dofs columns of Q need to be re-scaled in order to be unitary w.r.t constraints
             i.e. C_{pxn}*Q_{nxn} should be equal to [I_pxp | 0_pxd] (see check below)
             where n=size_of_constraint, p=primal_dofs, d=dual_dofs (n=p+d), I and 0 identity and null matrix resp. */
          ierr = PetscBLASIntCast(size_of_constraint,&Blas_M);CHKERRQ(ierr);
          ierr = PetscBLASIntCast(primal_dofs,&Blas_N);CHKERRQ(ierr);
          ierr = PetscBLASIntCast(primal_dofs,&Blas_K);CHKERRQ(ierr);
          ierr = PetscBLASIntCast(size_of_constraint,&Blas_LDA);CHKERRQ(ierr);
          ierr = PetscBLASIntCast(primal_dofs,&Blas_LDB);CHKERRQ(ierr);
          ierr = PetscBLASIntCast(size_of_constraint,&Blas_LDC);CHKERRQ(ierr);
          ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
          PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&Blas_M,&Blas_N,&Blas_K,&one,qr_basis,&Blas_LDA,trs_rhs,&Blas_LDB,&zero,&temp_quadrature_constraint[temp_indices[total_counts]],&Blas_LDC));
          ierr = PetscFPTrapPop();CHKERRQ(ierr);
          ierr = PetscMemcpy(qr_basis,&temp_quadrature_constraint[temp_indices[total_counts]],size_of_constraint*primal_dofs*sizeof(PetscScalar));CHKERRQ(ierr);

          /* insert values in change of basis matrix respecting global ordering of new primal dofs */ 
          start_rows = &temp_indices_to_constraint_B[temp_indices[total_counts]];
          /* insert cols for primal dofs */
          for (j=0;j<primal_dofs;j++) {
            start_vals = &qr_basis[j*size_of_constraint];
            start_cols = &temp_indices_to_constraint_B[temp_indices[total_counts]+aux_primal_minloc[primal_counter+j]];
            ierr = MatSetValues(pcbddc->ChangeOfBasisMatrix,size_of_constraint,start_rows,1,start_cols,start_vals,INSERT_VALUES);CHKERRQ(ierr);
          }
          /* insert cols for dual dofs */
          for (j=0,k=0;j<dual_dofs;k++) {
            if (!is_primal[temp_indices_to_constraint_B[temp_indices[total_counts]+k]]) {
              start_vals = &qr_basis[(primal_dofs+j)*size_of_constraint];
              start_cols = &temp_indices_to_constraint_B[temp_indices[total_counts]+k];
              ierr = MatSetValues(pcbddc->ChangeOfBasisMatrix,size_of_constraint,start_rows,1,start_cols,start_vals,INSERT_VALUES);CHKERRQ(ierr);
              j++;
            }
          }

          /* check change of basis */
          if (pcbddc->dbg_flag) {
            PetscInt   ii,jj;
            PetscBool valid_qr=PETSC_TRUE;
            ierr = PetscBLASIntCast(primal_dofs,&Blas_M);CHKERRQ(ierr);
            ierr = PetscBLASIntCast(size_of_constraint,&Blas_N);CHKERRQ(ierr);
            ierr = PetscBLASIntCast(size_of_constraint,&Blas_K);CHKERRQ(ierr);
            ierr = PetscBLASIntCast(size_of_constraint,&Blas_LDA);CHKERRQ(ierr);
            ierr = PetscBLASIntCast(size_of_constraint,&Blas_LDB);CHKERRQ(ierr);
            ierr = PetscBLASIntCast(primal_dofs,&Blas_LDC);CHKERRQ(ierr);
            ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
            PetscStackCallBLAS("BLASgemm",BLASgemm_("T","N",&Blas_M,&Blas_N,&Blas_K,&one,work,&Blas_LDA,qr_basis,&Blas_LDB,&zero,&work[size_of_constraint*primal_dofs],&Blas_LDC));
            ierr = PetscFPTrapPop();CHKERRQ(ierr);
            for (jj=0;jj<size_of_constraint;jj++) {
              for (ii=0;ii<primal_dofs;ii++) {
                if (ii != jj && PetscAbsScalar(work[size_of_constraint*primal_dofs+jj*primal_dofs+ii]) > 1.e-12) valid_qr = PETSC_FALSE;
                if (ii == jj && PetscAbsScalar(work[size_of_constraint*primal_dofs+jj*primal_dofs+ii]-1.0) > 1.e-12) valid_qr = PETSC_FALSE;
              }
            }
            if (!valid_qr) {
              ierr = PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"\t-> wrong change of basis!\n",PetscGlobalRank);CHKERRQ(ierr);
              for (jj=0;jj<size_of_constraint;jj++) {
                for (ii=0;ii<primal_dofs;ii++) {
                  if (ii != jj && PetscAbsScalar(work[size_of_constraint*primal_dofs+jj*primal_dofs+ii]) > 1.e-12) {
                    PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"\tQr basis function %d is not orthogonal to constraint %d (%1.14e)!\n",jj,ii,PetscAbsScalar(work[size_of_constraint*primal_dofs+jj*primal_dofs+ii]));
                  }
                  if (ii == jj && PetscAbsScalar(work[size_of_constraint*primal_dofs+jj*primal_dofs+ii]-1.0) > 1.e-12) {
                    PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"\tQr basis function %d is not unitary w.r.t constraint %d (%1.14e)!\n",jj,ii,PetscAbsScalar(work[size_of_constraint*primal_dofs+jj*primal_dofs+ii]));
                  }
                }
              }
            } else {
              ierr = PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"\t-> right change of basis!\n",PetscGlobalRank);CHKERRQ(ierr);
            }
          }
          /* increment primal counter */
          primal_counter += primal_dofs;
        } else {
          if (pcbddc->dbg_flag) {
            ierr = PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"Constraint %d does not need a change of basis (size %d)\n",total_counts,temp_indices[total_counts+1]-temp_indices[total_counts]);CHKERRQ(ierr);
          }
        }
        /* increment constraint counter total_counts */
        total_counts += primal_dofs;
      }
      if (pcbddc->dbg_flag) {
        ierr = PetscViewerFlush(pcbddc->dbg_viewer);CHKERRQ(ierr);
        ierr = PetscFree(work);CHKERRQ(ierr);
      }
      /* free workspace */
      ierr = PetscFree(trs_rhs);CHKERRQ(ierr);
      ierr = PetscFree(qr_tau);CHKERRQ(ierr);
      ierr = PetscFree(qr_work);CHKERRQ(ierr);
      ierr = PetscFree(gqr_work);CHKERRQ(ierr);
      ierr = PetscFree(is_primal);CHKERRQ(ierr);
      ierr = PetscFree(qr_basis);CHKERRQ(ierr);
    }
    /* assembling */
    ierr = MatAssemblyBegin(pcbddc->ChangeOfBasisMatrix,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(pcbddc->ChangeOfBasisMatrix,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    /*
    ierr = MatView(pcbddc->ChangeOfBasisMatrix,(PetscViewer)0);CHKERRQ(ierr);
    */
  }
  /* free workspace */
  ierr = PetscFree(aux_primal_numbering);CHKERRQ(ierr);
  ierr = PetscFree(aux_primal_minloc);CHKERRQ(ierr);
  ierr = PetscFree(temp_indices);CHKERRQ(ierr);
  ierr = PetscFree(change_basis);CHKERRQ(ierr);
  ierr = PetscFree(temp_indices_to_constraint);CHKERRQ(ierr);
  ierr = PetscFree(temp_indices_to_constraint_B);CHKERRQ(ierr);
  ierr = PetscFree(local_to_B);CHKERRQ(ierr);
  ierr = PetscFree(temp_quadrature_constraint);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCBDDCAnalyzeInterface"
PetscErrorCode PCBDDCAnalyzeInterface(PC pc)
{
  PC_BDDC     *pcbddc = (PC_BDDC*)pc->data;
  PC_IS       *pcis = (PC_IS*)pc->data;
  Mat_IS      *matis  = (Mat_IS*)pc->pmat->data;
  PetscInt    bs,ierr,i,vertex_size;
  PetscViewer viewer=pcbddc->dbg_viewer;

  PetscFunctionBegin;
  /* Init local Graph struct */
  ierr = PCBDDCGraphInit(pcbddc->mat_graph,matis->mapping);CHKERRQ(ierr);

  /* Check validity of the csr graph passed in by the user */
  if (pcbddc->mat_graph->nvtxs_csr != pcbddc->mat_graph->nvtxs) {
    ierr = PCBDDCGraphResetCSR(pcbddc->mat_graph);CHKERRQ(ierr);
  }
  /* Set default CSR adjacency of local dofs if not provided by the user with PCBDDCSetLocalAdjacencyGraph */
  if (!pcbddc->mat_graph->xadj || !pcbddc->mat_graph->adjncy) {
    Mat mat_adj;
    const PetscInt *xadj,*adjncy;
    PetscBool flg_row=PETSC_TRUE;

    ierr = MatConvert(matis->A,MATMPIADJ,MAT_INITIAL_MATRIX,&mat_adj);CHKERRQ(ierr);
    ierr = MatGetRowIJ(mat_adj,0,PETSC_TRUE,PETSC_FALSE,&i,&xadj,&adjncy,&flg_row);CHKERRQ(ierr);
    if (!flg_row) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Error in MatGetRowIJ called in %s\n",__FUNCT__);
    }
    ierr = PCBDDCSetLocalAdjacencyGraph(pc,i,xadj,adjncy,PETSC_COPY_VALUES);CHKERRQ(ierr);
    ierr = MatRestoreRowIJ(mat_adj,0,PETSC_TRUE,PETSC_FALSE,&i,&xadj,&adjncy,&flg_row);CHKERRQ(ierr);
    if (!flg_row) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Error in MatRestoreRowIJ called in %s\n",__FUNCT__);
    }
    ierr = MatDestroy(&mat_adj);CHKERRQ(ierr);
  }

  /* Set default dofs' splitting if no information has been provided by the user with PCBDDCSetDofsSplitting */
  vertex_size = 1;
  if (!pcbddc->n_ISForDofs) {
    IS *custom_ISForDofs;

    ierr = MatGetBlockSize(matis->A,&bs);CHKERRQ(ierr);
    ierr = PetscMalloc(bs*sizeof(IS),&custom_ISForDofs);CHKERRQ(ierr);
    for (i=0;i<bs;i++) {
      ierr = ISCreateStride(PETSC_COMM_SELF,pcis->n/bs,i,bs,&custom_ISForDofs[i]);CHKERRQ(ierr);
    }
    ierr = PCBDDCSetDofsSplitting(pc,bs,custom_ISForDofs);CHKERRQ(ierr);
    /* remove my references to IS objects */
    for (i=0;i<bs;i++) {
      ierr = ISDestroy(&custom_ISForDofs[i]);CHKERRQ(ierr);
    }
    ierr = PetscFree(custom_ISForDofs);CHKERRQ(ierr);
  } else { /* mat block size as vertex size (used for elasticity) */
    ierr = MatGetBlockSize(matis->A,&vertex_size);CHKERRQ(ierr);
  }

  /* Setup of Graph */
  ierr = PCBDDCGraphSetUp(pcbddc->mat_graph,vertex_size,pcbddc->NeumannBoundaries,pcbddc->DirichletBoundaries,pcbddc->n_ISForDofs,pcbddc->ISForDofs,pcbddc->user_primal_vertices);

  /* Graph's connected components analysis */
  ierr = PCBDDCGraphComputeConnectedComponents(pcbddc->mat_graph);CHKERRQ(ierr);

  /* print some info to stdout */
  if (pcbddc->dbg_flag) {
    ierr = PCBDDCGraphASCIIView(pcbddc->mat_graph,pcbddc->dbg_flag,viewer);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCBDDCGetPrimalVerticesLocalIdx"
PetscErrorCode  PCBDDCGetPrimalVerticesLocalIdx(PC pc, PetscInt *n_vertices, PetscInt *vertices_idx[])
{
  PC_BDDC        *pcbddc = (PC_BDDC*)(pc->data);
  PetscInt       *vertices,*row_cmat_indices,n,i,size_of_constraint,local_primal_size;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  n = 0;
  vertices = 0;
  if (pcbddc->ConstraintMatrix) {
    ierr = MatGetSize(pcbddc->ConstraintMatrix,&local_primal_size,&i);CHKERRQ(ierr);
    for (i=0;i<local_primal_size;i++) {
      ierr = MatGetRow(pcbddc->ConstraintMatrix,i,&size_of_constraint,NULL,NULL);CHKERRQ(ierr);
      if (size_of_constraint == 1) n++;
      ierr = MatRestoreRow(pcbddc->ConstraintMatrix,i,&size_of_constraint,NULL,NULL);CHKERRQ(ierr);
    }
    if (vertices_idx) {
      ierr = PetscMalloc(n*sizeof(PetscInt),&vertices);CHKERRQ(ierr);
      n = 0;
      for (i=0;i<local_primal_size;i++) {
        ierr = MatGetRow(pcbddc->ConstraintMatrix,i,&size_of_constraint,(const PetscInt**)&row_cmat_indices,NULL);CHKERRQ(ierr);
        if (size_of_constraint == 1) {
          vertices[n++]=row_cmat_indices[0];
        }
        ierr = MatRestoreRow(pcbddc->ConstraintMatrix,i,&size_of_constraint,(const PetscInt**)&row_cmat_indices,NULL);CHKERRQ(ierr);
      }
    }
  }
  *n_vertices = n;
  if (vertices_idx) *vertices_idx = vertices;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCBDDCGetPrimalConstraintsLocalIdx"
PetscErrorCode  PCBDDCGetPrimalConstraintsLocalIdx(PC pc, PetscInt *n_constraints, PetscInt *constraints_idx[])
{
  PC_BDDC        *pcbddc = (PC_BDDC*)(pc->data);
  PetscInt       *constraints_index,*row_cmat_indices,*row_cmat_global_indices;
  PetscInt       n,i,j,size_of_constraint,local_primal_size,local_size,max_size_of_constraint,min_index,min_loc;
  PetscBool      *touched;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  n = 0;
  constraints_index = 0;
  if (pcbddc->ConstraintMatrix) {
    ierr = MatGetSize(pcbddc->ConstraintMatrix,&local_primal_size,&local_size);CHKERRQ(ierr);
    max_size_of_constraint = 0;
    for (i=0;i<local_primal_size;i++) {
      ierr = MatGetRow(pcbddc->ConstraintMatrix,i,&size_of_constraint,NULL,NULL);CHKERRQ(ierr);
      if (size_of_constraint > 1) {
        n++;
      }
      max_size_of_constraint = PetscMax(size_of_constraint,max_size_of_constraint);
      ierr = MatRestoreRow(pcbddc->ConstraintMatrix,i,&size_of_constraint,NULL,NULL);CHKERRQ(ierr);
    }
    if (constraints_idx) {
      ierr = PetscMalloc(n*sizeof(PetscInt),&constraints_index);CHKERRQ(ierr);
      ierr = PetscMalloc(max_size_of_constraint*sizeof(PetscInt),&row_cmat_global_indices);CHKERRQ(ierr);
      ierr = PetscMalloc(local_size*sizeof(PetscBool),&touched);CHKERRQ(ierr);
      ierr = PetscMemzero(touched,local_size*sizeof(PetscBool));CHKERRQ(ierr);
      n = 0;
      for (i=0;i<local_primal_size;i++) {
        ierr = MatGetRow(pcbddc->ConstraintMatrix,i,&size_of_constraint,(const PetscInt**)&row_cmat_indices,NULL);CHKERRQ(ierr);
        if (size_of_constraint > 1) {
          ierr = ISLocalToGlobalMappingApply(pcbddc->mat_graph->l2gmap,size_of_constraint,row_cmat_indices,row_cmat_global_indices);CHKERRQ(ierr);
          /* find first untouched local node */
          j = 0;
          while(touched[row_cmat_indices[j]]) j++;
          min_index = row_cmat_global_indices[j];
          min_loc = j;
          /* search the minimum among nodes not yet touched on the connected component
             since there can be more than one constraint on a single cc */
          for (j=1;j<size_of_constraint;j++) {
            if (min_index > row_cmat_global_indices[j] && !touched[row_cmat_indices[j]]) {
              min_index = row_cmat_global_indices[j];
              min_loc = j;
            }
          }
          touched[row_cmat_indices[min_loc]] = PETSC_TRUE;
          constraints_index[n++] = row_cmat_indices[min_loc];
        }
        ierr = MatRestoreRow(pcbddc->ConstraintMatrix,i,&size_of_constraint,(const PetscInt**)&row_cmat_indices,NULL);CHKERRQ(ierr);
      }
      ierr = PetscFree(touched);CHKERRQ(ierr);
      ierr = PetscFree(row_cmat_global_indices);CHKERRQ(ierr);
    }
  }
  *n_constraints = n;
  if (constraints_idx) *constraints_idx = constraints_index;
  PetscFunctionReturn(0);
}

/* the next two functions has been adapted from pcis.c */
#undef __FUNCT__
#define __FUNCT__ "PCBDDCApplySchur"
PetscErrorCode  PCBDDCApplySchur(PC pc, Vec v, Vec vec1_B, Vec vec2_B, Vec vec1_D, Vec vec2_D)
{
  PetscErrorCode ierr;
  PC_IS          *pcis = (PC_IS*)(pc->data);

  PetscFunctionBegin;
  if (!vec2_B) { vec2_B = v; }
  ierr = MatMult(pcis->A_BB,v,vec1_B);CHKERRQ(ierr);
  ierr = MatMult(pcis->A_IB,v,vec1_D);CHKERRQ(ierr);
  ierr = KSPSolve(pcis->ksp_D,vec1_D,vec2_D);CHKERRQ(ierr);
  ierr = MatMult(pcis->A_BI,vec2_D,vec2_B);CHKERRQ(ierr);
  ierr = VecAXPY(vec1_B,-1.0,vec2_B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCBDDCApplySchurTranspose"
PetscErrorCode  PCBDDCApplySchurTranspose(PC pc, Vec v, Vec vec1_B, Vec vec2_B, Vec vec1_D, Vec vec2_D)
{
  PetscErrorCode ierr;
  PC_IS          *pcis = (PC_IS*)(pc->data);

  PetscFunctionBegin;
  if (!vec2_B) { vec2_B = v; }
  ierr = MatMultTranspose(pcis->A_BB,v,vec1_B);CHKERRQ(ierr);
  ierr = MatMultTranspose(pcis->A_BI,v,vec1_D);CHKERRQ(ierr);
  ierr = KSPSolveTranspose(pcis->ksp_D,vec1_D,vec2_D);CHKERRQ(ierr);
  ierr = MatMultTranspose(pcis->A_IB,vec2_D,vec2_B);CHKERRQ(ierr);
  ierr = VecAXPY(vec1_B,-1.0,vec2_B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCBDDCSubsetNumbering"
PetscErrorCode PCBDDCSubsetNumbering(MPI_Comm comm,ISLocalToGlobalMapping l2gmap, PetscInt n_local_dofs, PetscInt local_dofs[], PetscInt local_dofs_mult[], PetscInt* n_global_subset, PetscInt* global_numbering_subset[])
{
  Vec            local_vec,global_vec;
  IS             seqis,paris;
  VecScatter     scatter_ctx;
  PetscScalar    *array;
  PetscInt       *temp_global_dofs;
  PetscScalar    globalsum;
  PetscInt       i,j,s;
  PetscInt       nlocals,first_index,old_index,max_local;
  PetscMPIInt    rank_prec_comm,size_prec_comm,max_global;
  PetscMPIInt    *dof_sizes,*dof_displs;
  PetscBool      first_found;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* mpi buffers */
  MPI_Comm_size(comm,&size_prec_comm);
  MPI_Comm_rank(comm,&rank_prec_comm);
  j = ( !rank_prec_comm ? size_prec_comm : 0);
  ierr = PetscMalloc(j*sizeof(*dof_sizes),&dof_sizes);CHKERRQ(ierr);
  ierr = PetscMalloc(j*sizeof(*dof_displs),&dof_displs);CHKERRQ(ierr);
  /* get maximum size of subset */
  ierr = PetscMalloc(n_local_dofs*sizeof(PetscInt),&temp_global_dofs);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingApply(l2gmap,n_local_dofs,local_dofs,temp_global_dofs);CHKERRQ(ierr);
  max_local = 0;
  if (n_local_dofs) {
    max_local = temp_global_dofs[0];
    for (i=1;i<n_local_dofs;i++) {
      if (max_local < temp_global_dofs[i] ) {
        max_local = temp_global_dofs[i];
      }
    }
  }
  ierr = MPI_Allreduce(&max_local,&max_global,1,MPIU_INT,MPI_MAX,comm);
  max_global++;
  max_local = 0;
  if (n_local_dofs) {
    max_local = local_dofs[0];
    for (i=1;i<n_local_dofs;i++) {
      if (max_local < local_dofs[i] ) {
        max_local = local_dofs[i];
      }
    }
  }
  max_local++;
  /* allocate workspace */
  ierr = VecCreate(PETSC_COMM_SELF,&local_vec);CHKERRQ(ierr);
  ierr = VecSetSizes(local_vec,PETSC_DECIDE,max_local);CHKERRQ(ierr);
  ierr = VecSetType(local_vec,VECSEQ);CHKERRQ(ierr);
  ierr = VecCreate(comm,&global_vec);CHKERRQ(ierr);
  ierr = VecSetSizes(global_vec,PETSC_DECIDE,max_global);CHKERRQ(ierr);
  ierr = VecSetType(global_vec,VECMPI);CHKERRQ(ierr);
  /* create scatter */
  ierr = ISCreateGeneral(PETSC_COMM_SELF,n_local_dofs,local_dofs,PETSC_COPY_VALUES,&seqis);CHKERRQ(ierr);
  ierr = ISCreateGeneral(comm,n_local_dofs,temp_global_dofs,PETSC_COPY_VALUES,&paris);CHKERRQ(ierr);
  ierr = VecScatterCreate(local_vec,seqis,global_vec,paris,&scatter_ctx);CHKERRQ(ierr);
  ierr = ISDestroy(&seqis);CHKERRQ(ierr);
  ierr = ISDestroy(&paris);CHKERRQ(ierr);
  /* init array */
  ierr = VecSet(global_vec,0.0);CHKERRQ(ierr);
  ierr = VecSet(local_vec,0.0);CHKERRQ(ierr);
  ierr = VecGetArray(local_vec,&array);CHKERRQ(ierr);
  if (local_dofs_mult) {
    for (i=0;i<n_local_dofs;i++) {
      array[local_dofs[i]]=(PetscScalar)local_dofs_mult[i];
    }
  } else {
    for (i=0;i<n_local_dofs;i++) {
      array[local_dofs[i]]=1.0;
    }
  }
  ierr = VecRestoreArray(local_vec,&array);CHKERRQ(ierr);
  /* scatter into global vec and get total number of global dofs */
  ierr = VecScatterBegin(scatter_ctx,local_vec,global_vec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(scatter_ctx,local_vec,global_vec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecSum(global_vec,&globalsum);CHKERRQ(ierr);
  *n_global_subset = (PetscInt)PetscRealPart(globalsum);
  /* Fill global_vec with cumulative function for global numbering */
  ierr = VecGetArray(global_vec,&array);CHKERRQ(ierr);
  ierr = VecGetLocalSize(global_vec,&s);CHKERRQ(ierr);
  nlocals = 0;
  first_index = -1;
  first_found = PETSC_FALSE;
  for (i=0;i<s;i++) {
    if (!first_found && PetscRealPart(array[i]) > 0.0) {
      first_found = PETSC_TRUE;
      first_index = i;
    }
    nlocals += (PetscInt)PetscRealPart(array[i]);
  }
  ierr = MPI_Gather(&nlocals,1,MPIU_INT,dof_sizes,1,MPIU_INT,0,comm);CHKERRQ(ierr);
  if (!rank_prec_comm) {
    dof_displs[0]=0;
    for (i=1;i<size_prec_comm;i++) {
      dof_displs[i] = dof_displs[i-1]+dof_sizes[i-1];
    }
  }
  ierr = MPI_Scatter(dof_displs,1,MPIU_INT,&nlocals,1,MPIU_INT,0,comm);CHKERRQ(ierr);
  if (first_found) {
    array[first_index] += (PetscScalar)nlocals;
    old_index = first_index;
    for (i=first_index+1;i<s;i++) {
      if (PetscRealPart(array[i]) > 0.0) {
        array[i] += array[old_index];
        old_index = i;
      }
    }
  }
  ierr = VecRestoreArray(global_vec,&array);CHKERRQ(ierr);
  ierr = VecSet(local_vec,0.0);CHKERRQ(ierr);
  ierr = VecScatterBegin(scatter_ctx,global_vec,local_vec,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd  (scatter_ctx,global_vec,local_vec,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  /* get global ordering of local dofs */
  ierr = VecGetArray(local_vec,&array);CHKERRQ(ierr);
  if (local_dofs_mult) {
    for (i=0;i<n_local_dofs;i++) {
      temp_global_dofs[i] = (PetscInt)PetscRealPart(array[local_dofs[i]])-local_dofs_mult[i];
    }
  } else {
    for (i=0;i<n_local_dofs;i++) {
      temp_global_dofs[i] = (PetscInt)PetscRealPart(array[local_dofs[i]])-1;
    }
  }
  ierr = VecRestoreArray(local_vec,&array);CHKERRQ(ierr);
  /* free workspace */
  ierr = VecScatterDestroy(&scatter_ctx);CHKERRQ(ierr);
  ierr = VecDestroy(&local_vec);CHKERRQ(ierr);
  ierr = VecDestroy(&global_vec);CHKERRQ(ierr);
  ierr = PetscFree(dof_sizes);CHKERRQ(ierr);
  ierr = PetscFree(dof_displs);CHKERRQ(ierr);
  /* return pointer to global ordering of local dofs */
  *global_numbering_subset = temp_global_dofs;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCBDDCOrthonormalizeVecs"
PetscErrorCode PCBDDCOrthonormalizeVecs(PetscInt n, Vec vecs[])
{
  PetscInt       i,j;
  PetscScalar    *alphas;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* this implements stabilized Gram-Schmidt */
  ierr = PetscMalloc(n*sizeof(PetscScalar),&alphas);CHKERRQ(ierr);
  for (i=0;i<n;i++) {
    ierr = VecNormalize(vecs[i],NULL);CHKERRQ(ierr);
    if (i<n) { ierr = VecMDot(vecs[i],n-i-1,&vecs[i+1],&alphas[i+1]);CHKERRQ(ierr); }
    for (j=i+1;j<n;j++) { ierr = VecAXPY(vecs[j],PetscConj(-alphas[j]),vecs[i]);CHKERRQ(ierr); }
  }
  ierr = PetscFree(alphas);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* BDDC requires metis 5.0.1 for multilevel */
#if defined(PETSC_HAVE_METIS)
#include "metis.h"
#define MetisInt    idx_t
#define MetisScalar real_t
#endif

#undef __FUNCT__
#define __FUNCT__ "PCBDDCSetUpCoarseEnvironment"
PetscErrorCode PCBDDCSetUpCoarseEnvironment(PC pc,PetscScalar* coarse_submat_vals)
{
  Mat_IS    *matis    = (Mat_IS*)pc->pmat->data;
  PC_BDDC   *pcbddc   = (PC_BDDC*)pc->data;
  PC_IS     *pcis     = (PC_IS*)pc->data;
  MPI_Comm  prec_comm;
  MPI_Comm  coarse_comm;

  PetscInt  coarse_size;

  MatNullSpace CoarseNullSpace;

  /* common to all choiches */
  PetscScalar *temp_coarse_mat_vals;
  PetscScalar *ins_coarse_mat_vals;
  PetscInt    *ins_local_primal_indices;
  PetscMPIInt *localsizes2,*localdispl2;
  PetscMPIInt size_prec_comm;
  PetscMPIInt rank_prec_comm;
  PetscMPIInt active_rank=MPI_PROC_NULL;
  PetscMPIInt master_proc=0;
  PetscInt    ins_local_primal_size;
  /* specific to MULTILEVEL_BDDC */
  PetscMPIInt *ranks_recv=0;
  PetscMPIInt count_recv=0;
  PetscMPIInt rank_coarse_proc_send_to=-1;
  PetscMPIInt coarse_color = MPI_UNDEFINED;
  ISLocalToGlobalMapping coarse_ISLG;
  /* some other variables */
  PetscErrorCode ierr;
  MatType coarse_mat_type;
  PCType  coarse_pc_type;
  KSPType coarse_ksp_type;
  PC pc_temp;
  PetscInt i,j,k;
  PetscInt max_it_coarse_ksp=1;  /* don't increase this value */
  /* verbose output viewer */
  PetscViewer viewer=pcbddc->dbg_viewer;
  PetscInt    dbg_flag=pcbddc->dbg_flag;

  PetscInt      offset,offset2;
  PetscMPIInt   im_active,active_procs;
  PetscInt      *dnz,*onz;

  PetscBool     setsym,issym=PETSC_FALSE;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)pc,&prec_comm);CHKERRQ(ierr);
  ins_local_primal_indices = 0;
  ins_coarse_mat_vals      = 0;
  localsizes2              = 0;
  localdispl2              = 0;
  temp_coarse_mat_vals     = 0;
  coarse_ISLG              = 0;

  ierr = MPI_Comm_size(prec_comm,&size_prec_comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(prec_comm,&rank_prec_comm);CHKERRQ(ierr);
  ierr = MatIsSymmetricKnown(pc->pmat,&setsym,&issym);CHKERRQ(ierr);

  /* Assign global numbering to coarse dofs */
  {
    PetscInt     *auxlocal_primal,*aux_idx;
    PetscMPIInt  mpi_local_primal_size;
    PetscScalar  coarsesum,*array;

    mpi_local_primal_size = (PetscMPIInt)pcbddc->local_primal_size;

    /* Construct needed data structures for message passing */
    j = 0;
    if (rank_prec_comm == 0 || pcbddc->coarse_problem_type == REPLICATED_BDDC || pcbddc->coarse_problem_type == MULTILEVEL_BDDC) {
      j = size_prec_comm;
    }
    ierr = PetscMalloc(j*sizeof(PetscMPIInt),&pcbddc->local_primal_sizes);CHKERRQ(ierr);
    ierr = PetscMalloc(j*sizeof(PetscMPIInt),&pcbddc->local_primal_displacements);CHKERRQ(ierr);
    /* Gather local_primal_size information for all processes  */
    if (pcbddc->coarse_problem_type == REPLICATED_BDDC || pcbddc->coarse_problem_type == MULTILEVEL_BDDC) {
      ierr = MPI_Allgather(&mpi_local_primal_size,1,MPIU_INT,&pcbddc->local_primal_sizes[0],1,MPIU_INT,prec_comm);CHKERRQ(ierr);
    } else {
      ierr = MPI_Gather(&mpi_local_primal_size,1,MPIU_INT,&pcbddc->local_primal_sizes[0],1,MPIU_INT,0,prec_comm);CHKERRQ(ierr);
    }
    pcbddc->replicated_primal_size = 0;
    for (i=0; i<j; i++) {
      pcbddc->local_primal_displacements[i] = pcbddc->replicated_primal_size ;
      pcbddc->replicated_primal_size += pcbddc->local_primal_sizes[i];
    }

    /* This code fragment assumes that the number of local constraints per connected component
       is not greater than the number of nodes defined for the connected component
       (otherwise we will surely have linear dependence between constraints and thus a singular coarse problem) */
    ierr = PetscMalloc(pcbddc->local_primal_size*sizeof(PetscInt),&auxlocal_primal);CHKERRQ(ierr);
    ierr = PCBDDCGetPrimalVerticesLocalIdx(pc,&i,&aux_idx);CHKERRQ(ierr);
    ierr = PetscMemcpy(auxlocal_primal,aux_idx,i*sizeof(PetscInt));CHKERRQ(ierr);
    ierr = PetscFree(aux_idx);CHKERRQ(ierr);
    ierr = PCBDDCGetPrimalConstraintsLocalIdx(pc,&j,&aux_idx);CHKERRQ(ierr);
    ierr = PetscMemcpy(&auxlocal_primal[i],aux_idx,j*sizeof(PetscInt));CHKERRQ(ierr);
    ierr = PetscFree(aux_idx);CHKERRQ(ierr);
    /* Compute number of coarse dofs */
    ierr = PCBDDCSubsetNumbering(prec_comm,matis->mapping,pcbddc->local_primal_size,auxlocal_primal,NULL,&coarse_size,&pcbddc->local_primal_indices);CHKERRQ(ierr);

    if (dbg_flag) {
      ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"--------------------------------------------------\n");CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"Check coarse indices\n");CHKERRQ(ierr);
      ierr = VecSet(pcis->vec1_N,0.0);CHKERRQ(ierr);
      ierr = VecGetArray(pcis->vec1_N,&array);CHKERRQ(ierr);
      for (i=0;i<pcbddc->local_primal_size;i++) array[auxlocal_primal[i]]=1.0;
      ierr = VecRestoreArray(pcis->vec1_N,&array);CHKERRQ(ierr);
      ierr = VecSet(pcis->vec1_global,0.0);CHKERRQ(ierr);
      ierr = VecScatterBegin(matis->ctx,pcis->vec1_N,pcis->vec1_global,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
      ierr = VecScatterEnd  (matis->ctx,pcis->vec1_N,pcis->vec1_global,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
      ierr = VecScatterBegin(matis->ctx,pcis->vec1_global,pcis->vec1_N,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecScatterEnd  (matis->ctx,pcis->vec1_global,pcis->vec1_N,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecGetArray(pcis->vec1_N,&array);CHKERRQ(ierr);
      for (i=0;i<pcis->n;i++) {
        if (array[i] == 1.0) {
          ierr = ISLocalToGlobalMappingApply(matis->mapping,1,&i,&j);CHKERRQ(ierr);
          ierr = PetscViewerASCIISynchronizedPrintf(viewer,"Subdomain %04d: WRONG COARSE INDEX %d (local %d)\n",PetscGlobalRank,j,i);CHKERRQ(ierr);
        }
      }
      ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
      for (i=0;i<pcis->n;i++) {
        if (PetscRealPart(array[i]) > 0.0) array[i] = 1.0/PetscRealPart(array[i]);
      }
      ierr = VecRestoreArray(pcis->vec1_N,&array);CHKERRQ(ierr);
      ierr = VecSet(pcis->vec1_global,0.0);CHKERRQ(ierr);
      ierr = VecScatterBegin(matis->ctx,pcis->vec1_N,pcis->vec1_global,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
      ierr = VecScatterEnd  (matis->ctx,pcis->vec1_N,pcis->vec1_global,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
      ierr = VecSum(pcis->vec1_global,&coarsesum);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"Size of coarse problem SHOULD be %lf\n",coarsesum);CHKERRQ(ierr);
      ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
    }
    ierr = PetscFree(auxlocal_primal);CHKERRQ(ierr);
  }

  if (dbg_flag) {
    ierr = PetscViewerASCIIPrintf(viewer,"Size of coarse problem is %d\n",coarse_size);CHKERRQ(ierr);
    if (dbg_flag > 1) {
      ierr = PetscViewerASCIIPrintf(viewer,"Distribution of local primal indices\n");CHKERRQ(ierr);
      ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"Subdomain %04d\n",PetscGlobalRank);CHKERRQ(ierr);
      for (i=0;i<pcbddc->local_primal_size;i++) {
        ierr = PetscViewerASCIISynchronizedPrintf(viewer,"local_primal_indices[%d]=%d \n",i,pcbddc->local_primal_indices[i]);
      }
      ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
    }
  }

  im_active = 0;
  if (pcis->n) im_active = 1;
  ierr = MPI_Allreduce(&im_active,&active_procs,1,MPIU_INT,MPI_SUM,prec_comm);CHKERRQ(ierr);

  /* adapt coarse problem type */
#if defined(PETSC_HAVE_METIS)
  if (pcbddc->coarse_problem_type == MULTILEVEL_BDDC) {
    if (pcbddc->current_level < pcbddc->max_levels) {
      if ( (active_procs/pcbddc->coarsening_ratio) < 2 ) {
        if (dbg_flag) {
          ierr = PetscViewerASCIIPrintf(viewer,"Not enough active processes on level %d (active %d,ratio %d). Parallel direct solve for coarse problem\n",pcbddc->current_level,active_procs,pcbddc->coarsening_ratio);CHKERRQ(ierr);
         ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
        }
        pcbddc->coarse_problem_type = PARALLEL_BDDC;
      }
    } else {
      if (dbg_flag) {
        ierr = PetscViewerASCIIPrintf(viewer,"Max number of levels reached. Using parallel direct solve for coarse problem\n",pcbddc->max_levels,active_procs,pcbddc->coarsening_ratio);CHKERRQ(ierr);
        ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
      }
      pcbddc->coarse_problem_type = PARALLEL_BDDC;
    }
  }
#else
  pcbddc->coarse_problem_type = PARALLEL_BDDC;
#endif

  switch(pcbddc->coarse_problem_type){

    case(MULTILEVEL_BDDC):   /* we define a coarse mesh where subdomains are elements */
    {
#if defined(PETSC_HAVE_METIS)
      /* we need additional variables */
      MetisInt    n_subdomains,n_parts,objval,ncon,faces_nvtxs;
      MetisInt    *metis_coarse_subdivision;
      MetisInt    options[METIS_NOPTIONS];
      PetscMPIInt size_coarse_comm,rank_coarse_comm;
      PetscMPIInt procs_jumps_coarse_comm;
      PetscMPIInt *coarse_subdivision;
      PetscMPIInt *total_count_recv;
      PetscMPIInt *total_ranks_recv;
      PetscMPIInt *displacements_recv;
      PetscMPIInt *my_faces_connectivity;
      PetscMPIInt *petsc_faces_adjncy;
      MetisInt    *faces_adjncy;
      MetisInt    *faces_xadj;
      PetscMPIInt *number_of_faces;
      PetscMPIInt *faces_displacements;
      PetscInt    *array_int;
      PetscMPIInt my_faces=0;
      PetscMPIInt total_faces=0;
      PetscInt    ranks_stretching_ratio;

      /* define some quantities */
      pcbddc->coarse_communications_type = SCATTERS_BDDC;
      coarse_mat_type = MATIS;
      coarse_pc_type  = PCBDDC;
      coarse_ksp_type = KSPRICHARDSON;

      /* details of coarse decomposition */
      n_subdomains = active_procs;
      n_parts      = n_subdomains/pcbddc->coarsening_ratio;
      ranks_stretching_ratio = size_prec_comm/active_procs;
      procs_jumps_coarse_comm = pcbddc->coarsening_ratio*ranks_stretching_ratio;

#if 0
      PetscMPIInt *old_ranks;
      PetscInt    *new_ranks,*jj,*ii;
      MatPartitioning mat_part;
      IS coarse_new_decomposition,is_numbering;
      PetscViewer viewer_test;
      MPI_Comm    test_coarse_comm;
      PetscMPIInt test_coarse_color;
      Mat         mat_adj;
      /* Create new communicator for coarse problem splitting the old one */
      /* procs with coarse_color = MPI_UNDEFINED will have coarse_comm = MPI_COMM_NULL (from mpi standards)
         key = rank_prec_comm -> keep same ordering of ranks from the old to the new communicator */
      test_coarse_color = ( im_active ? 0 : MPI_UNDEFINED );
      test_coarse_comm = MPI_COMM_NULL;
      ierr = MPI_Comm_split(prec_comm,test_coarse_color,rank_prec_comm,&test_coarse_comm);CHKERRQ(ierr);
      if (im_active) {
        ierr = PetscMalloc(n_subdomains*sizeof(PetscMPIInt),&old_ranks);
        ierr = PetscMalloc(size_prec_comm*sizeof(PetscInt),&new_ranks);
        ierr = MPI_Comm_rank(test_coarse_comm,&rank_coarse_comm);CHKERRQ(ierr);
        ierr = MPI_Comm_size(test_coarse_comm,&j);CHKERRQ(ierr);
        ierr = MPI_Allgather(&rank_prec_comm,1,MPIU_INT,old_ranks,1,MPIU_INT,test_coarse_comm);CHKERRQ(ierr);
        for (i=0; i<size_prec_comm; i++) new_ranks[i] = -1;
        for (i=0; i<n_subdomains; i++) new_ranks[old_ranks[i]] = i;
        ierr = PetscViewerASCIIOpen(test_coarse_comm,"test_mat_part.out",&viewer_test);CHKERRQ(ierr);
        k = pcis->n_neigh-1;
        ierr = PetscMalloc(2*sizeof(PetscInt),&ii);
        ii[0]=0;
        ii[1]=k;
        ierr = PetscMalloc(k*sizeof(PetscInt),&jj);
        for (i=0; i<k; i++) jj[i]=new_ranks[pcis->neigh[i+1]];
        ierr = PetscSortInt(k,jj);CHKERRQ(ierr);
        ierr = MatCreateMPIAdj(test_coarse_comm,1,n_subdomains,ii,jj,NULL,&mat_adj);CHKERRQ(ierr);
        ierr = MatView(mat_adj,viewer_test);CHKERRQ(ierr);
        ierr = MatPartitioningCreate(test_coarse_comm,&mat_part);CHKERRQ(ierr);
        ierr = MatPartitioningSetAdjacency(mat_part,mat_adj);CHKERRQ(ierr);
        ierr = MatPartitioningSetFromOptions(mat_part);CHKERRQ(ierr);
        printf("Setting Nparts %d\n",n_parts);
        ierr = MatPartitioningSetNParts(mat_part,n_parts);CHKERRQ(ierr);
        ierr = MatPartitioningView(mat_part,viewer_test);CHKERRQ(ierr);
        ierr = MatPartitioningApply(mat_part,&coarse_new_decomposition);CHKERRQ(ierr);
        ierr = ISView(coarse_new_decomposition,viewer_test);CHKERRQ(ierr);
        ierr = ISPartitioningToNumbering(coarse_new_decomposition,&is_numbering);CHKERRQ(ierr);
        ierr = ISView(is_numbering,viewer_test);CHKERRQ(ierr);
        ierr = PetscViewerDestroy(&viewer_test);CHKERRQ(ierr);
        ierr = ISDestroy(&coarse_new_decomposition);CHKERRQ(ierr);
        ierr = ISDestroy(&is_numbering);CHKERRQ(ierr);
        ierr = MatPartitioningDestroy(&mat_part);CHKERRQ(ierr);
        ierr = PetscFree(old_ranks);CHKERRQ(ierr);
        ierr = PetscFree(new_ranks);CHKERRQ(ierr);
        ierr = MPI_Comm_free(&test_coarse_comm);CHKERRQ(ierr);
      }
#endif

      /* build CSR graph of subdomains' connectivity */
      ierr = PetscMalloc (pcis->n*sizeof(PetscInt),&array_int);CHKERRQ(ierr);
      ierr = PetscMemzero(array_int,pcis->n*sizeof(PetscInt));CHKERRQ(ierr);
      for (i=1;i<pcis->n_neigh;i++){/* i=1 so I don't count myself -> faces nodes counts to 1 */
        for (j=0;j<pcis->n_shared[i];j++){
          array_int[ pcis->shared[i][j] ]+=1;
        }
      }
      for (i=1;i<pcis->n_neigh;i++){
        for (j=0;j<pcis->n_shared[i];j++){
          if (array_int[ pcis->shared[i][j] ] > 0 ){
            my_faces++;
            break;
          }
        }
      }

      ierr = MPI_Reduce(&my_faces,&total_faces,1,MPIU_INT,MPI_SUM,master_proc,prec_comm);CHKERRQ(ierr);
      ierr = PetscMalloc (my_faces*sizeof(PetscInt),&my_faces_connectivity);CHKERRQ(ierr);
      my_faces=0;
      for (i=1;i<pcis->n_neigh;i++){
        for (j=0;j<pcis->n_shared[i];j++){
          if (array_int[ pcis->shared[i][j] ] > 0 ){
            my_faces_connectivity[my_faces]=pcis->neigh[i];
            my_faces++;
            break;
          }
        }
      }
      if (rank_prec_comm == master_proc) {
        ierr = PetscMalloc (total_faces*sizeof(PetscMPIInt),&petsc_faces_adjncy);CHKERRQ(ierr);
        ierr = PetscMalloc (size_prec_comm*sizeof(PetscMPIInt),&number_of_faces);CHKERRQ(ierr);
        ierr = PetscMalloc (total_faces*sizeof(MetisInt),&faces_adjncy);CHKERRQ(ierr);
        ierr = PetscMalloc ((n_subdomains+1)*sizeof(MetisInt),&faces_xadj);CHKERRQ(ierr);
        ierr = PetscMalloc ((size_prec_comm+1)*sizeof(PetscMPIInt),&faces_displacements);CHKERRQ(ierr);
      }
      ierr = MPI_Gather(&my_faces,1,MPIU_INT,&number_of_faces[0],1,MPIU_INT,master_proc,prec_comm);CHKERRQ(ierr);
      if (rank_prec_comm == master_proc) {
        faces_xadj[0]=0;
        faces_displacements[0]=0;
        j=0;
        for (i=1;i<size_prec_comm+1;i++) {
          faces_displacements[i]=faces_displacements[i-1]+number_of_faces[i-1];
          if (number_of_faces[i-1]) {
            j++;
            faces_xadj[j]=faces_xadj[j-1]+number_of_faces[i-1];
          }
        }
      }
      ierr = MPI_Gatherv(&my_faces_connectivity[0],my_faces,MPIU_INT,&petsc_faces_adjncy[0],number_of_faces,faces_displacements,MPIU_INT,master_proc,prec_comm);CHKERRQ(ierr);
      ierr = PetscFree(my_faces_connectivity);CHKERRQ(ierr);
      ierr = PetscFree(array_int);CHKERRQ(ierr);
      if (rank_prec_comm == master_proc) {
        for (i=0;i<total_faces;i++) faces_adjncy[i]=(MetisInt)(petsc_faces_adjncy[i]/ranks_stretching_ratio); /* cast to MetisInt */
        ierr = PetscFree(faces_displacements);CHKERRQ(ierr);
        ierr = PetscFree(number_of_faces);CHKERRQ(ierr);
        ierr = PetscFree(petsc_faces_adjncy);CHKERRQ(ierr);
      }

      if ( rank_prec_comm == master_proc ) {

        PetscInt heuristic_for_metis=3;

        ncon=1;
        faces_nvtxs=n_subdomains;
        /* partition graoh induced by face connectivity */
        ierr = PetscMalloc (n_subdomains*sizeof(MetisInt),&metis_coarse_subdivision);CHKERRQ(ierr);
        ierr = METIS_SetDefaultOptions(options);
        /* we need a contiguous partition of the coarse mesh */
        options[METIS_OPTION_CONTIG]=1;
        options[METIS_OPTION_NITER]=30;
        if (pcbddc->coarsening_ratio > 1) {
          if (n_subdomains>n_parts*heuristic_for_metis) {
            options[METIS_OPTION_IPTYPE]=METIS_IPTYPE_EDGE;
            options[METIS_OPTION_OBJTYPE]=METIS_OBJTYPE_CUT;
            ierr = METIS_PartGraphKway(&faces_nvtxs,&ncon,faces_xadj,faces_adjncy,NULL,NULL,NULL,&n_parts,NULL,NULL,options,&objval,metis_coarse_subdivision);
            if (ierr != METIS_OK) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in METIS_PartGraphKway (metis error code %D) called from PCBDDCSetUpCoarseEnvironment\n",ierr);
          } else {
            ierr = METIS_PartGraphRecursive(&faces_nvtxs,&ncon,faces_xadj,faces_adjncy,NULL,NULL,NULL,&n_parts,NULL,NULL,options,&objval,metis_coarse_subdivision);
            if (ierr != METIS_OK) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in METIS_PartGraphRecursive (metis error code %D) called from PCBDDCSetUpCoarseEnvironment\n",ierr);
          }
        } else {
          for (i=0;i<n_subdomains;i++) metis_coarse_subdivision[i]=i;
        }
        ierr = PetscFree(faces_xadj);CHKERRQ(ierr);
        ierr = PetscFree(faces_adjncy);CHKERRQ(ierr);
        ierr = PetscMalloc(size_prec_comm*sizeof(PetscMPIInt),&coarse_subdivision);CHKERRQ(ierr);

        /* copy/cast values avoiding possible type conflicts between PETSc, MPI and METIS */
        for (i=0;i<size_prec_comm;i++) coarse_subdivision[i]=MPI_PROC_NULL;
        for (i=0;i<n_subdomains;i++) coarse_subdivision[ranks_stretching_ratio*i]=(PetscInt)(metis_coarse_subdivision[i]);
        ierr = PetscFree(metis_coarse_subdivision);CHKERRQ(ierr);
      }

      /* Create new communicator for coarse problem splitting the old one */
      if ( !(rank_prec_comm%procs_jumps_coarse_comm) && rank_prec_comm < procs_jumps_coarse_comm*n_parts ){
        coarse_color=0;              /* for communicator splitting */
        active_rank=rank_prec_comm;  /* for insertion of matrix values */
      }
      /* procs with coarse_color = MPI_UNDEFINED will have coarse_comm = MPI_COMM_NULL (from mpi standards)
         key = rank_prec_comm -> keep same ordering of ranks from the old to the new communicator */
      ierr = MPI_Comm_split(prec_comm,coarse_color,rank_prec_comm,&coarse_comm);CHKERRQ(ierr);

      if ( coarse_color == 0 ) {
        ierr = MPI_Comm_size(coarse_comm,&size_coarse_comm);CHKERRQ(ierr);
        ierr = MPI_Comm_rank(coarse_comm,&rank_coarse_comm);CHKERRQ(ierr);
      } else {
        rank_coarse_comm = MPI_PROC_NULL;
      }

      /* master proc take care of arranging and distributing coarse information */
      if (rank_coarse_comm == master_proc) {
        ierr = PetscMalloc (size_coarse_comm*sizeof(PetscMPIInt),&displacements_recv);CHKERRQ(ierr);
        ierr = PetscMalloc (size_coarse_comm*sizeof(PetscMPIInt),&total_count_recv);CHKERRQ(ierr);
        ierr = PetscMalloc (n_subdomains*sizeof(PetscMPIInt),&total_ranks_recv);CHKERRQ(ierr);
        /* some initializations */
        displacements_recv[0]=0;
        ierr = PetscMemzero(total_count_recv,size_coarse_comm*sizeof(PetscMPIInt));CHKERRQ(ierr);
        /* count from how many processes the j-th process of the coarse decomposition will receive data */
        for (j=0;j<size_coarse_comm;j++) {
          for (i=0;i<size_prec_comm;i++) {
          if (coarse_subdivision[i]==j) total_count_recv[j]++;
          }
        }
        /* displacements needed for scatterv of total_ranks_recv */
      for (i=1; i<size_coarse_comm; i++) displacements_recv[i]=displacements_recv[i-1]+total_count_recv[i-1];

        /* Now fill properly total_ranks_recv -> each coarse process will receive the ranks (in prec_comm communicator) of its friend (sending) processes */
        ierr = PetscMemzero(total_count_recv,size_coarse_comm*sizeof(PetscMPIInt));CHKERRQ(ierr);
        for (j=0;j<size_coarse_comm;j++) {
          for (i=0;i<size_prec_comm;i++) {
            if (coarse_subdivision[i]==j) {
              total_ranks_recv[displacements_recv[j]+total_count_recv[j]]=i;
              total_count_recv[j]+=1;
            }
          }
        }
        /*for (j=0;j<size_coarse_comm;j++) {
          printf("process %d in new rank will receive from %d processes (original ranks follows)\n",j,total_count_recv[j]);
          for (i=0;i<total_count_recv[j];i++) {
            printf("%d ",total_ranks_recv[displacements_recv[j]+i]);
          }
          printf("\n");
        }*/

        /* identify new decomposition in terms of ranks in the old communicator */
        for (i=0;i<n_subdomains;i++) {
          coarse_subdivision[ranks_stretching_ratio*i]=coarse_subdivision[ranks_stretching_ratio*i]*procs_jumps_coarse_comm;
        }
        /*printf("coarse_subdivision in old end new ranks\n");
        for (i=0;i<size_prec_comm;i++)
          if (coarse_subdivision[i]!=MPI_PROC_NULL) {
            printf("%d=(%d %d), ",i,coarse_subdivision[i],coarse_subdivision[i]/procs_jumps_coarse_comm);
          } else {
            printf("%d=(%d %d), ",i,coarse_subdivision[i],coarse_subdivision[i]);
          }
        printf("\n");*/
      }

      /* Scatter new decomposition for send details */
      ierr = MPI_Scatter(&coarse_subdivision[0],1,MPIU_INT,&rank_coarse_proc_send_to,1,MPIU_INT,master_proc,prec_comm);CHKERRQ(ierr);
      /* Scatter receiving details to members of coarse decomposition */
      if ( coarse_color == 0) {
        ierr = MPI_Scatter(&total_count_recv[0],1,MPIU_INT,&count_recv,1,MPIU_INT,master_proc,coarse_comm);CHKERRQ(ierr);
        ierr = PetscMalloc (count_recv*sizeof(PetscMPIInt),&ranks_recv);CHKERRQ(ierr);
        ierr = MPI_Scatterv(&total_ranks_recv[0],total_count_recv,displacements_recv,MPIU_INT,&ranks_recv[0],count_recv,MPIU_INT,master_proc,coarse_comm);CHKERRQ(ierr);
      }

      /*printf("I will send my matrix data to proc  %d\n",rank_coarse_proc_send_to);
      if (coarse_color == 0) {
        printf("I will receive some matrix data from %d processes (ranks follows)\n",count_recv);
        for (i=0;i<count_recv;i++)
          printf("%d ",ranks_recv[i]);
        printf("\n");
      }*/

      if (rank_prec_comm == master_proc) {
        ierr = PetscFree(coarse_subdivision);CHKERRQ(ierr);
        ierr = PetscFree(total_count_recv);CHKERRQ(ierr);
        ierr = PetscFree(total_ranks_recv);CHKERRQ(ierr);
        ierr = PetscFree(displacements_recv);CHKERRQ(ierr);
      }
#endif
      break;
    }

    case(REPLICATED_BDDC):

      pcbddc->coarse_communications_type = GATHERS_BDDC;
      coarse_mat_type = MATSEQAIJ;
      coarse_pc_type  = PCLU;
      coarse_ksp_type  = KSPPREONLY;
      coarse_comm = PETSC_COMM_SELF;
      active_rank = rank_prec_comm;
      break;

    case(PARALLEL_BDDC):

      pcbddc->coarse_communications_type = SCATTERS_BDDC;
      coarse_mat_type = MATAIJ;
      coarse_pc_type  = PCREDUNDANT;
      coarse_ksp_type  = KSPPREONLY;
      coarse_comm = prec_comm;
      active_rank = rank_prec_comm;
      break;

    case(SEQUENTIAL_BDDC):
      pcbddc->coarse_communications_type = GATHERS_BDDC;
      coarse_mat_type = MATAIJ;
      coarse_pc_type = PCLU;
      coarse_ksp_type  = KSPPREONLY;
      coarse_comm = PETSC_COMM_SELF;
      active_rank = master_proc;
      break;
  }

  switch(pcbddc->coarse_communications_type){

    case(SCATTERS_BDDC):
      {
        if (pcbddc->coarse_problem_type==MULTILEVEL_BDDC) {

          IS coarse_IS;

          if(pcbddc->coarsening_ratio == 1) {
            ins_local_primal_size = pcbddc->local_primal_size;
            ins_local_primal_indices = pcbddc->local_primal_indices;
            if (coarse_color == 0) { ierr = PetscFree(ranks_recv);CHKERRQ(ierr); }
            /* nonzeros */
            ierr = PetscMalloc(ins_local_primal_size*sizeof(PetscInt),&dnz);CHKERRQ(ierr);
            ierr = PetscMemzero(dnz,ins_local_primal_size*sizeof(PetscInt));CHKERRQ(ierr);
            for (i=0;i<ins_local_primal_size;i++) {
              dnz[i] = ins_local_primal_size;
            }
          } else {
            PetscMPIInt send_size;
            PetscMPIInt *send_buffer;
            PetscInt    *aux_ins_indices;
            PetscInt    ii,jj;
            MPI_Request *requests;

            ierr = PetscMalloc(count_recv*sizeof(PetscMPIInt),&localdispl2);CHKERRQ(ierr);
            /* reusing pcbddc->local_primal_displacements and pcbddc->replicated_primal_size */
            ierr = PetscFree(pcbddc->local_primal_displacements);CHKERRQ(ierr);
            ierr = PetscMalloc((count_recv+1)*sizeof(PetscMPIInt),&pcbddc->local_primal_displacements);CHKERRQ(ierr);
            pcbddc->replicated_primal_size = count_recv;
            j = 0;
            for (i=0;i<count_recv;i++) {
              pcbddc->local_primal_displacements[i] = j;
              j += pcbddc->local_primal_sizes[ranks_recv[i]];
            }
            pcbddc->local_primal_displacements[count_recv] = j;
            ierr = PetscMalloc(j*sizeof(PetscMPIInt),&pcbddc->replicated_local_primal_indices);CHKERRQ(ierr);
            /* allocate auxiliary space */
            ierr = PetscMalloc(count_recv*sizeof(PetscMPIInt),&localsizes2);CHKERRQ(ierr);
            ierr = PetscMalloc(coarse_size*sizeof(PetscInt),&aux_ins_indices);CHKERRQ(ierr);
            ierr = PetscMemzero(aux_ins_indices,coarse_size*sizeof(PetscInt));CHKERRQ(ierr);
            /* allocate stuffs for message massing */
            ierr = PetscMalloc((count_recv+1)*sizeof(MPI_Request),&requests);CHKERRQ(ierr);
            for (i=0;i<count_recv+1;i++) { requests[i]=MPI_REQUEST_NULL; }
            /* send indices to be inserted */
            for (i=0;i<count_recv;i++) {
              send_size = pcbddc->local_primal_sizes[ranks_recv[i]];
              ierr = MPI_Irecv(&pcbddc->replicated_local_primal_indices[pcbddc->local_primal_displacements[i]],send_size,MPIU_INT,ranks_recv[i],999,prec_comm,&requests[i]);CHKERRQ(ierr);
            }
            if (rank_coarse_proc_send_to != MPI_PROC_NULL ) {
              send_size = pcbddc->local_primal_size;
              ierr = PetscMalloc(send_size*sizeof(PetscMPIInt),&send_buffer);CHKERRQ(ierr);
              for (i=0;i<send_size;i++) {
                send_buffer[i]=(PetscMPIInt)pcbddc->local_primal_indices[i];
              }
              ierr = MPI_Isend(send_buffer,send_size,MPIU_INT,rank_coarse_proc_send_to,999,prec_comm,&requests[count_recv]);CHKERRQ(ierr);
            }
            ierr = MPI_Waitall(count_recv+1,requests,MPI_STATUSES_IGNORE);CHKERRQ(ierr);
            if (rank_coarse_proc_send_to != MPI_PROC_NULL ) {
              ierr = PetscFree(send_buffer);CHKERRQ(ierr);
            }
            j = 0;
            for (i=0;i<count_recv;i++) {
              ii = pcbddc->local_primal_displacements[i+1]-pcbddc->local_primal_displacements[i];
              localsizes2[i] = ii*ii;
              localdispl2[i] = j;
              j += localsizes2[i];
              jj = pcbddc->local_primal_displacements[i];
              /* it counts the coarse subdomains sharing the coarse node */
              for (k=0;k<ii;k++) {
                aux_ins_indices[pcbddc->replicated_local_primal_indices[jj+k]] += 1;
              }
            }
            /* temp_coarse_mat_vals used to store matrix values to be received */
            ierr = PetscMalloc(j*sizeof(PetscScalar),&temp_coarse_mat_vals);CHKERRQ(ierr);
            /* evaluate how many values I will insert in coarse mat */
            ins_local_primal_size = 0;
            for (i=0;i<coarse_size;i++) {
              if (aux_ins_indices[i]) {
                ins_local_primal_size++;
              }
            }
            /* evaluate indices I will insert in coarse mat */
            ierr = PetscMalloc(ins_local_primal_size*sizeof(PetscInt),&ins_local_primal_indices);CHKERRQ(ierr);
            j = 0;
            for(i=0;i<coarse_size;i++) {
              if(aux_ins_indices[i]) {
                ins_local_primal_indices[j] = i;
                j++;
              }
            }
            /* processes partecipating in coarse problem receive matrix data from their friends */
            for (i=0;i<count_recv;i++) {
              ierr = MPI_Irecv(&temp_coarse_mat_vals[localdispl2[i]],localsizes2[i],MPIU_SCALAR,ranks_recv[i],666,prec_comm,&requests[i]);CHKERRQ(ierr);
            }
            if (rank_coarse_proc_send_to != MPI_PROC_NULL ) {
              send_size = pcbddc->local_primal_size*pcbddc->local_primal_size;
              ierr = MPI_Isend(&coarse_submat_vals[0],send_size,MPIU_SCALAR,rank_coarse_proc_send_to,666,prec_comm,&requests[count_recv]);CHKERRQ(ierr);
            }
            ierr = MPI_Waitall(count_recv+1,requests,MPI_STATUSES_IGNORE);CHKERRQ(ierr);
            /* nonzeros */
            ierr = PetscMalloc(ins_local_primal_size*sizeof(PetscInt),&dnz);CHKERRQ(ierr);
            ierr = PetscMemzero(dnz,ins_local_primal_size*sizeof(PetscInt));CHKERRQ(ierr);
            /* use aux_ins_indices to realize a global to local mapping */
            j=0;
            for(i=0;i<coarse_size;i++){
              if(aux_ins_indices[i]==0){
                aux_ins_indices[i]=-1;
              } else {
                aux_ins_indices[i]=j;
                j++;
              }
            }
            for (i=0;i<count_recv;i++) {
              j = pcbddc->local_primal_sizes[ranks_recv[i]];
              for (k=0;k<j;k++) {
                dnz[aux_ins_indices[pcbddc->replicated_local_primal_indices[pcbddc->local_primal_displacements[i]+k]]] += j;
              }
            }
            /* check */
            for (i=0;i<ins_local_primal_size;i++) {
              if (dnz[i] > ins_local_primal_size) {
                dnz[i] = ins_local_primal_size;
              }
            }
            ierr = PetscFree(requests);CHKERRQ(ierr);
            ierr = PetscFree(aux_ins_indices);CHKERRQ(ierr);
            if (coarse_color == 0) { ierr = PetscFree(ranks_recv);CHKERRQ(ierr); }
          }
          /* create local to global mapping needed by coarse MATIS */
          if (coarse_comm != MPI_COMM_NULL ) {ierr = MPI_Comm_free(&coarse_comm);CHKERRQ(ierr);}
          coarse_comm = prec_comm;
          active_rank = rank_prec_comm;
          ierr = ISCreateGeneral(coarse_comm,ins_local_primal_size,ins_local_primal_indices,PETSC_COPY_VALUES,&coarse_IS);CHKERRQ(ierr);
          ierr = ISLocalToGlobalMappingCreateIS(coarse_IS,&coarse_ISLG);CHKERRQ(ierr);
          ierr = ISDestroy(&coarse_IS);CHKERRQ(ierr);
        } else if (pcbddc->coarse_problem_type==PARALLEL_BDDC) {
          /* arrays for values insertion */
          ins_local_primal_size = pcbddc->local_primal_size;
          ierr = PetscMalloc(ins_local_primal_size*sizeof(PetscInt),&ins_local_primal_indices);CHKERRQ(ierr);
          ierr = PetscMalloc(ins_local_primal_size*ins_local_primal_size*sizeof(PetscScalar),&ins_coarse_mat_vals);CHKERRQ(ierr);
          for (j=0;j<ins_local_primal_size;j++){
            ins_local_primal_indices[j]=pcbddc->local_primal_indices[j];
            for (i=0;i<ins_local_primal_size;i++) {
              ins_coarse_mat_vals[j*ins_local_primal_size+i]=coarse_submat_vals[j*ins_local_primal_size+i];
            }
          }
        }
        break;

    }

    case(GATHERS_BDDC):
      {

        PetscMPIInt mysize,mysize2;
        PetscMPIInt *send_buffer;

        if (rank_prec_comm==active_rank) {
          ierr = PetscMalloc ( pcbddc->replicated_primal_size*sizeof(PetscMPIInt),&pcbddc->replicated_local_primal_indices);CHKERRQ(ierr);
          ierr = PetscMalloc ( pcbddc->replicated_primal_size*sizeof(PetscScalar),&pcbddc->replicated_local_primal_values);CHKERRQ(ierr);
          ierr = PetscMalloc ( size_prec_comm*sizeof(PetscMPIInt),&localsizes2);CHKERRQ(ierr);
          ierr = PetscMalloc ( size_prec_comm*sizeof(PetscMPIInt),&localdispl2);CHKERRQ(ierr);
          /* arrays for values insertion */
      for (i=0;i<size_prec_comm;i++) localsizes2[i]=pcbddc->local_primal_sizes[i]*pcbddc->local_primal_sizes[i];
          localdispl2[0]=0;
      for (i=1;i<size_prec_comm;i++) localdispl2[i]=localsizes2[i-1]+localdispl2[i-1];
          j=0;
      for (i=0;i<size_prec_comm;i++) j+=localsizes2[i];
          ierr = PetscMalloc ( j*sizeof(PetscScalar),&temp_coarse_mat_vals);CHKERRQ(ierr);
        }

        mysize=pcbddc->local_primal_size;
        mysize2=pcbddc->local_primal_size*pcbddc->local_primal_size;
        ierr = PetscMalloc(mysize*sizeof(PetscMPIInt),&send_buffer);CHKERRQ(ierr);
    for (i=0; i<mysize; i++) send_buffer[i]=(PetscMPIInt)pcbddc->local_primal_indices[i];

        if (pcbddc->coarse_problem_type == SEQUENTIAL_BDDC){
          ierr = MPI_Gatherv(send_buffer,mysize,MPIU_INT,&pcbddc->replicated_local_primal_indices[0],pcbddc->local_primal_sizes,pcbddc->local_primal_displacements,MPIU_INT,master_proc,prec_comm);CHKERRQ(ierr);
          ierr = MPI_Gatherv(&coarse_submat_vals[0],mysize2,MPIU_SCALAR,&temp_coarse_mat_vals[0],localsizes2,localdispl2,MPIU_SCALAR,master_proc,prec_comm);CHKERRQ(ierr);
        } else {
          ierr = MPI_Allgatherv(send_buffer,mysize,MPIU_INT,&pcbddc->replicated_local_primal_indices[0],pcbddc->local_primal_sizes,pcbddc->local_primal_displacements,MPIU_INT,prec_comm);CHKERRQ(ierr);
          ierr = MPI_Allgatherv(&coarse_submat_vals[0],mysize2,MPIU_SCALAR,&temp_coarse_mat_vals[0],localsizes2,localdispl2,MPIU_SCALAR,prec_comm);CHKERRQ(ierr);
        }
        ierr = PetscFree(send_buffer);CHKERRQ(ierr);
        break;
      }/* switch on coarse problem and communications associated with finished */
  }

  /* Now create and fill up coarse matrix */
  if ( rank_prec_comm == active_rank ) {

    Mat matis_coarse_local_mat;

    if (pcbddc->coarse_problem_type != MULTILEVEL_BDDC) {
      ierr = MatCreate(coarse_comm,&pcbddc->coarse_mat);CHKERRQ(ierr);
      ierr = MatSetSizes(pcbddc->coarse_mat,PETSC_DECIDE,PETSC_DECIDE,coarse_size,coarse_size);CHKERRQ(ierr);
      ierr = MatSetType(pcbddc->coarse_mat,coarse_mat_type);CHKERRQ(ierr);
      ierr = MatSetOptionsPrefix(pcbddc->coarse_mat,"coarse_");CHKERRQ(ierr);
      ierr = MatSetFromOptions(pcbddc->coarse_mat);CHKERRQ(ierr);
      ierr = MatSetUp(pcbddc->coarse_mat);CHKERRQ(ierr);
      ierr = MatSetOption(pcbddc->coarse_mat,MAT_ROW_ORIENTED,PETSC_FALSE);CHKERRQ(ierr); /* local values stored in column major */
      ierr = MatSetOption(pcbddc->coarse_mat,MAT_IGNORE_ZERO_ENTRIES,PETSC_TRUE);CHKERRQ(ierr);
    } else {
      ierr = MatCreateIS(coarse_comm,1,PETSC_DECIDE,PETSC_DECIDE,coarse_size,coarse_size,coarse_ISLG,&pcbddc->coarse_mat);CHKERRQ(ierr);
      ierr = MatSetUp(pcbddc->coarse_mat);CHKERRQ(ierr);
      ierr = MatISGetLocalMat(pcbddc->coarse_mat,&matis_coarse_local_mat);CHKERRQ(ierr);
      ierr = MatSetOptionsPrefix(pcbddc->coarse_mat,"coarse_");CHKERRQ(ierr);
      ierr = MatSetFromOptions(pcbddc->coarse_mat);CHKERRQ(ierr);
      ierr = MatSetUp(matis_coarse_local_mat);CHKERRQ(ierr);
      ierr = MatSetOption(matis_coarse_local_mat,MAT_ROW_ORIENTED,PETSC_FALSE);CHKERRQ(ierr); /* local values stored in column major */
      ierr = MatSetOption(matis_coarse_local_mat,MAT_IGNORE_ZERO_ENTRIES,PETSC_TRUE);CHKERRQ(ierr);
    }
    /* preallocation */
    if (pcbddc->coarse_problem_type != MULTILEVEL_BDDC) {

      PetscInt lrows,lcols,bs;

      ierr = MatGetLocalSize(pcbddc->coarse_mat,&lrows,&lcols);CHKERRQ(ierr);
      ierr = MatPreallocateInitialize(coarse_comm,lrows,lcols,dnz,onz);CHKERRQ(ierr);
      ierr = MatGetBlockSize(pcbddc->coarse_mat,&bs);CHKERRQ(ierr);

      if (pcbddc->coarse_problem_type == PARALLEL_BDDC) {

        Vec         vec_dnz,vec_onz;
        PetscScalar *my_dnz,*my_onz,*array;
        PetscInt    *mat_ranges,*row_ownership;
        PetscInt    coarse_index_row,coarse_index_col,owner;

        ierr = VecCreate(prec_comm,&vec_dnz);CHKERRQ(ierr);
        ierr = VecSetBlockSize(vec_dnz,bs);CHKERRQ(ierr);
        ierr = VecSetSizes(vec_dnz,PETSC_DECIDE,coarse_size);CHKERRQ(ierr);
        ierr = VecSetType(vec_dnz,VECMPI);CHKERRQ(ierr);
        ierr = VecDuplicate(vec_dnz,&vec_onz);CHKERRQ(ierr);

        ierr = PetscMalloc(pcbddc->local_primal_size*sizeof(PetscScalar),&my_dnz);CHKERRQ(ierr);
        ierr = PetscMalloc(pcbddc->local_primal_size*sizeof(PetscScalar),&my_onz);CHKERRQ(ierr);
        ierr = PetscMemzero(my_dnz,pcbddc->local_primal_size*sizeof(PetscScalar));CHKERRQ(ierr);
        ierr = PetscMemzero(my_onz,pcbddc->local_primal_size*sizeof(PetscScalar));CHKERRQ(ierr);

        ierr = PetscMalloc(coarse_size*sizeof(PetscInt),&row_ownership);CHKERRQ(ierr);
        ierr = MatGetOwnershipRanges(pcbddc->coarse_mat,(const PetscInt**)&mat_ranges);CHKERRQ(ierr);
        for (i=0;i<size_prec_comm;i++) {
          for (j=mat_ranges[i];j<mat_ranges[i+1];j++) {
            row_ownership[j]=i;
          }
        }

        for (i=0;i<pcbddc->local_primal_size;i++) {
          coarse_index_row = pcbddc->local_primal_indices[i];
          owner = row_ownership[coarse_index_row];
          for (j=i;j<pcbddc->local_primal_size;j++) {
            owner = row_ownership[coarse_index_row];
            coarse_index_col = pcbddc->local_primal_indices[j];
            if (coarse_index_col > mat_ranges[owner]-1 && coarse_index_col < mat_ranges[owner+1] ) {
              my_dnz[i] += 1.0;
            } else {
              my_onz[i] += 1.0;
            }
            if (i != j) {
              owner = row_ownership[coarse_index_col];
              if (coarse_index_row > mat_ranges[owner]-1 && coarse_index_row < mat_ranges[owner+1] ) {
                my_dnz[j] += 1.0;
              } else {
                my_onz[j] += 1.0;
              }
            }
          }
        }
        ierr = VecSet(vec_dnz,0.0);CHKERRQ(ierr);
        ierr = VecSet(vec_onz,0.0);CHKERRQ(ierr);
        if (pcbddc->local_primal_size) {
          ierr = VecSetValues(vec_dnz,pcbddc->local_primal_size,pcbddc->local_primal_indices,my_dnz,ADD_VALUES);CHKERRQ(ierr);
          ierr = VecSetValues(vec_onz,pcbddc->local_primal_size,pcbddc->local_primal_indices,my_onz,ADD_VALUES);CHKERRQ(ierr);
        }
        ierr = VecAssemblyBegin(vec_dnz);CHKERRQ(ierr);
        ierr = VecAssemblyBegin(vec_onz);CHKERRQ(ierr);
        ierr = VecAssemblyEnd(vec_dnz);CHKERRQ(ierr);
        ierr = VecAssemblyEnd(vec_onz);CHKERRQ(ierr);
        j = mat_ranges[rank_prec_comm+1]-mat_ranges[rank_prec_comm];
        ierr = VecGetArray(vec_dnz,&array);CHKERRQ(ierr);
        for (i=0; i<j; i++) dnz[i] = (PetscInt)PetscRealPart(array[i]);

        ierr = VecRestoreArray(vec_dnz,&array);CHKERRQ(ierr);
        ierr = VecGetArray(vec_onz,&array);CHKERRQ(ierr);
        for (i=0;i<j;i++) onz[i] = (PetscInt)PetscRealPart(array[i]);

        ierr = VecRestoreArray(vec_onz,&array);CHKERRQ(ierr);
        ierr = PetscFree(my_dnz);CHKERRQ(ierr);
        ierr = PetscFree(my_onz);CHKERRQ(ierr);
        ierr = PetscFree(row_ownership);CHKERRQ(ierr);
        ierr = VecDestroy(&vec_dnz);CHKERRQ(ierr);
        ierr = VecDestroy(&vec_onz);CHKERRQ(ierr);
      } else {
        for (k=0;k<size_prec_comm;k++){
          offset=pcbddc->local_primal_displacements[k];
          offset2=localdispl2[k];
          ins_local_primal_size = pcbddc->local_primal_sizes[k];
          ierr = PetscMalloc(ins_local_primal_size*sizeof(PetscInt),&ins_local_primal_indices);CHKERRQ(ierr);
          for (j=0;j<ins_local_primal_size;j++){
            ins_local_primal_indices[j]=(PetscInt)pcbddc->replicated_local_primal_indices[offset+j];
          }
          for (j=0;j<ins_local_primal_size;j++) {
            ierr = MatPreallocateSet(ins_local_primal_indices[j],ins_local_primal_size,ins_local_primal_indices,dnz,onz);CHKERRQ(ierr);
          }
          ierr = PetscFree(ins_local_primal_indices);CHKERRQ(ierr);
        }
      }

      /* check */
      for (i=0;i<lrows;i++) {
        if (dnz[i]>lcols) dnz[i]=lcols;
        if (onz[i]>coarse_size-lcols) onz[i]=coarse_size-lcols;
      }
      ierr = MatSeqAIJSetPreallocation(pcbddc->coarse_mat,0,dnz);CHKERRQ(ierr);
      ierr = MatMPIAIJSetPreallocation(pcbddc->coarse_mat,0,dnz,0,onz);CHKERRQ(ierr);
      ierr = MatPreallocateFinalize(dnz,onz);CHKERRQ(ierr);
    } else {
      ierr = MatSeqAIJSetPreallocation(matis_coarse_local_mat,0,dnz);CHKERRQ(ierr);
      ierr = PetscFree(dnz);CHKERRQ(ierr);
    }
    /* insert values */
    if (pcbddc->coarse_problem_type == PARALLEL_BDDC) {
      ierr = MatSetValues(pcbddc->coarse_mat,ins_local_primal_size,ins_local_primal_indices,ins_local_primal_size,ins_local_primal_indices,ins_coarse_mat_vals,ADD_VALUES);CHKERRQ(ierr);
    } else if (pcbddc->coarse_problem_type == MULTILEVEL_BDDC) {
      if (pcbddc->coarsening_ratio == 1) {
        ins_coarse_mat_vals = coarse_submat_vals;
        ierr = MatSetValues(pcbddc->coarse_mat,ins_local_primal_size,ins_local_primal_indices,ins_local_primal_size,ins_local_primal_indices,ins_coarse_mat_vals,INSERT_VALUES);CHKERRQ(ierr);
      } else {
        ierr = PetscFree(ins_local_primal_indices);CHKERRQ(ierr);
        for (k=0;k<pcbddc->replicated_primal_size;k++) {
          offset = pcbddc->local_primal_displacements[k];
          offset2 = localdispl2[k];
          ins_local_primal_size = pcbddc->local_primal_displacements[k+1]-pcbddc->local_primal_displacements[k];
          ierr = PetscMalloc(ins_local_primal_size*sizeof(PetscInt),&ins_local_primal_indices);CHKERRQ(ierr);
          for (j=0;j<ins_local_primal_size;j++){
            ins_local_primal_indices[j]=(PetscInt)pcbddc->replicated_local_primal_indices[offset+j];
          }
          ins_coarse_mat_vals = &temp_coarse_mat_vals[offset2];
          ierr = MatSetValues(pcbddc->coarse_mat,ins_local_primal_size,ins_local_primal_indices,ins_local_primal_size,ins_local_primal_indices,ins_coarse_mat_vals,ADD_VALUES);CHKERRQ(ierr);
          ierr = PetscFree(ins_local_primal_indices);CHKERRQ(ierr);
        }
      }
      ins_local_primal_indices = 0;
      ins_coarse_mat_vals = 0;
    } else {
      for (k=0;k<size_prec_comm;k++){
        offset=pcbddc->local_primal_displacements[k];
        offset2=localdispl2[k];
        ins_local_primal_size = pcbddc->local_primal_sizes[k];
        ierr = PetscMalloc(ins_local_primal_size*sizeof(PetscInt),&ins_local_primal_indices);CHKERRQ(ierr);
        for (j=0;j<ins_local_primal_size;j++){
          ins_local_primal_indices[j]=(PetscInt)pcbddc->replicated_local_primal_indices[offset+j];
        }
        ins_coarse_mat_vals = &temp_coarse_mat_vals[offset2];
        ierr = MatSetValues(pcbddc->coarse_mat,ins_local_primal_size,ins_local_primal_indices,ins_local_primal_size,ins_local_primal_indices,ins_coarse_mat_vals,ADD_VALUES);CHKERRQ(ierr);
        ierr = PetscFree(ins_local_primal_indices);CHKERRQ(ierr);
      }
      ins_local_primal_indices = 0;
      ins_coarse_mat_vals = 0;
    }
    ierr = MatAssemblyBegin(pcbddc->coarse_mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(pcbddc->coarse_mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    /* symmetry of coarse matrix */
    if (issym) {
      ierr = MatSetOption(pcbddc->coarse_mat,MAT_SYMMETRIC,PETSC_TRUE);CHKERRQ(ierr);
    }
    ierr = MatGetVecs(pcbddc->coarse_mat,&pcbddc->coarse_vec,&pcbddc->coarse_rhs);CHKERRQ(ierr);
  }

  /* create loc to glob scatters if needed */
  if (pcbddc->coarse_communications_type == SCATTERS_BDDC) {
     IS local_IS,global_IS;
     ierr = ISCreateStride(PETSC_COMM_SELF,pcbddc->local_primal_size,0,1,&local_IS);CHKERRQ(ierr);
     ierr = ISCreateGeneral(PETSC_COMM_SELF,pcbddc->local_primal_size,pcbddc->local_primal_indices,PETSC_COPY_VALUES,&global_IS);CHKERRQ(ierr);
     ierr = VecScatterCreate(pcbddc->vec1_P,local_IS,pcbddc->coarse_vec,global_IS,&pcbddc->coarse_loc_to_glob);CHKERRQ(ierr);
     ierr = ISDestroy(&local_IS);CHKERRQ(ierr);
     ierr = ISDestroy(&global_IS);CHKERRQ(ierr);
  }

  /* free memory no longer needed */
  if (coarse_ISLG)              { ierr = ISLocalToGlobalMappingDestroy(&coarse_ISLG);CHKERRQ(ierr); }
  if (ins_local_primal_indices) { ierr = PetscFree(ins_local_primal_indices);CHKERRQ(ierr); }
  if (ins_coarse_mat_vals)      { ierr = PetscFree(ins_coarse_mat_vals);CHKERRQ(ierr); }
  if (localsizes2)              { ierr = PetscFree(localsizes2);CHKERRQ(ierr); }
  if (localdispl2)              { ierr = PetscFree(localdispl2);CHKERRQ(ierr); }
  if (temp_coarse_mat_vals)     { ierr = PetscFree(temp_coarse_mat_vals);CHKERRQ(ierr); }

  /* Compute coarse null space */
  CoarseNullSpace = 0;
  if (pcbddc->NullSpace) {
    ierr = PCBDDCNullSpaceAssembleCoarse(pc,&CoarseNullSpace);CHKERRQ(ierr);
  }

  /* KSP for coarse problem */
  if (rank_prec_comm == active_rank) {
    PetscBool isbddc=PETSC_FALSE;

    ierr = KSPCreate(coarse_comm,&pcbddc->coarse_ksp);CHKERRQ(ierr);
    ierr = PetscObjectIncrementTabLevel((PetscObject)pcbddc->coarse_ksp,(PetscObject)pc,1);CHKERRQ(ierr);
    ierr = KSPSetOperators(pcbddc->coarse_ksp,pcbddc->coarse_mat,pcbddc->coarse_mat,SAME_PRECONDITIONER);CHKERRQ(ierr);
    ierr = KSPSetTolerances(pcbddc->coarse_ksp,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT,max_it_coarse_ksp);CHKERRQ(ierr);
    ierr = KSPSetType(pcbddc->coarse_ksp,coarse_ksp_type);CHKERRQ(ierr);
    ierr = KSPGetPC(pcbddc->coarse_ksp,&pc_temp);CHKERRQ(ierr);
    ierr = PCSetType(pc_temp,coarse_pc_type);CHKERRQ(ierr);
    /* Allow user's customization */
    ierr = KSPSetOptionsPrefix(pcbddc->coarse_ksp,"coarse_");CHKERRQ(ierr);
    /* Set Up PC for coarse problem BDDC */
    if (pcbddc->coarse_problem_type == MULTILEVEL_BDDC) {
      i = pcbddc->current_level+1;
      ierr = PCBDDCSetLevel(pc_temp,i);CHKERRQ(ierr);
      ierr = PCBDDCSetCoarseningRatio(pc_temp,pcbddc->coarsening_ratio);CHKERRQ(ierr);
      ierr = PCBDDCSetMaxLevels(pc_temp,pcbddc->max_levels);CHKERRQ(ierr);
      ierr = PCBDDCSetCoarseProblemType(pc_temp,MULTILEVEL_BDDC);CHKERRQ(ierr);
      if (CoarseNullSpace) {
        ierr = PCBDDCSetNullSpace(pc_temp,CoarseNullSpace);CHKERRQ(ierr);
      }
      if (dbg_flag) {
        ierr = PetscViewerASCIIPrintf(viewer,"----------------Level %d: Setting up level %d---------------\n",pcbddc->current_level,i);CHKERRQ(ierr);
        ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
      }
    } else {
      if (CoarseNullSpace) {
        ierr = KSPSetNullSpace(pcbddc->coarse_ksp,CoarseNullSpace);CHKERRQ(ierr);
      }
    }
    ierr = KSPSetFromOptions(pcbddc->coarse_ksp);CHKERRQ(ierr);
    ierr = KSPSetUp(pcbddc->coarse_ksp);CHKERRQ(ierr);

    ierr = KSPGetTolerances(pcbddc->coarse_ksp,NULL,NULL,NULL,&j);CHKERRQ(ierr);
    ierr = KSPGetPC(pcbddc->coarse_ksp,&pc_temp);CHKERRQ(ierr);
    ierr = PetscObjectTypeCompare((PetscObject)pc_temp,PCBDDC,&isbddc);CHKERRQ(ierr);
    if (j == 1) {
      ierr = KSPSetNormType(pcbddc->coarse_ksp,KSP_NORM_NONE);CHKERRQ(ierr);
      if (isbddc) {
        ierr = PCBDDCSetUseExactDirichlet(pc_temp,PETSC_FALSE);CHKERRQ(ierr);
      }
    }
  }
  /* Check coarse problem if requested */
  if ( dbg_flag && rank_prec_comm == active_rank ) {
    KSP check_ksp;
    PC  check_pc;
    Vec check_vec;
    PetscReal   abs_infty_error,infty_error,lambda_min,lambda_max;
    KSPType check_ksp_type;

    /* Create ksp object suitable for extreme eigenvalues' estimation */
    ierr = KSPCreate(coarse_comm,&check_ksp);CHKERRQ(ierr);
    ierr = KSPSetOperators(check_ksp,pcbddc->coarse_mat,pcbddc->coarse_mat,SAME_PRECONDITIONER);CHKERRQ(ierr);
    ierr = KSPSetTolerances(check_ksp,1.e-12,1.e-12,PETSC_DEFAULT,coarse_size);CHKERRQ(ierr);
    if (pcbddc->coarse_problem_type == MULTILEVEL_BDDC) {
      if (issym) check_ksp_type = KSPCG;
      else check_ksp_type = KSPGMRES;
      ierr = KSPSetComputeSingularValues(check_ksp,PETSC_TRUE);CHKERRQ(ierr);
    } else {
      check_ksp_type = KSPPREONLY;
    }
    ierr = KSPSetType(check_ksp,check_ksp_type);CHKERRQ(ierr);
    ierr = KSPGetPC(pcbddc->coarse_ksp,&check_pc);CHKERRQ(ierr);
    ierr = KSPSetPC(check_ksp,check_pc);CHKERRQ(ierr);
    ierr = KSPSetUp(check_ksp);CHKERRQ(ierr);
    /* create random vec */
    ierr = VecDuplicate(pcbddc->coarse_vec,&check_vec);CHKERRQ(ierr);
    ierr = VecSetRandom(check_vec,NULL);CHKERRQ(ierr);
    if (CoarseNullSpace) {
      ierr = MatNullSpaceRemove(CoarseNullSpace,check_vec);CHKERRQ(ierr);
    }
    ierr = MatMult(pcbddc->coarse_mat,check_vec,pcbddc->coarse_rhs);CHKERRQ(ierr);
    /* solve coarse problem */
    ierr = KSPSolve(check_ksp,pcbddc->coarse_rhs,pcbddc->coarse_vec);CHKERRQ(ierr);
    if (CoarseNullSpace) {
      ierr = MatNullSpaceRemove(CoarseNullSpace,pcbddc->coarse_vec);CHKERRQ(ierr);
    }
    /* check coarse problem residual error */
    ierr = VecAXPY(check_vec,-1.0,pcbddc->coarse_vec);CHKERRQ(ierr);
    ierr = VecNorm(check_vec,NORM_INFINITY,&infty_error);CHKERRQ(ierr);
    ierr = MatMult(pcbddc->coarse_mat,check_vec,pcbddc->coarse_rhs);CHKERRQ(ierr);
    ierr = VecNorm(pcbddc->coarse_rhs,NORM_INFINITY,&abs_infty_error);CHKERRQ(ierr);
    ierr = VecDestroy(&check_vec);CHKERRQ(ierr);
    /* get eigenvalue estimation if inexact */
    if (pcbddc->coarse_problem_type == MULTILEVEL_BDDC) {
      ierr = KSPComputeExtremeSingularValues(check_ksp,&lambda_max,&lambda_min);CHKERRQ(ierr);
      ierr = KSPGetIterationNumber(check_ksp,&k);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"Coarse problem eigenvalues estimated with %d iterations of %s.\n",k,check_ksp_type);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"Coarse problem eigenvalues: % 1.14e %1.14e\n",lambda_min,lambda_max);CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPrintf(viewer,"Coarse problem exact infty_error   : %1.14e\n",infty_error);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"Coarse problem residual infty_error: %1.14e\n",abs_infty_error);CHKERRQ(ierr);
    ierr = KSPDestroy(&check_ksp);CHKERRQ(ierr);
  }
  if (dbg_flag) {
    ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
  }
  ierr = MatNullSpaceDestroy(&CoarseNullSpace);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

