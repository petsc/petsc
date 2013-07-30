#include "bddc.h"
#include "bddcprivate.h"
#include <petscblaslapack.h>

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


