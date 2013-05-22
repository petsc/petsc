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
  ierr = VecDestroy(&pcbddc->temp_solution);CHKERRQ(ierr);
  ierr = VecDestroy(&pcbddc->original_rhs);CHKERRQ(ierr);
  ierr = MatDestroy(&pcbddc->local_mat);CHKERRQ(ierr);
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
  ierr = VecScatterDestroy(&pcbddc->R_to_B);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&pcbddc->R_to_D);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&pcbddc->coarse_loc_to_glob);CHKERRQ(ierr);
  ierr = KSPDestroy(&pcbddc->ksp_D);CHKERRQ(ierr);
  ierr = KSPDestroy(&pcbddc->ksp_R);CHKERRQ(ierr);
  ierr = PetscFree(pcbddc->local_primal_indices);CHKERRQ(ierr);
  ierr = PetscFree(pcbddc->replicated_local_primal_indices);CHKERRQ(ierr);
  ierr = PetscFree(pcbddc->replicated_local_primal_values);CHKERRQ(ierr);
  ierr = PetscFree(pcbddc->local_primal_displacements);CHKERRQ(ierr);
  ierr = PetscFree(pcbddc->local_primal_sizes);CHKERRQ(ierr);
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

#undef __FUNCT__
#define __FUNCT__ "PCBDDCConstraintsSetUp"
PetscErrorCode PCBDDCConstraintsSetUp(PC pc)
{
  PetscErrorCode ierr;
  PC_IS*         pcis = (PC_IS*)(pc->data);
  PC_BDDC*       pcbddc = (PC_BDDC*)pc->data;
  Mat_IS         *matis = (Mat_IS*)pc->pmat->data;
  PetscInt       *nnz,*is_indices;
  PetscScalar    *temp_quadrature_constraint;
  PetscInt       *temp_indices,*temp_indices_to_constraint,*temp_indices_to_constraint_B,*local_to_B;
  PetscInt       local_primal_size,i,j,k,total_counts,max_size_of_constraint;
  PetscInt       n_vertices,size_of_constraint;
  PetscReal      real_value;
  PetscBool      nnsp_has_cnst=PETSC_FALSE,use_nnsp_true=pcbddc->use_nnsp_true;
  PetscInt       nnsp_size=0,nnsp_addone=0,temp_constraints,temp_start_ptr,n_ISForFaces,n_ISForEdges;
  IS             *used_IS,ISForVertices,*ISForFaces,*ISForEdges;
  MatType        impMatType=MATSEQAIJ;
  PetscBLASInt   Bs,Bt,lwork,lierr;
  PetscReal      tol=1.0e-8;
  MatNullSpace   nearnullsp;
  const Vec      *nearnullvecs;
  Vec            *localnearnullsp;
  PetscScalar    *work,*temp_basis,*array_vector,*correlation_mat;
  PetscReal      *rwork,*singular_vals;
  PetscBLASInt   Bone=1,*ipiv;
  Vec            temp_vec;
  Mat            temp_mat;
  KSP            temp_ksp;
  PC             temp_pc;
  PetscInt       s,start_constraint,dual_dofs;
  PetscBool      compute_submatrix,useksp=PETSC_FALSE;
  PetscInt       *aux_primal_permutation,*aux_primal_numbering;
  PetscBool      boolforchange,*change_basis;
/* some ugly conditional declarations */
#if defined(PETSC_MISSING_LAPACK_GESVD)
  PetscScalar    one=1.0,zero=0.0;
  PetscInt       ii;
  PetscScalar    *singular_vectors;
  PetscBLASInt   *iwork,*ifail;
  PetscReal      dummy_real,abs_tol;
  PetscBLASInt   eigs_found;
#endif
  PetscBLASInt   dummy_int;
  PetscScalar    dummy_scalar;
  PetscBool      used_vertex,get_faces,get_edges,get_vertices;

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
  } else { /* if near null space is not provided it uses constants */
    nnsp_has_cnst = PETSC_TRUE;
    use_nnsp_true = PETSC_TRUE;
  }
  if (nnsp_has_cnst) {
    nnsp_addone = 1;
  }
  /*
       Evaluate maximum storage size needed by the procedure
       - temp_indices will contain start index of each constraint stored as follows
       - temp_indices_to_constraint  [temp_indices[i],...,temp[indices[i+1]-1] will contain the indices (in local numbering) on which the constraint acts
       - temp_indices_to_constraint_B[temp_indices[i],...,temp[indices[i+1]-1] will contain the indices (in boundary numbering) on which the constraint acts
       - temp_quadrature_constraint  [temp_indices[i],...,temp[indices[i+1]-1] will contain the scalars representing the constraint itself
                                                                                                                                                         */
  total_counts = n_ISForFaces+n_ISForEdges;
  total_counts *= (nnsp_addone+nnsp_size);
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
  total_counts *= (nnsp_addone+nnsp_size);
  total_counts += n_vertices;
  ierr = PetscMalloc(total_counts*sizeof(PetscScalar),&temp_quadrature_constraint);CHKERRQ(ierr);
  ierr = PetscMalloc(total_counts*sizeof(PetscInt),&temp_indices_to_constraint);CHKERRQ(ierr);
  ierr = PetscMalloc(total_counts*sizeof(PetscInt),&temp_indices_to_constraint_B);CHKERRQ(ierr);
  ierr = PetscMalloc(pcis->n*sizeof(PetscInt),&local_to_B);CHKERRQ(ierr);
  ierr = ISGetIndices(pcis->is_B_local,(const PetscInt**)&is_indices);CHKERRQ(ierr);
  for (i=0;i<pcis->n;i++) {
    local_to_B[i]=-1;
  }
  for (i=0;i<pcis->n_B;i++) {
    local_to_B[is_indices[i]]=i;
  }
  ierr = ISRestoreIndices(pcis->is_B_local,(const PetscInt**)&is_indices);CHKERRQ(ierr);

  /* First we issue queries to allocate optimal workspace for LAPACKgesvd or LAPACKsyev/LAPACKheev */
  rwork = 0;
  work = 0;
  singular_vals = 0;
  temp_basis = 0;
  correlation_mat = 0;
  if (!pcbddc->use_nnsp_true) {
    PetscScalar temp_work;
#if defined(PETSC_MISSING_LAPACK_GESVD)
    /* POD */
    PetscInt max_n;
    max_n = nnsp_addone+nnsp_size;
    /* using some techniques borrowed from Proper Orthogonal Decomposition */
    ierr = PetscMalloc(max_n*max_n*sizeof(PetscScalar),&correlation_mat);CHKERRQ(ierr);
    ierr = PetscMalloc(max_n*max_n*sizeof(PetscScalar),&singular_vectors);CHKERRQ(ierr);
    ierr = PetscMalloc(max_n*sizeof(PetscReal),&singular_vals);CHKERRQ(ierr);
    ierr = PetscMalloc(max_size_of_constraint*(nnsp_addone+nnsp_size)*sizeof(PetscScalar),&temp_basis);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
    ierr = PetscMalloc(3*max_n*sizeof(PetscReal),&rwork);CHKERRQ(ierr);
#endif
    ierr = PetscMalloc(5*max_n*sizeof(PetscBLASInt),&iwork);CHKERRQ(ierr);
    ierr = PetscMalloc(max_n*sizeof(PetscBLASInt),&ifail);CHKERRQ(ierr);
    /* now we evaluate the optimal workspace using query with lwork=-1 */
    ierr = PetscBLASIntCast(max_n,&Bt);CHKERRQ(ierr);
    lwork=-1;
    ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
    abs_tol=1.e-8;
/*    LAPACKsyev_("V","U",&Bt,correlation_mat,&Bt,singular_vals,&temp_work,&lwork,&lierr); */
    PetscStackCallBLAS("LAPACKsyevx",LAPACKsyevx_("V","A","U",&Bt,correlation_mat,&Bt,&dummy_real,&dummy_real,&dummy_int,&dummy_int,&abs_tol,&eigs_found,singular_vals,singular_vectors,&Bt,&temp_work,&lwork,iwork,ifail,&lierr));
#else
/*    LAPACKsyev_("V","U",&Bt,correlation_mat,&Bt,singular_vals,&temp_work,&lwork,rwork,&lierr); */
/*  LAPACK call is missing here! TODO */
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Not yet implemented for complexes when PETSC_MISSING_GESVD = 1");
#endif
    if ( lierr ) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in query to SYEVX Lapack routine %d",(int)lierr);
    }
    ierr = PetscFPTrapPop();CHKERRQ(ierr);
#else /* on missing GESVD */
    /* SVD */
    PetscInt max_n,min_n;
    max_n = max_size_of_constraint;
    min_n = nnsp_addone+nnsp_size;
    if (max_size_of_constraint < ( nnsp_addone+nnsp_size ) ) {
      min_n = max_size_of_constraint;
      max_n = nnsp_addone+nnsp_size;
    }
    ierr = PetscMalloc(min_n*sizeof(PetscReal),&singular_vals);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
    ierr = PetscMalloc(5*min_n*sizeof(PetscReal),&rwork);CHKERRQ(ierr);
#endif
    /* now we evaluate the optimal workspace using query with lwork=-1 */
    lwork=-1;
    ierr = PetscBLASIntCast(max_n,&Bs);CHKERRQ(ierr);
    ierr = PetscBLASIntCast(min_n,&Bt);CHKERRQ(ierr);
    dummy_int = Bs;
    ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
    PetscStackCallBLAS("LAPACKgesvd",LAPACKgesvd_("O","N",&Bs,&Bt,&temp_quadrature_constraint[0],&Bs,singular_vals,&dummy_scalar,&dummy_int,&dummy_scalar,&dummy_int,&temp_work,&lwork,&lierr));
#else
    PetscStackCallBLAS("LAPACKgesvd",LAPACKgesvd_("O","N",&Bs,&Bt,&temp_quadrature_constraint[0],&Bs,singular_vals,&dummy_scalar,&dummy_int,&dummy_scalar,&dummy_int,&temp_work,&lwork,rwork,&lierr));
#endif
    if ( lierr ) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in query to SVD Lapack routine %d",(int)lierr);
    }
    ierr = PetscFPTrapPop();CHKERRQ(ierr);
#endif
    /* Allocate optimal workspace */
    ierr = PetscBLASIntCast((PetscInt)PetscRealPart(temp_work),&lwork);CHKERRQ(ierr);
    total_counts = (PetscInt)lwork;
    ierr = PetscMalloc(total_counts*sizeof(PetscScalar),&work);CHKERRQ(ierr);
  }
  /* get local part of global near null space vectors */
  ierr = PetscMalloc(nnsp_size*sizeof(Vec),&localnearnullsp);CHKERRQ(ierr);
  for (k=0;k<nnsp_size;k++) {
    ierr = VecDuplicate(pcis->vec1_N,&localnearnullsp[k]);CHKERRQ(ierr);
    ierr = VecScatterBegin(matis->ctx,nearnullvecs[k],localnearnullsp[k],INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(matis->ctx,nearnullvecs[k],localnearnullsp[k],INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
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
      for (i=0;i<n_vertices;i++) {
        used_vertex=PETSC_FALSE;
        k=0;
        while (!used_vertex && k<nnsp_size) {
          ierr = VecGetArrayRead(localnearnullsp[k],(const PetscScalar**)&array_vector);CHKERRQ(ierr);
          if (PetscAbsScalar(array_vector[is_indices[i]])>0.0) {
            temp_indices_to_constraint[temp_indices[total_counts]]=is_indices[i];
            temp_indices_to_constraint_B[temp_indices[total_counts]]=local_to_B[is_indices[i]];
            temp_quadrature_constraint[temp_indices[total_counts]]=1.0;
            temp_indices[total_counts+1]=temp_indices[total_counts]+1;
            change_basis[total_counts]=PETSC_FALSE;
            total_counts++;
            used_vertex=PETSC_TRUE;
          }
          ierr = VecRestoreArrayRead(localnearnullsp[k],(const PetscScalar**)&array_vector);CHKERRQ(ierr);
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
      boolforchange = pcbddc->use_change_of_basis;
    } else {
      used_IS = &ISForFaces[i-n_ISForEdges];
      boolforchange = pcbddc->use_change_on_faces;
    }
    temp_constraints = 0;          /* zero the number of constraints I have on this conn comp */
    temp_start_ptr = total_counts; /* need to know the starting index of constraints stored */
    ierr = ISGetSize(*used_IS,&size_of_constraint);CHKERRQ(ierr);
    ierr = ISGetIndices(*used_IS,(const PetscInt**)&is_indices);CHKERRQ(ierr);
    /* HACK: change of basis should not performed on local periodic nodes */
    if (pcbddc->mat_graph->mirrors && pcbddc->mat_graph->mirrors[is_indices[0]]) {
      boolforchange = PETSC_FALSE;
    }
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
      ierr = VecGetArrayRead(localnearnullsp[k],(const PetscScalar**)&array_vector);CHKERRQ(ierr);
      for (j=0;j<size_of_constraint;j++) {
        temp_indices_to_constraint[temp_indices[total_counts]+j]=is_indices[j];
        temp_indices_to_constraint_B[temp_indices[total_counts]+j]=local_to_B[is_indices[j]];
        temp_quadrature_constraint[temp_indices[total_counts]+j]=array_vector[is_indices[j]];
      }
      ierr = VecRestoreArrayRead(localnearnullsp[k],(const PetscScalar**)&array_vector);CHKERRQ(ierr);
      real_value = 1.0;
      if (use_nnsp_true) { /* check if array is null on the connected component in case use_nnsp_true has been requested */
        ierr = PetscBLASIntCast(size_of_constraint,&Bs);CHKERRQ(ierr);
        PetscStackCallBLAS("BLASasum",real_value = BLASasum_(&Bs,&temp_quadrature_constraint[temp_indices[total_counts]],&Bone));
      }
      if (real_value > 0.0) { /* keep indices and values */
        temp_constraints++;
        temp_indices[total_counts+1]=temp_indices[total_counts]+size_of_constraint;  /* store new starting point */
        change_basis[total_counts]=boolforchange;
        total_counts++;
      }
    }
    ierr = ISRestoreIndices(*used_IS,(const PetscInt**)&is_indices);CHKERRQ(ierr);
    /* perform SVD on the constraint if use_nnsp_true has not be requested by the user */
    if (!use_nnsp_true) {
      ierr = PetscBLASIntCast(size_of_constraint,&Bs);CHKERRQ(ierr);
      ierr = PetscBLASIntCast(temp_constraints,&Bt);CHKERRQ(ierr);

#if defined(PETSC_MISSING_LAPACK_GESVD)
      ierr = PetscMemzero(correlation_mat,Bt*Bt*sizeof(PetscScalar));CHKERRQ(ierr);
      /* Store upper triangular part of correlation matrix */
      for (j=0;j<temp_constraints;j++) {
        for (k=0;k<j+1;k++) {
          PetscStackCallBLAS("BLASdot",correlation_mat[j*temp_constraints+k]=BLASdot_(&Bs,&temp_quadrature_constraint[temp_indices[temp_start_ptr+j]],&Bone,&temp_quadrature_constraint[temp_indices[temp_start_ptr+k]],&Bone));

        }
      }
      ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
/*      LAPACKsyev_("V","U",&Bt,correlation_mat,&Bt,singular_vals,work,&lwork,&lierr); */
      PetscStackCallBLAS("LAPACKsyevx",LAPACKsyevx_("V","A","U",&Bt,correlation_mat,&Bt,&dummy_real,&dummy_real,&dummy_int,&dummy_int,&abs_tol,&eigs_found,singular_vals,singular_vectors,&Bt,work,&lwork,iwork,ifail,&lierr));
#else
/*  LAPACK call is missing here! TODO */
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Not yet implemented for complexes when PETSC_MISSING_GESVD = 1");
#endif
      if (lierr) {
        SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in SYEVX Lapack routine %d",(int)lierr);
      }
      ierr = PetscFPTrapPop();CHKERRQ(ierr);
      /* retain eigenvalues greater than tol: note that lapack SYEV gives eigs in ascending order */
      j=0;
      while (j < Bt && singular_vals[j] < tol) j++;
      total_counts=total_counts-j;
      if (j<temp_constraints) {
        for (k=j;k<Bt;k++) {
          singular_vals[k]=1.0/PetscSqrtReal(singular_vals[k]);
        }
        ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
        PetscStackCallBLAS("BLASgemm_",BLASgemm_("N","N",&Bs,&Bt,&Bt,&one,&temp_quadrature_constraint[temp_indices[temp_start_ptr]],&Bs,correlation_mat,&Bt,&zero,temp_basis,&Bs));
        ierr = PetscFPTrapPop();CHKERRQ(ierr);
        /* copy POD basis into used quadrature memory */
        for (k=0;k<Bt-j;k++) {
          for (ii=0;ii<size_of_constraint;ii++) {
            temp_quadrature_constraint[temp_indices[temp_start_ptr+k]+ii]=singular_vals[Bt-1-k]*temp_basis[(Bt-1-k)*size_of_constraint+ii];
          }
        }
      }

#else  /* on missing GESVD */
      PetscInt min_n = temp_constraints;
      if (min_n > size_of_constraint) min_n = size_of_constraint;
      dummy_int = Bs;
      ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
      PetscStackCallBLAS("LAPACKgesvd",LAPACKgesvd_("O","N",&Bs,&Bt,&temp_quadrature_constraint[temp_indices[temp_start_ptr]],&Bs,singular_vals,&dummy_scalar,&dummy_int,&dummy_scalar,&dummy_int,work,&lwork,&lierr));
#else
      PetscStackCallBLAS("LAPACKgesvd",LAPACKgesvd_("O","N",&Bs,&Bt,&temp_quadrature_constraint[temp_indices[temp_start_ptr]],&Bs,singular_vals,&dummy_scalar,&dummy_int,&dummy_scalar,&dummy_int,work,&lwork,rwork,&lierr));
#endif
      if (lierr) {
        SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in SVD Lapack routine %d",(int)lierr);
      }
      ierr = PetscFPTrapPop();CHKERRQ(ierr);
      /* retain eigenvalues greater than tol: note that lapack SVD gives eigs in descending order */
      j = 0;
      while (j < min_n && singular_vals[min_n-j-1] < tol) j++;
      total_counts = total_counts-(PetscInt)Bt+(min_n-j);
#endif
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

  /* set quantities in pcbddc data structure */
  /* n_vertices defines the number of point primal dofs */
  /* n_constraints defines the number of averages (they can be point primal dofs if change of basis is requested) */
  local_primal_size = total_counts;
  pcbddc->n_vertices = n_vertices;
  pcbddc->n_constraints = total_counts-n_vertices;
  pcbddc->local_primal_size = local_primal_size;

  /* Create constraint matrix */
  /* The constraint matrix is used to compute the l2g map of primal dofs */
  /* so we need to set it up properly either with or without change of basis */
  ierr = MatCreate(PETSC_COMM_SELF,&pcbddc->ConstraintMatrix);CHKERRQ(ierr);
  ierr = MatSetType(pcbddc->ConstraintMatrix,impMatType);CHKERRQ(ierr);
  ierr = MatSetSizes(pcbddc->ConstraintMatrix,local_primal_size,pcis->n,local_primal_size,pcis->n);CHKERRQ(ierr);
  /* compute a local numbering of constraints : vertices first then constraints */
  ierr = VecSet(pcis->vec1_N,0.0);CHKERRQ(ierr);
  ierr = VecGetArray(pcis->vec1_N,&array_vector);CHKERRQ(ierr);
  ierr = PetscMalloc(local_primal_size*sizeof(PetscInt),&aux_primal_numbering);CHKERRQ(ierr);
  ierr = PetscMalloc(local_primal_size*sizeof(PetscInt),&aux_primal_permutation);CHKERRQ(ierr);
  total_counts=0;
  /* find vertices: subdomain corners plus dofs with basis changed */
  for (i=0;i<local_primal_size;i++) {
    size_of_constraint=temp_indices[i+1]-temp_indices[i];
    if (change_basis[i] || size_of_constraint == 1) {
      k=0;
      while(k < size_of_constraint && array_vector[temp_indices_to_constraint[temp_indices[i]+size_of_constraint-k-1]] != 0.0) {
        k=k+1;
      }
      j=temp_indices_to_constraint[temp_indices[i]+size_of_constraint-k-1];
      array_vector[j] = 1.0;
      aux_primal_numbering[total_counts]=j;
      aux_primal_permutation[total_counts]=total_counts;
      total_counts++;
    }
  }
  ierr = VecRestoreArray(pcis->vec1_N,&array_vector);CHKERRQ(ierr);
  /* permute indices in order to have a sorted set of vertices */
  ierr = PetscSortIntWithPermutation(total_counts,aux_primal_numbering,aux_primal_permutation);
  /* nonzero structure */
  ierr = PetscMalloc(local_primal_size*sizeof(PetscInt),&nnz);CHKERRQ(ierr);
  for (i=0;i<total_counts;i++) {
    nnz[i]=1;
  }
  j=total_counts;
  for (i=n_vertices;i<local_primal_size;i++) {
    if (!change_basis[i]) {
      nnz[j]=temp_indices[i+1]-temp_indices[i];
      j++;
    }
  }
  ierr = MatSeqAIJSetPreallocation(pcbddc->ConstraintMatrix,0,nnz);CHKERRQ(ierr);
  ierr = PetscFree(nnz);CHKERRQ(ierr);
  /* set values in constraint matrix */
  for (i=0;i<total_counts;i++) {
    j = aux_primal_permutation[i];
    k = aux_primal_numbering[j];
    ierr = MatSetValue(pcbddc->ConstraintMatrix,i,k,1.0,INSERT_VALUES);CHKERRQ(ierr);
  }
  for (i=n_vertices;i<local_primal_size;i++) {
    if (!change_basis[i]) {
      size_of_constraint=temp_indices[i+1]-temp_indices[i];
      ierr = MatSetValues(pcbddc->ConstraintMatrix,1,&total_counts,size_of_constraint,&temp_indices_to_constraint[temp_indices[i]],&temp_quadrature_constraint[temp_indices[i]],INSERT_VALUES);CHKERRQ(ierr);
      total_counts++;
    }
  }
  ierr = PetscFree(aux_primal_numbering);CHKERRQ(ierr);
  ierr = PetscFree(aux_primal_permutation);CHKERRQ(ierr);
  /* assembling */
  ierr = MatAssemblyBegin(pcbddc->ConstraintMatrix,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(pcbddc->ConstraintMatrix,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* Create matrix for change of basis. We don't need it in case pcbddc->use_change_of_basis is FALSE */
  if (pcbddc->use_change_of_basis) {
    ierr = MatCreate(PETSC_COMM_SELF,&pcbddc->ChangeOfBasisMatrix);CHKERRQ(ierr);
    ierr = MatSetType(pcbddc->ChangeOfBasisMatrix,impMatType);CHKERRQ(ierr);
    ierr = MatSetSizes(pcbddc->ChangeOfBasisMatrix,pcis->n_B,pcis->n_B,pcis->n_B,pcis->n_B);CHKERRQ(ierr);
    /* work arrays */
    /* we need to reuse these arrays, so we free them */
    ierr = PetscFree(temp_basis);CHKERRQ(ierr);
    ierr = PetscFree(work);CHKERRQ(ierr);
    ierr = PetscMalloc(pcis->n_B*sizeof(PetscInt),&nnz);CHKERRQ(ierr);
    ierr = PetscMalloc((nnsp_addone+nnsp_size)*(nnsp_addone+nnsp_size)*sizeof(PetscScalar),&temp_basis);CHKERRQ(ierr);
    ierr = PetscMalloc((nnsp_addone+nnsp_size)*sizeof(PetscScalar),&work);CHKERRQ(ierr);
    ierr = PetscMalloc((nnsp_addone+nnsp_size)*sizeof(PetscBLASInt),&ipiv);CHKERRQ(ierr);
    for (i=0;i<pcis->n_B;i++) {
      nnz[i]=1;
    }
    /* Overestimated nonzeros per row */
    k=1;
    for (i=pcbddc->n_vertices;i<local_primal_size;i++) {
      if (change_basis[i]) {
        size_of_constraint = temp_indices[i+1]-temp_indices[i];
        if (k < size_of_constraint) {
          k = size_of_constraint;
        }
        for (j=0;j<size_of_constraint;j++) {
          nnz[temp_indices_to_constraint_B[temp_indices[i]+j]] = size_of_constraint;
        }
      }
    }
    ierr = MatSeqAIJSetPreallocation(pcbddc->ChangeOfBasisMatrix,0,nnz);CHKERRQ(ierr);
    ierr = PetscFree(nnz);CHKERRQ(ierr);
    /* Temporary array to store indices */
    ierr = PetscMalloc(k*sizeof(PetscInt),&is_indices);CHKERRQ(ierr);
    /* Set initial identity in the matrix */
    for (i=0;i<pcis->n_B;i++) {
      ierr = MatSetValue(pcbddc->ChangeOfBasisMatrix,i,i,1.0,INSERT_VALUES);CHKERRQ(ierr);
    }
    /* Now we loop on the constraints which need a change of basis */
    /* Change of basis matrix is evaluated as the FIRST APPROACH in */
    /* Klawonn and Widlund, Dual-primal FETI-DP methods for linear elasticity, (6.2.1) */
    temp_constraints = 0;
    if (pcbddc->n_vertices < local_primal_size) {
      temp_start_ptr = temp_indices_to_constraint_B[temp_indices[pcbddc->n_vertices]];
    }
    for (i=pcbddc->n_vertices;i<local_primal_size;i++) {
      if (change_basis[i]) {
        compute_submatrix = PETSC_FALSE;
        useksp = PETSC_FALSE;
        if (temp_start_ptr == temp_indices_to_constraint_B[temp_indices[i]]) {
          temp_constraints++;
          if (i == local_primal_size -1 ||  temp_start_ptr != temp_indices_to_constraint_B[temp_indices[i+1]]) {
            compute_submatrix = PETSC_TRUE;
          }
        }
        if (compute_submatrix) {
          if (temp_constraints > 1 || pcbddc->use_nnsp_true) {
            useksp = PETSC_TRUE;
          }
          size_of_constraint = temp_indices[i+1]-temp_indices[i];
          if (useksp) { /* experimental TODO: reuse KSP and MAT instead of creating them each time */
            ierr = MatCreate(PETSC_COMM_SELF,&temp_mat);CHKERRQ(ierr);
            ierr = MatSetType(temp_mat,impMatType);CHKERRQ(ierr);
            ierr = MatSetSizes(temp_mat,size_of_constraint,size_of_constraint,size_of_constraint,size_of_constraint);CHKERRQ(ierr);
            ierr = MatSeqAIJSetPreallocation(temp_mat,size_of_constraint,NULL);CHKERRQ(ierr);
          }
          /* First _size_of_constraint-temp_constraints_ columns */
          dual_dofs = size_of_constraint-temp_constraints;
          start_constraint = i+1-temp_constraints;
          for (s=0;s<dual_dofs;s++) {
            is_indices[0] = s;
            for (j=0;j<temp_constraints;j++) {
              for (k=0;k<temp_constraints;k++) {
                temp_basis[j*temp_constraints+k]=temp_quadrature_constraint[temp_indices[start_constraint+k]+s+j+1];
              }
              work[j]=-temp_quadrature_constraint[temp_indices[start_constraint+j]+s];
              is_indices[j+1]=s+j+1;
            }
            Bt = temp_constraints;
            ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
            PetscStackCallBLAS("LAPACKgesv",LAPACKgesv_(&Bt,&Bone,temp_basis,&Bt,ipiv,work,&Bt,&lierr));
            if ( lierr ) {
              SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in GESV Lapack routine %d",(int)lierr);
            }
            ierr = PetscFPTrapPop();CHKERRQ(ierr);
            j = temp_indices_to_constraint_B[temp_indices[start_constraint]+s];
            ierr = MatSetValues(pcbddc->ChangeOfBasisMatrix,temp_constraints,&temp_indices_to_constraint_B[temp_indices[start_constraint]+s+1],1,&j,work,INSERT_VALUES);CHKERRQ(ierr);
            if (useksp) {
              /* temp mat with transposed rows and columns */
              ierr = MatSetValues(temp_mat,1,&s,temp_constraints,&is_indices[1],work,INSERT_VALUES);CHKERRQ(ierr);
              ierr = MatSetValue(temp_mat,is_indices[0],is_indices[0],1.0,INSERT_VALUES);CHKERRQ(ierr);
            }
          }
          if (useksp) {
            /* last rows of temp_mat */
            for (j=0;j<size_of_constraint;j++) {
              is_indices[j] = j;
            }
            for (s=0;s<temp_constraints;s++) {
              k = s + dual_dofs;
              ierr = MatSetValues(temp_mat,1,&k,size_of_constraint,is_indices,&temp_quadrature_constraint[temp_indices[start_constraint+s]],INSERT_VALUES);CHKERRQ(ierr);
            }
            ierr = MatAssemblyBegin(temp_mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
            ierr = MatAssemblyEnd(temp_mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
            ierr = MatGetVecs(temp_mat,&temp_vec,NULL);CHKERRQ(ierr);
            ierr = KSPCreate(PETSC_COMM_SELF,&temp_ksp);CHKERRQ(ierr);
            ierr = KSPSetOperators(temp_ksp,temp_mat,temp_mat,SAME_PRECONDITIONER);CHKERRQ(ierr);
            ierr = KSPSetType(temp_ksp,KSPPREONLY);CHKERRQ(ierr);
            ierr = KSPGetPC(temp_ksp,&temp_pc);CHKERRQ(ierr);
            ierr = PCSetType(temp_pc,PCLU);CHKERRQ(ierr);
            ierr = KSPSetUp(temp_ksp);CHKERRQ(ierr);
            for (s=0;s<temp_constraints;s++) {
              ierr = VecSet(temp_vec,0.0);CHKERRQ(ierr);
              ierr = VecSetValue(temp_vec,s+dual_dofs,1.0,INSERT_VALUES);CHKERRQ(ierr);
              ierr = VecAssemblyBegin(temp_vec);CHKERRQ(ierr);
              ierr = VecAssemblyEnd(temp_vec);CHKERRQ(ierr);
              ierr = KSPSolve(temp_ksp,temp_vec,temp_vec);CHKERRQ(ierr);
              ierr = VecGetArray(temp_vec,&array_vector);CHKERRQ(ierr);
              j = temp_indices_to_constraint_B[temp_indices[start_constraint+s]+size_of_constraint-s-1];
              /* last columns of change of basis matrix associated to new primal dofs */
              ierr = MatSetValues(pcbddc->ChangeOfBasisMatrix,size_of_constraint,&temp_indices_to_constraint_B[temp_indices[start_constraint+s]],1,&j,array_vector,INSERT_VALUES);CHKERRQ(ierr);
              ierr = VecRestoreArray(temp_vec,&array_vector);CHKERRQ(ierr);
            }
            ierr = MatDestroy(&temp_mat);CHKERRQ(ierr);
            ierr = KSPDestroy(&temp_ksp);CHKERRQ(ierr);
            ierr = VecDestroy(&temp_vec);CHKERRQ(ierr);
          } else {
            /* last columns of change of basis matrix associated to new primal dofs */
            for (s=0;s<temp_constraints;s++) {
              j = temp_indices_to_constraint_B[temp_indices[start_constraint+s]+size_of_constraint-s-1];
              ierr = MatSetValues(pcbddc->ChangeOfBasisMatrix,size_of_constraint,&temp_indices_to_constraint_B[temp_indices[start_constraint+s]],1,&j,&temp_quadrature_constraint[temp_indices[start_constraint+s]],INSERT_VALUES);CHKERRQ(ierr);
            }
          }
          /* prepare for the next cycle */
          temp_constraints = 0;
          if (i != local_primal_size -1 ) {
            temp_start_ptr = temp_indices_to_constraint_B[temp_indices[i+1]];
          }
        }
      }
    }
    /* assembling */
    ierr = MatAssemblyBegin(pcbddc->ChangeOfBasisMatrix,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(pcbddc->ChangeOfBasisMatrix,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = PetscFree(ipiv);CHKERRQ(ierr);
    ierr = PetscFree(is_indices);CHKERRQ(ierr);
  }
  /* free workspace no longer needed */
  ierr = PetscFree(rwork);CHKERRQ(ierr);
  ierr = PetscFree(work);CHKERRQ(ierr);
  ierr = PetscFree(temp_basis);CHKERRQ(ierr);
  ierr = PetscFree(singular_vals);CHKERRQ(ierr);
  ierr = PetscFree(correlation_mat);CHKERRQ(ierr);
  ierr = PetscFree(temp_indices);CHKERRQ(ierr);
  ierr = PetscFree(change_basis);CHKERRQ(ierr);
  ierr = PetscFree(temp_indices_to_constraint);CHKERRQ(ierr);
  ierr = PetscFree(temp_indices_to_constraint_B);CHKERRQ(ierr);
  ierr = PetscFree(local_to_B);CHKERRQ(ierr);
  ierr = PetscFree(temp_quadrature_constraint);CHKERRQ(ierr);
#if defined(PETSC_MISSING_LAPACK_GESVD)
  ierr = PetscFree(iwork);CHKERRQ(ierr);
  ierr = PetscFree(ifail);CHKERRQ(ierr);
  ierr = PetscFree(singular_vectors);CHKERRQ(ierr);
#endif
  for (k=0;k<nnsp_size;k++) {
    ierr = VecDestroy(&localnearnullsp[k]);CHKERRQ(ierr);
  }
  ierr = PetscFree(localnearnullsp);CHKERRQ(ierr);
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
  *n_vertices = n;
  *vertices_idx = vertices;
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
    ierr = PetscMalloc(n*sizeof(PetscInt),&constraints_index);CHKERRQ(ierr);
    ierr = PetscMalloc(max_size_of_constraint*sizeof(PetscInt),&row_cmat_global_indices);CHKERRQ(ierr);
    ierr = PetscMalloc(local_size*sizeof(PetscBool),&touched);CHKERRQ(ierr);
    ierr = PetscMemzero(touched,local_size*sizeof(PetscBool));CHKERRQ(ierr);
    n = 0;
    for (i=0;i<local_primal_size;i++) {
      ierr = MatGetRow(pcbddc->ConstraintMatrix,i,&size_of_constraint,(const PetscInt**)&row_cmat_indices,NULL);CHKERRQ(ierr);
      if (size_of_constraint > 1) {
        ierr = ISLocalToGlobalMappingApply(pcbddc->mat_graph->l2gmap,size_of_constraint,row_cmat_indices,row_cmat_global_indices);CHKERRQ(ierr);
        min_index = row_cmat_global_indices[0];
        min_loc = 0;
        for (j=1;j<size_of_constraint;j++) {
          /* there can be more than one constraint on a single connected component */
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
  }
  ierr = PetscFree(touched);CHKERRQ(ierr);
  ierr = PetscFree(row_cmat_global_indices);CHKERRQ(ierr);
  *n_constraints = n;
  *constraints_idx = constraints_index;
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
