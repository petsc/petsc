#include <petsc/private/pcbddcimpl.h>
#include <petsc/private/pcbddcprivateimpl.h>
#include <petscblaslapack.h>

static PetscErrorCode MatMult_BDdelta_deluxe_nonred(Mat A, Vec x, Vec y)
{
  BDdelta_DN     ctx;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(A,&ctx));
  PetscCall(MatMultTranspose(ctx->BD,x,ctx->work));
  PetscCall(KSPSolveTranspose(ctx->kBD,ctx->work,y));
  /* No PC so cannot propagate up failure in KSPSolveTranspose() */
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMultTranspose_BDdelta_deluxe_nonred(Mat A, Vec x, Vec y)
{
  BDdelta_DN     ctx;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(A,&ctx));
  PetscCall(KSPSolve(ctx->kBD,x,ctx->work));
  /* No PC so cannot propagate up failure in KSPSolve() */
  PetscCall(MatMult(ctx->BD,ctx->work,y));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDestroy_BDdelta_deluxe_nonred(Mat A)
{
  BDdelta_DN     ctx;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(A,&ctx));
  PetscCall(MatDestroy(&ctx->BD));
  PetscCall(KSPDestroy(&ctx->kBD));
  PetscCall(VecDestroy(&ctx->work));
  PetscCall(PetscFree(ctx));
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCCreateFETIDPMatContext(PC pc, FETIDPMat_ctx *fetidpmat_ctx)
{
  FETIDPMat_ctx  newctx;

  PetscFunctionBegin;
  PetscCall(PetscNew(&newctx));
  /* increase the reference count for BDDC preconditioner */
  PetscCall(PetscObjectReference((PetscObject)pc));
  newctx->pc              = pc;
  *fetidpmat_ctx          = newctx;
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCCreateFETIDPPCContext(PC pc, FETIDPPC_ctx *fetidppc_ctx)
{
  FETIDPPC_ctx   newctx;

  PetscFunctionBegin;
  PetscCall(PetscNew(&newctx));
  /* increase the reference count for BDDC preconditioner */
  PetscCall(PetscObjectReference((PetscObject)pc));
  newctx->pc              = pc;
  *fetidppc_ctx           = newctx;
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCDestroyFETIDPMat(Mat A)
{
  FETIDPMat_ctx  mat_ctx;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(A,&mat_ctx));
  PetscCall(VecDestroy(&mat_ctx->lambda_local));
  PetscCall(VecDestroy(&mat_ctx->temp_solution_D));
  PetscCall(VecDestroy(&mat_ctx->temp_solution_B));
  PetscCall(MatDestroy(&mat_ctx->B_delta));
  PetscCall(MatDestroy(&mat_ctx->B_Ddelta));
  PetscCall(MatDestroy(&mat_ctx->B_BB));
  PetscCall(MatDestroy(&mat_ctx->B_BI));
  PetscCall(MatDestroy(&mat_ctx->Bt_BB));
  PetscCall(MatDestroy(&mat_ctx->Bt_BI));
  PetscCall(MatDestroy(&mat_ctx->C));
  PetscCall(VecDestroy(&mat_ctx->rhs_flip));
  PetscCall(VecDestroy(&mat_ctx->vP));
  PetscCall(VecDestroy(&mat_ctx->xPg));
  PetscCall(VecDestroy(&mat_ctx->yPg));
  PetscCall(VecScatterDestroy(&mat_ctx->l2g_lambda));
  PetscCall(VecScatterDestroy(&mat_ctx->l2g_lambda_only));
  PetscCall(VecScatterDestroy(&mat_ctx->l2g_p));
  PetscCall(VecScatterDestroy(&mat_ctx->g2g_p));
  PetscCall(PCDestroy(&mat_ctx->pc)); /* decrease PCBDDC reference count */
  PetscCall(ISDestroy(&mat_ctx->pressure));
  PetscCall(ISDestroy(&mat_ctx->lagrange));
  PetscCall(PetscFree(mat_ctx));
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCDestroyFETIDPPC(PC pc)
{
  FETIDPPC_ctx   pc_ctx;

  PetscFunctionBegin;
  PetscCall(PCShellGetContext(pc,&pc_ctx));
  PetscCall(VecDestroy(&pc_ctx->lambda_local));
  PetscCall(MatDestroy(&pc_ctx->B_Ddelta));
  PetscCall(VecScatterDestroy(&pc_ctx->l2g_lambda));
  PetscCall(MatDestroy(&pc_ctx->S_j));
  PetscCall(PCDestroy(&pc_ctx->pc)); /* decrease PCBDDC reference count */
  PetscCall(VecDestroy(&pc_ctx->xPg));
  PetscCall(VecDestroy(&pc_ctx->yPg));
  PetscCall(PetscFree(pc_ctx));
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCSetupFETIDPMatContext(FETIDPMat_ctx fetidpmat_ctx)
{
  PC_IS          *pcis=(PC_IS*)fetidpmat_ctx->pc->data;
  PC_BDDC        *pcbddc=(PC_BDDC*)fetidpmat_ctx->pc->data;
  PCBDDCGraph    mat_graph=pcbddc->mat_graph;
  Mat_IS         *matis  = (Mat_IS*)fetidpmat_ctx->pc->pmat->data;
  MPI_Comm       comm;
  Mat            ScalingMat,BD1,BD2;
  Vec            fetidp_global;
  IS             IS_l2g_lambda;
  IS             subset,subset_mult,subset_n,isvert;
  PetscBool      skip_node,fully_redundant;
  PetscInt       i,j,k,s,n_boundary_dofs,n_global_lambda,n_vertices,partial_sum;
  PetscInt       cum,n_local_lambda,n_lambda_for_dof,dual_size,n_neg_values,n_pos_values;
  PetscMPIInt    rank,size,buf_size,neigh;
  PetscScalar    scalar_value;
  const PetscInt *vertex_indices;
  PetscInt       *dual_dofs_boundary_indices,*aux_local_numbering_1;
  const PetscInt *aux_global_numbering;
  PetscInt       *aux_sums,*cols_B_delta,*l2g_indices;
  PetscScalar    *array,*scaling_factors,*vals_B_delta;
  PetscScalar    **all_factors;
  PetscInt       *aux_local_numbering_2;
  PetscLayout    llay;

  /* saddlepoint */
  ISLocalToGlobalMapping l2gmap_p;
  PetscLayout            play;
  IS                     gP,pP;
  PetscInt               nPl,nPg,nPgl;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)(fetidpmat_ctx->pc),&comm));
  PetscCallMPI(MPI_Comm_rank(comm,&rank));
  PetscCallMPI(MPI_Comm_size(comm,&size));

  /* saddlepoint */
  nPl      = 0;
  nPg      = 0;
  nPgl     = 0;
  gP       = NULL;
  pP       = NULL;
  l2gmap_p = NULL;
  play     = NULL;
  PetscCall(PetscObjectQuery((PetscObject)fetidpmat_ctx->pc,"__KSPFETIDP_pP",(PetscObject*)&pP));
  if (pP) { /* saddle point */
    /* subdomain pressures in global numbering */
    PetscCall(PetscObjectQuery((PetscObject)fetidpmat_ctx->pc,"__KSPFETIDP_gP",(PetscObject*)&gP));
    PetscCheck(gP,PETSC_COMM_SELF,PETSC_ERR_PLIB,"gP not present");
    PetscCall(ISGetLocalSize(gP,&nPl));
    PetscCall(VecCreate(PETSC_COMM_SELF,&fetidpmat_ctx->vP));
    PetscCall(VecSetSizes(fetidpmat_ctx->vP,nPl,nPl));
    PetscCall(VecSetType(fetidpmat_ctx->vP,VECSTANDARD));
    PetscCall(VecSetUp(fetidpmat_ctx->vP));

    /* pressure matrix */
    PetscCall(PetscObjectQuery((PetscObject)fetidpmat_ctx->pc,"__KSPFETIDP_C",(PetscObject*)&fetidpmat_ctx->C));
    if (!fetidpmat_ctx->C) { /* null pressure block, compute layout and global numbering for pressures */
      IS Pg;

      PetscCall(ISRenumber(gP,NULL,&nPg,&Pg));
      PetscCall(ISLocalToGlobalMappingCreateIS(Pg,&l2gmap_p));
      PetscCall(ISDestroy(&Pg));
      PetscCall(PetscLayoutCreate(comm,&play));
      PetscCall(PetscLayoutSetBlockSize(play,1));
      PetscCall(PetscLayoutSetSize(play,nPg));
      PetscCall(ISGetLocalSize(pP,&nPgl));
      PetscCall(PetscLayoutSetLocalSize(play,nPgl));
      PetscCall(PetscLayoutSetUp(play));
    } else {
      PetscCall(PetscObjectReference((PetscObject)fetidpmat_ctx->C));
      PetscCall(MatISGetLocalToGlobalMapping(fetidpmat_ctx->C,&l2gmap_p,NULL));
      PetscCall(PetscObjectReference((PetscObject)l2gmap_p));
      PetscCall(MatGetSize(fetidpmat_ctx->C,&nPg,NULL));
      PetscCall(MatGetLocalSize(fetidpmat_ctx->C,NULL,&nPgl));
      PetscCall(MatGetLayouts(fetidpmat_ctx->C,NULL,&llay));
      PetscCall(PetscLayoutReference(llay,&play));
    }
    PetscCall(VecCreateMPIWithArray(comm,1,nPgl,nPg,NULL,&fetidpmat_ctx->xPg));
    PetscCall(VecCreateMPIWithArray(comm,1,nPgl,nPg,NULL,&fetidpmat_ctx->yPg));

    /* import matrices for pressures coupling */
    PetscCall(PetscObjectQuery((PetscObject)fetidpmat_ctx->pc,"__KSPFETIDP_B_BI",(PetscObject*)&fetidpmat_ctx->B_BI));
    PetscCheck(fetidpmat_ctx->B_BI,PETSC_COMM_SELF,PETSC_ERR_PLIB,"B_BI not present");
    PetscCall(PetscObjectReference((PetscObject)fetidpmat_ctx->B_BI));

    PetscCall(PetscObjectQuery((PetscObject)fetidpmat_ctx->pc,"__KSPFETIDP_B_BB",(PetscObject*)&fetidpmat_ctx->B_BB));
    PetscCheck(fetidpmat_ctx->B_BB,PETSC_COMM_SELF,PETSC_ERR_PLIB,"B_BB not present");
    PetscCall(PetscObjectReference((PetscObject)fetidpmat_ctx->B_BB));

    PetscCall(PetscObjectQuery((PetscObject)fetidpmat_ctx->pc,"__KSPFETIDP_Bt_BI",(PetscObject*)&fetidpmat_ctx->Bt_BI));
    PetscCheck(fetidpmat_ctx->Bt_BI,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Bt_BI not present");
    PetscCall(PetscObjectReference((PetscObject)fetidpmat_ctx->Bt_BI));

    PetscCall(PetscObjectQuery((PetscObject)fetidpmat_ctx->pc,"__KSPFETIDP_Bt_BB",(PetscObject*)&fetidpmat_ctx->Bt_BB));
    PetscCheck(fetidpmat_ctx->Bt_BB,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Bt_BB not present");
    PetscCall(PetscObjectReference((PetscObject)fetidpmat_ctx->Bt_BB));

    PetscCall(PetscObjectQuery((PetscObject)fetidpmat_ctx->pc,"__KSPFETIDP_flip" ,(PetscObject*)&fetidpmat_ctx->rhs_flip));
    if (fetidpmat_ctx->rhs_flip) {
      PetscCall(PetscObjectReference((PetscObject)fetidpmat_ctx->rhs_flip));
    }
  }

  /* Default type of lagrange multipliers is non-redundant */
  fully_redundant = fetidpmat_ctx->fully_redundant;

  /* Evaluate local and global number of lagrange multipliers */
  PetscCall(VecSet(pcis->vec1_N,0.0));
  n_local_lambda = 0;
  partial_sum = 0;
  n_boundary_dofs = 0;
  s = 0;

  /* Get Vertices used to define the BDDC */
  PetscCall(PCBDDCGraphGetCandidatesIS(pcbddc->mat_graph,NULL,NULL,NULL,NULL,&isvert));
  PetscCall(ISGetLocalSize(isvert,&n_vertices));
  PetscCall(ISGetIndices(isvert,&vertex_indices));

  dual_size = pcis->n_B-n_vertices;
  PetscCall(PetscMalloc1(dual_size,&dual_dofs_boundary_indices));
  PetscCall(PetscMalloc1(dual_size,&aux_local_numbering_1));
  PetscCall(PetscMalloc1(dual_size,&aux_local_numbering_2));

  PetscCall(VecGetArray(pcis->vec1_N,&array));
  for (i=0;i<pcis->n;i++) {
    j = mat_graph->count[i]; /* RECALL: mat_graph->count[i] does not count myself */
    if (j > 0) n_boundary_dofs++;
    skip_node = PETSC_FALSE;
    if (s < n_vertices && vertex_indices[s] == i) { /* it works for a sorted set of vertices */
      skip_node = PETSC_TRUE;
      s++;
    }
    if (j < 1) skip_node = PETSC_TRUE;
    if (mat_graph->special_dof[i] == PCBDDCGRAPH_DIRICHLET_MARK) skip_node = PETSC_TRUE;
    if (!skip_node) {
      if (fully_redundant) {
        /* fully redundant set of lagrange multipliers */
        n_lambda_for_dof = (j*(j+1))/2;
      } else {
        n_lambda_for_dof = j;
      }
      n_local_lambda += j;
      /* needed to evaluate global number of lagrange multipliers */
      array[i]=(1.0*n_lambda_for_dof)/(j+1.0); /* already scaled for the next global sum */
      /* store some data needed */
      dual_dofs_boundary_indices[partial_sum] = n_boundary_dofs-1;
      aux_local_numbering_1[partial_sum] = i;
      aux_local_numbering_2[partial_sum] = n_lambda_for_dof;
      partial_sum++;
    }
  }
  PetscCall(VecRestoreArray(pcis->vec1_N,&array));
  PetscCall(ISRestoreIndices(isvert,&vertex_indices));
  PetscCall(PCBDDCGraphRestoreCandidatesIS(pcbddc->mat_graph,NULL,NULL,NULL,NULL,&isvert));
  dual_size = partial_sum;

  /* compute global ordering of lagrange multipliers and associate l2g map */
  PetscCall(ISCreateGeneral(comm,partial_sum,aux_local_numbering_1,PETSC_COPY_VALUES,&subset_n));
  PetscCall(ISLocalToGlobalMappingApplyIS(pcis->mapping,subset_n,&subset));
  PetscCall(ISDestroy(&subset_n));
  PetscCall(ISCreateGeneral(comm,partial_sum,aux_local_numbering_2,PETSC_OWN_POINTER,&subset_mult));
  PetscCall(ISRenumber(subset,subset_mult,&fetidpmat_ctx->n_lambda,&subset_n));
  PetscCall(ISDestroy(&subset));

  if (PetscDefined(USE_DEBUG)) {
    PetscCall(VecSet(pcis->vec1_global,0.0));
    PetscCall(VecScatterBegin(matis->rctx,pcis->vec1_N,pcis->vec1_global,ADD_VALUES,SCATTER_REVERSE));
    PetscCall(VecScatterEnd(matis->rctx,pcis->vec1_N,pcis->vec1_global,ADD_VALUES,SCATTER_REVERSE));
    PetscCall(VecSum(pcis->vec1_global,&scalar_value));
    i = (PetscInt)PetscRealPart(scalar_value);
    PetscCheck(i == fetidpmat_ctx->n_lambda,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Global number of multipliers mismatch! (%" PetscInt_FMT " != %" PetscInt_FMT ")",fetidpmat_ctx->n_lambda,i);
  }

  /* init data for scaling factors exchange */
  if (!pcbddc->use_deluxe_scaling) {
    PetscInt    *ptrs_buffer,neigh_position;
    PetscScalar *send_buffer,*recv_buffer;
    MPI_Request *send_reqs,*recv_reqs;

    partial_sum = 0;
    PetscCall(PetscMalloc1(pcis->n_neigh,&ptrs_buffer));
    PetscCall(PetscMalloc1(PetscMax(pcis->n_neigh-1,0),&send_reqs));
    PetscCall(PetscMalloc1(PetscMax(pcis->n_neigh-1,0),&recv_reqs));
    PetscCall(PetscMalloc1(pcis->n+1,&all_factors));
    if (pcis->n_neigh > 0) ptrs_buffer[0]=0;
    for (i=1;i<pcis->n_neigh;i++) {
      partial_sum += pcis->n_shared[i];
      ptrs_buffer[i] = ptrs_buffer[i-1]+pcis->n_shared[i];
    }
    PetscCall(PetscMalloc1(partial_sum,&send_buffer));
    PetscCall(PetscMalloc1(partial_sum,&recv_buffer));
    PetscCall(PetscMalloc1(partial_sum,&all_factors[0]));
    for (i=0;i<pcis->n-1;i++) {
      j = mat_graph->count[i];
      all_factors[i+1]=all_factors[i]+j;
    }

    /* scatter B scaling to N vec */
    PetscCall(VecScatterBegin(pcis->N_to_B,pcis->D,pcis->vec1_N,INSERT_VALUES,SCATTER_REVERSE));
    PetscCall(VecScatterEnd(pcis->N_to_B,pcis->D,pcis->vec1_N,INSERT_VALUES,SCATTER_REVERSE));
    /* communications */
    PetscCall(VecGetArrayRead(pcis->vec1_N,(const PetscScalar**)&array));
    for (i=1;i<pcis->n_neigh;i++) {
      for (j=0;j<pcis->n_shared[i];j++) {
        send_buffer[ptrs_buffer[i-1]+j]=array[pcis->shared[i][j]];
      }
      PetscCall(PetscMPIIntCast(ptrs_buffer[i]-ptrs_buffer[i-1],&buf_size));
      PetscCall(PetscMPIIntCast(pcis->neigh[i],&neigh));
      PetscCallMPI(MPI_Isend(&send_buffer[ptrs_buffer[i-1]],buf_size,MPIU_SCALAR,neigh,0,comm,&send_reqs[i-1]));
      PetscCallMPI(MPI_Irecv(&recv_buffer[ptrs_buffer[i-1]],buf_size,MPIU_SCALAR,neigh,0,comm,&recv_reqs[i-1]));
    }
    PetscCall(VecRestoreArrayRead(pcis->vec1_N,(const PetscScalar**)&array));
    if (pcis->n_neigh > 0) {
      PetscCallMPI(MPI_Waitall(pcis->n_neigh-1,recv_reqs,MPI_STATUSES_IGNORE));
    }
    /* put values in correct places */
    for (i=1;i<pcis->n_neigh;i++) {
      for (j=0;j<pcis->n_shared[i];j++) {
        k = pcis->shared[i][j];
        neigh_position = 0;
        while (mat_graph->neighbours_set[k][neigh_position] != pcis->neigh[i]) {neigh_position++;}
        all_factors[k][neigh_position]=recv_buffer[ptrs_buffer[i-1]+j];
      }
    }
    if (pcis->n_neigh > 0) {
      PetscCallMPI(MPI_Waitall(pcis->n_neigh-1,send_reqs,MPI_STATUSES_IGNORE));
    }
    PetscCall(PetscFree(send_reqs));
    PetscCall(PetscFree(recv_reqs));
    PetscCall(PetscFree(send_buffer));
    PetscCall(PetscFree(recv_buffer));
    PetscCall(PetscFree(ptrs_buffer));
  }

  /* Compute B and B_delta (local actions) */
  PetscCall(PetscMalloc1(pcis->n_neigh,&aux_sums));
  PetscCall(PetscMalloc1(n_local_lambda,&l2g_indices));
  PetscCall(PetscMalloc1(n_local_lambda,&vals_B_delta));
  PetscCall(PetscMalloc1(n_local_lambda,&cols_B_delta));
  if (!pcbddc->use_deluxe_scaling) {
    PetscCall(PetscMalloc1(n_local_lambda,&scaling_factors));
  } else {
    scaling_factors = NULL;
    all_factors     = NULL;
  }
  PetscCall(ISGetIndices(subset_n,&aux_global_numbering));
  partial_sum=0;
  cum = 0;
  for (i=0;i<dual_size;i++) {
    n_global_lambda = aux_global_numbering[cum];
    j = mat_graph->count[aux_local_numbering_1[i]];
    aux_sums[0]=0;
    for (s=1;s<j;s++) {
      aux_sums[s]=aux_sums[s-1]+j-s+1;
    }
    if (all_factors) array = all_factors[aux_local_numbering_1[i]];
    n_neg_values = 0;
    while (n_neg_values < j && mat_graph->neighbours_set[aux_local_numbering_1[i]][n_neg_values] < rank) {n_neg_values++;}
    n_pos_values = j - n_neg_values;
    if (fully_redundant) {
      for (s=0;s<n_neg_values;s++) {
        l2g_indices    [partial_sum+s]=aux_sums[s]+n_neg_values-s-1+n_global_lambda;
        cols_B_delta   [partial_sum+s]=dual_dofs_boundary_indices[i];
        vals_B_delta   [partial_sum+s]=-1.0;
        if (!pcbddc->use_deluxe_scaling) scaling_factors[partial_sum+s]=array[s];
      }
      for (s=0;s<n_pos_values;s++) {
        l2g_indices    [partial_sum+s+n_neg_values]=aux_sums[n_neg_values]+s+n_global_lambda;
        cols_B_delta   [partial_sum+s+n_neg_values]=dual_dofs_boundary_indices[i];
        vals_B_delta   [partial_sum+s+n_neg_values]=1.0;
        if (!pcbddc->use_deluxe_scaling) scaling_factors[partial_sum+s+n_neg_values]=array[s+n_neg_values];
      }
      partial_sum += j;
    } else {
      /* l2g_indices and default cols and vals of B_delta */
      for (s=0;s<j;s++) {
        l2g_indices    [partial_sum+s]=n_global_lambda+s;
        cols_B_delta   [partial_sum+s]=dual_dofs_boundary_indices[i];
        vals_B_delta   [partial_sum+s]=0.0;
      }
      /* B_delta */
      if (n_neg_values > 0) { /* there's a rank next to me to the left */
        vals_B_delta   [partial_sum+n_neg_values-1]=-1.0;
      }
      if (n_neg_values < j) { /* there's a rank next to me to the right */
        vals_B_delta   [partial_sum+n_neg_values]=1.0;
      }
      /* scaling as in Klawonn-Widlund 1999 */
      if (!pcbddc->use_deluxe_scaling) {
        for (s=0;s<n_neg_values;s++) {
          scalar_value = 0.0;
          for (k=0;k<s+1;k++) scalar_value += array[k];
          scaling_factors[partial_sum+s] = -scalar_value;
        }
        for (s=0;s<n_pos_values;s++) {
          scalar_value = 0.0;
          for (k=s+n_neg_values;k<j;k++) scalar_value += array[k];
          scaling_factors[partial_sum+s+n_neg_values] = scalar_value;
        }
      }
      partial_sum += j;
    }
    cum += aux_local_numbering_2[i];
  }
  PetscCall(ISRestoreIndices(subset_n,&aux_global_numbering));
  PetscCall(ISDestroy(&subset_mult));
  PetscCall(ISDestroy(&subset_n));
  PetscCall(PetscFree(aux_sums));
  PetscCall(PetscFree(aux_local_numbering_1));
  PetscCall(PetscFree(dual_dofs_boundary_indices));
  if (all_factors) {
    PetscCall(PetscFree(all_factors[0]));
    PetscCall(PetscFree(all_factors));
  }

  /* Create local part of B_delta */
  PetscCall(MatCreate(PETSC_COMM_SELF,&fetidpmat_ctx->B_delta));
  PetscCall(MatSetSizes(fetidpmat_ctx->B_delta,n_local_lambda,pcis->n_B,n_local_lambda,pcis->n_B));
  PetscCall(MatSetType(fetidpmat_ctx->B_delta,MATSEQAIJ));
  PetscCall(MatSeqAIJSetPreallocation(fetidpmat_ctx->B_delta,1,NULL));
  PetscCall(MatSetOption(fetidpmat_ctx->B_delta,MAT_IGNORE_ZERO_ENTRIES,PETSC_TRUE));
  for (i=0;i<n_local_lambda;i++) {
    PetscCall(MatSetValue(fetidpmat_ctx->B_delta,i,cols_B_delta[i],vals_B_delta[i],INSERT_VALUES));
  }
  PetscCall(PetscFree(vals_B_delta));
  PetscCall(MatAssemblyBegin(fetidpmat_ctx->B_delta,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(fetidpmat_ctx->B_delta,MAT_FINAL_ASSEMBLY));

  BD1 = NULL;
  BD2 = NULL;
  if (fully_redundant) {
    PetscCheck(!pcbddc->use_deluxe_scaling,comm,PETSC_ERR_SUP,"Deluxe FETIDP with fully-redundant multipliers to be implemented");
    PetscCall(MatCreate(PETSC_COMM_SELF,&ScalingMat));
    PetscCall(MatSetSizes(ScalingMat,n_local_lambda,n_local_lambda,n_local_lambda,n_local_lambda));
    PetscCall(MatSetType(ScalingMat,MATSEQAIJ));
    PetscCall(MatSeqAIJSetPreallocation(ScalingMat,1,NULL));
    for (i=0;i<n_local_lambda;i++) {
      PetscCall(MatSetValue(ScalingMat,i,i,scaling_factors[i],INSERT_VALUES));
    }
    PetscCall(MatAssemblyBegin(ScalingMat,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(ScalingMat,MAT_FINAL_ASSEMBLY));
    PetscCall(MatMatMult(ScalingMat,fetidpmat_ctx->B_delta,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&fetidpmat_ctx->B_Ddelta));
    PetscCall(MatDestroy(&ScalingMat));
  } else {
    PetscCall(MatCreate(PETSC_COMM_SELF,&fetidpmat_ctx->B_Ddelta));
    PetscCall(MatSetSizes(fetidpmat_ctx->B_Ddelta,n_local_lambda,pcis->n_B,n_local_lambda,pcis->n_B));
    if (!pcbddc->use_deluxe_scaling || !pcbddc->sub_schurs) {
      PetscCall(MatSetType(fetidpmat_ctx->B_Ddelta,MATSEQAIJ));
      PetscCall(MatSeqAIJSetPreallocation(fetidpmat_ctx->B_Ddelta,1,NULL));
      for (i=0;i<n_local_lambda;i++) {
        PetscCall(MatSetValue(fetidpmat_ctx->B_Ddelta,i,cols_B_delta[i],scaling_factors[i],INSERT_VALUES));
      }
      PetscCall(MatAssemblyBegin(fetidpmat_ctx->B_Ddelta,MAT_FINAL_ASSEMBLY));
      PetscCall(MatAssemblyEnd(fetidpmat_ctx->B_Ddelta,MAT_FINAL_ASSEMBLY));
    } else {
      /* scaling as in Klawonn-Widlund 1999 */
      PCBDDCDeluxeScaling deluxe_ctx = pcbddc->deluxe_ctx;
      PCBDDCSubSchurs     sub_schurs = pcbddc->sub_schurs;
      Mat                 T;
      PetscScalar         *W,lwork,*Bwork;
      const PetscInt      *idxs = NULL;
      PetscInt            cum,mss,*nnz;
      PetscBLASInt        *pivots,B_lwork,B_N,B_ierr;

      PetscCheck(pcbddc->deluxe_singlemat,comm,PETSC_ERR_USER,"Cannot compute B_Ddelta! rerun with -pc_bddc_deluxe_singlemat");
      mss  = 0;
      PetscCall(PetscCalloc1(pcis->n_B,&nnz));
      if (sub_schurs->is_Ej_all) {
        PetscCall(ISGetIndices(sub_schurs->is_Ej_all,&idxs));
        for (i=0,cum=0;i<sub_schurs->n_subs;i++) {
          PetscInt subset_size;

          PetscCall(ISGetLocalSize(sub_schurs->is_subs[i],&subset_size));
          for (j=0;j<subset_size;j++) nnz[idxs[j+cum]] = subset_size;
          mss  = PetscMax(mss,subset_size);
          cum += subset_size;
        }
      }
      PetscCall(MatCreate(PETSC_COMM_SELF,&T));
      PetscCall(MatSetSizes(T,pcis->n_B,pcis->n_B,pcis->n_B,pcis->n_B));
      PetscCall(MatSetType(T,MATSEQAIJ));
      PetscCall(MatSeqAIJSetPreallocation(T,0,nnz));
      PetscCall(PetscFree(nnz));

      /* workspace allocation */
      B_lwork = 0;
      if (mss) {
        PetscScalar dummy = 1;

        B_lwork = -1;
        PetscCall(PetscBLASIntCast(mss,&B_N));
        PetscCall(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
        PetscStackCallBLAS("LAPACKgetri",LAPACKgetri_(&B_N,&dummy,&B_N,&B_N,&lwork,&B_lwork,&B_ierr));
        PetscCall(PetscFPTrapPop());
        PetscCheck(!B_ierr,PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in query to GETRI Lapack routine %d",(int)B_ierr);
        PetscCall(PetscBLASIntCast((PetscInt)PetscRealPart(lwork),&B_lwork));
      }
      PetscCall(PetscMalloc3(mss*mss,&W,mss,&pivots,B_lwork,&Bwork));

      for (i=0,cum=0;i<sub_schurs->n_subs;i++) {
        const PetscScalar *M;
        PetscInt          subset_size;

        PetscCall(ISGetLocalSize(sub_schurs->is_subs[i],&subset_size));
        PetscCall(PetscBLASIntCast(subset_size,&B_N));
        PetscCall(MatDenseGetArrayRead(deluxe_ctx->seq_mat[i],&M));
        PetscCall(PetscArraycpy(W,M,subset_size*subset_size));
        PetscCall(MatDenseRestoreArrayRead(deluxe_ctx->seq_mat[i],&M));
        PetscCall(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
        PetscStackCallBLAS("LAPACKgetrf",LAPACKgetrf_(&B_N,&B_N,W,&B_N,pivots,&B_ierr));
        PetscCheck(!B_ierr,PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in GETRF Lapack routine %d",(int)B_ierr);
        PetscStackCallBLAS("LAPACKgetri",LAPACKgetri_(&B_N,W,&B_N,pivots,Bwork,&B_lwork,&B_ierr));
        PetscCheck(!B_ierr,PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in GETRI Lapack routine %d",(int)B_ierr);
        PetscCall(PetscFPTrapPop());
        /* silent static analyzer */
        PetscCheck(idxs,PETSC_COMM_SELF,PETSC_ERR_PLIB,"IDXS not present");
        PetscCall(MatSetValues(T,subset_size,idxs+cum,subset_size,idxs+cum,W,INSERT_VALUES));
        cum += subset_size;
      }
      PetscCall(MatAssemblyBegin(T,MAT_FINAL_ASSEMBLY));
      PetscCall(MatAssemblyEnd(T,MAT_FINAL_ASSEMBLY));
      PetscCall(MatMatTransposeMult(T,fetidpmat_ctx->B_delta,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&BD1));
      PetscCall(MatMatMult(fetidpmat_ctx->B_delta,BD1,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&BD2));
      PetscCall(MatDestroy(&T));
      PetscCall(PetscFree3(W,pivots,Bwork));
      if (sub_schurs->is_Ej_all) {
        PetscCall(ISRestoreIndices(sub_schurs->is_Ej_all,&idxs));
      }
    }
  }
  PetscCall(PetscFree(scaling_factors));
  PetscCall(PetscFree(cols_B_delta));

  /* Layout of multipliers */
  PetscCall(PetscLayoutCreate(comm,&llay));
  PetscCall(PetscLayoutSetBlockSize(llay,1));
  PetscCall(PetscLayoutSetSize(llay,fetidpmat_ctx->n_lambda));
  PetscCall(PetscLayoutSetUp(llay));
  PetscCall(PetscLayoutGetLocalSize(llay,&fetidpmat_ctx->n));

  /* Local work vector of multipliers */
  PetscCall(VecCreate(PETSC_COMM_SELF,&fetidpmat_ctx->lambda_local));
  PetscCall(VecSetSizes(fetidpmat_ctx->lambda_local,n_local_lambda,n_local_lambda));
  PetscCall(VecSetType(fetidpmat_ctx->lambda_local,VECSEQ));

  if (BD2) {
    ISLocalToGlobalMapping l2g;
    Mat                    T,TA,*pT;
    IS                     is;
    PetscInt               nl,N;
    BDdelta_DN             ctx;

    PetscCall(PetscLayoutGetLocalSize(llay,&nl));
    PetscCall(PetscLayoutGetSize(llay,&N));
    PetscCall(MatCreate(comm,&T));
    PetscCall(MatSetSizes(T,nl,nl,N,N));
    PetscCall(MatSetType(T,MATIS));
    PetscCall(ISLocalToGlobalMappingCreate(comm,1,n_local_lambda,l2g_indices,PETSC_COPY_VALUES,&l2g));
    PetscCall(MatSetLocalToGlobalMapping(T,l2g,l2g));
    PetscCall(ISLocalToGlobalMappingDestroy(&l2g));
    PetscCall(MatISSetLocalMat(T,BD2));
    PetscCall(MatAssemblyBegin(T,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(T,MAT_FINAL_ASSEMBLY));
    PetscCall(MatDestroy(&BD2));
    PetscCall(MatConvert(T,MATAIJ,MAT_INITIAL_MATRIX,&TA));
    PetscCall(MatDestroy(&T));
    PetscCall(ISCreateGeneral(comm,n_local_lambda,l2g_indices,PETSC_USE_POINTER,&is));
    PetscCall(MatCreateSubMatrices(TA,1,&is,&is,MAT_INITIAL_MATRIX,&pT));
    PetscCall(MatDestroy(&TA));
    PetscCall(ISDestroy(&is));
    BD2  = pT[0];
    PetscCall(PetscFree(pT));

    /* B_Ddelta for non-redundant multipliers with deluxe scaling */
    PetscCall(PetscNew(&ctx));
    PetscCall(MatSetType(fetidpmat_ctx->B_Ddelta,MATSHELL));
    PetscCall(MatShellSetContext(fetidpmat_ctx->B_Ddelta,ctx));
    PetscCall(MatShellSetOperation(fetidpmat_ctx->B_Ddelta,MATOP_MULT,(void (*)(void))MatMult_BDdelta_deluxe_nonred));
    PetscCall(MatShellSetOperation(fetidpmat_ctx->B_Ddelta,MATOP_MULT_TRANSPOSE,(void (*)(void))MatMultTranspose_BDdelta_deluxe_nonred));
    PetscCall(MatShellSetOperation(fetidpmat_ctx->B_Ddelta,MATOP_DESTROY,(void (*)(void))MatDestroy_BDdelta_deluxe_nonred));
    PetscCall(MatSetUp(fetidpmat_ctx->B_Ddelta));

    PetscCall(PetscObjectReference((PetscObject)BD1));
    ctx->BD = BD1;
    PetscCall(KSPCreate(PETSC_COMM_SELF,&ctx->kBD));
    PetscCall(KSPSetOperators(ctx->kBD,BD2,BD2));
    PetscCall(VecDuplicate(fetidpmat_ctx->lambda_local,&ctx->work));
    fetidpmat_ctx->deluxe_nonred = PETSC_TRUE;
  }
  PetscCall(MatDestroy(&BD1));
  PetscCall(MatDestroy(&BD2));

  /* fetidpmat sizes */
  fetidpmat_ctx->n += nPgl;
  fetidpmat_ctx->N  = fetidpmat_ctx->n_lambda+nPg;

  /* Global vector for FETI-DP linear system */
  PetscCall(VecCreate(comm,&fetidp_global));
  PetscCall(VecSetSizes(fetidp_global,fetidpmat_ctx->n,fetidpmat_ctx->N));
  PetscCall(VecSetType(fetidp_global,VECMPI));
  PetscCall(VecSetUp(fetidp_global));

  /* Decide layout for fetidp dofs: if it is a saddle point problem
     pressure is ordered first in the local part of the global vector
     of the FETI-DP linear system */
  if (nPg) {
    Vec            v;
    IS             IS_l2g_p,ais;
    PetscLayout    alay;
    const PetscInt *idxs,*pranges,*aranges,*lranges;
    PetscInt       *l2g_indices_p,rst;
    PetscMPIInt    rank;

    PetscCall(PetscMalloc1(nPl,&l2g_indices_p));
    PetscCall(VecGetLayout(fetidp_global,&alay));
    PetscCall(PetscLayoutGetRanges(alay,&aranges));
    PetscCall(PetscLayoutGetRanges(play,&pranges));
    PetscCall(PetscLayoutGetRanges(llay,&lranges));

    PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)fetidp_global),&rank));
    PetscCall(ISCreateStride(PetscObjectComm((PetscObject)fetidp_global),pranges[rank+1]-pranges[rank],aranges[rank],1,&fetidpmat_ctx->pressure));
    PetscCall(PetscObjectSetName((PetscObject)fetidpmat_ctx->pressure,"F_P"));
    PetscCall(ISCreateStride(PetscObjectComm((PetscObject)fetidp_global),lranges[rank+1]-lranges[rank],aranges[rank]+pranges[rank+1]-pranges[rank],1,&fetidpmat_ctx->lagrange));
    PetscCall(PetscObjectSetName((PetscObject)fetidpmat_ctx->lagrange,"F_L"));
    PetscCall(ISLocalToGlobalMappingGetIndices(l2gmap_p,&idxs));
    /* shift local to global indices for pressure */
    for (i=0;i<nPl;i++) {
      PetscMPIInt owner;

      PetscCall(PetscLayoutFindOwner(play,idxs[i],&owner));
      l2g_indices_p[i] = idxs[i]-pranges[owner]+aranges[owner];
    }
    PetscCall(ISLocalToGlobalMappingRestoreIndices(l2gmap_p,&idxs));
    PetscCall(ISCreateGeneral(comm,nPl,l2g_indices_p,PETSC_OWN_POINTER,&IS_l2g_p));

    /* local to global scatter for pressure */
    PetscCall(VecScatterCreate(fetidpmat_ctx->vP,NULL,fetidp_global,IS_l2g_p,&fetidpmat_ctx->l2g_p));
    PetscCall(ISDestroy(&IS_l2g_p));

    /* scatter for lagrange multipliers only */
    PetscCall(VecCreate(comm,&v));
    PetscCall(VecSetType(v,VECSTANDARD));
    PetscCall(VecSetLayout(v,llay));
    PetscCall(VecSetUp(v));
    PetscCall(ISCreateGeneral(comm,n_local_lambda,l2g_indices,PETSC_COPY_VALUES,&ais));
    PetscCall(VecScatterCreate(fetidpmat_ctx->lambda_local,NULL,v,ais,&fetidpmat_ctx->l2g_lambda_only));
    PetscCall(ISDestroy(&ais));
    PetscCall(VecDestroy(&v));

    /* shift local to global indices for multipliers */
    for (i=0;i<n_local_lambda;i++) {
      PetscInt    ps;
      PetscMPIInt owner;

      PetscCall(PetscLayoutFindOwner(llay,l2g_indices[i],&owner));
      ps = pranges[owner+1]-pranges[owner];
      l2g_indices[i] = l2g_indices[i]-lranges[owner]+aranges[owner]+ps;
    }

    /* scatter from alldofs to pressures global fetidp vector */
    PetscCall(PetscLayoutGetRange(alay,&rst,NULL));
    PetscCall(ISCreateStride(comm,nPgl,rst,1,&ais));
    PetscCall(VecScatterCreate(pcis->vec1_global,pP,fetidp_global,ais,&fetidpmat_ctx->g2g_p));
    PetscCall(ISDestroy(&ais));
  }
  PetscCall(PetscLayoutDestroy(&llay));
  PetscCall(PetscLayoutDestroy(&play));
  PetscCall(ISCreateGeneral(comm,n_local_lambda,l2g_indices,PETSC_OWN_POINTER,&IS_l2g_lambda));

  /* scatter from local to global multipliers */
  PetscCall(VecScatterCreate(fetidpmat_ctx->lambda_local,NULL,fetidp_global,IS_l2g_lambda,&fetidpmat_ctx->l2g_lambda));
  PetscCall(ISDestroy(&IS_l2g_lambda));
  PetscCall(ISLocalToGlobalMappingDestroy(&l2gmap_p));
  PetscCall(VecDestroy(&fetidp_global));

  /* Create some work vectors needed by fetidp */
  PetscCall(VecDuplicate(pcis->vec1_B,&fetidpmat_ctx->temp_solution_B));
  PetscCall(VecDuplicate(pcis->vec1_D,&fetidpmat_ctx->temp_solution_D));
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCSetupFETIDPPCContext(Mat fetimat, FETIDPPC_ctx fetidppc_ctx)
{
  FETIDPMat_ctx  mat_ctx;
  PC_BDDC        *pcbddc = (PC_BDDC*)fetidppc_ctx->pc->data;
  PC_IS          *pcis = (PC_IS*)fetidppc_ctx->pc->data;
  PetscBool      lumped = PETSC_FALSE;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(fetimat,&mat_ctx));
  /* get references from objects created when setting up feti mat context */
  PetscCall(PetscObjectReference((PetscObject)mat_ctx->lambda_local));
  fetidppc_ctx->lambda_local = mat_ctx->lambda_local;
  PetscCall(PetscObjectReference((PetscObject)mat_ctx->B_Ddelta));
  fetidppc_ctx->B_Ddelta = mat_ctx->B_Ddelta;
  if (mat_ctx->deluxe_nonred) {
    PC               pc,mpc;
    BDdelta_DN       ctx;
    MatSolverType    solver;
    const char       *prefix;

    PetscCall(MatShellGetContext(mat_ctx->B_Ddelta,&ctx));
    PetscCall(KSPSetType(ctx->kBD,KSPPREONLY));
    PetscCall(KSPGetPC(ctx->kBD,&mpc));
    PetscCall(KSPGetPC(pcbddc->ksp_D,&pc));
    PetscCall(PCSetType(mpc,PCLU));
    PetscCall(PCFactorGetMatSolverType(pc,(MatSolverType*)&solver));
    if (solver) {
      PetscCall(PCFactorSetMatSolverType(mpc,solver));
    }
    PetscCall(MatGetOptionsPrefix(fetimat,&prefix));
    PetscCall(KSPSetOptionsPrefix(ctx->kBD,prefix));
    PetscCall(KSPAppendOptionsPrefix(ctx->kBD,"bddelta_"));
    PetscCall(KSPSetFromOptions(ctx->kBD));
  }

  if (mat_ctx->l2g_lambda_only) {
    PetscCall(PetscObjectReference((PetscObject)mat_ctx->l2g_lambda_only));
    fetidppc_ctx->l2g_lambda = mat_ctx->l2g_lambda_only;
  } else {
    PetscCall(PetscObjectReference((PetscObject)mat_ctx->l2g_lambda));
    fetidppc_ctx->l2g_lambda = mat_ctx->l2g_lambda;
  }
  /* Dirichlet preconditioner */
  PetscCall(PetscOptionsGetBool(NULL,((PetscObject)fetimat)->prefix,"-pc_lumped",&lumped,NULL));
  if (!lumped) {
    IS        iV;
    PetscBool discrete_harmonic = PETSC_FALSE;

    PetscCall(PetscObjectQuery((PetscObject)fetidppc_ctx->pc,"__KSPFETIDP_iV",(PetscObject*)&iV));
    if (iV) {
      PetscCall(PetscOptionsGetBool(NULL,((PetscObject)fetimat)->prefix,"-pc_discrete_harmonic",&discrete_harmonic,NULL));
    }
    if (discrete_harmonic) {
      KSP             sksp;
      PC              pc;
      PCBDDCSubSchurs sub_schurs = pcbddc->sub_schurs;
      Mat             A_II,A_IB,A_BI;
      IS              iP = NULL;
      PetscBool       isshell,reuse = PETSC_FALSE;
      KSPType         ksptype;
      const char      *prefix;

      /*
        We constructs a Schur complement for

        | A_II A_ID |
        | A_DI A_DD |

        instead of

        | A_II  B^t_II A_ID |
        | B_II -C_II   B_ID |
        | A_DI  B^t_ID A_DD |

      */
      if (sub_schurs && sub_schurs->reuse_solver) {
        PetscCall(PetscObjectQuery((PetscObject)sub_schurs->A,"__KSPFETIDP_iP",(PetscObject*)&iP));
        if (iP) reuse = PETSC_TRUE;
      }
      if (!reuse) {
        IS       aB;
        PetscInt nb;
        PetscCall(ISGetLocalSize(pcis->is_B_local,&nb));
        PetscCall(ISCreateStride(PetscObjectComm((PetscObject)pcis->A_II),nb,0,1,&aB));
        PetscCall(MatCreateSubMatrix(pcis->A_II,iV,iV,MAT_INITIAL_MATRIX,&A_II));
        PetscCall(MatCreateSubMatrix(pcis->A_IB,iV,aB,MAT_INITIAL_MATRIX,&A_IB));
        PetscCall(MatCreateSubMatrix(pcis->A_BI,aB,iV,MAT_INITIAL_MATRIX,&A_BI));
        PetscCall(ISDestroy(&aB));
      } else {
        PetscCall(MatCreateSubMatrix(sub_schurs->A,pcis->is_I_local,pcis->is_B_local,MAT_INITIAL_MATRIX,&A_IB));
        PetscCall(MatCreateSubMatrix(sub_schurs->A,pcis->is_B_local,pcis->is_I_local,MAT_INITIAL_MATRIX,&A_BI));
        PetscCall(PetscObjectReference((PetscObject)pcis->A_II));
        A_II = pcis->A_II;
      }
      PetscCall(MatCreateSchurComplement(A_II,A_II,A_IB,A_BI,pcis->A_BB,&fetidppc_ctx->S_j));

      /* propagate settings of solver */
      PetscCall(MatSchurComplementGetKSP(fetidppc_ctx->S_j,&sksp));
      PetscCall(KSPGetType(pcis->ksp_D,&ksptype));
      PetscCall(KSPSetType(sksp,ksptype));
      PetscCall(KSPGetPC(pcis->ksp_D,&pc));
      PetscCall(PetscObjectTypeCompare((PetscObject)pc,PCSHELL,&isshell));
      if (!isshell) {
        MatSolverType    solver;
        PCType           pctype;

        PetscCall(PCGetType(pc,&pctype));
        PetscCall(PCFactorGetMatSolverType(pc,(MatSolverType*)&solver));
        PetscCall(KSPGetPC(sksp,&pc));
        PetscCall(PCSetType(pc,pctype));
        if (solver) {
          PetscCall(PCFactorSetMatSolverType(pc,solver));
        }
      } else {
        PetscCall(KSPGetPC(sksp,&pc));
        PetscCall(PCSetType(pc,PCLU));
      }
      PetscCall(MatDestroy(&A_II));
      PetscCall(MatDestroy(&A_IB));
      PetscCall(MatDestroy(&A_BI));
      PetscCall(MatGetOptionsPrefix(fetimat,&prefix));
      PetscCall(KSPSetOptionsPrefix(sksp,prefix));
      PetscCall(KSPAppendOptionsPrefix(sksp,"harmonic_"));
      PetscCall(KSPSetFromOptions(sksp));
      if (reuse) {
        PetscCall(KSPSetPC(sksp,sub_schurs->reuse_solver->interior_solver));
        PetscCall(PetscObjectIncrementTabLevel((PetscObject)sub_schurs->reuse_solver->interior_solver,(PetscObject)sksp,0));
      }
    } else { /* default Dirichlet preconditioner is pde-harmonic */
      PetscCall(MatCreateSchurComplement(pcis->A_II,pcis->A_II,pcis->A_IB,pcis->A_BI,pcis->A_BB,&fetidppc_ctx->S_j));
      PetscCall(MatSchurComplementSetKSP(fetidppc_ctx->S_j,pcis->ksp_D));
    }
  } else {
    PetscCall(PetscObjectReference((PetscObject)pcis->A_BB));
    fetidppc_ctx->S_j = pcis->A_BB;
  }
  /* saddle-point */
  if (mat_ctx->xPg) {
    PetscCall(PetscObjectReference((PetscObject)mat_ctx->xPg));
    fetidppc_ctx->xPg = mat_ctx->xPg;
    PetscCall(PetscObjectReference((PetscObject)mat_ctx->yPg));
    fetidppc_ctx->yPg = mat_ctx->yPg;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode FETIDPMatMult_Kernel(Mat fetimat, Vec x, Vec y, PetscBool trans)
{
  FETIDPMat_ctx  mat_ctx;
  PC_BDDC        *pcbddc;
  PC_IS          *pcis;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(fetimat,&mat_ctx));
  pcis = (PC_IS*)mat_ctx->pc->data;
  pcbddc = (PC_BDDC*)mat_ctx->pc->data;
  /* Application of B_delta^T */
  PetscCall(VecSet(pcis->vec1_B,0.));
  PetscCall(VecScatterBegin(mat_ctx->l2g_lambda,x,mat_ctx->lambda_local,INSERT_VALUES,SCATTER_REVERSE));
  PetscCall(VecScatterEnd(mat_ctx->l2g_lambda,x,mat_ctx->lambda_local,INSERT_VALUES,SCATTER_REVERSE));
  PetscCall(MatMultTranspose(mat_ctx->B_delta,mat_ctx->lambda_local,pcis->vec1_B));

  /* Add contribution from saddle point */
  if (mat_ctx->l2g_p) {
    PetscCall(VecScatterBegin(mat_ctx->l2g_p,x,mat_ctx->vP,INSERT_VALUES,SCATTER_REVERSE));
    PetscCall(VecScatterEnd(mat_ctx->l2g_p,x,mat_ctx->vP,INSERT_VALUES,SCATTER_REVERSE));
    if (pcbddc->switch_static) {
      if (trans) {
        PetscCall(MatMultTranspose(mat_ctx->B_BI,mat_ctx->vP,pcis->vec1_D));
      } else {
        PetscCall(MatMult(mat_ctx->Bt_BI,mat_ctx->vP,pcis->vec1_D));
      }
    }
    if (trans) {
      PetscCall(MatMultTransposeAdd(mat_ctx->B_BB,mat_ctx->vP,pcis->vec1_B,pcis->vec1_B));
    } else {
      PetscCall(MatMultAdd(mat_ctx->Bt_BB,mat_ctx->vP,pcis->vec1_B,pcis->vec1_B));
    }
  } else {
    if (pcbddc->switch_static) {
      PetscCall(VecSet(pcis->vec1_D,0.0));
    }
  }
  /* Application of \widetilde{S}^-1 */
  PetscCall(PetscArrayzero(pcbddc->benign_p0,pcbddc->benign_n));
  PetscCall(PCBDDCApplyInterfacePreconditioner(mat_ctx->pc,trans));
  PetscCall(PetscArrayzero(pcbddc->benign_p0,pcbddc->benign_n));
  PetscCall(VecSet(y,0.0));
  /* Application of B_delta */
  PetscCall(MatMult(mat_ctx->B_delta,pcis->vec1_B,mat_ctx->lambda_local));
  /* Contribution from boundary pressures */
  if (mat_ctx->C) {
    const PetscScalar *lx;
    PetscScalar       *ly;

    /* pressure ordered first in the local part of x and y */
    PetscCall(VecGetArrayRead(x,&lx));
    PetscCall(VecGetArray(y,&ly));
    PetscCall(VecPlaceArray(mat_ctx->xPg,lx));
    PetscCall(VecPlaceArray(mat_ctx->yPg,ly));
    if (trans) {
      PetscCall(MatMultTranspose(mat_ctx->C,mat_ctx->xPg,mat_ctx->yPg));
    } else {
      PetscCall(MatMult(mat_ctx->C,mat_ctx->xPg,mat_ctx->yPg));
    }
    PetscCall(VecResetArray(mat_ctx->xPg));
    PetscCall(VecResetArray(mat_ctx->yPg));
    PetscCall(VecRestoreArrayRead(x,&lx));
    PetscCall(VecRestoreArray(y,&ly));
  }
  /* Add contribution from saddle point */
  if (mat_ctx->l2g_p) {
    if (trans) {
      PetscCall(MatMultTranspose(mat_ctx->Bt_BB,pcis->vec1_B,mat_ctx->vP));
    } else {
      PetscCall(MatMult(mat_ctx->B_BB,pcis->vec1_B,mat_ctx->vP));
    }
    if (pcbddc->switch_static) {
      if (trans) {
        PetscCall(MatMultTransposeAdd(mat_ctx->Bt_BI,pcis->vec1_D,mat_ctx->vP,mat_ctx->vP));
      } else {
        PetscCall(MatMultAdd(mat_ctx->B_BI,pcis->vec1_D,mat_ctx->vP,mat_ctx->vP));
      }
    }
    PetscCall(VecScatterBegin(mat_ctx->l2g_p,mat_ctx->vP,y,ADD_VALUES,SCATTER_FORWARD));
    PetscCall(VecScatterEnd(mat_ctx->l2g_p,mat_ctx->vP,y,ADD_VALUES,SCATTER_FORWARD));
  }
  PetscCall(VecScatterBegin(mat_ctx->l2g_lambda,mat_ctx->lambda_local,y,ADD_VALUES,SCATTER_FORWARD));
  PetscCall(VecScatterEnd(mat_ctx->l2g_lambda,mat_ctx->lambda_local,y,ADD_VALUES,SCATTER_FORWARD));
  PetscFunctionReturn(0);
}

PetscErrorCode FETIDPMatMult(Mat fetimat, Vec x, Vec y)
{
  PetscFunctionBegin;
  PetscCall(FETIDPMatMult_Kernel(fetimat,x,y,PETSC_FALSE));
  PetscFunctionReturn(0);
}

PetscErrorCode FETIDPMatMultTranspose(Mat fetimat, Vec x, Vec y)
{
  PetscFunctionBegin;
  PetscCall(FETIDPMatMult_Kernel(fetimat,x,y,PETSC_TRUE));
  PetscFunctionReturn(0);
}

PetscErrorCode FETIDPPCApply_Kernel(PC fetipc, Vec x, Vec y, PetscBool trans)
{
  FETIDPPC_ctx   pc_ctx;
  PC_IS          *pcis;

  PetscFunctionBegin;
  PetscCall(PCShellGetContext(fetipc,&pc_ctx));
  pcis = (PC_IS*)pc_ctx->pc->data;
  /* Application of B_Ddelta^T */
  PetscCall(VecScatterBegin(pc_ctx->l2g_lambda,x,pc_ctx->lambda_local,INSERT_VALUES,SCATTER_REVERSE));
  PetscCall(VecScatterEnd(pc_ctx->l2g_lambda,x,pc_ctx->lambda_local,INSERT_VALUES,SCATTER_REVERSE));
  PetscCall(VecSet(pcis->vec2_B,0.0));
  PetscCall(MatMultTranspose(pc_ctx->B_Ddelta,pc_ctx->lambda_local,pcis->vec2_B));
  /* Application of local Schur complement */
  if (trans) {
    PetscCall(MatMultTranspose(pc_ctx->S_j,pcis->vec2_B,pcis->vec1_B));
  } else {
    PetscCall(MatMult(pc_ctx->S_j,pcis->vec2_B,pcis->vec1_B));
  }
  /* Application of B_Ddelta */
  PetscCall(MatMult(pc_ctx->B_Ddelta,pcis->vec1_B,pc_ctx->lambda_local));
  PetscCall(VecSet(y,0.0));
  PetscCall(VecScatterBegin(pc_ctx->l2g_lambda,pc_ctx->lambda_local,y,ADD_VALUES,SCATTER_FORWARD));
  PetscCall(VecScatterEnd(pc_ctx->l2g_lambda,pc_ctx->lambda_local,y,ADD_VALUES,SCATTER_FORWARD));
  PetscFunctionReturn(0);
}

PetscErrorCode FETIDPPCApply(PC pc, Vec x, Vec y)
{
  PetscFunctionBegin;
  PetscCall(FETIDPPCApply_Kernel(pc,x,y,PETSC_FALSE));
  PetscFunctionReturn(0);
}

PetscErrorCode FETIDPPCApplyTranspose(PC pc, Vec x, Vec y)
{
  PetscFunctionBegin;
  PetscCall(FETIDPPCApply_Kernel(pc,x,y,PETSC_TRUE));
  PetscFunctionReturn(0);
}

PetscErrorCode FETIDPPCView(PC pc, PetscViewer viewer)
{
  FETIDPPC_ctx      pc_ctx;
  PetscBool         iascii;
  PetscViewer       sviewer;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii));
  if (iascii) {
    PetscMPIInt rank;
    PetscBool   isschur,isshell;

    PetscCall(PCShellGetContext(pc,&pc_ctx));
    PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)pc),&rank));
    PetscCall(PetscObjectTypeCompare((PetscObject)pc_ctx->S_j,MATSCHURCOMPLEMENT,&isschur));
    if (isschur) {
      PetscCall(PetscViewerASCIIPrintf(viewer,"  Dirichlet preconditioner (just from rank 0)\n"));
    } else {
      PetscCall(PetscViewerASCIIPrintf(viewer,"  Lumped preconditioner (just from rank 0)\n"));
    }
    PetscCall(PetscViewerGetSubViewer(viewer,PetscObjectComm((PetscObject)pc_ctx->S_j),&sviewer));
    if (rank == 0) {
      PetscCall(PetscViewerPushFormat(sviewer,PETSC_VIEWER_ASCII_INFO));
      PetscCall(PetscViewerASCIIPushTab(sviewer));
      PetscCall(MatView(pc_ctx->S_j,sviewer));
      PetscCall(PetscViewerASCIIPopTab(sviewer));
      PetscCall(PetscViewerPopFormat(sviewer));
    }
    PetscCall(PetscViewerRestoreSubViewer(viewer,PetscObjectComm((PetscObject)pc_ctx->S_j),&sviewer));
    PetscCall(PetscObjectTypeCompare((PetscObject)pc_ctx->B_Ddelta,MATSHELL,&isshell));
    if (isshell) {
      BDdelta_DN ctx;
      PetscCall(PetscViewerASCIIPrintf(viewer,"  FETI-DP BDdelta: DB^t * (B D^-1 B^t)^-1 for deluxe scaling (just from rank 0)\n"));
      PetscCall(MatShellGetContext(pc_ctx->B_Ddelta,&ctx));
      PetscCall(PetscViewerGetSubViewer(viewer,PetscObjectComm((PetscObject)pc_ctx->S_j),&sviewer));
      if (rank == 0) {
        PetscInt tl;

        PetscCall(PetscViewerASCIIGetTab(sviewer,&tl));
        PetscCall(PetscObjectSetTabLevel((PetscObject)ctx->kBD,tl));
        PetscCall(KSPView(ctx->kBD,sviewer));
        PetscCall(PetscViewerPushFormat(sviewer,PETSC_VIEWER_ASCII_INFO));
        PetscCall(MatView(ctx->BD,sviewer));
        PetscCall(PetscViewerPopFormat(sviewer));
      }
      PetscCall(PetscViewerRestoreSubViewer(viewer,PetscObjectComm((PetscObject)pc_ctx->S_j),&sviewer));
    }
    PetscCall(PetscViewerFlush(viewer));
  }
  PetscFunctionReturn(0);
}
