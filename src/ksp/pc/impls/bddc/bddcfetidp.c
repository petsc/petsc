#include <../src/ksp/pc/impls/bddc/bddc.h>
#include <../src/ksp/pc/impls/bddc/bddcprivate.h>
#include <petscblaslapack.h>

static PetscErrorCode MatMult_BDdelta_deluxe_nonred(Mat A, Vec x, Vec y)
{
  BDdelta_DN     ctx;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(A,&ctx));
  CHKERRQ(MatMultTranspose(ctx->BD,x,ctx->work));
  CHKERRQ(KSPSolveTranspose(ctx->kBD,ctx->work,y));
  /* No PC so cannot propagate up failure in KSPSolveTranspose() */
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMultTranspose_BDdelta_deluxe_nonred(Mat A, Vec x, Vec y)
{
  BDdelta_DN     ctx;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(A,&ctx));
  CHKERRQ(KSPSolve(ctx->kBD,x,ctx->work));
  /* No PC so cannot propagate up failure in KSPSolve() */
  CHKERRQ(MatMult(ctx->BD,ctx->work,y));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDestroy_BDdelta_deluxe_nonred(Mat A)
{
  BDdelta_DN     ctx;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(A,&ctx));
  CHKERRQ(MatDestroy(&ctx->BD));
  CHKERRQ(KSPDestroy(&ctx->kBD));
  CHKERRQ(VecDestroy(&ctx->work));
  CHKERRQ(PetscFree(ctx));
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCCreateFETIDPMatContext(PC pc, FETIDPMat_ctx *fetidpmat_ctx)
{
  FETIDPMat_ctx  newctx;

  PetscFunctionBegin;
  CHKERRQ(PetscNew(&newctx));
  /* increase the reference count for BDDC preconditioner */
  CHKERRQ(PetscObjectReference((PetscObject)pc));
  newctx->pc              = pc;
  *fetidpmat_ctx          = newctx;
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCCreateFETIDPPCContext(PC pc, FETIDPPC_ctx *fetidppc_ctx)
{
  FETIDPPC_ctx   newctx;

  PetscFunctionBegin;
  CHKERRQ(PetscNew(&newctx));
  /* increase the reference count for BDDC preconditioner */
  CHKERRQ(PetscObjectReference((PetscObject)pc));
  newctx->pc              = pc;
  *fetidppc_ctx           = newctx;
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCDestroyFETIDPMat(Mat A)
{
  FETIDPMat_ctx  mat_ctx;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(A,&mat_ctx));
  CHKERRQ(VecDestroy(&mat_ctx->lambda_local));
  CHKERRQ(VecDestroy(&mat_ctx->temp_solution_D));
  CHKERRQ(VecDestroy(&mat_ctx->temp_solution_B));
  CHKERRQ(MatDestroy(&mat_ctx->B_delta));
  CHKERRQ(MatDestroy(&mat_ctx->B_Ddelta));
  CHKERRQ(MatDestroy(&mat_ctx->B_BB));
  CHKERRQ(MatDestroy(&mat_ctx->B_BI));
  CHKERRQ(MatDestroy(&mat_ctx->Bt_BB));
  CHKERRQ(MatDestroy(&mat_ctx->Bt_BI));
  CHKERRQ(MatDestroy(&mat_ctx->C));
  CHKERRQ(VecDestroy(&mat_ctx->rhs_flip));
  CHKERRQ(VecDestroy(&mat_ctx->vP));
  CHKERRQ(VecDestroy(&mat_ctx->xPg));
  CHKERRQ(VecDestroy(&mat_ctx->yPg));
  CHKERRQ(VecScatterDestroy(&mat_ctx->l2g_lambda));
  CHKERRQ(VecScatterDestroy(&mat_ctx->l2g_lambda_only));
  CHKERRQ(VecScatterDestroy(&mat_ctx->l2g_p));
  CHKERRQ(VecScatterDestroy(&mat_ctx->g2g_p));
  CHKERRQ(PCDestroy(&mat_ctx->pc)); /* decrease PCBDDC reference count */
  CHKERRQ(ISDestroy(&mat_ctx->pressure));
  CHKERRQ(ISDestroy(&mat_ctx->lagrange));
  CHKERRQ(PetscFree(mat_ctx));
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCDestroyFETIDPPC(PC pc)
{
  FETIDPPC_ctx   pc_ctx;

  PetscFunctionBegin;
  CHKERRQ(PCShellGetContext(pc,&pc_ctx));
  CHKERRQ(VecDestroy(&pc_ctx->lambda_local));
  CHKERRQ(MatDestroy(&pc_ctx->B_Ddelta));
  CHKERRQ(VecScatterDestroy(&pc_ctx->l2g_lambda));
  CHKERRQ(MatDestroy(&pc_ctx->S_j));
  CHKERRQ(PCDestroy(&pc_ctx->pc)); /* decrease PCBDDC reference count */
  CHKERRQ(VecDestroy(&pc_ctx->xPg));
  CHKERRQ(VecDestroy(&pc_ctx->yPg));
  CHKERRQ(PetscFree(pc_ctx));
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
  CHKERRQ(PetscObjectGetComm((PetscObject)(fetidpmat_ctx->pc),&comm));
  CHKERRMPI(MPI_Comm_rank(comm,&rank));
  CHKERRMPI(MPI_Comm_size(comm,&size));

  /* saddlepoint */
  nPl      = 0;
  nPg      = 0;
  nPgl     = 0;
  gP       = NULL;
  pP       = NULL;
  l2gmap_p = NULL;
  play     = NULL;
  CHKERRQ(PetscObjectQuery((PetscObject)fetidpmat_ctx->pc,"__KSPFETIDP_pP",(PetscObject*)&pP));
  if (pP) { /* saddle point */
    /* subdomain pressures in global numbering */
    CHKERRQ(PetscObjectQuery((PetscObject)fetidpmat_ctx->pc,"__KSPFETIDP_gP",(PetscObject*)&gP));
    PetscCheckFalse(!gP,PETSC_COMM_SELF,PETSC_ERR_PLIB,"gP not present");
    CHKERRQ(ISGetLocalSize(gP,&nPl));
    CHKERRQ(VecCreate(PETSC_COMM_SELF,&fetidpmat_ctx->vP));
    CHKERRQ(VecSetSizes(fetidpmat_ctx->vP,nPl,nPl));
    CHKERRQ(VecSetType(fetidpmat_ctx->vP,VECSTANDARD));
    CHKERRQ(VecSetUp(fetidpmat_ctx->vP));

    /* pressure matrix */
    CHKERRQ(PetscObjectQuery((PetscObject)fetidpmat_ctx->pc,"__KSPFETIDP_C",(PetscObject*)&fetidpmat_ctx->C));
    if (!fetidpmat_ctx->C) { /* null pressure block, compute layout and global numbering for pressures */
      IS Pg;

      CHKERRQ(ISRenumber(gP,NULL,&nPg,&Pg));
      CHKERRQ(ISLocalToGlobalMappingCreateIS(Pg,&l2gmap_p));
      CHKERRQ(ISDestroy(&Pg));
      CHKERRQ(PetscLayoutCreate(comm,&play));
      CHKERRQ(PetscLayoutSetBlockSize(play,1));
      CHKERRQ(PetscLayoutSetSize(play,nPg));
      CHKERRQ(ISGetLocalSize(pP,&nPgl));
      CHKERRQ(PetscLayoutSetLocalSize(play,nPgl));
      CHKERRQ(PetscLayoutSetUp(play));
    } else {
      CHKERRQ(PetscObjectReference((PetscObject)fetidpmat_ctx->C));
      CHKERRQ(MatISGetLocalToGlobalMapping(fetidpmat_ctx->C,&l2gmap_p,NULL));
      CHKERRQ(PetscObjectReference((PetscObject)l2gmap_p));
      CHKERRQ(MatGetSize(fetidpmat_ctx->C,&nPg,NULL));
      CHKERRQ(MatGetLocalSize(fetidpmat_ctx->C,NULL,&nPgl));
      CHKERRQ(MatGetLayouts(fetidpmat_ctx->C,NULL,&llay));
      CHKERRQ(PetscLayoutReference(llay,&play));
    }
    CHKERRQ(VecCreateMPIWithArray(comm,1,nPgl,nPg,NULL,&fetidpmat_ctx->xPg));
    CHKERRQ(VecCreateMPIWithArray(comm,1,nPgl,nPg,NULL,&fetidpmat_ctx->yPg));

    /* import matrices for pressures coupling */
    CHKERRQ(PetscObjectQuery((PetscObject)fetidpmat_ctx->pc,"__KSPFETIDP_B_BI",(PetscObject*)&fetidpmat_ctx->B_BI));
    PetscCheckFalse(!fetidpmat_ctx->B_BI,PETSC_COMM_SELF,PETSC_ERR_PLIB,"B_BI not present");
    CHKERRQ(PetscObjectReference((PetscObject)fetidpmat_ctx->B_BI));

    CHKERRQ(PetscObjectQuery((PetscObject)fetidpmat_ctx->pc,"__KSPFETIDP_B_BB",(PetscObject*)&fetidpmat_ctx->B_BB));
    PetscCheckFalse(!fetidpmat_ctx->B_BB,PETSC_COMM_SELF,PETSC_ERR_PLIB,"B_BB not present");
    CHKERRQ(PetscObjectReference((PetscObject)fetidpmat_ctx->B_BB));

    CHKERRQ(PetscObjectQuery((PetscObject)fetidpmat_ctx->pc,"__KSPFETIDP_Bt_BI",(PetscObject*)&fetidpmat_ctx->Bt_BI));
    PetscCheckFalse(!fetidpmat_ctx->Bt_BI,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Bt_BI not present");
    CHKERRQ(PetscObjectReference((PetscObject)fetidpmat_ctx->Bt_BI));

    CHKERRQ(PetscObjectQuery((PetscObject)fetidpmat_ctx->pc,"__KSPFETIDP_Bt_BB",(PetscObject*)&fetidpmat_ctx->Bt_BB));
    PetscCheckFalse(!fetidpmat_ctx->Bt_BB,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Bt_BB not present");
    CHKERRQ(PetscObjectReference((PetscObject)fetidpmat_ctx->Bt_BB));

    CHKERRQ(PetscObjectQuery((PetscObject)fetidpmat_ctx->pc,"__KSPFETIDP_flip" ,(PetscObject*)&fetidpmat_ctx->rhs_flip));
    if (fetidpmat_ctx->rhs_flip) {
      CHKERRQ(PetscObjectReference((PetscObject)fetidpmat_ctx->rhs_flip));
    }
  }

  /* Default type of lagrange multipliers is non-redundant */
  fully_redundant = fetidpmat_ctx->fully_redundant;

  /* Evaluate local and global number of lagrange multipliers */
  CHKERRQ(VecSet(pcis->vec1_N,0.0));
  n_local_lambda = 0;
  partial_sum = 0;
  n_boundary_dofs = 0;
  s = 0;

  /* Get Vertices used to define the BDDC */
  CHKERRQ(PCBDDCGraphGetCandidatesIS(pcbddc->mat_graph,NULL,NULL,NULL,NULL,&isvert));
  CHKERRQ(ISGetLocalSize(isvert,&n_vertices));
  CHKERRQ(ISGetIndices(isvert,&vertex_indices));

  dual_size = pcis->n_B-n_vertices;
  CHKERRQ(PetscMalloc1(dual_size,&dual_dofs_boundary_indices));
  CHKERRQ(PetscMalloc1(dual_size,&aux_local_numbering_1));
  CHKERRQ(PetscMalloc1(dual_size,&aux_local_numbering_2));

  CHKERRQ(VecGetArray(pcis->vec1_N,&array));
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
  CHKERRQ(VecRestoreArray(pcis->vec1_N,&array));
  CHKERRQ(ISRestoreIndices(isvert,&vertex_indices));
  CHKERRQ(PCBDDCGraphRestoreCandidatesIS(pcbddc->mat_graph,NULL,NULL,NULL,NULL,&isvert));
  dual_size = partial_sum;

  /* compute global ordering of lagrange multipliers and associate l2g map */
  CHKERRQ(ISCreateGeneral(comm,partial_sum,aux_local_numbering_1,PETSC_COPY_VALUES,&subset_n));
  CHKERRQ(ISLocalToGlobalMappingApplyIS(pcis->mapping,subset_n,&subset));
  CHKERRQ(ISDestroy(&subset_n));
  CHKERRQ(ISCreateGeneral(comm,partial_sum,aux_local_numbering_2,PETSC_OWN_POINTER,&subset_mult));
  CHKERRQ(ISRenumber(subset,subset_mult,&fetidpmat_ctx->n_lambda,&subset_n));
  CHKERRQ(ISDestroy(&subset));

  if (PetscDefined(USE_DEBUG)) {
    CHKERRQ(VecSet(pcis->vec1_global,0.0));
    CHKERRQ(VecScatterBegin(matis->rctx,pcis->vec1_N,pcis->vec1_global,ADD_VALUES,SCATTER_REVERSE));
    CHKERRQ(VecScatterEnd(matis->rctx,pcis->vec1_N,pcis->vec1_global,ADD_VALUES,SCATTER_REVERSE));
    CHKERRQ(VecSum(pcis->vec1_global,&scalar_value));
    i = (PetscInt)PetscRealPart(scalar_value);
    PetscCheckFalse(i != fetidpmat_ctx->n_lambda,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Global number of multipliers mismatch! (%D != %D)",fetidpmat_ctx->n_lambda,i);
  }

  /* init data for scaling factors exchange */
  if (!pcbddc->use_deluxe_scaling) {
    PetscInt    *ptrs_buffer,neigh_position;
    PetscScalar *send_buffer,*recv_buffer;
    MPI_Request *send_reqs,*recv_reqs;

    partial_sum = 0;
    CHKERRQ(PetscMalloc1(pcis->n_neigh,&ptrs_buffer));
    CHKERRQ(PetscMalloc1(PetscMax(pcis->n_neigh-1,0),&send_reqs));
    CHKERRQ(PetscMalloc1(PetscMax(pcis->n_neigh-1,0),&recv_reqs));
    CHKERRQ(PetscMalloc1(pcis->n+1,&all_factors));
    if (pcis->n_neigh > 0) ptrs_buffer[0]=0;
    for (i=1;i<pcis->n_neigh;i++) {
      partial_sum += pcis->n_shared[i];
      ptrs_buffer[i] = ptrs_buffer[i-1]+pcis->n_shared[i];
    }
    CHKERRQ(PetscMalloc1(partial_sum,&send_buffer));
    CHKERRQ(PetscMalloc1(partial_sum,&recv_buffer));
    CHKERRQ(PetscMalloc1(partial_sum,&all_factors[0]));
    for (i=0;i<pcis->n-1;i++) {
      j = mat_graph->count[i];
      all_factors[i+1]=all_factors[i]+j;
    }

    /* scatter B scaling to N vec */
    CHKERRQ(VecScatterBegin(pcis->N_to_B,pcis->D,pcis->vec1_N,INSERT_VALUES,SCATTER_REVERSE));
    CHKERRQ(VecScatterEnd(pcis->N_to_B,pcis->D,pcis->vec1_N,INSERT_VALUES,SCATTER_REVERSE));
    /* communications */
    CHKERRQ(VecGetArrayRead(pcis->vec1_N,(const PetscScalar**)&array));
    for (i=1;i<pcis->n_neigh;i++) {
      for (j=0;j<pcis->n_shared[i];j++) {
        send_buffer[ptrs_buffer[i-1]+j]=array[pcis->shared[i][j]];
      }
      CHKERRQ(PetscMPIIntCast(ptrs_buffer[i]-ptrs_buffer[i-1],&buf_size));
      CHKERRQ(PetscMPIIntCast(pcis->neigh[i],&neigh));
      CHKERRMPI(MPI_Isend(&send_buffer[ptrs_buffer[i-1]],buf_size,MPIU_SCALAR,neigh,0,comm,&send_reqs[i-1]));
      CHKERRMPI(MPI_Irecv(&recv_buffer[ptrs_buffer[i-1]],buf_size,MPIU_SCALAR,neigh,0,comm,&recv_reqs[i-1]));
    }
    CHKERRQ(VecRestoreArrayRead(pcis->vec1_N,(const PetscScalar**)&array));
    if (pcis->n_neigh > 0) {
      CHKERRMPI(MPI_Waitall(pcis->n_neigh-1,recv_reqs,MPI_STATUSES_IGNORE));
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
      CHKERRMPI(MPI_Waitall(pcis->n_neigh-1,send_reqs,MPI_STATUSES_IGNORE));
    }
    CHKERRQ(PetscFree(send_reqs));
    CHKERRQ(PetscFree(recv_reqs));
    CHKERRQ(PetscFree(send_buffer));
    CHKERRQ(PetscFree(recv_buffer));
    CHKERRQ(PetscFree(ptrs_buffer));
  }

  /* Compute B and B_delta (local actions) */
  CHKERRQ(PetscMalloc1(pcis->n_neigh,&aux_sums));
  CHKERRQ(PetscMalloc1(n_local_lambda,&l2g_indices));
  CHKERRQ(PetscMalloc1(n_local_lambda,&vals_B_delta));
  CHKERRQ(PetscMalloc1(n_local_lambda,&cols_B_delta));
  if (!pcbddc->use_deluxe_scaling) {
    CHKERRQ(PetscMalloc1(n_local_lambda,&scaling_factors));
  } else {
    scaling_factors = NULL;
    all_factors     = NULL;
  }
  CHKERRQ(ISGetIndices(subset_n,&aux_global_numbering));
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
  CHKERRQ(ISRestoreIndices(subset_n,&aux_global_numbering));
  CHKERRQ(ISDestroy(&subset_mult));
  CHKERRQ(ISDestroy(&subset_n));
  CHKERRQ(PetscFree(aux_sums));
  CHKERRQ(PetscFree(aux_local_numbering_1));
  CHKERRQ(PetscFree(dual_dofs_boundary_indices));
  if (all_factors) {
    CHKERRQ(PetscFree(all_factors[0]));
    CHKERRQ(PetscFree(all_factors));
  }

  /* Create local part of B_delta */
  CHKERRQ(MatCreate(PETSC_COMM_SELF,&fetidpmat_ctx->B_delta));
  CHKERRQ(MatSetSizes(fetidpmat_ctx->B_delta,n_local_lambda,pcis->n_B,n_local_lambda,pcis->n_B));
  CHKERRQ(MatSetType(fetidpmat_ctx->B_delta,MATSEQAIJ));
  CHKERRQ(MatSeqAIJSetPreallocation(fetidpmat_ctx->B_delta,1,NULL));
  CHKERRQ(MatSetOption(fetidpmat_ctx->B_delta,MAT_IGNORE_ZERO_ENTRIES,PETSC_TRUE));
  for (i=0;i<n_local_lambda;i++) {
    CHKERRQ(MatSetValue(fetidpmat_ctx->B_delta,i,cols_B_delta[i],vals_B_delta[i],INSERT_VALUES));
  }
  CHKERRQ(PetscFree(vals_B_delta));
  CHKERRQ(MatAssemblyBegin(fetidpmat_ctx->B_delta,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(fetidpmat_ctx->B_delta,MAT_FINAL_ASSEMBLY));

  BD1 = NULL;
  BD2 = NULL;
  if (fully_redundant) {
    PetscCheckFalse(pcbddc->use_deluxe_scaling,comm,PETSC_ERR_SUP,"Deluxe FETIDP with fully-redundant multipliers to be implemented");
    CHKERRQ(MatCreate(PETSC_COMM_SELF,&ScalingMat));
    CHKERRQ(MatSetSizes(ScalingMat,n_local_lambda,n_local_lambda,n_local_lambda,n_local_lambda));
    CHKERRQ(MatSetType(ScalingMat,MATSEQAIJ));
    CHKERRQ(MatSeqAIJSetPreallocation(ScalingMat,1,NULL));
    for (i=0;i<n_local_lambda;i++) {
      CHKERRQ(MatSetValue(ScalingMat,i,i,scaling_factors[i],INSERT_VALUES));
    }
    CHKERRQ(MatAssemblyBegin(ScalingMat,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(ScalingMat,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatMatMult(ScalingMat,fetidpmat_ctx->B_delta,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&fetidpmat_ctx->B_Ddelta));
    CHKERRQ(MatDestroy(&ScalingMat));
  } else {
    CHKERRQ(MatCreate(PETSC_COMM_SELF,&fetidpmat_ctx->B_Ddelta));
    CHKERRQ(MatSetSizes(fetidpmat_ctx->B_Ddelta,n_local_lambda,pcis->n_B,n_local_lambda,pcis->n_B));
    if (!pcbddc->use_deluxe_scaling || !pcbddc->sub_schurs) {
      CHKERRQ(MatSetType(fetidpmat_ctx->B_Ddelta,MATSEQAIJ));
      CHKERRQ(MatSeqAIJSetPreallocation(fetidpmat_ctx->B_Ddelta,1,NULL));
      for (i=0;i<n_local_lambda;i++) {
        CHKERRQ(MatSetValue(fetidpmat_ctx->B_Ddelta,i,cols_B_delta[i],scaling_factors[i],INSERT_VALUES));
      }
      CHKERRQ(MatAssemblyBegin(fetidpmat_ctx->B_Ddelta,MAT_FINAL_ASSEMBLY));
      CHKERRQ(MatAssemblyEnd(fetidpmat_ctx->B_Ddelta,MAT_FINAL_ASSEMBLY));
    } else {
      /* scaling as in Klawonn-Widlund 1999 */
      PCBDDCDeluxeScaling deluxe_ctx = pcbddc->deluxe_ctx;
      PCBDDCSubSchurs     sub_schurs = pcbddc->sub_schurs;
      Mat                 T;
      PetscScalar         *W,lwork,*Bwork;
      const PetscInt      *idxs = NULL;
      PetscInt            cum,mss,*nnz;
      PetscBLASInt        *pivots,B_lwork,B_N,B_ierr;

      PetscCheckFalse(!pcbddc->deluxe_singlemat,comm,PETSC_ERR_USER,"Cannot compute B_Ddelta! rerun with -pc_bddc_deluxe_singlemat");
      mss  = 0;
      CHKERRQ(PetscCalloc1(pcis->n_B,&nnz));
      if (sub_schurs->is_Ej_all) {
        CHKERRQ(ISGetIndices(sub_schurs->is_Ej_all,&idxs));
        for (i=0,cum=0;i<sub_schurs->n_subs;i++) {
          PetscInt subset_size;

          CHKERRQ(ISGetLocalSize(sub_schurs->is_subs[i],&subset_size));
          for (j=0;j<subset_size;j++) nnz[idxs[j+cum]] = subset_size;
          mss  = PetscMax(mss,subset_size);
          cum += subset_size;
        }
      }
      CHKERRQ(MatCreate(PETSC_COMM_SELF,&T));
      CHKERRQ(MatSetSizes(T,pcis->n_B,pcis->n_B,pcis->n_B,pcis->n_B));
      CHKERRQ(MatSetType(T,MATSEQAIJ));
      CHKERRQ(MatSeqAIJSetPreallocation(T,0,nnz));
      CHKERRQ(PetscFree(nnz));

      /* workspace allocation */
      B_lwork = 0;
      if (mss) {
        PetscScalar dummy = 1;

        B_lwork = -1;
        CHKERRQ(PetscBLASIntCast(mss,&B_N));
        CHKERRQ(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
        PetscStackCallBLAS("LAPACKgetri",LAPACKgetri_(&B_N,&dummy,&B_N,&B_N,&lwork,&B_lwork,&B_ierr));
        CHKERRQ(PetscFPTrapPop());
        PetscCheckFalse(B_ierr,PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in query to GETRI Lapack routine %d",(int)B_ierr);
        CHKERRQ(PetscBLASIntCast((PetscInt)PetscRealPart(lwork),&B_lwork));
      }
      CHKERRQ(PetscMalloc3(mss*mss,&W,mss,&pivots,B_lwork,&Bwork));

      for (i=0,cum=0;i<sub_schurs->n_subs;i++) {
        const PetscScalar *M;
        PetscInt          subset_size;

        CHKERRQ(ISGetLocalSize(sub_schurs->is_subs[i],&subset_size));
        CHKERRQ(PetscBLASIntCast(subset_size,&B_N));
        CHKERRQ(MatDenseGetArrayRead(deluxe_ctx->seq_mat[i],&M));
        CHKERRQ(PetscArraycpy(W,M,subset_size*subset_size));
        CHKERRQ(MatDenseRestoreArrayRead(deluxe_ctx->seq_mat[i],&M));
        CHKERRQ(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
        PetscStackCallBLAS("LAPACKgetrf",LAPACKgetrf_(&B_N,&B_N,W,&B_N,pivots,&B_ierr));
        PetscCheckFalse(B_ierr,PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in GETRF Lapack routine %d",(int)B_ierr);
        PetscStackCallBLAS("LAPACKgetri",LAPACKgetri_(&B_N,W,&B_N,pivots,Bwork,&B_lwork,&B_ierr));
        PetscCheckFalse(B_ierr,PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in GETRI Lapack routine %d",(int)B_ierr);
        CHKERRQ(PetscFPTrapPop());
        /* silent static analyzer */
        PetscCheckFalse(!idxs,PETSC_COMM_SELF,PETSC_ERR_PLIB,"IDXS not present");
        CHKERRQ(MatSetValues(T,subset_size,idxs+cum,subset_size,idxs+cum,W,INSERT_VALUES));
        cum += subset_size;
      }
      CHKERRQ(MatAssemblyBegin(T,MAT_FINAL_ASSEMBLY));
      CHKERRQ(MatAssemblyEnd(T,MAT_FINAL_ASSEMBLY));
      CHKERRQ(MatMatTransposeMult(T,fetidpmat_ctx->B_delta,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&BD1));
      CHKERRQ(MatMatMult(fetidpmat_ctx->B_delta,BD1,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&BD2));
      CHKERRQ(MatDestroy(&T));
      CHKERRQ(PetscFree3(W,pivots,Bwork));
      if (sub_schurs->is_Ej_all) {
        CHKERRQ(ISRestoreIndices(sub_schurs->is_Ej_all,&idxs));
      }
    }
  }
  CHKERRQ(PetscFree(scaling_factors));
  CHKERRQ(PetscFree(cols_B_delta));

  /* Layout of multipliers */
  CHKERRQ(PetscLayoutCreate(comm,&llay));
  CHKERRQ(PetscLayoutSetBlockSize(llay,1));
  CHKERRQ(PetscLayoutSetSize(llay,fetidpmat_ctx->n_lambda));
  CHKERRQ(PetscLayoutSetUp(llay));
  CHKERRQ(PetscLayoutGetLocalSize(llay,&fetidpmat_ctx->n));

  /* Local work vector of multipliers */
  CHKERRQ(VecCreate(PETSC_COMM_SELF,&fetidpmat_ctx->lambda_local));
  CHKERRQ(VecSetSizes(fetidpmat_ctx->lambda_local,n_local_lambda,n_local_lambda));
  CHKERRQ(VecSetType(fetidpmat_ctx->lambda_local,VECSEQ));

  if (BD2) {
    ISLocalToGlobalMapping l2g;
    Mat                    T,TA,*pT;
    IS                     is;
    PetscInt               nl,N;
    BDdelta_DN             ctx;

    CHKERRQ(PetscLayoutGetLocalSize(llay,&nl));
    CHKERRQ(PetscLayoutGetSize(llay,&N));
    CHKERRQ(MatCreate(comm,&T));
    CHKERRQ(MatSetSizes(T,nl,nl,N,N));
    CHKERRQ(MatSetType(T,MATIS));
    CHKERRQ(ISLocalToGlobalMappingCreate(comm,1,n_local_lambda,l2g_indices,PETSC_COPY_VALUES,&l2g));
    CHKERRQ(MatSetLocalToGlobalMapping(T,l2g,l2g));
    CHKERRQ(ISLocalToGlobalMappingDestroy(&l2g));
    CHKERRQ(MatISSetLocalMat(T,BD2));
    CHKERRQ(MatAssemblyBegin(T,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(T,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatDestroy(&BD2));
    CHKERRQ(MatConvert(T,MATAIJ,MAT_INITIAL_MATRIX,&TA));
    CHKERRQ(MatDestroy(&T));
    CHKERRQ(ISCreateGeneral(comm,n_local_lambda,l2g_indices,PETSC_USE_POINTER,&is));
    CHKERRQ(MatCreateSubMatrices(TA,1,&is,&is,MAT_INITIAL_MATRIX,&pT));
    CHKERRQ(MatDestroy(&TA));
    CHKERRQ(ISDestroy(&is));
    BD2  = pT[0];
    CHKERRQ(PetscFree(pT));

    /* B_Ddelta for non-redundant multipliers with deluxe scaling */
    CHKERRQ(PetscNew(&ctx));
    CHKERRQ(MatSetType(fetidpmat_ctx->B_Ddelta,MATSHELL));
    CHKERRQ(MatShellSetContext(fetidpmat_ctx->B_Ddelta,ctx));
    CHKERRQ(MatShellSetOperation(fetidpmat_ctx->B_Ddelta,MATOP_MULT,(void (*)(void))MatMult_BDdelta_deluxe_nonred));
    CHKERRQ(MatShellSetOperation(fetidpmat_ctx->B_Ddelta,MATOP_MULT_TRANSPOSE,(void (*)(void))MatMultTranspose_BDdelta_deluxe_nonred));
    CHKERRQ(MatShellSetOperation(fetidpmat_ctx->B_Ddelta,MATOP_DESTROY,(void (*)(void))MatDestroy_BDdelta_deluxe_nonred));
    CHKERRQ(MatSetUp(fetidpmat_ctx->B_Ddelta));

    CHKERRQ(PetscObjectReference((PetscObject)BD1));
    ctx->BD = BD1;
    CHKERRQ(KSPCreate(PETSC_COMM_SELF,&ctx->kBD));
    CHKERRQ(KSPSetOperators(ctx->kBD,BD2,BD2));
    CHKERRQ(VecDuplicate(fetidpmat_ctx->lambda_local,&ctx->work));
    fetidpmat_ctx->deluxe_nonred = PETSC_TRUE;
  }
  CHKERRQ(MatDestroy(&BD1));
  CHKERRQ(MatDestroy(&BD2));

  /* fetidpmat sizes */
  fetidpmat_ctx->n += nPgl;
  fetidpmat_ctx->N  = fetidpmat_ctx->n_lambda+nPg;

  /* Global vector for FETI-DP linear system */
  CHKERRQ(VecCreate(comm,&fetidp_global));
  CHKERRQ(VecSetSizes(fetidp_global,fetidpmat_ctx->n,fetidpmat_ctx->N));
  CHKERRQ(VecSetType(fetidp_global,VECMPI));
  CHKERRQ(VecSetUp(fetidp_global));

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

    CHKERRQ(PetscMalloc1(nPl,&l2g_indices_p));
    CHKERRQ(VecGetLayout(fetidp_global,&alay));
    CHKERRQ(PetscLayoutGetRanges(alay,&aranges));
    CHKERRQ(PetscLayoutGetRanges(play,&pranges));
    CHKERRQ(PetscLayoutGetRanges(llay,&lranges));

    CHKERRMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)fetidp_global),&rank));
    CHKERRQ(ISCreateStride(PetscObjectComm((PetscObject)fetidp_global),pranges[rank+1]-pranges[rank],aranges[rank],1,&fetidpmat_ctx->pressure));
    CHKERRQ(PetscObjectSetName((PetscObject)fetidpmat_ctx->pressure,"F_P"));
    CHKERRQ(ISCreateStride(PetscObjectComm((PetscObject)fetidp_global),lranges[rank+1]-lranges[rank],aranges[rank]+pranges[rank+1]-pranges[rank],1,&fetidpmat_ctx->lagrange));
    CHKERRQ(PetscObjectSetName((PetscObject)fetidpmat_ctx->lagrange,"F_L"));
    CHKERRQ(ISLocalToGlobalMappingGetIndices(l2gmap_p,&idxs));
    /* shift local to global indices for pressure */
    for (i=0;i<nPl;i++) {
      PetscMPIInt owner;

      CHKERRQ(PetscLayoutFindOwner(play,idxs[i],&owner));
      l2g_indices_p[i] = idxs[i]-pranges[owner]+aranges[owner];
    }
    CHKERRQ(ISLocalToGlobalMappingRestoreIndices(l2gmap_p,&idxs));
    CHKERRQ(ISCreateGeneral(comm,nPl,l2g_indices_p,PETSC_OWN_POINTER,&IS_l2g_p));

    /* local to global scatter for pressure */
    CHKERRQ(VecScatterCreate(fetidpmat_ctx->vP,NULL,fetidp_global,IS_l2g_p,&fetidpmat_ctx->l2g_p));
    CHKERRQ(ISDestroy(&IS_l2g_p));

    /* scatter for lagrange multipliers only */
    CHKERRQ(VecCreate(comm,&v));
    CHKERRQ(VecSetType(v,VECSTANDARD));
    CHKERRQ(VecSetLayout(v,llay));
    CHKERRQ(VecSetUp(v));
    CHKERRQ(ISCreateGeneral(comm,n_local_lambda,l2g_indices,PETSC_COPY_VALUES,&ais));
    CHKERRQ(VecScatterCreate(fetidpmat_ctx->lambda_local,NULL,v,ais,&fetidpmat_ctx->l2g_lambda_only));
    CHKERRQ(ISDestroy(&ais));
    CHKERRQ(VecDestroy(&v));

    /* shift local to global indices for multipliers */
    for (i=0;i<n_local_lambda;i++) {
      PetscInt    ps;
      PetscMPIInt owner;

      CHKERRQ(PetscLayoutFindOwner(llay,l2g_indices[i],&owner));
      ps = pranges[owner+1]-pranges[owner];
      l2g_indices[i] = l2g_indices[i]-lranges[owner]+aranges[owner]+ps;
    }

    /* scatter from alldofs to pressures global fetidp vector */
    CHKERRQ(PetscLayoutGetRange(alay,&rst,NULL));
    CHKERRQ(ISCreateStride(comm,nPgl,rst,1,&ais));
    CHKERRQ(VecScatterCreate(pcis->vec1_global,pP,fetidp_global,ais,&fetidpmat_ctx->g2g_p));
    CHKERRQ(ISDestroy(&ais));
  }
  CHKERRQ(PetscLayoutDestroy(&llay));
  CHKERRQ(PetscLayoutDestroy(&play));
  CHKERRQ(ISCreateGeneral(comm,n_local_lambda,l2g_indices,PETSC_OWN_POINTER,&IS_l2g_lambda));

  /* scatter from local to global multipliers */
  CHKERRQ(VecScatterCreate(fetidpmat_ctx->lambda_local,NULL,fetidp_global,IS_l2g_lambda,&fetidpmat_ctx->l2g_lambda));
  CHKERRQ(ISDestroy(&IS_l2g_lambda));
  CHKERRQ(ISLocalToGlobalMappingDestroy(&l2gmap_p));
  CHKERRQ(VecDestroy(&fetidp_global));

  /* Create some work vectors needed by fetidp */
  CHKERRQ(VecDuplicate(pcis->vec1_B,&fetidpmat_ctx->temp_solution_B));
  CHKERRQ(VecDuplicate(pcis->vec1_D,&fetidpmat_ctx->temp_solution_D));
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCSetupFETIDPPCContext(Mat fetimat, FETIDPPC_ctx fetidppc_ctx)
{
  FETIDPMat_ctx  mat_ctx;
  PC_BDDC        *pcbddc = (PC_BDDC*)fetidppc_ctx->pc->data;
  PC_IS          *pcis = (PC_IS*)fetidppc_ctx->pc->data;
  PetscBool      lumped = PETSC_FALSE;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(fetimat,&mat_ctx));
  /* get references from objects created when setting up feti mat context */
  CHKERRQ(PetscObjectReference((PetscObject)mat_ctx->lambda_local));
  fetidppc_ctx->lambda_local = mat_ctx->lambda_local;
  CHKERRQ(PetscObjectReference((PetscObject)mat_ctx->B_Ddelta));
  fetidppc_ctx->B_Ddelta = mat_ctx->B_Ddelta;
  if (mat_ctx->deluxe_nonred) {
    PC               pc,mpc;
    BDdelta_DN       ctx;
    MatSolverType    solver;
    const char       *prefix;

    CHKERRQ(MatShellGetContext(mat_ctx->B_Ddelta,&ctx));
    CHKERRQ(KSPSetType(ctx->kBD,KSPPREONLY));
    CHKERRQ(KSPGetPC(ctx->kBD,&mpc));
    CHKERRQ(KSPGetPC(pcbddc->ksp_D,&pc));
    CHKERRQ(PCSetType(mpc,PCLU));
    CHKERRQ(PCFactorGetMatSolverType(pc,(MatSolverType*)&solver));
    if (solver) {
      CHKERRQ(PCFactorSetMatSolverType(mpc,solver));
    }
    CHKERRQ(MatGetOptionsPrefix(fetimat,&prefix));
    CHKERRQ(KSPSetOptionsPrefix(ctx->kBD,prefix));
    CHKERRQ(KSPAppendOptionsPrefix(ctx->kBD,"bddelta_"));
    CHKERRQ(KSPSetFromOptions(ctx->kBD));
  }

  if (mat_ctx->l2g_lambda_only) {
    CHKERRQ(PetscObjectReference((PetscObject)mat_ctx->l2g_lambda_only));
    fetidppc_ctx->l2g_lambda = mat_ctx->l2g_lambda_only;
  } else {
    CHKERRQ(PetscObjectReference((PetscObject)mat_ctx->l2g_lambda));
    fetidppc_ctx->l2g_lambda = mat_ctx->l2g_lambda;
  }
  /* Dirichlet preconditioner */
  CHKERRQ(PetscOptionsGetBool(NULL,((PetscObject)fetimat)->prefix,"-pc_lumped",&lumped,NULL));
  if (!lumped) {
    IS        iV;
    PetscBool discrete_harmonic = PETSC_FALSE;

    CHKERRQ(PetscObjectQuery((PetscObject)fetidppc_ctx->pc,"__KSPFETIDP_iV",(PetscObject*)&iV));
    if (iV) {
      CHKERRQ(PetscOptionsGetBool(NULL,((PetscObject)fetimat)->prefix,"-pc_discrete_harmonic",&discrete_harmonic,NULL));
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
        CHKERRQ(PetscObjectQuery((PetscObject)sub_schurs->A,"__KSPFETIDP_iP",(PetscObject*)&iP));
        if (iP) reuse = PETSC_TRUE;
      }
      if (!reuse) {
        IS       aB;
        PetscInt nb;
        CHKERRQ(ISGetLocalSize(pcis->is_B_local,&nb));
        CHKERRQ(ISCreateStride(PetscObjectComm((PetscObject)pcis->A_II),nb,0,1,&aB));
        CHKERRQ(MatCreateSubMatrix(pcis->A_II,iV,iV,MAT_INITIAL_MATRIX,&A_II));
        CHKERRQ(MatCreateSubMatrix(pcis->A_IB,iV,aB,MAT_INITIAL_MATRIX,&A_IB));
        CHKERRQ(MatCreateSubMatrix(pcis->A_BI,aB,iV,MAT_INITIAL_MATRIX,&A_BI));
        CHKERRQ(ISDestroy(&aB));
      } else {
        CHKERRQ(MatCreateSubMatrix(sub_schurs->A,pcis->is_I_local,pcis->is_B_local,MAT_INITIAL_MATRIX,&A_IB));
        CHKERRQ(MatCreateSubMatrix(sub_schurs->A,pcis->is_B_local,pcis->is_I_local,MAT_INITIAL_MATRIX,&A_BI));
        CHKERRQ(PetscObjectReference((PetscObject)pcis->A_II));
        A_II = pcis->A_II;
      }
      CHKERRQ(MatCreateSchurComplement(A_II,A_II,A_IB,A_BI,pcis->A_BB,&fetidppc_ctx->S_j));

      /* propagate settings of solver */
      CHKERRQ(MatSchurComplementGetKSP(fetidppc_ctx->S_j,&sksp));
      CHKERRQ(KSPGetType(pcis->ksp_D,&ksptype));
      CHKERRQ(KSPSetType(sksp,ksptype));
      CHKERRQ(KSPGetPC(pcis->ksp_D,&pc));
      CHKERRQ(PetscObjectTypeCompare((PetscObject)pc,PCSHELL,&isshell));
      if (!isshell) {
        MatSolverType    solver;
        PCType           pctype;

        CHKERRQ(PCGetType(pc,&pctype));
        CHKERRQ(PCFactorGetMatSolverType(pc,(MatSolverType*)&solver));
        CHKERRQ(KSPGetPC(sksp,&pc));
        CHKERRQ(PCSetType(pc,pctype));
        if (solver) {
          CHKERRQ(PCFactorSetMatSolverType(pc,solver));
        }
      } else {
        CHKERRQ(KSPGetPC(sksp,&pc));
        CHKERRQ(PCSetType(pc,PCLU));
      }
      CHKERRQ(MatDestroy(&A_II));
      CHKERRQ(MatDestroy(&A_IB));
      CHKERRQ(MatDestroy(&A_BI));
      CHKERRQ(MatGetOptionsPrefix(fetimat,&prefix));
      CHKERRQ(KSPSetOptionsPrefix(sksp,prefix));
      CHKERRQ(KSPAppendOptionsPrefix(sksp,"harmonic_"));
      CHKERRQ(KSPSetFromOptions(sksp));
      if (reuse) {
        CHKERRQ(KSPSetPC(sksp,sub_schurs->reuse_solver->interior_solver));
        CHKERRQ(PetscObjectIncrementTabLevel((PetscObject)sub_schurs->reuse_solver->interior_solver,(PetscObject)sksp,0));
      }
    } else { /* default Dirichlet preconditioner is pde-harmonic */
      CHKERRQ(MatCreateSchurComplement(pcis->A_II,pcis->A_II,pcis->A_IB,pcis->A_BI,pcis->A_BB,&fetidppc_ctx->S_j));
      CHKERRQ(MatSchurComplementSetKSP(fetidppc_ctx->S_j,pcis->ksp_D));
    }
  } else {
    CHKERRQ(PetscObjectReference((PetscObject)pcis->A_BB));
    fetidppc_ctx->S_j = pcis->A_BB;
  }
  /* saddle-point */
  if (mat_ctx->xPg) {
    CHKERRQ(PetscObjectReference((PetscObject)mat_ctx->xPg));
    fetidppc_ctx->xPg = mat_ctx->xPg;
    CHKERRQ(PetscObjectReference((PetscObject)mat_ctx->yPg));
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
  CHKERRQ(MatShellGetContext(fetimat,&mat_ctx));
  pcis = (PC_IS*)mat_ctx->pc->data;
  pcbddc = (PC_BDDC*)mat_ctx->pc->data;
  /* Application of B_delta^T */
  CHKERRQ(VecSet(pcis->vec1_B,0.));
  CHKERRQ(VecScatterBegin(mat_ctx->l2g_lambda,x,mat_ctx->lambda_local,INSERT_VALUES,SCATTER_REVERSE));
  CHKERRQ(VecScatterEnd(mat_ctx->l2g_lambda,x,mat_ctx->lambda_local,INSERT_VALUES,SCATTER_REVERSE));
  CHKERRQ(MatMultTranspose(mat_ctx->B_delta,mat_ctx->lambda_local,pcis->vec1_B));

  /* Add contribution from saddle point */
  if (mat_ctx->l2g_p) {
    CHKERRQ(VecScatterBegin(mat_ctx->l2g_p,x,mat_ctx->vP,INSERT_VALUES,SCATTER_REVERSE));
    CHKERRQ(VecScatterEnd(mat_ctx->l2g_p,x,mat_ctx->vP,INSERT_VALUES,SCATTER_REVERSE));
    if (pcbddc->switch_static) {
      if (trans) {
        CHKERRQ(MatMultTranspose(mat_ctx->B_BI,mat_ctx->vP,pcis->vec1_D));
      } else {
        CHKERRQ(MatMult(mat_ctx->Bt_BI,mat_ctx->vP,pcis->vec1_D));
      }
    }
    if (trans) {
      CHKERRQ(MatMultTransposeAdd(mat_ctx->B_BB,mat_ctx->vP,pcis->vec1_B,pcis->vec1_B));
    } else {
      CHKERRQ(MatMultAdd(mat_ctx->Bt_BB,mat_ctx->vP,pcis->vec1_B,pcis->vec1_B));
    }
  } else {
    if (pcbddc->switch_static) {
      CHKERRQ(VecSet(pcis->vec1_D,0.0));
    }
  }
  /* Application of \widetilde{S}^-1 */
  CHKERRQ(PetscArrayzero(pcbddc->benign_p0,pcbddc->benign_n));
  CHKERRQ(PCBDDCApplyInterfacePreconditioner(mat_ctx->pc,trans));
  CHKERRQ(PetscArrayzero(pcbddc->benign_p0,pcbddc->benign_n));
  CHKERRQ(VecSet(y,0.0));
  /* Application of B_delta */
  CHKERRQ(MatMult(mat_ctx->B_delta,pcis->vec1_B,mat_ctx->lambda_local));
  /* Contribution from boundary pressures */
  if (mat_ctx->C) {
    const PetscScalar *lx;
    PetscScalar       *ly;

    /* pressure ordered first in the local part of x and y */
    CHKERRQ(VecGetArrayRead(x,&lx));
    CHKERRQ(VecGetArray(y,&ly));
    CHKERRQ(VecPlaceArray(mat_ctx->xPg,lx));
    CHKERRQ(VecPlaceArray(mat_ctx->yPg,ly));
    if (trans) {
      CHKERRQ(MatMultTranspose(mat_ctx->C,mat_ctx->xPg,mat_ctx->yPg));
    } else {
      CHKERRQ(MatMult(mat_ctx->C,mat_ctx->xPg,mat_ctx->yPg));
    }
    CHKERRQ(VecResetArray(mat_ctx->xPg));
    CHKERRQ(VecResetArray(mat_ctx->yPg));
    CHKERRQ(VecRestoreArrayRead(x,&lx));
    CHKERRQ(VecRestoreArray(y,&ly));
  }
  /* Add contribution from saddle point */
  if (mat_ctx->l2g_p) {
    if (trans) {
      CHKERRQ(MatMultTranspose(mat_ctx->Bt_BB,pcis->vec1_B,mat_ctx->vP));
    } else {
      CHKERRQ(MatMult(mat_ctx->B_BB,pcis->vec1_B,mat_ctx->vP));
    }
    if (pcbddc->switch_static) {
      if (trans) {
        CHKERRQ(MatMultTransposeAdd(mat_ctx->Bt_BI,pcis->vec1_D,mat_ctx->vP,mat_ctx->vP));
      } else {
        CHKERRQ(MatMultAdd(mat_ctx->B_BI,pcis->vec1_D,mat_ctx->vP,mat_ctx->vP));
      }
    }
    CHKERRQ(VecScatterBegin(mat_ctx->l2g_p,mat_ctx->vP,y,ADD_VALUES,SCATTER_FORWARD));
    CHKERRQ(VecScatterEnd(mat_ctx->l2g_p,mat_ctx->vP,y,ADD_VALUES,SCATTER_FORWARD));
  }
  CHKERRQ(VecScatterBegin(mat_ctx->l2g_lambda,mat_ctx->lambda_local,y,ADD_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterEnd(mat_ctx->l2g_lambda,mat_ctx->lambda_local,y,ADD_VALUES,SCATTER_FORWARD));
  PetscFunctionReturn(0);
}

PetscErrorCode FETIDPMatMult(Mat fetimat, Vec x, Vec y)
{
  PetscFunctionBegin;
  CHKERRQ(FETIDPMatMult_Kernel(fetimat,x,y,PETSC_FALSE));
  PetscFunctionReturn(0);
}

PetscErrorCode FETIDPMatMultTranspose(Mat fetimat, Vec x, Vec y)
{
  PetscFunctionBegin;
  CHKERRQ(FETIDPMatMult_Kernel(fetimat,x,y,PETSC_TRUE));
  PetscFunctionReturn(0);
}

PetscErrorCode FETIDPPCApply_Kernel(PC fetipc, Vec x, Vec y, PetscBool trans)
{
  FETIDPPC_ctx   pc_ctx;
  PC_IS          *pcis;

  PetscFunctionBegin;
  CHKERRQ(PCShellGetContext(fetipc,&pc_ctx));
  pcis = (PC_IS*)pc_ctx->pc->data;
  /* Application of B_Ddelta^T */
  CHKERRQ(VecScatterBegin(pc_ctx->l2g_lambda,x,pc_ctx->lambda_local,INSERT_VALUES,SCATTER_REVERSE));
  CHKERRQ(VecScatterEnd(pc_ctx->l2g_lambda,x,pc_ctx->lambda_local,INSERT_VALUES,SCATTER_REVERSE));
  CHKERRQ(VecSet(pcis->vec2_B,0.0));
  CHKERRQ(MatMultTranspose(pc_ctx->B_Ddelta,pc_ctx->lambda_local,pcis->vec2_B));
  /* Application of local Schur complement */
  if (trans) {
    CHKERRQ(MatMultTranspose(pc_ctx->S_j,pcis->vec2_B,pcis->vec1_B));
  } else {
    CHKERRQ(MatMult(pc_ctx->S_j,pcis->vec2_B,pcis->vec1_B));
  }
  /* Application of B_Ddelta */
  CHKERRQ(MatMult(pc_ctx->B_Ddelta,pcis->vec1_B,pc_ctx->lambda_local));
  CHKERRQ(VecSet(y,0.0));
  CHKERRQ(VecScatterBegin(pc_ctx->l2g_lambda,pc_ctx->lambda_local,y,ADD_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterEnd(pc_ctx->l2g_lambda,pc_ctx->lambda_local,y,ADD_VALUES,SCATTER_FORWARD));
  PetscFunctionReturn(0);
}

PetscErrorCode FETIDPPCApply(PC pc, Vec x, Vec y)
{
  PetscFunctionBegin;
  CHKERRQ(FETIDPPCApply_Kernel(pc,x,y,PETSC_FALSE));
  PetscFunctionReturn(0);
}

PetscErrorCode FETIDPPCApplyTranspose(PC pc, Vec x, Vec y)
{
  PetscFunctionBegin;
  CHKERRQ(FETIDPPCApply_Kernel(pc,x,y,PETSC_TRUE));
  PetscFunctionReturn(0);
}

PetscErrorCode FETIDPPCView(PC pc, PetscViewer viewer)
{
  FETIDPPC_ctx      pc_ctx;
  PetscBool         iascii;
  PetscViewer       sviewer;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii));
  if (iascii) {
    PetscMPIInt rank;
    PetscBool   isschur,isshell;

    CHKERRQ(PCShellGetContext(pc,&pc_ctx));
    CHKERRMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)pc),&rank));
    CHKERRQ(PetscObjectTypeCompare((PetscObject)pc_ctx->S_j,MATSCHURCOMPLEMENT,&isschur));
    if (isschur) {
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"  Dirichlet preconditioner (just from rank 0)\n"));
    } else {
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"  Lumped preconditioner (just from rank 0)\n"));
    }
    CHKERRQ(PetscViewerGetSubViewer(viewer,PetscObjectComm((PetscObject)pc_ctx->S_j),&sviewer));
    if (rank == 0) {
      CHKERRQ(PetscViewerPushFormat(sviewer,PETSC_VIEWER_ASCII_INFO));
      CHKERRQ(PetscViewerASCIIPushTab(sviewer));
      CHKERRQ(MatView(pc_ctx->S_j,sviewer));
      CHKERRQ(PetscViewerASCIIPopTab(sviewer));
      CHKERRQ(PetscViewerPopFormat(sviewer));
    }
    CHKERRQ(PetscViewerRestoreSubViewer(viewer,PetscObjectComm((PetscObject)pc_ctx->S_j),&sviewer));
    CHKERRQ(PetscObjectTypeCompare((PetscObject)pc_ctx->B_Ddelta,MATSHELL,&isshell));
    if (isshell) {
      BDdelta_DN ctx;
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"  FETI-DP BDdelta: DB^t * (B D^-1 B^t)^-1 for deluxe scaling (just from rank 0)\n"));
      CHKERRQ(MatShellGetContext(pc_ctx->B_Ddelta,&ctx));
      CHKERRQ(PetscViewerGetSubViewer(viewer,PetscObjectComm((PetscObject)pc_ctx->S_j),&sviewer));
      if (rank == 0) {
        PetscInt tl;

        CHKERRQ(PetscViewerASCIIGetTab(sviewer,&tl));
        CHKERRQ(PetscObjectSetTabLevel((PetscObject)ctx->kBD,tl));
        CHKERRQ(KSPView(ctx->kBD,sviewer));
        CHKERRQ(PetscViewerPushFormat(sviewer,PETSC_VIEWER_ASCII_INFO));
        CHKERRQ(MatView(ctx->BD,sviewer));
        CHKERRQ(PetscViewerPopFormat(sviewer));
      }
      CHKERRQ(PetscViewerRestoreSubViewer(viewer,PetscObjectComm((PetscObject)pc_ctx->S_j),&sviewer));
    }
    CHKERRQ(PetscViewerFlush(viewer));
  }
  PetscFunctionReturn(0);
}
