#include <../src/ksp/pc/impls/bddc/bddc.h>
#include <../src/ksp/pc/impls/bddc/bddcprivate.h>
#include <petscblaslapack.h>

static PetscErrorCode MatMult_BDdelta_deluxe_nonred(Mat A, Vec x, Vec y)
{
  BDdelta_DN     ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(A,(void**)&ctx);CHKERRQ(ierr);
  ierr = MatMultTranspose(ctx->BD,x,ctx->work);CHKERRQ(ierr);
  ierr = KSPSolveTranspose(ctx->kBD,ctx->work,y);CHKERRQ(ierr);
  /* No PC so cannot propagate up failure in KSPSolveTranspose() */
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMultTranspose_BDdelta_deluxe_nonred(Mat A, Vec x, Vec y)
{
  BDdelta_DN     ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(A,(void**)&ctx);CHKERRQ(ierr);
  ierr = KSPSolve(ctx->kBD,x,ctx->work);CHKERRQ(ierr);
  /* No PC so cannot propagate up failure in KSPSolve() */
  ierr = MatMult(ctx->BD,ctx->work,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDestroy_BDdelta_deluxe_nonred(Mat A)
{
  BDdelta_DN     ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(A,(void**)&ctx);CHKERRQ(ierr);
  ierr = MatDestroy(&ctx->BD);CHKERRQ(ierr);
  ierr = KSPDestroy(&ctx->kBD);CHKERRQ(ierr);
  ierr = VecDestroy(&ctx->work);CHKERRQ(ierr);
  ierr = PetscFree(ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


PetscErrorCode PCBDDCCreateFETIDPMatContext(PC pc, FETIDPMat_ctx *fetidpmat_ctx)
{
  FETIDPMat_ctx  newctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNew(&newctx);CHKERRQ(ierr);
  /* increase the reference count for BDDC preconditioner */
  ierr = PetscObjectReference((PetscObject)pc);CHKERRQ(ierr);
  newctx->pc              = pc;
  *fetidpmat_ctx          = newctx;
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCCreateFETIDPPCContext(PC pc, FETIDPPC_ctx *fetidppc_ctx)
{
  FETIDPPC_ctx   newctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNew(&newctx);CHKERRQ(ierr);
  /* increase the reference count for BDDC preconditioner */
  ierr = PetscObjectReference((PetscObject)pc);CHKERRQ(ierr);
  newctx->pc              = pc;
  *fetidppc_ctx           = newctx;
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCDestroyFETIDPMat(Mat A)
{
  FETIDPMat_ctx  mat_ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(A,(void**)&mat_ctx);CHKERRQ(ierr);
  ierr = VecDestroy(&mat_ctx->lambda_local);CHKERRQ(ierr);
  ierr = VecDestroy(&mat_ctx->temp_solution_D);CHKERRQ(ierr);
  ierr = VecDestroy(&mat_ctx->temp_solution_B);CHKERRQ(ierr);
  ierr = MatDestroy(&mat_ctx->B_delta);CHKERRQ(ierr);
  ierr = MatDestroy(&mat_ctx->B_Ddelta);CHKERRQ(ierr);
  ierr = MatDestroy(&mat_ctx->B_BB);CHKERRQ(ierr);
  ierr = MatDestroy(&mat_ctx->B_BI);CHKERRQ(ierr);
  ierr = MatDestroy(&mat_ctx->Bt_BB);CHKERRQ(ierr);
  ierr = MatDestroy(&mat_ctx->Bt_BI);CHKERRQ(ierr);
  ierr = MatDestroy(&mat_ctx->C);CHKERRQ(ierr);
  ierr = VecDestroy(&mat_ctx->rhs_flip);CHKERRQ(ierr);
  ierr = VecDestroy(&mat_ctx->vP);CHKERRQ(ierr);
  ierr = VecDestroy(&mat_ctx->xPg);CHKERRQ(ierr);
  ierr = VecDestroy(&mat_ctx->yPg);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&mat_ctx->l2g_lambda);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&mat_ctx->l2g_lambda_only);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&mat_ctx->l2g_p);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&mat_ctx->g2g_p);CHKERRQ(ierr);
  ierr = PCDestroy(&mat_ctx->pc);CHKERRQ(ierr); /* decrease PCBDDC reference count */
  ierr = ISDestroy(&mat_ctx->pressure);CHKERRQ(ierr);
  ierr = ISDestroy(&mat_ctx->lagrange);CHKERRQ(ierr);
  ierr = PetscFree(mat_ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCDestroyFETIDPPC(PC pc)
{
  FETIDPPC_ctx   pc_ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PCShellGetContext(pc,(void**)&pc_ctx);CHKERRQ(ierr);
  ierr = VecDestroy(&pc_ctx->lambda_local);CHKERRQ(ierr);
  ierr = MatDestroy(&pc_ctx->B_Ddelta);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&pc_ctx->l2g_lambda);CHKERRQ(ierr);
  ierr = MatDestroy(&pc_ctx->S_j);CHKERRQ(ierr);
  ierr = PCDestroy(&pc_ctx->pc);CHKERRQ(ierr); /* decrease PCBDDC reference count */
  ierr = VecDestroy(&pc_ctx->xPg);CHKERRQ(ierr);
  ierr = VecDestroy(&pc_ctx->yPg);CHKERRQ(ierr);
  ierr = PetscFree(pc_ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCSetupFETIDPMatContext(FETIDPMat_ctx fetidpmat_ctx )
{
  PetscErrorCode ierr;
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
  ierr = PetscObjectGetComm((PetscObject)(fetidpmat_ctx->pc),&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);

  /* saddlepoint */
  nPl      = 0;
  nPg      = 0;
  nPgl     = 0;
  gP       = NULL;
  pP       = NULL;
  l2gmap_p = NULL;
  play     = NULL;
  ierr = PetscObjectQuery((PetscObject)fetidpmat_ctx->pc,"__KSPFETIDP_pP",(PetscObject*)&pP);CHKERRQ(ierr);
  if (pP) { /* saddle point */
    /* subdomain pressures in global numbering */
    ierr = PetscObjectQuery((PetscObject)fetidpmat_ctx->pc,"__KSPFETIDP_gP",(PetscObject*)&gP);CHKERRQ(ierr);
    if (!gP) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"gP not present");
    ierr = ISGetLocalSize(gP,&nPl);CHKERRQ(ierr);
    ierr = VecCreate(PETSC_COMM_SELF,&fetidpmat_ctx->vP);CHKERRQ(ierr);
    ierr = VecSetSizes(fetidpmat_ctx->vP,nPl,nPl);CHKERRQ(ierr);
    ierr = VecSetType(fetidpmat_ctx->vP,VECSTANDARD);CHKERRQ(ierr);
    ierr = VecSetUp(fetidpmat_ctx->vP);CHKERRQ(ierr);

    /* pressure matrix */
    ierr = PetscObjectQuery((PetscObject)fetidpmat_ctx->pc,"__KSPFETIDP_C",(PetscObject*)&fetidpmat_ctx->C);CHKERRQ(ierr);
    if (!fetidpmat_ctx->C) { /* null pressure block, compute layout and global numbering for pressures */
      IS Pg;

      ierr = ISRenumber(gP,NULL,&nPg,&Pg);CHKERRQ(ierr);
      ierr = ISLocalToGlobalMappingCreateIS(Pg,&l2gmap_p);CHKERRQ(ierr);
      ierr = ISDestroy(&Pg);CHKERRQ(ierr);
      ierr = PetscLayoutCreate(comm,&play);CHKERRQ(ierr);
      ierr = PetscLayoutSetBlockSize(play,1);CHKERRQ(ierr);
      ierr = PetscLayoutSetSize(play,nPg);CHKERRQ(ierr);
      ierr = ISGetLocalSize(pP,&nPgl);CHKERRQ(ierr);
      ierr = PetscLayoutSetLocalSize(play,nPgl);CHKERRQ(ierr);
      ierr = PetscLayoutSetUp(play);CHKERRQ(ierr);
    } else {
      ierr = PetscObjectReference((PetscObject)fetidpmat_ctx->C);CHKERRQ(ierr);
      ierr = MatGetLocalToGlobalMapping(fetidpmat_ctx->C,&l2gmap_p,NULL);CHKERRQ(ierr);
      ierr = PetscObjectReference((PetscObject)l2gmap_p);CHKERRQ(ierr);
      ierr = MatGetSize(fetidpmat_ctx->C,&nPg,NULL);CHKERRQ(ierr);
      ierr = MatGetLocalSize(fetidpmat_ctx->C,NULL,&nPgl);CHKERRQ(ierr);
      ierr = MatGetLayouts(fetidpmat_ctx->C,NULL,&llay);CHKERRQ(ierr);
      ierr = PetscLayoutReference(llay,&play);CHKERRQ(ierr);
    }
    ierr = VecCreateMPIWithArray(comm,1,nPgl,nPg,NULL,&fetidpmat_ctx->xPg);CHKERRQ(ierr);
    ierr = VecCreateMPIWithArray(comm,1,nPgl,nPg,NULL,&fetidpmat_ctx->yPg);CHKERRQ(ierr);

    /* import matrices for pressures coupling */
    ierr = PetscObjectQuery((PetscObject)fetidpmat_ctx->pc,"__KSPFETIDP_B_BI",(PetscObject*)&fetidpmat_ctx->B_BI);CHKERRQ(ierr);
    if (!fetidpmat_ctx->B_BI) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"B_BI not present");
    ierr = PetscObjectReference((PetscObject)fetidpmat_ctx->B_BI);CHKERRQ(ierr);

    ierr = PetscObjectQuery((PetscObject)fetidpmat_ctx->pc,"__KSPFETIDP_B_BB",(PetscObject*)&fetidpmat_ctx->B_BB);CHKERRQ(ierr);
    if (!fetidpmat_ctx->B_BB) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"B_BB not present");
    ierr = PetscObjectReference((PetscObject)fetidpmat_ctx->B_BB);CHKERRQ(ierr);

    ierr = PetscObjectQuery((PetscObject)fetidpmat_ctx->pc,"__KSPFETIDP_Bt_BI",(PetscObject*)&fetidpmat_ctx->Bt_BI);CHKERRQ(ierr);
    if (!fetidpmat_ctx->Bt_BI) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Bt_BI not present");
    ierr = PetscObjectReference((PetscObject)fetidpmat_ctx->Bt_BI);CHKERRQ(ierr);

    ierr = PetscObjectQuery((PetscObject)fetidpmat_ctx->pc,"__KSPFETIDP_Bt_BB",(PetscObject*)&fetidpmat_ctx->Bt_BB);CHKERRQ(ierr);
    if (!fetidpmat_ctx->Bt_BB) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Bt_BB not present");
    ierr = PetscObjectReference((PetscObject)fetidpmat_ctx->Bt_BB);CHKERRQ(ierr);

    ierr = PetscObjectQuery((PetscObject)fetidpmat_ctx->pc,"__KSPFETIDP_flip" ,(PetscObject*)&fetidpmat_ctx->rhs_flip);CHKERRQ(ierr);
    if (fetidpmat_ctx->rhs_flip) {
      ierr = PetscObjectReference((PetscObject)fetidpmat_ctx->rhs_flip);CHKERRQ(ierr);
    }
  }

  /* Default type of lagrange multipliers is non-redundant */
  fully_redundant = fetidpmat_ctx->fully_redundant;

  /* Evaluate local and global number of lagrange multipliers */
  ierr = VecSet(pcis->vec1_N,0.0);CHKERRQ(ierr);
  n_local_lambda = 0;
  partial_sum = 0;
  n_boundary_dofs = 0;
  s = 0;

  /* Get Vertices used to define the BDDC */
  ierr = PCBDDCGraphGetCandidatesIS(pcbddc->mat_graph,NULL,NULL,NULL,NULL,&isvert);CHKERRQ(ierr);
  ierr = ISGetLocalSize(isvert,&n_vertices);CHKERRQ(ierr);
  ierr = ISGetIndices(isvert,&vertex_indices);CHKERRQ(ierr);

  dual_size = pcis->n_B-n_vertices;
  ierr = PetscMalloc1(dual_size,&dual_dofs_boundary_indices);CHKERRQ(ierr);
  ierr = PetscMalloc1(dual_size,&aux_local_numbering_1);CHKERRQ(ierr);
  ierr = PetscMalloc1(dual_size,&aux_local_numbering_2);CHKERRQ(ierr);

  ierr = VecGetArray(pcis->vec1_N,&array);CHKERRQ(ierr);
  for (i=0;i<pcis->n;i++){
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
  ierr = VecRestoreArray(pcis->vec1_N,&array);CHKERRQ(ierr);
  ierr = ISRestoreIndices(isvert,&vertex_indices);CHKERRQ(ierr);
  ierr = PCBDDCGraphRestoreCandidatesIS(pcbddc->mat_graph,NULL,NULL,NULL,NULL,&isvert);CHKERRQ(ierr);
  dual_size = partial_sum;

  /* compute global ordering of lagrange multipliers and associate l2g map */
  ierr = ISCreateGeneral(comm,partial_sum,aux_local_numbering_1,PETSC_COPY_VALUES,&subset_n);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingApplyIS(pcis->mapping,subset_n,&subset);CHKERRQ(ierr);
  ierr = ISDestroy(&subset_n);CHKERRQ(ierr);
  ierr = ISCreateGeneral(comm,partial_sum,aux_local_numbering_2,PETSC_OWN_POINTER,&subset_mult);CHKERRQ(ierr);
  ierr = ISRenumber(subset,subset_mult,&fetidpmat_ctx->n_lambda,&subset_n);CHKERRQ(ierr);
  ierr = ISDestroy(&subset);CHKERRQ(ierr);

  if (PetscDefined(USE_DEBUG)) {
    ierr = VecSet(pcis->vec1_global,0.0);CHKERRQ(ierr);
    ierr = VecScatterBegin(matis->rctx,pcis->vec1_N,pcis->vec1_global,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterEnd(matis->rctx,pcis->vec1_N,pcis->vec1_global,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecSum(pcis->vec1_global,&scalar_value);CHKERRQ(ierr);
    i = (PetscInt)PetscRealPart(scalar_value);
    if (i != fetidpmat_ctx->n_lambda) SETERRQ2(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Global number of multipliers mismatch! (%D != %D)",fetidpmat_ctx->n_lambda,i);
  }

  /* init data for scaling factors exchange */
  if (!pcbddc->use_deluxe_scaling) {
    PetscInt    *ptrs_buffer,neigh_position;
    PetscScalar *send_buffer,*recv_buffer;
    MPI_Request *send_reqs,*recv_reqs;

    partial_sum = 0;
    ierr = PetscMalloc1(pcis->n_neigh,&ptrs_buffer);CHKERRQ(ierr);
    ierr = PetscMalloc1(PetscMax(pcis->n_neigh-1,0),&send_reqs);CHKERRQ(ierr);
    ierr = PetscMalloc1(PetscMax(pcis->n_neigh-1,0),&recv_reqs);CHKERRQ(ierr);
    ierr = PetscMalloc1(pcis->n+1,&all_factors);CHKERRQ(ierr);
    if (pcis->n_neigh > 0) ptrs_buffer[0]=0;
    for (i=1;i<pcis->n_neigh;i++) {
      partial_sum += pcis->n_shared[i];
      ptrs_buffer[i] = ptrs_buffer[i-1]+pcis->n_shared[i];
    }
    ierr = PetscMalloc1(partial_sum,&send_buffer);CHKERRQ(ierr);
    ierr = PetscMalloc1(partial_sum,&recv_buffer);CHKERRQ(ierr);
    ierr = PetscMalloc1(partial_sum,&all_factors[0]);CHKERRQ(ierr);
    for (i=0;i<pcis->n-1;i++) {
      j = mat_graph->count[i];
      all_factors[i+1]=all_factors[i]+j;
    }

    /* scatter B scaling to N vec */
    ierr = VecScatterBegin(pcis->N_to_B,pcis->D,pcis->vec1_N,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterEnd(pcis->N_to_B,pcis->D,pcis->vec1_N,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    /* communications */
    ierr = VecGetArrayRead(pcis->vec1_N,(const PetscScalar**)&array);CHKERRQ(ierr);
    for (i=1;i<pcis->n_neigh;i++) {
      for (j=0;j<pcis->n_shared[i];j++) {
        send_buffer[ptrs_buffer[i-1]+j]=array[pcis->shared[i][j]];
      }
      ierr = PetscMPIIntCast(ptrs_buffer[i]-ptrs_buffer[i-1],&buf_size);CHKERRQ(ierr);
      ierr = PetscMPIIntCast(pcis->neigh[i],&neigh);CHKERRQ(ierr);
      ierr = MPI_Isend(&send_buffer[ptrs_buffer[i-1]],buf_size,MPIU_SCALAR,neigh,0,comm,&send_reqs[i-1]);CHKERRQ(ierr);
      ierr = MPI_Irecv(&recv_buffer[ptrs_buffer[i-1]],buf_size,MPIU_SCALAR,neigh,0,comm,&recv_reqs[i-1]);CHKERRQ(ierr);
    }
    ierr = VecRestoreArrayRead(pcis->vec1_N,(const PetscScalar**)&array);CHKERRQ(ierr);
    if (pcis->n_neigh > 0) {
      ierr = MPI_Waitall(pcis->n_neigh-1,recv_reqs,MPI_STATUSES_IGNORE);CHKERRQ(ierr);
    }
    /* put values in correct places */
    for (i=1;i<pcis->n_neigh;i++) {
      for (j=0;j<pcis->n_shared[i];j++) {
        k = pcis->shared[i][j];
        neigh_position = 0;
        while(mat_graph->neighbours_set[k][neigh_position] != pcis->neigh[i]) {neigh_position++;}
        all_factors[k][neigh_position]=recv_buffer[ptrs_buffer[i-1]+j];
      }
    }
    if (pcis->n_neigh > 0) {
      ierr = MPI_Waitall(pcis->n_neigh-1,send_reqs,MPI_STATUSES_IGNORE);CHKERRQ(ierr);
    }
    ierr = PetscFree(send_reqs);CHKERRQ(ierr);
    ierr = PetscFree(recv_reqs);CHKERRQ(ierr);
    ierr = PetscFree(send_buffer);CHKERRQ(ierr);
    ierr = PetscFree(recv_buffer);CHKERRQ(ierr);
    ierr = PetscFree(ptrs_buffer);CHKERRQ(ierr);
  }

  /* Compute B and B_delta (local actions) */
  ierr = PetscMalloc1(pcis->n_neigh,&aux_sums);CHKERRQ(ierr);
  ierr = PetscMalloc1(n_local_lambda,&l2g_indices);CHKERRQ(ierr);
  ierr = PetscMalloc1(n_local_lambda,&vals_B_delta);CHKERRQ(ierr);
  ierr = PetscMalloc1(n_local_lambda,&cols_B_delta);CHKERRQ(ierr);
  if (!pcbddc->use_deluxe_scaling) {
    ierr = PetscMalloc1(n_local_lambda,&scaling_factors);CHKERRQ(ierr);
  } else {
    scaling_factors = NULL;
    all_factors     = NULL;
  }
  ierr = ISGetIndices(subset_n,&aux_global_numbering);CHKERRQ(ierr);
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
    while(n_neg_values < j && mat_graph->neighbours_set[aux_local_numbering_1[i]][n_neg_values] < rank) {n_neg_values++;}
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
      if ( n_neg_values > 0 ) { /* there's a rank next to me to the left */
        vals_B_delta   [partial_sum+n_neg_values-1]=-1.0;
      }
      if ( n_neg_values < j ) { /* there's a rank next to me to the right */
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
  ierr = ISRestoreIndices(subset_n,&aux_global_numbering);CHKERRQ(ierr);
  ierr = ISDestroy(&subset_mult);CHKERRQ(ierr);
  ierr = ISDestroy(&subset_n);CHKERRQ(ierr);
  ierr = PetscFree(aux_sums);CHKERRQ(ierr);
  ierr = PetscFree(aux_local_numbering_1);CHKERRQ(ierr);
  ierr = PetscFree(dual_dofs_boundary_indices);CHKERRQ(ierr);
  if (all_factors) {
    ierr = PetscFree(all_factors[0]);CHKERRQ(ierr);
    ierr = PetscFree(all_factors);CHKERRQ(ierr);
  }

  /* Create local part of B_delta */
  ierr = MatCreate(PETSC_COMM_SELF,&fetidpmat_ctx->B_delta);CHKERRQ(ierr);
  ierr = MatSetSizes(fetidpmat_ctx->B_delta,n_local_lambda,pcis->n_B,n_local_lambda,pcis->n_B);CHKERRQ(ierr);
  ierr = MatSetType(fetidpmat_ctx->B_delta,MATSEQAIJ);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(fetidpmat_ctx->B_delta,1,NULL);CHKERRQ(ierr);
  ierr = MatSetOption(fetidpmat_ctx->B_delta,MAT_IGNORE_ZERO_ENTRIES,PETSC_TRUE);CHKERRQ(ierr);
  for (i=0;i<n_local_lambda;i++) {
    ierr = MatSetValue(fetidpmat_ctx->B_delta,i,cols_B_delta[i],vals_B_delta[i],INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = PetscFree(vals_B_delta);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(fetidpmat_ctx->B_delta,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(fetidpmat_ctx->B_delta,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  BD1 = NULL;
  BD2 = NULL;
  if (fully_redundant) {
    if (pcbddc->use_deluxe_scaling) SETERRQ(comm,PETSC_ERR_SUP,"Deluxe FETIDP with fully-redundant multipliers to be implemented");
    ierr = MatCreate(PETSC_COMM_SELF,&ScalingMat);CHKERRQ(ierr);
    ierr = MatSetSizes(ScalingMat,n_local_lambda,n_local_lambda,n_local_lambda,n_local_lambda);CHKERRQ(ierr);
    ierr = MatSetType(ScalingMat,MATSEQAIJ);CHKERRQ(ierr);
    ierr = MatSeqAIJSetPreallocation(ScalingMat,1,NULL);CHKERRQ(ierr);
    for (i=0;i<n_local_lambda;i++) {
      ierr = MatSetValue(ScalingMat,i,i,scaling_factors[i],INSERT_VALUES);CHKERRQ(ierr);
    }
    ierr = MatAssemblyBegin(ScalingMat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(ScalingMat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatMatMult(ScalingMat,fetidpmat_ctx->B_delta,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&fetidpmat_ctx->B_Ddelta);CHKERRQ(ierr);
    ierr = MatDestroy(&ScalingMat);CHKERRQ(ierr);
  } else {
    ierr = MatCreate(PETSC_COMM_SELF,&fetidpmat_ctx->B_Ddelta);CHKERRQ(ierr);
    ierr = MatSetSizes(fetidpmat_ctx->B_Ddelta,n_local_lambda,pcis->n_B,n_local_lambda,pcis->n_B);CHKERRQ(ierr);
    if (!pcbddc->use_deluxe_scaling || !pcbddc->sub_schurs) {
      ierr = MatSetType(fetidpmat_ctx->B_Ddelta,MATSEQAIJ);CHKERRQ(ierr);
      ierr = MatSeqAIJSetPreallocation(fetidpmat_ctx->B_Ddelta,1,NULL);CHKERRQ(ierr);
      for (i=0;i<n_local_lambda;i++) {
        ierr = MatSetValue(fetidpmat_ctx->B_Ddelta,i,cols_B_delta[i],scaling_factors[i],INSERT_VALUES);CHKERRQ(ierr);
      }
      ierr = MatAssemblyBegin(fetidpmat_ctx->B_Ddelta,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      ierr = MatAssemblyEnd(fetidpmat_ctx->B_Ddelta,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    } else {
      /* scaling as in Klawonn-Widlund 1999 */
      PCBDDCDeluxeScaling deluxe_ctx = pcbddc->deluxe_ctx;
      PCBDDCSubSchurs     sub_schurs = pcbddc->sub_schurs;
      Mat                 T;
      PetscScalar         *W,lwork,*Bwork;
      const PetscInt      *idxs = NULL;
      PetscInt            cum,mss,*nnz;
      PetscBLASInt        *pivots,B_lwork,B_N,B_ierr;

      if (!pcbddc->deluxe_singlemat) SETERRQ(comm,PETSC_ERR_USER,"Cannot compute B_Ddelta! rerun with -pc_bddc_deluxe_singlemat");
      mss  = 0;
      ierr = PetscCalloc1(pcis->n_B,&nnz);CHKERRQ(ierr);
      if (sub_schurs->is_Ej_all) {
        ierr = ISGetIndices(sub_schurs->is_Ej_all,&idxs);CHKERRQ(ierr);
        for (i=0,cum=0;i<sub_schurs->n_subs;i++) {
          PetscInt subset_size;

          ierr = ISGetLocalSize(sub_schurs->is_subs[i],&subset_size);CHKERRQ(ierr);
          for (j=0;j<subset_size;j++) nnz[idxs[j+cum]] = subset_size;
          mss  = PetscMax(mss,subset_size);
          cum += subset_size;
        }
      }
      ierr = MatCreate(PETSC_COMM_SELF,&T);CHKERRQ(ierr);
      ierr = MatSetSizes(T,pcis->n_B,pcis->n_B,pcis->n_B,pcis->n_B);CHKERRQ(ierr);
      ierr = MatSetType(T,MATSEQAIJ);CHKERRQ(ierr);
      ierr = MatSeqAIJSetPreallocation(T,0,nnz);CHKERRQ(ierr);
      ierr = PetscFree(nnz);CHKERRQ(ierr);

      /* workspace allocation */
      B_lwork = 0;
      if (mss) {
        PetscScalar dummy = 1;

        B_lwork = -1;
        ierr = PetscBLASIntCast(mss,&B_N);CHKERRQ(ierr);
        ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
        PetscStackCallBLAS("LAPACKgetri",LAPACKgetri_(&B_N,&dummy,&B_N,&B_N,&lwork,&B_lwork,&B_ierr));
        ierr = PetscFPTrapPop();CHKERRQ(ierr);
        if (B_ierr) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in query to GETRI Lapack routine %d",(int)B_ierr);
        ierr = PetscBLASIntCast((PetscInt)PetscRealPart(lwork),&B_lwork);CHKERRQ(ierr);
      }
      ierr = PetscMalloc3(mss*mss,&W,mss,&pivots,B_lwork,&Bwork);CHKERRQ(ierr);

      for (i=0,cum=0;i<sub_schurs->n_subs;i++) {
        const PetscScalar *M;
        PetscInt          subset_size;

        ierr = ISGetLocalSize(sub_schurs->is_subs[i],&subset_size);CHKERRQ(ierr);
        ierr = PetscBLASIntCast(subset_size,&B_N);CHKERRQ(ierr);
        ierr = MatDenseGetArrayRead(deluxe_ctx->seq_mat[i],&M);CHKERRQ(ierr);
        ierr = PetscArraycpy(W,M,subset_size*subset_size);CHKERRQ(ierr);
        ierr = MatDenseRestoreArrayRead(deluxe_ctx->seq_mat[i],&M);CHKERRQ(ierr);
        ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
        PetscStackCallBLAS("LAPACKgetrf",LAPACKgetrf_(&B_N,&B_N,W,&B_N,pivots,&B_ierr));
        if (B_ierr) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in GETRF Lapack routine %d",(int)B_ierr);
        PetscStackCallBLAS("LAPACKgetri",LAPACKgetri_(&B_N,W,&B_N,pivots,Bwork,&B_lwork,&B_ierr));
        if (B_ierr) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in GETRI Lapack routine %d",(int)B_ierr);
        ierr = PetscFPTrapPop();CHKERRQ(ierr);
        /* silent static analyzer */
        if (!idxs) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"IDXS not present");
        ierr = MatSetValues(T,subset_size,idxs+cum,subset_size,idxs+cum,W,INSERT_VALUES);CHKERRQ(ierr);
        cum += subset_size;
      }
      ierr = MatAssemblyBegin(T,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      ierr = MatAssemblyEnd(T,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      ierr = MatMatTransposeMult(T,fetidpmat_ctx->B_delta,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&BD1);CHKERRQ(ierr);
      ierr = MatMatMult(fetidpmat_ctx->B_delta,BD1,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&BD2);CHKERRQ(ierr);
      ierr = MatDestroy(&T);CHKERRQ(ierr);
      ierr = PetscFree3(W,pivots,Bwork);CHKERRQ(ierr);
      if (sub_schurs->is_Ej_all) {
        ierr = ISRestoreIndices(sub_schurs->is_Ej_all,&idxs);CHKERRQ(ierr);
      }
    }
  }
  ierr = PetscFree(scaling_factors);CHKERRQ(ierr);
  ierr = PetscFree(cols_B_delta);CHKERRQ(ierr);

  /* Layout of multipliers */
  ierr = PetscLayoutCreate(comm,&llay);CHKERRQ(ierr);
  ierr = PetscLayoutSetBlockSize(llay,1);CHKERRQ(ierr);
  ierr = PetscLayoutSetSize(llay,fetidpmat_ctx->n_lambda);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(llay);CHKERRQ(ierr);
  ierr = PetscLayoutGetLocalSize(llay,&fetidpmat_ctx->n);CHKERRQ(ierr);

  /* Local work vector of multipliers */
  ierr = VecCreate(PETSC_COMM_SELF,&fetidpmat_ctx->lambda_local);CHKERRQ(ierr);
  ierr = VecSetSizes(fetidpmat_ctx->lambda_local,n_local_lambda,n_local_lambda);CHKERRQ(ierr);
  ierr = VecSetType(fetidpmat_ctx->lambda_local,VECSEQ);CHKERRQ(ierr);

  if (BD2) {
    ISLocalToGlobalMapping l2g;
    Mat                    T,TA,*pT;
    IS                     is;
    PetscInt               nl,N;
    BDdelta_DN             ctx;

    ierr = PetscLayoutGetLocalSize(llay,&nl);CHKERRQ(ierr);
    ierr = PetscLayoutGetSize(llay,&N);CHKERRQ(ierr);
    ierr = MatCreate(comm,&T);CHKERRQ(ierr);
    ierr = MatSetSizes(T,nl,nl,N,N);CHKERRQ(ierr);
    ierr = MatSetType(T,MATIS);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingCreate(comm,1,n_local_lambda,l2g_indices,PETSC_COPY_VALUES,&l2g);CHKERRQ(ierr);
    ierr = MatSetLocalToGlobalMapping(T,l2g,l2g);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingDestroy(&l2g);CHKERRQ(ierr);
    ierr = MatISSetLocalMat(T,BD2);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(T,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(T,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatDestroy(&BD2);CHKERRQ(ierr);
    ierr = MatConvert(T,MATAIJ,MAT_INITIAL_MATRIX,&TA);CHKERRQ(ierr);
    ierr = MatDestroy(&T);CHKERRQ(ierr);
    ierr = ISCreateGeneral(comm,n_local_lambda,l2g_indices,PETSC_USE_POINTER,&is);CHKERRQ(ierr);
    ierr = MatCreateSubMatrices(TA,1,&is,&is,MAT_INITIAL_MATRIX,&pT);CHKERRQ(ierr);
    ierr = MatDestroy(&TA);CHKERRQ(ierr);
    ierr = ISDestroy(&is);CHKERRQ(ierr);
    BD2  = pT[0];
    ierr = PetscFree(pT);CHKERRQ(ierr);

    /* B_Ddelta for non-redundant multipliers with deluxe scaling */
    ierr = PetscNew(&ctx);CHKERRQ(ierr);
    ierr = MatSetType(fetidpmat_ctx->B_Ddelta,MATSHELL);CHKERRQ(ierr);
    ierr = MatShellSetContext(fetidpmat_ctx->B_Ddelta,(void *)ctx);CHKERRQ(ierr);
    ierr = MatShellSetOperation(fetidpmat_ctx->B_Ddelta,MATOP_MULT,(void (*)(void))MatMult_BDdelta_deluxe_nonred);CHKERRQ(ierr);
    ierr = MatShellSetOperation(fetidpmat_ctx->B_Ddelta,MATOP_MULT_TRANSPOSE,(void (*)(void))MatMultTranspose_BDdelta_deluxe_nonred);CHKERRQ(ierr);
    ierr = MatShellSetOperation(fetidpmat_ctx->B_Ddelta,MATOP_DESTROY,(void (*)(void))MatDestroy_BDdelta_deluxe_nonred);CHKERRQ(ierr);
    ierr = MatSetUp(fetidpmat_ctx->B_Ddelta);CHKERRQ(ierr);

    ierr = PetscObjectReference((PetscObject)BD1);CHKERRQ(ierr);
    ctx->BD = BD1;
    ierr = KSPCreate(PETSC_COMM_SELF,&ctx->kBD);CHKERRQ(ierr);
    ierr = KSPSetOperators(ctx->kBD,BD2,BD2);CHKERRQ(ierr);
    ierr = VecDuplicate(fetidpmat_ctx->lambda_local,&ctx->work);CHKERRQ(ierr);
    fetidpmat_ctx->deluxe_nonred = PETSC_TRUE;
  }
  ierr = MatDestroy(&BD1);CHKERRQ(ierr);
  ierr = MatDestroy(&BD2);CHKERRQ(ierr);

  /* fetidpmat sizes */
  fetidpmat_ctx->n += nPgl;
  fetidpmat_ctx->N  = fetidpmat_ctx->n_lambda+nPg;

  /* Global vector for FETI-DP linear system */
  ierr = VecCreate(comm,&fetidp_global);CHKERRQ(ierr);
  ierr = VecSetSizes(fetidp_global,fetidpmat_ctx->n,fetidpmat_ctx->N);CHKERRQ(ierr);
  ierr = VecSetType(fetidp_global,VECMPI);CHKERRQ(ierr);
  ierr = VecSetUp(fetidp_global);CHKERRQ(ierr);

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

    ierr = PetscMalloc1(nPl,&l2g_indices_p);CHKERRQ(ierr);
    ierr = VecGetLayout(fetidp_global,&alay);CHKERRQ(ierr);
    ierr = PetscLayoutGetRanges(alay,&aranges);CHKERRQ(ierr);
    ierr = PetscLayoutGetRanges(play,&pranges);CHKERRQ(ierr);
    ierr = PetscLayoutGetRanges(llay,&lranges);CHKERRQ(ierr);

    ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)fetidp_global),&rank);CHKERRQ(ierr);
    ierr = ISCreateStride(PetscObjectComm((PetscObject)fetidp_global),pranges[rank+1]-pranges[rank],aranges[rank],1,&fetidpmat_ctx->pressure);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)fetidpmat_ctx->pressure,"F_P");CHKERRQ(ierr);
    ierr = ISCreateStride(PetscObjectComm((PetscObject)fetidp_global),lranges[rank+1]-lranges[rank],aranges[rank]+pranges[rank+1]-pranges[rank],1,&fetidpmat_ctx->lagrange);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)fetidpmat_ctx->lagrange,"F_L");CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingGetIndices(l2gmap_p,&idxs);CHKERRQ(ierr);
    /* shift local to global indices for pressure */
    for (i=0;i<nPl;i++) {
      PetscMPIInt owner;

      ierr = PetscLayoutFindOwner(play,idxs[i],&owner);CHKERRQ(ierr);
      l2g_indices_p[i] = idxs[i]-pranges[owner]+aranges[owner];
    }
    ierr = ISLocalToGlobalMappingRestoreIndices(l2gmap_p,&idxs);CHKERRQ(ierr);
    ierr = ISCreateGeneral(comm,nPl,l2g_indices_p,PETSC_OWN_POINTER,&IS_l2g_p);CHKERRQ(ierr);

    /* local to global scatter for pressure */
    ierr = VecScatterCreate(fetidpmat_ctx->vP,NULL,fetidp_global,IS_l2g_p,&fetidpmat_ctx->l2g_p);CHKERRQ(ierr);
    ierr = ISDestroy(&IS_l2g_p);CHKERRQ(ierr);

    /* scatter for lagrange multipliers only */
    ierr = VecCreate(comm,&v);CHKERRQ(ierr);
    ierr = VecSetType(v,VECSTANDARD);CHKERRQ(ierr);
    ierr = VecSetLayout(v,llay);CHKERRQ(ierr);
    ierr = VecSetUp(v);CHKERRQ(ierr);
    ierr = ISCreateGeneral(comm,n_local_lambda,l2g_indices,PETSC_COPY_VALUES,&ais);CHKERRQ(ierr);
    ierr = VecScatterCreate(fetidpmat_ctx->lambda_local,NULL,v,ais,&fetidpmat_ctx->l2g_lambda_only);CHKERRQ(ierr);
    ierr = ISDestroy(&ais);CHKERRQ(ierr);
    ierr = VecDestroy(&v);CHKERRQ(ierr);

    /* shift local to global indices for multipliers */
    for (i=0;i<n_local_lambda;i++) {
      PetscInt    ps;
      PetscMPIInt owner;

      ierr = PetscLayoutFindOwner(llay,l2g_indices[i],&owner);CHKERRQ(ierr);
      ps = pranges[owner+1]-pranges[owner];
      l2g_indices[i] = l2g_indices[i]-lranges[owner]+aranges[owner]+ps;
    }

    /* scatter from alldofs to pressures global fetidp vector */
    ierr = PetscLayoutGetRange(alay,&rst,NULL);CHKERRQ(ierr);
    ierr = ISCreateStride(comm,nPgl,rst,1,&ais);CHKERRQ(ierr);
    ierr = VecScatterCreate(pcis->vec1_global,pP,fetidp_global,ais,&fetidpmat_ctx->g2g_p);CHKERRQ(ierr);
    ierr = ISDestroy(&ais);CHKERRQ(ierr);
  }
  ierr = PetscLayoutDestroy(&llay);CHKERRQ(ierr);
  ierr = PetscLayoutDestroy(&play);CHKERRQ(ierr);
  ierr = ISCreateGeneral(comm,n_local_lambda,l2g_indices,PETSC_OWN_POINTER,&IS_l2g_lambda);CHKERRQ(ierr);

  /* scatter from local to global multipliers */
  ierr = VecScatterCreate(fetidpmat_ctx->lambda_local,NULL,fetidp_global,IS_l2g_lambda,&fetidpmat_ctx->l2g_lambda);CHKERRQ(ierr);
  ierr = ISDestroy(&IS_l2g_lambda);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingDestroy(&l2gmap_p);CHKERRQ(ierr);
  ierr = VecDestroy(&fetidp_global);CHKERRQ(ierr);

  /* Create some work vectors needed by fetidp */
  ierr = VecDuplicate(pcis->vec1_B,&fetidpmat_ctx->temp_solution_B);CHKERRQ(ierr);
  ierr = VecDuplicate(pcis->vec1_D,&fetidpmat_ctx->temp_solution_D);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


PetscErrorCode PCBDDCSetupFETIDPPCContext(Mat fetimat, FETIDPPC_ctx fetidppc_ctx)
{
  FETIDPMat_ctx  mat_ctx;
  PC_BDDC        *pcbddc = (PC_BDDC*)fetidppc_ctx->pc->data;
  PC_IS          *pcis = (PC_IS*)fetidppc_ctx->pc->data;
  PetscBool      lumped = PETSC_FALSE;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(fetimat,(void**)&mat_ctx);CHKERRQ(ierr);
  /* get references from objects created when setting up feti mat context */
  ierr = PetscObjectReference((PetscObject)mat_ctx->lambda_local);CHKERRQ(ierr);
  fetidppc_ctx->lambda_local = mat_ctx->lambda_local;
  ierr = PetscObjectReference((PetscObject)mat_ctx->B_Ddelta);CHKERRQ(ierr);
  fetidppc_ctx->B_Ddelta = mat_ctx->B_Ddelta;
  if (mat_ctx->deluxe_nonred) {
    PC               pc,mpc;
    BDdelta_DN       ctx;
    MatSolverType    solver;
    const char       *prefix;

    ierr = MatShellGetContext(mat_ctx->B_Ddelta,&ctx);CHKERRQ(ierr);
    ierr = KSPSetType(ctx->kBD,KSPPREONLY);CHKERRQ(ierr);
    ierr = KSPGetPC(ctx->kBD,&mpc);CHKERRQ(ierr);
    ierr = KSPGetPC(pcbddc->ksp_D,&pc);CHKERRQ(ierr);
    ierr = PCSetType(mpc,PCLU);CHKERRQ(ierr);
    ierr = PCFactorGetMatSolverType(pc,(MatSolverType*)&solver);CHKERRQ(ierr);
    if (solver) {
      ierr = PCFactorSetMatSolverType(mpc,solver);CHKERRQ(ierr);
    }
    ierr = MatGetOptionsPrefix(fetimat,&prefix);CHKERRQ(ierr);
    ierr = KSPSetOptionsPrefix(ctx->kBD,prefix);CHKERRQ(ierr);
    ierr = KSPAppendOptionsPrefix(ctx->kBD,"bddelta_");CHKERRQ(ierr);
    ierr = KSPSetFromOptions(ctx->kBD);CHKERRQ(ierr);
  }

  if (mat_ctx->l2g_lambda_only) {
    ierr = PetscObjectReference((PetscObject)mat_ctx->l2g_lambda_only);CHKERRQ(ierr);
    fetidppc_ctx->l2g_lambda = mat_ctx->l2g_lambda_only;
  } else {
    ierr = PetscObjectReference((PetscObject)mat_ctx->l2g_lambda);CHKERRQ(ierr);
    fetidppc_ctx->l2g_lambda = mat_ctx->l2g_lambda;
  }
  /* Dirichlet preconditioner */
  ierr = PetscOptionsGetBool(NULL,((PetscObject)fetimat)->prefix,"-pc_lumped",&lumped,NULL);CHKERRQ(ierr);
  if (!lumped) {
    IS        iV;
    PetscBool discrete_harmonic = PETSC_FALSE;

    ierr = PetscObjectQuery((PetscObject)fetidppc_ctx->pc,"__KSPFETIDP_iV",(PetscObject*)&iV);CHKERRQ(ierr);
    if (iV) {
      ierr = PetscOptionsGetBool(NULL,((PetscObject)fetimat)->prefix,"-pc_discrete_harmonic",&discrete_harmonic,NULL);CHKERRQ(ierr);
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
        ierr = PetscObjectQuery((PetscObject)sub_schurs->A,"__KSPFETIDP_iP",(PetscObject*)&iP);CHKERRQ(ierr);
        if (iP) reuse = PETSC_TRUE;
      }
      if (!reuse) {
        IS       aB;
        PetscInt nb;
        ierr = ISGetLocalSize(pcis->is_B_local,&nb);CHKERRQ(ierr);
        ierr = ISCreateStride(PetscObjectComm((PetscObject)pcis->A_II),nb,0,1,&aB);CHKERRQ(ierr);
        ierr = MatCreateSubMatrix(pcis->A_II,iV,iV,MAT_INITIAL_MATRIX,&A_II);CHKERRQ(ierr);
        ierr = MatCreateSubMatrix(pcis->A_IB,iV,aB,MAT_INITIAL_MATRIX,&A_IB);CHKERRQ(ierr);
        ierr = MatCreateSubMatrix(pcis->A_BI,aB,iV,MAT_INITIAL_MATRIX,&A_BI);CHKERRQ(ierr);
        ierr = ISDestroy(&aB);CHKERRQ(ierr);
      } else {
        ierr = MatCreateSubMatrix(sub_schurs->A,pcis->is_I_local,pcis->is_B_local,MAT_INITIAL_MATRIX,&A_IB);CHKERRQ(ierr);
        ierr = MatCreateSubMatrix(sub_schurs->A,pcis->is_B_local,pcis->is_I_local,MAT_INITIAL_MATRIX,&A_BI);CHKERRQ(ierr);
        ierr = PetscObjectReference((PetscObject)pcis->A_II);CHKERRQ(ierr);
        A_II = pcis->A_II;
      }
      ierr = MatCreateSchurComplement(A_II,A_II,A_IB,A_BI,pcis->A_BB,&fetidppc_ctx->S_j);CHKERRQ(ierr);

      /* propagate settings of solver */
      ierr = MatSchurComplementGetKSP(fetidppc_ctx->S_j,&sksp);CHKERRQ(ierr);
      ierr = KSPGetType(pcis->ksp_D,&ksptype);CHKERRQ(ierr);
      ierr = KSPSetType(sksp,ksptype);CHKERRQ(ierr);
      ierr = KSPGetPC(pcis->ksp_D,&pc);CHKERRQ(ierr);
      ierr = PetscObjectTypeCompare((PetscObject)pc,PCSHELL,&isshell);CHKERRQ(ierr);
      if (!isshell) {
        MatSolverType    solver;
        PCType           pctype;

        ierr = PCGetType(pc,&pctype);CHKERRQ(ierr);
        ierr = PCFactorGetMatSolverType(pc,(MatSolverType*)&solver);CHKERRQ(ierr);
        ierr = KSPGetPC(sksp,&pc);CHKERRQ(ierr);
        ierr = PCSetType(pc,pctype);CHKERRQ(ierr);
        if (solver) {
          ierr = PCFactorSetMatSolverType(pc,solver);CHKERRQ(ierr);
        }
      } else {
        ierr = KSPGetPC(sksp,&pc);CHKERRQ(ierr);
        ierr = PCSetType(pc,PCLU);CHKERRQ(ierr);
      }
      ierr = MatDestroy(&A_II);CHKERRQ(ierr);
      ierr = MatDestroy(&A_IB);CHKERRQ(ierr);
      ierr = MatDestroy(&A_BI);CHKERRQ(ierr);
      ierr = MatGetOptionsPrefix(fetimat,&prefix);CHKERRQ(ierr);
      ierr = KSPSetOptionsPrefix(sksp,prefix);CHKERRQ(ierr);
      ierr = KSPAppendOptionsPrefix(sksp,"harmonic_");CHKERRQ(ierr);
      ierr = KSPSetFromOptions(sksp);CHKERRQ(ierr);
      if (reuse) {
        ierr = KSPSetPC(sksp,sub_schurs->reuse_solver->interior_solver);CHKERRQ(ierr);
        ierr = PetscObjectIncrementTabLevel((PetscObject)sub_schurs->reuse_solver->interior_solver,(PetscObject)sksp,0);CHKERRQ(ierr);
      }
    } else { /* default Dirichlet preconditioner is pde-harmonic */
      ierr = MatCreateSchurComplement(pcis->A_II,pcis->A_II,pcis->A_IB,pcis->A_BI,pcis->A_BB,&fetidppc_ctx->S_j);CHKERRQ(ierr);
      ierr = MatSchurComplementSetKSP(fetidppc_ctx->S_j,pcis->ksp_D);CHKERRQ(ierr);
    }
  } else {
    ierr = PetscObjectReference((PetscObject)pcis->A_BB);CHKERRQ(ierr);
    fetidppc_ctx->S_j = pcis->A_BB;
  }
  /* saddle-point */
  if (mat_ctx->xPg) {
    ierr = PetscObjectReference((PetscObject)mat_ctx->xPg);CHKERRQ(ierr);
    fetidppc_ctx->xPg = mat_ctx->xPg;
    ierr = PetscObjectReference((PetscObject)mat_ctx->yPg);CHKERRQ(ierr);
    fetidppc_ctx->yPg = mat_ctx->yPg;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode FETIDPMatMult_Kernel(Mat fetimat, Vec x, Vec y, PetscBool trans)
{
  FETIDPMat_ctx  mat_ctx;
  PC_BDDC        *pcbddc;
  PC_IS          *pcis;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(fetimat,(void**)&mat_ctx);CHKERRQ(ierr);
  pcis = (PC_IS*)mat_ctx->pc->data;
  pcbddc = (PC_BDDC*)mat_ctx->pc->data;
  /* Application of B_delta^T */
  ierr = VecSet(pcis->vec1_B,0.);CHKERRQ(ierr);
  ierr = VecScatterBegin(mat_ctx->l2g_lambda,x,mat_ctx->lambda_local,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd(mat_ctx->l2g_lambda,x,mat_ctx->lambda_local,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = MatMultTranspose(mat_ctx->B_delta,mat_ctx->lambda_local,pcis->vec1_B);CHKERRQ(ierr);

  /* Add contribution from saddle point */
  if (mat_ctx->l2g_p) {
    ierr = VecScatterBegin(mat_ctx->l2g_p,x,mat_ctx->vP,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterEnd(mat_ctx->l2g_p,x,mat_ctx->vP,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    if (pcbddc->switch_static) {
      if (trans) {
        ierr = MatMultTranspose(mat_ctx->B_BI,mat_ctx->vP,pcis->vec1_D);CHKERRQ(ierr);
      } else {
        ierr = MatMult(mat_ctx->Bt_BI,mat_ctx->vP,pcis->vec1_D);CHKERRQ(ierr);
      }
    }
    if (trans) {
      ierr = MatMultTransposeAdd(mat_ctx->B_BB,mat_ctx->vP,pcis->vec1_B,pcis->vec1_B);CHKERRQ(ierr);
    } else {
      ierr = MatMultAdd(mat_ctx->Bt_BB,mat_ctx->vP,pcis->vec1_B,pcis->vec1_B);CHKERRQ(ierr);
    }
  } else {
    if (pcbddc->switch_static) {
      ierr = VecSet(pcis->vec1_D,0.0);CHKERRQ(ierr);
    }
  }
  /* Application of \widetilde{S}^-1 */
  ierr = PetscArrayzero(pcbddc->benign_p0,pcbddc->benign_n);CHKERRQ(ierr);
  ierr = PCBDDCApplyInterfacePreconditioner(mat_ctx->pc,trans);CHKERRQ(ierr);
  ierr = PetscArrayzero(pcbddc->benign_p0,pcbddc->benign_n);CHKERRQ(ierr);
  ierr = VecSet(y,0.0);CHKERRQ(ierr);
  /* Application of B_delta */
  ierr = MatMult(mat_ctx->B_delta,pcis->vec1_B,mat_ctx->lambda_local);CHKERRQ(ierr);
  /* Contribution from boundary pressures */
  if (mat_ctx->C) {
    const PetscScalar *lx;
    PetscScalar       *ly;

    /* pressure ordered first in the local part of x and y */
    ierr = VecGetArrayRead(x,&lx);CHKERRQ(ierr);
    ierr = VecGetArray(y,&ly);CHKERRQ(ierr);
    ierr = VecPlaceArray(mat_ctx->xPg,lx);CHKERRQ(ierr);
    ierr = VecPlaceArray(mat_ctx->yPg,ly);CHKERRQ(ierr);
    if (trans) {
      ierr = MatMultTranspose(mat_ctx->C,mat_ctx->xPg,mat_ctx->yPg);CHKERRQ(ierr);
    } else {
      ierr = MatMult(mat_ctx->C,mat_ctx->xPg,mat_ctx->yPg);CHKERRQ(ierr);
    }
    ierr = VecResetArray(mat_ctx->xPg);CHKERRQ(ierr);
    ierr = VecResetArray(mat_ctx->yPg);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(x,&lx);CHKERRQ(ierr);
    ierr = VecRestoreArray(y,&ly);CHKERRQ(ierr);
  }
  /* Add contribution from saddle point */
  if (mat_ctx->l2g_p) {
    if (trans) {
      ierr = MatMultTranspose(mat_ctx->Bt_BB,pcis->vec1_B,mat_ctx->vP);CHKERRQ(ierr);
    } else {
      ierr = MatMult(mat_ctx->B_BB,pcis->vec1_B,mat_ctx->vP);CHKERRQ(ierr);
    }
    if (pcbddc->switch_static) {
      if (trans) {
        ierr = MatMultTransposeAdd(mat_ctx->Bt_BI,pcis->vec1_D,mat_ctx->vP,mat_ctx->vP);CHKERRQ(ierr);
      } else {
        ierr = MatMultAdd(mat_ctx->B_BI,pcis->vec1_D,mat_ctx->vP,mat_ctx->vP);CHKERRQ(ierr);
      }
    }
    ierr = VecScatterBegin(mat_ctx->l2g_p,mat_ctx->vP,y,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(mat_ctx->l2g_p,mat_ctx->vP,y,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  }
  ierr = VecScatterBegin(mat_ctx->l2g_lambda,mat_ctx->lambda_local,y,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(mat_ctx->l2g_lambda,mat_ctx->lambda_local,y,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode FETIDPMatMult(Mat fetimat, Vec x, Vec y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = FETIDPMatMult_Kernel(fetimat,x,y,PETSC_FALSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode FETIDPMatMultTranspose(Mat fetimat, Vec x, Vec y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = FETIDPMatMult_Kernel(fetimat,x,y,PETSC_TRUE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode FETIDPPCApply_Kernel(PC fetipc, Vec x, Vec y, PetscBool trans)
{
  FETIDPPC_ctx   pc_ctx;
  PC_IS          *pcis;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PCShellGetContext(fetipc,(void**)&pc_ctx);CHKERRQ(ierr);
  pcis = (PC_IS*)pc_ctx->pc->data;
  /* Application of B_Ddelta^T */
  ierr = VecScatterBegin(pc_ctx->l2g_lambda,x,pc_ctx->lambda_local,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd(pc_ctx->l2g_lambda,x,pc_ctx->lambda_local,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecSet(pcis->vec2_B,0.0);CHKERRQ(ierr);
  ierr = MatMultTranspose(pc_ctx->B_Ddelta,pc_ctx->lambda_local,pcis->vec2_B);CHKERRQ(ierr);
  /* Application of local Schur complement */
  if (trans) {
    ierr = MatMultTranspose(pc_ctx->S_j,pcis->vec2_B,pcis->vec1_B);CHKERRQ(ierr);
  } else {
    ierr = MatMult(pc_ctx->S_j,pcis->vec2_B,pcis->vec1_B);CHKERRQ(ierr);
  }
  /* Application of B_Ddelta */
  ierr = MatMult(pc_ctx->B_Ddelta,pcis->vec1_B,pc_ctx->lambda_local);CHKERRQ(ierr);
  ierr = VecSet(y,0.0);CHKERRQ(ierr);
  ierr = VecScatterBegin(pc_ctx->l2g_lambda,pc_ctx->lambda_local,y,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(pc_ctx->l2g_lambda,pc_ctx->lambda_local,y,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode FETIDPPCApply(PC pc, Vec x, Vec y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = FETIDPPCApply_Kernel(pc,x,y,PETSC_FALSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode FETIDPPCApplyTranspose(PC pc, Vec x, Vec y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = FETIDPPCApply_Kernel(pc,x,y,PETSC_TRUE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode FETIDPPCView(PC pc, PetscViewer viewer)
{
  FETIDPPC_ctx      pc_ctx;
  PetscBool         iascii;
  PetscViewer       sviewer;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    PetscMPIInt rank;
    PetscBool   isschur,isshell;

    ierr = PCShellGetContext(pc,(void**)&pc_ctx);CHKERRQ(ierr);
    ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)pc),&rank);CHKERRQ(ierr);
    ierr = PetscObjectTypeCompare((PetscObject)pc_ctx->S_j,MATSCHURCOMPLEMENT,&isschur);CHKERRQ(ierr);
    if (isschur) {
      ierr = PetscViewerASCIIPrintf(viewer,"  Dirichlet preconditioner (just from rank 0)\n");CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIIPrintf(viewer,"  Lumped preconditioner (just from rank 0)\n");CHKERRQ(ierr);
    }
    ierr = PetscViewerGetSubViewer(viewer,PetscObjectComm((PetscObject)pc_ctx->S_j),&sviewer);CHKERRQ(ierr);
    if (!rank) {
      ierr = PetscViewerPushFormat(sviewer,PETSC_VIEWER_ASCII_INFO);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPushTab(sviewer);CHKERRQ(ierr);
      ierr = MatView(pc_ctx->S_j,sviewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopTab(sviewer);CHKERRQ(ierr);
      ierr = PetscViewerPopFormat(sviewer);CHKERRQ(ierr);
    }
    ierr = PetscViewerRestoreSubViewer(viewer,PetscObjectComm((PetscObject)pc_ctx->S_j),&sviewer);CHKERRQ(ierr);
    ierr = PetscObjectTypeCompare((PetscObject)pc_ctx->B_Ddelta,MATSHELL,&isshell);CHKERRQ(ierr);
    if (isshell) {
      BDdelta_DN ctx;
      ierr = PetscViewerASCIIPrintf(viewer,"  FETI-DP BDdelta: DB^t * (B D^-1 B^t)^-1 for deluxe scaling (just from rank 0)\n");CHKERRQ(ierr);
      ierr = MatShellGetContext(pc_ctx->B_Ddelta,&ctx);CHKERRQ(ierr);
      ierr = PetscViewerGetSubViewer(viewer,PetscObjectComm((PetscObject)pc_ctx->S_j),&sviewer);CHKERRQ(ierr);
      if (!rank) {
        PetscInt tl;

        ierr = PetscViewerASCIIGetTab(sviewer,&tl);CHKERRQ(ierr);
        ierr = PetscObjectSetTabLevel((PetscObject)ctx->kBD,tl);CHKERRQ(ierr);
        ierr = KSPView(ctx->kBD,sviewer);CHKERRQ(ierr);
        ierr = PetscViewerPushFormat(sviewer,PETSC_VIEWER_ASCII_INFO);CHKERRQ(ierr);
        ierr = MatView(ctx->BD,sviewer);CHKERRQ(ierr);
        ierr = PetscViewerPopFormat(sviewer);CHKERRQ(ierr);
      }
      ierr = PetscViewerRestoreSubViewer(viewer,PetscObjectComm((PetscObject)pc_ctx->S_j),&sviewer);CHKERRQ(ierr);
    }
    ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
