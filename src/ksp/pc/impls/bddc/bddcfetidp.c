#include <../src/ksp/pc/impls/bddc/bddc.h>
#include <../src/ksp/pc/impls/bddc/bddcprivate.h>

#undef __FUNCT__
#define __FUNCT__ "PCBDDCCreateFETIDPMatContext"
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

#undef __FUNCT__
#define __FUNCT__ "PCBDDCCreateFETIDPPCContext"
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

#undef __FUNCT__
#define __FUNCT__ "PCBDDCDestroyFETIDPMat"
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
  ierr = VecDestroy(&mat_ctx->vP);CHKERRQ(ierr);
  ierr = VecDestroy(&mat_ctx->xPg);CHKERRQ(ierr);
  ierr = VecDestroy(&mat_ctx->yPg);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&mat_ctx->l2g_lambda);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&mat_ctx->l2g_p);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&mat_ctx->g2g_p);CHKERRQ(ierr);
  ierr = PCDestroy(&mat_ctx->pc);CHKERRQ(ierr); /* decrease PCBDDC reference count */
  ierr = PetscFree(mat_ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCBDDCDestroyFETIDPPC"
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
  ierr = KSPDestroy(&pc_ctx->kP);CHKERRQ(ierr);
  ierr = VecDestroy(&pc_ctx->xPg);CHKERRQ(ierr);
  ierr = VecDestroy(&pc_ctx->yPg);CHKERRQ(ierr);
  ierr = PetscFree(pc_ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCBDDCSetupFETIDPMatContext"
PetscErrorCode PCBDDCSetupFETIDPMatContext(FETIDPMat_ctx fetidpmat_ctx )
{
  PetscErrorCode ierr;
  PC_IS          *pcis=(PC_IS*)fetidpmat_ctx->pc->data;
  PC_BDDC        *pcbddc=(PC_BDDC*)fetidpmat_ctx->pc->data;
  PCBDDCGraph    mat_graph=pcbddc->mat_graph;
  Mat_IS         *matis  = (Mat_IS*)fetidpmat_ctx->pc->pmat->data;
  MPI_Comm       comm;
  Mat            ScalingMat;
  Vec            fetidp_global;
  IS             IS_l2g_lambda;
  IS             subset,subset_mult,subset_n;
  PetscBool      skip_node,fully_redundant;
  PetscInt       i,j,k,s,n_boundary_dofs,n_global_lambda,n_vertices,partial_sum;
  PetscInt       cum,n_local_lambda,n_lambda_for_dof,dual_size,n_neg_values,n_pos_values;
  PetscMPIInt    rank,size,buf_size,neigh;
  PetscScalar    scalar_value;
  PetscInt       *vertex_indices;
  PetscInt       *dual_dofs_boundary_indices,*aux_local_numbering_1;
  const PetscInt *aux_global_numbering;
  PetscInt       *aux_sums,*cols_B_delta,*l2g_indices;
  PetscScalar    *array,*scaling_factors,*vals_B_delta;
  PetscInt       *aux_local_numbering_2;
  PetscLayout    llay;
  /* For communication of scaling factors */
  PetscInt       *ptrs_buffer,neigh_position;
  PetscScalar    **all_factors,*send_buffer,*recv_buffer;
  MPI_Request    *send_reqs,*recv_reqs;
  /* tests */
  PetscBool      test_fetidp;
  PetscViewer    viewer;
  /* saddlepoint */
  Mat                    oA;
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
  oA       = NULL;
  ierr = PetscObjectQuery((PetscObject)fetidpmat_ctx->pc,"__KSPFETIDP_A",(PetscObject*)&oA);CHKERRQ(ierr);
  if (oA) { /* oA is the original matrix */
    /* pressures in parallel layout of oA */
    ierr = PetscObjectQuery((PetscObject)fetidpmat_ctx->pc,"__KSPFETIDP_pP",(PetscObject*)&pP);CHKERRQ(ierr);
    if (!pP) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"pP not present");

    /* subdomain pressures in global numbering */
    ierr = PetscObjectQuery((PetscObject)fetidpmat_ctx->pc,"__KSPFETIDP_gP",(PetscObject*)&gP);CHKERRQ(ierr);
    if (!gP) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"gP not present");
    ierr = ISGetLocalSize(gP,&nPl);CHKERRQ(ierr);
    ierr = VecCreate(PETSC_COMM_SELF,&fetidpmat_ctx->vP);CHKERRQ(ierr);
    ierr = VecSetSizes(fetidpmat_ctx->vP,nPl,nPl);CHKERRQ(ierr);
    ierr = VecSetType(fetidpmat_ctx->vP,VECSTANDARD);CHKERRQ(ierr);
    ierr = VecSetUp(fetidpmat_ctx->vP);CHKERRQ(ierr);

    /* interface pressure matrix */
    ierr = PetscObjectQuery((PetscObject)fetidpmat_ctx->pc,"__KSPFETIDP_C",(PetscObject*)&fetidpmat_ctx->C);CHKERRQ(ierr);
    if (!fetidpmat_ctx->C) { /* null pressure block, compute layout and global numbering for interface pressures */
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

    /* import matrices for interface pressures coupling */
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
  n_vertices = pcbddc->n_vertices;
  vertex_indices = pcbddc->local_primal_ref_node;
  dual_size = pcis->n_B-n_vertices;
  ierr = PetscMalloc1(dual_size,&dual_dofs_boundary_indices);CHKERRQ(ierr);
  ierr = PetscMalloc1(dual_size,&aux_local_numbering_1);CHKERRQ(ierr);
  ierr = PetscMalloc1(dual_size,&aux_local_numbering_2);CHKERRQ(ierr);

  ierr = VecGetArray(pcis->vec1_N,&array);CHKERRQ(ierr);
  for (i=0;i<pcis->n;i++){
    j = mat_graph->count[i]; /* RECALL: mat_graph->count[i] does not count myself */
    if ( j > 0 ) {
      n_boundary_dofs++;
    }
    skip_node = PETSC_FALSE;
    if ( s < n_vertices && vertex_indices[s]==i) { /* it works for a sorted set of vertices */
      skip_node = PETSC_TRUE;
      s++;
    }
    if (j < 1) {
      skip_node = PETSC_TRUE;
    }
    if ( !skip_node ) {
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

  /* compute global ordering of lagrange multipliers and associate l2g map */
  ierr = ISCreateGeneral(comm,partial_sum,aux_local_numbering_1,PETSC_COPY_VALUES,&subset_n);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingApplyIS(pcis->mapping,subset_n,&subset);CHKERRQ(ierr);
  ierr = ISDestroy(&subset_n);CHKERRQ(ierr);
  ierr = ISCreateGeneral(comm,partial_sum,aux_local_numbering_2,PETSC_OWN_POINTER,&subset_mult);CHKERRQ(ierr);
  ierr = ISRenumber(subset,subset_mult,&fetidpmat_ctx->n_lambda,&subset_n);CHKERRQ(ierr);
  ierr = ISDestroy(&subset);CHKERRQ(ierr);

#if defined(PETSC_USE_DEBUG)
  ierr = VecSet(pcis->vec1_global,0.0);CHKERRQ(ierr);
  ierr = VecScatterBegin(matis->rctx,pcis->vec1_N,pcis->vec1_global,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd(matis->rctx,pcis->vec1_N,pcis->vec1_global,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecSum(pcis->vec1_global,&scalar_value);CHKERRQ(ierr);
  i = (PetscInt)PetscRealPart(scalar_value);
  if (i != fetidpmat_ctx->n_lambda) SETERRQ2(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Global number of multipliers mismatch! (%d!=%d)\n",fetidpmat_ctx->n_lambda,i);
#endif

  /* init data for scaling factors exchange */
  partial_sum = 0;
  ierr = PetscMalloc1(pcis->n_neigh,&ptrs_buffer);CHKERRQ(ierr);
  ierr = PetscMalloc1(PetscMax(pcis->n_neigh-1,0),&send_reqs);CHKERRQ(ierr);
  ierr = PetscMalloc1(PetscMax(pcis->n_neigh-1,0),&recv_reqs);CHKERRQ(ierr);
  ierr = PetscMalloc1(pcis->n,&all_factors);CHKERRQ(ierr);
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

  /* Compute B and B_delta (local actions) */
  ierr = PetscMalloc1(pcis->n_neigh,&aux_sums);CHKERRQ(ierr);
  ierr = PetscMalloc1(n_local_lambda,&l2g_indices);CHKERRQ(ierr);
  ierr = PetscMalloc1(n_local_lambda,&vals_B_delta);CHKERRQ(ierr);
  ierr = PetscMalloc1(n_local_lambda,&cols_B_delta);CHKERRQ(ierr);
  ierr = PetscMalloc1(n_local_lambda,&scaling_factors);CHKERRQ(ierr);
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
    array = all_factors[aux_local_numbering_1[i]];
    n_neg_values = 0;
    while(n_neg_values < j && mat_graph->neighbours_set[aux_local_numbering_1[i]][n_neg_values] < rank) {n_neg_values++;}
    n_pos_values = j - n_neg_values;
    if (fully_redundant) {
      for (s=0;s<n_neg_values;s++) {
        l2g_indices    [partial_sum+s]=aux_sums[s]+n_neg_values-s-1+n_global_lambda;
        cols_B_delta   [partial_sum+s]=dual_dofs_boundary_indices[i];
        vals_B_delta   [partial_sum+s]=-1.0;
        scaling_factors[partial_sum+s]=array[s];
      }
      for (s=0;s<n_pos_values;s++) {
        l2g_indices    [partial_sum+s+n_neg_values]=aux_sums[n_neg_values]+s+n_global_lambda;
        cols_B_delta   [partial_sum+s+n_neg_values]=dual_dofs_boundary_indices[i];
        vals_B_delta   [partial_sum+s+n_neg_values]=1.0;
        scaling_factors[partial_sum+s+n_neg_values]=array[s+n_neg_values];
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
      /* scaling as in Klawonn-Widlund 1999*/
      for (s=0;s<n_neg_values;s++) {
        scalar_value = 0.0;
        for (k=0;k<s+1;k++) {
          scalar_value += array[k];
        }
        scaling_factors[partial_sum+s] = -scalar_value;
      }
      for (s=0;s<n_pos_values;s++) {
        scalar_value = 0.0;
        for (k=s+n_neg_values;k<j;k++) {
          scalar_value += array[k];
        }
        scaling_factors[partial_sum+s+n_neg_values] = scalar_value;
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
  ierr = PetscFree(all_factors[0]);CHKERRQ(ierr);
  ierr = PetscFree(all_factors);CHKERRQ(ierr);

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
  ierr = MatAssemblyEnd  (fetidpmat_ctx->B_delta,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  if (fully_redundant) {
    ierr = MatCreate(PETSC_COMM_SELF,&ScalingMat);CHKERRQ(ierr);
    ierr = MatSetSizes(ScalingMat,n_local_lambda,n_local_lambda,n_local_lambda,n_local_lambda);CHKERRQ(ierr);
    ierr = MatSetType(ScalingMat,MATSEQAIJ);CHKERRQ(ierr);
    ierr = MatSeqAIJSetPreallocation(ScalingMat,1,NULL);CHKERRQ(ierr);
    for (i=0;i<n_local_lambda;i++) {
      ierr = MatSetValue(ScalingMat,i,i,scaling_factors[i],INSERT_VALUES);CHKERRQ(ierr);
    }
    ierr = MatAssemblyBegin(ScalingMat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd  (ScalingMat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatMatMult(ScalingMat,fetidpmat_ctx->B_delta,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&fetidpmat_ctx->B_Ddelta);CHKERRQ(ierr);
    ierr = MatDestroy(&ScalingMat);CHKERRQ(ierr);
  } else {
    ierr = MatCreate(PETSC_COMM_SELF,&fetidpmat_ctx->B_Ddelta);CHKERRQ(ierr);
    ierr = MatSetSizes(fetidpmat_ctx->B_Ddelta,n_local_lambda,pcis->n_B,n_local_lambda,pcis->n_B);CHKERRQ(ierr);
    ierr = MatSetType(fetidpmat_ctx->B_Ddelta,MATSEQAIJ);CHKERRQ(ierr);
    ierr = MatSeqAIJSetPreallocation(fetidpmat_ctx->B_Ddelta,1,NULL);CHKERRQ(ierr);
    for (i=0;i<n_local_lambda;i++) {
      ierr = MatSetValue(fetidpmat_ctx->B_Ddelta,i,cols_B_delta[i],scaling_factors[i],INSERT_VALUES);CHKERRQ(ierr);
    }
    ierr = MatAssemblyBegin(fetidpmat_ctx->B_Ddelta,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd  (fetidpmat_ctx->B_Ddelta,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  ierr = PetscFree(scaling_factors);CHKERRQ(ierr);
  ierr = PetscFree(cols_B_delta);CHKERRQ(ierr);

  /* Layout of multipliers */
  ierr = PetscLayoutCreate(comm,&llay);CHKERRQ(ierr);
  ierr = PetscLayoutSetBlockSize(llay,1);CHKERRQ(ierr);
  ierr = PetscLayoutSetSize(llay,fetidpmat_ctx->n_lambda);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(llay);CHKERRQ(ierr);
  ierr = PetscLayoutGetLocalSize(llay,&fetidpmat_ctx->n);CHKERRQ(ierr);

  /* fetidpmat sizes */
  fetidpmat_ctx->n += nPgl;
  fetidpmat_ctx->N  = fetidpmat_ctx->n_lambda+nPg;

  /* Local work vector of multipliers */
  ierr = VecCreate(PETSC_COMM_SELF,&fetidpmat_ctx->lambda_local);CHKERRQ(ierr);
  ierr = VecSetSizes(fetidpmat_ctx->lambda_local,n_local_lambda,n_local_lambda);CHKERRQ(ierr);
  ierr = VecSetType(fetidpmat_ctx->lambda_local,VECSEQ);CHKERRQ(ierr);

  /* Global vector for FETI-DP linear system */
  ierr = VecCreate(comm,&fetidp_global);CHKERRQ(ierr);
  ierr = VecSetSizes(fetidp_global,fetidpmat_ctx->n,fetidpmat_ctx->N);CHKERRQ(ierr);
  ierr = VecSetType(fetidp_global,VECMPI);CHKERRQ(ierr);
  ierr = VecSetUp(fetidp_global);CHKERRQ(ierr);

  /* Decide layout for fetidp dofs: if it is a saddle point problem
     pressure is ordered first in the local part of the global vector
     of the FETI-DP linear system */
  if (nPg) {
    IS             IS_l2g_p,ais;
    PetscLayout    alay;
    const PetscInt *idxs,*pranges,*aranges,*lranges;
    PetscInt       *l2g_indices_p,rst;

    ierr = PetscMalloc1(nPl,&l2g_indices_p);CHKERRQ(ierr);
    ierr = VecGetLayout(fetidp_global,&alay);CHKERRQ(ierr);
    ierr = PetscLayoutGetRanges(alay,&aranges);CHKERRQ(ierr);
    ierr = PetscLayoutGetRanges(play,&pranges);CHKERRQ(ierr);
    ierr = PetscLayoutGetRanges(llay,&lranges);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingGetIndices(l2gmap_p,&idxs);CHKERRQ(ierr);
    /* shift local to global indices for pressure */
    for (i=0;i<nPl;i++) {
      PetscInt owner;

      ierr = PetscLayoutFindOwner(play,idxs[i],&owner);CHKERRQ(ierr);
      l2g_indices_p[i] = idxs[i]-pranges[owner]+aranges[owner];
    }
    ierr = ISLocalToGlobalMappingRestoreIndices(l2gmap_p,&idxs);CHKERRQ(ierr);
    ierr = ISCreateGeneral(comm,nPl,l2g_indices_p,PETSC_OWN_POINTER,&IS_l2g_p);CHKERRQ(ierr);

    /* local to global scatter for interface pressure */
    ierr = VecScatterCreate(fetidpmat_ctx->vP,NULL,fetidp_global,IS_l2g_p,&fetidpmat_ctx->l2g_p);CHKERRQ(ierr);
    ierr = ISDestroy(&IS_l2g_p);CHKERRQ(ierr);

    /* shift local to global indices for multipliers */
    for (i=0;i<n_local_lambda;i++) {
      PetscInt owner,ps;

      ierr = PetscLayoutFindOwner(llay,l2g_indices[i],&owner);CHKERRQ(ierr);
      ps = pranges[owner+1]-pranges[owner];
      l2g_indices[i] = l2g_indices[i]-lranges[owner]+aranges[owner]+ps;
    }

    /* scatter from alldofs to interface pressures global fetidp vector */
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

  /* Create some vectors needed by fetidp */
  ierr = VecDuplicate(pcis->vec1_B,&fetidpmat_ctx->temp_solution_B);CHKERRQ(ierr);
  ierr = VecDuplicate(pcis->vec1_D,&fetidpmat_ctx->temp_solution_D);CHKERRQ(ierr);

  test_fetidp = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-fetidp_check",&test_fetidp,NULL);CHKERRQ(ierr);

  if (test_fetidp && !pcbddc->use_deluxe_scaling) {
    Vec       test_vec,test_vec_p = NULL;
    PetscReal real_value;

    ierr = PetscViewerASCIIGetStdout(comm,&viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushSynchronized(viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"----------FETI_DP SIZES--------------\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%04d]: Sizes %D %D %D %D %D\n",rank,fetidpmat_ctx->n,fetidpmat_ctx->N,nPgl,nPg,nPl);CHKERRQ(ierr);
    ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"----------FETI_DP TESTS--------------\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"All tests should return zero!\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"FETIDP MAT context in the ");CHKERRQ(ierr);
    if (fully_redundant) {
      ierr = PetscViewerASCIIPrintf(viewer,"fully redundant case for lagrange multipliers.\n");CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIIPrintf(viewer,"Non-fully redundant case for lagrange multiplier.\n");CHKERRQ(ierr);
    }
    ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);

    /******************************************************************/
    /* TEST A/B: Test numbering of global fetidp dofs             */
    /******************************************************************/

    ierr = VecDuplicate(fetidpmat_ctx->lambda_local,&test_vec);CHKERRQ(ierr);
    ierr = VecSet(fetidp_global,1.0);CHKERRQ(ierr);
    ierr = VecSet(test_vec,1.);CHKERRQ(ierr);
    ierr = VecScatterBegin(fetidpmat_ctx->l2g_lambda,fetidp_global,fetidpmat_ctx->lambda_local,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterEnd(fetidpmat_ctx->l2g_lambda,fetidp_global,fetidpmat_ctx->lambda_local,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    if (fetidpmat_ctx->l2g_p) {
      ierr = VecDuplicate(fetidpmat_ctx->vP,&test_vec_p);CHKERRQ(ierr);
      ierr = VecSet(test_vec_p,1.);CHKERRQ(ierr);
      ierr = VecScatterBegin(fetidpmat_ctx->l2g_p,fetidp_global,fetidpmat_ctx->vP,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
      ierr = VecScatterEnd(fetidpmat_ctx->l2g_p,fetidp_global,fetidpmat_ctx->vP,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    }
    scalar_value = -1.0;
    ierr = VecAXPY(test_vec,scalar_value,fetidpmat_ctx->lambda_local);CHKERRQ(ierr);
    ierr = VecNorm(test_vec,NORM_INFINITY,&real_value);CHKERRQ(ierr);
    ierr = VecDestroy(&test_vec);CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedPrintf(viewer,"A[%04d]: CHECK glob to loc: % 1.14e\n",rank,real_value);CHKERRQ(ierr);
    if (fetidpmat_ctx->l2g_p) {
      ierr = VecAXPY(test_vec_p,scalar_value,fetidpmat_ctx->vP);CHKERRQ(ierr);
      ierr = VecNorm(test_vec_p,NORM_INFINITY,&real_value);CHKERRQ(ierr);
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"A[%04d]: CHECK glob to loc (p): % 1.14e\n",rank,real_value);CHKERRQ(ierr);
    }
    ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
    if (fully_redundant) {
      ierr = VecSet(fetidp_global,0.0);CHKERRQ(ierr);
      ierr = VecSet(fetidpmat_ctx->lambda_local,0.5);CHKERRQ(ierr);
      ierr = VecScatterBegin(fetidpmat_ctx->l2g_lambda,fetidpmat_ctx->lambda_local,fetidp_global,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecScatterEnd(fetidpmat_ctx->l2g_lambda,fetidpmat_ctx->lambda_local,fetidp_global,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecSum(fetidp_global,&scalar_value);CHKERRQ(ierr);
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"B[%04d]: CHECK loc to glob: % 1.14e\n",rank,PetscRealPart(scalar_value)-fetidpmat_ctx->n_lambda);CHKERRQ(ierr);
      ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
    }
    if (fetidpmat_ctx->l2g_p) {
      ierr = VecSet(pcis->vec1_N,1.0);CHKERRQ(ierr);
      ierr = VecSet(pcis->vec1_global,0.0);CHKERRQ(ierr);
      ierr = VecScatterBegin(matis->rctx,pcis->vec1_N,pcis->vec1_global,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
      ierr = VecScatterEnd(matis->rctx,pcis->vec1_N,pcis->vec1_global,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);

      ierr = VecSet(fetidp_global,0.0);CHKERRQ(ierr);
      ierr = VecSet(fetidpmat_ctx->vP,-1.0);CHKERRQ(ierr);
      ierr = VecScatterBegin(fetidpmat_ctx->l2g_p,fetidpmat_ctx->vP,fetidp_global,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecScatterEnd(fetidpmat_ctx->l2g_p,fetidpmat_ctx->vP,fetidp_global,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecScatterBegin(fetidpmat_ctx->g2g_p,fetidp_global,pcis->vec1_global,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
      ierr = VecScatterEnd(fetidpmat_ctx->g2g_p,fetidp_global,pcis->vec1_global,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
      ierr = VecScatterBegin(fetidpmat_ctx->g2g_p,pcis->vec1_global,fetidp_global,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecScatterEnd(fetidpmat_ctx->g2g_p,pcis->vec1_global,fetidp_global,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecSum(fetidp_global,&scalar_value);CHKERRQ(ierr);
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"B[%04d]: CHECK loc to glob (p): % 1.14e\n",rank,PetscRealPart(scalar_value));CHKERRQ(ierr);
    }
    /******************************************************************/
    /* TEST C: It should holds B_delta*w=0, w\in\widehat{W}           */
    /* This is the meaning of the B matrix                            */
    /******************************************************************/

    ierr = VecSetRandom(pcis->vec1_N,NULL);CHKERRQ(ierr);
    ierr = VecSet(pcis->vec1_global,0.0);CHKERRQ(ierr);
    ierr = VecScatterBegin(matis->rctx,pcis->vec1_N,pcis->vec1_global,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterEnd(matis->rctx,pcis->vec1_N,pcis->vec1_global,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterBegin(matis->rctx,pcis->vec1_global,pcis->vec1_N,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(matis->rctx,pcis->vec1_global,pcis->vec1_N,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterBegin(pcis->N_to_B,pcis->vec1_N,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(pcis->N_to_B,pcis->vec1_N,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    /* Action of B_delta */
    ierr = MatMult(fetidpmat_ctx->B_delta,pcis->vec1_B,fetidpmat_ctx->lambda_local);CHKERRQ(ierr);
    ierr = VecSet(fetidp_global,0.0);CHKERRQ(ierr);
    ierr = VecScatterBegin(fetidpmat_ctx->l2g_lambda,fetidpmat_ctx->lambda_local,fetidp_global,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(fetidpmat_ctx->l2g_lambda,fetidpmat_ctx->lambda_local,fetidp_global,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecNorm(fetidp_global,NORM_INFINITY,&real_value);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"C[coll]: CHECK infty norm of B_delta*w (w continuous): % 1.14e\n",real_value);CHKERRQ(ierr);
    ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);

    /******************************************************************/
    /* TEST D: It should holds E_Dw = w - P_Dw w\in\widetilde{W}     */
    /* E_D = R_D^TR                                                   */
    /* P_D = B_{D,delta}^T B_{delta}                                  */
    /* eq.44 Mandel Tezaur and Dohrmann 2005                          */
    /******************************************************************/

    /* compute a random vector in \widetilde{W} */
    ierr = VecSetRandom(pcis->vec1_N,NULL);CHKERRQ(ierr);
    scalar_value = 0.0;  /* set zero at vertices */
    ierr = VecGetArray(pcis->vec1_N,&array);CHKERRQ(ierr);
    for (i=0;i<n_vertices;i++) { array[vertex_indices[i]]=scalar_value; }
    ierr = VecRestoreArray(pcis->vec1_N,&array);CHKERRQ(ierr);
    /* store w for final comparison */
    ierr = VecDuplicate(pcis->vec1_B,&test_vec);CHKERRQ(ierr);
    ierr = VecScatterBegin(pcis->N_to_B,pcis->vec1_N,test_vec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(pcis->N_to_B,pcis->vec1_N,test_vec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);

    /* Jump operator P_D : results stored in pcis->vec1_B */

    ierr = VecScatterBegin(pcis->N_to_B,pcis->vec1_N,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(pcis->N_to_B,pcis->vec1_N,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    /* Action of B_delta */
    ierr = MatMult(fetidpmat_ctx->B_delta,pcis->vec1_B,fetidpmat_ctx->lambda_local);CHKERRQ(ierr);
    ierr = VecSet(fetidp_global,0.0);CHKERRQ(ierr);
    ierr = VecScatterBegin(fetidpmat_ctx->l2g_lambda,fetidpmat_ctx->lambda_local,fetidp_global,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(fetidpmat_ctx->l2g_lambda,fetidpmat_ctx->lambda_local,fetidp_global,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    /* Action of B_Ddelta^T */
    ierr = VecScatterBegin(fetidpmat_ctx->l2g_lambda,fetidp_global,fetidpmat_ctx->lambda_local,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterEnd(fetidpmat_ctx->l2g_lambda,fetidp_global,fetidpmat_ctx->lambda_local,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = MatMultTranspose(fetidpmat_ctx->B_Ddelta,fetidpmat_ctx->lambda_local,pcis->vec1_B);CHKERRQ(ierr);

    /* Average operator E_D : results stored in pcis->vec2_B */
    ierr = VecScatterBegin(pcis->N_to_B,pcis->vec1_N,pcis->vec2_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(pcis->N_to_B,pcis->vec1_N,pcis->vec2_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = PCBDDCScalingExtension(fetidpmat_ctx->pc,pcis->vec2_B,pcis->vec1_global);CHKERRQ(ierr);
    ierr = VecScatterBegin(pcis->global_to_B,pcis->vec1_global,pcis->vec2_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(pcis->global_to_B,pcis->vec1_global,pcis->vec2_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);

    /* test E_D=I-P_D */
    scalar_value = 1.0;
    ierr = VecAXPY(pcis->vec1_B,scalar_value,pcis->vec2_B);CHKERRQ(ierr);
    scalar_value = -1.0;
    ierr = VecAXPY(pcis->vec1_B,scalar_value,test_vec);CHKERRQ(ierr);
    ierr = VecNorm(pcis->vec1_B,NORM_INFINITY,&real_value);CHKERRQ(ierr);
    ierr = VecDestroy(&test_vec);CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedPrintf(viewer,"D[%04d] CHECK infty norm of E_D + P_D - I: % 1.14e\n",rank,real_value);CHKERRQ(ierr);
    ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);

    /******************************************************************/
    /* TEST E: It should holds R_D^TP_Dw=0 w\in\widetilde{W}          */
    /* eq.48 Mandel Tezaur and Dohrmann 2005                          */
    /******************************************************************/

    ierr = VecSetRandom(pcis->vec1_N,NULL);CHKERRQ(ierr);
    ierr = VecGetArray(pcis->vec1_N,&array);CHKERRQ(ierr);
    scalar_value = 0.0;  /* set zero at vertices */
    for (i=0;i<n_vertices;i++) { array[vertex_indices[i]]=scalar_value; }
    ierr = VecRestoreArray(pcis->vec1_N,&array);CHKERRQ(ierr);

    /* Jump operator P_D : results stored in pcis->vec1_B */

    ierr = VecScatterBegin(pcis->N_to_B,pcis->vec1_N,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(pcis->N_to_B,pcis->vec1_N,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    /* Action of B_delta */
    ierr = MatMult(fetidpmat_ctx->B_delta,pcis->vec1_B,fetidpmat_ctx->lambda_local);CHKERRQ(ierr);
    ierr = VecSet(fetidp_global,0.0);CHKERRQ(ierr);
    ierr = VecScatterBegin(fetidpmat_ctx->l2g_lambda,fetidpmat_ctx->lambda_local,fetidp_global,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(fetidpmat_ctx->l2g_lambda,fetidpmat_ctx->lambda_local,fetidp_global,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    /* Action of B_Ddelta^T */
    ierr = VecScatterBegin(fetidpmat_ctx->l2g_lambda,fetidp_global,fetidpmat_ctx->lambda_local,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterEnd(fetidpmat_ctx->l2g_lambda,fetidp_global,fetidpmat_ctx->lambda_local,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = MatMultTranspose(fetidpmat_ctx->B_Ddelta,fetidpmat_ctx->lambda_local,pcis->vec1_B);CHKERRQ(ierr);
    /* scaling */
    ierr = PCBDDCScalingExtension(fetidpmat_ctx->pc,pcis->vec1_B,pcis->vec1_global);CHKERRQ(ierr);
    ierr = VecNorm(pcis->vec1_global,NORM_INFINITY,&real_value);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"E[coll]: CHECK infty norm of R^T_D P_D: % 1.14e\n",real_value);CHKERRQ(ierr);
    ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);

    if (!fully_redundant) {
      /******************************************************************/
      /* TEST F: It should holds B_{delta}B^T_{D,delta}=I               */
      /* Corollary thm 14 Mandel Tezaur and Dohrmann 2005               */
      /******************************************************************/
      ierr = VecDuplicate(fetidp_global,&test_vec);CHKERRQ(ierr);
      ierr = VecSetRandom(fetidp_global,NULL);CHKERRQ(ierr);
      if (fetidpmat_ctx->l2g_p) {
        ierr = VecSet(fetidpmat_ctx->vP,0.);CHKERRQ(ierr);
        ierr = VecScatterBegin(fetidpmat_ctx->l2g_p,fetidpmat_ctx->vP,fetidp_global,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
        ierr = VecScatterEnd(fetidpmat_ctx->l2g_p,fetidpmat_ctx->vP,fetidp_global,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      }
      /* Action of B_Ddelta^T */
      ierr = VecScatterBegin(fetidpmat_ctx->l2g_lambda,fetidp_global,fetidpmat_ctx->lambda_local,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
      ierr = VecScatterEnd(fetidpmat_ctx->l2g_lambda,fetidp_global,fetidpmat_ctx->lambda_local,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
      ierr = MatMultTranspose(fetidpmat_ctx->B_Ddelta,fetidpmat_ctx->lambda_local,pcis->vec1_B);CHKERRQ(ierr);
      /* Action of B_delta */
      ierr = MatMult(fetidpmat_ctx->B_delta,pcis->vec1_B,fetidpmat_ctx->lambda_local);CHKERRQ(ierr);
      ierr = VecSet(test_vec,0.0);CHKERRQ(ierr);
      ierr = VecScatterBegin(fetidpmat_ctx->l2g_lambda,fetidpmat_ctx->lambda_local,test_vec,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecScatterEnd(fetidpmat_ctx->l2g_lambda,fetidpmat_ctx->lambda_local,test_vec,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      scalar_value = -1.0;
      ierr = VecAXPY(fetidp_global,scalar_value,test_vec);CHKERRQ(ierr);
      ierr = VecNorm(fetidp_global,NORM_INFINITY,&real_value);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"E[coll]: CHECK infty norm of P^T_D - I: % 1.14e\n",real_value);CHKERRQ(ierr);
      ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
      ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
      ierr = VecDestroy(&test_vec);CHKERRQ(ierr);
    }
    ierr = VecDestroy(&test_vec_p);CHKERRQ(ierr);
  }
  /* final cleanup */
  ierr = VecDestroy(&fetidp_global);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCBDDCSetupFETIDPPCContext"
PetscErrorCode PCBDDCSetupFETIDPPCContext(Mat fetimat, FETIDPPC_ctx fetidppc_ctx)
{
  FETIDPMat_ctx  mat_ctx;
  PC_IS          *pcis;
  PetscBool      lumped = PETSC_FALSE;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(fetimat,(void**)&mat_ctx);CHKERRQ(ierr);
  /* get references from objects created when setting up feti mat context */
  ierr = PetscObjectReference((PetscObject)mat_ctx->lambda_local);CHKERRQ(ierr);
  fetidppc_ctx->lambda_local = mat_ctx->lambda_local;
  ierr = PetscObjectReference((PetscObject)mat_ctx->B_Ddelta);CHKERRQ(ierr);
  fetidppc_ctx->B_Ddelta = mat_ctx->B_Ddelta;
  ierr = PetscObjectReference((PetscObject)mat_ctx->l2g_lambda);CHKERRQ(ierr);
  fetidppc_ctx->l2g_lambda = mat_ctx->l2g_lambda;
  /* create preconditioner */
  pcis = (PC_IS*)fetidppc_ctx->pc->data;
  /* Dirichlet preconditioner */
  ierr = PetscOptionsGetBool(NULL,NULL,"-fetidp_lumped",&lumped,NULL);CHKERRQ(ierr);
  if (!lumped) {
    ierr = MatCreateSchurComplement(pcis->A_II,pcis->A_II,pcis->A_IB,pcis->A_BI,pcis->A_BB,&fetidppc_ctx->S_j);CHKERRQ(ierr);
    ierr = MatSchurComplementSetKSP(fetidppc_ctx->S_j,pcis->ksp_D);CHKERRQ(ierr);
  } else {
    ierr = PetscObjectReference((PetscObject)pcis->A_BB);CHKERRQ(ierr);
    fetidppc_ctx->S_j = pcis->A_BB;
  }
  /* saddle-point */
  if (mat_ctx->xPg) {
    Mat PAmat = NULL,PPmat = NULL;

    ierr = PetscObjectReference((PetscObject)mat_ctx->xPg);CHKERRQ(ierr);
    fetidppc_ctx->xPg = mat_ctx->xPg;
    ierr = PetscObjectReference((PetscObject)mat_ctx->yPg);CHKERRQ(ierr);
    fetidppc_ctx->yPg = mat_ctx->yPg;
    ierr = PetscObjectQuery((PetscObject)fetidppc_ctx->pc,"__KSPFETIDP_PPmat",(PetscObject*)&PPmat);CHKERRQ(ierr);
    ierr = PetscObjectQuery((PetscObject)fetidppc_ctx->pc,"__KSPFETIDP_PAmat",(PetscObject*)&PAmat);CHKERRQ(ierr);
    ierr = KSPCreate(PetscObjectComm((PetscObject)fetidppc_ctx->pc),&fetidppc_ctx->kP);CHKERRQ(ierr);
    ierr = KSPSetOperators(fetidppc_ctx->kP,PAmat,PPmat);CHKERRQ(ierr);
    ierr = KSPSetOptionsPrefix(fetidppc_ctx->kP,"fetidp_pmass_");CHKERRQ(ierr);
    ierr = KSPSetFromOptions(fetidppc_ctx->kP);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FETIDPMatMult"
PetscErrorCode FETIDPMatMult(Mat fetimat, Vec x, Vec y)
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
      ierr = MatMult(mat_ctx->Bt_BI,mat_ctx->vP,pcis->vec1_D);CHKERRQ(ierr);
    }
    ierr = MatMultAdd(mat_ctx->Bt_BB,mat_ctx->vP,pcis->vec1_B,pcis->vec1_B);CHKERRQ(ierr);
  } else {
    if (pcbddc->switch_static) {
      ierr = VecSet(pcis->vec1_D,0.0);CHKERRQ(ierr);
    }
  }
  /* Application of \widetilde{S}^-1 */
  ierr = PetscMemzero(pcbddc->benign_p0,pcbddc->benign_n*sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = PCBDDCApplyInterfacePreconditioner(mat_ctx->pc,PETSC_FALSE);CHKERRQ(ierr);
  ierr = PetscMemzero(pcbddc->benign_p0,pcbddc->benign_n*sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = VecSet(y,0.0);CHKERRQ(ierr);
  /* Application of B_delta */
  ierr = MatMult(mat_ctx->B_delta,pcis->vec1_B,mat_ctx->lambda_local);CHKERRQ(ierr);
  /* Contribution from boundary pressures */
  if (mat_ctx->C) {
    const PetscScalar *lx;
    PetscScalar       *ly;

    /* pressures ordered first in x and y */
    ierr = VecGetArrayRead(x,&lx);CHKERRQ(ierr);
    ierr = VecGetArray(y,&ly);CHKERRQ(ierr);
    ierr = VecPlaceArray(mat_ctx->xPg,lx);CHKERRQ(ierr);
    ierr = VecPlaceArray(mat_ctx->yPg,ly);CHKERRQ(ierr);
    ierr = MatMult(mat_ctx->C,mat_ctx->xPg,mat_ctx->yPg);CHKERRQ(ierr);
    ierr = VecResetArray(mat_ctx->xPg);CHKERRQ(ierr);
    ierr = VecResetArray(mat_ctx->yPg);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(x,&lx);CHKERRQ(ierr);
    ierr = VecRestoreArray(y,&ly);CHKERRQ(ierr);
  }
  /* Add contribution from saddle point */
  if (mat_ctx->l2g_p) {
    ierr = MatMult(mat_ctx->B_BB,pcis->vec1_B,mat_ctx->vP);CHKERRQ(ierr);
    if (pcbddc->switch_static) {
      ierr = MatMultAdd(mat_ctx->B_BI,pcis->vec1_D,mat_ctx->vP,mat_ctx->vP);CHKERRQ(ierr);
    }
    ierr = VecScatterBegin(mat_ctx->l2g_p,mat_ctx->vP,y,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(mat_ctx->l2g_p,mat_ctx->vP,y,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  }
  ierr = VecScatterBegin(mat_ctx->l2g_lambda,mat_ctx->lambda_local,y,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(mat_ctx->l2g_lambda,mat_ctx->lambda_local,y,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FETIDPMatMultTranspose"
PetscErrorCode FETIDPMatMultTranspose(Mat fetimat, Vec x, Vec y)
{
  FETIDPMat_ctx  mat_ctx;
  PC_IS          *pcis;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(fetimat,(void**)&mat_ctx);CHKERRQ(ierr);
  pcis = (PC_IS*)mat_ctx->pc->data;
  /* Application of B_delta^T */
  ierr = VecScatterBegin(mat_ctx->l2g_lambda,x,mat_ctx->lambda_local,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd(mat_ctx->l2g_lambda,x,mat_ctx->lambda_local,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = MatMultTranspose(mat_ctx->B_delta,mat_ctx->lambda_local,pcis->vec1_B);CHKERRQ(ierr);
  /* Application of \widetilde{S}^-1 */
  ierr = VecSet(pcis->vec1_D,0.0);CHKERRQ(ierr);
  ierr = PCBDDCApplyInterfacePreconditioner(mat_ctx->pc,PETSC_TRUE);CHKERRQ(ierr);
  /* Application of B_delta */
  ierr = MatMult(mat_ctx->B_delta,pcis->vec1_B,mat_ctx->lambda_local);CHKERRQ(ierr);
  ierr = VecSet(y,0.0);CHKERRQ(ierr);
  ierr = VecScatterBegin(mat_ctx->l2g_lambda,mat_ctx->lambda_local,y,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(mat_ctx->l2g_lambda,mat_ctx->lambda_local,y,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FETIDPPCApply"
PetscErrorCode FETIDPPCApply(PC fetipc, Vec x, Vec y)
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
  ierr = MatMult(pc_ctx->S_j,pcis->vec2_B,pcis->vec1_B);CHKERRQ(ierr);
  /* Application of B_Ddelta */
  ierr = MatMult(pc_ctx->B_Ddelta,pcis->vec1_B,pc_ctx->lambda_local);CHKERRQ(ierr);
  ierr = VecSet(y,0.0);CHKERRQ(ierr);
  ierr = VecScatterBegin(pc_ctx->l2g_lambda,pc_ctx->lambda_local,y,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(pc_ctx->l2g_lambda,pc_ctx->lambda_local,y,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FETIDPPCApplyTranspose"
PetscErrorCode FETIDPPCApplyTranspose(PC fetipc, Vec x, Vec y)
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
  ierr = MatMultTranspose(pc_ctx->S_j,pcis->vec2_B,pcis->vec1_B);CHKERRQ(ierr);
  /* Application of B_Ddelta */
  ierr = MatMult(pc_ctx->B_Ddelta,pcis->vec1_B,pc_ctx->lambda_local);CHKERRQ(ierr);
  ierr = VecSet(y,0.0);CHKERRQ(ierr);
  ierr = VecScatterBegin(pc_ctx->l2g_lambda,pc_ctx->lambda_local,y,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(pc_ctx->l2g_lambda,pc_ctx->lambda_local,y,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
