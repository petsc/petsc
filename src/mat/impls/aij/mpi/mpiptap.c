
/*
  Defines projective product routines where A is a MPIAIJ matrix
          C = P^T * A * P
*/

#include <../src/mat/impls/aij/seq/aij.h>   /*I "petscmat.h" I*/
#include <../src/mat/utils/freespace.h>
#include <../src/mat/impls/aij/mpi/mpiaij.h>
#include <petscbt.h>
#include <petsctime.h>

/* #define PTAP_PROFILE */

PetscErrorCode MatView_MPIAIJ_PtAP(Mat A,PetscViewer viewer)
{
  PetscErrorCode    ierr;
  Mat_MPIAIJ        *a=(Mat_MPIAIJ*)A->data;
  Mat_APMPI         *ptap=a->ap;
  PetscBool         iascii;
  PetscViewerFormat format;

  PetscFunctionBegin;
  if (!ptap) {
    /* hack: MatDuplicate() sets oldmat->ops->view to newmat which is a base mat class with null ptpa! */
    A->ops->view = MatView_MPIAIJ;
    ierr = (A->ops->view)(A,viewer);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
    if (format == PETSC_VIEWER_ASCII_INFO || format == PETSC_VIEWER_ASCII_INFO_DETAIL) {
      if (ptap->algType == 0) {
        ierr = PetscViewerASCIIPrintf(viewer,"using scalable MatPtAP() implementation\n");CHKERRQ(ierr);
      } else if (ptap->algType == 1) {
        ierr = PetscViewerASCIIPrintf(viewer,"using nonscalable MatPtAP() implementation\n");CHKERRQ(ierr);
      }
    }
  }
  ierr = (ptap->view)(A,viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatFreeIntermediateDataStructures_MPIAIJ_AP(Mat A)
{
  PetscErrorCode      ierr;
  Mat_MPIAIJ          *a=(Mat_MPIAIJ*)A->data;
  Mat_APMPI           *ptap=a->ap;
  Mat_Merge_SeqsToMPI *merge;

  PetscFunctionBegin;
  if (!ptap) PetscFunctionReturn(0);

  ierr = PetscFree2(ptap->startsj_s,ptap->startsj_r);CHKERRQ(ierr);
  ierr = PetscFree(ptap->bufa);CHKERRQ(ierr);
  ierr = MatDestroy(&ptap->P_loc);CHKERRQ(ierr);
  ierr = MatDestroy(&ptap->P_oth);CHKERRQ(ierr);
  ierr = MatDestroy(&ptap->A_loc);CHKERRQ(ierr); /* used by MatTransposeMatMult() */
  ierr = MatDestroy(&ptap->Rd);CHKERRQ(ierr);
  ierr = MatDestroy(&ptap->Ro);CHKERRQ(ierr);
  if (ptap->AP_loc) { /* used by alg_rap */
    Mat_SeqAIJ *ap = (Mat_SeqAIJ*)(ptap->AP_loc)->data;
    ierr = PetscFree(ap->i);CHKERRQ(ierr);
    ierr = PetscFree2(ap->j,ap->a);CHKERRQ(ierr);
    ierr = MatDestroy(&ptap->AP_loc);CHKERRQ(ierr);
  } else { /* used by alg_ptap */
    ierr = PetscFree(ptap->api);CHKERRQ(ierr);
    ierr = PetscFree(ptap->apj);CHKERRQ(ierr);
  }
  ierr = MatDestroy(&ptap->C_loc);CHKERRQ(ierr);
  ierr = MatDestroy(&ptap->C_oth);CHKERRQ(ierr);
  if (ptap->apa) {ierr = PetscFree(ptap->apa);CHKERRQ(ierr);}

  ierr = MatDestroy(&ptap->Pt);CHKERRQ(ierr);

  merge=ptap->merge;
  if (merge) { /* used by alg_ptap */
    ierr = PetscFree(merge->id_r);CHKERRQ(ierr);
    ierr = PetscFree(merge->len_s);CHKERRQ(ierr);
    ierr = PetscFree(merge->len_r);CHKERRQ(ierr);
    ierr = PetscFree(merge->bi);CHKERRQ(ierr);
    ierr = PetscFree(merge->bj);CHKERRQ(ierr);
    ierr = PetscFree(merge->buf_ri[0]);CHKERRQ(ierr);
    ierr = PetscFree(merge->buf_ri);CHKERRQ(ierr);
    ierr = PetscFree(merge->buf_rj[0]);CHKERRQ(ierr);
    ierr = PetscFree(merge->buf_rj);CHKERRQ(ierr);
    ierr = PetscFree(merge->coi);CHKERRQ(ierr);
    ierr = PetscFree(merge->coj);CHKERRQ(ierr);
    ierr = PetscFree(merge->owners_co);CHKERRQ(ierr);
    ierr = PetscLayoutDestroy(&merge->rowmap);CHKERRQ(ierr);
    ierr = PetscFree(ptap->merge);CHKERRQ(ierr);
  }
  ierr = ISLocalToGlobalMappingDestroy(&ptap->ltog);CHKERRQ(ierr);

  A->ops->destroy = ptap->destroy;
  ierr = PetscFree(a->ap);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatDestroy_MPIAIJ_PtAP(Mat A)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = (*A->ops->freeintermediatedatastructures)(A);CHKERRQ(ierr);
  ierr = (*A->ops->destroy)(A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatPtAP_MPIAIJ_MPIAIJ(Mat A,Mat P,MatReuse scall,PetscReal fill,Mat *C)
{
  PetscErrorCode ierr;
  PetscBool      flg;
  MPI_Comm       comm;
#if !defined(PETSC_HAVE_HYPRE)
  const char          *algTypes[2] = {"scalable","nonscalable"};
  PetscInt            nalg=2;
#else
  const char          *algTypes[3] = {"scalable","nonscalable","hypre"};
  PetscInt            nalg=3;
#endif
  PetscInt            pN=P->cmap->N,alg=1; /* set default algorithm */

  PetscFunctionBegin;
  /* check if matrix local sizes are compatible */
  ierr = PetscObjectGetComm((PetscObject)A,&comm);CHKERRQ(ierr);
  if (A->rmap->rstart != P->rmap->rstart || A->rmap->rend != P->rmap->rend) SETERRQ4(comm,PETSC_ERR_ARG_SIZ,"Matrix local dimensions are incompatible, Arow (%D, %D) != Prow (%D,%D)",A->rmap->rstart,A->rmap->rend,P->rmap->rstart,P->rmap->rend);
  if (A->cmap->rstart != P->rmap->rstart || A->cmap->rend != P->rmap->rend) SETERRQ4(comm,PETSC_ERR_ARG_SIZ,"Matrix local dimensions are incompatible, Acol (%D, %D) != Prow (%D,%D)",A->cmap->rstart,A->cmap->rend,P->rmap->rstart,P->rmap->rend);

  if (scall == MAT_INITIAL_MATRIX) {
    /* pick an algorithm */
    ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)A),((PetscObject)A)->prefix,"MatPtAP","Mat");CHKERRQ(ierr);
    ierr = PetscOptionsEList("-matptap_via","Algorithmic approach","MatPtAP",algTypes,nalg,algTypes[alg],&alg,&flg);CHKERRQ(ierr);
    ierr = PetscOptionsEnd();CHKERRQ(ierr);

    if (!flg && pN > 100000) { /* may switch to scalable algorithm as default */
      MatInfo     Ainfo,Pinfo;
      PetscInt    nz_local;
      PetscBool   alg_scalable_loc=PETSC_FALSE,alg_scalable;

      ierr = MatGetInfo(A,MAT_LOCAL,&Ainfo);CHKERRQ(ierr);
      ierr = MatGetInfo(P,MAT_LOCAL,&Pinfo);CHKERRQ(ierr);
      nz_local = (PetscInt)(Ainfo.nz_allocated + Pinfo.nz_allocated);

      if (pN > fill*nz_local) alg_scalable_loc = PETSC_TRUE;
      ierr = MPIU_Allreduce(&alg_scalable_loc,&alg_scalable,1,MPIU_BOOL,MPI_LOR,comm);CHKERRQ(ierr);

      if (alg_scalable) {
        alg = 0; /* scalable algorithm would 50% slower than nonscalable algorithm */
      }
    }

    switch (alg) {
    case 1:
      /* do R=P^T locally, then C=R*A*P -- nonscalable */
      ierr = PetscLogEventBegin(MAT_PtAPSymbolic,A,P,0,0);CHKERRQ(ierr);
      ierr = MatPtAPSymbolic_MPIAIJ_MPIAIJ(A,P,fill,C);CHKERRQ(ierr);
      ierr = PetscLogEventEnd(MAT_PtAPSymbolic,A,P,0,0);CHKERRQ(ierr);
      break;
#if defined(PETSC_HAVE_HYPRE)
    case 2:
      /* Use boomerAMGBuildCoarseOperator */
      ierr = PetscLogEventBegin(MAT_PtAPSymbolic,A,P,0,0);CHKERRQ(ierr);
      ierr = MatPtAPSymbolic_AIJ_AIJ_wHYPRE(A,P,fill,C);CHKERRQ(ierr);
      ierr = PetscLogEventEnd(MAT_PtAPSymbolic,A,P,0,0);CHKERRQ(ierr);
      break;
#endif
    default:
      /* do R=P^T locally, then C=R*A*P */
      ierr = PetscLogEventBegin(MAT_PtAPSymbolic,A,P,0,0);CHKERRQ(ierr);
      ierr = MatPtAPSymbolic_MPIAIJ_MPIAIJ_scalable(A,P,fill,C);CHKERRQ(ierr);
      ierr = PetscLogEventEnd(MAT_PtAPSymbolic,A,P,0,0);CHKERRQ(ierr);
      break;
    }

    if (alg == 0 || alg == 1) {
      Mat_MPIAIJ *c  = (Mat_MPIAIJ*)(*C)->data;
      Mat_APMPI  *ap = c->ap;
      ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)(*C)),((PetscObject)(*C))->prefix,"MatFreeIntermediateDataStructures","Mat");CHKERRQ(ierr);
      ap->freestruct = PETSC_FALSE;
      ierr = PetscOptionsBool("-mat_freeintermediatedatastructures","Free intermediate data structures", "MatFreeIntermediateDataStructures",ap->freestruct,&ap->freestruct, NULL);CHKERRQ(ierr);
      ierr = PetscOptionsEnd();CHKERRQ(ierr);
    }
  }

  ierr = PetscLogEventBegin(MAT_PtAPNumeric,A,P,0,0);CHKERRQ(ierr);
  ierr = (*(*C)->ops->ptapnumeric)(A,P,*C);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_PtAPNumeric,A,P,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatPtAPNumeric_MPIAIJ_MPIAIJ_scalable(Mat A,Mat P,Mat C)
{
  PetscErrorCode    ierr;
  Mat_MPIAIJ        *a=(Mat_MPIAIJ*)A->data,*p=(Mat_MPIAIJ*)P->data,*c=(Mat_MPIAIJ*)C->data;
  Mat_SeqAIJ        *ad=(Mat_SeqAIJ*)(a->A)->data,*ao=(Mat_SeqAIJ*)(a->B)->data;
  Mat_SeqAIJ        *ap,*p_loc,*p_oth,*c_seq;
  Mat_APMPI         *ptap = c->ap;
  Mat               AP_loc,C_loc,C_oth;
  PetscInt          i,rstart,rend,cm,ncols,row,*api,*apj,am = A->rmap->n,apnz,nout;
  PetscScalar       *apa;
  const PetscInt    *cols;
  const PetscScalar *vals;

  PetscFunctionBegin;
  if (!ptap) {
    MPI_Comm comm;
    ierr = PetscObjectGetComm((PetscObject)C,&comm);CHKERRQ(ierr);
    SETERRQ(comm,PETSC_ERR_ARG_WRONGSTATE,"PtAP cannot be reused. Do not call MatFreeIntermediateDataStructures() or use '-mat_freeintermediatedatastructures'");
  }

  ierr = MatZeroEntries(C);CHKERRQ(ierr);

  /* 1) get R = Pd^T,Ro = Po^T */
  if (ptap->reuse == MAT_REUSE_MATRIX) {
    ierr = MatTranspose(p->A,MAT_REUSE_MATRIX,&ptap->Rd);CHKERRQ(ierr);
    ierr = MatTranspose(p->B,MAT_REUSE_MATRIX,&ptap->Ro);CHKERRQ(ierr);
  }

  /* 2) get AP_loc */
  AP_loc = ptap->AP_loc;
  ap = (Mat_SeqAIJ*)AP_loc->data;

  /* 2-1) get P_oth = ptap->P_oth  and P_loc = ptap->P_loc */
  /*-----------------------------------------------------*/
  if (ptap->reuse == MAT_REUSE_MATRIX) {
    /* P_oth and P_loc are obtained in MatPtASymbolic() when reuse == MAT_INITIAL_MATRIX */
    ierr = MatGetBrowsOfAoCols_MPIAIJ(A,P,MAT_REUSE_MATRIX,&ptap->startsj_s,&ptap->startsj_r,&ptap->bufa,&ptap->P_oth);CHKERRQ(ierr);
    ierr = MatMPIAIJGetLocalMat(P,MAT_REUSE_MATRIX,&ptap->P_loc);CHKERRQ(ierr);
  }

  /* 2-2) compute numeric A_loc*P - dominating part */
  /* ---------------------------------------------- */
  /* get data from symbolic products */
  p_loc = (Mat_SeqAIJ*)(ptap->P_loc)->data;
  if (ptap->P_oth) p_oth = (Mat_SeqAIJ*)(ptap->P_oth)->data;
  else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"ptap->P_oth is NULL. Cannot proceed!");

  api   = ap->i;
  apj   = ap->j;
  ierr = ISLocalToGlobalMappingApply(ptap->ltog,api[AP_loc->rmap->n],apj,apj);CHKERRQ(ierr);
  for (i=0; i<am; i++) {
    /* AP[i,:] = A[i,:]*P = Ad*P_loc Ao*P_oth */
    apnz = api[i+1] - api[i];
    apa = ap->a + api[i];
    ierr = PetscMemzero(apa,sizeof(PetscScalar)*apnz);CHKERRQ(ierr);
    AProw_scalable(i,ad,ao,p_loc,p_oth,api,apj,apa);
    ierr = PetscLogFlops(2.0*apnz);CHKERRQ(ierr);
  }
  ierr = ISGlobalToLocalMappingApply(ptap->ltog,IS_GTOLM_DROP,api[AP_loc->rmap->n],apj,&nout,apj);CHKERRQ(ierr);
  if (api[AP_loc->rmap->n] != nout) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Incorrect mapping %D != %D\n",api[AP_loc->rmap->n],nout);

  /* 3) C_loc = Rd*AP_loc, C_oth = Ro*AP_loc */
  /* Always use scalable version since we are in the MPI scalable version */
  ierr = MatMatMultNumeric_SeqAIJ_SeqAIJ_Scalable(ptap->Rd,AP_loc,ptap->C_loc);CHKERRQ(ierr);
  ierr = MatMatMultNumeric_SeqAIJ_SeqAIJ_Scalable(ptap->Ro,AP_loc,ptap->C_oth);CHKERRQ(ierr);

  C_loc = ptap->C_loc;
  C_oth = ptap->C_oth;

  /* add C_loc and Co to to C */
  ierr = MatGetOwnershipRange(C,&rstart,&rend);CHKERRQ(ierr);

  /* C_loc -> C */
  cm    = C_loc->rmap->N;
  c_seq = (Mat_SeqAIJ*)C_loc->data;
  cols = c_seq->j;
  vals = c_seq->a;
  ierr = ISLocalToGlobalMappingApply(ptap->ltog,c_seq->i[C_loc->rmap->n],c_seq->j,c_seq->j);CHKERRQ(ierr);

  /* The (fast) MatSetValues_MPIAIJ_CopyFromCSRFormat function can only be used when C->was_assembled is PETSC_FALSE and */
  /* when there are no off-processor parts.  */
  /* If was_assembled is true, then the statement aj[rowstart_diag+dnz_row] = mat_j[col] - cstart; in MatSetValues_MPIAIJ_CopyFromCSRFormat */
  /* is no longer true. Then the more complex function MatSetValues_MPIAIJ() has to be used, where the column index is looked up from */
  /* a table, and other, more complex stuff has to be done. */
  if (C->assembled) {
    C->was_assembled = PETSC_TRUE;
    C->assembled     = PETSC_FALSE;
  }
  if (C->was_assembled) {
    for (i=0; i<cm; i++) {
      ncols = c_seq->i[i+1] - c_seq->i[i];
      row = rstart + i;
      ierr = MatSetValues_MPIAIJ(C,1,&row,ncols,cols,vals,ADD_VALUES);CHKERRQ(ierr);
      cols += ncols; vals += ncols;
    }
  } else {
    ierr = MatSetValues_MPIAIJ_CopyFromCSRFormat(C,c_seq->j,c_seq->i,c_seq->a);CHKERRQ(ierr);
  }
  ierr = ISGlobalToLocalMappingApply(ptap->ltog,IS_GTOLM_DROP,c_seq->i[C_loc->rmap->n],c_seq->j,&nout,c_seq->j);CHKERRQ(ierr);
  if (c_seq->i[C_loc->rmap->n] != nout) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Incorrect mapping %D != %D\n",c_seq->i[C_loc->rmap->n],nout);

  /* Co -> C, off-processor part */
  cm = C_oth->rmap->N;
  c_seq = (Mat_SeqAIJ*)C_oth->data;
  cols = c_seq->j;
  vals = c_seq->a;
  ierr = ISLocalToGlobalMappingApply(ptap->ltog,c_seq->i[C_oth->rmap->n],c_seq->j,c_seq->j);CHKERRQ(ierr);
  for (i=0; i<cm; i++) {
    ncols = c_seq->i[i+1] - c_seq->i[i];
    row = p->garray[i];
    ierr = MatSetValues(C,1,&row,ncols,cols,vals,ADD_VALUES);CHKERRQ(ierr);
    cols += ncols; vals += ncols;
  }
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ptap->reuse = MAT_REUSE_MATRIX;

  ierr = ISGlobalToLocalMappingApply(ptap->ltog,IS_GTOLM_DROP,c_seq->i[C_oth->rmap->n],c_seq->j,&nout,c_seq->j);CHKERRQ(ierr);
  if (c_seq->i[C_oth->rmap->n] != nout) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Incorrect mapping %D != %D\n",c_seq->i[C_loc->rmap->n],nout);

  /* supporting struct ptap consumes almost same amount of memory as C=PtAP, release it if C will not be updated by A and P */
  if (ptap->freestruct) {
    ierr = MatFreeIntermediateDataStructures(C);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatPtAPSymbolic_MPIAIJ_MPIAIJ_scalable(Mat A,Mat P,PetscReal fill,Mat *C)
{
  PetscErrorCode      ierr;
  Mat_APMPI           *ptap;
  Mat_MPIAIJ          *a=(Mat_MPIAIJ*)A->data,*p=(Mat_MPIAIJ*)P->data,*c;
  MPI_Comm            comm;
  PetscMPIInt         size,rank;
  Mat                 Cmpi,P_loc,P_oth;
  PetscFreeSpaceList  free_space=NULL,current_space=NULL;
  PetscInt            am=A->rmap->n,pm=P->rmap->n,pN=P->cmap->N,pn=P->cmap->n;
  PetscInt            *lnk,i,k,pnz,row,nsend;
  PetscMPIInt         tagi,tagj,*len_si,*len_s,*len_ri,icompleted=0,nrecv;
  PetscInt            **buf_rj,**buf_ri,**buf_ri_k;
  PetscInt            len,proc,*dnz,*onz,*owners,nzi,nspacedouble;
  PetscInt            nrows,*buf_s,*buf_si,*buf_si_i,**nextrow,**nextci;
  MPI_Request         *swaits,*rwaits;
  MPI_Status          *sstatus,rstatus;
  PetscLayout         rowmap;
  PetscInt            *owners_co,*coi,*coj;    /* i and j array of (p->B)^T*A*P - used in the communication */
  PetscMPIInt         *len_r,*id_r;    /* array of length of comm->size, store send/recv matrix values */
  PetscInt            *api,*apj,*Jptr,apnz,*prmap=p->garray,con,j,Crmax,*aj,*ai,*pi,nout;
  Mat_SeqAIJ          *p_loc,*p_oth=NULL,*ad=(Mat_SeqAIJ*)(a->A)->data,*ao=NULL,*c_loc,*c_oth;
  PetscScalar         *apv;
  PetscTable          ta;
  MatType             mtype;
  const char          *prefix;
#if defined(PETSC_USE_INFO)
  PetscReal           apfill;
#endif

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)A,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);

  if (size > 1) ao = (Mat_SeqAIJ*)(a->B)->data;

  /* create symbolic parallel matrix Cmpi */
  ierr = MatCreate(comm,&Cmpi);CHKERRQ(ierr);
  ierr = MatGetType(A,&mtype);CHKERRQ(ierr);
  ierr = MatSetType(Cmpi,mtype);CHKERRQ(ierr);

  /* create struct Mat_APMPI and attached it to C later */
  ierr        = PetscNew(&ptap);CHKERRQ(ierr);
  ptap->reuse = MAT_INITIAL_MATRIX;
  ptap->algType = 0;

  /* get P_oth by taking rows of P (= non-zero cols of local A) from other processors */
  ierr = MatGetBrowsOfAoCols_MPIAIJ(A,P,MAT_INITIAL_MATRIX,&ptap->startsj_s,&ptap->startsj_r,&ptap->bufa,&P_oth);CHKERRQ(ierr);
  /* get P_loc by taking all local rows of P */
  ierr = MatMPIAIJGetLocalMat(P,MAT_INITIAL_MATRIX,&P_loc);CHKERRQ(ierr);

  ptap->P_loc = P_loc;
  ptap->P_oth = P_oth;

  /* (0) compute Rd = Pd^T, Ro = Po^T  */
  /* --------------------------------- */
  ierr = MatTranspose(p->A,MAT_INITIAL_MATRIX,&ptap->Rd);CHKERRQ(ierr);
  ierr = MatTranspose(p->B,MAT_INITIAL_MATRIX,&ptap->Ro);CHKERRQ(ierr);

  /* (1) compute symbolic AP = A_loc*P = Ad*P_loc + Ao*P_oth (api,apj) */
  /* ----------------------------------------------------------------- */
  p_loc  = (Mat_SeqAIJ*)P_loc->data;
  if (P_oth) p_oth = (Mat_SeqAIJ*)P_oth->data;

  /* create and initialize a linked list */
  ierr = PetscTableCreate(pn,pN,&ta);CHKERRQ(ierr); /* for compute AP_loc and Cmpi */
  MatRowMergeMax_SeqAIJ(p_loc,P_loc->rmap->N,ta);
  MatRowMergeMax_SeqAIJ(p_oth,P_oth->rmap->N,ta);
  ierr = PetscTableGetCount(ta,&Crmax);CHKERRQ(ierr); /* Crmax = nnz(sum of Prows) */

  ierr = PetscLLCondensedCreate_Scalable(Crmax,&lnk);CHKERRQ(ierr);

  /* Initial FreeSpace size is fill*(nnz(A) + nnz(P)) */
  if (ao) {
    ierr = PetscFreeSpaceGet(PetscRealIntMultTruncate(fill,PetscIntSumTruncate(ad->i[am],PetscIntSumTruncate(ao->i[am],p_loc->i[pm]))),&free_space);CHKERRQ(ierr);
  } else {
    ierr = PetscFreeSpaceGet(PetscRealIntMultTruncate(fill,PetscIntSumTruncate(ad->i[am],p_loc->i[pm])),&free_space);CHKERRQ(ierr);
  }
  current_space = free_space;
  nspacedouble  = 0;

  ierr   = PetscMalloc1(am+1,&api);CHKERRQ(ierr);
  api[0] = 0;
  for (i=0; i<am; i++) {
    /* diagonal portion: Ad[i,:]*P */
    ai = ad->i; pi = p_loc->i;
    nzi = ai[i+1] - ai[i];
    aj  = ad->j + ai[i];
    for (j=0; j<nzi; j++) {
      row  = aj[j];
      pnz  = pi[row+1] - pi[row];
      Jptr = p_loc->j + pi[row];
      /* add non-zero cols of P into the sorted linked list lnk */
      ierr = PetscLLCondensedAddSorted_Scalable(pnz,Jptr,lnk);CHKERRQ(ierr);
    }
    /* off-diagonal portion: Ao[i,:]*P */
    if (ao) {
      ai = ao->i; pi = p_oth->i;
      nzi = ai[i+1] - ai[i];
      aj  = ao->j + ai[i];
      for (j=0; j<nzi; j++) {
        row  = aj[j];
        pnz  = pi[row+1] - pi[row];
        Jptr = p_oth->j + pi[row];
        ierr = PetscLLCondensedAddSorted_Scalable(pnz,Jptr,lnk);CHKERRQ(ierr);
      }
    }
    apnz     = lnk[0];
    api[i+1] = api[i] + apnz;

    /* if free space is not available, double the total space in the list */
    if (current_space->local_remaining<apnz) {
      ierr = PetscFreeSpaceGet(PetscIntSumTruncate(apnz,current_space->total_array_size),&current_space);CHKERRQ(ierr);
      nspacedouble++;
    }

    /* Copy data into free space, then initialize lnk */
    ierr = PetscLLCondensedClean_Scalable(apnz,current_space->array,lnk);CHKERRQ(ierr);

    current_space->array           += apnz;
    current_space->local_used      += apnz;
    current_space->local_remaining -= apnz;
  }
  /* Allocate space for apj and apv, initialize apj, and */
  /* destroy list of free space and other temporary array(s) */
  ierr = PetscCalloc2(api[am],&apj,api[am],&apv);CHKERRQ(ierr);
  ierr = PetscFreeSpaceContiguous(&free_space,apj);CHKERRQ(ierr);
  ierr = PetscLLCondensedDestroy_Scalable(lnk);CHKERRQ(ierr);

  /* Create AP_loc for reuse */
  ierr = MatCreateSeqAIJWithArrays(PETSC_COMM_SELF,am,pN,api,apj,apv,&ptap->AP_loc);CHKERRQ(ierr);
  ierr = MatSeqAIJCompactOutExtraColumns_SeqAIJ(ptap->AP_loc, &ptap->ltog);CHKERRQ(ierr);

#if defined(PETSC_USE_INFO)
  if (ao) {
    apfill = (PetscReal)api[am]/(ad->i[am]+ao->i[am]+p_loc->i[pm]+1);
  } else {
    apfill = (PetscReal)api[am]/(ad->i[am]+p_loc->i[pm]+1);
  }
  ptap->AP_loc->info.mallocs           = nspacedouble;
  ptap->AP_loc->info.fill_ratio_given  = fill;
  ptap->AP_loc->info.fill_ratio_needed = apfill;

  if (api[am]) {
    ierr = PetscInfo3(ptap->AP_loc,"Scalable algorithm, AP_loc reallocs %D; Fill ratio: given %g needed %g.\n",nspacedouble,(double)fill,(double)apfill);CHKERRQ(ierr);
    ierr = PetscInfo1(ptap->AP_loc,"Use MatPtAP(A,B,MatReuse,%g,&C) for best AP_loc performance.;\n",(double)apfill);CHKERRQ(ierr);
  } else {
    ierr = PetscInfo(ptap->AP_loc,"Scalable algorithm, AP_loc is empty \n");CHKERRQ(ierr);
  }
#endif

  /* (2-1) compute symbolic Co = Ro*AP_loc  */
  /* ------------------------------------ */
  ierr = MatGetOptionsPrefix(A,&prefix);CHKERRQ(ierr);
  ierr = MatSetOptionsPrefix(ptap->Ro,prefix);CHKERRQ(ierr);
  ierr = MatAppendOptionsPrefix(ptap->Ro,"inner_offdiag_");CHKERRQ(ierr);
  ierr = MatMatMultSymbolic_SeqAIJ_SeqAIJ(ptap->Ro,ptap->AP_loc,fill,&ptap->C_oth);CHKERRQ(ierr);

  /* (3) send coj of C_oth to other processors  */
  /* ------------------------------------------ */
  /* determine row ownership */
  ierr = PetscLayoutCreate(comm,&rowmap);CHKERRQ(ierr);
  rowmap->n  = pn;
  rowmap->bs = 1;
  ierr   = PetscLayoutSetUp(rowmap);CHKERRQ(ierr);
  owners = rowmap->range;

  /* determine the number of messages to send, their lengths */
  ierr = PetscMalloc4(size,&len_s,size,&len_si,size,&sstatus,size+2,&owners_co);CHKERRQ(ierr);
  ierr = PetscMemzero(len_s,size*sizeof(PetscMPIInt));CHKERRQ(ierr);
  ierr = PetscMemzero(len_si,size*sizeof(PetscMPIInt));CHKERRQ(ierr);

  c_oth = (Mat_SeqAIJ*)ptap->C_oth->data;
  coi   = c_oth->i; coj = c_oth->j;
  con   = ptap->C_oth->rmap->n;
  proc  = 0;
  ierr = ISLocalToGlobalMappingApply(ptap->ltog,coi[con],coj,coj);CHKERRQ(ierr);
  for (i=0; i<con; i++) {
    while (prmap[i] >= owners[proc+1]) proc++;
    len_si[proc]++;               /* num of rows in Co(=Pt*AP) to be sent to [proc] */
    len_s[proc] += coi[i+1] - coi[i]; /* num of nonzeros in Co to be sent to [proc] */
  }

  len          = 0; /* max length of buf_si[], see (4) */
  owners_co[0] = 0;
  nsend        = 0;
  for (proc=0; proc<size; proc++) {
    owners_co[proc+1] = owners_co[proc] + len_si[proc];
    if (len_s[proc]) {
      nsend++;
      len_si[proc] = 2*(len_si[proc] + 1); /* length of buf_si to be sent to [proc] */
      len         += len_si[proc];
    }
  }

  /* determine the number and length of messages to receive for coi and coj  */
  ierr = PetscGatherNumberOfMessages(comm,NULL,len_s,&nrecv);CHKERRQ(ierr);
  ierr = PetscGatherMessageLengths2(comm,nsend,nrecv,len_s,len_si,&id_r,&len_r,&len_ri);CHKERRQ(ierr);

  /* post the Irecv and Isend of coj */
  ierr = PetscCommGetNewTag(comm,&tagj);CHKERRQ(ierr);
  ierr = PetscPostIrecvInt(comm,tagj,nrecv,id_r,len_r,&buf_rj,&rwaits);CHKERRQ(ierr);
  ierr = PetscMalloc1(nsend+1,&swaits);CHKERRQ(ierr);
  for (proc=0, k=0; proc<size; proc++) {
    if (!len_s[proc]) continue;
    i    = owners_co[proc];
    ierr = MPI_Isend(coj+coi[i],len_s[proc],MPIU_INT,proc,tagj,comm,swaits+k);CHKERRQ(ierr);
    k++;
  }

  /* (2-2) compute symbolic C_loc = Rd*AP_loc */
  /* ---------------------------------------- */
  ierr = MatSetOptionsPrefix(ptap->Rd,prefix);CHKERRQ(ierr);
  ierr = MatAppendOptionsPrefix(ptap->Rd,"inner_diag_");CHKERRQ(ierr);
  ierr = MatMatMultSymbolic_SeqAIJ_SeqAIJ(ptap->Rd,ptap->AP_loc,fill,&ptap->C_loc);CHKERRQ(ierr);
  c_loc = (Mat_SeqAIJ*)ptap->C_loc->data;
  ierr = ISLocalToGlobalMappingApply(ptap->ltog,c_loc->i[ptap->C_loc->rmap->n],c_loc->j,c_loc->j);CHKERRQ(ierr);

  /* receives coj are complete */
  for (i=0; i<nrecv; i++) {
    ierr = MPI_Waitany(nrecv,rwaits,&icompleted,&rstatus);CHKERRQ(ierr);
  }
  ierr = PetscFree(rwaits);CHKERRQ(ierr);
  if (nsend) {ierr = MPI_Waitall(nsend,swaits,sstatus);CHKERRQ(ierr);}

  /* add received column indices into ta to update Crmax */
  for (k=0; k<nrecv; k++) {/* k-th received message */
    Jptr = buf_rj[k];
    for (j=0; j<len_r[k]; j++) {
      ierr = PetscTableAdd(ta,*(Jptr+j)+1,1,INSERT_VALUES);CHKERRQ(ierr); 
    }
  }
  ierr = PetscTableGetCount(ta,&Crmax);CHKERRQ(ierr);
  ierr = PetscTableDestroy(&ta);CHKERRQ(ierr);

  /* (4) send and recv coi */
  /*-----------------------*/
  ierr   = PetscCommGetNewTag(comm,&tagi);CHKERRQ(ierr);
  ierr   = PetscPostIrecvInt(comm,tagi,nrecv,id_r,len_ri,&buf_ri,&rwaits);CHKERRQ(ierr);
  ierr   = PetscMalloc1(len+1,&buf_s);CHKERRQ(ierr);
  buf_si = buf_s;  /* points to the beginning of k-th msg to be sent */
  for (proc=0,k=0; proc<size; proc++) {
    if (!len_s[proc]) continue;
    /* form outgoing message for i-structure:
         buf_si[0]:                 nrows to be sent
               [1:nrows]:           row index (global)
               [nrows+1:2*nrows+1]: i-structure index
    */
    /*-------------------------------------------*/
    nrows       = len_si[proc]/2 - 1; /* num of rows in Co to be sent to [proc] */
    buf_si_i    = buf_si + nrows+1;
    buf_si[0]   = nrows;
    buf_si_i[0] = 0;
    nrows       = 0;
    for (i=owners_co[proc]; i<owners_co[proc+1]; i++) {
      nzi = coi[i+1] - coi[i];
      buf_si_i[nrows+1] = buf_si_i[nrows] + nzi;  /* i-structure */
      buf_si[nrows+1]   = prmap[i] -owners[proc]; /* local row index */
      nrows++;
    }
    ierr = MPI_Isend(buf_si,len_si[proc],MPIU_INT,proc,tagi,comm,swaits+k);CHKERRQ(ierr);
    k++;
    buf_si += len_si[proc];
  }
  for (i=0; i<nrecv; i++) {
    ierr = MPI_Waitany(nrecv,rwaits,&icompleted,&rstatus);CHKERRQ(ierr);
  }
  ierr = PetscFree(rwaits);CHKERRQ(ierr);
  if (nsend) {ierr = MPI_Waitall(nsend,swaits,sstatus);CHKERRQ(ierr);}

  ierr = PetscFree4(len_s,len_si,sstatus,owners_co);CHKERRQ(ierr);
  ierr = PetscFree(len_ri);CHKERRQ(ierr);
  ierr = PetscFree(swaits);CHKERRQ(ierr);
  ierr = PetscFree(buf_s);CHKERRQ(ierr);

  /* (5) compute the local portion of Cmpi      */
  /* ------------------------------------------ */
  /* set initial free space to be Crmax, sufficient for holding nozeros in each row of Cmpi */
  ierr          = PetscFreeSpaceGet(Crmax,&free_space);CHKERRQ(ierr);
  current_space = free_space;

  ierr = PetscMalloc3(nrecv,&buf_ri_k,nrecv,&nextrow,nrecv,&nextci);CHKERRQ(ierr);
  for (k=0; k<nrecv; k++) {
    buf_ri_k[k] = buf_ri[k]; /* beginning of k-th recved i-structure */
    nrows       = *buf_ri_k[k];
    nextrow[k]  = buf_ri_k[k] + 1;  /* next row number of k-th recved i-structure */
    nextci[k]   = buf_ri_k[k] + (nrows + 1); /* poins to the next i-structure of k-th recved i-structure  */
  }

  ierr = MatPreallocateInitialize(comm,pn,pn,dnz,onz);CHKERRQ(ierr);
  ierr = PetscLLCondensedCreate_Scalable(Crmax,&lnk);CHKERRQ(ierr);
  for (i=0; i<pn; i++) {
    /* add C_loc into Cmpi */
    nzi  = c_loc->i[i+1] - c_loc->i[i];
    Jptr = c_loc->j + c_loc->i[i];
    ierr = PetscLLCondensedAddSorted_Scalable(nzi,Jptr,lnk);CHKERRQ(ierr);

    /* add received col data into lnk */
    for (k=0; k<nrecv; k++) { /* k-th received message */
      if (i == *nextrow[k]) { /* i-th row */
        nzi  = *(nextci[k]+1) - *nextci[k];
        Jptr = buf_rj[k] + *nextci[k];
        ierr = PetscLLCondensedAddSorted_Scalable(nzi,Jptr,lnk);CHKERRQ(ierr);
        nextrow[k]++; nextci[k]++;
      }
    }
    nzi = lnk[0];

    /* copy data into free space, then initialize lnk */
    ierr = PetscLLCondensedClean_Scalable(nzi,current_space->array,lnk);CHKERRQ(ierr);
    ierr = MatPreallocateSet(i+owners[rank],nzi,current_space->array,dnz,onz);CHKERRQ(ierr);
  }
  ierr = PetscFree3(buf_ri_k,nextrow,nextci);CHKERRQ(ierr);
  ierr = PetscLLCondensedDestroy_Scalable(lnk);CHKERRQ(ierr);
  ierr = PetscFreeSpaceDestroy(free_space);CHKERRQ(ierr);

  /* local sizes and preallocation */
  ierr = MatSetSizes(Cmpi,pn,pn,PETSC_DETERMINE,PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = MatSetBlockSizes(Cmpi,PetscAbs(P->cmap->bs),PetscAbs(P->cmap->bs));CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(Cmpi,0,dnz,0,onz);CHKERRQ(ierr);
  ierr = MatPreallocateFinalize(dnz,onz);CHKERRQ(ierr);

  /* members in merge */
  ierr = PetscFree(id_r);CHKERRQ(ierr);
  ierr = PetscFree(len_r);CHKERRQ(ierr);
  ierr = PetscFree(buf_ri[0]);CHKERRQ(ierr);
  ierr = PetscFree(buf_ri);CHKERRQ(ierr);
  ierr = PetscFree(buf_rj[0]);CHKERRQ(ierr);
  ierr = PetscFree(buf_rj);CHKERRQ(ierr);
  ierr = PetscLayoutDestroy(&rowmap);CHKERRQ(ierr);

  /* attach the supporting struct to Cmpi for reuse */
  c = (Mat_MPIAIJ*)Cmpi->data;
  c->ap           = ptap;
  ptap->duplicate = Cmpi->ops->duplicate;
  ptap->destroy   = Cmpi->ops->destroy;
  ptap->view      = Cmpi->ops->view;

  /* Cmpi is not ready for use - assembly will be done by MatPtAPNumeric() */
  Cmpi->assembled        = PETSC_FALSE;
  Cmpi->ops->ptapnumeric = MatPtAPNumeric_MPIAIJ_MPIAIJ_scalable;
  Cmpi->ops->destroy     = MatDestroy_MPIAIJ_PtAP;
  Cmpi->ops->view        = MatView_MPIAIJ_PtAP;
  Cmpi->ops->freeintermediatedatastructures = MatFreeIntermediateDataStructures_MPIAIJ_AP;
  *C                     = Cmpi;

   nout = 0;
   ierr = ISGlobalToLocalMappingApply(ptap->ltog,IS_GTOLM_DROP,c_oth->i[ptap->C_oth->rmap->n],c_oth->j,&nout,c_oth->j);CHKERRQ(ierr);
   if (c_oth->i[ptap->C_oth->rmap->n] != nout) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Incorrect mapping %D != %D\n",c_oth->i[ptap->C_oth->rmap->n],nout);
   ierr = ISGlobalToLocalMappingApply(ptap->ltog,IS_GTOLM_DROP,c_loc->i[ptap->C_loc->rmap->n],c_loc->j,&nout,c_loc->j);CHKERRQ(ierr);
   if (c_loc->i[ptap->C_loc->rmap->n] != nout) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Incorrect mapping %D != %D\n",c_loc->i[ptap->C_loc->rmap->n],nout);

  PetscFunctionReturn(0);
}

PetscErrorCode MatPtAPSymbolic_MPIAIJ_MPIAIJ(Mat A,Mat P,PetscReal fill,Mat *C)
{
  PetscErrorCode      ierr;
  Mat_APMPI           *ptap;
  Mat_MPIAIJ          *a=(Mat_MPIAIJ*)A->data,*p=(Mat_MPIAIJ*)P->data,*c;
  MPI_Comm            comm;
  PetscMPIInt         size,rank;
  Mat                 Cmpi;
  PetscFreeSpaceList  free_space=NULL,current_space=NULL;
  PetscInt            am=A->rmap->n,pm=P->rmap->n,pN=P->cmap->N,pn=P->cmap->n;
  PetscInt            *lnk,i,k,pnz,row,nsend;
  PetscBT             lnkbt;
  PetscMPIInt         tagi,tagj,*len_si,*len_s,*len_ri,icompleted=0,nrecv;
  PetscInt            **buf_rj,**buf_ri,**buf_ri_k;
  PetscInt            len,proc,*dnz,*onz,*owners,nzi,nspacedouble;
  PetscInt            nrows,*buf_s,*buf_si,*buf_si_i,**nextrow,**nextci;
  MPI_Request         *swaits,*rwaits;
  MPI_Status          *sstatus,rstatus;
  PetscLayout         rowmap;
  PetscInt            *owners_co,*coi,*coj;    /* i and j array of (p->B)^T*A*P - used in the communication */
  PetscMPIInt         *len_r,*id_r;    /* array of length of comm->size, store send/recv matrix values */
  PetscInt            *api,*apj,*Jptr,apnz,*prmap=p->garray,con,j,ap_rmax=0,Crmax,*aj,*ai,*pi;
  Mat_SeqAIJ          *p_loc,*p_oth=NULL,*ad=(Mat_SeqAIJ*)(a->A)->data,*ao=NULL,*c_loc,*c_oth;
  PetscScalar         *apv;
  PetscTable          ta;
  MatType             mtype;
  const char          *prefix;
#if defined(PETSC_USE_INFO)
  PetscReal           apfill;
#endif

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)A,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);

  if (size > 1) ao = (Mat_SeqAIJ*)(a->B)->data;

  /* create symbolic parallel matrix Cmpi */
  ierr = MatCreate(comm,&Cmpi);CHKERRQ(ierr);
  ierr = MatGetType(A,&mtype);CHKERRQ(ierr);
  ierr = MatSetType(Cmpi,mtype);CHKERRQ(ierr);

  /* Do dense axpy in MatPtAPNumeric_MPIAIJ_MPIAIJ() */
  Cmpi->ops->ptapnumeric = MatPtAPNumeric_MPIAIJ_MPIAIJ;

  /* create struct Mat_APMPI and attached it to C later */
  ierr        = PetscNew(&ptap);CHKERRQ(ierr);
  ptap->reuse = MAT_INITIAL_MATRIX;
  ptap->algType = 1;

  /* get P_oth by taking rows of P (= non-zero cols of local A) from other processors */
  ierr = MatGetBrowsOfAoCols_MPIAIJ(A,P,MAT_INITIAL_MATRIX,&ptap->startsj_s,&ptap->startsj_r,&ptap->bufa,&ptap->P_oth);CHKERRQ(ierr);
  /* get P_loc by taking all local rows of P */
  ierr = MatMPIAIJGetLocalMat(P,MAT_INITIAL_MATRIX,&ptap->P_loc);CHKERRQ(ierr);

  /* (0) compute Rd = Pd^T, Ro = Po^T  */
  /* --------------------------------- */
  ierr = MatTranspose(p->A,MAT_INITIAL_MATRIX,&ptap->Rd);CHKERRQ(ierr);
  ierr = MatTranspose(p->B,MAT_INITIAL_MATRIX,&ptap->Ro);CHKERRQ(ierr);

  /* (1) compute symbolic AP = A_loc*P = Ad*P_loc + Ao*P_oth (api,apj) */
  /* ----------------------------------------------------------------- */
  p_loc  = (Mat_SeqAIJ*)(ptap->P_loc)->data;
  if (ptap->P_oth) p_oth  = (Mat_SeqAIJ*)(ptap->P_oth)->data;

  /* create and initialize a linked list */
  ierr = PetscTableCreate(pn,pN,&ta);CHKERRQ(ierr); /* for compute AP_loc and Cmpi */
  MatRowMergeMax_SeqAIJ(p_loc,ptap->P_loc->rmap->N,ta);
  MatRowMergeMax_SeqAIJ(p_oth,ptap->P_oth->rmap->N,ta);
  ierr = PetscTableGetCount(ta,&Crmax);CHKERRQ(ierr); /* Crmax = nnz(sum of Prows) */
  /* printf("[%d] est %d, Crmax %d; pN %d\n",rank,5*(p_loc->rmax+p_oth->rmax + (PetscInt)(1.e-2*pN)),Crmax,pN); */

  ierr = PetscLLCondensedCreate(Crmax,pN,&lnk,&lnkbt);CHKERRQ(ierr);

  /* Initial FreeSpace size is fill*(nnz(A) + nnz(P)) */
  if (ao) {
    ierr = PetscFreeSpaceGet(PetscRealIntMultTruncate(fill,PetscIntSumTruncate(ad->i[am],PetscIntSumTruncate(ao->i[am],p_loc->i[pm]))),&free_space);CHKERRQ(ierr);
  } else {
    ierr = PetscFreeSpaceGet(PetscRealIntMultTruncate(fill,PetscIntSumTruncate(ad->i[am],p_loc->i[pm])),&free_space);CHKERRQ(ierr);
  }
  current_space = free_space;
  nspacedouble  = 0;

  ierr   = PetscMalloc1(am+1,&api);CHKERRQ(ierr);
  api[0] = 0;
  for (i=0; i<am; i++) {
    /* diagonal portion: Ad[i,:]*P */
    ai = ad->i; pi = p_loc->i;
    nzi = ai[i+1] - ai[i];
    aj  = ad->j + ai[i];
    for (j=0; j<nzi; j++) {
      row  = aj[j];
      pnz  = pi[row+1] - pi[row];
      Jptr = p_loc->j + pi[row];
      /* add non-zero cols of P into the sorted linked list lnk */
      ierr = PetscLLCondensedAddSorted(pnz,Jptr,lnk,lnkbt);CHKERRQ(ierr);
    }
    /* off-diagonal portion: Ao[i,:]*P */
    if (ao) {
      ai = ao->i; pi = p_oth->i;
      nzi = ai[i+1] - ai[i];
      aj  = ao->j + ai[i];
      for (j=0; j<nzi; j++) {
        row  = aj[j];
        pnz  = pi[row+1] - pi[row];
        Jptr = p_oth->j + pi[row];
        ierr = PetscLLCondensedAddSorted(pnz,Jptr,lnk,lnkbt);CHKERRQ(ierr);
      }
    }
    apnz     = lnk[0];
    api[i+1] = api[i] + apnz;
    if (ap_rmax < apnz) ap_rmax = apnz;

    /* if free space is not available, double the total space in the list */
    if (current_space->local_remaining<apnz) {
      ierr = PetscFreeSpaceGet(PetscIntSumTruncate(apnz,current_space->total_array_size),&current_space);CHKERRQ(ierr);
      nspacedouble++;
    }

    /* Copy data into free space, then initialize lnk */
    ierr = PetscLLCondensedClean(pN,apnz,current_space->array,lnk,lnkbt);CHKERRQ(ierr);

    current_space->array           += apnz;
    current_space->local_used      += apnz;
    current_space->local_remaining -= apnz;
  }
  /* Allocate space for apj and apv, initialize apj, and */
  /* destroy list of free space and other temporary array(s) */
  ierr   = PetscMalloc2(api[am],&apj,api[am],&apv);CHKERRQ(ierr);
  ierr   = PetscFreeSpaceContiguous(&free_space,apj);CHKERRQ(ierr);
  ierr = PetscLLDestroy(lnk,lnkbt);CHKERRQ(ierr);

  /* Create AP_loc for reuse */
  ierr = MatCreateSeqAIJWithArrays(PETSC_COMM_SELF,am,pN,api,apj,apv,&ptap->AP_loc);CHKERRQ(ierr);

#if defined(PETSC_USE_INFO)
  if (ao) {
    apfill = (PetscReal)api[am]/(ad->i[am]+ao->i[am]+p_loc->i[pm]+1);
  } else {
    apfill = (PetscReal)api[am]/(ad->i[am]+p_loc->i[pm]+1);
  }
  ptap->AP_loc->info.mallocs           = nspacedouble;
  ptap->AP_loc->info.fill_ratio_given  = fill;
  ptap->AP_loc->info.fill_ratio_needed = apfill;

  if (api[am]) {
    ierr = PetscInfo3(ptap->AP_loc,"Nonscalable algorithm, AP_loc reallocs %D; Fill ratio: given %g needed %g.\n",nspacedouble,(double)fill,(double)apfill);CHKERRQ(ierr);
    ierr = PetscInfo1(ptap->AP_loc,"Use MatPtAP(A,B,MatReuse,%g,&C) for best AP_loc performance.;\n",(double)apfill);CHKERRQ(ierr);
  } else {
    ierr = PetscInfo(ptap->AP_loc,"Nonscalable algorithm, AP_loc is empty \n");CHKERRQ(ierr);
  }
#endif

  /* (2-1) compute symbolic Co = Ro*AP_loc  */
  /* ------------------------------------ */
  ierr = MatGetOptionsPrefix(A,&prefix);CHKERRQ(ierr);
  ierr = MatSetOptionsPrefix(ptap->Ro,prefix);CHKERRQ(ierr);
  ierr = MatAppendOptionsPrefix(ptap->Ro,"inner_offdiag_");CHKERRQ(ierr);
  ierr = MatMatMultSymbolic_SeqAIJ_SeqAIJ(ptap->Ro,ptap->AP_loc,fill,&ptap->C_oth);CHKERRQ(ierr);

  /* (3) send coj of C_oth to other processors  */
  /* ------------------------------------------ */
  /* determine row ownership */
  ierr = PetscLayoutCreate(comm,&rowmap);CHKERRQ(ierr);
  rowmap->n  = pn;
  rowmap->bs = 1;
  ierr   = PetscLayoutSetUp(rowmap);CHKERRQ(ierr);
  owners = rowmap->range;

  /* determine the number of messages to send, their lengths */
  ierr = PetscMalloc4(size,&len_s,size,&len_si,size,&sstatus,size+2,&owners_co);CHKERRQ(ierr);
  ierr = PetscMemzero(len_s,size*sizeof(PetscMPIInt));CHKERRQ(ierr);
  ierr = PetscMemzero(len_si,size*sizeof(PetscMPIInt));CHKERRQ(ierr);

  c_oth = (Mat_SeqAIJ*)ptap->C_oth->data;
  coi   = c_oth->i; coj = c_oth->j;
  con   = ptap->C_oth->rmap->n;
  proc  = 0;
  for (i=0; i<con; i++) {
    while (prmap[i] >= owners[proc+1]) proc++;
    len_si[proc]++;               /* num of rows in Co(=Pt*AP) to be sent to [proc] */
    len_s[proc] += coi[i+1] - coi[i]; /* num of nonzeros in Co to be sent to [proc] */
  }

  len          = 0; /* max length of buf_si[], see (4) */
  owners_co[0] = 0;
  nsend        = 0;
  for (proc=0; proc<size; proc++) {
    owners_co[proc+1] = owners_co[proc] + len_si[proc];
    if (len_s[proc]) {
      nsend++;
      len_si[proc] = 2*(len_si[proc] + 1); /* length of buf_si to be sent to [proc] */
      len         += len_si[proc];
    }
  }

  /* determine the number and length of messages to receive for coi and coj  */
  ierr = PetscGatherNumberOfMessages(comm,NULL,len_s,&nrecv);CHKERRQ(ierr);
  ierr = PetscGatherMessageLengths2(comm,nsend,nrecv,len_s,len_si,&id_r,&len_r,&len_ri);CHKERRQ(ierr);

  /* post the Irecv and Isend of coj */
  ierr = PetscCommGetNewTag(comm,&tagj);CHKERRQ(ierr);
  ierr = PetscPostIrecvInt(comm,tagj,nrecv,id_r,len_r,&buf_rj,&rwaits);CHKERRQ(ierr);
  ierr = PetscMalloc1(nsend+1,&swaits);CHKERRQ(ierr);
  for (proc=0, k=0; proc<size; proc++) {
    if (!len_s[proc]) continue;
    i    = owners_co[proc];
    ierr = MPI_Isend(coj+coi[i],len_s[proc],MPIU_INT,proc,tagj,comm,swaits+k);CHKERRQ(ierr);
    k++;
  }

  /* (2-2) compute symbolic C_loc = Rd*AP_loc */
  /* ---------------------------------------- */
  ierr = MatSetOptionsPrefix(ptap->Rd,prefix);CHKERRQ(ierr);
  ierr = MatAppendOptionsPrefix(ptap->Rd,"inner_diag_");CHKERRQ(ierr);
  ierr = MatMatMultSymbolic_SeqAIJ_SeqAIJ(ptap->Rd,ptap->AP_loc,fill,&ptap->C_loc);CHKERRQ(ierr);
  c_loc = (Mat_SeqAIJ*)ptap->C_loc->data;

  /* receives coj are complete */
  for (i=0; i<nrecv; i++) {
    ierr = MPI_Waitany(nrecv,rwaits,&icompleted,&rstatus);CHKERRQ(ierr);
  }
  ierr = PetscFree(rwaits);CHKERRQ(ierr);
  if (nsend) {ierr = MPI_Waitall(nsend,swaits,sstatus);CHKERRQ(ierr);}

  /* add received column indices into ta to update Crmax */
  for (k=0; k<nrecv; k++) {/* k-th received message */
    Jptr = buf_rj[k];
    for (j=0; j<len_r[k]; j++) {
      ierr = PetscTableAdd(ta,*(Jptr+j)+1,1,INSERT_VALUES);CHKERRQ(ierr); 
    }
  }
  ierr = PetscTableGetCount(ta,&Crmax);CHKERRQ(ierr);
  ierr = PetscTableDestroy(&ta);CHKERRQ(ierr);

  /* (4) send and recv coi */
  /*-----------------------*/
  ierr   = PetscCommGetNewTag(comm,&tagi);CHKERRQ(ierr);
  ierr   = PetscPostIrecvInt(comm,tagi,nrecv,id_r,len_ri,&buf_ri,&rwaits);CHKERRQ(ierr);
  ierr   = PetscMalloc1(len+1,&buf_s);CHKERRQ(ierr);
  buf_si = buf_s;  /* points to the beginning of k-th msg to be sent */
  for (proc=0,k=0; proc<size; proc++) {
    if (!len_s[proc]) continue;
    /* form outgoing message for i-structure:
         buf_si[0]:                 nrows to be sent
               [1:nrows]:           row index (global)
               [nrows+1:2*nrows+1]: i-structure index
    */
    /*-------------------------------------------*/
    nrows       = len_si[proc]/2 - 1; /* num of rows in Co to be sent to [proc] */
    buf_si_i    = buf_si + nrows+1;
    buf_si[0]   = nrows;
    buf_si_i[0] = 0;
    nrows       = 0;
    for (i=owners_co[proc]; i<owners_co[proc+1]; i++) {
      nzi = coi[i+1] - coi[i];
      buf_si_i[nrows+1] = buf_si_i[nrows] + nzi;  /* i-structure */
      buf_si[nrows+1]   = prmap[i] -owners[proc]; /* local row index */
      nrows++;
    }
    ierr = MPI_Isend(buf_si,len_si[proc],MPIU_INT,proc,tagi,comm,swaits+k);CHKERRQ(ierr);
    k++;
    buf_si += len_si[proc];
  }
  for (i=0; i<nrecv; i++) {
    ierr = MPI_Waitany(nrecv,rwaits,&icompleted,&rstatus);CHKERRQ(ierr);
  }
  ierr = PetscFree(rwaits);CHKERRQ(ierr);
  if (nsend) {ierr = MPI_Waitall(nsend,swaits,sstatus);CHKERRQ(ierr);}

  ierr = PetscFree4(len_s,len_si,sstatus,owners_co);CHKERRQ(ierr);
  ierr = PetscFree(len_ri);CHKERRQ(ierr);
  ierr = PetscFree(swaits);CHKERRQ(ierr);
  ierr = PetscFree(buf_s);CHKERRQ(ierr);

  /* (5) compute the local portion of Cmpi      */
  /* ------------------------------------------ */
  /* set initial free space to be Crmax, sufficient for holding nozeros in each row of Cmpi */
  ierr          = PetscFreeSpaceGet(Crmax,&free_space);CHKERRQ(ierr);
  current_space = free_space;

  ierr = PetscMalloc3(nrecv,&buf_ri_k,nrecv,&nextrow,nrecv,&nextci);CHKERRQ(ierr);
  for (k=0; k<nrecv; k++) {
    buf_ri_k[k] = buf_ri[k]; /* beginning of k-th recved i-structure */
    nrows       = *buf_ri_k[k];
    nextrow[k]  = buf_ri_k[k] + 1;  /* next row number of k-th recved i-structure */
    nextci[k]   = buf_ri_k[k] + (nrows + 1); /* poins to the next i-structure of k-th recved i-structure  */
  }

  ierr = MatPreallocateInitialize(comm,pn,pn,dnz,onz);CHKERRQ(ierr);
  ierr = PetscLLCondensedCreate(Crmax,pN,&lnk,&lnkbt);CHKERRQ(ierr);
  for (i=0; i<pn; i++) {
    /* add C_loc into Cmpi */
    nzi  = c_loc->i[i+1] - c_loc->i[i];
    Jptr = c_loc->j + c_loc->i[i];
    ierr = PetscLLCondensedAddSorted(nzi,Jptr,lnk,lnkbt);CHKERRQ(ierr);

    /* add received col data into lnk */
    for (k=0; k<nrecv; k++) { /* k-th received message */
      if (i == *nextrow[k]) { /* i-th row */
        nzi  = *(nextci[k]+1) - *nextci[k];
        Jptr = buf_rj[k] + *nextci[k];
        ierr = PetscLLCondensedAddSorted(nzi,Jptr,lnk,lnkbt);CHKERRQ(ierr);
        nextrow[k]++; nextci[k]++;
      }
    }
    nzi = lnk[0];

    /* copy data into free space, then initialize lnk */
    ierr = PetscLLCondensedClean(pN,nzi,current_space->array,lnk,lnkbt);CHKERRQ(ierr);
    ierr = MatPreallocateSet(i+owners[rank],nzi,current_space->array,dnz,onz);CHKERRQ(ierr);
  }
  ierr = PetscFree3(buf_ri_k,nextrow,nextci);CHKERRQ(ierr);
  ierr = PetscLLDestroy(lnk,lnkbt);CHKERRQ(ierr);
  ierr = PetscFreeSpaceDestroy(free_space);CHKERRQ(ierr);

  /* local sizes and preallocation */
  ierr = MatSetSizes(Cmpi,pn,pn,PETSC_DETERMINE,PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = MatSetBlockSizes(Cmpi,PetscAbs(P->cmap->bs),PetscAbs(P->cmap->bs));CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(Cmpi,0,dnz,0,onz);CHKERRQ(ierr);
  ierr = MatPreallocateFinalize(dnz,onz);CHKERRQ(ierr);

  /* members in merge */
  ierr = PetscFree(id_r);CHKERRQ(ierr);
  ierr = PetscFree(len_r);CHKERRQ(ierr);
  ierr = PetscFree(buf_ri[0]);CHKERRQ(ierr);
  ierr = PetscFree(buf_ri);CHKERRQ(ierr);
  ierr = PetscFree(buf_rj[0]);CHKERRQ(ierr);
  ierr = PetscFree(buf_rj);CHKERRQ(ierr);
  ierr = PetscLayoutDestroy(&rowmap);CHKERRQ(ierr);

  /* attach the supporting struct to Cmpi for reuse */
  c = (Mat_MPIAIJ*)Cmpi->data;
  c->ap           = ptap;
  ptap->duplicate = Cmpi->ops->duplicate;
  ptap->destroy   = Cmpi->ops->destroy;
  ptap->view      = Cmpi->ops->view;
  ierr = PetscCalloc1(pN,&ptap->apa);CHKERRQ(ierr);

  /* Cmpi is not ready for use - assembly will be done by MatPtAPNumeric() */
  Cmpi->assembled        = PETSC_FALSE;
  Cmpi->ops->destroy     = MatDestroy_MPIAIJ_PtAP;
  Cmpi->ops->view        = MatView_MPIAIJ_PtAP;
  Cmpi->ops->freeintermediatedatastructures = MatFreeIntermediateDataStructures_MPIAIJ_AP;
  *C                     = Cmpi;
  PetscFunctionReturn(0);
}

PetscErrorCode MatPtAPNumeric_MPIAIJ_MPIAIJ(Mat A,Mat P,Mat C)
{
  PetscErrorCode    ierr;
  Mat_MPIAIJ        *a=(Mat_MPIAIJ*)A->data,*p=(Mat_MPIAIJ*)P->data,*c=(Mat_MPIAIJ*)C->data;
  Mat_SeqAIJ        *ad=(Mat_SeqAIJ*)(a->A)->data,*ao=(Mat_SeqAIJ*)(a->B)->data;
  Mat_SeqAIJ        *ap,*p_loc,*p_oth=NULL,*c_seq;
  Mat_APMPI         *ptap = c->ap;
  Mat               AP_loc,C_loc,C_oth;
  PetscInt          i,rstart,rend,cm,ncols,row;
  PetscInt          *api,*apj,am = A->rmap->n,j,col,apnz;
  PetscScalar       *apa;
  const PetscInt    *cols;
  const PetscScalar *vals;

  PetscFunctionBegin;
  if (!ptap) {
    MPI_Comm comm;
    ierr = PetscObjectGetComm((PetscObject)C,&comm);CHKERRQ(ierr);
    SETERRQ(comm,PETSC_ERR_ARG_WRONGSTATE,"PtAP cannot be reused. Do not call MatFreeIntermediateDataStructures() or use '-mat_freeintermediatedatastructures'");
  }

  ierr = MatZeroEntries(C);CHKERRQ(ierr);
  /* 1) get R = Pd^T,Ro = Po^T */
  if (ptap->reuse == MAT_REUSE_MATRIX) {
    ierr = MatTranspose(p->A,MAT_REUSE_MATRIX,&ptap->Rd);CHKERRQ(ierr);
    ierr = MatTranspose(p->B,MAT_REUSE_MATRIX,&ptap->Ro);CHKERRQ(ierr);
  }

  /* 2) get AP_loc */
  AP_loc = ptap->AP_loc;
  ap = (Mat_SeqAIJ*)AP_loc->data;

  /* 2-1) get P_oth = ptap->P_oth  and P_loc = ptap->P_loc */
  /*-----------------------------------------------------*/
  if (ptap->reuse == MAT_REUSE_MATRIX) {
    /* P_oth and P_loc are obtained in MatPtASymbolic() when reuse == MAT_INITIAL_MATRIX */
    ierr = MatGetBrowsOfAoCols_MPIAIJ(A,P,MAT_REUSE_MATRIX,&ptap->startsj_s,&ptap->startsj_r,&ptap->bufa,&ptap->P_oth);CHKERRQ(ierr);
    ierr = MatMPIAIJGetLocalMat(P,MAT_REUSE_MATRIX,&ptap->P_loc);CHKERRQ(ierr);
  }

  /* 2-2) compute numeric A_loc*P - dominating part */
  /* ---------------------------------------------- */
  /* get data from symbolic products */
  p_loc = (Mat_SeqAIJ*)(ptap->P_loc)->data;
  if (ptap->P_oth) {
    p_oth = (Mat_SeqAIJ*)(ptap->P_oth)->data;
  }
  apa   = ptap->apa;
  api   = ap->i;
  apj   = ap->j;
  for (i=0; i<am; i++) {
    /* AP[i,:] = A[i,:]*P = Ad*P_loc Ao*P_oth */
    AProw_nonscalable(i,ad,ao,p_loc,p_oth,apa);
    apnz = api[i+1] - api[i];
    for (j=0; j<apnz; j++) {
      col = apj[j+api[i]];
      ap->a[j+ap->i[i]] = apa[col];
      apa[col] = 0.0;
    }
    ierr = PetscLogFlops(2.0*apnz);CHKERRQ(ierr);
  }

  /* 3) C_loc = Rd*AP_loc, C_oth = Ro*AP_loc */
  ierr = ((ptap->C_loc)->ops->matmultnumeric)(ptap->Rd,AP_loc,ptap->C_loc);CHKERRQ(ierr);
  ierr = ((ptap->C_oth)->ops->matmultnumeric)(ptap->Ro,AP_loc,ptap->C_oth);CHKERRQ(ierr);
  C_loc = ptap->C_loc;
  C_oth = ptap->C_oth;

  /* add C_loc and Co to to C */
  ierr = MatGetOwnershipRange(C,&rstart,&rend);CHKERRQ(ierr);

  /* C_loc -> C */
  cm    = C_loc->rmap->N;
  c_seq = (Mat_SeqAIJ*)C_loc->data;
  cols = c_seq->j;
  vals = c_seq->a;


  /* The (fast) MatSetValues_MPIAIJ_CopyFromCSRFormat function can only be used when C->was_assembled is PETSC_FALSE and */
  /* when there are no off-processor parts.  */
  /* If was_assembled is true, then the statement aj[rowstart_diag+dnz_row] = mat_j[col] - cstart; in MatSetValues_MPIAIJ_CopyFromCSRFormat */
  /* is no longer true. Then the more complex function MatSetValues_MPIAIJ() has to be used, where the column index is looked up from */
  /* a table, and other, more complex stuff has to be done. */
  if (C->assembled) {
    C->was_assembled = PETSC_TRUE;
    C->assembled     = PETSC_FALSE;
  }
  if (C->was_assembled) {
    for (i=0; i<cm; i++) {
      ncols = c_seq->i[i+1] - c_seq->i[i];
      row = rstart + i;
      ierr = MatSetValues_MPIAIJ(C,1,&row,ncols,cols,vals,ADD_VALUES);CHKERRQ(ierr);
      cols += ncols; vals += ncols;
    }
  } else {
    ierr = MatSetValues_MPIAIJ_CopyFromCSRFormat(C,c_seq->j,c_seq->i,c_seq->a);CHKERRQ(ierr);
  }

  /* Co -> C, off-processor part */
  cm = C_oth->rmap->N;
  c_seq = (Mat_SeqAIJ*)C_oth->data;
  cols = c_seq->j;
  vals = c_seq->a;
  for (i=0; i<cm; i++) {
    ncols = c_seq->i[i+1] - c_seq->i[i];
    row = p->garray[i];
    ierr = MatSetValues(C,1,&row,ncols,cols,vals,ADD_VALUES);CHKERRQ(ierr);
    cols += ncols; vals += ncols;
  }

  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ptap->reuse = MAT_REUSE_MATRIX;

  /* supporting struct ptap consumes almost same amount of memory as C=PtAP, release it if C will not be updated by A and P */
  if (ptap->freestruct) {
    ierr = MatFreeIntermediateDataStructures(C);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
