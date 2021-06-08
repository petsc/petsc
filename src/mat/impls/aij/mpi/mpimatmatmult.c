
/*
  Defines matrix-matrix product routines for pairs of MPIAIJ matrices
          C = A * B
*/
#include <../src/mat/impls/aij/seq/aij.h> /*I "petscmat.h" I*/
#include <../src/mat/utils/freespace.h>
#include <../src/mat/impls/aij/mpi/mpiaij.h>
#include <petscbt.h>
#include <../src/mat/impls/dense/mpi/mpidense.h>
#include <petsc/private/vecimpl.h>
#include <petsc/private/sfimpl.h>

#if defined(PETSC_HAVE_HYPRE)
PETSC_INTERN PetscErrorCode MatMatMultSymbolic_AIJ_AIJ_wHYPRE(Mat,Mat,PetscReal,Mat);
#endif

PETSC_INTERN PetscErrorCode MatProductSymbolic_AB_MPIAIJ_MPIAIJ(Mat C)
{
  PetscErrorCode      ierr;
  Mat_Product         *product = C->product;
  Mat                 A=product->A,B=product->B;
  MatProductAlgorithm alg=product->alg;
  PetscReal           fill=product->fill;
  PetscBool           flg;

  PetscFunctionBegin;
  /* scalable */
  ierr = PetscStrcmp(alg,"scalable",&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = MatMatMultSymbolic_MPIAIJ_MPIAIJ(A,B,fill,C);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  /* nonscalable */
  ierr = PetscStrcmp(alg,"nonscalable",&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = MatMatMultSymbolic_MPIAIJ_MPIAIJ_nonscalable(A,B,fill,C);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  /* seqmpi */
  ierr = PetscStrcmp(alg,"seqmpi",&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = MatMatMultSymbolic_MPIAIJ_MPIAIJ_seqMPI(A,B,fill,C);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  /* backend general code */
  ierr = PetscStrcmp(alg,"backend",&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = MatProductSymbolic_MPIAIJBACKEND(C);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

#if defined(PETSC_HAVE_HYPRE)
  ierr = PetscStrcmp(alg,"hypre",&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = MatMatMultSymbolic_AIJ_AIJ_wHYPRE(A,B,fill,C);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
#endif
  SETERRQ(PetscObjectComm((PetscObject)C),PETSC_ERR_SUP,"Mat Product Algorithm is not supported");
}

PetscErrorCode MatDestroy_MPIAIJ_MatMatMult(void *data)
{
  PetscErrorCode ierr;
  Mat_APMPI      *ptap = (Mat_APMPI*)data;

  PetscFunctionBegin;
  ierr = PetscFree2(ptap->startsj_s,ptap->startsj_r);CHKERRQ(ierr);
  ierr = PetscFree(ptap->bufa);CHKERRQ(ierr);
  ierr = MatDestroy(&ptap->P_loc);CHKERRQ(ierr);
  ierr = MatDestroy(&ptap->P_oth);CHKERRQ(ierr);
  ierr = MatDestroy(&ptap->Pt);CHKERRQ(ierr);
  ierr = PetscFree(ptap->api);CHKERRQ(ierr);
  ierr = PetscFree(ptap->apj);CHKERRQ(ierr);
  ierr = PetscFree(ptap->apa);CHKERRQ(ierr);
  ierr = PetscFree(ptap);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatMatMultNumeric_MPIAIJ_MPIAIJ_nonscalable(Mat A,Mat P,Mat C)
{
  PetscErrorCode    ierr;
  Mat_MPIAIJ        *a  =(Mat_MPIAIJ*)A->data,*c=(Mat_MPIAIJ*)C->data;
  Mat_SeqAIJ        *ad =(Mat_SeqAIJ*)(a->A)->data,*ao=(Mat_SeqAIJ*)(a->B)->data;
  Mat_SeqAIJ        *cd =(Mat_SeqAIJ*)(c->A)->data,*co=(Mat_SeqAIJ*)(c->B)->data;
  PetscScalar       *cda=cd->a,*coa=co->a;
  Mat_SeqAIJ        *p_loc,*p_oth;
  PetscScalar       *apa,*ca;
  PetscInt          cm =C->rmap->n;
  Mat_APMPI         *ptap;
  PetscInt          *api,*apj,*apJ,i,k;
  PetscInt          cstart=C->cmap->rstart;
  PetscInt          cdnz,conz,k0,k1;
  const PetscScalar *dummy;
  MPI_Comm          comm;
  PetscMPIInt       size;

  PetscFunctionBegin;
  MatCheckProduct(C,3);
  ptap = (Mat_APMPI*)C->product->data;
  if (!ptap) SETERRQ(PetscObjectComm((PetscObject)C),PETSC_ERR_ARG_WRONGSTATE,"PtAP cannot be computed. Missing data");
  ierr = PetscObjectGetComm((PetscObject)A,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRMPI(ierr);

  if (!ptap->P_oth && size>1) SETERRQ(PetscObjectComm((PetscObject)C),PETSC_ERR_ARG_WRONGSTATE,"AP cannot be reused. Do not call MatProductClear()");

  /* 1) get P_oth = ptap->P_oth  and P_loc = ptap->P_loc */
  /*-----------------------------------------------------*/
  /* update numerical values of P_oth and P_loc */
  ierr = MatGetBrowsOfAoCols_MPIAIJ(A,P,MAT_REUSE_MATRIX,&ptap->startsj_s,&ptap->startsj_r,&ptap->bufa,&ptap->P_oth);CHKERRQ(ierr);
  ierr = MatMPIAIJGetLocalMat(P,MAT_REUSE_MATRIX,&ptap->P_loc);CHKERRQ(ierr);

  /* 2) compute numeric C_loc = A_loc*P = Ad*P_loc + Ao*P_oth */
  /*----------------------------------------------------------*/
  /* get data from symbolic products */
  p_loc = (Mat_SeqAIJ*)(ptap->P_loc)->data;
  p_oth = NULL;
  if (size >1) {
    p_oth = (Mat_SeqAIJ*)(ptap->P_oth)->data;
  }

  /* get apa for storing dense row A[i,:]*P */
  apa = ptap->apa;

  api = ptap->api;
  apj = ptap->apj;
  /* trigger copy to CPU */
  ierr = MatSeqAIJGetArrayRead(a->A,&dummy);CHKERRQ(ierr);
  ierr = MatSeqAIJRestoreArrayRead(a->A,&dummy);CHKERRQ(ierr);
  ierr = MatSeqAIJGetArrayRead(a->B,&dummy);CHKERRQ(ierr);
  ierr = MatSeqAIJRestoreArrayRead(a->B,&dummy);CHKERRQ(ierr);
  for (i=0; i<cm; i++) {
    /* compute apa = A[i,:]*P */
    AProw_nonscalable(i,ad,ao,p_loc,p_oth,apa);

    /* set values in C */
    apJ  = apj + api[i];
    cdnz = cd->i[i+1] - cd->i[i];
    conz = co->i[i+1] - co->i[i];

    /* 1st off-diagonal part of C */
    ca = coa + co->i[i];
    k  = 0;
    for (k0=0; k0<conz; k0++) {
      if (apJ[k] >= cstart) break;
      ca[k0]      = apa[apJ[k]];
      apa[apJ[k++]] = 0.0;
    }

    /* diagonal part of C */
    ca = cda + cd->i[i];
    for (k1=0; k1<cdnz; k1++) {
      ca[k1]      = apa[apJ[k]];
      apa[apJ[k++]] = 0.0;
    }

    /* 2nd off-diagonal part of C */
    ca = coa + co->i[i];
    for (; k0<conz; k0++) {
      ca[k0]      = apa[apJ[k]];
      apa[apJ[k++]] = 0.0;
    }
  }
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatMatMultSymbolic_MPIAIJ_MPIAIJ_nonscalable(Mat A,Mat P,PetscReal fill,Mat C)
{
  PetscErrorCode     ierr;
  MPI_Comm           comm;
  PetscMPIInt        size;
  Mat_APMPI          *ptap;
  PetscFreeSpaceList free_space=NULL,current_space=NULL;
  Mat_MPIAIJ         *a=(Mat_MPIAIJ*)A->data;
  Mat_SeqAIJ         *ad=(Mat_SeqAIJ*)(a->A)->data,*ao=(Mat_SeqAIJ*)(a->B)->data,*p_loc,*p_oth;
  PetscInt           *pi_loc,*pj_loc,*pi_oth,*pj_oth,*dnz,*onz;
  PetscInt           *adi=ad->i,*adj=ad->j,*aoi=ao->i,*aoj=ao->j,rstart=A->rmap->rstart;
  PetscInt           *lnk,i,pnz,row,*api,*apj,*Jptr,apnz,nspacedouble=0,j,nzi;
  PetscInt           am=A->rmap->n,pN=P->cmap->N,pn=P->cmap->n,pm=P->rmap->n;
  PetscBT            lnkbt;
  PetscReal          afill;
  MatType            mtype;

  PetscFunctionBegin;
  MatCheckProduct(C,4);
  if (C->product->data) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Extra product struct not empty");
  ierr = PetscObjectGetComm((PetscObject)A,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRMPI(ierr);

  /* create struct Mat_APMPI and attached it to C later */
  ierr = PetscNew(&ptap);CHKERRQ(ierr);

  /* get P_oth by taking rows of P (= non-zero cols of local A) from other processors */
  ierr = MatGetBrowsOfAoCols_MPIAIJ(A,P,MAT_INITIAL_MATRIX,&ptap->startsj_s,&ptap->startsj_r,&ptap->bufa,&ptap->P_oth);CHKERRQ(ierr);

  /* get P_loc by taking all local rows of P */
  ierr = MatMPIAIJGetLocalMat(P,MAT_INITIAL_MATRIX,&ptap->P_loc);CHKERRQ(ierr);

  p_loc  = (Mat_SeqAIJ*)(ptap->P_loc)->data;
  pi_loc = p_loc->i; pj_loc = p_loc->j;
  if (size > 1) {
    p_oth  = (Mat_SeqAIJ*)(ptap->P_oth)->data;
    pi_oth = p_oth->i; pj_oth = p_oth->j;
  } else {
    p_oth = NULL;
    pi_oth = NULL; pj_oth = NULL;
  }

  /* first, compute symbolic AP = A_loc*P = A_diag*P_loc + A_off*P_oth */
  /*-------------------------------------------------------------------*/
  ierr      = PetscMalloc1(am+2,&api);CHKERRQ(ierr);
  ptap->api = api;
  api[0]    = 0;

  /* create and initialize a linked list */
  ierr = PetscLLCondensedCreate(pN,pN,&lnk,&lnkbt);CHKERRQ(ierr);

  /* Initial FreeSpace size is fill*(nnz(A)+nnz(P)) */
  ierr = PetscFreeSpaceGet(PetscRealIntMultTruncate(fill,PetscIntSumTruncate(adi[am],PetscIntSumTruncate(aoi[am],pi_loc[pm]))),&free_space);CHKERRQ(ierr);
  current_space = free_space;

  ierr = MatPreallocateInitialize(comm,am,pn,dnz,onz);CHKERRQ(ierr);
  for (i=0; i<am; i++) {
    /* diagonal portion of A */
    nzi = adi[i+1] - adi[i];
    for (j=0; j<nzi; j++) {
      row  = *adj++;
      pnz  = pi_loc[row+1] - pi_loc[row];
      Jptr = pj_loc + pi_loc[row];
      /* add non-zero cols of P into the sorted linked list lnk */
      ierr = PetscLLCondensedAddSorted(pnz,Jptr,lnk,lnkbt);CHKERRQ(ierr);
    }
    /* off-diagonal portion of A */
    nzi = aoi[i+1] - aoi[i];
    for (j=0; j<nzi; j++) {
      row  = *aoj++;
      pnz  = pi_oth[row+1] - pi_oth[row];
      Jptr = pj_oth + pi_oth[row];
      ierr = PetscLLCondensedAddSorted(pnz,Jptr,lnk,lnkbt);CHKERRQ(ierr);
    }
    /* add possible missing diagonal entry */
    if (C->force_diagonals) {
      j = i + rstart; /* column index */
      ierr = PetscLLCondensedAddSorted(1,&j,lnk,lnkbt);CHKERRQ(ierr);
    }

    apnz     = lnk[0];
    api[i+1] = api[i] + apnz;

    /* if free space is not available, double the total space in the list */
    if (current_space->local_remaining<apnz) {
      ierr = PetscFreeSpaceGet(PetscIntSumTruncate(apnz,current_space->total_array_size),&current_space);CHKERRQ(ierr);
      nspacedouble++;
    }

    /* Copy data into free space, then initialize lnk */
    ierr = PetscLLCondensedClean(pN,apnz,current_space->array,lnk,lnkbt);CHKERRQ(ierr);
    ierr = MatPreallocateSet(i+rstart,apnz,current_space->array,dnz,onz);CHKERRQ(ierr);

    current_space->array           += apnz;
    current_space->local_used      += apnz;
    current_space->local_remaining -= apnz;
  }

  /* Allocate space for apj, initialize apj, and */
  /* destroy list of free space and other temporary array(s) */
  ierr = PetscMalloc1(api[am]+1,&ptap->apj);CHKERRQ(ierr);
  apj  = ptap->apj;
  ierr = PetscFreeSpaceContiguous(&free_space,ptap->apj);CHKERRQ(ierr);
  ierr = PetscLLDestroy(lnk,lnkbt);CHKERRQ(ierr);

  /* malloc apa to store dense row A[i,:]*P */
  ierr = PetscCalloc1(pN,&ptap->apa);CHKERRQ(ierr);

  /* set and assemble symbolic parallel matrix C */
  /*---------------------------------------------*/
  ierr = MatSetSizes(C,am,pn,PETSC_DETERMINE,PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = MatSetBlockSizesFromMats(C,A,P);CHKERRQ(ierr);

  ierr = MatGetType(A,&mtype);CHKERRQ(ierr);
  ierr = MatSetType(C,mtype);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(C,0,dnz,0,onz);CHKERRQ(ierr);
  ierr = MatPreallocateFinalize(dnz,onz);CHKERRQ(ierr);

  ierr = MatSetValues_MPIAIJ_CopyFromCSRFormat_Symbolic(C, apj, api);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatSetOption(C,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);

  C->ops->matmultnumeric = MatMatMultNumeric_MPIAIJ_MPIAIJ_nonscalable;
  C->ops->productnumeric = MatProductNumeric_AB;

  /* attach the supporting struct to C for reuse */
  C->product->data    = ptap;
  C->product->destroy = MatDestroy_MPIAIJ_MatMatMult;

  /* set MatInfo */
  afill = (PetscReal)api[am]/(adi[am]+aoi[am]+pi_loc[pm]+1) + 1.e-5;
  if (afill < 1.0) afill = 1.0;
  C->info.mallocs           = nspacedouble;
  C->info.fill_ratio_given  = fill;
  C->info.fill_ratio_needed = afill;

#if defined(PETSC_USE_INFO)
  if (api[am]) {
    ierr = PetscInfo3(C,"Reallocs %D; Fill ratio: given %g needed %g.\n",nspacedouble,(double)fill,(double)afill);CHKERRQ(ierr);
    ierr = PetscInfo1(C,"Use MatMatMult(A,B,MatReuse,%g,&C) for best performance.;\n",(double)afill);CHKERRQ(ierr);
  } else {
    ierr = PetscInfo(C,"Empty matrix product\n");CHKERRQ(ierr);
  }
#endif
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------- */
static PetscErrorCode MatMatMultSymbolic_MPIAIJ_MPIDense(Mat,Mat,PetscReal,Mat);
static PetscErrorCode MatMatMultNumeric_MPIAIJ_MPIDense(Mat,Mat,Mat);

static PetscErrorCode MatProductSetFromOptions_MPIAIJ_MPIDense_AB(Mat C)
{
  Mat_Product *product = C->product;
  Mat         A = product->A,B=product->B;

  PetscFunctionBegin;
  if (A->cmap->rstart != B->rmap->rstart || A->cmap->rend != B->rmap->rend)
    SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Matrix local dimensions are incompatible, (%D, %D) != (%D,%D)",A->cmap->rstart,A->cmap->rend,B->rmap->rstart,B->rmap->rend);

  C->ops->matmultsymbolic = MatMatMultSymbolic_MPIAIJ_MPIDense;
  C->ops->productsymbolic = MatProductSymbolic_AB;
  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------- */
static PetscErrorCode MatProductSetFromOptions_MPIAIJ_MPIDense_AtB(Mat C)
{
  Mat_Product *product = C->product;
  Mat         A = product->A,B=product->B;

  PetscFunctionBegin;
  if (A->rmap->rstart != B->rmap->rstart || A->rmap->rend != B->rmap->rend)
    SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Matrix local dimensions are incompatible, (%D, %D) != (%D,%D)",A->rmap->rstart,A->rmap->rend,B->rmap->rstart,B->rmap->rend);

  C->ops->transposematmultsymbolic = MatTransposeMatMultSymbolic_MPIAIJ_MPIDense;
  C->ops->productsymbolic          = MatProductSymbolic_AtB;
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------- */
PETSC_INTERN PetscErrorCode MatProductSetFromOptions_MPIAIJ_MPIDense(Mat C)
{
  PetscErrorCode ierr;
  Mat_Product    *product = C->product;

  PetscFunctionBegin;
  switch (product->type) {
  case MATPRODUCT_AB:
    ierr = MatProductSetFromOptions_MPIAIJ_MPIDense_AB(C);CHKERRQ(ierr);
    break;
  case MATPRODUCT_AtB:
    ierr = MatProductSetFromOptions_MPIAIJ_MPIDense_AtB(C);CHKERRQ(ierr);
    break;
  default:
    break;
  }
  PetscFunctionReturn(0);
}
/* ------------------------------------------------------- */

typedef struct {
  Mat          workB,workB1;
  MPI_Request  *rwaits,*swaits;
  PetscInt     nsends,nrecvs;
  MPI_Datatype *stype,*rtype;
  PetscInt     blda;
} MPIAIJ_MPIDense;

PetscErrorCode MatMPIAIJ_MPIDenseDestroy(void *ctx)
{
  MPIAIJ_MPIDense *contents = (MPIAIJ_MPIDense*)ctx;
  PetscErrorCode  ierr;
  PetscInt        i;

  PetscFunctionBegin;
  ierr = MatDestroy(&contents->workB);CHKERRQ(ierr);
  ierr = MatDestroy(&contents->workB1);CHKERRQ(ierr);
  for (i=0; i<contents->nsends; i++) {
    ierr = MPI_Type_free(&contents->stype[i]);CHKERRMPI(ierr);
  }
  for (i=0; i<contents->nrecvs; i++) {
    ierr = MPI_Type_free(&contents->rtype[i]);CHKERRMPI(ierr);
  }
  ierr = PetscFree4(contents->stype,contents->rtype,contents->rwaits,contents->swaits);CHKERRQ(ierr);
  ierr = PetscFree(contents);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMatMultSymbolic_MPIAIJ_MPIDense(Mat A,Mat B,PetscReal fill,Mat C)
{
  PetscErrorCode  ierr;
  Mat_MPIAIJ      *aij=(Mat_MPIAIJ*)A->data;
  PetscInt        nz=aij->B->cmap->n,nsends,nrecvs,i,nrows_to,j,blda,clda,m,M,n,N;
  MPIAIJ_MPIDense *contents;
  VecScatter      ctx=aij->Mvctx;
  PetscInt        Am=A->rmap->n,Bm=B->rmap->n,BN=B->cmap->N,Bbn,Bbn1,bs,nrows_from,numBb;
  MPI_Comm        comm;
  MPI_Datatype    type1,*stype,*rtype;
  const PetscInt  *sindices,*sstarts,*rstarts;
  PetscMPIInt     *disp;
  PetscBool       cisdense;

  PetscFunctionBegin;
  MatCheckProduct(C,4);
  if (C->product->data) SETERRQ(PetscObjectComm((PetscObject)C),PETSC_ERR_PLIB,"Product data not empty");
  ierr = PetscObjectGetComm((PetscObject)A,&comm);CHKERRQ(ierr);
  ierr = PetscObjectBaseTypeCompare((PetscObject)C,MATMPIDENSE,&cisdense);CHKERRQ(ierr);
  if (!cisdense) {
    ierr = MatSetType(C,((PetscObject)B)->type_name);CHKERRQ(ierr);
  }
  ierr = MatGetLocalSize(C,&m,&n);CHKERRQ(ierr);
  ierr = MatGetSize(C,&M,&N);CHKERRQ(ierr);
  if (m == PETSC_DECIDE || n == PETSC_DECIDE || M == PETSC_DECIDE || N == PETSC_DECIDE) {
    ierr = MatSetSizes(C,Am,B->cmap->n,A->rmap->N,BN);CHKERRQ(ierr);
  }
  ierr = MatSetBlockSizesFromMats(C,A,B);CHKERRQ(ierr);
  ierr = MatSetUp(C);CHKERRQ(ierr);
  ierr = MatDenseGetLDA(B,&blda);CHKERRQ(ierr);
  ierr = MatDenseGetLDA(C,&clda);CHKERRQ(ierr);
  ierr = PetscNew(&contents);CHKERRQ(ierr);

  ierr = VecScatterGetRemote_Private(ctx,PETSC_TRUE/*send*/,&nsends,&sstarts,&sindices,NULL,NULL);CHKERRQ(ierr);
  ierr = VecScatterGetRemoteOrdered_Private(ctx,PETSC_FALSE/*recv*/,&nrecvs,&rstarts,NULL,NULL,NULL);CHKERRQ(ierr);

  /* Create column block of B and C for memory scalability when BN is too large */
  /* Estimate Bbn, column size of Bb */
  if (nz) {
    Bbn1 = 2*Am*BN/nz;
    if (!Bbn1) Bbn1 = 1;
  } else Bbn1 = BN;

  bs = PetscAbs(B->cmap->bs);
  Bbn1 = Bbn1/bs *bs; /* Bbn1 is a multiple of bs */
  if (Bbn1 > BN) Bbn1 = BN;
  ierr = MPI_Allreduce(&Bbn1,&Bbn,1,MPIU_INT,MPI_MAX,comm);CHKERRMPI(ierr);

  /* Enable runtime option for Bbn */
  ierr = PetscOptionsBegin(comm,((PetscObject)C)->prefix,"MatMatMult","Mat");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-matmatmult_Bbn","Number of columns in Bb","MatMatMult",Bbn,&Bbn,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  Bbn  = PetscMin(Bbn,BN);

  if (Bbn > 0 && Bbn < BN) {
    numBb = BN/Bbn;
    Bbn1 = BN - numBb*Bbn;
  } else numBb = 0;

  if (numBb) {
    ierr = PetscInfo3(C,"use Bb, BN=%D, Bbn=%D; numBb=%D\n",BN,Bbn,numBb);CHKERRQ(ierr);
    if (Bbn1) { /* Create workB1 for the remaining columns */
      ierr = PetscInfo2(C,"use Bb1, BN=%D, Bbn1=%D\n",BN,Bbn1);CHKERRQ(ierr);
      /* Create work matrix used to store off processor rows of B needed for local product */
      ierr = MatCreateSeqDense(PETSC_COMM_SELF,nz,Bbn1,NULL,&contents->workB1);CHKERRQ(ierr);
    } else contents->workB1 = NULL;
  }

  /* Create work matrix used to store off processor rows of B needed for local product */
  ierr = MatCreateSeqDense(PETSC_COMM_SELF,nz,Bbn,NULL,&contents->workB);CHKERRQ(ierr);

  /* Use MPI derived data type to reduce memory required by the send/recv buffers */
  ierr = PetscMalloc4(nsends,&stype,nrecvs,&rtype,nrecvs,&contents->rwaits,nsends,&contents->swaits);CHKERRQ(ierr);
  contents->stype  = stype;
  contents->nsends = nsends;

  contents->rtype  = rtype;
  contents->nrecvs = nrecvs;
  contents->blda   = blda;

  ierr = PetscMalloc1(Bm+1,&disp);CHKERRQ(ierr);
  for (i=0; i<nsends; i++) {
    nrows_to = sstarts[i+1]-sstarts[i];
    for (j=0; j<nrows_to; j++) {
      disp[j] = sindices[sstarts[i]+j]; /* rowB to be sent */
    }
    ierr = MPI_Type_create_indexed_block(nrows_to,1,(const PetscMPIInt *)disp,MPIU_SCALAR,&type1);CHKERRMPI(ierr);

    ierr = MPI_Type_create_resized(type1,0,blda*sizeof(PetscScalar),&stype[i]);CHKERRMPI(ierr);
    ierr = MPI_Type_commit(&stype[i]);CHKERRMPI(ierr);
    ierr = MPI_Type_free(&type1);CHKERRMPI(ierr);
  }

  for (i=0; i<nrecvs; i++) {
    /* received values from a process form a (nrows_from x Bbn) row block in workB (column-wise) */
    nrows_from = rstarts[i+1]-rstarts[i];
    disp[0] = 0;
    ierr = MPI_Type_create_indexed_block(1, nrows_from, (const PetscMPIInt *)disp, MPIU_SCALAR, &type1);CHKERRMPI(ierr);
    ierr = MPI_Type_create_resized(type1, 0, nz*sizeof(PetscScalar), &rtype[i]);CHKERRMPI(ierr);
    ierr = MPI_Type_commit(&rtype[i]);CHKERRMPI(ierr);
    ierr = MPI_Type_free(&type1);CHKERRMPI(ierr);
  }

  ierr = PetscFree(disp);CHKERRQ(ierr);
  ierr = VecScatterRestoreRemote_Private(ctx,PETSC_TRUE/*send*/,&nsends,&sstarts,&sindices,NULL,NULL);CHKERRQ(ierr);
  ierr = VecScatterRestoreRemoteOrdered_Private(ctx,PETSC_FALSE/*recv*/,&nrecvs,&rstarts,NULL,NULL,NULL);CHKERRQ(ierr);
  ierr = MatSetOption(C,MAT_NO_OFF_PROC_ENTRIES,PETSC_TRUE);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatSetOption(C,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);

  C->product->data = contents;
  C->product->destroy = MatMPIAIJ_MPIDenseDestroy;
  C->ops->matmultnumeric = MatMatMultNumeric_MPIAIJ_MPIDense;
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatMatMultNumericAdd_SeqAIJ_SeqDense(Mat,Mat,Mat,const PetscBool);
/*
    Performs an efficient scatter on the rows of B needed by this process; this is
    a modification of the VecScatterBegin_() routines.

    Input: Bbidx = 0: B = Bb
                 = 1: B = Bb1, see MatMatMultSymbolic_MPIAIJ_MPIDense()
*/
PetscErrorCode MatMPIDenseScatter(Mat A,Mat B,PetscInt Bbidx,Mat C,Mat *outworkB)
{
  Mat_MPIAIJ        *aij = (Mat_MPIAIJ*)A->data;
  PetscErrorCode    ierr;
  const PetscScalar *b;
  PetscScalar       *rvalues;
  VecScatter        ctx = aij->Mvctx;
  const PetscInt    *sindices,*sstarts,*rstarts;
  const PetscMPIInt *sprocs,*rprocs;
  PetscInt          i,nsends,nrecvs;
  MPI_Request       *swaits,*rwaits;
  MPI_Comm          comm;
  PetscMPIInt       tag=((PetscObject)ctx)->tag,ncols=B->cmap->N,nrows=aij->B->cmap->n,nsends_mpi,nrecvs_mpi;
  MPIAIJ_MPIDense   *contents;
  Mat               workB;
  MPI_Datatype      *stype,*rtype;
  PetscInt          blda;

  PetscFunctionBegin;
  MatCheckProduct(C,4);
  if (!C->product->data) SETERRQ(PetscObjectComm((PetscObject)C),PETSC_ERR_PLIB,"Product data empty");
  contents = (MPIAIJ_MPIDense*)C->product->data;
  ierr = VecScatterGetRemote_Private(ctx,PETSC_TRUE/*send*/,&nsends,&sstarts,&sindices,&sprocs,NULL/*bs*/);CHKERRQ(ierr);
  ierr = VecScatterGetRemoteOrdered_Private(ctx,PETSC_FALSE/*recv*/,&nrecvs,&rstarts,NULL,&rprocs,NULL/*bs*/);CHKERRQ(ierr);
  ierr = PetscMPIIntCast(nsends,&nsends_mpi);CHKERRQ(ierr);
  ierr = PetscMPIIntCast(nrecvs,&nrecvs_mpi);CHKERRQ(ierr);
  if (Bbidx == 0) {
    workB = *outworkB = contents->workB;
  } else {
    workB = *outworkB = contents->workB1;
  }
  if (nrows != workB->rmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Number of rows of workB %D not equal to columns of aij->B %D",workB->cmap->n,nrows);
  swaits = contents->swaits;
  rwaits = contents->rwaits;

  ierr = MatDenseGetArrayRead(B,&b);CHKERRQ(ierr);
  ierr = MatDenseGetLDA(B,&blda);CHKERRQ(ierr);
  if (blda != contents->blda) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Cannot reuse an input matrix with lda %D != %D",blda,contents->blda);
  ierr = MatDenseGetArray(workB,&rvalues);CHKERRQ(ierr);

  /* Post recv, use MPI derived data type to save memory */
  ierr = PetscObjectGetComm((PetscObject)C,&comm);CHKERRQ(ierr);
  rtype = contents->rtype;
  for (i=0; i<nrecvs; i++) {
    ierr = MPI_Irecv(rvalues+(rstarts[i]-rstarts[0]),ncols,rtype[i],rprocs[i],tag,comm,rwaits+i);CHKERRMPI(ierr);
  }

  stype = contents->stype;
  for (i=0; i<nsends; i++) {
    ierr = MPI_Isend(b,ncols,stype[i],sprocs[i],tag,comm,swaits+i);CHKERRMPI(ierr);
  }

  if (nrecvs) {ierr = MPI_Waitall(nrecvs_mpi,rwaits,MPI_STATUSES_IGNORE);CHKERRMPI(ierr);}
  if (nsends) {ierr = MPI_Waitall(nsends_mpi,swaits,MPI_STATUSES_IGNORE);CHKERRMPI(ierr);}

  ierr = VecScatterRestoreRemote_Private(ctx,PETSC_TRUE/*send*/,&nsends,&sstarts,&sindices,&sprocs,NULL);CHKERRQ(ierr);
  ierr = VecScatterRestoreRemoteOrdered_Private(ctx,PETSC_FALSE/*recv*/,&nrecvs,&rstarts,NULL,&rprocs,NULL);CHKERRQ(ierr);
  ierr = MatDenseRestoreArrayRead(B,&b);CHKERRQ(ierr);
  ierr = MatDenseRestoreArray(workB,&rvalues);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMatMultNumeric_MPIAIJ_MPIDense(Mat A,Mat B,Mat C)
{
  PetscErrorCode  ierr;
  Mat_MPIAIJ      *aij    = (Mat_MPIAIJ*)A->data;
  Mat_MPIDense    *bdense = (Mat_MPIDense*)B->data;
  Mat_MPIDense    *cdense = (Mat_MPIDense*)C->data;
  Mat             workB;
  MPIAIJ_MPIDense *contents;

  PetscFunctionBegin;
  MatCheckProduct(C,3);
  if (!C->product->data) SETERRQ(PetscObjectComm((PetscObject)C),PETSC_ERR_PLIB,"Product data empty");
  contents = (MPIAIJ_MPIDense*)C->product->data;
  /* diagonal block of A times all local rows of B */
  /* TODO: this calls a symbolic multiplication every time, which could be avoided */
  ierr = MatMatMult(aij->A,bdense->A,MAT_REUSE_MATRIX,PETSC_DEFAULT,&cdense->A);CHKERRQ(ierr);
  if (contents->workB->cmap->n == B->cmap->N) {
    /* get off processor parts of B needed to complete C=A*B */
    ierr = MatMPIDenseScatter(A,B,0,C,&workB);CHKERRQ(ierr);

    /* off-diagonal block of A times nonlocal rows of B */
    ierr = MatMatMultNumericAdd_SeqAIJ_SeqDense(aij->B,workB,cdense->A,PETSC_TRUE);CHKERRQ(ierr);
  } else {
    Mat      Bb,Cb;
    PetscInt BN=B->cmap->N,n=contents->workB->cmap->n,i;
    if (n <= 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Column block size %D must be positive",n);

    for (i=0; i<BN; i+=n) {
      ierr = MatDenseGetSubMatrix(B,i,PetscMin(i+n,BN),&Bb);CHKERRQ(ierr);
      ierr = MatDenseGetSubMatrix(C,i,PetscMin(i+n,BN),&Cb);CHKERRQ(ierr);

      /* get off processor parts of B needed to complete C=A*B */
      ierr = MatMPIDenseScatter(A,Bb,i+n>BN,C,&workB);CHKERRQ(ierr);

      /* off-diagonal block of A times nonlocal rows of B */
      cdense = (Mat_MPIDense*)Cb->data;
      ierr = MatMatMultNumericAdd_SeqAIJ_SeqDense(aij->B,workB,cdense->A,PETSC_TRUE);CHKERRQ(ierr);

      ierr = MatDenseRestoreSubMatrix(B,&Bb);CHKERRQ(ierr);
      ierr = MatDenseRestoreSubMatrix(C,&Cb);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatMatMultNumeric_MPIAIJ_MPIAIJ(Mat A,Mat P,Mat C)
{
  PetscErrorCode    ierr;
  Mat_MPIAIJ        *a   = (Mat_MPIAIJ*)A->data,*c=(Mat_MPIAIJ*)C->data;
  Mat_SeqAIJ        *ad  = (Mat_SeqAIJ*)(a->A)->data,*ao=(Mat_SeqAIJ*)(a->B)->data;
  Mat_SeqAIJ        *cd  = (Mat_SeqAIJ*)(c->A)->data,*co=(Mat_SeqAIJ*)(c->B)->data;
  PetscInt          *adi = ad->i,*adj,*aoi=ao->i,*aoj;
  PetscScalar       *ada,*aoa,*cda=cd->a,*coa=co->a;
  Mat_SeqAIJ        *p_loc,*p_oth;
  PetscInt          *pi_loc,*pj_loc,*pi_oth,*pj_oth,*pj;
  PetscScalar       *pa_loc,*pa_oth,*pa,valtmp,*ca;
  PetscInt          cm    = C->rmap->n,anz,pnz;
  Mat_APMPI         *ptap;
  PetscScalar       *apa_sparse;
  const PetscScalar *dummy;
  PetscInt          *api,*apj,*apJ,i,j,k,row;
  PetscInt          cstart = C->cmap->rstart;
  PetscInt          cdnz,conz,k0,k1,nextp;
  MPI_Comm          comm;
  PetscMPIInt       size;

  PetscFunctionBegin;
  MatCheckProduct(C,3);
  ptap = (Mat_APMPI*)C->product->data;
  if (!ptap) SETERRQ(PetscObjectComm((PetscObject)C),PETSC_ERR_ARG_WRONGSTATE,"PtAP cannot be computed. Missing data");
  ierr = PetscObjectGetComm((PetscObject)C,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRMPI(ierr);
  if (!ptap->P_oth && size>1) SETERRQ(PetscObjectComm((PetscObject)C),PETSC_ERR_ARG_WRONGSTATE,"AP cannot be reused. Do not call MatProductClear()");

  apa_sparse = ptap->apa;

  /* 1) get P_oth = ptap->P_oth  and P_loc = ptap->P_loc */
  /*-----------------------------------------------------*/
  /* update numerical values of P_oth and P_loc */
  ierr = MatGetBrowsOfAoCols_MPIAIJ(A,P,MAT_REUSE_MATRIX,&ptap->startsj_s,&ptap->startsj_r,&ptap->bufa,&ptap->P_oth);CHKERRQ(ierr);
  ierr = MatMPIAIJGetLocalMat(P,MAT_REUSE_MATRIX,&ptap->P_loc);CHKERRQ(ierr);

  /* 2) compute numeric C_loc = A_loc*P = Ad*P_loc + Ao*P_oth */
  /*----------------------------------------------------------*/
  /* get data from symbolic products */
  p_loc = (Mat_SeqAIJ*)(ptap->P_loc)->data;
  pi_loc = p_loc->i; pj_loc = p_loc->j; pa_loc = p_loc->a;
  if (size >1) {
    p_oth = (Mat_SeqAIJ*)(ptap->P_oth)->data;
    pi_oth = p_oth->i; pj_oth = p_oth->j; pa_oth = p_oth->a;
  } else {
    p_oth = NULL; pi_oth = NULL; pj_oth = NULL; pa_oth = NULL;
  }

  /* trigger copy to CPU */
  ierr = MatSeqAIJGetArrayRead(a->A,&dummy);CHKERRQ(ierr);
  ierr = MatSeqAIJRestoreArrayRead(a->A,&dummy);CHKERRQ(ierr);
  ierr = MatSeqAIJGetArrayRead(a->B,&dummy);CHKERRQ(ierr);
  ierr = MatSeqAIJRestoreArrayRead(a->B,&dummy);CHKERRQ(ierr);
  api = ptap->api;
  apj = ptap->apj;
  for (i=0; i<cm; i++) {
    apJ = apj + api[i];

    /* diagonal portion of A */
    anz = adi[i+1] - adi[i];
    adj = ad->j + adi[i];
    ada = ad->a + adi[i];
    for (j=0; j<anz; j++) {
      row = adj[j];
      pnz = pi_loc[row+1] - pi_loc[row];
      pj  = pj_loc + pi_loc[row];
      pa  = pa_loc + pi_loc[row];
      /* perform sparse axpy */
      valtmp = ada[j];
      nextp  = 0;
      for (k=0; nextp<pnz; k++) {
        if (apJ[k] == pj[nextp]) { /* column of AP == column of P */
          apa_sparse[k] += valtmp*pa[nextp++];
        }
      }
      ierr = PetscLogFlops(2.0*pnz);CHKERRQ(ierr);
    }

    /* off-diagonal portion of A */
    anz = aoi[i+1] - aoi[i];
    aoj = ao->j + aoi[i];
    aoa = ao->a + aoi[i];
    for (j=0; j<anz; j++) {
      row = aoj[j];
      pnz = pi_oth[row+1] - pi_oth[row];
      pj  = pj_oth + pi_oth[row];
      pa  = pa_oth + pi_oth[row];
      /* perform sparse axpy */
      valtmp = aoa[j];
      nextp  = 0;
      for (k=0; nextp<pnz; k++) {
        if (apJ[k] == pj[nextp]) { /* column of AP == column of P */
          apa_sparse[k] += valtmp*pa[nextp++];
        }
      }
      ierr = PetscLogFlops(2.0*pnz);CHKERRQ(ierr);
    }

    /* set values in C */
    cdnz = cd->i[i+1] - cd->i[i];
    conz = co->i[i+1] - co->i[i];

    /* 1st off-diagonal part of C */
    ca = coa + co->i[i];
    k  = 0;
    for (k0=0; k0<conz; k0++) {
      if (apJ[k] >= cstart) break;
      ca[k0]        = apa_sparse[k];
      apa_sparse[k] = 0.0;
      k++;
    }

    /* diagonal part of C */
    ca = cda + cd->i[i];
    for (k1=0; k1<cdnz; k1++) {
      ca[k1]        = apa_sparse[k];
      apa_sparse[k] = 0.0;
      k++;
    }

    /* 2nd off-diagonal part of C */
    ca = coa + co->i[i];
    for (; k0<conz; k0++) {
      ca[k0]        = apa_sparse[k];
      apa_sparse[k] = 0.0;
      k++;
    }
  }
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* same as MatMatMultSymbolic_MPIAIJ_MPIAIJ_nonscalable(), except using LLCondensed to avoid O(BN) memory requirement */
PetscErrorCode MatMatMultSymbolic_MPIAIJ_MPIAIJ(Mat A,Mat P,PetscReal fill,Mat C)
{
  PetscErrorCode     ierr;
  MPI_Comm           comm;
  PetscMPIInt        size;
  Mat_APMPI          *ptap;
  PetscFreeSpaceList free_space = NULL,current_space=NULL;
  Mat_MPIAIJ         *a  = (Mat_MPIAIJ*)A->data;
  Mat_SeqAIJ         *ad = (Mat_SeqAIJ*)(a->A)->data,*ao=(Mat_SeqAIJ*)(a->B)->data,*p_loc,*p_oth;
  PetscInt           *pi_loc,*pj_loc,*pi_oth,*pj_oth,*dnz,*onz;
  PetscInt           *adi=ad->i,*adj=ad->j,*aoi=ao->i,*aoj=ao->j,rstart=A->rmap->rstart;
  PetscInt           i,pnz,row,*api,*apj,*Jptr,apnz,nspacedouble=0,j,nzi,*lnk,apnz_max=1;
  PetscInt           am=A->rmap->n,pn=P->cmap->n,pm=P->rmap->n,lsize=pn+20;
  PetscReal          afill;
  MatType            mtype;

  PetscFunctionBegin;
  MatCheckProduct(C,4);
  if (C->product->data) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Extra product struct not empty");
  ierr = PetscObjectGetComm((PetscObject)A,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRMPI(ierr);

  /* create struct Mat_APMPI and attached it to C later */
  ierr = PetscNew(&ptap);CHKERRQ(ierr);

  /* get P_oth by taking rows of P (= non-zero cols of local A) from other processors */
  ierr = MatGetBrowsOfAoCols_MPIAIJ(A,P,MAT_INITIAL_MATRIX,&ptap->startsj_s,&ptap->startsj_r,&ptap->bufa,&ptap->P_oth);CHKERRQ(ierr);

  /* get P_loc by taking all local rows of P */
  ierr = MatMPIAIJGetLocalMat(P,MAT_INITIAL_MATRIX,&ptap->P_loc);CHKERRQ(ierr);

  p_loc  = (Mat_SeqAIJ*)(ptap->P_loc)->data;
  pi_loc = p_loc->i; pj_loc = p_loc->j;
  if (size > 1) {
    p_oth  = (Mat_SeqAIJ*)(ptap->P_oth)->data;
    pi_oth = p_oth->i; pj_oth = p_oth->j;
  } else {
    p_oth  = NULL;
    pi_oth = NULL; pj_oth = NULL;
  }

  /* first, compute symbolic AP = A_loc*P = A_diag*P_loc + A_off*P_oth */
  /*-------------------------------------------------------------------*/
  ierr      = PetscMalloc1(am+2,&api);CHKERRQ(ierr);
  ptap->api = api;
  api[0]    = 0;

  ierr = PetscLLCondensedCreate_Scalable(lsize,&lnk);CHKERRQ(ierr);

  /* Initial FreeSpace size is fill*(nnz(A)+nnz(P)) */
  ierr = PetscFreeSpaceGet(PetscRealIntMultTruncate(fill,PetscIntSumTruncate(adi[am],PetscIntSumTruncate(aoi[am],pi_loc[pm]))),&free_space);CHKERRQ(ierr);
  current_space = free_space;
  ierr = MatPreallocateInitialize(comm,am,pn,dnz,onz);CHKERRQ(ierr);
  for (i=0; i<am; i++) {
    /* diagonal portion of A */
    nzi = adi[i+1] - adi[i];
    for (j=0; j<nzi; j++) {
      row  = *adj++;
      pnz  = pi_loc[row+1] - pi_loc[row];
      Jptr = pj_loc + pi_loc[row];
      /* Expand list if it is not long enough */
      if (pnz+apnz_max > lsize) {
        lsize = pnz+apnz_max;
        ierr = PetscLLCondensedExpand_Scalable(lsize, &lnk);CHKERRQ(ierr);
      }
      /* add non-zero cols of P into the sorted linked list lnk */
      ierr = PetscLLCondensedAddSorted_Scalable(pnz,Jptr,lnk);CHKERRQ(ierr);
      apnz     = *lnk; /* The first element in the list is the number of items in the list */
      api[i+1] = api[i] + apnz;
      if (apnz > apnz_max) apnz_max = apnz + 1; /* '1' for diagonal entry */
    }
    /* off-diagonal portion of A */
    nzi = aoi[i+1] - aoi[i];
    for (j=0; j<nzi; j++) {
      row  = *aoj++;
      pnz  = pi_oth[row+1] - pi_oth[row];
      Jptr = pj_oth + pi_oth[row];
      /* Expand list if it is not long enough */
      if (pnz+apnz_max > lsize) {
        lsize = pnz + apnz_max;
        ierr = PetscLLCondensedExpand_Scalable(lsize, &lnk);CHKERRQ(ierr);
      }
      /* add non-zero cols of P into the sorted linked list lnk */
      ierr = PetscLLCondensedAddSorted_Scalable(pnz,Jptr,lnk);CHKERRQ(ierr);
      apnz     = *lnk;  /* The first element in the list is the number of items in the list */
      api[i+1] = api[i] + apnz;
      if (apnz > apnz_max) apnz_max = apnz + 1; /* '1' for diagonal entry */
    }

    /* add missing diagonal entry */
    if (C->force_diagonals) {
      j = i + rstart; /* column index */
      ierr = PetscLLCondensedAddSorted_Scalable(1,&j,lnk);CHKERRQ(ierr);
    }

    apnz     = *lnk;
    api[i+1] = api[i] + apnz;
    if (apnz > apnz_max) apnz_max = apnz;

    /* if free space is not available, double the total space in the list */
    if (current_space->local_remaining<apnz) {
      ierr = PetscFreeSpaceGet(PetscIntSumTruncate(apnz,current_space->total_array_size),&current_space);CHKERRQ(ierr);
      nspacedouble++;
    }

    /* Copy data into free space, then initialize lnk */
    ierr = PetscLLCondensedClean_Scalable(apnz,current_space->array,lnk);CHKERRQ(ierr);
    ierr = MatPreallocateSet(i+rstart,apnz,current_space->array,dnz,onz);CHKERRQ(ierr);

    current_space->array           += apnz;
    current_space->local_used      += apnz;
    current_space->local_remaining -= apnz;
  }

  /* Allocate space for apj, initialize apj, and */
  /* destroy list of free space and other temporary array(s) */
  ierr = PetscMalloc1(api[am]+1,&ptap->apj);CHKERRQ(ierr);
  apj  = ptap->apj;
  ierr = PetscFreeSpaceContiguous(&free_space,ptap->apj);CHKERRQ(ierr);
  ierr = PetscLLCondensedDestroy_Scalable(lnk);CHKERRQ(ierr);

  /* create and assemble symbolic parallel matrix C */
  /*----------------------------------------------------*/
  ierr = MatSetSizes(C,am,pn,PETSC_DETERMINE,PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = MatSetBlockSizesFromMats(C,A,P);CHKERRQ(ierr);
  ierr = MatGetType(A,&mtype);CHKERRQ(ierr);
  ierr = MatSetType(C,mtype);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(C,0,dnz,0,onz);CHKERRQ(ierr);
  ierr = MatPreallocateFinalize(dnz,onz);CHKERRQ(ierr);

  /* malloc apa for assembly C */
  ierr = PetscCalloc1(apnz_max,&ptap->apa);CHKERRQ(ierr);

  ierr = MatSetValues_MPIAIJ_CopyFromCSRFormat_Symbolic(C, apj, api);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatSetOption(C,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);

  C->ops->matmultnumeric = MatMatMultNumeric_MPIAIJ_MPIAIJ;
  C->ops->productnumeric = MatProductNumeric_AB;

  /* attach the supporting struct to C for reuse */
  C->product->data    = ptap;
  C->product->destroy = MatDestroy_MPIAIJ_MatMatMult;

  /* set MatInfo */
  afill = (PetscReal)api[am]/(adi[am]+aoi[am]+pi_loc[pm]+1) + 1.e-5;
  if (afill < 1.0) afill = 1.0;
  C->info.mallocs           = nspacedouble;
  C->info.fill_ratio_given  = fill;
  C->info.fill_ratio_needed = afill;

#if defined(PETSC_USE_INFO)
  if (api[am]) {
    ierr = PetscInfo3(C,"Reallocs %D; Fill ratio: given %g needed %g.\n",nspacedouble,(double)fill,(double)afill);CHKERRQ(ierr);
    ierr = PetscInfo1(C,"Use MatMatMult(A,B,MatReuse,%g,&C) for best performance.;\n",(double)afill);CHKERRQ(ierr);
  } else {
    ierr = PetscInfo(C,"Empty matrix product\n");CHKERRQ(ierr);
  }
#endif
  PetscFunctionReturn(0);
}

/* This function is needed for the seqMPI matrix-matrix multiplication.  */
/* Three input arrays are merged to one output array. The size of the    */
/* output array is also output. Duplicate entries only show up once.     */
static void Merge3SortedArrays(PetscInt  size1, PetscInt *in1,
                               PetscInt  size2, PetscInt *in2,
                               PetscInt  size3, PetscInt *in3,
                               PetscInt *size4, PetscInt *out)
{
  int i = 0, j = 0, k = 0, l = 0;

  /* Traverse all three arrays */
  while (i<size1 && j<size2 && k<size3) {
    if (in1[i] < in2[j] && in1[i] < in3[k]) {
      out[l++] = in1[i++];
    }
    else if (in2[j] < in1[i] && in2[j] < in3[k]) {
      out[l++] = in2[j++];
    }
    else if (in3[k] < in1[i] && in3[k] < in2[j]) {
      out[l++] = in3[k++];
    }
    else if (in1[i] == in2[j] && in1[i] < in3[k]) {
      out[l++] = in1[i];
      i++, j++;
    }
    else if (in1[i] == in3[k] && in1[i] < in2[j]) {
      out[l++] = in1[i];
      i++, k++;
    }
    else if (in3[k] == in2[j] && in2[j] < in1[i])  {
      out[l++] = in2[j];
      k++, j++;
    }
    else if (in1[i] == in2[j] && in1[i] == in3[k]) {
      out[l++] = in1[i];
      i++, j++, k++;
    }
  }

  /* Traverse two remaining arrays */
  while (i<size1 && j<size2) {
    if (in1[i] < in2[j]) {
      out[l++] = in1[i++];
    }
    else if (in1[i] > in2[j]) {
      out[l++] = in2[j++];
    }
    else {
      out[l++] = in1[i];
      i++, j++;
    }
  }

  while (i<size1 && k<size3) {
    if (in1[i] < in3[k]) {
      out[l++] = in1[i++];
    }
    else if (in1[i] > in3[k]) {
      out[l++] = in3[k++];
    }
    else {
      out[l++] = in1[i];
      i++, k++;
    }
  }

  while (k<size3 && j<size2)  {
    if (in3[k] < in2[j]) {
      out[l++] = in3[k++];
    }
    else if (in3[k] > in2[j]) {
      out[l++] = in2[j++];
    }
    else {
      out[l++] = in3[k];
      k++, j++;
    }
  }

  /* Traverse one remaining array */
  while (i<size1) out[l++] = in1[i++];
  while (j<size2) out[l++] = in2[j++];
  while (k<size3) out[l++] = in3[k++];

  *size4 = l;
}

/* This matrix-matrix multiplication algorithm divides the multiplication into three multiplications and  */
/* adds up the products. Two of these three multiplications are performed with existing (sequential)      */
/* matrix-matrix multiplications.  */
PetscErrorCode MatMatMultSymbolic_MPIAIJ_MPIAIJ_seqMPI(Mat A, Mat P, PetscReal fill, Mat C)
{
  PetscErrorCode     ierr;
  MPI_Comm           comm;
  PetscMPIInt        size;
  Mat_APMPI          *ptap;
  PetscFreeSpaceList free_space_diag=NULL, current_space=NULL;
  Mat_MPIAIJ         *a  =(Mat_MPIAIJ*)A->data;
  Mat_SeqAIJ         *ad =(Mat_SeqAIJ*)(a->A)->data,*ao=(Mat_SeqAIJ*)(a->B)->data,*p_loc;
  Mat_MPIAIJ         *p  =(Mat_MPIAIJ*)P->data;
  Mat_SeqAIJ         *adpd_seq, *p_off, *aopoth_seq;
  PetscInt           adponz, adpdnz;
  PetscInt           *pi_loc,*dnz,*onz;
  PetscInt           *adi=ad->i,*adj=ad->j,*aoi=ao->i,rstart=A->rmap->rstart;
  PetscInt           *lnk,i, i1=0,pnz,row,*adpoi,*adpoj, *api, *adpoJ, *aopJ, *apJ,*Jptr, aopnz, nspacedouble=0,j,nzi,
                     *apj,apnz, *adpdi, *adpdj, *adpdJ, *poff_i, *poff_j, *j_temp, *aopothi, *aopothj;
  PetscInt           am=A->rmap->n,pN=P->cmap->N,pn=P->cmap->n,pm=P->rmap->n, p_colstart, p_colend;
  PetscBT            lnkbt;
  PetscReal          afill;
  PetscMPIInt        rank;
  Mat                adpd, aopoth;
  MatType            mtype;
  const char         *prefix;

  PetscFunctionBegin;
  MatCheckProduct(C,4);
  if (C->product->data) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Extra product struct not empty");
  ierr = PetscObjectGetComm((PetscObject)A,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRMPI(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRMPI(ierr);
  ierr = MatGetOwnershipRangeColumn(P, &p_colstart, &p_colend);CHKERRQ(ierr);

  /* create struct Mat_APMPI and attached it to C later */
  ierr = PetscNew(&ptap);CHKERRQ(ierr);

  /* get P_oth by taking rows of P (= non-zero cols of local A) from other processors */
  ierr = MatGetBrowsOfAoCols_MPIAIJ(A,P,MAT_INITIAL_MATRIX,&ptap->startsj_s,&ptap->startsj_r,&ptap->bufa,&ptap->P_oth);CHKERRQ(ierr);

  /* get P_loc by taking all local rows of P */
  ierr = MatMPIAIJGetLocalMat(P,MAT_INITIAL_MATRIX,&ptap->P_loc);CHKERRQ(ierr);

  p_loc  = (Mat_SeqAIJ*)(ptap->P_loc)->data;
  pi_loc = p_loc->i;

  /* Allocate memory for the i arrays of the matrices A*P, A_diag*P_off and A_offd * P */
  ierr      = PetscMalloc1(am+2,&api);CHKERRQ(ierr);
  ierr      = PetscMalloc1(am+2,&adpoi);CHKERRQ(ierr);

  adpoi[0]    = 0;
  ptap->api = api;
  api[0] = 0;

  /* create and initialize a linked list, will be used for both A_diag * P_loc_off and A_offd * P_oth */
  ierr = PetscLLCondensedCreate(pN,pN,&lnk,&lnkbt);CHKERRQ(ierr);
  ierr = MatPreallocateInitialize(comm,am,pn,dnz,onz);CHKERRQ(ierr);

  /* Symbolic calc of A_loc_diag * P_loc_diag */
  ierr = MatGetOptionsPrefix(A,&prefix);CHKERRQ(ierr);
  ierr = MatProductCreate(a->A,p->A,NULL,&adpd);CHKERRQ(ierr);
  ierr = MatGetOptionsPrefix(A,&prefix);CHKERRQ(ierr);
  ierr = MatSetOptionsPrefix(adpd,prefix);CHKERRQ(ierr);
  ierr = MatAppendOptionsPrefix(adpd,"inner_diag_");CHKERRQ(ierr);

  ierr = MatProductSetType(adpd,MATPRODUCT_AB);CHKERRQ(ierr);
  ierr = MatProductSetAlgorithm(adpd,"sorted");CHKERRQ(ierr);
  ierr = MatProductSetFill(adpd,fill);CHKERRQ(ierr);
  ierr = MatProductSetFromOptions(adpd);CHKERRQ(ierr);

  adpd->force_diagonals = C->force_diagonals;
  ierr = MatProductSymbolic(adpd);CHKERRQ(ierr);

  adpd_seq = (Mat_SeqAIJ*)((adpd)->data);
  adpdi = adpd_seq->i; adpdj = adpd_seq->j;
  p_off = (Mat_SeqAIJ*)((p->B)->data);
  poff_i = p_off->i; poff_j = p_off->j;

  /* j_temp stores indices of a result row before they are added to the linked list */
  ierr = PetscMalloc1(pN+2,&j_temp);CHKERRQ(ierr);

  /* Symbolic calc of the A_diag * p_loc_off */
  /* Initial FreeSpace size is fill*(nnz(A)+nnz(P)) */
  ierr = PetscFreeSpaceGet(PetscRealIntMultTruncate(fill,PetscIntSumTruncate(adi[am],PetscIntSumTruncate(aoi[am],pi_loc[pm]))),&free_space_diag);CHKERRQ(ierr);
  current_space = free_space_diag;

  for (i=0; i<am; i++) {
    /* A_diag * P_loc_off */
    nzi = adi[i+1] - adi[i];
    for (j=0; j<nzi; j++) {
      row  = *adj++;
      pnz  = poff_i[row+1] - poff_i[row];
      Jptr = poff_j + poff_i[row];
      for (i1 = 0; i1 < pnz; i1++) {
        j_temp[i1] = p->garray[Jptr[i1]];
      }
      /* add non-zero cols of P into the sorted linked list lnk */
      ierr = PetscLLCondensedAddSorted(pnz,j_temp,lnk,lnkbt);CHKERRQ(ierr);
    }

    adponz     = lnk[0];
    adpoi[i+1] = adpoi[i] + adponz;

    /* if free space is not available, double the total space in the list */
    if (current_space->local_remaining<adponz) {
      ierr = PetscFreeSpaceGet(PetscIntSumTruncate(adponz,current_space->total_array_size),&current_space);CHKERRQ(ierr);
      nspacedouble++;
    }

    /* Copy data into free space, then initialize lnk */
    ierr = PetscLLCondensedClean(pN,adponz,current_space->array,lnk,lnkbt);CHKERRQ(ierr);

    current_space->array           += adponz;
    current_space->local_used      += adponz;
    current_space->local_remaining -= adponz;
  }

  /* Symbolic calc of A_off * P_oth */
  ierr = MatSetOptionsPrefix(a->B,prefix);CHKERRQ(ierr);
  ierr = MatAppendOptionsPrefix(a->B,"inner_offdiag_");CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_SELF,&aopoth);CHKERRQ(ierr);
  ierr = MatMatMultSymbolic_SeqAIJ_SeqAIJ(a->B, ptap->P_oth, fill, aopoth);CHKERRQ(ierr);
  aopoth_seq = (Mat_SeqAIJ*)((aopoth)->data);
  aopothi = aopoth_seq->i; aopothj = aopoth_seq->j;

  /* Allocate space for apj, adpj, aopj, ... */
  /* destroy lists of free space and other temporary array(s) */

  ierr = PetscMalloc1(aopothi[am] + adpoi[am] + adpdi[am]+2, &ptap->apj);CHKERRQ(ierr);
  ierr = PetscMalloc1(adpoi[am]+2, &adpoj);CHKERRQ(ierr);

  /* Copy from linked list to j-array */
  ierr = PetscFreeSpaceContiguous(&free_space_diag,adpoj);CHKERRQ(ierr);
  ierr = PetscLLDestroy(lnk,lnkbt);CHKERRQ(ierr);

  adpoJ = adpoj;
  adpdJ = adpdj;
  aopJ = aopothj;
  apj  = ptap->apj;
  apJ = apj; /* still empty */

  /* Merge j-arrays of A_off * P, A_diag * P_loc_off, and */
  /* A_diag * P_loc_diag to get A*P */
  for (i = 0; i < am; i++) {
    aopnz  =  aopothi[i+1] -  aopothi[i];
    adponz = adpoi[i+1] - adpoi[i];
    adpdnz = adpdi[i+1] - adpdi[i];

    /* Correct indices from A_diag*P_diag */
    for (i1 = 0; i1 < adpdnz; i1++) {
      adpdJ[i1] += p_colstart;
    }
    /* Merge j-arrays of A_diag * P_loc_off and A_diag * P_loc_diag and A_off * P_oth */
    Merge3SortedArrays(adponz, adpoJ, adpdnz, adpdJ, aopnz, aopJ, &apnz, apJ);
    ierr = MatPreallocateSet(i+rstart, apnz, apJ, dnz, onz);CHKERRQ(ierr);

    aopJ += aopnz;
    adpoJ += adponz;
    adpdJ += adpdnz;
    apJ += apnz;
    api[i+1] = api[i] + apnz;
  }

  /* malloc apa to store dense row A[i,:]*P */
  ierr = PetscCalloc1(pN+2,&ptap->apa);CHKERRQ(ierr);

  /* create and assemble symbolic parallel matrix C */
  ierr = MatSetSizes(C,am,pn,PETSC_DETERMINE,PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = MatSetBlockSizesFromMats(C,A,P);CHKERRQ(ierr);
  ierr = MatGetType(A,&mtype);CHKERRQ(ierr);
  ierr = MatSetType(C,mtype);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(C,0,dnz,0,onz);CHKERRQ(ierr);
  ierr = MatPreallocateFinalize(dnz,onz);CHKERRQ(ierr);

  ierr = MatSetValues_MPIAIJ_CopyFromCSRFormat_Symbolic(C, apj, api);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatSetOption(C,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);

  C->ops->matmultnumeric = MatMatMultNumeric_MPIAIJ_MPIAIJ_nonscalable;
  C->ops->productnumeric = MatProductNumeric_AB;

  /* attach the supporting struct to C for reuse */
  C->product->data    = ptap;
  C->product->destroy = MatDestroy_MPIAIJ_MatMatMult;

  /* set MatInfo */
  afill = (PetscReal)api[am]/(adi[am]+aoi[am]+pi_loc[pm]+1) + 1.e-5;
  if (afill < 1.0) afill = 1.0;
  C->info.mallocs           = nspacedouble;
  C->info.fill_ratio_given  = fill;
  C->info.fill_ratio_needed = afill;

#if defined(PETSC_USE_INFO)
  if (api[am]) {
    ierr = PetscInfo3(C,"Reallocs %D; Fill ratio: given %g needed %g.\n",nspacedouble,(double)fill,(double)afill);CHKERRQ(ierr);
    ierr = PetscInfo1(C,"Use MatMatMult(A,B,MatReuse,%g,&C) for best performance.;\n",(double)afill);CHKERRQ(ierr);
  } else {
    ierr = PetscInfo(C,"Empty matrix product\n");CHKERRQ(ierr);
  }
#endif

  ierr = MatDestroy(&aopoth);CHKERRQ(ierr);
  ierr = MatDestroy(&adpd);CHKERRQ(ierr);
  ierr = PetscFree(j_temp);CHKERRQ(ierr);
  ierr = PetscFree(adpoj);CHKERRQ(ierr);
  ierr = PetscFree(adpoi);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*-------------------------------------------------------------------------*/
/* This routine only works when scall=MAT_REUSE_MATRIX! */
PetscErrorCode MatTransposeMatMultNumeric_MPIAIJ_MPIAIJ_matmatmult(Mat P,Mat A,Mat C)
{
  PetscErrorCode ierr;
  Mat_APMPI      *ptap;
  Mat            Pt;

  PetscFunctionBegin;
  MatCheckProduct(C,3);
  ptap = (Mat_APMPI*)C->product->data;
  if (!ptap) SETERRQ(PetscObjectComm((PetscObject)C),PETSC_ERR_ARG_WRONGSTATE,"PtAP cannot be computed. Missing data");
  if (!ptap->Pt) SETERRQ(PetscObjectComm((PetscObject)C),PETSC_ERR_ARG_WRONGSTATE,"PtA cannot be reused. Do not call MatProductClear()");

  Pt   = ptap->Pt;
  ierr = MatTranspose(P,MAT_REUSE_MATRIX,&Pt);CHKERRQ(ierr);
  ierr = MatMatMultNumeric_MPIAIJ_MPIAIJ(Pt,A,C);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* This routine is modified from MatPtAPSymbolic_MPIAIJ_MPIAIJ() */
PetscErrorCode MatTransposeMatMultSymbolic_MPIAIJ_MPIAIJ_nonscalable(Mat P,Mat A,PetscReal fill,Mat C)
{
  PetscErrorCode      ierr;
  Mat_APMPI           *ptap;
  Mat_MPIAIJ          *p=(Mat_MPIAIJ*)P->data;
  MPI_Comm            comm;
  PetscMPIInt         size,rank;
  PetscFreeSpaceList  free_space=NULL,current_space=NULL;
  PetscInt            pn=P->cmap->n,aN=A->cmap->N,an=A->cmap->n;
  PetscInt            *lnk,i,k,nsend,rstart;
  PetscBT             lnkbt;
  PetscMPIInt         tagi,tagj,*len_si,*len_s,*len_ri,nrecv;
  PETSC_UNUSED PetscMPIInt icompleted=0;
  PetscInt            **buf_rj,**buf_ri,**buf_ri_k,row,ncols,*cols;
  PetscInt            len,proc,*dnz,*onz,*owners,nzi;
  PetscInt            nrows,*buf_s,*buf_si,*buf_si_i,**nextrow,**nextci;
  MPI_Request         *swaits,*rwaits;
  MPI_Status          *sstatus,rstatus;
  PetscLayout         rowmap;
  PetscInt            *owners_co,*coi,*coj;    /* i and j array of (p->B)^T*A*P - used in the communication */
  PetscMPIInt         *len_r,*id_r;    /* array of length of comm->size, store send/recv matrix values */
  PetscInt            *Jptr,*prmap=p->garray,con,j,Crmax;
  Mat_SeqAIJ          *a_loc,*c_loc,*c_oth;
  PetscTable          ta;
  MatType             mtype;
  const char          *prefix;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)A,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRMPI(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRMPI(ierr);

  /* create symbolic parallel matrix C */
  ierr = MatGetType(A,&mtype);CHKERRQ(ierr);
  ierr = MatSetType(C,mtype);CHKERRQ(ierr);

  C->ops->transposematmultnumeric = MatTransposeMatMultNumeric_MPIAIJ_MPIAIJ_nonscalable;

  /* create struct Mat_APMPI and attached it to C later */
  ierr = PetscNew(&ptap);CHKERRQ(ierr);
  ptap->reuse = MAT_INITIAL_MATRIX;

  /* (0) compute Rd = Pd^T, Ro = Po^T  */
  /* --------------------------------- */
  ierr = MatTranspose_SeqAIJ(p->A,MAT_INITIAL_MATRIX,&ptap->Rd);CHKERRQ(ierr);
  ierr = MatTranspose_SeqAIJ(p->B,MAT_INITIAL_MATRIX,&ptap->Ro);CHKERRQ(ierr);

  /* (1) compute symbolic A_loc */
  /* ---------------------------*/
  ierr = MatMPIAIJGetLocalMat(A,MAT_INITIAL_MATRIX,&ptap->A_loc);CHKERRQ(ierr);

  /* (2-1) compute symbolic C_oth = Ro*A_loc  */
  /* ------------------------------------ */
  ierr = MatGetOptionsPrefix(A,&prefix);CHKERRQ(ierr);
  ierr = MatSetOptionsPrefix(ptap->Ro,prefix);CHKERRQ(ierr);
  ierr = MatAppendOptionsPrefix(ptap->Ro,"inner_offdiag_");CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_SELF,&ptap->C_oth);CHKERRQ(ierr);
  ierr = MatMatMultSymbolic_SeqAIJ_SeqAIJ(ptap->Ro,ptap->A_loc,fill,ptap->C_oth);CHKERRQ(ierr);

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
  ierr = PetscArrayzero(len_s,size);CHKERRQ(ierr);
  ierr = PetscArrayzero(len_si,size);CHKERRQ(ierr);

  c_oth = (Mat_SeqAIJ*)ptap->C_oth->data;
  coi   = c_oth->i; coj = c_oth->j;
  con   = ptap->C_oth->rmap->n;
  proc  = 0;
  for (i=0; i<con; i++) {
    while (prmap[i] >= owners[proc+1]) proc++;
    len_si[proc]++;               /* num of rows in Co(=Pt*A) to be sent to [proc] */
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
    ierr = MPI_Isend(coj+coi[i],len_s[proc],MPIU_INT,proc,tagj,comm,swaits+k);CHKERRMPI(ierr);
    k++;
  }

  /* (2-2) compute symbolic C_loc = Rd*A_loc */
  /* ---------------------------------------- */
  ierr = MatSetOptionsPrefix(ptap->Rd,prefix);CHKERRQ(ierr);
  ierr = MatAppendOptionsPrefix(ptap->Rd,"inner_diag_");CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_SELF,&ptap->C_loc);CHKERRQ(ierr);
  ierr = MatMatMultSymbolic_SeqAIJ_SeqAIJ(ptap->Rd,ptap->A_loc,fill,ptap->C_loc);CHKERRQ(ierr);
  c_loc = (Mat_SeqAIJ*)ptap->C_loc->data;

  /* receives coj are complete */
  for (i=0; i<nrecv; i++) {
    ierr = MPI_Waitany(nrecv,rwaits,&icompleted,&rstatus);CHKERRMPI(ierr);
  }
  ierr = PetscFree(rwaits);CHKERRQ(ierr);
  if (nsend) {ierr = MPI_Waitall(nsend,swaits,sstatus);CHKERRMPI(ierr);}

  /* add received column indices into ta to update Crmax */
  a_loc = (Mat_SeqAIJ*)(ptap->A_loc)->data;

  /* create and initialize a linked list */
  ierr = PetscTableCreate(an,aN,&ta);CHKERRQ(ierr); /* for compute Crmax */
  MatRowMergeMax_SeqAIJ(a_loc,ptap->A_loc->rmap->N,ta);

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
    ierr = MPI_Isend(buf_si,len_si[proc],MPIU_INT,proc,tagi,comm,swaits+k);CHKERRMPI(ierr);
    k++;
    buf_si += len_si[proc];
  }
  for (i=0; i<nrecv; i++) {
    ierr = MPI_Waitany(nrecv,rwaits,&icompleted,&rstatus);CHKERRMPI(ierr);
  }
  ierr = PetscFree(rwaits);CHKERRQ(ierr);
  if (nsend) {ierr = MPI_Waitall(nsend,swaits,sstatus);CHKERRMPI(ierr);}

  ierr = PetscFree4(len_s,len_si,sstatus,owners_co);CHKERRQ(ierr);
  ierr = PetscFree(len_ri);CHKERRQ(ierr);
  ierr = PetscFree(swaits);CHKERRQ(ierr);
  ierr = PetscFree(buf_s);CHKERRQ(ierr);

  /* (5) compute the local portion of C      */
  /* ------------------------------------------ */
  /* set initial free space to be Crmax, sufficient for holding nozeros in each row of C */
  ierr          = PetscFreeSpaceGet(Crmax,&free_space);CHKERRQ(ierr);
  current_space = free_space;

  ierr = PetscMalloc3(nrecv,&buf_ri_k,nrecv,&nextrow,nrecv,&nextci);CHKERRQ(ierr);
  for (k=0; k<nrecv; k++) {
    buf_ri_k[k] = buf_ri[k]; /* beginning of k-th recved i-structure */
    nrows       = *buf_ri_k[k];
    nextrow[k]  = buf_ri_k[k] + 1;  /* next row number of k-th recved i-structure */
    nextci[k]   = buf_ri_k[k] + (nrows + 1); /* poins to the next i-structure of k-th recved i-structure  */
  }

  ierr = MatPreallocateInitialize(comm,pn,an,dnz,onz);CHKERRQ(ierr);
  ierr = PetscLLCondensedCreate(Crmax,aN,&lnk,&lnkbt);CHKERRQ(ierr);
  for (i=0; i<pn; i++) { /* for each local row of C */
    /* add C_loc into C */
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

    /* add missing diagonal entry */
    if (C->force_diagonals) {
      k = i + owners[rank]; /* column index */
      ierr = PetscLLCondensedAddSorted(1,&k,lnk,lnkbt);CHKERRQ(ierr);
    }

    nzi = lnk[0];

    /* copy data into free space, then initialize lnk */
    ierr = PetscLLCondensedClean(aN,nzi,current_space->array,lnk,lnkbt);CHKERRQ(ierr);
    ierr = MatPreallocateSet(i+owners[rank],nzi,current_space->array,dnz,onz);CHKERRQ(ierr);
  }
  ierr = PetscFree3(buf_ri_k,nextrow,nextci);CHKERRQ(ierr);
  ierr = PetscLLDestroy(lnk,lnkbt);CHKERRQ(ierr);
  ierr = PetscFreeSpaceDestroy(free_space);CHKERRQ(ierr);

  /* local sizes and preallocation */
  ierr = MatSetSizes(C,pn,an,PETSC_DETERMINE,PETSC_DETERMINE);CHKERRQ(ierr);
  if (P->cmap->bs > 0) {ierr = PetscLayoutSetBlockSize(C->rmap,P->cmap->bs);CHKERRQ(ierr);}
  if (A->cmap->bs > 0) {ierr = PetscLayoutSetBlockSize(C->cmap,A->cmap->bs);CHKERRQ(ierr);}
  ierr = MatMPIAIJSetPreallocation(C,0,dnz,0,onz);CHKERRQ(ierr);
  ierr = MatPreallocateFinalize(dnz,onz);CHKERRQ(ierr);

  /* add C_loc and C_oth to C */
  ierr = MatGetOwnershipRange(C,&rstart,NULL);CHKERRQ(ierr);
  for (i=0; i<pn; i++) {
    ncols = c_loc->i[i+1] - c_loc->i[i];
    cols  = c_loc->j + c_loc->i[i];
    row   = rstart + i;
    ierr = MatSetValues(C,1,(const PetscInt*)&row,ncols,(const PetscInt*)cols,NULL,INSERT_VALUES);CHKERRQ(ierr);

    if (C->force_diagonals) {
      ierr = MatSetValues(C,1,(const PetscInt*)&row,1,(const PetscInt*)&row,NULL,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  for (i=0; i<con; i++) {
    ncols = c_oth->i[i+1] - c_oth->i[i];
    cols  = c_oth->j + c_oth->i[i];
    row   = prmap[i];
    ierr = MatSetValues(C,1,(const PetscInt*)&row,ncols,(const PetscInt*)cols,NULL,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatSetOption(C,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);

  /* members in merge */
  ierr = PetscFree(id_r);CHKERRQ(ierr);
  ierr = PetscFree(len_r);CHKERRQ(ierr);
  ierr = PetscFree(buf_ri[0]);CHKERRQ(ierr);
  ierr = PetscFree(buf_ri);CHKERRQ(ierr);
  ierr = PetscFree(buf_rj[0]);CHKERRQ(ierr);
  ierr = PetscFree(buf_rj);CHKERRQ(ierr);
  ierr = PetscLayoutDestroy(&rowmap);CHKERRQ(ierr);

  /* attach the supporting struct to C for reuse */
  C->product->data    = ptap;
  C->product->destroy = MatDestroy_MPIAIJ_PtAP;
  PetscFunctionReturn(0);
}

PetscErrorCode MatTransposeMatMultNumeric_MPIAIJ_MPIAIJ_nonscalable(Mat P,Mat A,Mat C)
{
  PetscErrorCode    ierr;
  Mat_MPIAIJ        *p=(Mat_MPIAIJ*)P->data;
  Mat_SeqAIJ        *c_seq;
  Mat_APMPI         *ptap;
  Mat               A_loc,C_loc,C_oth;
  PetscInt          i,rstart,rend,cm,ncols,row;
  const PetscInt    *cols;
  const PetscScalar *vals;

  PetscFunctionBegin;
  MatCheckProduct(C,3);
  ptap = (Mat_APMPI*)C->product->data;
  if (!ptap) SETERRQ(PetscObjectComm((PetscObject)C),PETSC_ERR_ARG_WRONGSTATE,"PtAP cannot be computed. Missing data");
  if (!ptap->A_loc) SETERRQ(PetscObjectComm((PetscObject)C),PETSC_ERR_ARG_WRONGSTATE,"PtA cannot be reused. Do not call MatProductClear()");
  ierr = MatZeroEntries(C);CHKERRQ(ierr);

  if (ptap->reuse == MAT_REUSE_MATRIX) {
    /* These matrices are obtained in MatTransposeMatMultSymbolic() */
    /* 1) get R = Pd^T, Ro = Po^T */
    /*----------------------------*/
    ierr = MatTranspose_SeqAIJ(p->A,MAT_REUSE_MATRIX,&ptap->Rd);CHKERRQ(ierr);
    ierr = MatTranspose_SeqAIJ(p->B,MAT_REUSE_MATRIX,&ptap->Ro);CHKERRQ(ierr);

    /* 2) compute numeric A_loc */
    /*--------------------------*/
    ierr = MatMPIAIJGetLocalMat(A,MAT_REUSE_MATRIX,&ptap->A_loc);CHKERRQ(ierr);
  }

  /* 3) C_loc = Rd*A_loc, C_oth = Ro*A_loc */
  A_loc = ptap->A_loc;
  ierr = ((ptap->C_loc)->ops->matmultnumeric)(ptap->Rd,A_loc,ptap->C_loc);CHKERRQ(ierr);
  ierr = ((ptap->C_oth)->ops->matmultnumeric)(ptap->Ro,A_loc,ptap->C_oth);CHKERRQ(ierr);
  C_loc = ptap->C_loc;
  C_oth = ptap->C_oth;

  /* add C_loc and C_oth to C */
  ierr = MatGetOwnershipRange(C,&rstart,&rend);CHKERRQ(ierr);

  /* C_loc -> C */
  cm    = C_loc->rmap->N;
  c_seq = (Mat_SeqAIJ*)C_loc->data;
  cols = c_seq->j;
  vals = c_seq->a;
  for (i=0; i<cm; i++) {
    ncols = c_seq->i[i+1] - c_seq->i[i];
    row = rstart + i;
    ierr = MatSetValues(C,1,&row,ncols,cols,vals,ADD_VALUES);CHKERRQ(ierr);
    cols += ncols; vals += ncols;
  }

  /* Co -> C, off-processor part */
  cm    = C_oth->rmap->N;
  c_seq = (Mat_SeqAIJ*)C_oth->data;
  cols  = c_seq->j;
  vals  = c_seq->a;
  for (i=0; i<cm; i++) {
    ncols = c_seq->i[i+1] - c_seq->i[i];
    row = p->garray[i];
    ierr = MatSetValues(C,1,&row,ncols,cols,vals,ADD_VALUES);CHKERRQ(ierr);
    cols += ncols; vals += ncols;
  }
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatSetOption(C,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);

  ptap->reuse = MAT_REUSE_MATRIX;
  PetscFunctionReturn(0);
}

PetscErrorCode MatTransposeMatMultNumeric_MPIAIJ_MPIAIJ(Mat P,Mat A,Mat C)
{
  PetscErrorCode      ierr;
  Mat_Merge_SeqsToMPI *merge;
  Mat_MPIAIJ          *p =(Mat_MPIAIJ*)P->data;
  Mat_SeqAIJ          *pd=(Mat_SeqAIJ*)(p->A)->data,*po=(Mat_SeqAIJ*)(p->B)->data;
  Mat_APMPI           *ptap;
  PetscInt            *adj;
  PetscInt            i,j,k,anz,pnz,row,*cj,nexta;
  MatScalar           *ada,*ca,valtmp;
  PetscInt            am=A->rmap->n,cm=C->rmap->n,pon=(p->B)->cmap->n;
  MPI_Comm            comm;
  PetscMPIInt         size,rank,taga,*len_s;
  PetscInt            *owners,proc,nrows,**buf_ri_k,**nextrow,**nextci;
  PetscInt            **buf_ri,**buf_rj;
  PetscInt            cnz=0,*bj_i,*bi,*bj,bnz,nextcj;  /* bi,bj,ba: local array of C(mpi mat) */
  MPI_Request         *s_waits,*r_waits;
  MPI_Status          *status;
  MatScalar           **abuf_r,*ba_i,*pA,*coa,*ba;
  const PetscScalar   *dummy;
  PetscInt            *ai,*aj,*coi,*coj,*poJ,*pdJ;
  Mat                 A_loc;
  Mat_SeqAIJ          *a_loc;

  PetscFunctionBegin;
  MatCheckProduct(C,3);
  ptap = (Mat_APMPI*)C->product->data;
  if (!ptap) SETERRQ(PetscObjectComm((PetscObject)C),PETSC_ERR_ARG_WRONGSTATE,"PtAP cannot be computed. Missing data");
  if (!ptap->A_loc) SETERRQ(PetscObjectComm((PetscObject)C),PETSC_ERR_ARG_WRONGSTATE,"PtA cannot be reused. Do not call MatProductClear()");
  ierr = PetscObjectGetComm((PetscObject)C,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRMPI(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRMPI(ierr);

  merge = ptap->merge;

  /* 2) compute numeric C_seq = P_loc^T*A_loc */
  /*------------------------------------------*/
  /* get data from symbolic products */
  coi    = merge->coi; coj = merge->coj;
  ierr   = PetscCalloc1(coi[pon]+1,&coa);CHKERRQ(ierr);
  bi     = merge->bi; bj = merge->bj;
  owners = merge->rowmap->range;
  ierr   = PetscCalloc1(bi[cm]+1,&ba);CHKERRQ(ierr);

  /* get A_loc by taking all local rows of A */
  A_loc = ptap->A_loc;
  ierr  = MatMPIAIJGetLocalMat(A,MAT_REUSE_MATRIX,&A_loc);CHKERRQ(ierr);
  a_loc = (Mat_SeqAIJ*)(A_loc)->data;
  ai    = a_loc->i;
  aj    = a_loc->j;

  /* trigger copy to CPU */
  ierr = MatSeqAIJGetArrayRead(p->A,&dummy);CHKERRQ(ierr);
  ierr = MatSeqAIJRestoreArrayRead(p->A,&dummy);CHKERRQ(ierr);
  ierr = MatSeqAIJGetArrayRead(p->B,&dummy);CHKERRQ(ierr);
  ierr = MatSeqAIJRestoreArrayRead(p->B,&dummy);CHKERRQ(ierr);
  for (i=0; i<am; i++) {
    anz = ai[i+1] - ai[i];
    adj = aj + ai[i];
    ada = a_loc->a + ai[i];

    /* 2-b) Compute Cseq = P_loc[i,:]^T*A[i,:] using outer product */
    /*-------------------------------------------------------------*/
    /* put the value into Co=(p->B)^T*A (off-diagonal part, send to others) */
    pnz = po->i[i+1] - po->i[i];
    poJ = po->j + po->i[i];
    pA  = po->a + po->i[i];
    for (j=0; j<pnz; j++) {
      row = poJ[j];
      cj  = coj + coi[row];
      ca  = coa + coi[row];
      /* perform sparse axpy */
      nexta  = 0;
      valtmp = pA[j];
      for (k=0; nexta<anz; k++) {
        if (cj[k] == adj[nexta]) {
          ca[k] += valtmp*ada[nexta];
          nexta++;
        }
      }
      ierr = PetscLogFlops(2.0*anz);CHKERRQ(ierr);
    }

    /* put the value into Cd (diagonal part) */
    pnz = pd->i[i+1] - pd->i[i];
    pdJ = pd->j + pd->i[i];
    pA  = pd->a + pd->i[i];
    for (j=0; j<pnz; j++) {
      row = pdJ[j];
      cj  = bj + bi[row];
      ca  = ba + bi[row];
      /* perform sparse axpy */
      nexta  = 0;
      valtmp = pA[j];
      for (k=0; nexta<anz; k++) {
        if (cj[k] == adj[nexta]) {
          ca[k] += valtmp*ada[nexta];
          nexta++;
        }
      }
      ierr = PetscLogFlops(2.0*anz);CHKERRQ(ierr);
    }
  }

  /* 3) send and recv matrix values coa */
  /*------------------------------------*/
  buf_ri = merge->buf_ri;
  buf_rj = merge->buf_rj;
  len_s  = merge->len_s;
  ierr   = PetscCommGetNewTag(comm,&taga);CHKERRQ(ierr);
  ierr   = PetscPostIrecvScalar(comm,taga,merge->nrecv,merge->id_r,merge->len_r,&abuf_r,&r_waits);CHKERRQ(ierr);

  ierr = PetscMalloc2(merge->nsend+1,&s_waits,size,&status);CHKERRQ(ierr);
  for (proc=0,k=0; proc<size; proc++) {
    if (!len_s[proc]) continue;
    i    = merge->owners_co[proc];
    ierr = MPI_Isend(coa+coi[i],len_s[proc],MPIU_MATSCALAR,proc,taga,comm,s_waits+k);CHKERRMPI(ierr);
    k++;
  }
  if (merge->nrecv) {ierr = MPI_Waitall(merge->nrecv,r_waits,status);CHKERRMPI(ierr);}
  if (merge->nsend) {ierr = MPI_Waitall(merge->nsend,s_waits,status);CHKERRMPI(ierr);}

  ierr = PetscFree2(s_waits,status);CHKERRQ(ierr);
  ierr = PetscFree(r_waits);CHKERRQ(ierr);
  ierr = PetscFree(coa);CHKERRQ(ierr);

  /* 4) insert local Cseq and received values into Cmpi */
  /*----------------------------------------------------*/
  ierr = PetscMalloc3(merge->nrecv,&buf_ri_k,merge->nrecv,&nextrow,merge->nrecv,&nextci);CHKERRQ(ierr);
  for (k=0; k<merge->nrecv; k++) {
    buf_ri_k[k] = buf_ri[k]; /* beginning of k-th recved i-structure */
    nrows       = *(buf_ri_k[k]);
    nextrow[k]  = buf_ri_k[k]+1;  /* next row number of k-th recved i-structure */
    nextci[k]   = buf_ri_k[k] + (nrows + 1); /* poins to the next i-structure of k-th recved i-structure  */
  }

  for (i=0; i<cm; i++) {
    row  = owners[rank] + i; /* global row index of C_seq */
    bj_i = bj + bi[i];  /* col indices of the i-th row of C */
    ba_i = ba + bi[i];
    bnz  = bi[i+1] - bi[i];
    /* add received vals into ba */
    for (k=0; k<merge->nrecv; k++) { /* k-th received message */
      /* i-th row */
      if (i == *nextrow[k]) {
        cnz    = *(nextci[k]+1) - *nextci[k];
        cj     = buf_rj[k] + *(nextci[k]);
        ca     = abuf_r[k] + *(nextci[k]);
        nextcj = 0;
        for (j=0; nextcj<cnz; j++) {
          if (bj_i[j] == cj[nextcj]) { /* bcol == ccol */
            ba_i[j] += ca[nextcj++];
          }
        }
        nextrow[k]++; nextci[k]++;
        ierr = PetscLogFlops(2.0*cnz);CHKERRQ(ierr);
      }
    }
    ierr = MatSetValues(C,1,&row,bnz,bj_i,ba_i,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = PetscFree(ba);CHKERRQ(ierr);
  ierr = PetscFree(abuf_r[0]);CHKERRQ(ierr);
  ierr = PetscFree(abuf_r);CHKERRQ(ierr);
  ierr = PetscFree3(buf_ri_k,nextrow,nextci);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatTransposeMatMultSymbolic_MPIAIJ_MPIAIJ(Mat P,Mat A,PetscReal fill,Mat C)
{
  PetscErrorCode      ierr;
  Mat                 A_loc;
  Mat_APMPI           *ptap;
  PetscFreeSpaceList  free_space=NULL,current_space=NULL;
  Mat_MPIAIJ          *p=(Mat_MPIAIJ*)P->data,*a=(Mat_MPIAIJ*)A->data;
  PetscInt            *pdti,*pdtj,*poti,*potj,*ptJ;
  PetscInt            nnz;
  PetscInt            *lnk,*owners_co,*coi,*coj,i,k,pnz,row;
  PetscInt            am  =A->rmap->n,pn=P->cmap->n;
  MPI_Comm            comm;
  PetscMPIInt         size,rank,tagi,tagj,*len_si,*len_s,*len_ri;
  PetscInt            **buf_rj,**buf_ri,**buf_ri_k;
  PetscInt            len,proc,*dnz,*onz,*owners;
  PetscInt            nzi,*bi,*bj;
  PetscInt            nrows,*buf_s,*buf_si,*buf_si_i,**nextrow,**nextci;
  MPI_Request         *swaits,*rwaits;
  MPI_Status          *sstatus,rstatus;
  Mat_Merge_SeqsToMPI *merge;
  PetscInt            *ai,*aj,*Jptr,anz,*prmap=p->garray,pon,nspacedouble=0,j;
  PetscReal           afill  =1.0,afill_tmp;
  PetscInt            rstart = P->cmap->rstart,rmax,aN=A->cmap->N,Armax;
  Mat_SeqAIJ          *a_loc;
  PetscTable          ta;
  MatType             mtype;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)A,&comm);CHKERRQ(ierr);
  /* check if matrix local sizes are compatible */
  if (A->rmap->rstart != P->rmap->rstart || A->rmap->rend != P->rmap->rend) SETERRQ4(comm,PETSC_ERR_ARG_SIZ,"Matrix local dimensions are incompatible, A (%D, %D) != P (%D,%D)",A->rmap->rstart,A->rmap->rend,P->rmap->rstart,P->rmap->rend);

  ierr = MPI_Comm_size(comm,&size);CHKERRMPI(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRMPI(ierr);

  /* create struct Mat_APMPI and attached it to C later */
  ierr = PetscNew(&ptap);CHKERRQ(ierr);

  /* get A_loc by taking all local rows of A */
  ierr = MatMPIAIJGetLocalMat(A,MAT_INITIAL_MATRIX,&A_loc);CHKERRQ(ierr);

  ptap->A_loc = A_loc;
  a_loc       = (Mat_SeqAIJ*)(A_loc)->data;
  ai          = a_loc->i;
  aj          = a_loc->j;

  /* determine symbolic Co=(p->B)^T*A - send to others */
  /*----------------------------------------------------*/
  ierr = MatGetSymbolicTranspose_SeqAIJ(p->A,&pdti,&pdtj);CHKERRQ(ierr);
  ierr = MatGetSymbolicTranspose_SeqAIJ(p->B,&poti,&potj);CHKERRQ(ierr);
  pon = (p->B)->cmap->n; /* total num of rows to be sent to other processors
                         >= (num of nonzero rows of C_seq) - pn */
  ierr   = PetscMalloc1(pon+1,&coi);CHKERRQ(ierr);
  coi[0] = 0;

  /* set initial free space to be fill*(nnz(p->B) + nnz(A)) */
  nnz           = PetscRealIntMultTruncate(fill,PetscIntSumTruncate(poti[pon],ai[am]));
  ierr          = PetscFreeSpaceGet(nnz,&free_space);CHKERRQ(ierr);
  current_space = free_space;

  /* create and initialize a linked list */
  ierr = PetscTableCreate(A->cmap->n + a->B->cmap->N,aN,&ta);CHKERRQ(ierr);
  MatRowMergeMax_SeqAIJ(a_loc,am,ta);
  ierr = PetscTableGetCount(ta,&Armax);CHKERRQ(ierr);

  ierr = PetscLLCondensedCreate_Scalable(Armax,&lnk);CHKERRQ(ierr);

  for (i=0; i<pon; i++) {
    pnz = poti[i+1] - poti[i];
    ptJ = potj + poti[i];
    for (j=0; j<pnz; j++) {
      row  = ptJ[j]; /* row of A_loc == col of Pot */
      anz  = ai[row+1] - ai[row];
      Jptr = aj + ai[row];
      /* add non-zero cols of AP into the sorted linked list lnk */
      ierr = PetscLLCondensedAddSorted_Scalable(anz,Jptr,lnk);CHKERRQ(ierr);
    }
    nnz = lnk[0];

    /* If free space is not available, double the total space in the list */
    if (current_space->local_remaining<nnz) {
      ierr = PetscFreeSpaceGet(PetscIntSumTruncate(nnz,current_space->total_array_size),&current_space);CHKERRQ(ierr);
      nspacedouble++;
    }

    /* Copy data into free space, and zero out denserows */
    ierr = PetscLLCondensedClean_Scalable(nnz,current_space->array,lnk);CHKERRQ(ierr);

    current_space->array           += nnz;
    current_space->local_used      += nnz;
    current_space->local_remaining -= nnz;

    coi[i+1] = coi[i] + nnz;
  }

  ierr = PetscMalloc1(coi[pon]+1,&coj);CHKERRQ(ierr);
  ierr = PetscFreeSpaceContiguous(&free_space,coj);CHKERRQ(ierr);
  ierr = PetscLLCondensedDestroy_Scalable(lnk);CHKERRQ(ierr); /* must destroy to get a new one for C */

  afill_tmp = (PetscReal)coi[pon]/(poti[pon] + ai[am]+1);
  if (afill_tmp > afill) afill = afill_tmp;

  /* send j-array (coj) of Co to other processors */
  /*----------------------------------------------*/
  /* determine row ownership */
  ierr = PetscNew(&merge);CHKERRQ(ierr);
  ierr = PetscLayoutCreate(comm,&merge->rowmap);CHKERRQ(ierr);

  merge->rowmap->n  = pn;
  merge->rowmap->bs = 1;

  ierr   = PetscLayoutSetUp(merge->rowmap);CHKERRQ(ierr);
  owners = merge->rowmap->range;

  /* determine the number of messages to send, their lengths */
  ierr = PetscCalloc1(size,&len_si);CHKERRQ(ierr);
  ierr = PetscCalloc1(size,&merge->len_s);CHKERRQ(ierr);

  len_s        = merge->len_s;
  merge->nsend = 0;

  ierr = PetscMalloc1(size+2,&owners_co);CHKERRQ(ierr);

  proc = 0;
  for (i=0; i<pon; i++) {
    while (prmap[i] >= owners[proc+1]) proc++;
    len_si[proc]++;  /* num of rows in Co to be sent to [proc] */
    len_s[proc] += coi[i+1] - coi[i];
  }

  len          = 0; /* max length of buf_si[] */
  owners_co[0] = 0;
  for (proc=0; proc<size; proc++) {
    owners_co[proc+1] = owners_co[proc] + len_si[proc];
    if (len_si[proc]) {
      merge->nsend++;
      len_si[proc] = 2*(len_si[proc] + 1);
      len         += len_si[proc];
    }
  }

  /* determine the number and length of messages to receive for coi and coj  */
  ierr = PetscGatherNumberOfMessages(comm,NULL,len_s,&merge->nrecv);CHKERRQ(ierr);
  ierr = PetscGatherMessageLengths2(comm,merge->nsend,merge->nrecv,len_s,len_si,&merge->id_r,&merge->len_r,&len_ri);CHKERRQ(ierr);

  /* post the Irecv and Isend of coj */
  ierr = PetscCommGetNewTag(comm,&tagj);CHKERRQ(ierr);
  ierr = PetscPostIrecvInt(comm,tagj,merge->nrecv,merge->id_r,merge->len_r,&buf_rj,&rwaits);CHKERRQ(ierr);
  ierr = PetscMalloc1(merge->nsend+1,&swaits);CHKERRQ(ierr);
  for (proc=0, k=0; proc<size; proc++) {
    if (!len_s[proc]) continue;
    i    = owners_co[proc];
    ierr = MPI_Isend(coj+coi[i],len_s[proc],MPIU_INT,proc,tagj,comm,swaits+k);CHKERRMPI(ierr);
    k++;
  }

  /* receives and sends of coj are complete */
  ierr = PetscMalloc1(size,&sstatus);CHKERRQ(ierr);
  for (i=0; i<merge->nrecv; i++) {
    PETSC_UNUSED PetscMPIInt icompleted;
    ierr = MPI_Waitany(merge->nrecv,rwaits,&icompleted,&rstatus);CHKERRMPI(ierr);
  }
  ierr = PetscFree(rwaits);CHKERRQ(ierr);
  if (merge->nsend) {ierr = MPI_Waitall(merge->nsend,swaits,sstatus);CHKERRMPI(ierr);}

  /* add received column indices into table to update Armax */
  /* Armax can be as large as aN if a P[row,:] is dense, see src/ksp/ksp/tutorials/ex56.c! */
  for (k=0; k<merge->nrecv; k++) {/* k-th received message */
    Jptr = buf_rj[k];
    for (j=0; j<merge->len_r[k]; j++) {
      ierr = PetscTableAdd(ta,*(Jptr+j)+1,1,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = PetscTableGetCount(ta,&Armax);CHKERRQ(ierr);
  /* printf("Armax %d, an %d + Bn %d = %d, aN %d\n",Armax,A->cmap->n,a->B->cmap->N,A->cmap->n+a->B->cmap->N,aN); */

  /* send and recv coi */
  /*-------------------*/
  ierr   = PetscCommGetNewTag(comm,&tagi);CHKERRQ(ierr);
  ierr   = PetscPostIrecvInt(comm,tagi,merge->nrecv,merge->id_r,len_ri,&buf_ri,&rwaits);CHKERRQ(ierr);
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
    nrows       = len_si[proc]/2 - 1;
    buf_si_i    = buf_si + nrows+1;
    buf_si[0]   = nrows;
    buf_si_i[0] = 0;
    nrows       = 0;
    for (i=owners_co[proc]; i<owners_co[proc+1]; i++) {
      nzi               = coi[i+1] - coi[i];
      buf_si_i[nrows+1] = buf_si_i[nrows] + nzi;  /* i-structure */
      buf_si[nrows+1]   = prmap[i] -owners[proc]; /* local row index */
      nrows++;
    }
    ierr = MPI_Isend(buf_si,len_si[proc],MPIU_INT,proc,tagi,comm,swaits+k);CHKERRMPI(ierr);
    k++;
    buf_si += len_si[proc];
  }
  i = merge->nrecv;
  while (i--) {
    PETSC_UNUSED PetscMPIInt icompleted;
    ierr = MPI_Waitany(merge->nrecv,rwaits,&icompleted,&rstatus);CHKERRMPI(ierr);
  }
  ierr = PetscFree(rwaits);CHKERRQ(ierr);
  if (merge->nsend) {ierr = MPI_Waitall(merge->nsend,swaits,sstatus);CHKERRMPI(ierr);}
  ierr = PetscFree(len_si);CHKERRQ(ierr);
  ierr = PetscFree(len_ri);CHKERRQ(ierr);
  ierr = PetscFree(swaits);CHKERRQ(ierr);
  ierr = PetscFree(sstatus);CHKERRQ(ierr);
  ierr = PetscFree(buf_s);CHKERRQ(ierr);

  /* compute the local portion of C (mpi mat) */
  /*------------------------------------------*/
  /* allocate bi array and free space for accumulating nonzero column info */
  ierr  = PetscMalloc1(pn+1,&bi);CHKERRQ(ierr);
  bi[0] = 0;

  /* set initial free space to be fill*(nnz(P) + nnz(AP)) */
  nnz           = PetscRealIntMultTruncate(fill,PetscIntSumTruncate(pdti[pn],PetscIntSumTruncate(poti[pon],ai[am])));
  ierr          = PetscFreeSpaceGet(nnz,&free_space);CHKERRQ(ierr);
  current_space = free_space;

  ierr = PetscMalloc3(merge->nrecv,&buf_ri_k,merge->nrecv,&nextrow,merge->nrecv,&nextci);CHKERRQ(ierr);
  for (k=0; k<merge->nrecv; k++) {
    buf_ri_k[k] = buf_ri[k]; /* beginning of k-th recved i-structure */
    nrows       = *buf_ri_k[k];
    nextrow[k]  = buf_ri_k[k] + 1;  /* next row number of k-th recved i-structure */
    nextci[k]   = buf_ri_k[k] + (nrows + 1); /* points to the next i-structure of k-th received i-structure  */
  }

  ierr = PetscLLCondensedCreate_Scalable(Armax,&lnk);CHKERRQ(ierr);
  ierr = MatPreallocateInitialize(comm,pn,A->cmap->n,dnz,onz);CHKERRQ(ierr);
  rmax = 0;
  for (i=0; i<pn; i++) {
    /* add pdt[i,:]*AP into lnk */
    pnz = pdti[i+1] - pdti[i];
    ptJ = pdtj + pdti[i];
    for (j=0; j<pnz; j++) {
      row  = ptJ[j];  /* row of AP == col of Pt */
      anz  = ai[row+1] - ai[row];
      Jptr = aj + ai[row];
      /* add non-zero cols of AP into the sorted linked list lnk */
      ierr = PetscLLCondensedAddSorted_Scalable(anz,Jptr,lnk);CHKERRQ(ierr);
    }

    /* add received col data into lnk */
    for (k=0; k<merge->nrecv; k++) { /* k-th received message */
      if (i == *nextrow[k]) { /* i-th row */
        nzi  = *(nextci[k]+1) - *nextci[k];
        Jptr = buf_rj[k] + *nextci[k];
        ierr = PetscLLCondensedAddSorted_Scalable(nzi,Jptr,lnk);CHKERRQ(ierr);
        nextrow[k]++; nextci[k]++;
      }
    }

    /* add missing diagonal entry */
    if (C->force_diagonals) {
      k = i + owners[rank]; /* column index */
      ierr = PetscLLCondensedAddSorted_Scalable(1,&k,lnk);CHKERRQ(ierr);
    }

    nnz = lnk[0];

    /* if free space is not available, make more free space */
    if (current_space->local_remaining<nnz) {
      ierr = PetscFreeSpaceGet(PetscIntSumTruncate(nnz,current_space->total_array_size),&current_space);CHKERRQ(ierr);
      nspacedouble++;
    }
    /* copy data into free space, then initialize lnk */
    ierr = PetscLLCondensedClean_Scalable(nnz,current_space->array,lnk);CHKERRQ(ierr);
    ierr = MatPreallocateSet(i+owners[rank],nnz,current_space->array,dnz,onz);CHKERRQ(ierr);

    current_space->array           += nnz;
    current_space->local_used      += nnz;
    current_space->local_remaining -= nnz;

    bi[i+1] = bi[i] + nnz;
    if (nnz > rmax) rmax = nnz;
  }
  ierr = PetscFree3(buf_ri_k,nextrow,nextci);CHKERRQ(ierr);

  ierr      = PetscMalloc1(bi[pn]+1,&bj);CHKERRQ(ierr);
  ierr      = PetscFreeSpaceContiguous(&free_space,bj);CHKERRQ(ierr);
  afill_tmp = (PetscReal)bi[pn]/(pdti[pn] + poti[pon] + ai[am]+1);
  if (afill_tmp > afill) afill = afill_tmp;
  ierr = PetscLLCondensedDestroy_Scalable(lnk);CHKERRQ(ierr);
  ierr = PetscTableDestroy(&ta);CHKERRQ(ierr);
  ierr = MatRestoreSymbolicTranspose_SeqAIJ(p->A,&pdti,&pdtj);CHKERRQ(ierr);
  ierr = MatRestoreSymbolicTranspose_SeqAIJ(p->B,&poti,&potj);CHKERRQ(ierr);

  /* create symbolic parallel matrix C - why cannot be assembled in Numeric part   */
  /*-------------------------------------------------------------------------------*/
  ierr = MatSetSizes(C,pn,A->cmap->n,PETSC_DETERMINE,PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = MatSetBlockSizes(C,PetscAbs(P->cmap->bs),PetscAbs(A->cmap->bs));CHKERRQ(ierr);
  ierr = MatGetType(A,&mtype);CHKERRQ(ierr);
  ierr = MatSetType(C,mtype);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(C,0,dnz,0,onz);CHKERRQ(ierr);
  ierr = MatPreallocateFinalize(dnz,onz);CHKERRQ(ierr);
  ierr = MatSetBlockSize(C,1);CHKERRQ(ierr);
  ierr = MatSetOption(C,MAT_NO_OFF_PROC_ENTRIES,PETSC_TRUE);CHKERRQ(ierr);
  for (i=0; i<pn; i++) {
    row  = i + rstart;
    nnz  = bi[i+1] - bi[i];
    Jptr = bj + bi[i];
    ierr = MatSetValues(C,1,&row,nnz,Jptr,NULL,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatSetOption(C,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);
  merge->bi        = bi;
  merge->bj        = bj;
  merge->coi       = coi;
  merge->coj       = coj;
  merge->buf_ri    = buf_ri;
  merge->buf_rj    = buf_rj;
  merge->owners_co = owners_co;

  /* attach the supporting struct to C for reuse */
  C->product->data    = ptap;
  C->product->destroy = MatDestroy_MPIAIJ_PtAP;
  ptap->merge         = merge;

  C->ops->mattransposemultnumeric = MatTransposeMatMultNumeric_MPIAIJ_MPIAIJ;

#if defined(PETSC_USE_INFO)
  if (bi[pn] != 0) {
    ierr = PetscInfo3(C,"Reallocs %D; Fill ratio: given %g needed %g.\n",nspacedouble,(double)fill,(double)afill);CHKERRQ(ierr);
    ierr = PetscInfo1(C,"Use MatTransposeMatMult(A,B,MatReuse,%g,&C) for best performance.\n",(double)afill);CHKERRQ(ierr);
  } else {
    ierr = PetscInfo(C,"Empty matrix product\n");CHKERRQ(ierr);
  }
#endif
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */
static PetscErrorCode MatProductSymbolic_AtB_MPIAIJ_MPIAIJ(Mat C)
{
  PetscErrorCode ierr;
  Mat_Product    *product = C->product;
  Mat            A=product->A,B=product->B;
  PetscReal      fill=product->fill;
  PetscBool      flg;

  PetscFunctionBegin;
  /* scalable */
  ierr = PetscStrcmp(product->alg,"scalable",&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = MatTransposeMatMultSymbolic_MPIAIJ_MPIAIJ(A,B,fill,C);CHKERRQ(ierr);
    goto next;
  }

  /* nonscalable */
  ierr = PetscStrcmp(product->alg,"nonscalable",&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = MatTransposeMatMultSymbolic_MPIAIJ_MPIAIJ_nonscalable(A,B,fill,C);CHKERRQ(ierr);
    goto next;
  }

  /* matmatmult */
  ierr = PetscStrcmp(product->alg,"at*b",&flg);CHKERRQ(ierr);
  if (flg) {
    Mat       At;
    Mat_APMPI *ptap;

    ierr = MatTranspose(A,MAT_INITIAL_MATRIX,&At);CHKERRQ(ierr);
    ierr = MatMatMultSymbolic_MPIAIJ_MPIAIJ(At,B,fill,C);CHKERRQ(ierr);
    ptap = (Mat_APMPI*)C->product->data;
    if (ptap) {
      ptap->Pt = At;
      C->product->destroy = MatDestroy_MPIAIJ_PtAP;
    }
    C->ops->transposematmultnumeric = MatTransposeMatMultNumeric_MPIAIJ_MPIAIJ_matmatmult;
    goto next;
  }

  /* backend general code */
  ierr = PetscStrcmp(product->alg,"backend",&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = MatProductSymbolic_MPIAIJBACKEND(C);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"MatProduct type is not supported");

next:
  C->ops->productnumeric = MatProductNumeric_AtB;
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */
/* Set options for MatMatMultxxx_MPIAIJ_MPIAIJ */
static PetscErrorCode MatProductSetFromOptions_MPIAIJ_AB(Mat C)
{
  PetscErrorCode ierr;
  Mat_Product    *product = C->product;
  Mat            A=product->A,B=product->B;
#if defined(PETSC_HAVE_HYPRE)
  const char     *algTypes[5] = {"scalable","nonscalable","seqmpi","backend","hypre"};
  PetscInt       nalg = 5;
#else
  const char     *algTypes[4] = {"scalable","nonscalable","seqmpi","backend",};
  PetscInt       nalg = 4;
#endif
  PetscInt       alg = 1; /* set nonscalable algorithm as default */
  PetscBool      flg;
  MPI_Comm       comm;

  PetscFunctionBegin;
  /* Check matrix local sizes */
  ierr = PetscObjectGetComm((PetscObject)C,&comm);CHKERRQ(ierr);
  if (A->cmap->rstart != B->rmap->rstart || A->cmap->rend != B->rmap->rend) SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Matrix local dimensions are incompatible, (%D, %D) != (%D,%D)",A->cmap->rstart,A->cmap->rend,B->rmap->rstart,B->rmap->rend);

  /* Set "nonscalable" as default algorithm */
  ierr = PetscStrcmp(C->product->alg,"default",&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = MatProductSetAlgorithm(C,(MatProductAlgorithm)algTypes[alg]);CHKERRQ(ierr);

    /* Set "scalable" as default if BN and local nonzeros of A and B are large */
    if (B->cmap->N > 100000) { /* may switch to scalable algorithm as default */
      MatInfo     Ainfo,Binfo;
      PetscInt    nz_local;
      PetscBool   alg_scalable_loc=PETSC_FALSE,alg_scalable;

      ierr = MatGetInfo(A,MAT_LOCAL,&Ainfo);CHKERRQ(ierr);
      ierr = MatGetInfo(B,MAT_LOCAL,&Binfo);CHKERRQ(ierr);
      nz_local = (PetscInt)(Ainfo.nz_allocated + Binfo.nz_allocated);

      if (B->cmap->N > product->fill*nz_local) alg_scalable_loc = PETSC_TRUE;
      ierr = MPIU_Allreduce(&alg_scalable_loc,&alg_scalable,1,MPIU_BOOL,MPI_LOR,comm);CHKERRMPI(ierr);

      if (alg_scalable) {
        alg  = 0; /* scalable algorithm would 50% slower than nonscalable algorithm */
        ierr = MatProductSetAlgorithm(C,(MatProductAlgorithm)algTypes[alg]);CHKERRQ(ierr);
        ierr = PetscInfo2(B,"Use scalable algorithm, BN %D, fill*nz_allocated %g\n",B->cmap->N,product->fill*nz_local);CHKERRQ(ierr);
      }
    }
  }

  /* Get runtime option */
  if (product->api_user) {
    ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)C),((PetscObject)C)->prefix,"MatMatMult","Mat");CHKERRQ(ierr);
    ierr = PetscOptionsEList("-matmatmult_via","Algorithmic approach","MatMatMult",algTypes,nalg,algTypes[alg],&alg,&flg);CHKERRQ(ierr);
    ierr = PetscOptionsEnd();CHKERRQ(ierr);
  } else {
    ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)C),((PetscObject)C)->prefix,"MatProduct_AB","Mat");CHKERRQ(ierr);
    ierr = PetscOptionsEList("-matproduct_ab_via","Algorithmic approach","MatMatMult",algTypes,nalg,algTypes[alg],&alg,&flg);CHKERRQ(ierr);
    ierr = PetscOptionsEnd();CHKERRQ(ierr);
  }
  if (flg) {
    ierr = MatProductSetAlgorithm(C,(MatProductAlgorithm)algTypes[alg]);CHKERRQ(ierr);
  }

  C->ops->productsymbolic = MatProductSymbolic_AB_MPIAIJ_MPIAIJ;
  PetscFunctionReturn(0);
}

/* Set options for MatTransposeMatMultXXX_MPIAIJ_MPIAIJ */
static PetscErrorCode MatProductSetFromOptions_MPIAIJ_AtB(Mat C)
{
  PetscErrorCode ierr;
  Mat_Product    *product = C->product;
  Mat            A=product->A,B=product->B;
  const char     *algTypes[4] = {"scalable","nonscalable","at*b","backend"};
  PetscInt       nalg = 4;
  PetscInt       alg = 1; /* set default algorithm  */
  PetscBool      flg;
  MPI_Comm       comm;

  PetscFunctionBegin;
  /* Check matrix local sizes */
  ierr = PetscObjectGetComm((PetscObject)C,&comm);CHKERRQ(ierr);
  if (A->rmap->rstart != B->rmap->rstart || A->rmap->rend != B->rmap->rend) SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Matrix local dimensions are incompatible, A (%D, %D) != B (%D,%D)",A->rmap->rstart,A->rmap->rend,B->rmap->rstart,B->rmap->rend);

  /* Set default algorithm */
  ierr = PetscStrcmp(C->product->alg,"default",&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = MatProductSetAlgorithm(C,(MatProductAlgorithm)algTypes[alg]);CHKERRQ(ierr);
  }

  /* Set "scalable" as default if BN and local nonzeros of A and B are large */
  if (alg && B->cmap->N > 100000) { /* may switch to scalable algorithm as default */
    MatInfo     Ainfo,Binfo;
    PetscInt    nz_local;
    PetscBool   alg_scalable_loc=PETSC_FALSE,alg_scalable;

    ierr = MatGetInfo(A,MAT_LOCAL,&Ainfo);CHKERRQ(ierr);
    ierr = MatGetInfo(B,MAT_LOCAL,&Binfo);CHKERRQ(ierr);
    nz_local = (PetscInt)(Ainfo.nz_allocated + Binfo.nz_allocated);

    if (B->cmap->N > product->fill*nz_local) alg_scalable_loc = PETSC_TRUE;
    ierr = MPIU_Allreduce(&alg_scalable_loc,&alg_scalable,1,MPIU_BOOL,MPI_LOR,comm);CHKERRMPI(ierr);

    if (alg_scalable) {
      alg  = 0; /* scalable algorithm would 50% slower than nonscalable algorithm */
      ierr = MatProductSetAlgorithm(C,(MatProductAlgorithm)algTypes[alg]);CHKERRQ(ierr);
      ierr = PetscInfo2(B,"Use scalable algorithm, BN %D, fill*nz_allocated %g\n",B->cmap->N,product->fill*nz_local);CHKERRQ(ierr);
    }
  }

  /* Get runtime option */
  if (product->api_user) {
    ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)C),((PetscObject)C)->prefix,"MatTransposeMatMult","Mat");CHKERRQ(ierr);
    ierr = PetscOptionsEList("-mattransposematmult_via","Algorithmic approach","MatTransposeMatMult",algTypes,nalg,algTypes[alg],&alg,&flg);CHKERRQ(ierr);
    ierr = PetscOptionsEnd();CHKERRQ(ierr);
  } else {
    ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)C),((PetscObject)C)->prefix,"MatProduct_AtB","Mat");CHKERRQ(ierr);
    ierr = PetscOptionsEList("-matproduct_atb_via","Algorithmic approach","MatTransposeMatMult",algTypes,nalg,algTypes[alg],&alg,&flg);CHKERRQ(ierr);
    ierr = PetscOptionsEnd();CHKERRQ(ierr);
  }
  if (flg) {
    ierr = MatProductSetAlgorithm(C,(MatProductAlgorithm)algTypes[alg]);CHKERRQ(ierr);
  }

  C->ops->productsymbolic = MatProductSymbolic_AtB_MPIAIJ_MPIAIJ;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatProductSetFromOptions_MPIAIJ_PtAP(Mat C)
{
  PetscErrorCode ierr;
  Mat_Product    *product = C->product;
  Mat            A=product->A,P=product->B;
  MPI_Comm       comm;
  PetscBool      flg;
  PetscInt       alg=1; /* set default algorithm */
#if !defined(PETSC_HAVE_HYPRE)
  const char     *algTypes[5] = {"scalable","nonscalable","allatonce","allatonce_merged","backend"};
  PetscInt       nalg=5;
#else
  const char     *algTypes[6] = {"scalable","nonscalable","allatonce","allatonce_merged","backend","hypre"};
  PetscInt       nalg=6;
#endif
  PetscInt       pN=P->cmap->N;

  PetscFunctionBegin;
  /* Check matrix local sizes */
  ierr = PetscObjectGetComm((PetscObject)C,&comm);CHKERRQ(ierr);
  if (A->rmap->rstart != P->rmap->rstart || A->rmap->rend != P->rmap->rend) SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Matrix local dimensions are incompatible, Arow (%D, %D) != Prow (%D,%D)",A->rmap->rstart,A->rmap->rend,P->rmap->rstart,P->rmap->rend);
  if (A->cmap->rstart != P->rmap->rstart || A->cmap->rend != P->rmap->rend) SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Matrix local dimensions are incompatible, Acol (%D, %D) != Prow (%D,%D)",A->cmap->rstart,A->cmap->rend,P->rmap->rstart,P->rmap->rend);

  /* Set "nonscalable" as default algorithm */
  ierr = PetscStrcmp(C->product->alg,"default",&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = MatProductSetAlgorithm(C,(MatProductAlgorithm)algTypes[alg]);CHKERRQ(ierr);

    /* Set "scalable" as default if BN and local nonzeros of A and B are large */
    if (pN > 100000) {
      MatInfo     Ainfo,Pinfo;
      PetscInt    nz_local;
      PetscBool   alg_scalable_loc=PETSC_FALSE,alg_scalable;

      ierr = MatGetInfo(A,MAT_LOCAL,&Ainfo);CHKERRQ(ierr);
      ierr = MatGetInfo(P,MAT_LOCAL,&Pinfo);CHKERRQ(ierr);
      nz_local = (PetscInt)(Ainfo.nz_allocated + Pinfo.nz_allocated);

      if (pN > product->fill*nz_local) alg_scalable_loc = PETSC_TRUE;
      ierr = MPIU_Allreduce(&alg_scalable_loc,&alg_scalable,1,MPIU_BOOL,MPI_LOR,comm);CHKERRMPI(ierr);

      if (alg_scalable) {
        alg = 0; /* scalable algorithm would 50% slower than nonscalable algorithm */
        ierr = MatProductSetAlgorithm(C,(MatProductAlgorithm)algTypes[alg]);CHKERRQ(ierr);
      }
    }
  }

  /* Get runtime option */
  if (product->api_user) {
    ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)C),((PetscObject)C)->prefix,"MatPtAP","Mat");CHKERRQ(ierr);
    ierr = PetscOptionsEList("-matptap_via","Algorithmic approach","MatPtAP",algTypes,nalg,algTypes[alg],&alg,&flg);CHKERRQ(ierr);
    ierr = PetscOptionsEnd();CHKERRQ(ierr);
  } else {
    ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)C),((PetscObject)C)->prefix,"MatProduct_PtAP","Mat");CHKERRQ(ierr);
    ierr = PetscOptionsEList("-matproduct_ptap_via","Algorithmic approach","MatPtAP",algTypes,nalg,algTypes[alg],&alg,&flg);CHKERRQ(ierr);
    ierr = PetscOptionsEnd();CHKERRQ(ierr);
  }
  if (flg) {
    ierr = MatProductSetAlgorithm(C,(MatProductAlgorithm)algTypes[alg]);CHKERRQ(ierr);
  }

  C->ops->productsymbolic = MatProductSymbolic_PtAP_MPIAIJ_MPIAIJ;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatProductSetFromOptions_MPIAIJ_RARt(Mat C)
{
  Mat_Product *product = C->product;
  Mat         A = product->A,R=product->B;

  PetscFunctionBegin;
  /* Check matrix local sizes */
  if (A->cmap->n != R->cmap->n || A->rmap->n != R->cmap->n) SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Matrix local dimensions are incompatible, A local (%D, %D), R local (%D,%D)",A->rmap->n,A->rmap->n,R->rmap->n,R->cmap->n);

  C->ops->productsymbolic = MatProductSymbolic_RARt_MPIAIJ_MPIAIJ;
  PetscFunctionReturn(0);
}

/*
 Set options for ABC = A*B*C = A*(B*C); ABC's algorithm must be chosen from AB's algorithm
*/
static PetscErrorCode MatProductSetFromOptions_MPIAIJ_ABC(Mat C)
{
  PetscErrorCode ierr;
  Mat_Product    *product = C->product;
  PetscBool      flg = PETSC_FALSE;
  PetscInt       alg = 1; /* default algorithm */
  const char     *algTypes[3] = {"scalable","nonscalable","seqmpi"};
  PetscInt       nalg = 3;

  PetscFunctionBegin;
  /* Set default algorithm */
  ierr = PetscStrcmp(C->product->alg,"default",&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = MatProductSetAlgorithm(C,(MatProductAlgorithm)algTypes[alg]);CHKERRQ(ierr);
  }

  /* Get runtime option */
  if (product->api_user) {
    ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)C),((PetscObject)C)->prefix,"MatMatMatMult","Mat");CHKERRQ(ierr);
    ierr = PetscOptionsEList("-matmatmatmult_via","Algorithmic approach","MatMatMatMult",algTypes,nalg,algTypes[alg],&alg,&flg);CHKERRQ(ierr);
    ierr = PetscOptionsEnd();CHKERRQ(ierr);
  } else {
    ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)C),((PetscObject)C)->prefix,"MatProduct_ABC","Mat");CHKERRQ(ierr);
    ierr = PetscOptionsEList("-matproduct_abc_via","Algorithmic approach","MatProduct_ABC",algTypes,nalg,algTypes[alg],&alg,&flg);CHKERRQ(ierr);
    ierr = PetscOptionsEnd();CHKERRQ(ierr);
  }
  if (flg) {
    ierr = MatProductSetAlgorithm(C,(MatProductAlgorithm)algTypes[alg]);CHKERRQ(ierr);
  }

  C->ops->matmatmultsymbolic = MatMatMatMultSymbolic_MPIAIJ_MPIAIJ_MPIAIJ;
  C->ops->productsymbolic    = MatProductSymbolic_ABC;
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatProductSetFromOptions_MPIAIJ(Mat C)
{
  PetscErrorCode ierr;
  Mat_Product    *product = C->product;

  PetscFunctionBegin;
  switch (product->type) {
  case MATPRODUCT_AB:
    ierr = MatProductSetFromOptions_MPIAIJ_AB(C);CHKERRQ(ierr);
    break;
  case MATPRODUCT_AtB:
    ierr = MatProductSetFromOptions_MPIAIJ_AtB(C);CHKERRQ(ierr);
    break;
  case MATPRODUCT_PtAP:
    ierr = MatProductSetFromOptions_MPIAIJ_PtAP(C);CHKERRQ(ierr);
    break;
  case MATPRODUCT_RARt:
    ierr = MatProductSetFromOptions_MPIAIJ_RARt(C);CHKERRQ(ierr);
    break;
  case MATPRODUCT_ABC:
    ierr = MatProductSetFromOptions_MPIAIJ_ABC(C);CHKERRQ(ierr);
    break;
  default:
    break;
  }
  PetscFunctionReturn(0);
}
