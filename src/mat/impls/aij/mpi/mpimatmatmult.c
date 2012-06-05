
/*
  Defines matrix-matrix product routines for pairs of MPIAIJ matrices
          C = A * B
*/
#include <../src/mat/impls/aij/seq/aij.h> /*I "petscmat.h" I*/
#include <../src/mat/utils/freespace.h>
#include <../src/mat/impls/aij/mpi/mpiaij.h>
#include <petscbt.h>
#include <../src/mat/impls/dense/mpi/mpidense.h>

#undef __FUNCT__
#define __FUNCT__ "MatMatMult_MPIAIJ_MPIAIJ"
PetscErrorCode MatMatMult_MPIAIJ_MPIAIJ(Mat A,Mat B,MatReuse scall,PetscReal fill, Mat *C) 
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (scall == MAT_INITIAL_MATRIX){
    ierr = PetscLogEventBegin(MAT_MatMultSymbolic,A,B,0,0);CHKERRQ(ierr);
    ierr = MatMatMultSymbolic_MPIAIJ_MPIAIJ(A,B,fill,C);CHKERRQ(ierr);
    ierr = PetscLogEventEnd(MAT_MatMultSymbolic,A,B,0,0);CHKERRQ(ierr);   
  }

  ierr = PetscLogEventBegin(MAT_MatMultNumeric,A,B,0,0);CHKERRQ(ierr); 
  ierr = (*(*C)->ops->matmultnumeric)(A,B,*C);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_MatMultNumeric,A,B,0,0);CHKERRQ(ierr); 
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatDestroy_MPIAIJ_MatMatMult"
PetscErrorCode MatDestroy_MPIAIJ_MatMatMult(Mat A)
{
  PetscErrorCode ierr;
  Mat_MPIAIJ     *a=(Mat_MPIAIJ*)A->data;
  Mat_PtAPMPI    *ptap=a->ptap;

  PetscFunctionBegin;
  ierr = PetscFree2(ptap->startsj_s,ptap->startsj_r);CHKERRQ(ierr);
  ierr = PetscFree(ptap->bufa);CHKERRQ(ierr);
  ierr = MatDestroy(&ptap->P_loc);CHKERRQ(ierr);
  ierr = MatDestroy(&ptap->P_oth);CHKERRQ(ierr);
  ierr = PetscFree(ptap->api);CHKERRQ(ierr);
  ierr = PetscFree(ptap->apj);CHKERRQ(ierr);
  ierr = PetscFree(ptap->apa);CHKERRQ(ierr);
  ierr = ptap->destroy(A);CHKERRQ(ierr);
  ierr = PetscFree(ptap);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatDuplicate_MPIAIJ_MatMatMult"
PetscErrorCode MatDuplicate_MPIAIJ_MatMatMult(Mat A, MatDuplicateOption op, Mat *M)
{
  PetscErrorCode     ierr;
  Mat_MPIAIJ         *a=(Mat_MPIAIJ*)A->data;
  Mat_PtAPMPI        *ptap=a->ptap;
  
  PetscFunctionBegin;
  ierr = (*ptap->duplicate)(A,op,M);CHKERRQ(ierr);
  (*M)->ops->destroy   = ptap->destroy;   /* = MatDestroy_MPIAIJ, *M doesn't duplicate A's special structure! */
  (*M)->ops->duplicate = ptap->duplicate; /* = MatDuplicate_MPIAIJ */
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMatMultNumeric_MPIAIJ_MPIAIJ"
PetscErrorCode MatMatMultNumeric_MPIAIJ_MPIAIJ(Mat A,Mat P,Mat C)
{
  PetscErrorCode     ierr;
  Mat_MPIAIJ         *a=(Mat_MPIAIJ*)A->data,*c=(Mat_MPIAIJ*)C->data;
  Mat_SeqAIJ         *ad=(Mat_SeqAIJ*)(a->A)->data,*ao=(Mat_SeqAIJ*)(a->B)->data;
  Mat_SeqAIJ         *cd=(Mat_SeqAIJ*)(c->A)->data,*co=(Mat_SeqAIJ*)(c->B)->data;
  PetscInt           *adi=ad->i,*adj,*aoi=ao->i,*aoj;
  PetscScalar        *ada,*aoa,*cda=cd->a,*coa=co->a;
  Mat_SeqAIJ         *p_loc,*p_oth; 
  PetscInt           *pi_loc,*pj_loc,*pi_oth,*pj_oth,*pj;
  PetscScalar        *pa_loc,*pa_oth,*pa,*apa,valtmp,*ca;
  PetscInt           cm=C->rmap->n,anz,pnz;
  Mat_PtAPMPI        *ptap=c->ptap;
  PetscInt           *api,*apj,*apJ,i,j,k,row;
  PetscInt           cstart=C->cmap->rstart;
  PetscInt           cdnz,conz,k0,k1;

  PetscFunctionBegin;
  /* 1) get P_oth = ptap->P_oth  and P_loc = ptap->P_loc */
  /*-----------------------------------------------------*/ 
  /* update numerical values of P_oth and P_loc */
  ierr = MatGetBrowsOfAoCols_MPIAIJ(A,P,MAT_REUSE_MATRIX,&ptap->startsj_s,&ptap->startsj_r,&ptap->bufa,&ptap->P_oth);CHKERRQ(ierr);  
  ierr = MatMPIAIJGetLocalMat(P,MAT_REUSE_MATRIX,&ptap->P_loc);CHKERRQ(ierr);

  /* 2) compute numeric C_loc = A_loc*P = Ad*P_loc + Ao*P_oth */
  /*----------------------------------------------------------*/
  /* get data from symbolic products */
  p_loc = (Mat_SeqAIJ*)(ptap->P_loc)->data;
  p_oth = (Mat_SeqAIJ*)(ptap->P_oth)->data;
  pi_loc=p_loc->i; pj_loc=p_loc->j; pa_loc=p_loc->a;   
  pi_oth=p_oth->i; pj_oth=p_oth->j; pa_oth=p_oth->a;
  
  /* get apa for storing dense row A[i,:]*P */ 
  apa = ptap->apa;

  api = ptap->api;
  apj = ptap->apj; 
  for (i=0; i<cm; i++) {
    /* diagonal portion of A */
    anz = adi[i+1] - adi[i];
    adj = ad->j + adi[i];
    ada = ad->a + adi[i];
    for (j=0; j<anz; j++) {
      row = adj[j]; 
      pnz = pi_loc[row+1] - pi_loc[row];
      pj  = pj_loc + pi_loc[row];
      pa  = pa_loc + pi_loc[row];

      /* perform dense axpy */
      valtmp = ada[j];
      for (k=0; k<pnz; k++){
        apa[pj[k]] += valtmp*pa[k];
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

      /* perform dense axpy */
      valtmp = aoa[j];
      for (k=0; k<pnz; k++){
        apa[pj[k]] += valtmp*pa[k];
      }
      ierr = PetscLogFlops(2.0*pnz);CHKERRQ(ierr);
    }

    /* set values in C */
    apJ = apj + api[i];
    cdnz = cd->i[i+1] - cd->i[i]; 
    conz = co->i[i+1] - co->i[i];

    /* 1st off-diagoanl part of C */
    ca = coa + co->i[i];
    k  = 0;
    for (k0=0; k0<conz; k0++){
      if (apJ[k] >= cstart) break;
      ca[k0]      = apa[apJ[k]]; 
      apa[apJ[k]] = 0.0;
      k++;
    }

    /* diagonal part of C */
    ca = cda + cd->i[i];
    for (k1=0; k1<cdnz; k1++){
      ca[k1]      = apa[apJ[k]]; 
      apa[apJ[k]] = 0.0;
      k++;
    }

    /* 2nd off-diagoanl part of C */
    ca = coa + co->i[i];
    for (; k0<conz; k0++){
      ca[k0]      = apa[apJ[k]]; 
      apa[apJ[k]] = 0.0;
      k++;
    }
  } 
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMatMultSymbolic_MPIAIJ_MPIAIJ"
PetscErrorCode MatMatMultSymbolic_MPIAIJ_MPIAIJ(Mat A,Mat P,PetscReal fill,Mat *C)
{
  PetscErrorCode       ierr;
  MPI_Comm             comm=((PetscObject)A)->comm;
  Mat                  Cmpi; 
  Mat_PtAPMPI          *ptap;
  PetscFreeSpaceList   free_space=PETSC_NULL,current_space=PETSC_NULL;
  Mat_MPIAIJ           *a=(Mat_MPIAIJ*)A->data,*c;
  Mat_SeqAIJ           *ad=(Mat_SeqAIJ*)(a->A)->data,*ao=(Mat_SeqAIJ*)(a->B)->data,*p_loc,*p_oth;
  PetscInt             *pi_loc,*pj_loc,*pi_oth,*pj_oth,*dnz,*onz;
  PetscInt             *adi=ad->i,*adj=ad->j,*aoi=ao->i,*aoj=ao->j,rstart=A->rmap->rstart; 
  PetscInt             *lnk,i,pnz,row,*api,*apj,*Jptr,apnz,nspacedouble=0,j,nzi;
  PetscInt             am=A->rmap->n,pN=P->cmap->N,pn=P->cmap->n,pm=P->rmap->n;  
  PetscBT              lnkbt;
  PetscScalar          *apa;
  PetscReal            afill;
  PetscBool            scalable=PETSC_FALSE;
  PetscInt             nlnk_max,armax,prmax;

  PetscFunctionBegin;
  if (A->cmap->rstart != P->rmap->rstart || A->cmap->rend != P->rmap->rend){
    SETERRQ4(comm,PETSC_ERR_ARG_SIZ,"Matrix local dimensions are incompatible, (%D, %D) != (%D,%D)",A->cmap->rstart,A->cmap->rend,P->rmap->rstart,P->rmap->rend);
  } 

  ierr = PetscObjectOptionsBegin((PetscObject)A);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-matmatmult_scalable","Use a scalable but slower C=A*B","",scalable,&scalable,PETSC_NULL);CHKERRQ(ierr);
    if (scalable){
      ierr = MatMatMultSymbolic_MPIAIJ_MPIAIJ_Scalable(A,P,fill,C);;CHKERRQ(ierr);
      PetscFunctionReturn(0);
    }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  /* create struct Mat_PtAPMPI and attached it to C later */
  ierr = PetscNew(Mat_PtAPMPI,&ptap);CHKERRQ(ierr); 

  /* get P_oth by taking rows of P (= non-zero cols of local A) from other processors */
  ierr = MatGetBrowsOfAoCols_MPIAIJ(A,P,MAT_INITIAL_MATRIX,&ptap->startsj_s,&ptap->startsj_r,&ptap->bufa,&ptap->P_oth);CHKERRQ(ierr);

  /* get P_loc by taking all local rows of P */
  ierr = MatMPIAIJGetLocalMat(P,MAT_INITIAL_MATRIX,&ptap->P_loc);CHKERRQ(ierr);

  p_loc = (Mat_SeqAIJ*)(ptap->P_loc)->data; 
  p_oth = (Mat_SeqAIJ*)(ptap->P_oth)->data;
  pi_loc = p_loc->i; pj_loc = p_loc->j; 
  pi_oth = p_oth->i; pj_oth = p_oth->j;

  /* first, compute symbolic AP = A_loc*P = A_diag*P_loc + A_off*P_oth */
  /*-------------------------------------------------------------------*/
  ierr  = PetscMalloc((am+2)*sizeof(PetscInt),&api);CHKERRQ(ierr);
  ptap->api = api;
  api[0]    = 0;

  /* create and initialize a linked list */
  armax = ad->rmax+ao->rmax;
  prmax = PetscMax(p_loc->rmax,p_oth->rmax);
  nlnk_max = armax*prmax;
  if (!nlnk_max || nlnk_max > pN) nlnk_max = pN;
  ierr = PetscLLCondensedCreate(nlnk_max,pN,&lnk,&lnkbt);CHKERRQ(ierr);

  /* Initial FreeSpace size is fill*(nnz(A)+nnz(P)) */
  ierr = PetscFreeSpaceGet((PetscInt)(fill*(adi[am]+aoi[am]+pi_loc[pm])),&free_space);CHKERRQ(ierr);
  current_space = free_space; 

  ierr = MatPreallocateInitialize(comm,am,pn,dnz,onz);CHKERRQ(ierr);
  for (i=0; i<am; i++) {
    apnz = 0;
    /* diagonal portion of A */
    nzi = adi[i+1] - adi[i];
    for (j=0; j<nzi; j++){
      row = *adj++; 
      pnz = pi_loc[row+1] - pi_loc[row];
      Jptr  = pj_loc + pi_loc[row];
      /* add non-zero cols of P into the sorted linked list lnk */
      ierr = PetscLLCondensedAddSorted(pnz,Jptr,lnk,lnkbt);CHKERRQ(ierr);
    }
    /* off-diagonal portion of A */
    nzi = aoi[i+1] - aoi[i];
    for (j=0; j<nzi; j++){   
      row = *aoj++; 
      pnz = pi_oth[row+1] - pi_oth[row];
      Jptr  = pj_oth + pi_oth[row];  
      ierr = PetscLLCondensedAddSorted(pnz,Jptr,lnk,lnkbt);CHKERRQ(ierr);
    }

    apnz     = lnk[0];
    api[i+1] = api[i] + apnz;

    /* if free space is not available, double the total space in the list */
    if (current_space->local_remaining<apnz) {
      ierr = PetscFreeSpaceGet(apnz+current_space->total_array_size,&current_space);CHKERRQ(ierr);
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
  ierr = PetscMalloc((api[am]+1)*sizeof(PetscInt),&ptap->apj);CHKERRQ(ierr);
  apj  = ptap->apj;
  ierr = PetscFreeSpaceContiguous(&free_space,ptap->apj);CHKERRQ(ierr);
  ierr = PetscLLDestroy(lnk,lnkbt);CHKERRQ(ierr);

  /* malloc apa to store dense row A[i,:]*P */ 
  ierr = PetscMalloc(pN*sizeof(PetscScalar),&apa);CHKERRQ(ierr);
  ierr = PetscMemzero(apa,pN*sizeof(PetscScalar));CHKERRQ(ierr);
  ptap->apa = apa;

  /* create and assemble symbolic parallel matrix Cmpi */
  /*----------------------------------------------------*/
  ierr = MatCreate(comm,&Cmpi);CHKERRQ(ierr);
  ierr = MatSetSizes(Cmpi,am,pn,PETSC_DETERMINE,PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = MatSetBlockSizes(Cmpi,A->rmap->bs,P->cmap->bs);CHKERRQ(ierr);

  ierr = MatSetType(Cmpi,MATMPIAIJ);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(Cmpi,0,dnz,0,onz);CHKERRQ(ierr);
  ierr = MatPreallocateFinalize(dnz,onz);CHKERRQ(ierr);
  for (i=0; i<am; i++){
    row  = i + rstart;
    apnz = api[i+1] - api[i];
    ierr = MatSetValues(Cmpi,1,&row,apnz,apj,apa,INSERT_VALUES);CHKERRQ(ierr);
    apj += apnz;
  }
  ierr = MatAssemblyBegin(Cmpi,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Cmpi,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ptap->destroy             = Cmpi->ops->destroy;
  ptap->duplicate           = Cmpi->ops->duplicate;
  Cmpi->ops->matmultnumeric = MatMatMultNumeric_MPIAIJ_MPIAIJ;
  Cmpi->ops->destroy        = MatDestroy_MPIAIJ_MatMatMult;  
  Cmpi->ops->duplicate      = MatDuplicate_MPIAIJ_MatMatMult;

  /* attach the supporting struct to Cmpi for reuse */
  c = (Mat_MPIAIJ*)Cmpi->data;
  c->ptap  = ptap;
  
  *C = Cmpi;

  /* set MatInfo */
  afill = (PetscReal)api[am]/(adi[am]+aoi[am]+pi_loc[pm]+1) + 1.e-5;
  if (afill < 1.0) afill = 1.0;
  Cmpi->info.mallocs           = nspacedouble;
  Cmpi->info.fill_ratio_given  = fill;               
  Cmpi->info.fill_ratio_needed = afill; 

#if defined(PETSC_USE_INFO)
  if (api[am]) {
    ierr = PetscInfo3(Cmpi,"Reallocs %D; Fill ratio: given %G needed %G.\n",nspacedouble,fill,afill);CHKERRQ(ierr);
    ierr = PetscInfo1(Cmpi,"Use MatMatMult(A,B,MatReuse,%G,&C) for best performance.;\n",afill);CHKERRQ(ierr);
  } else {
    ierr = PetscInfo(Cmpi,"Empty matrix product\n");CHKERRQ(ierr);
  }
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMatMult_MPIAIJ_MPIDense"
PetscErrorCode MatMatMult_MPIAIJ_MPIDense(Mat A,Mat B,MatReuse scall,PetscReal fill,Mat *C) 
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (scall == MAT_INITIAL_MATRIX){
    ierr = MatMatMultSymbolic_MPIAIJ_MPIDense(A,B,fill,C);CHKERRQ(ierr);
  }  
  ierr = MatMatMultNumeric_MPIAIJ_MPIDense(A,B,*C);CHKERRQ(ierr);  
  PetscFunctionReturn(0);
}

typedef struct {
  Mat         workB;
  PetscScalar *rvalues,*svalues;
  MPI_Request *rwaits,*swaits;
} MPIAIJ_MPIDense;

#undef __FUNCT__
#define __FUNCT__ "MatMPIAIJ_MPIDenseDestroy"
PetscErrorCode MatMPIAIJ_MPIDenseDestroy(void *ctx)
{
  MPIAIJ_MPIDense *contents = (MPIAIJ_MPIDense*) ctx;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = MatDestroy(&contents->workB);CHKERRQ(ierr);
  ierr = PetscFree4(contents->rvalues,contents->svalues,contents->rwaits,contents->swaits);CHKERRQ(ierr);
  ierr = PetscFree(contents);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMatMultSymbolic_MPIAIJ_MPIDense"
PetscErrorCode MatMatMultSymbolic_MPIAIJ_MPIDense(Mat A,Mat B,PetscReal fill,Mat *C)
{
  PetscErrorCode         ierr;
  Mat_MPIAIJ             *aij = (Mat_MPIAIJ*) A->data;
  PetscInt               nz = aij->B->cmap->n;
  PetscContainer         container;
  MPIAIJ_MPIDense        *contents;
  VecScatter             ctx = aij->Mvctx;
  VecScatter_MPI_General *from = (VecScatter_MPI_General*) ctx->fromdata;
  VecScatter_MPI_General *to   = ( VecScatter_MPI_General*) ctx->todata;
  PetscInt               m=A->rmap->n,n=B->cmap->n;

  PetscFunctionBegin;
  ierr = MatCreate(((PetscObject)B)->comm,C);CHKERRQ(ierr);
  ierr = MatSetSizes(*C,m,n,A->rmap->N,B->cmap->N);CHKERRQ(ierr);
  ierr = MatSetBlockSizes(*C,A->rmap->bs,B->cmap->bs);CHKERRQ(ierr);
  ierr = MatSetType(*C,MATMPIDENSE);CHKERRQ(ierr);
  ierr = MatMPIDenseSetPreallocation(*C,PETSC_NULL);CHKERRQ(ierr); 
  ierr = MatAssemblyBegin(*C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  (*C)->ops->matmult = MatMatMult_MPIAIJ_MPIDense;

  ierr = PetscNew(MPIAIJ_MPIDense,&contents);CHKERRQ(ierr);
  /* Create work matrix used to store off processor rows of B needed for local product */
  ierr = MatCreateSeqDense(PETSC_COMM_SELF,nz,B->cmap->N,PETSC_NULL,&contents->workB);CHKERRQ(ierr);
  /* Create work arrays needed */ 
  ierr = PetscMalloc4(B->cmap->N*from->starts[from->n],PetscScalar,&contents->rvalues,
                      B->cmap->N*to->starts[to->n],PetscScalar,&contents->svalues,
                      from->n,MPI_Request,&contents->rwaits,
                      to->n,MPI_Request,&contents->swaits);CHKERRQ(ierr);

  ierr = PetscContainerCreate(((PetscObject)A)->comm,&container);CHKERRQ(ierr);
  ierr = PetscContainerSetPointer(container,contents);CHKERRQ(ierr);
  ierr = PetscContainerSetUserDestroy(container,MatMPIAIJ_MPIDenseDestroy);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)(*C),"workB",(PetscObject)container);CHKERRQ(ierr);
  ierr = PetscContainerDestroy(&container);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMPIDenseScatter"
/*
    Performs an efficient scatter on the rows of B needed by this process; this is
    a modification of the VecScatterBegin_() routines.
*/
PetscErrorCode MatMPIDenseScatter(Mat A,Mat B,Mat C,Mat *outworkB)
{
  Mat_MPIAIJ             *aij = (Mat_MPIAIJ*)A->data;
  PetscErrorCode         ierr;
  PetscScalar            *b,*w,*svalues,*rvalues;
  VecScatter             ctx = aij->Mvctx; 
  VecScatter_MPI_General *from = (VecScatter_MPI_General*) ctx->fromdata;
  VecScatter_MPI_General *to   = ( VecScatter_MPI_General*) ctx->todata;
  PetscInt               i,j,k;
  PetscInt               *sindices,*sstarts,*rindices,*rstarts;
  PetscMPIInt            *sprocs,*rprocs,nrecvs;
  MPI_Request            *swaits,*rwaits;
  MPI_Comm               comm = ((PetscObject)A)->comm;
  PetscMPIInt            tag = ((PetscObject)ctx)->tag,ncols = B->cmap->N, nrows = aij->B->cmap->n,imdex,nrowsB = B->rmap->n;
  MPI_Status             status;
  MPIAIJ_MPIDense        *contents;
  PetscContainer         container;
  Mat                    workB;

  PetscFunctionBegin;
  ierr = PetscObjectQuery((PetscObject)C,"workB",(PetscObject*)&container);CHKERRQ(ierr);
  if (!container) SETERRQ(comm,PETSC_ERR_PLIB,"Container does not exist");
  ierr = PetscContainerGetPointer(container,(void**)&contents);CHKERRQ(ierr);

  workB = *outworkB = contents->workB;
  if (nrows != workB->rmap->n) SETERRQ2(comm,PETSC_ERR_PLIB,"Number of rows of workB %D not equal to columns of aij->B %D",nrows,workB->cmap->n);
  sindices  = to->indices;
  sstarts   = to->starts;
  sprocs    = to->procs;
  swaits    = contents->swaits;
  svalues   = contents->svalues;

  rindices  = from->indices;
  rstarts   = from->starts;
  rprocs    = from->procs;
  rwaits    = contents->rwaits;
  rvalues   = contents->rvalues;

  ierr = MatGetArray(B,&b);CHKERRQ(ierr);
  ierr = MatGetArray(workB,&w);CHKERRQ(ierr);

  for (i=0; i<from->n; i++) {
    ierr = MPI_Irecv(rvalues+ncols*rstarts[i],ncols*(rstarts[i+1]-rstarts[i]),MPIU_SCALAR,rprocs[i],tag,comm,rwaits+i);CHKERRQ(ierr);
  } 

  for (i=0; i<to->n; i++) {
    /* pack a message at a time */
    CHKMEMQ;
    for (j=0; j<sstarts[i+1]-sstarts[i]; j++){
      for (k=0; k<ncols; k++) {
        svalues[ncols*(sstarts[i] + j) + k] = b[sindices[sstarts[i]+j] + nrowsB*k];
      }
    }
    CHKMEMQ;
    ierr = MPI_Isend(svalues+ncols*sstarts[i],ncols*(sstarts[i+1]-sstarts[i]),MPIU_SCALAR,sprocs[i],tag,comm,swaits+i);CHKERRQ(ierr);
  }

  nrecvs = from->n;
  while (nrecvs) {
    ierr = MPI_Waitany(from->n,rwaits,&imdex,&status);CHKERRQ(ierr);
    nrecvs--;
    /* unpack a message at a time */
    CHKMEMQ;
    for (j=0; j<rstarts[imdex+1]-rstarts[imdex]; j++){
      for (k=0; k<ncols; k++) {
        w[rindices[rstarts[imdex]+j] + nrows*k] = rvalues[ncols*(rstarts[imdex] + j) + k];
      }
    }
    CHKMEMQ;
  }
  if (to->n) {ierr = MPI_Waitall(to->n,swaits,to->sstatus);CHKERRQ(ierr);}

  ierr = MatRestoreArray(B,&b);CHKERRQ(ierr);
  ierr = MatRestoreArray(workB,&w);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(workB,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(workB,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
extern PetscErrorCode MatMatMultNumericAdd_SeqAIJ_SeqDense(Mat,Mat,Mat);

#undef __FUNCT__  
#define __FUNCT__ "MatMatMultNumeric_MPIAIJ_MPIDense"
PetscErrorCode MatMatMultNumeric_MPIAIJ_MPIDense(Mat A,Mat B,Mat C)
{
  PetscErrorCode       ierr;
  Mat_MPIAIJ           *aij = (Mat_MPIAIJ*)A->data;
  Mat_MPIDense         *bdense = (Mat_MPIDense*)B->data;
  Mat_MPIDense         *cdense = (Mat_MPIDense*)C->data;
  Mat                  workB;

  PetscFunctionBegin;

  /* diagonal block of A times all local rows of B*/
  ierr = MatMatMultNumeric_SeqAIJ_SeqDense(aij->A,bdense->A,cdense->A);CHKERRQ(ierr);

  /* get off processor parts of B needed to complete the product */
  ierr = MatMPIDenseScatter(A,B,C,&workB);CHKERRQ(ierr);

  /* off-diagonal block of A times nonlocal rows of B */
  ierr = MatMatMultNumericAdd_SeqAIJ_SeqDense(aij->B,workB,cdense->A);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMatMultNumeric_MPIAIJ_MPIAIJ_Scalable"
PetscErrorCode MatMatMultNumeric_MPIAIJ_MPIAIJ_Scalable(Mat A,Mat P,Mat C)
{
  PetscErrorCode     ierr;
  Mat_MPIAIJ         *a=(Mat_MPIAIJ*)A->data,*c=(Mat_MPIAIJ*)C->data;
  Mat_SeqAIJ         *ad=(Mat_SeqAIJ*)(a->A)->data,*ao=(Mat_SeqAIJ*)(a->B)->data;
  Mat_SeqAIJ         *cd=(Mat_SeqAIJ*)(c->A)->data,*co=(Mat_SeqAIJ*)(c->B)->data;
  PetscInt           *adi=ad->i,*adj,*aoi=ao->i,*aoj;
  PetscScalar        *ada,*aoa,*cda=cd->a,*coa=co->a;
  Mat_SeqAIJ         *p_loc,*p_oth; 
  PetscInt           *pi_loc,*pj_loc,*pi_oth,*pj_oth,*pj;
  PetscScalar        *pa_loc,*pa_oth,*pa,valtmp,*ca;
  PetscInt           cm=C->rmap->n,anz,pnz;
  Mat_PtAPMPI        *ptap=c->ptap;
  PetscScalar        *apa_sparse=ptap->apa;
  PetscInt           *api,*apj,*apJ,i,j,k,row;
  PetscInt           cstart=C->cmap->rstart;
  PetscInt           cdnz,conz,k0,k1,nextp;

  PetscFunctionBegin;
  /* 1) get P_oth = ptap->P_oth  and P_loc = ptap->P_loc */
  /*-----------------------------------------------------*/ 
  /* update numerical values of P_oth and P_loc */
  ierr = MatGetBrowsOfAoCols_MPIAIJ(A,P,MAT_REUSE_MATRIX,&ptap->startsj_s,&ptap->startsj_r,&ptap->bufa,&ptap->P_oth);CHKERRQ(ierr);  
  ierr = MatMPIAIJGetLocalMat(P,MAT_REUSE_MATRIX,&ptap->P_loc);CHKERRQ(ierr);

  /* 2) compute numeric C_loc = A_loc*P = Ad*P_loc + Ao*P_oth */
  /*----------------------------------------------------------*/
  /* get data from symbolic products */
  p_loc = (Mat_SeqAIJ*)(ptap->P_loc)->data;
  p_oth = (Mat_SeqAIJ*)(ptap->P_oth)->data;
  pi_loc=p_loc->i; pj_loc=p_loc->j; pa_loc=p_loc->a;   
  pi_oth=p_oth->i; pj_oth=p_oth->j; pa_oth=p_oth->a;
  
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

    /* 1st off-diagoanl part of C */
    ca = coa + co->i[i];
    k  = 0;
    for (k0=0; k0<conz; k0++){
      if (apJ[k] >= cstart) break;
      ca[k0]      = apa_sparse[k]; 
      apa_sparse[k] = 0.0;
      k++;
    }

    /* diagonal part of C */
    ca = cda + cd->i[i];
    for (k1=0; k1<cdnz; k1++){
      ca[k1]      = apa_sparse[k]; 
      apa_sparse[k] = 0.0;
      k++;
    }

    /* 2nd off-diagoanl part of C */
    ca = coa + co->i[i];
    for (; k0<conz; k0++){
      ca[k0]      = apa_sparse[k]; 
      apa_sparse[k] = 0.0;
      k++;
    }
  } 
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* same as MatMatMultSymbolic_MPIAIJ_MPIAIJ(), except using LLCondensed to avoid O(BN) memory requirement */
#undef __FUNCT__
#define __FUNCT__ "MatMatMultSymbolic_MPIAIJ_MPIAIJ_Scalable"
PetscErrorCode MatMatMultSymbolic_MPIAIJ_MPIAIJ_Scalable(Mat A,Mat P,PetscReal fill,Mat *C)
{
  PetscErrorCode       ierr;
  MPI_Comm             comm=((PetscObject)A)->comm;
  Mat                  Cmpi; 
  Mat_PtAPMPI          *ptap;
  PetscFreeSpaceList   free_space=PETSC_NULL,current_space=PETSC_NULL;
  Mat_MPIAIJ           *a=(Mat_MPIAIJ*)A->data,*c;
  Mat_SeqAIJ           *ad=(Mat_SeqAIJ*)(a->A)->data,*ao=(Mat_SeqAIJ*)(a->B)->data,*p_loc,*p_oth;
  PetscInt             *pi_loc,*pj_loc,*pi_oth,*pj_oth,*dnz,*onz;
  PetscInt             *adi=ad->i,*adj=ad->j,*aoi=ao->i,*aoj=ao->j,rstart=A->rmap->rstart; 
  PetscInt             i,pnz,row,*api,*apj,*Jptr,apnz,nspacedouble=0,j,nzi,*lnk,apnz_max=0;
  PetscInt             am=A->rmap->n,pN=P->cmap->N,pn=P->cmap->n,pm=P->rmap->n;  
  PetscInt             nlnk_max,armax,prmax;
  PetscReal            afill;
  PetscScalar          *apa;

  PetscFunctionBegin;
  /* create struct Mat_PtAPMPI and attached it to C later */
  ierr = PetscNew(Mat_PtAPMPI,&ptap);CHKERRQ(ierr); 

  /* get P_oth by taking rows of P (= non-zero cols of local A) from other processors */
  ierr = MatGetBrowsOfAoCols_MPIAIJ(A,P,MAT_INITIAL_MATRIX,&ptap->startsj_s,&ptap->startsj_r,&ptap->bufa,&ptap->P_oth);CHKERRQ(ierr);

  /* get P_loc by taking all local rows of P */
  ierr = MatMPIAIJGetLocalMat(P,MAT_INITIAL_MATRIX,&ptap->P_loc);CHKERRQ(ierr);

  p_loc = (Mat_SeqAIJ*)(ptap->P_loc)->data; 
  p_oth = (Mat_SeqAIJ*)(ptap->P_oth)->data;
  pi_loc = p_loc->i; pj_loc = p_loc->j; 
  pi_oth = p_oth->i; pj_oth = p_oth->j;

  /* first, compute symbolic AP = A_loc*P = A_diag*P_loc + A_off*P_oth */
  /*-------------------------------------------------------------------*/
  ierr  = PetscMalloc((am+2)*sizeof(PetscInt),&api);CHKERRQ(ierr);
  ptap->api = api;
  api[0]    = 0;

  /* create and initialize a linked list */
  armax = ad->rmax+ao->rmax;
  prmax = PetscMax(p_loc->rmax,p_oth->rmax);
  nlnk_max = armax*prmax;
  if (!nlnk_max || nlnk_max > pN) nlnk_max = pN;
  ierr = PetscLLCondensedCreate_Scalable(nlnk_max,&lnk);CHKERRQ(ierr);

  /* Initial FreeSpace size is fill*(nnz(A)+nnz(P)) */
  ierr = PetscFreeSpaceGet((PetscInt)(fill*(adi[am]+aoi[am]+pi_loc[pm])),&free_space);CHKERRQ(ierr);
  current_space = free_space; 

  ierr = MatPreallocateInitialize(comm,am,pn,dnz,onz);CHKERRQ(ierr);
  for (i=0; i<am; i++) {
    apnz = 0;
    /* diagonal portion of A */
    nzi = adi[i+1] - adi[i];
    for (j=0; j<nzi; j++){
      row = *adj++; 
      pnz = pi_loc[row+1] - pi_loc[row];
      Jptr  = pj_loc + pi_loc[row];
      /* add non-zero cols of P into the sorted linked list lnk */
      ierr = PetscLLCondensedAddSorted_Scalable(pnz,Jptr,lnk);CHKERRQ(ierr);
    }
    /* off-diagonal portion of A */
    nzi = aoi[i+1] - aoi[i];
    for (j=0; j<nzi; j++){   
      row = *aoj++; 
      pnz = pi_oth[row+1] - pi_oth[row];
      Jptr  = pj_oth + pi_oth[row];  
      ierr = PetscLLCondensedAddSorted_Scalable(pnz,Jptr,lnk);CHKERRQ(ierr);
    }

    apnz     = *lnk;
    api[i+1] = api[i] + apnz;
    if (apnz > apnz_max) apnz_max = apnz;
    
    /* if free space is not available, double the total space in the list */
    if (current_space->local_remaining<apnz) {
      ierr = PetscFreeSpaceGet(apnz+current_space->total_array_size,&current_space);CHKERRQ(ierr);
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
  ierr = PetscMalloc((api[am]+1)*sizeof(PetscInt),&ptap->apj);CHKERRQ(ierr);
  apj  = ptap->apj;
  ierr = PetscFreeSpaceContiguous(&free_space,ptap->apj);CHKERRQ(ierr);
  ierr = PetscLLCondensedDestroy_Scalable(lnk);CHKERRQ(ierr);

  /* create and assemble symbolic parallel matrix Cmpi */
  /*----------------------------------------------------*/
  ierr = MatCreate(comm,&Cmpi);CHKERRQ(ierr);
  ierr = MatSetSizes(Cmpi,am,pn,PETSC_DETERMINE,PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = MatSetBlockSizes(Cmpi,A->rmap->bs,P->cmap->bs);CHKERRQ(ierr);
  ierr = MatSetType(Cmpi,MATMPIAIJ);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(Cmpi,0,dnz,0,onz);CHKERRQ(ierr);
  ierr = MatPreallocateFinalize(dnz,onz);CHKERRQ(ierr);
  ierr = MatSetBlockSize(Cmpi,1);CHKERRQ(ierr);

  /* malloc apa for assembly Cmpi */ 
  ierr = PetscMalloc(apnz_max*sizeof(PetscScalar),&apa);CHKERRQ(ierr);
  ierr = PetscMemzero(apa,apnz_max*sizeof(PetscScalar));CHKERRQ(ierr);
  ptap->apa = apa;
  for (i=0; i<am; i++){
    row  = i + rstart;
    apnz = api[i+1] - api[i];
    ierr = MatSetValues(Cmpi,1,&row,apnz,apj,apa,INSERT_VALUES);CHKERRQ(ierr);
    apj += apnz;
  }
  ierr = MatAssemblyBegin(Cmpi,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Cmpi,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ptap->destroy             = Cmpi->ops->destroy;
  ptap->duplicate           = Cmpi->ops->duplicate;
  Cmpi->ops->matmultnumeric = MatMatMultNumeric_MPIAIJ_MPIAIJ_Scalable;
  Cmpi->ops->destroy        = MatDestroy_MPIAIJ_MatMatMult;  
  Cmpi->ops->duplicate      = MatDuplicate_MPIAIJ_MatMatMult;

  /* attach the supporting struct to Cmpi for reuse */
  c = (Mat_MPIAIJ*)Cmpi->data;
  c->ptap  = ptap;
  
  *C = Cmpi;

  /* set MatInfo */
  afill = (PetscReal)api[am]/(adi[am]+aoi[am]+pi_loc[pm]+1) + 1.e-5;
  if (afill < 1.0) afill = 1.0;
  Cmpi->info.mallocs           = nspacedouble;
  Cmpi->info.fill_ratio_given  = fill;               
  Cmpi->info.fill_ratio_needed = afill; 

#if defined(PETSC_USE_INFO)
  if (api[am]) {
    ierr = PetscInfo3(Cmpi,"Reallocs %D; Fill ratio: given %G needed %G.\n",nspacedouble,fill,afill);CHKERRQ(ierr);
    ierr = PetscInfo1(Cmpi,"Use MatMatMult(A,B,MatReuse,%G,&C) for best performance.;\n",afill);CHKERRQ(ierr);
  } else {
    ierr = PetscInfo(Cmpi,"Empty matrix product\n");CHKERRQ(ierr);
  }
#endif
  PetscFunctionReturn(0);
}

/*-------------------------------------------------------------------------*/
#undef __FUNCT__
#define __FUNCT__ "MatTransposeMatMult_MPIAIJ_MPIAIJ"
PetscErrorCode MatTransposeMatMult_MPIAIJ_MPIAIJ(Mat P,Mat A,MatReuse scall,PetscReal fill,Mat *C) 
{
  PetscErrorCode ierr;
  PetscBool      scalable=PETSC_FALSE;

  PetscFunctionBegin;
  if (scall == MAT_INITIAL_MATRIX){
    ierr = PetscObjectOptionsBegin((PetscObject)A);CHKERRQ(ierr);
      ierr = PetscOptionsBool("-mattransposematmult_scalable","Use a scalable but slower C=Pt*A","",scalable,&scalable,PETSC_NULL);CHKERRQ(ierr);
      if  (scalable){
        ierr = MatTransposeMatMultSymbolic_MPIAIJ_MPIAIJ_Scalable(P,A,fill,C);CHKERRQ(ierr);
      } else {
        ierr = MatTransposeMatMultSymbolic_MPIAIJ_MPIAIJ(P,A,fill,C);CHKERRQ(ierr);
      }
    ierr = PetscOptionsEnd();CHKERRQ(ierr);
  }
  ierr = (*(*C)->ops->mattransposemultnumeric)(P,A,*C);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatTransposeMatMultNumeric_MPIAIJ_MPIAIJ"
PetscErrorCode MatTransposeMatMultNumeric_MPIAIJ_MPIAIJ(Mat P,Mat A,Mat C)
{
  PetscErrorCode       ierr;
  Mat_Merge_SeqsToMPI  *merge; 
  Mat_MPIAIJ           *p=(Mat_MPIAIJ*)P->data,*c=(Mat_MPIAIJ*)C->data;
  Mat_SeqAIJ           *pd=(Mat_SeqAIJ*)(p->A)->data,*po=(Mat_SeqAIJ*)(p->B)->data;
  Mat_PtAPMPI          *ptap;
  PetscInt             *adj,*aJ;
  PetscInt             i,j,k,anz,pnz,row,*cj;
  MatScalar            *ada,*aval,*ca,valtmp;
  PetscInt             am=A->rmap->n,cm=C->rmap->n,pon=(p->B)->cmap->n; 
  MPI_Comm             comm=((PetscObject)C)->comm;
  PetscMPIInt          size,rank,taga,*len_s;
  PetscInt             *owners,proc,nrows,**buf_ri_k,**nextrow,**nextci;
  PetscInt             **buf_ri,**buf_rj;  
  PetscInt             cnz=0,*bj_i,*bi,*bj,bnz,nextcj; /* bi,bj,ba: local array of C(mpi mat) */
  MPI_Request          *s_waits,*r_waits; 
  MPI_Status           *status;
  MatScalar            **abuf_r,*ba_i,*pA,*coa,*ba; 
  PetscInt             *ai,*aj,*coi,*coj; 
  PetscInt             *poJ=po->j,*pdJ=pd->j; 
  Mat                  A_loc;
  Mat_SeqAIJ           *a_loc;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);

  ptap  = c->ptap;
  merge = ptap->merge;

  /* 2) compute numeric C_seq = P_loc^T*A_loc*P - dominating part */
  /*--------------------------------------------------------------*/
  /* get data from symbolic products */
  coi = merge->coi; coj = merge->coj;
  ierr = PetscMalloc((coi[pon]+1)*sizeof(MatScalar),&coa);CHKERRQ(ierr);
  ierr = PetscMemzero(coa,coi[pon]*sizeof(MatScalar));CHKERRQ(ierr);

  bi     = merge->bi; bj = merge->bj;
  owners = merge->rowmap->range;
  ierr   = PetscMalloc((bi[cm]+1)*sizeof(MatScalar),&ba);CHKERRQ(ierr);
  ierr   = PetscMemzero(ba,bi[cm]*sizeof(MatScalar));CHKERRQ(ierr);
 
  /* get A_loc by taking all local rows of A */
  A_loc = ptap->A_loc;
  ierr = MatMPIAIJGetLocalMat(A,MAT_REUSE_MATRIX,&A_loc);CHKERRQ(ierr);
  a_loc = (Mat_SeqAIJ*)(A_loc)->data; 
  ai   = a_loc->i; 
  aj   = a_loc->j;

  ierr = PetscMalloc((A->cmap->N)*sizeof(PetscScalar),&aval);CHKERRQ(ierr); /* non-scalable!!! */
  ierr = PetscMemzero(aval,A->cmap->N*sizeof(PetscScalar));CHKERRQ(ierr);

    for (i=0; i<am; i++) {
      /* 2-a) put A[i,:] to dense array aval */
      anz = ai[i+1] - ai[i];
      adj = aj + ai[i];
      ada = a_loc->a + ai[i];
      for (j=0; j<anz; j++){
        aval[adj[j]] = ada[j];
      }

      /* 2-b) Compute Cseq = P_loc[i,:]^T*A[i,:] using outer product */
      /*--------------------------------------------------------------*/  
      /* put the value into Co=(p->B)^T*A (off-diagonal part, send to others) */
      pnz = po->i[i+1] - po->i[i];
      poJ = po->j + po->i[i];
      pA  = po->a + po->i[i];
      for (j=0; j<pnz; j++){ 
        row = poJ[j]; 
        cnz = coi[row+1] - coi[row];
        cj  = coj + coi[row]; 
        ca  = coa + coi[row];
        /* perform dense axpy */
        valtmp = pA[j];
        for (k=0; k<cnz; k++) { 
          ca[k] += valtmp*aval[cj[k]];
        }
        ierr = PetscLogFlops(2.0*cnz);CHKERRQ(ierr);      
      } 

      /* put the value into Cd (diagonal part) */
      pnz = pd->i[i+1] - pd->i[i];
      pdJ = pd->j + pd->i[i];
      pA  = pd->a + pd->i[i];
      for (j=0; j<pnz; j++){  
        row = pdJ[j]; 
        cnz = bi[row+1] - bi[row];
        cj  = bj + bi[row]; 
        ca  = ba + bi[row];
        /* perform dense axpy */
        valtmp = pA[j];
        for (k=0; k<cnz; k++) { 
          ca[k] += valtmp*aval[cj[k]]; 
        }
        ierr = PetscLogFlops(2.0*cnz);CHKERRQ(ierr);     
      }
     
      /* zero the current row of Pt*A */
      aJ = aj + ai[i];
      for (k=0; k<anz; k++) aval[aJ[k]] = 0.0;
    }

  /* 3) send and recv matrix values coa */
  /*------------------------------------*/
  buf_ri = merge->buf_ri;
  buf_rj = merge->buf_rj;
  len_s  = merge->len_s;
  ierr = PetscCommGetNewTag(comm,&taga);CHKERRQ(ierr);
  ierr = PetscPostIrecvScalar(comm,taga,merge->nrecv,merge->id_r,merge->len_r,&abuf_r,&r_waits);CHKERRQ(ierr);

  ierr = PetscMalloc2(merge->nsend+1,MPI_Request,&s_waits,size,MPI_Status,&status);CHKERRQ(ierr);
  for (proc=0,k=0; proc<size; proc++){  
    if (!len_s[proc]) continue;
    i = merge->owners_co[proc];
    ierr = MPI_Isend(coa+coi[i],len_s[proc],MPIU_MATSCALAR,proc,taga,comm,s_waits+k);CHKERRQ(ierr);
    k++;
  } 
  if (merge->nrecv) {ierr = MPI_Waitall(merge->nrecv,r_waits,status);CHKERRQ(ierr);}
  if (merge->nsend) {ierr = MPI_Waitall(merge->nsend,s_waits,status);CHKERRQ(ierr);}

  ierr = PetscFree2(s_waits,status);CHKERRQ(ierr);
  ierr = PetscFree(r_waits);CHKERRQ(ierr);
  ierr = PetscFree(coa);CHKERRQ(ierr);

  /* 4) insert local Cseq and received values into Cmpi */
  /*----------------------------------------------------*/
  ierr = PetscMalloc3(merge->nrecv,PetscInt**,&buf_ri_k,merge->nrecv,PetscInt*,&nextrow,merge->nrecv,PetscInt*,&nextci);CHKERRQ(ierr);
  for (k=0; k<merge->nrecv; k++){
    buf_ri_k[k] = buf_ri[k]; /* beginning of k-th recved i-structure */
    nrows       = *(buf_ri_k[k]);
    nextrow[k]  = buf_ri_k[k]+1;  /* next row number of k-th recved i-structure */
    nextci[k]   = buf_ri_k[k] + (nrows + 1);/* poins to the next i-structure of k-th recved i-structure  */
  }

  for (i=0; i<cm; i++) {
    row = owners[rank] + i; /* global row index of C_seq */
    bj_i = bj + bi[i];  /* col indices of the i-th row of C */
    ba_i = ba + bi[i]; 
    bnz  = bi[i+1] - bi[i];
    /* add received vals into ba */
    for (k=0; k<merge->nrecv; k++){ /* k-th received message */
      /* i-th row */
      if (i == *nextrow[k]) {
        cnz = *(nextci[k]+1) - *nextci[k]; 
        cj  = buf_rj[k] + *(nextci[k]);
        ca  = abuf_r[k] + *(nextci[k]);
        nextcj = 0;
        for (j=0; nextcj<cnz; j++){ 
          if (bj_i[j] == cj[nextcj]){ /* bcol == ccol */
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
  ierr = PetscFree(aval);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* This routine is modified from MatPtAPSymbolic_MPIAIJ_MPIAIJ() */
#undef __FUNCT__  
#define __FUNCT__ "MatTransposeMatMultSymbolic_MPIAIJ_MPIAIJ"
PetscErrorCode MatTransposeMatMultSymbolic_MPIAIJ_MPIAIJ(Mat P,Mat A,PetscReal fill,Mat *C)
{
  PetscErrorCode       ierr;
  Mat                  Cmpi,A_loc,POt,PDt; 
  Mat_PtAPMPI          *ptap;
  PetscFreeSpaceList   free_space=PETSC_NULL,current_space=PETSC_NULL;
  Mat_MPIAIJ           *p=(Mat_MPIAIJ*)P->data,*c;
  PetscInt             *pdti,*pdtj,*poti,*potj,*ptJ;
  PetscInt             nnz; 
  PetscInt             *lnk,*owners_co,*coi,*coj,i,k,pnz,row;
  PetscInt             am=A->rmap->n,pn=P->cmap->n;  
  PetscBT              lnkbt;
  MPI_Comm             comm=((PetscObject)A)->comm;
  PetscMPIInt          size,rank,tagi,tagj,*len_si,*len_s,*len_ri; 
  PetscInt             **buf_rj,**buf_ri,**buf_ri_k;
  PetscInt             len,proc,*dnz,*onz,*owners;
  PetscInt             nzi,*bi,*bj; 
  PetscInt             nrows,*buf_s,*buf_si,*buf_si_i,**nextrow,**nextci;
  MPI_Request          *swaits,*rwaits; 
  MPI_Status           *sstatus,rstatus;
  Mat_Merge_SeqsToMPI  *merge;
  PetscInt             *ai,*aj,*Jptr,anz,*prmap=p->garray,pon,nspacedouble=0,j;
  PetscReal            afill=1.0,afill_tmp;
  PetscInt             rstart = P->cmap->rstart,rmax,aN=A->cmap->N,Crmax;
  PetscScalar          *vals;
  Mat_SeqAIJ           *a_loc, *pdt,*pot;

  PetscFunctionBegin;
  /* check if matrix local sizes are compatible */
  if (A->rmap->rstart != P->rmap->rstart || A->rmap->rend != P->rmap->rend){
    SETERRQ4(comm,PETSC_ERR_ARG_SIZ,"Matrix local dimensions are incompatible, A (%D, %D) != P (%D,%D)",A->rmap->rstart,A->rmap->rend,P->rmap->rstart,P->rmap->rend);
  }

  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);

  /* create struct Mat_PtAPMPI and attached it to C later */
  ierr = PetscNew(Mat_PtAPMPI,&ptap);CHKERRQ(ierr);

  /* get A_loc by taking all local rows of A */
  ierr = MatMPIAIJGetLocalMat(A,MAT_INITIAL_MATRIX,&A_loc);CHKERRQ(ierr);
  ptap->A_loc = A_loc;
  a_loc = (Mat_SeqAIJ*)(A_loc)->data; 
  ai   = a_loc->i; 
  aj   = a_loc->j;
 
  /* determine symbolic Co=(p->B)^T*A - send to others */
  /*----------------------------------------------------*/
  ierr = MatTransposeSymbolic_SeqAIJ(p->A,&PDt);CHKERRQ(ierr); 
  pdt = (Mat_SeqAIJ*)PDt->data; 
  pdti = pdt->i; pdtj = pdt->j;

  ierr = MatTransposeSymbolic_SeqAIJ(p->B,&POt);CHKERRQ(ierr); 
  pot = (Mat_SeqAIJ*)POt->data; 
  poti = pot->i; potj = pot->j;

  /* then, compute symbolic Co = (p->B)^T*A */
  pon = (p->B)->cmap->n; /* total num of rows to be sent to other processors 
                         >= (num of nonzero rows of C_seq) - pn */
  ierr = PetscMalloc((pon+1)*sizeof(PetscInt),&coi);CHKERRQ(ierr);
  coi[0] = 0;

  /* set initial free space to be fill*(nnz(p->B) + nnz(A)) */
  nnz           = fill*(poti[pon] + ai[am]);
  ierr          = PetscFreeSpaceGet(nnz,&free_space);
  current_space = free_space;

  /* create and initialize a linked list */
  i = PetscMax(pdt->rmax,pot->rmax);
  Crmax = i*a_loc->rmax*size;
  if (!Crmax || Crmax > aN) Crmax = aN;
  ierr = PetscLLCondensedCreate(Crmax,aN,&lnk,&lnkbt);CHKERRQ(ierr);

  for (i=0; i<pon; i++) {
    pnz = poti[i+1] - poti[i];
    ptJ = potj + poti[i];
    for (j=0; j<pnz; j++){
      row  = ptJ[j]; /* row of A_loc == col of Pot */
      anz  = ai[row+1] - ai[row];
      Jptr = aj + ai[row];
      /* add non-zero cols of AP into the sorted linked list lnk */
      ierr = PetscLLCondensedAddSorted(anz,Jptr,lnk,lnkbt);CHKERRQ(ierr);
    }
    nnz = lnk[0];

    /* If free space is not available, double the total space in the list */
    if (current_space->local_remaining<nnz) {
      ierr = PetscFreeSpaceGet(nnz+current_space->total_array_size,&current_space);CHKERRQ(ierr);
      nspacedouble++;
    }

    /* Copy data into free space, and zero out denserows */
    ierr = PetscLLCondensedClean(aN,nnz,current_space->array,lnk,lnkbt);CHKERRQ(ierr);
    current_space->array           += nnz;
    current_space->local_used      += nnz;
    current_space->local_remaining -= nnz;
    coi[i+1] = coi[i] + nnz;
  }
  
  ierr = PetscMalloc((coi[pon]+1)*sizeof(PetscInt),&coj);CHKERRQ(ierr); 
  ierr = PetscFreeSpaceContiguous(&free_space,coj);CHKERRQ(ierr);
  afill_tmp = (PetscReal)coi[pon]/(poti[pon] + ai[am]+1);
  if (afill_tmp > afill) afill = afill_tmp;
 
  /* send j-array (coj) of Co to other processors */
  /*----------------------------------------------*/
  /* determine row ownership */
  ierr = PetscNew(Mat_Merge_SeqsToMPI,&merge);CHKERRQ(ierr);
  ierr = PetscLayoutCreate(comm,&merge->rowmap);CHKERRQ(ierr);
  merge->rowmap->n = pn;
  merge->rowmap->bs = 1;
  ierr = PetscLayoutSetUp(merge->rowmap);CHKERRQ(ierr);
  owners = merge->rowmap->range;

  /* determine the number of messages to send, their lengths */
  ierr = PetscMalloc(size*sizeof(PetscMPIInt),&len_si);CHKERRQ(ierr);
  ierr = PetscMemzero(len_si,size*sizeof(PetscMPIInt));CHKERRQ(ierr);
  ierr = PetscMalloc(size*sizeof(PetscMPIInt),&merge->len_s);CHKERRQ(ierr);
  len_s = merge->len_s;
  merge->nsend = 0;
  
  ierr = PetscMalloc((size+2)*sizeof(PetscInt),&owners_co);CHKERRQ(ierr);
  ierr = PetscMemzero(len_s,size*sizeof(PetscMPIInt));CHKERRQ(ierr);

  proc = 0;
  for (i=0; i<pon; i++){
    while (prmap[i] >= owners[proc+1]) proc++;
    len_si[proc]++;  /* num of rows in Co to be sent to [proc] */
    len_s[proc] += coi[i+1] - coi[i];
  }

  len   = 0;  /* max length of buf_si[] */
  owners_co[0] = 0;
  for (proc=0; proc<size; proc++){
    owners_co[proc+1] = owners_co[proc] + len_si[proc];
    if (len_si[proc]){ 
      merge->nsend++;
      len_si[proc] = 2*(len_si[proc] + 1);
      len += len_si[proc]; 
    }
  }

  /* determine the number and length of messages to receive for coi and coj  */
  ierr = PetscGatherNumberOfMessages(comm,PETSC_NULL,len_s,&merge->nrecv);CHKERRQ(ierr);
  ierr = PetscGatherMessageLengths2(comm,merge->nsend,merge->nrecv,len_s,len_si,&merge->id_r,&merge->len_r,&len_ri);CHKERRQ(ierr);      

  /* post the Irecv and Isend of coj */
  ierr = PetscCommGetNewTag(comm,&tagj);CHKERRQ(ierr);
  ierr = PetscPostIrecvInt(comm,tagj,merge->nrecv,merge->id_r,merge->len_r,&buf_rj,&rwaits);CHKERRQ(ierr);
  ierr = PetscMalloc((merge->nsend+1)*sizeof(MPI_Request),&swaits);CHKERRQ(ierr);
  for (proc=0, k=0; proc<size; proc++){  
    if (!len_s[proc]) continue;
    i = owners_co[proc];
    ierr = MPI_Isend(coj+coi[i],len_s[proc],MPIU_INT,proc,tagj,comm,swaits+k);CHKERRQ(ierr);
    k++;
  } 

  /* receives and sends of coj are complete */
  ierr = PetscMalloc(size*sizeof(MPI_Status),&sstatus);CHKERRQ(ierr); 
  for (i=0; i<merge->nrecv; i++){
    PetscMPIInt icompleted;
    ierr = MPI_Waitany(merge->nrecv,rwaits,&icompleted,&rstatus);CHKERRQ(ierr);
  }
  ierr = PetscFree(rwaits);CHKERRQ(ierr);
  if (merge->nsend) {ierr = MPI_Waitall(merge->nsend,swaits,sstatus);CHKERRQ(ierr);}
  
  /* send and recv coi */
  /*-------------------*/  
  ierr = PetscCommGetNewTag(comm,&tagi);CHKERRQ(ierr);
  ierr = PetscPostIrecvInt(comm,tagi,merge->nrecv,merge->id_r,len_ri,&buf_ri,&rwaits);CHKERRQ(ierr);
  ierr = PetscMalloc((len+1)*sizeof(PetscInt),&buf_s);CHKERRQ(ierr); 
  buf_si = buf_s;  /* points to the beginning of k-th msg to be sent */
  for (proc=0,k=0; proc<size; proc++){  
    if (!len_s[proc]) continue;
    /* form outgoing message for i-structure: 
         buf_si[0]:                 nrows to be sent
               [1:nrows]:           row index (global)
               [nrows+1:2*nrows+1]: i-structure index
    */
    /*-------------------------------------------*/      
    nrows = len_si[proc]/2 - 1; 
    buf_si_i    = buf_si + nrows+1;
    buf_si[0]   = nrows;
    buf_si_i[0] = 0;
    nrows = 0;
    for (i=owners_co[proc]; i<owners_co[proc+1]; i++){
      nzi = coi[i+1] - coi[i];
      buf_si_i[nrows+1] = buf_si_i[nrows] + nzi; /* i-structure */
      buf_si[nrows+1] =prmap[i] -owners[proc]; /* local row index */
      nrows++;
    }
    ierr = MPI_Isend(buf_si,len_si[proc],MPIU_INT,proc,tagi,comm,swaits+k);CHKERRQ(ierr);
    k++;
    buf_si += len_si[proc];
  } 
  i = merge->nrecv;
  while (i--) {
    PetscMPIInt icompleted;
    ierr = MPI_Waitany(merge->nrecv,rwaits,&icompleted,&rstatus);CHKERRQ(ierr);
  }
  ierr = PetscFree(rwaits);CHKERRQ(ierr);
  if (merge->nsend) {ierr = MPI_Waitall(merge->nsend,swaits,sstatus);CHKERRQ(ierr);}
  ierr = PetscFree(len_si);CHKERRQ(ierr);
  ierr = PetscFree(len_ri);CHKERRQ(ierr);
  ierr = PetscFree(swaits);CHKERRQ(ierr);
  ierr = PetscFree(sstatus);CHKERRQ(ierr);
  ierr = PetscFree(buf_s);CHKERRQ(ierr);

  /* compute the local portion of C (mpi mat) */
  /*------------------------------------------*/
  /* allocate bi array and free space for accumulating nonzero column info */
  ierr = PetscMalloc((pn+1)*sizeof(PetscInt),&bi);CHKERRQ(ierr);
  bi[0] = 0;

  /* set initial free space to be fill*(nnz(P) + nnz(A)) */
  nnz           = fill*(pdti[pn] + poti[pon] + ai[am]);
  ierr          = PetscFreeSpaceGet(nnz,&free_space);
  current_space = free_space;

  ierr = PetscMalloc3(merge->nrecv,PetscInt**,&buf_ri_k,merge->nrecv,PetscInt*,&nextrow,merge->nrecv,PetscInt*,&nextci);CHKERRQ(ierr);
  for (k=0; k<merge->nrecv; k++){
    buf_ri_k[k] = buf_ri[k]; /* beginning of k-th recved i-structure */
    nrows       = *buf_ri_k[k];
    nextrow[k]  = buf_ri_k[k] + 1;  /* next row number of k-th recved i-structure */
    nextci[k]   = buf_ri_k[k] + (nrows + 1);/* poins to the next i-structure of k-th recved i-structure  */
  }

  ierr = MatPreallocateInitialize(comm,pn,A->cmap->n,dnz,onz);CHKERRQ(ierr);
  rmax = 0;
  for (i=0; i<pn; i++) {
    /* add pdt[i,:]*AP into lnk */
    pnz = pdti[i+1] - pdti[i];
    ptJ = pdtj + pdti[i];
    for (j=0; j<pnz; j++){
      row  = ptJ[j];  /* row of AP == col of Pt */
      anz  = ai[row+1] - ai[row];
      Jptr = aj + ai[row];
      /* add non-zero cols of AP into the sorted linked list lnk */
      ierr = PetscLLCondensedAddSorted(anz,Jptr,lnk,lnkbt);CHKERRQ(ierr);
    }

    /* add received col data into lnk */
    for (k=0; k<merge->nrecv; k++){ /* k-th received message */
      if (i == *nextrow[k]) { /* i-th row */
        nzi = *(nextci[k]+1) - *nextci[k]; 
        Jptr  = buf_rj[k] + *nextci[k];
        ierr = PetscLLCondensedAddSorted(nzi,Jptr,lnk,lnkbt);CHKERRQ(ierr);
        nextrow[k]++; nextci[k]++;
      }
    }
    nnz = lnk[0]; 

    /* if free space is not available, make more free space */
    if (current_space->local_remaining<nnz) {
      ierr = PetscFreeSpaceGet(nnz+current_space->total_array_size,&current_space);CHKERRQ(ierr);
      nspacedouble++;
    }
    /* copy data into free space, then initialize lnk */
    ierr = PetscLLCondensedClean(aN,nnz,current_space->array,lnk,lnkbt);CHKERRQ(ierr);
    ierr = MatPreallocateSet(i+owners[rank],nnz,current_space->array,dnz,onz);CHKERRQ(ierr);
    current_space->array           += nnz;
    current_space->local_used      += nnz;
    current_space->local_remaining -= nnz;
    bi[i+1] = bi[i] + nnz;
    if (nnz > rmax) rmax = nnz;
  }
  ierr = PetscFree3(buf_ri_k,nextrow,nextci);CHKERRQ(ierr);

  ierr = PetscMalloc((bi[pn]+1)*sizeof(PetscInt),&bj);CHKERRQ(ierr);
  ierr = PetscFreeSpaceContiguous(&free_space,bj);CHKERRQ(ierr);
  afill_tmp = (PetscReal)bi[pn]/(pdti[pn] + poti[pon] + ai[am]+1); 
  if (afill_tmp > afill) afill = afill_tmp;
  ierr = PetscLLCondensedDestroy(lnk,lnkbt);CHKERRQ(ierr);
  ierr = MatDestroy(&POt);CHKERRQ(ierr);
  ierr = MatDestroy(&PDt);CHKERRQ(ierr);

  /* create symbolic parallel matrix Cmpi - why cannot be assembled in Numeric part   */
  /*----------------------------------------------------------------------------------*/
  ierr = PetscMalloc((rmax+1)*sizeof(PetscScalar),&vals);CHKERRQ(ierr); 
  ierr = PetscMemzero(vals,rmax*sizeof(PetscScalar));CHKERRQ(ierr);

  ierr = MatCreate(comm,&Cmpi);CHKERRQ(ierr);
  ierr = MatSetSizes(Cmpi,pn,A->cmap->n,PETSC_DETERMINE,PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = MatSetBlockSizes(Cmpi,P->cmap->bs,A->cmap->bs);CHKERRQ(ierr);
  ierr = MatSetType(Cmpi,MATMPIAIJ);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(Cmpi,0,dnz,0,onz);CHKERRQ(ierr);
  ierr = MatPreallocateFinalize(dnz,onz);CHKERRQ(ierr);
  ierr = MatSetBlockSize(Cmpi,1);CHKERRQ(ierr);
  for (i=0; i<pn; i++){
    row = i + rstart;
    nnz = bi[i+1] - bi[i];
    Jptr = bj + bi[i];
    ierr = MatSetValues(Cmpi,1,&row,nnz,Jptr,vals,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(Cmpi,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Cmpi,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = PetscFree(vals);CHKERRQ(ierr);

  merge->bi            = bi;
  merge->bj            = bj;
  merge->coi           = coi;
  merge->coj           = coj;
  merge->buf_ri        = buf_ri;
  merge->buf_rj        = buf_rj;
  merge->owners_co     = owners_co;
  merge->destroy       = Cmpi->ops->destroy;
  merge->duplicate     = Cmpi->ops->duplicate;

  Cmpi->ops->mattransposemultnumeric = MatTransposeMatMultNumeric_MPIAIJ_MPIAIJ;
  Cmpi->ops->destroy                 = MatDestroy_MPIAIJ_PtAP;  

  /* attach the supporting struct to Cmpi for reuse */
  c = (Mat_MPIAIJ*)Cmpi->data;
  c->ptap        = ptap;
  ptap->api      = PETSC_NULL;
  ptap->apj      = PETSC_NULL;
  ptap->merge    = merge;
  ptap->rmax     = rmax;
 
  *C = Cmpi;
#if defined(PETSC_USE_INFO)
  if (bi[pn] != 0) {
    ierr = PetscInfo3(Cmpi,"Reallocs %D; Fill ratio: given %G needed %G.\n",nspacedouble,fill,afill);CHKERRQ(ierr);
    ierr = PetscInfo1(Cmpi,"Use MatTransposeMatMult(A,B,MatReuse,%G,&C) for best performance.\n",afill);CHKERRQ(ierr);
  } else {
    ierr = PetscInfo(Cmpi,"Empty matrix product\n");CHKERRQ(ierr);
  }
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatTransposeMatMultNumeric_MPIAIJ_MPIAIJ_Scalable"
PetscErrorCode MatTransposeMatMultNumeric_MPIAIJ_MPIAIJ_Scalable(Mat P,Mat A,Mat C)
{
  PetscErrorCode       ierr;
  Mat_Merge_SeqsToMPI  *merge; 
  Mat_MPIAIJ           *p=(Mat_MPIAIJ*)P->data,*c=(Mat_MPIAIJ*)C->data;
  Mat_SeqAIJ           *pd=(Mat_SeqAIJ*)(p->A)->data,*po=(Mat_SeqAIJ*)(p->B)->data;
  Mat_PtAPMPI          *ptap;
  PetscInt             *adj;
  PetscInt             i,j,k,anz,pnz,row,*cj,nexta;
  MatScalar            *ada,*ca,valtmp;
  PetscInt             am=A->rmap->n,cm=C->rmap->n,pon=(p->B)->cmap->n; 
  MPI_Comm             comm=((PetscObject)C)->comm;
  PetscMPIInt          size,rank,taga,*len_s;
  PetscInt             *owners,proc,nrows,**buf_ri_k,**nextrow,**nextci;
  PetscInt             **buf_ri,**buf_rj;  
  PetscInt             cnz=0,*bj_i,*bi,*bj,bnz,nextcj; /* bi,bj,ba: local array of C(mpi mat) */
  MPI_Request          *s_waits,*r_waits; 
  MPI_Status           *status;
  MatScalar            **abuf_r,*ba_i,*pA,*coa,*ba; 
  PetscInt             *ai,*aj,*coi,*coj; 
  PetscInt             *poJ=po->j,*pdJ=pd->j; 
  Mat                  A_loc;
  Mat_SeqAIJ           *a_loc;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);

  ptap  = c->ptap;
  merge = ptap->merge;

  /* 2) compute numeric C_seq = P_loc^T*A_loc */
  /*------------------------------------------*/
  /* get data from symbolic products */
  coi = merge->coi; coj = merge->coj;
  ierr = PetscMalloc((coi[pon]+1)*sizeof(MatScalar),&coa);CHKERRQ(ierr);
  ierr = PetscMemzero(coa,coi[pon]*sizeof(MatScalar));CHKERRQ(ierr);
  bi     = merge->bi; bj = merge->bj;
  owners = merge->rowmap->range;
  ierr   = PetscMalloc((bi[cm]+1)*sizeof(MatScalar),&ba);CHKERRQ(ierr);
  ierr   = PetscMemzero(ba,bi[cm]*sizeof(MatScalar));CHKERRQ(ierr);
 
  /* get A_loc by taking all local rows of A */
  A_loc = ptap->A_loc;
  ierr = MatMPIAIJGetLocalMat(A,MAT_REUSE_MATRIX,&A_loc);CHKERRQ(ierr);
  a_loc = (Mat_SeqAIJ*)(A_loc)->data; 
  ai   = a_loc->i; 
  aj   = a_loc->j;

  for (i=0; i<am; i++) {
    /* 2-a) put A[i,:] to dense array aval */
    anz = ai[i+1] - ai[i];
    adj = aj + ai[i];
    ada = a_loc->a + ai[i];
      
    /* 2-b) Compute Cseq = P_loc[i,:]^T*A[i,:] using outer product */
    /*-------------------------------------------------------------*/  
    /* put the value into Co=(p->B)^T*A (off-diagonal part, send to others) */
    pnz = po->i[i+1] - po->i[i];
    poJ = po->j + po->i[i];
    pA  = po->a + po->i[i];
    for (j=0; j<pnz; j++){ 
      row = poJ[j]; 
      cnz = coi[row+1] - coi[row];
      cj  = coj + coi[row]; 
      ca  = coa + coi[row];
      /* perform sparse axpy */
      nexta  = 0; 
      valtmp = pA[j];
      for (k=0; nexta<anz; k++) { 
        if (cj[k] == adj[nexta]){
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
    for (j=0; j<pnz; j++){  
      row = pdJ[j]; 
      cnz = bi[row+1] - bi[row];
      cj  = bj + bi[row]; 
      ca  = ba + bi[row];
      /* perform sparse axpy */
      nexta  = 0;
      valtmp = pA[j];
      for (k=0; nexta<anz; k++) { 
        if (cj[k] == adj[nexta]){
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
  ierr = PetscCommGetNewTag(comm,&taga);CHKERRQ(ierr);
  ierr = PetscPostIrecvScalar(comm,taga,merge->nrecv,merge->id_r,merge->len_r,&abuf_r,&r_waits);CHKERRQ(ierr);

  ierr = PetscMalloc2(merge->nsend+1,MPI_Request,&s_waits,size,MPI_Status,&status);CHKERRQ(ierr);
  for (proc=0,k=0; proc<size; proc++){  
    if (!len_s[proc]) continue;
    i = merge->owners_co[proc];
    ierr = MPI_Isend(coa+coi[i],len_s[proc],MPIU_MATSCALAR,proc,taga,comm,s_waits+k);CHKERRQ(ierr);
    k++;
  } 
  if (merge->nrecv) {ierr = MPI_Waitall(merge->nrecv,r_waits,status);CHKERRQ(ierr);}
  if (merge->nsend) {ierr = MPI_Waitall(merge->nsend,s_waits,status);CHKERRQ(ierr);}

  ierr = PetscFree2(s_waits,status);CHKERRQ(ierr);
  ierr = PetscFree(r_waits);CHKERRQ(ierr);
  ierr = PetscFree(coa);CHKERRQ(ierr);

  /* 4) insert local Cseq and received values into Cmpi */
  /*----------------------------------------------------*/
  ierr = PetscMalloc3(merge->nrecv,PetscInt**,&buf_ri_k,merge->nrecv,PetscInt*,&nextrow,merge->nrecv,PetscInt*,&nextci);CHKERRQ(ierr);
  for (k=0; k<merge->nrecv; k++){
    buf_ri_k[k] = buf_ri[k]; /* beginning of k-th recved i-structure */
    nrows       = *(buf_ri_k[k]);
    nextrow[k]  = buf_ri_k[k]+1;  /* next row number of k-th recved i-structure */
    nextci[k]   = buf_ri_k[k] + (nrows + 1);/* poins to the next i-structure of k-th recved i-structure  */
  }

  for (i=0; i<cm; i++) {
    row = owners[rank] + i; /* global row index of C_seq */
    bj_i = bj + bi[i];  /* col indices of the i-th row of C */
    ba_i = ba + bi[i]; 
    bnz  = bi[i+1] - bi[i];
    /* add received vals into ba */
    for (k=0; k<merge->nrecv; k++){ /* k-th received message */
      /* i-th row */
      if (i == *nextrow[k]) {
        cnz = *(nextci[k]+1) - *nextci[k]; 
        cj  = buf_rj[k] + *(nextci[k]);
        ca  = abuf_r[k] + *(nextci[k]);
        nextcj = 0;
        for (j=0; nextcj<cnz; j++){ 
          if (bj_i[j] == cj[nextcj]){ /* bcol == ccol */
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

/* This routine is modified from MatPtAPSymbolic_MPIAIJ_MPIAIJ() */
#undef __FUNCT__  
#define __FUNCT__ "MatTransposeMatMultSymbolic_MPIAIJ_MPIAIJ_Scalable"
PetscErrorCode MatTransposeMatMultSymbolic_MPIAIJ_MPIAIJ_Scalable(Mat P,Mat A,PetscReal fill,Mat *C)
{
  PetscErrorCode       ierr;
  Mat                  Cmpi,A_loc,POt,PDt; 
  Mat_PtAPMPI          *ptap;
  PetscFreeSpaceList   free_space=PETSC_NULL,current_space=PETSC_NULL;
  Mat_MPIAIJ           *p=(Mat_MPIAIJ*)P->data,*c;
  PetscInt             *pdti,*pdtj,*poti,*potj,*ptJ;
  PetscInt             nnz; 
  PetscInt             *lnk,*owners_co,*coi,*coj,i,k,pnz,row;
  PetscInt             am=A->rmap->n,pn=P->cmap->n;  
  MPI_Comm             comm=((PetscObject)A)->comm;
  PetscMPIInt          size,rank,tagi,tagj,*len_si,*len_s,*len_ri; 
  PetscInt             **buf_rj,**buf_ri,**buf_ri_k;
  PetscInt             len,proc,*dnz,*onz,*owners;
  PetscInt             nzi,*bi,*bj; 
  PetscInt             nrows,*buf_s,*buf_si,*buf_si_i,**nextrow,**nextci;
  MPI_Request          *swaits,*rwaits; 
  MPI_Status           *sstatus,rstatus;
  Mat_Merge_SeqsToMPI  *merge;
  PetscInt             *ai,*aj,*Jptr,anz,*prmap=p->garray,pon,nspacedouble=0,j;
  PetscReal            afill=1.0,afill_tmp;
  PetscInt             rstart = P->cmap->rstart,rmax,aN=A->cmap->N,Crmax;
  PetscScalar          *vals;
  Mat_SeqAIJ           *a_loc, *pdt,*pot;

  PetscFunctionBegin;
  /* check if matrix local sizes are compatible */
  if (A->rmap->rstart != P->rmap->rstart || A->rmap->rend != P->rmap->rend){
    SETERRQ4(comm,PETSC_ERR_ARG_SIZ,"Matrix local dimensions are incompatible, A (%D, %D) != P (%D,%D)",A->rmap->rstart,A->rmap->rend,P->rmap->rstart,P->rmap->rend);
  }

  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);

  /* create struct Mat_PtAPMPI and attached it to C later */
  ierr = PetscNew(Mat_PtAPMPI,&ptap);CHKERRQ(ierr);

  /* get A_loc by taking all local rows of A */
  ierr = MatMPIAIJGetLocalMat(A,MAT_INITIAL_MATRIX,&A_loc);CHKERRQ(ierr);
  ptap->A_loc = A_loc;
  a_loc = (Mat_SeqAIJ*)(A_loc)->data; 
  ai   = a_loc->i; 
  aj   = a_loc->j;
 
  /* determine symbolic Co=(p->B)^T*A - send to others */
  /*----------------------------------------------------*/
  ierr = MatTransposeSymbolic_SeqAIJ(p->A,&PDt);CHKERRQ(ierr); 
  pdt = (Mat_SeqAIJ*)PDt->data; 
  pdti = pdt->i; pdtj = pdt->j;

  ierr = MatTransposeSymbolic_SeqAIJ(p->B,&POt);CHKERRQ(ierr); 
  pot = (Mat_SeqAIJ*)POt->data; 
  poti = pot->i; potj = pot->j;

  /* then, compute symbolic Co = (p->B)^T*A */
  pon = (p->B)->cmap->n; /* total num of rows to be sent to other processors 
                         >= (num of nonzero rows of C_seq) - pn */
  ierr = PetscMalloc((pon+1)*sizeof(PetscInt),&coi);CHKERRQ(ierr);
  coi[0] = 0;

  /* set initial free space to be fill*(nnz(p->B) + nnz(A)) */
  nnz           = fill*(poti[pon] + ai[am]);
  ierr          = PetscFreeSpaceGet(nnz,&free_space);
  current_space = free_space;

  /* create and initialize a linked list */
  i = PetscMax(pdt->rmax,pot->rmax);
  Crmax = i*a_loc->rmax*size; /* non-scalable! */
  if (!Crmax || Crmax > aN) Crmax = aN;
  ierr = PetscLLCondensedCreate_Scalable(Crmax,&lnk);CHKERRQ(ierr);

  for (i=0; i<pon; i++) {
    nnz = 0;
    pnz = poti[i+1] - poti[i];
    ptJ = potj + poti[i];
    for (j=0; j<pnz; j++){
      row  = ptJ[j]; /* row of A_loc == col of Pot */
      anz  = ai[row+1] - ai[row];
      Jptr = aj + ai[row];
      /* add non-zero cols of AP into the sorted linked list lnk */
      ierr = PetscLLCondensedAddSorted_Scalable(anz,Jptr,lnk);CHKERRQ(ierr);
    }
    nnz = lnk[0];

    /* If free space is not available, double the total space in the list */
    if (current_space->local_remaining<nnz) {
      ierr = PetscFreeSpaceGet(nnz+current_space->total_array_size,&current_space);CHKERRQ(ierr);
      nspacedouble++;
    }

    /* Copy data into free space, and zero out denserows */
    ierr = PetscLLCondensedClean_Scalable(nnz,current_space->array,lnk);CHKERRQ(ierr);
    current_space->array           += nnz;
    current_space->local_used      += nnz;
    current_space->local_remaining -= nnz;
    coi[i+1] = coi[i] + nnz;
  }
  
  ierr = PetscMalloc((coi[pon]+1)*sizeof(PetscInt),&coj);CHKERRQ(ierr);
  ierr = PetscFreeSpaceContiguous(&free_space,coj);CHKERRQ(ierr);
  afill_tmp = (PetscReal)coi[pon]/(poti[pon] + ai[am]+1);
  if (afill_tmp > afill) afill = afill_tmp;
 
  /* send j-array (coj) of Co to other processors */
  /*----------------------------------------------*/
  /* determine row ownership */
  ierr = PetscNew(Mat_Merge_SeqsToMPI,&merge);CHKERRQ(ierr);
  ierr = PetscLayoutCreate(comm,&merge->rowmap);CHKERRQ(ierr);
  merge->rowmap->n = pn;
  merge->rowmap->bs = 1;
  ierr = PetscLayoutSetUp(merge->rowmap);CHKERRQ(ierr);
  owners = merge->rowmap->range;

  /* determine the number of messages to send, their lengths */
  ierr = PetscMalloc(size*sizeof(PetscMPIInt),&len_si);CHKERRQ(ierr);
  ierr = PetscMemzero(len_si,size*sizeof(PetscMPIInt));CHKERRQ(ierr);
  ierr = PetscMalloc(size*sizeof(PetscMPIInt),&merge->len_s);CHKERRQ(ierr);
  len_s = merge->len_s;
  merge->nsend = 0;
  
  ierr = PetscMalloc((size+2)*sizeof(PetscInt),&owners_co);CHKERRQ(ierr);
  ierr = PetscMemzero(len_s,size*sizeof(PetscMPIInt));CHKERRQ(ierr);

  proc = 0;
  for (i=0; i<pon; i++){
    while (prmap[i] >= owners[proc+1]) proc++;
    len_si[proc]++;  /* num of rows in Co to be sent to [proc] */
    len_s[proc] += coi[i+1] - coi[i];
  }

  len   = 0;  /* max length of buf_si[] */
  owners_co[0] = 0;
  for (proc=0; proc<size; proc++){
    owners_co[proc+1] = owners_co[proc] + len_si[proc];
    if (len_si[proc]){ 
      merge->nsend++;
      len_si[proc] = 2*(len_si[proc] + 1);
      len += len_si[proc]; 
    }
  }

  /* determine the number and length of messages to receive for coi and coj  */
  ierr = PetscGatherNumberOfMessages(comm,PETSC_NULL,len_s,&merge->nrecv);CHKERRQ(ierr);
  ierr = PetscGatherMessageLengths2(comm,merge->nsend,merge->nrecv,len_s,len_si,&merge->id_r,&merge->len_r,&len_ri);CHKERRQ(ierr);      

  /* post the Irecv and Isend of coj */
  ierr = PetscCommGetNewTag(comm,&tagj);CHKERRQ(ierr);
  ierr = PetscPostIrecvInt(comm,tagj,merge->nrecv,merge->id_r,merge->len_r,&buf_rj,&rwaits);CHKERRQ(ierr);
  ierr = PetscMalloc((merge->nsend+1)*sizeof(MPI_Request),&swaits);CHKERRQ(ierr);
  for (proc=0, k=0; proc<size; proc++){  
    if (!len_s[proc]) continue;
    i = owners_co[proc];
    ierr = MPI_Isend(coj+coi[i],len_s[proc],MPIU_INT,proc,tagj,comm,swaits+k);CHKERRQ(ierr);
    k++;
  } 

  /* receives and sends of coj are complete */
  ierr = PetscMalloc(size*sizeof(MPI_Status),&sstatus);CHKERRQ(ierr); 
  for (i=0; i<merge->nrecv; i++){
    PetscMPIInt icompleted;
    ierr = MPI_Waitany(merge->nrecv,rwaits,&icompleted,&rstatus);CHKERRQ(ierr);
  }
  ierr = PetscFree(rwaits);CHKERRQ(ierr);
  if (merge->nsend) {ierr = MPI_Waitall(merge->nsend,swaits,sstatus);CHKERRQ(ierr);}
  
  /* send and recv coi */
  /*-------------------*/  
  ierr = PetscCommGetNewTag(comm,&tagi);CHKERRQ(ierr);
  ierr = PetscPostIrecvInt(comm,tagi,merge->nrecv,merge->id_r,len_ri,&buf_ri,&rwaits);CHKERRQ(ierr);
  ierr = PetscMalloc((len+1)*sizeof(PetscInt),&buf_s);CHKERRQ(ierr); 
  buf_si = buf_s;  /* points to the beginning of k-th msg to be sent */
  for (proc=0,k=0; proc<size; proc++){  
    if (!len_s[proc]) continue;
    /* form outgoing message for i-structure: 
         buf_si[0]:                 nrows to be sent
               [1:nrows]:           row index (global)
               [nrows+1:2*nrows+1]: i-structure index
    */
    /*-------------------------------------------*/      
    nrows = len_si[proc]/2 - 1; 
    buf_si_i    = buf_si + nrows+1;
    buf_si[0]   = nrows;
    buf_si_i[0] = 0;
    nrows = 0;
    for (i=owners_co[proc]; i<owners_co[proc+1]; i++){
      nzi = coi[i+1] - coi[i];
      buf_si_i[nrows+1] = buf_si_i[nrows] + nzi; /* i-structure */
      buf_si[nrows+1] =prmap[i] -owners[proc]; /* local row index */
      nrows++;
    }
    ierr = MPI_Isend(buf_si,len_si[proc],MPIU_INT,proc,tagi,comm,swaits+k);CHKERRQ(ierr);
    k++;
    buf_si += len_si[proc];
  } 
  i = merge->nrecv;
  while (i--) {
    PetscMPIInt icompleted;
    ierr = MPI_Waitany(merge->nrecv,rwaits,&icompleted,&rstatus);CHKERRQ(ierr);
  }
  ierr = PetscFree(rwaits);CHKERRQ(ierr);
  if (merge->nsend) {ierr = MPI_Waitall(merge->nsend,swaits,sstatus);CHKERRQ(ierr);}
  ierr = PetscFree(len_si);CHKERRQ(ierr);
  ierr = PetscFree(len_ri);CHKERRQ(ierr);
  ierr = PetscFree(swaits);CHKERRQ(ierr);
  ierr = PetscFree(sstatus);CHKERRQ(ierr);
  ierr = PetscFree(buf_s);CHKERRQ(ierr);

  /* compute the local portion of C (mpi mat) */
  /*------------------------------------------*/
  /* allocate bi array and free space for accumulating nonzero column info */
  ierr = PetscMalloc((pn+1)*sizeof(PetscInt),&bi);CHKERRQ(ierr);
  bi[0] = 0;

  /* set initial free space to be fill*(nnz(P) + nnz(AP)) */
  nnz           = fill*(pdti[pn] + poti[pon] + ai[am]);
  ierr          = PetscFreeSpaceGet(nnz,&free_space);
  current_space = free_space;

  ierr = PetscMalloc3(merge->nrecv,PetscInt**,&buf_ri_k,merge->nrecv,PetscInt*,&nextrow,merge->nrecv,PetscInt*,&nextci);CHKERRQ(ierr);
  for (k=0; k<merge->nrecv; k++){
    buf_ri_k[k] = buf_ri[k]; /* beginning of k-th recved i-structure */
    nrows       = *buf_ri_k[k];
    nextrow[k]  = buf_ri_k[k] + 1;  /* next row number of k-th recved i-structure */
    nextci[k]   = buf_ri_k[k] + (nrows + 1);/* poins to the next i-structure of k-th recved i-structure  */
  }

  ierr = MatPreallocateInitialize(comm,pn,A->cmap->n,dnz,onz);CHKERRQ(ierr);
  rmax = 0;
  for (i=0; i<pn; i++) {
    /* add pdt[i,:]*AP into lnk */
    pnz = pdti[i+1] - pdti[i];
    ptJ = pdtj + pdti[i];
    for (j=0; j<pnz; j++){
      row  = ptJ[j];  /* row of AP == col of Pt */
      anz  = ai[row+1] - ai[row];
      Jptr = aj + ai[row];
      /* add non-zero cols of AP into the sorted linked list lnk */
      ierr = PetscLLCondensedAddSorted_Scalable(anz,Jptr,lnk);CHKERRQ(ierr);
    }

    /* add received col data into lnk */
    for (k=0; k<merge->nrecv; k++){ /* k-th received message */
      if (i == *nextrow[k]) { /* i-th row */
        nzi = *(nextci[k]+1) - *nextci[k]; 
        Jptr  = buf_rj[k] + *nextci[k];
        ierr = PetscLLCondensedAddSorted_Scalable(nzi,Jptr,lnk);CHKERRQ(ierr);
        nextrow[k]++; nextci[k]++;
      }
    }
    nnz = lnk[0]; 

    /* if free space is not available, make more free space */
    if (current_space->local_remaining<nnz) {
      ierr = PetscFreeSpaceGet(nnz+current_space->total_array_size,&current_space);CHKERRQ(ierr);
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

  ierr = PetscMalloc((bi[pn]+1)*sizeof(PetscInt),&bj);CHKERRQ(ierr);
  ierr = PetscFreeSpaceContiguous(&free_space,bj);CHKERRQ(ierr);
  afill_tmp = (PetscReal)bi[pn]/(pdti[pn] + poti[pon] + ai[am]+1); 
  if (afill_tmp > afill) afill = afill_tmp;
  ierr = PetscLLCondensedDestroy_Scalable(lnk);CHKERRQ(ierr);
  ierr = MatDestroy(&POt);CHKERRQ(ierr);
  ierr = MatDestroy(&PDt);CHKERRQ(ierr);

  /* create symbolic parallel matrix Cmpi - why cannot be assembled in Numeric part   */
  /*----------------------------------------------------------------------------------*/
  ierr = PetscMalloc((rmax+1)*sizeof(PetscScalar),&vals);CHKERRQ(ierr); 
  ierr = PetscMemzero(vals,rmax*sizeof(PetscScalar));CHKERRQ(ierr);

  ierr = MatCreate(comm,&Cmpi);CHKERRQ(ierr);
  ierr = MatSetSizes(Cmpi,pn,A->cmap->n,PETSC_DETERMINE,PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = MatSetBlockSizes(Cmpi,P->cmap->bs,A->cmap->bs); CHKERRQ(ierr);
  ierr = MatSetType(Cmpi,MATMPIAIJ); CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(Cmpi,0,dnz,0,onz);CHKERRQ(ierr);
  ierr = MatPreallocateFinalize(dnz,onz);CHKERRQ(ierr);
  ierr = MatSetBlockSize(Cmpi,1);CHKERRQ(ierr);
  for (i=0; i<pn; i++){
    row = i + rstart;
    nnz = bi[i+1] - bi[i];
    Jptr = bj + bi[i];
    ierr = MatSetValues(Cmpi,1,&row,nnz,Jptr,vals,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(Cmpi,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Cmpi,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = PetscFree(vals);CHKERRQ(ierr);

  merge->bi            = bi;
  merge->bj            = bj;
  merge->coi           = coi;
  merge->coj           = coj;
  merge->buf_ri        = buf_ri;
  merge->buf_rj        = buf_rj;
  merge->owners_co     = owners_co;
  merge->destroy       = Cmpi->ops->destroy;
  merge->duplicate     = Cmpi->ops->duplicate;

  Cmpi->ops->mattransposemultnumeric = MatTransposeMatMultNumeric_MPIAIJ_MPIAIJ_Scalable;
  Cmpi->ops->destroy                 = MatDestroy_MPIAIJ_PtAP;  

  /* attach the supporting struct to Cmpi for reuse */
  c = (Mat_MPIAIJ*)Cmpi->data;
  c->ptap        = ptap;
  ptap->api      = PETSC_NULL;
  ptap->apj      = PETSC_NULL;
  ptap->merge    = merge;
  ptap->rmax     = rmax;
  ptap->apa      = PETSC_NULL;
 
  *C = Cmpi;
#if defined(PETSC_USE_INFO)
  if (bi[pn] != 0) {
    ierr = PetscInfo3(Cmpi,"Reallocs %D; Fill ratio: given %G needed %G.\n",nspacedouble,fill,afill);CHKERRQ(ierr);
    ierr = PetscInfo1(Cmpi,"Use MatTransposeMatMult(A,B,MatReuse,%G,&C) for best performance.\n",afill);CHKERRQ(ierr);
  } else {
    ierr = PetscInfo(Cmpi,"Empty matrix product\n");CHKERRQ(ierr);
  }
#endif
  PetscFunctionReturn(0);
}
