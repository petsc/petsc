
/*
  Defines matrix-matrix product routines for pairs of MPIAIJ matrices
          C = A * B
*/
#include <../src/mat/impls/aij/seq/aij.h> /*I "petscmat.h" I*/
#include <../src/mat/utils/freespace.h>
#include <../src/mat/impls/aij/mpi/mpiaij.h>
#include <petscbt.h>
#include <../src/mat/impls/dense/mpi/mpidense.h>
/*
#define DEBUG_MATMATMULT
 */

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
#define __FUNCT__ "PetscContainerDestroy_Mat_MatMatMultMPI"
PetscErrorCode PetscContainerDestroy_Mat_MatMatMultMPI(void *ptr)
{
  PetscErrorCode       ierr;
  Mat_MatMatMultMPI    *mult=(Mat_MatMatMultMPI*)ptr;

  PetscFunctionBegin;
  ierr = ISDestroy(&mult->isrowa);CHKERRQ(ierr);
  ierr = ISDestroy(&mult->isrowb);CHKERRQ(ierr);
  ierr = ISDestroy(&mult->iscolb);CHKERRQ(ierr);
  ierr = MatDestroy(&mult->C_seq);CHKERRQ(ierr);
  ierr = MatDestroy(&mult->A_loc);CHKERRQ(ierr);
  ierr = MatDestroy(&mult->B_seq);CHKERRQ(ierr);
  ierr = PetscFree(mult);CHKERRQ(ierr);
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
  ierr = PetscFree2(ptap->startsj,ptap->startsj_r);CHKERRQ(ierr);
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
#define __FUNCT__ "MatDestroy_MPIAIJ_MatMatMult_32"
PetscErrorCode MatDestroy_MPIAIJ_MatMatMult_32(Mat A)
{
  PetscErrorCode     ierr;
  PetscContainer     container;
  Mat_MatMatMultMPI  *mult=PETSC_NULL;

  PetscFunctionBegin;
  ierr = PetscObjectQuery((PetscObject)A,"Mat_MatMatMultMPI",(PetscObject *)&container);CHKERRQ(ierr);
  if (!container) SETERRQ(((PetscObject)A)->comm,PETSC_ERR_PLIB,"Container does not exist");
  ierr = PetscContainerGetPointer(container,(void **)&mult);CHKERRQ(ierr);
  A->ops->destroy   = mult->destroy;
  A->ops->duplicate = mult->duplicate;
  if (A->ops->destroy) {
    ierr = (*A->ops->destroy)(A);CHKERRQ(ierr);
  }
  ierr = PetscObjectCompose((PetscObject)A,"Mat_MatMatMultMPI",0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatDuplicate_MPIAIJ_MatMatMult_32"
PetscErrorCode MatDuplicate_MPIAIJ_MatMatMult_32(Mat A, MatDuplicateOption op, Mat *M)
{
  PetscErrorCode     ierr;
  Mat_MatMatMultMPI  *mult;
  PetscContainer     container;

  PetscFunctionBegin;
  ierr = PetscObjectQuery((PetscObject)A,"Mat_MatMatMultMPI",(PetscObject *)&container);CHKERRQ(ierr);
  if (!container) SETERRQ(((PetscObject)A)->comm,PETSC_ERR_PLIB,"Container does not exist");
  ierr  = PetscContainerGetPointer(container,(void **)&mult);CHKERRQ(ierr);
  /* Note: the container is not duplicated, because it requires deep copying of
     several large data sets (see PetscContainerDestroy_Mat_MatMatMultMPI()).
     These data sets are only used for repeated calling of MatMatMultNumeric().
     *M is unlikely being used in this way. Thus we create *M with pure mpiaij format */
  ierr = (*mult->duplicate)(A,op,M);CHKERRQ(ierr);
  (*M)->ops->destroy   = mult->destroy;   /* = MatDestroy_MPIAIJ, *M doesn't duplicate A's container! */
  (*M)->ops->duplicate = mult->duplicate; /* = MatDuplicate_MPIAIJ */
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
#if defined(DEBUG_MATMATMULT)
  PetscMPIInt        rank=a->rank;
#endif

  PetscFunctionBegin;
#if defined(DEBUG_MATMATMULT)
  if (!rank) ierr = PetscPrintf(PETSC_COMM_SELF,"[%d] call MatMatMultNumeric_MPIAIJ_MPIAIJ()...\n",rank);
#endif

  /* 1) get P_oth = ptap->P_oth  and P_loc = ptap->P_loc */
  /*-----------------------------------------------------*/ 
  /* update numerical values of P_oth and P_loc */
  ierr = MatGetBrowsOfAoCols_MPIAIJ(A,P,MAT_REUSE_MATRIX,&ptap->startsj,&ptap->startsj_r,&ptap->bufa,&ptap->P_oth);CHKERRQ(ierr);  
  ierr = MatMPIAIJGetLocalMat(P,MAT_REUSE_MATRIX,&ptap->P_loc);CHKERRQ(ierr);
#if defined(DEBUG_MATMATMULT)
  if (!rank) ierr = PetscPrintf(PETSC_COMM_SELF,"[%d] got P_oth and P_loc...\n",rank);
#endif

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
#if defined(DEBUG_MATMATMULT)
  PetscMPIInt          rank=a->rank;
#endif

  PetscFunctionBegin;
  if (A->cmap->rstart != P->rmap->rstart || A->cmap->rend != P->rmap->rend){
    SETERRQ4(comm,PETSC_ERR_ARG_SIZ,"Matrix local dimensions are incompatible, (%D, %D) != (%D,%D)",A->cmap->rstart,A->cmap->rend,P->rmap->rstart,P->rmap->rend);
  } 

  ierr = PetscObjectOptionsBegin((PetscObject)A);CHKERRQ(ierr);
    
    ierr = PetscOptionsBool("-matmatmult_32","Use a scalable but slower C=A*B","",scalable,&scalable,PETSC_NULL);CHKERRQ(ierr);
    if (scalable){
      ierr = MatMatMultSymbolic_MPIAIJ_MPIAIJ_Scalable_32(A,P,fill,C);;CHKERRQ(ierr);
      PetscFunctionReturn(0);
    }
    ierr = PetscOptionsBool("-matmatmult_scalable","Use a scalable but slower C=A*B","",scalable,&scalable,PETSC_NULL);CHKERRQ(ierr);
    if (scalable){
      ierr = MatMatMultSymbolic_MPIAIJ_MPIAIJ_Scalable(A,P,fill,C);;CHKERRQ(ierr);
      PetscFunctionReturn(0);
    }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  
#if defined(DEBUG_MATMATMULT)
  if (!rank) ierr = PetscPrintf(PETSC_COMM_SELF,"[%d] call MatMatMultSymbolic_MPIAIJ_MPIAIJ()...\n",rank);
#endif

  /* create struct Mat_PtAPMPI and attached it to C later */
  ierr = PetscNew(Mat_PtAPMPI,&ptap);CHKERRQ(ierr); 

  /* get P_oth by taking rows of P (= non-zero cols of local A) from other processors */
  ierr = MatGetBrowsOfAoCols_MPIAIJ(A,P,MAT_INITIAL_MATRIX,&ptap->startsj,&ptap->startsj_r,&ptap->bufa,&ptap->P_oth);CHKERRQ(ierr);
#if defined(DEBUG_MATMATMULT)
  if (!rank) ierr = PetscPrintf(PETSC_COMM_SELF,"[%d] P_oth is done...\n",rank);
#endif
  /* get P_loc by taking all local rows of P */
  ierr = MatMPIAIJGetLocalMat(P,MAT_INITIAL_MATRIX,&ptap->P_loc);CHKERRQ(ierr);

  p_loc = (Mat_SeqAIJ*)(ptap->P_loc)->data; 
  p_oth = (Mat_SeqAIJ*)(ptap->P_oth)->data;
  pi_loc = p_loc->i; pj_loc = p_loc->j; 
  pi_oth = p_oth->i; pj_oth = p_oth->j;

#if defined(DEBUG_MATMATMULT)
  if (!rank) ierr = PetscPrintf(PETSC_COMM_SELF,"[%d] P_loc is done, Annz %D * P_locnnz %D = %D...\n",rank,ad->i[am]+ao->i[am],p_loc->rmax,(ad->i[am]+ao->i[am])*p_loc->rmax);
#endif

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
#if defined(DEBUG_MATMATMULT)
  if (!rank) ierr = PetscPrintf(PETSC_COMM_SELF,"[%d] pN %d; nlnk_max %d; Armax %d+%d=%d; Prmax Max(%d,%d)=%d\n",rank,pN,nlnk_max,ad->rmax,ao->rmax,armax,p_loc->rmax,p_oth->rmax,prmax);
#endif
  ierr = PetscLLCondensedCreate(nlnk_max,pN,lnk,lnkbt);CHKERRQ(ierr);

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
      ierr = PetscLLCondensedAddSorted(nlnk_max,pN,pnz,Jptr,lnk,lnkbt);CHKERRQ(ierr);
    }
    /* off-diagonal portion of A */
    nzi = aoi[i+1] - aoi[i];
    for (j=0; j<nzi; j++){   
      row = *aoj++; 
      pnz = pi_oth[row+1] - pi_oth[row];
      Jptr  = pj_oth + pi_oth[row];  
      ierr = PetscLLCondensedAddSorted(nlnk_max,pN,pnz,Jptr,lnk,lnkbt);CHKERRQ(ierr);
    }

    apnz     = lnk[0];
    api[i+1] = api[i] + apnz;

    /* if free space is not available, double the total space in the list */
    if (current_space->local_remaining<apnz) {
      ierr = PetscFreeSpaceGet(apnz+current_space->total_array_size,&current_space);CHKERRQ(ierr);
      nspacedouble++;
    }

    /* Copy data into free space, then initialize lnk */
    ierr = PetscLLCondensedClean(nlnk_max,pN,apnz,current_space->array,lnk,lnkbt);CHKERRQ(ierr);
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
#if defined(DEBUG_MATMATMULT)
  if (!rank) ierr = PetscPrintf(PETSC_COMM_SELF,"[%d] AP is done...\n",rank);
#endif

  /* malloc apa to store dense row A[i,:]*P */ 
  ierr = PetscMalloc(pN*sizeof(PetscScalar),&apa);CHKERRQ(ierr);
  ierr = PetscMemzero(apa,pN*sizeof(PetscScalar));CHKERRQ(ierr);
  ptap->apa = apa;
#if defined(DEBUG_MATMATMULT)
  if (!rank) ierr = PetscPrintf(PETSC_COMM_SELF,"[%d] Malloc apa pN %D is done...\n",rank,pN);
#endif

  /* create and assemble symbolic parallel matrix Cmpi */
  /*----------------------------------------------------*/
  ierr = MatCreate(comm,&Cmpi);CHKERRQ(ierr);
  ierr = MatSetSizes(Cmpi,am,pn,PETSC_DETERMINE,PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = MatSetType(Cmpi,MATMPIAIJ);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(Cmpi,0,dnz,0,onz);CHKERRQ(ierr);
  ierr = MatPreallocateFinalize(dnz,onz);CHKERRQ(ierr);
  ierr = MatSetBlockSize(Cmpi,1);CHKERRQ(ierr);
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
  afill = (PetscReal)api[am]/(adi[am]+aoi[am]+pi_loc[pm]) + 1.e-5;
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

/* implementation used in PETSc-3.2 */
/* This routine is called ONLY in the case of reusing previously computed symbolic C */
#undef __FUNCT__
#define __FUNCT__ "MatMatMultNumeric_MPIAIJ_MPIAIJ_Scalable_32"
PetscErrorCode MatMatMultNumeric_MPIAIJ_MPIAIJ_Scalable_32(Mat A,Mat B,Mat C)
{
  PetscErrorCode     ierr;
  Mat                *seq;
  Mat_MatMatMultMPI  *mult;
  PetscContainer     container;

  PetscFunctionBegin;
  ierr = PetscObjectQuery((PetscObject)C,"Mat_MatMatMultMPI",(PetscObject *)&container);CHKERRQ(ierr);
  if (!container) SETERRQ(((PetscObject)A)->comm,PETSC_ERR_PLIB,"Container does not exist");
  ierr  = PetscContainerGetPointer(container,(void **)&mult);CHKERRQ(ierr);

  if (mult->skipNumeric){ /* first numeric product is done during symbolic product */
    mult->skipNumeric = PETSC_FALSE;
    PetscFunctionReturn(0);
  }
#if defined(DEBUG_MATMATMULT)
  PetscMPIInt rank;
  MPI_Comm    comm = ((PetscObject)C)->comm;
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  ierr = MPI_Barrier(comm);CHKERRQ(ierr);
  if (!rank) ierr = PetscPrintf(PETSC_COMM_SELF,"[%d] call MatMatMultNumeric_MPIAIJ_MPIAIJ_Scalable_32()...\n",rank);
#endif

  seq = &mult->B_seq;
  ierr = MatGetSubMatrices(B,1,&mult->isrowb,&mult->iscolb,MAT_REUSE_MATRIX,&seq);CHKERRQ(ierr);
  mult->B_seq = *seq;

  seq = &mult->A_loc;
  ierr = MatGetSubMatrices(A,1,&mult->isrowa,&mult->isrowb,MAT_REUSE_MATRIX,&seq);CHKERRQ(ierr);
  mult->A_loc = *seq;

  ierr = MatMatMultNumeric_SeqAIJ_SeqAIJ_Scalable(mult->A_loc,mult->B_seq,mult->C_seq);CHKERRQ(ierr);

  ierr = PetscObjectReference((PetscObject)mult->C_seq);CHKERRQ(ierr);
  ierr = MatMerge(((PetscObject)A)->comm,mult->C_seq,B->cmap->n,MAT_REUSE_MATRIX,&C);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* numeric product is computed as well */
#undef __FUNCT__
#define __FUNCT__ "MatMatMultSymbolic_MPIAIJ_MPIAIJ_Scalable_32"
PetscErrorCode MatMatMultSymbolic_MPIAIJ_MPIAIJ_Scalable_32(Mat A,Mat B,PetscReal fill,Mat *C)
{
  PetscErrorCode     ierr;
  Mat_MatMatMultMPI  *mult;
  PetscContainer     container;
  Mat                AB,*seq;
  Mat_MPIAIJ         *a=(Mat_MPIAIJ*)A->data;
  PetscInt           *idx,i,start,ncols,nzA,nzB,*cmap,imark;
#if defined(DEBUG_MATMATMULT)
  MPI_Comm             comm = ((PetscObject)A)->comm;
  PetscMPIInt          rank=a->rank;
#endif

  PetscFunctionBegin;
#if defined(DEBUG_MATMATMULT)
  if (!rank) ierr = PetscPrintf(PETSC_COMM_SELF,"[%d] call MatMatMultSymbolic_MPIAIJ_MPIAIJ_Scalable_32()...\n",rank);
#endif
  if (A->cmap->rstart != B->rmap->rstart || A->cmap->rend != B->rmap->rend){
    SETERRQ4(((PetscObject)A)->comm,PETSC_ERR_ARG_SIZ,"Matrix local dimensions are incompatible, (%D, %D) != (%D,%D)",A->cmap->rstart,A->cmap->rend,B->rmap->rstart,B->rmap->rend);
  }
  
  ierr = PetscNew(Mat_MatMatMultMPI,&mult);CHKERRQ(ierr);

  /* get isrowb: nonzero col of A */ 
  start = A->cmap->rstart;
  cmap  = a->garray;
  nzA   = a->A->cmap->n; 
  nzB   = a->B->cmap->n;
  ierr  = PetscMalloc((nzA+nzB)*sizeof(PetscInt), &idx);CHKERRQ(ierr);
  ncols = 0;
  for (i=0; i<nzB; i++) {  /* row < local row index */
    if (cmap[i] < start) idx[ncols++] = cmap[i];
    else break;
  }
  imark = i;
  for (i=0; i<nzA; i++) idx[ncols++] = start + i;  /* local rows */
  for (i=imark; i<nzB; i++) idx[ncols++] = cmap[i]; /* row > local row index */
  ierr = ISCreateGeneral(PETSC_COMM_SELF,ncols,idx,PETSC_OWN_POINTER,&mult->isrowb);CHKERRQ(ierr);
  ierr = ISCreateStride(PETSC_COMM_SELF,B->cmap->N,0,1,&mult->iscolb);CHKERRQ(ierr);

  /*  get isrowa: all local rows of A */
  ierr = ISCreateStride(PETSC_COMM_SELF,A->rmap->n,A->rmap->rstart,1,&mult->isrowa);CHKERRQ(ierr);

  /* Below should go to MatMatMultNumeric_MPIAIJ_MPIAIJ() - How to generate C there? */
  /* create a seq matrix B_seq = submatrix of B by taking rows of B that equal to nonzero col of A */
  ierr = MatGetSubMatrices(B,1,&mult->isrowb,&mult->iscolb,MAT_INITIAL_MATRIX,&seq);CHKERRQ(ierr);
  mult->B_seq = *seq;
  ierr = PetscFree(seq);CHKERRQ(ierr);
#if defined(DEBUG_MATMATMULT)
  if (!rank) ierr = PetscPrintf(PETSC_COMM_SELF,"[%d] B_seq is done...\n",rank);
#endif

  /*  create a seq matrix A_seq = submatrix of A by taking all local rows of A */
  ierr = MatGetSubMatrices(A,1,&mult->isrowa,&mult->isrowb,MAT_INITIAL_MATRIX,&seq);CHKERRQ(ierr); 
  mult->A_loc = *seq;
  ierr = PetscFree(seq);CHKERRQ(ierr);
#if defined(DEBUG_MATMATMULT)
  if (!rank) ierr = PetscPrintf(PETSC_COMM_SELF,"[%d] A_loc is done...\n",rank);
#endif

  /* compute C_seq = A_seq * B_seq */
  ierr = MatMatMultSymbolic_SeqAIJ_SeqAIJ_Scalable(mult->A_loc,mult->B_seq,fill,&mult->C_seq);CHKERRQ(ierr);
#if defined(DEBUG_MATMATMULT)
  ierr = MPI_Barrier(comm);CHKERRQ(ierr);
  if (!rank) ierr = PetscPrintf(PETSC_COMM_SELF,"[%d] C_seq Symbolic is done...\n",rank);
#endif  
  ierr = MatMatMultNumeric_SeqAIJ_SeqAIJ_Scalable(mult->A_loc,mult->B_seq,mult->C_seq);CHKERRQ(ierr);
#if defined(DEBUG_MATMATMULT)
  if (!rank) ierr = PetscPrintf(PETSC_COMM_SELF,"[%d] C_seq Numeric is done...\n",rank);
#endif
 
  /* create mpi matrix C by concatinating C_seq */
  ierr = PetscObjectReference((PetscObject)mult->C_seq);CHKERRQ(ierr); /* prevent C_seq being destroyed by MatMerge() */
  ierr = MatMergeSymbolic(((PetscObject)A)->comm,mult->C_seq,B->cmap->n,&AB);CHKERRQ(ierr);
  ierr = MatMergeNumeric(((PetscObject)A)->comm,mult->C_seq,B->cmap->n,AB);CHKERRQ(ierr);
#if defined(DEBUG_MATMATMULT)
  if (!rank) ierr = PetscPrintf(PETSC_COMM_SELF,"[%d] Merge is done...\n",rank);
#endif

  /* attach the supporting struct to C for reuse of symbolic C */
  ierr = PetscContainerCreate(PETSC_COMM_SELF,&container);CHKERRQ(ierr);
  ierr = PetscContainerSetPointer(container,mult);CHKERRQ(ierr);
  ierr = PetscContainerSetUserDestroy(container,PetscContainerDestroy_Mat_MatMatMultMPI);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)AB,"Mat_MatMatMultMPI",(PetscObject)container);CHKERRQ(ierr);
  ierr = PetscContainerDestroy(&container);CHKERRQ(ierr);
  mult->skipNumeric  = PETSC_TRUE; /* a numeric product is done here */
  mult->destroy      = AB->ops->destroy;
  mult->duplicate    = AB->ops->duplicate;
  AB->ops->matmultnumeric = MatMatMultNumeric_MPIAIJ_MPIAIJ_Scalable_32;
  AB->ops->destroy        = MatDestroy_MPIAIJ_MatMatMult_32;
  AB->ops->duplicate      = MatDuplicate_MPIAIJ_MatMatMult_32;
  AB->ops->matmult        = MatMatMult_MPIAIJ_MPIAIJ;
  *C                      = AB;
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
#define __FUNCT__ "MPIAIJ_MPIDenseDestroy"
PetscErrorCode MPIAIJ_MPIDenseDestroy(void *ctx)
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
  ierr = MatSetType(*C,MATMPIDENSE);CHKERRQ(ierr);
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
  ierr = PetscContainerSetUserDestroy(container,MPIAIJ_MPIDenseDestroy);CHKERRQ(ierr);
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
#if defined(DEBUG_MATMATMULT)
  PetscMPIInt        rank=a->rank;
#endif

  PetscFunctionBegin;
#if defined(DEBUG_MATMATMULT)
  if (!rank) ierr = PetscPrintf(PETSC_COMM_SELF,"[%d] call MatMatMultNumeric_MPIAIJ_MPIAIJ_Scalable()...\n",rank);
#endif

  /* 1) get P_oth = ptap->P_oth  and P_loc = ptap->P_loc */
  /*-----------------------------------------------------*/ 
  /* update numerical values of P_oth and P_loc */
  ierr = MatGetBrowsOfAoCols_MPIAIJ(A,P,MAT_REUSE_MATRIX,&ptap->startsj,&ptap->startsj_r,&ptap->bufa,&ptap->P_oth);CHKERRQ(ierr);  
  ierr = MatMPIAIJGetLocalMat(P,MAT_REUSE_MATRIX,&ptap->P_loc);CHKERRQ(ierr);
#if defined(DEBUG_MATMATMULT)
  if (!rank) ierr = PetscPrintf(PETSC_COMM_SELF,"[%d] got P_oth and P_loc...\n",rank);
#endif

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
#if defined(DEBUG_MATMATMULT)
  PetscMPIInt          rank=a->rank;
#endif

  PetscFunctionBegin;
#if defined(DEBUG_MATMATMULT)
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  if (!rank) ierr = PetscPrintf(PETSC_COMM_SELF,"[%d] call MatMatMultSymbolic_MPIAIJ_MPIAIJ_Scalable()...\n",rank);
#endif

  /* create struct Mat_PtAPMPI and attached it to C later */
  ierr = PetscNew(Mat_PtAPMPI,&ptap);CHKERRQ(ierr); 

  /* get P_oth by taking rows of P (= non-zero cols of local A) from other processors */
  ierr = MatGetBrowsOfAoCols_MPIAIJ(A,P,MAT_INITIAL_MATRIX,&ptap->startsj,&ptap->startsj_r,&ptap->bufa,&ptap->P_oth);CHKERRQ(ierr);
#if defined(DEBUG_MATMATMULT)
  if (!rank) ierr = PetscPrintf(PETSC_COMM_SELF,"[%d] P_oth is done...\n",rank);
#endif
  /* get P_loc by taking all local rows of P */
  ierr = MatMPIAIJGetLocalMat(P,MAT_INITIAL_MATRIX,&ptap->P_loc);CHKERRQ(ierr);

  p_loc = (Mat_SeqAIJ*)(ptap->P_loc)->data; 
  p_oth = (Mat_SeqAIJ*)(ptap->P_oth)->data;
  pi_loc = p_loc->i; pj_loc = p_loc->j; 
  pi_oth = p_oth->i; pj_oth = p_oth->j;

#if defined(DEBUG_MATMATMULT)
  if (!rank) ierr = PetscPrintf(PETSC_COMM_SELF,"[%d] P_loc is done, Annz %D * P_locnnz %D = %D...\n",rank,ad->i[am]+ao->i[am],p_loc->rmax,(ad->i[am]+ao->i[am])*p_loc->rmax);
#endif

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
#if defined(DEBUG_MATMATMULT)
  if (!rank) ierr = PetscPrintf(PETSC_COMM_SELF,"[%d] pN %d; nlnk_max %d; Armax %d+%d=%d; Prmax Max(%d,%d)=%d\n",rank,pN,nlnk_max,ad->rmax,ao->rmax,armax,p_loc->rmax,p_oth->rmax,prmax);
#endif
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
#if defined(DEBUG_MATMATMULT)
  if (!rank) ierr = PetscPrintf(PETSC_COMM_SELF,"[%d] AP is done..., apnz_max %d\n",rank,apnz_max);
#endif

  /* create and assemble symbolic parallel matrix Cmpi */
  /*----------------------------------------------------*/
  ierr = MatCreate(comm,&Cmpi);CHKERRQ(ierr);
  ierr = MatSetSizes(Cmpi,am,pn,PETSC_DETERMINE,PETSC_DETERMINE);CHKERRQ(ierr);
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
  afill = (PetscReal)api[am]/(adi[am]+aoi[am]+pi_loc[pm]) + 1.e-5;
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
