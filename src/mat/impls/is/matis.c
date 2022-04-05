/*
    Creates a matrix class for using the Neumann-Neumann type preconditioners.
    This stores the matrices in globally unassembled form. Each processor
    assembles only its local Neumann problem and the parallel matrix vector
    product is handled "implicitly".

    Currently this allows for only one subdomain per processor.
*/

#include <../src/mat/impls/is/matis.h>      /*I "petscmat.h" I*/
#include <petsc/private/sfimpl.h>
#include <petsc/private/vecimpl.h>
#include <petsc/private/hashseti.h>

#define MATIS_MAX_ENTRIES_INSERTION 2048
static PetscErrorCode MatSetValuesLocal_IS(Mat,PetscInt,const PetscInt*,PetscInt,const PetscInt*,const PetscScalar*,InsertMode);
static PetscErrorCode MatSetValuesBlockedLocal_IS(Mat,PetscInt,const PetscInt*,PetscInt,const PetscInt*,const PetscScalar*,InsertMode);
static PetscErrorCode MatISSetUpScatters_Private(Mat);

static PetscErrorCode MatISContainerDestroyPtAP_Private(void *ptr)
{
  MatISPtAP      ptap = (MatISPtAP)ptr;

  PetscFunctionBegin;
  PetscCall(MatDestroySubMatrices(ptap->ris1 ? 2 : 1,&ptap->lP));
  PetscCall(ISDestroy(&ptap->cis0));
  PetscCall(ISDestroy(&ptap->cis1));
  PetscCall(ISDestroy(&ptap->ris0));
  PetscCall(ISDestroy(&ptap->ris1));
  PetscCall(PetscFree(ptap));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatPtAPNumeric_IS_XAIJ(Mat A, Mat P, Mat C)
{
  MatISPtAP      ptap;
  Mat_IS         *matis = (Mat_IS*)A->data;
  Mat            lA,lC;
  MatReuse       reuse;
  IS             ris[2],cis[2];
  PetscContainer c;
  PetscInt       n;

  PetscFunctionBegin;
  PetscCall(PetscObjectQuery((PetscObject)C,"_MatIS_PtAP",(PetscObject*)&c));
  PetscCheck(c,PetscObjectComm((PetscObject)C),PETSC_ERR_PLIB,"Missing PtAP information");
  PetscCall(PetscContainerGetPointer(c,(void**)&ptap));
  ris[0] = ptap->ris0;
  ris[1] = ptap->ris1;
  cis[0] = ptap->cis0;
  cis[1] = ptap->cis1;
  n      = ptap->ris1 ? 2 : 1;
  reuse  = ptap->lP ? MAT_REUSE_MATRIX : MAT_INITIAL_MATRIX;
  PetscCall(MatCreateSubMatrices(P,n,ris,cis,reuse,&ptap->lP));

  PetscCall(MatISGetLocalMat(A,&lA));
  PetscCall(MatISGetLocalMat(C,&lC));
  if (ptap->ris1) { /* unsymmetric A mapping */
    Mat lPt;

    PetscCall(MatTranspose(ptap->lP[1],MAT_INITIAL_MATRIX,&lPt));
    PetscCall(MatMatMatMult(lPt,lA,ptap->lP[0],reuse,ptap->fill,&lC));
    if (matis->storel2l) {
      PetscCall(PetscObjectCompose((PetscObject)(A),"_MatIS_PtAP_l2l",(PetscObject)lPt));
    }
    PetscCall(MatDestroy(&lPt));
  } else {
    PetscCall(MatPtAP(lA,ptap->lP[0],reuse,ptap->fill,&lC));
    if (matis->storel2l) {
     PetscCall(PetscObjectCompose((PetscObject)C,"_MatIS_PtAP_l2l",(PetscObject)ptap->lP[0]));
    }
  }
  if (reuse == MAT_INITIAL_MATRIX) {
    PetscCall(MatISSetLocalMat(C,lC));
    PetscCall(MatDestroy(&lC));
  }
  PetscCall(MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatGetNonzeroColumnsLocal_Private(Mat PT,IS *cis)
{
  Mat            Po,Pd;
  IS             zd,zo;
  const PetscInt *garray;
  PetscInt       *aux,i,bs;
  PetscInt       dc,stc,oc,ctd,cto;
  PetscBool      ismpiaij,ismpibaij,isseqaij,isseqbaij;
  MPI_Comm       comm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(PT,MAT_CLASSID,1);
  PetscValidPointer(cis,2);
  PetscCall(PetscObjectGetComm((PetscObject)PT,&comm));
  bs   = 1;
  PetscCall(PetscObjectBaseTypeCompare((PetscObject)PT,MATMPIAIJ,&ismpiaij));
  PetscCall(PetscObjectBaseTypeCompare((PetscObject)PT,MATMPIBAIJ,&ismpibaij));
  PetscCall(PetscObjectBaseTypeCompare((PetscObject)PT,MATSEQAIJ,&isseqaij));
  PetscCall(PetscObjectTypeCompare((PetscObject)PT,MATSEQBAIJ,&isseqbaij));
  if (isseqaij || isseqbaij) {
    Pd = PT;
    Po = NULL;
    garray = NULL;
  } else if (ismpiaij) {
    PetscCall(MatMPIAIJGetSeqAIJ(PT,&Pd,&Po,&garray));
  } else if (ismpibaij) {
    PetscCall(MatMPIBAIJGetSeqBAIJ(PT,&Pd,&Po,&garray));
    PetscCall(MatGetBlockSize(PT,&bs));
  } else SETERRQ(comm,PETSC_ERR_SUP,"Not for matrix type %s",((PetscObject)(PT))->type_name);

  /* identify any null columns in Pd or Po */
  /* We use a tolerance comparison since it may happen that, with geometric multigrid,
     some of the columns are not really zero, but very close to */
  zo = zd = NULL;
  if (Po) {
    PetscCall(MatFindNonzeroRowsOrCols_Basic(Po,PETSC_TRUE,PETSC_SMALL,&zo));
  }
  PetscCall(MatFindNonzeroRowsOrCols_Basic(Pd,PETSC_TRUE,PETSC_SMALL,&zd));

  PetscCall(MatGetLocalSize(PT,NULL,&dc));
  PetscCall(MatGetOwnershipRangeColumn(PT,&stc,NULL));
  if (Po) PetscCall(MatGetLocalSize(Po,NULL,&oc));
  else oc = 0;
  PetscCall(PetscMalloc1((dc+oc)/bs,&aux));
  if (zd) {
    const PetscInt *idxs;
    PetscInt       nz;

    /* this will throw an error if bs is not valid */
    PetscCall(ISSetBlockSize(zd,bs));
    PetscCall(ISGetLocalSize(zd,&nz));
    PetscCall(ISGetIndices(zd,&idxs));
    ctd  = nz/bs;
    for (i=0; i<ctd; i++) aux[i] = (idxs[bs*i]+stc)/bs;
    PetscCall(ISRestoreIndices(zd,&idxs));
  } else {
    ctd = dc/bs;
    for (i=0; i<ctd; i++) aux[i] = i+stc/bs;
  }
  if (zo) {
    const PetscInt *idxs;
    PetscInt       nz;

    /* this will throw an error if bs is not valid */
    PetscCall(ISSetBlockSize(zo,bs));
    PetscCall(ISGetLocalSize(zo,&nz));
    PetscCall(ISGetIndices(zo,&idxs));
    cto  = nz/bs;
    for (i=0; i<cto; i++) aux[i+ctd] = garray[idxs[bs*i]/bs];
    PetscCall(ISRestoreIndices(zo,&idxs));
  } else {
    cto = oc/bs;
    for (i=0; i<cto; i++) aux[i+ctd] = garray[i];
  }
  PetscCall(ISCreateBlock(comm,bs,ctd+cto,aux,PETSC_OWN_POINTER,cis));
  PetscCall(ISDestroy(&zd));
  PetscCall(ISDestroy(&zo));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatPtAPSymbolic_IS_XAIJ(Mat A,Mat P,PetscReal fill,Mat C)
{
  Mat                    PT,lA;
  MatISPtAP              ptap;
  ISLocalToGlobalMapping Crl2g,Ccl2g,rl2g,cl2g;
  PetscContainer         c;
  MatType                lmtype;
  const PetscInt         *garray;
  PetscInt               ibs,N,dc;
  MPI_Comm               comm;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)A,&comm));
  PetscCall(MatSetType(C,MATIS));
  PetscCall(MatISGetLocalMat(A,&lA));
  PetscCall(MatGetType(lA,&lmtype));
  PetscCall(MatISSetLocalMatType(C,lmtype));
  PetscCall(MatGetSize(P,NULL,&N));
  PetscCall(MatGetLocalSize(P,NULL,&dc));
  PetscCall(MatSetSizes(C,dc,dc,N,N));
/* Not sure about this
  PetscCall(MatGetBlockSizes(P,NULL,&ibs));
  PetscCall(MatSetBlockSize(*C,ibs));
*/

  PetscCall(PetscNew(&ptap));
  PetscCall(PetscContainerCreate(PETSC_COMM_SELF,&c));
  PetscCall(PetscContainerSetPointer(c,ptap));
  PetscCall(PetscContainerSetUserDestroy(c,MatISContainerDestroyPtAP_Private));
  PetscCall(PetscObjectCompose((PetscObject)C,"_MatIS_PtAP",(PetscObject)c));
  PetscCall(PetscContainerDestroy(&c));
  ptap->fill = fill;

  PetscCall(MatISGetLocalToGlobalMapping(A,&rl2g,&cl2g));

  PetscCall(ISLocalToGlobalMappingGetBlockSize(cl2g,&ibs));
  PetscCall(ISLocalToGlobalMappingGetSize(cl2g,&N));
  PetscCall(ISLocalToGlobalMappingGetBlockIndices(cl2g,&garray));
  PetscCall(ISCreateBlock(comm,ibs,N/ibs,garray,PETSC_COPY_VALUES,&ptap->ris0));
  PetscCall(ISLocalToGlobalMappingRestoreBlockIndices(cl2g,&garray));

  PetscCall(MatCreateSubMatrix(P,ptap->ris0,NULL,MAT_INITIAL_MATRIX,&PT));
  PetscCall(MatGetNonzeroColumnsLocal_Private(PT,&ptap->cis0));
  PetscCall(ISLocalToGlobalMappingCreateIS(ptap->cis0,&Ccl2g));
  PetscCall(MatDestroy(&PT));

  Crl2g = NULL;
  if (rl2g != cl2g) { /* unsymmetric A mapping */
    PetscBool same,lsame = PETSC_FALSE;
    PetscInt  N1,ibs1;

    PetscCall(ISLocalToGlobalMappingGetSize(rl2g,&N1));
    PetscCall(ISLocalToGlobalMappingGetBlockSize(rl2g,&ibs1));
    PetscCall(ISLocalToGlobalMappingGetBlockIndices(rl2g,&garray));
    PetscCall(ISCreateBlock(comm,ibs,N/ibs,garray,PETSC_COPY_VALUES,&ptap->ris1));
    PetscCall(ISLocalToGlobalMappingRestoreBlockIndices(rl2g,&garray));
    if (ibs1 == ibs && N1 == N) { /* check if the l2gmaps are the same */
      const PetscInt *i1,*i2;

      PetscCall(ISBlockGetIndices(ptap->ris0,&i1));
      PetscCall(ISBlockGetIndices(ptap->ris1,&i2));
      PetscCall(PetscArraycmp(i1,i2,N,&lsame));
    }
    PetscCall(MPIU_Allreduce(&lsame,&same,1,MPIU_BOOL,MPI_LAND,comm));
    if (same) {
      PetscCall(ISDestroy(&ptap->ris1));
    } else {
      PetscCall(MatCreateSubMatrix(P,ptap->ris1,NULL,MAT_INITIAL_MATRIX,&PT));
      PetscCall(MatGetNonzeroColumnsLocal_Private(PT,&ptap->cis1));
      PetscCall(ISLocalToGlobalMappingCreateIS(ptap->cis1,&Crl2g));
      PetscCall(MatDestroy(&PT));
    }
  }
/* Not sure about this
  if (!Crl2g) {
    PetscCall(MatGetBlockSize(C,&ibs));
    PetscCall(ISLocalToGlobalMappingSetBlockSize(Ccl2g,ibs));
  }
*/
  PetscCall(MatSetLocalToGlobalMapping(C,Crl2g ? Crl2g : Ccl2g,Ccl2g));
  PetscCall(ISLocalToGlobalMappingDestroy(&Crl2g));
  PetscCall(ISLocalToGlobalMappingDestroy(&Ccl2g));

  C->ops->ptapnumeric = MatPtAPNumeric_IS_XAIJ;
  PetscFunctionReturn(0);
}

/* ----------------------------------------- */
static PetscErrorCode MatProductSymbolic_PtAP_IS_XAIJ(Mat C)
{
  Mat_Product    *product = C->product;
  Mat            A=product->A,P=product->B;
  PetscReal      fill=product->fill;

  PetscFunctionBegin;
  PetscCall(MatPtAPSymbolic_IS_XAIJ(A,P,fill,C));
  C->ops->productnumeric = MatProductNumeric_PtAP;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatProductSetFromOptions_IS_XAIJ_PtAP(Mat C)
{
  PetscFunctionBegin;
  C->ops->productsymbolic = MatProductSymbolic_PtAP_IS_XAIJ;
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatProductSetFromOptions_IS_XAIJ(Mat C)
{
  Mat_Product    *product = C->product;

  PetscFunctionBegin;
  if (product->type == MATPRODUCT_PtAP) {
    PetscCall(MatProductSetFromOptions_IS_XAIJ_PtAP(C));
  }
  PetscFunctionReturn(0);
}

/* ----------------------------------------- */
static PetscErrorCode MatISContainerDestroyFields_Private(void *ptr)
{
  MatISLocalFields lf = (MatISLocalFields)ptr;
  PetscInt         i;

  PetscFunctionBegin;
  for (i=0;i<lf->nr;i++) {
    PetscCall(ISDestroy(&lf->rf[i]));
  }
  for (i=0;i<lf->nc;i++) {
    PetscCall(ISDestroy(&lf->cf[i]));
  }
  PetscCall(PetscFree2(lf->rf,lf->cf));
  PetscCall(PetscFree(lf));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatConvert_SeqXAIJ_IS(Mat A,MatType type,MatReuse reuse,Mat *newmat)
{
  Mat            B,lB;

  PetscFunctionBegin;
  if (reuse != MAT_REUSE_MATRIX) {
    ISLocalToGlobalMapping rl2g,cl2g;
    PetscInt               bs;
    IS                     is;

    PetscCall(MatGetBlockSize(A,&bs));
    PetscCall(ISCreateStride(PetscObjectComm((PetscObject)A),A->rmap->n/bs,0,1,&is));
    if (bs > 1) {
      IS       is2;
      PetscInt i,*aux;

      PetscCall(ISGetLocalSize(is,&i));
      PetscCall(ISGetIndices(is,(const PetscInt**)&aux));
      PetscCall(ISCreateBlock(PetscObjectComm((PetscObject)A),bs,i,aux,PETSC_COPY_VALUES,&is2));
      PetscCall(ISRestoreIndices(is,(const PetscInt**)&aux));
      PetscCall(ISDestroy(&is));
      is   = is2;
    }
    PetscCall(ISSetIdentity(is));
    PetscCall(ISLocalToGlobalMappingCreateIS(is,&rl2g));
    PetscCall(ISDestroy(&is));
    PetscCall(ISCreateStride(PetscObjectComm((PetscObject)A),A->cmap->n/bs,0,1,&is));
    if (bs > 1) {
      IS       is2;
      PetscInt i,*aux;

      PetscCall(ISGetLocalSize(is,&i));
      PetscCall(ISGetIndices(is,(const PetscInt**)&aux));
      PetscCall(ISCreateBlock(PetscObjectComm((PetscObject)A),bs,i,aux,PETSC_COPY_VALUES,&is2));
      PetscCall(ISRestoreIndices(is,(const PetscInt**)&aux));
      PetscCall(ISDestroy(&is));
      is   = is2;
    }
    PetscCall(ISSetIdentity(is));
    PetscCall(ISLocalToGlobalMappingCreateIS(is,&cl2g));
    PetscCall(ISDestroy(&is));
    PetscCall(MatCreateIS(PetscObjectComm((PetscObject)A),bs,A->rmap->n,A->cmap->n,A->rmap->N,A->cmap->N,rl2g,cl2g,&B));
    PetscCall(ISLocalToGlobalMappingDestroy(&rl2g));
    PetscCall(ISLocalToGlobalMappingDestroy(&cl2g));
    PetscCall(MatDuplicate(A,MAT_COPY_VALUES,&lB));
    if (reuse == MAT_INITIAL_MATRIX) *newmat = B;
  } else {
    B    = *newmat;
    PetscCall(PetscObjectReference((PetscObject)A));
    lB   = A;
  }
  PetscCall(MatISSetLocalMat(B,lB));
  PetscCall(MatDestroy(&lB));
  PetscCall(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));
  if (reuse == MAT_INPLACE_MATRIX) {
    PetscCall(MatHeaderReplace(A,&B));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatISScaleDisassembling_Private(Mat A)
{
  Mat_IS         *matis = (Mat_IS*)(A->data);
  PetscScalar    *aa;
  const PetscInt *ii,*jj;
  PetscInt       i,n,m;
  PetscInt       *ecount,**eneighs;
  PetscBool      flg;

  PetscFunctionBegin;
  PetscCall(MatGetRowIJ(matis->A,0,PETSC_FALSE,PETSC_FALSE,&m,&ii,&jj,&flg));
  PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_SUP,"Cannot get IJ structure");
  PetscCall(ISLocalToGlobalMappingGetNodeInfo(matis->rmapping,&n,&ecount,&eneighs));
  PetscCheck(m == n,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Unexpected %" PetscInt_FMT " != %" PetscInt_FMT,m,n);
  PetscCall(MatSeqAIJGetArray(matis->A,&aa));
  for (i=0;i<n;i++) {
    if (ecount[i] > 1) {
      PetscInt j;

      for (j=ii[i];j<ii[i+1];j++) {
        PetscInt    i2 = jj[j],p,p2;
        PetscReal   scal = 0.0;

        for (p=0;p<ecount[i];p++) {
          for (p2=0;p2<ecount[i2];p2++) {
            if (eneighs[i][p] == eneighs[i2][p2]) { scal += 1.0; break; }
          }
        }
        if (scal) aa[j] /= scal;
      }
    }
  }
  PetscCall(ISLocalToGlobalMappingRestoreNodeInfo(matis->rmapping,&n,&ecount,&eneighs));
  PetscCall(MatSeqAIJRestoreArray(matis->A,&aa));
  PetscCall(MatRestoreRowIJ(matis->A,0,PETSC_FALSE,PETSC_FALSE,&m,&ii,&jj,&flg));
  PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_SUP,"Cannot restore IJ structure");
  PetscFunctionReturn(0);
}

typedef enum {MAT_IS_DISASSEMBLE_L2G_NATURAL,MAT_IS_DISASSEMBLE_L2G_MAT, MAT_IS_DISASSEMBLE_L2G_ND} MatISDisassemblel2gType;

static PetscErrorCode MatMPIXAIJComputeLocalToGlobalMapping_Private(Mat A, ISLocalToGlobalMapping *l2g)
{
  Mat                     Ad,Ao;
  IS                      is,ndmap,ndsub;
  MPI_Comm                comm;
  const PetscInt          *garray,*ndmapi;
  PetscInt                bs,i,cnt,nl,*ncount,*ndmapc;
  PetscBool               ismpiaij,ismpibaij;
  const char *const       MatISDisassemblel2gTypes[] = {"NATURAL","MAT","ND","MatISDisassemblel2gType","MAT_IS_DISASSEMBLE_L2G_",NULL};
  MatISDisassemblel2gType mode = MAT_IS_DISASSEMBLE_L2G_NATURAL;
  MatPartitioning         part;
  PetscSF                 sf;
  PetscErrorCode          ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)A),((PetscObject)A)->prefix,"MatIS l2g disassembling options","Mat");PetscCall(ierr);
  PetscCall(PetscOptionsEnum("-mat_is_disassemble_l2g_type","Type of local-to-global mapping to be used for disassembling","MatISDisassemblel2gType",MatISDisassemblel2gTypes,(PetscEnum)mode,(PetscEnum*)&mode,NULL));
  ierr = PetscOptionsEnd();PetscCall(ierr);
  if (mode == MAT_IS_DISASSEMBLE_L2G_MAT) {
    PetscCall(MatGetLocalToGlobalMapping(A,l2g,NULL));
    PetscFunctionReturn(0);
  }
  PetscCall(PetscObjectGetComm((PetscObject)A,&comm));
  PetscCall(PetscObjectBaseTypeCompare((PetscObject)A,MATMPIAIJ ,&ismpiaij));
  PetscCall(PetscObjectBaseTypeCompare((PetscObject)A,MATMPIBAIJ,&ismpibaij));
  PetscCall(MatGetBlockSize(A,&bs));
  switch (mode) {
  case MAT_IS_DISASSEMBLE_L2G_ND:
    PetscCall(MatPartitioningCreate(comm,&part));
    PetscCall(MatPartitioningSetAdjacency(part,A));
    PetscCall(PetscObjectSetOptionsPrefix((PetscObject)part,((PetscObject)A)->prefix));
    PetscCall(MatPartitioningSetFromOptions(part));
    PetscCall(MatPartitioningApplyND(part,&ndmap));
    PetscCall(MatPartitioningDestroy(&part));
    PetscCall(ISBuildTwoSided(ndmap,NULL,&ndsub));
    PetscCall(MatMPIAIJSetUseScalableIncreaseOverlap(A,PETSC_TRUE));
    PetscCall(MatIncreaseOverlap(A,1,&ndsub,1));
    PetscCall(ISLocalToGlobalMappingCreateIS(ndsub,l2g));

    /* it may happen that a separator node is not properly shared */
    PetscCall(ISLocalToGlobalMappingGetNodeInfo(*l2g,&nl,&ncount,NULL));
    PetscCall(PetscSFCreate(comm,&sf));
    PetscCall(ISLocalToGlobalMappingGetIndices(*l2g,&garray));
    PetscCall(PetscSFSetGraphLayout(sf,A->rmap,nl,NULL,PETSC_OWN_POINTER,garray));
    PetscCall(ISLocalToGlobalMappingRestoreIndices(*l2g,&garray));
    PetscCall(PetscCalloc1(A->rmap->n,&ndmapc));
    PetscCall(PetscSFReduceBegin(sf,MPIU_INT,ncount,ndmapc,MPI_REPLACE));
    PetscCall(PetscSFReduceEnd(sf,MPIU_INT,ncount,ndmapc,MPI_REPLACE));
    PetscCall(ISLocalToGlobalMappingRestoreNodeInfo(*l2g,NULL,&ncount,NULL));
    PetscCall(ISGetIndices(ndmap,&ndmapi));
    for (i = 0, cnt = 0; i < A->rmap->n; i++)
      if (ndmapi[i] < 0 && ndmapc[i] < 2)
        cnt++;

    PetscCall(MPIU_Allreduce(&cnt,&i,1,MPIU_INT,MPI_MAX,comm));
    if (i) { /* we detected isolated separator nodes */
      Mat                    A2,A3;
      IS                     *workis,is2;
      PetscScalar            *vals;
      PetscInt               gcnt = i,*dnz,*onz,j,*lndmapi;
      ISLocalToGlobalMapping ll2g;
      PetscBool              flg;
      const PetscInt         *ii,*jj;

      /* communicate global id of separators */
      ierr = MatPreallocateInitialize(comm,A->rmap->n,A->cmap->n,dnz,onz);PetscCall(ierr);
      for (i = 0, cnt = 0; i < A->rmap->n; i++)
        dnz[i] = ndmapi[i] < 0 ? i + A->rmap->rstart : -1;

      PetscCall(PetscMalloc1(nl,&lndmapi));
      PetscCall(PetscSFBcastBegin(sf,MPIU_INT,dnz,lndmapi,MPI_REPLACE));

      /* compute adjacency of isolated separators node */
      PetscCall(PetscMalloc1(gcnt,&workis));
      for (i = 0, cnt = 0; i < A->rmap->n; i++) {
        if (ndmapi[i] < 0 && ndmapc[i] < 2) {
          PetscCall(ISCreateStride(comm,1,i+A->rmap->rstart,1,&workis[cnt++]));
        }
      }
      for (i = cnt; i < gcnt; i++) {
        PetscCall(ISCreateStride(comm,0,0,1,&workis[i]));
      }
      for (i = 0; i < gcnt; i++) {
        PetscCall(PetscObjectSetName((PetscObject)workis[i],"ISOLATED"));
        PetscCall(ISViewFromOptions(workis[i],NULL,"-view_isolated_separators"));
      }

      /* no communications since all the ISes correspond to locally owned rows */
      PetscCall(MatIncreaseOverlap(A,gcnt,workis,1));

      /* end communicate global id of separators */
      PetscCall(PetscSFBcastEnd(sf,MPIU_INT,dnz,lndmapi,MPI_REPLACE));

      /* communicate new layers : create a matrix and transpose it */
      PetscCall(PetscArrayzero(dnz,A->rmap->n));
      PetscCall(PetscArrayzero(onz,A->rmap->n));
      for (i = 0, j = 0; i < A->rmap->n; i++) {
        if (ndmapi[i] < 0 && ndmapc[i] < 2) {
          const PetscInt* idxs;
          PetscInt        s;

          PetscCall(ISGetLocalSize(workis[j],&s));
          PetscCall(ISGetIndices(workis[j],&idxs));
          PetscCall(MatPreallocateSet(i+A->rmap->rstart,s,idxs,dnz,onz));
          j++;
        }
      }
      PetscCheck(j == cnt,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Unexpected local count %" PetscInt_FMT " != %" PetscInt_FMT,j,cnt);

      for (i = 0; i < gcnt; i++) {
        PetscCall(PetscObjectSetName((PetscObject)workis[i],"EXTENDED"));
        PetscCall(ISViewFromOptions(workis[i],NULL,"-view_isolated_separators"));
      }

      for (i = 0, j = 0; i < A->rmap->n; i++) j = PetscMax(j,dnz[i]+onz[i]);
      PetscCall(PetscMalloc1(j,&vals));
      for (i = 0; i < j; i++) vals[i] = 1.0;

      PetscCall(MatCreate(comm,&A2));
      PetscCall(MatSetType(A2,MATMPIAIJ));
      PetscCall(MatSetSizes(A2,A->rmap->n,A->cmap->n,A->rmap->N,A->cmap->N));
      PetscCall(MatMPIAIJSetPreallocation(A2,0,dnz,0,onz));
      PetscCall(MatSetOption(A2,MAT_NO_OFF_PROC_ENTRIES,PETSC_TRUE));
      for (i = 0, j = 0; i < A2->rmap->n; i++) {
        PetscInt        row = i+A2->rmap->rstart,s = dnz[i] + onz[i];
        const PetscInt* idxs;

        if (s) {
          PetscCall(ISGetIndices(workis[j],&idxs));
          PetscCall(MatSetValues(A2,1,&row,s,idxs,vals,INSERT_VALUES));
          PetscCall(ISRestoreIndices(workis[j],&idxs));
          j++;
        }
      }
      PetscCheck(j == cnt,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Unexpected local count %" PetscInt_FMT " != %" PetscInt_FMT,j,cnt);
      PetscCall(PetscFree(vals));
      PetscCall(MatAssemblyBegin(A2,MAT_FINAL_ASSEMBLY));
      PetscCall(MatAssemblyEnd(A2,MAT_FINAL_ASSEMBLY));
      PetscCall(MatTranspose(A2,MAT_INPLACE_MATRIX,&A2));

      /* extract submatrix corresponding to the coupling "owned separators" x "isolated separators" */
      for (i = 0, j = 0; i < nl; i++)
        if (lndmapi[i] >= 0) lndmapi[j++] = lndmapi[i];
      PetscCall(ISCreateGeneral(comm,j,lndmapi,PETSC_USE_POINTER,&is));
      PetscCall(MatMPIAIJGetLocalMatCondensed(A2,MAT_INITIAL_MATRIX,&is,NULL,&A3));
      PetscCall(ISDestroy(&is));
      PetscCall(MatDestroy(&A2));

      /* extend local to global map to include connected isolated separators */
      PetscCall(PetscObjectQuery((PetscObject)A3,"_petsc_GetLocalMatCondensed_iscol",(PetscObject*)&is));
      PetscCheck(is,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Missing column map");
      PetscCall(ISLocalToGlobalMappingCreateIS(is,&ll2g));
      PetscCall(MatGetRowIJ(A3,0,PETSC_FALSE,PETSC_FALSE,&i,&ii,&jj,&flg));
      PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_SUP,"Cannot get IJ structure");
      PetscCall(ISCreateGeneral(PETSC_COMM_SELF,ii[i],jj,PETSC_COPY_VALUES,&is));
      PetscCall(MatRestoreRowIJ(A3,0,PETSC_FALSE,PETSC_FALSE,&i,&ii,&jj,&flg));
      PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_SUP,"Cannot get IJ structure");
      PetscCall(ISLocalToGlobalMappingApplyIS(ll2g,is,&is2));
      PetscCall(ISDestroy(&is));
      PetscCall(ISLocalToGlobalMappingDestroy(&ll2g));

      /* add new nodes to the local-to-global map */
      PetscCall(ISLocalToGlobalMappingDestroy(l2g));
      PetscCall(ISExpand(ndsub,is2,&is));
      PetscCall(ISDestroy(&is2));
      PetscCall(ISLocalToGlobalMappingCreateIS(is,l2g));
      PetscCall(ISDestroy(&is));

      PetscCall(MatDestroy(&A3));
      PetscCall(PetscFree(lndmapi));
      ierr = MatPreallocateFinalize(dnz,onz);PetscCall(ierr);
      for (i = 0; i < gcnt; i++) {
        PetscCall(ISDestroy(&workis[i]));
      }
      PetscCall(PetscFree(workis));
    }
    PetscCall(ISRestoreIndices(ndmap,&ndmapi));
    PetscCall(PetscSFDestroy(&sf));
    PetscCall(PetscFree(ndmapc));
    PetscCall(ISDestroy(&ndmap));
    PetscCall(ISDestroy(&ndsub));
    PetscCall(ISLocalToGlobalMappingSetBlockSize(*l2g,bs));
    PetscCall(ISLocalToGlobalMappingViewFromOptions(*l2g,NULL,"-matis_nd_l2g_view"));
    break;
  case MAT_IS_DISASSEMBLE_L2G_NATURAL:
    if (ismpiaij) {
      PetscCall(MatMPIAIJGetSeqAIJ(A,&Ad,&Ao,&garray));
    } else if (ismpibaij) {
      PetscCall(MatMPIBAIJGetSeqBAIJ(A,&Ad,&Ao,&garray));
    } else SETERRQ(comm,PETSC_ERR_SUP,"Type %s",((PetscObject)A)->type_name);
    PetscCheck(garray,comm,PETSC_ERR_ARG_WRONGSTATE,"garray not present");
    if (A->rmap->n) {
      PetscInt dc,oc,stc,*aux;

      PetscCall(MatGetLocalSize(A,NULL,&dc));
      PetscCall(MatGetLocalSize(Ao,NULL,&oc));
      PetscCall(MatGetOwnershipRangeColumn(A,&stc,NULL));
      PetscCall(PetscMalloc1((dc+oc)/bs,&aux));
      for (i=0; i<dc/bs; i++) aux[i]       = i+stc/bs;
      for (i=0; i<oc/bs; i++) aux[i+dc/bs] = garray[i];
      PetscCall(ISCreateBlock(comm,bs,(dc+oc)/bs,aux,PETSC_OWN_POINTER,&is));
    } else {
      PetscCall(ISCreateBlock(comm,1,0,NULL,PETSC_OWN_POINTER,&is));
    }
    PetscCall(ISLocalToGlobalMappingCreateIS(is,l2g));
    PetscCall(ISDestroy(&is));
    break;
  default:
    SETERRQ(comm,PETSC_ERR_ARG_WRONG,"Unsupported l2g disassembling type %d",mode);
  }
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatConvert_XAIJ_IS(Mat A,MatType type,MatReuse reuse,Mat *newmat)
{
  Mat                    lA,Ad,Ao,B = NULL;
  ISLocalToGlobalMapping rl2g,cl2g;
  IS                     is;
  MPI_Comm               comm;
  void                   *ptrs[2];
  const char             *names[2] = {"_convert_csr_aux","_convert_csr_data"};
  const PetscInt         *garray;
  PetscScalar            *dd,*od,*aa,*data;
  const PetscInt         *di,*dj,*oi,*oj;
  const PetscInt         *odi,*odj,*ooi,*ooj;
  PetscInt               *aux,*ii,*jj;
  PetscInt               bs,lc,dr,dc,oc,str,stc,nnz,i,jd,jo,cum;
  PetscBool              flg,ismpiaij,ismpibaij,was_inplace = PETSC_FALSE;
  PetscMPIInt            size;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)A,&comm));
  PetscCallMPI(MPI_Comm_size(comm,&size));
  if (size == 1) {
    PetscCall(MatConvert_SeqXAIJ_IS(A,type,reuse,newmat));
    PetscFunctionReturn(0);
  }
  if (reuse != MAT_REUSE_MATRIX && A->cmap->N == A->rmap->N) {
    PetscCall(MatMPIXAIJComputeLocalToGlobalMapping_Private(A,&rl2g));
    PetscCall(MatCreate(comm,&B));
    PetscCall(MatSetType(B,MATIS));
    PetscCall(MatSetSizes(B,A->rmap->n,A->cmap->n,A->rmap->N,A->cmap->N));
    PetscCall(MatSetLocalToGlobalMapping(B,rl2g,rl2g));
    PetscCall(MatGetBlockSize(A,&bs));
    PetscCall(MatSetBlockSize(B,bs));
    PetscCall(ISLocalToGlobalMappingDestroy(&rl2g));
    if (reuse == MAT_INPLACE_MATRIX) was_inplace = PETSC_TRUE;
    reuse = MAT_REUSE_MATRIX;
  }
  if (reuse == MAT_REUSE_MATRIX) {
    Mat            *newlA, lA;
    IS             rows, cols;
    const PetscInt *ridx, *cidx;
    PetscInt       rbs, cbs, nr, nc;

    if (!B) B = *newmat;
    PetscCall(MatISGetLocalToGlobalMapping(B,&rl2g,&cl2g));
    PetscCall(ISLocalToGlobalMappingGetBlockIndices(rl2g,&ridx));
    PetscCall(ISLocalToGlobalMappingGetBlockIndices(cl2g,&cidx));
    PetscCall(ISLocalToGlobalMappingGetSize(rl2g,&nr));
    PetscCall(ISLocalToGlobalMappingGetSize(cl2g,&nc));
    PetscCall(ISLocalToGlobalMappingGetBlockSize(rl2g,&rbs));
    PetscCall(ISLocalToGlobalMappingGetBlockSize(cl2g,&cbs));
    PetscCall(ISCreateBlock(comm,rbs,nr/rbs,ridx,PETSC_USE_POINTER,&rows));
    if (rl2g != cl2g) {
      PetscCall(ISCreateBlock(comm,cbs,nc/cbs,cidx,PETSC_USE_POINTER,&cols));
    } else {
      PetscCall(PetscObjectReference((PetscObject)rows));
      cols = rows;
    }
    PetscCall(MatISGetLocalMat(B,&lA));
    PetscCall(MatCreateSubMatrices(A,1,&rows,&cols,MAT_INITIAL_MATRIX,&newlA));
    PetscCall(MatConvert(newlA[0],MATSEQAIJ,MAT_INPLACE_MATRIX,&newlA[0]));
    PetscCall(ISLocalToGlobalMappingRestoreBlockIndices(rl2g,&ridx));
    PetscCall(ISLocalToGlobalMappingRestoreBlockIndices(cl2g,&cidx));
    PetscCall(ISDestroy(&rows));
    PetscCall(ISDestroy(&cols));
    if (!lA->preallocated) { /* first time */
      PetscCall(MatDuplicate(newlA[0],MAT_COPY_VALUES,&lA));
      PetscCall(MatISSetLocalMat(B,lA));
      PetscCall(PetscObjectDereference((PetscObject)lA));
    }
    PetscCall(MatCopy(newlA[0],lA,SAME_NONZERO_PATTERN));
    PetscCall(MatDestroySubMatrices(1,&newlA));
    PetscCall(MatISScaleDisassembling_Private(B));
    PetscCall(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));
    if (was_inplace) PetscCall(MatHeaderReplace(A,&B));
    else *newmat = B;
    PetscFunctionReturn(0);
  }
  /* rectangular case, just compress out the column space */
  PetscCall(PetscObjectBaseTypeCompare((PetscObject)A,MATMPIAIJ ,&ismpiaij));
  PetscCall(PetscObjectBaseTypeCompare((PetscObject)A,MATMPIBAIJ,&ismpibaij));
  if (ismpiaij) {
    bs   = 1;
    PetscCall(MatMPIAIJGetSeqAIJ(A,&Ad,&Ao,&garray));
  } else if (ismpibaij) {
    PetscCall(MatGetBlockSize(A,&bs));
    PetscCall(MatMPIBAIJGetSeqBAIJ(A,&Ad,&Ao,&garray));
    PetscCall(MatConvert(Ad,MATSEQAIJ,MAT_INITIAL_MATRIX,&Ad));
    PetscCall(MatConvert(Ao,MATSEQAIJ,MAT_INITIAL_MATRIX,&Ao));
  } else SETERRQ(comm,PETSC_ERR_SUP,"Type %s",((PetscObject)A)->type_name);
  PetscCall(MatSeqAIJGetArray(Ad,&dd));
  PetscCall(MatSeqAIJGetArray(Ao,&od));
  PetscCheck(garray,comm,PETSC_ERR_ARG_WRONGSTATE,"garray not present");

  /* access relevant information from MPIAIJ */
  PetscCall(MatGetOwnershipRange(A,&str,NULL));
  PetscCall(MatGetOwnershipRangeColumn(A,&stc,NULL));
  PetscCall(MatGetLocalSize(A,&dr,&dc));
  PetscCall(MatGetLocalSize(Ao,NULL,&oc));
  PetscCall(MatGetRowIJ(Ad,0,PETSC_FALSE,PETSC_FALSE,&i,&di,&dj,&flg));
  PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_SUP,"Cannot get IJ structure");
  PetscCall(MatGetRowIJ(Ao,0,PETSC_FALSE,PETSC_FALSE,&i,&oi,&oj,&flg));
  PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_SUP,"Cannot get IJ structure");
  nnz = di[dr] + oi[dr];
  /* store original pointers to be restored later */
  odi = di; odj = dj; ooi = oi; ooj = oj;

  /* generate l2g maps for rows and cols */
  PetscCall(ISCreateStride(comm,dr/bs,str/bs,1,&is));
  if (bs > 1) {
    IS is2;

    PetscCall(ISGetLocalSize(is,&i));
    PetscCall(ISGetIndices(is,(const PetscInt**)&aux));
    PetscCall(ISCreateBlock(comm,bs,i,aux,PETSC_COPY_VALUES,&is2));
    PetscCall(ISRestoreIndices(is,(const PetscInt**)&aux));
    PetscCall(ISDestroy(&is));
    is   = is2;
  }
  PetscCall(ISLocalToGlobalMappingCreateIS(is,&rl2g));
  PetscCall(ISDestroy(&is));
  if (dr) {
    PetscCall(PetscMalloc1((dc+oc)/bs,&aux));
    for (i=0; i<dc/bs; i++) aux[i]       = i+stc/bs;
    for (i=0; i<oc/bs; i++) aux[i+dc/bs] = garray[i];
    PetscCall(ISCreateBlock(comm,bs,(dc+oc)/bs,aux,PETSC_OWN_POINTER,&is));
    lc   = dc+oc;
  } else {
    PetscCall(ISCreateBlock(comm,bs,0,NULL,PETSC_OWN_POINTER,&is));
    lc   = 0;
  }
  PetscCall(ISLocalToGlobalMappingCreateIS(is,&cl2g));
  PetscCall(ISDestroy(&is));

  /* create MATIS object */
  PetscCall(MatCreate(comm,&B));
  PetscCall(MatSetSizes(B,dr,dc,PETSC_DECIDE,PETSC_DECIDE));
  PetscCall(MatSetType(B,MATIS));
  PetscCall(MatSetBlockSize(B,bs));
  PetscCall(MatSetLocalToGlobalMapping(B,rl2g,cl2g));
  PetscCall(ISLocalToGlobalMappingDestroy(&rl2g));
  PetscCall(ISLocalToGlobalMappingDestroy(&cl2g));

  /* merge local matrices */
  PetscCall(PetscMalloc1(nnz+dr+1,&aux));
  PetscCall(PetscMalloc1(nnz,&data));
  ii   = aux;
  jj   = aux+dr+1;
  aa   = data;
  *ii  = *(di++) + *(oi++);
  for (jd=0,jo=0,cum=0;*ii<nnz;cum++)
  {
     for (;jd<*di;jd++) { *jj++ = *dj++;      *aa++ = *dd++; }
     for (;jo<*oi;jo++) { *jj++ = *oj++ + dc; *aa++ = *od++; }
     *(++ii) = *(di++) + *(oi++);
  }
  for (;cum<dr;cum++) *(++ii) = nnz;

  PetscCall(MatRestoreRowIJ(Ad,0,PETSC_FALSE,PETSC_FALSE,&i,&odi,&odj,&flg));
  PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_SUP,"Cannot restore IJ structure");
  PetscCall(MatRestoreRowIJ(Ao,0,PETSC_FALSE,PETSC_FALSE,&i,&ooi,&ooj,&flg));
  PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_SUP,"Cannot restore IJ structure");
  PetscCall(MatSeqAIJRestoreArray(Ad,&dd));
  PetscCall(MatSeqAIJRestoreArray(Ao,&od));

  ii   = aux;
  jj   = aux+dr+1;
  aa   = data;
  PetscCall(MatCreateSeqAIJWithArrays(PETSC_COMM_SELF,dr,lc,ii,jj,aa,&lA));

  /* create containers to destroy the data */
  ptrs[0] = aux;
  ptrs[1] = data;
  for (i=0; i<2; i++) {
    PetscContainer c;

    PetscCall(PetscContainerCreate(PETSC_COMM_SELF,&c));
    PetscCall(PetscContainerSetPointer(c,ptrs[i]));
    PetscCall(PetscContainerSetUserDestroy(c,PetscContainerUserDestroyDefault));
    PetscCall(PetscObjectCompose((PetscObject)lA,names[i],(PetscObject)c));
    PetscCall(PetscContainerDestroy(&c));
  }
  if (ismpibaij) { /* destroy converted local matrices */
    PetscCall(MatDestroy(&Ad));
    PetscCall(MatDestroy(&Ao));
  }

  /* finalize matrix */
  PetscCall(MatISSetLocalMat(B,lA));
  PetscCall(MatDestroy(&lA));
  PetscCall(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));
  if (reuse == MAT_INPLACE_MATRIX) {
    PetscCall(MatHeaderReplace(A,&B));
  } else *newmat = B;
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatConvert_Nest_IS(Mat A,MatType type,MatReuse reuse,Mat *newmat)
{
  Mat                    **nest,*snest,**rnest,lA,B;
  IS                     *iscol,*isrow,*islrow,*islcol;
  ISLocalToGlobalMapping rl2g,cl2g;
  MPI_Comm               comm;
  PetscInt               *lr,*lc,*l2gidxs;
  PetscInt               i,j,nr,nc,rbs,cbs;
  PetscBool              convert,lreuse,*istrans;

  PetscFunctionBegin;
  PetscCall(MatNestGetSubMats(A,&nr,&nc,&nest));
  lreuse = PETSC_FALSE;
  rnest  = NULL;
  if (reuse == MAT_REUSE_MATRIX) {
    PetscBool ismatis,isnest;

    PetscCall(PetscObjectTypeCompare((PetscObject)*newmat,MATIS,&ismatis));
    PetscCheck(ismatis,PetscObjectComm((PetscObject)*newmat),PETSC_ERR_USER,"Cannot reuse matrix of type %s",((PetscObject)(*newmat))->type_name);
    PetscCall(MatISGetLocalMat(*newmat,&lA));
    PetscCall(PetscObjectTypeCompare((PetscObject)lA,MATNEST,&isnest));
    if (isnest) {
      PetscCall(MatNestGetSubMats(lA,&i,&j,&rnest));
      lreuse = (PetscBool)(i == nr && j == nc);
      if (!lreuse) rnest = NULL;
    }
  }
  PetscCall(PetscObjectGetComm((PetscObject)A,&comm));
  PetscCall(PetscCalloc2(nr,&lr,nc,&lc));
  PetscCall(PetscCalloc6(nr,&isrow,nc,&iscol,nr,&islrow,nc,&islcol,nr*nc,&snest,nr*nc,&istrans));
  PetscCall(MatNestGetISs(A,isrow,iscol));
  for (i=0;i<nr;i++) {
    for (j=0;j<nc;j++) {
      PetscBool ismatis;
      PetscInt  l1,l2,lb1,lb2,ij=i*nc+j;

      /* Null matrix pointers are allowed in MATNEST */
      if (!nest[i][j]) continue;

      /* Nested matrices should be of type MATIS */
      PetscCall(PetscObjectTypeCompare((PetscObject)nest[i][j],MATTRANSPOSEMAT,&istrans[ij]));
      if (istrans[ij]) {
        Mat T,lT;
        PetscCall(MatTransposeGetMat(nest[i][j],&T));
        PetscCall(PetscObjectTypeCompare((PetscObject)T,MATIS,&ismatis));
        PetscCheck(ismatis,comm,PETSC_ERR_SUP,"Cannot convert from MATNEST to MATIS! Matrix block (%" PetscInt_FMT ",%" PetscInt_FMT ") (transposed) is not of type MATIS",i,j);
        PetscCall(MatISGetLocalMat(T,&lT));
        PetscCall(MatCreateTranspose(lT,&snest[ij]));
      } else {
        PetscCall(PetscObjectTypeCompare((PetscObject)nest[i][j],MATIS,&ismatis));
        PetscCheck(ismatis,comm,PETSC_ERR_SUP,"Cannot convert from MATNEST to MATIS! Matrix block (%" PetscInt_FMT ",%" PetscInt_FMT ") is not of type MATIS",i,j);
        PetscCall(MatISGetLocalMat(nest[i][j],&snest[ij]));
      }

      /* Check compatibility of local sizes */
      PetscCall(MatGetSize(snest[ij],&l1,&l2));
      PetscCall(MatGetBlockSizes(snest[ij],&lb1,&lb2));
      if (!l1 || !l2) continue;
      PetscCheckFalse(lr[i] && l1 != lr[i],PETSC_COMM_SELF,PETSC_ERR_SUP,"Cannot convert from MATNEST to MATIS! Matrix block (%" PetscInt_FMT ",%" PetscInt_FMT ") has invalid local size %" PetscInt_FMT " != %" PetscInt_FMT,i,j,lr[i],l1);
      PetscCheckFalse(lc[j] && l2 != lc[j],PETSC_COMM_SELF,PETSC_ERR_SUP,"Cannot convert from MATNEST to MATIS! Matrix block (%" PetscInt_FMT ",%" PetscInt_FMT ") has invalid local size %" PetscInt_FMT " != %" PetscInt_FMT,i,j,lc[j],l2);
      lr[i] = l1;
      lc[j] = l2;

      /* check compatibilty for local matrix reusage */
      if (rnest && !rnest[i][j] != !snest[ij]) lreuse = PETSC_FALSE;
    }
  }

  if (PetscDefined (USE_DEBUG)) {
    /* Check compatibility of l2g maps for rows */
    for (i=0;i<nr;i++) {
      rl2g = NULL;
      for (j=0;j<nc;j++) {
        PetscInt n1,n2;

        if (!nest[i][j]) continue;
        if (istrans[i*nc+j]) {
          Mat T;

          PetscCall(MatTransposeGetMat(nest[i][j],&T));
          PetscCall(MatISGetLocalToGlobalMapping(T,NULL,&cl2g));
        } else {
          PetscCall(MatISGetLocalToGlobalMapping(nest[i][j],&cl2g,NULL));
        }
        PetscCall(ISLocalToGlobalMappingGetSize(cl2g,&n1));
        if (!n1) continue;
        if (!rl2g) {
          rl2g = cl2g;
        } else {
          const PetscInt *idxs1,*idxs2;
          PetscBool      same;

          PetscCall(ISLocalToGlobalMappingGetSize(rl2g,&n2));
          PetscCheck(n1 == n2,PETSC_COMM_SELF,PETSC_ERR_SUP,"Cannot convert from MATNEST to MATIS! Matrix block (%" PetscInt_FMT ",%" PetscInt_FMT ") has invalid row l2gmap size %" PetscInt_FMT " != %" PetscInt_FMT,i,j,n1,n2);
          PetscCall(ISLocalToGlobalMappingGetIndices(cl2g,&idxs1));
          PetscCall(ISLocalToGlobalMappingGetIndices(rl2g,&idxs2));
          PetscCall(PetscArraycmp(idxs1,idxs2,n1,&same));
          PetscCall(ISLocalToGlobalMappingRestoreIndices(cl2g,&idxs1));
          PetscCall(ISLocalToGlobalMappingRestoreIndices(rl2g,&idxs2));
          PetscCheck(same,PETSC_COMM_SELF,PETSC_ERR_SUP,"Cannot convert from MATNEST to MATIS! Matrix block (%" PetscInt_FMT ",%" PetscInt_FMT ") has invalid row l2gmap",i,j);
        }
      }
    }
    /* Check compatibility of l2g maps for columns */
    for (i=0;i<nc;i++) {
      rl2g = NULL;
      for (j=0;j<nr;j++) {
        PetscInt n1,n2;

        if (!nest[j][i]) continue;
        if (istrans[j*nc+i]) {
          Mat T;

          PetscCall(MatTransposeGetMat(nest[j][i],&T));
          PetscCall(MatISGetLocalToGlobalMapping(T,&cl2g,NULL));
        } else {
          PetscCall(MatISGetLocalToGlobalMapping(nest[j][i],NULL,&cl2g));
        }
        PetscCall(ISLocalToGlobalMappingGetSize(cl2g,&n1));
        if (!n1) continue;
        if (!rl2g) {
          rl2g = cl2g;
        } else {
          const PetscInt *idxs1,*idxs2;
          PetscBool      same;

          PetscCall(ISLocalToGlobalMappingGetSize(rl2g,&n2));
          PetscCheck(n1 == n2,PETSC_COMM_SELF,PETSC_ERR_SUP,"Cannot convert from MATNEST to MATIS! Matrix block (%" PetscInt_FMT ",%" PetscInt_FMT ") has invalid column l2gmap size %" PetscInt_FMT " != %" PetscInt_FMT,j,i,n1,n2);
          PetscCall(ISLocalToGlobalMappingGetIndices(cl2g,&idxs1));
          PetscCall(ISLocalToGlobalMappingGetIndices(rl2g,&idxs2));
          PetscCall(PetscArraycmp(idxs1,idxs2,n1,&same));
          PetscCall(ISLocalToGlobalMappingRestoreIndices(cl2g,&idxs1));
          PetscCall(ISLocalToGlobalMappingRestoreIndices(rl2g,&idxs2));
          PetscCheck(same,PETSC_COMM_SELF,PETSC_ERR_SUP,"Cannot convert from MATNEST to MATIS! Matrix block (%" PetscInt_FMT ",%" PetscInt_FMT ") has invalid column l2gmap",j,i);
        }
      }
    }
  }

  B = NULL;
  if (reuse != MAT_REUSE_MATRIX) {
    PetscInt stl;

    /* Create l2g map for the rows of the new matrix and index sets for the local MATNEST */
    for (i=0,stl=0;i<nr;i++) stl += lr[i];
    PetscCall(PetscMalloc1(stl,&l2gidxs));
    for (i=0,stl=0;i<nr;i++) {
      Mat            usedmat;
      Mat_IS         *matis;
      const PetscInt *idxs;

      /* local IS for local NEST */
      PetscCall(ISCreateStride(PETSC_COMM_SELF,lr[i],stl,1,&islrow[i]));

      /* l2gmap */
      j = 0;
      usedmat = nest[i][j];
      while (!usedmat && j < nc-1) usedmat = nest[i][++j];
      PetscCheck(usedmat,comm,PETSC_ERR_SUP,"Cannot find valid row mat");

      if (istrans[i*nc+j]) {
        Mat T;
        PetscCall(MatTransposeGetMat(usedmat,&T));
        usedmat = T;
      }
      matis = (Mat_IS*)(usedmat->data);
      PetscCall(ISGetIndices(isrow[i],&idxs));
      if (istrans[i*nc+j]) {
        PetscCall(PetscSFBcastBegin(matis->csf,MPIU_INT,idxs,l2gidxs+stl,MPI_REPLACE));
        PetscCall(PetscSFBcastEnd(matis->csf,MPIU_INT,idxs,l2gidxs+stl,MPI_REPLACE));
      } else {
        PetscCall(PetscSFBcastBegin(matis->sf,MPIU_INT,idxs,l2gidxs+stl,MPI_REPLACE));
        PetscCall(PetscSFBcastEnd(matis->sf,MPIU_INT,idxs,l2gidxs+stl,MPI_REPLACE));
      }
      PetscCall(ISRestoreIndices(isrow[i],&idxs));
      stl += lr[i];
    }
    PetscCall(ISLocalToGlobalMappingCreate(comm,1,stl,l2gidxs,PETSC_OWN_POINTER,&rl2g));

    /* Create l2g map for columns of the new matrix and index sets for the local MATNEST */
    for (i=0,stl=0;i<nc;i++) stl += lc[i];
    PetscCall(PetscMalloc1(stl,&l2gidxs));
    for (i=0,stl=0;i<nc;i++) {
      Mat            usedmat;
      Mat_IS         *matis;
      const PetscInt *idxs;

      /* local IS for local NEST */
      PetscCall(ISCreateStride(PETSC_COMM_SELF,lc[i],stl,1,&islcol[i]));

      /* l2gmap */
      j = 0;
      usedmat = nest[j][i];
      while (!usedmat && j < nr-1) usedmat = nest[++j][i];
      PetscCheck(usedmat,comm,PETSC_ERR_SUP,"Cannot find valid column mat");
      if (istrans[j*nc+i]) {
        Mat T;
        PetscCall(MatTransposeGetMat(usedmat,&T));
        usedmat = T;
      }
      matis = (Mat_IS*)(usedmat->data);
      PetscCall(ISGetIndices(iscol[i],&idxs));
      if (istrans[j*nc+i]) {
        PetscCall(PetscSFBcastBegin(matis->sf,MPIU_INT,idxs,l2gidxs+stl,MPI_REPLACE));
        PetscCall(PetscSFBcastEnd(matis->sf,MPIU_INT,idxs,l2gidxs+stl,MPI_REPLACE));
      } else {
        PetscCall(PetscSFBcastBegin(matis->csf,MPIU_INT,idxs,l2gidxs+stl,MPI_REPLACE));
        PetscCall(PetscSFBcastEnd(matis->csf,MPIU_INT,idxs,l2gidxs+stl,MPI_REPLACE));
      }
      PetscCall(ISRestoreIndices(iscol[i],&idxs));
      stl += lc[i];
    }
    PetscCall(ISLocalToGlobalMappingCreate(comm,1,stl,l2gidxs,PETSC_OWN_POINTER,&cl2g));

    /* Create MATIS */
    PetscCall(MatCreate(comm,&B));
    PetscCall(MatSetSizes(B,A->rmap->n,A->cmap->n,A->rmap->N,A->cmap->N));
    PetscCall(MatGetBlockSizes(A,&rbs,&cbs));
    PetscCall(MatSetBlockSizes(B,rbs,cbs));
    PetscCall(MatSetType(B,MATIS));
    PetscCall(MatISSetLocalMatType(B,MATNEST));
    { /* hack : avoid setup of scatters */
      Mat_IS *matis = (Mat_IS*)(B->data);
      matis->islocalref = PETSC_TRUE;
    }
    PetscCall(MatSetLocalToGlobalMapping(B,rl2g,cl2g));
    PetscCall(ISLocalToGlobalMappingDestroy(&rl2g));
    PetscCall(ISLocalToGlobalMappingDestroy(&cl2g));
    PetscCall(MatCreateNest(PETSC_COMM_SELF,nr,islrow,nc,islcol,snest,&lA));
    PetscCall(MatNestSetVecType(lA,VECNEST));
    for (i=0;i<nr*nc;i++) {
      if (istrans[i]) {
        PetscCall(MatDestroy(&snest[i]));
      }
    }
    PetscCall(MatISSetLocalMat(B,lA));
    PetscCall(MatDestroy(&lA));
    { /* hack : setup of scatters done here */
      Mat_IS *matis = (Mat_IS*)(B->data);

      matis->islocalref = PETSC_FALSE;
      PetscCall(MatISSetUpScatters_Private(B));
    }
    PetscCall(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));
    if (reuse == MAT_INPLACE_MATRIX) {
      PetscCall(MatHeaderReplace(A,&B));
    } else {
      *newmat = B;
    }
  } else {
    if (lreuse) {
      PetscCall(MatISGetLocalMat(*newmat,&lA));
      for (i=0;i<nr;i++) {
        for (j=0;j<nc;j++) {
          if (snest[i*nc+j]) {
            PetscCall(MatNestSetSubMat(lA,i,j,snest[i*nc+j]));
            if (istrans[i*nc+j]) {
              PetscCall(MatDestroy(&snest[i*nc+j]));
            }
          }
        }
      }
    } else {
      PetscInt stl;
      for (i=0,stl=0;i<nr;i++) {
        PetscCall(ISCreateStride(PETSC_COMM_SELF,lr[i],stl,1,&islrow[i]));
        stl  += lr[i];
      }
      for (i=0,stl=0;i<nc;i++) {
        PetscCall(ISCreateStride(PETSC_COMM_SELF,lc[i],stl,1,&islcol[i]));
        stl  += lc[i];
      }
      PetscCall(MatCreateNest(PETSC_COMM_SELF,nr,islrow,nc,islcol,snest,&lA));
      for (i=0;i<nr*nc;i++) {
        if (istrans[i]) {
          PetscCall(MatDestroy(&snest[i]));
        }
      }
      PetscCall(MatISSetLocalMat(*newmat,lA));
      PetscCall(MatDestroy(&lA));
    }
    PetscCall(MatAssemblyBegin(*newmat,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(*newmat,MAT_FINAL_ASSEMBLY));
  }

  /* Create local matrix in MATNEST format */
  convert = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(NULL,((PetscObject)A)->prefix,"-matis_convert_local_nest",&convert,NULL));
  if (convert) {
    Mat              M;
    MatISLocalFields lf;
    PetscContainer   c;

    PetscCall(MatISGetLocalMat(*newmat,&lA));
    PetscCall(MatConvert(lA,MATAIJ,MAT_INITIAL_MATRIX,&M));
    PetscCall(MatISSetLocalMat(*newmat,M));
    PetscCall(MatDestroy(&M));

    /* attach local fields to the matrix */
    PetscCall(PetscNew(&lf));
    PetscCall(PetscMalloc2(nr,&lf->rf,nc,&lf->cf));
    for (i=0;i<nr;i++) {
      PetscInt n,st;

      PetscCall(ISGetLocalSize(islrow[i],&n));
      PetscCall(ISStrideGetInfo(islrow[i],&st,NULL));
      PetscCall(ISCreateStride(comm,n,st,1,&lf->rf[i]));
    }
    for (i=0;i<nc;i++) {
      PetscInt n,st;

      PetscCall(ISGetLocalSize(islcol[i],&n));
      PetscCall(ISStrideGetInfo(islcol[i],&st,NULL));
      PetscCall(ISCreateStride(comm,n,st,1,&lf->cf[i]));
    }
    lf->nr = nr;
    lf->nc = nc;
    PetscCall(PetscContainerCreate(PetscObjectComm((PetscObject)(*newmat)),&c));
    PetscCall(PetscContainerSetPointer(c,lf));
    PetscCall(PetscContainerSetUserDestroy(c,MatISContainerDestroyFields_Private));
    PetscCall(PetscObjectCompose((PetscObject)(*newmat),"_convert_nest_lfields",(PetscObject)c));
    PetscCall(PetscContainerDestroy(&c));
  }

  /* Free workspace */
  for (i=0;i<nr;i++) {
    PetscCall(ISDestroy(&islrow[i]));
  }
  for (i=0;i<nc;i++) {
    PetscCall(ISDestroy(&islcol[i]));
  }
  PetscCall(PetscFree6(isrow,iscol,islrow,islcol,snest,istrans));
  PetscCall(PetscFree2(lr,lc));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDiagonalScale_IS(Mat A, Vec l, Vec r)
{
  Mat_IS            *matis = (Mat_IS*)A->data;
  Vec               ll,rr;
  const PetscScalar *Y,*X;
  PetscScalar       *x,*y;

  PetscFunctionBegin;
  if (l) {
    ll   = matis->y;
    PetscCall(VecGetArrayRead(l,&Y));
    PetscCall(VecGetArray(ll,&y));
    PetscCall(PetscSFBcastBegin(matis->sf,MPIU_SCALAR,Y,y,MPI_REPLACE));
  } else {
    ll = NULL;
  }
  if (r) {
    rr   = matis->x;
    PetscCall(VecGetArrayRead(r,&X));
    PetscCall(VecGetArray(rr,&x));
    PetscCall(PetscSFBcastBegin(matis->csf,MPIU_SCALAR,X,x,MPI_REPLACE));
  } else {
    rr = NULL;
  }
  if (ll) {
    PetscCall(PetscSFBcastEnd(matis->sf,MPIU_SCALAR,Y,y,MPI_REPLACE));
    PetscCall(VecRestoreArrayRead(l,&Y));
    PetscCall(VecRestoreArray(ll,&y));
  }
  if (rr) {
    PetscCall(PetscSFBcastEnd(matis->csf,MPIU_SCALAR,X,x,MPI_REPLACE));
    PetscCall(VecRestoreArrayRead(r,&X));
    PetscCall(VecRestoreArray(rr,&x));
  }
  PetscCall(MatDiagonalScale(matis->A,ll,rr));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatGetInfo_IS(Mat A,MatInfoType flag,MatInfo *ginfo)
{
  Mat_IS         *matis = (Mat_IS*)A->data;
  MatInfo        info;
  PetscLogDouble isend[6],irecv[6];
  PetscInt       bs;

  PetscFunctionBegin;
  PetscCall(MatGetBlockSize(A,&bs));
  if (matis->A->ops->getinfo) {
    PetscCall(MatGetInfo(matis->A,MAT_LOCAL,&info));
    isend[0] = info.nz_used;
    isend[1] = info.nz_allocated;
    isend[2] = info.nz_unneeded;
    isend[3] = info.memory;
    isend[4] = info.mallocs;
  } else {
    isend[0] = 0.;
    isend[1] = 0.;
    isend[2] = 0.;
    isend[3] = 0.;
    isend[4] = 0.;
  }
  isend[5] = matis->A->num_ass;
  if (flag == MAT_LOCAL) {
    ginfo->nz_used      = isend[0];
    ginfo->nz_allocated = isend[1];
    ginfo->nz_unneeded  = isend[2];
    ginfo->memory       = isend[3];
    ginfo->mallocs      = isend[4];
    ginfo->assemblies   = isend[5];
  } else if (flag == MAT_GLOBAL_MAX) {
    PetscCall(MPIU_Allreduce(isend,irecv,6,MPIU_PETSCLOGDOUBLE,MPI_MAX,PetscObjectComm((PetscObject)A)));

    ginfo->nz_used      = irecv[0];
    ginfo->nz_allocated = irecv[1];
    ginfo->nz_unneeded  = irecv[2];
    ginfo->memory       = irecv[3];
    ginfo->mallocs      = irecv[4];
    ginfo->assemblies   = irecv[5];
  } else if (flag == MAT_GLOBAL_SUM) {
    PetscCall(MPIU_Allreduce(isend,irecv,5,MPIU_PETSCLOGDOUBLE,MPI_SUM,PetscObjectComm((PetscObject)A)));

    ginfo->nz_used      = irecv[0];
    ginfo->nz_allocated = irecv[1];
    ginfo->nz_unneeded  = irecv[2];
    ginfo->memory       = irecv[3];
    ginfo->mallocs      = irecv[4];
    ginfo->assemblies   = A->num_ass;
  }
  ginfo->block_size        = bs;
  ginfo->fill_ratio_given  = 0;
  ginfo->fill_ratio_needed = 0;
  ginfo->factor_mallocs    = 0;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatTranspose_IS(Mat A,MatReuse reuse,Mat *B)
{
  Mat                    C,lC,lA;

  PetscFunctionBegin;
  if (reuse == MAT_INITIAL_MATRIX || reuse == MAT_INPLACE_MATRIX) {
    ISLocalToGlobalMapping rl2g,cl2g;
    PetscCall(MatCreate(PetscObjectComm((PetscObject)A),&C));
    PetscCall(MatSetSizes(C,A->cmap->n,A->rmap->n,A->cmap->N,A->rmap->N));
    PetscCall(MatSetBlockSizes(C,PetscAbs(A->cmap->bs),PetscAbs(A->rmap->bs)));
    PetscCall(MatSetType(C,MATIS));
    PetscCall(MatGetLocalToGlobalMapping(A,&rl2g,&cl2g));
    PetscCall(MatSetLocalToGlobalMapping(C,cl2g,rl2g));
  } else C = *B;

  /* perform local transposition */
  PetscCall(MatISGetLocalMat(A,&lA));
  PetscCall(MatTranspose(lA,MAT_INITIAL_MATRIX,&lC));
  PetscCall(MatSetLocalToGlobalMapping(lC,lA->cmap->mapping,lA->rmap->mapping));
  PetscCall(MatISSetLocalMat(C,lC));
  PetscCall(MatDestroy(&lC));

  if (reuse == MAT_INITIAL_MATRIX || reuse == MAT_REUSE_MATRIX) {
    *B = C;
  } else {
    PetscCall(MatHeaderMerge(A,&C));
  }
  PetscCall(MatAssemblyBegin(*B,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*B,MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDiagonalSet_IS(Mat A,Vec D,InsertMode insmode)
{
  Mat_IS         *is = (Mat_IS*)A->data;

  PetscFunctionBegin;
  if (D) { /* MatShift_IS pass D = NULL */
    PetscCall(VecScatterBegin(is->rctx,D,is->y,INSERT_VALUES,SCATTER_FORWARD));
    PetscCall(VecScatterEnd(is->rctx,D,is->y,INSERT_VALUES,SCATTER_FORWARD));
  }
  PetscCall(VecPointwiseDivide(is->y,is->y,is->counter));
  PetscCall(MatDiagonalSet(is->A,is->y,insmode));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatShift_IS(Mat A,PetscScalar a)
{
  Mat_IS         *is = (Mat_IS*)A->data;

  PetscFunctionBegin;
  PetscCall(VecSet(is->y,a));
  PetscCall(MatDiagonalSet_IS(A,NULL,ADD_VALUES));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSetValuesLocal_SubMat_IS(Mat A,PetscInt m,const PetscInt *rows, PetscInt n,const PetscInt *cols,const PetscScalar *values,InsertMode addv)
{
  PetscInt       rows_l[MATIS_MAX_ENTRIES_INSERTION],cols_l[MATIS_MAX_ENTRIES_INSERTION];

  PetscFunctionBegin;
  PetscCheckFalse(m > MATIS_MAX_ENTRIES_INSERTION || n > MATIS_MAX_ENTRIES_INSERTION,PETSC_COMM_SELF,PETSC_ERR_SUP,"Number of row/column indices must be <= %d: they are %" PetscInt_FMT " %" PetscInt_FMT,MATIS_MAX_ENTRIES_INSERTION,m,n);
  PetscCall(ISLocalToGlobalMappingApply(A->rmap->mapping,m,rows,rows_l));
  PetscCall(ISLocalToGlobalMappingApply(A->cmap->mapping,n,cols,cols_l));
  PetscCall(MatSetValuesLocal_IS(A,m,rows_l,n,cols_l,values,addv));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSetValuesBlockedLocal_SubMat_IS(Mat A,PetscInt m,const PetscInt *rows, PetscInt n,const PetscInt *cols,const PetscScalar *values,InsertMode addv)
{
  PetscInt       rows_l[MATIS_MAX_ENTRIES_INSERTION],cols_l[MATIS_MAX_ENTRIES_INSERTION];

  PetscFunctionBegin;
  PetscCheckFalse(m > MATIS_MAX_ENTRIES_INSERTION || n > MATIS_MAX_ENTRIES_INSERTION,PETSC_COMM_SELF,PETSC_ERR_SUP,"Number of row/column block indices must be <= %d: they are %" PetscInt_FMT " %" PetscInt_FMT,MATIS_MAX_ENTRIES_INSERTION,m,n);
  PetscCall(ISLocalToGlobalMappingApplyBlock(A->rmap->mapping,m,rows,rows_l));
  PetscCall(ISLocalToGlobalMappingApplyBlock(A->cmap->mapping,n,cols,cols_l));
  PetscCall(MatSetValuesBlockedLocal_IS(A,m,rows_l,n,cols_l,values,addv));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatCreateSubMatrix_IS(Mat mat,IS irow,IS icol,MatReuse scall,Mat *newmat)
{
  Mat               locmat,newlocmat;
  Mat_IS            *newmatis;
  const PetscInt    *idxs;
  PetscInt          i,m,n;

  PetscFunctionBegin;
  if (scall == MAT_REUSE_MATRIX) {
    PetscBool ismatis;

    PetscCall(PetscObjectTypeCompare((PetscObject)*newmat,MATIS,&ismatis));
    PetscCheck(ismatis,PetscObjectComm((PetscObject)*newmat),PETSC_ERR_ARG_WRONG,"Cannot reuse matrix! Not of MATIS type");
    newmatis = (Mat_IS*)(*newmat)->data;
    PetscCheck(newmatis->getsub_ris,PetscObjectComm((PetscObject)*newmat),PETSC_ERR_ARG_WRONG,"Cannot reuse matrix! Misses local row IS");
    PetscCheck(newmatis->getsub_cis,PetscObjectComm((PetscObject)*newmat),PETSC_ERR_ARG_WRONG,"Cannot reuse matrix! Misses local col IS");
  }
  /* irow and icol may not have duplicate entries */
  if (PetscDefined(USE_DEBUG)) {
    Vec               rtest,ltest;
    const PetscScalar *array;

    PetscCall(MatCreateVecs(mat,&ltest,&rtest));
    PetscCall(ISGetLocalSize(irow,&n));
    PetscCall(ISGetIndices(irow,&idxs));
    for (i=0;i<n;i++) {
      PetscCall(VecSetValue(rtest,idxs[i],1.0,ADD_VALUES));
    }
    PetscCall(VecAssemblyBegin(rtest));
    PetscCall(VecAssemblyEnd(rtest));
    PetscCall(VecGetLocalSize(rtest,&n));
    PetscCall(VecGetOwnershipRange(rtest,&m,NULL));
    PetscCall(VecGetArrayRead(rtest,&array));
    for (i=0;i<n;i++) PetscCheckFalse(array[i] != 0. && array[i] != 1.,PETSC_COMM_SELF,PETSC_ERR_SUP,"Index %" PetscInt_FMT " counted %" PetscInt_FMT " times! Irow may not have duplicate entries",i+m,(PetscInt)PetscRealPart(array[i]));
    PetscCall(VecRestoreArrayRead(rtest,&array));
    PetscCall(ISRestoreIndices(irow,&idxs));
    PetscCall(ISGetLocalSize(icol,&n));
    PetscCall(ISGetIndices(icol,&idxs));
    for (i=0;i<n;i++) {
      PetscCall(VecSetValue(ltest,idxs[i],1.0,ADD_VALUES));
    }
    PetscCall(VecAssemblyBegin(ltest));
    PetscCall(VecAssemblyEnd(ltest));
    PetscCall(VecGetLocalSize(ltest,&n));
    PetscCall(VecGetOwnershipRange(ltest,&m,NULL));
    PetscCall(VecGetArrayRead(ltest,&array));
    for (i=0;i<n;i++) PetscCheckFalse(array[i] != 0. && array[i] != 1.,PETSC_COMM_SELF,PETSC_ERR_SUP,"Index %" PetscInt_FMT " counted %" PetscInt_FMT " times! Icol may not have duplicate entries",i+m,(PetscInt)PetscRealPart(array[i]));
    PetscCall(VecRestoreArrayRead(ltest,&array));
    PetscCall(ISRestoreIndices(icol,&idxs));
    PetscCall(VecDestroy(&rtest));
    PetscCall(VecDestroy(&ltest));
  }
  if (scall == MAT_INITIAL_MATRIX) {
    Mat_IS                 *matis = (Mat_IS*)mat->data;
    ISLocalToGlobalMapping rl2g;
    IS                     is;
    PetscInt               *lidxs,*lgidxs,*newgidxs;
    PetscInt               ll,newloc,irbs,icbs,arbs,acbs,rbs,cbs;
    PetscBool              cong;
    MPI_Comm               comm;

    PetscCall(PetscObjectGetComm((PetscObject)mat,&comm));
    PetscCall(MatGetBlockSizes(mat,&arbs,&acbs));
    PetscCall(ISGetBlockSize(irow,&irbs));
    PetscCall(ISGetBlockSize(icol,&icbs));
    rbs  = arbs == irbs ? irbs : 1;
    cbs  = acbs == icbs ? icbs : 1;
    PetscCall(ISGetLocalSize(irow,&m));
    PetscCall(ISGetLocalSize(icol,&n));
    PetscCall(MatCreate(comm,newmat));
    PetscCall(MatSetType(*newmat,MATIS));
    PetscCall(MatSetSizes(*newmat,m,n,PETSC_DECIDE,PETSC_DECIDE));
    PetscCall(MatSetBlockSizes(*newmat,rbs,cbs));
    /* communicate irow to their owners in the layout */
    PetscCall(ISGetIndices(irow,&idxs));
    PetscCall(PetscLayoutMapLocal(mat->rmap,m,idxs,&ll,&lidxs,&lgidxs));
    PetscCall(ISRestoreIndices(irow,&idxs));
    PetscCall(PetscArrayzero(matis->sf_rootdata,matis->sf->nroots));
    for (i=0;i<ll;i++) matis->sf_rootdata[lidxs[i]] = lgidxs[i]+1;
    PetscCall(PetscFree(lidxs));
    PetscCall(PetscFree(lgidxs));
    PetscCall(PetscSFBcastBegin(matis->sf,MPIU_INT,matis->sf_rootdata,matis->sf_leafdata,MPI_REPLACE));
    PetscCall(PetscSFBcastEnd(matis->sf,MPIU_INT,matis->sf_rootdata,matis->sf_leafdata,MPI_REPLACE));
    for (i=0,newloc=0;i<matis->sf->nleaves;i++) if (matis->sf_leafdata[i]) newloc++;
    PetscCall(PetscMalloc1(newloc,&newgidxs));
    PetscCall(PetscMalloc1(newloc,&lidxs));
    for (i=0,newloc=0;i<matis->sf->nleaves;i++)
      if (matis->sf_leafdata[i]) {
        lidxs[newloc] = i;
        newgidxs[newloc++] = matis->sf_leafdata[i]-1;
      }
    PetscCall(ISCreateGeneral(comm,newloc,newgidxs,PETSC_OWN_POINTER,&is));
    PetscCall(ISLocalToGlobalMappingCreateIS(is,&rl2g));
    PetscCall(ISLocalToGlobalMappingSetBlockSize(rl2g,rbs));
    PetscCall(ISDestroy(&is));
    /* local is to extract local submatrix */
    newmatis = (Mat_IS*)(*newmat)->data;
    PetscCall(ISCreateGeneral(comm,newloc,lidxs,PETSC_OWN_POINTER,&newmatis->getsub_ris));
    PetscCall(MatHasCongruentLayouts(mat,&cong));
    if (cong && irow == icol && matis->csf == matis->sf) {
      PetscCall(MatSetLocalToGlobalMapping(*newmat,rl2g,rl2g));
      PetscCall(PetscObjectReference((PetscObject)newmatis->getsub_ris));
      newmatis->getsub_cis = newmatis->getsub_ris;
    } else {
      ISLocalToGlobalMapping cl2g;

      /* communicate icol to their owners in the layout */
      PetscCall(ISGetIndices(icol,&idxs));
      PetscCall(PetscLayoutMapLocal(mat->cmap,n,idxs,&ll,&lidxs,&lgidxs));
      PetscCall(ISRestoreIndices(icol,&idxs));
      PetscCall(PetscArrayzero(matis->csf_rootdata,matis->csf->nroots));
      for (i=0;i<ll;i++) matis->csf_rootdata[lidxs[i]] = lgidxs[i]+1;
      PetscCall(PetscFree(lidxs));
      PetscCall(PetscFree(lgidxs));
      PetscCall(PetscSFBcastBegin(matis->csf,MPIU_INT,matis->csf_rootdata,matis->csf_leafdata,MPI_REPLACE));
      PetscCall(PetscSFBcastEnd(matis->csf,MPIU_INT,matis->csf_rootdata,matis->csf_leafdata,MPI_REPLACE));
      for (i=0,newloc=0;i<matis->csf->nleaves;i++) if (matis->csf_leafdata[i]) newloc++;
      PetscCall(PetscMalloc1(newloc,&newgidxs));
      PetscCall(PetscMalloc1(newloc,&lidxs));
      for (i=0,newloc=0;i<matis->csf->nleaves;i++)
        if (matis->csf_leafdata[i]) {
          lidxs[newloc] = i;
          newgidxs[newloc++] = matis->csf_leafdata[i]-1;
        }
      PetscCall(ISCreateGeneral(comm,newloc,newgidxs,PETSC_OWN_POINTER,&is));
      PetscCall(ISLocalToGlobalMappingCreateIS(is,&cl2g));
      PetscCall(ISLocalToGlobalMappingSetBlockSize(cl2g,cbs));
      PetscCall(ISDestroy(&is));
      /* local is to extract local submatrix */
      PetscCall(ISCreateGeneral(comm,newloc,lidxs,PETSC_OWN_POINTER,&newmatis->getsub_cis));
      PetscCall(MatSetLocalToGlobalMapping(*newmat,rl2g,cl2g));
      PetscCall(ISLocalToGlobalMappingDestroy(&cl2g));
    }
    PetscCall(ISLocalToGlobalMappingDestroy(&rl2g));
  } else {
    PetscCall(MatISGetLocalMat(*newmat,&newlocmat));
  }
  PetscCall(MatISGetLocalMat(mat,&locmat));
  newmatis = (Mat_IS*)(*newmat)->data;
  PetscCall(MatCreateSubMatrix(locmat,newmatis->getsub_ris,newmatis->getsub_cis,scall,&newlocmat));
  if (scall == MAT_INITIAL_MATRIX) {
    PetscCall(MatISSetLocalMat(*newmat,newlocmat));
    PetscCall(MatDestroy(&newlocmat));
  }
  PetscCall(MatAssemblyBegin(*newmat,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*newmat,MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatCopy_IS(Mat A,Mat B,MatStructure str)
{
  Mat_IS         *a = (Mat_IS*)A->data,*b;
  PetscBool      ismatis;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)B,MATIS,&ismatis));
  PetscCheck(ismatis,PetscObjectComm((PetscObject)B),PETSC_ERR_SUP,"Need to be implemented");
  b = (Mat_IS*)B->data;
  PetscCall(MatCopy(a->A,b->A,str));
  PetscCall(PetscObjectStateIncrease((PetscObject)B));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMissingDiagonal_IS(Mat A,PetscBool  *missing,PetscInt *d)
{
  Vec               v;
  const PetscScalar *array;
  PetscInt          i,n;

  PetscFunctionBegin;
  *missing = PETSC_FALSE;
  PetscCall(MatCreateVecs(A,NULL,&v));
  PetscCall(MatGetDiagonal(A,v));
  PetscCall(VecGetLocalSize(v,&n));
  PetscCall(VecGetArrayRead(v,&array));
  for (i=0;i<n;i++) if (array[i] == 0.) break;
  PetscCall(VecRestoreArrayRead(v,&array));
  PetscCall(VecDestroy(&v));
  if (i != n) *missing = PETSC_TRUE;
  if (d) {
    *d = -1;
    if (*missing) {
      PetscInt rstart;
      PetscCall(MatGetOwnershipRange(A,&rstart,NULL));
      *d = i+rstart;
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatISSetUpSF_IS(Mat B)
{
  Mat_IS         *matis = (Mat_IS*)(B->data);
  const PetscInt *gidxs;
  PetscInt       nleaves;

  PetscFunctionBegin;
  if (matis->sf) PetscFunctionReturn(0);
  PetscCall(PetscSFCreate(PetscObjectComm((PetscObject)B),&matis->sf));
  PetscCall(ISLocalToGlobalMappingGetIndices(matis->rmapping,&gidxs));
  PetscCall(ISLocalToGlobalMappingGetSize(matis->rmapping,&nleaves));
  PetscCall(PetscSFSetGraphLayout(matis->sf,B->rmap,nleaves,NULL,PETSC_OWN_POINTER,gidxs));
  PetscCall(ISLocalToGlobalMappingRestoreIndices(matis->rmapping,&gidxs));
  PetscCall(PetscMalloc2(matis->sf->nroots,&matis->sf_rootdata,matis->sf->nleaves,&matis->sf_leafdata));
  if (matis->rmapping != matis->cmapping) { /* setup SF for columns */
    PetscCall(ISLocalToGlobalMappingGetSize(matis->cmapping,&nleaves));
    PetscCall(PetscSFCreate(PetscObjectComm((PetscObject)B),&matis->csf));
    PetscCall(ISLocalToGlobalMappingGetIndices(matis->cmapping,&gidxs));
    PetscCall(PetscSFSetGraphLayout(matis->csf,B->cmap,nleaves,NULL,PETSC_OWN_POINTER,gidxs));
    PetscCall(ISLocalToGlobalMappingRestoreIndices(matis->cmapping,&gidxs));
    PetscCall(PetscMalloc2(matis->csf->nroots,&matis->csf_rootdata,matis->csf->nleaves,&matis->csf_leafdata));
  } else {
    matis->csf = matis->sf;
    matis->csf_leafdata = matis->sf_leafdata;
    matis->csf_rootdata = matis->sf_rootdata;
  }
  PetscFunctionReturn(0);
}

/*@
   MatISStoreL2L - Store local-to-local operators during the Galerkin process of MatPtAP.

   Collective

   Input Parameters:
+  A - the matrix
-  store - the boolean flag

   Level: advanced

   Notes:

.seealso: MatCreate(), MatCreateIS(), MatISSetPreallocation(), MatPtAP()
@*/
PetscErrorCode MatISStoreL2L(Mat A, PetscBool store)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidType(A,1);
  PetscValidLogicalCollectiveBool(A,store,2);
  PetscTryMethod(A,"MatISStoreL2L_C",(Mat,PetscBool),(A,store));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatISStoreL2L_IS(Mat A, PetscBool store)
{
  Mat_IS         *matis = (Mat_IS*)(A->data);

  PetscFunctionBegin;
  matis->storel2l = store;
  if (!store) {
    PetscCall(PetscObjectCompose((PetscObject)(A),"_MatIS_PtAP_l2l",NULL));
  }
  PetscFunctionReturn(0);
}

/*@
   MatISFixLocalEmpty - Compress out zero local rows from the local matrices

   Collective

   Input Parameters:
+  A - the matrix
-  fix - the boolean flag

   Level: advanced

   Notes: When fix is true, new local matrices and l2g maps are generated during the final assembly process.

.seealso: MatCreate(), MatCreateIS(), MatISSetPreallocation(), MatAssemblyEnd(), MAT_FINAL_ASSEMBLY
@*/
PetscErrorCode MatISFixLocalEmpty(Mat A, PetscBool fix)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidType(A,1);
  PetscValidLogicalCollectiveBool(A,fix,2);
  PetscTryMethod(A,"MatISFixLocalEmpty_C",(Mat,PetscBool),(A,fix));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatISFixLocalEmpty_IS(Mat A, PetscBool fix)
{
  Mat_IS *matis = (Mat_IS*)(A->data);

  PetscFunctionBegin;
  matis->locempty = fix;
  PetscFunctionReturn(0);
}

/*@
   MatISSetPreallocation - Preallocates memory for a MATIS parallel matrix.

   Collective

   Input Parameters:
+  B - the matrix
.  d_nz  - number of nonzeros per row in DIAGONAL portion of local submatrix
           (same value is used for all local rows)
.  d_nnz - array containing the number of nonzeros in the various rows of the
           DIAGONAL portion of the local submatrix (possibly different for each row)
           or NULL, if d_nz is used to specify the nonzero structure.
           The size of this array is equal to the number of local rows, i.e 'm'.
           For matrices that will be factored, you must leave room for (and set)
           the diagonal entry even if it is zero.
.  o_nz  - number of nonzeros per row in the OFF-DIAGONAL portion of local
           submatrix (same value is used for all local rows).
-  o_nnz - array containing the number of nonzeros in the various rows of the
           OFF-DIAGONAL portion of the local submatrix (possibly different for
           each row) or NULL, if o_nz is used to specify the nonzero
           structure. The size of this array is equal to the number
           of local rows, i.e 'm'.

   If the *_nnz parameter is given then the *_nz parameter is ignored

   Level: intermediate

   Notes:
    This function has the same interface as the MPIAIJ preallocation routine in order to simplify the transition
          from the asssembled format to the unassembled one. It overestimates the preallocation of MATIS local
          matrices; for exact preallocation, the user should set the preallocation directly on local matrix objects.

.seealso: MatCreate(), MatCreateIS(), MatMPIAIJSetPreallocation(), MatISGetLocalMat(), MATIS
@*/
PetscErrorCode MatISSetPreallocation(Mat B,PetscInt d_nz,const PetscInt d_nnz[],PetscInt o_nz,const PetscInt o_nnz[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(B,MAT_CLASSID,1);
  PetscValidType(B,1);
  PetscTryMethod(B,"MatISSetPreallocation_C",(Mat,PetscInt,const PetscInt[],PetscInt,const PetscInt[]),(B,d_nz,d_nnz,o_nz,o_nnz));
  PetscFunctionReturn(0);
}

/* this is used by DMDA */
PETSC_EXTERN PetscErrorCode MatISSetPreallocation_IS(Mat B,PetscInt d_nz,const PetscInt d_nnz[],PetscInt o_nz,const PetscInt o_nnz[])
{
  Mat_IS         *matis = (Mat_IS*)(B->data);
  PetscInt       bs,i,nlocalcols;

  PetscFunctionBegin;
  PetscCall(MatSetUp(B));
  if (!d_nnz) for (i=0;i<matis->sf->nroots;i++) matis->sf_rootdata[i] = d_nz;
  else for (i=0;i<matis->sf->nroots;i++) matis->sf_rootdata[i] = d_nnz[i];

  if (!o_nnz) for (i=0;i<matis->sf->nroots;i++) matis->sf_rootdata[i] += o_nz;
  else for (i=0;i<matis->sf->nroots;i++) matis->sf_rootdata[i] += o_nnz[i];

  PetscCall(PetscSFBcastBegin(matis->sf,MPIU_INT,matis->sf_rootdata,matis->sf_leafdata,MPI_REPLACE));
  PetscCall(MatGetSize(matis->A,NULL,&nlocalcols));
  PetscCall(MatGetBlockSize(matis->A,&bs));
  PetscCall(PetscSFBcastEnd(matis->sf,MPIU_INT,matis->sf_rootdata,matis->sf_leafdata,MPI_REPLACE));

  for (i=0;i<matis->sf->nleaves;i++) matis->sf_leafdata[i] = PetscMin(matis->sf_leafdata[i],nlocalcols);
  PetscCall(MatSeqAIJSetPreallocation(matis->A,0,matis->sf_leafdata));
#if defined(PETSC_HAVE_HYPRE)
  PetscCall(MatHYPRESetPreallocation(matis->A,0,matis->sf_leafdata,0,NULL));
#endif

  for (i=0;i<matis->sf->nleaves/bs;i++) {
    PetscInt b;

    matis->sf_leafdata[i] = matis->sf_leafdata[i*bs]/bs;
    for (b=1;b<bs;b++) {
      matis->sf_leafdata[i] = PetscMax(matis->sf_leafdata[i],matis->sf_leafdata[i*bs+b]/bs);
    }
  }
  PetscCall(MatSeqBAIJSetPreallocation(matis->A,bs,0,matis->sf_leafdata));

  nlocalcols /= bs;
  for (i=0;i<matis->sf->nleaves/bs;i++) matis->sf_leafdata[i] = PetscMin(matis->sf_leafdata[i],nlocalcols - i);
  PetscCall(MatSeqSBAIJSetPreallocation(matis->A,bs,0,matis->sf_leafdata));

  /* for other matrix types */
  PetscCall(MatSetUp(matis->A));
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode MatISSetMPIXAIJPreallocation_Private(Mat A, Mat B, PetscBool maxreduce)
{
  Mat_IS          *matis = (Mat_IS*)(A->data);
  PetscInt        *my_dnz,*my_onz,*dnz,*onz,*mat_ranges,*row_ownership;
  const PetscInt  *global_indices_r,*global_indices_c;
  PetscInt        i,j,bs,rows,cols;
  PetscInt        lrows,lcols;
  PetscInt        local_rows,local_cols;
  PetscMPIInt     size;
  PetscBool       isdense,issbaij;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)A),&size));
  PetscCall(MatGetSize(A,&rows,&cols));
  PetscCall(MatGetBlockSize(A,&bs));
  PetscCall(MatGetSize(matis->A,&local_rows,&local_cols));
  PetscCall(PetscObjectBaseTypeCompare((PetscObject)matis->A,MATSEQDENSE,&isdense));
  PetscCall(PetscObjectBaseTypeCompare((PetscObject)matis->A,MATSEQSBAIJ,&issbaij));
  PetscCall(ISLocalToGlobalMappingGetIndices(matis->rmapping,&global_indices_r));
  if (matis->rmapping != matis->cmapping) {
    PetscCall(ISLocalToGlobalMappingGetIndices(matis->cmapping,&global_indices_c));
  } else global_indices_c = global_indices_r;

  if (issbaij) PetscCall(MatGetRowUpperTriangular(matis->A));
  /*
     An SF reduce is needed to sum up properly on shared rows.
     Note that generally preallocation is not exact, since it overestimates nonzeros
  */
  PetscCall(MatGetLocalSize(A,&lrows,&lcols));
  ierr = MatPreallocateInitialize(PetscObjectComm((PetscObject)A),lrows,lcols,dnz,onz);PetscCall(ierr);
  /* All processes need to compute entire row ownership */
  PetscCall(PetscMalloc1(rows,&row_ownership));
  PetscCall(MatGetOwnershipRanges(A,(const PetscInt**)&mat_ranges));
  for (i=0;i<size;i++) {
    for (j=mat_ranges[i];j<mat_ranges[i+1];j++) row_ownership[j] = i;
  }
  PetscCall(MatGetOwnershipRangesColumn(A,(const PetscInt**)&mat_ranges));

  /*
     my_dnz and my_onz contains exact contribution to preallocation from each local mat
     then, they will be summed up properly. This way, preallocation is always sufficient
  */
  PetscCall(PetscCalloc2(local_rows,&my_dnz,local_rows,&my_onz));
  /* preallocation as a MATAIJ */
  if (isdense) { /* special case for dense local matrices */
    for (i=0;i<local_rows;i++) {
      PetscInt owner = row_ownership[global_indices_r[i]];
      for (j=0;j<local_cols;j++) {
        PetscInt index_col = global_indices_c[j];
        if (index_col > mat_ranges[owner]-1 && index_col < mat_ranges[owner+1]) { /* diag block */
          my_dnz[i] += 1;
        } else { /* offdiag block */
          my_onz[i] += 1;
        }
      }
    }
  } else if (matis->A->ops->getrowij) {
    const PetscInt *ii,*jj,*jptr;
    PetscBool      done;
    PetscCall(MatGetRowIJ(matis->A,0,PETSC_FALSE,PETSC_FALSE,&local_rows,&ii,&jj,&done));
    PetscCheck(done,PetscObjectComm((PetscObject)(matis->A)),PETSC_ERR_PLIB,"Error in MatGetRowIJ");
    jptr = jj;
    for (i=0;i<local_rows;i++) {
      PetscInt index_row = global_indices_r[i];
      for (j=0;j<ii[i+1]-ii[i];j++,jptr++) {
        PetscInt owner = row_ownership[index_row];
        PetscInt index_col = global_indices_c[*jptr];
        if (index_col > mat_ranges[owner]-1 && index_col < mat_ranges[owner+1]) { /* diag block */
          my_dnz[i] += 1;
        } else { /* offdiag block */
          my_onz[i] += 1;
        }
        /* same as before, interchanging rows and cols */
        if (issbaij && index_col != index_row) {
          owner = row_ownership[index_col];
          if (index_row > mat_ranges[owner]-1 && index_row < mat_ranges[owner+1]) {
            my_dnz[*jptr] += 1;
          } else {
            my_onz[*jptr] += 1;
          }
        }
      }
    }
    PetscCall(MatRestoreRowIJ(matis->A,0,PETSC_FALSE,PETSC_FALSE,&local_rows,&ii,&jj,&done));
    PetscCheck(done,PetscObjectComm((PetscObject)(matis->A)),PETSC_ERR_PLIB,"Error in MatRestoreRowIJ");
  } else { /* loop over rows and use MatGetRow */
    for (i=0;i<local_rows;i++) {
      const PetscInt *cols;
      PetscInt       ncols,index_row = global_indices_r[i];
      PetscCall(MatGetRow(matis->A,i,&ncols,&cols,NULL));
      for (j=0;j<ncols;j++) {
        PetscInt owner = row_ownership[index_row];
        PetscInt index_col = global_indices_c[cols[j]];
        if (index_col > mat_ranges[owner]-1 && index_col < mat_ranges[owner+1]) { /* diag block */
          my_dnz[i] += 1;
        } else { /* offdiag block */
          my_onz[i] += 1;
        }
        /* same as before, interchanging rows and cols */
        if (issbaij && index_col != index_row) {
          owner = row_ownership[index_col];
          if (index_row > mat_ranges[owner]-1 && index_row < mat_ranges[owner+1]) {
            my_dnz[cols[j]] += 1;
          } else {
            my_onz[cols[j]] += 1;
          }
        }
      }
      PetscCall(MatRestoreRow(matis->A,i,&ncols,&cols,NULL));
    }
  }
  if (global_indices_c != global_indices_r) {
    PetscCall(ISLocalToGlobalMappingRestoreIndices(matis->cmapping,&global_indices_c));
  }
  PetscCall(ISLocalToGlobalMappingRestoreIndices(matis->rmapping,&global_indices_r));
  PetscCall(PetscFree(row_ownership));

  /* Reduce my_dnz and my_onz */
  if (maxreduce) {
    PetscCall(PetscSFReduceBegin(matis->sf,MPIU_INT,my_dnz,dnz,MPI_MAX));
    PetscCall(PetscSFReduceBegin(matis->sf,MPIU_INT,my_onz,onz,MPI_MAX));
    PetscCall(PetscSFReduceEnd(matis->sf,MPIU_INT,my_dnz,dnz,MPI_MAX));
    PetscCall(PetscSFReduceEnd(matis->sf,MPIU_INT,my_onz,onz,MPI_MAX));
  } else {
    PetscCall(PetscSFReduceBegin(matis->sf,MPIU_INT,my_dnz,dnz,MPI_SUM));
    PetscCall(PetscSFReduceBegin(matis->sf,MPIU_INT,my_onz,onz,MPI_SUM));
    PetscCall(PetscSFReduceEnd(matis->sf,MPIU_INT,my_dnz,dnz,MPI_SUM));
    PetscCall(PetscSFReduceEnd(matis->sf,MPIU_INT,my_onz,onz,MPI_SUM));
  }
  PetscCall(PetscFree2(my_dnz,my_onz));

  /* Resize preallocation if overestimated */
  for (i=0;i<lrows;i++) {
    dnz[i] = PetscMin(dnz[i],lcols);
    onz[i] = PetscMin(onz[i],cols-lcols);
  }

  /* Set preallocation */
  PetscCall(MatSetBlockSizesFromMats(B,A,A));
  PetscCall(MatSeqAIJSetPreallocation(B,0,dnz));
  PetscCall(MatMPIAIJSetPreallocation(B,0,dnz,0,onz));
  for (i=0;i<lrows;i+=bs) {
    PetscInt b, d = dnz[i],o = onz[i];

    for (b=1;b<bs;b++) {
      d = PetscMax(d,dnz[i+b]);
      o = PetscMax(o,onz[i+b]);
    }
    dnz[i/bs] = PetscMin(d/bs + d%bs,lcols/bs);
    onz[i/bs] = PetscMin(o/bs + o%bs,(cols-lcols)/bs);
  }
  PetscCall(MatSeqBAIJSetPreallocation(B,bs,0,dnz));
  PetscCall(MatMPIBAIJSetPreallocation(B,bs,0,dnz,0,onz));
  PetscCall(MatMPISBAIJSetPreallocation(B,bs,0,dnz,0,onz));
  ierr = MatPreallocateFinalize(dnz,onz);PetscCall(ierr);
  if (issbaij) PetscCall(MatRestoreRowUpperTriangular(matis->A));
  PetscCall(MatSetOption(B,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_TRUE));
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatConvert_IS_XAIJ(Mat mat, MatType mtype, MatReuse reuse, Mat *M)
{
  Mat_IS            *matis = (Mat_IS*)(mat->data);
  Mat               local_mat,MT;
  PetscInt          rbs,cbs,rows,cols,lrows,lcols;
  PetscInt          local_rows,local_cols;
  PetscBool         isseqdense,isseqsbaij,isseqaij,isseqbaij;
  PetscMPIInt       size;
  const PetscScalar *array;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)mat),&size));
  if (size == 1 && mat->rmap->N == matis->A->rmap->N && mat->cmap->N == matis->A->cmap->N) {
    Mat      B;
    IS       irows = NULL,icols = NULL;
    PetscInt rbs,cbs;

    PetscCall(ISLocalToGlobalMappingGetBlockSize(matis->rmapping,&rbs));
    PetscCall(ISLocalToGlobalMappingGetBlockSize(matis->cmapping,&cbs));
    if (reuse != MAT_REUSE_MATRIX) { /* check if l2g maps are one-to-one */
      IS             rows,cols;
      const PetscInt *ridxs,*cidxs;
      PetscInt       i,nw,*work;

      PetscCall(ISLocalToGlobalMappingGetBlockIndices(matis->rmapping,&ridxs));
      PetscCall(ISLocalToGlobalMappingGetSize(matis->rmapping,&nw));
      nw   = nw/rbs;
      PetscCall(PetscCalloc1(nw,&work));
      for (i=0;i<nw;i++) work[ridxs[i]] += 1;
      for (i=0;i<nw;i++) if (!work[i] || work[i] > 1) break;
      if (i == nw) {
        PetscCall(ISCreateBlock(PETSC_COMM_SELF,rbs,nw,ridxs,PETSC_USE_POINTER,&rows));
        PetscCall(ISSetPermutation(rows));
        PetscCall(ISInvertPermutation(rows,PETSC_DECIDE,&irows));
        PetscCall(ISDestroy(&rows));
      }
      PetscCall(ISLocalToGlobalMappingRestoreBlockIndices(matis->rmapping,&ridxs));
      PetscCall(PetscFree(work));
      if (irows && matis->rmapping != matis->cmapping) {
        PetscCall(ISLocalToGlobalMappingGetBlockIndices(matis->cmapping,&cidxs));
        PetscCall(ISLocalToGlobalMappingGetSize(matis->cmapping,&nw));
        nw   = nw/cbs;
        PetscCall(PetscCalloc1(nw,&work));
        for (i=0;i<nw;i++) work[cidxs[i]] += 1;
        for (i=0;i<nw;i++) if (!work[i] || work[i] > 1) break;
        if (i == nw) {
          PetscCall(ISCreateBlock(PETSC_COMM_SELF,cbs,nw,cidxs,PETSC_USE_POINTER,&cols));
          PetscCall(ISSetPermutation(cols));
          PetscCall(ISInvertPermutation(cols,PETSC_DECIDE,&icols));
          PetscCall(ISDestroy(&cols));
        }
        PetscCall(ISLocalToGlobalMappingRestoreBlockIndices(matis->cmapping,&cidxs));
        PetscCall(PetscFree(work));
      } else if (irows) {
        PetscCall(PetscObjectReference((PetscObject)irows));
        icols = irows;
      }
    } else {
      PetscCall(PetscObjectQuery((PetscObject)(*M),"_MatIS_IS_XAIJ_irows",(PetscObject*)&irows));
      PetscCall(PetscObjectQuery((PetscObject)(*M),"_MatIS_IS_XAIJ_icols",(PetscObject*)&icols));
      if (irows) PetscCall(PetscObjectReference((PetscObject)irows));
      if (icols) PetscCall(PetscObjectReference((PetscObject)icols));
    }
    if (!irows || !icols) {
      PetscCall(ISDestroy(&icols));
      PetscCall(ISDestroy(&irows));
      goto general_assembly;
    }
    PetscCall(MatConvert(matis->A,mtype,MAT_INITIAL_MATRIX,&B));
    if (reuse != MAT_INPLACE_MATRIX) {
      PetscCall(MatCreateSubMatrix(B,irows,icols,reuse,M));
      PetscCall(PetscObjectCompose((PetscObject)(*M),"_MatIS_IS_XAIJ_irows",(PetscObject)irows));
      PetscCall(PetscObjectCompose((PetscObject)(*M),"_MatIS_IS_XAIJ_icols",(PetscObject)icols));
    } else {
      Mat C;

      PetscCall(MatCreateSubMatrix(B,irows,icols,MAT_INITIAL_MATRIX,&C));
      PetscCall(MatHeaderReplace(mat,&C));
    }
    PetscCall(MatDestroy(&B));
    PetscCall(ISDestroy(&icols));
    PetscCall(ISDestroy(&irows));
    PetscFunctionReturn(0);
  }
general_assembly:
  PetscCall(MatGetSize(mat,&rows,&cols));
  PetscCall(ISLocalToGlobalMappingGetBlockSize(matis->rmapping,&rbs));
  PetscCall(ISLocalToGlobalMappingGetBlockSize(matis->cmapping,&cbs));
  PetscCall(MatGetLocalSize(mat,&lrows,&lcols));
  PetscCall(MatGetSize(matis->A,&local_rows,&local_cols));
  PetscCall(PetscObjectBaseTypeCompare((PetscObject)matis->A,MATSEQDENSE,&isseqdense));
  PetscCall(PetscObjectBaseTypeCompare((PetscObject)matis->A,MATSEQAIJ,&isseqaij));
  PetscCall(PetscObjectBaseTypeCompare((PetscObject)matis->A,MATSEQBAIJ,&isseqbaij));
  PetscCall(PetscObjectBaseTypeCompare((PetscObject)matis->A,MATSEQSBAIJ,&isseqsbaij));
  PetscCheckFalse(!isseqdense && !isseqaij && !isseqbaij && !isseqsbaij,PETSC_COMM_SELF,PETSC_ERR_SUP,"Not for matrix type %s",((PetscObject)(matis->A))->type_name);
  if (PetscDefined (USE_DEBUG)) {
    PetscBool         lb[4],bb[4];

    lb[0] = isseqdense;
    lb[1] = isseqaij;
    lb[2] = isseqbaij;
    lb[3] = isseqsbaij;
    PetscCall(MPIU_Allreduce(lb,bb,4,MPIU_BOOL,MPI_LAND,PetscObjectComm((PetscObject)mat)));
    PetscCheckFalse(!bb[0] && !bb[1] && !bb[2] && !bb[3],PETSC_COMM_SELF,PETSC_ERR_SUP,"Local matrices must have the same type");
  }

  if (reuse != MAT_REUSE_MATRIX) {
    PetscCall(MatCreate(PetscObjectComm((PetscObject)mat),&MT));
    PetscCall(MatSetSizes(MT,lrows,lcols,rows,cols));
    PetscCall(MatSetType(MT,mtype));
    PetscCall(MatSetBlockSizes(MT,rbs,cbs));
    PetscCall(MatISSetMPIXAIJPreallocation_Private(mat,MT,PETSC_FALSE));
  } else {
    PetscInt mrbs,mcbs,mrows,mcols,mlrows,mlcols;

    /* some checks */
    MT   = *M;
    PetscCall(MatGetBlockSizes(MT,&mrbs,&mcbs));
    PetscCall(MatGetSize(MT,&mrows,&mcols));
    PetscCall(MatGetLocalSize(MT,&mlrows,&mlcols));
    PetscCheck(mrows == rows,PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Cannot reuse matrix. Wrong number of rows (%" PetscInt_FMT " != %" PetscInt_FMT ")",rows,mrows);
    PetscCheck(mcols == cols,PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Cannot reuse matrix. Wrong number of cols (%" PetscInt_FMT " != %" PetscInt_FMT ")",cols,mcols);
    PetscCheck(mlrows == lrows,PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Cannot reuse matrix. Wrong number of local rows (%" PetscInt_FMT " != %" PetscInt_FMT ")",lrows,mlrows);
    PetscCheck(mlcols == lcols,PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Cannot reuse matrix. Wrong number of local cols (%" PetscInt_FMT " != %" PetscInt_FMT ")",lcols,mlcols);
    PetscCheck(mrbs == rbs,PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Cannot reuse matrix. Wrong row block size (%" PetscInt_FMT " != %" PetscInt_FMT ")",rbs,mrbs);
    PetscCheck(mcbs == cbs,PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Cannot reuse matrix. Wrong col block size (%" PetscInt_FMT " != %" PetscInt_FMT ")",cbs,mcbs);
    PetscCall(MatZeroEntries(MT));
  }

  if (isseqsbaij || isseqbaij) {
    PetscCall(MatConvert(matis->A,MATSEQAIJ,MAT_INITIAL_MATRIX,&local_mat));
    isseqaij = PETSC_TRUE;
  } else {
    PetscCall(PetscObjectReference((PetscObject)matis->A));
    local_mat = matis->A;
  }

  /* Set values */
  PetscCall(MatSetLocalToGlobalMapping(MT,matis->rmapping,matis->cmapping));
  if (isseqdense) { /* special case for dense local matrices */
    PetscInt          i,*dummy;

    PetscCall(PetscMalloc1(PetscMax(local_rows,local_cols),&dummy));
    for (i=0;i<PetscMax(local_rows,local_cols);i++) dummy[i] = i;
    PetscCall(MatSetOption(MT,MAT_ROW_ORIENTED,PETSC_FALSE));
    PetscCall(MatDenseGetArrayRead(local_mat,&array));
    PetscCall(MatSetValuesLocal(MT,local_rows,dummy,local_cols,dummy,array,ADD_VALUES));
    PetscCall(MatDenseRestoreArrayRead(local_mat,&array));
    PetscCall(PetscFree(dummy));
  } else if (isseqaij) {
    const PetscInt *blocks;
    PetscInt       i,nvtxs,*xadj,*adjncy, nb;
    PetscBool      done;
    PetscScalar    *sarray;

    PetscCall(MatGetRowIJ(local_mat,0,PETSC_FALSE,PETSC_FALSE,&nvtxs,(const PetscInt**)&xadj,(const PetscInt**)&adjncy,&done));
    PetscCheck(done,PetscObjectComm((PetscObject)local_mat),PETSC_ERR_PLIB,"Error in MatGetRowIJ");
    PetscCall(MatSeqAIJGetArray(local_mat,&sarray));
    PetscCall(MatGetVariableBlockSizes(local_mat,&nb,&blocks));
    if (nb) { /* speed up assembly for special blocked matrices (used by BDDC) */
      PetscInt sum;

      for (i=0,sum=0;i<nb;i++) sum += blocks[i];
      if (sum == nvtxs) {
        PetscInt r;

        for (i=0,r=0;i<nb;i++) {
          PetscAssert(blocks[i] == xadj[r+1] - xadj[r],PETSC_COMM_SELF,PETSC_ERR_PLIB,"Invalid block sizes prescribed for block %" PetscInt_FMT ": expected %" PetscInt_FMT ", got %" PetscInt_FMT,i,blocks[i],xadj[r+1] - xadj[r]);
          PetscCall(MatSetValuesLocal(MT,blocks[i],adjncy+xadj[r],blocks[i],adjncy+xadj[r],sarray+xadj[r],ADD_VALUES));
          r   += blocks[i];
        }
      } else {
        for (i=0;i<nvtxs;i++) {
          PetscCall(MatSetValuesLocal(MT,1,&i,xadj[i+1]-xadj[i],adjncy+xadj[i],sarray+xadj[i],ADD_VALUES));
        }
      }
    } else {
      for (i=0;i<nvtxs;i++) {
        PetscCall(MatSetValuesLocal(MT,1,&i,xadj[i+1]-xadj[i],adjncy+xadj[i],sarray+xadj[i],ADD_VALUES));
      }
    }
    PetscCall(MatRestoreRowIJ(local_mat,0,PETSC_FALSE,PETSC_FALSE,&nvtxs,(const PetscInt**)&xadj,(const PetscInt**)&adjncy,&done));
    PetscCheck(done,PetscObjectComm((PetscObject)local_mat),PETSC_ERR_PLIB,"Error in MatRestoreRowIJ");
    PetscCall(MatSeqAIJRestoreArray(local_mat,&sarray));
  } else { /* very basic values insertion for all other matrix types */
    PetscInt i;

    for (i=0;i<local_rows;i++) {
      PetscInt       j;
      const PetscInt *local_indices_cols;

      PetscCall(MatGetRow(local_mat,i,&j,&local_indices_cols,&array));
      PetscCall(MatSetValuesLocal(MT,1,&i,j,local_indices_cols,array,ADD_VALUES));
      PetscCall(MatRestoreRow(local_mat,i,&j,&local_indices_cols,&array));
    }
  }
  PetscCall(MatAssemblyBegin(MT,MAT_FINAL_ASSEMBLY));
  PetscCall(MatDestroy(&local_mat));
  PetscCall(MatAssemblyEnd(MT,MAT_FINAL_ASSEMBLY));
  if (isseqdense) {
    PetscCall(MatSetOption(MT,MAT_ROW_ORIENTED,PETSC_TRUE));
  }
  if (reuse == MAT_INPLACE_MATRIX) {
    PetscCall(MatHeaderReplace(mat,&MT));
  } else if (reuse == MAT_INITIAL_MATRIX) {
    *M = MT;
  }
  PetscFunctionReturn(0);
}

/*@
    MatISGetMPIXAIJ - Converts MATIS matrix into a parallel AIJ format

  Input Parameters:
+  mat - the matrix (should be of type MATIS)
-  reuse - either MAT_INITIAL_MATRIX or MAT_REUSE_MATRIX

  Output Parameter:
.  newmat - the matrix in AIJ format

  Level: developer

  Notes:
    This function has been deprecated and it will be removed in future releases. Update your code to use the MatConvert() interface.

.seealso: MATIS, MatConvert()
@*/
PetscErrorCode MatISGetMPIXAIJ(Mat mat, MatReuse reuse, Mat *newmat)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidLogicalCollectiveEnum(mat,reuse,2);
  PetscValidPointer(newmat,3);
  if (reuse == MAT_REUSE_MATRIX) {
    PetscValidHeaderSpecific(*newmat,MAT_CLASSID,3);
    PetscCheckSameComm(mat,1,*newmat,3);
    PetscCheck(mat != *newmat,PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Cannot reuse the same matrix");
  }
  PetscUseMethod(mat,"MatISGetMPIXAIJ_C",(Mat,MatType,MatReuse,Mat*),(mat,MATAIJ,reuse,newmat));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDuplicate_IS(Mat mat,MatDuplicateOption op,Mat *newmat)
{
  Mat_IS         *matis = (Mat_IS*)(mat->data);
  PetscInt       rbs,cbs,m,n,M,N;
  Mat            B,localmat;

  PetscFunctionBegin;
  PetscCall(ISLocalToGlobalMappingGetBlockSize(mat->rmap->mapping,&rbs));
  PetscCall(ISLocalToGlobalMappingGetBlockSize(mat->cmap->mapping,&cbs));
  PetscCall(MatGetSize(mat,&M,&N));
  PetscCall(MatGetLocalSize(mat,&m,&n));
  PetscCall(MatCreate(PetscObjectComm((PetscObject)mat),&B));
  PetscCall(MatSetSizes(B,m,n,M,N));
  PetscCall(MatSetBlockSize(B,rbs == cbs ? rbs : 1));
  PetscCall(MatSetType(B,MATIS));
  PetscCall(MatISSetLocalMatType(B,matis->lmattype));
  PetscCall(MatSetLocalToGlobalMapping(B,mat->rmap->mapping,mat->cmap->mapping));
  PetscCall(MatDuplicate(matis->A,op,&localmat));
  PetscCall(MatSetLocalToGlobalMapping(localmat,matis->A->rmap->mapping,matis->A->cmap->mapping));
  PetscCall(MatISSetLocalMat(B,localmat));
  PetscCall(MatDestroy(&localmat));
  PetscCall(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));
  *newmat = B;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatIsHermitian_IS(Mat A,PetscReal tol,PetscBool  *flg)
{
  Mat_IS         *matis = (Mat_IS*)A->data;
  PetscBool      local_sym;

  PetscFunctionBegin;
  PetscCall(MatIsHermitian(matis->A,tol,&local_sym));
  PetscCall(MPIU_Allreduce(&local_sym,flg,1,MPIU_BOOL,MPI_LAND,PetscObjectComm((PetscObject)A)));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatIsSymmetric_IS(Mat A,PetscReal tol,PetscBool *flg)
{
  Mat_IS         *matis = (Mat_IS*)A->data;
  PetscBool      local_sym;

  PetscFunctionBegin;
  if (matis->rmapping != matis->cmapping) {
    *flg = PETSC_FALSE;
    PetscFunctionReturn(0);
  }
  PetscCall(MatIsSymmetric(matis->A,tol,&local_sym));
  PetscCall(MPIU_Allreduce(&local_sym,flg,1,MPIU_BOOL,MPI_LAND,PetscObjectComm((PetscObject)A)));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatIsStructurallySymmetric_IS(Mat A,PetscBool *flg)
{
  Mat_IS         *matis = (Mat_IS*)A->data;
  PetscBool      local_sym;

  PetscFunctionBegin;
  if (matis->rmapping != matis->cmapping) {
    *flg = PETSC_FALSE;
    PetscFunctionReturn(0);
  }
  PetscCall(MatIsStructurallySymmetric(matis->A,&local_sym));
  PetscCall(MPIU_Allreduce(&local_sym,flg,1,MPIU_BOOL,MPI_LAND,PetscObjectComm((PetscObject)A)));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDestroy_IS(Mat A)
{
  Mat_IS         *b = (Mat_IS*)A->data;

  PetscFunctionBegin;
  PetscCall(PetscFree(b->bdiag));
  PetscCall(PetscFree(b->lmattype));
  PetscCall(MatDestroy(&b->A));
  PetscCall(VecScatterDestroy(&b->cctx));
  PetscCall(VecScatterDestroy(&b->rctx));
  PetscCall(VecDestroy(&b->x));
  PetscCall(VecDestroy(&b->y));
  PetscCall(VecDestroy(&b->counter));
  PetscCall(ISDestroy(&b->getsub_ris));
  PetscCall(ISDestroy(&b->getsub_cis));
  if (b->sf != b->csf) {
    PetscCall(PetscSFDestroy(&b->csf));
    PetscCall(PetscFree2(b->csf_rootdata,b->csf_leafdata));
  } else b->csf = NULL;
  PetscCall(PetscSFDestroy(&b->sf));
  PetscCall(PetscFree2(b->sf_rootdata,b->sf_leafdata));
  PetscCall(ISLocalToGlobalMappingDestroy(&b->rmapping));
  PetscCall(ISLocalToGlobalMappingDestroy(&b->cmapping));
  PetscCall(PetscFree(A->data));
  PetscCall(PetscObjectChangeTypeName((PetscObject)A,NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatISSetLocalMatType_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatISGetLocalMat_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatISSetLocalMat_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatISGetMPIXAIJ_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatISSetPreallocation_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatISStoreL2L_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatISFixLocalEmpty_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatISGetLocalToGlobalMapping_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatConvert_is_mpiaij_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatConvert_is_mpibaij_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatConvert_is_mpisbaij_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatConvert_is_seqaij_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatConvert_is_seqbaij_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatConvert_is_seqsbaij_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatConvert_is_aij_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatSetPreallocationCOOLocal_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatSetPreallocationCOO_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatSetValuesCOO_C",NULL));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMult_IS(Mat A,Vec x,Vec y)
{
  Mat_IS         *is  = (Mat_IS*)A->data;
  PetscScalar    zero = 0.0;

  PetscFunctionBegin;
  /*  scatter the global vector x into the local work vector */
  PetscCall(VecScatterBegin(is->cctx,x,is->x,INSERT_VALUES,SCATTER_FORWARD));
  PetscCall(VecScatterEnd(is->cctx,x,is->x,INSERT_VALUES,SCATTER_FORWARD));

  /* multiply the local matrix */
  PetscCall(MatMult(is->A,is->x,is->y));

  /* scatter product back into global memory */
  PetscCall(VecSet(y,zero));
  PetscCall(VecScatterBegin(is->rctx,is->y,y,ADD_VALUES,SCATTER_REVERSE));
  PetscCall(VecScatterEnd(is->rctx,is->y,y,ADD_VALUES,SCATTER_REVERSE));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMultAdd_IS(Mat A,Vec v1,Vec v2,Vec v3)
{
  Vec            temp_vec;

  PetscFunctionBegin; /*  v3 = v2 + A * v1.*/
  if (v3 != v2) {
    PetscCall(MatMult(A,v1,v3));
    PetscCall(VecAXPY(v3,1.0,v2));
  } else {
    PetscCall(VecDuplicate(v2,&temp_vec));
    PetscCall(MatMult(A,v1,temp_vec));
    PetscCall(VecAXPY(temp_vec,1.0,v2));
    PetscCall(VecCopy(temp_vec,v3));
    PetscCall(VecDestroy(&temp_vec));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMultTranspose_IS(Mat A,Vec y,Vec x)
{
  Mat_IS         *is = (Mat_IS*)A->data;

  PetscFunctionBegin;
  /*  scatter the global vector x into the local work vector */
  PetscCall(VecScatterBegin(is->rctx,y,is->y,INSERT_VALUES,SCATTER_FORWARD));
  PetscCall(VecScatterEnd(is->rctx,y,is->y,INSERT_VALUES,SCATTER_FORWARD));

  /* multiply the local matrix */
  PetscCall(MatMultTranspose(is->A,is->y,is->x));

  /* scatter product back into global vector */
  PetscCall(VecSet(x,0));
  PetscCall(VecScatterBegin(is->cctx,is->x,x,ADD_VALUES,SCATTER_REVERSE));
  PetscCall(VecScatterEnd(is->cctx,is->x,x,ADD_VALUES,SCATTER_REVERSE));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMultTransposeAdd_IS(Mat A,Vec v1,Vec v2,Vec v3)
{
  Vec            temp_vec;

  PetscFunctionBegin; /*  v3 = v2 + A' * v1.*/
  if (v3 != v2) {
    PetscCall(MatMultTranspose(A,v1,v3));
    PetscCall(VecAXPY(v3,1.0,v2));
  } else {
    PetscCall(VecDuplicate(v2,&temp_vec));
    PetscCall(MatMultTranspose(A,v1,temp_vec));
    PetscCall(VecAXPY(temp_vec,1.0,v2));
    PetscCall(VecCopy(temp_vec,v3));
    PetscCall(VecDestroy(&temp_vec));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatView_IS(Mat A,PetscViewer viewer)
{
  Mat_IS         *a = (Mat_IS*)A->data;
  PetscViewer    sviewer;
  PetscBool      isascii,view = PETSC_TRUE;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii));
  if (isascii)  {
    PetscViewerFormat format;

    PetscCall(PetscViewerGetFormat(viewer,&format));
    if (format == PETSC_VIEWER_ASCII_INFO) view = PETSC_FALSE;
  }
  if (!view) PetscFunctionReturn(0);
  PetscCall(PetscViewerGetSubViewer(viewer,PETSC_COMM_SELF,&sviewer));
  PetscCall(MatView(a->A,sviewer));
  PetscCall(PetscViewerRestoreSubViewer(viewer,PETSC_COMM_SELF,&sviewer));
  PetscCall(PetscViewerFlush(viewer));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatInvertBlockDiagonal_IS(Mat mat,const PetscScalar **values)
{
  Mat_IS            *is = (Mat_IS*)mat->data;
  MPI_Datatype      nodeType;
  const PetscScalar *lv;
  PetscInt          bs;

  PetscFunctionBegin;
  PetscCall(MatGetBlockSize(mat,&bs));
  PetscCall(MatSetBlockSize(is->A,bs));
  PetscCall(MatInvertBlockDiagonal(is->A,&lv));
  if (!is->bdiag) {
    PetscCall(PetscMalloc1(bs*mat->rmap->n,&is->bdiag));
  }
  PetscCallMPI(MPI_Type_contiguous(bs,MPIU_SCALAR,&nodeType));
  PetscCallMPI(MPI_Type_commit(&nodeType));
  PetscCall(PetscSFReduceBegin(is->sf,nodeType,lv,is->bdiag,MPI_REPLACE));
  PetscCall(PetscSFReduceEnd(is->sf,nodeType,lv,is->bdiag,MPI_REPLACE));
  PetscCallMPI(MPI_Type_free(&nodeType));
  if (values) *values = is->bdiag;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatISSetUpScatters_Private(Mat A)
{
  Vec            cglobal,rglobal;
  IS             from;
  Mat_IS         *is = (Mat_IS*)A->data;
  PetscScalar    sum;
  const PetscInt *garray;
  PetscInt       nr,rbs,nc,cbs;
  VecType        rtype;

  PetscFunctionBegin;
  PetscCall(ISLocalToGlobalMappingGetSize(is->rmapping,&nr));
  PetscCall(ISLocalToGlobalMappingGetBlockSize(is->rmapping,&rbs));
  PetscCall(ISLocalToGlobalMappingGetSize(is->cmapping,&nc));
  PetscCall(ISLocalToGlobalMappingGetBlockSize(is->cmapping,&cbs));
  PetscCall(VecDestroy(&is->x));
  PetscCall(VecDestroy(&is->y));
  PetscCall(VecDestroy(&is->counter));
  PetscCall(VecScatterDestroy(&is->rctx));
  PetscCall(VecScatterDestroy(&is->cctx));
  PetscCall(MatCreateVecs(is->A,&is->x,&is->y));
  PetscCall(VecBindToCPU(is->y,PETSC_TRUE));
  PetscCall(VecGetRootType_Private(is->y,&rtype));
  PetscCall(PetscFree(A->defaultvectype));
  PetscCall(PetscStrallocpy(rtype,&A->defaultvectype));
  PetscCall(MatCreateVecs(A,&cglobal,&rglobal));
  PetscCall(VecBindToCPU(rglobal,PETSC_TRUE));
  PetscCall(ISLocalToGlobalMappingGetBlockIndices(is->rmapping,&garray));
  PetscCall(ISCreateBlock(PetscObjectComm((PetscObject)A),rbs,nr/rbs,garray,PETSC_USE_POINTER,&from));
  PetscCall(VecScatterCreate(rglobal,from,is->y,NULL,&is->rctx));
  PetscCall(ISLocalToGlobalMappingRestoreBlockIndices(is->rmapping,&garray));
  PetscCall(ISDestroy(&from));
  if (is->rmapping != is->cmapping) {
    PetscCall(ISLocalToGlobalMappingGetBlockIndices(is->cmapping,&garray));
    PetscCall(ISCreateBlock(PetscObjectComm((PetscObject)A),cbs,nc/cbs,garray,PETSC_USE_POINTER,&from));
    PetscCall(VecScatterCreate(cglobal,from,is->x,NULL,&is->cctx));
    PetscCall(ISLocalToGlobalMappingRestoreBlockIndices(is->cmapping,&garray));
    PetscCall(ISDestroy(&from));
  } else {
    PetscCall(PetscObjectReference((PetscObject)is->rctx));
    is->cctx = is->rctx;
  }
  PetscCall(VecDestroy(&cglobal));

  /* interface counter vector (local) */
  PetscCall(VecDuplicate(is->y,&is->counter));
  PetscCall(VecBindToCPU(is->counter,PETSC_TRUE));
  PetscCall(VecSet(is->y,1.));
  PetscCall(VecScatterBegin(is->rctx,is->y,rglobal,ADD_VALUES,SCATTER_REVERSE));
  PetscCall(VecScatterEnd(is->rctx,is->y,rglobal,ADD_VALUES,SCATTER_REVERSE));
  PetscCall(VecScatterBegin(is->rctx,rglobal,is->counter,INSERT_VALUES,SCATTER_FORWARD));
  PetscCall(VecScatterEnd(is->rctx,rglobal,is->counter,INSERT_VALUES,SCATTER_FORWARD));
  PetscCall(VecBindToCPU(is->y,PETSC_FALSE));
  PetscCall(VecBindToCPU(is->counter,PETSC_FALSE));

  /* special functions for block-diagonal matrices */
  PetscCall(VecSum(rglobal,&sum));
  A->ops->invertblockdiagonal = NULL;
  if ((PetscInt)(PetscRealPart(sum)) == A->rmap->N && A->rmap->N == A->cmap->N && is->rmapping == is->cmapping) A->ops->invertblockdiagonal = MatInvertBlockDiagonal_IS;
  PetscCall(VecDestroy(&rglobal));

  /* setup SF for general purpose shared indices based communications */
  PetscCall(MatISSetUpSF_IS(A));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatISFilterL2GMap(Mat A, ISLocalToGlobalMapping map, ISLocalToGlobalMapping *nmap, ISLocalToGlobalMapping *lmap)
{
  IS                         is;
  ISLocalToGlobalMappingType l2gtype;
  const PetscInt             *idxs;
  PetscHSetI                 ht;
  PetscInt                   *nidxs;
  PetscInt                   i,n,bs,c;
  PetscBool                  flg[] = {PETSC_FALSE,PETSC_FALSE};

  PetscFunctionBegin;
  PetscCall(ISLocalToGlobalMappingGetSize(map,&n));
  PetscCall(ISLocalToGlobalMappingGetBlockSize(map,&bs));
  PetscCall(ISLocalToGlobalMappingGetBlockIndices(map,&idxs));
  PetscCall(PetscHSetICreate(&ht));
  PetscCall(PetscMalloc1(n/bs,&nidxs));
  for (i=0,c=0;i<n/bs;i++) {
    PetscBool missing;
    if (idxs[i] < 0) { flg[0] = PETSC_TRUE; continue; }
    PetscCall(PetscHSetIQueryAdd(ht,idxs[i],&missing));
    if (!missing) flg[1] = PETSC_TRUE;
    else nidxs[c++] = idxs[i];
  }
  PetscCall(PetscHSetIDestroy(&ht));
  PetscCall(MPIU_Allreduce(MPI_IN_PLACE,flg,2,MPIU_BOOL,MPI_LOR,PetscObjectComm((PetscObject)A)));
  if (!flg[0] && !flg[1]) { /* Entries are all non negative and unique */
    *nmap = NULL;
    *lmap = NULL;
    PetscCall(PetscFree(nidxs));
    PetscCall(ISLocalToGlobalMappingRestoreBlockIndices(map,&idxs));
    PetscFunctionReturn(0);
  }

  /* New l2g map without negative or repeated indices */
  PetscCall(ISCreateBlock(PetscObjectComm((PetscObject)A),bs,c,nidxs,PETSC_USE_POINTER,&is));
  PetscCall(ISLocalToGlobalMappingCreateIS(is,nmap));
  PetscCall(ISDestroy(&is));
  PetscCall(ISLocalToGlobalMappingGetType(map,&l2gtype));
  PetscCall(ISLocalToGlobalMappingSetType(*nmap,l2gtype));

  /* New local l2g map for repeated indices */
  PetscCall(ISGlobalToLocalMappingApplyBlock(*nmap,IS_GTOLM_MASK,n/bs,idxs,NULL,nidxs));
  PetscCall(ISCreateBlock(PETSC_COMM_SELF,bs,n/bs,nidxs,PETSC_USE_POINTER,&is));
  PetscCall(ISLocalToGlobalMappingCreateIS(is,lmap));
  PetscCall(ISDestroy(&is));

  PetscCall(PetscFree(nidxs));
  PetscCall(ISLocalToGlobalMappingRestoreBlockIndices(map,&idxs));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSetLocalToGlobalMapping_IS(Mat A,ISLocalToGlobalMapping rmapping,ISLocalToGlobalMapping cmapping)
{
  Mat_IS                 *is = (Mat_IS*)A->data;
  ISLocalToGlobalMapping localrmapping = NULL, localcmapping = NULL;
  PetscBool              cong, freem[] = { PETSC_FALSE, PETSC_FALSE };
  PetscInt               nr,rbs,nc,cbs;

  PetscFunctionBegin;
  if (rmapping) PetscCheckSameComm(A,1,rmapping,2);
  if (cmapping) PetscCheckSameComm(A,1,cmapping,3);

  PetscCall(ISLocalToGlobalMappingDestroy(&is->rmapping));
  PetscCall(ISLocalToGlobalMappingDestroy(&is->cmapping));
  PetscCall(PetscLayoutSetUp(A->rmap));
  PetscCall(PetscLayoutSetUp(A->cmap));
  PetscCall(MatHasCongruentLayouts(A,&cong));

  /* If NULL, local space matches global space */
  if (!rmapping) {
    IS is;

    PetscCall(ISCreateStride(PetscObjectComm((PetscObject)A),A->rmap->N,0,1,&is));
    PetscCall(ISLocalToGlobalMappingCreateIS(is,&rmapping));
    if (A->rmap->bs > 0) PetscCall(ISLocalToGlobalMappingSetBlockSize(rmapping,A->rmap->bs));
    PetscCall(ISDestroy(&is));
    freem[0] = PETSC_TRUE;
    if (!cmapping && cong && A->rmap->bs == A->cmap->bs) cmapping = rmapping;
  } else if (!is->islocalref) { /* check if the l2g map has negative or repeated entries */
    PetscCall(MatISFilterL2GMap(A,rmapping,&is->rmapping,&localrmapping));
    if (rmapping == cmapping) {
      PetscCall(PetscObjectReference((PetscObject)is->rmapping));
      is->cmapping = is->rmapping;
      PetscCall(PetscObjectReference((PetscObject)localrmapping));
      localcmapping = localrmapping;
    }
  }
  if (!cmapping) {
    IS is;

    PetscCall(ISCreateStride(PetscObjectComm((PetscObject)A),A->cmap->N,0,1,&is));
    PetscCall(ISLocalToGlobalMappingCreateIS(is,&cmapping));
    if (A->cmap->bs > 0) PetscCall(ISLocalToGlobalMappingSetBlockSize(cmapping,A->cmap->bs));
    PetscCall(ISDestroy(&is));
    freem[1] = PETSC_TRUE;
  } else if (cmapping != rmapping && !is->islocalref) { /* check if the l2g map has negative or repeated entries */
    PetscCall(MatISFilterL2GMap(A,cmapping,&is->cmapping,&localcmapping));
  }
  if (!is->rmapping) {
    PetscCall(PetscObjectReference((PetscObject)rmapping));
    is->rmapping = rmapping;
  }
  if (!is->cmapping) {
    PetscCall(PetscObjectReference((PetscObject)cmapping));
    is->cmapping = cmapping;
  }

  /* Clean up */
  PetscCall(MatDestroy(&is->A));
  if (is->csf != is->sf) {
    PetscCall(PetscSFDestroy(&is->csf));
    PetscCall(PetscFree2(is->csf_rootdata,is->csf_leafdata));
  } else is->csf = NULL;
  PetscCall(PetscSFDestroy(&is->sf));
  PetscCall(PetscFree2(is->sf_rootdata,is->sf_leafdata));
  PetscCall(PetscFree(is->bdiag));

  /* check if the two mappings are actually the same for square matrices since MATIS has some optimization for this case
     (DOLFIN passes 2 different objects) */
  PetscCall(ISLocalToGlobalMappingGetSize(is->rmapping,&nr));
  PetscCall(ISLocalToGlobalMappingGetBlockSize(is->rmapping,&rbs));
  PetscCall(ISLocalToGlobalMappingGetSize(is->cmapping,&nc));
  PetscCall(ISLocalToGlobalMappingGetBlockSize(is->cmapping,&cbs));
  if (is->rmapping != is->cmapping && cong) {
    PetscBool same = PETSC_FALSE;
    if (nr == nc && cbs == rbs) {
      const PetscInt *idxs1,*idxs2;

      PetscCall(ISLocalToGlobalMappingGetBlockIndices(is->rmapping,&idxs1));
      PetscCall(ISLocalToGlobalMappingGetBlockIndices(is->cmapping,&idxs2));
      PetscCall(PetscArraycmp(idxs1,idxs2,nr/rbs,&same));
      PetscCall(ISLocalToGlobalMappingRestoreBlockIndices(is->rmapping,&idxs1));
      PetscCall(ISLocalToGlobalMappingRestoreBlockIndices(is->cmapping,&idxs2));
    }
    PetscCall(MPIU_Allreduce(MPI_IN_PLACE,&same,1,MPIU_BOOL,MPI_LAND,PetscObjectComm((PetscObject)A)));
    if (same) {
      PetscCall(ISLocalToGlobalMappingDestroy(&is->cmapping));
      PetscCall(PetscObjectReference((PetscObject)is->rmapping));
      is->cmapping = is->rmapping;
    }
  }
  PetscCall(PetscLayoutSetBlockSize(A->rmap,rbs));
  PetscCall(PetscLayoutSetBlockSize(A->cmap,cbs));
  /* Pass the user defined maps to the layout */
  PetscCall(PetscLayoutSetISLocalToGlobalMapping(A->rmap,rmapping));
  PetscCall(PetscLayoutSetISLocalToGlobalMapping(A->cmap,cmapping));
  if (freem[0]) PetscCall(ISLocalToGlobalMappingDestroy(&rmapping));
  if (freem[1]) PetscCall(ISLocalToGlobalMappingDestroy(&cmapping));

  /* Create the local matrix A */
  PetscCall(MatCreate(PETSC_COMM_SELF,&is->A));
  PetscCall(MatSetType(is->A,is->lmattype));
  PetscCall(MatSetSizes(is->A,nr,nc,nr,nc));
  PetscCall(MatSetBlockSizes(is->A,rbs,cbs));
  PetscCall(MatSetOptionsPrefix(is->A,"is_"));
  PetscCall(MatAppendOptionsPrefix(is->A,((PetscObject)A)->prefix));
  PetscCall(PetscLayoutSetUp(is->A->rmap));
  PetscCall(PetscLayoutSetUp(is->A->cmap));
  PetscCall(MatSetLocalToGlobalMapping(is->A,localrmapping,localcmapping));
  PetscCall(ISLocalToGlobalMappingDestroy(&localrmapping));
  PetscCall(ISLocalToGlobalMappingDestroy(&localcmapping));

  /* setup scatters and local vectors for MatMult */
  if (!is->islocalref) PetscCall(MatISSetUpScatters_Private(A));
  A->preallocated = PETSC_TRUE;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSetUp_IS(Mat A)
{
  ISLocalToGlobalMapping rmap, cmap;

  PetscFunctionBegin;
  PetscCall(MatGetLocalToGlobalMapping(A,&rmap,&cmap));
  if (!rmap && !cmap) {
    PetscCall(MatSetLocalToGlobalMapping(A,NULL,NULL));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSetValues_IS(Mat mat, PetscInt m,const PetscInt *rows, PetscInt n,const PetscInt *cols, const PetscScalar *values, InsertMode addv)
{
  Mat_IS         *is = (Mat_IS*)mat->data;
  PetscInt       rows_l[MATIS_MAX_ENTRIES_INSERTION],cols_l[MATIS_MAX_ENTRIES_INSERTION];

  PetscFunctionBegin;
  PetscCall(ISGlobalToLocalMappingApply(is->rmapping,IS_GTOLM_MASK,m,rows,&m,rows_l));
  if (m != n || rows != cols || is->cmapping != is->rmapping) {
    PetscCall(ISGlobalToLocalMappingApply(is->cmapping,IS_GTOLM_MASK,n,cols,&n,cols_l));
    PetscCall(MatSetValues(is->A,m,rows_l,n,cols_l,values,addv));
  } else {
    PetscCall(MatSetValues(is->A,m,rows_l,m,rows_l,values,addv));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSetValuesBlocked_IS(Mat mat, PetscInt m,const PetscInt *rows, PetscInt n,const PetscInt *cols, const PetscScalar *values, InsertMode addv)
{
  Mat_IS         *is = (Mat_IS*)mat->data;
  PetscInt       rows_l[MATIS_MAX_ENTRIES_INSERTION],cols_l[MATIS_MAX_ENTRIES_INSERTION];

  PetscFunctionBegin;
  PetscCall(ISGlobalToLocalMappingApplyBlock(is->rmapping,IS_GTOLM_MASK,m,rows,&m,rows_l));
  if (m != n || rows != cols || is->cmapping != is->rmapping) {
    PetscCall(ISGlobalToLocalMappingApply(is->cmapping,IS_GTOLM_MASK,n,cols,&n,cols_l));
    PetscCall(MatSetValuesBlocked(is->A,m,rows_l,n,cols_l,values,addv));
  } else {
    PetscCall(MatSetValuesBlocked(is->A,m,rows_l,n,rows_l,values,addv));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSetValuesLocal_IS(Mat A,PetscInt m,const PetscInt *rows, PetscInt n,const PetscInt *cols,const PetscScalar *values,InsertMode addv)
{
  Mat_IS         *is = (Mat_IS*)A->data;

  PetscFunctionBegin;
  if (is->A->rmap->mapping || is->A->cmap->mapping) {
    PetscCall(MatSetValuesLocal(is->A,m,rows,n,cols,values,addv));
  } else {
    PetscCall(MatSetValues(is->A,m,rows,n,cols,values,addv));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSetValuesBlockedLocal_IS(Mat A,PetscInt m,const PetscInt *rows, PetscInt n,const PetscInt *cols,const PetscScalar *values,InsertMode addv)
{
  Mat_IS         *is = (Mat_IS*)A->data;

  PetscFunctionBegin;
  if (is->A->rmap->mapping || is->A->cmap->mapping) {
    PetscCall(MatSetValuesBlockedLocal(is->A,m,rows,n,cols,values,addv));
  } else {
    PetscCall(MatSetValuesBlocked(is->A,m,rows,n,cols,values,addv));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatISZeroRowsColumnsLocal_Private(Mat A,PetscInt n,const PetscInt rows[],PetscScalar diag,PetscBool columns)
{
  Mat_IS         *is = (Mat_IS*)A->data;

  PetscFunctionBegin;
  if (!n) {
    is->pure_neumann = PETSC_TRUE;
  } else {
    PetscInt i;
    is->pure_neumann = PETSC_FALSE;

    if (columns) {
      PetscCall(MatZeroRowsColumns(is->A,n,rows,diag,NULL,NULL));
    } else {
      PetscCall(MatZeroRows(is->A,n,rows,diag,NULL,NULL));
    }
    if (diag != 0.) {
      const PetscScalar *array;
      PetscCall(VecGetArrayRead(is->counter,&array));
      for (i=0; i<n; i++) {
        PetscCall(MatSetValue(is->A,rows[i],rows[i],diag/(array[rows[i]]),INSERT_VALUES));
      }
      PetscCall(VecRestoreArrayRead(is->counter,&array));
    }
    PetscCall(MatAssemblyBegin(is->A,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(is->A,MAT_FINAL_ASSEMBLY));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatZeroRowsColumns_Private_IS(Mat A,PetscInt n,const PetscInt rows[],PetscScalar diag,Vec x,Vec b,PetscBool columns)
{
  Mat_IS         *matis = (Mat_IS*)A->data;
  PetscInt       nr,nl,len,i;
  PetscInt       *lrows;

  PetscFunctionBegin;
  if (PetscUnlikelyDebug(columns || diag != 0. || (x && b))) {
    PetscBool cong;

    PetscCall(PetscLayoutCompare(A->rmap,A->cmap,&cong));
    cong = (PetscBool)(cong && matis->sf == matis->csf);
    PetscCheck(cong || !columns,PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Columns can be zeroed if and only if A->rmap and A->cmap are congruent and the l2g maps are the same for MATIS");
    PetscCheckFalse(!cong && diag != 0.,PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Nonzero diagonal value supported if and only if A->rmap and A->cmap are congruent and the l2g maps are the same for MATIS");
    PetscCheckFalse(!cong && x && b,PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"A->rmap and A->cmap need to be congruent, and the l2g maps be the same");
  }
  /* get locally owned rows */
  PetscCall(PetscLayoutMapLocal(A->rmap,n,rows,&len,&lrows,NULL));
  /* fix right hand side if needed */
  if (x && b) {
    const PetscScalar *xx;
    PetscScalar       *bb;

    PetscCall(VecGetArrayRead(x, &xx));
    PetscCall(VecGetArray(b, &bb));
    for (i=0;i<len;++i) bb[lrows[i]] = diag*xx[lrows[i]];
    PetscCall(VecRestoreArrayRead(x, &xx));
    PetscCall(VecRestoreArray(b, &bb));
  }
  /* get rows associated to the local matrices */
  PetscCall(MatGetSize(matis->A,&nl,NULL));
  PetscCall(PetscArrayzero(matis->sf_leafdata,nl));
  PetscCall(PetscArrayzero(matis->sf_rootdata,A->rmap->n));
  for (i=0;i<len;i++) matis->sf_rootdata[lrows[i]] = 1;
  PetscCall(PetscFree(lrows));
  PetscCall(PetscSFBcastBegin(matis->sf,MPIU_INT,matis->sf_rootdata,matis->sf_leafdata,MPI_REPLACE));
  PetscCall(PetscSFBcastEnd(matis->sf,MPIU_INT,matis->sf_rootdata,matis->sf_leafdata,MPI_REPLACE));
  PetscCall(PetscMalloc1(nl,&lrows));
  for (i=0,nr=0;i<nl;i++) if (matis->sf_leafdata[i]) lrows[nr++] = i;
  PetscCall(MatISZeroRowsColumnsLocal_Private(A,nr,lrows,diag,columns));
  PetscCall(PetscFree(lrows));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatZeroRows_IS(Mat A,PetscInt n,const PetscInt rows[],PetscScalar diag,Vec x,Vec b)
{
  PetscFunctionBegin;
  PetscCall(MatZeroRowsColumns_Private_IS(A,n,rows,diag,x,b,PETSC_FALSE));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatZeroRowsColumns_IS(Mat A,PetscInt n,const PetscInt rows[],PetscScalar diag,Vec x,Vec b)
{
  PetscFunctionBegin;
  PetscCall(MatZeroRowsColumns_Private_IS(A,n,rows,diag,x,b,PETSC_TRUE));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatAssemblyBegin_IS(Mat A,MatAssemblyType type)
{
  Mat_IS         *is = (Mat_IS*)A->data;

  PetscFunctionBegin;
  PetscCall(MatAssemblyBegin(is->A,type));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatAssemblyEnd_IS(Mat A,MatAssemblyType type)
{
  Mat_IS         *is = (Mat_IS*)A->data;

  PetscFunctionBegin;
  PetscCall(MatAssemblyEnd(is->A,type));
  /* fix for local empty rows/cols */
  if (is->locempty && type == MAT_FINAL_ASSEMBLY) {
    Mat                    newlA;
    ISLocalToGlobalMapping rl2g,cl2g;
    IS                     nzr,nzc;
    PetscInt               nr,nc,nnzr,nnzc;
    PetscBool              lnewl2g,newl2g;

    PetscCall(MatGetSize(is->A,&nr,&nc));
    PetscCall(MatFindNonzeroRowsOrCols_Basic(is->A,PETSC_FALSE,PETSC_SMALL,&nzr));
    if (!nzr) {
      PetscCall(ISCreateStride(PetscObjectComm((PetscObject)is->A),nr,0,1,&nzr));
    }
    PetscCall(MatFindNonzeroRowsOrCols_Basic(is->A,PETSC_TRUE,PETSC_SMALL,&nzc));
    if (!nzc) {
      PetscCall(ISCreateStride(PetscObjectComm((PetscObject)is->A),nc,0,1,&nzc));
    }
    PetscCall(ISGetSize(nzr,&nnzr));
    PetscCall(ISGetSize(nzc,&nnzc));
    if (nnzr != nr || nnzc != nc) { /* need new global l2g map */
      lnewl2g = PETSC_TRUE;
      PetscCallMPI(MPI_Allreduce(&lnewl2g,&newl2g,1,MPIU_BOOL,MPI_LOR,PetscObjectComm((PetscObject)A)));

      /* extract valid submatrix */
      PetscCall(MatCreateSubMatrix(is->A,nzr,nzc,MAT_INITIAL_MATRIX,&newlA));
    } else { /* local matrix fully populated */
      lnewl2g = PETSC_FALSE;
      PetscCallMPI(MPI_Allreduce(&lnewl2g,&newl2g,1,MPIU_BOOL,MPI_LOR,PetscObjectComm((PetscObject)A)));
      PetscCall(PetscObjectReference((PetscObject)is->A));
      newlA   = is->A;
    }

    /* attach new global l2g map if needed */
    if (newl2g) {
      IS              zr,zc;
      const  PetscInt *ridxs,*cidxs,*zridxs,*zcidxs;
      PetscInt        *nidxs,i;

      PetscCall(ISComplement(nzr,0,nr,&zr));
      PetscCall(ISComplement(nzc,0,nc,&zc));
      PetscCall(PetscMalloc1(PetscMax(nr,nc),&nidxs));
      PetscCall(ISLocalToGlobalMappingGetIndices(is->rmapping,&ridxs));
      PetscCall(ISLocalToGlobalMappingGetIndices(is->cmapping,&cidxs));
      PetscCall(ISGetIndices(zr,&zridxs));
      PetscCall(ISGetIndices(zc,&zcidxs));
      PetscCall(ISGetLocalSize(zr,&nnzr));
      PetscCall(ISGetLocalSize(zc,&nnzc));

      PetscCall(PetscArraycpy(nidxs,ridxs,nr));
      for (i = 0; i < nnzr; i++) nidxs[zridxs[i]] = -1;
      PetscCall(ISLocalToGlobalMappingCreate(PetscObjectComm((PetscObject)A),1,nr,nidxs,PETSC_COPY_VALUES,&rl2g));
      PetscCall(PetscArraycpy(nidxs,cidxs,nc));
      for (i = 0; i < nnzc; i++) nidxs[zcidxs[i]] = -1;
      PetscCall(ISLocalToGlobalMappingCreate(PetscObjectComm((PetscObject)A),1,nc,nidxs,PETSC_COPY_VALUES,&cl2g));

      PetscCall(ISRestoreIndices(zr,&zridxs));
      PetscCall(ISRestoreIndices(zc,&zcidxs));
      PetscCall(ISLocalToGlobalMappingRestoreIndices(is->rmapping,&ridxs));
      PetscCall(ISLocalToGlobalMappingRestoreIndices(is->cmapping,&cidxs));
      PetscCall(ISDestroy(&nzr));
      PetscCall(ISDestroy(&nzc));
      PetscCall(ISDestroy(&zr));
      PetscCall(ISDestroy(&zc));
      PetscCall(PetscFree(nidxs));
      PetscCall(MatSetLocalToGlobalMapping(A,rl2g,cl2g));
      PetscCall(ISLocalToGlobalMappingDestroy(&rl2g));
      PetscCall(ISLocalToGlobalMappingDestroy(&cl2g));
    }
    PetscCall(MatISSetLocalMat(A,newlA));
    PetscCall(MatDestroy(&newlA));
    PetscCall(ISDestroy(&nzr));
    PetscCall(ISDestroy(&nzc));
    is->locempty = PETSC_FALSE;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatISGetLocalMat_IS(Mat mat,Mat *local)
{
  Mat_IS *is = (Mat_IS*)mat->data;

  PetscFunctionBegin;
  *local = is->A;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatISRestoreLocalMat_IS(Mat mat,Mat *local)
{
  PetscFunctionBegin;
  *local = NULL;
  PetscFunctionReturn(0);
}

/*@
    MatISGetLocalMat - Gets the local matrix stored inside a MATIS matrix.

  Input Parameter:
.  mat - the matrix

  Output Parameter:
.  local - the local matrix

  Level: advanced

  Notes:
    This can be called if you have precomputed the nonzero structure of the
  matrix and want to provide it to the inner matrix object to improve the performance
  of the MatSetValues() operation.

  Call MatISRestoreLocalMat() when finished with the local matrix.

.seealso: MATIS
@*/
PetscErrorCode MatISGetLocalMat(Mat mat,Mat *local)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidPointer(local,2);
  PetscUseMethod(mat,"MatISGetLocalMat_C",(Mat,Mat*),(mat,local));
  PetscFunctionReturn(0);
}

/*@
    MatISRestoreLocalMat - Restores the local matrix obtained with MatISGetLocalMat()

  Input Parameter:
.  mat - the matrix

  Output Parameter:
.  local - the local matrix

  Level: advanced

.seealso: MATIS
@*/
PetscErrorCode MatISRestoreLocalMat(Mat mat,Mat *local)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidPointer(local,2);
  PetscUseMethod(mat,"MatISRestoreLocalMat_C",(Mat,Mat*),(mat,local));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatISSetLocalMatType_IS(Mat mat,MatType mtype)
{
  Mat_IS         *is = (Mat_IS*)mat->data;

  PetscFunctionBegin;
  if (is->A) {
    PetscCall(MatSetType(is->A,mtype));
  }
  PetscCall(PetscFree(is->lmattype));
  PetscCall(PetscStrallocpy(mtype,&is->lmattype));
  PetscFunctionReturn(0);
}

/*@
    MatISSetLocalMatType - Specifies the type of local matrix

  Input Parameters:
+  mat - the matrix
-  mtype - the local matrix type

  Output Parameter:

  Level: advanced

.seealso: MATIS, MatSetType(), MatType
@*/
PetscErrorCode MatISSetLocalMatType(Mat mat,MatType mtype)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscUseMethod(mat,"MatISSetLocalMatType_C",(Mat,MatType),(mat,mtype));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatISSetLocalMat_IS(Mat mat,Mat local)
{
  Mat_IS         *is = (Mat_IS*)mat->data;
  PetscInt       nrows,ncols,orows,ocols;
  MatType        mtype,otype;
  PetscBool      sametype = PETSC_TRUE;

  PetscFunctionBegin;
  if (is->A && !is->islocalref) {
    PetscCall(MatGetSize(is->A,&orows,&ocols));
    PetscCall(MatGetSize(local,&nrows,&ncols));
    PetscCheckFalse(orows != nrows || ocols != ncols,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Local MATIS matrix should be of size %" PetscInt_FMT "x%" PetscInt_FMT " (you passed a %" PetscInt_FMT "x%" PetscInt_FMT " matrix)",orows,ocols,nrows,ncols);
    PetscCall(MatGetType(local,&mtype));
    PetscCall(MatGetType(is->A,&otype));
    PetscCall(PetscStrcmp(mtype,otype,&sametype));
  }
  PetscCall(PetscObjectReference((PetscObject)local));
  PetscCall(MatDestroy(&is->A));
  is->A = local;
  PetscCall(MatGetType(is->A,&mtype));
  PetscCall(MatISSetLocalMatType(mat,mtype));
  if (!sametype && !is->islocalref) {
    PetscCall(MatISSetUpScatters_Private(mat));
  }
  PetscFunctionReturn(0);
}

/*@
    MatISSetLocalMat - Replace the local matrix stored inside a MATIS object.

  Collective on Mat

  Input Parameters:
+  mat - the matrix
-  local - the local matrix

  Output Parameter:

  Level: advanced

  Notes:
    This can be called if you have precomputed the local matrix and
  want to provide it to the matrix object MATIS.

.seealso: MATIS
@*/
PetscErrorCode MatISSetLocalMat(Mat mat,Mat local)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidHeaderSpecific(local,MAT_CLASSID,2);
  PetscUseMethod(mat,"MatISSetLocalMat_C",(Mat,Mat),(mat,local));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatZeroEntries_IS(Mat A)
{
  Mat_IS         *a = (Mat_IS*)A->data;

  PetscFunctionBegin;
  PetscCall(MatZeroEntries(a->A));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatScale_IS(Mat A,PetscScalar a)
{
  Mat_IS         *is = (Mat_IS*)A->data;

  PetscFunctionBegin;
  PetscCall(MatScale(is->A,a));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatGetDiagonal_IS(Mat A, Vec v)
{
  Mat_IS         *is = (Mat_IS*)A->data;

  PetscFunctionBegin;
  /* get diagonal of the local matrix */
  PetscCall(MatGetDiagonal(is->A,is->y));

  /* scatter diagonal back into global vector */
  PetscCall(VecSet(v,0));
  PetscCall(VecScatterBegin(is->rctx,is->y,v,ADD_VALUES,SCATTER_REVERSE));
  PetscCall(VecScatterEnd(is->rctx,is->y,v,ADD_VALUES,SCATTER_REVERSE));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSetOption_IS(Mat A,MatOption op,PetscBool flg)
{
  Mat_IS         *a = (Mat_IS*)A->data;

  PetscFunctionBegin;
  PetscCall(MatSetOption(a->A,op,flg));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatAXPY_IS(Mat Y,PetscScalar a,Mat X,MatStructure str)
{
  Mat_IS         *y = (Mat_IS*)Y->data;
  Mat_IS         *x;

  PetscFunctionBegin;
  if (PetscDefined(USE_DEBUG)) {
    PetscBool      ismatis;
    PetscCall(PetscObjectTypeCompare((PetscObject)X,MATIS,&ismatis));
    PetscCheck(ismatis,PetscObjectComm((PetscObject)Y),PETSC_ERR_SUP,"Cannot call MatAXPY(Y,a,X,str) with X not of type MATIS");
  }
  x = (Mat_IS*)X->data;
  PetscCall(MatAXPY(y->A,a,x->A,str));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatGetLocalSubMatrix_IS(Mat A,IS row,IS col,Mat *submat)
{
  Mat                    lA;
  Mat_IS                 *matis = (Mat_IS*)(A->data);
  ISLocalToGlobalMapping rl2g,cl2g;
  IS                     is;
  const PetscInt         *rg,*rl;
  PetscInt               nrg;
  PetscInt               N,M,nrl,i,*idxs;

  PetscFunctionBegin;
  PetscCall(ISLocalToGlobalMappingGetIndices(A->rmap->mapping,&rg));
  PetscCall(ISGetLocalSize(row,&nrl));
  PetscCall(ISGetIndices(row,&rl));
  PetscCall(ISLocalToGlobalMappingGetSize(A->rmap->mapping,&nrg));
  if (PetscDefined(USE_DEBUG)) {
    for (i=0; i<nrl; i++) PetscCheck(rl[i]<nrg,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Local row index %" PetscInt_FMT " -> %" PetscInt_FMT " greater then maximum possible %" PetscInt_FMT,i,rl[i],nrg);
  }
  PetscCall(PetscMalloc1(nrg,&idxs));
  /* map from [0,nrl) to row */
  for (i=0;i<nrl;i++) idxs[i] = rl[i];
  for (i=nrl;i<nrg;i++) idxs[i] = -1;
  PetscCall(ISRestoreIndices(row,&rl));
  PetscCall(ISLocalToGlobalMappingRestoreIndices(A->rmap->mapping,&rg));
  PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)A),nrg,idxs,PETSC_OWN_POINTER,&is));
  PetscCall(ISLocalToGlobalMappingCreateIS(is,&rl2g));
  PetscCall(ISDestroy(&is));
  /* compute new l2g map for columns */
  if (col != row || matis->rmapping != matis->cmapping || matis->A->rmap->mapping != matis->A->cmap->mapping) {
    const PetscInt *cg,*cl;
    PetscInt       ncg;
    PetscInt       ncl;

    PetscCall(ISLocalToGlobalMappingGetIndices(A->cmap->mapping,&cg));
    PetscCall(ISGetLocalSize(col,&ncl));
    PetscCall(ISGetIndices(col,&cl));
    PetscCall(ISLocalToGlobalMappingGetSize(A->cmap->mapping,&ncg));
    if (PetscDefined(USE_DEBUG)) {
      for (i=0; i<ncl; i++) PetscCheck(cl[i]<ncg,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Local column index %" PetscInt_FMT " -> %" PetscInt_FMT " greater then maximum possible %" PetscInt_FMT,i,cl[i],ncg);
    }
    PetscCall(PetscMalloc1(ncg,&idxs));
    /* map from [0,ncl) to col */
    for (i=0;i<ncl;i++) idxs[i] = cl[i];
    for (i=ncl;i<ncg;i++) idxs[i] = -1;
    PetscCall(ISRestoreIndices(col,&cl));
    PetscCall(ISLocalToGlobalMappingRestoreIndices(A->cmap->mapping,&cg));
    PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)A),ncg,idxs,PETSC_OWN_POINTER,&is));
    PetscCall(ISLocalToGlobalMappingCreateIS(is,&cl2g));
    PetscCall(ISDestroy(&is));
  } else {
    PetscCall(PetscObjectReference((PetscObject)rl2g));
    cl2g = rl2g;
  }
  /* create the MATIS submatrix */
  PetscCall(MatGetSize(A,&M,&N));
  PetscCall(MatCreate(PetscObjectComm((PetscObject)A),submat));
  PetscCall(MatSetSizes(*submat,PETSC_DECIDE,PETSC_DECIDE,M,N));
  PetscCall(MatSetType(*submat,MATIS));
  matis = (Mat_IS*)((*submat)->data);
  matis->islocalref = PETSC_TRUE;
  PetscCall(MatSetLocalToGlobalMapping(*submat,rl2g,cl2g));
  PetscCall(MatISGetLocalMat(A,&lA));
  PetscCall(MatISSetLocalMat(*submat,lA));
  PetscCall(MatSetUp(*submat));
  PetscCall(MatAssemblyBegin(*submat,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*submat,MAT_FINAL_ASSEMBLY));
  PetscCall(ISLocalToGlobalMappingDestroy(&rl2g));
  PetscCall(ISLocalToGlobalMappingDestroy(&cl2g));

  /* remove unsupported ops */
  PetscCall(PetscMemzero((*submat)->ops,sizeof(struct _MatOps)));
  (*submat)->ops->destroy               = MatDestroy_IS;
  (*submat)->ops->setvalueslocal        = MatSetValuesLocal_SubMat_IS;
  (*submat)->ops->setvaluesblockedlocal = MatSetValuesBlockedLocal_SubMat_IS;
  (*submat)->ops->assemblybegin         = MatAssemblyBegin_IS;
  (*submat)->ops->assemblyend           = MatAssemblyEnd_IS;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSetFromOptions_IS(PetscOptionItems *PetscOptionsObject, Mat A)
{
  Mat_IS         *a = (Mat_IS*)A->data;
  char           type[256];
  PetscBool      flg;

  PetscFunctionBegin;
  PetscCall(PetscOptionsHead(PetscOptionsObject,"MATIS options"));
  PetscCall(PetscOptionsBool("-matis_fixempty","Fix local matrices in case of empty local rows/columns","MatISFixLocalEmpty",a->locempty,&a->locempty,NULL));
  PetscCall(PetscOptionsBool("-matis_storel2l","Store local-to-local matrices generated from PtAP operations","MatISStoreL2L",a->storel2l,&a->storel2l,NULL));
  PetscCall(PetscOptionsFList("-matis_localmat_type","Matrix type","MatISSetLocalMatType",MatList,a->lmattype,type,256,&flg));
  if (flg) {
    PetscCall(MatISSetLocalMatType(A,type));
  }
  if (a->A) {
    PetscCall(MatSetFromOptions(a->A));
  }
  PetscCall(PetscOptionsTail());
  PetscFunctionReturn(0);
}

/*@
    MatCreateIS - Creates a "process" unassembled matrix, assembled on each
       process but not across processes.

   Input Parameters:
+     comm    - MPI communicator that will share the matrix
.     bs      - block size of the matrix
.     m,n,M,N - local and/or global sizes of the left and right vector used in matrix vector products
.     rmap    - local to global map for rows
-     cmap    - local to global map for cols

   Output Parameter:
.    A - the resulting matrix

   Level: advanced

   Notes:
    See MATIS for more details.
    m and n are NOT related to the size of the map; they represent the size of the local parts of the distributed vectors
    used in MatMult operations. The sizes of rmap and cmap define the size of the local matrices.
    If rmap (cmap) is NULL, then the local row (column) spaces matches the global space.

.seealso: MATIS, MatSetLocalToGlobalMapping()
@*/
PetscErrorCode MatCreateIS(MPI_Comm comm,PetscInt bs,PetscInt m,PetscInt n,PetscInt M,PetscInt N,ISLocalToGlobalMapping rmap,ISLocalToGlobalMapping cmap,Mat *A)
{
  PetscFunctionBegin;
  PetscCall(MatCreate(comm,A));
  PetscCall(MatSetSizes(*A,m,n,M,N));
  if (bs > 0) {
    PetscCall(MatSetBlockSize(*A,bs));
  }
  PetscCall(MatSetType(*A,MATIS));
  PetscCall(MatSetLocalToGlobalMapping(*A,rmap,cmap));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatHasOperation_IS(Mat A, MatOperation op, PetscBool *has)
{
  Mat_IS         *a = (Mat_IS*)A->data;

  PetscFunctionBegin;
  *has = PETSC_FALSE;
  if (!((void**)A->ops)[op]) PetscFunctionReturn(0);
  PetscCall(MatHasOperation(a->A,op,has));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSetValuesCOO_IS(Mat A,const PetscScalar v[],InsertMode imode)
{
  Mat_IS         *a = (Mat_IS*)A->data;

  PetscFunctionBegin;
  PetscCall(MatSetValuesCOO(a->A,v,imode));
  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSetPreallocationCOOLocal_IS(Mat A,PetscCount ncoo,PetscInt coo_i[],PetscInt coo_j[])
{
  Mat_IS         *a = (Mat_IS*)A->data;

  PetscFunctionBegin;
  PetscCheck(a->A,PetscObjectComm((PetscObject)A),PETSC_ERR_ORDER,"Need to provide l2g map first via MatSetLocalToGlobalMapping");
  if (a->A->rmap->mapping || a->A->cmap->mapping) {
    PetscCall(MatSetPreallocationCOOLocal(a->A,ncoo,coo_i,coo_j));
  } else {
    PetscCall(MatSetPreallocationCOO(a->A,ncoo,coo_i,coo_j));
  }
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatSetValuesCOO_C",MatSetValuesCOO_IS));
  A->preallocated = PETSC_TRUE;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSetPreallocationCOO_IS(Mat A,PetscCount ncoo,const PetscInt coo_i[],const PetscInt coo_j[])
{
  Mat_IS         *a = (Mat_IS*)A->data;
  PetscInt       *coo_il, *coo_jl, incoo;

  PetscFunctionBegin;
  PetscCheck(a->A,PetscObjectComm((PetscObject)A),PETSC_ERR_ORDER,"Need to provide l2g map first via MatSetLocalToGlobalMapping");
  PetscCheck(ncoo <= PETSC_MAX_INT,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"ncoo %" PetscCount_FMT " overflowed PetscInt; configure --with-64-bit-indices or request support",ncoo);
  PetscCall(PetscMalloc2(ncoo,&coo_il,ncoo,&coo_jl));
  PetscCall(ISGlobalToLocalMappingApply(a->rmapping,IS_GTOLM_MASK,ncoo,coo_i,&incoo,coo_il));
  PetscCall(ISGlobalToLocalMappingApply(a->cmapping,IS_GTOLM_MASK,ncoo,coo_j,&incoo,coo_jl));
  PetscCall(MatSetPreallocationCOO(a->A,ncoo,coo_il,coo_jl));
  PetscCall(PetscFree2(coo_il,coo_jl));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatSetValuesCOO_C",MatSetValuesCOO_IS));
  A->preallocated = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*@
   MatISGetLocalToGlobalMapping - Gets the local-to-global numbering of the MATIS object

   Not Collective

   Input Parameter:
.  A - the matrix

   Output Parameters:
+  rmapping - row mapping
-  cmapping - column mapping

   Notes: The returned map can be different from the one used to construct the MATIS object, since it will not contain negative or repeated indices.

   Level: advanced

.seealso:  MatSetLocalToGlobalMapping()
@*/
PetscErrorCode MatISGetLocalToGlobalMapping(Mat A,ISLocalToGlobalMapping *rmapping,ISLocalToGlobalMapping *cmapping)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidType(A,1);
  if (rmapping) PetscValidPointer(rmapping,2);
  if (cmapping) PetscValidPointer(cmapping,3);
  PetscUseMethod(A,"MatISGetLocalToGlobalMapping_C",(Mat,ISLocalToGlobalMapping*,ISLocalToGlobalMapping*),(A,rmapping,cmapping));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatISGetLocalToGlobalMapping_IS(Mat A,ISLocalToGlobalMapping *r, ISLocalToGlobalMapping *c)
{
  Mat_IS *a = (Mat_IS*)A->data;

  PetscFunctionBegin;
  if (r) *r = a->rmapping;
  if (c) *c = a->cmapping;
  PetscFunctionReturn(0);
}

/*MC
   MATIS - MATIS = "is" - A matrix type to be used for using the non-overlapping domain decomposition methods (e.g. PCBDDC or KSPFETIDP).
   This stores the matrices in globally unassembled form. Each processor assembles only its local Neumann problem and the parallel matrix vector
   product is handled "implicitly".

   Options Database Keys:
+ -mat_type is - sets the matrix type to "is" during a call to MatSetFromOptions()
. -matis_fixempty - Fixes local matrices in case of empty local rows/columns.
- -matis_storel2l - stores the local-to-local operators generated by the Galerkin process of MatPtAP().

   Notes:
    Options prefix for the inner matrix are given by -is_mat_xxx

          You must call MatSetLocalToGlobalMapping() before using this matrix type.

          You can do matrix preallocation on the local matrix after you obtain it with
          MatISGetLocalMat(); otherwise, you could use MatISSetPreallocation()

  Level: advanced

.seealso: Mat, MatISGetLocalMat(), MatSetLocalToGlobalMapping(), MatISSetPreallocation(), MatCreateIS(), PCBDDC, KSPFETIDP

M*/
PETSC_EXTERN PetscErrorCode MatCreate_IS(Mat A)
{
  Mat_IS         *a;

  PetscFunctionBegin;
  PetscCall(PetscNewLog(A,&a));
  PetscCall(PetscStrallocpy(MATAIJ,&a->lmattype));
  A->data = (void*)a;

  /* matrix ops */
  PetscCall(PetscMemzero(A->ops,sizeof(struct _MatOps)));
  A->ops->mult                    = MatMult_IS;
  A->ops->multadd                 = MatMultAdd_IS;
  A->ops->multtranspose           = MatMultTranspose_IS;
  A->ops->multtransposeadd        = MatMultTransposeAdd_IS;
  A->ops->destroy                 = MatDestroy_IS;
  A->ops->setlocaltoglobalmapping = MatSetLocalToGlobalMapping_IS;
  A->ops->setvalues               = MatSetValues_IS;
  A->ops->setvaluesblocked        = MatSetValuesBlocked_IS;
  A->ops->setvalueslocal          = MatSetValuesLocal_IS;
  A->ops->setvaluesblockedlocal   = MatSetValuesBlockedLocal_IS;
  A->ops->zerorows                = MatZeroRows_IS;
  A->ops->zerorowscolumns         = MatZeroRowsColumns_IS;
  A->ops->assemblybegin           = MatAssemblyBegin_IS;
  A->ops->assemblyend             = MatAssemblyEnd_IS;
  A->ops->view                    = MatView_IS;
  A->ops->zeroentries             = MatZeroEntries_IS;
  A->ops->scale                   = MatScale_IS;
  A->ops->getdiagonal             = MatGetDiagonal_IS;
  A->ops->setoption               = MatSetOption_IS;
  A->ops->ishermitian             = MatIsHermitian_IS;
  A->ops->issymmetric             = MatIsSymmetric_IS;
  A->ops->isstructurallysymmetric = MatIsStructurallySymmetric_IS;
  A->ops->duplicate               = MatDuplicate_IS;
  A->ops->missingdiagonal         = MatMissingDiagonal_IS;
  A->ops->copy                    = MatCopy_IS;
  A->ops->getlocalsubmatrix       = MatGetLocalSubMatrix_IS;
  A->ops->createsubmatrix         = MatCreateSubMatrix_IS;
  A->ops->axpy                    = MatAXPY_IS;
  A->ops->diagonalset             = MatDiagonalSet_IS;
  A->ops->shift                   = MatShift_IS;
  A->ops->transpose               = MatTranspose_IS;
  A->ops->getinfo                 = MatGetInfo_IS;
  A->ops->diagonalscale           = MatDiagonalScale_IS;
  A->ops->setfromoptions          = MatSetFromOptions_IS;
  A->ops->setup                   = MatSetUp_IS;
  A->ops->hasoperation            = MatHasOperation_IS;

  /* special MATIS functions */
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatISSetLocalMatType_C",MatISSetLocalMatType_IS));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatISGetLocalMat_C",MatISGetLocalMat_IS));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatISRestoreLocalMat_C",MatISRestoreLocalMat_IS));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatISSetLocalMat_C",MatISSetLocalMat_IS));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatISGetMPIXAIJ_C",MatConvert_IS_XAIJ));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatISSetPreallocation_C",MatISSetPreallocation_IS));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatISStoreL2L_C",MatISStoreL2L_IS));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatISFixLocalEmpty_C",MatISFixLocalEmpty_IS));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatISGetLocalToGlobalMapping_C",MatISGetLocalToGlobalMapping_IS));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatConvert_is_mpiaij_C",MatConvert_IS_XAIJ));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatConvert_is_mpibaij_C",MatConvert_IS_XAIJ));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatConvert_is_mpisbaij_C",MatConvert_IS_XAIJ));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatConvert_is_seqaij_C",MatConvert_IS_XAIJ));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatConvert_is_seqbaij_C",MatConvert_IS_XAIJ));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatConvert_is_seqsbaij_C",MatConvert_IS_XAIJ));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatConvert_is_aij_C",MatConvert_IS_XAIJ));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatSetPreallocationCOOLocal_C",MatSetPreallocationCOOLocal_IS));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatSetPreallocationCOO_C",MatSetPreallocationCOO_IS));
  PetscCall(PetscObjectChangeTypeName((PetscObject)A,MATIS));
  PetscFunctionReturn(0);
}
