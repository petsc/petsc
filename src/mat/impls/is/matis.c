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

#define MATIS_MAX_ENTRIES_INSERTION 2048
static PetscErrorCode MatSetValuesLocal_IS(Mat,PetscInt,const PetscInt*,PetscInt,const PetscInt*,const PetscScalar*,InsertMode);
static PetscErrorCode MatSetValuesBlockedLocal_IS(Mat,PetscInt,const PetscInt*,PetscInt,const PetscInt*,const PetscScalar*,InsertMode);
static PetscErrorCode MatISSetUpScatters_Private(Mat);

static PetscErrorCode MatISContainerDestroyPtAP_Private(void *ptr)
{
  MatISPtAP      ptap = (MatISPtAP)ptr;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatDestroySubMatrices(ptap->ris1 ? 2 : 1,&ptap->lP);CHKERRQ(ierr);
  ierr = ISDestroy(&ptap->cis0);CHKERRQ(ierr);
  ierr = ISDestroy(&ptap->cis1);CHKERRQ(ierr);
  ierr = ISDestroy(&ptap->ris0);CHKERRQ(ierr);
  ierr = ISDestroy(&ptap->ris1);CHKERRQ(ierr);
  ierr = PetscFree(ptap);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectQuery((PetscObject)C,"_MatIS_PtAP",(PetscObject*)&c);CHKERRQ(ierr);
  PetscCheckFalse(!c,PetscObjectComm((PetscObject)C),PETSC_ERR_PLIB,"Missing PtAP information");
  ierr   = PetscContainerGetPointer(c,(void**)&ptap);CHKERRQ(ierr);
  ris[0] = ptap->ris0;
  ris[1] = ptap->ris1;
  cis[0] = ptap->cis0;
  cis[1] = ptap->cis1;
  n      = ptap->ris1 ? 2 : 1;
  reuse  = ptap->lP ? MAT_REUSE_MATRIX : MAT_INITIAL_MATRIX;
  ierr   = MatCreateSubMatrices(P,n,ris,cis,reuse,&ptap->lP);CHKERRQ(ierr);

  ierr = MatISGetLocalMat(A,&lA);CHKERRQ(ierr);
  ierr = MatISGetLocalMat(C,&lC);CHKERRQ(ierr);
  if (ptap->ris1) { /* unsymmetric A mapping */
    Mat lPt;

    ierr = MatTranspose(ptap->lP[1],MAT_INITIAL_MATRIX,&lPt);CHKERRQ(ierr);
    ierr = MatMatMatMult(lPt,lA,ptap->lP[0],reuse,ptap->fill,&lC);CHKERRQ(ierr);
    if (matis->storel2l) {
      ierr = PetscObjectCompose((PetscObject)(A),"_MatIS_PtAP_l2l",(PetscObject)lPt);CHKERRQ(ierr);
    }
    ierr = MatDestroy(&lPt);CHKERRQ(ierr);
  } else {
    ierr = MatPtAP(lA,ptap->lP[0],reuse,ptap->fill,&lC);CHKERRQ(ierr);
    if (matis->storel2l) {
     ierr = PetscObjectCompose((PetscObject)C,"_MatIS_PtAP_l2l",(PetscObject)ptap->lP[0]);CHKERRQ(ierr);
    }
  }
  if (reuse == MAT_INITIAL_MATRIX) {
    ierr = MatISSetLocalMat(C,lC);CHKERRQ(ierr);
    ierr = MatDestroy(&lC);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(PT,MAT_CLASSID,1);
  PetscValidPointer(cis,2);
  ierr = PetscObjectGetComm((PetscObject)PT,&comm);CHKERRQ(ierr);
  bs   = 1;
  ierr = PetscObjectBaseTypeCompare((PetscObject)PT,MATMPIAIJ,&ismpiaij);CHKERRQ(ierr);
  ierr = PetscObjectBaseTypeCompare((PetscObject)PT,MATMPIBAIJ,&ismpibaij);CHKERRQ(ierr);
  ierr = PetscObjectBaseTypeCompare((PetscObject)PT,MATSEQAIJ,&isseqaij);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)PT,MATSEQBAIJ,&isseqbaij);CHKERRQ(ierr);
  if (isseqaij || isseqbaij) {
    Pd = PT;
    Po = NULL;
    garray = NULL;
  } else if (ismpiaij) {
    ierr = MatMPIAIJGetSeqAIJ(PT,&Pd,&Po,&garray);CHKERRQ(ierr);
  } else if (ismpibaij) {
    ierr = MatMPIBAIJGetSeqBAIJ(PT,&Pd,&Po,&garray);CHKERRQ(ierr);
    ierr = MatGetBlockSize(PT,&bs);CHKERRQ(ierr);
  } else SETERRQ(comm,PETSC_ERR_SUP,"Not for matrix type %s",((PetscObject)(PT))->type_name);

  /* identify any null columns in Pd or Po */
  /* We use a tolerance comparison since it may happen that, with geometric multigrid,
     some of the columns are not really zero, but very close to */
  zo = zd = NULL;
  if (Po) {
    ierr = MatFindNonzeroRowsOrCols_Basic(Po,PETSC_TRUE,PETSC_SMALL,&zo);CHKERRQ(ierr);
  }
  ierr = MatFindNonzeroRowsOrCols_Basic(Pd,PETSC_TRUE,PETSC_SMALL,&zd);CHKERRQ(ierr);

  ierr = MatGetLocalSize(PT,NULL,&dc);CHKERRQ(ierr);
  ierr = MatGetOwnershipRangeColumn(PT,&stc,NULL);CHKERRQ(ierr);
  if (Po) { ierr = MatGetLocalSize(Po,NULL,&oc);CHKERRQ(ierr); }
  else oc = 0;
  ierr = PetscMalloc1((dc+oc)/bs,&aux);CHKERRQ(ierr);
  if (zd) {
    const PetscInt *idxs;
    PetscInt       nz;

    /* this will throw an error if bs is not valid */
    ierr = ISSetBlockSize(zd,bs);CHKERRQ(ierr);
    ierr = ISGetLocalSize(zd,&nz);CHKERRQ(ierr);
    ierr = ISGetIndices(zd,&idxs);CHKERRQ(ierr);
    ctd  = nz/bs;
    for (i=0; i<ctd; i++) aux[i] = (idxs[bs*i]+stc)/bs;
    ierr = ISRestoreIndices(zd,&idxs);CHKERRQ(ierr);
  } else {
    ctd = dc/bs;
    for (i=0; i<ctd; i++) aux[i] = i+stc/bs;
  }
  if (zo) {
    const PetscInt *idxs;
    PetscInt       nz;

    /* this will throw an error if bs is not valid */
    ierr = ISSetBlockSize(zo,bs);CHKERRQ(ierr);
    ierr = ISGetLocalSize(zo,&nz);CHKERRQ(ierr);
    ierr = ISGetIndices(zo,&idxs);CHKERRQ(ierr);
    cto  = nz/bs;
    for (i=0; i<cto; i++) aux[i+ctd] = garray[idxs[bs*i]/bs];
    ierr = ISRestoreIndices(zo,&idxs);CHKERRQ(ierr);
  } else {
    cto = oc/bs;
    for (i=0; i<cto; i++) aux[i+ctd] = garray[i];
  }
  ierr = ISCreateBlock(comm,bs,ctd+cto,aux,PETSC_OWN_POINTER,cis);CHKERRQ(ierr);
  ierr = ISDestroy(&zd);CHKERRQ(ierr);
  ierr = ISDestroy(&zo);CHKERRQ(ierr);
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
  PetscErrorCode         ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)A,&comm);CHKERRQ(ierr);
  ierr = MatSetType(C,MATIS);CHKERRQ(ierr);
  ierr = MatISGetLocalMat(A,&lA);CHKERRQ(ierr);
  ierr = MatGetType(lA,&lmtype);CHKERRQ(ierr);
  ierr = MatISSetLocalMatType(C,lmtype);CHKERRQ(ierr);
  ierr = MatGetSize(P,NULL,&N);CHKERRQ(ierr);
  ierr = MatGetLocalSize(P,NULL,&dc);CHKERRQ(ierr);
  ierr = MatSetSizes(C,dc,dc,N,N);CHKERRQ(ierr);
/* Not sure about this
  ierr = MatGetBlockSizes(P,NULL,&ibs);CHKERRQ(ierr);
  ierr = MatSetBlockSize(*C,ibs);CHKERRQ(ierr);
*/

  ierr = PetscNew(&ptap);CHKERRQ(ierr);
  ierr = PetscContainerCreate(PETSC_COMM_SELF,&c);CHKERRQ(ierr);
  ierr = PetscContainerSetPointer(c,ptap);CHKERRQ(ierr);
  ierr = PetscContainerSetUserDestroy(c,MatISContainerDestroyPtAP_Private);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)C,"_MatIS_PtAP",(PetscObject)c);CHKERRQ(ierr);
  ierr = PetscContainerDestroy(&c);CHKERRQ(ierr);
  ptap->fill = fill;

  ierr = MatGetLocalToGlobalMapping(A,&rl2g,&cl2g);CHKERRQ(ierr);

  ierr = ISLocalToGlobalMappingGetBlockSize(cl2g,&ibs);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingGetSize(cl2g,&N);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingGetBlockIndices(cl2g,&garray);CHKERRQ(ierr);
  ierr = ISCreateBlock(comm,ibs,N/ibs,garray,PETSC_COPY_VALUES,&ptap->ris0);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingRestoreBlockIndices(cl2g,&garray);CHKERRQ(ierr);

  ierr = MatCreateSubMatrix(P,ptap->ris0,NULL,MAT_INITIAL_MATRIX,&PT);CHKERRQ(ierr);
  ierr = MatGetNonzeroColumnsLocal_Private(PT,&ptap->cis0);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingCreateIS(ptap->cis0,&Ccl2g);CHKERRQ(ierr);
  ierr = MatDestroy(&PT);CHKERRQ(ierr);

  Crl2g = NULL;
  if (rl2g != cl2g) { /* unsymmetric A mapping */
    PetscBool same,lsame = PETSC_FALSE;
    PetscInt  N1,ibs1;

    ierr = ISLocalToGlobalMappingGetSize(rl2g,&N1);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingGetBlockSize(rl2g,&ibs1);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingGetBlockIndices(rl2g,&garray);CHKERRQ(ierr);
    ierr = ISCreateBlock(comm,ibs,N/ibs,garray,PETSC_COPY_VALUES,&ptap->ris1);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingRestoreBlockIndices(rl2g,&garray);CHKERRQ(ierr);
    if (ibs1 == ibs && N1 == N) { /* check if the l2gmaps are the same */
      const PetscInt *i1,*i2;

      ierr = ISBlockGetIndices(ptap->ris0,&i1);CHKERRQ(ierr);
      ierr = ISBlockGetIndices(ptap->ris1,&i2);CHKERRQ(ierr);
      ierr = PetscArraycmp(i1,i2,N,&lsame);CHKERRQ(ierr);
    }
    ierr = MPIU_Allreduce(&lsame,&same,1,MPIU_BOOL,MPI_LAND,comm);CHKERRMPI(ierr);
    if (same) {
      ierr = ISDestroy(&ptap->ris1);CHKERRQ(ierr);
    } else {
      ierr = MatCreateSubMatrix(P,ptap->ris1,NULL,MAT_INITIAL_MATRIX,&PT);CHKERRQ(ierr);
      ierr = MatGetNonzeroColumnsLocal_Private(PT,&ptap->cis1);CHKERRQ(ierr);
      ierr = ISLocalToGlobalMappingCreateIS(ptap->cis1,&Crl2g);CHKERRQ(ierr);
      ierr = MatDestroy(&PT);CHKERRQ(ierr);
    }
  }
/* Not sure about this
  if (!Crl2g) {
    ierr = MatGetBlockSize(C,&ibs);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingSetBlockSize(Ccl2g,ibs);CHKERRQ(ierr);
  }
*/
  ierr = MatSetLocalToGlobalMapping(C,Crl2g ? Crl2g : Ccl2g,Ccl2g);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingDestroy(&Crl2g);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingDestroy(&Ccl2g);CHKERRQ(ierr);

  C->ops->ptapnumeric = MatPtAPNumeric_IS_XAIJ;
  PetscFunctionReturn(0);
}

/* ----------------------------------------- */
static PetscErrorCode MatProductSymbolic_PtAP_IS_XAIJ(Mat C)
{
  PetscErrorCode ierr;
  Mat_Product    *product = C->product;
  Mat            A=product->A,P=product->B;
  PetscReal      fill=product->fill;

  PetscFunctionBegin;
  ierr = MatPtAPSymbolic_IS_XAIJ(A,P,fill,C);CHKERRQ(ierr);
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
  PetscErrorCode ierr;
  Mat_Product    *product = C->product;

  PetscFunctionBegin;
  if (product->type == MATPRODUCT_PtAP) {
    ierr = MatProductSetFromOptions_IS_XAIJ_PtAP(C);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* ----------------------------------------- */
static PetscErrorCode MatISContainerDestroyFields_Private(void *ptr)
{
  MatISLocalFields lf = (MatISLocalFields)ptr;
  PetscInt         i;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  for (i=0;i<lf->nr;i++) {
    ierr = ISDestroy(&lf->rf[i]);CHKERRQ(ierr);
  }
  for (i=0;i<lf->nc;i++) {
    ierr = ISDestroy(&lf->cf[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree2(lf->rf,lf->cf);CHKERRQ(ierr);
  ierr = PetscFree(lf);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatConvert_SeqXAIJ_IS(Mat A,MatType type,MatReuse reuse,Mat *newmat)
{
  Mat            B,lB;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (reuse != MAT_REUSE_MATRIX) {
    ISLocalToGlobalMapping rl2g,cl2g;
    PetscInt               bs;
    IS                     is;

    ierr = MatGetBlockSize(A,&bs);CHKERRQ(ierr);
    ierr = ISCreateStride(PetscObjectComm((PetscObject)A),A->rmap->n/bs,0,1,&is);CHKERRQ(ierr);
    if (bs > 1) {
      IS       is2;
      PetscInt i,*aux;

      ierr = ISGetLocalSize(is,&i);CHKERRQ(ierr);
      ierr = ISGetIndices(is,(const PetscInt**)&aux);CHKERRQ(ierr);
      ierr = ISCreateBlock(PetscObjectComm((PetscObject)A),bs,i,aux,PETSC_COPY_VALUES,&is2);CHKERRQ(ierr);
      ierr = ISRestoreIndices(is,(const PetscInt**)&aux);CHKERRQ(ierr);
      ierr = ISDestroy(&is);CHKERRQ(ierr);
      is   = is2;
    }
    ierr = ISSetIdentity(is);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingCreateIS(is,&rl2g);CHKERRQ(ierr);
    ierr = ISDestroy(&is);CHKERRQ(ierr);
    ierr = ISCreateStride(PetscObjectComm((PetscObject)A),A->cmap->n/bs,0,1,&is);CHKERRQ(ierr);
    if (bs > 1) {
      IS       is2;
      PetscInt i,*aux;

      ierr = ISGetLocalSize(is,&i);CHKERRQ(ierr);
      ierr = ISGetIndices(is,(const PetscInt**)&aux);CHKERRQ(ierr);
      ierr = ISCreateBlock(PetscObjectComm((PetscObject)A),bs,i,aux,PETSC_COPY_VALUES,&is2);CHKERRQ(ierr);
      ierr = ISRestoreIndices(is,(const PetscInt**)&aux);CHKERRQ(ierr);
      ierr = ISDestroy(&is);CHKERRQ(ierr);
      is   = is2;
    }
    ierr = ISSetIdentity(is);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingCreateIS(is,&cl2g);CHKERRQ(ierr);
    ierr = ISDestroy(&is);CHKERRQ(ierr);
    ierr = MatCreateIS(PetscObjectComm((PetscObject)A),bs,A->rmap->n,A->cmap->n,A->rmap->N,A->cmap->N,rl2g,cl2g,&B);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingDestroy(&rl2g);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingDestroy(&cl2g);CHKERRQ(ierr);
    ierr = MatDuplicate(A,MAT_COPY_VALUES,&lB);CHKERRQ(ierr);
    if (reuse == MAT_INITIAL_MATRIX) *newmat = B;
  } else {
    B    = *newmat;
    ierr = PetscObjectReference((PetscObject)A);CHKERRQ(ierr);
    lB   = A;
  }
  ierr = MatISSetLocalMat(B,lB);CHKERRQ(ierr);
  ierr = MatDestroy(&lB);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (reuse == MAT_INPLACE_MATRIX) {
    ierr = MatHeaderReplace(A,&B);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatGetRowIJ(matis->A,0,PETSC_FALSE,PETSC_FALSE,&m,&ii,&jj,&flg);CHKERRQ(ierr);
  PetscCheckFalse(!flg,PETSC_COMM_SELF,PETSC_ERR_SUP,"Cannot get IJ structure");
  ierr = ISLocalToGlobalMappingGetNodeInfo(A->rmap->mapping,&n,&ecount,&eneighs);CHKERRQ(ierr);
  PetscCheckFalse(m != n,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Unexpected %" PetscInt_FMT " != %" PetscInt_FMT,m,n);
  ierr = MatSeqAIJGetArray(matis->A,&aa);CHKERRQ(ierr);
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
  ierr = ISLocalToGlobalMappingRestoreNodeInfo(A->rmap->mapping,&n,&ecount,&eneighs);CHKERRQ(ierr);
  ierr = MatSeqAIJRestoreArray(matis->A,&aa);CHKERRQ(ierr);
  ierr = MatRestoreRowIJ(matis->A,0,PETSC_FALSE,PETSC_FALSE,&m,&ii,&jj,&flg);CHKERRQ(ierr);
  PetscCheckFalse(!flg,PETSC_COMM_SELF,PETSC_ERR_SUP,"Cannot restore IJ structure");
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
  ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)A),((PetscObject)A)->prefix,"MatIS l2g disassembling options","Mat");CHKERRQ(ierr);
  ierr = PetscOptionsEnum("-mat_is_disassemble_l2g_type","Type of local-to-global mapping to be used for disassembling","MatISDisassemblel2gType",MatISDisassemblel2gTypes,(PetscEnum)mode,(PetscEnum*)&mode,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  if (mode == MAT_IS_DISASSEMBLE_L2G_MAT) {
    ierr = MatGetLocalToGlobalMapping(A,l2g,NULL);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  ierr = PetscObjectGetComm((PetscObject)A,&comm);CHKERRQ(ierr);
  ierr = PetscObjectBaseTypeCompare((PetscObject)A,MATMPIAIJ ,&ismpiaij);CHKERRQ(ierr);
  ierr = PetscObjectBaseTypeCompare((PetscObject)A,MATMPIBAIJ,&ismpibaij);CHKERRQ(ierr);
  ierr = MatGetBlockSize(A,&bs);CHKERRQ(ierr);
  switch (mode) {
  case MAT_IS_DISASSEMBLE_L2G_ND:
    ierr = MatPartitioningCreate(comm,&part);CHKERRQ(ierr);
    ierr = MatPartitioningSetAdjacency(part,A);CHKERRQ(ierr);
    ierr = PetscObjectSetOptionsPrefix((PetscObject)part,((PetscObject)A)->prefix);CHKERRQ(ierr);
    ierr = MatPartitioningSetFromOptions(part);CHKERRQ(ierr);
    ierr = MatPartitioningApplyND(part,&ndmap);CHKERRQ(ierr);
    ierr = MatPartitioningDestroy(&part);CHKERRQ(ierr);
    ierr = ISBuildTwoSided(ndmap,NULL,&ndsub);CHKERRQ(ierr);
    ierr = MatMPIAIJSetUseScalableIncreaseOverlap(A,PETSC_TRUE);CHKERRQ(ierr);
    ierr = MatIncreaseOverlap(A,1,&ndsub,1);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingCreateIS(ndsub,l2g);CHKERRQ(ierr);

    /* it may happen that a separator node is not properly shared */
    ierr = ISLocalToGlobalMappingGetNodeInfo(*l2g,&nl,&ncount,NULL);CHKERRQ(ierr);
    ierr = PetscSFCreate(comm,&sf);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingGetIndices(*l2g,&garray);CHKERRQ(ierr);
    ierr = PetscSFSetGraphLayout(sf,A->rmap,nl,NULL,PETSC_OWN_POINTER,garray);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingRestoreIndices(*l2g,&garray);CHKERRQ(ierr);
    ierr = PetscCalloc1(A->rmap->n,&ndmapc);CHKERRQ(ierr);
    ierr = PetscSFReduceBegin(sf,MPIU_INT,ncount,ndmapc,MPI_REPLACE);CHKERRQ(ierr);
    ierr = PetscSFReduceEnd(sf,MPIU_INT,ncount,ndmapc,MPI_REPLACE);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingRestoreNodeInfo(*l2g,NULL,&ncount,NULL);CHKERRQ(ierr);
    ierr = ISGetIndices(ndmap,&ndmapi);CHKERRQ(ierr);
    for (i = 0, cnt = 0; i < A->rmap->n; i++)
      if (ndmapi[i] < 0 && ndmapc[i] < 2)
        cnt++;

    ierr = MPIU_Allreduce(&cnt,&i,1,MPIU_INT,MPI_MAX,comm);CHKERRMPI(ierr);
    if (i) { /* we detected isolated separator nodes */
      Mat                    A2,A3;
      IS                     *workis,is2;
      PetscScalar            *vals;
      PetscInt               gcnt = i,*dnz,*onz,j,*lndmapi;
      ISLocalToGlobalMapping ll2g;
      PetscBool              flg;
      const PetscInt         *ii,*jj;

      /* communicate global id of separators */
      ierr = MatPreallocateInitialize(comm,A->rmap->n,A->cmap->n,dnz,onz);CHKERRQ(ierr);
      for (i = 0, cnt = 0; i < A->rmap->n; i++)
        dnz[i] = ndmapi[i] < 0 ? i + A->rmap->rstart : -1;

      ierr = PetscMalloc1(nl,&lndmapi);CHKERRQ(ierr);
      ierr = PetscSFBcastBegin(sf,MPIU_INT,dnz,lndmapi,MPI_REPLACE);CHKERRQ(ierr);

      /* compute adjacency of isolated separators node */
      ierr = PetscMalloc1(gcnt,&workis);CHKERRQ(ierr);
      for (i = 0, cnt = 0; i < A->rmap->n; i++) {
        if (ndmapi[i] < 0 && ndmapc[i] < 2) {
          ierr = ISCreateStride(comm,1,i+A->rmap->rstart,1,&workis[cnt++]);CHKERRQ(ierr);
        }
      }
      for (i = cnt; i < gcnt; i++) {
        ierr = ISCreateStride(comm,0,0,1,&workis[i]);CHKERRQ(ierr);
      }
      for (i = 0; i < gcnt; i++) {
        ierr = PetscObjectSetName((PetscObject)workis[i],"ISOLATED");CHKERRQ(ierr);
        ierr = ISViewFromOptions(workis[i],NULL,"-view_isolated_separators");CHKERRQ(ierr);
      }

      /* no communications since all the ISes correspond to locally owned rows */
      ierr = MatIncreaseOverlap(A,gcnt,workis,1);CHKERRQ(ierr);

      /* end communicate global id of separators */
      ierr = PetscSFBcastEnd(sf,MPIU_INT,dnz,lndmapi,MPI_REPLACE);CHKERRQ(ierr);

      /* communicate new layers : create a matrix and transpose it */
      ierr = PetscArrayzero(dnz,A->rmap->n);CHKERRQ(ierr);
      ierr = PetscArrayzero(onz,A->rmap->n);CHKERRQ(ierr);
      for (i = 0, j = 0; i < A->rmap->n; i++) {
        if (ndmapi[i] < 0 && ndmapc[i] < 2) {
          const PetscInt* idxs;
          PetscInt        s;

          ierr = ISGetLocalSize(workis[j],&s);CHKERRQ(ierr);
          ierr = ISGetIndices(workis[j],&idxs);CHKERRQ(ierr);
          ierr = MatPreallocateSet(i+A->rmap->rstart,s,idxs,dnz,onz);CHKERRQ(ierr);
          j++;
        }
      }
      PetscCheckFalse(j != cnt,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Unexpected local count %" PetscInt_FMT " != %" PetscInt_FMT,j,cnt);

      for (i = 0; i < gcnt; i++) {
        ierr = PetscObjectSetName((PetscObject)workis[i],"EXTENDED");CHKERRQ(ierr);
        ierr = ISViewFromOptions(workis[i],NULL,"-view_isolated_separators");CHKERRQ(ierr);
      }

      for (i = 0, j = 0; i < A->rmap->n; i++) j = PetscMax(j,dnz[i]+onz[i]);
      ierr = PetscMalloc1(j,&vals);CHKERRQ(ierr);
      for (i = 0; i < j; i++) vals[i] = 1.0;

      ierr = MatCreate(comm,&A2);CHKERRQ(ierr);
      ierr = MatSetType(A2,MATMPIAIJ);CHKERRQ(ierr);
      ierr = MatSetSizes(A2,A->rmap->n,A->cmap->n,A->rmap->N,A->cmap->N);CHKERRQ(ierr);
      ierr = MatMPIAIJSetPreallocation(A2,0,dnz,0,onz);CHKERRQ(ierr);
      ierr = MatSetOption(A2,MAT_NO_OFF_PROC_ENTRIES,PETSC_TRUE);CHKERRQ(ierr);
      for (i = 0, j = 0; i < A2->rmap->n; i++) {
        PetscInt        row = i+A2->rmap->rstart,s = dnz[i] + onz[i];
        const PetscInt* idxs;

        if (s) {
          ierr = ISGetIndices(workis[j],&idxs);CHKERRQ(ierr);
          ierr = MatSetValues(A2,1,&row,s,idxs,vals,INSERT_VALUES);CHKERRQ(ierr);
          ierr = ISRestoreIndices(workis[j],&idxs);CHKERRQ(ierr);
          j++;
        }
      }
      PetscCheckFalse(j != cnt,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Unexpected local count %" PetscInt_FMT " != %" PetscInt_FMT,j,cnt);
      ierr = PetscFree(vals);CHKERRQ(ierr);
      ierr = MatAssemblyBegin(A2,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      ierr = MatAssemblyEnd(A2,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      ierr = MatTranspose(A2,MAT_INPLACE_MATRIX,&A2);CHKERRQ(ierr);

      /* extract submatrix corresponding to the coupling "owned separators" x "isolated separators" */
      for (i = 0, j = 0; i < nl; i++)
        if (lndmapi[i] >= 0) lndmapi[j++] = lndmapi[i];
      ierr = ISCreateGeneral(comm,j,lndmapi,PETSC_USE_POINTER,&is);CHKERRQ(ierr);
      ierr = MatMPIAIJGetLocalMatCondensed(A2,MAT_INITIAL_MATRIX,&is,NULL,&A3);CHKERRQ(ierr);
      ierr = ISDestroy(&is);CHKERRQ(ierr);
      ierr = MatDestroy(&A2);CHKERRQ(ierr);

      /* extend local to global map to include connected isolated separators */
      ierr = PetscObjectQuery((PetscObject)A3,"_petsc_GetLocalMatCondensed_iscol",(PetscObject*)&is);CHKERRQ(ierr);
      PetscCheckFalse(!is,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Missing column map");
      ierr = ISLocalToGlobalMappingCreateIS(is,&ll2g);CHKERRQ(ierr);
      ierr = MatGetRowIJ(A3,0,PETSC_FALSE,PETSC_FALSE,&i,&ii,&jj,&flg);CHKERRQ(ierr);
      PetscCheckFalse(!flg,PETSC_COMM_SELF,PETSC_ERR_SUP,"Cannot get IJ structure");
      ierr = ISCreateGeneral(PETSC_COMM_SELF,ii[i],jj,PETSC_COPY_VALUES,&is);CHKERRQ(ierr);
      ierr = MatRestoreRowIJ(A3,0,PETSC_FALSE,PETSC_FALSE,&i,&ii,&jj,&flg);CHKERRQ(ierr);
      PetscCheckFalse(!flg,PETSC_COMM_SELF,PETSC_ERR_SUP,"Cannot get IJ structure");
      ierr = ISLocalToGlobalMappingApplyIS(ll2g,is,&is2);CHKERRQ(ierr);
      ierr = ISDestroy(&is);CHKERRQ(ierr);
      ierr = ISLocalToGlobalMappingDestroy(&ll2g);CHKERRQ(ierr);

      /* add new nodes to the local-to-global map */
      ierr = ISLocalToGlobalMappingDestroy(l2g);CHKERRQ(ierr);
      ierr = ISExpand(ndsub,is2,&is);CHKERRQ(ierr);
      ierr = ISDestroy(&is2);CHKERRQ(ierr);
      ierr = ISLocalToGlobalMappingCreateIS(is,l2g);CHKERRQ(ierr);
      ierr = ISDestroy(&is);CHKERRQ(ierr);

      ierr = MatDestroy(&A3);CHKERRQ(ierr);
      ierr = PetscFree(lndmapi);CHKERRQ(ierr);
      ierr = MatPreallocateFinalize(dnz,onz);CHKERRQ(ierr);
      for (i = 0; i < gcnt; i++) {
        ierr = ISDestroy(&workis[i]);CHKERRQ(ierr);
      }
      ierr = PetscFree(workis);CHKERRQ(ierr);
    }
    ierr = ISRestoreIndices(ndmap,&ndmapi);CHKERRQ(ierr);
    ierr = PetscSFDestroy(&sf);CHKERRQ(ierr);
    ierr = PetscFree(ndmapc);CHKERRQ(ierr);
    ierr = ISDestroy(&ndmap);CHKERRQ(ierr);
    ierr = ISDestroy(&ndsub);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingSetBlockSize(*l2g,bs);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingViewFromOptions(*l2g,NULL,"-matis_nd_l2g_view");CHKERRQ(ierr);
    break;
  case MAT_IS_DISASSEMBLE_L2G_NATURAL:
    if (ismpiaij) {
      ierr = MatMPIAIJGetSeqAIJ(A,&Ad,&Ao,&garray);CHKERRQ(ierr);
    } else if (ismpibaij) {
      ierr = MatMPIBAIJGetSeqBAIJ(A,&Ad,&Ao,&garray);CHKERRQ(ierr);
    } else SETERRQ(comm,PETSC_ERR_SUP,"Type %s",((PetscObject)A)->type_name);
    PetscCheckFalse(!garray,comm,PETSC_ERR_ARG_WRONGSTATE,"garray not present");
    if (A->rmap->n) {
      PetscInt dc,oc,stc,*aux;

      ierr = MatGetLocalSize(A,NULL,&dc);CHKERRQ(ierr);
      ierr = MatGetLocalSize(Ao,NULL,&oc);CHKERRQ(ierr);
      ierr = MatGetOwnershipRangeColumn(A,&stc,NULL);CHKERRQ(ierr);
      ierr = PetscMalloc1((dc+oc)/bs,&aux);CHKERRQ(ierr);
      for (i=0; i<dc/bs; i++) aux[i]       = i+stc/bs;
      for (i=0; i<oc/bs; i++) aux[i+dc/bs] = garray[i];
      ierr = ISCreateBlock(comm,bs,(dc+oc)/bs,aux,PETSC_OWN_POINTER,&is);CHKERRQ(ierr);
    } else {
      ierr = ISCreateBlock(comm,1,0,NULL,PETSC_OWN_POINTER,&is);CHKERRQ(ierr);
    }
    ierr = ISLocalToGlobalMappingCreateIS(is,l2g);CHKERRQ(ierr);
    ierr = ISDestroy(&is);CHKERRQ(ierr);
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
  PetscErrorCode         ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)A,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRMPI(ierr);
  if (size == 1) {
    ierr = MatConvert_SeqXAIJ_IS(A,type,reuse,newmat);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  if (reuse != MAT_REUSE_MATRIX && A->cmap->N == A->rmap->N) {
    ierr = MatMPIXAIJComputeLocalToGlobalMapping_Private(A,&rl2g);CHKERRQ(ierr);
    ierr = MatCreate(comm,&B);CHKERRQ(ierr);
    ierr = MatSetType(B,MATIS);CHKERRQ(ierr);
    ierr = MatSetSizes(B,A->rmap->n,A->cmap->n,A->rmap->N,A->cmap->N);CHKERRQ(ierr);
    ierr = MatSetLocalToGlobalMapping(B,rl2g,rl2g);CHKERRQ(ierr);
    ierr = MatGetBlockSize(A,&bs);CHKERRQ(ierr);
    ierr = MatSetBlockSize(B,bs);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingDestroy(&rl2g);CHKERRQ(ierr);
    if (reuse == MAT_INPLACE_MATRIX) was_inplace = PETSC_TRUE;
    reuse = MAT_REUSE_MATRIX;
  }
  if (reuse == MAT_REUSE_MATRIX) {
    Mat            *newlA, lA;
    IS             rows, cols;
    const PetscInt *ridx, *cidx;
    PetscInt       rbs, cbs, nr, nc;

    if (!B) B = *newmat;
    ierr = MatGetLocalToGlobalMapping(B,&rl2g,&cl2g);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingGetBlockIndices(rl2g,&ridx);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingGetBlockIndices(cl2g,&cidx);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingGetSize(rl2g,&nr);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingGetSize(cl2g,&nc);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingGetBlockSize(rl2g,&rbs);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingGetBlockSize(cl2g,&cbs);CHKERRQ(ierr);
    ierr = ISCreateBlock(comm,rbs,nr/rbs,ridx,PETSC_USE_POINTER,&rows);CHKERRQ(ierr);
    if (rl2g != cl2g) {
      ierr = ISCreateBlock(comm,cbs,nc/cbs,cidx,PETSC_USE_POINTER,&cols);CHKERRQ(ierr);
    } else {
      ierr = PetscObjectReference((PetscObject)rows);CHKERRQ(ierr);
      cols = rows;
    }
    ierr = MatISGetLocalMat(B,&lA);CHKERRQ(ierr);
    ierr = MatCreateSubMatrices(A,1,&rows,&cols,MAT_INITIAL_MATRIX,&newlA);CHKERRQ(ierr);
    ierr = MatConvert(newlA[0],MATSEQAIJ,MAT_INPLACE_MATRIX,&newlA[0]);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingRestoreBlockIndices(rl2g,&ridx);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingRestoreBlockIndices(cl2g,&cidx);CHKERRQ(ierr);
    ierr = ISDestroy(&rows);CHKERRQ(ierr);
    ierr = ISDestroy(&cols);CHKERRQ(ierr);
    if (!lA->preallocated) { /* first time */
      ierr = MatDuplicate(newlA[0],MAT_COPY_VALUES,&lA);CHKERRQ(ierr);
      ierr = MatISSetLocalMat(B,lA);CHKERRQ(ierr);
      ierr = PetscObjectDereference((PetscObject)lA);CHKERRQ(ierr);
    }
    ierr = MatCopy(newlA[0],lA,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = MatDestroySubMatrices(1,&newlA);CHKERRQ(ierr);
    ierr = MatISScaleDisassembling_Private(B);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    if (was_inplace) { ierr = MatHeaderReplace(A,&B);CHKERRQ(ierr); }
    else *newmat = B;
    PetscFunctionReturn(0);
  }
  /* rectangular case, just compress out the column space */
  ierr = PetscObjectBaseTypeCompare((PetscObject)A,MATMPIAIJ ,&ismpiaij);CHKERRQ(ierr);
  ierr = PetscObjectBaseTypeCompare((PetscObject)A,MATMPIBAIJ,&ismpibaij);CHKERRQ(ierr);
  if (ismpiaij) {
    bs   = 1;
    ierr = MatMPIAIJGetSeqAIJ(A,&Ad,&Ao,&garray);CHKERRQ(ierr);
  } else if (ismpibaij) {
    ierr = MatGetBlockSize(A,&bs);CHKERRQ(ierr);
    ierr = MatMPIBAIJGetSeqBAIJ(A,&Ad,&Ao,&garray);CHKERRQ(ierr);
    ierr = MatConvert(Ad,MATSEQAIJ,MAT_INITIAL_MATRIX,&Ad);CHKERRQ(ierr);
    ierr = MatConvert(Ao,MATSEQAIJ,MAT_INITIAL_MATRIX,&Ao);CHKERRQ(ierr);
  } else SETERRQ(comm,PETSC_ERR_SUP,"Type %s",((PetscObject)A)->type_name);
  ierr = MatSeqAIJGetArray(Ad,&dd);CHKERRQ(ierr);
  ierr = MatSeqAIJGetArray(Ao,&od);CHKERRQ(ierr);
  PetscCheckFalse(!garray,comm,PETSC_ERR_ARG_WRONGSTATE,"garray not present");

  /* access relevant information from MPIAIJ */
  ierr = MatGetOwnershipRange(A,&str,NULL);CHKERRQ(ierr);
  ierr = MatGetOwnershipRangeColumn(A,&stc,NULL);CHKERRQ(ierr);
  ierr = MatGetLocalSize(A,&dr,&dc);CHKERRQ(ierr);
  ierr = MatGetLocalSize(Ao,NULL,&oc);CHKERRQ(ierr);
  ierr = MatGetRowIJ(Ad,0,PETSC_FALSE,PETSC_FALSE,&i,&di,&dj,&flg);CHKERRQ(ierr);
  PetscCheckFalse(!flg,PETSC_COMM_SELF,PETSC_ERR_SUP,"Cannot get IJ structure");
  ierr = MatGetRowIJ(Ao,0,PETSC_FALSE,PETSC_FALSE,&i,&oi,&oj,&flg);CHKERRQ(ierr);
  PetscCheckFalse(!flg,PETSC_COMM_SELF,PETSC_ERR_SUP,"Cannot get IJ structure");
  nnz = di[dr] + oi[dr];
  /* store original pointers to be restored later */
  odi = di; odj = dj; ooi = oi; ooj = oj;

  /* generate l2g maps for rows and cols */
  ierr = ISCreateStride(comm,dr/bs,str/bs,1,&is);CHKERRQ(ierr);
  if (bs > 1) {
    IS is2;

    ierr = ISGetLocalSize(is,&i);CHKERRQ(ierr);
    ierr = ISGetIndices(is,(const PetscInt**)&aux);CHKERRQ(ierr);
    ierr = ISCreateBlock(comm,bs,i,aux,PETSC_COPY_VALUES,&is2);CHKERRQ(ierr);
    ierr = ISRestoreIndices(is,(const PetscInt**)&aux);CHKERRQ(ierr);
    ierr = ISDestroy(&is);CHKERRQ(ierr);
    is   = is2;
  }
  ierr = ISLocalToGlobalMappingCreateIS(is,&rl2g);CHKERRQ(ierr);
  ierr = ISDestroy(&is);CHKERRQ(ierr);
  if (dr) {
    ierr = PetscMalloc1((dc+oc)/bs,&aux);CHKERRQ(ierr);
    for (i=0; i<dc/bs; i++) aux[i]       = i+stc/bs;
    for (i=0; i<oc/bs; i++) aux[i+dc/bs] = garray[i];
    ierr = ISCreateBlock(comm,bs,(dc+oc)/bs,aux,PETSC_OWN_POINTER,&is);CHKERRQ(ierr);
    lc   = dc+oc;
  } else {
    ierr = ISCreateBlock(comm,bs,0,NULL,PETSC_OWN_POINTER,&is);CHKERRQ(ierr);
    lc   = 0;
  }
  ierr = ISLocalToGlobalMappingCreateIS(is,&cl2g);CHKERRQ(ierr);
  ierr = ISDestroy(&is);CHKERRQ(ierr);

  /* create MATIS object */
  ierr = MatCreate(comm,&B);CHKERRQ(ierr);
  ierr = MatSetSizes(B,dr,dc,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = MatSetType(B,MATIS);CHKERRQ(ierr);
  ierr = MatSetBlockSize(B,bs);CHKERRQ(ierr);
  ierr = MatSetLocalToGlobalMapping(B,rl2g,cl2g);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingDestroy(&rl2g);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingDestroy(&cl2g);CHKERRQ(ierr);

  /* merge local matrices */
  ierr = PetscMalloc1(nnz+dr+1,&aux);CHKERRQ(ierr);
  ierr = PetscMalloc1(nnz,&data);CHKERRQ(ierr);
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

  ierr = MatRestoreRowIJ(Ad,0,PETSC_FALSE,PETSC_FALSE,&i,&odi,&odj,&flg);CHKERRQ(ierr);
  PetscCheckFalse(!flg,PETSC_COMM_SELF,PETSC_ERR_SUP,"Cannot restore IJ structure");
  ierr = MatRestoreRowIJ(Ao,0,PETSC_FALSE,PETSC_FALSE,&i,&ooi,&ooj,&flg);CHKERRQ(ierr);
  PetscCheckFalse(!flg,PETSC_COMM_SELF,PETSC_ERR_SUP,"Cannot restore IJ structure");
  ierr = MatSeqAIJRestoreArray(Ad,&dd);CHKERRQ(ierr);
  ierr = MatSeqAIJRestoreArray(Ao,&od);CHKERRQ(ierr);

  ii   = aux;
  jj   = aux+dr+1;
  aa   = data;
  ierr = MatCreateSeqAIJWithArrays(PETSC_COMM_SELF,dr,lc,ii,jj,aa,&lA);CHKERRQ(ierr);

  /* create containers to destroy the data */
  ptrs[0] = aux;
  ptrs[1] = data;
  for (i=0; i<2; i++) {
    PetscContainer c;

    ierr = PetscContainerCreate(PETSC_COMM_SELF,&c);CHKERRQ(ierr);
    ierr = PetscContainerSetPointer(c,ptrs[i]);CHKERRQ(ierr);
    ierr = PetscContainerSetUserDestroy(c,PetscContainerUserDestroyDefault);CHKERRQ(ierr);
    ierr = PetscObjectCompose((PetscObject)lA,names[i],(PetscObject)c);CHKERRQ(ierr);
    ierr = PetscContainerDestroy(&c);CHKERRQ(ierr);
  }
  if (ismpibaij) { /* destroy converted local matrices */
    ierr = MatDestroy(&Ad);CHKERRQ(ierr);
    ierr = MatDestroy(&Ao);CHKERRQ(ierr);
  }

  /* finalize matrix */
  ierr = MatISSetLocalMat(B,lA);CHKERRQ(ierr);
  ierr = MatDestroy(&lA);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (reuse == MAT_INPLACE_MATRIX) {
    ierr = MatHeaderReplace(A,&B);CHKERRQ(ierr);
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
  PetscErrorCode         ierr;

  PetscFunctionBegin;
  ierr   = MatNestGetSubMats(A,&nr,&nc,&nest);CHKERRQ(ierr);
  lreuse = PETSC_FALSE;
  rnest  = NULL;
  if (reuse == MAT_REUSE_MATRIX) {
    PetscBool ismatis,isnest;

    ierr = PetscObjectTypeCompare((PetscObject)*newmat,MATIS,&ismatis);CHKERRQ(ierr);
    PetscCheckFalse(!ismatis,PetscObjectComm((PetscObject)*newmat),PETSC_ERR_USER,"Cannot reuse matrix of type %s",((PetscObject)(*newmat))->type_name);
    ierr = MatISGetLocalMat(*newmat,&lA);CHKERRQ(ierr);
    ierr = PetscObjectTypeCompare((PetscObject)lA,MATNEST,&isnest);CHKERRQ(ierr);
    if (isnest) {
      ierr   = MatNestGetSubMats(lA,&i,&j,&rnest);CHKERRQ(ierr);
      lreuse = (PetscBool)(i == nr && j == nc);
      if (!lreuse) rnest = NULL;
    }
  }
  ierr = PetscObjectGetComm((PetscObject)A,&comm);CHKERRQ(ierr);
  ierr = PetscCalloc2(nr,&lr,nc,&lc);CHKERRQ(ierr);
  ierr = PetscCalloc6(nr,&isrow,nc,&iscol,nr,&islrow,nc,&islcol,nr*nc,&snest,nr*nc,&istrans);CHKERRQ(ierr);
  ierr = MatNestGetISs(A,isrow,iscol);CHKERRQ(ierr);
  for (i=0;i<nr;i++) {
    for (j=0;j<nc;j++) {
      PetscBool ismatis;
      PetscInt  l1,l2,lb1,lb2,ij=i*nc+j;

      /* Null matrix pointers are allowed in MATNEST */
      if (!nest[i][j]) continue;

      /* Nested matrices should be of type MATIS */
      ierr = PetscObjectTypeCompare((PetscObject)nest[i][j],MATTRANSPOSEMAT,&istrans[ij]);CHKERRQ(ierr);
      if (istrans[ij]) {
        Mat T,lT;
        ierr = MatTransposeGetMat(nest[i][j],&T);CHKERRQ(ierr);
        ierr = PetscObjectTypeCompare((PetscObject)T,MATIS,&ismatis);CHKERRQ(ierr);
        PetscCheckFalse(!ismatis,comm,PETSC_ERR_SUP,"Cannot convert from MATNEST to MATIS! Matrix block (%" PetscInt_FMT ",%" PetscInt_FMT ") (transposed) is not of type MATIS",i,j);
        ierr = MatISGetLocalMat(T,&lT);CHKERRQ(ierr);
        ierr = MatCreateTranspose(lT,&snest[ij]);CHKERRQ(ierr);
      } else {
        ierr = PetscObjectTypeCompare((PetscObject)nest[i][j],MATIS,&ismatis);CHKERRQ(ierr);
        PetscCheckFalse(!ismatis,comm,PETSC_ERR_SUP,"Cannot convert from MATNEST to MATIS! Matrix block (%" PetscInt_FMT ",%" PetscInt_FMT ") is not of type MATIS",i,j);
        ierr = MatISGetLocalMat(nest[i][j],&snest[ij]);CHKERRQ(ierr);
      }

      /* Check compatibility of local sizes */
      ierr = MatGetSize(snest[ij],&l1,&l2);CHKERRQ(ierr);
      ierr = MatGetBlockSizes(snest[ij],&lb1,&lb2);CHKERRQ(ierr);
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

          ierr = MatTransposeGetMat(nest[i][j],&T);CHKERRQ(ierr);
          ierr = MatGetLocalToGlobalMapping(T,NULL,&cl2g);CHKERRQ(ierr);
        } else {
          ierr = MatGetLocalToGlobalMapping(nest[i][j],&cl2g,NULL);CHKERRQ(ierr);
        }
        ierr = ISLocalToGlobalMappingGetSize(cl2g,&n1);CHKERRQ(ierr);
        if (!n1) continue;
        if (!rl2g) {
          rl2g = cl2g;
        } else {
          const PetscInt *idxs1,*idxs2;
          PetscBool      same;

          ierr = ISLocalToGlobalMappingGetSize(rl2g,&n2);CHKERRQ(ierr);
          PetscCheckFalse(n1 != n2,PETSC_COMM_SELF,PETSC_ERR_SUP,"Cannot convert from MATNEST to MATIS! Matrix block (%" PetscInt_FMT ",%" PetscInt_FMT ") has invalid row l2gmap size %" PetscInt_FMT " != %" PetscInt_FMT,i,j,n1,n2);
          ierr = ISLocalToGlobalMappingGetIndices(cl2g,&idxs1);CHKERRQ(ierr);
          ierr = ISLocalToGlobalMappingGetIndices(rl2g,&idxs2);CHKERRQ(ierr);
          ierr = PetscArraycmp(idxs1,idxs2,n1,&same);CHKERRQ(ierr);
          ierr = ISLocalToGlobalMappingRestoreIndices(cl2g,&idxs1);CHKERRQ(ierr);
          ierr = ISLocalToGlobalMappingRestoreIndices(rl2g,&idxs2);CHKERRQ(ierr);
          PetscCheckFalse(!same,PETSC_COMM_SELF,PETSC_ERR_SUP,"Cannot convert from MATNEST to MATIS! Matrix block (%" PetscInt_FMT ",%" PetscInt_FMT ") has invalid row l2gmap",i,j);
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

          ierr = MatTransposeGetMat(nest[j][i],&T);CHKERRQ(ierr);
          ierr = MatGetLocalToGlobalMapping(T,&cl2g,NULL);CHKERRQ(ierr);
        } else {
          ierr = MatGetLocalToGlobalMapping(nest[j][i],NULL,&cl2g);CHKERRQ(ierr);
        }
        ierr = ISLocalToGlobalMappingGetSize(cl2g,&n1);CHKERRQ(ierr);
        if (!n1) continue;
        if (!rl2g) {
          rl2g = cl2g;
        } else {
          const PetscInt *idxs1,*idxs2;
          PetscBool      same;

          ierr = ISLocalToGlobalMappingGetSize(rl2g,&n2);CHKERRQ(ierr);
          PetscCheckFalse(n1 != n2,PETSC_COMM_SELF,PETSC_ERR_SUP,"Cannot convert from MATNEST to MATIS! Matrix block (%" PetscInt_FMT ",%" PetscInt_FMT ") has invalid column l2gmap size %" PetscInt_FMT " != %" PetscInt_FMT,j,i,n1,n2);
          ierr = ISLocalToGlobalMappingGetIndices(cl2g,&idxs1);CHKERRQ(ierr);
          ierr = ISLocalToGlobalMappingGetIndices(rl2g,&idxs2);CHKERRQ(ierr);
          ierr = PetscArraycmp(idxs1,idxs2,n1,&same);CHKERRQ(ierr);
          ierr = ISLocalToGlobalMappingRestoreIndices(cl2g,&idxs1);CHKERRQ(ierr);
          ierr = ISLocalToGlobalMappingRestoreIndices(rl2g,&idxs2);CHKERRQ(ierr);
          PetscCheckFalse(!same,PETSC_COMM_SELF,PETSC_ERR_SUP,"Cannot convert from MATNEST to MATIS! Matrix block (%" PetscInt_FMT ",%" PetscInt_FMT ") has invalid column l2gmap",j,i);
        }
      }
    }
  }

  B = NULL;
  if (reuse != MAT_REUSE_MATRIX) {
    PetscInt stl;

    /* Create l2g map for the rows of the new matrix and index sets for the local MATNEST */
    for (i=0,stl=0;i<nr;i++) stl += lr[i];
    ierr = PetscMalloc1(stl,&l2gidxs);CHKERRQ(ierr);
    for (i=0,stl=0;i<nr;i++) {
      Mat            usedmat;
      Mat_IS         *matis;
      const PetscInt *idxs;

      /* local IS for local NEST */
      ierr  = ISCreateStride(PETSC_COMM_SELF,lr[i],stl,1,&islrow[i]);CHKERRQ(ierr);

      /* l2gmap */
      j = 0;
      usedmat = nest[i][j];
      while (!usedmat && j < nc-1) usedmat = nest[i][++j];
      PetscCheckFalse(!usedmat,comm,PETSC_ERR_SUP,"Cannot find valid row mat");

      if (istrans[i*nc+j]) {
        Mat T;
        ierr    = MatTransposeGetMat(usedmat,&T);CHKERRQ(ierr);
        usedmat = T;
      }
      matis = (Mat_IS*)(usedmat->data);
      ierr  = ISGetIndices(isrow[i],&idxs);CHKERRQ(ierr);
      if (istrans[i*nc+j]) {
        ierr = PetscSFBcastBegin(matis->csf,MPIU_INT,idxs,l2gidxs+stl,MPI_REPLACE);CHKERRQ(ierr);
        ierr = PetscSFBcastEnd(matis->csf,MPIU_INT,idxs,l2gidxs+stl,MPI_REPLACE);CHKERRQ(ierr);
      } else {
        ierr = PetscSFBcastBegin(matis->sf,MPIU_INT,idxs,l2gidxs+stl,MPI_REPLACE);CHKERRQ(ierr);
        ierr = PetscSFBcastEnd(matis->sf,MPIU_INT,idxs,l2gidxs+stl,MPI_REPLACE);CHKERRQ(ierr);
      }
      ierr = ISRestoreIndices(isrow[i],&idxs);CHKERRQ(ierr);
      stl += lr[i];
    }
    ierr = ISLocalToGlobalMappingCreate(comm,1,stl,l2gidxs,PETSC_OWN_POINTER,&rl2g);CHKERRQ(ierr);

    /* Create l2g map for columns of the new matrix and index sets for the local MATNEST */
    for (i=0,stl=0;i<nc;i++) stl += lc[i];
    ierr = PetscMalloc1(stl,&l2gidxs);CHKERRQ(ierr);
    for (i=0,stl=0;i<nc;i++) {
      Mat            usedmat;
      Mat_IS         *matis;
      const PetscInt *idxs;

      /* local IS for local NEST */
      ierr  = ISCreateStride(PETSC_COMM_SELF,lc[i],stl,1,&islcol[i]);CHKERRQ(ierr);

      /* l2gmap */
      j = 0;
      usedmat = nest[j][i];
      while (!usedmat && j < nr-1) usedmat = nest[++j][i];
      PetscCheckFalse(!usedmat,comm,PETSC_ERR_SUP,"Cannot find valid column mat");
      if (istrans[j*nc+i]) {
        Mat T;
        ierr    = MatTransposeGetMat(usedmat,&T);CHKERRQ(ierr);
        usedmat = T;
      }
      matis = (Mat_IS*)(usedmat->data);
      ierr  = ISGetIndices(iscol[i],&idxs);CHKERRQ(ierr);
      if (istrans[j*nc+i]) {
        ierr = PetscSFBcastBegin(matis->sf,MPIU_INT,idxs,l2gidxs+stl,MPI_REPLACE);CHKERRQ(ierr);
        ierr = PetscSFBcastEnd(matis->sf,MPIU_INT,idxs,l2gidxs+stl,MPI_REPLACE);CHKERRQ(ierr);
      } else {
        ierr = PetscSFBcastBegin(matis->csf,MPIU_INT,idxs,l2gidxs+stl,MPI_REPLACE);CHKERRQ(ierr);
        ierr = PetscSFBcastEnd(matis->csf,MPIU_INT,idxs,l2gidxs+stl,MPI_REPLACE);CHKERRQ(ierr);
      }
      ierr = ISRestoreIndices(iscol[i],&idxs);CHKERRQ(ierr);
      stl += lc[i];
    }
    ierr = ISLocalToGlobalMappingCreate(comm,1,stl,l2gidxs,PETSC_OWN_POINTER,&cl2g);CHKERRQ(ierr);

    /* Create MATIS */
    ierr = MatCreate(comm,&B);CHKERRQ(ierr);
    ierr = MatSetSizes(B,A->rmap->n,A->cmap->n,A->rmap->N,A->cmap->N);CHKERRQ(ierr);
    ierr = MatGetBlockSizes(A,&rbs,&cbs);CHKERRQ(ierr);
    ierr = MatSetBlockSizes(B,rbs,cbs);CHKERRQ(ierr);
    ierr = MatSetType(B,MATIS);CHKERRQ(ierr);
    ierr = MatISSetLocalMatType(B,MATNEST);CHKERRQ(ierr);
    { /* hack : avoid setup of scatters */
      Mat_IS *matis = (Mat_IS*)(B->data);
      matis->islocalref = PETSC_TRUE;
    }
    ierr = MatSetLocalToGlobalMapping(B,rl2g,cl2g);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingDestroy(&rl2g);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingDestroy(&cl2g);CHKERRQ(ierr);
    ierr = MatCreateNest(PETSC_COMM_SELF,nr,islrow,nc,islcol,snest,&lA);CHKERRQ(ierr);
    ierr = MatNestSetVecType(lA,VECNEST);CHKERRQ(ierr);
    for (i=0;i<nr*nc;i++) {
      if (istrans[i]) {
        ierr = MatDestroy(&snest[i]);CHKERRQ(ierr);
      }
    }
    ierr = MatISSetLocalMat(B,lA);CHKERRQ(ierr);
    ierr = MatDestroy(&lA);CHKERRQ(ierr);
    { /* hack : setup of scatters done here */
      Mat_IS *matis = (Mat_IS*)(B->data);

      matis->islocalref = PETSC_FALSE;
      ierr = MatISSetUpScatters_Private(B);CHKERRQ(ierr);
    }
    ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    if (reuse == MAT_INPLACE_MATRIX) {
      ierr = MatHeaderReplace(A,&B);CHKERRQ(ierr);
    } else {
      *newmat = B;
    }
  } else {
    if (lreuse) {
      ierr = MatISGetLocalMat(*newmat,&lA);CHKERRQ(ierr);
      for (i=0;i<nr;i++) {
        for (j=0;j<nc;j++) {
          if (snest[i*nc+j]) {
            ierr = MatNestSetSubMat(lA,i,j,snest[i*nc+j]);CHKERRQ(ierr);
            if (istrans[i*nc+j]) {
              ierr = MatDestroy(&snest[i*nc+j]);CHKERRQ(ierr);
            }
          }
        }
      }
    } else {
      PetscInt stl;
      for (i=0,stl=0;i<nr;i++) {
        ierr  = ISCreateStride(PETSC_COMM_SELF,lr[i],stl,1,&islrow[i]);CHKERRQ(ierr);
        stl  += lr[i];
      }
      for (i=0,stl=0;i<nc;i++) {
        ierr  = ISCreateStride(PETSC_COMM_SELF,lc[i],stl,1,&islcol[i]);CHKERRQ(ierr);
        stl  += lc[i];
      }
      ierr = MatCreateNest(PETSC_COMM_SELF,nr,islrow,nc,islcol,snest,&lA);CHKERRQ(ierr);
      for (i=0;i<nr*nc;i++) {
        if (istrans[i]) {
          ierr = MatDestroy(&snest[i]);CHKERRQ(ierr);
        }
      }
      ierr = MatISSetLocalMat(*newmat,lA);CHKERRQ(ierr);
      ierr = MatDestroy(&lA);CHKERRQ(ierr);
    }
    ierr = MatAssemblyBegin(*newmat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(*newmat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }

  /* Create local matrix in MATNEST format */
  convert = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,((PetscObject)A)->prefix,"-matis_convert_local_nest",&convert,NULL);CHKERRQ(ierr);
  if (convert) {
    Mat              M;
    MatISLocalFields lf;
    PetscContainer   c;

    ierr = MatISGetLocalMat(*newmat,&lA);CHKERRQ(ierr);
    ierr = MatConvert(lA,MATAIJ,MAT_INITIAL_MATRIX,&M);CHKERRQ(ierr);
    ierr = MatISSetLocalMat(*newmat,M);CHKERRQ(ierr);
    ierr = MatDestroy(&M);CHKERRQ(ierr);

    /* attach local fields to the matrix */
    ierr = PetscNew(&lf);CHKERRQ(ierr);
    ierr = PetscMalloc2(nr,&lf->rf,nc,&lf->cf);CHKERRQ(ierr);
    for (i=0;i<nr;i++) {
      PetscInt n,st;

      ierr = ISGetLocalSize(islrow[i],&n);CHKERRQ(ierr);
      ierr = ISStrideGetInfo(islrow[i],&st,NULL);CHKERRQ(ierr);
      ierr = ISCreateStride(comm,n,st,1,&lf->rf[i]);CHKERRQ(ierr);
    }
    for (i=0;i<nc;i++) {
      PetscInt n,st;

      ierr = ISGetLocalSize(islcol[i],&n);CHKERRQ(ierr);
      ierr = ISStrideGetInfo(islcol[i],&st,NULL);CHKERRQ(ierr);
      ierr = ISCreateStride(comm,n,st,1,&lf->cf[i]);CHKERRQ(ierr);
    }
    lf->nr = nr;
    lf->nc = nc;
    ierr = PetscContainerCreate(PetscObjectComm((PetscObject)(*newmat)),&c);CHKERRQ(ierr);
    ierr = PetscContainerSetPointer(c,lf);CHKERRQ(ierr);
    ierr = PetscContainerSetUserDestroy(c,MatISContainerDestroyFields_Private);CHKERRQ(ierr);
    ierr = PetscObjectCompose((PetscObject)(*newmat),"_convert_nest_lfields",(PetscObject)c);CHKERRQ(ierr);
    ierr = PetscContainerDestroy(&c);CHKERRQ(ierr);
  }

  /* Free workspace */
  for (i=0;i<nr;i++) {
    ierr = ISDestroy(&islrow[i]);CHKERRQ(ierr);
  }
  for (i=0;i<nc;i++) {
    ierr = ISDestroy(&islcol[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree6(isrow,iscol,islrow,islcol,snest,istrans);CHKERRQ(ierr);
  ierr = PetscFree2(lr,lc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDiagonalScale_IS(Mat A, Vec l, Vec r)
{
  Mat_IS            *matis = (Mat_IS*)A->data;
  Vec               ll,rr;
  const PetscScalar *Y,*X;
  PetscScalar       *x,*y;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (l) {
    ll   = matis->y;
    ierr = VecGetArrayRead(l,&Y);CHKERRQ(ierr);
    ierr = VecGetArray(ll,&y);CHKERRQ(ierr);
    ierr = PetscSFBcastBegin(matis->sf,MPIU_SCALAR,Y,y,MPI_REPLACE);CHKERRQ(ierr);
  } else {
    ll = NULL;
  }
  if (r) {
    rr   = matis->x;
    ierr = VecGetArrayRead(r,&X);CHKERRQ(ierr);
    ierr = VecGetArray(rr,&x);CHKERRQ(ierr);
    ierr = PetscSFBcastBegin(matis->csf,MPIU_SCALAR,X,x,MPI_REPLACE);CHKERRQ(ierr);
  } else {
    rr = NULL;
  }
  if (ll) {
    ierr = PetscSFBcastEnd(matis->sf,MPIU_SCALAR,Y,y,MPI_REPLACE);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(l,&Y);CHKERRQ(ierr);
    ierr = VecRestoreArray(ll,&y);CHKERRQ(ierr);
  }
  if (rr) {
    ierr = PetscSFBcastEnd(matis->csf,MPIU_SCALAR,X,x,MPI_REPLACE);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(r,&X);CHKERRQ(ierr);
    ierr = VecRestoreArray(rr,&x);CHKERRQ(ierr);
  }
  ierr = MatDiagonalScale(matis->A,ll,rr);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatGetInfo_IS(Mat A,MatInfoType flag,MatInfo *ginfo)
{
  Mat_IS         *matis = (Mat_IS*)A->data;
  MatInfo        info;
  PetscLogDouble isend[6],irecv[6];
  PetscInt       bs;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatGetBlockSize(A,&bs);CHKERRQ(ierr);
  if (matis->A->ops->getinfo) {
    ierr     = MatGetInfo(matis->A,MAT_LOCAL,&info);CHKERRQ(ierr);
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
    ierr = MPIU_Allreduce(isend,irecv,6,MPIU_PETSCLOGDOUBLE,MPI_MAX,PetscObjectComm((PetscObject)A));CHKERRMPI(ierr);

    ginfo->nz_used      = irecv[0];
    ginfo->nz_allocated = irecv[1];
    ginfo->nz_unneeded  = irecv[2];
    ginfo->memory       = irecv[3];
    ginfo->mallocs      = irecv[4];
    ginfo->assemblies   = irecv[5];
  } else if (flag == MAT_GLOBAL_SUM) {
    ierr = MPIU_Allreduce(isend,irecv,5,MPIU_PETSCLOGDOUBLE,MPI_SUM,PetscObjectComm((PetscObject)A));CHKERRMPI(ierr);

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
  PetscErrorCode         ierr;

  PetscFunctionBegin;
  if (reuse == MAT_INITIAL_MATRIX || reuse == MAT_INPLACE_MATRIX) {
    ISLocalToGlobalMapping rl2g,cl2g;
    ierr = MatCreate(PetscObjectComm((PetscObject)A),&C);CHKERRQ(ierr);
    ierr = MatSetSizes(C,A->cmap->n,A->rmap->n,A->cmap->N,A->rmap->N);CHKERRQ(ierr);
    ierr = MatSetBlockSizes(C,PetscAbs(A->cmap->bs),PetscAbs(A->rmap->bs));CHKERRQ(ierr);
    ierr = MatSetType(C,MATIS);CHKERRQ(ierr);
    ierr = MatGetLocalToGlobalMapping(A,&rl2g,&cl2g);CHKERRQ(ierr);
    ierr = MatSetLocalToGlobalMapping(C,cl2g,rl2g);CHKERRQ(ierr);
  } else {
    C = *B;
  }

  /* perform local transposition */
  ierr = MatISGetLocalMat(A,&lA);CHKERRQ(ierr);
  ierr = MatTranspose(lA,MAT_INITIAL_MATRIX,&lC);CHKERRQ(ierr);
  ierr = MatISSetLocalMat(C,lC);CHKERRQ(ierr);
  ierr = MatDestroy(&lC);CHKERRQ(ierr);

  if (reuse == MAT_INITIAL_MATRIX || reuse == MAT_REUSE_MATRIX) {
    *B = C;
  } else {
    ierr = MatHeaderMerge(A,&C);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(*B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDiagonalSet_IS(Mat A,Vec D,InsertMode insmode)
{
  Mat_IS         *is = (Mat_IS*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (D) { /* MatShift_IS pass D = NULL */
    ierr = VecScatterBegin(is->rctx,D,is->y,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(is->rctx,D,is->y,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  }
  ierr = VecPointwiseDivide(is->y,is->y,is->counter);CHKERRQ(ierr);
  ierr = MatDiagonalSet(is->A,is->y,insmode);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatShift_IS(Mat A,PetscScalar a)
{
  Mat_IS         *is = (Mat_IS*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecSet(is->y,a);CHKERRQ(ierr);
  ierr = MatDiagonalSet_IS(A,NULL,ADD_VALUES);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSetValuesLocal_SubMat_IS(Mat A,PetscInt m,const PetscInt *rows, PetscInt n,const PetscInt *cols,const PetscScalar *values,InsertMode addv)
{
  PetscErrorCode ierr;
  PetscInt       rows_l[MATIS_MAX_ENTRIES_INSERTION],cols_l[MATIS_MAX_ENTRIES_INSERTION];

  PetscFunctionBegin;
  PetscCheckFalse(m > MATIS_MAX_ENTRIES_INSERTION || n > MATIS_MAX_ENTRIES_INSERTION,PETSC_COMM_SELF,PETSC_ERR_SUP,"Number of row/column indices must be <= %d: they are %" PetscInt_FMT " %" PetscInt_FMT,MATIS_MAX_ENTRIES_INSERTION,m,n);
  ierr = ISLocalToGlobalMappingApply(A->rmap->mapping,m,rows,rows_l);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingApply(A->cmap->mapping,n,cols,cols_l);CHKERRQ(ierr);
  ierr = MatSetValuesLocal_IS(A,m,rows_l,n,cols_l,values,addv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSetValuesBlockedLocal_SubMat_IS(Mat A,PetscInt m,const PetscInt *rows, PetscInt n,const PetscInt *cols,const PetscScalar *values,InsertMode addv)
{
  PetscErrorCode ierr;
  PetscInt       rows_l[MATIS_MAX_ENTRIES_INSERTION],cols_l[MATIS_MAX_ENTRIES_INSERTION];

  PetscFunctionBegin;
  PetscCheckFalse(m > MATIS_MAX_ENTRIES_INSERTION || n > MATIS_MAX_ENTRIES_INSERTION,PETSC_COMM_SELF,PETSC_ERR_SUP,"Number of row/column block indices must be <= %d: they are %" PetscInt_FMT " %" PetscInt_FMT,MATIS_MAX_ENTRIES_INSERTION,m,n);
  ierr = ISLocalToGlobalMappingApplyBlock(A->rmap->mapping,m,rows,rows_l);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingApplyBlock(A->cmap->mapping,n,cols,cols_l);CHKERRQ(ierr);
  ierr = MatSetValuesBlockedLocal_IS(A,m,rows_l,n,cols_l,values,addv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatCreateSubMatrix_IS(Mat mat,IS irow,IS icol,MatReuse scall,Mat *newmat)
{
  Mat               locmat,newlocmat;
  Mat_IS            *newmatis;
  const PetscInt    *idxs;
  PetscInt          i,m,n;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (scall == MAT_REUSE_MATRIX) {
    PetscBool ismatis;

    ierr = PetscObjectTypeCompare((PetscObject)*newmat,MATIS,&ismatis);CHKERRQ(ierr);
    PetscCheckFalse(!ismatis,PetscObjectComm((PetscObject)*newmat),PETSC_ERR_ARG_WRONG,"Cannot reuse matrix! Not of MATIS type");
    newmatis = (Mat_IS*)(*newmat)->data;
    PetscCheckFalse(!newmatis->getsub_ris,PetscObjectComm((PetscObject)*newmat),PETSC_ERR_ARG_WRONG,"Cannot reuse matrix! Misses local row IS");
    PetscCheckFalse(!newmatis->getsub_cis,PetscObjectComm((PetscObject)*newmat),PETSC_ERR_ARG_WRONG,"Cannot reuse matrix! Misses local col IS");
  }
  /* irow and icol may not have duplicate entries */
  if (PetscDefined(USE_DEBUG)) {
    Vec               rtest,ltest;
    const PetscScalar *array;

    ierr = MatCreateVecs(mat,&ltest,&rtest);CHKERRQ(ierr);
    ierr = ISGetLocalSize(irow,&n);CHKERRQ(ierr);
    ierr = ISGetIndices(irow,&idxs);CHKERRQ(ierr);
    for (i=0;i<n;i++) {
      ierr = VecSetValue(rtest,idxs[i],1.0,ADD_VALUES);CHKERRQ(ierr);
    }
    ierr = VecAssemblyBegin(rtest);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(rtest);CHKERRQ(ierr);
    ierr = VecGetLocalSize(rtest,&n);CHKERRQ(ierr);
    ierr = VecGetOwnershipRange(rtest,&m,NULL);CHKERRQ(ierr);
    ierr = VecGetArrayRead(rtest,&array);CHKERRQ(ierr);
    for (i=0;i<n;i++) PetscCheckFalse(array[i] != 0. && array[i] != 1.,PETSC_COMM_SELF,PETSC_ERR_SUP,"Index %" PetscInt_FMT " counted %" PetscInt_FMT " times! Irow may not have duplicate entries",i+m,(PetscInt)PetscRealPart(array[i]));
    ierr = VecRestoreArrayRead(rtest,&array);CHKERRQ(ierr);
    ierr = ISRestoreIndices(irow,&idxs);CHKERRQ(ierr);
    ierr = ISGetLocalSize(icol,&n);CHKERRQ(ierr);
    ierr = ISGetIndices(icol,&idxs);CHKERRQ(ierr);
    for (i=0;i<n;i++) {
      ierr = VecSetValue(ltest,idxs[i],1.0,ADD_VALUES);CHKERRQ(ierr);
    }
    ierr = VecAssemblyBegin(ltest);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(ltest);CHKERRQ(ierr);
    ierr = VecGetLocalSize(ltest,&n);CHKERRQ(ierr);
    ierr = VecGetOwnershipRange(ltest,&m,NULL);CHKERRQ(ierr);
    ierr = VecGetArrayRead(ltest,&array);CHKERRQ(ierr);
    for (i=0;i<n;i++) PetscCheckFalse(array[i] != 0. && array[i] != 1.,PETSC_COMM_SELF,PETSC_ERR_SUP,"Index %" PetscInt_FMT " counted %" PetscInt_FMT " times! Icol may not have duplicate entries",i+m,(PetscInt)PetscRealPart(array[i]));
    ierr = VecRestoreArrayRead(ltest,&array);CHKERRQ(ierr);
    ierr = ISRestoreIndices(icol,&idxs);CHKERRQ(ierr);
    ierr = VecDestroy(&rtest);CHKERRQ(ierr);
    ierr = VecDestroy(&ltest);CHKERRQ(ierr);
  }
  if (scall == MAT_INITIAL_MATRIX) {
    Mat_IS                 *matis = (Mat_IS*)mat->data;
    ISLocalToGlobalMapping rl2g;
    IS                     is;
    PetscInt               *lidxs,*lgidxs,*newgidxs;
    PetscInt               ll,newloc,irbs,icbs,arbs,acbs,rbs,cbs;
    PetscBool              cong;
    MPI_Comm               comm;

    ierr = PetscObjectGetComm((PetscObject)mat,&comm);CHKERRQ(ierr);
    ierr = MatGetBlockSizes(mat,&arbs,&acbs);CHKERRQ(ierr);
    ierr = ISGetBlockSize(irow,&irbs);CHKERRQ(ierr);
    ierr = ISGetBlockSize(icol,&icbs);CHKERRQ(ierr);
    rbs  = arbs == irbs ? irbs : 1;
    cbs  = acbs == icbs ? icbs : 1;
    ierr = ISGetLocalSize(irow,&m);CHKERRQ(ierr);
    ierr = ISGetLocalSize(icol,&n);CHKERRQ(ierr);
    ierr = MatCreate(comm,newmat);CHKERRQ(ierr);
    ierr = MatSetType(*newmat,MATIS);CHKERRQ(ierr);
    ierr = MatSetSizes(*newmat,m,n,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
    ierr = MatSetBlockSizes(*newmat,rbs,cbs);CHKERRQ(ierr);
    /* communicate irow to their owners in the layout */
    ierr = ISGetIndices(irow,&idxs);CHKERRQ(ierr);
    ierr = PetscLayoutMapLocal(mat->rmap,m,idxs,&ll,&lidxs,&lgidxs);CHKERRQ(ierr);
    ierr = ISRestoreIndices(irow,&idxs);CHKERRQ(ierr);
    ierr = PetscArrayzero(matis->sf_rootdata,matis->sf->nroots);CHKERRQ(ierr);
    for (i=0;i<ll;i++) matis->sf_rootdata[lidxs[i]] = lgidxs[i]+1;
    ierr = PetscFree(lidxs);CHKERRQ(ierr);
    ierr = PetscFree(lgidxs);CHKERRQ(ierr);
    ierr = PetscSFBcastBegin(matis->sf,MPIU_INT,matis->sf_rootdata,matis->sf_leafdata,MPI_REPLACE);CHKERRQ(ierr);
    ierr = PetscSFBcastEnd(matis->sf,MPIU_INT,matis->sf_rootdata,matis->sf_leafdata,MPI_REPLACE);CHKERRQ(ierr);
    for (i=0,newloc=0;i<matis->sf->nleaves;i++) if (matis->sf_leafdata[i]) newloc++;
    ierr = PetscMalloc1(newloc,&newgidxs);CHKERRQ(ierr);
    ierr = PetscMalloc1(newloc,&lidxs);CHKERRQ(ierr);
    for (i=0,newloc=0;i<matis->sf->nleaves;i++)
      if (matis->sf_leafdata[i]) {
        lidxs[newloc] = i;
        newgidxs[newloc++] = matis->sf_leafdata[i]-1;
      }
    ierr = ISCreateGeneral(comm,newloc,newgidxs,PETSC_OWN_POINTER,&is);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingCreateIS(is,&rl2g);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingSetBlockSize(rl2g,rbs);CHKERRQ(ierr);
    ierr = ISDestroy(&is);CHKERRQ(ierr);
    /* local is to extract local submatrix */
    newmatis = (Mat_IS*)(*newmat)->data;
    ierr = ISCreateGeneral(comm,newloc,lidxs,PETSC_OWN_POINTER,&newmatis->getsub_ris);CHKERRQ(ierr);
    ierr = MatHasCongruentLayouts(mat,&cong);CHKERRQ(ierr);
    if (cong && irow == icol && matis->csf == matis->sf) {
      ierr = MatSetLocalToGlobalMapping(*newmat,rl2g,rl2g);CHKERRQ(ierr);
      ierr = PetscObjectReference((PetscObject)newmatis->getsub_ris);CHKERRQ(ierr);
      newmatis->getsub_cis = newmatis->getsub_ris;
    } else {
      ISLocalToGlobalMapping cl2g;

      /* communicate icol to their owners in the layout */
      ierr = ISGetIndices(icol,&idxs);CHKERRQ(ierr);
      ierr = PetscLayoutMapLocal(mat->cmap,n,idxs,&ll,&lidxs,&lgidxs);CHKERRQ(ierr);
      ierr = ISRestoreIndices(icol,&idxs);CHKERRQ(ierr);
      ierr = PetscArrayzero(matis->csf_rootdata,matis->csf->nroots);CHKERRQ(ierr);
      for (i=0;i<ll;i++) matis->csf_rootdata[lidxs[i]] = lgidxs[i]+1;
      ierr = PetscFree(lidxs);CHKERRQ(ierr);
      ierr = PetscFree(lgidxs);CHKERRQ(ierr);
      ierr = PetscSFBcastBegin(matis->csf,MPIU_INT,matis->csf_rootdata,matis->csf_leafdata,MPI_REPLACE);CHKERRQ(ierr);
      ierr = PetscSFBcastEnd(matis->csf,MPIU_INT,matis->csf_rootdata,matis->csf_leafdata,MPI_REPLACE);CHKERRQ(ierr);
      for (i=0,newloc=0;i<matis->csf->nleaves;i++) if (matis->csf_leafdata[i]) newloc++;
      ierr = PetscMalloc1(newloc,&newgidxs);CHKERRQ(ierr);
      ierr = PetscMalloc1(newloc,&lidxs);CHKERRQ(ierr);
      for (i=0,newloc=0;i<matis->csf->nleaves;i++)
        if (matis->csf_leafdata[i]) {
          lidxs[newloc] = i;
          newgidxs[newloc++] = matis->csf_leafdata[i]-1;
        }
      ierr = ISCreateGeneral(comm,newloc,newgidxs,PETSC_OWN_POINTER,&is);CHKERRQ(ierr);
      ierr = ISLocalToGlobalMappingCreateIS(is,&cl2g);CHKERRQ(ierr);
      ierr = ISLocalToGlobalMappingSetBlockSize(cl2g,cbs);CHKERRQ(ierr);
      ierr = ISDestroy(&is);CHKERRQ(ierr);
      /* local is to extract local submatrix */
      ierr = ISCreateGeneral(comm,newloc,lidxs,PETSC_OWN_POINTER,&newmatis->getsub_cis);CHKERRQ(ierr);
      ierr = MatSetLocalToGlobalMapping(*newmat,rl2g,cl2g);CHKERRQ(ierr);
      ierr = ISLocalToGlobalMappingDestroy(&cl2g);CHKERRQ(ierr);
    }
    ierr = ISLocalToGlobalMappingDestroy(&rl2g);CHKERRQ(ierr);
  } else {
    ierr = MatISGetLocalMat(*newmat,&newlocmat);CHKERRQ(ierr);
  }
  ierr = MatISGetLocalMat(mat,&locmat);CHKERRQ(ierr);
  newmatis = (Mat_IS*)(*newmat)->data;
  ierr = MatCreateSubMatrix(locmat,newmatis->getsub_ris,newmatis->getsub_cis,scall,&newlocmat);CHKERRQ(ierr);
  if (scall == MAT_INITIAL_MATRIX) {
    ierr = MatISSetLocalMat(*newmat,newlocmat);CHKERRQ(ierr);
    ierr = MatDestroy(&newlocmat);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(*newmat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*newmat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatCopy_IS(Mat A,Mat B,MatStructure str)
{
  Mat_IS         *a = (Mat_IS*)A->data,*b;
  PetscBool      ismatis;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)B,MATIS,&ismatis);CHKERRQ(ierr);
  PetscCheckFalse(!ismatis,PetscObjectComm((PetscObject)B),PETSC_ERR_SUP,"Need to be implemented");
  b = (Mat_IS*)B->data;
  ierr = MatCopy(a->A,b->A,str);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMissingDiagonal_IS(Mat A,PetscBool  *missing,PetscInt *d)
{
  Vec               v;
  const PetscScalar *array;
  PetscInt          i,n;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  *missing = PETSC_FALSE;
  ierr = MatCreateVecs(A,NULL,&v);CHKERRQ(ierr);
  ierr = MatGetDiagonal(A,v);CHKERRQ(ierr);
  ierr = VecGetLocalSize(v,&n);CHKERRQ(ierr);
  ierr = VecGetArrayRead(v,&array);CHKERRQ(ierr);
  for (i=0;i<n;i++) if (array[i] == 0.) break;
  ierr = VecRestoreArrayRead(v,&array);CHKERRQ(ierr);
  ierr = VecDestroy(&v);CHKERRQ(ierr);
  if (i != n) *missing = PETSC_TRUE;
  if (d) {
    *d = -1;
    if (*missing) {
      PetscInt rstart;
      ierr = MatGetOwnershipRange(A,&rstart,NULL);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (matis->sf) PetscFunctionReturn(0);
  ierr = PetscSFCreate(PetscObjectComm((PetscObject)B),&matis->sf);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingGetIndices(B->rmap->mapping,&gidxs);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingGetSize(B->rmap->mapping,&nleaves);CHKERRQ(ierr);
  ierr = PetscSFSetGraphLayout(matis->sf,B->rmap,nleaves,NULL,PETSC_OWN_POINTER,gidxs);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingRestoreIndices(B->rmap->mapping,&gidxs);CHKERRQ(ierr);
  ierr = PetscMalloc2(matis->sf->nroots,&matis->sf_rootdata,matis->sf->nleaves,&matis->sf_leafdata);CHKERRQ(ierr);
  if (B->rmap->mapping != B->cmap->mapping) { /* setup SF for columns */
    ierr = ISLocalToGlobalMappingGetSize(B->cmap->mapping,&nleaves);CHKERRQ(ierr);
    ierr = PetscSFCreate(PetscObjectComm((PetscObject)B),&matis->csf);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingGetIndices(B->cmap->mapping,&gidxs);CHKERRQ(ierr);
    ierr = PetscSFSetGraphLayout(matis->csf,B->cmap,nleaves,NULL,PETSC_OWN_POINTER,gidxs);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingRestoreIndices(B->cmap->mapping,&gidxs);CHKERRQ(ierr);
    ierr = PetscMalloc2(matis->csf->nroots,&matis->csf_rootdata,matis->csf->nleaves,&matis->csf_leafdata);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidType(A,1);
  PetscValidLogicalCollectiveBool(A,store,2);
  ierr = PetscTryMethod(A,"MatISStoreL2L_C",(Mat,PetscBool),(A,store));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatISStoreL2L_IS(Mat A, PetscBool store)
{
  Mat_IS         *matis = (Mat_IS*)(A->data);
  PetscErrorCode ierr;

  PetscFunctionBegin;
  matis->storel2l = store;
  if (!store) {
    ierr = PetscObjectCompose((PetscObject)(A),"_MatIS_PtAP_l2l",NULL);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidType(A,1);
  PetscValidLogicalCollectiveBool(A,fix,2);
  ierr = PetscTryMethod(A,"MatISFixLocalEmpty_C",(Mat,PetscBool),(A,fix));CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(B,MAT_CLASSID,1);
  PetscValidType(B,1);
  ierr = PetscTryMethod(B,"MatISSetPreallocation_C",(Mat,PetscInt,const PetscInt[],PetscInt,const PetscInt[]),(B,d_nz,d_nnz,o_nz,o_nnz));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* this is used by DMDA */
PETSC_EXTERN PetscErrorCode MatISSetPreallocation_IS(Mat B,PetscInt d_nz,const PetscInt d_nnz[],PetscInt o_nz,const PetscInt o_nnz[])
{
  Mat_IS         *matis = (Mat_IS*)(B->data);
  PetscInt       bs,i,nlocalcols;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatSetUp(B);CHKERRQ(ierr);
  if (!d_nnz) for (i=0;i<matis->sf->nroots;i++) matis->sf_rootdata[i] = d_nz;
  else for (i=0;i<matis->sf->nroots;i++) matis->sf_rootdata[i] = d_nnz[i];

  if (!o_nnz) for (i=0;i<matis->sf->nroots;i++) matis->sf_rootdata[i] += o_nz;
  else for (i=0;i<matis->sf->nroots;i++) matis->sf_rootdata[i] += o_nnz[i];

  ierr = PetscSFBcastBegin(matis->sf,MPIU_INT,matis->sf_rootdata,matis->sf_leafdata,MPI_REPLACE);CHKERRQ(ierr);
  ierr = MatGetSize(matis->A,NULL,&nlocalcols);CHKERRQ(ierr);
  ierr = MatGetBlockSize(matis->A,&bs);CHKERRQ(ierr);
  ierr = PetscSFBcastEnd(matis->sf,MPIU_INT,matis->sf_rootdata,matis->sf_leafdata,MPI_REPLACE);CHKERRQ(ierr);

  for (i=0;i<matis->sf->nleaves;i++) matis->sf_leafdata[i] = PetscMin(matis->sf_leafdata[i],nlocalcols);
  ierr = MatSeqAIJSetPreallocation(matis->A,0,matis->sf_leafdata);CHKERRQ(ierr);
#if defined(PETSC_HAVE_HYPRE)
  ierr = MatHYPRESetPreallocation(matis->A,0,matis->sf_leafdata,0,NULL);CHKERRQ(ierr);
#endif

  for (i=0;i<matis->sf->nleaves/bs;i++) {
    PetscInt b;

    matis->sf_leafdata[i] = matis->sf_leafdata[i*bs]/bs;
    for (b=1;b<bs;b++) {
      matis->sf_leafdata[i] = PetscMax(matis->sf_leafdata[i],matis->sf_leafdata[i*bs+b]/bs);
    }
  }
  ierr = MatSeqBAIJSetPreallocation(matis->A,bs,0,matis->sf_leafdata);CHKERRQ(ierr);

  nlocalcols /= bs;
  for (i=0;i<matis->sf->nleaves/bs;i++) matis->sf_leafdata[i] = PetscMin(matis->sf_leafdata[i],nlocalcols - i);
  ierr = MatSeqSBAIJSetPreallocation(matis->A,bs,0,matis->sf_leafdata);CHKERRQ(ierr);

  /* for other matrix types */
  ierr = MatSetUp(matis->A);CHKERRQ(ierr);
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
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)A),&size);CHKERRMPI(ierr);
  ierr = MatGetSize(A,&rows,&cols);CHKERRQ(ierr);
  ierr = MatGetBlockSize(A,&bs);CHKERRQ(ierr);
  ierr = MatGetSize(matis->A,&local_rows,&local_cols);CHKERRQ(ierr);
  ierr = PetscObjectBaseTypeCompare((PetscObject)matis->A,MATSEQDENSE,&isdense);CHKERRQ(ierr);
  ierr = PetscObjectBaseTypeCompare((PetscObject)matis->A,MATSEQSBAIJ,&issbaij);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingGetIndices(A->rmap->mapping,&global_indices_r);CHKERRQ(ierr);
  if (A->rmap->mapping != A->cmap->mapping) {
    ierr = ISLocalToGlobalMappingGetIndices(A->cmap->mapping,&global_indices_c);CHKERRQ(ierr);
  } else {
    global_indices_c = global_indices_r;
  }

  if (issbaij) {
    ierr = MatGetRowUpperTriangular(matis->A);CHKERRQ(ierr);
  }
  /*
     An SF reduce is needed to sum up properly on shared rows.
     Note that generally preallocation is not exact, since it overestimates nonzeros
  */
  ierr = MatGetLocalSize(A,&lrows,&lcols);CHKERRQ(ierr);
  ierr = MatPreallocateInitialize(PetscObjectComm((PetscObject)A),lrows,lcols,dnz,onz);CHKERRQ(ierr);
  /* All processes need to compute entire row ownership */
  ierr = PetscMalloc1(rows,&row_ownership);CHKERRQ(ierr);
  ierr = MatGetOwnershipRanges(A,(const PetscInt**)&mat_ranges);CHKERRQ(ierr);
  for (i=0;i<size;i++) {
    for (j=mat_ranges[i];j<mat_ranges[i+1];j++) {
      row_ownership[j] = i;
    }
  }
  ierr = MatGetOwnershipRangesColumn(A,(const PetscInt**)&mat_ranges);CHKERRQ(ierr);

  /*
     my_dnz and my_onz contains exact contribution to preallocation from each local mat
     then, they will be summed up properly. This way, preallocation is always sufficient
  */
  ierr = PetscCalloc2(local_rows,&my_dnz,local_rows,&my_onz);CHKERRQ(ierr);
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
    ierr = MatGetRowIJ(matis->A,0,PETSC_FALSE,PETSC_FALSE,&local_rows,&ii,&jj,&done);CHKERRQ(ierr);
    PetscCheckFalse(!done,PetscObjectComm((PetscObject)(matis->A)),PETSC_ERR_PLIB,"Error in MatGetRowIJ");
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
    ierr = MatRestoreRowIJ(matis->A,0,PETSC_FALSE,PETSC_FALSE,&local_rows,&ii,&jj,&done);CHKERRQ(ierr);
    PetscCheckFalse(!done,PetscObjectComm((PetscObject)(matis->A)),PETSC_ERR_PLIB,"Error in MatRestoreRowIJ");
  } else { /* loop over rows and use MatGetRow */
    for (i=0;i<local_rows;i++) {
      const PetscInt *cols;
      PetscInt       ncols,index_row = global_indices_r[i];
      ierr = MatGetRow(matis->A,i,&ncols,&cols,NULL);CHKERRQ(ierr);
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
      ierr = MatRestoreRow(matis->A,i,&ncols,&cols,NULL);CHKERRQ(ierr);
    }
  }
  if (global_indices_c != global_indices_r) {
    ierr = ISLocalToGlobalMappingRestoreIndices(A->cmap->mapping,&global_indices_c);CHKERRQ(ierr);
  }
  ierr = ISLocalToGlobalMappingRestoreIndices(A->rmap->mapping,&global_indices_r);CHKERRQ(ierr);
  ierr = PetscFree(row_ownership);CHKERRQ(ierr);

  /* Reduce my_dnz and my_onz */
  if (maxreduce) {
    ierr = PetscSFReduceBegin(matis->sf,MPIU_INT,my_dnz,dnz,MPI_MAX);CHKERRQ(ierr);
    ierr = PetscSFReduceBegin(matis->sf,MPIU_INT,my_onz,onz,MPI_MAX);CHKERRQ(ierr);
    ierr = PetscSFReduceEnd(matis->sf,MPIU_INT,my_dnz,dnz,MPI_MAX);CHKERRQ(ierr);
    ierr = PetscSFReduceEnd(matis->sf,MPIU_INT,my_onz,onz,MPI_MAX);CHKERRQ(ierr);
  } else {
    ierr = PetscSFReduceBegin(matis->sf,MPIU_INT,my_dnz,dnz,MPI_SUM);CHKERRQ(ierr);
    ierr = PetscSFReduceBegin(matis->sf,MPIU_INT,my_onz,onz,MPI_SUM);CHKERRQ(ierr);
    ierr = PetscSFReduceEnd(matis->sf,MPIU_INT,my_dnz,dnz,MPI_SUM);CHKERRQ(ierr);
    ierr = PetscSFReduceEnd(matis->sf,MPIU_INT,my_onz,onz,MPI_SUM);CHKERRQ(ierr);
  }
  ierr = PetscFree2(my_dnz,my_onz);CHKERRQ(ierr);

  /* Resize preallocation if overestimated */
  for (i=0;i<lrows;i++) {
    dnz[i] = PetscMin(dnz[i],lcols);
    onz[i] = PetscMin(onz[i],cols-lcols);
  }

  /* Set preallocation */
  ierr = MatSeqAIJSetPreallocation(B,0,dnz);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(B,0,dnz,0,onz);CHKERRQ(ierr);
  for (i=0;i<lrows;i+=bs) {
    PetscInt b, d = dnz[i],o = onz[i];

    for (b=1;b<bs;b++) {
      d = PetscMax(d,dnz[i+b]);
      o = PetscMax(o,onz[i+b]);
    }
    dnz[i/bs] = PetscMin(d/bs + d%bs,lcols/bs);
    onz[i/bs] = PetscMin(o/bs + o%bs,(cols-lcols)/bs);
  }
  ierr = MatSeqBAIJSetPreallocation(B,bs,0,dnz);CHKERRQ(ierr);
  ierr = MatMPIBAIJSetPreallocation(B,bs,0,dnz,0,onz);CHKERRQ(ierr);
  ierr = MatMPISBAIJSetPreallocation(B,bs,0,dnz,0,onz);CHKERRQ(ierr);
  ierr = MatPreallocateFinalize(dnz,onz);CHKERRQ(ierr);
  if (issbaij) {
    ierr = MatRestoreRowUpperTriangular(matis->A);CHKERRQ(ierr);
  }
  ierr = MatSetOption(B,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);
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
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)mat),&size);CHKERRMPI(ierr);
  if (size == 1 && mat->rmap->N == matis->A->rmap->N && mat->cmap->N == matis->A->cmap->N) {
    Mat      B;
    IS       irows = NULL,icols = NULL;
    PetscInt rbs,cbs;

    ierr = ISLocalToGlobalMappingGetBlockSize(mat->rmap->mapping,&rbs);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingGetBlockSize(mat->cmap->mapping,&cbs);CHKERRQ(ierr);
    if (reuse != MAT_REUSE_MATRIX) { /* check if l2g maps are one-to-one */
      IS             rows,cols;
      const PetscInt *ridxs,*cidxs;
      PetscInt       i,nw,*work;

      ierr = ISLocalToGlobalMappingGetBlockIndices(mat->rmap->mapping,&ridxs);CHKERRQ(ierr);
      ierr = ISLocalToGlobalMappingGetSize(mat->rmap->mapping,&nw);CHKERRQ(ierr);
      nw   = nw/rbs;
      ierr = PetscCalloc1(nw,&work);CHKERRQ(ierr);
      for (i=0;i<nw;i++) work[ridxs[i]] += 1;
      for (i=0;i<nw;i++) if (!work[i] || work[i] > 1) break;
      if (i == nw) {
        ierr = ISCreateBlock(PETSC_COMM_SELF,rbs,nw,ridxs,PETSC_USE_POINTER,&rows);CHKERRQ(ierr);
        ierr = ISSetPermutation(rows);CHKERRQ(ierr);
        ierr = ISInvertPermutation(rows,PETSC_DECIDE,&irows);CHKERRQ(ierr);
        ierr = ISDestroy(&rows);CHKERRQ(ierr);
      }
      ierr = ISLocalToGlobalMappingRestoreBlockIndices(mat->rmap->mapping,&ridxs);CHKERRQ(ierr);
      ierr = PetscFree(work);CHKERRQ(ierr);
      if (irows && mat->rmap->mapping != mat->cmap->mapping) {
        ierr = ISLocalToGlobalMappingGetBlockIndices(mat->cmap->mapping,&cidxs);CHKERRQ(ierr);
        ierr = ISLocalToGlobalMappingGetSize(mat->cmap->mapping,&nw);CHKERRQ(ierr);
        nw   = nw/cbs;
        ierr = PetscCalloc1(nw,&work);CHKERRQ(ierr);
        for (i=0;i<nw;i++) work[cidxs[i]] += 1;
        for (i=0;i<nw;i++) if (!work[i] || work[i] > 1) break;
        if (i == nw) {
          ierr = ISCreateBlock(PETSC_COMM_SELF,cbs,nw,cidxs,PETSC_USE_POINTER,&cols);CHKERRQ(ierr);
          ierr = ISSetPermutation(cols);CHKERRQ(ierr);
          ierr = ISInvertPermutation(cols,PETSC_DECIDE,&icols);CHKERRQ(ierr);
          ierr = ISDestroy(&cols);CHKERRQ(ierr);
        }
        ierr = ISLocalToGlobalMappingRestoreBlockIndices(mat->cmap->mapping,&cidxs);CHKERRQ(ierr);
        ierr = PetscFree(work);CHKERRQ(ierr);
      } else if (irows) {
        ierr  = PetscObjectReference((PetscObject)irows);CHKERRQ(ierr);
        icols = irows;
      }
    } else {
      ierr = PetscObjectQuery((PetscObject)(*M),"_MatIS_IS_XAIJ_irows",(PetscObject*)&irows);CHKERRQ(ierr);
      ierr = PetscObjectQuery((PetscObject)(*M),"_MatIS_IS_XAIJ_icols",(PetscObject*)&icols);CHKERRQ(ierr);
      if (irows) { ierr = PetscObjectReference((PetscObject)irows);CHKERRQ(ierr); }
      if (icols) { ierr = PetscObjectReference((PetscObject)icols);CHKERRQ(ierr); }
    }
    if (!irows || !icols) {
      ierr = ISDestroy(&icols);CHKERRQ(ierr);
      ierr = ISDestroy(&irows);CHKERRQ(ierr);
      goto general_assembly;
    }
    ierr = MatConvert(matis->A,mtype,MAT_INITIAL_MATRIX,&B);CHKERRQ(ierr);
    if (reuse != MAT_INPLACE_MATRIX) {
      ierr = MatCreateSubMatrix(B,irows,icols,reuse,M);CHKERRQ(ierr);
      ierr = PetscObjectCompose((PetscObject)(*M),"_MatIS_IS_XAIJ_irows",(PetscObject)irows);CHKERRQ(ierr);
      ierr = PetscObjectCompose((PetscObject)(*M),"_MatIS_IS_XAIJ_icols",(PetscObject)icols);CHKERRQ(ierr);
    } else {
      Mat C;

      ierr = MatCreateSubMatrix(B,irows,icols,MAT_INITIAL_MATRIX,&C);CHKERRQ(ierr);
      ierr = MatHeaderReplace(mat,&C);CHKERRQ(ierr);
    }
    ierr = MatDestroy(&B);CHKERRQ(ierr);
    ierr = ISDestroy(&icols);CHKERRQ(ierr);
    ierr = ISDestroy(&irows);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
general_assembly:
  ierr = MatGetSize(mat,&rows,&cols);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingGetBlockSize(mat->rmap->mapping,&rbs);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingGetBlockSize(mat->cmap->mapping,&cbs);CHKERRQ(ierr);
  ierr = MatGetLocalSize(mat,&lrows,&lcols);CHKERRQ(ierr);
  ierr = MatGetSize(matis->A,&local_rows,&local_cols);CHKERRQ(ierr);
  ierr = PetscObjectBaseTypeCompare((PetscObject)matis->A,MATSEQDENSE,&isseqdense);CHKERRQ(ierr);
  ierr = PetscObjectBaseTypeCompare((PetscObject)matis->A,MATSEQAIJ,&isseqaij);CHKERRQ(ierr);
  ierr = PetscObjectBaseTypeCompare((PetscObject)matis->A,MATSEQBAIJ,&isseqbaij);CHKERRQ(ierr);
  ierr = PetscObjectBaseTypeCompare((PetscObject)matis->A,MATSEQSBAIJ,&isseqsbaij);CHKERRQ(ierr);
  PetscCheckFalse(!isseqdense && !isseqaij && !isseqbaij && !isseqsbaij,PETSC_COMM_SELF,PETSC_ERR_SUP,"Not for matrix type %s",((PetscObject)(matis->A))->type_name);
  if (PetscDefined (USE_DEBUG)) {
    PetscBool         lb[4],bb[4];

    lb[0] = isseqdense;
    lb[1] = isseqaij;
    lb[2] = isseqbaij;
    lb[3] = isseqsbaij;
    ierr = MPIU_Allreduce(lb,bb,4,MPIU_BOOL,MPI_LAND,PetscObjectComm((PetscObject)mat));CHKERRMPI(ierr);
    PetscCheckFalse(!bb[0] && !bb[1] && !bb[2] && !bb[3],PETSC_COMM_SELF,PETSC_ERR_SUP,"Local matrices must have the same type");
  }

  if (reuse != MAT_REUSE_MATRIX) {
    ierr = MatCreate(PetscObjectComm((PetscObject)mat),&MT);CHKERRQ(ierr);
    ierr = MatSetSizes(MT,lrows,lcols,rows,cols);CHKERRQ(ierr);
    ierr = MatSetType(MT,mtype);CHKERRQ(ierr);
    ierr = MatSetBlockSizes(MT,rbs,cbs);CHKERRQ(ierr);
    ierr = MatISSetMPIXAIJPreallocation_Private(mat,MT,PETSC_FALSE);CHKERRQ(ierr);
  } else {
    PetscInt mrbs,mcbs,mrows,mcols,mlrows,mlcols;

    /* some checks */
    MT   = *M;
    ierr = MatGetBlockSizes(MT,&mrbs,&mcbs);CHKERRQ(ierr);
    ierr = MatGetSize(MT,&mrows,&mcols);CHKERRQ(ierr);
    ierr = MatGetLocalSize(MT,&mlrows,&mlcols);CHKERRQ(ierr);
    PetscCheckFalse(mrows != rows,PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Cannot reuse matrix. Wrong number of rows (%" PetscInt_FMT " != %" PetscInt_FMT ")",rows,mrows);
    PetscCheckFalse(mcols != cols,PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Cannot reuse matrix. Wrong number of cols (%" PetscInt_FMT " != %" PetscInt_FMT ")",cols,mcols);
    PetscCheckFalse(mlrows != lrows,PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Cannot reuse matrix. Wrong number of local rows (%" PetscInt_FMT " != %" PetscInt_FMT ")",lrows,mlrows);
    PetscCheckFalse(mlcols != lcols,PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Cannot reuse matrix. Wrong number of local cols (%" PetscInt_FMT " != %" PetscInt_FMT ")",lcols,mlcols);
    PetscCheckFalse(mrbs != rbs,PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Cannot reuse matrix. Wrong row block size (%" PetscInt_FMT " != %" PetscInt_FMT ")",rbs,mrbs);
    PetscCheckFalse(mcbs != cbs,PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Cannot reuse matrix. Wrong col block size (%" PetscInt_FMT " != %" PetscInt_FMT ")",cbs,mcbs);
    ierr = MatZeroEntries(MT);CHKERRQ(ierr);
  }

  if (isseqsbaij || isseqbaij) {
    ierr = MatConvert(matis->A,MATSEQAIJ,MAT_INITIAL_MATRIX,&local_mat);CHKERRQ(ierr);
    isseqaij = PETSC_TRUE;
  } else {
    ierr = PetscObjectReference((PetscObject)matis->A);CHKERRQ(ierr);
    local_mat = matis->A;
  }

  /* Set values */
  ierr = MatSetLocalToGlobalMapping(MT,mat->rmap->mapping,mat->cmap->mapping);CHKERRQ(ierr);
  if (isseqdense) { /* special case for dense local matrices */
    PetscInt          i,*dummy;

    ierr = PetscMalloc1(PetscMax(local_rows,local_cols),&dummy);CHKERRQ(ierr);
    for (i=0;i<PetscMax(local_rows,local_cols);i++) dummy[i] = i;
    ierr = MatSetOption(MT,MAT_ROW_ORIENTED,PETSC_FALSE);CHKERRQ(ierr);
    ierr = MatDenseGetArrayRead(local_mat,&array);CHKERRQ(ierr);
    ierr = MatSetValuesLocal(MT,local_rows,dummy,local_cols,dummy,array,ADD_VALUES);CHKERRQ(ierr);
    ierr = MatDenseRestoreArrayRead(local_mat,&array);CHKERRQ(ierr);
    ierr = PetscFree(dummy);CHKERRQ(ierr);
  } else if (isseqaij) {
    const PetscInt *blocks;
    PetscInt       i,nvtxs,*xadj,*adjncy, nb;
    PetscBool      done;
    PetscScalar    *sarray;

    ierr = MatGetRowIJ(local_mat,0,PETSC_FALSE,PETSC_FALSE,&nvtxs,(const PetscInt**)&xadj,(const PetscInt**)&adjncy,&done);CHKERRQ(ierr);
    PetscCheckFalse(!done,PetscObjectComm((PetscObject)local_mat),PETSC_ERR_PLIB,"Error in MatGetRowIJ");
    ierr = MatSeqAIJGetArray(local_mat,&sarray);CHKERRQ(ierr);
    ierr = MatGetVariableBlockSizes(local_mat,&nb,&blocks);CHKERRQ(ierr);
    if (nb) { /* speed up assembly for special blocked matrices (used by BDDC) */
      PetscInt sum;

      for (i=0,sum=0;i<nb;i++) sum += blocks[i];
      if (sum == nvtxs) {
        PetscInt r;

        for (i=0,r=0;i<nb;i++) {
          PetscAssertFalse(blocks[i] != xadj[r+1] - xadj[r],PETSC_COMM_SELF,PETSC_ERR_PLIB,"Invalid block sizes prescribed for block %" PetscInt_FMT ": expected %" PetscInt_FMT ", got %" PetscInt_FMT,i,blocks[i],xadj[r+1] - xadj[r]);
          ierr = MatSetValuesLocal(MT,blocks[i],adjncy+xadj[r],blocks[i],adjncy+xadj[r],sarray+xadj[r],ADD_VALUES);CHKERRQ(ierr);
          r   += blocks[i];
        }
      } else {
        for (i=0;i<nvtxs;i++) {
          ierr = MatSetValuesLocal(MT,1,&i,xadj[i+1]-xadj[i],adjncy+xadj[i],sarray+xadj[i],ADD_VALUES);CHKERRQ(ierr);
        }
      }
    } else {
      for (i=0;i<nvtxs;i++) {
        ierr = MatSetValuesLocal(MT,1,&i,xadj[i+1]-xadj[i],adjncy+xadj[i],sarray+xadj[i],ADD_VALUES);CHKERRQ(ierr);
      }
    }
    ierr = MatRestoreRowIJ(local_mat,0,PETSC_FALSE,PETSC_FALSE,&nvtxs,(const PetscInt**)&xadj,(const PetscInt**)&adjncy,&done);CHKERRQ(ierr);
    PetscCheckFalse(!done,PetscObjectComm((PetscObject)local_mat),PETSC_ERR_PLIB,"Error in MatRestoreRowIJ");
    ierr = MatSeqAIJRestoreArray(local_mat,&sarray);CHKERRQ(ierr);
  } else { /* very basic values insertion for all other matrix types */
    PetscInt i;

    for (i=0;i<local_rows;i++) {
      PetscInt       j;
      const PetscInt *local_indices_cols;

      ierr = MatGetRow(local_mat,i,&j,&local_indices_cols,&array);CHKERRQ(ierr);
      ierr = MatSetValuesLocal(MT,1,&i,j,local_indices_cols,array,ADD_VALUES);CHKERRQ(ierr);
      ierr = MatRestoreRow(local_mat,i,&j,&local_indices_cols,&array);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(MT,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatDestroy(&local_mat);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(MT,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (isseqdense) {
    ierr = MatSetOption(MT,MAT_ROW_ORIENTED,PETSC_TRUE);CHKERRQ(ierr);
  }
  if (reuse == MAT_INPLACE_MATRIX) {
    ierr = MatHeaderReplace(mat,&MT);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidLogicalCollectiveEnum(mat,reuse,2);
  PetscValidPointer(newmat,3);
  if (reuse == MAT_REUSE_MATRIX) {
    PetscValidHeaderSpecific(*newmat,MAT_CLASSID,3);
    PetscCheckSameComm(mat,1,*newmat,3);
    PetscCheckFalse(mat == *newmat,PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Cannot reuse the same matrix");
  }
  ierr = PetscUseMethod(mat,"MatISGetMPIXAIJ_C",(Mat,MatType,MatReuse,Mat*),(mat,MATAIJ,reuse,newmat));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDuplicate_IS(Mat mat,MatDuplicateOption op,Mat *newmat)
{
  PetscErrorCode ierr;
  Mat_IS         *matis = (Mat_IS*)(mat->data);
  PetscInt       rbs,cbs,m,n,M,N;
  Mat            B,localmat;

  PetscFunctionBegin;
  ierr = ISLocalToGlobalMappingGetBlockSize(mat->rmap->mapping,&rbs);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingGetBlockSize(mat->cmap->mapping,&cbs);CHKERRQ(ierr);
  ierr = MatGetSize(mat,&M,&N);CHKERRQ(ierr);
  ierr = MatGetLocalSize(mat,&m,&n);CHKERRQ(ierr);
  ierr = MatCreate(PetscObjectComm((PetscObject)mat),&B);CHKERRQ(ierr);
  ierr = MatSetSizes(B,m,n,M,N);CHKERRQ(ierr);
  ierr = MatSetBlockSize(B,rbs == cbs ? rbs : 1);CHKERRQ(ierr);
  ierr = MatSetType(B,MATIS);CHKERRQ(ierr);
  ierr = MatISSetLocalMatType(B,matis->lmattype);CHKERRQ(ierr);
  ierr = MatSetLocalToGlobalMapping(B,mat->rmap->mapping,mat->cmap->mapping);CHKERRQ(ierr);
  ierr = MatDuplicate(matis->A,op,&localmat);CHKERRQ(ierr);
  ierr = MatISSetLocalMat(B,localmat);CHKERRQ(ierr);
  ierr = MatDestroy(&localmat);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  *newmat = B;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatIsHermitian_IS(Mat A,PetscReal tol,PetscBool  *flg)
{
  PetscErrorCode ierr;
  Mat_IS         *matis = (Mat_IS*)A->data;
  PetscBool      local_sym;

  PetscFunctionBegin;
  ierr = MatIsHermitian(matis->A,tol,&local_sym);CHKERRQ(ierr);
  ierr = MPIU_Allreduce(&local_sym,flg,1,MPIU_BOOL,MPI_LAND,PetscObjectComm((PetscObject)A));CHKERRMPI(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatIsSymmetric_IS(Mat A,PetscReal tol,PetscBool  *flg)
{
  PetscErrorCode ierr;
  Mat_IS         *matis = (Mat_IS*)A->data;
  PetscBool      local_sym;

  PetscFunctionBegin;
  ierr = MatIsSymmetric(matis->A,tol,&local_sym);CHKERRQ(ierr);
  ierr = MPIU_Allreduce(&local_sym,flg,1,MPIU_BOOL,MPI_LAND,PetscObjectComm((PetscObject)A));CHKERRMPI(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatIsStructurallySymmetric_IS(Mat A,PetscBool  *flg)
{
  PetscErrorCode ierr;
  Mat_IS         *matis = (Mat_IS*)A->data;
  PetscBool      local_sym;

  PetscFunctionBegin;
  if (A->rmap->mapping != A->cmap->mapping) {
    *flg = PETSC_FALSE;
    PetscFunctionReturn(0);
  }
  ierr = MatIsStructurallySymmetric(matis->A,&local_sym);CHKERRQ(ierr);
  ierr = MPIU_Allreduce(&local_sym,flg,1,MPIU_BOOL,MPI_LAND,PetscObjectComm((PetscObject)A));CHKERRMPI(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDestroy_IS(Mat A)
{
  PetscErrorCode ierr;
  Mat_IS         *b = (Mat_IS*)A->data;

  PetscFunctionBegin;
  ierr = PetscFree(b->bdiag);CHKERRQ(ierr);
  ierr = PetscFree(b->lmattype);CHKERRQ(ierr);
  ierr = MatDestroy(&b->A);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&b->cctx);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&b->rctx);CHKERRQ(ierr);
  ierr = VecDestroy(&b->x);CHKERRQ(ierr);
  ierr = VecDestroy(&b->y);CHKERRQ(ierr);
  ierr = VecDestroy(&b->counter);CHKERRQ(ierr);
  ierr = ISDestroy(&b->getsub_ris);CHKERRQ(ierr);
  ierr = ISDestroy(&b->getsub_cis);CHKERRQ(ierr);
  if (b->sf != b->csf) {
    ierr = PetscSFDestroy(&b->csf);CHKERRQ(ierr);
    ierr = PetscFree2(b->csf_rootdata,b->csf_leafdata);CHKERRQ(ierr);
  } else b->csf = NULL;
  ierr = PetscSFDestroy(&b->sf);CHKERRQ(ierr);
  ierr = PetscFree2(b->sf_rootdata,b->sf_leafdata);CHKERRQ(ierr);
  ierr = PetscFree(A->data);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)A,NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatISSetLocalMatType_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatISGetLocalMat_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatISSetLocalMat_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatISGetMPIXAIJ_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatISSetPreallocation_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatISStoreL2L_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatISFixLocalEmpty_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatConvert_is_mpiaij_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatConvert_is_mpibaij_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatConvert_is_mpisbaij_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatConvert_is_seqaij_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatConvert_is_seqbaij_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatConvert_is_seqsbaij_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatConvert_is_aij_C",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMult_IS(Mat A,Vec x,Vec y)
{
  PetscErrorCode ierr;
  Mat_IS         *is  = (Mat_IS*)A->data;
  PetscScalar    zero = 0.0;

  PetscFunctionBegin;
  /*  scatter the global vector x into the local work vector */
  ierr = VecScatterBegin(is->cctx,x,is->x,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(is->cctx,x,is->x,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);

  /* multiply the local matrix */
  ierr = MatMult(is->A,is->x,is->y);CHKERRQ(ierr);

  /* scatter product back into global memory */
  ierr = VecSet(y,zero);CHKERRQ(ierr);
  ierr = VecScatterBegin(is->rctx,is->y,y,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd(is->rctx,is->y,y,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMultAdd_IS(Mat A,Vec v1,Vec v2,Vec v3)
{
  Vec            temp_vec;
  PetscErrorCode ierr;

  PetscFunctionBegin; /*  v3 = v2 + A * v1.*/
  if (v3 != v2) {
    ierr = MatMult(A,v1,v3);CHKERRQ(ierr);
    ierr = VecAXPY(v3,1.0,v2);CHKERRQ(ierr);
  } else {
    ierr = VecDuplicate(v2,&temp_vec);CHKERRQ(ierr);
    ierr = MatMult(A,v1,temp_vec);CHKERRQ(ierr);
    ierr = VecAXPY(temp_vec,1.0,v2);CHKERRQ(ierr);
    ierr = VecCopy(temp_vec,v3);CHKERRQ(ierr);
    ierr = VecDestroy(&temp_vec);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMultTranspose_IS(Mat A,Vec y,Vec x)
{
  Mat_IS         *is = (Mat_IS*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /*  scatter the global vector x into the local work vector */
  ierr = VecScatterBegin(is->rctx,y,is->y,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(is->rctx,y,is->y,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);

  /* multiply the local matrix */
  ierr = MatMultTranspose(is->A,is->y,is->x);CHKERRQ(ierr);

  /* scatter product back into global vector */
  ierr = VecSet(x,0);CHKERRQ(ierr);
  ierr = VecScatterBegin(is->cctx,is->x,x,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd(is->cctx,is->x,x,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMultTransposeAdd_IS(Mat A,Vec v1,Vec v2,Vec v3)
{
  Vec            temp_vec;
  PetscErrorCode ierr;

  PetscFunctionBegin; /*  v3 = v2 + A' * v1.*/
  if (v3 != v2) {
    ierr = MatMultTranspose(A,v1,v3);CHKERRQ(ierr);
    ierr = VecAXPY(v3,1.0,v2);CHKERRQ(ierr);
  } else {
    ierr = VecDuplicate(v2,&temp_vec);CHKERRQ(ierr);
    ierr = MatMultTranspose(A,v1,temp_vec);CHKERRQ(ierr);
    ierr = VecAXPY(temp_vec,1.0,v2);CHKERRQ(ierr);
    ierr = VecCopy(temp_vec,v3);CHKERRQ(ierr);
    ierr = VecDestroy(&temp_vec);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatView_IS(Mat A,PetscViewer viewer)
{
  Mat_IS         *a = (Mat_IS*)A->data;
  PetscErrorCode ierr;
  PetscViewer    sviewer;
  PetscBool      isascii,view = PETSC_TRUE;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (isascii)  {
    PetscViewerFormat format;

    ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
    if (format == PETSC_VIEWER_ASCII_INFO) view = PETSC_FALSE;
  }
  if (!view) PetscFunctionReturn(0);
  ierr = PetscViewerGetSubViewer(viewer,PETSC_COMM_SELF,&sviewer);CHKERRQ(ierr);
  ierr = MatView(a->A,sviewer);CHKERRQ(ierr);
  ierr = PetscViewerRestoreSubViewer(viewer,PETSC_COMM_SELF,&sviewer);CHKERRQ(ierr);
  ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatInvertBlockDiagonal_IS(Mat mat,const PetscScalar **values)
{
  Mat_IS            *is = (Mat_IS*)mat->data;
  MPI_Datatype      nodeType;
  const PetscScalar *lv;
  PetscInt          bs;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = MatGetBlockSize(mat,&bs);CHKERRQ(ierr);
  ierr = MatSetBlockSize(is->A,bs);CHKERRQ(ierr);
  ierr = MatInvertBlockDiagonal(is->A,&lv);CHKERRQ(ierr);
  if (!is->bdiag) {
    ierr = PetscMalloc1(bs*mat->rmap->n,&is->bdiag);CHKERRQ(ierr);
  }
  ierr = MPI_Type_contiguous(bs,MPIU_SCALAR,&nodeType);CHKERRMPI(ierr);
  ierr = MPI_Type_commit(&nodeType);CHKERRMPI(ierr);
  ierr = PetscSFReduceBegin(is->sf,nodeType,lv,is->bdiag,MPI_REPLACE);CHKERRQ(ierr);
  ierr = PetscSFReduceEnd(is->sf,nodeType,lv,is->bdiag,MPI_REPLACE);CHKERRQ(ierr);
  ierr = MPI_Type_free(&nodeType);CHKERRMPI(ierr);
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
  PetscBool      iscuda;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = ISLocalToGlobalMappingGetSize(A->rmap->mapping,&nr);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingGetBlockSize(A->rmap->mapping,&rbs);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingGetSize(A->cmap->mapping,&nc);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingGetBlockSize(A->cmap->mapping,&cbs);CHKERRQ(ierr);
  ierr = VecDestroy(&is->x);CHKERRQ(ierr);
  ierr = VecDestroy(&is->y);CHKERRQ(ierr);
  ierr = VecDestroy(&is->counter);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&is->rctx);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&is->cctx);CHKERRQ(ierr);
  ierr = MatCreateVecs(is->A,&is->x,&is->y);CHKERRQ(ierr);
  ierr = VecBindToCPU(is->y,PETSC_TRUE);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)is->y,VECSEQCUDA,&iscuda);CHKERRQ(ierr);
  if (iscuda) {
    ierr = PetscFree(A->defaultvectype);CHKERRQ(ierr);
    ierr = PetscStrallocpy(VECCUDA,&A->defaultvectype);CHKERRQ(ierr);
  }
  ierr = MatCreateVecs(A,&cglobal,&rglobal);CHKERRQ(ierr);
  ierr = VecBindToCPU(rglobal,PETSC_TRUE);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingGetBlockIndices(A->rmap->mapping,&garray);CHKERRQ(ierr);
  ierr = ISCreateBlock(PetscObjectComm((PetscObject)A),rbs,nr/rbs,garray,PETSC_USE_POINTER,&from);CHKERRQ(ierr);
  ierr = VecScatterCreate(rglobal,from,is->y,NULL,&is->rctx);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingRestoreBlockIndices(A->rmap->mapping,&garray);CHKERRQ(ierr);
  ierr = ISDestroy(&from);CHKERRQ(ierr);
  if (A->rmap->mapping != A->cmap->mapping) {
    ierr = ISLocalToGlobalMappingGetBlockIndices(A->cmap->mapping,&garray);CHKERRQ(ierr);
    ierr = ISCreateBlock(PetscObjectComm((PetscObject)A),cbs,nc/cbs,garray,PETSC_USE_POINTER,&from);CHKERRQ(ierr);
    ierr = VecScatterCreate(cglobal,from,is->x,NULL,&is->cctx);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingRestoreBlockIndices(A->cmap->mapping,&garray);CHKERRQ(ierr);
    ierr = ISDestroy(&from);CHKERRQ(ierr);
  } else {
    ierr = PetscObjectReference((PetscObject)is->rctx);CHKERRQ(ierr);
    is->cctx = is->rctx;
  }
  ierr = VecDestroy(&cglobal);CHKERRQ(ierr);

  /* interface counter vector (local) */
  ierr = VecDuplicate(is->y,&is->counter);CHKERRQ(ierr);
  ierr = VecBindToCPU(is->counter,PETSC_TRUE);CHKERRQ(ierr);
  ierr = VecSet(is->y,1.);CHKERRQ(ierr);
  ierr = VecScatterBegin(is->rctx,is->y,rglobal,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd(is->rctx,is->y,rglobal,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterBegin(is->rctx,rglobal,is->counter,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(is->rctx,rglobal,is->counter,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecBindToCPU(is->y,PETSC_FALSE);CHKERRQ(ierr);
  ierr = VecBindToCPU(is->counter,PETSC_FALSE);CHKERRQ(ierr);

  /* special functions for block-diagonal matrices */
  ierr = VecSum(rglobal,&sum);CHKERRQ(ierr);
  if ((PetscInt)(PetscRealPart(sum)) == A->rmap->N && A->rmap->N == A->cmap->N && A->rmap->mapping == A->cmap->mapping) {
    A->ops->invertblockdiagonal = MatInvertBlockDiagonal_IS;
  } else {
    A->ops->invertblockdiagonal = NULL;
  }
  ierr = VecDestroy(&rglobal);CHKERRQ(ierr);

  /* setup SF for general purpose shared indices based communications */
  ierr = MatISSetUpSF_IS(A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSetLocalToGlobalMapping_IS(Mat A,ISLocalToGlobalMapping rmapping,ISLocalToGlobalMapping cmapping)
{
  PetscErrorCode ierr;
  PetscInt       nr,rbs,nc,cbs;
  Mat_IS         *is = (Mat_IS*)A->data;
  PetscBool      cong, same = PETSC_FALSE;

  PetscFunctionBegin;
  if (rmapping) PetscCheckSameComm(A,1,rmapping,2);
  if (cmapping) PetscCheckSameComm(A,1,cmapping,3);
  ierr = PetscLayoutSetUp(A->rmap);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(A->cmap);CHKERRQ(ierr);
  ierr = MatHasCongruentLayouts(A,&cong);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)rmapping);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)cmapping);CHKERRQ(ierr);
  /* If NULL, local space matches global space */
  if (!rmapping) {
    IS is;

    ierr = ISCreateStride(PetscObjectComm((PetscObject)A),A->rmap->N,0,1,&is);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingCreateIS(is,&rmapping);CHKERRQ(ierr);
    if (A->rmap->bs > 0) { ierr = ISLocalToGlobalMappingSetBlockSize(rmapping,A->rmap->bs);CHKERRQ(ierr); }
    ierr = ISDestroy(&is);CHKERRQ(ierr);

    if (!cmapping && cong) {
      ierr = PetscObjectReference((PetscObject)rmapping);CHKERRQ(ierr);
      cmapping = rmapping;
    }
  }
  if (!cmapping) {
    IS is;

    ierr = ISCreateStride(PetscObjectComm((PetscObject)A),A->cmap->N,0,1,&is);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingCreateIS(is,&cmapping);CHKERRQ(ierr);
    if (A->cmap->bs > 0) { ierr = ISLocalToGlobalMappingSetBlockSize(cmapping,A->cmap->bs);CHKERRQ(ierr); }
    ierr = ISDestroy(&is);CHKERRQ(ierr);
  }

  /* Clean up */
  ierr = MatDestroy(&is->A);CHKERRQ(ierr);
  if (is->csf != is->sf) {
    ierr = PetscSFDestroy(&is->csf);CHKERRQ(ierr);
    ierr = PetscFree2(is->csf_rootdata,is->csf_leafdata);CHKERRQ(ierr);
  } else is->csf = NULL;
  ierr = PetscSFDestroy(&is->sf);CHKERRQ(ierr);
  ierr = PetscFree2(is->sf_rootdata,is->sf_leafdata);CHKERRQ(ierr);
  ierr = PetscFree(is->bdiag);CHKERRQ(ierr);

  /* check if the two mappings are actually the same for square matrices since MATIS has some optimization for this case
     (DOLFIN passes 2 different objects) */
  ierr = ISLocalToGlobalMappingGetSize(rmapping,&nr);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingGetBlockSize(rmapping,&rbs);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingGetSize(cmapping,&nc);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingGetBlockSize(cmapping,&cbs);CHKERRQ(ierr);
  if (rmapping != cmapping && cong) {
    if (nr == nc && cbs == rbs) {
      const PetscInt *idxs1,*idxs2;

      ierr = ISLocalToGlobalMappingGetBlockIndices(rmapping,&idxs1);CHKERRQ(ierr);
      ierr = ISLocalToGlobalMappingGetBlockIndices(cmapping,&idxs2);CHKERRQ(ierr);
      ierr = PetscArraycmp(idxs1,idxs2,nr/rbs,&same);CHKERRQ(ierr);
      ierr = ISLocalToGlobalMappingRestoreBlockIndices(rmapping,&idxs1);CHKERRQ(ierr);
      ierr = ISLocalToGlobalMappingRestoreBlockIndices(cmapping,&idxs2);CHKERRQ(ierr);
    }
    ierr = MPIU_Allreduce(MPI_IN_PLACE,&same,1,MPIU_BOOL,MPI_LAND,PetscObjectComm((PetscObject)A));CHKERRMPI(ierr);
  }
  ierr = PetscLayoutSetBlockSize(A->rmap,rbs);CHKERRQ(ierr);
  ierr = PetscLayoutSetBlockSize(A->cmap,cbs);CHKERRQ(ierr);
  ierr = PetscLayoutSetISLocalToGlobalMapping(A->rmap,rmapping);CHKERRQ(ierr);
  ierr = PetscLayoutSetISLocalToGlobalMapping(A->cmap,same ? rmapping : cmapping);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingDestroy(&rmapping);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingDestroy(&cmapping);CHKERRQ(ierr);

  /* Create the local matrix A */
  ierr = MatCreate(PETSC_COMM_SELF,&is->A);CHKERRQ(ierr);
  ierr = MatSetType(is->A,is->lmattype);CHKERRQ(ierr);
  ierr = MatSetSizes(is->A,nr,nc,nr,nc);CHKERRQ(ierr);
  ierr = MatSetBlockSizes(is->A,rbs,cbs);CHKERRQ(ierr);
  ierr = MatSetOptionsPrefix(is->A,"is_");CHKERRQ(ierr);
  ierr = MatAppendOptionsPrefix(is->A,((PetscObject)A)->prefix);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(is->A->rmap);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(is->A->cmap);CHKERRQ(ierr);

  /* setup scatters and local vectors for MatMult */
  if (!is->islocalref) {
    ierr = MatISSetUpScatters_Private(A);CHKERRQ(ierr);
  }
  A->preallocated = PETSC_TRUE;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSetUp_IS(Mat A)
{
  ISLocalToGlobalMapping rmap, cmap;
  PetscErrorCode         ierr;

  PetscFunctionBegin;
  ierr = MatGetLocalToGlobalMapping(A,&rmap,&cmap);CHKERRQ(ierr);
  PetscCheckFalse(rmap && !cmap,PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Missing column mapping");
  PetscCheckFalse(cmap && !rmap,PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Missing row mapping");
  if (!rmap && !cmap) {
    ierr = MatSetLocalToGlobalMapping(A,NULL,NULL);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSetValues_IS(Mat mat, PetscInt m,const PetscInt *rows, PetscInt n,const PetscInt *cols, const PetscScalar *values, InsertMode addv)
{
  Mat_IS         *is = (Mat_IS*)mat->data;
  PetscErrorCode ierr;
  PetscInt       i,zm,zn;
  PetscInt       rows_l[MATIS_MAX_ENTRIES_INSERTION],cols_l[MATIS_MAX_ENTRIES_INSERTION];

  PetscFunctionBegin;
  if (PetscDefined(USE_DEBUG)) {
    PetscCheckFalse(m > MATIS_MAX_ENTRIES_INSERTION || n > MATIS_MAX_ENTRIES_INSERTION,PETSC_COMM_SELF,PETSC_ERR_SUP,"Number of row/column indices must be <= %d: they are %" PetscInt_FMT " %" PetscInt_FMT,MATIS_MAX_ENTRIES_INSERTION,m,n);
    /* count negative indices */
    for (i=0,zm=0;i<m;i++) if (rows[i] < 0) zm++;
    for (i=0,zn=0;i<n;i++) if (cols[i] < 0) zn++;
  }
  ierr = ISGlobalToLocalMappingApply(mat->rmap->mapping,IS_GTOLM_MASK,m,rows,&m,rows_l);CHKERRQ(ierr);
  ierr = ISGlobalToLocalMappingApply(mat->cmap->mapping,IS_GTOLM_MASK,n,cols,&n,cols_l);CHKERRQ(ierr);
  if (PetscDefined(USE_DEBUG)) {
    /* count negative indices (should be the same as before) */
    for (i=0;i<m;i++) if (rows_l[i] < 0) zm--;
    for (i=0;i<n;i++) if (cols_l[i] < 0) zn--;
    PetscCheckFalse(!is->A->rmap->mapping && zm,PETSC_COMM_SELF,PETSC_ERR_SUP,"Some of the row indices can not be mapped! Maybe you should not use MATIS");
    PetscCheckFalse(!is->A->cmap->mapping && zn,PETSC_COMM_SELF,PETSC_ERR_SUP,"Some of the column indices can not be mapped! Maybe you should not use MATIS");
  }
  ierr = MatSetValues(is->A,m,rows_l,n,cols_l,values,addv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSetValuesBlocked_IS(Mat mat, PetscInt m,const PetscInt *rows, PetscInt n,const PetscInt *cols, const PetscScalar *values, InsertMode addv)
{
  Mat_IS         *is = (Mat_IS*)mat->data;
  PetscErrorCode ierr;
  PetscInt       i,zm,zn;
  PetscInt       rows_l[MATIS_MAX_ENTRIES_INSERTION],cols_l[MATIS_MAX_ENTRIES_INSERTION];

  PetscFunctionBegin;
  if (PetscDefined(USE_DEBUG)) {
    PetscCheckFalse(m > MATIS_MAX_ENTRIES_INSERTION || n > MATIS_MAX_ENTRIES_INSERTION,PETSC_COMM_SELF,PETSC_ERR_SUP,"Number of row/column block indices must be <= %d: they are %" PetscInt_FMT " %" PetscInt_FMT,MATIS_MAX_ENTRIES_INSERTION,m,n);
    /* count negative indices */
    for (i=0,zm=0;i<m;i++) if (rows[i] < 0) zm++;
    for (i=0,zn=0;i<n;i++) if (cols[i] < 0) zn++;
  }
  ierr = ISGlobalToLocalMappingApplyBlock(mat->rmap->mapping,IS_GTOLM_MASK,m,rows,&m,rows_l);CHKERRQ(ierr);
  ierr = ISGlobalToLocalMappingApplyBlock(mat->cmap->mapping,IS_GTOLM_MASK,n,cols,&n,cols_l);CHKERRQ(ierr);
  if (PetscDefined(USE_DEBUG)) {
    /* count negative indices (should be the same as before) */
    for (i=0;i<m;i++) if (rows_l[i] < 0) zm--;
    for (i=0;i<n;i++) if (cols_l[i] < 0) zn--;
    PetscCheckFalse(!is->A->rmap->mapping && zm,PETSC_COMM_SELF,PETSC_ERR_SUP,"Some of the row indices can not be mapped! Maybe you should not use MATIS");
    PetscCheckFalse(!is->A->cmap->mapping && zn,PETSC_COMM_SELF,PETSC_ERR_SUP,"Some of the column indices can not be mapped! Maybe you should not use MATIS");
  }
  ierr = MatSetValuesBlocked(is->A,m,rows_l,n,cols_l,values,addv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSetValuesLocal_IS(Mat A,PetscInt m,const PetscInt *rows, PetscInt n,const PetscInt *cols,const PetscScalar *values,InsertMode addv)
{
  PetscErrorCode ierr;
  Mat_IS         *is = (Mat_IS*)A->data;

  PetscFunctionBegin;
  if (is->A->rmap->mapping) {
    ierr = MatSetValuesLocal(is->A,m,rows,n,cols,values,addv);CHKERRQ(ierr);
  } else {
    ierr = MatSetValues(is->A,m,rows,n,cols,values,addv);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSetValuesBlockedLocal_IS(Mat A,PetscInt m,const PetscInt *rows, PetscInt n,const PetscInt *cols,const PetscScalar *values,InsertMode addv)
{
  PetscErrorCode ierr;
  Mat_IS         *is = (Mat_IS*)A->data;

  PetscFunctionBegin;
  if (is->A->rmap->mapping) {
    if (PetscDefined(USE_DEBUG)) {
      PetscInt ibs,bs;

      ierr = ISLocalToGlobalMappingGetBlockSize(is->A->rmap->mapping,&ibs);CHKERRQ(ierr);
      ierr = MatGetBlockSize(is->A,&bs);CHKERRQ(ierr);
      PetscCheckFalse(ibs != bs,PETSC_COMM_SELF,PETSC_ERR_SUP,"Different block sizes! local mat %" PetscInt_FMT ", local l2g map %" PetscInt_FMT,bs,ibs);
    }
    ierr = MatSetValuesBlockedLocal(is->A,m,rows,n,cols,values,addv);CHKERRQ(ierr);
  } else {
    ierr = MatSetValuesBlocked(is->A,m,rows,n,cols,values,addv);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatISZeroRowsColumnsLocal_Private(Mat A,PetscInt n,const PetscInt rows[],PetscScalar diag,PetscBool columns)
{
  Mat_IS         *is = (Mat_IS*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!n) {
    is->pure_neumann = PETSC_TRUE;
  } else {
    PetscInt i;
    is->pure_neumann = PETSC_FALSE;

    if (columns) {
      ierr = MatZeroRowsColumns(is->A,n,rows,diag,NULL,NULL);CHKERRQ(ierr);
    } else {
      ierr = MatZeroRows(is->A,n,rows,diag,NULL,NULL);CHKERRQ(ierr);
    }
    if (diag != 0.) {
      const PetscScalar *array;
      ierr = VecGetArrayRead(is->counter,&array);CHKERRQ(ierr);
      for (i=0; i<n; i++) {
        ierr = MatSetValue(is->A,rows[i],rows[i],diag/(array[rows[i]]),INSERT_VALUES);CHKERRQ(ierr);
      }
      ierr = VecRestoreArrayRead(is->counter,&array);CHKERRQ(ierr);
    }
    ierr = MatAssemblyBegin(is->A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(is->A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatZeroRowsColumns_Private_IS(Mat A,PetscInt n,const PetscInt rows[],PetscScalar diag,Vec x,Vec b,PetscBool columns)
{
  Mat_IS         *matis = (Mat_IS*)A->data;
  PetscInt       nr,nl,len,i;
  PetscInt       *lrows;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (PetscUnlikelyDebug(columns || diag != 0. || (x && b))) {
    PetscBool cong;

    ierr = PetscLayoutCompare(A->rmap,A->cmap,&cong);CHKERRQ(ierr);
    cong = (PetscBool)(cong && matis->sf == matis->csf);
    PetscCheckFalse(!cong && columns,PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Columns can be zeroed if and only if A->rmap and A->cmap are congruent and the l2g maps are the same for MATIS");
    PetscCheckFalse(!cong && diag != 0.,PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Nonzero diagonal value supported if and only if A->rmap and A->cmap are congruent and the l2g maps are the same for MATIS");
    PetscCheckFalse(!cong && x && b,PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"A->rmap and A->cmap need to be congruent, and the l2g maps be the same");
  }
  /* get locally owned rows */
  ierr = PetscLayoutMapLocal(A->rmap,n,rows,&len,&lrows,NULL);CHKERRQ(ierr);
  /* fix right hand side if needed */
  if (x && b) {
    const PetscScalar *xx;
    PetscScalar       *bb;

    ierr = VecGetArrayRead(x, &xx);CHKERRQ(ierr);
    ierr = VecGetArray(b, &bb);CHKERRQ(ierr);
    for (i=0;i<len;++i) bb[lrows[i]] = diag*xx[lrows[i]];
    ierr = VecRestoreArrayRead(x, &xx);CHKERRQ(ierr);
    ierr = VecRestoreArray(b, &bb);CHKERRQ(ierr);
  }
  /* get rows associated to the local matrices */
  ierr = MatGetSize(matis->A,&nl,NULL);CHKERRQ(ierr);
  ierr = PetscArrayzero(matis->sf_leafdata,nl);CHKERRQ(ierr);
  ierr = PetscArrayzero(matis->sf_rootdata,A->rmap->n);CHKERRQ(ierr);
  for (i=0;i<len;i++) matis->sf_rootdata[lrows[i]] = 1;
  ierr = PetscFree(lrows);CHKERRQ(ierr);
  ierr = PetscSFBcastBegin(matis->sf,MPIU_INT,matis->sf_rootdata,matis->sf_leafdata,MPI_REPLACE);CHKERRQ(ierr);
  ierr = PetscSFBcastEnd(matis->sf,MPIU_INT,matis->sf_rootdata,matis->sf_leafdata,MPI_REPLACE);CHKERRQ(ierr);
  ierr = PetscMalloc1(nl,&lrows);CHKERRQ(ierr);
  for (i=0,nr=0;i<nl;i++) if (matis->sf_leafdata[i]) lrows[nr++] = i;
  ierr = MatISZeroRowsColumnsLocal_Private(A,nr,lrows,diag,columns);CHKERRQ(ierr);
  ierr = PetscFree(lrows);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatZeroRows_IS(Mat A,PetscInt n,const PetscInt rows[],PetscScalar diag,Vec x,Vec b)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatZeroRowsColumns_Private_IS(A,n,rows,diag,x,b,PETSC_FALSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatZeroRowsColumns_IS(Mat A,PetscInt n,const PetscInt rows[],PetscScalar diag,Vec x,Vec b)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatZeroRowsColumns_Private_IS(A,n,rows,diag,x,b,PETSC_TRUE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatAssemblyBegin_IS(Mat A,MatAssemblyType type)
{
  Mat_IS         *is = (Mat_IS*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatAssemblyBegin(is->A,type);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatAssemblyEnd_IS(Mat A,MatAssemblyType type)
{
  Mat_IS         *is = (Mat_IS*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatAssemblyEnd(is->A,type);CHKERRQ(ierr);
  /* fix for local empty rows/cols */
  if (is->locempty && type == MAT_FINAL_ASSEMBLY) {
    Mat                    newlA;
    ISLocalToGlobalMapping rl2g,cl2g;
    IS                     nzr,nzc;
    PetscInt               nr,nc,nnzr,nnzc;
    PetscBool              lnewl2g,newl2g;

    ierr = MatGetSize(is->A,&nr,&nc);CHKERRQ(ierr);
    ierr = MatFindNonzeroRowsOrCols_Basic(is->A,PETSC_FALSE,PETSC_SMALL,&nzr);CHKERRQ(ierr);
    if (!nzr) {
      ierr = ISCreateStride(PetscObjectComm((PetscObject)is->A),nr,0,1,&nzr);CHKERRQ(ierr);
    }
    ierr = MatFindNonzeroRowsOrCols_Basic(is->A,PETSC_TRUE,PETSC_SMALL,&nzc);CHKERRQ(ierr);
    if (!nzc) {
      ierr = ISCreateStride(PetscObjectComm((PetscObject)is->A),nc,0,1,&nzc);CHKERRQ(ierr);
    }
    ierr = ISGetSize(nzr,&nnzr);CHKERRQ(ierr);
    ierr = ISGetSize(nzc,&nnzc);CHKERRQ(ierr);
    if (nnzr != nr || nnzc != nc) {
      ISLocalToGlobalMapping l2g;
      IS                     is1,is2;

      /* need new global l2g map */
      lnewl2g = PETSC_TRUE;
      ierr    = MPI_Allreduce(&lnewl2g,&newl2g,1,MPIU_BOOL,MPI_LOR,PetscObjectComm((PetscObject)A));CHKERRMPI(ierr);

      /* extract valid submatrix */
      ierr = MatCreateSubMatrix(is->A,nzr,nzc,MAT_INITIAL_MATRIX,&newlA);CHKERRQ(ierr);

      /* attach local l2g maps for successive calls of MatSetValues on the local matrix */
      ierr = ISLocalToGlobalMappingCreateIS(nzr,&l2g);CHKERRQ(ierr);
      ierr = ISCreateStride(PetscObjectComm((PetscObject)is->A),nr,0,1,&is1);CHKERRQ(ierr);
      ierr = ISGlobalToLocalMappingApplyIS(l2g,IS_GTOLM_MASK,is1,&is2);CHKERRQ(ierr);
      ierr = ISLocalToGlobalMappingDestroy(&l2g);CHKERRQ(ierr);
      if (is->A->rmap->mapping) { /* the matrix has a local-to-local map already attached (probably DMDA generated) */
        const PetscInt *idxs1,*idxs2;
        PetscInt       j,i,nl,*tidxs;

        ierr = ISLocalToGlobalMappingGetSize(is->A->rmap->mapping,&nl);CHKERRQ(ierr);
        ierr = ISLocalToGlobalMappingGetIndices(is->A->rmap->mapping,&idxs1);CHKERRQ(ierr);
        ierr = PetscMalloc1(nl,&tidxs);CHKERRQ(ierr);
        ierr = ISGetIndices(is2,&idxs2);CHKERRQ(ierr);
        for (i=0,j=0;i<nl;i++) tidxs[i] = idxs1[i] < 0 ? -1 : idxs2[j++];
        PetscCheckFalse(j != nr,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Unexpected count %" PetscInt_FMT " != %" PetscInt_FMT,j,nr);
        ierr = ISLocalToGlobalMappingRestoreIndices(is->A->rmap->mapping,&idxs1);CHKERRQ(ierr);
        ierr = ISRestoreIndices(is2,&idxs2);CHKERRQ(ierr);
        ierr = ISDestroy(&is2);CHKERRQ(ierr);
        ierr = ISCreateGeneral(PetscObjectComm((PetscObject)is->A->rmap->mapping),nl,tidxs,PETSC_OWN_POINTER,&is2);CHKERRQ(ierr);
      }
      ierr = ISLocalToGlobalMappingCreateIS(is2,&rl2g);CHKERRQ(ierr);
      ierr = ISDestroy(&is1);CHKERRQ(ierr);
      ierr = ISDestroy(&is2);CHKERRQ(ierr);

      ierr = ISLocalToGlobalMappingCreateIS(nzc,&l2g);CHKERRQ(ierr);
      ierr = ISCreateStride(PetscObjectComm((PetscObject)is->A),nc,0,1,&is1);CHKERRQ(ierr);
      ierr = ISGlobalToLocalMappingApplyIS(l2g,IS_GTOLM_MASK,is1,&is2);CHKERRQ(ierr);
      ierr = ISLocalToGlobalMappingDestroy(&l2g);CHKERRQ(ierr);
      if (is->A->cmap->mapping) { /* the matrix has a local-to-local map already attached (probably DMDA generated) */
        const PetscInt *idxs1,*idxs2;
        PetscInt       j,i,nl,*tidxs;

        ierr = ISLocalToGlobalMappingGetSize(is->A->cmap->mapping,&nl);CHKERRQ(ierr);
        ierr = ISLocalToGlobalMappingGetIndices(is->A->cmap->mapping,&idxs1);CHKERRQ(ierr);
        ierr = PetscMalloc1(nl,&tidxs);CHKERRQ(ierr);
        ierr = ISGetIndices(is2,&idxs2);CHKERRQ(ierr);
        for (i=0,j=0;i<nl;i++) tidxs[i] = idxs1[i] < 0 ? -1 : idxs2[j++];
        PetscCheckFalse(j != nc,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Unexpected count %" PetscInt_FMT " != %" PetscInt_FMT,j,nc);
        ierr = ISLocalToGlobalMappingRestoreIndices(is->A->cmap->mapping,&idxs1);CHKERRQ(ierr);
        ierr = ISRestoreIndices(is2,&idxs2);CHKERRQ(ierr);
        ierr = ISDestroy(&is2);CHKERRQ(ierr);
        ierr = ISCreateGeneral(PetscObjectComm((PetscObject)is->A->cmap->mapping),nl,tidxs,PETSC_OWN_POINTER,&is2);CHKERRQ(ierr);
      }
      ierr = ISLocalToGlobalMappingCreateIS(is2,&cl2g);CHKERRQ(ierr);
      ierr = ISDestroy(&is1);CHKERRQ(ierr);
      ierr = ISDestroy(&is2);CHKERRQ(ierr);

      ierr = MatSetLocalToGlobalMapping(newlA,rl2g,cl2g);CHKERRQ(ierr);

      ierr = ISLocalToGlobalMappingDestroy(&rl2g);CHKERRQ(ierr);
      ierr = ISLocalToGlobalMappingDestroy(&cl2g);CHKERRQ(ierr);
    } else { /* local matrix fully populated */
      lnewl2g = PETSC_FALSE;
      ierr    = MPI_Allreduce(&lnewl2g,&newl2g,1,MPIU_BOOL,MPI_LOR,PetscObjectComm((PetscObject)A));CHKERRMPI(ierr);
      ierr    = PetscObjectReference((PetscObject)is->A);CHKERRQ(ierr);
      newlA   = is->A;
    }
    /* attach new global l2g map if needed */
    if (newl2g) {
      IS             gnzr,gnzc;
      const PetscInt *grid,*gcid;

      ierr = ISLocalToGlobalMappingApplyIS(A->rmap->mapping,nzr,&gnzr);CHKERRQ(ierr);
      ierr = ISLocalToGlobalMappingApplyIS(A->cmap->mapping,nzc,&gnzc);CHKERRQ(ierr);
      ierr = ISGetIndices(gnzr,&grid);CHKERRQ(ierr);
      ierr = ISGetIndices(gnzc,&gcid);CHKERRQ(ierr);
      ierr = ISLocalToGlobalMappingCreate(PetscObjectComm((PetscObject)A),1,nnzr,grid,PETSC_COPY_VALUES,&rl2g);CHKERRQ(ierr);
      ierr = ISLocalToGlobalMappingCreate(PetscObjectComm((PetscObject)A),1,nnzc,gcid,PETSC_COPY_VALUES,&cl2g);CHKERRQ(ierr);
      ierr = ISRestoreIndices(gnzr,&grid);CHKERRQ(ierr);
      ierr = ISRestoreIndices(gnzc,&gcid);CHKERRQ(ierr);
      ierr = ISDestroy(&gnzr);CHKERRQ(ierr);
      ierr = ISDestroy(&gnzc);CHKERRQ(ierr);
      ierr = MatSetLocalToGlobalMapping(A,rl2g,cl2g);CHKERRQ(ierr);
      ierr = ISLocalToGlobalMappingDestroy(&rl2g);CHKERRQ(ierr);
      ierr = ISLocalToGlobalMappingDestroy(&cl2g);CHKERRQ(ierr);
    }
    ierr = MatISSetLocalMat(A,newlA);CHKERRQ(ierr);
    ierr = MatDestroy(&newlA);CHKERRQ(ierr);
    ierr = ISDestroy(&nzr);CHKERRQ(ierr);
    ierr = ISDestroy(&nzc);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidPointer(local,2);
  ierr = PetscUseMethod(mat,"MatISGetLocalMat_C",(Mat,Mat*),(mat,local));CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidPointer(local,2);
  ierr = PetscUseMethod(mat,"MatISRestoreLocalMat_C",(Mat,Mat*),(mat,local));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatISSetLocalMatType_IS(Mat mat,MatType mtype)
{
  Mat_IS         *is = (Mat_IS*)mat->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (is->A) {
    ierr = MatSetType(is->A,mtype);CHKERRQ(ierr);
  }
  ierr = PetscFree(is->lmattype);CHKERRQ(ierr);
  ierr = PetscStrallocpy(mtype,&is->lmattype);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  ierr = PetscUseMethod(mat,"MatISSetLocalMatType_C",(Mat,MatType),(mat,mtype));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatISSetLocalMat_IS(Mat mat,Mat local)
{
  Mat_IS         *is = (Mat_IS*)mat->data;
  PetscInt       nrows,ncols,orows,ocols;
  PetscErrorCode ierr;
  MatType        mtype,otype;
  PetscBool      sametype = PETSC_TRUE;

  PetscFunctionBegin;
  if (is->A) {
    ierr = MatGetSize(is->A,&orows,&ocols);CHKERRQ(ierr);
    ierr = MatGetSize(local,&nrows,&ncols);CHKERRQ(ierr);
    PetscCheckFalse(orows != nrows || ocols != ncols,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Local MATIS matrix should be of size %" PetscInt_FMT "x%" PetscInt_FMT " (you passed a %" PetscInt_FMT "x%" PetscInt_FMT " matrix)",orows,ocols,nrows,ncols);
    ierr = MatGetType(local,&mtype);CHKERRQ(ierr);
    ierr = MatGetType(is->A,&otype);CHKERRQ(ierr);
    ierr = PetscStrcmp(mtype,otype,&sametype);CHKERRQ(ierr);
  }
  ierr  = PetscObjectReference((PetscObject)local);CHKERRQ(ierr);
  ierr  = MatDestroy(&is->A);CHKERRQ(ierr);
  is->A = local;
  ierr  = MatGetType(is->A,&mtype);CHKERRQ(ierr);
  ierr  = MatISSetLocalMatType(mat,mtype);CHKERRQ(ierr);
  if (!sametype && !is->islocalref) {
    ierr = MatISSetUpScatters_Private(mat);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidHeaderSpecific(local,MAT_CLASSID,2);
  ierr = PetscUseMethod(mat,"MatISSetLocalMat_C",(Mat,Mat),(mat,local));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatZeroEntries_IS(Mat A)
{
  Mat_IS         *a = (Mat_IS*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatZeroEntries(a->A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatScale_IS(Mat A,PetscScalar a)
{
  Mat_IS         *is = (Mat_IS*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatScale(is->A,a);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatGetDiagonal_IS(Mat A, Vec v)
{
  Mat_IS         *is = (Mat_IS*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* get diagonal of the local matrix */
  ierr = MatGetDiagonal(is->A,is->y);CHKERRQ(ierr);

  /* scatter diagonal back into global vector */
  ierr = VecSet(v,0);CHKERRQ(ierr);
  ierr = VecScatterBegin(is->rctx,is->y,v,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd(is->rctx,is->y,v,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSetOption_IS(Mat A,MatOption op,PetscBool flg)
{
  Mat_IS         *a = (Mat_IS*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatSetOption(a->A,op,flg);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatAXPY_IS(Mat Y,PetscScalar a,Mat X,MatStructure str)
{
  Mat_IS         *y = (Mat_IS*)Y->data;
  Mat_IS         *x;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (PetscDefined(USE_DEBUG)) {
    PetscBool      ismatis;
    ierr = PetscObjectTypeCompare((PetscObject)X,MATIS,&ismatis);CHKERRQ(ierr);
    PetscCheckFalse(!ismatis,PetscObjectComm((PetscObject)Y),PETSC_ERR_SUP,"Cannot call MatAXPY(Y,a,X,str) with X not of type MATIS");
  }
  x = (Mat_IS*)X->data;
  ierr = MatAXPY(y->A,a,x->A,str);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatGetLocalSubMatrix_IS(Mat A,IS row,IS col,Mat *submat)
{
  Mat                    lA;
  Mat_IS                 *matis;
  ISLocalToGlobalMapping rl2g,cl2g;
  IS                     is;
  const PetscInt         *rg,*rl;
  PetscInt               nrg;
  PetscInt               N,M,nrl,i,*idxs;
  PetscErrorCode         ierr;

  PetscFunctionBegin;
  ierr = ISLocalToGlobalMappingGetIndices(A->rmap->mapping,&rg);CHKERRQ(ierr);
  ierr = ISGetLocalSize(row,&nrl);CHKERRQ(ierr);
  ierr = ISGetIndices(row,&rl);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingGetSize(A->rmap->mapping,&nrg);CHKERRQ(ierr);
  if (PetscDefined(USE_DEBUG)) {
    for (i=0; i<nrl; i++) PetscCheckFalse(rl[i]>=nrg,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Local row index %" PetscInt_FMT " -> %" PetscInt_FMT " greater then maximum possible %" PetscInt_FMT,i,rl[i],nrg);
  }
  ierr = PetscMalloc1(nrg,&idxs);CHKERRQ(ierr);
  /* map from [0,nrl) to row */
  for (i=0;i<nrl;i++) idxs[i] = rl[i];
  for (i=nrl;i<nrg;i++) idxs[i] = -1;
  ierr = ISRestoreIndices(row,&rl);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingRestoreIndices(A->rmap->mapping,&rg);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PetscObjectComm((PetscObject)A),nrg,idxs,PETSC_OWN_POINTER,&is);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingCreateIS(is,&rl2g);CHKERRQ(ierr);
  ierr = ISDestroy(&is);CHKERRQ(ierr);
  /* compute new l2g map for columns */
  if (col != row || A->rmap->mapping != A->cmap->mapping) {
    const PetscInt *cg,*cl;
    PetscInt       ncg;
    PetscInt       ncl;

    ierr = ISLocalToGlobalMappingGetIndices(A->cmap->mapping,&cg);CHKERRQ(ierr);
    ierr = ISGetLocalSize(col,&ncl);CHKERRQ(ierr);
    ierr = ISGetIndices(col,&cl);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingGetSize(A->cmap->mapping,&ncg);CHKERRQ(ierr);
    if (PetscDefined(USE_DEBUG)) {
      for (i=0; i<ncl; i++) PetscCheckFalse(cl[i]>=ncg,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Local column index %" PetscInt_FMT " -> %" PetscInt_FMT " greater then maximum possible %" PetscInt_FMT,i,cl[i],ncg);
    }
    ierr = PetscMalloc1(ncg,&idxs);CHKERRQ(ierr);
    /* map from [0,ncl) to col */
    for (i=0;i<ncl;i++) idxs[i] = cl[i];
    for (i=ncl;i<ncg;i++) idxs[i] = -1;
    ierr = ISRestoreIndices(col,&cl);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingRestoreIndices(A->cmap->mapping,&cg);CHKERRQ(ierr);
    ierr = ISCreateGeneral(PetscObjectComm((PetscObject)A),ncg,idxs,PETSC_OWN_POINTER,&is);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingCreateIS(is,&cl2g);CHKERRQ(ierr);
    ierr = ISDestroy(&is);CHKERRQ(ierr);
  } else {
    ierr = PetscObjectReference((PetscObject)rl2g);CHKERRQ(ierr);
    cl2g = rl2g;
  }
  /* create the MATIS submatrix */
  ierr = MatGetSize(A,&M,&N);CHKERRQ(ierr);
  ierr = MatCreate(PetscObjectComm((PetscObject)A),submat);CHKERRQ(ierr);
  ierr = MatSetSizes(*submat,PETSC_DECIDE,PETSC_DECIDE,M,N);CHKERRQ(ierr);
  ierr = MatSetType(*submat,MATIS);CHKERRQ(ierr);
  matis = (Mat_IS*)((*submat)->data);
  matis->islocalref = PETSC_TRUE;
  ierr = MatSetLocalToGlobalMapping(*submat,rl2g,cl2g);CHKERRQ(ierr);
  ierr = MatISGetLocalMat(A,&lA);CHKERRQ(ierr);
  ierr = MatISSetLocalMat(*submat,lA);CHKERRQ(ierr);
  ierr = MatSetUp(*submat);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(*submat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*submat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingDestroy(&rl2g);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingDestroy(&cl2g);CHKERRQ(ierr);
  /* remove unsupported ops */
  ierr = PetscMemzero((*submat)->ops,sizeof(struct _MatOps));CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"MATIS options");CHKERRQ(ierr);
  ierr = PetscOptionsBool("-matis_fixempty","Fix local matrices in case of empty local rows/columns","MatISFixLocalEmpty",a->locempty,&a->locempty,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-matis_storel2l","Store local-to-local matrices generated from PtAP operations","MatISStoreL2L",a->storel2l,&a->storel2l,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsFList("-matis_localmat_type","Matrix type","MatISSetLocalMatType",MatList,a->lmattype,type,256,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = MatISSetLocalMatType(A,type);CHKERRQ(ierr);
  }
  if (a->A) {
    ierr = MatSetFromOptions(a->A);CHKERRQ(ierr);
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatCreate(comm,A);CHKERRQ(ierr);
  ierr = MatSetSizes(*A,m,n,M,N);CHKERRQ(ierr);
  if (bs > 0) {
    ierr = MatSetBlockSize(*A,bs);CHKERRQ(ierr);
  }
  ierr = MatSetType(*A,MATIS);CHKERRQ(ierr);
  ierr = MatSetLocalToGlobalMapping(*A,rmap,cmap);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatHasOperation_IS(Mat A, MatOperation op, PetscBool *has)
{
  Mat_IS         *a = (Mat_IS*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  *has = PETSC_FALSE;
  if (!((void**)A->ops)[op]) PetscFunctionReturn(0);
  ierr = MatHasOperation(a->A,op,has);CHKERRQ(ierr);
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
  PetscErrorCode ierr;
  Mat_IS         *b;

  PetscFunctionBegin;
  ierr    = PetscNewLog(A,&b);CHKERRQ(ierr);
  ierr    = PetscStrallocpy(MATAIJ,&b->lmattype);CHKERRQ(ierr);
  A->data = (void*)b;

  /* matrix ops */
  ierr = PetscMemzero(A->ops,sizeof(struct _MatOps));CHKERRQ(ierr);
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
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatISSetLocalMatType_C",MatISSetLocalMatType_IS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatISGetLocalMat_C",MatISGetLocalMat_IS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatISRestoreLocalMat_C",MatISRestoreLocalMat_IS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatISSetLocalMat_C",MatISSetLocalMat_IS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatISGetMPIXAIJ_C",MatConvert_IS_XAIJ);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatISSetPreallocation_C",MatISSetPreallocation_IS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatISStoreL2L_C",MatISStoreL2L_IS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatISFixLocalEmpty_C",MatISFixLocalEmpty_IS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatConvert_is_mpiaij_C",MatConvert_IS_XAIJ);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatConvert_is_mpibaij_C",MatConvert_IS_XAIJ);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatConvert_is_mpisbaij_C",MatConvert_IS_XAIJ);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatConvert_is_seqaij_C",MatConvert_IS_XAIJ);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatConvert_is_seqbaij_C",MatConvert_IS_XAIJ);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatConvert_is_seqsbaij_C",MatConvert_IS_XAIJ);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatConvert_is_aij_C",MatConvert_IS_XAIJ);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)A,MATIS);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
