/*
    Creates a matrix class for using the Neumann-Neumann type preconditioners.
    This stores the matrices in globally unassembled form. Each processor
    assembles only its local Neumann problem and the parallel matrix vector
    product is handled "implicitly".

    Currently this allows for only one subdomain per processor.
*/

#include <../src/mat/impls/is/matis.h>      /*I "petscmat.h" I*/
#include <../src/mat/impls/aij/mpi/mpiaij.h>
#include <petsc/private/sfimpl.h>

#define MATIS_MAX_ENTRIES_INSERTION 2048

static PetscErrorCode MatISContainerDestroyFields_Private(void *ptr)
{
  MatISLocalFields lf = (MatISLocalFields)ptr;
  PetscInt         i;
  PetscErrorCode   ierr;

  PetscFunctionBeginUser;
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

static PetscErrorCode MatISContainerDestroyArray_Private(void *ptr)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscFree(ptr);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatConvert_MPIAIJ_IS(Mat A,MatType type,MatReuse reuse,Mat *newmat)
{
  Mat_MPIAIJ             *aij  = (Mat_MPIAIJ*)A->data;
  Mat_SeqAIJ             *diag = (Mat_SeqAIJ*)(aij->A->data);
  Mat_SeqAIJ             *offd = (Mat_SeqAIJ*)(aij->B->data);
  Mat                    lA;
  ISLocalToGlobalMapping rl2g,cl2g;
  IS                     is;
  MPI_Comm               comm;
  void                   *ptrs[2];
  const char             *names[2] = {"_convert_csr_aux","_convert_csr_data"};
  PetscScalar            *dd,*od,*aa,*data;
  PetscInt               *di,*dj,*oi,*oj;
  PetscInt               *aux,*ii,*jj;
  PetscInt               lc,dr,dc,oc,str,stc,nnz,i,jd,jo,cum;
  PetscErrorCode         ierr;

  PetscFunctionBeginUser;
  ierr = PetscObjectGetComm((PetscObject)A,&comm);CHKERRQ(ierr);
  if (!aij->garray) SETERRQ(comm,PETSC_ERR_SUP,"garray not present");

  /* access relevant information from MPIAIJ */
  ierr = MatGetOwnershipRange(A,&str,NULL);CHKERRQ(ierr);
  ierr = MatGetOwnershipRangeColumn(A,&stc,NULL);CHKERRQ(ierr);
  ierr = MatGetLocalSize(A,&dr,&dc);CHKERRQ(ierr);
  di   = diag->i;
  dj   = diag->j;
  dd   = diag->a;
  oc   = aij->B->cmap->n;
  oi   = offd->i;
  oj   = offd->j;
  od   = offd->a;
  nnz  = diag->i[dr] + offd->i[dr];

  /* generate l2g maps for rows and cols */
  ierr = ISCreateStride(comm,dr,str,1,&is);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingCreateIS(is,&rl2g);CHKERRQ(ierr);
  ierr = ISDestroy(&is);CHKERRQ(ierr);
  if (dr) {
    ierr = PetscMalloc1(dc+oc,&aux);CHKERRQ(ierr);
    for (i=0; i<dc; i++) aux[i]    = i+stc;
    for (i=0; i<oc; i++) aux[i+dc] = aij->garray[i];
    ierr = ISCreateGeneral(comm,dc+oc,aux,PETSC_OWN_POINTER,&is);CHKERRQ(ierr);
    lc   = dc+oc;
  } else {
    ierr = ISCreateGeneral(comm,0,NULL,PETSC_OWN_POINTER,&is);CHKERRQ(ierr);
    lc   = 0;
  }
  ierr = ISLocalToGlobalMappingCreateIS(is,&cl2g);CHKERRQ(ierr);
  ierr = ISDestroy(&is);CHKERRQ(ierr);

  /* create MATIS object */
  ierr = MatCreate(comm,newmat);CHKERRQ(ierr);
  ierr = MatSetSizes(*newmat,dr,dc,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = MatSetType(*newmat,MATIS);CHKERRQ(ierr);
  ierr = MatSetLocalToGlobalMapping(*newmat,rl2g,cl2g);CHKERRQ(ierr);
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
    ierr = PetscContainerSetUserDestroy(c,MatISContainerDestroyArray_Private);CHKERRQ(ierr);
    ierr = PetscObjectCompose((PetscObject)lA,names[i],(PetscObject)c);CHKERRQ(ierr);
    ierr = PetscContainerDestroy(&c);CHKERRQ(ierr);
  }

  /* finalize matrix */
  ierr = MatISSetLocalMat(*newmat,lA);CHKERRQ(ierr);
  ierr = MatDestroy(&lA);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(*newmat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*newmat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   MatISSetUpSF - Setup star forest objects used by MatIS.

   Collective on MPI_Comm

   Input Parameters:
+  A - the matrix

   Level: advanced

   Notes: This function does not need to be called by the user.

.keywords: matrix

.seealso: MatCreate(), MatCreateIS(), MatISSetPreallocation(), MatISGetLocalMat()
@*/
PetscErrorCode  MatISSetUpSF(Mat A)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidType(A,1);
  ierr = PetscTryMethod(A,"MatISSetUpSF_C",(Mat),(A));CHKERRQ(ierr);
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

  PetscFunctionBeginUser;
  ierr   = MatNestGetSubMats(A,&nr,&nc,&nest);CHKERRQ(ierr);
  lreuse = PETSC_FALSE;
  rnest  = NULL;
  if (reuse == MAT_REUSE_MATRIX) {
    PetscBool ismatis,isnest;

    ierr = PetscObjectTypeCompare((PetscObject)*newmat,MATIS,&ismatis);CHKERRQ(ierr);
    if (!ismatis) SETERRQ1(PetscObjectComm((PetscObject)*newmat),PETSC_ERR_USER,"Cannot reuse matrix of type %s",((PetscObject)(*newmat))->type);
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
  ierr = PetscCalloc6(nr,&isrow,nc,&iscol,
                      nr,&islrow,nc,&islcol,
                      nr*nc,&snest,nr*nc,&istrans);CHKERRQ(ierr);
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
        if (!ismatis) SETERRQ2(comm,PETSC_ERR_SUP,"Cannot convert from MATNEST to MATIS! Matrix block (%D,%D) (transposed) is not of type MATIS",i,j);
        ierr = MatISGetLocalMat(T,&lT);CHKERRQ(ierr);
        ierr = MatCreateTranspose(lT,&snest[ij]);CHKERRQ(ierr);
      } else {
        ierr = PetscObjectTypeCompare((PetscObject)nest[i][j],MATIS,&ismatis);CHKERRQ(ierr);
        if (!ismatis) SETERRQ2(comm,PETSC_ERR_SUP,"Cannot convert from MATNEST to MATIS! Matrix block (%D,%D) is not of type MATIS",i,j);
        ierr = MatISGetLocalMat(nest[i][j],&snest[ij]);CHKERRQ(ierr);
      }

      /* Check compatibility of local sizes */
      ierr = MatGetSize(snest[ij],&l1,&l2);CHKERRQ(ierr);
      ierr = MatGetBlockSizes(snest[ij],&lb1,&lb2);CHKERRQ(ierr);
      if (!l1 || !l2) continue;
      if (lr[i] && l1 != lr[i]) SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_SUP,"Cannot convert from MATNEST to MATIS! Matrix block (%D,%D) has invalid local size %D != %D",i,j,lr[i],l1);
      if (lc[j] && l2 != lc[j]) SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_SUP,"Cannot convert from MATNEST to MATIS! Matrix block (%D,%D) has invalid local size %D != %D",i,j,lc[j],l2);
      lr[i] = l1;
      lc[j] = l2;

      /* check compatibilty for local matrix reusage */
      if (rnest && !rnest[i][j] != !snest[ij]) lreuse = PETSC_FALSE;
    }
  }

#if defined (PETSC_USE_DEBUG)
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
        if (n1 != n2) SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_SUP,"Cannot convert from MATNEST to MATIS! Matrix block (%D,%D) has invalid row l2gmap size %D != %D",i,j,n1,n2);
        ierr = ISLocalToGlobalMappingGetIndices(cl2g,&idxs1);CHKERRQ(ierr);
        ierr = ISLocalToGlobalMappingGetIndices(rl2g,&idxs2);CHKERRQ(ierr);
        ierr = PetscMemcmp(idxs1,idxs2,n1*sizeof(PetscInt),&same);CHKERRQ(ierr);
        ierr = ISLocalToGlobalMappingRestoreIndices(cl2g,&idxs1);CHKERRQ(ierr);
        ierr = ISLocalToGlobalMappingRestoreIndices(rl2g,&idxs2);CHKERRQ(ierr);
        if (!same) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_SUP,"Cannot convert from MATNEST to MATIS! Matrix block (%D,%D) has invalid row l2gmap",i,j);
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
        if (n1 != n2) SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_SUP,"Cannot convert from MATNEST to MATIS! Matrix block (%D,%D) has invalid column l2gmap size %D != %D",j,i,n1,n2);
        ierr = ISLocalToGlobalMappingGetIndices(cl2g,&idxs1);CHKERRQ(ierr);
        ierr = ISLocalToGlobalMappingGetIndices(rl2g,&idxs2);CHKERRQ(ierr);
        ierr = PetscMemcmp(idxs1,idxs2,n1*sizeof(PetscInt),&same);CHKERRQ(ierr);
        ierr = ISLocalToGlobalMappingRestoreIndices(cl2g,&idxs1);CHKERRQ(ierr);
        ierr = ISLocalToGlobalMappingRestoreIndices(rl2g,&idxs2);CHKERRQ(ierr);
        if (!same) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_SUP,"Cannot convert from MATNEST to MATIS! Matrix block (%D,%D) has invalid column l2gmap",j,i);
      }
    }
  }
#endif

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
      if (!usedmat) SETERRQ(comm,PETSC_ERR_SUP,"Cannot find valid row mat");

      if (istrans[i*nc+j]) {
        Mat T;
        ierr    = MatTransposeGetMat(usedmat,&T);CHKERRQ(ierr);
        usedmat = T;
      }
      ierr  = MatISSetUpSF(usedmat);CHKERRQ(ierr);
      matis = (Mat_IS*)(usedmat->data);
      ierr  = ISGetIndices(isrow[i],&idxs);CHKERRQ(ierr);
      if (istrans[i*nc+j]) {
        ierr = PetscSFBcastBegin(matis->csf,MPIU_INT,idxs,l2gidxs+stl);CHKERRQ(ierr);
        ierr = PetscSFBcastEnd(matis->csf,MPIU_INT,idxs,l2gidxs+stl);CHKERRQ(ierr);
      } else {
        ierr = PetscSFBcastBegin(matis->sf,MPIU_INT,idxs,l2gidxs+stl);CHKERRQ(ierr);
        ierr = PetscSFBcastEnd(matis->sf,MPIU_INT,idxs,l2gidxs+stl);CHKERRQ(ierr);
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
      if (!usedmat) SETERRQ(comm,PETSC_ERR_SUP,"Cannot find valid column mat");
      if (istrans[j*nc+i]) {
        Mat T;
        ierr    = MatTransposeGetMat(usedmat,&T);CHKERRQ(ierr);
        usedmat = T;
      }
      ierr  = MatISSetUpSF(usedmat);CHKERRQ(ierr);
      matis = (Mat_IS*)(usedmat->data);
      ierr  = ISGetIndices(iscol[i],&idxs);CHKERRQ(ierr);
      if (istrans[j*nc+i]) {
        ierr = PetscSFBcastBegin(matis->sf,MPIU_INT,idxs,l2gidxs+stl);CHKERRQ(ierr);
        ierr = PetscSFBcastEnd(matis->sf,MPIU_INT,idxs,l2gidxs+stl);CHKERRQ(ierr);
      } else {
        ierr = PetscSFBcastBegin(matis->csf,MPIU_INT,idxs,l2gidxs+stl);CHKERRQ(ierr);
        ierr = PetscSFBcastEnd(matis->csf,MPIU_INT,idxs,l2gidxs+stl);CHKERRQ(ierr);
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
    ierr = MatSetLocalToGlobalMapping(B,rl2g,cl2g);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingDestroy(&rl2g);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingDestroy(&cl2g);CHKERRQ(ierr);
    ierr = MatCreateNest(PETSC_COMM_SELF,nr,islrow,nc,islcol,snest,&lA);CHKERRQ(ierr);
    for (i=0;i<nr*nc;i++) {
      if (istrans[i]) {
        ierr = MatDestroy(&snest[i]);CHKERRQ(ierr);
      }
    }
    ierr = MatISSetLocalMat(B,lA);CHKERRQ(ierr);
    ierr = MatDestroy(&lA);CHKERRQ(ierr);
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
      if (istrans[i]) {
        ierr = MatDestroy(&snest[i]);CHKERRQ(ierr);
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
    ierr = PetscCalloc2(nr,&lf->rf,nc,&lf->cf);CHKERRQ(ierr);
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
  ierr = MatISSetUpSF(A);CHKERRQ(ierr);
  if (l) {
    ll   = matis->y;
    ierr = VecGetArrayRead(l,&Y);CHKERRQ(ierr);
    ierr = VecGetArray(ll,&y);CHKERRQ(ierr);
    ierr = PetscSFBcastBegin(matis->sf,MPIU_SCALAR,Y,y);CHKERRQ(ierr);
  } else {
    ll = NULL;
  }
  if (r) {
    rr   = matis->x;
    ierr = VecGetArrayRead(r,&X);CHKERRQ(ierr);
    ierr = VecGetArray(rr,&x);CHKERRQ(ierr);
    ierr = PetscSFBcastBegin(matis->csf,MPIU_SCALAR,X,x);CHKERRQ(ierr);
  } else {
    rr = NULL;
  }
  if (ll) {
    ierr = PetscSFBcastEnd(matis->sf,MPIU_SCALAR,Y,y);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(l,&Y);CHKERRQ(ierr);
    ierr = VecRestoreArray(ll,&y);CHKERRQ(ierr);
  }
  if (rr) {
    ierr = PetscSFBcastEnd(matis->csf,MPIU_SCALAR,X,x);CHKERRQ(ierr);
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
  PetscReal      isend[6],irecv[6];
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
    ierr = MPIU_Allreduce(isend,irecv,6,MPIU_REAL,MPIU_MAX,PetscObjectComm((PetscObject)A));CHKERRQ(ierr);

    ginfo->nz_used      = irecv[0];
    ginfo->nz_allocated = irecv[1];
    ginfo->nz_unneeded  = irecv[2];
    ginfo->memory       = irecv[3];
    ginfo->mallocs      = irecv[4];
    ginfo->assemblies   = irecv[5];
  } else if (flag == MAT_GLOBAL_SUM) {
    ierr = MPIU_Allreduce(isend,irecv,5,MPIU_REAL,MPIU_SUM,PetscObjectComm((PetscObject)A));CHKERRQ(ierr);

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

PetscErrorCode MatTranspose_IS(Mat A,MatReuse reuse,Mat *B)
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

PetscErrorCode  MatDiagonalSet_IS(Mat A,Vec D,InsertMode insmode)
{
  Mat_IS         *is = (Mat_IS*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!D) { /* this code branch is used by MatShift_IS */
    ierr = VecSet(is->y,1.);CHKERRQ(ierr);
  } else {
    ierr = VecScatterBegin(is->rctx,D,is->y,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(is->rctx,D,is->y,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  }
  ierr = VecPointwiseDivide(is->y,is->y,is->counter);CHKERRQ(ierr);
  ierr = MatDiagonalSet(is->A,is->y,insmode);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode  MatShift_IS(Mat A,PetscScalar a)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatDiagonalSet_IS(A,NULL,ADD_VALUES);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSetValuesLocal_SubMat_IS(Mat A,PetscInt m,const PetscInt *rows, PetscInt n,const PetscInt *cols,const PetscScalar *values,InsertMode addv)
{
  PetscErrorCode ierr;
  Mat_IS         *is = (Mat_IS*)A->data;
  PetscInt       rows_l[MATIS_MAX_ENTRIES_INSERTION],cols_l[MATIS_MAX_ENTRIES_INSERTION];

  PetscFunctionBegin;
#if defined(PETSC_USE_DEBUG)
  if (m > MATIS_MAX_ENTRIES_INSERTION || n > MATIS_MAX_ENTRIES_INSERTION) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_SUP,"Number of row/column indices must be <= %D: they are %D %D",MATIS_MAX_ENTRIES_INSERTION,m,n);
#endif
  ierr = ISLocalToGlobalMappingApply(A->rmap->mapping,m,rows,rows_l);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingApply(A->cmap->mapping,n,cols,cols_l);CHKERRQ(ierr);
  ierr = MatSetValues(is->A,m,rows_l,n,cols_l,values,addv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSetValuesBlockedLocal_SubMat_IS(Mat A,PetscInt m,const PetscInt *rows, PetscInt n,const PetscInt *cols,const PetscScalar *values,InsertMode addv)
{
  PetscErrorCode ierr;
  Mat_IS         *is = (Mat_IS*)A->data;
  PetscInt       rows_l[MATIS_MAX_ENTRIES_INSERTION],cols_l[MATIS_MAX_ENTRIES_INSERTION];

  PetscFunctionBegin;
#if defined(PETSC_USE_DEBUG)
  if (m > MATIS_MAX_ENTRIES_INSERTION || n > MATIS_MAX_ENTRIES_INSERTION) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_SUP,"Number of row/column block indices must be <= %D: they are %D %D",MATIS_MAX_ENTRIES_INSERTION,m,n);
#endif
  ierr = ISLocalToGlobalMappingApplyBlock(A->rmap->mapping,m,rows,rows_l);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingApplyBlock(A->cmap->mapping,n,cols,cols_l);CHKERRQ(ierr);
  ierr = MatSetValuesBlocked(is->A,m,rows_l,n,cols_l,values,addv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscLayoutMapLocal_Private(PetscLayout map,PetscInt N,const PetscInt idxs[], PetscInt *on,PetscInt **oidxs,PetscInt **ogidxs)
{
  PetscInt      *owners = map->range;
  PetscInt       n      = map->n;
  PetscSF        sf;
  PetscInt      *lidxs,*work = NULL;
  PetscSFNode   *ridxs;
  PetscMPIInt    rank;
  PetscInt       r, p = 0, len = 0;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (on) *on = 0;              /* squelch -Wmaybe-uninitialized */
  /* Create SF where leaves are input idxs and roots are owned idxs (code adapted from MatZeroRowsMapLocal_Private) */
  ierr = MPI_Comm_rank(map->comm,&rank);CHKERRQ(ierr);
  ierr = PetscMalloc1(n,&lidxs);CHKERRQ(ierr);
  for (r = 0; r < n; ++r) lidxs[r] = -1;
  ierr = PetscMalloc1(N,&ridxs);CHKERRQ(ierr);
  for (r = 0; r < N; ++r) {
    const PetscInt idx = idxs[r];
    if (idx < 0 || map->N <= idx) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Index %D out of range [0,%D)",idx,map->N);
    if (idx < owners[p] || owners[p+1] <= idx) { /* short-circuit the search if the last p owns this idx too */
      ierr = PetscLayoutFindOwner(map,idx,&p);CHKERRQ(ierr);
    }
    ridxs[r].rank = p;
    ridxs[r].index = idxs[r] - owners[p];
  }
  ierr = PetscSFCreate(map->comm,&sf);CHKERRQ(ierr);
  ierr = PetscSFSetGraph(sf,n,N,NULL,PETSC_OWN_POINTER,ridxs,PETSC_OWN_POINTER);CHKERRQ(ierr);
  ierr = PetscSFReduceBegin(sf,MPIU_INT,(PetscInt*)idxs,lidxs,MPI_LOR);CHKERRQ(ierr);
  ierr = PetscSFReduceEnd(sf,MPIU_INT,(PetscInt*)idxs,lidxs,MPI_LOR);CHKERRQ(ierr);
  if (ogidxs) { /* communicate global idxs */
    PetscInt cum = 0,start,*work2;

    ierr = PetscMalloc1(n,&work);CHKERRQ(ierr);
    ierr = PetscCalloc1(N,&work2);CHKERRQ(ierr);
    for (r = 0; r < N; ++r) if (idxs[r] >=0) cum++;
    ierr = MPI_Scan(&cum,&start,1,MPIU_INT,MPI_SUM,map->comm);CHKERRQ(ierr);
    start -= cum;
    cum = 0;
    for (r = 0; r < N; ++r) if (idxs[r] >=0) work2[r] = start+cum++;
    ierr = PetscSFReduceBegin(sf,MPIU_INT,work2,work,MPIU_REPLACE);CHKERRQ(ierr);
    ierr = PetscSFReduceEnd(sf,MPIU_INT,work2,work,MPIU_REPLACE);CHKERRQ(ierr);
    ierr = PetscFree(work2);CHKERRQ(ierr);
  }
  ierr = PetscSFDestroy(&sf);CHKERRQ(ierr);
  /* Compress and put in indices */
  for (r = 0; r < n; ++r)
    if (lidxs[r] >= 0) {
      if (work) work[len] = work[r];
      lidxs[len++] = r;
    }
  if (on) *on = len;
  if (oidxs) *oidxs = lidxs;
  if (ogidxs) *ogidxs = work;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatCreateSubMatrix_IS(Mat mat,IS irow,IS icol,MatReuse scall,Mat *newmat)
{
  Mat               locmat,newlocmat;
  Mat_IS            *newmatis;
#if defined(PETSC_USE_DEBUG)
  Vec               rtest,ltest;
  const PetscScalar *array;
#endif
  const PetscInt    *idxs;
  PetscInt          i,m,n;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (scall == MAT_REUSE_MATRIX) {
    PetscBool ismatis;

    ierr = PetscObjectTypeCompare((PetscObject)*newmat,MATIS,&ismatis);CHKERRQ(ierr);
    if (!ismatis) SETERRQ(PetscObjectComm((PetscObject)*newmat),PETSC_ERR_ARG_WRONG,"Cannot reuse matrix! Not of MATIS type");
    newmatis = (Mat_IS*)(*newmat)->data;
    if (!newmatis->getsub_ris) SETERRQ(PetscObjectComm((PetscObject)*newmat),PETSC_ERR_ARG_WRONG,"Cannot reuse matrix! Misses local row IS");
    if (!newmatis->getsub_cis) SETERRQ(PetscObjectComm((PetscObject)*newmat),PETSC_ERR_ARG_WRONG,"Cannot reuse matrix! Misses local col IS");
  }
  /* irow and icol may not have duplicate entries */
#if defined(PETSC_USE_DEBUG)
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
  for (i=0;i<n;i++) if (array[i] != 0. && array[i] != 1.) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_SUP,"Index %D counted %D times! Irow may not have duplicate entries",i+m,(PetscInt)PetscRealPart(array[i]));
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
  for (i=0;i<n;i++) if (array[i] != 0. && array[i] != 1.) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_SUP,"Index %D counted %D times! Icol may not have duplicate entries",i+m,(PetscInt)PetscRealPart(array[i]));
  ierr = VecRestoreArrayRead(ltest,&array);CHKERRQ(ierr);
  ierr = ISRestoreIndices(icol,&idxs);CHKERRQ(ierr);
  ierr = VecDestroy(&rtest);CHKERRQ(ierr);
  ierr = VecDestroy(&ltest);CHKERRQ(ierr);
#endif
  if (scall == MAT_INITIAL_MATRIX) {
    Mat_IS                 *matis = (Mat_IS*)mat->data;
    ISLocalToGlobalMapping rl2g;
    IS                     is;
    PetscInt               *lidxs,*lgidxs,*newgidxs;
    PetscInt               ll,newloc;
    MPI_Comm               comm;

    ierr = PetscObjectGetComm((PetscObject)mat,&comm);CHKERRQ(ierr);
    ierr = ISGetLocalSize(irow,&m);CHKERRQ(ierr);
    ierr = ISGetLocalSize(icol,&n);CHKERRQ(ierr);
    ierr = MatCreate(comm,newmat);CHKERRQ(ierr);
    ierr = MatSetType(*newmat,MATIS);CHKERRQ(ierr);
    ierr = MatSetSizes(*newmat,m,n,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
    /* communicate irow to their owners in the layout */
    ierr = ISGetIndices(irow,&idxs);CHKERRQ(ierr);
    ierr = PetscLayoutMapLocal_Private(mat->rmap,m,idxs,&ll,&lidxs,&lgidxs);CHKERRQ(ierr);
    ierr = ISRestoreIndices(irow,&idxs);CHKERRQ(ierr);
    ierr = MatISSetUpSF(mat);CHKERRQ(ierr);
    ierr = PetscMemzero(matis->sf_rootdata,matis->sf->nroots*sizeof(PetscInt));CHKERRQ(ierr);
    for (i=0;i<ll;i++) matis->sf_rootdata[lidxs[i]] = lgidxs[i]+1;
    ierr = PetscFree(lidxs);CHKERRQ(ierr);
    ierr = PetscFree(lgidxs);CHKERRQ(ierr);
    ierr = PetscSFBcastBegin(matis->sf,MPIU_INT,matis->sf_rootdata,matis->sf_leafdata);CHKERRQ(ierr);
    ierr = PetscSFBcastEnd(matis->sf,MPIU_INT,matis->sf_rootdata,matis->sf_leafdata);CHKERRQ(ierr);
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
    ierr = ISDestroy(&is);CHKERRQ(ierr);
    /* local is to extract local submatrix */
    newmatis = (Mat_IS*)(*newmat)->data;
    ierr = ISCreateGeneral(comm,newloc,lidxs,PETSC_OWN_POINTER,&newmatis->getsub_ris);CHKERRQ(ierr);
    if (mat->congruentlayouts == -1) { /* first time we compare rows and cols layouts */
      PetscBool cong;
      ierr = PetscLayoutCompare(mat->rmap,mat->cmap,&cong);CHKERRQ(ierr);
      if (cong) mat->congruentlayouts = 1;
      else      mat->congruentlayouts = 0;
    }
    if (mat->congruentlayouts && irow == icol) {
      ierr = MatSetLocalToGlobalMapping(*newmat,rl2g,rl2g);CHKERRQ(ierr);
      ierr = PetscObjectReference((PetscObject)newmatis->getsub_ris);CHKERRQ(ierr);
      newmatis->getsub_cis = newmatis->getsub_ris;
    } else {
      ISLocalToGlobalMapping cl2g;

      /* communicate icol to their owners in the layout */
      ierr = ISGetIndices(icol,&idxs);CHKERRQ(ierr);
      ierr = PetscLayoutMapLocal_Private(mat->cmap,n,idxs,&ll,&lidxs,&lgidxs);CHKERRQ(ierr);
      ierr = ISRestoreIndices(icol,&idxs);CHKERRQ(ierr);
      ierr = PetscMemzero(matis->csf_rootdata,matis->csf->nroots*sizeof(PetscInt));CHKERRQ(ierr);
      for (i=0;i<ll;i++) matis->csf_rootdata[lidxs[i]] = lgidxs[i]+1;
      ierr = PetscFree(lidxs);CHKERRQ(ierr);
      ierr = PetscFree(lgidxs);CHKERRQ(ierr);
      ierr = PetscSFBcastBegin(matis->csf,MPIU_INT,matis->csf_rootdata,matis->csf_leafdata);CHKERRQ(ierr);
      ierr = PetscSFBcastEnd(matis->csf,MPIU_INT,matis->csf_rootdata,matis->csf_leafdata);CHKERRQ(ierr);
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
  if (!ismatis) SETERRQ(PetscObjectComm((PetscObject)B),PETSC_ERR_SUP,"Need to be implemented");
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
   MatISSetPreallocation - Preallocates memory for a MATIS parallel matrix.

   Collective on MPI_Comm

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

   Notes: This function has the same interface as the MPIAIJ preallocation routine in order to simplify the transition
          from the asssembled format to the unassembled one. It overestimates the preallocation of MATIS local
          matrices; for exact preallocation, the user should set the preallocation directly on local matrix objects.

.keywords: matrix

.seealso: MatCreate(), MatCreateIS(), MatMPIAIJSetPreallocation(), MatISGetLocalMat(), MATIS
@*/
PetscErrorCode  MatISSetPreallocation(Mat B,PetscInt d_nz,const PetscInt d_nnz[],PetscInt o_nz,const PetscInt o_nnz[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(B,MAT_CLASSID,1);
  PetscValidType(B,1);
  ierr = PetscTryMethod(B,"MatISSetPreallocation_C",(Mat,PetscInt,const PetscInt[],PetscInt,const PetscInt[]),(B,d_nz,d_nnz,o_nz,o_nnz));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode  MatISSetPreallocation_IS(Mat B,PetscInt d_nz,const PetscInt d_nnz[],PetscInt o_nz,const PetscInt o_nnz[])
{
  Mat_IS         *matis = (Mat_IS*)(B->data);
  PetscInt       bs,i,nlocalcols;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!matis->A) SETERRQ(PetscObjectComm((PetscObject)B),PETSC_ERR_SUP,"You should first call MatSetLocalToGlobalMapping");
  ierr = MatISSetUpSF(B);CHKERRQ(ierr);

  if (!d_nnz) for (i=0;i<matis->sf->nroots;i++) matis->sf_rootdata[i] = d_nz;
  else for (i=0;i<matis->sf->nroots;i++) matis->sf_rootdata[i] = d_nnz[i];

  if (!o_nnz) for (i=0;i<matis->sf->nroots;i++) matis->sf_rootdata[i] += o_nz;
  else for (i=0;i<matis->sf->nroots;i++) matis->sf_rootdata[i] += o_nnz[i];

  ierr = PetscSFBcastBegin(matis->sf,MPIU_INT,matis->sf_rootdata,matis->sf_leafdata);CHKERRQ(ierr);
  ierr = MatGetSize(matis->A,NULL,&nlocalcols);CHKERRQ(ierr);
  ierr = MatGetBlockSize(matis->A,&bs);CHKERRQ(ierr);
  ierr = PetscSFBcastEnd(matis->sf,MPIU_INT,matis->sf_rootdata,matis->sf_leafdata);CHKERRQ(ierr);

  for (i=0;i<matis->sf->nleaves;i++) matis->sf_leafdata[i] = PetscMin(matis->sf_leafdata[i],nlocalcols);
  ierr = MatSeqAIJSetPreallocation(matis->A,0,matis->sf_leafdata);CHKERRQ(ierr);
#if defined(PETSC_HAVE_HYPRE)
  ierr = MatHYPRESetPreallocation(matis->A,0,matis->sf_leafdata,0,NULL);CHKERRQ(ierr);
#endif

  for (i=0;i<matis->sf->nleaves/bs;i++) matis->sf_leafdata[i] = matis->sf_leafdata[i*bs]/bs;
  ierr = MatSeqBAIJSetPreallocation(matis->A,bs,0,matis->sf_leafdata);CHKERRQ(ierr);

  for (i=0;i<matis->sf->nleaves/bs;i++) matis->sf_leafdata[i] = matis->sf_leafdata[i]-i;
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
  PetscMPIInt     nsubdomains;
  PetscBool       isdense,issbaij;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)A),&nsubdomains);CHKERRQ(ierr);
  ierr = MatGetSize(A,&rows,&cols);CHKERRQ(ierr);
  ierr = MatGetBlockSize(A,&bs);CHKERRQ(ierr);
  ierr = MatGetSize(matis->A,&local_rows,&local_cols);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)matis->A,MATSEQDENSE,&isdense);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)matis->A,MATSEQSBAIJ,&issbaij);CHKERRQ(ierr);
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
  ierr = MatISSetUpSF(A);CHKERRQ(ierr);
  ierr = MatGetLocalSize(A,&lrows,&lcols);CHKERRQ(ierr);
  ierr = MatPreallocateInitialize(PetscObjectComm((PetscObject)A),lrows,lcols,dnz,onz);CHKERRQ(ierr);
  /* All processes need to compute entire row ownership */
  ierr = PetscMalloc1(rows,&row_ownership);CHKERRQ(ierr);
  ierr = MatGetOwnershipRanges(A,(const PetscInt**)&mat_ranges);CHKERRQ(ierr);
  for (i=0;i<nsubdomains;i++) {
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
        if (index_col > mat_ranges[owner]-1 && index_col < mat_ranges[owner+1] ) { /* diag block */
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
    if (!done) SETERRQ(PetscObjectComm((PetscObject)(matis->A)),PETSC_ERR_PLIB,"Error in MatGetRowIJ");
    jptr = jj;
    for (i=0;i<local_rows;i++) {
      PetscInt index_row = global_indices_r[i];
      for (j=0;j<ii[i+1]-ii[i];j++,jptr++) {
        PetscInt owner = row_ownership[index_row];
        PetscInt index_col = global_indices_c[*jptr];
        if (index_col > mat_ranges[owner]-1 && index_col < mat_ranges[owner+1] ) { /* diag block */
          my_dnz[i] += 1;
        } else { /* offdiag block */
          my_onz[i] += 1;
        }
        /* same as before, interchanging rows and cols */
        if (issbaij && index_col != index_row) {
          owner = row_ownership[index_col];
          if (index_row > mat_ranges[owner]-1 && index_row < mat_ranges[owner+1] ) {
            my_dnz[*jptr] += 1;
          } else {
            my_onz[*jptr] += 1;
          }
        }
      }
    }
    ierr = MatRestoreRowIJ(matis->A,0,PETSC_FALSE,PETSC_FALSE,&local_rows,&ii,&jj,&done);CHKERRQ(ierr);
    if (!done) SETERRQ(PetscObjectComm((PetscObject)(matis->A)),PETSC_ERR_PLIB,"Error in MatRestoreRowIJ");
  } else { /* loop over rows and use MatGetRow */
    for (i=0;i<local_rows;i++) {
      const PetscInt *cols;
      PetscInt       ncols,index_row = global_indices_r[i];
      ierr = MatGetRow(matis->A,i,&ncols,&cols,NULL);CHKERRQ(ierr);
      for (j=0;j<ncols;j++) {
        PetscInt owner = row_ownership[index_row];
        PetscInt index_col = global_indices_c[cols[j]];
        if (index_col > mat_ranges[owner]-1 && index_col < mat_ranges[owner+1] ) { /* diag block */
          my_dnz[i] += 1;
        } else { /* offdiag block */
          my_onz[i] += 1;
        }
        /* same as before, interchanging rows and cols */
        if (issbaij && index_col != index_row) {
          owner = row_ownership[index_col];
          if (index_row > mat_ranges[owner]-1 && index_row < mat_ranges[owner+1] ) {
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
  ierr = MatMPIAIJSetPreallocation(B,0,dnz,0,onz);CHKERRQ(ierr);
  for (i=0;i<lrows/bs;i++) {
    dnz[i] = dnz[i*bs]/bs;
    onz[i] = onz[i*bs]/bs;
  }
  ierr = MatMPIBAIJSetPreallocation(B,bs,0,dnz,0,onz);CHKERRQ(ierr);
  ierr = MatMPISBAIJSetPreallocation(B,bs,0,dnz,0,onz);CHKERRQ(ierr);
  ierr = MatPreallocateFinalize(dnz,onz);CHKERRQ(ierr);
  if (issbaij) {
    ierr = MatRestoreRowUpperTriangular(matis->A);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatISGetMPIXAIJ_IS(Mat mat, MatReuse reuse, Mat *M)
{
  Mat_IS         *matis = (Mat_IS*)(mat->data);
  Mat            local_mat;
  /* info on mat */
  PetscInt       bs,rows,cols,lrows,lcols;
  PetscInt       local_rows,local_cols;
  PetscBool      isseqdense,isseqsbaij,isseqaij,isseqbaij;
#if defined (PETSC_USE_DEBUG)
  PetscBool      lb[4],bb[4];
#endif
  PetscMPIInt    nsubdomains;
  /* values insertion */
  PetscScalar    *array;
  /* work */
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* get info from mat */
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)mat),&nsubdomains);CHKERRQ(ierr);
  if (nsubdomains == 1) {
    Mat            B;
    IS             rows,cols;
    IS             irows,icols;
    const PetscInt *ridxs,*cidxs;

    ierr = ISLocalToGlobalMappingGetIndices(mat->rmap->mapping,&ridxs);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingGetIndices(mat->cmap->mapping,&cidxs);CHKERRQ(ierr);
    ierr = ISCreateGeneral(PETSC_COMM_SELF,mat->rmap->n,ridxs,PETSC_USE_POINTER,&rows);CHKERRQ(ierr);
    ierr = ISCreateGeneral(PETSC_COMM_SELF,mat->cmap->n,cidxs,PETSC_USE_POINTER,&cols);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingRestoreIndices(mat->rmap->mapping,&ridxs);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingRestoreIndices(mat->cmap->mapping,&cidxs);CHKERRQ(ierr);
    ierr = ISSetPermutation(rows);CHKERRQ(ierr);
    ierr = ISSetPermutation(cols);CHKERRQ(ierr);
    ierr = ISInvertPermutation(rows,mat->rmap->n,&irows);CHKERRQ(ierr);
    ierr = ISInvertPermutation(cols,mat->cmap->n,&icols);CHKERRQ(ierr);
    ierr = ISDestroy(&cols);CHKERRQ(ierr);
    ierr = ISDestroy(&rows);CHKERRQ(ierr);
    ierr = MatConvert(matis->A,MATSEQAIJ,MAT_INITIAL_MATRIX,&B);CHKERRQ(ierr);
    ierr = MatCreateSubMatrix(B,irows,icols,reuse,M);CHKERRQ(ierr);
    ierr = MatDestroy(&B);CHKERRQ(ierr);
    ierr = ISDestroy(&icols);CHKERRQ(ierr);
    ierr = ISDestroy(&irows);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  ierr = MatGetSize(mat,&rows,&cols);CHKERRQ(ierr);
  ierr = MatGetBlockSize(mat,&bs);CHKERRQ(ierr);
  ierr = MatGetLocalSize(mat,&lrows,&lcols);CHKERRQ(ierr);
  ierr = MatGetSize(matis->A,&local_rows,&local_cols);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)matis->A,MATSEQDENSE,&isseqdense);CHKERRQ(ierr);
  ierr = PetscObjectBaseTypeCompare((PetscObject)matis->A,MATSEQAIJ,&isseqaij);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)matis->A,MATSEQBAIJ,&isseqbaij);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)matis->A,MATSEQSBAIJ,&isseqsbaij);CHKERRQ(ierr);
  if (!isseqdense && !isseqaij && !isseqbaij && !isseqsbaij) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Not for matrix type %s",((PetscObject)(matis->A))->type_name);
#if defined (PETSC_USE_DEBUG)
  lb[0] = isseqdense;
  lb[1] = isseqaij;
  lb[2] = isseqbaij;
  lb[3] = isseqsbaij;
  ierr = MPIU_Allreduce(lb,bb,4,MPIU_BOOL,MPI_LAND,PetscObjectComm((PetscObject)mat));CHKERRQ(ierr);
  if (!bb[0] && !bb[1] && !bb[2] && !bb[3]) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Local matrices must have the same type");
#endif

  if (reuse == MAT_INITIAL_MATRIX) {
    ierr = MatCreate(PetscObjectComm((PetscObject)mat),M);CHKERRQ(ierr);
    ierr = MatSetSizes(*M,lrows,lcols,rows,cols);CHKERRQ(ierr);
    ierr = MatSetBlockSize(*M,bs);CHKERRQ(ierr);
    if (!isseqsbaij) {
      ierr = MatSetType(*M,MATAIJ);CHKERRQ(ierr);
    } else {
      ierr = MatSetType(*M,MATSBAIJ);CHKERRQ(ierr);
    }
    ierr = MatISSetMPIXAIJPreallocation_Private(mat,*M,PETSC_FALSE);CHKERRQ(ierr);
  } else {
    PetscInt mbs,mrows,mcols,mlrows,mlcols;
    /* some checks */
    ierr = MatGetBlockSize(*M,&mbs);CHKERRQ(ierr);
    ierr = MatGetSize(*M,&mrows,&mcols);CHKERRQ(ierr);
    ierr = MatGetLocalSize(*M,&mlrows,&mlcols);CHKERRQ(ierr);
    if (mrows != rows) SETERRQ2(PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Cannot reuse matrix. Wrong number of rows (%d != %d)",rows,mrows);
    if (mcols != cols) SETERRQ2(PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Cannot reuse matrix. Wrong number of cols (%d != %d)",cols,mcols);
    if (mlrows != lrows) SETERRQ2(PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Cannot reuse matrix. Wrong number of local rows (%d != %d)",lrows,mlrows);
    if (mlcols != lcols) SETERRQ2(PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Cannot reuse matrix. Wrong number of local cols (%d != %d)",lcols,mlcols);
    if (mbs != bs) SETERRQ2(PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Cannot reuse matrix. Wrong block size (%d != %d)",bs,mbs);
    ierr = MatZeroEntries(*M);CHKERRQ(ierr);
  }

  if (isseqsbaij) {
    ierr = MatConvert(matis->A,MATSEQBAIJ,MAT_INITIAL_MATRIX,&local_mat);CHKERRQ(ierr);
  } else {
    ierr = PetscObjectReference((PetscObject)matis->A);CHKERRQ(ierr);
    local_mat = matis->A;
  }

  /* Set values */
  ierr = MatSetLocalToGlobalMapping(*M,mat->rmap->mapping,mat->cmap->mapping);CHKERRQ(ierr);
  if (isseqdense) { /* special case for dense local matrices */
    PetscInt i,*dummy;

    ierr = PetscMalloc1(PetscMax(local_rows,local_cols),&dummy);CHKERRQ(ierr);
    for (i=0;i<PetscMax(local_rows,local_cols);i++) dummy[i] = i;
    ierr = MatSetOption(*M,MAT_ROW_ORIENTED,PETSC_FALSE);CHKERRQ(ierr);
    ierr = MatDenseGetArray(local_mat,&array);CHKERRQ(ierr);
    ierr = MatSetValuesLocal(*M,local_rows,dummy,local_cols,dummy,array,ADD_VALUES);CHKERRQ(ierr);
    ierr = MatDenseRestoreArray(local_mat,&array);CHKERRQ(ierr);
    ierr = PetscFree(dummy);CHKERRQ(ierr);
  } else if (isseqaij) {
    PetscInt  i,nvtxs,*xadj,*adjncy;
    PetscBool done;

    ierr = MatGetRowIJ(local_mat,0,PETSC_FALSE,PETSC_FALSE,&nvtxs,(const PetscInt**)&xadj,(const PetscInt**)&adjncy,&done);CHKERRQ(ierr);
    if (!done) SETERRQ(PetscObjectComm((PetscObject)local_mat),PETSC_ERR_PLIB,"Error in MatGetRowIJ");
    ierr = MatSeqAIJGetArray(local_mat,&array);CHKERRQ(ierr);
    for (i=0;i<nvtxs;i++) {
      ierr = MatSetValuesLocal(*M,1,&i,xadj[i+1]-xadj[i],adjncy+xadj[i],array+xadj[i],ADD_VALUES);CHKERRQ(ierr);
    }
    ierr = MatRestoreRowIJ(local_mat,0,PETSC_FALSE,PETSC_FALSE,&nvtxs,(const PetscInt**)&xadj,(const PetscInt**)&adjncy,&done);CHKERRQ(ierr);
    if (!done) SETERRQ(PetscObjectComm((PetscObject)local_mat),PETSC_ERR_PLIB,"Error in MatRestoreRowIJ");
    ierr = MatSeqAIJRestoreArray(local_mat,&array);CHKERRQ(ierr);
  } else { /* very basic values insertion for all other matrix types */
    PetscInt i;

    for (i=0;i<local_rows;i++) {
      PetscInt       j;
      const PetscInt *local_indices_cols;

      ierr = MatGetRow(local_mat,i,&j,&local_indices_cols,(const PetscScalar**)&array);CHKERRQ(ierr);
      ierr = MatSetValuesLocal(*M,1,&i,j,local_indices_cols,array,ADD_VALUES);CHKERRQ(ierr);
      ierr = MatRestoreRow(local_mat,i,&j,&local_indices_cols,(const PetscScalar**)&array);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(*M,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatDestroy(&local_mat);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*M,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (isseqdense) {
    ierr = MatSetOption(*M,MAT_ROW_ORIENTED,PETSC_TRUE);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@
    MatISGetMPIXAIJ - Converts MATIS matrix into a parallel AIJ format

  Input Parameter:
.  mat - the matrix (should be of type MATIS)
.  reuse - either MAT_INITIAL_MATRIX or MAT_REUSE_MATRIX

  Output Parameter:
.  newmat - the matrix in AIJ format

  Level: developer

  Notes: mat and *newmat cannot be the same object when MAT_REUSE_MATRIX is requested.

.seealso: MATIS
@*/
PetscErrorCode MatISGetMPIXAIJ(Mat mat, MatReuse reuse, Mat *newmat)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidLogicalCollectiveEnum(mat,reuse,2);
  PetscValidPointer(newmat,3);
  if (reuse != MAT_INITIAL_MATRIX) {
    PetscValidHeaderSpecific(*newmat,MAT_CLASSID,3);
    PetscCheckSameComm(mat,1,*newmat,3);
    if (mat == *newmat) SETERRQ(PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Cannot reuse the same matrix");
  }
  ierr = PetscUseMethod(mat,"MatISGetMPIXAIJ_C",(Mat,MatReuse,Mat*),(mat,reuse,newmat));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatDuplicate_IS(Mat mat,MatDuplicateOption op,Mat *newmat)
{
  PetscErrorCode ierr;
  Mat_IS         *matis = (Mat_IS*)(mat->data);
  PetscInt       bs,m,n,M,N;
  Mat            B,localmat;

  PetscFunctionBegin;
  ierr = MatGetBlockSize(mat,&bs);CHKERRQ(ierr);
  ierr = MatGetSize(mat,&M,&N);CHKERRQ(ierr);
  ierr = MatGetLocalSize(mat,&m,&n);CHKERRQ(ierr);
  ierr = MatCreateIS(PetscObjectComm((PetscObject)mat),bs,m,n,M,N,mat->rmap->mapping,mat->cmap->mapping,&B);CHKERRQ(ierr);
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
  ierr = MPIU_Allreduce(&local_sym,flg,1,MPIU_BOOL,MPI_LAND,PetscObjectComm((PetscObject)A));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatIsSymmetric_IS(Mat A,PetscReal tol,PetscBool  *flg)
{
  PetscErrorCode ierr;
  Mat_IS         *matis = (Mat_IS*)A->data;
  PetscBool      local_sym;

  PetscFunctionBegin;
  ierr = MatIsSymmetric(matis->A,tol,&local_sym);CHKERRQ(ierr);
  ierr = MPIU_Allreduce(&local_sym,flg,1,MPIU_BOOL,MPI_LAND,PetscObjectComm((PetscObject)A));CHKERRQ(ierr);
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
  ierr = MPIU_Allreduce(&local_sym,flg,1,MPIU_BOOL,MPI_LAND,PetscObjectComm((PetscObject)A));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDestroy_IS(Mat A)
{
  PetscErrorCode ierr;
  Mat_IS         *b = (Mat_IS*)A->data;

  PetscFunctionBegin;
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
  }
  ierr = PetscSFDestroy(&b->sf);CHKERRQ(ierr);
  ierr = PetscFree2(b->sf_rootdata,b->sf_leafdata);CHKERRQ(ierr);
  ierr = PetscFree(A->data);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)A,0);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatISGetLocalMat_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatISSetLocalMat_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatISGetMPIXAIJ_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatISSetPreallocation_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatISSetUpSF_C",NULL);CHKERRQ(ierr);
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

static PetscErrorCode MatSetLocalToGlobalMapping_IS(Mat A,ISLocalToGlobalMapping rmapping,ISLocalToGlobalMapping cmapping)
{
  PetscErrorCode ierr;
  PetscInt       nr,rbs,nc,cbs;
  Mat_IS         *is = (Mat_IS*)A->data;
  IS             from,to;
  Vec            cglobal,rglobal;

  PetscFunctionBegin;
  PetscCheckSameComm(A,1,rmapping,2);
  PetscCheckSameComm(A,1,cmapping,3);
  /* Destroy any previous data */
  ierr = VecDestroy(&is->x);CHKERRQ(ierr);
  ierr = VecDestroy(&is->y);CHKERRQ(ierr);
  ierr = VecDestroy(&is->counter);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&is->rctx);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&is->cctx);CHKERRQ(ierr);
  ierr = MatDestroy(&is->A);CHKERRQ(ierr);
  if (is->csf != is->sf) {
    ierr = PetscSFDestroy(&is->csf);CHKERRQ(ierr);
    ierr = PetscFree2(is->csf_rootdata,is->csf_leafdata);CHKERRQ(ierr);
  }
  ierr = PetscSFDestroy(&is->sf);CHKERRQ(ierr);
  ierr = PetscFree2(is->sf_rootdata,is->sf_leafdata);CHKERRQ(ierr);

  /* Setup Layout and set local to global maps */
  ierr = PetscLayoutSetUp(A->rmap);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(A->cmap);CHKERRQ(ierr);
  ierr = PetscLayoutSetISLocalToGlobalMapping(A->rmap,rmapping);CHKERRQ(ierr);
  ierr = PetscLayoutSetISLocalToGlobalMapping(A->cmap,cmapping);CHKERRQ(ierr);

  /* Create the local matrix A */
  ierr = ISLocalToGlobalMappingGetSize(rmapping,&nr);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingGetBlockSize(rmapping,&rbs);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingGetSize(cmapping,&nc);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingGetBlockSize(cmapping,&cbs);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_SELF,&is->A);CHKERRQ(ierr);
  ierr = MatSetType(is->A,MATAIJ);CHKERRQ(ierr);
  ierr = MatSetSizes(is->A,nr,nc,nr,nc);CHKERRQ(ierr);
  ierr = MatSetBlockSizes(is->A,rbs,cbs);CHKERRQ(ierr);
  ierr = MatSetOptionsPrefix(is->A,((PetscObject)A)->prefix);CHKERRQ(ierr);
  ierr = MatAppendOptionsPrefix(is->A,"is_");CHKERRQ(ierr);
  ierr = MatSetFromOptions(is->A);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(is->A->rmap);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(is->A->cmap);CHKERRQ(ierr);

  if (!is->islocalref) { /* setup scatters and local vectors for MatMult */
    /* Create the local work vectors */
    ierr = MatCreateVecs(is->A,&is->x,&is->y);CHKERRQ(ierr);

    /* setup the global to local scatters */
    ierr = MatCreateVecs(A,&cglobal,&rglobal);CHKERRQ(ierr);
    ierr = ISCreateStride(PETSC_COMM_SELF,nr,0,1,&to);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingApplyIS(rmapping,to,&from);CHKERRQ(ierr);
    ierr = VecScatterCreate(rglobal,from,is->y,to,&is->rctx);CHKERRQ(ierr);
    if (rmapping != cmapping) {
      ierr = ISDestroy(&to);CHKERRQ(ierr);
      ierr = ISDestroy(&from);CHKERRQ(ierr);
      ierr = ISCreateStride(PETSC_COMM_SELF,nc,0,1,&to);CHKERRQ(ierr);
      ierr = ISLocalToGlobalMappingApplyIS(cmapping,to,&from);CHKERRQ(ierr);
      ierr = VecScatterCreate(cglobal,from,is->x,to,&is->cctx);CHKERRQ(ierr);
    } else {
      ierr = PetscObjectReference((PetscObject)is->rctx);CHKERRQ(ierr);
      is->cctx = is->rctx;
    }

    /* interface counter vector (local) */
    ierr = VecDuplicate(is->y,&is->counter);CHKERRQ(ierr);
    ierr = VecSet(is->y,1.);CHKERRQ(ierr);
    ierr = VecScatterBegin(is->rctx,is->y,rglobal,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterEnd(is->rctx,is->y,rglobal,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterBegin(is->rctx,rglobal,is->counter,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(is->rctx,rglobal,is->counter,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);

    /* free workspace */
    ierr = VecDestroy(&rglobal);CHKERRQ(ierr);
    ierr = VecDestroy(&cglobal);CHKERRQ(ierr);
    ierr = ISDestroy(&to);CHKERRQ(ierr);
    ierr = ISDestroy(&from);CHKERRQ(ierr);
  }
  ierr = MatSetUp(A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSetValues_IS(Mat mat, PetscInt m,const PetscInt *rows, PetscInt n,const PetscInt *cols, const PetscScalar *values, InsertMode addv)
{
  Mat_IS         *is = (Mat_IS*)mat->data;
  PetscErrorCode ierr;
#if defined(PETSC_USE_DEBUG)
  PetscInt       i,zm,zn;
#endif
  PetscInt       rows_l[MATIS_MAX_ENTRIES_INSERTION],cols_l[MATIS_MAX_ENTRIES_INSERTION];

  PetscFunctionBegin;
#if defined(PETSC_USE_DEBUG)
  if (m > MATIS_MAX_ENTRIES_INSERTION || n > MATIS_MAX_ENTRIES_INSERTION) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_SUP,"Number of row/column indices must be <= %D: they are %D %D",MATIS_MAX_ENTRIES_INSERTION,m,n);
  /* count negative indices */
  for (i=0,zm=0;i<m;i++) if (rows[i] < 0) zm++;
  for (i=0,zn=0;i<n;i++) if (cols[i] < 0) zn++;
#endif
  ierr = ISGlobalToLocalMappingApply(mat->rmap->mapping,IS_GTOLM_MASK,m,rows,&m,rows_l);CHKERRQ(ierr);
  ierr = ISGlobalToLocalMappingApply(mat->cmap->mapping,IS_GTOLM_MASK,n,cols,&n,cols_l);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
  /* count negative indices (should be the same as before) */
  for (i=0;i<m;i++) if (rows_l[i] < 0) zm--;
  for (i=0;i<n;i++) if (cols_l[i] < 0) zn--;
  /* disable check when usesetlocal is true */
  if (!is->usesetlocal && zm) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Some of the row indices can not be mapped! Maybe you should not use MATIS");
  if (!is->usesetlocal && zn) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Some of the column indices can not be mapped! Maybe you should not use MATIS");
#endif
  if (is->usesetlocal) {
    ierr = MatSetValuesLocal(is->A,m,rows_l,n,cols_l,values,addv);CHKERRQ(ierr);
  } else {
    ierr = MatSetValues(is->A,m,rows_l,n,cols_l,values,addv);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSetValuesBlocked_IS(Mat mat, PetscInt m,const PetscInt *rows, PetscInt n,const PetscInt *cols, const PetscScalar *values, InsertMode addv)
{
  Mat_IS         *is = (Mat_IS*)mat->data;
  PetscErrorCode ierr;
#if defined(PETSC_USE_DEBUG)
  PetscInt       i,zm,zn;
#endif
  PetscInt       rows_l[MATIS_MAX_ENTRIES_INSERTION],cols_l[MATIS_MAX_ENTRIES_INSERTION];

  PetscFunctionBegin;
#if defined(PETSC_USE_DEBUG)
  if (m > MATIS_MAX_ENTRIES_INSERTION || n > MATIS_MAX_ENTRIES_INSERTION) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_SUP,"Number of row/column block indices must be <= %D: they are %D %D",MATIS_MAX_ENTRIES_INSERTION,m,n);
  /* count negative indices */
  for (i=0,zm=0;i<m;i++) if (rows[i] < 0) zm++;
  for (i=0,zn=0;i<n;i++) if (cols[i] < 0) zn++;
#endif
  ierr = ISGlobalToLocalMappingApplyBlock(mat->rmap->mapping,IS_GTOLM_MASK,m,rows,&m,rows_l);CHKERRQ(ierr);
  ierr = ISGlobalToLocalMappingApplyBlock(mat->cmap->mapping,IS_GTOLM_MASK,n,cols,&n,cols_l);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
  /* count negative indices (should be the same as before) */
  for (i=0;i<m;i++) if (rows_l[i] < 0) zm--;
  for (i=0;i<n;i++) if (cols_l[i] < 0) zn--;
  if (zm) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Some of the row indices can not be mapped! Maybe you should not use MATIS");
  if (zn) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Some of the column indices can not be mapped! Maybe you should not use MATIS");
#endif
  ierr = MatSetValuesBlocked(is->A,m,rows_l,n,cols_l,values,addv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSetValuesLocal_IS(Mat A,PetscInt m,const PetscInt *rows, PetscInt n,const PetscInt *cols,const PetscScalar *values,InsertMode addv)
{
  PetscErrorCode ierr;
  Mat_IS         *is = (Mat_IS*)A->data;

  PetscFunctionBegin;
  if (is->usesetlocal) {
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
  ierr = MatSetValuesBlocked(is->A,m,rows,n,cols,values,addv);CHKERRQ(ierr);
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
      ierr = MatZeroRowsColumns(is->A,n,rows,diag,0,0);CHKERRQ(ierr);
    } else {
      ierr = MatZeroRows(is->A,n,rows,diag,0,0);CHKERRQ(ierr);
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
#if defined(PETSC_USE_DEBUG)
  if (columns || diag != 0. || (x && b)) {
    PetscBool cong;
    ierr = PetscLayoutCompare(A->rmap,A->cmap,&cong);CHKERRQ(ierr);
    if (!cong && columns) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Columns can be zeroed if and only if A->rmap and A->cmap are congruent for MATIS");
    if (!cong && diag != 0.) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Nonzero diagonal value supported if and only if A->rmap and A->cmap are congruent for MATIS");
    if (!cong && x && b) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"A->rmap and A->cmap need to be congruent");
  }
#endif
  /* get locally owned rows */
  ierr = PetscLayoutMapLocal_Private(A->rmap,n,rows,&len,&lrows,NULL);CHKERRQ(ierr);
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
  ierr = MatISSetUpSF(A);CHKERRQ(ierr);
  ierr = MatGetSize(matis->A,&nl,NULL);CHKERRQ(ierr);
  ierr = PetscMemzero(matis->sf_leafdata,nl*sizeof(PetscInt));CHKERRQ(ierr);
  ierr = PetscMemzero(matis->sf_rootdata,A->rmap->n*sizeof(PetscInt));CHKERRQ(ierr);
  for (i=0;i<len;i++) matis->sf_rootdata[lrows[i]] = 1;
  ierr = PetscFree(lrows);CHKERRQ(ierr);
  ierr = PetscSFBcastBegin(matis->sf,MPIU_INT,matis->sf_rootdata,matis->sf_leafdata);CHKERRQ(ierr);
  ierr = PetscSFBcastEnd(matis->sf,MPIU_INT,matis->sf_rootdata,matis->sf_leafdata);CHKERRQ(ierr);
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
    ISLocalToGlobalMapping l2g;
    IS                     tis;
    const PetscScalar      *v;
    PetscInt               i,n,cf,*loce,*locf;
    PetscBool              sym;

    ierr = MatGetRowMaxAbs(is->A,is->y,NULL);CHKERRQ(ierr);
    ierr = MatIsSymmetric(is->A,PETSC_SMALL,&sym);CHKERRQ(ierr);
    if (!sym) SETERRQ(PetscObjectComm((PetscObject)is->A),PETSC_ERR_SUP,"Not yet implemented for unsymmetric case");
    ierr = VecGetLocalSize(is->y,&n);CHKERRQ(ierr);
    ierr = PetscMalloc1(n,&loce);CHKERRQ(ierr);
    ierr = PetscMalloc1(n,&locf);CHKERRQ(ierr);
    ierr = VecGetArrayRead(is->y,&v);CHKERRQ(ierr);
    for (i=0,cf=0;i<n;i++) {
      if (v[i] == 0.0) {
        loce[i] = -1;
      } else {
        loce[i]    = cf;
        locf[cf++] = i;
      }
    }
    ierr = VecRestoreArrayRead(is->y,&v);CHKERRQ(ierr);
    /* extract valid submatrix */
    ierr = ISCreateGeneral(PetscObjectComm((PetscObject)is->A),cf,locf,PETSC_USE_POINTER,&tis);CHKERRQ(ierr);
    ierr = MatCreateSubMatrix(is->A,tis,tis,MAT_INITIAL_MATRIX,&newlA);CHKERRQ(ierr);
    ierr = ISDestroy(&tis);CHKERRQ(ierr);
    /* attach local l2g map for successive calls of MatSetValues */
    ierr = ISLocalToGlobalMappingCreate(PetscObjectComm((PetscObject)is->A),1,n,loce,PETSC_OWN_POINTER,&l2g);CHKERRQ(ierr);
    ierr = MatSetLocalToGlobalMapping(newlA,l2g,l2g);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingDestroy(&l2g);CHKERRQ(ierr);
    /* flag MatSetValues */
    is->usesetlocal = PETSC_TRUE;
    /* attach new global l2g map */
    ierr = ISLocalToGlobalMappingApply(A->rmap->mapping,cf,locf,locf);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingCreate(PetscObjectComm((PetscObject)A),1,cf,locf,PETSC_OWN_POINTER,&l2g);CHKERRQ(ierr);
    ierr = MatSetLocalToGlobalMapping(A,l2g,l2g);CHKERRQ(ierr);
    ierr = MatISSetLocalMat(A,newlA);CHKERRQ(ierr);
    ierr = MatDestroy(&newlA);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingDestroy(&l2g);CHKERRQ(ierr);
  }
  is->locempty = PETSC_FALSE;
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

static PetscErrorCode MatISSetLocalMat_IS(Mat mat,Mat local)
{
  Mat_IS         *is = (Mat_IS*)mat->data;
  PetscInt       nrows,ncols,orows,ocols;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (is->A) {
    ierr = MatGetSize(is->A,&orows,&ocols);CHKERRQ(ierr);
    ierr = MatGetSize(local,&nrows,&ncols);CHKERRQ(ierr);
    if (orows != nrows || ocols != ncols) SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Local MATIS matrix should be of size %Dx%D (you passed a %Dx%D matrix)",orows,ocols,nrows,ncols);
  }
  ierr  = PetscObjectReference((PetscObject)local);CHKERRQ(ierr);
  ierr  = MatDestroy(&is->A);CHKERRQ(ierr);
  is->A = local;
  PetscFunctionReturn(0);
}

/*@
    MatISSetLocalMat - Replace the local matrix stored inside a MATIS object.

  Input Parameter:
.  mat - the matrix
.  local - the local matrix

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
#if defined(PETSC_USE_DEBUG)
  PetscBool      ismatis;
#endif
  PetscErrorCode ierr;

  PetscFunctionBegin;
#if defined(PETSC_USE_DEBUG)
  ierr = PetscObjectTypeCompare((PetscObject)X,MATIS,&ismatis);CHKERRQ(ierr);
  if (!ismatis) SETERRQ(PetscObjectComm((PetscObject)Y),PETSC_ERR_SUP,"Cannot call MatAXPY(Y,a,X,str) with X not of type MATIS");
#endif
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
#if defined(PETSC_USE_DEBUG)
  for (i=0;i<nrl;i++) if (rl[i]>=nrg) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Local row index %D -> %D greater than maximum possible %D",i,rl[i],nrg);
#endif
  ierr = PetscMalloc1(nrg,&idxs);CHKERRQ(ierr);
  /* map from [0,nrl) to row */
  for (i=0;i<nrl;i++) idxs[i] = rl[i];
#if defined(PETSC_USE_DEBUG)
  for (i=nrl;i<nrg;i++) idxs[i] = nrg;
#else
  for (i=nrl;i<nrg;i++) idxs[i] = -1;
#endif
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
#if defined(PETSC_USE_DEBUG)
    for (i=0;i<ncl;i++) if (cl[i]>=ncg) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Local column index %D -> %D greater than maximum possible %D",i,cl[i],ncg);
#endif
    ierr = PetscMalloc1(ncg,&idxs);CHKERRQ(ierr);
    /* map from [0,ncl) to col */
    for (i=0;i<ncl;i++) idxs[i] = cl[i];
#if defined(PETSC_USE_DEBUG)
    for (i=ncl;i<ncg;i++) idxs[i] = ncg;
#else
    for (i=ncl;i<ncg;i++) idxs[i] = -1;
#endif
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"MATIS options");CHKERRQ(ierr);
  ierr = PetscObjectOptionsBegin((PetscObject)A);
  ierr = PetscOptionsBool("-matis_fixempty","Fix local matrices in case of empty local rows/columns",NULL,a->locempty,&a->locempty,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
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

   Notes: See MATIS for more details.
          m and n are NOT related to the size of the map; they represent the size of the local parts of the vectors
          used in MatMult operations. The sizes of rmap and cmap define the size of the local matrices.
          If either rmap or cmap are NULL, then the matrix is assumed to be square.

.seealso: MATIS, MatSetLocalToGlobalMapping()
@*/
PetscErrorCode  MatCreateIS(MPI_Comm comm,PetscInt bs,PetscInt m,PetscInt n,PetscInt M,PetscInt N,ISLocalToGlobalMapping rmap,ISLocalToGlobalMapping cmap,Mat *A)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!rmap && !cmap) SETERRQ(comm,PETSC_ERR_USER,"You need to provide at least one of the mappings");
  ierr = MatCreate(comm,A);CHKERRQ(ierr);
  ierr = MatSetSizes(*A,m,n,M,N);CHKERRQ(ierr);
  if (bs > 0) {
    ierr = MatSetBlockSize(*A,bs);CHKERRQ(ierr);
  }
  ierr = MatSetType(*A,MATIS);CHKERRQ(ierr);
  if (rmap && cmap) {
    ierr = MatSetLocalToGlobalMapping(*A,rmap,cmap);CHKERRQ(ierr);
  } else if (!rmap) {
    ierr = MatSetLocalToGlobalMapping(*A,cmap,cmap);CHKERRQ(ierr);
  } else {
    ierr = MatSetLocalToGlobalMapping(*A,rmap,rmap);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*MC
   MATIS - MATIS = "is" - A matrix type to be used for using the non-overlapping domain decomposition methods (e.g. PCBDDC or KSPFETIDP).
   This stores the matrices in globally unassembled form. Each processor
   assembles only its local Neumann problem and the parallel matrix vector
   product is handled "implicitly".

   Operations Provided:
+  MatMult()
.  MatMultAdd()
.  MatMultTranspose()
.  MatMultTransposeAdd()
.  MatZeroEntries()
.  MatSetOption()
.  MatZeroRows()
.  MatSetValues()
.  MatSetValuesBlocked()
.  MatSetValuesLocal()
.  MatSetValuesBlockedLocal()
.  MatScale()
.  MatGetDiagonal()
.  MatMissingDiagonal()
.  MatDuplicate()
.  MatCopy()
.  MatAXPY()
.  MatCreateSubMatrix()
.  MatGetLocalSubMatrix()
.  MatTranspose()
-  MatSetLocalToGlobalMapping()

   Options Database Keys:
. -mat_type is - sets the matrix type to "is" during a call to MatSetFromOptions()

   Notes: Options prefix for the inner matrix are given by -is_mat_xxx

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
  A->data = (void*)b;

  /* matrix ops */
  ierr    = PetscMemzero(A->ops,sizeof(struct _MatOps));CHKERRQ(ierr);
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

  /* special MATIS functions */
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatISGetLocalMat_C",MatISGetLocalMat_IS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatISRestoreLocalMat_C",MatISRestoreLocalMat_IS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatISSetLocalMat_C",MatISSetLocalMat_IS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatISGetMPIXAIJ_C",MatISGetMPIXAIJ_IS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatISSetPreallocation_C",MatISSetPreallocation_IS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatISSetUpSF_C",MatISSetUpSF_IS);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)A,MATIS);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
