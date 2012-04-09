
/*
   Support for the parallel AIJ matrix vector multiply
*/
#include <../src/mat/impls/aij/mpi/mpiaij.h>

#undef __FUNCT__  
#define __FUNCT__ "MatSetUpMultiply_MPIAIJ"
PetscErrorCode MatSetUpMultiply_MPIAIJ(Mat mat)
{
  Mat_MPIAIJ         *aij = (Mat_MPIAIJ*)mat->data;
  Mat_SeqAIJ         *B = (Mat_SeqAIJ*)(aij->B->data);  
  PetscErrorCode     ierr;
  PetscInt           i,j,*aj = B->j,ec = 0,*garray;
  IS                 from,to;
  Vec                gvec;
  PetscBool          useblockis;
#if defined (PETSC_USE_CTABLE)
  PetscTable         gid1_lid1;
  PetscTablePosition tpos;
  PetscInt           gid,lid; 
#else
  PetscInt           N = mat->cmap->N,*indices;
#endif

  PetscFunctionBegin;

#if defined (PETSC_USE_CTABLE)
  /* use a table - Mark Adams */
  ierr = PetscTableCreate(aij->B->rmap->n,mat->cmap->N+1,&gid1_lid1);CHKERRQ(ierr);
  for (i=0; i<aij->B->rmap->n; i++) {
    for (j=0; j<B->ilen[i]; j++) {
      PetscInt data,gid1 = aj[B->i[i] + j] + 1;
      ierr = PetscTableFind(gid1_lid1,gid1,&data);CHKERRQ(ierr);
      if (!data) {
        /* one based table */ 
        ierr = PetscTableAdd(gid1_lid1,gid1,++ec,INSERT_VALUES);CHKERRQ(ierr); 
      }
    }
  }
  /* form array of columns we need */
  ierr = PetscMalloc((ec+1)*sizeof(PetscInt),&garray);CHKERRQ(ierr);
  ierr = PetscTableGetHeadPosition(gid1_lid1,&tpos);CHKERRQ(ierr); 
  while (tpos) {  
    ierr = PetscTableGetNext(gid1_lid1,&tpos,&gid,&lid);CHKERRQ(ierr); 
    gid--;
    lid--;
    garray[lid] = gid; 
  }
  ierr = PetscSortInt(ec,garray);CHKERRQ(ierr); /* sort, and rebuild */
  ierr = PetscTableRemoveAll(gid1_lid1);CHKERRQ(ierr);
  for (i=0; i<ec; i++) {
    ierr = PetscTableAdd(gid1_lid1,garray[i]+1,i+1,INSERT_VALUES);CHKERRQ(ierr); 
  }
  /* compact out the extra columns in B */
  for (i=0; i<aij->B->rmap->n; i++) {
    for (j=0; j<B->ilen[i]; j++) {
      PetscInt gid1 = aj[B->i[i] + j] + 1;
      ierr = PetscTableFind(gid1_lid1,gid1,&lid);CHKERRQ(ierr);
      lid --;
      aj[B->i[i] + j]  = lid;
    }
  }
  aij->B->cmap->n = aij->B->cmap->N = ec;
  ierr = PetscLayoutSetUp((aij->B->cmap));CHKERRQ(ierr);
  ierr = PetscTableDestroy(&gid1_lid1);CHKERRQ(ierr);
  /* Mark Adams */
#else
  /* Make an array as long as the number of columns */
  /* mark those columns that are in aij->B */
  ierr = PetscMalloc((N+1)*sizeof(PetscInt),&indices);CHKERRQ(ierr);
  ierr = PetscMemzero(indices,N*sizeof(PetscInt));CHKERRQ(ierr);
  for (i=0; i<aij->B->rmap->n; i++) {
    for (j=0; j<B->ilen[i]; j++) {
      if (!indices[aj[B->i[i] + j] ]) ec++; 
      indices[aj[B->i[i] + j] ] = 1;
    }
  }

  /* form array of columns we need */
  ierr = PetscMalloc((ec+1)*sizeof(PetscInt),&garray);CHKERRQ(ierr);
  ec = 0;
  for (i=0; i<N; i++) {
    if (indices[i]) garray[ec++] = i;
  }

  /* make indices now point into garray */
  for (i=0; i<ec; i++) {
    indices[garray[i]] = i;
  }

  /* compact out the extra columns in B */
  for (i=0; i<aij->B->rmap->n; i++) {
    for (j=0; j<B->ilen[i]; j++) {
      aj[B->i[i] + j] = indices[aj[B->i[i] + j]];
    }
  }
  aij->B->cmap->n = aij->B->cmap->N = ec;
  ierr = PetscLayoutSetUp((aij->B->cmap));CHKERRQ(ierr);
  ierr = PetscFree(indices);CHKERRQ(ierr);
#endif  
  /* create local vector that is used to scatter into */
  ierr = VecCreateSeq(PETSC_COMM_SELF,ec,&aij->lvec);CHKERRQ(ierr);

  /* create two temporary Index sets for build scatter gather */
  /*  check for the special case where blocks are communicated for faster VecScatterXXX */
  useblockis = PETSC_TRUE;
  if (mat->rmap->bs > 1) {
    PetscInt bs = mat->rmap->bs,ibs,ga;
    if (!(ec % bs)) {
      for (i=0; i<ec/bs; i++) {
        if ((ga = garray[ibs = i*bs]) % bs) {
          useblockis = PETSC_FALSE;
          break;
        }
        for (j=1; j<bs; j++) {
          if (garray[ibs+j] != ga+j) {
            useblockis = PETSC_FALSE;
            break;
          }
        }
        if (!useblockis) break;
      }
    }
  }
  if (useblockis) {
    PetscInt *ga,bs = mat->rmap->bs,iec = ec/bs;
    ierr = PetscInfo(mat,"Using block index set to define scatter\n");
    ierr = PetscMalloc((ec/mat->rmap->bs)*sizeof(PetscInt),&ga);CHKERRQ(ierr);
    for (i=0; i<iec; i++) ga[i] = garray[i*bs]/bs;
    ierr = ISCreateBlock(((PetscObject)mat)->comm,bs,iec,ga,PETSC_OWN_POINTER,&from);CHKERRQ(ierr);
  } else {
    ierr = ISCreateGeneral(((PetscObject)mat)->comm,ec,garray,PETSC_COPY_VALUES,&from);CHKERRQ(ierr);
  }
  ierr = ISCreateStride(PETSC_COMM_SELF,ec,0,1,&to);CHKERRQ(ierr);

  /* create temporary global vector to generate scatter context */
  /* This does not allocate the array's memory so is efficient */
  ierr = VecCreateMPIWithArray(((PetscObject)mat)->comm,1,mat->cmap->n,mat->cmap->N,PETSC_NULL,&gvec);CHKERRQ(ierr);

  /* generate the scatter context */
  ierr = VecScatterCreate(gvec,from,aij->lvec,to,&aij->Mvctx);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(mat,aij->Mvctx);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(mat,aij->lvec);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(mat,from);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(mat,to);CHKERRQ(ierr);
  aij->garray = garray;
  ierr = PetscLogObjectMemory(mat,(ec+1)*sizeof(PetscInt));CHKERRQ(ierr);
  ierr = ISDestroy(&from);CHKERRQ(ierr);
  ierr = ISDestroy(&to);CHKERRQ(ierr);
  ierr = VecDestroy(&gvec);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "MatDisAssemble_MPIAIJ"
/*
     Takes the local part of an already assembled MPIAIJ matrix
   and disassembles it. This is to allow new nonzeros into the matrix
   that require more communication in the matrix vector multiply. 
   Thus certain data-structures must be rebuilt.

   Kind of slow! But that's what application programmers get when 
   they are sloppy.
*/
PetscErrorCode MatDisAssemble_MPIAIJ(Mat A)
{
  Mat_MPIAIJ     *aij = (Mat_MPIAIJ*)A->data;
  Mat            B = aij->B,Bnew;
  Mat_SeqAIJ     *Baij = (Mat_SeqAIJ*)B->data;
  PetscErrorCode ierr;
  PetscInt       i,j,m = B->rmap->n,n = A->cmap->N,col,ct = 0,*garray = aij->garray,*nz,ec;
  PetscScalar    v;

  PetscFunctionBegin;
  /* free stuff related to matrix-vec multiply */
  ierr = VecGetSize(aij->lvec,&ec);CHKERRQ(ierr); /* needed for PetscLogObjectMemory below */
  ierr = VecDestroy(&aij->lvec);CHKERRQ(ierr); 
  ierr = VecScatterDestroy(&aij->Mvctx);CHKERRQ(ierr); 
  if (aij->colmap) {
#if defined (PETSC_USE_CTABLE)
    ierr = PetscTableDestroy(&aij->colmap);CHKERRQ(ierr);
#else
    ierr = PetscFree(aij->colmap);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory(A,-aij->B->cmap->n*sizeof(PetscInt));CHKERRQ(ierr);
#endif
  }

  /* make sure that B is assembled so we can access its values */
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* invent new B and copy stuff over */
  ierr = PetscMalloc((m+1)*sizeof(PetscInt),&nz);CHKERRQ(ierr);
  for (i=0; i<m; i++) {
    nz[i] = Baij->i[i+1] - Baij->i[i];
  }
  ierr = MatCreate(PETSC_COMM_SELF,&Bnew);CHKERRQ(ierr);
  ierr = MatSetSizes(Bnew,m,n,m,n);CHKERRQ(ierr);
  ierr = MatSetType(Bnew,((PetscObject)B)->type_name);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(Bnew,0,nz);CHKERRQ(ierr);
  ((Mat_SeqAIJ*)Bnew->data)->nonew = Baij->nonew; /* Inherit insertion error options. */
  ierr = PetscFree(nz);CHKERRQ(ierr);
  for (i=0; i<m; i++) {
    for (j=Baij->i[i]; j<Baij->i[i+1]; j++) {
      col  = garray[Baij->j[ct]];
      v    = Baij->a[ct++];
      ierr = MatSetValues(Bnew,1,&i,1,&col,&v,B->insertmode);CHKERRQ(ierr);
    }
  }
  ierr = PetscFree(aij->garray);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory(A,-ec*sizeof(PetscInt));CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(A,Bnew);CHKERRQ(ierr);
  aij->B = Bnew;
  A->was_assembled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*      ugly stuff added for Glenn someday we should fix this up */

static PetscInt *auglyrmapd = 0,*auglyrmapo = 0;  /* mapping from the local ordering to the "diagonal" and "off-diagonal"
                                      parts of the local matrix */
static Vec auglydd = 0,auglyoo = 0;   /* work vectors used to scale the two parts of the local matrix */


#undef __FUNCT__
#define __FUNCT__ "MatMPIAIJDiagonalScaleLocalSetUp"
PetscErrorCode MatMPIAIJDiagonalScaleLocalSetUp(Mat inA,Vec scale)
{
  Mat_MPIAIJ     *ina = (Mat_MPIAIJ*) inA->data; /*access private part of matrix */
  PetscErrorCode ierr;
  PetscInt       i,n,nt,cstart,cend,no,*garray = ina->garray,*lindices;
  PetscInt       *r_rmapd,*r_rmapo;
  
  PetscFunctionBegin;
  ierr = MatGetOwnershipRange(inA,&cstart,&cend);CHKERRQ(ierr);
  ierr = MatGetSize(ina->A,PETSC_NULL,&n);CHKERRQ(ierr);
  ierr = PetscMalloc((inA->rmap->mapping->n+1)*sizeof(PetscInt),&r_rmapd);CHKERRQ(ierr);
  ierr = PetscMemzero(r_rmapd,inA->rmap->mapping->n*sizeof(PetscInt));CHKERRQ(ierr);
  nt   = 0;
  for (i=0; i<inA->rmap->mapping->n; i++) {
    if (inA->rmap->mapping->indices[i] >= cstart && inA->rmap->mapping->indices[i] < cend) {
      nt++;
      r_rmapd[i] = inA->rmap->mapping->indices[i] + 1;
    }
  }
  if (nt != n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Hmm nt %D n %D",nt,n);
  ierr = PetscMalloc((n+1)*sizeof(PetscInt),&auglyrmapd);CHKERRQ(ierr);
  for (i=0; i<inA->rmap->mapping->n; i++) {
    if (r_rmapd[i]){
      auglyrmapd[(r_rmapd[i]-1)-cstart] = i;
    }
  }
  ierr = PetscFree(r_rmapd);CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF,n,&auglydd);CHKERRQ(ierr);

  ierr = PetscMalloc((inA->cmap->N+1)*sizeof(PetscInt),&lindices);CHKERRQ(ierr);
  ierr = PetscMemzero(lindices,inA->cmap->N*sizeof(PetscInt));CHKERRQ(ierr);
  for (i=0; i<ina->B->cmap->n; i++) {
    lindices[garray[i]] = i+1;
  }
  no   = inA->rmap->mapping->n - nt;
  ierr = PetscMalloc((inA->rmap->mapping->n+1)*sizeof(PetscInt),&r_rmapo);CHKERRQ(ierr);
  ierr = PetscMemzero(r_rmapo,inA->rmap->mapping->n*sizeof(PetscInt));CHKERRQ(ierr);
  nt   = 0;
  for (i=0; i<inA->rmap->mapping->n; i++) {
    if (lindices[inA->rmap->mapping->indices[i]]) {
      nt++;
      r_rmapo[i] = lindices[inA->rmap->mapping->indices[i]];
    }
  }
  if (nt > no) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Hmm nt %D no %D",nt,n);
  ierr = PetscFree(lindices);CHKERRQ(ierr);
  ierr = PetscMalloc((nt+1)*sizeof(PetscInt),&auglyrmapo);CHKERRQ(ierr);
  for (i=0; i<inA->rmap->mapping->n; i++) {
    if (r_rmapo[i]){
      auglyrmapo[(r_rmapo[i]-1)] = i;
    }
  }
  ierr = PetscFree(r_rmapo);CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF,nt,&auglyoo);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMPIAIJDiagonalScaleLocal"
PetscErrorCode MatMPIAIJDiagonalScaleLocal(Mat A,Vec scale)
{
  /* This routine should really be abandoned as it duplicates MatDiagonalScaleLocal */
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscTryMethod(A,"MatDiagonalScaleLocal_C",(Mat,Vec),(A,scale));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatDiagonalScaleLocal_MPIAIJ"
PetscErrorCode  MatDiagonalScaleLocal_MPIAIJ(Mat A,Vec scale)
{
  Mat_MPIAIJ     *a = (Mat_MPIAIJ*) A->data; /*access private part of matrix */
  PetscErrorCode ierr;
  PetscInt       n,i;
  PetscScalar    *d,*o,*s;
  
  PetscFunctionBegin;
  if (!auglyrmapd) {
    ierr = MatMPIAIJDiagonalScaleLocalSetUp(A,scale);CHKERRQ(ierr);
  }

  ierr = VecGetArray(scale,&s);CHKERRQ(ierr);
  
  ierr = VecGetLocalSize(auglydd,&n);CHKERRQ(ierr);
  ierr = VecGetArray(auglydd,&d);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    d[i] = s[auglyrmapd[i]]; /* copy "diagonal" (true local) portion of scale into dd vector */
  }
  ierr = VecRestoreArray(auglydd,&d);CHKERRQ(ierr);
  /* column scale "diagonal" portion of local matrix */
  ierr = MatDiagonalScale(a->A,PETSC_NULL,auglydd);CHKERRQ(ierr);

  ierr = VecGetLocalSize(auglyoo,&n);CHKERRQ(ierr);
  ierr = VecGetArray(auglyoo,&o);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    o[i] = s[auglyrmapo[i]]; /* copy "off-diagonal" portion of scale into oo vector */
  }
  ierr = VecRestoreArray(scale,&s);CHKERRQ(ierr);
  ierr = VecRestoreArray(auglyoo,&o);CHKERRQ(ierr);
  /* column scale "off-diagonal" portion of local matrix */
  ierr = MatDiagonalScale(a->B,PETSC_NULL,auglyoo);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
EXTERN_C_END


