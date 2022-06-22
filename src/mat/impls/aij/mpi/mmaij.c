
/*
   Support for the parallel AIJ matrix vector multiply
*/
#include <../src/mat/impls/aij/mpi/mpiaij.h>
#include <petsc/private/vecimpl.h>
#include <petsc/private/isimpl.h>    /* needed because accesses data structure of ISLocalToGlobalMapping directly */

PetscErrorCode MatSetUpMultiply_MPIAIJ(Mat mat)
{
  Mat_MPIAIJ         *aij = (Mat_MPIAIJ*)mat->data;
  Mat_SeqAIJ         *B   = (Mat_SeqAIJ*)(aij->B->data);
  PetscInt           i,j,*aj = B->j,*garray;
  PetscInt           ec = 0; /* Number of nonzero external columns */
  IS                 from,to;
  Vec                gvec;
#if defined(PETSC_USE_CTABLE)
  PetscTable         gid1_lid1;
  PetscTablePosition tpos;
  PetscInt           gid,lid;
#else
  PetscInt           N = mat->cmap->N,*indices;
#endif

  PetscFunctionBegin;
  if (!aij->garray) {
    PetscCheck(aij->B,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Missing B mat");
#if defined(PETSC_USE_CTABLE)
    /* use a table */
    PetscCall(PetscTableCreate(aij->B->rmap->n,mat->cmap->N+1,&gid1_lid1));
    for (i=0; i<aij->B->rmap->n; i++) {
      for (j=0; j<B->ilen[i]; j++) {
        PetscInt data,gid1 = aj[B->i[i] + j] + 1;
        PetscCall(PetscTableFind(gid1_lid1,gid1,&data));
        if (!data) {
          /* one based table */
          PetscCall(PetscTableAdd(gid1_lid1,gid1,++ec,INSERT_VALUES));
        }
      }
    }
    /* form array of columns we need */
    PetscCall(PetscMalloc1(ec,&garray));
    PetscCall(PetscTableGetHeadPosition(gid1_lid1,&tpos));
    while (tpos) {
      PetscCall(PetscTableGetNext(gid1_lid1,&tpos,&gid,&lid));
      gid--;
      lid--;
      garray[lid] = gid;
    }
    PetscCall(PetscSortInt(ec,garray)); /* sort, and rebuild */
    PetscCall(PetscTableRemoveAll(gid1_lid1));
    for (i=0; i<ec; i++) {
      PetscCall(PetscTableAdd(gid1_lid1,garray[i]+1,i+1,INSERT_VALUES));
    }
    /* compact out the extra columns in B */
    for (i=0; i<aij->B->rmap->n; i++) {
      for (j=0; j<B->ilen[i]; j++) {
        PetscInt gid1 = aj[B->i[i] + j] + 1;
        PetscCall(PetscTableFind(gid1_lid1,gid1,&lid));
        lid--;
        aj[B->i[i] + j] = lid;
      }
    }
    PetscCall(PetscLayoutDestroy(&aij->B->cmap));
    PetscCall(PetscLayoutCreateFromSizes(PetscObjectComm((PetscObject)aij->B),ec,ec,1,&aij->B->cmap));
    PetscCall(PetscTableDestroy(&gid1_lid1));
#else
    /* Make an array as long as the number of columns */
    /* mark those columns that are in aij->B */
    PetscCall(PetscCalloc1(N,&indices));
    for (i=0; i<aij->B->rmap->n; i++) {
      for (j=0; j<B->ilen[i]; j++) {
        if (!indices[aj[B->i[i] + j]]) ec++;
        indices[aj[B->i[i] + j]] = 1;
      }
    }

    /* form array of columns we need */
    PetscCall(PetscMalloc1(ec,&garray));
    ec   = 0;
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
    PetscCall(PetscLayoutDestroy(&aij->B->cmap));
    PetscCall(PetscLayoutCreateFromSizes(PetscObjectComm((PetscObject)aij->B),ec,ec,1,&aij->B->cmap));
    PetscCall(PetscFree(indices));
#endif
  } else {
    garray = aij->garray;
  }

  if (!aij->lvec) {
    PetscCheck(aij->B,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Missing B mat");
    PetscCall(MatCreateVecs(aij->B,&aij->lvec,NULL));
  }
  PetscCall(VecGetSize(aij->lvec,&ec));

  /* create two temporary Index sets for build scatter gather */
  PetscCall(ISCreateGeneral(PETSC_COMM_SELF,ec,garray,PETSC_COPY_VALUES,&from));
  PetscCall(ISCreateStride(PETSC_COMM_SELF,ec,0,1,&to));

  /* create temporary global vector to generate scatter context */
  /* This does not allocate the array's memory so is efficient */
  PetscCall(VecCreateMPIWithArray(PetscObjectComm((PetscObject)mat),1,mat->cmap->n,mat->cmap->N,NULL,&gvec));

  /* generate the scatter context */
  PetscCall(VecScatterDestroy(&aij->Mvctx));
  PetscCall(VecScatterCreate(gvec,from,aij->lvec,to,&aij->Mvctx));
  PetscCall(VecScatterViewFromOptions(aij->Mvctx,(PetscObject)mat,"-matmult_vecscatter_view"));
  PetscCall(PetscLogObjectParent((PetscObject)mat,(PetscObject)aij->Mvctx));
  PetscCall(PetscLogObjectParent((PetscObject)mat,(PetscObject)aij->lvec));
  PetscCall(PetscLogObjectMemory((PetscObject)mat,ec*sizeof(PetscInt)));
  aij->garray = garray;

  PetscCall(PetscLogObjectParent((PetscObject)mat,(PetscObject)from));
  PetscCall(PetscLogObjectParent((PetscObject)mat,(PetscObject)to));

  PetscCall(ISDestroy(&from));
  PetscCall(ISDestroy(&to));
  PetscCall(VecDestroy(&gvec));
  PetscFunctionReturn(0);
}

/* Disassemble the off-diag portion of the MPIAIJXxx matrix.
   In other words, change the B from reduced format using local col ids
   to expanded format using global col ids, which will make it easier to
   insert new nonzeros (with global col ids) into the matrix.
   The off-diag B determines communication in the matrix vector multiply.
*/
PetscErrorCode MatDisAssemble_MPIAIJ(Mat A)
{
  Mat_MPIAIJ        *aij  = (Mat_MPIAIJ*)A->data;
  Mat               B     = aij->B,Bnew;
  Mat_SeqAIJ        *Baij = (Mat_SeqAIJ*)B->data;
  PetscInt          i,j,m = B->rmap->n,n = A->cmap->N,col,ct = 0,*garray = aij->garray,*nz,ec;
  PetscScalar       v;
  const PetscScalar *ba;

  PetscFunctionBegin;
  /* free stuff related to matrix-vec multiply */
  PetscCall(VecGetSize(aij->lvec,&ec)); /* needed for PetscLogObjectMemory below */
  PetscCall(VecDestroy(&aij->lvec));
  if (aij->colmap) {
#if defined(PETSC_USE_CTABLE)
    PetscCall(PetscTableDestroy(&aij->colmap));
#else
    PetscCall(PetscFree(aij->colmap));
    PetscCall(PetscLogObjectMemory((PetscObject)A,-aij->B->cmap->n*sizeof(PetscInt)));
#endif
  }

  /* make sure that B is assembled so we can access its values */
  PetscCall(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));

  /* invent new B and copy stuff over */
  PetscCall(PetscMalloc1(m+1,&nz));
  for (i=0; i<m; i++) {
    nz[i] = Baij->i[i+1] - Baij->i[i];
  }
  PetscCall(MatCreate(PETSC_COMM_SELF,&Bnew));
  PetscCall(MatSetSizes(Bnew,m,n,m,n)); /* Bnew now uses A->cmap->N as its col size */
  PetscCall(MatSetBlockSizesFromMats(Bnew,A,A));
  PetscCall(MatSetType(Bnew,((PetscObject)B)->type_name));
  PetscCall(MatSeqAIJSetPreallocation(Bnew,0,nz));

  if (Baij->nonew >= 0) { /* Inherit insertion error options (if positive). */
    ((Mat_SeqAIJ*)Bnew->data)->nonew = Baij->nonew;
  }

  /*
   Ensure that B's nonzerostate is monotonically increasing.
   Or should this follow the MatSetValues() loop to preserve B's nonzerstate across a MatDisAssemble() call?
   */
  Bnew->nonzerostate = B->nonzerostate;

  PetscCall(PetscFree(nz));
  PetscCall(MatSeqAIJGetArrayRead(B,&ba));
  for (i=0; i<m; i++) {
    for (j=Baij->i[i]; j<Baij->i[i+1]; j++) {
      col  = garray[Baij->j[ct]];
      v    = ba[ct++];
      PetscCall(MatSetValues(Bnew,1,&i,1,&col,&v,B->insertmode));
    }
  }
  PetscCall(MatSeqAIJRestoreArrayRead(B,&ba));

  PetscCall(PetscFree(aij->garray));
  PetscCall(PetscLogObjectMemory((PetscObject)A,-ec*sizeof(PetscInt)));
  PetscCall(MatDestroy(&B));
  PetscCall(PetscLogObjectParent((PetscObject)A,(PetscObject)Bnew));

  aij->B           = Bnew;
  A->was_assembled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*      ugly stuff added for Glenn someday we should fix this up */

static PetscInt *auglyrmapd = NULL,*auglyrmapo = NULL; /* mapping from the local ordering to the "diagonal" and "off-diagonal" parts of the local matrix */
static Vec auglydd          = NULL,auglyoo     = NULL; /* work vectors used to scale the two parts of the local matrix */

PetscErrorCode MatMPIAIJDiagonalScaleLocalSetUp(Mat inA,Vec scale)
{
  Mat_MPIAIJ     *ina = (Mat_MPIAIJ*) inA->data; /*access private part of matrix */
  PetscInt       i,n,nt,cstart,cend,no,*garray = ina->garray,*lindices;
  PetscInt       *r_rmapd,*r_rmapo;

  PetscFunctionBegin;
  PetscCall(MatGetOwnershipRange(inA,&cstart,&cend));
  PetscCall(MatGetSize(ina->A,NULL,&n));
  PetscCall(PetscCalloc1(inA->rmap->mapping->n+1,&r_rmapd));
  nt   = 0;
  for (i=0; i<inA->rmap->mapping->n; i++) {
    if (inA->rmap->mapping->indices[i] >= cstart && inA->rmap->mapping->indices[i] < cend) {
      nt++;
      r_rmapd[i] = inA->rmap->mapping->indices[i] + 1;
    }
  }
  PetscCheck(nt == n,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Hmm nt %" PetscInt_FMT " n %" PetscInt_FMT,nt,n);
  PetscCall(PetscMalloc1(n+1,&auglyrmapd));
  for (i=0; i<inA->rmap->mapping->n; i++) {
    if (r_rmapd[i]) {
      auglyrmapd[(r_rmapd[i]-1)-cstart] = i;
    }
  }
  PetscCall(PetscFree(r_rmapd));
  PetscCall(VecCreateSeq(PETSC_COMM_SELF,n,&auglydd));

  PetscCall(PetscCalloc1(inA->cmap->N+1,&lindices));
  for (i=0; i<ina->B->cmap->n; i++) {
    lindices[garray[i]] = i+1;
  }
  no   = inA->rmap->mapping->n - nt;
  PetscCall(PetscCalloc1(inA->rmap->mapping->n+1,&r_rmapo));
  nt   = 0;
  for (i=0; i<inA->rmap->mapping->n; i++) {
    if (lindices[inA->rmap->mapping->indices[i]]) {
      nt++;
      r_rmapo[i] = lindices[inA->rmap->mapping->indices[i]];
    }
  }
  PetscCheck(nt <= no,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Hmm nt %" PetscInt_FMT " no %" PetscInt_FMT,nt,n);
  PetscCall(PetscFree(lindices));
  PetscCall(PetscMalloc1(nt+1,&auglyrmapo));
  for (i=0; i<inA->rmap->mapping->n; i++) {
    if (r_rmapo[i]) {
      auglyrmapo[(r_rmapo[i]-1)] = i;
    }
  }
  PetscCall(PetscFree(r_rmapo));
  PetscCall(VecCreateSeq(PETSC_COMM_SELF,nt,&auglyoo));
  PetscFunctionReturn(0);
}

PetscErrorCode MatMPIAIJDiagonalScaleLocal(Mat A,Vec scale)
{
  /* This routine should really be abandoned as it duplicates MatDiagonalScaleLocal */

  PetscFunctionBegin;
  PetscTryMethod(A,"MatDiagonalScaleLocal_C",(Mat,Vec),(A,scale));
  PetscFunctionReturn(0);
}

PetscErrorCode  MatDiagonalScaleLocal_MPIAIJ(Mat A,Vec scale)
{
  Mat_MPIAIJ        *a = (Mat_MPIAIJ*) A->data; /*access private part of matrix */
  PetscInt          n,i;
  PetscScalar       *d,*o;
  const PetscScalar *s;

  PetscFunctionBegin;
  if (!auglyrmapd) {
    PetscCall(MatMPIAIJDiagonalScaleLocalSetUp(A,scale));
  }

  PetscCall(VecGetArrayRead(scale,&s));

  PetscCall(VecGetLocalSize(auglydd,&n));
  PetscCall(VecGetArray(auglydd,&d));
  for (i=0; i<n; i++) {
    d[i] = s[auglyrmapd[i]]; /* copy "diagonal" (true local) portion of scale into dd vector */
  }
  PetscCall(VecRestoreArray(auglydd,&d));
  /* column scale "diagonal" portion of local matrix */
  PetscCall(MatDiagonalScale(a->A,NULL,auglydd));

  PetscCall(VecGetLocalSize(auglyoo,&n));
  PetscCall(VecGetArray(auglyoo,&o));
  for (i=0; i<n; i++) {
    o[i] = s[auglyrmapo[i]]; /* copy "off-diagonal" portion of scale into oo vector */
  }
  PetscCall(VecRestoreArrayRead(scale,&s));
  PetscCall(VecRestoreArray(auglyoo,&o));
  /* column scale "off-diagonal" portion of local matrix */
  PetscCall(MatDiagonalScale(a->B,NULL,auglyoo));
  PetscFunctionReturn(0);
}
