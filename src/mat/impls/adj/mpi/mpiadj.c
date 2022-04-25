
/*
    Defines the basic matrix operations for the ADJ adjacency list matrix data-structure.
*/
#include <../src/mat/impls/adj/mpi/mpiadj.h>    /*I "petscmat.h" I*/
#include <petscsf.h>

/*
 * The interface should be easy to use for both MatCreateSubMatrix (parallel sub-matrix) and MatCreateSubMatrices (sequential sub-matrices)
 * */
static PetscErrorCode MatCreateSubMatrix_MPIAdj_data(Mat adj,IS irows, IS icols, PetscInt **sadj_xadj,PetscInt **sadj_adjncy,PetscInt **sadj_values)
{
  PetscInt           nlrows_is,icols_n,i,j,nroots,nleaves,rlocalindex,*ncols_send,*ncols_recv;
  PetscInt           nlrows_mat,*adjncy_recv,Ncols_recv,Ncols_send,*xadj_recv,*values_recv;
  PetscInt          *ncols_recv_offsets,loc,rnclos,*sadjncy,*sxadj,*svalues;
  const PetscInt    *irows_indices,*icols_indices,*xadj, *adjncy;
  PetscMPIInt        owner;
  Mat_MPIAdj        *a = (Mat_MPIAdj*)adj->data;
  PetscLayout        rmap;
  MPI_Comm           comm;
  PetscSF            sf;
  PetscSFNode       *iremote;
  PetscBool          done;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)adj,&comm));
  PetscCall(MatGetLayouts(adj,&rmap,NULL));
  PetscCall(ISGetLocalSize(irows,&nlrows_is));
  PetscCall(ISGetIndices(irows,&irows_indices));
  PetscCall(PetscMalloc1(nlrows_is,&iremote));
  /* construct sf graph*/
  nleaves = nlrows_is;
  for (i=0; i<nlrows_is; i++) {
    owner = -1;
    rlocalindex = -1;
    PetscCall(PetscLayoutFindOwnerIndex(rmap,irows_indices[i],&owner,&rlocalindex));
    iremote[i].rank  = owner;
    iremote[i].index = rlocalindex;
  }
  PetscCall(MatGetRowIJ(adj,0,PETSC_FALSE,PETSC_FALSE,&nlrows_mat,&xadj,&adjncy,&done));
  PetscCall(PetscCalloc4(nlrows_mat,&ncols_send,nlrows_is,&xadj_recv,nlrows_is+1,&ncols_recv_offsets,nlrows_is,&ncols_recv));
  nroots = nlrows_mat;
  for (i=0; i<nlrows_mat; i++) {
    ncols_send[i] = xadj[i+1]-xadj[i];
  }
  PetscCall(PetscSFCreate(comm,&sf));
  PetscCall(PetscSFSetGraph(sf,nroots,nleaves,NULL,PETSC_OWN_POINTER,iremote,PETSC_OWN_POINTER));
  PetscCall(PetscSFSetType(sf,PETSCSFBASIC));
  PetscCall(PetscSFSetFromOptions(sf));
  PetscCall(PetscSFBcastBegin(sf,MPIU_INT,ncols_send,ncols_recv,MPI_REPLACE));
  PetscCall(PetscSFBcastEnd(sf,MPIU_INT,ncols_send,ncols_recv,MPI_REPLACE));
  PetscCall(PetscSFBcastBegin(sf,MPIU_INT,xadj,xadj_recv,MPI_REPLACE));
  PetscCall(PetscSFBcastEnd(sf,MPIU_INT,xadj,xadj_recv,MPI_REPLACE));
  PetscCall(PetscSFDestroy(&sf));
  Ncols_recv =0;
  for (i=0; i<nlrows_is; i++) {
    Ncols_recv             += ncols_recv[i];
    ncols_recv_offsets[i+1] = ncols_recv[i]+ncols_recv_offsets[i];
  }
  Ncols_send = 0;
  for (i=0; i<nlrows_mat; i++) {
    Ncols_send += ncols_send[i];
  }
  PetscCall(PetscCalloc1(Ncols_recv,&iremote));
  PetscCall(PetscCalloc1(Ncols_recv,&adjncy_recv));
  nleaves = Ncols_recv;
  Ncols_recv = 0;
  for (i=0; i<nlrows_is; i++) {
    PetscCall(PetscLayoutFindOwner(rmap,irows_indices[i],&owner));
    for (j=0; j<ncols_recv[i]; j++) {
      iremote[Ncols_recv].rank    = owner;
      iremote[Ncols_recv++].index = xadj_recv[i]+j;
    }
  }
  PetscCall(ISRestoreIndices(irows,&irows_indices));
  /*if we need to deal with edge weights ???*/
  if (a->useedgeweights) PetscCall(PetscCalloc1(Ncols_recv,&values_recv));
  nroots = Ncols_send;
  PetscCall(PetscSFCreate(comm,&sf));
  PetscCall(PetscSFSetGraph(sf,nroots,nleaves,NULL,PETSC_OWN_POINTER,iremote,PETSC_OWN_POINTER));
  PetscCall(PetscSFSetType(sf,PETSCSFBASIC));
  PetscCall(PetscSFSetFromOptions(sf));
  PetscCall(PetscSFBcastBegin(sf,MPIU_INT,adjncy,adjncy_recv,MPI_REPLACE));
  PetscCall(PetscSFBcastEnd(sf,MPIU_INT,adjncy,adjncy_recv,MPI_REPLACE));
  if (a->useedgeweights) {
    PetscCall(PetscSFBcastBegin(sf,MPIU_INT,a->values,values_recv,MPI_REPLACE));
    PetscCall(PetscSFBcastEnd(sf,MPIU_INT,a->values,values_recv,MPI_REPLACE));
  }
  PetscCall(PetscSFDestroy(&sf));
  PetscCall(MatRestoreRowIJ(adj,0,PETSC_FALSE,PETSC_FALSE,&nlrows_mat,&xadj,&adjncy,&done));
  PetscCall(ISGetLocalSize(icols,&icols_n));
  PetscCall(ISGetIndices(icols,&icols_indices));
  rnclos = 0;
  for (i=0; i<nlrows_is; i++) {
    for (j=ncols_recv_offsets[i]; j<ncols_recv_offsets[i+1]; j++) {
      PetscCall(PetscFindInt(adjncy_recv[j], icols_n, icols_indices, &loc));
      if (loc<0) {
        adjncy_recv[j] = -1;
        if (a->useedgeweights) values_recv[j] = -1;
        ncols_recv[i]--;
      } else {
        rnclos++;
      }
    }
  }
  PetscCall(ISRestoreIndices(icols,&icols_indices));
  PetscCall(PetscCalloc1(rnclos,&sadjncy));
  if (a->useedgeweights) PetscCall(PetscCalloc1(rnclos,&svalues));
  PetscCall(PetscCalloc1(nlrows_is+1,&sxadj));
  rnclos = 0;
  for (i=0; i<nlrows_is; i++) {
    for (j=ncols_recv_offsets[i]; j<ncols_recv_offsets[i+1]; j++) {
      if (adjncy_recv[j]<0) continue;
      sadjncy[rnclos] = adjncy_recv[j];
      if (a->useedgeweights) svalues[rnclos] = values_recv[j];
      rnclos++;
    }
  }
  for (i=0; i<nlrows_is; i++) {
    sxadj[i+1] = sxadj[i]+ncols_recv[i];
  }
  if (sadj_xadj)  { *sadj_xadj = sxadj;} else    PetscCall(PetscFree(sxadj));
  if (sadj_adjncy) { *sadj_adjncy = sadjncy;} else PetscCall(PetscFree(sadjncy));
  if (sadj_values) {
    if (a->useedgeweights) *sadj_values = svalues; else *sadj_values=NULL;
  } else {
    if (a->useedgeweights) PetscCall(PetscFree(svalues));
  }
  PetscCall(PetscFree4(ncols_send,xadj_recv,ncols_recv_offsets,ncols_recv));
  PetscCall(PetscFree(adjncy_recv));
  if (a->useedgeweights) PetscCall(PetscFree(values_recv));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatCreateSubMatrices_MPIAdj_Private(Mat mat,PetscInt n,const IS irow[],const IS icol[],PetscBool subcomm,MatReuse scall,Mat *submat[])
{
  PetscInt           i,irow_n,icol_n,*sxadj,*sadjncy,*svalues;
  PetscInt          *indices,nindx,j,k,loc;
  PetscMPIInt        issame;
  const PetscInt    *irow_indices,*icol_indices;
  MPI_Comm           scomm_row,scomm_col,scomm_mat;

  PetscFunctionBegin;
  nindx = 0;
  /*
   * Estimate a maximum number for allocating memory
   */
  for (i=0; i<n; i++) {
    PetscCall(ISGetLocalSize(irow[i],&irow_n));
    PetscCall(ISGetLocalSize(icol[i],&icol_n));
    nindx = nindx>(irow_n+icol_n)? nindx:(irow_n+icol_n);
  }
  PetscCall(PetscMalloc1(nindx,&indices));
  /* construct a submat */
  for (i=0; i<n; i++) {
    if (subcomm) {
      PetscCall(PetscObjectGetComm((PetscObject)irow[i],&scomm_row));
      PetscCall(PetscObjectGetComm((PetscObject)icol[i],&scomm_col));
      PetscCallMPI(MPI_Comm_compare(scomm_row,scomm_col,&issame));
      PetscCheck(issame == MPI_IDENT,PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"row index set must have the same comm as the col index set");
      PetscCallMPI(MPI_Comm_compare(scomm_row,PETSC_COMM_SELF,&issame));
      PetscCheck(issame != MPI_IDENT,PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP," can not use PETSC_COMM_SELF as comm when extracting a parallel submatrix");
    } else {
      scomm_row = PETSC_COMM_SELF;
    }
    /*get sub-matrix data*/
    sxadj=NULL; sadjncy=NULL; svalues=NULL;
    PetscCall(MatCreateSubMatrix_MPIAdj_data(mat,irow[i],icol[i],&sxadj,&sadjncy,&svalues));
    PetscCall(ISGetLocalSize(irow[i],&irow_n));
    PetscCall(ISGetLocalSize(icol[i],&icol_n));
    PetscCall(ISGetIndices(irow[i],&irow_indices));
    PetscCall(PetscArraycpy(indices,irow_indices,irow_n));
    PetscCall(ISRestoreIndices(irow[i],&irow_indices));
    PetscCall(ISGetIndices(icol[i],&icol_indices));
    PetscCall(PetscArraycpy(indices+irow_n,icol_indices,icol_n));
    PetscCall(ISRestoreIndices(icol[i],&icol_indices));
    nindx = irow_n+icol_n;
    PetscCall(PetscSortRemoveDupsInt(&nindx,indices));
    /* renumber columns */
    for (j=0; j<irow_n; j++) {
      for (k=sxadj[j]; k<sxadj[j+1]; k++) {
        PetscCall(PetscFindInt(sadjncy[k],nindx,indices,&loc));
        PetscCheck(loc>=0,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"can not find col %" PetscInt_FMT,sadjncy[k]);
        sadjncy[k] = loc;
      }
    }
    if (scall==MAT_INITIAL_MATRIX) {
      PetscCall(MatCreateMPIAdj(scomm_row,irow_n,icol_n,sxadj,sadjncy,svalues,submat[i]));
    } else {
       Mat                sadj = *(submat[i]);
       Mat_MPIAdj         *sa  = (Mat_MPIAdj*)((sadj)->data);
       PetscCall(PetscObjectGetComm((PetscObject)sadj,&scomm_mat));
       PetscCallMPI(MPI_Comm_compare(scomm_row,scomm_mat,&issame));
       PetscCheck(issame == MPI_IDENT,PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"submatrix  must have the same comm as the col index set");
       PetscCall(PetscArraycpy(sa->i,sxadj,irow_n+1));
       PetscCall(PetscArraycpy(sa->j,sadjncy,sxadj[irow_n]));
       if (svalues) PetscCall(PetscArraycpy(sa->values,svalues,sxadj[irow_n]));
       PetscCall(PetscFree(sxadj));
       PetscCall(PetscFree(sadjncy));
       if (svalues) PetscCall(PetscFree(svalues));
    }
  }
  PetscCall(PetscFree(indices));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatCreateSubMatricesMPI_MPIAdj(Mat mat,PetscInt n, const IS irow[],const IS icol[],MatReuse scall,Mat *submat[])
{
  /*get sub-matrices across a sub communicator */
  PetscFunctionBegin;
  PetscCall(MatCreateSubMatrices_MPIAdj_Private(mat,n,irow,icol,PETSC_TRUE,scall,submat));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatCreateSubMatrices_MPIAdj(Mat mat,PetscInt n,const IS irow[],const IS icol[],MatReuse scall,Mat *submat[])
{
  PetscFunctionBegin;
  /*get sub-matrices based on PETSC_COMM_SELF */
  PetscCall(MatCreateSubMatrices_MPIAdj_Private(mat,n,irow,icol,PETSC_FALSE,scall,submat));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatView_MPIAdj_ASCII(Mat A,PetscViewer viewer)
{
  Mat_MPIAdj        *a = (Mat_MPIAdj*)A->data;
  PetscInt          i,j,m = A->rmap->n;
  const char        *name;
  PetscViewerFormat format;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetName((PetscObject)A,&name));
  PetscCall(PetscViewerGetFormat(viewer,&format));
  if (format == PETSC_VIEWER_ASCII_INFO) {
    PetscFunctionReturn(0);
  } else PetscCheck(format != PETSC_VIEWER_ASCII_MATLAB,PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"MATLAB format not supported");
  else {
    PetscCall(PetscViewerASCIIUseTabs(viewer,PETSC_FALSE));
    PetscCall(PetscViewerASCIIPushSynchronized(viewer));
    for (i=0; i<m; i++) {
      PetscCall(PetscViewerASCIISynchronizedPrintf(viewer,"row %" PetscInt_FMT ":",i+A->rmap->rstart));
      for (j=a->i[i]; j<a->i[i+1]; j++) {
        if (a->values) {
          PetscCall(PetscViewerASCIISynchronizedPrintf(viewer," (%" PetscInt_FMT ", %" PetscInt_FMT ") ",a->j[j], a->values[j]));
        } else {
          PetscCall(PetscViewerASCIISynchronizedPrintf(viewer," %" PetscInt_FMT " ",a->j[j]));
        }
      }
      PetscCall(PetscViewerASCIISynchronizedPrintf(viewer,"\n"));
    }
    PetscCall(PetscViewerASCIIUseTabs(viewer,PETSC_TRUE));
    PetscCall(PetscViewerFlush(viewer));
    PetscCall(PetscViewerASCIIPopSynchronized(viewer));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatView_MPIAdj(Mat A,PetscViewer viewer)
{
  PetscBool      iascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii));
  if (iascii) {
    PetscCall(MatView_MPIAdj_ASCII(A,viewer));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDestroy_MPIAdj(Mat mat)
{
  Mat_MPIAdj     *a = (Mat_MPIAdj*)mat->data;

  PetscFunctionBegin;
#if defined(PETSC_USE_LOG)
  PetscLogObjectState((PetscObject)mat,"Rows=%" PetscInt_FMT ", Cols=%" PetscInt_FMT ", NZ=%" PetscInt_FMT,mat->rmap->n,mat->cmap->n,a->nz);
#endif
  PetscCall(PetscFree(a->diag));
  if (a->freeaij) {
    if (a->freeaijwithfree) {
      if (a->i) free(a->i);
      if (a->j) free(a->j);
    } else {
      PetscCall(PetscFree(a->i));
      PetscCall(PetscFree(a->j));
      PetscCall(PetscFree(a->values));
    }
  }
  PetscCall(PetscFree(a->rowvalues));
  PetscCall(PetscFree(mat->data));
  PetscCall(PetscObjectChangeTypeName((PetscObject)mat,NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatMPIAdjSetPreallocation_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatMPIAdjCreateNonemptySubcommMat_C",NULL));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSetOption_MPIAdj(Mat A,MatOption op,PetscBool flg)
{
  Mat_MPIAdj     *a = (Mat_MPIAdj*)A->data;

  PetscFunctionBegin;
  switch (op) {
  case MAT_SYMMETRIC:
  case MAT_STRUCTURALLY_SYMMETRIC:
  case MAT_HERMITIAN:
    a->symmetric = flg;
    break;
  case MAT_SYMMETRY_ETERNAL:
    break;
  default:
    PetscCall(PetscInfo(A,"Option %s ignored\n",MatOptions[op]));
    break;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatGetRow_MPIAdj(Mat A,PetscInt row,PetscInt *nz,PetscInt **idx,PetscScalar **v)
{
  Mat_MPIAdj *a = (Mat_MPIAdj*)A->data;

  PetscFunctionBegin;
  row -= A->rmap->rstart;
  PetscCheckFalse(row < 0 || row >= A->rmap->n,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Row out of range");
  *nz = a->i[row+1] - a->i[row];
  if (v) {
    PetscInt j;
    if (a->rowvalues_alloc < *nz) {
      PetscCall(PetscFree(a->rowvalues));
      a->rowvalues_alloc = PetscMax(a->rowvalues_alloc*2, *nz);
      PetscCall(PetscMalloc1(a->rowvalues_alloc,&a->rowvalues));
    }
    for (j=0; j<*nz; j++) {
      a->rowvalues[j] = a->values ? a->values[a->i[row]+j]:1.0;
    }
    *v = (*nz) ? a->rowvalues : NULL;
  }
  if (idx) *idx = (*nz) ? a->j + a->i[row] : NULL;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatRestoreRow_MPIAdj(Mat A,PetscInt row,PetscInt *nz,PetscInt **idx,PetscScalar **v)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatEqual_MPIAdj(Mat A,Mat B,PetscBool * flg)
{
  Mat_MPIAdj     *a = (Mat_MPIAdj*)A->data,*b = (Mat_MPIAdj*)B->data;
  PetscBool      flag;

  PetscFunctionBegin;
  /* If the  matrix dimensions are not equal,or no of nonzeros */
  if ((A->rmap->n != B->rmap->n) ||(a->nz != b->nz)) {
    flag = PETSC_FALSE;
  }

  /* if the a->i are the same */
  PetscCall(PetscArraycmp(a->i,b->i,A->rmap->n+1,&flag));

  /* if a->j are the same */
  PetscCall(PetscMemcmp(a->j,b->j,(a->nz)*sizeof(PetscInt),&flag));

  PetscCall(MPIU_Allreduce(&flag,flg,1,MPIU_BOOL,MPI_LAND,PetscObjectComm((PetscObject)A)));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatGetRowIJ_MPIAdj(Mat A,PetscInt oshift,PetscBool symmetric,PetscBool blockcompressed,PetscInt *m,const PetscInt *inia[],const PetscInt *inja[],PetscBool  *done)
{
  PetscInt   i;
  Mat_MPIAdj *a   = (Mat_MPIAdj*)A->data;
  PetscInt   **ia = (PetscInt**)inia,**ja = (PetscInt**)inja;

  PetscFunctionBegin;
  *m    = A->rmap->n;
  *ia   = a->i;
  *ja   = a->j;
  *done = PETSC_TRUE;
  if (oshift) {
    for (i=0; i<(*ia)[*m]; i++) {
      (*ja)[i]++;
    }
    for (i=0; i<=(*m); i++) (*ia)[i]++;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatRestoreRowIJ_MPIAdj(Mat A,PetscInt oshift,PetscBool symmetric,PetscBool blockcompressed,PetscInt *m,const PetscInt *inia[],const PetscInt *inja[],PetscBool  *done)
{
  PetscInt   i;
  Mat_MPIAdj *a   = (Mat_MPIAdj*)A->data;
  PetscInt   **ia = (PetscInt**)inia,**ja = (PetscInt**)inja;

  PetscFunctionBegin;
  PetscCheckFalse(ia && a->i != *ia,PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONGSTATE,"ia passed back is not one obtained with MatGetRowIJ()");
  PetscCheckFalse(ja && a->j != *ja,PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONGSTATE,"ja passed back is not one obtained with MatGetRowIJ()");
  if (oshift) {
    PetscCheck(ia,PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONGSTATE,"If oshift then you must passed in inia[] argument");
    PetscCheck(ja,PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONGSTATE,"If oshift then you must passed in inja[] argument");
    for (i=0; i<=(*m); i++) (*ia)[i]--;
    for (i=0; i<(*ia)[*m]; i++) {
      (*ja)[i]--;
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode  MatConvertFrom_MPIAdj(Mat A,MatType type,MatReuse reuse,Mat *newmat)
{
  Mat               B;
  PetscInt          i,m,N,nzeros = 0,*ia,*ja,len,rstart,cnt,j,*a;
  const PetscInt    *rj;
  const PetscScalar *ra;
  MPI_Comm          comm;

  PetscFunctionBegin;
  PetscCall(MatGetSize(A,NULL,&N));
  PetscCall(MatGetLocalSize(A,&m,NULL));
  PetscCall(MatGetOwnershipRange(A,&rstart,NULL));

  /* count the number of nonzeros per row */
  for (i=0; i<m; i++) {
    PetscCall(MatGetRow(A,i+rstart,&len,&rj,NULL));
    for (j=0; j<len; j++) {
      if (rj[j] == i+rstart) {len--; break;}    /* don't count diagonal */
    }
    nzeros += len;
    PetscCall(MatRestoreRow(A,i+rstart,&len,&rj,NULL));
  }

  /* malloc space for nonzeros */
  PetscCall(PetscMalloc1(nzeros+1,&a));
  PetscCall(PetscMalloc1(N+1,&ia));
  PetscCall(PetscMalloc1(nzeros+1,&ja));

  nzeros = 0;
  ia[0]  = 0;
  for (i=0; i<m; i++) {
    PetscCall(MatGetRow(A,i+rstart,&len,&rj,&ra));
    cnt  = 0;
    for (j=0; j<len; j++) {
      if (rj[j] != i+rstart) { /* if not diagonal */
        a[nzeros+cnt]    = (PetscInt) PetscAbsScalar(ra[j]);
        ja[nzeros+cnt++] = rj[j];
      }
    }
    PetscCall(MatRestoreRow(A,i+rstart,&len,&rj,&ra));
    nzeros += cnt;
    ia[i+1] = nzeros;
  }

  PetscCall(PetscObjectGetComm((PetscObject)A,&comm));
  PetscCall(MatCreate(comm,&B));
  PetscCall(MatSetSizes(B,m,PETSC_DETERMINE,PETSC_DETERMINE,N));
  PetscCall(MatSetType(B,type));
  PetscCall(MatMPIAdjSetPreallocation(B,ia,ja,a));

  if (reuse == MAT_INPLACE_MATRIX) {
    PetscCall(MatHeaderReplace(A,&B));
  } else {
    *newmat = B;
  }
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------*/
static struct _MatOps MatOps_Values = {NULL,
                                       MatGetRow_MPIAdj,
                                       MatRestoreRow_MPIAdj,
                                       NULL,
                                /* 4*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                /*10*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                /*15*/ NULL,
                                       MatEqual_MPIAdj,
                                       NULL,
                                       NULL,
                                       NULL,
                                /*20*/ NULL,
                                       NULL,
                                       MatSetOption_MPIAdj,
                                       NULL,
                                /*24*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                /*29*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                /*34*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                /*39*/ NULL,
                                       MatCreateSubMatrices_MPIAdj,
                                       NULL,
                                       NULL,
                                       NULL,
                                /*44*/ NULL,
                                       NULL,
                                       MatShift_Basic,
                                       NULL,
                                       NULL,
                                /*49*/ NULL,
                                       MatGetRowIJ_MPIAdj,
                                       MatRestoreRowIJ_MPIAdj,
                                       NULL,
                                       NULL,
                                /*54*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                /*59*/ NULL,
                                       MatDestroy_MPIAdj,
                                       MatView_MPIAdj,
                                       MatConvertFrom_MPIAdj,
                                       NULL,
                                /*64*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                /*69*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                /*74*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                /*79*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                /*84*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                /*89*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                /*94*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                /*99*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                               /*104*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                               /*109*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                               /*114*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                               /*119*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                               /*124*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       MatCreateSubMatricesMPI_MPIAdj,
                               /*129*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                               /*134*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                               /*139*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                /*144*/NULL,
                                       NULL,
                                       NULL,
                                       NULL
};

static PetscErrorCode  MatMPIAdjSetPreallocation_MPIAdj(Mat B,PetscInt *i,PetscInt *j,PetscInt *values)
{
  Mat_MPIAdj     *b = (Mat_MPIAdj*)B->data;
  PetscBool       useedgeweights;

  PetscFunctionBegin;
  PetscCall(PetscLayoutSetUp(B->rmap));
  PetscCall(PetscLayoutSetUp(B->cmap));
  if (values) useedgeweights = PETSC_TRUE; else useedgeweights = PETSC_FALSE;
  /* Make everybody knows if they are using edge weights or not */
  PetscCall(MPIU_Allreduce((int*)&useedgeweights,(int*)&b->useedgeweights,1,MPI_INT,MPI_MAX,PetscObjectComm((PetscObject)B)));

  if (PetscDefined(USE_DEBUG)) {
    PetscInt ii;

    PetscCheck(i[0] == 0,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"First i[] index must be zero, instead it is %" PetscInt_FMT,i[0]);
    for (ii=1; ii<B->rmap->n; ii++) {
      PetscCheckFalse(i[ii] < 0 || i[ii] < i[ii-1],PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"i[%" PetscInt_FMT "]=%" PetscInt_FMT " index is out of range: i[%" PetscInt_FMT "]=%" PetscInt_FMT,ii,i[ii],ii-1,i[ii-1]);
    }
    for (ii=0; ii<i[B->rmap->n]; ii++) {
      PetscCheckFalse(j[ii] < 0 || j[ii] >= B->cmap->N,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Column index %" PetscInt_FMT " out of range %" PetscInt_FMT,ii,j[ii]);
    }
  }
  B->preallocated = PETSC_TRUE;

  b->j      = j;
  b->i      = i;
  b->values = values;

  b->nz        = i[B->rmap->n];
  b->diag      = NULL;
  b->symmetric = PETSC_FALSE;
  b->freeaij   = PETSC_TRUE;

  PetscCall(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMPIAdjCreateNonemptySubcommMat_MPIAdj(Mat A,Mat *B)
{
  Mat_MPIAdj     *a = (Mat_MPIAdj*)A->data;
  const PetscInt *ranges;
  MPI_Comm       acomm,bcomm;
  MPI_Group      agroup,bgroup;
  PetscMPIInt    i,rank,size,nranks,*ranks;

  PetscFunctionBegin;
  *B    = NULL;
  PetscCall(PetscObjectGetComm((PetscObject)A,&acomm));
  PetscCallMPI(MPI_Comm_size(acomm,&size));
  PetscCallMPI(MPI_Comm_size(acomm,&rank));
  PetscCall(MatGetOwnershipRanges(A,&ranges));
  for (i=0,nranks=0; i<size; i++) {
    if (ranges[i+1] - ranges[i] > 0) nranks++;
  }
  if (nranks == size) {         /* All ranks have a positive number of rows, so we do not need to create a subcomm; */
    PetscCall(PetscObjectReference((PetscObject)A));
    *B   = A;
    PetscFunctionReturn(0);
  }

  PetscCall(PetscMalloc1(nranks,&ranks));
  for (i=0,nranks=0; i<size; i++) {
    if (ranges[i+1] - ranges[i] > 0) ranks[nranks++] = i;
  }
  PetscCallMPI(MPI_Comm_group(acomm,&agroup));
  PetscCallMPI(MPI_Group_incl(agroup,nranks,ranks,&bgroup));
  PetscCall(PetscFree(ranks));
  PetscCallMPI(MPI_Comm_create(acomm,bgroup,&bcomm));
  PetscCallMPI(MPI_Group_free(&agroup));
  PetscCallMPI(MPI_Group_free(&bgroup));
  if (bcomm != MPI_COMM_NULL) {
    PetscInt   m,N;
    Mat_MPIAdj *b;
    PetscCall(MatGetLocalSize(A,&m,NULL));
    PetscCall(MatGetSize(A,NULL,&N));
    PetscCall(MatCreateMPIAdj(bcomm,m,N,a->i,a->j,a->values,B));
    b          = (Mat_MPIAdj*)(*B)->data;
    b->freeaij = PETSC_FALSE;
    PetscCallMPI(MPI_Comm_free(&bcomm));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode  MatMPIAdjToSeq_MPIAdj(Mat A,Mat *B)
{
  PetscInt       M,N,*II,*J,NZ,nz,m,nzstart,i;
  PetscInt       *Values = NULL;
  Mat_MPIAdj     *adj = (Mat_MPIAdj*)A->data;
  PetscMPIInt    mnz,mm,*allnz,*allm,size,*dispnz,*dispm;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)A),&size));
  PetscCall(MatGetSize(A,&M,&N));
  PetscCall(MatGetLocalSize(A,&m,NULL));
  nz   = adj->nz;
  PetscCheck(adj->i[m] == nz,PETSC_COMM_SELF,PETSC_ERR_PLIB,"nz %" PetscInt_FMT " not correct i[m] %" PetscInt_FMT,nz,adj->i[m]);
  PetscCallMPI(MPI_Allreduce(&nz,&NZ,1,MPIU_INT,MPI_SUM,PetscObjectComm((PetscObject)A)));

  PetscCall(PetscMPIIntCast(nz,&mnz));
  PetscCall(PetscMalloc2(size,&allnz,size,&dispnz));
  PetscCallMPI(MPI_Allgather(&mnz,1,MPI_INT,allnz,1,MPI_INT,PetscObjectComm((PetscObject)A)));
  dispnz[0] = 0; for (i=1; i<size; i++) dispnz[i] = dispnz[i-1]+ allnz[i-1];
  if (adj->values) {
    PetscCall(PetscMalloc1(NZ,&Values));
    PetscCallMPI(MPI_Allgatherv(adj->values,mnz,MPIU_INT,Values,allnz,dispnz,MPIU_INT,PetscObjectComm((PetscObject)A)));
  }
  PetscCall(PetscMalloc1(NZ,&J));
  PetscCallMPI(MPI_Allgatherv(adj->j,mnz,MPIU_INT,J,allnz,dispnz,MPIU_INT,PetscObjectComm((PetscObject)A)));
  PetscCall(PetscFree2(allnz,dispnz));
  PetscCallMPI(MPI_Scan(&nz,&nzstart,1,MPIU_INT,MPI_SUM,PetscObjectComm((PetscObject)A)));
  nzstart -= nz;
  /* shift the i[] values so they will be correct after being received */
  for (i=0; i<m; i++) adj->i[i] += nzstart;
  PetscCall(PetscMalloc1(M+1,&II));
  PetscCall(PetscMPIIntCast(m,&mm));
  PetscCall(PetscMalloc2(size,&allm,size,&dispm));
  PetscCallMPI(MPI_Allgather(&mm,1,MPI_INT,allm,1,MPI_INT,PetscObjectComm((PetscObject)A)));
  dispm[0] = 0; for (i=1; i<size; i++) dispm[i] = dispm[i-1]+ allm[i-1];
  PetscCallMPI(MPI_Allgatherv(adj->i,mm,MPIU_INT,II,allm,dispm,MPIU_INT,PetscObjectComm((PetscObject)A)));
  PetscCall(PetscFree2(allm,dispm));
  II[M] = NZ;
  /* shift the i[] values back */
  for (i=0; i<m; i++) adj->i[i] -= nzstart;
  PetscCall(MatCreateMPIAdj(PETSC_COMM_SELF,M,N,II,J,Values,B));
  PetscFunctionReturn(0);
}

/*@
   MatMPIAdjCreateNonemptySubcommMat - create the same MPIAdj matrix on a subcommunicator containing only processes owning a positive number of rows

   Collective

   Input Parameter:
.  A - original MPIAdj matrix

   Output Parameter:
.  B - matrix on subcommunicator, NULL on ranks that owned zero rows of A

   Level: developer

   Note:
   This function is mostly useful for internal use by mesh partitioning packages that require that every process owns at least one row.

   The matrix B should be destroyed with MatDestroy(). The arrays are not copied, so B should be destroyed before A is destroyed.

.seealso: `MatCreateMPIAdj()`
@*/
PetscErrorCode MatMPIAdjCreateNonemptySubcommMat(Mat A,Mat *B)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscUseMethod(A,"MatMPIAdjCreateNonemptySubcommMat_C",(Mat,Mat*),(A,B));
  PetscFunctionReturn(0);
}

/*MC
   MATMPIADJ - MATMPIADJ = "mpiadj" - A matrix type to be used for distributed adjacency matrices,
   intended for use constructing orderings and partitionings.

  Level: beginner

.seealso: `MatCreateMPIAdj`
M*/

PETSC_EXTERN PetscErrorCode MatCreate_MPIAdj(Mat B)
{
  Mat_MPIAdj     *b;

  PetscFunctionBegin;
  PetscCall(PetscNewLog(B,&b));
  B->data      = (void*)b;
  PetscCall(PetscMemcpy(B->ops,&MatOps_Values,sizeof(struct _MatOps)));
  B->assembled = PETSC_FALSE;

  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatMPIAdjSetPreallocation_C",MatMPIAdjSetPreallocation_MPIAdj));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatMPIAdjCreateNonemptySubcommMat_C",MatMPIAdjCreateNonemptySubcommMat_MPIAdj));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatMPIAdjToSeq_C",MatMPIAdjToSeq_MPIAdj));
  PetscCall(PetscObjectChangeTypeName((PetscObject)B,MATMPIADJ));
  PetscFunctionReturn(0);
}

/*@C
   MatMPIAdjToSeq - Converts an parallel MPIAdj matrix to complete MPIAdj on each process (needed by sequential preconditioners)

   Logically Collective

   Input Parameter:
.  A - the matrix

   Output Parameter:
.  B - the same matrix on all processes

   Level: intermediate

.seealso: `MatCreate()`, `MatCreateMPIAdj()`, `MatSetValues()`
@*/
PetscErrorCode  MatMPIAdjToSeq(Mat A,Mat *B)
{
  PetscFunctionBegin;
  PetscUseMethod(A,"MatMPIAdjToSeq_C",(Mat,Mat*),(A,B));
  PetscFunctionReturn(0);
}

/*@C
   MatMPIAdjSetPreallocation - Sets the array used for storing the matrix elements

   Logically Collective

   Input Parameters:
+  A - the matrix
.  i - the indices into j for the start of each row
.  j - the column indices for each row (sorted for each row).
       The indices in i and j start with zero (NOT with one).
-  values - [optional] edge weights

   Level: intermediate

.seealso: `MatCreate()`, `MatCreateMPIAdj()`, `MatSetValues()`
@*/
PetscErrorCode  MatMPIAdjSetPreallocation(Mat B,PetscInt *i,PetscInt *j,PetscInt *values)
{
  PetscFunctionBegin;
  PetscTryMethod(B,"MatMPIAdjSetPreallocation_C",(Mat,PetscInt*,PetscInt*,PetscInt*),(B,i,j,values));
  PetscFunctionReturn(0);
}

/*@C
   MatCreateMPIAdj - Creates a sparse matrix representing an adjacency list.
   The matrix does not have numerical values associated with it, but is
   intended for ordering (to reduce bandwidth etc) and partitioning.

   Collective

   Input Parameters:
+  comm - MPI communicator
.  m - number of local rows
.  N - number of global columns
.  i - the indices into j for the start of each row
.  j - the column indices for each row (sorted for each row).
       The indices in i and j start with zero (NOT with one).
-  values -[optional] edge weights

   Output Parameter:
.  A - the matrix

   Level: intermediate

   Notes:
    This matrix object does not support most matrix operations, include
   MatSetValues().
   You must NOT free the ii, values and jj arrays yourself. PETSc will free them
   when the matrix is destroyed; you must allocate them with PetscMalloc(). If you
    call from Fortran you need not create the arrays with PetscMalloc().
   Should not include the matrix diagonals.

   If you already have a matrix, you can create its adjacency matrix by a call
   to MatConvert, specifying a type of MATMPIADJ.

   Possible values for MatSetOption() - MAT_STRUCTURALLY_SYMMETRIC

.seealso: `MatCreate()`, `MatConvert()`, `MatGetOrdering()`
@*/
PetscErrorCode  MatCreateMPIAdj(MPI_Comm comm,PetscInt m,PetscInt N,PetscInt *i,PetscInt *j,PetscInt *values,Mat *A)
{
  PetscFunctionBegin;
  PetscCall(MatCreate(comm,A));
  PetscCall(MatSetSizes(*A,m,PETSC_DETERMINE,PETSC_DETERMINE,N));
  PetscCall(MatSetType(*A,MATMPIADJ));
  PetscCall(MatMPIAdjSetPreallocation(*A,i,j,values));
  PetscFunctionReturn(0);
}
