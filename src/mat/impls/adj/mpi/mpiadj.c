
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
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)adj,&comm);CHKERRQ(ierr);
  ierr = MatGetLayouts(adj,&rmap,NULL);CHKERRQ(ierr);
  ierr = ISGetLocalSize(irows,&nlrows_is);CHKERRQ(ierr);
  ierr = ISGetIndices(irows,&irows_indices);CHKERRQ(ierr);
  ierr = PetscMalloc1(nlrows_is,&iremote);CHKERRQ(ierr);
  /* construct sf graph*/
  nleaves = nlrows_is;
  for (i=0; i<nlrows_is; i++){
    owner = -1;
    rlocalindex = -1;
    ierr = PetscLayoutFindOwnerIndex(rmap,irows_indices[i],&owner,&rlocalindex);CHKERRQ(ierr);
    iremote[i].rank  = owner;
    iremote[i].index = rlocalindex;
  }
  ierr = MatGetRowIJ(adj,0,PETSC_FALSE,PETSC_FALSE,&nlrows_mat,&xadj,&adjncy,&done);CHKERRQ(ierr);
  ierr = PetscCalloc4(nlrows_mat,&ncols_send,nlrows_is,&xadj_recv,nlrows_is+1,&ncols_recv_offsets,nlrows_is,&ncols_recv);CHKERRQ(ierr);
  nroots = nlrows_mat;
  for (i=0; i<nlrows_mat; i++){
    ncols_send[i] = xadj[i+1]-xadj[i];
  }
  ierr = PetscSFCreate(comm,&sf);CHKERRQ(ierr);
  ierr = PetscSFSetGraph(sf,nroots,nleaves,NULL,PETSC_OWN_POINTER,iremote,PETSC_OWN_POINTER);CHKERRQ(ierr);
  ierr = PetscSFSetType(sf,PETSCSFBASIC);CHKERRQ(ierr);
  ierr = PetscSFSetFromOptions(sf);CHKERRQ(ierr);
  ierr = PetscSFBcastBegin(sf,MPIU_INT,ncols_send,ncols_recv);CHKERRQ(ierr);
  ierr = PetscSFBcastEnd(sf,MPIU_INT,ncols_send,ncols_recv);CHKERRQ(ierr);
  ierr = PetscSFBcastBegin(sf,MPIU_INT,xadj,xadj_recv);CHKERRQ(ierr);
  ierr = PetscSFBcastEnd(sf,MPIU_INT,xadj,xadj_recv);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&sf);CHKERRQ(ierr);
  Ncols_recv =0;
  for (i=0; i<nlrows_is; i++){
    Ncols_recv             += ncols_recv[i];
    ncols_recv_offsets[i+1] = ncols_recv[i]+ncols_recv_offsets[i];
  }
  Ncols_send = 0;
  for(i=0; i<nlrows_mat; i++){
    Ncols_send += ncols_send[i];
  }
  ierr = PetscCalloc1(Ncols_recv,&iremote);CHKERRQ(ierr);
  ierr = PetscCalloc1(Ncols_recv,&adjncy_recv);CHKERRQ(ierr);
  nleaves = Ncols_recv;
  Ncols_recv = 0;
  for (i=0; i<nlrows_is; i++){
    ierr = PetscLayoutFindOwner(rmap,irows_indices[i],&owner);CHKERRQ(ierr);
    for (j=0; j<ncols_recv[i]; j++){
      iremote[Ncols_recv].rank    = owner;
      iremote[Ncols_recv++].index = xadj_recv[i]+j;
    }
  }
  ierr = ISRestoreIndices(irows,&irows_indices);CHKERRQ(ierr);
  /*if we need to deal with edge weights ???*/
  if (a->useedgeweights){ierr = PetscCalloc1(Ncols_recv,&values_recv);CHKERRQ(ierr);}
  nroots = Ncols_send;
  ierr = PetscSFCreate(comm,&sf);CHKERRQ(ierr);
  ierr = PetscSFSetGraph(sf,nroots,nleaves,NULL,PETSC_OWN_POINTER,iremote,PETSC_OWN_POINTER);CHKERRQ(ierr);
  ierr = PetscSFSetType(sf,PETSCSFBASIC);CHKERRQ(ierr);
  ierr = PetscSFSetFromOptions(sf);CHKERRQ(ierr);
  ierr = PetscSFBcastBegin(sf,MPIU_INT,adjncy,adjncy_recv);CHKERRQ(ierr);
  ierr = PetscSFBcastEnd(sf,MPIU_INT,adjncy,adjncy_recv);CHKERRQ(ierr);
  if (a->useedgeweights){
    ierr = PetscSFBcastBegin(sf,MPIU_INT,a->values,values_recv);CHKERRQ(ierr);
    ierr = PetscSFBcastEnd(sf,MPIU_INT,a->values,values_recv);CHKERRQ(ierr);
  }
  ierr = PetscSFDestroy(&sf);CHKERRQ(ierr);
  ierr = MatRestoreRowIJ(adj,0,PETSC_FALSE,PETSC_FALSE,&nlrows_mat,&xadj,&adjncy,&done);CHKERRQ(ierr);
  ierr = ISGetLocalSize(icols,&icols_n);CHKERRQ(ierr);
  ierr = ISGetIndices(icols,&icols_indices);CHKERRQ(ierr);
  rnclos = 0;
  for (i=0; i<nlrows_is; i++){
    for (j=ncols_recv_offsets[i]; j<ncols_recv_offsets[i+1]; j++){
      ierr = PetscFindInt(adjncy_recv[j], icols_n, icols_indices, &loc);CHKERRQ(ierr);
      if (loc<0){
        adjncy_recv[j] = -1;
        if (a->useedgeweights) values_recv[j] = -1;
        ncols_recv[i]--;
      } else {
        rnclos++;
      }
    }
  }
  ierr = ISRestoreIndices(icols,&icols_indices);CHKERRQ(ierr);
  ierr = PetscCalloc1(rnclos,&sadjncy);CHKERRQ(ierr);
  if (a->useedgeweights) {ierr = PetscCalloc1(rnclos,&svalues);CHKERRQ(ierr);}
  ierr = PetscCalloc1(nlrows_is+1,&sxadj);CHKERRQ(ierr);
  rnclos = 0;
  for(i=0; i<nlrows_is; i++){
    for(j=ncols_recv_offsets[i]; j<ncols_recv_offsets[i+1]; j++){
      if (adjncy_recv[j]<0) continue;
      sadjncy[rnclos] = adjncy_recv[j];
      if (a->useedgeweights) svalues[rnclos] = values_recv[j];
      rnclos++;
    }
  }
  for(i=0; i<nlrows_is; i++){
    sxadj[i+1] = sxadj[i]+ncols_recv[i];
  }
  if (sadj_xadj)  { *sadj_xadj = sxadj;} else    { ierr = PetscFree(sxadj);CHKERRQ(ierr);}
  if (sadj_adjncy){ *sadj_adjncy = sadjncy;} else { ierr = PetscFree(sadjncy);CHKERRQ(ierr);}
  if (sadj_values){
    if (a->useedgeweights) *sadj_values = svalues; else *sadj_values=NULL;
  } else {
    if (a->useedgeweights) {ierr = PetscFree(svalues);CHKERRQ(ierr);}
  }
  ierr = PetscFree4(ncols_send,xadj_recv,ncols_recv_offsets,ncols_recv);CHKERRQ(ierr);
  ierr = PetscFree(adjncy_recv);CHKERRQ(ierr);
  if (a->useedgeweights) {ierr = PetscFree(values_recv);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

static PetscErrorCode MatCreateSubMatrices_MPIAdj_Private(Mat mat,PetscInt n,const IS irow[],const IS icol[],PetscBool subcomm,MatReuse scall,Mat *submat[])
{
  PetscInt           i,irow_n,icol_n,*sxadj,*sadjncy,*svalues;
  PetscInt          *indices,nindx,j,k,loc;
  PetscMPIInt        issame;
  const PetscInt    *irow_indices,*icol_indices;
  MPI_Comm           scomm_row,scomm_col,scomm_mat;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  nindx = 0;
  /*
   * Estimate a maximum number for allocating memory
   */
  for(i=0; i<n; i++){
    ierr = ISGetLocalSize(irow[i],&irow_n);CHKERRQ(ierr);
    ierr = ISGetLocalSize(icol[i],&icol_n);CHKERRQ(ierr);
    nindx = nindx>(irow_n+icol_n)? nindx:(irow_n+icol_n);
  }
  ierr = PetscMalloc1(nindx,&indices);CHKERRQ(ierr);
  /* construct a submat */
  for (i=0; i<n; i++){
    if (subcomm){
      ierr = PetscObjectGetComm((PetscObject)irow[i],&scomm_row);CHKERRQ(ierr);
      ierr = PetscObjectGetComm((PetscObject)icol[i],&scomm_col);CHKERRQ(ierr);
      ierr = MPI_Comm_compare(scomm_row,scomm_col,&issame);CHKERRQ(ierr);
      if (issame != MPI_IDENT) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"row index set must have the same comm as the col index set\n");
      ierr = MPI_Comm_compare(scomm_row,PETSC_COMM_SELF,&issame);CHKERRQ(ierr);
      if (issame == MPI_IDENT) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP," can not use PETSC_COMM_SELF as comm when extracting a parallel submatrix\n");
    } else {
      scomm_row = PETSC_COMM_SELF;
    }
    /*get sub-matrix data*/
    sxadj=NULL; sadjncy=NULL; svalues=NULL;
    ierr = MatCreateSubMatrix_MPIAdj_data(mat,irow[i],icol[i],&sxadj,&sadjncy,&svalues);CHKERRQ(ierr);
    ierr = ISGetLocalSize(irow[i],&irow_n);CHKERRQ(ierr);
    ierr = ISGetLocalSize(icol[i],&icol_n);CHKERRQ(ierr);
    ierr = ISGetIndices(irow[i],&irow_indices);CHKERRQ(ierr);
    ierr = PetscArraycpy(indices,irow_indices,irow_n);CHKERRQ(ierr);
    ierr = ISRestoreIndices(irow[i],&irow_indices);CHKERRQ(ierr);
    ierr = ISGetIndices(icol[i],&icol_indices);CHKERRQ(ierr);
    ierr = PetscArraycpy(indices+irow_n,icol_indices,icol_n);CHKERRQ(ierr);
    ierr = ISRestoreIndices(icol[i],&icol_indices);CHKERRQ(ierr);
    nindx = irow_n+icol_n;
    ierr = PetscSortRemoveDupsInt(&nindx,indices);CHKERRQ(ierr);
    /* renumber columns */
    for (j=0; j<irow_n; j++){
      for (k=sxadj[j]; k<sxadj[j+1]; k++){
        ierr = PetscFindInt(sadjncy[k],nindx,indices,&loc);CHKERRQ(ierr);
        if (loc<0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"can not find col %D",sadjncy[k]);
        sadjncy[k] = loc;
      }
    }
    if (scall==MAT_INITIAL_MATRIX){
      ierr = MatCreateMPIAdj(scomm_row,irow_n,icol_n,sxadj,sadjncy,svalues,submat[i]);CHKERRQ(ierr);
    } else {
       Mat                sadj = *(submat[i]);
       Mat_MPIAdj         *sa  = (Mat_MPIAdj*)((sadj)->data);
       ierr = PetscObjectGetComm((PetscObject)sadj,&scomm_mat);CHKERRQ(ierr);
       ierr = MPI_Comm_compare(scomm_row,scomm_mat,&issame);CHKERRQ(ierr);
       if (issame != MPI_IDENT) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"submatrix  must have the same comm as the col index set\n");
       ierr = PetscArraycpy(sa->i,sxadj,irow_n+1);CHKERRQ(ierr);
       ierr = PetscArraycpy(sa->j,sadjncy,sxadj[irow_n]);CHKERRQ(ierr);
       if (svalues){ierr = PetscArraycpy(sa->values,svalues,sxadj[irow_n]);CHKERRQ(ierr);}
       ierr = PetscFree(sxadj);CHKERRQ(ierr);
       ierr = PetscFree(sadjncy);CHKERRQ(ierr);
       if (svalues) {ierr = PetscFree(svalues);CHKERRQ(ierr);}
    }
  }
  ierr = PetscFree(indices);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatCreateSubMatricesMPI_MPIAdj(Mat mat,PetscInt n, const IS irow[],const IS icol[],MatReuse scall,Mat *submat[])
{
  PetscErrorCode     ierr;
  /*get sub-matrices across a sub communicator */
  PetscFunctionBegin;
  ierr = MatCreateSubMatrices_MPIAdj_Private(mat,n,irow,icol,PETSC_TRUE,scall,submat);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatCreateSubMatrices_MPIAdj(Mat mat,PetscInt n,const IS irow[],const IS icol[],MatReuse scall,Mat *submat[])
{
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  /*get sub-matrices based on PETSC_COMM_SELF */
  ierr = MatCreateSubMatrices_MPIAdj_Private(mat,n,irow,icol,PETSC_FALSE,scall,submat);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatView_MPIAdj_ASCII(Mat A,PetscViewer viewer)
{
  Mat_MPIAdj        *a = (Mat_MPIAdj*)A->data;
  PetscErrorCode    ierr;
  PetscInt          i,j,m = A->rmap->n;
  const char        *name;
  PetscViewerFormat format;

  PetscFunctionBegin;
  ierr = PetscObjectGetName((PetscObject)A,&name);CHKERRQ(ierr);
  ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
  if (format == PETSC_VIEWER_ASCII_INFO) {
    PetscFunctionReturn(0);
  } else if (format == PETSC_VIEWER_ASCII_MATLAB) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"MATLAB format not supported");
  else {
    ierr = PetscViewerASCIIUseTabs(viewer,PETSC_FALSE);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushSynchronized(viewer);CHKERRQ(ierr);
    for (i=0; i<m; i++) {
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"row %D:",i+A->rmap->rstart);CHKERRQ(ierr);
      for (j=a->i[i]; j<a->i[i+1]; j++) {
        if (a->values) {
          ierr = PetscViewerASCIISynchronizedPrintf(viewer," (%D, %D) ",a->j[j], a->values[j]);CHKERRQ(ierr);
        } else {
          ierr = PetscViewerASCIISynchronizedPrintf(viewer," %D ",a->j[j]);CHKERRQ(ierr);
        }
      }
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"\n");CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIUseTabs(viewer,PETSC_TRUE);CHKERRQ(ierr);
    ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopSynchronized(viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatView_MPIAdj(Mat A,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscBool      iascii;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = MatView_MPIAdj_ASCII(A,viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDestroy_MPIAdj(Mat mat)
{
  Mat_MPIAdj     *a = (Mat_MPIAdj*)mat->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
#if defined(PETSC_USE_LOG)
  PetscLogObjectState((PetscObject)mat,"Rows=%D, Cols=%D, NZ=%D",mat->rmap->n,mat->cmap->n,a->nz);
#endif
  ierr = PetscFree(a->diag);CHKERRQ(ierr);
  if (a->freeaij) {
    if (a->freeaijwithfree) {
      if (a->i) free(a->i);
      if (a->j) free(a->j);
    } else {
      ierr = PetscFree(a->i);CHKERRQ(ierr);
      ierr = PetscFree(a->j);CHKERRQ(ierr);
      ierr = PetscFree(a->values);CHKERRQ(ierr);
    }
  }
  ierr = PetscFree(a->rowvalues);CHKERRQ(ierr);
  ierr = PetscFree(mat->data);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)mat,NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatMPIAdjSetPreallocation_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatMPIAdjCreateNonemptySubcommMat_C",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSetOption_MPIAdj(Mat A,MatOption op,PetscBool flg)
{
  Mat_MPIAdj     *a = (Mat_MPIAdj*)A->data;
  PetscErrorCode ierr;

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
    ierr = PetscInfo1(A,"Option %s ignored\n",MatOptions[op]);CHKERRQ(ierr);
    break;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatGetRow_MPIAdj(Mat A,PetscInt row,PetscInt *nz,PetscInt **idx,PetscScalar **v)
{
  Mat_MPIAdj *a = (Mat_MPIAdj*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  row -= A->rmap->rstart;
  if (row < 0 || row >= A->rmap->n) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Row out of range");
  *nz = a->i[row+1] - a->i[row];
  if (v) {
    PetscInt j;
    if (a->rowvalues_alloc < *nz) {
      ierr = PetscFree(a->rowvalues);CHKERRQ(ierr);
      a->rowvalues_alloc = PetscMax(a->rowvalues_alloc*2, *nz);
      ierr = PetscMalloc1(a->rowvalues_alloc,&a->rowvalues);CHKERRQ(ierr);
    }
    for (j=0; j<*nz; j++){
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
  PetscErrorCode ierr;
  PetscBool      flag;

  PetscFunctionBegin;
  /* If the  matrix dimensions are not equal,or no of nonzeros */
  if ((A->rmap->n != B->rmap->n) ||(a->nz != b->nz)) {
    flag = PETSC_FALSE;
  }

  /* if the a->i are the same */
  ierr = PetscArraycmp(a->i,b->i,A->rmap->n+1,&flag);CHKERRQ(ierr);

  /* if a->j are the same */
  ierr = PetscMemcmp(a->j,b->j,(a->nz)*sizeof(PetscInt),&flag);CHKERRQ(ierr);

  ierr = MPIU_Allreduce(&flag,flg,1,MPIU_BOOL,MPI_LAND,PetscObjectComm((PetscObject)A));CHKERRQ(ierr);
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
  if (ia && a->i != *ia) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONGSTATE,"ia passed back is not one obtained with MatGetRowIJ()");
  if (ja && a->j != *ja) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONGSTATE,"ja passed back is not one obtained with MatGetRowIJ()");
  if (oshift) {
    if (!ia) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONGSTATE,"If oshift then you must passed in inia[] argument");
    if (!ja) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONGSTATE,"If oshift then you must passed in inja[] argument");
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
  PetscErrorCode    ierr;
  PetscInt          i,m,N,nzeros = 0,*ia,*ja,len,rstart,cnt,j,*a;
  const PetscInt    *rj;
  const PetscScalar *ra;
  MPI_Comm          comm;

  PetscFunctionBegin;
  ierr = MatGetSize(A,NULL,&N);CHKERRQ(ierr);
  ierr = MatGetLocalSize(A,&m,NULL);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(A,&rstart,NULL);CHKERRQ(ierr);

  /* count the number of nonzeros per row */
  for (i=0; i<m; i++) {
    ierr = MatGetRow(A,i+rstart,&len,&rj,NULL);CHKERRQ(ierr);
    for (j=0; j<len; j++) {
      if (rj[j] == i+rstart) {len--; break;}    /* don't count diagonal */
    }
    nzeros += len;
    ierr    = MatRestoreRow(A,i+rstart,&len,&rj,NULL);CHKERRQ(ierr);
  }

  /* malloc space for nonzeros */
  ierr = PetscMalloc1(nzeros+1,&a);CHKERRQ(ierr);
  ierr = PetscMalloc1(N+1,&ia);CHKERRQ(ierr);
  ierr = PetscMalloc1(nzeros+1,&ja);CHKERRQ(ierr);

  nzeros = 0;
  ia[0]  = 0;
  for (i=0; i<m; i++) {
    ierr = MatGetRow(A,i+rstart,&len,&rj,&ra);CHKERRQ(ierr);
    cnt  = 0;
    for (j=0; j<len; j++) {
      if (rj[j] != i+rstart) { /* if not diagonal */
        a[nzeros+cnt]    = (PetscInt) PetscAbsScalar(ra[j]);
        ja[nzeros+cnt++] = rj[j];
      }
    }
    ierr    = MatRestoreRow(A,i+rstart,&len,&rj,&ra);CHKERRQ(ierr);
    nzeros += cnt;
    ia[i+1] = nzeros;
  }

  ierr = PetscObjectGetComm((PetscObject)A,&comm);CHKERRQ(ierr);
  ierr = MatCreate(comm,&B);CHKERRQ(ierr);
  ierr = MatSetSizes(B,m,PETSC_DETERMINE,PETSC_DETERMINE,N);CHKERRQ(ierr);
  ierr = MatSetType(B,type);CHKERRQ(ierr);
  ierr = MatMPIAdjSetPreallocation(B,ia,ja,a);CHKERRQ(ierr);

  if (reuse == MAT_INPLACE_MATRIX) {
    ierr = MatHeaderReplace(A,&B);CHKERRQ(ierr);
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
                                       NULL
};

static PetscErrorCode  MatMPIAdjSetPreallocation_MPIAdj(Mat B,PetscInt *i,PetscInt *j,PetscInt *values)
{
  Mat_MPIAdj     *b = (Mat_MPIAdj*)B->data;
  PetscBool       useedgeweights;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLayoutSetUp(B->rmap);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(B->cmap);CHKERRQ(ierr);
  if (values) useedgeweights = PETSC_TRUE; else useedgeweights = PETSC_FALSE;
  /* Make everybody knows if they are using edge weights or not */
  ierr = MPIU_Allreduce((int*)&useedgeweights,(int*)&b->useedgeweights,1,MPI_INT,MPI_MAX,PetscObjectComm((PetscObject)B));CHKERRQ(ierr);

  if (PetscDefined(USE_DEBUG)) {
    PetscInt ii;

    if (i[0] != 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"First i[] index must be zero, instead it is %D\n",i[0]);
    for (ii=1; ii<B->rmap->n; ii++) {
      if (i[ii] < 0 || i[ii] < i[ii-1]) SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"i[%D]=%D index is out of range: i[%D]=%D",ii,i[ii],ii-1,i[ii-1]);
    }
    for (ii=0; ii<i[B->rmap->n]; ii++) {
      if (j[ii] < 0 || j[ii] >= B->cmap->N) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Column index %D out of range %D\n",ii,j[ii]);
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

  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMPIAdjCreateNonemptySubcommMat_MPIAdj(Mat A,Mat *B)
{
  Mat_MPIAdj     *a = (Mat_MPIAdj*)A->data;
  PetscErrorCode ierr;
  const PetscInt *ranges;
  MPI_Comm       acomm,bcomm;
  MPI_Group      agroup,bgroup;
  PetscMPIInt    i,rank,size,nranks,*ranks;

  PetscFunctionBegin;
  *B    = NULL;
  ierr  = PetscObjectGetComm((PetscObject)A,&acomm);CHKERRQ(ierr);
  ierr  = MPI_Comm_size(acomm,&size);CHKERRQ(ierr);
  ierr  = MPI_Comm_size(acomm,&rank);CHKERRQ(ierr);
  ierr  = MatGetOwnershipRanges(A,&ranges);CHKERRQ(ierr);
  for (i=0,nranks=0; i<size; i++) {
    if (ranges[i+1] - ranges[i] > 0) nranks++;
  }
  if (nranks == size) {         /* All ranks have a positive number of rows, so we do not need to create a subcomm; */
    ierr = PetscObjectReference((PetscObject)A);CHKERRQ(ierr);
    *B   = A;
    PetscFunctionReturn(0);
  }

  ierr = PetscMalloc1(nranks,&ranks);CHKERRQ(ierr);
  for (i=0,nranks=0; i<size; i++) {
    if (ranges[i+1] - ranges[i] > 0) ranks[nranks++] = i;
  }
  ierr = MPI_Comm_group(acomm,&agroup);CHKERRQ(ierr);
  ierr = MPI_Group_incl(agroup,nranks,ranks,&bgroup);CHKERRQ(ierr);
  ierr = PetscFree(ranks);CHKERRQ(ierr);
  ierr = MPI_Comm_create(acomm,bgroup,&bcomm);CHKERRQ(ierr);
  ierr = MPI_Group_free(&agroup);CHKERRQ(ierr);
  ierr = MPI_Group_free(&bgroup);CHKERRQ(ierr);
  if (bcomm != MPI_COMM_NULL) {
    PetscInt   m,N;
    Mat_MPIAdj *b;
    ierr       = MatGetLocalSize(A,&m,NULL);CHKERRQ(ierr);
    ierr       = MatGetSize(A,NULL,&N);CHKERRQ(ierr);
    ierr       = MatCreateMPIAdj(bcomm,m,N,a->i,a->j,a->values,B);CHKERRQ(ierr);
    b          = (Mat_MPIAdj*)(*B)->data;
    b->freeaij = PETSC_FALSE;
    ierr       = MPI_Comm_free(&bcomm);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode  MatMPIAdjToSeq_MPIAdj(Mat A,Mat *B)
{
  PetscErrorCode ierr;
  PetscInt       M,N,*II,*J,NZ,nz,m,nzstart,i;
  PetscInt       *Values = NULL;
  Mat_MPIAdj     *adj = (Mat_MPIAdj*)A->data;
  PetscMPIInt    mnz,mm,*allnz,*allm,size,*dispnz,*dispm;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)A),&size);CHKERRQ(ierr);
  ierr = MatGetSize(A,&M,&N);CHKERRQ(ierr);
  ierr = MatGetLocalSize(A,&m,NULL);CHKERRQ(ierr);
  nz   = adj->nz;
  if (adj->i[m] != nz) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"nz %D not correct i[m] %d",nz,adj->i[m]);
  ierr = MPI_Allreduce(&nz,&NZ,1,MPIU_INT,MPI_SUM,PetscObjectComm((PetscObject)A));CHKERRQ(ierr);

  ierr = PetscMPIIntCast(nz,&mnz);CHKERRQ(ierr);
  ierr = PetscMalloc2(size,&allnz,size,&dispnz);CHKERRQ(ierr);
  ierr = MPI_Allgather(&mnz,1,MPI_INT,allnz,1,MPI_INT,PetscObjectComm((PetscObject)A));CHKERRQ(ierr);
  dispnz[0] = 0; for (i=1; i<size; i++) dispnz[i] = dispnz[i-1]+ allnz[i-1];
  if (adj->values) {
    ierr = PetscMalloc1(NZ,&Values);CHKERRQ(ierr);
    ierr = MPI_Allgatherv(adj->values,mnz,MPIU_INT,Values,allnz,dispnz,MPIU_INT,PetscObjectComm((PetscObject)A));CHKERRQ(ierr);
  }
  ierr = PetscMalloc1(NZ,&J);CHKERRQ(ierr);
  ierr = MPI_Allgatherv(adj->j,mnz,MPIU_INT,J,allnz,dispnz,MPIU_INT,PetscObjectComm((PetscObject)A));CHKERRQ(ierr);
  ierr = PetscFree2(allnz,dispnz);CHKERRQ(ierr);
  ierr = MPI_Scan(&nz,&nzstart,1,MPIU_INT,MPI_SUM,PetscObjectComm((PetscObject)A));CHKERRQ(ierr);
  nzstart -= nz;
  /* shift the i[] values so they will be correct after being received */
  for (i=0; i<m; i++) adj->i[i] += nzstart;
  ierr = PetscMalloc1(M+1,&II);CHKERRQ(ierr);
  ierr = PetscMPIIntCast(m,&mm);CHKERRQ(ierr);
  ierr = PetscMalloc2(size,&allm,size,&dispm);CHKERRQ(ierr);
  ierr = MPI_Allgather(&mm,1,MPI_INT,allm,1,MPI_INT,PetscObjectComm((PetscObject)A));CHKERRQ(ierr);
  dispm[0] = 0; for (i=1; i<size; i++) dispm[i] = dispm[i-1]+ allm[i-1];
  ierr = MPI_Allgatherv(adj->i,mm,MPIU_INT,II,allm,dispm,MPIU_INT,PetscObjectComm((PetscObject)A));CHKERRQ(ierr);
  ierr = PetscFree2(allm,dispm);CHKERRQ(ierr);
  II[M] = NZ;
  /* shift the i[] values back */
  for (i=0; i<m; i++) adj->i[i] -= nzstart;
  ierr = MatCreateMPIAdj(PETSC_COMM_SELF,M,N,II,J,Values,B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   MatMPIAdjCreateNonemptySubcommMat - create the same MPIAdj matrix on a subcommunicator containing only processes owning a positive number of rows

   Collective

   Input Arguments:
.  A - original MPIAdj matrix

   Output Arguments:
.  B - matrix on subcommunicator, NULL on ranks that owned zero rows of A

   Level: developer

   Note:
   This function is mostly useful for internal use by mesh partitioning packages that require that every process owns at least one row.

   The matrix B should be destroyed with MatDestroy(). The arrays are not copied, so B should be destroyed before A is destroyed.

.seealso: MatCreateMPIAdj()
@*/
PetscErrorCode MatMPIAdjCreateNonemptySubcommMat(Mat A,Mat *B)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  ierr = PetscUseMethod(A,"MatMPIAdjCreateNonemptySubcommMat_C",(Mat,Mat*),(A,B));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
   MATMPIADJ - MATMPIADJ = "mpiadj" - A matrix type to be used for distributed adjacency matrices,
   intended for use constructing orderings and partitionings.

  Level: beginner

.seealso: MatCreateMPIAdj
M*/

PETSC_EXTERN PetscErrorCode MatCreate_MPIAdj(Mat B)
{
  Mat_MPIAdj     *b;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr         = PetscNewLog(B,&b);CHKERRQ(ierr);
  B->data      = (void*)b;
  ierr         = PetscMemcpy(B->ops,&MatOps_Values,sizeof(struct _MatOps));CHKERRQ(ierr);
  B->assembled = PETSC_FALSE;

  ierr = PetscObjectComposeFunction((PetscObject)B,"MatMPIAdjSetPreallocation_C",MatMPIAdjSetPreallocation_MPIAdj);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatMPIAdjCreateNonemptySubcommMat_C",MatMPIAdjCreateNonemptySubcommMat_MPIAdj);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatMPIAdjToSeq_C",MatMPIAdjToSeq_MPIAdj);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)B,MATMPIADJ);CHKERRQ(ierr);
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

.seealso: MatCreate(), MatCreateMPIAdj(), MatSetValues()
@*/
PetscErrorCode  MatMPIAdjToSeq(Mat A,Mat *B)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscUseMethod(A,"MatMPIAdjToSeq_C",(Mat,Mat*),(A,B));CHKERRQ(ierr);
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

.seealso: MatCreate(), MatCreateMPIAdj(), MatSetValues()
@*/
PetscErrorCode  MatMPIAdjSetPreallocation(Mat B,PetscInt *i,PetscInt *j,PetscInt *values)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscTryMethod(B,"MatMPIAdjSetPreallocation_C",(Mat,PetscInt*,PetscInt*,PetscInt*),(B,i,j,values));CHKERRQ(ierr);
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

.seealso: MatCreate(), MatConvert(), MatGetOrdering()
@*/
PetscErrorCode  MatCreateMPIAdj(MPI_Comm comm,PetscInt m,PetscInt N,PetscInt *i,PetscInt *j,PetscInt *values,Mat *A)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatCreate(comm,A);CHKERRQ(ierr);
  ierr = MatSetSizes(*A,m,PETSC_DETERMINE,PETSC_DETERMINE,N);CHKERRQ(ierr);
  ierr = MatSetType(*A,MATMPIADJ);CHKERRQ(ierr);
  ierr = MatMPIAdjSetPreallocation(*A,i,j,values);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
