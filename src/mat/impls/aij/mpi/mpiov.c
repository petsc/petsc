/*
   Routines to compute overlapping regions of a parallel MPI matrix
  and to find submatrices that were shared across processors.
*/
#include <../src/mat/impls/aij/seq/aij.h>
#include <../src/mat/impls/aij/mpi/mpiaij.h>
#include <petscbt.h>
#include <petscsf.h>

static PetscErrorCode MatIncreaseOverlap_MPIAIJ_Once(Mat,PetscInt,IS*);
static PetscErrorCode MatIncreaseOverlap_MPIAIJ_Local(Mat,PetscInt,char**,PetscInt*,PetscInt**,PetscTable*);
static PetscErrorCode MatIncreaseOverlap_MPIAIJ_Receive(Mat,PetscInt,PetscInt**,PetscInt**,PetscInt*);
extern PetscErrorCode MatGetRow_MPIAIJ(Mat,PetscInt,PetscInt*,PetscInt**,PetscScalar**);
extern PetscErrorCode MatRestoreRow_MPIAIJ(Mat,PetscInt,PetscInt*,PetscInt**,PetscScalar**);

static PetscErrorCode MatIncreaseOverlap_MPIAIJ_Once_Scalable(Mat,PetscInt,IS*);
static PetscErrorCode MatIncreaseOverlap_MPIAIJ_Local_Scalable(Mat,PetscInt,IS*);
static PetscErrorCode MatIncreaseOverlap_MPIAIJ_Send_Scalable(Mat,PetscInt,PetscMPIInt,PetscMPIInt *,PetscInt *, PetscInt *,PetscInt **,PetscInt **);
static PetscErrorCode MatIncreaseOverlap_MPIAIJ_Receive_Scalable(Mat,PetscInt,IS*,PetscInt,PetscInt *);

PetscErrorCode MatIncreaseOverlap_MPIAIJ(Mat C,PetscInt imax,IS is[],PetscInt ov)
{
  PetscInt       i;

  PetscFunctionBegin;
  PetscCheckFalse(ov < 0,PetscObjectComm((PetscObject)C),PETSC_ERR_ARG_OUTOFRANGE,"Negative overlap specified");
  for (i=0; i<ov; ++i) {
    PetscCall(MatIncreaseOverlap_MPIAIJ_Once(C,imax,is));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatIncreaseOverlap_MPIAIJ_Scalable(Mat C,PetscInt imax,IS is[],PetscInt ov)
{
  PetscInt       i;

  PetscFunctionBegin;
  PetscCheckFalse(ov < 0,PetscObjectComm((PetscObject)C),PETSC_ERR_ARG_OUTOFRANGE,"Negative overlap specified");
  for (i=0; i<ov; ++i) {
    PetscCall(MatIncreaseOverlap_MPIAIJ_Once_Scalable(C,imax,is));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatIncreaseOverlap_MPIAIJ_Once_Scalable(Mat mat,PetscInt nidx,IS is[])
{
  MPI_Comm       comm;
  PetscInt       *length,length_i,tlength,*remoterows,nrrows,reducednrrows,*rrow_ranks,*rrow_isids,i,j;
  PetscInt       *tosizes,*tosizes_temp,*toffsets,*fromsizes,*todata,*fromdata;
  PetscInt       nrecvrows,*sbsizes = NULL,*sbdata = NULL;
  const PetscInt *indices_i,**indices;
  PetscLayout    rmap;
  PetscMPIInt    rank,size,*toranks,*fromranks,nto,nfrom,owner;
  PetscSF        sf;
  PetscSFNode    *remote;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)mat,&comm));
  PetscCallMPI(MPI_Comm_rank(comm,&rank));
  PetscCallMPI(MPI_Comm_size(comm,&size));
  /* get row map to determine where rows should be going */
  PetscCall(MatGetLayouts(mat,&rmap,NULL));
  /* retrieve IS data and put all together so that we
   * can optimize communication
   *  */
  PetscCall(PetscMalloc2(nidx,(PetscInt ***)&indices,nidx,&length));
  for (i=0,tlength=0; i<nidx; i++) {
    PetscCall(ISGetLocalSize(is[i],&length[i]));
    tlength += length[i];
    PetscCall(ISGetIndices(is[i],&indices[i]));
  }
  /* find these rows on remote processors */
  PetscCall(PetscCalloc3(tlength,&remoterows,tlength,&rrow_ranks,tlength,&rrow_isids));
  PetscCall(PetscCalloc3(size,&toranks,2*size,&tosizes,size,&tosizes_temp));
  nrrows = 0;
  for (i=0; i<nidx; i++) {
    length_i  = length[i];
    indices_i = indices[i];
    for (j=0; j<length_i; j++) {
      owner = -1;
      PetscCall(PetscLayoutFindOwner(rmap,indices_i[j],&owner));
      /* remote processors */
      if (owner != rank) {
        tosizes_temp[owner]++; /* number of rows to owner */
        rrow_ranks[nrrows]  = owner; /* processor */
        rrow_isids[nrrows]   = i; /* is id */
        remoterows[nrrows++] = indices_i[j]; /* row */
      }
    }
    PetscCall(ISRestoreIndices(is[i],&indices[i]));
  }
  PetscCall(PetscFree2(*(PetscInt***)&indices,length));
  /* test if we need to exchange messages
   * generally speaking, we do not need to exchange
   * data when overlap is 1
   * */
  PetscCallMPI(MPIU_Allreduce(&nrrows,&reducednrrows,1,MPIU_INT,MPIU_MAX,comm));
  /* we do not have any messages
   * It usually corresponds to overlap 1
   * */
  if (!reducednrrows) {
    PetscCall(PetscFree3(toranks,tosizes,tosizes_temp));
    PetscCall(PetscFree3(remoterows,rrow_ranks,rrow_isids));
    PetscCall(MatIncreaseOverlap_MPIAIJ_Local_Scalable(mat,nidx,is));
    PetscFunctionReturn(0);
  }
  nto = 0;
  /* send sizes and ranks for building a two-sided communcation */
  for (i=0; i<size; i++) {
    if (tosizes_temp[i]) {
      tosizes[nto*2]  = tosizes_temp[i]*2; /* size */
      tosizes_temp[i] = nto; /* a map from processor to index */
      toranks[nto++]  = i; /* processor */
    }
  }
  PetscCall(PetscMalloc1(nto+1,&toffsets));
  toffsets[0] = 0;
  for (i=0; i<nto; i++) {
    toffsets[i+1]  = toffsets[i]+tosizes[2*i]; /* offsets */
    tosizes[2*i+1] = toffsets[i]; /* offsets to send */
  }
  /* send information to other processors */
  PetscCall(PetscCommBuildTwoSided(comm,2,MPIU_INT,nto,toranks,tosizes,&nfrom,&fromranks,&fromsizes));
  nrecvrows = 0;
  for (i=0; i<nfrom; i++) nrecvrows += fromsizes[2*i];
  PetscCall(PetscMalloc1(nrecvrows,&remote));
  nrecvrows = 0;
  for (i=0; i<nfrom; i++) {
    for (j=0; j<fromsizes[2*i]; j++) {
      remote[nrecvrows].rank    = fromranks[i];
      remote[nrecvrows++].index = fromsizes[2*i+1]+j;
    }
  }
  PetscCall(PetscSFCreate(comm,&sf));
  PetscCall(PetscSFSetGraph(sf,nrecvrows,nrecvrows,NULL,PETSC_OWN_POINTER,remote,PETSC_OWN_POINTER));
  /* use two-sided communication by default since OPENMPI has some bugs for one-sided one */
  PetscCall(PetscSFSetType(sf,PETSCSFBASIC));
  PetscCall(PetscSFSetFromOptions(sf));
  /* message pair <no of is, row>  */
  PetscCall(PetscCalloc2(2*nrrows,&todata,nrecvrows,&fromdata));
  for (i=0; i<nrrows; i++) {
    owner = rrow_ranks[i]; /* processor */
    j     = tosizes_temp[owner]; /* index */
    todata[toffsets[j]++] = rrow_isids[i];
    todata[toffsets[j]++] = remoterows[i];
  }
  PetscCall(PetscFree3(toranks,tosizes,tosizes_temp));
  PetscCall(PetscFree3(remoterows,rrow_ranks,rrow_isids));
  PetscCall(PetscFree(toffsets));
  PetscCall(PetscSFBcastBegin(sf,MPIU_INT,todata,fromdata,MPI_REPLACE));
  PetscCall(PetscSFBcastEnd(sf,MPIU_INT,todata,fromdata,MPI_REPLACE));
  PetscCall(PetscSFDestroy(&sf));
  /* send rows belonging to the remote so that then we could get the overlapping data back */
  PetscCall(MatIncreaseOverlap_MPIAIJ_Send_Scalable(mat,nidx,nfrom,fromranks,fromsizes,fromdata,&sbsizes,&sbdata));
  PetscCall(PetscFree2(todata,fromdata));
  PetscCall(PetscFree(fromsizes));
  PetscCall(PetscCommBuildTwoSided(comm,2,MPIU_INT,nfrom,fromranks,sbsizes,&nto,&toranks,&tosizes));
  PetscCall(PetscFree(fromranks));
  nrecvrows = 0;
  for (i=0; i<nto; i++) nrecvrows += tosizes[2*i];
  PetscCall(PetscCalloc1(nrecvrows,&todata));
  PetscCall(PetscMalloc1(nrecvrows,&remote));
  nrecvrows = 0;
  for (i=0; i<nto; i++) {
    for (j=0; j<tosizes[2*i]; j++) {
      remote[nrecvrows].rank    = toranks[i];
      remote[nrecvrows++].index = tosizes[2*i+1]+j;
    }
  }
  PetscCall(PetscSFCreate(comm,&sf));
  PetscCall(PetscSFSetGraph(sf,nrecvrows,nrecvrows,NULL,PETSC_OWN_POINTER,remote,PETSC_OWN_POINTER));
  /* use two-sided communication by default since OPENMPI has some bugs for one-sided one */
  PetscCall(PetscSFSetType(sf,PETSCSFBASIC));
  PetscCall(PetscSFSetFromOptions(sf));
  /* overlap communication and computation */
  PetscCall(PetscSFBcastBegin(sf,MPIU_INT,sbdata,todata,MPI_REPLACE));
  PetscCall(MatIncreaseOverlap_MPIAIJ_Local_Scalable(mat,nidx,is));
  PetscCall(PetscSFBcastEnd(sf,MPIU_INT,sbdata,todata,MPI_REPLACE));
  PetscCall(PetscSFDestroy(&sf));
  PetscCall(PetscFree2(sbdata,sbsizes));
  PetscCall(MatIncreaseOverlap_MPIAIJ_Receive_Scalable(mat,nidx,is,nrecvrows,todata));
  PetscCall(PetscFree(toranks));
  PetscCall(PetscFree(tosizes));
  PetscCall(PetscFree(todata));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatIncreaseOverlap_MPIAIJ_Receive_Scalable(Mat mat,PetscInt nidx, IS is[], PetscInt nrecvs, PetscInt *recvdata)
{
  PetscInt         *isz,isz_i,i,j,is_id, data_size;
  PetscInt          col,lsize,max_lsize,*indices_temp, *indices_i;
  const PetscInt   *indices_i_temp;
  MPI_Comm         *iscomms;

  PetscFunctionBegin;
  max_lsize = 0;
  PetscCall(PetscMalloc1(nidx,&isz));
  for (i=0; i<nidx; i++) {
    PetscCall(ISGetLocalSize(is[i],&lsize));
    max_lsize = lsize>max_lsize ? lsize:max_lsize;
    isz[i]    = lsize;
  }
  PetscCall(PetscMalloc2((max_lsize+nrecvs)*nidx,&indices_temp,nidx,&iscomms));
  for (i=0; i<nidx; i++) {
    PetscCall(PetscCommDuplicate(PetscObjectComm((PetscObject)is[i]),&iscomms[i],NULL));
    PetscCall(ISGetIndices(is[i],&indices_i_temp));
    PetscCall(PetscArraycpy(indices_temp+i*(max_lsize+nrecvs),indices_i_temp, isz[i]));
    PetscCall(ISRestoreIndices(is[i],&indices_i_temp));
    PetscCall(ISDestroy(&is[i]));
  }
  /* retrieve information to get row id and its overlap */
  for (i=0; i<nrecvs;) {
    is_id     = recvdata[i++];
    data_size = recvdata[i++];
    indices_i = indices_temp+(max_lsize+nrecvs)*is_id;
    isz_i     = isz[is_id];
    for (j=0; j< data_size; j++) {
      col = recvdata[i++];
      indices_i[isz_i++] = col;
    }
    isz[is_id] = isz_i;
  }
  /* remove duplicate entities */
  for (i=0; i<nidx; i++) {
    indices_i = indices_temp+(max_lsize+nrecvs)*i;
    isz_i     = isz[i];
    PetscCall(PetscSortRemoveDupsInt(&isz_i,indices_i));
    PetscCall(ISCreateGeneral(iscomms[i],isz_i,indices_i,PETSC_COPY_VALUES,&is[i]));
    PetscCall(PetscCommDestroy(&iscomms[i]));
  }
  PetscCall(PetscFree(isz));
  PetscCall(PetscFree2(indices_temp,iscomms));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatIncreaseOverlap_MPIAIJ_Send_Scalable(Mat mat,PetscInt nidx, PetscMPIInt nfrom,PetscMPIInt *fromranks,PetscInt *fromsizes, PetscInt *fromrows, PetscInt **sbrowsizes, PetscInt **sbrows)
{
  PetscLayout       rmap,cmap;
  PetscInt          i,j,k,l,*rows_i,*rows_data_ptr,**rows_data,max_fszs,rows_pos,*rows_pos_i;
  PetscInt          is_id,tnz,an,bn,rstart,cstart,row,start,end,col,totalrows,*sbdata;
  PetscInt         *indv_counts,indvc_ij,*sbsizes,*indices_tmp,*offsets;
  const PetscInt   *gcols,*ai,*aj,*bi,*bj;
  Mat               amat,bmat;
  PetscMPIInt       rank;
  PetscBool         done;
  MPI_Comm          comm;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)mat,&comm));
  PetscCallMPI(MPI_Comm_rank(comm,&rank));
  PetscCall(MatMPIAIJGetSeqAIJ(mat,&amat,&bmat,&gcols));
  /* Even if the mat is symmetric, we still assume it is not symmetric */
  PetscCall(MatGetRowIJ(amat,0,PETSC_FALSE,PETSC_FALSE,&an,&ai,&aj,&done));
  PetscCheck(done,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"can not get row IJ ");
  PetscCall(MatGetRowIJ(bmat,0,PETSC_FALSE,PETSC_FALSE,&bn,&bi,&bj,&done));
  PetscCheck(done,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"can not get row IJ ");
  /* total number of nonzero values is used to estimate the memory usage in the next step */
  tnz  = ai[an]+bi[bn];
  PetscCall(MatGetLayouts(mat,&rmap,&cmap));
  PetscCall(PetscLayoutGetRange(rmap,&rstart,NULL));
  PetscCall(PetscLayoutGetRange(cmap,&cstart,NULL));
  /* to find the longest message */
  max_fszs = 0;
  for (i=0; i<nfrom; i++) max_fszs = fromsizes[2*i]>max_fszs ? fromsizes[2*i]:max_fszs;
  /* better way to estimate number of nonzero in the mat??? */
  PetscCall(PetscCalloc5(max_fszs*nidx,&rows_data_ptr,nidx,&rows_data,nidx,&rows_pos_i,nfrom*nidx,&indv_counts,tnz,&indices_tmp));
  for (i=0; i<nidx; i++) rows_data[i] = rows_data_ptr+max_fszs*i;
  rows_pos  = 0;
  totalrows = 0;
  for (i=0; i<nfrom; i++) {
    PetscCall(PetscArrayzero(rows_pos_i,nidx));
    /* group data together */
    for (j=0; j<fromsizes[2*i]; j+=2) {
      is_id                       = fromrows[rows_pos++];/* no of is */
      rows_i                      = rows_data[is_id];
      rows_i[rows_pos_i[is_id]++] = fromrows[rows_pos++];/* row */
    }
    /* estimate a space to avoid multiple allocations  */
    for (j=0; j<nidx; j++) {
      indvc_ij = 0;
      rows_i   = rows_data[j];
      for (l=0; l<rows_pos_i[j]; l++) {
        row    = rows_i[l]-rstart;
        start  = ai[row];
        end    = ai[row+1];
        for (k=start; k<end; k++) { /* Amat */
          col = aj[k] + cstart;
          indices_tmp[indvc_ij++] = col;/* do not count the rows from the original rank */
        }
        start = bi[row];
        end   = bi[row+1];
        for (k=start; k<end; k++) { /* Bmat */
          col = gcols[bj[k]];
          indices_tmp[indvc_ij++] = col;
        }
      }
      PetscCall(PetscSortRemoveDupsInt(&indvc_ij,indices_tmp));
      indv_counts[i*nidx+j] = indvc_ij;
      totalrows            += indvc_ij;
    }
  }
  /* message triple <no of is, number of rows, rows> */
  PetscCall(PetscCalloc2(totalrows+nidx*nfrom*2,&sbdata,2*nfrom,&sbsizes));
  totalrows = 0;
  rows_pos  = 0;
  /* use this code again */
  for (i=0;i<nfrom;i++) {
    PetscCall(PetscArrayzero(rows_pos_i,nidx));
    for (j=0; j<fromsizes[2*i]; j+=2) {
      is_id                       = fromrows[rows_pos++];
      rows_i                      = rows_data[is_id];
      rows_i[rows_pos_i[is_id]++] = fromrows[rows_pos++];
    }
    /* add data  */
    for (j=0; j<nidx; j++) {
      if (!indv_counts[i*nidx+j]) continue;
      indvc_ij = 0;
      sbdata[totalrows++] = j;
      sbdata[totalrows++] = indv_counts[i*nidx+j];
      sbsizes[2*i]       += 2;
      rows_i              = rows_data[j];
      for (l=0; l<rows_pos_i[j]; l++) {
        row   = rows_i[l]-rstart;
        start = ai[row];
        end   = ai[row+1];
        for (k=start; k<end; k++) { /* Amat */
          col = aj[k] + cstart;
          indices_tmp[indvc_ij++] = col;
        }
        start = bi[row];
        end   = bi[row+1];
        for (k=start; k<end; k++) { /* Bmat */
          col = gcols[bj[k]];
          indices_tmp[indvc_ij++] = col;
        }
      }
      PetscCall(PetscSortRemoveDupsInt(&indvc_ij,indices_tmp));
      sbsizes[2*i]  += indvc_ij;
      PetscCall(PetscArraycpy(sbdata+totalrows,indices_tmp,indvc_ij));
      totalrows += indvc_ij;
    }
  }
  PetscCall(PetscMalloc1(nfrom+1,&offsets));
  offsets[0] = 0;
  for (i=0; i<nfrom; i++) {
    offsets[i+1]   = offsets[i] + sbsizes[2*i];
    sbsizes[2*i+1] = offsets[i];
  }
  PetscCall(PetscFree(offsets));
  if (sbrowsizes) *sbrowsizes = sbsizes;
  if (sbrows) *sbrows = sbdata;
  PetscCall(PetscFree5(rows_data_ptr,rows_data,rows_pos_i,indv_counts,indices_tmp));
  PetscCall(MatRestoreRowIJ(amat,0,PETSC_FALSE,PETSC_FALSE,&an,&ai,&aj,&done));
  PetscCall(MatRestoreRowIJ(bmat,0,PETSC_FALSE,PETSC_FALSE,&bn,&bi,&bj,&done));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatIncreaseOverlap_MPIAIJ_Local_Scalable(Mat mat,PetscInt nidx, IS is[])
{
  const PetscInt   *gcols,*ai,*aj,*bi,*bj, *indices;
  PetscInt          tnz,an,bn,i,j,row,start,end,rstart,cstart,col,k,*indices_temp;
  PetscInt          lsize,lsize_tmp;
  PetscMPIInt       rank,owner;
  Mat               amat,bmat;
  PetscBool         done;
  PetscLayout       cmap,rmap;
  MPI_Comm          comm;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)mat,&comm));
  PetscCallMPI(MPI_Comm_rank(comm,&rank));
  PetscCall(MatMPIAIJGetSeqAIJ(mat,&amat,&bmat,&gcols));
  PetscCall(MatGetRowIJ(amat,0,PETSC_FALSE,PETSC_FALSE,&an,&ai,&aj,&done));
  PetscCheck(done,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"can not get row IJ ");
  PetscCall(MatGetRowIJ(bmat,0,PETSC_FALSE,PETSC_FALSE,&bn,&bi,&bj,&done));
  PetscCheck(done,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"can not get row IJ ");
  /* is it a safe way to compute number of nonzero values ? */
  tnz  = ai[an]+bi[bn];
  PetscCall(MatGetLayouts(mat,&rmap,&cmap));
  PetscCall(PetscLayoutGetRange(rmap,&rstart,NULL));
  PetscCall(PetscLayoutGetRange(cmap,&cstart,NULL));
  /* it is a better way to estimate memory than the old implementation
   * where global size of matrix is used
   * */
  PetscCall(PetscMalloc1(tnz,&indices_temp));
  for (i=0; i<nidx; i++) {
    MPI_Comm iscomm;

    PetscCall(ISGetLocalSize(is[i],&lsize));
    PetscCall(ISGetIndices(is[i],&indices));
    lsize_tmp = 0;
    for (j=0; j<lsize; j++) {
      owner = -1;
      row   = indices[j];
      PetscCall(PetscLayoutFindOwner(rmap,row,&owner));
      if (owner != rank) continue;
      /* local number */
      row  -= rstart;
      start = ai[row];
      end   = ai[row+1];
      for (k=start; k<end; k++) { /* Amat */
        col = aj[k] + cstart;
        indices_temp[lsize_tmp++] = col;
      }
      start = bi[row];
      end   = bi[row+1];
      for (k=start; k<end; k++) { /* Bmat */
        col = gcols[bj[k]];
        indices_temp[lsize_tmp++] = col;
      }
    }
   PetscCall(ISRestoreIndices(is[i],&indices));
   PetscCall(PetscCommDuplicate(PetscObjectComm((PetscObject)is[i]),&iscomm,NULL));
   PetscCall(ISDestroy(&is[i]));
   PetscCall(PetscSortRemoveDupsInt(&lsize_tmp,indices_temp));
   PetscCall(ISCreateGeneral(iscomm,lsize_tmp,indices_temp,PETSC_COPY_VALUES,&is[i]));
   PetscCall(PetscCommDestroy(&iscomm));
  }
  PetscCall(PetscFree(indices_temp));
  PetscCall(MatRestoreRowIJ(amat,0,PETSC_FALSE,PETSC_FALSE,&an,&ai,&aj,&done));
  PetscCall(MatRestoreRowIJ(bmat,0,PETSC_FALSE,PETSC_FALSE,&bn,&bi,&bj,&done));
  PetscFunctionReturn(0);
}

/*
  Sample message format:
  If a processor A wants processor B to process some elements corresponding
  to index sets is[1],is[5]
  mesg [0] = 2   (no of index sets in the mesg)
  -----------
  mesg [1] = 1 => is[1]
  mesg [2] = sizeof(is[1]);
  -----------
  mesg [3] = 5  => is[5]
  mesg [4] = sizeof(is[5]);
  -----------
  mesg [5]
  mesg [n]  datas[1]
  -----------
  mesg[n+1]
  mesg[m]  data(is[5])
  -----------

  Notes:
  nrqs - no of requests sent (or to be sent out)
  nrqr - no of requests received (which have to be or which have been processed)
*/
static PetscErrorCode MatIncreaseOverlap_MPIAIJ_Once(Mat C,PetscInt imax,IS is[])
{
  Mat_MPIAIJ     *c = (Mat_MPIAIJ*)C->data;
  PetscMPIInt    *w1,*w2,nrqr,*w3,*w4,*onodes1,*olengths1,*onodes2,*olengths2;
  const PetscInt **idx,*idx_i;
  PetscInt       *n,**data,len;
#if defined(PETSC_USE_CTABLE)
  PetscTable     *table_data,table_data_i;
  PetscInt       *tdata,tcount,tcount_max;
#else
  PetscInt       *data_i,*d_p;
#endif
  PetscMPIInt    size,rank,tag1,tag2,proc = 0;
  PetscInt       M,i,j,k,**rbuf,row,nrqs,msz,**outdat,**ptr;
  PetscInt       *ctr,*pa,*tmp,*isz,*isz1,**xdata,**rbuf2;
  PetscBT        *table;
  MPI_Comm       comm;
  MPI_Request    *s_waits1,*r_waits1,*s_waits2,*r_waits2;
  MPI_Status     *recv_status;
  MPI_Comm       *iscomms;
  char           *t_p;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)C,&comm));
  size = c->size;
  rank = c->rank;
  M    = C->rmap->N;

  PetscCall(PetscObjectGetNewTag((PetscObject)C,&tag1));
  PetscCall(PetscObjectGetNewTag((PetscObject)C,&tag2));

  PetscCall(PetscMalloc2(imax,(PetscInt***)&idx,imax,&n));

  for (i=0; i<imax; i++) {
    PetscCall(ISGetIndices(is[i],&idx[i]));
    PetscCall(ISGetLocalSize(is[i],&n[i]));
  }

  /* evaluate communication - mesg to who,length of mesg, and buffer space
     required. Based on this, buffers are allocated, and data copied into them  */
  PetscCall(PetscCalloc4(size,&w1,size,&w2,size,&w3,size,&w4));
  for (i=0; i<imax; i++) {
    PetscCall(PetscArrayzero(w4,size)); /* initialise work vector*/
    idx_i = idx[i];
    len   = n[i];
    for (j=0; j<len; j++) {
      row = idx_i[j];
      PetscCheckFalse(row < 0,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Index set cannot have negative entries");
      PetscCall(PetscLayoutFindOwner(C->rmap,row,&proc));
      w4[proc]++;
    }
    for (j=0; j<size; j++) {
      if (w4[j]) { w1[j] += w4[j]; w3[j]++;}
    }
  }

  nrqs     = 0;              /* no of outgoing messages */
  msz      = 0;              /* total mesg length (for all proc */
  w1[rank] = 0;              /* no mesg sent to intself */
  w3[rank] = 0;
  for (i=0; i<size; i++) {
    if (w1[i])  {w2[i] = 1; nrqs++;} /* there exists a message to proc i */
  }
  /* pa - is list of processors to communicate with */
  PetscCall(PetscMalloc1(nrqs,&pa));
  for (i=0,j=0; i<size; i++) {
    if (w1[i]) {pa[j] = i; j++;}
  }

  /* Each message would have a header = 1 + 2*(no of IS) + data */
  for (i=0; i<nrqs; i++) {
    j      = pa[i];
    w1[j] += w2[j] + 2*w3[j];
    msz   += w1[j];
  }

  /* Determine the number of messages to expect, their lengths, from from-ids */
  PetscCall(PetscGatherNumberOfMessages(comm,w2,w1,&nrqr));
  PetscCall(PetscGatherMessageLengths(comm,nrqs,nrqr,w1,&onodes1,&olengths1));

  /* Now post the Irecvs corresponding to these messages */
  PetscCall(PetscPostIrecvInt(comm,tag1,nrqr,onodes1,olengths1,&rbuf,&r_waits1));

  /* Allocate Memory for outgoing messages */
  PetscCall(PetscMalloc4(size,&outdat,size,&ptr,msz,&tmp,size,&ctr));
  PetscCall(PetscArrayzero(outdat,size));
  PetscCall(PetscArrayzero(ptr,size));

  {
    PetscInt *iptr = tmp,ict  = 0;
    for (i=0; i<nrqs; i++) {
      j         = pa[i];
      iptr     +=  ict;
      outdat[j] = iptr;
      ict       = w1[j];
    }
  }

  /* Form the outgoing messages */
  /* plug in the headers */
  for (i=0; i<nrqs; i++) {
    j            = pa[i];
    outdat[j][0] = 0;
    PetscCall(PetscArrayzero(outdat[j]+1,2*w3[j]));
    ptr[j]       = outdat[j] + 2*w3[j] + 1;
  }

  /* Memory for doing local proc's work */
  {
    PetscInt M_BPB_imax = 0;
#if defined(PETSC_USE_CTABLE)
    PetscCall(PetscIntMultError((M/PETSC_BITS_PER_BYTE+1),imax, &M_BPB_imax));
    PetscCall(PetscMalloc1(imax,&table_data));
    for (i=0; i<imax; i++) {
      PetscCall(PetscTableCreate(n[i],M,&table_data[i]));
    }
    PetscCall(PetscCalloc4(imax,&table, imax,&data, imax,&isz, M_BPB_imax,&t_p));
    for (i=0; i<imax; i++) {
      table[i] = t_p + (M/PETSC_BITS_PER_BYTE+1)*i;
    }
#else
    PetscInt Mimax = 0;
    PetscCall(PetscIntMultError(M,imax, &Mimax));
    PetscCall(PetscIntMultError((M/PETSC_BITS_PER_BYTE+1),imax, &M_BPB_imax));
    PetscCall(PetscCalloc5(imax,&table, imax,&data, imax,&isz, Mimax,&d_p, M_BPB_imax,&t_p));
    for (i=0; i<imax; i++) {
      table[i] = t_p + (M/PETSC_BITS_PER_BYTE+1)*i;
      data[i]  = d_p + M*i;
    }
#endif
  }

  /* Parse the IS and update local tables and the outgoing buf with the data */
  {
    PetscInt n_i,isz_i,*outdat_j,ctr_j;
    PetscBT  table_i;

    for (i=0; i<imax; i++) {
      PetscCall(PetscArrayzero(ctr,size));
      n_i     = n[i];
      table_i = table[i];
      idx_i   = idx[i];
#if defined(PETSC_USE_CTABLE)
      table_data_i = table_data[i];
#else
      data_i  = data[i];
#endif
      isz_i   = isz[i];
      for (j=0; j<n_i; j++) {   /* parse the indices of each IS */
        row  = idx_i[j];
        PetscCall(PetscLayoutFindOwner(C->rmap,row,&proc));
        if (proc != rank) { /* copy to the outgoing buffer */
          ctr[proc]++;
          *ptr[proc] = row;
          ptr[proc]++;
        } else if (!PetscBTLookupSet(table_i,row)) {
#if defined(PETSC_USE_CTABLE)
          PetscCall(PetscTableAdd(table_data_i,row+1,isz_i+1,INSERT_VALUES));
#else
          data_i[isz_i] = row; /* Update the local table */
#endif
          isz_i++;
        }
      }
      /* Update the headers for the current IS */
      for (j=0; j<size; j++) { /* Can Optimise this loop by using pa[] */
        if ((ctr_j = ctr[j])) {
          outdat_j        = outdat[j];
          k               = ++outdat_j[0];
          outdat_j[2*k]   = ctr_j;
          outdat_j[2*k-1] = i;
        }
      }
      isz[i] = isz_i;
    }
  }

  /*  Now  post the sends */
  PetscCall(PetscMalloc1(nrqs,&s_waits1));
  for (i=0; i<nrqs; ++i) {
    j    = pa[i];
    PetscCallMPI(MPI_Isend(outdat[j],w1[j],MPIU_INT,j,tag1,comm,s_waits1+i));
  }

  /* No longer need the original indices */
  PetscCall(PetscMalloc1(imax,&iscomms));
  for (i=0; i<imax; ++i) {
    PetscCall(ISRestoreIndices(is[i],idx+i));
    PetscCall(PetscCommDuplicate(PetscObjectComm((PetscObject)is[i]),&iscomms[i],NULL));
  }
  PetscCall(PetscFree2(*(PetscInt***)&idx,n));

  for (i=0; i<imax; ++i) {
    PetscCall(ISDestroy(&is[i]));
  }

  /* Do Local work */
#if defined(PETSC_USE_CTABLE)
  PetscCall(MatIncreaseOverlap_MPIAIJ_Local(C,imax,table,isz,NULL,table_data));
#else
  PetscCall(MatIncreaseOverlap_MPIAIJ_Local(C,imax,table,isz,data,NULL));
#endif

  /* Receive messages */
  PetscCall(PetscMalloc1(nrqr,&recv_status));
  PetscCallMPI(MPI_Waitall(nrqr,r_waits1,recv_status));
  PetscCallMPI(MPI_Waitall(nrqs,s_waits1,MPI_STATUSES_IGNORE));

  /* Phase 1 sends are complete - deallocate buffers */
  PetscCall(PetscFree4(outdat,ptr,tmp,ctr));
  PetscCall(PetscFree4(w1,w2,w3,w4));

  PetscCall(PetscMalloc1(nrqr,&xdata));
  PetscCall(PetscMalloc1(nrqr,&isz1));
  PetscCall(MatIncreaseOverlap_MPIAIJ_Receive(C,nrqr,rbuf,xdata,isz1));
  PetscCall(PetscFree(rbuf[0]));
  PetscCall(PetscFree(rbuf));

  /* Send the data back */
  /* Do a global reduction to know the buffer space req for incoming messages */
  {
    PetscMPIInt *rw1;

    PetscCall(PetscCalloc1(size,&rw1));

    for (i=0; i<nrqr; ++i) {
      proc = recv_status[i].MPI_SOURCE;

      PetscCheckFalse(proc != onodes1[i],PETSC_COMM_SELF,PETSC_ERR_PLIB,"MPI_SOURCE mismatch");
      rw1[proc] = isz1[i];
    }
    PetscCall(PetscFree(onodes1));
    PetscCall(PetscFree(olengths1));

    /* Determine the number of messages to expect, their lengths, from from-ids */
    PetscCall(PetscGatherMessageLengths(comm,nrqr,nrqs,rw1,&onodes2,&olengths2));
    PetscCall(PetscFree(rw1));
  }
  /* Now post the Irecvs corresponding to these messages */
  PetscCall(PetscPostIrecvInt(comm,tag2,nrqs,onodes2,olengths2,&rbuf2,&r_waits2));

  /* Now  post the sends */
  PetscCall(PetscMalloc1(nrqr,&s_waits2));
  for (i=0; i<nrqr; ++i) {
    j    = recv_status[i].MPI_SOURCE;
    PetscCallMPI(MPI_Isend(xdata[i],isz1[i],MPIU_INT,j,tag2,comm,s_waits2+i));
  }

  /* receive work done on other processors */
  {
    PetscInt    is_no,ct1,max,*rbuf2_i,isz_i,jmax;
    PetscMPIInt idex;
    PetscBT     table_i;

    for (i=0; i<nrqs; ++i) {
      PetscCallMPI(MPI_Waitany(nrqs,r_waits2,&idex,MPI_STATUS_IGNORE));
      /* Process the message */
      rbuf2_i = rbuf2[idex];
      ct1     = 2*rbuf2_i[0]+1;
      jmax    = rbuf2[idex][0];
      for (j=1; j<=jmax; j++) {
        max     = rbuf2_i[2*j];
        is_no   = rbuf2_i[2*j-1];
        isz_i   = isz[is_no];
        table_i = table[is_no];
#if defined(PETSC_USE_CTABLE)
        table_data_i = table_data[is_no];
#else
        data_i  = data[is_no];
#endif
        for (k=0; k<max; k++,ct1++) {
          row = rbuf2_i[ct1];
          if (!PetscBTLookupSet(table_i,row)) {
#if defined(PETSC_USE_CTABLE)
            PetscCall(PetscTableAdd(table_data_i,row+1,isz_i+1,INSERT_VALUES));
#else
            data_i[isz_i] = row;
#endif
            isz_i++;
          }
        }
        isz[is_no] = isz_i;
      }
    }

    PetscCallMPI(MPI_Waitall(nrqr,s_waits2,MPI_STATUSES_IGNORE));
  }

#if defined(PETSC_USE_CTABLE)
  tcount_max = 0;
  for (i=0; i<imax; ++i) {
    table_data_i = table_data[i];
    PetscCall(PetscTableGetCount(table_data_i,&tcount));
    if (tcount_max < tcount) tcount_max = tcount;
  }
  PetscCall(PetscMalloc1(tcount_max+1,&tdata));
#endif

  for (i=0; i<imax; ++i) {
#if defined(PETSC_USE_CTABLE)
    PetscTablePosition tpos;
    table_data_i = table_data[i];

    PetscCall(PetscTableGetHeadPosition(table_data_i,&tpos));
    while (tpos) {
      PetscCall(PetscTableGetNext(table_data_i,&tpos,&k,&j));
      tdata[--j] = --k;
    }
    PetscCall(ISCreateGeneral(iscomms[i],isz[i],tdata,PETSC_COPY_VALUES,is+i));
#else
    PetscCall(ISCreateGeneral(iscomms[i],isz[i],data[i],PETSC_COPY_VALUES,is+i));
#endif
    PetscCall(PetscCommDestroy(&iscomms[i]));
  }

  PetscCall(PetscFree(iscomms));
  PetscCall(PetscFree(onodes2));
  PetscCall(PetscFree(olengths2));

  PetscCall(PetscFree(pa));
  PetscCall(PetscFree(rbuf2[0]));
  PetscCall(PetscFree(rbuf2));
  PetscCall(PetscFree(s_waits1));
  PetscCall(PetscFree(r_waits1));
  PetscCall(PetscFree(s_waits2));
  PetscCall(PetscFree(r_waits2));
  PetscCall(PetscFree(recv_status));
  if (xdata) {
    PetscCall(PetscFree(xdata[0]));
    PetscCall(PetscFree(xdata));
  }
  PetscCall(PetscFree(isz1));
#if defined(PETSC_USE_CTABLE)
  for (i=0; i<imax; i++) {
    PetscCall(PetscTableDestroy((PetscTable*)&table_data[i]));
  }
  PetscCall(PetscFree(table_data));
  PetscCall(PetscFree(tdata));
  PetscCall(PetscFree4(table,data,isz,t_p));
#else
  PetscCall(PetscFree5(table,data,isz,d_p,t_p));
#endif
  PetscFunctionReturn(0);
}

/*
   MatIncreaseOverlap_MPIAIJ_Local - Called by MatincreaseOverlap, to do
       the work on the local processor.

     Inputs:
      C      - MAT_MPIAIJ;
      imax - total no of index sets processed at a time;
      table  - an array of char - size = m bits.

     Output:
      isz    - array containing the count of the solution elements corresponding
               to each index set;
      data or table_data  - pointer to the solutions
*/
static PetscErrorCode MatIncreaseOverlap_MPIAIJ_Local(Mat C,PetscInt imax,PetscBT *table,PetscInt *isz,PetscInt **data,PetscTable *table_data)
{
  Mat_MPIAIJ *c = (Mat_MPIAIJ*)C->data;
  Mat        A  = c->A,B = c->B;
  Mat_SeqAIJ *a = (Mat_SeqAIJ*)A->data,*b = (Mat_SeqAIJ*)B->data;
  PetscInt   start,end,val,max,rstart,cstart,*ai,*aj;
  PetscInt   *bi,*bj,*garray,i,j,k,row,isz_i;
  PetscBT    table_i;
#if defined(PETSC_USE_CTABLE)
  PetscTable         table_data_i;
  PetscTablePosition tpos;
  PetscInt           tcount,*tdata;
#else
  PetscInt           *data_i;
#endif

  PetscFunctionBegin;
  rstart = C->rmap->rstart;
  cstart = C->cmap->rstart;
  ai     = a->i;
  aj     = a->j;
  bi     = b->i;
  bj     = b->j;
  garray = c->garray;

  for (i=0; i<imax; i++) {
#if defined(PETSC_USE_CTABLE)
    /* copy existing entries of table_data_i into tdata[] */
    table_data_i = table_data[i];
    PetscCall(PetscTableGetCount(table_data_i,&tcount));
    PetscCheckFalse(tcount != isz[i],PETSC_COMM_SELF,PETSC_ERR_PLIB," tcount %" PetscInt_FMT " != isz[%" PetscInt_FMT "] %" PetscInt_FMT,tcount,i,isz[i]);

    PetscCall(PetscMalloc1(tcount,&tdata));
    PetscCall(PetscTableGetHeadPosition(table_data_i,&tpos));
    while (tpos) {
      PetscCall(PetscTableGetNext(table_data_i,&tpos,&row,&j));
      tdata[--j] = --row;
      PetscCheckFalse(j > tcount - 1,PETSC_COMM_SELF,PETSC_ERR_PLIB," j %" PetscInt_FMT " >= tcount %" PetscInt_FMT,j,tcount);
    }
#else
    data_i  = data[i];
#endif
    table_i = table[i];
    isz_i   = isz[i];
    max     = isz[i];

    for (j=0; j<max; j++) {
#if defined(PETSC_USE_CTABLE)
      row   = tdata[j] - rstart;
#else
      row   = data_i[j] - rstart;
#endif
      start = ai[row];
      end   = ai[row+1];
      for (k=start; k<end; k++) { /* Amat */
        val = aj[k] + cstart;
        if (!PetscBTLookupSet(table_i,val)) {
#if defined(PETSC_USE_CTABLE)
          PetscCall(PetscTableAdd(table_data_i,val+1,isz_i+1,INSERT_VALUES));
#else
          data_i[isz_i] = val;
#endif
          isz_i++;
        }
      }
      start = bi[row];
      end   = bi[row+1];
      for (k=start; k<end; k++) { /* Bmat */
        val = garray[bj[k]];
        if (!PetscBTLookupSet(table_i,val)) {
#if defined(PETSC_USE_CTABLE)
          PetscCall(PetscTableAdd(table_data_i,val+1,isz_i+1,INSERT_VALUES));
#else
          data_i[isz_i] = val;
#endif
          isz_i++;
        }
      }
    }
    isz[i] = isz_i;

#if defined(PETSC_USE_CTABLE)
    PetscCall(PetscFree(tdata));
#endif
  }
  PetscFunctionReturn(0);
}

/*
      MatIncreaseOverlap_MPIAIJ_Receive - Process the received messages,
         and return the output

         Input:
           C    - the matrix
           nrqr - no of messages being processed.
           rbuf - an array of pointers to the received requests

         Output:
           xdata - array of messages to be sent back
           isz1  - size of each message

  For better efficiency perhaps we should malloc separately each xdata[i],
then if a remalloc is required we need only copy the data for that one row
rather then all previous rows as it is now where a single large chunk of
memory is used.

*/
static PetscErrorCode MatIncreaseOverlap_MPIAIJ_Receive(Mat C,PetscInt nrqr,PetscInt **rbuf,PetscInt **xdata,PetscInt * isz1)
{
  Mat_MPIAIJ     *c = (Mat_MPIAIJ*)C->data;
  Mat            A  = c->A,B = c->B;
  Mat_SeqAIJ     *a = (Mat_SeqAIJ*)A->data,*b = (Mat_SeqAIJ*)B->data;
  PetscInt       rstart,cstart,*ai,*aj,*bi,*bj,*garray,i,j,k;
  PetscInt       row,total_sz,ct,ct1,ct2,ct3,mem_estimate,oct2,l,start,end;
  PetscInt       val,max1,max2,m,no_malloc =0,*tmp,new_estimate,ctr;
  PetscInt       *rbuf_i,kmax,rbuf_0;
  PetscBT        xtable;

  PetscFunctionBegin;
  m      = C->rmap->N;
  rstart = C->rmap->rstart;
  cstart = C->cmap->rstart;
  ai     = a->i;
  aj     = a->j;
  bi     = b->i;
  bj     = b->j;
  garray = c->garray;

  for (i=0,ct=0,total_sz=0; i<nrqr; ++i) {
    rbuf_i =  rbuf[i];
    rbuf_0 =  rbuf_i[0];
    ct    += rbuf_0;
    for (j=1; j<=rbuf_0; j++) total_sz += rbuf_i[2*j];
  }

  if (C->rmap->n) max1 = ct*(a->nz + b->nz)/C->rmap->n;
  else max1 = 1;
  mem_estimate = 3*((total_sz > max1 ? total_sz : max1)+1);
  if (nrqr) {
    PetscCall(PetscMalloc1(mem_estimate,&xdata[0]));
    ++no_malloc;
  }
  PetscCall(PetscBTCreate(m,&xtable));
  PetscCall(PetscArrayzero(isz1,nrqr));

  ct3 = 0;
  for (i=0; i<nrqr; i++) { /* for easch mesg from proc i */
    rbuf_i =  rbuf[i];
    rbuf_0 =  rbuf_i[0];
    ct1    =  2*rbuf_0+1;
    ct2    =  ct1;
    ct3   += ct1;
    for (j=1; j<=rbuf_0; j++) { /* for each IS from proc i*/
      PetscCall(PetscBTMemzero(m,xtable));
      oct2 = ct2;
      kmax = rbuf_i[2*j];
      for (k=0; k<kmax; k++,ct1++) {
        row = rbuf_i[ct1];
        if (!PetscBTLookupSet(xtable,row)) {
          if (!(ct3 < mem_estimate)) {
            new_estimate = (PetscInt)(1.5*mem_estimate)+1;
            PetscCall(PetscMalloc1(new_estimate,&tmp));
            PetscCall(PetscArraycpy(tmp,xdata[0],mem_estimate));
            PetscCall(PetscFree(xdata[0]));
            xdata[0]     = tmp;
            mem_estimate = new_estimate; ++no_malloc;
            for (ctr=1; ctr<=i; ctr++) xdata[ctr] = xdata[ctr-1] + isz1[ctr-1];
          }
          xdata[i][ct2++] = row;
          ct3++;
        }
      }
      for (k=oct2,max2=ct2; k<max2; k++) {
        row   = xdata[i][k] - rstart;
        start = ai[row];
        end   = ai[row+1];
        for (l=start; l<end; l++) {
          val = aj[l] + cstart;
          if (!PetscBTLookupSet(xtable,val)) {
            if (!(ct3 < mem_estimate)) {
              new_estimate = (PetscInt)(1.5*mem_estimate)+1;
              PetscCall(PetscMalloc1(new_estimate,&tmp));
              PetscCall(PetscArraycpy(tmp,xdata[0],mem_estimate));
              PetscCall(PetscFree(xdata[0]));
              xdata[0]     = tmp;
              mem_estimate = new_estimate; ++no_malloc;
              for (ctr=1; ctr<=i; ctr++) xdata[ctr] = xdata[ctr-1] + isz1[ctr-1];
            }
            xdata[i][ct2++] = val;
            ct3++;
          }
        }
        start = bi[row];
        end   = bi[row+1];
        for (l=start; l<end; l++) {
          val = garray[bj[l]];
          if (!PetscBTLookupSet(xtable,val)) {
            if (!(ct3 < mem_estimate)) {
              new_estimate = (PetscInt)(1.5*mem_estimate)+1;
              PetscCall(PetscMalloc1(new_estimate,&tmp));
              PetscCall(PetscArraycpy(tmp,xdata[0],mem_estimate));
              PetscCall(PetscFree(xdata[0]));
              xdata[0]     = tmp;
              mem_estimate = new_estimate; ++no_malloc;
              for (ctr =1; ctr <=i; ctr++) xdata[ctr] = xdata[ctr-1] + isz1[ctr-1];
            }
            xdata[i][ct2++] = val;
            ct3++;
          }
        }
      }
      /* Update the header*/
      xdata[i][2*j]   = ct2 - oct2; /* Undo the vector isz1 and use only a var*/
      xdata[i][2*j-1] = rbuf_i[2*j-1];
    }
    xdata[i][0] = rbuf_0;
    if (i+1<nrqr) xdata[i+1]  = xdata[i] + ct2;
    isz1[i]     = ct2; /* size of each message */
  }
  PetscCall(PetscBTDestroy(&xtable));
  PetscCall(PetscInfo(C,"Allocated %" PetscInt_FMT " bytes, required %" PetscInt_FMT " bytes, no of mallocs = %" PetscInt_FMT "\n",mem_estimate,ct3,no_malloc));
  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------*/
extern PetscErrorCode MatCreateSubMatrices_MPIAIJ_Local(Mat,PetscInt,const IS[],const IS[],MatReuse,Mat*);
/*
    Every processor gets the entire matrix
*/
PetscErrorCode MatCreateSubMatrix_MPIAIJ_All(Mat A,MatCreateSubMatrixOption flag,MatReuse scall,Mat *Bin[])
{
  Mat            B;
  Mat_MPIAIJ     *a = (Mat_MPIAIJ*)A->data;
  Mat_SeqAIJ     *b,*ad = (Mat_SeqAIJ*)a->A->data,*bd = (Mat_SeqAIJ*)a->B->data;
  PetscMPIInt    size,rank,*recvcounts = NULL,*displs = NULL;
  PetscInt       sendcount,i,*rstarts = A->rmap->range,n,cnt,j;
  PetscInt       m,*b_sendj,*garray = a->garray,*lens,*jsendbuf,*a_jsendbuf,*b_jsendbuf;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)A),&size));
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)A),&rank));
  if (scall == MAT_INITIAL_MATRIX) {
    /* ----------------------------------------------------------------
         Tell every processor the number of nonzeros per row
    */
    PetscCall(PetscMalloc1(A->rmap->N,&lens));
    for (i=A->rmap->rstart; i<A->rmap->rend; i++) {
      lens[i] = ad->i[i-A->rmap->rstart+1] - ad->i[i-A->rmap->rstart] + bd->i[i-A->rmap->rstart+1] - bd->i[i-A->rmap->rstart];
    }
    PetscCall(PetscMalloc2(size,&recvcounts,size,&displs));
    for (i=0; i<size; i++) {
      recvcounts[i] = A->rmap->range[i+1] - A->rmap->range[i];
      displs[i]     = A->rmap->range[i];
    }
    PetscCallMPI(MPI_Allgatherv(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,lens,recvcounts,displs,MPIU_INT,PetscObjectComm((PetscObject)A)));
    /* ---------------------------------------------------------------
         Create the sequential matrix of the same type as the local block diagonal
    */
    PetscCall(MatCreate(PETSC_COMM_SELF,&B));
    PetscCall(MatSetSizes(B,A->rmap->N,A->cmap->N,PETSC_DETERMINE,PETSC_DETERMINE));
    PetscCall(MatSetBlockSizesFromMats(B,A,A));
    PetscCall(MatSetType(B,((PetscObject)a->A)->type_name));
    PetscCall(MatSeqAIJSetPreallocation(B,0,lens));
    PetscCall(PetscCalloc1(2,Bin));
    **Bin = B;
    b     = (Mat_SeqAIJ*)B->data;

    /*--------------------------------------------------------------------
       Copy my part of matrix column indices over
    */
    sendcount  = ad->nz + bd->nz;
    jsendbuf   = b->j + b->i[rstarts[rank]];
    a_jsendbuf = ad->j;
    b_jsendbuf = bd->j;
    n          = A->rmap->rend - A->rmap->rstart;
    cnt        = 0;
    for (i=0; i<n; i++) {
      /* put in lower diagonal portion */
      m = bd->i[i+1] - bd->i[i];
      while (m > 0) {
        /* is it above diagonal (in bd (compressed) numbering) */
        if (garray[*b_jsendbuf] > A->rmap->rstart + i) break;
        jsendbuf[cnt++] = garray[*b_jsendbuf++];
        m--;
      }

      /* put in diagonal portion */
      for (j=ad->i[i]; j<ad->i[i+1]; j++) {
        jsendbuf[cnt++] = A->rmap->rstart + *a_jsendbuf++;
      }

      /* put in upper diagonal portion */
      while (m-- > 0) {
        jsendbuf[cnt++] = garray[*b_jsendbuf++];
      }
    }
    PetscCheckFalse(cnt != sendcount,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Corrupted PETSc matrix: nz given %" PetscInt_FMT " actual nz %" PetscInt_FMT,sendcount,cnt);

    /*--------------------------------------------------------------------
       Gather all column indices to all processors
    */
    for (i=0; i<size; i++) {
      recvcounts[i] = 0;
      for (j=A->rmap->range[i]; j<A->rmap->range[i+1]; j++) {
        recvcounts[i] += lens[j];
      }
    }
    displs[0] = 0;
    for (i=1; i<size; i++) {
      displs[i] = displs[i-1] + recvcounts[i-1];
    }
    PetscCallMPI(MPI_Allgatherv(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,b->j,recvcounts,displs,MPIU_INT,PetscObjectComm((PetscObject)A)));
    /*--------------------------------------------------------------------
        Assemble the matrix into useable form (note numerical values not yet set)
    */
    /* set the b->ilen (length of each row) values */
    PetscCall(PetscArraycpy(b->ilen,lens,A->rmap->N));
    /* set the b->i indices */
    b->i[0] = 0;
    for (i=1; i<=A->rmap->N; i++) {
      b->i[i] = b->i[i-1] + lens[i-1];
    }
    PetscCall(PetscFree(lens));
    PetscCall(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));

  } else {
    B = **Bin;
    b = (Mat_SeqAIJ*)B->data;
  }

  /*--------------------------------------------------------------------
       Copy my part of matrix numerical values into the values location
  */
  if (flag == MAT_GET_VALUES) {
    const PetscScalar *ada,*bda,*a_sendbuf,*b_sendbuf;
    MatScalar         *sendbuf,*recvbuf;

    PetscCall(MatSeqAIJGetArrayRead(a->A,&ada));
    PetscCall(MatSeqAIJGetArrayRead(a->B,&bda));
    sendcount = ad->nz + bd->nz;
    sendbuf   = b->a + b->i[rstarts[rank]];
    a_sendbuf = ada;
    b_sendbuf = bda;
    b_sendj   = bd->j;
    n         = A->rmap->rend - A->rmap->rstart;
    cnt       = 0;
    for (i=0; i<n; i++) {
      /* put in lower diagonal portion */
      m = bd->i[i+1] - bd->i[i];
      while (m > 0) {
        /* is it above diagonal (in bd (compressed) numbering) */
        if (garray[*b_sendj] > A->rmap->rstart + i) break;
        sendbuf[cnt++] = *b_sendbuf++;
        m--;
        b_sendj++;
      }

      /* put in diagonal portion */
      for (j=ad->i[i]; j<ad->i[i+1]; j++) {
        sendbuf[cnt++] = *a_sendbuf++;
      }

      /* put in upper diagonal portion */
      while (m-- > 0) {
        sendbuf[cnt++] = *b_sendbuf++;
        b_sendj++;
      }
    }
    PetscCheckFalse(cnt != sendcount,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Corrupted PETSc matrix: nz given %" PetscInt_FMT " actual nz %" PetscInt_FMT,sendcount,cnt);

    /* -----------------------------------------------------------------
       Gather all numerical values to all processors
    */
    if (!recvcounts) {
      PetscCall(PetscMalloc2(size,&recvcounts,size,&displs));
    }
    for (i=0; i<size; i++) {
      recvcounts[i] = b->i[rstarts[i+1]] - b->i[rstarts[i]];
    }
    displs[0] = 0;
    for (i=1; i<size; i++) {
      displs[i] = displs[i-1] + recvcounts[i-1];
    }
    recvbuf = b->a;
    PetscCallMPI(MPI_Allgatherv(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,recvbuf,recvcounts,displs,MPIU_SCALAR,PetscObjectComm((PetscObject)A)));
    PetscCall(MatSeqAIJRestoreArrayRead(a->A,&ada));
    PetscCall(MatSeqAIJRestoreArrayRead(a->B,&bda));
  }  /* endof (flag == MAT_GET_VALUES) */
  PetscCall(PetscFree2(recvcounts,displs));

  if (A->symmetric) {
    PetscCall(MatSetOption(B,MAT_SYMMETRIC,PETSC_TRUE));
  } else if (A->hermitian) {
    PetscCall(MatSetOption(B,MAT_HERMITIAN,PETSC_TRUE));
  } else if (A->structurally_symmetric) {
    PetscCall(MatSetOption(B,MAT_STRUCTURALLY_SYMMETRIC,PETSC_TRUE));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatCreateSubMatrices_MPIAIJ_SingleIS_Local(Mat C,PetscInt ismax,const IS isrow[],const IS iscol[],MatReuse scall,PetscBool allcolumns,Mat *submats)
{
  Mat_MPIAIJ     *c = (Mat_MPIAIJ*)C->data;
  Mat            submat,A = c->A,B = c->B;
  Mat_SeqAIJ     *a = (Mat_SeqAIJ*)A->data,*b = (Mat_SeqAIJ*)B->data,*subc;
  PetscInt       *ai=a->i,*aj=a->j,*bi=b->i,*bj=b->j,nzA,nzB;
  PetscInt       cstart = C->cmap->rstart,cend = C->cmap->rend,rstart = C->rmap->rstart,*bmap = c->garray;
  const PetscInt *icol,*irow;
  PetscInt       nrow,ncol,start;
  PetscMPIInt    rank,size,tag1,tag2,tag3,tag4,*w1,*w2,nrqr;
  PetscInt       **sbuf1,**sbuf2,i,j,k,l,ct1,ct2,ct3,**rbuf1,row,proc;
  PetscInt       nrqs=0,msz,**ptr,*req_size,*ctr,*pa,*tmp,tcol,*iptr;
  PetscInt       **rbuf3,*req_source1,*req_source2,**sbuf_aj,**rbuf2,max1,nnz;
  PetscInt       *lens,rmax,ncols,*cols,Crow;
#if defined(PETSC_USE_CTABLE)
  PetscTable     cmap,rmap;
  PetscInt       *cmap_loc,*rmap_loc;
#else
  PetscInt       *cmap,*rmap;
#endif
  PetscInt       ctr_j,*sbuf1_j,*sbuf_aj_i,*rbuf1_i,kmax,*sbuf1_i,*rbuf2_i,*rbuf3_i;
  PetscInt       *cworkB,lwrite,*subcols,*row2proc;
  PetscScalar    *vworkA,*vworkB,*a_a,*b_a,*subvals=NULL;
  MPI_Request    *s_waits1,*r_waits1,*s_waits2,*r_waits2,*r_waits3;
  MPI_Request    *r_waits4,*s_waits3 = NULL,*s_waits4;
  MPI_Status     *r_status1,*r_status2,*s_status1,*s_status3 = NULL,*s_status2;
  MPI_Status     *r_status3 = NULL,*r_status4,*s_status4;
  MPI_Comm       comm;
  PetscScalar    **rbuf4,**sbuf_aa,*vals,*sbuf_aa_i,*rbuf4_i;
  PetscMPIInt    *onodes1,*olengths1,idex,end;
  Mat_SubSppt    *smatis1;
  PetscBool      isrowsorted,iscolsorted;

  PetscFunctionBegin;
  PetscValidLogicalCollectiveInt(C,ismax,2);
  PetscValidLogicalCollectiveEnum(C,scall,5);
  PetscCheckFalse(ismax != 1,PETSC_COMM_SELF,PETSC_ERR_SUP,"This routine only works when all processes have ismax=1");
  PetscCall(MatSeqAIJGetArrayRead(A,(const PetscScalar**)&a_a));
  PetscCall(MatSeqAIJGetArrayRead(B,(const PetscScalar**)&b_a));
  PetscCall(PetscObjectGetComm((PetscObject)C,&comm));
  size = c->size;
  rank = c->rank;

  PetscCall(ISSorted(iscol[0],&iscolsorted));
  PetscCall(ISSorted(isrow[0],&isrowsorted));
  PetscCall(ISGetIndices(isrow[0],&irow));
  PetscCall(ISGetLocalSize(isrow[0],&nrow));
  if (allcolumns) {
    icol = NULL;
    ncol = C->cmap->N;
  } else {
    PetscCall(ISGetIndices(iscol[0],&icol));
    PetscCall(ISGetLocalSize(iscol[0],&ncol));
  }

  if (scall == MAT_INITIAL_MATRIX) {
    PetscInt *sbuf2_i,*cworkA,lwrite,ctmp;

    /* Get some new tags to keep the communication clean */
    tag1 = ((PetscObject)C)->tag;
    PetscCall(PetscObjectGetNewTag((PetscObject)C,&tag2));
    PetscCall(PetscObjectGetNewTag((PetscObject)C,&tag3));

    /* evaluate communication - mesg to who, length of mesg, and buffer space
     required. Based on this, buffers are allocated, and data copied into them */
    PetscCall(PetscCalloc2(size,&w1,size,&w2));
    PetscCall(PetscMalloc1(nrow,&row2proc));

    /* w1[proc] = num of rows owned by proc -- to be requested */
    proc = 0;
    nrqs = 0; /* num of outgoing messages */
    for (j=0; j<nrow; j++) {
      row  = irow[j];
      if (!isrowsorted) proc = 0;
      while (row >= C->rmap->range[proc+1]) proc++;
      w1[proc]++;
      row2proc[j] = proc; /* map row index to proc */

      if (proc != rank && !w2[proc]) {
        w2[proc] = 1; nrqs++;
      }
    }
    w1[rank] = 0;  /* rows owned by self will not be requested */

    PetscCall(PetscMalloc1(nrqs,&pa)); /*(proc -array)*/
    for (proc=0,j=0; proc<size; proc++) {
      if (w1[proc]) { pa[j++] = proc;}
    }

    /* Each message would have a header = 1 + 2*(num of IS) + data (here,num of IS = 1) */
    msz = 0;              /* total mesg length (for all procs) */
    for (i=0; i<nrqs; i++) {
      proc      = pa[i];
      w1[proc] += 3;
      msz      += w1[proc];
    }
    PetscCall(PetscInfo(0,"Number of outgoing messages %" PetscInt_FMT " Total message length %" PetscInt_FMT "\n",nrqs,msz));

    /* Determine nrqr, the number of messages to expect, their lengths, from from-ids */
    /* if w2[proc]=1, a message of length w1[proc] will be sent to proc; */
    PetscCall(PetscGatherNumberOfMessages(comm,w2,w1,&nrqr));

    /* Input: nrqs: nsend; nrqr: nrecv; w1: msg length to be sent;
       Output: onodes1: recv node-ids; olengths1: corresponding recv message length */
    PetscCall(PetscGatherMessageLengths(comm,nrqs,nrqr,w1,&onodes1,&olengths1));

    /* Now post the Irecvs corresponding to these messages */
    PetscCall(PetscPostIrecvInt(comm,tag1,nrqr,onodes1,olengths1,&rbuf1,&r_waits1));

    PetscCall(PetscFree(onodes1));
    PetscCall(PetscFree(olengths1));

    /* Allocate Memory for outgoing messages */
    PetscCall(PetscMalloc4(size,&sbuf1,size,&ptr,2*msz,&tmp,size,&ctr));
    PetscCall(PetscArrayzero(sbuf1,size));
    PetscCall(PetscArrayzero(ptr,size));

    /* subf1[pa[0]] = tmp, subf1[pa[i]] = subf1[pa[i-1]] + w1[pa[i-1]] */
    iptr = tmp;
    for (i=0; i<nrqs; i++) {
      proc        = pa[i];
      sbuf1[proc] = iptr;
      iptr       += w1[proc];
    }

    /* Form the outgoing messages */
    /* Initialize the header space */
    for (i=0; i<nrqs; i++) {
      proc      = pa[i];
      PetscCall(PetscArrayzero(sbuf1[proc],3));
      ptr[proc] = sbuf1[proc] + 3;
    }

    /* Parse the isrow and copy data into outbuf */
    PetscCall(PetscArrayzero(ctr,size));
    for (j=0; j<nrow; j++) {  /* parse the indices of each IS */
      proc = row2proc[j];
      if (proc != rank) { /* copy to the outgoing buf*/
        *ptr[proc] = irow[j];
        ctr[proc]++; ptr[proc]++;
      }
    }

    /* Update the headers for the current IS */
    for (j=0; j<size; j++) { /* Can Optimise this loop too */
      if ((ctr_j = ctr[j])) {
        sbuf1_j        = sbuf1[j];
        k              = ++sbuf1_j[0];
        sbuf1_j[2*k]   = ctr_j;
        sbuf1_j[2*k-1] = 0;
      }
    }

    /* Now post the sends */
    PetscCall(PetscMalloc1(nrqs,&s_waits1));
    for (i=0; i<nrqs; ++i) {
      proc = pa[i];
      PetscCallMPI(MPI_Isend(sbuf1[proc],w1[proc],MPIU_INT,proc,tag1,comm,s_waits1+i));
    }

    /* Post Receives to capture the buffer size */
    PetscCall(PetscMalloc4(nrqs,&r_status2,nrqr,&s_waits2,nrqs,&r_waits2,nrqr,&s_status2));
    PetscCall(PetscMalloc3(nrqs,&req_source2,nrqs,&rbuf2,nrqs,&rbuf3));

    if (nrqs) rbuf2[0] = tmp + msz;
    for (i=1; i<nrqs; ++i) rbuf2[i] = rbuf2[i-1] + w1[pa[i-1]];

    for (i=0; i<nrqs; ++i) {
      proc = pa[i];
      PetscCallMPI(MPI_Irecv(rbuf2[i],w1[proc],MPIU_INT,proc,tag2,comm,r_waits2+i));
    }

    PetscCall(PetscFree2(w1,w2));

    /* Send to other procs the buf size they should allocate */
    /* Receive messages*/
    PetscCall(PetscMalloc1(nrqr,&r_status1));
    PetscCall(PetscMalloc3(nrqr,&sbuf2,nrqr,&req_size,nrqr,&req_source1));

    PetscCallMPI(MPI_Waitall(nrqr,r_waits1,r_status1));
    for (i=0; i<nrqr; ++i) {
      req_size[i] = 0;
      rbuf1_i        = rbuf1[i];
      start          = 2*rbuf1_i[0] + 1;
      PetscCallMPI(MPI_Get_count(r_status1+i,MPIU_INT,&end));
      PetscCall(PetscMalloc1(end,&sbuf2[i]));
      sbuf2_i        = sbuf2[i];
      for (j=start; j<end; j++) {
        k            = rbuf1_i[j] - rstart;
        ncols        = ai[k+1] - ai[k] + bi[k+1] - bi[k];
        sbuf2_i[j]   = ncols;
        req_size[i] += ncols;
      }
      req_source1[i] = r_status1[i].MPI_SOURCE;

      /* form the header */
      sbuf2_i[0] = req_size[i];
      for (j=1; j<start; j++) sbuf2_i[j] = rbuf1_i[j];

      PetscCallMPI(MPI_Isend(sbuf2_i,end,MPIU_INT,req_source1[i],tag2,comm,s_waits2+i));
    }

    PetscCall(PetscFree(r_status1));
    PetscCall(PetscFree(r_waits1));

    /* rbuf2 is received, Post recv column indices a->j */
    PetscCallMPI(MPI_Waitall(nrqs,r_waits2,r_status2));

    PetscCall(PetscMalloc4(nrqs,&r_waits3,nrqr,&s_waits3,nrqs,&r_status3,nrqr,&s_status3));
    for (i=0; i<nrqs; ++i) {
      PetscCall(PetscMalloc1(rbuf2[i][0],&rbuf3[i]));
      req_source2[i] = r_status2[i].MPI_SOURCE;
      PetscCallMPI(MPI_Irecv(rbuf3[i],rbuf2[i][0],MPIU_INT,req_source2[i],tag3,comm,r_waits3+i));
    }

    /* Wait on sends1 and sends2 */
    PetscCall(PetscMalloc1(nrqs,&s_status1));
    PetscCallMPI(MPI_Waitall(nrqs,s_waits1,s_status1));
    PetscCall(PetscFree(s_waits1));
    PetscCall(PetscFree(s_status1));

    PetscCallMPI(MPI_Waitall(nrqr,s_waits2,s_status2));
    PetscCall(PetscFree4(r_status2,s_waits2,r_waits2,s_status2));

    /* Now allocate sending buffers for a->j, and send them off */
    PetscCall(PetscMalloc1(nrqr,&sbuf_aj));
    for (i=0,j=0; i<nrqr; i++) j += req_size[i];
    if (nrqr) PetscCall(PetscMalloc1(j,&sbuf_aj[0]));
    for (i=1; i<nrqr; i++) sbuf_aj[i] = sbuf_aj[i-1] + req_size[i-1];

    for (i=0; i<nrqr; i++) { /* for each requested message */
      rbuf1_i   = rbuf1[i];
      sbuf_aj_i = sbuf_aj[i];
      ct1       = 2*rbuf1_i[0] + 1;
      ct2       = 0;
      /* max1=rbuf1_i[0]; PetscCheckFalse(max1 != 1,PETSC_COMM_SELF,PETSC_ERR_PLIB,"max1 %d != 1",max1); */

      kmax = rbuf1[i][2];
      for (k=0; k<kmax; k++,ct1++) { /* for each row */
        row    = rbuf1_i[ct1] - rstart;
        nzA    = ai[row+1] - ai[row];
        nzB    = bi[row+1] - bi[row];
        ncols  = nzA + nzB;
        cworkA = aj + ai[row]; cworkB = bj + bi[row];

        /* load the column indices for this row into cols*/
        cols = sbuf_aj_i + ct2;

        lwrite = 0;
        for (l=0; l<nzB; l++) {
          if ((ctmp = bmap[cworkB[l]]) < cstart) cols[lwrite++] = ctmp;
        }
        for (l=0; l<nzA; l++) cols[lwrite++] = cstart + cworkA[l];
        for (l=0; l<nzB; l++) {
          if ((ctmp = bmap[cworkB[l]]) >= cend) cols[lwrite++] = ctmp;
        }

        ct2 += ncols;
      }
      PetscCallMPI(MPI_Isend(sbuf_aj_i,req_size[i],MPIU_INT,req_source1[i],tag3,comm,s_waits3+i));
    }

    /* create column map (cmap): global col of C -> local col of submat */
#if defined(PETSC_USE_CTABLE)
    if (!allcolumns) {
      PetscCall(PetscTableCreate(ncol,C->cmap->N,&cmap));
      PetscCall(PetscCalloc1(C->cmap->n,&cmap_loc));
      for (j=0; j<ncol; j++) { /* use array cmap_loc[] for local col indices */
        if (icol[j] >= cstart && icol[j] <cend) {
          cmap_loc[icol[j] - cstart] = j+1;
        } else { /* use PetscTable for non-local col indices */
          PetscCall(PetscTableAdd(cmap,icol[j]+1,j+1,INSERT_VALUES));
        }
      }
    } else {
      cmap     = NULL;
      cmap_loc = NULL;
    }
    PetscCall(PetscCalloc1(C->rmap->n,&rmap_loc));
#else
    if (!allcolumns) {
      PetscCall(PetscCalloc1(C->cmap->N,&cmap));
      for (j=0; j<ncol; j++) cmap[icol[j]] = j+1;
    } else {
      cmap = NULL;
    }
#endif

    /* Create lens for MatSeqAIJSetPreallocation() */
    PetscCall(PetscCalloc1(nrow,&lens));

    /* Compute lens from local part of C */
    for (j=0; j<nrow; j++) {
      row  = irow[j];
      proc = row2proc[j];
      if (proc == rank) {
        /* diagonal part A = c->A */
        ncols = ai[row-rstart+1] - ai[row-rstart];
        cols  = aj + ai[row-rstart];
        if (!allcolumns) {
          for (k=0; k<ncols; k++) {
#if defined(PETSC_USE_CTABLE)
            tcol = cmap_loc[cols[k]];
#else
            tcol = cmap[cols[k]+cstart];
#endif
            if (tcol) lens[j]++;
          }
        } else { /* allcolumns */
          lens[j] = ncols;
        }

        /* off-diagonal part B = c->B */
        ncols = bi[row-rstart+1] - bi[row-rstart];
        cols  = bj + bi[row-rstart];
        if (!allcolumns) {
          for (k=0; k<ncols; k++) {
#if defined(PETSC_USE_CTABLE)
            PetscCall(PetscTableFind(cmap,bmap[cols[k]]+1,&tcol));
#else
            tcol = cmap[bmap[cols[k]]];
#endif
            if (tcol) lens[j]++;
          }
        } else { /* allcolumns */
          lens[j] += ncols;
        }
      }
    }

    /* Create row map (rmap): global row of C -> local row of submat */
#if defined(PETSC_USE_CTABLE)
    PetscCall(PetscTableCreate(nrow,C->rmap->N,&rmap));
    for (j=0; j<nrow; j++) {
      row  = irow[j];
      proc = row2proc[j];
      if (proc == rank) { /* a local row */
        rmap_loc[row - rstart] = j;
      } else {
        PetscCall(PetscTableAdd(rmap,irow[j]+1,j+1,INSERT_VALUES));
      }
    }
#else
    PetscCall(PetscCalloc1(C->rmap->N,&rmap));
    for (j=0; j<nrow; j++) {
      rmap[irow[j]] = j;
    }
#endif

    /* Update lens from offproc data */
    /* recv a->j is done */
    PetscCallMPI(MPI_Waitall(nrqs,r_waits3,r_status3));
    for (i=0; i<nrqs; i++) {
      proc    = pa[i];
      sbuf1_i = sbuf1[proc];
      /* jmax    = sbuf1_i[0]; PetscCheckFalse(jmax != 1,PETSC_COMM_SELF,PETSC_ERR_PLIB,"jmax !=1"); */
      ct1     = 2 + 1;
      ct2     = 0;
      rbuf2_i = rbuf2[i]; /* received length of C->j */
      rbuf3_i = rbuf3[i]; /* received C->j */

      /* is_no  = sbuf1_i[2*j-1]; PetscCheckFalse(is_no != 0,PETSC_COMM_SELF,PETSC_ERR_PLIB,"is_no !=0"); */
      max1   = sbuf1_i[2];
      for (k=0; k<max1; k++,ct1++) {
#if defined(PETSC_USE_CTABLE)
        PetscCall(PetscTableFind(rmap,sbuf1_i[ct1]+1,&row));
        row--;
        PetscCheckFalse(row < 0,PETSC_COMM_SELF,PETSC_ERR_PLIB,"row not found in table");
#else
        row = rmap[sbuf1_i[ct1]]; /* the row index in submat */
#endif
        /* Now, store row index of submat in sbuf1_i[ct1] */
        sbuf1_i[ct1] = row;

        nnz = rbuf2_i[ct1];
        if (!allcolumns) {
          for (l=0; l<nnz; l++,ct2++) {
#if defined(PETSC_USE_CTABLE)
            if (rbuf3_i[ct2] >= cstart && rbuf3_i[ct2] <cend) {
              tcol = cmap_loc[rbuf3_i[ct2] - cstart];
            } else {
              PetscCall(PetscTableFind(cmap,rbuf3_i[ct2]+1,&tcol));
            }
#else
            tcol = cmap[rbuf3_i[ct2]]; /* column index in submat */
#endif
            if (tcol) lens[row]++;
          }
        } else { /* allcolumns */
          lens[row] += nnz;
        }
      }
    }
    PetscCallMPI(MPI_Waitall(nrqr,s_waits3,s_status3));
    PetscCall(PetscFree4(r_waits3,s_waits3,r_status3,s_status3));

    /* Create the submatrices */
    PetscCall(MatCreate(PETSC_COMM_SELF,&submat));
    PetscCall(MatSetSizes(submat,nrow,ncol,PETSC_DETERMINE,PETSC_DETERMINE));

    PetscCall(ISGetBlockSize(isrow[0],&i));
    PetscCall(ISGetBlockSize(iscol[0],&j));
    PetscCall(MatSetBlockSizes(submat,i,j));
    PetscCall(MatSetType(submat,((PetscObject)A)->type_name));
    PetscCall(MatSeqAIJSetPreallocation(submat,0,lens));

    /* create struct Mat_SubSppt and attached it to submat */
    PetscCall(PetscNew(&smatis1));
    subc = (Mat_SeqAIJ*)submat->data;
    subc->submatis1 = smatis1;

    smatis1->id          = 0;
    smatis1->nrqs        = nrqs;
    smatis1->nrqr        = nrqr;
    smatis1->rbuf1       = rbuf1;
    smatis1->rbuf2       = rbuf2;
    smatis1->rbuf3       = rbuf3;
    smatis1->sbuf2       = sbuf2;
    smatis1->req_source2 = req_source2;

    smatis1->sbuf1       = sbuf1;
    smatis1->ptr         = ptr;
    smatis1->tmp         = tmp;
    smatis1->ctr         = ctr;

    smatis1->pa           = pa;
    smatis1->req_size     = req_size;
    smatis1->req_source1  = req_source1;

    smatis1->allcolumns  = allcolumns;
    smatis1->singleis    = PETSC_TRUE;
    smatis1->row2proc    = row2proc;
    smatis1->rmap        = rmap;
    smatis1->cmap        = cmap;
#if defined(PETSC_USE_CTABLE)
    smatis1->rmap_loc    = rmap_loc;
    smatis1->cmap_loc    = cmap_loc;
#endif

    smatis1->destroy     = submat->ops->destroy;
    submat->ops->destroy = MatDestroySubMatrix_SeqAIJ;
    submat->factortype   = C->factortype;

    /* compute rmax */
    rmax = 0;
    for (i=0; i<nrow; i++) rmax = PetscMax(rmax,lens[i]);

  } else { /* scall == MAT_REUSE_MATRIX */
    submat = submats[0];
    PetscCheckFalse(submat->rmap->n != nrow || submat->cmap->n != ncol,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Cannot reuse matrix. wrong size");

    subc    = (Mat_SeqAIJ*)submat->data;
    rmax    = subc->rmax;
    smatis1 = subc->submatis1;
    nrqs        = smatis1->nrqs;
    nrqr        = smatis1->nrqr;
    rbuf1       = smatis1->rbuf1;
    rbuf2       = smatis1->rbuf2;
    rbuf3       = smatis1->rbuf3;
    req_source2 = smatis1->req_source2;

    sbuf1     = smatis1->sbuf1;
    sbuf2     = smatis1->sbuf2;
    ptr       = smatis1->ptr;
    tmp       = smatis1->tmp;
    ctr       = smatis1->ctr;

    pa         = smatis1->pa;
    req_size   = smatis1->req_size;
    req_source1 = smatis1->req_source1;

    allcolumns = smatis1->allcolumns;
    row2proc   = smatis1->row2proc;
    rmap       = smatis1->rmap;
    cmap       = smatis1->cmap;
#if defined(PETSC_USE_CTABLE)
    rmap_loc   = smatis1->rmap_loc;
    cmap_loc   = smatis1->cmap_loc;
#endif
  }

  /* Post recv matrix values */
  PetscCall(PetscMalloc3(nrqs,&rbuf4, rmax,&subcols, rmax,&subvals));
  PetscCall(PetscMalloc4(nrqs,&r_waits4,nrqr,&s_waits4,nrqs,&r_status4,nrqr,&s_status4));
  PetscCall(PetscObjectGetNewTag((PetscObject)C,&tag4));
  for (i=0; i<nrqs; ++i) {
    PetscCall(PetscMalloc1(rbuf2[i][0],&rbuf4[i]));
    PetscCallMPI(MPI_Irecv(rbuf4[i],rbuf2[i][0],MPIU_SCALAR,req_source2[i],tag4,comm,r_waits4+i));
  }

  /* Allocate sending buffers for a->a, and send them off */
  PetscCall(PetscMalloc1(nrqr,&sbuf_aa));
  for (i=0,j=0; i<nrqr; i++) j += req_size[i];
  if (nrqr) PetscCall(PetscMalloc1(j,&sbuf_aa[0]));
  for (i=1; i<nrqr; i++) sbuf_aa[i] = sbuf_aa[i-1] + req_size[i-1];

  for (i=0; i<nrqr; i++) {
    rbuf1_i   = rbuf1[i];
    sbuf_aa_i = sbuf_aa[i];
    ct1       = 2*rbuf1_i[0]+1;
    ct2       = 0;
    /* max1=rbuf1_i[0]; PetscCheckFalse(max1 != 1,PETSC_COMM_SELF,PETSC_ERR_PLIB,"max1 !=1"); */

    kmax = rbuf1_i[2];
    for (k=0; k<kmax; k++,ct1++) {
      row = rbuf1_i[ct1] - rstart;
      nzA = ai[row+1] - ai[row];
      nzB = bi[row+1] - bi[row];
      ncols  = nzA + nzB;
      cworkB = bj + bi[row];
      vworkA = a_a + ai[row];
      vworkB = b_a + bi[row];

      /* load the column values for this row into vals*/
      vals = sbuf_aa_i + ct2;

      lwrite = 0;
      for (l=0; l<nzB; l++) {
        if ((bmap[cworkB[l]]) < cstart) vals[lwrite++] = vworkB[l];
      }
      for (l=0; l<nzA; l++) vals[lwrite++] = vworkA[l];
      for (l=0; l<nzB; l++) {
        if ((bmap[cworkB[l]]) >= cend) vals[lwrite++] = vworkB[l];
      }

      ct2 += ncols;
    }
    PetscCallMPI(MPI_Isend(sbuf_aa_i,req_size[i],MPIU_SCALAR,req_source1[i],tag4,comm,s_waits4+i));
  }

  /* Assemble submat */
  /* First assemble the local rows */
  for (j=0; j<nrow; j++) {
    row  = irow[j];
    proc = row2proc[j];
    if (proc == rank) {
      Crow = row - rstart;  /* local row index of C */
#if defined(PETSC_USE_CTABLE)
      row = rmap_loc[Crow]; /* row index of submat */
#else
      row = rmap[row];
#endif

      if (allcolumns) {
        /* diagonal part A = c->A */
        ncols = ai[Crow+1] - ai[Crow];
        cols  = aj + ai[Crow];
        vals  = a_a + ai[Crow];
        i     = 0;
        for (k=0; k<ncols; k++) {
          subcols[i]   = cols[k] + cstart;
          subvals[i++] = vals[k];
        }

        /* off-diagonal part B = c->B */
        ncols = bi[Crow+1] - bi[Crow];
        cols  = bj + bi[Crow];
        vals  = b_a + bi[Crow];
        for (k=0; k<ncols; k++) {
          subcols[i]   = bmap[cols[k]];
          subvals[i++] = vals[k];
        }

        PetscCall(MatSetValues_SeqAIJ(submat,1,&row,i,subcols,subvals,INSERT_VALUES));

      } else { /* !allcolumns */
#if defined(PETSC_USE_CTABLE)
        /* diagonal part A = c->A */
        ncols = ai[Crow+1] - ai[Crow];
        cols  = aj + ai[Crow];
        vals  = a_a + ai[Crow];
        i     = 0;
        for (k=0; k<ncols; k++) {
          tcol = cmap_loc[cols[k]];
          if (tcol) {
            subcols[i]   = --tcol;
            subvals[i++] = vals[k];
          }
        }

        /* off-diagonal part B = c->B */
        ncols = bi[Crow+1] - bi[Crow];
        cols  = bj + bi[Crow];
        vals  = b_a + bi[Crow];
        for (k=0; k<ncols; k++) {
          PetscCall(PetscTableFind(cmap,bmap[cols[k]]+1,&tcol));
          if (tcol) {
            subcols[i]   = --tcol;
            subvals[i++] = vals[k];
          }
        }
#else
        /* diagonal part A = c->A */
        ncols = ai[Crow+1] - ai[Crow];
        cols  = aj + ai[Crow];
        vals  = a_a + ai[Crow];
        i     = 0;
        for (k=0; k<ncols; k++) {
          tcol = cmap[cols[k]+cstart];
          if (tcol) {
            subcols[i]   = --tcol;
            subvals[i++] = vals[k];
          }
        }

        /* off-diagonal part B = c->B */
        ncols = bi[Crow+1] - bi[Crow];
        cols  = bj + bi[Crow];
        vals  = b_a + bi[Crow];
        for (k=0; k<ncols; k++) {
          tcol = cmap[bmap[cols[k]]];
          if (tcol) {
            subcols[i]   = --tcol;
            subvals[i++] = vals[k];
          }
        }
#endif
        PetscCall(MatSetValues_SeqAIJ(submat,1,&row,i,subcols,subvals,INSERT_VALUES));
      }
    }
  }

  /* Now assemble the off-proc rows */
  for (i=0; i<nrqs; i++) { /* for each requested message */
    /* recv values from other processes */
    PetscCallMPI(MPI_Waitany(nrqs,r_waits4,&idex,r_status4+i));
    proc    = pa[idex];
    sbuf1_i = sbuf1[proc];
    /* jmax    = sbuf1_i[0]; PetscCheckFalse(jmax != 1,PETSC_COMM_SELF,PETSC_ERR_PLIB,"jmax %d != 1",jmax); */
    ct1     = 2 + 1;
    ct2     = 0; /* count of received C->j */
    ct3     = 0; /* count of received C->j that will be inserted into submat */
    rbuf2_i = rbuf2[idex]; /* int** received length of C->j from other processes */
    rbuf3_i = rbuf3[idex]; /* int** received C->j from other processes */
    rbuf4_i = rbuf4[idex]; /* scalar** received C->a from other processes */

    /* is_no = sbuf1_i[2*j-1]; PetscCheckFalse(is_no != 0,PETSC_COMM_SELF,PETSC_ERR_PLIB,"is_no !=0"); */
    max1 = sbuf1_i[2];             /* num of rows */
    for (k=0; k<max1; k++,ct1++) { /* for each recved row */
      row = sbuf1_i[ct1]; /* row index of submat */
      if (!allcolumns) {
        idex = 0;
        if (scall == MAT_INITIAL_MATRIX || !iscolsorted) {
          nnz  = rbuf2_i[ct1]; /* num of C entries in this row */
          for (l=0; l<nnz; l++,ct2++) { /* for each recved column */
#if defined(PETSC_USE_CTABLE)
            if (rbuf3_i[ct2] >= cstart && rbuf3_i[ct2] <cend) {
              tcol = cmap_loc[rbuf3_i[ct2] - cstart];
            } else {
              PetscCall(PetscTableFind(cmap,rbuf3_i[ct2]+1,&tcol));
            }
#else
            tcol = cmap[rbuf3_i[ct2]];
#endif
            if (tcol) {
              subcols[idex]   = --tcol; /* may not be sorted */
              subvals[idex++] = rbuf4_i[ct2];

              /* We receive an entire column of C, but a subset of it needs to be inserted into submat.
               For reuse, we replace received C->j with index that should be inserted to submat */
              if (iscolsorted) rbuf3_i[ct3++] = ct2;
            }
          }
          PetscCall(MatSetValues_SeqAIJ(submat,1,&row,idex,subcols,subvals,INSERT_VALUES));
        } else { /* scall == MAT_REUSE_MATRIX */
          submat = submats[0];
          subc   = (Mat_SeqAIJ*)submat->data;

          nnz = subc->i[row+1] - subc->i[row]; /* num of submat entries in this row */
          for (l=0; l<nnz; l++) {
            ct2 = rbuf3_i[ct3++]; /* index of rbuf4_i[] which needs to be inserted into submat */
            subvals[idex++] = rbuf4_i[ct2];
          }

          bj = subc->j + subc->i[row]; /* sorted column indices */
          PetscCall(MatSetValues_SeqAIJ(submat,1,&row,nnz,bj,subvals,INSERT_VALUES));
        }
      } else { /* allcolumns */
        nnz  = rbuf2_i[ct1]; /* num of C entries in this row */
        PetscCall(MatSetValues_SeqAIJ(submat,1,&row,nnz,rbuf3_i+ct2,rbuf4_i+ct2,INSERT_VALUES));
        ct2 += nnz;
      }
    }
  }

  /* sending a->a are done */
  PetscCallMPI(MPI_Waitall(nrqr,s_waits4,s_status4));
  PetscCall(PetscFree4(r_waits4,s_waits4,r_status4,s_status4));

  PetscCall(MatAssemblyBegin(submat,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(submat,MAT_FINAL_ASSEMBLY));
  submats[0] = submat;

  /* Restore the indices */
  PetscCall(ISRestoreIndices(isrow[0],&irow));
  if (!allcolumns) {
    PetscCall(ISRestoreIndices(iscol[0],&icol));
  }

  /* Destroy allocated memory */
  for (i=0; i<nrqs; ++i) {
    PetscCall(PetscFree(rbuf4[i]));
  }
  PetscCall(PetscFree3(rbuf4,subcols,subvals));
  if (sbuf_aa) {
    PetscCall(PetscFree(sbuf_aa[0]));
    PetscCall(PetscFree(sbuf_aa));
  }

  if (scall == MAT_INITIAL_MATRIX) {
    PetscCall(PetscFree(lens));
    if (sbuf_aj) {
      PetscCall(PetscFree(sbuf_aj[0]));
      PetscCall(PetscFree(sbuf_aj));
    }
  }
  PetscCall(MatSeqAIJRestoreArrayRead(A,(const PetscScalar**)&a_a));
  PetscCall(MatSeqAIJRestoreArrayRead(B,(const PetscScalar**)&b_a));
  PetscFunctionReturn(0);
}

PetscErrorCode MatCreateSubMatrices_MPIAIJ_SingleIS(Mat C,PetscInt ismax,const IS isrow[],const IS iscol[],MatReuse scall,Mat *submat[])
{
  PetscInt       ncol;
  PetscBool      colflag,allcolumns=PETSC_FALSE;

  PetscFunctionBegin;
  /* Allocate memory to hold all the submatrices */
  if (scall == MAT_INITIAL_MATRIX) {
    PetscCall(PetscCalloc1(2,submat));
  }

  /* Check for special case: each processor gets entire matrix columns */
  PetscCall(ISIdentity(iscol[0],&colflag));
  PetscCall(ISGetLocalSize(iscol[0],&ncol));
  if (colflag && ncol == C->cmap->N) allcolumns = PETSC_TRUE;

  PetscCall(MatCreateSubMatrices_MPIAIJ_SingleIS_Local(C,ismax,isrow,iscol,scall,allcolumns,*submat));
  PetscFunctionReturn(0);
}

PetscErrorCode MatCreateSubMatrices_MPIAIJ(Mat C,PetscInt ismax,const IS isrow[],const IS iscol[],MatReuse scall,Mat *submat[])
{
  PetscInt       nmax,nstages=0,i,pos,max_no,nrow,ncol,in[2],out[2];
  PetscBool      rowflag,colflag,wantallmatrix=PETSC_FALSE;
  Mat_SeqAIJ     *subc;
  Mat_SubSppt    *smat;

  PetscFunctionBegin;
  /* Check for special case: each processor has a single IS */
  if (C->submat_singleis) { /* flag is set in PCSetUp_ASM() to skip MPI_Allreduce() */
    PetscCall(MatCreateSubMatrices_MPIAIJ_SingleIS(C,ismax,isrow,iscol,scall,submat));
    C->submat_singleis = PETSC_FALSE; /* resume its default value in case C will be used for non-single IS */
    PetscFunctionReturn(0);
  }

  /* Collect global wantallmatrix and nstages */
  if (!C->cmap->N) nmax=20*1000000/sizeof(PetscInt);
  else nmax = 20*1000000 / (C->cmap->N * sizeof(PetscInt));
  if (!nmax) nmax = 1;

  if (scall == MAT_INITIAL_MATRIX) {
    /* Collect global wantallmatrix and nstages */
    if (ismax == 1 && C->rmap->N == C->cmap->N) {
      PetscCall(ISIdentity(*isrow,&rowflag));
      PetscCall(ISIdentity(*iscol,&colflag));
      PetscCall(ISGetLocalSize(*isrow,&nrow));
      PetscCall(ISGetLocalSize(*iscol,&ncol));
      if (rowflag && colflag && nrow == C->rmap->N && ncol == C->cmap->N) {
        wantallmatrix = PETSC_TRUE;

        PetscCall(PetscOptionsGetBool(((PetscObject)C)->options,((PetscObject)C)->prefix,"-use_fast_submatrix",&wantallmatrix,NULL));
      }
    }

    /* Determine the number of stages through which submatrices are done
       Each stage will extract nmax submatrices.
       nmax is determined by the matrix column dimension.
       If the original matrix has 20M columns, only one submatrix per stage is allowed, etc.
    */
    nstages = ismax/nmax + ((ismax % nmax) ? 1 : 0); /* local nstages */

    in[0] = -1*(PetscInt)wantallmatrix;
    in[1] = nstages;
    PetscCallMPI(MPIU_Allreduce(in,out,2,MPIU_INT,MPI_MAX,PetscObjectComm((PetscObject)C)));
    wantallmatrix = (PetscBool)(-out[0]);
    nstages       = out[1]; /* Make sure every processor loops through the global nstages */

  } else { /* MAT_REUSE_MATRIX */
    if (ismax) {
      subc = (Mat_SeqAIJ*)(*submat)[0]->data;
      smat = subc->submatis1;
    } else { /* (*submat)[0] is a dummy matrix */
      smat = (Mat_SubSppt*)(*submat)[0]->data;
    }
    if (!smat) {
      /* smat is not generated by MatCreateSubMatrix_MPIAIJ_All(...,MAT_INITIAL_MATRIX,...) */
      wantallmatrix = PETSC_TRUE;
    } else if (smat->singleis) {
      PetscCall(MatCreateSubMatrices_MPIAIJ_SingleIS(C,ismax,isrow,iscol,scall,submat));
      PetscFunctionReturn(0);
    } else {
      nstages = smat->nstages;
    }
  }

  if (wantallmatrix) {
    PetscCall(MatCreateSubMatrix_MPIAIJ_All(C,MAT_GET_VALUES,scall,submat));
    PetscFunctionReturn(0);
  }

  /* Allocate memory to hold all the submatrices and dummy submatrices */
  if (scall == MAT_INITIAL_MATRIX) {
    PetscCall(PetscCalloc1(ismax+nstages,submat));
  }

  for (i=0,pos=0; i<nstages; i++) {
    if (pos+nmax <= ismax) max_no = nmax;
    else if (pos >= ismax) max_no = 0;
    else                   max_no = ismax-pos;

    PetscCall(MatCreateSubMatrices_MPIAIJ_Local(C,max_no,isrow+pos,iscol+pos,scall,*submat+pos));
    if (!max_no) {
      if (scall == MAT_INITIAL_MATRIX) { /* submat[pos] is a dummy matrix */
        smat = (Mat_SubSppt*)(*submat)[pos]->data;
        smat->nstages = nstages;
      }
      pos++; /* advance to next dummy matrix if any */
    } else pos += max_no;
  }

  if (ismax && scall == MAT_INITIAL_MATRIX) {
    /* save nstages for reuse */
    subc = (Mat_SeqAIJ*)(*submat)[0]->data;
    smat = subc->submatis1;
    smat->nstages = nstages;
  }
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------*/
PetscErrorCode MatCreateSubMatrices_MPIAIJ_Local(Mat C,PetscInt ismax,const IS isrow[],const IS iscol[],MatReuse scall,Mat *submats)
{
  Mat_MPIAIJ     *c = (Mat_MPIAIJ*)C->data;
  Mat            A  = c->A;
  Mat_SeqAIJ     *a = (Mat_SeqAIJ*)A->data,*b = (Mat_SeqAIJ*)c->B->data,*subc;
  const PetscInt **icol,**irow;
  PetscInt       *nrow,*ncol,start;
  PetscMPIInt    rank,size,tag0,tag2,tag3,tag4,*w1,*w2,*w3,*w4,nrqr;
  PetscInt       **sbuf1,**sbuf2,i,j,k,l,ct1,ct2,**rbuf1,row,proc=-1;
  PetscInt       nrqs=0,msz,**ptr=NULL,*req_size=NULL,*ctr=NULL,*pa,*tmp=NULL,tcol;
  PetscInt       **rbuf3=NULL,*req_source1=NULL,*req_source2,**sbuf_aj,**rbuf2=NULL,max1,max2;
  PetscInt       **lens,is_no,ncols,*cols,mat_i,*mat_j,tmp2,jmax;
#if defined(PETSC_USE_CTABLE)
  PetscTable     *cmap,cmap_i=NULL,*rmap,rmap_i;
#else
  PetscInt       **cmap,*cmap_i=NULL,**rmap,*rmap_i;
#endif
  const PetscInt *irow_i;
  PetscInt       ctr_j,*sbuf1_j,*sbuf_aj_i,*rbuf1_i,kmax,*lens_i;
  MPI_Request    *s_waits1,*r_waits1,*s_waits2,*r_waits2,*r_waits3;
  MPI_Request    *r_waits4,*s_waits3,*s_waits4;
  MPI_Comm       comm;
  PetscScalar    **rbuf4,*rbuf4_i,**sbuf_aa,*vals,*mat_a,*imat_a,*sbuf_aa_i;
  PetscMPIInt    *onodes1,*olengths1,end;
  PetscInt       **row2proc,*row2proc_i,ilen_row,*imat_ilen,*imat_j,*imat_i,old_row;
  Mat_SubSppt    *smat_i;
  PetscBool      *issorted,*allcolumns,colflag,iscsorted=PETSC_TRUE;
  PetscInt       *sbuf1_i,*rbuf2_i,*rbuf3_i,ilen;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)C,&comm));
  size = c->size;
  rank = c->rank;

  PetscCall(PetscMalloc4(ismax,&row2proc,ismax,&cmap,ismax,&rmap,ismax+1,&allcolumns));
  PetscCall(PetscMalloc5(ismax,(PetscInt***)&irow,ismax,(PetscInt***)&icol,ismax,&nrow,ismax,&ncol,ismax,&issorted));

  for (i=0; i<ismax; i++) {
    PetscCall(ISSorted(iscol[i],&issorted[i]));
    if (!issorted[i]) iscsorted = issorted[i];

    PetscCall(ISSorted(isrow[i],&issorted[i]));

    PetscCall(ISGetIndices(isrow[i],&irow[i]));
    PetscCall(ISGetLocalSize(isrow[i],&nrow[i]));

    /* Check for special case: allcolumn */
    PetscCall(ISIdentity(iscol[i],&colflag));
    PetscCall(ISGetLocalSize(iscol[i],&ncol[i]));
    if (colflag && ncol[i] == C->cmap->N) {
      allcolumns[i] = PETSC_TRUE;
      icol[i] = NULL;
    } else {
      allcolumns[i] = PETSC_FALSE;
      PetscCall(ISGetIndices(iscol[i],&icol[i]));
    }
  }

  if (scall == MAT_REUSE_MATRIX) {
    /* Assumes new rows are same length as the old rows */
    for (i=0; i<ismax; i++) {
      PetscCheckFalse(!submats[i],PETSC_COMM_SELF,PETSC_ERR_ARG_NULL,"submats[%" PetscInt_FMT "] is null, cannot reuse",i);
      subc = (Mat_SeqAIJ*)submats[i]->data;
      PetscCheckFalse((submats[i]->rmap->n != nrow[i]) || (submats[i]->cmap->n != ncol[i]),PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Cannot reuse matrix. wrong size");

      /* Initial matrix as if empty */
      PetscCall(PetscArrayzero(subc->ilen,submats[i]->rmap->n));

      smat_i   = subc->submatis1;

      nrqs        = smat_i->nrqs;
      nrqr        = smat_i->nrqr;
      rbuf1       = smat_i->rbuf1;
      rbuf2       = smat_i->rbuf2;
      rbuf3       = smat_i->rbuf3;
      req_source2 = smat_i->req_source2;

      sbuf1     = smat_i->sbuf1;
      sbuf2     = smat_i->sbuf2;
      ptr       = smat_i->ptr;
      tmp       = smat_i->tmp;
      ctr       = smat_i->ctr;

      pa          = smat_i->pa;
      req_size    = smat_i->req_size;
      req_source1 = smat_i->req_source1;

      allcolumns[i] = smat_i->allcolumns;
      row2proc[i]   = smat_i->row2proc;
      rmap[i]       = smat_i->rmap;
      cmap[i]       = smat_i->cmap;
    }

    if (!ismax) { /* Get dummy submatrices and retrieve struct submatis1 */
      PetscCheckFalse(!submats[0],PETSC_COMM_SELF,PETSC_ERR_ARG_NULL,"submats are null, cannot reuse");
      smat_i = (Mat_SubSppt*)submats[0]->data;

      nrqs        = smat_i->nrqs;
      nrqr        = smat_i->nrqr;
      rbuf1       = smat_i->rbuf1;
      rbuf2       = smat_i->rbuf2;
      rbuf3       = smat_i->rbuf3;
      req_source2 = smat_i->req_source2;

      sbuf1       = smat_i->sbuf1;
      sbuf2       = smat_i->sbuf2;
      ptr         = smat_i->ptr;
      tmp         = smat_i->tmp;
      ctr         = smat_i->ctr;

      pa          = smat_i->pa;
      req_size    = smat_i->req_size;
      req_source1 = smat_i->req_source1;

      allcolumns[0] = PETSC_FALSE;
    }
  } else { /* scall == MAT_INITIAL_MATRIX */
    /* Get some new tags to keep the communication clean */
    PetscCall(PetscObjectGetNewTag((PetscObject)C,&tag2));
    PetscCall(PetscObjectGetNewTag((PetscObject)C,&tag3));

    /* evaluate communication - mesg to who, length of mesg, and buffer space
     required. Based on this, buffers are allocated, and data copied into them*/
    PetscCall(PetscCalloc4(size,&w1,size,&w2,size,&w3,size,&w4));   /* mesg size, initialize work vectors */

    for (i=0; i<ismax; i++) {
      jmax   = nrow[i];
      irow_i = irow[i];

      PetscCall(PetscMalloc1(jmax,&row2proc_i));
      row2proc[i] = row2proc_i;

      if (issorted[i]) proc = 0;
      for (j=0; j<jmax; j++) {
        if (!issorted[i]) proc = 0;
        row = irow_i[j];
        while (row >= C->rmap->range[proc+1]) proc++;
        w4[proc]++;
        row2proc_i[j] = proc; /* map row index to proc */
      }
      for (j=0; j<size; j++) {
        if (w4[j]) { w1[j] += w4[j];  w3[j]++; w4[j] = 0;}
      }
    }

    nrqs     = 0;              /* no of outgoing messages */
    msz      = 0;              /* total mesg length (for all procs) */
    w1[rank] = 0;              /* no mesg sent to self */
    w3[rank] = 0;
    for (i=0; i<size; i++) {
      if (w1[i])  { w2[i] = 1; nrqs++;} /* there exists a message to proc i */
    }
    PetscCall(PetscMalloc1(nrqs,&pa)); /*(proc -array)*/
    for (i=0,j=0; i<size; i++) {
      if (w1[i]) { pa[j] = i; j++; }
    }

    /* Each message would have a header = 1 + 2*(no of IS) + data */
    for (i=0; i<nrqs; i++) {
      j      = pa[i];
      w1[j] += w2[j] + 2* w3[j];
      msz   += w1[j];
    }
    PetscCall(PetscInfo(0,"Number of outgoing messages %" PetscInt_FMT " Total message length %" PetscInt_FMT "\n",nrqs,msz));

    /* Determine the number of messages to expect, their lengths, from from-ids */
    PetscCall(PetscGatherNumberOfMessages(comm,w2,w1,&nrqr));
    PetscCall(PetscGatherMessageLengths(comm,nrqs,nrqr,w1,&onodes1,&olengths1));

    /* Now post the Irecvs corresponding to these messages */
    PetscCall(PetscObjectGetNewTag((PetscObject)C,&tag0));
    PetscCall(PetscPostIrecvInt(comm,tag0,nrqr,onodes1,olengths1,&rbuf1,&r_waits1));

    /* Allocate Memory for outgoing messages */
    PetscCall(PetscMalloc4(size,&sbuf1,size,&ptr,2*msz,&tmp,size,&ctr));
    PetscCall(PetscArrayzero(sbuf1,size));
    PetscCall(PetscArrayzero(ptr,size));

    {
      PetscInt *iptr = tmp;
      k    = 0;
      for (i=0; i<nrqs; i++) {
        j        = pa[i];
        iptr    += k;
        sbuf1[j] = iptr;
        k        = w1[j];
      }
    }

    /* Form the outgoing messages. Initialize the header space */
    for (i=0; i<nrqs; i++) {
      j           = pa[i];
      sbuf1[j][0] = 0;
      PetscCall(PetscArrayzero(sbuf1[j]+1,2*w3[j]));
      ptr[j]      = sbuf1[j] + 2*w3[j] + 1;
    }

    /* Parse the isrow and copy data into outbuf */
    for (i=0; i<ismax; i++) {
      row2proc_i = row2proc[i];
      PetscCall(PetscArrayzero(ctr,size));
      irow_i = irow[i];
      jmax   = nrow[i];
      for (j=0; j<jmax; j++) {  /* parse the indices of each IS */
        proc = row2proc_i[j];
        if (proc != rank) { /* copy to the outgoing buf*/
          ctr[proc]++;
          *ptr[proc] = irow_i[j];
          ptr[proc]++;
        }
      }
      /* Update the headers for the current IS */
      for (j=0; j<size; j++) { /* Can Optimise this loop too */
        if ((ctr_j = ctr[j])) {
          sbuf1_j        = sbuf1[j];
          k              = ++sbuf1_j[0];
          sbuf1_j[2*k]   = ctr_j;
          sbuf1_j[2*k-1] = i;
        }
      }
    }

    /*  Now  post the sends */
    PetscCall(PetscMalloc1(nrqs,&s_waits1));
    for (i=0; i<nrqs; ++i) {
      j    = pa[i];
      PetscCallMPI(MPI_Isend(sbuf1[j],w1[j],MPIU_INT,j,tag0,comm,s_waits1+i));
    }

    /* Post Receives to capture the buffer size */
    PetscCall(PetscMalloc1(nrqs,&r_waits2));
    PetscCall(PetscMalloc3(nrqs,&req_source2,nrqs,&rbuf2,nrqs,&rbuf3));
    if (nrqs) rbuf2[0] = tmp + msz;
    for (i=1; i<nrqs; ++i) {
      rbuf2[i] = rbuf2[i-1]+w1[pa[i-1]];
    }
    for (i=0; i<nrqs; ++i) {
      j    = pa[i];
      PetscCallMPI(MPI_Irecv(rbuf2[i],w1[j],MPIU_INT,j,tag2,comm,r_waits2+i));
    }

    /* Send to other procs the buf size they should allocate */
    /* Receive messages*/
    PetscCall(PetscMalloc1(nrqr,&s_waits2));
    PetscCall(PetscMalloc3(nrqr,&sbuf2,nrqr,&req_size,nrqr,&req_source1));
    {
      PetscInt   *sAi = a->i,*sBi = b->i,id,rstart = C->rmap->rstart;
      PetscInt   *sbuf2_i;

      PetscCallMPI(MPI_Waitall(nrqr,r_waits1,MPI_STATUSES_IGNORE));
      for (i=0; i<nrqr; ++i) {
        req_size[i] = 0;
        rbuf1_i        = rbuf1[i];
        start          = 2*rbuf1_i[0] + 1;
        end            = olengths1[i];
        PetscCall(PetscMalloc1(end,&sbuf2[i]));
        sbuf2_i        = sbuf2[i];
        for (j=start; j<end; j++) {
          id              = rbuf1_i[j] - rstart;
          ncols           = sAi[id+1] - sAi[id] + sBi[id+1] - sBi[id];
          sbuf2_i[j]      = ncols;
          req_size[i] += ncols;
        }
        req_source1[i] = onodes1[i];
        /* form the header */
        sbuf2_i[0] = req_size[i];
        for (j=1; j<start; j++) sbuf2_i[j] = rbuf1_i[j];

        PetscCallMPI(MPI_Isend(sbuf2_i,end,MPIU_INT,req_source1[i],tag2,comm,s_waits2+i));
      }
    }

    PetscCall(PetscFree(onodes1));
    PetscCall(PetscFree(olengths1));
    PetscCall(PetscFree(r_waits1));
    PetscCall(PetscFree4(w1,w2,w3,w4));

    /* Receive messages*/
    PetscCall(PetscMalloc1(nrqs,&r_waits3));
    PetscCallMPI(MPI_Waitall(nrqs,r_waits2,MPI_STATUSES_IGNORE));
    for (i=0; i<nrqs; ++i) {
      PetscCall(PetscMalloc1(rbuf2[i][0],&rbuf3[i]));
      req_source2[i] = pa[i];
      PetscCallMPI(MPI_Irecv(rbuf3[i],rbuf2[i][0],MPIU_INT,req_source2[i],tag3,comm,r_waits3+i));
    }
    PetscCall(PetscFree(r_waits2));

    /* Wait on sends1 and sends2 */
    PetscCallMPI(MPI_Waitall(nrqs,s_waits1,MPI_STATUSES_IGNORE));
    PetscCallMPI(MPI_Waitall(nrqr,s_waits2,MPI_STATUSES_IGNORE));
    PetscCall(PetscFree(s_waits1));
    PetscCall(PetscFree(s_waits2));

    /* Now allocate sending buffers for a->j, and send them off */
    PetscCall(PetscMalloc1(nrqr,&sbuf_aj));
    for (i=0,j=0; i<nrqr; i++) j += req_size[i];
    if (nrqr) PetscCall(PetscMalloc1(j,&sbuf_aj[0]));
    for (i=1; i<nrqr; i++) sbuf_aj[i] = sbuf_aj[i-1] + req_size[i-1];

    PetscCall(PetscMalloc1(nrqr,&s_waits3));
    {
      PetscInt nzA,nzB,*a_i = a->i,*b_i = b->i,lwrite;
      PetscInt *cworkA,*cworkB,cstart = C->cmap->rstart,rstart = C->rmap->rstart,*bmap = c->garray;
      PetscInt cend = C->cmap->rend;
      PetscInt *a_j = a->j,*b_j = b->j,ctmp;

      for (i=0; i<nrqr; i++) {
        rbuf1_i   = rbuf1[i];
        sbuf_aj_i = sbuf_aj[i];
        ct1       = 2*rbuf1_i[0] + 1;
        ct2       = 0;
        for (j=1,max1=rbuf1_i[0]; j<=max1; j++) {
          kmax = rbuf1[i][2*j];
          for (k=0; k<kmax; k++,ct1++) {
            row    = rbuf1_i[ct1] - rstart;
            nzA    = a_i[row+1] - a_i[row];
            nzB    = b_i[row+1] - b_i[row];
            ncols  = nzA + nzB;
            cworkA = a_j + a_i[row];
            cworkB = b_j + b_i[row];

            /* load the column indices for this row into cols */
            cols = sbuf_aj_i + ct2;

            lwrite = 0;
            for (l=0; l<nzB; l++) {
              if ((ctmp = bmap[cworkB[l]]) < cstart) cols[lwrite++] = ctmp;
            }
            for (l=0; l<nzA; l++) cols[lwrite++] = cstart + cworkA[l];
            for (l=0; l<nzB; l++) {
              if ((ctmp = bmap[cworkB[l]]) >= cend) cols[lwrite++] = ctmp;
            }

            ct2 += ncols;
          }
        }
        PetscCallMPI(MPI_Isend(sbuf_aj_i,req_size[i],MPIU_INT,req_source1[i],tag3,comm,s_waits3+i));
      }
    }

    /* create col map: global col of C -> local col of submatrices */
    {
      const PetscInt *icol_i;
#if defined(PETSC_USE_CTABLE)
      for (i=0; i<ismax; i++) {
        if (!allcolumns[i]) {
          PetscCall(PetscTableCreate(ncol[i],C->cmap->N,&cmap[i]));

          jmax   = ncol[i];
          icol_i = icol[i];
          cmap_i = cmap[i];
          for (j=0; j<jmax; j++) {
            PetscCall(PetscTableAdd(cmap[i],icol_i[j]+1,j+1,INSERT_VALUES));
          }
        } else cmap[i] = NULL;
      }
#else
      for (i=0; i<ismax; i++) {
        if (!allcolumns[i]) {
          PetscCall(PetscCalloc1(C->cmap->N,&cmap[i]));
          jmax   = ncol[i];
          icol_i = icol[i];
          cmap_i = cmap[i];
          for (j=0; j<jmax; j++) {
            cmap_i[icol_i[j]] = j+1;
          }
        } else cmap[i] = NULL;
      }
#endif
    }

    /* Create lens which is required for MatCreate... */
    for (i=0,j=0; i<ismax; i++) j += nrow[i];
    PetscCall(PetscMalloc1(ismax,&lens));

    if (ismax) {
      PetscCall(PetscCalloc1(j,&lens[0]));
    }
    for (i=1; i<ismax; i++) lens[i] = lens[i-1] + nrow[i-1];

    /* Update lens from local data */
    for (i=0; i<ismax; i++) {
      row2proc_i = row2proc[i];
      jmax = nrow[i];
      if (!allcolumns[i]) cmap_i = cmap[i];
      irow_i = irow[i];
      lens_i = lens[i];
      for (j=0; j<jmax; j++) {
        row = irow_i[j];
        proc = row2proc_i[j];
        if (proc == rank) {
          PetscCall(MatGetRow_MPIAIJ(C,row,&ncols,&cols,NULL));
          if (!allcolumns[i]) {
            for (k=0; k<ncols; k++) {
#if defined(PETSC_USE_CTABLE)
              PetscCall(PetscTableFind(cmap_i,cols[k]+1,&tcol));
#else
              tcol = cmap_i[cols[k]];
#endif
              if (tcol) lens_i[j]++;
            }
          } else { /* allcolumns */
            lens_i[j] = ncols;
          }
          PetscCall(MatRestoreRow_MPIAIJ(C,row,&ncols,&cols,NULL));
        }
      }
    }

    /* Create row map: global row of C -> local row of submatrices */
#if defined(PETSC_USE_CTABLE)
    for (i=0; i<ismax; i++) {
      PetscCall(PetscTableCreate(nrow[i],C->rmap->N,&rmap[i]));
      irow_i = irow[i];
      jmax   = nrow[i];
      for (j=0; j<jmax; j++) {
      PetscCall(PetscTableAdd(rmap[i],irow_i[j]+1,j+1,INSERT_VALUES));
      }
    }
#else
    for (i=0; i<ismax; i++) {
      PetscCall(PetscCalloc1(C->rmap->N,&rmap[i]));
      rmap_i = rmap[i];
      irow_i = irow[i];
      jmax   = nrow[i];
      for (j=0; j<jmax; j++) {
        rmap_i[irow_i[j]] = j;
      }
    }
#endif

    /* Update lens from offproc data */
    {
      PetscInt *rbuf2_i,*rbuf3_i,*sbuf1_i;

      PetscCallMPI(MPI_Waitall(nrqs,r_waits3,MPI_STATUSES_IGNORE));
      for (tmp2=0; tmp2<nrqs; tmp2++) {
        sbuf1_i = sbuf1[pa[tmp2]];
        jmax    = sbuf1_i[0];
        ct1     = 2*jmax+1;
        ct2     = 0;
        rbuf2_i = rbuf2[tmp2];
        rbuf3_i = rbuf3[tmp2];
        for (j=1; j<=jmax; j++) {
          is_no  = sbuf1_i[2*j-1];
          max1   = sbuf1_i[2*j];
          lens_i = lens[is_no];
          if (!allcolumns[is_no]) cmap_i = cmap[is_no];
          rmap_i = rmap[is_no];
          for (k=0; k<max1; k++,ct1++) {
#if defined(PETSC_USE_CTABLE)
            PetscCall(PetscTableFind(rmap_i,sbuf1_i[ct1]+1,&row));
            row--;
            PetscCheckFalse(row < 0,PETSC_COMM_SELF,PETSC_ERR_PLIB,"row not found in table");
#else
            row = rmap_i[sbuf1_i[ct1]]; /* the val in the new matrix to be */
#endif
            max2 = rbuf2_i[ct1];
            for (l=0; l<max2; l++,ct2++) {
              if (!allcolumns[is_no]) {
#if defined(PETSC_USE_CTABLE)
                PetscCall(PetscTableFind(cmap_i,rbuf3_i[ct2]+1,&tcol));
#else
                tcol = cmap_i[rbuf3_i[ct2]];
#endif
                if (tcol) lens_i[row]++;
              } else { /* allcolumns */
                lens_i[row]++; /* lens_i[row] += max2 ? */
              }
            }
          }
        }
      }
    }
    PetscCall(PetscFree(r_waits3));
    PetscCallMPI(MPI_Waitall(nrqr,s_waits3,MPI_STATUSES_IGNORE));
    PetscCall(PetscFree(s_waits3));

    /* Create the submatrices */
    for (i=0; i<ismax; i++) {
      PetscInt    rbs,cbs;

      PetscCall(ISGetBlockSize(isrow[i],&rbs));
      PetscCall(ISGetBlockSize(iscol[i],&cbs));

      PetscCall(MatCreate(PETSC_COMM_SELF,submats+i));
      PetscCall(MatSetSizes(submats[i],nrow[i],ncol[i],PETSC_DETERMINE,PETSC_DETERMINE));

      PetscCall(MatSetBlockSizes(submats[i],rbs,cbs));
      PetscCall(MatSetType(submats[i],((PetscObject)A)->type_name));
      PetscCall(MatSeqAIJSetPreallocation(submats[i],0,lens[i]));

      /* create struct Mat_SubSppt and attached it to submat */
      PetscCall(PetscNew(&smat_i));
      subc = (Mat_SeqAIJ*)submats[i]->data;
      subc->submatis1 = smat_i;

      smat_i->destroy          = submats[i]->ops->destroy;
      submats[i]->ops->destroy = MatDestroySubMatrix_SeqAIJ;
      submats[i]->factortype   = C->factortype;

      smat_i->id          = i;
      smat_i->nrqs        = nrqs;
      smat_i->nrqr        = nrqr;
      smat_i->rbuf1       = rbuf1;
      smat_i->rbuf2       = rbuf2;
      smat_i->rbuf3       = rbuf3;
      smat_i->sbuf2       = sbuf2;
      smat_i->req_source2 = req_source2;

      smat_i->sbuf1       = sbuf1;
      smat_i->ptr         = ptr;
      smat_i->tmp         = tmp;
      smat_i->ctr         = ctr;

      smat_i->pa           = pa;
      smat_i->req_size     = req_size;
      smat_i->req_source1  = req_source1;

      smat_i->allcolumns  = allcolumns[i];
      smat_i->singleis    = PETSC_FALSE;
      smat_i->row2proc    = row2proc[i];
      smat_i->rmap        = rmap[i];
      smat_i->cmap        = cmap[i];
    }

    if (!ismax) { /* Create dummy submats[0] for reuse struct subc */
      PetscCall(MatCreate(PETSC_COMM_SELF,&submats[0]));
      PetscCall(MatSetSizes(submats[0],0,0,PETSC_DETERMINE,PETSC_DETERMINE));
      PetscCall(MatSetType(submats[0],MATDUMMY));

      /* create struct Mat_SubSppt and attached it to submat */
      PetscCall(PetscNewLog(submats[0],&smat_i));
      submats[0]->data = (void*)smat_i;

      smat_i->destroy          = submats[0]->ops->destroy;
      submats[0]->ops->destroy = MatDestroySubMatrix_Dummy;
      submats[0]->factortype   = C->factortype;

      smat_i->id          = 0;
      smat_i->nrqs        = nrqs;
      smat_i->nrqr        = nrqr;
      smat_i->rbuf1       = rbuf1;
      smat_i->rbuf2       = rbuf2;
      smat_i->rbuf3       = rbuf3;
      smat_i->sbuf2       = sbuf2;
      smat_i->req_source2 = req_source2;

      smat_i->sbuf1       = sbuf1;
      smat_i->ptr         = ptr;
      smat_i->tmp         = tmp;
      smat_i->ctr         = ctr;

      smat_i->pa           = pa;
      smat_i->req_size     = req_size;
      smat_i->req_source1  = req_source1;

      smat_i->allcolumns  = PETSC_FALSE;
      smat_i->singleis    = PETSC_FALSE;
      smat_i->row2proc    = NULL;
      smat_i->rmap        = NULL;
      smat_i->cmap        = NULL;
    }

    if (ismax) PetscCall(PetscFree(lens[0]));
    PetscCall(PetscFree(lens));
    if (sbuf_aj) {
      PetscCall(PetscFree(sbuf_aj[0]));
      PetscCall(PetscFree(sbuf_aj));
    }

  } /* endof scall == MAT_INITIAL_MATRIX */

  /* Post recv matrix values */
  PetscCall(PetscObjectGetNewTag((PetscObject)C,&tag4));
  PetscCall(PetscMalloc1(nrqs,&rbuf4));
  PetscCall(PetscMalloc1(nrqs,&r_waits4));
  for (i=0; i<nrqs; ++i) {
    PetscCall(PetscMalloc1(rbuf2[i][0],&rbuf4[i]));
    PetscCallMPI(MPI_Irecv(rbuf4[i],rbuf2[i][0],MPIU_SCALAR,req_source2[i],tag4,comm,r_waits4+i));
  }

  /* Allocate sending buffers for a->a, and send them off */
  PetscCall(PetscMalloc1(nrqr,&sbuf_aa));
  for (i=0,j=0; i<nrqr; i++) j += req_size[i];
  if (nrqr) PetscCall(PetscMalloc1(j,&sbuf_aa[0]));
  for (i=1; i<nrqr; i++) sbuf_aa[i] = sbuf_aa[i-1] + req_size[i-1];

  PetscCall(PetscMalloc1(nrqr,&s_waits4));
  {
    PetscInt    nzA,nzB,*a_i = a->i,*b_i = b->i, *cworkB,lwrite;
    PetscInt    cstart = C->cmap->rstart,rstart = C->rmap->rstart,*bmap = c->garray;
    PetscInt    cend   = C->cmap->rend;
    PetscInt    *b_j   = b->j;
    PetscScalar *vworkA,*vworkB,*a_a,*b_a;

    PetscCall(MatSeqAIJGetArrayRead(A,(const PetscScalar**)&a_a));
    PetscCall(MatSeqAIJGetArrayRead(c->B,(const PetscScalar**)&b_a));
    for (i=0; i<nrqr; i++) {
      rbuf1_i   = rbuf1[i];
      sbuf_aa_i = sbuf_aa[i];
      ct1       = 2*rbuf1_i[0]+1;
      ct2       = 0;
      for (j=1,max1=rbuf1_i[0]; j<=max1; j++) {
        kmax = rbuf1_i[2*j];
        for (k=0; k<kmax; k++,ct1++) {
          row    = rbuf1_i[ct1] - rstart;
          nzA    = a_i[row+1] - a_i[row];
          nzB    = b_i[row+1] - b_i[row];
          ncols  = nzA + nzB;
          cworkB = b_j + b_i[row];
          vworkA = a_a + a_i[row];
          vworkB = b_a + b_i[row];

          /* load the column values for this row into vals*/
          vals = sbuf_aa_i+ct2;

          lwrite = 0;
          for (l=0; l<nzB; l++) {
            if ((bmap[cworkB[l]]) < cstart) vals[lwrite++] = vworkB[l];
          }
          for (l=0; l<nzA; l++) vals[lwrite++] = vworkA[l];
          for (l=0; l<nzB; l++) {
            if ((bmap[cworkB[l]]) >= cend) vals[lwrite++] = vworkB[l];
          }

          ct2 += ncols;
        }
      }
      PetscCallMPI(MPI_Isend(sbuf_aa_i,req_size[i],MPIU_SCALAR,req_source1[i],tag4,comm,s_waits4+i));
    }
    PetscCall(MatSeqAIJRestoreArrayRead(A,(const PetscScalar**)&a_a));
    PetscCall(MatSeqAIJRestoreArrayRead(c->B,(const PetscScalar**)&b_a));
  }

  /* Assemble the matrices */
  /* First assemble the local rows */
  for (i=0; i<ismax; i++) {
    row2proc_i = row2proc[i];
    subc      = (Mat_SeqAIJ*)submats[i]->data;
    imat_ilen = subc->ilen;
    imat_j    = subc->j;
    imat_i    = subc->i;
    imat_a    = subc->a;

    if (!allcolumns[i]) cmap_i = cmap[i];
    rmap_i = rmap[i];
    irow_i = irow[i];
    jmax   = nrow[i];
    for (j=0; j<jmax; j++) {
      row  = irow_i[j];
      proc = row2proc_i[j];
      if (proc == rank) {
        old_row = row;
#if defined(PETSC_USE_CTABLE)
        PetscCall(PetscTableFind(rmap_i,row+1,&row));
        row--;
#else
        row = rmap_i[row];
#endif
        ilen_row = imat_ilen[row];
        PetscCall(MatGetRow_MPIAIJ(C,old_row,&ncols,&cols,&vals));
        mat_i    = imat_i[row];
        mat_a    = imat_a + mat_i;
        mat_j    = imat_j + mat_i;
        if (!allcolumns[i]) {
          for (k=0; k<ncols; k++) {
#if defined(PETSC_USE_CTABLE)
            PetscCall(PetscTableFind(cmap_i,cols[k]+1,&tcol));
#else
            tcol = cmap_i[cols[k]];
#endif
            if (tcol) {
              *mat_j++ = tcol - 1;
              *mat_a++ = vals[k];
              ilen_row++;
            }
          }
        } else { /* allcolumns */
          for (k=0; k<ncols; k++) {
            *mat_j++ = cols[k];  /* global col index! */
            *mat_a++ = vals[k];
            ilen_row++;
          }
        }
        PetscCall(MatRestoreRow_MPIAIJ(C,old_row,&ncols,&cols,&vals));

        imat_ilen[row] = ilen_row;
      }
    }
  }

  /* Now assemble the off proc rows */
  PetscCallMPI(MPI_Waitall(nrqs,r_waits4,MPI_STATUSES_IGNORE));
  for (tmp2=0; tmp2<nrqs; tmp2++) {
    sbuf1_i = sbuf1[pa[tmp2]];
    jmax    = sbuf1_i[0];
    ct1     = 2*jmax + 1;
    ct2     = 0;
    rbuf2_i = rbuf2[tmp2];
    rbuf3_i = rbuf3[tmp2];
    rbuf4_i = rbuf4[tmp2];
    for (j=1; j<=jmax; j++) {
      is_no     = sbuf1_i[2*j-1];
      rmap_i    = rmap[is_no];
      if (!allcolumns[is_no]) cmap_i = cmap[is_no];
      subc      = (Mat_SeqAIJ*)submats[is_no]->data;
      imat_ilen = subc->ilen;
      imat_j    = subc->j;
      imat_i    = subc->i;
      imat_a    = subc->a;
      max1      = sbuf1_i[2*j];
      for (k=0; k<max1; k++,ct1++) {
        row = sbuf1_i[ct1];
#if defined(PETSC_USE_CTABLE)
        PetscCall(PetscTableFind(rmap_i,row+1,&row));
        row--;
#else
        row = rmap_i[row];
#endif
        ilen  = imat_ilen[row];
        mat_i = imat_i[row];
        mat_a = imat_a + mat_i;
        mat_j = imat_j + mat_i;
        max2  = rbuf2_i[ct1];
        if (!allcolumns[is_no]) {
          for (l=0; l<max2; l++,ct2++) {
#if defined(PETSC_USE_CTABLE)
            PetscCall(PetscTableFind(cmap_i,rbuf3_i[ct2]+1,&tcol));
#else
            tcol = cmap_i[rbuf3_i[ct2]];
#endif
            if (tcol) {
              *mat_j++ = tcol - 1;
              *mat_a++ = rbuf4_i[ct2];
              ilen++;
            }
          }
        } else { /* allcolumns */
          for (l=0; l<max2; l++,ct2++) {
            *mat_j++ = rbuf3_i[ct2]; /* same global column index of C */
            *mat_a++ = rbuf4_i[ct2];
            ilen++;
          }
        }
        imat_ilen[row] = ilen;
      }
    }
  }

  if (!iscsorted) { /* sort column indices of the rows */
    for (i=0; i<ismax; i++) {
      subc      = (Mat_SeqAIJ*)submats[i]->data;
      imat_j    = subc->j;
      imat_i    = subc->i;
      imat_a    = subc->a;
      imat_ilen = subc->ilen;

      if (allcolumns[i]) continue;
      jmax = nrow[i];
      for (j=0; j<jmax; j++) {
        mat_i = imat_i[j];
        mat_a = imat_a + mat_i;
        mat_j = imat_j + mat_i;
        PetscCall(PetscSortIntWithScalarArray(imat_ilen[j],mat_j,mat_a));
      }
    }
  }

  PetscCall(PetscFree(r_waits4));
  PetscCallMPI(MPI_Waitall(nrqr,s_waits4,MPI_STATUSES_IGNORE));
  PetscCall(PetscFree(s_waits4));

  /* Restore the indices */
  for (i=0; i<ismax; i++) {
    PetscCall(ISRestoreIndices(isrow[i],irow+i));
    if (!allcolumns[i]) {
      PetscCall(ISRestoreIndices(iscol[i],icol+i));
    }
  }

  for (i=0; i<ismax; i++) {
    PetscCall(MatAssemblyBegin(submats[i],MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(submats[i],MAT_FINAL_ASSEMBLY));
  }

  /* Destroy allocated memory */
  if (sbuf_aa) {
    PetscCall(PetscFree(sbuf_aa[0]));
    PetscCall(PetscFree(sbuf_aa));
  }
  PetscCall(PetscFree5(*(PetscInt***)&irow,*(PetscInt***)&icol,nrow,ncol,issorted));

  for (i=0; i<nrqs; ++i) {
    PetscCall(PetscFree(rbuf4[i]));
  }
  PetscCall(PetscFree(rbuf4));

  PetscCall(PetscFree4(row2proc,cmap,rmap,allcolumns));
  PetscFunctionReturn(0);
}

/*
 Permute A & B into C's *local* index space using rowemb,dcolemb for A and rowemb,ocolemb for B.
 Embeddings are supposed to be injections and the above implies that the range of rowemb is a subset
 of [0,m), dcolemb is in [0,n) and ocolemb is in [N-n).
 If pattern == DIFFERENT_NONZERO_PATTERN, C is preallocated according to A&B.
 After that B's columns are mapped into C's global column space, so that C is in the "disassembled"
 state, and needs to be "assembled" later by compressing B's column space.

 This function may be called in lieu of preallocation, so C should not be expected to be preallocated.
 Following this call, C->A & C->B have been created, even if empty.
 */
PetscErrorCode MatSetSeqMats_MPIAIJ(Mat C,IS rowemb,IS dcolemb,IS ocolemb,MatStructure pattern,Mat A,Mat B)
{
  /* If making this function public, change the error returned in this function away from _PLIB. */
  Mat_MPIAIJ     *aij;
  Mat_SeqAIJ     *Baij;
  PetscBool      seqaij,Bdisassembled;
  PetscInt       m,n,*nz,i,j,ngcol,col,rstart,rend,shift,count;
  PetscScalar    v;
  const PetscInt *rowindices,*colindices;

  PetscFunctionBegin;
  /* Check to make sure the component matrices (and embeddings) are compatible with C. */
  if (A) {
    PetscCall(PetscObjectBaseTypeCompare((PetscObject)A,MATSEQAIJ,&seqaij));
    PetscCheck(seqaij,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Diagonal matrix is of wrong type");
    if (rowemb) {
      PetscCall(ISGetLocalSize(rowemb,&m));
      PetscCheckFalse(m != A->rmap->n,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Row IS of size %" PetscInt_FMT " is incompatible with diag matrix row size %" PetscInt_FMT,m,A->rmap->n);
    } else {
      if (C->rmap->n != A->rmap->n) {
        SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Diag seq matrix is row-incompatible with the MPIAIJ matrix");
      }
    }
    if (dcolemb) {
      PetscCall(ISGetLocalSize(dcolemb,&n));
      PetscCheckFalse(n != A->cmap->n,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Diag col IS of size %" PetscInt_FMT " is incompatible with diag matrix col size %" PetscInt_FMT,n,A->cmap->n);
    } else {
      PetscCheckFalse(C->cmap->n != A->cmap->n,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Diag seq matrix is col-incompatible with the MPIAIJ matrix");
    }
  }
  if (B) {
    PetscCall(PetscObjectBaseTypeCompare((PetscObject)B,MATSEQAIJ,&seqaij));
    PetscCheck(seqaij,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Off-diagonal matrix is of wrong type");
    if (rowemb) {
      PetscCall(ISGetLocalSize(rowemb,&m));
      PetscCheckFalse(m != B->rmap->n,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Row IS of size %" PetscInt_FMT " is incompatible with off-diag matrix row size %" PetscInt_FMT,m,A->rmap->n);
    } else {
      if (C->rmap->n != B->rmap->n) {
        SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Off-diag seq matrix is row-incompatible with the MPIAIJ matrix");
      }
    }
    if (ocolemb) {
      PetscCall(ISGetLocalSize(ocolemb,&n));
      PetscCheckFalse(n != B->cmap->n,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Off-diag col IS of size %" PetscInt_FMT " is incompatible with off-diag matrix col size %" PetscInt_FMT,n,B->cmap->n);
    } else {
      PetscCheckFalse(C->cmap->N - C->cmap->n != B->cmap->n,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Off-diag seq matrix is col-incompatible with the MPIAIJ matrix");
    }
  }

  aij = (Mat_MPIAIJ*)C->data;
  if (!aij->A) {
    /* Mimic parts of MatMPIAIJSetPreallocation() */
    PetscCall(MatCreate(PETSC_COMM_SELF,&aij->A));
    PetscCall(MatSetSizes(aij->A,C->rmap->n,C->cmap->n,C->rmap->n,C->cmap->n));
    PetscCall(MatSetBlockSizesFromMats(aij->A,C,C));
    PetscCall(MatSetType(aij->A,MATSEQAIJ));
    PetscCall(PetscLogObjectParent((PetscObject)C,(PetscObject)aij->A));
  }
  if (A) {
    PetscCall(MatSetSeqMat_SeqAIJ(aij->A,rowemb,dcolemb,pattern,A));
  } else {
    PetscCall(MatSetUp(aij->A));
  }
  if (B) { /* Destroy the old matrix or the column map, depending on the sparsity pattern. */
    /*
      If pattern == DIFFERENT_NONZERO_PATTERN, we reallocate B and
      need to "disassemble" B -- convert it to using C's global indices.
      To insert the values we take the safer, albeit more expensive, route of MatSetValues().

      If pattern == SUBSET_NONZERO_PATTERN, we do not "disassemble" B and do not reallocate;
      we MatZeroValues(B) first, so there may be a bunch of zeros that, perhaps, could be compacted out.

      TODO: Put B's values into aij->B's aij structure in place using the embedding ISs?
      At least avoid calling MatSetValues() and the implied searches?
    */

    if (B && pattern == DIFFERENT_NONZERO_PATTERN) {
#if defined(PETSC_USE_CTABLE)
      PetscCall(PetscTableDestroy(&aij->colmap));
#else
      PetscCall(PetscFree(aij->colmap));
      /* A bit of a HACK: ideally we should deal with case aij->B all in one code block below. */
      if (aij->B) {
        PetscCall(PetscLogObjectMemory((PetscObject)C,-aij->B->cmap->n*sizeof(PetscInt)));
      }
#endif
      ngcol = 0;
      if (aij->lvec) {
        PetscCall(VecGetSize(aij->lvec,&ngcol));
      }
      if (aij->garray) {
        PetscCall(PetscFree(aij->garray));
        PetscCall(PetscLogObjectMemory((PetscObject)C,-ngcol*sizeof(PetscInt)));
      }
      PetscCall(VecDestroy(&aij->lvec));
      PetscCall(VecScatterDestroy(&aij->Mvctx));
    }
    if (aij->B && B && pattern == DIFFERENT_NONZERO_PATTERN) {
      PetscCall(MatDestroy(&aij->B));
    }
    if (aij->B && B && pattern == SUBSET_NONZERO_PATTERN) {
      PetscCall(MatZeroEntries(aij->B));
    }
  }
  Bdisassembled = PETSC_FALSE;
  if (!aij->B) {
    PetscCall(MatCreate(PETSC_COMM_SELF,&aij->B));
    PetscCall(PetscLogObjectParent((PetscObject)C,(PetscObject)aij->B));
    PetscCall(MatSetSizes(aij->B,C->rmap->n,C->cmap->N,C->rmap->n,C->cmap->N));
    PetscCall(MatSetBlockSizesFromMats(aij->B,B,B));
    PetscCall(MatSetType(aij->B,MATSEQAIJ));
    Bdisassembled = PETSC_TRUE;
  }
  if (B) {
    Baij = (Mat_SeqAIJ*)B->data;
    if (pattern == DIFFERENT_NONZERO_PATTERN) {
      PetscCall(PetscMalloc1(B->rmap->n,&nz));
      for (i=0; i<B->rmap->n; i++) {
        nz[i] = Baij->i[i+1] - Baij->i[i];
      }
      PetscCall(MatSeqAIJSetPreallocation(aij->B,0,nz));
      PetscCall(PetscFree(nz));
    }

    PetscCall(PetscLayoutGetRange(C->rmap,&rstart,&rend));
    shift = rend-rstart;
    count = 0;
    rowindices = NULL;
    colindices = NULL;
    if (rowemb) {
      PetscCall(ISGetIndices(rowemb,&rowindices));
    }
    if (ocolemb) {
      PetscCall(ISGetIndices(ocolemb,&colindices));
    }
    for (i=0; i<B->rmap->n; i++) {
      PetscInt row;
      row = i;
      if (rowindices) row = rowindices[i];
      for (j=Baij->i[i]; j<Baij->i[i+1]; j++) {
        col  = Baij->j[count];
        if (colindices) col = colindices[col];
        if (Bdisassembled && col>=rstart) col += shift;
        v    = Baij->a[count];
        PetscCall(MatSetValues(aij->B,1,&row,1,&col,&v,INSERT_VALUES));
        ++count;
      }
    }
    /* No assembly for aij->B is necessary. */
    /* FIXME: set aij->B's nonzerostate correctly. */
  } else {
    PetscCall(MatSetUp(aij->B));
  }
  C->preallocated  = PETSC_TRUE;
  C->was_assembled = PETSC_FALSE;
  C->assembled     = PETSC_FALSE;
   /*
      C will need to be assembled so that aij->B can be compressed into local form in MatSetUpMultiply_MPIAIJ().
      Furthermore, its nonzerostate will need to be based on that of aij->A's and aij->B's.
   */
  PetscFunctionReturn(0);
}

/*
  B uses local indices with column indices ranging between 0 and N-n; they  must be interpreted using garray.
 */
PetscErrorCode MatGetSeqMats_MPIAIJ(Mat C,Mat *A,Mat *B)
{
  Mat_MPIAIJ *aij = (Mat_MPIAIJ*)C->data;

  PetscFunctionBegin;
  PetscValidPointer(A,2);
  PetscValidPointer(B,3);
  /* FIXME: make sure C is assembled */
  *A = aij->A;
  *B = aij->B;
  /* Note that we don't incref *A and *B, so be careful! */
  PetscFunctionReturn(0);
}

/*
  Extract MPI submatrices encoded by pairs of IS that may live on subcomms of C.
  NOT SCALABLE due to the use of ISGetNonlocalIS() (see below).
*/
PetscErrorCode MatCreateSubMatricesMPI_MPIXAIJ(Mat C,PetscInt ismax,const IS isrow[],const IS iscol[],MatReuse scall,Mat *submat[],
                                               PetscErrorCode(*getsubmats_seq)(Mat,PetscInt,const IS[],const IS[],MatReuse,Mat**),
                                               PetscErrorCode(*getlocalmats)(Mat,Mat*,Mat*),
                                               PetscErrorCode(*setseqmat)(Mat,IS,IS,MatStructure,Mat),
                                               PetscErrorCode(*setseqmats)(Mat,IS,IS,IS,MatStructure,Mat,Mat))
{
  PetscMPIInt    size,flag;
  PetscInt       i,ii,cismax,ispar;
  Mat            *A,*B;
  IS             *isrow_p,*iscol_p,*cisrow,*ciscol,*ciscol_p;

  PetscFunctionBegin;
  if (!ismax) PetscFunctionReturn(0);

  for (i = 0, cismax = 0; i < ismax; ++i) {
    PetscCallMPI(MPI_Comm_compare(((PetscObject)isrow[i])->comm,((PetscObject)iscol[i])->comm,&flag));
    PetscCheckFalse(flag != MPI_IDENT,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Row and column index sets must have the same communicator");
    PetscCallMPI(MPI_Comm_size(((PetscObject)isrow[i])->comm, &size));
    if (size > 1) ++cismax;
  }

  /*
     If cismax is zero on all C's ranks, then and only then can we use purely sequential matrix extraction.
     ispar counts the number of parallel ISs across C's comm.
  */
  PetscCallMPI(MPIU_Allreduce(&cismax,&ispar,1,MPIU_INT,MPI_MAX,PetscObjectComm((PetscObject)C)));
  if (!ispar) { /* Sequential ISs only across C's comm, so can call the sequential matrix extraction subroutine. */
    PetscCall((*getsubmats_seq)(C,ismax,isrow,iscol,scall,submat));
    PetscFunctionReturn(0);
  }

  /* if (ispar) */
  /*
    Construct the "complements" -- the off-processor indices -- of the iscol ISs for parallel ISs only.
    These are used to extract the off-diag portion of the resulting parallel matrix.
    The row IS for the off-diag portion is the same as for the diag portion,
    so we merely alias (without increfing) the row IS, while skipping those that are sequential.
  */
  PetscCall(PetscMalloc2(cismax,&cisrow,cismax,&ciscol));
  PetscCall(PetscMalloc1(cismax,&ciscol_p));
  for (i = 0, ii = 0; i < ismax; ++i) {
    PetscCallMPI(MPI_Comm_size(((PetscObject)isrow[i])->comm,&size));
    if (size > 1) {
      /*
         TODO: This is the part that's ***NOT SCALABLE***.
         To fix this we need to extract just the indices of C's nonzero columns
         that lie on the intersection of isrow[i] and ciscol[ii] -- the nonlocal
         part of iscol[i] -- without actually computing ciscol[ii]. This also has
         to be done without serializing on the IS list, so, most likely, it is best
         done by rewriting MatCreateSubMatrices_MPIAIJ() directly.
      */
      PetscCall(ISGetNonlocalIS(iscol[i],&(ciscol[ii])));
      /* Now we have to
         (a) make sure ciscol[ii] is sorted, since, even if the off-proc indices
             were sorted on each rank, concatenated they might no longer be sorted;
         (b) Use ISSortPermutation() to construct ciscol_p, the mapping from the
             indices in the nondecreasing order to the original index positions.
         If ciscol[ii] is strictly increasing, the permutation IS is NULL.
      */
      PetscCall(ISSortPermutation(ciscol[ii],PETSC_FALSE,ciscol_p+ii));
      PetscCall(ISSort(ciscol[ii]));
      ++ii;
    }
  }
  PetscCall(PetscMalloc2(ismax,&isrow_p,ismax,&iscol_p));
  for (i = 0, ii = 0; i < ismax; ++i) {
    PetscInt       j,issize;
    const PetscInt *indices;

    /*
       Permute the indices into a nondecreasing order. Reject row and col indices with duplicates.
     */
    PetscCall(ISSortPermutation(isrow[i],PETSC_FALSE,isrow_p+i));
    PetscCall(ISSort(isrow[i]));
    PetscCall(ISGetLocalSize(isrow[i],&issize));
    PetscCall(ISGetIndices(isrow[i],&indices));
    for (j = 1; j < issize; ++j) {
      PetscCheckFalse(indices[j] == indices[j-1],PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Repeated indices in row IS %" PetscInt_FMT ": indices at %" PetscInt_FMT " and %" PetscInt_FMT " are both %" PetscInt_FMT,i,j-1,j,indices[j]);
    }
    PetscCall(ISRestoreIndices(isrow[i],&indices));
    PetscCall(ISSortPermutation(iscol[i],PETSC_FALSE,iscol_p+i));
    PetscCall(ISSort(iscol[i]));
    PetscCall(ISGetLocalSize(iscol[i],&issize));
    PetscCall(ISGetIndices(iscol[i],&indices));
    for (j = 1; j < issize; ++j) {
      PetscCheckFalse(indices[j-1] == indices[j],PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Repeated indices in col IS %" PetscInt_FMT ": indices at %" PetscInt_FMT " and %" PetscInt_FMT " are both %" PetscInt_FMT,i,j-1,j,indices[j]);
    }
    PetscCall(ISRestoreIndices(iscol[i],&indices));
    PetscCallMPI(MPI_Comm_size(((PetscObject)isrow[i])->comm,&size));
    if (size > 1) {
      cisrow[ii] = isrow[i];
      ++ii;
    }
  }
  /*
    Allocate the necessary arrays to hold the resulting parallel matrices as well as the intermediate
    array of sequential matrices underlying the resulting parallel matrices.
    Which arrays to allocate is based on the value of MatReuse scall and whether ISs are sorted and/or
    contain duplicates.

    There are as many diag matrices as there are original index sets. There are only as many parallel
    and off-diag matrices, as there are parallel (comm size > 1) index sets.

    ARRAYS that can hold Seq matrices get allocated in any event -- either here or by getsubmats_seq():
    - If the array of MPI matrices already exists and is being reused, we need to allocate the array
      and extract the underlying seq matrices into it to serve as placeholders, into which getsubmats_seq
      will deposite the extracted diag and off-diag parts. Thus, we allocate the A&B arrays and fill them
      with A[i] and B[ii] extracted from the corresponding MPI submat.
    - However, if the rows, A's column indices or B's column indices are not sorted, the extracted A[i] & B[ii]
      will have a different order from what getsubmats_seq expects.  To handle this case -- indicated
      by a nonzero isrow_p[i], iscol_p[i], or ciscol_p[ii] -- we duplicate A[i] --> AA[i], B[ii] --> BB[ii]
      (retrieve composed AA[i] or BB[ii]) and reuse them here. AA[i] and BB[ii] are then used to permute its
      values into A[i] and B[ii] sitting inside the corresponding submat.
    - If no reuse is taking place then getsubmats_seq will allocate the A&B arrays and create the corresponding
      A[i], B[ii], AA[i] or BB[ii] matrices.
  */
  /* Parallel matrix array is allocated here only if no reuse is taking place. If reused, it is passed in by the caller. */
  if (scall == MAT_INITIAL_MATRIX) {
    PetscCall(PetscMalloc1(ismax,submat));
  }

  /* Now obtain the sequential A and B submatrices separately. */
  /* scall=MAT_REUSE_MATRIX is not handled yet, because getsubmats_seq() requires reuse of A and B */
  PetscCall((*getsubmats_seq)(C,ismax,isrow,iscol,MAT_INITIAL_MATRIX,&A));
  PetscCall((*getsubmats_seq)(C,cismax,cisrow,ciscol,MAT_INITIAL_MATRIX,&B));

  /*
    If scall == MAT_REUSE_MATRIX AND the permutations are NULL, we are done, since the sequential
    matrices A & B have been extracted directly into the parallel matrices containing them, or
    simply into the sequential matrix identical with the corresponding A (if size == 1).
    Note that in that case colmap doesn't need to be rebuilt, since the matrices are expected
    to have the same sparsity pattern.
    Otherwise, A and/or B have to be properly embedded into C's index spaces and the correct colmap
    must be constructed for C. This is done by setseqmat(s).
  */
  for (i = 0, ii = 0; i < ismax; ++i) {
    /*
       TODO: cache ciscol, permutation ISs and maybe cisrow? What about isrow & iscol?
       That way we can avoid sorting and computing permutations when reusing.
       To this end:
        - remove the old cache, if it exists, when extracting submatrices with MAT_INITIAL_MATRIX
        - if caching arrays to hold the ISs, make and compose a container for them so that it can
          be destroyed upon destruction of C (use PetscContainerUserDestroy() to clear out the contents).
    */
    MatStructure pattern = DIFFERENT_NONZERO_PATTERN;

    PetscCallMPI(MPI_Comm_size(((PetscObject)isrow[i])->comm,&size));
    /* Construct submat[i] from the Seq pieces A (and B, if necessary). */
    if (size > 1) {
      if (scall == MAT_INITIAL_MATRIX) {
        PetscCall(MatCreate(((PetscObject)isrow[i])->comm,(*submat)+i));
        PetscCall(MatSetSizes((*submat)[i],A[i]->rmap->n,A[i]->cmap->n,PETSC_DETERMINE,PETSC_DETERMINE));
        PetscCall(MatSetType((*submat)[i],MATMPIAIJ));
        PetscCall(PetscLayoutSetUp((*submat)[i]->rmap));
        PetscCall(PetscLayoutSetUp((*submat)[i]->cmap));
      }
      /*
        For each parallel isrow[i], insert the extracted sequential matrices into the parallel matrix.
      */
      {
        Mat AA = A[i],BB = B[ii];

        if (AA || BB) {
          PetscCall(setseqmats((*submat)[i],isrow_p[i],iscol_p[i],ciscol_p[ii],pattern,AA,BB));
          PetscCall(MatAssemblyBegin((*submat)[i],MAT_FINAL_ASSEMBLY));
          PetscCall(MatAssemblyEnd((*submat)[i],MAT_FINAL_ASSEMBLY));
        }
        PetscCall(MatDestroy(&AA));
      }
      PetscCall(ISDestroy(ciscol+ii));
      PetscCall(ISDestroy(ciscol_p+ii));
      ++ii;
    } else { /* if (size == 1) */
      if (scall == MAT_REUSE_MATRIX) {
        PetscCall(MatDestroy(&(*submat)[i]));
      }
      if (isrow_p[i] || iscol_p[i]) {
        PetscCall(MatDuplicate(A[i],MAT_DO_NOT_COPY_VALUES,(*submat)+i));
        PetscCall(setseqmat((*submat)[i],isrow_p[i],iscol_p[i],pattern,A[i]));
        /* Otherwise A is extracted straight into (*submats)[i]. */
        /* TODO: Compose A[i] on (*submat([i] for future use, if ((isrow_p[i] || iscol_p[i]) && MAT_INITIAL_MATRIX). */
        PetscCall(MatDestroy(A+i));
      } else (*submat)[i] = A[i];
    }
    PetscCall(ISDestroy(&isrow_p[i]));
    PetscCall(ISDestroy(&iscol_p[i]));
  }
  PetscCall(PetscFree2(cisrow,ciscol));
  PetscCall(PetscFree2(isrow_p,iscol_p));
  PetscCall(PetscFree(ciscol_p));
  PetscCall(PetscFree(A));
  PetscCall(MatDestroySubMatrices(cismax,&B));
  PetscFunctionReturn(0);
}

PetscErrorCode MatCreateSubMatricesMPI_MPIAIJ(Mat C,PetscInt ismax,const IS isrow[],const IS iscol[],MatReuse scall,Mat *submat[])
{
  PetscFunctionBegin;
  PetscCall(MatCreateSubMatricesMPI_MPIXAIJ(C,ismax,isrow,iscol,scall,submat,MatCreateSubMatrices_MPIAIJ,MatGetSeqMats_MPIAIJ,MatSetSeqMat_SeqAIJ,MatSetSeqMats_MPIAIJ));
  PetscFunctionReturn(0);
}
