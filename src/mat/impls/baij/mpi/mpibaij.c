#ifndef lint
static char vcid[] = "$Id: mpibaij.c,v 1.13 1996/07/11 04:06:08 balay Exp balay $";
#endif

#include "mpibaij.h"


#include "draw.h"
#include "pinclude/pviewer.h"

extern int MatSetUpMultiply_MPIBAIJ(Mat); 
extern int DisAssemble_MPIBAIJ(Mat);
extern int MatIncreaseOverlap_MPIBAIJ(Mat,int,IS *,int);
extern int MatGetSubMatrices_MPIBAIJ(Mat,int,IS *,IS *,MatGetSubMatrixCall,Mat **);

/* local utility routine that creates a mapping from the global column 
number to the local number in the off-diagonal part of the local 
storage of the matrix.  This is done in a non scable way since the 
length of colmap equals the global matrix length. 
*/
static int CreateColmap_MPIBAIJ_Private(Mat mat)
{
  Mat_MPIBAIJ *baij = (Mat_MPIBAIJ *) mat->data;
  Mat_SeqBAIJ *B = (Mat_SeqBAIJ*) baij->B->data;
  int        nbs = B->nbs,i;

  baij->colmap = (int *) PetscMalloc(baij->Nbs*sizeof(int));CHKPTRQ(baij->colmap);
  PLogObjectMemory(mat,baij->Nbs*sizeof(int));
  PetscMemzero(baij->colmap,baij->Nbs*sizeof(int));
  for ( i=0; i<nbs; i++ ) baij->colmap[baij->garray[i]] = i;
  return 0;
}


static int MatGetReordering_MPIBAIJ(Mat mat,MatReordering type,IS *rperm,IS *cperm)
{
  Mat_MPIBAIJ *baij = (Mat_MPIBAIJ *) mat->data;
  int         ierr;
  if (baij->size == 1) {
    ierr = MatGetReordering(baij->A,type,rperm,cperm); CHKERRQ(ierr);
  } else SETERRQ(1,"MatGetReordering_MPIBAIJ:not supported in parallel");
  return 0;
}

static int MatSetValues_MPIBAIJ(Mat mat,int m,int *im,int n,int *in,Scalar *v,InsertMode addv)
{
  Mat_MPIBAIJ *baij = (Mat_MPIBAIJ *) mat->data;
  Scalar      value;
  int         ierr,i,j, rstart = baij->rstart, rend = baij->rend;
  int         cstart = baij->cstart, cend = baij->cend,row,col;
  int         roworiented = baij->roworiented,rstart_orig,rend_orig;
  int         cstart_orig,cend_orig,bs=baij->bs;

  if (baij->insertmode != NOT_SET_VALUES && baij->insertmode != addv) {
    SETERRQ(1,"MatSetValues_MPIBAIJ:Cannot mix inserts and adds");
  }
  baij->insertmode = addv;
  rstart_orig = rstart*bs;
  rend_orig   = rend*bs;
  cstart_orig = cstart*bs;
  cend_orig   = cend*bs;
  for ( i=0; i<m; i++ ) {
    if (im[i] < 0) SETERRQ(1,"MatSetValues_MPIBAIJ:Negative row");
    if (im[i] >= baij->M) SETERRQ(1,"MatSetValues_MPIBAIJ:Row too large");
    if (im[i] >= rstart_orig && im[i] < rend_orig) {
      row = im[i] - rstart_orig;
      for ( j=0; j<n; j++ ) {
        if (in[j] < 0) SETERRQ(1,"MatSetValues_MPIBAIJ:Negative column");
        if (in[j] >= baij->N) SETERRQ(1,"MatSetValues_MPIBAIJ:Col too large");
        if (in[j] >= cstart_orig && in[j] < cend_orig){
          col = in[j] - cstart_orig;
          if (roworiented) value = v[i*n+j]; else value = v[i+j*m];
          ierr = MatSetValues(baij->A,1,&row,1,&col,&value,addv);CHKERRQ(ierr);
        }
        else {
          if (mat->was_assembled) {
            if (!baij->colmap) {ierr = CreateColmap_MPIBAIJ_Private(mat);CHKERRQ(ierr);}
            col = baij->colmap[in[j]/bs] +in[j]%bs;
            if (col < 0 && !((Mat_SeqBAIJ*)(baij->A->data))->nonew) {
              ierr = DisAssemble_MPIBAIJ(mat); CHKERRQ(ierr); 
              col =  in[j];              
            }
          }
          else col = in[j];
          if (roworiented) value = v[i*n+j]; else value = v[i+j*m];
          ierr = MatSetValues(baij->B,1,&row,1,&col,&value,addv);CHKERRQ(ierr);
        }
      }
    } 
    else {
      if (roworiented) {
        ierr = StashValues_Private(&baij->stash,im[i],n,in,v+i*n,addv);CHKERRQ(ierr);
      }
      else {
        row = im[i];
        for ( j=0; j<n; j++ ) {
          ierr = StashValues_Private(&baij->stash,row,1,in+j,v+i+j*m,addv);CHKERRQ(ierr);
        }
      }
    }
  }
  return 0;
}

static int MatGetValues_MPIBAIJ(Mat mat,int m,int *idxm,int n,int *idxn,Scalar *v)
{
  Mat_MPIBAIJ *baij = (Mat_MPIBAIJ *) mat->data;
  int        bs=baij->bs,ierr,i,j, bsrstart = baij->rstart*bs, bsrend = baij->rend*bs;
  int        bscstart = baij->cstart*bs, bscend = baij->cend*bs,row,col;

  for ( i=0; i<m; i++ ) {
    if (idxm[i] < 0) SETERRQ(1,"MatGetValues_MPIBAIJ:Negative row");
    if (idxm[i] >= baij->M) SETERRQ(1,"MatGetValues_MPIBAIJ:Row too large");
    if (idxm[i] >= bsrstart && idxm[i] < bsrend) {
      row = idxm[i] - bsrstart;
      for ( j=0; j<n; j++ ) {
        if (idxn[j] < 0) SETERRQ(1,"MatGetValues_MPIBAIJ:Negative column");
        if (idxn[j] >= baij->N) SETERRQ(1,"MatGetValues_MPIBAIJ:Col too large");
        if (idxn[j] >= bscstart && idxn[j] < bscend){
          col = idxn[j] - bscstart;
          ierr = MatGetValues(baij->A,1,&row,1,&col,v+i*n+j); CHKERRQ(ierr);
        }
        else {
          if (!baij->colmap) {ierr = CreateColmap_MPIBAIJ_Private(mat);CHKERRQ(ierr);} 
          if ( baij->garray[baij->colmap[idxn[j]/bs]] != idxn[j]/bs ) *(v+i*n+j) = 0.0;
          else {
            col = baij->colmap[idxn[j]/bs]*bs + idxn[j]%bs;
            ierr = MatGetValues(baij->B,1,&row,1,&col,v+i*n+j); CHKERRQ(ierr);
          } 
        }
      }
    } 
    else {
      SETERRQ(1,"MatGetValues_MPIBAIJ:Only local values currently supported");
    }
  }
  return 0;
}

static int MatNorm_MPIBAIJ(Mat mat,NormType type,double *norm)
{
  Mat_MPIBAIJ *baij = (Mat_MPIBAIJ *) mat->data;
  Mat_SeqBAIJ *amat = (Mat_SeqBAIJ*) baij->A->data, *bmat = (Mat_SeqBAIJ*) baij->B->data;
  int        ierr, i,bs2=baij->bs2;
  double     sum = 0.0;
  Scalar     *v;

  if (baij->size == 1) {
    ierr =  MatNorm(baij->A,type,norm); CHKERRQ(ierr);
  } else {
    if (type == NORM_FROBENIUS) {
      v = amat->a;
      for (i=0; i<amat->nz*bs2; i++ ) {
#if defined(PETSC_COMPLEX)
        sum += real(conj(*v)*(*v)); v++;
#else
        sum += (*v)*(*v); v++;
#endif
      }
      v = bmat->a;
      for (i=0; i<bmat->nz*bs2; i++ ) {
#if defined(PETSC_COMPLEX)
        sum += real(conj(*v)*(*v)); v++;
#else
        sum += (*v)*(*v); v++;
#endif
      }
      MPI_Allreduce(&sum,norm,1,MPI_DOUBLE,MPI_SUM,mat->comm);
      *norm = sqrt(*norm);
    }
    else
      SETERRQ(1,"MatNorm_SeqBAIJ:No support for this norm yet");
  }
  return 0;
}

static int MatAssemblyBegin_MPIBAIJ(Mat mat,MatAssemblyType mode)
{ 
  Mat_MPIBAIJ  *baij = (Mat_MPIBAIJ *) mat->data;
  MPI_Comm    comm = mat->comm;
  int         size = baij->size, *owners = baij->rowners,bs=baij->bs;
  int         rank = baij->rank,tag = mat->tag, *owner,*starts,count,ierr;
  MPI_Request *send_waits,*recv_waits;
  int         *nprocs,i,j,idx,*procs,nsends,nreceives,nmax,*work;
  InsertMode  addv;
  Scalar      *rvalues,*svalues;

  /* make sure all processors are either in INSERTMODE or ADDMODE */
  MPI_Allreduce(&baij->insertmode,&addv,1,MPI_INT,MPI_BOR,comm);
  if (addv == (ADD_VALUES|INSERT_VALUES)) {
    SETERRQ(1,"MatAssemblyBegin_MPIBAIJ:Some processors inserted others added");
  }
  baij->insertmode = addv; /* in case this processor had no cache */

  /*  first count number of contributors to each processor */
  nprocs = (int *) PetscMalloc( 2*size*sizeof(int) ); CHKPTRQ(nprocs);
  PetscMemzero(nprocs,2*size*sizeof(int)); procs = nprocs + size;
  owner = (int *) PetscMalloc( (baij->stash.n+1)*sizeof(int) ); CHKPTRQ(owner);
  for ( i=0; i<baij->stash.n; i++ ) {
    idx = baij->stash.idx[i];
    for ( j=0; j<size; j++ ) {
      if (idx >= owners[j]*bs && idx < owners[j+1]*bs) {
        nprocs[j]++; procs[j] = 1; owner[i] = j; break;
      }
    }
  }
  nsends = 0;  for ( i=0; i<size; i++ ) { nsends += procs[i];} 

  /* inform other processors of number of messages and max length*/
  work = (int *) PetscMalloc( size*sizeof(int) ); CHKPTRQ(work);
  MPI_Allreduce(procs, work,size,MPI_INT,MPI_SUM,comm);
  nreceives = work[rank]; 
  MPI_Allreduce( nprocs, work,size,MPI_INT,MPI_MAX,comm);
  nmax = work[rank];
  PetscFree(work);

  /* post receives: 
       1) each message will consist of ordered pairs 
     (global index,value) we store the global index as a double 
     to simplify the message passing. 
       2) since we don't know how long each individual message is we 
     allocate the largest needed buffer for each receive. Potentially 
     this is a lot of wasted space.


       This could be done better.
  */
  rvalues = (Scalar *) PetscMalloc(3*(nreceives+1)*(nmax+1)*sizeof(Scalar));
  CHKPTRQ(rvalues);
  recv_waits = (MPI_Request *) PetscMalloc((nreceives+1)*sizeof(MPI_Request));
  CHKPTRQ(recv_waits);
  for ( i=0; i<nreceives; i++ ) {
    MPI_Irecv(rvalues+3*nmax*i,3*nmax,MPIU_SCALAR,MPI_ANY_SOURCE,tag,
              comm,recv_waits+i);
  }

  /* do sends:
      1) starts[i] gives the starting index in svalues for stuff going to 
         the ith processor
  */
  svalues = (Scalar *) PetscMalloc(3*(baij->stash.n+1)*sizeof(Scalar));CHKPTRQ(svalues);
  send_waits = (MPI_Request *) PetscMalloc( (nsends+1)*sizeof(MPI_Request));
  CHKPTRQ(send_waits);
  starts = (int *) PetscMalloc( size*sizeof(int) ); CHKPTRQ(starts);
  starts[0] = 0; 
  for ( i=1; i<size; i++ ) { starts[i] = starts[i-1] + nprocs[i-1];} 
  for ( i=0; i<baij->stash.n; i++ ) {
    svalues[3*starts[owner[i]]]       = (Scalar)  baij->stash.idx[i];
    svalues[3*starts[owner[i]]+1]     = (Scalar)  baij->stash.idy[i];
    svalues[3*(starts[owner[i]]++)+2] =  baij->stash.array[i];
  }
  PetscFree(owner);
  starts[0] = 0;
  for ( i=1; i<size; i++ ) { starts[i] = starts[i-1] + nprocs[i-1];} 
  count = 0;
  for ( i=0; i<size; i++ ) {
    if (procs[i]) {
      MPI_Isend(svalues+3*starts[i],3*nprocs[i],MPIU_SCALAR,i,tag,
                comm,send_waits+count++);
    }
  }
  PetscFree(starts); PetscFree(nprocs);

  /* Free cache space */
  PLogInfo(0,"[%d]MatAssemblyBegin_MPIBAIJ:Number of off processor values %d\n",rank,baij->stash.n);
  ierr = StashDestroy_Private(&baij->stash); CHKERRQ(ierr);

  baij->svalues    = svalues;    baij->rvalues    = rvalues;
  baij->nsends     = nsends;     baij->nrecvs     = nreceives;
  baij->send_waits = send_waits; baij->recv_waits = recv_waits;
  baij->rmax       = nmax;

  return 0;
}


static int MatAssemblyEnd_MPIBAIJ(Mat mat,MatAssemblyType mode)
{ 
  Mat_MPIBAIJ *baij = (Mat_MPIBAIJ *) mat->data;
  MPI_Status  *send_status,recv_status;
  int         imdex,nrecvs = baij->nrecvs, count = nrecvs, i, n, ierr;
  int         bs=baij->bs,row,col,other_disassembled;
  Scalar      *values,val;
  InsertMode  addv = baij->insertmode;

  /*  wait on receives */
  while (count) {
    MPI_Waitany(nrecvs,baij->recv_waits,&imdex,&recv_status);
    /* unpack receives into our local space */
    values = baij->rvalues + 3*imdex*baij->rmax;
    MPI_Get_count(&recv_status,MPIU_SCALAR,&n);
    n = n/3;
    for ( i=0; i<n; i++ ) {
      row = (int) PetscReal(values[3*i]) - baij->rstart*bs;
      col = (int) PetscReal(values[3*i+1]);
      val = values[3*i+2];
      if (col >= baij->cstart*bs && col < baij->cend*bs) {
        col -= baij->cstart*bs;
        MatSetValues(baij->A,1,&row,1,&col,&val,addv);
      } 
      else {
        if (mat->was_assembled) {
          if (!baij->colmap) {ierr = CreateColmap_MPIBAIJ_Private(mat); CHKERRQ(ierr);}
          col = baij->colmap[col/bs]*bs + col%bs;
          if (col < 0  && !((Mat_SeqBAIJ*)(baij->A->data))->nonew) {
            ierr = DisAssemble_MPIBAIJ(mat); CHKERRQ(ierr); 
            col = (int) PetscReal(values[3*i+1]);
          }
        }
        MatSetValues(baij->B,1,&row,1,&col,&val,addv);
      }
    }
    count--;
  }
  PetscFree(baij->recv_waits); PetscFree(baij->rvalues);
 
  /* wait on sends */
  if (baij->nsends) {
    send_status = (MPI_Status *) PetscMalloc(baij->nsends*sizeof(MPI_Status));
    CHKPTRQ(send_status);
    MPI_Waitall(baij->nsends,baij->send_waits,send_status);
    PetscFree(send_status);
  }
  PetscFree(baij->send_waits); PetscFree(baij->svalues);

  baij->insertmode = NOT_SET_VALUES;
  ierr = MatAssemblyBegin(baij->A,mode); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(baij->A,mode); CHKERRQ(ierr);

  /* determine if any processor has disassembled, if so we must 
     also disassemble ourselfs, in order that we may reassemble. */
  MPI_Allreduce(&mat->was_assembled,&other_disassembled,1,MPI_INT,MPI_PROD,mat->comm);
  if (mat->was_assembled && !other_disassembled) {
    ierr = DisAssemble_MPIBAIJ(mat); CHKERRQ(ierr);
  }

  if (!mat->was_assembled && mode == MAT_FINAL_ASSEMBLY) {
    ierr = MatSetUpMultiply_MPIBAIJ(mat); CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(baij->B,mode); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(baij->B,mode); CHKERRQ(ierr);

  if (baij->rowvalues) {PetscFree(baij->rowvalues); baij->rowvalues = 0;}
  return 0;
}

static int MatView_MPIBAIJ_Binary(Mat mat,Viewer viewer)
{
  Mat_MPIBAIJ  *baij = (Mat_MPIBAIJ *) mat->data;
  int          ierr;

  if (baij->size == 1) {
    ierr = MatView(baij->A,viewer); CHKERRQ(ierr);
  }
  else SETERRQ(1,"MatView_MPIBAIJ_Binary:Only uniprocessor output supported");
  return 0;
}

static int MatView_MPIBAIJ_ASCIIorDraworMatlab(Mat mat,Viewer viewer)
{
  Mat_MPIBAIJ  *baij = (Mat_MPIBAIJ *) mat->data;
  int          ierr, format,rank,bs=baij->bs;
  FILE         *fd;
  ViewerType   vtype;

  ierr = ViewerGetType(viewer,&vtype); CHKERRQ(ierr);
  if (vtype  == ASCII_FILES_VIEWER || vtype == ASCII_FILE_VIEWER) { 
    ierr = ViewerGetFormat(viewer,&format);
    if (format == ASCII_FORMAT_INFO_DETAILED) {
      int nz, nzalloc, mem;
      MPI_Comm_rank(mat->comm,&rank);
      ierr = ViewerASCIIGetPointer(viewer,&fd); CHKERRQ(ierr);
      ierr = MatGetInfo(mat,MAT_LOCAL,&nz,&nzalloc,&mem); 
      PetscSequentialPhaseBegin(mat->comm,1);
      fprintf(fd,"[%d] Local rows %d nz %d nz alloced %d bs %d mem %d\n",
              rank,baij->m,nz*bs,nzalloc*bs,baij->bs,mem);       
      ierr = MatGetInfo(baij->A,MAT_LOCAL,&nz,&nzalloc,&mem); 
      fprintf(fd,"[%d] on-diagonal part: nz %d \n",rank,nz*bs);
      ierr = MatGetInfo(baij->B,MAT_LOCAL,&nz,&nzalloc,&mem); 
      fprintf(fd,"[%d] off-diagonal part: nz %d \n",rank,nz*bs); 
      fflush(fd);
      PetscSequentialPhaseEnd(mat->comm,1);
      ierr = VecScatterView(baij->Mvctx,viewer); CHKERRQ(ierr);
      return 0; 
    }
    else if (format == ASCII_FORMAT_INFO) {
      return 0;
    }
  }

  if (vtype == DRAW_VIEWER) {
    Draw       draw;
    PetscTruth isnull;
    ierr = ViewerDrawGetDraw(viewer,&draw); CHKERRQ(ierr);
    ierr = DrawIsNull(draw,&isnull); CHKERRQ(ierr); if (isnull) return 0;
  }

  if (vtype == ASCII_FILE_VIEWER) {
    ierr = ViewerASCIIGetPointer(viewer,&fd); CHKERRQ(ierr);
    PetscSequentialPhaseBegin(mat->comm,1);
    fprintf(fd,"[%d] rows %d starts %d ends %d cols %d starts %d ends %d\n",
           baij->rank,baij->m,baij->rstart*bs,baij->rend*bs,baij->n,
            baij->cstart*bs,baij->cend*bs);
    ierr = MatView(baij->A,viewer); CHKERRQ(ierr);
    ierr = MatView(baij->B,viewer); CHKERRQ(ierr);
    fflush(fd);
    PetscSequentialPhaseEnd(mat->comm,1);
  }
  else {
    int size = baij->size;
    rank = baij->rank;
    if (size == 1) {
      ierr = MatView(baij->A,viewer); CHKERRQ(ierr);
    }
    else {
      /* assemble the entire matrix onto first processor. */
      Mat         A;
      Mat_SeqBAIJ *Aloc;
      int         M = baij->M, N = baij->N,*ai,*aj,row,col,i,j,k,*rvals;
      int         mbs=baij->mbs;
      Scalar      *a;

      if (!rank) {
        ierr = MatCreateMPIBAIJ(mat->comm,baij->bs,M,N,M,N,0,PETSC_NULL,0,PETSC_NULL,&A);
        CHKERRQ(ierr);
      }
      else {
        ierr = MatCreateMPIBAIJ(mat->comm,baij->bs,0,0,M,N,0,PETSC_NULL,0,PETSC_NULL,&A);
        CHKERRQ(ierr);
      }
      PLogObjectParent(mat,A);

      /* copy over the A part */
      Aloc = (Mat_SeqBAIJ*) baij->A->data;
      ai = Aloc->i; aj = Aloc->j; a = Aloc->a;
      row = baij->rstart;
      rvals = (int *) PetscMalloc(bs*sizeof(int)); CHKPTRQ(rvals);

      for ( i=0; i<mbs; i++ ) {
        rvals[0] = bs*(baij->rstart + i);
        for ( j=1; j<bs; j++ ) { rvals[j] = rvals[j-1] + 1; }
        for ( j=ai[i]; j<ai[i+1]; j++ ) {
          col = (baij->cstart+aj[j])*bs;
          for (k=0; k<bs; k++ ) {
            ierr = MatSetValues(A,bs,rvals,1,&col,a,INSERT_VALUES);CHKERRQ(ierr);
            col++; a += bs;
          }
        }
      } 
      /* copy over the B part */
      Aloc = (Mat_SeqBAIJ*) baij->B->data;
      ai = Aloc->i; aj = Aloc->j; a = Aloc->a;
      row = baij->rstart*bs;
      for ( i=0; i<mbs; i++ ) {
        rvals[0] = bs*(baij->rstart + i);
        for ( j=1; j<bs; j++ ) { rvals[j] = rvals[j-1] + 1; }
        for ( j=ai[i]; j<ai[i+1]; j++ ) {
          col = baij->garray[aj[j]]*bs;
          for (k=0; k<bs; k++ ) { 
            ierr = MatSetValues(A,bs,rvals,1,&col,a,INSERT_VALUES);CHKERRQ(ierr);
            col++; a += bs;
          }
        }
      } 
      PetscFree(rvals);
      ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
      ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
      if (!rank) {
        ierr = MatView(((Mat_MPIBAIJ*)(A->data))->A,viewer); CHKERRQ(ierr);
      }
      ierr = MatDestroy(A); CHKERRQ(ierr);
    }
  }
  return 0;
}



static int MatView_MPIBAIJ(PetscObject obj,Viewer viewer)
{
  Mat         mat = (Mat) obj;
  int         ierr;
  ViewerType  vtype;
 
  ierr = ViewerGetType(viewer,&vtype); CHKERRQ(ierr);
  if (vtype == ASCII_FILE_VIEWER || vtype == ASCII_FILES_VIEWER ||
      vtype == DRAW_VIEWER       || vtype == MATLAB_VIEWER) { 
    ierr = MatView_MPIBAIJ_ASCIIorDraworMatlab(mat,viewer); CHKERRQ(ierr);
  }
  else if (vtype == BINARY_FILE_VIEWER) {
    return MatView_MPIBAIJ_Binary(mat,viewer);
  }
  return 0;
}

static int MatDestroy_MPIBAIJ(PetscObject obj)
{
  Mat         mat = (Mat) obj;
  Mat_MPIBAIJ *baij = (Mat_MPIBAIJ *) mat->data;
  int         ierr;

#if defined(PETSC_LOG)
  PLogObjectState(obj,"Rows=%d, Cols=%d",baij->M,baij->N);
#endif

  PetscFree(baij->rowners); 
  ierr = MatDestroy(baij->A); CHKERRQ(ierr);
  ierr = MatDestroy(baij->B); CHKERRQ(ierr);
  if (baij->colmap) PetscFree(baij->colmap);
  if (baij->garray) PetscFree(baij->garray);
  if (baij->lvec)   VecDestroy(baij->lvec);
  if (baij->Mvctx)  VecScatterDestroy(baij->Mvctx);
  if (baij->rowvalues) PetscFree(baij->rowvalues);
  PetscFree(baij); 
  PLogObjectDestroy(mat);
  PetscHeaderDestroy(mat);
  return 0;
}

static int MatMult_MPIBAIJ(Mat A,Vec xx,Vec yy)
{
  Mat_MPIBAIJ *a = (Mat_MPIBAIJ *) A->data;
  int         ierr, nt;

  ierr = VecGetLocalSize(xx,&nt);  CHKERRQ(ierr);
  if (nt != a->n) {
    SETERRQ(1,"MatMult_MPIAIJ:Incompatible parition of A and xx");
  }
  ierr = VecGetLocalSize(yy,&nt);  CHKERRQ(ierr);
  if (nt != a->m) {
    SETERRQ(1,"MatMult_MPIAIJ:Incompatible parition of A and yy");
  }
  ierr = VecScatterBegin(xx,a->lvec,INSERT_VALUES,SCATTER_ALL,a->Mvctx); CHKERRQ(ierr);
  ierr = (*a->A->ops.mult)(a->A,xx,yy); CHKERRQ(ierr);
  ierr = VecScatterEnd(xx,a->lvec,INSERT_VALUES,SCATTER_ALL,a->Mvctx); CHKERRQ(ierr);
  ierr = (*a->B->ops.multadd)(a->B,a->lvec,yy,yy); CHKERRQ(ierr);
  return 0;
}

static int MatMultAdd_MPIBAIJ(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_MPIBAIJ *a = (Mat_MPIBAIJ *) A->data;
  int        ierr;
  ierr = VecScatterBegin(xx,a->lvec,INSERT_VALUES,SCATTER_ALL,a->Mvctx);CHKERRQ(ierr);
  ierr = (*a->A->ops.multadd)(a->A,xx,yy,zz); CHKERRQ(ierr);
  ierr = VecScatterEnd(xx,a->lvec,INSERT_VALUES,SCATTER_ALL,a->Mvctx);CHKERRQ(ierr);
  ierr = (*a->B->ops.multadd)(a->B,a->lvec,zz,zz); CHKERRQ(ierr);
  return 0;
}

static int MatMultTrans_MPIBAIJ(Mat A,Vec xx,Vec yy)
{
  Mat_MPIBAIJ *a = (Mat_MPIBAIJ *) A->data;
  int        ierr;

  /* do nondiagonal part */
  ierr = (*a->B->ops.multtrans)(a->B,xx,a->lvec); CHKERRQ(ierr);
  /* send it on its way */
  ierr = VecScatterBegin(a->lvec,yy,ADD_VALUES,
                (ScatterMode)(SCATTER_ALL|SCATTER_REVERSE),a->Mvctx); CHKERRQ(ierr);
  /* do local part */
  ierr = (*a->A->ops.multtrans)(a->A,xx,yy); CHKERRQ(ierr);
  /* receive remote parts: note this assumes the values are not actually */
  /* inserted in yy until the next line, which is true for my implementation*/
  /* but is not perhaps always true. */
  ierr = VecScatterEnd(a->lvec,yy,ADD_VALUES,
                  (ScatterMode)(SCATTER_ALL|SCATTER_REVERSE),a->Mvctx); CHKERRQ(ierr);
  return 0;
}

static int MatMultTransAdd_MPIBAIJ(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_MPIBAIJ *a = (Mat_MPIBAIJ *) A->data;
  int        ierr;

  /* do nondiagonal part */
  ierr = (*a->B->ops.multtrans)(a->B,xx,a->lvec); CHKERRQ(ierr);
  /* send it on its way */
  ierr = VecScatterBegin(a->lvec,zz,ADD_VALUES,
                 (ScatterMode)(SCATTER_ALL|SCATTER_REVERSE),a->Mvctx); CHKERRQ(ierr);
  /* do local part */
  ierr = (*a->A->ops.multtransadd)(a->A,xx,yy,zz); CHKERRQ(ierr);
  /* receive remote parts: note this assumes the values are not actually */
  /* inserted in yy until the next line, which is true for my implementation*/
  /* but is not perhaps always true. */
  ierr = VecScatterEnd(a->lvec,zz,ADD_VALUES,
                  (ScatterMode)(SCATTER_ALL|SCATTER_REVERSE),a->Mvctx); CHKERRQ(ierr);
  return 0;
}

/*
  This only works correctly for square matrices where the subblock A->A is the 
   diagonal block
*/
static int MatGetDiagonal_MPIBAIJ(Mat A,Vec v)
{
  Mat_MPIBAIJ *a = (Mat_MPIBAIJ *) A->data;
  if (a->M != a->N) 
    SETERRQ(1,"MatGetDiagonal_MPIBAIJ:Supports only square matrix where A->A is diag block");
  return MatGetDiagonal(a->A,v);
}

static int MatScale_MPIBAIJ(Scalar *aa,Mat A)
{
  Mat_MPIBAIJ *a = (Mat_MPIBAIJ *) A->data;
  int        ierr;
  ierr = MatScale(aa,a->A); CHKERRQ(ierr);
  ierr = MatScale(aa,a->B); CHKERRQ(ierr);
  return 0;
}
static int MatGetSize_MPIBAIJ(Mat matin,int *m,int *n)
{
  Mat_MPIBAIJ *mat = (Mat_MPIBAIJ *) matin->data;
  *m = mat->M; *n = mat->N;
  return 0;
}

static int MatGetLocalSize_MPIBAIJ(Mat matin,int *m,int *n)
{
  Mat_MPIBAIJ *mat = (Mat_MPIBAIJ *) matin->data;
  *m = mat->m; *n = mat->N;
  return 0;
}

static int MatGetOwnershipRange_MPIBAIJ(Mat matin,int *m,int *n)
{
  Mat_MPIBAIJ *mat = (Mat_MPIBAIJ *) matin->data;
  *m = mat->rstart*mat->bs; *n = mat->rend*mat->bs;
  return 0;
}

extern int MatGetRow_SeqBAIJ(Mat,int,int*,int**,Scalar**);
extern int MatRestoreRow_SeqBAIJ(Mat,int,int*,int**,Scalar**);

int MatGetRow_MPIBAIJ(Mat matin,int row,int *nz,int **idx,Scalar **v)
{
  Mat_MPIBAIJ *mat = (Mat_MPIBAIJ *) matin->data;
  Scalar     *vworkA, *vworkB, **pvA, **pvB,*v_p;
  int        bs = mat->bs, bs2 = mat->bs2, i, ierr, *cworkA, *cworkB, **pcA, **pcB;
  int        nztot, nzA, nzB, lrow, brstart = mat->rstart*bs, brend = mat->rend*bs;
  int        *cmap, *idx_p,cstart = mat->cstart;

  if (mat->getrowactive == PETSC_TRUE) SETERRQ(1,"MatGetRow_MPIBAIJ:Already active");
  mat->getrowactive = PETSC_TRUE;

  if (!mat->rowvalues && (idx || v)) {
    /*
        allocate enough space to hold information from the longest row.
    */
    Mat_SeqBAIJ *Aa = (Mat_SeqBAIJ *) mat->A->data,*Ba = (Mat_SeqBAIJ *) mat->B->data; 
    int     max = 1,mbs = mat->mbs,tmp;
    for ( i=0; i<mbs; i++ ) {
      tmp = Aa->i[i+1] - Aa->i[i] + Ba->i[i+1] - Ba->i[i];
      if (max < tmp) { max = tmp; }
    }
    mat->rowvalues = (Scalar *) PetscMalloc( max*bs2*(sizeof(int)+sizeof(Scalar))); 
    CHKPTRQ(mat->rowvalues);
    mat->rowindices = (int *) (mat->rowvalues + max*bs2);
  }
       

  if (row < brstart || row >= brend) SETERRQ(1,"MatGetRow_MPIBAIJ:Only local rows")
  lrow = row - brstart;

  pvA = &vworkA; pcA = &cworkA; pvB = &vworkB; pcB = &cworkB;
  if (!v)   {pvA = 0; pvB = 0;}
  if (!idx) {pcA = 0; if (!v) pcB = 0;}
  ierr = (*mat->A->ops.getrow)(mat->A,lrow,&nzA,pcA,pvA); CHKERRQ(ierr);
  ierr = (*mat->B->ops.getrow)(mat->B,lrow,&nzB,pcB,pvB); CHKERRQ(ierr);
  nztot = nzA + nzB;

  cmap  = mat->garray;
  if (v  || idx) {
    if (nztot) {
      /* Sort by increasing column numbers, assuming A and B already sorted */
      int imark = -1;
      if (v) {
        *v = v_p = mat->rowvalues;
        for ( i=0; i<nzB; i++ ) {
          if (cmap[cworkB[i]/bs] < cstart)   v_p[i] = vworkB[i];
          else break;
        }
        imark = i;
        for ( i=0; i<nzA; i++ )     v_p[imark+i] = vworkA[i];
        for ( i=imark; i<nzB; i++ ) v_p[nzA+i]   = vworkB[i];
      }
      if (idx) {
        *idx = idx_p = mat->rowindices;
        if (imark > -1) {
          for ( i=0; i<imark; i++ ) {
            idx_p[i] = cmap[cworkB[i]/bs]*bs + cworkB[i]%bs;
          }
        } else {
          for ( i=0; i<nzB; i++ ) {
            if (cmap[cworkB[i]/bs] < cstart)   
              idx_p[i] = cmap[cworkB[i]/bs]*bs + cworkB[i]%bs ;
            else break;
          }
          imark = i;
        }
        for ( i=0; i<nzA; i++ )     idx_p[imark+i] = cstart*bs + cworkA[i];
        for ( i=imark; i<nzB; i++ ) idx_p[nzA+i]   = cmap[cworkB[i]/bs]*bs + cworkB[i]%bs ;
      } 
    } 
    else {
      if (idx) *idx = 0;
      if (v)   *v   = 0;
    }
  }
  *nz = nztot;
  ierr = (*mat->A->ops.restorerow)(mat->A,lrow,&nzA,pcA,pvA); CHKERRQ(ierr);
  ierr = (*mat->B->ops.restorerow)(mat->B,lrow,&nzB,pcB,pvB); CHKERRQ(ierr);
  return 0;
}

int MatRestoreRow_MPIBAIJ(Mat mat,int row,int *nz,int **idx,Scalar **v)
{
  Mat_MPIBAIJ *baij = (Mat_MPIBAIJ *) mat->data;
  if (baij->getrowactive == PETSC_FALSE) {
    SETERRQ(1,"MatRestoreRow_MPIBAIJ:MatGetRow not called");
  }
  baij->getrowactive = PETSC_FALSE;
  return 0;
}

static int MatGetBlockSize_MPIBAIJ(Mat mat, int *bs)
{
  Mat_MPIBAIJ *baij = (Mat_MPIBAIJ *) mat->data;
  *bs = baij->bs;
  return 0;
}

static int MatZeroEntries_MPIBAIJ(Mat A)
{
  Mat_MPIBAIJ *l = (Mat_MPIBAIJ *) A->data;
  int         ierr;
  ierr = MatZeroEntries(l->A); CHKERRQ(ierr);
  ierr = MatZeroEntries(l->B); CHKERRQ(ierr);
  return 0;
}
static int MatSetOption_MPIBAIJ(Mat A,MatOption op)
{
  Mat_MPIBAIJ *a = (Mat_MPIBAIJ *) A->data;

  if (op == MAT_NO_NEW_NONZERO_LOCATIONS ||
      op == MAT_YES_NEW_NONZERO_LOCATIONS ||
      op == MAT_COLUMNS_SORTED ||
      op == MAT_ROW_ORIENTED) {
        MatSetOption(a->A,op);
        MatSetOption(a->B,op);
  }
  else if (op == MAT_ROWS_SORTED || 
           op == MAT_SYMMETRIC ||
           op == MAT_STRUCTURALLY_SYMMETRIC ||
           op == MAT_YES_NEW_DIAGONALS)
    PLogInfo(A,"Info:MatSetOption_MPIBAIJ:Option ignored\n");
  else if (op == MAT_COLUMN_ORIENTED) {
    a->roworiented = 0;
    MatSetOption(a->A,op);
    MatSetOption(a->B,op);
  }
  else if (op == MAT_NO_NEW_DIAGONALS)
    {SETERRQ(PETSC_ERR_SUP,"MatSetOption_MPIAIJ:MAT_NO_NEW_DIAGONALS");}
  else 
    {SETERRQ(PETSC_ERR_SUP,"MatSetOption_MPIAIJ:unknown option");}
  return 0;
}

/* -------------------------------------------------------------------*/
static struct _MatOps MatOps = {
  MatSetValues_MPIBAIJ,MatGetRow_MPIBAIJ,MatRestoreRow_MPIBAIJ,MatMult_MPIBAIJ,
  MatMultAdd_MPIBAIJ,MatMultTrans_MPIBAIJ,MatMultTransAdd_MPIBAIJ,0,
  0,0,0,0,
  0,0,0,0,
  0,MatGetDiagonal_MPIBAIJ,0,MatNorm_MPIBAIJ,
  MatAssemblyBegin_MPIBAIJ,MatAssemblyEnd_MPIBAIJ,0,MatSetOption_MPIBAIJ,
  MatZeroEntries_MPIBAIJ,0,MatGetReordering_MPIBAIJ,0,
  0,0,0,MatGetSize_MPIBAIJ,
  MatGetLocalSize_MPIBAIJ,MatGetOwnershipRange_MPIBAIJ,0,0,
  0,0,0,0,
  0,0,0,0,
  0,0,0,MatGetSubMatrices_MPIBAIJ,
  MatIncreaseOverlap_MPIBAIJ,MatGetValues_MPIBAIJ,0,0,
  MatScale_MPIBAIJ,0,0,0,MatGetBlockSize_MPIBAIJ};
                                

/*@C
   MatCreateMPIBAIJ - Creates a sparse parallel matrix in block AIJ format
   (block compressed row).  For good matrix assembly performance
   the user should preallocate the matrix storage by setting the parameters 
   d_nz (or d_nnz) and o_nz (or o_nnz).  By setting these parameters accurately,
   performance can be increased by more than a factor of 50.

   Input Parameters:
.  comm - MPI communicator
.  bs   - size of blockk
.  m - number of local rows (or PETSC_DECIDE to have calculated if M is given)
           This value should be the same as the local size used in creating the 
           y vector for the matrix-vector product y = Ax.
.  n - number of local columns (or PETSC_DECIDE to have calculated if N is given)
           This value should be the same as the local size used in creating the 
           x vector for the matrix-vector product y = Ax.
.  M - number of global rows (or PETSC_DECIDE to have calculated if m is given)
.  N - number of global columns (or PETSC_DECIDE to have calculated if n is given)
.  d_nz  - number of block nonzeros per block row in diagonal portion of local 
           submatrix  (same for all local rows)
.  d_nzz - array containing the number of block nonzeros in the various block rows 
           of the in diagonal portion of the local (possibly different for each block
           row) or PETSC_NULL.  You must leave room for the diagonal entry even if
           it is zero.
.  o_nz  - number of block nonzeros per block row in the off-diagonal portion of local
           submatrix (same for all local rows).
.  o_nzz - array containing the number of nonzeros in the various block rows of the
           off-diagonal portion of the local submatrix (possibly different for
           each block row) or PETSC_NULL.

   Output Parameter:
.  A - the matrix 

   Notes:
   The user MUST specify either the local or global matrix dimensions
   (possibly both).

   Storage Information:
   For a square global matrix we define each processor's diagonal portion 
   to be its local rows and the corresponding columns (a square submatrix);  
   each processor's off-diagonal portion encompasses the remainder of the
   local matrix (a rectangular submatrix). 

   The user can specify preallocated storage for the diagonal part of
   the local submatrix with either d_nz or d_nnz (not both).  Set 
   d_nz=PETSC_DEFAULT and d_nnz=PETSC_NULL for PETSc to control dynamic
   memory allocation.  Likewise, specify preallocated storage for the
   off-diagonal part of the local submatrix with o_nz or o_nnz (not both).

   Consider a processor that owns rows 3, 4 and 5 of a parallel matrix. In
   the figure below we depict these three local rows and all columns (0-11).

$          0 1 2 3 4 5 6 7 8 9 10 11
$         -------------------
$  row 3  |  o o o d d d o o o o o o
$  row 4  |  o o o d d d o o o o o o
$  row 5  |  o o o d d d o o o o o o
$         -------------------
$ 

   Thus, any entries in the d locations are stored in the d (diagonal) 
   submatrix, and any entries in the o locations are stored in the
   o (off-diagonal) submatrix.  Note that the d and the o submatrices are
   stored simply in the MATSEQBAIJ format for compressed row storage.

   Now d_nz should indicate the number of nonzeros per row in the d matrix,
   and o_nz should indicate the number of nonzeros per row in the o matrix.
   In general, for PDE problems in which most nonzeros are near the diagonal,
   one expects d_nz >> o_nz.   For large problems you MUST preallocate memory
   or you will get TERRIBLE performance; see the users' manual chapter on
   matrices and the file $(PETSC_DIR)/Performance.

.keywords: matrix, block, aij, compressed row, sparse, parallel

.seealso: MatCreate(), MatCreateSeqBAIJ(), MatSetValues()
@*/
int MatCreateMPIBAIJ(MPI_Comm comm,int bs,int m,int n,int M,int N,
                    int d_nz,int *d_nnz,int o_nz,int *o_nnz,Mat *A)
{
  Mat          B;
  Mat_MPIBAIJ  *b;
  int          ierr, i,sum[2],work[2],mbs,nbs,Mbs=PETSC_DECIDE,Nbs=PETSC_DECIDE;

  if (bs < 1) SETERRQ(1,"MatCreateMPIBAIJ: invalid block size specified");
  *A = 0;
  PetscHeaderCreate(B,_Mat,MAT_COOKIE,MATMPIBAIJ,comm);
  PLogObjectCreate(B);
  B->data       = (void *) (b = PetscNew(Mat_MPIBAIJ)); CHKPTRQ(b);
  PetscMemzero(b,sizeof(Mat_MPIBAIJ));
  PetscMemcpy(&B->ops,&MatOps,sizeof(struct _MatOps));
  B->destroy    = MatDestroy_MPIBAIJ;
  B->view       = MatView_MPIBAIJ;

  B->factor     = 0;
  B->assembled  = PETSC_FALSE;

  b->insertmode = NOT_SET_VALUES;
  MPI_Comm_rank(comm,&b->rank);
  MPI_Comm_size(comm,&b->size);

  if ( m == PETSC_DECIDE && (d_nnz != PETSC_NULL || o_nnz != PETSC_NULL)) 
    SETERRQ(1,"MatCreateMPIBAIJ:Cannot have PETSC_DECIDE rows but set d_nnz or o_nnz");
  if ( M == PETSC_DECIDE && m == PETSC_DECIDE) SETERRQ(1,"MatCreateMPIBAIJ: either M or m should be specified");
  if ( M == PETSC_DECIDE && n == PETSC_DECIDE)SETERRQ(1,"MatCreateMPIBAIJ: either N or n should be specified"); 
  if ( M != PETSC_DECIDE && m != PETSC_DECIDE) M = PETSC_DECIDE;
  if ( N != PETSC_DECIDE && n != PETSC_DECIDE) N = PETSC_DECIDE;

  if (M == PETSC_DECIDE || N == PETSC_DECIDE) {
    work[0] = m; work[1] = n;
    mbs = m/bs; nbs = n/bs;
    MPI_Allreduce( work, sum,2,MPI_INT,MPI_SUM,comm );
    if (M == PETSC_DECIDE) {M = sum[0]; Mbs = M/bs;}
    if (N == PETSC_DECIDE) {N = sum[1]; Nbs = N/bs;}
  }
  if (m == PETSC_DECIDE) {
    Mbs = M/bs;
    if (Mbs*bs != M) SETERRQ(1,"MatCreateMPIBAIJ: No of global rows must be divisible by blocksize");
    mbs = Mbs/b->size + ((Mbs % b->size) > b->rank);
    m   = mbs*bs;
  }
  if (n == PETSC_DECIDE) {
    Nbs = N/bs;
    if (Nbs*bs != N) SETERRQ(1,"MatCreateMPIBAIJ: No of global cols must be divisible by blocksize");
    nbs = Nbs/b->size + ((Nbs % b->size) > b->rank);
    n   = nbs*bs;
  }
  if (mbs*bs != m || nbs*bs != n) SETERRQ(1,"MatCreateMPIBAIJ: No of local rows, cols must be divisible by blocksize");

  b->m = m; B->m = m;
  b->n = n; B->n = n;
  b->N = N; B->N = N;
  b->M = M; B->M = M;
  b->bs  = bs;
  b->bs2 = bs*bs;
  b->mbs = mbs;
  b->nbs = nbs;
  b->Mbs = Mbs;
  b->Nbs = Nbs;

  /* build local table of row and column ownerships */
  b->rowners = (int *) PetscMalloc(2*(b->size+2)*sizeof(int)); CHKPTRQ(b->rowners);
  PLogObjectMemory(B,2*(b->size+2)*sizeof(int)+sizeof(struct _Mat)+sizeof(Mat_MPIBAIJ));
  b->cowners = b->rowners + b->size + 1;
  MPI_Allgather(&mbs,1,MPI_INT,b->rowners+1,1,MPI_INT,comm);
  b->rowners[0] = 0;
  for ( i=2; i<=b->size; i++ ) {
    b->rowners[i] += b->rowners[i-1];
  }
  b->rstart = b->rowners[b->rank]; 
  b->rend   = b->rowners[b->rank+1]; 
  MPI_Allgather(&nbs,1,MPI_INT,b->cowners+1,1,MPI_INT,comm);
  b->cowners[0] = 0;
  for ( i=2; i<=b->size; i++ ) {
    b->cowners[i] += b->cowners[i-1];
  }
  b->cstart = b->cowners[b->rank]; 
  b->cend   = b->cowners[b->rank+1]; 
  
  if (d_nz == PETSC_DEFAULT) d_nz = 5;
  ierr = MatCreateSeqBAIJ(MPI_COMM_SELF,bs,m,n,d_nz,d_nnz,&b->A); CHKERRQ(ierr);
  PLogObjectParent(B,b->A);
  if (o_nz == PETSC_DEFAULT) o_nz = 0;
  ierr = MatCreateSeqBAIJ(MPI_COMM_SELF,bs,m,N,o_nz,o_nnz,&b->B); CHKERRQ(ierr);
  PLogObjectParent(B,b->B);

  /* build cache for off array entries formed */
  ierr = StashBuild_Private(&b->stash); CHKERRQ(ierr);
  b->colmap      = 0;
  b->garray      = 0;
  b->roworiented = 1;

  /* stuff used for matrix vector multiply */
  b->lvec      = 0;
  b->Mvctx     = 0;

  /* stuff for MatGetRow() */
  b->rowindices   = 0;
  b->rowvalues    = 0;
  b->getrowactive = PETSC_FALSE;

  *A = B;
  return 0;
}

#include "sys.h"

int MatLoad_MPIBAIJ(Viewer viewer,MatType type,Mat *newmat)
{
  Mat          A;
  int          i, nz, ierr, j,rstart, rend, fd;
  Scalar       *vals,*buf;
  MPI_Comm     comm = ((PetscObject)viewer)->comm;
  MPI_Status   status;
  int          header[4],rank,size,*rowlengths = 0,M,N,m,*rowners,*browners,maxnz,*cols;
  int          *locrowlens,*sndcounts = 0,*procsnz = 0, jj,*mycols,*ibuf;
  int          flg,tag = ((PetscObject)viewer)->tag,bs=1,bs2,Mbs,mbs,extra_rows;
  int          *dlens,*odlens,*mask,*masked1,*masked2,rowcount,odcount;
  int          dcount,kmax,k,nzcount,tmp;

 
  ierr = OptionsGetInt(PETSC_NULL,"-matload_block_size",&bs,&flg);CHKERRQ(ierr);
  bs2  = bs*bs;

  MPI_Comm_size(comm,&size); MPI_Comm_rank(comm,&rank);
  if (!rank) {
    ierr = ViewerBinaryGetDescriptor(viewer,&fd); CHKERRQ(ierr);
    ierr = PetscBinaryRead(fd,(char *)header,4,BINARY_INT); CHKERRQ(ierr);
    if (header[0] != MAT_COOKIE) SETERRQ(1,"MatLoad_MPIBAIJ:not matrix object");
  }
    
  MPI_Bcast(header+1,3,MPI_INT,0,comm);
  M = header[1]; N = header[2];


  if (M != N) SETERRQ(1,"MatLoad_SeqBAIJ:Can only do square matrices");

  /* 
     This code adds extra rows to make sure the number of rows is 
     divisible by the blocksize
  */
  Mbs        = M/bs;
  extra_rows = bs - M + bs*(Mbs);
  if (extra_rows == bs) extra_rows = 0;
  else                  Mbs++;
  if (extra_rows &&!rank) {
    PLogInfo(0,"MatLoad_SeqBAIJ:Padding loaded matrix to match blocksize");
  }
  /* determine ownership of all rows */
  mbs = Mbs/size + ((Mbs % size) > rank);
  m   = mbs * bs;
  rowners = (int *) PetscMalloc(2*(size+2)*sizeof(int)); CHKPTRQ(rowners);
  browners = rowners + size + 1;
  MPI_Allgather(&mbs,1,MPI_INT,rowners+1,1,MPI_INT,comm);
  rowners[0] = 0;
  for ( i=2; i<=size; i++ ) rowners[i] += rowners[i-1];
  for ( i=0; i<=size;  i++ ) browners[i] = rowners[i]*bs;
  rstart = rowners[rank]; 
  rend   = rowners[rank+1]; 

  /* distribute row lengths to all processors */
  locrowlens = (int*) PetscMalloc( (rend-rstart)*bs*sizeof(int) ); CHKPTRQ(locrowlens);
  if (!rank) {
    rowlengths = (int*) PetscMalloc( (M+extra_rows)*sizeof(int) ); CHKPTRQ(rowlengths);
    ierr = PetscBinaryRead(fd,rowlengths,M,BINARY_INT); CHKERRQ(ierr);
    for ( i=0; i<extra_rows; i++ ) rowlengths[M+i] = 1;
    sndcounts = (int*) PetscMalloc( size*sizeof(int) ); CHKPTRQ(sndcounts);
    for ( i=0; i<size; i++ ) sndcounts[i] = browners[i+1] - browners[i];
    MPI_Scatterv(rowlengths,sndcounts,browners,MPI_INT,locrowlens,(rend-rstart)*bs,MPI_INT,0,comm);
    PetscFree(sndcounts);
  }
  else {
    MPI_Scatterv(0,0,0,MPI_INT,locrowlens,(rend-rstart)*bs,MPI_INT, 0,comm);
  }

  if (!rank) {
    /* calculate the number of nonzeros on each processor */
    procsnz = (int*) PetscMalloc( size*sizeof(int) ); CHKPTRQ(procsnz);
    PetscMemzero(procsnz,size*sizeof(int));
    for ( i=0; i<size; i++ ) {
      for ( j=rowners[i]*bs; j< rowners[i+1]*bs; j++ ) {
        procsnz[i] += rowlengths[j];
      }
    }
    PetscFree(rowlengths);
    
    /* determine max buffer needed and allocate it */
    maxnz = 0;
    for ( i=0; i<size; i++ ) {
      maxnz = PetscMax(maxnz,procsnz[i]);
    }
    cols = (int *) PetscMalloc( maxnz*sizeof(int) ); CHKPTRQ(cols);

    /* read in my part of the matrix column indices  */
    nz = procsnz[0];
    ibuf = (int *) PetscMalloc( nz*sizeof(int) ); CHKPTRQ(ibuf);
    mycols = ibuf;
    if (size == 1)  nz -= extra_rows;
    ierr = PetscBinaryRead(fd,mycols,nz,BINARY_INT); CHKERRQ(ierr);
    if (size == 1)  for (i=0; i< extra_rows; i++) { mycols[nz+i] = M+i; }

    /* read in every ones (except the last) and ship off */
    for ( i=1; i<size-1; i++ ) {
      nz = procsnz[i];
      ierr = PetscBinaryRead(fd,cols,nz,BINARY_INT); CHKERRQ(ierr);
      MPI_Send(cols,nz,MPI_INT,i,tag,comm);
    }
    /* read in the stuff for the last proc */
    if ( size != 1 ) {
      nz = procsnz[size-1] - extra_rows;  /* the extra rows are not on the disk */
      ierr = PetscBinaryRead(fd,cols,nz,BINARY_INT); CHKERRQ(ierr);
      for ( i=0; i<extra_rows; i++ ) cols[nz+i] = M+i;
      MPI_Send(cols,nz+extra_rows,MPI_INT,size-1,tag,comm);
    }
    PetscFree(cols);
  }
  else {
    /* determine buffer space needed for message */
    nz = 0;
    for ( i=0; i<m; i++ ) {
      nz += locrowlens[i];
    }
    ibuf = (int*) PetscMalloc( nz*sizeof(int) ); CHKPTRQ(ibuf);
    mycols = ibuf;
    /* receive message of column indices*/
    MPI_Recv(mycols,nz,MPI_INT,0,tag,comm,&status);
    MPI_Get_count(&status,MPI_INT,&maxnz);
    if (maxnz != nz) SETERRQ(1,"MatLoad_MPIBAIJ:something is wrong with file");
  }
  
  /* loop over local rows, determining number of off diagonal entries */
  dlens  = (int *) PetscMalloc( 2*(rend-rstart+1)*sizeof(int) ); CHKPTRQ(dlens);
  odlens = dlens + (rend-rstart);
  mask   = (int *) PetscMalloc( 3*Mbs*sizeof(int) ); CHKPTRQ(mask);
  PetscMemzero(mask,3*Mbs*sizeof(int));
  masked1 = mask    + Mbs;
  masked2 = masked1 + Mbs;
  rowcount = 0; nzcount = 0;
  for ( i=0; i<mbs; i++ ) {
    dcount  = 0;
    odcount = 0;
    for ( j=0; j<bs; j++ ) {
      kmax = locrowlens[rowcount];
      for ( k=0; k<kmax; k++ ) {
        tmp = mycols[nzcount++]/bs;
        if (!mask[tmp]) {
          mask[tmp] = 1;
          if (tmp < rstart || tmp >= rend ) masked2[odcount++] = tmp;
          else masked1[dcount++] = tmp;
        }
      }
      rowcount++;
    }
  
    dlens[i]  = dcount;
    odlens[i] = odcount;

    /* zero out the mask elements we set */
    for ( j=0; j<dcount; j++ ) mask[masked1[j]] = 0;
    for ( j=0; j<odcount; j++ ) mask[masked2[j]] = 0; 
  }

  /* create our matrix */
  ierr = MatCreateMPIBAIJ(comm,bs,m,PETSC_DECIDE,M+extra_rows,N+extra_rows,0,dlens,0,odlens,newmat);CHKERRQ(ierr);
  A = *newmat;
  MatSetOption(A,MAT_COLUMNS_SORTED); 
  
  if (!rank) {
    buf = (Scalar *) PetscMalloc( maxnz*sizeof(Scalar) ); CHKPTRQ(buf);
    /* read in my part of the matrix numerical values  */
    nz = procsnz[0];
    vals = buf;
    mycols = ibuf;
    if (size == 1)  nz -= extra_rows;
    ierr = PetscBinaryRead(fd,vals,nz,BINARY_SCALAR); CHKERRQ(ierr);
    if (size == 1)  for (i=0; i< extra_rows; i++) { vals[nz+i] = 1.0; }
    /* insert into matrix */
    jj      = rstart*bs;
    for ( i=0; i<m; i++ ) {
      ierr = MatSetValues(A,1,&jj,locrowlens[i],mycols,vals,INSERT_VALUES);CHKERRQ(ierr);
      mycols += locrowlens[i];
      vals   += locrowlens[i];
      jj++;
    }
    /* read in other processors( except the last one) and ship out */
    for ( i=1; i<size-1; i++ ) {
      nz = procsnz[i];
      vals = buf;
      ierr = PetscBinaryRead(fd,vals,nz,BINARY_SCALAR); CHKERRQ(ierr);
      MPI_Send(vals,nz,MPIU_SCALAR,i,A->tag,comm);
    }
    /* the last proc */
    if ( size != 1 ){
      nz = procsnz[i] - extra_rows;
      vals = buf;
      ierr = PetscBinaryRead(fd,vals,nz,BINARY_SCALAR); CHKERRQ(ierr);
      for ( i=0; i<extra_rows; i++ ) vals[nz+i] = 1.0;
      MPI_Send(vals,nz+extra_rows,MPIU_SCALAR,size-1,A->tag,comm);
    }
    PetscFree(procsnz);
  }
  else {
    /* receive numeric values */
    buf = (Scalar*) PetscMalloc( nz*sizeof(Scalar) ); CHKPTRQ(buf);

    /* receive message of values*/
    vals = buf;
    mycols = ibuf;
    MPI_Recv(vals,nz,MPIU_SCALAR,0,A->tag,comm,&status);
    MPI_Get_count(&status,MPIU_SCALAR,&maxnz);
    if (maxnz != nz) SETERRQ(1,"MatLoad_MPIBAIJ:something is wrong with file");

    /* insert into matrix */
    jj      = rstart*bs;
    for ( i=0; i<m; i++ ) {
      ierr = MatSetValues(A,1,&jj,locrowlens[i],mycols,vals,INSERT_VALUES);CHKERRQ(ierr);
      mycols += locrowlens[i];
      vals   += locrowlens[i];
      jj++;
    }
  }
  PetscFree(locrowlens); 
  PetscFree(buf); 
  PetscFree(ibuf); 
  PetscFree(rowners);
  PetscFree(dlens);
  PetscFree(mask);
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  return 0;
}


