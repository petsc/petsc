
/*$Id: mesi.c,v 1.1 2001/09/12 03:30:08 bsmith Exp bsmith $*/
/*
    Defines the basic matrix operations for the AIJ (compressed row)
  matrix storage format.
*/

#include "petscsys.h"       
#include "esi/ESI.h"
#include "esi/petsc/vector.h"
#include "esi/petsc/matrix.h"
#include "src/mat/matimpl.h"   /*I "petscmat.h" I*/

typedef struct { 
  int                                   rstart,rend; /* range of local rows */
  esi::Operator<double,int>             *eop;
  esi::MatrixData<int>                  *emat;
  esi::MatrixRowReadAccess<double,int>  *rmat;
  esi::MatrixRowWriteAccess<double,int> *wmat;
} Mat_ESI;

#undef __FUNCT__  
#define __FUNCT__ "MatESISetOperator"
/*@C
     MatESISetOperator - Takes a PETSc matrix sets it to type ESI and 
       provides the ESI operator that it wraps to look like a PETSc matrix.

@*/
 int MatESISetOperator(Mat xin,esi::Operator<double,int> *v)
{
  Mat_ESI    *x = (Mat_ESI*)xin->data;
  PetscTruth tesi;
  int        ierr;

  PetscFunctionBegin;

  ierr = v->getInterface("esi::MatrixData",static_cast<void*>(x->emat));
  ierr = v->getInterface("esi::MatrixRowReadAccess",static_cast<void*>(x->rmat));CHKERRQ(ierr);
  ierr = v->getInterface("esi::MatrixRowWriteAccess",static_cast<void*>(x->wmat));CHKERRQ(ierr);
  if (!x->emat) SETERRQ(1,"PETSc currently requires esi::Operator to support esi::MatrixData interface");

  ierr = PetscTypeCompare((PetscObject)xin,0,&tesi);CHKERRQ(ierr);
  if (tesi) {
    ierr = MatSetType(xin,MATESI);CHKERRQ(ierr);
  }
  ierr = PetscTypeCompare((PetscObject)xin,MATESI,&tesi);CHKERRQ(ierr);
  if (tesi) {
    int                    m,n,M,N;
    esi::IndexSpace<int>   *rmap,*cmap;

    ierr = x->emat->getIndexSpaces(rmap,cmap);CHKERRQ(ierr);

    ierr = rmap->getGlobalSize(M);CHKERRQ(ierr);
    if (xin->M == -1) xin->M = M;
    else if (xin->M != M) SETERRQ2(1,"Global rows of Mat %d not equal size of esi::MatrixData %d",xin->M,M);

    ierr = cmap->getGlobalSize(N);CHKERRQ(ierr);
    if (xin->N == -1) xin->N = N;
    else if (xin->N != N) SETERRQ2(1,"Global columns of Mat %d not equal size of esi::MatrixData %d",xin->N,N);

    ierr = rmap->getLocalSize(m);CHKERRQ(ierr);
    if (xin->m == -1) xin->m = m;
    else if (xin->m != m) SETERRQ2(1,"Local rows of Mat %d not equal size of esi::MatrixData %d",xin->n,n);

    ierr = cmap->getLocalSize(n);CHKERRQ(ierr);
    if (xin->n == -1) xin->n = n;
    else if (xin->n != n) SETERRQ2(1,"Local columns of Mat %d not equal size of esi::MatrixData %d",xin->n,n);

    x->eop  = v;
    v->addReference();
    if (!xin->rmap){
      ierr = PetscMapCreateMPI(xin->comm,m,M,&xin->rmap);CHKERRQ(ierr);
    }
    if (!xin->cmap){
      ierr = PetscMapCreateMPI(xin->comm,n,N,&xin->cmap);CHKERRQ(ierr);
    }
    ierr = PetscMapGetLocalRange(xin->rmap,&x->rstart,&x->rend);CHKERRQ(ierr);
    ierr = MatStashCreate_Private(xin->comm,1,&xin->stash);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

extern PetscFList CCAList;

#undef __FUNCT__  
#define __FUNCT__ "MatESISetType"
/*@
    MatESISetType - Given a PETSc matrix of type ESI loads the ESI constructor
          by name and wraps the ESI operator to look like a PETSc matrix.
@*/
int MatESISetType(Mat V,char *name)
{
  int                                ierr;
  ::esi::Operator<double,int>        *ve;
  ::esi::OperatorFactory<double,int> *f;
#if defined(PETSC_HAVE_CCA)
  ::gov::cca::Component              *(*r)(void);   
#else
  ::esi::OperatorFactory<double,int> *(*r)(void);
#endif
  ::esi::IndexSpace<int>             *rmap,*cmap;

  PetscFunctionBegin;
  ierr = PetscFListFind(V->comm,CCAList,name,(void(**)(void))&r);CHKERRQ(ierr);
  if (!r) SETERRQ1(1,"Unable to load esi::OperatorFactory constructor %s",name);
#if defined(PETSC_HAVE_CCA)
  gov::cca::Component *component = (*r)();
  gov::cca::Port      *port      = dynamic_cast<gov::cca::Port*>(component);
  f    = dynamic_cast<esi::OperatorFactory<double,int>*>(port);
#else
  f    = (*r)();
#endif
  if (V->m == PETSC_DECIDE) {
    ierr = PetscSplitOwnership(V->comm,&V->m,&V->M);CHKERRQ(ierr);
  }
  ierr = ESICreateIndexSpace("MPI",&V->comm,V->m,rmap);CHKERRQ(ierr);
  if (V->n == PETSC_DECIDE) {
    ierr = PetscSplitOwnership(V->comm,&V->n,&V->N);CHKERRQ(ierr);
  }
  ierr = ESICreateIndexSpace("MPI",&V->comm,V->n,cmap);CHKERRQ(ierr);
  ierr = f->getOperator(*rmap,*cmap,ve);CHKERRQ(ierr);
  ierr = rmap->deleteReference();CHKERRQ(ierr);
  ierr = cmap->deleteReference();CHKERRQ(ierr);
  delete f;
  ierr = MatESISetOperator(V,ve);CHKERRQ(ierr);
  ierr = ve->deleteReference();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatESISetFromOptions"
int MatESISetFromOptions(Mat V)
{
  char       string[1024];
  PetscTruth flg;
  int        ierr;
 
  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)V,MATESI,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscOptionsGetString(V->prefix,"-mat_esi_type",string,1024,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = MatESISetType(V,string);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------------------------*/

#undef __FUNCT__  
#define __FUNCT__ "MatSetValues_ESI"
int MatSetValues_ESI(Mat mat,int m,int *im,int n,int *in,PetscScalar *v,InsertMode addv)
{
  Mat_ESI *esi = (Mat_ESI*)mat->data;
  int      ierr,row,i,j,rstart = esi->rstart,rend = esi->rend;
 
  PetscFunctionBegin;
  for (i=0; i<m; i++) {
    if (im[i] < 0) continue;
#if defined(PETSC_USE_BOPT_g)
    if (im[i] >= mat->M) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Row too large");
#endif
    if (im[i] >= rstart && im[i] < rend) {
      row = im[i] - rstart;
      for (j=0; j<n; j++) {
          ierr = esi->wmat->copyIntoRow(im[i],&v[i+j*m],&in[j],1);CHKERRQ(ierr);
       }
    } else {
      ierr = MatStashValuesCol_Private(&mat->stash,im[i],n,in,v+i,m);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatAssemblyBegin_ESI"
int MatAssemblyBegin_ESI(Mat mat,MatAssemblyType mode)
{ 
  int         ierr,nstash,reallocs,*rowners;
  InsertMode  addv;

  PetscFunctionBegin;

  /* make sure all processors are either in INSERTMODE or ADDMODE */
  ierr = MPI_Allreduce(&mat->insertmode,&addv,1,MPI_INT,MPI_BOR,mat->comm);CHKERRQ(ierr);
  if (addv == (ADD_VALUES|INSERT_VALUES)) {
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Some processors inserted others added");
  }
  mat->insertmode = addv; /* in case this processor had no cache */

  ierr = PetscMapGetGlobalRange(mat->rmap,&rowners);CHKERRQ(ierr);
  ierr = MatStashScatterBegin_Private(&mat->stash,rowners);CHKERRQ(ierr);
  ierr = MatStashGetInfo_Private(&mat->stash,&nstash,&reallocs);CHKERRQ(ierr);
  PetscLogInfo(0,"MatAssemblyBegin_ESI:Stash has %d entries, uses %d mallocs.\n",nstash,reallocs);
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "MatAssemblyEnd_ESI"
int MatAssemblyEnd_ESI(Mat mat,MatAssemblyType mode)
{ 
  Mat_ESI     *a = (Mat_ESI*)mat->data;
  int         i,j,rstart,ncols,n,ierr,flg;
  int         *row,*col;
  PetscScalar *val;
  InsertMode  addv = mat->insertmode;

  PetscFunctionBegin;
  while (1) {
    ierr = MatStashScatterGetMesg_Private(&mat->stash,&n,&row,&col,&val,&flg);CHKERRQ(ierr);
    if (!flg) break;
     for (i=0; i<n;) {
      /* Now identify the consecutive vals belonging to the same row */
      for (j=i,rstart=row[j]; j<n; j++) { if (row[j] != rstart) break; }
      if (j < n) ncols = j-i;
      else       ncols = n-i;
      /* Now assemble all these values with a single function call */
      ierr = MatSetValues_ESI(mat,1,row+i,ncols,col+i,val+i,addv);CHKERRQ(ierr);
      i = j;
    }
  }
  ierr = MatStashScatterEnd_Private(&mat->stash);CHKERRQ(ierr);
  ierr = a->wmat->loadComplete();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMult_ESI"
int MatMult_ESI(Mat A,Vec xx,Vec yy)
{
  Mat_ESI                 *a = (Mat_ESI*)A->data;
  int                     ierr;
  esi::Vector<double,int> *x,*y;

  PetscFunctionBegin;
  ierr = VecESIWrap(xx,&x);CHKERRQ(ierr);
  ierr = VecESIWrap(yy,&y);CHKERRQ(ierr);
  ierr = a->eop->apply(*x,*y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatDestroy_ESI"
int MatDestroy_ESI(Mat v)
{
  Mat_ESI *vs = (Mat_ESI*)v->data;
  int     ierr;

  PetscFunctionBegin;
  if (vs->eop) {
    vs->eop->deleteReference();
  }
  ierr = MatStashDestroy_Private(&v->bstash);CHKERRQ(ierr);
  ierr = MatStashDestroy_Private(&v->stash);CHKERRQ(ierr);
  ierr = PetscFree(vs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatView_ESI"
int MatView_ESI(Mat A,PetscViewer viewer)
{
  Mat_ESI              *a = (Mat_ESI*)A->data;
  int                  ierr,i,rstart,m,*cols,nz,j;
  PetscTruth           issocket,isascii,isbinary,isdraw;
  esi::IndexSpace<int> *rmap,*cmap;

  PetscScalar *values;

  PetscFunctionBegin;  
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_SOCKET,&issocket);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&isascii);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_BINARY,&isbinary);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_DRAW,&isdraw);CHKERRQ(ierr);
  if (isascii) {
    ierr   = PetscViewerASCIIUseTabs(viewer,PETSC_NO);CHKERRQ(ierr);
    cols   = new int[100];
    values = new PetscScalar[100];
    ierr   = a->emat->getIndexSpaces(rmap,cmap);CHKERRQ(ierr);
    ierr   = rmap->getLocalPartitionOffset(rstart);CHKERRQ(ierr);
    ierr   = rmap->getLocalSize(m);CHKERRQ(ierr);
    for (i=rstart; i<rstart+m; i++) {
      ierr = PetscViewerASCIIPrintf(viewer,"row %d:",i);CHKERRQ(ierr);
      ierr = a->rmat->copyOutRow(i,values,cols,100,nz);CHKERRQ(ierr);
      for (j=0; j<nz; j++) {
        ierr = PetscViewerASCIIPrintf(viewer," %d %g ",cols[j],values[j]);CHKERRQ(ierr);
      }
      ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIUseTabs(viewer,PETSC_YES);CHKERRQ(ierr);
    ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
  } else {
    SETERRQ1(1,"Viewer type %s not supported by SeqAIJ matrices",((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}


/* -------------------------------------------------------------------*/
static struct _MatOps MatOps_Values = {
       MatSetValues_ESI,
       0,
       0,
       MatMult_ESI,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       MatAssemblyBegin_ESI,
       MatAssemblyEnd_ESI,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       MatDestroy_ESI,
       MatView_ESI,
       0};

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatCreate_ESI"
int MatCreate_ESI(Mat B)
{
  int        ierr;
  Mat_ESI    *b;

  PetscFunctionBegin;

  ierr                = PetscNew(Mat_ESI,&b);CHKERRQ(ierr);
  B->data             = (void*)b;
  ierr                = PetscMemzero(b,sizeof(Mat_ESI));CHKERRQ(ierr);
  ierr                = PetscMemcpy(B->ops,&MatOps_Values,sizeof(struct _MatOps));CHKERRQ(ierr);
  B->factor           = 0;
  B->lupivotthreshold = 1.0;
  B->mapping          = 0;
  ierr = PetscOptionsGetReal(PETSC_NULL,"-mat_lu_pivotthreshold",&B->lupivotthreshold,PETSC_NULL);CHKERRQ(ierr);

  b->emat = 0;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatLoad_ESI"
int MatLoad_ESI(PetscViewer viewer,MatType type,Mat *newmat)
{
  Mat          A;
  PetscScalar  *vals,*svals;
  MPI_Comm     comm = ((PetscObject)viewer)->comm;
  MPI_Status   status;
  int          i,nz,ierr,j,rstart,rend,fd;
  int          header[4],rank,size,*rowlengths = 0,M,N,m,*rowners,maxnz,*cols;
  int          *ourlens,*sndcounts = 0,*procsnz = 0,*offlens,jj,*mycols,*smycols;
  int          tag = ((PetscObject)viewer)->tag,cend,cstart,n;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  if (!rank) {
    ierr = PetscViewerBinaryGetDescriptor(viewer,&fd);CHKERRQ(ierr);
    ierr = PetscBinaryRead(fd,(char *)header,4,PETSC_INT);CHKERRQ(ierr);
    if (header[0] != MAT_FILE_COOKIE) SETERRQ(PETSC_ERR_FILE_UNEXPECTED,"not matrix object");
    if (header[3] < 0) {
      SETERRQ(PETSC_ERR_FILE_UNEXPECTED,"Matrix in special format on disk, cannot load as MPIAIJ");
    }
  }

  ierr = MPI_Bcast(header+1,3,MPI_INT,0,comm);CHKERRQ(ierr);
  M = header[1]; N = header[2];
  /* determine ownership of all rows */
  m = M/size + ((M % size) > rank);
  ierr = PetscMalloc((size+2)*sizeof(int),&rowners);CHKERRQ(ierr);
  ierr = MPI_Allgather(&m,1,MPI_INT,rowners+1,1,MPI_INT,comm);CHKERRQ(ierr);
  rowners[0] = 0;
  for (i=2; i<=size; i++) {
    rowners[i] += rowners[i-1];
  }
  rstart = rowners[rank]; 
  rend   = rowners[rank+1]; 

  /* distribute row lengths to all processors */
  ierr    = PetscMalloc(2*(rend-rstart+1)*sizeof(int),&ourlens);CHKERRQ(ierr);
  offlens = ourlens + (rend-rstart);
  if (!rank) {
    ierr = PetscMalloc(M*sizeof(int),&rowlengths);CHKERRQ(ierr);
    ierr = PetscBinaryRead(fd,rowlengths,M,PETSC_INT);CHKERRQ(ierr);
    ierr = PetscMalloc(size*sizeof(int),&sndcounts);CHKERRQ(ierr);
    for (i=0; i<size; i++) sndcounts[i] = rowners[i+1] - rowners[i];
    ierr = MPI_Scatterv(rowlengths,sndcounts,rowners,MPI_INT,ourlens,rend-rstart,MPI_INT,0,comm);CHKERRQ(ierr);
    ierr = PetscFree(sndcounts);CHKERRQ(ierr);
  } else {
    ierr = MPI_Scatterv(0,0,0,MPI_INT,ourlens,rend-rstart,MPI_INT,0,comm);CHKERRQ(ierr);
  }

  if (!rank) {
    /* calculate the number of nonzeros on each processor */
    ierr = PetscMalloc(size*sizeof(int),&procsnz);CHKERRQ(ierr);
    ierr = PetscMemzero(procsnz,size*sizeof(int));CHKERRQ(ierr);
    for (i=0; i<size; i++) {
      for (j=rowners[i]; j< rowners[i+1]; j++) {
        procsnz[i] += rowlengths[j];
      }
    }
    ierr = PetscFree(rowlengths);CHKERRQ(ierr);

    /* determine max buffer needed and allocate it */
    maxnz = 0;
    for (i=0; i<size; i++) {
      maxnz = PetscMax(maxnz,procsnz[i]);
    }
    ierr = PetscMalloc(maxnz*sizeof(int),&cols);CHKERRQ(ierr);

    /* read in my part of the matrix column indices  */
    nz   = procsnz[0];
    ierr = PetscMalloc(nz*sizeof(int),&mycols);CHKERRQ(ierr);
    ierr = PetscBinaryRead(fd,mycols,nz,PETSC_INT);CHKERRQ(ierr);

    /* read in every one elses and ship off */
    for (i=1; i<size; i++) {
      nz   = procsnz[i];
      ierr = PetscBinaryRead(fd,cols,nz,PETSC_INT);CHKERRQ(ierr);
      ierr = MPI_Send(cols,nz,MPI_INT,i,tag,comm);CHKERRQ(ierr);
    }
    ierr = PetscFree(cols);CHKERRQ(ierr);
  } else {
    /* determine buffer space needed for message */
    nz = 0;
    for (i=0; i<m; i++) {
      nz += ourlens[i];
    }
    ierr = PetscMalloc((nz+1)*sizeof(int),&mycols);CHKERRQ(ierr);

    /* receive message of column indices*/
    ierr = MPI_Recv(mycols,nz,MPI_INT,0,tag,comm,&status);CHKERRQ(ierr);
    ierr = MPI_Get_count(&status,MPI_INT,&maxnz);CHKERRQ(ierr);
    if (maxnz != nz) SETERRQ(PETSC_ERR_FILE_UNEXPECTED,"something is wrong with file");
  }

  /* determine column ownership if matrix is not square */
  if (N != M) {
    n      = N/size + ((N % size) > rank);
    ierr   = MPI_Scan(&n,&cend,1,MPI_INT,MPI_SUM,comm);CHKERRQ(ierr);
    cstart = cend - n;
  } else {
    cstart = rstart;
    cend   = rend;
    n      = cend - cstart;
  }

  /* loop over local rows, determining number of off diagonal entries */
  ierr = PetscMemzero(offlens,m*sizeof(int));CHKERRQ(ierr);
  jj = 0;
  for (i=0; i<m; i++) {
    for (j=0; j<ourlens[i]; j++) {
      if (mycols[jj] < cstart || mycols[jj] >= cend) offlens[i]++;
      jj++;
    }
  }

  /* create our matrix */
  for (i=0; i<m; i++) {
    ourlens[i] -= offlens[i];
  }
  ierr = MatCreate(comm,m,n,M,N,newmat);CHKERRQ(ierr);
  ierr = MatSetType(*newmat,MATESI);CHKERRQ(ierr);
  ierr = MatSetFromOptions(*newmat);CHKERRQ(ierr);
  A = *newmat;
  ierr = MatSetOption(A,MAT_COLUMNS_SORTED);CHKERRQ(ierr);
  for (i=0; i<m; i++) {
    ourlens[i] += offlens[i];
  }

  if (!rank) {
    ierr = PetscMalloc(maxnz*sizeof(PetscScalar),&vals);CHKERRQ(ierr);

    /* read in my part of the matrix numerical values  */
    nz   = procsnz[0];
    ierr = PetscBinaryRead(fd,vals,nz,PETSC_SCALAR);CHKERRQ(ierr);
    
    /* insert into matrix */
    jj      = rstart;
    smycols = mycols;
    svals   = vals;
    for (i=0; i<m; i++) {
      ierr = MatSetValues(A,1,&jj,ourlens[i],smycols,svals,INSERT_VALUES);CHKERRQ(ierr);
      smycols += ourlens[i];
      svals   += ourlens[i];
      jj++;
    }

    /* read in other processors and ship out */
    for (i=1; i<size; i++) {
      nz   = procsnz[i];
      ierr = PetscBinaryRead(fd,vals,nz,PETSC_SCALAR);CHKERRQ(ierr);
      ierr = MPI_Send(vals,nz,MPIU_SCALAR,i,A->tag,comm);CHKERRQ(ierr);
    }
    ierr = PetscFree(procsnz);CHKERRQ(ierr);
  } else {
    /* receive numeric values */
    ierr = PetscMalloc((nz+1)*sizeof(PetscScalar),&vals);CHKERRQ(ierr);

    /* receive message of values*/
    ierr = MPI_Recv(vals,nz,MPIU_SCALAR,0,A->tag,comm,&status);CHKERRQ(ierr);
    ierr = MPI_Get_count(&status,MPIU_SCALAR,&maxnz);CHKERRQ(ierr);
    if (maxnz != nz) SETERRQ(PETSC_ERR_FILE_UNEXPECTED,"something is wrong with file");

    /* insert into matrix */
    jj      = rstart;
    smycols = mycols;
    svals   = vals;
    for (i=0; i<m; i++) {
      ierr     = MatSetValues(A,1,&jj,ourlens[i],smycols,svals,INSERT_VALUES);CHKERRQ(ierr);
      smycols += ourlens[i];
      svals   += ourlens[i];
      jj++;
    }
  }
  ierr = PetscFree(ourlens);CHKERRQ(ierr);
  ierr = PetscFree(vals);CHKERRQ(ierr);
  ierr = PetscFree(mycols);CHKERRQ(ierr);
  ierr = PetscFree(rowners);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END


EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatCreate_PetscESI"
int MatCreate_PetscESI(Mat V)
{
  int                            ierr;
  Mat                            v;
  esi::petsc::Matrix<double,int> *ve;

  PetscFunctionBegin;
  V->ops->destroy = 0;  /* since this is called from MatSetType() we have to make sure it doesn't get destroyed twice */
  ierr = MatSetType(V,MATESI);CHKERRQ(ierr);
  ierr = MatCreate(V->comm,V->m,V->n,V->M,V->N,&v);CHKERRQ(ierr);
  ierr = MatSetType(v,MATMPIAIJ);CHKERRQ(ierr);
  ve   = new esi::petsc::Matrix<double,int>(v);
  ierr = MatESISetOperator(V,ve);CHKERRQ(ierr);
  ierr = ve->deleteReference();CHKERRQ(ierr);
  ierr = PetscObjectDereference((PetscObject)v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

