#define PETSCMAT_DLL

/*
    Creates hypre ijmatrix from PETSc matrix
*/

#include "private/matimpl.h"          /*I "petscmat.h" I*/
EXTERN_C_BEGIN
#include "HYPRE.h"
#include "HYPRE_parcsr_ls.h"
EXTERN_C_END

#undef __FUNCT__
#define __FUNCT__ "MatHYPRE_IJMatrixPreallocate"
PetscErrorCode MatHYPRE_IJMatrixPreallocate(Mat A_d, Mat A_o,HYPRE_IJMatrix ij)
{
  PetscErrorCode ierr;
  PetscInt       i;
  PetscInt       n_d,*ia_d,n_o,*ia_o;
  PetscTruth     done_d=PETSC_FALSE,done_o=PETSC_FALSE;
  PetscInt       *nnz_d=PETSC_NULL,*nnz_o=PETSC_NULL;
  
  PetscFunctionBegin;
  if (A_d) { /* determine number of nonzero entries in local diagonal part */
    ierr = MatGetRowIJ(A_d,0,PETSC_FALSE,PETSC_FALSE,&n_d,&ia_d,PETSC_NULL,&done_d);CHKERRQ(ierr);
    if (done_d) {
      ierr = PetscMalloc(n_d*sizeof(PetscInt),&nnz_d);CHKERRQ(ierr);
      for (i=0; i<n_d; i++) {
        nnz_d[i] = ia_d[i+1] - ia_d[i];
      }
    }
    ierr = MatRestoreRowIJ(A_d,0,PETSC_FALSE,PETSC_FALSE,&n_d,&ia_d,PETSC_NULL,&done_d);CHKERRQ(ierr);
  }
  if (A_o) { /* determine number of nonzero entries in local off-diagonal part */
    ierr = MatGetRowIJ(A_o,0,PETSC_FALSE,PETSC_FALSE,&n_o,&ia_o,PETSC_NULL,&done_o);CHKERRQ(ierr);
    if (done_o) {
      ierr = PetscMalloc(n_o*sizeof(PetscInt),&nnz_o);CHKERRQ(ierr);
      for (i=0; i<n_o; i++) {
        nnz_o[i] = ia_o[i+1] - ia_o[i];
      }
    }
    ierr = MatRestoreRowIJ(A_o,0,PETSC_FALSE,PETSC_FALSE,&n_o,&ia_o,PETSC_NULL,&done_o);CHKERRQ(ierr);
  }
  if (done_d) {    /* set number of nonzeros in HYPRE IJ matrix */
    if (!done_o) { /* only diagonal part */
      ierr = PetscMalloc(n_d*sizeof(PetscInt),&nnz_o);CHKERRQ(ierr);
      for (i=0; i<n_d; i++) {
        nnz_o[i] = 0;
      }
    }
    ierr = HYPRE_IJMatrixSetDiagOffdSizes(ij,nnz_d,nnz_o);CHKERRQ(ierr);
    ierr = PetscFree(nnz_d);CHKERRQ(ierr);
    ierr = PetscFree(nnz_o);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "MatHYPRE_IJMatrixCreate"
PetscErrorCode MatHYPRE_IJMatrixCreate(Mat A,HYPRE_IJMatrix *ij)
{
  PetscErrorCode ierr;
  int            rstart,rend,cstart,cend;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_COOKIE,1);
  PetscValidType(A,1);
  PetscValidPointer(ij,2);
  ierr = MatPreallocated(A);CHKERRQ(ierr);
  rstart = A->rmap->rstart;
  rend   = A->rmap->rend;
  cstart = A->cmap->rstart;
  cend   = A->cmap->rend;
  ierr = HYPRE_IJMatrixCreate(((PetscObject)A)->comm,rstart,rend-1,cstart,cend-1,ij);CHKERRQ(ierr);
  ierr = HYPRE_IJMatrixSetObjectType(*ij,HYPRE_PARCSR);CHKERRQ(ierr);
  {
    PetscTruth  same;
    Mat         A_d,A_o;
    PetscInt    *colmap;
    ierr = PetscTypeCompare((PetscObject)A,MATMPIAIJ,&same);CHKERRQ(ierr);
    if (same) {
      ierr = MatMPIAIJGetSeqAIJ(A,&A_d,&A_o,&colmap);CHKERRQ(ierr);
      ierr = MatHYPRE_IJMatrixPreallocate(A_d,A_o,*ij);CHKERRQ(ierr);
      PetscFunctionReturn(0);
    }
    ierr = PetscTypeCompare((PetscObject)A,MATMPIBAIJ,&same);CHKERRQ(ierr);
    if (same) {
      ierr = MatMPIBAIJGetSeqBAIJ(A,&A_d,&A_o,&colmap);CHKERRQ(ierr);
      ierr = MatHYPRE_IJMatrixPreallocate(A_d,A_o,*ij);CHKERRQ(ierr);
      PetscFunctionReturn(0);
    }
    ierr = PetscTypeCompare((PetscObject)A,MATSEQAIJ,&same);CHKERRQ(ierr);
    if (same) {
      ierr = MatHYPRE_IJMatrixPreallocate(A,PETSC_NULL,*ij);CHKERRQ(ierr);
      PetscFunctionReturn(0);
    }
    ierr = PetscTypeCompare((PetscObject)A,MATSEQBAIJ,&same);CHKERRQ(ierr);
    if (same) {
      ierr = MatHYPRE_IJMatrixPreallocate(A,PETSC_NULL,*ij);CHKERRQ(ierr);
      PetscFunctionReturn(0);
    }
  }
  PetscFunctionReturn(0);
}

extern PetscErrorCode MatHYPRE_IJMatrixFastCopy_MPIAIJ(Mat,HYPRE_IJMatrix);
extern PetscErrorCode MatHYPRE_IJMatrixFastCopy_SeqAIJ(Mat,HYPRE_IJMatrix);
/*
    Copies the data over (column indices, numerical values) to hypre matrix  
*/

#undef __FUNCT__
#define __FUNCT__ "MatHYPRE_IJMatrixCopy"
PetscErrorCode MatHYPRE_IJMatrixCopy(Mat A,HYPRE_IJMatrix ij)
{
  PetscErrorCode    ierr;
  PetscInt          i,rstart,rend,ncols;
  const PetscScalar *values;
  const PetscInt    *cols;
  PetscTruth        flg;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)A,MATMPIAIJ,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = MatHYPRE_IJMatrixFastCopy_MPIAIJ(A,ij);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  ierr = PetscTypeCompare((PetscObject)A,MATSEQAIJ,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = MatHYPRE_IJMatrixFastCopy_SeqAIJ(A,ij);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  ierr = PetscLogEventBegin(MAT_Convert,A,0,0,0);CHKERRQ(ierr);
  ierr = HYPRE_IJMatrixInitialize(ij);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(A,&rstart,&rend);CHKERRQ(ierr);
  for (i=rstart; i<rend; i++) {
    ierr = MatGetRow(A,i,&ncols,&cols,&values);CHKERRQ(ierr);
    ierr = HYPRE_IJMatrixSetValues(ij,1,&ncols,&i,cols,values);CHKERRQ(ierr);
    ierr = MatRestoreRow(A,i,&ncols,&cols,&values);CHKERRQ(ierr);
  }
  ierr = HYPRE_IJMatrixAssemble(ij);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_Convert,A,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
    This copies the CSR format directly from the PETSc data structure to the hypre 
    data structure without calls to MatGetRow() or hypre's set values.

*/
#include "_hypre_IJ_mv.h"
#include "HYPRE_IJ_mv.h"
#include "../src/mat/impls/aij/mpi/mpiaij.h"

#undef __FUNCT__
#define __FUNCT__ "MatHYPRE_IJMatrixFastCopy_SeqIJ"
PetscErrorCode MatHYPRE_IJMatrixFastCopy_SeqAIJ(Mat A,HYPRE_IJMatrix ij)
{
  PetscErrorCode        ierr;
  Mat_SeqAIJ            *pdiag = (Mat_SeqAIJ*)A->data;;

  hypre_ParCSRMatrix    *par_matrix;
  hypre_AuxParCSRMatrix *aux_matrix;
  hypre_CSRMatrix       *hdiag,*hoffd;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_COOKIE,1);
  PetscValidType(A,1);
  PetscValidPointer(ij,2);

  ierr = PetscLogEventBegin(MAT_Convert,A,0,0,0);CHKERRQ(ierr);
  ierr = HYPRE_IJMatrixInitialize(ij);CHKERRQ(ierr);
  par_matrix = (hypre_ParCSRMatrix*)hypre_IJMatrixObject(ij);
  aux_matrix = (hypre_AuxParCSRMatrix*)hypre_IJMatrixTranslator(ij);
  hdiag = hypre_ParCSRMatrixDiag(par_matrix);
  hoffd = hypre_ParCSRMatrixOffd(par_matrix);

  /* 
       this is the Hack part where we monkey directly with the hypre datastructures
  */

  ierr = PetscMemcpy(hdiag->i,pdiag->i,(A->rmap->n + 1)*sizeof(PetscInt));
  ierr = PetscMemcpy(hdiag->j,pdiag->j,pdiag->nz*sizeof(PetscInt));
  ierr = PetscMemcpy(hdiag->data,pdiag->a,pdiag->nz*sizeof(PetscScalar));

  hypre_AuxParCSRMatrixNeedAux(aux_matrix) = 0;
  ierr = HYPRE_IJMatrixAssemble(ij);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_Convert,A,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatHYPRE_IJMatrixFastCopy_MPIAIJ"
PetscErrorCode MatHYPRE_IJMatrixFastCopy_MPIAIJ(Mat A,HYPRE_IJMatrix ij)
{
  PetscErrorCode        ierr;
  Mat_MPIAIJ            *pA = (Mat_MPIAIJ*)A->data;
  Mat_SeqAIJ            *pdiag,*poffd;
  PetscInt              i,*garray = pA->garray,*jj,cstart,*pjj;

  hypre_ParCSRMatrix    *par_matrix;
  hypre_AuxParCSRMatrix *aux_matrix;
  hypre_CSRMatrix       *hdiag,*hoffd;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_COOKIE,1);
  PetscValidType(A,1);
  PetscValidPointer(ij,2);
  pdiag = (Mat_SeqAIJ*) pA->A->data;
  poffd = (Mat_SeqAIJ*) pA->B->data;
  /* cstart is only valid for square MPIAIJ layed out in the usual way */
  ierr = MatGetOwnershipRange(A,&cstart,PETSC_NULL);CHKERRQ(ierr);

  ierr = PetscLogEventBegin(MAT_Convert,A,0,0,0);CHKERRQ(ierr);

  ierr = HYPRE_IJMatrixInitialize(ij);CHKERRQ(ierr);
  par_matrix = (hypre_ParCSRMatrix*)hypre_IJMatrixObject(ij);
  aux_matrix = (hypre_AuxParCSRMatrix*)hypre_IJMatrixTranslator(ij);
  hdiag = hypre_ParCSRMatrixDiag(par_matrix);
  hoffd = hypre_ParCSRMatrixOffd(par_matrix);

  /* 
       this is the Hack part where we monkey directly with the hypre datastructures
  */

  ierr = PetscMemcpy(hdiag->i,pdiag->i,(pA->A->rmap->n + 1)*sizeof(PetscInt));
  /* need to shift the diag column indices (hdiag->j) back to global numbering since hypre is expecting this */
  jj  = hdiag->j;
  pjj = pdiag->j;
  for (i=0; i<pdiag->nz; i++) {
    jj[i] = cstart + pjj[i];
  }
  ierr = PetscMemcpy(hdiag->data,pdiag->a,pdiag->nz*sizeof(PetscScalar));

  ierr = PetscMemcpy(hoffd->i,poffd->i,(pA->A->rmap->n + 1)*sizeof(PetscInt));
  /* need to move the offd column indices (hoffd->j) back to global numbering since hypre is expecting this
     If we hacked a hypre a bit more we might be able to avoid this step */
  jj  = hoffd->j;
  pjj = poffd->j;
  for (i=0; i<poffd->nz; i++) {
    jj[i] = garray[pjj[i]];
  }
  ierr = PetscMemcpy(hoffd->data,poffd->a,poffd->nz*sizeof(PetscScalar));

  hypre_AuxParCSRMatrixNeedAux(aux_matrix) = 0;
  ierr = HYPRE_IJMatrixAssemble(ij);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_Convert,A,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
    Does NOT copy the data over, instead uses DIRECTLY the pointers from the PETSc MPIAIJ format

    This is UNFINISHED and does NOT work! The problem is that hypre puts the diagonal entry first
    which will corrupt the PETSc data structure if we did this. Need a work around to this problem.
*/
#include "_hypre_IJ_mv.h"
#include "HYPRE_IJ_mv.h"

#undef __FUNCT__
#define __FUNCT__ "MatHYPRE_IJMatrixLink"
PetscErrorCode MatHYPRE_IJMatrixLink(Mat A,HYPRE_IJMatrix *ij)
{
  PetscErrorCode        ierr;
  int                   rstart,rend,cstart,cend;
  PetscTruth            flg;
  hypre_ParCSRMatrix    *par_matrix;
  hypre_AuxParCSRMatrix *aux_matrix;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_COOKIE,1);
  PetscValidType(A,1);
  PetscValidPointer(ij,2);
  ierr = PetscTypeCompare((PetscObject)A,MATMPIAIJ,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_ERR_SUP,"Can only use with PETSc MPIAIJ matrices");
  ierr = MatPreallocated(A);CHKERRQ(ierr);

  ierr = PetscLogEventBegin(MAT_Convert,A,0,0,0);CHKERRQ(ierr);
  rstart = A->rmap->rstart;
  rend   = A->rmap->rend;
  cstart = A->cmap->rstart;
  cend   = A->cmap->rend;
  ierr = HYPRE_IJMatrixCreate(((PetscObject)A)->comm,rstart,rend-1,cstart,cend-1,ij);CHKERRQ(ierr);
  ierr = HYPRE_IJMatrixSetObjectType(*ij,HYPRE_PARCSR);CHKERRQ(ierr);
 
  ierr = HYPRE_IJMatrixInitialize(*ij);CHKERRQ(ierr);
  par_matrix = (hypre_ParCSRMatrix*)hypre_IJMatrixObject(*ij);
  aux_matrix = (hypre_AuxParCSRMatrix*)hypre_IJMatrixTranslator(*ij);

  hypre_AuxParCSRMatrixNeedAux(aux_matrix) = 0;

  /* this is the Hack part where we monkey directly with the hypre datastructures */

  ierr = HYPRE_IJMatrixAssemble(*ij);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_Convert,A,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}



/*MC
   MATHYPRESTRUCT - MATHYPRESTRUCT = "hyprestruct" - A matrix type to be used for parallel sparse matrices
          based on the hypre HYPRE_StructMatrix.


.seealso: MatCreateMPIAIJ()
M*/

#include "petscda.h"   /*I "petscda.h" I*/
#include "../src/mat/impls/hypre/mhyp.h"

#undef __FUNCT__
#define __FUNCT__ "MatSetUp_HYPREStruct"
PetscErrorCode MatSetUp_HYPREStruct(Mat mat)
{
  Mat_HYPREStruct *ex = (Mat_HYPREStruct*) mat->data;

  PetscFunctionBegin;
  if (!ex->da) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"You are using the MATHYPRESTRUCT preconditioner but never called MatSetDA()");
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSetValuesLocal_HYPREStruct_3d"
PetscErrorCode PETSCMAT_DLLEXPORT MatSetValuesLocal_HYPREStruct_3d(Mat mat,PetscInt nrow,const PetscInt irow[],PetscInt ncol,const PetscInt icol[],const PetscScalar y[],InsertMode addv) 
{
  PetscErrorCode  ierr;
  PetscInt        i,j,stencil,nx,ny,gnx,gny,rstart,*gindices,index[3],row,entries[7] = {0,1,2,3,4,5,6};
  PetscScalar     values[7];
  Mat_HYPREStruct *ex = (Mat_HYPREStruct*) mat->data;

  PetscFunctionBegin;
  ierr = MatGetOwnershipRange(mat,&rstart,PETSC_NULL);CHKERRQ(ierr);
  ierr = DAGetGlobalIndices(ex->da,PETSC_NULL,&gindices);CHKERRQ(ierr);
  ierr = DAGetGhostCorners(ex->da,0,0,0,&gnx,&gny,0);CHKERRQ(ierr);
  ierr = DAGetCorners(ex->da,0,0,0,&nx,&ny,0);CHKERRQ(ierr);
  for (i=0; i<nrow; i++) {
    ierr = PetscMemzero(values,7*sizeof(PetscScalar));CHKERRQ(ierr);
    for (j=0; j<ncol; j++) {
      stencil = icol[j] - irow[i];
      if (!stencil) {
        values[3] = *y++;
      } else if (stencil == -1) {
        values[2] = *y++;
      } else if (stencil == 1) {
        values[4] = *y++;
      } else if (stencil == -gnx) {
        values[1] = *y++;
      } else if (stencil == gnx) {
        values[5] = *y++;
      } else if (stencil == -gnx*gny) {
        values[0] = *y++;
      } else if (stencil == gnx*gny) {
        values[6] = *y++;
      } else SETERRQ3(PETSC_ERR_ARG_WRONG,"Local row %D local column %D have bad stencil %D",irow[i],icol[j],stencil);
    }
    row = gindices[irow[i]] - rstart;
    index[0] = row % nx;
    index[1] = (row/nx) % ny;
    index[2] = row/(nx*ny);
    ierr = HYPRE_StructMatrixSetValues(ex->hmat,index,7,entries,values);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSetDA_HYPREStruct"
PetscErrorCode PETSCKSP_DLLEXPORT MatSetDA_HYPREStruct(Mat mat,DA da)
{
  PetscErrorCode  ierr;
  Mat_HYPREStruct *ex = (Mat_HYPREStruct*) mat->data;
  PetscInt         dim,dof,sw[3],nx,ny,nz;
  int              ilower[3],iupper[3],ssize,i;
  DAPeriodicType   p;
  DAStencilType    st;

  PetscFunctionBegin;
  ex->da = da;
  ierr   = PetscObjectReference((PetscObject)da);CHKERRQ(ierr); 

  ierr = DAGetInfo(ex->da,&dim,0,0,0,0,0,0,&dof,&sw[0],&p,&st);CHKERRQ(ierr);
  ierr = DAGetCorners(ex->da,&ilower[0],&ilower[1],&ilower[2],&iupper[0],&iupper[1],&iupper[2]);CHKERRQ(ierr);
  iupper[0] += ilower[0] - 1;    
  iupper[1] += ilower[1] - 1;    
  iupper[2] += ilower[2] - 1;    

  /* create the hypre grid object and set its information */
  if (dof > 1) SETERRQ(PETSC_ERR_SUP,"Currently only support for scalar problems");
  if (p) SETERRQ(PETSC_ERR_SUP,"Ask us to add periodic support by calling HYPRE_StructGridSetPeriodic()");
  ierr = HYPRE_StructGridCreate(ex->hcomm,dim,&ex->hgrid);CHKERRQ(ierr);

  ierr = HYPRE_StructGridSetExtents(ex->hgrid,ilower,iupper);CHKERRQ(ierr);
  ierr = HYPRE_StructGridAssemble(ex->hgrid);CHKERRQ(ierr);
    
  sw[1] = sw[0];
  sw[2] = sw[1];
  ierr = HYPRE_StructGridSetNumGhost(ex->hgrid,sw);CHKERRQ(ierr);

  /* create the hypre stencil object and set its information */
  if (sw[0] > 1) SETERRQ(PETSC_ERR_SUP,"Ask us to add support for wider stencils"); 
  if (st == DA_STENCIL_BOX) SETERRQ(PETSC_ERR_SUP,"Ask us to add support for box stencils"); 
  if (dim == 1) {
    int offsets[3][1] = {{-1},{0},{1}};
    ssize = 3;
    ierr = HYPRE_StructStencilCreate(dim,ssize,&ex->hstencil);CHKERRQ(ierr);
    for (i=0; i<ssize; i++) {
      ierr = HYPRE_StructStencilSetElement(ex->hstencil,i,offsets[i]);CHKERRQ(ierr);
    }
  } else if (dim == 2) {
    int offsets[5][2] = {{0,-1},{-1,0},{0,0},{1,0},{0,1}};
    ssize = 5;
    ierr = HYPRE_StructStencilCreate(dim,ssize,&ex->hstencil);CHKERRQ(ierr);
    for (i=0; i<ssize; i++) {
      ierr = HYPRE_StructStencilSetElement(ex->hstencil,i,offsets[i]);CHKERRQ(ierr);
    }
  } else if (dim == 3) {
    int offsets[7][3] = {{0,0,-1},{0,-1,0},{-1,0,0},{0,0,0},{1,0,0},{0,1,0},{0,0,1}}; 
    ssize = 7;
    ierr = HYPRE_StructStencilCreate(dim,ssize,&ex->hstencil);CHKERRQ(ierr);
    for (i=0; i<ssize; i++) {
      ierr = HYPRE_StructStencilSetElement(ex->hstencil,i,offsets[i]);CHKERRQ(ierr);
    }
  }
  
  /* create the HYPRE vector for rhs and solution */
  ierr = HYPRE_StructVectorCreate(ex->hcomm,ex->hgrid,&ex->hb);CHKERRQ(ierr);
  ierr = HYPRE_StructVectorCreate(ex->hcomm,ex->hgrid,&ex->hx);CHKERRQ(ierr);
  ierr = HYPRE_StructVectorInitialize(ex->hb);CHKERRQ(ierr);
  ierr = HYPRE_StructVectorInitialize(ex->hx);CHKERRQ(ierr);
  ierr = HYPRE_StructVectorAssemble(ex->hb);CHKERRQ(ierr);
  ierr = HYPRE_StructVectorAssemble(ex->hx);CHKERRQ(ierr);

  /* create the hypre matrix object and set its information */
  ierr = HYPRE_StructMatrixCreate(ex->hcomm,ex->hgrid,ex->hstencil,&ex->hmat);CHKERRQ(ierr);
  ierr = HYPRE_StructMatrixInitialize(ex->hmat);CHKERRQ(ierr);

  /* set the global and local sizes of the matrix */
  ierr = DAGetCorners(da,0,0,0,&nx,&ny,&nz);CHKERRQ(ierr);
  ierr = MatSetSizes(mat,dof*nx*ny*nz,dof*nx*ny*nz,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = PetscMapSetBlockSize(mat->rmap,1);CHKERRQ(ierr);
  ierr = PetscMapSetBlockSize(mat->cmap,1);CHKERRQ(ierr);
  ierr = PetscMapSetUp(mat->rmap);CHKERRQ(ierr);
  ierr = PetscMapSetUp(mat->cmap);CHKERRQ(ierr);

  if (dim == 3) {
    mat->ops->setvalueslocal = MatSetValuesLocal_HYPREStruct_3d;
  } else SETERRQ(PETSC_ERR_SUP,"Only support for 3d DA currently");

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMult_HYPREStruct"
PetscErrorCode MatMult_HYPREStruct(Mat A,Vec x,Vec y)
{
  PetscErrorCode  ierr;
  PetscScalar     *xx,*yy;
  int             ilower[3],iupper[3];
  Mat_HYPREStruct *mx = (Mat_HYPREStruct *)(A->data);

  PetscFunctionBegin;
  ierr = DAGetCorners(mx->da,&ilower[0],&ilower[1],&ilower[2],&iupper[0],&iupper[1],&iupper[2]);CHKERRQ(ierr);
  iupper[0] += ilower[0] - 1;    
  iupper[1] += ilower[1] - 1;    
  iupper[2] += ilower[2] - 1;    

  /* copy x values over to hypre */
  ierr = HYPRE_StructVectorInitialize(mx->hb);CHKERRQ(ierr);
  ierr = VecGetArray(x,&xx);CHKERRQ(ierr);
  ierr = HYPRE_StructVectorSetBoxValues(mx->hb,ilower,iupper,xx);CHKERRQ(ierr);
  ierr = VecRestoreArray(x,&xx);CHKERRQ(ierr);
  ierr = HYPRE_StructVectorAssemble(mx->hb);CHKERRQ(ierr);

  ierr = HYPRE_StructMatrixMatvec(1.0,mx->hmat,mx->hb,0.0,mx->hx);CHKERRQ(ierr);

  /* copy solution values back to PETSc */
  ierr = VecGetArray(y,&yy);CHKERRQ(ierr);
  ierr = HYPRE_StructVectorGetBoxValues(mx->hx,ilower,iupper,yy);CHKERRQ(ierr);
  ierr = VecRestoreArray(y,&yy);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}



#undef __FUNCT__  
#define __FUNCT__ "MatAssemblyEnd_HYPREStruct"
PetscErrorCode MatAssemblyEnd_HYPREStruct(Mat mat,MatAssemblyType mode)
{
  Mat_HYPREStruct *ex = (Mat_HYPREStruct*) mat->data;
  PetscInt         dim,dof,sw[3];
  int              ilower[3],iupper[3],i;
  DAPeriodicType   p;
  DAStencilType    st;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = DAGetInfo(ex->da,&dim,0,0,0,0,0,0,&dof,&sw[0],&p,&st);CHKERRQ(ierr);
  ierr = DAGetCorners(ex->da,&ilower[0],&ilower[1],&ilower[2],&iupper[0],&iupper[1],&iupper[2]);CHKERRQ(ierr);
  iupper[0] += ilower[0] - 1;    
  iupper[1] += ilower[1] - 1;    
  iupper[2] += ilower[2] - 1;    

  /* {

    int dummy = 3;
    PetscScalar values[10000];
   
    for (i=0; i<10000; i++) values[i] = 1.0;
    dummy = 3; ierr = HYPRE_StructMatrixSetBoxValues(ex->hmat,ilower,iupper,1,&dummy,values);CHKERRQ(ierr);

    for (i=0; i<10000; i++) values[i] = 0.0;
    dummy = 0; ierr = HYPRE_StructMatrixSetBoxValues(ex->hmat,ilower,iupper,1,&dummy,values);CHKERRQ(ierr);
    dummy = 1; ierr = HYPRE_StructMatrixSetBoxValues(ex->hmat,ilower,iupper,1,&dummy,values);CHKERRQ(ierr);
    dummy = 2; ierr = HYPRE_StructMatrixSetBoxValues(ex->hmat,ilower,iupper,1,&dummy,values);CHKERRQ(ierr);
    dummy = 4; ierr = HYPRE_StructMatrixSetBoxValues(ex->hmat,ilower,iupper,1,&dummy,values);CHKERRQ(ierr);
    dummy = 5; ierr = HYPRE_StructMatrixSetBoxValues(ex->hmat,ilower,iupper,1,&dummy,values);CHKERRQ(ierr);
    dummy = 6; ierr = HYPRE_StructMatrixSetBoxValues(ex->hmat,ilower,iupper,1,&dummy,values);CHKERRQ(ierr);
    } */
  ierr = HYPRE_StructMatrixAssemble(ex->hmat);CHKERRQ(ierr);
  ierr = HYPRE_StructMatrixPrint("dummy",ex->hmat,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSetDA"
/*@
   MatSetDA - Sets the DA that is to be used by the HYPRE_StructMatrix PETSc matrix

   Collective on Mat

   Input Parameters:
+  mat - the matrix
-  da - the da

   Level: intermediate


@*/
PetscErrorCode PETSCKSP_DLLEXPORT MatSetDA(Mat mat,DA da)
{
  PetscErrorCode ierr,(*f)(Mat,DA);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidHeaderSpecific(da,DM_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)mat,"MatSetDA_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(mat,da);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatCreate_HYPREStruct"
PetscErrorCode PETSCMAT_DLLEXPORT MatCreate_HYPREStruct(Mat B)
{
  Mat_HYPREStruct *ex;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr            = PetscNewLog(B,Mat_HYPREStruct,&ex);CHKERRQ(ierr);
  B->data         = (void*)ex;
  B->rmap->bs     = 1;
  B->assembled    = PETSC_FALSE;
  B->mapping      = 0;

  B->insertmode   = NOT_SET_VALUES;

  B->ops->assemblyend    = MatAssemblyEnd_HYPREStruct;
  B->ops->mult           = MatMult_HYPREStruct;

  ierr = MPI_Comm_dup(((PetscObject)B)->comm,&(ex->hcomm));CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatSetDA_C","MatSetDA_HYPREStruct",MatSetDA_HYPREStruct);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)B,MATHYPRESTRUCT);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END
