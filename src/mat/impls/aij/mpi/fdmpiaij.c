
#include "src/mat/impls/aij/mpi/mpiaij.h"

EXTERN PetscErrorCode CreateColmap_MPIAIJ_Private(Mat);
EXTERN PetscErrorCode MatGetColumnIJ_SeqAIJ(Mat,PetscInt,PetscTruth,PetscInt*,PetscInt*[],PetscInt*[],PetscTruth*);
EXTERN PetscErrorCode MatRestoreColumnIJ_SeqAIJ(Mat,PetscInt,PetscTruth,PetscInt*,PetscInt*[],PetscInt*[],PetscTruth*);

#undef __FUNCT__  
#define __FUNCT__ "MatFDColoringCreate_MPIAIJ"
PetscErrorCode MatFDColoringCreate_MPIAIJ(Mat mat,ISColoring iscoloring,MatFDColoring c)
{
  Mat_MPIAIJ     *aij = (Mat_MPIAIJ*)mat->data;
  PetscErrorCode ierr;
  PetscMPIInt    size,*ncolsonproc,*disp,nn;
  PetscInt       i,*is,n,nrows,j,k,m,*rows = 0,*A_ci,*A_cj,ncols,col;
  PetscInt       nis = iscoloring->n,nctot,*cols,*B_ci,*B_cj;
  PetscInt       *rowhit,M = mat->m,cstart = aij->cstart,cend = aij->cend,colb;
  PetscInt       *columnsforrow,l;
  IS             *isa;
  PetscTruth     done,flg;

  PetscFunctionBegin;
  if (!mat->assembled) {
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Matrix must be assembled first; MatAssemblyBegin/End();");
  }

  ierr = ISColoringGetIS(iscoloring,PETSC_IGNORE,&isa);CHKERRQ(ierr);
  c->M             = mat->M;  /* set the global rows and columns and local rows */
  c->N             = mat->N;
  c->m             = mat->m;
  c->rstart        = aij->rstart;

  c->ncolors       = nis;
  ierr             = PetscMalloc(nis*sizeof(PetscInt),&c->ncolumns);CHKERRQ(ierr);
  ierr             = PetscMalloc(nis*sizeof(PetscInt*),&c->columns);CHKERRQ(ierr); 
  ierr             = PetscMalloc(nis*sizeof(PetscInt),&c->nrows);CHKERRQ(ierr);
  ierr             = PetscMalloc(nis*sizeof(PetscInt*),&c->rows);CHKERRQ(ierr);
  ierr             = PetscMalloc(nis*sizeof(PetscInt*),&c->columnsforrow);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory(c,5*nis*sizeof(PetscInt));CHKERRQ(ierr);

  /* Allow access to data structures of local part of matrix */
  if (!aij->colmap) {
    ierr = CreateColmap_MPIAIJ_Private(mat);CHKERRQ(ierr);
  }
  /*
      Calls the _SeqAIJ() version of these routines to make sure it does not 
     get the reduced (by inodes) version of I and J
  */
  ierr = MatGetColumnIJ_SeqAIJ(aij->A,0,PETSC_FALSE,&ncols,&A_ci,&A_cj,&done);CHKERRQ(ierr); 
  ierr = MatGetColumnIJ_SeqAIJ(aij->B,0,PETSC_FALSE,&ncols,&B_ci,&B_cj,&done);CHKERRQ(ierr); 

  ierr = MPI_Comm_size(mat->comm,&size);CHKERRQ(ierr); 
  ierr = PetscMalloc(2*size*sizeof(PetscInt*),&ncolsonproc);CHKERRQ(ierr);
  disp = ncolsonproc + size;

  ierr = PetscMalloc((M+1)*sizeof(PetscInt),&rowhit);CHKERRQ(ierr);
  ierr = PetscMalloc((M+1)*sizeof(PetscInt),&columnsforrow);CHKERRQ(ierr);

  /*
     Temporary option to allow for debugging/testing
  */
  ierr = PetscOptionsHasName(PETSC_NULL,"-matfdcoloring_slow",&flg);CHKERRQ(ierr);

  for (i=0; i<nis; i++) {
    ierr = ISGetLocalSize(isa[i],&n);CHKERRQ(ierr);
    ierr = ISGetIndices(isa[i],&is);CHKERRQ(ierr);
    c->ncolumns[i] = n;
    c->ncolumns[i] = n;
    if (n) {
      ierr = PetscMalloc(n*sizeof(PetscInt),&c->columns[i]);CHKERRQ(ierr);
      ierr = PetscLogObjectMemory(c,n*sizeof(PetscInt));CHKERRQ(ierr);
      ierr = PetscMemcpy(c->columns[i],is,n*sizeof(PetscInt));CHKERRQ(ierr);
    } else {
      c->columns[i]  = 0;
    }

    /* Determine the total (parallel) number of columns of this color */
    nn   = (PetscMPIInt)n;
    ierr = MPI_Allgather(&nn,1,MPI_INT,ncolsonproc,1,MPI_INT,mat->comm);CHKERRQ(ierr);
    nctot = 0; for (j=0; j<size; j++) {nctot += ncolsonproc[j];}
    if (!nctot) {
      ierr = PetscLogInfo(((PetscObject)mat,"MatFDColoringCreate_MPIAIJ: Coloring of matrix has some unneeded colors with no corresponding rows\n"));CHKERRQ(ierr);
    }

    disp[0] = 0;
    for (j=1; j<size; j++) {
      disp[j] = disp[j-1] + ncolsonproc[j-1];
    }
    
    /* Get complete list of columns for color on each processor */
    ierr = PetscMalloc((nctot+1)*sizeof(PetscInt),&cols);CHKERRQ(ierr);
    ierr = MPI_Allgatherv(is,n,MPIU_INT,cols,ncolsonproc,disp,MPIU_INT,mat->comm);CHKERRQ(ierr);

    /*
       Mark all rows affect by these columns
    */
    if (!flg) {/*-----------------------------------------------------------------------------*/
      /* crude, fast version */
      ierr = PetscMemzero(rowhit,M*sizeof(PetscInt));CHKERRQ(ierr);
      /* loop over columns*/
      for (j=0; j<nctot; j++) {
        col  = cols[j];
        if (col >= cstart && col < cend) {
          /* column is in diagonal block of matrix */
          rows = A_cj + A_ci[col-cstart]; 
          m    = A_ci[col-cstart+1] - A_ci[col-cstart];
        } else {
#if defined (PETSC_USE_CTABLE)
          ierr = PetscTableFind(aij->colmap,col+1,&colb);CHKERRQ(ierr)
	  colb --;
#else
          colb = aij->colmap[col] - 1;
#endif
          if (colb == -1) {
            m = 0; 
          } else {
            rows = B_cj + B_ci[colb]; 
            m    = B_ci[colb+1] - B_ci[colb];
          }
        }
        /* loop over columns marking them in rowhit */
        for (k=0; k<m; k++) {
          rowhit[*rows++] = col + 1;
        }
      }

      /* count the number of hits */
      nrows = 0;
      for (j=0; j<M; j++) {
        if (rowhit[j]) nrows++;
      }
      c->nrows[i]         = nrows;
      ierr                = PetscMalloc((nrows+1)*sizeof(PetscInt),&c->rows[i]);CHKERRQ(ierr);
      ierr                = PetscMalloc((nrows+1)*sizeof(PetscInt),&c->columnsforrow[i]);CHKERRQ(ierr);
      ierr = PetscLogObjectMemory(c,2*(nrows+1)*sizeof(PetscInt));CHKERRQ(ierr);
      nrows = 0;
      for (j=0; j<M; j++) {
        if (rowhit[j]) {
          c->rows[i][nrows]           = j;
          c->columnsforrow[i][nrows] = rowhit[j] - 1;
          nrows++;
        }
      }
    } else {/*-------------------------------------------------------------------------------*/
      /* slow version, using rowhit as a linked list */
      PetscInt currentcol,fm,mfm;
      rowhit[M] = M;
      nrows     = 0;
      /* loop over columns*/
      for (j=0; j<nctot; j++) {
        col  = cols[j];
        if (col >= cstart && col < cend) {
          /* column is in diagonal block of matrix */
          rows = A_cj + A_ci[col-cstart]; 
          m    = A_ci[col-cstart+1] - A_ci[col-cstart];
        } else {
#if defined (PETSC_USE_CTABLE)
	  ierr = PetscTableFind(aij->colmap,col+1,&colb);CHKERRQ(ierr);
          colb --;
#else
          colb = aij->colmap[col] - 1;
#endif
          if (colb == -1) {
            m = 0; 
          } else {
            rows = B_cj + B_ci[colb]; 
            m    = B_ci[colb+1] - B_ci[colb];
          }
        }
        /* loop over columns marking them in rowhit */
        fm    = M; /* fm points to first entry in linked list */
        for (k=0; k<m; k++) {
          currentcol = *rows++;
	  /* is it already in the list? */
          do {
            mfm  = fm;
            fm   = rowhit[fm];
          } while (fm < currentcol);
          /* not in list so add it */
          if (fm != currentcol) {
            nrows++;
            columnsforrow[currentcol] = col;
            /* next three lines insert new entry into linked list */
            rowhit[mfm]               = currentcol;
            rowhit[currentcol]        = fm;
            fm                        = currentcol; 
            /* fm points to present position in list since we know the columns are sorted */
          } else {
            SETERRQ(PETSC_ERR_PLIB,"Invalid coloring of matrix detected");
          }
        }
      }
      c->nrows[i]         = nrows;
      ierr = PetscMalloc((nrows+1)*sizeof(PetscInt),&c->rows[i]);CHKERRQ(ierr);
      ierr = PetscMalloc((nrows+1)*sizeof(PetscInt),&c->columnsforrow[i]);CHKERRQ(ierr);
      ierr = PetscLogObjectMemory(c,(nrows+1)*sizeof(PetscInt));CHKERRQ(ierr);
      /* now store the linked list of rows into c->rows[i] */
      nrows = 0;
      fm    = rowhit[M];
      do {
        c->rows[i][nrows]            = fm;
        c->columnsforrow[i][nrows++] = columnsforrow[fm];
        fm                           = rowhit[fm];
      } while (fm < M);
    } /* ---------------------------------------------------------------------------------------*/
    ierr = PetscFree(cols);CHKERRQ(ierr);
  }

  /* Optimize by adding the vscale, and scaleforrow[][] fields */
  /*
       vscale will contain the "diagonal" on processor scalings followed by the off processor
  */
  ierr = VecCreateGhost(mat->comm,aij->A->m,PETSC_DETERMINE,aij->B->n,aij->garray,&c->vscale);CHKERRQ(ierr)
  ierr = PetscMalloc(c->ncolors*sizeof(PetscInt*),&c->vscaleforrow);CHKERRQ(ierr);
  for (k=0; k<c->ncolors; k++) { 
    ierr = PetscMalloc((c->nrows[k]+1)*sizeof(PetscInt),&c->vscaleforrow[k]);CHKERRQ(ierr);
    for (l=0; l<c->nrows[k]; l++) {
      col = c->columnsforrow[k][l];
      if (col >= cstart && col < cend) {
        /* column is in diagonal block of matrix */
        colb = col - cstart;
      } else {
        /* column  is in "off-processor" part */
#if defined (PETSC_USE_CTABLE)
        ierr = PetscTableFind(aij->colmap,col+1,&colb);CHKERRQ(ierr);
        colb --;
#else
        colb = aij->colmap[col] - 1;
#endif
        colb += cend - cstart;
      }
      c->vscaleforrow[k][l] = colb;
    }
  }
  ierr = ISColoringRestoreIS(iscoloring,&isa);CHKERRQ(ierr);

  ierr = PetscFree(rowhit);CHKERRQ(ierr);
  ierr = PetscFree(columnsforrow);CHKERRQ(ierr);
  ierr = PetscFree(ncolsonproc);CHKERRQ(ierr);
  ierr = MatRestoreColumnIJ_SeqAIJ(aij->A,0,PETSC_FALSE,&ncols,&A_ci,&A_cj,&done);CHKERRQ(ierr); 
  ierr = MatRestoreColumnIJ_SeqAIJ(aij->B,0,PETSC_FALSE,&ncols,&B_ci,&B_cj,&done);CHKERRQ(ierr); 
  PetscFunctionReturn(0);
}






