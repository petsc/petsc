/*$Id: fdaij.c,v 1.33 2000/07/10 03:39:32 bsmith Exp bsmith $*/

#include "src/mat/impls/aij/seq/aij.h"
#include "src/vec/vecimpl.h"

EXTERN int MatGetColumnIJ_SeqAIJ(Mat,int,PetscTruth,int*,int**,int**,PetscTruth*);
EXTERN int MatRestoreColumnIJ_SeqAIJ(Mat,int,PetscTruth,int*,int**,int**,PetscTruth*);

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"MatFDColoringCreate_SeqAIJ"
int MatFDColoringCreate_SeqAIJ(Mat mat,ISColoring iscoloring,MatFDColoring c)
{
  int        i,*is,n,nrows,N = mat->N,j,k,m,*rows,ierr,*ci,*cj,ncols,col;
  int        nis = iscoloring->n,*rowhit,*columnsforrow,l;
  IS         *isa = iscoloring->is;
  PetscTruth done,flg;

  PetscFunctionBegin;
  if (!mat->assembled) {
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Matrix must be assembled by calls to MatAssemblyBegin/End();");
  }

  c->M             = mat->M;  /* set total rows, columns and local rows */
  c->N             = mat->N;
  c->m             = mat->M;
  c->rstart        = 0;

  c->ncolors       = nis;
  c->ncolumns      = (int*)PetscMalloc(nis*sizeof(int));CHKPTRQ(c->ncolumns);
  c->columns       = (int**)PetscMalloc(nis*sizeof(int *));CHKPTRQ(c->columns); 
  c->nrows         = (int*)PetscMalloc(nis*sizeof(int));CHKPTRQ(c->nrows);
  c->rows          = (int**)PetscMalloc(nis*sizeof(int *));CHKPTRQ(c->rows);
  c->columnsforrow = (int**)PetscMalloc(nis*sizeof(int *));CHKPTRQ(c->columnsforrow);

  /*
      Calls the _SeqAIJ() version of these routines to make sure it does not 
     get the reduced (by inodes) version of I and J
  */
  ierr = MatGetColumnIJ_SeqAIJ(mat,0,PETSC_FALSE,&ncols,&ci,&cj,&done);CHKERRQ(ierr);

  /*
     Temporary option to allow for debugging/testing
  */
  ierr = OptionsHasName(PETSC_NULL,"-matfdcoloring_slow",&flg);CHKERRQ(ierr);

  rowhit        = (int*)PetscMalloc((N+1)*sizeof(int));CHKPTRQ(rowhit);
  columnsforrow = (int*)PetscMalloc((N+1)*sizeof(int));CHKPTRQ(columnsforrow);

  for (i=0; i<nis; i++) {
    ierr = ISGetLocalSize(isa[i],&n);CHKERRQ(ierr);
    ierr = ISGetIndices(isa[i],&is);CHKERRQ(ierr);
    c->ncolumns[i] = n;
    if (n) {
      c->columns[i]  = (int*)PetscMalloc(n*sizeof(int));CHKPTRQ(c->columns[i]);
      ierr = PetscMemcpy(c->columns[i],is,n*sizeof(int));CHKERRQ(ierr);
    } else {
      c->columns[i]  = 0;
    }

    if (flg) { /* ------------------------------------------------------------------------------*/
      /* crude version requires O(N*N) work */
      ierr = PetscMemzero(rowhit,N*sizeof(int));CHKERRQ(ierr);
      /* loop over columns*/
      for (j=0; j<n; j++) {
        col  = is[j];
        rows = cj + ci[col]; 
        m    = ci[col+1] - ci[col];
        /* loop over columns marking them in rowhit */
        for (k=0; k<m; k++) {
          rowhit[*rows++] = col + 1;
        }
      }
      /* count the number of hits */
      nrows = 0;
      for (j=0; j<N; j++) {
        if (rowhit[j]) nrows++;
      }
      c->nrows[i]         = nrows;
      c->rows[i]          = (int*)PetscMalloc(nrows*sizeof(int));CHKPTRQ(c->rows[i]);
      c->columnsforrow[i] = (int*)PetscMalloc(nrows*sizeof(int));CHKPTRQ(c->columnsforrow[i]);
      nrows = 0;
      for (j=0; j<N; j++) {
        if (rowhit[j]) {
          c->rows[i][nrows]           = j;
          c->columnsforrow[i][nrows] = rowhit[j] - 1;
          nrows++;
        }
      }
    } else {  /*-------------------------------------------------------------------------------*/
      /* efficient version, using rowhit as a linked list */
      int currentcol,fm,mfm;
      rowhit[N] = N;
      nrows     = 0;
      /* loop over columns */
      for (j=0; j<n; j++) {
        col   = is[j];
        rows  = cj + ci[col]; 
        m     = ci[col+1] - ci[col];
        /* loop over columns marking them in rowhit */
        fm    = N; /* fm points to first entry in linked list */
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
            SETERRQ(PETSC_ERR_PLIB,"Detected invalid coloring");
          }

        }
      }
      c->nrows[i]         = nrows;
      c->rows[i]          = (int *)PetscMalloc((nrows+1)*sizeof(int));CHKPTRQ(c->rows[i]);
      c->columnsforrow[i] = (int *)PetscMalloc((nrows+1)*sizeof(int));CHKPTRQ(c->columnsforrow[i]);
      /* now store the linked list of rows into c->rows[i] */
      nrows = 0;
      fm    = rowhit[N];
      do {
        c->rows[i][nrows]            = fm;
        c->columnsforrow[i][nrows++] = columnsforrow[fm];
        fm                           = rowhit[fm];
      } while (fm < N);
    } /* ---------------------------------------------------------------------------------------*/
    ierr = ISRestoreIndices(isa[i],&is);CHKERRQ(ierr);  
  }
  ierr = MatRestoreColumnIJ_SeqAIJ(mat,0,PETSC_FALSE,&ncols,&ci,&cj,&done);CHKERRQ(ierr);

  ierr = PetscFree(rowhit);CHKERRQ(ierr);
  ierr = PetscFree(columnsforrow);CHKERRQ(ierr);

  /* Optimize by adding the vscale, and scaleforrow[][] fields */
  /*
       see the version for MPIAIJ
  */
  ierr = VecCreateGhost(mat->comm,mat->m,PETSC_DETERMINE,0,PETSC_NULL,&c->vscale);CHKERRQ(ierr)
  c->vscaleforrow   = (int**)PetscMalloc(c->ncolors*sizeof(int*));CHKPTRQ(c->vscaleforrow);
  for (k=0; k<c->ncolors; k++) { 
    c->vscaleforrow[k] = (int*)PetscMalloc((c->nrows[k]+1)*sizeof(int));CHKPTRQ(c->vscaleforrow[k]);
    for (l=0; l<c->nrows[k]; l++) {
      col = c->columnsforrow[k][l];
      c->vscaleforrow[k][l] = col;
    }
  }

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"MatColoringPatch_SeqAIJ"
int MatColoringPatch_SeqAIJ(Mat mat,int ncolors,int *coloring,ISColoring *iscoloring)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ*)mat->data;
  int        n = a->n,*sizes,i,**ii,ierr,tag;
  IS         *is;

  PetscFunctionBegin;
  /* construct the index sets from the coloring array */
  sizes = (int*)PetscMalloc(ncolors*sizeof(int));CHKPTRQ(sizes);
  ierr = PetscMemzero(sizes,ncolors*sizeof(int));CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    sizes[coloring[i]-1]++;
  }
  ii    = (int**)PetscMalloc(ncolors*sizeof(int*));CHKPTRQ(ii);
  ii[0] = (int*)PetscMalloc(n*sizeof(int));CHKPTRQ(ii[0]);
  for (i=1; i<ncolors; i++) {
    ii[i] = ii[i-1] + sizes[i-1];
  }
  ierr = PetscMemzero(sizes,ncolors*sizeof(int));CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    ii[coloring[i]-1][sizes[coloring[i]-1]++] = i;
  }
  is  = (IS*)PetscMalloc(ncolors*sizeof(IS));CHKPTRQ(is);
  for (i=0; i<ncolors; i++) {
    ierr = ISCreateGeneral(PETSC_COMM_SELF,sizes[i],ii[i],is+i);CHKERRQ(ierr);
  }

  *iscoloring         = (ISColoring)PetscMalloc(sizeof(struct _p_ISColoring));CHKPTRQ(*iscoloring);
  (*iscoloring)->n    = ncolors;
  (*iscoloring)->is   = is;
  ierr = PetscCommDuplicate_Private(mat->comm,&(*iscoloring)->comm,&tag);CHKERRQ(ierr);
  ierr = PetscFree(sizes);CHKERRQ(ierr);
  ierr = PetscFree(ii[0]);CHKERRQ(ierr);
  ierr = PetscFree(ii);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
     Makes a longer coloring[] array and calls the usual code with that
*/
#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"MatColoringPatch_SeqAIJ_Inode"
int MatColoringPatch_SeqAIJ_Inode(Mat mat,int ncolors,int *coloring,ISColoring *iscoloring)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ*)mat->data;
  int        n = a->n,ierr,m = a->inode.node_count,j,*ns = a->inode.size,row;
  int        *colorused,i,*newcolor;

  PetscFunctionBegin;
  newcolor = (int*)PetscMalloc((n+1)*sizeof(int));CHKPTRQ(newcolor);

  /* loop over inodes, marking a color for each column*/
  row = 0;
  for (i=0; i<m; i++){
    for (j=0; j<ns[i]; j++) {
      newcolor[row++] = coloring[i] + j*ncolors;
    }
  }

  /* eliminate unneeded colors */
  colorused = (int*)PetscMalloc(5*ncolors*sizeof(int));CHKPTRQ(colorused);
  ierr      = PetscMemzero(colorused,5*ncolors*sizeof(int));CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    colorused[newcolor[i]-1] = 1;
  }

  for (i=1; i<5*ncolors; i++) {
    colorused[i] += colorused[i-1];
  }
  ncolors = colorused[5*ncolors-1];
  for (i=0; i<n; i++) {
    newcolor[i] = colorused[newcolor[i]-1];
  }
  ierr = PetscFree(colorused);CHKERRQ(ierr);

  ierr = MatColoringPatch_SeqAIJ(mat,ncolors,newcolor,iscoloring);CHKERRQ(ierr);
  ierr = PetscFree(newcolor);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}






