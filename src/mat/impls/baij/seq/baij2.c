#ifndef lint
static char vcid[] = "$Id: baij2.c,v 1.1 1996/04/30 18:51:06 balay Exp balay $";
#endif

#include "baij.h"
#include "petsc.h"


static int MatIncreaseOverlap_SeqAIJ(Mat A, int is_max, IS *is, int ov)
{
  Mat_SeqBAIJ *a = (Mat_SeqBAIJ *) A->data;
  int         row, i,j,k,l,m,n, *idx,ierr, *nidx, isz, val;
  int         start, end, *ai, *aj;
  char        *table;

  m     = a->mbs;
  ai    = a->i;
  aj    = a->j;

  if (ov < 0)  SETERRQ(1,"MatIncreaseOverlap_SeqBAIJ: illegal overlap value used");

  table = (char *) PetscMalloc((m/BITSPERBYTE +1)*sizeof(char)); CHKPTRQ(table); 
  nidx  = (int *) PetscMalloc((m+1)*sizeof(int)); CHKPTRQ(nidx); 

  for ( i=0; i<is_max; i++ ) {
    /* Initialise the two local arrays */
    isz  = 0;
    PetscMemzero(table,(m/BITSPERBYTE +1)*sizeof(char));
                 
                /* Extract the indices, assume there can be duplicate entries */
    ierr = ISGetIndices(is[i],&idx);  CHKERRQ(ierr);
    ierr = ISGetSize(is[i],&n);  CHKERRQ(ierr);
    
    /* Enter these into the temp arrays i.e mark table[row], enter row into new index */
    for ( j=0; j<n ; ++j){
      if(!BT_LOOKUP(table, idx[j])) { nidx[isz++] = idx[j];}
    }
    ierr = ISRestoreIndices(is[i],&idx);  CHKERRQ(ierr);
    ierr = ISDestroy(is[i]); CHKERRQ(ierr);
    
    k = 0;
    for ( j=0; j<ov; j++){ /* for each overlap*/
      n = isz;
      for ( ; k<n ; k++){ /* do only those rows in nidx[k], which are not done yet */
        row   = nidx[k];
        start = ai[row];
        end   = ai[row+1];
        for ( l = start; l<end ; l++){
          val = aj[l];
          if (!BT_LOOKUP(table,val)) {nidx[isz++] = val;}
        }
      }
    }
    ierr = ISCreateSeq(MPI_COMM_SELF, isz, nidx, (is+i)); CHKERRQ(ierr);
  }
  PetscFree(table);
  PetscFree(nidx);
  return 0;
}
