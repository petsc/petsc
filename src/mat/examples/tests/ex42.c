/*$Id: ex42.c,v 1.19 2001/01/15 21:46:09 bsmith Exp balay $*/

static char help[] = 
"Tests MatIncreaseOverlap() and MatGetSubmatrices() for the parallel case.\n\
This example is similar to ex40.c; here the index sets used are random.\n\
Input arguments are:\n\
  -f <input_file> : file to load.  For a 5X5 example of the 5-pt. stencil,\n\
                       use the file petsc/src/mat/examples/matbinary.ex\n\
  -nd <size>      : > 0  no of domains per processor \n\
  -ov <overlap>   : >=0  amount of overlap between domains\n\n";

#include "petscsles.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **args)
{
  int         ierr,nd = 2,ov=1,i,j,size,m,n,rank,*idx;
  PetscTruth  flg;
  Mat         A,B,*submatA,*submatB;
  char        file[128]; 
  PetscViewer      fd;
  IS          *is1,*is2;
  PetscRandom r;
  Scalar      rand;

  PetscInitialize(&argc,&args,(char *)0,help);
#if defined(PETSC_USE_COMPLEX)
  SETERRA(1,"This example does not work with complex numbers");
#else
  
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);  CHKERRA(ierr);
  ierr = PetscOptionsGetString(PETSC_NULL,"-f",file,127,PETSC_NULL);CHKERRA(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-nd",&nd,PETSC_NULL);CHKERRA(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-ov",&ov,PETSC_NULL);CHKERRA(ierr);

  /* Read matrix and RHS */
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,PETSC_BINARY_RDONLY,&fd);CHKERRA(ierr);
  ierr = MatLoad(fd,MATMPIAIJ,&A);CHKERRA(ierr);
  ierr = PetscViewerDestroy(fd);CHKERRA(ierr);

  /* Read the matrix again as a seq matrix */
  ierr = PetscViewerBinaryOpen(PETSC_COMM_SELF,file,PETSC_BINARY_RDONLY,&fd);CHKERRA(ierr);
  ierr = MatLoad(fd,MATSEQAIJ,&B);CHKERRA(ierr);
  ierr = PetscViewerDestroy(fd);CHKERRA(ierr);
  
  /* Create the Random no generator */
  ierr = MatGetSize(A,&m,&n);CHKERRA(ierr);  
  ierr = PetscRandomCreate(PETSC_COMM_SELF,RANDOM_DEFAULT,&r);CHKERRA(ierr);

  /* Create the IS corresponding to subdomains */
  ierr = PetscMalloc(nd*sizeof(IS **),&is1);CHKERRA(ierr);
  ierr = PetscMalloc(nd*sizeof(IS **),&is2);CHKERRA(ierr);
  ierr = PetscMalloc(m *sizeof(int),&idx);CHKERRA(ierr);
  
  /* Create the random Index Sets */
  for (i=0; i<nd; i++) {
    /* Skip a few,so that the IS on different procs are diffeent*/
    for (j=0; j<rank; j++) {
      ierr   = PetscRandomGetValue(r,&rand);CHKERRA(ierr);
    }
    ierr   = PetscRandomGetValue(r,&rand);CHKERRA(ierr);
    size   = (int)(rand*m);
    for (j=0; j<size; j++) {
      ierr   = PetscRandomGetValue(r,&rand);CHKERRA(ierr);
      idx[j] = (int)(rand*m);
    }
    ierr = PetscSortInt(size,idx);CHKERRA(ierr);
    ierr = ISCreateGeneral(PETSC_COMM_SELF,size,idx,is1+i);CHKERRA(ierr);
    ierr = ISCreateGeneral(PETSC_COMM_SELF,size,idx,is2+i);CHKERRA(ierr);
  }

  ierr = MatIncreaseOverlap(A,nd,is1,ov);CHKERRA(ierr);
  ierr = MatIncreaseOverlap(B,nd,is2,ov);CHKERRA(ierr);

  for (i=0; i<nd; ++i) { 
    ierr = ISSort(is1[i]);CHKERRQ(ierr);
    ierr = ISSort(is2[i]);CHKERRQ(ierr);
  }
  
  ierr = MatGetSubMatrices(A,nd,is1,is1,MAT_INITIAL_MATRIX,&submatA);CHKERRA(ierr);
  ierr = MatGetSubMatrices(B,nd,is2,is2,MAT_INITIAL_MATRIX,&submatB);CHKERRA(ierr);
  
  /* Now see if the serial and parallel case have the same answers */
  for (i=0; i<nd; ++i) { 
    ierr = MatEqual(submatA[i],submatB[i],&flg);CHKERRA(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF,"proc:[%d], i=%d, flg =%d\n",rank,i,flg);CHKERRQ(ierr);
  }

  /* Free Allocated Memory */
  for (i=0; i<nd; ++i) { 
    ierr = ISDestroy(is1[i]);CHKERRA(ierr);
    ierr = ISDestroy(is2[i]);CHKERRA(ierr);
    ierr = MatDestroy(submatA[i]);CHKERRA(ierr);
    ierr = MatDestroy(submatB[i]);CHKERRA(ierr);
  }
  ierr = PetscFree(submatA);CHKERRA(ierr);
  ierr = PetscFree(submatB);CHKERRA(ierr);
  ierr = PetscRandomDestroy(r);CHKERRA(ierr);
  ierr = PetscFree(is1);CHKERRA(ierr);
  ierr = PetscFree(is2);CHKERRA(ierr);
  ierr = MatDestroy(A);CHKERRA(ierr);
  ierr = MatDestroy(B);CHKERRA(ierr);
  ierr = PetscFree(idx);CHKERRA(ierr);

  PetscFinalize();
#endif
  return 0;
}

