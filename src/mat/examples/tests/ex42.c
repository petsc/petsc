#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex42.c,v 1.8 1998/12/03 04:01:49 bsmith Exp bsmith $";
#endif

static char help[] = 
"Tests MatIncreaseOverlap() and MatGetSubmatrices() for the parallel case.\n\
This example is similar to ex40.c; here the index sets used are random.\n\
Input arguments are:\n\
  -f <input_file> : file to load.  For a 5X5 example of the 5-pt. stencil,\n\
                       use the file petsc/src/mat/examples/matbinary.ex\n\
  -nd <size>      : > 0  no of domains per processor \n\
  -ov <overlap>   : >=0  amount of overlap between domains\n\n";

  
#include "mat.h"
#include "sles.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **args)
{
  int        ierr, flg, nd = 2, ov=1,i ,j,size, m, n, rank, *idx;
  Mat        A, B, *submatA, *submatB;
  char       file[128]; 
  Viewer     fd;
  IS         *is1, *is2;
  PetscRandom   r;
  Scalar     rand;
  PetscInitialize(&argc,&args,(char *)0,help);
#if defined(USE_PETSC_COMPLEX)
  SETERRA(1,0,"This example does not work with complex numbers");
#else
  
  MPI_Comm_rank(PETSC_COMM_WORLD,&rank);  
  ierr = OptionsGetString(PETSC_NULL,"-f",file,127, &flg); CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-nd",&nd, &flg); CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-ov",&ov, &flg); CHKERRA(ierr);

  /* Read matrix and RHS */
  ierr = ViewerBinaryOpen(PETSC_COMM_WORLD,file,BINARY_RDONLY,&fd); CHKERRA(ierr);
  ierr = MatLoad(fd,MATMPIAIJ,&A); CHKERRA(ierr);
  ierr = ViewerDestroy(fd); CHKERRA(ierr);

  /* Read the matrix again as a seq matrix */
  ierr = ViewerBinaryOpen(PETSC_COMM_SELF,file,BINARY_RDONLY,&fd); CHKERRA(ierr);
  ierr = MatLoad(fd,MATSEQAIJ,&B); CHKERRA(ierr);
  ierr = ViewerDestroy(fd); CHKERRA(ierr);
  
  /* Create the Random no generator */
  ierr = MatGetSize(A,&m, &n); CHKERRA(ierr);  
  ierr = PetscRandomCreate(PETSC_COMM_SELF,RANDOM_DEFAULT,&r); CHKERRA(ierr);

  /* Create the IS corresponding to subdomains */
  is1    = (IS *) PetscMalloc( nd*sizeof(IS **) ); CHKPTRA(is1);
  is2    = (IS *) PetscMalloc( nd*sizeof(IS **) ); CHKPTRA(is2);
  idx    = (int*) PetscMalloc( m *sizeof(int )  ); CHKPTRA(idx);
  
  /* Create the random Index Sets */
  for (i=0; i<nd; i++) {
    /* Skip a few, so that the IS on different procs are diffeent*/
    for (j=0; j<rank; j++) {
      ierr   = PetscRandomGetValue(r, &rand); CHKERRA(ierr);
    }
    ierr   = PetscRandomGetValue(r, &rand); CHKERRA(ierr);
    size   = (int)(rand*m);
    for (j=0; j<size; j++) {
      ierr   = PetscRandomGetValue(r, &rand); CHKERRA(ierr);
      idx[j] = (int)(rand*m);
    }
    ierr = PetscSortInt(size,idx); CHKERRA(ierr);
    ierr = ISCreateGeneral(PETSC_COMM_SELF,size,idx,is1+i); CHKERRA(ierr);
    ierr = ISCreateGeneral(PETSC_COMM_SELF,size,idx,is2+i); CHKERRA(ierr);
  }

  ierr = MatIncreaseOverlap(A, nd, is1, ov); CHKERRA(ierr);
  ierr = MatIncreaseOverlap(B, nd, is2, ov); CHKERRA(ierr);

  for (i=0; i<nd; ++i) { 
    ierr = ISSort(is1[i]); CHKERRQ(ierr);
    ierr = ISSort(is2[i]); CHKERRQ(ierr);
  }
  
  ierr = MatGetSubMatrices(A,nd,is1,is1,MAT_INITIAL_MATRIX,&submatA);CHKERRA(ierr);
  ierr = MatGetSubMatrices(B,nd,is2,is2,MAT_INITIAL_MATRIX,&submatB);CHKERRA(ierr);
  
  /* Now see if the serial and parallel case have the same answers */
  for (i=0; i<nd; ++i) { 
    ierr = MatEqual(submatA[i], submatB[i],(PetscTruth*)&flg);CHKERRA(ierr);
    PetscPrintf(PETSC_COMM_SELF,"proc:[%d], i=%d, flg =%d\n",rank,i,flg);
  }

  /* Free Allocated Memory */
  for (i=0; i<nd; ++i) { 
    ISDestroy(is1[i]); 
    ISDestroy(is2[i]); 
    MatDestroy(submatA[i]);
    MatDestroy(submatB[i]);
  }
  PetscFree(submatA);
  PetscFree(submatB);
  PetscRandomDestroy(r);
  PetscFree(is1);
  PetscFree(is2);
  MatDestroy(A);
  MatDestroy(B);
  PetscFree(idx);

  PetscFinalize();
#endif
  return 0;
}

