/*$Id: ex41.c,v 1.17 2000/05/05 22:16:17 balay Exp bsmith $*/

static char help[] = "Tests MatIncreaseOverlap() - the parallel case. This example\n\
is similar to ex40.c; here the index sets used are random. Input arguments are:\n\
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
  Mat         A,B;
  char        file[128]; 
  Viewer      fd;
  IS          *is1,*is2;
  PetscRandom r;
  Scalar      rand;
  PetscInitialize(&argc,&args,(char *)0,help);
#if defined(PETSC_USE_COMPLEX)
  SETERRA(1,"This example does not work with complex numbers");
#else
  
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRA(ierr);
  ierr = OptionsGetString(PETSC_NULL,"-f",file,127,PETSC_NULL);CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-nd",&nd,PETSC_NULL);CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-ov",&ov,PETSC_NULL);CHKERRA(ierr);

  /* Read matrix and RHS */
  ierr = ViewerBinaryOpen(PETSC_COMM_WORLD,file,BINARY_RDONLY,&fd);CHKERRA(ierr);
  ierr = MatLoad(fd,MATMPIAIJ,&A);CHKERRA(ierr);
  ierr = ViewerDestroy(fd);CHKERRA(ierr);

  /* Read the matrix again as a seq matrix */
  ierr = ViewerBinaryOpen(PETSC_COMM_SELF,file,BINARY_RDONLY,&fd);CHKERRA(ierr);
  ierr = MatLoad(fd,MATSEQAIJ,&B);CHKERRA(ierr);
  ierr = ViewerDestroy(fd);CHKERRA(ierr);
  
  /* Create the Random no generator */
  ierr = MatGetSize(A,&m,&n);CHKERRA(ierr);  
  ierr = PetscRandomCreate(PETSC_COMM_SELF,RANDOM_DEFAULT,&r);CHKERRA(ierr);
  
  /* Create the IS corresponding to subdomains */
  is1    = (IS*)PetscMalloc(nd*sizeof(IS **));CHKPTRA(is1);
  is2    = (IS*)PetscMalloc(nd*sizeof(IS **));CHKPTRA(is2);
  idx    = (int*)PetscMalloc(m *sizeof(int));CHKPTRA(idx);

  /* Create the random Index Sets */
  for (i=0; i<nd; i++) {
    for (j=0; j<rank; j++) {
      ierr   = PetscRandomGetValue(r,&rand);CHKERRA(ierr);
    }   
    ierr   = PetscRandomGetValue(r,&rand);CHKERRA(ierr);
    size   = (int)(rand*m);
    for (j=0; j<size; j++) {
      ierr   = PetscRandomGetValue(r,&rand);CHKERRA(ierr);
      idx[j] = (int)(rand*m);
    }
    ierr = ISCreateGeneral(PETSC_COMM_SELF,size,idx,is1+i);CHKERRQ(ierr);
    ierr = ISCreateGeneral(PETSC_COMM_SELF,size,idx,is2+i);CHKERRQ(ierr);
  }
  
  ierr = MatIncreaseOverlap(A,nd,is1,ov);CHKERRA(ierr);
  ierr = MatIncreaseOverlap(B,nd,is2,ov);CHKERRA(ierr);
  
  /* Now see if the serial and parallel case have the same answers */
  for (i=0; i<nd; ++i) { 
    ierr = ISEqual(is1[i],is2[i],&flg);CHKERRA(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF,"proc:[%d], i=%d, flg =%d\n",rank,i,flg);CHKERRA(ierr);
  }

  /* Free Allocated Memory */
  for (i=0; i<nd; ++i) { 
    ierr = ISDestroy(is1[i]);CHKERRA(ierr);
    ierr = ISDestroy(is2[i]);CHKERRA(ierr);
  }
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

