/*$Id: ex41.c,v 1.25 2001/08/07 03:03:07 balay Exp $*/

static char help[] = "Tests MatIncreaseOverlap() - the parallel case. This example\n\
is similar to ex40.c; here the index sets used are random. Input arguments are:\n\
  -f <input_file> : file to load.  For a 5X5 example of the 5-pt. stencil,\n\
                       use the file petsc/src/mat/examples/matbinary.ex\n\
  -nd <size>      : > 0  no of domains per processor \n\
  -ov <overlap>   : >=0  amount of overlap between domains\n\n";

#include "petscsles.h"

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  int         ierr,nd = 2,ov=1,i,j,size,m,n,rank,*idx;
  PetscTruth  flg;
  Mat         A,B;
  char        file[128]; 
  PetscViewer fd;
  IS          *is1,*is2;
  PetscRandom r;
  PetscScalar rand;

  PetscInitialize(&argc,&args,(char *)0,help);
#if defined(PETSC_USE_COMPLEX)
  SETERRQ(1,"This example does not work with complex numbers");
#else
  
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = PetscOptionsGetString(PETSC_NULL,"-f",file,127,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-nd",&nd,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-ov",&ov,PETSC_NULL);CHKERRQ(ierr);

  /* Read matrix and RHS */
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,PETSC_FILE_RDONLY,&fd);CHKERRQ(ierr);
  ierr = MatLoad(fd,MATMPIAIJ,&A);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(fd);CHKERRQ(ierr);

  /* Read the matrix again as a seq matrix */
  ierr = PetscViewerBinaryOpen(PETSC_COMM_SELF,file,PETSC_FILE_RDONLY,&fd);CHKERRQ(ierr);
  ierr = MatLoad(fd,MATSEQAIJ,&B);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(fd);CHKERRQ(ierr);
  
  /* Create the Random no generator */
  ierr = MatGetSize(A,&m,&n);CHKERRQ(ierr);  
  ierr = PetscRandomCreate(PETSC_COMM_SELF,RANDOM_DEFAULT,&r);CHKERRQ(ierr);
  
  /* Create the IS corresponding to subdomains */
  ierr = PetscMalloc(nd*sizeof(IS **),&is1);CHKERRQ(ierr);
  ierr = PetscMalloc(nd*sizeof(IS **),&is2);CHKERRQ(ierr);
  ierr = PetscMalloc(m *sizeof(int),&idx);CHKERRQ(ierr);

  /* Create the random Index Sets */
  for (i=0; i<nd; i++) {
    for (j=0; j<rank; j++) {
      ierr   = PetscRandomGetValue(r,&rand);CHKERRQ(ierr);
    }   
    ierr   = PetscRandomGetValue(r,&rand);CHKERRQ(ierr);
    size   = (int)(rand*m);
    for (j=0; j<size; j++) {
      ierr   = PetscRandomGetValue(r,&rand);CHKERRQ(ierr);
      idx[j] = (int)(rand*m);
    }
    ierr = ISCreateGeneral(PETSC_COMM_SELF,size,idx,is1+i);CHKERRQ(ierr);
    ierr = ISCreateGeneral(PETSC_COMM_SELF,size,idx,is2+i);CHKERRQ(ierr);
  }
  
  ierr = MatIncreaseOverlap(A,nd,is1,ov);CHKERRQ(ierr);
  ierr = MatIncreaseOverlap(B,nd,is2,ov);CHKERRQ(ierr);
  
  /* Now see if the serial and parallel case have the same answers */
  for (i=0; i<nd; ++i) { 
    int sz1,sz2;
    ierr = ISEqual(is1[i],is2[i],&flg);CHKERRQ(ierr);
    ierr = ISGetSize(is1[i],&sz1);CHKERRQ(ierr);
    ierr = ISGetSize(is2[i],&sz2);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF,"proc:[%d], i=%d, flg =%d sz1 = %d sz2 = %d\n",rank,i,flg,sz1,sz2);CHKERRQ(ierr);
    /* ISView(is1[i],PETSC_VIEWER_STDOUT_SELF);
    ISView(is2[i],PETSC_VIEWER_STDOUT_SELF); */
  }

  /* Free Allocated Memory */
  for (i=0; i<nd; ++i) { 
    ierr = ISDestroy(is1[i]);CHKERRQ(ierr);
    ierr = ISDestroy(is2[i]);CHKERRQ(ierr);
  }
  ierr = PetscRandomDestroy(r);CHKERRQ(ierr);
  ierr = PetscFree(is1);CHKERRQ(ierr);
  ierr = PetscFree(is2);CHKERRQ(ierr);
  ierr = MatDestroy(A);CHKERRQ(ierr);
  ierr = MatDestroy(B);CHKERRQ(ierr);
  ierr = PetscFree(idx);CHKERRQ(ierr);

  ierr = PetscFinalize();CHKERRQ(ierr);
#endif
  return 0;
}

