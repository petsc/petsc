/*$Id: ex40.c,v 1.19 2001/01/15 21:46:09 bsmith Exp balay $*/

static char help[] = "Tests the parallel case for MatIncreaseOverlap(). Input arguments are:\n\
  -f <input_file> : file to load.  For a 5X5 example of the 5-pt. stencil,\n\
                       use the file petsc/src/mat/examples/matbinary.ex\n\
  -nd <size>      : > 0  number of domains per processor \n\
  -ov <overlap>   : >=0  amount of overlap between domains\n\n";

#include "petscsles.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **args)
{
  int         ierr,nd = 2,ov=1,i,size,start,m,n,end,rank;
  PetscTruth  flg;
  Mat         A,B;
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

  /* Read the matrix again as a sequential matrix */
  ierr = PetscViewerBinaryOpen(PETSC_COMM_SELF,file,PETSC_BINARY_RDONLY,&fd);CHKERRA(ierr);
  ierr = MatLoad(fd,MATSEQAIJ,&B);CHKERRA(ierr);
  ierr = PetscViewerDestroy(fd);CHKERRA(ierr);
  
  /* Create the IS corresponding to subdomains */
  ierr = PetscMalloc(nd*sizeof(IS **),&is1);CHKERRA(ierr);
  ierr = PetscMalloc(nd*sizeof(IS **),&is2);CHKERRA(ierr);

  /* Create the random Index Sets */
  ierr = MatGetSize(A,&m,&n);CHKERRA(ierr);
  ierr = PetscRandomCreate(PETSC_COMM_SELF,RANDOM_DEFAULT,&r);CHKERRA(ierr);
  for (i=0; i<nd; i++) {
    ierr = PetscRandomGetValue(r,&rand);CHKERRA(ierr);
    start = (int)(rand*m);
    ierr = PetscRandomGetValue(r,&rand);CHKERRA(ierr);
    end  = (int)(rand*m);
    size =  end - start;
    if (start > end) { start = end; size = -size ;}
    ierr = ISCreateStride(PETSC_COMM_SELF,size,start,1,is1+i);CHKERRA(ierr);
    ierr = ISCreateStride(PETSC_COMM_SELF,size,start,1,is2+i);CHKERRA(ierr);
  }
  ierr = MatIncreaseOverlap(A,nd,is1,ov);CHKERRA(ierr);
  ierr = MatIncreaseOverlap(B,nd,is2,ov);CHKERRA(ierr);

  /* Now see if the serial and parallel case have the same answers */
  for (i=0; i<nd; ++i) { 
    ierr = ISEqual(is1[i],is2[i],&flg);CHKERRA(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF,"proc:[%d], i=%d, flg =%d\n",rank,i,flg);CHKERRA(ierr);
  }

  /* Free allocated memory */
  for (i=0; i<nd; ++i) { 
    ierr = ISDestroy(is1[i]);CHKERRA(ierr);
    ierr = ISDestroy(is2[i]);CHKERRA(ierr);
  }
  ierr = PetscFree(is1);CHKERRA(ierr);
  ierr = PetscFree(is2);CHKERRA(ierr);
  ierr = PetscRandomDestroy(r);CHKERRA(ierr);
  ierr = MatDestroy(A);CHKERRA(ierr);
  ierr = MatDestroy(B);CHKERRA(ierr);

  PetscFinalize();
#endif
  return 0;
}

