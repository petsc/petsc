/*$Id: ex49.c,v 1.17 2000/10/24 20:26:04 bsmith Exp bsmith $*/

static char help[] = "Tests MatTranspose(), MatNorm(), MatValid(), and MatAXPY().\n\n";

#include "petscmat.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **argv)
{
  Mat        mat,tmat = 0;
  int        m = 4,n,i,j,ierr,size,rank;
  int        rstart,rend,rect = 0,nd,bs,*diag,*bdlen;
  PetscTruth flg,isbdiag;
  Scalar     v,**diagv;
  double     normf,normi,norm1;
  MatInfo    info;
  
  PetscInitialize(&argc,&argv,(char*)0,help);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-m",&m,PETSC_NULL);CHKERRA(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRA(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRA(ierr);
  n = m;
  ierr = PetscOptionsHasName(PETSC_NULL,"-rect1",&flg);CHKERRA(ierr);
  if (flg) {n += 2; rect = 1;}
  ierr = PetscOptionsHasName(PETSC_NULL,"-rect2",&flg);CHKERRA(ierr);
  if (flg) {n -= 2; rect = 1;}

  /* Create and assemble matrix */
  ierr = MatCreate(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,m,n,&mat);CHKERRA(ierr);
  ierr = MatSetFromOptions(mat);CHKERRA(ierr);
  ierr = MatGetOwnershipRange(mat,&rstart,&rend);CHKERRA(ierr);
  for (i=rstart; i<rend; i++) { 
    for (j=0; j<n; j++) { 
      v=10*i+j; 
      ierr = MatSetValues(mat,1,&i,1,&j,&v,INSERT_VALUES);CHKERRA(ierr);
    }
  }
  ierr = MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY);CHKERRA(ierr);
  ierr = MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY);CHKERRA(ierr);

  /* Test whether matrix has been corrupted (just to demonstrate this
     routine) not needed in most application codes. */
  ierr = MatValid(mat,(PetscTruth*)&flg);CHKERRA(ierr);
  if (!flg) SETERRA(1,"Corrupted matrix.");

  /* Print info about original matrix */
  ierr = MatGetInfo(mat,MAT_GLOBAL_SUM,&info);CHKERRA(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"original matrix nonzeros = %d, allocated nonzeros = %d\n",
                    (int)info.nz_used,(int)info.nz_allocated);CHKERRA(ierr);
  ierr = MatNorm(mat,NORM_FROBENIUS,&normf);CHKERRA(ierr);
  ierr = MatNorm(mat,NORM_1,&norm1);CHKERRA(ierr);
  ierr = MatNorm(mat,NORM_INFINITY,&normi);CHKERRA(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"original: Frobenious norm = %g, one norm = %g, infinity norm = %g\n",
                     normf,norm1,normi);CHKERRA(ierr);
  ierr = MatView(mat,PETSC_VIEWER_STDOUT_WORLD);CHKERRA(ierr);

  ierr = PetscTypeCompare((PetscObject)mat,MATSEQBDIAG,&isbdiag);CHKERRQ(ierr);
  if (!isbdiag) {
    ierr = PetscTypeCompare((PetscObject)mat,MATMPIBDIAG,&isbdiag);CHKERRQ(ierr);
  }
  if (isbdiag) {
    ierr = MatBDiagGetData(mat,&nd,&bs,&diag,&bdlen,&diagv);CHKERRA(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"original matrix: block diag format: %d diagonals, block size = %d\n",nd,bs);CHKERRA(ierr);
    for (i=0; i<nd; i++) {
      ierr = PetscPrintf(PETSC_COMM_WORLD," diag=%d, bdlength=%d\n",diag[i],bdlen[i]);CHKERRA(ierr);
    }
  }

  /* Form matrix transpose */
  ierr = PetscOptionsHasName(PETSC_NULL,"-in_place",&flg);CHKERRA(ierr);
  if (!rect && flg) {
    ierr = MatTranspose(mat,0);CHKERRA(ierr);   /* in-place transpose */
    tmat = mat; mat = 0;
  } else {      /* out-of-place transpose */
    ierr = MatTranspose(mat,&tmat);CHKERRA(ierr); 
  }

  /* Print info about transpose matrix */
  ierr = MatGetInfo(tmat,MAT_GLOBAL_SUM,&info);CHKERRA(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"transpose matrix nonzeros = %d, allocated nonzeros = %d\n",
                     (int)info.nz_used,(int)info.nz_allocated);CHKERRA(ierr);
  ierr = MatNorm(tmat,NORM_FROBENIUS,&normf);CHKERRA(ierr);
  ierr = MatNorm(tmat,NORM_1,&norm1);CHKERRA(ierr);
  ierr = MatNorm(tmat,NORM_INFINITY,&normi);CHKERRA(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"transpose: Frobenious norm = %g, one norm = %g, infinity norm = %g\n",
                     normf,norm1,normi);CHKERRA(ierr);
  ierr = MatView(tmat,PETSC_VIEWER_STDOUT_WORLD);CHKERRA(ierr);

  if (isbdiag) {
    ierr = MatBDiagGetData(tmat,&nd,&bs,&diag,&bdlen,&diagv);CHKERRA(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"transpose matrix: block diag format: %d diagonals, block size = %d\n",nd,bs);CHKERRA(ierr);
    for (i=0; i<nd; i++) {
      ierr = PetscPrintf(PETSC_COMM_WORLD," diag=%d, bdlength=%d\n",diag[i],bdlen[i]);CHKERRA(ierr);
    }
  }

  /* Test MatAXPY */
  if (mat && !rect) {
    Scalar alpha = 1.0;
    ierr = PetscOptionsGetScalar(PETSC_NULL,"-alpha",&alpha,PETSC_NULL);CHKERRA(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"matrix addition:  B = B + alpha * A\n");CHKERRA(ierr);
    ierr = MatAXPY(&alpha,mat,tmat);CHKERRA(ierr); 
    ierr = MatView(tmat,PETSC_VIEWER_STDOUT_WORLD);CHKERRA(ierr);
  }

  /* Free data structures */  
  ierr = MatDestroy(tmat);CHKERRA(ierr);
  if (mat) {ierr = MatDestroy(mat);CHKERRA(ierr);}

  PetscFinalize();
  return 0;
}
 
