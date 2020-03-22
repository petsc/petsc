
static char help[] = "Creates a matrix using 9 pt stencil, and uses it to test MatIncreaseOverlap (needed for additive Schwarz preconditioner). \n\
  -m <size>       : problem size\n\
  -x1, -x2 <size> : no of subdomains in x and y directions\n\n";

#include <petscksp.h>

PetscErrorCode FormElementStiffness(PetscReal H,PetscScalar *Ke)
{
  Ke[0]  = H/6.0;    Ke[1]  = -.125*H; Ke[2]  = H/12.0;   Ke[3]  = -.125*H;
  Ke[4]  = -.125*H;  Ke[5]  = H/6.0;   Ke[6]  = -.125*H;  Ke[7]  = H/12.0;
  Ke[8]  = H/12.0;   Ke[9]  = -.125*H; Ke[10] = H/6.0;    Ke[11] = -.125*H;
  Ke[12] = -.125*H;  Ke[13] = H/12.0;  Ke[14] = -.125*H;  Ke[15] = H/6.0;
  return 0;
}
PetscErrorCode FormElementRhs(PetscReal x,PetscReal y,PetscReal H,PetscScalar *r)
{
  r[0] = 0.; r[1] = 0.; r[2] = 0.; r[3] = 0.0;
  return 0;
}

int main(int argc,char **args)
{
  Mat            C;
  PetscErrorCode ierr;
  PetscInt       i,m = 2,N,M,idx[4],Nsub1,Nsub2,ol=1,x1,x2;
  PetscScalar    Ke[16];
  PetscReal      h;
  IS             *is1,*is2,*islocal1,*islocal2;
  PetscBool      flg;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL);CHKERRQ(ierr);
  N    = (m+1)*(m+1); /* dimension of matrix */
  M    = m*m; /* number of elements */
  h    = 1.0/m;    /* mesh width */
  x1   = (m+1)/2;
  x2   = x1;
  ierr = PetscOptionsGetInt(NULL,NULL,"-x1",&x1,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-x2",&x2,NULL);CHKERRQ(ierr);
  /* create stiffness matrix */
  ierr = MatCreateSeqAIJ(PETSC_COMM_SELF,N,N,9,NULL,&C);CHKERRQ(ierr);

  /* forms the element stiffness for the Laplacian */
  ierr = FormElementStiffness(h*h,Ke);CHKERRQ(ierr);
  for (i=0; i<M; i++) {
    /* node numbers for the four corners of element */
    idx[0] = (m+1)*(i/m) + (i % m);
    idx[1] = idx[0]+1; idx[2] = idx[1] + m + 1; idx[3] = idx[2] - 1;
    ierr   = MatSetValues(C,4,idx,4,idx,Ke,ADD_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  for (ol=0; ol<m+2; ++ol) {

    ierr = PCASMCreateSubdomains2D(m+1,m+1,x1,x2,1,0,&Nsub1,&is1,&islocal1);CHKERRQ(ierr);
    ierr = MatIncreaseOverlap(C,Nsub1,is1,ol);CHKERRQ(ierr);
    ierr = PCASMCreateSubdomains2D(m+1,m+1,x1,x2,1,ol,&Nsub2,&is2,&islocal2);CHKERRQ(ierr);

    ierr = PetscPrintf(PETSC_COMM_SELF,"flg == 1 => both index sets are same\n");CHKERRQ(ierr);
    if (Nsub1 != Nsub2) {
      ierr = PetscPrintf(PETSC_COMM_SELF,"Error: No of indes sets don't match\n");CHKERRQ(ierr);
    }

    for (i=0; i<Nsub1; ++i) {
      ierr = ISEqual(is1[i],is2[i],&flg);CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_SELF,"i =  %D,flg = %d \n",i,(int)flg);CHKERRQ(ierr);

    }
    for (i=0; i<Nsub1; ++i) {ierr = ISDestroy(&is1[i]);CHKERRQ(ierr);}
    for (i=0; i<Nsub2; ++i) {ierr = ISDestroy(&is2[i]);CHKERRQ(ierr);}
    for (i=0; i<Nsub1; ++i) {ierr = ISDestroy(&islocal1[i]);CHKERRQ(ierr);}
    for (i=0; i<Nsub2; ++i) {ierr = ISDestroy(&islocal2[i]);CHKERRQ(ierr);}


    ierr = PetscFree(is1);CHKERRQ(ierr);
    ierr = PetscFree(is2);CHKERRQ(ierr);
    ierr = PetscFree(islocal1);CHKERRQ(ierr);
    ierr = PetscFree(islocal2);CHKERRQ(ierr);
  }
  ierr = MatDestroy(&C);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}



/*TEST

   test:
      args: -m 7

TEST*/
