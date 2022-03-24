
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
  PetscInt       i,m = 2,N,M,idx[4],Nsub1,Nsub2,ol=1,x1,x2;
  PetscScalar    Ke[16];
  PetscReal      h;
  IS             *is1,*is2,*islocal1,*islocal2;
  PetscBool      flg;

  CHKERRQ(PetscInitialize(&argc,&args,(char*)0,help));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL));
  N    = (m+1)*(m+1); /* dimension of matrix */
  M    = m*m; /* number of elements */
  h    = 1.0/m;    /* mesh width */
  x1   = (m+1)/2;
  x2   = x1;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-x1",&x1,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-x2",&x2,NULL));
  /* create stiffness matrix */
  CHKERRQ(MatCreateSeqAIJ(PETSC_COMM_SELF,N,N,9,NULL,&C));

  /* forms the element stiffness for the Laplacian */
  CHKERRQ(FormElementStiffness(h*h,Ke));
  for (i=0; i<M; i++) {
    /* node numbers for the four corners of element */
    idx[0] = (m+1)*(i/m) + (i % m);
    idx[1] = idx[0]+1; idx[2] = idx[1] + m + 1; idx[3] = idx[2] - 1;
    CHKERRQ(MatSetValues(C,4,idx,4,idx,Ke,ADD_VALUES));
  }
  CHKERRQ(MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY));

  for (ol=0; ol<m+2; ++ol) {

    CHKERRQ(PCASMCreateSubdomains2D(m+1,m+1,x1,x2,1,0,&Nsub1,&is1,&islocal1));
    CHKERRQ(MatIncreaseOverlap(C,Nsub1,is1,ol));
    CHKERRQ(PCASMCreateSubdomains2D(m+1,m+1,x1,x2,1,ol,&Nsub2,&is2,&islocal2));

    CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"flg == 1 => both index sets are same\n"));
    if (Nsub1 != Nsub2) {
      CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"Error: No of indes sets don't match\n"));
    }

    for (i=0; i<Nsub1; ++i) {
      CHKERRQ(ISEqual(is1[i],is2[i],&flg));
      CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"i =  %D,flg = %d \n",i,(int)flg));

    }
    for (i=0; i<Nsub1; ++i) CHKERRQ(ISDestroy(&is1[i]));
    for (i=0; i<Nsub2; ++i) CHKERRQ(ISDestroy(&is2[i]));
    for (i=0; i<Nsub1; ++i) CHKERRQ(ISDestroy(&islocal1[i]));
    for (i=0; i<Nsub2; ++i) CHKERRQ(ISDestroy(&islocal2[i]));

    CHKERRQ(PetscFree(is1));
    CHKERRQ(PetscFree(is2));
    CHKERRQ(PetscFree(islocal1));
    CHKERRQ(PetscFree(islocal2));
  }
  CHKERRQ(MatDestroy(&C));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

   test:
      args: -m 7

TEST*/
