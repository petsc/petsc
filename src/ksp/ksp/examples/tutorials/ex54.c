
static char help[] = "Creates a matrix using simple quadirlateral finite elements, and uses it to test GAMG\n\
  -m <size>       : problem size\n                                      \
  -alpha <v>      : scaling of material coeficient in embedded circle\n\n";

#include <petscksp.h>

#undef __FUNCT__
#define __FUNCT__ "FormElementRhs"
PetscErrorCode FormElementRhs(PetscReal x,PetscReal y,PetscReal H,PetscScalar *r)
{
  r[0] = 0.; r[1] = 0.; r[2] = 0.; r[3] = 0.0;
  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  Mat            Amat,Pmat;
  PetscErrorCode ierr;
  PetscInt       i,m = 20,N,rdim,cdim,its,Istart,Iend,j,Ii,bs;
  PetscReal      x,y,h;
  Vec            xx,bb;
  KSP            ksp;
  PetscReal      soft_alpha = 1.e-3, *coords;
  PetscScalar    v;
  PetscInt       ii,jj;

  PetscInitialize(&argc,&args,(char *)0,help);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-m",&m,PETSC_NULL); CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(PETSC_NULL,"-alpha",&soft_alpha,PETSC_NULL); CHKERRQ(ierr);
  N = (m+1)*(m+1); /* dimension of matrix */
  /*M = m*m; number of elements */
  h = 1.0/(PetscReal)m;
  /* create stiffness matrix */
  ierr = MatCreateSeqAIJ(PETSC_COMM_SELF,N,N,9,PETSC_NULL,&Amat);CHKERRQ(ierr);
  ierr = MatCreateSeqAIJ(PETSC_COMM_SELF,N,N,9,PETSC_NULL,&Pmat);CHKERRQ(ierr);
  bs = 1;

  /* Generate vectors */
  ierr = MatGetSize(Amat,&rdim,&cdim);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_WORLD,&xx);CHKERRQ(ierr);
  ierr = VecSetSizes(xx,PETSC_DECIDE,rdim);CHKERRQ(ierr);
  ierr = VecSetFromOptions(xx);CHKERRQ(ierr);
  ierr = VecDuplicate(xx,&bb);CHKERRQ(ierr);

  /* forms the element stiffness for the Laplacian and coordinates */
  ierr = PetscMalloc(2*N*sizeof(PetscReal),&coords);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(Amat,&Istart,&Iend);CHKERRQ(ierr);
#define EPS 0.0
  PetscScalar DD1[4][4] = { {5.0, -2.0, -1.0, -2.0},
                            {-2.0, 5.0, -2.0, -1.0},
                            {-1.0, -2.0, 5.0, -2.0},
                            {-2.0, -1.0, -2.0, 5.0} };

  PetscScalar DD2[4][4] = {{1.0, EPS,  EPS,  EPS},
                           {EPS, 5.0,  -2.0, EPS},
                           {EPS, -2.0, 5.0,  EPS},
                           {EPS, EPS,  EPS,  1.0}};
  PetscScalar DD[4][4];
  v = 1.0;
  ierr = VecSet(bb,v); CHKERRQ(ierr);
  for (Ii=Istart; Ii<Iend; Ii++) {
    j = Ii/(m+1); i = Ii - j*(m+1);
    /* coords */
    x = h*(Ii % (m+1)); y = h*(Ii/(m+1));
    coords[2*Ii] = x; coords[2*Ii+1] = y;
    if( i<m && j<m ) {
      PetscInt idx[4] = {Ii, Ii+1, Ii + (m+1) + 1, Ii + (m+1)};
      /* radius */
      PetscReal radius = sqrt( (x-.5+h/2)*(x-.5+h/2) +  (y-.5+h/2)*(y-.5+h/2) );
      PetscReal alpha = 1.0;
      if( radius < 0.25 ) alpha = soft_alpha;
      for(ii=0;ii<4;ii++)for(jj=0;jj<4;jj++) DD[ii][jj] = alpha*DD1[ii][jj];
      /* no BCs in Pamt */
      ierr = MatSetValues(Pmat,4,idx,4,idx,(const PetscScalar*)DD,ADD_VALUES);CHKERRQ(ierr);
      if( i>0 ) {
        ierr = MatSetValues(Amat,4,idx,4,idx,(const PetscScalar*)DD,ADD_VALUES);CHKERRQ(ierr);
      }
      else {
        /* a BC */
        for(ii=0;ii<4;ii++)for(jj=0;jj<4;jj++) DD[ii][jj] = alpha*DD2[ii][jj];
        ierr = MatSetValues(Amat,4,idx,4,idx,(const PetscScalar*)DD,ADD_VALUES);CHKERRQ(ierr);
      }
    }
  }
  ierr = MatAssemblyBegin(Amat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Amat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(Pmat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Pmat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(bb);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(bb);CHKERRQ(ierr);

  /* Setup solver */
  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);  CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,Amat,Pmat,SAME_NONZERO_PATTERN); CHKERRQ(ierr);
  ierr = KSPSetType(ksp,KSPCG);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  {
    PC subpc;
    ierr = KSPGetPC(ksp,&subpc);CHKERRQ(ierr);
    ierr = PCSetType(subpc,PCGAMG); CHKERRQ(ierr);
    ierr = PCSetCoordinates( subpc, 2, coords ); CHKERRQ(ierr);
    ierr = PetscFree( coords );  CHKERRQ(ierr);
  }
  if(!PETSC_TRUE) {
    PetscViewer        viewer;
    ierr = PetscViewerASCIIOpen(PETSC_COMM_SELF, "Amat.m", &viewer);  CHKERRQ(ierr);
    ierr = PetscViewerSetFormat( viewer, PETSC_VIEWER_ASCII_MATLAB);  CHKERRQ(ierr);
    ierr = MatView(Amat,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy( &viewer );
  }

  /* solve */
  ierr = KSPSolve(ksp,bb,xx);CHKERRQ(ierr);
  ierr = KSPGetIterationNumber(ksp,&its);CHKERRQ(ierr);

  /* Free work space */
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = VecDestroy(&xx);CHKERRQ(ierr);
  ierr = VecDestroy(&bb);CHKERRQ(ierr);
  ierr = MatDestroy(&Amat);CHKERRQ(ierr);
  ierr = MatDestroy(&Pmat);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return 0;
}

