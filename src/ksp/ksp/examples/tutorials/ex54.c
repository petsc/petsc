
static char help[] = "Creates a matrix using simple quadirlateral finite elements, and uses it to test GAMG\n\
  -m <size>       : problem size\n                                      \
  -alpha <v>      : scaling of material coeficient in embedded circle\n\n";

#include <petscksp.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  Mat            Amat,Pmat;
  PetscErrorCode ierr;
  PetscInt       npx,npy,i,m,M,its,Istart,Iend,j,Ii,bs,ix,ne=4;
  PetscReal      x,y,h;
  Vec            xx,bb;
  KSP            ksp;
  PetscReal      soft_alpha = 1.e-3;
  MPI_Comm       wcomm;
  PetscMPIInt    npe,mype,myXpe,myYpe;
  PC pc;
  PetscScalar DD1[4][4] = { {5.0, -2.0, -1.0, -2.0},
                            {-2.0, 5.0, -2.0, -1.0},
                            {-1.0, -2.0, 5.0, -2.0},
                            {-2.0, -1.0, -2.0, 5.0} };
#define EPS 0.0
  PetscScalar DD2[4][4] = {{1.0, EPS,  EPS,  EPS},
                           {EPS, 5.0,  -2.0, EPS},
                           {EPS, -2.0, 5.0,  EPS},
                           {EPS, EPS,  EPS,  1.0}};
  PetscScalar DD[4][4];

  PetscInitialize(&argc,&args,(char *)0,help);
  wcomm = PETSC_COMM_WORLD;
  ierr = MPI_Comm_rank( wcomm, &mype );   CHKERRQ(ierr);
  ierr = MPI_Comm_size( wcomm, &npe );    CHKERRQ(ierr);
  npx = npe; npy=1;
  while( npx > npy ) { npx /= 2; npy *= 2; }
  myXpe = mype%npx; myYpe = mype/npx;
PetscPrintf(PETSC_COMM_SELF,"[%d]%s npe=%d, npx=%d, npy=%d, myXpe=%d, myYpe=%d\n",mype,__FUNCT__,npe,npx,npy,myXpe,myYpe);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-ne",&ne,PETSC_NULL); CHKERRQ(ierr);
  h = 1./(npx*ne);
  /* ne*ne; number of local elements */
  ierr = PetscOptionsGetReal(PETSC_NULL,"-alpha",&soft_alpha,PETSC_NULL); CHKERRQ(ierr);
  M = (ne*npx+1)*(ne*npy+1); /* global number of nodes */
  m = (ne+1)*(ne+1); /* local number of nodes */
  if( myXpe != 0 ) m -= (ne+1);
  if( myYpe != 0 ) m -= (ne+1);
  if( myYpe != 0 && myXpe != 0 ) m++;
  /* create stiffness matrix */
  ierr = MatCreateMPIAIJ(wcomm,m,m,M,M,
                         18,PETSC_NULL,6,PETSC_NULL,&Amat);CHKERRQ(ierr);
  ierr = MatCreateMPIAIJ(wcomm,m,m,M,M,
                         18,PETSC_NULL,6,PETSC_NULL,&Pmat);CHKERRQ(ierr);
  bs = 1;
  /* Generate vectors */
  ierr = VecCreate(wcomm,&xx);   CHKERRQ(ierr);
  ierr = VecSetSizes(xx,m,M);    CHKERRQ(ierr);
  ierr = VecSetFromOptions(xx);  CHKERRQ(ierr);
  ierr = VecDuplicate(xx,&bb);   CHKERRQ(ierr);
  ierr = VecSet(bb,1.0);         CHKERRQ(ierr);

  ierr = MatGetOwnershipRange(Amat,&Istart,&Iend);CHKERRQ(ierr);
  PetscReal coords[2*m]; 
  /* forms the element stiffness for the Laplacian and coordinates */
  for (Ii=Istart,ix=0; Ii<Iend; Ii++,ix++) {
    j = Ii/(ne*npx+1); i = Ii - j*(ne*npx+1);
    /* coords */
    x = h*(Ii % (ne*npx+1)); y = h*(Ii/(ne*npx+1));
    coords[2*ix] = x; coords[2*ix+1] = y;
    if( i<ne*npx && j<ne*npy ) {
      PetscInt idx[4] = {Ii, Ii+1, Ii + (ne*npx+1) + 1, Ii + (ne*npx+1)};
      /* radius */
      PetscReal radius = sqrt( (x-.5+h/2)*(x-.5+h/2) + (y-.5+h/2)*(y-.5+h/2) );
      PetscReal alpha = 1.0;
      if( radius < 0.25 ){ 
        alpha = soft_alpha;
      }
      for(int ii=0;ii<4;ii++)for(int jj=0;jj<4;jj++) DD[ii][jj] = alpha*DD1[ii][jj];
      /* no BCs in Pamt */
      ierr = MatSetValues(Pmat,4,idx,4,idx,(const PetscScalar*)DD,ADD_VALUES);CHKERRQ(ierr);
      if( i>0 ) {
        ierr = MatSetValues(Amat,4,idx,4,idx,(const PetscScalar*)DD,ADD_VALUES);CHKERRQ(ierr);
      }
      else {
        /* a BC */
        for(int ii=0;ii<4;ii++)for(int jj=0;jj<4;jj++) DD[ii][jj] = alpha*DD2[ii][jj];
        ierr = MatSetValues(Amat,4,idx,4,idx,(const PetscScalar*)DD,ADD_VALUES);CHKERRQ(ierr);
      }
    }
  }
  ierr = MatAssemblyBegin(Amat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Amat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(Pmat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Pmat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  
  /* Setup solver */
  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);  CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,Amat,Pmat,SAME_NONZERO_PATTERN); CHKERRQ(ierr);
  ierr = KSPSetType(ksp,KSPCG);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  
  ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
  ierr = PCSetType(pc,PCGAMG); CHKERRQ(ierr);
  ierr = PCSetCoordinates( pc, 2, coords ); CHKERRQ(ierr);

  if(!PETSC_TRUE) {
    PetscViewer        viewer;
    ierr = PetscViewerASCIIOpen(wcomm, "Amat.m", &viewer);  CHKERRQ(ierr);
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

