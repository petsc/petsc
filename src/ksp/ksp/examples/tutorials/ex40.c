
static char help[] = "Lattice Gauge 2D model.\n"
"Parameters:\n"
"-size n          to use a grid size of n, i.e n space and n time steps\n"
"-beta b          controls the randomness of the gauge field\n"
"-rho r           the quark mass (?)";

#include "petscksp.h"
#include "petscasa.h"
#include "petscda.h"

PetscErrorCode computeMinEigVal(Mat A, PetscInt its, PetscScalar *eig);

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int Argc,char **Args)
{
  PetscTruth      flg;
  PetscInt        n = 6,i;
  PetscScalar     rho = 1.0;
  PetscReal       h;
  PetscReal       beta = 1.0;
  ADDA            adda;
  PetscInt        nodes[2];
  PetscTruth      periodic[2];
  PetscInt        refine[2];
  PetscRandom     rctx;
  PetscMPIInt     comm_size;
  Mat             H;
  PetscInt        *lcs, *lce;
  PetscInt        x, y;
  PetscReal       r1, r2;
  PetscScalar     uxy1, uxy2;
  ADDAIdx         sxy, sxy_m;
  PetscScalar     val, valconj;
  Mat             HtH;
  Vec             b, Htb;
  Vec             xvec;
  KSP             kspmg;
  PC              pcmg;
  PetscErrorCode  ierr;

  PetscInitialize(&Argc,&Args,(char *)0,help);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-size",&n,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(PETSC_NULL,"-beta",&beta,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetScalar(PETSC_NULL,"-rho",&rho,&flg);CHKERRQ(ierr);

  /* Set the fudge parameters, we scale the whole thing by 1/(2*h) later */
  h = 1.;
  rho *= 1./(2.*h);
  
  /* Geometry info */
  for(i=0; i<2; i++) {
    nodes[i] = n;
    periodic[i] = PETSC_TRUE;
    refine[i] = 3;
  }
  ierr = ADDACreate(PETSC_COMM_WORLD, 2, nodes, PETSC_NULL, 2 /* this is the # of dof's */,
		    periodic, &adda);CHKERRQ(ierr);
  ierr = ADDASetRefinement(adda, refine, 2);CHKERRQ(ierr);
  
  /* Random numbers */
  ierr = PetscRandomCreate(PETSC_COMM_WORLD,&rctx);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rctx);CHKERRQ(ierr);

  /* Single or multi processor ? */
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&comm_size);CHKERRQ(ierr);

  /* construct matrix */
  if( comm_size == 1 ) {
    ierr = ADDAGetMatrix(adda, MATSEQAIJ, &H);CHKERRQ(ierr);
  } else {
    ierr = ADDAGetMatrix(adda, MATMPIAIJ, &H);CHKERRQ(ierr);
  }

  /* get local corners for this processor, user is responsible for freeing lcs,lce */
  ierr = ADDAGetCorners(adda, &lcs, &lce);CHKERRQ(ierr);

  /* Allocate space for the indices that we use to construct the matrix */
  ierr = PetscMalloc(2*sizeof(PetscInt), &(sxy.x));CHKERRQ(ierr);
  ierr = PetscMalloc(2*sizeof(PetscInt), &(sxy_m.x));CHKERRQ(ierr);

  /* Assemble the matrix */
  for( x=lcs[0]; x<lce[0]; x++ ) {
    for( y=lcs[1]; y<lce[1]; y++ ) {
      /* each lattice point sets only the *forward* pointing parameters (right, down),
	 i.e. Nabla_1^+ and Nabla_2^+.
	 In this way we can use only local random number creation. That means
	 we also have to set the corresponding backward pointing entries. */
      /* Compute some normally distributed random numbers via Box-Muller */
      ierr = PetscRandomGetValueReal(rctx, &r1);CHKERRQ(ierr);
      r1 = 1.-r1; /* to change from [0,1) to (0,1], which we need for the log */
      ierr = PetscRandomGetValueReal(rctx, &r2);CHKERRQ(ierr);
      PetscReal R = sqrt(-2.*log(r1));
      PetscReal c = cos(2.*PETSC_PI*r2);
      PetscReal s = sin(2.*PETSC_PI*r2);

      /* use those to set the field */
      uxy1 = PetscExpScalar( ((PetscScalar) (R*c/beta))*PETSC_i);
      uxy2 = PetscExpScalar( ((PetscScalar) (R*s/beta))*PETSC_i);
      
      sxy.x[0] = x; sxy.x[1] = y; /* the point where we are */

      /* center action */
      sxy.d = 0; /* spin 0, 0 */
      ierr = ADDAMatSetValues(H, adda, 1, &sxy, adda, 1, &sxy, &rho, ADD_VALUES);CHKERRQ(ierr);
      sxy.d = 1; /* spin 1, 1 */
      val = -rho;
      ierr = ADDAMatSetValues(H, adda, 1, &sxy, adda, 1, &sxy, &val, ADD_VALUES);CHKERRQ(ierr);
      
      sxy_m.x[0] = x+1; sxy_m.x[1] = y; /* right action */
      sxy.d = 0; sxy_m.d = 0; /* spin 0, 0 */
      val = -uxy1; valconj = PetscConj(val);
      ierr = ADDAMatSetValues(H, adda, 1, &sxy_m, adda, 1, &sxy, &val, ADD_VALUES);CHKERRQ(ierr);
      ierr = ADDAMatSetValues(H, adda, 1, &sxy, adda, 1, &sxy_m, &valconj, ADD_VALUES);CHKERRQ(ierr);
      sxy.d = 0; sxy_m.d = 1; /* spin 0, 1 */
      val = -uxy1; valconj = PetscConj(val);
      ierr = ADDAMatSetValues(H, adda, 1, &sxy_m, adda, 1, &sxy, &val, ADD_VALUES);CHKERRQ(ierr);
      ierr = ADDAMatSetValues(H, adda, 1, &sxy, adda, 1, &sxy_m, &valconj, ADD_VALUES);CHKERRQ(ierr);
      sxy.d = 1; sxy_m.d = 0; /* spin 1, 0 */
      val = uxy1; valconj = PetscConj(val);
      ierr = ADDAMatSetValues(H, adda, 1, &sxy_m, adda, 1, &sxy, &val, ADD_VALUES);CHKERRQ(ierr);
      ierr = ADDAMatSetValues(H, adda, 1, &sxy, adda, 1, &sxy_m, &valconj, ADD_VALUES);CHKERRQ(ierr);
      sxy.d = 1; sxy_m.d = 1; /* spin 1, 1 */
      val = uxy1; valconj = PetscConj(val);
      ierr = ADDAMatSetValues(H, adda, 1, &sxy_m, adda, 1, &sxy, &val, ADD_VALUES);CHKERRQ(ierr);
      ierr = ADDAMatSetValues(H, adda, 1, &sxy, adda, 1, &sxy_m, &valconj, ADD_VALUES);CHKERRQ(ierr);

      sxy_m.x[0] = x; sxy_m.x[1] = y+1; /* down action */
      sxy.d = 0; sxy_m.d = 0; /* spin 0, 0 */
      val = -uxy2; valconj = PetscConj(val);
      ierr = ADDAMatSetValues(H, adda, 1, &sxy_m, adda, 1, &sxy, &val, ADD_VALUES);CHKERRQ(ierr);
      ierr = ADDAMatSetValues(H, adda, 1, &sxy, adda, 1, &sxy_m, &valconj, ADD_VALUES);CHKERRQ(ierr);
      sxy.d = 0; sxy_m.d = 1; /* spin 0, 1 */
      val = -PETSC_i*uxy2; valconj = PetscConj(val);
      ierr = ADDAMatSetValues(H, adda, 1, &sxy_m, adda, 1, &sxy, &val, ADD_VALUES);CHKERRQ(ierr);
      ierr = ADDAMatSetValues(H, adda, 1, &sxy, adda, 1, &sxy_m, &valconj, ADD_VALUES);CHKERRQ(ierr);
      sxy.d = 1; sxy_m.d = 0; /* spin 1, 0 */
      val = -PETSC_i*uxy2; valconj = PetscConj(val);
      ierr = ADDAMatSetValues(H, adda, 1, &sxy_m, adda, 1, &sxy, &val, ADD_VALUES);CHKERRQ(ierr);
      ierr = ADDAMatSetValues(H, adda, 1, &sxy, adda, 1, &sxy_m, &valconj, ADD_VALUES);CHKERRQ(ierr);
      sxy.d = 1; sxy_m.d = 1; /* spin 1, 1 */
      val = PetscConj(uxy2); valconj = PetscConj(val);
      ierr = ADDAMatSetValues(H, adda, 1, &sxy_m, adda, 1, &sxy, &val, ADD_VALUES);CHKERRQ(ierr);
      ierr = ADDAMatSetValues(H, adda, 1, &sxy, adda, 1, &sxy_m, &valconj, ADD_VALUES);CHKERRQ(ierr);
    }
  }
  
  ierr = PetscFree(sxy.x);CHKERRQ(ierr);
  ierr = PetscFree(sxy_m.x);CHKERRQ(ierr);

  ierr = PetscFree(lcs);CHKERRQ(ierr);
  ierr = PetscFree(lce);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(H, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(H, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* scale H */
  ierr = MatScale(H, 1./(2.*h));CHKERRQ(ierr);

  /* construct normal equations */
  ierr = MatMatMult(H, H, MAT_INITIAL_MATRIX, 1., &HtH);CHKERRQ(ierr);

  PetscScalar mineval;
  ierr = computeMinEigVal(HtH, 1000, &mineval);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "Minimum eigenvalue of H^{dag} H is %f\n", PetscAbsScalar(mineval));CHKERRQ(ierr);

  /* permutation matrix to check whether H and HtH are identical to the ones in the paper */
/*   Mat perm; */
/*   ierr = ADDAGetMatrix(adda, MATSEQAIJ, &perm);CHKERRQ(ierr); */
/*   PetscInt row, col; */
/*   PetscScalar one = 1.0; */
/*   for(PetscInt i=0; i<n; i++) { */
/*     for(PetscInt j=0; j<n; j++) { */
/*       row = (i*n+j)*2; col = i*n+j; */
/*       ierr = MatSetValues(perm, 1, &row, 1, &col, &one, INSERT_VALUES);CHKERRQ(ierr); */
/*       row = (i*n+j)*2+1; col = i*n+j + n*n; */
/*       ierr = MatSetValues(perm, 1, &row, 1, &col, &one, INSERT_VALUES);CHKERRQ(ierr); */
/*     } */
/*   } */
/*   ierr = MatAssemblyBegin(perm, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr); */
/*   ierr = MatAssemblyEnd(perm, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr); */

/*   Mat Hperm; */
/*   ierr = MatPtAP(H, perm, MAT_INITIAL_MATRIX, 1.0, &Hperm);CHKERRQ(ierr); */
/*   ierr = PetscPrintf(PETSC_COMM_WORLD, "Matrix H after construction\n");CHKERRQ(ierr); */
/*   ierr = MatView(Hperm, PETSC_VIEWER_STDOUT_(PETSC_COMM_WORLD));CHKERRQ(ierr); */

/*   Mat HtHperm; */
/*   ierr = MatPtAP(HtH, perm, MAT_INITIAL_MATRIX, 1.0, &HtHperm);CHKERRQ(ierr); */
/*   ierr = PetscPrintf(PETSC_COMM_WORLD, "Matrix HtH:\n");CHKERRQ(ierr); */
/*   ierr = MatView(HtHperm, PETSC_VIEWER_STDOUT_(PETSC_COMM_WORLD));CHKERRQ(ierr); */

  /* right hand side */
  ierr = ADDACreateGlobalVector(adda, &b);CHKERRQ(ierr);
  ierr = VecSet(b,0.0);CHKERRQ(ierr);
  PetscInt ix[1] = {0};
  PetscScalar vals[1] = {1.0};
  ierr = VecSetValues(b, 1, ix, vals, INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(b);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(b);CHKERRQ(ierr);
/*   ierr = VecSetRandom(b, rctx);CHKERRQ(ierr); */
  ierr = VecDuplicate(b, &Htb);CHKERRQ(ierr);
  ierr = MatMultTranspose(H, b, Htb);CHKERRQ(ierr);

  /* construct solver */
  ierr = KSPCreate(PETSC_COMM_WORLD,&kspmg);CHKERRQ(ierr);
  ierr = KSPSetType(kspmg, KSPCG);CHKERRQ(ierr);

  ierr = KSPGetPC(kspmg,&pcmg);CHKERRQ(ierr);
  ierr = PCSetType(pcmg,PCASA);CHKERRQ(ierr);

  /* maybe user wants to override some of the choices */
  ierr = KSPSetFromOptions(kspmg);CHKERRQ(ierr);

  ierr = KSPSetOperators(kspmg, HtH, HtH, DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);

  ierr = PCASASetDM(pcmg, (DM) adda);CHKERRQ(ierr);
  ierr = ADDADestroy(adda);CHKERRQ(ierr);

  ierr = PCASASetTolerances(pcmg, 1.e-6, 1.e-10,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);

  ierr = VecDuplicate(b, &xvec);CHKERRQ(ierr);
  ierr = VecSet(xvec, 0.0);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Solve the linear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = KSPSolve(kspmg, Htb, xvec);CHKERRQ(ierr);

/*   ierr = VecView(xvec, PETSC_VIEWER_STDOUT_(PETSC_COMM_WORLD));CHKERRQ(ierr); */

  ierr = KSPDestroy(kspmg);CHKERRQ(ierr);

  ierr = VecDestroy(xvec);CHKERRQ(ierr);
  ierr = VecDestroy(b);CHKERRQ(ierr);
  ierr = VecDestroy(Htb);CHKERRQ(ierr);
  ierr = MatDestroy(H);CHKERRQ(ierr);
  ierr = MatDestroy(HtH);CHKERRQ(ierr);

  ierr = PetscRandomDestroy(rctx);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}

/* --------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "computeMinEigVal"
PetscErrorCode computeMinEigVal(Mat A, PetscInt its, PetscScalar *eig) {
  PetscErrorCode  ierr;
  PetscRandom     rctx;     /* random number generator context */
  Vec             x0, x, x_1, tmp;
  PetscScalar     lambda_its, lambda_its_1;
  PetscReal       norm;
  Mat             G;
  PetscInt        i;
  
  PetscFunctionBegin;
  ierr = PetscRandomCreate(PETSC_COMM_WORLD,&rctx);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rctx);CHKERRQ(ierr);

  /* compute G = I-1/norm(A)*A */
  ierr = MatNorm(A, NORM_1, &norm);CHKERRQ(ierr);
  ierr = MatConvert(A, MATSAME, MAT_INITIAL_MATRIX, &G);CHKERRQ(ierr);
  ierr = MatShift(G, -norm);CHKERRQ(ierr);
  ierr = MatScale(G, -1./norm);CHKERRQ(ierr);

  ierr = MatGetVecs(G, &x_1, &x);CHKERRQ(ierr);
  ierr = VecSetRandom(x, rctx);CHKERRQ(ierr);
  ierr = VecDuplicate(x, &x0);CHKERRQ(ierr);
  ierr = VecCopy(x, x0);CHKERRQ(ierr);

  ierr = MatMult(G, x, x_1);CHKERRQ(ierr);
  for(i=0; i<its; i++) {
    tmp = x; x = x_1; x_1 = tmp;
    ierr = MatMult(G, x, x_1);CHKERRQ(ierr);
  }
  ierr = VecDot(x0, x, &lambda_its);CHKERRQ(ierr);
  ierr = VecDot(x0, x_1, &lambda_its_1);CHKERRQ(ierr);

  *eig = norm*(1.-lambda_its_1/lambda_its);

  ierr = VecDestroy(x0);CHKERRQ(ierr);
  ierr = VecDestroy(x);CHKERRQ(ierr);
  ierr = VecDestroy(x_1);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(rctx);CHKERRQ(ierr);
  ierr = MatDestroy(G);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
