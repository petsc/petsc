
static char help[] = "Lattice Gauge 2D model.\n"
"Parameters:\n"
"-size n          to use a grid size of n, i.e n space and n time steps\n"
"-beta b          controls the randomness of the gauge field\n"
"-rho r           the quark mass (?)";

#include <petscksp.h>
#include <petscpcasa.h>
#include <petscdmda.h>

PetscErrorCode computeMaxEigVal(Mat A, PetscInt its, PetscScalar *eig);

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int Argc,char **Args)
{
  PetscBool       flg;
  PetscInt        n = -6;
  PetscScalar     rho = 1.0;
  PetscReal       h;
  PetscReal       beta = 1.0;
  DM              da;
  PetscRandom     rctx;
  PetscMPIInt     comm_size;
  Mat             H,HtH;
  PetscInt        x, y, xs, ys, xm, ym;
  PetscReal       r1, r2;
  PetscScalar     uxy1, uxy2;
  MatStencil      sxy, sxy_m;
  PetscScalar     val, valconj;
  Vec             b, Htb,xvec;
  KSP             kspmg;
  PC              pcmg;
  PetscErrorCode  ierr;
  PetscInt        ix[1] = {0};
  PetscScalar     vals[1] = {1.0};

  PetscInitialize(&Argc,&Args,(char *)0,help);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-size",&n,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(PETSC_NULL,"-beta",&beta,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetScalar(PETSC_NULL,"-rho",&rho,&flg);CHKERRQ(ierr);

  /* Set the fudge parameters, we scale the whole thing by 1/(2*h) later */
  h = 1.;
  rho *= 1./(2.*h);
  
  /* Geometry info */
  ierr = DMDACreate2d(PETSC_COMM_WORLD, DMDA_BOUNDARY_PERIODIC,DMDA_BOUNDARY_PERIODIC, DMDA_STENCIL_STAR, n, n,
		    PETSC_DECIDE, PETSC_DECIDE, 2 /* this is the # of dof's */,
		    1, PETSC_NULL, PETSC_NULL, &da);CHKERRQ(ierr);
  
  /* Random numbers */
  ierr = PetscRandomCreate(PETSC_COMM_WORLD,&rctx);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rctx);CHKERRQ(ierr);

  /* Single or multi processor ? */
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&comm_size);CHKERRQ(ierr);

  /* construct matrix */
  if( comm_size == 1 ) {
    ierr = DMCreateMatrix(da, MATSEQAIJ, &H);CHKERRQ(ierr);
  } else {
    ierr = DMCreateMatrix(da, MATMPIAIJ, &H);CHKERRQ(ierr);
  }

  /* get local corners for this processor */
  ierr = DMDAGetCorners(da,&xs,&ys,0,&xm,&ym,0);CHKERRQ(ierr);

  /* Assemble the matrix */
  for( x=xs; x<xs+xm; x++ ) {
    for( y=ys; y<ys+ym; y++ ) {
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
      
      sxy.i = x; sxy.j = y; /* the point where we are */

      /* center action */
      sxy.c = 0; /* spin 0, 0 */
      ierr = MatSetValuesStencil(H, 1, &sxy, 1, &sxy, &rho, ADD_VALUES);CHKERRQ(ierr);
      sxy.c = 1; /* spin 1, 1 */
      val = -rho;
      ierr = MatSetValuesStencil(H, 1, &sxy, 1, &sxy, &val, ADD_VALUES);CHKERRQ(ierr);
      
      sxy_m.i = x+1; sxy_m.j = y; /* right action */
      sxy.c = 0; sxy_m.c = 0; /* spin 0, 0 */
      val = -uxy1; valconj = PetscConj(val);
      ierr = MatSetValuesStencil(H, 1, &sxy_m, 1, &sxy, &val, ADD_VALUES);CHKERRQ(ierr);
      ierr = MatSetValuesStencil(H, 1, &sxy, 1, &sxy_m, &valconj, ADD_VALUES);CHKERRQ(ierr);
      sxy.c = 0; sxy_m.c = 1; /* spin 0, 1 */
      val = -uxy1; valconj = PetscConj(val);
      ierr = MatSetValuesStencil(H, 1, &sxy_m, 1, &sxy, &val, ADD_VALUES);CHKERRQ(ierr);
      ierr = MatSetValuesStencil(H, 1, &sxy, 1, &sxy_m, &valconj, ADD_VALUES);CHKERRQ(ierr);
      sxy.c = 1; sxy_m.c = 0; /* spin 1, 0 */
      val = uxy1; valconj = PetscConj(val);
      ierr = MatSetValuesStencil(H, 1, &sxy_m, 1, &sxy, &val, ADD_VALUES);CHKERRQ(ierr);
      ierr = MatSetValuesStencil(H, 1, &sxy, 1, &sxy_m, &valconj, ADD_VALUES);CHKERRQ(ierr);
      sxy.c = 1; sxy_m.c = 1; /* spin 1, 1 */
      val = uxy1; valconj = PetscConj(val);
      ierr = MatSetValuesStencil(H, 1, &sxy_m, 1, &sxy, &val, ADD_VALUES);CHKERRQ(ierr);
      ierr = MatSetValuesStencil(H, 1, &sxy, 1, &sxy_m, &valconj, ADD_VALUES);CHKERRQ(ierr);

      sxy_m.i = x; sxy_m.j = y+1; /* down action */
      sxy.c = 0; sxy_m.c = 0; /* spin 0, 0 */
      val = -uxy2; valconj = PetscConj(val);
      ierr = MatSetValuesStencil(H, 1, &sxy_m, 1, &sxy, &val, ADD_VALUES);CHKERRQ(ierr);
      ierr = MatSetValuesStencil(H, 1, &sxy, 1, &sxy_m, &valconj, ADD_VALUES);CHKERRQ(ierr);
      sxy.c = 0; sxy_m.c = 1; /* spin 0, 1 */
      val = -PETSC_i*uxy2; valconj = PetscConj(val);
      ierr = MatSetValuesStencil(H, 1, &sxy_m, 1, &sxy, &val, ADD_VALUES);CHKERRQ(ierr);
      ierr = MatSetValuesStencil(H, 1, &sxy, 1, &sxy_m, &valconj, ADD_VALUES);CHKERRQ(ierr);
      sxy.c = 1; sxy_m.c = 0; /* spin 1, 0 */
      val = -PETSC_i*uxy2; valconj = PetscConj(val);
      ierr = MatSetValuesStencil(H, 1, &sxy_m, 1, &sxy, &val, ADD_VALUES);CHKERRQ(ierr);
      ierr = MatSetValuesStencil(H, 1, &sxy, 1, &sxy_m, &valconj, ADD_VALUES);CHKERRQ(ierr);
      sxy.c = 1; sxy_m.c = 1; /* spin 1, 1 */
      val = PetscConj(uxy2); valconj = PetscConj(val);
      ierr = MatSetValuesStencil(H, 1, &sxy_m, 1, &sxy, &val, ADD_VALUES);CHKERRQ(ierr);
      ierr = MatSetValuesStencil(H, 1, &sxy, 1, &sxy_m, &valconj, ADD_VALUES);CHKERRQ(ierr);
    }
  }
  
  ierr = MatAssemblyBegin(H, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(H, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* scale H */
  ierr = MatScale(H, 1./(2.*h));CHKERRQ(ierr);

  /* it looks like H is Hermetian */
  /* construct normal equations */
  ierr = MatMatMult(H, H, MAT_INITIAL_MATRIX, 1., &HtH);CHKERRQ(ierr);

  /* permutation matrix to check whether H and HtH are identical to the ones in the paper */
/*   Mat perm; */
/*   ierr = DMCreateMatrix(da, MATSEQAIJ, &perm);CHKERRQ(ierr); */
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
  ierr = DMCreateGlobalVector(da, &b);CHKERRQ(ierr);
  ierr = VecSet(b,0.0);CHKERRQ(ierr);
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

  ierr = DMDASetRefinementFactor(da, 3, 3, 3);CHKERRQ(ierr);
  ierr = PCASASetDM(pcmg, (DM) da);CHKERRQ(ierr);

  ierr = PCASASetTolerances(pcmg, 1.e-6, 1.e-10,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);

  ierr = VecDuplicate(b, &xvec);CHKERRQ(ierr);
  ierr = VecSet(xvec, 0.0);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Solve the linear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = KSPSolve(kspmg, Htb, xvec);CHKERRQ(ierr);

/*   ierr = VecView(xvec, PETSC_VIEWER_STDOUT_(PETSC_COMM_WORLD));CHKERRQ(ierr); */

  ierr = KSPDestroy(&kspmg);CHKERRQ(ierr);
  ierr = VecDestroy(&xvec);CHKERRQ(ierr);

  /*   seems to be destroyed by KSPDestroy */
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = VecDestroy(&Htb);CHKERRQ(ierr);
  ierr = MatDestroy(&HtH);CHKERRQ(ierr);
  ierr = MatDestroy(&H);CHKERRQ(ierr);

  ierr = DMDestroy(&da);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&rctx);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}

/* --------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "computeMaxEigVal"
PetscErrorCode computeMaxEigVal(Mat A, PetscInt its, PetscScalar *eig) {
  PetscErrorCode  ierr;
  PetscRandom     rctx;     /* random number generator context */
  Vec             x0, x, x_1, tmp;
  PetscScalar     lambda_its, lambda_its_1;
  PetscInt        i;
  
  PetscFunctionBegin;
  ierr = PetscRandomCreate(PETSC_COMM_WORLD,&rctx);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rctx);CHKERRQ(ierr);
  ierr = MatGetVecs(A, &x_1, &x);CHKERRQ(ierr);
  ierr = VecSetRandom(x, rctx);CHKERRQ(ierr);
  ierr = VecDuplicate(x, &x0);CHKERRQ(ierr);
  ierr = VecCopy(x, x0);CHKERRQ(ierr);

  ierr = MatMult(A, x, x_1);CHKERRQ(ierr);
  for(i=0; i<its; i++) {
    tmp = x; x = x_1; x_1 = tmp;
    ierr = MatMult(A, x, x_1);CHKERRQ(ierr);
  }
  ierr = VecDot(x0, x, &lambda_its);CHKERRQ(ierr);
  ierr = VecDot(x0, x_1, &lambda_its_1);CHKERRQ(ierr);

  *eig = lambda_its_1/lambda_its;

  ierr = VecDestroy(&x0);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&x_1);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
