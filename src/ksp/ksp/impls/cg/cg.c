
/*
    This file implements the conjugate gradient method in PETSc as part of
    KSP. You can use this as a starting point for implementing your own
    Krylov method that is not provided with PETSc.

    The following basic routines are required for each Krylov method.
        KSPCreate_XXX()          - Creates the Krylov context
        KSPSetFromOptions_XXX()  - Sets runtime options
        KSPSolve_XXX()           - Runs the Krylov method
        KSPDestroy_XXX()         - Destroys the Krylov context, freeing all
                                   memory it needed
    Here the "_XXX" denotes a particular implementation, in this case
    we use _CG (e.g. KSPCreate_CG, KSPDestroy_CG). These routines are
    are actually called vai the common user interface routines
    KSPSetType(), KSPSetFromOptions(), KSPSolve(), and KSPDestroy() so the
    application code interface remains identical for all preconditioners.

    Other basic routines for the KSP objects include
        KSPSetUp_XXX()
        KSPView_XXX()             - Prints details of solver being used.

    Detailed notes:
    By default, this code implements the CG (Conjugate Gradient) method,
    which is valid for real symmetric (and complex Hermitian) positive
    definite matrices. Note that for the complex Hermitian case, the
    VecDot() arguments within the code MUST remain in the order given
    for correct computation of inner products.

    Reference: Hestenes and Steifel, 1952.

    By switching to the indefinite vector inner product, VecTDot(), the
    same code is used for the complex symmetric case as well.  The user
    must call KSPCGSetType(ksp,KSP_CG_SYMMETRIC) or use the option
    -ksp_cg_type symmetric to invoke this variant for the complex case.
    Note, however, that the complex symmetric code is NOT valid for
    all such matrices ... and thus we don't recommend using this method.
*/
/*
       cgimpl.h defines the simple data structured used to store information
    related to the type of matrix (e.g. complex symmetric) being solved and
    data used during the optional Lanczo process used to compute eigenvalues
*/
#include <../src/ksp/ksp/impls/cg/cgimpl.h>       /*I "petscksp.h" I*/
extern PetscErrorCode KSPComputeExtremeSingularValues_CG(KSP,PetscReal *,PetscReal *);
extern PetscErrorCode KSPComputeEigenvalues_CG(KSP,PetscInt,PetscReal *,PetscReal *,PetscInt *);

/*
     KSPSetUp_CG - Sets up the workspace needed by the CG method.

      This is called once, usually automatically by KSPSolve() or KSPSetUp()
     but can be called directly by KSPSetUp()
*/
#undef __FUNCT__
#define __FUNCT__ "KSPSetUp_CG"
PetscErrorCode KSPSetUp_CG(KSP ksp)
{
  KSP_CG         *cgP = (KSP_CG*)ksp->data;
  PetscErrorCode ierr;
  PetscInt        maxit = ksp->max_it,nwork = 3;

  PetscFunctionBegin;
  /* get work vectors needed by CG */
  if (cgP->singlereduction) nwork += 2;
  ierr = KSPDefaultGetWork(ksp,nwork);CHKERRQ(ierr);

  /*
     If user requested computations of eigenvalues then allocate work
     work space needed
  */
  if (ksp->calc_sings) {
    /* get space to store tridiagonal matrix for Lanczos */
    ierr = PetscMalloc4(maxit+1,PetscScalar,&cgP->e,maxit+1,PetscScalar,&cgP->d,maxit+1,PetscReal,&cgP->ee,maxit+1,PetscReal,&cgP->dd);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory(ksp,2*(maxit+1)*(sizeof(PetscScalar)+sizeof(PetscReal)));CHKERRQ(ierr);
    ksp->ops->computeextremesingularvalues = KSPComputeExtremeSingularValues_CG;
    ksp->ops->computeeigenvalues           = KSPComputeEigenvalues_CG;
  }
  PetscFunctionReturn(0);
}

/*
       KSPSolve_CG - This routine actually applies the conjugate gradient  method

   This routine is MUCH too messy. I has too many options (norm type and single reduction) embedded making the code confusing and likely to be buggy.

   Input Parameter:
.     ksp - the Krylov space object that was set to use conjugate gradient, by, for
            example, KSPCreate(MPI_Comm,KSP *ksp); KSPSetType(ksp,KSPCG);
*/
#undef __FUNCT__
#define __FUNCT__ "KSPSolve_CG"
PetscErrorCode  KSPSolve_CG(KSP ksp)
{
  PetscErrorCode ierr;
  PetscInt       i,stored_max_it,eigs;
  PetscScalar    dpi = 0.0,a = 1.0,beta,betaold = 1.0,b = 0,*e = 0,*d = 0,delta,dpiold;
  PetscReal      dp = 0.0;
  Vec            X,B,Z,R,P,S,W;
  KSP_CG         *cg;
  Mat            Amat,Pmat;
  MatStructure   pflag;
  PetscBool      diagonalscale;

  PetscFunctionBegin;
  ierr    = PCGetDiagonalScale(ksp->pc,&diagonalscale);CHKERRQ(ierr);
  if (diagonalscale) SETERRQ1(((PetscObject)ksp)->comm,PETSC_ERR_SUP,"Krylov method %s does not support diagonal scaling",((PetscObject)ksp)->type_name);

  cg            = (KSP_CG*)ksp->data;
  eigs          = ksp->calc_sings;
  stored_max_it = ksp->max_it;
  X             = ksp->vec_sol;
  B             = ksp->vec_rhs;
  R             = ksp->work[0];
  Z             = ksp->work[1];
  P             = ksp->work[2];
  if (cg->singlereduction) {
    S           = ksp->work[3];
    W           = ksp->work[4];
  } else {
    S           = 0;            /* unused */
    W           = Z;
  }

#define VecXDot(x,y,a) (((cg->type) == (KSP_CG_HERMITIAN)) ? VecDot(x,y,a) : VecTDot(x,y,a))

  if (eigs) {e = cg->e; d = cg->d; e[0] = 0.0; }
  ierr = PCGetOperators(ksp->pc,&Amat,&Pmat,&pflag);CHKERRQ(ierr);

  ksp->its = 0;
  if (!ksp->guess_zero) {
    ierr = KSP_MatMult(ksp,Amat,X,R);CHKERRQ(ierr);            /*     r <- b - Ax     */
    ierr = VecAYPX(R,-1.0,B);CHKERRQ(ierr);
  } else {
    ierr = VecCopy(B,R);CHKERRQ(ierr);                         /*     r <- b (x is 0) */
  }

  switch (ksp->normtype) {
  case KSP_NORM_PRECONDITIONED:
    ierr = KSP_PCApply(ksp,R,Z);CHKERRQ(ierr);                   /*     z <- Br         */
    ierr = VecNorm(Z,NORM_2,&dp);CHKERRQ(ierr);                /*    dp <- z'*z = e'*A'*B'*B*A'*e'     */
    break;
  case KSP_NORM_UNPRECONDITIONED:
    ierr = VecNorm(R,NORM_2,&dp);CHKERRQ(ierr);                /*    dp <- r'*r = e'*A'*A*e            */
    break;
  case KSP_NORM_NATURAL:
    ierr = KSP_PCApply(ksp,R,Z);CHKERRQ(ierr);                   /*     z <- Br         */
    if (cg->singlereduction) {
      ierr = KSP_MatMult(ksp,Amat,Z,S);CHKERRQ(ierr);
      ierr = VecXDot(Z,S,&delta);CHKERRQ(ierr);
    }
    ierr = VecXDot(Z,R,&beta);CHKERRQ(ierr);                     /*  beta <- z'*r       */
    if (PetscIsInfOrNanScalar(beta)) SETERRQ(((PetscObject)ksp)->comm,PETSC_ERR_FP,"Infinite or not-a-number generated in dot product");
    dp = PetscSqrtReal(PetscAbsScalar(beta));                           /*    dp <- r'*z = r'*B*r = e'*A'*B*A*e */
    break;
  case KSP_NORM_NONE:
    dp = 0.0;
    break;
  default: SETERRQ1(((PetscObject)ksp)->comm,PETSC_ERR_SUP,"%s",KSPNormTypes[ksp->normtype]);
  }
  KSPLogResidualHistory(ksp,dp);
  ierr = KSPMonitor(ksp,0,dp);CHKERRQ(ierr);
  ksp->rnorm = dp;

  ierr = (*ksp->converged)(ksp,0,dp,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);      /* test for convergence */
  if (ksp->reason) PetscFunctionReturn(0);

  if (ksp->normtype != KSP_NORM_PRECONDITIONED && (ksp->normtype != KSP_NORM_NATURAL)){
    ierr = KSP_PCApply(ksp,R,Z);CHKERRQ(ierr);                   /*     z <- Br         */
  }
  if (ksp->normtype != KSP_NORM_NATURAL){
    if (cg->singlereduction) {
      ierr = KSP_MatMult(ksp,Amat,Z,S);CHKERRQ(ierr);
      ierr = VecXDot(Z,S,&delta);CHKERRQ(ierr);
    }
    ierr = VecXDot(Z,R,&beta);CHKERRQ(ierr);         /*  beta <- z'*r       */
    if (PetscIsInfOrNanScalar(beta)) SETERRQ(((PetscObject)ksp)->comm,PETSC_ERR_FP,"Infinite or not-a-number generated in dot product");
  }

  i = 0;
  do {
     ksp->its = i+1;
     if (beta == 0.0) {
       ksp->reason = KSP_CONVERGED_ATOL;
       ierr = PetscInfo(ksp,"converged due to beta = 0\n");CHKERRQ(ierr);
       break;
#if !defined(PETSC_USE_COMPLEX)
     } else if ((i > 0) && (beta*betaold < 0.0)) {
       ksp->reason = KSP_DIVERGED_INDEFINITE_PC;
       ierr = PetscInfo(ksp,"diverging due to indefinite preconditioner\n");CHKERRQ(ierr);
       break;
#endif
     }
     if (!i) {
       ierr = VecCopy(Z,P);CHKERRQ(ierr);         /*     p <- z          */
       b = 0.0;
     } else {
       b = beta/betaold;
       if (eigs) {
         if (ksp->max_it != stored_max_it) SETERRQ(((PetscObject)ksp)->comm,PETSC_ERR_SUP,"Can not change maxit AND calculate eigenvalues");
         e[i] = PetscSqrtReal(PetscAbsScalar(b))/a;
       }
       ierr = VecAYPX(P,b,Z);CHKERRQ(ierr);    /*     p <- z + b* p   */
     }
     dpiold = dpi;
     if (!cg->singlereduction || !i) {
       ierr = KSP_MatMult(ksp,Amat,P,W);CHKERRQ(ierr);          /*     w <- Ap         */
       ierr = VecXDot(P,W,&dpi);CHKERRQ(ierr);                  /*     dpi <- p'w     */
     } else {
	ierr = VecAYPX(W,beta/betaold,S);CHKERRQ(ierr);                  /*     w <- Ap         */
        dpi = delta - beta*beta*dpiold/(betaold*betaold);              /*     dpi <- p'w     */
     }
     betaold = beta;
     if (PetscIsInfOrNanScalar(dpi)) SETERRQ(((PetscObject)ksp)->comm,PETSC_ERR_FP,"Infinite or not-a-number generated in dot product");

     if ((dpi == 0.0) || ((i > 0) && (PetscRealPart(dpi*dpiold) <= 0.0))) {
       ksp->reason = KSP_DIVERGED_INDEFINITE_MAT;
       ierr = PetscInfo(ksp,"diverging due to indefinite or negative definite matrix\n");CHKERRQ(ierr);
       break;
     }
     a = beta/dpi;                                 /*     a = beta/p'w   */
     if (eigs) {
       d[i] = PetscSqrtReal(PetscAbsScalar(b))*e[i] + 1.0/a;
     }
     ierr = VecAXPY(X,a,P);CHKERRQ(ierr);          /*     x <- x + ap     */
     ierr = VecAXPY(R,-a,W);CHKERRQ(ierr);                      /*     r <- r - aw    */
     if (ksp->normtype == KSP_NORM_PRECONDITIONED && ksp->chknorm < i+2) {
       ierr = KSP_PCApply(ksp,R,Z);CHKERRQ(ierr);               /*     z <- Br         */
       if (cg->singlereduction) {
         ierr = KSP_MatMult(ksp,Amat,Z,S);CHKERRQ(ierr);
       }
       ierr = VecNorm(Z,NORM_2,&dp);CHKERRQ(ierr);              /*    dp <- z'*z       */
     } else if (ksp->normtype == KSP_NORM_UNPRECONDITIONED && ksp->chknorm < i+2) {
       ierr = VecNorm(R,NORM_2,&dp);CHKERRQ(ierr);              /*    dp <- r'*r       */
     } else if (ksp->normtype == KSP_NORM_NATURAL) {
       ierr = KSP_PCApply(ksp,R,Z);CHKERRQ(ierr);               /*     z <- Br         */
       if (cg->singlereduction) {
         PetscScalar tmp[2];
         Vec         vecs[2];
         vecs[0] = S; vecs[1] = R;
         ierr = KSP_MatMult(ksp,Amat,Z,S);CHKERRQ(ierr);
         /*ierr = VecXDot(Z,S,&delta);CHKERRQ(ierr);
	   ierr = VecXDot(Z,R,&beta);CHKERRQ(ierr); */    /*  beta <- r'*z       */
         ierr = VecMDot(Z,2,vecs,tmp);CHKERRQ(ierr);
         delta = tmp[0]; beta = tmp[1];
       } else {
         ierr = VecXDot(Z,R,&beta);CHKERRQ(ierr);     /*  beta <- r'*z       */
       }
       if (PetscIsInfOrNanScalar(beta)) SETERRQ(((PetscObject)ksp)->comm,PETSC_ERR_FP,"Infinite or not-a-number generated in dot product");
       dp = PetscSqrtReal(PetscAbsScalar(beta));
     } else {
       dp = 0.0;
     }
     ksp->rnorm = dp;
     KSPLogResidualHistory(ksp,dp);
     ierr = KSPMonitor(ksp,i+1,dp);CHKERRQ(ierr);
     ierr = (*ksp->converged)(ksp,i+1,dp,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);
     if (ksp->reason) break;

     if ((ksp->normtype != KSP_NORM_PRECONDITIONED && (ksp->normtype != KSP_NORM_NATURAL)) || (ksp->chknorm >= i+2)){
       ierr = KSP_PCApply(ksp,R,Z);CHKERRQ(ierr);                   /*     z <- Br         */
       if (cg->singlereduction) {
         ierr = KSP_MatMult(ksp,Amat,Z,S);CHKERRQ(ierr);
       }
     }
     if ((ksp->normtype != KSP_NORM_NATURAL) || (ksp->chknorm >= i+2)){
       if (cg->singlereduction) {
         PetscScalar tmp[2];
         Vec         vecs[2];
         vecs[0] = S; vecs[1] = R;
	 /* ierr = VecXDot(Z,R,&beta);CHKERRQ(ierr);   */     /*  beta <- z'*r       */
         /* ierr = VecXDot(Z,S,&delta);CHKERRQ(ierr);*/
         ierr = VecMDot(Z,2,vecs,tmp);CHKERRQ(ierr);
         delta = tmp[0]; beta = tmp[1];
       } else {
	 ierr = VecXDot(Z,R,&beta);CHKERRQ(ierr);        /*  beta <- z'*r       */
       }
       if (PetscIsInfOrNanScalar(beta)) SETERRQ(((PetscObject)ksp)->comm,PETSC_ERR_FP,"Infinite or not-a-number generated in dot product");
     }

     i++;
  } while (i<ksp->max_it);
  if (i >= ksp->max_it) {
    ksp->reason = KSP_DIVERGED_ITS;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPDestroy_CG"
PetscErrorCode KSPDestroy_CG(KSP ksp)
{
  KSP_CG         *cg = (KSP_CG*)ksp->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* free space used for singular value calculations */
  if (ksp->calc_sings) {
    ierr = PetscFree4(cg->e,cg->d,cg->ee,cg->dd);CHKERRQ(ierr);
  }
  ierr = KSPDefaultDestroy(ksp);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ksp,"KSPCGSetType_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ksp,"KSPCGUseSingleReduction_C","",PETSC_NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
     KSPView_CG - Prints information about the current Krylov method being used

      Currently this only prints information to a file (or stdout) about the
      symmetry of the problem. If your Krylov method has special options or
      flags that information should be printed here.

*/
#undef __FUNCT__
#define __FUNCT__ "KSPView_CG"
PetscErrorCode KSPView_CG(KSP ksp,PetscViewer viewer)
{
#if defined(PETSC_USE_COMPLEX)
  KSP_CG         *cg = (KSP_CG *)ksp->data;
  PetscErrorCode ierr;
  PetscBool      iascii;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"  CG or CGNE: variant %s\n",KSPCGTypes[cg->type]);CHKERRQ(ierr);
  } else {
    SETERRQ1(((PetscObject)ksp)->comm,PETSC_ERR_SUP,"Viewer type %s not supported for KSP cg",((PetscObject)viewer)->type_name);
  }
#endif
  PetscFunctionReturn(0);
}

/*
    KSPSetFromOptions_CG - Checks the options database for options related to the
                           conjugate gradient method.
*/
#undef __FUNCT__
#define __FUNCT__ "KSPSetFromOptions_CG"
PetscErrorCode KSPSetFromOptions_CG(KSP ksp)
{
  PetscErrorCode ierr;
  KSP_CG         *cg = (KSP_CG *)ksp->data;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("KSP CG and CGNE options");CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
  ierr = PetscOptionsEnum("-ksp_cg_type","Matrix is Hermitian or complex symmetric","KSPCGSetType",KSPCGTypes,(PetscEnum)cg->type,
                          (PetscEnum*)&cg->type,PETSC_NULL);CHKERRQ(ierr);
#endif
  ierr = PetscOptionsBool("-ksp_cg_single_reduction","Merge inner products into single MPI_Allreduce()",
                           "KSPCGUseSingleReduction",cg->singlereduction,&cg->singlereduction,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
    KSPCGSetType_CG - This is an option that is SPECIFIC to this particular Krylov method.
                      This routine is registered below in KSPCreate_CG() and called from the
                      routine KSPCGSetType() (see the file cgtype.c).

        This must be wrapped in an EXTERN_C_BEGIN to be dynamically linkable in C++
*/
EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "KSPCGSetType_CG"
PetscErrorCode  KSPCGSetType_CG(KSP ksp,KSPCGType type)
{
  KSP_CG *cg = (KSP_CG *)ksp->data;

  PetscFunctionBegin;
  cg->type = type;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "KSPCGUseSingleReduction_CG"
PetscErrorCode  KSPCGUseSingleReduction_CG(KSP ksp,PetscBool  flg)
{
  KSP_CG *cg  = (KSP_CG *)ksp->data;

  PetscFunctionBegin;
  cg->singlereduction = flg;
  PetscFunctionReturn(0);
}
EXTERN_C_END

/*
    KSPCreate_CG - Creates the data structure for the Krylov method CG and sets the
       function pointers for all the routines it needs to call (KSPSolve_CG() etc)

    It must be wrapped in EXTERN_C_BEGIN to be dynamically linkable in C++
*/
/*MC
     KSPCG - The preconditioned conjugate gradient (PCG) iterative method

   Options Database Keys:
+   -ksp_cg_type Hermitian - (for complex matrices only) indicates the matrix is Hermitian, see KSPCGSetType()
.   -ksp_cg_type symmetric - (for complex matrices only) indicates the matrix is symmetric
-   -ksp_cg_single_reduction - performs both inner products needed in the algorithm with a single MPI_Allreduce() call, see KSPCGUseSingleReduction()

   Level: beginner

   Notes: The PCG method requires both the matrix and preconditioner to be symmetric positive (or negative) (semi) definite
          Only left preconditioning is supported.

   For complex numbers there are two different CG methods. One for Hermitian symmetric matrices and one for non-Hermitian symmetric matrices. Use
   KSPCGSetType() to indicate which type you are using.

   Developer Notes: KSPSolve_CG() should actually query the matrix to determine if it is Hermitian symmetric or not and NOT require the user to
   indicate it to the KSP object.

   References:
   Methods of Conjugate Gradients for Solving Linear Systems, Magnus R. Hestenes and Eduard Stiefel,
   Journal of Research of the National Bureau of Standards Vol. 49, No. 6, December 1952 Research Paper 2379
   pp. 409--436.

.seealso:  KSPCreate(), KSPSetType(), KSPType (for list of available types), KSP,
           KSPCGSetType(), KSPCGUseSingleReduction()

M*/
EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "KSPCreate_CG"
PetscErrorCode  KSPCreate_CG(KSP ksp)
{
  PetscErrorCode ierr;
  KSP_CG         *cg;

  PetscFunctionBegin;
  ierr = PetscNewLog(ksp,KSP_CG,&cg);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
  cg->type                       = KSP_CG_SYMMETRIC;
#else
  cg->type                       = KSP_CG_HERMITIAN;
#endif
  ksp->data                      = (void*)cg;

  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_PRECONDITIONED,PC_LEFT,2);CHKERRQ(ierr);
  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_UNPRECONDITIONED,PC_LEFT,1);CHKERRQ(ierr);
  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_NATURAL,PC_LEFT,1);CHKERRQ(ierr);
  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_NONE,PC_LEFT,1);CHKERRQ(ierr);

  /*
       Sets the functions that are associated with this data structure
       (in C++ this is the same as defining virtual functions)
  */
  ksp->ops->setup                = KSPSetUp_CG;
  ksp->ops->solve                = KSPSolve_CG;
  ksp->ops->destroy              = KSPDestroy_CG;
  ksp->ops->view                 = KSPView_CG;
  ksp->ops->setfromoptions       = KSPSetFromOptions_CG;
  ksp->ops->buildsolution        = KSPDefaultBuildSolution;
  ksp->ops->buildresidual        = KSPDefaultBuildResidual;

  /*
      Attach the function KSPCGSetType_CG() to this object. The routine
      KSPCGSetType() checks for this attached function and calls it if it finds
      it. (Sort of like a dynamic member function that can be added at run time
  */
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ksp,"KSPCGSetType_C","KSPCGSetType_CG", KSPCGSetType_CG);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ksp,"KSPCGUseSingleReduction_C","KSPCGUseSingleReduction_CG", KSPCGUseSingleReduction_CG);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END




