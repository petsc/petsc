/*$Id: itcreate.c,v 1.185 2000/04/30 22:07:34 bsmith Exp balay $*/
/*
     The basic KSP routines, Create, View etc. are here.
*/
#include "src/sles/ksp/kspimpl.h"      /*I "petscksp.h" I*/
#include "petscsys.h"

PetscTruth KSPRegisterAllCalled = PETSC_FALSE;

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"KSPView"
/*@C 
   KSPView - Prints the KSP data structure.

   Collective on KSP

   Input Parameters:
+  ksp - the Krylov space context
-  viewer - visualization context

   Note:
   The available visualization contexts include
+     VIEWER_STDOUT_SELF - standard output (default)
-     VIEWER_STDOUT_WORLD - synchronized standard
         output where only the first processor opens
         the file.  All other processors send their 
         data to the first processor to print. 

   The user can open an alternative visualization context with
   ViewerASCIIOpen() - output to a specified file.

   Level: developer

.keywords: KSP, view

.seealso: PCView(), ViewerASCIIOpen()
@*/
int KSPView(KSP ksp,Viewer viewer)
{
  char        *type;
  int         ierr;
  PetscTruth  isascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  if (!viewer) viewer = VIEWER_STDOUT_(ksp->comm);
  PetscValidHeaderSpecific(viewer,VIEWER_COOKIE);
  PetscCheckSameComm(ksp,viewer);

  ierr = PetscTypeCompare((PetscObject)viewer,ASCII_VIEWER,&isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = KSPGetType(ksp,&type);CHKERRQ(ierr);
    ierr = ViewerASCIIPrintf(viewer,"KSP Object:\n");CHKERRQ(ierr);
    if (type) {
      ierr = ViewerASCIIPrintf(viewer,"  type: %s\n",type);CHKERRQ(ierr);
    } else {
      ierr = ViewerASCIIPrintf(viewer,"  type: not yet set\n");CHKERRQ(ierr);
    }
    if (ksp->ops->view) {
      ierr = ViewerASCIIPushTab(viewer);CHKERRQ(ierr);
      ierr = (*ksp->ops->view)(ksp,viewer);CHKERRQ(ierr);
      ierr = ViewerASCIIPopTab(viewer);CHKERRQ(ierr);
    }
    if (ksp->guess_zero) {ierr = ViewerASCIIPrintf(viewer,"  maximum iterations=%d, initial guess is zero\n",ksp->max_it);CHKERRQ(ierr);}
    else                 {ierr = ViewerASCIIPrintf(viewer,"  maximum iterations=%d\n", ksp->max_it);CHKERRQ(ierr);}
    ierr = ViewerASCIIPrintf(viewer,"  tolerances:  relative=%g, absolute=%g, divergence=%g\n",ksp->rtol,ksp->atol,ksp->divtol);CHKERRQ(ierr);
    if (ksp->pc_side == PC_RIGHT)          {ierr = ViewerASCIIPrintf(viewer,"  right preconditioning\n");CHKERRQ(ierr);}
    else if (ksp->pc_side == PC_SYMMETRIC) {ierr = ViewerASCIIPrintf(viewer,"  symmetric preconditioning\n");CHKERRQ(ierr);}
    else                                   {ierr = ViewerASCIIPrintf(viewer,"  left preconditioning\n");CHKERRQ(ierr);}
  } else {
    if (ksp->ops->view) {
      ierr = (*ksp->ops->view)(ksp,viewer);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

/*
   Contains the list of registered KSP routines
*/
FList KSPList = 0;

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"KSPSetAvoidNorms"
/*@C
   KSPSetAvoidNorms - Sets the KSP solver to avoid computing the residual norm
   when possible.  This, for example, reduces the number of collective operations
   when using the Krylov method as a smoother.

   Collective on KSP

   Input Parameter:
.  ksp - Krylov solver context

   Notes: 
   One cannot use the default convergence test routines when this option is 
   set, since these are based on decreases in the residual norms.  Thus, this
   option automatically switches to activate the KSPSkipConverged() test function.

   Currently only works with the CG, Richardson, Bi-CG-stab, CR, and CGS methods.

   Level: advanced

.keywords: KSP, create, context, norms

.seealso: KSPSetUp(), KSPSolve(), KSPDestroy(), KSPSkipConverged()
@*/
int KSPSetAvoidNorms(KSP ksp)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  ksp->avoidnorms = PETSC_TRUE;
  ierr = KSPSetConvergenceTest(ksp,KSPSkipConverged,PETSC_NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"KSPPublish_Petsc"
static int KSPPublish_Petsc(PetscObject obj)
{
#if defined(PETSC_HAVE_AMS)
  KSP          v = (KSP) obj;
  int          ierr;
#endif

  PetscFunctionBegin;

#if defined(PETSC_HAVE_AMS)
  /* if it is already published then return */
  if (v->amem >=0) PetscFunctionReturn(0);

  ierr = PetscObjectPublishBaseBegin(obj);CHKERRQ(ierr);
  ierr = AMS_Memory_add_field((AMS_Memory)v->amem,"Iteration",&v->its,1,AMS_INT,AMS_READ,
                                AMS_COMMON,AMS_REDUCT_UNDEF);CHKERRQ(ierr);
  ierr = AMS_Memory_add_field((AMS_Memory)v->amem,"Residual",&v->rnorm,1,AMS_DOUBLE,AMS_READ,
                                AMS_COMMON,AMS_REDUCT_UNDEF);CHKERRQ(ierr);
  ierr = PetscObjectPublishBaseEnd(obj);CHKERRQ(ierr);
#endif

  PetscFunctionReturn(0);
}


#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"KSPCreate"
/*@C
   KSPCreate - Creates the default KSP context.

   Collective on MPI_Comm

   Input Parameter:
.  comm - MPI communicator

   Output Parameter:
.  ksp - location to put the KSP context

   Notes:
   The default KSP type is GMRES with a restart of 30, using modified Gram-Schmidt
   orthogonalization.

   Level: developer

.keywords: KSP, create, context

.seealso: KSPSetUp(), KSPSolve(), KSPDestroy()
@*/
int KSPCreate(MPI_Comm comm,KSP *inksp)
{
  KSP ksp;

  PetscFunctionBegin;
  *inksp = 0;
  PetscHeaderCreate(ksp,_p_KSP,struct _KSPOps,KSP_COOKIE,-1,"KSP",comm,KSPDestroy,KSPView);
  PLogObjectCreate(ksp);
  *inksp             = ksp;
  ksp->bops->publish = KSPPublish_Petsc;

  ksp->type          = -1;
  ksp->max_it        = 10000;
  ksp->pc_side       = PC_LEFT;
  ksp->use_pres      = PETSC_FALSE;
  ksp->rtol          = 1.e-5;
  ksp->atol          = 1.e-50;
  ksp->divtol        = 1.e4;
  ksp->avoidnorms    = PETSC_FALSE;

  ksp->rnorm               = 0.0;
  ksp->its                 = 0;
  ksp->guess_zero          = PETSC_TRUE;
  ksp->calc_sings          = PETSC_FALSE;
  ksp->calc_res            = PETSC_FALSE;
  ksp->res_hist            = PETSC_NULL;
  ksp->res_hist_len        = 0;
  ksp->res_hist_max        = 0;
  ksp->res_hist_reset      = PETSC_TRUE;
  ksp->numbermonitors      = 0;
  ksp->converged           = KSPDefaultConverged;
  ksp->ops->buildsolution  = KSPDefaultBuildSolution;
  ksp->ops->buildresidual  = KSPDefaultBuildResidual;

  ksp->ops->setfromoptions = 0;
  ksp->ops->printhelp      = 0;

  ksp->vec_sol         = 0;
  ksp->vec_rhs         = 0;
  ksp->B               = 0;

  ksp->ops->solve      = 0;
  ksp->ops->setup      = 0;
  ksp->ops->destroy    = 0;

  ksp->data            = 0;
  ksp->nwork           = 0;
  ksp->work            = 0;

  ksp->cnvP            = 0;

  ksp->reason          = KSP_CONVERGED_ITERATING;

  ksp->setupcalled     = 0;
  PetscPublishAll(ksp);
  PetscFunctionReturn(0);
}
 
#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"KSPSetType"
/*@C
   KSPSetType - Builds KSP for a particular solver. 

   Collective on KSP

   Input Parameters:
+  ksp      - the Krylov space context
-  type - a known method

   Options Database Key:
.  -ksp_type  <method> - Sets the method; use -help for a list 
    of available methods (for instance, cg or gmres)

   Notes:  
   See "petsc/include/petscksp.h" for available methods (for instance,
   KSPCG or KSPGMRES).

  Normally, it is best to use the SLESSetFromOptions() command and
  then set the KSP type from the options database rather than by using
  this routine.  Using the options database provides the user with
  maximum flexibility in evaluating the many different Krylov methods.
  The KSPSetType() routine is provided for those situations where it
  is necessary to set the iterative solver independently of the command
  line or options database.  This might be the case, for example, when
  the choice of iterative solver changes during the execution of the
  program, and the user's application is taking responsibility for
  choosing the appropriate method.  In other words, this routine is
  not for beginners.

  Level: intermediate

.keywords: KSP, set, method

.seealso: PCSetType()
@*/
int KSPSetType(KSP ksp,KSPType type)
{
  int        ierr,(*r)(KSP);
  PetscTruth match;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  PetscValidCharPointer(type);

  ierr = PetscTypeCompare((PetscObject)ksp,type,&match);CHKERRQ(ierr);
  if (match) PetscFunctionReturn(0);

  if (ksp->data) {
    /* destroy the old private KSP context */
    ierr = (*ksp->ops->destroy)(ksp);CHKERRQ(ierr);
    ksp->data = 0;
  }
  /* Get the function pointers for the iterative method requested */
  if (!KSPRegisterAllCalled) {ierr = KSPRegisterAll(PETSC_NULL);CHKERRQ(ierr);}

  ierr =  FListFind(ksp->comm,KSPList,type,(int (**)(void *)) &r);CHKERRQ(ierr);

  if (!r) SETERRQ1(1,1,"Unknown KSP type given: %s",type);

  ksp->setupcalled = 0;
  ierr = (*r)(ksp);CHKERRQ(ierr);

  ierr = PetscObjectChangeTypeName((PetscObject)ksp,type);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"KSPRegisterDestroy"
/*@C
   KSPRegisterDestroy - Frees the list of KSP methods that were
   registered by KSPRegisterDynamic().

   Not Collective

   Level: advanced

.keywords: KSP, register, destroy

.seealso: KSPRegisterDynamic(), KSPRegisterAll()
@*/
int KSPRegisterDestroy(void)
{
  int ierr;

  PetscFunctionBegin;
  if (KSPList) {
    ierr = FListDestroy(KSPList);CHKERRQ(ierr);
    KSPList = 0;
  }
  KSPRegisterAllCalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"KSPGetType"
/*@C
   KSPGetType - Gets the KSP type as a string from the KSP object.

   Not Collective

   Input Parameter:
.  ksp - Krylov context 

   Output Parameter:
.  name - name of KSP method 

   Level: intermediate

.keywords: KSP, get, method, name

.seealso: KSPSetType()
@*/
int KSPGetType(KSP ksp,KSPType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  *type = ksp->type_name;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"KSPPrintHelp"
/*@ 
   KSPPrintHelp - Prints all options for the KSP component.

   Collective on KSP

   Input Parameter:
.  ksp - the KSP context

   Options Database Keys:
+  -help - Prints KSP options
-  -h - Prints KSP options

   Level: developer

.keywords: KSP, help

.seealso: KSPSetFromOptions()
@*/
int KSPPrintHelp(KSP ksp)
{
  char p[64];
  int  ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  ierr = PetscStrcpy(p,"-");CHKERRQ(ierr);
  if (ksp->prefix)  {ierr = PetscStrcat(p,ksp->prefix);CHKERRQ(ierr);}

  if (!KSPRegisterAllCalled) {ierr = KSPRegisterAll(PETSC_NULL);CHKERRQ(ierr);}
  ierr = (*PetscHelpPrintf)(ksp->comm,"KSP options -------------------------------------------------\n");CHKERRQ(ierr);
  ierr = FListPrintTypes(ksp->comm,stdout,ksp->prefix,"ksp_type",KSPList);CHKERRQ(ierr);CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(ksp->comm," %sksp_rtol <tol>: relative tolerance, defaults to %g\n",
                   p,ksp->rtol);CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(ksp->comm," %sksp_atol <tol>: absolute tolerance, defaults to %g\n",
                   p,ksp->atol);CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(ksp->comm," %sksp_divtol <tol>: divergence tolerance, defaults to %g\n",
                     p,ksp->divtol);CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(ksp->comm," %sksp_max_it <maxit>: maximum iterations, defaults to %d\n",
                     p,ksp->max_it);CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(ksp->comm," %sksp_preres: use preconditioned residual norm in convergence test\n",p);CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(ksp->comm," %sksp_right_pc: use right preconditioner instead of left\n",p);CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(ksp->comm," %sksp_avoid_norms: do not compute residual norms (CG and Bi-CG-stab)\n",p);CHKERRQ(ierr);

  ierr = (*PetscHelpPrintf)(ksp->comm," KSP Monitoring Options: Choose any of the following\n");CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(ksp->comm,"   %sksp_cancelmonitors: cancel all monitors hardwired in code\n",p);CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(ksp->comm,"   %sksp_monitor: at each iteration print (usually preconditioned) \n\
  residual norm to stdout\n",p);CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(ksp->comm,"   %sksp_smonitor: same as the above, but prints fewer digits of the\n\
    residual norm for small residual norms. This is useful to conceal\n\
    meaningless digits that may be different on different machines.\n",p);CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(ksp->comm,"   %sksp_xmonitor: use X graphics monitor of (usually \n\
    preconditioned) residual norm\n",p);CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(ksp->comm,"   %sksp_truemonitor: at each iteration print true and preconditioned\n",p);CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(ksp->comm,"                      residual norms to stdout\n");CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(ksp->comm,"   %sksp_xtruemonitor: use X graphics monitor of true\n",p);CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(ksp->comm,"                                 residual norm\n");CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(ksp->comm,"   %sksp_singmonitor: calculate singular values during linear solve\n",p);CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(ksp->comm,"       (only for CG and GMRES)\n");CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(ksp->comm,"   %sksp_plot_eigenvalues_explicitly\n",p);CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(ksp->comm,"   %sksp_plot_eigenvalues\n",p);CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(ksp->comm,"   %sksp_compute_eigenvalues\n",p);CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(ksp->comm,"   %sksp_compute_singularvalues\n",p);CHKERRQ(ierr);

  if (ksp->ops->printhelp) {
    ierr = (*ksp->ops->printhelp)(ksp,p);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#define MAXSETFROMOPTIONS 5
extern int numberofsetfromoptions;
extern int (*othersetfromoptions[MAXSETFROMOPTIONS])(KSP);

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"KSPSetTypeFromOptions"
/*@
   KSPSetTypeFromOptions - Sets KSP type from the options database, if not
       given then sets default.

   Collective on KSP

   Input Parameters:
.  ksp - the Krylov space context

   Level: developer

.keywords: KSP, set, from, options, database

.seealso: KSPPrintHelp(), KSPSetFromOptions(), SLESSetFromOptions(),
          SLESSetTypeFromOptions()
@*/
int KSPSetTypeFromOptions(KSP ksp)
{
  PetscTruth flg;
  int        ierr;
  char       type[256];

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);

  ierr = OptionsGetString(ksp->prefix,"-ksp_type",type,256,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = KSPSetType(ksp,type);CHKERRQ(ierr);
  }
  /*
    Set the type if it was never set.
  */
  if (!ksp->type_name) {
    ierr = KSPSetType(ksp,KSPGMRES);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_AMS)
#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"KSPOptionsPublish"
int KSPOptionsPublish(KSP ksp)
{
  int  ierr,ntypes;
  char **types;

  PetscFunctionBegin;
  ierr = OptionsSelectBegin(ksp->comm,ksp->prefix,"KSP Options");CHKERRQ(ierr);

  ierr = FListGet(KSPList,&types,&ntypes);CHKERRQ(ierr);
  ierr = OptionsSelectList(ksp->comm,"-ksp_type","Krylov method",types,ntypes,ksp->type_name ? ksp->type_name : KSPGMRES);CHKERRQ(ierr);
  ierr = PetscFree(types);CHKERRQ(ierr);

  ierr = OptionsSelectInt(ksp->comm,"-ksp_max_it","Maximum number of iterations",ksp->max_it);CHKERRQ(ierr);
  ierr = OptionsSelectDouble(ksp->comm,"-ksp_rtol","Relative decrease in residual norm",ksp->rtol);CHKERRQ(ierr);
  ierr = OptionsSelectDouble(ksp->comm,"-ksp_atol","Absolute value of residual norm",ksp->atol);CHKERRQ(ierr);
  ierr = OptionsSelectDouble(ksp->comm,"-ksp_divtol","Residual norm increase cause divergence",ksp->divtol);CHKERRQ(ierr);
  ierr = OptionsSelectName(ksp->comm,"-ksp_monitor","Monitor preconditioned residual norm");CHKERRQ(ierr);
  ierr = OptionsSelectName(ksp->comm,"-ksp_vecmonitor","Monitor solution graphically");CHKERRQ(ierr);
  ierr = OptionsSelectName(ksp->comm,"-ksp_truemonitor","Monitor true residual norm");CHKERRQ(ierr);
  ierr = OptionsSelectName(ksp->comm,"-ksp_singmonitor","Monitor singular values");CHKERRQ(ierr);
  ierr = OptionsSelectName(ksp->comm,"-ksp_xmonitor","Monitor graphically preconditioned residual norm");CHKERRQ(ierr);
  ierr = OptionsSelectName(ksp->comm,"-ksp_xtruemonitor","Monitor graphically true residual norm");CHKERRQ(ierr);
  ierr = OptionsSelectName(ksp->comm,"-ksp_plot_eigenvalues","Scatter plot extreme eigenvalues");CHKERRQ(ierr);
  ierr = OptionsSelectEnd(ksp->comm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"KSPSetFromOptions"
/*@
   KSPSetFromOptions - Sets KSP options from the options database.
   This routine must be called before KSPSetUp() if the user is to be 
   allowed to set the Krylov type. 

   Collective on KSP

   Input Parameters:
.  ksp - the Krylov space context

   Options Database Keys:
+   -ksp_max_it - maximum number of linear iterations
.   -ksp_rtol rtol - relative tolerance used in default determination of convergence, i.e.
                if residual norm decreases by this factor than convergence is declared
.   -ksp_atol atol - absolute tolerance used in default convergence test, i.e. if residual 
                norm is less than this then convergence is declared
.   -ksp_divtol tol - if residual norm increases by this factor than divergence is declared
.   -ksp_avoid_norms - skip norms used in convergence tests (useful only when not using 
                       convergence test (say you always want to run with 5 iterations) to 
                       save on communication overhead
.   -ksp_cancelmonitors - cancel all previous convergene monitor routines set
.   -ksp_monitor - print residual norm at each iteration
.   -ksp_xmonitor - plot residual norm at each iteration
.   -ksp_vecmonitor - plot solution at each iteration
-   -ksp_singmonitor - monitor extremem singular values at each iteration

   Notes:  
   To see all options, run your program with the -help option
   or consult the users manual.

   Level: developer

.keywords: KSP, set, from, options, database

.seealso: KSPPrintHelp()
@*/
int KSPSetFromOptions(KSP ksp)
{
  int        ierr,i;
  PetscTruth flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);

#if defined(PETSC_HAVE_AMS)
  if (PetscPublishOptions) {
    ierr = KSPOptionsPublish(ksp);CHKERRQ(ierr);
  }
#endif

  ierr = KSPSetTypeFromOptions(ksp);CHKERRQ(ierr);

  ierr = OptionsGetInt(ksp->prefix,"-ksp_max_it",&ksp->max_it,PETSC_NULL);CHKERRQ(ierr);
  ierr = OptionsGetDouble(ksp->prefix,"-ksp_rtol",&ksp->rtol,PETSC_NULL);CHKERRQ(ierr);
  ierr = OptionsGetDouble(ksp->prefix,"-ksp_atol",&ksp->atol,PETSC_NULL);CHKERRQ(ierr);
  ierr = OptionsGetDouble(ksp->prefix,"-ksp_divtol",&ksp->divtol,PETSC_NULL);CHKERRQ(ierr);

  ierr = OptionsHasName(ksp->prefix,"-ksp_avoid_norms",&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = KSPSetAvoidNorms(ksp);CHKERRQ(ierr);
  }

  /* -----------------------------------------------------------------------*/
  /*
    Cancels all monitors hardwired into code before call to KSPSetFromOptions()
  */
  ierr = OptionsHasName(ksp->prefix,"-ksp_cancelmonitors",&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = KSPClearMonitor(ksp);CHKERRQ(ierr);
  }
  /*
    Prints preconditioned residual norm at each iteration
  */
  ierr = OptionsHasName(ksp->prefix,"-ksp_monitor",&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = KSPSetMonitor(ksp,KSPDefaultMonitor,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  }
  /*
    Plots the vector solution 
  */
  ierr = OptionsHasName(ksp->prefix,"-ksp_vecmonitor",&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = KSPSetMonitor(ksp,KSPVecViewMonitor,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  }
  /*
    Prints preconditioned and true residual norm at each iteration
  */
  ierr = OptionsHasName(ksp->prefix,"-ksp_truemonitor",&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = KSPSetMonitor(ksp,KSPTrueMonitor,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  }
  /*
    Prints extreme eigenvalue estimates at each iteration
  */
  ierr = OptionsHasName(ksp->prefix,"-ksp_singmonitor",&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = KSPSetComputeSingularValues(ksp);CHKERRQ(ierr);
    ierr = KSPSetMonitor(ksp,KSPSingularValueMonitor,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr); 
  }
  /*
    Prints preconditioned residual norm with fewer digits
    */
  ierr = OptionsHasName(ksp->prefix,"-ksp_smonitor",&flg);CHKERRQ(ierr); 
  if (flg) {
    ierr = KSPSetMonitor(ksp,KSPDefaultSMonitor,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  }
  /*
    Graphically plots preconditioned residual norm
  */
  ierr = OptionsHasName(ksp->prefix,"-ksp_xmonitor",&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = KSPSetMonitor(ksp,KSPLGMonitor,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  }
  /*
    Graphically plots preconditioned and true residual norm
  */
  ierr = OptionsHasName(ksp->prefix,"-ksp_xtruemonitor",&flg);CHKERRQ(ierr);
  if (flg){
    ierr = KSPSetMonitor(ksp,KSPLGTrueMonitor,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  }
  /* -----------------------------------------------------------------------*/
  ierr = OptionsHasName(ksp->prefix,"-ksp_preres",&flg);CHKERRQ(ierr);
  if (flg) { ierr = KSPSetUsePreconditionedResidual(ksp);CHKERRQ(ierr);}
  ierr = OptionsHasName(ksp->prefix,"-ksp_left_pc",&flg);CHKERRQ(ierr);
  if (flg) { ierr = KSPSetPreconditionerSide(ksp,PC_LEFT);CHKERRQ(ierr); }
  ierr = OptionsHasName(ksp->prefix,"-ksp_right_pc",&flg);CHKERRQ(ierr);
  if (flg) { ierr = KSPSetPreconditionerSide(ksp,PC_RIGHT);CHKERRQ(ierr);}
  ierr = OptionsHasName(ksp->prefix,"-ksp_symmetric_pc",&flg);CHKERRQ(ierr);
  if (flg) { ierr = KSPSetPreconditionerSide(ksp,PC_SYMMETRIC);CHKERRQ(ierr);}

  ierr = OptionsHasName(ksp->prefix,"-ksp_compute_singularvalues",&flg);CHKERRQ(ierr);
  if (flg) { ierr = KSPSetComputeSingularValues(ksp);CHKERRQ(ierr); }
  ierr = OptionsHasName(ksp->prefix,"-ksp_compute_eigenvalues",&flg);CHKERRQ(ierr);
  if (flg) { ierr = KSPSetComputeSingularValues(ksp);CHKERRQ(ierr); }
  ierr = OptionsHasName(ksp->prefix,"-ksp_plot_eigenvalues",&flg);CHKERRQ(ierr);
  if (flg) { ierr = KSPSetComputeSingularValues(ksp);CHKERRQ(ierr); }

  for (i=0; i<numberofsetfromoptions; i++) {
    ierr = (*othersetfromoptions[i])(ksp);CHKERRQ(ierr);
  }

  ierr = OptionsHasName(PETSC_NULL,"-help",&flg);CHKERRQ(ierr);
  if (flg) { ierr = KSPPrintHelp(ksp);CHKERRQ(ierr);  }

#if defined(PETSC_HAVE_AMS)
  if (PetscPublishOptions && ksp->ops->publishoptions) {
    ierr = (*ksp->ops->publishoptions)(ksp);CHKERRQ(ierr);
  }
#endif
  if (ksp->ops->setfromoptions) {
    ierr = (*ksp->ops->setfromoptions)(ksp);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

/*MC
   KSPRegisterDynamic - Adds a method to the Krylov subspace solver package.

   Synopsis:
   KSPRegisterDynamic(char *name_solver,char *path,char *name_create,int (*routine_create)(KSP))

   Not Collective

   Input Parameters:
+  name_solver - name of a new user-defined solver
.  path - path (either absolute or relative) the library containing this solver
.  name_create - name of routine to create method context
-  routine_create - routine to create method context

   Notes:
   KSPRegisterDynamic() may be called multiple times to add several user-defined solvers.

   If dynamic libraries are used, then the fourth input argument (routine_create)
   is ignored.

   Sample usage:
.vb
   KSPRegisterDynamic("my_solver",/home/username/my_lib/lib/libO/solaris/mylib.a,
               "MySolverCreate",MySolverCreate);
.ve

   Then, your solver can be chosen with the procedural interface via
$     KSPSetType(ksp,"my_solver")
   or at runtime via the option
$     -ksp_type my_solver

   Level: advanced

   Environmental variables such as ${PETSC_ARCH}, ${PETSC_DIR}, ${PETSC_LDIR}, ${BOPT},
   and others of the form ${any_environmental_variable} occuring in pathname will be 
   replaced with appropriate values.

.keywords: KSP, register

.seealso: KSPRegisterAll(), KSPRegisterDestroy()

M*/

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"KSPRegister"
int KSPRegister(char *sname,char *path,char *name,int (*function)(KSP))
{
  int  ierr;
  char fullname[256];

  PetscFunctionBegin;
  ierr = FListConcat(path,name,fullname);CHKERRQ(ierr);
  ierr = FListAdd(&KSPList,sname,fullname,(int (*)(void*))function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
