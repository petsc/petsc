/* $Id: modpcf.c,v 1.3 1999/11/24 21:54:59 bsmith Exp bsmith $*/

#include "sles.h" 
#undef __FUNC__  
#define __FUNC__ "KSPFGMRESSetModifyPC"
/*@
   KSPFGMRESSetModifyPC - Sets the routine used by FGMRES to modify the preconditioner.

   Collective on KSP

   Input Parameters:
+  ksp - iterative context obtained from KSPCreate
-  fcn - modifypc function

   Calling Sequence of function:
    ierr = int fcn(KSP ksp,int total_its,int max_total_its,int loc_its,int,max_loc_its,double res_norm);

    ksp - the ksp context being used.
    total_its     - the total number of FGMRES iterations that have occurred.    
    max_total_its - the maximum number of iterations allowed for the method.
    loc_its       - the number of FGMRES iterations since last restart.
    max_loc_its   - the maximum number of iterations that can occur before
                    a restart (so number of Krylov directions to be computed)
    res_norm      - the current residual norm.


   Notes:
   Several modifypc routines are predefined, including
    KSPFGMRESModifyPCNoChange()
    KSPFGMRESModifyPCSLES()

   Options Database Keys:
   -ksp_fgmres_modifypcnochange
   -ksp_fgmres_modifypcsles

@*/
int KSPFGMRESSetModifyPC( KSP ksp, int (*fcn)( KSP, int, int, int, int, double) )
{
  int ierr, (*f)(KSP, int (*)( KSP, int, int, int, int, double ));

  PetscFunctionBegin;
  PetscValidHeaderSpecific( ksp, KSP_COOKIE );
  ierr = PetscObjectQueryFunction( (PetscObject)ksp, "KSPFGMRESSetModifyPC_C", (void **)&f); CHKERRQ(ierr);
  if (f) {
    ierr = (*f)( ksp, fcn ); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}



/* The following are different routines used to modify the preconditioner */

#undef __FUNC__  
#define __FUNC__ "KSPFGMRESModifyPCNoChange"
/*@

  FGMRESModifyPCNoChange - this is the default used by fgmres - it doesn't change the preconditioner. 

  Input Parameters:
+    ksp - the ksp context being used.
.    total_its     - the total number of FGMRES iterations that have occurred.    
.    max_total_its - the maximum number of iterations allowed for the method.
.    loc_its       - the number of FGMRES iterations since last restart.
.    max_loc_its   - the maximum number of iterations that can occur before
                    a restart (so number of Krylov directions to be computed)
-    res_norm      - the current residual norm.


You can use this as a template!

@*/
int KSPFGMRESModifyPCNoChange(KSP ksp,int total_its,int max_total_its,int loc_its,int max_loc_its,double res_norm)
{
  PC         pc;
  int        ierr;

  PetscFunctionBegin;
  ierr = KSPGetPC( ksp, &pc ); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "KSPFGMRESModifyPCSLES"
/*@

 KSPFGMRESModifyPCSLES - modifies the attributes of the
     GMRES preconditioner.  It serves as an example (not as something 
     useful!) 

  Input Parameters:
+    ksp - the ksp context being used.
.    total_its     - the total number of FGMRES iterations that have occurred.    
.    max_total_its - the maximum number of iterations allowed for the method.
.    loc_its       - the number of FGMRES iterations since last restart.
.    max_loc_its   - the maximum number of iterations that can occur before
                    a restart (so number of Krylov directions to be computed)
-    res_norm      - the current residual norm.


 This could be used as a template!

@*/
int KSPFGMRESModifyPCSLES(KSP ksp,int total_its,int max_total_its,int loc_its,int max_loc_its,double res_norm)
{
  PC         pc;
  int        ierr,maxits;
  SLES       sub_sles;
  KSP        sub_ksp;
  double     rtol, atol, dtol;
  PetscTruth issles;

  PetscFunctionBegin;

  ierr = KSPGetPC( ksp, &pc ); CHKERRQ(ierr);

  ierr = PetscTypeCompare((PetscObject) pc, PCSLES,&issles );CHKERRQ(ierr);
  if (issles) { 
    ierr = PCSLESGetSLES( pc, &sub_sles ); CHKERRQ(ierr);
    ierr = SLESGetKSP( sub_sles, &sub_ksp ); CHKERRQ(ierr);
  
    /* note that at this point you could check the type of KSP with KSPGetType() */  

    /* Now we can use functions such as KSPGMRESSetRestart() or 
      KSPGMRESSetOrthogonalization() or KSPSetTolerances() */

    /* we can vary the tolerances depending on the iteration number we are on.  For
      example, do more iterations in our GMRES inner iteration for the 
      first 4 Krylov directions (note loc_it will start at 0) */
    ierr = KSPGetTolerances( sub_ksp, &rtol, &atol, &dtol, &maxits ); CHKERRQ(ierr);
    if (loc_its == 0 ) {
       maxits = 20;
       rtol = 1.e-2;
    } else if (loc_its == 4) { 
       maxits = 5;
       rtol = 1.e-1;
    }
    ierr = KSPSetTolerances( sub_ksp, rtol, atol, dtol, maxits ); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}





