/* $Id: modpcf.c,v 1.2 1999/11/10 03:20:57 bsmith Exp bsmith $*/

#include "src/sles/ksp/impls/fgmres/fgmresp.h"      /*I  "ksp.h"  I*/
#include "/home/baker/working/allisonpc.h"
#include "src/sles/pc/pcimpl.h"   /*I "pc.h" I*/
#include "src/sles/slesimpl.h"   /*I "sles.h" I*/
 
/*
   KSPFGMRESSetModifyPC - Sets the routine used by FGMRES to modify the preconditioner
    note pcfamily must be used as the precondtioner type.

   Collective on KSP

   Input Parameters:
+  ksp - iterative context obtained from KSPCreate
-  fcn - modifypc function

   Calling Sequence of function:
    errorcode = int fcn( KSP ksp, int total_its, int max_total_its, int loc_its, int, max_loc_its, double res_norm);

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
    KSPFGMRESModifyPCGMRESVariableEx()
    KSPGMRESModifyPCEx()
   Options Database Keys:
   -ksp_fgmres_modifypcnochange
   -ksp_fgmres_modifypcgmresvariableex
   -ksp_fgmres_modifypcex

*/
#undef __FUNC__  
#define __FUNC__ "KSPFGMRESSetModifyPC"
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

/*

FGMRESModifyPCNoChange - this is the default used by fgmres - it doesn't change the preconditioner. 

Input:
    ksp - the ksp context being used.
    total_its     - the total number of FGMRES iterations that have occurred.    
    max_total_its - the maximum number of iterations allowed for the method.
    loc_its       - the number of FGMRES iterations since last restart.
    max_loc_its   - the maximum number of iterations that can occur before
                    a restart (so number of Krylov directions to be computed)
    res_norm      - the current residual norm.


You can use this as a template!

*/

#undef __FUNC__  
#define __FUNC__ "KSPFGMRESModifyPCNoChange"
int KSPFGMRESModifyPCNoChange( KSP ksp, int total_its, int max_total_its, int loc_its, int max_loc_its, double res_norm)
{
  PC         pc;
  int        ierr;
  PCType     familytype;
  PetscTruth isfamily;

  PetscFunctionBegin;
  ierr = KSPGetPC( ksp, &pc ); CHKERRQ(ierr);

  ierr = PetscTypeCompare((PetscObject)pc, PCFAMILY,&isfamily );CHKERRQ(ierr);
  if (isfamily) { 
    /* currently we can only modify the pc is pcfamily is being used */
    /* might want to add a warning message that pcfamily should be used */
    PetscFunctionReturn(0);
  }


   /* Use PCFamilyGetPCType to get the type of preconditioner currently
      being used by pcfamily. */
  ierr = PCFamilyGetPCType(pc, &familytype); CHKERRQ(ierr);

   
   /* To change the type of precondioner being used, call PCFamilyChangePC,
      and supply the type you wish to change to, for example PCJACOBI, PCSLES, etc.
      This next call won't actually do anything since we are requesting a change 
      to the current type */ 
  ierr = PCFamilyChangePC( pc, familytype); CHKERRQ(ierr);


   /* To modify attributes of the current pc being used (after changing the type 
      or instead of changing the type), first get a reference to that object as 
      below using PCFamilyGetPC.  Then change any attributes using the normal 
      PC functions.  For example if familytype = PCILU, you could call 
      PCILUSetLevels(pcfamilypc, 10).  Or if the type PCSLES, then use 
      PCSLESGetSLES() followed by any sles/ksp/pc options you wish! 
      Also see KSPFGMRESModifyGMRESVariable().*/

  /*ierr = PCFamilyGetPC( pc, &pcfamilypc ); CHKERRQ(ierr);*/


  PetscFunctionReturn(0);
}

/*

 KSPFGMRESModifyPCGMRESVariableEx - modifies the attributes of the
     GMRES preconditioner.  It serves as an example (not as something 
     useful!)  It assumes you left did not initially specify the pcfamily 
     preconditioner, so that the default GMRES is used. 

Input:
    ksp - the ksp context being used.
    total_its     - the total number of FGMRES iterations that have occurred.    
    max_total_its - the maximum number of iterations allowed for the method.
    loc_its       - the number of FGMRES iterations since last restart.
    max_loc_its   - the maximum number of iterations that can occur before
                    a restart (so number of Krylov directions to be computed)
    res_norm      - the current residual norm.


 This could be used as a template!

*/
#undef __FUNC__  
#define __FUNC__ "KSPFGMRESModifyPCGMRESVariableEx"
int KSPFGMRESModifyPCGMRESVariableEx( KSP ksp, int total_its, int max_total_its, int loc_its, int max_loc_its, double res_norm)
{
  PC         pc, pcfamilypc;
  int        ierr;
  SLES       sub_sles;
  KSP        sub_ksp;
  double     rtol, atol, dtol;
  int        maxits; 
  PetscTruth isfamily,issles;

  PetscFunctionBegin;

  ierr = KSPGetPC( ksp, &pc ); CHKERRQ(ierr);

  ierr = PetscTypeCompare((PetscObject)pc, PCFAMILY,&isfamily ); CHKERRQ(ierr);
  if (isfamily) { 
    /* currently we can only modify the pc if pcfamily is being used */
    /* might want to add a warning message that pcfamily should be used */
    PetscFunctionReturn(0);
  }
  ierr = PetscTypeCompare((PetscObject) pc, PCSLES,&issles );CHKERRQ(ierr);
  if (issles) { 
    /* we only want to use the default GMRES for this contrived example */
    PetscFunctionReturn(0);
  }
 

  ierr = PCFamilyGetPC( pc, &pcfamilypc ); CHKERRQ(ierr);
  ierr = PCSLESGetSLES( pcfamilypc, &sub_sles ); CHKERRQ(ierr);
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


  PetscFunctionReturn(0);

}

/*

 KSPFGMRESModifyPCEx - Just another example.

Input:
    ksp - the ksp context being used.
    total_its     - the total number of FGMRES iterations that have occurred.    
    max_total_its - the maximum number of iterations allowed for the method.
    loc_its       - the number of FGMRES iterations since last restart.
    max_loc_its   - the maximum number of iterations that can occur before
                    a restart (so number of Krylov directions to be computed)
    res_norm      - the current residual norm.

 This could be used as a template!

*/
#undef __FUNC__  
#define __FUNC__ "KSPFGMRESModifyPCEx"
int KSPFGMRESModifyPCEx( KSP ksp, int total_its, int max_total_its, int loc_its, int max_loc_its, double res_norm)
{
  PC         pc, pcfamilypc;
  int        ierr;
  SLES       sub_sles;
  KSP        sub_ksp;
  double     rtol, atol, dtol;
  int        maxits; 
  PetscTruth isfamily;

  PetscFunctionBegin;
  ierr = KSPGetPC( ksp, &pc ); CHKERRQ(ierr);

  ierr = PetscTypeCompare((PetscObject)pc, PCFAMILY,&isfamily);CHKERRQ(ierr);
  if (isfamily) { 
    /* currently we can only modify the pc if pcfamily is being used */
    /* might want to add a warning message that pcfamily should be used */
    PetscFunctionReturn(0);
  }
 

  /* use no preconditioner for the first 3 iterations (after each 
     restart), then switch to gmres */
  if (loc_its == 0 ) {
     ierr = PCFamilyChangePC( pc, PCNONE); CHKERRQ(ierr);
  } else if (loc_its == 3)  { 
     ierr = PCFamilyChangePC( pc, PCSLES); CHKERRQ(ierr);
     ierr = PCFamilyGetPC( pc, &pcfamilypc ); CHKERRQ(ierr);

     ierr = PCSLESGetSLES( pcfamilypc, &sub_sles ); CHKERRQ(ierr);
     /* using the default SLES, which is gmres */
     ierr = SLESGetKSP( sub_sles, &sub_ksp ); CHKERRQ(ierr);
     ierr = KSPGetTolerances( sub_ksp, &rtol, &atol, &dtol, &maxits ); CHKERRQ(ierr)
     maxits = 20;
     rtol = 1.e-1;
     ierr = KSPSetTolerances( sub_ksp, rtol, atol, dtol, maxits ); CHKERRQ(ierr);
  }


  PetscFunctionReturn(0);

}





