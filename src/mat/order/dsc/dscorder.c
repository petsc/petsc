/*$Id: dscorder.c,v 1.4 2001/01/16 18:18:44 balay Exp bsmith $*/
/* 
        Provides an interface to the DSCPACK-S ordering routines.
*/
#include "petscmat.h"
#if defined(PETSC_HAVE_DSCPACK) && !defined(PETSC_USE_COMPLEX) 
extern int dsc_s_nz; /* golbal communication mechanism for dscpack */

EXTERN_C_BEGIN
#include "dscmain.h"

#undef __FUNC__
#define __FUNC__ "MatOrdering_DSC"
int MatOrdering_DSC(Mat mat,MatOrderingType type,IS *row,IS *col)
{
  int             ierr,order_code,m,*ai,*aj;
  int             s_nz, *perm, *iperm;
  PetscTruth      flg;

  PetscFunctionBegin;

  ierr = PetscStrcmp(type,MATORDERING_DSC_ND,&flg); 
  if (flg) { 
    order_code = 1; 
  } else {
    ierr = PetscStrcmp(type,MATORDERING_DSC_MMD,&flg);
    if (flg) {
      order_code = 2;
    } else {
      ierr = PetscStrcmp(type,MATORDERING_DSC_MDF,&flg);
      if (flg) {
        order_code = 3;
      } else {
        printf(" default ordering: MATORDERING_DSC_ND is used \n");
        order_code = 1;
      }
    }
  }
  
  ierr = MatGetRowIJ(mat,0,PETSC_TRUE,&m,&ai,&aj,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ1(PETSC_ERR_SUP,"Cannot get rows for matrix type %s",((PetscObject)mat)->type_name);

  /* check for multiple calls? something like PetscInitialize() ? **/
  DSC_Open0();

  DSC_Order(order_code, m, ai, aj, &s_nz, &perm, &iperm);        
  if (DSC_STATUS.cont_or_stop == DSC_STOP_TYPE) goto ERROR_HANDLE;
  /* fix the error handling? */

  ierr = ISCreateGeneral(PETSC_COMM_SELF,m,perm,row);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_SELF,m,perm,col);CHKERRQ(ierr);                                          
  
  /* set some global structures in case they are required by DSC-ICC */
  dsc_s_nz  = s_nz;

ERROR_HANDLE:  
  if (DSC_STATUS.error_code != DSC_NO_ERROR) {
    DSC_Error_Display();
    SETERRQ(PETSC_ERR_ARG_SIZ, "DSC_ERROR");
  }

  ierr = MatRestoreRowIJ(mat,0,PETSC_TRUE,&m,&ai,&aj,&flg);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_END

#endif
