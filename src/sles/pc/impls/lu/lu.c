#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: lu.c,v 1.93 1998/04/24 21:21:22 curfman Exp bsmith $";
#endif
/*
   Defines a direct factorization preconditioner for any Mat implementation
   Note: this need not be consided a preconditioner since it supplies
         a direct solver.
*/
#include "src/pc/pcimpl.h"                /*I "pc.h" I*/
#include "pinclude/pviewer.h"

typedef struct {
  Mat               fact;             /* factored matrix */
  double            fill, actualfill; /* expected and actual fill in factor */
  int               inplace;          /* flag indicating in-place factorization */
  IS                row, col;         /* index sets used for reordering */
  MatReorderingType ordering;         /* matrix ordering */
} PC_LU;


#undef __FUNC__  
#define __FUNC__ "PCSetFromOptions_LU"
static int PCSetFromOptions_LU(PC pc)
{
  int    ierr,flg;
  double fill;

  PetscFunctionBegin;
  ierr = OptionsHasName(pc->prefix,"-pc_lu_in_place",&flg); CHKERRQ(ierr);
  if (flg) {
    ierr = PCLUSetUseInPlace(pc); CHKERRQ(ierr);
  }
  ierr = OptionsGetDouble(pc->prefix,"-pc_lu_fill",&fill,&flg); CHKERRQ(ierr);
  if (flg) {
    ierr = PCLUSetFill(pc,fill); CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCPrintHelp_LU"
static int PCPrintHelp_LU(PC pc,char *p)
{
  PetscFunctionBegin;
  (*PetscHelpPrintf)(pc->comm," Options for PCLU preconditioner:\n");
  (*PetscHelpPrintf)(pc->comm," %spc_lu_in_place: do factorization in place\n",p);
  (*PetscHelpPrintf)(pc->comm," %spc_lu_fill <fill>: expected fill in factor\n",p);
  (*PetscHelpPrintf)(pc->comm," -mat_order <name>: ordering to reduce fill",p);
  (*PetscHelpPrintf)(pc->comm," (nd,natural,1wd,rcm,qmd)\n");
  (*PetscHelpPrintf)(pc->comm," %spc_lu_nonzeros_along_diagonal <tol>: changes column ordering\n",p);
  (*PetscHelpPrintf)(pc->comm,"    to reduce the change of obtaining zero pivot during LU.\n");
  (*PetscHelpPrintf)(pc->comm,"    If <tol> not given defaults to 1.e-10.\n");
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCView_LU"
static int PCView_LU(PC pc,Viewer viewer)
{
  FILE       *fd;
  PC_LU      *lu = (PC_LU *) pc->data;
  int        ierr;
  char       *order;
  ViewerType vtype;

  PetscFunctionBegin;
  ierr = MatReorderingGetName(lu->ordering,&order); CHKERRQ(ierr);
  ViewerGetType(viewer,&vtype);
  if (vtype  == ASCII_FILE_VIEWER || vtype == ASCII_FILES_VIEWER) {
    MatInfo info;
    ierr = ViewerASCIIGetPointer(viewer,&fd); CHKERRQ(ierr);
    if (lu->inplace) PetscFPrintf(pc->comm,fd,"  LU: in-place factorization\n");
    else PetscFPrintf(pc->comm,fd,"  LU: out-of-place factorization\n");
    PetscFPrintf(pc->comm,fd,"      matrix ordering: %s\n",order);
    if (lu->fact) {
      ierr = MatGetInfo(lu->fact,MAT_LOCAL,&info); CHKERRQ(ierr);
      PetscFPrintf(pc->comm,fd,"      LU nonzeros %g\n",info.nz_used);
    }
  } else if (vtype == STRING_VIEWER) {
    ViewerStringSPrintf(viewer," order=%s",order);
  } else {
    SETERRQ(1,1,"Viewer type not supported for this object");
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCGetFactoredMatrix_LU"
static int PCGetFactoredMatrix_LU(PC pc,Mat *mat)
{
  PC_LU *dir = (PC_LU *) pc->data;

  PetscFunctionBegin;
  *mat = dir->fact;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCSetUp_LU"
static int PCSetUp_LU(PC pc)
{
  int         ierr,flg;
  PC_LU       *dir = (PC_LU *) pc->data;
  MatType     type;

  PetscFunctionBegin;
  ierr = MatGetType(pc->pmat,&type,PETSC_NULL); CHKERRQ(ierr);
  if (dir->inplace) {
    ierr = MatGetReorderingTypeFromOptions(0,&dir->ordering); CHKERRQ(ierr);
    ierr = MatGetReordering(pc->pmat,dir->ordering,&dir->row,&dir->col); CHKERRQ(ierr);
    if (dir->row) {PLogObjectParent(pc,dir->row); PLogObjectParent(pc,dir->col);}
    ierr = MatLUFactor(pc->pmat,dir->row,dir->col,dir->fill); CHKERRQ(ierr);
    dir->fact = pc->pmat;
  } else {
    if (!pc->setupcalled) {
      ierr = MatGetReorderingTypeFromOptions(0,&dir->ordering); CHKERRQ(ierr);
      ierr = MatGetReordering(pc->pmat,dir->ordering,&dir->row,&dir->col); CHKERRQ(ierr);
      ierr = OptionsHasName(pc->prefix,"-pc_lu_nonzeros_along_diagonal",&flg);CHKERRQ(ierr);
      if (flg) {
        double tol = 1.e-10;
        ierr = OptionsGetDouble(pc->prefix,"-pc_lu_nonzeros_along_diagonal",&tol,&flg);CHKERRQ(ierr);
        ierr = MatReorderForNonzeroDiagonal(pc->pmat,tol,dir->row,dir->col);CHKERRQ(ierr);
      }
      if (dir->row) {PLogObjectParent(pc,dir->row); PLogObjectParent(pc,dir->col);}
      ierr = MatLUFactorSymbolic(pc->pmat,dir->row,dir->col,dir->fill,&dir->fact); CHKERRQ(ierr);
      PLogObjectParent(pc,dir->fact);
    } else if (pc->flag != SAME_NONZERO_PATTERN) { 
      ierr = MatDestroy(dir->fact); CHKERRQ(ierr);
      if (dir->row && dir->col && dir->row != dir->col) {ierr = ISDestroy(dir->row);CHKERRQ(ierr);}
      if (dir->col) {ierr = ISDestroy(dir->col); CHKERRQ(ierr);}
      ierr = MatGetReordering(pc->pmat,dir->ordering,&dir->row,&dir->col); CHKERRQ(ierr);
      ierr = OptionsHasName(pc->prefix,"-pc_lu_nonzeros_along_diagonal",&flg);CHKERRQ(ierr);
      if (flg) {
        double tol = 1.e-10;
        ierr = OptionsGetDouble(pc->prefix,"-pc_lu_nonzeros_along_diagonal",&tol,&flg);CHKERRQ(ierr);
        ierr = MatReorderForNonzeroDiagonal(pc->pmat,tol,dir->row,dir->col);CHKERRQ(ierr);
      }
      if (dir->row) {PLogObjectParent(pc,dir->row); PLogObjectParent(pc,dir->col);}
      ierr = MatLUFactorSymbolic(pc->pmat,dir->row,dir->col,dir->fill,&dir->fact); CHKERRQ(ierr);
      PLogObjectParent(pc,dir->fact);
    }
    ierr = MatLUFactorNumeric(pc->pmat,&dir->fact); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCDestroy_LU"
static int PCDestroy_LU(PC pc)
{
  PC_LU *dir = (PC_LU*) pc->data;
  int   ierr;

  PetscFunctionBegin;
  if (!dir->inplace && dir->fact) {ierr = MatDestroy(dir->fact); CHKERRQ(ierr);}
  if (dir->row && dir->col && dir->row != dir->col) {ierr = ISDestroy(dir->row); CHKERRQ(ierr);}
  if (dir->col) {ierr = ISDestroy(dir->col); CHKERRQ(ierr);}
  PetscFree(dir); 
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCApply_LU"
static int PCApply_LU(PC pc,Vec x,Vec y)
{
  PC_LU *dir = (PC_LU *) pc->data;
  int   ierr;

  PetscFunctionBegin;
  if (dir->inplace) {ierr = MatSolve(pc->pmat,x,y); CHKERRQ(ierr);}
  else              {ierr = MatSolve(dir->fact,x,y); CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

/* -----------------------------------------------------------------------------------*/

#undef __FUNC__  
#define __FUNC__ "PCLUSetFill_LU"
int PCLUSetFill_LU(PC pc,double fill)
{
  PC_LU *dir;

  PetscFunctionBegin;
  dir = (PC_LU *) pc->data;
  dir->fill = fill;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCLUSetUseInPlace_LU"
int PCLUSetUseInPlace_LU(PC pc)
{
  PC_LU *dir;

  PetscFunctionBegin;
  dir = (PC_LU *) pc->data;
  dir->inplace = 1;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCLUSetMatReordering_LU"
int PCLUSetMatReordering_LU(PC pc, MatReorderingType ordering)
{
  PC_LU *dir = (PC_LU *) pc->data;

  PetscFunctionBegin;
  dir->ordering = ordering;
  PetscFunctionReturn(0);
}

/* -----------------------------------------------------------------------------------*/

#undef __FUNC__  
#define __FUNC__ "PCLUSetFill"
/*@
   PCLUSetFill - Indicate the amount of fill you expect in the factored matrix,
       fill = number nonzeros in factor/number nonzeros in original matrix.

   Collective on PC
   
   Input Parameters:
+  pc - the preconditioner context
-  fill - amount of expected fill

   Options Database Key:
.  -pc_lu_fill <fill> - Sets fill amount

   Note:
   For sparse matrix factorizations it is difficult to predict how much 
   fill to expect. By running with the option -log_info PETSc will print the 
   actual amount of fill used; allowing you to set the value accurately for
   future runs. Bt default PETSc uses a value of 5.0

.keywords: PC, set, factorization, direct, fill

.seealso: PCILUSetFill()
@*/
int PCLUSetFill(PC pc,double fill)
{
  int ierr, (*f)(PC,double);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE);
  if (fill < 1.0) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,1,"Fill factor cannot be less then 1.0");
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCLUSetFill_C",(void **)&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,fill);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCLUSetUseInPlace"
/*@
   PCLUSetUseInPlace - Tells the system to do an in-place factorization.
   For some implementations, for instance, dense matrices, this enables the 
   solution of much larger problems. 

   Collective on PC

   Input Parameters:
.  pc - the preconditioner context

   Options Database Key:
.  -pc_lu_in_place - Activates in-place factorization

   Note:
   PCLUSetUseInplace() can only be used with the KSP method KSPPREONLY.
   This is because the Krylov space methods require an application of the 
   matrix multiplication, which is not possible here because the matrix has 
   been factored in-place, replacing the original matrix.

.keywords: PC, set, factorization, direct, inplace, in-place, LU

.seealso: PCILUSetUseInPlace()
@*/
int PCLUSetUseInPlace(PC pc)
{
  int ierr, (*f)(PC);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE);
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCLUSetUseInPlace_C",(void **)&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

/*@
    PCLUSetMatReordering - Sets the ordering routine (to reduce fill) to 
    be used it the LU factorization.

    Collective on PC

    Input Parameters:
+   pc - the preconditioner context
-   ordering - the matrix ordering name, for example, ORDER_ND or ORDER_RCM

   Options Database Key:
.  -mat_order <nd,rcm,...> - Sets ordering routine

.seealso: PCILUSetMatReordering()
@*/
int PCLUSetMatReordering(PC pc, MatReorderingType ordering)
{
  int ierr, (*f)(PC,MatReorderingType);

  PetscFunctionBegin;
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCLUSetMatReodering_C",(void **)&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,ordering);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------------ */

#undef __FUNC__  
#define __FUNC__ "PCCreate_LU"
int PCCreate_LU(PC pc)
{
  int   ierr;
  PC_LU *dir     = PetscNew(PC_LU); CHKPTRQ(dir);

  PetscFunctionBegin;
  PLogObjectMemory(pc,sizeof(PC_LU));

  dir->fact             = 0;
  dir->inplace          = 0;
  dir->fill             = 5.0;
  dir->col              = 0;
  dir->row              = 0;
  dir->ordering         = ORDER_ND;
  pc->destroy           = PCDestroy_LU;
  pc->apply             = PCApply_LU;
  pc->setup             = PCSetUp_LU;
  pc->data              = (void *) dir;
  pc->setfromoptions    = PCSetFromOptions_LU;
  pc->printhelp         = PCPrintHelp_LU;
  pc->view              = PCView_LU;
  pc->applyrich         = 0;
  pc->getfactoredmatrix = PCGetFactoredMatrix_LU;

  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCLUSetFill_C","PCLUSetFill_LU",
                    (void*)PCLUSetFill_LU);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCLUSetUseInPlace_C","PCLUSetUseInPlace_LU",
                    (void*)PCLUSetUseInPlace_LU);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCLUSetMatReordering_C","PCLUSetMatReordering_LU",
                    (void*)PCLUSetMatReordering_LU);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
