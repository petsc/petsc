#ifndef lint
static char vcid[] = "$Id: lu.c,v 1.61 1996/03/19 21:25:10 bsmith Exp bsmith $";
#endif
/*
   Defines a direct factorization preconditioner for any Mat implementation
   Note: this need not be consided a preconditioner since it supplies
         a direct solver.
*/
#include "pcimpl.h"                /*I "pc.h" I*/
#include "pinclude/pviewer.h"

typedef struct {
  Mat         fact;       /* factored matrix */
  int         inplace;    /* flag indicating in-place factorization */
  IS          row, col;   /* index sets used for reordering */
  MatOrdering ordering;   /* matrix ordering */
} PC_LU;

/*@
   PCLUSetUseInPlace - Tells the system to do an in-place factorization.
   For some implementations, for instance, dense matrices, this enables the 
   solution of much larger problems. 

   Input Parameters:
.  pc - the preconditioner context

   Options Database Key:
$  -pc_lu_in_place

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
  PC_LU *dir;
  PetscValidHeaderSpecific(pc,PC_COOKIE);
  dir = (PC_LU *) pc->data;
  if (pc->type != PCLU) return 0;
  dir->inplace = 1;
  return 0;
}

static int PCSetFromOptions_LU(PC pc)
{
  int ierr,flg;

  ierr = OptionsHasName(pc->prefix,"-pc_lu_in_place",&flg); CHKERRQ(ierr);
  if (flg) {
    PCLUSetUseInPlace(pc);
  }
  return 0;
}

static int PCPrintHelp_LU(PC pc,char *p)
{
  PetscPrintf(pc->comm," Options for PCLU preconditioner:\n");
  PetscPrintf(pc->comm," %spc_lu_in_place: do factorization in place\n",p);
  PetscPrintf(pc->comm," -mat_order name: ordering to reduce fill",p);
  PetscPrintf(pc->comm," (nd,natural,1wd,rcm,qmd)\n");
  return 0;
}

static int PCView_LU(PetscObject obj,Viewer viewer)
{
  PC         pc = (PC)obj;
  FILE       *fd;
  PC_LU      *lu = (PC_LU *) pc->data;
  int        ierr,nzlu;
  char       *order;
  ViewerType vtype;

  MatReorderingGetName(lu->ordering,&order);
  ViewerGetType(viewer,&vtype);
  if (vtype  == ASCII_FILE_VIEWER || vtype == ASCII_FILES_VIEWER) {
    ierr = ViewerASCIIGetPointer(viewer,&fd); CHKERRQ(ierr);
    if (lu->inplace) PetscFPrintf(pc->comm,fd,"  LU: in-place factorization\n");
    else PetscFPrintf(pc->comm,fd,"  LU: out-of-place factorization\n");
    PetscFPrintf(pc->comm,fd,"      matrix ordering: %s\n",order);
    ierr = MatGetInfo(lu->fact,MAT_LOCAL,&nzlu,PETSC_NULL,PETSC_NULL);
    PetscFPrintf(pc->comm,fd,"      LU nonzeros %d\n",nzlu);
  }
  else if (vtype == STRING_VIEWER) {
    ViewerStringSPrintf(viewer," order=%s",order);
  }
  return 0;
}

static int PCGetFactoredMatrix_LU(PC pc,Mat *mat)
{
  PC_LU *dir = (PC_LU *) pc->data;
  *mat = dir->fact;
  return 0;
}

static int PCSetUp_LU(PC pc)
{
  int         ierr;
  PC_LU       *dir = (PC_LU *) pc->data;
  MatType     type;

  ierr = MatGetType(pc->pmat,&type,PETSC_NULL); CHKERRQ(ierr);
  if (type == MATSEQBDIAG) dir->ordering = ORDER_NATURAL;
  if (dir->inplace) {
    ierr = MatGetReorderingTypeFromOptions(0,&dir->ordering); CHKERRQ(ierr);
    ierr = MatGetReordering(pc->pmat,dir->ordering,&dir->row,&dir->col); CHKERRQ(ierr);
    if (dir->row) {PLogObjectParent(pc,dir->row); PLogObjectParent(pc,dir->col);}

    /* this uses an arbritrary 5.0 as the fill factor! User may set
       with the option -mat_lu_fill */
    ierr = MatLUFactor(pc->pmat,dir->row,dir->col,5.0); CHKERRQ(ierr);
    dir->fact = pc->pmat;
  }
  else {
    if (!pc->setupcalled) {
      ierr = MatGetReorderingTypeFromOptions(0,&dir->ordering); CHKERRQ(ierr);
      ierr = MatGetReordering(pc->pmat,dir->ordering,&dir->row,&dir->col); CHKERRQ(ierr);
      if (dir->row) {PLogObjectParent(pc,dir->row); PLogObjectParent(pc,dir->col);}
      ierr = MatLUFactorSymbolic(pc->pmat,dir->row,dir->col,5.0,&dir->fact); CHKERRQ(ierr);
      PLogObjectParent(pc,dir->fact);
    }
    else if (pc->flag != SAME_NONZERO_PATTERN) { 
      ierr = MatDestroy(dir->fact); CHKERRQ(ierr);
      ierr = ISDestroy(dir->row); CHKERRQ(ierr);
      ierr = ISDestroy(dir->col); CHKERRQ(ierr);
      ierr = MatGetReordering(pc->pmat,dir->ordering,&dir->row,&dir->col); CHKERRQ(ierr);
      if (dir->row) {PLogObjectParent(pc,dir->row);PLogObjectParent(pc,dir->col);}
      ierr = MatLUFactorSymbolic(pc->pmat,dir->row,dir->col,5.0,&dir->fact); CHKERRQ(ierr);
      PLogObjectParent(pc,dir->fact);
    }
    ierr = MatLUFactorNumeric(pc->pmat,&dir->fact); CHKERRQ(ierr);
  }
  return 0;
}

static int PCDestroy_LU(PetscObject obj)
{
  PC        pc   = (PC) obj;
  PC_LU *dir = (PC_LU*) pc->data;

  if (!dir->inplace) MatDestroy(dir->fact);
  if (dir->row && dir->col && dir->row != dir->col) ISDestroy(dir->row);
  if (dir->col) ISDestroy(dir->col);
  PetscFree(dir); 
  return 0;
}

static int PCApply_LU(PC pc,Vec x,Vec y)
{
  PC_LU *dir = (PC_LU *) pc->data;
  int   ierr;

  if (dir->inplace) {ierr = MatSolve(pc->pmat,x,y); CHKERRQ(ierr);}
  else              {ierr = MatSolve(dir->fact,x,y); CHKERRQ(ierr);}
  return 0;
}

int PCCreate_LU(PC pc)
{
  PC_LU *dir     = PetscNew(PC_LU); CHKPTRQ(dir);
  dir->fact      = 0;
  dir->inplace   = 0;
  dir->col       = 0;
  dir->row       = 0;
  dir->ordering  = ORDER_ND;
  pc->destroy    = PCDestroy_LU;
  pc->apply      = PCApply_LU;
  pc->setup      = PCSetUp_LU;
  pc->type       = PCLU;
  pc->data       = (void *) dir;
  pc->setfrom    = PCSetFromOptions_LU;
  pc->printhelp  = PCPrintHelp_LU;
  pc->view       = PCView_LU;
  pc->getfactoredmatrix = PCGetFactoredMatrix_LU;
  return 0;
}
