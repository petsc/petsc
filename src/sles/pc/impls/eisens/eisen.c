#ifndef lint
static char vcid[] = "$Id: eisen.c,v 1.49 1996/04/07 22:44:30 curfman Exp curfman $";
#endif

/*
   Defines a  Eisenstat trick SSOR  preconditioner. This uses about 
 %50 of the usual amount of floating point ops used for SSOR + Krylov 
 method. But it requires actually solving the preconditioned problem 
 with both left and right preconditioning. 
*/
#include "pcimpl.h"           /*I "pc.h" I*/
#include "pinclude/pviewer.h"

typedef struct {
  Mat    shell,A;
  Vec    b,diag;     /* temporary storage for true right hand side */
  double omega;
  int    usediag;    /* indicates preconditioner should include diagonal scaling*/
} PC_Eisenstat;

/*@
   PCEisenstatUseDiagonalScaling - Causes the Eisenstat preconditioner
   to do an additional diagonal preconditioning. For matrices with very 
   different values along the diagonal this may improve convergence.

   Input Parameter:
.  pc - the preconditioner context

   Options Database Key:
$  -pc_eisenstat_diagonal_scaling

.keywords: PC, Eisenstat, use, diagonal, scaling, SSOR

.seealso: PCEisenstatSetOmega()
@*/
int PCEisenstatUseDiagonalScaling(PC pc)
{
  PC_Eisenstat *eis = (PC_Eisenstat *) pc->data;
  if (pc->type != PCEISENSTAT) return 0;
  eis->usediag = 1;
  return 0;
}

static int PCMult_Eisenstat(Mat mat,Vec b,Vec x)
{
  PC           pc;
  PC_Eisenstat *eis;
  MatShellGetContext(mat,(void **)&pc);
  eis = (PC_Eisenstat *) pc->data;
  return MatRelax(eis->A,b,eis->omega,SOR_EISENSTAT,0.0,1,x);
}

static int PCApply_Eisenstat(PC pc,Vec x,Vec y)
{
  PC_Eisenstat *eis = (PC_Eisenstat *) pc->data;
  int          ierr;

  if (eis->usediag)  {ierr = VecPointwiseMult(x,eis->diag,y);CHKERRQ(ierr);}
  else               {ierr = VecCopy(x,y);  CHKERRQ(ierr);}
  return 0; 
}

/* this cheats and looks inside KSP to determine if nonzero initial guess*/
#include "src/ksp/kspimpl.h"

static int PCPre_Eisenstat(PC pc,KSP ksp)
{
  PC_Eisenstat *eis = (PC_Eisenstat *) pc->data;
  Vec          b,x;
  int          ierr;

  if (pc->mat != pc->pmat) SETERRQ(1,"PCPre_Eisenstat:cannot have different mat+pmat"); 
 
  /* swap shell matrix and true matrix */
  eis->A    = pc->mat;
  pc->mat   = eis->shell;

  KSPGetRhs(ksp,&b);
  if (!eis->b) {
    ierr = VecDuplicate(b,&eis->b); CHKERRQ(ierr);
    PLogObjectParent(pc,eis->b);
  }
  
  /* save true b, other option is to swap pointers */
  ierr = VecCopy(b,eis->b); CHKERRQ(ierr);

  /* if nonzero initial guess, modify x */
  if (!ksp->guess_zero) {
    KSPGetSolution(ksp,&x);
    ierr = MatRelax(eis->A,x,eis->omega,SOR_APPLY_UPPER,0.0,1,x);CHKERRQ(ierr);
  }

  /* modify b by (L + D)^{-1} */
  ierr =   MatRelax(eis->A,b,eis->omega,(MatSORType)(SOR_ZERO_INITIAL_GUESS | 
                                        SOR_FORWARD_SWEEP),0.0,1,b); CHKERRQ(ierr);  
  return 0;
}

static int PCPost_Eisenstat(PC pc,KSP ksp)
{
  PC_Eisenstat *eis = (PC_Eisenstat *) pc->data;
  Vec          x,b;
  int          ierr;
  KSPGetSolution(ksp,&x);
  ierr =   MatRelax(eis->A,x,eis->omega,(MatSORType)(SOR_ZERO_INITIAL_GUESS | 
                                 SOR_BACKWARD_SWEEP),0.0,1,x); CHKERRQ(ierr);
  pc->mat = eis->A;
  /* get back true b */
  KSPGetRhs(ksp,&b);
  VecCopy(eis->b,b);
  return 0;
}

static int PCDestroy_Eisenstat(PetscObject obj)
{
  PC           pc = (PC) obj;
  PC_Eisenstat *eis = ( PC_Eisenstat  *) pc->data; 
  if (eis->b) VecDestroy(eis->b);
  if (eis->shell) MatDestroy(eis->shell);
  if (eis->diag) VecDestroy(eis->diag);
  PetscFree(eis);
  return 0;
}

static int PCSetFromOptions_Eisenstat(PC pc)
{
  double  omega;
  int     ierr,flg;

  ierr = OptionsGetDouble(pc->prefix,"-pc_eisenstat_omega",&omega,&flg); CHKERRQ(ierr);
  if (flg) {
    PCEisenstatSetOmega(pc,omega);
  }
  ierr = OptionsHasName(pc->prefix,"-pc_eisenstat_diagonal_scaling",&flg);CHKERRQ(ierr);
  if (flg) {
    PCEisenstatUseDiagonalScaling(pc);
  }
  return 0;
}

static int PCPrintHelp_Eisenstat(PC pc,char *p)
{
  PetscPrintf(pc->comm," Options for PCEisenstat preconditioner:\n");
  PetscPrintf(pc->comm," %spc_eisenstat_omega omega: relaxation factor (0<omega<2)\n",p);
  PetscPrintf(pc->comm," %spc_eisenstat_diagonal_scaling\n",p);
  return 0;
}

static int PCView_Eisenstat(PetscObject obj,Viewer viewer)
{
  PC            pc = (PC)obj;
  FILE          *fd;
  PC_Eisenstat  *eis = ( PC_Eisenstat  *) pc->data; 
  int           ierr;
  ViewerType    vtype;

  ierr = ViewerGetType(viewer,&vtype); CHKERRQ(ierr);
  if (vtype == ASCII_FILE_VIEWER || vtype == ASCII_FILES_VIEWER) {
    ierr = ViewerASCIIGetPointer(viewer,&fd); CHKERRQ(ierr);
    PetscFPrintf(pc->comm,fd,"    Eisenstat: omega = %g\n",eis->omega);
  }
  return 0;
}

static int PCSetUp_Eisenstat(PC pc)
{
  int          ierr, M, N, m, n;
  PC_Eisenstat *eis = (PC_Eisenstat *) pc->data;
  Vec          diag;

  if (pc->setupcalled == 0) {
    ierr = MatGetSize(pc->mat,&M,&N); CHKERRA(ierr);
    ierr = MatGetLocalSize(pc->mat,&m,&n); CHKERRA(ierr);
    ierr = MatCreateShell(pc->comm,m,N,M,N,(void*)pc,&eis->shell); CHKERRQ(ierr);
    PLogObjectParent(pc,eis->shell);
    ierr = MatShellSetOperation(eis->shell,MAT_MULT,(void*)PCMult_Eisenstat); 
           CHKERRQ(ierr);
  }
  if (!eis->usediag) return 0;
  if (pc->setupcalled == 0) {
    ierr = VecDuplicate(pc->vec,&diag); CHKERRQ(ierr);
    PLogObjectParent(pc,diag);
  }
  else {
    diag = eis->diag;
  }
  ierr = MatGetDiagonal(pc->pmat,diag); CHKERRQ(ierr);
  ierr = VecReciprocal(diag); CHKERRQ(ierr);
  eis->diag = diag;
  return 0;
}

int PCCreate_Eisenstat(PC pc)
{
  PC_Eisenstat *eis = PetscNew(PC_Eisenstat); CHKPTRQ(eis);
  pc->apply         = PCApply_Eisenstat;
  pc->presolve      = PCPre_Eisenstat;
  pc->postsolve     = PCPost_Eisenstat;
  pc->applyrich     = 0;
  pc->setfrom       = PCSetFromOptions_Eisenstat;
  pc->printhelp     = PCPrintHelp_Eisenstat ;
  pc->destroy       = PCDestroy_Eisenstat;
  pc->view          = PCView_Eisenstat;
  pc->type          = PCEISENSTAT;
  pc->data          = (void *) eis;
  pc->setup         = PCSetUp_Eisenstat;
  eis->omega        = 1.0;
  eis->b            = 0;
  eis->diag         = 0;
  eis->usediag      = 0;
  return 0;
}

/*@ 
   PCEisenstatSetOmega - Sets the SSOR relaxation coefficient, omega,
   to use with Eisenstat's trick (where omega = 1.0 by default).

   Input Parameters:
.  pc - the preconditioner context
.  omega - relaxation coefficient (0 < omega < 2)

   Options Database Key:
$  -pc_eisenstat_omega  omega

   Notes: 
   The Eisenstat trick implementation of SSOR requires about 50% of the
   usual amount of floating point operations used for SSOR + Krylov method;
   however, the preconditioned problem must be solved with both left 
   and right preconditioning.

   To use SSOR without the Eisenstat trick, employ the PCSOR preconditioner, 
   which can be chosen with the database options
$    -pc_type  sor  -pc_sor_symmetric

.keywords: PC, Eisenstat, set, SOR, SSOR, relaxation, omega

.seealso: PCSORSetOmega()
@*/
int PCEisenstatSetOmega(PC pc,double omega)
{
  PC_Eisenstat  *eis;
  PetscValidHeaderSpecific(pc,PC_COOKIE);
  if (pc->type != PCEISENSTAT) return 0;
  if (omega >= 2.0 || omega <= 0.0) SETERRQ(1,"PCEisenstatSetOmega:Relaxation out of range");
  eis = (PC_Eisenstat *) pc->data;
  eis->omega = omega;
  return 0;
}


