#ifndef lint
static char vcid[] = "$Id: sor.c,v 1.50 1996/03/19 21:25:08 bsmith Exp curfman $";
#endif

/*
   Defines a  (S)SOR  preconditioner for any Mat implementation
*/
#include "pcimpl.h"               /*I "pc.h" I*/
#include "pinclude/pviewer.h"

typedef struct {
  int        its;        /* inner iterations, number of sweeps */
  MatSORType sym;        /* forward, reverse, symmetric etc. */
  double     omega;
} PC_SOR;

static int PCApply_SOR(PC pc,Vec x,Vec y)
{
  PC_SOR *jac = (PC_SOR *) pc->data;
  int    ierr, flag = jac->sym | SOR_ZERO_INITIAL_GUESS;
  ierr = MatRelax(pc->pmat,x,jac->omega,(MatSORType)flag,0.0,jac->its,y);CHKERRQ(ierr);
  return 0;
}

static int PCApplyRichardson_SOR(PC pc,Vec b,Vec y,Vec w,int its)
{
  PC_SOR *jac = (PC_SOR *) pc->data;
  int    ierr, flag;
  flag = jac->sym;
  ierr = MatRelax(pc->mat,b,jac->omega,(MatSORType)flag,0.0,its,y);CHKERRQ(ierr);
  return 0;
}

/* parses arguments of the form -pc_sor [symmetric,forward,back][omega=...] */
static int PCSetFromOptions_SOR(PC pc)
{
  int    its,ierr,flg;
  double omega;

  ierr = OptionsGetDouble(pc->prefix,"-pc_sor_omega",&omega,&flg); CHKERRQ(ierr);
  if (flg) {PCSORSetOmega(pc,omega);} 
  ierr = OptionsGetInt(pc->prefix,"-pc_sor_its",&its,&flg); CHKERRQ(ierr);
  if (flg) {PCSORSetIterations(pc,its);}
  ierr = OptionsHasName(pc->prefix,"-pc_sor_symmetric",&flg); CHKERRQ(ierr);
  if (flg) {PCSORSetSymmetric(pc,SOR_SYMMETRIC_SWEEP);}
  ierr = OptionsHasName(pc->prefix,"-pc_sor_backward",&flg); CHKERRQ(ierr);
  if (flg) {PCSORSetSymmetric(pc,SOR_BACKWARD_SWEEP);}
  ierr = OptionsHasName(pc->prefix,"-pc_sor_local_symmetric",&flg); CHKERRQ(ierr);
  if (flg) {PCSORSetSymmetric(pc,SOR_LOCAL_SYMMETRIC_SWEEP);}
  ierr = OptionsHasName(pc->prefix,"-pc_sor_local_backward",&flg); CHKERRQ(ierr);
  if (flg) {PCSORSetSymmetric(pc,SOR_LOCAL_BACKWARD_SWEEP);}
  ierr = OptionsHasName(pc->prefix,"-pc_sor_local_forward",&flg); CHKERRQ(ierr);
  if (flg) {PCSORSetSymmetric(pc,SOR_LOCAL_FORWARD_SWEEP);}
  return 0;
}

static int PCPrintHelp_SOR(PC pc,char *p)
{
  PetscPrintf(pc->comm," Options for PCSOR preconditioner:\n");
  PetscPrintf(pc->comm," %spc_sor_omega omega: relaxation factor (0 < omega < 2)\n",p);
  PetscPrintf(pc->comm," %spc_sor_symmetric: use SSOR\n",p);
  PetscPrintf(pc->comm," %spc_sor_backward: use backward sweep instead of forward\n",p);
  PetscPrintf(pc->comm," %spc_sor_local_symmetric: use SSOR on each processor\n",p);
  PetscPrintf(pc->comm," %spc_sor_local_backward: use backward sweep locally\n",p);
  PetscPrintf(pc->comm," %spc_sor_local_forward: use forward sweep locally\n",p);
  PetscPrintf(pc->comm," %spc_sor_its its: number of inner SOR iterations to use\n",p);
  return 0;
}

static int PCView_SOR(PetscObject obj,Viewer viewer)
{
  PC         pc = (PC)obj;
  PC_SOR     *jac = (PC_SOR *) pc->data;
  FILE       *fd;
  MatSORType sym = jac->sym;
  char       *sortype;
  int        ierr;
  ViewerType vtype;

  ierr = ViewerGetType(viewer,&vtype); CHKERRQ(ierr);
  if (vtype  == ASCII_FILE_VIEWER || vtype == ASCII_FILES_VIEWER) {  
    ierr = ViewerASCIIGetPointer(viewer,&fd); CHKERRQ(ierr);
    if (sym & SOR_ZERO_INITIAL_GUESS) 
      PetscFPrintf(pc->comm,fd,"    SOR:  zero initial guess\n");
    if (sym == SOR_APPLY_UPPER)              sortype = "apply_upper";
    else if (sym == SOR_APPLY_LOWER)         sortype = "apply_lower";
    else if (sym & SOR_EISENSTAT)            sortype = "Eisenstat";
    else if ((sym & SOR_SYMMETRIC_SWEEP) == SOR_SYMMETRIC_SWEEP)
                                             sortype = "symmetric";
    else if (sym & SOR_BACKWARD_SWEEP)       sortype = "backward";
    else if (sym & SOR_FORWARD_SWEEP)        sortype = "forward";
    else if ((sym & SOR_LOCAL_SYMMETRIC_SWEEP) == SOR_LOCAL_SYMMETRIC_SWEEP)
                                           sortype = "local_symmetric";
    else if (sym & SOR_LOCAL_FORWARD_SWEEP)  sortype = "local_forward";
    else if (sym & SOR_LOCAL_BACKWARD_SWEEP) sortype = "local_backward"; 
    else                                     sortype = "unknown";
    PetscFPrintf(pc->comm,fd,
       "    SOR: type = %s, iterations = %d, omega = %g\n",
       sortype,jac->its,jac->omega);
  }
  return 0;
}

int PCCreate_SOR(PC pc)
{
  PC_SOR *jac   = PetscNew(PC_SOR); CHKPTRQ(jac);
  pc->apply     = PCApply_SOR;
  pc->applyrich = PCApplyRichardson_SOR;
  pc->setfrom   = PCSetFromOptions_SOR;
  pc->printhelp = PCPrintHelp_SOR;
  pc->setup     = 0;
  pc->type      = PCSOR;
  pc->data      = (void *) jac;
  pc->view      = PCView_SOR;
  jac->sym      = SOR_FORWARD_SWEEP;
  jac->omega    = 1.0;
  jac->its      = 1;
  return 0;
}

/*@
   PCSORSetSymmetric - Sets the SOR preconditioner to use symmetric (SSOR), 
   backward, or forward relaxation.  The local variants perform SOR on
   each processor.  By default forward relaxation is used.

   Input Parameters:
.  pc - the preconditioner context
.  flag - one of the following:
$    SOR_FORWARD_SWEEP
$    SOR_BACKWARD_SWEEP
$    SOR_SYMMETRIC_SWEEP
$    SOR_LOCAL_FORWARD_SWEEP
$    SOR_LOCAL_BACKWARD_SWEEP
$    SOR_LOCAL_SYMMETRIC_SWEEP

   Options Database Keys:
$  -pc_sor_symmetric
$  -pc_sor_backward
$  -pc_sor_local_forward
$  -pc_sor_local_symmetric
$  -pc_sor_local_backward

   Notes: 
   To use the Eisenstat trick with SSOR, employ the PCESOR preconditioner,
   which can be chosen with the database option 
$     -pc_type  eisenstat

.keywords: PC, SOR, SSOR, set, relaxation, sweep, forward, backward, symmetric

.seealso: PCEisenstatSetOmega(), PCSORSetIterations(), PCSORSetOmega()
@*/
int PCSORSetSymmetric(PC pc, MatSORType flag)
{
  PC_SOR *jac = (PC_SOR *) pc->data; 
  PetscValidHeaderSpecific(pc,PC_COOKIE);
  if (pc->type != PCSOR) return 0;
  jac->sym = flag;
  return 0;
}
/*@
   PCSORSetOmega - Sets the SOR relaxation coefficient, omega
   (where omega = 1.0 by default).

   Input Parameters:
.  pc - the preconditioner context
.  omega - relaxation coefficient (0 < omega < 2). 

   Options Database Key:
$  -pc_sor_omega  omega

.keywords: PC, SOR, SSOR, set, relaxation, omega

.seealso: PCSORSetSymmetric(), PCSORSetIterations(), PCEisenstatSetOmega()
@*/
int PCSORSetOmega(PC pc, double omega)
{
  PC_SOR *jac = (PC_SOR *) pc->data; 
  PetscValidHeaderSpecific(pc,PC_COOKIE);
  if (pc->type != PCSOR) return 0;
  if (omega >= 2.0 || omega <= 0.0) SETERRQ(1,"PCSORSetOmega:Relaxation out of range");
  jac->omega = omega;
  return 0;
}
/*@
   PCSORSetIterations - Sets the number of inner iterations to 
   be used by the SOR preconditioner. The default is 1.

   Input Parameters:
.  pc - the preconditioner context
.  its - number of iterations to use

   Options Database Key:
$  -pc_sor_its  its

.keywords: PC, SOR, SSOR, set, iterations

.seealso: PCSORSetOmega(), PCSORSetSymmetric()
@*/
int PCSORSetIterations(PC pc, int its)
{
  PC_SOR *jac = (PC_SOR *) pc->data; 
  PetscValidHeaderSpecific(pc,PC_COOKIE);
  if (pc->type != PCSOR) return 0;
  jac->its = its;
  return 0;
}
