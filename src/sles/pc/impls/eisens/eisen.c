#ifndef lint
static char vcid[] = "$Id: eisen.c,v 1.22 1995/07/07 16:16:10 bsmith Exp bsmith $";
#endif

/*
   Defines a  Eisenstat trick SSOR  preconditioner. This uses about 
 %50 of the usual amount of floating point ops used for SSOR + Krylov 
 method. But it requires actually solving the preconditioned problem 
 with both left and right preconditioning. 

*/
#include "pcimpl.h"

typedef struct {
  Mat    shell,A;
  Vec    b;
  double omega;
} PC_Eisenstat;

static int PCMult_Eisenstat(void *ptr,Vec b,Vec x)
{
  PC      pc = (PC) ptr;
  PC_Eisenstat *jac = (PC_Eisenstat *) pc->data;
  return MatRelax(jac->A,b,jac->omega,SOR_EISENSTAT,0.0,1,x);
}

static int PCApply_Eisenstat(PC ptr,Vec x,Vec y)
{
  return VecCopy(x,y);
}

/* this cheats and looks inside KSP to determine if nonzero initial guess*/
#include "src/ksp/kspimpl.h"

static int PCPre_Eisenstat(PC pc,KSP ksp)
{
  PC_Eisenstat *jac = (PC_Eisenstat *) pc->data;
  Vec     b,x;
  int     ierr;

  if (pc->mat != pc->pmat) {
    SETERRQ(1,"PCPre_Eisenstat: cannot have different mat from pmat"); 
  }
 
  /* swap shell matrix and true matrix */
  jac->A    = pc->mat;
  pc->mat   = jac->shell;

  KSPGetRhs(ksp,&b);
  if (!jac->b) {
    ierr = VecDuplicate(b,&jac->b); CHKERRQ(ierr);
    PLogObjectParent(pc,jac->b);
  }
  
  /* save true b, other option is to swap pointers */
  ierr = VecCopy(b,jac->b); CHKERRQ(ierr);

  /* if nonzero initial guess, modify x */
  if (!ksp->guess_zero) {
    KSPGetSolution(ksp,&x);
    ierr = MatRelax(jac->A,x,jac->omega,SOR_APPLY_UPPER,0.0,1,x); CHKERRQ(ierr);
  }

  /* modify b by (L + D)^{-1} */
  ierr =   MatRelax(jac->A,b,jac->omega,(MatSORType)(SOR_ZERO_INITIAL_GUESS | 
                                        SOR_FORWARD_SWEEP),0.0,1,b); 
  CHKERRQ(ierr);  
  return 0;
}

static int PCPost_Eisenstat(PC pc,KSP ksp)
{
  PC_Eisenstat *jac = (PC_Eisenstat *) pc->data;
  Vec     x,b;
  int     ierr;
  KSPGetSolution(ksp,&x);
  ierr =   MatRelax(jac->A,x,jac->omega,(MatSORType)(SOR_ZERO_INITIAL_GUESS | 
                                 SOR_BACKWARD_SWEEP),0.0,1,x); CHKERRQ(ierr);
  pc->mat = jac->A;
  /* get back true b */
  KSPGetRhs(ksp,&b);
  VecCopy(jac->b,b);
  return 0;
}

int PCDestroy_Eisenstat(PetscObject obj)
{
  PC       pc = (PC) obj;
  PC_Eisenstat  *jac = ( PC_Eisenstat  *) pc->data; 
  if (jac->b) VecDestroy(jac->b);
  if (jac->shell) MatDestroy(jac->shell);
  PETSCFREE(jac);
  return 0;
}

static int PCSetFrom_Eisenstat(PC pc)
{
  double  omega;

  if (OptionsGetDouble(pc->prefix,"-pc_sor_omega",&omega)) {
    PCEisenstatSetOmega(pc,omega);
  }
  return 0;
}

static int PCPrintHelp_Eisenstat(PC pc)
{
  char *p;
  if (pc->prefix) p = pc->prefix; else p = "-";
  MPIU_print(pc->comm," %spc_sor_omega omega: relaxation factor (0 < omega < 2)\n",p);
  return 0;
}
int PCCreate_Eisenstat(PC pc)
{
  int      ierr;
  PC_Eisenstat  *jac;
  jac           = PETSCNEW(PC_Eisenstat); CHKPTRQ(jac);
  pc->apply     = PCApply_Eisenstat;
  pc->presolve  = PCPre_Eisenstat;
  pc->postsolve = PCPost_Eisenstat;
  pc->applyrich = 0;
  pc->setfrom   = PCSetFrom_Eisenstat;
  pc->printhelp = PCPrintHelp_Eisenstat ;
  pc->destroy   = PCDestroy_Eisenstat;
  pc->type      = PCESOR;
  pc->data      = (void *) jac;
  pc->setup     = 0;
  jac->omega    = 1.0;
  jac->b        = 0;
  ierr = MatShellCreate(pc->comm,0,0,(void*) pc,&jac->shell); CHKERRQ(ierr);
  ierr = MatShellSetMult(jac->shell, PCMult_Eisenstat); CHKERRQ(ierr);
  return 0;
}

/*@ 
   PCEisenstatSetOmega - Sets the SSOR relaxation coefficient, omega,
   to use with Eisenstat's trick (where omega = 1.0 by default).

   Input Parameters:
.  pc - the preconditioner context
.  omega - relaxation coefficient (0 < omega < 2)

   Options Database Key:
$  -pc_sor_omega  omega

   Notes: 
   The Eisenstat trick implementation of SSOR requires about 50% of the
   usual amount of floating point operations used for SSOR + Krylov method;
   however, the preconditioned problem must be solved with both left 
   and right preconditioning.

   To use SSOR without the Eisenstat trick, employ the PCSOR preconditioner, 
   which can be chosen with the database options
$    -pc_method  sor  -pc_sor_symmetric

.keywords: PC, Eisenstat, set, SOR, SSOR, relaxation, omega

.seealso: PCSORSetOmega()
@*/
int PCEisenstatSetOmega(PC pc,double omega)
{
  PC_Eisenstat  *jac;
  VALIDHEADER(pc,PC_COOKIE);
  if (pc->type != PCESOR) return 0;
  jac = (PC_Eisenstat *) pc->data;
  jac->omega = omega;
  return 0;
}


