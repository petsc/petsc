#ifndef lint
static char vcid[] = "$Id: sor.c,v 1.12 1995/04/16 03:43:21 curfman Exp bsmith $";
#endif

/*
   Defines a  (S)SOR  preconditioner for any Mat implementation
*/
#include "pcimpl.h"
#include "options.h"

typedef struct {
  int    its,sym;
  double omega;
} PC_SOR;

static int PCApply_SOR(PC pc,Vec x,Vec y)
{
  PC_SOR *jac = (PC_SOR *) pc->data;
  int    ierr, flag = jac->sym | SOR_ZERO_INITIAL_GUESS;
  if ((ierr = MatRelax(pc->mat,x,jac->omega,flag,0.0,jac->its,y))) return ierr;
  return 0;
}

static int PCApplyRichardson_SOR(PC pc,Vec b,Vec y,Vec w,int its)
{
  PC_SOR *jac = (PC_SOR *) pc->data;
  int    ierr, flag;
  flag = jac->sym;
  if ((ierr = MatRelax(pc->mat,b,jac->omega,flag,0.0,its,y))) return ierr;
  return 0;
}

/* parses arguments of the form -sor [symmetric,forward,back][omega=...] */
static int PCSetFromOptions_SOR(PC pc)
{
  int    its;
  double omega;

  if (OptionsGetDouble(0,pc->prefix,"-sor_omega",&omega)) {
    PCSORSetOmega(pc,omega);
  } 
  if (OptionsGetInt(0,pc->prefix,"-sor_its",&its)) {
    PCSORSetIterations(pc,its);
  }
  if (OptionsHasName(0,pc->prefix,"-sor_symmetric")) {
    PCSORSetSymmetric(pc,SOR_SYMMETRIC_SWEEP);
  }
  if (OptionsHasName(0,pc->prefix,"-sor_backward")) {
    PCSORSetSymmetric(pc,SOR_BACKWARD_SWEEP);
  }
  if (OptionsHasName(0,pc->prefix,"-sor_local_symmetric")) {
    PCSORSetSymmetric(pc,SOR_LOCAL_SYMMETRIC_SWEEP);
  }
  if (OptionsHasName(0,pc->prefix,"-sor_local_backward")) {
    PCSORSetSymmetric(pc,SOR_LOCAL_BACKWARD_SWEEP);
  }
  if (OptionsHasName(0,pc->prefix,"-sor_local_forward")) {
    PCSORSetSymmetric(pc,SOR_LOCAL_FORWARD_SWEEP);
  }
  return 0;
}

int PCPrintHelp_SOR(PC pc)
{
  char *p;
  if (pc->prefix) p = pc->prefix; else p = "-";
  fprintf(stderr," %ssor_omega omega: relaxation factor (0 < omega < 2)\n",p);
  fprintf(stderr," %ssor_symmetric: use SSOR\n",p);
  fprintf(stderr," %ssor_backward: use backward sweep instead of forward\n",p);
  fprintf(stderr," %ssor_local_symmetric: use SSOR on each processor\n",p);
  fprintf(stderr," %ssor_local_backward: use backward sweep locally\n",p);
  fprintf(stderr," %ssor_local_forward: use forward sweep locally\n",p);
  fprintf(stderr," %ssor_its its: number of inner SOR iterations to use\n",p);
  return 0;
}
int PCCreate_SOR(PC pc)
{
  PC_SOR *jac   = NEW(PC_SOR); CHKPTR(jac);
  pc->apply     = PCApply_SOR;
  pc->applyrich = PCApplyRichardson_SOR;
  pc->setfrom   = PCSetFromOptions_SOR;
  pc->printhelp = PCPrintHelp_SOR;
  pc->setup     = 0;
  pc->type      = PCSOR;
  pc->data      = (void *) jac;
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
$      SOR_FORWARD_SWEEP
$      SOR_SYMMETRIC_SWEEP
$      SOR_BACKWARD_SWEEP
$      SOR_LOCAL_FORWARD_SWEEP
$      SOR_LOCAL_SYMMETRIC_SWEEP
$      SOR_LOCAL_BACKWARD_SWEEP

   Options Database Keys:
$  -sor_symmetric
$  -sor_backward
$  -sor_local_forward
$  -sor_local_symmetric
$  -sor_local_backward

   Notes: 
   To use the Eisenstat trick with SSOR, employ the PCESOR preconditioner,
   which can be chosen with the database option 
$     -pc_method eisenstat

.keywords: PC, SOR, SSOR, set, relaxation, sweep, forward, backward, symmetric

.seealso: PCEisenstatSetOmega(), PCSORSetOmega(), PCSORSetIterations()
@*/
int PCSORSetSymmetric(PC pc, int flag)
{
  PC_SOR *jac = (PC_SOR *) pc->data; 
  VALIDHEADER(pc,PC_COOKIE);
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
$  -sor_omega  omega

.keywords: PC, SOR, SSOR, set, relaxation, omega

.seealso: PCSORSetSymmetric(), PCSORSetIterations(), PCEisenstatSetOmega()
@*/
int PCSORSetOmega(PC pc, double omega)
{
  PC_SOR *jac = (PC_SOR *) pc->data; 
  VALIDHEADER(pc,PC_COOKIE);
  if (omega >= 2.0 || omega <= 0.0) { SETERR(1,"Relaxation out of range");}
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
$  -sor_its  its

.keywords: PC, SOR, SSOR, set, iterations

.seealso: PCSORSetOmega(), PCSORSetSymmetric()
@*/
int PCSORSetIterations(PC pc, int its)
{
  PC_SOR *jac = (PC_SOR *) pc->data; 
  VALIDHEADER(pc,PC_COOKIE);
  jac->its = its;
  return 0;
}
