
/*
   Defines a  (S)SOR  preconditioner for any Mat implementation
*/
#include "pcimpl.h"
#include "options.h"

typedef struct {
  int    sym;
  double omega;
} PCiSOR;

static int PCiSORApply(PC pc,Vec x,Vec y)
{
  PCiSOR *jac = (PCiSOR *) pc->data;
  int    ierr, flag = jac->sym | SOR_ZERO_INITIAL_GUESS;
  if (ierr = MatRelax(pc->mat,x,jac->omega,flag,0,1,y)) return ierr;
  return 0;
}

static int PCiSORApplyrich(PC pc,Vec b,Vec y,Vec w,int its)
{
  PCiSOR *jac = (PCiSOR *) pc->data;
  int    ierr, flag;
  flag = jac->sym;
  if (ierr = MatRelax(pc->mat,b,jac->omega,flag,0,its,y)) return ierr;
  return 0;
}

/* parses arguments of the form -sor [symmetric,forward,back][omega=...] */
static int PCisetfrom(PC pc)
{
  PCiSOR *jac = (PCiSOR *) pc->data;
  double omega;

  if (OptionsGetDouble(0,"-sor_omega",&omega)) {
    PCSORSetOmega(pc,omega);
  }
  if (OptionsHasName(0,"-sor_symmetric")) {
    PCSORSetSymmetric(pc,SOR_SYMMETRIC_SWEEP);
  }
  if (OptionsHasName(0,"-sor_backward")) {
    PCSORSetSymmetric(pc,SOR_BACKWARD_SWEEP);
  }
  return 0;
}

int PCiSORprinthelp(PC pc)
{
  fprintf(stderr,"-sor_omega omega: relaxation factor. 0 < omega <2\n");
  fprintf(stderr,"-sor_symmetric: use SSOR\n");
  fprintf(stderr,"-sor_backward: use backward sweep instead of forward\n");
  return 0;
}
int PCiSORCreate(PC pc)
{
  PCiSOR *jac   = NEW(PCiSOR); CHKPTR(jac);
  pc->apply     = PCiSORApply;
  pc->applyrich = PCiSORApplyrich;
  pc->setfrom   = PCisetfrom;
  pc->printhelp = PCiSORprinthelp;
  pc->setup     = 0;
  pc->type      = PCSOR;
  pc->data      = (void *) jac;
  jac->sym      = SOR_FORWARD_SWEEP;
  jac->omega    = 1.0;
  return 0;
}

/*@
     PCSORSetSymmetric - Sets the SOR preconditioner to use SSOR, or 
       backward, or forward relaxation. By default it uses forward.

  Input Parameters:
.   pc - the preconditioner context
.   flag - one of SOR_FORWARD_SWEEP, SOR_SYMMETRIC_SWEEP,SOR_BACKWARD_SWEEP 
@*/
int PCSORSetSymmetric(PC pc, int flag)
{
  PCiSOR *jac = (PCiSOR *) pc->data; 
  VALIDHEADER(pc,PC_COOKIE);
  jac->sym = flag;
  return 0;
}
/*@
     PCSORSetOmega - Sets the SOR relaxation coefficient. By default
         uses 1.0;

  Input Parameters:
.   pc - the preconditioner context
.   omega - relaxation coefficient, 0 < omega < 2. 
@*/
int PCSORSetOmega(PC pc, double omega)
{
  PCiSOR *jac = (PCiSOR *) pc->data; 
  VALIDHEADER(pc,PC_COOKIE);
  if (omega >= 2.0 || omega <= 0.0) { SETERR(1,"Relaxation out of range");}
  jac->omega = omega;
  return 0;
}
