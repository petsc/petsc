#ifndef lint
static char vcid[] = "$Id: eisen.c,v 1.5 1995/03/10 04:44:35 bsmith Exp bsmith $";
#endif

/*
   Defines a  Eisenstat trick SSOR  preconditioner. This uses about 
 %50 of the usual amount of floating point ops used for SSOR + Krylov 
 method. But it requires actually solving the preconditioned problem 
 with both left and right preconditioning. 

*/
#include "pcimpl.h"
#include "options.h"

typedef struct {
  Mat    shell,A;
  Vec    b;
  double omega;
} PCiESOR;

static int PCiESORmult(void *ptr,Vec b,Vec x)
{
  PC      pc = (PC) ptr;
  PCiESOR *jac = (PCiESOR *) pc->data;
  return MatRelax(jac->A,b,jac->omega,SOR_EISENSTAT,0.0,1,x);
}

static int PCiNoneApply(PC ptr,Vec x,Vec y)
{
  return VecCopy(x,y);
}

/* this cheats and looks inside KSP to determine if nonzero initial guess*/
#include "src/ksp/kspimpl.h"

static int PCiPre(PC pc,KSP ksp)
{
  PCiESOR *jac = (PCiESOR *) pc->data;
  Vec     b,x;
  int     ierr;

  if (pc->mat != pc->pmat) {
    SETERR(1,"Eisenstat preconditioner cannot have different mat from pmat"); 
  }
 
  /* swap shell matrix and true matrix */
  jac->A    = pc->mat;
  pc->mat   = jac->shell;

  KSPGetRhs(ksp,&b);
  if (!jac->b) {
    ierr = VecCreate(b,&jac->b); CHKERR(ierr);
  }
  
  /* save true b, other option is to swap pointers */
  ierr = VecCopy(b,jac->b); CHKERR(ierr);

  /* if nonzero initial guess, modify x */
  if (!ksp->guess_zero) {
    KSPGetSolution(ksp,&x);
    ierr = MatRelax(jac->A,x,jac->omega,SOR_APPLY_UPPER,0.0,1,x); CHKERR(ierr);
  }

  /* modify b by (L + D)^{-1} */
  ierr =   MatRelax(jac->A,b,jac->omega,SOR_ZERO_INITIAL_GUESS | 
                                        SOR_FORWARD_SWEEP,0.0,1,b); 
  CHKERR(ierr);  
  return 0;
}

static int PCiPost(PC pc,KSP ksp)
{
  PCiESOR *jac = (PCiESOR *) pc->data;
  Vec     x,b;
  int     ierr;
  KSPGetSolution(ksp,&x);
  ierr =   MatRelax(jac->A,x,jac->omega,SOR_ZERO_INITIAL_GUESS | 
                                 SOR_BACKWARD_SWEEP,0.0,1,x); CHKERR(ierr);
  pc->mat = jac->A;
  /* get back true b */
  KSPGetRhs(ksp,&b);
  VecCopy(jac->b,b);
  return 0;
}

int PCiESORDestroy(PetscObject obj)
{
  PC       pc = (PC) obj;
  PCiESOR  *jac = ( PCiESOR  *) pc->data; 
  if (jac->b) VecDestroy(jac->b);
  if (jac->shell) MatDestroy(jac->shell);
  FREE(jac);
  PETSCHEADERDESTROY(pc);
  return 0;
}

static int PCisetfrom(PC pc)
{
  double  omega;

  if (OptionsGetDouble(0,pc->prefix,"-sor_omega",&omega)) {
    PCESORSetOmega(pc,omega);
  }
  return 0;
}

static int PCiprinthelp(PC pc)
{
  char *p;
  if (pc->prefix) p = pc->prefix; else p = "-";
  fprintf(stderr,"%ssor_omega omega: relaxation factor. 0 < omega <2\n",p);
  return 0;
}
int PCiESORCreate(PC pc)
{
  int      ierr;
  PCiESOR  *jac;
  jac           = NEW(PCiESOR); CHKPTR(jac);
  pc->apply     = PCiNoneApply;
  pc->presolve  = PCiPre;
  pc->postsolve = PCiPost;
  pc->applyrich = 0;
  pc->setfrom   = PCisetfrom;
  pc->printhelp = PCiprinthelp ;
  pc->destroy   = PCiESORDestroy;
  pc->type      = PCESOR;
  pc->data      = (void *) jac;
  pc->setup     = 0;
  jac->omega    = 1.0;
  jac->b        = 0;
  ierr = MatShellCreate(0,0,(void*) pc,&jac->shell); CHKERR(ierr);
  ierr = MatShellSetMult(jac->shell, PCiESORmult); CHKERR(ierr);
  return 0;
}

/*@ 
      PCESORSetOmega - Sets relaxation factor to use with SSOR using 
                       Eisenstat's trick.

  Input Parameters:
.  pc - the preconditioner context
.  omega - relaxation factor between 0 and 2.

@*/
int PCESORSetOmega(PC pc,double omega)
{
  PCiESOR  *jac;
  VALIDHEADER(pc,PC_COOKIE);
  if (pc->type != PCESOR) return 0;
  jac = (PCiESOR *) pc->data;
  jac->omega = omega;
  return 0;
}


