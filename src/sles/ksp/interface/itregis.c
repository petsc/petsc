#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: itregis.c,v 1.29 1998/04/20 19:37:53 curfman Exp curfman $";
#endif

#include "src/ksp/kspimpl.h"  /*I "ksp.h" I*/

extern int KSPCreate_Richardson(KSP);
extern int KSPCreate_Chebychev(KSP);
extern int KSPCreate_CG(KSP);
extern int KSPCreate_TCQMR(KSP);
extern int KSPCreate_GMRES(KSP);
extern int KSPCreate_BCGS(KSP);
extern int KSPCreate_CGS(KSP);
extern int KSPCreate_TFQMR(KSP);
extern int KSPCreate_LSQR(KSP);
extern int KSPCreate_PREONLY(KSP);
extern int KSPCreate_CR(KSP);
extern int KSPCreate_QCG(KSP);

/*M
   KSPRegister - Adds a method to the Krylov subspace solver package.

   Synopsis:
   KSPRegister(char *name_solver,char *path,char *name_create,int (*routine_create)(KSP))

   Input Parameters:
.  name_solver - name of a new user-defined solver
.  path - path (either absolute or relative) the library containing this solver
.  name_create - name of routine to create method context
.  routine_create - routine to create method context

   Notes:
   KSPRegister() may be called multiple times to add several user-defined solvers.

   If dynamic libraries are used, then the fourth input argument (routine_create)
   is ignored.

   Sample usage:
   KSPRegister("my_solver",/home/username/my_lib/lib/libO/solaris/mylib.a,
                "MySolverCreate",MySolverCreate);

   Then, your solver can be chosen with the procedural interface via
$     KSPSetType(ksp,"my_solver")
$   or at runtime via the option
$     -ksp_type my_solver

.keywords: KSP, register

.seealso: KSPRegisterAll(), KSPRegisterDestroy()
M*/

#if defined(USE_DYNAMIC_LIBRARIES)
#define KSPRegister(a,b,c,d) KSPRegister_Private(a,b,c,0)
#else
#define KSPRegister(a,b,c,d) KSPRegister_Private(a,b,c,d)
#endif

#undef __FUNC__  
#define __FUNC__ "KSPRegister_Private"
static int KSPRegister_Private(char *sname,char *path,char *name,int (*function)(KSP))
{
  int ierr;
  char fullname[256];

  PetscFunctionBegin;
  PetscStrcpy(fullname,path); PetscStrcat(fullname,":");PetscStrcat(fullname,name);
  ierr = DLRegister(&KSPList,sname,fullname,(int (*)(void*))function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

  
/*
      This is used by KSPSetType() to make sure that at least one 
    KSPRegisterAll() is called. In general, if there is more than one
    DLL then KSPRegisterAll() may be called several times.
*/
extern int KSPRegisterAllCalled;

#undef __FUNC__  
#define __FUNC__ "KSPRegisterAll"
/*@C
  KSPRegisterAll - Registers all of the Krylov subspace methods in the KSP package.

   Note Collective

.keywords: KSP, register, all

.seealso:  KSPRegisterDestroy()
@*/
int KSPRegisterAll(char *path)
{
  int ierr;

  PetscFunctionBegin;
  KSPRegisterAllCalled = 1;

  ierr = KSPRegister(KSPCG,         path,"KSPCreate_CG",        KSPCreate_CG);CHKERRQ(ierr);
  ierr = KSPRegister(KSPRICHARDSON, path,"KSPCreate_Richardson",KSPCreate_Richardson);CHKERRQ(ierr);
  ierr = KSPRegister(KSPCHEBYCHEV,  path,"KSPCreate_Chebychev", KSPCreate_Chebychev);CHKERRQ(ierr);
  ierr = KSPRegister(KSPGMRES,      path,"KSPCreate_GMRES",     KSPCreate_GMRES);CHKERRQ(ierr);
  ierr = KSPRegister(KSPTCQMR,      path,"KSPCreate_TCQMR",     KSPCreate_TCQMR);CHKERRQ(ierr);
  ierr = KSPRegister(KSPBCGS,       path,"KSPCreate_BCGS",      KSPCreate_BCGS);CHKERRQ(ierr);
  ierr = KSPRegister(KSPCGS,        path,"KSPCreate_CGS",       KSPCreate_CGS);CHKERRQ(ierr);
  ierr = KSPRegister(KSPTFQMR,      path,"KSPCreate_TFQMR",     KSPCreate_TFQMR);CHKERRQ(ierr);
  ierr = KSPRegister(KSPCR,         path,"KSPCreate_CR",        KSPCreate_CR);CHKERRQ(ierr);
  ierr = KSPRegister(KSPLSQR,       path,"KSPCreate_LSQR",      KSPCreate_LSQR);CHKERRQ(ierr);
  ierr = KSPRegister(KSPPREONLY,    path,"KSPCreate_PREONLY",   KSPCreate_PREONLY);CHKERRQ(ierr);
  ierr = KSPRegister(KSPQCG,        path,"KSPCreate_QCG",       KSPCreate_QCG);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
