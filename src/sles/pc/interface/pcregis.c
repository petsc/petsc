#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: pcregis.c,v 1.42 1998/04/20 19:49:14 curfman Exp curfman $";
#endif

#include "petsc.h"
#include "src/pc/pcimpl.h"          /*I   "pc.h"   I*/

extern int PCCreate_Jacobi(PC);
extern int PCCreate_BJacobi(PC);
extern int PCCreate_ILU(PC);
extern int PCCreate_None(PC);
extern int PCCreate_LU(PC);
extern int PCCreate_SOR(PC);
extern int PCCreate_Shell(PC);
extern int PCCreate_MG(PC);
extern int PCCreate_Eisenstat(PC);
extern int PCCreate_ICC(PC);
extern int PCCreate_ASM(PC);
extern int PCCreate_BGS(PC);
extern int PCCreate_SLES(PC);
extern int PCCreate_Composite(PC);

/*M
   PCRegister - Adds a method to the preconditioner package.

   Synopsis:
   PCRegister(char *name_solver,char *path,char *name_create,int (*routine_create)(PC))

   Input Parameters:
.  name_solver - name of a new user-defined solver
.  path - path (either absolute or relative) the library containing this solver
.  name_create - name of routine to create method context
.  routine_create - routine to create method context

   Notes:
   PCRegister() may be called multiple times to add several user-defined preconditioners.

   If dynamic libraries are used, then the fourth input argument (routine_create)
   is ignored.

   Sample usage:
   PCRegister("my_solver",/home/username/my_lib/lib/libO/solaris/mylib.a,
                "MySolverCreate",MySolverCreate);

   Then, your solver can be chosen with the procedural interface via
$     PCSetType(pc,"my_solver")
   or at runtime via the option
$     -pc_type my_solver

.keywords: PC, register

.seealso: PCRegisterAll(), PCRegisterDestroy()
M*/

#if defined(USE_DYNAMIC_LIBRARIES)
#define PCRegister(a,b,c,d) PCRegister_Private(a,b,c,0)
#else
#define PCRegister(a,b,c,d) PCRegister_Private(a,b,c,d)
#endif

#undef __FUNC__  
#define __FUNC__ "PCRegister_Private"
static int PCRegister_Private(char *sname,char *path,char *name,int (*function)(PC))
{
  int  ierr;
  char fullname[256];

  PetscFunctionBegin;
  PetscStrcpy(fullname,path); PetscStrcat(fullname,":");PetscStrcat(fullname,name);
  ierr = DLRegister(&PCList,sname,fullname,(int (*)(void*))function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCRegisterAll"
/*@C
  PCRegisterAll - Registers all of the preconditioners in the PC package.

  Input Parameter:
.   path - the library where the routines are to be found (optional)

  Not Collective

.keywords: PC, register, all

.seealso: PCRegister(), PCRegisterDestroy()
@*/
int PCRegisterAll(char *path)
{
  int ierr;

  PetscFunctionBegin;
  PCRegisterAllCalled = 1;

  ierr = PCRegister(PCNONE         ,path,"PCCreate_None",PCCreate_None);CHKERRQ(ierr);
  ierr = PCRegister(PCJACOBI       ,path,"PCCreate_Jacobi",PCCreate_Jacobi);CHKERRQ(ierr);
  ierr = PCRegister(PCBJACOBI      ,path,"PCCreate_BJacobi",PCCreate_BJacobi);CHKERRQ(ierr);
  ierr = PCRegister(PCSOR          ,path,"PCCreate_SOR",PCCreate_SOR);CHKERRQ(ierr);
  ierr = PCRegister(PCLU           ,path,"PCCreate_LU",PCCreate_LU);CHKERRQ(ierr);
  ierr = PCRegister(PCSHELL        ,path,"PCCreate_Shell",PCCreate_Shell);CHKERRQ(ierr);
  ierr = PCRegister(PCMG           ,path,"PCCreate_MG",PCCreate_MG);CHKERRQ(ierr);
  ierr = PCRegister(PCEISENSTAT    ,path,"PCCreate_Eisenstat",PCCreate_Eisenstat);CHKERRQ(ierr);
  ierr = PCRegister(PCILU          ,path,"PCCreate_ILU",PCCreate_ILU);CHKERRQ(ierr);
  ierr = PCRegister(PCICC          ,path,"PCCreate_ICC",PCCreate_ICC);CHKERRQ(ierr);
  ierr = PCRegister(PCASM          ,path,"PCCreate_ASM",PCCreate_ASM);CHKERRQ(ierr);
  ierr = PCRegister(PCBGS          ,path,"PCCreate_BGS",PCCreate_BGS);CHKERRQ(ierr);
  ierr = PCRegister(PCSLES         ,path,"PCCreate_SLES",PCCreate_SLES);CHKERRQ(ierr);
  ierr = PCRegister(PCCOMPOSITE    ,path,"PCCreate_Composite",PCCreate_Composite);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


