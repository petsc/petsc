#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: mmmethods.c,v 1.5 1998/05/13 20:02:16 curfman Exp curfman $";
#endif

#include "mmimpl.h"

typedef struct {
  void *stuff;
} MM_Euler;

/* Euler model only */
#undef __FUNC__  
#define __FUNC__ "MMCreate_Euler"
int MMCreate_Euler(MM mm)
{
  MM_Euler *euler = PetscNew(MM_Euler); CHKPTRQ(euler);

  PetscFunctionBegin;
  PetscMemzero(euler,sizeof(MM_Euler));
  PLogObjectMemory(mm,sizeof(MM_Euler));

  euler->stuff       = 0;
  mm->data           = (void *) euler;
  mm->ncomponents    = 5;
  mm->setupcalled    = 0;
  mm->printhelp      = 0;
  mm->setfromoptions = 0;

  PetscFunctionReturn(0);
}

/* ------------------------------------------------------- */
/* these should really be in separate files/directories */
/* ------------------------------------------------------- */

typedef struct {
  void *stuff;
} MM_Hybrid_EF;

/* Hybrid Euler and full potential */
#undef __FUNC__  
#define __FUNC__ "MMCreate_Hybrid_EF"
int MMCreate_Hybrid_EF1(MM mm)
{
  int flg, ierr;
  MM_Hybrid_EF *hybrid = PetscNew(MM_Hybrid_EF); CHKPTRQ(hybrid);

  PetscFunctionBegin;
  PetscMemzero(mm,sizeof(MM_Hybrid_EF));
  PLogObjectMemory(mm,sizeof(MM_Hybrid_EF));

  hybrid->stuff      = 0;
  mm->data           = (void *) hybrid;
  mm->ncomponents    = 6;
  mm->setupcalled    = 0;
  mm->printhelp      = 0;
  mm->setfromoptions = 0;

  /*
  mm->type          = MMHYBRID_EF1;
  ierr = OptionsHasName(PETSC_NULL,"-mm_hybrid_euler",&flg); CHKERRQ(ierr);
  if (flg) mm->type = MMHYBRID_E;
  ierr = OptionsHasName(PETSC_NULL,"-mm_hybrid_fp",&flg); CHKERRQ(ierr);
  if (flg) mm->type = MMHYBRID_F;
  */
  PetscFunctionReturn(0);
}

typedef struct {
  void *stuff;
} MM_FullPotential;

/* Full potential */
#undef __FUNC__  
#define __FUNC__ "MMCreate_Fp_EF"
int MMCreate_FullPotential(MM mm)
{
  MM_FullPotential *fp = PetscNew(MM_FullPotential); CHKPTRQ(fp);

  PetscFunctionBegin;
  PetscMemzero(mm,sizeof(MM_FullPotential));
  PLogObjectMemory(mm,sizeof(MM_FullPotential));

  fp->stuff          = 0;
  mm->data           = (void *) fp;
  mm->ncomponents    = 1;
  mm->setupcalled    = 0;
  mm->printhelp      = 0;
  mm->setfromoptions = 0;

  PetscFunctionReturn(0);
}
