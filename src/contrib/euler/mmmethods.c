#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex1.c,v 1.63 1997/09/22 15:21:33 balay Exp $";
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
  PLogObjectMemory(mm,sizeof(MM_Euler));

  euler->stuff    = 0;
  mm->data        = (void *) euler;
  mm->ncomponents = 5;
  mm->setupcalled = 0;
  mm->printhelp   = 0;
  mm->setfrom     = 0;
  mm->type        = MMEULER;

  return 0;
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
  PLogObjectMemory(mm,sizeof(MM_Hybrid_EF));

  hybrid->stuff     = 0;
  mm->data          = (void *) hybrid;
  mm->ncomponents   = 6;
  mm->setupcalled   = 0;
  mm->printhelp     = 0;
  mm->setfrom       = 0;

  mm->type          = MMHYBRID_EF1;
  ierr = OptionsHasName(PETSC_NULL,"-mm_hybrid_euler",&flg); CHKERRQ(ierr);
  if (flg) mm->type = MMHYBRID_E;
  ierr = OptionsHasName(PETSC_NULL,"-mm_hybrid_fp",&flg); CHKERRQ(ierr);
  if (flg) mm->type = MMHYBRID_F;
  return 0;
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
  PLogObjectMemory(mm,sizeof(MM_FullPotential));

  fp->stuff   = 0;
  mm->data        = (void *) fp;
  mm->ncomponents = 1;
  mm->setupcalled = 0;
  mm->printhelp   = 0;
  mm->setfrom     = 0;
  mm->type        = MMFP;
  return 0;
}
