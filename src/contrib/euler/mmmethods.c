
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
  MM_Hybrid_EF *hybrid = PetscNew(MM_Hybrid_EF); CHKPTRQ(hybrid);
  PLogObjectMemory(mm,sizeof(MM_Hybrid_EF));

  hybrid->stuff   = 0;
  mm->data        = (void *) hybrid;
  mm->ncomponents = 6;
  mm->setupcalled = 0;
  mm->printhelp   = 0;
  mm->setfrom     = 0;
  return 0;
}
