
static char help[] = "Tests nested events.\n\n";

#include <petscsys.h>

int main(int argc,char **argv)
{
  int            event1,event2,event3;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  CHKERRQ(PetscLogEventRegister("Event2",0,&event2));
  CHKERRQ(PetscLogEventRegister("Event1",0,&event1));
  CHKERRQ(PetscLogEventRegister("Event3",0,&event3));

  CHKERRQ(PetscLogEventBegin(event1,0,0,0,0));
  CHKERRQ(PetscSleep(1.0));
  CHKERRQ(PetscLogEventBegin(event2,0,0,0,0));
  CHKERRQ(PetscSleep(1.0));
  CHKERRQ(PetscLogEventBegin(event3,0,0,0,0));
  CHKERRQ(PetscSleep(1.0));
  CHKERRQ(PetscLogEventEnd(event3,0,0,0,0));
  CHKERRQ(PetscLogEventEnd(event2,0,0,0,0));
  CHKERRQ(PetscLogEventEnd(event1,0,0,0,0));
  ierr = PetscFinalize();
  return ierr;
}
