
static char help[] = "Tests nested events.\n\n";

#include <petscsys.h>

int main(int argc,char **argv)
{
  int            event1,event2,event3;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = PetscLogEventRegister("Event2",0,&event2);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("Event1",0,&event1);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("Event3",0,&event3);CHKERRQ(ierr);

  ierr = PetscLogEventBegin(event1,0,0,0,0);CHKERRQ(ierr);
  ierr = PetscSleep(1.0);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(event2,0,0,0,0);CHKERRQ(ierr);
  ierr = PetscSleep(1.0);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(event3,0,0,0,0);CHKERRQ(ierr);
  ierr = PetscSleep(1.0);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(event3,0,0,0,0);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(event2,0,0,0,0);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(event1,0,0,0,0);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

