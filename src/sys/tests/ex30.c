
static char help[] = "Tests nested events.\n\n";

#include <petscsys.h>

int main(int argc,char **argv)
{
  int            event1,event2,event3;

  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  PetscCall(PetscLogEventRegister("Event2",0,&event2));
  PetscCall(PetscLogEventRegister("Event1",0,&event1));
  PetscCall(PetscLogEventRegister("Event3",0,&event3));

  PetscCall(PetscLogEventBegin(event1,0,0,0,0));
  PetscCall(PetscSleep(1.0));
  PetscCall(PetscLogEventBegin(event2,0,0,0,0));
  PetscCall(PetscSleep(1.0));
  PetscCall(PetscLogEventBegin(event3,0,0,0,0));
  PetscCall(PetscSleep(1.0));
  PetscCall(PetscLogEventEnd(event3,0,0,0,0));
  PetscCall(PetscLogEventEnd(event2,0,0,0,0));
  PetscCall(PetscLogEventEnd(event1,0,0,0,0));
  PetscCall(PetscFinalize());
  return 0;
}
