/*$Id: ex4.c,v 1.9 2001/03/23 23:20:59 balay Exp $*/

static char help[] = "Tests PetscStrtok().\n\n";

#include "petsc.h"

#undef __FUNCT__
#define __FUNCT__ "main"
/*
       The system command strtok() is broken under some versions of linux.
*/
int main(int argc,char **argv)
{
  char *string = "Greetings, Linux just come to check your\nstrtok(). Hah";
  char *sub;
  int  ierr;

  PetscInitialize(&argc,&argv,(char *)0,help);

  ierr = PetscStrtok(string," ",&sub);CHKERRQ(ierr);
  printf("%s\n",sub);
  ierr = PetscStrtok(0," ",&sub);CHKERRQ(ierr);
  printf("%s\n",sub);
  ierr = PetscStrtok(0," ",&sub);CHKERRQ(ierr);
  printf("%s\n",sub);
  ierr = PetscStrtok(0," ",&sub);CHKERRQ(ierr);
  printf("%s\n",sub);
  ierr = PetscStrtok(0," ",&sub);CHKERRQ(ierr);
  printf("%s\n",sub);
  ierr = PetscStrtok(0," ",&sub);CHKERRQ(ierr);
  printf("%s\n",sub);
  ierr = PetscStrtok(0,"\n",&sub);CHKERRQ(ierr);
  printf("%s\n",sub);
  ierr = PetscStrtok(0," ",&sub);CHKERRQ(ierr);
  printf("%s\n",sub);
  ierr = PetscStrtok(0," ",&sub);CHKERRQ(ierr);
  printf("%s\n",sub);

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
 
