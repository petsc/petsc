
/*
     This defines part of the private API for logging performance information. It is intended to be used only by the
   PETSc PetscLog...() interface and not elsewhere, nor by users. Hence the prototypes for these functions are NOT
   in the public PETSc include files.

*/
#include <petsc-private/logimpl.h> /*I    "petscsys.h"   I*/

#undef __FUNCT__  
#define __FUNCT__ "PetscIntStackDestroy"
/*@C
  PetscIntStackDestroy - This function destroys a stack.

  Not Collective

  Input Parameter:
. stack - The stack

  Level: developer

.keywords: log, stack, destroy
.seealso: PetscIntStackCreate(), PetscIntStackEmpty(), PetscIntStackPush(), PetscIntStackPop(), PetscIntStackTop()
@*/
PetscErrorCode PetscIntStackDestroy(PetscIntStack stack)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(stack->stack);CHKERRQ(ierr);
  ierr = PetscFree(stack);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscIntStackEmpty"
/*@C
  PetscIntStackEmpty - This function determines whether any items have been pushed.

  Not Collective

  Input Parameter:
. stack - The stack

  Output Parameter:
. empty - PETSC_TRUE if the stack is empty

  Level: developer

.keywords: log, stack, empty
.seealso: PetscIntStackCreate(), PetscIntStackDestroy(), PetscIntStackPush(), PetscIntStackPop(), PetscIntStackTop()
@*/
PetscErrorCode PetscIntStackEmpty(PetscIntStack stack, PetscBool  *empty)
{
  PetscFunctionBegin;
  PetscValidIntPointer(empty,2);
  if (stack->top == -1) {
    *empty = PETSC_TRUE;
  } else {
    *empty = PETSC_FALSE;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscIntStackTop"
/*@C
  PetscIntStackTop - This function returns the top of the stack.

  Not Collective

  Input Parameter:
. stack - The stack

  Output Parameter:
. top - The integer on top of the stack

  Level: developer

.keywords: log, stack, top
.seealso: PetscIntStackCreate(), PetscIntStackDestroy(), PetscIntStackEmpty(), PetscIntStackPush(), PetscIntStackPop()
@*/
PetscErrorCode PetscIntStackTop(PetscIntStack stack, int *top)
{
  PetscFunctionBegin;
  PetscValidIntPointer(top,2);
  *top = stack->stack[stack->top];
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscIntStackPush"
/*@C
  PetscIntStackPush - This function pushes an integer on the stack.

  Not Collective

  Input Parameters:
+ stack - The stack
- item  - The integer to push

  Level: developer

.keywords: log, stack, push
.seealso: PetscIntStackCreate(), PetscIntStackDestroy(), PetscIntStackEmpty(), PetscIntStackPop(), PetscIntStackTop()
@*/
PetscErrorCode PetscIntStackPush(PetscIntStack stack, int item)
{
  int            *array;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  stack->top++;
  if (stack->top >= stack->max) {
    ierr = PetscMalloc(stack->max*2 * sizeof(int), &array);CHKERRQ(ierr);
    ierr = PetscMemcpy(array, stack->stack, stack->max * sizeof(int));CHKERRQ(ierr);
    ierr = PetscFree(stack->stack);CHKERRQ(ierr);
    stack->stack = array;
    stack->max  *= 2;
  }
  stack->stack[stack->top] = item;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscIntStackPop"
/*@C
  PetscIntStackPop - This function pops an integer from the stack.

  Not Collective

  Input Parameter:
. stack - The stack

  Output Parameter:
. item  - The integer popped

  Level: developer

.keywords: log, stack, pop
.seealso: PetscIntStackCreate(), PetscIntStackDestroy(), PetscIntStackEmpty(), PetscIntStackPush(), PetscIntStackTop()
@*/
PetscErrorCode PetscIntStackPop(PetscIntStack stack, int *item)
{
  PetscFunctionBegin;
  PetscValidPointer(item,2);
  if (stack->top == -1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE, "Stack is empty");
  *item = stack->stack[stack->top--];
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscIntStackCreate"
/*@C
  PetscIntStackCreate - This function creates a stack.

  Not Collective

  Output Parameter:
. stack - The stack

  Level: developer

.keywords: log, stack, pop
.seealso: PetscIntStackDestroy(), PetscIntStackEmpty(), PetscIntStackPush(), PetscIntStackPop(), PetscIntStackTop()
@*/
PetscErrorCode PetscIntStackCreate(PetscIntStack *stack)
{
  PetscIntStack       s;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(stack,1);
  ierr = PetscNew(struct _n_PetscIntStack, &s);CHKERRQ(ierr);
  s->top = -1;
  s->max = 128;
  ierr = PetscMalloc(s->max * sizeof(int), &s->stack);CHKERRQ(ierr);
  ierr = PetscMemzero(s->stack, s->max * sizeof(int));CHKERRQ(ierr);
  *stack = s;
  PetscFunctionReturn(0);
}
