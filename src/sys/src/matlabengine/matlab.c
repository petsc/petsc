/* $Id: makefile,v 1.1 2000/04/02 04:46:38 bsmith Exp bsmith $ #include "petsc.h" */

#include "petsc.h"
#include "engine.h"   /* Matlab include file  */

/* Pointer to Matlab engine data structure */
static Engine *ep = PETSC_NULL;
static char   buffer[1024];

#undef __FUNC__  
#define __FUNC__ "PetscMatlabEngineInitialize"
int PetscMatlabEngineInitialize(MPI_Comm comm,char *machine)
{

  PetscFunctionBegin;
  if (ep) PetscFunctionReturn(0);
  if (!machine) machine = "\0";
  ep = engOpen(machine);
  if (!ep) SETERRQ1(1,1,"Unable to start Matlab engine on %s\n",machine);
  engOutputBuffer(ep,buffer,1024);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PetscMatlabEngineFinalize"
int PetscMatlabEngineFinalize(MPI_Comm comm)
{
  PetscFunctionBegin;
  if (!ep) PetscFunctionReturn(0);
  engClose(ep);
  ep = PETSC_NULL;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PetscMatlabEngineEvaluate"
int PetscMatlabEngineEvaluate(MPI_Comm comm,char *string)
{
  PetscFunctionBegin;  
  engEvalString(ep, string);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PetscMatlabEngineGetOutput"
int PetscMatlabEngineGetOutput(MPI_Comm comm,char **string)
{
  PetscFunctionBegin;  
  *string = buffer + 2;
  PetscFunctionReturn(0);
}

#define  BUFSIZE 256
int dummy()
{
	mxArray *T = NULL, *result = NULL;
	char buffer[BUFSIZE];
	double time[10] = { 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 };

	/*
	 * PART I
	 *
	 * For the first half of this demonstration, we will send data
	 * to MATLAB, analyze the data, and plot the result.
	 */

	/* 
	 * Create a variable for our data
	 */
	T = mxCreateDoubleMatrix(1, 10, mxREAL);
	mxSetName(T, "T");
	/* memcpy((void *)mxGetPr(T), (void *)time, sizeof(time)); */
	/*
	 * Place the variable T into the MATLAB workspace
	 */
	engPutArray(ep, T);

	mxDestroyArray(T);



	/*
	 * PART II
	 *
	 * For the second half of this demonstration, we will request
	 * a MATLAB string, which should define a variable X.  MATLAB
	 * will evaluate the string and create the variable.  We
	 * will then recover the variable, and determine its type.
	 */
	  
	/*
	 * Use engOutputBuffer to capture MATLAB output, so we can
	 * echo it back.
	 */

	engOutputBuffer(ep, buffer, BUFSIZE);
	while (result == NULL) {
	    char str[BUFSIZE];
	    /*
	     * Get a string input from the user
	     */
	    printf("Enter a MATLAB command to evaluate.  This command should\n");
	    printf("create a variable X.  This program will then determine\n");
	    printf("what kind of variable you created.\n");
	    printf("For example: X = 1:5\n");
	    printf(">> ");

	    fgets(str, BUFSIZE-1, stdin);
	  
	    /*
	     * Evaluate input with engEvalString
	     */
	    engEvalString(ep, str);
	    
	    /*
	     * Echo the output from the command.  First two characters are
	     * always the double prompt (>>).
	     */
	    printf("%s", buffer+2);
	    
	    /*
	     * Get result of computation
	     */
	    printf("\nRetrieving X...\n");
	    if ((result = engGetArray(ep,"X")) == NULL)
	      printf("Oops! You didn't create a variable X.\n\n");
	    else {
		printf("X is class %s\t\n", mxGetClassName(result));
	    }
	}
return 0;
}






