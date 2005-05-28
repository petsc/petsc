#include <stdio.h>
#include <stdlib.h>
#include "knr-compat.h"
#include "report-once.h"

#define ADINTRINSICS_C
#include "adintrinsics.h"

/* Global Variable */
/* Variable initialized in automatically generated file */
/* double ADIntr_Partials[ADINTR_FUNC_MAX][ADINTR_PARTIALS_MAX]; */
#include "initcommon.h"

#if defined(__cplusplus)
extern "C" {
#endif

typedef enum ADIntr_Modes Mode;

Mode ADIntr_Mode = ADINTR_REPORTONCE;

/* This provides the first 10 spots in the mode stack. 
   Most normal programs will probably not grow beyond this. */

static Mode *ADIntr_mode_stack = 0;
static int ADIntr_mode_depth = 0;
static int ADIntr_mode_max_depth = 10;

#define ADINTR_STACK_GROWTH_INCREMENT 10

Mode
adintr_current_mode ARG0(void)
{
     return ADIntr_Mode;
}


static void
ADIntr_die_malloc_failure ARG0(void)
{
     fprintf(stderr,"ADIntrinsics: out of virtual memory\n");
     fflush(stderr);
     abort();
}


void
adintr_mode_push ARG1(Mode, new_mode)
{
     if (!ADIntr_mode_stack) 
     {
	  ADIntr_mode_stack = (Mode *) malloc(ADIntr_mode_max_depth *
					      sizeof(Mode));
	  if (!ADIntr_mode_stack) 
	  {
	       ADIntr_die_malloc_failure();
	  }
     }

     if (ADIntr_mode_depth >= ADIntr_mode_max_depth) 
     {
	  ADIntr_mode_max_depth += ADINTR_STACK_GROWTH_INCREMENT;

	  ADIntr_mode_stack = 
	       (Mode *) realloc (ADIntr_mode_stack,
				 ADIntr_mode_max_depth * sizeof(Mode));

	  if (!ADIntr_mode_stack) 
	  {
	       ADIntr_die_malloc_failure();
	  }
     }
     
     ADIntr_mode_stack[ADIntr_mode_depth] = ADIntr_Mode;
     ADIntr_mode_depth++;

     ADIntr_Mode = new_mode;
}


void
adintr_mode_pop ARG0(void)
{
     if (!ADIntr_mode_stack || !ADIntr_mode_depth)
     {
	  fprintf (stderr,"ADIntrinsics warning: more mode POP's than PUSH's (arising from AD_EXCEPTION_BEGIN_IGNORE\n");
	  fprintf (stderr,"ADIntrinsics: Ignoring POP request\n");
	  fflush(stderr);
     } 
     else 
     {
	  ADIntr_mode_depth --;
	  ADIntr_Mode = ADIntr_mode_stack[ADIntr_mode_depth];
     }
}

/************************************************************************/

void
adintr_ehsup ARG3(enum ADIntr_Funcs, func, 
		  enum ADIntr_Partials, partial,
		  double, value)
{
     ADIntr_Partials[func][partial] = value;
}

double
adintr_ehgup ARG2(enum ADIntr_Funcs, func, 
		  enum ADIntr_Partials, partial)
{
     return ADIntr_Partials[func][partial];
}

void
adintr_ehsout ARG1(FILE *,the_file)
{
     reportonce_set_raw_output(the_file);
}

void 
adintr_ehrpt ARG0(void)
{
     reportonce_summary();
}

void
adintr_ehrst ARG0(void)
{
     reportonce_reset();
}

void
adintr_ehsfid ARG3(int*,g_ehfid, char *,routine, char *,filename)
{
     reportonce_ehsfid(g_ehfid, routine, filename);
}

#if defined(__cplusplus)
}
#endif

