# define HLINE "-------------------------------------------------------------\n"

# ifndef MIN
# define MIN(x,y) ((x)<(y)?(x):(y))
# endif
# ifndef MAX
# define MAX(x,y) ((x)>(y)?(x):(y))
# endif

#define OFFSET 0
#define NTIMES 10
#define N 2000000

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <float.h>

static double	a[N+OFFSET],
		b[N+OFFSET],
		c[N+OFFSET];

static double	rmstime[4] = {0}, maxtime[4] = {0},
		mintime[4] = {FLT_MAX,FLT_MAX,FLT_MAX,FLT_MAX};

static char	*label[4] = {"Copy:      ", "Scale:     ",
   "Add:       ", "Triad:     "};

static double	bytes[4] = {
   2 * sizeof(double) * N,
   2 * sizeof(double) * N,
   3 * sizeof(double) * N,
   3 * sizeof(double) * N
   };

extern double second();

int
main()
   {
   int			quantum, checktick();
   int			BytesPerWord;
   register int	j, k;
   double		scalar, t, times[4][NTIMES];

   /* --- SETUP --- determine precision and check timing --- */

   printf(HLINE);
   BytesPerWord = sizeof(double);
   printf("This system uses %d bytes per DOUBLE PRECISION word.\n",
	BytesPerWord);

   printf(HLINE);
   printf("Array size = %d, Offset = %d\n" , N, OFFSET);
   printf("Total memory required = %.1f MB.\n",
	(3 * N * BytesPerWord) / 1048576.0);
   printf("Each test is run %d times, but only\n", NTIMES);
   printf("the *best* time for each is used.\n");

   /* Get initial value for system clock. */

   for (j=0; j<N; j++) {
	a[j] = 1.0;
	b[j] = 2.0;
	c[j] = 0.0;
	}

   printf(HLINE);

   if  ( (quantum = checktick()) >= 1) 
	printf("Your clock granularity/precision appears to be "
	    "%d microseconds.\n", quantum);
   else
	printf("Your clock granularity appears to be "
	    "less than one microsecond.\n");

   t = second();
   for (j = 0; j < N; j++)
	a[j] = 2.0E0 * a[j];
   t = 1.0E6 * (second() - t);

   printf("Each test below will take on the order"
	" of %d microseconds.\n", (int) t  );
   printf("   (= %d clock ticks)\n", (int) (t/quantum) );
   printf("Increase the size of the arrays if this shows that\n");
   printf("you are not getting at least 20 clock ticks per test.\n");

   printf(HLINE);

   printf("WARNING -- The above is only a rough guideline.\n");
   printf("For best results, please be sure you know the\n");
   printf("precision of your system timer.\n");
   printf(HLINE);

   /*	--- MAIN LOOP --- repeat test cases NTIMES times --- */

   scalar = 3.0;
   for (k=0; k<NTIMES; k++)
	{
	times[0][k] = second();
	for (j=0; j<N; j++)
	    c[j] = a[j];
	times[0][k] = second() - times[0][k];
	
	times[1][k] = second();
	for (j=0; j<N; j++)
	    b[j] = scalar*c[j];
	times[1][k] = second() - times[1][k];
	
	times[2][k] = second();
	for (j=0; j<N; j++)
	    c[j] = a[j]+b[j];
	times[2][k] = second() - times[2][k];
	
	times[3][k] = second();
	for (j=0; j<N; j++)
	    a[j] = b[j]+scalar*c[j];
	times[3][k] = second() - times[3][k];
	}

   /*	--- SUMMARY --- */

       for (k=0; k<NTIMES; k++)
	{
	for (j=0; j<4; j++)
	    {
	    rmstime[j] = rmstime[j] + (times[j][k] * times[j][k]);
	    mintime[j] = MIN(mintime[j], times[j][k]);
	    maxtime[j] = MAX(maxtime[j], times[j][k]);
	    }
	}

   printf("Function      Rate (MB/s)   RMS time     Min time     Max time\n");
   for (j=0; j<4; j++) {
	rmstime[j] = sqrt(rmstime[j]/(double)NTIMES);

	printf("%s%11.4f  %11.4f  %11.4f  %11.4f\n", label[j],
	       1.0E-06 * bytes[j]/mintime[j],
	       rmstime[j],
	       mintime[j],
	       maxtime[j]);
   }
   return 0;
}

# define	M	20

int
checktick()
   {
   int		i, minDelta, Delta;
   double	t1, t2, timesfound[M];

/*  Collect a sequence of M unique time values from the system. */

   for (i = 0; i < M; i++) {
	t1 = second();
	while( ((t2=second()) - t1) < 1.0E-6 )
	    ;
	timesfound[i] = t1 = t2;
	}

/*
* Determine the minimum difference between these M values.
* This result will be our estimate (in microseconds) for the
* clock granularity.
*/

   minDelta = 1000000;
   for (i = 1; i < M; i++) {
	Delta = (int)( 1.0E6 * (timesfound[i]-timesfound[i-1]));
	minDelta = MIN(minDelta, MAX(Delta,0));
	}

   return(minDelta);
   }
