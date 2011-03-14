#include <stdio.h>
#include <petscconf.h>
#if defined(PETSC_HAVE_STRINGS_H)
#include <strings.h>
#endif
#if defined(PETSC_HAVE_STRING_H)
#include <string.h>
#endif
#if defined(PETSC_HAVE_STDLIB_H)
#include <stdlib.h>
#endif

#if defined(__cplusplus)
extern "C" {
#endif

#ifdef EVERYTHING_STATIC
#define RO_EXTERN static
#else
#define RO_EXTERN
#endif
#include <knr-compat.h>
#ifndef DISABLE_FORTRAN
#define MAX_PREPRO_ARGS 31 /* This is an option for cfortran.h */
#include <cfortran.h>
#endif /* DISABLE_FORTRAN */
#include "report-once.h"
#include "config.h"
static void *xmalloc _P((size_t));
static void *xcalloc _P((size_t, size_t));
static void *xrealloc _P((void*, size_t));
typedef struct exception_info {
     int line;
     int exception_type;
     unsigned long int count;

     struct exception_info *down;
} exception_info;
static const char *exceptions[] = {
#include "names.h"
};

static int hash_size = HASH_SIZE;
static int initial_max_files = INITIAL_MAX_FILES;
static int current_max_files = 0;
static int file_growth_increment = FILE_GROWTH_INCREMENT;
static int initial_store_created = 0;
static exception_info ***exception_info_store;
static int *line_numbers_count;

static int allocated = 0;
static int used = 0;

static char **filenames;
static char **routine_names;
static FILE *ERROR_FILE = 0;
static void *
xmalloc ARG1(size_t,size)
{
     void *tmp = malloc (size);
     if (!tmp) {
          fprintf(stderr,"report once mode: out of virtual memory\n");
          fflush(stderr);
          abort();
     }
     return tmp;
}
static void *
xcalloc ARG2(size_t, number, size_t, size_of_one)
{
     void *tmp = calloc ( number, size_of_one );
     if (!tmp)
     {
          fprintf (stderr,"report once mode: virtual memory exhausted\n");
          fflush(stderr);
          abort();
     }

     return tmp;
}
static void *
xrealloc ARG2(void*, ptr, size_t, new_size)
{
     void *tmp = realloc (ptr, new_size);
     if (!tmp)
     {
          fprintf (stderr,"report once mode: virtual memory exhausted\n");
          fflush(stderr);
          abort();
     }

     return tmp;
}

/* This depends on what the Fortran AD tool thinks. */
#define FORTRAN_UNDEFINED_FID  0 

#define ALREADY_ASSIGNED(fid) (fid != FORTRAN_UNDEFINED_FID)

RO_EXTERN void
reportonce_ehsfid ARG3(int*,g_ehfid, char *,routine, char *,filename)
{
     int routine_len;
     int filename_len;

     if ( ALREADY_ASSIGNED(*g_ehfid) )
     {
          return;
     }

     routine_len = strlen(routine);
     filename_len = strlen(filename);

     {
          if (!allocated)
          {
               allocated = initial_max_files;

               filenames = (char **) xmalloc (allocated * sizeof (char**));
               routine_names = (char **) xmalloc (allocated * sizeof (char**));
          }
          else if ( used >= allocated ) /* Should never be strictly greater */
          {
               allocated += file_growth_increment;

               filenames = (char **) xrealloc (filenames,
                                               allocated * sizeof(char*));

               routine_names = (char **) realloc (routine_names,
                                                  allocated * sizeof(char*));
          }
     }

     filenames[used] = (char *) xcalloc (filename_len+1, sizeof(char));
     routine_names[used] = (char *) xcalloc (routine_len+1, sizeof(char));

     strcpy (filenames[used], filename);
     strcpy (routine_names[used], routine);

     *g_ehfid = (used + 1); /* Fortran likes stuff numbered from 1 */
     used++;
}


RO_EXTERN void
reportonce_report_one ARG4(int, fid, int, line, 
                           int, exception_type, long int, count)
{
     if (!ERROR_FILE) ERROR_FILE = stderr;
     fprintf (ERROR_FILE,
              "At line %d in file \"%s\", while executing routine \"%s\",\n",
              line,
              filenames[fid],
              routine_names[fid]
              );
     fprintf (ERROR_FILE,
              "an exception occurred evaluating %.30s : %ld %s.\n",
              exceptions[exception_type],
              count,
              (count == 1) ? "time" : "times"
              );
     fprintf (ERROR_FILE, "\n");
}
RO_EXTERN void
reportonce_files ARG1(int, new_initial_size)
{
     initial_max_files = new_initial_size;
}
RO_EXTERN void
reportonce_accumulate ARG3(int, file, int, line, int, exception)
{
     /* Adjust to internally number from 0 */
     file = file - 1; 

     if ( ! initial_store_created )
     {
          {
               int i;
               
               /* We depend on calloc'ed memory to read as integer 0 */
               
               exception_info_store =
                    (exception_info ***) xcalloc ( initial_max_files,
                                                   sizeof ( exception_info **) );
               
               line_numbers_count =
                    (int*) xcalloc ( initial_max_files, sizeof (int));

               for (i=0; i < initial_max_files; i++ )
               {
                    exception_info_store[i] =
                         (exception_info **) xcalloc (hash_size,
                                                      sizeof (exception_info *));
               }

               initial_store_created = 1;
               current_max_files = initial_max_files;
          }
     }

     {
          while ( file >= current_max_files )
          {
               int i;
               
               exception_info_store =
                    (exception_info ***) xrealloc ( exception_info_store, 
                                 (current_max_files + file_growth_increment ) * 
                                 sizeof ( exception_info ** ) );

               line_numbers_count =
                    (int*) xrealloc (line_numbers_count,
                                      (current_max_files + file_growth_increment)*
                                      sizeof (int) );

               for (i = current_max_files;
                    i < current_max_files + file_growth_increment;
                    i++)
               {
                    exception_info_store[i] =
                         (exception_info **) xcalloc (hash_size,
                                                      sizeof (exception_info *));
                    line_numbers_count[i] = 0;
               }

               current_max_files += file_growth_increment;
          }
     }
     do {
          int hashed_line = line % hash_size;
          exception_info *our_loc;
          exception_info *previous_loc = 0;

          {
               if (!exception_info_store[file][hashed_line])
               {
                    exception_info_store[file][hashed_line] =
                         (exception_info*)xcalloc (1, sizeof(exception_info));
                    
                    our_loc = exception_info_store[file][hashed_line];
                    
                    our_loc->line = line;
                    our_loc->exception_type = exception;
                    our_loc->count = 1;
                    our_loc->down = NULL; 
                    
                    line_numbers_count[file] += 1;

                    break;
               }
          }
               /* (This routine does a "break" to leave this section.) */

          /* We know this is not zero now */
          our_loc = exception_info_store[file][hashed_line];

          {
               while ((our_loc != NULL) && (our_loc->line != line))
               {
                    previous_loc = our_loc;
                    our_loc = our_loc->down;
               }
          }

          if (!our_loc)
          {
               {
                    exception_info *old_first_elt = exception_info_store[file][hashed_line];

                    exception_info_store[file][hashed_line] =
                         (exception_info*)xcalloc (1, sizeof(exception_info));
                    
                    our_loc = exception_info_store[file][hashed_line];
                    
                    our_loc->line = line;
                    our_loc->exception_type = exception;
                    our_loc->count = 1;
                    our_loc->down = old_first_elt; 
                         
                    line_numbers_count[file] += 1;
               }
          }
          else
          {
               /* Move up to the start of the line if we are not already first */
               if ( previous_loc != 0 )
               {
                    /* Save the first node's next pointer in case
                       we are swapping #2 and #1. */

                    exception_info *first_next =
                         exception_info_store[file][hashed_line];

                    /* We are not first (yet...) */
                    previous_loc->down = our_loc->down;
                    our_loc->down = first_next;

                    /* Now we are first */
                    exception_info_store[file][hashed_line] = our_loc;
               }
               our_loc->count += 1;
          }

     } while (0);


}
RO_EXTERN void
reportonce_summary ARG0(void)
{
     {
          int current_file;
          struct exception_info switch_tmp;
          struct exception_info * elts;
          int i,j;

          for (current_file = 0 ; current_file < current_max_files ; current_file++)
          {
               int found_count = 0;

               /* Just skip this iteration if there's nothing to be done. */
               if (!line_numbers_count[current_file]) 
                    continue;

              /* Make an array big enough to hold all of the extracted
                 info, then sort it in that array.
              */
              elts = (struct exception_info * ) 
                   xcalloc (line_numbers_count[current_file] + 1,
                            sizeof(struct exception_info));

              /* 
                 For a given file, walk along each main bucket of the array.
              */
              for (i = 0; i < hash_size; i++)
              {
                   /* Anybody home? */
                   if ( (exception_info_store[current_file][i] != 0)
                        && (exception_info_store[current_file][i]->line != 0) )
                   {
                        exception_info current_elt;

                        /* Yes. */
                        current_elt = *exception_info_store[current_file][i];
                        elts[found_count] = current_elt;
                        found_count++;

                        /* Check for more folks chained off the bottom */
                        while (current_elt.down != 0)
                        {
                             current_elt = *(current_elt.down);
                             elts[found_count] = current_elt;
                             found_count++;
                        }
                   }
              }

              if ( found_count != line_numbers_count[current_file])
              {        
                   fprintf(stderr, "report once: Failed internal consistency check.\n");
                   abort();
              }

               /* Sort the elements: Bubblesort */
               for (i=0;i<found_count; i++)
               {
                    for (j=i; j<found_count; j++)
                    {
                         if ( elts[i].line > elts[j].line )
                         {
                              switch_tmp = elts[i];
                              elts[i] = elts[j];
                              elts[j] = switch_tmp;
                         }
                    }
               }

               /* Now print them out. */
               
               for ( i=0; i<found_count; i++)
               {
                    reportonce_report_one (current_file,
                                           elts[i].line,
                                           elts[i].exception_type,
                                           elts[i].count);
               }

               /* Clean up */
               free (elts);
          }
     }

}
RO_EXTERN void reportonce_reset ARG0(void)
{
     int file_count;
     int line_hash_count;
     
     for (file_count = 0; file_count < current_max_files; file_count++)
     {
          line_numbers_count[file_count] = 0;
          
          for (line_hash_count = 0;
               line_hash_count < hash_size ;
               line_hash_count++)
          {
               if ( exception_info_store[file_count][line_hash_count] != 0 )
               {
                    free(exception_info_store[file_count][line_hash_count]);
                    exception_info_store[file_count][line_hash_count] = 0;
               }
          }
     }
}
RO_EXTERN void
reportonce_set_output_file ARG1(char *,output_filename)
{
     FILE *check_file;
     check_file = fopen(output_filename,"w");
     if (!check_file)
     {
          fprintf(stderr,"Unable to open reportonce output file: %s\n",
                  output_filename);
          fprintf(stderr,"Proceding to emit errors to standard error.\n");
          fflush(stderr);
     }
     else
     {
          ERROR_FILE = check_file;
     }
}
RO_EXTERN void
reportonce_set_raw_output ARG1(FILE *,outfile)
{
     ERROR_FILE = outfile;
}

RO_EXTERN char *
reportonce_get_filename ARG1(int, file_id)
{
     return filenames[file_id];
}

RO_EXTERN char *
reportonce_get_routine_name ARG1(int, file_id)
{
     return routine_names[file_id];
}


/* Long names are disabled unless ENABLE_LONG_FORTRAN_NAMES is defined */
/* Prototypes put here for clarity; real work is done by CFORTRAN.H */
#if 0
void once_summary (void);
void once_reset (void);
void once_accumulate (int *file, int *line, int *exception);
void once_max_files (int *new_files);
void once_output_file (char *filename);
void ehsfid (int *g_ehfid, char *routine, char *filename);
#endif

#ifndef DISABLE_FORTRAN
#ifdef ENABLE_LONG_FORTRAN_NAMES
/* Long names */
FCALLSCSUB0(reportonce_summary,ONCE_SUMMARY,once_summary)
FCALLSCSUB3(reportonce_accumulate,ONCE_ACCUMULATE,once_accumulate,INT,INT,INT)
FCALLSCSUB1(reportonce_files,ONCE_MAX_FILES,once_max_files,INT)
#endif

/* Short (<=6 characters) names */
FCALLSCSUB0(reportonce_summary,EHORPT,ehorpt)
FCALLSCSUB3(reportonce_accumulate,EHOACC,ehoacc,INT,INT,INT)
FCALLSCSUB1(reportonce_files,EHOMXF,ehomxf,INT)

FCALLSCSUB3(reportonce_ehsfid,EHSFID,ehsfid,PINT,STRING,STRING)
#ifdef ENABLE_LONG_FORTRAN_NAMES
FCALLSCSUB1(reportonce_set_output_file,ONCE_OUTPUT_FILE,once_output_file,STRING)
#endif
FCALLSCSUB1(reportonce_set_output_file,EHOFIL,ehofil,STRING)
#ifdef ENABLE_LONG_FORTRAN_NAMES
/* Long name */
FCALLSCSUB0(reportonce_reset,ONCE_RESET,once_reset)
#endif
/* Short name */
FCALLSCSUB0(reportonce_reset,EHORST,ehorst)

#endif /* DISABLE_FORTRAN */

#if defined(__cplusplus)
}
#endif

