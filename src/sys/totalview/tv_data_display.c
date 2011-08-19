/* 
 * $Header: /home/tv/src/debugger/src/datadisp/tv_data_display.c,v 1.4 2010-04-21 15:32:50 tringali Exp $
 * $Locker:  $

   Copyright (c) 2010, Rogue Wave Software, Inc.

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
   THE SOFTWARE.

 * Update log
 *
 * Jan 28 2010 SJT: Bug 12100, bump base size to 16K and recognize if it is
 *                  resized further.
 * Sep 24 2009 SJT: Remove pre/post callback to reduce function call overhead.
 * Jul 1  2009 SJT: Created.
 *
 */

#include <../src/sys/totalview/tv_data_display.h>
#include <petscconf.h>

#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>

#define DATA_FORMAT_BUFFER_SIZE 1048576
#define TV_FORMAT_INACTIVE 0
#define TV_FORMAT_FIRST_CALL 1
#define TV_FORMAT_APPEND_CALL 2

volatile int TV_data_format_control = TV_FORMAT_INACTIVE;
   
/* TV_data_format_buffer should not be static for icc 11, and others */
char TV_data_format_buffer[DATA_FORMAT_BUFFER_SIZE];
static char *TV_data_buffer_ptr = TV_data_format_buffer;

int TV_add_row(const char *field_name,
               const char *type_name,
               const void *value)
{
  size_t remaining;
  int out;

  /* Called at the wrong time */
  if (TV_data_format_control == TV_FORMAT_INACTIVE)
    return EPERM;
    
  if (strpbrk(field_name, "\n\t") != NULL)
    return EINVAL;

  if (strpbrk(type_name, "\n\t") != NULL)
    return EINVAL;

  if (TV_data_format_control == TV_FORMAT_FIRST_CALL)
    {
      /* Zero out the buffer to avoid confusion, and set the write point 
         to the top of the buffer. */

      memset(TV_data_format_buffer, 0, DATA_FORMAT_BUFFER_SIZE);
      TV_data_buffer_ptr = TV_data_format_buffer;
      TV_data_format_control = TV_FORMAT_APPEND_CALL;
    }
        
  remaining = TV_data_buffer_ptr + DATA_FORMAT_BUFFER_SIZE - TV_data_format_buffer;
  
#if defined(PETSC_HAVE__SNPRINTF) && !defined(PETSC_HAVE_SNPRINTF)
#define snprintf _snprintf
#endif
  out = snprintf(TV_data_buffer_ptr,remaining, "%s\t%s\t%p\n",field_name, type_name, value);
  
  if (out < 1)
    return ENOMEM;
    
  TV_data_buffer_ptr += out;
  
  return 0;
}

void TV_pre_display_callback(void)
{
  TV_data_format_control = TV_FORMAT_FIRST_CALL;
}

void TV_post_display_callback(void)
{
  TV_data_format_control = TV_FORMAT_INACTIVE;
}
