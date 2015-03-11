/*
 * $Header: /home/tv/src/debugger/src/datadisp/tv_data_display.h,v 1.3 2010-04-21 15:32:50 tringali Exp $
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
 * Sep 25 2009 SJT: Add idempotence header.
 * Jul 1  2009 SJT: Created.
 *
 */

#if !defined(TV_DATA_DISPLAY_H_INCLUDED)
#define TV_DATA_DISPLAY_H_INCLUDED 1

#include <petscsys.h>

#if defined(__cplusplus)
extern "C" {
#endif

enum TV_format_result
{
  TV_format_OK,             /* Type is known, and successfully converted */
  TV_format_failed,         /* Type is known, but could not convert it */
  TV_format_raw,            /* Just display it as a regular type for now */
  TV_format_never           /* Don't know about this type, and please don't ask again */
};

#define TV_ascii_string_type "$string"
#define TV_int_type "$int"

PETSC_EXTERN int TV_add_row(const char*,const char*,const void*);

/*
       0: Success
   EPERM: Called with no active callback to TV_display_type
  EINVAL: field_name or type_name has illegal characters
  ENOMEM: No more room left for display data
*/

#if defined(__cplusplus)
}
#endif

#endif
