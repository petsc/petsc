//
//  AppDelegate.m
//  PETSc cocoa example
//
//  Created by Barry Smith on 8/2/12.
//  Copyright (c) 2012 Barry Smith. All rights reserved.
//

#import "AppDelegate.h"
#import "PETSc/petsc.h"


/*
 This is called by PETSc for all print calls.
 
 Need to create a place in Cocoa to put the print messages; commented out code below is from iOS example
 
 */
PetscErrorCode PetscVFPrintfiPhone(FILE *fd,const char *format,va_list Argp)
{
    size_t len;
    char   str[1024];
    
    PetscVSNPrintf(str,1024,format,&len,Argp);
    // globalTextView.text = [NSString stringWithFormat:@"%@%s", globalTextView.text,str];
    return 0;
}

extern PetscErrorCode PetscVFPrintfiPhone(FILE *,const char *,va_list);

@implementation AppDelegate

- (void)dealloc
{
    [super dealloc];
}

#define main ex19
#define PETSC_APPLE_FRAMEWORK
#include "../../../../../../src/snes/examples/tutorials/ex19.c"

- (void)applicationDidFinishLaunching:(NSNotification *)aNotification
{
    // Insert code here to initialize your application
    PetscVFPrintf = PetscVFPrintfiPhone;
    char **args;
    int  argc;
 
    /* this example is silly because it just runs a PETSc example when the graphics window appears
        but it does test the use of the PETSc framework */
    
    PetscStrToArray("ex19 -ksp_monitor",&argc,&args);
    ex19(argc,args);
    
    
}

@end
