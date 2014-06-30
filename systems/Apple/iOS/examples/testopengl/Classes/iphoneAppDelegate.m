//
//  iphoneAppDelegate.m
//  iphone
//
//  Created by Barry Smith on 5/12/10.
//  Copyright __MyCompanyName__ 2010. All rights reserved.
//

#import "iphoneAppDelegate.h"
#import "iphoneViewController.h"
#import <PETSc/petscsys.h>

extern PetscErrorCode PetscVFPrintfiPhone(FILE *,const char *,va_list);

@implementation iphoneAppDelegate

@synthesize window;
@synthesize viewController;


- (BOOL)application:(UIApplication *)application didFinishLaunchingWithOptions:(NSDictionary *)launchOptions {    
     // Override point for customization after app launch    
    [window addSubview:viewController.view];
    [window makeKeyAndVisible];
    MPI_Init(0,0);
    PetscVFPrintf = PetscVFPrintfiPhone;
	  return YES;
}


- (void)dealloc {
     MPI_Finalize();
    [viewController release];
    [window release];
    [super dealloc];
}


@end
