//
//  iphoneViewController.m
//  iphone
//
//  Created by Barry Smith on 5/12/10.
//  Copyright __MyCompanyName__ 2010. All rights reserved.
//

#import "iphoneViewController.h"
#include "petsc.h"

@implementation iphoneViewController
@synthesize glkView;
@synthesize textField;
@synthesize textView;


/*
// The designated initializer. Override to perform setup that is required before the view is loaded.
- (id)initWithNibName:(NSString *)nibNameOrNil bundle:(NSBundle *)nibBundleOrNil {
    if ((self = [super initWithNibName:nibNameOrNil bundle:nibBundleOrNil])) {
        // Custom initialization
    }
    return self;
}
*/


/*
 // Implement loadView to create a view hierarchy programmatically, without using a nib.
- (void)loadView {
}
*/


// Implement viewDidLoad to do additional setup after loading the view, typically from a nib.
- (void)viewDidLoad {
    [super viewDidLoad];
    EAGLContext *context = [[EAGLContext alloc] initWithAPI:kEAGLRenderingAPIOpenGLES2];
    
    if (!context) {
        NSLog(@"Failed to create ES context");
    }
    NSLog(@"Created ES context");
    
    glkView.context = context;
    [EAGLContext setCurrentContext:context];
}



/*
// Override to allow orientations other than the default portrait orientation.
- (BOOL)shouldAutorotateToInterfaceOrientation:(UIInterfaceOrientation)interfaceOrientation {
    // Return YES for supported orientations
    return (interfaceOrientation == UIInterfaceOrientationPortrait);
}
*/

- (void)didReceiveMemoryWarning {
	// Releases the view if it doesn't have a superview.
    [super didReceiveMemoryWarning];
	
	// Release any cached data, images, etc that aren't in use.
}

- (void)viewDidUnload {
    [glkView release];
    glkView = nil;
    [glkView release];
    glkView = nil;
    [self setGlkView:nil];
	// Release any retained subviews of the main view.
	// e.g. self.myOutlet = nil;
}


- (void)dealloc {
  [textField release];
    [glkView release];
    [glkView release];
    [glkView release];
    [super dealloc];
}

/*
    Not sure if this does anything
*/ 
void SwapBuffers()
{
    EAGLContext* context = [EAGLContext currentContext];
    
    [context presentRenderbuffer:GL_RENDERBUFFER];
}

UITextView *globalTextView;


/*
   This is called by PETSc for all print calls.

   Simply addeds to the NSString in globalTextView and it gets displayed in the UITextView in the display
*/
PetscErrorCode PetscVFPrintfiPhone(FILE *fd,const char *format,va_list Argp)
{
  size_t len;
  char   str[1024];
  
  PetscVSNPrintf(str,1024,format,&len,Argp);
  globalTextView.text = [NSString stringWithFormat:@"%@%s", globalTextView.text,str];
  return 0;
}

#define main ex19
#define help help19
#define Field Field19
#include <../src/snes/examples/tutorials/ex19.c>
#undef main 
#undef help
#undef Field
#define main ex48
#define help help48
#define Field Field48
#include <../src/snes/examples/tutorials/ex48.c>
#undef main
#undef help
#undef Field
#define main ex4
#define help help4
#include <../src/sys/draw/examples/tests/ex4.c>
#undef main
#undef help
#undef Field
#define main ex3
#define help help3
#include <../src/sys/draw/examples/tests/ex3.c>

extern PetscErrorCode  PetscDrawOpenGLESRegisterGLKView(GLKView *);

/*
    This is called each time one hits return in the TextField.

    Converts the string to a collection of arguments that are then passed on to PETSc
*/
- (BOOL) textFieldShouldReturn: (UITextField*) theTextField {
  [theTextField resignFirstResponder]; /* makes the keyboard disappear */
  textView.text = @"";   /* clears the UITextView */

  globalTextView = textView;   /* we make this class member a global so can use in PetscVFPrintfiPhone() */

  PetscDrawOpenGLESRegisterGLKView(glkView);  /* Let PETSc know about this window it may use */
  textView.font = [UIFont fontWithName:@"Courier" size:8.0]; /* make the font size in the UITextView a more reasonable size  and use fixed width*/
 
    
   
    
  const char *str = [textField.text UTF8String];
  char **args;
  int argc;
  PetscBool flg1,flg2,flg3,flg4;
  
  PetscErrorCode ierr = PetscStrToArray(str,&argc,&args);
  ierr = PetscStrncmp(str, "./ex19", 6, &flg1);
  ierr = PetscStrncmp(str, "./ex48", 6, &flg2);
  ierr = PetscStrncmp(str, "./ex4", 5, &flg3);
  ierr = PetscStrncmp(str, "./ex3", 5, &flg4);
  if (flg1) {
    ex19(argc,args);
  } else if (flg2) {
    ex48(argc,args);
  } else if (flg3) {
    ex4(argc,args);
  } else if (flg4) {
    ex3(argc,args);
  } else {
    textView.text =@"Must start with ./ex3, ./ex4,  ./ex19 or ./ex48 ";
    ierr = PetscStrToArrayDestroy(argc,args);
    return YES;
  }
  ierr = PetscStrToArrayDestroy(argc,args);
    
    
  return YES;
}



@end
