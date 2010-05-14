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
@synthesize textField;
@synthesize textView;
@synthesize outPut;


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


/*
// Implement viewDidLoad to do additional setup after loading the view, typically from a nib.
- (void)viewDidLoad {
    [super viewDidLoad];
}
*/


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
	// Release any retained subviews of the main view.
	// e.g. self.myOutlet = nil;
}


- (void)dealloc {
  [textField release];
    [super dealloc];
}

UITextView *globalTextView;

PetscErrorCode PetscVFPrintfiPhone(FILE *fd,const char *format,va_list Argp)
{
  int len;
  char str[1024];
  
  PetscVSNPrintf(str,1024,format,&len,Argp);
  NSLog(@"%s",str);
  globalTextView.text = [NSString stringWithFormat:@"%@%s", globalTextView.text,str];
  
  return 0;
}

- (BOOL) textFieldShouldReturn: (UITextField*) theTextField {
  [theTextField resignFirstResponder];
  globalTextView = textView;
  char **args;
  int argc;
  // Override point for customization after app launch    
 
  const char *str = [textField.text UTF8String];
  PetscErrorCode ierr = PetscStrToArray(str,&argc,&args);
textView.text = @"";
  textView.font = [UIFont systemFontOfSize: 10.0f];

  ierr = PetscInitialize(&argc,&args,0,0); 
  if (ierr) {
    NSLog(@"Failed to initialize, likely command line mistake");
    return YES;
  }
  ierr = PetscFinalize();
  ierr = PetscStrToArrayDestroy(argc,args);
  
    return YES;
}

@end
