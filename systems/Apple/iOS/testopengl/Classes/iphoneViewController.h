//
//  iphoneViewController.h
//  iphone
//
//  Created by Barry Smith on 5/12/10.
//  Copyright __MyCompanyName__ 2010. All rights reserved.
//

#import <UIKit/UIKit.h>
#import <GLKit/GLKit.h>
#import <OpenGLES/EAGLDrawable.h>

@interface iphoneViewController : UIViewController <UITextFieldDelegate>{
IBOutlet UITextField *textField;
IBOutlet UITextView *textView;
IBOutlet GLKView *glkView;
 
}
@property (retain, nonatomic) GLKView *glkView;
@property (nonatomic,retain) UITextField *textField;
@property (nonatomic,retain) UITextView *textView;


@end

