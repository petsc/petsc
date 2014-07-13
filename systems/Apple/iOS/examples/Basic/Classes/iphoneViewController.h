//
//  iphoneViewController.h
//  iphone
//
//  Created by Barry Smith on 5/12/10.
//  Copyright __MyCompanyName__ 2010. All rights reserved.
//

#import <UIKit/UIKit.h>

@interface iphoneViewController : UIViewController <UITextFieldDelegate>{
IBOutlet UITextField *textField;
IBOutlet UITextView *textView;
         NSString *outPut;
}
@property (nonatomic,retain) UITextField *textField;
@property (nonatomic,retain) UITextView *textView;
@property (nonatomic,retain) NSString *outPut;

@end

