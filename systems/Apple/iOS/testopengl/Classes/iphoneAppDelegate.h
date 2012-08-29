//
//  iphoneAppDelegate.h
//  iphone
//
//  Created by Barry Smith on 5/12/10.
//  Copyright __MyCompanyName__ 2010. All rights reserved.
//

#import <UIKit/UIKit.h>

@class iphoneViewController;

@interface iphoneAppDelegate : NSObject <UIApplicationDelegate> {
    UIWindow *window;
    iphoneViewController *viewController;
}

@property (nonatomic, retain) IBOutlet UIWindow *window;
@property (nonatomic, retain) IBOutlet iphoneViewController *viewController;

@end

