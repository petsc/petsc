//
//  main.m
//  iphone
//
//  Created by Barry Smith on 5/12/10.
//  Copyright __MyCompanyName__ 2010. All rights reserved.
//

#import <UIKit/UIKit.h>


int main(int argc, char *argv[]) {
    
    NSAutoreleasePool * pool = [[NSAutoreleasePool alloc] init];
 @try {
 int retVal = UIApplicationMain(argc, argv, nil, nil);
 }
 @catch (NSException *exception) {
 NSLog(@"Exception - %@",[exception description]);
 exit(EXIT_FAILURE);
 }
 
    int retVal = UIApplicationMain(argc, argv, nil, nil);
    [pool release];
    return retVal;
}

/*
int main(int argc, char *argv[])
{
    int retVal = 0;
    @autoreleasepool {
        NSString *classString = NSStringFromClass([sortaAppDelegate class]);
        @try {
            retVal = UIApplicationMain(argc, argv, nil, classString);
        }
        @catch (NSException *exception) {
            NSLog(@"Exception - %@",[exception description]);
            exit(EXIT_FAILURE);
        }
    }
    return retVal;
}
*/