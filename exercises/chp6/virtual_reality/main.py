import cv2
import numpy as np


def imshow(x):
    cv2.imshow('frame', x) 

# now let's initialize the list of reference point


def shape_selection(event, x, y, flags, param):
    # grab references to the global variables
    global ref_point, crop, frame

    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being performed
    if event == cv2.EVENT_LBUTTONDOWN:
        ref_point = [(x, y)]

    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        ref_point.append((x, y))

        # draw a rectangle around the region of interest
        cv2.rectangle(frame, ref_point[0], ref_point[1], (0, 255, 0), 2)
        cv2.imshow("frame", frame)    
    
if __name__ == '__main__':
    cv2.namedWindow('frame')
    cv2.setMouseCallback("frame", shape_selection)
    video_capture_device_index = 0
    webcam = cv2.VideoCapture(video_capture_device_index)
    adv = cv2.imread('lena.jpg')
    ref_point = None
    descriptor = cv2.ORB_create()
    crop_des = None
    crop_kp = None
    crop_rect = None
    matcher = cv2.BFMatcher_create()    
    while(True):
        ret, frame = webcam.read()
        if cv2.waitKey(1) & 0xFF == ord('d'):
            print("Draw your rectangle!")
            while(True):
                cv2.imshow('frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('d'):
                    break
                
        if ref_point is not None:
            print(ref_point)            
            bc, br = ref_point[0]
            ec, er = ref_point[1]            
            crop = frame[br:er, bc:ec, :]
            cv2.imshow('crop', crop)
            cv2.waitKey(0)
            crop_kp, crop_des = descriptor.detectAndCompute(crop, None)
            crop_rect = ref_point
            ref_point = None

        if crop_des is not None:
            kp2, des2 = descriptor.detectAndCompute(frame, None)
            matcher = cv2.BFMatcher_create()
            raw_matches = matcher.knnMatch(crop_des, des2, k=2)
            # store all the good matches as per Lowe's ratio test.
            good = []
            for m, n in raw_matches:
                if m.distance < 0.75*n.distance:
                    good.append(m)
            if len(good)>10:
                src_pts = np.float32([crop_kp[m.queryIdx].pt for m in good]).reshape(-1,1,2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                matchesMask = mask.ravel().tolist()
                print("MATCH!")
                h, w, d = crop.shape
                pts = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)
                dst = cv2.perspectiveTransform(pts,M)
                
                h, w, d = frame.shape                
                proj = cv2.warpPerspective(adv, M, dsize=(w, h))
                mask = proj > 0

                frame[mask] = proj[mask]
                print(dst)
                #frame = cv2.polylines(frame,[np.int32(dst)],True,255,3, cv2.LINE_AA)                
        imshow(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    webcam.release() 

    cv2.destroyAllWindows() 
