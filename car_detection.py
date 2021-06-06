import cv2
import numpy as np

# Change Cars.mp4 with your road traffic video
vid = cv2.VideoCapture("Cars.mp4")
arrow = 60
car_count = []
mask = None
prev_frame = None
roi = None
kernel_erode = np.ones((4,4), np.uint8)
kernel_dilate = np.ones((20,20), np.uint8)

def drawMask(event,x,y,flag,param):
    global ix2,iy2,drawing, arrow
    #left click pressed
    if event == cv2.EVENT_LBUTTONDOWN: 
        drawing = True
        #pointer position
        ix2,iy2 = x,y   
        # mouse movement
    elif event == cv2.EVENT_MOUSEMOVE: 
        if drawing == True:
            #draws circle in x,y pos
            cv2.circle(mask,(x,y), arrow,(255,255,255),-1) 
    elif event == cv2.EVENT_LBUTTONUP:  
        #button not pressed
        drawing =False
        cv2.circle(mask,(x,y), arrow,(255,255,255),-1)

def frame_comparison(actual_frame, previous_frame, image_mask, previous_comparison_frame):
    #frame variable assignment and comparison between actual and previous frame
    comparison_frame = cv2.bitwise_and(actual_frame, previous_frame, mask = image_mask)
    comparison = cv2.absdiff(previous_comparison_frame, comparison_frame)
    previous_comparison_frame = comparison_frame
    previous_frame = actual_frame
    _, comparison = cv2.threshold(comparison, 30, 255, cv2.THRESH_BINARY)
    comparison = cv2.morphologyEx(comparison, cv2.MORPH_ERODE, kernel_erode)
    comparison = cv2.morphologyEx(comparison, cv2.MORPH_DILATE, kernel_dilate)
    comparison = cv2.GaussianBlur(comparison, (11,11), 5)
    return comparison, previous_comparison_frame, previous_frame

def car_contour_counter(compared_images, roi_coords, car_counter_list):
                contours, hierarchy = cv2.findContours(comparison, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                for car in contours:
                    x,y,w,h = cv2.boundingRect(car)
                    roi_area = ((roi[0] + roi[2]) * (roi[1] + roi[3]))/1.5
                    if (x <= (roi[0] + roi[2])) & (y >= (roi[1] + roi[3])):
                        car_count.append(car)
                return contours, car_count

while(vid.isOpened()):
    _, image = vid.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Selects region of interest if one hasn't been picked yet
    if roi is None:
        roi = cv2.selectROI(image)
    # Selects a mask if one hasn't been selected yet
    elif mask is None:
        drawing = False
        alpha= 0.7  
        mask = np.zeros(gray.shape,np.uint8)
        cv2.namedWindow('Create Mask')
        # Mouse events inside window within cv
        cv2.setMouseCallback('Create Mask',drawMask) 
        # alpha color over the image
        overlay = gray.copy() 
        # creates a copy of the image
        output =gray.copy() 
        # Mask creation - Separate window to draw mask region
        while True:
            cv2.addWeighted(overlay,alpha,mask,1-alpha,0,output)
            cv2.imshow('Create Mask',output)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        cv2.destroyAllWindows()
    else:
        # draws the ROI rectangle in the main colored image
        cv2.rectangle(image, (roi[0], roi[1]), (roi[0] + roi[2], roi[1] + roi[3]), (0,0,255), 1)
        cv2.rectangle(gray, (roi[0], roi[1]), (roi[0] + roi[2], roi[1] + roi[3]), (0,0,255), 1)
        # initialization comparison, verifies if there is a previous frame, if not assigns one
        if prev_frame is None:
            prev_frame = gray
            # frame comparison within mask region
            prev_comp_frame = cv2.bitwise_and(gray, prev_frame, mask = mask)
        else:
            comparison, prev_comp_frame, prev_frame = frame_comparison(gray, prev_frame, mask, prev_comp_frame)
            contours, car_count = car_contour_counter(comparison, roi, car_count)
            cv2.drawContours(image, contours, -1, (0, 90, 200))
            #cv2.imshow("Comparacion", comparison)
        cv2.putText(image,f"Vehiculos: {round(len(car_count)/30, 2)}", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.imshow("Vehicle Detection", image)
    # exits program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
      
cv2.destroyAllWindows()