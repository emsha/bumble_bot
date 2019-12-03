import pyscreenshot as ImageGrab
import numpy as np
import cv2 as cv




def extractFacesFromScreen(full_screen_path, save_path):

    # grab fullscreen
    im = ImageGrab.grab()
    im_np = np.array(im)
    # save image file
    # print(0)
    # img_name = full_screen_path+'screenshot.png'
    # print(1)
    # im.save(img_name)
    # print(2)
    # show image in a window
    # im.show()
    
    face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_cascade = cv.CascadeClassifier('haarcascade_eye.xml')
    # img = cv.imread(img_name)
    img = im_np[:,:,:3] # strip alpha channel
    img = cv.cvtColor(img,cv.COLOR_BGR2RGB) # convert numpy rgb to cv bgr
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5,) #minSize=(50,50))
    face_crops = []
    for (x,y,w,h) in faces:
        
        # scale rectangles

        scale_factor = 1.5

        w_s = w * scale_factor
        h_s = h * scale_factor
        x_diff = (w_s - w)/2
        y_diff = (h_s - h)/2

        x -= x_diff
        y -= y_diff
        w = w_s
        h = h_s

        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)

        # make crops
        crop_img = img[y:y+h, x:x+w]
        face_crops.append(crop_img)

    return face_crops
if __name__ == '__main__':
    crops = extractFacesFromScreen('./screenshots/', './screenshots/crops/')
    for c in crops:
        cv.imshow('e',c)
        cv.waitKey(0)
        cv.destroyAllWindows()