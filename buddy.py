from selenium import webdriver
from selenium import common
import cv2 as cv
from time import sleep
import random
import extract_face_img as e
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
from torch.autograd import Variable

def rate_crop_with_model(crop, model_ft):
    c = crop
    d = Image.fromarray(c)
    data_transform = transforms.Compose([
    transforms.Scale(350),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    img = data_transform(d).float()
    img = Variable(img, requires_grad=True)
    img = img.unsqueeze(0)
    outputs = model_ft(img)
    _, preds = torch.max(outputs.data, 1)
    return preds.item()

driver = webdriver.Chrome("/Users/maxshashoua/Documents/Developer/chromedriver")

driver.get("https://bumble.com/get-started")

# sign in using fb
success = False
while not success:
    try:
        driver.find_element_by_xpath('//*[@id="main"]/div/div[1]/div[2]/main/div/div[2]/form/div[1]/div').click()
        success = True
    except (common.exceptions.NoSuchElementException, common.exceptions.ElementClickInterceptedException):
        continue
# driver.find_element_by_xpath('//*[@id="main"]/div/div[1]/div[2]/main/div/div[2]/form/div[1]/div').click()        
main_page=driver.current_window_handle

print('loading model...')
model_path = '/Users/maxshashoua/Documents/Developer/faces/models/2019-09-07'
model_ft = torch.load(model_path)
model_ft.eval()
criterion = nn.CrossEntropyLoss()
print('model is ready...')

s = input('press enter to continue')
l = [True, False, False, True, True, False]




for i in range(1000): 
    print('profile {}'.format(i))
    # for yes in l:
    cv.destroyAllWindows()
    crops = e.extractFacesFromScreen('./screenshots/', './screenshots/crops/')
    ratings = []
    try:
        for c in crops:
            r = rate_crop_with_model(c, model_ft)
            ratings.append(r)
            # cv.imshow(str(r), c)
            # sleep(1)
    except ValueError:
        continue
    if len(ratings)==0:
        print('no face found')
        driver.find_element_by_xpath('//*[@id="main"]/div/div[1]/main/div[2]/div/div/span/div[2]/div/div[2]/div/div[1]/div/span/span').click()
        continue
    rating = sum(ratings)/len(ratings)
    print(ratings, rating)
    yes = True if rating>=3 else False
    sleep(random.randint(1, 3))
    stuck = True
    stuck_c = 0
    while stuck:
        stuck_c += 1
        if stuck_c > 100:
            print("broke stuck")
            break
        try:
            
            if yes: driver.find_element_by_xpath('//*[@id="main"]/div/div[1]/main/div[2]/div/div/span/div[2]/div/div[2]/div/div[3]/div/span/span').click()
            else: driver.find_element_by_xpath('//*[@id="main"]/div/div[1]/main/div[2]/div/div/span/div[2]/div/div[2]/div/div[1]/div/span/span').click()
            stuck = False
        except common.exceptions.NoSuchElementException:
            pass

    # check for match page:
    try:
        driver.find_element_by_xpath('//*[@id="main"]/div/div[1]/main/div[2]/article/div/footer/div/div[2]/div/span/span/span').click()
    except common.exceptions.NoSuchElementException:
        pass


    
sleep(2)

driver.quit()
