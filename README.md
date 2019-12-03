# bumble_bot
Automated API-less bumble robot using vision and browser automation (pytorch, opencv, selenium).

Quick one day project for the online dater on the go (el oh el)

The automation is buggy. Bumble does not want people to be able to do this. How to do and how it does:

how to run:
 1. run python3 buddy.py
  the login page will open up
 2. login with your credentials.
 3. watch it go!
 
how it works:

after loging in, make sure that you don't have any images of faces visible except for the bumble webpage because...

OpenCV finds all the faces on your screen and runs them through a pytorch resnet trained on a small dateset of attractivness-rated face images that I found online. It will average the scores together and swipe right (selenium automation searching for the right button, sometimes breaks) if the avg  score is above some threshold specified in the code. 

Things that go right:
  - facial detection works so well and was very easy to implement. openCV for the win!
  - pytorch makes it so easy to set up and train neural nets. 
  
Things that go wrong:
  - detecting faces outside of the webpage
  - dataset is small and not good enough and just kinda strange...
  - selenium has trouble pushing the right buttons
  
funny project though!
