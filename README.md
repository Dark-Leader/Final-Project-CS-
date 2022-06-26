# NoteRecognition

Welcome to NoteRecognition - The app that teaches you how to play the piano!

# How to use the app:
1. the server needs to run on a linux based system.
2. create a conda env with the given requirements.txt file.
3. download FluidSynth with the following commands:

sudo apt-get update

sudo apt-get install fluidsynth

4. download the Pytorch model from: https://drive.google.com/file/d/1U-i_oH_MSRE7TMF597jHNevtVyDTGAP0/view?usp=sharing
and place it inside the following path: "algorithm/ML"
6. optional - if you wish the train the model differently - I classified manually around 3000 images and made the following dataset: https://drive.google.com/drive/folders/13XoVZcMSgTMSWky8C1QHETadLqnW6v1S?usp=sharing
you may do as you wish with it - if you want to classify more symbols - e.g support of black piano keys then you will need to add said classes to the dataset.
6. run the flask app with: "python3 main.py"
7. go to "localhost:5000" and upload an image to the server - image of the melody you wish to learn how to play - I provide you with images inside the test images folder.
8. press the play button to see the piano animation and use the play, pause, reset buttons to control the playback and you can download the output image or the output the server made from the detections from the image provided.


I made a short video to illustrate how to run it: https://youtu.be/_hcmd1JnRAI


supported notes:
you may check the dataset classes to better understand what the model can classify correctly.
1. whole.
2. half, dotted half.
3. quarter, dotted quarter.
4. eighth, dotted eighth.
5. sixteenth, dotted sixteenth.
6. rest whole.
7. rest half.
8. rest quarter, dotted quarter rest.
9. rest eighth, dotted rest eighth.
10. rest sixteenth, dotted rest sixteenth.

unsupported notes:
1. black piano keys - sharp or 'b': 

![ApplicationFrameHost_YF5FxQpRXq](https://user-images.githubusercontent.com/53357564/175818046-26df9651-f78b-465e-a2da-5b8b8f95eafe.png)
![ApplicationFrameHost_wiL0vyEo6i](https://user-images.githubusercontent.com/53357564/175818110-27633545-c20a-475a-8737-b691807f9ee5.png)


2. conneted notes with different durations e.g:

![ApplicationFrameHost_nzoqKMfENV](https://user-images.githubusercontent.com/53357564/175818138-5bf9c811-066e-469f-b3c8-8eaff2430627.png)


3. notes longer than a full rest.
4. tie notes:


![ApplicationFrameHost_8AAiscKcNR](https://user-images.githubusercontent.com/53357564/175818233-179de638-9a14-4ab8-a826-80078e43911a.png)
5. gracenotes.
