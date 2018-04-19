# -*- coding: utf-8 -*-
import numpy as np
import cv2
import itertools
import tensorflow as tf
import os
from PIL import Image
import pickle
import pprint
cls = ['ApplyEyeMakeup', 'ApplyLipstick', 'Archery', 'BabyCrawling', 'BalanceBeam',
       'BandMarching', 'BaseballPitch', 'Basketball', 'BasketballDunk', 'BenchPress', 'Biking',
       'Billiards', 'BlowDryHair', 'BlowingCandles', 'BodyWeightSquats', 'Bowling', 'BoxingPunchingBag',
       'BoxingSpeedBag', 'BreastStroke', 'BrushingTeeth', 'CleanAndJerk', 'CliffDiving', 'CricketBowling',
       'CricketShot', 'CuttingInKitchen', 'Diving', 'Drumming', 'Fencing', 'FieldHockeyPenalty',
       'FloorGymnastics', 'FrisbeeCatch', 'FrontCrawl', 'GolfSwing', 'Haircut', 'Hammering', 'HammerThrow',
       'HandStandPushups', 'HandstandWalking', 'HeadMassage', 'HighJump', 'HorseRace', 'HorseRiding',
       'HulaHoop', 'IceDancing', 'JavelinThrow', 'JugglingBalls', 'JumpingJack', 'JumpRope', 'Kayaking',
       'Knitting', 'LongJump', 'Lunges', 'MilitaryParade', 'Mixing', 'MoppingFloor', 'Nunchucks', 'ParallelBars',
       'PizzaTossing', 'PlayingCello', 'PlayingDaf', 'PlayingDhol', 'PlayingFlute', 'PlayingGuitar', 'PlayingPiano',
       'PlayingSitar', 'PlayingTabla', 'PlayingViolin', 'PoleVault', 'PommelHorse', 'PullUps', 'Punch', 'PushUps',
       'Rafting', 'RockClimbingIndoor', 'RopeClimbing', 'Rowing', 'SalsaSpin', 'ShavingBeard', 'Shotput',
       'SkateBoarding', 'Skiing', 'Skijet', 'SkyDiving', 'SoccerJuggling', 'SoccerPenalty', 'StillRings',
       'SumoWrestling', 'Surfing', 'Swing', 'TableTennisShot', 'TaiChi', 'TennisSwing', 'ThrowDiscus',
       'TrampolineJumping', 'Typing', 'UnevenBars', 'VolleyballSpiking', 'WalkingWithDog', 'WallPushups', 'WritingOnBoard', 'YoYo']

UCF101_path = '/home/yutong/Documents/cnn/UCF_101/'
trainlist01 = '/home/yutong/Documents/cnn/ucfTrainTestlist/trainlist01'
testlist01 = '/home/yutong/Documents/cnn/ucfTrainTestlist/testlist01'
imagepath = '/home/yutong/Documents/cnn/image'
global index
index = 0


def DownSample(image):  #transfer image from(240,320) to (120,160)
  w,h = image.shape
  img_new = np.zeros([w/2,h/2])
  img_fin = np.zeros([w/4,h/4])
  for i in range(w/2):
    for j in range(h/2):
      img_new[i,j] = image[2*i-1,2*j-1]
  for i in range(w/4):
    for j in range(h/4):
      img_fin[i,j] = img_new[2*i-1,2*j-1]
  return img_fin


def transfer3D():
  matrix = np.zeros([7, 60 * 80 * 16])
  for j in range(7):
    for i in range(16*j,16*j+16):
      path = 'image/' + str(i) + '.png'
      img = cv2.imread(path).astype(np.uint8)
      gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      hei = gray.shape[0]  # 60
      wid = gray.shape[1]  # 80
      size = wid * hei
      vector = np.reshape(gray, [1, size])
      matrix[j, j * 80 * 60: (j + 1) * 80 * 60] = vector[0, :]
      #matrix is (7,80*60*16)

  return matrix


def processvideo(path):
  count = 0
  cap = cv2.VideoCapture(path)
  video_length = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
  nFrames = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
  fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)  # zhen per second 25
  size = (int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)),
          int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)))  # (240,320)
  if video_length > 113:
    for i in range(nFrames):
      ret, frame = cap.read()  # ret bool;frame(240,320,3)
      if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # gray (240,320)    
        if (0<=i<=111):
          img = DownSample(gray)
          cv2.imwrite('image/' + str(i) + '.png', img)
          count =1   
      else:
        break
  return count


def Getdata():

  outX = file('data_x10.p', 'wb')
  outY = file('data_y10.p', 'wb')

  global index

  for a in ['trainlist_10']:

    for l in file('ucfTrainTestlist/' + a + '.txt'):
      f = l.split(" ")[0].split(
          "/")[1].split("\r\n")[0]  # f is name of video
      cl = cls.index(l.split("_")[1])  # cl is the number of video type

      cl_name = l.split("_")[1]  # cl_name is video type

      path = UCF101_path + cl_name + '/' + f

      count = processvideo(path) #each video
      if count == 1:
        frame_four = transfer3D() #get a matrix have 4 channel,each channel have 16 frames

        if(frame_four.shape == (7, 60*80*16)):
          y_ini = np.zeros(10)
          y_ini[cl] = 1
          y = y_ini
          v1 = frame_four[0].astype('uint8')
          v2 = frame_four[1].astype('uint8')
          v3 = frame_four[2].astype('uint8')
          v4 = frame_four[3].astype('uint8')
          v5 = frame_four[4].astype('uint8')
          v6 = frame_four[5].astype('uint8')
          v7 = frame_four[6].astype('uint8')
          
          WriteData(y,outY)
          WriteData(y,outY)
          WriteData(y,outY)
          WriteData(y,outY)
          WriteData(y,outY)
          WriteData(y,outY)
          WriteData(y,outY)

          WriteData(v1,outX)
          WriteData(v2,outX)
          WriteData(v3,outX)
          WriteData(v4,outX)
          WriteData(v5,outX)
          WriteData(v6,outX)
          WriteData(v7,outX)
  
      index += 1
      print index

  outX.close()
  outY.close()
  return
# main
Getdata()



