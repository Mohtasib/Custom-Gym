import numpy as np
import copy
import time
import sys
import pybullet as p
from scipy.spatial import distance

class Robot():

    def __init__(self, pepper):
        self.joint_parameters = dict()
        self.percentage_speed = 1
        self.joint_data = pepper.joint_dict.items()
        self.initJointParameters()

    #initialise the dictionary of the angles
    def initJointParameters(self):
        for name, joint in self.joint_data:
            if "Finger" not in name and "Thumb" not in name:
                self.joint_parameters[name] = 0
    

    #set an angle for a given joint in radians
    def setAngleParameter(self, jointName, angleValue):

      lowerLimit, upperLimit = self.getJointRangeValue(jointName)

      if angleValue >= lowerLimit and angleValue <= upperLimit :
        self.joint_parameters[jointName] = angleValue
      else:
        print("[Lower, Upper] limit of " + jointName + " is ["+str(lowerLimit)+", "+str(upperLimit)+"]")
        sys.exit(1)

    #get the lower and upper angle values for a given joint
    def getJointRangeValue(self, jointName):
      if jointName not in self.joint_parameters.keys():
            print("Error, joint "+ str(jointName) +" does not exist !")
            sys.exit(1)
      else:
        for name, joint in self.joint_data:
          if jointName == name :
            return joint.lower_limit, joint.upper_limit



    #set the value of the max speed percentage for the robot mvts
    def setPercentageSpeed(self, speed):
        if speed >= 0.0 and speed <=1.0:
            self.percentage_speed = speed
        else:
            print("Error, speed percentage " +str(speed)+ " is not acceptable ! range is [0, 1]")
            sys.exit(1)

    def getPercentageSpeed(self):
        return self.percentage_speed

    #wait for the robot to end his move
    def waitEndMove(self, pepper):
        oldAngles = pepper.getAnglesPosition(list(self.joint_parameters.keys()))
        currentAngles = np.zeros(17)
        start = time.time()
        now = start
        while list(np.around(np.array(oldAngles),1)) != list(np.around(np.array(currentAngles),1)) and (now - start) < 5 :
            oldAngles= copy.deepcopy(currentAngles)
            currentAngles = pepper.getAnglesPosition(list(self.joint_parameters.keys()))
            now = time.time()
            time.sleep(0.02)

    #sets the angles to the default position
    def resetAngles(self):
      self.setAngleParameter("RShoulderPitch",1.55)
      self.setAngleParameter("RShoulderRoll",-0.24)
      self.setAngleParameter("RElbowRoll",0.02)
      self.setAngleParameter("RElbowYaw",1.23)
      self.setAngleParameter("RWristYaw",0.03)
      self.setAngleParameter("RHand",0.58)
      self.setAngleParameter("LShoulderPitch",1.55)
      self.setAngleParameter("LShoulderRoll",0.24)
      self.setAngleParameter("LElbowRoll",-0.02)
      self.setAngleParameter("LElbowYaw",-1.23)
      self.setAngleParameter("LWristYaw",-0.03)
      self.setAngleParameter("LHand",0.58)
      self.setAngleParameter("HeadPitch",0.14)
      self.setAngleParameter("HeadYaw",0.0)

    def customized_stand_posture(self):
      self.setAngleParameter("HeadPitch",0.14)
      self.setAngleParameter("HeadYaw",0.0)
      self.setAngleParameter("HipPitch",-0.1)

    #get the distance between the hands and the object to pick up
    def getDistHands(self,leftHandPos,rightHandPos,objPos):
        dstR = distance.euclidean(rightHandPos, objPos[0])
        dstL = distance.euclidean(leftHandPos,  objPos[0])
        print("Right hand dist : " + str(dstR))
        print("Left hand dist : "  + str(dstL))
        return [dstL, dstR]
