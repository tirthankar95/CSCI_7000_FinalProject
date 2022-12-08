# -*- coding: utf-8 -*-
"""mylava_env.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1PESHOczXNTOsc7mvOicuHAplvINqyCUI
"""

import numpy as np
from enum import IntEnum
import math

class lava:
  # Enumeration of possible actions
  class Actions(IntEnum):
    # Turn left, turn right, move forward
    left = 0
    right = 1
    forward = 2
    backward = 3
    pickup = 4

  #Enumeration of elements
  class Elements(IntEnum):
    neutral=0
    key=1
    door=2
    agent=4
    lava=8
    block=16

  def getVision(self):
    start_i=self.ipos-self.visi 
    start_j=self.jpos-self.visi
    obs=[]
    for _ in range(start_i,self.ipos+self.visi+1):
      obs_t=[]
      for __ in range(start_j,self.jpos+self.visi+1):
        obs_t.append(int(self.grid[_][__]))
      obs.append(obs_t)
    return [obs]

  def reset_m(self):
    self.grid=[[self.Elements.block for i in range(self.size+2*self.visi)] for j in range(self.size+2*self.visi)]
    for _ in range(self.visi,self.visi+self.size):
      for __ in range(self.visi,self.visi+self.size):
        self.grid[_][__]=self.Elements.neutral
    lavaCnt=math.ceil((self.size*self.size*self.difficulty)/100)
    while lavaCnt>0:
      self.lipos=np.random.randint(0,self.size)+self.visi
      self.ljpos=np.random.randint(0,self.size)+self.visi
      if self.grid[self.lipos][self.ljpos]==0:
        self.grid[self.lipos][self.ljpos]=self.Elements.lava
        lavaCnt-=1     
    while True:
      self.ipos=np.random.randint(0,self.size)+self.visi
      self.jpos=np.random.randint(0,self.size)+self.visi
      if self.grid[self.ipos][self.jpos]==0:
        self.grid[self.ipos][self.jpos]=self.Elements.agent
        break
    obs=self.getVision()
    self.oipos=self.ipos;self.ojpos=self.jpos
    self.okeyiPos=self.keyiPos;self.okeyjPos=self.keyjPos 
    self.gridOrig=self.grid
    return obs,0,False,(self.ipos,self.jpos)

  def reset_prev(self):
    self.grid=self.gridOrig
    self.ipos=self.oipos;self.jpos=self.ojpos
    self.grid[self.ipos][self.jpos]=self.Elements.agent
    self.agentMask=self.Elements.agent
    obs=self.getVision()
    return obs,0,False,(self.ipos,self.jpos)
    
  def __init__(self,size=4,visi=2,difficulty=10):
    self.delta=[[0,-1],[0,1],[-1,0],[1,0],[0,0]]	
    self.size=size	
    self.visi=visi
    self.difficulty=difficulty
    self.agentMask=self.Elements.agent	
    	
  def step_m(self,action):	
    reward=-1	
    terminate=False	
    self.grid[self.ipos][self.jpos]&=~(self.agentMask)	
    ic=self.ipos+self.delta[action][0]	
    jc=self.jpos+self.delta[action][1]	
    if self.grid[ic][jc]!=self.Elements.block:	
      self.ipos=ic	
      self.jpos=jc 	
    if self.Actions.pickup==action and self.grid[ic][jc]==self.Elements.key:	
      self.agentMask=(self.Elements.agent|self.Elements.key)	
    if self.grid[ic][jc]==self.Elements.door and self.agentMask==(self.Elements.agent|self.Elements.key):	
      terminate=True 	
      reward=10	
    if self.grid[ic][jc]==self.Elements.lava:	
      terminate=True 	
      reward=-10	
    self.grid[self.ipos][self.jpos]|=self.agentMask	
    obs=self.getVision()	
    return obs, reward, terminate, (self.ipos,self.jpos)	

  def printEnv(self):	
    for r in self.grid:	
      for c in r:	
        if c==self.Elements.block:	
          continue	
        print(int(c),end=" ")	
      print()	
    print()

def test():
  numActions=5 #maximum there are 5 actions.
  env=lava()
  state_next, reward, done, _ =env.reset_m()
  #env.printEnv()
  steps=1000
  c_reward=0
  while not done and steps>0:
    action=np.random.randint(0,numActions)
    # print(action)
    # print()
    state_next, reward, done, _ =env.step_m(action)
    # env.printEnv()
    # print()
    c_reward+=reward
    steps-=1
  print(c_reward)
#test()