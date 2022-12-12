import numpy as np
from enum import IntEnum
import copy

class mixedEnv:
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
    doorN=32 #This door will open without a key.

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

  def reset(self,grid=None):
    assert(grid!=None)
    self.grid=[[self.Elements.block for i in range(self.size+2*self.visi)] for j in range(self.size+2*self.visi)]
    for _ in range(len(grid)):
      for __ in range(len(grid[_])):
        self.grid[self.visi+_][self.visi+__]=grid[_][__]
    for _ in range(len(self.grid)):
        for __ in range(len(self.grid[_])):
            if self.grid[_][__]&self.Elements.agent!=0:
                self.ipos=_
                self.jpos=__
                self.grid[self.ipos][self.jpos]=self.Elements.neutral
    self.agentMask=self.Elements.agent
    self.grid[self.ipos][self.jpos]=self.Elements.agent
    obs=self.getVision()
    self.oipos=self.ipos;self.ojpos=self.jpos
    return obs,0,False,(self.ipos,self.jpos)

  def __init__(self,size=4,visi=2):
    self.delta=[[0,-1],[0,1],[-1,0],[1,0],[0,0]]
    self.size=size
    self.visi=visi

  def step(self,action):
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
  numActions=5
  env3=mixedEnv()
  state_next, reward, done, _ =env3.reset(mixedGrid)
  env3.printEnv()
  steps=1000
  c_reward=0
  while not done and steps>0:
    action=np.random.randint(0,numActions)
    print(action)
    print()
    state_next, reward, done, _ =env3.step(action)
    env3.printEnv()
    print()
    c_reward+=reward
    steps-=1
  print(c_reward)
# test()