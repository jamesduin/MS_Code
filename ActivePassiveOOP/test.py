class Parent:        # define parent class
   parentAttr = 100
   def __init__(self,name):
      self.name = name
      self.car = 'big boy car'
      print("Calling parent constructor")

   def parentMethod(self):
      print('Calling parent method')

   def setAttr(self, attr):
      Parent.parentAttr = attr

   def getAttr(self):
       print("Parent attribute :", Parent.parentAttr)

   def printName(self):
       print(self.chair)

class Child(Parent): # define child class
    def __init__(self,name):
        Parent.__init__(self,name)
        self.chair = 'little chair'
        print("Calling child constructor")

    def childMethod(self):
        print('Calling child method')

    def printCar(self):
        print(self.car)


c = Child('Jerry')          # instance of child
c.childMethod()      # child calls its method
c.parentMethod()     # calls parent's method
c.setAttr(200)       # again call parent's method
c.getAttr()          # again call parent's method
c.printName()
c.printCar()