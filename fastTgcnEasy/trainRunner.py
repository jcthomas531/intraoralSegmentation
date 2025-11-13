

#note 1
#in the future i want to move this to just a bash script where python is opened
#and train.py is imported and the function is ran but now to keep things easy
#i am going to use the same framework that I had set up in fastTgcn



#should alredy be in the proper working directory
import train
train.fastTgcnEasy(arch = "l",
                   testPath = "Y:\\dissModels\\intraoralSegmentation\\IOSSegData\\test-L-Small",
                   trainPath = "Y:\\dissModels\\intraoralSegmentation\\IOSSegData\\train-L-Small",
                   batch_size = 1,
                   k = 32,
                   numWorkers = 8,
                   epochs = 31)





