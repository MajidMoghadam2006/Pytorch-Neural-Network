import torch
from torch import nn, optim

class Classifier():
    def __init__(self, model, criterion, optimizer, cuda=False, flatten_input=False):
        
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.cuda = cuda
        self.flatten_input = flatten_input
         
        if self.cuda:
            self.model.cuda()

    def fit(self, trainloader, testloader=None, epochs=5, print_every=1):
        ''' train on the trainloader dataset and validate on testloader dataset'''
        
        steps = 0
        for e in range(epochs):
            # Model in training mode, dropout is on
            self.model.train()
            running_loss = 0
            for data, target in trainloader:
                
                # move tensors to GPU if CUDA is available
                if self.cuda:
                    data, target = data.cuda(), target.cuda()
                
                # Batch training
                steps += 1

                # Flatten images into a input_size long vector => images: batch_size * input_size
                if self.flatten_input:
                    data.resize_(data.size()[0], data.size()[-1]*data.size()[-2])
                
                # set grads to zero to avoid accumulating them
                self.optimizer.zero_grad()

                # forward pass +  backpropagation + update weights
                output = self.model.forward(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            if e % print_every == 0 and testloader is not None:
                # Validation
                # Model in inference mode, dropout is off
                self.model.eval()

                # Turn off gradients for validation, will speed up inference
                with torch.no_grad():
                    test_loss, accuracy = self.evaluate(testloader)

                print("Epoch: {}/{}.. ".format(e + 1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/len(trainloader)),
                      "Test Loss: {:.3f}.. ".format(test_loss),
                      "Test Accuracy: {:.3f}".format(accuracy))

    def evaluate(self, testloader):
        '''evaluate model accuracy and loss on test/validation dataset'''
        accuracy = 0
        test_loss = 0
        for data, target in testloader:
            
            # move tensors to GPU if CUDA is available
            if self.cuda:
                data, target = data.cuda(), target.cuda()
            
            # Flatten images into a input_size long vector => images: batch_size * input_size
            if self.flatten_input:
                data.resize_(data.size()[0], data.size()[-1]*data.size()[-2])

            output = self.model.forward(data)
            test_loss += self.criterion(output, target).item()

            ## Calculating the accuracy
            # Model's output is log-softmax, take exponential to get the probabilities
            ps = torch.exp(output)
            # Class with highest probability is our predicted class, compare with true label
            equality = (target.data == ps.max(1)[1])
            # Accuracy is number of correct predictions divided by all predictions, just take the mean
            accuracy += equality.type_as(torch.FloatTensor()).mean()

        return test_loss/len(testloader), accuracy/len(testloader)

    def predict(self, testloader):
        '''generate model predictions (probability vector) for the input dataset'''
        for data, target in testloader:
            
            # move tensors to GPU if CUDA is available
            if self.cuda:
                data, target = data.cuda(), target.cuda()
            
            # Flatten images into a input_size long vector => images: batch_size * input_size
            if self.flatten_input:
                data.resize_(data.size()[0], data.size()[-1]*data.size()[-2])

            output = self.model.forward(data)
            test_loss += self.criterion(output, target).item()

            ## Calculating the accuracy
            # Model's output is log-softmax, take exponential to get the probabilities
            ps = torch.exp(output)
        return ps
    
    def save_model(self, model_path):
        '''save torch model in model_path.pt'''
        torch.save(self.model.state_dict(), model_path + '.pt')
        
        
        
        
        
