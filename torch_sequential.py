import torch
from torch import nn, optim

class Classifier():
    def __init__(self, model, criterion, optimizer, cuda=False):
        
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
         
        if cuda:
            self.model.cuda()

    def fit(self, trainloader, testloader=None, epochs=5, print_every=1):
        ''' train on the trainloader dataset and validate on testloader dataset'''
        steps = 0
        for e in range(epochs):
            # Model in training mode, dropout is on
            self.model.train()
            running_loss = 0
            for images, labels in trainloader:
                # Batch training
                steps += 1

                # Flatten images into a input_size long vector => images: batch_size * input_size
                images.resize_(images.size()[0], images.size()[-1]*images.size()[-2])
                
                # set grads to zero to avoid accumulating them
                self.optimizer.zero_grad()

                # forward pass +  backpropagation + update weights
                output = self.model.forward(images)
                loss = self.criterion(output, labels)
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
                      "Training Loss: {:.3f}.. ".format(running_loss / print_every),
                      "Test Loss: {:.3f}.. ".format(test_loss),
                      "Test Accuracy: {:.3f}".format(accuracy))

    def evaluate(self, testloader):
        accuracy = 0
        test_loss = 0
        for images, labels in testloader:
            # Flatten images into a input_size long vector => images: batch_size * input_size
            images.resize_(images.size()[0], images.size()[-1]*images.size()[-2])

            output = self.model.forward(images)
            test_loss += self.criterion(output, labels).item()

            ## Calculating the accuracy
            # Model's output is log-softmax, take exponential to get the probabilities
            ps = torch.exp(output)
            # Class with highest probability is our predicted class, compare with true label
            equality = (labels.data == ps.max(1)[1])
            # Accuracy is number of correct predictions divided by all predictions, just take the mean
            accuracy += equality.type_as(torch.FloatTensor()).mean()

        return test_loss/len(testloader), accuracy/len(testloader)

    def predict(self, testloader):
        pass
