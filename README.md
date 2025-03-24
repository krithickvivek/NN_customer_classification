# Developing a Neural Network Classification Model

## AIM

To develop a neural network classification model for the given dataset.

## Problem Statement

An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model

![image](https://github.com/user-attachments/assets/c57a0545-8667-491f-8bc6-6a82c1907eb5)

## DESIGN STEPS

### STEP 1:
Import the necessary libraries and load the dataset into the program.
### STEP 2:
Preprocess the dataset by handling missing values (if any) and normalizing the data.
### STEP 3:
Split the dataset into training and testing sets to evaluate model performance.
### STEP 4:
Define the neural network architecture with input, hidden, and output layers.
### STEP 5:
Compile the model using a suitable loss function and optimizer for regression tasks.
### STEP 6:
Train the model using the training dataset and monitor loss reduction over epochs.
### STEP 7:
Plot the training loss versus iteration graph to analyze model convergence.
### STEP 8:
Evaluate the model performance using the testing dataset.
### STEP 9:
Use the trained model to make predictions on new data samples.
## PROGRAM

### Name: Krithick Vivekananda
### Register Number: 212223240075

```python
class PeopleClassifier(nn.Module):
    def __init__(self, input_size):
        super(PeopleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 8)
        self.fc4 = nn.Linear(8, 4)


    def forward(self, x):
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=F.relu(self.fc3(x))
        x=self.fc4(x)
        return x

```
```python
# Initialize the Model, Loss Function, and Optimizer
model =PeopleClassifier(input_size=X_train.shape[1])
criterion =nn.CrossEntropyLoss()
optimizer =optim.Adam(model.parameters(),lr=0.001)

```
```python
def train_model(model,train_loader,criterion,optimizer,epochs):
  for epoch in range(epochs):
    model.train()
    for X_batch,y_batch in train_loader:
      optimizer.zero_grad()
      outputs=model(X_batch)
      loss=criterion(outputs,y_batch)
      loss.backward()
      optimizer.step()

  if(epoch+1)%10==0:
    print(f'Epoch [{epoch+1}/{epochs}],Loss:{loss.item():.4f}')
```



## Dataset Information

![image](https://github.com/user-attachments/assets/3a6cc943-e166-4509-aaa4-5399adc38efd)

## OUTPUT

### Confusion Matrix
![image](https://github.com/user-attachments/assets/9e7841a5-9f22-4615-8a13-4a41e1aa4906)
### Classification Report
![image](https://github.com/user-attachments/assets/e4f5f3a0-3717-47c0-b323-cf36c0300444)
### New Sample Data Prediction
![image](https://github.com/user-attachments/assets/f64f54f6-d2eb-42c5-a2b6-5a5d4794144b)
## RESULT
The neural network classification model was successfully developed and trained. The model accurately classified new customers into segments (A, B, C, D) based on the given dataset, as verified by the confusion matrix and classification report.
