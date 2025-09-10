import numpy as np
import torch
import torch.nn as nn

import torch.optim as optim
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
import open3d as o3d

"""
Multi-Layer Perceptron (MLP) model for ground surface estimation.
This neural network takes 2D coordinates (x,y) as input and predicts the z-coordinate (height).
"""
class MLP(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, output_dim=1):
        super(MLP, self).__init__()
        
        # Define network layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # Input to hidden layer
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # Hidden to hidden layer
        self.fc3 = nn.Linear(hidden_dim, output_dim)  # Hidden to output layer
        
        # Activation function
        self.activation = nn.ReLU()  # ReLU activation for non-linearity

    def forward(self, x):
        # Forward pass through the network
        x = self.activation(self.fc1(x))  # First hidden layer
        x = self.activation(self.fc2(x))  # Second hidden layer
        x = self.fc3(x)  # Output layer (no activation for regression tasks)
        return x

def optim_loss(prediction,target):
    c1 =nn.MSELoss()
    c2 = nn.HuberLoss(delta=0.7)
    above_mask = prediction<target
    #print(prediction)
    return c2(prediction[above_mask],target[above_mask])+c1(prediction[~above_mask],target[~above_mask]) 


"""
MLP-based ground surface model.
This class wraps the MLP model with training and inference methods for ground surface estimation.
"""
class MLP_model:
    
    def __init__(self, hidden_dim=64,
                 pc_range=[-30, -30, -3, 30, 30, 2],
                 batch_size=512,
                 lr=0.005,
                 device='cuda:0'):
        self.device = device
        self.model = MLP(input_dim=2, hidden_dim=hidden_dim, output_dim=1).to(device)
        self.pc_range = pc_range
        self.batch_size=batch_size
        self.lr=lr
    
    def train(self,pcd,test_size=0.2):
        '''
        input: pcd [N,3]
        '''
        
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)  # Adam optimizer
        scheduler = StepLR(optimizer, step_size=200, gamma=0.4)
        
        mask = (pcd[:, 2] > self.pc_range[2]) & (pcd[:, 2] < self.pc_range[5]) \
            &(pcd[:,0]> self.pc_range[0]) &(pcd[:,0]< self.pc_range[3]) \
            &(pcd[:,1]> self.pc_range[1]) &(pcd[:,1]< self.pc_range[4])
        
        pcd = pcd[mask]
        print('Point Cloud Size:',pcd.shape)
        X_train, X_test, y_train, y_test = train_test_split(pcd[:,:2], pcd[:,2], test_size=test_size, random_state=42)
        X_train = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_train = torch.tensor(y_train, dtype=torch.float32).to(self.device)
    
        X_test = torch.tensor(X_test, dtype=torch.float32).to(self.device)
        y_test = torch.tensor(y_test, dtype=torch.float32).to(self.device)

        # Training loop
        num_epochs = 10
        for epoch in range(num_epochs):
            # Forward pass
            num_sample = len(X_train)
            num_batch = num_sample//self.batch_size
            for i in range(num_batch+1):
                if i==num_batch:
                    start, end = num_batch*self.batch_size, num_sample
                else:
                    start, end = i*self.batch_size, (i+1)*self.batch_size
                X_input,y_input = X_train[start:end,:], y_train[start:end]
                outputs = self.model(X_input)
                loss = optim_loss(outputs, y_input.unsqueeze(1))
                optimizer.zero_grad()
                loss.backward()  
                optimizer.step()

            scheduler.step()
            
            # Print loss every 10 epochs
            if (epoch + 1) % 1 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")
        
            print('Finish Training!')
            
    def inference(self, location):
        location = torch.tensor(location, dtype=torch.float32).to(self.device)
        return self.model(location[:,:2])
    
    def test(self,pcd,threshold=0.15):        
        ground_point = []
        all_mask = []
        
        mask = (pcd[:, 2] > self.pc_range[2]) & (pcd[:, 2] < self.pc_range[5]) \
            &(pcd[:,0]> self.pc_range[0]) &(pcd[:,0]< self.pc_range[3]) \
            &(pcd[:,1]> self.pc_range[1]) &(pcd[:,1]< self.pc_range[4])
        
        pcd = pcd[mask]
        
        X_test,y_test = pcd[:,:2], pcd[:,2]
        X_test = torch.tensor(X_test, dtype=torch.float32).to(self.device)
        y_test = torch.tensor(y_test, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():        
            num_sample = len(X_test)
            num_batch = num_sample//self.batch_size
            for i in range(num_batch+1):
                if i==num_batch:
                    start, end = num_batch*self.batch_size, num_sample
                else:
                    start, end = i*self.batch_size, (i+1)*self.batch_size
                X_input,y_input = X_test[start:end,:], y_test[start:end]
                outputs = self.model(X_input)
                
                mask = torch.abs(outputs[:,0]-y_input)<threshold 
                points = torch.cat([X_input,y_input.unsqueeze(1)],dim=-1)
                all_mask.append(mask.cpu().numpy())
                ground_point.append(points[mask].cpu().numpy())
                
        ground_point = np.concatenate(ground_point,axis=0)
        all_mask = np.concatenate(all_mask,axis=0)
        return ground_point, all_mask
    
"""
RANSAC-based ground plane estimation model.
This class implements RANSAC algorithm to fit a plane to point cloud data.
"""
class RANSAC_model:
    def __init__(self,
                 pc_range=[-30, -30, -3, 30, 30, 2],
                 ):
        # Point cloud range filter [min_x, min_y, min_z, max_x, max_y, max_z]
        self.pc_range = pc_range
    
    def train(self,pcd):
        mask = (pcd[:, 2] > self.pc_range[2]) & (pcd[:, 2] < self.pc_range[5]) \
            &(pcd[:,0]> self.pc_range[0]) &(pcd[:,0]< self.pc_range[3]) \
            &(pcd[:,1]> self.pc_range[1]) &(pcd[:,1]< self.pc_range[4])
        pcd = pcd[mask]
        
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(pcd)
        
        plane_model, inliers = pc.segment_plane(distance_threshold=0.25, ransac_n=4, num_iterations=5000)
        self.model = plane_model
        ground_cloud = pc.select_by_index(inliers)
        ground_pc = np.asarray(ground_cloud.points)
        return ground_pc
    
    def inference(self,location):
        return (-self.model[3]-self.model[0]*location[0]-self.model[1]*location[1])/(self.model[2])