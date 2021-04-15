import torch

def tensor_convertAngleToVector(yaw, pitch, roll):
    p, y, r = pitch, yaw, roll
    Rotate_matrix = torch.tensor([
        [torch.cos(y)*torch.cos(r), -torch.cos(y)*torch.sin(r), torch.sin(y)],
        [torch.cos(p)*torch.sin(r)+torch.cos(r)*torch.sin(p)*torch.sin(y), torch.cos(p)*torch.cos(r)-torch.sin(p)*torch.sin(y)*torch.sin(r), -torch.cos(y)*torch.sin(p)],
        [torch.sin(p)*torch.sin(r)-torch.cos(p)*torch.cos(r)*torch.sin(y), torch.cos(r)*torch.sin(p)+torch.cos(p)*torch.sin(y)*torch.sin(r), torch.cos(p)*torch.cos(y)]
    ], requires_grad=True).type(torch.FloatTensor)
    Rotate_matrix = Rotate_matrix / torch.norm(Rotate_matrix)
    Rotate_matrix = torch.flatten(torch.t(Rotate_matrix))

    return Rotate_matrix

def tensor_convertListAngleToVector(yaws, pitchs, rolls):
    Rotate_matrixs = []
    for yaw, pitch, roll in zip(yaws, pitchs, rolls):
        Rotate_matrix = tensor_convertAngleToVector(yaw, pitch, roll)
        Rotate_matrixs.append(Rotate_matrix.view(1, Rotate_matrix.size()[0]))
    
    return torch.cat(Rotate_matrixs, axis=0)
