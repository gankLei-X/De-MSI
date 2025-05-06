from preprocessing import *
from model import *
from torch.utils.data import DataLoader, TensorDataset, random_split

def DeNoising(oridata,peak,mono_iso,input_shape,mask,patch_size = 256):

    print('Training from mono-isotope to isotope...')

    m,n = input_shape
    epoch = 2000

    monoisoPeak = mono_iso[:,0]
    isoPeak = mono_iso[:,1]

    trainMat = Peak2Mat(oridata,peak,isoPeak,m,n)
    testMat = Peak2Mat(oridata,peak,monoisoPeak,m,n)

    for i in range(len(trainMat)):
        inputMatrix = np.hstack((trainMat[i,0].reshape(-1,1),testMat[i,0].reshape(-1,1)))
        returnMat = Thist_match(inputMatrix, m, n)
        trainMat[i,0] = returnMat[:,0].reshape(m,n)
        testMat[i,0] = returnMat[:,1].reshape(m,n)

    trainMat, _ = extract_patches_with_overlap(trainMat, patch_size = patch_size, step = 16)
    testMat, _ = extract_patches_with_overlap(testMat, patch_size = patch_size, step = 16)

    trainMat = torch.from_numpy(trainMat).float()
    testMat = torch.from_numpy(testMat).float()

    dataset = TensorDataset(trainMat, testMat)
    train_loader = DataLoader(dataset, batch_size = 128, shuffle=True)

    model = UNet(1, 1).cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr = 0.005)
    scheduler_unet = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epoch)
    criteron1 = torch.nn.L1Loss(reduction='mean')

    for i in range(epoch):
        model.train()
        loss_sum = 0
        for batch in train_loader:
            x,y = batch
            x = x.cuda()
            y = y.cuda()
            outputsx = model(x)
            loss = criteron1(outputsx[y != 0], y[y != 0])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()

        print("Epoch %d Train Loss:" % i + str(loss_sum / len(train_loader)))

        scheduler_unet.step()

    print('Interfacing the denoised ions...')

    trainMat = Peak2Mat(oridata,peak,peak,m,n)
    trainMat,patchNUMPer = extract_patches_with_overlap(trainMat,patch_size=patch_size,step=16)
    print(trainMat.shape)
    trainMat = torch.from_numpy(trainMat).float()

    return_matrix = np.zeros((len(trainMat),patch_size * patch_size))
    num = len(trainMat)

    mini_batch = 32
    for batch in range(num // mini_batch):
        with torch.no_grad():
            x = trainMat[batch * mini_batch : (batch + 1) * mini_batch]
            x = x.cuda()
            outputs= model(x)
            outputs = outputs.detach().cpu().numpy()
            return_matrix[batch * mini_batch : (batch + 1) * mini_batch] = outputs.reshape(mini_batch,patch_size * patch_size)

    with torch.no_grad():
        x = trainMat[(batch + 1) * mini_batch:]
        x = x.cuda()
        outputs = model(x)
        outputs = outputs.detach().cpu().numpy()
        return_matrix[(batch + 1) * mini_batch:] = outputs.reshape(len(x), patch_size * patch_size)

    return_matrix = return_matrix.reshape(len(return_matrix), 1, patch_size, patch_size)
    reconstructed_matrix = reconstruct_image_from_patches(return_matrix, input_shape, patchNUMPer,patch_size=patch_size,step=16)
    reconstructed_matrix = reconstructed_matrix.reshape(len(oridata[1]), len(oridata))
    reconstructed_matrix = np.transpose(reconstructed_matrix)
    reconstructed_matrix = reconstructed_matrix * mask.reshape(-1,1)

    return reconstructed_matrix
