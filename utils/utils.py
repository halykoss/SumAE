import torch
import torchvision
import matplotlib.pyplot as plt # plotting library
import glob
from PIL import Image
from pretrained_model.MNIST_model import Net

def plot_ae_outputs(encoder, decoder, device, test_dataset, epoch, directory, n=5):
    fig = plt.figure(figsize=(10,7))

    for i in range(n):
      ax = plt.subplot(4,n,i+1)
      img_stacked = torch.unbind(test_dataset.get(i)[0])
      img = img_stacked[0].unsqueeze(0).to(device)
      sec_img = img_stacked[1].unsqueeze(0).to(device)
      img_to = test_dataset.get(i)[0].unsqueeze(0).to(device)
      with torch.no_grad():
         rec_img  = decoder(encoder(img_to))[0]
      img_dec_stacked = torch.unbind(rec_img)
      f_img  = img_dec_stacked[0].unsqueeze(0).to(device)
      s_img  = img_dec_stacked[1].unsqueeze(0).to(device)
      plt.imshow(img.cpu().squeeze().numpy(), cmap='gist_gray')
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)  
      if i == n//2:
        ax.set_title('First number (n1) ')
      ax = plt.subplot(4, n, i + 1 + n)
      plt.imshow(sec_img.cpu().squeeze().numpy(), cmap='gist_gray')  
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)  
      if i == n//2:
         ax.set_title('Second number (n2) ')
      ax = plt.subplot(4, n, i + 1 + 2 * n)
      plt.imshow(f_img.cpu().squeeze().numpy(), cmap='gist_gray')  
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)  
      if i == n//2:
         ax.set_title('First digit (n1 + n2)')
      ax = plt.subplot(4, n, i + 1 + 3 * n)
      plt.imshow(s_img.cpu().squeeze().numpy(), cmap='gist_gray')  
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)  
      if i == n//2:
         ax.set_title('Second digit (n1 + n2)')
    
    fig.suptitle("SumAE - Epoch " + str(epoch))
    if epoch < 10:
        plt.savefig(directory + 'step_0' + str(epoch) + '.png')
    else:
        plt.savefig(directory + 'step_' + str(epoch) + '.png')
    plt.show()

def generate_gif():
   # filepaths
   fp_in = "outputs/step_*.png"
   fp_out = "outputs/image.gif"

   # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
   img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]
   img.save(fp=fp_out, format='GIF', append_images=imgs,
            save_all=True, duration=600, loop=0)

### Testing function
### Controllare che la somma sia corretta
def test_acc(encoder, decoder, device, dataloader):
    # Set evaluation mode for encoder and decoder
    encoder.eval()
    decoder.eval()
    model = Net()
    model.load_state_dict(torch.load("pretrained_model/model.pth"))
    correct = 0
    total = 0

    with torch.no_grad(): # No need to track the gradients
        for image_batch, (real_label, y_batch) in dataloader:
            # Move tensor to the proper device
            image_batch = image_batch.to(device)
            y_batch = y_batch.to(device)
            # Encode data
            encoded_data = encoder(image_batch)
            # Decode data
            decoded_data = decoder(encoded_data)
            for idx, img_in in enumerate(image_batch):
                #img_enc = torch.unbind(img_in)
                img_dec_stacked = torch.unbind(decoded_data[idx])
                # in_1 = model(img_enc[0].unsqueeze(0).unsqueeze(0).to("cpu")).data.max(1, keepdim=True)[1]
                # in_2 = model(img_enc[1].unsqueeze(0).unsqueeze(0).to("cpu")).data.max(1, keepdim=True)[1]
                in_3 = model(img_dec_stacked[0].unsqueeze(0).unsqueeze(0).to("cpu")).data.max(1, keepdim=True)[1]
                in_4 = model(img_dec_stacked[1].unsqueeze(0).unsqueeze(0).to("cpu")).data.max(1, keepdim=True)[1]
                if (real_label[0][idx] + real_label[1][idx]) == in_3.item() *  10 + in_4.item():
                    correct += 1
                total += 1
    
    return correct / total * 100
            
