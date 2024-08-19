import numpy as np
from keras.datasets import mnist

#MNIST DATA
def load_mnist():
    (x_train, y_train), (x_val, y_val) = mnist.load_data()
    x_train = x_train.astype(np.float32) / 255.
    # x_test = x_test.astype(np.float32) / 255.
    x_val = x_val.astype(np.float32) / 255.
    return x_train,y_train,x_val,y_val

#Plot Latent Class Distribution
def plot_classes(z,label,epoch,model_config):
    batch=model_config['batch_size']
    model_name=model_config['model_name']

    latent=np.array(z)
    # print(latent.shape)
    fig=plt.figure(figsize=(10, 10))
    plt.scatter(latent[:, 0], latent[:, 1], c=label)
    plt.colorbar()
    fig.savefig("{}/classes_{}_{}.png".format(model_name,epoch,batch))
    plt.clf()
    plt.close(fig)

#Plot Image Generated
def imagegrid(images,epoch,model_config):
    batch=model_config['batch_size']
    model_name=model_config['model_name']

    fig = plt.figure(figsize=[20, 20])
    # images = self.generateImages(decoder,100)
    for index,img in enumerate(images):
        img = img.reshape((28, 28))
        ax = fig.add_subplot(10, 10, index+1)
        ax.set_axis_off()
        ax.imshow(img, cmap="gray")
    fig.savefig("{}/grid_{}_{}.png".format(model_name,epoch,batch))
    plt.show()
    plt.close(fig)

def plot_losses(train_losses,val_losses,loss_name,model_config):
    epoch=model_config['ep']
    batch=model_config['batch_size']
    model_name=model_config['model_name']

    #train loss
    plt.plot(range(len(train_losses)),train_losses,c='r')
    #val loss
    plt.plot(range(len(val_losses)),val_losses,c='b')
    plt.savefig('./plot/{}_{}_{}_{}.png'.format(model_name,epoch,batch,loss_name))
    plt.clf()
