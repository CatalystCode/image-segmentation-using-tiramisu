import os
import numpy as np
import imageio
import azure.storage.blob as azureblob
from PIL import Image
import glob


_STORAGE_ACCOUNT_NAME = 'yourstgacct'
_STORAGE_ACCOUNT_KEY = 'youraccountkey=='
_STORAGE_INPUT_CONTAINER = 'data'
_PREFIX_='yoursubfolder/foreground_segmented/'
_SAVE_DIR='save_dir/'
filepath=os.path.join(os.path.realpath('.'),'save_dir')
npy_path=os.path.join(filepath,'12031854sd.quick.npy')
msk_path=os.path.join(filepath,'12031854sd.quick.mask.gif')


def open_image(fn,img_sz = (516,516)):
    return np.array(Image.open(fn).resize(img_sz, Image.NEAREST))

def list_files_in_container(blob_client,container,prefix):
    blobs = [blob.name for blob in blob_client.list_blobs(container) if prefix in blob.name and 'npy' in blob.name]
    return blobs

def download_blob(blob_client, container,prefix,savedir, blob):
    blob_client.get_blob_to_path(container_name=container, blob_name= blob, file_path= savedir+blob.replace(prefix,''))
    return savedir+blob.replace(prefix,'')

def upload_blob(blob_client, container, blobname, localfile):
    blob_client.create_blob_from_path(container,
                                    blobname,
                                    localfile)

def npy_to_gif(npy_array_path, gif_path):
    np_array=np.load(npy_array_path)*255
    imageio.imsave(gif_path,np_array)
    np_array2=imageio.imread(gif_path)

def gif_to_3dnpy_and_upload(mask_dir,blob_vfolder , blob_client, container):
    lnames = sorted(glob.glob(mask_dir+'/*.gif'))
    labels = np.stack(([open_image(fn) for fn in lnames]))
    labels_3d = labels[:, :, :, None] * np.ones(3, dtype=int)[None, None,None, :]
    for i,fn in enumerate(lnames):
        npa = labels_3d[i,:,:,:]
        npname = fn.replace('.mask.gif','.3dmask.npy').replace('_Mask_Img/','_Mask_Npy/')
        np.save(npname,npa)
        blobname = blob_vfolder + fn.replace('.mask.gif','.3dmask.npy').split('/')[-1]
        upload_blob(blob_client, container, blobname, npname)

def gif_to_3dnpy(mask_dir,npy_dir, sz):
    lnames = sorted(glob.glob(mask_dir+'/*.gif'))
    labels = np.stack(([open_image(fn, img_sz=sz) for fn in lnames]))
    labels_3d = labels[:, :, :, None] * np.ones(3, dtype=int)[None, None,None, :]
    for i,fn in enumerate(lnames):
        npa = labels_3d[i,:,:,:]
        npname = fn.replace('.mask.gif','.3dmask.npy').replace(mask_dir,npy_dir)
        np.save(npname,npa)
        

if __name__ == '__main__':
    #npy_to_gif(npy_path,msk_path)
    blob_client = azureblob.BlockBlobService(
        account_name=_STORAGE_ACCOUNT_NAME,
        account_key=_STORAGE_ACCOUNT_KEY)
    blobs=list_files_in_container(blob_client, _STORAGE_INPUT_CONTAINER, _PREFIX_)
    print('got {} blobs'.format(len(blobs)))

    i=0
    for blob in blobs:
        i=i+1
        npy_array=download_blob(blob_client,_STORAGE_INPUT_CONTAINER,_PREFIX_,_SAVE_DIR,blob)
        npy_to_gif(npy_array, npy_array.replace('.npy','.mask.gif'))
        blob_name=_PREFIX_+npy_array.replace('.npy','.mask.gif').replace(_SAVE_DIR,'')
        upload_blob(blob_client,_STORAGE_INPUT_CONTAINER,blob_name,npy_array.replace('.npy','.mask.gif'))
        os.remove(npy_array)
        os.remove(npy_array.replace('.npy','.mask.gif'))
        print('progress {} out of {}'.format(i,len(blobs)))
