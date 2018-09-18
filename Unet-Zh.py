from zhixuhao.modelUnet import *
from zhixuhao.data import *

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"


data_gen_args = dict(
                    rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=False,
                    fill_mode='nearest'
                    )
#membrane
#reflex
myGene = trainGenerator(1,'membrane/train','image','label',data_gen_args, save_to_dir = None)
model = unet()
callbacks = []
model_checkpoint = ModelCheckpoint('membrane.hdf5', monitor='loss',verbose=1, save_best_only=True)
callbacks.append(model_checkpoint)
model.fit_generator(myGene,steps_per_epoch=300,epochs=1,callbacks=callbacks)
# model.load_weights("reflex.hdf5")
model.save_weights('membrane.all.hdf5')
testGene = testGenerator("membrane/test")
results = model.predict_generator(testGene,30,verbose=1)
saveResult("membrane/test/predict",results)