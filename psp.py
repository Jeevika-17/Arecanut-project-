{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "2860cb32",
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import json\n",
    "import numpy as np\n",
    "from tensorflow.keras.preprocessing.image import load_img, img_to_array\n",
    "from tensorflow.keras.utils import Sequence, to_categorical\n",
    "from tensorflow.keras.applications import InceptionV3\n",
    "from tensorflow.keras.models import Model\n",
    "from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, UpSampling2D, Concatenate, Conv2D\n",
    "import tensorflow as tf"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "0b05c353",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "{0: 'Arecanut', 1: 'good', 2: 'karigot', 3: 'phatora', 4: 'phattora'}\n"
     ]
    }
   ],
   "source": [
    "data_dir = r'train'\n",
    "annotation_path = r'_annotations.coco.json'\n",
    "\n",
    "# Read annotation file\n",
    "with open(annotation_path, 'r') as f:\n",
    "    annotations = json.load(f)\n",
    "\n",
    "# Extract categories and create a mapping\n",
    "categories = {cat['id']: cat['name'] for cat in annotations['categories']}\n",
    "category_to_index = {name: idx for idx, name in enumerate(categories.values())}\n",
    "# Flatten images list with category IDs\n",
    "print(categories)\n",
    "images = []\n",
    "for image_info in annotations['images']:\n",
    "    image_annotations = [ann for ann in annotations['annotations'] if ann['image_id'] == image_info['id']]\n",
    "    for annotation in image_annotations:\n",
    "        images.append({\n",
    "            \"file_name\": image_info['file_name'],\n",
    "            \"category\": categories[annotation['category_id']]\n",
    "        })     \n",
    "\n",
    "# Custom Data Generator\n",
    "class CustomDataGenerator(Sequence):\n",
    "    def __init__(self, data_dir, images, category_to_index, batch_size, input_size, shuffle=True):\n",
    "        self.data_dir = data_dir\n",
    "        self.images = images\n",
    "        self.category_to_index = category_to_index\n",
    "        self.batch_size = batch_size\n",
    "        self.input_size = input_size\n",
    "        self.shuffle = shuffle\n",
    "        self.on_epoch_end()\n",
    "        \n",
    "    def __len__(self):\n",
    "        return int(np.floor(len(self.images) / self.batch_size))\n",
    "    \n",
    "    def on_epoch_end(self):\n",
    "        self.indexes = np.arange(len(self.images))\n",
    "        if self.shuffle:\n",
    "            np.random.shuffle(self.indexes)\n",
    "    \n",
    "    def __getitem__(self, index):\n",
    "        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]\n",
    "        batch_images = [self.images[k] for k in indexes]\n",
    "        X, y = self.__data_generation(batch_images)\n",
    "        return X, y\n",
    "    \n",
    "    def __data_generation(self, batch_images):\n",
    "        X = np.empty((self.batch_size, *self.input_size, 3))\n",
    "        y = np.empty((self.batch_size), dtype=int)\n",
    "        \n",
    "        for i, image_info in enumerate(batch_images):\n",
    "            image_path = os.path.join(self.data_dir, image_info['file_name'])\n",
    "            
    "            image = load_img(image_path, target_size=self.input_size)\n",
    "            image = img_to_array(image) / 255.0  # Normalize image\n",
    "            label = self.category_to_index[image_info['category']]\n",
    "            \n",
    "            X[i,] = image\n",
    "            y[i] = label\n",
    "        \n",
    "        return X, to_categorical(y, num_classes=len(self.category_to_index))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "9c987343",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Parameters\n",
    "input_height, input_width = 640,640\n",
    "batch_size = 20\n",
    "num_classes = len(categories)\n",
    "input_size = (input_height, input_width)\n",
    "\n",
    "# Create Generators\n",
    "train_generator = CustomDataGenerator(\n",
    "    data_dir=data_dir,\n",
    "    images=images,\n",
    "    category_to_index=category_to_index,\n",
    "    batch_size=batch_size,\n",
    "    input_size=input_size,\n",
    ")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "4a737072",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Define Model (PSP-Net + InceptionV3)\n",
    "input_tensor = Input(shape=(input_height, input_width, 3))\n",
    "inception_base = InceptionV3(weights='imagenet', include_top=False, input_tensor=input_tensor)\n",
    "\n",
    "def pyramid_pooling_module(x, bin_sizes):\n",
    "    concat_list = [x]\n",
    "    h, w = x.shape[1], x.shape[2]\n",
    "    for bin_size in bin_sizes:\n",
    "        pool_size = (h // bin_size, w // bin_size)\n",
    "        pool = tf.keras.layers.AveragePooling2D(pool_size, strides=pool_size, padding='same')(x)\n",
    "        pool = Conv2D(512, (1, 1), padding='same', activation='relu')(pool)\n",
    "        pool = UpSampling2D(size=pool_size, interpolation='bilinear')(pool)\n",
    "        concat_list.append(pool)\n",
    "    \n",
    "    return Concatenate()(concat_list)\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "275e6660",
   "metadata": {},
   "outputs": [],
   "source": [
    "psp = pyramid_pooling_module(inception_base.output, bin_sizes=[1, 2, 3, 6])\n",
    "\n",
    "x = GlobalAveragePooling2D()(psp)\n",
    "x = Dense(512, activation='relu')(x)\n",
    "output_tensor = Dense(num_classes, activation='softmax')(x)\n",
    "\n",
    "model = Model(inputs=input_tensor, outputs=output_tensor)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "aa18c729",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text":[
      "Epoch 1/15\n"
     ]
    },
  
    
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m205s\u001b[0m 205s/step - accuracy: 0.3000 - loss: 1.6849\n",
      "Epoch 2/15\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m168s\u001b[0m 168s/step - accuracy: 0.3000 - loss: 2.8821\n",
      "Epoch 3/15\n",
      
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m164s\u001b[0m 164s/step - accuracy: 0.4000 - loss: 3.9712\n",
      "Epoch 4/15\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m159s\u001b[0m 159s/step - accuracy: 0.1500 - loss: 1.4835\n",
      "Epoch 5/15\n"
      
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m164s\u001b[0m 164s/step - accuracy: 0.3000 - loss: 1.3515\n",
      "Epoch 6/15\n",
      
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m167s\u001b[0m 167s/step - accuracy: 0.1000 - loss: 1.3037\n",
      "Epoch 7/15\n",
      
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m157s\u001b[0m 157s/step - accuracy: 0.1500 - loss: 1.2292\n",
      "Epoch 8/15\n"
           ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m169s\u001b[0m 169s/step - accuracy: 0.3500 - loss: 1.2125\n",
      "Epoch 9/15\n",
      
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m158s\u001b[0m 158s/step - accuracy: 0.3500 - loss: 1.2359\n",
      "Epoch 10/15\n",
      
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m161s\u001b[0m 161s/step - accuracy: 0.6000 - loss: 1.0026\n",
      "Epoch 11/15\n"
     
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m169s\u001b[0m 169s/step - accuracy: 0.5000 - loss: 1.0240\n",
      "Epoch 12/15\n",
      
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m171s\u001b[0m 171s/step - accuracy: 0.5500 - loss: 1.0107\n",
      "Epoch 13/15\n",
      
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m167s\u001b[0m 167s/step - accuracy: 0.6000 - loss: 0.8915\n",
      "Epoch 14/15\n"
       
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m164s\u001b[0m 164s/step - accuracy: 0.3000 - loss: 1.6826\n",
      "Epoch 15/15\n",
      
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m153s\u001b[0m 153s/step - accuracy: 0.4500 - loss: 1.0400\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<keras.src.callbacks.history.History at 0x1fb61d29180>"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])\n",
    "\n",
    "# Training the model\n",
    "model.fit(\n",
    "    train_generator,\n",
    "    epochs=15,\n",
    "    steps_per_epoch=1\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e3bfcd5a",
   "metadata": {},
   "outputs": [],
   "source": [
    "model.save('arecanut.h5')\n",
    "class TestDataGenerator(Sequence):\n",
    "    def __init__(self, data_dir, file_names, batch_size, input_size, shuffle=False):\n",
    "        self.data_dir = data_dir\n",
    "        self.file_names = file_names\n",
    "        self.batch_size = batch_size\n",
    "        self.input_size = input_size\n",
    "        self.shuffle = shuffle\n",
    "        self.on_epoch_end()\n",
    "        \n",
    "    def __len__(self):\n",
    "        return int(np.floor(len(self.file_names) / self.batch_size))\n",
    "    \n",
    "    def on_epoch_end(self):\n",
    "        self.indexes = np.arange(len(self.file_names))\n",
    "        if self.shuffle:\n",
    "            np.random.shuffle(self.indexes)\n",
    "    \n",
    "    def __getitem__(self, index):\n",
    "        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]\n",
    "        batch_file_names = [self.file_names[k] for k in indexes]\n",
    "        X = self.__data_generation(batch_file_names)\n",
    "        return X\n",
    "    \n",
    "    def __data_generation(self, batch_file_names):\n",
    "        X = np.empty((self.batch_size, *self.input_size, 3))\n",
    "        \n",
    "        for i, file_name in enumerate(batch_file_names):\n",
    "            image_path = os.path.join(self.data_dir, file_name)\n",
    "            image = load_img(image_path, target_size=self.input_size)\n",
    "            image = img_to_array(image) / 255.0  # Normalize image\n",
    "            X[i,] = image\n",
    "        \n",
    "        return X\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9a5bf8b4",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Path to test data\n",
    "test_data_dir = r'test'\n",
    "\n",
    "# List of test image file names\n",
    "test_file_names = [f for f in os.listdir(test_data_dir)]\n",
    "\n",
    "# Parameters\n",
    "batch_size = 20\n",
    "input_size = (640,640)\n",
    "\n",
    "# Create Test Data Generator\n",
    "test_generator = TestDataGenerator(\n",
    "    data_dir=test_data_dir,\n",
    "    file_names=test_file_names,\n",
    "    batch_size=batch_size,\n",
    "    input_size=input_size\n",
    ")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "93c2faee",
   "metadata": {
    "collapsed": true
   },
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'model' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "Cell \u001b[1;32mIn[2], line 2\u001b[0m\n\u001b[0;32m      1\u001b[0m \u001b[38;5;66;03m# Make predictions\u001b[39;00m\n\u001b[1;32m----> 2\u001b[0m predictions \u001b[38;5;241m=\u001b[39m \u001b[43mmodel\u001b[49m\u001b[38;5;241m.\u001b[39mpredict(test_generator, steps\u001b[38;5;241m=\u001b[39m\u001b[38;5;28mlen\u001b[39m(test_generator))\n\u001b[0;32m      4\u001b[0m \u001b[38;5;66;03m# Get the predicted class indices\u001b[39;00m\n\u001b[0;32m      5\u001b[0m predicted_class_indices \u001b[38;5;241m=\u001b[39m np\u001b[38;5;241m.\u001b[39margmax(predictions, axis\u001b[38;5;241m=\u001b[39m\u001b[38;5;241m1\u001b[39m)\n",
      "\u001b[1;31mNameError\u001b[0m: name 'model' is not defined"
     ]
    }
   ],
   "source": [
    "# Make predictions\n",
    "predictions = model.predict(test_generator, steps=len(test_generator))\n",
    "\n",
    "# Get the predicted class indices\n",
    "predicted_class_indices = np.argmax(predictions, axis=1)\n",
    "print(predicted_class_indices)\n",
    "# Map predicted indices to category names\n",
    "index_to_category = {v: k for k, v in category_to_index.items()}\n",
    "predicted_categories = [index_to_category[idx] for idx in predicted_class_indices]\n",
    "\n",
    "# Print predictions\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "87220a97",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Image: IMG_20230308_120224_jpg.rf.46bebe80955cc7106f89e0f129e12a62.jpg - Predicted Category: karigot\n",
      "Image: IMG_20230308_120329_jpg.rf.7597427e24bb1e683b64c00de41349d5.jpg - Predicted Category: karigot\n",
      "Image: IMG_20230308_120408_jpg.rf.62fe563984f40d675292a7a531354e97.jpg - Predicted Category: karigot\n",
      "Image: IMG_20230308_120416_jpg.rf.2ef95a8d529da58d58cbb53d0769948b.jpg - Predicted Category: karigot\n",
      "Image: IMG_20230308_120426_jpg.rf.753c73666a4138352ef8372e501c6494.jpg - Predicted Category: karigot\n",
      "Image: IMG_20230308_120433_jpg.rf.52b25e04e59d06b95c6e56441207f2cb.jpg - Predicted Category: karigot\n",
      "Image: IMG_20230308_120533_jpg.rf.034d6247d242ac041d15bfbcab0304eb.jpg - Predicted Category: karigot\n",
      "Image: IMG_20230308_120604_jpg.rf.bc305e1c60ffeb1f253f11c29ed756e2.jpg - Predicted Category: karigot\n",
      "Image: IMG_20230308_120952_jpg.rf.81513754ec337323f598ff76925dfcfa.jpg - Predicted Category: karigot\n",
      "Image: IMG_20230308_120957_jpg.rf.3d1f781ea852e78b2852a343665729cd.jpg - Predicted Category: karigot\n",
      "Image: IMG_20230308_121219_jpg.rf.a52c431b6e3e54c7509eb98898d7db4e.jpg - Predicted Category: karigot\n",
      "Image: IMG_20230308_121234_jpg.rf.851cfdb4adab5ea51ea3ffa2252f1386.jpg - Predicted Category: karigot\n",
      "Image: IMG_20230308_121310_jpg.rf.3a0dc6a46bf9135a8b37bf55389e3f52.jpg - Predicted Category: karigot\n",
      "Image: IMG_20230308_121908_jpg.rf.f88a5f0788b548d1c59b5fb15fa0b4f1.jpg - Predicted Category: karigot\n",
      "Image: IMG_20230308_121933_jpg.rf.59507ee33e91194d49fecae603d288e5.jpg - Predicted Category: karigot\n",
      "Image: IMG_20230308_121951_jpg.rf.9da8ba18286b2bbe489bf90b773e10ec.jpg - Predicted Category: karigot\n",
      "Image: IMG_20230308_122109_jpg.rf.72786631868a399283e64f29824aeec2.jpg - Predicted Category: karigot\n",
      "Image: IMG_20230308_122739_jpg.rf.8f4b1019522ad2ba385ab9e4dafd300f.jpg - Predicted Category: Arecanut\n",
      "Image: IMG_20230308_123012_jpg.rf.52e1b36b0836be805bb9bcc3525311ad.jpg - Predicted Category: karigot\n",
      "Image: IMG_20230308_123232_jpg.rf.0dad9c10b958566b548934c206bd519c.jpg - Predicted Category: karigot\n"
     ]
    }
   ],
   "source": [
    "for file_name, category in zip(test_file_names, predicted_categories):\n",
    "    print(f'Image: {file_name} - Predicted Category: {category}')\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a80a5930",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "16d76ae3",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "03215222",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b8a17f51",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6ce64b75",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7904aa86",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4880963c",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "baae61bb",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ddb76466",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "13ac6926",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
