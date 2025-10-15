# 🦈 Shark classification

using streamlit and ngrok, build website for shark image classification. 
using pretrained model (ResNet 50) with ImageNet dataset. 

<p align="center">
<img src="/image/shark.gif" width="450">
</p>


-----------------------------------------------------------------------

### Table of contents
- [ ] prerequisite
- [ ] Dataset + model 
- [ ] Create website
- [ ] RWebsite and upload shark image 
-----------------------------------------------------------------------

### prerequisite

Install libraries 

```
!pip install streamlit -q
!pip install pyngrok
```

create token on [pyngrok website](https://ngrok.com/)


```
from pyngrok import ngrok
ngrok.set_auth_token('Your token')
```

### Dataset + model 

First, you read csv file and check all variables. 

```
%%writefile app.py

import streamlit as st
import tensorflow as tf
import PIL
from PIL import Image, ImageOps
import numpy as np
from tensorflow.keras.applications.imagenet_utils import decode_predictions

#..... Load pretrained model 
resnet50_pre = tf.keras.applications.resnet.ResNet50(weights='imagenet',input_shape=(224,224,3))

#..... title of the website 
st.title('Shark image classification')
#..... the section where new image is uploaded 
file = st.file_uploader('please upload your image (only jpg or png format).', type = ['jpg','png'])

#..... if no file has been uplodaed, show additional msg 
if file is None:
  st.text('please upload your image first.')
else:
  #..... open uploaded data 
  image = Image.open(file)
  st.image(image,use_column_width=True)
  #..... resize image 
  img_resized = ImageOps.fit(image,(224,224),PIL.Image.Resampling.LANCZOS)
  img_resized = img_resized.convert("RGB")
  img_resized = np.array(img_resized)
  #..... predict 
  pred = resnet50_pre.predict(img_resized.reshape([1,224,224,3]))
  decoded_pred = decode_predictions(pred)
  result = ''
  #..... sshows the results according to ranking
  for i, instance in enumerate(decoded_pred[0]):
    result += 'rank {}: {} ({:.2f}%)'.format(i+1, instance[1], instance[2]*100)
  st.success(result)

```

### Create website 

```
!nohup streamlit run app.py --server.port 80 &
nohup python3 -m streamlit run app.py --server.port 80 &
```

create URL

```
url = ngrok.connect(addr='80')
url
```

the output is 



### Website and upload shark image 



-----------------------------------------------------------------------

### Reference
- [Youtube]()


### image sources
- [Giphy](https://giphy.com/gifs/sharkweek-discovery-shark-week-tyson-vs-jaws-ggnetrSQzSUHBRLIK9)
