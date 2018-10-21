# Sign2Text

## Sign Language Reader

Deep Learning and Computer Vision Sign Language real-time Translator to English using Drone technologies.

---

Recognition model that streams video from a drone and detects and recognises sign language:
```
(env) $ python recognize.py
```

Sign Language single symbol classification: multi-label classification ConvNet that distinguishes between the letters in the Sign Language Alphabet:
```
(env) $ python classify.py
```

Hand Detection: binary classification ConvNet that distinguishes between a hand or not:
```
(env) $ python detect.py
```

---

### Technologies:
* Deep Learning & Computer Vision: Keras, TensorFlow, Sklearn, Skimage, OpenCV, Numpy, Pandas
* DJI Tello Drone 

---

### Instructions:

```
(env) $ pip install -r requirements.txt
```

```
(env) $ cat requirements.txt
keras
tensorflow
opencv-python
tellopy
image
imutils
pygame
av
```

---

### Run it!:
* DJI Tello Drone:
```
(env) $ python flight.py
```
* Your webcam:
```
(env) $ python webcam.py
```

---

<table>
    <tr>
        <td align="center" width="200">
            <img src='img/dcu.png' width='120'/>
            <div style='word-wrap: break-word;width:120px;vertical-align:text-bottom'>Dublin City University</div></td>
        <td align="center" width="200">
            <img src='img/insight-centre.png' width='120'/>
            <div style='word-wrap:break-word;width:140px;vertical-align:text-bottom'>Insight Centre for Data Analytics</div>
        </td>
        <td align="center" width="200">
            <img src='img/talent-garden.png' width='120'/>
            <div style='word-wrap: break-word;width:120px;vertical-align:text-bottom'>Talent Garden Dublin</div>
        </td>
    </tr>
</table>

---

### Sign Language Alphabet:
<img src='img/letters.png' width='700'/>