from openvino.runtime import Core
import cv2,time
import numpy as np

## 須注意使用的model所需的input data size
## Image, name: data, shape: 1, 3, 384, 672 in the format B, C, H, W, 
#                        B , C, H,  W
data = np.ndarray(shape=(1, 3 ,384, 672), dtype=np.float32)  # for face detection
data1 = np.ndarray(shape=(1, 3 ,62, 62), dtype=np.float32)   # for age-gender recognition
data2 = np.ndarray(shape=(1, 3 ,64, 64), dtype=np.float32)   # for emotion recognition

#載入不同的模型
ie = Core()
#compiled model , device_name : CPU,GPU, AUTO, 
## face detection model ###
compiled_model = ie.compile_model('intel/face-detection-adas-0001/FP32/face-detection-adas-0001.xml', device_name="AUTO")
output_face = compiled_model.output(0) #取得輸出層

### age/gender model ###
age_gender_model = ie.compile_model('intel/age-gender-recognition-retail-0013/FP32/age-gender-recognition-retail-0013.xml', device_name="AUTO")
output_prob = age_gender_model.output(0) #取得第0輸出層, gender probability  ,  classes [0 - female, 1 - male].
output_age = age_gender_model.output(1) #取得第1輸出層 , age 是除100的數值,所以實際年齡取出後需要x100

### emotion model ###
emotion_model = ie.compile_model('intel/emotions-recognition-retail-0003/FP32/emotions-recognition-retail-0003.xml', device_name="AUTO")
output_prob_em = emotion_model.output(0) #取得輸出層  ,  Softmax output across five emotions (0 - ‘neutral’, 1 - ‘happy’, 2 - ‘sad’, 3 - ‘surprise’, 4 - ‘anger’)
### class or labels ###
GENDERS = ['Female', 'Male']
EMOTION = ['neutral', 'happy', 'sad', 'surprise', 'anger']

## input data ##
### 使用webcam
cap = cv2.VideoCapture(0)#自行選擇攝影機編號
### 使用動態影像, intel 官網下載的data
# target ='open_model_zoo/data/video/people-detection.mp4'
# cap = cv2.VideoCapture(target)

#儲存 mp4 格式 , 辨識後的結果儲存下來
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # (*'mpv4') 用這種表示會error,無法存檔
out = cv2.VideoWriter('output.mp4', fourcc, 20, (640, 480))

while cap.isOpened():
    stime = time.time()
    ret, frame = cap.read() #ret=retval,frame=image 
    ## 判斷是否有讀到東西, 要加入，避免讀完檔案後報錯
    if not ret:
      print("Ignoring empty camera frame.")
      break
    #### 先使用face detection 辨識出人臉 ###
    resizeframe = cv2.resize(frame,(672,384))  # W x H
    # resizeframe = cv2.cvtColor(resizeframe, cv2.COLOR_BGR2RGB)  #CV2顏色BGR，其他都是RGB   
    # cv2使用方法imread读出来图片的数据是shape [h w c] , 轉成 C, H, W 
    image_array = resizeframe.transpose(2,0,1)
    data[0] = image_array
    #預測結果
    prediction = compiled_model(data)[output_face]
		##開始將inference的結果自行進行處理，以下為劃出bounding box, 並標出類別與信心水準
    for detection in prediction.reshape(-1, 7):   #開始對每一個辨識出來的人臉進行處理
        confidence = float(detection[2])
        if confidence > 0.6:
            print('detection=', detection)
            xmin = int(detection[3] * frame.shape[1])
            ymin = int(detection[4] * frame.shape[0])
            xmax = int(detection[5] * frame.shape[1])
            ymax = int(detection[6] * frame.shape[0])
            
            ## 取出人臉圖像
            if ymin > 0 and xmin > 0:
                facepic = frame[ymin:ymax, xmin:xmax]
                facepic_ag = cv2.resize(facepic,(62,62))
                image_array1 = facepic_ag.transpose(2,0,1)
                data1[0] = image_array1
                gender = age_gender_model(data1)[output_prob]
                age = age_gender_model(data1)[output_age]
                
                GE = GENDERS[gender[0].argmax()]
                AG = age[0][0][0][0] * 100
                       
                text = "{}, age: {:.0f}".format(GE, AG)
                if GE == 'Male':
                    color = (255, 0, 0)
                else:
                    color = (255, 0, 255)
                
                ## 辨識 emotion ##
                facepic_em = cv2.resize(facepic,(64,64))
                image_array2 = facepic_em.transpose(2,0,1)
                data2[0] = image_array2
                emotion = emotion_model(data2)[output_prob_em]
                
                # # print('emotion = ', emotion)
                emotext = EMOTION[emotion[0].argmax()]
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color=color)
                cv2.putText(frame, emotext + ' FACE', (xmin + 3, (ymin + 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
                cv2.putText(frame, text, (xmin + 3, (ymin - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
            ##############
            

    cv2.putText(frame, 'fps: ' + str(int(1 / (time.time() - stime))), (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.imshow('OpenVINO face/age/gender/emotion detection', frame)
    # save
    out.write(frame)
    
    key=cv2.waitKey(1)
    # 按q離開
    if key == ord('q'):
        break
# 釋放攝影機
cap.release()
out.release()
# 關閉所有 OpenCV 視窗
cv2.destroyAllWindows()