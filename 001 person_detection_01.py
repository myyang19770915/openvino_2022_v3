from openvino.runtime import Core
import cv2,time
import numpy as np

## 須注意使用的model所需的input data size
#                        B , C, H,  W
data = np.ndarray(shape=(1, 3 ,320, 544), dtype=np.float32)

#載入模型
ie = Core()
#compiled model , device_name : CPU,GPU, AUTO, 
compiled_model = ie.compile_model('intel/person-detection-retail-0013/FP32/person-detection-retail-0013.xml', device_name="AUTO")
output_key = compiled_model.output(0) #取得輸出層
### 使用webcam
# cap = cv2.VideoCapture(0)#自行選擇攝影機編號
### 使用動態影像, intel 官網下載的data
target ='open_model_zoo/data/video/people-detection.mp4'
cap = cv2.VideoCapture(target)

#儲存 mp4 格式 , 辨識後的結果儲存下來
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # (*'mpv4') 用這種表示會error,無法存檔
out = cv2.VideoWriter('output.mp4', fourcc, 30, (544, 320))

while cap.isOpened():
    stime = time.time()
    ret, frame = cap.read() #ret=retval,frame=image 
    ## 判斷是否有讀到東西, 要加入，避免讀完檔案後報錯
    if not ret:
      print("Ignoring empty camera frame.")
      break
    ####################
    resizeframe = cv2.resize(frame,(544,320))  # W x H
    # resizeframe = cv2.cvtColor(resizeframe, cv2.COLOR_BGR2RGB)  #CV2顏色BGR，其他都是RGB   
    # cv2使用方法imread读出来图片的数据是shape [h w c] , 轉成 C, H, W 
    normalized_image_array = resizeframe.transpose(2,0,1)
    data[0] = normalized_image_array
    #預測結果
    prediction = compiled_model(data)[output_key]
		##開始將inference的結果自行進行處理，以下為劃出bounding box, 並標出類別與信心水準
    for detection in prediction.reshape(-1, 7):
        confidence = float(detection[2])
        if confidence > 0.25:
            print('detection=', detection)
            xmin2 = int(detection[3] * resizeframe.shape[1])
            ymin2 = int(detection[4] * resizeframe.shape[0])
            xmax2 = int(detection[5] * resizeframe.shape[1])
            ymax2 = int(detection[6] * resizeframe.shape[0])
            cv2.rectangle(resizeframe, (xmin2, ymin2), (xmax2, ymax2), color=(0, 255, 255))
            cv2.putText(resizeframe,'PERSON '+str(int(confidence*100))+ '%',(xmin2, (ymin2-10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4,(0, 255, 255),1,cv2.LINE_AA)
    cv2.putText(resizeframe, 'fps: ' + str(int(1 / (time.time() - stime))), (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.imshow('OpenVINO person detection', resizeframe)
    # save
    out.write(resizeframe)
    
    key=cv2.waitKey(1)
    # 按q離開
    if key == ord('q'):
        break
# 釋放攝影機
cap.release()
out.release()
# 關閉所有 OpenCV 視窗
cv2.destroyAllWindows()