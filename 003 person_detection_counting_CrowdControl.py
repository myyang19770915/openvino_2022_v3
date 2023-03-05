from openvino.runtime import Core
import cv2,time
import numpy as np

# 人流計算初始值
ObjList=[]
objIndex=0
#物件追蹤程式
def TrackObj(newCenterList,ObjList1,objIndex):
    ObjListTemp=[]
    try:
        for i in range(len(newCenterList)):
            newP=(newCenterList[i][0],newCenterList[i][1])
            Obj,objIndex = MatchObj(newP,ObjList1,objIndex)
            ObjListTemp.append(Obj)
        return ObjListTemp,objIndex
    except Exception as err:
        print(err)
        return None,objIndex

#位置比對程式
def MatchObj(newP,ObjList1,ObjIndex):
    dtlist=[]
    dtIndex=[]
    #計算物件與所有前一畫面物件的距離
    for i in range(len(ObjList1)):
        distant = ((ObjList1[i][1]-newP[0])**2 + (ObjList1[i][2]-newP[1])**2)**0.5
        if distant<25:
            dtIndex.append(i)
            dtlist.append(distant)

    #設定回傳
    if dtlist==[]: #新物件
        ObjIndex+=1
        Obj=ObjIndex,newP[0],newP[1]
    else: #舊物件
        #找出最小距離
        Obj=list(ObjList1[dtIndex[np.argmin(dtlist)]])
        Obj[1],Obj[2]=newP[0],newP[1]
    return Obj,ObjIndex


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
out = cv2.VideoWriter('output.mp4', fourcc, 30, (768, 432))  # frame size要一致， 才可以存檔

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
    #計算人數, 起始值0
    PredDict={'person':0}
    
    #偵測區域
    cv2.rectangle(frame, (30, 250), (720, 300), (0,0,255), 3, cv2.LINE_AA)
    #新畫面中所有物件中心點        
    newCenterList=[]
    
    for detection in prediction.reshape(-1, 7):
        confidence = float(detection[2])
        
        
        
        if confidence > 0.9:
            print('detection=', detection)
            xmin2 = int(detection[3] * frame.shape[1])
            ymin2 = int(detection[4] * frame.shape[0])
            xmax2 = int(detection[5] * frame.shape[1])
            ymax2 = int(detection[6] * frame.shape[0])
            cv2.rectangle(frame, (xmin2, ymin2), (xmax2, ymax2), color=(0, 255, 255))
            cv2.putText(frame,'PERSON '+str(int(confidence*100))+ '%',(xmin2, (ymin2-10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4,(0, 255, 255),1,cv2.LINE_AA)
            # 計算當下圖片中人數####
            if detection[1]==1:
              PredDict['person'] += 1
            ################
            
            #中心點
            (x,y)=(int((xmin2+xmax2)/2),int((ymin2+ymax2)/2))
            if x>30 and x<720 and y>250 and y<300:
              newCenterList.append((x,y))
              cv2.circle(frame,(x,y), 3, (255,255,0), 3)
    
    #將新中心點與現有中心點比對
    ObjList,objIndex=TrackObj(newCenterList,ObjList,objIndex)
    for i in range(len(ObjList)):
      cv2.putText(frame, str(ObjList[i][0]), (ObjList[i][1], ObjList[i][2]), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 0, 0), 2)

    ##
    print('Person sum: {}'.format(objIndex))
    ##
            
    cv2.putText(frame, 'fps: ' + str(int(1 / (time.time() - stime))), (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
    # 標註人數
    cv2.putText(frame, 'person counting: ' + str(objIndex), (20, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
    
    cv2.imshow('OpenVINO person detection', frame)
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



